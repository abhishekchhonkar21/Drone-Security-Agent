"""
app/agents/alert_agent.py
Sliding-window alert agent. Reads recent frame documents from MongoDB,
builds a rich prompt, sends to Azure OpenAI for structured classification,
deduplicates, and persists alerts.
"""
import json
import re
import uuid
from datetime import datetime, timezone

from langchain_openai import AzureChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.db.mongo import frames_col, alerts_col
from app.models.schemas import AlertDocument, AlertReasoningTrace
from app.utils.logger import get_logger


log = get_logger(__name__)


_client = AzureChatOpenAI(
    azure_deployment=settings.azure_openai_deployment,
    api_version=settings.azure_openai_api_version,
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_key,
)


_SYSTEM_PROMPT = """\
You are a security alert classifier for a drone surveillance system.
You receive a sequence of recent frame analyses from a fixed monitoring zone.
Your task is to decide whether the sequence represents a security alert.

Return ONLY valid JSON with this exact schema — no markdown, no extra text:
{
  "is_alert": true | false,
  "alert_type": "<loitering | repeated_vehicle | after_hours_activity | restricted_zone_intrusion | crowd_formation | suspicious_package | tailgating | unknown>",
  "severity": "<low | medium | high | critical>",
  "title": "<concise alert title>",
  "description": "<1–2 sentence description of the threat>",
  "evidence_frame_ids": ["<frame_id_1>", "<frame_id_2>"],
  "reasoning": "<brief reasoning explaining why this is or is not an alert>"
}

Alert classification rules:
- loitering: same person/vehicle visible across ≥3 consecutive frames without clear purpose
- repeated_vehicle: same vehicle description (color + type) seen multiple times
- after_hours_activity: any activity in sensitive zone between 22:00 and 06:00 UTC
- restricted_zone_intrusion: human presence in restricted / sensitive zones (gate, server room)
- crowd_formation: sudden increase in people_count beyond 3 in a zone
- suspicious_package: unattended bags, boxes, or containers detected
- tailgating: two vehicles / persons entering a controlled zone in rapid succession

If is_alert is false, still provide a reasoning explanation.
Severity guide: low=informational, medium=monitor, high=respond soon, critical=immediate response.
"""


def _build_frame_context_block(frame: dict) -> str:
    ctx = frame.get("llm_context", {})
    tel = frame.get("telemetry", {})
    return (
        f"Frame ID: {frame['_id']}\n"
        f"Time: {frame.get('video_timestamp', 'N/A')}\n"
        f"Zone: {tel.get('zone_name', 'unknown')}\n"
        f"GPS: lat={tel.get('latitude')}, lon={tel.get('longitude')}, alt={tel.get('altitude')}m\n"
        f"Scene: {ctx.get('scene_summary', '')}\n"
        f"Objects: {', '.join(ctx.get('objects_detected', []))}\n"
        f"People: {ctx.get('people_count', 0)} | Vehicles: {ctx.get('vehicles_count', 0)}\n"
        f"Activities: {', '.join(ctx.get('activity_detected', []))}\n"
        f"Risk signals: {', '.join(ctx.get('risk_signals', []))}\n"
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def _call_alert_llm(user_prompt: str) -> str:
    response = _client.invoke([
        ("system", _SYSTEM_PROMPT),
        ("human", user_prompt),
    ])

    content = response.content
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts).strip()

    return str(content).strip()


def _parse_alert_json(raw: str) -> dict | None:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except Exception as exc:
        log.warning(f"[AlertAgent] Failed to parse alert JSON: {exc} | raw={raw[:200]}")
        return None


async def _is_duplicate(video_id: str, alert_type: str, zone_name: str) -> bool:
    """
    Simple deduplication: skip if an alert of the same type + zone
    already exists for this session with status 'new'.
    """
    existing = await alerts_col().find_one({
        "video_id": video_id,
        "alert_type": alert_type,
        "zone_name": zone_name,
        "status": "new",
    })
    return existing is not None


async def run_alert_check(video_id: str, recent_frame_ids: list[str]) -> None:
    """
    Fetch the given frame documents from MongoDB, build a prompt,
    call the alert LLM, and persist any new alerts.

    Args:
        video_id:          Session identifier.
        recent_frame_ids:  List of frame _ids to include in the window.
    """
    log.info(
        f"[AlertAgent] Checking {len(recent_frame_ids)} frames "
        f"for video_id={video_id}"
    )

    cursor = frames_col().find({"_id": {"$in": recent_frame_ids}})
    frames = await cursor.to_list(length=len(recent_frame_ids))

    if not frames:
        log.warning("[AlertAgent] No frames found for alert check — skipping")
        return

    context_blocks = "\n---\n".join(_build_frame_context_block(f) for f in frames)
    zone_names = list({f.get("telemetry", {}).get("zone_name", "unknown") for f in frames})
    now_utc = datetime.now(tz=timezone.utc).isoformat()

    user_prompt = (
        f"Current UTC time: {now_utc}\n"
        f"Zone(s) under review: {', '.join(zone_names)}\n"
        f"Sensitive zones (always alert for intrusions): main_gate, server_room, restricted_area\n\n"
        f"Recent frame sequence ({len(frames)} frames):\n\n"
        f"{context_blocks}\n\n"
        "Based on the above sequence, determine whether a security alert should be raised."
    )

    try:
        raw = _call_alert_llm(user_prompt)
        log.debug(f"[AlertAgent] Raw LLM response: {raw[:200]}")
        result = _parse_alert_json(raw)
    except Exception as exc:
        log.error(f"[AlertAgent] LLM call failed: {exc}", exc_info=True)
        return

    if not result:
        log.warning("[AlertAgent] Could not parse alert response — skipping")
        return

    log.info(
        f"[AlertAgent] is_alert={result.get('is_alert')} | "
        f"type={result.get('alert_type')} | severity={result.get('severity')} | "
        f"reasoning={result.get('reasoning', '')[:80]}"
    )

    if not result.get("is_alert"):
        return

    zone_name = zone_names[0] if zone_names else "unknown"
    alert_type = result.get("alert_type", "unknown")

    if await _is_duplicate(video_id, alert_type, zone_name):
        log.info(
            f"[AlertAgent] Duplicate alert suppressed: "
            f"type={alert_type} zone={zone_name}"
        )
        return

    frames_sorted = sorted(
        frames,
        key=lambda f: f.get("video_timestamp", datetime.min),
        reverse=True,
    )
    trigger_ts = frames_sorted[0].get("video_timestamp") or datetime.now(tz=timezone.utc)

    if isinstance(trigger_ts, str):
        trigger_ts = datetime.fromisoformat(trigger_ts)

    alert = AlertDocument(
        _id=str(uuid.uuid4()),
        video_id=video_id,
        frame_ids=result.get("evidence_frame_ids", recent_frame_ids),
        alert_type=alert_type,
        severity=result.get("severity", "medium"),
        title=result.get("title", "Security Alert"),
        description=result.get("description", ""),
        trigger_timestamp=trigger_ts,
        zone_name=zone_name,
        status="new",
        generated_by="llm_alert_agent",
        reasoning_trace=AlertReasoningTrace(
            prompt_version="alert_v1",
            supporting_signals=result.get("reasoning", "").split(". "),
        ),
    )

    await alerts_col().insert_one(alert.model_dump(by_alias=True))
    log.info(
        f"[AlertAgent] 🚨 Alert persisted: {alert.title} | "
        f"severity={alert.severity} | zone={alert.zone_name}"
    )