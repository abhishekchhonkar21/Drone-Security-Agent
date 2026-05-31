"""
app/agents/summary_agent.py
Fetches all frame documents for a session, compresses the timeline
into major events, and generates a natural language session summary.
"""
from langchain_openai import AzureChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.db.mongo import frames_col, alerts_col, sessions_col
from app.utils.logger import get_logger


log = get_logger(__name__)


_client = AzureChatOpenAI(
    azure_deployment=settings.azure_openai_deployment,
    api_version=settings.azure_openai_api_version,
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_key,
    temperature=0.3,
)


_SYSTEM_PROMPT = """\
You are a security operations centre analyst writing a shift handover summary for a drone patrol session.
You will receive a chronological sequence of frame-level security analyses.
Write a clear, concise summary covering:
- Key objects and entities observed
- Notable movements or entries/exits
- Any suspicious activity or risk signals
- Alert events that were triggered
- Overall threat assessment

Format: 3–5 paragraphs of plain prose. No bullet points. Be factual and specific.
End with a one-sentence "Session conclusion" on its own line prefixed with "CONCLUSION: ".
"""


def _compress_frame_timeline(frames: list[dict]) -> str:
    """
    Compress potentially hundreds of frame summaries into a token-efficient
    timeline string for the LLM. Groups frames by zone and summarises.
    """
    if not frames:
        return "No frames recorded."

    lines = []
    for f in frames:
        ctx = f.get("llm_context", {})
        ts = str(f.get("video_timestamp", "?"))[:19]
        zone = f.get("telemetry", {}).get("zone_name", "?")
        summary = ctx.get("scene_summary", "")
        risks = ", ".join(ctx.get("risk_signals", []))
        line = f"[{ts}] zone={zone} | {summary}"
        if risks:
            line += f" | ⚠ {risks}"
        lines.append(line)

    if len(lines) > 200:
        step = max(1, len(lines) // 200)
        lines = lines[::step]
        lines.append(f"... (sub-sampled from {len(frames)} total frames)")

    return "\n".join(lines)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def _call_summary_llm(user_prompt: str) -> str:
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


async def generate_session_summary(video_id: str) -> dict:
    """
    Generate (or refresh) a summary for a video session.
    Stores the summary text back into the video_sessions document.

    Returns:
        dict with keys: summary_text, frame_count, alert_count, zones_visited
    """
    log.info(f"[SummaryAgent] Generating summary for video_id={video_id}")

    cursor = frames_col().find(
        {"video_id": video_id},
        sort=[("video_timestamp", 1)],
    )
    frames = await cursor.to_list(length=5000)
    log.info(f"[SummaryAgent] Fetched {len(frames)} frames for summarisation")

    alert_cursor = alerts_col().find({"video_id": video_id})
    alerts = await alert_cursor.to_list(length=200)

    timeline_str = _compress_frame_timeline(frames)
    zones = list({f.get("telemetry", {}).get("zone_name", "unknown") for f in frames})
    alert_summaries = "\n".join(
        f"- [{a.get('severity', '?').upper()}] {a.get('title', '')}: {a.get('description', '')}"
        for a in alerts
    )

    user_prompt = (
        f"Video session ID: {video_id}\n"
        f"Total frames: {len(frames)}\n"
        f"Zones monitored: {', '.join(zones)}\n\n"
        f"Frame timeline:\n{timeline_str}\n\n"
        f"Alerts raised ({len(alerts)}):\n{alert_summaries or 'None'}\n\n"
        "Write the session summary now."
    )

    try:
        summary_text = _call_summary_llm(user_prompt)
        log.info(f"[SummaryAgent] Summary generated ({len(summary_text)} chars)")
    except Exception as exc:
        log.error(f"[SummaryAgent] LLM call failed: {exc}", exc_info=True)
        summary_text = f"[Summary generation failed: {exc}]"

    await sessions_col().update_one(
        {"_id": video_id},
        {"$set": {"summary_text": summary_text}},
    )

    return {
        "summary_text": summary_text,
        "frame_count": len(frames),
        "alert_count": len(alerts),
        "zones_visited": zones,
    }