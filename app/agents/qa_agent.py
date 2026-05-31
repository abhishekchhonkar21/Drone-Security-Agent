"""
app/agents/qa_agent.py
Follow-up QA agent. Translates natural language operator questions into
MongoDB queries, fetches matching frames, and synthesises a grounded answer.
The agent uses a two-step approach:
  Step 1 — Query planner: LLM extracts structured filter params from the question.
  Step 2 — Answer synthesiser: LLM produces a natural language answer over results.
"""
import json
import re

from langchain_openai import AzureChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.db.mongo import frames_col
from app.utils.logger import get_logger


log = get_logger(__name__)


_client = AzureChatOpenAI(
    azure_deployment=settings.azure_openai_deployment,
    api_version=settings.azure_openai_api_version,
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_key,
)


# ---------------------------------------------------------------------------
# Step 1: Query Planner
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are a database query planner for a drone security monitoring system.
The MongoDB 'frames' collection has these queryable fields:
  - video_id: string
  - telemetry.zone_name: string (e.g. "main_gate", "garage", "north_fence")
  - video_timestamp: ISO datetime
  - llm_context.objects_detected: array of strings
  - llm_context.activity_detected: array of strings
  - llm_context.risk_signals: array of strings
  - llm_context.people_count: integer
  - llm_context.vehicles_count: integer
  - normalized_entities.label: string (e.g. "blue_pickup_truck", "person_1")
  - normalized_entities.entity_type: string ("person" | "vehicle" | "object")

Given a user question and a video_id, return a JSON object with MongoDB-style filter conditions:
{
  "zone_name": "<string or null>",
  "object_keyword": "<keyword to search in objects_detected array or null>",
  "activity_keyword": "<keyword to search in activity_detected array or null>",
  "entity_label_keyword": "<keyword to search in normalized_entities.label or null>",
  "entity_type": "<person | vehicle | object | null>",
  "after_hour": <0-23 or null>,
  "before_hour": <0-23 or null>,
  "min_people": <integer or null>,
  "min_vehicles": <integer or null>,
  "limit": <integer, default 10>
}
Return ONLY valid JSON — no markdown, no explanation.
"""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
def _plan_query(question: str, video_id: str) -> dict:
    response = _client.invoke([
        ("system", _PLANNER_SYSTEM),
        ("human", f'video_id: "{video_id}"\nQuestion: "{question}"'),
    ])

    content = response.content
    if isinstance(content, str):
        raw = content.strip()
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        raw = "\n".join(text_parts).strip()
    else:
        raw = str(content).strip()

    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Step 2: MongoDB query builder
# ---------------------------------------------------------------------------

def _build_mongo_filter(video_id: str, plan: dict) -> dict:
    """Translate the planner output into a MongoDB filter dict."""
    filt: dict = {"video_id": video_id}

    if plan.get("zone_name"):
        filt["telemetry.zone_name"] = {
            "$regex": plan["zone_name"], "$options": "i"
        }

    if plan.get("object_keyword"):
        filt["llm_context.objects_detected"] = {
            "$elemMatch": {"$regex": plan["object_keyword"], "$options": "i"}
        }

    if plan.get("activity_keyword"):
        filt["llm_context.activity_detected"] = {
            "$elemMatch": {"$regex": plan["activity_keyword"], "$options": "i"}
        }

    if plan.get("entity_label_keyword"):
        filt["normalized_entities.label"] = {
            "$regex": plan["entity_label_keyword"], "$options": "i"
        }

    if plan.get("entity_type"):
        filt["normalized_entities.entity_type"] = plan["entity_type"]

    if plan.get("min_people") is not None:
        filt["llm_context.people_count"] = {"$gte": plan["min_people"]}

    if plan.get("min_vehicles") is not None:
        filt["llm_context.vehicles_count"] = {"$gte": plan["min_vehicles"]}

    after_h = plan.get("after_hour")
    before_h = plan.get("before_hour")
    if after_h is not None or before_h is not None:
        expr_parts = []
        if after_h is not None:
            expr_parts.append({"$gte": [{"$hour": "$video_timestamp"}, after_h]})
        if before_h is not None:
            expr_parts.append({"$lte": [{"$hour": "$video_timestamp"}, before_h]})
        filt["$expr"] = {"$and": expr_parts} if len(expr_parts) > 1 else expr_parts

    return filt


# ---------------------------------------------------------------------------
# Step 3: Answer synthesiser
# ---------------------------------------------------------------------------

_SYNTH_SYSTEM = """\
You are a security analyst assistant. You have retrieved a set of drone camera frame records
that match the operator's query. Synthesise a clear, factual answer based only on the evidence provided.
Mention specific timestamps, zones, and entities where relevant.
Keep the answer to 2–4 sentences. Do not invent information not present in the frames.
"""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
def _synthesise_answer(question: str, frames: list[dict]) -> str:
    if not frames:
        return "No matching frames were found in the indexed data for this query."

    evidence = "\n".join(
        f"[{str(f.get('video_timestamp', '?'))[:19]}] "
        f"zone={f.get('telemetry', {}).get('zone_name', '?')} — "
        f"{f.get('llm_context', {}).get('scene_summary', '')}"
        for f in frames[:20]
    )

    response = _client.invoke([
        ("system", _SYNTH_SYSTEM),
        (
            "human",
            (
                f"Operator question: {question}\n\n"
                f"Matching frame evidence ({len(frames)} frames):\n{evidence}\n\n"
                "Answer the question based on the evidence."
            ),
        ),
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def answer_question(video_id: str, question: str) -> dict:
    """
    Answer a natural language question over indexed frame data.

    Returns:
        dict with keys: question, answer, supporting_frames, frame_count
    """
    log.info(f"[QAAgent] Question for {video_id}: {question!r}")

    try:
        plan = _plan_query(question, video_id)
        log.info(f"[QAAgent] Query plan: {plan}")
    except Exception as exc:
        log.error(f"[QAAgent] Query planning failed: {exc}", exc_info=True)
        return {
            "question": question,
            "answer": "Failed to plan the query. Please try rephrasing.",
            "supporting_frames": [],
            "frame_count": 0,
        }

    limit = min(int(plan.get("limit", 10)), 50)
    mongo_filter = _build_mongo_filter(video_id, plan)
    log.debug(f"[QAAgent] MongoDB filter: {mongo_filter}")

    cursor = frames_col().find(
        mongo_filter,
        sort=[("video_timestamp", 1)],
    ).limit(limit)
    frames = await cursor.to_list(length=limit)
    log.info(f"[QAAgent] Retrieved {len(frames)} matching frames")

    try:
        answer = _synthesise_answer(question, frames)
    except Exception as exc:
        log.error(f"[QAAgent] Synthesis failed: {exc}", exc_info=True)
        answer = f"Found {len(frames)} matching frames but could not synthesise an answer."

    supporting = [
        {
            "_id": str(f.get("_id")),
            "frame_no": f.get("frame_no"),
            "video_timestamp": str(f.get("video_timestamp", "")),
            "zone_name": f.get("telemetry", {}).get("zone_name", ""),
            "scene_summary": f.get("llm_context", {}).get("scene_summary", ""),
            "frame_path": f.get("frame_path", ""),
        }
        for f in frames
    ]

    log.info(f"[QAAgent] Answer: {answer[:120]}")
    return {
        "question": question,
        "answer": answer,
        "supporting_frames": supporting,
        "frame_count": len(frames),
    }