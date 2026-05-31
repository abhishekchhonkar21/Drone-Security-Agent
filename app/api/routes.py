"""
app/api/routes.py
All FastAPI route handlers. Thin layer — delegates to pipeline/agents.
"""
import uuid
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse

from app.db.mongo import frames_col, alerts_col, sessions_col
from app.models.schemas import (
    IngestSessionRequest, IngestSessionResponse,
    AskQuestionRequest, AskQuestionResponse,
    VideoSessionDocument,
)
from app.pipeline.ingestion import run_ingestion
from app.agents.summary_agent import generate_session_summary
from app.agents.qa_agent import answer_question
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _serialize(doc: dict) -> dict:
    """Convert MongoDB document to JSON-serialisable dict."""
    if doc is None:
        return {}
    doc["_id"] = str(doc.get("_id", ""))
    for k, v in doc.items():
        if isinstance(v, datetime):
            doc[k] = v.isoformat()
    return doc


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@router.post("/ingest/session", response_model=IngestSessionResponse)
async def ingest_session(
    req: IngestSessionRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start ingestion of a video + telemetry session.
    Runs the pipeline as a background task and returns immediately.
    """
    log.info(f"[API] POST /ingest/session | video_id={req.video_id}")

    # Check for duplicate session
    existing = await sessions_col().find_one({"_id": req.video_id})
    if existing and existing.get("status") in ("processing", "processed"):
        raise HTTPException(
            status_code=409,
            detail=f"Session {req.video_id} already exists with status={existing['status']}",
        )

    # Create session document
    session = VideoSessionDocument(
        _id=req.video_id,
        drone_id=req.drone_id,
        source_video_path=req.video_path,
        status="pending",
    )
    await sessions_col().replace_one(
        {"_id": req.video_id},
        session.model_dump(by_alias=True),
        upsert=True,
    )

    # Fire background pipeline
    background_tasks.add_task(
        run_ingestion,
        video_id=req.video_id,
        drone_id=req.drone_id,
        video_path=req.video_path,
        telemetry_path=req.telemetry_path,
    )

    log.info(f"[API] Ingestion started for {req.video_id}")
    return IngestSessionResponse(
        video_id=req.video_id,
        status="pending",
        message="Ingestion started in background. Poll /api/sessions/{video_id} for status.",
    )


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@router.get("/sessions")
async def list_sessions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """List all video sessions, newest first."""
    skip = (page - 1) * page_size
    cursor = sessions_col().find({}, sort=[("session_start", -1)]).skip(skip).limit(page_size)
    sessions = await cursor.to_list(length=page_size)
    total = await sessions_col().count_documents({})
    return {"sessions": [_serialize(s) for s in sessions], "total": total, "page": page}


@router.get("/sessions/{video_id}")
async def get_session(video_id: str):
    """Return session metadata and status."""
    log.info(f"[API] GET /sessions/{video_id}")
    doc = await sessions_col().find_one({"_id": video_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Session {video_id} not found")
    return _serialize(doc)


@router.get("/sessions/{video_id}/frames")
async def get_frames(
    video_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """Return paginated indexed frames for a session."""
    log.info(f"[API] GET /sessions/{video_id}/frames (page={page})")
    skip = (page - 1) * page_size
    cursor = (
        frames_col()
        .find({"video_id": video_id}, sort=[("video_timestamp", 1)])
        .skip(skip)
        .limit(page_size)
    )
    frames = await cursor.to_list(length=page_size)
    total = await frames_col().count_documents({"video_id": video_id})
    return {
        "video_id": video_id,
        "frames": [_serialize(f) for f in frames],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/sessions/{video_id}/alerts")
async def get_alerts(video_id: str):
    """Return all alerts for a session."""
    log.info(f"[API] GET /sessions/{video_id}/alerts")
    cursor = alerts_col().find(
        {"video_id": video_id}, sort=[("trigger_timestamp", -1)]
    )
    alerts = await cursor.to_list(length=500)
    return {
        "video_id": video_id,
        "alerts": [_serialize(a) for a in alerts],
        "total": len(alerts),
    }


@router.patch("/sessions/{video_id}/alerts/{alert_id}/status")
async def update_alert_status(video_id: str, alert_id: str, status: str = Query(...)):
    """Update alert status (new → acknowledged → resolved)."""
    valid = {"new", "acknowledged", "resolved"}
    if status not in valid:
        raise HTTPException(status_code=400, detail=f"status must be one of {valid}")
    result = await alerts_col().update_one(
        {"_id": alert_id, "video_id": video_id},
        {"$set": {"status": status}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"alert_id": alert_id, "status": status}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@router.post("/sessions/{video_id}/summary")
async def session_summary(video_id: str):
    """Generate (or refresh) a session summary using the Summary Agent."""
    log.info(f"[API] POST /sessions/{video_id}/summary")
    doc = await sessions_col().find_one({"_id": video_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Session {video_id} not found")
    if doc.get("status") not in ("processed", "processing"):
        raise HTTPException(
            status_code=400,
            detail=f"Session must be processed first (current status: {doc.get('status')})",
        )
    result = await generate_session_summary(video_id)
    result["video_id"] = video_id
    return result


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------

@router.post("/sessions/{video_id}/ask", response_model=AskQuestionResponse)
async def ask_question(video_id: str, req: AskQuestionRequest):
    """Answer a follow-up question over indexed frame data."""
    log.info(f"[API] POST /sessions/{video_id}/ask | question={req.question!r}")
    doc = await sessions_col().find_one({"_id": video_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Session {video_id} not found")
    result = await answer_question(video_id=video_id, question=req.question)
    return AskQuestionResponse(**result)


# ---------------------------------------------------------------------------
# Frame search
# ---------------------------------------------------------------------------

@router.get("/frames/search")
async def search_frames(
    video_id: str | None = Query(default=None),
    zone: str | None = Query(default=None),
    object_kw: str | None = Query(default=None),
    activity_kw: str | None = Query(default=None),
    after: str | None = Query(default=None, description="ISO datetime"),
    before: str | None = Query(default=None, description="ISO datetime"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """Search frames by object keyword, zone, activity, or time range."""
    log.info(f"[API] GET /frames/search | video_id={video_id} zone={zone} obj={object_kw}")

    filt: dict = {}
    if video_id:
        filt["video_id"] = video_id
    if zone:
        filt["telemetry.zone_name"] = {"$regex": zone, "$options": "i"}
    if object_kw:
        filt["llm_context.objects_detected"] = {
            "$elemMatch": {"$regex": object_kw, "$options": "i"}
        }
    if activity_kw:
        filt["llm_context.activity_detected"] = {
            "$elemMatch": {"$regex": activity_kw, "$options": "i"}
        }
    if after or before:
        ts_filter: dict = {}
        if after:
            ts_filter["$gte"] = datetime.fromisoformat(after)
        if before:
            ts_filter["$lte"] = datetime.fromisoformat(before)
        filt["video_timestamp"] = ts_filter

    skip = (page - 1) * page_size
    cursor = (
        frames_col()
        .find(filt, sort=[("video_timestamp", 1)])
        .skip(skip)
        .limit(page_size)
    )
    frames = await cursor.to_list(length=page_size)
    total = await frames_col().count_documents(filt)

    return {
        "frames": [_serialize(f) for f in frames],
        "total": total,
        "page": page,
        "page_size": page_size,
        "filters_applied": filt,
    }
