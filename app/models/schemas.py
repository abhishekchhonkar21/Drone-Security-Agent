"""
app/models/schemas.py
Pydantic models for all data structures: frames, alerts, sessions, requests, responses.
These are shared between the pipeline, agents, and API layers.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field
import uuid


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class TelemetryData(BaseModel):
    latitude: float
    longitude: float
    altitude: float
    zone_name: str
    speed: Optional[float] = None
    heading: Optional[float] = None


class LLMContext(BaseModel):
    scene_summary: str
    objects_detected: list[str] = Field(default_factory=list)
    people_count: int = 0
    vehicles_count: int = 0
    vehicle_attributes: list[str] = Field(default_factory=list)
    activity_detected: list[str] = Field(default_factory=list)
    risk_signals: list[str] = Field(default_factory=list)
    confidence_notes: str = ""


class NormalizedEntity(BaseModel):
    entity_type: str          # person | vehicle | object
    label: str
    subtype: Optional[str] = None
    color: Optional[str] = None
    make_model: Optional[str] = None


class IngestionMeta(BaseModel):
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    model_name: str
    pipeline_version: str = "v1"


class AlertReasoningTrace(BaseModel):
    prompt_version: str
    supporting_signals: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Collection documents
# ---------------------------------------------------------------------------

class FrameDocument(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    video_id: str
    drone_id: str
    frame_no: int
    frame_path: str
    video_timestamp: datetime
    telemetry: TelemetryData
    llm_context: LLMContext
    normalized_entities: list[NormalizedEntity] = Field(default_factory=list)
    ingestion_meta: IngestionMeta

    class Config:
        populate_by_name = True


class AlertDocument(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    video_id: str
    frame_ids: list[str]
    alert_type: str
    severity: str              # low | medium | high | critical
    title: str
    description: str
    trigger_timestamp: datetime
    zone_name: str
    status: str = "new"        # new | acknowledged | resolved
    generated_by: str = "llm_alert_agent"
    reasoning_trace: AlertReasoningTrace

    class Config:
        populate_by_name = True


class VideoSessionDocument(BaseModel):
    id: str = Field(alias="_id")
    drone_id: str
    source_video_path: str
    session_start: Optional[datetime] = None
    session_end: Optional[datetime] = None
    status: str = "pending"    # pending | processing | processed | failed
    frame_count: int = 0
    sampled_frame_count: int = 0
    summary_text: Optional[str] = None
    zones_visited: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None

    class Config:
        populate_by_name = True


# ---------------------------------------------------------------------------
# API Request / Response models
# ---------------------------------------------------------------------------

class IngestSessionRequest(BaseModel):
    video_id: str
    drone_id: str
    video_path: str
    telemetry_path: str


class IngestSessionResponse(BaseModel):
    video_id: str
    status: str
    message: str


class FrameSearchParams(BaseModel):
    video_id: Optional[str] = None
    zone_name: Optional[str] = None
    object_keyword: Optional[str] = None
    after: Optional[datetime] = None
    before: Optional[datetime] = None
    page: int = 1
    page_size: int = 20


class AskQuestionRequest(BaseModel):
    question: str


class AskQuestionResponse(BaseModel):
    question: str
    answer: str
    supporting_frames: list[dict[str, Any]] = Field(default_factory=list)
    frame_count: int = 0


class SummaryResponse(BaseModel):
    video_id: str
    summary_text: str
    frame_count: int
    alert_count: int
    zones_visited: list[str]


class AlertListResponse(BaseModel):
    video_id: str
    alerts: list[dict[str, Any]]
    total: int


class TelemetryRecord(BaseModel):
    """Single record from the simulated telemetry JSON file."""
    drone_id: str
    video_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    altitude: float
    zone_name: str
    speed: Optional[float] = None
    heading: Optional[float] = None
