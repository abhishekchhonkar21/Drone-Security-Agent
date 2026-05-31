"""
app/pipeline/ingestion.py
Orchestrates the full ingestion pipeline for a video session:
  1. Frame sampling
  2. Telemetry merging
  3. VLM captioning (GPT-4o Vision)
  4. Entity normalization
  5. MongoDB persistence
  6. Alert agent trigger (sliding window)

Designed to run as a background task (FastAPI BackgroundTasks).
"""
import uuid
from datetime import datetime, timezone
from app.config import settings
from app.db.mongo import frames_col, sessions_col
from app.models.schemas import (
    FrameDocument, VideoSessionDocument, IngestionMeta, TelemetryData,
)
from app.pipeline.frame_sampler import sample_frames
from app.pipeline.telemetry_merger import TelemetryMerger
from app.pipeline.vlm_captioner import generate_frame_description
from app.pipeline.entity_normalizer import normalize_entities
from app.agents.alert_agent import run_alert_check
from app.utils.logger import get_logger

log = get_logger(__name__)


async def run_ingestion(
    video_id: str,
    drone_id: str,
    video_path: str,
    telemetry_path: str,
) -> None:
    """
    Full ingestion pipeline. Called as a background task.
    Updates the video_sessions document throughout processing.
    """
    log.info(f"[Ingestion] ▶ Starting ingestion for video_id={video_id}")
    started_at = datetime.now(tz=timezone.utc)

    # Mark session as processing
    await sessions_col().update_one(
        {"_id": video_id},
        {"$set": {"status": "processing", "session_start": started_at}},
    )

    try:
        # ------------------------------------------------------------------ #
        # Stage 1: Sample frames
        # ------------------------------------------------------------------ #
        log.info(f"[Ingestion] Stage 1 — Sampling frames from: {video_path}")
        sampled = sample_frames(video_path, video_id)
        log.info(f"[Ingestion] Stage 1 ✓  — {len(sampled)} frames sampled")

        await sessions_col().update_one(
            {"_id": video_id},
            {"$set": {"frame_count": len(sampled)}},
        )

        # ------------------------------------------------------------------ #
        # Stage 2: Load telemetry merger
        # ------------------------------------------------------------------ #
        log.info(f"[Ingestion] Stage 2 — Loading telemetry from: {telemetry_path}")
        merger = TelemetryMerger(telemetry_path)
        log.info("[Ingestion] Stage 2 ✓  — Telemetry loaded")

        # ------------------------------------------------------------------ #
        # Stage 3–5: Per-frame captioning + normalization + storage
        # ------------------------------------------------------------------ #
        log.info("[Ingestion] Stage 3–5 — Captioning and storing frames …")
        zones_seen: set[str] = set()
        inserted_frame_ids: list[str] = []

        for idx, sf in enumerate(sampled, start=1):
            frame_id = str(uuid.uuid4())
            sf.frame_id = frame_id

            # Telemetry merge
            telemetry = merger.get_for_timestamp(sf.video_timestamp)
            zones_seen.add(telemetry.zone_name)

            # VLM caption
            log.info(
                f"[Ingestion] Frame {idx}/{len(sampled)} "
                f"(zone={telemetry.zone_name}) — calling GPT-4o Vision …"
            )
            llm_context = generate_frame_description(
                frame_path=sf.frame_path,
                telemetry=telemetry,
                video_timestamp=sf.video_timestamp,
            )

            # Entity normalization
            entities = normalize_entities(llm_context)

            # Build document
            doc = FrameDocument(
                _id=frame_id,
                video_id=video_id,
                drone_id=drone_id,
                frame_no=sf.frame_no,
                frame_path=sf.frame_path,
                video_timestamp=sf.video_timestamp,
                telemetry=telemetry,
                llm_context=llm_context,
                normalized_entities=entities,
                ingestion_meta=IngestionMeta(
                    model_name=settings.azure_openai_deployment,
                    pipeline_version="v1",
                ),
            )

            # Persist frame to MongoDB
            await frames_col().insert_one(doc.model_dump(by_alias=True))
            inserted_frame_ids.append(frame_id)

            log.info(
                f"[Ingestion] ✓ Frame {idx}/{len(sampled)} stored | "
                f"summary: {llm_context.scene_summary[:80]}"
            )

            # ---------------------------------------------------------------- #
            # Stage 6: Sliding-window alert check
            # Every N frames, hand the last window to the alert agent
            # ---------------------------------------------------------------- #
            window = settings.alert_window_size
            if len(inserted_frame_ids) >= window and idx % window == 0:
                log.info(f"[Ingestion] Triggering alert check (window={window})")
                await run_alert_check(
                    video_id=video_id,
                    recent_frame_ids=inserted_frame_ids[-window:],
                )

        # Final alert pass over tail frames
        tail = inserted_frame_ids[-settings.alert_window_size:]
        if tail:
            log.info("[Ingestion] Final tail alert check …")
            await run_alert_check(video_id=video_id, recent_frame_ids=tail)

        # ------------------------------------------------------------------ #
        # Finalise session
        # ------------------------------------------------------------------ #
        ended_at = datetime.now(tz=timezone.utc)
        await sessions_col().update_one(
            {"_id": video_id},
            {
                "$set": {
                    "status": "processed",
                    "session_end": ended_at,
                    "sampled_frame_count": len(sampled),
                    "zones_visited": list(zones_seen),
                }
            },
        )
        log.info(
            f"[Ingestion] ✅ Ingestion complete for video_id={video_id} | "
            f"frames={len(sampled)} | zones={zones_seen} | "
            f"duration={(ended_at - started_at).total_seconds():.1f}s"
        )

    except Exception as exc:
        log.error(f"[Ingestion] ❌ Pipeline failed for {video_id}: {exc}", exc_info=True)
        await sessions_col().update_one(
            {"_id": video_id},
            {"$set": {"status": "failed", "error_message": str(exc)[:500]}},
        )
        raise
