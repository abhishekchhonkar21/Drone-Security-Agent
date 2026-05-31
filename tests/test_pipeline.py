"""
tests/test_pipeline.py
Unit tests for the ingestion pipeline stages.
Uses mocking to avoid real API calls and DB writes.
"""
import json
import os
import tempfile
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from pathlib import Path


# ── Frame Sampler ──────────────────────────────────────────────────
class TestFrameSampler:

    def test_sample_from_directory(self, tmp_path):
        """Should import frames from a directory of images."""
        import cv2
        import numpy as np
        from app.pipeline.frame_sampler import sample_frames

        # Create 3 dummy JPEG files
        frames_dir = tmp_path / "frames_input"
        frames_dir.mkdir()
        for i in range(3):
            img = np.zeros((360, 640, 3), dtype="uint8")
            cv2.imwrite(str(frames_dir / f"frame_{i:04d}.jpg"), img)

        result = sample_frames(
            video_path=str(frames_dir),
            video_id="test_vid",
            interval_sec=1,
        )
        assert len(result) == 3
        assert result[0].frame_no == 1
        assert result[2].frame_no == 3
        for sf in result:
            assert Path(sf.frame_path).exists()

    def test_missing_video_raises(self):
        """Should raise FileNotFoundError for nonexistent video."""
        from app.pipeline.frame_sampler import sample_frames
        with pytest.raises(FileNotFoundError):
            sample_frames("/nonexistent/video.mp4", "vid_test")


# ── Telemetry Merger ───────────────────────────────────────────────
class TestTelemetryMerger:

    def _write_telemetry(self, path: Path, records: list) -> None:
        path.write_text(json.dumps(records))

    def test_exact_match(self, tmp_path):
        from app.pipeline.telemetry_merger import TelemetryMerger

        records = [
            {
                "drone_id": "d1", "video_id": "v1",
                "timestamp": "2026-05-30T00:00:05+00:00",
                "latitude": 18.5, "longitude": 73.8,
                "altitude": 12.0, "zone_name": "main_gate",
            }
        ]
        p = tmp_path / "tel.json"
        self._write_telemetry(p, records)
        merger = TelemetryMerger(str(p))
        ts = datetime(2026, 5, 30, 0, 0, 5, tzinfo=timezone.utc)
        result = merger.get_for_timestamp(ts)
        assert result.zone_name == "main_gate"
        assert result.latitude == 18.5

    def test_nearest_match(self, tmp_path):
        from app.pipeline.telemetry_merger import TelemetryMerger

        records = [
            {
                "drone_id": "d1", "video_id": "v1",
                "timestamp": "2026-05-30T00:00:00+00:00",
                "latitude": 18.1, "longitude": 73.1,
                "altitude": 10.0, "zone_name": "garage",
            },
            {
                "drone_id": "d1", "video_id": "v1",
                "timestamp": "2026-05-30T00:00:10+00:00",
                "latitude": 18.2, "longitude": 73.2,
                "altitude": 11.0, "zone_name": "north_fence",
            },
        ]
        p = tmp_path / "tel.json"
        self._write_telemetry(p, records)
        merger = TelemetryMerger(str(p))

        # Query at t=8s — closer to t=10s record
        ts = datetime(2026, 5, 30, 0, 0, 8, tzinfo=timezone.utc)
        result = merger.get_for_timestamp(ts)
        assert result.zone_name == "north_fence"

    def test_fallback_when_out_of_tolerance(self, tmp_path):
        from app.pipeline.telemetry_merger import TelemetryMerger

        records = [
            {
                "drone_id": "d1", "video_id": "v1",
                "timestamp": "2026-05-30T00:00:00+00:00",
                "latitude": 18.1, "longitude": 73.1,
                "altitude": 10.0, "zone_name": "garage",
            }
        ]
        p = tmp_path / "tel.json"
        self._write_telemetry(p, records)
        merger = TelemetryMerger(str(p))

        # Query 60 seconds away — beyond default tolerance of 5s
        ts = datetime(2026, 5, 30, 0, 1, 0, tzinfo=timezone.utc)
        result = merger.get_for_timestamp(ts)
        # Fallback zone
        assert result.zone_name == "unknown"

    def test_empty_telemetry_returns_fallback(self, tmp_path):
        from app.pipeline.telemetry_merger import TelemetryMerger

        p = tmp_path / "empty.json"
        p.write_text("[]")
        merger = TelemetryMerger(str(p))
        ts = datetime(2026, 5, 30, 0, 0, 0, tzinfo=timezone.utc)
        result = merger.get_for_timestamp(ts)
        assert result.zone_name == "unknown"


# ── Entity Normalizer ──────────────────────────────────────────────
class TestEntityNormalizer:

    def _make_ctx(self, **kwargs):
        from app.models.schemas import LLMContext
        defaults = dict(
            scene_summary="test",
            objects_detected=[],
            people_count=0,
            vehicles_count=0,
            vehicle_attributes=[],
            activity_detected=[],
            risk_signals=[],
            confidence_notes="",
        )
        defaults.update(kwargs)
        return LLMContext(**defaults)

    def test_person_extraction(self):
        from app.pipeline.entity_normalizer import normalize_entities

        ctx = self._make_ctx(people_count=2)
        entities = normalize_entities(ctx)
        person_entities = [e for e in entities if e.entity_type == "person"]
        assert len(person_entities) == 2
        assert person_entities[0].label == "person_1"
        assert person_entities[1].label == "person_2"

    def test_vehicle_extraction_from_attributes(self):
        from app.pipeline.entity_normalizer import normalize_entities

        ctx = self._make_ctx(vehicle_attributes=["blue pickup truck", "black sedan"])
        entities = normalize_entities(ctx)
        vehicle_entities = [e for e in entities if e.entity_type == "vehicle"]
        assert len(vehicle_entities) == 2
        labels = [e.label for e in vehicle_entities]
        assert any("blue" in l for l in labels)
        assert any("black" in l for l in labels)

    def test_vehicle_color_extraction(self):
        from app.pipeline.entity_normalizer import normalize_entities

        ctx = self._make_ctx(vehicle_attributes=["red ford truck"])
        entities = normalize_entities(ctx)
        vehicle = next(e for e in entities if e.entity_type == "vehicle")
        assert vehicle.color == "red"

    def test_vehicle_make_extraction(self):
        from app.pipeline.entity_normalizer import normalize_entities

        ctx = self._make_ctx(vehicle_attributes=["blue toyota suv"])
        entities = normalize_entities(ctx)
        vehicle = next(e for e in entities if e.entity_type == "vehicle")
        assert vehicle.make_model == "Toyota"

    def test_vehicle_from_objects_detected_fallback(self):
        from app.pipeline.entity_normalizer import normalize_entities

        ctx = self._make_ctx(objects_detected=["blue truck", "gate"])
        entities = normalize_entities(ctx)
        vehicle_entities = [e for e in entities if e.entity_type == "vehicle"]
        assert len(vehicle_entities) == 1

    def test_no_entities_empty_context(self):
        from app.pipeline.entity_normalizer import normalize_entities

        ctx = self._make_ctx()
        entities = normalize_entities(ctx)
        assert entities == []

    def test_deduplicated_labels(self):
        from app.pipeline.entity_normalizer import normalize_entities

        ctx = self._make_ctx(vehicle_attributes=["blue truck", "blue truck", "blue truck"])
        entities = normalize_entities(ctx)
        labels = [e.label for e in entities]
        assert len(labels) == len(set(labels)), "Labels should be unique"


# ── VLM Captioner JSON Parser ──────────────────────────────────────
class TestVLMCaptionerParser:

    def test_parse_valid_json(self):
        from app.pipeline.vlm_captioner import _parse_llm_json

        raw = json.dumps({
            "scene_summary": "A person near the gate.",
            "objects_detected": ["person", "gate"],
            "people_count": 1,
            "vehicles_count": 0,
            "vehicle_attributes": [],
            "activity_detected": ["person_standing"],
            "risk_signals": ["person_near_gate"],
            "confidence_notes": "",
        })
        ctx = _parse_llm_json(raw)
        assert ctx.scene_summary == "A person near the gate."
        assert "person" in ctx.objects_detected
        assert ctx.people_count == 1

    def test_parse_with_markdown_fences(self):
        from app.pipeline.vlm_captioner import _parse_llm_json

        raw = '```json\n{"scene_summary":"test","objects_detected":[],"people_count":0,"vehicles_count":0,"vehicle_attributes":[],"activity_detected":[],"risk_signals":[],"confidence_notes":""}\n```'
        ctx = _parse_llm_json(raw)
        assert ctx.scene_summary == "test"

    def test_parse_invalid_returns_fallback(self):
        from app.pipeline.vlm_captioner import _parse_llm_json

        ctx = _parse_llm_json("this is not json at all!!!")
        assert "Parse error" in ctx.scene_summary or "error" in ctx.confidence_notes.lower()


# ── Alert Agent JSON Parser ────────────────────────────────────────
class TestAlertAgentParser:

    def test_parse_valid_alert(self):
        from app.agents.alert_agent import _parse_alert_json

        raw = json.dumps({
            "is_alert": True,
            "alert_type": "loitering",
            "severity": "high",
            "title": "Person loitering at main gate",
            "description": "A person remained near the main gate for 5 consecutive frames.",
            "evidence_frame_ids": ["frame_001", "frame_002"],
            "reasoning": "Repeated presence in sensitive zone at night.",
        })
        result = _parse_alert_json(raw)
        assert result["is_alert"] is True
        assert result["alert_type"] == "loitering"
        assert result["severity"] == "high"

    def test_parse_no_alert(self):
        from app.agents.alert_agent import _parse_alert_json

        raw = json.dumps({
            "is_alert": False,
            "alert_type": "unknown",
            "severity": "low",
            "title": "",
            "description": "",
            "evidence_frame_ids": [],
            "reasoning": "No suspicious activity.",
        })
        result = _parse_alert_json(raw)
        assert result["is_alert"] is False

    def test_parse_invalid_returns_none(self):
        from app.agents.alert_agent import _parse_alert_json

        result = _parse_alert_json("not valid json {{{{")
        assert result is None

    def test_parse_strips_markdown_fences(self):
        from app.agents.alert_agent import _parse_alert_json

        raw = '```json\n{"is_alert":false,"alert_type":"unknown","severity":"low","title":"","description":"","evidence_frame_ids":[],"reasoning":"ok"}\n```'
        result = _parse_alert_json(raw)
        assert result is not None
        assert result["is_alert"] is False


# ── Pydantic Models ────────────────────────────────────────────────
class TestSchemas:

    def test_frame_document_serialises(self):
        from app.models.schemas import (
            FrameDocument, TelemetryData, LLMContext, IngestionMeta
        )

        frame = FrameDocument(
            _id="test-uuid",
            video_id="vid_001",
            drone_id="drone_alpha",
            frame_no=1,
            frame_path="storage/videos/vid_001/frames/frame_0001.jpg",
            video_timestamp=datetime(2026, 5, 30, 0, 0, 0, tzinfo=timezone.utc),
            telemetry=TelemetryData(
                latitude=18.5, longitude=73.8, altitude=12.0, zone_name="main_gate"
            ),
            llm_context=LLMContext(
                scene_summary="A person near the gate.",
                objects_detected=["person"],
                people_count=1,
            ),
            ingestion_meta=IngestionMeta(model_name="gpt-4o"),
        )
        dumped = frame.model_dump(by_alias=True)
        assert dumped["_id"] == "test-uuid"
        assert dumped["video_id"] == "vid_001"
        assert dumped["telemetry"]["zone_name"] == "main_gate"

    def test_alert_document_defaults(self):
        from app.models.schemas import AlertDocument, AlertReasoningTrace

        alert = AlertDocument(
            _id="alert-uuid",
            video_id="vid_001",
            frame_ids=["f1", "f2"],
            alert_type="loitering",
            severity="high",
            title="Test Alert",
            description="Test description",
            trigger_timestamp=datetime(2026, 5, 30, 0, 1, 0, tzinfo=timezone.utc),
            zone_name="main_gate",
            reasoning_trace=AlertReasoningTrace(
                prompt_version="alert_v1",
                supporting_signals=["night", "repeat"],
            ),
        )
        assert alert.status == "new"
        assert alert.generated_by == "llm_alert_agent"
