"""
tests/test_api.py
Integration tests for FastAPI endpoints using TestClient.
MongoDB and LLM calls are mocked so these run without external services.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone

# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def mock_settings(monkeypatch):
    """Patch settings so no real .env is needed."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("MONGODB_DB_NAME", "drone_test")


@pytest.fixture
def client(mock_settings):
    """
    Build a TestClient with DB and background tasks mocked out.
    """
    with patch("app.db.mongo.connect_db", new_callable=AsyncMock), \
         patch("app.db.mongo.close_db", new_callable=AsyncMock), \
         patch("app.db.mongo._db") as mock_db:

        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── Health ─────────────────────────────────────────────────────────
class TestHealth:

    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ── Ingest session ─────────────────────────────────────────────────
class TestIngestSession:

    @patch("app.api.routes.sessions_col")
    @patch("app.api.routes.run_ingestion")
    def test_ingest_session_starts(self, mock_ingest, mock_sessions, client):
        # Simulate no existing session
        mock_col = AsyncMock()
        mock_col.find_one = AsyncMock(return_value=None)
        mock_col.replace_one = AsyncMock()
        mock_sessions.return_value = mock_col

        r = client.post("/api/ingest/session", json={
            "video_id": "vid_test",
            "drone_id": "drone_1",
            "video_path": "input/demo.mp4",
            "telemetry_path": "input/demo.json",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["video_id"] == "vid_test"
        assert data["status"] == "pending"

    @patch("app.api.routes.sessions_col")
    def test_ingest_duplicate_session_returns_409(self, mock_sessions, client):
        mock_col = AsyncMock()
        mock_col.find_one = AsyncMock(return_value={
            "_id": "vid_test", "status": "processed"
        })
        mock_sessions.return_value = mock_col

        r = client.post("/api/ingest/session", json={
            "video_id": "vid_test",
            "drone_id": "drone_1",
            "video_path": "input/demo.mp4",
            "telemetry_path": "input/demo.json",
        })
        assert r.status_code == 409


# ── Get session ────────────────────────────────────────────────────
class TestGetSession:

    @patch("app.api.routes.sessions_col")
    def test_get_existing_session(self, mock_sessions, client):
        mock_col = AsyncMock()
        mock_col.find_one = AsyncMock(return_value={
            "_id": "vid_001",
            "drone_id": "drone_alpha",
            "status": "processed",
            "sampled_frame_count": 30,
            "session_start": datetime(2026, 5, 30, 0, 0, 0, tzinfo=timezone.utc),
        })
        mock_sessions.return_value = mock_col

        r = client.get("/api/sessions/vid_001")
        assert r.status_code == 200
        data = r.json()
        assert data["_id"] == "vid_001"
        assert data["status"] == "processed"

    @patch("app.api.routes.sessions_col")
    def test_get_nonexistent_session_returns_404(self, mock_sessions, client):
        mock_col = AsyncMock()
        mock_col.find_one = AsyncMock(return_value=None)
        mock_sessions.return_value = mock_col

        r = client.get("/api/sessions/nonexistent")
        assert r.status_code == 404


# ── Frames ─────────────────────────────────────────────────────────
class TestFrames:

    @patch("app.api.routes.frames_col")
    def test_get_frames_paginated(self, mock_frames, client):
        mock_col = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.skip.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.to_list = AsyncMock(return_value=[
            {
                "_id": "frame_001",
                "video_id": "vid_001",
                "frame_no": 1,
                "video_timestamp": datetime(2026, 5, 30, 0, 0, 1, tzinfo=timezone.utc),
                "telemetry": {"zone_name": "main_gate", "latitude": 18.5, "longitude": 73.8, "altitude": 12.0},
                "llm_context": {"scene_summary": "Clear area.", "objects_detected": []},
                "normalized_entities": [],
            }
        ])
        mock_col.find.return_value = mock_cursor
        mock_col.count_documents = AsyncMock(return_value=1)
        mock_frames.return_value = mock_col

        r = client.get("/api/sessions/vid_001/frames?page=1&page_size=20")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert len(data["frames"]) == 1


# ── Alerts ─────────────────────────────────────────────────────────
class TestAlerts:

    @patch("app.api.routes.alerts_col")
    def test_get_alerts(self, mock_alerts, client):
        mock_col = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {
                "_id": "alert_001",
                "video_id": "vid_001",
                "alert_type": "loitering",
                "severity": "high",
                "title": "Person loitering",
                "description": "Test alert",
                "trigger_timestamp": datetime(2026, 5, 30, 0, 1, 0, tzinfo=timezone.utc),
                "zone_name": "main_gate",
                "status": "new",
                "frame_ids": ["f1"],
            }
        ])
        mock_col.find.return_value = mock_cursor
        mock_alerts.return_value = mock_col

        r = client.get("/api/sessions/vid_001/alerts")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["alerts"][0]["alert_type"] == "loitering"


# ── Search frames ──────────────────────────────────────────────────
class TestFrameSearch:

    @patch("app.api.routes.frames_col")
    def test_search_by_zone(self, mock_frames, client):
        mock_col = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.skip.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_col.find.return_value = mock_cursor
        mock_col.count_documents = AsyncMock(return_value=0)
        mock_frames.return_value = mock_col

        r = client.get("/api/frames/search?zone=main_gate&video_id=vid_001")
        assert r.status_code == 200
        data = r.json()
        assert "frames" in data
        assert data["total"] == 0

    @patch("app.api.routes.frames_col")
    def test_search_by_object_keyword(self, mock_frames, client):
        mock_col = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.skip.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_col.find.return_value = mock_cursor
        mock_col.count_documents = AsyncMock(return_value=0)
        mock_frames.return_value = mock_col

        r = client.get("/api/frames/search?object_kw=truck")
        assert r.status_code == 200
