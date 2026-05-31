"""
tests/test_agents.py
Unit tests for alert, summary, and QA agents.
All external I/O (MongoDB, OpenAI/Azure OpenAI) is mocked.
"""
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def make_frame(frame_id: str, zone: str = "main_gate", risk: list | None = None, people: int = 0) -> dict:
    return {
        "_id": frame_id,
        "video_id": "vid_001",
        "video_timestamp": datetime(2026, 5, 30, 0, 1, 0, tzinfo=timezone.utc),
        "telemetry": {
            "latitude": 18.5,
            "longitude": 73.8,
            "altitude": 12.0,
            "zone_name": zone,
        },
        "llm_context": {
            "scene_summary": f"Scene in {zone}",
            "objects_detected": ["person"] if people else [],
            "people_count": people,
            "vehicles_count": 0,
            "activity_detected": [],
            "risk_signals": risk or [],
        },
        "normalized_entities": [],
    }


class TestAlertAgent:

    @pytest.mark.asyncio
    @patch("app.agents.alert_agent.alerts_col")
    @patch("app.agents.alert_agent.frames_col")
    @patch("app.agents.alert_agent._call_alert_llm")
    async def test_alert_is_persisted_when_raised(
        self, mock_llm, mock_frames_col, mock_alerts_col
    ):
        frames = [make_frame(f"f{i}", "main_gate", ["person_near_gate"], 1) for i in range(3)]

        mock_frames_cursor = MagicMock()
        mock_frames_cursor.to_list = AsyncMock(return_value=frames)

        mock_fc = MagicMock()
        mock_fc.find.return_value = mock_frames_cursor
        mock_frames_col.return_value = mock_fc

        mock_ac = MagicMock()
        mock_ac.find_one = AsyncMock(return_value=None)
        mock_ac.insert_one = AsyncMock()
        mock_alerts_col.return_value = mock_ac

        mock_llm.return_value = json.dumps({
            "is_alert": True,
            "alert_type": "loitering",
            "severity": "high",
            "title": "Person loitering at main gate",
            "description": "Loitering detected.",
            "evidence_frame_ids": ["f0", "f1", "f2"],
            "reasoning": "Repeated presence.",
        })

        from app.agents.alert_agent import run_alert_check
        await run_alert_check("vid_001", ["f0", "f1", "f2"])

        mock_ac.insert_one.assert_called_once()
        inserted = mock_ac.insert_one.call_args[0][0]
        assert inserted["alert_type"] == "loitering"
        assert inserted["severity"] == "high"
        assert inserted["zone_name"] == "main_gate"

    @pytest.mark.asyncio
    @patch("app.agents.alert_agent.alerts_col")
    @patch("app.agents.alert_agent.frames_col")
    @patch("app.agents.alert_agent._call_alert_llm")
    async def test_no_alert_not_inserted(
        self, mock_llm, mock_frames_col, mock_alerts_col
    ):
        frames = [make_frame("f0", "garage")]

        mock_fc = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=frames)
        mock_fc.find.return_value = mock_cursor
        mock_frames_col.return_value = mock_fc

        mock_ac = MagicMock()
        mock_ac.insert_one = AsyncMock()
        mock_alerts_col.return_value = mock_ac

        mock_llm.return_value = json.dumps({
            "is_alert": False,
            "alert_type": "unknown",
            "severity": "low",
            "title": "",
            "description": "",
            "evidence_frame_ids": [],
            "reasoning": "Nothing suspicious.",
        })

        from app.agents.alert_agent import run_alert_check
        await run_alert_check("vid_001", ["f0"])

        mock_ac.insert_one.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.agents.alert_agent.alerts_col")
    @patch("app.agents.alert_agent.frames_col")
    @patch("app.agents.alert_agent._call_alert_llm")
    async def test_duplicate_suppression(
        self, mock_llm, mock_frames_col, mock_alerts_col
    ):
        frames = [make_frame("f0", "main_gate", ["risk"], 1)]

        mock_fc = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=frames)
        mock_fc.find.return_value = mock_cursor
        mock_frames_col.return_value = mock_fc

        mock_ac = MagicMock()
        mock_ac.find_one = AsyncMock(return_value={"_id": "existing_alert"})
        mock_ac.insert_one = AsyncMock()
        mock_alerts_col.return_value = mock_ac

        mock_llm.return_value = json.dumps({
            "is_alert": True,
            "alert_type": "loitering",
            "severity": "high",
            "title": "Duplicate",
            "description": "",
            "evidence_frame_ids": ["f0"],
            "reasoning": "Dup.",
        })

        from app.agents.alert_agent import run_alert_check
        await run_alert_check("vid_001", ["f0"])

        mock_ac.insert_one.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.agents.alert_agent.frames_col")
    async def test_empty_frames_returns_early(self, mock_frames_col):
        mock_fc = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_fc.find.return_value = mock_cursor
        mock_frames_col.return_value = mock_fc

        from app.agents.alert_agent import run_alert_check
        await run_alert_check("vid_001", [])


class TestSummaryAgent:

    @pytest.mark.asyncio
    @patch("app.agents.summary_agent.sessions_col")
    @patch("app.agents.summary_agent.alerts_col")
    @patch("app.agents.summary_agent.frames_col")
    @patch("app.agents.summary_agent._call_summary_llm")
    async def test_summary_generated_and_stored(
        self, mock_llm, mock_fc, mock_ac, mock_sc
    ):
        frames = [make_frame(f"f{i}", "garage") for i in range(5)]

        fc = MagicMock()
        fc_cursor = MagicMock()
        fc_cursor.to_list = AsyncMock(return_value=frames)
        fc.find.return_value = fc_cursor
        mock_fc.return_value = fc

        ac = MagicMock()
        ac_cursor = MagicMock()
        ac_cursor.to_list = AsyncMock(return_value=[])
        ac.find.return_value = ac_cursor
        mock_ac.return_value = ac

        sc = MagicMock()
        sc.update_one = AsyncMock()
        mock_sc.return_value = sc

        mock_llm.return_value = (
            "The patrol session observed 5 frames in the garage zone.\n"
            "No suspicious activity was detected.\n"
            "CONCLUSION: Routine patrol with no incidents."
        )

        from app.agents.summary_agent import generate_session_summary
        result = await generate_session_summary("vid_001")

        assert result["frame_count"] == 5
        assert result["alert_count"] == 0
        assert "garage" in result["zones_visited"]
        assert "CONCLUSION" in result["summary_text"]
        sc.update_one.assert_called_once()


class TestQAAgent:

    @pytest.mark.asyncio
    @patch("app.agents.qa_agent.frames_col")
    @patch("app.agents.qa_agent._plan_query")
    @patch("app.agents.qa_agent._synthesise_answer")
    async def test_qa_returns_answer_with_frames(
        self, mock_synth, mock_plan, mock_fc
    ):
        frames = [make_frame("f0", "main_gate")]

        fc = MagicMock()
        find_result = MagicMock()
        cursor = MagicMock()
        cursor.to_list = AsyncMock(return_value=frames)
        find_result.limit.return_value = cursor
        fc.find.return_value = find_result
        mock_fc.return_value = fc

        mock_plan.return_value = {
            "zone_name": "main_gate",
            "object_keyword": None,
            "activity_keyword": None,
            "entity_label_keyword": None,
            "entity_type": None,
            "after_hour": None,
            "before_hour": None,
            "min_people": None,
            "min_vehicles": None,
            "limit": 10,
        }
        mock_synth.return_value = "One person was seen near the main gate at 00:01."

        from app.agents.qa_agent import answer_question
        result = await answer_question("vid_001", "Who was near the main gate?")

        assert result["frame_count"] == 1
        assert "main gate" in result["answer"]
        assert len(result["supporting_frames"]) == 1

    @pytest.mark.asyncio
    @patch("app.agents.qa_agent._plan_query")
    async def test_qa_returns_error_on_plan_failure(self, mock_plan):
        mock_plan.side_effect = Exception("LLM unavailable")

        from app.agents.qa_agent import answer_question
        result = await answer_question("vid_001", "What happened?")

        assert "Failed" in result["answer"]
        assert result["frame_count"] == 0