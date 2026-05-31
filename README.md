# Drone Security Analyst Agent

README — Setup, Architecture & AI Integration Guide

**AI Engineer Assignment · FlytBase · May 2026**

## Overview

The Drone Security Analyst Agent is an AI prototype for autonomous drone security monitoring. It processes a simulated video patrol alongside aligned drone telemetry, generates structured per-frame intelligence using GPT-4o Vision, and exposes three LLM-powered agents — Alert, Summary, and QA — through a FastAPI backend and a dark-themed single-page web interface.

### Capabilities

| Capability | What it does |
|---|---|
| Frame-level VLM analysis | GPT-4o Vision describes every sampled frame as structured JSON: objects, people count, vehicle attributes, activities, and risk signals. |
| Telemetry enrichment | Each frame is merged with the nearest GPS, altitude, and zone telemetry record, grounding visual analysis in physical space. |
| LLM Alert Agent | Evaluates sliding windows of frames and classifies security events such as loitering, repeated vehicle sightings, after-hours activity, and restricted-zone intrusion. |
| Session Summary Agent | Generates a 3–5 paragraph natural language patrol report from the full frame timeline. |
| Follow-up QA Agent | Answers operator questions over indexed frames using a two-step query planner and synthesiser pattern. |
| Queryable frame index | MongoDB stores every frame with structured indexes enabling retrieval by zone, time, object keyword, or entity label. |

## Project Structure

```text
drone-security-agent/
├── app/
│   ├── main.py                      # FastAPI app factory + lifespan events
│   ├── config.py                    # Pydantic settings (reads from .env)
│   ├── db/mongo.py                  # Motor async client + index bootstrap
│   ├── models/schemas.py            # All Pydantic data models
│   ├── pipeline/
│   │   ├── frame_sampler.py         # OpenCV video → sampled JPEG frames
│   │   ├── telemetry_merger.py      # Nearest-timestamp JSON alignment
│   │   ├── vlm_captioner.py         # GPT-4o Vision → structured LLMContext
│   │   ├── entity_normalizer.py     # Heuristic → canonical NormalizedEntity[]
│   │   └── ingestion.py             # Orchestrator (FastAPI background task)
│   ├── agents/
│   │   ├── alert_agent.py           # Sliding-window LLM security classifier
│   │   ├── summary_agent.py         # Session timeline → prose summary
│   │   └── qa_agent.py              # Two-step query planner + synthesiser
│   ├── api/routes.py                # REST endpoints
│   └── utils/logger.py              # Rich-coloured structured logging
├── frontend/
│   ├── templates/index.html         # Single-page security dashboard
│   └── static/{css,js}              # Industrial dark-theme UI
├── scripts/generate_demo_data.py    # Synthetic video + telemetry generator
├── tests/                           # Full pytest suite
├── .env.example                     # Environment variables with descriptions
└── requirements.txt
```

## Quick Start

Complete these steps in order:

1. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
2. Copy the environment file and fill in the required values.
   ```bash
   cp .env.example .env
   ```
3. Generate synthetic demo data.
   ```bash
   python scripts/generate_demo_data.py
   ```
4. Start the FastAPI server.
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
5. Open `http://localhost:8000`.
6. Click **NEW** and enter:
   - `vid_001`
   - `drone_alpha`
   - `input/demo_patrol.mp4`
   - `input/demo_telemetry.json`

## Detailed Setup

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Earlier versions lack some typing features used in the codebase. |
| pip | Ensure it is up to date with `pip install --upgrade pip`. |
| OpenAI API key | Must have GPT-4o access. Free trial tier does not include GPT-4o vision. |
| MongoDB Atlas account | A free M0 cluster is sufficient for the prototype. |
| Git (optional) | Needed only if cloning from a repository. |

### MongoDB Atlas Setup

1. Create a free M0 cluster in MongoDB Atlas.
2. Under **Database Access**, create a user and password.
3. Under **Network Access**, whitelist your IP or use `0.0.0.0/0` for development.
4. Copy the Python driver connection string.
5. Add it to `.env` as `MONGODB_URI`.

The database and collections are created automatically on first run.

### Environment Configuration

Copy the example file:

```bash
cp .env.example .env
```

Minimum required values:

```env
OPENAI_API_KEY=sk-proj-...your-key...
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
```

Optional tuning:

```env
FRAME_SAMPLE_INTERVAL_SEC=2
ALERT_WINDOW_SIZE=5
TELEMETRY_MATCH_TOLERANCE_SEC=5
LOG_LEVEL=INFO
```

### Generate Demo Data

The demo generator creates:
- A 60-second synthetic surveillance video with HUD overlays.
- Seven patrol scenarios, including person near gate, blue truck, loitering, night activity, crowd formation, and multiple vehicles.
- Aligned telemetry at 1 Hz across five zones.

Run:

```bash
python scripts/generate_demo_data.py
```

Output files:
- `input/demo_patrol.mp4`
- `input/demo_telemetry.json`

### Start the Server

```bash
uvicorn app.main:app --reload --port 8000
```

API docs:
- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

## Running the Demo

### Via the Web UI

1. Go to `http://localhost:8000`.
2. Click **NEW** in the Sessions panel.
3. Fill in the session form.
4. Click **START INGESTION**.
5. Watch frames appear in the **FRAMES** tab.
6. Once the session is processed, use:
   - **ALERTS** for generated security events.
   - **SUMMARY** to generate a patrol report.
   - **ASK AI** for natural language questions.
   - **SEARCH** to filter by zone, object keyword, or activity.

### Via curl / API

Start ingestion:

```bash
curl -X POST http://localhost:8000/api/ingest/session \
  -H "Content-Type: application/json" \
  -d '{"video_id":"vid_001","drone_id":"drone_alpha","video_path":"input/demo_patrol.mp4","telemetry_path":"input/demo_telemetry.json"}'
```

Poll session status:

```bash
curl http://localhost:8000/api/sessions/vid_001
```

Ask a question:

```bash
curl -X POST http://localhost:8000/api/sessions/vid_001/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What vehicles were seen near the gate?"}'
```

Search frames by object:

```bash
curl 'http://localhost:8000/api/frames/search?video_id=vid_001&object_kw=truck'
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/ingest/session` | Start video and telemetry ingestion in the background. |
| GET | `/api/sessions` | List all sessions with status, frame count, and drone ID. |
| GET | `/api/sessions/{video_id}` | Return full session metadata including status, zones visited, and summary text. |
| GET | `/api/sessions/{video_id}/frames` | Return paginated indexed frames ordered by timestamp. |
| GET | `/api/sessions/{video_id}/alerts` | Return all alerts newest-first. |
| PATCH | `/api/sessions/{video_id}/alerts/{alert_id}/status` | Update alert lifecycle status. |
| POST | `/api/sessions/{video_id}/summary` | Trigger the Summary Agent. |
| POST | `/api/sessions/{video_id}/ask` | Run natural language QA over indexed frames. |
| GET | `/api/frames/search` | Search frames by video, zone, object keyword, activity keyword, and time filters. |

## Architecture and Design Decisions

### Five-Layer Single-Node Architecture

The system is split into five layers:
- Input simulation layer
- Ingestion pipeline
- Storage layer
- Agent layer
- API and frontend layer

This prototype stays single-node to reduce infrastructure complexity while keeping the system easy to debug and demonstrate.

### Ingestion as a FastAPI Background Task

Ingestion runs as a FastAPI background task instead of a blocking request. This allows the API to return immediately while frame sampling, telemetry merging, and VLM processing continue asynchronously.

Production upgrade path:
- Replace `BackgroundTasks` with Celery + Redis.
- Add retries and horizontal scaling with worker processes.

### Fixed-Interval Frame Sampling

Frames are sampled every `FRAME_SAMPLE_INTERVAL_SEC` seconds using OpenCV. This is preferred over scene-cut detection because surveillance footage changes gradually, and fixed sampling captures slow-developing events more reliably.

### Nearest-Timestamp Telemetry Merge

Each sampled frame is matched to the closest telemetry point within a tolerance window. If the nearest match is too far away, the system falls back to `zone=unknown` instead of attaching stale telemetry.

### Structured JSON from GPT-4o

The VLM captioner returns structured JSON rather than free-text captions. This makes frame content queryable by MongoDB and usable by the downstream alert, summary, and QA agents.

### Alert Agent Design

The LLM classifies potential alerts, while the application:
- validates the response,
- deduplicates alerts,
- persists records to MongoDB.

This keeps the LLM responsible for reasoning and the application responsible for control flow and data integrity.

### QA Agent Design

The QA flow uses:
1. Query planner
2. MongoDB query execution
3. Answer synthesiser

This is used instead of vector RAG because most surveillance questions are structured and map well to exact filters such as zone, time window, object type, and activity.

### MongoDB Schema Choice

MongoDB stores flexible JSON documents that fit variable LLM output better than a rigid relational schema. Targeted indexes on fields like `video_id`, `video_timestamp`, `telemetry.zone_name`, `llm_context.objects_detected`, and `normalized_entities.label` support efficient retrieval.

### Frontend Choice

The frontend is a vanilla JavaScript SPA with no framework or build step. This keeps the prototype lightweight and focused on AI engineering rather than frontend tooling.

## AI Tools Integration and Impact

### Claude as Development Assistant

Claude was used for:
- architecture design,
- Pydantic model generation,
- prompt engineering,
- module scaffolding,
- test suite generation,
- documentation drafting.

Human review was then used to customise models, prompts, tests, and implementation details.

### GPT-4o Runtime Components

| Component | GPT-4o Call Type | Output |
|---|---|---|
| VLM Frame Captioner | Chat + Vision | Structured `LLMContext` JSON per frame |
| Alert Agent | Chat | Alert classification JSON |
| Summary Agent | Chat | 3–5 paragraph session report |
| QA Query Planner | Chat | Structured filter parameters JSON |
| QA Synthesiser | Chat | 2–4 sentence grounded answer |

### Reliability Engineering

All GPT-4o calls are wrapped with Tenacity retries using exponential backoff. JSON parsing strips markdown fences and uses fallback handling so the pipeline does not crash on malformed LLM output.

### Cost Management

The VLM captioner uses low-detail image processing to reduce token usage significantly compared with high-resolution processing. Token limits are also constrained per call type.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenAI API key with GPT-4o vision access. |
| `OPENAI_MODEL` | `gpt-4o` | Model name. |
| `MONGODB_URI` | required | MongoDB Atlas connection string. |
| `MONGODB_DB_NAME` | `drone_security` | Database name. |
| `FRAME_SAMPLE_INTERVAL_SEC` | `2` | Sampling interval in seconds. |
| `ALERT_WINDOW_SIZE` | `5` | Frames per sliding alert window. |
| `TELEMETRY_MATCH_TOLERANCE_SEC` | `5` | Max telemetry merge gap in seconds. |
| `STORAGE_BASE_PATH` | `storage` | Root directory for frame files. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |
| `APP_HOST` | `0.0.0.0` | Uvicorn bind host. |
| `APP_PORT` | `8000` | Uvicorn bind port. |

## Testing

### Running Tests

All tests run with external services mocked.

```bash
pytest tests/ -v
pytest tests/ --cov=app --cov-report=term-missing
pytest tests/test_pipeline.py -v
pytest tests/test_agents.py -v
pytest tests/test_api.py -v
```

### Coverage Summary

| Test File | Cases | Coverage |
|---|---:|---|
| `tests/test_pipeline.py` | 10 | Frame sampling, telemetry merge, entity normalization, VLM JSON parser, alert JSON parser, and Pydantic serialization. |
| `tests/test_api.py` | 8 | Ingestion endpoint, duplicate session handling, sessions lookup, frame listing, alerts, alert status update, and frame search. |
| `tests/test_agents.py` | 9 | Alert persistence, duplicate suppression, empty-frame path, summary generation, QA answer generation, and QA planning failure. |

## Troubleshooting

| Problem | Solution |
|---|---|
| MongoDB connection fails on startup | Verify `MONGODB_URI` and Atlas network access settings. |
| OpenAI rate limit during ingestion | Increase `FRAME_SAMPLE_INTERVAL_SEC` or add pacing between frame calls. |
| `input/demo_patrol.mp4` not found | Run `python scripts/generate_demo_data.py` from the project root. |
| Frames tab is empty | Wait for background ingestion and check logs for progress or failures. |
| Alert Agent not raising alerts | Lower `ALERT_WINDOW_SIZE` or ensure the configured model supports the required capability. |
| Broken frame images | Verify `storage/` exists and `frame_path` matches the actual file location. |
| Session stuck in processing | Check terminal logs for the background-task traceback and manually reset the DB status if needed. |
| Tests fail with ImportError | Reinstall dependencies and verify settings are loaded lazily. |

## Future Scope

| Feature | Layer | Implementation Path |
|---|---|---|
| Live RTSP ingestion | Input | Replace file-based capture with `rtsp://` streaming and reconnect logic. |
| Blob storage for frames | Storage | Move frame files to S3 or Azure Blob and serve via signed URLs. |
| Distributed frame workers | Ingestion | Add Celery + Redis or Kafka-backed worker queues. |
| Hybrid RAG retrieval | QA Agent | Add Atlas Vector Search and combine with structured filters. |
| Person re-identification | Ingestion / Agents | Integrate a re-ID model for consistent person tracking. |
| Motion-detection pre-filter | Ingestion | Skip static frames before VLM calls to reduce cost. |
| Operator dashboard | Frontend | Add auth, alert workflow, map view, and alert pins. |
| Multi-drone support | All layers | Extend the existing session model to a multi-drone overview. |