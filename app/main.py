"""
app/main.py
FastAPI application factory. Registers lifespan events, mounts static files,
configures Jinja2 templates, and includes all routers.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.db.mongo import connect_db, close_db
from app.api.routes import router
from app.utils.logger import get_logger

log = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "frontend" / "templates"
STATIC_DIR = BASE_DIR / "frontend" / "static"


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("  Drone Security Analyst Agent — starting up")
    log.info(f"  Model     : {settings.azure_openai_deployment}")
    log.info(f"  DB        : {settings.mongodb_db_name}")
    log.info(f"  Frame int : {settings.frame_sample_interval_sec}s")
    log.info(f"  Alert win : {settings.alert_window_size} frames")
    log.info("=" * 60)

    # Initialise MongoDB connection
    await connect_db()

    # Ensure storage directories exist
    Path(settings.storage_base_path).mkdir(parents=True, exist_ok=True)
    log.info(f"Storage root: {settings.storage_base_path}")

    yield

    # Shutdown
    await close_db()
    log.info("Drone Security Analyst Agent — shut down cleanly")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    application = FastAPI(
        title="Drone Security Analyst Agent",
        description=(
            "AI-powered drone security monitoring system. "
            "Processes video + telemetry, generates frame descriptions, "
            "detects alerts, and answers operator questions."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Mount static files
    if STATIC_DIR.exists():
        application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Include API router
    application.include_router(router)

    # Mount frame image files for serving
    storage_path = Path(settings.storage_base_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    application.mount(
        "/storage",
        StaticFiles(directory=str(storage_path)),
        name="storage",
    )

    # Jinja2 templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # ---------------------------------------------------------------------------
    # Frontend routes
    # ---------------------------------------------------------------------------

    @application.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @application.get("/health")
    async def health():
        return {"status": "ok", "service": "drone-security-analyst-agent"}

    return application


app = create_app()
