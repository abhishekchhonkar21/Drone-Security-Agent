"""
app/db/mongo.py
Async MongoDB client using Motor. Provides collection accessors and index setup.
Call `connect_db()` on app startup and `close_db()` on shutdown.
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING
from app.config import settings
from app.utils.logger import get_logger

log = get_logger(__name__)

# Module-level client + db — initialised in connect_db()
_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_db() -> None:
    """Create the Motor client and ensure indexes exist."""
    global _client, _db
    log.info(f"[DB] Connecting to MongoDB (db={settings.mongodb_db_name}) …")
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    _db = _client[settings.mongodb_db_name]
    await _ensure_indexes()
    log.info("[DB] MongoDB connection established ✓")


async def close_db() -> None:
    """Close the Motor client gracefully."""
    global _client
    if _client:
        _client.close()
        log.info("[DB] MongoDB connection closed.")


def get_db() -> AsyncIOMotorDatabase:
    """Return the database instance. Must be called after connect_db()."""
    if _db is None:
        raise RuntimeError("Database not initialised. Call connect_db() first.")
    return _db


# ---------------------------------------------------------------------------
# Collection shortcuts
# ---------------------------------------------------------------------------

def frames_col():
    return get_db()["frames"]

def alerts_col():
    return get_db()["alerts"]

def sessions_col():
    return get_db()["video_sessions"]


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

async def _ensure_indexes() -> None:
    """Create all indexes if they do not exist. Idempotent."""
    log.info("[DB] Ensuring MongoDB indexes …")
    db = get_db()

    # frames collection
    await db["frames"].create_index([("video_id", ASCENDING)])
    await db["frames"].create_index([("video_timestamp", ASCENDING)])
    await db["frames"].create_index([("telemetry.zone_name", ASCENDING)])
    await db["frames"].create_index([("llm_context.objects_detected", ASCENDING)])
    await db["frames"].create_index([("normalized_entities.label", ASCENDING)])
    await db["frames"].create_index([
        ("video_id", ASCENDING), ("video_timestamp", ASCENDING)
    ])

    # alerts collection
    await db["alerts"].create_index([("video_id", ASCENDING)])
    await db["alerts"].create_index([("alert_type", ASCENDING)])
    await db["alerts"].create_index([("trigger_timestamp", DESCENDING)])
    await db["alerts"].create_index([("zone_name", ASCENDING)])
    await db["alerts"].create_index([("status", ASCENDING)])

    # video_sessions collection
    await db["video_sessions"].create_index([("status", ASCENDING)])
    await db["video_sessions"].create_index([("drone_id", ASCENDING)])

    log.info("[DB] Indexes ready ✓")
