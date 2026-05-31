"""
app/pipeline/telemetry_merger.py
Loads a telemetry JSON file and provides nearest-timestamp matching
for each sampled frame. Configurable tolerance window.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from app.config import settings
from app.models.schemas import TelemetryData, TelemetryRecord
from app.utils.logger import get_logger

log = get_logger(__name__)

# Fallback telemetry used when no match is found within tolerance
_FALLBACK_TELEMETRY = TelemetryData(
    latitude=0.0,
    longitude=0.0,
    altitude=0.0,
    zone_name="unknown",
)


class TelemetryMerger:
    """
    Loads telemetry records from a JSON file and exposes
    `get_for_timestamp()` for O(n) nearest-match lookup.
    For a prototype this is fast enough; in production, use a sorted
    structure or timeseries DB for sub-millisecond lookup.
    """

    def __init__(self, telemetry_path: str) -> None:
        self._records: list[TelemetryRecord] = []
        self._tolerance_sec = settings.telemetry_match_tolerance_sec
        self._load(telemetry_path)

    def _load(self, path: str) -> None:
        log.info(f"[Telemetry] Loading telemetry from: {path}")
        raw = json.loads(Path(path).read_text())
        if isinstance(raw, dict) and "records" in raw:
            raw = raw["records"]
        self._records = [TelemetryRecord(**r) for r in raw]
        # Ensure UTC-aware datetimes for comparison
        for r in self._records:
            if r.timestamp.tzinfo is None:
                r.timestamp = r.timestamp.replace(tzinfo=timezone.utc)
        log.info(f"[Telemetry] Loaded {len(self._records)} telemetry records ✓")

    def get_for_timestamp(self, ts: datetime) -> TelemetryData:
        """
        Return the TelemetryData whose timestamp is closest to `ts`.
        Returns fallback telemetry if no record is within tolerance.
        """
        if not self._records:
            log.warning("[Telemetry] No records loaded — using fallback telemetry")
            return _FALLBACK_TELEMETRY

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        best: TelemetryRecord | None = None
        best_delta = float("inf")

        for rec in self._records:
            delta = abs((rec.timestamp - ts).total_seconds())
            if delta < best_delta:
                best_delta = delta
                best = rec

        if best_delta > self._tolerance_sec:
            log.warning(
                f"[Telemetry] No record within {self._tolerance_sec}s of "
                f"{ts.isoformat()} (closest gap={best_delta:.1f}s) — fallback used"
            )
            return _FALLBACK_TELEMETRY

        return TelemetryData(
            latitude=best.latitude,
            longitude=best.longitude,
            altitude=best.altitude,
            zone_name=best.zone_name,
            speed=best.speed,
            heading=best.heading,
        )
