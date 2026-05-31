"""
scripts/generate_demo_data.py
Generates:
  1. A synthetic MP4 surveillance video (colored frames with text overlays)
  2. A matching telemetry JSON file

Run from the project root:
    python scripts/generate_demo_data.py
"""
import json
import math
import random
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

OUTPUT_VIDEO = Path("input/demo_patrol.mp4")
OUTPUT_TELEMETRY = Path("input/demo_telemetry.json")
DURATION_SEC = 60          # 1 minute demo
FPS = 10
FRAME_W, FRAME_H = 640, 360

ZONES = [
    {"name": "main_gate",      "lat": 18.5204, "lon": 73.8567, "alt": 12.0},
    {"name": "garage",         "lat": 18.5206, "lon": 73.8570, "alt": 10.0},
    {"name": "north_fence",    "lat": 18.5210, "lon": 73.8565, "alt": 15.0},
    {"name": "parking_lot",    "lat": 18.5203, "lon": 73.8575, "alt": 11.0},
    {"name": "restricted_area","lat": 18.5215, "lon": 73.8560, "alt": 13.0},
]

SCENARIOS = [
    {"label": "Clear area",              "color": (20, 35, 50),   "objects": []},
    {"label": "Person near gate",        "color": (25, 40, 30),   "objects": ["PERSON"]},
    {"label": "Blue truck parked",       "color": (20, 30, 45),   "objects": ["TRUCK"]},
    {"label": "Two people loitering",    "color": (30, 45, 30),   "objects": ["PERSON", "PERSON"]},
    {"label": "Vehicle + person",        "color": (25, 40, 50),   "objects": ["TRUCK", "PERSON"]},
    {"label": "Night activity detected", "color": (10, 15, 25),   "objects": ["PERSON"]},
    {"label": "Multiple vehicles",       "color": (15, 25, 40),   "objects": ["TRUCK", "CAR"]},
]


def draw_frame(frame_idx: int, scenario: dict, zone: dict, ts: datetime) -> np.ndarray:
    """Draw a synthetic surveillance-style frame."""
    img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    r, g, b = scenario["color"]

    # Background gradient
    for y in range(FRAME_H):
        ratio = y / FRAME_H
        img[y, :] = [
            int(b + ratio * 10),
            int(g + ratio * 8),
            int(r + ratio * 5),
        ]

    # Ground line
    cv2.line(img, (0, FRAME_H - 80), (FRAME_W, FRAME_H - 80), (40, 60, 50), 1)

    # Draw objects
    obj_x = 100
    for obj in scenario["objects"]:
        if obj == "PERSON":
            # Simple stick-figure silhouette
            cx, cy = obj_x, FRAME_H - 120
            cv2.circle(img, (cx, cy), 15, (180, 190, 200), -1)
            cv2.line(img, (cx, cy + 15), (cx, cy + 55), (180, 190, 200), 5)
            cv2.line(img, (cx - 18, cy + 30), (cx + 18, cy + 30), (180, 190, 200), 4)
            cv2.line(img, (cx, cy + 55), (cx - 12, cy + 90), (180, 190, 200), 4)
            cv2.line(img, (cx, cy + 55), (cx + 12, cy + 90), (180, 190, 200), 4)
            obj_x += 60
        elif obj in ("TRUCK", "CAR"):
            # Vehicle shape
            vw = 110 if obj == "TRUCK" else 80
            vh = 45 if obj == "TRUCK" else 38
            vx, vy = obj_x, FRAME_H - 130
            color = (80, 100, 160) if obj == "TRUCK" else (60, 160, 80)
            cv2.rectangle(img, (vx, vy), (vx + vw, vy + vh), color, -1)
            cv2.rectangle(img, (vx + 10, vy - 25), (vx + vw - 10, vy), (color[0] + 20, color[1] + 20, color[2] + 20), -1)
            # Wheels
            cv2.circle(img, (vx + 18, vy + vh), 14, (30, 30, 30), -1)
            cv2.circle(img, (vx + vw - 18, vy + vh), 14, (30, 30, 30), -1)
            obj_x += vw + 20

    # Scan-line overlay (CRT effect)
    for y in range(0, FRAME_H, 4):
        img[y, :] = (img[y, :] * 0.85).astype(np.uint8)

    # Noise
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # HUD overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"CAM-{zone['name'].upper()}", (10, 25),
                font, 0.5, (0, 220, 160), 1, cv2.LINE_AA)
    cv2.putText(img, ts.strftime("%H:%M:%S UTC"), (10, 45),
                font, 0.4, (150, 180, 160), 1, cv2.LINE_AA)
    cv2.putText(img, f"LAT:{zone['lat']:.4f} LON:{zone['lon']:.4f}",
                (10, FRAME_H - 10), font, 0.35, (100, 130, 110), 1, cv2.LINE_AA)
    cv2.putText(img, scenario["label"], (FRAME_W - 200, 25),
                font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    # Recording indicator
    cv2.circle(img, (FRAME_W - 15, 15), 5, (0, 0, 220), -1)

    # Corner brackets (surveillance style)
    blen = 20
    thick = 1
    col = (0, 200, 120)
    for cx, cy in [(0, 0), (FRAME_W, 0), (0, FRAME_H), (FRAME_W, FRAME_H)]:
        sx = 1 if cx == 0 else -1
        sy = 1 if cy == 0 else -1
        px, py = cx + sx * 5, cy + sy * 5
        cv2.line(img, (px, py), (px + sx * blen, py), col, thick)
        cv2.line(img, (px, py), (px, py + sy * blen), col, thick)

    return img


def generate_video_and_telemetry(
    video_path: Path,
    telemetry_path: Path,
    duration_sec: int = DURATION_SEC,
):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, FPS, (FRAME_W, FRAME_H))

    total_frames = duration_sec * FPS
    base_ts = datetime(2026, 5, 30, 0, 0, 0, tzinfo=timezone.utc)  # midnight patrol

    telemetry_records = []
    print(f"Generating {total_frames} frames ({duration_sec}s @ {FPS}fps) …")

    for i in range(total_frames):
        ts = base_ts + timedelta(seconds=i / FPS)
        zone_idx = int((i / total_frames) * len(ZONES))
        zone = ZONES[min(zone_idx, len(ZONES) - 1)]

        # Slightly vary coordinates per frame for realism
        jitter = 0.0001
        zone_var = {
            **zone,
            "lat": zone["lat"] + random.uniform(-jitter, jitter),
            "lon": zone["lon"] + random.uniform(-jitter, jitter),
        }

        # Pick scenario (change every ~3 seconds, with some randomness)
        scenario_idx = (i // (FPS * 3) + random.randint(0, 1)) % len(SCENARIOS)
        scenario = SCENARIOS[scenario_idx]

        frame = draw_frame(i, scenario, zone_var, ts)
        out.write(frame)

        # Telemetry at ~1Hz (every FPS frames)
        if i % FPS == 0:
            telemetry_records.append({
                "drone_id": "drone_alpha",
                "video_id": "vid_001",
                "timestamp": ts.isoformat(),
                "latitude": zone_var["lat"],
                "longitude": zone_var["lon"],
                "altitude": zone["alt"] + random.uniform(-0.5, 0.5),
                "zone_name": zone["name"],
                "speed": round(random.uniform(2.0, 8.0), 2),
                "heading": round(random.uniform(0, 360), 1),
            })

        if (i + 1) % (FPS * 5) == 0:
            print(f"  … {i + 1}/{total_frames} frames written")

    out.release()
    print(f"✓ Video saved: {video_path}")

    with open(telemetry_path, "w") as f:
        json.dump(telemetry_records, f, indent=2)
    print(f"✓ Telemetry saved: {telemetry_path} ({len(telemetry_records)} records)")


if __name__ == "__main__":
    print("=" * 50)
    print("  Drone Security — Demo Data Generator")
    print("=" * 50)
    generate_video_and_telemetry(OUTPUT_VIDEO, OUTPUT_TELEMETRY)
    print("\nDemo data ready. Now run:")
    print("  uvicorn app.main:app --reload --port 8000")
