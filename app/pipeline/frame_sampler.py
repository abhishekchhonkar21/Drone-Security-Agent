"""
app/pipeline/frame_sampler.py
Samples frames from a video file at a fixed time interval using OpenCV.
Saves frames to local disk under storage/videos/<video_id>/frames/.
"""
import cv2
import os
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from app.config import settings
from app.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SampledFrame:
    """Lightweight struct for a single sampled frame before DB storage."""
    frame_id: str          # unique UUID assigned later
    frame_no: int          # sequential sampler index (1-based)
    video_timestamp: datetime
    frame_path: str        # absolute local path to saved JPEG
    raw_frame_no: int      # original OpenCV frame number


def sample_frames(
    video_path: str,
    video_id: str,
    interval_sec: int | None = None,
) -> list[SampledFrame]:
    """
    Read the video at `video_path` and save one frame every `interval_sec`
    seconds to disk. Returns a list of SampledFrame metadata objects.

    Args:
        video_path: Path to the source .mp4 (or directory of frames).
        video_id:   Session identifier used to build the output path.
        interval_sec: Seconds between samples; defaults to settings value.

    Returns:
        List of SampledFrame objects, one per saved frame.
    """
    interval_sec = interval_sec or settings.frame_sample_interval_sec

    # --- Resolve output directory ---
    output_dir = Path(settings.storage_base_path) / "videos" / video_id / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[Sampler] Output dir: {output_dir}")

    # --- Handle frame-directory input ---
    if os.path.isdir(video_path):
        return _sample_from_directory(video_path, video_id, output_dir)

    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"[Sampler] Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    sample_every_n = max(1, int(fps * interval_sec))

    log.info(
        f"[Sampler] Video: {video_path} | FPS={fps:.1f} | "
        f"Total frames={total_frames} | Duration={duration_sec:.1f}s | "
        f"Sampling every {sample_every_n} frames ({interval_sec}s)"
    )

    sampled: list[SampledFrame] = []
    raw_frame_no = 0
    sampler_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if raw_frame_no % sample_every_n == 0:
            sampler_index += 1
            timestamp_sec = raw_frame_no / fps
            video_ts = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)

            filename = f"frame_{sampler_index:04d}.jpg"
            frame_path = str(output_dir / filename)
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            sampled.append(SampledFrame(
                frame_id="",              # assigned by ingestion orchestrator
                frame_no=sampler_index,
                video_timestamp=video_ts,
                frame_path=frame_path,
                raw_frame_no=raw_frame_no,
            ))

            if sampler_index % 20 == 0:
                log.info(f"[Sampler] … sampled {sampler_index} frames so far")

        raw_frame_no += 1

    cap.release()
    log.info(f"[Sampler] Done — {len(sampled)} frames saved to {output_dir}")
    return sampled


def _sample_from_directory(
    frames_dir: str,
    video_id: str,
    output_dir: Path,
) -> list[SampledFrame]:
    """
    Treat a directory of pre-existing JPEG/PNG files as a frame sequence.
    Each file is treated as 1 second of footage.
    """
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(
        f for f in Path(frames_dir).iterdir()
        if f.suffix.lower() in exts
    )
    log.info(f"[Sampler] Frame directory mode — {len(files)} images found")

    sampled: list[SampledFrame] = []
    for idx, src_path in enumerate(files, start=1):
        dst_path = output_dir / f"frame_{idx:04d}.jpg"
        import shutil
        shutil.copy2(src_path, dst_path)
        sampled.append(SampledFrame(
            frame_id="",
            frame_no=idx,
            video_timestamp=datetime.fromtimestamp(idx - 1, tz=timezone.utc),
            frame_path=str(dst_path),
            raw_frame_no=idx - 1,
        ))
    return sampled
