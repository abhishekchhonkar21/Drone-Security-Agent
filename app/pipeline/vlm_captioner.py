"""
app/pipeline/vlm_captioner.py
Sends a frame image + telemetry context to Azure OpenAI Vision and returns
a structured LLMContext describing the scene.
Includes retry logic and graceful fallback on API errors.
"""
import base64
import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.models.schemas import LLMContext, TelemetryData
from app.utils.logger import get_logger


log = get_logger(__name__)


_client = AzureChatOpenAI(
    azure_deployment=settings.azure_openai_deployment,
    api_version=settings.azure_openai_api_version,
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_key,

)


_SYSTEM_PROMPT = """\
You are a security analyst AI for a drone surveillance system.
You will receive a frame image from a drone camera along with metadata about the drone's location.
Your job is to produce a STRUCTURED JSON security analysis of the scene.

Return ONLY valid JSON — no markdown fences, no extra text. Use this exact schema:
{
  "scene_summary": "<one-sentence description of what is happening>",
  "objects_detected": ["<object1>", "<object2>"],
  "people_count": <integer>,
  "vehicles_count": <integer>,
  "vehicle_attributes": ["<color make/model or description>"],
  "activity_detected": ["<activity1>", "<activity2>"],
  "risk_signals": ["<signal1>"],
  "confidence_notes": "<any notes about image quality or uncertainty>"
}

Be specific. Use security-relevant vocabulary.
Examples of good activity_detected values: vehicle_entry, person_loitering, crowd_forming, after_hours_access, vehicle_parked_long.
Examples of good risk_signals: person_near_restricted_zone, unidentified_vehicle, after_hours_activity, tailgating_risk.
If nothing notable, use empty arrays and state that clearly in scene_summary.
"""


_USER_PROMPT_TEMPLATE = """\
Frame captured at: {timestamp}
Drone zone: {zone_name}
GPS: lat={latitude}, lon={longitude}, alt={altitude}m

Analyse this security camera frame and respond in the required JSON format.
"""


def _encode_image(frame_path: str) -> str:
    """Base64-encode a JPEG frame for the vision API."""
    return base64.b64encode(Path(frame_path).read_bytes()).decode("utf-8")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_openai(image_b64: str, user_prompt: str) -> str:
    """Make the Azure OpenAI vision call with retry."""
    response = _client.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "low",
                    },
                },
            ]
        ),
    ])

    content = response.content
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts).strip()

    return str(content).strip()


def _parse_llm_json(raw: str) -> LLMContext:
    """
    Parse raw LLM output into LLMContext.
    Strips markdown code fences if the model includes them despite instructions.
    Falls back to a minimal valid context on parse failure.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(cleaned)
        return LLMContext(**data)
    except Exception as exc:
        log.warning(f"[Captioner] Failed to parse LLM JSON: {exc} | raw={raw[:200]}")
        return LLMContext(
            scene_summary="[Parse error — raw model output stored in confidence_notes]",
            confidence_notes=raw[:500],
        )


def generate_frame_description(
    frame_path: str,
    telemetry: TelemetryData,
    video_timestamp,
) -> LLMContext:
    """
    Generate a structured LLMContext for a single frame.

    Args:
        frame_path:      Path to the saved JPEG on disk.
        telemetry:       Merged telemetry for this frame.
        video_timestamp: UTC datetime of the frame within the video.

    Returns:
        LLMContext with scene analysis.
    """
    log.debug(f"[Captioner] Describing frame: {Path(frame_path).name}")

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        timestamp=video_timestamp.isoformat(),
        zone_name=telemetry.zone_name,
        latitude=telemetry.latitude,
        longitude=telemetry.longitude,
        altitude=telemetry.altitude,
    )

    try:
        image_b64 = _encode_image(frame_path)
        raw_output = _call_openai(image_b64, user_prompt)
        log.debug(f"[Captioner] Raw LLM output: {raw_output[:150]} …")
        context = _parse_llm_json(raw_output)
        log.debug(f"[Captioner] Scene summary: {context.scene_summary}")
        return context
    except Exception as exc:
        log.error(f"[Captioner] Captioning error for {frame_path}: {exc}", exc_info=True)
        return LLMContext(
            scene_summary="[Captioning error — frame skipped]",
            confidence_notes=str(exc)[:200],
        )