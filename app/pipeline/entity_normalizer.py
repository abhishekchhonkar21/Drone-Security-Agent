"""
app/pipeline/entity_normalizer.py
Converts raw LLMContext objects_detected + vehicle_attributes into
canonical NormalizedEntity structs using heuristic rules + optional
LLM assist for complex cases.
"""
import re
from app.models.schemas import LLMContext, NormalizedEntity
from app.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Heuristic lookup tables
# ---------------------------------------------------------------------------

_VEHICLE_TYPES = {
    "truck": "truck",
    "pickup": "pickup_truck",
    "van": "van",
    "suv": "suv",
    "car": "car",
    "sedan": "sedan",
    "motorcycle": "motorcycle",
    "bike": "motorcycle",
    "bus": "bus",
    "lorry": "truck",
    "hatchback": "hatchback",
    "jeep": "suv",
    "forklift": "forklift",
}

_COLORS = {
    "red", "blue", "green", "black", "white", "silver", "grey", "gray",
    "yellow", "orange", "brown", "maroon", "navy", "dark", "light",
}

_PERSON_KEYWORDS = {"person", "people", "man", "woman", "individual", "guard", "pedestrian", "suspect"}

_MAKES = {
    "ford": "Ford", "toyota": "Toyota", "honda": "Honda", "bmw": "BMW",
    "mercedes": "Mercedes-Benz", "audi": "Audi", "chevrolet": "Chevrolet",
    "hyundai": "Hyundai", "kia": "Kia", "nissan": "Nissan", "vw": "Volkswagen",
    "volkswagen": "Volkswagen", "tesla": "Tesla",
}


def _is_vehicle(token: str) -> bool:
    return any(v in token.lower() for v in _VEHICLE_TYPES)


def _is_person(token: str) -> bool:
    return any(p in token.lower() for p in _PERSON_KEYWORDS)


def _extract_color(text: str) -> str | None:
    words = text.lower().split()
    for w in words:
        if w in _COLORS:
            return w
    return None


def _extract_vehicle_subtype(text: str) -> str:
    text_lower = text.lower()
    for keyword, subtype in _VEHICLE_TYPES.items():
        if keyword in text_lower:
            return subtype
    return "vehicle"


def _extract_make(text: str) -> str | None:
    text_lower = text.lower()
    for keyword, make in _MAKES.items():
        if keyword in text_lower:
            return make
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_entities(context: LLMContext) -> list[NormalizedEntity]:
    """
    Derive a list of NormalizedEntity objects from an LLMContext.
    Combines objects_detected (for persons) and vehicle_attributes (for vehicles).

    Returns:
        List of deduplicated NormalizedEntity objects.
    """
    entities: list[NormalizedEntity] = []
    seen_labels: set[str] = set()

    # --- Person detection ---
    person_count = context.people_count
    if person_count == 0:
        # Fall back to scanning objects_detected
        person_count = sum(1 for o in context.objects_detected if _is_person(o))

    for i in range(max(person_count, 0)):
        label = f"person_{i + 1}"
        if label not in seen_labels:
            entities.append(NormalizedEntity(entity_type="person", label=label))
            seen_labels.add(label)

    # --- Vehicle detection from vehicle_attributes ---
    for attr_str in context.vehicle_attributes:
        color = _extract_color(attr_str)
        subtype = _extract_vehicle_subtype(attr_str)
        make = _extract_make(attr_str)

        # Build a canonical label like "blue_pickup_truck" or "vehicle_1"
        parts = [p for p in [color, subtype] if p]
        label = "_".join(parts) if parts else "unknown_vehicle"
        # Deduplicate by appending an index if collision
        base_label = label
        counter = 1
        while label in seen_labels:
            label = f"{base_label}_{counter}"
            counter += 1
        seen_labels.add(label)

        entities.append(NormalizedEntity(
            entity_type="vehicle",
            label=label,
            subtype=subtype,
            color=color,
            make_model=make,
        ))

    # --- Vehicle detection fallback from objects_detected ---
    if not context.vehicle_attributes:
        for obj in context.objects_detected:
            if _is_vehicle(obj):
                color = _extract_color(obj)
                subtype = _extract_vehicle_subtype(obj)
                parts = [p for p in [color, subtype] if p]
                label = "_".join(parts) if parts else "unknown_vehicle"
                base_label = label
                counter = 1
                while label in seen_labels:
                    label = f"{base_label}_{counter}"
                    counter += 1
                seen_labels.add(label)
                entities.append(NormalizedEntity(
                    entity_type="vehicle",
                    label=label,
                    subtype=subtype,
                    color=color,
                ))

    # --- Other notable objects ---
    other_keywords = {"package", "bag", "weapon", "luggage", "box", "container", "drone"}
    for obj in context.objects_detected:
        obj_lower = obj.lower()
        if not _is_person(obj) and not _is_vehicle(obj):
            for kw in other_keywords:
                if kw in obj_lower and obj_lower not in seen_labels:
                    entities.append(NormalizedEntity(entity_type="object", label=obj_lower))
                    seen_labels.add(obj_lower)
                    break

    log.debug(f"[Normalizer] Extracted {len(entities)} entities from frame context")
    return entities
