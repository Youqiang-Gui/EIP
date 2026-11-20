"""Six independent semantic emotion conditions, plus none for silent chunks."""

from __future__ import annotations

from typing import Any, Mapping

EMOTION_LABEL_TO_TOKEN_ID = {
    "happy": 151614,
    "sad": 151615,
    "angry": 151616,
    "surprise": 151617,
    "normal": 151618,
    "urgent": 151619,
}
NO_EMOTION_ID = 0
EMOTION_LABELS = tuple(EMOTION_LABEL_TO_TOKEN_ID)
NUM_EMOTION_CONDITIONS = 1 + len(EMOTION_LABELS)
EMOTION_MAPPING_VERSION = "semantic_six_emotions_v1"
EMOTION_LABEL_TO_CONDITION_ID = {
    label: index for index, label in enumerate(EMOTION_LABELS, start=1)
}
EMOTION_TOKEN_ID_TO_CONDITION_ID = {
    token_id: EMOTION_LABEL_TO_CONDITION_ID[label]
    for label, token_id in EMOTION_LABEL_TO_TOKEN_ID.items()
}


def emotion_label(value: Any, *, source: str) -> str:
    """Validate one index emotion object and return its normalized label."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: target_emotion must be an object")
    label = value.get("label")
    normalized = label.lower() if isinstance(label, str) else None
    if normalized not in EMOTION_LABEL_TO_TOKEN_ID:
        raise ValueError(
            f"{source}: invalid target_emotion.label={label!r}; "
            f"expected one of {sorted(EMOTION_LABEL_TO_TOKEN_ID)}"
        )
    token_id = value.get("token_id")
    expected_token_id = EMOTION_LABEL_TO_TOKEN_ID[normalized]
    if token_id is not None and int(token_id) != expected_token_id:
        raise ValueError(f"{source}: label/token_id mismatch for {label!r}/{token_id!r}")
    return normalized


def emotion_condition_id(value: Any, *, source: str) -> int:
    """Encode the annotated emotion directly without merging categories."""

    return EMOTION_LABEL_TO_CONDITION_ID[
        emotion_label(value, source=source)
    ]


def emotion_name(condition_id: int) -> str:
    names = {
        NO_EMOTION_ID: "none",
        **{index: label for label, index in EMOTION_LABEL_TO_CONDITION_ID.items()},
    }
    try:
        return names[int(condition_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid emotion id: {condition_id!r}") from exc
