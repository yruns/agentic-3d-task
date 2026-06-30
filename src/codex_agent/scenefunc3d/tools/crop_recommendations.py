"""Crop recommendations derived from visible-object evidence."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

from typing_extensions import NotRequired

_AFFORDANCE_CROP_QUERY_TRIGGERS: tuple[str, ...] = (
    "button",
    "buttons",
    "control",
    "controls",
    "dial",
    "dials",
    "handle",
    "handles",
    "knob",
    "knobs",
    "pull",
    "switch",
    "switches",
    "valve",
    "valves",
)
_RIGHT_LOWER_CROP_LEFT_FRACTION = 0.65
_RIGHT_LOWER_CROP_RIGHT_EXPAND_FRACTION = 0.10
_RIGHT_LOWER_CROP_TOP_FRACTION = 0.70
_RIGHT_LOWER_CROP_BOTTOM_EXPAND_FRACTION = 0.05


class MatchedObjectCropPayload(TypedDict):
    """Visible object metadata sufficient to propose a crop."""

    object_id: str
    label: str
    score: float
    source: str
    bbox_xyxy: NotRequired[list[float]]
    bbox_format: NotRequired[Literal["pixel_xyxy"]]


class RecommendedCropPayload(TypedDict):
    """JSON-ready recommended crop derived from visible-object evidence."""

    bbox_xyxy: list[float]
    bbox_format: Literal["pixel_xyxy"]
    reason: str
    source_object_id: str
    source_object_label: str


@dataclass(frozen=True)
class RecommendedCrop:
    """One crop recommendation for a keyframe or follow-up view."""

    bbox_xyxy: tuple[float, float, float, float]
    reason: str
    source_object_id: str
    source_object_label: str

    def __post_init__(self) -> None:
        """Validate crop recommendation invariants."""
        _validate_pixel_crop_bbox(self.bbox_xyxy)
        _validate_non_empty_text("reason", self.reason)
        _validate_non_empty_text("source_object_id", self.source_object_id)
        _validate_non_empty_text("source_object_label", self.source_object_label)

    def to_payload(self) -> RecommendedCropPayload:
        """Return the JSON-ready recommended crop."""
        return {
            "bbox_xyxy": [float(value) for value in self.bbox_xyxy],
            "bbox_format": "pixel_xyxy",
            "reason": self.reason,
            "source_object_id": self.source_object_id,
            "source_object_label": self.source_object_label,
        }


def recommended_crops_for_matched_objects(
    matched_objects: Sequence[MatchedObjectCropPayload],
    task_description: str | None,
) -> tuple[RecommendedCrop, ...]:
    """Return object-bbox crops and small-affordance subcrops."""
    crops: list[RecommendedCrop] = []
    should_add_affordance_crop = _task_requests_small_affordance(task_description)
    for matched_object in matched_objects:
        bbox_xyxy = _matched_object_pixel_bbox(matched_object)
        if bbox_xyxy is None:
            continue
        crops.append(
            RecommendedCrop(
                bbox_xyxy=bbox_xyxy,
                reason="matched_object_full_bbox",
                source_object_id=matched_object["object_id"],
                source_object_label=matched_object["label"],
            )
        )
        if should_add_affordance_crop:
            crops.append(
                RecommendedCrop(
                    bbox_xyxy=_right_lower_affordance_crop(bbox_xyxy),
                    reason="right_lower_affordance_crop_from_matched_object_bbox",
                    source_object_id=matched_object["object_id"],
                    source_object_label=matched_object["label"],
                )
            )
    return tuple(crops)


def _matched_object_pixel_bbox(
    matched_object: MatchedObjectCropPayload,
) -> tuple[float, float, float, float] | None:
    if matched_object.get("bbox_format") != "pixel_xyxy":
        return None
    bbox_values = matched_object.get("bbox_xyxy")
    if bbox_values is None or len(bbox_values) != 4:
        return None
    bbox_xyxy = (
        float(bbox_values[0]),
        float(bbox_values[1]),
        float(bbox_values[2]),
        float(bbox_values[3]),
    )
    _validate_pixel_crop_bbox(bbox_xyxy)
    return bbox_xyxy


def _right_lower_affordance_crop(
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    left, top, right, bottom = bbox_xyxy
    width = right - left
    height = bottom - top
    return (
        left + width * _RIGHT_LOWER_CROP_LEFT_FRACTION,
        top + height * _RIGHT_LOWER_CROP_TOP_FRACTION,
        right + width * _RIGHT_LOWER_CROP_RIGHT_EXPAND_FRACTION,
        bottom + height * _RIGHT_LOWER_CROP_BOTTOM_EXPAND_FRACTION,
    )


def _task_requests_small_affordance(task_description: str | None) -> bool:
    if task_description is None:
        return False
    query_lower = task_description.lower()
    return any(
        _text_contains_word(query_lower, trigger)
        for trigger in _AFFORDANCE_CROP_QUERY_TRIGGERS
    )


def _text_contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _validate_non_empty_text(field_name: str, field_value: str) -> None:
    if field_value.strip() == "":
        raise ValueError(f"{field_name} must be non-empty")


def _validate_pixel_crop_bbox(
    bbox_xyxy: tuple[float, float, float, float],
) -> None:
    left, top, right, bottom = bbox_xyxy
    if any(not math.isfinite(value) for value in bbox_xyxy):
        raise ValueError(f"bbox_xyxy coordinates must be finite: {bbox_xyxy!r}")
    if any(value < 0.0 for value in bbox_xyxy):
        raise ValueError(f"bbox_xyxy coordinates must be non-negative: {bbox_xyxy!r}")
    if left >= right or top >= bottom:
        raise ValueError("bbox_xyxy must satisfy left < right and top < bottom")


__all__ = [
    "MatchedObjectCropPayload",
    "RecommendedCrop",
    "RecommendedCropPayload",
    "recommended_crops_for_matched_objects",
]
