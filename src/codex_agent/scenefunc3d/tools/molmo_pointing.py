"""Data contracts and parser for SceneFunc3D Molmo pointing output."""

from __future__ import annotations

import html
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ...errors import SceneFunc3dDataError

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_POINT_TAG_RE = re.compile(
    r"<point\b(?P<attributes>[^>]*)>(?P<label_text>.*?)</point>",
    flags=re.IGNORECASE | re.DOTALL,
)
_POINT_ATTR_RE = re.compile(
    r"(?P<name>[A-Za-z_:][\w:.-]*)\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    flags=re.DOTALL,
)
_PERCENT_MIN = 0.0
_PERCENT_MAX = 100.0


class MolmoPointPayload(TypedDict):
    """JSON-ready payload for one Molmo point."""

    x_px: float
    y_px: float
    source: str
    label: str


class MolmoPointResultPayload(TypedDict):
    """JSON-ready payload for Molmo pointing results."""

    frame_id: str
    prompt: str
    points: list[MolmoPointPayload]
    raw_text_path: str
    overlay_path: str


@dataclass(frozen=True)
class MolmoPoint:
    """One Molmo point converted from percent image coordinates to pixels."""

    x_px: float
    y_px: float
    source: str
    label: str

    def __post_init__(self) -> None:
        """Validate directly constructed point contracts."""
        if not math.isfinite(self.x_px) or self.x_px < 0.0:
            raise SceneFunc3dDataError(f"invalid Molmo point x_px={self.x_px!r}")
        if not math.isfinite(self.y_px) or self.y_px < 0.0:
            raise SceneFunc3dDataError(f"invalid Molmo point y_px={self.y_px!r}")

    def to_payload(self) -> MolmoPointPayload:
        """Return this point as a JSON-ready mapping."""
        return {
            "x_px": self.x_px,
            "y_px": self.y_px,
            "source": self.source,
            "label": self.label,
        }


class MolmoPointArgs(BaseModel):
    """Arguments for a future Molmo pointing tool."""

    model_config = ConfigDict(extra="forbid")

    frame_id: NonEmptyText
    image_path: Path
    prompt: NonEmptyText
    image_width: int = Field(gt=0, strict=True)
    image_height: int = Field(gt=0, strict=True)


@dataclass(frozen=True)
class MolmoPointResult:
    """Molmo pointing result metadata for one SceneFunc3D frame."""

    frame_id: str
    prompt: str
    points: tuple[MolmoPoint, ...]
    raw_text_path: Path
    overlay_path: Path

    def to_payload(self) -> MolmoPointResultPayload:
        """Return the JSON-ready CLI payload."""
        return {
            "frame_id": self.frame_id,
            "prompt": self.prompt,
            "points": [point.to_payload() for point in self.points],
            "raw_text_path": str(self.raw_text_path),
            "overlay_path": str(self.overlay_path),
        }


def parse_molmo_points(
    raw_text: str, *, image_width: int, image_height: int
) -> tuple[MolmoPoint, ...]:
    """Parse Molmo ``<point>`` percent coordinates into pixel coordinates.

    Molmo emits XML-ish point tags such as ``<point x="81" y="62">label</point>``.
    Coordinates are percentages of the image dimensions and are converted to
    pixel-space floats. Invalid point tags fail fast because downstream mask
    generation should not silently continue from malformed model output.
    """
    _validate_image_dimensions(image_width=image_width, image_height=image_height)

    points: list[MolmoPoint] = []
    for match in _POINT_TAG_RE.finditer(raw_text):
        attributes = _parse_point_attributes(match.group("attributes"))
        x_percent = _parse_percent_attribute(attributes, "x")
        y_percent = _parse_percent_attribute(attributes, "y")
        label = _point_label(attributes, match.group("label_text"))
        points.append(
            MolmoPoint(
                x_px=image_width * x_percent / 100.0,
                y_px=image_height * y_percent / 100.0,
                source=match.group(0),
                label=label,
            )
        )
    return tuple(points)


def _validate_image_dimensions(*, image_width: int, image_height: int) -> None:
    if image_width <= 0:
        raise SceneFunc3dDataError(f"image_width must be positive; got {image_width!r}")
    if image_height <= 0:
        raise SceneFunc3dDataError(
            f"image_height must be positive; got {image_height!r}"
        )


def _parse_point_attributes(raw_attributes: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for match in _POINT_ATTR_RE.finditer(raw_attributes):
        attributes[match.group("name").lower()] = html.unescape(match.group("value"))
    return attributes


def _parse_percent_attribute(
    attributes: Mapping[str, str], attribute_name: str
) -> float:
    if attribute_name not in attributes:
        raise SceneFunc3dDataError(
            f"missing Molmo point {attribute_name} percent attribute"
        )
    raw_value = attributes[attribute_name]
    try:
        percent = float(raw_value)
    except ValueError as exc:
        raise SceneFunc3dDataError(
            f"invalid Molmo point {attribute_name} percent={raw_value!r}; "
            f"expected a number from {_PERCENT_MIN:g} to {_PERCENT_MAX:g}"
        ) from exc
    if not _PERCENT_MIN <= percent <= _PERCENT_MAX:
        raise SceneFunc3dDataError(
            f"invalid Molmo point {attribute_name} percent={raw_value!r}; "
            f"expected a number from {_PERCENT_MIN:g} to {_PERCENT_MAX:g}"
        )
    return percent


def _point_label(attributes: Mapping[str, str], label_text: str) -> str:
    alt_text = attributes.get("alt", "").strip()
    if alt_text:
        return alt_text
    return html.unescape(label_text).strip()


__all__ = [
    "MolmoPoint",
    "MolmoPointArgs",
    "MolmoPointPayload",
    "MolmoPointResult",
    "MolmoPointResultPayload",
    "parse_molmo_points",
]
