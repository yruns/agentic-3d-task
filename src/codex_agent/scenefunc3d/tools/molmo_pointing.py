"""Data contracts and parser for SceneFunc3D Molmo pointing output."""

from __future__ import annotations

import html
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field, FilePath, StringConstraints

from ...errors import SceneFunc3dDataError
from .models import ToolInputError

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SafeFrameIdText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
]

_POINT_TAG_RE = re.compile(
    r"<point\b(?P<attributes>[^>]*)>(?P<label_text>.*?)</point>",
    flags=re.IGNORECASE | re.DOTALL,
)
_POINT_START_RE = re.compile(r"<point\b", flags=re.IGNORECASE)
_POINT_ATTR_RE = re.compile(
    r"(?P<name>[A-Za-z_:][\w:.-]*)\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    flags=re.DOTALL,
)
_PERCENT_MIN = 0.0
_PERCENT_MAX = 100.0
_OVERLAY_POINT_RADIUS_PX = 5
_OVERLAY_POINT_OUTLINE_WIDTH_PX = 2
_JPEG_QUALITY = 95


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

    frame_id: SafeFrameIdText
    image_path: FilePath
    prompt: NonEmptyText
    image_width: int = Field(gt=0, strict=True)
    image_height: int = Field(gt=0, strict=True)


@dataclass(frozen=True)
class MolmoBackendConfig:
    """Explicit Molmo backend selection."""

    model_name: str
    model_path: Path

    def __post_init__(self) -> None:
        """Validate backend identity before any heavy loading attempt."""
        if not self.model_name.strip():
            raise ToolInputError("Molmo backend model_name must not be empty")


@dataclass(frozen=True)
class MolmoPointResult:
    """Molmo pointing result metadata for one SceneFunc3D frame."""

    frame_id: str
    prompt: str
    points: tuple[MolmoPoint, ...]
    raw_text_path: Path
    overlay_path: Path

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready CLI payload."""
        return {
            "frame_id": self.frame_id,
            "prompt": self.prompt,
            "points": [point.to_payload() for point in self.points],
            "raw_text_path": str(self.raw_text_path),
            "overlay_path": str(self.overlay_path),
        }


def run_molmo_backend(config: MolmoBackendConfig) -> None:
    """Validate configured Molmo backend before heavy model loading."""
    if not config.model_path.exists():
        raise ToolInputError(
            f"Molmo backend unavailable: {config.model_name} at {config.model_path}"
        )
    raise ToolInputError(
        "Molmo backend execution is not configured: "
        f"{config.model_name} at {config.model_path}"
    )


def molmo_point(
    args: MolmoPointArgs,
    *,
    out_dir: Path,
    backend_config_path: Path | None,
) -> MolmoPointResult:
    """Call Molmo sidecar, parse points, and write raw/overlay artifacts."""
    if backend_config_path is None:
        raise ToolInputError("molmo_point backend config is required")

    from codex_agent.scenefunc3d.backends.config import (
        ensure_path_under_roots,
        load_backend_settings,
    )
    from codex_agent.scenefunc3d.backends.molmo_rpc import request_molmo_point

    settings = load_backend_settings(backend_config_path)
    artifact_out_dir = ensure_path_under_roots(
        out_dir,
        roots=settings.allowed_output_roots,
        field_name="out_dir",
    )
    request_id = f"{args.frame_id}_molmo"
    response = request_molmo_point(settings, request_id=request_id, args=args)
    raw_text_path = _write_raw_text(artifact_out_dir, args.frame_id, response.raw_text)
    try:
        points = parse_molmo_points(
            response.raw_text,
            image_width=args.image_width,
            image_height=args.image_height,
        )
    except SceneFunc3dDataError as exc:
        raise ToolInputError(
            "Molmo point response could not be parsed: "
            f"frame_id={args.frame_id!r}; raw_text_path={raw_text_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc
    if not points:
        raise ToolInputError(
            "Molmo point response contained no <point> tags: "
            f"frame_id={args.frame_id!r}; raw_text_path={raw_text_path}"
        )
    overlay_path = _write_point_overlay(
        args.image_path,
        artifact_out_dir,
        args.frame_id,
        points,
    )
    return MolmoPointResult(
        frame_id=args.frame_id,
        prompt=args.prompt,
        points=points,
        raw_text_path=raw_text_path,
        overlay_path=overlay_path,
    )


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

    point_matches = tuple(_POINT_TAG_RE.finditer(raw_text))
    _validate_complete_point_tags(raw_text, point_matches)

    points: list[MolmoPoint] = []
    for match in point_matches:
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


def _validate_complete_point_tags(
    raw_text: str, point_matches: tuple[re.Match[str], ...]
) -> None:
    point_start_offsets = tuple(
        match.start() for match in _POINT_START_RE.finditer(raw_text)
    )
    complete_tag_offsets = {match.start() for match in point_matches}
    malformed_offsets = tuple(
        offset for offset in point_start_offsets if offset not in complete_tag_offsets
    )
    if malformed_offsets:
        raise SceneFunc3dDataError(
            "malformed Molmo point tag: every '<point' fragment must have a "
            f"matching '</point>' tag; malformed offsets={malformed_offsets}"
        )


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


def _write_raw_text(out_dir: Path, frame_id: str, raw_text: str) -> Path:
    path = out_dir / "molmo" / f"{frame_id}_raw.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw_text, encoding="utf-8")
    except OSError as exc:
        raise ToolInputError(
            "could not write Molmo raw text artifact: "
            f"path={path}; error_type={exc.__class__.__name__}"
        ) from exc
    return path


def _write_point_overlay(
    image_path: Path,
    out_dir: Path,
    frame_id: str,
    points: tuple[MolmoPoint, ...],
) -> Path:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ToolInputError(
            "Pillow is required to render Molmo point overlays; install the "
            "'vision' extra"
        ) from exc

    overlay_path = out_dir / "molmo" / f"{frame_id}_points.jpg"
    try:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            overlay = image.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for point in points:
            left = point.x_px - _OVERLAY_POINT_RADIUS_PX
            top = point.y_px - _OVERLAY_POINT_RADIUS_PX
            right = point.x_px + _OVERLAY_POINT_RADIUS_PX
            bottom = point.y_px + _OVERLAY_POINT_RADIUS_PX
            draw.ellipse(
                (left, top, right, bottom),
                outline=(255, 0, 0),
                width=_OVERLAY_POINT_OUTLINE_WIDTH_PX,
            )
            draw.line(
                (
                    point.x_px - _OVERLAY_POINT_RADIUS_PX,
                    point.y_px,
                    point.x_px + _OVERLAY_POINT_RADIUS_PX,
                    point.y_px,
                ),
                fill=(255, 0, 0),
                width=_OVERLAY_POINT_OUTLINE_WIDTH_PX,
            )
            draw.line(
                (
                    point.x_px,
                    point.y_px - _OVERLAY_POINT_RADIUS_PX,
                    point.x_px,
                    point.y_px + _OVERLAY_POINT_RADIUS_PX,
                ),
                fill=(255, 0, 0),
                width=_OVERLAY_POINT_OUTLINE_WIDTH_PX,
            )
        overlay.save(overlay_path, format="JPEG", quality=_JPEG_QUALITY)
    except OSError as exc:
        raise ToolInputError(
            "could not render Molmo point overlay: "
            f"image_path={image_path}; overlay_path={overlay_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc
    return overlay_path


__all__ = [
    "MolmoBackendConfig",
    "MolmoPoint",
    "MolmoPointArgs",
    "MolmoPointPayload",
    "MolmoPointResult",
    "MolmoPointResultPayload",
    "molmo_point",
    "parse_molmo_points",
    "run_molmo_backend",
]
