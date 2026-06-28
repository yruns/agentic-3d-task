"""Lightweight SceneFunc3D evidence-view tools."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .models import ToolInputError
from .scene_context import SceneFunc3dToolScene

_RAW_RGB_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg")
_ERROR_FRAME_ID_PREVIEW = 8
_JPEG_QUALITY = 90
_CROP_HASH_LENGTH = 12

BboxFormat = Literal["normalized", "pixel_xyxy"]


class SceneSummaryArgs(BaseModel):
    """Arguments for ``scene_summary``."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class SceneSummaryResult:
    """Summary payload for one prepared SceneFunc3D scene."""

    visit_id: str
    total_rgb_frames: int
    rgb_frame_ids: tuple[str, ...]
    has_conceptgraph: bool

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready CLI payload."""
        return {
            "visit_id": self.visit_id,
            "total_rgb_frames": self.total_rgb_frames,
            "rgb_frame_ids": list(self.rgb_frame_ids),
            "has_conceptgraph": self.has_conceptgraph,
        }


def scene_summary(
    tool_scene: SceneFunc3dToolScene, args: SceneSummaryArgs
) -> SceneSummaryResult:
    """Return lightweight filesystem metadata for one SceneFunc3D scene."""
    _ = args
    return SceneSummaryResult(
        visit_id=tool_scene.visit_id,
        total_rgb_frames=len(tool_scene.rgb_frame_ids),
        rgb_frame_ids=tool_scene.rgb_frame_ids,
        has_conceptgraph=tool_scene.conceptgraph_dir.is_dir(),
    )


class ViewFrameArgs(BaseModel):
    """Arguments for ``view_frame``."""

    model_config = ConfigDict(extra="forbid")

    frame_ids: tuple[str, ...] = Field(min_length=1)


class FrameObjectPayload(TypedDict, total=False):
    """JSON-ready visible object metadata for one frame."""

    object_id: str
    label: str
    score: float
    bbox_xyxy: list[float]
    bbox_format: BboxFormat
    source: str


class FrameObjectsArgs(BaseModel):
    """Arguments for ``frame_objects``."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(min_length=1)


@dataclass(frozen=True)
class FrameObjectsResult:
    """Visible object summary for one frame."""

    frame_id: str
    objects: tuple[FrameObjectPayload, ...]

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready CLI payload."""
        return {"frame_id": self.frame_id, "objects": list(self.objects)}


class ViewCropArgs(BaseModel):
    """Arguments for ``view_crop``."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(min_length=1)
    bbox: tuple[float, float, float, float]
    bbox_format: BboxFormat = "normalized"

    @field_validator("bbox")
    @classmethod
    def validate_finite_bbox(
        cls, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Validate finite crop coordinates."""
        left, top, right, bottom = bbox
        coordinates = (left, top, right, bottom)
        if any(not math.isfinite(value) for value in coordinates):
            raise ValueError("bbox coordinates must be finite")
        if left >= right or top >= bottom:
            raise ValueError("bbox must satisfy left < right and top < bottom")
        return bbox

    @model_validator(mode="after")
    def validate_bbox_format(self) -> ViewCropArgs:
        """Validate coordinates against the declared bbox coordinate system."""
        if self.bbox_format == "normalized":
            _validate_normalized_bbox(self.bbox)
        else:
            _validate_pixel_bbox(self.bbox)
        return self


class ViewBevArgs(BaseModel):
    """Arguments for ``view_bev``."""

    model_config = ConfigDict(extra="forbid")


class _ObjectFrameMapObject(BaseModel):
    """One visible object entry from ``object_frame_map.json``."""

    model_config = ConfigDict(extra="ignore")

    object_id: int | str
    class_name: str = Field(min_length=1)
    score: float = Field(ge=0.0, allow_inf_nan=False)
    bbox_xyxy: tuple[float, float, float, float] | None = None

    @field_validator("bbox_xyxy")
    @classmethod
    def validate_bbox_xyxy(
        cls, bbox_xyxy: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        """Validate ConceptGraph pixel bbox metadata at the JSON boundary."""
        if bbox_xyxy is None:
            return None
        _validate_pixel_bbox(bbox_xyxy)
        return bbox_xyxy


class _ObjectFrameMapFrame(BaseModel):
    """One frame entry from ``object_frame_map.json``."""

    model_config = ConfigDict(extra="ignore")

    view_id: int
    frame_name: str = Field(min_length=1)
    objects: tuple[_ObjectFrameMapObject, ...] = ()


class _ObjectFrameMapDocument(BaseModel):
    """Validated ``object_frame_map.json`` payload."""

    model_config = ConfigDict(extra="ignore")

    frame_to_objects: dict[str, _ObjectFrameMapFrame]


@dataclass(frozen=True)
class ViewFrame:
    """One rendered SceneFunc3D RGB evidence frame."""

    frame_id: str
    image_path: Path
    depth_path: Path | None = None
    intrinsics_path: Path | None = None
    pose_path: Path | None = None

    def to_payload(self) -> dict[str, object]:
        """Return this frame as a JSON-ready mapping."""
        payload: dict[str, object] = {
            "frame_id": self.frame_id,
            "image_path": str(self.image_path),
        }
        if self.depth_path is not None:
            payload["depth_path"] = str(self.depth_path)
        if self.intrinsics_path is not None:
            payload["intrinsics_path"] = str(self.intrinsics_path)
        if self.pose_path is not None:
            payload["pose_path"] = str(self.pose_path)
        return payload


@dataclass(frozen=True)
class ViewFrameResult:
    """Rendered RGB evidence frames."""

    frames: tuple[ViewFrame, ...]

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready CLI payload."""
        frame_payloads: list[dict[str, object]] = [
            frame.to_payload() for frame in self.frames
        ]
        return {"frames": frame_payloads}


def view_frame(
    tool_scene: SceneFunc3dToolScene, args: ViewFrameArgs, *, out_dir: Path
) -> ViewFrameResult:
    """Copy requested RGB frames to the writable evidence-image directory."""
    _validate_frame_ids(tool_scene, args.frame_ids)
    frames: list[ViewFrame] = []
    for frame_id in args.frame_ids:
        geometry_paths = _resolve_frame_geometry_paths(tool_scene, frame_id)
        frames.append(
            ViewFrame(
                frame_id=frame_id,
                image_path=_write_jpeg_copy(
                    _resolve_rgb_source(tool_scene, frame_id),
                    out_dir / tool_scene.visit_id / f"{frame_id}.jpg",
                ),
                depth_path=geometry_paths.depth_path,
                intrinsics_path=geometry_paths.intrinsics_path,
                pose_path=geometry_paths.pose_path,
            )
        )
    return ViewFrameResult(frames=tuple(frames))


def frame_objects(
    tool_scene: SceneFunc3dToolScene, args: FrameObjectsArgs
) -> FrameObjectsResult:
    """Return ConceptGraph visible objects for one frame."""
    _validate_frame_ids(tool_scene, (args.frame_id,))
    object_frame_map = _load_object_frame_map(tool_scene)
    frame_record = _object_frame_record_for(object_frame_map, args.frame_id)
    objects = tuple(
        _object_payload(visible_object) for visible_object in frame_record.objects
    )
    return FrameObjectsResult(frame_id=args.frame_id, objects=objects)


def view_crop(
    tool_scene: SceneFunc3dToolScene, args: ViewCropArgs, *, out_dir: Path
) -> ViewFrameResult:
    """Render a crop evidence image for one frame."""
    _validate_frame_ids(tool_scene, (args.frame_id,))
    crop_path = _write_crop_jpeg(
        _resolve_rgb_source(tool_scene, args.frame_id),
        out_dir
        / tool_scene.visit_id
        / _crop_filename(
            args.frame_id,
            bbox=args.bbox,
            bbox_format=args.bbox_format,
        ),
        bbox=args.bbox,
        bbox_format=args.bbox_format,
    )
    return ViewFrameResult(
        frames=(ViewFrame(frame_id=args.frame_id, image_path=crop_path),)
    )


def view_bev(
    tool_scene: SceneFunc3dToolScene, args: ViewBevArgs, *, out_dir: Path
) -> ViewFrameResult:
    """Return the top-down BEV image when the prepared scene provides one."""
    _ = args
    _ = out_dir
    bev_path = tool_scene.conceptgraph_dir / "bev" / "scene_bev.png"
    if not bev_path.is_file():
        raise ToolInputError(f"BEV asset is not available: {bev_path}")
    return ViewFrameResult(frames=(ViewFrame(frame_id="bev", image_path=bev_path),))


def _validate_frame_ids(
    tool_scene: SceneFunc3dToolScene, requested_frame_ids: tuple[str, ...]
) -> None:
    available_frame_ids = set(tool_scene.rgb_frame_ids)
    missing_frame_ids = [
        frame_id
        for frame_id in requested_frame_ids
        if frame_id not in available_frame_ids
    ]
    if missing_frame_ids:
        raise ToolInputError(
            f"frame ids not in this SceneFunc3D scene: {missing_frame_ids}; "
            f"available frame ids include: "
            f"{list(tool_scene.rgb_frame_ids[:_ERROR_FRAME_ID_PREVIEW])} "
            f"({len(tool_scene.rgb_frame_ids)} total)"
        )


def _load_object_frame_map(tool_scene: SceneFunc3dToolScene) -> _ObjectFrameMapDocument:
    object_frame_map_path = (
        tool_scene.conceptgraph_dir / "indices" / "object_frame_map.json"
    )
    if not object_frame_map_path.is_file():
        raise ToolInputError(
            "frame_objects requires a visible-object index, which is not available: "
            f"{object_frame_map_path}"
        )
    try:
        payload: object = json.loads(object_frame_map_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ToolInputError(
            "could not read visible-object index: "
            f"path={object_frame_map_path}; error_type={exc.__class__.__name__}"
        ) from exc
    except JSONDecodeError as exc:
        raise ToolInputError(
            f"visible-object index is not valid JSON: {object_frame_map_path}"
        ) from exc
    try:
        return _ObjectFrameMapDocument.model_validate(payload)
    except ValidationError as exc:
        raise ToolInputError(
            "visible-object index failed validation: "
            f"path={object_frame_map_path}; error={_format_validation_error(exc)}"
        ) from exc


def _object_frame_record_for(
    object_frame_map: _ObjectFrameMapDocument, frame_id: str
) -> _ObjectFrameMapFrame:
    for frame_record in object_frame_map.frame_to_objects.values():
        if _frame_id_from_frame_name(frame_record.frame_name) == frame_id:
            return frame_record
    if frame_id in object_frame_map.frame_to_objects:
        return object_frame_map.frame_to_objects[frame_id]
    raise ToolInputError(
        "frame_objects found no visible-object entry for frame: "
        f"frame_id={frame_id!r}"
    )


def _frame_id_from_frame_name(frame_name: str) -> str | None:
    stem = Path(frame_name).stem
    if stem.endswith("-rgb"):
        stem = stem.removesuffix("-rgb")
    if stem.isdigit():
        return stem.zfill(6)
    return None


def _object_payload(visible_object: _ObjectFrameMapObject) -> FrameObjectPayload:
    payload: FrameObjectPayload = {
        "object_id": str(visible_object.object_id),
        "label": visible_object.class_name,
        "score": visible_object.score,
        "source": "object_frame_map",
    }
    if visible_object.bbox_xyxy is not None:
        payload["bbox_xyxy"] = [float(value) for value in visible_object.bbox_xyxy]
        payload["bbox_format"] = "pixel_xyxy"
    return payload


def _validate_normalized_bbox(bbox: tuple[float, float, float, float]) -> None:
    if any(value < 0.0 or value > 1.0 for value in bbox):
        raise ValueError("bbox coordinates must be normalized to [0, 1]")


def _validate_pixel_bbox(bbox: tuple[float, float, float, float]) -> None:
    left, top, right, bottom = bbox
    if any(not math.isfinite(value) for value in bbox):
        raise ValueError("pixel_xyxy bbox coordinates must be finite")
    if any(value < 0.0 for value in bbox):
        raise ValueError("pixel_xyxy bbox coordinates must be non-negative")
    if left >= right or top >= bottom:
        raise ValueError("pixel_xyxy bbox must satisfy left < right and top < bottom")


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ())) or "(root)"
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return "; ".join(parts)


def _resolve_rgb_source(tool_scene: SceneFunc3dToolScene, frame_id: str) -> Path:
    conceptgraph_rgb_path = tool_scene.conceptgraph_rgb_vis_dir / f"{frame_id}-rgb.jpg"
    if conceptgraph_rgb_path.is_file():
        return conceptgraph_rgb_path
    source_frame_rgb_path = tool_scene.source_frame_raw_rgb_path(frame_id)
    if source_frame_rgb_path is not None and source_frame_rgb_path.is_file():
        return source_frame_rgb_path
    for suffix in _RAW_RGB_SUFFIXES:
        raw_rgb_path = tool_scene.raw_dir / f"{frame_id}-rgb{suffix}"
        if raw_rgb_path.is_file():
            return raw_rgb_path
    source_frame_message = (
        f", source_frames RGB path {source_frame_rgb_path}"
        if source_frame_rgb_path is not None
        else ""
    )
    raise ToolInputError(
        f"no RGB image found for frame id {frame_id!r}; checked "
        f"{conceptgraph_rgb_path}{source_frame_message} and raw RGB files under "
        f"{tool_scene.raw_dir}"
    )


@dataclass(frozen=True)
class _FrameGeometryPaths:
    """Optional geometry paths surfaced with view_frame payloads."""

    depth_path: Path | None
    intrinsics_path: Path | None
    pose_path: Path | None


def _resolve_frame_geometry_paths(
    tool_scene: SceneFunc3dToolScene, frame_id: str
) -> _FrameGeometryPaths:
    from codex_agent.scenefunc3d.backends.frame_assets import (
        resolve_available_frame_geometry_assets,
    )

    raw_assets = resolve_available_frame_geometry_assets(tool_scene, frame_id)
    return _FrameGeometryPaths(
        depth_path=_prefer_existing_file(
            raw_assets.depth_path,
            tool_scene.source_frame_depth_path(frame_id),
        ),
        intrinsics_path=_prefer_existing_file(
            raw_assets.intrinsics_path,
            tool_scene.source_frame_intrinsics_path(frame_id),
        ),
        pose_path=_prefer_existing_file(
            raw_assets.pose_path,
            tool_scene.source_frame_pose_path(frame_id),
        ),
    )


def _prefer_existing_file(
    primary_path: Path | None, fallback_path: Path | None
) -> Path | None:
    if primary_path is not None and primary_path.is_file():
        return primary_path
    if fallback_path is not None and fallback_path.is_file():
        return fallback_path
    return None


def _write_jpeg_copy(source_path: Path, destination_path: Path) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ToolInputError(
            "Pillow is required to render SceneFunc3D evidence frames; install the "
            "'vision' extra"
        ) from exc

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            image.convert("RGB").save(
                destination_path, format="JPEG", quality=_JPEG_QUALITY
            )
    except OSError as exc:
        raise ToolInputError(
            f"could not render RGB image {source_path} as JPEG: {exc}"
        ) from exc
    return destination_path


def _write_crop_jpeg(
    source_path: Path,
    destination_path: Path,
    *,
    bbox: tuple[float, float, float, float],
    bbox_format: BboxFormat,
) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ToolInputError(
            "Pillow is required to render SceneFunc3D evidence crops; install the "
            "'vision' extra"
        ) from exc

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            rgb_image = image.convert("RGB")
            crop_box = _pixel_crop_box(
                bbox,
                bbox_format=bbox_format,
                width=rgb_image.width,
                height=rgb_image.height,
            )
            rgb_image.crop(crop_box).save(
                destination_path,
                format="JPEG",
                quality=_JPEG_QUALITY,
            )
    except OSError as exc:
        raise ToolInputError(
            "could not render RGB crop: "
            f"source_path={source_path}; destination_path={destination_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc
    return destination_path


def _crop_filename(
    frame_id: str,
    *,
    bbox: tuple[float, float, float, float],
    bbox_format: BboxFormat,
) -> str:
    digest = hashlib.sha1(
        f"{bbox_format}:{','.join(str(value) for value in bbox)}".encode()
    ).hexdigest()[:_CROP_HASH_LENGTH]
    return f"{frame_id}_crop_{bbox_format}_{digest}.jpg"


def _pixel_crop_box(
    bbox: tuple[float, float, float, float],
    *,
    bbox_format: BboxFormat,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if bbox_format == "pixel_xyxy":
        return _clamped_pixel_crop_box(bbox, width=width, height=height)
    return _normalized_pixel_crop_box(bbox, width=width, height=height)


def _normalized_pixel_crop_box(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    left_px = max(0, min(width - 1, math.floor(left * width)))
    top_px = max(0, min(height - 1, math.floor(top * height)))
    right_px = max(left_px + 1, min(width, math.ceil(right * width)))
    bottom_px = max(top_px + 1, min(height, math.ceil(bottom * height)))
    return (left_px, top_px, right_px, bottom_px)


def _clamped_pixel_crop_box(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    left_px = max(0, min(width - 1, math.floor(left)))
    top_px = max(0, min(height - 1, math.floor(top)))
    right_px = max(left_px + 1, min(width, math.ceil(right)))
    bottom_px = max(top_px + 1, min(height, math.ceil(bottom)))
    return (left_px, top_px, right_px, bottom_px)


__all__ = [
    "FrameObjectPayload",
    "FrameObjectsArgs",
    "FrameObjectsResult",
    "SceneSummaryArgs",
    "SceneSummaryResult",
    "ViewBevArgs",
    "ViewCropArgs",
    "ViewFrame",
    "ViewFrameArgs",
    "ViewFrameResult",
    "frame_objects",
    "scene_summary",
    "view_bev",
    "view_crop",
    "view_frame",
]
