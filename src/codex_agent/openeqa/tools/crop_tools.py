"""Zoom / crop (image) tool: ``view_crop``.

Returns a **high-resolution crop** of a frame so fine detail the ≤768 px frame
view loses — a brand label, a small object's identity, a book's color — becomes
legible. Two ways to aim it:

* ``frame_id`` + ``bbox`` — crop an explicit region (normalized ``[0,1]`` or raw
  pixels) the agent picked after seeing the downsized frame. Deterministic, the
  primary mode.
* ``object_id`` — crop around a detected object using its **stored 2D detection
  box** in the object's best view (reuses the detector's own box; no fragile 3D
  projection). Falls back to a centered crop if the object has no usable box.

Needs Pillow for the crop; ``object_id`` mode additionally builds the
ConceptGraph selector. Imported lazily by the dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..scene import DEFAULT_MAX_IMAGE_SIZE
from .imaging import box_looks_normalized, crop_rgb_for_view, normalized_box_to_pixels
from .models import ToolInputError
from .scene_context import OpenEqaToolScene, build_selector

#: Detector image dimensions the stored ``xyxy`` boxes live in (see
#: ``keyframe.keyframe_selector._VIS_IMG_WIDTH/_HEIGHT``). Object-mode boxes are
#: scaled from this space to each clip's raw RGB size.
_DETECTION_IMG_WIDTH = 1200
_DETECTION_IMG_HEIGHT = 680


class ViewCropArgs(BaseModel):
    """Arguments for :func:`view_crop`."""

    model_config = ConfigDict(extra="forbid")

    frame_id: int | None = None
    bbox: list[float] = Field(default_factory=list)
    object_id: int | None = None
    margin: float = Field(default=0.15, ge=0.0, le=2.0)
    max_image_size: int = Field(default=DEFAULT_MAX_IMAGE_SIZE, ge=64, le=2048)

    @model_validator(mode="after")
    def _check_target(self) -> ViewCropArgs:
        if self.object_id is not None:
            return self
        if self.frame_id is None or len(self.bbox) != 4:
            raise ValueError(
                'pass {"object_id": 12} or {"frame_id": 120, "bbox": [x0,y0,x1,y1]} '
                "(bbox normalized 0-1 or raw pixels)"
            )
        return self


@dataclass(frozen=True)
class ViewCropResult:
    """A high-res crop plus what region of which frame it shows."""

    image_path: Path
    frame_id: int
    box: tuple[int, int, int, int]
    source: str
    note: str
    object_id: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image_path": str(self.image_path),
            "frame_id": self.frame_id,
            "crop_box_px": list(self.box),
            "source": self.source,
        }
        if self.object_id is not None:
            payload["object_id"] = self.object_id
        if self.note:
            payload["note"] = self.note
        return payload


def view_crop(
    tool_scene: OpenEqaToolScene, args: ViewCropArgs, *, out_dir: Path
) -> ViewCropResult:
    """Crop a frame to a region (explicit bbox) or around a detected object."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.object_id is not None:
        return _crop_by_object(tool_scene, args, out_dir=out_dir)
    return _crop_by_bbox(tool_scene, args, out_dir=out_dir)


def _crop_by_bbox(
    tool_scene: OpenEqaToolScene, args: ViewCropArgs, *, out_dir: Path
) -> ViewCropResult:
    scene = tool_scene.scene
    frame_id = args.frame_id
    assert frame_id is not None  # guaranteed by the validator
    if frame_id not in set(scene.rgb_frame_ids):
        raise ToolInputError(
            f"frame id {frame_id} not in this clip; valid range is "
            f"[{scene.rgb_frame_ids[0]}, {scene.rgb_frame_ids[-1]}] "
            f"({scene.total_frames} frames)"
        )
    raw_path = scene.raw_rgb_path(frame_id)
    box = tuple(args.bbox)
    if box_looks_normalized(box):  # type: ignore[arg-type]
        width, height = _image_size(raw_path)
        box = normalized_box_to_pixels(box, width, height)  # type: ignore[arg-type]
    destination = out_dir / f"{tool_scene.clip_id}_crop_{frame_id:06d}.jpg"
    saved, applied = crop_rgb_for_view(
        raw_path, destination, box=box, max_size=args.max_image_size  # type: ignore[arg-type]
    )
    return ViewCropResult(
        image_path=saved,
        frame_id=frame_id,
        box=applied,
        source="bbox",
        note="",
    )


def _crop_by_object(
    tool_scene: OpenEqaToolScene, args: ViewCropArgs, *, out_dir: Path
) -> ViewCropResult:
    conceptgraph_dir = tool_scene.require_conceptgraph()
    selector = build_selector(conceptgraph_dir)
    obj = _find_object(selector, args.object_id)
    view_id, det_box = _best_view_and_box(selector, obj)
    stride = int(getattr(selector, "stride", 1) or 1)
    frame_id = _nearest_available(tool_scene, view_id * stride)
    raw_path = tool_scene.scene.raw_rgb_path(frame_id)
    width, height = _image_size(raw_path)

    note = ""
    if det_box is not None:
        box = _scale_detection_box(det_box, width, height, margin=args.margin)
    else:
        box = _center_box(width, height)
        note = (
            f"object {args.object_id} has no stored 2D detection box; "
            "showing a centered crop of its best view"
        )
    destination = (
        out_dir / f"{tool_scene.clip_id}_objcrop_{args.object_id}_{frame_id:06d}.jpg"
    )
    saved, applied = crop_rgb_for_view(
        raw_path, destination, box=box, max_size=args.max_image_size
    )
    return ViewCropResult(
        image_path=saved,
        frame_id=frame_id,
        box=applied,
        source="object",
        note=note,
        object_id=args.object_id,
    )


def _find_object(selector: Any, object_id: int | None) -> Any:
    for obj in selector.objects:
        if obj.obj_id == object_id:
            return obj
    raise ToolInputError(
        f"object id {object_id} not in this scene; call list_objects for valid ids"
    )


def _best_view_and_box(
    selector: Any, obj: Any
) -> tuple[int, tuple[float, float, float, float] | None]:
    """Pick the object's highest-visibility view and its stored 2D box, if any."""
    views = getattr(selector, "object_to_views", {}).get(obj.obj_id, [])
    ranked = [int(v) for v, _ in views] if views else []
    image_idx = list(getattr(obj, "image_idx", []) or [])
    boxes = list(getattr(obj, "xyxy", []) or [])

    for view_id in ranked:
        if view_id in image_idx:
            det_index = image_idx.index(view_id)
            if det_index < len(boxes):
                return view_id, _as_box(boxes[det_index])
        return view_id, None  # best view known, but no box stored for it
    if image_idx:  # no visibility ranking; use the first detection
        box = _as_box(boxes[0]) if boxes else None
        return int(image_idx[0]), box
    return 0, None


def _as_box(raw: Any) -> tuple[float, float, float, float] | None:
    values = [float(v) for v in list(raw)]
    if len(values) < 4:
        return None
    return (values[0], values[1], values[2], values[3])


def _scale_detection_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    margin: float,
) -> tuple[float, float, float, float]:
    sx = width / _DETECTION_IMG_WIDTH
    sy = height / _DETECTION_IMG_HEIGHT
    left, top, right, bottom = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    pad_x = (right - left) * margin
    pad_y = (bottom - top) * margin
    return (left - pad_x, top - pad_y, right + pad_x, bottom + pad_y)


def _center_box(width: int, height: int) -> tuple[float, float, float, float]:
    return (width * 0.25, height * 0.25, width * 0.75, height * 0.75)


def _nearest_available(tool_scene: OpenEqaToolScene, frame_id: int) -> int:
    available = tool_scene.scene.rgb_frame_ids
    if frame_id in set(available):
        return frame_id
    return min(available, key=lambda fid: abs(fid - frame_id))


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:  # optional vision dependency
        from ...errors import OpenEqaDataError

        raise OpenEqaDataError(
            "Pillow is required for OpenEQA image tools; install the 'vision' extra"
        ) from exc
    with Image.open(path) as image:
        return image.width, image.height


__all__ = [
    "ViewCropArgs",
    "ViewCropResult",
    "view_crop",
]
