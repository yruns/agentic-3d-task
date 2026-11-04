"""Zoom / crop (image) tool: ``view_crop``.

Returns a **high-resolution crop** of a frame so fine detail the ≤768 px frame
view loses — a brand label, a small object's identity, a book's color — becomes
legible. Three ways to aim it:

* ``frame_id`` + ``bbox`` — crop an explicit region (normalized ``[0,1]`` or raw
  pixels) the agent picked after seeing a downsized frame.
* ``object_id`` — crop around a detected object using its **stored 2D detection
  box** in the object's highest-visibility view with a valid detection box.
* ``object_id`` + ``frame_id`` — crop around ``object_id`` specifically on
  ``frame_id`` (using its stored 2D detection box on that frame if available, or
  its nearest detected view's box).

Needs Pillow for the crop; ``object_id`` mode additionally uses the cached
ConceptGraph selector. Imported lazily by the dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..scene import DEFAULT_MAX_IMAGE_SIZE
from .imaging import (
    box_looks_normalized,
    clamp_box,
    crop_rgb_for_view,
    normalized_box_to_pixels,
)
from .models import ToolInputError
from .scene_context import OpenEqaToolScene, build_selector

#: OpenEQA ConceptGraph GSA detections are extracted at ScanNet 1296x968 RGB
#: resolution. Most clips store 1296x968 raw RGBs (1:1 with xyxy); a few clips
#: store 640x480 raw RGBs and need scaling from (1296, 968).
_DETECTION_IMG_WIDTH = 1296
_DETECTION_IMG_HEIGHT = 968


class ViewCropArgs(BaseModel):
    """Arguments for :func:`view_crop`."""

    model_config = ConfigDict(extra="forbid")

    frame_id: int | None = None
    bbox: list[float] = Field(default_factory=list)
    object_id: int | None = None
    margin: float = Field(default=0.18, ge=0.0, le=2.0)
    max_image_size: int = Field(default=DEFAULT_MAX_IMAGE_SIZE, ge=64, le=2048)

    @model_validator(mode="after")
    def _check_target(self) -> ViewCropArgs:
        if self.object_id is not None:
            if self.bbox and len(self.bbox) != 4:
                raise ValueError(
                    "when bbox is provided it must have 4 values [x0, y0, x1, y1]"
                )
            return self
        if self.frame_id is None or len(self.bbox) != 4:
            raise ValueError(
                'pass {"object_id": 12} (optionally with "frame_id") or '
                '{"frame_id": 120, "bbox": [x0,y0,x1,y1]} '
                "(bbox normalized 0-1 or raw pixels)"
            )
        return self


@dataclass(frozen=True)
class ViewCropResult:
    """A high-res crop plus what region of which frame it shows."""

    image_path: Path
    frame_id: int
    box: tuple[int, int, int, int]
    box_norm: tuple[float, float, float, float]
    source: str
    note: str
    object_id: int | None = None
    object_category: str = ""
    candidate_frame_ids: tuple[int, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image_path": str(self.image_path),
            "frame_id": self.frame_id,
            "crop_box_px": list(self.box),
            "crop_box_norm": [round(v, 4) for v in self.box_norm],
            "source": self.source,
        }
        if self.object_id is not None:
            payload["object_id"] = self.object_id
        if self.object_category:
            payload["object_category"] = self.object_category
        if self.candidate_frame_ids:
            payload["other_frames_for_object"] = list(self.candidate_frame_ids)
        if self.note:
            payload["note"] = self.note
        return payload


def view_crop(
    tool_scene: OpenEqaToolScene, args: ViewCropArgs, *, out_dir: Path
) -> ViewCropResult:
    """Crop a frame to a region (explicit bbox) or around a detected object."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.object_id is not None and len(args.bbox) != 4:
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
    width, height = _image_size(raw_path)
    box = tuple(args.bbox)
    if box_looks_normalized(box):  # type: ignore[arg-type]
        box = normalized_box_to_pixels(box, width, height)  # type: ignore[arg-type]
    clamped = clamp_box(box, width, height)  # type: ignore[arg-type]
    box_token = "_".join(str(coord) for coord in clamped)
    destination = out_dir / f"{tool_scene.clip_id}_crop_{frame_id:06d}_{box_token}.jpg"
    saved, applied = crop_rgb_for_view(
        raw_path, destination, box=clamped, max_size=args.max_image_size
    )
    return ViewCropResult(
        image_path=saved,
        frame_id=frame_id,
        box=applied,
        box_norm=_to_norm(applied, width, height),
        source="bbox",
        note="",
        object_id=args.object_id,
    )


def _crop_by_object(
    tool_scene: OpenEqaToolScene, args: ViewCropArgs, *, out_dir: Path
) -> ViewCropResult:
    conceptgraph_dir = tool_scene.require_conceptgraph()
    selector = build_selector(conceptgraph_dir)
    obj = _find_object(selector, args.object_id)
    stride = max(1, int(getattr(selector, "stride", 1) or 1))

    requested_view_id: int | None = None
    if args.frame_id is not None:
        if args.frame_id not in set(tool_scene.scene.rgb_frame_ids):
            raise ToolInputError(
                f"frame id {args.frame_id} not in this clip; valid range is "
                f"[{tool_scene.scene.rgb_frame_ids[0]}, "
                f"{tool_scene.scene.rgb_frame_ids[-1]}]"
            )
        requested_view_id = args.frame_id // stride

    view_id, det_box = _best_view_and_box(
        selector, obj, requested_view_id=requested_view_id
    )
    frame_id = (
        args.frame_id
        if args.frame_id is not None
        else _nearest_available(tool_scene, view_id * stride)
    )
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
    clamped = clamp_box(box, width, height)
    box_token = "_".join(str(coord) for coord in clamped)
    destination = (
        out_dir
        / f"{tool_scene.clip_id}_objcrop_{args.object_id}_{frame_id:06d}_{box_token}.jpg"
    )
    saved, applied = crop_rgb_for_view(
        raw_path, destination, box=clamped, max_size=args.max_image_size
    )
    other_frames = _top_object_frame_ids(
        tool_scene, selector, obj, stride=stride, exclude_frame_id=frame_id
    )
    category = str(getattr(obj, "object_tag", "") or getattr(obj, "category", "") or "")
    return ViewCropResult(
        image_path=saved,
        frame_id=frame_id,
        box=applied,
        box_norm=_to_norm(applied, width, height),
        source="object",
        note=note,
        object_id=args.object_id,
        object_category=category,
        candidate_frame_ids=other_frames,
    )


def _find_object(selector: Any, object_id: int | None) -> Any:
    for obj in selector.objects:
        if obj.obj_id == object_id:
            return obj
    raise ToolInputError(
        f"object id {object_id} not in this scene; call list_objects for valid ids"
    )


def _best_box_for_view(
    image_idx: list[int], boxes: list[Any], view_id: int
) -> tuple[float, float, float, float] | None:
    """Return the largest stored 2D detection box for ``view_id``, if any."""
    best_box: tuple[float, float, float, float] | None = None
    best_area = -1.0
    for det_index, vid in enumerate(image_idx):
        if int(vid) != view_id or det_index >= len(boxes):
            continue
        candidate = _as_box(boxes[det_index])
        if candidate is None:
            continue
        area = max(0.0, candidate[2] - candidate[0]) * max(
            0.0, candidate[3] - candidate[1]
        )
        if area > best_area:
            best_area = area
            best_box = candidate
    return best_box


def _best_view_and_box(
    selector: Any,
    obj: Any,
    *,
    requested_view_id: int | None = None,
) -> tuple[int, tuple[float, float, float, float] | None]:
    """Pick the object's highest-visibility view and its stored 2D box, if any."""
    views = getattr(selector, "object_to_views", {}).get(obj.obj_id, [])
    ranked = [int(v) for v, _ in views] if views else []
    image_idx = [int(v) for v in (getattr(obj, "image_idx", []) or [])]
    boxes = list(getattr(obj, "xyxy", []) or [])

    if requested_view_id is not None:
        direct_box = _best_box_for_view(image_idx, boxes, requested_view_id)
        if direct_box is not None:
            return requested_view_id, direct_box
        # If the exact requested view has no stored detection box, check nearby views
        if image_idx and boxes:
            nearest_vid = min(image_idx, key=lambda vid: abs(vid - requested_view_id))
            if abs(nearest_vid - requested_view_id) <= 3:
                near_box = _best_box_for_view(image_idx, boxes, nearest_vid)
                if near_box is not None:
                    return requested_view_id, near_box
        return requested_view_id, None

    # 1. Search ranked visibility views for the first view that has a valid 2D box
    for view_id in ranked:
        box = _best_box_for_view(image_idx, boxes, view_id)
        if box is not None:
            return view_id, box

    # 2. Fallback to any detection in image_idx (picking the largest box)
    if image_idx and boxes:
        best_vid = image_idx[0]
        best_box: tuple[float, float, float, float] | None = None
        best_area = -1.0
        for vid in dict.fromkeys(image_idx):
            b = _best_box_for_view(image_idx, boxes, vid)
            if b is not None:
                area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                if area > best_area:
                    best_area = area
                    best_box = b
                    best_vid = vid
        if best_box is not None:
            return best_vid, best_box

    # 3. Last resort: highest-visibility view without a box
    if ranked:
        return ranked[0], None
    return 0, None


def _top_object_frame_ids(
    tool_scene: OpenEqaToolScene,
    selector: Any,
    obj: Any,
    *,
    stride: int,
    exclude_frame_id: int,
    limit: int = 4,
) -> tuple[int, ...]:
    views = getattr(selector, "object_to_views", {}).get(obj.obj_id, [])
    ranked = [int(v) for v, _ in views] if views else []
    out: list[int] = []
    for vid in ranked:
        fid = _nearest_available(tool_scene, vid * stride)
        if fid != exclude_frame_id and fid not in out:
            out.append(fid)
        if len(out) >= limit:
            break
    return tuple(out)


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
    # OpenEQA detections are in raw RGB pixel coordinates (1296x968), or in the
    # synthetic test image's own pixel space when boxes fit inside (width, height).
    # Only scale when the raw RGB on disk is smaller than the detection space
    # (e.g. the two 640x480 clips whose detections live in 1296x968).
    if box[2] <= width + 2.0 and box[3] <= height + 2.0:
        sx, sy = 1.0, 1.0
    else:
        sx = width / _DETECTION_IMG_WIDTH
        sy = height / _DETECTION_IMG_HEIGHT
    left, top, right, bottom = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    pad_x = (right - left) * margin
    pad_y = (bottom - top) * margin
    return (left - pad_x, top - pad_y, right + pad_x, bottom + pad_y)


def _to_norm(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[float, float, float, float]:
    w = max(1, width)
    h = max(1, height)
    return (box[0] / w, box[1] / h, box[2] / w, box[3] / h)


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
