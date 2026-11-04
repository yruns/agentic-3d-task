"""Frame-switching & trajectory-panning (image) tool: ``view_frame``.

Fetches one or more raw first-person frames by ``frame_id`` / ``frame_ids``,
by a trajectory neighborhood around ``around_frame_id`` (with ``step``), or by
BEV ``grid_cells`` ``[[row, col], ...]``, downsizes them into a writable
scratch dir, and builds a single labelled ``contact_sheet`` when multiple
frames are returned.

When ConceptGraph assets are present for the clip, each returned frame is
also annotated with ``visible_objects`` (ordered left-to-right with normalized
2D ``bbox_norm`` ``[x0, y0, x1, y1]`` ready for ``view_crop``) and
``covered_grids``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..scene import DEFAULT_MAX_IMAGE_SIZE, downsize_rgb_for_view
from .imaging import compose_contact_sheet
from .models import ToolInputError
from .scene_context import OpenEqaToolScene, build_selector

#: Most frames returned by a single call (keeps the turn's image budget bounded).
_MAX_FRAMES = 4
_MAX_VISIBLE_OBJECTS = 8
_MAX_COVERED_GRIDS = 6
_DETECTION_IMG_WIDTH = 1296
_DETECTION_IMG_HEIGHT = 968


class ViewFrameArgs(BaseModel):
    """Arguments for :func:`view_frame`."""

    model_config = ConfigDict(extra="forbid")

    frame_id: int | None = None
    frame_ids: list[int] = Field(default_factory=list)
    around_frame_id: int | None = None
    step: int = Field(default=10, ge=1, le=200)
    grid_cells: list[tuple[int, int]] = Field(default_factory=list)
    max_image_size: int = Field(default=DEFAULT_MAX_IMAGE_SIZE, ge=64, le=2048)


@dataclass(frozen=True)
class ViewFrame:
    """One fetched, downsized first-person frame with optional scene annotations."""

    frame_id: int
    image_path: Path
    visible_objects: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    covered_grids: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frame_id": self.frame_id,
            "image_path": str(self.image_path),
        }
        if self.visible_objects:
            payload["visible_objects"] = list(self.visible_objects)
        if self.covered_grids:
            payload["covered_grids"] = [list(c) for c in self.covered_grids]
        return payload


@dataclass(frozen=True)
class ViewFrameResult:
    """Fetched frames plus the clip's available frame range."""

    frames: tuple[ViewFrame, ...]
    total_frames: int
    frame_id_range: tuple[int, int]
    note: str
    contact_sheet_path: Path | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frames": [frame.to_payload() for frame in self.frames],
            "total_frames": self.total_frames,
            "frame_id_range": list(self.frame_id_range),
            "note": self.note,
        }
        if self.contact_sheet_path is not None:
            payload["contact_sheet"] = str(self.contact_sheet_path)
            payload["tip"] = (
                "view_image the contact_sheet to see all frames at once; "
                "use view_crop with object_id or bbox_norm for a high-res zoom"
            )
        return payload


def view_frame(
    tool_scene: OpenEqaToolScene, args: ViewFrameArgs, *, out_dir: Path
) -> ViewFrameResult:
    """Fetch one or more raw frames by id, neighborhood, or grid cell."""
    scene = tool_scene.scene
    selector = _try_get_selector(tool_scene)
    requested = _requested_ids(args, tool_scene=tool_scene, selector=selector)
    available = set(scene.rgb_frame_ids)
    missing = [fid for fid in requested if fid not in available]
    if missing:
        raise ToolInputError(
            f"frame ids not in this clip: {missing}; valid range is "
            f"[{scene.rgb_frame_ids[0]}, {scene.rgb_frame_ids[-1]}] "
            f"({scene.total_frames} frames). Nearest available to {missing[0]}: "
            f"{_nearest_frame_id(scene.rgb_frame_ids, missing[0])}"
        )

    capped = requested[:_MAX_FRAMES]
    note = (
        ""
        if len(requested) <= _MAX_FRAMES
        else f"(showing {_MAX_FRAMES} of {len(requested)} requested)"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = tuple(
        _build_view_frame(
            tool_scene,
            selector,
            frame_id,
            out_dir=out_dir,
            max_image_size=args.max_image_size,
        )
        for frame_id in capped
    )
    contact_sheet = _contact_sheet(tool_scene, frames, out_dir)
    return ViewFrameResult(
        frames=frames,
        total_frames=scene.total_frames,
        frame_id_range=(scene.rgb_frame_ids[0], scene.rgb_frame_ids[-1]),
        note=note,
        contact_sheet_path=contact_sheet,
    )


def _build_view_frame(
    tool_scene: OpenEqaToolScene,
    selector: Any | None,
    frame_id: int,
    *,
    out_dir: Path,
    max_image_size: int,
) -> ViewFrame:
    raw_path = tool_scene.scene.raw_rgb_path(frame_id)
    image_path = downsize_rgb_for_view(
        raw_path,
        out_dir / f"{tool_scene.clip_id}_frame_{frame_id:06d}.jpg",
        max_size=max_image_size,
    )
    vis_objects, cov_grids = annotate_frame_from_selector(selector, frame_id)
    return ViewFrame(
        frame_id=frame_id,
        image_path=image_path,
        visible_objects=vis_objects,
        covered_grids=cov_grids,
    )


def annotate_frame_from_selector(
    selector: Any | None, frame_id: int
) -> tuple[tuple[dict[str, Any], ...], tuple[tuple[int, int], ...]]:
    """Extract visible objects (with normalized 2D boxes) and grid cells for ``frame_id``."""
    if selector is None:
        return (), ()
    stride = max(1, int(getattr(selector, "stride", 1) or 1))
    view_id = frame_id // stride
    view_to_objects = getattr(selector, "view_to_objects", {}) or {}
    view_to_grids = getattr(selector, "view_to_grids", {}) or {}
    by_id = {obj.obj_id: obj for obj in (getattr(selector, "objects", []) or [])}

    raw_objs = view_to_objects.get(view_id, [])
    entries: list[tuple[float, dict[str, Any]]] = []
    for oid, score in raw_objs[:12]:
        obj = by_id.get(int(oid))
        if obj is None:
            continue
        category = str(obj.object_tag or obj.category or "object")
        item: dict[str, Any] = {"obj_id": int(oid), "category": category}
        norm_box = _norm_box_for_object_in_view(obj, view_id)
        if norm_box is not None:
            item["bbox_norm"] = [round(v, 3) for v in norm_box]
            sort_x = 0.5 * (norm_box[0] + norm_box[2])
        else:
            sort_x = 10.0 - float(score)
        entries.append((sort_x, item))

    entries.sort(key=lambda pair: pair[0])
    visible_objects = tuple(item for _, item in entries[:_MAX_VISIBLE_OBJECTS])

    raw_grids = view_to_grids.get(view_id, [])
    covered_grids = tuple(
        (int(cell[0]), int(cell[1])) for cell, _ in raw_grids[:_MAX_COVERED_GRIDS]
    )
    return visible_objects, covered_grids


def _norm_box_for_object_in_view(
    obj: Any, view_id: int
) -> tuple[float, float, float, float] | None:
    image_idx = [int(v) for v in (getattr(obj, "image_idx", []) or [])]
    boxes = list(getattr(obj, "xyxy", []) or [])
    best_box: tuple[float, float, float, float] | None = None
    best_area = -1.0
    for det_idx, vid in enumerate(image_idx):
        if vid != view_id or det_idx >= len(boxes):
            continue
        vals = [float(v) for v in list(boxes[det_idx])]
        if len(vals) < 4:
            continue
        area = max(0.0, vals[2] - vals[0]) * max(0.0, vals[3] - vals[1])
        if area > best_area:
            best_area = area
            best_box = (vals[0], vals[1], vals[2], vals[3])
    if best_box is None:
        return None
    x0, y0, x1, y1 = best_box
    if x1 <= 1.0 and y1 <= 1.0:
        return (
            max(0.0, min(1.0, x0)),
            max(0.0, min(1.0, y0)),
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
        )
    w = _DETECTION_IMG_WIDTH if x1 > 64.0 or y1 > 64.0 else 48.0
    h = _DETECTION_IMG_HEIGHT if x1 > 64.0 or y1 > 64.0 else 48.0
    return (
        max(0.0, min(1.0, x0 / w)),
        max(0.0, min(1.0, y0 / h)),
        max(0.0, min(1.0, x1 / w)),
        max(0.0, min(1.0, y1 / h)),
    )


def _try_get_selector(tool_scene: OpenEqaToolScene) -> Any | None:
    if not tool_scene.conceptgraph_dir.is_dir():
        return None
    try:
        return build_selector(tool_scene.conceptgraph_dir)
    except Exception:
        return None


def _contact_sheet(
    tool_scene: OpenEqaToolScene, frames: tuple[ViewFrame, ...], out_dir: Path
) -> Path | None:
    if len(frames) < 2:
        return None
    tiles = [(f"frame {frame.frame_id}", frame.image_path) for frame in frames]
    ids_token = "_".join(f"{frame.frame_id:06d}" for frame in frames)
    destination = out_dir / f"{tool_scene.clip_id}_frame_sheet_{ids_token}.jpg"
    try:
        return compose_contact_sheet(tiles, destination)
    except Exception:  # a sheet is a convenience; never fail the tool over it
        return None


def _requested_ids(
    args: ViewFrameArgs,
    *,
    tool_scene: OpenEqaToolScene,
    selector: Any | None,
) -> list[int]:
    ids: list[int] = []
    if args.frame_id is not None:
        ids.append(args.frame_id)
    ids.extend(args.frame_ids)

    scene_ids = tool_scene.scene.rgb_frame_ids
    if args.around_frame_id is not None and scene_ids:
        center = _nearest_frame_id(scene_ids, args.around_frame_id)
        step = max(1, args.step)
        offsets = (-2 * step, -step, 0, step)
        for offset in offsets:
            cand = _nearest_frame_id(scene_ids, center + offset)
            ids.append(cand)

    if args.grid_cells and selector is not None and scene_ids:
        stride = max(1, int(getattr(selector, "stride", 1) or 1))
        grid_to_views = getattr(selector, "grid_to_views", {}) or {}
        scored_views: dict[int, float] = {}
        for r, c in args.grid_cells:
            for vid, score in grid_to_views.get((int(r), int(c)), []):
                scored_views[int(vid)] = scored_views.get(int(vid), 0.0) + float(score)
        for vid, _ in sorted(scored_views.items(), key=lambda kv: (-kv[1], kv[0]))[
            :_MAX_FRAMES
        ]:
            ids.append(_nearest_frame_id(scene_ids, vid * stride))

    if not ids:
        raise ToolInputError(
            'view_frame needs a frame to show: pass {"frame_id": 120}, '
            '{"frame_ids": [60, 120, 240]}, {"around_frame_id": 120, "step": 10}, '
            'or {"grid_cells": [[2, 3]]}'
        )
    deduped: list[int] = []
    seen: set[int] = set()
    for frame_id in ids:
        if frame_id not in seen:
            seen.add(frame_id)
            deduped.append(frame_id)
    return deduped


def _nearest_frame_id(frame_ids: tuple[int, ...], target: int) -> int:
    return min(frame_ids, key=lambda fid: abs(fid - target))


__all__ = [
    "ViewFrameArgs",
    "ViewFrame",
    "ViewFrameResult",
    "annotate_frame_from_selector",
    "view_frame",
]
