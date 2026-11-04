"""Scene-object inventory & detail (text) tool: ``list_objects``.

Lists and searches the ConceptGraph objects detected in the clip — their ids,
categories, 3D centers/sizes, primary BEV ``grid_cell``, top-visibility
``best_frame_ids``, colors, spatial location notes, nearby objects, usability
(affordances), and appearance descriptions.

Supports filtering by:
* ``category`` — substring match on object category (with fallback to full-text
  enrichment fields so synonyms or attribute terms still surface matches).
* ``query`` — multi-token keyword search across category, description, color,
  location, usability, and nearby objects.
* ``object_ids`` — inspect specific object ids with full, untruncated text.
* ``grid_cells`` — filter to objects inside specific ``[row, col]`` BEV cells.

Needs the ``conceptgraph/`` assets; imported lazily by the dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import ToolInputError
from .scene_context import OpenEqaToolScene, build_selector

_NOTE_MAX_CHARS = 220
_LOC_MAX_CHARS = 180
_USABILITY_MAX_CHARS = 160
_FULL_MAX_CHARS = 600
_MAX_BEST_FRAMES = 4
_MAX_NEARBY = 6


class ListObjectsArgs(BaseModel):
    """Arguments for :func:`list_objects`."""

    model_config = ConfigDict(extra="forbid")

    category: str = ""
    query: str = ""
    object_ids: list[int] = Field(default_factory=list)
    grid_cells: list[tuple[int, int]] = Field(default_factory=list)
    limit: int = Field(default=50, ge=1, le=500)


@dataclass(frozen=True)
class ObjectEntry:
    """One scene object's compact, JSON-ready record."""

    obj_id: int
    category: str
    center: tuple[float, float, float]
    size: tuple[float, float, float] | None
    note: str
    grid_cell: tuple[int, int] | None = None
    best_frame_ids: tuple[int, ...] = ()
    color: tuple[str, ...] = ()
    location: str = ""
    nearby_objects: tuple[str, ...] = ()
    usability: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "obj_id": self.obj_id,
            "category": self.category,
            "center": [round(v, 3) for v in self.center],
        }
        if self.size is not None:
            payload["size"] = [round(v, 3) for v in self.size]
        if self.grid_cell is not None:
            payload["grid_cell"] = list(self.grid_cell)
        if self.best_frame_ids:
            payload["best_frame_ids"] = list(self.best_frame_ids)
        if self.color:
            payload["color"] = list(self.color)
        if self.note:
            payload["note"] = self.note
        if self.location:
            payload["location"] = self.location
        if self.nearby_objects:
            payload["nearby_objects"] = list(self.nearby_objects)
        if self.usability:
            payload["usability"] = self.usability
        return payload


@dataclass(frozen=True)
class ListObjectsResult:
    """A filtered slice of the scene's object inventory."""

    objects: tuple[ObjectEntry, ...]
    count: int
    total: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total": self.total,
            "objects": [entry.to_payload() for entry in self.objects],
        }


def list_objects(
    tool_scene: OpenEqaToolScene, args: ListObjectsArgs
) -> ListObjectsResult:
    """List or search scene objects with rich geometric, frame, and semantic metadata."""
    conceptgraph_dir = tool_scene.require_conceptgraph()
    selector = build_selector(conceptgraph_dir)

    stride = max(1, int(getattr(selector, "stride", 1) or 1))
    object_to_views: dict[int, list[tuple[int, float]]] = (
        getattr(selector, "object_to_views", {}) or {}
    )
    object_to_grids: dict[int, list[tuple[int, int]]] = (
        getattr(selector, "object_to_grids", {}) or {}
    )
    available_frames = tool_scene.scene.rgb_frame_ids
    available_set = set(available_frames)

    wanted_ids = {int(oid) for oid in args.object_ids}
    wanted_grids = {(int(r), int(c)) for r, c in args.grid_cells}
    wanted_cat = args.category.strip().lower()
    wanted_query = args.query.strip().lower()
    full_detail = bool(wanted_ids)

    all_valid_objects = [obj for obj in selector.objects if obj.centroid is not None]
    if wanted_ids:
        known_ids = {obj.obj_id for obj in all_valid_objects}
        missing = sorted(wanted_ids - known_ids)
        if missing:
            raise ToolInputError(
                f"object_ids {missing} not in this scene; "
                "call list_objects with no filters to see valid ids"
            )

    matched: list[tuple[float, Any]] = []
    for obj in all_valid_objects:
        if wanted_ids and obj.obj_id not in wanted_ids:
            continue
        if wanted_grids:
            obj_grids = set(object_to_grids.get(obj.obj_id, []))
            if not (obj_grids & wanted_grids):
                continue
        score = _match_score(
            obj,
            wanted_cat=wanted_cat,
            wanted_query=wanted_query,
            allow_cat_fallback=False,
        )
        if score <= 0.0:
            continue
        matched.append((score, obj))

    if wanted_cat and not matched:
        for obj in all_valid_objects:
            if wanted_ids and obj.obj_id not in wanted_ids:
                continue
            if wanted_grids:
                obj_grids = set(object_to_grids.get(obj.obj_id, []))
                if not (obj_grids & wanted_grids):
                    continue
            score = _match_score(
                obj,
                wanted_cat=wanted_cat,
                wanted_query=wanted_query,
                allow_cat_fallback=True,
            )
            if score > 0.0:
                matched.append((score, obj))

    if (wanted_cat or wanted_query) and not matched:
        label = args.category or args.query
        raise ToolInputError(
            f"no objects match category/query {label!r}; "
            "call list_objects with no category to see what is in the scene, "
            "or inspect frames directly with keyframe_selector / view_frame "
            "(small or wall-mounted items may not be in the 3D object catalog)"
        )

    # Sort by match relevance descending, then obj_id ascending
    matched.sort(key=lambda item: (-item[0], item[1].obj_id))

    entries: list[ObjectEntry] = []
    for _, obj in matched:
        assert obj.centroid is not None
        category = obj.object_tag or obj.category or "object"
        cells = object_to_grids.get(obj.obj_id, [])
        primary_cell = (int(cells[0][0]), int(cells[0][1])) if cells else None
        best_frames = _resolve_best_frames(
            object_to_views.get(obj.obj_id, []),
            stride=stride,
            available_frames=available_frames,
            available_set=available_set,
        )
        raw_colors = getattr(obj, "color", []) or []
        raw_nearby = getattr(obj, "co_objects", []) or []
        entries.append(
            ObjectEntry(
                obj_id=obj.obj_id,
                category=category,
                center=(
                    float(obj.centroid[0]),
                    float(obj.centroid[1]),
                    float(obj.centroid[2]),
                ),
                size=_extent(obj),
                grid_cell=primary_cell,
                best_frame_ids=best_frames,
                color=tuple(str(c) for c in raw_colors[:4] if str(c).strip()),
                note=_truncate(
                    obj.summary,
                    _FULL_MAX_CHARS if full_detail else _NOTE_MAX_CHARS,
                ),
                location=_truncate(
                    getattr(obj, "location", ""),
                    _FULL_MAX_CHARS if full_detail else _LOC_MAX_CHARS,
                ),
                nearby_objects=tuple(
                    str(n) for n in raw_nearby[:_MAX_NEARBY] if str(n).strip()
                ),
                usability=_truncate(
                    getattr(obj, "affordance_category", ""),
                    _FULL_MAX_CHARS if full_detail else _USABILITY_MAX_CHARS,
                ),
            )
        )

    total = len(entries)
    return ListObjectsResult(
        objects=tuple(entries[: args.limit]),
        count=min(total, args.limit),
        total=total,
    )


def _match_score(
    obj: Any,
    *,
    wanted_cat: str,
    wanted_query: str,
    allow_cat_fallback: bool = False,
) -> float:
    if not wanted_cat and not wanted_query:
        return 1.0
    category = (obj.object_tag or obj.category or "object").lower()
    class_names = " ".join(
        str(c).lower() for c in (getattr(obj, "class_name", []) or [])
    )
    summary = str(getattr(obj, "summary", "") or "").lower()
    location = str(getattr(obj, "location", "") or "").lower()
    usability = str(getattr(obj, "affordance_category", "") or "").lower()
    colors = " ".join(str(c).lower() for c in (getattr(obj, "color", []) or []))
    nearby = " ".join(str(n).lower() for n in (getattr(obj, "co_objects", []) or []))
    self_blob = " ".join((category, class_names, summary, usability, colors))
    full_blob = " ".join((self_blob, location, nearby))

    score = 0.0
    if wanted_cat:
        if wanted_cat == category:
            score += 4.0
        elif wanted_cat in category or category in wanted_cat:
            score += 3.0
        elif wanted_cat in class_names:
            score += 2.0
        elif allow_cat_fallback and wanted_cat in self_blob:
            score += 1.0
        else:
            return 0.0

    if wanted_query:
        tokens = [
            tok
            for tok in wanted_query.replace(",", " ").split()
            if len(tok) >= 2 and tok not in {"the", "and", "for", "with", "near"}
        ]
        if not tokens:
            tokens = [wanted_query]
        hits = 0
        for tok in tokens:
            if tok in category:
                score += 3.0
                hits += 1
            elif tok in colors or tok in summary:
                score += 2.0
                hits += 1
            elif tok in full_blob:
                score += 1.0
                hits += 1
        if hits == 0:
            return 0.0
    return max(score, 0.1)


def _resolve_best_frames(
    views: list[tuple[int, float]],
    *,
    stride: int,
    available_frames: tuple[int, ...],
    available_set: set[int],
) -> tuple[int, ...]:
    if not views or not available_frames:
        return ()
    out: list[int] = []
    for view_id, _ in views:
        target = int(view_id) * stride
        fid = (
            target
            if target in available_set
            else min(available_frames, key=lambda f: abs(f - target))
        )
        if fid not in out:
            out.append(fid)
        if len(out) >= _MAX_BEST_FRAMES:
            break
    return tuple(out)


def _extent(obj: Any) -> tuple[float, float, float] | None:
    import numpy as np

    bbox = obj.bbox_np
    if bbox is None:
        return None
    array = np.asarray(bbox, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 3:
        return None
    spans = array[:, :3].max(axis=0) - array[:, :3].min(axis=0)
    return (float(spans[0]), float(spans[1]), float(spans[2]))


def _truncate(value: str, max_chars: int) -> str:
    normalized = " ".join(str(value).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


__all__ = [
    "ListObjectsArgs",
    "ObjectEntry",
    "ListObjectsResult",
    "list_objects",
]
