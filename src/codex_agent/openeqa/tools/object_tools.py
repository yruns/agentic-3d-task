"""Scene-object inventory (text) tool: ``list_objects``.

Lists the ConceptGraph objects detected in the clip — their ids, categories, 3D
centers/sizes, and enrichment descriptions — so the agent can ground the BEV
(``view_bev`` highlights take these ids), phrase ``keyframe_selector`` queries,
and reason about scene layout from text before fetching pixels. The ids match
the ones the BEV and keyframe tools use (the selector's load order).

Needs the ``conceptgraph/`` assets; imported lazily by the dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import ToolInputError
from .scene_context import OpenEqaToolScene, build_selector

_NOTE_MAX_CHARS = 160


class ListObjectsArgs(BaseModel):
    """Arguments for :func:`list_objects`."""

    model_config = ConfigDict(extra="forbid")

    category: str = ""
    limit: int = Field(default=40, ge=1, le=500)


@dataclass(frozen=True)
class ObjectEntry:
    """One scene object's compact, JSON-ready record."""

    obj_id: int
    category: str
    center: tuple[float, float, float]
    size: tuple[float, float, float] | None
    note: str

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "obj_id": self.obj_id,
            "category": self.category,
            "center": [round(v, 3) for v in self.center],
        }
        if self.size is not None:
            payload["size"] = [round(v, 3) for v in self.size]
        if self.note:
            payload["note"] = self.note
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
    """List scene objects (optionally filtered by category substring)."""
    conceptgraph_dir = tool_scene.require_conceptgraph()
    selector = build_selector(conceptgraph_dir)

    wanted = args.category.strip().lower()
    entries: list[ObjectEntry] = []
    for obj in selector.objects:
        if obj.centroid is None:
            continue
        category = obj.object_tag or obj.category or "object"
        if wanted and wanted not in category.lower():
            continue
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
                note=_truncate(obj.summary, _NOTE_MAX_CHARS),
            )
        )
    total = len(entries)
    if wanted and not entries:
        raise ToolInputError(
            f"no objects match category {args.category!r}; "
            "call list_objects with no category to see what is in the scene"
        )
    return ListObjectsResult(
        objects=tuple(entries[: args.limit]), count=min(total, args.limit), total=total
    )


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
