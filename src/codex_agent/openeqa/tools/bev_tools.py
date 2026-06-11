"""Global-alignment (image) tool: ``view_bev``.

Renders the mesh-free top-down *schematic* BEV for the clip (object footprints +
camera trajectory) so the agent can read scene layout and tie first-person
frames to a map. With ``highlight``/``categories`` the named objects are drawn in
the highlight color. The render is written into a writable scratch dir and its
path returned for ``view_image``.

OpenEQA clips have no mesh, so this uses
:meth:`keyframe.keyframe_selector.KeyframeSelector.generate_scene_bev` with the
``openeqa`` schematic builder. Needs the ``conceptgraph/`` assets and the
``vision`` extra; imported lazily by the dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import ToolInputError
from .scene_context import OpenEqaToolScene, build_selector

_BEV_RENDERER_VERSION = "v1"


class ViewBevArgs(BaseModel):
    """Arguments for :func:`view_bev`."""

    model_config = ConfigDict(extra="forbid")

    highlight: list[int] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ViewBevResult:
    """A BEV image path plus what was highlighted on it."""

    image_path: Path
    highlight_ids: tuple[int, ...]
    missing_categories: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image_path": str(self.image_path),
            "highlight_ids": list(self.highlight_ids),
            "view": "default" if not self.highlight_ids else "highlighted",
        }
        if self.missing_categories:
            payload["categories_with_no_matches"] = list(self.missing_categories)
        return payload


def view_bev(
    tool_scene: OpenEqaToolScene, args: ViewBevArgs, *, out_dir: Path
) -> ViewBevResult:
    """Render the schematic BEV, optionally highlighting objects by id/category."""
    conceptgraph_dir = tool_scene.require_conceptgraph()
    selector = build_selector(conceptgraph_dir)

    highlight_ids, missing = _resolve_highlight_ids(selector, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    token = "_".join(str(i) for i in highlight_ids) if highlight_ids else "default"
    out_path = out_dir / f"{tool_scene.clip_id}_bev_{token}_{_BEV_RENDERER_VERSION}.png"
    try:
        rendered = selector.generate_scene_bev(
            output_path=out_path,
            use_cache=False,
            highlight_ids=frozenset(highlight_ids),
        )
    except ToolInputError:
        raise
    except Exception as exc:  # re-raised as a recoverable tool error for the agent
        raise ToolInputError(
            f"could not render the BEV for this scene " f"({type(exc).__name__}: {exc})"
        ) from exc
    return ViewBevResult(
        image_path=rendered,
        highlight_ids=tuple(highlight_ids),
        missing_categories=tuple(missing),
    )


def _resolve_highlight_ids(
    selector: Any, args: ViewBevArgs
) -> tuple[list[int], list[str]]:
    by_id = {obj.obj_id: obj for obj in selector.objects}
    missing_ids = sorted({pid for pid in args.highlight if pid not in by_id})
    if missing_ids:
        raise ToolInputError(
            f"highlight ids not in this scene: {missing_ids}; "
            "call list_objects to see valid object ids"
        )
    ids: set[int] = set(args.highlight)

    by_category: dict[str, list[int]] = {}
    for obj in selector.objects:
        category = (obj.object_tag or obj.category or "object").strip().lower()
        by_category.setdefault(category, []).append(obj.obj_id)
    missing_categories: list[str] = []
    for raw in args.categories:
        key = raw.strip().lower()
        if not key:
            continue
        hits = [oid for cat, oids in by_category.items() if key in cat for oid in oids]
        if hits:
            ids.update(hits)
        else:
            missing_categories.append(raw)
    return sorted(ids), missing_categories


__all__ = [
    "ViewBevArgs",
    "ViewBevResult",
    "view_bev",
]
