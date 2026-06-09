"""Language→frame retrieval (image) tool: ``keyframe_selector``.

Wraps the production :meth:`keyframe.keyframe_selector.KeyframeSelector.select_keyframes_v2`
(Stage-1 query parsing + geometric grounding + joint-coverage frame selection)
and maps the returned frames back onto the prepared catalog's frame ids so the
agent can fetch first-person views for a free-form description when the catalog
shortlist is not obvious.

This is the one tool that needs the heavier ``conceptgraph/`` scene assets and a
network LLM call for query parsing; it is imported lazily by the dispatcher and
reports failures as recoverable tool errors so the agent can fall back to the
catalog-based selectors.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..sample import Nr3dScene
from .frame_tools import SelectedFrame, _build_selected_frame
from .models import ToolInputError

_MAX_FRAMES = 3
_CONCEPTGRAPH_DIRNAME = "conceptgraph"


class KeyframeSelectorArgs(BaseModel):
    """Arguments for :func:`keyframe_selector`."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    k: int = Field(default=3, ge=1)


@dataclass(frozen=True)
class KeyframeSelectorResult:
    """Language-retrieved frames mapped onto the catalog's frame ids."""

    frames: tuple[SelectedFrame, ...]
    hypothesis_summary: str
    note: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "hypothesis_summary": self.hypothesis_summary,
            "note": self.note,
            "frames": [frame.to_payload() for frame in self.frames],
        }


def keyframe_selector(
    scene: Nr3dScene, args: KeyframeSelectorArgs
) -> KeyframeSelectorResult:
    """Retrieve up to ``k`` first-person frames for a free-form query."""
    conceptgraph_dir = scene.scene_dir.parent / _CONCEPTGRAPH_DIRNAME
    if not conceptgraph_dir.is_dir():
        raise ToolInputError(
            "keyframe_selector is unavailable for this scene (no conceptgraph "
            f"assets at {conceptgraph_dir}); use select_by_proposal / "
            "list_scene_proposals instead"
        )
    capped_k = min(args.k, _MAX_FRAMES)
    note = "" if args.k == capped_k else f"(k capped at {_MAX_FRAMES} from {args.k})"

    result = _run_selector(conceptgraph_dir, args.query, capped_k)
    frame_ids = _map_result_to_frame_ids(scene, result)
    frames = tuple(
        frame
        for frame_id in frame_ids[:capped_k]
        if (frame := _build_selected_frame(scene, frame_id)) is not None
    )
    return KeyframeSelectorResult(
        frames=frames,
        hypothesis_summary=_summarize(result),
        note=note,
    )


def _run_selector(conceptgraph_dir: Path, query: str, k: int) -> Any:
    try:
        from keyframe.keyframe_selector import KeyframeSelector

        selector = KeyframeSelector.from_scene_path(conceptgraph_dir, dataset="nr3d")
        return selector.select_keyframes_v2(query, k=k, viewpoint_aware=True)
    except ToolInputError:
        raise
    except Exception as exc:  # re-raised as a recoverable tool error for the agent
        raise ToolInputError(
            f"keyframe_selector failed ({type(exc).__name__}: {exc}); "
            "fall back to select_by_proposal on candidate ids"
        ) from exc


def _map_result_to_frame_ids(scene: Nr3dScene, result: Any) -> list[int]:
    """Map a ``KeyframeResult`` to catalog frame ids.

    ``keyframe_indices`` come from the scene's per-frame visibility index, which
    is keyed by the same frame ids the catalog uses, so they map directly. Only
    when that yields nothing do we fall back to matching the resolved RGB paths'
    basenames against the catalog's per-frame views.
    """
    valid_frames = set(scene.proposal_pool.frame_to_proposal_ids())
    direct = _dedupe_present(
        (int(view_id) for view_id in getattr(result, "keyframe_indices", []) or []),
        valid_frames,
    )
    if direct:
        return direct
    name_to_frame = {
        Path(view.raw_rgb_path).name: view.frame_id
        for proposal in scene.proposal_pool.proposals
        for view in proposal.frame_views.values()
    }
    paths = getattr(result, "keyframe_paths", []) or []
    return _dedupe_present(
        (
            frame_id
            for path in paths
            if (frame_id := name_to_frame.get(Path(path).name)) is not None
        ),
        valid_frames,
    )


def _dedupe_present(frame_ids: Iterable[int], valid_frames: set[int]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for frame_id in frame_ids:
        if frame_id in seen or frame_id not in valid_frames:
            continue
        seen.add(frame_id)
        ordered.append(frame_id)
    return ordered


def _summarize(result: Any) -> str:
    target = getattr(result, "target_term", None)
    anchor = getattr(result, "anchor_term", None)
    parts = [f"target={target!r}"]
    if anchor:
        parts.append(f"anchor={anchor!r}")
    return " ".join(parts)


__all__ = [
    "KeyframeSelectorArgs",
    "KeyframeSelectorResult",
    "keyframe_selector",
]
