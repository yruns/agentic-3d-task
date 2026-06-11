"""Language->frame retrieval (image) tool: ``keyframe_selector``.

Wraps the production
:meth:`keyframe.keyframe_selector.KeyframeSelector.select_keyframes_v2`
(Stage-1 query parsing + geometric grounding + joint-coverage frame selection)
so the agent can ask, in free-form language, for the first-person frames most
likely to show the objects a question is about — instead of being limited to the
uniform sample attached to the turn.

This is the one tool that needs the heavier ``conceptgraph/`` assets and a
network LLM call for query parsing; failures are reported as recoverable tool
errors so the agent can fall back to ``view_frame``. With ``use_bev`` the
selector also renders the mesh-free schematic BEV and feeds it to the parser as
visual context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..scene import DEFAULT_MAX_IMAGE_SIZE, downsize_rgb_for_view
from .models import ToolInputError
from .scene_context import OpenEqaToolScene, build_selector

#: Cap on frames returned (keeps the turn's image budget bounded).
_MAX_FRAMES = 4
_RGB_FRAME_RE = re.compile(r"(\d+)-rgb\.")


class KeyframeSelectorArgs(BaseModel):
    """Arguments for :func:`keyframe_selector`."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    k: int = Field(default=3, ge=1)
    use_bev: bool = False
    max_image_size: int = Field(default=DEFAULT_MAX_IMAGE_SIZE, ge=64, le=2048)


@dataclass(frozen=True)
class KeyframeHit:
    """One language-retrieved frame, downsized for viewing."""

    frame_id: int
    image_path: Path

    def to_payload(self) -> dict[str, Any]:
        return {"frame_id": self.frame_id, "image_path": str(self.image_path)}


@dataclass(frozen=True)
class KeyframeSelectorResult:
    """Language-retrieved frames plus a short grounding summary."""

    frames: tuple[KeyframeHit, ...]
    hypothesis_summary: str
    note: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "hypothesis_summary": self.hypothesis_summary,
            "note": self.note,
            "frames": [frame.to_payload() for frame in self.frames],
        }


def keyframe_selector(
    tool_scene: OpenEqaToolScene, args: KeyframeSelectorArgs, *, out_dir: Path
) -> KeyframeSelectorResult:
    """Retrieve up to ``k`` first-person frames for a free-form query."""
    conceptgraph_dir = tool_scene.require_conceptgraph()
    capped_k = min(args.k, _MAX_FRAMES)
    note = "" if args.k == capped_k else f"(k capped at {_MAX_FRAMES} from {args.k})"

    selector = build_selector(conceptgraph_dir)
    result = _run_selector(selector, args.query, capped_k, use_bev=args.use_bev)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[KeyframeHit] = []
    seen: set[int] = set()
    for path in result.keyframe_paths[:capped_k]:
        frame_id = _frame_id_from_path(Path(path))
        if frame_id is None or frame_id in seen:
            continue
        seen.add(frame_id)
        frames.append(
            KeyframeHit(
                frame_id=frame_id,
                image_path=downsize_rgb_for_view(
                    Path(path),
                    out_dir / f"{tool_scene.clip_id}_kf_{frame_id:06d}.jpg",
                    max_size=args.max_image_size,
                ),
            )
        )
    return KeyframeSelectorResult(
        frames=tuple(frames),
        hypothesis_summary=_summarize(result),
        note=note or _empty_note(frames),
    )


def _run_selector(selector: Any, query: str, k: int, *, use_bev: bool) -> Any:
    try:
        return selector.select_keyframes_v2(
            query, k=k, viewpoint_aware=True, use_visual_context=use_bev
        )
    except Exception as exc:  # re-raised as a recoverable tool error for the agent
        raise ToolInputError(
            f"keyframe_selector failed ({type(exc).__name__}: {exc}); "
            "fall back to view_frame"
        ) from exc


def _frame_id_from_path(path: Path) -> int | None:
    match = _RGB_FRAME_RE.search(path.name)
    return int(match.group(1)) if match is not None else None


def _summarize(result: Any) -> str:
    target = getattr(result, "target_term", None)
    anchor = getattr(result, "anchor_term", None)
    parts = [f"target={target!r}"]
    if anchor:
        parts.append(f"anchor={anchor!r}")
    return " ".join(parts)


def _empty_note(frames: list[KeyframeHit]) -> str:
    if frames:
        return ""
    return "no frames grounded for this query; try view_frame or rephrase"


__all__ = [
    "KeyframeSelectorArgs",
    "KeyframeHit",
    "KeyframeSelectorResult",
    "keyframe_selector",
]
