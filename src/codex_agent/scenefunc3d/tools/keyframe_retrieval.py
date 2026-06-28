"""Lightweight SceneFunc3D keyframe retrieval tool."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .scene_context import SceneFunc3dToolScene

QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class KeyframeSelectorArgs(BaseModel):
    """Arguments for ``keyframe_selector``."""

    model_config = ConfigDict(extra="forbid")

    query: QueryText
    k: int = Field(default=4, ge=1, le=8, strict=True)


@dataclass(frozen=True)
class KeyframeSelection:
    """One ranked SceneFunc3D frame candidate."""

    frame_id: str
    rank: int

    def to_payload(self) -> dict[str, object]:
        """Return this selection as a JSON-ready mapping."""
        return {"frame_id": self.frame_id, "rank": self.rank}


@dataclass(frozen=True)
class KeyframeSelectorResult:
    """Ranked frame candidates for a query."""

    query: str
    frames: tuple[KeyframeSelection, ...]

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready CLI payload."""
        frame_payloads: list[dict[str, object]] = [
            frame.to_payload() for frame in self.frames
        ]
        return {"query": self.query, "frames": frame_payloads}


def keyframe_selector(
    tool_scene: SceneFunc3dToolScene, args: KeyframeSelectorArgs, *, out_dir: Path
) -> KeyframeSelectorResult:
    """Return the first ``k`` RGB frame ids as a deterministic placeholder."""
    _ = out_dir
    frames = tuple(
        KeyframeSelection(frame_id=frame_id, rank=rank)
        for rank, frame_id in enumerate(tool_scene.rgb_frame_ids[: args.k], start=1)
    )
    return KeyframeSelectorResult(query=args.query, frames=frames)


__all__ = [
    "KeyframeSelection",
    "KeyframeSelectorArgs",
    "KeyframeSelectorResult",
    "keyframe_selector",
]
