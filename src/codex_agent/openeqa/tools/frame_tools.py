"""Frame-switching (image) tool: ``view_frame``.

The QA turn only attaches a small uniform sample of first-person frames. When
the answer needs a frame that was not attached (a closer view, a different part
of the room, a specific moment in the trajectory), ``view_frame`` fetches any
raw frame id (or a few) from the clip, downsizes it into a writable scratch dir,
and returns the path(s) so the agent can ``view_image`` the new evidence.

Requires the ``vision`` extra (Pillow), imported lazily by the downsize helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..scene import DEFAULT_MAX_IMAGE_SIZE, downsize_rgb_for_view
from .models import ToolInputError
from .scene_context import OpenEqaToolScene

#: Most frames returned by a single call (keeps the turn's image budget bounded).
_MAX_FRAMES = 4


class ViewFrameArgs(BaseModel):
    """Arguments for :func:`view_frame`."""

    model_config = ConfigDict(extra="forbid")

    frame_id: int | None = None
    frame_ids: list[int] = Field(default_factory=list)
    max_image_size: int = Field(default=DEFAULT_MAX_IMAGE_SIZE, ge=64, le=2048)


@dataclass(frozen=True)
class ViewFrame:
    """One fetched, downsized first-person frame."""

    frame_id: int
    image_path: Path

    def to_payload(self) -> dict[str, Any]:
        return {"frame_id": self.frame_id, "image_path": str(self.image_path)}


@dataclass(frozen=True)
class ViewFrameResult:
    """Fetched frames plus the clip's available frame range."""

    frames: tuple[ViewFrame, ...]
    total_frames: int
    frame_id_range: tuple[int, int]
    note: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "frames": [frame.to_payload() for frame in self.frames],
            "total_frames": self.total_frames,
            "frame_id_range": list(self.frame_id_range),
            "note": self.note,
        }


def view_frame(
    tool_scene: OpenEqaToolScene, args: ViewFrameArgs, *, out_dir: Path
) -> ViewFrameResult:
    """Fetch one or more raw frames by id, downsized for viewing."""
    requested = _requested_ids(args)
    scene = tool_scene.scene
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
        else f"(showing {_MAX_FRAMES} of " f"{len(requested)} requested)"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = tuple(
        ViewFrame(
            frame_id=frame_id,
            image_path=downsize_rgb_for_view(
                scene.raw_rgb_path(frame_id),
                out_dir / f"{tool_scene.clip_id}_frame_{frame_id:06d}.jpg",
                max_size=args.max_image_size,
            ),
        )
        for frame_id in capped
    )
    return ViewFrameResult(
        frames=frames,
        total_frames=scene.total_frames,
        frame_id_range=(scene.rgb_frame_ids[0], scene.rgb_frame_ids[-1]),
        note=note,
    )


def _requested_ids(args: ViewFrameArgs) -> list[int]:
    ids: list[int] = []
    if args.frame_id is not None:
        ids.append(args.frame_id)
    ids.extend(args.frame_ids)
    if not ids:
        raise ToolInputError(
            'view_frame needs a frame to show: pass {"frame_id": 120} or '
            '{"frame_ids": [60, 120, 240]}'
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
    "view_frame",
]
