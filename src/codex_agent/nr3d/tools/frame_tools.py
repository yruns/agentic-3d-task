"""Frame fetch/inventory (text) tools.

``select_by_proposal`` fetches first-person frames that show a set of proposal
ids — crucially supporting ``require_all`` so the agent can fetch a frame where a
target *and* its anchor are co-visible. ``list_frame_proposals`` lists which
proposals are in a frame, ordered left→right by 2D box center. Neither draws on
the image; the agent pairs them with ``mark_frame_with_bbox`` + ``view_image``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..sample import Nr3dScene
from .models import ToolInputError

_MAX_FRAMES = 3


class SelectByProposalArgs(BaseModel):
    """Arguments for :func:`select_by_proposal`."""

    model_config = ConfigDict(extra="forbid")

    proposal_ids: list[int] = Field(min_length=1)
    require_all: bool = False
    k: int = Field(default=3, ge=1)


class ListFrameProposalsArgs(BaseModel):
    """Arguments for :func:`list_frame_proposals`."""

    model_config = ConfigDict(extra="forbid")

    frame_id: int


@dataclass(frozen=True)
class SelectedFrame:
    """One fetched first-person frame with its visible proposals + pose."""

    frame_id: int
    visible_proposal_ids: tuple[int, ...]
    image_path: Path
    bev_xy: tuple[float, float] | None
    camera_yaw: float | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "visible_proposal_ids": list(self.visible_proposal_ids),
            "image_path": str(self.image_path),
            "bev_xy": list(self.bev_xy) if self.bev_xy is not None else None,
            "camera_yaw": self.camera_yaw,
        }


@dataclass(frozen=True)
class SelectByProposalResult:
    """A small set of frames showing the requested proposals."""

    frames: tuple[SelectedFrame, ...]
    note: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "note": self.note,
            "frames": [frame.to_payload() for frame in self.frames],
        }


@dataclass(frozen=True)
class FrameInventoryResult:
    """All proposals visible in one frame, ordered left→right."""

    frame_id: int
    visible_proposal_ids: tuple[int, ...]
    left_to_right: tuple[str, ...]
    categories: dict[int, str]
    boxes_2d: dict[int, list[int]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "visible_proposal_ids": list(self.visible_proposal_ids),
            "left_to_right": list(self.left_to_right),
            "categories": self.categories,
            "boxes_2d": self.boxes_2d,
        }


def select_by_proposal(
    scene: Nr3dScene, args: SelectByProposalArgs
) -> SelectByProposalResult:
    """Fetch up to ``k`` frames that show the given proposal ids."""
    pool = scene.proposal_pool
    missing = sorted({pid for pid in args.proposal_ids if pool.get(pid) is None})
    if missing:
        raise ToolInputError(f"proposal ids not in pool: {missing}")
    frame_sets = [set(pool.require(pid).frame_views) for pid in args.proposal_ids]
    if args.require_all:
        frame_ids = set.intersection(*frame_sets)
    else:
        frame_ids = set().union(*frame_sets)

    capped_k, note = _cap_k(args.k)
    if args.require_all and not frame_ids:
        note = (
            f"{note} no single frame shows all of {args.proposal_ids}; "
            "retry with require_all=false to get per-id frames"
        ).strip()
    chosen = sorted(frame_ids)[:capped_k]
    frames = tuple(
        frame
        for frame_id in chosen
        if (frame := _build_selected_frame(scene, frame_id)) is not None
    )
    return SelectByProposalResult(frames=frames, note=note)


def list_frame_proposals(
    scene: Nr3dScene, args: ListFrameProposalsArgs
) -> FrameInventoryResult:
    """List the proposals visible in one frame, ordered left→right."""
    frame_to_ids = scene.proposal_pool.frame_to_proposal_ids()
    if args.frame_id not in frame_to_ids:
        available = sorted(frame_to_ids)[:20]
        raise ToolInputError(
            f"frame_id={args.frame_id} has no catalog proposals; "
            f"available frame_ids[:20]={available}"
        )
    ordered = _proposals_left_to_right(scene, args.frame_id)
    return FrameInventoryResult(
        frame_id=args.frame_id,
        visible_proposal_ids=tuple(pid for pid, _, _ in ordered),
        left_to_right=tuple(f"#{pid} {category}" for pid, category, _ in ordered),
        categories={pid: category for pid, category, _ in ordered},
        boxes_2d={pid: list(box) for pid, _, box in ordered},
    )


def _proposals_left_to_right(
    scene: Nr3dScene, frame_id: int
) -> list[tuple[int, str, tuple[int, int, int, int]]]:
    rows: list[tuple[float, int, str, tuple[int, int, int, int]]] = []
    for proposal in scene.proposal_pool.proposals:
        view = proposal.frame_views.get(frame_id)
        if view is None:
            continue
        rows.append(
            (view.center_x, proposal.proposal_id, proposal.category, view.bbox_2d)
        )
    rows.sort(key=lambda row: (row[0], row[1]))
    return [(pid, category, box) for _, pid, category, box in rows]


def _build_selected_frame(scene: Nr3dScene, frame_id: int) -> SelectedFrame | None:
    visible = scene.proposal_pool.frame_to_proposal_ids().get(frame_id, [])
    image_path = _raw_path_for_frame(scene, frame_id)
    if image_path is None:
        return None
    pose = (
        scene.camera_trajectory.pose(frame_id)
        if scene.camera_trajectory is not None
        else None
    )
    bev_xy = (pose[0], pose[1]) if pose is not None else None
    camera_yaw = pose[2] if pose is not None else None
    return SelectedFrame(
        frame_id=frame_id,
        visible_proposal_ids=tuple(visible),
        image_path=image_path,
        bev_xy=bev_xy,
        camera_yaw=camera_yaw,
    )


def _raw_path_for_frame(scene: Nr3dScene, frame_id: int) -> Path | None:
    for proposal in scene.proposal_pool.proposals:
        view = proposal.frame_views.get(frame_id)
        if view is not None:
            return scene.resolve_raw_rgb(view)
    return None


def _cap_k(k: int) -> tuple[int, str]:
    if k <= _MAX_FRAMES:
        return k, ""
    return _MAX_FRAMES, f"(k capped at {_MAX_FRAMES} from {k})"


__all__ = [
    "SelectByProposalArgs",
    "ListFrameProposalsArgs",
    "SelectedFrame",
    "SelectByProposalResult",
    "FrameInventoryResult",
    "select_by_proposal",
    "list_frame_proposals",
]
