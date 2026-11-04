"""Frame fetch/inventory (text) tools.

``select_by_proposal`` fetches first-person frames that show a set of proposal
ids — supporting ``require_all`` for co-visible target+anchor frames (with
automatic fallback to maximal partial co-visibility when >2 proposals cannot all
fit in one view) and balanced, spatially-diverse coverage when
``require_all=False``. ``list_frame_proposals`` lists which proposals are in a
frame, ordered left→right by 2D box center.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..proposals import FrameView, Proposal
from ..sample import Nr3dScene
from .models import ToolInputError

_MAX_FRAMES = 4
_REF_IMG_W = 1296.0
_REF_IMG_H = 968.0
_MIN_POS_SEPARATION_M = 0.35
_MIN_YAW_SEPARATION_RAD = 0.25
_MIN_FRAME_ID_GAP = 4


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
    requested_visible_ids: tuple[int, ...] = ()
    requested_left_to_right: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frame_id": self.frame_id,
            "visible_proposal_ids": list(self.visible_proposal_ids),
            "image_path": str(self.image_path),
            "bev_xy": list(self.bev_xy) if self.bev_xy is not None else None,
            "camera_yaw": self.camera_yaw,
        }
        if self.requested_visible_ids:
            payload["requested_visible_ids"] = list(self.requested_visible_ids)
        if self.requested_left_to_right:
            payload["requested_left_to_right"] = list(self.requested_left_to_right)
        return payload


@dataclass(frozen=True)
class SelectByProposalResult:
    """A small set of frames showing the requested proposals."""

    frames: tuple[SelectedFrame, ...]
    note: str
    per_proposal_best_frames: dict[int, list[int]] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "note": self.note,
            "frames": [frame.to_payload() for frame in self.frames],
        }
        if self.per_proposal_best_frames:
            payload["per_proposal_best_frames"] = self.per_proposal_best_frames
        return payload


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
    """Fetch up to ``k`` high-quality, diverse frames for ``proposal_ids``."""
    pool = scene.proposal_pool
    requested_ids = list(dict.fromkeys(int(pid) for pid in args.proposal_ids))
    missing = sorted({pid for pid in requested_ids if pool.get(pid) is None})
    if missing:
        raise ToolInputError(f"proposal ids not in pool: {missing}")

    proposals = [pool.require(pid) for pid in requested_ids]
    per_prop_best = {
        p.proposal_id: _best_frames_for_proposal(scene, p, limit=3) for p in proposals
    }
    capped_k, note = _cap_k(args.k)

    if args.require_all:
        chosen, covis_note = _select_covisible_frames(scene, proposals, capped_k)
        if covis_note:
            note = f"{note} {covis_note}".strip()
    else:
        chosen = _select_balanced_frames(scene, proposals, capped_k)

    frames = tuple(
        frame
        for frame_id in chosen
        if (frame := _build_selected_frame(scene, frame_id, requested_ids)) is not None
    )
    return SelectByProposalResult(
        frames=frames,
        note=note,
        per_proposal_best_frames=per_prop_best,
    )


def _view_quality(view: FrameView) -> float:
    x1, y1, x2, y2 = view.bbox_2d
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    area_frac = min(1.0, (w * h) / (_REF_IMG_W * _REF_IMG_H * 0.25))
    clipped = x1 < 12 or y1 < 12 or x2 > (_REF_IMG_W - 12) or y2 > (_REF_IMG_H - 12)
    cx_norm = 0.5 * (x1 + x2) / _REF_IMG_W
    cy_norm = 0.5 * (y1 + y2) / _REF_IMG_H
    center_bonus = max(0.0, 1.0 - math.hypot(cx_norm - 0.5, cy_norm - 0.5))
    base = 0.65 * area_frac + 0.35 * center_bonus
    return base * (0.75 if clipped else 1.0)


def _best_frames_for_proposal(
    scene: Nr3dScene, proposal: Proposal, *, limit: int = 3
) -> list[int]:
    if not proposal.frame_views:
        return list(proposal.visible_frame_ids[:limit])
    scored = sorted(
        (
            (_view_quality(view), -frame_id, frame_id)
            for frame_id, view in proposal.frame_views.items()
        ),
        reverse=True,
    )
    candidates = [fid for _, _, fid in scored]
    return _pick_diverse_frames(scene, candidates, limit)


def _select_covisible_frames(
    scene: Nr3dScene, proposals: list[Proposal], k: int
) -> tuple[list[int], str]:
    frame_sets = [set(p.frame_views) for p in proposals]
    full_inter = set.intersection(*frame_sets) if frame_sets else set()
    if full_inter:
        ranked = sorted(
            full_inter,
            key=lambda fid: (
                -sum(_view_quality(p.frame_views[fid]) for p in proposals),
                fid,
            ),
        )
        return _pick_diverse_frames(scene, ranked, k), ""

    # No single frame shows all N proposals. Check if >2 proposals were requested
    # and some frames co-view at least 2 of them.
    all_frames = set().union(*frame_sets) if frame_sets else set()
    by_count: list[tuple[int, float, int]] = []
    for fid in all_frames:
        vis_props = [p for p in proposals if fid in p.frame_views]
        if len(vis_props) >= 2:
            qual = sum(_view_quality(p.frame_views[fid]) for p in vis_props)
            by_count.append((len(vis_props), qual, -fid))

    req_ids = [p.proposal_id for p in proposals]
    if by_count and len(proposals) > 2:
        by_count.sort(reverse=True)
        max_cnt = by_count[0][0]
        ranked_partial = [-neg_fid for _, _, neg_fid in by_count]
        chosen = _pick_diverse_frames(scene, ranked_partial, k)
        return (
            chosen,
            f"no single frame shows all of {req_ids}; returning frames that "
            f"co-view up to {max_cnt}/{len(proposals)} of them "
            "(see per_proposal_best_frames for each id's top views)",
        )

    return (
        [],
        f"no single frame shows all of {req_ids}; "
        "retry with require_all=false or inspect per_proposal_best_frames",
    )


def _select_balanced_frames(
    scene: Nr3dScene, proposals: list[Proposal], k: int
) -> list[int]:
    all_frames = sorted({fid for p in proposals for fid in p.frame_views})
    if not all_frames:
        return []
    if len(all_frames) <= k:
        return all_frames

    # Precompute per-frame quality per requested proposal
    frame_prop_qual: dict[int, dict[int, float]] = {}
    for fid in all_frames:
        frame_prop_qual[fid] = {
            p.proposal_id: _view_quality(p.frame_views[fid])
            for p in proposals
            if fid in p.frame_views
        }

    picked: list[int] = []
    covered_pids: set[int] = set()
    target_pids = {p.proposal_id for p in proposals}

    for _ in range(k):
        uncovered = target_pids - covered_pids
        best_fid: int | None = None
        best_key: tuple[int, int, float, int, int] | None = None

        for enforce_sep in (True, False):
            for fid in all_frames:
                if fid in picked:
                    continue
                if enforce_sep and not _is_frame_separated(scene, fid, picked):
                    continue
                quals = frame_prop_qual[fid]
                new_cov = sum(1 for pid in quals if pid in uncovered)
                total_cov = len(quals)
                qual_sum = sum(
                    q * (2.0 if pid in uncovered else 1.0) for pid, q in quals.items()
                )
                sep_bonus = 1 if _is_frame_separated(scene, fid, picked) else 0
                key = (new_cov, sep_bonus, round(qual_sum, 6), total_cov, -fid)
                if best_key is None or key > best_key:
                    best_key = key
                    best_fid = fid
            if best_fid is not None:
                break

        if best_fid is None:
            break
        picked.append(best_fid)
        covered_pids.update(frame_prop_qual[best_fid].keys())

    return picked


def _pick_diverse_frames(
    scene: Nr3dScene, ranked_frame_ids: list[int], k: int
) -> list[int]:
    if len(ranked_frame_ids) <= k:
        return list(ranked_frame_ids)
    for enforce_sep in (True, False):
        picked: list[int] = []
        for fid in ranked_frame_ids:
            if fid in picked:
                continue
            if enforce_sep and not _is_frame_separated(scene, fid, picked):
                continue
            picked.append(fid)
            if len(picked) >= k:
                return picked
        if len(picked) >= k:
            return picked
    # Fill any remaining slots from ranked order
    out = list(picked)
    for fid in ranked_frame_ids:
        if fid not in out:
            out.append(fid)
            if len(out) >= k:
                break
    return out


def _is_frame_separated(
    scene: Nr3dScene, candidate_fid: int, picked: list[int]
) -> bool:
    if not picked:
        return True
    traj = scene.camera_trajectory
    cand_pose = traj.pose(candidate_fid) if traj is not None else None
    for other_fid in picked:
        if other_fid == candidate_fid:
            return False
        other_pose = traj.pose(other_fid) if traj is not None else None
        if cand_pose is not None and other_pose is not None:
            dist = math.hypot(
                cand_pose[0] - other_pose[0], cand_pose[1] - other_pose[1]
            )
            dyaw = abs(
                math.atan2(
                    math.sin(cand_pose[2] - other_pose[2]),
                    math.cos(cand_pose[2] - other_pose[2]),
                )
            )
            if dist < _MIN_POS_SEPARATION_M and dyaw < _MIN_YAW_SEPARATION_RAD:
                return False
        else:
            if abs(candidate_fid - other_fid) < _MIN_FRAME_ID_GAP:
                return False
    return True


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


def _build_selected_frame(
    scene: Nr3dScene,
    frame_id: int,
    requested_ids: list[int] | None = None,
) -> SelectedFrame | None:
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

    req_vis: tuple[int, ...] = ()
    req_ltr: tuple[str, ...] = ()
    if requested_ids:
        req_set = set(requested_ids)
        ordered = _proposals_left_to_right(scene, frame_id)
        req_ordered = [(pid, cat) for pid, cat, _ in ordered if pid in req_set]
        req_vis = tuple(pid for pid, _ in req_ordered)
        req_ltr = tuple(f"#{pid} {cat}" for pid, cat in req_ordered)

    return SelectedFrame(
        frame_id=frame_id,
        visible_proposal_ids=tuple(visible),
        image_path=image_path,
        bev_xy=bev_xy,
        camera_yaw=camera_yaw,
        requested_visible_ids=req_vis,
        requested_left_to_right=req_ltr,
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
