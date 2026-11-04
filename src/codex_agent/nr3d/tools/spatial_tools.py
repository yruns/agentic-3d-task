"""Spatial-verdict (text) tools: deterministic 3D + co-viewed 2D + virtual-view ranking.

Metric relations (``closest_to`` / ``near`` / ``next_to`` / ``farthest_from`` /
``above`` / ``below`` / ``between`` / ``opposite``) are decided from 3D box
centers. View-dependent relations (``left_of`` / ``right_of`` / ``in_front_of``
/ ``behind``) combine co-visible first-person frame votes (2D horizontal offset
and camera-to-object depth) with a 3D virtual-observer projection (standing at
``observer_id`` or the room center facing the anchor/candidate group) so
non-co-visible or hypothetical viewpoint queries ("facing the monitors, the one
on the right", "looking from the door, the front right table") also resolve
deterministically.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..proposals import Proposal
from ..sample import Nr3dScene
from .models import ToolInputError

SPATIAL_RELATIONS: tuple[str, ...] = (
    "closest_to",
    "near",
    "next_to",
    "farthest_from",
    "above",
    "below",
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "between",
    "opposite",
)
MULTI_ANCHOR_SPATIAL_RELATIONS: tuple[str, ...] = (
    "closest_to",
    "near",
    "next_to",
    "farthest_from",
    "between",
    "above",
    "below",
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
)
_RELATION_ALIASES: dict[str, str] = {
    "closer_to": "closest_to",
    "closest": "closest_to",
    "nearest": "closest_to",
    "nearest_to": "closest_to",
    "nearer_to": "closest_to",
    "farther_from": "farthest_from",
    "further_from": "farthest_from",
    "furthest_from": "farthest_from",
    "furthest": "farthest_from",
    "farthest": "farthest_from",
    "far_from": "farthest_from",
    "near_to": "near",
    "nearby": "near",
    "beside": "next_to",
    "adjacent": "next_to",
    "adjacent_to": "next_to",
    "left": "left_of",
    "right": "right_of",
    "to_the_left_of": "left_of",
    "to_the_right_of": "right_of",
    "on_the_left": "left_of",
    "on_the_right": "right_of",
    "front": "in_front_of",
    "front_of": "in_front_of",
    "in_front": "in_front_of",
    "back": "behind",
    "back_of": "behind",
    "in_back_of": "behind",
    "behind_of": "behind",
    "middle": "between",
    "in_between": "between",
    "center": "between",
    "across": "opposite",
    "across_from": "opposite",
    "opposite_to": "opposite",
    "opposite_of": "opposite",
    "over": "above",
    "on_top_of": "above",
    "higher_than": "above",
    "under": "below",
    "underneath": "below",
    "beneath": "below",
    "lower_than": "below",
}


def normalize_relation(relation: str) -> str:
    """Lowercase + underscore-normalize a relation string."""
    return str(relation or "").strip().lower().replace("-", "_").replace(" ", "_")


def canonical_spatial_relation(relation: str) -> str:
    """Map a relation (with synonyms) to its canonical supported form."""
    norm = normalize_relation(relation)
    return _RELATION_ALIASES.get(norm, norm)


class CompareProposalsSpatialArgs(BaseModel):
    """Arguments for :func:`compare_proposals_spatial`."""

    model_config = ConfigDict(extra="forbid")

    candidate_ids: list[int]
    anchor_id: int | None = None
    observer_id: int | None = None
    relation: str


class CompareCandidatesToAnchorsArgs(BaseModel):
    """Arguments for :func:`compare_candidates_to_anchors`."""

    model_config = ConfigDict(extra="forbid")

    candidate_ids: list[int]
    anchor_ids: list[int]
    observer_id: int | None = None
    relation: str


@dataclass(frozen=True)
class _CoviewVotes:
    """2D left/right and front/behind evidence from co-visible frames."""

    shared_frame_count: int
    mean_offset_x: float | None
    left_frame_count: int
    right_frame_count: int
    front_frame_count: int = 0
    behind_frame_count: int = 0


@dataclass(frozen=True)
class _CandidateScore:
    """Per-candidate spatial metrics against a single anchor or group center."""

    proposal_id: int
    distance: float
    horizontal_distance: float
    vertical_offset: float
    horizontal_offset: float
    coview: _CoviewVotes
    facing_right_offset: float = 0.0
    facing_depth_offset: float = 0.0
    between_score: float = 0.0
    is_self_anchor: bool = False


@dataclass(frozen=True)
class CompareSpatialResult:
    """Ranking of candidates against one anchor (or group) for a relation."""

    candidate_ids: tuple[int, ...]
    anchor_id: int | None
    relation: str
    requested_relation: str
    scored: tuple[_CandidateScore, ...]
    observer_id: int | None = None

    def to_payload(self) -> dict[str, Any]:
        non_self = [s for s in self.scored if not s.is_self_anchor] or list(self.scored)
        facing_ltr = [
            s.proposal_id for s in sorted(non_self, key=lambda s: s.facing_right_offset)
        ]
        facing_ftb = [
            s.proposal_id for s in sorted(non_self, key=lambda s: s.facing_depth_offset)
        ]
        payload: dict[str, Any] = {
            "candidate_ids": list(self.candidate_ids),
            "anchor_id": self.anchor_id,
            "relation": self.relation,
            "requested_relation": self.requested_relation,
            "ranked_ids": [s.proposal_id for s in self.scored],
            "distances": [round(s.distance, 4) for s in self.scored],
            "horizontal_distances": [
                round(s.horizontal_distance, 4) for s in self.scored
            ],
            "vertical_offsets": [round(s.vertical_offset, 4) for s in self.scored],
            "x_offsets": [round(s.horizontal_offset, 4) for s in self.scored],
            "shared_frame_counts": [s.coview.shared_frame_count for s in self.scored],
            "mean_2d_center_offsets_x": [
                (
                    round(s.coview.mean_offset_x, 2)
                    if s.coview.mean_offset_x is not None
                    else None
                )
                for s in self.scored
            ],
            "supporting_frame_counts": [
                _supporting_frame_count(s.coview, self.relation) for s in self.scored
            ],
            "contradicting_frame_counts": [
                _contradicting_frame_count(s.coview, self.relation) for s in self.scored
            ],
            "facing_left_to_right_ids": facing_ltr,
            "facing_front_to_back_ids": facing_ftb,
            "facing_right_offsets_m": [
                round(s.facing_right_offset, 3) for s in self.scored
            ],
            "facing_depth_offsets_m": [
                round(s.facing_depth_offset, 3) for s in self.scored
            ],
        }
        if self.observer_id is not None:
            payload["observer_id"] = self.observer_id
        return payload


@dataclass(frozen=True)
class _AnchorRanking:
    """Ranking of the candidate set against one of several anchors."""

    anchor_id: int
    scored: tuple[_CandidateScore, ...]


@dataclass(frozen=True)
class CompareMultiAnchorResult:
    """Per-anchor rankings plus a cross-anchor consistency / between verdict."""

    candidate_ids: tuple[int, ...]
    anchor_ids: tuple[int, ...]
    relation: str
    requested_relation: str
    per_anchor: tuple[_AnchorRanking, ...]
    joint_ranked_ids: tuple[int, ...] = ()
    between_distances: dict[int, float] | None = None

    def to_payload(self) -> dict[str, Any]:
        per_anchor_rows: list[dict[str, Any]] = []
        top1_by_anchor: dict[int, int] = {}
        for ranking in self.per_anchor:
            ranked_ids = [s.proposal_id for s in ranking.scored]
            if ranked_ids:
                top1_by_anchor[ranking.anchor_id] = ranked_ids[0]
            per_anchor_rows.append(
                {
                    "anchor_id": ranking.anchor_id,
                    "ranked_ids": ranked_ids,
                    "distances": [round(s.distance, 4) for s in ranking.scored],
                    "horizontal_distances": [
                        round(s.horizontal_distance, 4) for s in ranking.scored
                    ],
                }
            )
        top1s = list(top1_by_anchor.values())
        if self.joint_ranked_ids:
            globally_consistent: int | None = self.joint_ranked_ids[0]
            disagreement = False
        else:
            globally_consistent = top1s[0] if top1s and len(set(top1s)) == 1 else None
            disagreement = len(set(top1s)) > 1
        payload: dict[str, Any] = {
            "candidate_ids": list(self.candidate_ids),
            "anchor_ids": list(self.anchor_ids),
            "relation": self.relation,
            "requested_relation": self.requested_relation,
            "per_anchor": per_anchor_rows,
            "top1_by_anchor": top1_by_anchor,
            "anchor_disagreement": disagreement,
            "globally_consistent_top1": globally_consistent,
        }
        if self.joint_ranked_ids:
            payload["joint_ranked_ids"] = list(self.joint_ranked_ids)
        if self.between_distances is not None:
            payload["between_segment_distances"] = {
                k: round(v, 4) for k, v in self.between_distances.items()
            }
        return payload


def compare_proposals_spatial(
    scene: Nr3dScene, args: CompareProposalsSpatialArgs
) -> CompareSpatialResult:
    """Rank ``candidate_ids`` against ``anchor_id`` (or group) for a spatial relation."""
    relation = _require_relation(args.relation, allowed=SPATIAL_RELATIONS)
    candidates = _require_candidates(scene, args.candidate_ids)
    anchor = (
        _require_proposal(scene, args.anchor_id, role="anchor_id")
        if args.anchor_id is not None
        else None
    )
    observer = (
        _require_proposal(scene, args.observer_id, role="observer_id")
        if args.observer_id is not None
        else None
    )
    scored_raw = _score_candidates(scene, candidates, anchor=anchor, observer=observer)
    scored = _rank(scored_raw, relation)
    return CompareSpatialResult(
        candidate_ids=tuple(args.candidate_ids),
        anchor_id=args.anchor_id,
        relation=relation,
        requested_relation=args.relation,
        scored=scored,
        observer_id=args.observer_id,
    )


def compare_candidates_to_anchors(
    scene: Nr3dScene, args: CompareCandidatesToAnchorsArgs
) -> CompareMultiAnchorResult:
    """Rank candidates against several plausible anchors and check agreement."""
    relation = _require_relation(args.relation, allowed=MULTI_ANCHOR_SPATIAL_RELATIONS)
    candidates = _require_candidates(scene, args.candidate_ids)
    if not args.anchor_ids:
        raise ToolInputError("anchor_ids must contain at least one proposal id")
    anchors = [
        _require_proposal(scene, anchor_id, role="anchor_ids")
        for anchor_id in args.anchor_ids
    ]
    observer = (
        _require_proposal(scene, args.observer_id, role="observer_id")
        if args.observer_id is not None
        else None
    )
    rankings = tuple(
        _AnchorRanking(
            anchor_id=anchor.proposal_id,
            scored=_rank(
                _score_candidates(scene, candidates, anchor=anchor, observer=observer),
                relation,
            ),
        )
        for anchor in anchors
    )

    joint_ranked: tuple[int, ...] = ()
    between_dists: dict[int, float] | None = None
    if relation == "between" and len(anchors) >= 2:
        joint_ranked, between_dists = _rank_between_anchors(candidates, anchors)
    elif len(anchors) >= 2 and relation in ("closest_to", "near", "next_to"):
        # Also compute joint sum-of-distances ranking across all anchors
        joint_ranked = _rank_joint_distance(candidates, anchors)

    return CompareMultiAnchorResult(
        candidate_ids=tuple(args.candidate_ids),
        anchor_ids=tuple(args.anchor_ids),
        relation=relation,
        requested_relation=args.relation,
        per_anchor=rankings,
        joint_ranked_ids=joint_ranked,
        between_distances=between_dists,
    )


def _require_relation(relation: str, *, allowed: Sequence[str]) -> str:
    canonical = canonical_spatial_relation(relation)
    if canonical not in allowed:
        raise ToolInputError(
            f"unsupported relation {relation!r}; allowed: {' | '.join(allowed)}"
        )
    return canonical


def _require_proposal(scene: Nr3dScene, proposal_id: int, *, role: str) -> Proposal:
    proposal = scene.proposal_pool.get(proposal_id)
    if proposal is None:
        raise ToolInputError(
            f"{role}={proposal_id} is not in the pool for scene {scene.scene_id}"
        )
    return proposal


def _require_candidates(
    scene: Nr3dScene, candidate_ids: Sequence[int]
) -> list[Proposal]:
    if not candidate_ids:
        raise ToolInputError("candidate_ids must contain at least one proposal id")
    pool = scene.proposal_pool
    missing = sorted({cid for cid in candidate_ids if pool.get(cid) is None})
    if missing:
        raise ToolInputError(f"candidate ids not in pool: {missing}")
    return [pool.require(cid) for cid in candidate_ids]


def _score_candidates(
    scene: Nr3dScene,
    candidates: list[Proposal],
    *,
    anchor: Proposal | None,
    observer: Proposal | None,
) -> tuple[_CandidateScore, ...]:
    # Reference 3D center (anchor if provided, else mean of candidates)
    if anchor is not None:
        ax, ay, az = anchor.position_3d
    else:
        ax = sum(c.position_3d[0] for c in candidates) / len(candidates)
        ay = sum(c.position_3d[1] for c in candidates) / len(candidates)
        az = sum(c.position_3d[2] for c in candidates) / len(candidates)

    # Virtual observer XY position: explicit observer_id if given, else room center
    all_proposals = scene.proposal_pool.proposals
    room_cx = (
        sum(p.position_3d[0] for p in all_proposals) / len(all_proposals)
        if all_proposals
        else 0.0
    )
    room_cy = (
        sum(p.position_3d[1] for p in all_proposals) / len(all_proposals)
        if all_proposals
        else 0.0
    )
    if observer is not None:
        obs_x, obs_y = observer.position_3d[0], observer.position_3d[1]
    else:
        obs_x, obs_y = room_cx, room_cy

    # Look-at XY target: if observer is specified, look at candidate group center;
    # otherwise look from room center toward anchor/candidate centroid
    cand_mx = sum(c.position_3d[0] for c in candidates) / len(candidates)
    cand_my = sum(c.position_3d[1] for c in candidates) / len(candidates)
    if (
        observer is not None
        or anchor is None
        or (
            anchor is not None
            and len(candidates) > 1
            and any(c.proposal_id == anchor.proposal_id for c in candidates)
        )
    ):
        look_x, look_y = cand_mx, cand_my
    else:
        look_x, look_y = ax, ay

    fx, fy = look_x - obs_x, look_y - obs_y
    fnorm = math.hypot(fx, fy)
    if fnorm < 1e-6:
        # Fallback if observer == target: look from room center toward candidate mean
        fx, fy = cand_mx - room_cx, cand_my - room_cy
        fnorm = math.hypot(fx, fy)
    if fnorm < 1e-6:
        fx, fy = 1.0, 0.0
    else:
        fx, fy = fx / fnorm, fy / fnorm

    # In right-handed 2D (+X right, +Y up): forward=(fx, fy) -> right=(fy, -fx)
    rx, ry = fy, -fx

    # Group median distance for "between/middle" among candidates
    between_scores = _compute_group_middle_scores(candidates)

    out: list[_CandidateScore] = []
    for candidate in candidates:
        cx, cy, cz = candidate.position_3d
        dx, dy, dz = cx - ax, cy - ay, cz - az
        is_self = (
            anchor is not None
            and candidate.proposal_id == anchor.proposal_id
            and len(candidates) > 1
        )
        coview = (
            _coviewed_votes(scene, candidate, anchor)
            if anchor is not None and not is_self
            else _CoviewVotes(0, None, 0, 0, 0, 0)
        )
        right_off = (cx - look_x) * rx + (cy - look_y) * ry
        depth_off = (cx - look_x) * fx + (cy - look_y) * fy
        out.append(
            _CandidateScore(
                proposal_id=candidate.proposal_id,
                distance=math.sqrt(dx * dx + dy * dy + dz * dz),
                horizontal_distance=math.hypot(dx, dy),
                vertical_offset=dz,
                horizontal_offset=dx,
                coview=coview,
                facing_right_offset=right_off,
                facing_depth_offset=depth_off,
                between_score=between_scores.get(candidate.proposal_id, 0.0),
                is_self_anchor=is_self,
            )
        )
    return tuple(out)


def _compute_group_middle_scores(candidates: list[Proposal]) -> dict[int, float]:
    """Distance from the group centroid (smaller = more 'in the middle')."""
    if len(candidates) <= 2:
        return {c.proposal_id: 0.0 for c in candidates}
    mx = sum(c.position_3d[0] for c in candidates) / len(candidates)
    my = sum(c.position_3d[1] for c in candidates) / len(candidates)
    mz = sum(c.position_3d[2] for c in candidates) / len(candidates)
    return {
        c.proposal_id: math.sqrt(
            (c.position_3d[0] - mx) ** 2
            + (c.position_3d[1] - my) ** 2
            + (c.position_3d[2] - mz) ** 2
        )
        for c in candidates
    }


def _coviewed_votes(
    scene: Nr3dScene, candidate: Proposal, anchor: Proposal
) -> _CoviewVotes:
    shared = sorted(set(candidate.frame_views) & set(anchor.frame_views))
    offsets: list[float] = []
    front_votes = 0
    behind_votes = 0
    traj = scene.camera_trajectory
    cx, cy, _ = candidate.position_3d
    ax, ay, _ = anchor.position_3d

    for frame_id in shared:
        candidate_view = candidate.frame_views[frame_id]
        anchor_view = anchor.frame_views[frame_id]
        offsets.append(candidate_view.center_x - anchor_view.center_x)
        pose = traj.pose(frame_id) if traj is not None else None
        if pose is not None:
            cam_x, cam_y, _ = pose
            d_cand = math.hypot(cx - cam_x, cy - cam_y)
            d_anc = math.hypot(ax - cam_x, ay - cam_y)
            if d_cand + 0.05 < d_anc:
                front_votes += 1
            elif d_cand > d_anc + 0.05:
                behind_votes += 1

    mean_offset = sum(offsets) / len(offsets) if offsets else None
    return _CoviewVotes(
        shared_frame_count=len(offsets),
        mean_offset_x=mean_offset,
        left_frame_count=sum(1 for offset in offsets if offset < 0.0),
        right_frame_count=sum(1 for offset in offsets if offset > 0.0),
        front_frame_count=front_votes,
        behind_frame_count=behind_votes,
    )


def _rank(
    scored: tuple[_CandidateScore, ...], relation: str
) -> tuple[_CandidateScore, ...]:
    if relation == "closest_to":
        return tuple(sorted(scored, key=lambda s: (s.is_self_anchor, s.distance)))
    if relation in ("near", "next_to"):
        return tuple(
            sorted(
                scored,
                key=lambda s: (s.is_self_anchor, s.horizontal_distance, s.distance),
            )
        )
    if relation == "farthest_from":
        return tuple(sorted(scored, key=lambda s: (s.is_self_anchor, -s.distance)))
    if relation == "above":
        return tuple(
            sorted(
                scored,
                key=lambda s: (
                    s.is_self_anchor,
                    s.vertical_offset <= 0.0,
                    s.horizontal_distance,
                    -s.vertical_offset,
                ),
            )
        )
    if relation == "below":
        return tuple(
            sorted(
                scored,
                key=lambda s: (
                    s.is_self_anchor,
                    s.vertical_offset >= 0.0,
                    s.horizontal_distance,
                    s.vertical_offset,
                ),
            )
        )
    if relation == "between":
        return tuple(
            sorted(
                scored,
                key=lambda s: (s.is_self_anchor, s.between_score, s.distance),
            )
        )
    if relation == "opposite":
        # Opposite: across the group/anchor (prefer non-adjacent, facing across)
        return tuple(
            sorted(
                scored,
                key=lambda s: (
                    s.is_self_anchor,
                    abs(s.facing_right_offset),
                    -s.horizontal_distance,
                ),
            )
        )
    if relation in ("in_front_of", "behind"):
        return tuple(sorted(scored, key=lambda s: _front_behind_sort_key(s, relation)))
    direction = "left" if relation == "left_of" else "right"
    return tuple(sorted(scored, key=lambda s: _left_right_sort_key(s, direction)))


def _left_right_sort_key(
    score: _CandidateScore, direction: str
) -> tuple[bool, int, int, int, float, float]:
    coview = score.coview
    relation = "left_of" if direction == "left" else "right_of"
    supporting = _supporting_frame_count(coview, relation)
    contradicting = _contradicting_frame_count(coview, relation)
    # Virtual 3D observer offset: smaller (more negative) = further left;
    # larger (more positive) = further right
    virtual_dir_key = (
        score.facing_right_offset if direction == "left" else -score.facing_right_offset
    )
    return (
        score.is_self_anchor,
        _directional_bucket(coview, relation=relation),
        -supporting,
        contradicting,
        virtual_dir_key,
        score.horizontal_distance,
    )


def _front_behind_sort_key(
    score: _CandidateScore, relation: str
) -> tuple[bool, int, int, int, float, float]:
    coview = score.coview
    supporting = _supporting_frame_count(coview, relation)
    contradicting = _contradicting_frame_count(coview, relation)
    # Virtual 3D observer depth: smaller = closer (in front); larger = further (behind)
    virtual_depth_key = (
        score.facing_depth_offset
        if relation == "in_front_of"
        else -score.facing_depth_offset
    )
    return (
        score.is_self_anchor,
        _directional_bucket(coview, relation=relation),
        -supporting,
        contradicting,
        virtual_depth_key,
        score.horizontal_distance,
    )


def _directional_bucket(coview: _CoviewVotes, *, relation: str) -> int:
    supporting = _supporting_frame_count(coview, relation)
    contradicting = _contradicting_frame_count(coview, relation)
    if supporting > contradicting and supporting > 0:
        return 0
    if coview.shared_frame_count == 0 or (supporting == 0 and contradicting == 0):
        return 1
    return 2


def _supporting_frame_count(coview: _CoviewVotes, relation: str) -> int:
    if relation == "left_of":
        return coview.left_frame_count
    if relation == "right_of":
        return coview.right_frame_count
    if relation == "in_front_of":
        return coview.front_frame_count
    if relation == "behind":
        return coview.behind_frame_count
    return 0


def _contradicting_frame_count(coview: _CoviewVotes, relation: str) -> int:
    if relation == "left_of":
        return coview.right_frame_count
    if relation == "right_of":
        return coview.left_frame_count
    if relation == "in_front_of":
        return coview.behind_frame_count
    if relation == "behind":
        return coview.front_frame_count
    return 0


def _rank_between_anchors(
    candidates: list[Proposal], anchors: list[Proposal]
) -> tuple[tuple[int, ...], dict[int, float]]:
    """Rank candidates by how well they lie on the 3D segment between ``anchors``."""
    # Pick the two furthest anchors as segment endpoints
    best_pair = (anchors[0], anchors[-1])
    max_span = -1.0
    for i, a1 in enumerate(anchors):
        for a2 in anchors[i + 1 :]:
            span = math.dist(a1.position_3d, a2.position_3d)
            if span > max_span:
                max_span = span
                best_pair = (a1, a2)

    p1 = best_pair[0].position_3d
    p2 = best_pair[1].position_3d
    vx, vy, vz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    seg_len_sq = vx * vx + vy * vy + vz * vz
    anchor_id_set = {a.proposal_id for a in anchors}

    rows: list[tuple[bool, int, float, int]] = []
    dist_map: dict[int, float] = {}
    for c in candidates:
        cx, cy, cz = c.position_3d
        is_self = c.proposal_id in anchor_id_set and len(candidates) > len(anchors)
        if seg_len_sq < 1e-8:
            d = math.dist((cx, cy, cz), p1)
            dist_map[c.proposal_id] = d
            rows.append((is_self, 0, d, c.proposal_id))
            continue
        t = ((cx - p1[0]) * vx + (cy - p1[1]) * vy + (cz - p1[2]) * vz) / seg_len_sq
        t_clamped = max(0.0, min(1.0, t))
        proj = (p1[0] + t_clamped * vx, p1[1] + t_clamped * vy, p1[2] + t_clamped * vz)
        perp_dist = math.dist((cx, cy, cz), proj)
        in_segment = 0 if (-0.15 <= t <= 1.15) else 1
        score = perp_dist + 0.25 * abs(t - 0.5) * math.sqrt(seg_len_sq)
        dist_map[c.proposal_id] = perp_dist
        rows.append((is_self, in_segment, score, c.proposal_id))

    rows.sort()
    return tuple(pid for _, _, _, pid in rows), dist_map


def _rank_joint_distance(
    candidates: list[Proposal], anchors: list[Proposal]
) -> tuple[int, ...]:
    anchor_id_set = {a.proposal_id for a in anchors}
    rows: list[tuple[bool, float, int]] = []
    for c in candidates:
        is_self = c.proposal_id in anchor_id_set and len(candidates) > len(anchors)
        total_d = sum(
            math.hypot(
                c.position_3d[0] - a.position_3d[0],
                c.position_3d[1] - a.position_3d[1],
            )
            for a in anchors
        )
        rows.append((is_self, total_d, c.proposal_id))
    rows.sort()
    return tuple(pid for _, _, pid in rows)


__all__ = [
    "SPATIAL_RELATIONS",
    "MULTI_ANCHOR_SPATIAL_RELATIONS",
    "normalize_relation",
    "canonical_spatial_relation",
    "CompareProposalsSpatialArgs",
    "CompareCandidatesToAnchorsArgs",
    "CompareSpatialResult",
    "CompareMultiAnchorResult",
    "compare_proposals_spatial",
    "compare_candidates_to_anchors",
]
