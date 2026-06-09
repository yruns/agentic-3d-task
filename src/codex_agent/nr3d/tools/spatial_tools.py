"""Spatial-verdict (text) tools: deterministic 3D + co-viewed 2D ranking.

Metric relations (``closest_to`` / ``near`` / ``next_to`` / ``farthest_from`` /
``above`` / ``below``) are decided from 3D box centers. View-dependent relations
(``left_of`` / ``right_of``) are decided by voting over frames where the
candidate and the anchor are *co-visible*, using their 2D box centers — because
left/right only has meaning from a concrete viewpoint, not from world XYZ.
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
)
MULTI_ANCHOR_SPATIAL_RELATIONS: tuple[str, ...] = (
    "closest_to",
    "near",
    "next_to",
    "farthest_from",
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
    anchor_id: int
    relation: str


class CompareCandidatesToAnchorsArgs(BaseModel):
    """Arguments for :func:`compare_candidates_to_anchors`."""

    model_config = ConfigDict(extra="forbid")

    candidate_ids: list[int]
    anchor_ids: list[int]
    relation: str


@dataclass(frozen=True)
class _CoviewVotes:
    """2D left/right evidence from frames where two proposals co-appear."""

    shared_frame_count: int
    mean_offset_x: float | None
    left_frame_count: int
    right_frame_count: int


@dataclass(frozen=True)
class _CandidateScore:
    """Per-candidate spatial metrics against a single anchor."""

    proposal_id: int
    distance: float
    horizontal_distance: float
    vertical_offset: float
    horizontal_offset: float
    coview: _CoviewVotes


@dataclass(frozen=True)
class CompareSpatialResult:
    """Ranking of candidates against one anchor for a relation."""

    candidate_ids: tuple[int, ...]
    anchor_id: int
    relation: str
    requested_relation: str
    scored: tuple[_CandidateScore, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
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
        }


@dataclass(frozen=True)
class _AnchorRanking:
    """Ranking of the candidate set against one of several anchors."""

    anchor_id: int
    scored: tuple[_CandidateScore, ...]


@dataclass(frozen=True)
class CompareMultiAnchorResult:
    """Per-anchor rankings plus a cross-anchor consistency verdict."""

    candidate_ids: tuple[int, ...]
    anchor_ids: tuple[int, ...]
    relation: str
    requested_relation: str
    per_anchor: tuple[_AnchorRanking, ...]

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
        globally_consistent = top1s[0] if top1s and len(set(top1s)) == 1 else None
        return {
            "candidate_ids": list(self.candidate_ids),
            "anchor_ids": list(self.anchor_ids),
            "relation": self.relation,
            "requested_relation": self.requested_relation,
            "per_anchor": per_anchor_rows,
            "top1_by_anchor": top1_by_anchor,
            "anchor_disagreement": len(set(top1s)) > 1,
            "globally_consistent_top1": globally_consistent,
        }


def compare_proposals_spatial(
    scene: Nr3dScene, args: CompareProposalsSpatialArgs
) -> CompareSpatialResult:
    """Rank ``candidate_ids`` against ``anchor_id`` for a spatial relation."""
    relation = _require_relation(args.relation, allowed=SPATIAL_RELATIONS)
    anchor = _require_proposal(scene, args.anchor_id, role="anchor_id")
    candidates = _require_candidates(scene, args.candidate_ids)
    scored = _rank(
        tuple(_score_candidate(candidate, anchor) for candidate in candidates),
        relation,
    )
    return CompareSpatialResult(
        candidate_ids=tuple(args.candidate_ids),
        anchor_id=args.anchor_id,
        relation=relation,
        requested_relation=args.relation,
        scored=scored,
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
    rankings = tuple(
        _AnchorRanking(
            anchor_id=anchor.proposal_id,
            scored=_rank(
                tuple(_score_candidate(candidate, anchor) for candidate in candidates),
                relation,
            ),
        )
        for anchor in anchors
    )
    return CompareMultiAnchorResult(
        candidate_ids=tuple(args.candidate_ids),
        anchor_ids=tuple(args.anchor_ids),
        relation=relation,
        requested_relation=args.relation,
        per_anchor=rankings,
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


def _score_candidate(candidate: Proposal, anchor: Proposal) -> _CandidateScore:
    cx, cy, cz = candidate.position_3d
    ax, ay, az = anchor.position_3d
    dx, dy, dz = cx - ax, cy - ay, cz - az
    return _CandidateScore(
        proposal_id=candidate.proposal_id,
        distance=math.sqrt(dx * dx + dy * dy + dz * dz),
        horizontal_distance=math.hypot(dx, dy),
        vertical_offset=dz,
        horizontal_offset=dx,
        coview=_coviewed_votes(candidate, anchor),
    )


def _coviewed_votes(candidate: Proposal, anchor: Proposal) -> _CoviewVotes:
    shared = sorted(set(candidate.frame_views) & set(anchor.frame_views))
    offsets: list[float] = []
    for frame_id in shared:
        candidate_view = candidate.frame_views[frame_id]
        anchor_view = anchor.frame_views[frame_id]
        offsets.append(candidate_view.center_x - anchor_view.center_x)
    mean_offset = sum(offsets) / len(offsets) if offsets else None
    return _CoviewVotes(
        shared_frame_count=len(offsets),
        mean_offset_x=mean_offset,
        left_frame_count=sum(1 for offset in offsets if offset < 0.0),
        right_frame_count=sum(1 for offset in offsets if offset > 0.0),
    )


def _rank(
    scored: tuple[_CandidateScore, ...], relation: str
) -> tuple[_CandidateScore, ...]:
    if relation == "closest_to":
        return tuple(sorted(scored, key=lambda s: s.distance))
    if relation in ("near", "next_to"):
        return tuple(sorted(scored, key=lambda s: (s.horizontal_distance, s.distance)))
    if relation == "farthest_from":
        return tuple(sorted(scored, key=lambda s: s.distance, reverse=True))
    if relation == "above":
        return tuple(
            sorted(
                scored,
                key=lambda s: (
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
                    s.vertical_offset >= 0.0,
                    s.horizontal_distance,
                    s.vertical_offset,
                ),
            )
        )
    direction = "left" if relation == "left_of" else "right"
    return tuple(sorted(scored, key=lambda s: _left_right_sort_key(s, direction)))


def _left_right_sort_key(
    score: _CandidateScore, direction: str
) -> tuple[int, int, int, float, float]:
    coview = score.coview
    relation = "left_of" if direction == "left" else "right_of"
    supporting = _supporting_frame_count(coview, relation)
    contradicting = _contradicting_frame_count(coview, relation)
    abs_offset = (
        abs(coview.mean_offset_x) if coview.mean_offset_x is not None else float("inf")
    )
    return (
        _left_right_bucket(coview, direction=direction),
        -supporting,
        contradicting,
        abs_offset,
        score.horizontal_distance,
    )


def _left_right_bucket(coview: _CoviewVotes, *, direction: str) -> int:
    relation = "left_of" if direction == "left" else "right_of"
    supporting = _supporting_frame_count(coview, relation)
    contradicting = _contradicting_frame_count(coview, relation)
    if supporting > contradicting and supporting > 0:
        return 0
    if coview.shared_frame_count == 0:
        return 1
    return 2


def _supporting_frame_count(coview: _CoviewVotes, relation: str) -> int:
    if relation == "left_of":
        return coview.left_frame_count
    if relation == "right_of":
        return coview.right_frame_count
    return 0


def _contradicting_frame_count(coview: _CoviewVotes, relation: str) -> int:
    if relation == "left_of":
        return coview.right_frame_count
    if relation == "right_of":
        return coview.left_frame_count
    return 0


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
