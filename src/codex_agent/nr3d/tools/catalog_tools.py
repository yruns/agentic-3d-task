"""Catalog (text) tools: narrow the candidate pool without fetching images.

``inspect_proposal`` returns one proposal's full enrichment (color, description,
nearby objects, every frame it appears in) — strictly more than the truncated
note carried in the prompt. ``list_scene_proposals`` filters the pool by
category or BEV region so the agent can build a shortlist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..proposals import Proposal
from ..sample import Nr3dScene
from .models import ToolInputError

_REGION_BEV_LEN = 4


class InspectProposalArgs(BaseModel):
    """Arguments for :func:`inspect_proposal`."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: int


class ListSceneProposalsArgs(BaseModel):
    """Arguments for :func:`list_scene_proposals`."""

    model_config = ConfigDict(extra="forbid")

    category: str | None = None
    region_bev: list[float] | None = None
    limit: int | None = Field(default=None, ge=0)


@dataclass(frozen=True)
class InspectProposalResult:
    """Full detail for a single proposal."""

    proposal: Proposal
    pool_source: str

    def to_payload(self) -> dict[str, Any]:
        proposal = self.proposal
        payload: dict[str, Any] = {
            "proposal_id": proposal.proposal_id,
            "category": proposal.category,
            "score": proposal.score,
            "source": self.pool_source,
            "position_3d": [round(v, 4) for v in proposal.position_3d],
            "bbox_3d_9dof": [round(v, 4) for v in proposal.bbox_3d_9dof],
            "frame_count": len(proposal.frame_views),
            "frames_appeared": sorted(proposal.frame_views),
        }
        if proposal.enriched_category is not None:
            payload["enriched_category"] = proposal.enriched_category
        if proposal.compact_note is not None:
            payload["compact_note"] = proposal.compact_note
        if proposal.enrichment is not None:
            payload["enrichment"] = proposal.enrichment.to_payload()
        return payload


@dataclass(frozen=True)
class ListSceneProposalsResult:
    """A filtered shortlist of proposals."""

    proposals: tuple[Proposal, ...]

    def to_payload(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for proposal in self.proposals:
            row: dict[str, Any] = {
                "proposal_id": proposal.proposal_id,
                "category": proposal.category,
                "position_3d": [round(v, 4) for v in proposal.position_3d],
                "frame_count": len(proposal.frame_views),
            }
            if proposal.enriched_category is not None:
                row["enriched_category"] = proposal.enriched_category
            if proposal.compact_note is not None:
                row["compact_note"] = proposal.compact_note
            rows.append(row)
        return {"count": len(rows), "proposals": rows}


def inspect_proposal(
    scene: Nr3dScene, args: InspectProposalArgs
) -> InspectProposalResult:
    """Return one proposal's full enrichment and frame appearances."""
    proposal = scene.proposal_pool.get(args.proposal_id)
    if proposal is None:
        available = sorted(scene.proposal_pool.ids())
        raise ToolInputError(
            f"proposal_id={args.proposal_id} is not in the pool for scene "
            f"{scene.scene_id}; available ids={available}"
        )
    return InspectProposalResult(
        proposal=proposal, pool_source=scene.proposal_pool.source
    )


def list_scene_proposals(
    scene: Nr3dScene, args: ListSceneProposalsArgs
) -> ListSceneProposalsResult:
    """Return proposals filtered by category and/or BEV region."""
    proposals = sorted(scene.proposal_pool.proposals, key=lambda p: p.proposal_id)
    if args.category is not None:
        wanted = args.category.strip().lower()
        proposals = [p for p in proposals if p.category.strip().lower() == wanted]
    if args.region_bev is not None:
        box = _validated_region(args.region_bev)
        proposals = [p for p in proposals if _in_bev_box(p.position_3d, box)]
    if args.limit is not None:
        proposals = proposals[: args.limit]
    return ListSceneProposalsResult(proposals=tuple(proposals))


def _validated_region(region_bev: list[float]) -> tuple[float, float, float, float]:
    if len(region_bev) != _REGION_BEV_LEN:
        raise ToolInputError(
            "region_bev must be [xmin, ymin, xmax, ymax]; "
            f"got {len(region_bev)} values"
        )
    xmin, ymin, xmax, ymax = (float(v) for v in region_bev)
    return xmin, ymin, xmax, ymax


def _in_bev_box(
    position_3d: tuple[float, float, float],
    box: tuple[float, float, float, float],
) -> bool:
    xmin, ymin, xmax, ymax = box
    return xmin <= position_3d[0] <= xmax and ymin <= position_3d[1] <= ymax


__all__ = [
    "InspectProposalArgs",
    "ListSceneProposalsArgs",
    "InspectProposalResult",
    "ListSceneProposalsResult",
    "inspect_proposal",
    "list_scene_proposals",
]
