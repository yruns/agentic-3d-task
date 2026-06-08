"""Typed NR3D proposal pool loaded from prepared scene artifacts.

A *proposal* is one candidate object in the scene (an oriented 3D box plus
category and optional enrichment). For a ``source: gt`` pool the proposal boxes
are the EmbodiedScan ground-truth annotations, so visual grounding reduces to
*selecting* the proposal that the referring expression describes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import Nr3dDataError

_BBOX_DOF = 9
_VALID_SOURCES: frozenset[str] = frozenset({"gt", "vdetr", "conceptgraph"})


@dataclass(frozen=True)
class Proposal:
    """One candidate object proposal in a scene."""

    proposal_id: int
    bbox_3d_9dof: tuple[float, ...]
    category: str
    score: float
    enriched_category: str | None = None
    compact_note: str | None = None
    visible_frame_ids: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.bbox_3d_9dof) != _BBOX_DOF:
            raise Nr3dDataError(
                f"proposal {self.proposal_id} bbox_3d_9dof must have "
                f"{_BBOX_DOF} values, got {len(self.bbox_3d_9dof)}"
            )


@dataclass(frozen=True)
class ProposalPool:
    """An immutable collection of proposals for one scene."""

    source: str
    scene_id: str
    proposals: tuple[Proposal, ...]

    def get(self, proposal_id: int) -> Proposal | None:
        """Return the proposal with ``proposal_id`` or ``None``."""
        return self._by_id().get(proposal_id)

    def require(self, proposal_id: int) -> Proposal:
        """Return the proposal with ``proposal_id`` or raise."""
        proposal = self.get(proposal_id)
        if proposal is None:
            raise Nr3dDataError(
                f"proposal_id={proposal_id} is not in the pool for "
                f"scene {self.scene_id}"
            )
        return proposal

    def ids(self) -> set[int]:
        """Return the set of all proposal ids in the pool."""
        return {proposal.proposal_id for proposal in self.proposals}

    def ids_by_category(self) -> dict[str, list[int]]:
        """Return proposal ids grouped (and sorted) by category."""
        grouped: dict[str, list[int]] = defaultdict(list)
        for proposal in self.proposals:
            grouped[proposal.category].append(proposal.proposal_id)
        return {category: sorted(ids) for category, ids in sorted(grouped.items())}

    def _by_id(self) -> dict[int, Proposal]:
        return {proposal.proposal_id: proposal for proposal in self.proposals}

    @classmethod
    def from_files(cls, proposals_path: Path, visibility_path: Path) -> ProposalPool:
        """Load a pool from ``proposals.jsonl`` and ``visibility.json``.

        Args:
            proposals_path: JSON file with a top-level ``proposals`` list.
            visibility_path: JSON map ``frame_id -> [proposal_id, ...]``.

        Raises:
            Nr3dDataError: If a file is missing or malformed.
        """
        raw = _load_json_object(proposals_path)
        source = str(raw.get("source", ""))
        if source not in _VALID_SOURCES:
            raise Nr3dDataError(
                f"{proposals_path}: source must be one of {sorted(_VALID_SOURCES)}, "
                f"got {source!r}"
            )
        scene_id = str(raw.get("scene_id", ""))
        raw_proposals = raw.get("proposals")
        if not isinstance(raw_proposals, list):
            raise Nr3dDataError(
                f"{proposals_path}: 'proposals' must be a list, got "
                f"{type(raw_proposals).__name__}"
            )

        frames_by_proposal = _invert_visibility(visibility_path)
        proposals = tuple(
            _parse_proposal(item, index, proposals_path, frames_by_proposal)
            for index, item in enumerate(raw_proposals)
        )
        return cls(source=source, scene_id=scene_id, proposals=proposals)


def _parse_proposal(
    item: Any,
    index: int,
    source_path: Path,
    frames_by_proposal: dict[int, list[int]],
) -> Proposal:
    if not isinstance(item, dict):
        raise Nr3dDataError(
            f"{source_path}: proposals[{index}] must be an object, got "
            f"{type(item).__name__}"
        )
    for required_key in ("bbox_3d", "score", "label"):
        if required_key not in item:
            raise Nr3dDataError(
                f"{source_path}: proposals[{index}].{required_key} is required"
            )
    bbox = item["bbox_3d"]
    if not isinstance(bbox, list) or len(bbox) != _BBOX_DOF:
        raise Nr3dDataError(
            f"{source_path}: proposals[{index}].bbox_3d must be a "
            f"{_BBOX_DOF}-element list"
        )
    proposal_id = int(item["id"]) if "id" in item else index
    return Proposal(
        proposal_id=proposal_id,
        bbox_3d_9dof=tuple(float(x) for x in bbox),
        category=str(item["label"]),
        score=float(item["score"]),
        enriched_category=_optional_str(item.get("enriched_category")),
        compact_note=_optional_str(item.get("compact_note")),
        visible_frame_ids=tuple(sorted(frames_by_proposal.get(proposal_id, []))),
    )


def _invert_visibility(visibility_path: Path) -> dict[int, list[int]]:
    raw = _load_json_object(visibility_path)
    frames_by_proposal: dict[int, list[int]] = defaultdict(list)
    for frame_id, proposal_ids in raw.items():
        if not isinstance(proposal_ids, list):
            raise Nr3dDataError(
                f"{visibility_path}: value for frame {frame_id!r} must be a list"
            )
        frame = int(frame_id)
        for proposal_id in proposal_ids:
            frames_by_proposal[int(proposal_id)].append(frame)
    return frames_by_proposal


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise Nr3dDataError(f"required NR3D artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Nr3dDataError(
            f"{path}: expected a JSON object, got {type(payload).__name__}"
        )
    return payload


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["Proposal", "ProposalPool"]
