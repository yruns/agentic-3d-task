"""Typed NR3D proposal pool loaded from prepared scene artifacts.

A *proposal* is one candidate object in the scene (an oriented 3D box plus
category and optional enrichment). For a ``source: gt`` pool the proposal boxes
are the EmbodiedScan ground-truth annotations, so visual grounding reduces to
*selecting* the proposal that the referring expression describes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import Nr3dDataError

_BBOX_DOF = 9
_BBOX_2D_LEN = 4
_VALID_SOURCES: frozenset[str] = frozenset({"gt", "vdetr", "conceptgraph"})


@dataclass(frozen=True)
class FrameView:
    """One proposal's 2D appearance in a single first-person frame."""

    frame_id: int
    bbox_2d: tuple[int, int, int, int]
    raw_rgb_path: str

    @property
    def center_x(self) -> float:
        """Horizontal pixel center of the 2D box (used for left/right voting)."""
        return (float(self.bbox_2d[0]) + float(self.bbox_2d[2])) / 2.0


@dataclass(frozen=True)
class ProposalEnrichment:
    """Structured VLM enrichment for a proposal (color/description/neighbors).

    Modeled explicitly instead of a bare ``dict`` so downstream tools get a
    typed, stable surface. Unknown/absent fields default to empty values.
    """

    category: str = ""
    color: str = ""
    description: str = ""
    location: str = ""
    usability: str = ""
    nearby_objects: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ProposalEnrichment:
        """Build enrichment from an artifact mapping, coercing types safely."""
        nearby = raw.get("nearby_objects")
        nearby_objects = (
            tuple(str(item) for item in nearby) if isinstance(nearby, list) else ()
        )
        return cls(
            category=str(raw.get("category", "")),
            color=str(raw.get("color", "")),
            description=str(raw.get("description", "")),
            location=str(raw.get("location", "")),
            usability=str(raw.get("usability", "")),
            nearby_objects=nearby_objects,
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable view, omitting empty fields."""
        payload: dict[str, Any] = {}
        if self.category:
            payload["category"] = self.category
        if self.color:
            payload["color"] = self.color
        if self.description:
            payload["description"] = self.description
        if self.location:
            payload["location"] = self.location
        if self.usability:
            payload["usability"] = self.usability
        if self.nearby_objects:
            payload["nearby_objects"] = list(self.nearby_objects)
        return payload


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
    frame_views: Mapping[int, FrameView] = field(default_factory=dict)
    enrichment: ProposalEnrichment | None = None

    def __post_init__(self) -> None:
        if len(self.bbox_3d_9dof) != _BBOX_DOF:
            raise Nr3dDataError(
                f"proposal {self.proposal_id} bbox_3d_9dof must have "
                f"{_BBOX_DOF} values, got {len(self.bbox_3d_9dof)}"
            )

    @property
    def position_3d(self) -> tuple[float, float, float]:
        """The 3D box center ``(cx, cy, cz)`` (first three 9-DoF values)."""
        return (
            self.bbox_3d_9dof[0],
            self.bbox_3d_9dof[1],
            self.bbox_3d_9dof[2],
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

    def frame_to_proposal_ids(self) -> dict[int, list[int]]:
        """Return ``frame_id -> sorted proposal ids`` from per-frame 2D views."""
        grouped: dict[int, list[int]] = defaultdict(list)
        for proposal in self.proposals:
            for frame_id in proposal.frame_views:
                grouped[int(frame_id)].append(proposal.proposal_id)
        return {frame_id: sorted(ids) for frame_id, ids in grouped.items()}

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
    frame_views = _parse_frame_views(item.get("frame_views"), proposal_id, source_path)
    visible_from_views = sorted(frame_views)
    visible_frame_ids = (
        tuple(visible_from_views)
        if visible_from_views
        else tuple(sorted(frames_by_proposal.get(proposal_id, [])))
    )
    return Proposal(
        proposal_id=proposal_id,
        bbox_3d_9dof=tuple(float(x) for x in bbox),
        category=str(item["label"]),
        score=float(item["score"]),
        enriched_category=_optional_str(item.get("enriched_category")),
        compact_note=_optional_str(item.get("compact_note")),
        visible_frame_ids=visible_frame_ids,
        frame_views=frame_views,
        enrichment=_parse_enrichment(item.get("enrichment")),
    )


def _parse_frame_views(
    raw: Any, proposal_id: int, source_path: Path
) -> dict[int, FrameView]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise Nr3dDataError(
            f"{source_path}: proposal {proposal_id} frame_views must be an "
            f"object, got {type(raw).__name__}"
        )
    frame_views: dict[int, FrameView] = {}
    for frame_key, view in raw.items():
        if not isinstance(view, dict):
            raise Nr3dDataError(
                f"{source_path}: proposal {proposal_id} frame_views[{frame_key!r}] "
                f"must be an object, got {type(view).__name__}"
            )
        bbox_2d = view.get("bbox_2d")
        if not isinstance(bbox_2d, list) or len(bbox_2d) != _BBOX_2D_LEN:
            raise Nr3dDataError(
                f"{source_path}: proposal {proposal_id} frame_views[{frame_key!r}]"
                f".bbox_2d must be a {_BBOX_2D_LEN}-element list"
            )
        raw_rgb_path = view.get("raw_rgb_path")
        if not isinstance(raw_rgb_path, str) or not raw_rgb_path:
            raise Nr3dDataError(
                f"{source_path}: proposal {proposal_id} frame_views[{frame_key!r}]"
                ".raw_rgb_path must be a non-empty string"
            )
        frame_id = int(frame_key)
        frame_views[frame_id] = FrameView(
            frame_id=frame_id,
            bbox_2d=(
                int(round(float(bbox_2d[0]))),
                int(round(float(bbox_2d[1]))),
                int(round(float(bbox_2d[2]))),
                int(round(float(bbox_2d[3]))),
            ),
            raw_rgb_path=raw_rgb_path,
        )
    return frame_views


def _parse_enrichment(raw: Any) -> ProposalEnrichment | None:
    if not isinstance(raw, dict) or not raw:
        return None
    return ProposalEnrichment.from_mapping(raw)


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


__all__ = ["FrameView", "ProposalEnrichment", "Proposal", "ProposalPool"]
