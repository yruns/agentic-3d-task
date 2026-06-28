"""Mask inspection, multi-view suggestion, and fusion contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


class MaskInspectionPayload(TypedDict):
    """JSON-ready mask inspection result."""

    artifact_path: str
    overlay_paths: list[str]
    lifted_point_count: int
    status: str


class SuggestedViewPayload(TypedDict):
    """JSON-ready suggested view."""

    frame_id: str
    reason: str
    rank: int


class SuggestedViewsPayload(TypedDict):
    """JSON-ready suggested view result."""

    seed_fragment_id: str
    views: list[SuggestedViewPayload]


class AcceptedFragmentPayload(TypedDict):
    """JSON-ready accepted mask fragment."""

    fragment_id: str
    point_count: int


class FusedMaskPayload(TypedDict):
    """JSON-ready fused mask result."""

    accepted_fragments: list[AcceptedFragmentPayload]
    mask_npz_path: str
    mask_ply_path: str


@dataclass(frozen=True)
class MaskInspectionResult:
    """Artifact inspection result for agent review."""

    artifact_path: Path
    overlay_paths: tuple[Path, ...]
    lifted_point_count: int
    status: str

    def __post_init__(self) -> None:
        """Validate domain invariants for one inspection result."""
        _validate_non_negative_int("lifted_point_count", self.lifted_point_count)
        _validate_non_empty_text("status", self.status)

    def to_payload(self) -> MaskInspectionPayload:
        """Return the JSON-ready inspection result."""
        return {
            "artifact_path": str(self.artifact_path),
            "overlay_paths": [str(path) for path in self.overlay_paths],
            "lifted_point_count": self.lifted_point_count,
            "status": self.status,
        }


@dataclass(frozen=True)
class SuggestedView:
    """One additional view suggested from an accepted 3D seed."""

    frame_id: str
    reason: str
    rank: int

    def __post_init__(self) -> None:
        """Validate domain invariants for one suggested view."""
        _validate_non_empty_text("frame_id", self.frame_id)
        _validate_non_empty_text("reason", self.reason)
        _validate_positive_int("rank", self.rank)

    def to_payload(self) -> SuggestedViewPayload:
        """Return the JSON-ready suggested view."""
        return {"frame_id": self.frame_id, "reason": self.reason, "rank": self.rank}


@dataclass(frozen=True)
class SuggestedViewsResult:
    """Additional views suggested for multi-view expansion."""

    seed_fragment_id: str
    views: tuple[SuggestedView, ...]

    def __post_init__(self) -> None:
        """Validate domain invariants for a suggested view set."""
        _validate_non_empty_text("seed_fragment_id", self.seed_fragment_id)

    def to_payload(self) -> SuggestedViewsPayload:
        """Return the JSON-ready suggested view set."""
        return {
            "seed_fragment_id": self.seed_fragment_id,
            "views": [view.to_payload() for view in self.views],
        }


@dataclass(frozen=True)
class AcceptedFragment:
    """One accepted 3D mask fragment."""

    fragment_id: str
    point_count: int

    def __post_init__(self) -> None:
        """Validate domain invariants for one accepted fragment."""
        _validate_non_empty_text("fragment_id", self.fragment_id)
        _validate_non_negative_int("point_count", self.point_count)

    def to_payload(self) -> AcceptedFragmentPayload:
        """Return the JSON-ready accepted fragment."""
        return {"fragment_id": self.fragment_id, "point_count": self.point_count}


@dataclass(frozen=True)
class FusedMaskResult:
    """Fused mask artifact result."""

    accepted_fragments: tuple[AcceptedFragment, ...]
    mask_npz_path: Path
    mask_ply_path: Path

    def to_payload(self) -> FusedMaskPayload:
        """Return the JSON-ready fused mask result."""
        return {
            "accepted_fragments": [
                fragment.to_payload() for fragment in self.accepted_fragments
            ],
            "mask_npz_path": str(self.mask_npz_path),
            "mask_ply_path": str(self.mask_ply_path),
        }


def _validate_non_empty_text(field_name: str, field_value: str) -> None:
    if field_value.strip() == "":
        raise ValueError(f"{field_name} must be non-empty")


def _validate_non_negative_int(field_name: str, field_value: int) -> None:
    if field_value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_positive_int(field_name: str, field_value: int) -> None:
    if field_value <= 0:
        raise ValueError(f"{field_name} must be positive")


__all__ = [
    "AcceptedFragment",
    "AcceptedFragmentPayload",
    "FusedMaskPayload",
    "FusedMaskResult",
    "MaskInspectionPayload",
    "MaskInspectionResult",
    "SuggestedView",
    "SuggestedViewPayload",
    "SuggestedViewsPayload",
    "SuggestedViewsResult",
]
