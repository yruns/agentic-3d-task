"""Data contracts for future SceneFunc3D SAM mask candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field, StrictStr, StringConstraints

from ...errors import SceneFunc3dDataError

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
StrictPixelCoordinate = Annotated[float, Field(ge=0.0, strict=True)]


class SamCandidatePayload(TypedDict):
    """JSON-ready payload for one SAM mask candidate."""

    candidate_id: str
    score: float
    pixel_count: int
    coverage_percent: float
    overlay_path: str


class SamMaskResultPayload(TypedDict):
    """JSON-ready payload for SAM mask candidate results."""

    frame_id: str
    candidates: list[SamCandidatePayload]
    contact_sheet_path: str


class SamMaskArgsPayload(TypedDict):
    """JSON-ready payload for a future SAM mask request."""

    frame_id: str
    image_path: str
    points: list[SamPointInputPayload]


class SamPointInputPayload(TypedDict):
    """JSON-ready payload for one SAM point prompt."""

    x_px: float
    y_px: float
    source: str
    label: str


class SamPointInput(BaseModel):
    """Strict JSON-boundary point input for future SAM mask generation."""

    model_config = ConfigDict(extra="forbid")

    x_px: StrictPixelCoordinate
    y_px: StrictPixelCoordinate
    source: StrictStr = ""
    label: StrictStr = ""

    def to_payload(self) -> SamPointInputPayload:
        """Return this point prompt as a JSON-ready mapping."""
        return {
            "x_px": self.x_px,
            "y_px": self.y_px,
            "source": self.source,
            "label": self.label,
        }


class SamMaskArgs(BaseModel):
    """Arguments for a future SAM mask candidate tool."""

    model_config = ConfigDict(extra="forbid")

    frame_id: NonEmptyText
    image_path: Path
    points: tuple[SamPointInput, ...] = Field(min_length=1)

    def to_payload(self) -> SamMaskArgsPayload:
        """Return this request as a JSON-ready mapping."""
        return {
            "frame_id": self.frame_id,
            "image_path": str(self.image_path),
            "points": [point.to_payload() for point in self.points],
        }


@dataclass(frozen=True)
class SamCandidate:
    """One SAM mask candidate proposed from a Molmo point prompt."""

    candidate_id: str
    score: float
    pixel_count: int
    coverage_percent: float
    overlay_path: Path

    def __post_init__(self) -> None:
        """Validate directly constructed candidate contracts."""
        if not self.candidate_id.strip():
            raise SceneFunc3dDataError("candidate_id must not be empty")
        if not 0.0 <= self.score <= 1.0:
            raise SceneFunc3dDataError(
                f"candidate score must be in [0, 1]; got {self.score!r}"
            )
        if self.pixel_count < 0:
            raise SceneFunc3dDataError(
                f"candidate pixel_count must be non-negative; got {self.pixel_count!r}"
            )
        if not 0.0 <= self.coverage_percent <= 100.0:
            raise SceneFunc3dDataError(
                "candidate coverage_percent must be in [0, 100]; "
                f"got {self.coverage_percent!r}"
            )

    def to_payload(self) -> SamCandidatePayload:
        """Return this candidate as a JSON-ready mapping."""
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "pixel_count": self.pixel_count,
            "coverage_percent": self.coverage_percent,
            "overlay_path": str(self.overlay_path),
        }


@dataclass(frozen=True)
class SamMaskResult:
    """SAM mask candidates and rendered contact-sheet artifact for one frame."""

    frame_id: str
    candidates: tuple[SamCandidate, ...]
    contact_sheet_path: Path

    def to_payload(self) -> SamMaskResultPayload:
        """Return the JSON-ready CLI payload."""
        return {
            "frame_id": self.frame_id,
            "candidates": [candidate.to_payload() for candidate in self.candidates],
            "contact_sheet_path": str(self.contact_sheet_path),
        }


__all__ = [
    "SamCandidate",
    "SamCandidatePayload",
    "SamMaskArgs",
    "SamMaskArgsPayload",
    "SamMaskResult",
    "SamMaskResultPayload",
    "SamPointInput",
    "SamPointInputPayload",
]
