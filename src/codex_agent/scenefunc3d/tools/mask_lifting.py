"""Data contracts for future SceneFunc3D 2D-mask-to-3D lifting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, StringConstraints

from ...errors import SceneFunc3dDataError

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class LiftMaskResultPayload(TypedDict):
    """JSON-ready payload for a lifted 3D mask result."""

    frame_id: str
    candidate_id: str
    lifted_point_count: int
    mask_npz_path: str
    mask_ply_path: str
    overlay_path: str


class LiftMaskArgs(BaseModel):
    """Arguments for a future 2D-mask-to-3D lifting tool."""

    model_config = ConfigDict(extra="forbid")

    frame_id: NonEmptyText
    candidate_id: NonEmptyText
    mask_path: Path
    depth_path: Path
    intrinsics_path: Path
    pose_path: Path


@dataclass(frozen=True)
class LiftMaskResult:
    """Lifted point-mask artifact paths for one 2D mask candidate."""

    frame_id: str
    candidate_id: str
    lifted_point_count: int
    mask_npz_path: Path
    mask_ply_path: Path
    overlay_path: Path

    def __post_init__(self) -> None:
        """Validate directly constructed lifting result contracts."""
        if not self.frame_id.strip():
            raise SceneFunc3dDataError("frame_id must not be empty")
        if not self.candidate_id.strip():
            raise SceneFunc3dDataError("candidate_id must not be empty")
        if self.lifted_point_count < 0:
            raise SceneFunc3dDataError(
                "lifted_point_count must be non-negative; "
                f"got {self.lifted_point_count!r}"
            )

    def to_payload(self) -> LiftMaskResultPayload:
        """Return this lifting result as a JSON-ready mapping."""
        return {
            "frame_id": self.frame_id,
            "candidate_id": self.candidate_id,
            "lifted_point_count": self.lifted_point_count,
            "mask_npz_path": str(self.mask_npz_path),
            "mask_ply_path": str(self.mask_ply_path),
            "overlay_path": str(self.overlay_path),
        }


__all__ = ["LiftMaskArgs", "LiftMaskResult", "LiftMaskResultPayload"]
