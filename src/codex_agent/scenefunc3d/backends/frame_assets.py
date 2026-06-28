"""Resolve raw frame geometry assets for deterministic mask lifting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from re import Pattern

from codex_agent.scenefunc3d.tools.models import ToolInputError
from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene

_SAFE_FRAME_ID_RE: Pattern[str] = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class FrameGeometryAssets:
    """Raw depth, intrinsics, and pose paths for one source frame."""

    frame_id: str
    depth_path: Path
    intrinsics_path: Path
    pose_path: Path

    def __post_init__(self) -> None:
        """Validate directly constructed frame-geometry asset records."""
        if not _SAFE_FRAME_ID_RE.fullmatch(self.frame_id):
            raise ToolInputError(
                "frame_geometry_asset_invalid_frame_id: " f"frame_id={self.frame_id!r}"
            )


def resolve_frame_geometry_assets(
    tool_scene: SceneFunc3dToolScene,
    frame_id: str,
) -> FrameGeometryAssets:
    """Resolve narrow raw-layout geometry assets for ``frame_id``."""
    if not _SAFE_FRAME_ID_RE.fullmatch(frame_id):
        raise ToolInputError(
            "frame_geometry_asset_invalid_frame_id: " f"frame_id={frame_id!r}"
        )

    depth_path = _first_existing_path(_depth_candidates(tool_scene.raw_dir, frame_id))
    intrinsics_path = _first_existing_path(
        _intrinsics_candidates(tool_scene.raw_dir, frame_id)
    )
    pose_path = _first_existing_path(_pose_candidates(tool_scene.raw_dir, frame_id))
    if depth_path is None or intrinsics_path is None or pose_path is None:
        missing_fields: list[str] = []
        if depth_path is None:
            missing_fields.append("depth")
        if intrinsics_path is None:
            missing_fields.append("intrinsics")
        if pose_path is None:
            missing_fields.append("pose")
        raise ToolInputError(
            "frame_geometry_asset_missing: "
            f"frame_id={frame_id!r}; raw_dir={tool_scene.raw_dir}; "
            f"missing={','.join(missing_fields)}"
        )
    return FrameGeometryAssets(
        frame_id=frame_id,
        depth_path=depth_path,
        intrinsics_path=intrinsics_path,
        pose_path=pose_path,
    )


def _depth_candidates(raw_dir: Path, frame_id: str) -> tuple[Path, ...]:
    return (
        raw_dir / f"{frame_id}-depth.png",
        raw_dir / f"{frame_id}_depth.png",
        raw_dir / f"{frame_id}.depth.png",
        raw_dir / "depth" / f"{frame_id}.png",
        raw_dir / "depths" / f"{frame_id}.png",
    )


def _intrinsics_candidates(raw_dir: Path, frame_id: str) -> tuple[Path, ...]:
    return (
        raw_dir / "intrinsics.txt",
        raw_dir / "intrinsic.txt",
        raw_dir / f"{frame_id}-intrinsics.txt",
        raw_dir / f"{frame_id}_intrinsics.txt",
        raw_dir / "intrinsics" / f"{frame_id}.txt",
        raw_dir / "intrinsic" / f"{frame_id}.txt",
    )


def _pose_candidates(raw_dir: Path, frame_id: str) -> tuple[Path, ...]:
    return (
        raw_dir / f"{frame_id}-pose.txt",
        raw_dir / f"{frame_id}_pose.txt",
        raw_dir / f"{frame_id}.pose.txt",
        raw_dir / "pose" / f"{frame_id}.txt",
        raw_dir / "poses" / f"{frame_id}.txt",
        raw_dir / "extrinsic" / f"{frame_id}.txt",
        raw_dir / "extrinsics" / f"{frame_id}.txt",
    )


def _first_existing_path(candidates: tuple[Path, ...]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


__all__ = ["FrameGeometryAssets", "resolve_frame_geometry_assets"]
