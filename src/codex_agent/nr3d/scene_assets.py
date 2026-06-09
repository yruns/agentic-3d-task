"""Auxiliary per-scene assets used by the NR3D agent tools.

The proposal pool (``proposals.py``) carries the objects and their per-frame 2D
views. Two more prepared artifacts are needed by the tool layer and are loaded
here:

* ``camera_trajectory.json`` — ``frame_id -> [x, y, yaw]`` BEV poses, used to
  report where a first-person frame was taken and for region selection;
* ``bev/scene_bev_nr3d.view.json`` — the look-down virtual-camera parameters
  written next to the base BEV render, used to project a proposal's 3D center
  onto the BEV image for ``view_bev`` highlights.

Both are optional: a scene without them still supports the text/frame tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import Nr3dDataError

_VIEW_ROTATION_DIM = 3
_VIEW_TRANSLATION_DIM = 3
_VIEW_CROP_OFFSET_DIM = 2
_TRAJECTORY_ENTRY_LEN = 3

BevPose = tuple[float, float, float]


@dataclass(frozen=True)
class CameraTrajectory:
    """BEV camera poses keyed by frame id (``x``, ``y`` meters; ``yaw`` radians)."""

    poses_by_frame: dict[int, BevPose]

    def pose(self, frame_id: int) -> BevPose | None:
        """Return the ``(x, y, yaw)`` pose for ``frame_id`` or ``None``."""
        return self.poses_by_frame.get(int(frame_id))

    @classmethod
    def load(cls, path: Path) -> CameraTrajectory:
        """Load a trajectory map; raise on malformed content."""
        payload = _load_json(path)
        if not isinstance(payload, dict):
            raise Nr3dDataError(f"{path}: camera trajectory must be a JSON object")
        poses_by_frame: dict[int, BevPose] = {}
        for frame_key, entry in payload.items():
            if not isinstance(entry, list) or len(entry) != _TRAJECTORY_ENTRY_LEN:
                raise Nr3dDataError(
                    f"{path}: trajectory[{frame_key!r}] must be a "
                    f"{_TRAJECTORY_ENTRY_LEN}-element [x, y, yaw] list"
                )
            poses_by_frame[int(frame_key)] = (
                float(entry[0]),
                float(entry[1]),
                float(entry[2]),
            )
        return cls(poses_by_frame=poses_by_frame)


@dataclass(frozen=True)
class BevViewParams:
    """Look-down virtual-camera parameters for the prepared BEV render.

    Mirrors :class:`keyframe.bev.render.CameraView` but stays dependency-free
    (no numpy / OpenCV) so it can be loaded in lightweight tool processes.
    """

    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]
    focal: float
    center: float
    image_size: int
    crop_offset: tuple[int, int]

    @classmethod
    def from_mapping(cls, raw: Any, *, source: Path) -> BevViewParams:
        """Validate and build view parameters from the ``.view.json`` mapping."""
        if not isinstance(raw, dict):
            raise Nr3dDataError(f"{source}: BEV view params must be a JSON object")
        rotation = _coerce_rotation(raw.get("R"), source=source)
        translation = _coerce_vec3(raw.get("t"), field_name="t", source=source)
        crop = raw.get("crop_offset")
        if not isinstance(crop, list) or len(crop) != _VIEW_CROP_OFFSET_DIM:
            raise Nr3dDataError(
                f"{source}: crop_offset must be a {_VIEW_CROP_OFFSET_DIM}-element list"
            )
        return cls(
            rotation=rotation,
            translation=translation,
            focal=float(raw["f"]),
            center=float(raw["c"]),
            image_size=int(raw["image_size"]),
            crop_offset=(int(crop[0]), int(crop[1])),
        )

    @classmethod
    def load(cls, path: Path) -> BevViewParams:
        """Load BEV view parameters from a ``.view.json`` sidecar."""
        return cls.from_mapping(_load_json(path), source=path)


def _coerce_rotation(
    raw: Any, *, source: Path
) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(raw, list) or len(raw) != _VIEW_ROTATION_DIM:
        raise Nr3dDataError(f"{source}: R must be a {_VIEW_ROTATION_DIM}x3 matrix")
    rows: list[tuple[float, float, float]] = []
    for row in raw:
        rows.append(_coerce_vec3(row, field_name="R row", source=source))
    return tuple(rows)


def _coerce_vec3(
    raw: Any, *, field_name: str, source: Path
) -> tuple[float, float, float]:
    if not isinstance(raw, list) or len(raw) != _VIEW_TRANSLATION_DIM:
        raise Nr3dDataError(
            f"{source}: {field_name} must be a {_VIEW_TRANSLATION_DIM}-element list"
        )
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise Nr3dDataError(f"required NR3D asset is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "BevPose",
    "CameraTrajectory",
    "BevViewParams",
]
