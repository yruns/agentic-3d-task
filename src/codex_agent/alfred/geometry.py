"""Validated RGB-D observations and camera geometry for incremental memory.

Camera coordinates are OpenCV right/down/forward; world coordinates use Z up.
Simulator-specific axis conversion belongs at the observation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
GridCell = tuple[int, int]


@dataclass(frozen=True)
class Detection:
    """One observed image region, without privileged simulator object identity."""

    category: str
    mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError(
                f"category must be a nonempty string, got {self.category!r}"
            )
        raw = np.asarray(self.mask)
        if raw.ndim != 2 or not raw.size or not np.isin(raw, [0, 1]).all():
            raise ValueError("mask must be a nonempty binary 2D array")
        mask = raw.astype(bool, copy=True)
        if not mask.any():
            raise ValueError("mask must contain at least one observed pixel")
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class Observation:
    """A captured frame with metric geometry, ready for online integration.

    Depth zero means unavailable; other depth samples must be finite and positive.
    The camera transform maps OpenCV camera points to the common Z-up world frame.
    ``agent_position`` is the agent base position, not necessarily the camera center.
    """

    frame_id: int
    rgb_path: Path
    depth_m: FloatArray
    intrinsics: FloatArray
    camera_to_world: FloatArray
    agent_position: tuple[float, float, float]
    detections: tuple[Detection, ...]

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int):
            raise ValueError(
                f"frame_id must be a nonnegative int, got {self.frame_id!r}"
            )
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be nonnegative, got {self.frame_id}")
        object.__setattr__(self, "rgb_path", Path(self.rgb_path))
        for name in ("depth_m", "intrinsics", "camera_to_world"):
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if (
            self.depth_m.ndim != 2
            or not self.depth_m.size
            or np.any(self.depth_m < 0)
            or not np.any(self.depth_m > 0)
        ):
            raise ValueError("depth_m must be nonnegative HxW with positive samples")
        if (
            self.intrinsics.shape != (3, 3)
            or self.intrinsics[0, 0] <= 0
            or self.intrinsics[1, 1] <= 0
            or not np.allclose(self.intrinsics[2], [0, 0, 1])
            or not np.isclose(self.intrinsics[0, 1], 0)
            or not np.isclose(self.intrinsics[1, 0], 0)
        ):
            raise ValueError("intrinsics must be a 3x3 zero-skew pinhole matrix")
        if self.camera_to_world.shape != (4, 4) or not np.allclose(
            self.camera_to_world[3], [0, 0, 0, 1]
        ):
            raise ValueError("camera_to_world must be a homogeneous 4x4 transform")
        rotation = self.camera_to_world[:3, :3]
        # THOR's left-handed world converted to (x,z,y) may introduce reflection.
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("camera_to_world rotation must be orthonormal")
        if len(self.agent_position) != 3 or not np.isfinite(self.agent_position).all():
            raise ValueError(
                "agent_position must contain three finite world coordinates"
            )
        object.__setattr__(
            self, "agent_position", tuple(float(value) for value in self.agent_position)
        )
        object.__setattr__(self, "detections", tuple(self.detections))
        for detection in self.detections:
            if not isinstance(detection, Detection):
                raise ValueError("detections must contain Detection instances")
            if detection.mask.shape != self.depth_m.shape:
                raise ValueError("detection mask shape must match depth_m")


def project_pixels(
    observation: Observation, rows: NDArray[np.int64], cols: NDArray[np.int64]
) -> FloatArray:
    """Back-project selected nonzero metric depths into the Z-up world frame."""
    depths = observation.depth_m[rows, cols]
    intrinsics = observation.intrinsics
    camera_points = np.column_stack(
        (
            (cols - intrinsics[0, 2]) * depths / intrinsics[0, 0],
            (rows - intrinsics[1, 2]) * depths / intrinsics[1, 1],
            depths,
        )
    )
    return np.asarray(
        camera_points @ observation.camera_to_world[:3, :3].T
        + observation.camera_to_world[:3, 3],
        dtype=np.float64,
    )


def appearance_histogram(colors: NDArray[np.uint8]) -> FloatArray:
    """Return an L2-normalized RGB histogram, a lightweight appearance descriptor."""
    histogram = np.concatenate(
        [
            np.histogram(colors[:, channel], bins=8, range=(0, 256))[0]
            for channel in range(3)
        ]
    ).astype(np.float64)
    return np.asarray(histogram / np.linalg.norm(histogram), dtype=np.float64)


def bound_rows(array: FloatArray, limit: int) -> FloatArray:
    """Deterministically retain spatial samples with a fixed memory bound."""
    if len(array) <= limit:
        return array
    indices = np.linspace(0, len(array) - 1, limit, dtype=int)
    return np.asarray(array[indices], dtype=np.float64)
