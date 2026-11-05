"""Validated RGB-D observations and camera geometry for incremental memory.

Camera coordinates are OpenCV right/down/forward; world coordinates use Z up.
Simulator-specific axis conversion belongs at the observation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
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
    confidence: float
    image_feature: FloatArray
    text_feature: FloatArray
    bbox_xyxy: tuple[float, float, float, float]

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
        box = np.asarray(self.bbox_xyxy, dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("bbox_xyxy must contain four finite detector coordinates")
        x1, y1, x2, y2 = (float(value) for value in box)
        height, width = mask.shape
        if x2 <= x1 or y2 <= y1 or x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
            raise ValueError("bbox_xyxy must be nonempty and intersect the image")
        object.__setattr__(self, "bbox_xyxy", (x1, y1, x2, y2))
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, Real)
            or not np.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        for name in ("image_feature", "text_feature"):
            object.__setattr__(self, name, validated_feature(getattr(self, name), name))
        if self.image_feature.shape != self.text_feature.shape:
            raise ValueError("image_feature and text_feature dimensions must match")


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
    frame_feature: FloatArray

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
        object.__setattr__(
            self,
            "frame_feature",
            validated_feature(self.frame_feature, "frame_feature"),
        )
        for detection in self.detections:
            if not isinstance(detection, Detection):
                raise ValueError("detections must contain Detection instances")
            if detection.mask.shape != self.depth_m.shape:
                raise ValueError("detection mask shape must match depth_m")
            if detection.image_feature.shape != self.frame_feature.shape:
                raise ValueError("detection and frame_feature dimensions must match")


def project_pixels(
    observation: Observation, rows: NDArray[np.int64], cols: NDArray[np.int64]
) -> FloatArray:
    """Back-project selected nonzero metric depths into the Z-up world frame."""
    if (
        rows.ndim != 1
        or cols.shape != rows.shape
        or rows.dtype.kind not in "iu"
        or cols.dtype.kind not in "iu"
    ):
        raise ValueError(
            "rows and cols must be equally sized one-dimensional integer arrays"
        )
    height, width = observation.depth_m.shape
    if (
        np.any(rows < 0)
        or np.any(rows >= height)
        or np.any(cols < 0)
        or np.any(cols >= width)
    ):
        raise ValueError("pixel coordinates must lie inside the observation")
    depths = observation.depth_m[rows, cols]
    if np.any(depths <= 0):
        raise ValueError("selected pixels must have positive metric depth")
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


def depth_visible_points(
    points: FloatArray,
    depth_m: FloatArray,
    intrinsics: FloatArray,
    camera_to_world: FloatArray,
    *,
    tolerance_m: float,
) -> NDArray[np.bool_]:
    """Test world points against a pinhole image and its measured depth surface.

    Points behind the camera, outside the image, or hidden behind a nearer depth
    surface are not supporting evidence for an object-view link.
    """
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("points must be a finite Nx3 array")
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("tolerance_m must be positive and finite")
    if (
        depth_m.ndim != 2
        or intrinsics.shape != (3, 3)
        or camera_to_world.shape != (4, 4)
    ):
        raise ValueError(
            "visibility requires HxW depth, 3x3 intrinsics and 4x4 camera transform"
        )
    camera = (points - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    visible = np.zeros(len(points), dtype=bool)
    indices = np.flatnonzero(camera[:, 2] > 1e-8)
    if not len(indices):
        return visible
    projected = camera[indices] @ intrinsics.T
    cols = np.rint(projected[:, 0] / projected[:, 2]).astype(np.int64)
    rows = np.rint(projected[:, 1] / projected[:, 2]).astype(np.int64)
    height, width = depth_m.shape
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    valid_indices = indices[inside]
    observed = depth_m[rows[inside], cols[inside]]
    visible[valid_indices] = (observed > 0) & (
        np.abs(observed - camera[valid_indices, 2]) <= tolerance_m
    )
    return visible


def validated_feature(feature: FloatArray, name: str) -> FloatArray:
    """Copy a finite unit feature vector; reject absent or malformed embeddings."""
    vector = np.array(feature, dtype=np.float64, copy=True)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional vector")
    if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-4):
        raise ValueError(f"{name} must be L2-normalized")
    vector.setflags(write=False)
    return vector


def bound_rows(array: FloatArray, limit: int) -> FloatArray:
    """Deterministically retain spatial samples with a fixed memory bound."""
    if len(array) <= limit:
        return array
    indices = np.linspace(0, len(array) - 1, limit, dtype=int)
    return np.asarray(array[indices], dtype=np.float64)
