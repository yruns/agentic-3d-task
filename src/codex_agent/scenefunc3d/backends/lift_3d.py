"""Deterministic 2D mask to 3D point lifting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from codex_agent.scenefunc3d.tools.models import ToolInputError

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
_POSE_LAST_ROW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
_POSE_LAST_ROW_ATOL = 1e-8
_MIN_ROTATION_DETERMINANT_ABS = 1e-12


@dataclass(frozen=True)
class CameraGeometry:
    """Validated pinhole camera intrinsics and camera-to-world pose."""

    intrinsics: FloatArray
    camera_to_world: FloatArray

    def __post_init__(self) -> None:
        """Validate camera matrices before lifting any external frame data."""
        intrinsics = _coerce_float_array(
            self.intrinsics,
            field_name="intrinsics",
            expected_shape=(3, 3),
        )
        camera_to_world = _coerce_float_array(
            self.camera_to_world,
            field_name="camera_to_world",
            expected_shape=(4, 4),
        )
        if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
            raise ToolInputError(
                "invalid camera intrinsics: focal lengths must be positive"
            )
        if not bool(
            np.allclose(
                camera_to_world[3, :],
                _POSE_LAST_ROW,
                atol=_POSE_LAST_ROW_ATOL,
                rtol=0.0,
            )
        ):
            raise ToolInputError(
                "invalid camera_to_world pose: last row must be [0, 0, 0, 1]"
            )
        rotation_determinant = float(np.linalg.det(camera_to_world[:3, :3]))
        if abs(rotation_determinant) <= _MIN_ROTATION_DETERMINANT_ABS:
            raise ToolInputError(
                "invalid camera_to_world pose: rotation block must be non-degenerate"
            )
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", camera_to_world)


def load_mask_npz(path: Path) -> BoolArray:
    """Load a 2D boolean mask from a SAM-style NPZ artifact."""
    try:
        with np.load(path) as archive:
            if "mask" not in archive.files:
                raise ToolInputError(
                    "mask npz is missing required key 'mask': " f"path={path}"
                )
            mask_array = np.asarray(archive["mask"])
    except ToolInputError:
        raise
    except (OSError, ValueError) as exc:
        raise ToolInputError(
            "could not load mask npz artifact: "
            f"path={path}; error_type={exc.__class__.__name__}"
        ) from exc

    return _coerce_binary_mask(mask_array, field_name=f"mask array at {path}")


def backproject_mask_to_world(
    mask: BoolArray,
    depth_meters: FloatArray,
    geometry: CameraGeometry,
) -> FloatArray:
    """Back-project positive-depth mask pixels into world-space XYZ points."""
    boolean_mask = _coerce_bool_mask(mask, field_name="mask")
    depth = _coerce_depth(depth_meters)
    if depth.shape != boolean_mask.shape:
        raise ToolInputError(
            "depth_meters shape must match mask shape: "
            f"mask_shape={boolean_mask.shape}; depth_shape={depth.shape}"
        )

    valid_depth_mask = np.isfinite(depth) & (depth > 0.0)
    valid_mask = boolean_mask & valid_depth_mask
    if not bool(np.any(valid_mask)):
        raise ToolInputError(
            "lift_too_sparse: mask contains no pixels with finite positive depth"
        )
    invalid_masked_depth = boolean_mask & ~valid_depth_mask
    if bool(np.any(invalid_masked_depth)):
        raise ToolInputError(
            "lift_invalid_depth: mask contains non-finite or non-positive depth pixels"
        )

    rows, columns = np.nonzero(boolean_mask)
    z_values = depth[rows, columns]
    fx = geometry.intrinsics[0, 0]
    fy = geometry.intrinsics[1, 1]
    cx = geometry.intrinsics[0, 2]
    cy = geometry.intrinsics[1, 2]
    x_values = (columns.astype(np.float64) - cx) * z_values / fx
    y_values = (rows.astype(np.float64) - cy) * z_values / fy
    camera_points = np.column_stack((x_values, y_values, z_values))
    homogeneous_camera_points = np.column_stack(
        (camera_points, np.ones(camera_points.shape[0], dtype=np.float64))
    )
    world_homogeneous = homogeneous_camera_points @ geometry.camera_to_world.T
    world_points = cast(FloatArray, world_homogeneous[:, :3].astype(np.float64))
    return _validate_points_world(world_points, field_name="points_world")


def write_lift_npz(path: Path, points_world: FloatArray) -> Path:
    """Write lifted world points to a compressed NPZ artifact."""
    valid_points = _validate_points_world(points_world, field_name="points_world")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, points_world=valid_points)
    except OSError as exc:
        raise ToolInputError(
            "could not write lifted points npz: "
            f"path={path}; error_type={exc.__class__.__name__}"
        ) from exc
    return path


def write_lift_ply(path: Path, points_world: FloatArray) -> Path:
    """Write lifted world points as an ASCII PLY point cloud."""
    valid_points = _validate_points_world(points_world, field_name="points_world")
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {valid_points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    vertex_lines = "\n".join(
        " ".join(_format_float(coordinate) for coordinate in point)
        for point in valid_points
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{header}{vertex_lines}\n", encoding="ascii")
    except OSError as exc:
        raise ToolInputError(
            "could not write lifted points ply: "
            f"path={path}; error_type={exc.__class__.__name__}"
        ) from exc
    return path


def _coerce_float_array(
    value: FloatArray,
    *,
    field_name: str,
    expected_shape: tuple[int, int],
) -> FloatArray:
    array = cast(FloatArray, np.asarray(value, dtype=np.float64))
    if array.shape != expected_shape:
        raise ToolInputError(
            f"{field_name} must have shape {expected_shape}: shape={array.shape}"
        )
    if not bool(np.all(np.isfinite(array))):
        raise ToolInputError(f"{field_name} must contain only finite values")
    return array


def _coerce_bool_mask(mask: BoolArray, *, field_name: str) -> BoolArray:
    mask_array = np.asarray(mask)
    return _coerce_binary_mask(mask_array, field_name=field_name)


def _coerce_binary_mask(
    mask_array: npt.NDArray[np.generic],
    *,
    field_name: str,
) -> BoolArray:
    if mask_array.ndim != 2:
        raise ToolInputError(f"{field_name} must be a 2D mask: ndim={mask_array.ndim}")
    if mask_array.dtype == np.dtype(np.bool_):
        return cast(BoolArray, mask_array.astype(np.bool_, copy=False))
    if np.issubdtype(mask_array.dtype, np.complexfloating):
        raise ToolInputError(f"{field_name} must be a binary numeric or bool mask")
    if not np.issubdtype(mask_array.dtype, np.number):
        raise ToolInputError(f"{field_name} must be a binary numeric or bool mask")
    numeric_mask = cast(FloatArray, np.asarray(mask_array, dtype=np.float64))
    if not bool(np.all(np.isfinite(numeric_mask))):
        raise ToolInputError(f"{field_name} must be a finite binary mask")
    is_binary = (numeric_mask == 0.0) | (numeric_mask == 1.0)
    if not bool(np.all(is_binary)):
        raise ToolInputError(f"{field_name} must be a binary mask with values 0 or 1")
    return cast(BoolArray, numeric_mask.astype(np.bool_, copy=False))


def _coerce_depth(depth_meters: FloatArray) -> FloatArray:
    depth = cast(FloatArray, np.asarray(depth_meters, dtype=np.float64))
    if depth.ndim != 2:
        raise ToolInputError(f"depth_meters must be 2D: ndim={depth.ndim}")
    return depth


def _validate_points_world(points_world: FloatArray, *, field_name: str) -> FloatArray:
    points = cast(FloatArray, np.asarray(points_world, dtype=np.float64))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ToolInputError(
            f"{field_name} must have shape (N, 3): shape={points.shape}"
        )
    if points.shape[0] == 0:
        raise ToolInputError(f"{field_name} must contain at least one point")
    if not bool(np.all(np.isfinite(points))):
        raise ToolInputError(f"{field_name} must contain only finite values")
    return points


def _format_float(value: np.float64) -> str:
    return format(float(value), ".15g")


__all__ = [
    "BoolArray",
    "CameraGeometry",
    "FloatArray",
    "backproject_mask_to_world",
    "load_mask_npz",
    "write_lift_npz",
    "write_lift_ply",
]
