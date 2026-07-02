"""Shared depth-image and camera-matrix readers for SceneFunc3D backends.

Extracted from ``tools/mask_lifting.py`` so the multi-view pipeline and the
lift tool read camera geometry through one validated path (DRY). Depth PNGs are
16-bit millimetre ARKit depth; matrices are whitespace-delimited text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry, FloatArray
from codex_agent.scenefunc3d.tools.models import ToolInputError

_DEPTH_MILLIMETRES_PER_METRE = 1000.0


def read_depth_meters(depth_path: Path) -> FloatArray:
    """Read a single-channel 16-bit depth PNG as float64 metres."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ToolInputError(
            "Pillow is required to read depth images; install the 'vision' extra"
        ) from exc

    try:
        with Image.open(depth_path) as image:
            depth_pixels = np.asarray(image, dtype=np.float64)
    except OSError as exc:
        raise ToolInputError(
            "could not read depth image: "
            f"path={depth_path}; error_type={exc.__class__.__name__}"
        ) from exc
    except ValueError as exc:
        raise ToolInputError(
            "could not parse depth image: "
            f"path={depth_path}; error_type={exc.__class__.__name__}"
        ) from exc

    if depth_pixels.ndim != 2:
        raise ToolInputError(
            "depth image must be a single-channel 2D image: "
            f"path={depth_path}; shape={depth_pixels.shape}"
        )
    depth_meters: FloatArray = depth_pixels / _DEPTH_MILLIMETRES_PER_METRE
    return depth_meters


def read_camera_matrix(
    matrix_path: Path,
    *,
    expected_shape: tuple[int, int],
    field_name: str,
) -> FloatArray:
    """Read a whitespace-delimited matrix and reshape to ``expected_shape``."""
    expected_size = expected_shape[0] * expected_shape[1]
    try:
        raw_matrix = np.loadtxt(matrix_path, dtype=np.float64)
        matrix = np.asarray(raw_matrix, dtype=np.float64)
    except OSError as exc:
        raise ToolInputError(
            "could not read camera matrix file: "
            f"field={field_name}; path={matrix_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc
    except ValueError as exc:
        raise ToolInputError(
            "could not parse camera matrix file: "
            f"field={field_name}; path={matrix_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc

    if matrix.size != expected_size:
        raise ToolInputError(
            "camera matrix has wrong element count: "
            f"field={field_name}; path={matrix_path}; "
            f"expected={expected_size}; actual={matrix.size}"
        )
    reshaped_matrix: FloatArray = matrix.reshape(expected_shape)
    return reshaped_matrix


def load_camera_geometry(*, intrinsics_path: Path, pose_path: Path) -> CameraGeometry:
    """Load intrinsics + camera-to-world pose into a validated ``CameraGeometry``."""
    intrinsics = read_camera_matrix(
        intrinsics_path, expected_shape=(3, 3), field_name="intrinsics"
    )
    camera_to_world = read_camera_matrix(
        pose_path, expected_shape=(4, 4), field_name="camera_to_world"
    )
    try:
        return CameraGeometry(intrinsics=intrinsics, camera_to_world=camera_to_world)
    except ToolInputError as exc:
        raise ToolInputError(
            "invalid camera geometry: "
            f"intrinsics_path={intrinsics_path}; pose_path={pose_path}; error={exc}"
        ) from exc


__all__ = ["load_camera_geometry", "read_camera_matrix", "read_depth_meters"]
