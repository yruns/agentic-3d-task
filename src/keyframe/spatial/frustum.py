"""Camera-frustum overlap estimation for pose-aware keyframe selection.

Two estimators are provided:

- :func:`frustum_overlap_l1` uses pose geometry only (cheap, but
  underestimates parallax for pure forward/backward dolly motion).
- :func:`frustum_overlap_l2` reprojects valid depth pixels into the other
  camera (accurate, requires a depth map).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# ScanNet color images are a fixed 1296x968 across the corpus.
_SCANNET_IMAGE_WH: tuple[int, int] = (1296, 968)


def load_scene_intrinsic(
    scene_raw_dir: Path,
) -> tuple[NDArray[np.float64], tuple[int, int]]:
    """Load the 3x3 color intrinsic and image (W, H) from raw ScanNet data."""
    intrinsic_path = scene_raw_dir / "intrinsic_color.txt"
    if not intrinsic_path.exists():
        raise FileNotFoundError(intrinsic_path)

    intrinsic = np.loadtxt(intrinsic_path, dtype=np.float64)
    if intrinsic.shape != (4, 4):
        raise ValueError(f"intrinsic_color.txt must be 4x4, got {intrinsic.shape}")

    k = intrinsic[:3, :3]
    if not np.isfinite(k).all():
        raise ValueError("intrinsic matrix contains non-finite values")
    if np.allclose(k, 0.0) or np.any(np.isclose(np.diag(k), 0.0)):
        raise ValueError("intrinsic matrix is degenerate")
    return k, _SCANNET_IMAGE_WH


def frustum_overlap_l1(
    pose_a: NDArray[np.float64],
    pose_b: NDArray[np.float64],
    k: NDArray[np.float64],
    img_wh: tuple[int, int],
    scene_depth: float = 2.0,
) -> float:
    """Approximate view overlap in [0, 1] using pose geometry only."""
    pose_a = _validate_pose(pose_a, "pose_a")
    pose_b = _validate_pose(pose_b, "pose_b")
    fx, _fy = _validate_intrinsic(k)
    width, _height = _validate_img_wh(img_wh)
    if scene_depth <= 0:
        raise ValueError("scene_depth must be positive")

    fov_h = 2.0 * np.arctan(width / (2.0 * fx))
    forward_a = _normalize(-pose_a[:3, 2], "pose_a forward")
    forward_b = _normalize(-pose_b[:3, 2], "pose_b forward")
    origin_a = pose_a[:3, 3]
    origin_b = pose_b[:3, 3]

    cos_theta = float(np.clip(np.dot(forward_a, forward_b), -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    angular_drop = min(1.0, (theta / np.pi) ** 1.5)

    delta = origin_b - origin_a
    lateral = delta - np.dot(delta, forward_a) * forward_a
    d_perp = float(np.linalg.norm(lateral))
    half_w_at_depth = float(scene_depth * np.tan(fov_h / 2.0))
    if half_w_at_depth <= 0:
        raise ValueError("horizontal frustum half-width is non-positive")
    parallax_drop = min(1.0, d_perp / half_w_at_depth)

    overlap = max(
        0.0, 1.0 - angular_drop - parallax_drop + angular_drop * parallax_drop
    )
    if np.isclose(overlap, 0.0, atol=1e-7):
        return 0.0
    if np.isclose(overlap, 1.0, atol=1e-7):
        return 1.0
    return float(overlap)


def frustum_overlap_l2(
    depth_a: NDArray[np.float64],
    pose_a: NDArray[np.float64],
    pose_b: NDArray[np.float64],
    k: NDArray[np.float64],
    img_wh: tuple[int, int],
    subsample: int = 8,
    depth_min: float = 0.1,
    depth_max: float = 10.0,
) -> float:
    """Estimate overlap in [0, 1] by reprojecting depth pixels of A into B."""
    if subsample <= 0:
        raise ValueError("subsample must be positive")
    if depth_min <= 0:
        raise ValueError("depth_min must be positive")
    if depth_max <= depth_min:
        raise ValueError("depth_max must be greater than depth_min")

    pose_a = _validate_pose(pose_a, "pose_a")
    pose_b = _validate_pose(pose_b, "pose_b")
    _validate_intrinsic(k)
    width, height = _validate_img_wh(img_wh)

    depth = np.asarray(depth_a, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth_a must be HxW, got shape {depth.shape}")
    if not np.isfinite(depth).all():
        raise ValueError("depth_a contains non-finite values")

    depth_h, depth_w = depth.shape
    if depth_h == 0 or depth_w == 0:
        raise ValueError("depth_a is empty")
    if not np.any((depth > depth_min) & (depth < depth_max)):
        raise ValueError("depth_a has no valid depth samples")

    effective_k = _rescale_intrinsic(k, (width, height), (depth_w, depth_h))
    fx = effective_k[0, 0]
    fy = effective_k[1, 1]
    cx = effective_k[0, 2]
    cy = effective_k[1, 2]

    us = np.arange(0, depth_w, subsample, dtype=np.float64)
    vs = np.arange(0, depth_h, subsample, dtype=np.float64)
    grid_u, grid_v = np.meshgrid(us, vs, indexing="xy")

    sampled_depth = depth[grid_v.astype(np.int64), grid_u.astype(np.int64)]
    sampled_valid = (sampled_depth > depth_min) & (sampled_depth < depth_max)
    if not np.any(sampled_valid):
        raise ValueError("depth_a has no valid depth samples after subsampling")

    u = grid_u[sampled_valid]
    v = grid_v[sampled_valid]
    z = sampled_depth[sampled_valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_cam_a = np.stack([x, y, z, np.ones_like(z)], axis=1)
    points_world = (pose_a @ points_cam_a.T).T

    points_cam_b = (np.linalg.inv(pose_b) @ points_world.T).T
    z_b = points_cam_b[:, 2]
    x_b = points_cam_b[:, 0]
    y_b = points_cam_b[:, 1]
    valid_b = z_b > depth_min
    if not np.any(valid_b):
        return 0.0

    u_b = fx * x_b / z_b + cx
    v_b = fy * y_b / z_b + cy
    in_frustum = (
        valid_b & (u_b >= 0.0) & (u_b < depth_w) & (v_b >= 0.0) & (v_b < depth_h)
    )
    return float(np.count_nonzero(in_frustum) / np.count_nonzero(sampled_valid))


def _validate_pose(pose: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} contains non-finite values")
    return pose


def _validate_intrinsic(k: NDArray[np.float64]) -> tuple[float, float]:
    intrinsic = np.asarray(k, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"k must be 3x3, got {intrinsic.shape}")
    if not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic matrix contains non-finite values")
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    if np.isclose(fx, 0.0) or np.isclose(fy, 0.0):
        raise ValueError("intrinsic matrix is degenerate")
    return fx, fy


def _validate_img_wh(img_wh: tuple[int, int]) -> tuple[int, int]:
    width, height = img_wh
    if width <= 0 or height <= 0:
        raise ValueError(f"img_wh must be positive, got {img_wh}")
    return int(width), int(height)


def _normalize(vector: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"{name} is degenerate")
    return vector / norm


def _rescale_intrinsic(
    k: NDArray[np.float64],
    original_wh: tuple[int, int],
    target_wh: tuple[int, int],
) -> NDArray[np.float64]:
    original_w, original_h = original_wh
    target_w, target_h = target_wh
    scale_x = target_w / original_w
    scale_y = target_h / original_h

    scaled = np.asarray(k, dtype=np.float64).copy()
    scaled[0, 0] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[0, 2] *= scale_x
    scaled[1, 2] *= scale_y
    return scaled  # type: ignore[no-any-return]
