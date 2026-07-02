"""3D->2D visibility projection for SceneFun3D anchor-driven frame selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from codex_agent.scenefunc3d.backends.anchor import TargetAnchor
from codex_agent.scenefunc3d.backends.lift_3d import (
    BoolArray,
    CameraGeometry,
    FloatArray,
    _validate_points_world,
)

if TYPE_CHECKING:
    from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene

_DEFAULT_DEPTH_TOLERANCE = 0.25


@dataclass(frozen=True)
class FrameVisibility:
    """Anchor visibility of one frame."""

    frame_id: str
    visible: bool
    unoccluded_fraction: float
    centeredness: float
    quality_score: float


@dataclass(frozen=True)
class FrameCamera:
    """One frame's in-memory camera + depth, for visibility scoring."""

    frame_id: str
    geometry: CameraGeometry
    depth_meters: FloatArray


def project_world_to_pixels(
    points_world: FloatArray, geometry: CameraGeometry
) -> tuple[FloatArray, FloatArray]:
    """Project world points into one camera; return (pixels_uv (N,2), camera_z (N,)).

    Inverse of ``lift_3d.backproject_mask_to_world``. Pixel u/v are NaN for
    points at or behind the image plane (``camera_z <= 0``); callers must gate on
    ``camera_z > 0`` before using pixels.
    """
    points = _validate_points_world(points_world, field_name="points_world")
    world_to_camera = np.linalg.inv(geometry.camera_to_world)
    homogeneous = np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))
    camera = homogeneous @ world_to_camera.T
    camera_xyz = camera[:, :3]
    camera_z = camera_xyz[:, 2]
    fx = geometry.intrinsics[0, 0]
    fy = geometry.intrinsics[1, 1]
    cx = geometry.intrinsics[0, 2]
    cy = geometry.intrinsics[1, 2]
    safe_z = np.where(camera_z > 0.0, camera_z, np.nan)
    u = fx * camera_xyz[:, 0] / safe_z + cx
    v = fy * camera_xyz[:, 1] / safe_z + cy
    pixels: FloatArray = np.column_stack((u, v)).astype(np.float64)
    return pixels, camera_z.astype(np.float64)


def point_visibility(
    points_world: FloatArray,
    geometry: CameraGeometry,
    depth_meters: FloatArray,
    *,
    depth_tolerance: float = _DEFAULT_DEPTH_TOLERANCE,
) -> BoolArray:
    """Per-point un-occluded visibility in one frame.

    Visible = in front of camera, projects inside the image, and camera-space
    depth matches the observed depth within ``depth_tolerance`` (relative).
    """
    depth = np.asarray(depth_meters, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth_meters must be 2D; got shape {depth.shape}")
    if not np.isfinite(depth_tolerance) or depth_tolerance < 0.0:
        raise ValueError(
            f"depth_tolerance must be finite and non-negative: {depth_tolerance!r}"
        )
    pixels, camera_z = project_world_to_pixels(points_world, geometry)
    height, width = int(depth.shape[0]), int(depth.shape[1])
    # Non-finite pixels (points behind the camera) are excluded by ``in_bounds``
    # below; zero them first so the int cast does not emit a NaN-cast warning.
    finite_pixels = np.where(np.isfinite(pixels), pixels, 0.0)
    columns = np.floor(finite_pixels[:, 0]).astype(np.int64, copy=False)
    rows = np.floor(finite_pixels[:, 1]).astype(np.int64, copy=False)
    in_bounds = (
        (camera_z > 0.0)
        & np.isfinite(pixels[:, 0])
        & np.isfinite(pixels[:, 1])
        & (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < height)
    )
    safe_rows = np.where(in_bounds, rows, 0)
    safe_columns = np.where(in_bounds, columns, 0)
    observed = depth[safe_rows, safe_columns]
    valid_observed = np.isfinite(observed) & (observed > 0.0)
    occlusion_ok = np.abs(camera_z - observed) <= depth_tolerance * observed
    visible: BoolArray = (in_bounds & valid_observed & occlusion_ok).astype(np.bool_)
    return visible


def _anchor_probe_points(anchor: TargetAnchor) -> FloatArray:
    center = np.asarray(anchor.centroid, dtype=np.float64)
    radius = anchor.radius_m
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=np.float64,
    )
    return (center[np.newaxis, :] + offsets).astype(np.float64)


def _centeredness(
    anchor: TargetAnchor, geometry: CameraGeometry, depth_meters: FloatArray
) -> float:
    height, width = int(depth_meters.shape[0]), int(depth_meters.shape[1])
    pixels, _camera_z = project_world_to_pixels(
        np.asarray([anchor.centroid], dtype=np.float64), geometry
    )
    u, v = float(pixels[0, 0]), float(pixels[0, 1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return 0.0
    center_u, center_v = width / 2.0, height / 2.0
    distance = float(np.hypot(u - center_u, v - center_v))
    half_diagonal = float(np.hypot(center_u, center_v))
    if half_diagonal <= 0.0:
        return 0.0
    return max(0.0, 1.0 - distance / half_diagonal)


def score_anchor_visibility(
    frame_id: str,
    anchor: TargetAnchor,
    geometry: CameraGeometry,
    depth_meters: FloatArray,
    *,
    depth_tolerance: float = _DEFAULT_DEPTH_TOLERANCE,
) -> FrameVisibility:
    """Score how well one frame sees the anchor (probe visibility x centeredness)."""
    probes = _anchor_probe_points(anchor)
    visible = point_visibility(
        probes, geometry, depth_meters, depth_tolerance=depth_tolerance
    )
    if not bool(visible[0]):
        return FrameVisibility(
            frame_id=frame_id,
            visible=False,
            unoccluded_fraction=0.0,
            centeredness=0.0,
            quality_score=0.0,
        )
    unoccluded_fraction = float(np.mean(visible))
    centeredness = _centeredness(anchor, geometry, depth_meters)
    return FrameVisibility(
        frame_id=frame_id,
        visible=True,
        unoccluded_fraction=unoccluded_fraction,
        centeredness=centeredness,
        quality_score=unoccluded_fraction * centeredness,
    )


def select_visible_frames(
    anchor: TargetAnchor,
    cameras: Sequence[FrameCamera],
    *,
    frame_cap: int,
    depth_tolerance: float = _DEFAULT_DEPTH_TOLERANCE,
) -> tuple[FrameVisibility, ...]:
    """Score all frames, keep the ones that see the anchor, return the top N.

    Ties break deterministically by ascending ``frame_id``.
    """
    if frame_cap <= 0:
        raise ValueError(f"frame_cap must be positive: {frame_cap!r}")
    scored = [
        score_anchor_visibility(
            camera.frame_id,
            anchor,
            camera.geometry,
            camera.depth_meters,
            depth_tolerance=depth_tolerance,
        )
        for camera in cameras
    ]
    visible = [result for result in scored if result.visible]
    visible.sort(key=lambda result: (-result.quality_score, result.frame_id))
    return tuple(visible[:frame_cap])


def select_scene_visible_frames(
    anchor: TargetAnchor,
    scene: SceneFunc3dToolScene,
    *,
    frame_cap: int,
    depth_tolerance: float = _DEFAULT_DEPTH_TOLERANCE,
) -> tuple[FrameVisibility, ...]:
    """Stream a scene's frames, score anchor visibility, return top-N + seed.

    Depth is read one frame at a time and discarded to bound memory. The
    anchor's ``seed_frame_id`` is included on a best-effort basis (when it is
    among the scene's geometry-complete frames) so a fully-occluded scene
    degrades to the single-frame seed rather than an empty selection.
    """
    if frame_cap <= 0:
        raise ValueError(f"frame_cap must be positive: {frame_cap!r}")

    from codex_agent.scenefunc3d.backends.frame_loader import (
        iter_frame_geometry,
        read_frame_depth,
    )

    scored: list[FrameVisibility] = []
    for frame_id, geometry in iter_frame_geometry(scene):
        depth = read_frame_depth(scene, frame_id)
        scored.append(
            score_anchor_visibility(
                frame_id, anchor, geometry, depth, depth_tolerance=depth_tolerance
            )
        )

    visible = [result for result in scored if result.visible]
    visible.sort(key=lambda result: (-result.quality_score, result.frame_id))
    selected = list(visible[:frame_cap])
    selected_ids = {result.frame_id for result in selected}

    if anchor.seed_frame_id not in selected_ids:
        seed_score = next(
            (r for r in scored if r.frame_id == anchor.seed_frame_id), None
        )
        # Seed inclusion is best-effort: only frames that were streamed (i.e. have
        # complete geometry) can be scored/lifted. A seed with no complete geometry
        # (or a synthetic Tier-2 seed id) is simply not force-included.
        if seed_score is not None:
            selected.append(seed_score)
    return tuple(selected)


def count_vertex_visibility(
    vertex_ids: Sequence[int],
    vertex_coords: FloatArray,
    scene: SceneFunc3dToolScene,
    frame_ids: Sequence[str],
    *,
    depth_tolerance: float = _DEFAULT_DEPTH_TOLERANCE,
) -> dict[int, int]:
    """Count, per candidate vertex, how many ``frame_ids`` see it un-occluded.

    Reads each frame's depth once (streamed, discarded) and tests all vertices
    against it. ``vertex_ids[k]`` labels row ``k`` of ``vertex_coords``.
    """
    coords = np.asarray(vertex_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"vertex_coords must have shape (N, 3): {coords.shape}")
    if len(vertex_ids) != coords.shape[0]:
        raise ValueError(
            "vertex_ids length must match vertex_coords rows: "
            f"ids={len(vertex_ids)}; rows={coords.shape[0]}"
        )
    if coords.shape[0] == 0:
        return {}

    from codex_agent.scenefunc3d.backends.frame_loader import (
        load_frame_geometry,
        read_frame_depth,
    )

    int_ids = [int(vertex_id) for vertex_id in vertex_ids]
    counts = dict.fromkeys(int_ids, 0)
    for frame_id in frame_ids:
        geometry = load_frame_geometry(scene, frame_id)
        depth = read_frame_depth(scene, frame_id)
        visible = point_visibility(
            coords, geometry, depth, depth_tolerance=depth_tolerance
        )
        for row, vertex_id in enumerate(int_ids):
            if bool(visible[row]):
                counts[vertex_id] += 1
    return counts


__all__ = [
    "FrameCamera",
    "FrameVisibility",
    "count_vertex_visibility",
    "point_visibility",
    "project_world_to_pixels",
    "score_anchor_visibility",
    "select_scene_visible_frames",
    "select_visible_frames",
]
