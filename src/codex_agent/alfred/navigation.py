"""Fast-marching navigation through observed space with robot-body clearance."""

from __future__ import annotations

import importlib
import math
from collections.abc import Collection
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .geometry import GridCell
from .spatial_map import NavigationRaster, SpatialMap

if TYPE_CHECKING:
    from .memory import IncrementalSceneMemory

MOVE_DISTANCE_M = 0.25


def compute_fmm_distances(
    raster: NavigationRaster, goals: Collection[GridCell]
) -> NDArray[np.float64]:
    """Solve Eikonal distances with scikit-fmm; unreachable cells stay infinite.

    A missing dependency fails explicitly. No graph-search substitute is used.
    """
    try:
        backend = importlib.import_module("skfmm")
    except ModuleNotFoundError as exc:
        raise RuntimeError("ALFRED FMM navigation requires scikit-fmm") from exc
    target_indices = [
        index
        for cell in goals
        if raster.contains(cell) and (index := raster.index(cell)) is not None
    ]
    distances = np.full(raster.traversable.shape, np.inf, dtype=np.float64)
    if not target_indices:
        return distances
    level_set = np.ones(raster.traversable.shape, dtype=np.float64)
    for index in target_indices:
        level_set[index] = 0.0
    masked: NDArray[np.float64] = np.ma.MaskedArray(level_set, mask=~raster.traversable)
    # The untyped external numerical result is converted at this boundary.
    solved = backend.distance(masked, dx=raster.resolution_m)
    distances = np.asarray(np.ma.filled(solved, np.inf), dtype=np.float64)
    if distances.shape != raster.traversable.shape or np.isnan(distances).any():
        raise RuntimeError("scikit-fmm returned an invalid distance field")
    distances[~raster.traversable] = np.inf
    for index in target_indices:
        distances[index] = 0.0
    return distances


def rotation_towards(current: float, desired: float) -> str | None:
    """Choose a 90-degree turn toward a heading on ALFRED's discrete lattice."""
    if not math.isfinite(current) or not math.isfinite(desired):
        raise ValueError("current and desired yaw must be finite")
    difference = (desired - current + 180) % 360 - 180
    if abs(difference) <= 45:
        return None
    return "RotateRight" if difference > 0 else "RotateLeft"


def navigation_step(
    spatial_map: SpatialMap, goals: Collection[GridCell], *, yaw_degrees: float
) -> tuple[str | None, str]:
    """Recompute FMM and check one atomic move against inflated known obstacles."""
    if spatial_map.agent_position is None:
        raise ValueError("navigation requires an integrated observation")
    if not math.isfinite(yaw_degrees):
        raise ValueError("yaw_degrees must be finite")
    raster = spatial_map.navigation_raster()
    start = spatial_map.world_to_cell(*spatial_map.agent_position[:2])
    start_index = raster.index(start)
    if start_index is None:
        raise RuntimeError("agent is outside its own navigation raster")
    distances = compute_fmm_distances(raster, goals)
    current_distance = float(distances[start_index])
    if not math.isfinite(current_distance):
        return None, "No reachable stance through observed space with body clearance."
    if start in goals or current_distance <= MOVE_DISTANCE_M * 0.45:
        return (
            None,
            "Reached an observed stance; visibility and interaction readiness remain unverified.",
        )
    step_cells = round(MOVE_DISTANCE_M / raster.resolution_m)
    if step_cells < 1 or not math.isclose(
        step_cells * raster.resolution_m, MOVE_DISTANCE_M
    ):
        raise ValueError("fine_cell_size_m must divide the 0.25 m primitive move")
    candidates: list[tuple[float, float, float]] = []
    position = spatial_map.agent_position
    for offset in (0, 90, 180, 270):
        heading = (yaw_degrees + offset) % 360
        radians = math.radians(heading)
        swept = [
            spatial_map.world_to_cell(
                position[0]
                + MOVE_DISTANCE_M * step / (2 * step_cells) * math.sin(radians),
                position[1]
                + MOVE_DISTANCE_M * step / (2 * step_cells) * math.cos(radians),
            )
            for step in range(1, 2 * step_cells + 1)
        ]
        if not all(raster.contains(cell) for cell in swept):
            continue
        endpoint = raster.index(swept[-1])
        if endpoint is None:
            continue
        remaining = float(distances[endpoint])
        if remaining >= current_distance - raster.resolution_m * 0.25:
            continue
        turn = abs((heading - yaw_degrees + 180) % 360 - 180)
        candidates.append((remaining, turn, heading))
    if not candidates:
        return (
            None,
            "FMM route has no safe 0.25 m action from this stance; acquire another view.",
        )
    _, _, heading = min(candidates)
    return (
        rotation_towards(yaw_degrees, heading) or "MoveAhead",
        "Following the current FMM route through inflated observed space.",
    )


def next_navigation_action(
    memory: IncrementalSceneMemory,
    *,
    yaw_degrees: float,
    horizon_degrees: float,
    object_id: int | None,
    cell: GridCell | None,
    camera_height_m: float = 1.576,
) -> tuple[str | None, str]:
    """Navigate to an observed object's vicinity or a stable coarse region ID."""
    if (object_id is None) == (cell is None):
        raise ValueError("navigation requires exactly one object_id or coarse cell")
    spatial_map = memory.spatial_map
    position = spatial_map.agent_position
    if position is None:
        raise ValueError("navigation requires an integrated observation")
    raster = spatial_map.navigation_raster()
    rows, cols = np.nonzero(raster.traversable)
    available = [
        raster.cell(int(row), int(col)) for row, col in zip(rows, cols, strict=True)
    ]
    target: NDArray[np.float64] | None = None
    if object_id is not None:
        try:
            object_id = memory.resolve_object_id(object_id)
        except KeyError:
            return None, "Requested object has not been observed."
        node = memory.objects[object_id]
        if not node.geometry_valid or object_id == memory.held_object_id:
            return None, "Object location is stale or held; obtain a new observation."
        target = node.centroid
        bounds = node.bounds
        goals = set()
        for candidate in available:
            point = np.asarray(spatial_map.cell_center(candidate))
            nearest = np.clip(point, bounds[0, :2], bounds[1, :2])
            distance = float(np.linalg.norm(point - nearest))
            if 0.35 <= distance <= 0.85:
                goals.add(candidate)
    else:
        assert cell is not None
        goals = {
            candidate
            for candidate in available
            if spatial_map.region_for_cell(candidate) == cell
        }
    action, feedback = navigation_step(spatial_map, goals, yaw_degrees=yaw_degrees)
    if action is not None or not feedback.startswith("Reached") or target is None:
        return action, feedback
    dx, dy = float(target[0] - position[0]), float(target[1] - position[1])
    rotation = rotation_towards(yaw_degrees, math.degrees(math.atan2(dx, dy)))
    if rotation is not None:
        return (
            rotation,
            "Orienting toward the remembered object; reacquire its current mask.",
        )
    desired_pitch = math.degrees(
        math.atan2(camera_height_m - float(target[2]), max(0.01, math.hypot(dx, dy)))
    )
    desired_pitch = max(-30.0, min(60.0, desired_pitch))
    if abs(horizon_degrees - desired_pitch) > 15:
        return (
            "LookDown" if desired_pitch > horizon_degrees else "LookUp"
        ), "Adjusting observation pitch; interaction readiness remains unverified."
    return None, feedback
