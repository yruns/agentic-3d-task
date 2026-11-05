"""Metric occupancy, visibility and robot-clearance regressions using RGB-D rays."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from codex_agent.alfred.geometry import Observation
from codex_agent.alfred.spatial_map import SpatialMap, SpatialMapConfig


def wall_observation(frame_id: int = 0) -> Observation:
    """A synthetic wall two metres ahead, without semantic detections."""
    pose = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 1.0], [0, 0, 0, 1]], dtype=float
    )
    return Observation(
        frame_id,
        Path("unused-rgb.png"),
        np.full((21, 21), 2.0),
        np.array([[20, 0, 10], [0, 20, 10], [0, 0, 1.0]]),
        pose,
        (0.0, 0.0, 0.0),
        (),
        np.array([1.0, 0.0]),
    )


def floor_map() -> SpatialMap:
    """A metric floor patch reconstructed from a synthetic downward RGB-D view."""
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 2.0
    observation = Observation(
        0,
        Path("unused-rgb.png"),
        np.full((41, 41), 2.0),
        np.array([[20, 0, 20], [0, 20, 20], [0, 0, 1.0]]),
        pose,
        (0.0, 0.0, 0.0),
        (),
        np.array([1.0, 0.0]),
    )
    spatial_map = SpatialMap()
    spatial_map.update(observation)
    return spatial_map


def test_depth_without_detections_marks_wall_and_never_sees_behind_it() -> None:
    spatial_map = SpatialMap()
    regions = spatial_map.update(wall_observation())
    assert spatial_map.cell_state(spatial_map.world_to_cell(0.0, 2.0)) == "occupied"
    assert spatial_map.cell_state(spatial_map.world_to_cell(0.0, 3.0)) == "unknown"
    assert spatial_map.world_to_cell(0.0, 3.0) not in spatial_map.seen_cells
    assert spatial_map.world_to_region(0.0, 2.0) in regions
    assert spatial_map.coverage[spatial_map.world_to_region(0.0, 2.0)].observations == 1


def test_semantic_region_and_fine_cell_ids_stay_fixed_when_map_expands() -> None:
    spatial_map = SpatialMap()
    assert spatial_map.world_to_cell(-0.01, -0.01) == (-1, -1)
    assert spatial_map.world_to_region(-0.01, -0.01) == (-1, -1)
    assert spatial_map.world_to_cell(-0.25, -4e-17) == (-5, 0)
    cell, region = spatial_map.world_to_cell(0.31, 1.12), spatial_map.world_to_region(
        0.31, 1.12
    )
    spatial_map.update(wall_observation())
    moved = wall_observation(1)
    pose = moved.camera_to_world.copy()
    pose[0, 3] = -5.0
    spatial_map.update(
        replace(moved, camera_to_world=pose, agent_position=(-5.0, 0.0, 0.0))
    )
    assert spatial_map.world_to_cell(0.31, 1.12) == cell
    assert spatial_map.world_to_region(0.31, 1.12) == region


def test_overhead_visibility_does_not_prove_ground_traversability() -> None:
    observation = wall_observation()
    pose = observation.camera_to_world.copy()
    pose[2, 3] = 4.0
    spatial_map = SpatialMap()
    spatial_map.update(replace(observation, camera_to_world=pose))
    cell = spatial_map.world_to_cell(0.0, 1.0)
    assert cell in spatial_map.seen_cells
    assert spatial_map.cell_state(cell) == "unknown"


def test_held_mask_does_not_leave_obstacles_and_collisions_persist() -> None:
    spatial_map = SpatialMap()
    observation = wall_observation()
    spatial_map.update(observation, ignored_mask=np.ones((21, 21), dtype=bool))
    assert all(
        spatial_map.cell_state(cell) != "occupied" for cell in spatial_map.known_cells
    )
    blocked = spatial_map.world_to_cell(0.0, 1.0)
    spatial_map.record_collision(blocked)
    spatial_map.update(wall_observation(1))
    assert spatial_map.cell_state(blocked) == "occupied"
    region = spatial_map.region_for_cell(blocked)
    old_coverage = spatial_map.coverage[region].observations
    spatial_map.invalidate_region({region})
    assert spatial_map.cell_state(blocked) == "unknown"
    assert spatial_map.coverage[region].observations == old_coverage


def test_robot_footprint_inflates_obstacles_and_unknown_boundaries() -> None:
    spatial_map = floor_map()
    center = spatial_map.world_to_cell(0.0, 1.0)
    spatial_map.record_collision(center)
    raster = spatial_map.navigation_raster()
    assert not raster.contains(center)
    assert not raster.contains((center[0] + 3, center[1]))
    assert not raster.contains(spatial_map.world_to_cell(4.0, 0.0))
    assert raster.contains(spatial_map.world_to_cell(0.0, 0.0))


def test_rejects_inconsistent_grid_scale_and_repeated_frames() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        SpatialMapConfig(fine_cell_size_m=0.07)
    spatial_map = SpatialMap()
    spatial_map.update(wall_observation())
    with pytest.raises(ValueError, match="monotonically"):
        spatial_map.update(wall_observation())


def test_adaptive_floor_calibration_marks_elevated_planar_floor_free() -> None:
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 2.0
    # Simulate 8.5% monocular depth underestimation: camera at z=2.0m measures
    # depth=1.83m on a flat floor, placing raw z at +0.17m (>0.12m obstacle threshold).
    observation = Observation(
        0,
        Path("unused-rgb.png"),
        np.full((81, 81), 1.83),
        np.array([[20, 0, 40], [0, 20, 40], [0, 0, 1.0]]),
        pose,
        (0.0, 0.0, 0.0),
        (),
        np.array([1.0, 0.0]),
    )
    spatial_map = SpatialMap()
    spatial_map.update(observation)
    assert spatial_map.cell_state(spatial_map.world_to_cell(0.0, 0.5)) == "free"
