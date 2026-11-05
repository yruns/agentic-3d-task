"""FMM distances, safe primitive movement and persistent frontier commitments."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from codex_agent.alfred.exploration import next_exploration_action, suggest_frontiers
from codex_agent.alfred.geometry import Observation
from codex_agent.alfred.memory import IncrementalSceneMemory
from codex_agent.alfred.navigation import (
    compute_fmm_distances,
    navigation_step,
    next_navigation_action,
)
from codex_agent.alfred.object_store import ObjectMemory
from codex_agent.alfred.odometry import ActionOdometry, camera_intrinsics
from codex_agent.alfred.rendering import render_memory_bev
from codex_agent.alfred.spatial_map import NavigationRaster
from codex_agent.tests.test_alfred_spatial_map import floor_map, wall_observation


def test_fmm_respects_blocked_cells_and_disconnected_free_islands() -> None:
    free = np.ones((9, 11), dtype=bool)
    free[:, 5] = False
    raster = NavigationRaster((0, 0), free, 0.05)
    distances = compute_fmm_distances(raster, {(1, 4)})
    assert distances[4, 1] == 0
    assert 0 < distances[4, 4] < 0.25
    assert np.isinf(distances[:, 5:]).all()


def test_fmm_takes_a_detour_around_an_observed_barrier() -> None:
    free = np.ones((15, 15), dtype=bool)
    free[3:12, 7] = False
    raster = NavigationRaster((0, 0), free, 0.05)
    distances = compute_fmm_distances(raster, {(11, 7)})
    assert distances[7, 3] > 0.4
    assert np.isinf(distances[7, 7])
    assert np.isfinite(distances[1, 7])


def test_missing_fmm_dependency_fails_without_an_algorithm_substitute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="requires scikit-fmm"):
        compute_fmm_distances(
            NavigationRaster((0, 0), np.ones((3, 3), dtype=bool), 0.05), {(1, 1)}
        )


def test_navigation_rotates_then_advances_and_replans_after_collision() -> None:
    spatial_map = floor_map()
    goal = spatial_map.world_to_cell(0.0, 1.0)
    action, _ = navigation_step(spatial_map, {goal}, yaw_degrees=90)
    assert action == "RotateLeft"
    action, _ = navigation_step(spatial_map, {goal}, yaw_degrees=0)
    assert action == "MoveAhead"
    spatial_map.record_collision(spatial_map.world_to_cell(0.0, 0.25))
    action, message = navigation_step(spatial_map, {goal}, yaw_degrees=0)
    assert action != "MoveAhead"
    assert message


def test_frontier_target_remains_locked_across_navigation_steps(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path)
    memory.spatial_map = floor_map()
    goal = memory.spatial_map.world_to_cell(0.75, 0.0)
    memory.spatial_map.exploration.goal = goal
    action, _ = next_exploration_action(memory, yaw_degrees=0, horizon_degrees=30)
    assert action == "RotateRight"
    assert memory.spatial_map.exploration.goal == goal
    action, _ = next_exploration_action(memory, yaw_degrees=90, horizon_degrees=30)
    assert action == "MoveAhead"
    assert memory.spatial_map.exploration.goal == goal


def test_object_navigation_resolves_retired_identity_aliases(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path)
    memory.spatial_map = floor_map()
    memory.objects[0] = ObjectMemory(
        object_id=0,
        category="Mug",
        points=np.array([[0.0, 1.0, 0.8]]),
        colors=np.array([[1.0, 0.0, 0.0]]),
        image_feature=np.array([1.0, 0.0]),
        text_feature=np.array([1.0, 0.0]),
        first_seen=0,
        last_seen=0,
    )
    memory.aliases[7] = 0
    actual = next_navigation_action(
        memory, yaw_degrees=0, horizon_degrees=30, object_id=7, cell=None
    )
    expected = next_navigation_action(
        memory, yaw_degrees=0, horizon_degrees=30, object_id=0, cell=None
    )
    assert actual == expected
    assert actual[0] == "MoveAhead"


def test_frontiers_are_reachable_and_revisits_reduce_priority(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path)
    memory.spatial_map = floor_map()
    candidates = suggest_frontiers(memory)
    assert candidates
    first = candidates[0]
    assert first.information_gain_m2 > 0
    assert memory.spatial_map.navigation_raster().contains(first.stance_cell)
    memory.spatial_map.exploration.attempts[first.region] = 3
    assert all(
        candidate.region != first.region
        for candidate in suggest_frontiers(memory, limit=100)
    )


def test_four_sixty_degree_views_enable_motion_without_claiming_blind_gaps_seen(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path)
    odometry = ActionOdometry()
    for index, view_yaw in enumerate((0, 90, 180, 270)):
        observation = Observation(
            frame_id=index,
            rgb_path=tmp_path / "unused.png",
            depth_m=np.full((41, 41), 2.0),
            intrinsics=camera_intrinsics(width=41, height=41),
            camera_to_world=odometry.camera_to_world(
                yaw_degrees=view_yaw, horizon_degrees=0
            ),
            agent_position=(0.0, 0.0, 0.0),
            detections=(),
            frame_feature=np.array([1.0, 0.0]),
        )
        memory.spatial_map.update(observation)
    blind = memory.spatial_map.world_to_cell(0.75, 0.75)
    assert blind not in memory.spatial_map.seen_cells
    assert memory.spatial_map.cell_state(blind) == "unknown"
    assert not memory.spatial_map.navigation_raster().contains(blind)
    candidates = suggest_frontiers(memory)
    assert candidates and candidates[0].path_cost_m > 0
    yaw = 0.0
    for _ in range(4):
        action, feedback = next_exploration_action(
            memory, yaw_degrees=yaw, horizon_degrees=30
        )
        assert action is not None
        if action == "MoveAhead":
            assert "locked frontier" in feedback
            break
        assert action in {"RotateLeft", "RotateRight"}
        yaw = (yaw + (90 if action == "RotateRight" else -90)) % 360
    else:
        pytest.fail("frontier exploration never produced a physical move")
    assert blind not in memory.spatial_map.seen_cells


def test_no_reachable_frontier_has_a_bounded_scan_not_endless_noops(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path)
    observation = wall_observation()
    memory.spatial_map.update(
        observation, ignored_mask=np.ones_like(observation.depth_m, dtype=bool)
    )
    coverage = next(iter(memory.spatial_map.coverage.values()))
    coverage.direction_bins.update({0, 2, 4, 6})
    actions = [
        next_exploration_action(memory, yaw_degrees=0, horizon_degrees=30)[0]
        for _ in range(5)
    ]
    assert actions == ["RotateRight"] * 4 + [None]


def test_bev_renders_occupied_unknown_and_free_geometry(tmp_path: Path) -> None:
    from PIL import Image

    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.spatial_map = floor_map()
    memory.spatial_map.record_collision(memory.spatial_map.world_to_cell(0.0, 1.0))
    path = render_memory_bev(memory, tmp_path / "bev.png")
    with Image.open(path) as image:
        colors = set(image.convert("RGB").getdata())
    assert (255, 255, 255) in colors
    assert (66, 75, 88) in colors
    assert (203, 208, 213) in colors
