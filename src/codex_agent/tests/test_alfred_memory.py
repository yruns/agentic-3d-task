"""Synthetic RGB-D regressions for incremental, observation-only ALFRED memory."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from codex_agent.alfred.memory import (
    Detection,
    IncrementalSceneMemory,
    MemoryConfig,
    Observation,
)


def _observation(
    root: Path,
    frame_id: int,
    *,
    translation_x: float = 0.0,
    category: str = "Mug",
    color: tuple[int, int, int] = (200, 30, 20),
    with_detection: bool = True,
) -> Observation:
    rgb_path = root / f"rgb_{frame_id}.png"
    Image.new("RGB", (8, 8), color).save(rgb_path)
    pose = np.array(
        [[1, 0, 0, translation_x], [0, 0, 1, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]],
        dtype=float,
    )
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    return Observation(
        frame_id=frame_id,
        rgb_path=rgb_path,
        depth_m=np.full((8, 8), 2.0),
        intrinsics=np.array([[20, 0, 3.5], [0, 20, 3.5], [0, 0, 1.0]]),
        camera_to_world=pose,
        agent_position=(translation_x, 0.0, 0.0),
        detections=(Detection(category, mask),) if with_detection else (),
    )


def test_projects_observed_masks_into_z_up_world_and_saves_sources(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    frame = memory.integrate(_observation(tmp_path, 0))
    node = memory.objects[0]
    np.testing.assert_allclose(node.centroid, [0.0, 2.0, 1.5], atol=0.01)
    assert memory.current_detection_objects == {0: 0}
    assert memory.object_to_views[0] == {0}
    assert memory.view_to_objects[0] == {0}
    assert frame.rgb_path.is_file()
    assert frame.geometry_path.is_file()
    index = json.loads(frame.geometry_path.with_suffix(".json").read_text())
    assert index["detection_object_ids"] == {"0": 0}
    assert index["object_ids"] == [0]
    assert {tuple(cell) for cell in index["cells"]} == memory.view_to_grids[0]
    with np.load(frame.geometry_path) as saved:
        assert saved["depth_m"].shape == (8, 8)
        assert saved["masks"].shape == (1, 8, 8)
    exported = memory.to_scene_objects()
    assert exported[0].category == "Mug"
    assert exported[0].image_idx == [0]


def test_repeated_views_fuse_but_separated_same_class_objects_remain_distinct(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0))
    memory.integrate(_observation(tmp_path, 1, translation_x=0.05))
    assert len(memory.objects) == 1
    assert memory.object_to_views[0] == {0, 1}
    memory.integrate(_observation(tmp_path, 2, translation_x=2.0))
    assert len(memory.objects) == 2
    assert memory.current_detection_objects == {0: 1}


def test_same_geometry_different_appearance_does_not_merge(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0))
    memory.integrate(_observation(tmp_path, 1, color=(0, 200, 255)))
    assert len(memory.objects) == 2


def test_cell_ids_do_not_shift_when_observed_extent_expands(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    cell = memory.world_to_cell(0.1, 1.1)
    memory.integrate(_observation(tmp_path, 0))
    memory.integrate(_observation(tmp_path, 1, translation_x=-8.0))
    assert memory.world_to_cell(0.1, 1.1) == cell
    assert memory.world_to_cell(-0.01, -0.01) == (-1, -1)


def test_observation_coverage_is_not_navigation_free_space(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory", MemoryConfig(depth_stride=1))
    memory.integrate(_observation(tmp_path, 0, with_detection=False))
    assert memory.cell_state(memory.world_to_cell(0.0, 0.0)) == "free"
    assert memory.cell_state(memory.world_to_cell(0.0, 1.0)) == "unknown"
    assert memory.world_to_cell(0.0, 1.0) in memory.grid_to_views
    assert memory.cell_state((1000, 1000)) == "unknown"
    assert not memory.objects


def test_failed_interaction_does_not_invalidate_but_successful_pickup_does(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0))
    memory.record_interaction("PickupObject", 0, success=False)
    assert memory.objects[0].geometry_valid
    memory.record_interaction("PickupObject", 0, success=True)
    assert not memory.objects[0].geometry_valid
    assert memory.held_object_id == 0
    assert memory.retrieve_frames(object_ids=(0,)) == ()
    assert memory.retrieve_frames(object_ids=(0,), include_stale=True)[0].frame_id == 0
    assert memory.to_scene_objects() == []


def test_open_invalidates_history_until_a_new_observation(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0, category="Cabinet"))
    memory.record_interaction("OpenObject", 0, success=True)
    memory.integrate(_observation(tmp_path, 1, category="Cabinet"))
    assert len(memory.objects) == 1
    assert memory.objects[0].geometry_valid
    assert [frame.frame_id for frame in memory.retrieve_frames(object_ids=(0,))] == [1]


def test_visible_held_instance_does_not_create_a_stationary_duplicate(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0))
    memory.record_interaction("PickupObject", 0, success=True)
    carried = _observation(tmp_path, 1, translation_x=1.0)
    carried = replace(carried, depth_m=np.full((8, 8), 0.4))
    memory.integrate(carried)
    assert len(memory.objects) == 1
    assert memory.current_detection_objects == {0: 0}
    assert memory.objects[0].state == "held"
    assert not memory.objects[0].geometry_valid
    assert not memory.to_scene_objects()


def test_put_invalidates_held_object_and_receptacle(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0))
    memory.record_interaction("PickupObject", 0, success=True)
    memory.integrate(_observation(tmp_path, 1, category="CounterTop"))
    memory.record_interaction("PutObject", 1, success=True)
    assert memory.held_object_id is None
    assert not memory.objects[0].geometry_valid
    assert not memory.objects[1].geometry_valid


def test_retrieval_retains_distinct_requested_regions(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0))
    memory.integrate(_observation(tmp_path, 1, translation_x=3.0))
    result = memory.retrieve_frames(object_ids=(0, 1), limit=2)
    assert {frame.frame_id for frame in result} == {0, 1}
    assert memory.retrieve_frames(cells=((999, 999),)) == ()


def test_memory_rejects_repeated_or_out_of_order_frame_ids(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 2))
    with pytest.raises(ValueError, match="frame_id"):
        memory.integrate(_observation(tmp_path, 1))


def test_bound_point_cloud_and_frame_catalog_and_render(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(
        tmp_path / "memory", MemoryConfig(max_points_per_object=8, max_object_views=2)
    )
    for frame_id in range(4):
        memory.integrate(_observation(tmp_path, frame_id))
    assert len(memory.objects[0].points) <= 8
    assert len(memory.to_scene_objects()[0].image_idx) <= 2
    snapshot = memory.snapshot()
    assert snapshot["frame_count"] == 4
    assert snapshot["coordinate_system"] == "world_z_up"
    assert memory.render_bev(tmp_path / "bev.png").is_file()


@pytest.mark.parametrize("field", ["depth_m", "intrinsics", "camera_to_world"])
def test_observation_rejects_nonfinite_geometry(tmp_path: Path, field: str) -> None:
    from dataclasses import replace

    observation = _observation(tmp_path, 0)
    broken = getattr(observation, field).copy()
    broken.flat[0] = np.nan
    with pytest.raises(ValueError, match=field):
        replace(observation, **{field: broken})


def test_detection_rejects_nonbinary_mask() -> None:
    with pytest.raises(ValueError, match="mask"):
        Detection("Mug", np.full((2, 2), 0.3))


def test_unknown_requested_object_is_an_error(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    with pytest.raises(KeyError, match="object_id"):
        memory.retrieve_frames(object_ids=(17,))


def _surface_observation(
    root: Path, frame_id: int, *, floor: bool, offset_x: float = 0.0
) -> Observation:
    """Observe a narrow region at y=1.6 m as either an obstacle or the floor."""
    observation = _observation(root, frame_id, with_detection=False)
    pose = observation.camera_to_world.copy()
    pose[0, 3] = offset_x + 0.125
    distance = 1.6
    if floor:
        pitch = math.atan2(1.5, 1.6)
        pose[:3, 1] = (0.0, -math.sin(pitch), -math.cos(pitch))
        pose[:3, 2] = (0.0, math.cos(pitch), -math.sin(pitch))
        distance = math.hypot(1.5, 1.6)
    return replace(
        observation,
        depth_m=np.full((8, 8), distance),
        intrinsics=np.array([[10000, 0, 3.5], [0, 10000, 3.5], [0, 0, 1.0]]),
        camera_to_world=pose,
        agent_position=(offset_x, 0.0, 0.0),
    )


def test_current_floor_evidence_clears_old_perceptual_obstacle(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_surface_observation(tmp_path, 0, floor=False))
    assert memory.cell_state((0, 6)) == "occupied"
    memory.integrate(_surface_observation(tmp_path, 1, floor=True))
    assert memory.cell_state((0, 6)) == "free"


def test_floor_update_preserves_obstacles_outside_current_observation(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_surface_observation(tmp_path, 0, floor=False, offset_x=2.0))
    memory.integrate(_surface_observation(tmp_path, 1, floor=False))
    memory.integrate(_surface_observation(tmp_path, 2, floor=True))
    assert memory.cell_state((0, 6)) == "free"
    assert memory.cell_state((8, 6)) == "occupied"


def test_collision_survives_predicted_floor_until_actual_traversal(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_surface_observation(tmp_path, 0, floor=False))
    memory.record_collision((0, 6))
    memory.integrate(_surface_observation(tmp_path, 1, floor=True))
    assert memory.cell_state((0, 6)) == "occupied"
    assert memory.collision_cells == {(0, 6)}
    assert memory.snapshot()["collision_cells"] == [(0, 6)]
    traversed = _surface_observation(tmp_path, 2, floor=True)
    pose = traversed.camera_to_world.copy()
    pose[:2, 3] = (0.125, 1.625)
    memory.integrate(
        replace(traversed, agent_position=(0.125, 1.625, 0.0), camera_to_world=pose)
    )
    assert memory.cell_state((0, 6)) == "free"
    assert not memory.collision_cells


def test_opening_object_invalidates_collision_evidence_in_its_old_region(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0, category="Cabinet"))
    cell = memory.world_to_cell(*memory.objects[0].points[0, :2])
    memory.record_collision(cell)
    memory.record_interaction("OpenObject", 0, success=True)
    assert cell not in memory.collision_cells
    assert memory.cell_state(cell) == "unknown"


@pytest.mark.parametrize("cell", [(True, 1), (1.5, 2), (1,)])
def test_collision_requires_a_pair_of_integer_cells(
    tmp_path: Path, cell: object
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    with pytest.raises(ValueError, match="collision cell"):
        memory.record_collision(cell)  # type: ignore[arg-type]
