"""Synthetic regressions for CLIP object mapping, temporal evidence and identity."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from codex_agent.alfred.memory import (
    Detection,
    IncrementalSceneMemory,
    MappingConfig,
    MemoryConfig,
    Observation,
)


def _observation(
    root: Path,
    frame_id: int,
    *,
    translation_x: float = 0.0,
    category: str = "Mug",
    feature: tuple[float, float, float] = (1.0, 0.0, 0.0),
    with_detection: bool = True,
) -> Observation:
    rgb_path = root / f"rgb_{frame_id}.png"
    Image.new("RGB", (10, 10), (200, 30, 20)).save(rgb_path)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    vector = np.array(feature)
    return Observation(
        frame_id=frame_id,
        rgb_path=rgb_path,
        depth_m=np.full((10, 10), 2.0),
        intrinsics=np.array([[20, 0, 4.5], [0, 20, 4.5], [0, 0, 1.0]]),
        camera_to_world=np.array(
            [[1, 0, 0, translation_x], [0, 0, 1, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]],
            dtype=float,
        ),
        agent_position=(translation_x, 0.0, 0.0),
        detections=(
            (
                Detection(
                    category,
                    mask,
                    0.9,
                    vector,
                    np.array([1.0, 0.0, 0.0]),
                    bbox_xyxy=(2, 2, 8, 8),
                ),
            )
            if with_detection
            else ()
        ),
        frame_feature=vector,
    )


def test_projects_and_persists_all_observation_features(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    observation = _observation(tmp_path, 0)
    frame = memory.integrate(observation)
    np.testing.assert_allclose(memory.objects[0].centroid, [0, 2, 1.5], atol=1e-12)
    assert memory.current_detection_objects == {0: 0}
    assert memory.object_to_views[0] == {0}
    assert memory.view_to_objects[0] == {0}
    index = json.loads(frame.geometry_path.with_suffix(".json").read_text())
    assert index["detection_object_ids"] == {"0": 0}
    assert {tuple(cell) for cell in index["cells"]} == memory.view_to_grids[0]
    for cell in frame.cells:
        assert memory.grid_to_views[cell] == {0}
    with np.load(frame.geometry_path, allow_pickle=False) as saved:
        np.testing.assert_allclose(
            saved["image_features"][0], observation.detections[0].image_feature
        )
        np.testing.assert_allclose(
            saved["text_features"][0], observation.detections[0].text_feature
        )
        np.testing.assert_allclose(saved["frame_feature"], observation.frame_feature)
        assert saved["masks"].shape == (1, 10, 10)
        assert saved["confidences"].tolist() == [0.9]
    exported = memory.to_scene_objects()
    assert exported[0].clip_ft is not None
    np.testing.assert_allclose(exported[0].clip_ft, [1, 0, 0])
    assert exported[0].image_idx == [0]


def test_overlap_and_clip_fuse_nearby_observations_not_separated_objects(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0))
    memory.integrate(_observation(tmp_path, 1, translation_x=0.01))
    assert len(memory.objects) == 1
    assert memory.objects[0].detection_count == 2
    memory.integrate(_observation(tmp_path, 2, translation_x=2))
    assert memory.current_detection_objects == {0: 1}


def test_appearance_is_clip_not_rgb_histogram(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0))
    memory.integrate(_observation(tmp_path, 1, feature=(0, 1, 0)))
    assert len(memory.objects) == 2  # Identical RGB colors and geometry.


def test_label_jitter_does_not_block_geometric_visual_identity(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0, category="Mug"))
    memory.integrate(_observation(tmp_path, 1, category="Cup"))
    assert len(memory.objects) == 1
    assert set(memory.objects[0].category_evidence) == {"Mug", "Cup"}


def test_provisional_objects_are_queryable_and_confirm_on_distinct_frames(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0))
    assert memory.objects[0].provisional
    assert memory.retrieve_frames(object_ids=(0,))
    memory.integrate(_observation(tmp_path, 1))
    memory.integrate(_observation(tmp_path, 2))
    assert not memory.objects[0].provisional


def test_same_frame_duplicates_do_not_inflate_support(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    frame = _observation(tmp_path, 0)
    duplicate = replace(frame.detections[0], category="Cup", confidence=0.8)
    memory.integrate(replace(frame, detections=frame.detections + (duplicate,)))
    assert len(memory.objects) == 1
    assert memory.current_detection_objects == {0: 0, 1: 0}
    assert memory.suppressed_detection_indices == {1: 0}
    assert memory.objects[0].detection_count == 1


def test_simultaneous_distinct_instances_never_merge_during_maintenance(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(
        tmp_path / "memory",
        MemoryConfig(
            mapping=MappingConfig(min_detection_points=8, merge_overlap_threshold=0.0)
        ),
    )
    frame = _observation(tmp_path, 0)
    left, right = np.zeros((10, 10), bool), np.zeros((10, 10), bool)
    left[:, :5], right[:, 5:] = True, True
    memory.integrate(
        replace(
            frame,
            detections=(
                replace(frame.detections[0], mask=left, bbox_xyxy=(0, 0, 5, 10)),
                replace(frame.detections[0], mask=right, bbox_xyxy=(5, 0, 10, 10)),
            ),
        )
    )
    memory.maintain_objects()
    assert len(memory.objects) == 2
    assert set(memory.current_detection_objects.values()) == {0, 1}


def test_merge_aliases_preserve_views_and_ids_are_not_reused(tmp_path: Path) -> None:
    mapping = MappingConfig(
        association_threshold=1.9, merge_image_similarity=0.6, maintenance_interval=2
    )
    memory = IncrementalSceneMemory(tmp_path / "memory", MemoryConfig(mapping=mapping))
    memory.integrate(_observation(tmp_path, 0))
    memory.integrate(_observation(tmp_path, 1, feature=(0.7, np.sqrt(0.51), 0)))
    assert set(memory.objects) == {0}
    assert memory.aliases == {1: 0}
    assert memory.objects[0].category_evidence == {"Mug": 2}
    assert memory.resolve_object_id(1) == 0
    assert memory.object_to_views[0] == {0, 1}
    assert memory.view_to_objects[1] == {0}
    assert memory.frames[1].object_ids == (0,)
    assert {frame.frame_id for frame in memory.retrieve_frames(object_ids=(1,))} == {
        0,
        1,
    }
    memory.integrate(_observation(tmp_path, 2, translation_x=3))
    assert set(memory.objects) == {0, 2}


def test_failed_interaction_preserves_evidence_and_pickup_invalidates_it(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0))
    memory.record_interaction("PickupObject", 0, success=False)
    assert memory.objects[0].geometry_valid
    memory.record_interaction("PickupObject", 0, success=True)
    assert memory.held_object_id == 0
    assert not memory.objects[0].geometry_valid
    assert memory.objects[0].geometry_epoch == 1
    assert memory.retrieve_frames(object_ids=(0,)) == ()
    assert memory.retrieve_frames(object_ids=(0,), include_stale=True)[0].frame_id == 0


def test_held_then_placed_identity_replaces_old_geometry(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0))
    memory.record_interaction("PickupObject", 0, success=True)
    carried = replace(
        _observation(tmp_path, 1, translation_x=3), depth_m=np.full((10, 10), 0.4)
    )
    memory.integrate(carried)
    assert memory.current_detection_objects == {0: 0}
    assert not memory.objects[0].geometry_valid
    receptacle = _observation(
        tmp_path, 2, translation_x=3, category="CounterTop", feature=(0, 1, 0)
    )
    memory.integrate(receptacle)
    memory.record_interaction("PutObject", 1, success=True)
    memory.integrate(_observation(tmp_path, 3, translation_x=3))
    assert memory.current_detection_objects == {0: 0}
    assert len(memory.objects) == 2
    np.testing.assert_allclose(memory.objects[0].centroid, [3, 2, 1.5], atol=0.01)
    assert memory.objects[0].points[:, 0].min() > 2.5
    assert memory.objects[0].geometry_epoch == 2
    assert [frame.frame_id for frame in memory.retrieve_frames(object_ids=(0,))] == [3]


def test_open_requires_new_visual_evidence_but_preserves_identity(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0, category="Cabinet"))
    memory.record_interaction("OpenObject", 0, success=True)
    assert memory.retrieve_frames(object_ids=(0,)) == ()
    memory.integrate(_observation(tmp_path, 1, category="Cabinet"))
    assert len(memory.objects) == 1
    assert memory.objects[0].geometry_valid
    assert [frame.frame_id for frame in memory.retrieve_frames(object_ids=(0,))] == [1]


def test_slice_creates_child_identity_instead_of_reanimating_parent(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0, category="Apple"))
    memory.record_interaction("SliceObject", 0, success=True)
    memory.integrate(_observation(tmp_path, 1, category="AppleSliced"))
    assert memory.objects[0].state == "sliced"
    assert memory.objects[1].parent_object_id == 0
    assert memory.current_detection_objects == {0: 1}


def test_retrieval_covers_regions_without_requiring_a_detected_object(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    first = memory.integrate(_observation(tmp_path, 0, with_detection=False))
    memory.integrate(_observation(tmp_path, 1, translation_x=3))
    assert first.cells
    assert memory.retrieve_frames(cells=(first.cells[-1],))
    assert memory.retrieve_frames(cells=((999, 999),)) == ()


def test_retrieval_covers_disjoint_objects_and_excludes_delivered_frames(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0))
    memory.integrate(_observation(tmp_path, 1, translation_x=3))
    assert {
        frame.frame_id for frame in memory.retrieve_frames(object_ids=(0, 1), limit=2)
    } == {0, 1}
    assert memory.retrieve_frames(object_ids=(0, 1), exclude_frame_ids=(0, 1)) == ()
    memory.integrate(_observation(tmp_path, 2))
    assert [
        frame.frame_id
        for frame in memory.retrieve_frames(object_ids=(0,), exclude_frame_ids=(0, 1))
    ] == [2]


def test_view_selection_prefers_camera_complementarity(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    for frame_id, x in enumerate((0.0, 1.0, 1.0)):
        memory.integrate(
            _observation(tmp_path, frame_id, translation_x=x, with_detection=False)
        )
    assert [frame.frame_id for frame in memory.retrieve_frames(limit=2)] == [2, 0]


def test_geometry_storage_and_view_history_are_bounded(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(
        tmp_path / "memory",
        MemoryConfig(
            mapping=MappingConfig(max_points_per_object=8), max_object_views=2
        ),
    )
    for frame_id in range(4):
        memory.integrate(_observation(tmp_path, frame_id))
    assert len(memory.objects[0].points) <= 8
    assert len(memory.to_scene_objects()[0].image_idx) <= 2
    assert memory.snapshot()["frame_count"] == 4


def test_unusable_depth_keeps_raw_detection_without_fictitious_geometry(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    observation = _observation(tmp_path, 0)
    depth = observation.depth_m.copy()
    depth[observation.detections[0].mask] = 0
    frame = memory.integrate(replace(observation, depth_m=depth))
    assert not memory.objects
    with np.load(frame.geometry_path, allow_pickle=False) as saved:
        assert saved["masks"].shape[0] == 1


@pytest.mark.parametrize("excluded", [(-1,), (True,), (1.5,)])
def test_invalid_retrieval_exclusions_are_rejected(
    tmp_path: Path, excluded: tuple[object, ...]
) -> None:
    with pytest.raises(ValueError, match="exclude_frame_ids"):
        IncrementalSceneMemory(tmp_path).retrieve_frames(exclude_frame_ids=excluded)  # type: ignore[arg-type]


def test_unknown_identity_and_repeated_frame_are_rejected(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    with pytest.raises(KeyError, match="object_id"):
        memory.retrieve_frames(object_ids=(17,))
    memory.integrate(_observation(tmp_path, 1))
    with pytest.raises(ValueError, match="frame_id"):
        memory.integrate(_observation(tmp_path, 0))


def test_action_state_is_retained_as_feedback_prior_after_reobservation(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0, category="Cabinet"))
    memory.record_interaction("OpenObject", 0, success=True)
    memory.integrate(_observation(tmp_path, 1, category="Cabinet"))
    state = memory.objects[0].action_states["open"]
    assert state.value is True and state.frame_id == 0
    catalog = memory.snapshot()["objects"]
    assert isinstance(catalog, list)
    assert catalog[0]["action_states"]["open"]["source"] == "successful_action"


def test_observation_rejects_feature_space_change_without_writing_artifact(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    memory.integrate(_observation(tmp_path, 0))
    next_frame = _observation(tmp_path, 1, with_detection=False)
    with pytest.raises(ValueError, match="feature dimension"):
        memory.integrate(replace(next_frame, frame_feature=np.array([1.0, 0])))
    assert not (tmp_path / "memory/observations/frame_000001.npz").exists()


def test_contained_mask_subtraction_preserves_raw_evidence_and_supplies_interaction_mask(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(
        tmp_path / "memory",
        MemoryConfig(mapping=MappingConfig(max_bbox_area_ratio=1.0)),
    )
    observation = _observation(tmp_path, 0)
    parent_mask = np.ones((10, 10), bool)
    child_mask = np.zeros((10, 10), bool)
    child_mask[2:7, 2:7] = True
    parent = Detection(
        "CounterTop",
        parent_mask,
        0.9,
        np.array([1.0, 0, 0]),
        np.array([1.0, 0, 0]),
        bbox_xyxy=(0, 0, 10, 10),
    )
    child = Detection(
        "Mug",
        child_mask,
        0.9,
        np.array([0, 1.0, 0]),
        np.array([0, 1.0, 0]),
        bbox_xyxy=(2, 2, 7, 7),
    )
    frame = memory.integrate(replace(observation, detections=(parent, child)))
    assert (
        set(memory.current_detection_masks)
        == set(memory.current_detection_objects)
        == {0, 1}
    )
    np.testing.assert_array_equal(
        memory.current_detection_masks[0], parent_mask & ~child_mask
    )
    np.testing.assert_array_equal(memory.current_detection_masks[1], child_mask)
    assert not memory.current_detection_masks[0].flags.writeable
    assert observation.detections[0].mask.sum() == 36
    assert parent.mask.sum() == 100
    index = json.loads(frame.geometry_path.with_suffix(".json").read_text())
    with np.load(frame.geometry_path, allow_pickle=False) as raw:
        assert raw["masks"][0].sum() == 100
        np.testing.assert_array_equal(
            raw["detector_boxes"], [[0, 0, 10, 10], [2, 2, 7, 7]]
        )
    with np.load(index["effective_masks_path"], allow_pickle=False) as effective:
        assert effective["masks"][0].sum() == 75
        assert effective["detection_indices"].tolist() == [0, 1]


def test_held_object_self_occlusion_suppresses_misclassified_ghosts_and_obstacles(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(
        tmp_path / "memory",
        MemoryConfig(mapping=MappingConfig(max_bbox_area_ratio=1.0)),
    )
    memory.integrate(_observation(tmp_path, 0, category="Plate"))
    memory.record_interaction("PickupObject", 0, success=True)

    # Frame 1: while holding Plate (#0), close-up detector hallucinates both CD and Ladle at 0.40m.
    base = _observation(tmp_path, 1, translation_x=1.0, with_detection=False)
    depth = np.full((10, 10), 2.5, dtype=float)
    depth[5:10, 2:8] = 0.40
    mask_cd = np.zeros((10, 10), dtype=bool)
    mask_cd[5:10, 2:8] = True
    mask_ladle = np.zeros((10, 10), dtype=bool)
    mask_ladle[6:9, 3:7] = True
    det_cd = Detection(
        "CD",
        mask_cd,
        0.85,
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        bbox_xyxy=(2, 5, 8, 10),
    )
    det_ladle = Detection(
        "Ladle",
        mask_ladle,
        0.75,
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        bbox_xyxy=(3, 6, 7, 9),
    )
    memory.integrate(replace(base, depth_m=depth, detections=(det_cd, det_ladle)))
    assert len(memory.objects) == 1
    assert memory.objects[0].category == "Plate"
    assert set(memory.current_detection_objects.values()) == {0}


def test_spatial_fallback_rejects_incompatible_categories(tmp_path: Path) -> None:
    memory = IncrementalSceneMemory(
        tmp_path / "memory",
        MemoryConfig(mapping=MappingConfig(max_bbox_area_ratio=1.0)),
    )
    # Frame 0: observe SinkBasin (#0)
    memory.integrate(_observation(tmp_path, 0, category="SinkBasin"))
    # Frame 1: SinkBasin is missed, but a SaltShaker (#1) is detected at the same spot
    memory.integrate(
        _observation(
            tmp_path,
            1,
            category="SaltShaker",
            feature=(0.0, 1.0, 0.0),
        )
    )
    assert len(memory.objects) == 2
    # SaltShaker must NOT be used as a fallback mask for PutObject on SinkBasin (#0)
    assert memory.select_interaction_mask(0, action="PutObject") is None


def test_spatial_fallback_rejects_unrelated_pickupables_and_updates_placed_bounds(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(
        tmp_path / "memory",
        MemoryConfig(mapping=MappingConfig(max_bbox_area_ratio=1.0)),
    )
    # Frame 0: observe TissueBox (#0) at x=0
    memory.integrate(
        _observation(tmp_path, 0, category="TissueBox", feature=(1.0, 0.0, 0.0))
    )
    # Frame 1: TissueBox is missed, but an unrelated Candle (#1) is detected at x=0
    memory.integrate(
        _observation(tmp_path, 1, category="Candle", feature=(0.0, 1.0, 0.0))
    )
    assert len(memory.objects) == 2
    # Candle (#1) must NOT be used as a fallback mask for PickupObject on TissueBox (#0)
    assert memory.select_interaction_mask(0, action="PickupObject") is None

    # Pick up TissueBox (#0) and place it on Fridge (#2) at translation_x=3
    memory.record_interaction("PickupObject", 0, success=True)
    memory.integrate(
        _observation(
            tmp_path, 2, translation_x=3, category="Fridge", feature=(0.0, 0.0, 1.0)
        )
    )
    memory.record_interaction("PutObject", 2, success=True)
    # Before TissueBox (#0) is directly re-observed, its centroid/bounds reflect Fridge (#2)
    np.testing.assert_allclose(
        memory.objects[0].centroid, memory.objects[2].centroid, atol=0.05
    )
