"""Independent numerical references for the OpenEQA-to-online mapping transfer.

All geometry and features are synthetic. No cached pickle, model, GPU, dataset
mutation or legacy pipeline import is involved in these reference calculations.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from codex_agent.alfred.geometry import Detection, Observation
from codex_agent.alfred.memory import (
    IncrementalSceneMemory,
    MappingConfig,
    MemoryConfig,
)
from codex_agent.alfred.object_mapping import (
    association_similarity,
    largest_density_cluster,
    point_overlap,
)


@pytest.mark.parametrize(("bias", "expected"), [(0.0, 1.2), (0.25, 1.1), (-0.5, 1.4)])
def test_phys_bias_matches_independently_calculated_weighted_sum(
    bias: float, expected: float
) -> None:
    assert association_similarity(0.4, 0.8, bias) == pytest.approx(expected)
    assert MappingConfig().phys_bias == 0.0


def test_directed_overlap_matches_brute_force_euclidean_reference() -> None:
    rng = np.random.default_rng(1701)
    existing = rng.uniform(-1, 1, size=(40, 3))
    observed = np.concatenate((existing[:17] + 0.002, np.full((9, 3), 10.0)))
    radius = 0.025
    pairwise = np.sqrt(
        np.sum((observed[:, None, :] - existing[None, :, :]) ** 2, axis=2)
    )
    reference = sum(min(row) < radius for row in pairwise) / len(observed)
    assert point_overlap(observed, existing, radius) == reference == 17 / 26
    reverse_reference = sum(min(column) < radius for column in pairwise.T) / len(
        existing
    )
    assert point_overlap(existing, observed, radius) == reverse_reference == 17 / 40


def test_overlap_radius_is_strict_not_squared_twice() -> None:
    existing = np.zeros((1, 3))
    observed = np.array([[0.024, 0, 0], [0.025, 0, 0], [0.026, 0, 0]])
    assert point_overlap(observed, existing, 0.025) == pytest.approx(1 / 3)


def test_dbscan_preserves_original_when_only_tiny_cluster_is_supported() -> None:
    points = np.array(
        [[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=float
    )
    assert largest_density_cluster(points, eps_m=0.1, min_points=3).tolist() == list(
        range(5)
    )


def _frame(
    root: Path, frame_id: int, image_feature: list[float], text_feature: list[float]
) -> Observation:
    rgb_path = root / f"reference_{frame_id}.png"
    Image.new("RGB", (10, 10), (120, 50, 20)).save(rgb_path)
    mask = np.zeros((10, 10), bool)
    mask[2:8, 2:8] = True
    return Observation(
        frame_id=frame_id,
        rgb_path=rgb_path,
        depth_m=np.full((10, 10), 2.0),
        intrinsics=np.array([[20, 0, 4.5], [0, 20, 4.5], [0, 0, 1.0]]),
        camera_to_world=np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]], dtype=float
        ),
        agent_position=(0.0, 0.0, 0.0),
        detections=(
            Detection(
                "Mug",
                mask,
                0.9,
                np.array(image_feature),
                np.array(text_feature),
                bbox_xyxy=(2, 2, 8, 8),
            ),
        ),
        frame_feature=np.array(image_feature),
    )


def test_online_fusion_updates_image_and_text_by_detection_count(
    tmp_path: Path,
) -> None:
    memory = IncrementalSceneMemory(tmp_path / "memory")
    for index in range(3):
        memory.integrate(_frame(tmp_path, index, [1.0, 0, 0], [0, 0, 1.0]))
    memory.integrate(_frame(tmp_path, 3, [0.6, 0.8, 0], [0, 1.0, 0]))
    assert len(memory.objects) == 1
    node = memory.objects[0]
    expected_image = np.array([3.6, 0.8, 0]) / np.sqrt(3.6**2 + 0.8**2)
    expected_text = np.array([0, 1.0, 3.0]) / np.sqrt(10)
    np.testing.assert_allclose(node.image_feature, expected_image)
    np.testing.assert_allclose(node.text_feature, expected_text)
    assert node.detection_count == 4


@pytest.mark.parametrize(
    ("image_feature", "text_feature"),
    [([0.8, 0.6, 0], [1.0, 0, 0]), ([0.9, np.sqrt(0.19), 0], [0.8, 0.6, 0])],
)
def test_periodic_merge_requires_strict_image_and_text_similarity(
    tmp_path: Path, image_feature: list[float], text_feature: list[float]
) -> None:
    memory = IncrementalSceneMemory(
        tmp_path / "memory",
        MemoryConfig(
            mapping=MappingConfig(association_threshold=2.0, maintenance_interval=2)
        ),
    )
    memory.integrate(_frame(tmp_path, 0, [1.0, 0, 0], [1.0, 0, 0]))
    memory.integrate(_frame(tmp_path, 1, image_feature, text_feature))
    assert len(memory.objects) == 2
    assert memory.aliases == {}


def test_mask_filter_uses_original_confidence_area_and_box_boundaries() -> None:
    from dataclasses import replace

    from codex_agent.alfred.object_mapping import mapping_masks

    mask = np.zeros((10, 10), bool)
    mask[:5, :5] = True
    base = Detection(
        "Mug",
        mask,
        0.25,
        np.array([1.0, 0]),
        np.array([1.0, 0]),
        bbox_xyxy=(0, 0, 10, 5),
    )
    smaller = mask.copy()
    smaller[0, 0] = False
    detections = (
        base,
        replace(base, confidence=0.249),
        replace(base, mask=smaller),
        replace(base, bbox_xyxy=(0, 0, 10, 5.001)),
    )
    retained, rejected = mapping_masks(detections, MappingConfig())
    assert set(retained) == {0}
    assert rejected == {
        1: "confidence_below_threshold",
        2: "mask_area_below_threshold",
        3: "detector_bbox_area_above_threshold",
    }
    np.testing.assert_array_equal(retained[0], mask)


def test_point_cap_is_applied_after_density_cluster_selection() -> None:
    from codex_agent.alfred.object_mapping import process_cloud

    dense = np.column_stack((np.arange(20) * 0.002, np.zeros((20, 2))))
    isolated = np.column_stack((np.arange(1, 6), np.zeros((5, 2))))
    points = np.concatenate((dense, isolated))
    config = MappingConfig(
        voxel_size_m=0.001,
        dbscan_eps_m=0.02,
        dbscan_min_points=5,
        max_points_per_object=5,
    )
    reduced, _ = process_cloud(points, np.full(points.shape, 0.5), config, denoise=True)
    assert len(reduced) == 5
    assert reduced[:, 0].max() < 0.04


def test_cloud_voxel_origin_matches_translation_of_input_cloud() -> None:
    from codex_agent.alfred.object_mapping import voxel_reduce

    points = np.array([[0.049, 0, 0], [0.051, 0, 0], [0.2, 0, 0]])
    colors = np.eye(3)
    reduced, _ = voxel_reduce(points, colors, 0.1)
    shifted, _ = voxel_reduce(points + np.array([0.049, 0.017, -0.012]), colors, 0.1)
    np.testing.assert_allclose(shifted, reduced + [0.049, 0.017, -0.012])


def test_oriented_gate_rejects_separated_rotated_slabs_with_overlapping_aabbs() -> None:
    from itertools import product

    from codex_agent.alfred.object_mapping import oriented_envelopes_overlap

    corners = np.array(list(product((-1.0, 1.0), (-0.05, 0.05), (-0.05, 0.05))))
    angle = np.pi / 4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    first = corners @ rotation.T
    second = first + [-0.15, 0.15, 0]
    assert np.all(first.min(axis=0) < second.max(axis=0))
    assert np.all(second.min(axis=0) < first.max(axis=0))
    assert not oriented_envelopes_overlap(first, second)
    assert oriented_envelopes_overlap(first, first + [0.01, 0.01, 0])


def test_category_votes_are_unweighted_even_with_stronger_opposing_confidence(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    memory = IncrementalSceneMemory(tmp_path / "memory")
    for frame_id, category, confidence in (
        (0, "Mug", 0.3),
        (1, "Mug", 0.3),
        (2, "Cup", 0.95),
    ):
        observation = _frame(tmp_path, frame_id, [1.0, 0, 0], [1.0, 0, 0])
        detection = replace(
            observation.detections[0], category=category, confidence=confidence
        )
        memory.integrate(replace(observation, detections=(detection,)))
    assert len(memory.objects) == 1
    node = memory.objects[0]
    assert node.category == "Mug"
    assert node.category_evidence == {"Mug": 2, "Cup": 1}
    assert [view.confidence for view in node.views] == [0.3, 0.3, 0.95]
