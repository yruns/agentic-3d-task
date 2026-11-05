"""Regression tests for geometry reduction and weighted CLIP object evidence."""

import numpy as np
import pytest

from codex_agent.alfred.geometry import Detection
from codex_agent.alfred.object_mapping import (
    fuse_features,
    largest_density_cluster,
    point_overlap,
    voxel_reduce,
)


def test_voxel_reduction_averages_colors_and_geometry_deterministically() -> None:
    points = np.array([[0.01, 0, 0], [0.02, 0, 0], [0.2, 0, 0]])
    colors = np.eye(3)
    reduced, averaged = voxel_reduce(points, colors, 0.1)
    np.testing.assert_allclose(reduced, [[0.015, 0, 0], [0.2, 0, 0]])
    np.testing.assert_allclose(averaged, np.array([[0.5, 0.5, 0], [0, 0, 1]]))


def test_density_filter_removes_distant_outlier_but_preserves_sparse_new_object() -> (
    None
):
    points = np.array(
        [[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0], [0.04, 0, 0], [4, 0, 0]],
        dtype=float,
    )
    assert largest_density_cluster(points, eps_m=0.1, min_points=3).tolist() == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert largest_density_cluster(points, eps_m=0.001, min_points=3).tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]


def test_overlap_is_observed_point_support_not_centroid_distance() -> None:
    existing = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    assert point_overlap(np.array([[0, 0, 0], [0.5, 0, 0]]), existing, 0.025) == 0.5
    assert point_overlap(np.array([[0.5, 0, 0]]), existing, 0.025) == 0


def test_feature_fusion_weights_number_of_observations() -> None:
    feature = fuse_features(np.array([1.0, 0]), np.array([0.0, 1]), 3, 1)
    np.testing.assert_allclose(feature, np.array([3, 1]) / np.sqrt(10))


@pytest.mark.parametrize(
    "feature", [np.zeros(3), np.ones(3), np.array([np.nan, 0, 0]), np.ones((1, 3))]
)
def test_detection_rejects_invalid_features(feature: np.ndarray) -> None:
    with pytest.raises(ValueError, match="image_feature"):
        Detection(
            "Mug",
            np.ones((2, 2), bool),
            0.9,
            feature,
            np.array([1.0, 0, 0]),
            bbox_xyxy=(0, 0, 2, 2),
        )


def test_detection_copies_mask_and_features_before_retaining_evidence() -> None:
    mask = np.ones((2, 2), bool)
    feature = np.array([1.0, 0, 0])
    detection = Detection("Mug", mask, 0.9, feature, feature, bbox_xyxy=(0, 0, 2, 2))
    mask[:] = False
    feature[:] = 0
    assert detection.mask.all()
    np.testing.assert_array_equal(detection.image_feature, [1, 0, 0])
    assert not detection.image_feature.flags.writeable


@pytest.mark.parametrize("confidence", [-1.0, 1.1, np.nan, True])
def test_detection_confidence_is_validated(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        Detection(
            "Mug",
            np.ones((2, 2), bool),
            confidence,
            np.array([1.0, 0]),
            np.array([1.0, 0]),
            bbox_xyxy=(0, 0, 2, 2),
        )


def test_depth_visibility_rejects_hidden_outside_and_behind_camera_points() -> None:
    from codex_agent.alfred.geometry import depth_visible_points

    points = np.array([[0, 0, 2], [0, 0, 3], [20, 0, 2], [0, 0, -2]], dtype=float)
    mask = depth_visible_points(
        points,
        np.full((8, 8), 2.0),
        np.array([[20, 0, 3.5], [0, 20, 3.5], [0, 0, 1.0]]),
        np.eye(4),
        tolerance_m=0.1,
    )
    assert mask.tolist() == [True, False, False, False]


def test_oriented_envelope_margin_and_clip_gate() -> None:
    from codex_agent.alfred.object_mapping import (
        MappingConfig,
        mapping_masks,
        oriented_envelopes_overlap,
    )

    slab1 = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.2, 0.2, 0.01]]
    )
    slab2 = slab1 + np.array([0.0, 0.0, 0.06])
    assert not oriented_envelopes_overlap(slab1, slab2, margin_m=0.0)
    assert oriented_envelopes_overlap(slab1, slab2, margin_m=0.08)

    det_good = Detection(
        "Mug",
        np.ones((6, 6), bool),
        0.30,
        np.array([1.0, 0.0]),
        np.array([0.25, np.sqrt(1 - 0.25**2)]),
        bbox_xyxy=(0, 0, 6, 6),
    )
    det_bad_clip = Detection(
        "Mug",
        np.ones((6, 6), bool),
        0.30,
        np.array([1.0, 0.0]),
        np.array([0.05, np.sqrt(1 - 0.05**2)]),
        bbox_xyxy=(0, 0, 6, 6),
    )
    retained, rejected = mapping_masks(
        (det_good, det_bad_clip),
        MappingConfig(min_clip_similarity=0.18, max_bbox_area_ratio=1.0),
    )
    assert set(retained) == {0}
    assert rejected == {1: "clip_similarity_below_threshold"}
