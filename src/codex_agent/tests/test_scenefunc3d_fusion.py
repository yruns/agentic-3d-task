"""Tests for SceneFunc3D layered multi-view fusion."""

from __future__ import annotations

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.anchor import build_anchor
from codex_agent.scenefunc3d.backends.fusion import (
    FrameLift,
    FusedMask,
    FusionParams,
    MultiViewLiftBundle,
    agreement_scores,
    anchor_component_indices,
    connected_components,
    fuse_multiview_points,
)
from codex_agent.scenefunc3d.tools.models import ToolInputError


def test_frame_lift_rejects_negative_index() -> None:
    with pytest.raises(ToolInputError):
        FrameLift(frame_id="000001", point_indices=(-1, 2))


def test_bundle_requires_at_least_one_frame() -> None:
    anchor = build_anchor(
        np.zeros((3, 3)), motion_type="rotate", seed_frame_id="000001"
    )
    with pytest.raises(ToolInputError):
        MultiViewLiftBundle(frames=(), anchor=anchor)


def test_fusion_params_defaults() -> None:
    params = FusionParams()
    assert params.agreement_tau == pytest.approx(0.5)
    assert params.radius_scale == pytest.approx(1.0)
    assert params.cluster_link_eps_m == pytest.approx(0.02)
    assert params.min_cluster_points == 10


def test_agreement_uses_visibility_denominator() -> None:
    frames = (
        FrameLift(frame_id="000001", point_indices=(5, 6)),
        FrameLift(frame_id="000002", point_indices=(5,)),
    )
    visibility = {5: 4, 6: 2}
    scores = agreement_scores(frames, visibility)
    assert scores[5] == pytest.approx(0.5)
    assert scores[6] == pytest.approx(0.5)


def test_agreement_without_visibility_uses_frame_count() -> None:
    frames = (
        FrameLift(frame_id="000001", point_indices=(5, 6)),
        FrameLift(frame_id="000002", point_indices=(5,)),
    )
    scores = agreement_scores(frames, {})
    assert scores[5] == pytest.approx(1.0)
    assert scores[6] == pytest.approx(0.5)


def test_agreement_clamped_to_one() -> None:
    frames = (FrameLift(frame_id="000001", point_indices=(5, 5)),)
    scores = agreement_scores(frames, {5: 1})
    assert scores[5] == pytest.approx(1.0)


def test_connected_components_splits_far_blobs() -> None:
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [1.01, 0.0, 0.0]]
    )
    components = connected_components(coords, eps=0.05)
    assert len(components) == 2
    assert {frozenset(c) for c in components} == {frozenset({0, 1}), frozenset({2, 3})}


def test_anchor_component_keeps_blob_nearest_anchor() -> None:
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [1.01, 0.0, 0.0]]
    )
    kept = anchor_component_indices(
        coords, anchor_centroid=(0.0, 0.0, 0.0), eps=0.05, min_points=1
    )
    assert set(kept) == {0, 1}


def test_anchor_component_drops_small_components() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    kept = anchor_component_indices(
        coords, anchor_centroid=(0.0, 0.0, 0.0), eps=0.05, min_points=5
    )
    assert kept == ()


def _scene_vertices() -> np.ndarray:
    target = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    wall = np.stack([np.linspace(1.0, 1.09, 10), np.zeros(10), np.zeros(10)], axis=1)
    return np.concatenate([target, wall], axis=0)


def test_fusion_keeps_consensus_target_drops_far_wall_by_anchor_gate() -> None:
    anchor = build_anchor(
        _scene_vertices()[:3], motion_type="pinch_pull", seed_frame_id="000001"
    )
    frames = (
        FrameLift(frame_id="000001", point_indices=(0, 1, 2)),
        FrameLift(frame_id="000002", point_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
    )
    bundle = MultiViewLiftBundle(frames=frames, anchor=anchor)
    params = FusionParams(
        agreement_tau=0.5, cluster_link_eps_m=0.05, min_cluster_points=2
    )
    fused = fuse_multiview_points(bundle, _scene_vertices(), params)
    assert isinstance(fused, FusedMask)
    assert set(fused.point_indices) == {0, 1, 2}
    assert fused.confidence == pytest.approx(1.0)


def test_fusion_empty_when_nothing_passes_agreement() -> None:
    anchor = build_anchor(
        _scene_vertices()[:3], motion_type="pinch_pull", seed_frame_id="000001"
    )
    frames = (FrameLift(frame_id="000001", point_indices=(0,)),)
    bundle = MultiViewLiftBundle(
        frames=frames, anchor=anchor, visibility_counts={0: 10}
    )
    fused = fuse_multiview_points(
        bundle, _scene_vertices(), FusionParams(agreement_tau=0.5)
    )
    assert fused.point_indices == ()
    assert fused.confidence == pytest.approx(0.0)
