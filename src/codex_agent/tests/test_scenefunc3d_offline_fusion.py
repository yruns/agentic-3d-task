"""Tests for the offline SceneFunc3D fusion re-scoring harness."""

from __future__ import annotations

import numpy as np

from codex_agent.scenefunc3d.backends.fusion import FusionParams
from codex_agent.scenefunc3d.evaluation.offline_fusion import (
    SampleFusionInput,
    fuse_and_score_sample,
)


def _line_scene(n: int = 40) -> np.ndarray:
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, 0] = np.linspace(0.0, 0.39, n)  # 1 cm spacing
    return coords


def test_fuse_and_score_recovers_precision_over_union() -> None:
    scene = _line_scene()
    gt_ids = frozenset(range(0, 3))  # tight target well inside the anchor radius
    sample = SampleFusionInput(
        sample_id="420673::demo",
        motion_type="pinch_pull",
        frames_point_indices=(
            (0, 1, 2),
            (0, 1, 2, 30, 31, 32),
        ),
        anchor_point_indices=(0, 1, 2),
        scene_vertices=scene,
        gt_ids=gt_ids,
    )
    params = FusionParams(
        agreement_tau=0.5, cluster_link_eps_m=0.03, min_cluster_points=2
    )
    score = fuse_and_score_sample(sample, params)
    assert score.metrics.precision == 1.0  # spurious tail removed
    assert score.metrics.iou == 1.0
