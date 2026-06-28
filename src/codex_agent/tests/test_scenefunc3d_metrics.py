"""Tests for SceneFunc3D mask metrics."""

from __future__ import annotations

from codex_agent.scenefunc3d.evaluation.metrics import (
    MaskMetrics,
    compute_mask_metrics,
)
from codex_agent.scenefunc3d.evaluation.scorer import score_point_ids


def test_compute_mask_metrics() -> None:
    metrics = compute_mask_metrics(predicted_ids={1, 2, 3}, gt_ids={2, 3, 4, 5})
    assert metrics == MaskMetrics(
        iou=0.4,
        precision=2 / 3,
        recall=0.5,
        f1=4 / 7,
        predicted_count=3,
        gt_count=4,
    )


def test_compute_empty_metrics() -> None:
    metrics = compute_mask_metrics(predicted_ids=set(), gt_ids={1, 2})
    assert metrics.iou == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_compute_both_empty_metrics() -> None:
    metrics = compute_mask_metrics(predicted_ids=set(), gt_ids=set())
    assert metrics == MaskMetrics(
        iou=0.0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        predicted_count=0,
        gt_count=0,
    )


def test_compute_perfect_match_metrics() -> None:
    metrics = compute_mask_metrics(predicted_ids={10, 20}, gt_ids={10, 20})
    assert metrics == MaskMetrics(
        iou=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        predicted_count=2,
        gt_count=2,
    )


def test_score_point_ids_wraps_metrics_and_failure_type() -> None:
    score = score_point_ids(
        sample_id="visit_001_desc_002",
        predicted_ids={1, 5},
        gt_ids={1, 2, 3},
        failure_type="partial_lift",
    )

    assert score.sample_id == "visit_001_desc_002"
    assert score.failure_type == "partial_lift"
    assert score.metrics == MaskMetrics(
        iou=0.25,
        precision=0.5,
        recall=1 / 3,
        f1=0.4,
        predicted_count=2,
        gt_count=3,
    )
