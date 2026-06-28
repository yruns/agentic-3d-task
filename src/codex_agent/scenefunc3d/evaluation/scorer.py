"""Scoring contracts for SceneFunc3D mask evaluation."""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass

from .metrics import MaskMetrics, compute_mask_metrics


@dataclass(frozen=True)
class SceneFunc3dScore:
    """SceneFunc3D score for one sample."""

    sample_id: str
    metrics: MaskMetrics
    failure_type: str


def score_point_ids(
    *,
    sample_id: str,
    predicted_ids: Set[int],
    gt_ids: Set[int],
    failure_type: str = "",
) -> SceneFunc3dScore:
    """Score predicted point ids against ground truth ids for one sample."""
    return SceneFunc3dScore(
        sample_id=sample_id,
        metrics=compute_mask_metrics(predicted_ids=predicted_ids, gt_ids=gt_ids),
        failure_type=failure_type,
    )


__all__ = ["SceneFunc3dScore", "score_point_ids"]
