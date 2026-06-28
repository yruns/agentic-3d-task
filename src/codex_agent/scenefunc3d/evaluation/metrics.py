"""Mask metric contracts for SceneFunc3D evaluation."""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass


@dataclass(frozen=True)
class MaskMetrics:
    """Point-id mask overlap metrics."""

    iou: float
    precision: float
    recall: float
    f1: float
    predicted_count: int
    gt_count: int


def compute_mask_metrics(predicted_ids: Set[int], gt_ids: Set[int]) -> MaskMetrics:
    """Compute overlap metrics between predicted and ground-truth point ids."""
    predicted_count = len(predicted_ids)
    gt_count = len(gt_ids)
    intersection_count = sum(1 for point_id in predicted_ids if point_id in gt_ids)
    union_count = predicted_count + gt_count - intersection_count

    iou = intersection_count / union_count if union_count > 0 else 0.0
    precision = intersection_count / predicted_count if predicted_count > 0 else 0.0
    recall = intersection_count / gt_count if gt_count > 0 else 0.0
    f1 = (
        2 * intersection_count / (predicted_count + gt_count)
        if precision + recall > 0.0
        else 0.0
    )

    return MaskMetrics(
        iou=iou,
        precision=precision,
        recall=recall,
        f1=f1,
        predicted_count=predicted_count,
        gt_count=gt_count,
    )


__all__ = ["MaskMetrics", "compute_mask_metrics"]
