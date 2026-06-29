"""JSON payload helpers for SceneFunc3D evaluation outputs."""

from __future__ import annotations

from typing import TypedDict

from .scorer import SceneFunc3dScore


class MaskMetricsPayload(TypedDict):
    """JSON-ready payload for one SceneFunc3D mask metric bundle."""

    iou: float
    precision: float
    recall: float
    f1: float
    predicted_count: int
    gt_count: int


class SceneFunc3dScorePayload(TypedDict):
    """JSON-ready score payload for one SceneFunc3D runner result."""

    sample_id: str
    failure_type: str
    metrics: MaskMetricsPayload


def score_to_payload(score: SceneFunc3dScore) -> SceneFunc3dScorePayload:
    """Convert an internal score object into a public JSON payload."""
    return {
        "sample_id": score.sample_id,
        "failure_type": score.failure_type,
        "metrics": {
            "iou": score.metrics.iou,
            "precision": score.metrics.precision,
            "recall": score.metrics.recall,
            "f1": score.metrics.f1,
            "predicted_count": score.metrics.predicted_count,
            "gt_count": score.metrics.gt_count,
        },
    }


__all__ = [
    "MaskMetricsPayload",
    "SceneFunc3dScorePayload",
    "score_to_payload",
]
