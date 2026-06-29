"""JSON payload helpers for SceneFunc3D evaluation outputs."""

from __future__ import annotations

from collections.abc import Sequence
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


class SceneFunc3dScoreManifestPayload(TypedDict):
    """JSON-ready manifest for scoring multiple SceneFunc3D runner results."""

    result_count: int
    result_paths: list[str]
    scores: list[SceneFunc3dScorePayload]


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


def score_manifest_to_payload(
    *,
    scores: Sequence[SceneFunc3dScore],
    result_paths: Sequence[str],
) -> SceneFunc3dScoreManifestPayload:
    """Convert multiple scores into a deterministic evaluation manifest."""
    return {
        "result_count": len(scores),
        "result_paths": list(result_paths),
        "scores": [score_to_payload(score) for score in scores],
    }


__all__ = [
    "MaskMetricsPayload",
    "SceneFunc3dScorePayload",
    "SceneFunc3dScoreManifestPayload",
    "score_manifest_to_payload",
    "score_to_payload",
]
