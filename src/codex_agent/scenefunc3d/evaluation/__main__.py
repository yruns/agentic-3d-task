"""Score one SceneFunc3D runner ``result.json`` from the command line."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from codex_agent.errors import CodexAgentError

from .scorer import SceneFunc3dScore, score_result_file


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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the SceneFunc3D result scoring CLI parser."""
    parser = argparse.ArgumentParser(
        prog="codex_agent.scenefunc3d.evaluation",
        description=__doc__,
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="SceneFunc3D dataset root containing per-visit scene directories.",
    )
    parser.add_argument(
        "--result-path",
        required=True,
        type=Path,
        help="Path to one SceneFunc3D runner result.json.",
    )
    parser.add_argument(
        "--failure-type",
        default="",
        help="Optional failure taxonomy label to attach to the emitted score.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Score one SceneFunc3D runner result and print compact JSON metrics."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        score = score_result_file(
            data_root=_namespace_path(args, "data_root"),
            result_path=_namespace_path(args, "result_path"),
            failure_type=_namespace_str(args, "failure_type"),
        )
    except CodexAgentError as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")
    print(json.dumps(score_to_payload(score), ensure_ascii=False))
    return 0


def score_to_payload(score: SceneFunc3dScore) -> SceneFunc3dScorePayload:
    """Convert an internal score object into the public CLI JSON payload."""
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


def _namespace_path(args: argparse.Namespace, name: str) -> Path:
    value: object = getattr(args, name)
    if not isinstance(value, Path):
        raise TypeError(f"argparse field {name!r} must be a Path")
    return value


def _namespace_str(args: argparse.Namespace, name: str) -> str:
    value: object = getattr(args, name)
    if not isinstance(value, str):
        raise TypeError(f"argparse field {name!r} must be a string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
