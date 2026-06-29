"""Score SceneFunc3D runner ``result.json`` files from the command line."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

from codex_agent.errors import CodexAgentError, SceneFunc3dDataError

from .payloads import (
    SceneFunc3dScoreManifestPayload,
    SceneFunc3dScorePayload,
    score_manifest_to_payload,
    score_to_payload,
)
from .scorer import SceneFunc3dScore, score_result_file

SceneFunc3dEvaluationPayload: TypeAlias = (
    SceneFunc3dScorePayload | SceneFunc3dScoreManifestPayload
)


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
    result_source = parser.add_mutually_exclusive_group(required=True)
    result_source.add_argument(
        "--result-path",
        type=Path,
        help="Path to one SceneFunc3D runner result.json.",
    )
    result_source.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing SceneFunc3D runner result.json files.",
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
        payload = _score_cli_payload(args)
    except CodexAgentError as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _score_cli_payload(args: argparse.Namespace) -> SceneFunc3dEvaluationPayload:
    data_root = _namespace_path(args, "data_root")
    failure_type = _namespace_str(args, "failure_type")
    result_path = _namespace_optional_path(args, "result_path")
    if result_path is not None:
        return score_to_payload(
            score_result_file(
                data_root=data_root,
                result_path=result_path,
                failure_type=failure_type,
            )
        )

    results_dir = _namespace_optional_path(args, "results_dir")
    if results_dir is None:
        raise SceneFunc3dDataError("one of --result-path or --results-dir is required")
    result_paths = _result_paths_in_dir(results_dir)
    scores: list[SceneFunc3dScore] = []
    for result_file_path in result_paths:
        scores.append(
            score_result_file(
                data_root=data_root,
                result_path=result_file_path,
                failure_type=failure_type,
            )
        )
    return score_manifest_to_payload(
        scores=scores,
        result_paths=[str(path) for path in result_paths],
    )


def _result_paths_in_dir(results_dir: Path) -> tuple[Path, ...]:
    root = Path(results_dir)
    if not root.is_dir():
        raise SceneFunc3dDataError(f"results directory is missing: {root}")
    result_paths = tuple(sorted(root.rglob("result.json")))
    if not result_paths:
        raise SceneFunc3dDataError(f"results directory contains no result.json: {root}")
    return result_paths


def _namespace_path(args: argparse.Namespace, name: str) -> Path:
    value: object = getattr(args, name)
    if not isinstance(value, Path):
        raise TypeError(f"argparse field {name!r} must be a Path")
    return value


def _namespace_optional_path(args: argparse.Namespace, name: str) -> Path | None:
    value: object = getattr(args, name)
    if value is None:
        return None
    if not isinstance(value, Path):
        raise TypeError(f"argparse field {name!r} must be a Path or None")
    return value


def _namespace_str(args: argparse.Namespace, name: str) -> str:
    value: object = getattr(args, name)
    if not isinstance(value, str):
        raise TypeError(f"argparse field {name!r} must be a string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
