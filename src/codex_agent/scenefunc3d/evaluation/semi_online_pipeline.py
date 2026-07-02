"""Semi-online SceneFunc3D validation (spec tier 2).

For each sample under a completed run root, take the largest saved fragment as
a fixed seed anchor and run the full deterministic B/C/D pipeline through the
real Molmo/SAM sidecars, then re-score against hidden GT. Needs healthy Molmo
and SAM sidecars; does NOT need the ModelHub adapter.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence, Set
from pathlib import Path
from typing import TypedDict

from codex_agent.errors import CodexAgentError
from codex_agent.scenefunc3d.backends.anchor import TargetAnchor, build_anchor
from codex_agent.scenefunc3d.backends.fusion import FusionParams
from codex_agent.scenefunc3d.backends.lift_3d import load_scene_mesh_vertices
from codex_agent.scenefunc3d.evaluation.offline_fusion import (
    load_sample_fusion_input,
)
from codex_agent.scenefunc3d.evaluation.scorer import (
    SceneFunc3dScore,
    load_gt_point_ids,
    score_point_ids,
)
from codex_agent.scenefunc3d.pipeline import (
    FrameProposer,
    run_anchor_multiview_pipeline,
)
from codex_agent.scenefunc3d.pipeline_backends import SidecarFrameProposer
from codex_agent.scenefunc3d.sample import load_sample, scene_dir_for
from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene


class SemiOnlineRowPayload(TypedDict):
    """JSON-ready metrics for one semi-online sample."""

    sample_id: str
    iou: float
    precision: float
    recall: float
    predicted_count: int


class SemiOnlineResultPayload(TypedDict):
    """JSON-ready result of a semi-online run."""

    sample_count: int
    mean_iou: float
    mean_precision: float
    ap50: float
    rows: list[SemiOnlineRowPayload]


def score_semi_online_sample(
    *,
    sample_id: str,
    scene: SceneFunc3dToolScene,
    anchor: TargetAnchor,
    gt_ids: Set[int],
    proposer: FrameProposer,
    params: FusionParams,
    frame_cap: int,
) -> SceneFunc3dScore:
    """Run the full pipeline for one sample and score it (pure assembly)."""
    scene_vertices = load_scene_mesh_vertices(scene.raw_mesh_path)
    result = run_anchor_multiview_pipeline(
        anchor=anchor,
        scene=scene,
        scene_vertices=scene_vertices,
        proposer=proposer,
        params=params,
        frame_cap=frame_cap,
    )
    return score_point_ids(
        sample_id=sample_id,
        predicted_ids=frozenset(result.fused.point_indices),
        gt_ids=gt_ids,
    )


def run_semi_online(
    *,
    run_root: Path,
    data_root: Path,
    backend_config_path: Path,
    out_dir: Path,
    params: FusionParams,
    frame_cap: int,
    sample_ids: Sequence[str],
) -> SemiOnlineResultPayload:
    """Run the semi-online pipeline for each sample and aggregate metrics."""
    rows: list[SemiOnlineRowPayload] = []
    ious: list[float] = []
    precisions: list[float] = []
    hits50 = 0
    for sample_id in sample_ids:
        sample = load_sample(data_root, sample_id)
        scene = SceneFunc3dToolScene.load(scene_dir_for(data_root, sample.visit_id))
        fusion_input = load_sample_fusion_input(
            run_root=run_root, data_root=data_root, sample_id=sample_id
        )
        anchor = build_anchor(
            fusion_input.scene_vertices[list(fusion_input.anchor_point_indices)],
            motion_type=fusion_input.motion_type,
            seed_frame_id="semi_online_seed",
        )
        proposer: FrameProposer = SidecarFrameProposer(
            scene=scene,
            out_dir=out_dir / sample.visit_id / sample.desc_id,
            backend_config_path=backend_config_path,
            affordance_concept=sample.task_description,
            task_description=sample.task_description,
        )
        score = score_semi_online_sample(
            sample_id=sample_id,
            scene=scene,
            anchor=anchor,
            gt_ids=load_gt_point_ids(data_root, sample_id),
            proposer=proposer,
            params=params,
            frame_cap=frame_cap,
        )
        ious.append(score.metrics.iou)
        precisions.append(score.metrics.precision)
        hits50 += int(score.metrics.iou >= 0.50)
        rows.append(
            {
                "sample_id": sample_id,
                "iou": score.metrics.iou,
                "precision": score.metrics.precision,
                "recall": score.metrics.recall,
                "predicted_count": score.metrics.predicted_count,
            }
        )
    n = max(len(sample_ids), 1)
    return {
        "sample_count": len(sample_ids),
        "mean_iou": sum(ious) / n,
        "mean_precision": sum(precisions) / n,
        "ap50": hits50 / n,
        "rows": rows,
    }


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


def _namespace_int(args: argparse.Namespace, name: str) -> int:
    value: object = getattr(args, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"argparse field {name!r} must be an int")
    return value


def _namespace_float(args: argparse.Namespace, name: str) -> float:
    value: object = getattr(args, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"argparse field {name!r} must be a float")
    return float(value)


def _sample_ids_arg(args: argparse.Namespace) -> tuple[str, ...]:
    from codex_agent.scenefunc3d.evaluation.offline_fusion import (
        _sample_ids_from_run_root,
    )

    value: object = args.sample_ids
    if value:
        if not isinstance(value, list):
            raise TypeError("sample_ids must be a list")
        return tuple(str(item) for item in value)
    return _sample_ids_from_run_root(_namespace_path(args, "run_root"))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semi-online SceneFunc3D pipeline")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backend-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frame-cap", type=int, default=12)
    parser.add_argument("--agreement-tau", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: run the semi-online pipeline and print JSON."""
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        summary = run_semi_online(
            run_root=_namespace_path(args, "run_root"),
            data_root=_namespace_path(args, "data_root"),
            backend_config_path=_namespace_path(args, "backend_config"),
            out_dir=_namespace_path(args, "out_dir"),
            params=FusionParams(agreement_tau=_namespace_float(args, "agreement_tau")),
            frame_cap=_namespace_int(args, "frame_cap"),
            sample_ids=_sample_ids_arg(args),
        )
    except CodexAgentError as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")
    text = json.dumps(summary, indent=2)
    output = _namespace_optional_path(args, "output")
    if output is not None:
        output.write_text(text, encoding="utf-8")
    print(text)
    return 0


__all__ = [
    "SemiOnlineResultPayload",
    "SemiOnlineRowPayload",
    "main",
    "run_semi_online",
    "score_semi_online_sample",
]


if __name__ == "__main__":
    raise SystemExit(main())
