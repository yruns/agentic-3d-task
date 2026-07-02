"""Offline SceneFunc3D fusion sweep: rebuild bundles from saved artifacts,
re-fuse, and re-score against hidden GT. No model/adapter/sidecar calls."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
from zipfile import BadZipFile

import numpy as np

from codex_agent.errors import CodexAgentError, SceneFunc3dDataError
from codex_agent.scenefunc3d.backends.anchor import build_anchor
from codex_agent.scenefunc3d.backends.fusion import (
    FrameLift,
    FusionParams,
    MultiViewLiftBundle,
    fuse_multiview_points,
)
from codex_agent.scenefunc3d.backends.lift_3d import (
    FloatArray,
    load_scene_mesh_vertices,
)
from codex_agent.scenefunc3d.evaluation.scorer import (
    SceneFunc3dScore,
    load_gt_point_ids,
    score_point_ids,
)
from codex_agent.scenefunc3d.sample import load_sample, scene_dir_for

_FRAGMENT_GLOB = "*/mask_data.npz"
_FRAGMENT_POINT_INDICES_KEY = "point_indices"


class SweepParamsPayload(TypedDict):
    """JSON-ready fusion parameters for one offline sweep row."""

    agreement_tau: float
    radius_scale: float
    cluster_link_eps_m: float
    min_cluster_points: int


class SweepRowPayload(TypedDict):
    """JSON-ready metrics for one offline sweep parameter combination."""

    params: SweepParamsPayload
    mean_iou: float
    mean_precision: float
    ap25: float
    ap50: float


class SweepResultPayload(TypedDict):
    """JSON-ready result of an offline SceneFunc3D fusion sweep."""

    sample_count: int
    results: list[SweepRowPayload]


@dataclass(frozen=True)
class SampleFusionInput:
    """Everything needed to fuse + score one sample offline."""

    sample_id: str
    motion_type: str
    frames_point_indices: tuple[tuple[int, ...], ...]
    anchor_point_indices: tuple[int, ...]
    scene_vertices: FloatArray
    gt_ids: Set[int]


def fuse_and_score_sample(
    sample: SampleFusionInput, params: FusionParams
) -> SceneFunc3dScore:
    """Fuse one sample's frames and score the result against its GT ids.

    The anchor is built from ``anchor_point_indices`` (the seed fragment), not
    the union, so a far spurious fragment cannot pull the anchor off-target.
    """
    if not sample.anchor_point_indices or not any(sample.frames_point_indices):
        return score_point_ids(
            sample_id=sample.sample_id, predicted_ids=frozenset(), gt_ids=sample.gt_ids
        )
    anchor = build_anchor(
        sample.scene_vertices[list(sample.anchor_point_indices)],
        motion_type=sample.motion_type,
        seed_frame_id="offline_seed",
    )
    frames = tuple(
        FrameLift(frame_id=f"offline_{position:03d}", point_indices=indices)
        for position, indices in enumerate(sample.frames_point_indices)
    )
    bundle = MultiViewLiftBundle(frames=frames, anchor=anchor)
    fused = fuse_multiview_points(bundle, sample.scene_vertices, params)
    return score_point_ids(
        sample_id=sample.sample_id,
        predicted_ids=frozenset(fused.point_indices),
        gt_ids=sample.gt_ids,
    )


def load_sample_fusion_input(
    *, run_root: Path, data_root: Path, sample_id: str
) -> SampleFusionInput:
    """Build a SampleFusionInput from a saved run root + dataset root."""
    sample = load_sample(data_root, sample_id)
    motion_type = sample.motion_hints[0].motion_type if sample.motion_hints else ""
    fragment_dir = run_root / sample.visit_id / sample.desc_id / "fragments"
    frames = _load_fragment_frames(fragment_dir)
    # Offline seed proxy: use the largest fragment as the anchor source. In the
    # online pipeline the anchor comes from the agent-confirmed seed instead.
    anchor_point_indices = max(frames, key=len)
    scene_vertices = load_scene_mesh_vertices(
        scene_dir_for(data_root, sample.visit_id) / "raw" / "mesh.ply"
    )
    return SampleFusionInput(
        sample_id=sample_id,
        motion_type=motion_type,
        frames_point_indices=frames,
        anchor_point_indices=anchor_point_indices,
        scene_vertices=scene_vertices,
        gt_ids=load_gt_point_ids(data_root, sample_id),
    )


def _load_fragment_frames(fragment_dir: Path) -> tuple[tuple[int, ...], ...]:
    """Load per-fragment raw-mesh vertex ids under one sample's fragments dir.

    Fails closed with :class:`SceneFunc3dDataError` when the directory is
    missing or holds no ``*/mask_data.npz`` fragments, so an offline sweep never
    silently scores an empty prediction for a sample with no saved evidence.
    """
    if not fragment_dir.is_dir():
        raise SceneFunc3dDataError(
            f"SceneFunc3D fragments directory is missing: {fragment_dir}"
        )
    frames = [
        _load_fragment_point_indices(npz_path)
        for npz_path in sorted(fragment_dir.glob(_FRAGMENT_GLOB))
    ]
    if not frames:
        raise SceneFunc3dDataError(
            "SceneFunc3D fragments directory contains no "
            f"{_FRAGMENT_GLOB} fragments: {fragment_dir}"
        )
    return tuple(frames)


def _load_fragment_point_indices(npz_path: Path) -> tuple[int, ...]:
    """Load one fragment NPZ's raw-mesh vertex ids, failing closed on bad data."""
    try:
        with np.load(npz_path) as archive:
            if _FRAGMENT_POINT_INDICES_KEY not in archive.files:
                raise SceneFunc3dDataError(
                    "SceneFunc3D fragment NPZ is missing required key "
                    f"{_FRAGMENT_POINT_INDICES_KEY!r}: npz_path={npz_path}"
                )
            point_indices_array = np.asarray(archive[_FRAGMENT_POINT_INDICES_KEY])
            indices = point_indices_array.astype(np.int64).ravel()
    except SceneFunc3dDataError:
        raise
    except (BadZipFile, OSError, ValueError) as exc:
        raise SceneFunc3dDataError(
            "could not load SceneFunc3D fragment NPZ: "
            f"npz_path={npz_path}; error_type={exc.__class__.__name__}"
        ) from exc
    return tuple(int(value) for value in indices)


def _sample_ids_from_run_root(run_root: Path) -> tuple[str, ...]:
    """Return every ``<visit_id>::<desc_id>`` sample saved under a run root.

    Fails closed with :class:`SceneFunc3dDataError` when the run root is missing
    or holds no fragment samples, mirroring the results-dir contract in
    ``evaluation/__main__.py``.
    """
    if not run_root.is_dir():
        raise SceneFunc3dDataError(f"SceneFunc3D run root is missing: {run_root}")
    sample_ids: list[str] = []
    for visit_dir in sorted(
        p for p in run_root.iterdir() if p.is_dir() and p.name.isdigit()
    ):
        for desc_dir in sorted(
            p for p in visit_dir.iterdir() if (p / "fragments").is_dir()
        ):
            sample_ids.append(f"{visit_dir.name}::{desc_dir.name}")
    if not sample_ids:
        raise SceneFunc3dDataError(
            f"SceneFunc3D run root contains no fragment samples: {run_root}"
        )
    return tuple(sample_ids)


def run_offline_sweep(
    *, run_root: Path, data_root: Path, param_grid: tuple[FusionParams, ...]
) -> SweepResultPayload:
    """Fuse+score every sample under run_root for each params set."""
    sample_ids = _sample_ids_from_run_root(run_root)
    results: list[SweepRowPayload] = []
    for params in param_grid:
        ious: list[float] = []
        precisions: list[float] = []
        hits25 = 0
        hits50 = 0
        for sample_id in sample_ids:
            sample = load_sample_fusion_input(
                run_root=run_root, data_root=data_root, sample_id=sample_id
            )
            score = fuse_and_score_sample(sample, params)
            ious.append(score.metrics.iou)
            precisions.append(score.metrics.precision)
            hits25 += int(score.metrics.iou >= 0.25)
            hits50 += int(score.metrics.iou >= 0.50)
        n = len(sample_ids)
        results.append(
            {
                "params": {
                    "agreement_tau": params.agreement_tau,
                    "radius_scale": params.radius_scale,
                    "cluster_link_eps_m": params.cluster_link_eps_m,
                    "min_cluster_points": params.min_cluster_points,
                },
                "mean_iou": sum(ious) / n,
                "mean_precision": sum(precisions) / n,
                "ap25": hits25 / n,
                "ap50": hits50 / n,
            }
        )
    return {"sample_count": len(sample_ids), "results": results}


def _default_param_grid() -> tuple[FusionParams, ...]:
    grid: list[FusionParams] = []
    for tau in (0.3, 0.5, 0.7):
        for eps in (0.02, 0.04):
            grid.append(FusionParams(agreement_tau=tau, cluster_link_eps_m=eps))
    return tuple(grid)


def _sweep_cli_payload(args: argparse.Namespace) -> SweepResultPayload:
    """Run the sweep from validated CLI arguments (keeps main free of logic)."""
    return run_offline_sweep(
        run_root=_namespace_path(args, "run_root"),
        data_root=_namespace_path(args, "data_root"),
        param_grid=_default_param_grid(),
    )


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


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the offline SceneFunc3D fusion sweep CLI parser."""
    parser = argparse.ArgumentParser(description="Offline SceneFunc3D fusion sweep")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: run the offline fusion sweep and print JSON."""
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        summary = _sweep_cli_payload(args)
    except CodexAgentError as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")
    output_path = _namespace_optional_path(args, "output")
    text = json.dumps(summary, indent=2)
    if output_path is not None:
        output_path.write_text(text, encoding="utf-8")
    print(text)
    return 0


__all__ = [
    "SampleFusionInput",
    "SweepParamsPayload",
    "SweepResultPayload",
    "SweepRowPayload",
    "fuse_and_score_sample",
    "load_sample_fusion_input",
    "main",
    "run_offline_sweep",
]


if __name__ == "__main__":
    raise SystemExit(main())
