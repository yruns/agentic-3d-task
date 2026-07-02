"""Offline SceneFunc3D fusion sweep: rebuild bundles from saved artifacts,
re-fuse, and re-score against hidden GT. No model/adapter/sidecar calls."""

from __future__ import annotations

import argparse
import json
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    frames: list[tuple[int, ...]] = []
    for npz_path in sorted(fragment_dir.glob("*/mask_data.npz")):
        with np.load(npz_path) as archive:
            indices = np.asarray(archive["point_indices"]).astype(np.int64).ravel()
        frames.append(tuple(int(value) for value in indices))
    # Offline seed proxy: use the largest fragment as the anchor source. In the
    # online pipeline the anchor comes from the agent-confirmed seed instead.
    anchor_point_indices = max(frames, key=len) if frames else ()
    scene_vertices = load_scene_mesh_vertices(
        scene_dir_for(data_root, sample.visit_id) / "raw" / "mesh.ply"
    )
    return SampleFusionInput(
        sample_id=sample_id,
        motion_type=motion_type,
        frames_point_indices=tuple(frames),
        anchor_point_indices=anchor_point_indices,
        scene_vertices=scene_vertices,
        gt_ids=load_gt_point_ids(data_root, sample_id),
    )


def _sample_ids_from_run_root(run_root: Path) -> tuple[str, ...]:
    sample_ids: list[str] = []
    for visit_dir in sorted(
        p for p in run_root.iterdir() if p.is_dir() and p.name.isdigit()
    ):
        for desc_dir in sorted(
            p for p in visit_dir.iterdir() if (p / "fragments").is_dir()
        ):
            sample_ids.append(f"{visit_dir.name}::{desc_dir.name}")
    return tuple(sample_ids)


def run_offline_sweep(
    *, run_root: Path, data_root: Path, param_grid: tuple[FusionParams, ...]
) -> dict[str, object]:
    """Fuse+score every sample under run_root for each params set."""
    sample_ids = _sample_ids_from_run_root(run_root)
    results: list[dict[str, object]] = []
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
        n = max(len(sample_ids), 1)
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


def main(argv: list[str] | None = None) -> int:
    """CLI entry: run the offline fusion sweep and print JSON."""
    parser = argparse.ArgumentParser(description="Offline SceneFunc3D fusion sweep")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = run_offline_sweep(
        run_root=args.run_root,
        data_root=args.data_root,
        param_grid=_default_param_grid(),
    )
    text = json.dumps(summary, indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
