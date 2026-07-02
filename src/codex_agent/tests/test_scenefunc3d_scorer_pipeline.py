"""Tests for scoring Phase-2c pipeline result files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.lift_3d import write_lift_npz
from codex_agent.scenefunc3d.evaluation.scorer import (
    score_pipeline_result_file,
    score_result_file,
)
from codex_agent.tests.scenefunc3d_scene_fixtures import write_scoring_scene


def _pipeline_result_json(result_path: Path, *, mask_npz_path: Path) -> None:
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": "421254::desc-a",
                "result_kind": "anchor_multiview_pipeline",
                "outcome": {
                    "result_kind": "anchor_multiview_pipeline",
                    "mask_npz_path": str(mask_npz_path),
                    "mask_ply_path": str(mask_npz_path.with_name("lifted_points.ply")),
                    "confidence": 0.5,
                    "selected_frame_ids": ["000010"],
                    "seed": {
                        "seed_frame_id": "000010",
                        "seed_candidate_id": "cand-a",
                        "seed_fragment_id": "000010_cand-a",
                        "affordance_concept": "the handle",
                        "seed_point_count": 2,
                    },
                    "per_frame": [],
                    "fusion_params": {
                        "agreement_tau": 0.5,
                        "radius_scale": 1.0,
                        "cluster_link_eps_m": 0.02,
                        "min_cluster_points": 10,
                    },
                    "uncertainties": [],
                },
                "turn": {
                    "turn_id": "t",
                    "status": "completed",
                    "duration_ms": None,
                    "reasoning_summary": None,
                    "run_home": None,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_fused(npz_path: Path, ids: tuple[int, ...]) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    write_lift_npz(
        npz_path,
        np.zeros((len(ids), 3), dtype=np.float64),
        point_indices=np.array(ids, dtype=np.int64),
    )


def test_score_pipeline_result_file_scores_fused_npz(tmp_path: Path) -> None:
    write_scoring_scene(tmp_path / "data", gt_point_ids=(0, 1, 2))
    fused = tmp_path / "run" / "fused" / "mask_data.npz"
    _write_fused(fused, (0, 1, 2))
    result_path = tmp_path / "run" / "result.json"
    _pipeline_result_json(result_path, mask_npz_path=fused)
    score = score_pipeline_result_file(
        data_root=tmp_path / "data", result_path=result_path
    )
    assert score.metrics.iou == pytest.approx(1.0)


def test_score_result_file_dispatches_on_result_kind(tmp_path: Path) -> None:
    write_scoring_scene(tmp_path / "data", gt_point_ids=(0, 1, 2))
    fused = tmp_path / "run" / "fused" / "mask_data.npz"
    _write_fused(fused, (0, 1))
    result_path = tmp_path / "run" / "result.json"
    _pipeline_result_json(result_path, mask_npz_path=fused)
    score = score_result_file(data_root=tmp_path / "data", result_path=result_path)
    assert score.metrics.iou == pytest.approx(2.0 / 3.0)
