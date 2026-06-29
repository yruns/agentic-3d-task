"""Tests for SceneFunc3D mask metrics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.evaluation import __main__ as evaluation_cli
from codex_agent.scenefunc3d.evaluation.metrics import (
    MaskMetrics,
    compute_mask_metrics,
)
from codex_agent.scenefunc3d.evaluation.scorer import (
    load_gt_point_ids,
    load_predicted_point_ids,
    score_mask_artifact,
    score_mask_npz,
    score_point_ids,
    score_result_file,
)
from codex_agent.scenefunc3d.task import FRAGMENT_APPROVAL_ACTIONS
from codex_agent.scenefunc3d.tools.mask_lifting import LiftMaskArgs, LiftMaskResult


def test_compute_mask_metrics() -> None:
    metrics = compute_mask_metrics(predicted_ids={1, 2, 3}, gt_ids={2, 3, 4, 5})
    assert metrics == MaskMetrics(
        iou=0.4,
        precision=2 / 3,
        recall=0.5,
        f1=4 / 7,
        predicted_count=3,
        gt_count=4,
    )


def test_compute_empty_metrics() -> None:
    metrics = compute_mask_metrics(predicted_ids=set(), gt_ids={1, 2})
    assert metrics.iou == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_compute_both_empty_metrics() -> None:
    metrics = compute_mask_metrics(predicted_ids=set(), gt_ids=set())
    assert metrics == MaskMetrics(
        iou=0.0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        predicted_count=0,
        gt_count=0,
    )


def test_compute_perfect_match_metrics() -> None:
    metrics = compute_mask_metrics(predicted_ids={10, 20}, gt_ids={10, 20})
    assert metrics == MaskMetrics(
        iou=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        predicted_count=2,
        gt_count=2,
    )


def test_score_point_ids_wraps_metrics_and_failure_type() -> None:
    score = score_point_ids(
        sample_id="visit_001_desc_002",
        predicted_ids={1, 5},
        gt_ids={1, 2, 3},
        failure_type="partial_lift",
    )

    assert score.sample_id == "visit_001_desc_002"
    assert score.failure_type == "partial_lift"
    assert score.metrics == MaskMetrics(
        iou=0.25,
        precision=0.5,
        recall=1 / 3,
        f1=0.4,
        predicted_count=2,
        gt_count=3,
    )


def test_load_gt_point_ids_reads_hidden_annotation_indices(tmp_path: Path) -> None:
    _write_scoring_scene(tmp_path)

    gt_ids = load_gt_point_ids(tmp_path, "421254::desc-a")

    assert gt_ids == frozenset({3, 5, 8, 13, 21})


def test_load_predicted_point_ids_accepts_point_indices_npz(tmp_path: Path) -> None:
    mask_npz_path = tmp_path / "mask_data.npz"
    np.savez_compressed(mask_npz_path, point_indices=np.array([3, 8, 99]))

    predicted_ids = load_predicted_point_ids(mask_npz_path)

    assert predicted_ids == frozenset({3, 8, 99})


def test_load_predicted_point_ids_accepts_point_ids_npz(tmp_path: Path) -> None:
    mask_npz_path = tmp_path / "mask_data.npz"
    np.savez_compressed(mask_npz_path, point_ids=np.array([5, 21]))

    predicted_ids = load_predicted_point_ids(mask_npz_path)

    assert predicted_ids == frozenset({5, 21})


def test_load_predicted_point_ids_rejects_points_world_only_npz(
    tmp_path: Path,
) -> None:
    mask_npz_path = tmp_path / "mask_data.npz"
    points_world = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    np.savez_compressed(mask_npz_path, points_world=points_world)

    with pytest.raises(SceneFunc3dDataError, match="point ids"):
        load_predicted_point_ids(mask_npz_path)


def test_score_mask_npz_loads_prediction_and_hidden_gt(tmp_path: Path) -> None:
    _write_scoring_scene(tmp_path)
    mask_npz_path = tmp_path / "mask_data.npz"
    np.savez_compressed(mask_npz_path, point_indices=np.array([3, 8, 99]))

    score = score_mask_npz(
        data_root=tmp_path,
        sample_id="421254::desc-a",
        mask_npz_path=mask_npz_path,
        failure_type="partial_lift",
    )

    assert score.sample_id == "421254::desc-a"
    assert score.failure_type == "partial_lift"
    assert score.metrics == MaskMetrics(
        iou=2 / 6,
        precision=2 / 3,
        recall=2 / 5,
        f1=0.5,
        predicted_count=3,
        gt_count=5,
    )


def test_score_mask_artifact_uses_final_artifact_npz_path(tmp_path: Path) -> None:
    _write_scoring_scene(tmp_path)
    mask_npz_path = tmp_path / "mask_data.npz"
    np.savez_compressed(mask_npz_path, point_indices=np.array([3, 5, 8, 13, 21]))
    artifact_path = tmp_path / "mask_artifact.json"
    _write_scoring_mask_artifact(artifact_path, mask_npz_path=mask_npz_path)

    score = score_mask_artifact(
        data_root=tmp_path,
        sample_id="421254::desc-a",
        mask_artifact_path=artifact_path,
    )

    assert score.metrics == MaskMetrics(
        iou=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        predicted_count=5,
        gt_count=5,
    )


def test_score_result_file_uses_runner_result_artifact_path(tmp_path: Path) -> None:
    _write_scoring_scene(tmp_path)
    result_dir = tmp_path / "result"
    fused_dir = result_dir / "fused"
    artifact_path = fused_dir / "mask_artifact.json"
    mask_npz_path = fused_dir / "mask_data.npz"
    mask_ply_path = fused_dir / "lifted_points.ply"
    _write_valid_scoring_mask_artifact(
        artifact_path,
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
        point_indices=(3, 5, 8, 13, 21),
    )
    result_path = result_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": "421254::desc-a",
                "outcome": {
                    "mask_artifact_path": "fused/mask_artifact.json",
                    "mask_npz_path": "fused/mask_data.npz",
                    "mask_ply_path": "fused/lifted_points.ply",
                    "selected_frame_ids": ["000010"],
                    "accepted_fragment_ids": ["frag-a"],
                    "confidence": 0.9,
                    "uncertainties": [],
                },
                "turn": {},
            }
        ),
        encoding="utf-8",
    )

    score = score_result_file(data_root=tmp_path, result_path=result_path)

    assert score.sample_id == "421254::desc-a"
    assert score.metrics == MaskMetrics(
        iou=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        predicted_count=5,
        gt_count=5,
    )


def test_score_result_file_rejects_outcome_mask_npz_mismatch(tmp_path: Path) -> None:
    _write_scoring_scene(tmp_path)
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    mask_npz_path = result_dir / "fused" / "mask_data.npz"
    mask_ply_path = result_dir / "fused" / "lifted_points.ply"
    artifact_path = result_dir / "fused" / "mask_artifact.json"
    _write_valid_scoring_mask_artifact(
        artifact_path,
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
        point_indices=(3, 5),
    )
    wrong_mask_npz_path = result_dir / "fused" / "wrong_mask_data.npz"
    result_path = result_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": "421254::desc-a",
                "outcome": {
                    "mask_artifact_path": str(artifact_path),
                    "mask_npz_path": str(wrong_mask_npz_path),
                    "mask_ply_path": str(mask_ply_path),
                    "selected_frame_ids": ["000010"],
                    "accepted_fragment_ids": ["frag-a"],
                    "confidence": 0.9,
                    "uncertainties": [],
                },
                "turn": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SceneFunc3dDataError, match="mask_npz_path"):
        score_result_file(data_root=tmp_path, result_path=result_path)


def test_score_result_file_rejects_existing_artifact_document_mask_npz_mismatch(
    tmp_path: Path,
) -> None:
    _write_scoring_scene(tmp_path)
    result_dir = tmp_path / "copied_case"
    fused_dir = result_dir / "fused"
    fused_dir.mkdir(parents=True)
    mask_npz_path = fused_dir / "mask_data.npz"
    mask_ply_path = fused_dir / "lifted_points.ply"
    artifact_path = fused_dir / "mask_artifact.json"
    original_run_root = tmp_path / "original_run" / "agent_outputs" / "421254"
    original_mask_npz_path = original_run_root / "desc-a" / "fused" / "mask_data.npz"
    _write_valid_scoring_mask_artifact_with_recorded_paths(
        artifact_path,
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
        point_indices=(3, 5, 8, 13, 21),
        recorded_mask_npz_path=original_mask_npz_path,
        recorded_mask_ply_path=mask_ply_path,
    )
    _write_scoring_npz(original_mask_npz_path, point_indices=(99,))
    result_path = result_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": "421254::desc-a",
                "outcome": {
                    "mask_artifact_path": str(artifact_path),
                    "mask_npz_path": str(mask_npz_path),
                    "mask_ply_path": str(mask_ply_path),
                    "selected_frame_ids": ["000010"],
                    "accepted_fragment_ids": ["frag-a"],
                    "confidence": 0.9,
                    "uncertainties": [],
                },
                "turn": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SceneFunc3dDataError, match="mask_npz_path"):
        score_result_file(data_root=tmp_path, result_path=result_path)


def test_score_result_file_falls_back_to_copied_artifact_members(
    tmp_path: Path,
) -> None:
    _write_scoring_scene(tmp_path)
    result_dir = tmp_path / "copied_case"
    fused_dir = result_dir / "fused"
    fused_dir.mkdir(parents=True)
    mask_npz_path = fused_dir / "mask_data.npz"
    mask_ply_path = fused_dir / "lifted_points.ply"
    artifact_path = fused_dir / "mask_artifact.json"
    stale_run_root = tmp_path / "missing_run" / "agent_outputs" / "421254" / "desc-a"
    _write_valid_scoring_mask_artifact(
        artifact_path,
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
        point_indices=(3, 5, 8, 13, 21),
    )
    result_path = result_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": "421254::desc-a",
                "outcome": {
                    "mask_artifact_path": str(
                        stale_run_root / "fused" / "mask_artifact.json"
                    ),
                    "mask_npz_path": str(stale_run_root / "fused" / "mask_data.npz"),
                    "mask_ply_path": str(
                        stale_run_root / "fused" / "lifted_points.ply"
                    ),
                    "selected_frame_ids": ["000010"],
                    "accepted_fragment_ids": ["frag-a"],
                    "confidence": 0.9,
                    "uncertainties": [],
                },
                "turn": {},
            }
        ),
        encoding="utf-8",
    )

    score = score_result_file(data_root=tmp_path, result_path=result_path)

    assert score.metrics == MaskMetrics(
        iou=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        predicted_count=5,
        gt_count=5,
    )


def test_score_result_file_falls_back_to_copied_artifact_document_members(
    tmp_path: Path,
) -> None:
    _write_scoring_scene(tmp_path)
    result_dir = tmp_path / "copied_case"
    fused_dir = result_dir / "fused"
    fused_dir.mkdir(parents=True)
    mask_npz_path = fused_dir / "mask_data.npz"
    mask_ply_path = fused_dir / "lifted_points.ply"
    artifact_path = fused_dir / "mask_artifact.json"
    stale_run_root = tmp_path / "missing_run" / "agent_outputs" / "421254" / "desc-a"
    stale_mask_npz_path = stale_run_root / "fused" / "mask_data.npz"
    stale_mask_ply_path = stale_run_root / "fused" / "lifted_points.ply"
    _write_valid_scoring_mask_artifact_with_recorded_paths(
        artifact_path,
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
        point_indices=(3, 5, 8, 13, 21),
        recorded_mask_npz_path=stale_mask_npz_path,
        recorded_mask_ply_path=stale_mask_ply_path,
    )
    result_path = result_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": "421254::desc-a",
                "outcome": {
                    "mask_artifact_path": str(
                        stale_run_root / "fused" / "mask_artifact.json"
                    ),
                    "mask_npz_path": str(stale_mask_npz_path),
                    "mask_ply_path": str(stale_mask_ply_path),
                    "selected_frame_ids": ["000010"],
                    "accepted_fragment_ids": ["frag-a"],
                    "confidence": 0.9,
                    "uncertainties": [],
                },
                "turn": {},
            }
        ),
        encoding="utf-8",
    )

    score = score_result_file(data_root=tmp_path, result_path=result_path)

    assert score.metrics == MaskMetrics(
        iou=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        predicted_count=5,
        gt_count=5,
    )


def test_score_result_file_falls_back_to_copied_standard_fragment_final_mask(
    tmp_path: Path,
) -> None:
    _write_scoring_scene(tmp_path)
    result_dir = tmp_path / "copied_case"
    fused_dir = result_dir / "fused"
    fused_dir.mkdir(parents=True)
    mask_npz_path = fused_dir / "mask_data.npz"
    mask_ply_path = fused_dir / "lifted_points.ply"
    artifact_path = fused_dir / "mask_artifact.json"
    stale_run_root = tmp_path / "missing_run" / "agent_outputs" / "421254" / "desc-a"
    stale_mask_npz_path = stale_run_root / "fused" / "mask_data.npz"
    stale_mask_ply_path = stale_run_root / "fused" / "lifted_points.ply"
    point_indices = (3, 5, 8, 13, 21)
    _write_scoring_mask_members(
        mask_npz_path,
        mask_ply_path=mask_ply_path,
        point_indices=point_indices,
    )
    review_artifacts = _write_standard_review_artifacts(
        result_dir,
        frame_id="000010",
        candidate_id="mask_00",
    )
    artifact_path.write_text(
        json.dumps(
            {
                "accepted_frame_ids": ["000010"],
                "accepted_fragments": [
                    {
                        "fragment_id": "000010_mask_00",
                        "frame_id": "000010",
                        "point_count": len(point_indices),
                        "lift_geometry": _lift_geometry_payload_for_point_count(
                            len(point_indices)
                        ),
                        "approval_actions": [
                            action.value for action in FRAGMENT_APPROVAL_ACTIONS
                        ],
                        "review_artifacts": review_artifacts,
                    }
                ],
                "multi_view_decision": {
                    "seed_fragment_id": "000010_mask_00",
                    "action": "stop",
                    "reason": "first lift covers the target part for this fixture",
                    "suggested_frame_ids": [],
                },
                "mask_npz_path": str(stale_mask_npz_path),
                "mask_ply_path": str(stale_mask_ply_path),
            }
        ),
        encoding="utf-8",
    )
    result_path = result_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": "421254::desc-a",
                "outcome": {
                    "mask_artifact_path": str(
                        stale_run_root / "fused" / "mask_artifact.json"
                    ),
                    "mask_npz_path": str(stale_mask_npz_path),
                    "mask_ply_path": str(stale_mask_ply_path),
                    "selected_frame_ids": ["000010"],
                    "accepted_fragment_ids": ["000010_mask_00"],
                    "confidence": 0.9,
                    "uncertainties": [],
                },
                "turn": {},
            }
        ),
        encoding="utf-8",
    )

    score = score_result_file(data_root=tmp_path, result_path=result_path)

    assert score.metrics == MaskMetrics(
        iou=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        predicted_count=5,
        gt_count=5,
    )


def test_score_result_file_rejects_ambiguous_copied_artifact_member(
    tmp_path: Path,
) -> None:
    _write_scoring_scene(tmp_path)
    result_dir = tmp_path / "copied_case"
    fused_dir = result_dir / "fused"
    fused_dir.mkdir(parents=True)
    mask_npz_path = fused_dir / "mask_data.npz"
    stale_run_root = tmp_path / "missing_run" / "agent_outputs" / "421254" / "desc-a"
    artifact_path = fused_dir / "mask_artifact.json"
    _write_scoring_mask_artifact(
        artifact_path,
        mask_npz_path=stale_run_root / "fused" / "mask_data.npz",
    )
    _write_scoring_mask_artifact(
        result_dir / "mask_artifact.json",
        mask_npz_path=stale_run_root / "fused" / "mask_data.npz",
    )
    np.savez_compressed(mask_npz_path, point_indices=np.array([3, 5]))
    result_path = result_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": "421254::desc-a",
                "outcome": {
                    "mask_artifact_path": str(
                        stale_run_root / "fused" / "mask_artifact.json"
                    ),
                    "mask_npz_path": str(stale_run_root / "fused" / "mask_data.npz"),
                    "mask_ply_path": str(stale_run_root / "fused" / "mask.ply"),
                    "selected_frame_ids": ["000010"],
                    "accepted_fragment_ids": ["frag-a"],
                    "confidence": 0.9,
                    "uncertainties": [],
                },
                "turn": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SceneFunc3dDataError, match="ambiguous"):
        score_result_file(data_root=tmp_path, result_path=result_path)


def test_score_result_file_rejects_unrelated_copied_artifact_basename(
    tmp_path: Path,
) -> None:
    _write_scoring_scene(tmp_path)
    result_dir = tmp_path / "copied_case"
    unrelated_fused_dir = result_dir / "other_case" / "fused"
    unrelated_fused_dir.mkdir(parents=True)
    unrelated_mask_npz_path = unrelated_fused_dir / "mask_data.npz"
    np.savez_compressed(
        unrelated_mask_npz_path,
        point_indices=np.array([3, 5, 8, 13, 21]),
    )
    _write_scoring_mask_artifact(
        unrelated_fused_dir / "mask_artifact.json",
        mask_npz_path=unrelated_mask_npz_path,
    )
    stale_run_root = tmp_path / "missing_run" / "agent_outputs" / "421254" / "desc-a"
    result_path = result_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": "421254::desc-a",
                "outcome": {
                    "mask_artifact_path": str(
                        stale_run_root / "fused" / "mask_artifact.json"
                    ),
                    "mask_npz_path": str(stale_run_root / "fused" / "mask_data.npz"),
                    "mask_ply_path": str(stale_run_root / "fused" / "mask.ply"),
                    "selected_frame_ids": ["000010"],
                    "accepted_fragment_ids": ["frag-a"],
                    "confidence": 0.9,
                    "uncertainties": [],
                },
                "turn": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SceneFunc3dDataError, match="mask_artifact_path"):
        score_result_file(data_root=tmp_path, result_path=result_path)


def test_evaluation_cli_scores_runner_result_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_scoring_scene(tmp_path)
    fused_dir = tmp_path / "fused"
    mask_npz_path = fused_dir / "mask_data.npz"
    mask_ply_path = fused_dir / "lifted_points.ply"
    artifact_path = fused_dir / "mask_artifact.json"
    _write_valid_scoring_mask_artifact(
        artifact_path,
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
        point_indices=(3, 5, 8, 13, 21),
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": "421254::desc-a",
                "outcome": {
                    "mask_artifact_path": str(artifact_path),
                    "mask_npz_path": str(mask_npz_path),
                    "mask_ply_path": str(mask_ply_path),
                    "selected_frame_ids": ["000010"],
                    "accepted_fragment_ids": ["frag-a"],
                    "confidence": 0.9,
                    "uncertainties": [],
                },
                "turn": {},
            }
        ),
        encoding="utf-8",
    )

    exit_code = evaluation_cli.main(
        [
            "--data-root",
            str(tmp_path),
            "--result-path",
            str(result_path),
        ]
    )

    assert exit_code == 0
    payload = _json_object(capsys.readouterr().out)
    assert payload["sample_id"] == "421254::desc-a"
    assert payload["failure_type"] == ""
    metrics = _json_object_member(payload, "metrics")
    assert metrics["iou"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["predicted_count"] == 5
    assert metrics["gt_count"] == 5


def test_evaluation_cli_scores_result_directory_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_scoring_scene(tmp_path)
    second_result_path = _write_scoring_result_file(
        tmp_path / "runs" / "421254" / "second" / "result.json",
        mask_point_indices=(3, 99),
    )
    first_result_path = _write_scoring_result_file(
        tmp_path / "runs" / "421254" / "first" / "result.json",
        mask_point_indices=(3, 5, 8, 13, 21),
    )

    exit_code = evaluation_cli.main(
        [
            "--data-root",
            str(tmp_path),
            "--results-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert exit_code == 0
    payload = _json_object(capsys.readouterr().out)
    assert payload["result_count"] == 2
    assert payload["result_paths"] == [
        str(first_result_path),
        str(second_result_path),
    ]
    scores = cast(list[object], payload["scores"])
    first_score = cast(Mapping[str, object], scores[0])
    second_score = cast(Mapping[str, object], scores[1])
    assert first_score["sample_id"] == "421254::desc-a"
    assert second_score["sample_id"] == "421254::desc-a"
    first_metrics = _json_object_member(first_score, "metrics")
    second_metrics = _json_object_member(second_score, "metrics")
    assert first_metrics["iou"] == 1.0
    assert second_metrics["iou"] == 1 / 6
    assert second_metrics["predicted_count"] == 2


def test_lift_mask_args_accepts_existing_input_files(tmp_path: Path) -> None:
    payload = _lift_mask_args_payload(tmp_path)

    args = LiftMaskArgs.model_validate(payload)

    assert args.frame_id == "000050"
    assert args.candidate_id == "mask_00"
    assert args.mask_path == payload["mask_path"]


def test_lift_mask_args_accepts_sam_candidate_mask_npz_path(
    tmp_path: Path,
) -> None:
    payload = _lift_mask_args_payload(tmp_path)
    mask_path = payload.pop("mask_path")
    payload["mask_npz_path"] = mask_path

    args = LiftMaskArgs.model_validate(payload)

    assert args.mask_path == mask_path


def test_lift_mask_args_ignore_redundant_mask_overlay_path(tmp_path: Path) -> None:
    payload = _lift_mask_args_payload(tmp_path)
    payload["mask_overlay_path"] = str(tmp_path / "mask_00_overlay.jpg")

    args = LiftMaskArgs.model_validate(payload)

    assert args.mask_path == payload["mask_path"]


def test_lift_mask_args_rejects_nonexistent_input_file(tmp_path: Path) -> None:
    payload = _lift_mask_args_payload(tmp_path)
    payload["mask_path"] = tmp_path / "missing-mask.npz"

    with pytest.raises(ValidationError):
        LiftMaskArgs.model_validate(payload)


def test_lift_mask_args_rejects_empty_path_string(tmp_path: Path) -> None:
    payload = _lift_mask_args_payload(tmp_path)
    payload["mask_path"] = ""

    with pytest.raises(ValidationError):
        LiftMaskArgs.model_validate(payload)


def test_lift_mask_args_rejects_extra_fields(tmp_path: Path) -> None:
    payload = _lift_mask_args_payload(tmp_path)
    payload["unexpected"] = "value"

    with pytest.raises(ValidationError):
        LiftMaskArgs.model_validate(payload)


@pytest.mark.parametrize("field_name", ["frame_id", "candidate_id"])
def test_lift_mask_args_rejects_blank_ids(tmp_path: Path, field_name: str) -> None:
    payload = _lift_mask_args_payload(tmp_path)
    payload[field_name] = "   "

    with pytest.raises(ValidationError):
        LiftMaskArgs.model_validate(payload)


@pytest.mark.parametrize("field_name", ["frame_id", "candidate_id"])
def test_lift_mask_args_rejects_unsafe_ids(tmp_path: Path, field_name: str) -> None:
    payload = _lift_mask_args_payload(tmp_path)
    payload[field_name] = "../escape"

    with pytest.raises(ValidationError):
        LiftMaskArgs.model_validate(payload)


def test_lift_mask_result_payload_is_json_serializable() -> None:
    result = LiftMaskResult(
        frame_id="000050",
        candidate_id="mask_00",
        lifted_point_count=12,
        mask_npz_path=Path("/tmp/mask_00.npz"),
        mask_ply_path=Path("/tmp/mask_00.ply"),
        overlay_path=Path("/tmp/mask_00_overlay.png"),
    )

    payload = result.to_payload()

    assert isinstance(payload["mask_npz_path"], str)
    assert isinstance(payload["mask_ply_path"], str)
    assert isinstance(payload["overlay_path"], str)
    assert json.loads(json.dumps(payload)) == payload


def test_lift_mask_result_rejects_zero_points() -> None:
    with pytest.raises(SceneFunc3dDataError, match="positive"):
        LiftMaskResult(
            frame_id="000050",
            candidate_id="mask_00",
            lifted_point_count=0,
            mask_npz_path=Path("/tmp/mask_00.npz"),
            mask_ply_path=Path("/tmp/mask_00.ply"),
            overlay_path=Path("/tmp/mask_00_overlay.png"),
        )


def _lift_mask_args_payload(tmp_path: Path) -> dict[str, object]:
    mask_path = tmp_path / "mask_00.npz"
    depth_path = tmp_path / "000050.depth.png"
    intrinsics_path = tmp_path / "intrinsics.txt"
    pose_path = tmp_path / "pose.txt"
    for input_path in (mask_path, depth_path, intrinsics_path, pose_path):
        input_path.write_text("placeholder\n", encoding="utf-8")

    return {
        "frame_id": "000050",
        "candidate_id": "mask_00",
        "mask_path": mask_path,
        "depth_path": depth_path,
        "intrinsics_path": intrinsics_path,
        "pose_path": pose_path,
    }


def _write_scoring_scene(root: Path) -> None:
    scene_dir = root / "421254"
    scene_dir.mkdir(parents=True)
    (scene_dir / "conceptgraph").mkdir()
    (scene_dir / "421254_descriptions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "descriptions": [
                    {
                        "desc_id": "desc-a",
                        "annot_id": ["annot-a", "annot-b"],
                        "description": "Open the lower drawer.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_motions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "motions": [
                    {
                        "motion_id": "motion-a",
                        "annot_id": "annot-a",
                        "motion_type": "trans",
                        "motion_dir": [1.0, 0.0, 0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_annotations.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "annotations": [
                    {
                        "annot_id": "annot-a",
                        "label": "pinch_pull",
                        "indices": [3, 5, 8],
                    },
                    {
                        "annot_id": "annot-b",
                        "label": "pinch_pull",
                        "indices": [8, 13, 21],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_scoring_mask_artifact(artifact_path: Path, *, mask_npz_path: Path) -> None:
    artifact_path.write_text(
        json.dumps(
            {
                "accepted_frame_ids": ["000010"],
                "accepted_fragments": [
                    {
                        "fragment_id": "frag-a",
                        "frame_id": "000010",
                        "point_count": 5,
                        "lift_geometry": _lift_geometry_payload_for_point_count(5),
                        "approval_actions": [
                            action.value for action in FRAGMENT_APPROVAL_ACTIONS
                        ],
                        "review_artifacts": {
                            "molmo_raw_text_path": "/tmp/molmo_raw.txt",
                            "molmo_overlay_path": "/tmp/molmo_overlay.jpg",
                            "sam_contact_sheet_path": "/tmp/sam_contact.jpg",
                            "sam_candidate_overlay_path": "/tmp/sam_candidate.jpg",
                            "lift_overlay_path": "/tmp/lift_overlay.jpg",
                        },
                    }
                ],
                "multi_view_decision": {
                    "seed_fragment_id": "frag-a",
                    "action": "stop",
                    "reason": "first lift covers the target part for this fixture",
                    "suggested_frame_ids": [],
                },
                "mask_npz_path": str(mask_npz_path),
                "mask_ply_path": "/tmp/lifted_points.ply",
            }
        ),
        encoding="utf-8",
    )


def _write_valid_scoring_mask_artifact(
    artifact_path: Path,
    *,
    mask_npz_path: Path,
    mask_ply_path: Path,
    point_indices: tuple[int, ...],
) -> None:
    _write_valid_scoring_mask_artifact_with_recorded_paths(
        artifact_path,
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
        point_indices=point_indices,
        recorded_mask_npz_path=mask_npz_path,
        recorded_mask_ply_path=mask_ply_path,
    )


def _write_valid_scoring_mask_artifact_with_recorded_paths(
    artifact_path: Path,
    *,
    mask_npz_path: Path,
    mask_ply_path: Path,
    point_indices: tuple[int, ...],
    recorded_mask_npz_path: Path,
    recorded_mask_ply_path: Path,
) -> None:
    _write_scoring_mask_members(
        mask_npz_path,
        mask_ply_path=mask_ply_path,
        point_indices=point_indices,
    )
    review_artifacts = _write_existing_review_artifacts(artifact_path.parent / "frag-a")
    artifact_path.write_text(
        json.dumps(
            {
                "accepted_frame_ids": ["000010"],
                "accepted_fragments": [
                    {
                        "fragment_id": "frag-a",
                        "frame_id": "000010",
                        "point_count": len(point_indices),
                        "lift_geometry": _lift_geometry_payload_for_point_count(
                            len(point_indices)
                        ),
                        "approval_actions": [
                            action.value for action in FRAGMENT_APPROVAL_ACTIONS
                        ],
                        "review_artifacts": review_artifacts,
                    }
                ],
                "multi_view_decision": {
                    "seed_fragment_id": "frag-a",
                    "action": "stop",
                    "reason": "first lift covers the target part for this fixture",
                    "suggested_frame_ids": [],
                },
                "mask_npz_path": str(recorded_mask_npz_path),
                "mask_ply_path": str(recorded_mask_ply_path),
            }
        ),
        encoding="utf-8",
    )


def _write_scoring_mask_members(
    mask_npz_path: Path,
    *,
    mask_ply_path: Path,
    point_indices: tuple[int, ...],
) -> None:
    from codex_agent.scenefunc3d.backends.lift_3d import write_lift_npz, write_lift_ply

    points_world = np.array(
        [
            [float(index), float(index + 1), float(index + 2)]
            for index, _point_id in enumerate(point_indices)
        ],
        dtype=np.float64,
    )
    point_indices_array = np.array(point_indices, dtype=np.int64)
    write_lift_npz(mask_npz_path, points_world, point_indices=point_indices_array)
    write_lift_ply(mask_ply_path, points_world)


def _write_scoring_npz(path: Path, *, point_indices: tuple[int, ...]) -> None:
    from codex_agent.scenefunc3d.backends.lift_3d import write_lift_npz

    points_world = np.array(
        [
            [float(index), float(index + 1), float(index + 2)]
            for index, _point_id in enumerate(point_indices)
        ],
        dtype=np.float64,
    )
    point_indices_array = np.array(point_indices, dtype=np.int64)
    write_lift_npz(path, points_world, point_indices=point_indices_array)


def _write_existing_review_artifacts(root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "molmo_raw_text_path": root / "molmo_raw.txt",
        "molmo_overlay_path": root / "molmo_overlay.jpg",
        "sam_contact_sheet_path": root / "sam_contact_sheet.jpg",
        "sam_candidate_overlay_path": root / "sam_candidate_overlay.jpg",
        "lift_overlay_path": root / "lift_overlay.txt",
    }
    for path in paths.values():
        path.write_text("reviewed\n", encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def _write_standard_review_artifacts(
    artifact_root: Path, *, frame_id: str, candidate_id: str
) -> dict[str, str]:
    fragment_id = f"{frame_id}_{candidate_id}"
    paths = {
        "molmo_raw_text_path": artifact_root / "molmo" / f"{frame_id}_raw.txt",
        "molmo_overlay_path": artifact_root / "molmo" / f"{frame_id}_points.jpg",
        "sam_contact_sheet_path": (
            artifact_root / "sam" / frame_id / "contact_sheet.jpg"
        ),
        "sam_candidate_overlay_path": (
            artifact_root / "sam" / frame_id / f"{candidate_id}_overlay.jpg"
        ),
        "lift_overlay_path": (
            artifact_root / "fragments" / fragment_id / "lift_overlay.txt"
        ),
    }
    for field_name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if field_name == "lift_overlay_path":
            path.write_text(
                f"frame_id={frame_id}\ncandidate_id={candidate_id}\n",
                encoding="utf-8",
            )
        else:
            path.write_text("reviewed\n", encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def _lift_geometry_payload_for_point_count(point_count: int) -> dict[str, object]:
    max_index = float(max(point_count - 1, 0))
    return {
        "bbox_min_xyz": [0.0, 1.0, 2.0],
        "bbox_max_xyz": [max_index, max_index + 1.0, max_index + 2.0],
        "bbox_extent_xyz": [max_index, max_index, max_index],
        "max_extent_meters": max_index,
    }


def _write_scoring_result_file(
    result_path: Path,
    *,
    mask_point_indices: tuple[int, ...],
) -> Path:
    result_path.parent.mkdir(parents=True)
    fused_dir = result_path.parent / "fused"
    mask_npz_path = fused_dir / "mask_data.npz"
    mask_ply_path = fused_dir / "lifted_points.ply"
    artifact_path = fused_dir / "mask_artifact.json"
    _write_valid_scoring_mask_artifact(
        artifact_path,
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
        point_indices=mask_point_indices,
    )
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": "421254::desc-a",
                "outcome": {
                    "mask_artifact_path": str(artifact_path),
                    "mask_npz_path": str(mask_npz_path),
                    "mask_ply_path": str(mask_ply_path),
                    "selected_frame_ids": ["000010"],
                    "accepted_fragment_ids": ["frag-a"],
                    "confidence": 0.9,
                    "uncertainties": [],
                },
                "turn": {},
            }
        ),
        encoding="utf-8",
    )
    return result_path


def _json_object(raw_json: str) -> Mapping[str, object]:
    payload: object = json.loads(raw_json)
    assert isinstance(payload, dict)
    return cast(Mapping[str, object], payload)


def _json_object_member(
    payload: Mapping[str, object], member_name: str
) -> Mapping[str, object]:
    member = payload[member_name]
    assert isinstance(member, dict)
    return cast(Mapping[str, object], member)
