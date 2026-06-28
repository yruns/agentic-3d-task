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
    mask_npz_path = tmp_path / "mask_data.npz"
    np.savez_compressed(mask_npz_path, point_indices=np.array([3, 5, 8, 13, 21]))
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    artifact_path = result_dir / "mask_artifact.json"
    _write_scoring_mask_artifact(artifact_path, mask_npz_path=mask_npz_path)
    result_path = result_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": "421254::desc-a",
                "outcome": {
                    "mask_artifact_path": "mask_artifact.json",
                    "mask_npz_path": str(tmp_path / "wrong_prediction.npz"),
                    "mask_ply_path": "/tmp/lifted_points.ply",
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


def test_evaluation_cli_scores_runner_result_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_scoring_scene(tmp_path)
    mask_npz_path = tmp_path / "mask_data.npz"
    np.savez_compressed(mask_npz_path, point_indices=np.array([3, 5, 8, 13, 21]))
    artifact_path = tmp_path / "mask_artifact.json"
    _write_scoring_mask_artifact(artifact_path, mask_npz_path=mask_npz_path)
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": "421254::desc-a",
                "outcome": {
                    "mask_artifact_path": str(artifact_path),
                    "mask_npz_path": str(tmp_path / "unused_prediction.npz"),
                    "mask_ply_path": "/tmp/lifted_points.ply",
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


def test_lift_mask_args_accepts_existing_input_files(tmp_path: Path) -> None:
    payload = _lift_mask_args_payload(tmp_path)

    args = LiftMaskArgs.model_validate(payload)

    assert args.frame_id == "000050"
    assert args.candidate_id == "mask_00"
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
                "mask_npz_path": str(mask_npz_path),
                "mask_ply_path": "/tmp/lifted_points.ply",
            }
        ),
        encoding="utf-8",
    )


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
