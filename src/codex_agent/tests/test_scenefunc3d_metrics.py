"""Tests for SceneFunc3D mask metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from codex_agent.scenefunc3d.evaluation.metrics import (
    MaskMetrics,
    compute_mask_metrics,
)
from codex_agent.scenefunc3d.evaluation.scorer import score_point_ids
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
