"""Depth comparison measures raw meters and keeps train/GT boundaries explicit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codex_agent.alfred.depth_calibration import (
    CalibrationFrame,
    CalibrationManifest,
    aggregate_depth_metrics,
    depth_metrics,
    plane_metrics,
    select_depth_backend,
    temporal_depth_consistency,
)


def _frame(frame_id: str = "0") -> CalibrationFrame:
    return CalibrationFrame(
        frame_id=frame_id,
        scene_id="FloorPlan1",
        rgb_path=Path("rgb.png"),
        gt_depth_path=Path("depth.npy"),
        rgb_sha256="a" * 64,
        gt_depth_sha256="b" * 64,
        intrinsics=((100, 0, 8), (0, 100, 8), (0, 0, 1)),
        pitch_degrees=0,
        camera_to_world=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    )


def test_depth_metrics_penalize_scale_error_without_gt_alignment() -> None:
    truth = np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)
    result = depth_metrics(truth * 2, truth)
    assert result.abs_rel == 1
    assert result.median_scale_ratio == 2
    assert result.rmse_m == pytest.approx(np.sqrt(2))
    assert result.near_abs_rel == 1
    assert result.edge_gradient_mae_m == 1
    assert result.edge_pairs == 2


def test_invalid_predictions_reduce_coverage_and_saturated_gt_is_separate() -> None:
    truth = np.array([[1, 2, 5], [0, np.nan, 5]], dtype=np.float32)
    prediction = np.array([[1, np.nan, 20], [4, 4, 20]], dtype=np.float32)
    result = depth_metrics(prediction, truth)
    assert result.valid_gt_pixels == 2
    assert result.valid_prediction_pixels == 1
    assert result.valid_coverage == 0.5
    assert result.saturated_gt_pixels == 2
    assert result.saturation_fraction == pytest.approx(1 / 3)
    assert result.abs_rel == 0


def test_selection_has_no_default_preference_for_da3() -> None:
    truth = np.ones((4, 4), dtype=np.float32)
    summaries = {
        "da3metric-large": aggregate_depth_metrics([depth_metrics(truth * 1.2, truth)]),
        "unidepth-v2-vitl14": aggregate_depth_metrics([depth_metrics(truth, truth)]),
    }
    assert select_depth_backend(summaries)["selected_backend"] == "unidepth-v2-vitl14"
    summaries["unidepth-v2-vitl14"]["minimum_frame_coverage"] = 0.9
    assert select_depth_backend(summaries)["selected_backend"] == "da3metric-large"
    summaries["unidepth-v2-vitl14"] = summaries["da3metric-large"].copy()
    assert select_depth_backend(summaries)["status"] == "tie_requires_review"


def test_manifest_rejects_validation_split_and_duplicate_frames() -> None:
    payload = {
        "schema_version": 1,
        "split": "valid_unseen",
        "capture_protocol": "frozen",
        "source_sha256": "a" * 64,
        "frames": [_frame().model_dump(mode="json")],
    }
    with pytest.raises(ValueError):
        CalibrationManifest.model_validate(payload)
    payload["split"] = "train"
    payload["frames"] = [_frame().model_dump(mode="json")] * 2
    with pytest.raises(ValueError, match="unique"):
        CalibrationManifest.model_validate(payload)


def test_plane_measurement_does_not_remove_metric_offset() -> None:
    truth = np.ones((16, 16), dtype=np.float32)
    assert plane_metrics(truth, truth, np.eye(3), None)["status"] == "unavailable"
    result = plane_metrics(truth * 2, truth, np.eye(3), np.ones_like(truth, dtype=bool))
    assert result["normal_error_degrees"] == pytest.approx(0)
    assert result["offset_error_m"] == pytest.approx(1)


def test_multiframe_geometry_detects_disagreement_at_same_pose() -> None:
    truth = np.ones((16, 16), dtype=np.float32)
    source, target = _frame("0"), _frame("1")
    result = temporal_depth_consistency(truth, truth * 2, truth, truth, source, target)
    assert result["status"] == "computed"
    assert result["mae_m"] == pytest.approx(1)


def test_frozen_manifest_hashes_are_checked_before_inference(tmp_path: Path) -> None:
    import json

    from codex_agent.alfred.depth_calibration import load_calibration_manifest
    from codex_agent.alfred.provenance import file_sha256

    rgb, depth = tmp_path / "rgb.png", tmp_path / "depth.npy"
    rgb.write_bytes(b"fixture image bytes")
    np.save(depth, np.ones((4, 4), dtype=np.float32))
    row = _frame().model_dump(mode="json")
    row.update(
        rgb_path="rgb.png",
        gt_depth_path="depth.npy",
        rgb_sha256=file_sha256(rgb),
        gt_depth_sha256=file_sha256(depth),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "train",
                "capture_protocol": "frozen",
                "source_sha256": "a" * 64,
                "frames": [row],
            }
        )
    )
    assert load_calibration_manifest(manifest).frames[0].rgb_path == rgb
    rgb.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_calibration_manifest(manifest)


def test_comparison_model_receives_only_rgb_and_camera(tmp_path: Path) -> None:
    from PIL import Image

    from codex_agent.alfred.depth_calibration_runner import evaluate_depth_model
    from codex_agent.alfred.provenance import file_sha256

    rgb, depth = tmp_path / "rgb.png", tmp_path / "depth.npy"
    Image.new("RGB", (16, 16), (23, 45, 67)).save(rgb)
    np.save(depth, np.ones((16, 16), dtype=np.float32))
    frame = _frame().model_copy(
        update={
            "rgb_path": rgb,
            "gt_depth_path": depth,
            "rgb_sha256": file_sha256(rgb),
            "gt_depth_sha256": file_sha256(depth),
        }
    )
    manifest = CalibrationManifest(
        schema_version=1,
        split="train",
        capture_protocol="frozen",
        source_sha256="a" * 64,
        frames=(frame,),
    )
    seen: list[tuple[int, ...]] = []

    class ExternalDepthModel:
        identifier = "test-external-boundary"

        def predict(self, rgb: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
            seen.append(rgb.shape)
            assert rgb.dtype == np.uint8
            assert intrinsics.shape == (3, 3)
            return np.full(rgb.shape[:2], 2, dtype=np.float32)

    result = evaluate_depth_model(
        ExternalDepthModel(),
        manifest,
        output_dir=tmp_path / "prediction",
        gpu_lock_path=tmp_path / "gpu.lock",
        lock_timeout_seconds=1,
    )
    assert seen == [(16, 16, 3)]
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert summary["abs_rel"] == 1
    assert summary["median_scale_ratio"] == 2
    assert (tmp_path / "prediction/frames.jsonl").is_file()


def test_calibration_uses_explicit_physical_gpu_index_with_logical_cuda_zero(
    tmp_path: Path,
) -> None:
    from dataclasses import replace
    from typing import cast

    from codex_agent.alfred.depth_calibration_runner import CalibrationRunConfig

    config = CalibrationRunConfig(
        manifest_path=tmp_path / "manifest.json",
        da3_model_dir=tmp_path / "da3",
        unidepth_model_dir=tmp_path / "unidepth",
        output_dir=tmp_path / "output",
        gpu_lock_path=tmp_path / "gpu4.lock",
        project_root=tmp_path,
        physical_gpu_index=4,
    )
    assert config.device == "cuda:0"
    assert config.physical_gpu_index == 4
    assert replace(config, physical_gpu_index=0).physical_gpu_index == 0
    for invalid in (1, -1, True, 4.0):
        with pytest.raises(ValueError, match="physical_gpu_index"):
            replace(config, physical_gpu_index=cast(int, invalid))
