"""Perception RPC contracts without loading optional external neural models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codex_agent.alfred import perception as perception_module
from codex_agent.alfred.perception import (
    FILM_CHECKPOINTS,
    FilmPerception,
    FilmPerceptionConfig,
    PerceptionDetection,
    PerceptionObservation,
    validate_prediction_artifacts,
)
from codex_agent.alfred.perception_worker import (
    choose_depth_view,
    normalize_depth_state,
    prepare_model_images,
)


def test_depth_gets_bgr_while_detector_and_saved_source_remain_rgb() -> None:
    rgb = np.array([[[255, 31, 7], [12, 80, 199]]], dtype=np.uint8)
    original = rgb.copy()
    detector_rgb, depth_bgr = prepare_model_images(rgb)
    np.testing.assert_array_equal(detector_rgb, original)
    np.testing.assert_array_equal(depth_bgr, [[[7, 31, 255], [199, 80, 12]]])
    np.testing.assert_array_equal(rgb, original)
    assert detector_rgb.flags.c_contiguous and depth_bgr.flags.c_contiguous
    assert all(stride >= 0 for stride in depth_bgr.strides)


def test_depth_checkpoint_routing_is_explicit() -> None:
    assert choose_depth_view(0) == "level"
    assert choose_depth_view(15) == "level"
    assert choose_depth_view(30) == "downward"
    assert choose_depth_view(45) == "downward"
    assert choose_depth_view(60.00001) == "downward"
    assert choose_depth_view(-30.00001) == "level"
    with pytest.raises(ValueError, match="horizon"):
        choose_depth_view(float("nan"))
    with pytest.raises(ValueError, match="horizon"):
        choose_depth_view(60.01)


def test_depth_state_removes_only_valid_parallel_prefixes() -> None:
    assert normalize_depth_state({"module.conv.weight": 1}) == {"conv.weight": 1}
    with pytest.raises(ValueError, match="module"):
        normalize_depth_state({"conv.weight": 1})


def test_config_rejects_gpu_one_and_missing_checkpoints(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="device"):
        FilmPerceptionConfig(
            python_executable=Path(__file__),
            film_root=tmp_path,
            weights_root=tmp_path,
            device="cuda:1",  # type: ignore[arg-type]  # Exercise invalid runtime input.
        )
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        FilmPerceptionConfig(
            python_executable=Path(__file__),
            film_root=tmp_path,
            weights_root=tmp_path,
        )


def test_validates_saved_depth_masks_and_paths(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(depth_path, np.ones((300, 300), dtype=np.float32))
    np.save(mask_path, np.ones((300, 300), dtype=np.uint8))
    observation = PerceptionObservation(
        frame_id="0001",
        depth_path=depth_path,
        depth_model="level",
        detections=(
            PerceptionDetection(
                category="Apple",
                mask_path=mask_path,
                bbox_xyxy=(0, 0, 300, 300),
                confidence=0.9,
            ),
        ),
    )
    validate_prediction_artifacts(observation, output_dir=tmp_path)
    np.save(depth_path, np.full((300, 300), float("nan"), dtype=np.float32))
    with pytest.raises(ValueError, match="depth"):
        validate_prediction_artifacts(observation, output_dir=tmp_path)


def test_rejects_mask_outside_artifact_directory(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, np.ones((300, 300), dtype=np.float32))
    observation = PerceptionObservation(
        frame_id="0001",
        depth_path=depth_path,
        depth_model="level",
        detections=(
            PerceptionDetection(
                category="Apple",
                mask_path=tmp_path.parent / "outside.npy",
                bbox_xyxy=(1, 1, 3, 3),
                confidence=0.9,
            ),
        ),
    )
    with pytest.raises(ValueError, match="output_dir"):
        validate_prediction_artifacts(observation, output_dir=tmp_path)


def test_worker_is_persistent_and_pins_physical_gpu_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for relative in FILM_CHECKPOINTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    captured: dict[str, object] = {}
    commands: list[str] = []
    horizons: list[object] = []

    class FakeWorker:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def request(self, payload: dict[str, object]) -> dict[str, object]:
            commands.append(str(payload["command"]))
            if payload["command"] == "initialize":
                return {"backend": "film_predicted_rgb_v1"}
            horizons.append(payload["horizon_degrees"])
            depth_path = tmp_path / (str(payload["frame_id"]) + "_depth.npy")
            np.save(depth_path, np.ones((300, 300), dtype=np.float32))
            return {
                "frame_id": payload["frame_id"],
                "depth_path": str(depth_path),
                "depth_model": "level",
                "detections": [],
            }

        def close(self) -> None:
            commands.append("close")

    monkeypatch.setattr(perception_module, "JsonWorker", FakeWorker)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    config = FilmPerceptionConfig(
        python_executable=Path(__file__), film_root=tmp_path, weights_root=tmp_path
    )
    with FilmPerception(config, log_path=tmp_path / "worker.log") as predictor:
        for frame_id in ("first", "second"):
            predictor.predict(
                Path(__file__),
                frame_id=frame_id,
                horizon_degrees=60.00001,
                output_dir=tmp_path,
            )
        with pytest.raises(ValueError, match="horizon"):
            predictor.predict(
                Path(__file__),
                frame_id="invalid",
                horizon_degrees=60.01,
                output_dir=tmp_path,
            )
    assert commands == ["initialize", "predict", "predict", "close"]
    assert horizons == [60.0, 60.0]
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
