"""Perception RPC and artifact boundaries without optional neural dependencies."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

from codex_agent.alfred import perception as perception_module
from codex_agent.alfred import provenance
from codex_agent.alfred.perception import (
    PERCEPTION_PROFILE,
    PerceptionDetection,
    PerceptionDiagnostics,
    PerceptionObservation,
    ScenePerception,
    ScenePerceptionConfig,
    validate_prediction_artifacts,
)
from codex_agent.alfred.perception_worker import PredictionRequest


def _configuration(root: Path) -> ScenePerceptionConfig:
    for name in ("dino.py", "dino.pth", "sam.pth", "clip.safetensors"):
        (root / name).write_text("local test fixture")
    for folder, names in (
        (
            "bert",
            ("config.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"),
        ),
        ("depth", ("config.json", "model.safetensors")),
    ):
        directory = root / folder
        directory.mkdir()
        for name in names:
            (directory / name).write_text(json.dumps({"model_name": "da3metric-large"}))
    return ScenePerceptionConfig(
        python_executable=Path(sys.executable),
        dino_config=root / "dino.py",
        dino_checkpoint=root / "dino.pth",
        dino_text_encoder_dir=root / "bert",
        sam_checkpoint=root / "sam.pth",
        clip_checkpoint=root / "clip.safetensors",
        depth_model_dir=root / "depth",
        gpu_lock_path=root / "gpu.lock",
    )


def _runtime_provenance(*, visible_devices: str) -> dict[str, object]:
    return provenance.RuntimeProvenance(
        executable="/separate/worker/bin/python",
        prefix="/separate/worker",
        package_root=str(Path(provenance.__file__).parent),
        package_source_sha256=provenance.alfred_source_sha256(),
        visible_devices=visible_devices,
        packages={
            name: provenance.PackageProvenance(
                version="worker-version", distribution_record_sha256="a" * 64
            )
            for name in provenance.PERCEPTION_PACKAGES
        },
    ).model_dump(mode="json")


def _observation(
    root: Path, *, frame_id: str = "frame", with_detection: bool = True
) -> PerceptionObservation:
    depth_path, features_path = (
        root / f"{frame_id}_depth.npy",
        root / f"{frame_id}_features.npz",
    )
    np.save(depth_path, np.ones((12, 16), dtype=np.float32))
    detections: tuple[PerceptionDetection, ...] = ()
    if with_detection:
        mask_path = root / f"{frame_id}_mask.npy"
        mask = np.zeros((12, 16), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        np.save(mask_path, mask)
        detections = (
            PerceptionDetection(
                category="Faucet",
                mask_path=mask_path,
                bbox_xyxy=(3, 2, 7, 5),
                confidence=0.9,
            ),
        )
    feature = np.array([0.6, 0.8], dtype=np.float32)
    objects = np.tile(feature, (len(detections), 1))
    np.savez(
        features_path,
        image_features=objects,
        text_features=objects,
        frame_feature=feature,
    )
    return PerceptionObservation(
        frame_id=frame_id,
        depth_path=depth_path,
        depth_model="da3metric-large",
        image_width=16,
        image_height=12,
        features_path=features_path,
        detections=detections,
        diagnostics=PerceptionDiagnostics(
            vocabulary_categories=114,
            caption_batches=5,
            candidate_boxes=len(detections),
            nms_suppressed=0,
            capped_detections=0,
            empty_masks=0,
        ),
    )


def test_config_requires_explicit_local_assets_and_rejects_gpu_one(
    tmp_path: Path,
) -> None:
    config = _configuration(tmp_path)
    payload = config.to_payload()
    payload["device"] = "cuda:1"
    with pytest.raises(ValueError, match="device"):
        ScenePerceptionConfig(**payload)  # type: ignore[arg-type]
    config.sam_checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="sam_checkpoint"):
        ScenePerceptionConfig(**config.to_payload())  # type: ignore[arg-type]


@pytest.mark.parametrize("index", [1, -1, True, "4"])
def test_physical_gpu_index_is_strict_and_forbids_gpu_one(
    tmp_path: Path, index: object
) -> None:
    config = _configuration(tmp_path)
    payload = {**config.to_payload(), "physical_gpu_index": index}
    with pytest.raises(ValueError, match="physical_gpu_index"):
        ScenePerceptionConfig(**payload)  # type: ignore[arg-type]


def test_relative_depth_checkpoint_cannot_substitute_metric_model(
    tmp_path: Path,
) -> None:
    config = _configuration(tmp_path)
    (config.depth_model_dir / "config.json").write_text(
        '{"model_name":"da3mono-large"}'
    )
    with pytest.raises(ValueError, match="model_name"):
        ScenePerceptionConfig(**config.to_payload())  # type: ignore[arg-type]


def test_validates_rgb_sized_depth_masks_and_normalized_features(
    tmp_path: Path,
) -> None:
    observation = _observation(tmp_path)
    validate_prediction_artifacts(observation, output_dir=tmp_path)
    np.save(observation.depth_path, np.full((12, 16), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="depth"):
        validate_prediction_artifacts(observation, output_dir=tmp_path)


def test_zero_detections_retain_frame_feature_and_empty_aligned_arrays(
    tmp_path: Path,
) -> None:
    observation = _observation(tmp_path, with_detection=False)
    validate_prediction_artifacts(observation, output_dir=tmp_path)
    with np.load(observation.features_path) as archive:
        assert (
            archive["image_features"].shape == archive["text_features"].shape == (0, 2)
        )
        assert archive["frame_feature"].shape == (2,)


def test_preserves_float_detector_box_independently_of_mask_bounds(
    tmp_path: Path,
) -> None:
    observation = _observation(tmp_path)
    payload = observation.model_dump()
    payload["detections"][0]["bbox_xyxy"] = (-1.5, 0.25, 16.5, 10.75)
    changed = PerceptionObservation.model_validate(payload)
    validate_prediction_artifacts(changed, output_dir=tmp_path)
    assert changed.detections[0].bbox_xyxy == (-1.5, 0.25, 16.5, 10.75)
    payload["detections"][0]["bbox_xyxy"] = (20, 0, 25, 10)
    with pytest.raises(ValueError, match="intersect"):
        PerceptionObservation.model_validate(payload)


def test_rejects_misaligned_or_unnormalized_features(tmp_path: Path) -> None:
    observation = _observation(tmp_path)
    np.savez(
        observation.features_path,
        image_features=np.ones((2, 2), dtype=np.float32),
        text_features=np.ones((2, 2), dtype=np.float32),
        frame_feature=np.array([1, 0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="dimensions/order"):
        validate_prediction_artifacts(observation, output_dir=tmp_path)
    np.savez(
        observation.features_path,
        image_features=np.ones((1, 2), dtype=np.float32),
        text_features=np.ones((1, 2), dtype=np.float32),
        frame_feature=np.array([1, 0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="normalized"):
        validate_prediction_artifacts(observation, output_dir=tmp_path)


def test_rejects_masks_outside_owned_directory(tmp_path: Path) -> None:
    observation = _observation(tmp_path)
    changed = observation.model_copy(
        update={"features_path": tmp_path.parent / "foreign.npz"}
    )
    with pytest.raises(ValueError, match="output_dir"):
        validate_prediction_artifacts(changed, output_dir=tmp_path)


def test_rejects_unaccounted_detector_truncation(tmp_path: Path) -> None:
    observation = _observation(tmp_path)
    payload = observation.model_dump()
    payload["diagnostics"]["candidate_boxes"] = 50
    with pytest.raises(ValueError, match="account"):
        PerceptionObservation.model_validate(payload)


def test_worker_request_forbids_metadata_and_horizon_routing() -> None:
    payload = {
        "command": "predict",
        "rgb_path": "image.png",
        "frame_id": "0",
        "intrinsics": np.eye(3).tolist(),
        "output_dir": "/tmp/predictions",
    }
    for field in ("metadata", "task_type", "horizon_degrees", "depth_frame"):
        with pytest.raises(ValueError, match="Extra inputs"):
            PredictionRequest.model_validate({**payload, field: 1})


def test_persistent_worker_pins_gpu_and_passes_only_rgb_and_intrinsics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = replace(_configuration(tmp_path), physical_gpu_index=4)
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (16, 12)).save(image_path)
    captured: dict[str, object] = {}
    commands: list[str] = []
    requests: list[dict[str, object]] = []
    selected_gpus: list[int] = []
    gpu_uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"

    def select_gpu(index: int) -> str:
        selected_gpus.append(index)
        return gpu_uuid

    monkeypatch.setattr(perception_module, "physical_gpu_uuid", select_gpu)

    class FakeWorker:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def request(self, payload: dict[str, object]) -> dict[str, object]:
            commands.append(str(payload["command"]))
            if payload["command"] == "initialize":
                return {
                    "backend": PERCEPTION_PROFILE,
                    "runtime_provenance": _runtime_provenance(visible_devices=gpu_uuid),
                }
            requests.append(payload)
            return _observation(
                tmp_path, frame_id=str(payload["frame_id"]), with_detection=False
            ).model_dump(mode="json")

        def close(self) -> None:
            commands.append("close")

    monkeypatch.setattr(perception_module, "JsonWorker", FakeWorker)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with ScenePerception(configuration, log_path=tmp_path / "worker.log") as predictor:
        assert predictor.runtime_provenance.prefix == "/separate/worker"
        assert (
            predictor.runtime_provenance.packages["torch"].version == "worker-version"
        )
        for frame in ("first", "second"):
            predictor.predict(
                image_path, frame_id=frame, intrinsics=np.eye(3), output_dir=tmp_path
            )
        with pytest.raises(ValueError, match="intrinsics"):
            predictor.predict(
                image_path,
                frame_id="invalid",
                intrinsics=np.zeros((3, 3)),
                output_dir=tmp_path,
            )
    assert commands == ["initialize", "predict", "predict", "close"]
    assert all(
        set(request) == {"command", "rgb_path", "frame_id", "intrinsics", "output_dir"}
        for request in requests
    )
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["CUDA_VISIBLE_DEVICES"] == gpu_uuid
    assert selected_gpus == [4]
    assert environment["HF_HUB_OFFLINE"] == environment["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize("problem", ["missing", "source", "accelerator", "packages"])
def test_initialization_rejects_unknown_or_mismatched_worker_provenance_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    config = replace(_configuration(tmp_path), device="cpu")
    commands: list[str] = []
    reported = _runtime_provenance(visible_devices="")
    if problem == "source":
        reported["package_source_sha256"] = "0" * 64
    elif problem == "accelerator":
        reported["visible_devices"] = "unexpected-device"
    elif problem == "packages":
        cast(dict[str, object], reported["packages"]).pop("torch")

    class FakeWorker:
        def __init__(self, **kwargs: object) -> None:
            pass

        def request(self, payload: dict[str, object]) -> dict[str, object]:
            commands.append(str(payload["command"]))
            return {
                "backend": PERCEPTION_PROFILE,
                **({"runtime_provenance": reported} if problem != "missing" else {}),
            }

        def close(self) -> None:
            commands.append("close")

    def forbidden_gpu_query(index: int) -> str:
        raise AssertionError("CPU perception must not query or select a GPU")

    monkeypatch.setattr(perception_module, "JsonWorker", FakeWorker)
    monkeypatch.setattr(perception_module, "physical_gpu_uuid", forbidden_gpu_query)
    with pytest.raises(ValueError):
        ScenePerception(config, log_path=tmp_path / "worker.log")
    assert commands == ["initialize", "close"]


def test_initialization_cancellation_closes_owned_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_configuration(tmp_path), device="cpu")
    closed: list[bool] = []

    class InterruptedWorker:
        def __init__(self, **kwargs: object) -> None:
            pass

        def request(self, payload: dict[str, object]) -> dict[str, object]:
            raise KeyboardInterrupt("cancelled initialization")

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(perception_module, "JsonWorker", InterruptedWorker)
    with pytest.raises(KeyboardInterrupt, match="cancelled initialization"):
        ScenePerception(config, log_path=tmp_path / "worker.log")
    assert closed == [True]
