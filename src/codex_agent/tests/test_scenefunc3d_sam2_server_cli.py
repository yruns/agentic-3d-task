"""Tests for the SceneFunc3D SAM2 sidecar server CLI and handlers."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Literal, cast

import numpy as np
import pytest
from numpy.typing import DTypeLike, NDArray

from codex_agent.scenefunc3d.servers.http_json import JsonRoute, make_json_handler
from codex_agent.scenefunc3d.servers.schemas import (
    SamMaskCandidateResponse,
    SamMaskRequest,
)


class _FakeSamRunner:
    model_name: str = "fake-sam2"

    def masks(self, request: SamMaskRequest) -> tuple[SamMaskCandidateResponse, ...]:
        image_path = request.require_image_path()
        if not image_path.is_file():
            raise AssertionError(f"expected materialized image file: {image_path}")
        staging_dir = request.require_staging_dir()
        staging_dir.mkdir(parents=True, exist_ok=True)
        mask_path = staging_dir / "mask_00.npz"
        mask = np.array([[True, False, True], [False, False, True]], dtype=np.bool_)
        np.savez_compressed(mask_path, mask=mask)
        return (
            SamMaskCandidateResponse(
                candidate_id="mask_00",
                score=0.91,
                mask_npz_path=mask_path,
                pixel_count=3,
                coverage_percent=50.0,
            ),
        )


class _OutOfMemorySamRunner:
    model_name: str = "fake-sam2"

    def masks(self, request: SamMaskRequest) -> tuple[SamMaskCandidateResponse, ...]:
        raise RuntimeError("CUDA out of memory while allocating SAM tensors")


class _EmptyMaskSamRunner:
    model_name: str = "fake-sam2"

    def masks(self, request: SamMaskRequest) -> tuple[SamMaskCandidateResponse, ...]:
        staging_dir = request.require_staging_dir()
        staging_dir.mkdir(parents=True, exist_ok=True)
        mask_path = staging_dir / "mask_00.npz"
        np.savez_compressed(
            mask_path,
            mask=np.zeros((0, 3), dtype=np.bool_),
        )
        return (
            SamMaskCandidateResponse(
                candidate_id="mask_00",
                score=0.91,
                mask_npz_path=mask_path,
                pixel_count=0,
                coverage_percent=0.0,
            ),
        )


class _RecordingSamRunner(_FakeSamRunner):
    seen_image_path: Path | None = None
    seen_staging_dir: Path | None = None
    image_existed_during_call: bool = False

    def masks(self, request: SamMaskRequest) -> tuple[SamMaskCandidateResponse, ...]:
        image_path = request.require_image_path()
        self.seen_image_path = image_path
        self.seen_staging_dir = request.require_staging_dir()
        self.image_existed_during_call = image_path.is_file()
        return super().masks(request)


class _RootCheckingSamRunner(_FakeSamRunner):
    def __init__(self, staging_root: Path) -> None:
        self._staging_root = staging_root.resolve()
        self.seen_staging_dir: Path | None = None

    def masks(self, request: SamMaskRequest) -> tuple[SamMaskCandidateResponse, ...]:
        staging_dir = request.require_staging_dir().resolve()
        staging_dir.relative_to(self._staging_root)
        self.seen_staging_dir = staging_dir
        return super().masks(request)


def test_build_arg_parser_accepts_runtime_options() -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--backend",
            "official",
            "--host",
            "127.0.0.1",
            "--port",
            "8712",
            "--model-name",
            "SAM2.1-Hiera-L",
            "--checkpoint-path",
            "/models/sam2.pt",
            "--config-path",
            "/models/sam2.yaml",
            "--staging-root",
            "/runs/scenefunc3d",
            "--device",
            "cuda:0",
        ]
    )

    assert args.backend == "official"
    assert args.host == "127.0.0.1"
    assert args.port == 8712
    assert args.model_name == "SAM2.1-Hiera-L"
    assert args.checkpoint_path == Path("/models/sam2.pt")
    assert args.config_path == Path("/models/sam2.yaml")
    assert args.staging_root == Path("/runs/scenefunc3d")
    assert args.device == "cuda:0"


def test_write_candidate_masks_includes_inline_npz_payload(tmp_path: Path) -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import _write_candidate_masks

    mask_batch = np.array(
        [[[True, False, True], [False, False, True]]],
        dtype=np.bool_,
    )
    candidates = _write_candidate_masks(
        mask_batch,
        np.array([0.91], dtype=np.float64),
        staging_dir=tmp_path,
    )

    candidate = candidates[0]
    assert candidate.mask_npz_path is not None
    mask_npz_bytes = candidate.mask_npz_path.read_bytes()
    assert candidate.mask_npz_base64 == base64.b64encode(mask_npz_bytes).decode("ascii")
    assert candidate.mask_npz_sha256 == hashlib.sha256(mask_npz_bytes).hexdigest()


def test_build_arg_parser_accepts_transformers_model_path() -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--backend",
            "transformers",
            "--model-path",
            "/hf/models--facebook--sam2.1-hiera-large/snapshots/665f8",
            "--staging-root",
            "/runs/scenefunc3d",
            "--device",
            "cuda:0",
        ]
    )

    assert args.backend == "transformers"
    assert args.model_path == Path(
        "/hf/models--facebook--sam2.1-hiera-large/snapshots/665f8"
    )
    assert args.model_id == ""
    assert args.staging_root == Path("/runs/scenefunc3d")
    assert args.device == "cuda:0"


def test_build_arg_parser_requires_staging_root() -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            [
                "--checkpoint-path",
                "/models/sam2.pt",
                "--config-path",
                "/models/sam2.yaml",
            ]
        )


def test_build_arg_parser_requires_official_checkpoint_and_config() -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            [
                "--backend",
                "official",
                "--staging-root",
                "/runs/scenefunc3d",
            ]
        )


def test_build_arg_parser_requires_transformers_model_reference() -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            [
                "--backend",
                "transformers",
                "--staging-root",
                "/runs/scenefunc3d",
            ]
        )


def test_build_arg_parser_rejects_blank_transformers_model_id() -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            [
                "--backend",
                "transformers",
                "--model-id",
                "   ",
                "--staging-root",
                "/runs/scenefunc3d",
            ]
        )


def test_module_import_does_not_import_heavy_dependencies() -> None:
    module_name = "codex_agent.scenefunc3d.servers.sam2_mask_server"
    heavy_modules = (
        "PIL",
        "PIL.Image",
        "sam2",
        "sam2.build_sam",
        "sam2.sam2_image_predictor",
        "torch",
        "transformers",
    )
    preserved_modules = {
        dependency_name: sys.modules.pop(dependency_name)
        for dependency_name in heavy_modules
        if dependency_name in sys.modules
    }
    sys.modules.pop(module_name, None)

    try:
        imported_module = importlib.import_module(module_name)
        assert isinstance(imported_module, ModuleType)
        for dependency_name in heavy_modules:
            assert dependency_name not in sys.modules
    finally:
        _restore_modules(preserved_modules)


def test_official_runner_validates_model_files_before_heavy_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_agent.scenefunc3d.servers import sam2_mask_server

    config_path = tmp_path / "sam2.yaml"
    config_path.write_text("model: fake\n", encoding="utf-8")

    def fail_import_module(module_name: str) -> ModuleType:
        raise AssertionError(f"unexpected heavy import: {module_name}")

    monkeypatch.setattr(sam2_mask_server.importlib, "import_module", fail_import_module)

    with pytest.raises(RuntimeError, match="checkpoint_path"):
        sam2_mask_server.OfficialSam2Runner(
            model_name="SAM2.1-Hiera-L",
            checkpoint_path=tmp_path / "missing.pt",
            config_path=config_path,
            staging_root=tmp_path / "out",
            device="cuda:0",
        )


def test_official_runner_rejects_non_cuda_device_before_heavy_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_agent.scenefunc3d.servers import sam2_mask_server

    checkpoint_path = tmp_path / "sam2.pt"
    config_path = tmp_path / "sam2.yaml"
    checkpoint_path.write_bytes(b"checkpoint")
    config_path.write_text("model: fake\n", encoding="utf-8")

    def fail_import_module(module_name: str) -> ModuleType:
        raise AssertionError(f"unexpected heavy import: {module_name}")

    monkeypatch.setattr(sam2_mask_server.importlib, "import_module", fail_import_module)

    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        sam2_mask_server.OfficialSam2Runner(
            model_name="SAM2.1-Hiera-L",
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            staging_root=tmp_path / "out",
            device="cpu",
        )


def test_official_runner_writes_valid_candidate_masks_with_fake_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_agent.scenefunc3d.servers import sam2_mask_server

    checkpoint_path = tmp_path / "sam2.pt"
    config_path = tmp_path / "sam2.yaml"
    image_path = tmp_path / "frame.jpg"
    staging_dir = tmp_path / "out" / "sam" / "000050" / "candidates"
    checkpoint_path.write_bytes(b"checkpoint")
    config_path.write_text("model: fake\n", encoding="utf-8")
    image_path.write_bytes(b"image")
    fake_build_module = _FakeSam2BuildModule()
    fake_predictor_module = _FakePredictorModule()
    fake_torch_module = _FakeTorchModule()
    fake_image_module = _FakeImageModule()

    def fake_import_module(module_name: str) -> ModuleType:
        modules_by_name: dict[str, ModuleType] = {
            "PIL.Image": fake_image_module,
            "sam2.build_sam": fake_build_module,
            "sam2.sam2_image_predictor": fake_predictor_module,
            "torch": fake_torch_module,
        }
        return modules_by_name[module_name]

    monkeypatch.setattr(sam2_mask_server.importlib, "import_module", fake_import_module)

    runner = sam2_mask_server.OfficialSam2Runner(
        model_name="SAM2.1-Hiera-L",
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        staging_root=tmp_path / "out",
        device="cuda:0",
    )
    request = SamMaskRequest.model_validate(
        {
            "request_id": "req-1",
            "image_path": image_path,
            "points": [
                {
                    "x_px": 1.0,
                    "y_px": 0.5,
                    "label": "drawer",
                    "source": '<point x="1" y="0.5">drawer</point>',
                }
            ],
            "staging_dir": staging_dir,
        }
    )
    candidates = runner.masks(request)

    assert fake_build_module.config_path == str(config_path.resolve())
    assert fake_build_module.checkpoint_path == str(checkpoint_path.resolve())
    assert fake_build_module.device == "cuda:0"
    assert fake_predictor_module.predictor.model == "fake-sam2-model"
    assert fake_image_module.last_path == image_path
    assert len(candidates) == 2
    assert candidates[0].candidate_id == "mask_00"
    assert candidates[0].score == 0.8
    assert candidates[0].pixel_count == 3
    assert candidates[0].coverage_percent == 50.0
    assert candidates[1].candidate_id == "mask_01"
    assert candidates[1].score == 0.25
    assert candidates[1].pixel_count == 2
    assert candidates[1].coverage_percent == pytest.approx(33.33333333333333)
    assert candidates[0].mask_npz_path is not None
    first_mask = _load_saved_mask(candidates[0].mask_npz_path)
    assert first_mask.dtype == np.bool_
    assert first_mask.shape == (2, 3)
    assert first_mask.tolist() == [
        [True, False, True],
        [False, False, True],
    ]
    predictor = fake_predictor_module.predictor
    assert predictor.image_shape == (2, 3, 3)
    point_coords = predictor.point_coords
    point_labels = predictor.point_labels
    assert point_coords is not None
    assert point_labels is not None
    assert point_coords.dtype == np.float32
    assert point_coords.tolist() == [[1.0, 0.5]]
    assert point_labels.dtype == np.int32
    assert point_labels.tolist() == [1]
    assert predictor.multimask_output is True


def test_official_runner_rejects_staging_dir_outside_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_agent.scenefunc3d.servers import sam2_mask_server

    checkpoint_path = tmp_path / "sam2.pt"
    config_path = tmp_path / "sam2.yaml"
    image_path = tmp_path / "frame.jpg"
    allowed_root = tmp_path / "allowed"
    checkpoint_path.write_bytes(b"checkpoint")
    config_path.write_text("model: fake\n", encoding="utf-8")
    image_path.write_bytes(b"image")
    _install_fake_runtime(monkeypatch, sam2_mask_server)
    runner = sam2_mask_server.OfficialSam2Runner(
        model_name="SAM2.1-Hiera-L",
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        staging_root=allowed_root,
        device="cuda:0",
    )
    request = _sam_request(
        image_path=image_path,
        staging_dir=tmp_path / "outside" / "candidates",
    )

    with pytest.raises(RuntimeError, match="outside configured staging root"):
        runner.masks(request)


def test_official_runner_rejects_non_binary_predictor_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_agent.scenefunc3d.servers import sam2_mask_server

    checkpoint_path = tmp_path / "sam2.pt"
    config_path = tmp_path / "sam2.yaml"
    image_path = tmp_path / "frame.jpg"
    checkpoint_path.write_bytes(b"checkpoint")
    config_path.write_text("model: fake\n", encoding="utf-8")
    image_path.write_bytes(b"image")
    _install_fake_runtime(
        monkeypatch,
        sam2_mask_server,
        predictor_module=_FloatMaskPredictorModule(),
    )
    runner = sam2_mask_server.OfficialSam2Runner(
        model_name="SAM2.1-Hiera-L",
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        staging_root=tmp_path / "out",
        device="cuda:0",
    )

    with pytest.raises(RuntimeError, match="binary"):
        runner.masks(
            _sam_request(
                image_path=image_path,
                staging_dir=tmp_path / "out" / "sam" / "000050" / "candidates",
            )
        )


def test_official_runner_maps_missing_pillow_to_image_load_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_agent.scenefunc3d.servers import sam2_mask_server

    checkpoint_path = tmp_path / "sam2.pt"
    config_path = tmp_path / "sam2.yaml"
    image_path = tmp_path / "frame.jpg"
    checkpoint_path.write_bytes(b"checkpoint")
    config_path.write_text("model: fake\n", encoding="utf-8")
    image_path.write_bytes(b"image")
    _install_fake_runtime(monkeypatch, sam2_mask_server)
    runner = sam2_mask_server.OfficialSam2Runner(
        model_name="SAM2.1-Hiera-L",
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        staging_root=tmp_path / "out",
        device="cuda:0",
    )

    def missing_pillow_import(module_name: str) -> ModuleType:
        if module_name == "PIL.Image":
            raise ImportError("missing pillow")
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(
        sam2_mask_server.importlib, "import_module", missing_pillow_import
    )

    with pytest.raises(RuntimeError, match="PIL.Image"):
        runner.masks(
            _sam_request(
                image_path=image_path,
                staging_dir=tmp_path / "out" / "sam" / "000050" / "candidates",
            )
        )


def test_official_runner_rejects_unparseable_predictor_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_agent.scenefunc3d.servers import sam2_mask_server

    checkpoint_path = tmp_path / "sam2.pt"
    config_path = tmp_path / "sam2.yaml"
    image_path = tmp_path / "frame.jpg"
    checkpoint_path.write_bytes(b"checkpoint")
    config_path.write_text("model: fake\n", encoding="utf-8")
    image_path.write_bytes(b"image")
    _install_fake_runtime(
        monkeypatch,
        sam2_mask_server,
        predictor_module=_BadScorePredictorModule(),
    )
    runner = sam2_mask_server.OfficialSam2Runner(
        model_name="SAM2.1-Hiera-L",
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        staging_root=tmp_path / "out",
        device="cuda:0",
    )

    with pytest.raises(RuntimeError, match="scores"):
        runner.masks(
            _sam_request(
                image_path=image_path,
                staging_dir=tmp_path / "out" / "sam" / "000050" / "candidates",
            )
        )


def test_transformers_runner_writes_valid_candidate_masks_with_fake_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_agent.scenefunc3d.servers import sam2_mask_server

    model_path = tmp_path / "hf_snapshot"
    image_path = tmp_path / "frame.jpg"
    staging_dir = tmp_path / "out" / "sam" / "000050" / "candidates"
    model_path.mkdir()
    image_path.write_bytes(b"image")
    fake_torch_module = _FakeTorchModule()
    fake_image_module = _FakeImageModule()
    fake_transformers_module = _FakeTransformersModule(torch_module=fake_torch_module)

    def fake_import_module(module_name: str) -> ModuleType:
        modules_by_name: dict[str, ModuleType] = {
            "PIL.Image": fake_image_module,
            "torch": fake_torch_module,
            "transformers": fake_transformers_module,
        }
        return modules_by_name[module_name]

    monkeypatch.setattr(sam2_mask_server.importlib, "import_module", fake_import_module)

    runner = sam2_mask_server.TransformersSam2Runner(
        model_name="SAM2.1-Hiera-L",
        model_reference=str(model_path),
        staging_root=tmp_path / "out",
        device="cuda:0",
    )
    request = _sam_request(image_path=image_path, staging_dir=staging_dir)
    candidates = runner.masks(request)

    assert fake_transformers_module.model_loader.model_reference == str(model_path)
    assert fake_transformers_module.model_loader.torch_dtype == "fake-bfloat16"
    assert fake_transformers_module.model.device == "cuda:0"
    assert fake_transformers_module.model.eval_called is True
    assert fake_transformers_module.processor_loader.model_reference == str(model_path)
    assert fake_image_module.last_path == image_path
    assert len(candidates) == 2
    assert candidates[0].candidate_id == "mask_00"
    assert candidates[0].score == pytest.approx(0.8)
    assert candidates[0].pixel_count == 3
    assert candidates[0].coverage_percent == 50.0
    assert candidates[1].candidate_id == "mask_01"
    assert candidates[1].score == pytest.approx(0.25)
    assert candidates[1].pixel_count == 2
    assert candidates[1].coverage_percent == pytest.approx(33.33333333333333)
    assert candidates[0].mask_npz_path is not None
    assert _load_saved_mask(candidates[0].mask_npz_path).tolist() == [
        [True, False, True],
        [False, False, True],
    ]
    assert fake_transformers_module.processor.image_shape == (2, 3, 3)
    assert fake_transformers_module.processor.input_points == [[[[1.0, 0.5]]]]
    assert fake_transformers_module.processor.input_labels == [[[1]]]
    assert fake_transformers_module.processor.device == "cuda:0"
    assert fake_transformers_module.model.input_devices == {
        "pixel_values": "cuda:0",
        "input_points": "cuda:0",
        "input_labels": "cuda:0",
    }
    assert fake_transformers_module.model.input_dtypes["pixel_values"] == (
        "fake-bfloat16"
    )
    assert fake_transformers_module.model.received_keys == (
        "input_labels",
        "input_points",
        "pixel_values",
    )
    assert fake_torch_module.inference_mode_enter_count == 1


def test_transformers_runner_casts_bfloat16_scores_before_numpy_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_agent.scenefunc3d.servers import sam2_mask_server

    model_path = tmp_path / "hf_snapshot"
    image_path = tmp_path / "frame.jpg"
    staging_dir = tmp_path / "out" / "sam" / "000050" / "candidates"
    model_path.mkdir()
    image_path.write_bytes(b"image")
    fake_torch_module = _FakeTorchModule()
    fake_transformers_module = _FakeTransformersModule(
        torch_module=fake_torch_module,
        score_dtype=fake_torch_module.bfloat16,
    )

    def fake_import_module(module_name: str) -> ModuleType:
        modules_by_name: dict[str, ModuleType] = {
            "PIL.Image": _FakeImageModule(),
            "torch": fake_torch_module,
            "transformers": fake_transformers_module,
        }
        return modules_by_name[module_name]

    monkeypatch.setattr(sam2_mask_server.importlib, "import_module", fake_import_module)
    runner = sam2_mask_server.TransformersSam2Runner(
        model_name="SAM2.1-Hiera-L",
        model_reference=str(model_path),
        staging_root=tmp_path / "out",
        device="cuda:0",
    )

    candidates = runner.masks(
        _sam_request(image_path=image_path, staging_dir=staging_dir)
    )

    assert [candidate.score for candidate in candidates] == pytest.approx([0.8, 0.25])


def test_transformers_runner_rejects_non_binary_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_agent.scenefunc3d.servers import sam2_mask_server

    model_path = tmp_path / "hf_snapshot"
    image_path = tmp_path / "frame.jpg"
    model_path.mkdir()
    image_path.write_bytes(b"image")
    fake_torch_module = _FakeTorchModule()
    fake_transformers_module = _FakeTransformersModule(
        torch_module=fake_torch_module,
        post_processed_masks=np.array(
            [[[[0.1, 0.0, 0.8], [0.0, 0.2, 0.0]]]],
            dtype=np.float32,
        ),
    )

    def fake_import_module(module_name: str) -> ModuleType:
        modules_by_name: dict[str, ModuleType] = {
            "PIL.Image": _FakeImageModule(),
            "torch": fake_torch_module,
            "transformers": fake_transformers_module,
        }
        return modules_by_name[module_name]

    monkeypatch.setattr(sam2_mask_server.importlib, "import_module", fake_import_module)
    runner = sam2_mask_server.TransformersSam2Runner(
        model_name="SAM2.1-Hiera-L",
        model_reference=str(model_path),
        staging_root=tmp_path / "out",
        device="cuda:0",
    )

    with pytest.raises(RuntimeError, match="binary"):
        runner.masks(
            _sam_request(
                image_path=image_path,
                staging_dir=tmp_path / "out" / "sam" / "000050" / "candidates",
            )
        )


def test_transformers_runner_rejects_non_tensor_original_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codex_agent.scenefunc3d.servers import sam2_mask_server

    model_path = tmp_path / "hf_snapshot"
    image_path = tmp_path / "frame.jpg"
    model_path.mkdir()
    image_path.write_bytes(b"image")
    fake_torch_module = _FakeTorchModule()
    fake_transformers_module = _FakeTransformersModule(
        torch_module=fake_torch_module,
        original_sizes_value="not-a-tensor",
    )

    def fake_import_module(module_name: str) -> ModuleType:
        modules_by_name: dict[str, ModuleType] = {
            "PIL.Image": _FakeImageModule(),
            "torch": fake_torch_module,
            "transformers": fake_transformers_module,
        }
        return modules_by_name[module_name]

    monkeypatch.setattr(sam2_mask_server.importlib, "import_module", fake_import_module)
    runner = sam2_mask_server.TransformersSam2Runner(
        model_name="SAM2.1-Hiera-L",
        model_reference=str(model_path),
        staging_root=tmp_path / "out",
        device="cuda:0",
    )

    with pytest.raises(RuntimeError, match="original_sizes"):
        runner.masks(
            _sam_request(
                image_path=image_path,
                staging_dir=tmp_path / "out" / "sam" / "000050" / "candidates",
            )
        )


def test_handler_supports_health_and_masks_routes(tmp_path: Path) -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import _build_routes

    image_path = tmp_path / "frame.jpg"
    staging_dir = tmp_path / "out" / "sam" / "000050" / "candidates"
    image_path.write_bytes(b"image")
    server = _start_json_server(_build_routes(_FakeSamRunner()))
    try:
        health_payload = _get_json(f"http://127.0.0.1:{server.server_port}/health")
        mask_payload = _post_json(
            f"http://127.0.0.1:{server.server_port}/v1/masks",
            {
                "request_id": "req-1",
                "image_path": str(image_path),
                "points": [
                    {
                        "x_px": 10.0,
                        "y_px": 20.0,
                        "label": "drawer",
                        "source": '<point x="10" y="20">drawer</point>',
                    }
                ],
                "staging_dir": str(staging_dir),
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    assert health_payload == {
        "status": "ok",
        "model_name": "fake-sam2",
        "model_loaded": True,
    }
    assert mask_payload["request_id"] == "req-1"
    assert mask_payload["model_name"] == "fake-sam2"
    assert mask_payload["candidates"] == [
        {
            "candidate_id": "mask_00",
            "score": 0.91,
            "mask_rle": {
                "encoding": "row_major_counts",
                "height": 2,
                "width": 3,
                "counts": [0, 1, 1, 1, 2, 1],
            },
            "mask_npz_base64": "",
            "mask_npz_sha256": "",
            "pixel_count": 3,
            "coverage_percent": 50.0,
        }
    ]
    assert isinstance(mask_payload["latency_ms"], float)


def test_handler_materializes_inline_mask_request() -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import _build_routes

    runner = _RecordingSamRunner()
    server = _start_json_server(_build_routes(runner))
    try:
        mask_payload = _post_json(
            f"http://127.0.0.1:{server.server_port}/v1/masks",
            {
                "request_id": "req-inline",
                "image": _inline_image_payload(b"inline image bytes"),
                "points": [
                    {
                        "x_px": 10.0,
                        "y_px": 20.0,
                        "label": "drawer",
                        "source": '<point x="10" y="20">drawer</point>',
                    }
                ],
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    assert mask_payload["request_id"] == "req-inline"
    assert runner.image_existed_during_call is True
    assert runner.seen_image_path is not None
    assert runner.seen_staging_dir is not None
    assert not runner.seen_image_path.exists()
    assert not runner.seen_staging_dir.exists()


def test_inline_mask_request_uses_runner_staging_root(tmp_path: Path) -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import run_sam_mask_request

    staging_root = tmp_path / "staging-root"
    staging_root.mkdir()
    runner = _RootCheckingSamRunner(staging_root)
    request = SamMaskRequest.model_validate(
        {
            "request_id": "req-inline-root",
            "image": _inline_image_payload(b"inline image bytes"),
            "points": [
                {
                    "x_px": 10.0,
                    "y_px": 20.0,
                    "label": "drawer",
                    "source": '<point x="10" y="20">drawer</point>',
                }
            ],
        }
    )

    candidates = run_sam_mask_request(runner, request)

    assert candidates[0].candidate_id == "mask_00"
    assert runner.seen_staging_dir is not None
    assert not runner.seen_staging_dir.exists()


def test_invalid_mask_request_returns_recoverable_http_error(
    tmp_path: Path,
) -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import _build_routes

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    server = _start_json_server(_build_routes(_FakeSamRunner()))
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_json(
                f"http://127.0.0.1:{server.server_port}/v1/masks",
                {
                    "request_id": "req-1",
                    "image_path": str(image_path),
                    "points": [],
                    "staging_dir": str(tmp_path / "out"),
                },
            )
        error_payload = json.loads(exc_info.value.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()

    assert exc_info.value.code == 400
    assert error_payload["error"] == "invalid_request"
    assert "points" in json.dumps(error_payload)


def test_runner_oom_returns_resource_http_error(tmp_path: Path) -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import _build_routes

    image_path = tmp_path / "frame.jpg"
    staging_dir = tmp_path / "out" / "sam" / "000050" / "candidates"
    image_path.write_bytes(b"image")
    server = _start_json_server(_build_routes(_OutOfMemorySamRunner()))
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_json(
                f"http://127.0.0.1:{server.server_port}/v1/masks",
                {
                    "request_id": "req-1",
                    "image_path": str(image_path),
                    "points": [
                        {
                            "x_px": 10.0,
                            "y_px": 20.0,
                            "label": "drawer",
                            "source": '<point x="10" y="20">drawer</point>',
                        }
                    ],
                    "staging_dir": str(staging_dir),
                },
            )
        error_payload = json.loads(exc_info.value.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()

    assert exc_info.value.code == 503
    assert error_payload == {"error": "sam_resource_unavailable"}


def test_runner_invalid_output_returns_structured_http_error(tmp_path: Path) -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import (
        SamInvalidOutputError,
        _build_routes,
    )

    class _InvalidOutputSamRunner:
        model_name: str = "fake-sam2"

        def masks(
            self, request: SamMaskRequest
        ) -> tuple[SamMaskCandidateResponse, ...]:
            raise SamInvalidOutputError("predictor returned invalid masks")

    image_path = tmp_path / "frame.jpg"
    staging_dir = tmp_path / "out" / "sam" / "000050" / "candidates"
    image_path.write_bytes(b"image")
    server = _start_json_server(_build_routes(_InvalidOutputSamRunner()))
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_json(
                f"http://127.0.0.1:{server.server_port}/v1/masks",
                {
                    "request_id": "req-1",
                    "image_path": str(image_path),
                    "points": [
                        {
                            "x_px": 10.0,
                            "y_px": 20.0,
                            "label": "drawer",
                            "source": '<point x="10" y="20">drawer</point>',
                        }
                    ],
                    "staging_dir": str(staging_dir),
                },
            )
        error_payload = json.loads(exc_info.value.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()

    assert exc_info.value.code == 500
    assert error_payload == {"error": "sam_invalid_output", "request_id": "req-1"}


def test_empty_mask_artifact_returns_structured_invalid_output_error(
    tmp_path: Path,
) -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import _build_routes

    image_path = tmp_path / "frame.jpg"
    staging_dir = tmp_path / "out" / "sam" / "000050" / "candidates"
    image_path.write_bytes(b"image")
    server = _start_json_server(_build_routes(_EmptyMaskSamRunner()))
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_json(
                f"http://127.0.0.1:{server.server_port}/v1/masks",
                {
                    "request_id": "req-empty-mask",
                    "image_path": str(image_path),
                    "points": [
                        {
                            "x_px": 10.0,
                            "y_px": 20.0,
                            "label": "drawer",
                            "source": '<point x="10" y="20">drawer</point>',
                        }
                    ],
                    "staging_dir": str(staging_dir),
                },
            )
        error_payload = json.loads(exc_info.value.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()

    assert exc_info.value.code == 500
    assert error_payload == {
        "error": "sam_invalid_output",
        "request_id": "req-empty-mask",
    }


def _restore_modules(modules_by_name: dict[str, ModuleType]) -> None:
    for module_name, module in modules_by_name.items():
        sys.modules[module_name] = module


def _load_saved_mask(path: Path) -> NDArray[np.bool_]:
    with np.load(path) as loaded_npz:
        mask_array = loaded_npz["mask"]
    if mask_array.dtype != np.bool_:
        raise AssertionError(f"expected bool mask, got {mask_array.dtype}")
    return cast(NDArray[np.bool_], mask_array)


def _start_json_server(routes: dict[str, JsonRoute]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_json_handler(routes))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _get_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=2.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("expected JSON object response")
    return payload


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2.0) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(response_payload, dict):
        raise AssertionError("expected JSON object response")
    return response_payload


def _sam_request(*, image_path: Path, staging_dir: Path) -> SamMaskRequest:
    return SamMaskRequest.model_validate(
        {
            "request_id": "req-1",
            "image_path": image_path,
            "points": [
                {
                    "x_px": 1.0,
                    "y_px": 0.5,
                    "label": "drawer",
                    "source": '<point x="1" y="0.5">drawer</point>',
                }
            ],
            "staging_dir": staging_dir,
        }
    )


def _inline_image_payload(image_bytes: bytes) -> dict[str, object]:
    return {
        "filename": "frame.jpg",
        "mime_type": "image/jpeg",
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "data_base64": base64.b64encode(image_bytes).decode("ascii"),
    }


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    sam2_mask_server: ModuleType,
    *,
    predictor_module: ModuleType | None = None,
) -> None:
    fake_build_module = _FakeSam2BuildModule()
    fake_predictor_module = (
        predictor_module if predictor_module is not None else _FakePredictorModule()
    )
    fake_torch_module = _FakeTorchModule()
    fake_image_module = _FakeImageModule()

    def fake_import_module(module_name: str) -> ModuleType:
        modules_by_name: dict[str, ModuleType] = {
            "PIL.Image": fake_image_module,
            "sam2.build_sam": fake_build_module,
            "sam2.sam2_image_predictor": fake_predictor_module,
            "torch": fake_torch_module,
        }
        return modules_by_name[module_name]

    monkeypatch.setattr(sam2_mask_server.importlib, "import_module", fake_import_module)


class _FakeTorchCuda:
    def is_available(self) -> bool:
        return True


class _FakeTorchModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.cuda = _FakeTorchCuda()
        self.bfloat16 = "fake-bfloat16"
        self.inference_mode_depth = 0
        self.inference_mode_enter_count = 0

    def inference_mode(self) -> _FakeInferenceMode:
        return _FakeInferenceMode(torch_module=self)


class _FakeInferenceMode:
    def __init__(self, *, torch_module: _FakeTorchModule) -> None:
        self._torch_module = torch_module

    def __enter__(self) -> None:
        self._torch_module.inference_mode_depth += 1
        self._torch_module.inference_mode_enter_count += 1

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> Literal[False]:
        self._torch_module.inference_mode_depth -= 1
        return False


class _FakeSam2BuildModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("sam2.build_sam")
        self.config_path = ""
        self.checkpoint_path = ""
        self.device = ""

    def build_sam2(
        self,
        config_file: str,
        ckpt_path: str,
        *,
        device: str,
    ) -> str:
        self.config_path = config_file
        self.checkpoint_path = ckpt_path
        self.device = device
        return "fake-sam2-model"


class _FakePredictorModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("sam2.sam2_image_predictor")
        self.predictor = _FakePredictor(model="")
        self.SAM2ImagePredictor = self._build_predictor

    def _build_predictor(self, model: object) -> _FakePredictor:
        self.predictor = _FakePredictor(model=model)
        return self.predictor


class _FloatMaskPredictorModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("sam2.sam2_image_predictor")
        self.predictor = _FloatMaskPredictor(model="")
        self.SAM2ImagePredictor = self._build_predictor

    def _build_predictor(self, model: object) -> _FloatMaskPredictor:
        self.predictor = _FloatMaskPredictor(model=model)
        return self.predictor


class _BadScorePredictorModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("sam2.sam2_image_predictor")
        self.predictor = _BadScorePredictor(model="")
        self.SAM2ImagePredictor = self._build_predictor

    def _build_predictor(self, model: object) -> _BadScorePredictor:
        self.predictor = _BadScorePredictor(model=model)
        return self.predictor


class _FakeTransformersModule(ModuleType):
    def __init__(
        self,
        *,
        torch_module: _FakeTorchModule,
        post_processed_masks: NDArray[np.bool_] | NDArray[np.float32] | None = None,
        original_sizes_value: object = None,
        score_dtype: object = None,
    ) -> None:
        super().__init__("transformers")
        self.model = _FakeSam2TransformersModel(
            torch_module=torch_module,
            score_dtype=score_dtype,
        )
        self.processor = _FakeSam2Processor(
            post_processed_masks=post_processed_masks,
            original_sizes_value=original_sizes_value,
        )
        self.model_loader = _FakeSam2ModelLoader(self.model)
        self.processor_loader = _FakeSam2ProcessorLoader(self.processor)
        self.Sam2Model = self.model_loader
        self.Sam2Processor = self.processor_loader


class _FakeSam2ModelLoader:
    def __init__(self, model: _FakeSam2TransformersModel) -> None:
        self._model = model
        self.model_reference = ""
        self.torch_dtype: object = None

    def from_pretrained(
        self,
        model_reference: str,
        *,
        torch_dtype: object,
    ) -> _FakeSam2TransformersModel:
        self.model_reference = model_reference
        self.torch_dtype = torch_dtype
        return self._model


class _FakeSam2ProcessorLoader:
    def __init__(self, processor: _FakeSam2Processor) -> None:
        self._processor = processor
        self.model_reference = ""

    def from_pretrained(self, model_reference: str) -> _FakeSam2Processor:
        self.model_reference = model_reference
        return self._processor


class _FakeSam2TransformersModel:
    def __init__(
        self,
        *,
        torch_module: _FakeTorchModule,
        score_dtype: object,
    ) -> None:
        self._torch_module = torch_module
        self._score_dtype = score_dtype
        self.device = ""
        self.eval_called = False
        self.input_devices: dict[str, str] = {}
        self.input_dtypes: dict[str, object] = {}
        self.received_keys: tuple[str, ...] = ()

    def to(self, device: str) -> _FakeSam2TransformersModel:
        self.device = device
        return self

    def eval(self) -> _FakeSam2TransformersModel:
        self.eval_called = True
        return self

    def __call__(self, **model_inputs: object) -> _FakeSam2Output:
        expected_keys = ("input_labels", "input_points", "pixel_values")
        self.received_keys = tuple(sorted(model_inputs))
        if self.received_keys != expected_keys:
            raise AssertionError(f"unexpected model input keys: {self.received_keys}")
        if self._torch_module.inference_mode_depth <= 0:
            raise AssertionError("expected torch.inference_mode around model call")
        input_devices: dict[str, str] = {}
        for key in expected_keys:
            value = model_inputs[key]
            if not isinstance(value, _FakeTensor):
                raise AssertionError(f"expected fake tensor for {key}")
            input_devices[key] = value.device
            self.input_dtypes[key] = value.dtype
        self.input_devices = input_devices
        return _FakeSam2Output(
            pred_masks=_FakeTensor(np.zeros((1, 1, 2, 2, 3), dtype=np.float32)),
            iou_scores=_FakeTensor(
                np.array([[[0.8, 0.25]]], dtype=np.float32),
                dtype=self._score_dtype,
            ),
        )


class _FakeSam2Output:
    def __init__(self, *, pred_masks: _FakeTensor, iou_scores: _FakeTensor) -> None:
        self.pred_masks = pred_masks
        self.iou_scores = iou_scores


class _FakeSam2Processor:
    def __init__(
        self,
        *,
        post_processed_masks: NDArray[np.bool_] | NDArray[np.float32] | None,
        original_sizes_value: object,
    ) -> None:
        self._post_processed_masks: NDArray[np.bool_] | NDArray[np.float32]
        if post_processed_masks is None:
            self._post_processed_masks = np.array(
                [
                    [
                        [[True, False, True], [False, False, True]],
                        [[False, True, False], [True, False, False]],
                    ]
                ],
                dtype=np.bool_,
            )
        else:
            self._post_processed_masks = post_processed_masks
        self.image_shape: tuple[int, ...] = ()
        self.input_points: list[list[list[list[float]]]] = []
        self.input_labels: list[list[list[int]]] = []
        self.device = ""
        self.original_sizes_value = original_sizes_value

    def __call__(
        self,
        *,
        images: _FakeImage,
        input_points: list[list[list[list[float]]]],
        input_labels: list[list[list[int]]],
        return_tensors: str,
    ) -> _FakeProcessorInputs:
        if return_tensors != "pt":
            raise AssertionError(f"unexpected return_tensors: {return_tensors}")
        self.image_shape = np.asarray(images, dtype=np.uint8).shape
        self.input_points = input_points
        self.input_labels = input_labels
        return _FakeProcessorInputs(processor=self)

    def post_process_masks(
        self,
        pred_masks: object,
        original_sizes: object,
    ) -> tuple[NDArray[np.bool_] | NDArray[np.float32], ...]:
        if not isinstance(pred_masks, _FakeTensor):
            raise AssertionError("expected fake pred_masks tensor")
        if not isinstance(original_sizes, _FakeTensor):
            raise AssertionError("expected fake original_sizes tensor")
        return (self._post_processed_masks,)


class _FakeProcessorInputs(dict[str, object]):
    def __init__(self, *, processor: _FakeSam2Processor) -> None:
        original_sizes: object = processor.original_sizes_value
        if original_sizes is None:
            original_sizes = _FakeTensor(np.array([[2, 3]], dtype=np.int64))
        super().__init__(
            {
                "pixel_values": _FakeTensor(np.zeros((1, 3, 2, 3), dtype=np.float32)),
                "input_points": _FakeTensor(
                    np.array([[[[1.0, 0.5]]]], dtype=np.float32)
                ),
                "input_labels": _FakeTensor(np.array([[[1]]], dtype=np.int64)),
                "original_sizes": original_sizes,
                "reshaped_input_sizes": _FakeTensor(
                    np.array([[1024, 1024]], dtype=np.int64)
                ),
            }
        )
        self._processor = processor

    def to(self, device: str) -> _FakeProcessorInputs:
        self._processor.device = device
        for value in self.values():
            if isinstance(value, _FakeTensor):
                value.to(device)
        return self


class _FakeTensor:
    def __init__(self, value: NDArray[np.generic], *, dtype: object = None) -> None:
        self._value = value
        self.device = ""
        self.dtype: object = dtype

    def to(
        self,
        device: str | None = None,
        *,
        dtype: object | None = None,
    ) -> _FakeTensor:
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def float(self) -> _FakeTensor:
        return _FakeTensor(self._value.astype(np.float32), dtype="fake-float32")

    def __getitem__(self, key: int) -> _FakeTensor:
        return _FakeTensor(np.asarray(self._value[key]), dtype=self.dtype)

    def __array__(self, dtype: DTypeLike | None = None) -> NDArray[np.generic]:
        if self.dtype == "fake-bfloat16":
            raise TypeError("Got unsupported ScalarType BFloat16")
        if dtype is None:
            return self._value
        return np.asarray(self._value, dtype=dtype)


class _FakeImageModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("PIL.Image")
        self.last_path = Path()

    def open(self, fp: Path) -> _FakeImage:
        self.last_path = fp
        return _FakeImage()


class _FakeImage:
    def __enter__(self) -> _FakeImage:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> Literal[False]:
        return False

    def convert(self, mode: str) -> _FakeImage:
        if mode != "RGB":
            raise AssertionError(f"unexpected image mode: {mode}")
        return self

    def __array__(self, dtype: object | None = None) -> NDArray[np.uint8]:
        return np.zeros((2, 3, 3), dtype=np.uint8)


class _FakePredictor:
    def __init__(self, *, model: object) -> None:
        self.model = model
        self.image_shape: tuple[int, ...] = ()
        self.point_coords: NDArray[np.float32] | None = None
        self.point_labels: NDArray[np.int32] | None = None
        self.multimask_output = False

    def set_image(self, image: NDArray[np.uint8]) -> None:
        self.image_shape = image.shape

    def predict(
        self,
        *,
        point_coords: NDArray[np.float32],
        point_labels: NDArray[np.int32],
        multimask_output: bool,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float32]]:
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.multimask_output = multimask_output
        masks = np.array(
            [
                [[True, False, True], [False, False, True]],
                [[False, True, False], [True, False, False]],
            ],
            dtype=np.bool_,
        )
        scores = np.array([0.8, 0.25], dtype=np.float64)
        logits = np.zeros((2, 2, 3), dtype=np.float32)
        return masks, scores, logits


class _FloatMaskPredictor:
    def __init__(self, *, model: object) -> None:
        self.model = model
        self.image_shape: tuple[int, ...] = ()
        self.point_coords: NDArray[np.float32] | None = None
        self.point_labels: NDArray[np.int32] | None = None
        self.multimask_output = False

    def set_image(self, image: NDArray[np.uint8]) -> None:
        self.image_shape = image.shape

    def predict(
        self,
        *,
        point_coords: NDArray[np.float32],
        point_labels: NDArray[np.int32],
        multimask_output: bool,
    ) -> tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.float32]]:
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.multimask_output = multimask_output
        masks = np.array(
            [
                [[0.1, 0.0, 0.8], [0.0, 0.2, 0.0]],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.8], dtype=np.float64)
        logits = np.zeros((1, 2, 3), dtype=np.float32)
        return masks, scores, logits


class _BadScorePredictor:
    def __init__(self, *, model: object) -> None:
        self.model = model
        self.image_shape: tuple[int, ...] = ()
        self.point_coords: NDArray[np.float32] | None = None
        self.point_labels: NDArray[np.int32] | None = None
        self.multimask_output = False

    def set_image(self, image: NDArray[np.uint8]) -> None:
        self.image_shape = image.shape

    def predict(
        self,
        *,
        point_coords: NDArray[np.float32],
        point_labels: NDArray[np.int32],
        multimask_output: bool,
    ) -> tuple[NDArray[np.bool_], list[str], NDArray[np.float32]]:
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.multimask_output = multimask_output
        masks = np.array(
            [
                [[True, False, True], [False, False, True]],
            ],
            dtype=np.bool_,
        )
        logits = np.zeros((1, 2, 3), dtype=np.float32)
        return masks, ["not-a-score"], logits
