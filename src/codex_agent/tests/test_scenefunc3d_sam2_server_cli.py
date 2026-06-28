"""Tests for the SceneFunc3D SAM2 sidecar server CLI and handlers."""

from __future__ import annotations

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
from numpy.typing import NDArray

from codex_agent.scenefunc3d.servers.http_json import JsonRoute, make_json_handler
from codex_agent.scenefunc3d.servers.schemas import (
    SamMaskCandidateResponse,
    SamMaskRequest,
)


class _FakeSamRunner:
    model_name: str = "fake-sam2"

    def masks(self, request: SamMaskRequest) -> tuple[SamMaskCandidateResponse, ...]:
        request.staging_dir.mkdir(parents=True, exist_ok=True)
        mask_path = request.staging_dir / "mask_00.npz"
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


def test_build_arg_parser_accepts_runtime_options() -> None:
    from codex_agent.scenefunc3d.servers.sam2_mask_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(
        [
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

    assert args.host == "127.0.0.1"
    assert args.port == 8712
    assert args.model_name == "SAM2.1-Hiera-L"
    assert args.checkpoint_path == Path("/models/sam2.pt")
    assert args.config_path == Path("/models/sam2.yaml")
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


def test_module_import_does_not_import_heavy_dependencies() -> None:
    module_name = "codex_agent.scenefunc3d.servers.sam2_mask_server"
    heavy_modules = (
        "PIL",
        "PIL.Image",
        "sam2",
        "sam2.build_sam",
        "sam2.sam2_image_predictor",
        "torch",
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
            "mask_npz_path": str(staging_dir / "mask_00.npz"),
            "pixel_count": 3,
            "coverage_percent": 50.0,
        }
    ]
    assert isinstance(mask_payload["latency_ms"], float)


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
