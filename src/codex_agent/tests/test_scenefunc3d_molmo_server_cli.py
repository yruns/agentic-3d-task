"""Tests for the SceneFunc3D Molmo sidecar server CLI and handlers."""

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
from typing import Literal

import pytest

from codex_agent.scenefunc3d.servers.http_json import JsonRoute, make_json_handler
from codex_agent.scenefunc3d.servers.schemas import MolmoPointRequest


class _FakeMolmoRunner:
    model_name: str = "fake-molmo"

    def point(self, request: MolmoPointRequest) -> str:
        return f'<point x="50" y="50">{request.prompt}</point>'


class _OutOfMemoryMolmoRunner:
    model_name: str = "fake-molmo"

    def point(self, request: MolmoPointRequest) -> str:
        raise RuntimeError("CUDA out of memory while allocating tensor")


def test_build_arg_parser_accepts_runtime_options() -> None:
    from codex_agent.scenefunc3d.servers.molmo_point_server import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "8711",
            "--model-name",
            "MolmoPoint-8B",
            "--model-path",
            "/models/molmo",
            "--device",
            "cuda:0",
        ]
    )

    assert args.host == "127.0.0.1"
    assert args.port == 8711
    assert args.model_name == "MolmoPoint-8B"
    assert args.model_path == Path("/models/molmo")
    assert args.device == "cuda:0"


def test_build_arg_parser_requires_local_model_path() -> None:
    from codex_agent.scenefunc3d.servers.molmo_point_server import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--port", "8711"])


def test_module_import_does_not_import_heavy_dependencies() -> None:
    module_name = "codex_agent.scenefunc3d.servers.molmo_point_server"
    preserved_modules = {
        dependency_name: sys.modules.pop(dependency_name)
        for dependency_name in ("PIL", "torch", "transformers")
        if dependency_name in sys.modules
    }
    sys.modules.pop(module_name, None)

    try:
        imported_module = importlib.import_module(module_name)
        assert isinstance(imported_module, ModuleType)
        assert "PIL" not in sys.modules
        assert "torch" not in sys.modules
        assert "transformers" not in sys.modules
    finally:
        _restore_modules(preserved_modules)


def test_transformers_runner_uses_local_molmo_runtime_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_agent.scenefunc3d.servers import molmo_point_server

    model_path = tmp_path / "molmo"
    model_path.mkdir()
    image_processor_path = model_path / "image_preprocessing_molmo.py"
    image_processor_path.write_text(
        "\nimport tensorflow as tf\n"
        "def resize(resize_method):\n"
        '    if resize_method == "tensorflow":\n'
        "        return tf\n",
        encoding="utf-8",
    )
    modeling_path = model_path / "modeling_molmo.py"
    modeling_path.write_text(
        "class MolmoForCausalLM:\n"
        '    _no_split_modules = ["MolmoBlock"]\n'
        "    def tie_weights(self):\n"
        "        pass\n"
        "    def prepare_inputs_for_generation(self, model_kwargs, num_new_tokens):\n"
        '        if "cache_position" in model_kwargs:\n'
        '            model_kwargs["cache_position"] = '
        'model_kwargs["cache_position"][-1:] + num_new_tokens\n',
        encoding="utf-8",
    )
    fake_torch = _FakeTorchModule()
    fake_transformers = _FakeTransformersModule()
    fake_pil_image = _FakeImageModule()

    def fake_import_module(module_name: str) -> ModuleType:
        modules_by_name: dict[str, ModuleType] = {
            "torch": fake_torch,
            "transformers": fake_transformers,
            "PIL.Image": fake_pil_image,
        }
        return modules_by_name[module_name]

    monkeypatch.setattr(
        molmo_point_server.importlib, "import_module", fake_import_module
    )

    runner = molmo_point_server.TransformersMolmoRunner(
        model_name="MolmoPoint-8B",
        model_path=model_path,
        device="cuda:0",
    )
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    raw_text = runner.point(
        MolmoPointRequest(
            request_id="req-1",
            image_path=image_path,
            prompt="drawer handle",
            image_width=10,
            image_height=10,
        )
    )

    assert raw_text == '<point x="50" y="50">handle</point>'
    assert fake_transformers.model_loader.last_kwargs["local_files_only"] is True
    assert fake_transformers.model_loader.last_kwargs["low_cpu_mem_usage"] is True
    assert (
        fake_transformers.model_loader.last_kwargs["torch_dtype"] is fake_torch.bfloat16
    )
    assert fake_transformers.processor_loader.last_source == str(model_path)
    assert fake_transformers.model.to_device == "device:cuda:0"
    assert fake_transformers.model.eval_called is True
    assert fake_transformers.model._supports_default_dynamic_cache() is False
    assert fake_torch.inference_mode_entered is True
    assert fake_torch.autocast_device_type == "cuda"
    assert fake_transformers.generation_config_kwargs["use_cache"] is True
    assert "\nimport tensorflow as tf\n" not in image_processor_path.read_text(
        encoding="utf-8"
    )
    assert "all_tied_weights_keys = {}" in modeling_path.read_text(encoding="utf-8")


def test_patch_molmo_remote_code_requires_tensorflow_resize_branch(
    tmp_path: Path,
) -> None:
    from codex_agent.scenefunc3d.servers.molmo_point_server import (
        patch_molmo_remote_code,
    )

    model_path = tmp_path / "molmo"
    model_path.mkdir()
    image_processor_path = model_path / "image_preprocessing_molmo.py"
    image_processor_path.write_text("\nimport tensorflow as tf\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="tensorflow resize branch"):
        patch_molmo_remote_code(model_path)


def test_patch_molmo_remote_code_requires_molmo_model_anchor(
    tmp_path: Path,
) -> None:
    from codex_agent.scenefunc3d.servers.molmo_point_server import (
        patch_molmo_remote_code,
    )

    model_path = tmp_path / "molmo"
    model_path.mkdir()
    modeling_path = model_path / "modeling_molmo.py"
    modeling_path.write_text("class OtherModel:\n    pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Molmo model class anchor"):
        patch_molmo_remote_code(model_path)


def test_transformers_runner_rejects_non_cuda_device(
    tmp_path: Path,
) -> None:
    from codex_agent.scenefunc3d.servers import molmo_point_server

    model_path = tmp_path / "molmo"
    model_path.mkdir()

    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        molmo_point_server.TransformersMolmoRunner(
            model_name="MolmoPoint-8B",
            model_path=model_path,
            device="cpu",
        )


def test_handler_supports_health_and_point_routes(tmp_path: Path) -> None:
    from codex_agent.scenefunc3d.servers.molmo_point_server import _build_routes

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    server = _start_json_server(_build_routes(_FakeMolmoRunner()))
    try:
        health_payload = _get_json(f"http://127.0.0.1:{server.server_port}/health")
        point_payload = _post_json(
            f"http://127.0.0.1:{server.server_port}/v1/point",
            {
                "request_id": "req-1",
                "image_path": str(image_path),
                "prompt": "drawer handle",
                "image_width": 640,
                "image_height": 480,
            },
        )
    finally:
        server.shutdown()
        server.server_close()

    assert health_payload == {
        "status": "ok",
        "model_name": "fake-molmo",
        "model_loaded": True,
    }
    assert point_payload["request_id"] == "req-1"
    assert point_payload["model_name"] == "fake-molmo"
    assert point_payload["raw_text"] == '<point x="50" y="50">drawer handle</point>'
    assert isinstance(point_payload["latency_ms"], float)


def test_invalid_point_request_returns_recoverable_http_error(tmp_path: Path) -> None:
    from codex_agent.scenefunc3d.servers.molmo_point_server import _build_routes

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    server = _start_json_server(_build_routes(_FakeMolmoRunner()))
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_json(
                f"http://127.0.0.1:{server.server_port}/v1/point",
                {
                    "request_id": "req-1",
                    "image_path": str(image_path),
                    "prompt": "drawer handle",
                    "image_width": "640",
                    "image_height": 480,
                },
            )
        error_payload = json.loads(exc_info.value.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()

    assert exc_info.value.code == 400
    assert error_payload["error"] == "invalid_request"
    assert "image_width" in json.dumps(error_payload)


def test_runner_oom_returns_resource_http_error(tmp_path: Path) -> None:
    from codex_agent.scenefunc3d.servers.molmo_point_server import _build_routes

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    server = _start_json_server(_build_routes(_OutOfMemoryMolmoRunner()))
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_json(
                f"http://127.0.0.1:{server.server_port}/v1/point",
                {
                    "request_id": "req-1",
                    "image_path": str(image_path),
                    "prompt": "drawer handle",
                    "image_width": 640,
                    "image_height": 480,
                },
            )
        error_payload = json.loads(exc_info.value.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()

    assert exc_info.value.code == 503
    assert error_payload == {"error": "molmo_resource_unavailable"}


def test_handle_point_validates_payload_and_returns_response(tmp_path: Path) -> None:
    from codex_agent.scenefunc3d.servers.molmo_point_server import _handle_point

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    response_payload = _handle_point(
        _FakeMolmoRunner(),
        {
            "request_id": "req-2",
            "image_path": str(image_path),
            "prompt": "cabinet knob",
            "image_width": 320,
            "image_height": 240,
        },
    )

    assert response_payload["request_id"] == "req-2"
    assert response_payload["model_name"] == "fake-molmo"
    assert response_payload["raw_text"] == '<point x="50" y="50">cabinet knob</point>'


def _restore_modules(modules_by_name: dict[str, ModuleType]) -> None:
    for module_name, module in modules_by_name.items():
        sys.modules[module_name] = module


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


class _FakeContext:
    def __init__(self, on_enter: object) -> None:
        self._on_enter = on_enter

    def __enter__(self) -> object:
        if callable(self._on_enter):
            self._on_enter()
        return None

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> Literal[False]:
        return False


class _FakeCuda:
    def is_available(self) -> bool:
        return True


class _FakeTorchModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.bfloat16 = object()
        self.cuda = _FakeCuda()
        self.inference_mode_entered = False
        self.autocast_device_type = ""

    def device(self, value: str) -> str:
        return f"device:{value}"

    def inference_mode(self) -> _FakeContext:
        return _FakeContext(self._mark_inference_mode_entered)

    def autocast(self, device_type: str, *, dtype: object) -> _FakeContext:
        self.autocast_device_type = device_type
        return _FakeContext(lambda: None)

    def _mark_inference_mode_entered(self) -> None:
        self.inference_mode_entered = True


class _FakeFromPretrainedLoader:
    def __init__(self, loaded_object: object) -> None:
        self._loaded_object = loaded_object
        self.last_source = ""
        self.last_kwargs: dict[str, object] = {}

    def from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        **kwargs: object,
    ) -> object:
        self.last_source = pretrained_model_name_or_path
        self.last_kwargs = dict(kwargs)
        return self._loaded_object


class _FakeTensor:
    def __init__(self, token_count: int) -> None:
        self._token_count = token_count

    def to(self, device: object) -> _FakeTensor:
        return self

    def unsqueeze(self, dim: int) -> _FakeTensor:
        return self

    def size(self, dim: int) -> int:
        return self._token_count


class _FakeGeneratedBatch:
    def __getitem__(self, key: object) -> list[int]:
        return [1, 2, 3]


class _FakeTokenizer:
    def decode(self, token_ids: object, *, skip_special_tokens: bool) -> str:
        return '<point x="50" y="50">handle</point>'


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def process(self, *, images: object, text: str) -> dict[str, _FakeTensor]:
        return {"input_ids": _FakeTensor(3), "images": _FakeTensor(0)}


class _FakeModel:
    device = "cuda:0"

    def __init__(self) -> None:
        self.to_device = ""
        self.eval_called = False
        self._supports_default_dynamic_cache = lambda: True

    def to(self, device: object) -> _FakeModel:
        self.to_device = str(device)
        return self

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    def generate_from_batch(
        self,
        inputs: object,
        generation_config: object,
        *,
        tokenizer: object,
    ) -> _FakeGeneratedBatch:
        return _FakeGeneratedBatch()


class _FakeTransformersModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("transformers")
        self.processor = _FakeProcessor()
        self.model = _FakeModel()
        self.processor_loader = _FakeFromPretrainedLoader(self.processor)
        self.model_loader = _FakeFromPretrainedLoader(self.model)
        self.AutoProcessor = self.processor_loader
        self.AutoModelForCausalLM = self.model_loader
        self.generation_config_kwargs: dict[str, object] = {}
        self.GenerationConfig = self._build_generation_config

    def _build_generation_config(self, **kwargs: object) -> dict[str, object]:
        self.generation_config_kwargs = dict(kwargs)
        return dict(kwargs)


class _FakeImageObject:
    def __enter__(self) -> _FakeImageObject:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> Literal[False]:
        return False

    def convert(self, mode: str) -> _FakeImageObject:
        return self


class _FakeImageModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("PIL.Image")

    def open(self, fp: Path) -> _FakeImageObject:
        return _FakeImageObject()
