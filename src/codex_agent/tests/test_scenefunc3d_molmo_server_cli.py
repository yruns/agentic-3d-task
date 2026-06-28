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


def test_transformers_runner_uses_molmopoint_image_text_to_text_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codex_agent.scenefunc3d.servers import molmo_point_server

    model_path = tmp_path / "molmo"
    model_path.mkdir()
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
    assert fake_transformers.model_loader.last_source == str(model_path)
    assert fake_transformers.model_loader.last_kwargs["local_files_only"] is True
    assert fake_transformers.model_loader.last_kwargs["device_map"] == "auto"
    assert (
        fake_transformers.model_loader.last_kwargs["torch_dtype"] is fake_torch.bfloat16
    )
    assert fake_transformers.processor_loader.last_source == str(model_path)
    assert fake_transformers.processor_loader.last_kwargs["padding_side"] == "left"
    assert fake_transformers.model.to_device == ""
    assert fake_transformers.model.eval_called is True
    assert fake_transformers.processor.prompt_text == "drawer handle"
    assert fake_transformers.processor.return_pointing_metadata is True
    assert fake_transformers.model.generated_input_ids is fake_transformers.input_ids
    assert (
        fake_transformers.model.generated_pixel_values is fake_transformers.pixel_values
    )
    assert (
        fake_transformers.model.logit_processor_input_ids is fake_transformers.input_ids
    )
    assert fake_transformers.model.generated_max_new_tokens == 200
    assert fake_transformers.model.generated_metadata_keys == ()
    assert fake_transformers.model.extract_called is True
    assert fake_transformers.model.extracted_text == raw_text
    assert (
        fake_transformers.model.extracted_token_pooling
        is fake_transformers.image_token_pooling
    )
    assert (
        fake_transformers.model.extracted_subpatch_mapping
        is fake_transformers.subpatch_mapping
    )
    assert (
        fake_transformers.model.extracted_image_sizes is fake_transformers.image_sizes
    )
    assert fake_torch.inference_mode_entered is True
    assert fake_torch.autocast_device_type == "cuda"


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


class _FakeGeneratedTokens:
    def __getitem__(self, key: object) -> object:
        return object()


class _FakeProcessor:
    def __init__(self, model_inputs: dict[str, object]) -> None:
        self._model_inputs = model_inputs
        self.prompt_text = ""
        self.return_pointing_metadata = False

    def apply_chat_template(
        self,
        messages: object,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
        return_dict: bool,
        padding: bool,
        return_pointing_metadata: bool,
    ) -> dict[str, object]:
        self.prompt_text = _extract_prompt_text(messages)
        self.return_pointing_metadata = return_pointing_metadata
        return dict(self._model_inputs)

    def post_process_image_text_to_text(
        self,
        generated_tokens: object,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]:
        return ['<point x="50" y="50">handle</point>']


class _FakeModel:
    device = "cuda:0"
    _metadata_keys = ("image_token_pooling_np", "subpatch_mapping", "image_sizes")

    def __init__(self) -> None:
        self.to_device = ""
        self.eval_called = False
        self.logit_processor_input_ids: object = None
        self.generated_input_ids: object = None
        self.generated_pixel_values: object = None
        self.generated_max_new_tokens = 0
        self.generated_metadata_keys: tuple[str, ...] = ()
        self.extract_called = False
        self.extracted_text = ""
        self.extracted_token_pooling: object = None
        self.extracted_subpatch_mapping: object = None
        self.extracted_image_sizes: object = None

    def to(self, device: object) -> _FakeModel:
        self.to_device = str(device)
        return self

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    def build_logit_processor_from_inputs(self, model_inputs: object) -> object:
        if not isinstance(model_inputs, dict):
            raise AssertionError("expected model input mapping")
        self.logit_processor_input_ids = model_inputs["input_ids"]
        return "fake-logit-processor"

    def generate(
        self,
        **kwargs: object,
    ) -> _FakeGeneratedTokens:
        self.generated_input_ids = kwargs["input_ids"]
        self.generated_pixel_values = kwargs["pixel_values"]
        self.generated_metadata_keys = tuple(
            key for key in kwargs if key in self._metadata_keys
        )
        if self.generated_metadata_keys:
            raise AssertionError("MolmoPoint metadata must not be passed to generate")
        max_new_tokens = kwargs["max_new_tokens"]
        if not isinstance(max_new_tokens, int):
            raise AssertionError("expected integer max_new_tokens")
        self.generated_max_new_tokens = max_new_tokens
        return _FakeGeneratedTokens()

    def extract_image_points(
        self,
        generated_text: str,
        token_pooling: object,
        subpatch_mapping: object,
        image_sizes: object,
    ) -> object:
        self.extract_called = True
        self.extracted_text = generated_text
        self.extracted_token_pooling = token_pooling
        self.extracted_subpatch_mapping = subpatch_mapping
        self.extracted_image_sizes = image_sizes
        return object()


class _FakeTransformersModule(ModuleType):
    def __init__(self) -> None:
        super().__init__("transformers")
        self.input_ids = _FakeTensor(3)
        self.pixel_values = _FakeTensor(0)
        self.image_token_pooling = object()
        self.subpatch_mapping = object()
        self.image_sizes = object()
        self.model_inputs: dict[str, object] = {
            "input_ids": self.input_ids,
            "pixel_values": self.pixel_values,
            "image_token_pooling_np": self.image_token_pooling,
            "subpatch_mapping": self.subpatch_mapping,
            "image_sizes": self.image_sizes,
        }
        self.processor = _FakeProcessor(self.model_inputs)
        self.model = _FakeModel()
        self.processor_loader = _FakeFromPretrainedLoader(self.processor)
        self.model_loader = _FakeFromPretrainedLoader(self.model)
        self.AutoProcessor = self.processor_loader
        self.AutoModelForImageTextToText = self.model_loader


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


def _extract_prompt_text(messages: object) -> str:
    if not isinstance(messages, list):
        raise AssertionError("expected chat messages list")
    first_message = messages[0]
    if not isinstance(first_message, dict):
        raise AssertionError("expected chat message object")
    content = first_message["content"]
    if not isinstance(content, list):
        raise AssertionError("expected chat content list")
    first_content = content[0]
    if not isinstance(first_content, dict):
        raise AssertionError("expected chat content object")
    text = first_content["text"]
    if not isinstance(text, str):
        raise AssertionError("expected prompt text")
    return text
