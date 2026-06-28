"""Molmo point sidecar server for SceneFunc3D agent tools."""

from __future__ import annotations

import argparse
import importlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypedDict, cast

from pydantic import ValidationError

from codex_agent.scenefunc3d.servers.http_json import (
    JsonHttpError,
    JsonObject,
    JsonRoute,
    make_json_handler,
)
from codex_agent.scenefunc3d.servers.schemas import (
    HealthResponse,
    MolmoPointRequest,
    MolmoPointResponse,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8711
DEFAULT_MODEL_NAME = "MolmoPoint-8B"
DEFAULT_DEVICE = "cuda:0"
MAX_NEW_TOKENS = 200
MOLMO_STOP_STRING = "<|endoftext|>"

_CUDA_OOM_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
)


class MolmoRunner(Protocol):
    """Inference boundary used by the Molmo HTTP route."""

    model_name: str

    def point(self, request: MolmoPointRequest) -> str:
        """Return raw Molmo generated text for one pointing request."""


class _FromPretrainedLoader(Protocol):
    def from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        **kwargs: object,
    ) -> object:
        """Load an object from a local path or model name."""


class _GenerationConfigBuilder(Protocol):
    def __call__(
        self,
        *,
        max_new_tokens: int,
        stop_strings: str,
        use_cache: bool,
    ) -> object:
        """Build a Transformers generation config."""


class _TokenizerProtocol(Protocol):
    def decode(self, token_ids: object, *, skip_special_tokens: bool) -> str:
        """Decode generated token ids into text."""


class _TensorLike(Protocol):
    def to(self, device: object) -> _TensorLike:
        """Move a tensor-like value to the target device."""

    def unsqueeze(self, dim: int) -> _TensorLike:
        """Add a batch dimension."""

    def size(self, dim: int) -> int:
        """Return the tensor size for a dimension."""


class _GeneratedBatch(Protocol):
    def __getitem__(self, key: object) -> object:
        """Return generated tokens by tensor-style indexing."""


class _MolmoProcessor(Protocol):
    tokenizer: _TokenizerProtocol

    def process(
        self,
        *,
        images: Sequence[object],
        text: str,
    ) -> Mapping[str, _TensorLike]:
        """Build model inputs for Molmo generation."""


class _MolmoModel(Protocol):
    device: object
    _supports_default_dynamic_cache: Callable[[], bool]

    def to(self, device: object) -> object:
        """Move this model to a device."""

    def eval(self) -> object:
        """Switch the model to inference mode."""

    def generate_from_batch(
        self,
        inputs: Mapping[str, _TensorLike],
        generation_config: object,
        *,
        tokenizer: _TokenizerProtocol,
    ) -> _GeneratedBatch:
        """Generate output tokens from prepared Molmo inputs."""


class _TorchCuda(Protocol):
    def is_available(self) -> bool:
        """Return whether CUDA is available."""


class _ContextManagerLike(Protocol):
    def __enter__(self) -> object:
        """Enter the context manager."""

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool | None:
        """Exit the context manager."""


class _TorchModule(Protocol):
    bfloat16: object
    cuda: _TorchCuda

    def device(self, value: str) -> object:
        """Build a torch device object."""

    def inference_mode(self) -> _ContextManagerLike:
        """Disable autograd for generation."""

    def autocast(self, device_type: str, *, dtype: object) -> _ContextManagerLike:
        """Create an autocast context."""


class _ImageObject(Protocol):
    def __enter__(self) -> _ImageObject:
        """Return the open image object."""

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool | None:
        """Close the image context."""

    def convert(self, mode: str) -> _ImageObject:
        """Convert the image color mode."""


class _ImageModule(Protocol):
    def open(self, fp: Path) -> _ImageObject:
        """Open an image from disk."""


@dataclass(frozen=True)
class MolmoPatchReport:
    """One idempotent compatibility patch applied to local Molmo remote code."""

    file_path: str
    changed: bool
    operations: tuple[str, ...]


class ValidationIssuePayload(TypedDict):
    """JSON-ready details for one request validation issue."""

    field: str
    message: str


class TransformersMolmoRunner:
    """Molmo runner backed by Hugging Face Transformers.

    Heavy vision/model dependencies are imported only when this runner is
    instantiated or called, keeping package import paths lightweight.
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_path: Path,
        device: str,
    ) -> None:
        self.model_name = model_name
        self._device = device
        model_dir = model_path.expanduser().resolve()
        if not model_dir.is_dir():
            raise RuntimeError(
                f"Molmo model_path must be an existing directory: {model_dir}"
            )
        if not device.startswith("cuda"):
            raise RuntimeError(f"Molmo sidecar requires a CUDA device; got {device!r}")

        torch_module = cast(_TorchModule, importlib.import_module("torch"))
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is required for Molmo sidecar inference")
        transformers_module = importlib.import_module("transformers")
        auto_processor = cast(
            _FromPretrainedLoader,
            _require_module_attribute(transformers_module, "AutoProcessor"),
        )
        auto_model = cast(
            _FromPretrainedLoader,
            _require_module_attribute(transformers_module, "AutoModelForCausalLM"),
        )
        generation_config_type = cast(
            _GenerationConfigBuilder,
            _require_module_attribute(transformers_module, "GenerationConfig"),
        )
        self._torch = torch_module
        self._patch_reports = patch_molmo_remote_code(model_dir)

        self._processor = cast(
            _MolmoProcessor,
            auto_processor.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                local_files_only=True,
            ),
        )
        self._model = cast(
            _MolmoModel,
            auto_model.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                torch_dtype=torch_module.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=True,
            ),
        )
        self._model.to(torch_module.device(device))
        force_legacy_generation_cache(self._model)
        self._model.eval()
        self._generation_config = generation_config_type(
            max_new_tokens=MAX_NEW_TOKENS,
            stop_strings=MOLMO_STOP_STRING,
            use_cache=True,
        )

    def point(self, request: MolmoPointRequest) -> str:
        image_module = cast(_ImageModule, importlib.import_module("PIL.Image"))
        with image_module.open(request.image_path) as image:
            rgb_image = image.convert("RGB")
            raw_inputs = self._processor.process(
                images=[rgb_image],
                text=request.prompt,
            )
            model_inputs = _prepare_model_inputs(
                raw_inputs,
                device=self._model.device,
            )
            with (
                self._torch.inference_mode(),
                self._torch.autocast("cuda", dtype=self._torch.bfloat16),
            ):
                generated_batch = self._model.generate_from_batch(
                    model_inputs,
                    self._generation_config,
                    tokenizer=self._processor.tokenizer,
                )

        prompt_token_count = model_inputs["input_ids"].size(1)
        generated_tokens = generated_batch[0, prompt_token_count:]
        decoded = self._processor.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        if not isinstance(decoded, str):
            raise TypeError("Molmo tokenizer.decode returned a non-string result")
        return decoded


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the Molmo sidecar command line parser."""

    parser = argparse.ArgumentParser(
        description="Serve the SceneFunc3D Molmo pointing sidecar."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    return parser


def serve(runner: MolmoRunner, *, host: str, port: int) -> None:
    """Run the blocking Molmo sidecar HTTP server."""

    server = ThreadingHTTPServer((host, port), make_json_handler(_build_routes(runner)))
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _build_routes(runner: MolmoRunner) -> dict[str, JsonRoute]:
    return {
        "/health": lambda payload: _handle_health(runner),
        "/v1/point": lambda payload: _handle_point(runner, payload),
    }


def _handle_health(runner: MolmoRunner) -> JsonObject:
    response = HealthResponse(
        status="ok",
        model_name=runner.model_name,
        model_loaded=True,
    )
    return response.model_dump(mode="json")


def _handle_point(runner: MolmoRunner, payload: JsonObject) -> JsonObject:
    try:
        request = MolmoPointRequest.model_validate(payload)
    except ValidationError as exc:
        raise JsonHttpError(
            400,
            {
                "error": "invalid_request",
                "details": _validation_error_details(exc),
            },
        ) from exc
    start_time = time.perf_counter()
    try:
        raw_text = runner.point(request)
    except RuntimeError as exc:
        if _is_gpu_resource_error(exc):
            raise JsonHttpError(503, {"error": "molmo_resource_unavailable"}) from exc
        raise
    latency_ms = (time.perf_counter() - start_time) * 1000
    response = MolmoPointResponse(
        request_id=request.request_id,
        model_name=runner.model_name,
        raw_text=raw_text,
        latency_ms=latency_ms,
    )
    return response.model_dump(mode="json")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the Molmo point sidecar server."""

    args = build_arg_parser().parse_args(argv)
    runner = TransformersMolmoRunner(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
    )
    try:
        serve(runner, host=args.host, port=args.port)
    except KeyboardInterrupt:
        return 130
    return 0


def _require_module_attribute(module: ModuleType, attribute_name: str) -> object:
    return getattr(module, attribute_name)


def _prepare_model_inputs(
    raw_inputs: Mapping[str, _TensorLike],
    *,
    device: object,
) -> dict[str, _TensorLike]:
    return {name: tensor.to(device).unsqueeze(0) for name, tensor in raw_inputs.items()}


def patch_molmo_remote_code(model_dir: Path) -> tuple[MolmoPatchReport, ...]:
    """Apply local Molmo remote-code compatibility patches idempotently."""

    reports: list[MolmoPatchReport] = []
    image_processor = model_dir / "image_preprocessing_molmo.py"
    if image_processor.is_file():
        reports.append(_patch_molmo_image_processor(image_processor))
    modeling = model_dir / "modeling_molmo.py"
    if modeling.is_file():
        reports.append(_patch_molmo_modeling(modeling))
    return tuple(reports)


def force_legacy_generation_cache(model: _MolmoModel) -> None:
    """Keep Molmo remote code on tuple-style KV cache with newer transformers."""

    def uses_default_dynamic_cache() -> bool:
        return False

    model._supports_default_dynamic_cache = uses_default_dynamic_cache


def _patch_molmo_image_processor(path: Path) -> MolmoPatchReport:
    original_text = path.read_text(encoding="utf-8")
    updated_text = original_text
    operations: list[str] = []
    if 'tf = import_module("tensorflow")' not in updated_text:
        branch = '    if resize_method == "tensorflow":\n'
        if branch not in updated_text:
            raise RuntimeError(
                f"cannot locate tensorflow resize branch in Molmo remote code: {path}"
            )
        replacement = (
            branch
            + "        from importlib import import_module\n\n"
            + '        tf = import_module("tensorflow")\n'
        )
        updated_text = updated_text.replace(branch, replacement, 1)
        operations.append("added lazy tensorflow import")
    if "\nimport tensorflow as tf\n" in updated_text:
        updated_text = updated_text.replace("\nimport tensorflow as tf\n", "\n", 1)
        operations.append("removed eager tensorflow import")
    return _write_patch_report(path, original_text, updated_text, tuple(operations))


def _patch_molmo_modeling(path: Path) -> MolmoPatchReport:
    original_text = path.read_text(encoding="utf-8")
    updated_text = original_text
    operations: list[str] = []
    if "all_tied_weights_keys" not in updated_text:
        marker = '    _no_split_modules = ["MolmoBlock"]\n'
        if marker not in updated_text:
            raise RuntimeError(
                f"cannot locate Molmo model class anchor in remote code: {path}"
            )
        updated_text = updated_text.replace(
            marker, marker + "    all_tied_weights_keys = {}\n", 1
        )
        operations.append("added all_tied_weights_keys")
    if "def tie_weights(self):" in updated_text:
        updated_text = updated_text.replace(
            "def tie_weights(self):", "def tie_weights(self, *args, **kwargs):", 1
        )
        operations.append("allowed tie_weights compatibility args")
    old_cache_update = (
        '        if "cache_position" in model_kwargs:\n'
        '            model_kwargs["cache_position"] = '
        'model_kwargs["cache_position"][-1:] + num_new_tokens\n'
    )
    new_cache_update = (
        '        cache_position = model_kwargs.get("cache_position")\n'
        "        if cache_position is not None:\n"
        '            model_kwargs["cache_position"] = '
        "cache_position[-1:] + num_new_tokens\n"
    )
    if old_cache_update in updated_text:
        updated_text = updated_text.replace(old_cache_update, new_cache_update, 1)
        operations.append("guarded cache_position update")
    return _write_patch_report(path, original_text, updated_text, tuple(operations))


def _write_patch_report(
    path: Path,
    original_text: str,
    updated_text: str,
    operations: tuple[str, ...],
) -> MolmoPatchReport:
    if updated_text != original_text:
        path.write_text(updated_text, encoding="utf-8")
    return MolmoPatchReport(
        file_path=str(path),
        changed=updated_text != original_text,
        operations=operations,
    )


def _validation_error_details(exc: ValidationError) -> list[ValidationIssuePayload]:
    details: list[ValidationIssuePayload] = []
    for error in exc.errors():
        field_name = ".".join(str(part) for part in error.get("loc", ())) or "(root)"
        details.append(
            {
                "field": field_name,
                "message": str(error.get("msg", "invalid")),
            }
        )
    return details


def _is_gpu_resource_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _CUDA_OOM_MARKERS)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MolmoRunner",
    "MolmoPatchReport",
    "TransformersMolmoRunner",
    "build_arg_parser",
    "force_legacy_generation_cache",
    "main",
    "patch_molmo_remote_code",
    "serve",
]
