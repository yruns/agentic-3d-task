"""Molmo point sidecar server for SceneFunc3D agent tools."""

from __future__ import annotations

import argparse
import html
import importlib
import math
import re
import time
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Literal, Protocol, TypedDict, cast, runtime_checkable

from pydantic import ValidationError

from codex_agent.scenefunc3d.backends.image_payload import materialize_inline_image
from codex_agent.scenefunc3d.servers.http_json import (
    JsonHttpError,
    JsonObject,
    JsonRoute,
    make_json_handler,
)
from codex_agent.scenefunc3d.servers.schemas import (
    HealthResponse,
    MolmoImagePoint,
    MolmoPointRequest,
    MolmoPointResponse,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8711
DEFAULT_MODEL_NAME = "MolmoPoint-8B"
DEFAULT_DEVICE = "cuda:0"
MAX_NEW_TOKENS = 200
MOLMOPOINT_TOKEN_POOLING_KEY = "image_token_pooling_np"
MOLMOPOINT_SUBPATCH_MAPPING_KEY = "subpatch_mapping"
MOLMOPOINT_IMAGE_SIZES_KEY = "image_sizes"
MOLMOPOINT_LEGACY_METADATA_KEY = "metadata"
MOLMOPOINT_LEGACY_TOKEN_POOLING_KEY = "token_pooling"
_MOLMOPOINT_LABEL_RE = re.compile(
    r"<points?\b(?:[^'\">]|\"[^\"]*\"|'[^']*')*>(?P<label_text>.*?)</point>",
    flags=re.IGNORECASE | re.DOTALL,
)

_CUDA_OOM_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
)


class MolmoRunner(Protocol):
    """Inference boundary used by the Molmo HTTP route."""

    model_name: str

    def point(self, request: MolmoPointRequest) -> MolmoRunnerPointResult:
        """Return raw Molmo text and parsed image points for one request."""


class _FromPretrainedLoader(Protocol):
    def from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        **kwargs: object,
    ) -> object:
        """Load an object from a local path or model name."""


@runtime_checkable
class _DeviceMovable(Protocol):
    def to(self, device: object) -> object:
        """Move a tensor-like value to the target device."""


@runtime_checkable
class _TokenIdsLike(Protocol):
    def size(self, dim: int) -> int:
        """Return the tensor size for a dimension."""


class _GeneratedTokens(Protocol):
    def __getitem__(self, key: object) -> object:
        """Return generated tokens by tensor-style indexing."""


class _MolmoPointTextContent(TypedDict):
    type: Literal["text"]
    text: str


class _MolmoPointImageContent(TypedDict):
    type: Literal["image"]
    image: object


class _MolmoPointMessage(TypedDict):
    role: Literal["user"]
    content: list[_MolmoPointTextContent | _MolmoPointImageContent]


class _MolmoPointProcessor(Protocol):
    def apply_chat_template(
        self,
        messages: Sequence[_MolmoPointMessage],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
        return_dict: bool,
        padding: bool,
        return_pointing_metadata: bool,
    ) -> MutableMapping[str, object]:
        """Build MolmoPoint model inputs and pointing metadata."""

    def post_process_image_text_to_text(
        self,
        generated_tokens: object,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> Sequence[str]:
        """Decode generated token ids into generated text."""


class _MolmoPointModel(Protocol):
    def eval(self) -> object:
        """Switch the model to inference mode."""

    def build_logit_processor_from_inputs(
        self,
        model_inputs: Mapping[str, object],
    ) -> object:
        """Build MolmoPoint constrained decoding processors."""

    def generate(self, **kwargs: object) -> _GeneratedTokens:
        """Generate output tokens from prepared MolmoPoint inputs."""

    def extract_image_points(
        self,
        generated_text: str,
        token_pooling: object,
        subpatch_mapping: object,
        image_sizes: object,
    ) -> object:
        """Validate generated pointing text against MolmoPoint image metadata."""


class _LegacyMolmoModel(Protocol):
    _supports_default_dynamic_cache: object

    def to(self, device: object) -> object:
        """Move this model to a device."""


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


@dataclass(frozen=True)
class MolmoPointMetadata:
    """MolmoPoint pointing metadata needed for image-point sanity checking."""

    token_pooling: object
    subpatch_mapping: object
    image_sizes: object


@dataclass(frozen=True)
class MolmoPointGenerationInputs:
    """Separated model inputs and non-model pointing metadata."""

    model_inputs: dict[str, object]
    metadata: MolmoPointMetadata


@dataclass(frozen=True)
class MolmoRunnerPointResult:
    """Molmo sidecar result before HTTP response serialization."""

    raw_text: str
    image_points: tuple[MolmoImagePoint, ...]


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
            _require_module_attribute(
                transformers_module,
                "AutoModelForImageTextToText",
            ),
        )
        self._torch = torch_module
        self._device_value = torch_module.device(device)

        self._processor = cast(
            _MolmoPointProcessor,
            auto_processor.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                padding_side="left",
                local_files_only=True,
            ),
        )
        self._model = cast(
            _MolmoPointModel,
            auto_model.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                torch_dtype=torch_module.bfloat16,
                device_map="auto",
                local_files_only=True,
            ),
        )
        self._model.eval()

    def point(self, request: MolmoPointRequest) -> MolmoRunnerPointResult:
        image_module = cast(_ImageModule, importlib.import_module("PIL.Image"))
        with image_module.open(request.require_image_path()) as image:
            rgb_image = image.convert("RGB")
            raw_inputs = self._processor.apply_chat_template(
                _build_molmopoint_messages(rgb_image, request.prompt),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                return_pointing_metadata=True,
            )
            generation_inputs = _prepare_molmopoint_generation_inputs(
                raw_inputs,
                device=self._device_value,
            )
            with (
                self._torch.inference_mode(),
                self._torch.autocast("cuda", dtype=self._torch.bfloat16),
            ):
                output = self._model.generate(
                    **generation_inputs.model_inputs,
                    logits_processor=self._model.build_logit_processor_from_inputs(
                        generation_inputs.model_inputs
                    ),
                    max_new_tokens=MAX_NEW_TOKENS,
                )

        input_ids = _require_token_ids(generation_inputs.model_inputs)
        prompt_token_count = input_ids.size(1)
        generated_tokens = output[:, prompt_token_count:]
        generated_texts = self._processor.post_process_image_text_to_text(
            generated_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        generated_text = _first_generated_text(generated_texts)
        image_points = _extract_generated_points(
            self._model,
            generated_text,
            generation_inputs.metadata,
            fallback_label=request.prompt,
            image_width=request.image_width,
            image_height=request.image_height,
        )
        return MolmoRunnerPointResult(
            raw_text=generated_text, image_points=image_points
        )


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
        point_result = run_molmo_point_request(runner, request)
    except RuntimeError as exc:
        if _is_gpu_resource_error(exc):
            raise JsonHttpError(503, {"error": "molmo_resource_unavailable"}) from exc
        raise
    latency_ms = (time.perf_counter() - start_time) * 1000
    response = MolmoPointResponse(
        request_id=request.request_id,
        model_name=runner.model_name,
        raw_text=point_result.raw_text,
        image_points=point_result.image_points,
        latency_ms=latency_ms,
    )
    return response.model_dump(mode="json")


def run_molmo_point_request(
    runner: MolmoRunner,
    request: MolmoPointRequest,
) -> MolmoRunnerPointResult:
    """Run Molmo with a path-backed request, materializing inline images briefly."""
    if request.image_source == "path":
        return runner.point(request)

    with TemporaryDirectory(prefix="scenefunc3d-molmo-image-") as temporary_dir:
        image_path = materialize_inline_image(
            request.require_inline_image(),
            parent_dir=Path(temporary_dir),
        )
        return runner.point(request.with_image_path(image_path))


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


def _build_molmopoint_messages(
    image: object,
    prompt: str,
) -> list[_MolmoPointMessage]:
    message: _MolmoPointMessage = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image},
        ],
    }
    return [message]


def _prepare_molmopoint_generation_inputs(
    raw_inputs: MutableMapping[str, object],
    *,
    device: object,
) -> MolmoPointGenerationInputs:
    metadata = _extract_molmopoint_metadata(raw_inputs)
    model_inputs = {
        name: _move_input_to_device(value, device) for name, value in raw_inputs.items()
    }
    return MolmoPointGenerationInputs(model_inputs=model_inputs, metadata=metadata)


def _extract_molmopoint_metadata(
    raw_inputs: MutableMapping[str, object],
) -> MolmoPointMetadata:
    legacy_metadata = raw_inputs.pop(MOLMOPOINT_LEGACY_METADATA_KEY, None)
    if MOLMOPOINT_TOKEN_POOLING_KEY in raw_inputs:
        return MolmoPointMetadata(
            token_pooling=_pop_required_value(raw_inputs, MOLMOPOINT_TOKEN_POOLING_KEY),
            subpatch_mapping=_pop_required_value(
                raw_inputs,
                MOLMOPOINT_SUBPATCH_MAPPING_KEY,
            ),
            image_sizes=_pop_required_value(raw_inputs, MOLMOPOINT_IMAGE_SIZES_KEY),
        )

    if legacy_metadata is None:
        raise RuntimeError(
            "MolmoPoint processor metadata missing field: "
            f"{MOLMOPOINT_TOKEN_POOLING_KEY}"
        )
    if not isinstance(legacy_metadata, Mapping):
        raise TypeError("MolmoPoint processor metadata must be a mapping")
    return MolmoPointMetadata(
        token_pooling=_require_mapping_value(
            legacy_metadata,
            MOLMOPOINT_LEGACY_TOKEN_POOLING_KEY,
        ),
        subpatch_mapping=_require_mapping_value(
            legacy_metadata,
            MOLMOPOINT_SUBPATCH_MAPPING_KEY,
        ),
        image_sizes=_require_mapping_value(legacy_metadata, MOLMOPOINT_IMAGE_SIZES_KEY),
    )


def _pop_required_value(raw_inputs: MutableMapping[str, object], key: str) -> object:
    if key not in raw_inputs:
        raise RuntimeError(f"MolmoPoint processor metadata missing field: {key}")
    value = raw_inputs.pop(key)
    if value is None:
        raise TypeError(f"MolmoPoint processor metadata field cannot be None: {key}")
    return value


def _require_mapping_value(metadata: Mapping[object, object], key: str) -> object:
    if key not in metadata:
        raise RuntimeError(f"MolmoPoint processor metadata missing field: {key}")
    value = metadata[key]
    if value is None:
        raise TypeError(f"MolmoPoint processor metadata field cannot be None: {key}")
    return value


def _move_input_to_device(value: object, device: object) -> object:
    if isinstance(value, _DeviceMovable):
        return value.to(device)
    return value


def _require_token_ids(model_inputs: Mapping[str, object]) -> _TokenIdsLike:
    input_ids = model_inputs.get("input_ids")
    if not isinstance(input_ids, _TokenIdsLike):
        raise TypeError("MolmoPoint processor did not return sized input_ids")
    return input_ids


def _first_generated_text(generated_texts: Sequence[str]) -> str:
    if len(generated_texts) == 0:
        raise RuntimeError("MolmoPoint processor returned no generated text")
    generated_text = generated_texts[0]
    if not isinstance(generated_text, str):
        raise TypeError("MolmoPoint processor returned a non-string generated text")
    return generated_text


def _extract_generated_points(
    model: _MolmoPointModel,
    generated_text: str,
    metadata: MolmoPointMetadata,
    *,
    fallback_label: str,
    image_width: int,
    image_height: int,
) -> tuple[MolmoImagePoint, ...]:
    try:
        raw_points = model.extract_image_points(
            generated_text,
            metadata.token_pooling,
            metadata.subpatch_mapping,
            metadata.image_sizes,
        )
    except Exception as exc:
        raise RuntimeError(
            "MolmoPoint generated text failed image-point extraction sanity check"
        ) from exc
    return _parse_extracted_image_points(
        raw_points,
        source_text=generated_text,
        label=_extract_point_label(generated_text, fallback_label=fallback_label),
        image_width=image_width,
        image_height=image_height,
    )


def _parse_extracted_image_points(
    raw_points: object,
    *,
    source_text: str,
    label: str,
    image_width: int,
    image_height: int,
) -> tuple[MolmoImagePoint, ...]:
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, str | bytes):
        raise RuntimeError("MolmoPoint extracted image points must be a sequence")
    image_points: list[MolmoImagePoint] = []
    for raw_point in raw_points:
        image_points.append(
            _parse_extracted_image_point(
                raw_point,
                source_text=source_text,
                label=label,
                image_width=image_width,
                image_height=image_height,
            )
        )
    return tuple(image_points)


def _parse_extracted_image_point(
    raw_point: object,
    *,
    source_text: str,
    label: str,
    image_width: int,
    image_height: int,
) -> MolmoImagePoint:
    if not isinstance(raw_point, Sequence) or isinstance(raw_point, str | bytes):
        raise RuntimeError("MolmoPoint extracted image point must be a sequence")
    if len(raw_point) < 4:
        raise RuntimeError(
            "MolmoPoint extracted image point must include object, image, x, and y"
        )
    return MolmoImagePoint(
        x_px=_require_bounded_pixel_coordinate(
            raw_point[2],
            field_name="x_px",
            upper_bound=image_width,
        ),
        y_px=_require_bounded_pixel_coordinate(
            raw_point[3],
            field_name="y_px",
            upper_bound=image_height,
        ),
        source=source_text,
        label=label,
    )


def _require_bounded_pixel_coordinate(
    value: object,
    *,
    field_name: str,
    upper_bound: int,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(f"MolmoPoint {field_name} must be a real number")
    float_value = float(value)
    if not math.isfinite(float_value) or float_value < 0.0:
        raise RuntimeError(f"MolmoPoint {field_name} must be finite and nonnegative")
    if float_value >= float(upper_bound):
        raise RuntimeError(
            f"MolmoPoint {field_name} must be inside the request image bounds"
        )
    return float_value


def _extract_point_label(generated_text: str, *, fallback_label: str) -> str:
    match = _MOLMOPOINT_LABEL_RE.search(generated_text)
    if match is not None:
        label = html.unescape(match.group("label_text")).strip()
        if label:
            return label
    return fallback_label.strip()


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


def force_legacy_generation_cache(model: _LegacyMolmoModel) -> None:
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
    "run_molmo_point_request",
    "serve",
]
