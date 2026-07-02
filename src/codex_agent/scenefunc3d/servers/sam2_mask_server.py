"""SAM2 mask sidecar server for SceneFunc3D agent tools."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import time
from collections.abc import Iterable, Sequence
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import ModuleType
from typing import (
    Literal,
    Protocol,
    TypedDict,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from codex_agent.scenefunc3d.backends.image_payload import materialize_inline_image
from codex_agent.scenefunc3d.backends.mask_codec import encode_bool_mask_rle
from codex_agent.scenefunc3d.servers.http_json import (
    JsonHttpError,
    JsonObject,
    JsonRoute,
    make_json_handler,
)
from codex_agent.scenefunc3d.servers.schemas import (
    HealthResponse,
    SamMaskCandidateResponse,
    SamMaskRequest,
    SamMaskResponse,
    SamMaskRle,
    SamPointPrompt,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8712
DEFAULT_MODEL_NAME = "SAM2.1-Hiera-L"
DEFAULT_DEVICE = "cuda:0"
Sam2Backend = Literal["official", "transformers"]
_ArgparseNamespace = TypeVar("_ArgparseNamespace")
_TRANSFORMERS_SAM2_MODEL_INPUT_KEYS = frozenset(
    (
        "pixel_values",
        "input_points",
        "input_labels",
        "input_boxes",
        "input_masks",
        "image_embeddings",
    )
)

_CUDA_RESOURCE_ERROR_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cuda error",
    "resource exhausted",
)


class Sam2Runner(Protocol):
    """Inference boundary used by the SAM2 HTTP route."""

    model_name: str

    def masks(self, request: SamMaskRequest) -> tuple[SamMaskCandidateResponse, ...]:
        """Return candidate mask metadata for one SAM2 request."""


class _TorchCuda(Protocol):
    def is_available(self) -> bool:
        """Return whether CUDA is available."""


class _TorchInferenceMode(Protocol):
    def __enter__(self) -> object:
        """Enter inference-only tensor mode."""

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool | None:
        """Exit inference-only tensor mode."""


class _TorchModule(Protocol):
    cuda: _TorchCuda
    bfloat16: object

    def inference_mode(self) -> _TorchInferenceMode:
        """Return the torch inference-mode context manager."""


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


class _BuildSam2Function(Protocol):
    def __call__(
        self,
        config_file: str,
        ckpt_path: str,
        *,
        device: str,
    ) -> object:
        """Build a SAM2 model object."""


class _Sam2ImagePredictor(Protocol):
    def set_image(self, image: NDArray[np.uint8]) -> None:
        """Set the current RGB image for subsequent SAM2 prediction."""

    def predict(
        self,
        *,
        point_coords: NDArray[np.float32],
        point_labels: NDArray[np.int32],
        multimask_output: bool,
    ) -> tuple[object, object, object] | tuple[object, object]:
        """Predict masks and scores from point prompts."""


class _Sam2ImagePredictorFactory(Protocol):
    def __call__(self, sam_model: object) -> _Sam2ImagePredictor:
        """Build a SAM2 image predictor for a loaded model."""


@runtime_checkable
class _TensorLike(Protocol):
    def detach(self) -> _TensorLike:
        """Detach the tensor from any autograd graph."""

    def cpu(self) -> _TensorLike:
        """Return a CPU-backed representation."""

    def float(self) -> _TensorLike:
        """Return a float32 representation."""

    def __getitem__(self, key: int) -> _TensorLike:
        """Return an indexed tensor-like value."""


@runtime_checkable
class _DTypeMovable(Protocol):
    def to(
        self,
        device: str | None = None,
        *,
        dtype: object | None = None,
    ) -> object:
        """Move or cast a tensor-like value."""


class _ProcessorInputs(Protocol):
    def to(self, device: str) -> _ProcessorInputs:
        """Move processor inputs to a device."""

    def __getitem__(self, key: str) -> object:
        """Return one processor input by key."""

    def keys(self) -> Iterable[str]:
        """Return processor input keys."""


class _TransformersSam2Output(Protocol):
    pred_masks: _TensorLike
    iou_scores: _TensorLike


class _TransformersSam2Model(Protocol):
    def to(self, device: str) -> _TransformersSam2Model:
        """Move the model to a device."""

    def eval(self) -> object:
        """Put the model in inference mode."""

    def __call__(self, **model_inputs: object) -> _TransformersSam2Output:
        """Run SAM2 inference."""


class _Sam2ModelLoader(Protocol):
    def from_pretrained(
        self,
        model_reference: str,
        *,
        torch_dtype: object,
    ) -> _TransformersSam2Model:
        """Load a Transformers SAM2 model."""


class _TransformersSam2Processor(Protocol):
    def __call__(
        self,
        *,
        images: _ImageObject,
        input_points: list[list[list[list[float]]]],
        input_labels: list[list[list[int]]],
        return_tensors: str,
    ) -> _ProcessorInputs:
        """Build model inputs for SAM2 point prompts."""

    def post_process_masks(
        self,
        pred_masks: object,
        original_sizes: object,
    ) -> Sequence[object]:
        """Resize model masks back to the input image sizes."""


class _Sam2ProcessorLoader(Protocol):
    def from_pretrained(self, model_reference: str) -> _TransformersSam2Processor:
        """Load a Transformers SAM2 processor."""


class _TransformersModule(Protocol):
    Sam2Model: _Sam2ModelLoader
    Sam2Processor: _Sam2ProcessorLoader


class ValidationIssuePayload(TypedDict):
    """JSON-ready details for one request validation issue."""

    field: str
    message: str


class SamStagingPathError(RuntimeError):
    """Raised when a SAM2 request tries to write outside the staging root."""


class SamImageLoadError(RuntimeError):
    """Raised when the request image cannot be loaded as RGB pixels."""


class SamInvalidOutputError(RuntimeError):
    """Raised when SAM2 produces invalid masks or scores."""


class SamArtifactWriteError(RuntimeError):
    """Raised when SAM2 mask artifacts cannot be written."""


class _Sam2ArgumentParser(argparse.ArgumentParser):
    @overload
    def parse_args(
        self,
        args: Iterable[str] | None = None,
        namespace: None = None,
    ) -> argparse.Namespace: ...

    @overload
    def parse_args(
        self,
        args: Iterable[str] | None,
        namespace: _ArgparseNamespace,
    ) -> _ArgparseNamespace: ...

    @overload
    def parse_args(
        self,
        *,
        namespace: _ArgparseNamespace,
    ) -> _ArgparseNamespace: ...

    def parse_args(
        self,
        args: Iterable[str] | None = None,
        namespace: _ArgparseNamespace | None = None,
    ) -> argparse.Namespace | _ArgparseNamespace:
        parsed_args = super().parse_args(args, namespace)
        _validate_cli_args(self, cast(argparse.Namespace, parsed_args))
        return parsed_args


class OfficialSam2Runner:
    """SAM2 runner backed by the official facebookresearch/sam2 package.

    Heavy vision/model dependencies are imported only when this runner is
    instantiated or called, keeping package import paths lightweight.
    """

    def __init__(
        self,
        *,
        model_name: str,
        checkpoint_path: Path,
        config_path: Path,
        staging_root: Path,
        device: str,
    ) -> None:
        self.model_name = model_name
        self._checkpoint_path = _require_existing_file(
            checkpoint_path,
            field_name="checkpoint_path",
        )
        self._config_path = _require_existing_file(
            config_path,
            field_name="config_path",
        )
        self._staging_root = _prepare_staging_root(staging_root)
        if not device.startswith("cuda"):
            raise RuntimeError(f"SAM2 sidecar requires a CUDA device; got {device!r}")

        torch_module = cast(_TorchModule, importlib.import_module("torch"))
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is required for SAM2 sidecar inference")

        build_sam_module = importlib.import_module("sam2.build_sam")
        predictor_module = importlib.import_module("sam2.sam2_image_predictor")
        build_sam2 = cast(
            _BuildSam2Function,
            _require_module_attribute(build_sam_module, "build_sam2"),
        )
        predictor_factory = cast(
            _Sam2ImagePredictorFactory,
            _require_module_attribute(predictor_module, "SAM2ImagePredictor"),
        )
        model = build_sam2(
            str(self._config_path),
            str(self._checkpoint_path),
            device=device,
        )
        self._predictor = predictor_factory(model)
        self._prediction_lock = Lock()

    def masks(self, request: SamMaskRequest) -> tuple[SamMaskCandidateResponse, ...]:
        staging_dir = _resolve_staging_dir(
            request.require_staging_dir(),
            staging_root=self._staging_root,
        )
        try:
            image_module = cast(_ImageModule, importlib.import_module("PIL.Image"))
            with image_module.open(request.require_image_path()) as image:
                rgb_image = image.convert("RGB")
                image_array = _image_to_rgb_array(rgb_image)
        except SamImageLoadError:
            raise
        except ImportError as exc:
            raise SamImageLoadError(
                "PIL.Image is required to load SAM2 request images"
            ) from exc
        except (OSError, ValueError) as exc:
            raise SamImageLoadError(
                "could not load SAM2 request image: "
                f"image_path={request.image_path}; error_type={exc.__class__.__name__}"
            ) from exc

        point_coords = _build_point_coords(request.points)
        point_labels = np.ones(len(request.points), dtype=np.int32)
        with self._prediction_lock:
            self._predictor.set_image(image_array)
            predict_output = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        raw_masks, raw_scores = _split_predict_output(predict_output)
        mask_batch = _validate_mask_batch(
            raw_masks,
            image_height=image_array.shape[0],
            image_width=image_array.shape[1],
        )
        scores = _validate_scores(
            raw_scores,
            candidate_count=mask_batch.shape[0],
        )
        try:
            staging_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SamArtifactWriteError(
                "could not create SAM2 staging directory: "
                f"staging_dir={staging_dir}; error_type={exc.__class__.__name__}"
            ) from exc
        return _write_candidate_masks(
            mask_batch,
            scores,
            staging_dir=staging_dir,
        )


class TransformersSam2Runner:
    """SAM2 runner backed by Hugging Face Transformers.

    This backend accepts either a model id or a local HF snapshot directory and
    uses the same output validation/writer path as the official SAM2 backend.
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_reference: str,
        staging_root: Path,
        device: str,
    ) -> None:
        self.model_name = model_name
        if not model_reference:
            raise RuntimeError("model_reference must be a non-empty model id or path")
        self._staging_root = _prepare_staging_root(staging_root)
        self._device = device
        if not device.startswith("cuda"):
            raise RuntimeError(f"SAM2 sidecar requires a CUDA device; got {device!r}")

        torch_module = cast(_TorchModule, importlib.import_module("torch"))
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is required for SAM2 sidecar inference")
        self._torch_module = torch_module

        transformers_module = cast(
            _TransformersModule,
            importlib.import_module("transformers"),
        )
        self._model = transformers_module.Sam2Model.from_pretrained(
            model_reference,
            torch_dtype=torch_module.bfloat16,
        ).to(device)
        self._model.eval()
        self._processor = transformers_module.Sam2Processor.from_pretrained(
            model_reference
        )
        self._prediction_lock = Lock()

    def masks(self, request: SamMaskRequest) -> tuple[SamMaskCandidateResponse, ...]:
        staging_dir = _resolve_staging_dir(
            request.require_staging_dir(),
            staging_root=self._staging_root,
        )
        try:
            image_module = cast(_ImageModule, importlib.import_module("PIL.Image"))
            with image_module.open(request.require_image_path()) as image:
                rgb_image = image.convert("RGB")
                image_array = _image_to_rgb_array(rgb_image)
        except SamImageLoadError:
            raise
        except ImportError as exc:
            raise SamImageLoadError(
                "PIL.Image is required to load SAM2 request images"
            ) from exc
        except (OSError, ValueError) as exc:
            raise SamImageLoadError(
                "could not load SAM2 request image: "
                f"image_path={request.image_path}; error_type={exc.__class__.__name__}"
            ) from exc

        input_points = _build_transformers_input_points(request.points)
        input_labels = _build_transformers_input_labels(request.points)
        with self._prediction_lock:
            processor_inputs = self._processor(
                images=rgb_image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            ).to(self._device)
            model_inputs = _cast_transformers_pixel_values(
                _filter_transformers_model_inputs(processor_inputs),
                dtype=self._torch_module.bfloat16,
            )
            with self._torch_module.inference_mode():
                outputs = self._model(**model_inputs)
            post_processed_masks = self._processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                _require_tensor_input(processor_inputs, "original_sizes")
                .detach()
                .cpu(),
            )
            raw_scores = _extract_single_prompt_scores(
                outputs.iou_scores.detach().float().cpu()[0]
            )

        raw_masks = _extract_single_prompt_masks(post_processed_masks)
        mask_batch = _validate_mask_batch(
            raw_masks,
            image_height=image_array.shape[0],
            image_width=image_array.shape[1],
        )
        scores = _validate_scores(
            raw_scores,
            candidate_count=mask_batch.shape[0],
        )
        try:
            staging_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SamArtifactWriteError(
                "could not create SAM2 staging directory: "
                f"staging_dir={staging_dir}; error_type={exc.__class__.__name__}"
            ) from exc
        return _write_candidate_masks(
            mask_batch,
            scores,
            staging_dir=staging_dir,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the SAM2 sidecar command line parser."""

    parser = _Sam2ArgumentParser(description="Serve the SceneFunc3D SAM2 mask sidecar.")
    parser.add_argument(
        "--backend",
        choices=("official", "transformers"),
        default="official",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    return parser


def serve(runner: Sam2Runner, *, host: str, port: int) -> None:
    """Run the blocking SAM2 sidecar HTTP server."""

    server = ThreadingHTTPServer((host, port), make_json_handler(_build_routes(runner)))
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _build_routes(runner: Sam2Runner) -> dict[str, JsonRoute]:
    return {
        "/health": lambda payload: _handle_health(runner),
        "/v1/masks": lambda payload: _handle_masks(runner, payload),
    }


def _handle_health(runner: Sam2Runner) -> JsonObject:
    response = HealthResponse(
        status="ok",
        model_name=runner.model_name,
        model_loaded=True,
    )
    return cast(JsonObject, response.model_dump(mode="json"))


def _handle_masks(runner: Sam2Runner, payload: JsonObject) -> JsonObject:
    try:
        request = SamMaskRequest.model_validate(payload)
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
        candidates = run_sam_mask_request(runner, request)
    except SamStagingPathError as exc:
        raise JsonHttpError(
            400,
            {
                "error": "invalid_staging_dir",
                "request_id": request.request_id,
            },
        ) from exc
    except SamImageLoadError as exc:
        raise JsonHttpError(
            400,
            {
                "error": "sam_image_load_failed",
                "request_id": request.request_id,
            },
        ) from exc
    except SamInvalidOutputError as exc:
        raise JsonHttpError(
            500,
            {
                "error": "sam_invalid_output",
                "request_id": request.request_id,
            },
        ) from exc
    except SamArtifactWriteError as exc:
        raise JsonHttpError(
            500,
            {
                "error": "sam_artifact_write_failed",
                "request_id": request.request_id,
            },
        ) from exc
    except RuntimeError as exc:
        if _is_gpu_resource_error(exc):
            raise JsonHttpError(503, {"error": "sam_resource_unavailable"}) from exc
        raise
    latency_ms = (time.perf_counter() - start_time) * 1000
    response = SamMaskResponse(
        request_id=request.request_id,
        model_name=runner.model_name,
        candidates=candidates,
        latency_ms=latency_ms,
    )
    return cast(JsonObject, response.model_dump(mode="json", exclude_none=True))


def run_sam_mask_request(
    runner: Sam2Runner,
    request: SamMaskRequest,
) -> tuple[SamMaskCandidateResponse, ...]:
    """Run SAM with path-backed inputs, using temporary files for inline requests."""
    if request.image_source == "path" and request.staging_dir is not None:
        return _response_candidates_by_value(runner.masks(request))

    with TemporaryDirectory(
        prefix="scenefunc3d-sam-request-",
        dir=_runner_temporary_root_parent(runner),
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        path_request = request
        if request.image_source == "inline":
            image_path = materialize_inline_image(
                request.require_inline_image(),
                parent_dir=temporary_root / "image",
            )
            path_request = request.with_image_path(image_path)
        if path_request.staging_dir is None:
            path_request = path_request.with_staging_dir(temporary_root / "staging")
        return _response_candidates_by_value(runner.masks(path_request))


def _runner_temporary_root_parent(runner: Sam2Runner) -> Path | None:
    staging_root = getattr(runner, "_staging_root", None)
    if isinstance(staging_root, Path):
        return staging_root
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the SAM2 mask sidecar server."""

    args = build_arg_parser().parse_args(argv)
    try:
        runner = _build_runner_from_cli_args(args)
        serve(runner, host=args.host, port=args.port)
    except KeyboardInterrupt:
        return 130
    return 0


def _build_runner_from_cli_args(args: argparse.Namespace) -> Sam2Runner:
    backend = _namespace_backend(args)
    if backend == "official":
        return OfficialSam2Runner(
            model_name=_namespace_str(args, "model_name"),
            checkpoint_path=_namespace_required_path(args, "checkpoint_path"),
            config_path=_namespace_required_path(args, "config_path"),
            staging_root=_namespace_required_path(args, "staging_root"),
            device=_namespace_str(args, "device"),
        )
    return TransformersSam2Runner(
        model_name=_namespace_str(args, "model_name"),
        model_reference=_transformers_model_reference_from_cli_args(args),
        staging_root=_namespace_required_path(args, "staging_root"),
        device=_namespace_str(args, "device"),
    )


def _validate_cli_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    backend = _namespace_backend(args)
    checkpoint_path = _namespace_optional_path(args, "checkpoint_path")
    config_path = _namespace_optional_path(args, "config_path")
    model_path = _namespace_optional_path(args, "model_path")
    model_id = _namespace_trimmed_str(args, "model_id")

    if backend == "official":
        if checkpoint_path is None or config_path is None:
            parser.error(
                "--backend official requires --checkpoint-path and --config-path"
            )
        return

    has_model_path = model_path is not None
    has_model_id = bool(model_id)
    if has_model_path == has_model_id:
        parser.error(
            "--backend transformers requires exactly one of --model-path or --model-id"
        )


def _namespace_backend(args: argparse.Namespace) -> Sam2Backend:
    backend: object = args.backend
    if backend == "official" or backend == "transformers":
        return backend
    raise RuntimeError(f"unsupported SAM2 backend: {backend!r}")


def _namespace_str(args: argparse.Namespace, name: str) -> str:
    value: object = getattr(args, name)
    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be a string: {value!r}")
    return value


def _namespace_trimmed_str(args: argparse.Namespace, name: str) -> str:
    return _namespace_str(args, name).strip()


def _namespace_required_path(args: argparse.Namespace, name: str) -> Path:
    value = _namespace_optional_path(args, name)
    if value is None:
        raise RuntimeError(f"{name} is required")
    return value


def _namespace_optional_path(args: argparse.Namespace, name: str) -> Path | None:
    value: object = getattr(args, name)
    if value is None:
        return None
    if not isinstance(value, Path):
        raise RuntimeError(f"{name} must be a path: {value!r}")
    return value


def _transformers_model_reference_from_cli_args(args: argparse.Namespace) -> str:
    model_path = _namespace_optional_path(args, "model_path")
    if model_path is not None:
        return _resolve_transformers_model_path(model_path)
    return _namespace_trimmed_str(args, "model_id")


def _require_existing_file(path: Path, *, field_name: str) -> Path:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise RuntimeError(f"{field_name} must be an existing file: {resolved_path}")
    return resolved_path


def _resolve_transformers_model_path(path: Path) -> str:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise RuntimeError(f"model_path must exist: {resolved_path}")
    if resolved_path.is_file():
        if resolved_path.name != "model.safetensors":
            raise RuntimeError(
                "model_path file must be model.safetensors or a HF snapshot directory: "
                f"{resolved_path}"
            )
        return str(resolved_path.parent)
    if not resolved_path.is_dir():
        raise RuntimeError(f"model_path must be a file or directory: {resolved_path}")
    return str(resolved_path)


def _prepare_staging_root(path: Path) -> Path:
    resolved_path = path.expanduser().resolve()
    try:
        resolved_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            "staging_root must be creatable: "
            f"path={resolved_path}; error_type={exc.__class__.__name__}"
        ) from exc
    if not resolved_path.is_dir():
        raise RuntimeError(f"staging_root must be a directory: {resolved_path}")
    return resolved_path


def _resolve_staging_dir(path: Path, *, staging_root: Path) -> Path:
    resolved_path = path.expanduser().resolve()
    try:
        resolved_path.relative_to(staging_root)
    except ValueError as exc:
        raise SamStagingPathError(
            "SAM2 staging_dir is outside configured staging root: "
            f"staging_dir={resolved_path}; staging_root={staging_root}"
        ) from exc
    return resolved_path


def _require_module_attribute(module: ModuleType, attribute_name: str) -> object:
    return getattr(module, attribute_name)


def _image_to_rgb_array(image: _ImageObject) -> NDArray[np.uint8]:
    image_array = np.asarray(image, dtype=np.uint8)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise SamImageLoadError(
            "SAM2 input image must convert to an RGB array with shape HxWx3"
        )
    return cast(NDArray[np.uint8], image_array)


def _build_point_coords(
    points: tuple[SamPointPrompt, ...],
) -> NDArray[np.float32]:
    return np.array(
        [[point.x_px, point.y_px] for point in points],
        dtype=np.float32,
    )


def _build_transformers_input_points(
    points: tuple[SamPointPrompt, ...],
) -> list[list[list[list[float]]]]:
    return [[[[float(point.x_px), float(point.y_px)] for point in points]]]


def _build_transformers_input_labels(
    points: tuple[SamPointPrompt, ...],
) -> list[list[list[int]]]:
    return [[[1 for _point in points]]]


def _filter_transformers_model_inputs(
    processor_inputs: _ProcessorInputs,
) -> dict[str, object]:
    model_inputs: dict[str, object] = {}
    for key in processor_inputs.keys():
        if key in _TRANSFORMERS_SAM2_MODEL_INPUT_KEYS:
            model_inputs[key] = processor_inputs[key]
    if "pixel_values" not in model_inputs:
        raise SamInvalidOutputError("SAM2 processor did not return pixel_values")
    return model_inputs


def _cast_transformers_pixel_values(
    model_inputs: dict[str, object],
    *,
    dtype: object,
) -> dict[str, object]:
    pixel_values = model_inputs["pixel_values"]
    if not isinstance(pixel_values, _DTypeMovable):
        raise SamInvalidOutputError("SAM2 processor returned non-castable pixel_values")
    model_inputs["pixel_values"] = pixel_values.to(dtype=dtype)
    return model_inputs


def _require_tensor_input(
    processor_inputs: _ProcessorInputs,
    key: str,
) -> _TensorLike:
    try:
        value = processor_inputs[key]
    except KeyError as exc:
        raise SamInvalidOutputError(f"SAM2 processor did not return {key}") from exc
    if not isinstance(value, _TensorLike):
        raise SamInvalidOutputError(f"SAM2 processor returned non-tensor {key}")
    return cast(_TensorLike, value)


def _split_predict_output(
    predict_output: tuple[object, object, object] | tuple[object, object],
) -> tuple[object, object]:
    if len(predict_output) < 2:
        raise SamInvalidOutputError(
            "SAM2 predictor.predict returned no masks or scores"
        )
    return predict_output[0], predict_output[1]


def _extract_single_prompt_masks(raw_post_processed_masks: Sequence[object]) -> object:
    if not raw_post_processed_masks:
        raise SamInvalidOutputError("SAM2 processor returned no post-processed masks")
    mask_array = np.asarray(raw_post_processed_masks[0])
    if mask_array.ndim == 4:
        if mask_array.shape[0] != 1:
            raise SamInvalidOutputError(
                "SAM2 processor masks must contain exactly one prompt group"
            )
        return mask_array[0]
    if mask_array.ndim == 3:
        return mask_array
    raise SamInvalidOutputError(
        "SAM2 processor masks must have shape group x candidates x height x width"
    )


def _extract_single_prompt_scores(raw_scores: object) -> object:
    score_array = np.asarray(raw_scores)
    if score_array.ndim == 2:
        if score_array.shape[0] != 1:
            raise SamInvalidOutputError(
                "SAM2 processor scores must contain exactly one prompt group"
            )
        return score_array[0]
    if score_array.ndim == 1:
        return score_array
    raise SamInvalidOutputError("SAM2 processor scores must be a 1D candidate array")


def _validate_mask_batch(
    raw_masks: object,
    *,
    image_height: int,
    image_width: int,
) -> NDArray[np.bool_]:
    masks_array = np.asarray(raw_masks)
    if masks_array.ndim != 3:
        raise SamInvalidOutputError(
            "SAM2 predictor masks must have shape candidate_count x height x width"
        )
    if masks_array.shape[0] < 1:
        raise SamInvalidOutputError("SAM2 predictor returned no mask candidates")
    if masks_array.shape[1] != image_height or masks_array.shape[2] != image_width:
        raise SamInvalidOutputError(
            "SAM2 predictor mask size must match the input image size: "
            f"mask={masks_array.shape[1]}x{masks_array.shape[2]}, "
            f"image={image_height}x{image_width}"
        )
    if masks_array.dtype == np.bool_:
        return cast(NDArray[np.bool_], masks_array)
    if not np.issubdtype(masks_array.dtype, np.integer):
        raise SamInvalidOutputError(
            "SAM2 predictor masks must be binary bool or integer 0/1 arrays"
        )
    unique_values = np.unique(masks_array)
    if not bool(np.all((unique_values == 0) | (unique_values == 1))):
        raise SamInvalidOutputError(
            "SAM2 predictor masks must be binary bool or integer 0/1 arrays"
        )
    return cast(NDArray[np.bool_], masks_array.astype(np.bool_, copy=False))


def _validate_scores(
    raw_scores: object, *, candidate_count: int
) -> NDArray[np.float64]:
    try:
        scores = np.asarray(raw_scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SamInvalidOutputError(
            "SAM2 predictor scores must be convertible to finite floats"
        ) from exc
    if scores.ndim != 1:
        raise SamInvalidOutputError("SAM2 predictor scores must be a 1D array")
    if scores.shape[0] != candidate_count:
        raise SamInvalidOutputError(
            "SAM2 predictor masks and scores must have the same candidate count: "
            f"masks={candidate_count}, scores={scores.shape[0]}"
        )
    if not bool(np.all(np.isfinite(scores))):
        raise SamInvalidOutputError("SAM2 predictor scores must be finite")
    if not bool(np.all((scores >= 0.0) & (scores <= 1.0))):
        raise SamInvalidOutputError("SAM2 predictor scores must be in [0, 1]")
    return cast(NDArray[np.float64], scores)


def _write_candidate_masks(
    mask_batch: NDArray[np.bool_],
    scores: NDArray[np.float64],
    *,
    staging_dir: Path,
) -> tuple[SamMaskCandidateResponse, ...]:
    total_pixels = mask_batch.shape[1] * mask_batch.shape[2]
    if total_pixels <= 0:
        raise SamInvalidOutputError("SAM2 masks must contain at least one pixel")

    candidates: list[SamMaskCandidateResponse] = []
    for index in range(mask_batch.shape[0]):
        mask = mask_batch[index]
        candidate_id = f"mask_{index:02d}"
        mask_npz_path = staging_dir / f"{candidate_id}.npz"
        try:
            np.savez_compressed(mask_npz_path, mask=mask)
            mask_npz_bytes = mask_npz_path.read_bytes()
        except OSError as exc:
            raise SamArtifactWriteError(
                "could not write SAM2 mask artifact: "
                f"path={mask_npz_path}; error_type={exc.__class__.__name__}"
            ) from exc
        pixel_count = int(np.count_nonzero(mask))
        coverage_percent = (pixel_count / total_pixels) * 100.0
        candidates.append(
            SamMaskCandidateResponse(
                candidate_id=candidate_id,
                score=float(scores[index]),
                mask_npz_path=mask_npz_path,
                mask_npz_base64=base64.b64encode(mask_npz_bytes).decode("ascii"),
                mask_npz_sha256=hashlib.sha256(mask_npz_bytes).hexdigest(),
                pixel_count=pixel_count,
                coverage_percent=coverage_percent,
            )
        )
    return tuple(candidates)


def _response_candidates_by_value(
    candidates: tuple[SamMaskCandidateResponse, ...],
) -> tuple[SamMaskCandidateResponse, ...]:
    return tuple(_response_candidate_by_value(candidate) for candidate in candidates)


def _response_candidate_by_value(
    candidate: SamMaskCandidateResponse,
) -> SamMaskCandidateResponse:
    if candidate.mask_rle is not None:
        return SamMaskCandidateResponse(
            candidate_id=candidate.candidate_id,
            score=candidate.score,
            mask_rle=candidate.mask_rle,
            pixel_count=candidate.pixel_count,
            coverage_percent=candidate.coverage_percent,
        )
    if candidate.mask_npz_path is None:
        return candidate

    mask = _load_candidate_mask_for_response(candidate.mask_npz_path)
    try:
        mask_rle_payload = encode_bool_mask_rle(mask)
    except ValueError as exc:
        raise SamInvalidOutputError(
            "SAM2 response mask artifact could not be encoded as RLE: "
            f"path={candidate.mask_npz_path}; error_type={exc.__class__.__name__}"
        ) from exc
    return SamMaskCandidateResponse(
        candidate_id=candidate.candidate_id,
        score=candidate.score,
        mask_rle=SamMaskRle(
            encoding="row_major_counts",
            height=mask_rle_payload.height,
            width=mask_rle_payload.width,
            counts=mask_rle_payload.counts,
        ),
        pixel_count=candidate.pixel_count,
        coverage_percent=candidate.coverage_percent,
    )


def _load_candidate_mask_for_response(
    mask_npz_path: Path,
) -> NDArray[np.bool_]:
    try:
        with np.load(mask_npz_path) as archive:
            if "mask" not in archive.files:
                raise SamArtifactWriteError(
                    "SAM2 mask artifact is missing required key 'mask': "
                    f"path={mask_npz_path}"
                )
            mask_array = archive["mask"]
    except SamArtifactWriteError:
        raise
    except (OSError, ValueError) as exc:
        raise SamArtifactWriteError(
            "could not load SAM2 mask artifact for response: "
            f"path={mask_npz_path}; error_type={exc.__class__.__name__}"
        ) from exc
    return _validate_single_mask_array_for_response(mask_array, path=mask_npz_path)


def _validate_single_mask_array_for_response(
    raw_mask: object,
    *,
    path: Path,
) -> NDArray[np.bool_]:
    mask_array = np.asarray(raw_mask)
    if mask_array.ndim != 2:
        raise SamInvalidOutputError(
            "SAM2 response mask artifact must contain a 2D mask: "
            f"path={path}; ndim={mask_array.ndim}"
        )
    if mask_array.shape[0] <= 0 or mask_array.shape[1] <= 0:
        raise SamInvalidOutputError(
            "SAM2 response mask artifact must contain a non-empty 2D mask: "
            f"path={path}; shape={mask_array.shape}"
        )
    if mask_array.dtype == np.bool_:
        return cast(NDArray[np.bool_], mask_array)
    if not np.issubdtype(mask_array.dtype, np.integer):
        raise SamInvalidOutputError(
            "SAM2 response mask artifact must contain a binary bool or integer "
            f"0/1 mask: path={path}"
        )
    unique_values = np.unique(mask_array)
    if not bool(np.all((unique_values == 0) | (unique_values == 1))):
        raise SamInvalidOutputError(
            "SAM2 response mask artifact must contain a binary bool or integer "
            f"0/1 mask: path={path}"
        )
    return cast(NDArray[np.bool_], mask_array.astype(np.bool_, copy=False))


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
    return any(marker in message for marker in _CUDA_RESOURCE_ERROR_MARKERS)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OfficialSam2Runner",
    "SamArtifactWriteError",
    "SamImageLoadError",
    "SamInvalidOutputError",
    "SamStagingPathError",
    "Sam2Runner",
    "TransformersSam2Runner",
    "build_arg_parser",
    "main",
    "run_sam_mask_request",
    "serve",
]
