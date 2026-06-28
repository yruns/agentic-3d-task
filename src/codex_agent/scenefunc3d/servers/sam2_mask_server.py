"""SAM2 mask sidecar server for SceneFunc3D agent tools."""

from __future__ import annotations

import argparse
import importlib
import time
from collections.abc import Sequence
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from types import ModuleType
from typing import Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

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
    SamPointPrompt,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8712
DEFAULT_MODEL_NAME = "SAM2.1-Hiera-L"
DEFAULT_DEVICE = "cuda:0"

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


class _TorchModule(Protocol):
    cuda: _TorchCuda


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
            request.staging_dir,
            staging_root=self._staging_root,
        )
        try:
            image_module = cast(_ImageModule, importlib.import_module("PIL.Image"))
            with image_module.open(request.image_path) as image:
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the SAM2 sidecar command line parser."""

    parser = argparse.ArgumentParser(
        description="Serve the SceneFunc3D SAM2 mask sidecar."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
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
        candidates = runner.masks(request)
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
    return cast(JsonObject, response.model_dump(mode="json"))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the SAM2 mask sidecar server."""

    args = build_arg_parser().parse_args(argv)
    try:
        runner = OfficialSam2Runner(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint_path,
            config_path=args.config_path,
            staging_root=args.staging_root,
            device=args.device,
        )
        serve(runner, host=args.host, port=args.port)
    except KeyboardInterrupt:
        return 130
    return 0


def _require_existing_file(path: Path, *, field_name: str) -> Path:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise RuntimeError(f"{field_name} must be an existing file: {resolved_path}")
    return resolved_path


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


def _split_predict_output(
    predict_output: tuple[object, object, object] | tuple[object, object],
) -> tuple[object, object]:
    if len(predict_output) < 2:
        raise SamInvalidOutputError(
            "SAM2 predictor.predict returned no masks or scores"
        )
    return predict_output[0], predict_output[1]


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
                pixel_count=pixel_count,
                coverage_percent=coverage_percent,
            )
        )
    return tuple(candidates)


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
    "build_arg_parser",
    "main",
    "serve",
]
