"""Explicit local metric-depth backends; no simulator data or fallback models."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .model_runtime import active_model

FloatArray = NDArray[np.float32]


def validate_intrinsics(intrinsics: NDArray[np.floating[Any]]) -> FloatArray:
    """Require a finite calibrated pinhole camera instead of estimating its scale."""
    matrix = np.asarray(intrinsics, dtype=np.float32)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], [0, 0, 1], atol=1e-6)
        or abs(float(matrix[0, 1])) > 1e-6
        or abs(float(matrix[1, 0])) > 1e-6
    ):
        raise ValueError(
            "intrinsics must be finite 3x3 pinhole K with positive focal lengths and last row [0,0,1]"
        )
    return matrix.copy()


def canonical_depth_to_meters(
    depth: NDArray[np.floating[Any]], processed_intrinsics: NDArray[np.floating[Any]]
) -> FloatArray:
    """Apply DA3Metric's canonical focal conversion once at its output resolution.

    The official input processor scales K with the image. The standalone metric
    network's raw forward result is canonical depth; the nested network, which
    already scales metric depth, is deliberately unsupported here.
    """
    matrix = validate_intrinsics(processed_intrinsics)
    values = np.asarray(depth, dtype=np.float32)
    if (
        values.ndim != 2
        or not values.size
        or not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        raise ValueError("canonical depth must be a nonempty finite positive HxW array")
    meters = values * float((matrix[0, 0] + matrix[1, 1]) / 600)
    if not np.isfinite(meters).all() or np.any(meters <= 0):
        raise ValueError("metric depth conversion produced invalid meters")
    return np.asarray(meters, dtype=np.float32)


def require_local_depth_model(directory: Path, *, model_name: str) -> None:
    """Check the exact model identity and local Hub files before framework import."""
    for name in ("config.json", "model.safetensors"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"depth model artifact missing: {directory / name}")
    if model_name == "da3metric-large":
        configuration = json.loads((directory / "config.json").read_text())
        if configuration.get("model_name") != model_name:
            raise ValueError(
                f"depth config model_name={configuration.get('model_name')!r} must be {model_name!r}"
            )


def strip_da3_wrapper_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the observed API wrapper prefix before strict network loading."""
    if not state or any(not key.startswith("model.") for key in state):
        raise ValueError(
            "DA3 checkpoint keys must all start with the model. wrapper prefix"
        )
    return {key[len("model.") :]: value for key, value in state.items()}


class DA3MetricDepth:
    """One local DA3Metric-Large model, using calibrated scale and no pose input."""

    identifier = "da3metric-large"

    def __init__(self, directory: Path, *, device: str) -> None:
        require_local_depth_model(directory, model_name=self.identifier)
        configuration = importlib.import_module("depth_anything_3.cfg")
        registry = importlib.import_module("depth_anything_3.registry")
        processor = importlib.import_module("depth_anything_3.utils.io.input_processor")
        output = importlib.import_module("depth_anything_3.utils.io.output_processor")
        safetensors = importlib.import_module("safetensors.torch")
        omegaconf = importlib.import_module("omegaconf")
        architecture = configuration.load_config(
            registry.MODEL_REGISTRY[self.identifier]
        )
        saved_config = json.loads((directory / "config.json").read_text())
        if saved_config.get("config") != omegaconf.OmegaConf.to_container(
            architecture, resolve=True
        ):
            raise ValueError(
                "DA3 checkpoint architecture differs from the pinned metric model"
            )
        self.model: Any = configuration.create_object(architecture).eval()
        state = safetensors.load_file(
            str(directory / "model.safetensors"), device="cpu"
        )
        self.model.load_state_dict(strip_da3_wrapper_prefix(state), strict=True)
        self.input_processor: Any = processor.InputProcessor()
        self.output_processor: Any = output.OutputProcessor()
        self.device = device

    def predict(
        self, rgb: NDArray[np.uint8], intrinsics: NDArray[np.floating[Any]]
    ) -> FloatArray:
        """Return positive camera-axis depth in meters at the original RGB size."""
        matrix = validate_intrinsics(intrinsics)
        cv2 = importlib.import_module("cv2")
        images, _, cameras = self.input_processor(
            [rgb],
            intrinsics=matrix[None],
            process_res=504,
            process_res_method="upper_bound_resize",
            num_workers=1,
            sequential=True,
        )
        if cameras is None or tuple(cameras.shape) != (1, 3, 3):
            raise ValueError("DA3 preprocessing did not return the calibrated camera")
        with active_model(self.model, self.device) as model:
            torch = importlib.import_module("torch")
            autocast_dtype = (
                torch.bfloat16
                if self.device.startswith("cuda") and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            with torch.autocast(
                device_type=self.device.split(":")[0],
                dtype=autocast_dtype,
                enabled=self.device.startswith("cuda"),
            ):
                raw = model.forward(
                    images[None].to(self.device).float(), export_feat_layers=[]
                )
            if bool(raw.get("is_metric", False)):
                raise ValueError(
                    "DA3 output is already metric; refusing double scaling"
                )
            prediction = self.output_processor(raw)
            canonical = np.asarray(prediction.depth)
            if canonical.shape != (1, *images.shape[-2:]):
                raise ValueError(
                    f"DA3 depth shape={canonical.shape} differs from processed image"
                )
            meters = canonical_depth_to_meters(canonical[0], cameras[0].cpu().numpy())
            del raw, prediction
        resized = cv2.resize(
            meters, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        return np.asarray(resized, dtype=np.float32)


class UniDepthMetricDepth:
    """Explicit offline calibration comparator, never selected by production RPC."""

    identifier = "unidepth-v2-vitl14"

    def __init__(self, directory: Path, *, device: str) -> None:
        require_local_depth_model(directory, model_name=self.identifier)
        module = importlib.import_module("unidepth.models")
        self.model: Any = module.UniDepthV2.from_pretrained(
            str(directory), local_files_only=True, strict=True
        ).eval()
        self.device = device

    def predict(
        self, rgb: NDArray[np.uint8], intrinsics: NDArray[np.floating[Any]]
    ) -> FloatArray:
        """Use supplied intrinsics and return UniDepth's existing meter scale."""
        torch = importlib.import_module("torch")
        matrix = validate_intrinsics(intrinsics)
        with active_model(self.model, self.device) as model:
            image = (
                torch.from_numpy(np.ascontiguousarray(rgb))
                .permute(2, 0, 1)
                .to(self.device)
            )
            result = model.infer(image, torch.from_numpy(matrix).to(self.device))
            depth = np.asarray(
                result["depth"].squeeze().float().cpu().numpy(), dtype=np.float32
            )
            del image, result
        if (
            depth.shape != rgb.shape[:2]
            or not np.isfinite(depth).all()
            or np.any(depth <= 0)
        ):
            raise ValueError("UniDepth returned invalid metric depth")
        return depth
