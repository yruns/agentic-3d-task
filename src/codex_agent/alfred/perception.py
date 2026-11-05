"""Validated RGB perception boundary for local depth, masks and CLIP features."""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .depth_models import require_local_depth_model, validate_intrinsics
from .model_runtime import physical_gpu_uuid
from .provenance import PERCEPTION_PACKAGES, RuntimeProvenance, alfred_source_sha256
from .vocabulary import ALFRED_CATEGORIES, detection_batches
from .worker import JsonWorker

PERCEPTION_PROFILE = "grounded_scene_rgb_v1"


@dataclass(frozen=True)
class ScenePerceptionConfig:
    """Explicit local model assets; no implicit download or replacement backend."""

    python_executable: Path
    dino_config: Path
    dino_checkpoint: Path
    dino_text_encoder_dir: Path
    sam_checkpoint: Path
    clip_checkpoint: Path
    depth_model_dir: Path
    gpu_lock_path: Path
    device: Literal["cuda:0", "cpu"] = "cuda:0"
    physical_gpu_index: int = 0
    sam_model_type: Literal["vit_h", "vit_l", "vit_b"] = "vit_h"
    clip_model_name: str = "ViT-H-14"
    box_threshold: float = 0.2
    text_threshold: float = 0.2
    nms_iou_threshold: float = 0.5
    caption_batch_size: int = 29
    max_detections: int = 100
    timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if (
            type(self.physical_gpu_index) is not int
            or self.physical_gpu_index < 0
            or self.physical_gpu_index == 1
        ):
            raise ValueError(
                "physical_gpu_index must be a nonnegative integer other than forbidden GPU 1"
            )
        if self.device not in ("cuda:0", "cpu"):
            raise ValueError(f"device={self.device!r} must be 'cuda:0' or 'cpu'")
        if self.sam_model_type not in ("vit_h", "vit_l", "vit_b"):
            raise ValueError(f"sam_model_type={self.sam_model_type!r} is unsupported")
        if not self.clip_model_name or not re.fullmatch(
            r"[A-Za-z0-9_-]+", self.clip_model_name
        ):
            raise ValueError(
                "clip_model_name must be a local OpenCLIP architecture name"
            )
        for name in ("box_threshold", "text_threshold", "nms_iou_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name}={value!r} must be finite in (0,1)")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        detection_batches(self.caption_batch_size)
        if type(self.max_detections) is not int or not 1 <= self.max_detections <= 1000:
            raise ValueError("max_detections must be an integer in [1,1000]")
        for name in (
            "python_executable",
            "dino_config",
            "dino_checkpoint",
            "dino_text_encoder_dir",
            "sam_checkpoint",
            "clip_checkpoint",
            "depth_model_dir",
            "gpu_lock_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).absolute())
        for name in (
            "python_executable",
            "dino_config",
            "dino_checkpoint",
            "sam_checkpoint",
            "clip_checkpoint",
        ):
            path = getattr(self, name)
            if not path.is_file():
                raise FileNotFoundError(f"{name} artifact missing: {path}")
        for name in ("config.json", "tokenizer_config.json", "vocab.txt"):
            if not (self.dino_text_encoder_dir / name).is_file():
                raise FileNotFoundError(
                    f"DINO text encoder artifact missing: {self.dino_text_encoder_dir / name}"
                )
        if not any(
            (self.dino_text_encoder_dir / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        ):
            raise FileNotFoundError(
                f"DINO text encoder weights missing: {self.dino_text_encoder_dir}"
            )
        require_local_depth_model(self.depth_model_dir, model_name="da3metric-large")

    def to_payload(self) -> dict[str, object]:
        """Serialize the validated configuration for the isolated model worker."""
        return {
            name: str(value) if isinstance(value, Path) else value
            for name, value in asdict(self).items()
        }


class PerceptionDetection(BaseModel):
    """One mask and its original floating-point detector box and category."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    category: str = Field(min_length=1)
    mask_path: Path
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_detection(self) -> PerceptionDetection:
        x1, y1, x2, y2 = self.bbox_xyxy
        if not x1 < x2 or not y1 < y2:
            raise ValueError(
                f"bbox_xyxy={self.bbox_xyxy!r} must bound a nonempty region"
            )
        if self.category not in ALFRED_CATEGORIES:
            raise ValueError(
                f"category={self.category!r} is outside fixed ALFRED vocabulary"
            )
        return self


class PerceptionDiagnostics(BaseModel):
    """Visible accounting for fixed vocabulary coverage and filtered detections."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    vocabulary_categories: int = Field(ge=0)
    caption_batches: int = Field(ge=0)
    candidate_boxes: int = Field(ge=0)
    nms_suppressed: int = Field(ge=0)
    capped_detections: int = Field(ge=0)
    empty_masks: int = Field(ge=0)


class PerceptionObservation(BaseModel):
    """Meter-scale depth, instance masks and paired normalized CLIP features."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    frame_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    depth_path: Path
    depth_model: Literal["da3metric-large"]
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    features_path: Path
    detections: tuple[PerceptionDetection, ...]
    diagnostics: PerceptionDiagnostics

    @model_validator(mode="after")
    def validate_shape_and_counts(self) -> PerceptionObservation:
        for detection in self.detections:
            if (
                detection.bbox_xyxy[0] >= self.image_width
                or detection.bbox_xyxy[1] >= self.image_height
                or detection.bbox_xyxy[2] <= 0
                or detection.bbox_xyxy[3] <= 0
            ):
                raise ValueError("detection bbox must intersect observation image")
        counts = self.diagnostics
        if counts.vocabulary_categories != len(ALFRED_CATEGORIES):
            raise ValueError("perception did not cover the complete fixed vocabulary")
        if (
            counts.candidate_boxes
            != counts.nms_suppressed
            + counts.capped_detections
            + counts.empty_masks
            + len(self.detections)
        ):
            raise ValueError("detection diagnostics do not account for every candidate")
        return self


class PerceptionRuntimeStats(BaseModel):
    """Measured costs and explicit suppression counts for one predictor lifetime."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    load_seconds: float = Field(ge=0)
    inference_seconds: float = Field(ge=0)
    frames: int = Field(ge=0)
    peak_cuda_memory_bytes: int = Field(ge=0)
    device: str
    depth_seconds: float = Field(ge=0)
    detection_seconds: float = Field(ge=0)
    segmentation_seconds: float = Field(ge=0)
    feature_seconds: float = Field(ge=0)
    capped_detections: int = Field(ge=0)
    nms_suppressed: int = Field(ge=0)
    empty_masks: int = Field(ge=0)
    gpu_wait_seconds: float = Field(ge=0)
    residency: Literal["sequential_cpu_offload"]


def validate_prediction_artifacts(
    observation: PerceptionObservation, *, output_dir: Path
) -> None:
    """Check external arrays, ordering, normalization and path ownership."""
    root = output_dir.resolve()
    paths = [
        observation.depth_path,
        observation.features_path,
        *(item.mask_path for item in observation.detections),
    ]
    if len(set(paths)) != len(paths):
        raise ValueError("prediction artifact paths must be distinct")
    for path in paths:
        if not path.is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError(f"prediction path={path} must be within output_dir={root}")
    shape = (observation.image_height, observation.image_width)
    depth = np.load(observation.depth_path, allow_pickle=False)
    if (
        depth.shape != shape
        or depth.dtype != np.float32
        or not np.isfinite(depth).all()
        or np.any(depth <= 0)
    ):
        raise ValueError("predicted depth must be positive finite float32 HxW meters")
    for detection in observation.detections:
        mask = np.load(detection.mask_path, allow_pickle=False)
        if (
            mask.shape != shape
            or mask.dtype != np.uint8
            or not np.isin(mask, [0, 1]).all()
            or not mask.any()
        ):
            raise ValueError(
                f"predicted mask={detection.mask_path} must be nonempty binary uint8 HxW"
            )
    with np.load(observation.features_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"image_features", "text_features", "frame_feature"}:
            raise ValueError(
                "features archive must contain image_features/text_features/frame_feature"
            )
        image, text, frame = (
            arrays["image_features"],
            arrays["text_features"],
            arrays["frame_feature"],
        )
        if (
            frame.ndim != 1
            or not frame.size
            or image.shape != (len(observation.detections), frame.size)
            or text.shape != image.shape
        ):
            raise ValueError("feature dimensions/order differ from detections")
        for name, values in (
            ("image_features", image),
            ("text_features", text),
            ("frame_feature", frame),
        ):
            if (
                values.dtype != np.float32
                or not np.isfinite(values).all()
                or not np.allclose(np.linalg.norm(values, axis=-1), 1, atol=1e-3)
            ):
                raise ValueError(
                    f"{name} must contain finite unit-normalized float32 features"
                )


class ScenePerception:
    """Own one persistent isolated perception worker across sequential episodes."""

    def __init__(self, config: ScenePerceptionConfig, *, log_path: Path) -> None:
        self.config = config
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": (
                    physical_gpu_uuid(config.physical_gpu_index)
                    if config.device == "cuda:0"
                    else ""
                ),
                "NVIDIA_TF32_OVERRIDE": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        self._worker = JsonWorker(
            command=[
                str(config.python_executable),
                "-u",
                "-m",
                "codex_agent.alfred.perception_worker",
            ],
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            log_path=log_path,
            timeout_seconds=config.timeout_seconds,
        )
        try:
            initialized = self._worker.request(
                {"command": "initialize", "config": config.to_payload()}
            )
            if initialized.get("backend") != PERCEPTION_PROFILE:
                raise ValueError("perception worker initialized an unexpected backend")
            self._runtime_provenance = RuntimeProvenance.model_validate(
                initialized.get("runtime_provenance")
            )
            if set(self._runtime_provenance.packages) != set(PERCEPTION_PACKAGES):
                raise ValueError("perception worker package provenance is incomplete")
            if self._runtime_provenance.package_source_sha256 != alfred_source_sha256():
                raise ValueError(
                    "perception worker imported different project source bytes"
                )
            if (
                self._runtime_provenance.visible_devices
                != environment["CUDA_VISIBLE_DEVICES"]
            ):
                raise ValueError(
                    "perception worker accelerator identity differs from configured UUID"
                )
        except BaseException:
            # Process ownership begins before initialization; cancellation must
            # close our worker even when the context manager was never entered.
            self._worker.close()
            raise

    @property
    def runtime_provenance(self) -> RuntimeProvenance:
        """Return validated provenance captured inside the actual model worker."""
        return self._runtime_provenance

    def predict(
        self,
        rgb_path: Path,
        *,
        frame_id: str,
        intrinsics: NDArray[np.floating[Any]],
        output_dir: Path,
    ) -> PerceptionObservation:
        """Predict only from the given RGB and calibrated intrinsic matrix."""
        from PIL import Image

        if not isinstance(frame_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", frame_id
        ):
            raise ValueError(f"frame_id={frame_id!r} must be a safe filename component")
        matrix = validate_intrinsics(intrinsics)
        path = rgb_path.resolve(strict=True)
        with Image.open(path) as image:
            if image.mode != "RGB" or min(image.size) < 2:
                raise ValueError(
                    "rgb_path must contain an RGB image with positive dimensions"
                )
            width, height = image.size
        observation = PerceptionObservation.model_validate(
            self._worker.request(
                {
                    "command": "predict",
                    "rgb_path": str(path),
                    "frame_id": frame_id,
                    "intrinsics": matrix.tolist(),
                    "output_dir": str(output_dir.absolute()),
                }
            )
        )
        if observation.frame_id != frame_id or (
            observation.image_width,
            observation.image_height,
        ) != (width, height):
            raise ValueError(
                "perception response frame identity/dimensions differ from request"
            )
        validate_prediction_artifacts(observation, output_dir=output_dir)
        return observation

    def statistics(self) -> PerceptionRuntimeStats:
        return PerceptionRuntimeStats.model_validate(
            self._worker.request({"command": "statistics"})
        )

    def close(self) -> None:
        self._worker.close()

    def __enter__(self) -> ScenePerception:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
