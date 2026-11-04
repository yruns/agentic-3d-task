"""RGB-only FILM perception in an explicitly configured legacy subprocess.

The main agent retains its existing environment. ALFRED-trained model classes
and weights are loaded once in FILM's compatible interpreter, then shared across
sequential episodes. No simulator metadata enters this boundary.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .perception_worker import normalize_horizon_degrees
from .worker import JsonWorker

FILM_CHECKPOINTS = (
    "maskrcnn_alfworld/objects_lr5e-3_005.pth",
    "maskrcnn_alfworld/receps_lr5e-3_003.pth",
    "depth_models/valts0/model-102500-best_silog_17.00430",
    "depth_models/valts/model-2000-best_silog_10.13741",
)
_FRAME_SIZE = (300, 300)


@dataclass(frozen=True)
class FilmPerceptionConfig:
    """Explicit model locations, thresholds and physical GPU 0 or CPU execution."""

    python_executable: Path
    film_root: Path
    weights_root: Path
    device: Literal["cuda:0", "cpu"] = "cuda:0"
    score_threshold: float = 0.5
    mask_threshold: float = 0.5
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.device not in ("cuda:0", "cpu"):
            raise ValueError(f"device={self.device!r} must be 'cuda:0' or 'cpu'")
        for name in ("score_threshold", "mask_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name}={value!r} must be finite and in (0, 1)")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        for name in ("python_executable", "film_root", "weights_root"):
            object.__setattr__(self, name, Path(getattr(self, name)).absolute())
        if not self.python_executable.is_file():
            raise FileNotFoundError(
                f"FILM interpreter missing: {self.python_executable}"
            )
        if not self.film_root.is_dir():
            raise FileNotFoundError(f"FILM checkout missing: {self.film_root}")
        for relative in FILM_CHECKPOINTS:
            if not (self.weights_root / relative).is_file():
                raise FileNotFoundError(
                    f"FILM checkpoint missing: {self.weights_root / relative}"
                )


class PerceptionDetection(BaseModel):
    """One predicted instance; the mask is image-local, never a simulator ID."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    category: str = Field(min_length=1)
    mask_path: Path
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_bbox(self) -> PerceptionDetection:
        x1, y1, x2, y2 = self.bbox_xyxy
        if not (0 <= x1 < x2 <= 300 and 0 <= y1 < y2 <= 300):
            raise ValueError(
                f"bbox_xyxy={self.bbox_xyxy!r} must bound a nonempty 300x300 region"
            )
        return self


class PerceptionObservation(BaseModel):
    """Metric depth and masks predicted only from the current RGB frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    depth_path: Path
    depth_model: Literal["level", "downward"]
    detections: tuple[PerceptionDetection, ...]


class PerceptionRuntimeStats(BaseModel):
    """Measured worker costs for the current predictor lifetime."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    load_seconds: float = Field(ge=0)
    inference_seconds: float = Field(ge=0)
    frames: int = Field(ge=0)
    peak_cuda_memory_bytes: int = Field(ge=0)
    device: str


def validate_prediction_artifacts(
    observation: PerceptionObservation, *, output_dir: Path
) -> None:
    """Validate external worker arrays before they enter geometry or memory."""
    root = output_dir.resolve()
    paths = [
        observation.depth_path,
        *(item.mask_path for item in observation.detections),
    ]
    for path in paths:
        if not path.is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError(f"prediction path={path} must be within output_dir={root}")
    depth = np.load(observation.depth_path, allow_pickle=False)
    if (
        depth.shape != _FRAME_SIZE
        or depth.dtype != np.float32
        or not np.isfinite(depth).all()
        or np.any(depth < 0)
        or np.any(depth > 5)
    ):
        raise ValueError(
            "predicted depth must be finite float32 300x300 meters in [0, 5]"
        )
    for detection in observation.detections:
        mask = np.load(detection.mask_path, allow_pickle=False)
        if (
            mask.shape != _FRAME_SIZE
            or mask.dtype != np.uint8
            or not np.isin(mask, [0, 1]).all()
            or not mask.any()
        ):
            raise ValueError(
                f"predicted mask={detection.mask_path} must be nonempty binary uint8 300x300"
            )


class FilmPerception:
    """Own persistent neural models and expose bounded RGB prediction requests."""

    def __init__(self, config: FilmPerceptionConfig, *, log_path: Path) -> None:
        self.config = config
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "0" if config.device == "cuda:0" else "",
                "NVIDIA_TF32_OVERRIDE": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        self._worker = JsonWorker(
            command=[
                str(config.python_executable),
                "-u",
                str(Path(__file__).with_name("perception_worker.py")),
                "--film-root",
                str(config.film_root),
                "--weights-root",
                str(config.weights_root),
                "--device",
                config.device,
                "--score-threshold",
                str(config.score_threshold),
                "--mask-threshold",
                str(config.mask_threshold),
            ],
            cwd=config.film_root,
            env=environment,
            log_path=log_path,
            timeout_seconds=config.timeout_seconds,
        )
        try:
            initialized = self._worker.request({"command": "initialize"})
            if initialized != {"backend": "film_predicted_rgb_v1"}:
                raise ValueError("perception worker returned an unexpected backend")
        except BaseException:
            self._worker.close()
            raise

    def predict(
        self, rgb_path: Path, *, frame_id: str, horizon_degrees: float, output_dir: Path
    ) -> PerceptionObservation:
        """Predict one 300x300 RGB frame and retain metric depth and instance masks."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", frame_id):
            raise ValueError(
                f"frame_id={frame_id!r} must contain letters, digits, '_' or '-'"
            )
        horizon_degrees = normalize_horizon_degrees(horizon_degrees)
        rgb_path = rgb_path.resolve(strict=True)
        output_dir = output_dir.absolute()
        output_dir.mkdir(parents=True, exist_ok=True)
        response = self._worker.request(
            {
                "command": "predict",
                "rgb_path": str(rgb_path),
                "frame_id": frame_id,
                "horizon_degrees": horizon_degrees,
                "output_dir": str(output_dir),
            }
        )
        observation = PerceptionObservation.model_validate(response)
        if observation.frame_id != frame_id:
            raise ValueError(
                "perception response frame_id differs from the requested frame"
            )
        validate_prediction_artifacts(observation, output_dir=output_dir)
        return observation

    def statistics(self) -> PerceptionRuntimeStats:
        """Read measured initialization/inference costs from the running worker."""
        return PerceptionRuntimeStats.model_validate(
            self._worker.request({"command": "statistics"})
        )

    def close(self) -> None:
        """Release this predictor and its subprocess only."""
        self._worker.close()

    def __enter__(self) -> FilmPerception:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
