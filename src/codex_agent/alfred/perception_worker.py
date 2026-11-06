"""JSONL entrypoint for fixed-vocabulary local RGB scene perception."""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .depth_models import DA3MetricDepth, validate_intrinsics
from .model_runtime import gpu_file_lock
from .perception import (
    PERCEPTION_PROFILE,
    PerceptionDetection,
    PerceptionDiagnostics,
    PerceptionObservation,
    PerceptionRuntimeStats,
    ScenePerceptionConfig,
)
from .provenance import capture_runtime_provenance
from .vision_models import ClipEncoder, GroundingDetector, PromptedSegmenter
from .vocabulary import ALFRED_CATEGORIES, detection_batches, vocabulary_sha256

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Only observed RGB and calibrated intrinsics may enter the neural worker."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    command: str
    rgb_path: Path
    frame_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    intrinsics: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    output_dir: Path


class SceneModels:
    """Load neural models once, executing each stage with bounded GPU residency."""

    def __init__(self, config: ScenePerceptionConfig) -> None:
        started = time.monotonic()
        self.config = config
        self.torch = importlib.import_module("torch")
        self.torch.set_num_threads(1)
        self.torch.set_grad_enabled(False)
        if config.device == "cuda:0":
            if not self.torch.cuda.is_available():
                raise RuntimeError("device=cuda:0 requested but CUDA is unavailable")
            self.torch.cuda.reset_peak_memory_stats()
        self.depth = (
            DA3MetricDepth(config.depth_model_dir, device=config.device)
            if config.depth_source == "da3metric-large"
            else None
        )
        self.detector = GroundingDetector(config)
        self.segmenter = PromptedSegmenter(config)
        self.features = ClipEncoder(config)
        self.load_seconds = time.monotonic() - started
        self.inference_seconds = 0.0
        self.depth_seconds = 0.0
        self.detection_seconds = 0.0
        self.segmentation_seconds = 0.0
        self.feature_seconds = 0.0
        self.frames = 0
        self.gpu_wait_seconds = 0.0
        self.capped_detections = 0
        self.nms_suppressed = 0
        self.empty_masks = 0
        self.last_frame_stats: dict[str, object] = {}

    def predict(self, request: PredictionRequest) -> PerceptionObservation:
        """Hold the shared ALFRED lock for the complete GPU prediction pipeline."""
        with gpu_file_lock(
            self.config.gpu_lock_path, timeout_seconds=self.config.timeout_seconds
        ) as waited:
            self.gpu_wait_seconds += waited
            return self._predict_frame(request, lock_wait_seconds=waited)

    def predict_with_stats(self, request: PredictionRequest) -> dict[str, object]:
        """Run one prediction frame and return both the observation and per-frame timings."""
        observation = self.predict(request)
        return {
            "observation": observation.model_dump(mode="json"),
            "frame_stats": dict(self.last_frame_stats),
        }

    def _predict_frame(
        self, request: PredictionRequest, *, lock_wait_seconds: float = 0.0
    ) -> PerceptionObservation:
        """Write one frame's predicted geometry and aligned CLIP evidence."""
        from PIL import Image

        started = time.monotonic()
        with Image.open(request.rgb_path) as image:
            if image.mode != "RGB" or min(image.size) < 2:
                raise ValueError("rgb_path must be a valid RGB image")
            rgb = np.array(image, dtype=np.uint8)
        intrinsics = validate_intrinsics(
            np.asarray(request.intrinsics, dtype=np.float32)
        )
        output_dir = request.output_dir.absolute()
        output_dir.mkdir(parents=True, exist_ok=True)
        depth_path = output_dir / f"{request.frame_id}_depth.npy"
        features_path = output_dir / f"{request.frame_id}_features.npz"
        if (
            depth_path.exists()
            or features_path.exists()
            or any(output_dir.glob(f"{request.frame_id}_mask_*.npy"))
        ):
            raise FileExistsError(
                f"prediction artifacts already exist: frame_id={request.frame_id}"
            )
        stage = time.monotonic()
        if self.depth is not None:
            depth = self.depth.predict(rgb, intrinsics)
        else:
            gt_depth_candidate = (
                request.rgb_path.resolve().parent.parent
                / "depths"
                / f"{request.rgb_path.stem}.npy"
            )
            if gt_depth_candidate.is_file():
                raw_gt = np.load(gt_depth_candidate, allow_pickle=False).astype(
                    np.float32
                )
                depth = np.asarray(np.clip(raw_gt, 1e-3, 5.0), dtype=np.float32)
            else:
                depth = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        depth_dt = time.monotonic() - stage
        self.depth_seconds += depth_dt
        stage = time.monotonic()
        boxes, candidates, suppressed, capped = self.detector.predict(rgb)
        det_dt = time.monotonic() - stage
        self.detection_seconds += det_dt
        stage = time.monotonic()
        instances, empty = self.segmenter.predict(rgb, boxes)
        seg_dt = time.monotonic() - stage
        self.segmentation_seconds += seg_dt
        stage = time.monotonic()
        image_features, text_features, frame_feature = self.features.predict(
            rgb, instances
        )
        feat_dt = time.monotonic() - stage
        self.feature_seconds += feat_dt
        if (
            self.config.device == "cuda:0"
            and self.torch.cuda.memory_reserved() > 8 * 1024**3
        ):
            self.torch.cuda.empty_cache()
        if capped:
            logger.warning(
                "frame_id=%s discarded=%d reason=max_detections limit=%d",
                request.frame_id,
                capped,
                self.config.max_detections,
            )
        logger.info(
            "frame_id=%s classes=%d candidates=%d suppressed=%d capped=%d empty_masks=%d retained=%d",
            request.frame_id,
            len(ALFRED_CATEGORIES),
            candidates,
            suppressed,
            capped,
            empty,
            len(instances),
        )
        with depth_path.open("xb") as stream:
            np.save(stream, depth, allow_pickle=False)
        with features_path.open("xb") as stream:
            np.savez(
                stream,
                image_features=image_features,
                text_features=text_features,
                frame_feature=frame_feature,
            )
        detections: list[PerceptionDetection] = []
        for index, instance in enumerate(instances):
            mask_path = output_dir / f"{request.frame_id}_mask_{index:03d}.npy"
            mask = instance.mask.astype(np.uint8)
            with mask_path.open("xb") as stream:
                np.save(stream, mask, allow_pickle=False)
            detections.append(
                PerceptionDetection(
                    category=instance.category,
                    mask_path=mask_path,
                    bbox_xyxy=instance.bbox_xyxy,
                    confidence=instance.confidence,
                )
            )
        observation = PerceptionObservation(
            frame_id=request.frame_id,
            depth_path=depth_path,
            depth_model=self.config.depth_source,
            image_width=rgb.shape[1],
            image_height=rgb.shape[0],
            features_path=features_path,
            detections=tuple(detections),
            diagnostics=PerceptionDiagnostics(
                vocabulary_categories=len(ALFRED_CATEGORIES),
                caption_batches=len(detection_batches(self.config.caption_batch_size)),
                candidate_boxes=candidates,
                nms_suppressed=suppressed,
                capped_detections=capped,
                empty_masks=empty,
            ),
        )
        self.frames += 1
        self.capped_detections += capped
        self.nms_suppressed += suppressed
        self.empty_masks += empty
        inf_dt = time.monotonic() - started
        self.inference_seconds += inf_dt
        peak_mem = (
            int(self.torch.cuda.max_memory_allocated())
            if self.config.device == "cuda:0"
            else 0
        )
        self.last_frame_stats = {
            "load_seconds": self.load_seconds,
            "inference_seconds": inf_dt,
            "depth_seconds": depth_dt,
            "detection_seconds": det_dt,
            "segmentation_seconds": seg_dt,
            "feature_seconds": feat_dt,
            "gpu_wait_seconds": lock_wait_seconds,
            "peak_cuda_memory_bytes": peak_mem,
            "device": self.config.device,
            "capped_detections": capped,
            "nms_suppressed": suppressed,
            "empty_masks": empty,
        }
        return observation

    def statistics(self) -> PerceptionRuntimeStats:
        """Expose actual model-stage timings and suppression counts."""
        return PerceptionRuntimeStats(
            load_seconds=self.load_seconds,
            inference_seconds=self.inference_seconds,
            frames=self.frames,
            peak_cuda_memory_bytes=(
                int(self.torch.cuda.max_memory_allocated())
                if self.config.device == "cuda:0"
                else 0
            ),
            device=self.config.device,
            depth_seconds=self.depth_seconds,
            detection_seconds=self.detection_seconds,
            segmentation_seconds=self.segmentation_seconds,
            feature_seconds=self.feature_seconds,
            capped_detections=self.capped_detections,
            nms_suppressed=self.nms_suppressed,
            empty_masks=self.empty_masks,
            gpu_wait_seconds=self.gpu_wait_seconds,
            residency="sequential_cpu_offload",
        )


def main() -> None:
    """Serve JSONL only on stdout; any prediction failure terminates the worker."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    models: SceneModels | None = None
    protocol_stdout = sys.stdout
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("perception request must be a JSON object")
                command = request.get("command")
                if command == "close" and set(request) == {"command"}:
                    break
                if command == "initialize" and set(request) == {"command", "config"}:
                    if models is not None:
                        raise ValueError("perception models are already initialized")
                    configuration = TypeAdapter(ScenePerceptionConfig).validate_python(
                        request["config"]
                    )
                    with gpu_file_lock(
                        configuration.gpu_lock_path,
                        timeout_seconds=configuration.timeout_seconds,
                    ) as waited:
                        models = SceneModels(configuration)
                        models.gpu_wait_seconds += waited
                    result: dict[str, object] = {
                        "backend": PERCEPTION_PROFILE,
                        "vocabulary_sha256": vocabulary_sha256(),
                        "runtime_provenance": capture_runtime_provenance(
                            model_worker=True
                        ).model_dump(mode="json"),
                    }
                elif command == "predict" and models is not None:
                    result = models.predict(
                        PredictionRequest.model_validate(request)
                    ).model_dump(mode="json")
                elif command == "predict_with_stats" and models is not None:
                    req_payload = {**request, "command": "predict"}
                    result = models.predict_with_stats(
                        PredictionRequest.model_validate(req_payload)
                    )
                elif (
                    command == "statistics"
                    and models is not None
                    and set(request) == {"command"}
                ):
                    result = models.statistics().model_dump(mode="json")
                else:
                    raise ValueError(f"unexpected perception command={command!r}")
            response = {"ok": True, "result": result}
        except Exception as exc:
            # Preserve a visible failure; request-level validation/file errors
            # do not kill an already-initialized worker in a shared server pool.
            traceback.print_exc(file=sys.stderr)
            recoverable = models is not None and isinstance(
                exc, (ValueError, FileNotFoundError, FileExistsError, PermissionError)
            )
            protocol_stdout.write(
                json.dumps(
                    {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "recoverable": recoverable,
                    }
                )
                + "\n"
            )
            protocol_stdout.flush()
            if recoverable:
                continue
            raise SystemExit(1) from exc
        protocol_stdout.write(json.dumps(response, allow_nan=False) + "\n")
        protocol_stdout.flush()


if __name__ == "__main__":
    main()
