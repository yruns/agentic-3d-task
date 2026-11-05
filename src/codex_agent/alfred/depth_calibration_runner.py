"""Offline comparison runner; ground truth never enters either model adapter."""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .depth_calibration import (
    SELECTION_RULE,
    CalibrationFrame,
    CalibrationManifest,
    DepthMetrics,
    aggregate_depth_metrics,
    depth_metrics,
    load_calibration_manifest,
    plane_metrics,
    select_depth_backend,
    temporal_depth_consistency,
)
from .depth_models import DA3MetricDepth, UniDepthMetricDepth, require_local_depth_model
from .model_runtime import gpu_file_lock, physical_gpu_uuid
from .provenance import code_provenance, file_sha256


class DepthPredictor(Protocol):
    identifier: str

    def predict(
        self, rgb: NDArray[np.uint8], intrinsics: NDArray[np.floating[Any]]
    ) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class CalibrationRunConfig:
    manifest_path: Path
    da3_model_dir: Path
    unidepth_model_dir: Path
    output_dir: Path
    gpu_lock_path: Path
    project_root: Path
    device: Literal["cuda:0", "cpu"] = "cuda:0"
    physical_gpu_index: int = 0
    lock_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if self.device not in ("cuda:0", "cpu"):
            raise ValueError("calibration device must be logical cuda:0 or cpu")
        if (
            type(self.physical_gpu_index) is not int
            or self.physical_gpu_index < 0
            or self.physical_gpu_index == 1
        ):
            raise ValueError(
                f"physical_gpu_index={self.physical_gpu_index!r} must be a nonnegative integer other than 1"
            )
        if not np.isfinite(self.lock_timeout_seconds) or self.lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be finite and positive")


def _write_new_json(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _read_frame(
    frame: CalibrationFrame,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    from PIL import Image

    with Image.open(frame.rgb_path) as image:
        if image.mode != "RGB":
            raise ValueError(f"calibration RGB is not RGB mode: {frame.frame_id}")
        rgb = np.array(image)
    target = np.load(frame.gt_depth_path, allow_pickle=False)
    if target.dtype != np.float32 or target.shape != rgb.shape[:2]:
        raise ValueError(
            f"GT depth must be RGB-aligned float32 meters: {frame.frame_id}"
        )
    return rgb, target


def evaluate_depth_model(
    predictor: DepthPredictor,
    manifest: CalibrationManifest,
    *,
    output_dir: Path,
    gpu_lock_path: Path,
    lock_timeout_seconds: float,
) -> dict[str, object]:
    """Score a complete same-frame set; persist per-frame predictions and timing."""
    output_dir.mkdir()
    rows: list[dict[str, object]] = []
    metrics: list[DepthMetrics] = []
    previous: dict[
        str, tuple[CalibrationFrame, NDArray[np.float32], NDArray[np.float32]]
    ] = {}
    consistency: list[dict[str, object]] = []
    total_inference_seconds = 0.0
    total_wait_seconds = 0.0
    for frame in manifest.frames:
        rgb, target = _read_frame(frame)
        started = time.monotonic()
        with gpu_file_lock(
            gpu_lock_path, timeout_seconds=lock_timeout_seconds
        ) as waited:
            inference_started = time.monotonic()
            prediction = predictor.predict(rgb, np.asarray(frame.intrinsics))
            inference_seconds = time.monotonic() - inference_started
        wall_seconds = time.monotonic() - started
        total_inference_seconds += inference_seconds
        total_wait_seconds += waited
        if prediction.dtype != np.float32 or prediction.shape != target.shape:
            raise ValueError(
                f"depth adapter returned invalid shape/dtype: {frame.frame_id}"
            )
        score = depth_metrics(prediction, target)
        prediction_path = output_dir / f"{frame.frame_id}_depth.npy"
        with prediction_path.open("xb") as stream:
            np.save(stream, prediction, allow_pickle=False)
        floor_mask = None
        if frame.floor_mask_path is not None:
            raw_mask = np.load(frame.floor_mask_path, allow_pickle=False)
            if raw_mask.shape != target.shape or not np.isin(raw_mask, [0, 1]).all():
                raise ValueError("frozen floor mask must be binary and RGB aligned")
            floor_mask = raw_mask.astype(bool)
        row: dict[str, object] = {
            "frame_id": frame.frame_id,
            "scene_id": frame.scene_id,
            "pitch_degrees": frame.pitch_degrees,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": file_sha256(prediction_path),
            "metrics": score.model_dump(),
            "plane": plane_metrics(
                prediction, target, np.asarray(frame.intrinsics), floor_mask
            ),
            "inference_seconds": inference_seconds,
            "gpu_wait_seconds": waited,
            "wall_seconds": wall_seconds,
        }
        with (output_dir / "frames.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if frame.scene_id in previous:
            prior, prior_prediction, prior_target = previous[frame.scene_id]
            consistency.append(
                {
                    "source_frame_id": prior.frame_id,
                    "target_frame_id": frame.frame_id,
                    **temporal_depth_consistency(
                        prior_prediction, prediction, prior_target, target, prior, frame
                    ),
                }
            )
        previous[frame.scene_id] = frame, prediction, target
        rows.append(row)
        metrics.append(score)
    stratified: dict[str, dict[str, object]] = {}
    for field in ("scene_id", "pitch_degrees"):
        stratified[field] = {
            str(value): aggregate_depth_metrics(
                [
                    score
                    for frame, score in zip(manifest.frames, metrics, strict=True)
                    if getattr(frame, field) == value
                ]
            )
            for value in sorted({getattr(frame, field) for frame in manifest.frames})
        }
    return {
        "backend": predictor.identifier,
        "summary": aggregate_depth_metrics(metrics),
        "stratified": stratified,
        "frames": rows,
        "temporal_consistency": consistency,
        "plane_metrics_note": "Unavailable without a separately frozen floor mask; no plane is invented from depth.",
        "inference_seconds": total_inference_seconds,
        "gpu_wait_seconds": total_wait_seconds,
    }


def _backend_source_provenance() -> dict[str, object]:
    """Fingerprint imported backend code/config without importing their models."""
    result: dict[str, object] = {}
    for name in ("depth_anything_3", "unidepth"):
        specification = importlib.util.find_spec(name)
        if specification is None or not specification.submodule_search_locations:
            raise ModuleNotFoundError(
                f"required calibration backend is not installed: {name}"
            )
        roots = [
            Path(path).resolve() for path in specification.submodule_search_locations
        ]
        digest = hashlib.sha256()
        count = 0
        for root in roots:
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix in {".py", ".json", ".yaml"}:
                    digest.update(str(path.relative_to(root)).encode() + b"\0")
                    digest.update(path.read_bytes())
                    digest.update(b"\0")
                    count += 1
        result[name] = {
            "package_roots": [str(root) for root in roots],
            "source_config_sha256": digest.hexdigest(),
            "files": count,
        }
    return result


def run_depth_calibration(config: CalibrationRunConfig) -> dict[str, object]:
    """Compare exactly two metric models, recording failures without substitution."""
    manifest = load_calibration_manifest(config.manifest_path)
    source = code_provenance(config.project_root)
    if config.output_dir.exists():
        raise FileExistsError(f"calibration output already exists: {config.output_dir}")
    require_local_depth_model(config.da3_model_dir, model_name="da3metric-large")
    require_local_depth_model(
        config.unidepth_model_dir, model_name="unidepth-v2-vitl14"
    )
    gpu_uuid = (
        physical_gpu_uuid(config.physical_gpu_index)
        if config.device == "cuda:0"
        else ""
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    config.output_dir.mkdir(parents=True)
    metadata = {
        "scope": "Frozen train frames only; raw meters; no model sees GT, pose, task goals, or pilot validation data",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "manifest_sha256": file_sha256(config.manifest_path),
        "capture_source_sha256": manifest.source_sha256,
        "cuda_visible_devices": gpu_uuid,
        "selection_rule": SELECTION_RULE,
        "code": source,
        "backend_sources": _backend_source_provenance(),
        "checkpoint_sha256": {
            str(path): file_sha256(path)
            for directory in (config.da3_model_dir, config.unidepth_model_dir)
            for path in (directory / "config.json", directory / "model.safetensors")
        },
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "safetensors", "huggingface-hub")
        },
    }
    _write_new_json(config.output_dir / "protocol.json", metadata)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    results: dict[str, dict[str, object]] = {}
    try:
        for factory, directory in (
            (DA3MetricDepth, config.da3_model_dir),
            (UniDepthMetricDepth, config.unidepth_model_dir),
        ):
            if config.device == "cuda:0":
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            predictor = factory(directory, device=config.device)
            load_seconds = time.monotonic() - started
            result = evaluate_depth_model(
                predictor,
                manifest,
                output_dir=config.output_dir / predictor.identifier,
                gpu_lock_path=config.gpu_lock_path,
                lock_timeout_seconds=config.lock_timeout_seconds,
            )
            result.update(
                load_seconds=load_seconds,
                peak_cuda_memory_bytes=(
                    int(torch.cuda.max_memory_allocated())
                    if config.device == "cuda:0"
                    else 0
                ),
            )
            results[predictor.identifier] = result
            _write_new_json(config.output_dir / f"{predictor.identifier}.json", result)
            del predictor
            gc.collect()
        summaries = {name: result["summary"] for name, result in results.items()}
        if not all(isinstance(summary, dict) for summary in summaries.values()):
            raise ValueError("calibration model summary is malformed")
        report = {
            "status": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": metadata["manifest_sha256"],
            "frames_per_backend": len(manifest.frames),
            "results": results,
            "selection": select_depth_backend(
                {
                    name: dict(summary)
                    for name, summary in summaries.items()
                    if isinstance(summary, dict)
                }
            ),
        }
        _write_new_json(config.output_dir / "comparison.json", report)
        return report
    except Exception as exc:
        _write_new_json(
            config.output_dir / "error.json",
            {
                "status": "error",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_backends": list(results),
                "selection": None,
                "note": "No fallback or GT scale alignment was performed; preserve partial frame logs.",
            },
        )
        raise
