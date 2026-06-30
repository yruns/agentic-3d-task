"""Molmo + SAM + depth lifting smoke for SceneFuncVal-CG."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

PointSource = Literal["manual_pixel", "molmo_percent", "pixel_tuple"]
MaskSelection = Literal["score", "smallest"]
SmokeStatus = Literal["passed", "no_mask", "empty_lift", "mask_too_large"]

BoolArray: TypeAlias = NDArray[np.bool_]
UInt8Array: TypeAlias = NDArray[np.uint8]
Int64Array: TypeAlias = NDArray[np.int64]
Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
TensorBatch: TypeAlias = Mapping[str, Tensor]


class MolmoProcessor(Protocol):
    """Protocol for Molmo's remote-code processor."""

    tokenizer: PreTrainedTokenizerBase

    def process(self, *, images: Sequence[Image.Image], text: str) -> TensorBatch:
        """Process images and prompt text for Molmo generation."""


class MolmoModel(Protocol):
    """Protocol for Molmo's remote-code causal language model."""

    @property
    def device(self) -> torch.device:
        """Return the active model device."""

    def generate_from_batch(
        self,
        batch: TensorBatch,
        generation_config: GenerationConfig,
        *,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Tensor:
        """Generate tokens from a processed Molmo batch."""

    def eval(self) -> None:
        """Switch the model to evaluation mode."""


class DynamicCacheSwitchable(Protocol):
    """Protocol for transformer generation cache compatibility override."""

    _supports_default_dynamic_cache: Callable[[], bool]


class SamPredictorLike(Protocol):
    """Protocol for the subset of Segment Anything's predictor used here."""

    def set_image(self, image: UInt8Array) -> None:
        """Set the RGB image used for point-prompted prediction."""

    def predict(
        self,
        *,
        point_coords: Float32Array,
        point_labels: Int64Array,
        multimask_output: bool,
    ) -> tuple[BoolArray, Float32Array, Float32Array]:
        """Predict SAM masks and scores for one point prompt."""


@dataclass(frozen=True)
class SmokeConfig:
    dataset_root: Path
    scene_id: str
    frame_id: str
    prompt: str
    output_dir: Path
    molmo_model_id: str
    molmo_cache_dir: Path
    sam_checkpoint: Path
    desc_id: str
    depth_scale: float
    max_new_tokens: int
    max_lift_points: int
    manual_point: str
    sam_selection: MaskSelection
    max_success_coverage_percent: float

    def __post_init__(self) -> None:
        _require_non_empty("scene_id", self.scene_id)
        _require_non_empty("frame_id", self.frame_id)
        _require_non_empty("prompt", self.prompt)
        _require_positive_float("depth_scale", self.depth_scale)
        _require_positive_int("max_new_tokens", self.max_new_tokens)
        _require_positive_int("max_lift_points", self.max_lift_points)
        _require_positive_float(
            "max_success_coverage_percent", self.max_success_coverage_percent
        )
        if self.max_success_coverage_percent > 100.0:
            raise ValueError(
                "max_success_coverage_percent must be <= 100.0, "
                f"got {self.max_success_coverage_percent}"
            )


@dataclass(frozen=True)
class CliRequest:
    config: SmokeConfig
    allow_failure: bool


@dataclass(frozen=True)
class ImagePoint:
    x: float
    y: float
    x_percent: float
    y_percent: float
    source: PointSource


@dataclass(frozen=True)
class SamMaskPrediction:
    mask: BoolArray
    score: float


@dataclass(frozen=True)
class LiftedPoints:
    world_points: Float64Array
    colors: UInt8Array
    lifted_point_count: int


@dataclass(frozen=True)
class MaskSummary:
    mask_index: int
    sam_score: float
    pixel_count: int
    image_coverage_percent: float
    lifted_point_count: int
    saved_point_count: int
    overlay_path: str
    mask_npz_path: str
    ply_path: str


@dataclass(frozen=True)
class SuccessAssessment:
    success: bool
    status: SmokeStatus
    reason: str


@dataclass(frozen=True)
class MolmoPatchReport:
    file_path: str
    changed: bool
    operations: tuple[str, ...]


@dataclass(frozen=True)
class SmokeSummary:
    success: bool
    status: SmokeStatus
    status_reason: str
    scene_id: str
    frame_id: str
    desc_id: str
    prompt: str
    image_path: str
    depth_path: str
    intrinsic_path: str
    pose_path: str
    molmo_model_id: str
    sam_checkpoint: str
    sam_selection: MaskSelection
    max_success_coverage_percent: float
    molmo_patch_reports: tuple[MolmoPatchReport, ...]
    molmo_generated_text: str
    points: tuple[ImagePoint, ...]
    masks: tuple[MaskSummary, ...]


@dataclass(frozen=True)
class MolmoBatch:
    tensors: TensorBatch

    @classmethod
    def from_processor_output(
        cls, processor_output: TensorBatch, *, device: torch.device
    ) -> MolmoBatch:
        if "input_ids" not in processor_output:
            keys = ", ".join(sorted(processor_output))
            raise ValueError(f"Molmo processor output missing input_ids; keys={keys}")
        tensors = {
            key: value.to(device).unsqueeze(0)
            for key, value in processor_output.items()
        }
        return cls(tensors=tensors)

    @property
    def input_token_count(self) -> int:
        return int(self.tensors["input_ids"].size(1))


def _require_non_empty(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_positive_float(field_name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{field_name} must be > 0, got {value}")


def _require_positive_int(field_name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0, got {value}")


def positive_float(raw_value: str) -> float:
    value = float(raw_value)
    _require_positive_float("value", value)
    return value


def positive_int(raw_value: str) -> int:
    value = int(raw_value)
    _require_positive_int("value", value)
    return value


def normalize_frame_id(frame_id: str) -> str:
    if not frame_id.isdigit():
        raise ValueError(f"frame_id must contain only digits: {frame_id}")
    return f"{int(frame_id):06d}"


def read_matrix(path: Path, *, shape: tuple[int, int]) -> Float64Array:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != shape:
        raise ValueError(f"expected matrix shape {shape} at {path}, got {matrix.shape}")
    return matrix


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return cast(Image.Image, image.convert("RGB"))


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Molmo + SAM inference")


def patch_molmo_remote_code(model_dir: Path) -> tuple[MolmoPatchReport, ...]:
    """Apply local Molmo remote-code compatibility patches idempotently."""

    if not model_dir.exists() or not model_dir.is_dir():
        return ()

    reports: list[MolmoPatchReport] = []
    image_processor = model_dir / "image_preprocessing_molmo.py"
    if image_processor.exists():
        reports.append(_patch_molmo_image_processor(image_processor))

    modeling = model_dir / "modeling_molmo.py"
    if modeling.exists():
        reports.append(_patch_molmo_modeling(modeling))

    return tuple(reports)


def _patch_molmo_image_processor(path: Path) -> MolmoPatchReport:
    original = path.read_text(encoding="utf-8")
    updated = original
    operations: list[str] = []
    if "\nimport tensorflow as tf\n" in updated:
        updated = updated.replace("\nimport tensorflow as tf\n", "\n", 1)
        operations.append("removed eager tensorflow import")
    if 'tf = import_module("tensorflow")' not in updated:
        branch = '    if resize_method == "tensorflow":\n'
        replacement = (
            branch
            + "        from importlib import import_module\n\n"
            + '        tf = import_module("tensorflow")\n'
        )
        if branch not in updated:
            raise ValueError(f"cannot locate tensorflow resize branch in {path}")
        updated = updated.replace(branch, replacement, 1)
        operations.append("added lazy tensorflow import")
    return _write_patch_report(path, original, updated, tuple(operations))


def _patch_molmo_modeling(path: Path) -> MolmoPatchReport:
    original = path.read_text(encoding="utf-8")
    updated = original
    operations: list[str] = []
    if "all_tied_weights_keys" not in updated:
        marker = '    _no_split_modules = ["MolmoBlock"]\n'
        if marker not in updated:
            raise ValueError(f"cannot locate MolmoForCausalLM class header in {path}")
        updated = updated.replace(
            marker, marker + "    all_tied_weights_keys = {}\n", 1
        )
        operations.append("added all_tied_weights_keys")
    if "def tie_weights(self):" in updated:
        updated = updated.replace(
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
    if old_cache_update in updated:
        updated = updated.replace(old_cache_update, new_cache_update, 1)
        operations.append("guarded cache_position update")
    return _write_patch_report(path, original, updated, tuple(operations))


def _write_patch_report(
    path: Path, original: str, updated: str, operations: tuple[str, ...]
) -> MolmoPatchReport:
    changed = updated != original
    if changed:
        path.write_text(updated, encoding="utf-8")
    return MolmoPatchReport(
        file_path=str(path),
        changed=changed,
        operations=operations,
    )


def force_legacy_generation_cache(model: DynamicCacheSwitchable) -> None:
    """Keep Molmo remote code on tuple-style KV cache with newer transformers."""

    def uses_default_dynamic_cache() -> bool:
        return False

    model._supports_default_dynamic_cache = uses_default_dynamic_cache


def load_molmo_model(
    config: SmokeConfig,
) -> tuple[MolmoModel, MolmoProcessor, tuple[MolmoPatchReport, ...]]:
    require_cuda()
    model_path = Path(config.molmo_model_id).expanduser()
    local_files_only = model_path.exists()
    patch_reports = patch_molmo_remote_code(model_path) if local_files_only else ()
    processor = AutoProcessor.from_pretrained(
        config.molmo_model_id,
        trust_remote_code=True,
        cache_dir=config.molmo_cache_dir,
        local_files_only=local_files_only,
    )
    raw_model = AutoModelForCausalLM.from_pretrained(
        config.molmo_model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        cache_dir=config.molmo_cache_dir,
        local_files_only=local_files_only,
    )
    cast(torch.nn.Module, raw_model).to(torch.device("cuda"))
    force_legacy_generation_cache(cast(DynamicCacheSwitchable, raw_model))
    molmo_model = cast(MolmoModel, raw_model)
    molmo_model.eval()
    return molmo_model, cast(MolmoProcessor, processor), patch_reports


def generate_molmo_text(
    model: MolmoModel,
    processor: MolmoProcessor,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    inputs = processor.process(images=[image], text=prompt)
    batch = MolmoBatch.from_processor_output(inputs, device=model.device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate_from_batch(
            batch.tensors,
            GenerationConfig(
                max_new_tokens=max_new_tokens,
                stop_strings="<|endoftext|>",
                use_cache=True,
            ),
            tokenizer=processor.tokenizer,
        )
    generated_tokens = output[0, batch.input_token_count :]
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    if not isinstance(decoded, str):
        raise TypeError("Molmo tokenizer.decode returned multiple strings")
    return decoded


def parse_molmo_points(
    generated_text: str, *, image_width: int, image_height: int
) -> tuple[ImagePoint, ...]:
    points: list[ImagePoint] = []
    seen_pixels: set[tuple[int, int]] = set()
    percent_pattern = re.compile(
        r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"'
    )
    for match in percent_pattern.finditer(generated_text):
        x_percent = float(match.group(1))
        y_percent = float(match.group(2))
        if x_percent < 0 or x_percent > 100 or y_percent < 0 or y_percent > 100:
            continue
        x = x_percent * image_width / 100.0
        y = y_percent * image_height / 100.0
        pixel_key = (round(x), round(y))
        if pixel_key in seen_pixels:
            continue
        points.append(
            ImagePoint(
                x=x,
                y=y,
                x_percent=x_percent,
                y_percent=y_percent,
                source="molmo_percent",
            )
        )
        seen_pixels.add(pixel_key)

    tuple_pattern = re.compile(r"\((\d{1,4}),\s*(\d{1,4})\)")
    for match in tuple_pattern.finditer(generated_text):
        x = float(match.group(1))
        y = float(match.group(2))
        if x < 0 or x >= image_width or y < 0 or y >= image_height:
            continue
        pixel_key = (round(x), round(y))
        if pixel_key in seen_pixels:
            continue
        points.append(
            ImagePoint(
                x=x,
                y=y,
                x_percent=100.0 * x / image_width,
                y_percent=100.0 * y / image_height,
                source="pixel_tuple",
            )
        )
        seen_pixels.add(pixel_key)
    return tuple(points)


def extract_single_molmo_point(
    generated_text: str, *, image_width: int, image_height: int
) -> ImagePoint:
    points = parse_molmo_points(
        generated_text, image_width=image_width, image_height=image_height
    )
    if len(points) != 1:
        raise ValueError(
            "Molmo output must contain exactly one valid point, "
            f"got {len(points)}: {generated_text!r}"
        )
    return points[0]


def parse_manual_point(
    manual_point: str, *, image_width: int, image_height: int
) -> ImagePoint:
    parts = [part.strip() for part in manual_point.split(",")]
    if len(parts) != 2:
        raise ValueError(f"manual_point must use 'x,y' format: {manual_point}")
    x = float(parts[0])
    y = float(parts[1])
    if x < 0 or x >= image_width or y < 0 or y >= image_height:
        raise ValueError(
            f"manual_point ({x}, {y}) outside image bounds {image_width}x{image_height}"
        )
    return ImagePoint(
        x=x,
        y=y,
        x_percent=100.0 * x / image_width,
        y_percent=100.0 * y / image_height,
        source="manual_pixel",
    )


def load_sam_predictor(checkpoint: Path) -> SamPredictorLike:
    require_cuda()
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint))
    sam.to(device="cuda")
    return cast(SamPredictorLike, SamPredictor(sam))


def select_sam_candidate(
    mask_candidates: BoolArray,
    score_candidates: Float32Array,
    *,
    selection: MaskSelection,
) -> SamMaskPrediction:
    if mask_candidates.shape[0] != score_candidates.shape[0]:
        raise ValueError(
            "SAM candidate mask/score count mismatch: "
            f"{mask_candidates.shape[0]} masks vs {score_candidates.shape[0]} scores"
        )
    if mask_candidates.shape[0] == 0:
        raise ValueError("SAM returned zero mask candidates")
    if selection == "score":
        best_index = int(np.argmax(score_candidates))
    else:
        pixel_counts = np.asarray(
            [np.count_nonzero(candidate) for candidate in mask_candidates],
            dtype=np.int64,
        )
        non_empty_indices = np.nonzero(pixel_counts > 0)[0]
        if non_empty_indices.size == 0:
            best_index = int(np.argmin(pixel_counts))
        else:
            smallest_relative_index = int(np.argmin(pixel_counts[non_empty_indices]))
            best_index = int(non_empty_indices[smallest_relative_index])
    return SamMaskPrediction(
        mask=mask_candidates[best_index].astype(bool),
        score=float(score_candidates[best_index]),
    )


def predict_sam_masks(
    predictor: SamPredictorLike,
    image: Image.Image,
    points: Sequence[ImagePoint],
    *,
    selection: MaskSelection,
) -> tuple[SamMaskPrediction, ...]:
    predictor.set_image(np.asarray(image, dtype=np.uint8))
    predictions: list[SamMaskPrediction] = []
    for point in points:
        point_coords = np.asarray([[point.x, point.y]], dtype=np.float32)
        point_labels = np.asarray([1], dtype=np.int64)
        mask_candidates, score_candidates, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        predictions.append(
            select_sam_candidate(
                mask_candidates.astype(bool),
                score_candidates.astype(np.float32),
                selection=selection,
            )
        )
    return tuple(predictions)


def lift_mask_to_world_points(
    mask: BoolArray,
    rgb: UInt8Array,
    depth: NDArray[np.uint16],
    intrinsic: Float64Array,
    pose: Float64Array,
    *,
    depth_scale: float,
    max_lift_points: int,
) -> LiftedPoints:
    _require_positive_float("depth_scale", depth_scale)
    _require_positive_int("max_lift_points", max_lift_points)
    valid_mask = np.logical_and(mask, depth > 0)
    ys, xs = np.nonzero(valid_mask)
    lifted_count = int(xs.size)
    if lifted_count == 0:
        return LiftedPoints(
            world_points=np.empty((0, 3), dtype=np.float64),
            colors=np.empty((0, 3), dtype=np.uint8),
            lifted_point_count=0,
        )

    if lifted_count > max_lift_points:
        indices = np.linspace(0, lifted_count - 1, max_lift_points, dtype=np.int64)
        xs = xs[indices]
        ys = ys[indices]

    z = depth[ys, xs].astype(np.float64) / depth_scale
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    x_camera = (xs.astype(np.float64) - cx) * z / fx
    y_camera = (ys.astype(np.float64) - cy) * z / fy
    camera_points = np.stack(
        [x_camera, y_camera, z, np.ones_like(z, dtype=np.float64)], axis=1
    )
    world_points = (pose @ camera_points.T).T[:, :3].astype(np.float64)
    colors = rgb[ys, xs].astype(np.uint8)
    return LiftedPoints(
        world_points=world_points,
        colors=colors,
        lifted_point_count=lifted_count,
    )


def save_ply(path: Path, points: Float64Array, colors: UInt8Array) -> None:
    if points.shape[0] != colors.shape[0]:
        raise ValueError(
            f"points and colors must have same length, got {points.shape[0]} and {colors.shape[0]}"
        )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def save_overlay(
    path: Path, image: Image.Image, mask: BoolArray, point: ImagePoint
) -> None:
    mask_alpha = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    mask_alpha[mask] = np.asarray([255, 0, 0, 110], dtype=np.uint8)
    overlay = Image.fromarray(mask_alpha)
    composed = Image.alpha_composite(image.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(composed)
    radius = 10
    x = int(round(point.x))
    y = int(round(point.y))
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius), outline="white", width=4
    )
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="red")
    composed.convert("RGB").save(path, quality=95)


def assess_smoke_success(
    masks: Sequence[MaskSummary], *, max_success_coverage_percent: float
) -> SuccessAssessment:
    _require_positive_float(
        "max_success_coverage_percent", max_success_coverage_percent
    )
    if not masks:
        return SuccessAssessment(
            success=False,
            status="no_mask",
            reason="SAM did not produce a selected mask",
        )
    if any(
        mask.image_coverage_percent > max_success_coverage_percent for mask in masks
    ):
        return SuccessAssessment(
            success=False,
            status="mask_too_large",
            reason=(
                "selected SAM mask exceeds max_success_coverage_percent="
                f"{max_success_coverage_percent}"
            ),
        )
    if not any(mask.lifted_point_count > 0 for mask in masks):
        return SuccessAssessment(
            success=False,
            status="empty_lift",
            reason="selected SAM mask has no valid depth pixels to lift",
        )
    return SuccessAssessment(
        success=True,
        status="passed",
        reason="selected SAM mask passed coverage and depth-lift checks",
    )


def write_summary(path: Path, summary: SmokeSummary) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def run_smoke(config: SmokeConfig) -> SmokeSummary:
    frame_id = normalize_frame_id(config.frame_id)
    raw_dir = config.dataset_root / config.scene_id / "raw"
    image_path = raw_dir / f"{frame_id}-rgb.jpg"
    depth_path = raw_dir / f"{frame_id}-depth.png"
    intrinsic_path = raw_dir / f"{frame_id}-intrinsic.txt"
    pose_path = raw_dir / f"{frame_id}.txt"
    for path in (
        image_path,
        depth_path,
        intrinsic_path,
        pose_path,
        config.sam_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    run_dir = config.output_dir / config.scene_id / frame_id
    run_dir.mkdir(parents=True, exist_ok=True)
    image = load_rgb_image(image_path)
    patch_reports: tuple[MolmoPatchReport, ...] = ()
    if config.manual_point:
        generated_text = "MANUAL_POINT_DIAGNOSTIC_MODE"
        points = (
            parse_manual_point(
                config.manual_point, image_width=image.width, image_height=image.height
            ),
        )
    else:
        molmo_model, molmo_processor, patch_reports = load_molmo_model(config)
        generated_text = generate_molmo_text(
            molmo_model,
            molmo_processor,
            image,
            config.prompt,
            max_new_tokens=config.max_new_tokens,
        )
        points = (
            extract_single_molmo_point(
                generated_text, image_width=image.width, image_height=image.height
            ),
        )
    (run_dir / "molmo_output.txt").write_text(generated_text, encoding="utf-8")
    masks: list[MaskSummary] = []
    predictor = load_sam_predictor(config.sam_checkpoint)
    sam_predictions = predict_sam_masks(
        predictor,
        image,
        points,
        selection=config.sam_selection,
    )
    depth = np.asarray(Image.open(depth_path), dtype=np.uint16)
    rgb = np.asarray(image, dtype=np.uint8)
    intrinsic = read_matrix(intrinsic_path, shape=(3, 3))
    pose = read_matrix(pose_path, shape=(4, 4))
    for index, (point, prediction) in enumerate(
        zip(points, sam_predictions, strict=True)
    ):
        point_dir = run_dir / f"mask_{index:02d}"
        point_dir.mkdir(parents=True, exist_ok=True)
        mask_npz_path = point_dir / "mask_data.npz"
        np.savez_compressed(
            mask_npz_path,
            mask=prediction.mask.astype(np.uint8),
            point=np.asarray([point.x, point.y], dtype=np.float32),
            score=np.asarray([prediction.score], dtype=np.float32),
        )
        overlay_path = point_dir / "overlay.jpg"
        save_overlay(overlay_path, image, prediction.mask, point)
        lifted = lift_mask_to_world_points(
            prediction.mask,
            rgb,
            depth,
            intrinsic,
            pose,
            depth_scale=config.depth_scale,
            max_lift_points=config.max_lift_points,
        )
        ply_path = point_dir / "lifted_points.ply"
        save_ply(ply_path, lifted.world_points, lifted.colors)
        pixel_count = int(np.count_nonzero(prediction.mask))
        image_coverage = 100.0 * pixel_count / float(prediction.mask.size)
        masks.append(
            MaskSummary(
                mask_index=index,
                sam_score=prediction.score,
                pixel_count=pixel_count,
                image_coverage_percent=image_coverage,
                lifted_point_count=lifted.lifted_point_count,
                saved_point_count=int(lifted.world_points.shape[0]),
                overlay_path=str(overlay_path),
                mask_npz_path=str(mask_npz_path),
                ply_path=str(ply_path),
            )
        )

    mask_summaries = tuple(masks)
    assessment = assess_smoke_success(
        mask_summaries,
        max_success_coverage_percent=config.max_success_coverage_percent,
    )
    summary = SmokeSummary(
        success=assessment.success,
        status=assessment.status,
        status_reason=assessment.reason,
        scene_id=config.scene_id,
        frame_id=frame_id,
        desc_id=config.desc_id,
        prompt=config.prompt,
        image_path=str(image_path),
        depth_path=str(depth_path),
        intrinsic_path=str(intrinsic_path),
        pose_path=str(pose_path),
        molmo_model_id=config.molmo_model_id,
        sam_checkpoint=str(config.sam_checkpoint),
        sam_selection=config.sam_selection,
        max_success_coverage_percent=config.max_success_coverage_percent,
        molmo_patch_reports=patch_reports,
        molmo_generated_text=generated_text,
        points=points,
        masks=mask_summaries,
    )
    write_summary(run_dir / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> CliRequest:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG"
        ),
    )
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--desc-id", default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/"
            "molmo_sam3d_smoke_20260627"
        ),
    )
    parser.add_argument("--molmo-model-id", default="allenai/Molmo-7B-D-0924")
    parser.add_argument(
        "--molmo-cache-dir",
        type=Path,
        default=Path(
            "/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/"
            "models/cache/huggingface"
        ),
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path(
            "/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/"
            "models/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
        ),
    )
    parser.add_argument("--depth-scale", type=positive_float, default=1000.0)
    parser.add_argument("--max-new-tokens", type=positive_int, default=120)
    parser.add_argument("--max-lift-points", type=positive_int, default=50000)
    parser.add_argument(
        "--max-success-coverage-percent",
        type=positive_float,
        default=5.0,
        help="Maximum selected-mask image coverage considered a successful small-object smoke.",
    )
    parser.add_argument(
        "--sam-selection",
        choices=("score", "smallest"),
        default="smallest",
        help="How to select from SAM multimask candidates for each Molmo point.",
    )
    parser.add_argument(
        "--manual-point",
        default="",
        help=(
            "Diagnostic-only pixel point as 'x,y'. When set, skips Molmo and "
            "only validates SAM plus 3D lifting."
        ),
    )
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Write summary and exit zero even when the smoke status is not passed.",
    )
    args = parser.parse_args(argv)
    selection = cast(MaskSelection, args.sam_selection)
    return CliRequest(
        config=SmokeConfig(
            dataset_root=args.dataset_root,
            scene_id=args.scene_id,
            frame_id=args.frame_id,
            prompt=args.prompt,
            output_dir=args.output_dir,
            molmo_model_id=args.molmo_model_id,
            molmo_cache_dir=args.molmo_cache_dir,
            sam_checkpoint=args.sam_checkpoint,
            desc_id=args.desc_id,
            depth_scale=args.depth_scale,
            max_new_tokens=args.max_new_tokens,
            max_lift_points=args.max_lift_points,
            manual_point=args.manual_point,
            sam_selection=selection,
            max_success_coverage_percent=args.max_success_coverage_percent,
        ),
        allow_failure=bool(args.allow_failure),
    )


def main(argv: Sequence[str] | None = None) -> int:
    request = parse_args(argv)
    request.config.molmo_cache_dir.mkdir(parents=True, exist_ok=True)
    summary = run_smoke(request.config)
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    if not summary.success and not request.allow_failure:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
