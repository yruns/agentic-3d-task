"""Molmo + SAM + depth lifting smoke for SceneFuncVal-CG."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

PointSource = Literal["manual_pixel", "molmo_percent", "pixel_tuple"]
MaskSelection = Literal["score", "smallest"]


class MolmoProcessor(Protocol):
    """Protocol for Molmo's remote-code processor."""

    tokenizer: PreTrainedTokenizerBase

    def process(self, *, images: list[Image.Image], text: str) -> dict[str, Tensor]:
        """Process images and prompt text for Molmo generation."""


class MolmoModel(Protocol):
    """Protocol for Molmo's remote-code causal language model."""

    @property
    def device(self) -> torch.device:
        """Return the active model device."""

    def generate_from_batch(
        self,
        batch: dict[str, Tensor],
        generation_config: GenerationConfig,
        *,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Tensor:
        """Generate tokens from a processed Molmo batch."""


class DynamicCacheSwitchable(Protocol):
    """Protocol for transformer generation cache compatibility override."""

    _supports_default_dynamic_cache: Callable[[], bool]


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


@dataclass(frozen=True)
class ImagePoint:
    x: float
    y: float
    x_percent: float
    y_percent: float
    source: PointSource


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
class SmokeSummary:
    success: bool
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
    molmo_generated_text: str
    points: list[ImagePoint]
    masks: list[MaskSummary]


def normalize_frame_id(frame_id: str) -> str:
    if not frame_id.isdigit():
        raise ValueError(f"frame_id must contain only digits: {frame_id}")
    return f"{int(frame_id):06d}"


def read_matrix(path: Path, *, shape: tuple[int, int]) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != shape:
        raise ValueError(f"expected matrix shape {shape} at {path}, got {matrix.shape}")
    return matrix


def load_rgb_image(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def force_legacy_generation_cache(model: DynamicCacheSwitchable) -> None:
    """Keep Molmo remote code on tuple-style KV cache with newer transformers."""

    def uses_default_dynamic_cache() -> bool:
        return False

    model._supports_default_dynamic_cache = uses_default_dynamic_cache


def load_molmo_model(config: SmokeConfig) -> tuple[MolmoModel, MolmoProcessor]:
    local_files_only = Path(config.molmo_model_id).expanduser().exists()
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(
        config.molmo_model_id,
        trust_remote_code=True,
        cache_dir=config.molmo_cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.molmo_model_id,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        cache_dir=config.molmo_cache_dir,
        local_files_only=local_files_only,
    )
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    force_legacy_generation_cache(cast(DynamicCacheSwitchable, model))
    model.eval()
    return cast(MolmoModel, model), cast(MolmoProcessor, processor)


def generate_molmo_text(
    model: MolmoModel,
    processor: MolmoProcessor,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    inputs = processor.process(images=[image], text=prompt)
    batch = {key: value.to(model.device).unsqueeze(0) for key, value in inputs.items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate_from_batch(
            batch,
            GenerationConfig(
                max_new_tokens=max_new_tokens,
                stop_strings="<|endoftext|>",
                use_cache=True,
            ),
            tokenizer=processor.tokenizer,
        )
    generated_tokens = output[0, batch["input_ids"].size(1) :]
    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def extract_points(
    generated_text: str, *, image_width: int, image_height: int
) -> list[ImagePoint]:
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
    return points


def parse_manual_point(
    manual_point: str, *, image_width: int, image_height: int
) -> list[ImagePoint]:
    if not manual_point:
        return []
    parts = [part.strip() for part in manual_point.split(",")]
    if len(parts) != 2:
        raise ValueError(f"manual point must use 'x,y' format: {manual_point}")
    x = float(parts[0])
    y = float(parts[1])
    if x < 0 or x >= image_width or y < 0 or y >= image_height:
        raise ValueError(
            f"manual point ({x}, {y}) outside image bounds {image_width}x{image_height}"
        )
    return [
        ImagePoint(
            x=x,
            y=y,
            x_percent=100.0 * x / image_width,
            y_percent=100.0 * y / image_height,
            source="manual_pixel",
        )
    ]


def load_sam_predictor(checkpoint: Path) -> object:
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint))
    sam.to(device="cuda")
    return SamPredictor(sam)


def predict_sam_masks(
    predictor: object,
    image: Image.Image,
    points: list[ImagePoint],
    *,
    selection: MaskSelection,
) -> tuple[list[np.ndarray], list[float]]:
    from segment_anything import SamPredictor

    typed_predictor = cast(SamPredictor, predictor)
    typed_predictor.set_image(np.asarray(image))
    masks: list[np.ndarray] = []
    scores: list[float] = []
    for point in points:
        point_coords = np.asarray([[point.x, point.y]], dtype=np.float32)
        point_labels = np.asarray([1], dtype=np.int64)
        mask_candidates, score_candidates, _ = typed_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
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
                smallest_relative_index = int(
                    np.argmin(pixel_counts[non_empty_indices])
                )
                best_index = int(non_empty_indices[smallest_relative_index])
        masks.append(mask_candidates[best_index].astype(bool))
        scores.append(float(score_candidates[best_index]))
    return masks, scores


def lift_mask_to_world_points(
    mask: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    pose: np.ndarray,
    *,
    depth_scale: float,
    max_lift_points: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    valid_mask = np.logical_and(mask, depth > 0)
    ys, xs = np.nonzero(valid_mask)
    lifted_count = int(xs.size)
    if lifted_count == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8), 0

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
    world_points = (pose @ camera_points.T).T[:, :3]
    colors = rgb[ys, xs].astype(np.uint8)
    return world_points, colors, lifted_count


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
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
    path: Path, image: Image.Image, mask: np.ndarray, point: ImagePoint
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_alpha = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    mask_alpha[mask] = np.asarray([255, 0, 0, 110], dtype=np.uint8)
    overlay = Image.fromarray(mask_alpha, mode="RGBA")
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
    for path in [
        image_path,
        depth_path,
        intrinsic_path,
        pose_path,
        config.sam_checkpoint,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    run_dir = config.output_dir / config.scene_id / frame_id
    run_dir.mkdir(parents=True, exist_ok=True)
    image = load_rgb_image(image_path)
    if config.manual_point:
        generated_text = "MANUAL_POINT_DIAGNOSTIC_MODE"
        points = parse_manual_point(
            config.manual_point, image_width=image.width, image_height=image.height
        )
    else:
        molmo_model, molmo_processor = load_molmo_model(config)
        generated_text = generate_molmo_text(
            molmo_model,
            molmo_processor,
            image,
            config.prompt,
            max_new_tokens=config.max_new_tokens,
        )
        points = extract_points(
            generated_text, image_width=image.width, image_height=image.height
        )
    (run_dir / "molmo_output.txt").write_text(generated_text, encoding="utf-8")
    masks: list[MaskSummary] = []
    if points:
        predictor = load_sam_predictor(config.sam_checkpoint)
        sam_masks, sam_scores = predict_sam_masks(
            predictor,
            image,
            points,
            selection=config.sam_selection,
        )
        depth = np.asarray(Image.open(depth_path))
        rgb = np.asarray(image)
        intrinsic = read_matrix(intrinsic_path, shape=(3, 3))
        pose = read_matrix(pose_path, shape=(4, 4))
        for index, (point, mask, score) in enumerate(
            zip(points, sam_masks, sam_scores, strict=True)
        ):
            point_dir = run_dir / f"mask_{index:02d}"
            point_dir.mkdir(parents=True, exist_ok=True)
            mask_npz_path = point_dir / "mask_data.npz"
            np.savez_compressed(
                mask_npz_path,
                mask=mask.astype(np.uint8),
                point=np.asarray([point.x, point.y], dtype=np.float32),
                score=np.asarray([score], dtype=np.float32),
            )
            overlay_path = point_dir / "overlay.jpg"
            save_overlay(overlay_path, image, mask, point)
            world_points, colors, lifted_count = lift_mask_to_world_points(
                mask,
                rgb,
                depth,
                intrinsic,
                pose,
                depth_scale=config.depth_scale,
                max_lift_points=config.max_lift_points,
            )
            ply_path = point_dir / "lifted_points.ply"
            save_ply(ply_path, world_points, colors)
            pixel_count = int(np.count_nonzero(mask))
            image_coverage = 100.0 * pixel_count / float(mask.size)
            masks.append(
                MaskSummary(
                    mask_index=index,
                    sam_score=score,
                    pixel_count=pixel_count,
                    image_coverage_percent=image_coverage,
                    lifted_point_count=lifted_count,
                    saved_point_count=int(world_points.shape[0]),
                    overlay_path=str(overlay_path),
                    mask_npz_path=str(mask_npz_path),
                    ply_path=str(ply_path),
                )
            )

    summary = SmokeSummary(
        success=bool(
            points and masks and any(mask.lifted_point_count > 0 for mask in masks)
        ),
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
        molmo_generated_text=generated_text,
        points=points,
        masks=masks,
    )
    write_summary(run_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--max-lift-points", type=int, default=50000)
    parser.add_argument(
        "--sam-selection",
        choices=("score", "smallest"),
        default="score",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SmokeConfig(
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
        sam_selection=args.sam_selection,
    )
    config.molmo_cache_dir.mkdir(parents=True, exist_ok=True)
    summary = run_smoke(config)
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
