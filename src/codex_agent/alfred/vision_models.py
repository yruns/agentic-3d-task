"""Local GroundingDINO, SAM1 and OpenCLIP adapters with fixed class prompts.

Dynamic values are confined to the optional neural-framework boundary. The
rest of the application receives validated arrays and canonical ALFRED labels.
"""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .model_runtime import active_model
from .vocabulary import ALFRED_CATEGORIES, category_prompt, detection_batches

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL.Image import Image

    from .perception import ScenePerceptionConfig


@dataclass(frozen=True)
class GroundedBox:
    """A scored fixed-vocabulary box in original RGB pixel coordinates."""

    category: str
    xyxy: tuple[float, float, float, float]
    confidence: float

    def __post_init__(self) -> None:
        if self.category not in ALFRED_CATEGORIES:
            raise ValueError(f"category={self.category!r} is outside ALFRED vocabulary")
        x1, y1, x2, y2 = self.xyxy
        if not all(math.isfinite(v) for v in self.xyxy) or not x1 < x2 or not y1 < y2:
            raise ValueError(f"xyxy={self.xyxy!r} must be finite nonempty pixel bounds")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence={self.confidence!r} must be in [0,1]")


@dataclass(frozen=True)
class MaskedInstance:
    category: str
    mask: NDArray[np.bool_]
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


def padded_detection_crop(
    image: Image, bbox_xyxy: tuple[float, float, float, float]
) -> Image:
    """Use the prepared-scene CLIP crop: original detector box plus 20 pixels.

    Preserve the RGB background and PIL's floating-coordinate crop rounding.
    Segmentation masks do not alter this appearance feature's input pixels.
    """
    left, top, right, bottom = bbox_xyxy
    return image.crop(
        (
            max(0.0, left - 20),
            max(0.0, top - 20),
            min(float(image.width), right + 20),
            min(float(image.height), bottom + 20),
        )
    )


def token_spans_for_categories(
    caption: str, phrases: Sequence[str], offsets: Sequence[tuple[int, int]]
) -> tuple[tuple[int, ...], ...]:
    """Bind whole caption spans to labels, avoiding substring label collisions."""
    expected = ". ".join(phrases) + "."
    if caption != expected:
        raise ValueError(
            "caption must contain the fixed phrases in their original order"
        )
    result: list[tuple[int, ...]] = []
    start = 0
    for phrase in phrases:
        end = start + len(phrase)
        indices = tuple(
            index
            for index, (left, right) in enumerate(offsets)
            if right > left and left >= start and right <= end
        )
        if not indices:
            raise ValueError(f"tokenizer omitted category phrase={phrase!r}")
        result.append(indices)
        start = end + 2
    return tuple(result)


def suppress_boxes(
    boxes: Sequence[GroundedBox], *, iou_threshold: float, limit: int
) -> tuple[tuple[GroundedBox, ...], int, int]:
    """Suppress duplicate boxes across caption batches; report every cap discard."""
    if not 0 < iou_threshold < 1 or type(limit) is not int or limit < 1:
        raise ValueError("NMS requires iou_threshold in (0,1) and positive limit")
    selected: list[GroundedBox] = []
    suppressed = 0
    for candidate in sorted(boxes, key=lambda item: -item.confidence):
        if any(
            _intersection_over_union(candidate.xyxy, previous.xyxy) > iou_threshold
            for previous in selected
        ):
            suppressed += 1
        else:
            selected.append(candidate)
    capped = max(0, len(selected) - limit)
    return tuple(selected[:limit]), suppressed, capped


def _intersection_over_union(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (area_left + area_right - intersection)


def decode_grounded_boxes(
    logits: NDArray[np.floating[Any]],
    boxes: NDArray[np.floating[Any]],
    categories: Sequence[str],
    spans: Sequence[Sequence[int]],
    *,
    image_size: tuple[int, int],
    box_threshold: float,
    text_threshold: float,
) -> tuple[GroundedBox, ...]:
    """Preserve upstream max-token scoring while binding exact category spans.

    GroundingDINO's box confidence is the maximum over the complete token row;
    both thresholds use strict greater-than comparisons. Only label binding
    differs from substring phrase matching, which confuses sliced/base labels.
    """
    if (
        logits.ndim != 2
        or boxes.shape != (len(logits), 4)
        or len(categories) != len(spans)
        or not len(categories)
    ):
        raise ValueError("GroundingDINO output dimensions differ from caption schema")
    if (
        not np.isfinite(logits).all()
        or not np.isfinite(boxes).all()
        or np.any((logits < 0) | (logits > 1))
    ):
        raise ValueError(
            "GroundingDINO returned nonfinite boxes or invalid probabilities"
        )
    if any(not span or min(span) < 0 or max(span) >= logits.shape[1] for span in spans):
        raise ValueError("category tokens exceed GroundingDINO output length")
    scores = np.stack([logits[:, list(span)].max(axis=1) for span in spans], axis=1)
    width, height = image_size
    result: list[GroundedBox] = []
    for index, category_index in enumerate(scores.argmax(axis=1)):
        confidence = float(logits[index].max())
        if (
            confidence <= box_threshold
            or float(scores[index, category_index]) <= text_threshold
        ):
            continue
        cx, cy, bw, bh = boxes[index]
        x1, x2 = (cx - bw / 2) * width, (cx + bw / 2) * width
        y1, y2 = (cy - bh / 2) * height, (cy + bh / 2) * height
        if x2 <= x1 or y2 <= y1:
            continue
        result.append(
            GroundedBox(
                categories[category_index],
                (float(x1), float(y1), float(x2), float(y2)),
                confidence,
            )
        )
    return tuple(result)


def grounding_inference_state(
    state: Mapping[str, Any],
    *,
    labelbook_size: int,
    hidden_dim: int,
    max_positions: int,
) -> dict[str, Any]:
    """Remove two verified non-inference entries from the released DINO weights.

    The official inference builder fixes dn_number=0 and has no training label
    embedding. Modern BERT registers its deterministic position IDs as a
    nonpersistent buffer. All remaining parameters still load with strict=True.
    """
    result = dict(state)
    label_key = "label_enc.weight"
    if label_key in result:
        # Released Swin-T OGC weights have 2001 rows for labelbook_size=2000.
        expected_shape = (labelbook_size + 1, hidden_dim)
        if tuple(result[label_key].shape) != expected_shape:
            raise ValueError(
                f"GroundingDINO training label embedding shape={tuple(result[label_key].shape)}"
                f" differs from expected={expected_shape}"
            )
        del result[label_key]
        logger.info("checkpoint entry=%s excluded: denoising training only", label_key)
    position_key = "bert.embeddings.position_ids"
    if position_key in result:
        positions = np.asarray(result[position_key].cpu())
        if not np.array_equal(positions, np.arange(max_positions)[None]):
            raise ValueError("GroundingDINO BERT position IDs differ from fixed buffer")
        del result[position_key]
        logger.info(
            "checkpoint entry=%s verified: deterministic BERT buffer", position_key
        )
    return result


class GroundingDetector:
    """Run every fixed category in bounded captions, without goal-conditioned pruning."""

    def __init__(self, config: ScenePerceptionConfig) -> None:
        torch = importlib.import_module("torch")
        models = importlib.import_module("groundingdino.models")
        slconfig = importlib.import_module("groundingdino.util.slconfig")
        misc = importlib.import_module("groundingdino.util.misc")
        transforms = importlib.import_module("groundingdino.datasets.transforms")
        options = slconfig.SLConfig.fromfile(str(config.dino_config))
        options.device = "cpu"
        options.text_encoder_type = str(config.dino_text_encoder_dir)
        self.model: Any = models.build_model(options)
        state = torch.load(
            str(config.dino_checkpoint), map_location="cpu", weights_only=True
        )
        inference_state = grounding_inference_state(
            misc.clean_state_dict(state["model"]),
            labelbook_size=options.dn_labelbook_size,
            hidden_dim=self.model.hidden_dim,
            max_positions=self.model.bert.config.max_position_embeddings,
        )
        self.model.load_state_dict(inference_state, strict=True)
        self.model.eval()
        self.transform: Any = transforms.Compose(
            [
                transforms.RandomResize([800], max_size=1333),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.config = config

    def predict(
        self, rgb: NDArray[np.uint8]
    ) -> tuple[tuple[GroundedBox, ...], int, int, int]:
        image_module = importlib.import_module("PIL.Image")
        tensor, _ = self.transform(image_module.fromarray(rgb), None)
        predictions: list[GroundedBox] = []
        with active_model(self.model, self.config.device) as model:
            image = tensor[None].to(self.config.device)
            for categories in detection_batches(self.config.caption_batch_size):
                phrases = tuple(category_prompt(category) for category in categories)
                caption = ". ".join(phrases) + "."
                tokenized = model.tokenizer(caption, return_offsets_mapping=True)
                if len(tokenized["input_ids"]) > model.max_text_len:
                    raise ValueError(
                        "fixed vocabulary batch exceeds detector token limit; reduce caption_batch_size"
                    )
                spans = token_spans_for_categories(
                    caption, phrases, tokenized["offset_mapping"]
                )
                output = model(image, captions=[caption])
                logits = output["pred_logits"][0].sigmoid().float().cpu().numpy()
                boxes = output["pred_boxes"][0].float().cpu().numpy()
                predictions.extend(
                    decode_grounded_boxes(
                        logits,
                        boxes,
                        categories,
                        spans,
                        image_size=(rgb.shape[1], rgb.shape[0]),
                        box_threshold=self.config.box_threshold,
                        text_threshold=self.config.text_threshold,
                    )
                )
                del output
            del image
        selected, suppressed, capped = suppress_boxes(
            predictions,
            iou_threshold=self.config.nms_iou_threshold,
            limit=self.config.max_detections,
        )
        return selected, len(predictions), suppressed, capped


class PromptedSegmenter:
    """SAM1 box prompts, with one cached image embedding per observed frame."""

    def __init__(self, config: ScenePerceptionConfig) -> None:
        module = importlib.import_module("segment_anything")
        self.model: Any = module.sam_model_registry[config.sam_model_type](
            checkpoint=str(config.sam_checkpoint)
        ).eval()
        self.predictor: Any = module.SamPredictor(self.model)
        self.device = config.device

    def predict(
        self, rgb: NDArray[np.uint8], boxes: Sequence[GroundedBox]
    ) -> tuple[tuple[MaskedInstance, ...], int]:
        if not boxes:
            return (), 0
        instances: list[MaskedInstance] = []
        empty_masks = 0
        with active_model(self.model, self.device):
            try:
                self.predictor.set_image(rgb, image_format="RGB")
                for box in boxes:
                    masks, qualities, _ = self.predictor.predict(
                        box=np.asarray(box.xyxy, dtype=np.float32),
                        multimask_output=True,
                    )
                    if (
                        masks.shape != (3, *rgb.shape[:2])
                        or qualities.shape != (3,)
                        or not np.isfinite(qualities).all()
                    ):
                        raise ValueError("SAM1 returned invalid mask shape or quality")
                    mask = np.asarray(masks[int(np.argmax(qualities))], dtype=bool)
                    if not mask.any():
                        empty_masks += 1
                        continue
                    instances.append(
                        MaskedInstance(box.category, mask, box.confidence, box.xyxy)
                    )
            finally:
                self.predictor.reset_image()
        return tuple(instances), empty_masks


class ClipEncoder:
    """Encode padded RGB detection crops, class text and full frames in CLIP space."""

    def __init__(self, config: ScenePerceptionConfig) -> None:
        module = importlib.import_module("open_clip")
        self.model, _, self.preprocess = module.create_model_and_transforms(
            config.clip_model_name, pretrained=str(config.clip_checkpoint), device="cpu"
        )
        self.model.eval()
        self.tokenizer: Any = module.get_tokenizer(config.clip_model_name)
        self.device = config.device
        torch = importlib.import_module("torch")
        with active_model(self.model, self.device) as model:
            tokens = self.tokenizer(
                [category_prompt(category) for category in ALFRED_CATEGORIES]
            ).to(self.device)
            encoded = model.encode_text(tokens).float()
            encoded = torch.nn.functional.normalize(encoded, dim=-1)
            self.text_features = encoded.cpu().numpy().astype(np.float32)
            del tokens, encoded
        self.category_indices = {
            category: index for index, category in enumerate(ALFRED_CATEGORIES)
        }

    def predict(
        self, rgb: NDArray[np.uint8], instances: Sequence[MaskedInstance]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        torch = importlib.import_module("torch")
        image_module = importlib.import_module("PIL.Image")
        image = image_module.fromarray(rgb)
        crops = [image]
        for instance in instances:
            crops.append(padded_detection_crop(image, instance.bbox_xyxy))
        chunks: list[NDArray[np.float32]] = []
        with active_model(self.model, self.device) as model:
            for start in range(0, len(crops), 8):
                batch = torch.stack(
                    [self.preprocess(crop) for crop in crops[start : start + 8]]
                ).to(self.device)
                encoded = model.encode_image(batch).float()
                encoded = torch.nn.functional.normalize(encoded, dim=-1)
                chunks.append(encoded.cpu().numpy().astype(np.float32))
                del batch, encoded
        features = np.concatenate(chunks)
        indices = [self.category_indices[instance.category] for instance in instances]
        return features[1:], self.text_features[indices], features[0]
