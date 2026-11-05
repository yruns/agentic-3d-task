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
    require_all_span_tokens: bool = False,
) -> tuple[GroundedBox, ...]:
    """Preserve upstream max-token scoring while binding exact category spans.

    GroundingDINO's box confidence is the maximum over the complete token row;
    both thresholds use strict greater-than comparisons. Only label binding
    differs from substring phrase matching, which confuses sliced/base labels.
    When ``require_all_span_tokens=True``, compound phrases require support
    across all constituent tokens and report the matched span score as box
    confidence to prevent modifier tokens (e.g. ``sliced``) or punctuation from
    inflating unrelated false positives.
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
    if require_all_span_tokens:
        span_scores: list[NDArray[np.floating[Any]]] = []
        for cat, span in zip(categories, spans, strict=True):
            sub = logits[:, list(span)]
            if len(span) == 1:
                span_scores.append(sub[:, 0])
            else:
                min_tok = sub.min(axis=1)
                mean_tok = sub.mean(axis=1)
                max_tok = sub.max(axis=1)
                min_req = 0.16 if cat.endswith("Sliced") else 0.10
                combined = np.where(
                    min_tok >= min_req, 0.5 * mean_tok + 0.5 * max_tok, 0.0
                )
                span_scores.append(combined)
        scores = np.stack(span_scores, axis=1)
    else:
        scores = np.stack([logits[:, list(span)].max(axis=1) for span in spans], axis=1)
    width, height = image_size
    result: list[GroundedBox] = []
    for index, category_index in enumerate(scores.argmax(axis=1)):
        span_conf = float(scores[index, category_index])
        confidence = (
            span_conf if require_all_span_tokens else float(logits[index].max())
        )
        if confidence <= box_threshold or span_conf <= text_threshold:
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
        options.use_checkpoint = False
        options.use_transformer_ckpt = False
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
        self._batch_specs: list[
            tuple[tuple[str, ...], str, tuple[tuple[int, ...], ...]]
        ] = []
        for categories in detection_batches(self.config.caption_batch_size):
            phrases = tuple(category_prompt(category) for category in categories)
            caption = ". ".join(phrases) + "."
            tokenized = self.model.tokenizer(caption, return_offsets_mapping=True)
            if len(tokenized["input_ids"]) > self.model.max_text_len:
                raise ValueError(
                    "fixed vocabulary batch exceeds detector token limit; reduce caption_batch_size"
                )
            spans = token_spans_for_categories(
                caption, phrases, tokenized["offset_mapping"]
            )
            self._batch_specs.append((categories, caption, spans))
        self._cached_text_dict: dict[str, Any] | None = None

    def _ensure_cached_text_dict(self, model: Any) -> dict[str, Any]:
        if self._cached_text_dict is not None:
            return {k: v.clone() for k, v in self._cached_text_dict.items()}
        bertwarper = importlib.import_module(
            "groundingdino.models.GroundingDINO.bertwarper"
        )
        captions = [spec[1] for spec in self._batch_specs]
        tokenized = model.tokenizer(
            captions, padding="longest", return_tensors="pt"
        ).to(self.config.device)
        (
            text_self_attention_masks,
            position_ids,
            _,
        ) = bertwarper.generate_masks_with_special_tokens_and_transfer_map(
            tokenized, model.specical_tokens, model.tokenizer
        )
        if text_self_attention_masks.shape[1] > model.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : model.max_text_len, : model.max_text_len
            ]
            position_ids = position_ids[:, : model.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : model.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][
                :, : model.max_text_len
            ]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][
                :, : model.max_text_len
            ]
        if model.sub_sentence_present:
            tokenized_for_encoder = {
                k: v for k, v in tokenized.items() if k != "attention_mask"
            }
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized
        bert_output = model.bert(**tokenized_for_encoder)
        encoded_text = model.feat_map(bert_output["last_hidden_state"])
        text_token_mask = tokenized.attention_mask.bool()
        if encoded_text.shape[1] > model.max_text_len:
            encoded_text = encoded_text[:, : model.max_text_len, :]
            text_token_mask = text_token_mask[:, : model.max_text_len]
            position_ids = position_ids[:, : model.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : model.max_text_len, : model.max_text_len
            ]
        self._cached_text_dict = {
            "encoded_text": encoded_text.detach(),
            "text_token_mask": text_token_mask.detach(),
            "position_ids": position_ids.detach(),
            "text_self_attention_masks": text_self_attention_masks.detach(),
        }
        return {k: v.clone() for k, v in self._cached_text_dict.items()}

    def predict(
        self, rgb: NDArray[np.uint8]
    ) -> tuple[tuple[GroundedBox, ...], int, int, int]:
        torch = importlib.import_module("torch")
        F = importlib.import_module("torch.nn.functional")
        misc = importlib.import_module("groundingdino.util.misc")
        image_module = importlib.import_module("PIL.Image")
        tensor, _ = self.transform(image_module.fromarray(rgb), None)
        predictions: list[GroundedBox] = []
        with (
            torch.no_grad(),
            active_model(self.model, self.config.device) as model,
        ):
            image = tensor[None].to(self.config.device)
            samples = misc.nested_tensor_from_tensor_list(image)
            features, poss = model.backbone(samples)
            srcs: list[Any] = []
            masks: list[Any] = []
            for level, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(model.input_proj[level](src))
                masks.append(mask)
            if model.num_feature_levels > len(srcs):
                len_srcs = len(srcs)
                for level in range(len_srcs, model.num_feature_levels):
                    if level == len_srcs:
                        src = model.input_proj[level](features[-1].tensors)
                    else:
                        src = model.input_proj[level](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                        torch.bool
                    )[0]
                    pos_l = model.backbone[1](misc.NestedTensor(src, mask)).to(
                        src.dtype
                    )
                    srcs.append(src)
                    masks.append(mask)
                    poss.append(pos_l)
            bs = len(self._batch_specs)
            srcs_b = [s.expand(bs, -1, -1, -1).contiguous() for s in srcs]
            masks_b = [m.expand(bs, -1, -1).contiguous() for m in masks]
            poss_b = [p.expand(bs, -1, -1, -1).contiguous() for p in poss]
            text_dict = self._ensure_cached_text_dict(model)
            hs, reference, _, _, _ = model.transformer(
                srcs_b, masks_b, None, poss_b, None, None, text_dict
            )
            last_coord = (
                model.bbox_embed[-1](hs[-1]) + misc.inverse_sigmoid(reference[-2])
            ).sigmoid()
            last_class = model.class_embed[-1](hs[-1], text_dict).sigmoid()
            logits_all = last_class.detach().float().cpu().numpy()
            boxes_all = last_coord.detach().float().cpu().numpy()
            for batch_idx, (categories, _, spans) in enumerate(self._batch_specs):
                logits = logits_all[batch_idx, :, : logits_all.shape[2]]
                boxes = boxes_all[batch_idx]
                predictions.extend(
                    decode_grounded_boxes(
                        logits,
                        boxes,
                        categories,
                        spans,
                        image_size=(rgb.shape[1], rgb.shape[0]),
                        box_threshold=self.config.box_threshold,
                        text_threshold=self.config.text_threshold,
                        require_all_span_tokens=True,
                    )
                )
            del image, samples, features, poss, srcs, masks, hs, reference
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
        torch = importlib.import_module("torch")
        instances: list[MaskedInstance] = []
        empty_masks = 0
        use_cuda_amp = self.device.startswith("cuda")
        amp_dtype = (
            torch.bfloat16
            if use_cuda_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with (
            torch.no_grad(),
            active_model(self.model, self.device),
            torch.autocast(
                device_type=self.device.split(":")[0],
                dtype=amp_dtype,
                enabled=use_cuda_amp,
            ),
        ):
            try:
                self.predictor.set_image(rgb, image_format="RGB")
                if hasattr(self.predictor, "predict_torch") and hasattr(
                    self.predictor, "transform"
                ):
                    box_tensor = torch.as_tensor(
                        [box.xyxy for box in boxes],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    transformed_boxes = self.predictor.transform.apply_boxes_torch(
                        box_tensor, rgb.shape[:2]
                    )
                    masks_t, qualities_t, _ = self.predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=True,
                    )
                    masks_np = masks_t.detach().cpu().numpy()
                    qualities_np = qualities_t.detach().float().cpu().numpy()
                    for idx, box in enumerate(boxes):
                        masks = masks_np[idx]
                        qualities = qualities_np[idx]
                        if (
                            masks.shape != (3, *rgb.shape[:2])
                            or qualities.shape != (3,)
                            or not np.isfinite(qualities).all()
                        ):
                            raise ValueError(
                                "SAM1 returned invalid mask shape or quality"
                            )
                        mask = np.asarray(masks[int(np.argmax(qualities))], dtype=bool)
                        if not mask.any():
                            empty_masks += 1
                            continue
                        instances.append(
                            MaskedInstance(box.category, mask, box.confidence, box.xyxy)
                        )
                else:
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
                            raise ValueError(
                                "SAM1 returned invalid mask shape or quality"
                            )
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
        use_cuda_amp = self.device.startswith("cuda")
        amp_dtype = (
            torch.bfloat16
            if use_cuda_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with (
            torch.no_grad(),
            active_model(self.model, self.device) as model,
            torch.autocast(
                device_type=self.device.split(":")[0],
                dtype=amp_dtype,
                enabled=use_cuda_amp,
            ),
        ):
            for start in range(0, len(crops), 32):
                batch = torch.stack(
                    [self.preprocess(crop) for crop in crops[start : start + 32]]
                ).to(self.device)
                encoded = model.encode_image(batch).float()
                encoded = torch.nn.functional.normalize(encoded, dim=-1)
                chunks.append(encoded.cpu().numpy().astype(np.float32))
                del batch, encoded
        features = np.concatenate(chunks)
        indices = [self.category_indices[instance.category] for instance in instances]
        return features[1:], self.text_features[indices], features[0]
