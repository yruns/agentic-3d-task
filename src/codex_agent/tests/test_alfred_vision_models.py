"""Deterministic grounding label selection and suppression boundaries."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from PIL import Image

from codex_agent.alfred.perception import ScenePerceptionConfig
from codex_agent.alfred.vision_models import (
    ClipEncoder,
    GroundedBox,
    MaskedInstance,
    PromptedSegmenter,
    decode_grounded_boxes,
    grounding_inference_state,
    padded_detection_crop,
    suppress_boxes,
    token_spans_for_categories,
)
from codex_agent.alfred.vocabulary import ALFRED_CATEGORIES, category_prompt


def test_caption_offsets_keep_compound_category_identity() -> None:
    caption = "apple. sliced apple. faucet."
    offsets = [
        (0, 0),
        (0, 5),
        (5, 6),
        (7, 13),
        (14, 19),
        (19, 20),
        (21, 27),
        (27, 28),
        (0, 0),
    ]
    spans = token_spans_for_categories(
        caption, ("apple", "sliced apple", "faucet"), offsets
    )
    assert spans == ((1,), (3, 4), (6,))


def test_suppression_preserves_different_instances_and_reports_all_drops() -> None:
    boxes = [
        GroundedBox("Apple", (0, 0, 10, 10), 0.9),
        GroundedBox("Apple", (1, 1, 10, 10), 0.8),
        GroundedBox("AppleSliced", (1, 1, 10, 10), 0.7),
        GroundedBox("Faucet", (20, 20, 25, 25), 0.6),
    ]
    selected, suppressed, limited = suppress_boxes(boxes, iou_threshold=0.5, limit=1)
    assert [box.category for box in selected] == ["Apple"]
    assert (suppressed, limited) == (2, 1)
    assert np.isfinite([box.confidence for box in selected]).all()


def test_grounding_uses_upstream_max_confidence_and_strict_thresholds() -> None:
    # The second word is weak, but upstream's max-token box confidence is .8.
    probabilities = np.array(
        [[0, 0.3, 0, 0.8, 0.01], [0, 0.2, 0, 0.1, 0.1]], dtype=np.float64
    )
    boxes = np.array([[0.1, 0.5, 0.4, 0.5], [0.5, 0.5, 0.2, 0.2]])
    result = decode_grounded_boxes(
        probabilities,
        boxes,
        ("Apple", "AppleSliced"),
        ((1,), (3, 4)),
        image_size=(100, 80),
        box_threshold=0.2,
        text_threshold=0.2,
    )
    assert len(result) == 1
    assert result[0].category == "AppleSliced"
    assert result[0].confidence == pytest.approx(0.8)
    # Upstream box conversion retains the detector box outside the image.
    assert result[0].xyxy == pytest.approx((-10, 20, 30, 60))
    assert not decode_grounded_boxes(
        probabilities[:1],
        boxes[:1],
        ("Apple", "AppleSliced"),
        ((1,), (3, 4)),
        image_size=(100, 80),
        box_threshold=0.2,
        text_threshold=0.8,
    )


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((25.2, 30.7, 40.8, 45.1), (5.2, 10.7, 60.8, 65.1)),
        ((-3.0, 5.0, 85.0, 78.0), (0.0, 0.0, 100.0, 80.0)),
    ],
)
def test_clip_crop_matches_padded_detector_box_with_original_background(
    box: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
) -> None:
    rgb = np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3)
    # Explicit reference bounds reproduce legacy PIL crop rounding, not the
    # production helper, and retain every surrounding background pixel.
    reference = Image.fromarray(rgb).crop(expected)
    actual = padded_detection_crop(Image.fromarray(rgb), box)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))


def test_suppression_matches_torchvision_class_agnostic_reference() -> None:
    import torch
    from torchvision.ops import nms

    boxes = [
        GroundedBox("Apple", (-2, 0, 10, 10), 0.7),
        GroundedBox("AppleSliced", (0, 0, 10, 10), 0.9),
        GroundedBox("Faucet", (20, 20, 30, 30), 0.8),
        GroundedBox("SinkBasin", (21, 21, 31, 31), 0.6),
    ]
    expected = nms(
        torch.tensor([box.xyxy for box in boxes], dtype=torch.float32),
        torch.tensor([box.confidence for box in boxes]),
        0.5,
    ).tolist()
    actual, suppressed, capped = suppress_boxes(boxes, iou_threshold=0.5, limit=100)
    assert actual == tuple(boxes[index] for index in expected)
    assert suppressed == len(boxes) - len(expected)
    assert capped == 0


def test_clip_inputs_match_legacy_class_text_and_unmasked_padded_crops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    token_inputs: list[list[str]] = []

    class ReferenceEncoder(torch.nn.Module):
        def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
            return tokens

        def encode_image(self, images: torch.Tensor) -> torch.Tensor:
            return images

    def tokenize(classes: list[str]) -> torch.Tensor:
        token_inputs.append(classes)
        return torch.tensor([[index + 1, 2, 3] for index in range(len(classes))])

    def preprocess(image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.asarray(image).mean(axis=(0, 1)).astype(np.float32))

    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        SimpleNamespace(
            create_model_and_transforms=lambda *args, **kwargs: (
                ReferenceEncoder(),
                None,
                preprocess,
            ),
            get_tokenizer=lambda name: tokenize,
        ),
    )
    config = cast(
        ScenePerceptionConfig,
        SimpleNamespace(
            clip_model_name="reference",
            clip_checkpoint=Path("unused.safetensors"),
            device="cpu",
        ),
    )
    encoder = ClipEncoder(config)
    assert token_inputs == [[category_prompt(name) for name in ALFRED_CATEGORIES]]
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    rgb[:, :] = [10, 60, 120]
    rgb[35:40, 30:35] = [200, 10, 20]
    mask = np.zeros((80, 100), dtype=bool)
    mask[35:40, 30:35] = True
    image_features, text_features, frame_feature = encoder.predict(
        rgb, (MaskedInstance("AppleSliced", mask, 0.8, (25, 30, 40, 45)),)
    )
    # Independent reference: RGB region from original box + 20px, not mask.
    reference = rgb[10:65, 5:60].mean(axis=(0, 1))
    np.testing.assert_allclose(image_features[0], reference / np.linalg.norm(reference))
    full = rgb.mean(axis=(0, 1))
    np.testing.assert_allclose(frame_feature, full / np.linalg.norm(full))
    category_index = ALFRED_CATEGORIES.index("AppleSliced")
    text = np.array([category_index + 1, 2, 3], dtype=np.float32)
    np.testing.assert_allclose(text_features[0], text / np.linalg.norm(text))


def test_sam_selects_best_quality_mask_and_retains_detector_box() -> None:
    import torch

    class ReferencePredictor:
        def set_image(self, rgb: np.ndarray, *, image_format: str) -> None:
            assert image_format == "RGB"

        def predict(
            self, *, box: np.ndarray, multimask_output: bool
        ) -> tuple[np.ndarray, np.ndarray, None]:
            assert multimask_output is True
            np.testing.assert_array_equal(box, [-2, 1, 7.5, 9])
            masks = np.zeros((3, 10, 10), dtype=bool)
            masks[0, 2, 2] = True
            masks[1, 4:6, 4:6] = True
            masks[2, 7, 7] = True
            return masks, np.array([0.2, 0.9, 0.3]), None

        def reset_image(self) -> None:
            pass

    segmenter = object.__new__(PromptedSegmenter)
    segmenter.model = torch.nn.Identity()
    segmenter.predictor = ReferencePredictor()
    segmenter.device = "cpu"
    box = GroundedBox("Faucet", (-2, 1, 7.5, 9), 0.7)
    instances, empty = segmenter.predict(np.zeros((10, 10, 3), dtype=np.uint8), (box,))
    assert empty == 0
    assert len(instances) == 1
    assert instances[0].bbox_xyxy == box.xyxy
    expected = np.zeros((10, 10), dtype=bool)
    expected[4:6, 4:6] = True
    np.testing.assert_array_equal(instances[0].mask, expected)


def test_released_dino_training_entries_are_verified_before_strict_loading() -> None:
    import torch

    state = {
        "label_enc.weight": torch.zeros(2001, 256),
        "bert.embeddings.position_ids": torch.arange(512)[None],
        "inference.weight": torch.ones(2, 3),
        "unexpected.weight": torch.zeros(1),
    }
    actual = grounding_inference_state(
        state, labelbook_size=2000, hidden_dim=256, max_positions=512
    )
    # Unexpected inference entries remain present so strict loading rejects them.
    assert set(actual) == {"inference.weight", "unexpected.weight"}
    assert set(state) == {
        "label_enc.weight",
        "bert.embeddings.position_ids",
        "inference.weight",
        "unexpected.weight",
    }
    state["bert.embeddings.position_ids"][0, 1] = 99
    with pytest.raises(ValueError, match="position IDs"):
        grounding_inference_state(
            state, labelbook_size=2000, hidden_dim=256, max_positions=512
        )


def test_require_all_span_tokens_prevents_modifier_token_stealing() -> None:
    # Token 1 ("apple") is 0.45; token 3 ("sliced") is 0.80 but token 4 ("apple") is 0.01.
    probabilities = np.array([[0.95, 0.45, 0.0, 0.80, 0.01]], dtype=np.float64)
    boxes = np.array([[0.5, 0.5, 0.4, 0.5]])
    result = decode_grounded_boxes(
        probabilities,
        boxes,
        ("Apple", "AppleSliced"),
        ((1,), (3, 4)),
        image_size=(100, 80),
        box_threshold=0.25,
        text_threshold=0.25,
        require_all_span_tokens=True,
    )
    assert len(result) == 1
    assert result[0].category == "Apple"
    assert result[0].confidence == pytest.approx(0.45)
