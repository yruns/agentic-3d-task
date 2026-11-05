"""Metric scale and camera boundaries without loading neural checkpoints."""

from __future__ import annotations

import numpy as np
import pytest

from codex_agent.alfred.depth_models import (
    canonical_depth_to_meters,
    validate_intrinsics,
)


def test_metric_conversion_uses_resized_focal_exactly_once() -> None:
    original = np.array([[260.0, 0, 149.5], [0, 260.0, 149.5], [0, 0, 1]])
    resized = original.copy()
    resized[:2] *= 504 / 300
    canonical = np.full((504, 504), 2 * 300 / resized[0, 0], dtype=np.float32)
    actual = canonical_depth_to_meters(canonical, resized)
    np.testing.assert_allclose(actual, 2.0, rtol=1e-6)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(original[2], [0, 0, 1])


@pytest.mark.parametrize("value", [float("nan"), 0.0, -1.0])
def test_rejects_nonmetric_depth(value: float) -> None:
    with pytest.raises(ValueError, match="depth"):
        canonical_depth_to_meters(np.full((2, 3), value), np.eye(3))


def test_rejects_invalid_camera_without_guessing_focal() -> None:
    for matrix in (np.zeros((3, 3)), np.eye(4), np.full((3, 3), np.nan)):
        with pytest.raises(ValueError, match="intrinsics"):
            validate_intrinsics(matrix)


def test_da3_adapter_restores_pixels_without_second_metric_scaling() -> None:
    from types import SimpleNamespace

    from codex_agent.alfred.depth_models import DA3MetricDepth

    torch = pytest.importorskip("torch")
    camera = np.array([[260, 0, 149.5], [0, 260, 149.5], [0, 0, 1]], dtype=np.float32)
    calls: list[str] = []

    class ExternalModel:
        def input_processor(self, images: object, **kwargs: object) -> object:
            intrinsics = np.asarray(kwargs["intrinsics"]).copy()
            intrinsics[:, :2] *= 504 / 300
            return torch.zeros((1, 3, 504, 504)), None, torch.from_numpy(intrinsics)

        def to(self, device: str) -> ExternalModel:
            assert device == "cpu"
            return self

        def forward(self, image: object, **kwargs: object) -> dict[str, object]:
            calls.append("canonical_forward")
            return {
                "depth": np.full(
                    (1, 504, 504), 2 * 300 / (260 * 504 / 300), dtype=np.float32
                )
            }

        def output_processor(self, output: dict[str, object]) -> SimpleNamespace:
            return SimpleNamespace(depth=output["depth"])

    predictor = DA3MetricDepth.__new__(DA3MetricDepth)
    predictor.model = ExternalModel()
    predictor.input_processor = predictor.model.input_processor
    predictor.output_processor = predictor.model.output_processor
    predictor.device = "cpu"
    depth = predictor.predict(np.zeros((300, 300, 3), dtype=np.uint8), camera)
    assert depth.shape == (300, 300)
    np.testing.assert_allclose(depth, 2.0, rtol=1e-6)
    np.testing.assert_array_equal(camera[0], [260, 0, 149.5])
    assert calls == ["canonical_forward"]


def test_da3_checkpoint_mapping_is_exact_and_rejects_unexpected_prefixes() -> None:
    from codex_agent.alfred.depth_models import strip_da3_wrapper_prefix

    assert strip_da3_wrapper_prefix({"model.head.weight": 1}) == {"head.weight": 1}
    with pytest.raises(ValueError, match="wrapper prefix"):
        strip_da3_wrapper_prefix({"other.head.weight": 1})
