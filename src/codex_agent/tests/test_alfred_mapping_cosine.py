"""Cosine references for normalized CLIP features with finite-precision drift."""

import numpy as np
import pytest
from numpy.typing import NDArray

from codex_agent.alfred.object_mapping import feature_cosine


def test_near_unit_roundoff_does_not_cross_strict_merge_threshold() -> None:
    first = np.array([1.0, 0.0])
    second = 1.00009 * np.array([0.8, 0.6])

    # Both vectors satisfy the observation boundary's unit-norm tolerance.
    # Their angle is the 3-4-5 triangle regardless of the small scale drift.
    assert np.isclose(np.linalg.norm(second), 1.0, atol=1e-4)
    assert float(first @ second) > 0.8
    similarity = feature_cosine(first, second)
    assert similarity == pytest.approx(0.8, abs=1e-14)
    assert not similarity > 0.8


def test_near_unit_roundoff_does_not_hide_a_match_above_threshold() -> None:
    cosine = 0.80001
    first = np.array([1.0, 0.0])
    second = 0.99991 * np.array([cosine, np.sqrt(1.0 - cosine**2)])

    assert float(first @ second) < 0.8
    assert feature_cosine(first, second) == pytest.approx(cosine, abs=1e-14)
    assert feature_cosine(first, second) > 0.8


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]])),
        (np.empty(0), np.empty(0)),
    ],
)
def test_feature_cosine_rejects_malformed_dimensions(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> None:
    with pytest.raises(ValueError):
        feature_cosine(first, second)


@pytest.mark.parametrize("invalid", [np.array([0.0, 0.0]), np.array([np.nan, 0.0])])
def test_feature_cosine_rejects_undefined_angles(
    invalid: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError):
        feature_cosine(np.array([1.0, 0.0]), invalid)
