"""Unit tests for oriented 3D IoU."""

from __future__ import annotations

import math

import pytest

from codex_agent.errors import Nr3dDataError
from codex_agent.nr3d.geometry import compute_oriented_iou_3d, oriented_bbox_to_corners


def test_identical_boxes_have_iou_one() -> None:
    box = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert compute_oriented_iou_3d(box, box) == pytest.approx(1.0)


def test_disjoint_boxes_have_iou_zero() -> None:
    a = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    b = [10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert compute_oriented_iou_3d(a, b) == 0.0


def test_half_offset_overlap_matches_analytic_value() -> None:
    a = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]
    # intersection 1x2x2=4, union 8+8-4=12 -> 1/3.
    assert compute_oriented_iou_3d(a, b) == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_iou_invariant_to_full_rotation_when_boxes_match() -> None:
    box = [0.5, -1.0, 0.2, 1.0, 2.0, 0.5, 0.3, -0.2, 0.1]
    assert compute_oriented_iou_3d(box, box) == pytest.approx(1.0, abs=1e-6)


def test_zero_volume_box_returns_zero() -> None:
    flat = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    solid = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert compute_oriented_iou_3d(flat, solid) == 0.0


def test_rotated_overlap_is_symmetric() -> None:
    a = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]
    b = [0.5, 0.5, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, math.pi / 4]
    assert compute_oriented_iou_3d(a, b) == pytest.approx(
        compute_oriented_iou_3d(b, a), abs=1e-9
    )


def test_wrong_length_bbox_raises() -> None:
    with pytest.raises(Nr3dDataError):
        compute_oriented_iou_3d([0.0, 0.0, 0.0], [0.0] * 9)


def test_non_finite_bbox_raises() -> None:
    bad = [float("nan"), 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    with pytest.raises(Nr3dDataError):
        compute_oriented_iou_3d(bad, [0.0] * 9)


def test_corners_shape() -> None:
    corners = oriented_bbox_to_corners([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert corners.shape == (8, 3)
