"""Tests for SceneFunc3D target-anchor construction."""

from __future__ import annotations

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.anchor import TargetAnchor, build_anchor
from codex_agent.scenefunc3d.tools.models import ToolInputError


def test_anchor_centroid_is_point_mean() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    anchor = build_anchor(points, motion_type="pinch_pull", seed_frame_id="000012")
    assert isinstance(anchor, TargetAnchor)
    assert anchor.centroid == pytest.approx((1.0 / 3.0, 1.0 / 3.0, 0.0))
    assert anchor.seed_frame_id == "000012"
    assert anchor.source_point_count == 3


def test_anchor_radius_capped_by_motion_prior() -> None:
    # Points spread ~1 m from centroid; pinch_pull caps radius at 0.20 m.
    points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    anchor = build_anchor(points, motion_type="pinch_pull", seed_frame_id="000001")
    assert anchor.radius_m == pytest.approx(0.20)


def test_anchor_radius_uses_percentile_when_below_cap() -> None:
    points = np.zeros((100, 3))
    points[:, 0] = np.linspace(0.0, 0.10, 100)  # spread 0.10 m along x
    anchor = build_anchor(
        points, motion_type="pinch_pull", seed_frame_id="000001", radius_percentile=95.0
    )
    assert 0.02 < anchor.radius_m < 0.20


def test_empty_points_raise() -> None:
    with pytest.raises(ToolInputError):
        build_anchor(np.zeros((0, 3)), motion_type="rotate", seed_frame_id="000001")
