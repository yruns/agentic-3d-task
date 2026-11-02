"""Tests for the semi-online SceneFunc3D pipeline harness (fake proposer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.anchor import build_anchor
from codex_agent.scenefunc3d.backends.fusion import FusionParams
from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry
from codex_agent.scenefunc3d.evaluation.semi_online_pipeline import (
    score_semi_online_sample,
)
from codex_agent.scenefunc3d.pipeline import FrameProposal
from codex_agent.tests.scenefunc3d_synthetic_scene import (
    SyntheticFrame,
    write_synthetic_scene,
)


class _FakeProposer:
    def __init__(self, indices: tuple[int, ...]) -> None:
        self._indices = indices

    def propose(
        self,
        *,
        frame_id: str,
        geometry: CameraGeometry,
        projected_anchor_xy: tuple[float, float] | None,
    ) -> FrameProposal:
        return FrameProposal(
            frame_id=frame_id, point_indices=self._indices, molmo_fallback_used=False
        )


def test_score_semi_online_sample_recovers_target(tmp_path: Path) -> None:
    intrinsics = np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]])
    vertices = np.array(
        [
            [0.00, 0.0, 2.0],
            [0.01, 0.0, 2.0],
            [0.00, 0.01, 2.0],
            [0.01, 0.01, 2.0],
            [0.005, 0.005, 2.0],
            [3.00, 3.0, 2.0],
        ]
    )
    frames = (
        SyntheticFrame("000000", np.eye(4), depth_value_m=2.0),
        SyntheticFrame("000001", np.eye(4), depth_value_m=2.0),
    )
    scene = write_synthetic_scene(
        tmp_path / "scene",
        frames=frames,
        vertices_world=vertices,
        intrinsics=intrinsics,
    )
    anchor = build_anchor(
        vertices[:5], motion_type="pinch_pull", seed_frame_id="000000"
    )
    score = score_semi_online_sample(
        sample_id="123::0",
        scene=scene,
        anchor=anchor,
        gt_ids=frozenset({0, 1, 2, 3, 4}),
        proposer=_FakeProposer((0, 1, 2, 3, 4)),
        params=FusionParams(agreement_tau=0.5, min_cluster_points=3),
        frame_cap=10,
    )
    assert score.metrics.iou == pytest.approx(1.0)
