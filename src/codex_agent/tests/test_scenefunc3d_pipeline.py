"""Tests for the pure anchor multi-view pipeline driver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.anchor import build_anchor
from codex_agent.scenefunc3d.backends.fusion import FusionParams
from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry, FloatArray
from codex_agent.scenefunc3d.backends.visibility import FrameVisibility
from codex_agent.scenefunc3d.pipeline import (
    FrameProposal,
    fuse_selected_frames,
)

_INTRINSICS = np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]])


def _geometry() -> CameraGeometry:
    return CameraGeometry(intrinsics=_INTRINSICS, camera_to_world=np.eye(4))


class _FakeProposer:
    """Returns pre-baked point ids per frame; records projected anchors seen."""

    def __init__(self, per_frame: Mapping[str, tuple[int, ...]]) -> None:
        self._per_frame = per_frame
        self.seen_projected: list[tuple[float, float] | None] = []

    def propose(
        self,
        *,
        frame_id: str,
        geometry: CameraGeometry,
        projected_anchor_xy: tuple[float, float] | None,
    ) -> FrameProposal:
        self.seen_projected.append(projected_anchor_xy)
        return FrameProposal(
            frame_id=frame_id,
            point_indices=self._per_frame.get(frame_id, ()),
            molmo_fallback_used=False,
        )


def _all_visible_counts(
    vertex_ids: Sequence[int], vertex_coords: FloatArray, frame_ids: Sequence[str]
) -> Mapping[int, int]:
    return {int(v): len(frame_ids) for v in vertex_ids}


def test_pipeline_fuses_consensus_target_and_drops_off_anchor_frame() -> None:
    # 5 tightly-clustered target vertices near the anchor, 1 far wall vertex.
    scene_vertices = np.array(
        [
            [0.00, 0.0, 2.0],
            [0.01, 0.0, 2.0],
            [0.00, 0.01, 2.0],
            [0.01, 0.01, 2.0],
            [0.005, 0.005, 2.0],
            [3.00, 3.0, 2.0],  # id 5: far wall
        ]
    )
    anchor = build_anchor(
        scene_vertices[:5], motion_type="pinch_pull", seed_frame_id="000000"
    )
    selected = (
        FrameVisibility("000000", True, 1.0, 1.0, 1.0),
        FrameVisibility("000001", True, 1.0, 1.0, 1.0),
        FrameVisibility("000002", True, 1.0, 1.0, 1.0),
    )
    geometry_by_frame = {fid: _geometry() for fid in ("000000", "000001", "000002")}
    proposer = _FakeProposer(
        {
            "000000": (0, 1, 2, 3, 4),
            "000001": (0, 1, 2, 3, 4),
            "000002": (5,),  # off-anchor -> must be gated out
        }
    )
    result = fuse_selected_frames(
        anchor=anchor,
        scene_vertices=scene_vertices,
        selected_frames=selected,
        geometry_by_frame=geometry_by_frame,
        proposer=proposer,
        params=FusionParams(agreement_tau=0.5, min_cluster_points=3),
        visibility_counts_fn=_all_visible_counts,
    )
    assert set(result.fused.point_indices) == {0, 1, 2, 3, 4}
    rejected = {o.frame_id for o in result.per_frame if not o.accepted}
    assert rejected == {"000002"}
    # The anchor centroid (0.005, 0.005, 2.0) projects just off the principal
    # point (20, 20): u = v = 20 * 0.005 / 2.0 + 20 = 20.05.
    assert proposer.seen_projected[0] == pytest.approx((20.05, 20.05))


def test_pipeline_returns_empty_when_no_frame_accepted() -> None:
    scene_vertices = np.array([[0.0, 0.0, 2.0], [5.0, 5.0, 2.0]])
    anchor = build_anchor(
        np.array([[0.0, 0.0, 2.0]]), motion_type="rotate", seed_frame_id="000000"
    )
    selected = (FrameVisibility("000000", True, 1.0, 1.0, 1.0),)
    result = fuse_selected_frames(
        anchor=anchor,
        scene_vertices=scene_vertices,
        selected_frames=selected,
        geometry_by_frame={"000000": _geometry()},
        proposer=_FakeProposer({"000000": (1,)}),  # off-anchor only
        params=FusionParams(),
        visibility_counts_fn=_all_visible_counts,
    )
    assert result.fused.point_indices == ()
    assert all(not o.accepted for o in result.per_frame)
