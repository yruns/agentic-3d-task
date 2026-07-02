"""Tests for SceneFunc3D 3D->2D visibility projection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.anchor import TargetAnchor, build_anchor
from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry
from codex_agent.scenefunc3d.backends.visibility import (
    FrameCamera,
    FrameVisibility,
    count_vertex_visibility,
    point_visibility,
    project_world_to_pixels,
    score_anchor_visibility,
    select_scene_visible_frames,
    select_visible_frames,
)
from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene
from codex_agent.tests.scenefunc3d_synthetic_scene import (
    SyntheticFrame,
    write_synthetic_scene,
)


def _identity_camera(
    fx: float = 100.0, cx: float = 50.0, cy: float = 50.0
) -> CameraGeometry:
    intrinsics = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])
    return CameraGeometry(intrinsics=intrinsics, camera_to_world=np.eye(4))


def _flat_depth(value: float = 2.0, size: int = 100) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float64)


def _centered_anchor() -> TargetAnchor:
    # Symmetric points -> centroid exactly (0, 0, 2) -> projects to (50, 50),
    # so centeredness is exactly 1.0.
    points = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.02, 0.0, 2.0],
            [-0.02, 0.0, 2.0],
            [0.0, 0.02, 2.0],
            [0.0, -0.02, 2.0],
        ]
    )
    return build_anchor(points, motion_type="pinch_pull", seed_frame_id="000001")


def test_projects_on_axis_point_to_principal_point() -> None:
    geometry = _identity_camera()
    pixels, camera_z = project_world_to_pixels(np.array([[0.0, 0.0, 2.0]]), geometry)
    assert pixels[0] == pytest.approx((50.0, 50.0))
    assert camera_z[0] == pytest.approx(2.0)


def test_projects_off_axis_point_with_perspective() -> None:
    geometry = _identity_camera()
    pixels, _camera_z = project_world_to_pixels(np.array([[0.1, 0.0, 2.0]]), geometry)
    assert pixels[0][0] == pytest.approx(55.0)
    assert pixels[0][1] == pytest.approx(50.0)


def test_behind_camera_point_has_non_positive_z() -> None:
    geometry = _identity_camera()
    _pixels, camera_z = project_world_to_pixels(np.array([[0.0, 0.0, -1.0]]), geometry)
    assert camera_z[0] < 0.0


def test_point_on_depth_surface_is_visible() -> None:
    visible = point_visibility(
        np.array([[0.0, 0.0, 2.0]]), _identity_camera(), _flat_depth(2.0)
    )
    assert bool(visible[0]) is True


def test_point_behind_surface_is_occluded() -> None:
    visible = point_visibility(
        np.array([[0.0, 0.0, 2.0]]), _identity_camera(), _flat_depth(1.0)
    )
    assert bool(visible[0]) is False


def test_out_of_frame_point_is_not_visible() -> None:
    visible = point_visibility(
        np.array([[10.0, 0.0, 2.0]]), _identity_camera(), _flat_depth(2.0)
    )
    assert bool(visible[0]) is False


def test_behind_camera_point_is_not_visible() -> None:
    visible = point_visibility(
        np.array([[0.0, 0.0, -1.0]]), _identity_camera(), _flat_depth(2.0)
    )
    assert bool(visible[0]) is False


def test_fully_visible_centered_anchor_scores_high() -> None:
    result = score_anchor_visibility(
        "000001", _centered_anchor(), _identity_camera(), _flat_depth(2.0)
    )
    assert isinstance(result, FrameVisibility)
    assert result.visible is True
    assert result.unoccluded_fraction == pytest.approx(1.0)
    assert result.centeredness == pytest.approx(1.0)
    assert result.quality_score == pytest.approx(1.0)


def test_occluded_anchor_scores_zero_and_not_visible() -> None:
    result = score_anchor_visibility(
        "000001", _centered_anchor(), _identity_camera(), _flat_depth(1.0)
    )
    assert result.visible is False
    assert result.quality_score == pytest.approx(0.0)


def test_selection_drops_occluded_and_ranks_by_quality() -> None:
    anchor = _centered_anchor()
    geometry = _identity_camera()
    cameras = (
        FrameCamera(
            frame_id="000001", geometry=geometry, depth_meters=_flat_depth(2.0)
        ),
        FrameCamera(
            frame_id="000002", geometry=geometry, depth_meters=_flat_depth(1.0)
        ),
    )
    selected = select_visible_frames(anchor, cameras, frame_cap=10)
    assert tuple(v.frame_id for v in selected) == ("000001",)
    assert all(v.visible for v in selected)


def test_selection_respects_frame_cap() -> None:
    anchor = _centered_anchor()
    geometry = _identity_camera()
    cameras = tuple(
        FrameCamera(
            frame_id=f"{index:06d}", geometry=geometry, depth_meters=_flat_depth(2.0)
        )
        for index in range(5)
    )
    selected = select_visible_frames(anchor, cameras, frame_cap=3)
    assert len(selected) == 3


def _pipeline_scene(tmp_path: Path) -> SceneFunc3dToolScene:
    intrinsics = np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]])
    frames = (
        # 000000 sees the anchor at depth 2.0 (unoccluded).
        SyntheticFrame("000000", np.eye(4), depth_value_m=2.0),
        # 000001 has a near wall at 1.0 -> anchor at z=2.0 is occluded.
        SyntheticFrame("000001", np.eye(4), depth_value_m=1.0),
    )
    return write_synthetic_scene(
        tmp_path / "scene",
        frames=frames,
        vertices_world=np.array([[0.0, 0.0, 2.0], [0.02, 0.0, 2.0]]),
        intrinsics=intrinsics,
    )


def test_select_scene_visible_frames_streams_and_ranks(tmp_path: Path) -> None:
    scene = _pipeline_scene(tmp_path)
    anchor = build_anchor(
        np.array([[0.0, 0.0, 2.0], [0.02, 0.0, 2.0], [-0.02, 0.0, 2.0]]),
        motion_type="pinch_pull",
        seed_frame_id="000000",
    )
    selected = select_scene_visible_frames(anchor, scene, frame_cap=10)
    assert "000000" in {v.frame_id for v in selected}
    # The occluded frame must be dropped unless it is the seed frame.
    assert "000001" not in {v.frame_id for v in selected}


def test_select_scene_visible_frames_always_includes_seed(tmp_path: Path) -> None:
    scene = _pipeline_scene(tmp_path)
    # Seed is the occluded frame; it must still appear (single-frame degrade).
    anchor = build_anchor(
        np.array([[0.0, 0.0, 2.0], [0.02, 0.0, 2.0], [-0.02, 0.0, 2.0]]),
        motion_type="pinch_pull",
        seed_frame_id="000001",
    )
    selected = select_scene_visible_frames(anchor, scene, frame_cap=10)
    assert "000001" in {v.frame_id for v in selected}


def test_count_vertex_visibility_counts_unoccluded_frames(tmp_path: Path) -> None:
    scene = _pipeline_scene(tmp_path)
    # Vertex 0 at z=2.0 is visible in 000000 (depth 2.0) but occluded in 000001
    # (near wall at 1.0). Vertex ids are arbitrary raw-mesh ids for the counter.
    vertex_ids = [7, 9]
    vertex_coords = np.array([[0.0, 0.0, 2.0], [0.02, 0.0, 2.0]])
    counts = count_vertex_visibility(
        vertex_ids, vertex_coords, scene, ("000000", "000001")
    )
    assert counts[7] == 1
    assert counts[9] == 1


def test_count_vertex_visibility_accumulates_across_frames(tmp_path: Path) -> None:
    intrinsics = np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]])
    frames = (
        SyntheticFrame("000000", np.eye(4), depth_value_m=2.0),
        SyntheticFrame("000001", np.eye(4), depth_value_m=2.0),
    )
    scene = write_synthetic_scene(
        tmp_path / "scene",
        frames=frames,
        vertices_world=np.array([[0.0, 0.0, 2.0]]),
        intrinsics=intrinsics,
    )
    counts = count_vertex_visibility(
        [5], np.array([[0.0, 0.0, 2.0]]), scene, ("000000", "000001")
    )
    assert counts[5] == 2


def test_count_vertex_visibility_rejects_length_mismatch(tmp_path: Path) -> None:
    scene = _pipeline_scene(tmp_path)
    with pytest.raises(ValueError):
        count_vertex_visibility([1], np.zeros((2, 3)), scene, ("000000",))
