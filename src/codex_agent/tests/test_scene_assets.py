"""Unit tests for auxiliary NR3D scene-asset loaders and rich scene loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_agent.errors import Nr3dDataError
from codex_agent.nr3d.sample import Nr3dScene
from codex_agent.nr3d.scene_assets import BevViewParams, CameraTrajectory
from codex_agent.tests.conftest import Nr3dToolsFixture


def test_camera_trajectory_pose_lookup(tmp_path: Path) -> None:
    path = tmp_path / "camera_trajectory.json"
    path.write_text(json.dumps({"0": [1.0, 2.0, 0.5], "3": [4.0, 5.0, 1.0]}))
    trajectory = CameraTrajectory.load(path)
    assert trajectory.pose(0) == (1.0, 2.0, 0.5)
    assert trajectory.pose(3) == (4.0, 5.0, 1.0)
    assert trajectory.pose(99) is None


def test_camera_trajectory_rejects_bad_entry(tmp_path: Path) -> None:
    path = tmp_path / "camera_trajectory.json"
    path.write_text(json.dumps({"0": [1.0, 2.0]}))
    with pytest.raises(Nr3dDataError):
        CameraTrajectory.load(path)


def test_bev_view_params_load(tmp_path: Path) -> None:
    path = tmp_path / "bev.view.json"
    path.write_text(
        json.dumps(
            {
                "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "t": [0, 0, 10],
                "f": 100.0,
                "c": 50.0,
                "image_size": 100,
                "crop_offset": [3, 4],
            }
        )
    )
    view = BevViewParams.load(path)
    assert view.focal == 100.0
    assert view.center == 50.0
    assert view.crop_offset == (3, 4)
    assert view.rotation[0] == (1.0, 0.0, 0.0)


def test_bev_view_params_rejects_bad_rotation(tmp_path: Path) -> None:
    path = tmp_path / "bev.view.json"
    path.write_text(
        json.dumps(
            {
                "R": [[1, 0, 0]],
                "t": [0, 0, 1],
                "f": 1,
                "c": 1,
                "image_size": 1,
                "crop_offset": [0, 0],
            }
        )
    )
    with pytest.raises(Nr3dDataError):
        BevViewParams.load(path)


def test_scene_load_carries_assets(nr3d_tools_fixture: Nr3dToolsFixture) -> None:
    scene = Nr3dScene.load(nr3d_tools_fixture.scene_dir)
    assert scene.valid_frame_ids == (0, 1)
    assert scene.camera_trajectory is not None
    assert scene.camera_trajectory.pose(1) == (1.0, 0.5, 1.57)
    assert scene.bev_view_params is not None
    assert scene.scene_dir == nr3d_tools_fixture.scene_dir


def test_scene_resolves_raw_rgb_against_scene_root(
    nr3d_tools_fixture: Nr3dToolsFixture,
) -> None:
    scene = Nr3dScene.load(nr3d_tools_fixture.scene_dir)
    chair = scene.proposal_pool.require(3)
    resolved = scene.resolve_raw_rgb(chair.frame_views[0])
    assert resolved.exists()
    assert resolved.name == "000000-rgb.png"
