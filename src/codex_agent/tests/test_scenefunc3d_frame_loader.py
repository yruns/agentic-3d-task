"""Tests for the SceneFunc3D per-frame camera/depth loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.backends.frame_loader import (
    frame_rgb_path,
    iter_frame_geometry,
    load_frame_geometry,
    read_frame_depth,
)
from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry
from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene
from codex_agent.tests.scenefunc3d_synthetic_scene import (
    SyntheticFrame,
    write_synthetic_scene,
)

_INTRINSICS = np.array([[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]])


def _scene(tmp_path: Path) -> SceneFunc3dToolScene:
    frames = (
        SyntheticFrame("000000", np.eye(4), depth_value_m=2.0),
        SyntheticFrame("000001", np.eye(4), depth_value_m=3.0),
    )
    return write_synthetic_scene(
        tmp_path / "scene",
        frames=frames,
        vertices_world=np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
        intrinsics=_INTRINSICS,
    )


def test_iter_frame_geometry_yields_all_frames(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    items = list(iter_frame_geometry(scene))
    assert [frame_id for frame_id, _geometry in items] == ["000000", "000001"]
    assert all(isinstance(g, CameraGeometry) for _f, g in items)


def test_load_frame_geometry_reads_one_frame(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    geometry = load_frame_geometry(scene, "000001")
    assert geometry.intrinsics[0, 0] == pytest.approx(20.0)


def test_read_frame_depth_scales_metres(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    depth = read_frame_depth(scene, "000000")
    assert depth.shape == (40, 40)
    assert float(depth[0, 0]) == pytest.approx(2.0)


def test_frame_rgb_path_exists(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    rgb = frame_rgb_path(scene, "000000")
    assert rgb.is_file()
    assert rgb.name == "000000-rgb.jpg"


def test_iter_frame_geometry_skips_incomplete_frames(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    (scene.raw_dir / "000001.txt").unlink()
    items = list(iter_frame_geometry(scene))
    assert [frame_id for frame_id, _geometry in items] == ["000000"]


def test_iter_frame_geometry_fails_closed_when_no_geometry(tmp_path: Path) -> None:
    scene_root = tmp_path / "empty"
    (scene_root / "raw").mkdir(parents=True)
    (scene_root / "raw" / "000000-rgb.jpg").write_bytes(b"")
    (scene_root / "raw" / "source_frames.json").write_text(
        '[{"frame_id": "000000", "rgb": "000000-rgb.jpg"}]', encoding="utf-8"
    )
    scene = SceneFunc3dToolScene.load(scene_root)
    with pytest.raises(SceneFunc3dDataError):
        list(iter_frame_geometry(scene))
