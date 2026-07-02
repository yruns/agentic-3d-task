"""Tests for shared SceneFunc3D camera/depth I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from codex_agent.scenefunc3d.backends.camera_io import (
    load_camera_geometry,
    read_camera_matrix,
    read_depth_meters,
)
from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry
from codex_agent.scenefunc3d.tools.models import ToolInputError


def _write_depth_png(path: Path, depth_mm: np.ndarray) -> None:
    Image.fromarray(depth_mm.astype(np.uint16)).save(path)


def test_read_depth_meters_scales_millimetres(tmp_path: Path) -> None:
    depth_png = tmp_path / "000000-depth.png"
    _write_depth_png(depth_png, np.array([[1000, 2000], [3000, 0]], dtype=np.uint16))
    depth = read_depth_meters(depth_png)
    assert depth.shape == (2, 2)
    assert depth[0, 0] == pytest.approx(1.0)
    assert depth[1, 0] == pytest.approx(3.0)


def test_read_camera_matrix_reshapes(tmp_path: Path) -> None:
    intrinsic_txt = tmp_path / "000000-intrinsic.txt"
    intrinsic_txt.write_text("100 0 50\n0 100 60\n0 0 1\n", encoding="utf-8")
    matrix = read_camera_matrix(
        intrinsic_txt, expected_shape=(3, 3), field_name="intrinsics"
    )
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == pytest.approx(100.0)


def test_read_camera_matrix_rejects_wrong_size(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("1 2 3\n", encoding="utf-8")
    with pytest.raises(ToolInputError):
        read_camera_matrix(bad, expected_shape=(3, 3), field_name="intrinsics")


def test_load_camera_geometry_builds_validated_geometry(tmp_path: Path) -> None:
    (tmp_path / "000000-intrinsic.txt").write_text(
        "100 0 50\n0 100 60\n0 0 1\n", encoding="utf-8"
    )
    (tmp_path / "000000.txt").write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8"
    )
    geometry = load_camera_geometry(
        intrinsics_path=tmp_path / "000000-intrinsic.txt",
        pose_path=tmp_path / "000000.txt",
    )
    assert isinstance(geometry, CameraGeometry)
    assert geometry.intrinsics[1, 1] == pytest.approx(100.0)
