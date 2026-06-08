"""Unit tests for the strongly-typed BEV rendering module.

These exercise mesh I/O, the look-down renderer and the NR3D asset resolver
using tiny synthetic meshes (no real ScanNet data). They require the ``vision``
extra (opencv + plyfile) and are skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("plyfile")

from keyframe.bev import (  # noqa: E402
    BEVMarker,
    Nr3dSceneBEVBuilder,
    SceneBEVConfig,
    TriangleMesh,
    load_ply_mesh,
    render_scene_bev,
)
from keyframe.bev.builder import (  # noqa: E402
    _find_scannet_mesh,
    _load_intrinsic,
    _load_poses,
)


def _floor_mesh() -> TriangleMesh:
    verts = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], dtype=np.float64)
    colors = np.full((4, 3), 0.5, dtype=np.float32)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return TriangleMesh(vertices=verts, colors=colors, triangles=tris)


def _write_floor_ply(path: Path) -> None:
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "element face 2\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
        "-2 -2 0 128 128 128\n"
        "2 -2 0 128 128 128\n"
        "2 2 0 128 128 128\n"
        "-2 2 0 128 128 128\n"
        "3 0 1 2\n"
        "3 0 2 3\n"
    )


def _look_down_poses(height: float = 1.5, count: int = 3) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [0.0, 0.0, height]
    return np.repeat(pose[None], count, axis=0)


def _intrinsic() -> np.ndarray:
    return np.array(
        [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


# ----- mesh -----------------------------------------------------------------


def test_triangle_normals_point_up_for_floor() -> None:
    normals = _floor_mesh().triangle_normals()
    assert np.allclose(normals[:, 2], 1.0)


def test_transformed_translates_vertices() -> None:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    moved = _floor_mesh().transformed(matrix)
    assert np.allclose(moved.vertices[0], [-1.0, 0.0, 3.0])
    assert np.array_equal(moved.triangles, _floor_mesh().triangles)


def test_transformed_rejects_non_4x4() -> None:
    with pytest.raises(ValueError, match="4x4"):
        _floor_mesh().transformed(np.eye(3, dtype=np.float64))


def test_load_ply_mesh_roundtrip(tmp_path: Path) -> None:
    ply = tmp_path / "floor.ply"
    _write_floor_ply(ply)
    mesh = load_ply_mesh(ply)
    assert mesh.vertices.shape == (4, 3)
    assert mesh.colors.shape == (4, 3)
    assert mesh.triangles.shape == (2, 3)
    assert mesh.colors.max() <= 1.0


def test_load_ply_mesh_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ply_mesh(tmp_path / "nope.ply")


# ----- renderer -------------------------------------------------------------


def test_render_scene_bev_produces_cropped_rgb() -> None:
    cfg = SceneBEVConfig(image_size=400)
    out = render_scene_bev(_floor_mesh(), _look_down_poses(), _intrinsic(), config=cfg)
    assert out.image.ndim == 3 and out.image.shape[2] == 3
    assert out.image.dtype == np.uint8
    # Cropped tightly: smaller than the full canvas but non-empty.
    assert 0 < out.image.shape[0] <= 400
    assert int((out.image < 245).any(axis=2).sum()) > 0


def test_render_scene_bev_marker_and_highlight_do_not_crash() -> None:
    cfg = SceneBEVConfig(image_size=300)
    markers = [
        BEVMarker(obj_id=1, category="sofa", position=(0.0, 0.0, 0.1)),
        BEVMarker(obj_id=2, category="lamp", position=(1.0, 1.0, 0.1)),
    ]
    out = render_scene_bev(
        _floor_mesh(),
        _look_down_poses(),
        _intrinsic(),
        markers,
        highlight_ids=frozenset({1}),
        config=cfg,
    )
    assert out.view.image_size == 300
    assert out.view.crop_offset[0] >= 0


def test_render_scene_bev_rejects_bad_pose_shape() -> None:
    with pytest.raises(ValueError, match="camera_poses"):
        render_scene_bev(
            _floor_mesh(), np.zeros((4, 4), dtype=np.float64), _intrinsic()
        )


def test_render_scene_bev_rejects_bad_intrinsic() -> None:
    with pytest.raises(ValueError, match="intrinsic"):
        render_scene_bev(_floor_mesh(), _look_down_poses(), np.eye(4, dtype=np.float64))


# ----- NR3D builder asset resolution ----------------------------------------


def _make_nr3d_layout(tmp_path: Path, scene_id: str) -> tuple[Path, Path]:
    """Create data_root/<scene>/raw/{traj,intrinsic} + aux mesh; return roots."""
    data_root = tmp_path / "nr3d" / "scannet"
    scene_dir = data_root / scene_id
    (scene_dir / "raw").mkdir(parents=True)
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 1.5]
    (scene_dir / "raw" / "traj.txt").write_text(
        " ".join(str(v) for v in pose.flatten())
    )
    (scene_dir / "raw" / "intrinsic_color.txt").write_text(
        "577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n"
    )
    mesh_dir = tmp_path / "nr3d" / "scannet_aux_meshes" / scene_id
    mesh_dir.mkdir(parents=True)
    _write_floor_ply(mesh_dir / f"{scene_id}_vh_clean_2.ply")
    return data_root, mesh_dir


def test_nr3d_resolve_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCANNET_DATA_ROOT", raising=False)
    data_root, mesh_dir = _make_nr3d_layout(tmp_path, "scene0000_00")
    paths = Nr3dSceneBEVBuilder().resolve_paths("scene0000_00", data_root)
    assert paths.mesh == mesh_dir / "scene0000_00_vh_clean_2.ply"
    assert paths.trajectory == data_root / "scene0000_00" / "raw" / "traj.txt"
    assert paths.intrinsic == data_root / "scene0000_00" / "raw" / "intrinsic_color.txt"


def test_nr3d_resolve_axis_align(tmp_path: Path) -> None:
    data_root, _ = _make_nr3d_layout(tmp_path, "scene0000_00")
    aux = tmp_path / "nr3d" / "scannet_aux" / "scene0000_00"
    aux.mkdir(parents=True)
    (aux / "scene0000_00.txt").write_text(
        "axisAlignment = 1 0 0 1 0 1 0 2 0 0 1 3 0 0 0 1\n"
    )
    matrix = Nr3dSceneBEVBuilder().resolve_axis_align("scene0000_00", data_root)
    assert matrix is not None
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix[:3, 3], [1.0, 2.0, 3.0])


def test_nr3d_resolve_axis_align_absent_returns_none(tmp_path: Path) -> None:
    data_root, _ = _make_nr3d_layout(tmp_path, "scene0000_00")
    assert Nr3dSceneBEVBuilder().resolve_axis_align("scene0000_00", data_root) is None


def test_find_scannet_mesh_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SCANNET_DATA_ROOT", raising=False)
    data_root = tmp_path / "nr3d" / "scannet"
    data_root.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="ScanNet mesh"):
        _find_scannet_mesh("scene9999_99", data_root)


def test_load_poses_single_line(tmp_path: Path) -> None:
    path = tmp_path / "traj.txt"
    eye = " ".join(str(v) for v in np.eye(4).flatten())
    path.write_text(f"{eye}\n{eye}\n")
    poses = _load_poses(path)
    assert poses.shape == (2, 4, 4)


def test_load_intrinsic_from_4x4(tmp_path: Path) -> None:
    path = tmp_path / "intr.txt"
    path.write_text("577 0 320 0\n0 577 240 0\n0 0 1 0\n0 0 0 1\n")
    k = _load_intrinsic(path)
    assert k.shape == (3, 3)
    assert k[0, 0] == pytest.approx(577.0)


# ----- end-to-end builder ---------------------------------------------------


def test_builder_build_renders_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SCANNET_DATA_ROOT", raising=False)
    data_root, _ = _make_nr3d_layout(tmp_path, "scene0000_00")
    builder = Nr3dSceneBEVBuilder(SceneBEVConfig(image_size=300))
    markers = [BEVMarker(obj_id=0, category="floor", position=(0.0, 0.0, 0.0))]
    output = tmp_path / "out" / "bev.png"

    result = builder.build(
        scene_id="scene0000_00",
        data_root=data_root,
        markers=markers,
        output_path=output,
    )
    assert result == output
    assert output.exists() and output.stat().st_size > 0
    cache_dir = data_root / "scene0000_00" / "bev_cache"
    cached = list(cache_dir.glob("keyframe_bev_*.png"))
    assert len(cached) == 1

    # Second call is a cache hit (same bytes, no re-render needed).
    output.unlink()
    builder.build(
        scene_id="scene0000_00",
        data_root=data_root,
        markers=markers,
        output_path=output,
    )
    assert output.exists()
