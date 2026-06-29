"""Tests for deterministic SceneFunc3D mask lifting."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from codex_agent.scenefunc3d.backends.frame_assets import (
    FrameGeometryAssets,
    resolve_frame_geometry_assets,
)
from codex_agent.scenefunc3d.backends.lift_3d import (
    CameraGeometry,
    assign_nearest_scene_point_indices,
    backproject_mask_to_world,
    load_mask_npz,
    load_scene_mesh_vertices,
    write_lift_npz,
    write_lift_ply,
)
from codex_agent.scenefunc3d.tools.models import ToolInputError
from codex_agent.scenefunc3d.tools.scene_context import (
    SceneFunc3dToolScene,
    SourceFrameIndex,
)


def test_backproject_mask_to_world_with_identity_pose() -> None:
    np = pytest.importorskip("numpy")
    intrinsics = np.array(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    camera_to_world = np.eye(4, dtype=np.float64)
    geometry = CameraGeometry(
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
    )
    mask = np.array([[False, True], [False, False]], dtype=np.bool_)
    depth_meters = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=np.float64)

    points_world = backproject_mask_to_world(mask, depth_meters, geometry)

    np.testing.assert_allclose(points_world, np.array([[1.0, 0.0, 2.0]]))


def test_backproject_mask_to_world_rejects_sparse_mask() -> None:
    np = pytest.importorskip("numpy")
    geometry = CameraGeometry(
        intrinsics=np.eye(3, dtype=np.float64),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    mask = np.array([[False, True], [False, False]], dtype=np.bool_)
    depth_meters = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(ToolInputError, match="lift_too_sparse"):
        backproject_mask_to_world(mask, depth_meters, geometry)


def test_backproject_mask_to_world_filters_mixed_invalid_depth_under_mask() -> None:
    np = pytest.importorskip("numpy")
    geometry = CameraGeometry(
        intrinsics=np.eye(3, dtype=np.float64),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    mask = np.array([[True, True]], dtype=np.bool_)
    depth_meters = np.array([[1.0, np.nan]], dtype=np.float64)

    points_world = backproject_mask_to_world(mask, depth_meters, geometry)

    np.testing.assert_allclose(points_world, np.array([[0.0, 0.0, 1.0]]))


def test_load_mask_npz_round_trip_returns_bool_2d(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    mask_path = tmp_path / "mask.npz"
    np.savez(mask_path, mask=np.array([[0, 1], [1, 0]], dtype=np.uint8))

    mask = load_mask_npz(mask_path)

    assert mask.dtype == np.bool_
    assert mask.shape == (2, 2)
    assert mask.tolist() == [[False, True], [True, False]]


@pytest.mark.parametrize(
    "mask_values",
    [
        [[0, 2]],
        [[0.0, float("nan")]],
        [[1.0 + 2.0j, 0.0 + 0.0j]],
        [["", "mask"]],
    ],
)
def test_load_mask_npz_rejects_non_binary_masks(
    tmp_path: Path, mask_values: list[list[object]]
) -> None:
    np = pytest.importorskip("numpy")
    mask_path = tmp_path / "mask.npz"
    np.savez(mask_path, mask=np.array(mask_values))

    with pytest.raises(ToolInputError, match="binary"):
        load_mask_npz(mask_path)


def test_backproject_mask_to_world_rejects_non_binary_mask_array() -> None:
    np = pytest.importorskip("numpy")
    geometry = CameraGeometry(
        intrinsics=np.eye(3, dtype=np.float64),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    mask = np.array([[0, 2]], dtype=np.uint8)
    depth_meters = np.array([[1.0, 1.0]], dtype=np.float64)

    with pytest.raises(ToolInputError, match="binary"):
        backproject_mask_to_world(mask, depth_meters, geometry)


def test_backproject_mask_to_world_rejects_complex_mask_array() -> None:
    np = pytest.importorskip("numpy")
    geometry = CameraGeometry(
        intrinsics=np.eye(3, dtype=np.float64),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    mask = np.array([[1.0 + 2.0j, 0.0 + 0.0j]], dtype=np.complex128)
    depth_meters = np.array([[1.0, 1.0]], dtype=np.float64)

    with pytest.raises(ToolInputError, match="binary"):
        backproject_mask_to_world(mask, depth_meters, geometry)


def test_camera_geometry_rejects_negative_focal_length() -> None:
    np = pytest.importorskip("numpy")
    intrinsics = np.eye(3, dtype=np.float64)
    intrinsics[0, 0] = -1.0

    with pytest.raises(ToolInputError, match="positive"):
        CameraGeometry(intrinsics=intrinsics, camera_to_world=np.eye(4))


def test_camera_geometry_rejects_non_affine_pose() -> None:
    np = pytest.importorskip("numpy")
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[3, 0] = 1.0

    with pytest.raises(ToolInputError, match="last row"):
        CameraGeometry(intrinsics=np.eye(3), camera_to_world=camera_to_world)


def test_camera_geometry_rejects_degenerate_rotation_block() -> None:
    np = pytest.importorskip("numpy")
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[2, 2] = 0.0

    with pytest.raises(ToolInputError, match="non-degenerate"):
        CameraGeometry(intrinsics=np.eye(3), camera_to_world=camera_to_world)


def test_write_lift_npz_and_ascii_ply_artifacts(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    points_world = np.array([[1.0, 0.0, 2.0], [3.5, 4.0, 5.0]], dtype=np.float64)
    point_indices = np.array([10, 12], dtype=np.int64)
    npz_path = tmp_path / "lifted_points.npz"
    ply_path = tmp_path / "lifted_points.ply"

    written_npz_path = write_lift_npz(
        npz_path, points_world, point_indices=point_indices
    )
    written_ply_path = write_lift_ply(ply_path, points_world)

    assert written_npz_path == npz_path
    assert written_ply_path == ply_path
    with np.load(npz_path) as archive:
        np.testing.assert_allclose(archive["points_world"], points_world)
        np.testing.assert_array_equal(archive["point_indices"], point_indices)
    ply_text = ply_path.read_text(encoding="ascii")
    assert ply_text.startswith("ply\nformat ascii 1.0\n")
    assert "element vertex 2\n" in ply_text
    assert "1 0 2\n" in ply_text
    assert "3.5 4 5\n" in ply_text


def test_write_lift_npz_rejects_point_index_count_mismatch(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    points_world = np.array([[1.0, 0.0, 2.0], [3.5, 4.0, 5.0]], dtype=np.float64)
    point_indices = np.array([10], dtype=np.int64)

    with pytest.raises(ToolInputError, match="point_indices"):
        write_lift_npz(
            tmp_path / "lifted_points.npz",
            points_world,
            point_indices=point_indices,
        )


def test_load_scene_mesh_vertices_reads_binary_little_endian_ply(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    mesh_path = _write_binary_scene_mesh(
        tmp_path / "mesh.ply",
        points=((1.0, 0.0, 2.0), (3.5, 4.0, 5.0)),
    )

    vertices = load_scene_mesh_vertices(mesh_path)

    np.testing.assert_allclose(
        vertices,
        np.array([[1.0, 0.0, 2.0], [3.5, 4.0, 5.0]], dtype=np.float64),
    )


def test_assign_nearest_scene_point_indices_returns_vertex_indices() -> None:
    np = pytest.importorskip("numpy")
    points_world = np.array([[1.02, 0.0, 2.0], [3.45, 4.0, 5.0]], dtype=np.float64)
    scene_points_world = np.array(
        [[1.0, 0.0, 2.0], [3.5, 4.0, 5.0], [10.0, 0.0, 0.0]], dtype=np.float64
    )

    point_indices = assign_nearest_scene_point_indices(
        points_world,
        scene_points_world,
        max_distance_meters=0.1,
    )

    np.testing.assert_array_equal(point_indices, np.array([0, 1], dtype=np.int64))


def test_assign_nearest_scene_point_indices_rejects_far_points() -> None:
    np = pytest.importorskip("numpy")
    points_world = np.array([[1.2, 0.0, 2.0]], dtype=np.float64)
    scene_points_world = np.array([[1.0, 0.0, 2.0]], dtype=np.float64)

    with pytest.raises(ToolInputError, match="nearest_scene_point_too_far"):
        assign_nearest_scene_point_indices(
            points_world,
            scene_points_world,
            max_distance_meters=0.05,
        )


def test_write_lift_ply_rejects_nonfinite_points(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    points_world = np.array([[1.0, np.nan, 2.0]], dtype=np.float64)

    with pytest.raises(ToolInputError, match="finite"):
        write_lift_ply(tmp_path / "bad.ply", points_world)


def test_resolve_frame_geometry_assets_from_raw_scene(tmp_path: Path) -> None:
    scene = _make_tool_scene(tmp_path)
    depth_path = scene.raw_dir / "000050-depth.png"
    intrinsics_path = scene.raw_dir / "intrinsics.txt"
    pose_path = scene.raw_dir / "pose" / "000050.txt"
    depth_path.write_bytes(b"depth")
    intrinsics_path.write_text("1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
    pose_path.parent.mkdir()
    pose_path.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")

    assets = resolve_frame_geometry_assets(scene, "000050")

    assert assets == FrameGeometryAssets(
        frame_id="000050",
        depth_path=depth_path,
        intrinsics_path=intrinsics_path,
        pose_path=pose_path,
    )


def test_resolve_frame_geometry_assets_reports_missing_asset(tmp_path: Path) -> None:
    scene = _make_tool_scene(tmp_path)
    (scene.raw_dir / "000050-depth.png").write_bytes(b"depth")

    with pytest.raises(ToolInputError, match="frame_geometry_asset_missing"):
        resolve_frame_geometry_assets(scene, "000050")


def _make_tool_scene(tmp_path: Path) -> SceneFunc3dToolScene:
    raw_dir = tmp_path / "scene" / "raw"
    raw_dir.mkdir(parents=True)
    return SceneFunc3dToolScene(
        visit_id="scene",
        scene_root=tmp_path / "scene",
        rgb_frame_ids=("000050",),
        source_frame_index=SourceFrameIndex(records=()),
    )


def _write_binary_scene_mesh(
    path: Path, *, points: tuple[tuple[float, float, float], ...]
) -> Path:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for x_value, y_value, z_value in points:
            handle.write(
                struct.pack(
                    "<fffBBB",
                    x_value,
                    y_value,
                    z_value,
                    0,
                    0,
                    0,
                )
            )
    return path
