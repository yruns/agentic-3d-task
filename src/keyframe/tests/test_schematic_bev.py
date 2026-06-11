"""Unit tests for the mesh-free schematic BEV (renderer, builder, selector branch).

These exercise the OpenEQA top-down floor-plan path with tiny synthetic data (no
real ScanNet). They require the ``vision`` extra (OpenCV) and are skipped
otherwise.
"""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from keyframe import KeyframeSelector  # noqa: E402
from keyframe.bev import (  # noqa: E402
    DEFAULT_SCHEMATIC_BEV_CONFIG,
    BEVMarker,
    OpenEqaSceneBEVBuilder,
    OrthoBevView,
    SchematicBevConfig,
    render_schematic_bev,
)


def _markers() -> list[BEVMarker]:
    return [
        BEVMarker(
            obj_id=0, category="desk", position=(0.0, 0.0, 0.4), extent=(1.2, 0.6, 0.7)
        ),
        BEVMarker(
            obj_id=1, category="chair", position=(1.5, 0.5, 0.3), extent=(0.5, 0.5, 0.9)
        ),
        BEVMarker(obj_id=2, category="door", position=(3.0, 2.0, 1.0)),  # no extent
    ]


def _camera_xy(n: int = 4) -> np.ndarray:
    return np.array([[0.3 * i, -1.0] for i in range(n)], dtype=np.float64)


# ----- OrthoBevView -----------------------------------------------------------


def test_ortho_view_project_and_roundtrip() -> None:
    view = OrthoBevView(min_x=-1.0, max_y=3.0, scale=100.0, width=400, height=400)
    # top-left of world maps near image origin; +y is up so larger y -> smaller py.
    assert view.project(-1.0, 3.0) == (0, 0)
    px, py = view.project(0.0, 2.0)
    assert px == 100 and py == 100
    payload = view.to_payload()
    assert payload == {
        "min_x": -1.0,
        "max_y": 3.0,
        "scale": 100.0,
        "width": 400,
        "height": 400,
    }


# ----- render_schematic_bev ---------------------------------------------------


def test_render_schematic_bev_shape_and_view() -> None:
    image, view = render_schematic_bev(_markers(), _camera_xy())
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8
    assert image.shape[0] == view.height and image.shape[1] == view.width
    # longest side bounded by the configured budget.
    assert max(image.shape[:2]) <= DEFAULT_SCHEMATIC_BEV_CONFIG.image_size


def test_render_schematic_bev_highlight_changes_pixels() -> None:
    plain, _ = render_schematic_bev(_markers(), _camera_xy())
    highlighted, _ = render_schematic_bev(
        _markers(), _camera_xy(), highlight_ids=frozenset({1})
    )
    assert plain.shape == highlighted.shape
    assert not np.array_equal(plain, highlighted)


def test_render_schematic_bev_camera_only() -> None:
    image, _ = render_schematic_bev([], _camera_xy(5))
    assert image.size > 0


def test_render_schematic_bev_requires_geometry() -> None:
    with pytest.raises(ValueError, match="at least one object or camera pose"):
        render_schematic_bev([], np.empty((0, 2), dtype=np.float64))


def test_render_schematic_bev_respects_small_image_size() -> None:
    cfg = SchematicBevConfig(image_size=200, max_pixels_per_meter=50.0)
    image, _ = render_schematic_bev(_markers(), _camera_xy(), config=cfg)
    assert max(image.shape[:2]) <= 200 + 2 * cfg.margin_px


# ----- OpenEqaSceneBEVBuilder -------------------------------------------------


def _write_traj(conceptgraph_dir: Path, n: int = 5) -> None:
    conceptgraph_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        pose = np.eye(4)
        pose[:3, 3] = [0.3 * i, 0.0, 1.0]
        lines.append(" ".join(str(v) for v in pose.flatten()))
    (conceptgraph_dir / "traj.txt").write_text("\n".join(lines), encoding="utf-8")


def test_openeqa_builder_writes_png_view_json_and_cache(tmp_path: Path) -> None:
    scene_id = "010-scannet-scene0011_00"
    _write_traj(tmp_path / scene_id / "conceptgraph")
    output = tmp_path / "out" / "bev.png"
    builder = OpenEqaSceneBEVBuilder()

    result = builder.build(
        scene_id=scene_id,
        data_root=tmp_path,
        markers=_markers(),
        output_path=output,
    )
    assert result == output
    assert output.exists() and output.stat().st_size > 0
    assert output.with_suffix(".view.json").exists()
    cache_dir = tmp_path / scene_id / "bev_cache"
    assert len(list(cache_dir.glob("schematic_bev_*.png"))) == 1


def test_openeqa_builder_cache_hit(tmp_path: Path) -> None:
    scene_id = "010-scannet-scene0011_00"
    _write_traj(tmp_path / scene_id / "conceptgraph")
    builder = OpenEqaSceneBEVBuilder()
    output = tmp_path / "out" / "bev.png"
    builder.build(
        scene_id=scene_id, data_root=tmp_path, markers=_markers(), output_path=output
    )
    output.unlink()
    builder.build(
        scene_id=scene_id, data_root=tmp_path, markers=_markers(), output_path=output
    )
    assert output.exists()


def test_openeqa_builder_no_cache_does_not_write_cache(tmp_path: Path) -> None:
    scene_id = "010-scannet-scene0011_00"
    _write_traj(tmp_path / scene_id / "conceptgraph")
    output = tmp_path / "out" / "bev.png"
    OpenEqaSceneBEVBuilder().build(
        scene_id=scene_id,
        data_root=tmp_path,
        markers=_markers(),
        output_path=output,
        use_cache=False,
    )
    assert output.exists()
    assert not (tmp_path / scene_id / "bev_cache").exists()


# ----- KeyframeSelector OpenEQA branch ----------------------------------------


def _build_openeqa_clip(tmp_path: Path) -> Path:
    """Build a tiny OpenEQA clip and return its conceptgraph dir."""
    clip = tmp_path / "010-scannet-scene0011_00"
    raw = clip / "raw"
    raw.mkdir(parents=True)
    for fid in range(4):
        raw.joinpath(f"{fid:06d}-rgb.png").write_bytes(b"\xff\xd8\xff")
    conceptgraph = clip / "conceptgraph"
    (conceptgraph / "pcd_saves").mkdir(parents=True)

    def _cube(center: tuple[float, float, float], half: float) -> np.ndarray:
        cx, cy, cz = center
        return np.array(
            [
                [cx + dx, cy + dy, cz + dz]
                for dx in (-half, half)
                for dy in (-half, half)
                for dz in (-half, half)
            ],
            dtype=np.float64,
        )

    objects = [
        {
            "class_name": ["desk"] * 3,
            "pcd_np": _cube((0.0, 0.0, 0.4), 0.6),
            "bbox_np": _cube((0.0, 0.0, 0.4), 0.6),
            "image_idx": [0, 1],
            "xyxy": np.array([[10, 10, 40, 40], [10, 10, 40, 40]], dtype=float),
            "num_detections": 2,
        },
        {
            "class_name": ["chair"] * 3,
            "pcd_np": _cube((1.5, 0.5, 0.3), 0.25),
            "bbox_np": _cube((1.5, 0.5, 0.3), 0.25),
            "image_idx": [1],
            "xyxy": np.array([[20, 20, 35, 45]], dtype=float),
            "num_detections": 1,
        },
    ]
    with gzip.open(conceptgraph / "pcd_saves" / "full_scene_post.pkl.gz", "wb") as fh:
        pickle.dump({"objects": objects}, fh)
    (conceptgraph / "enriched_objects.json").write_text(
        json.dumps({"objects": []}), encoding="utf-8"
    )
    _write_traj(conceptgraph, n=4)
    return conceptgraph


def test_generate_scene_bev_openeqa(tmp_path: Path) -> None:
    conceptgraph = _build_openeqa_clip(tmp_path)
    selector = KeyframeSelector.from_scene_path(conceptgraph, dataset="openeqa")
    output = tmp_path / "scratch" / "bev.png"
    result = selector.generate_scene_bev(output_path=output, use_cache=False)
    assert result == output
    assert output.exists() and output.stat().st_size > 0


def test_generate_scene_bev_highlight_openeqa(tmp_path: Path) -> None:
    conceptgraph = _build_openeqa_clip(tmp_path)
    selector = KeyframeSelector.from_scene_path(conceptgraph, dataset="openeqa")
    output = tmp_path / "scratch" / "bev_hl.png"
    selector.generate_scene_bev(
        output_path=output, use_cache=False, highlight_ids=frozenset({0})
    )
    assert output.exists()


def test_generate_scene_bev_unknown_dataset_raises(tmp_path: Path) -> None:
    conceptgraph = _build_openeqa_clip(tmp_path)
    selector = KeyframeSelector.from_scene_path(conceptgraph, dataset="mystery")
    with pytest.raises(ValueError, match="No BEV builder"):
        selector.generate_scene_bev(use_cache=False)
