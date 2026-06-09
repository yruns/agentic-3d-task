"""Shared fixtures: a tiny on-disk NR3D scene + sample, and a fake Codex home."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

# A 1x1 transparent PNG; the loaders only check existence, but a real file keeps
# the fixture honest for any code that opens it.
_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000154a24f1d0000000049454e44ae42"
    "6082"
)

SCENE_ID = "scene0001_00"
SAMPLE_ID = "scannet/scene0001_00::3::100"
PACK_NAME = "pack_nr3d_v9_catalog_first"


@dataclass(frozen=True)
class Nr3dFixture:
    """Paths describing a prepared single-scene NR3D fixture."""

    data_root: Path
    scene_dir: Path
    sample_id: str = SAMPLE_ID
    scene_id: str = SCENE_ID
    target_id: int = 3
    gt_bbox: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)


@pytest.fixture
def nr3d_fixture(tmp_path: Path) -> Nr3dFixture:
    data_root = tmp_path / "data" / "nr3d" / "scannet"
    scene_dir = data_root / SCENE_ID / PACK_NAME
    (scene_dir / "samples").mkdir(parents=True)
    (scene_dir / "bev").mkdir(parents=True)

    proposals = {
        "source": "gt",
        "scene_id": SCENE_ID,
        "axis_align_matrix": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        "proposals": [
            {
                "id": 3,
                "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0],
                "score": 1.0,
                "label": "chair",
                "enriched_category": "office chair",
                "compact_note": "a small red office chair",
            },
            {
                "id": 7,
                "bbox_3d": [2, 0, 0, 1, 1, 1, 0, 0, 0],
                "score": 1.0,
                "label": "table",
                "compact_note": "a wooden table",
            },
        ],
    }
    (scene_dir / "proposals.jsonl").write_text(json.dumps(proposals), encoding="utf-8")
    (scene_dir / "visibility.json").write_text(
        json.dumps({"10": [3, 7], "11": [3]}), encoding="utf-8"
    )
    (scene_dir / "scene_catalog.json").write_text(
        json.dumps(
            {
                "scene_id": SCENE_ID,
                "scene_category": "office",
                "total_frames": 12,
                "frame_id_range": [0, 11],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "bev" / "scene_bev_nr3d.png").write_bytes(_PNG_1X1)

    sample = {
        "sample_id": SAMPLE_ID,
        "scene_id": SCENE_ID,
        "target_id": 3,
        "category": "chair",
        "query": "the red chair",
        "gt_bbox_3d_9dof": [0, 0, 0, 1, 1, 1, 0, 0, 0],
    }
    (scene_dir / "samples" / "scannet__scene0001_00__3__100.json").write_text(
        json.dumps(sample), encoding="utf-8"
    )
    return Nr3dFixture(data_root=data_root, scene_dir=scene_dir)


@dataclass(frozen=True)
class Nr3dToolsFixture:
    """A richer single-scene fixture with per-frame 2D views and BEV assets."""

    data_root: Path
    scene_dir: Path
    scene_id: str = SCENE_ID


def _solid_png(path: Path, size: int) -> None:
    from PIL import Image

    Image.new("RGB", (size, size), color=(120, 120, 120)).save(path, format="PNG")


@pytest.fixture
def nr3d_tools_fixture(tmp_path: Path) -> Nr3dToolsFixture:
    """Build a scene with frame_views, camera trajectory, and BEV view params.

    Layout::

        <root>/<scene>/pack_nr3d_v9_catalog_first/  proposals.jsonl, visibility.json,
                                                     scene_catalog.json, camera_trajectory.json,
                                                     bev/{png, view.json}
        <root>/<scene>/raw/                          000000-rgb.png, 000001-rgb.png
    """
    data_root = tmp_path / "data" / "nr3d" / "scannet"
    scene_dir = data_root / SCENE_ID / PACK_NAME
    (scene_dir / "bev").mkdir(parents=True)
    raw_dir = data_root / SCENE_ID / "raw"
    raw_dir.mkdir(parents=True)
    for frame_id in (0, 1):
        _solid_png(raw_dir / f"{frame_id:06d}-rgb.png", size=128)

    def raw(frame_id: int) -> str:
        return f"data/nr3d/scannet/{SCENE_ID}/raw/{frame_id:06d}-rgb.png"

    proposals = {
        "source": "gt",
        "scene_id": SCENE_ID,
        "proposals": [
            {
                "id": 3,
                "bbox_3d": [0, 0, 0, 1, 1, 1, 0, 0, 0],
                "score": 1.0,
                "label": "chair",
                "enriched_category": "office chair",
                "compact_note": "a small red office chair",
                "enrichment": {
                    "category": "office chair",
                    "color": "red",
                    "nearby_objects": ["table"],
                },
                "frame_views": {
                    "0": {"bbox_2d": [10, 10, 40, 60], "raw_rgb_path": raw(0)},
                    "1": {"bbox_2d": [70, 20, 100, 70], "raw_rgb_path": raw(1)},
                },
            },
            {
                "id": 7,
                "bbox_3d": [2, 0, 0, 1, 1, 1, 0, 0, 0],
                "score": 1.0,
                "label": "table",
                "compact_note": "a wooden table",
                "frame_views": {
                    "0": {"bbox_2d": [60, 10, 95, 60], "raw_rgb_path": raw(0)},
                },
            },
            {
                "id": 9,
                "bbox_3d": [0, 0, 2, 1, 1, 1, 0, 0, 0],
                "score": 1.0,
                "label": "lamp",
                "frame_views": {
                    "1": {"bbox_2d": [10, 10, 30, 30], "raw_rgb_path": raw(1)},
                },
            },
        ],
    }
    (scene_dir / "proposals.jsonl").write_text(json.dumps(proposals), encoding="utf-8")
    (scene_dir / "visibility.json").write_text(
        json.dumps({"0": [3, 7], "1": [3, 9]}), encoding="utf-8"
    )
    (scene_dir / "scene_catalog.json").write_text(
        json.dumps(
            {
                "scene_id": SCENE_ID,
                "scene_category": "office",
                "total_frames": 2,
                "frame_id_range": [0, 1],
                "valid_frame_ids": [0, 1],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "camera_trajectory.json").write_text(
        json.dumps({"0": [0.0, 0.0, 0.0], "1": [1.0, 0.5, 1.57]}),
        encoding="utf-8",
    )
    _solid_png(scene_dir / "bev" / "scene_bev_nr3d.png", size=100)
    (scene_dir / "bev" / "scene_bev_nr3d.view.json").write_text(
        json.dumps(
            {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [0.0, 0.0, 10.0],
                "f": 100.0,
                "c": 50.0,
                "image_size": 100,
                "crop_offset": [0, 0],
            }
        ),
        encoding="utf-8",
    )
    return Nr3dToolsFixture(data_root=data_root, scene_dir=scene_dir)


@pytest.fixture
def fake_codex_home(tmp_path: Path) -> Path:
    """A minimal CODEX_HOME with a config.toml the runtime can copy."""
    home = tmp_path / ".codex-home"
    home.mkdir()
    (home / "config.toml").write_text('model = "test"\n', encoding="utf-8")
    (home / "installation_id").write_text("test-install\n", encoding="utf-8")
    return home
