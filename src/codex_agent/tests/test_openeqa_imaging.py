"""Unit tests for the OpenEQA imaging helpers (crop + contact sheet)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from codex_agent.errors import OpenEqaDataError  # noqa: E402
from codex_agent.openeqa.tools.imaging import (  # noqa: E402
    box_looks_normalized,
    clamp_box,
    compose_contact_sheet,
    crop_rgb_for_view,
    normalized_box_to_pixels,
)


def _solid(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> Path:
    Image.new("RGB", size, color=color).save(path, format="PNG")
    return path


# ----- clamp_box --------------------------------------------------------------


def test_clamp_box_orders_and_clamps() -> None:
    # corners given out of order and out of bounds -> sorted + clamped inside image
    box = clamp_box((90.0, 80.0, 10.0, 5.0), width=100, height=100)
    left, top, right, bottom = box
    assert 0 <= left < right <= 100
    assert 0 <= top < bottom <= 100


def test_clamp_box_grows_tiny_box_to_minimum() -> None:
    box = clamp_box((50.0, 50.0, 51.0, 51.0), width=200, height=200)
    left, top, right, bottom = box
    assert right - left >= 32
    assert bottom - top >= 32


def test_clamp_box_rejects_empty_image() -> None:
    with pytest.raises(OpenEqaDataError):
        clamp_box((0.0, 0.0, 1.0, 1.0), width=0, height=10)


# ----- normalized helpers -----------------------------------------------------


def test_box_looks_normalized() -> None:
    assert box_looks_normalized((0.1, 0.2, 0.9, 1.0))
    assert not box_looks_normalized((0.1, 0.2, 0.9, 40.0))


def test_normalized_box_to_pixels() -> None:
    assert normalized_box_to_pixels((0.0, 0.5, 1.0, 1.0), 200, 100) == (
        0.0,
        50.0,
        200.0,
        100.0,
    )


# ----- crop_rgb_for_view ------------------------------------------------------


def test_crop_keeps_region_at_higher_detail(tmp_path: Path) -> None:
    src = _solid(tmp_path / "src.png", (1000, 800), (10, 20, 30))
    dst = tmp_path / "crop.jpg"
    saved, applied = crop_rgb_for_view(src, dst, box=(100, 100, 400, 500), max_size=768)
    assert saved.exists()
    left, top, right, bottom = applied
    assert (left, top, right, bottom) == (100, 100, 400, 500)
    with Image.open(saved) as out:
        # 300x400 crop is under the 768 cap, so it is kept at native crop size
        assert out.size == (300, 400)


def test_crop_downsizes_when_larger_than_max(tmp_path: Path) -> None:
    src = _solid(tmp_path / "src.png", (2000, 2000), (0, 0, 0))
    dst = tmp_path / "crop.jpg"
    saved, _ = crop_rgb_for_view(src, dst, box=(0, 0, 2000, 2000), max_size=256)
    with Image.open(saved) as out:
        assert max(out.size) == 256


# ----- compose_contact_sheet --------------------------------------------------


def test_contact_sheet_tiles_all_inputs(tmp_path: Path) -> None:
    paths = [
        _solid(tmp_path / f"f{i}.png", (120, 90), (i * 40, 0, 0)) for i in range(3)
    ]
    sheet = compose_contact_sheet(
        [(f"frame {i}", p) for i, p in enumerate(paths)],
        tmp_path / "sheet.jpg",
        tile_px=100,
    )
    assert sheet.exists()
    with Image.open(sheet) as out:
        # 3 tiles -> 2 columns x 2 rows of (100 x 100+label) cells
        assert out.width >= 200 and out.height >= 200


def test_contact_sheet_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(OpenEqaDataError):
        compose_contact_sheet([], tmp_path / "sheet.jpg")
