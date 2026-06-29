"""Tests for SceneFunc3D crop-to-full-frame metadata validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_agent.scenefunc3d.tools.crop_metadata import (
    crop_metadata_path_for_image,
    load_crop_metadata_for_image,
)
from codex_agent.scenefunc3d.tools.models import ToolInputError


def test_load_crop_metadata_rejects_mismatched_crop_image_path(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    source_image_path = tmp_path / "source.png"
    crop_image_path = tmp_path / "000000_crop.jpg"
    other_crop_path = tmp_path / "other_crop.jpg"
    Image.new("RGB", (100, 80)).save(source_image_path)
    Image.new("RGB", (40, 40)).save(crop_image_path)
    Image.new("RGB", (40, 40)).save(other_crop_path)
    crop_metadata_path_for_image(crop_image_path).write_text(
        json.dumps(
            {
                "frame_id": "000000",
                "crop_image_path": str(other_crop_path),
                "source_image_path": str(source_image_path),
                "source_image_width": 100,
                "source_image_height": 80,
                "crop_image_width": 40,
                "crop_image_height": 40,
                "crop_bbox_xyxy": [10, 20, 50, 60],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ToolInputError, match="does not describe image_path"):
        load_crop_metadata_for_image(
            crop_image_path,
            expected_frame_id="000000",
            allowed_image_roots=(tmp_path,),
        )


def test_load_crop_metadata_rejects_source_dimension_mismatch(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    source_image_path = tmp_path / "source.png"
    crop_image_path = tmp_path / "000000_crop.jpg"
    Image.new("RGB", (100, 80)).save(source_image_path)
    Image.new("RGB", (40, 40)).save(crop_image_path)
    crop_metadata_path_for_image(crop_image_path).write_text(
        json.dumps(
            {
                "frame_id": "000000",
                "crop_image_path": str(crop_image_path),
                "source_image_path": str(source_image_path),
                "source_image_width": 99,
                "source_image_height": 80,
                "crop_image_width": 40,
                "crop_image_height": 40,
                "crop_bbox_xyxy": [10, 20, 50, 60],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ToolInputError, match="source image size mismatch"):
        load_crop_metadata_for_image(
            crop_image_path,
            expected_frame_id="000000",
            allowed_image_roots=(tmp_path,),
        )


def test_load_crop_metadata_rejects_crop_dimension_mismatch(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    source_image_path = tmp_path / "source.png"
    crop_image_path = tmp_path / "000000_crop.jpg"
    Image.new("RGB", (100, 80)).save(source_image_path)
    Image.new("RGB", (40, 40)).save(crop_image_path)
    crop_metadata_path_for_image(crop_image_path).write_text(
        json.dumps(
            {
                "frame_id": "000000",
                "crop_image_path": str(crop_image_path),
                "source_image_path": str(source_image_path),
                "source_image_width": 100,
                "source_image_height": 80,
                "crop_image_width": 41,
                "crop_image_height": 40,
                "crop_bbox_xyxy": [10, 20, 51, 60],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ToolInputError, match="crop image size mismatch"):
        load_crop_metadata_for_image(
            crop_image_path,
            expected_frame_id="000000",
            allowed_image_roots=(tmp_path,),
        )


def test_load_crop_metadata_rejects_non_strict_bbox_coordinates(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    source_image_path = tmp_path / "source.png"
    crop_image_path = tmp_path / "000000_crop.jpg"
    Image.new("RGB", (100, 80)).save(source_image_path)
    Image.new("RGB", (40, 40)).save(crop_image_path)
    crop_metadata_path_for_image(crop_image_path).write_text(
        json.dumps(
            {
                "frame_id": "000000",
                "crop_image_path": str(crop_image_path),
                "source_image_path": str(source_image_path),
                "source_image_width": 100,
                "source_image_height": 80,
                "crop_image_width": 40,
                "crop_image_height": 40,
                "crop_bbox_xyxy": ["10", 20, 50, 60],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ToolInputError, match="crop_bbox_xyxy"):
        load_crop_metadata_for_image(
            crop_image_path,
            expected_frame_id="000000",
            allowed_image_roots=(tmp_path,),
        )


def test_load_crop_metadata_rejects_image_path_outside_allowed_roots_before_sidecar(
    tmp_path: Path,
) -> None:
    outside_root = tmp_path.parent / f"{tmp_path.name}_outside"
    outside_root.mkdir()
    outside_crop_path = outside_root / "000000_crop.jpg"

    with pytest.raises(ToolInputError, match="image_path is outside configured roots"):
        load_crop_metadata_for_image(
            outside_crop_path,
            expected_frame_id="000000",
            allowed_image_roots=(tmp_path,),
        )
