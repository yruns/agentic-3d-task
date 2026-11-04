"""Unit tests for the OpenEQA ``view_crop`` zoom tool."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from codex_agent.openeqa.tools.crop_tools import (  # noqa: E402
    ViewCropArgs,
    _best_view_and_box,
    _scale_detection_box,
    view_crop,
)
from codex_agent.openeqa.tools.dispatch import run_tool  # noqa: E402
from codex_agent.openeqa.tools.models import ToolInputError  # noqa: E402
from codex_agent.openeqa.tools.scene_context import OpenEqaToolScene  # noqa: E402
from codex_agent.tests.conftest import OpenEqaToolsFixture  # noqa: E402


@pytest.fixture
def tool_scene(openeqa_tools_fixture: OpenEqaToolsFixture) -> OpenEqaToolScene:
    return OpenEqaToolScene.load(openeqa_tools_fixture.scene_dir)


# ----- bbox mode (no selector needed) -----------------------------------------


def test_crop_by_normalized_bbox(tool_scene: OpenEqaToolScene, tmp_path: Path) -> None:
    result = view_crop(
        tool_scene,
        ViewCropArgs(frame_id=3, bbox=[0.25, 0.25, 0.75, 0.75]),
        out_dir=tmp_path,
    )
    payload = result.to_payload()
    assert payload["source"] == "bbox"
    assert payload["frame_id"] == 3
    assert Path(payload["image_path"]).exists()
    left, top, right, bottom = payload["crop_box_px"]
    assert 0 <= left < right <= 48
    assert 0 <= top < bottom <= 48
    assert "crop_box_norm" in payload


def test_crop_by_pixel_bbox(tool_scene: OpenEqaToolScene, tmp_path: Path) -> None:
    result = view_crop(
        tool_scene, ViewCropArgs(frame_id=2, bbox=[4, 4, 40, 40]), out_dir=tmp_path
    )
    assert result.source == "bbox"
    with Image.open(result.image_path) as out:
        assert out.size[0] > 0 and out.size[1] > 0


def test_crop_distinct_boxes_on_same_frame_use_unique_paths(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    first = view_crop(
        tool_scene,
        ViewCropArgs(frame_id=3, bbox=[0.0, 0.0, 0.7, 0.7]),
        out_dir=tmp_path,
    )
    second = view_crop(
        tool_scene,
        ViewCropArgs(frame_id=3, bbox=[0.25, 0.25, 1.0, 1.0]),
        out_dir=tmp_path,
    )
    assert first.image_path != second.image_path
    assert first.image_path.exists()
    assert second.image_path.exists()


def test_crop_bad_frame_is_recoverable(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    with pytest.raises(ToolInputError, match="not in this clip"):
        view_crop(
            tool_scene, ViewCropArgs(frame_id=999, bbox=[0, 0, 1, 1]), out_dir=tmp_path
        )


def test_crop_requires_target(tool_scene: OpenEqaToolScene, tmp_path: Path) -> None:
    with pytest.raises(ToolInputError, match="invalid arguments"):
        run_tool(tool_scene, "view_crop", {}, out_dir=tmp_path)


def test_crop_bbox_needs_four_values(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    with pytest.raises(ToolInputError, match="invalid arguments"):
        run_tool(
            tool_scene,
            "view_crop",
            {"frame_id": 1, "bbox": [0.1, 0.2]},
            out_dir=tmp_path,
        )


def test_dispatch_routes_view_crop(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    payload = run_tool(
        tool_scene,
        "view_crop",
        {"frame_id": 0, "bbox": [0.2, 0.2, 0.8, 0.8]},
        out_dir=tmp_path,
    ).to_payload()
    assert Path(payload["image_path"]).exists()


# ----- object mode (real ConceptGraph selector; needs NumPy + OpenCV) ---------


def test_crop_by_object_uses_detection_box(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    pytest.importorskip("cv2")
    result = view_crop(tool_scene, ViewCropArgs(object_id=1), out_dir=tmp_path)
    payload = result.to_payload()
    assert payload["source"] == "object"
    assert payload["object_id"] == 1
    assert payload["object_category"] == "chair"
    assert Path(payload["image_path"]).exists()
    assert payload["frame_id"] in {0, 5}


def test_crop_by_object_with_explicit_frame_id(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    pytest.importorskip("cv2")
    result = view_crop(
        tool_scene, ViewCropArgs(object_id=1, frame_id=5), out_dir=tmp_path
    )
    payload = result.to_payload()
    assert payload["source"] == "object"
    assert payload["object_id"] == 1
    assert payload["frame_id"] == 5
    assert Path(payload["image_path"]).exists()


def test_best_view_and_box_continues_past_view_without_detection() -> None:
    # Regression test: if ranked[0] (view 0) has no 2D detection box in image_idx,
    # _best_view_and_box must continue to ranked[1] (view 1) instead of returning None!
    selector = SimpleNamespace(object_to_views={1: [(0, 0.95), (1, 0.85)]})
    obj = SimpleNamespace(
        obj_id=1,
        image_idx=[1],
        xyxy=[[100.0, 200.0, 300.0, 400.0]],
    )
    vid, box = _best_view_and_box(selector, obj)
    assert vid == 1
    assert box == (100.0, 200.0, 300.0, 400.0)


def test_scale_detection_box_preserves_native_1296x968_coords() -> None:
    # Regression test: when raw image is 1296x968 and box is in 1296x968,
    # _scale_detection_box must NOT distort Y by 968/680!
    scaled = _scale_detection_box((100.0, 400.0, 300.0, 800.0), 1296, 968, margin=0.0)
    assert scaled == (100.0, 400.0, 300.0, 800.0)

    # When raw image is 640x480 and detection box exceeds 640x480 (in 1296x968 space),
    # it scales proportionally to 640x480.
    scaled_640 = _scale_detection_box(
        (648.0, 484.0, 1296.0, 968.0), 640, 480, margin=0.0
    )
    assert scaled_640 == pytest.approx((320.0, 240.0, 640.0, 480.0))


def test_crop_unknown_object_is_recoverable(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    pytest.importorskip("cv2")
    with pytest.raises(ToolInputError, match="not in this scene"):
        view_crop(tool_scene, ViewCropArgs(object_id=99), out_dir=tmp_path)
