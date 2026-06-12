"""Unit tests for the OpenEQA ``view_crop`` zoom tool."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from codex_agent.openeqa.tools.crop_tools import (  # noqa: E402
    ViewCropArgs,
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


def test_crop_by_pixel_bbox(tool_scene: OpenEqaToolScene, tmp_path: Path) -> None:
    result = view_crop(
        tool_scene, ViewCropArgs(frame_id=2, bbox=[4, 4, 40, 40]), out_dir=tmp_path
    )
    assert result.source == "bbox"
    with Image.open(result.image_path) as out:
        assert out.size[0] > 0 and out.size[1] > 0


def test_crop_bad_frame_is_recoverable(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    with pytest.raises(ToolInputError, match="not in this clip"):
        view_crop(
            tool_scene, ViewCropArgs(frame_id=999, bbox=[0, 0, 1, 1]), out_dir=tmp_path
        )


def test_crop_requires_target(tool_scene: OpenEqaToolScene, tmp_path: Path) -> None:
    # no object_id and no (frame_id + bbox) -> validation error surfaced as ToolInputError
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
    assert Path(payload["image_path"]).exists()
    # chair (obj 1) best view is 0 -> frame 0 (view * stride)
    assert payload["frame_id"] in {0, 5}


def test_crop_unknown_object_is_recoverable(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    pytest.importorskip("cv2")
    with pytest.raises(ToolInputError, match="not in this scene"):
        view_crop(tool_scene, ViewCropArgs(object_id=99), out_dir=tmp_path)
