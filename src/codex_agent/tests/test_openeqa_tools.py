"""Unit tests for the OpenEQA agent tool layer (frame/object/bev/keyframe + dispatch).

``view_frame`` needs Pillow; ``list_objects`` / ``view_bev`` build a real
ConceptGraph selector (NumPy) and ``view_bev`` additionally needs OpenCV. The
file is skipped when those optional deps are absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("cv2")
pytest.importorskip("PIL")

from codex_agent.openeqa.tools import bev_tools  # noqa: E402
from codex_agent.openeqa.tools import keyframe_retrieval as kf  # noqa: E402
from codex_agent.openeqa.tools.bev_tools import ViewBevArgs, view_bev  # noqa: E402
from codex_agent.openeqa.tools.dispatch import (  # noqa: E402
    TOOL_NAMES,
    run_tool,
)
from codex_agent.openeqa.tools.frame_tools import (  # noqa: E402
    ViewFrameArgs,
    view_frame,
)
from codex_agent.openeqa.tools.keyframe_retrieval import (  # noqa: E402
    KeyframeSelectorArgs,
    keyframe_selector,
)
from codex_agent.openeqa.tools.models import ToolInputError  # noqa: E402
from codex_agent.openeqa.tools.object_tools import (  # noqa: E402
    ListObjectsArgs,
    list_objects,
)
from codex_agent.openeqa.tools.scene_context import OpenEqaToolScene  # noqa: E402
from codex_agent.tests.conftest import (  # noqa: E402
    OpenEqaFixture,
    OpenEqaToolsFixture,
)


@pytest.fixture
def tool_scene(openeqa_tools_fixture: OpenEqaToolsFixture) -> OpenEqaToolScene:
    return OpenEqaToolScene.load(openeqa_tools_fixture.scene_dir)


@pytest.fixture
def raw_only_scene(openeqa_fixture: OpenEqaFixture) -> OpenEqaToolScene:
    """A clip with raw frames but no ConceptGraph pack."""
    return OpenEqaToolScene.load(openeqa_fixture.data_root / openeqa_fixture.clip_id)


# ----- view_frame -------------------------------------------------------------


def test_view_frame_single(tool_scene: OpenEqaToolScene, tmp_path: Path) -> None:
    result = view_frame(tool_scene, ViewFrameArgs(frame_id=3), out_dir=tmp_path)
    payload = result.to_payload()
    assert payload["total_frames"] == 10
    assert payload["frame_id_range"] == [0, 9]
    assert len(payload["frames"]) == 1
    frame = payload["frames"][0]
    assert frame["frame_id"] == 3
    assert Path(frame["image_path"]).exists()


def test_view_frame_multiple_dedupes(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    result = view_frame(
        tool_scene, ViewFrameArgs(frame_ids=[0, 5, 5, 9]), out_dir=tmp_path
    )
    ids = [f.frame_id for f in result.frames]
    assert ids == [0, 5, 9]


def test_view_frame_caps_at_four(tool_scene: OpenEqaToolScene, tmp_path: Path) -> None:
    result = view_frame(
        tool_scene, ViewFrameArgs(frame_ids=[0, 1, 2, 3, 4, 5]), out_dir=tmp_path
    )
    assert len(result.frames) == 4
    assert "showing 4 of 6" in result.note


def test_view_frame_unknown_id_is_recoverable(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    with pytest.raises(ToolInputError, match="valid range is"):
        view_frame(tool_scene, ViewFrameArgs(frame_id=999), out_dir=tmp_path)


def test_view_frame_requires_a_frame(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    with pytest.raises(ToolInputError, match="needs a frame"):
        view_frame(tool_scene, ViewFrameArgs(), out_dir=tmp_path)


# ----- list_objects -----------------------------------------------------------


def test_list_objects_all(tool_scene: OpenEqaToolScene) -> None:
    result = list_objects(tool_scene, ListObjectsArgs())
    payload = result.to_payload()
    assert payload["total"] == 3
    categories = {entry["category"] for entry in payload["objects"]}
    assert categories == {"desk", "chair", "door"}
    desk = next(e for e in payload["objects"] if e["category"] == "desk")
    assert "size" in desk and len(desk["size"]) == 3


def test_list_objects_category_filter(tool_scene: OpenEqaToolScene) -> None:
    result = list_objects(tool_scene, ListObjectsArgs(category="chair"))
    assert result.count == 1
    assert result.objects[0].obj_id == 1


def test_list_objects_limit(tool_scene: OpenEqaToolScene) -> None:
    result = list_objects(tool_scene, ListObjectsArgs(limit=2))
    assert result.total == 3
    assert result.count == 2


def test_list_objects_missing_category_is_recoverable(
    tool_scene: OpenEqaToolScene,
) -> None:
    with pytest.raises(ToolInputError, match="no objects match category"):
        list_objects(tool_scene, ListObjectsArgs(category="spaceship"))


def test_list_objects_without_conceptgraph(
    raw_only_scene: OpenEqaToolScene,
) -> None:
    with pytest.raises(ToolInputError, match="no ConceptGraph assets"):
        list_objects(raw_only_scene, ListObjectsArgs())


# ----- view_bev ---------------------------------------------------------------


def test_view_bev_default(tool_scene: OpenEqaToolScene, tmp_path: Path) -> None:
    result = view_bev(tool_scene, ViewBevArgs(), out_dir=tmp_path)
    payload = result.to_payload()
    assert payload["view"] == "default"
    assert payload["highlight_ids"] == []
    assert Path(payload["image_path"]).exists()


def test_view_bev_highlight_ids(tool_scene: OpenEqaToolScene, tmp_path: Path) -> None:
    result = view_bev(tool_scene, ViewBevArgs(highlight=[0, 1]), out_dir=tmp_path)
    assert result.highlight_ids == (0, 1)
    assert result.to_payload()["view"] == "highlighted"
    assert Path(result.image_path).exists()


def test_view_bev_category_highlight(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    result = view_bev(tool_scene, ViewBevArgs(categories=["chair"]), out_dir=tmp_path)
    assert result.highlight_ids == (1,)


def test_view_bev_missing_category_reported(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    result = view_bev(tool_scene, ViewBevArgs(categories=["sofa"]), out_dir=tmp_path)
    assert result.missing_categories == ("sofa",)
    assert result.highlight_ids == ()


def test_view_bev_unknown_id_is_recoverable(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    with pytest.raises(ToolInputError, match="highlight ids not in this scene"):
        view_bev(tool_scene, ViewBevArgs(highlight=[99]), out_dir=tmp_path)


def test_view_bev_without_conceptgraph(
    raw_only_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    with pytest.raises(ToolInputError, match="no ConceptGraph assets"):
        view_bev(raw_only_scene, ViewBevArgs(), out_dir=tmp_path)


def test_view_bev_render_failure_is_recoverable(
    tool_scene: OpenEqaToolScene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BoomSelector:
        objects: list[Any] = []

        def generate_scene_bev(self, **kwargs: Any) -> Path:
            raise RuntimeError("render failed")

    monkeypatch.setattr(
        bev_tools, "build_selector", lambda cg, model=None: _BoomSelector()
    )
    with pytest.raises(ToolInputError, match="could not render the BEV"):
        view_bev(tool_scene, ViewBevArgs(), out_dir=tmp_path)


# ----- keyframe_selector (LLM path stubbed) ----------------------------------


class _FakeKeyframeResult:
    def __init__(self, paths: list[str]) -> None:
        self.keyframe_paths = paths
        self.target_term = "chair"
        self.anchor_term = "desk"


class _FakeSelector:
    def __init__(self, paths: list[str], *, fail: bool = False) -> None:
        self._paths = paths
        self._fail = fail

    def select_keyframes_v2(self, query: str, **kwargs: Any) -> _FakeKeyframeResult:
        if self._fail:
            raise RuntimeError("boom")
        k = int(kwargs.get("k", len(self._paths)))
        return _FakeKeyframeResult(self._paths[:k])


def test_keyframe_selector_maps_frames(
    tool_scene: OpenEqaToolScene,
    openeqa_tools_fixture: OpenEqaToolsFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = openeqa_tools_fixture.scene_dir / "raw"
    paths = [str(raw / "000003-rgb.png"), str(raw / "000006-rgb.png")]
    monkeypatch.setattr(
        kf, "build_selector", lambda cg, model=None: _FakeSelector(paths)
    )
    result = keyframe_selector(
        tool_scene,
        KeyframeSelectorArgs(query="the office chair", k=2),
        out_dir=tmp_path,
    )
    payload = result.to_payload()
    assert [f["frame_id"] for f in payload["frames"]] == [3, 6]
    assert "chair" in payload["hypothesis_summary"]
    assert all(Path(f["image_path"]).exists() for f in payload["frames"])


def test_keyframe_selector_failure_is_recoverable(
    tool_scene: OpenEqaToolScene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        kf, "build_selector", lambda cg, model=None: _FakeSelector([], fail=True)
    )
    with pytest.raises(ToolInputError, match="keyframe_selector failed"):
        keyframe_selector(tool_scene, KeyframeSelectorArgs(query="x"), out_dir=tmp_path)


def test_keyframe_selector_without_conceptgraph(
    raw_only_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    with pytest.raises(ToolInputError, match="no ConceptGraph assets"):
        keyframe_selector(
            raw_only_scene, KeyframeSelectorArgs(query="x"), out_dir=tmp_path
        )


# ----- dispatch ---------------------------------------------------------------


def test_dispatch_routes_view_frame(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    payload = run_tool(
        tool_scene, "view_frame", {"frame_id": 0}, out_dir=tmp_path
    ).to_payload()
    assert payload["frames"][0]["frame_id"] == 0


def test_dispatch_routes_list_objects(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    payload = run_tool(
        tool_scene, "list_objects", {"limit": 2}, out_dir=tmp_path
    ).to_payload()
    assert payload["total"] == 3


def test_dispatch_routes_view_bev(tool_scene: OpenEqaToolScene, tmp_path: Path) -> None:
    payload = run_tool(tool_scene, "view_bev", {}, out_dir=tmp_path).to_payload()
    assert Path(payload["image_path"]).exists()


def test_dispatch_routes_keyframe_selector(
    tool_scene: OpenEqaToolScene,
    openeqa_tools_fixture: OpenEqaToolsFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = openeqa_tools_fixture.scene_dir / "raw"
    monkeypatch.setattr(
        kf,
        "build_selector",
        lambda cg, model=None: _FakeSelector([str(raw / "000001-rgb.png")]),
    )
    payload = run_tool(
        tool_scene, "keyframe_selector", {"query": "the chair"}, out_dir=tmp_path
    ).to_payload()
    assert payload["frames"][0]["frame_id"] == 1


def test_dispatch_unknown_tool(tool_scene: OpenEqaToolScene, tmp_path: Path) -> None:
    with pytest.raises(ToolInputError, match="unknown tool"):
        run_tool(tool_scene, "nope", {}, out_dir=tmp_path)


def test_dispatch_validates_arguments(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    with pytest.raises(ToolInputError, match="invalid arguments"):
        run_tool(tool_scene, "view_frame", {"frame_id": "x"}, out_dir=tmp_path)


def test_tool_names_are_stable() -> None:
    assert set(TOOL_NAMES) == {
        "list_objects",
        "keyframe_selector",
        "view_frame",
        "view_bev",
    }
