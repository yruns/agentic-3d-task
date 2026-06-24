"""Unit tests for keyframe_selector diversity / fallback / contact sheet.

These exercise the v4-trace-analysis fixes without the network LLM: a fake
selector stands in for the ConceptGraph + query-parser path, exposing only the
public attributes the tool reads (``camera_poses``, ``stride``, ``objects``,
``object_to_views``, ``select_keyframes_v2``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("PIL")

import numpy as np  # noqa: E402

from codex_agent.openeqa.tools import keyframe_retrieval as kf  # noqa: E402
from codex_agent.openeqa.tools.frame_tools import (  # noqa: E402
    ViewFrameArgs,
    view_frame,
)
from codex_agent.openeqa.tools.keyframe_retrieval import (  # noqa: E402
    KeyframeSelectorArgs,
    keyframe_selector,
)
from codex_agent.openeqa.tools.scene_context import OpenEqaToolScene  # noqa: E402
from codex_agent.tests.conftest import OpenEqaToolsFixture  # noqa: E402


@pytest.fixture
def tool_scene(openeqa_tools_fixture: OpenEqaToolsFixture) -> OpenEqaToolScene:
    return OpenEqaToolScene.load(openeqa_tools_fixture.scene_dir)


def _pose(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [x, 0.0, 0.0]
    return pose


class _Result:
    def __init__(self, paths: list[str], *, anchor: str | None = "desk") -> None:
        self.keyframe_paths = paths
        self.target_term = "chair"
        self.anchor_term = anchor


class _FakeSelector:
    """Minimal stand-in exposing only what keyframe_selector reads."""

    def __init__(
        self,
        *,
        positions: list[float],
        paths: list[str],
        objects: list[Any] | None = None,
        object_to_views: dict[int, list[tuple[int, float]]] | None = None,
        empty: bool = False,
        stride: int = 1,
    ) -> None:
        self.stride = stride
        self.camera_poses = [_pose(x) for x in positions]
        self.objects = objects or []
        self.object_to_views = object_to_views or {}
        self._paths = paths
        self._empty = empty

    def select_keyframes_v2(self, query: str, **kwargs: Any) -> _Result:
        if self._empty:
            return _Result([], anchor=None)
        k = int(kwargs.get("k", len(self._paths)))
        return _Result(self._paths[:k])


def _raw(fixture: OpenEqaToolsFixture, frame_id: int) -> str:
    return str(fixture.scene_dir / "raw" / f"{frame_id:06d}-rgb.png")


def test_spatial_dedup_drops_near_duplicate_views(
    tool_scene: OpenEqaToolScene,
    openeqa_tools_fixture: OpenEqaToolsFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # views 0 and 1 share a camera position -> 1 should be dropped as a duplicate
    fake = _FakeSelector(
        positions=[0.0, 0.0, 1.0, 2.0],
        paths=[_raw(openeqa_tools_fixture, i) for i in range(4)],
    )
    monkeypatch.setattr(kf, "build_selector", lambda cg, model=None: fake)
    result = keyframe_selector(
        tool_scene, KeyframeSelectorArgs(query="the chair", k=3), out_dir=tmp_path
    )
    ids = [f.frame_id for f in result.frames]
    assert ids == [0, 2, 3]
    assert 1 not in ids


def test_keyframe_selector_emits_contact_sheet(
    tool_scene: OpenEqaToolScene,
    openeqa_tools_fixture: OpenEqaToolsFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSelector(
        positions=[0.0, 1.0, 2.0],
        paths=[_raw(openeqa_tools_fixture, i) for i in range(3)],
    )
    monkeypatch.setattr(kf, "build_selector", lambda cg, model=None: fake)
    payload = keyframe_selector(
        tool_scene, KeyframeSelectorArgs(query="the chair", k=3), out_dir=tmp_path
    ).to_payload()
    assert "contact_sheet" in payload
    assert Path(payload["contact_sheet"]).exists()


def test_grounding_fallback_uses_category_visibility(
    tool_scene: OpenEqaToolScene,
    openeqa_tools_fixture: OpenEqaToolsFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = [SimpleNamespace(obj_id=5, object_tag="chair", category="chair")]
    fake = _FakeSelector(
        positions=[0.0, 1.0, 2.0, 3.0],
        paths=[],
        objects=objects,
        object_to_views={5: [(2, 0.9), (0, 0.8)]},
        empty=True,
    )
    monkeypatch.setattr(kf, "build_selector", lambda cg, model=None: fake)
    result = keyframe_selector(
        tool_scene,
        KeyframeSelectorArgs(query="where is the chair", k=2),
        out_dir=tmp_path,
    )
    assert result.frames  # non-empty despite empty grounding
    assert "category-visibility" in result.to_payload()["fallback"]
    assert {f.frame_id for f in result.frames} <= {0, 2}


def test_grounding_fallback_sweeps_trajectory(
    tool_scene: OpenEqaToolScene,
    openeqa_tools_fixture: OpenEqaToolsFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSelector(positions=[0.0, 1.0, 2.0, 3.0], paths=[], empty=True)
    monkeypatch.setattr(kf, "build_selector", lambda cg, model=None: fake)
    result = keyframe_selector(
        tool_scene,
        KeyframeSelectorArgs(query="zzz nothing matches", k=3),
        out_dir=tmp_path,
    )
    assert result.frames
    assert "trajectory sweep" in result.to_payload()["fallback"]


def test_cross_call_dedup_returns_new_views(
    tool_scene: OpenEqaToolScene,
    openeqa_tools_fixture: OpenEqaToolsFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CODEX_HOME present -> per-turn cross-call memory is enabled
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    fake = _FakeSelector(
        positions=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        paths=[_raw(openeqa_tools_fixture, i) for i in range(6)],
    )
    monkeypatch.setattr(kf, "build_selector", lambda cg, model=None: fake)
    args = KeyframeSelectorArgs(query="the chair", k=3)
    first = keyframe_selector(tool_scene, args, out_dir=tmp_path)
    second = keyframe_selector(tool_scene, args, out_dir=tmp_path)
    first_ids = {f.frame_id for f in first.frames}
    second_ids = {f.frame_id for f in second.frames}
    assert first_ids == {0, 1, 2}
    assert second_ids.isdisjoint(first_ids)


def test_view_frame_emits_contact_sheet(
    tool_scene: OpenEqaToolScene, tmp_path: Path
) -> None:
    result = view_frame(
        tool_scene, ViewFrameArgs(frame_ids=[0, 4, 8]), out_dir=tmp_path
    )
    payload = result.to_payload()
    assert "contact_sheet" in payload
    assert Path(payload["contact_sheet"]).exists()
