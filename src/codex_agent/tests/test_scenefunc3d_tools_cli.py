"""Tests for the SceneFunc3D tools CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_agent.scenefunc3d.tools.__main__ import main


def _write_raw_rgb_scene(root: Path) -> Path:
    scene_dir = root / "421254"
    raw_dir = scene_dir / "raw"
    conceptgraph_dir = scene_dir / "conceptgraph"
    raw_dir.mkdir(parents=True)
    conceptgraph_dir.mkdir()
    (raw_dir / "000000-rgb.png").write_bytes(b"not-a-real-image")
    (raw_dir / "000010-rgb.png").write_bytes(b"not-a-real-image")
    (conceptgraph_dir / "indices").mkdir()
    return scene_dir


def _write_source_frame_scene(root: Path) -> Path:
    scene_dir = root / "421254"
    raw_dir = scene_dir / "raw"
    conceptgraph_vis_dir = scene_dir / "conceptgraph" / "gsa_vis_ram_withbg_allclasses"
    raw_dir.mkdir(parents=True)
    conceptgraph_vis_dir.mkdir(parents=True)
    source_frames = [
        {
            "frame_id": "000000",
            "visit_id": "421254",
            "video_id": "42444754",
            "timestamp": "80966.897",
            "rgb": "/datasets/SceneFunVal/421254/42444754/hires_wide/42444754_80966.897.jpg",
            "depth": "/datasets/SceneFunVal/421254/42444754/hires_depth/42444754_80966.897.png",
            "intrinsic": "/datasets/SceneFunVal/421254/42444754/hires_wide_intrinsics/42444754_80966.897.pincam",
        },
        {
            "frame_id": "000010",
            "visit_id": "421254",
            "video_id": "42444754",
            "timestamp": "80976.882",
            "rgb": "/datasets/SceneFunVal/421254/42444754/hires_wide/42444754_80976.882.jpg",
            "depth": "/datasets/SceneFunVal/421254/42444754/hires_depth/42444754_80976.882.png",
            "intrinsic": "/datasets/SceneFunVal/421254/42444754/hires_wide_intrinsics/42444754_80976.882.pincam",
        },
    ]
    (raw_dir / "source_frames.json").write_text(
        json.dumps(source_frames), encoding="utf-8"
    )
    return scene_dir


def test_cli_scene_summary_reads_source_frame_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_source_frame_scene(tmp_path)
    code = main(["scene_summary", "--scene-root", str(scene_dir), "--args", "{}"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["visit_id"] == "421254"
    assert payload["total_rgb_frames"] == 2
    assert payload["rgb_frame_ids"] == ["000000", "000010"]
    assert payload["has_conceptgraph"] is True


def test_cli_scene_summary_falls_back_to_raw_rgb_frames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    code = main(["scene_summary", "--scene-root", str(scene_dir), "--args", "{}"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["visit_id"] == "421254"
    assert payload["total_rgb_frames"] == 2
    assert payload["rgb_frame_ids"] == ["000000", "000010"]
    assert payload["has_conceptgraph"] is True


def test_cli_bad_args_json_is_recoverable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    code = main(["scene_summary", "--scene-root", str(scene_dir), "--args", "not-json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "must be valid JSON" in payload["error"]


def test_cli_missing_scene_root_exits_nonzero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            ["scene_summary", "--scene-root", str(tmp_path / "missing"), "--args", "{}"]
        )
    assert excinfo.value.code == 1
