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


def _write_source_frame_scene(root: Path, source_frames: object) -> Path:
    scene_dir = root / "421254"
    raw_dir = scene_dir / "raw"
    conceptgraph_vis_dir = scene_dir / "conceptgraph" / "gsa_vis_ram_withbg_allclasses"
    raw_dir.mkdir(parents=True)
    conceptgraph_vis_dir.mkdir(parents=True)
    (raw_dir / "source_frames.json").write_text(
        json.dumps(source_frames), encoding="utf-8"
    )
    return scene_dir


def _source_frame_records() -> list[dict[str, str]]:
    return [
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


def test_cli_scene_summary_reads_source_frame_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_source_frame_scene(tmp_path, _source_frame_records())
    code = main(["scene_summary", "--scene-root", str(scene_dir), "--args", "{}"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["visit_id"] == "421254"
    assert payload["total_rgb_frames"] == 2
    assert payload["rgb_frame_ids"] == ["000000", "000010"]
    assert payload["has_conceptgraph"] is True


def test_cli_source_frame_list_rejects_invalid_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_frames = [
        {"frame_id": "000000"},
        {"timestamp": "80967.880"},
    ]
    scene_dir = _write_source_frame_scene(tmp_path, source_frames)

    with pytest.raises(SystemExit) as excinfo:
        main(["scene_summary", "--scene-root", str(scene_dir), "--args", "{}"])

    assert excinfo.value.code == 1
    assert "source frame record 1" in capsys.readouterr().err


def test_cli_source_frame_mapping_rejects_invalid_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_frames = {
        "000000": {"timestamp": "80966.897"},
        "bad-entry": {"timestamp": "80967.880"},
    }
    scene_dir = _write_source_frame_scene(tmp_path, source_frames)

    with pytest.raises(SystemExit) as excinfo:
        main(["scene_summary", "--scene-root", str(scene_dir), "--args", "{}"])

    assert excinfo.value.code == 1
    assert "source frame entry 'bad-entry'" in capsys.readouterr().err


def test_cli_source_frame_index_rejects_duplicate_frame_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_frames = [
        {"frame_id": "000000"},
        {"rgb": "/datasets/SceneFunVal/421254/42444754/hires_wide/000000.jpg"},
    ]
    scene_dir = _write_source_frame_scene(tmp_path, source_frames)

    with pytest.raises(SystemExit) as excinfo:
        main(["scene_summary", "--scene-root", str(scene_dir), "--args", "{}"])

    assert excinfo.value.code == 1
    assert "duplicate frame id '000000'" in capsys.readouterr().err


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


def test_cli_view_frame_returns_image_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    scene_dir = _write_raw_rgb_scene(tmp_path)
    Image.new("RGB", (12, 10), color=(10, 20, 30)).save(
        scene_dir / "raw" / "000000-rgb.png"
    )

    code = main(
        [
            "view_frame",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_ids": ["000000"]}),
            "--out-dir",
            str(tmp_path / "scratch"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["frames"][0]["frame_id"] == "000000"
    assert Path(payload["frames"][0]["image_path"]).exists()


def test_cli_view_frame_uses_conceptgraph_visualization_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    scene_dir = _write_source_frame_scene(tmp_path, _source_frame_records())
    vis_dir = scene_dir / "conceptgraph" / "gsa_vis_ram_withbg_allclasses"
    Image.new("RGB", (16, 12), color=(30, 40, 50)).save(vis_dir / "000000-rgb.jpg")

    code = main(
        [
            "view_frame",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_ids": ["000000"]}),
            "--out-dir",
            str(tmp_path / "scratch"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["frames"][0]["frame_id"] == "000000"
    assert Path(payload["frames"][0]["image_path"]).exists()


def test_cli_view_frame_uses_source_frame_raw_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    source_rgb_path = tmp_path / "source_rgb" / "42444754_80966.897.jpg"
    source_rgb_path.parent.mkdir()
    Image.new("RGB", (18, 14), color=(90, 80, 70)).save(source_rgb_path)
    scene_dir = _write_source_frame_scene(
        tmp_path,
        [
            {
                "frame_id": "000000",
                "rgb": str(source_rgb_path),
            }
        ],
    )

    code = main(
        [
            "view_frame",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_ids": ["000000"]}),
            "--out-dir",
            str(tmp_path / "scratch"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["frames"][0]["frame_id"] == "000000"
    assert Path(payload["frames"][0]["image_path"]).exists()


def test_cli_view_frame_prefers_conceptgraph_over_source_frame_raw_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    source_rgb_path = tmp_path / "source_rgb" / "42444754_80966.897.jpg"
    source_rgb_path.parent.mkdir()
    Image.new("RGB", (18, 14), color=(200, 10, 10)).save(source_rgb_path)
    scene_dir = _write_source_frame_scene(
        tmp_path,
        [
            {
                "frame_id": "000000",
                "rgb": str(source_rgb_path),
            }
        ],
    )
    vis_dir = scene_dir / "conceptgraph" / "gsa_vis_ram_withbg_allclasses"
    Image.new("RGB", (18, 14), color=(10, 200, 10)).save(vis_dir / "000000-rgb.jpg")

    code = main(
        [
            "view_frame",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_ids": ["000000"]}),
            "--out-dir",
            str(tmp_path / "scratch"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    output_path = Path(payload["frames"][0]["image_path"])
    with Image.open(output_path) as image:
        red, green, blue = image.convert("RGB").getpixel((0, 0))
    assert green > red
    assert green > blue


def test_cli_view_frame_missing_image_is_recoverable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_source_frame_scene(tmp_path, _source_frame_records())

    code = main(
        [
            "view_frame",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_ids": ["000000"]}),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "no RGB image found" in payload["error"]


def test_cli_frame_objects_returns_empty_visible_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)

    code = main(
        [
            "frame_objects",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_id": "000000"}),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload == {"frame_id": "000000", "objects": []}


def test_cli_view_crop_requires_frame_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)

    code = main(
        [
            "view_crop",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"bbox": [0.1, 0.1, 0.5, 0.5]}),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "frame_id" in payload["error"]


def test_cli_view_bev_reports_unavailable_without_bev_asset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)

    code = main(["view_bev", "--scene-root", str(scene_dir), "--args", "{}"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "BEV asset is not available" in payload["error"]


def test_cli_keyframe_selector_returns_first_k_frames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)

    code = main(
        [
            "keyframe_selector",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"query": "drawer handle", "k": 1}),
            "--out-dir",
            str(tmp_path / "scratch"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["query"] == "drawer handle"
    assert payload["frames"] == [{"frame_id": "000000", "rank": 1}]


def test_cli_keyframe_selector_rejects_blank_query(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)

    code = main(
        [
            "keyframe_selector",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"query": "   ", "k": 1}),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "query" in payload["error"]


def test_cli_keyframe_selector_rejects_string_k(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)

    code = main(
        [
            "keyframe_selector",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"query": "drawer handle", "k": "2"}),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "k" in payload["error"]
