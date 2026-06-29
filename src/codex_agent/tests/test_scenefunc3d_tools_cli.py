"""Tests for the SceneFunc3D tools CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_agent.scenefunc3d.task import ApprovalAction
from codex_agent.scenefunc3d.tools.__main__ import main

_APPROVED_FRAGMENT_ACTIONS = (
    ApprovalAction.SELECT_EVIDENCE.value,
    ApprovalAction.PROPOSE_MOLMO_POINT.value,
    ApprovalAction.APPROVE_MOLMO_POINT.value,
    ApprovalAction.PROPOSE_SAM_CANDIDATES.value,
    ApprovalAction.APPROVE_SAM_CANDIDATE.value,
    ApprovalAction.CREATE_FIRST_LIFT.value,
    ApprovalAction.APPROVE_FIRST_LIFT.value,
)


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
    assert payload["has_bev"] is False
    assert "bev_path" not in payload
    assert payload["has_scene_mesh"] is False
    assert "scene_mesh_path" not in payload
    assert payload["has_visible_object_index"] is False
    assert "visible_object_index_path" not in payload
    assert payload["visible_object_frame_count"] == 0
    assert payload["visible_object_total_count"] == 0


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


def test_cli_scene_summary_exposes_indexed_evidence_assets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    bev_path = scene_dir / "conceptgraph" / "bev" / "scene_bev.png"
    mesh_path = scene_dir / "conceptgraph" / "mesh.ply"
    object_frame_map_path = (
        scene_dir / "conceptgraph" / "indices" / "object_frame_map.json"
    )
    bev_path.parent.mkdir()
    bev_path.write_bytes(b"bev")
    mesh_path.write_bytes(b"ply\n")
    object_frame_map_path.write_text(
        json.dumps(
            {
                "frame_to_objects": {
                    "0": {
                        "view_id": 0,
                        "frame_name": "000000-rgb.jpg",
                        "objects": [
                            {
                                "object_id": 12,
                                "class_name": "drawer",
                                "score": 0.87,
                            },
                            {
                                "object_id": 13,
                                "class_name": "cabinet",
                                "score": 0.42,
                            },
                        ],
                    },
                    "1": {
                        "view_id": 1,
                        "frame_name": "000010-rgb.jpg",
                        "objects": [
                            {
                                "object_id": 14,
                                "class_name": "handle",
                                "score": 0.91,
                            }
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    code = main(["scene_summary", "--scene-root", str(scene_dir), "--args", "{}"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["has_bev"] is True
    assert payload["bev_path"] == str(bev_path)
    assert payload["has_scene_mesh"] is True
    assert payload["scene_mesh_path"] == str(mesh_path)
    assert payload["has_visible_object_index"] is True
    assert payload["visible_object_index_path"] == str(object_frame_map_path)
    assert payload["visible_object_frame_count"] == 2
    assert payload["visible_object_total_count"] == 3


def test_cli_scene_summary_validates_visible_object_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    object_frame_map_path = (
        scene_dir / "conceptgraph" / "indices" / "object_frame_map.json"
    )
    object_frame_map_path.write_text(
        json.dumps(
            {
                "frame_to_objects": {
                    "0": {
                        "view_id": 0,
                        "frame_name": "000000-rgb.jpg",
                        "objects": [
                            {
                                "object_id": 12,
                                "class_name": "drawer",
                                "score": float("inf"),
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    code = main(["scene_summary", "--scene-root", str(scene_dir), "--args", "{}"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "visible-object index failed validation" in payload["error"]
    assert "score" in payload["error"]


def test_cli_bad_args_json_is_recoverable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    code = main(["scene_summary", "--scene-root", str(scene_dir), "--args", "not-json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "must be valid JSON" in payload["error"]


def test_cli_writes_tool_completed_event(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    out_dir = tmp_path / "run"

    code = main(
        [
            "scene_summary",
            "--scene-root",
            str(scene_dir),
            "--args",
            "{}",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    event_lines = (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(event_lines) == 1
    event = json.loads(event_lines[0])
    assert event["event_type"] == "tool_completed"
    assert event["tool_name"] == "scene_summary"
    assert event["status"] == "success"
    assert event["args"] == {}
    assert event["error"] == ""
    assert event["result"]["visit_id"] == payload["visit_id"]
    assert event["result"]["rgb_frame_ids"] == payload["rgb_frame_ids"]


def test_cli_writes_tool_failed_event_for_recoverable_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_source_frame_scene(tmp_path, _source_frame_records())
    out_dir = tmp_path / "run"

    code = main(
        [
            "view_frame",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_ids": ["000000"]}),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    event_lines = (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(event_lines) == 1
    event = json.loads(event_lines[0])
    assert event["event_type"] == "tool_failed"
    assert event["tool_name"] == "view_frame"
    assert event["status"] == "failed"
    assert event["args"] == {"frame_ids": ["000000"]}
    assert event["result"] is None
    assert event["error"] == payload["error"]


def test_cli_appends_tool_events_for_multiple_invocations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    out_dir = tmp_path / "run"

    first_code = main(
        [
            "scene_summary",
            "--scene-root",
            str(scene_dir),
            "--args",
            "{}",
            "--out-dir",
            str(out_dir),
        ]
    )
    _ = capsys.readouterr()
    second_code = main(
        [
            "frame_objects",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_id": "000000"}),
            "--out-dir",
            str(out_dir),
        ]
    )
    _ = capsys.readouterr()

    assert first_code == 0
    assert second_code == 0
    event_lines = (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(event_lines) == 2
    first_event = json.loads(event_lines[0])
    second_event = json.loads(event_lines[1])
    assert first_event["event_type"] == "tool_completed"
    assert first_event["tool_name"] == "scene_summary"
    assert second_event["event_type"] == "tool_failed"
    assert second_event["tool_name"] == "frame_objects"
    assert second_event["args"] == {"frame_id": "000000"}


def test_cli_tool_failed_event_write_error_exits_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_source_frame_scene(tmp_path, _source_frame_records())
    out_dir = tmp_path / "not-a-directory"
    out_dir.write_text("file blocks event directory creation\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "view_frame",
                "--scene-root",
                str(scene_dir),
                "--args",
                json.dumps({"frame_ids": ["000000"]}),
                "--out-dir",
                str(out_dir),
            ]
        )

    assert excinfo.value.code == 1
    assert "could not append SceneFunc3D tool event" in capsys.readouterr().err


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
    assert payload["frames"][0]["image_width"] == 12
    assert payload["frames"][0]["image_height"] == 10


def test_cli_view_frame_exposes_raw_geometry_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    scene_dir = _write_raw_rgb_scene(tmp_path)
    raw_dir = scene_dir / "raw"
    Image.new("RGB", (12, 10), color=(10, 20, 30)).save(raw_dir / "000000-rgb.png")
    depth_path = raw_dir / "000000-depth.png"
    intrinsics_path = raw_dir / "intrinsics.txt"
    pose_path = raw_dir / "pose" / "000000.txt"
    depth_path.write_bytes(b"depth")
    intrinsics_path.write_text("1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
    pose_path.parent.mkdir()
    pose_path.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")

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
    frame_payload = payload["frames"][0]
    assert frame_payload["depth_path"] == str(depth_path)
    assert frame_payload["intrinsics_path"] == str(intrinsics_path)
    assert frame_payload["pose_path"] == str(pose_path)


def test_cli_view_frame_exposes_source_frame_geometry_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    source_root = tmp_path / "source"
    source_rgb_path = source_root / "frame.jpg"
    depth_path = source_root / "frame_depth.png"
    intrinsics_path = source_root / "frame_intrinsics.txt"
    pose_path = source_root / "frame_pose.txt"
    source_root.mkdir()
    Image.new("RGB", (18, 14), color=(90, 80, 70)).save(source_rgb_path)
    depth_path.write_bytes(b"depth")
    intrinsics_path.write_text("1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
    pose_path.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")
    scene_dir = _write_source_frame_scene(
        tmp_path,
        [
            {
                "frame_id": "000000",
                "rgb": str(source_rgb_path),
                "depth": str(depth_path),
                "intrinsics": str(intrinsics_path),
                "pose": str(pose_path),
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
    frame_payload = payload["frames"][0]
    assert frame_payload["depth_path"] == str(depth_path)
    assert frame_payload["intrinsics_path"] == str(intrinsics_path)
    assert frame_payload["pose_path"] == str(pose_path)


def test_cli_view_frame_exposes_mixed_raw_and_source_geometry_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    source_root = tmp_path / "source"
    source_rgb_path = source_root / "frame.jpg"
    source_intrinsics_path = source_root / "frame_intrinsics.txt"
    source_pose_path = source_root / "frame_pose.txt"
    source_root.mkdir()
    Image.new("RGB", (18, 14), color=(90, 80, 70)).save(source_rgb_path)
    source_intrinsics_path.write_text("1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
    source_pose_path.write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n",
        encoding="utf-8",
    )
    scene_dir = _write_source_frame_scene(
        tmp_path,
        [
            {
                "frame_id": "000000",
                "rgb": str(source_rgb_path),
                "intrinsics": str(source_intrinsics_path),
                "pose": str(source_pose_path),
            }
        ],
    )
    raw_depth_path = scene_dir / "raw" / "000000-depth.png"
    raw_depth_path.write_bytes(b"depth")

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
    frame_payload = payload["frames"][0]
    assert frame_payload["depth_path"] == str(raw_depth_path)
    assert frame_payload["intrinsics_path"] == str(source_intrinsics_path)
    assert frame_payload["pose_path"] == str(source_pose_path)


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


def test_cli_frame_objects_reports_unavailable_index(
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
    assert "frame_objects requires a visible-object index" in payload["error"]


def test_cli_frame_objects_reads_conceptgraph_object_frame_map(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    object_frame_map_path = (
        scene_dir / "conceptgraph" / "indices" / "object_frame_map.json"
    )
    object_frame_map_path.write_text(
        json.dumps(
            {
                "metadata": {"num_objects": 2, "num_views": 1},
                "frame_to_objects": {
                    "0": {
                        "view_id": 0,
                        "frame_name": "000000-rgb.jpg",
                        "num_objects": 2,
                        "objects": [
                            {
                                "object_id": 12,
                                "class_name": "drawer",
                                "score": 0.87,
                                "bbox_xyxy": [10, 20, 110, 120],
                            },
                            {
                                "object_id": 13,
                                "class_name": "cabinet",
                                "score": 0.42,
                                "bbox_xyxy": None,
                            },
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

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
    assert payload["frame_id"] == "000000"
    assert payload["objects"] == [
        {
            "object_id": "12",
            "label": "drawer",
            "score": 0.87,
            "bbox_xyxy": [10.0, 20.0, 110.0, 120.0],
            "bbox_format": "pixel_xyxy",
            "source": "object_frame_map",
        },
        {
            "object_id": "13",
            "label": "cabinet",
            "score": 0.42,
            "source": "object_frame_map",
        },
    ]


def test_cli_frame_objects_rejects_invalid_object_bbox(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    object_frame_map_path = (
        scene_dir / "conceptgraph" / "indices" / "object_frame_map.json"
    )
    object_frame_map_path.write_text(
        json.dumps(
            {
                "frame_to_objects": {
                    "0": {
                        "view_id": 0,
                        "frame_name": "000000-rgb.jpg",
                        "objects": [
                            {
                                "object_id": 12,
                                "class_name": "drawer",
                                "score": 0.87,
                                "bbox_xyxy": [50, 20, 10, 120],
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

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
    assert "visible-object index failed validation" in payload["error"]
    assert "bbox_xyxy" in payload["error"]


def test_cli_frame_objects_rejects_non_finite_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    object_frame_map_path = (
        scene_dir / "conceptgraph" / "indices" / "object_frame_map.json"
    )
    object_frame_map_path.write_text(
        json.dumps(
            {
                "frame_to_objects": {
                    "0": {
                        "view_id": 0,
                        "frame_name": "000000-rgb.jpg",
                        "objects": [
                            {
                                "object_id": 12,
                                "class_name": "drawer",
                                "score": float("inf"),
                                "bbox_xyxy": [10, 20, 50, 120],
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

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
    assert "visible-object index failed validation" in payload["error"]
    assert "score" in payload["error"]


def test_cli_frame_objects_prefers_frame_name_over_numeric_key_collision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    object_frame_map_path = (
        scene_dir / "conceptgraph" / "indices" / "object_frame_map.json"
    )
    object_frame_map_path.write_text(
        json.dumps(
            {
                "frame_to_objects": {
                    "10": {
                        "view_id": 10,
                        "frame_name": "000123-rgb.jpg",
                        "objects": [
                            {
                                "object_id": "wrong-frame",
                                "class_name": "lamp",
                                "score": 0.5,
                            }
                        ],
                    },
                    "4": {
                        "view_id": 4,
                        "frame_name": "000010-rgb.jpg",
                        "objects": [
                            {
                                "object_id": "correct-frame",
                                "class_name": "drawer",
                                "score": 0.9,
                            }
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "frame_objects",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_id": "000010"}),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["objects"] == [
        {
            "object_id": "correct-frame",
            "label": "drawer",
            "score": 0.9,
            "source": "object_frame_map",
        }
    ]


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


def test_cli_view_crop_rejects_invalid_bbox(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)

    code = main(
        [
            "view_crop",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_id": "000000", "bbox": [-1.0, 0.1, 2.0, 0.5]}),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "bbox" in payload["error"]


def test_cli_view_crop_renders_normalized_rgb_crop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    scene_dir = _write_raw_rgb_scene(tmp_path)
    image = Image.new("RGB", (100, 80), color=(220, 10, 10))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 0, 99, 79), fill=(10, 220, 10))
    image.save(scene_dir / "raw" / "000000-rgb.png")

    code = main(
        [
            "view_crop",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_id": "000000", "bbox": [0.5, 0.0, 0.9, 0.4]}),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    crop_path = Path(payload["frames"][0]["image_path"])
    assert payload["frames"][0]["frame_id"] == "000000"
    assert payload["frames"][0]["image_width"] == 40
    assert payload["frames"][0]["image_height"] == 32
    assert crop_path.exists()
    with Image.open(crop_path) as crop_image:
        assert crop_image.size == (40, 32)
        red_mean, green_mean, _ = crop_image.resize((1, 1)).getpixel((0, 0))
        assert green_mean > red_mean


def test_cli_view_crop_accepts_frame_object_pixel_bbox(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    scene_dir = _write_raw_rgb_scene(tmp_path)
    image = Image.new("RGB", (100, 80), color=(220, 10, 10))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 0, 99, 79), fill=(10, 220, 10))
    image.save(scene_dir / "raw" / "000000-rgb.png")
    object_frame_map_path = (
        scene_dir / "conceptgraph" / "indices" / "object_frame_map.json"
    )
    object_frame_map_path.write_text(
        json.dumps(
            {
                "frame_to_objects": {
                    "0": {
                        "view_id": 0,
                        "frame_name": "000000-rgb.jpg",
                        "objects": [
                            {
                                "object_id": "handle-1",
                                "class_name": "handle",
                                "score": 0.91,
                                "bbox_xyxy": [50, 0, 90, 32],
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

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
    object_payload = json.loads(capsys.readouterr().out.strip())["objects"][0]

    code = main(
        [
            "view_crop",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps(
                {
                    "frame_id": "000000",
                    "bbox": object_payload["bbox_xyxy"],
                    "bbox_format": object_payload["bbox_format"],
                }
            ),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    crop_path = Path(payload["frames"][0]["image_path"])
    with Image.open(crop_path) as crop_image:
        assert crop_image.size == (40, 32)
        red_mean, green_mean, _ = crop_image.resize((1, 1)).getpixel((0, 0))
        assert green_mean > red_mean


def test_cli_view_crop_uses_unique_paths_for_distinct_crops(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    scene_dir = _write_raw_rgb_scene(tmp_path)
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(
        scene_dir / "raw" / "000000-rgb.png"
    )

    first_code = main(
        [
            "view_crop",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_id": "000000", "bbox": [0.1, 0.1, 0.5, 0.5]}),
            "--out-dir",
            str(tmp_path / "scratch"),
        ]
    )
    first_payload = json.loads(capsys.readouterr().out.strip())
    second_code = main(
        [
            "view_crop",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_id": "000000", "bbox": [0.2, 0.2, 0.6, 0.6]}),
            "--out-dir",
            str(tmp_path / "scratch"),
        ]
    )
    second_payload = json.loads(capsys.readouterr().out.strip())

    assert first_code == 0
    assert second_code == 0
    assert (
        first_payload["frames"][0]["image_path"]
        != second_payload["frames"][0]["image_path"]
    )


def test_cli_view_bev_reports_unavailable_without_bev_asset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)

    code = main(["view_bev", "--scene-root", str(scene_dir), "--args", "{}"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "BEV asset is not available" in payload["error"]


def test_cli_keyframe_selector_prioritizes_query_frame_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)

    code = main(
        [
            "keyframe_selector",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"query": "drawer handle in frame 000010", "k": 1}),
            "--out-dir",
            str(tmp_path / "scratch"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["query"] == "drawer handle in frame 000010"
    assert payload["strategy"] == "frame_id_match"
    assert payload["frames"] == [
        {
            "frame_id": "000010",
            "rank": 1,
            "score": 100.0,
            "reason": "query_mentions_frame_id",
        }
    ]


def test_cli_keyframe_selector_uses_coverage_fallback_without_query_match(
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
    assert payload["strategy"] == "coverage_fallback_no_query_match"
    assert payload["frames"] == [
        {
            "frame_id": "000010",
            "rank": 1,
            "score": 0.0,
            "reason": "coverage_fallback_no_query_match",
        }
    ]


def test_cli_keyframe_selector_does_not_treat_counts_as_frame_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = tmp_path / "421254"
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir(parents=True)
    for frame_id in ("000000", "000003", "000010"):
        (raw_dir / f"{frame_id}-rgb.png").write_bytes(b"not-a-real-image")
    (scene_dir / "conceptgraph").mkdir()

    code = main(
        [
            "keyframe_selector",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"query": "open 3 drawers", "k": 1}),
            "--out-dir",
            str(tmp_path / "scratch"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["strategy"] == "coverage_fallback_no_query_match"
    assert payload["frames"][0]["score"] == 0.0


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


def test_cli_inspect_mask_artifact_validates_npz_ply(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    mask_npz_path, mask_ply_path = _write_points_artifact(tmp_path / "fragment-a")

    code = main(
        [
            "inspect_mask_artifact",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps(
                {
                    "mask_npz_path": str(mask_npz_path),
                    "mask_ply_path": str(mask_ply_path),
                }
            ),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "valid"
    assert payload["lifted_point_count"] == 2


def test_cli_fuse_accepted_masks_writes_final_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    first_npz_path, first_ply_path = _write_points_artifact(tmp_path / "frag-a")
    second_npz_path, second_ply_path = _write_points_artifact(tmp_path / "frag-b")
    first_review_artifacts = _write_review_artifacts(tmp_path / "review-a")
    second_review_artifacts = _write_review_artifacts(tmp_path / "review-b")

    code = main(
        [
            "fuse_accepted_masks",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps(
                {
                    "fragments": [
                        {
                            "fragment_id": "frag-a",
                            "frame_id": "000000",
                            "mask_npz_path": str(first_npz_path),
                            "mask_ply_path": str(first_ply_path),
                            "approval_actions": list(_APPROVED_FRAGMENT_ACTIONS),
                            "review_artifacts": first_review_artifacts,
                        },
                        {
                            "fragment_id": "frag-b",
                            "frame_id": "000010",
                            "mask_npz_path": str(second_npz_path),
                            "mask_ply_path": str(second_ply_path),
                            "approval_actions": list(_APPROVED_FRAGMENT_ACTIONS),
                            "review_artifacts": second_review_artifacts,
                        },
                    ],
                    "multi_view_decision": {
                        "seed_fragment_id": "frag-a",
                        "action": "expand",
                        "reason": "first lift is sparse and needs another view",
                        "suggested_frame_ids": ["000010"],
                    },
                }
            ),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert Path(payload["mask_artifact_path"]).exists()
    assert Path(payload["mask_npz_path"]).exists()
    assert Path(payload["mask_ply_path"]).exists()
    assert payload["accepted_frame_ids"] == ["000000", "000010"]
    assert payload["multi_view_decision"]["action"] == "expand"
    assert payload["multi_view_decision"]["suggested_frame_ids"] == ["000010"]
    assert [fragment["fragment_id"] for fragment in payload["accepted_fragments"]] == [
        "frag-a",
        "frag-b",
    ]
    assert [fragment["frame_id"] for fragment in payload["accepted_fragments"]] == [
        "000000",
        "000010",
    ]
    assert [
        fragment["approval_actions"] for fragment in payload["accepted_fragments"]
    ] == [
        list(_APPROVED_FRAGMENT_ACTIONS),
        list(_APPROVED_FRAGMENT_ACTIONS),
    ]
    assert [
        fragment["review_artifacts"] for fragment in payload["accepted_fragments"]
    ] == [
        first_review_artifacts,
        second_review_artifacts,
    ]


def test_cli_fuse_accepted_masks_requires_fragment_approval_actions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    mask_npz_path, mask_ply_path = _write_points_artifact(tmp_path / "frag-a")

    code = main(
        [
            "fuse_accepted_masks",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps(
                {
                    "fragments": [
                        {
                            "fragment_id": "frag-a",
                            "frame_id": "000000",
                            "mask_npz_path": str(mask_npz_path),
                            "mask_ply_path": str(mask_ply_path),
                        },
                    ],
                    "multi_view_decision": {
                        "seed_fragment_id": "frag-a",
                        "action": "stop",
                        "reason": "first lift covers the target part",
                        "suggested_frame_ids": [],
                    },
                }
            ),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "approval_actions" in payload["error"]


def test_cli_fuse_accepted_masks_requires_fragment_review_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    mask_npz_path, mask_ply_path = _write_points_artifact(tmp_path / "frag-a")

    code = main(
        [
            "fuse_accepted_masks",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps(
                {
                    "fragments": [
                        {
                            "fragment_id": "frag-a",
                            "frame_id": "000000",
                            "mask_npz_path": str(mask_npz_path),
                            "mask_ply_path": str(mask_ply_path),
                            "approval_actions": list(_APPROVED_FRAGMENT_ACTIONS),
                        },
                    ],
                    "multi_view_decision": {
                        "seed_fragment_id": "frag-a",
                        "action": "stop",
                        "reason": "first lift covers the target part",
                        "suggested_frame_ids": [],
                    },
                }
            ),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "review_artifacts" in payload["error"]


def test_cli_suggest_additional_views_returns_scene_neighbors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_raw_rgb_scene(tmp_path)
    seed_dir = tmp_path / "fragments" / "000000_mask_00"
    mask_npz_path, mask_ply_path = _write_points_artifact(seed_dir)
    lift_overlay_path = _write_lift_overlay(
        seed_dir / "lift_overlay.txt",
        frame_id="000000",
        candidate_id="mask_00",
    )

    code = main(
        [
            "suggest_additional_views",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps(
                {
                    "seed_fragment_id": "000000_mask_00",
                    "accepted_frame_id": "000000",
                    "seed_mask_npz_path": str(mask_npz_path),
                    "seed_mask_ply_path": str(mask_ply_path),
                    "seed_lift_overlay_path": str(lift_overlay_path),
                    "min_seed_point_count": 4,
                    "k": 1,
                }
            ),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["seed_fragment_id"] == "000000_mask_00"
    assert payload["expansion_recommendation"] == "expand"
    assert payload["views"] == [
        {
            "frame_id": "000010",
            "reason": (
                "seed_geometry_sparse: point_count=2 below min_seed_point_count=4; "
                "candidate is missing depth, intrinsics, and pose; "
                "view_diversity_score=0.000; "
                "temporal_distance=10"
            ),
            "rank": 1,
            "has_depth": False,
            "has_intrinsics": False,
            "has_pose": False,
            "view_diversity_score": 0.0,
        }
    ]


def _write_points_artifact(root: Path) -> tuple[Path, Path]:
    import numpy as np

    from codex_agent.scenefunc3d.backends.lift_3d import write_lift_npz, write_lift_ply

    points_world = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float64,
    )
    point_indices = np.array([10, 12], dtype=np.int64)
    mask_npz_path = write_lift_npz(
        root / "mask_data.npz", points_world, point_indices=point_indices
    )
    mask_ply_path = write_lift_ply(root / "lifted_points.ply", points_world)
    return mask_npz_path, mask_ply_path


def _write_lift_overlay(path: Path, *, frame_id: str, candidate_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"frame_id={frame_id}\n"
        f"candidate_id={candidate_id}\n"
        "lifted_point_count=2\n",
        encoding="utf-8",
    )
    return path


def _write_review_artifacts(root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "molmo_raw_text_path": root / "molmo_raw.txt",
        "molmo_overlay_path": root / "molmo_overlay.jpg",
        "sam_contact_sheet_path": root / "sam_contact_sheet.jpg",
        "sam_candidate_overlay_path": root / "sam_candidate_overlay.jpg",
        "lift_overlay_path": root / "lift_overlay.txt",
    }
    for path in paths.values():
        path.write_text("reviewed\n", encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}
