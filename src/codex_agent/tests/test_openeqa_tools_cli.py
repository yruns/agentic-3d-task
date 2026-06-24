"""Tests for the OpenEQA tools CLI entry point (``python -m codex_agent.openeqa.tools``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from codex_agent.openeqa.tools.__main__ import main  # noqa: E402
from codex_agent.tests.conftest import OpenEqaToolsFixture  # noqa: E402


def test_cli_view_frame_prints_json(
    openeqa_tools_fixture: OpenEqaToolsFixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "view_frame",
            "--scene-dir",
            str(openeqa_tools_fixture.scene_dir),
            "--args",
            json.dumps({"frame_ids": [0, 4]}),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert [f["frame_id"] for f in payload["frames"]] == [0, 4]


def test_cli_recoverable_error_is_json(
    openeqa_tools_fixture: OpenEqaToolsFixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "view_frame",
            "--scene-dir",
            str(openeqa_tools_fixture.scene_dir),
            "--args",
            json.dumps({"frame_id": 9999}),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "error" in payload


def test_cli_bad_args_json_is_recoverable(
    openeqa_tools_fixture: OpenEqaToolsFixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "view_frame",
            "--scene-dir",
            str(openeqa_tools_fixture.scene_dir),
            "--args",
            "not-json",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "must be valid JSON" in payload["error"]


def test_cli_missing_scene_dir_exits_nonzero(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "view_frame",
                "--scene-dir",
                str(tmp_path / "does_not_exist"),
                "--args",
                "{}",
                "--out-dir",
                str(tmp_path / "scratch"),
            ]
        )
    assert excinfo.value.code == 1
