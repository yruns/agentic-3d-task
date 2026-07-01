"""Unit tests for the SceneFunc3D CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from codex_agent.cli import run_scenefunc3d
from codex_agent.scenefunc3d import runner


def test_shell_entrypoint_uses_cli_module_and_portable_python_default() -> None:
    script_text = Path("scripts/scenefunc3d/run_single_case_e2e.sh").read_text(
        encoding="utf-8"
    )

    assert "codex_agent.cli.run_scenefunc3d" in script_text
    assert "codex_agent.scenefunc3d.runner" not in script_text
    assert "${PYTHON:-python3}" in script_text


def test_sidecar_shell_scripts_use_portable_python_default() -> None:
    for script_path in (
        Path("scripts/scenefunc3d/check_sidecars.sh"),
        Path("scripts/scenefunc3d/serve_molmo_point.sh"),
        Path("scripts/scenefunc3d/serve_native_sidecar.sh"),
        Path("scripts/scenefunc3d/serve_sam2.sh"),
    ):
        script_text = script_path.read_text(encoding="utf-8")

        assert "${PYTHON:-python3}" in script_text
        assert "${PYTHON:-python}" not in script_text


def test_build_arg_parser_uses_scene_func_runner_parser() -> None:
    parser = run_scenefunc3d.build_arg_parser()

    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "codex_agent.cli.run_scenefunc3d"
    assert "codex_agent.cli.run_scenefunc3d" in parser.format_usage()


def test_main_parses_with_cli_prog_and_runs_parsed_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, argparse.Namespace] = {}

    def _fake_main(argv: list[str] | None = None) -> int:
        _ = argv
        raise AssertionError("CLI must not delegate to runner.main")

    def _fake_run_from_args(args: argparse.Namespace) -> int:
        captured["args"] = args
        return 17

    monkeypatch.setattr(runner, "main", _fake_main)
    monkeypatch.setattr(runner, "run_from_args", _fake_run_from_args, raising=False)

    exit_code = run_scenefunc3d.main(
        [
            "--dataset-root",
            str(tmp_path / "data"),
            "--sample-id",
            "421254::desc-a",
            "--backend-config",
            str(tmp_path / "backends.toml"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 17
    assert captured["args"].dataset_root == tmp_path / "data"
    assert captured["args"].sample_id == "421254::desc-a"
