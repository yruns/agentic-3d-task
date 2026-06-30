"""Tests for the project-local Codex home bootstrap script."""

from __future__ import annotations

import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 test environment.
    import tomli as tomllib


def test_bootstrap_codex_home_writes_safe_config(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "codex_agent" / "bootstrap_codex_home.sh"
    codex_home = tmp_path / ".codex-home"

    command: list[str] = [
        "bash",
        str(script_path),
        "--project-root",
        str(repo_root),
        "--codex-home",
        str(codex_home),
    ]
    subprocess.run(command, check=True, cwd=repo_root)

    config_path = codex_home / "config.toml"
    config_text = config_path.read_text(encoding="utf-8")
    lowered_config = config_text.lower()

    assert 'model = "gpt-5.4-2026-03-05"' in config_text
    assert 'model_provider = "modelhub_adapter"' in config_text
    assert 'base_url = "http://127.0.0.1:8787/v1"' in config_text
    assert f'[projects."{repo_root}"]' in config_text
    assert "replace" not in lowered_config
    assert "ak" not in lowered_config
    assert "token" not in lowered_config
    assert "secret" not in lowered_config


def test_bootstrap_codex_home_escapes_toml_strings(tmp_path: Path) -> None:
    repo_root = tmp_path / 'repo"root'
    codex_home = tmp_path / ".codex-home"
    adapter_url = 'http://127.0.0.1:8787/v1?name="adapter"'
    model_name = 'gpt-"quoted"'
    script_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "codex_agent"
        / "bootstrap_codex_home.sh"
    )

    command: list[str] = [
        "bash",
        str(script_path),
        "--project-root",
        str(repo_root),
        "--codex-home",
        str(codex_home),
        "--adapter-base-url",
        adapter_url,
        "--model",
        model_name,
    ]
    subprocess.run(command, check=True)

    config_path = codex_home / "config.toml"
    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert parsed["model"] == model_name
    assert parsed["model_providers"]["modelhub_adapter"]["base_url"] == adapter_url
    assert parsed["projects"][str(repo_root)]["trust_level"] == "trusted"
