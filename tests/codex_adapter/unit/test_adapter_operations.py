"""Operator commands validate configuration and initialize state explicitly."""

from __future__ import annotations

import base64
import json
import runpy
import sys
from pathlib import Path

import pytest

from codex_adapter.interfaces.cli import main


@pytest.fixture
def operator_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(
        "TEST_OPERATOR_STATE_KEY", base64.urlsafe_b64encode(b"k" * 32).decode()
    )
    monkeypatch.setenv("TEST_OPERATOR_MODEL_KEY", "fake-operator-key")
    path = tmp_path / "adapter.toml"
    path.write_text("""schema_version = 1
[state]
database = "state.sqlite"
active_key_id = "v1"
[[state.keys]]
id = "v1"
secret_env = "TEST_OPERATOR_STATE_KEY"
[[models]]
id = "gpt-test"
protocol = "responses"
endpoint = "https://upstream.example/responses"
[[models.credentials]]
id = "one"
version = "v1"
secret_env = "TEST_OPERATOR_MODEL_KEY"
weight = 1
""")
    return path


@pytest.mark.parametrize(
    "command,status",
    [("check", "configuration_valid"), ("init-state", "state_initialized")],
)
def test_operator_commands_do_not_probe_upstream(
    operator_config: Path, command: str, status: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command, "--config", str(operator_config)]) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == status


def test_models_output_contains_no_secrets(
    operator_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["models", "--config", str(operator_config)]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["models"][0]["model"] == "gpt-test"
    assert "fake-operator-key" not in output
    assert "upstream.example" not in output


def test_invalid_configuration_has_safe_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["check", "--config", str(tmp_path / "missing.toml")]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_config"


def test_serve_uses_one_worker_without_access_logs(
    operator_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    observed: list[dict[str, object]] = []

    def run(app: object, **options: object) -> None:
        observed.append(options)

    monkeypatch.setattr(uvicorn, "run", run)
    assert main(["serve", "--config", str(operator_config)]) == 0
    assert observed[0]["workers"] == 1
    assert observed[0]["access_log"] is False


def test_module_entrypoint(
    operator_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["codex_adapter", "check", "--config", str(operator_config)]
    )
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("codex_adapter", run_name="__main__")
    assert raised.value.code == 0
