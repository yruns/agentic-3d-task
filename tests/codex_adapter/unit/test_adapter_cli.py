"""The operator CLI is explicit and never creates missing state on serve."""

import pytest

from codex_adapter.interfaces.cli import build_parser


def test_cli_requires_an_explicit_configuration_file() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve"])


@pytest.mark.parametrize("command", ["check", "init-state", "serve", "models"])
def test_cli_declares_operational_commands(command: str) -> None:
    parsed = build_parser().parse_args([command, "--config", "configs/adapter.toml"])
    assert parsed.command == command
    assert str(parsed.config) == "configs/adapter.toml"
