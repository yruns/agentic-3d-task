"""Tests for LLM configuration loading and the weighted client pool."""

from __future__ import annotations

from pathlib import Path

import pytest

from keyframe.llm import LLMClient, LLMConfig
from keyframe.llm.client import is_rate_limit_error

_TOML = """
default_model = "primary"

[models."primary"]
api_version = "2024-03-01-preview"
temperature = 0.0
keys = [
    { api_key = "k1", weight = 1, endpoint = "https://a/deploy" },
    { api_key = "k2", weight = 3, endpoint = "https://b/crawl" },
]

[models."secondary"]
endpoint = "https://c/crawl"
api_version = "2024-02-15-preview"
keys = [{ api_key = "k3" }]
"""


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "llm.toml"
    path.write_text(_TOML)
    return path


def test_config_loads_and_weights(tmp_path: Path) -> None:
    config = LLMConfig.from_toml(_write_config(tmp_path))
    assert config.default_model == "primary"
    primary = config.get_model()
    assert len(primary.keys) == 2
    assert len(primary.weighted_keys()) == 4  # weights 1 + 3


def test_config_rejects_unknown_default(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        'default_model = "missing"\n[models."primary"]\napi_version="v"\nendpoint="https://e"\nkeys=[{api_key="k"}]\n'
    )
    with pytest.raises(ValueError):
        LLMConfig.from_toml(path)


def test_client_builds_models(tmp_path: Path) -> None:
    client = LLMClient.from_toml(str(_write_config(tmp_path)))
    assert client.key_count() == 2
    chat = client.get_chat_model()
    assert chat.deployment_name == "primary"
    assert client.get_chat_model("secondary").deployment_name == "secondary"


def test_rate_limit_detection() -> None:
    assert is_rate_limit_error(RuntimeError("429 Too Many Requests"))
    assert not is_rate_limit_error(RuntimeError("connection refused"))
