"""Configuration loads references, never inlines or guesses credentials."""

import base64
from pathlib import Path

import pytest

from codex_adapter.config import load_settings
from codex_adapter.domain.errors import AdapterError

CONFIG = """
schema_version = 1
[state]
database = "state/adapter.sqlite"
active_key_id = "v1"
[[state.keys]]
id = "v1"
secret_env = "STATE_KEY"
[[models]]
id = "gpt-test"
protocol = "responses"
endpoint = "https://upstream.example/responses"
[[models.credentials]]
id = "primary"
version = "v1"
secret_env = "GPT_KEY"
weight = 3
"""


@pytest.fixture
def environment() -> dict[str, str]:
    return {
        "STATE_KEY": base64.urlsafe_b64encode(b"x" * 32).decode(),
        "GPT_KEY": "fake-config-secret",
    }


def config_file(tmp_path: Path, content: str = CONFIG) -> Path:
    path = tmp_path / "adapter.toml"
    path.write_text(content)
    return path


def test_loads_only_registered_models_and_env_references(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    settings = load_settings(config_file(tmp_path), environ=environment)
    assert (
        settings.registry.get("gpt-test").credentials[0].secret
        == environment["GPT_KEY"]
    )
    assert settings.state.database == tmp_path / "state/adapter.sqlite"
    assert environment["GPT_KEY"] not in repr(settings)


def test_missing_secret_is_a_startup_error(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    with pytest.raises(AdapterError, match="GPT_KEY"):
        load_settings(
            config_file(tmp_path), environ={"STATE_KEY": environment["STATE_KEY"]}
        )


def test_inline_credentials_are_rejected_without_echo(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    content = CONFIG.replace('secret_env = "GPT_KEY"', 'secret = "never-echo-this"')
    with pytest.raises(AdapterError) as raised:
        load_settings(config_file(tmp_path, content), environ=environment)
    assert "never-echo-this" not in str(raised.value)


@pytest.mark.parametrize(
    "replacement", ["weight = true", "weight = -1", 'weight = "3"']
)
def test_invalid_weights_fail_at_startup(
    tmp_path: Path, environment: dict[str, str], replacement: str
) -> None:
    with pytest.raises(AdapterError):
        load_settings(
            config_file(tmp_path, CONFIG.replace("weight = 3", replacement)),
            environ=environment,
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:password@upstream.example/responses",
        "https://upstream.example/responses?ak=secret",
        "http://upstream.example/responses",
        "file:///tmp/model",
    ],
)
def test_untrusted_or_secret_bearing_endpoint_is_rejected(
    tmp_path: Path, environment: dict[str, str], endpoint: str
) -> None:
    with pytest.raises(AdapterError):
        load_settings(
            config_file(
                tmp_path, CONFIG.replace("https://upstream.example/responses", endpoint)
            ),
            environ=environment,
        )


def test_public_listen_requires_adapter_authentication(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    with pytest.raises(AdapterError, match="authentication"):
        load_settings(
            config_file(tmp_path, CONFIG + '\n[server]\nhost="0.0.0.0"\n'),
            environ=environment,
        )


def test_gemini_defaults_do_not_claim_unverified_capabilities(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    content = CONFIG.replace('id = "gpt-test"', 'id = "gemini-test"').replace(
        'protocol = "responses"', 'protocol = "gemini_chat"'
    )
    model = load_settings(
        config_file(tmp_path, content), environ=environment
    ).registry.get("gemini-test")
    assert model.capabilities.default_reasoning_effort is None
    assert model.capabilities.reasoning_efforts == ()
    assert not model.capabilities.structured_output
    assert not model.capabilities.images


def test_gemini_cannot_advertise_unimplemented_summary_mapping(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    content = (
        CONFIG.replace('id = "gpt-test"', 'id = "gemini-test"').replace(
            'protocol = "responses"', 'protocol = "gemini_chat"'
        )
        + "\n[models.capabilities]\nreasoning_summary = true\n"
    )
    with pytest.raises(AdapterError, match="summary"):
        load_settings(config_file(tmp_path, content), environ=environment)
