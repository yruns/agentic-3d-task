"""Direct AK values belong to individually validated weighted upstreams."""

import base64
from pathlib import Path

import pytest

from codex_adapter.config import load_settings
from codex_adapter.domain.errors import AdapterError

CONFIG = """
schema_version = 2
[state]
database = "state/adapter.sqlite"
active_key_id = "v1"
[[state.keys]]
id = "v1"
secret_env = "STATE_KEY"
[[models]]
id = "gpt-test"
protocol = "responses"
[[models.upstreams]]
id = "primary"
version = "v1"
url = "https://upstream.example/responses"
ak = "fake-direct-file-ak"
weight = 3
"""


@pytest.fixture
def environment() -> dict[str, str]:
    return {
        "STATE_KEY": base64.urlsafe_b64encode(b"x" * 32).decode(),
        "GPT_KEY": "must-not-use-this-environment-ak",
    }


def config_file(tmp_path: Path, content: str = CONFIG) -> Path:
    path = tmp_path / "adapter.toml"
    path.write_text(content)
    return path


def test_upstream_uses_file_ak_and_does_not_expose_it_in_repr(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    settings = load_settings(config_file(tmp_path), environ=environment)
    upstream = settings.registry.get("gpt-test").upstreams[0]
    assert upstream.secret == "fake-direct-file-ak"
    assert upstream.url == "https://upstream.example/responses"
    assert settings.state.database == tmp_path / "state/adapter.sqlite"
    assert upstream.secret not in repr(settings)


def test_ak_environment_changes_cannot_override_file_value(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    path = config_file(tmp_path)
    first = load_settings(path, environ=environment)
    second = load_settings(
        path,
        environ={
            **environment,
            "GPT_KEY": "different-env-ak",
            "AIDP_GPT_AK": "legacy-env-ak",
        },
    )
    assert (
        first.registry.get("gpt-test").upstreams
        == second.registry.get("gpt-test").upstreams
    )


def test_missing_ak_fails_even_when_legacy_environment_key_exists(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    content = CONFIG.replace('ak = "fake-direct-file-ak"\n', "")
    with pytest.raises(AdapterError):
        load_settings(config_file(tmp_path, content), environ=environment)


@pytest.mark.parametrize("field", ["ak_env", "secret_env"])
def test_ak_environment_references_are_not_supported(
    tmp_path: Path, environment: dict[str, str], field: str
) -> None:
    content = CONFIG.replace('ak = "fake-direct-file-ak"', f'{field} = "GPT_KEY"')
    with pytest.raises(AdapterError):
        load_settings(config_file(tmp_path, content), environ=environment)


def test_environment_ak_field_is_rejected_even_with_a_direct_ak(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    content = CONFIG.replace(
        'ak = "fake-direct-file-ak"', 'ak = "fake-direct-file-ak"\nak_env = "GPT_KEY"'
    )
    with pytest.raises(AdapterError):
        load_settings(config_file(tmp_path, content), environ=environment)


def test_old_schema_requires_explicit_migration(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    with pytest.raises(AdapterError, match="schema_version.*2"):
        load_settings(
            config_file(
                tmp_path, CONFIG.replace("schema_version = 2", "schema_version = 1")
            ),
            environ=environment,
        )


@pytest.mark.parametrize(
    "literal", ['""', '"   "', '"private-marker\\nsecond-line"', "123", "true"]
)
def test_invalid_direct_ak_does_not_echo_input(
    tmp_path: Path, environment: dict[str, str], literal: str
) -> None:
    content = CONFIG.replace('"fake-direct-file-ak"', literal)
    with pytest.raises(AdapterError) as raised:
        load_settings(config_file(tmp_path, content), environ=environment)
    assert "private-marker" not in str(raised.value)
    assert "fake-direct-file-ak" not in str(raised.value)


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
    "url",
    [
        "https://user:password@upstream.example/responses",
        "https://upstream.example/responses?ak=secret",
        "http://upstream.example/responses",
        "file:///tmp/model",
    ],
)
def test_each_upstream_rejects_untrusted_or_secret_bearing_url(
    tmp_path: Path, environment: dict[str, str], url: str
) -> None:
    with pytest.raises(AdapterError):
        load_settings(
            config_file(
                tmp_path, CONFIG.replace("https://upstream.example/responses", url)
            ),
            environ=environment,
        )


@pytest.mark.parametrize("suffix", ["?", "#"])
def test_old_empty_url_delimiters_remain_exact_for_existing_bindings(
    tmp_path: Path, environment: dict[str, str], suffix: str
) -> None:
    url = "https://upstream.example/responses" + suffix
    content = CONFIG.replace("https://upstream.example/responses", url)
    upstream = (
        load_settings(config_file(tmp_path, content), environ=environment)
        .registry.get("gpt-test")
        .upstreams[0]
    )
    assert upstream.url == url


def test_one_model_can_contain_different_upstream_urls(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    content = CONFIG + """
[[models.upstreams]]
id = "secondary"
version = "v1"
url = "https://other-upstream.example/responses"
ak = "fake-second-file-ak"
weight = 1
"""
    model = load_settings(
        config_file(tmp_path, content), environ=environment
    ).registry.get("gpt-test")
    assert tuple(upstream.url for upstream in model.upstreams) == (
        "https://upstream.example/responses",
        "https://other-upstream.example/responses",
    )
    assert tuple(upstream.secret for upstream in model.upstreams) == (
        "fake-direct-file-ak",
        "fake-second-file-ak",
    )


def test_public_listen_requires_adapter_authentication(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    with pytest.raises(AdapterError, match="authentication"):
        load_settings(
            config_file(tmp_path, CONFIG + '\n[server]\nhost="0.0.0.0"\n'),
            environ=environment,
        )


def test_shipped_gemini_profile_enables_images_and_structured_output() -> None:
    path = Path(__file__).resolve().parents[3] / "configs/codex_adapter.example.toml"
    settings = load_settings(path, environ={})
    capability = settings.registry.get("gemini-3.8-flash").capabilities
    assert capability.images is True
    assert capability.structured_output is True


def test_state_config_does_not_require_state_keys(tmp_path: Path) -> None:
    minimal = """
schema_version = 2
[state]
database = "state/adapter.sqlite"
[[models]]
id = "gpt-test"
protocol = "responses"
[[models.upstreams]]
id = "primary"
version = "v1"
url = "https://upstream.example/responses"
ak = "fake-direct-file-ak"
weight = 3
"""
    settings = load_settings(config_file(tmp_path, minimal), environ={})
    assert settings.state.database == tmp_path / "state/adapter.sqlite"


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
