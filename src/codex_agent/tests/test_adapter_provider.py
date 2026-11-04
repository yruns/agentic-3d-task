"""Codex-adapter discovery, capability validation, and session regressions."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

import pytest

from codex_agent.cli.runtime_preflight import preflight_codex_runtime
from codex_agent.config import CodexAgentConfig
from codex_agent.errors import CodexConfigError
from codex_agent.runtime import CodexAgentRuntime
from codex_agent.tests.test_adapter_fixtures import adapter_profile, adapter_test_server


def write_adapter_home(tmp_path: Path, base_url: str, *, extra: str = "") -> Path:
    home = tmp_path / "adapter-home"
    home.mkdir()
    (home / "config.toml").write_text(
        'model_provider = "codex_adapter"\n'
        '[model_providers.codex_adapter]\nname = "Test codex adapter"\n'
        f'base_url = {json.dumps(base_url)}\nwire_api = "responses"\n{extra}',
        encoding="utf-8",
    )
    return home


def test_preflight_validates_new_identity_and_exact_model(tmp_path: Path) -> None:
    model = "gpt-test/space + name"
    profile = adapter_profile(model=model, protocol="responses")
    with adapter_test_server(profile) as server:
        home = write_adapter_home(tmp_path, server.base_url)
        result = preflight_codex_runtime(CodexAgentConfig(codex_home=home, model=model))
    assert result.status == "checked"
    assert result.model_profile is not None
    assert result.model_profile.model == model
    assert [call.path for call in server.requests] == [
        "/health",
        f"/v1/models/{quote(model, safe='')}/capabilities",
    ]


@pytest.mark.parametrize(
    "health",
    [
        {},
        {"status": "ok"},
        {
            "status": "ready",
            "service": "old-adapter",
            "version": "1.0.0",
            "schema_version": 1,
            "models": ["gpt-test"],
        },
        {
            "status": "ready",
            "service": "codex-adapter",
            "version": "1.0.0",
            "schema_version": 2,
            "models": ["gpt-test"],
        },
        {
            "status": "ready",
            "service": "codex-adapter",
            "version": "1.0.0",
            "schema_version": 1,
            "models": ["another-model"],
        },
    ],
)
def test_preflight_rejects_wrong_service_or_unregistered_model(
    tmp_path: Path, health: object
) -> None:
    with adapter_test_server(
        adapter_profile(model="gpt-test", protocol="responses"), health=health
    ) as server:
        home = write_adapter_home(tmp_path, server.base_url)
        with pytest.raises(CodexConfigError, match="(identity|registered|health)"):
            preflight_codex_runtime(CodexAgentConfig(codex_home=home, model="gpt-test"))


@pytest.mark.parametrize(
    "capabilities",
    [
        {"structured_output": False},
        {"structured_output": "yes"},
        {"context_window": -1},
        {"reasoning_efforts": ["medium"], "default_reasoning_effort": "high"},
    ],
)
def test_preflight_rejects_invalid_or_insufficient_capabilities(
    tmp_path: Path, capabilities: dict[str, object]
) -> None:
    profile = adapter_profile(model="gpt-test", protocol="responses", **capabilities)
    with adapter_test_server(profile) as server:
        home = write_adapter_home(tmp_path, server.base_url)
        with pytest.raises(CodexConfigError, match="(capabilit|structured_output)"):
            preflight_codex_runtime(CodexAgentConfig(codex_home=home, model="gpt-test"))


def test_preflight_does_not_guess_missing_capability_fields(tmp_path: Path) -> None:
    profile = {
        **adapter_profile(model="gpt-test", protocol="responses"),
        "capabilities": {},
    }
    with adapter_test_server(profile) as server:
        home = write_adapter_home(tmp_path, server.base_url)
        with pytest.raises(CodexConfigError, match="capabilit"):
            preflight_codex_runtime(CodexAgentConfig(codex_home=home, model="gpt-test"))


def test_preflight_uses_only_optional_local_bearer_token(tmp_path: Path) -> None:
    with adapter_test_server(
        adapter_profile(model="gpt-test", protocol="responses")
    ) as server:
        home = write_adapter_home(tmp_path, server.base_url)
        config = CodexAgentConfig(
            codex_home=home, model="gpt-test", adapter_token="fake-local-token"
        )
        preflight_codex_runtime(config)
    assert len(server.requests) == 2
    assert all(
        call.headers.get("Authorization") == "Bearer fake-local-token"
        for call in server.requests
    )
    assert all(
        not any(key.lower().startswith("x-codex-upstream") for key in call.headers)
        for call in server.requests
    )


@pytest.mark.parametrize(
    "extra",
    [
        'env_http_headers = { "X-Codex-Upstream-AK" = "PRIVATE_KEY" }\n',
        'http_headers = { "X-Codex-Upstream-URL" = "https://invalid.test" }\n',
        'env_key = "PRIVATE_UPSTREAM_KEY"\n',
        'experimental_bearer_token = "fake-must-not-forward"\n',
        'http_headers = { "Authorization" = "Bearer fake-upstream" }\n',
    ],
)
def test_new_provider_rejects_client_upstream_overrides(
    tmp_path: Path, extra: str
) -> None:
    with adapter_test_server(
        adapter_profile(model="gpt-test", protocol="responses")
    ) as server:
        home = write_adapter_home(tmp_path, server.base_url, extra=extra)
        with pytest.raises(CodexConfigError, match="upstream"):
            preflight_codex_runtime(CodexAgentConfig(codex_home=home, model="gpt-test"))
    assert not server.requests


@pytest.mark.parametrize("effort", ["medium", "high"])
def test_gemini_rejects_unsupported_effort_including_default(
    tmp_path: Path, effort: str
) -> None:
    with adapter_test_server(adapter_profile()) as server:
        home = write_adapter_home(tmp_path, server.base_url)
        config = CodexAgentConfig(codex_home=home, model="gemini-test", reasoning_effort=effort)  # type: ignore[arg-type]
        with pytest.raises(CodexConfigError, match="provider-default"):
            preflight_codex_runtime(config)


def test_preflight_rejects_pinned_sdk_parallel_tools_mismatch(tmp_path: Path) -> None:
    with adapter_test_server(adapter_profile(parallel_tools=False)) as server:
        home = write_adapter_home(tmp_path, server.base_url)
        config = CodexAgentConfig(
            codex_home=home, model="gemini-test", reasoning_effort="provider-default"
        )
        with pytest.raises(CodexConfigError, match="parallel_tools.*0.163.0"):
            preflight_codex_runtime(config)


def test_gemini_rejects_positive_context_override(tmp_path: Path) -> None:
    with adapter_test_server(adapter_profile()) as server:
        home = write_adapter_home(tmp_path, server.base_url)
        config = CodexAgentConfig(
            codex_home=home,
            model="gemini-test",
            reasoning_effort="provider-default",
            model_context_window=1000000,
        )
        with pytest.raises(CodexConfigError, match="compaction"):
            preflight_codex_runtime(config)


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_new_session_route_is_unique_untruncated_even_without_cache(
    tmp_path: Path, cache_enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_ADAPTER_SESSION_ID", "codex_agent")
    runtime = CodexAgentRuntime(
        CodexAgentConfig(
            enable_prefix_cache=cache_enabled,
            prefix_cache_session_id="x" * 500,
            adapter_token="fake-local-token",
        )
    )
    first = runtime._build_turn_env(tmp_path / "run1")
    second = runtime._build_turn_env(tmp_path / "run2")
    session = first["CODEX_ADAPTER_SESSION_ID"]
    assert str(UUID(session)) == session
    assert first["CODEX_ADAPTER_SESSION_ID"] != second["CODEX_ADAPTER_SESSION_ID"]
    assert first["CODEX_ADAPTER_AUTHORIZATION"] == "Bearer fake-local-token"
    assert not any("UPSTREAM" in key for key in first)
    assert "CODEX_AGENT_MODELHUB_EXTRA_HEADER" not in first
    assert any(
        'model_providers.codex_adapter.env_http_headers.X-Codex-Session-ID="CODEX_ADAPTER_SESSION_ID"'
        == override
        for override in runtime._build_config_overrides()
    )


def test_preflight_new_provider_rejects_non_http_discovery(tmp_path: Path) -> None:
    home = write_adapter_home(tmp_path, "stdio://adapter")
    with pytest.raises(CodexConfigError, match="HTTP"):
        preflight_codex_runtime(CodexAgentConfig(codex_home=home))


def test_adapter_environment_clears_ambient_authorization_when_optional_token_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_ADAPTER_AUTHORIZATION", "Bearer unrelated-ambient-token")
    env = CodexAgentRuntime(CodexAgentConfig())._build_turn_env(tmp_path)
    assert env["CODEX_ADAPTER_AUTHORIZATION"] == ""
