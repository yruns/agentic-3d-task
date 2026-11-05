"""Actor runtime tests for tool-enabled single-turn ALFRED execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_agent.alfred.actor_runtime import AlfredPolicyRuntime, build_actor_catalog
from codex_agent.config import CodexAgentConfig
from codex_agent.errors import CodexConfigError
from codex_agent.provider_profile import AdapterModelProfile
from codex_agent.tests.test_adapter_fixtures import adapter_profile, adapter_test_server
from codex_agent.tests.test_adapter_provider import write_adapter_home


def test_actor_catalog_enables_shell_and_view_image_while_disabling_ambient_tools() -> (
    None
):
    profile = AdapterModelProfile.model_validate(
        adapter_profile(model="gpt-exact-version", protocol="responses")
    )
    catalog = build_actor_catalog(profile)
    model = catalog["models"][0]
    assert model["slug"] == "gpt-exact-version"
    assert model["apply_patch_tool_type"] is None
    assert model["multi_agent_version"] == "disabled"
    assert model["node_repl_disabled"] is True
    assert model["include_skills_usage_instructions"] is False
    assert model["include_plugin_usage_instructions"] is False
    assert model["include_apps_usage_instructions"] is False
    assert model["input_modalities"] == ["text", "image"]


def test_actor_run_catalog_and_overrides_are_isolated_per_turn(tmp_path: Path) -> None:
    with adapter_test_server(
        adapter_profile(model="gpt-test", protocol="responses")
    ) as server:
        home = write_adapter_home(tmp_path, server.base_url)
        runtime = AlfredPolicyRuntime(
            CodexAgentConfig(codex_home=home, model="gpt-test")
        )
        first = runtime._prepare_run_home()
        catalog = json.loads((first / "alfred_actor_catalog.json").read_text())
        assert catalog["models"][0]["slug"] == "gpt-test"
        overrides = runtime._build_config_overrides()
        assert (
            overrides[-1]
            == f"model_catalog_json={json.dumps(str(first / 'alfred_actor_catalog.json'))}"
        )
        assert "features.multi_agent_v2=false" in overrides
        assert "features.plugins=false" in overrides
        assert "features.code_mode=false" in overrides
        assert "tools.experimental_request_user_input.enabled=false" in overrides
        env = runtime._build_turn_env(first)
        assert "PYTHONPATH" in env
        assert "127.0.0.1" in env["no_proxy"]
        runtime._cleanup_run_home(first)
        second = runtime._prepare_run_home()
        assert second != first
        runtime._cleanup_run_home(second)
    assert not first.exists() and not second.exists()


@pytest.mark.parametrize(
    "unsafe_section",
    [
        '[mcp_servers.hidden]\ncommand="cat"\n',
        "[plugins.hidden]\nenabled=true\n",
        '[profiles.custom.mcp_servers.hidden]\ncommand="cat"\n',
    ],
)
def test_actor_rejects_inherited_mcp_and_plugins_before_starting_tools(
    tmp_path: Path, unsafe_section: str
) -> None:
    home = write_adapter_home(tmp_path, "http://127.0.0.1:1/v1")
    with (home / "config.toml").open("a") as stream:
        stream.write(unsafe_section)
    runtime = AlfredPolicyRuntime(CodexAgentConfig(codex_home=home, model="gpt-test"))
    with pytest.raises(CodexConfigError, match="MCP.*plugins"):
        runtime._prepare_run_home()
    assert not (home / "runs").exists()


def test_catalog_write_failure_removes_partial_home_even_when_traces_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with adapter_test_server(
        adapter_profile(model="gpt-test", protocol="responses")
    ) as server:
        home = write_adapter_home(tmp_path, server.base_url)
        original = Path.write_text

        def reject_catalog(path: Path, *args: object, **kwargs: object) -> int:
            if path.name == "alfred_actor_catalog.json":
                raise PermissionError("synthetic artifact write failure")
            return original(path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "write_text", reject_catalog)
        runtime = AlfredPolicyRuntime(
            CodexAgentConfig(codex_home=home, model="gpt-test", keep_run_home=True)
        )
        with pytest.raises(PermissionError, match="synthetic"):
            runtime._prepare_run_home()
    assert list((home / "runs").iterdir()) == []
    with pytest.raises(CodexConfigError, match="prepared"):
        runtime._build_config_overrides()


def test_actor_rejects_provider_outside_registered_adapter() -> None:
    runtime = AlfredPolicyRuntime(CodexAgentConfig(model_provider="openai"))
    with pytest.raises(CodexConfigError, match="codex_adapter"):
        runtime._prepare_run_home()


def test_actor_requires_image_capability() -> None:
    profile = AdapterModelProfile.model_validate(
        adapter_profile(model="gpt-test", protocol="responses", images=False)
    )
    with pytest.raises(CodexConfigError, match="images"):
        build_actor_catalog(profile)
