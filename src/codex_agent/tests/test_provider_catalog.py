"""Capability-driven custom catalog and pre-execution rejection regressions."""

from __future__ import annotations

from pathlib import Path

import pytest

import codex_agent.runtime as runtime_module
from codex_agent.config import CodexAgentConfig
from codex_agent.errors import CodexConfigError
from codex_agent.models import CodexTurnRequest
from codex_agent.runtime import CodexAgentRuntime
from codex_agent.tests.test_adapter_fixtures import adapter_profile, adapter_test_server
from codex_agent.tests.test_adapter_provider import write_adapter_home


@pytest.fixture(autouse=True)
def reject_unexpected_sdk_start(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_sdk(**_: object) -> None:
        raise AssertionError("Invalid capabilities must be rejected before SDK startup")

    monkeypatch.setattr(runtime_module, "Codex", unexpected_sdk)


def test_gemini_catalog_keeps_function_tools_and_disables_implicit_reasoning() -> None:
    from codex_agent.provider_catalog import build_model_catalog
    from codex_agent.provider_profile import AdapterModelProfile

    profile = AdapterModelProfile.model_validate(adapter_profile())
    catalog = build_model_catalog(profile)
    model = catalog["models"][0]
    assert model["slug"] == "gemini-test"
    assert model["shell_type"] == "unified_exec"
    assert model["input_modalities"] == ["text", "image"]
    assert model["supported_reasoning_levels"] == []
    assert model["default_reasoning_level"] is None
    assert model["supports_reasoning_summary_parameter"] is False
    assert model["default_reasoning_summary"] == "none"
    assert model["context_window"] is None
    assert model["max_context_window"] is None
    assert model["auto_compact_token_limit"] is None
    assert model["tool_mode"] == "direct"


@pytest.mark.parametrize(
    "setting",
    [
        "model_context_window = 1000000\n",
        "model_auto_compact_token_limit = 999999999\n",
        'model_reasoning_effort = "medium"\n',
    ],
)
def test_gemini_rejects_inherited_compaction_or_reasoning_config(
    tmp_path: Path, setting: str
) -> None:
    with adapter_test_server(adapter_profile()) as server:
        home = write_adapter_home(tmp_path, server.base_url)
        path = home / "config.toml"
        path.write_text(setting + path.read_text(), encoding="utf-8")
        runtime = CodexAgentRuntime(
            CodexAgentConfig(
                project_root=tmp_path,
                codex_home=home,
                model="gemini-test",
                reasoning_effort="provider-default",
            )
        )
        with pytest.raises(CodexConfigError, match="(compaction|reasoning_effort)"):
            runtime.run_turn(
                CodexTurnRequest(prompt="test", output_schema={"type": "object"})
            )
    assert all(call.method == "GET" for call in server.requests)


@pytest.mark.parametrize("flag", ["structured_output", "images"])
def test_runtime_rejects_undeclared_task_capability_before_sdk(
    tmp_path: Path, flag: str
) -> None:
    profile = (
        adapter_profile(structured_output=False)
        if flag == "structured_output"
        else adapter_profile(images=False)
    )
    with adapter_test_server(profile) as server:
        home = write_adapter_home(tmp_path, server.base_url)
        image = tmp_path / "image.png"
        image.write_bytes(b"fake-existing-image")
        runtime = CodexAgentRuntime(
            CodexAgentConfig(
                project_root=tmp_path,
                codex_home=home,
                model="gemini-test",
                reasoning_effort="provider-default",
            )
        )
        with pytest.raises(CodexConfigError, match=flag):
            runtime.run_turn(
                CodexTurnRequest(
                    prompt="test",
                    output_schema={"type": "object"},
                    image_paths=(image,),
                )
            )
    assert all(call.method == "GET" for call in server.requests)
