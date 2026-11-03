"""Pinned real-SDK/core integration against a loopback-only scripted adapter.

No Codex class, thread, or turn is mocked. Only the adapter HTTP boundary is
fake; this does not test the separately owned HTTP application or upstreams.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import partial
from importlib.metadata import version
from pathlib import Path
from uuid import UUID

import pytest
from PIL import Image

from codex_agent.config import CodexAgentConfig
from codex_agent.models import CodexTurnRequest, CodexTurnResult
from codex_agent.runtime import CodexAgentRuntime
from codex_agent.tests.test_adapter_fixtures import (
    AdapterReply,
    AdapterRequest,
    adapter_profile,
    adapter_test_server,
    sdk_message,
    sdk_response_events,
)
from codex_agent.tests.test_adapter_provider import write_adapter_home

_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _function_names(body: Mapping[str, object]) -> set[str]:
    tools = body.get("tools")
    assert isinstance(tools, list)
    assert all(
        isinstance(tool, dict) and tool.get("type") == "function" for tool in tools
    )
    return {str(tool["name"]) for tool in tools}


def _shell_call(call: AdapterRequest, root: Path) -> dict[str, object]:
    names = _function_names(call.body)
    assert "view_image" in names
    if "exec_command" in names:
        name, arguments = "exec_command", {
            "cmd": "printf fixture-shell-evidence",
            "workdir": str(root),
        }
    elif "shell_command" in names:
        name, arguments = "shell_command", {
            "command": "printf fixture-shell-evidence",
            "workdir": str(root),
        }
    else:
        raise AssertionError(f"Missing usable SDK shell function: {names}")
    return {
        "type": "function_call",
        "id": "fc-shell",
        "call_id": "call-shell",
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def _scripted_response(
    call: AdapterRequest, *, posts: list[AdapterRequest], root: Path, image: Path
) -> AdapterReply:
    assert call.method == "POST" and call.path == "/v1/responses", call.path
    posts.append(call)
    index = len(posts)
    if index == 1:
        output = _shell_call(call, root)
    elif index == 2:
        output = {
            "type": "function_call",
            "id": "fc-image",
            "call_id": "call-image",
            "name": "view_image",
            "arguments": json.dumps({"path": str(image)}),
            "status": "completed",
        }
    elif index == 3:
        output = sdk_message("not JSON, please finalize")
    else:
        output = sdk_message('{"answer":"fixture-complete"}')
    # Deliberately exceed the declared window: a mistakenly enabled automatic
    # compaction would cause an observable /compact or summary request here.
    return sdk_response_events(
        [output], response_id=f"fake-{index}", input_tokens=2000000
    )


def _runtime(tmp_path: Path, base_url: str) -> CodexAgentRuntime:
    home = write_adapter_home(
        tmp_path,
        base_url,
        extra=(
            "requires_openai_auth = false\nsupports_websockets = false\n"
            "request_max_retries = 0\nstream_max_retries = 0\n"
        ),
    )
    return CodexAgentRuntime(
        CodexAgentConfig(
            project_root=tmp_path,
            codex_home=home,
            model="gemini-test",
            reasoning_effort="provider-default",
            adapter_token="fake-local-token",
            enable_prefix_cache=False,
            keep_run_home=True,
            max_tool_calls=8,
            turn_timeout_s=20,
        )
    )


def _assert_request_contract(posts: list[AdapterRequest]) -> None:
    assert len(posts) == 5
    ids = [call.header("X-Codex-Session-ID") for call in posts]
    assert len(set(ids[:4])) == 1
    assert ids[4] != ids[0]
    assert all(str(UUID(session)) == session for session in ids)
    assert all(
        call.header("Authorization") == "Bearer fake-local-token" for call in posts
    )
    for call in posts:
        assert not any(
            key.lower().startswith("x-codex-upstream") for key in call.headers
        )
        assert call.body["model"] == "gemini-test"
        assert call.body["text"]["format"]["schema"] == _SCHEMA  # type: ignore[index]
        assert call.body.get("reasoning") in (None, {}), call.body.get("reasoning")
        assert call.body.get("parallel_tool_calls") is True
        _function_names(call.body)
    assert "fixture-shell-evidence" in json.dumps(posts[1].body)
    assert '"type": "input_image"' in json.dumps(posts[2].body)
    assert '"call_id": "call-image"' in json.dumps(posts[2].body)


def _assert_real_catalog(result: CodexTurnResult) -> None:
    assert result.metadata.run_home is not None
    path = Path(result.metadata.run_home) / "adapter-model-catalog.json"
    model = json.loads(path.read_text(encoding="utf-8"))["models"][0]
    assert model["slug"] == "gemini-test"
    assert model["context_window"] is None
    assert model["max_context_window"] is None
    assert model["auto_compact_token_limit"] is None
    assert model["supports_reasoning_summary_parameter"] is False


def test_real_sdk_keeps_session_schema_tools_and_disables_gemini_compaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fail before launching a subprocess on an unimplemented catalog path (RED).
    from codex_agent.provider_catalog import build_model_catalog

    assert callable(build_model_catalog)
    assert version("openai-codex") == "0.163.0"
    monkeypatch.setenv("CODEX_ADAPTER_SESSION_ID", "codex_agent")
    monkeypatch.setenv("PROBE_UPSTREAM_AK", "fake-must-not-be-forwarded")
    image = tmp_path / "fixture.png"
    Image.new("RGB", (4, 4), color=(20, 60, 100)).save(image)
    posts: list[AdapterRequest] = []
    respond = partial(_scripted_response, posts=posts, root=tmp_path, image=image)
    with adapter_test_server(adapter_profile(), respond=respond) as server:
        runtime = _runtime(tmp_path, server.base_url)
        request = CodexTurnRequest(
            prompt="Read the fixture and return JSON.", output_schema=_SCHEMA
        )
        first = runtime.run_turn(
            request, response_validator=lambda text: '"answer"' in text
        )
        second = runtime.run_turn(request)
    assert (
        first.final_response == second.final_response == '{"answer":"fixture-complete"}'
    )
    assert len(first.metadata.attempts) == 2
    _assert_request_contract(posts)
    _assert_real_catalog(first)
    assert all("compact" not in call.path for call in server.requests)
