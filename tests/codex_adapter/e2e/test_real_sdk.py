"""Real Codex uses HTTP/SSE against the gateway; upstream stays a local fake."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from openai_codex import CodexConfig, Thread
from openai_codex.client import CodexClient
from openai_codex.models import JsonObject
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from codex_adapter.config import AdapterSettings, ServerSettings
from codex_adapter.domain.models import (
    ModelCapabilities,
    RegisteredModel,
    Registry,
    Upstream,
)
from codex_adapter.domain.state import StateKey, StateSettings
from codex_adapter.interfaces.app import create_app

_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
_TOOL: JsonObject = {
    "name": "lookup_record",
    "description": "Look up one test record",
    "inputSchema": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    },
}


@contextmanager
def running_adapter(
    settings: AdapterSettings,
    transport: httpx.MockTransport,
    wire: list[JsonObject] | None = None,
) -> Iterator[str]:
    app = create_app(settings, upstream_transport=transport, initialize_state=True)
    if wire is not None:
        _capture_test_wire(app, wire)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="critical", access_log=False, lifespan="on")
        )
        worker = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        worker.start()
        try:
            _wait_started(server, worker)
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            server.should_exit = True
            worker.join(timeout=10)
            if worker.is_alive():
                raise RuntimeError("Local test adapter did not stop")


def _capture_test_wire(app: FastAPI, wire: list[JsonObject]) -> None:
    @app.middleware("http")
    async def capture(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == "POST":
            wire.append(json.loads(await request.body()))
        return await call_next(request)


def _wait_started(server: uvicorn.Server, worker: threading.Thread) -> None:
    deadline = time.monotonic() + 10
    while not server.started:
        if not worker.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("Local test adapter failed to start")
        time.sleep(0.01)


def model_catalog(model_id: str) -> dict[str, object]:
    model: dict[str, object] = {
        "slug": model_id,
        "display_name": "Local protocol contract",
        "description": None,
        "supported_reasoning_levels": [],
        "shell_type": "disabled",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 1,
        "upgrade": None,
        "model_messages": {
            "instructions_template": "Use the provided test tool when requested."
        },
        "include_skills_usage_instructions": False,
        "include_plugin_usage_instructions": False,
        "include_apps_usage_instructions": False,
        "supports_reasoning_summary_parameter": False,
        "default_reasoning_summary": "none",
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_image_detail_original": False,
        "context_window": None,
        "max_context_window": None,
        "auto_compact_token_limit": None,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "node_repl_disabled": True,
        "tool_mode": "direct",
        "multi_agent_version": "disabled",
    }
    return {"models": [model]}


def sdk_config(tmp_path: Path, base_url: str, model_id: str) -> CodexConfig:
    home, codex_home = tmp_path / "home", tmp_path / "codex-home"
    home.mkdir()
    codex_home.mkdir()
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(model_catalog(model_id)))
    features = (
        "shell_tool",
        "view_image",
        "unified_exec",
        "plugins",
        "hooks",
        "multi_agent",
        "context_management",
        "async_compaction",
        "responses_websockets",
        "responses_websockets_v2",
    )
    text = f'model={json.dumps(model_id)}\nmodel_provider="codex_adapter"\nmodel_catalog_json={json.dumps(str(catalog))}\nproject_doc_max_bytes=0\nweb_search="disabled"\nmodel_post_turn_compact_threshold_percent=0\n[features]\n'
    text += "".join(f"{feature}=false\n" for feature in features)
    text += f'[model_providers.codex_adapter]\nname="Local test"\nbase_url={json.dumps(base_url)}\nwire_api="responses"\nrequires_openai_auth=false\nsupports_websockets=false\nrequest_max_retries=0\nstream_max_retries=0\n[model_providers.codex_adapter.env_http_headers]\nX-Codex-Session-ID="TEST_ADAPTER_SESSION"\n'
    (codex_home / "config.toml").write_text(text)
    return CodexConfig(
        cwd=str(tmp_path),
        env={
            "HOME": str(home),
            "CODEX_HOME": str(codex_home),
            "TEST_ADAPTER_SESSION": "real-sdk-unique-session-123",
            "RUST_LOG": "warn",
        },
    )


def adapter_settings(tmp_path: Path) -> AdapterSettings:
    model = RegisteredModel(
        "gemini-sdk-test",
        "gemini_chat",
        (Upstream("one", "v1", "https://fake-upstream.example/chat", "fake-sdk-key"),),
        ModelCapabilities(
            reasoning_efforts=(), default_reasoning_effort=None, reasoning_summary=False
        ),
    )
    return AdapterSettings(
        Registry((model,)),
        StateSettings(tmp_path / "state.sqlite", (StateKey("v1", b"k" * 32),), "v1"),
        ServerSettings(upstream_timeout_seconds=10.0),
    )


def _native_message(index: int, signature: str) -> JsonObject:
    if index not in {1, 2, 4}:
        return {
            "role": "assistant",
            "content": json.dumps({"answer": "348"}),
            "signature": signature,
        }
    return {
        "role": "assistant",
        "content": None,
        "signature": signature,
        "tool_calls": [
            {
                "id": "",
                "type": "function",
                "function": {
                    "name": "lookup_record",
                    "arguments": json.dumps(
                        {"name": "beta" if index == 2 else "alpha"}
                    ),
                },
                "signature": f"tool-{signature}",
            }
        ],
    }


class FakeGemini:
    """Assert complete native replay, rather than mocking gateway behavior."""

    def __init__(self) -> None:
        self.requests: tuple[httpx.Request, ...] = ()
        self.signatures: tuple[str, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests = (*self.requests, request)
        index = len(self.requests)
        assert body["response_format"]["type"] == "json_schema"
        restored = tuple(
            message["signature"]
            for message in body["messages"]
            if message["role"] == "assistant"
        )
        assert restored == self.signatures
        signature = f"opaque-native-signature-{index}"
        self.signatures = (*self.signatures, signature)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": _native_message(index, signature),
                        "finish_reason": "tool_calls" if index in {1, 2, 4} else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100 + index,
                    "completion_tokens": 10,
                    "reasoning_tokens": 2,
                    "total_tokens": 112 + index,
                    "prompt_tokens_details": {"cached_tokens": 64},
                },
            },
        )


class TestTools:
    """A real SDK dynamic tool handler, with observable executions."""

    __test__ = False

    def __init__(self) -> None:
        self.executions: tuple[str, ...] = ()

    def __call__(self, method: str, params: JsonObject | None) -> JsonObject:
        assert method == "item/tool/call" and params is not None
        arguments = params["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        assert isinstance(arguments, dict)
        name = arguments["name"]
        assert isinstance(name, str)
        self.executions = (*self.executions, name)
        return {
            "success": True,
            "contentItems": [
                {
                    "type": "inputText",
                    "text": json.dumps({"value": 137 if name == "alpha" else 211}),
                }
            ],
        }


def _run_sdk(config: CodexConfig, cwd: Path, tools: TestTools) -> None:
    with CodexClient(config, approval_handler=tools) as client:
        client.initialize()
        started = client.thread_start(
            {
                "model": "gemini-sdk-test",
                "modelProvider": "codex_adapter",
                "cwd": str(cwd),
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "baseInstructions": "Use the test tool.",
                "environments": [],
                "dynamicTools": [_TOOL],
                "allowProviderModelFallback": False,
            }
        )
        thread = Thread(client, started.thread.id)
        for prompt in (
            "Look up alpha then beta and return the sum.",
            "Verify alpha and repeat the answer.",
        ):
            result = thread.run(prompt, output_schema=_SCHEMA)
            assert result.final_response is not None
            assert json.loads(result.final_response)["answer"] == "348"


def input_shapes(wire: list[JsonObject]) -> tuple[tuple[str, ...], ...]:
    shapes: tuple[tuple[str, ...], ...] = ()
    for body in wire:
        items = body.get("input", [])
        assert isinstance(items, list)
        shapes = (
            *shapes,
            *(tuple(sorted(item)) for item in items if isinstance(item, dict)),
        )
    return shapes


class FakeTaskGemini:
    """Exercise runtime-native shell and image tools through the actual codec."""

    def __init__(self, root: Path, image: Path) -> None:
        self.root, self.image = root, image
        self.requests: tuple[httpx.Request, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        index = len(self.requests)
        body = json.loads(request.content)
        tools = {tool["function"]["name"] for tool in body["tools"]}
        if index == 1:
            name = "exec_command" if "exec_command" in tools else "shell_command"
            arguments = {
                (
                    "cmd" if name == "exec_command" else "command"
                ): "printf adapter-shell-evidence",
                "workdir": str(self.root),
            }
        elif index == 2:
            name, arguments = "view_image", {"path": str(self.image)}
        else:
            return native_json(
                {
                    "role": "assistant",
                    "content": json.dumps({"answer": "done"}),
                    "signature": f"sig-{index}",
                }
            )
        assert name in tools
        message = {
            "role": "assistant",
            "content": None,
            "signature": f"sig-{index}",
            "tool_calls": [
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                    "signature": f"tool-sig-{index}",
                }
            ],
        }
        return native_json(message, finish="tool_calls")


def native_json(
    message: Mapping[str, object], *, finish: str = "stop"
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "reasoning_tokens": 2,
                "total_tokens": 112,
            },
        },
    )


def test_runtime_native_tools_through_actual_gemini_bridge(tmp_path: Path) -> None:
    from PIL import Image

    from codex_agent.config import CodexAgentConfig
    from codex_agent.models import CodexTurnRequest
    from codex_agent.runtime import CodexAgentRuntime

    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(image)
    upstream = FakeTaskGemini(tmp_path, image)
    with running_adapter(
        adapter_settings(tmp_path), httpx.MockTransport(upstream)
    ) as base_url:
        home = tmp_path / "runtime-home"
        home.mkdir()
        (home / "config.toml").write_text(
            f'model_provider="codex_adapter"\n[model_providers.codex_adapter]\nname="Local test"\nbase_url={json.dumps(base_url)}\nwire_api="responses"\nrequires_openai_auth=false\n'
        )
        runtime = CodexAgentRuntime(
            CodexAgentConfig(
                project_root=tmp_path,
                codex_home=home,
                model="gemini-sdk-test",
                reasoning_effort="",
                sandbox="workspace_write",
                max_tool_calls=6,
                turn_timeout_s=15,
            )
        )
        result = runtime.run_turn(
            CodexTurnRequest(prompt="Inspect local evidence.", output_schema=_SCHEMA)
        )
    assert json.loads(result.final_response)["answer"] == "done"
    assert len(upstream.requests) == 3
    assert "adapter-shell-evidence" in upstream.requests[1].content.decode()
    assert "image_url" in upstream.requests[2].content.decode()


def test_real_sdk_tool_continuation_schema_and_multi_turn_state(tmp_path: Path) -> None:
    upstream, tools = FakeGemini(), TestTools()
    wire: list[JsonObject] = []
    with running_adapter(
        adapter_settings(tmp_path), httpx.MockTransport(upstream), wire
    ) as base_url:
        try:
            _run_sdk(sdk_config(tmp_path, base_url, "gemini-sdk-test"), tmp_path, tools)
        except RuntimeError as error:
            shapes = input_shapes(wire)
            raise AssertionError(
                f"SDK failed: {error}; input item shapes: {shapes}"
            ) from None
    assert tools.executions == ("alpha", "beta", "alpha")
    assert len(upstream.requests) == 5
    assert {request.url.params["ak"] for request in upstream.requests} == {
        "fake-sdk-key"
    }
    assert {
        json.loads(request.headers["extra"])["session_id"]
        for request in upstream.requests
    } == {"real-sdk-unique-session-123"}
