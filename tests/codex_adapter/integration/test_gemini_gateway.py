"""Full HTTP/SQLite Gemini state continuity, replay and restart contracts."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from codex_adapter.config import AdapterSettings, ServerSettings
from codex_adapter.domain.json_codec import JsonObject
from codex_adapter.domain.models import (
    ModelCapabilities,
    RegisteredModel,
    Registry,
    Upstream,
)
from codex_adapter.domain.state import StateSettings
from codex_adapter.interfaces.app import create_app

HEADERS = {"X-Codex-Session-ID": "session-gemini-http-123"}
NATIVE_TOOL: JsonObject = {
    "role": "assistant",
    "content": None,
    "signature": "message-opaque-test-signature",
    "tool_calls": [
        {
            "id": "",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{ "name": "alpha" }'},
            "signature": "tool-opaque-test-signature",
        }
    ],
    "multimodal_contents": [
        {"type": "text", "text": "", "signature": "part-opaque-test-signature"}
    ],
    "future_extension": {"preserve": [1, "yes"]},
}


def settings(tmp_path: Path) -> AdapterSettings:
    capabilities = ModelCapabilities(
        reasoning_efforts=(), default_reasoning_effort=None, reasoning_summary=False
    )
    model = RegisteredModel(
        "gemini-test",
        "gemini_chat",
        (
            Upstream(
                "test-key", "v1", "https://upstream.example/chat", "fake-http-test-key"
            ),
        ),
        capabilities,
    )
    return AdapterSettings(
        Registry((model,)),
        StateSettings(tmp_path / "state.sqlite"),
        ServerSettings(),
    )


def native_response(message: JsonObject, *, finish: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "native-result",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "reasoning_tokens": 3,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 7},
            },
        },
    )


def seed_request() -> JsonObject:
    return {
        "model": "gemini-test",
        "input": [{"role": "user", "content": "Look up alpha"}],
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up a record",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }
        ],
    }


def continuation_request(response: JsonObject) -> JsonObject:
    output = response["output"]
    assert isinstance(output, list)
    wire_call = next(
        item
        for item in output
        if isinstance(item, dict) and item.get("type") == "function_call"
    )
    assert isinstance(wire_call, dict)
    initial = seed_request()["input"]
    assert isinstance(initial, list)
    return {
        **seed_request(),
        "input": [
            *initial,
            *output,
            {
                "type": "function_call_output",
                "call_id": wire_call["call_id"],
                "output": "receipt-alpha",
            },
        ],
    }


def test_tool_signatures_replay_and_restart_remain_on_one_key(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    requests: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        messages = json.loads(request.content)["messages"]
        if len(requests) == 1:
            return native_response(NATIVE_TOOL, finish="tool_calls")
        assert messages[1] == NATIVE_TOOL
        assert messages[2] == {
            "role": "tool",
            "tool_call_id": "",
            "content": "receipt-alpha",
        }
        return native_response(
            {
                "role": "assistant",
                "content": "done",
                "signature": "final-opaque-signature",
            }
        )

    transport = httpx.MockTransport(upstream)
    with TestClient(
        create_app(configured, upstream_transport=transport, initialize_state=True)
    ) as client:
        first = client.post("/v1/responses", json=seed_request(), headers=HEADERS)
        assert first.status_code == 200, first.text
        continuation = continuation_request(first.json())
        second = client.post("/v1/responses", json=continuation, headers=HEADERS)
        assert second.status_code == 200, second.text
        replay = client.post("/v1/responses", json=continuation, headers=HEADERS)
        assert replay.content == second.content
        assert replay.headers["x-codex-adapter-replay"] == "true"
    with TestClient(create_app(configured, upstream_transport=transport)) as client:
        replay_after_restart = client.post(
            "/v1/responses", json=continuation, headers=HEADERS
        )
        assert replay_after_restart.content == second.content
    assert len(requests) == 2
    assert {request.url.params["ak"] for request in requests} == {"fake-http-test-key"}
    assert {
        json.loads(request.headers["extra"])["session_id"] for request in requests
    } == {HEADERS["X-Codex-Session-ID"]}
    assert b"fake-http-test-key" not in configured.state.database.read_bytes()
    assert b"tool-opaque-test-signature" in configured.state.database.read_bytes()


def test_same_idempotency_key_with_different_payload_fails_without_new_generation(
    tmp_path: Path,
) -> None:
    calls: list[bytes] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request.content)
        return native_response({"role": "assistant", "content": "done"})

    with TestClient(
        create_app(
            settings(tmp_path),
            upstream_transport=httpx.MockTransport(upstream),
            initialize_state=True,
        )
    ) as client:
        headers = {**HEADERS, "Idempotency-Key": "one-logical-request"}
        first = client.post("/v1/responses", json=seed_request(), headers=headers)
        changed = client.post(
            "/v1/responses",
            json={**seed_request(), "input": "different"},
            headers=headers,
        )
    assert first.status_code == 200, first.text
    assert changed.status_code == 409
    assert len(calls) == 1
