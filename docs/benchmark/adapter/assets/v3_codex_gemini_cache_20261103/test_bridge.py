"""Protocol round-trip tests; all upstream traffic uses MockTransport."""

import copy
import json

import httpx
import pytest
from bridge import MODEL, create_app
from fastapi.testclient import TestClient

KEY = "TEST-KEY-NEVER-IN-TRACE"
HEADERS = {"X-Codex-Upstream-AK": KEY, "X-Codex-Session-ID": "test-session"}
TOOLS = [
    {
        "type": "function",
        "name": "lookup_record",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
    }
]


def native_result(message, finish="stop"):
    return {
        "model": MODEL,
        "choices": [{"finish_reason": finish, "message": message}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "reasoning_tokens": 20,
            "total_tokens": 130,
            "prompt_tokens_details": {"cached_tokens": 80},
        },
    }


def tool_message(name):
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": "PRIVATE-REASONING",
        "signature": "PRIVATE-MESSAGE-SIGNATURE",
        "tool_calls": [
            {
                "index": 0,
                "id": "",
                "type": "function",
                "function": {
                    "name": "lookup_record",
                    "arguments": json.dumps({"name": name}),
                },
                "signature": "PRIVATE-TOOL-SIGNATURE",
            }
        ],
    }


def text_message(text):
    return {
        "role": "assistant",
        "content": text,
        "reasoning_content": "PRIVATE-REASONING",
        "signature": "PRIVATE-MESSAGE-SIGNATURE",
        "multimodal_contents": [
            {"type": "text", "text": text, "signature": "PRIVATE-MULTIMODAL-SIGNATURE"}
        ],
    }


def request_body(inputs, **kwargs):
    return {
        "model": MODEL,
        "instructions": "Follow the tool protocol.",
        "tools": TOOLS,
        "input": inputs,
        "include": ["reasoning.encrypted_content"],
        **kwargs,
    }


def test_two_tool_rounds_restore_all_native_fields_and_text_state(tmp_path):
    native_messages = [
        tool_message("alpha"),
        tool_message("beta"),
        text_message("done"),
        text_message("again"),
    ]
    sent = []

    def upstream(request):
        assert request.url.params["ak"] == KEY
        body = json.loads(request.content)
        sent.append(body)
        index = len(sent) - 1
        return httpx.Response(
            200,
            json=native_result(
                native_messages[index], "tool_calls" if index < 2 else "stop"
            ),
        )

    client = TestClient(
        create_app(transport=httpx.MockTransport(upstream), trace_dir=tmp_path)
    )
    inputs = [{"role": "user", "content": "look up alpha and beta"}]
    for index in range(3):
        result = client.post(
            "/v1/responses", headers=HEADERS, json=request_body(inputs)
        )
        assert result.status_code == 200, result.text
        output = result.json()["output"]
        assert output[0]["encrypted_content"].startswith("a3d-gemini-state-v1:")
        inputs.extend(output)
        if index < 2:
            assert output[1]["call_id"]
            inputs.append(
                {
                    "type": "function_call_output",
                    "call_id": output[1]["call_id"],
                    "output": json.dumps({"value": index + 1}),
                }
            )
    inputs.append({"role": "user", "content": "confirm again"})
    result = client.post("/v1/responses", headers=HEADERS, json=request_body(inputs))
    assert result.status_code == 200, result.text
    assert len(sent) == 4  # No model-answer cache.
    assert [
        message for message in sent[3]["messages"] if message["role"] == "assistant"
    ] == native_messages[:3]
    tool_results = [
        message for message in sent[3]["messages"] if message["role"] == "tool"
    ]
    assert tool_results == [
        {"role": "tool", "tool_call_id": "", "content": '{"value": 1}'},
        {"role": "tool", "tool_call_id": "", "content": '{"value": 2}'},
    ]
    assert result.json()["usage"] == {
        "input_tokens": 100,
        "input_tokens_details": {"cached_tokens": 80},
        "output_tokens": 30,
        "output_tokens_details": {"reasoning_tokens": 20},
        "total_tokens": 130,
    }
    trace_text = (tmp_path / "bridge_trace.jsonl").read_text()
    for forbidden in (
        KEY,
        "PRIVATE-MESSAGE-SIGNATURE",
        "PRIVATE-TOOL-SIGNATURE",
        "PRIVATE-MULTIMODAL-SIGNATURE",
        "PRIVATE-REASONING",
        output[0]["encrypted_content"],
    ):
        assert forbidden not in trace_text
    rows = [json.loads(line) for line in trace_text.splitlines()]
    assert len(rows[-1]["restored_states"]) == 3
    assert rows[-1]["raw_usage"]["reasoning_tokens"] == 20
    assert rows[-1]["restored_states"] == [row["saved_state"] for row in rows[:3]]


def test_binding_rejects_changed_key_and_gpt_without_upstream_call(tmp_path):
    sent = []

    def upstream(request):
        sent.append(request)
        return httpx.Response(200, json=native_result(text_message("done")))

    client = TestClient(
        create_app(transport=httpx.MockTransport(upstream), trace_dir=tmp_path)
    )
    assert (
        client.post(
            "/v1/responses", headers=HEADERS, json=request_body("hello")
        ).status_code
        == 200
    )
    changed = {**HEADERS, "X-Codex-Upstream-AK": "DIFFERENT-SECRET"}
    result = client.post("/v1/responses", headers=changed, json=request_body("hello"))
    assert result.status_code == 409
    assert result.json()["error"]["code"] == "immutable_session_binding_mismatch"
    result = client.post(
        "/v1/responses", headers=HEADERS, json=request_body("hello", model="gpt-5.5")
    )
    assert result.status_code == 400
    assert len(sent) == 1
    assert "DIFFERENT-SECRET" not in (tmp_path / "bridge_trace.jsonl").read_text()


@pytest.mark.parametrize(
    "mutation", ["missing", "bad_handle", "changed_projection", "foreign_session"]
)
def test_state_replay_fails_loudly(mutation):
    sent = []

    def upstream(request):
        sent.append(request)
        return httpx.Response(200, json=native_result(text_message("done")))

    client = TestClient(create_app(transport=httpx.MockTransport(upstream)))
    initial = [{"role": "user", "content": "hello"}]
    result = client.post(
        "/v1/responses", headers=HEADERS, json=request_body(initial)
    ).json()
    inputs = initial + copy.deepcopy(result["output"])
    headers = HEADERS
    if mutation == "missing":
        inputs = initial
    elif mutation == "bad_handle":
        inputs[1]["encrypted_content"] += "bad"
    elif mutation == "changed_projection":
        inputs[2]["content"][0]["text"] = "tampered"
    else:
        headers = {**HEADERS, "X-Codex-Session-ID": "foreign-session"}
    response = client.post("/v1/responses", headers=headers, json=request_body(inputs))
    assert response.status_code in {400, 409}
    assert len(sent) == 1


def test_sse_lifecycle_contains_full_carrier_tools_and_final_output():
    message = tool_message("alpha")
    message["content"] = "I will look it up."
    client = TestClient(
        create_app(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json=native_result(message, "tool_calls")
                )
            )
        )
    )
    response = client.post(
        "/v1/responses", headers=HEADERS, json=request_body("hello", stream=True)
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.completed"
    done_items = [
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done"
    ]
    assert done_items == events[-1]["response"]["output"]
    event_types = {event["type"] for event in events}
    assert {
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.added",
        "response.content_part.done",
    } <= event_types
    assert done_items[0]["encrypted_content"].startswith("a3d-gemini-state-v1:")


def test_transport_error_never_logs_or_returns_credential(tmp_path):
    def upstream(request):
        raise httpx.ConnectError(
            f"intentional test exception {request.url}", request=request
        )

    client = TestClient(
        create_app(transport=httpx.MockTransport(upstream), trace_dir=tmp_path)
    )
    result = client.post("/v1/responses", headers=HEADERS, json=request_body("hello"))
    assert result.status_code == 502
    assert KEY not in result.text
    assert KEY not in (tmp_path / "bridge_trace.jsonl").read_text()


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "custom", "name": "shell"},
        {"type": "namespace", "name": "tools", "tools": []},
    ],
)
def test_unsupported_tools_rejected_before_upstream(tool):
    def upstream(request):
        pytest.fail("unsupported request reached upstream")

    client = TestClient(create_app(transport=httpx.MockTransport(upstream)))
    result = client.post(
        "/v1/responses", headers=HEADERS, json=request_body("hello", tools=[tool])
    )
    assert result.status_code == 400


def test_tracking_metadata_is_logged_without_entering_model_input(tmp_path):
    sent = []

    def upstream(request):
        sent.append(json.loads(request.content))
        assert (
            json.loads(request.headers["extra"])["session_id"]
            == HEADERS["X-Codex-Session-ID"]
        )
        return httpx.Response(200, json=native_result(text_message("done")))

    client = TestClient(
        create_app(transport=httpx.MockTransport(upstream), trace_dir=tmp_path)
    )
    body = request_body(
        "hello",
        client_metadata={"trace_id": "CLIENT-TRACE"},
        prompt_cache_key="CACHE-KEY",
    )
    assert client.post("/v1/responses", headers=HEADERS, json=body).status_code == 200
    assert "client_metadata" not in sent[0] and "prompt_cache_key" not in sent[0]
    assert sent[0]["max_tokens"] == 8192
    assert sent[0]["messages"] == [
        {"role": "system", "content": "Follow the tool protocol."},
        {"role": "user", "content": "hello"},
    ]
    trace = json.loads((tmp_path / "bridge_trace.jsonl").read_text())
    assert trace["ingress"]["client_metadata"]["trace_id"] == "CLIENT-TRACE"
    assert trace["ingress"]["prompt_cache_key"] == "CACHE-KEY"
    assert trace["resolved_config"]["max_output_tokens_source"] == "probe_fixed_default"


def test_service_tier_rejected_explicitly():
    client = TestClient(
        create_app(
            transport=httpx.MockTransport(
                lambda request: pytest.fail("unexpected upstream")
            )
        )
    )
    result = client.post(
        "/v1/responses",
        headers=HEADERS,
        json=request_body("hello", service_tier="priority"),
    )
    assert result.status_code == 400
    assert result.json()["error"]["code"] == "service_tier_not_supported_by_probe"


def test_developer_role_maps_to_system_preserving_text_and_order(tmp_path):
    sent = []

    def upstream(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=native_result(text_message("done")))

    client = TestClient(
        create_app(transport=httpx.MockTransport(upstream), trace_dir=tmp_path)
    )
    inputs = [
        {"role": "developer", "content": "first developer"},
        {
            "role": "developer",
            "content": [{"type": "input_text", "text": "second developer"}],
        },
        {"role": "user", "content": "hello"},
    ]
    result = client.post("/v1/responses", headers=HEADERS, json=request_body(inputs))
    assert result.status_code == 200
    assert sent[0]["messages"] == [
        {"role": "system", "content": "Follow the tool protocol."},
        {"role": "system", "content": "first developer"},
        {"role": "system", "content": "second developer"},
        {"role": "user", "content": "hello"},
    ]
    trace = json.loads((tmp_path / "bridge_trace.jsonl").read_text())
    assert trace["ingress"]["input"] == inputs
    assert (
        trace["client_metadata_policy"]["developer_role"]
        == "map_to_system_preserve_text_and_order"
    )


def test_upstream_error_retains_diagnostics_without_secrets_or_opaque_state(tmp_path):
    opaque = "abcDEF012_+/-" * 40
    message = f"Unsupported role developer; token={KEY}; url=https://host.example/api?ak={KEY}&state={opaque}; signature={opaque}"
    payload = {
        "error": {
            "code": "invalid_request",
            "type": "bad_request",
            "message": message,
            "signature": "PRIVATE-SIGNATURE",
        }
    }
    client = TestClient(
        create_app(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(400, json=payload)
            ),
            trace_dir=tmp_path,
        )
    )
    result = client.post("/v1/responses", headers=HEADERS, json=request_body("hello"))
    assert result.status_code == 502
    raw_trace = (tmp_path / "bridge_trace.jsonl").read_text()
    for forbidden in (KEY, opaque, "PRIVATE-SIGNATURE", "host.example"):
        assert forbidden not in raw_trace
        assert forbidden not in result.text
    trace = json.loads(raw_trace)
    assert trace["upstream_error"]["code"] == "invalid_request"
    assert trace["upstream_error"]["type"] == "bad_request"
    assert "Unsupported role developer" in trace["upstream_error"]["message"]
    assert "[REDACTED_OPAQUE]" in trace["upstream_error"]["message"]


def test_visible_token_control_removes_only_signatures_without_mutating_history(
    tmp_path,
):
    sent = []

    def upstream(request):
        assert request.url.params["ak"] == KEY
        assert (
            json.loads(request.headers["extra"])["session_id"]
            == HEADERS["X-Codex-Session-ID"]
        )
        sent.append(json.loads(request.content))
        return httpx.Response(
            200, json=native_result(text_message(f"answer {len(sent)}"))
        )

    application = create_app(
        transport=httpx.MockTransport(upstream), trace_dir=tmp_path
    )
    client = TestClient(application)
    inputs = [{"role": "user", "content": "hello"}]
    first = client.post(
        "/v1/responses", headers=HEADERS, json=request_body(inputs, tools=[])
    )
    inputs.extend(first.json()["output"])
    inputs.append({"role": "user", "content": "continue"})
    second = client.post(
        "/v1/responses", headers=HEADERS, json=request_body(inputs, tools=[])
    )
    assert second.status_code == 200
    bridge = application.state.bridge
    before_carriers = list(bridge.sessions["test-session"].carriers)
    before_turns = copy.deepcopy(bridge.turns)
    original = copy.deepcopy(bridge.last_requests["test-session"])
    response = client.post("/probe/visible-token-control", headers=HEADERS)
    assert response.status_code == 200, response.text
    assert set(response.json()) == {"upstream_status", "raw_usage"}
    expected = copy.deepcopy(original)
    expected["max_tokens"] = 128
    del expected["messages"][2]["signature"]
    del expected["messages"][2]["multimodal_contents"][0]["signature"]
    assert sent[-1] == expected
    assert bridge.sessions["test-session"].carriers == before_carriers
    assert bridge.turns == before_turns
    assert bridge.last_requests["test-session"] == original
    assert KEY not in json.dumps(bridge.last_requests)
    inputs.extend(second.json()["output"])
    inputs.append({"role": "user", "content": "after control"})
    followup = client.post(
        "/v1/responses", headers=HEADERS, json=request_body(inputs, tools=[])
    )
    assert followup.status_code == 200, followup.text
    trace_text = (tmp_path / "bridge_trace.jsonl").read_text()
    control = next(
        json.loads(line)
        for line in trace_text.splitlines()
        if json.loads(line).get("kind") == "visible_token_control"
    )
    assert len(control["removed_state_fields"]) == 2
    assert control["removed_state_fields"][0]["path"] == "$.messages[2].signature"
    assert control["original_body"]["sha256"]
    assert KEY not in trace_text
    assert "PRIVATE-MESSAGE-SIGNATURE" not in trace_text
    assert "PRIVATE-MULTIMODAL-SIGNATURE" not in trace_text


def test_visible_token_control_enforces_key_binding_and_existing_request():
    sent = []

    def upstream(request):
        sent.append(request)
        return httpx.Response(200, json=native_result(text_message("answer")))

    client = TestClient(create_app(transport=httpx.MockTransport(upstream)))
    assert (
        client.post("/probe/visible-token-control", headers=HEADERS).status_code == 409
    )
    assert not sent
    assert (
        client.post(
            "/v1/responses", headers=HEADERS, json=request_body("hello", tools=[])
        ).status_code
        == 200
    )
    bad_headers = {**HEADERS, "X-Codex-Upstream-AK": "WRONG-KEY"}
    response = client.post("/probe/visible-token-control", headers=bad_headers)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "immutable_session_binding_mismatch"
    assert len(sent) == 1
