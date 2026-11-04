"""Replay stability and fail-closed option validation for Gemini Chat."""

from __future__ import annotations

from dataclasses import replace

import pytest
from test_gemini_chat import (
    CAPABILITIES,
    MODEL,
    TOOL,
    lease,
    native_message,
    native_result,
    output,
    request,
    tool_call,
    two_rounds,
)

from codex_adapter.codecs.gemini_chat import convert_response, prepare_request
from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import JsonObject, decode_object


@pytest.mark.parametrize(
    "changed",
    [
        {"max_output_tokens": 2000},
        {"temperature": 1.7},
        {"top_p": 0.95},
        {"reasoning": {"effort": "high"}},
        {"tool_choice": "none"},
        {"parallel_tool_calls": True},
        {
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "result",
                    "schema": {"type": "object"},
                    "strict": True,
                }
            }
        },
    ],
)
def test_per_turn_generation_options_do_not_change_saved_history(
    changed: JsonObject,
) -> None:
    initial: list[JsonObject] = [{"role": "user", "content": "lookup"}]
    settings: JsonObject = {
        "max_output_tokens": 1000,
        "temperature": 0.1,
        "top_p": 0.6,
        "reasoning": {"effort": "low"},
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "text": {"format": {"type": "text"}},
    }
    prepared = prepare_request(request(initial, **settings), MODEL, ())
    message = native_message(tool_call("lookup", "native-call"))
    completed = convert_response(
        native_result(message, "tool_calls"), prepared, lease(), MODEL
    )
    continuation: list[JsonObject] = [
        *initial,
        *output(completed),
        {"type": "function_call_output", "call_id": "native-call", "output": "receipt"},
    ]
    replay = prepare_request(
        request(continuation, **{**settings, **changed}), MODEL, (completed.turn,)
    )
    payload = decode_object(replay.payload)
    assert replay.options_digest == prepared.options_digest
    assert decode_object(replay.messages[-2]) == message
    assert decode_object(replay.messages[-1]) == {
        "role": "tool",
        "tool_call_id": "native-call",
        "content": "receipt",
    }
    if "max_output_tokens" in changed:
        assert payload["max_tokens"] == 2000
    if "text" in changed:
        assert isinstance(payload["response_format"], dict)
        assert payload["response_format"]["type"] == "json_schema"


@pytest.mark.parametrize("mutation", ["user", "tool_output", "projection", "carrier"])
def test_changed_generation_options_do_not_bypass_replay_checks(mutation: str) -> None:
    inputs, turns, _ = two_rounds()
    positions = {"user": 0, "tool_output": 3, "projection": 2, "carrier": 1}
    alterations: dict[str, JsonObject] = {
        "user": {"content": "changed"},
        "tool_output": {"output": "changed"},
        "projection": {"arguments": "{}"},
        "carrier": {"encrypted_content": "other"},
    }
    position = positions[mutation]
    altered = [
        *inputs[:position],
        {**inputs[position], **alterations[mutation]},
        *inputs[position + 1 :],
    ]
    with pytest.raises(AdapterError) as raised:
        prepare_request(request(altered, max_output_tokens=2000), MODEL, turns)
    assert raised.value.code == "state_conflict"


_CHANGED_TOOLS: list[JsonObject] = [
    {"tools": []},
    {"tools": [{**TOOL, "description": "different"}]},
    {
        "tools": [
            {
                **TOOL,
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                },
            }
        ]
    },
    {"tools": [{**TOOL, "strict": False}]},
]


@pytest.mark.parametrize("changed", _CHANGED_TOOLS)
def test_tool_definitions_remain_bound_to_previous_turn(changed: JsonObject) -> None:
    initial: list[JsonObject] = [{"role": "user", "content": "hello"}]
    prepared = prepare_request(request(initial), MODEL, ())
    completed = convert_response(
        native_result(native_message(text="done")), prepared, lease(), MODEL
    )
    with pytest.raises(AdapterError) as raised:
        prepare_request(
            request([*initial, *output(completed)], **changed), MODEL, (completed.turn,)
        )
    assert raised.value.code == "state_conflict"


@pytest.mark.parametrize(
    "option",
    [
        {"metadata": {"trace": 1}},
        {"client_metadata": {"request": False}},
        {"include": ["unknown.item"]},
        {"include": "reasoning.encrypted_content"},
        {"service_tier": "priority"},
        {"temperature": -0.01},
        {"temperature": 2.01},
        {"top_p": -0.1},
        {"top_p": 1.1},
        {"max_output_tokens": 2.5},
        {
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "result",
                    "schema": {},
                    "strict": "true",
                }
            }
        },
        {"text": {"format": {"type": "text", "extra": True}}},
        {"reasoning": {"effort": 5}},
        {"tool_choice": "unexpected"},
        {"tool_choice": {"type": "custom", "name": "lookup"}},
        {"tool_choice": {"type": "function", "name": "bad name"}},
    ],
)
def test_invalid_request_options_fail_before_upstream(option: JsonObject) -> None:
    with pytest.raises(AdapterError) as raised:
        prepare_request(request("hello", **option), MODEL, ())
    assert raised.value.status_code == 400


_INVALID_TOOLS: list[JsonObject] = [
    {"type": "custom", "name": "shell"},
    {**TOOL, "name": "bad name"},
    {**TOOL, "parameters": {"type": "array"}},
    {**TOOL, "parameters": {"type": "object", "properties": []}},
    {**TOOL, "parameters": {"type": "object", "required": [1]}},
    {**TOOL, "parameters": {"type": "object", "required": ["x", "x"]}},
    {**TOOL, "strict": "yes"},
    {"type": "namespace", "name": "records", "tools": []},
    {"type": "namespace", "name": "records", "tools": [{**TOOL, "namespace": "other"}]},
]


@pytest.mark.parametrize("tool", _INVALID_TOOLS)
def test_malformed_tool_definitions_are_rejected(tool: JsonObject) -> None:
    with pytest.raises(AdapterError) as raised:
        prepare_request(request("hello", tools=[tool]), MODEL, ())
    assert raised.value.status_code == 400


def test_duplicate_and_colliding_tool_names_are_rejected() -> None:
    with pytest.raises(AdapterError):
        prepare_request(request("hello", tools=[TOOL, TOOL]), MODEL, ())
    namespace: JsonObject = {"type": "namespace", "name": "records", "tools": [TOOL]}
    prepared = prepare_request(request("hello", tools=[namespace]), MODEL, ())
    colliding: JsonObject = {**TOOL, "name": prepared.tool_names[0].upstream_name}
    with pytest.raises(AdapterError):
        prepare_request(request("hello", tools=[namespace, colliding]), MODEL, ())


def test_namespace_description_is_combined_with_function_description() -> None:
    namespace: JsonObject = {
        "type": "namespace",
        "name": "records",
        "description": "Records namespace",
        "tools": [TOOL],
    }
    prepared = prepare_request(request("hello", tools=[namespace]), MODEL, ())
    tools = decode_object(prepared.payload)["tools"]
    assert isinstance(tools, list) and isinstance(tools[0], dict)
    function = tools[0]["function"]
    assert isinstance(function, dict)
    assert (
        function["description"]
        == "Namespace description: Records namespace\n\nLook up a record"
    )


def test_namespace_description_survives_without_function_description() -> None:
    namespace: JsonObject = {
        "type": "namespace",
        "name": "records",
        "description": "Records namespace",
        "tools": [
            {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
        ],
    }
    prepared = prepare_request(request("hello", tools=[namespace]), MODEL, ())
    tools = decode_object(prepared.payload)["tools"]
    assert isinstance(tools, list) and isinstance(tools[0], dict)
    function = tools[0]["function"]
    assert isinstance(function, dict)
    assert function["description"] == "Namespace description: Records namespace"


def test_tool_choice_requires_tools_and_namespace_capability_is_enforced() -> None:
    with pytest.raises(AdapterError):
        prepare_request(request("hello", tools=[], tool_choice="required"), MODEL, ())
    restricted = replace(
        MODEL, capabilities=replace(CAPABILITIES, namespace_tools=False)
    )
    with pytest.raises(AdapterError):
        prepare_request(
            request("hello", tools=[{**TOOL, "namespace": "records"}]), restricted, ()
        )


def test_invalid_saved_snapshot_is_a_state_conflict() -> None:
    initial: list[JsonObject] = [{"role": "user", "content": "hello"}]
    prepared = prepare_request(request(initial), MODEL, ())
    completed = convert_response(
        native_result(native_message(text="done")), prepared, lease(), MODEL
    )
    forged = replace(completed.turn, native_message=b"not-json")
    with pytest.raises(AdapterError) as raised:
        prepare_request(request([*initial, *output(completed)]), MODEL, (forged,))
    assert raised.value.code == "state_conflict"


@pytest.mark.parametrize("bad_number", [float("nan"), float("inf")])
def test_nonfinite_json_is_rejected_without_exposing_values(bad_number: float) -> None:
    with pytest.raises(AdapterError) as raised:
        prepare_request(request("hello", temperature=bad_number), MODEL, ())
    assert raised.value.code == "invalid_request"


def test_metadata_fields_are_validated_then_consumed_locally() -> None:
    body = request(
        "hello",
        metadata={"trace": "id"},
        client_metadata={"client": "codex"},
        prompt_cache_key="key",
    )
    upstream = decode_object(prepare_request(body, MODEL, ()).payload)
    assert (
        "metadata" not in upstream
        and "client_metadata" not in upstream
        and "prompt_cache_key" not in upstream
    )


def test_text_schema_description_is_preserved() -> None:
    fmt: JsonObject = {
        "type": "json_schema",
        "name": "result",
        "description": "expected",
        "schema": {"type": "object"},
    }
    upstream = decode_object(
        prepare_request(request("hello", text={"format": fmt}), MODEL, ()).payload
    )
    response_format = upstream["response_format"]
    assert isinstance(response_format, dict) and response_format["json_schema"] == {
        "name": "result",
        "description": "expected",
        "schema": {"type": "object"},
    }


def test_nullable_reasoning_and_text_format_options_use_defaults() -> None:
    formats: tuple[JsonObject, ...] = ({}, {"format": {"type": "text"}})
    for text in formats:
        upstream = decode_object(
            prepare_request(request("hello", text=text), MODEL, ()).payload
        )
        assert "response_format" not in upstream
