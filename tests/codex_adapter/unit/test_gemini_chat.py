"""Gemini ModelHub Chat translation, replay, and streaming contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

import pytest

from codex_adapter.codecs.gemini_chat import (
    convert_response,
    last_carrier,
    prepare_request,
    response_events,
)
from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import (
    JsonObject,
    JsonValue,
    decode_object,
    encode_json,
)
from codex_adapter.domain.models import ModelCapabilities, RegisteredModel, Upstream
from codex_adapter.domain.state import Binding, Completion, RequestLease, SavedTurn

MODEL_ID = "gemini-test"
CAPABILITIES = ModelCapabilities(
    reasoning_summary=False,
    reasoning_efforts=("low", "medium", "high"),
    default_reasoning_effort=None,
)
MODEL = RegisteredModel(
    id=MODEL_ID,
    protocol="gemini_chat",
    upstreams=(
        Upstream(
            id="fake",
            version="1",
            url="https://example.invalid/chat",
            secret="test-placeholder",
        ),
    ),
    capabilities=CAPABILITIES,
)
BINDING = Binding(
    id="binding-1",
    principal="test-principal",
    session_id="test-session",
    model_id=MODEL_ID,
    protocol="gemini_chat",
    codec_version=MODEL.codec_version,
    endpoint=MODEL.upstreams[0].url,
    credential_id="fake",
    credential_version="1",
    credential_fingerprint="fake-fingerprint",
    created_at=1700000000.0,
    expires_at=1800000000.0,
)
TOOL: JsonObject = {
    "type": "function",
    "name": "lookup",
    "description": "Look up a record",
    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
    "strict": True,
}


def lease(index: int = 1) -> RequestLease:
    return RequestLease(
        binding=BINDING,
        request_id=f"request-{index}",
        request_digest="fake-digest",
        idempotency_key=f"key-{index}",
        carrier=f"test-carrier-{index}",
        response_id=f"resp_test_{index}",
        sequence=index,
        previous_carrier=f"test-carrier-{index - 1}" if index > 1 else None,
        created_at=1700000100.0 + index,
    )


def request(
    items: str | Sequence[Mapping[str, JsonValue]], **options: JsonValue
) -> JsonObject:
    input_value: JsonValue = (
        items if isinstance(items, str) else [dict(item) for item in items]
    )
    result: JsonObject = {
        "model": MODEL_ID,
        "instructions": "Follow the tool protocol.",
        "tools": [TOOL],
        "input": input_value,
    }
    return {**result, **options}


def tool_call(
    name: str, wire_id: str, arguments: str = '{"name":"alpha"}'
) -> JsonObject:
    return {
        "id": wire_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
        "signature": "tool-signature-secret",
        "extension": {"opaque": "preserve"},
    }


def native_message(*calls: JsonObject, text: str | None = None) -> JsonObject:
    return {
        "role": "assistant",
        "content": text,
        "reasoning_content": "private-reasoning",
        "signature": "message-signature-secret",
        "multimodal_contents": [
            {"type": "text", "text": text or "", "signature": "multimodal-secret"}
        ],
        "tool_calls": list(calls),
        "unknown_extension": {"nested": [1, {"future": True}]},
    }


def native_result(
    message: JsonObject,
    finish: str = "stop",
    usage: JsonObject | None = None,
) -> JsonObject:
    return {
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": (
            usage
            if usage is not None
            else {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "reasoning_tokens": 20,
                "total_tokens": 130,
                "prompt_tokens_details": {"cached_tokens": 80},
            }
        ),
    }


def output(completion: Completion) -> list[JsonObject]:
    value = decode_object(completion.response)["output"]
    assert isinstance(value, list) and all(isinstance(item, dict) for item in value)
    # decode_object validates JSON recursively; the assertion checks every item is an object.
    return cast(list[JsonObject], value)


def object_list(value: object) -> list[JsonObject]:
    assert isinstance(value, list) and all(isinstance(item, dict) for item in value)
    return value


def two_rounds() -> tuple[list[JsonObject], tuple[SavedTurn, ...], list[JsonObject]]:
    inputs: list[JsonObject] = [{"role": "user", "content": "find records"}]
    turns: tuple[SavedTurn, ...] = ()
    native_messages = [
        native_message(tool_call("lookup", "", '{ "name" : "alpha" }')),
        native_message(tool_call("lookup", "", '{"name":"beta"}')),
    ]
    for index, message in enumerate(native_messages, start=1):
        prepared = prepare_request(request(inputs), MODEL, turns)
        completion = convert_response(
            native_result(message, "tool_calls"), prepared, lease(index), MODEL
        )
        projection = output(completion)
        call = next(item for item in projection if item["type"] == "function_call")
        inputs = [
            *inputs,
            *projection,
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": f"result-{index}",
            },
        ]
        turns = (*turns, completion.turn)
    return inputs, turns, native_messages


@pytest.mark.parametrize("content", [None, []])
def test_sdk_empty_reasoning_content_does_not_change_native_state(
    content: JsonValue,
) -> None:
    inputs, turns, originals = two_rounds()
    sdk_inputs = [
        {**item, "content": content} if item.get("type") == "reasoning" else item
        for item in inputs
    ]
    restored = prepare_request(request(sdk_inputs), MODEL, turns)
    messages = object_list(decode_object(restored.payload)["messages"])
    assert [item for item in messages if item["role"] == "assistant"] == originals


def test_two_empty_native_ids_across_turns_round_trip_all_native_bytes() -> None:
    inputs, turns, originals = two_rounds()
    prepared = prepare_request(request(inputs), MODEL, turns)
    upstream = decode_object(prepared.payload)
    messages = object_list(upstream["messages"])
    assert upstream["stream"] is False
    assert [m for m in messages if m["role"] == "assistant"] == originals
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["", ""]
    assert [turn.native_message for turn in turns] == [
        encode_json(item) for item in originals
    ]
    assert all(
        wire and native == "" for turn in turns for wire, native in turn.call_ids
    )
    assert prepared.messages == tuple(encode_json(item) for item in messages)
    assert last_carrier(request(inputs)) == turns[-1].carrier
    assert b"signature-secret" not in encode_json(output_from_turn(turns[0]))


def output_from_turn(turn: SavedTurn) -> list[JsonValue]:
    return [decode_object(value) for value in turn.projections]


def test_saved_prefix_count_and_digest_include_full_native_assistant() -> None:
    prepared = prepare_request(request("hello"), MODEL, ())
    message = native_message(text="answer")
    completion = convert_response(native_result(message), prepared, lease(), MODEL)
    expected = (*prepared.messages, encode_json(message))
    assert completion.turn.prefix_count == len(expected)
    assert (
        completion.turn.prefix_digest
        == hashlib.sha256(b"[" + b",".join(expected) + b"]").hexdigest()
    )
    assert completion.turn.options_digest == prepared.options_digest
    assert decode_object(completion.response)["created_at"] == 1700000101


def test_matching_projection_must_not_be_inserted_as_second_assistant() -> None:
    initial: list[JsonObject] = [{"role": "user", "content": "hello"}]
    prepared = prepare_request(request(initial), MODEL, ())
    message = native_message(text="answer")
    completion = convert_response(native_result(message), prepared, lease(), MODEL)
    replay = prepare_request(
        request([*initial, *output(completion), {"role": "user", "content": "next"}]),
        MODEL,
        (completion.turn,),
    )
    messages = object_list(decode_object(replay.payload)["messages"])
    assert [item for item in messages if item["role"] == "assistant"] == [message]
    assert b"message-signature-secret" not in completion.response
    assert b"private-reasoning" not in completion.response


def test_parallel_tool_outputs_are_associated_by_id_not_order() -> None:
    prepared = prepare_request(request("go", parallel_tool_calls=True), MODEL, ())
    calls = (
        tool_call("lookup", "first"),
        tool_call("lookup", "second"),
        tool_call("lookup", ""),
    )
    completion = convert_response(
        native_result(native_message(*calls), "tool_calls"), prepared, lease(), MODEL
    )
    projections = output(completion)
    projected_calls = [item for item in projections if item["type"] == "function_call"]
    wire_ids = [item["call_id"] for item in projected_calls]
    assert wire_ids[:2] == ["first", "second"]
    assert isinstance(wire_ids[2], str) and wire_ids[2] not in ("", *wire_ids[:2])
    results: list[JsonObject] = [
        {"type": "function_call_output", "call_id": wire_ids[2], "output": "empty"},
        {"type": "function_call_output", "call_id": wire_ids[1], "output": "second"},
        {"type": "function_call_output", "call_id": wire_ids[0], "output": "first"},
    ]
    body = request(
        [{"role": "user", "content": "go"}, *projections, *results],
        parallel_tool_calls=True,
    )
    replay = prepare_request(body, MODEL, (completion.turn,))
    messages = object_list(decode_object(replay.payload)["messages"])
    assert [
        (m["tool_call_id"], m["content"]) for m in messages if m["role"] == "tool"
    ] == [("", "empty"), ("second", "second"), ("first", "first")]


@pytest.mark.parametrize("ids", [("", ""), ("same", "same")])
def test_ambiguous_native_tool_ids_are_rejected(ids: tuple[str, str]) -> None:
    prepared = prepare_request(request("go"), MODEL, ())
    message = native_message(*(tool_call("lookup", value) for value in ids))
    with pytest.raises(AdapterError) as raised:
        convert_response(native_result(message, "tool_calls"), prepared, lease(), MODEL)
    assert raised.value.status_code == 502


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "foreign",
        "reordered",
        "projection",
        "user",
        "instructions",
        "tool_result",
        "tools",
    ],
)
def test_replay_rejects_modified_or_incomplete_history(mutation: str) -> None:
    inputs, turns, _ = two_rounds()
    body = request(inputs)
    if mutation == "missing":
        body = request(inputs[:1])
    elif mutation == "foreign":
        reason = {**inputs[1], "encrypted_content": "foreign"}
        body = request([inputs[0], reason, *inputs[2:]])
    elif mutation == "reordered":
        body = request([inputs[0], *inputs[4:7], *inputs[1:4], *inputs[7:]])
    elif mutation == "projection":
        changed = {**inputs[2], "arguments": '{"name":"tampered"}'}
        body = request([*inputs[:2], changed, *inputs[3:]])
    elif mutation == "user":
        body = request([{**inputs[0], "content": "tampered"}, *inputs[1:]])
    elif mutation == "instructions":
        body = {**body, "instructions": "tampered"}
    elif mutation == "tool_result":
        body = request([*inputs[:3], {**inputs[3], "output": "tampered"}, *inputs[4:]])
    else:
        body = {**body, "tools": [{**TOOL, "strict": False}]}
    with pytest.raises(AdapterError):
        prepare_request(body, MODEL, turns)


def test_tampered_native_snapshot_or_prefix_count_is_rejected() -> None:
    inputs, turns, _ = two_rounds()
    native = decode_object(turns[0].native_message)
    forged = replace(
        turns[0], native_message=encode_json({**native, "signature": "changed"})
    )
    with pytest.raises(AdapterError):
        prepare_request(request(inputs), MODEL, (forged, turns[1]))
    with pytest.raises(AdapterError):
        prepare_request(
            request(inputs), MODEL, (replace(turns[0], prefix_count=999), turns[1])
        )


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "unknown", "message_before_result"]
)
def test_tool_results_must_be_complete_once_and_precede_messages(mutation: str) -> None:
    inputs, turns, _ = two_rounds()
    last = inputs[-1]
    if mutation == "missing":
        changed = inputs[:-1]
    elif mutation == "duplicate":
        changed = [*inputs, last]
    elif mutation == "unknown":
        changed = [*inputs[:-1], {**last, "call_id": "unknown"}]
    else:
        changed = [*inputs[:-1], {"role": "user", "content": "before result"}, last]
    with pytest.raises(AdapterError):
        prepare_request(request(changed), MODEL, turns)


def namespace_schema_body() -> tuple[JsonObject, JsonObject]:
    namespace: JsonObject = {"type": "namespace", "name": "records", "tools": [TOOL]}
    schema: JsonObject = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    body = request(
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "inspect"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,YQ==",
                        "detail": "high",
                    },
                    {
                        "type": "input_image",
                        "image_url": "https://example.invalid/image.png",
                    },
                ],
            }
        ],
        tools=[namespace],
        tool_choice={"type": "function", "namespace": "records", "name": "lookup"},
        parallel_tool_calls=True,
        text={
            "format": {
                "type": "json_schema",
                "name": "result",
                "strict": True,
                "schema": schema,
            }
        },
        reasoning={"effort": "high", "summary": "none"},
        max_output_tokens=301,
    )
    return body, schema


def test_schema_namespace_tool_choice_and_effort_are_translated() -> None:
    body, schema = namespace_schema_body()
    prepared = prepare_request(body, MODEL, ())
    upstream = decode_object(prepared.payload)
    names = prepared.tool_names
    assert (
        len(names) == 1
        and names[0].name == "lookup"
        and names[0].namespace == "records"
    )
    assert object_list(upstream["tools"])[0]["function"] == {
        "name": names[0].upstream_name,
        "description": TOOL["description"],
        "parameters": TOOL["parameters"],
        "strict": True,
    }
    assert upstream["tool_choice"] == {
        "type": "function",
        "function": {"name": names[0].upstream_name},
    }
    assert upstream["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "result", "strict": True, "schema": schema},
    }
    assert upstream["max_tokens"] == 301 and upstream["reasoning_effort"] == "high"
    assert upstream["parallel_tool_calls"] is True
    assert prepared.tool_names == prepare_request(body, MODEL, ()).tool_names
    native = native_message(tool_call(names[0].upstream_name, "native-id"))
    completion = convert_response(
        native_result(native, "tool_calls"), prepared, lease(), MODEL
    )
    call = next(item for item in output(completion) if item["type"] == "function_call")
    assert call["name"] == "lookup" and call["namespace"] == "records"
    assert call["arguments"] == '{"name":"alpha"}'


def test_mixed_data_and_https_images_are_translated() -> None:
    body, _ = namespace_schema_body()
    upstream = decode_object(prepare_request(body, MODEL, ()).payload)
    message = object_list(upstream["messages"])[-1]
    assert message["content"] == [
        {"type": "text", "text": "inspect"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,YQ==", "detail": "high"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/image.png"},
        },
    ]


def test_developer_and_system_messages_and_text_parts_map_correctly() -> None:
    inputs: list[JsonObject] = [
        {"role": "developer", "content": [{"type": "input_text", "text": "dev"}]},
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "a"},
                {"type": "input_text", "text": "b"},
            ],
        },
    ]
    prepared = prepare_request(request(inputs), MODEL, ())
    messages = object_list(decode_object(prepared.payload)["messages"])
    assert messages == [
        {"role": "system", "content": "Follow the tool protocol."},
        {"role": "system", "content": "dev"},
        {"role": "system", "content": "system"},
        {"role": "user", "content": "ab"},
    ]


@pytest.mark.parametrize(
    "url",
    [
        "file:///private/image.png",
        "/tmp/image.png",
        "http://example.invalid/a.png",
        "https://",
        "data:text/html;base64,YQ==",
    ],
)
def test_unsafe_or_invalid_image_urls_are_rejected(url: str) -> None:
    body = request(
        [{"role": "user", "content": [{"type": "input_image", "image_url": url}]}]
    )
    with pytest.raises(AdapterError):
        prepare_request(body, MODEL, ())


@pytest.mark.parametrize(
    "options",
    [
        {"store": True},
        {"stream": "true"},
        {"previous_response_id": "resp_old"},
        {"conversation": "conv_old"},
        {"truncation": "auto"},
        {"unknown": 1},
        {"metadata": "wrong"},
        {"client_metadata": ["wrong"]},
        {"prompt_cache_key": 5},
        {"max_output_tokens": True},
        {"max_output_tokens": 0},
        {"reasoning": {"effort": "invented"}},
        {"reasoning": {"summary": "detailed"}},
        {"text": {"format": {"type": "json_schema", "name": "x", "schema": []}}},
        {"tool_choice": {"type": "function", "name": "missing"}},
        {"parallel_tool_calls": "true"},
        {"temperature": True},
    ],
)
def test_unsupported_or_malformed_options_fail_closed(
    options: JsonObject,
) -> None:
    with pytest.raises(AdapterError):
        prepare_request(request("hello", **options), MODEL, ())


@pytest.mark.parametrize(
    "capability,options",
    [
        (
            "structured_output",
            {
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "x",
                        "schema": {"type": "object"},
                    }
                }
            },
        ),
        ("function_tools", {}),
        (
            "namespace_tools",
            {"tools": [{"type": "namespace", "name": "ns", "tools": [TOOL]}]},
        ),
        ("parallel_tools", {"parallel_tool_calls": True}),
    ],
)
def test_model_capabilities_are_enforced(capability: str, options: JsonObject) -> None:
    configurations = {
        "structured_output": replace(CAPABILITIES, structured_output=False),
        "function_tools": replace(CAPABILITIES, function_tools=False),
        "namespace_tools": replace(CAPABILITIES, namespace_tools=False),
        "parallel_tools": replace(CAPABILITIES, parallel_tools=False),
    }
    model = replace(MODEL, capabilities=configurations[capability])
    with pytest.raises(AdapterError):
        prepare_request(request("hello", **options), model, ())


def test_image_and_reasoning_effort_capability_are_enforced() -> None:
    no_images = replace(MODEL, capabilities=replace(CAPABILITIES, images=False))
    body = request(
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "https://example.invalid/a"}
                ],
            }
        ]
    )
    with pytest.raises(AdapterError):
        prepare_request(body, no_images, ())
    no_reasoning = replace(
        MODEL, capabilities=replace(CAPABILITIES, reasoning_efforts=())
    )
    with pytest.raises(AdapterError):
        prepare_request(
            request("hello", reasoning={"effort": "high"}), no_reasoning, ()
        )


def test_usage_accounting_preserves_raw_and_keeps_absent_details_unknown() -> None:
    prepared = prepare_request(request("hello"), MODEL, ())
    message = native_message(text="answer")
    full = native_result(message)
    completion = convert_response(full, prepared, lease(), MODEL)
    assert completion.raw_usage == encode_json(full["usage"])
    assert decode_object(completion.response)["usage"] == {
        "input_tokens": 100,
        "input_tokens_details": {"cached_tokens": 80},
        "output_tokens": 30,
        "output_tokens_details": {"reasoning_tokens": 20},
        "total_tokens": 130,
    }
    missing: JsonObject = {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 15,
    }
    result = convert_response(
        native_result(message, usage=missing), prepared, lease(), MODEL
    )
    assert decode_object(result.response)["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


@pytest.mark.parametrize(
    "usage",
    [
        {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "reasoning_tokens": 2,
            "total_tokens": 14,
        },
        {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 12},
        {"prompt_tokens": True, "completion_tokens": 3, "total_tokens": 14},
        {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "total_tokens": 13,
            "prompt_tokens_details": {"cached_tokens": 11},
        },
    ],
)
def test_bad_usage_is_rejected_not_guessed(usage: JsonObject) -> None:
    prepared = prepare_request(request("hello"), MODEL, ())
    with pytest.raises(AdapterError) as raised:
        convert_response(
            native_result(native_message(text="answer"), usage=usage),
            prepared,
            lease(),
            MODEL,
        )
    assert raised.value.status_code == 502


@pytest.mark.parametrize(
    "call",
    [
        {
            "id": "x",
            "type": "function",
            "function": {"name": "lookup", "arguments": "not json"},
        },
        {
            "id": "x",
            "type": "function",
            "function": {"name": "lookup", "arguments": "[]"},
        },
        {
            "id": "x",
            "type": "function",
            "function": {"name": "unknown", "arguments": "{}"},
        },
        {
            "id": "x",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"a":1,"a":2}'},
        },
        {"type": "function", "function": {"name": "lookup", "arguments": "{}"}},
    ],
)
def test_invalid_or_unregistered_native_calls_fail(call: JsonObject) -> None:
    prepared = prepare_request(request("hello"), MODEL, ())
    with pytest.raises(AdapterError) as raised:
        convert_response(
            native_result(native_message(call), "tool_calls"), prepared, lease(), MODEL
        )
    assert raised.value.status_code == 502


def events(completion: Completion) -> list[JsonObject]:
    frames = response_events(completion.response)
    result: list[JsonObject] = []
    for frame in frames:
        assert frame.endswith(b"\n\n")
        event_line, data_line = frame.rstrip(b"\n").split(b"\n", 1)
        assert event_line.startswith(b"event: ") and data_line.startswith(b"data: ")
        decoded = decode_object(data_line[6:])
        assert event_line[7:].decode() == decoded["type"]
        result = [*result, decoded]
    return result


def test_sse_complete_lifecycle_matches_final_response_and_ids() -> None:
    prepared = prepare_request(request("hello"), MODEL, ())
    message = native_message(tool_call("lookup", "call-1"), text="Using a tool.")
    completion = convert_response(
        native_result(message, "tool_calls"), prepared, lease(), MODEL
    )
    rows = events(completion)
    assert [event["sequence_number"] for event in rows] == list(range(len(rows)))
    assert rows[0]["type"] == "response.created"
    assert rows[1]["type"] == "response.in_progress"
    assert rows[-1]["type"] == "response.completed"
    assert rows[-1]["response"] == decode_object(completion.response)
    assert [
        event["item"] for event in rows if event["type"] == "response.output_item.done"
    ] == output(completion)
    assert {event["type"] for event in rows} >= {
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
    }
    assert response_events(completion.response) == response_events(completion.response)


def test_length_finishes_as_incomplete_in_json_and_sse() -> None:
    prepared = prepare_request(request("hello"), MODEL, ())
    completion = convert_response(
        native_result(native_message(text="partial"), "length"),
        prepared,
        lease(),
        MODEL,
    )
    response = decode_object(completion.response)
    assert response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}
    assert events(completion)[-1]["type"] == "response.incomplete"
    assert output(completion)[1]["status"] == "incomplete"


def test_incomplete_tool_call_is_never_turned_into_completed_response() -> None:
    prepared = prepare_request(request("hello"), MODEL, ())
    with pytest.raises(AdapterError):
        convert_response(
            native_result(native_message(tool_call("lookup", "x")), "length"),
            prepared,
            lease(),
            MODEL,
        )
