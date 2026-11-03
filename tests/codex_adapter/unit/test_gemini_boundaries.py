"""Additional malformed-input and replay-boundary regressions."""

from __future__ import annotations

from dataclasses import replace

import pytest
from test_gemini_chat import (
    MODEL,
    TOOL,
    lease,
    native_message,
    native_result,
    output,
    request,
    tool_call,
)

from codex_adapter.codecs.gemini_chat import convert_response, prepare_request
from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import JsonObject, JsonValue, decode_object


# These are valid JSON values but intentionally invalid content-part type fields.
@pytest.mark.parametrize("part_type", [[], {}, None, 1, True])
def test_malformed_content_type_fails_as_adapter_error(part_type: JsonValue) -> None:
    body = request([{"role": "user", "content": [{"type": part_type, "text": "x"}]}])
    with pytest.raises(AdapterError):
        prepare_request(body, MODEL, ())


@pytest.mark.parametrize(
    "image",
    [
        "https://example.invalid\\@localhost/image.png",
        "https://example.invalid/image path.png",
        "https://127.0.0.1/image.png",
        "https://localhost/image.png",
        "data:image/png;base64,=",
    ],
)
def test_invalid_image_authorities_are_rejected(image: str) -> None:
    body = request(
        [{"role": "user", "content": [{"type": "input_image", "image_url": image}]}]
    )
    with pytest.raises(AdapterError):
        prepare_request(body, MODEL, ())


@pytest.mark.parametrize("value", ["none", "auto", {}, {"type": "invalid"}])
def test_non_disabled_truncation_is_explicitly_unsupported(value: JsonValue) -> None:
    with pytest.raises(AdapterError):
        prepare_request(request("hello", truncation=value), MODEL, ())


def test_replaying_null_tool_calls_preserves_native_message() -> None:
    initial: list[JsonObject] = [{"role": "user", "content": "hello"}]
    prepared = prepare_request(request(initial), MODEL, ())
    message = {**native_message(text="hello"), "tool_calls": None}
    completion = convert_response(native_result(message), prepared, lease(), MODEL)
    replay = prepare_request(
        request([*initial, *output(completion)]), MODEL, (completion.turn,)
    )
    upstream = decode_object(replay.payload)
    messages = upstream["messages"]
    assert isinstance(messages, list) and messages[-1] == message


def test_incomplete_output_can_be_continued_without_marking_it_completed() -> None:
    initial: list[JsonObject] = [{"role": "user", "content": "hello"}]
    prepared = prepare_request(request(initial), MODEL, ())
    completion = convert_response(
        native_result(native_message(text="part"), "length"), prepared, lease(), MODEL
    )
    replay = prepare_request(
        request(
            [*initial, *output(completion), {"role": "user", "content": "continue"}]
        ),
        MODEL,
        (completion.turn,),
    )
    assert len(replay.messages) == 4


def test_replay_checks_optional_previous_tool_calls_against_native_snapshot() -> None:
    initial: list[JsonObject] = [{"role": "user", "content": "hello"}]
    prepared = prepare_request(request(initial), MODEL, ())
    completion = convert_response(
        native_result(native_message(tool_call("lookup", "x")), "tool_calls"),
        prepared,
        lease(),
        MODEL,
    )
    inputs: list[JsonObject] = [
        *initial,
        *output(completion),
        {"type": "function_call_output", "call_id": "x", "output": "ok"},
    ]
    forged = replace(completion.turn, call_ids=(("x", "other"),))
    with pytest.raises(AdapterError):
        prepare_request(request(inputs), MODEL, (forged,))


def test_namespace_identity_is_part_of_stable_options_digest() -> None:
    namespace: JsonObject = {"type": "namespace", "name": "records", "tools": [TOOL]}
    base = request("hello", tools=[namespace])
    original = prepare_request(base, MODEL, ())
    renamed = prepare_request(
        request("hello", tools=[{**namespace, "name": "other"}]), MODEL, ()
    )
    assert original.options_digest != renamed.options_digest
