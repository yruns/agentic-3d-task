"""Every previous native assistant and signature must survive a growing history."""

from __future__ import annotations

import pytest
from test_gemini_chat import (
    MODEL,
    lease,
    native_message,
    native_result,
    object_list,
    output,
    request,
    tool_call,
)

from codex_adapter.codecs.gemini_chat import (
    convert_response,
    last_carrier,
    prepare_request,
)
from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import JsonObject, decode_object
from codex_adapter.domain.state import SavedTurn


def native_assistant(index: int, *, tools: bool) -> JsonObject:
    calls: tuple[JsonObject, ...] = (
        (
            {
                **tool_call("lookup", f"native-{index}"),
                "signature": f"call-signature-{index}",
            },
        )
        if tools
        else ()
    )
    return {
        **native_message(*calls, text=f"reply-{index}"),
        "signature": f"message-signature-{index}",
        "multimodal_contents": [
            {
                "type": "text",
                "text": f"part-{index}",
                "signature": f"part-signature-{index}",
            }
        ],
    }


def history(
    rounds: int, *, tools: bool = False
) -> tuple[list[JsonObject], tuple[SavedTurn, ...], tuple[JsonObject, ...]]:
    inputs: list[JsonObject] = [{"role": "user", "content": "initial question"}]
    turns: tuple[SavedTurn, ...] = ()
    originals: tuple[JsonObject, ...] = ()
    for index in range(1, rounds + 1):
        prepared = prepare_request(request(inputs), MODEL, turns)
        restored = object_list(decode_object(prepared.payload)["messages"])
        assert (
            tuple(item for item in restored if item["role"] == "assistant") == originals
        )
        native = native_assistant(index, tools=tools)
        completion = convert_response(
            native_result(native, "tool_calls" if tools else "stop"),
            prepared,
            lease(index),
            MODEL,
        )
        projection = output(completion)
        results: list[JsonObject] = [
            {
                "type": "function_call_output",
                "call_id": item["call_id"],
                "output": f"tool-result-{index}",
            }
            for item in projection
            if item["type"] == "function_call"
        ]
        users: list[JsonObject] = (
            [] if tools else [{"role": "user", "content": f"followup-{index}"}]
        )
        inputs = [*inputs, *projection, *results, *users]
        turns = (*turns, completion.turn)
        originals = (*originals, native)
    return inputs, turns, originals


@pytest.mark.parametrize("rounds", [1, 4, 12])
@pytest.mark.parametrize("tools", [False, True])
def test_all_previous_native_signatures_are_restored(rounds: int, tools: bool) -> None:
    inputs, turns, originals = history(rounds, tools=tools)
    prepared = prepare_request(request(inputs), MODEL, turns)
    messages = object_list(decode_object(prepared.payload)["messages"])
    restored = tuple(item for item in messages if item["role"] == "assistant")
    assert restored == originals
    assert len(restored) == rounds
    assert tuple(item["signature"] for item in restored) == tuple(
        f"message-signature-{index}" for index in range(1, rounds + 1)
    )
    if tools:
        assert tuple(
            item["tool_call_id"] for item in messages if item["role"] == "tool"
        ) == tuple(f"native-{index}" for index in range(1, rounds + 1))


def test_missing_old_carrier_is_rejected_even_when_latest_carrier_matches() -> None:
    inputs, turns, _ = history(4)
    first = next(
        index
        for index, item in enumerate(inputs)
        if item.get("encrypted_content") == turns[1].carrier
    )
    after_projection = first + 1 + len(turns[1].projections)
    body = request([*inputs[:first], *inputs[after_projection:]])
    assert last_carrier(body) == turns[-1].carrier
    with pytest.raises(AdapterError) as raised:
        prepare_request(body, MODEL, turns)
    assert raised.value.code == "state_conflict"
    assert raised.value.status_code == 409


def test_reordered_old_carriers_are_rejected_even_when_latest_carrier_matches() -> None:
    inputs, turns, _ = history(4)
    substitutions = {
        turns[0].carrier: turns[1].carrier,
        turns[1].carrier: turns[0].carrier,
    }
    reordered = [
        (
            {
                **item,
                "encrypted_content": substitutions[str(item.get("encrypted_content"))],
            }
            if item.get("encrypted_content") in substitutions
            else item
        )
        for item in inputs
    ]
    body = request(reordered)
    assert last_carrier(body) == turns[-1].carrier
    with pytest.raises(AdapterError) as raised:
        prepare_request(body, MODEL, turns)
    assert raised.value.code == "state_conflict"
    assert raised.value.status_code == 409
