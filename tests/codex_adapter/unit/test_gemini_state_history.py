"""Every previous native assistant and signature must survive a growing history."""

from __future__ import annotations

import pytest
from test_gemini_chat import MODEL, lease, native_message, native_result, object_list, output, request

from codex_adapter.codecs.gemini_chat import convert_response, last_carrier, prepare_request
from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import JsonObject, decode_object
from codex_adapter.domain.state import SavedTurn


def history(rounds: int) -> tuple[list[JsonObject], tuple[SavedTurn, ...], tuple[JsonObject, ...]]:
    inputs: list[JsonObject] = [{"role": "user", "content": "initial question"}]
    turns: tuple[SavedTurn, ...] = ()
    originals: tuple[JsonObject, ...] = ()
    for index in range(1, rounds + 1):
        prepared = prepare_request(request(inputs), MODEL, turns)
        restored = object_list(decode_object(prepared.payload)["messages"])
        assert tuple(item for item in restored if item["role"] == "assistant") == originals
        native: JsonObject = {
            **native_message(text=f"reply-{index}"),
            "signature": f"message-signature-{index}",
            "multimodal_contents": [{"type": "text", "text": f"part-{index}", "signature": f"part-signature-{index}"}],
        }
        completion = convert_response(native_result(native), prepared, lease(index), MODEL)
        inputs = [*inputs, *output(completion), {"role": "user", "content": f"followup-{index}"}]
        turns = (*turns, completion.turn)
        originals = (*originals, native)
    return inputs, turns, originals


@pytest.mark.parametrize("rounds", [1, 4, 12])
def test_all_previous_native_signatures_are_restored(rounds: int) -> None:
    inputs, turns, originals = history(rounds)
    prepared = prepare_request(request(inputs), MODEL, turns)
    messages = object_list(decode_object(prepared.payload)["messages"])
    restored = tuple(item for item in messages if item["role"] == "assistant")
    assert restored == originals
    assert len(restored) == rounds
    assert tuple(item["signature"] for item in restored) == tuple(f"message-signature-{index}" for index in range(1, rounds + 1))


def test_missing_old_carrier_is_rejected_even_when_latest_carrier_matches() -> None:
    inputs, turns, _ = history(4)
    missing = [item for item in inputs if item.get("encrypted_content") != turns[1].carrier]
    body = request(missing)
    assert last_carrier(body) == turns[-1].carrier
    with pytest.raises(AdapterError):
        prepare_request(body, MODEL, turns)


def test_reordered_old_carriers_are_rejected_even_when_latest_carrier_matches() -> None:
    inputs, turns, _ = history(4)
    substitutions = {turns[0].carrier: turns[1].carrier, turns[1].carrier: turns[0].carrier}
    reordered = [{**item, "encrypted_content": substitutions[str(item.get("encrypted_content"))]} if item.get("encrypted_content") in substitutions else item for item in inputs]
    body = request(reordered)
    assert last_carrier(body) == turns[-1].carrier
    with pytest.raises(AdapterError):
        prepare_request(body, MODEL, turns)
