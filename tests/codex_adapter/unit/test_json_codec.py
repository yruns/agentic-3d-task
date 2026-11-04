"""JSON boundaries must not silently reinterpret ambiguous wire requests."""

import pytest

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import JsonObject, decode_object, encode_json


@pytest.mark.parametrize(
    "payload", [b"null", b"[]", b'{"model":"a","model":"b"}', b'{"x":NaN}', b"not-json"]
)
def test_invalid_or_ambiguous_json_is_rejected(payload: bytes) -> None:
    with pytest.raises(AdapterError):
        decode_object(payload)


def test_nested_extension_fields_round_trip_without_mutation() -> None:
    payload: JsonObject = {
        "model": "gpt-test",
        "reasoning": {"encrypted_content": "opaque:+/="},
        "unknown": [None, True, 3],
    }
    encoded = encode_json(payload)
    assert decode_object(encoded) == payload
    assert payload["reasoning"] == {"encrypted_content": "opaque:+/="}
