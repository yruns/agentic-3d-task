"""Regression tests for adapter foundation boundaries and secret safety."""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from traceback import format_exception
from typing import Any

import pytest

from codex_adapter.domain import models as model_module
from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import decode_object, encode_json
from codex_adapter.domain.models import (
    ModelCapabilities,
    RegisteredModel,
    Registry,
    Upstream,
)
from codex_adapter.domain.routing import choose_upstream
from codex_adapter.domain.state import (
    Binding,
    BoundRoute,
    Completion,
    PreparedGemini,
    Replay,
    RequestLease,
    SavedTurn,
    StateKey,
    StateSettings,
    ToolName,
)


def sample_model() -> RegisteredModel:
    return RegisteredModel(
        id="gpt-test",
        protocol="responses",
        upstreams=(
            Upstream(
                id="key",
                version="1",
                url="https://example.invalid/responses",
                secret="test-secret",
            ),
        ),
        capabilities=ModelCapabilities(),
    )


def sample_settings() -> StateSettings:
    return StateSettings(
        database=Path("state.db"),
        keys=(StateKey(id="key", secret=b"x" * 32),),
        active_key_id="key",
    )


def sample_binding() -> Binding:
    return Binding(
        id="sensitive-marker",
        principal="sensitive-marker",
        session_id="sensitive-marker",
        model_id="gpt-test",
        protocol="responses",
        codec_version="responses:1",
        endpoint="https://example.invalid/?token=sensitive-marker",
        credential_id="sensitive-marker",
        credential_version="sensitive-marker",
        credential_fingerprint="sensitive-marker",
        key_id="sensitive-marker",
        created_at=1.0,
        expires_at=2.0,
    )


def sample_turn() -> SavedTurn:
    return SavedTurn(
        carrier="sensitive-marker",
        native_message=b"sensitive-marker",
        projections=(b"sensitive-marker",),
        call_ids=(("sensitive-marker", "sensitive-marker"),),
        prefix_count=1,
        prefix_digest="sensitive-marker",
        options_digest="sensitive-marker",
    )


@pytest.mark.parametrize("weight", [True, -1, 1.5, "1", None, float("nan")])
def test_credential_weight_must_be_a_nonnegative_integer(weight: Any) -> None:
    with pytest.raises(AdapterError, match="nonnegative integer"):
        Upstream(
            id="key",
            version="1",
            url="https://example.invalid/responses",
            secret="test-secret",
            weight=weight,
        )


@pytest.mark.parametrize("slot", [True, -1, 1, 0.5, "0", None])
def test_selector_must_return_an_in_range_integer(slot: Any) -> None:
    with pytest.raises(AdapterError, match="invalid slot"):
        choose_upstream(sample_model(), draw=lambda total: slot)


def test_unknown_model_message_does_not_echo_client_input() -> None:
    with pytest.raises(AdapterError) as raised:
        Registry(models=(sample_model(),)).get("sensitive-marker")
    assert raised.value.code == "model_not_registered"
    assert raised.value.param == "model"
    assert "sensitive-marker" not in str(raised.value)
    assert "sensitive-marker" not in repr(raised.value.payload())


@pytest.mark.parametrize(
    "changes",
    [
        {"reasoning_efforts": ["low", "medium"]},
        {"reasoning_efforts": ("",)},
        {"reasoning_efforts": ("low", "low")},
        {"reasoning_efforts": ("low",), "default_reasoning_effort": "medium"},
        {"context_window": 0},
        {"context_window": -1},
        {"context_window": True},
        {"context_window": 1.5},
        {"default_max_output_tokens": 0},
        {"default_max_output_tokens": True},
        {"structured_output": "true"},
    ],
)
def test_invalid_capabilities_fail_at_construction(changes: dict[str, Any]) -> None:
    with pytest.raises(AdapterError):
        replace(ModelCapabilities(), **changes)


def test_capabilities_can_disable_reasoning_efforts() -> None:
    capabilities = ModelCapabilities(
        reasoning_efforts=(), default_reasoning_effort=None
    )
    assert capabilities.reasoning_efforts == ()


def test_registry_rejects_mutable_nested_collections() -> None:
    model = sample_model()
    with pytest.raises(AdapterError):
        replace(model, upstreams=list(model.upstreams))  # type: ignore[arg-type]  # Deliberately mutable boundary input.
    with pytest.raises(AdapterError):
        Registry(models=[model])  # type: ignore[arg-type]


def test_each_protocol_uses_its_own_codec_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_module,
        "_CODEC_VERSIONS",
        {"responses": "2", "gemini_chat": "1"},
        raising=False,
    )
    native = sample_model()
    gemini = replace(native, id="gemini-test", protocol="gemini_chat")
    assert native.codec_version == "responses:2"
    assert gemini.codec_version == "gemini_chat:1"


@pytest.mark.parametrize(
    "id_value,secret",
    [
        ("", b"x" * 32),
        (" ", b"x" * 32),
        ("key", b""),
        ("key", b"x" * 16),
        ("key", b"x" * 33),
        ("key", "not-bytes"),
        ("key", bytearray(b"x" * 32)),
    ],
)
def test_invalid_state_key_fails(id_value: Any, secret: Any) -> None:
    with pytest.raises(AdapterError):
        StateKey(id=id_value, secret=secret)


@pytest.mark.parametrize(
    "changes",
    [
        {"keys": ()},
        {"keys": (StateKey(id="key", secret=b"x" * 32),) * 2},
        {"keys": [StateKey(id="key", secret=b"x" * 32)]},
        {"active_key_id": "missing"},
        {"session_ttl_seconds": 0},
        {"session_ttl_seconds": -1},
        {"session_ttl_seconds": True},
        {"max_sessions": 0},
        {"max_turns": -1},
        {"max_state_bytes": 1.5},
    ],
)
def test_invalid_state_settings_fail_at_construction(changes: dict[str, Any]) -> None:
    with pytest.raises(AdapterError):
        replace(sample_settings(), **changes)


def test_state_and_codec_records_reject_mutable_collections() -> None:
    turn = sample_turn()
    with pytest.raises(AdapterError):
        replace(turn, projections=[b"message"])  # type: ignore[arg-type]  # Deliberately mutable boundary input.
    with pytest.raises(AdapterError):
        replace(turn, call_ids=(["call", "name"],))  # type: ignore[arg-type]  # Deliberately mutable boundary input.
    with pytest.raises(AdapterError):
        PreparedGemini(payload=b"{}", messages=[b"{}"], options_digest="digest")  # type: ignore[arg-type]
    with pytest.raises(AdapterError):
        PreparedGemini(payload=b"{}", messages=(), options_digest="digest", tool_names=[])  # type: ignore[arg-type]


def test_secret_and_payload_fields_are_hidden_from_nested_repr() -> None:
    marker = "sensitive-marker"
    original = sample_model()
    model = replace(
        original,
        upstreams=(
            replace(original.upstreams[0], url=f"https://example.invalid/{marker}"),
        ),
    )
    binding = sample_binding()
    turn = sample_turn()
    tool = ToolName(upstream_name=marker, name=marker, namespace=marker)
    prepared = PreparedGemini(
        payload=marker.encode(),
        messages=(marker.encode(),),
        options_digest=marker,
        tool_names=(tool,),
    )
    completion = Completion(
        response=marker.encode(), turn=turn, raw_usage=marker.encode()
    )
    lease = RequestLease(
        binding=binding,
        request_id=marker,
        request_digest=marker,
        idempotency_key=marker,
        carrier=marker,
        response_id=marker,
        sequence=1,
        previous_carrier=marker,
    )
    records = (
        StateKey(id="key", secret=marker.encode().ljust(32, b"x")),
        model,
        binding,
        BoundRoute(binding=binding, model=model, upstream=model.upstreams[0]),
        turn,
        tool,
        prepared,
        completion,
        lease,
        Replay(response=marker.encode(), raw_usage=marker.encode()),
    )
    for record in records:
        assert marker not in repr(record), type(record).__name__


def test_lease_created_at_is_backwards_compatible() -> None:
    lease = RequestLease(
        binding=sample_binding(),
        request_id="request",
        request_digest="digest",
        idempotency_key="idempotency",
        carrier="carrier",
        response_id="response",
        sequence=1,
        previous_carrier=None,
    )
    assert lease.created_at == 0.0
    assert replace(lease, created_at=42.0).created_at == 42.0


@pytest.mark.parametrize(
    "number",
    [
        "1e400",
        "-1e400",
        "1e-9999",
        "1.0000000000000001",
        "0.10000000000000001",
        "9007199254740993.0",
    ],
)
def test_lossy_json_floats_are_rejected(number: str) -> None:
    with pytest.raises(AdapterError, match="JSON"):
        decode_object(('{"value":' + number + "}").encode())


@pytest.mark.parametrize(
    "number",
    ["0.1", "-0.0", "5e-324", "1e300", "1.000000000", "123456789012345678901234567890"],
)
def test_json_numbers_that_round_trip_are_accepted(number: str) -> None:
    value = decode_object(('{"value":' + number + "}").encode())
    encoded = encode_json(value).decode()
    encoded_number = encoded.removeprefix('{"value":').removesuffix("}")
    assert Decimal(encoded_number) == Decimal(number)


@pytest.mark.parametrize(
    "payload", [b'{"x":"\\ud800"}', b'{"\\udfff":true}', b'{"x":["\\ud800"]}']
)
def test_json_rejects_unpaired_surrogates_in_keys_and_values(payload: bytes) -> None:
    with pytest.raises(AdapterError, match="JSON"):
        decode_object(payload)


def test_json_accepts_valid_surrogate_pair() -> None:
    result = decode_object(b'{"x":"\\ud83d\\ude00"}')
    assert decode_object(encode_json(result)) == result


@pytest.mark.parametrize(
    "payload", [{"x": float("nan")}, {"x": float("inf")}, {"x": "\ud800"}]
)
def test_internal_encoder_fails_with_safe_adapter_error(payload: Any) -> None:
    with pytest.raises(AdapterError, match="JSON") as raised:
        encode_json(payload)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_json_validation_traceback_does_not_expose_request_payload() -> None:
    marker = "sensitive-marker"
    payload = (
        '{"x":' + "[" * 300 + '{"token":"' + marker + '"}' + "]" * 300 + "}"
    ).encode()
    with pytest.raises(AdapterError) as raised:
        decode_object(payload)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert marker not in "".join(format_exception(raised.value))
