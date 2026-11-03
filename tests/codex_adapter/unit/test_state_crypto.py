"""AEAD envelopes and adapter-owned, authenticated state references."""

import base64
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

import pytest

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.state import Binding, StateKey, StateSettings
from codex_adapter.infrastructure.state_crypto import StateCrypto


class BindingChanges(TypedDict, total=False):
    id: str
    session_id: str
    principal: str
    model_id: str
    codec_version: str
    credential_id: str
    credential_version: str
    endpoint: str


def settings() -> StateSettings:
    return StateSettings(Path("unused.sqlite"), (StateKey("old", b"o" * 32),), "old")


def binding() -> Binding:
    return Binding(
        "binding-test",
        "local",
        "session-test",
        "gemini-test",
        "gemini_chat",
        "gemini_chat:1",
        "https://upstream.example/chat/completions",
        "credential",
        "version-1",
        "fingerprint",
        "old",
        100.0,
        200.0,
    )


def test_aead_is_randomized_and_round_trips_binary_without_mutating_input() -> None:
    crypto = StateCrypto(settings())
    plaintext = b"private-native-message\x00\xff"
    first = crypto.encrypt(plaintext, context=b"native-message")
    second = crypto.encrypt(plaintext, context=b"native-message")
    assert first != second
    assert plaintext not in first
    assert plaintext == b"private-native-message\x00\xff"
    assert crypto.decrypt(first, context=b"native-message") == plaintext


def test_ciphertext_cannot_be_moved_to_a_different_record_context() -> None:
    crypto = StateCrypto(settings())
    encrypted = crypto.encrypt(b"private", context=b"session-one")
    with pytest.raises(AdapterError) as raised:
        crypto.decrypt(encrypted, context=b"session-two")
    assert raised.value.code == "state_corrupt"
    assert "private" not in str(raised.value)


@pytest.mark.parametrize("ciphertext", [b"", b"malformed", b"CAS1\x00\x01x"])
def test_malformed_ciphertext_is_a_safe_domain_error(ciphertext: bytes) -> None:
    with pytest.raises(AdapterError) as raised:
        StateCrypto(settings()).decrypt(ciphertext, context=b"context")
    assert raised.value.code == "state_corrupt"


def test_modified_ciphertext_fails_authentication() -> None:
    crypto = StateCrypto(settings())
    envelope = crypto.encrypt(b"private body", context=b"record")
    modified = envelope[:-1] + bytes([envelope[-1] ^ 1])
    with pytest.raises(AdapterError) as raised:
        crypto.decrypt(modified, context=b"record")
    assert raised.value.code == "state_corrupt"
    assert "private body" not in str(raised.value)


def test_keyring_reads_old_envelopes_and_writes_with_the_active_key() -> None:
    old = StateCrypto(settings())
    rotated_settings = replace(
        settings(),
        keys=(*settings().keys, StateKey("new", b"n" * 32)),
        active_key_id="new",
    )
    rotated = StateCrypto(rotated_settings)
    old_envelope = old.encrypt(b"old-state", context=b"record")
    assert rotated.decrypt(old_envelope, context=b"record") == b"old-state"
    new_envelope = rotated.encrypt(b"new-state", context=b"record")
    assert rotated.decrypt(new_envelope, context=b"record") == b"new-state"
    with pytest.raises(AdapterError) as raised:
        old.decrypt(new_envelope, context=b"record")
    assert raised.value.code == "state_key_missing"


def test_replaced_key_material_is_not_accepted_for_existing_ciphertext() -> None:
    envelope = StateCrypto(settings()).encrypt(b"body", context=b"record")
    replaced = replace(settings(), keys=(StateKey("old", b"x" * 32),))
    with pytest.raises(AdapterError):
        StateCrypto(replaced).decrypt(envelope, context=b"record")


def test_carriers_are_explicitly_adapter_owned_and_verify_the_chain() -> None:
    crypto = StateCrypto(settings())
    first = crypto.carrier(binding(), 1, None)
    second = crypto.carrier(binding(), 2, first)
    assert first.startswith("codex_adapter.state.v2.")
    assert len(second) < len(first) + 100
    crypto.verify_carrier(first, binding(), sequence=1, previous_carrier=None)
    crypto.verify_carrier(second, binding(), sequence=2, previous_carrier=first)
    with pytest.raises(AdapterError):
        crypto.verify_carrier(second, binding(), sequence=2, previous_carrier=None)


@pytest.mark.parametrize(
    "changed",
    [
        {"id": "different-binding"},
        {"session_id": "different-session"},
        {"principal": "another-user"},
        {"model_id": "different-model"},
        {"codec_version": "gemini_chat:2"},
        {"credential_id": "another-key"},
        {"credential_version": "version-2"},
        {"endpoint": "https://other.example"},
    ],
)
def test_carrier_is_bound_to_the_exact_route(changed: BindingChanges) -> None:
    crypto = StateCrypto(settings())
    carrier = crypto.carrier(binding(), 1, None)
    with pytest.raises(AdapterError):
        crypto.verify_carrier(
            carrier, replace(binding(), **changed), sequence=1, previous_carrier=None
        )


@pytest.mark.parametrize(
    "carrier", ["", "opaque-openai-token", "codex_adapter.state.v1.x.y"]
)
def test_unrecognized_carriers_fail_closed(carrier: str) -> None:
    with pytest.raises(AdapterError):
        StateCrypto(settings()).verify_carrier(
            carrier, binding(), sequence=1, previous_carrier=None
        )


def test_carrier_tampering_and_wrong_sequence_are_rejected() -> None:
    crypto = StateCrypto(settings())
    carrier = crypto.carrier(binding(), 1, None)
    modified = carrier[:-3] + ("AAA" if carrier[-3:] != "AAA" else "BBB")
    with pytest.raises(AdapterError):
        crypto.verify_carrier(modified, binding(), sequence=1, previous_carrier=None)
    with pytest.raises(AdapterError):
        crypto.verify_carrier(carrier, binding(), sequence=2, previous_carrier=None)


def test_binding_metadata_has_a_separate_authenticator() -> None:
    crypto = StateCrypto(settings())
    tag = crypto.binding_tag(binding())
    crypto.verify_binding(binding(), tag)
    with pytest.raises(AdapterError):
        crypto.verify_binding(replace(binding(), expires_at=300.0), tag)
    with pytest.raises(AdapterError):
        crypto.verify_binding(binding(), b"invalid-tag")


def test_carrier_base64_never_exposes_private_route_or_key_identity() -> None:
    key_id = "private-state-key-identity"
    configured = replace(
        settings(), keys=(StateKey(key_id, b"s" * 32),), active_key_id=key_id
    )
    private = replace(
        binding(),
        id="private-binding-identity",
        principal="private-principal-identity",
        session_id="private-session-identity",
        model_id="private-model-identity",
        endpoint="https://private-endpoint.example/chat/completions",
        credential_id="private-credential-identity",
        credential_version="private-credential-version",
        credential_fingerprint="f" * 64,
        key_id=key_id,
    )
    crypto = StateCrypto(configured)
    carrier = crypto.carrier(private, 1, None)
    payload = carrier.split(".", 3)[3].split(".", 1)[0]
    decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
    for value in (
        private.id,
        private.principal,
        private.session_id,
        private.model_id,
        private.endpoint,
        private.credential_id,
        private.credential_version,
        private.credential_fingerprint,
        key_id,
    ):
        assert value.encode() not in decoded
    crypto.verify_carrier(carrier, private, sequence=1, previous_carrier=None)


def test_sealed_carriers_are_randomized_and_support_keyring_rotation() -> None:
    original = StateCrypto(settings())
    first = original.carrier(binding(), 1, None)
    assert original.carrier(binding(), 1, None) != first
    rotated_settings = replace(
        settings(),
        keys=(*settings().keys, StateKey("new", b"n" * 32)),
        active_key_id="new",
    )
    rotated = StateCrypto(rotated_settings)
    rotated.verify_carrier(first, binding(), sequence=1, previous_carrier=None)
    second = rotated.carrier(binding(), 2, first)
    rotated.verify_carrier(second, binding(), sequence=2, previous_carrier=first)
    without_old = StateCrypto(
        replace(rotated_settings, keys=(rotated_settings.keys[1],))
    )
    with pytest.raises(AdapterError) as raised:
        without_old.verify_carrier(first, binding(), sequence=1, previous_carrier=None)
    assert raised.value.code == "state_key_missing"
