"""AEAD records and versioned adapter-owned references, never provider ciphertext."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import secrets
from types import MappingProxyType

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import JsonObject, decode_object, encode_json
from codex_adapter.domain.state import Binding, StateSettings

_ENVELOPE_MAGIC = b"CAS1"
_NONCE_BYTES = 12
_TAG_BYTES = 16
_CARRIER_PREFIX = "codex_adapter.state.v2."
_CARRIER_MAGIC = b"CAR2"
_CARRIER_HINT_BYTES = 16
_MAX_CARRIER_BYTES = 16384
_MAX_KEY_ID_BYTES = 256
_KDF_SALT = b"codex_adapter.state.v1.key-derivation"


def corrupt_state() -> AdapterError:
    """Return a safe failure without echoing ciphertext, SQL, or private evidence."""
    return AdapterError(
        "state_corrupt",
        "Stored adapter state failed integrity validation",
        status_code=500,
    )


def binding_snapshot(binding: Binding) -> JsonObject:
    """Canonical, secret-free route identity authenticated by all record types."""
    return {
        "id": binding.id,
        "principal": binding.principal,
        "session_id": binding.session_id,
        "model_id": binding.model_id,
        "protocol": binding.protocol,
        "codec_version": binding.codec_version,
        "endpoint": binding.endpoint,
        "credential_id": binding.credential_id,
        "credential_version": binding.credential_version,
        "credential_fingerprint": binding.credential_fingerprint,
        "key_id": binding.key_id,
        "created_at": binding.created_at,
        "expires_at": binding.expires_at,
    }


def credential_fingerprint(secret: str) -> str:
    """Persist a one-way credential identity, not its secret or a callable token."""
    try:
        encoded = secret.encode("utf-8")
    except UnicodeError:
        raise AdapterError(
            "invalid_config", "Credential secret must be valid UTF-8"
        ) from None
    return hashlib.sha256(b"codex_adapter.credential.v1\x00" + encoded).hexdigest()


def _encode64(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _decode64(payload: str) -> bytes:
    try:
        encoded = payload.encode("ascii")
        decoded = base64.b64decode(
            encoded + b"=" * (-len(encoded) % 4), altchars=b"-_", validate=True
        )
    except (ValueError, UnicodeError, binascii.Error):
        raise corrupt_state() from None
    if _encode64(decoded) != payload:
        raise corrupt_state()
    return decoded


def _previous_digest(previous_carrier: str | None) -> str | None:
    if previous_carrier is None:
        return None
    return hashlib.sha256(previous_carrier.encode("utf-8")).hexdigest()


class StateCrypto:
    """Immutable keyring: read old keys, write active, and separate AEAD/MAC keys."""

    def __init__(self, settings: StateSettings) -> None:
        self._keys = MappingProxyType({key.id: key.secret for key in settings.keys})
        self.active_key_id = settings.active_key_id
        if self.active_key_id not in self._keys:
            raise AdapterError("invalid_config", "The active state key is unavailable")
        if any(
            not isinstance(secret, bytes) or len(secret) != 32
            for secret in self._keys.values()
        ):
            raise AdapterError(
                "invalid_config", "State keys must contain exactly 32 bytes"
            )
        for key_id in self._keys:
            self._validate_key_id(key_id)
        self._carrier_key_ids = MappingProxyType(
            {self._carrier_hint(key_id): key_id for key_id in self._keys}
        )
        if len(self._carrier_key_ids) != len(self._keys):
            raise AdapterError(
                "invalid_config", "State carrier key hints must be unique"
            )

    def _validate_key_id(self, key_id: str) -> None:
        try:
            encoded = key_id.encode("utf-8")
        except UnicodeError:
            raise AdapterError(
                "invalid_config", "Invalid state key identifier"
            ) from None
        if not encoded or len(encoded) > _MAX_KEY_ID_BYTES:
            raise AdapterError("invalid_config", "Invalid state key identifier")

    def _derive(self, key_id: str, purpose: bytes) -> bytes:
        secret = self._keys.get(key_id)
        if secret is None:
            raise AdapterError(
                "state_key_missing",
                "A required state key is unavailable",
                status_code=503,
            )
        return HKDF(
            algorithm=hashes.SHA256(), length=32, salt=_KDF_SALT, info=purpose
        ).derive(secret)

    def encrypt(
        self, plaintext: bytes, *, context: bytes, key_id: str | None = None
    ) -> bytes:
        """Seal bytes with AES-256-GCM and caller-supplied record identity as AAD."""
        if not isinstance(plaintext, bytes) or not isinstance(context, bytes):
            raise corrupt_state()
        selected = self.active_key_id if key_id is None else key_id
        encoded_id = selected.encode("utf-8")
        header = _ENVELOPE_MAGIC + len(encoded_id).to_bytes(2, "big") + encoded_id
        nonce = secrets.token_bytes(_NONCE_BYTES)
        cipher = AESGCM(self._derive(selected, b"record-aead"))
        return header + nonce + cipher.encrypt(nonce, plaintext, header + context)

    def decrypt(self, envelope: bytes, *, context: bytes) -> bytes:
        """Authenticate both the envelope header and record context before release."""
        key_id, header, nonce, ciphertext = self._unpack(envelope)
        try:
            return AESGCM(self._derive(key_id, b"record-aead")).decrypt(
                nonce, ciphertext, header + context
            )
        except (InvalidTag, ValueError, TypeError):
            raise corrupt_state() from None

    def _unpack(self, envelope: bytes) -> tuple[str, bytes, bytes, bytes]:
        if (
            not isinstance(envelope, bytes)
            or not envelope.startswith(_ENVELOPE_MAGIC)
            or len(envelope) < 6
        ):
            raise corrupt_state()
        key_size = int.from_bytes(envelope[4:6], "big")
        header_size = 6 + key_size
        if (
            not 0 < key_size <= _MAX_KEY_ID_BYTES
            or len(envelope) < header_size + _NONCE_BYTES + _TAG_BYTES
        ):
            raise corrupt_state()
        try:
            key_id = envelope[6:header_size].decode("utf-8")
        except UnicodeError:
            raise corrupt_state() from None
        return (
            key_id,
            envelope[:header_size],
            envelope[header_size : header_size + _NONCE_BYTES],
            envelope[header_size + _NONCE_BYTES :],
        )

    def binding_tag(self, binding: Binding) -> bytes:
        """Authenticate stored routing metadata with a dedicated MAC key."""
        return hmac.digest(
            self._derive(binding.key_id, b"binding-mac"),
            encode_json(binding_snapshot(binding)),
            "sha256",
        )

    def verify_binding(self, binding: Binding, tag: bytes) -> None:
        if not isinstance(tag, bytes) or not hmac.compare_digest(
            self.binding_tag(binding), tag
        ):
            raise corrupt_state()

    def binding_id(
        self, principal: str, session_id: str, *, key_id: str | None = None
    ) -> str:
        """Authenticate the lookup identity before consulting mutable SQL columns."""
        selected = self.active_key_id if key_id is None else key_id
        identity = encode_json({"principal": principal, "session_id": session_id})
        return (
            "bind_"
            + hmac.digest(
                self._derive(selected, b"session-lookup-v2"), identity, "sha256"
            ).hex()
        )

    def binding_ids(self, principal: str, session_id: str) -> tuple[str, ...]:
        return tuple(
            self.binding_id(principal, session_id, key_id=key_id)
            for key_id in self._keys
        )

    def _carrier_hint(self, key_id: str) -> bytes:
        return hmac.digest(
            self._derive(key_id, b"carrier-key-hint-v2"),
            key_id.encode("utf-8"),
            "sha256",
        )[:_CARRIER_HINT_BYTES]

    def carrier(
        self, binding: Binding, sequence: int, previous_carrier: str | None
    ) -> str:
        """Seal adapter-owned v2 claims; no route, credential, or key ID is public."""
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise corrupt_state()
        claims = self._carrier_claims(
            binding, sequence, previous_carrier, self.active_key_id
        )
        header = _CARRIER_MAGIC + self._carrier_hint(self.active_key_id)
        nonce = secrets.token_bytes(_NONCE_BYTES)
        cipher = AESGCM(self._derive(self.active_key_id, b"carrier-aead-v2"))
        encrypted = cipher.encrypt(
            nonce, encode_json(claims), _CARRIER_PREFIX.encode("ascii") + header
        )
        carrier = _CARRIER_PREFIX + _encode64(header + nonce + encrypted)
        if len(carrier) > _MAX_CARRIER_BYTES:
            raise corrupt_state()
        return carrier

    def _carrier_claims(
        self, binding: Binding, sequence: int, previous_carrier: str | None, key_id: str
    ) -> JsonObject:
        return {
            "version": 2,
            "binding": binding_snapshot(binding),
            "key": key_id,
            "sequence": sequence,
            "previous": _previous_digest(previous_carrier),
        }

    def verify_carrier(
        self,
        carrier: str,
        binding: Binding,
        *,
        sequence: int,
        previous_carrier: str | None,
    ) -> None:
        """Authenticate opaque claims against the exact route and predecessor."""
        try:
            claims, key_id = self._parse_carrier(carrier)
            if claims != self._carrier_claims(
                binding, sequence, previous_carrier, key_id
            ):
                raise corrupt_state()
        except AdapterError as error:
            if error.code == "state_key_missing":
                raise
            raise AdapterError(
                "state_conflict",
                "Adapter state reference does not match this session",
                status_code=409,
            ) from None

    def _parse_carrier(self, carrier: str) -> tuple[JsonObject, str]:
        if (
            not isinstance(carrier, str)
            or not carrier.startswith(_CARRIER_PREFIX)
            or len(carrier) > _MAX_CARRIER_BYTES
        ):
            raise corrupt_state()
        envelope = _decode64(carrier[len(_CARRIER_PREFIX) :])
        header_size = len(_CARRIER_MAGIC) + _CARRIER_HINT_BYTES
        if (
            not envelope.startswith(_CARRIER_MAGIC)
            or len(envelope) < header_size + _NONCE_BYTES + _TAG_BYTES
        ):
            raise corrupt_state()
        header = envelope[:header_size]
        key_id = self._carrier_key_ids.get(header[len(_CARRIER_MAGIC) :])
        if key_id is None:
            raise AdapterError(
                "state_key_missing",
                "A required state key is unavailable",
                status_code=503,
            )
        nonce = envelope[header_size : header_size + _NONCE_BYTES]
        try:
            cipher = AESGCM(self._derive(key_id, b"carrier-aead-v2"))
            plaintext = cipher.decrypt(
                nonce,
                envelope[header_size + _NONCE_BYTES :],
                _CARRIER_PREFIX.encode("ascii") + header,
            )
        except (InvalidTag, ValueError, TypeError):
            raise corrupt_state() from None
        return decode_object(plaintext), key_id
