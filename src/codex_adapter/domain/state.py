"""Immutable records shared by codecs, persistence and request orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .errors import AdapterError
from .models import Credential, RegisteredModel

_STATE_KEY_BYTES = 32


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _bytes_tuple(value: object) -> bool:
    return isinstance(value, tuple) and all(isinstance(item, bytes) for item in value)


def _optional_bytes(value: object) -> bool:
    return value is None or isinstance(value, bytes)


@dataclass(frozen=True)
class StateKey:
    id: str
    secret: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise AdapterError("invalid_config", "State key id must be nonempty")
        if not isinstance(self.secret, bytes) or len(self.secret) != _STATE_KEY_BYTES:
            raise AdapterError(
                "invalid_config", "State key must contain exactly 32 bytes"
            )


@dataclass(frozen=True)
class StateSettings:
    database: Path
    keys: tuple[StateKey, ...]
    active_key_id: str
    session_ttl_seconds: int = 86400
    max_sessions: int = 10000
    max_turns: int = 256
    max_state_bytes: int = 33554432

    def __post_init__(self) -> None:
        if (
            not isinstance(self.keys, tuple)
            or not self.keys
            or not all(isinstance(key, StateKey) for key in self.keys)
        ):
            raise AdapterError(
                "invalid_config", "State keys must be a nonempty immutable tuple"
            )
        ids = tuple(key.id for key in self.keys)
        if len(set(ids)) != len(ids):
            raise AdapterError("invalid_config", "State key ids must be unique")
        if self.active_key_id not in ids:
            raise AdapterError("invalid_config", "Active state key must be configured")
        limits = (
            self.session_ttl_seconds,
            self.max_sessions,
            self.max_turns,
            self.max_state_bytes,
        )
        if not all(_positive_integer(value) for value in limits):
            raise AdapterError(
                "invalid_config", "State limits must be positive integers"
            )


@dataclass(frozen=True)
class Binding:
    id: str = field(repr=False)
    principal: str = field(repr=False)
    session_id: str = field(repr=False)
    model_id: str = field(repr=False)
    protocol: str
    codec_version: str
    endpoint: str = field(repr=False)
    credential_id: str = field(repr=False)
    credential_version: str = field(repr=False)
    credential_fingerprint: str = field(repr=False)
    key_id: str = field(repr=False)
    created_at: float
    expires_at: float


@dataclass(frozen=True)
class BoundRoute:
    binding: Binding
    model: RegisteredModel
    credential: Credential


@dataclass(frozen=True)
class SavedTurn:
    carrier: str = field(repr=False)
    native_message: bytes = field(repr=False)
    projections: tuple[bytes, ...] = field(repr=False)
    call_ids: tuple[tuple[str, str], ...] = field(repr=False)
    prefix_count: int
    prefix_digest: str = field(repr=False)
    options_digest: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.native_message, bytes) or not _bytes_tuple(
            self.projections
        ):
            raise AdapterError(
                "invalid_state",
                "Turn messages must be immutable bytes",
                status_code=500,
            )
        if not isinstance(self.call_ids, tuple) or not all(
            isinstance(pair, tuple)
            and len(pair) == 2
            and all(isinstance(value, str) for value in pair)
            for pair in self.call_ids
        ):
            raise AdapterError(
                "invalid_state",
                "Turn call ids must be immutable pairs",
                status_code=500,
            )


@dataclass(frozen=True)
class ToolName:
    """Stable native name and its Codex function/namespace projection."""

    upstream_name: str = field(repr=False)
    name: str = field(repr=False)
    namespace: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.upstream_name, str) or not isinstance(self.name, str):
            raise AdapterError(
                "invalid_state", "Tool names must be strings", status_code=500
            )
        if self.namespace is not None and not isinstance(self.namespace, str):
            raise AdapterError(
                "invalid_state", "Tool namespace must be a string", status_code=500
            )


@dataclass(frozen=True)
class PreparedGemini:
    payload: bytes = field(repr=False)
    messages: tuple[bytes, ...] = field(repr=False)
    options_digest: str = field(repr=False)
    tool_names: tuple[ToolName, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes) or not _bytes_tuple(self.messages):
            raise AdapterError(
                "invalid_state",
                "Prepared messages must be immutable bytes",
                status_code=500,
            )
        if not isinstance(self.tool_names, tuple) or not all(
            isinstance(name, ToolName) for name in self.tool_names
        ):
            raise AdapterError(
                "invalid_state",
                "Prepared tool names must be immutable",
                status_code=500,
            )


@dataclass(frozen=True)
class Completion:
    response: bytes = field(repr=False)
    turn: SavedTurn
    raw_usage: bytes | None = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.response, bytes) or not _optional_bytes(self.raw_usage):
            raise AdapterError(
                "invalid_state",
                "Completion data must be immutable bytes",
                status_code=500,
            )
        if not isinstance(self.turn, SavedTurn):
            raise AdapterError(
                "invalid_state", "Completion requires a saved turn", status_code=500
            )


@dataclass(frozen=True)
class RequestLease:
    binding: Binding
    request_id: str = field(repr=False)
    request_digest: str = field(repr=False)
    idempotency_key: str = field(repr=False)
    carrier: str = field(repr=False)
    response_id: str = field(repr=False)
    sequence: int
    previous_carrier: str | None = field(repr=False)
    created_at: float = 0.0


@dataclass(frozen=True)
class Replay:
    response: bytes = field(repr=False)
    raw_usage: bytes | None = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.response, bytes) or not _optional_bytes(self.raw_usage):
            raise AdapterError(
                "invalid_state", "Replay data must be immutable bytes", status_code=500
            )
