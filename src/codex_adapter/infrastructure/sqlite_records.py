"""Typed, validated plaintext JSON serialization at the SQLite boundary."""

from __future__ import annotations

import hashlib
import math
import sqlite3
from dataclasses import dataclass
from typing import Literal, cast

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import JsonObject, decode_object, encode_json
from codex_adapter.domain.state import Binding, Replay, RequestLease, SavedTurn

JournalStatus = Literal["inflight", "committed", "rejected", "indeterminate"]
_CARRIER_PREFIX = "codex_adapter.turn.v1."


def corrupt_state() -> AdapterError:
    """Return a safe failure without echoing SQL or private evidence."""
    return AdapterError(
        "state_corrupt",
        "Stored adapter state failed integrity validation",
        status_code=500,
    )


def credential_fingerprint(secret: str) -> str:
    """Persist a one-way credential identity, not its raw secret."""
    try:
        encoded = secret.encode("utf-8")
    except UnicodeError:
        raise AdapterError(
            "invalid_config", "Credential secret must be valid UTF-8"
        ) from None
    return hashlib.sha256(b"codex_adapter.credential.v1\x00" + encoded).hexdigest()


def binding_id(principal: str, session_id: str) -> str:
    """Deterministic lookup identity derived from principal and session_id."""
    identity = encode_json({"principal": principal, "session_id": session_id})
    return "bind_" + hashlib.sha256(identity).hexdigest()


def format_carrier(binding_identifier: str, sequence: int) -> str:
    """Build a plaintext turn reference for local session continuity."""
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise corrupt_state()
    return f"{_CARRIER_PREFIX}{binding_identifier}.{sequence}"


def verify_carrier(carrier: str, binding: Binding, *, sequence: int) -> None:
    """Verify that the turn carrier matches this binding and sequence."""
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 1
        or not isinstance(carrier, str)
        or carrier != format_carrier(binding.id, sequence)
    ):
        raise AdapterError(
            "state_conflict",
            "Adapter state reference does not match this session",
            status_code=409,
        )


@dataclass(frozen=True)
class Head:
    sequence: int = 0
    carrier: str | None = None
    pending_request_id: str | None = None
    pending_status: Literal["inflight", "indeterminate"] | None = None


@dataclass(frozen=True)
class JournalRecord:
    lease: RequestLease
    status: JournalStatus
    replay: Replay | None = None


def row_text(row: sqlite3.Row, field: str) -> str:
    value: object = row[field]
    if not isinstance(value, str) or not value:
        raise corrupt_state()
    return value


def row_payload(row: sqlite3.Row, field: str) -> bytes:
    value: object = row[field]
    if isinstance(value, str) and value:
        try:
            return value.encode("utf-8")
        except UnicodeError:
            raise corrupt_state() from None
    if isinstance(value, bytes) and value:
        return value
    raise corrupt_state()


def row_number(row: sqlite3.Row, field: str) -> float:
    value: object = row[field]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise corrupt_state()
    return float(value)


def row_int(row: sqlite3.Row, field: str) -> int:
    value: object = row[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise corrupt_state()
    return value


def snapshot(payload: bytes) -> JsonObject:
    try:
        return decode_object(payload)
    except AdapterError:
        raise corrupt_state() from None


def text(value: JsonObject, field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise corrupt_state()
    return result


def optional_text(value: JsonObject, field: str) -> str | None:
    result = value.get(field)
    if result is not None and (not isinstance(result, str) or not result):
        raise corrupt_state()
    return result


def integer(value: JsonObject, field: str) -> int:
    result = value.get(field)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise corrupt_state()
    return result


def number(value: JsonObject, field: str) -> float:
    result = value.get(field)
    if (
        isinstance(result, bool)
        or not isinstance(result, (int, float))
        or not math.isfinite(result)
    ):
        raise corrupt_state()
    return float(result)


def object_field(value: JsonObject, field: str) -> JsonObject:
    result = value.get(field)
    if not isinstance(result, dict):
        raise corrupt_state()
    return result


def utf8_text(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise corrupt_state()
    try:
        return value.decode("utf-8")
    except UnicodeError:
        raise corrupt_state() from None


def utf8_bytes(value: object) -> bytes:
    if not isinstance(value, str):
        raise corrupt_state()
    try:
        return value.encode("utf-8")
    except UnicodeError:
        raise corrupt_state() from None


def encode_head(head: Head) -> str:
    return encode_json(
        {
            "sequence": head.sequence,
            "carrier": head.carrier,
            "pending_request_id": head.pending_request_id,
            "pending_status": head.pending_status,
        }
    ).decode("utf-8")


def decode_head(payload: bytes) -> Head:
    value = snapshot(payload)
    if not {"sequence", "carrier", "pending_request_id", "pending_status"}.issubset(
        value
    ):
        raise corrupt_state()
    status = optional_text(value, "pending_status")
    if status not in {None, "inflight", "indeterminate"}:
        raise corrupt_state()
    head = Head(
        integer(value, "sequence"),
        optional_text(value, "carrier"),
        optional_text(value, "pending_request_id"),
        cast(Literal["inflight", "indeterminate"] | None, status),
    )
    if (head.sequence == 0) != (head.carrier is None) or (
        head.pending_request_id is None
    ) != (head.pending_status is None):
        raise corrupt_state()
    return head


def encode_turn(turn: SavedTurn) -> str:
    validate_turn(turn)
    return encode_json(
        {
            "carrier": turn.carrier,
            "native_message": utf8_text(turn.native_message),
            "projections": [utf8_text(item) for item in turn.projections],
            "call_ids": [[native, projected] for native, projected in turn.call_ids],
            "prefix_count": turn.prefix_count,
            "prefix_digest": turn.prefix_digest,
            "options_digest": turn.options_digest,
        }
    ).decode("utf-8")


def validate_turn(turn: SavedTurn) -> None:
    if not isinstance(turn, SavedTurn) or not isinstance(turn.native_message, bytes):
        raise corrupt_state()
    if not isinstance(turn.projections, tuple) or any(
        not isinstance(item, bytes) for item in turn.projections
    ):
        raise corrupt_state()
    if not isinstance(turn.call_ids, tuple) or any(
        not isinstance(pair, tuple)
        or len(pair) != 2
        or any(not isinstance(item, str) for item in pair)
        or not pair[0]
        for pair in turn.call_ids
    ):
        raise corrupt_state()
    if (
        isinstance(turn.prefix_count, bool)
        or not isinstance(turn.prefix_count, int)
        or turn.prefix_count < 0
    ):
        raise corrupt_state()
    if any(
        not isinstance(item, str) or not item
        for item in (turn.carrier, turn.prefix_digest, turn.options_digest)
    ):
        raise corrupt_state()


def decode_turn(payload: bytes) -> SavedTurn:
    value = snapshot(payload)
    projections = value.get("projections")
    calls = value.get("call_ids")
    if not isinstance(projections, list) or not isinstance(calls, list):
        raise corrupt_state()
    turn = SavedTurn(
        text(value, "carrier"),
        utf8_bytes(value.get("native_message")),
        tuple(utf8_bytes(item) for item in projections),
        tuple(_decode_call(item) for item in calls),
        integer(value, "prefix_count"),
        text(value, "prefix_digest"),
        text(value, "options_digest"),
    )
    validate_turn(turn)
    return turn


def _decode_call(value: object) -> tuple[str, str]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, str) for item in value)
        or not value[0]
    ):
        raise corrupt_state()
    return cast(str, value[0]), cast(str, value[1])


def lease_snapshot(lease: RequestLease) -> JsonObject:
    return {
        "request_id": lease.request_id,
        "request_digest": lease.request_digest,
        "idempotency_key": lease.idempotency_key,
        "carrier": lease.carrier,
        "response_id": lease.response_id,
        "sequence": lease.sequence,
        "previous_carrier": lease.previous_carrier,
        "created_at": lease.created_at,
    }


def _decode_lease(value: JsonObject, binding: Binding) -> RequestLease:
    lease = RequestLease(
        binding,
        text(value, "request_id"),
        text(value, "request_digest"),
        text(value, "idempotency_key"),
        text(value, "carrier"),
        text(value, "response_id"),
        integer(value, "sequence"),
        optional_text(value, "previous_carrier"),
        created_at=number(value, "created_at"),
    )
    if lease.sequence < 1:
        raise corrupt_state()
    return lease


def encode_journal(record: JournalRecord) -> str:
    replay = record.replay
    return encode_json(
        {
            "lease": lease_snapshot(record.lease),
            "status": record.status,
            "replay": (
                None
                if replay is None
                else {
                    "response": utf8_text(replay.response),
                    "raw_usage": (
                        None
                        if replay.raw_usage is None
                        else utf8_text(replay.raw_usage)
                    ),
                }
            ),
        }
    ).decode("utf-8")


def decode_journal(payload: bytes, binding: Binding) -> JournalRecord:
    value = snapshot(payload)
    status = text(value, "status")
    if status not in {"inflight", "committed", "rejected", "indeterminate"}:
        raise corrupt_state()
    lease = _decode_lease(object_field(value, "lease"), binding)
    replay_value = value.get("replay")
    if status != "committed":
        if replay_value is not None:
            raise corrupt_state()
        return JournalRecord(lease, cast(JournalStatus, status))
    if not isinstance(replay_value, dict):
        raise corrupt_state()
    raw_usage = replay_value.get("raw_usage")
    replay = Replay(
        utf8_bytes(replay_value.get("response")),
        None if raw_usage is None else utf8_bytes(raw_usage),
    )
    return JournalRecord(lease, "committed", replay)
