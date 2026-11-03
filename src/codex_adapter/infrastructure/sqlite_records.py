"""Typed, validated serialization at the untrusted SQLite/JSON boundary."""

from __future__ import annotations

import base64
import binascii
import math
import sqlite3
from dataclasses import dataclass
from typing import Literal, cast

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import JsonObject, decode_object, encode_json
from codex_adapter.domain.state import Binding, Replay, RequestLease, SavedTurn

from .state_crypto import corrupt_state

JournalStatus = Literal["inflight", "committed", "rejected", "indeterminate"]


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


def row_bytes(row: sqlite3.Row, field: str) -> bytes:
    value: object = row[field]
    if not isinstance(value, bytes):
        raise corrupt_state()
    return value


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


def binary(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise corrupt_state()
    return base64.b64encode(value).decode("ascii")


def unbinary(value: object) -> bytes:
    if not isinstance(value, str):
        raise corrupt_state()
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeError, binascii.Error):
        raise corrupt_state() from None


def encode_head(head: Head) -> bytes:
    return encode_json(
        {
            "sequence": head.sequence,
            "carrier": head.carrier,
            "pending_request_id": head.pending_request_id,
            "pending_status": head.pending_status,
        }
    )


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


def encode_turn(turn: SavedTurn) -> bytes:
    validate_turn(turn)
    return encode_json(
        {
            "carrier": turn.carrier,
            "native_message": binary(turn.native_message),
            "projections": [binary(item) for item in turn.projections],
            "call_ids": [[native, projected] for native, projected in turn.call_ids],
            "prefix_count": turn.prefix_count,
            "prefix_digest": turn.prefix_digest,
            "options_digest": turn.options_digest,
        }
    )


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
        unbinary(value.get("native_message")),
        tuple(unbinary(item) for item in projections),
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


def encode_journal(record: JournalRecord) -> bytes:
    replay = record.replay
    return encode_json(
        {
            "lease": lease_snapshot(record.lease),
            "status": record.status,
            "replay": (
                None
                if replay is None
                else {
                    "response": binary(replay.response),
                    "raw_usage": (
                        None if replay.raw_usage is None else binary(replay.raw_usage)
                    ),
                }
            ),
        }
    )


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
        unbinary(replay_value.get("response")),
        None if raw_usage is None else unbinary(raw_usage),
    )
    return JournalRecord(lease, "committed", replay)
