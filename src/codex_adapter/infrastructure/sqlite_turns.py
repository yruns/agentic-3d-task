"""Plaintext native evidence and contiguous turn chains."""

from __future__ import annotations

import sqlite3

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.state import Binding, RequestLease, SavedTurn, StateSettings

from .sqlite_bindings import SQLiteBindings, state_conflict
from .sqlite_records import (
    Head,
    corrupt_state,
    decode_turn,
    encode_turn,
    row_int,
    row_payload,
    row_text,
    verify_carrier,
)


class SQLiteTurns:
    """Turn helpers assume the caller holds the database transaction lock."""

    def __init__(self, settings: StateSettings, bindings: SQLiteBindings) -> None:
        self.settings = settings
        self.bindings = bindings

    def verify_head(
        self, connection: sqlite3.Connection, binding: Binding, head: Head
    ) -> None:
        """Verify turn chain continuity via indexed metadata without loading payloads."""
        row = connection.execute(
            """SELECT COUNT(*) AS count, COALESCE(MAX(sequence), 0) AS max_seq
               FROM turns WHERE binding_id = ?""",
            (binding.id,),
        ).fetchone()
        if (
            row is None
            or row_int(row, "count") != head.sequence
            or row_int(row, "max_seq") != head.sequence
        ):
            raise corrupt_state()
        if head.sequence == 0:
            if head.carrier is not None:
                raise corrupt_state()
            return
        tail = connection.execute(
            "SELECT carrier FROM turns WHERE binding_id = ? AND sequence = ?",
            (binding.id, head.sequence),
        ).fetchone()
        if tail is None or row_text(tail, "carrier") != head.carrier:
            raise corrupt_state()

    def load(
        self, connection: sqlite3.Connection, binding: Binding
    ) -> tuple[SavedTurn, ...]:
        head = self.bindings.head(connection, binding)
        rows = connection.execute(
            "SELECT sequence, carrier, payload FROM turns WHERE binding_id = ? ORDER BY sequence",
            (binding.id,),
        ).fetchall()
        if len(rows) != head.sequence:
            raise corrupt_state()
        previous: str | None = None
        turns: tuple[SavedTurn, ...] = ()
        for sequence, row in enumerate(rows, start=1):
            if row_int(row, "sequence") != sequence:
                raise corrupt_state()
            turn = self._decode(row, binding, sequence)
            turns = (*turns, turn)
            previous = turn.carrier
        if previous != head.carrier:
            raise corrupt_state()
        return turns

    def _decode(self, row: sqlite3.Row, binding: Binding, sequence: int) -> SavedTurn:
        try:
            turn = decode_turn(row_payload(row, "payload"))
            if turn.carrier != row_text(row, "carrier"):
                raise corrupt_state()
            verify_carrier(turn.carrier, binding, sequence=sequence)
            return turn
        except AdapterError:
            raise corrupt_state() from None

    def append(
        self, connection: sqlite3.Connection, lease: RequestLease, turn: SavedTurn
    ) -> None:
        head = self.bindings.head(connection, lease.binding)
        self.verify_head(connection, lease.binding, head)
        if (
            lease.sequence != head.sequence + 1
            or lease.previous_carrier != head.carrier
            or turn.carrier != lease.carrier
        ):
            raise state_conflict()
        verify_carrier(lease.carrier, lease.binding, sequence=lease.sequence)
        payload = encode_turn(turn)
        connection.execute(
            "INSERT INTO turns (binding_id, sequence, carrier, payload) VALUES (?, ?, ?, ?)",
            (lease.binding.id, lease.sequence, lease.carrier, payload),
        )
        self.bindings.advance(
            connection, lease.binding, Head(lease.sequence, lease.carrier)
        )

    def check_budget(self, connection: sqlite3.Connection, binding: Binding) -> None:
        row = connection.execute(
            """SELECT
                COALESCE((SELECT SUM(LENGTH(CAST(payload AS BLOB))) FROM turns WHERE binding_id = ?), 0)
                + COALESCE((SELECT SUM(LENGTH(CAST(payload AS BLOB))) FROM requests WHERE binding_id = ?), 0)
                + COALESCE((SELECT SUM(LENGTH(key_hash) + LENGTH(request_id))
                    FROM idempotency_keys WHERE binding_id = ?), 0) AS size""",
            (binding.id, binding.id, binding.id),
        ).fetchone()
        if row is None or row_int(row, "size") > self.settings.max_state_bytes:
            raise AdapterError(
                "state_quota",
                "Session state byte capacity has been reached",
                status_code=413,
            )
