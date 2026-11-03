"""Encrypted native evidence and authenticated contiguous turn chains."""

from __future__ import annotations

import sqlite3

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.state import Binding, RequestLease, SavedTurn, StateSettings

from .sqlite_bindings import SQLiteBindings, record_context, state_conflict
from .sqlite_records import Head, decode_turn, encode_turn, row_bytes, row_int, row_text
from .state_crypto import StateCrypto, corrupt_state


class SQLiteTurns:
    """Turn helpers assume the caller holds the database transaction lock."""

    def __init__(
        self, settings: StateSettings, crypto: StateCrypto, bindings: SQLiteBindings
    ) -> None:
        self.settings = settings
        self.crypto = crypto
        self.bindings = bindings

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
            turn = self._decode(row, binding, sequence, previous)
            turns = (*turns, turn)
            previous = turn.carrier
        if previous != head.carrier:
            raise corrupt_state()
        return turns

    def _decode(
        self, row: sqlite3.Row, binding: Binding, sequence: int, previous: str | None
    ) -> SavedTurn:
        try:
            plaintext = self.crypto.decrypt(
                row_bytes(row, "payload"),
                context=record_context(binding, "turn", sequence),
            )
            turn = decode_turn(plaintext)
            if turn.carrier != row_text(row, "carrier"):
                raise corrupt_state()
            self.crypto.verify_carrier(
                turn.carrier, binding, sequence=sequence, previous_carrier=previous
            )
            return turn
        except AdapterError as error:
            if error.code == "state_key_missing":
                raise
            raise corrupt_state() from None

    def append(
        self, connection: sqlite3.Connection, lease: RequestLease, turn: SavedTurn
    ) -> None:
        head = self.bindings.head(connection, lease.binding)
        if (
            lease.sequence != head.sequence + 1
            or lease.previous_carrier != head.carrier
            or turn.carrier != lease.carrier
        ):
            raise state_conflict()
        self.crypto.verify_carrier(
            lease.carrier,
            lease.binding,
            sequence=lease.sequence,
            previous_carrier=lease.previous_carrier,
        )
        payload = self.crypto.encrypt(
            encode_turn(turn),
            context=record_context(lease.binding, "turn", lease.sequence),
        )
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
                COALESCE((SELECT SUM(LENGTH(payload)) FROM turns WHERE binding_id = ?), 0)
                + COALESCE((SELECT SUM(LENGTH(payload)) FROM requests WHERE binding_id = ?), 0)
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
