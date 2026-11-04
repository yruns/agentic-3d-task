"""Plaintext native evidence, turn caching, and contiguous turn chains."""

from __future__ import annotations

import sqlite3
from collections import OrderedDict
from threading import RLock

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

_MAX_CACHED_TURNS = 4096


class SQLiteTurns:
    """Turn helpers assume the caller holds a read or write database transaction."""

    def __init__(self, settings: StateSettings, bindings: SQLiteBindings) -> None:
        self.settings = settings
        self.bindings = bindings
        self._cache_lock = RLock()
        self._turn_cache: OrderedDict[tuple[str, int, str], tuple[str, SavedTurn]] = (
            OrderedDict()
        )
        self._budget_cache: dict[str, int] = {}

    def invalidate_budget(self, binding_id: str | None = None) -> None:
        with self._cache_lock:
            if binding_id is None:
                self._budget_cache.clear()
            else:
                self._budget_cache.pop(binding_id, None)

    def _get_cached_turn(
        self, key: tuple[str, int, str], raw_payload: str
    ) -> SavedTurn | None:
        with self._cache_lock:
            entry = self._turn_cache.get(key)
            if entry is not None and entry[0] == raw_payload:
                self._turn_cache.move_to_end(key)
                return entry[1]
            return None

    def _put_cached_turn(
        self, key: tuple[str, int, str], raw_payload: str, turn: SavedTurn
    ) -> None:
        with self._cache_lock:
            self._turn_cache[key] = (raw_payload, turn)
            self._turn_cache.move_to_end(key)
            while len(self._turn_cache) > _MAX_CACHED_TURNS:
                self._turn_cache.popitem(last=False)

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
        turns: list[SavedTurn] = []
        for sequence, row in enumerate(rows, start=1):
            if row_int(row, "sequence") != sequence:
                raise corrupt_state()
            turn = self._decode(row, binding, sequence)
            turns.append(turn)
            previous = turn.carrier
        if previous != head.carrier:
            raise corrupt_state()
        return tuple(turns)

    def _decode(self, row: sqlite3.Row, binding: Binding, sequence: int) -> SavedTurn:
        try:
            carrier = row_text(row, "carrier")
            verify_carrier(carrier, binding, sequence=sequence)
            raw_value: object = row["payload"]
            cache_key = (binding.id, sequence, carrier)
            if isinstance(raw_value, str) and raw_value:
                cached = self._get_cached_turn(cache_key, raw_value)
                if cached is not None:
                    return cached
            turn = decode_turn(row_payload(row, "payload"))
            if turn.carrier != carrier:
                raise corrupt_state()
            if isinstance(raw_value, str) and raw_value:
                self._put_cached_turn(cache_key, raw_value, turn)
            return turn
        except AdapterError:
            raise corrupt_state() from None

    def append(
        self, connection: sqlite3.Connection, lease: RequestLease, turn: SavedTurn
    ) -> int:
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
        self._put_cached_turn(
            (lease.binding.id, lease.sequence, lease.carrier), payload, turn
        )
        return len(payload.encode("utf-8"))

    def check_budget(
        self,
        connection: sqlite3.Connection,
        binding: Binding,
        *,
        delta_bytes: int | None = None,
    ) -> None:
        with self._cache_lock:
            cached_total = self._budget_cache.get(binding.id)
        if cached_total is not None and delta_bytes is not None:
            size = cached_total + delta_bytes
        else:
            size = self._query_budget(connection, binding.id)
        if size > self.settings.max_state_bytes:
            self.invalidate_budget(binding.id)
            raise AdapterError(
                "state_quota",
                "Session state byte capacity has been reached",
                status_code=413,
            )
        with self._cache_lock:
            self._budget_cache[binding.id] = size

    def _query_budget(self, connection: sqlite3.Connection, binding_id: str) -> int:
        row = connection.execute(
            """SELECT
                COALESCE((SELECT SUM(LENGTH(CAST(payload AS BLOB))) FROM turns WHERE binding_id = ?), 0)
                + COALESCE((SELECT SUM(LENGTH(CAST(payload AS BLOB))) FROM requests WHERE binding_id = ?), 0)
                + COALESCE((SELECT SUM(LENGTH(key_hash) + LENGTH(request_id))
                    FROM idempotency_keys WHERE binding_id = ?), 0) AS size""",
            (binding_id, binding_id, binding_id),
        ).fetchone()
        if row is None:
            raise corrupt_state()
        return row_int(row, "size")
