"""Single-worker durable StateStore with plaintext JSON evidence and exact replay."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable

from codex_adapter.domain.models import RegisteredModel
from codex_adapter.domain.state import (
    Binding,
    BoundRoute,
    Completion,
    Replay,
    RequestLease,
    SavedTurn,
    StateSettings,
)

from .sqlite_bindings import SQLiteBindings, state_conflict
from .sqlite_database import SQLiteDatabase
from .sqlite_journal import SQLiteJournal
from .sqlite_turns import SQLiteTurns


class SQLiteStateStore:
    """Explicit lifecycle; one owning process, serialized threads, no blind retries."""

    def __init__(
        self,
        settings: StateSettings,
        *,
        clock: Callable[[], float] = time.time,
        draw: Callable[[int], int] = secrets.randbelow,
    ) -> None:
        self.settings = settings
        self._database = SQLiteDatabase(settings)
        self._bindings = SQLiteBindings(settings, clock, draw)
        self._turns = SQLiteTurns(settings, self._bindings)
        self._journal = SQLiteJournal(settings, self._bindings, self._turns)

    def initialize(self) -> None:
        """Explicitly create new storage; existing files are never overwritten."""
        self._database.initialize()

    def ready(self) -> None:
        """Validate storage and quarantine only reservations left by an old owner."""
        if self._database.ready():
            try:
                with self._database.transaction() as connection:
                    self._journal.recover(connection)
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        """Release the connection and exclusive worker-ownership lock."""
        self._database.close()

    def bind(
        self, session_id: str, model: RegisteredModel, *, principal: str = "local"
    ) -> BoundRoute:
        """Atomically choose once, or verify the exact persisted route reference."""
        if not isinstance(model, RegisteredModel):
            raise state_conflict()
        with self._database.read_transaction() as connection:
            existing = self._bindings.lookup_existing(
                connection, session_id, model, principal
            )
        if existing is not None:
            return existing
        with self._database.transaction() as connection:
            return self._bindings.bind(connection, session_id, model, principal)

    def turns(self, binding: Binding) -> tuple[SavedTurn, ...]:
        """Load contiguous turn history, never synthesize missing turns."""
        with self._database.read_transaction() as connection:
            persisted = self._bindings.validate(connection, binding)
            return self._turns.load(connection, persisted)

    def reserve(
        self,
        binding: Binding,
        request_digest: str,
        *,
        previous_carrier: str | None,
        idempotency_key: str | None = None,
    ) -> RequestLease | Replay:
        """Replay before codec restoration, or reserve exactly one session write."""
        with self._database.transaction() as connection:
            persisted = self._bindings.validate(connection, binding)
            return self._journal.reserve(
                connection, persisted, request_digest, previous_carrier, idempotency_key
            )

    def commit(self, lease: RequestLease, completion: Completion) -> None:
        """Persist response bytes, evidence, usage and head before publication."""
        self._check_lease_type(lease)
        with self._database.transaction() as connection:
            self._bindings.validate(connection, lease.binding)
            self._journal.commit(connection, lease, completion)

    def reject(self, lease: RequestLease) -> None:
        """Release a known-safe failure without deleting its idempotency identity."""
        self._check_lease_type(lease)
        with self._database.transaction() as connection:
            self._bindings.validate(connection, lease.binding)
            self._journal.transition(connection, lease, "rejected")

    def indeterminate(self, lease: RequestLease) -> None:
        """Quarantine an ambiguous upstream outcome; it cannot be auto-retried."""
        self._check_lease_type(lease)
        with self._database.transaction() as connection:
            self._bindings.validate(connection, lease.binding)
            self._journal.transition(connection, lease, "indeterminate")

    def _check_lease_type(self, lease: RequestLease) -> None:
        if not isinstance(lease, RequestLease):
            raise state_conflict()
