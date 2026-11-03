"""Durable at-most-one-inflight journal; ambiguous upstream outcomes quarantine."""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass, replace
from typing import Literal

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import encode_json
from codex_adapter.domain.state import (
    Binding,
    Completion,
    Replay,
    RequestLease,
    StateSettings,
)

from .sqlite_bindings import (
    SQLiteBindings,
    record_context,
    state_conflict,
    validate_identity,
)
from .sqlite_records import (
    Head,
    JournalRecord,
    decode_journal,
    encode_journal,
    row_bytes,
    row_text,
)
from .sqlite_turns import SQLiteTurns
from .state_crypto import StateCrypto, corrupt_state


@dataclass(frozen=True)
class RequestIdentity:
    digest: str
    previous_carrier: str | None
    idempotency_key: str
    request_hash: str
    key_hash: str


def request_hash(binding: Binding, digest: str, previous: str | None) -> str:
    return hashlib.sha256(
        encode_json(
            {
                "scope": "codex_adapter.request.v1",
                "binding": binding.id,
                "digest": digest,
                "previous": previous,
            }
        )
    ).hexdigest()


def key_hash(binding: Binding, key: str) -> str:
    return hashlib.sha256(
        encode_json(
            {"scope": "codex_adapter.idempotency.v1", "binding": binding.id, "key": key}
        )
    ).hexdigest()


def request_identity(
    binding: Binding, digest: str, previous: str | None, idem: str | None
) -> RequestIdentity:
    validate_identity(digest, field="request_digest")
    if previous is not None and (
        not isinstance(previous, str) or not previous or len(previous) > 16384
    ):
        raise state_conflict()
    fingerprint = request_hash(binding, digest, previous)
    selected = "derived:" + fingerprint if idem is None else idem
    validate_identity(selected, field="idempotency_key")
    return RequestIdentity(
        digest, previous, selected, fingerprint, key_hash(binding, selected)
    )


class SQLiteJournal:
    """Encrypted reservations and exact response replay inside SQLite transactions."""

    def __init__(
        self,
        settings: StateSettings,
        crypto: StateCrypto,
        bindings: SQLiteBindings,
        turns: SQLiteTurns,
    ) -> None:
        self.settings = settings
        self.crypto = crypto
        self.bindings = bindings
        self.turns = turns

    def reserve(
        self,
        connection: sqlite3.Connection,
        binding: Binding,
        digest: str,
        previous: str | None,
        idem: str | None,
    ) -> RequestLease | Replay:
        identity = request_identity(binding, digest, previous, idem)
        self.turns.load(connection, binding)
        pending = self._pending(connection, binding)
        record = self._existing(connection, binding, identity)
        if record is not None and record.status == "committed":
            if record.replay is None:
                raise corrupt_state()
            if self._alias(connection, binding, identity, record.lease.request_id):
                self.turns.check_budget(connection, binding)
            return record.replay
        if pending is not None:
            raise AdapterError(
                "request_in_progress",
                "Session has an unfinished or indeterminate request",
                status_code=409,
            )
        return self._reserve_lease(connection, binding, identity, record)

    def _existing(
        self,
        connection: sqlite3.Connection,
        binding: Binding,
        identity: RequestIdentity,
    ) -> JournalRecord | None:
        alias = connection.execute(
            "SELECT request_id FROM idempotency_keys WHERE binding_id = ? AND key_hash = ?",
            (binding.id, identity.key_hash),
        ).fetchone()
        if alias is None:
            row = connection.execute(
                "SELECT * FROM requests WHERE binding_id = ? AND request_hash = ?",
                (binding.id, identity.request_hash),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM requests WHERE request_id = ? AND binding_id = ?",
                (row_text(alias, "request_id"), binding.id),
            ).fetchone()
            if row is None:
                raise corrupt_state()
        if row is None:
            return None
        record = self._decode(row, binding)
        if (
            record.lease.request_digest != identity.digest
            or record.lease.previous_carrier != identity.previous_carrier
        ):
            raise state_conflict()
        return record

    def _reserve_lease(
        self,
        connection: sqlite3.Connection,
        binding: Binding,
        identity: RequestIdentity,
        record: JournalRecord | None,
    ) -> RequestLease:
        head = self.bindings.head(connection, binding)
        if head.carrier != identity.previous_carrier:
            raise state_conflict()
        if head.sequence >= self.settings.max_turns:
            raise AdapterError(
                "turn_quota", "Session turn capacity has been reached", status_code=413
            )
        lease = (
            self._new_lease(binding, identity, head) if record is None else record.lease
        )
        self._save(connection, JournalRecord(lease, "inflight"))
        self._alias(connection, binding, identity, lease.request_id)
        self.bindings.advance(
            connection,
            binding,
            replace(
                head, pending_request_id=lease.request_id, pending_status="inflight"
            ),
        )
        self.turns.check_budget(connection, binding)
        return lease

    def _pending(
        self, connection: sqlite3.Connection, binding: Binding
    ) -> JournalRecord | None:
        """Cross-check the authenticated head against at most two indexed rows."""
        head = self.bindings.head(connection, binding)
        rows = connection.execute(
            "SELECT * FROM requests WHERE binding_id = ? AND status IN (?, ?) LIMIT 2",
            (binding.id, "inflight", "indeterminate"),
        ).fetchall()
        if head.pending_request_id is None:
            if rows:
                raise corrupt_state()
            return None
        if len(rows) != 1 or row_text(rows[0], "request_id") != head.pending_request_id:
            raise corrupt_state()
        record = self._decode(rows[0], binding)
        if (
            record.status != head.pending_status
            or record.lease.sequence != head.sequence + 1
            or record.lease.previous_carrier != head.carrier
        ):
            raise corrupt_state()
        return record

    def _new_lease(
        self, binding: Binding, identity: RequestIdentity, head: Head
    ) -> RequestLease:
        sequence = head.sequence + 1
        return RequestLease(
            binding,
            "req_" + uuid.uuid4().hex,
            identity.digest,
            identity.idempotency_key,
            self.crypto.carrier(binding, sequence, identity.previous_carrier),
            "resp_" + uuid.uuid4().hex,
            sequence,
            identity.previous_carrier,
            created_at=self.bindings.now(),
        )

    def _alias(
        self,
        connection: sqlite3.Connection,
        binding: Binding,
        identity: RequestIdentity,
        request_id: str,
    ) -> bool:
        row = connection.execute(
            "SELECT request_id FROM idempotency_keys WHERE binding_id = ? AND key_hash = ?",
            (binding.id, identity.key_hash),
        ).fetchone()
        if row is not None:
            if row_text(row, "request_id") != request_id:
                raise state_conflict()
            return False
        connection.execute(
            "INSERT INTO idempotency_keys (binding_id, key_hash, request_id) VALUES (?, ?, ?)",
            (binding.id, identity.key_hash, request_id),
        )
        return True

    def _decode(self, row: sqlite3.Row, binding: Binding) -> JournalRecord:
        request_id = row_text(row, "request_id")
        try:
            plaintext = self.crypto.decrypt(
                row_bytes(row, "payload"),
                context=record_context(binding, "journal", request_id),
            )
            record = decode_journal(plaintext, binding)
            lease = record.lease
            if (
                lease.request_id != request_id
                or row_text(row, "binding_id") != binding.id
                or row_text(row, "status") != record.status
                or row_text(row, "request_hash")
                != request_hash(binding, lease.request_digest, lease.previous_carrier)
            ):
                raise corrupt_state()
            self.crypto.verify_carrier(
                lease.carrier,
                binding,
                sequence=lease.sequence,
                previous_carrier=lease.previous_carrier,
            )
            return record
        except AdapterError as error:
            if error.code == "state_key_missing":
                raise
            raise corrupt_state() from None

    def _save(self, connection: sqlite3.Connection, record: JournalRecord) -> None:
        lease = record.lease
        encrypted = self.crypto.encrypt(
            encode_journal(record),
            context=record_context(lease.binding, "journal", lease.request_id),
        )
        connection.execute(
            """INSERT INTO requests (request_id, binding_id, request_hash, status, payload)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(request_id) DO UPDATE SET status = excluded.status, payload = excluded.payload""",
            (
                lease.request_id,
                lease.binding.id,
                request_hash(
                    lease.binding, lease.request_digest, lease.previous_carrier
                ),
                record.status,
                encrypted,
            ),
        )

    def _require_inflight(
        self, connection: sqlite3.Connection, lease: RequestLease
    ) -> JournalRecord:
        record = self._pending(connection, lease.binding)
        if record is None or record.status != "inflight" or record.lease != lease:
            raise state_conflict()
        return record

    def commit(
        self,
        connection: sqlite3.Connection,
        lease: RequestLease,
        completion: Completion,
    ) -> None:
        record = self._require_inflight(connection, lease)
        self._validate_completion(completion)
        self.turns.load(connection, lease.binding)
        self.turns.append(connection, lease, completion.turn)
        replay = Replay(completion.response, completion.raw_usage)
        self._save(connection, replace(record, status="committed", replay=replay))
        self.turns.check_budget(connection, lease.binding)

    def _validate_completion(self, completion: Completion) -> None:
        if (
            not isinstance(completion, Completion)
            or not isinstance(completion.response, bytes)
            or (
                completion.raw_usage is not None
                and not isinstance(completion.raw_usage, bytes)
            )
        ):
            raise AdapterError(
                "invalid_completion",
                "Completion evidence must contain immutable bytes",
                status_code=500,
            )

    def transition(
        self,
        connection: sqlite3.Connection,
        lease: RequestLease,
        status: Literal["rejected", "indeterminate"],
    ) -> None:
        record = self._require_inflight(connection, lease)
        head = self.bindings.head(connection, lease.binding)
        self._save(connection, replace(record, status=status))
        pending_id = None if status == "rejected" else lease.request_id
        pending_status: Literal["indeterminate"] | None = (
            None if status == "rejected" else "indeterminate"
        )
        self.bindings.advance(
            connection,
            lease.binding,
            replace(head, pending_request_id=pending_id, pending_status=pending_status),
        )

    def recover(self, connection: sqlite3.Connection) -> None:
        """Recover indexed inflight records; missing pending rows fail on reserve."""
        rows = connection.execute(
            "SELECT * FROM requests WHERE status = ?", ("inflight",)
        ).fetchall()
        for row in rows:
            binding = self.bindings.load(
                connection, row_text(row, "binding_id"), check_expiry=False
            )
            record = self._decode(row, binding)
            self.transition(connection, record.lease, "indeterminate")
