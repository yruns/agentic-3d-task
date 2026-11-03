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
    state_conflict,
    validate_identity,
)
from .sqlite_records import (
    Head,
    JournalRecord,
    corrupt_state,
    decode_journal,
    encode_journal,
    format_carrier,
    row_payload,
    row_text,
    verify_carrier,
)
from .sqlite_turns import SQLiteTurns


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
    """Plaintext reservations and exact response replay inside SQLite transactions."""

    def __init__(
        self,
        settings: StateSettings,
        bindings: SQLiteBindings,
        turns: SQLiteTurns,
    ) -> None:
        self.settings = settings
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
        head = self.bindings.head(connection, binding)
        self.turns.verify_head(connection, binding, head)
        pending = self._pending(connection, binding, head=head)
        existing = self._existing(connection, binding, identity)
        record = existing[0] if existing is not None else None
        old_bytes = existing[1] if existing is not None else 0
        if record is not None and record.status == "committed":
            if record.replay is None:
                raise corrupt_state()
            alias_bytes = self._alias(
                connection, binding, identity, record.lease.request_id
            )
            if alias_bytes > 0:
                self.turns.check_budget(connection, binding, delta_bytes=alias_bytes)
            return record.replay
        if pending is not None:
            raise AdapterError(
                "request_in_progress",
                "Session has an unfinished or indeterminate request",
                status_code=409,
            )
        return self._reserve_lease(
            connection, binding, identity, record, head, old_bytes=old_bytes
        )

    def _existing(
        self,
        connection: sqlite3.Connection,
        binding: Binding,
        identity: RequestIdentity,
    ) -> tuple[JournalRecord, int] | None:
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
        record, payload_bytes = self._decode_with_size(row, binding)
        if (
            record.lease.request_digest != identity.digest
            or record.lease.previous_carrier != identity.previous_carrier
        ):
            raise state_conflict()
        return record, payload_bytes

    def _reserve_lease(
        self,
        connection: sqlite3.Connection,
        binding: Binding,
        identity: RequestIdentity,
        record: JournalRecord | None,
        head: Head,
        *,
        old_bytes: int,
    ) -> RequestLease:
        if head.carrier != identity.previous_carrier:
            raise state_conflict()
        if head.sequence >= self.settings.max_turns:
            raise AdapterError(
                "turn_quota", "Session turn capacity has been reached", status_code=413
            )
        lease = (
            self._new_lease(binding, identity, head) if record is None else record.lease
        )
        new_bytes = self._save(connection, JournalRecord(lease, "inflight"))
        alias_bytes = self._alias(connection, binding, identity, lease.request_id)
        self.bindings.advance(
            connection,
            binding,
            replace(
                head, pending_request_id=lease.request_id, pending_status="inflight"
            ),
        )
        self.turns.check_budget(
            connection, binding, delta_bytes=(new_bytes - old_bytes) + alias_bytes
        )
        return lease

    def _pending(
        self,
        connection: sqlite3.Connection,
        binding: Binding,
        *,
        head: Head | None = None,
    ) -> tuple[JournalRecord, int] | None:
        """Cross-check the head against at most two indexed rows."""
        if head is None:
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
        record, payload_bytes = self._decode_with_size(rows[0], binding)
        if (
            record.status != head.pending_status
            or record.lease.sequence != head.sequence + 1
            or record.lease.previous_carrier != head.carrier
        ):
            raise corrupt_state()
        return record, payload_bytes

    def _new_lease(
        self, binding: Binding, identity: RequestIdentity, head: Head
    ) -> RequestLease:
        sequence = head.sequence + 1
        return RequestLease(
            binding,
            "req_" + uuid.uuid4().hex,
            identity.digest,
            identity.idempotency_key,
            format_carrier(binding.id, sequence),
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
    ) -> int:
        row = connection.execute(
            "SELECT request_id FROM idempotency_keys WHERE binding_id = ? AND key_hash = ?",
            (binding.id, identity.key_hash),
        ).fetchone()
        if row is not None:
            if row_text(row, "request_id") != request_id:
                raise state_conflict()
            return 0
        connection.execute(
            "INSERT INTO idempotency_keys (binding_id, key_hash, request_id) VALUES (?, ?, ?)",
            (binding.id, identity.key_hash, request_id),
        )
        return len(identity.key_hash) + len(request_id)

    def _decode_with_size(
        self, row: sqlite3.Row, binding: Binding
    ) -> tuple[JournalRecord, int]:
        request_id = row_text(row, "request_id")
        try:
            payload = row_payload(row, "payload")
            record = decode_journal(payload, binding)
            lease = record.lease
            if (
                lease.request_id != request_id
                or row_text(row, "binding_id") != binding.id
                or row_text(row, "status") != record.status
                or row_text(row, "request_hash")
                != request_hash(binding, lease.request_digest, lease.previous_carrier)
            ):
                raise corrupt_state()
            verify_carrier(lease.carrier, binding, sequence=lease.sequence)
            return record, len(payload)
        except AdapterError:
            raise corrupt_state() from None

    def _save(self, connection: sqlite3.Connection, record: JournalRecord) -> int:
        lease = record.lease
        payload = encode_journal(record)
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
                payload,
            ),
        )
        return len(payload.encode("utf-8"))

    def _require_inflight(
        self, connection: sqlite3.Connection, lease: RequestLease
    ) -> tuple[JournalRecord, int]:
        pending = self._pending(connection, lease.binding)
        if (
            pending is None
            or pending[0].status != "inflight"
            or pending[0].lease != lease
        ):
            raise state_conflict()
        return pending

    def commit(
        self,
        connection: sqlite3.Connection,
        lease: RequestLease,
        completion: Completion,
    ) -> None:
        record, old_req_bytes = self._require_inflight(connection, lease)
        self._validate_completion(completion)
        turn_bytes = self.turns.append(connection, lease, completion.turn)
        replay = Replay(completion.response, completion.raw_usage)
        new_req_bytes = self._save(
            connection, replace(record, status="committed", replay=replay)
        )
        self.turns.check_budget(
            connection,
            lease.binding,
            delta_bytes=turn_bytes + (new_req_bytes - old_req_bytes),
        )

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
        record, old_req_bytes = self._require_inflight(connection, lease)
        head = self.bindings.head(connection, lease.binding)
        new_req_bytes = self._save(connection, replace(record, status=status))
        pending_id = None if status == "rejected" else lease.request_id
        pending_status: Literal["indeterminate"] | None = (
            None if status == "rejected" else "indeterminate"
        )
        self.bindings.advance(
            connection,
            lease.binding,
            replace(head, pending_request_id=pending_id, pending_status=pending_status),
        )
        if new_req_bytes != old_req_bytes:
            self.turns.invalidate_budget(lease.binding.id)

    def recover(self, connection: sqlite3.Connection) -> None:
        """Recover indexed inflight records; missing pending rows fail on reserve."""
        rows = connection.execute(
            "SELECT * FROM requests WHERE status = ?", ("inflight",)
        ).fetchall()
        for row in rows:
            binding = self.bindings.load(
                connection, row_text(row, "binding_id"), check_expiry=False
            )
            record, _ = self._decode_with_size(row, binding)
            self.transition(connection, record.lease, "indeterminate")
