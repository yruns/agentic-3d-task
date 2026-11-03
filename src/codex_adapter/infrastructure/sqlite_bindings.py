"""Durable first-bind routing and authenticated session/head identities."""

from __future__ import annotations

import hmac
import math
import sqlite3
from collections.abc import Callable

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import encode_json
from codex_adapter.domain.models import RegisteredModel
from codex_adapter.domain.routing import choose_credential
from codex_adapter.domain.state import Binding, BoundRoute, StateSettings

from .sqlite_records import (
    Head,
    decode_head,
    encode_head,
    row_bytes,
    row_int,
    row_number,
    row_text,
)
from .state_crypto import (
    StateCrypto,
    binding_snapshot,
    corrupt_state,
    credential_fingerprint,
)

_MAX_IDENTITY_BYTES = 1024


def validate_identity(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise AdapterError("invalid_request", f"Invalid {field}", param=field)
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        raise AdapterError("invalid_request", f"Invalid {field}", param=field) from None
    if len(encoded) > _MAX_IDENTITY_BYTES:
        raise AdapterError("invalid_request", f"Invalid {field}", param=field)


def state_conflict() -> AdapterError:
    return AdapterError(
        "state_conflict",
        "Request does not match the durable session state",
        status_code=409,
    )


def record_context(
    binding: Binding, kind: str, identifier: str | int | None = None
) -> bytes:
    return encode_json(
        {"record": kind, "binding": binding_snapshot(binding), "identifier": identifier}
    )


class SQLiteBindings:
    """SQL helpers run only inside the owning database transaction."""

    def __init__(
        self,
        settings: StateSettings,
        crypto: StateCrypto,
        clock: Callable[[], float],
        draw: Callable[[int], int],
    ) -> None:
        self.settings = settings
        self.crypto = crypto
        self.clock = clock
        self.draw = draw

    def now(self) -> float:
        value = self.clock()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise AdapterError(
                "invalid_clock",
                "State clock must return a finite nonnegative timestamp",
                status_code=500,
            )
        return float(value)

    def bind(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        model: RegisteredModel,
        principal: str,
    ) -> BoundRoute:
        validate_identity(session_id, field="session_id")
        validate_identity(principal, field="principal")
        row = self._lookup(connection, principal, session_id)
        if row is None:
            return self._create(connection, session_id, model, principal)
        binding = self._decode_binding(row)
        if binding.principal != principal or binding.session_id != session_id:
            raise corrupt_state()
        self._check_expiry(binding)
        if (
            binding.model_id,
            binding.protocol,
            binding.codec_version,
            binding.endpoint,
        ) != (model.id, model.protocol, model.codec_version, model.endpoint):
            raise state_conflict()
        credential = next(
            (
                key
                for key in model.credentials
                if (key.id, key.version)
                == (binding.credential_id, binding.credential_version)
            ),
            None,
        )
        if (
            credential is None
            or credential.status == "revoked"
            or not hmac.compare_digest(
                binding.credential_fingerprint,
                credential_fingerprint(credential.secret),
            )
        ):
            raise state_conflict()
        return BoundRoute(binding, model, credential)

    def _lookup(
        self, connection: sqlite3.Connection, principal: str, session_id: str
    ) -> sqlite3.Row | None:
        matches: tuple[sqlite3.Row, ...] = tuple(
            row
            for binding_id in self.crypto.binding_ids(principal, session_id)
            if (
                row := connection.execute(
                    "SELECT * FROM bindings WHERE id = ?", (binding_id,)
                ).fetchone()
            )
            is not None
        )
        if len(matches) > 1:
            raise corrupt_state()
        return matches[0] if matches else None

    def _create(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        model: RegisteredModel,
        principal: str,
    ) -> BoundRoute:
        count = connection.execute("SELECT COUNT(*) AS count FROM bindings").fetchone()
        if count is None or row_int(count, "count") >= self.settings.max_sessions:
            raise AdapterError(
                "session_quota",
                "State session capacity has been reached",
                status_code=429,
            )
        credential = choose_credential(model, draw=self.draw)
        created = self.now()
        binding = Binding(
            self.crypto.binding_id(principal, session_id),
            principal,
            session_id,
            model.id,
            model.protocol,
            model.codec_version,
            model.endpoint,
            credential.id,
            credential.version,
            credential_fingerprint(credential.secret),
            self.crypto.active_key_id,
            created,
            created + self.settings.session_ttl_seconds,
        )
        self._insert(connection, binding)
        return BoundRoute(binding, model, credential)

    def _insert(self, connection: sqlite3.Connection, binding: Binding) -> None:
        head = self.crypto.encrypt(
            encode_head(Head()), context=record_context(binding, "head")
        )
        connection.execute(
            """INSERT INTO bindings (
                id, principal, session_id, model_id, protocol, codec_version, endpoint,
                credential_id, credential_version, credential_fingerprint, key_id,
                created_at, expires_at, binding_mac, head_state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                binding.id,
                binding.principal,
                binding.session_id,
                binding.model_id,
                binding.protocol,
                binding.codec_version,
                binding.endpoint,
                binding.credential_id,
                binding.credential_version,
                binding.credential_fingerprint,
                binding.key_id,
                binding.created_at,
                binding.expires_at,
                self.crypto.binding_tag(binding),
                head,
            ),
        )

    def _decode_binding(self, row: sqlite3.Row) -> Binding:
        try:
            binding = Binding(
                row_text(row, "id"),
                row_text(row, "principal"),
                row_text(row, "session_id"),
                row_text(row, "model_id"),
                row_text(row, "protocol"),
                row_text(row, "codec_version"),
                row_text(row, "endpoint"),
                row_text(row, "credential_id"),
                row_text(row, "credential_version"),
                row_text(row, "credential_fingerprint"),
                row_text(row, "key_id"),
                row_number(row, "created_at"),
                row_number(row, "expires_at"),
            )
            self.crypto.verify_binding(binding, row_bytes(row, "binding_mac"))
            if binding.id != self.crypto.binding_id(
                binding.principal, binding.session_id, key_id=binding.key_id
            ):
                raise corrupt_state()
            return binding
        except AdapterError as error:
            if error.code in {"state_corrupt", "state_key_missing"}:
                raise
            raise corrupt_state() from None

    def _check_expiry(self, binding: Binding) -> None:
        if self.now() >= binding.expires_at:
            raise AdapterError(
                "session_expired",
                "Adapter session has expired and cannot be rebound",
                status_code=410,
            )

    def load(
        self,
        connection: sqlite3.Connection,
        binding_id: str,
        *,
        check_expiry: bool = True,
    ) -> Binding:
        row = connection.execute(
            "SELECT * FROM bindings WHERE id = ?", (binding_id,)
        ).fetchone()
        if row is None:
            raise state_conflict()
        binding = self._decode_binding(row)
        if check_expiry:
            self._check_expiry(binding)
        return binding

    def validate(self, connection: sqlite3.Connection, binding: Binding) -> Binding:
        if not isinstance(binding, Binding):
            raise state_conflict()
        persisted = self.load(connection, binding.id)
        if persisted != binding:
            raise state_conflict()
        return persisted

    def head(self, connection: sqlite3.Connection, binding: Binding) -> Head:
        row = connection.execute(
            "SELECT head_state FROM bindings WHERE id = ?", (binding.id,)
        ).fetchone()
        if row is None:
            raise corrupt_state()
        return decode_head(
            self.crypto.decrypt(
                row_bytes(row, "head_state"), context=record_context(binding, "head")
            )
        )

    def advance(
        self, connection: sqlite3.Connection, binding: Binding, head: Head
    ) -> None:
        payload = self.crypto.encrypt(
            encode_head(head), context=record_context(binding, "head")
        )
        connection.execute(
            "UPDATE bindings SET head_state = ? WHERE id = ?", (payload, binding.id)
        )
