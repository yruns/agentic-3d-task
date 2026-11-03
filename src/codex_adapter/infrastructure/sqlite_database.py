"""Explicit SQLite lifecycle, process ownership, and serialized transactions."""

from __future__ import annotations

import fcntl
import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Event, RLock

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.json_codec import encode_json
from codex_adapter.domain.state import StateSettings

from .sqlite_records import row_bytes, row_text
from .sqlite_schema import create_schema, verify_schema
from .state_crypto import StateCrypto

_KEY_PROOF = b"codex_adapter.state.v1.verified-master-key"


def unavailable_state() -> AdapterError:
    return AdapterError(
        "state_unavailable", "Adapter state storage is unavailable", status_code=503
    )


class SQLiteDatabase:
    """One connection and advisory ownership lock per worker; threads serialize IO."""

    def __init__(self, settings: StateSettings, crypto: StateCrypto) -> None:
        self.settings = settings
        self.crypto = crypto
        self._connection: sqlite3.Connection | None = None
        self._file_descriptor: int | None = None
        self._lock = RLock()
        self._closing = Event()

    def initialize(self) -> None:
        """Create a new database explicitly, never replace any existing file."""
        with self._lock:
            if self._connection is not None:
                raise AdapterError(
                    "state_exists", "State storage already exists", status_code=409
                )
            self._closing.clear()
            self._open(create=True)
            try:
                with self.transaction() as connection:
                    database_id = create_schema(connection)
                    self._register_active_key(connection, database_id)
            except BaseException:
                self.close()
                raise

    def ready(self) -> bool:
        """Verify schema/key proofs without creating a DB; return whether reopened."""
        with self._lock:
            reopened = self._connection is None
            if not reopened and self._closing.is_set():
                raise unavailable_state()
            if reopened:
                self._closing.clear()
                self._open(create=False)
            try:
                with self.transaction() as connection:
                    database_id = verify_schema(connection)
                    self._verify_keys(connection, database_id)
                    self._register_active_key(connection, database_id)
            except BaseException:
                self.close()
                raise
            return reopened

    def _open(self, *, create: bool) -> None:
        descriptor = self._acquire_file(create=create)
        try:
            uri = self.settings.database.absolute().as_uri() + "?mode=rw"
            connection = sqlite3.connect(
                uri,
                uri=True,
                isolation_level=None,
                check_same_thread=False,
                timeout=5.0,
            )
            connection.row_factory = sqlite3.Row
            self._connection = connection
            self._file_descriptor = descriptor
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA journal_mode = WAL")
        except sqlite3.Error:
            self._file_descriptor = descriptor
            self.close()
            raise unavailable_state() from None

    def _acquire_file(self, *, create: bool) -> int:
        descriptor = self._acquire_owner_lock()
        try:
            self._probe_database(create=create)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    def _acquire_owner_lock(self) -> int:
        lock_path = self.settings.database.with_name(
            self.settings.database.name + ".owner.lock"
        )
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError:
            raise unavailable_state() from None
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise unavailable_state()
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, AdapterError):
            os.close(descriptor)
            raise unavailable_state() from None
        return descriptor

    def _probe_database(self, *, create: bool) -> None:
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(self.settings.database, flags, 0o600)
        except FileExistsError:
            raise AdapterError(
                "state_exists", "State storage already exists", status_code=409
            ) from None
        except OSError:
            raise unavailable_state() from None
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise unavailable_state()
        finally:
            os.close(descriptor)

    def _proof_context(self, database_id: str, key_id: str) -> bytes:
        return encode_json(
            {"record": "key-proof", "database": database_id, "key": key_id}
        )

    def _verify_keys(self, connection: sqlite3.Connection, database_id: str) -> None:
        rows = connection.execute("SELECT key_id, proof FROM key_proofs").fetchall()
        if not rows:
            raise AdapterError(
                "state_key_missing",
                "State key verification metadata is missing",
                status_code=503,
            )
        for row in rows:
            key_id = row_text(row, "key_id")
            try:
                plaintext = self.crypto.decrypt(
                    row_bytes(row, "proof"),
                    context=self._proof_context(database_id, key_id),
                )
            except AdapterError as error:
                if error.code == "state_key_missing":
                    raise
                raise AdapterError(
                    "state_key_mismatch",
                    "State master key verification failed",
                    status_code=503,
                ) from None
            if plaintext != _KEY_PROOF:
                raise AdapterError(
                    "state_key_mismatch",
                    "State master key verification failed",
                    status_code=503,
                )

    def _register_active_key(
        self, connection: sqlite3.Connection, database_id: str
    ) -> None:
        key_id = self.crypto.active_key_id
        if (
            connection.execute(
                "SELECT key_id FROM key_proofs WHERE key_id = ?", (key_id,)
            ).fetchone()
            is not None
        ):
            return
        proof = self.crypto.encrypt(
            _KEY_PROOF, context=self._proof_context(database_id, key_id)
        )
        connection.execute(
            "INSERT INTO key_proofs (key_id, proof) VALUES (?, ?)", (key_id, proof)
        )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialize the entire operation and sanitize all SQLite error boundaries."""
        with self._lock:
            connection = self._connection
            if connection is None or self._closing.is_set():
                raise unavailable_state()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except sqlite3.Error:
                connection.rollback()
                raise unavailable_state() from None
            except BaseException:
                connection.rollback()
                raise

    def close(self) -> None:
        """Gate queued work, drain active SQL, then release connection/ownership.

        Once this returns, no SQL remains active. Previously queued transaction
        calls can still wake, but fail the closing gate without touching SQLite.
        """
        self._closing.set()
        with self._lock:
            connection, descriptor = self._connection, self._file_descriptor
            self._connection = None
            self._file_descriptor = None
            try:
                if connection is not None:
                    connection.close()
            except sqlite3.Error:
                raise unavailable_state() from None
            finally:
                if descriptor is not None:
                    os.close(descriptor)
