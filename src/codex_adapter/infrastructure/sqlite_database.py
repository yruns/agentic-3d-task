"""Explicit SQLite lifecycle, process ownership, and concurrent WAL read/write."""

from __future__ import annotations

import fcntl
import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Condition, Event, RLock

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.state import StateSettings

from .sqlite_schema import create_schema, verify_schema

_MAX_IDLE_READERS = 8


def unavailable_state() -> AdapterError:
    return AdapterError(
        "state_unavailable", "Adapter state storage is unavailable", status_code=503
    )


def _configure_connection(connection: sqlite3.Connection, *, query_only: bool) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA foreign_keys = ON")
    if not query_only:
        connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA mmap_size = 268435456")
    connection.execute("PRAGMA cache_size = -65536")
    if query_only:
        connection.execute("PRAGMA query_only = ON")


class SQLiteDatabase:
    """One writer and pooled concurrent WAL readers per worker process."""

    def __init__(self, settings: StateSettings) -> None:
        self.settings = settings
        self._connection: sqlite3.Connection | None = None
        self._file_descriptor: int | None = None
        self._lock = RLock()
        self._closing = Event()
        self._reader_cond = Condition(RLock())
        self._idle_readers: list[sqlite3.Connection] = []
        self._active_readers = 0

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
                    create_schema(connection)
            except BaseException:
                self.close()
                raise

    def ready(self) -> bool:
        """Verify schema on open without creating a DB; return whether reopened."""
        with self._lock:
            reopened = self._connection is None
            if not reopened and self._closing.is_set():
                raise unavailable_state()
            if reopened:
                self._closing.clear()
                self._open(create=False)
                try:
                    with self.transaction() as connection:
                        verify_schema(connection)
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
            _configure_connection(connection, query_only=False)
            self._connection = connection
            self._file_descriptor = descriptor
        except sqlite3.Error:
            self._file_descriptor = descriptor
            self.close()
            raise unavailable_state() from None

    def _open_reader(self) -> sqlite3.Connection:
        uri = self.settings.database.absolute().as_uri() + "?mode=rw"
        try:
            connection = sqlite3.connect(
                uri,
                uri=True,
                isolation_level=None,
                check_same_thread=False,
                timeout=5.0,
            )
            _configure_connection(connection, query_only=True)
            return connection
        except sqlite3.Error:
            raise unavailable_state() from None

    def _acquire_read_connection(self) -> sqlite3.Connection:
        with self._reader_cond:
            if self._connection is None or self._closing.is_set():
                raise unavailable_state()
            self._active_readers += 1
            if self._idle_readers:
                return self._idle_readers.pop()
        try:
            return self._open_reader()
        except BaseException:
            with self._reader_cond:
                self._active_readers -= 1
                if self._active_readers == 0:
                    self._reader_cond.notify_all()
            raise

    def _release_read_connection(
        self, connection: sqlite3.Connection, *, discard: bool
    ) -> None:
        to_close: sqlite3.Connection | None = None
        with self._reader_cond:
            if (
                discard
                or self._closing.is_set()
                or self._connection is None
                or len(self._idle_readers) >= _MAX_IDLE_READERS
            ):
                to_close = connection
            else:
                self._idle_readers.append(connection)
            self._active_readers -= 1
            if self._active_readers == 0:
                self._reader_cond.notify_all()
        if to_close is not None:
            try:
                to_close.close()
            except sqlite3.Error:
                pass

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

    @contextmanager
    def read_transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a concurrent WAL read snapshot without blocking writers or other readers."""
        connection = self._acquire_read_connection()
        discard = False
        try:
            connection.execute("BEGIN DEFERRED")
            yield connection
            connection.rollback()
        except sqlite3.Error:
            discard = True
            try:
                connection.rollback()
            except sqlite3.Error:
                pass
            raise unavailable_state() from None
        except BaseException:
            try:
                connection.rollback()
            except sqlite3.Error:
                discard = True
            raise
        finally:
            self._release_read_connection(connection, discard=discard)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialize write operations and sanitize all SQLite error boundaries."""
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
        """Gate queued work, drain active SQL readers/writers, then release ownership."""
        self._closing.set()
        with self._reader_cond:
            while self._active_readers > 0:
                self._reader_cond.wait()
            idle_readers = list(self._idle_readers)
            self._idle_readers.clear()
        for reader in idle_readers:
            try:
                reader.close()
            except sqlite3.Error:
                pass
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
