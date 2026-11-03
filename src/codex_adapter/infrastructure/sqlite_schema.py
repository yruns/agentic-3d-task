"""Versioned schema definitions; schema drift is an explicit startup failure."""

from __future__ import annotations

import sqlite3
import uuid

from .sqlite_records import corrupt_state, row_int, row_text

SCHEMA_VERSION = 2
SCHEMA = (
    """CREATE TABLE metadata (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        schema_version INTEGER NOT NULL,
        database_id TEXT NOT NULL
    )""",
    """CREATE TABLE bindings (
        id TEXT PRIMARY KEY,
        principal TEXT NOT NULL,
        session_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        protocol TEXT NOT NULL,
        codec_version TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        credential_id TEXT NOT NULL,
        credential_version TEXT NOT NULL,
        credential_fingerprint TEXT NOT NULL,
        created_at REAL NOT NULL,
        expires_at REAL NOT NULL,
        head_state TEXT NOT NULL,
        UNIQUE (principal, session_id)
    )""",
    """CREATE TABLE turns (
        binding_id TEXT NOT NULL REFERENCES bindings(id),
        sequence INTEGER NOT NULL CHECK (sequence > 0),
        carrier TEXT NOT NULL UNIQUE,
        payload TEXT NOT NULL,
        PRIMARY KEY (binding_id, sequence)
    )""",
    """CREATE TABLE requests (
        request_id TEXT PRIMARY KEY,
        binding_id TEXT NOT NULL REFERENCES bindings(id),
        request_hash TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('inflight', 'committed', 'rejected', 'indeterminate')),
        payload TEXT NOT NULL,
        UNIQUE (binding_id, request_hash)
    )""",
    """CREATE INDEX requests_status ON requests (status, binding_id)""",
    """CREATE TABLE idempotency_keys (
        binding_id TEXT NOT NULL REFERENCES bindings(id),
        key_hash TEXT NOT NULL,
        request_id TEXT NOT NULL REFERENCES requests(request_id),
        PRIMARY KEY (binding_id, key_hash)
    )""",
)


def create_schema(connection: sqlite3.Connection) -> str:
    for statement in SCHEMA:
        connection.execute(statement)
    database_id = uuid.uuid4().hex
    connection.execute(
        "INSERT INTO metadata (singleton, schema_version, database_id) VALUES (?, ?, ?)",
        (1, SCHEMA_VERSION, database_id),
    )
    return database_id


def verify_schema(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE ? ORDER BY name",
        ("sqlite_%",),
    ).fetchall()
    actual = {" ".join(row_text(row, "sql").split()) for row in rows}
    expected = {" ".join(statement.split()) for statement in SCHEMA}
    if actual != expected:
        raise corrupt_state()
    metadata = connection.execute(
        "SELECT * FROM metadata WHERE singleton = ?", (1,)
    ).fetchone()
    if metadata is None or row_int(metadata, "schema_version") != SCHEMA_VERSION:
        raise corrupt_state()
    check = connection.execute("PRAGMA quick_check").fetchall()
    if (
        len(check) != 1
        or check[0][0] != "ok"
        or connection.execute("PRAGMA foreign_key_check").fetchone() is not None
    ):
        raise corrupt_state()
    return row_text(metadata, "database_id")
