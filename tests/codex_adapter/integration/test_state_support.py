"""Shared test-only constructors for the SQLite persistence boundary."""

from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path

import pytest

from codex_adapter.domain.models import Credential, ModelCapabilities, RegisteredModel
from codex_adapter.domain.state import (
    Completion,
    RequestLease,
    SavedTurn,
    StateKey,
    StateSettings,
)
from codex_adapter.infrastructure.sqlite_state import SQLiteStateStore


def state_settings(directory: Path) -> StateSettings:
    return StateSettings(
        directory / "state.sqlite", (StateKey("key-1", b"k" * 32),), "key-1"
    )


def registered_model() -> RegisteredModel:
    return RegisteredModel(
        "gemini-test",
        "gemini_chat",
        "https://upstream.example/chat/completions",
        (
            Credential("credential-a", "v1", "test-ak-first", 3),
            Credential("credential-b", "v1", "test-ak-second", 1),
        ),
        ModelCapabilities(),
    )


def digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def completion(
    lease: RequestLease,
    *,
    response: bytes = b'{"id":"recorded-response","output":[]}',
    native_message: bytes = b'{"role":"assistant","content":"private-answer"}',
    raw_usage: bytes | None = b'{"upstream_usage":{"private":17}}',
) -> Completion:
    turn = SavedTurn(
        lease.carrier,
        native_message,
        (b'{"type":"message","id":"message-1"}',),
        (("native-call", "codex-call"),),
        1,
        digest("prefix"),
        digest("options"),
    )
    return Completion(response, turn, raw_usage)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteStateStore]:
    instance = SQLiteStateStore(
        state_settings(tmp_path), clock=lambda: 100.0, draw=lambda total: 0
    )
    instance.initialize()
    instance.ready()
    try:
        yield instance
    finally:
        instance.close()
