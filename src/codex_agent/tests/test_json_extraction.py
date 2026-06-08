"""Unit tests for tolerant JSON-object extraction from Codex responses."""

from __future__ import annotations

import pytest

from codex_agent.errors import CodexResponseError
from codex_agent.json_extraction import extract_json_object


def test_parses_bare_object() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_parses_fenced_json() -> None:
    text = 'Here is the answer:\n```json\n{"a": 2}\n```\n'
    assert extract_json_object(text) == {"a": 2}


def test_parses_object_after_prose() -> None:
    text = 'Final answer: {"proposal_id": 5, "confidence": 0.8} done.'
    assert extract_json_object(text) == {"proposal_id": 5, "confidence": 0.8}


def test_empty_text_raises() -> None:
    with pytest.raises(CodexResponseError):
        extract_json_object("   ")


def test_non_object_json_raises() -> None:
    with pytest.raises(CodexResponseError):
        extract_json_object("[1, 2, 3]")


def test_garbage_raises() -> None:
    with pytest.raises(CodexResponseError):
        extract_json_object("not json at all")
