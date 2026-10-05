"""Identify only the released SCOPE empty object-list request."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from scripts.scope.prefilter_response_contract import (
    allows_empty_object_prefilter_response,
)


class _TextPart(TypedDict):
    type: Literal["text"]
    text: str


class _SystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class _UserMessage(TypedDict):
    role: Literal["user"]
    content: list[_TextPart]


_MESSAGES_ADAPTER: TypeAdapter[tuple[_SystemMessage, _UserMessage]] = TypeAdapter(
    tuple[_SystemMessage, _UserMessage]
)
_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures/scope_empty_object_prefilter_messages.json"
)


def _load_messages() -> tuple[_SystemMessage, _UserMessage]:
    """Validate the original JSON fixture before constructing test variants."""
    return _MESSAGES_ADAPTER.validate_json(_FIXTURE_PATH.read_text())


def test_identifies_original_empty_list_prefilter() -> None:
    messages = _load_messages()
    assert len(messages[1]["content"]) == 4
    assert allows_empty_object_prefilter_response(messages)


def test_accepts_another_nonempty_question_in_original_prompt() -> None:
    messages = _load_messages()
    question = messages[1]["content"][2]["text"]
    messages[1]["content"][2]["text"] = question.replace(
        "Can you find the decorative plant?", "Can you find the chair?"
    )
    assert allows_empty_object_prefilter_response(messages)


def test_rejects_nonempty_candidate_list() -> None:
    messages = _load_messages()
    objects = messages[1]["content"][3]["text"]
    messages[1]["content"][3]["text"] = objects.replace(
        "\nAnswer: ", "\nchair\nAnswer: "
    )
    assert not allows_empty_object_prefilter_response(messages)


@pytest.mark.parametrize("part_index", [0, 1])
def test_rejects_modified_instructions_or_example(part_index: int) -> None:
    messages = _load_messages()
    messages[1]["content"][part_index]["text"] += " changed"
    assert not allows_empty_object_prefilter_response(messages)


@pytest.mark.parametrize(
    "replacement",
    [
        "Question: Can you find the decorative plant?\n",
        (
            "Following is the concrete content of the task and you should retrieve "
            "helpful objects in order:\nQuestion: \n"
        ),
        (
            "Following is the concrete content of the task and you should retrieve "
            "helpful objects in order:\nQuestion:    \n"
        ),
        (
            "Following is the concrete content of the task and you should retrieve "
            "helpful objects in order:\nQuestion: Can you find the decorative plant?"
        ),
    ],
)
def test_rejects_invalid_question_part(replacement: str) -> None:
    messages = _load_messages()
    messages[1]["content"][2]["text"] = replacement
    assert not allows_empty_object_prefilter_response(messages)


@pytest.mark.parametrize("suffix", ["\n", " ", "None"])
def test_rejects_any_change_to_empty_list_terminator(suffix: str) -> None:
    messages = _load_messages()
    messages[1]["content"][3]["text"] += suffix
    assert not allows_empty_object_prefilter_response(messages)


def test_rejects_modified_system_prompt() -> None:
    messages = _load_messages()
    messages[0]["content"] = messages[0]["content"].rstrip("\n")
    assert not allows_empty_object_prefilter_response(messages)


@pytest.mark.parametrize("part_count", [3, 5])
def test_requires_exactly_four_user_text_parts(part_count: int) -> None:
    messages = _load_messages()
    if part_count == 3:
        messages[1]["content"].pop()
    else:
        messages[1]["content"].append({"type": "text", "text": "extra"})
    assert not allows_empty_object_prefilter_response(messages)


@pytest.mark.parametrize(
    "invalid_messages",
    [
        None,
        True,
        "messages",
        {},
        [],
        [{"role": "system", "content": "system"}],
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "plain text"},
        ],
    ],
)
def test_rejects_malformed_messages(invalid_messages: object) -> None:
    assert not allows_empty_object_prefilter_response(invalid_messages)


def test_rejects_reversed_message_roles() -> None:
    system, user = _load_messages()
    assert not allows_empty_object_prefilter_response([user, system])


def test_rejects_extra_message() -> None:
    system, user = _load_messages()
    assert not allows_empty_object_prefilter_response([system, user, user])


def test_rejects_images_even_when_text_matches() -> None:
    system, user = _load_messages()
    image_part: object = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,test", "detail": "high"},
    }
    content: list[object] = [*user["content"][:3], image_part]
    request: object = [system, {"role": "user", "content": content}]
    assert not allows_empty_object_prefilter_response(request)


@pytest.mark.parametrize("invalid_text", [None, True, 123, ["text"]])
def test_rejects_nonstring_text_without_coercion(invalid_text: object) -> None:
    system, user = _load_messages()
    malformed_part: object = {"type": "text", "text": invalid_text}
    content: list[object] = [malformed_part, *user["content"][1:]]
    request: object = [system, {"role": "user", "content": content}]
    assert not allows_empty_object_prefilter_response(request)


def test_rejects_additional_fields_in_original_request() -> None:
    system, user = _load_messages()
    request: object = [
        system,
        {"role": "user", "content": user["content"], "name": "extra"},
    ]
    assert not allows_empty_object_prefilter_response(request)
