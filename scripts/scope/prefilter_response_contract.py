"""Recognize the released SCOPE request whose object candidate list is empty.

The prompt is pinned to SCOPE 46f1e1f78077219c30e35440540e4ef22995194c,
``src/eval_utils_gpt_goatbench.py:290-325``. This module only recognizes the
request; callers independently validate response content and finish reason.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

EMPTY_PREFILTER_HEADER: Final[str] = "X-SCOPE-Empty-Prefilter"

_SYSTEM_PROMPT: Final[str] = "You are an AI agent in a 3D indoor scene.\n"
_INSTRUCTIONS: Final[str] = (
    "Your goal is to answer questions about the scene through exploration.\n"
    "To efficiently solve the problem, you should first rank objects in the scene "
    "based on their importance.\n"
    "These are the rules for the task.\n"
    "1. Read through the whole object list.\n"
    "2. Rank objects in the list based on how well they can help your exploration "
    "given the question.\n"
    "3. Reprint the name of all objects that may help your exploration given the "
    "question. "
    "4. Do not print any object not included in the list or include any additional "
    "information in your response.\n"
)
_EXAMPLE: Final[str] = (
    "Here is an example of selecting helpful objects:\n"
    "Question: What can I use to watch my favorite shows and movies?\n"
    "Following is a list of objects that you can choose, each object one line\n"
    "painting\nspeaker\nbox\ncabinet\nlamp\ntv\nbook rack\nsofa\noven\nbed\ncurtain\n"
    "Answer: tv\nspeaker\nsofa\nbed\n"
)
_QUESTION_PREFIX: Final[str] = (
    "Following is the concrete content of the task and you should retrieve helpful "
    "objects in order:\nQuestion: "
)
_EMPTY_OBJECT_LIST: Final[str] = (
    "Following is a list of objects that you can choose, each object one line\n"
    "Answer: "
)


class _TextPart(BaseModel):
    """One original plain-text prompt part, with no additional content fields."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    type: Literal["text"]
    text: str


class _SystemMessage(BaseModel):
    """The single system message preceding the original SCOPE user prompt."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    role: Literal["system"]
    content: str


class _UserMessage(BaseModel):
    """Exactly four text parts; JSON arrays become immutable typed tuples."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    role: Literal["user"]
    content: tuple[_TextPart, _TextPart, _TextPart, _TextPart] = Field(strict=False)


_MESSAGES_ADAPTER: TypeAdapter[tuple[_SystemMessage, _UserMessage]] = TypeAdapter(
    tuple[_SystemMessage, _UserMessage]
)


def allows_empty_object_prefilter_response(messages: object) -> bool:
    """Return whether messages exactly identify SCOPE's empty-list prefilter.

    Malformed inputs, changed instructions, images and nonempty candidate lists
    do not qualify. Recognition permits no inference about the response itself.
    """
    try:
        system_message, user_message = _MESSAGES_ADAPTER.validate_python(messages)
    except ValidationError:
        return False

    instructions, example, question, object_list = user_message.content
    return (
        system_message.content == _SYSTEM_PROMPT
        and instructions.text == _INSTRUCTIONS
        and example.text == _EXAMPLE
        and question.text.startswith(_QUESTION_PREFIX)
        and question.text.endswith("\n")
        and bool(question.text[len(_QUESTION_PREFIX) : -1].strip())
        and object_list.text == _EMPTY_OBJECT_LIST
    )
