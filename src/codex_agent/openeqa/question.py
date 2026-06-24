"""Loaders and domain model for OpenEQA questions.

OpenEQA ships as a single JSON array (``open-eqa-v0.json``) where each record is
one open-ended question about an embodied scene. This module turns that
(untrusted) file into validated :class:`OpenEqaQuestion` domain objects.

Only the ScanNet split is locally runnable in this repository (the HM3D split
needs frames that are not prepared here), so loading defaults to filtering on the
``scannet-v0/`` episode prefix. A question's ``episode_history`` (for example
``scannet-v0/002-scannet-scene0709_00``) maps to the on-disk ``clip_id``
(``002-scannet-scene0709_00``) that names the prepared scene directory.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..errors import OpenEqaDataError

#: Episode prefix for the locally-prepared ScanNet split.
SCANNET_EPISODE_PREFIX = "scannet-v0/"
#: Infix that separates the clip index from the ScanNet scene id in a clip id
#: (``002-scannet-scene0709_00`` -> scene ``scene0709_00``).
_SCANNET_INFIX = "-scannet-"


class _RawOpenEqaRecord(BaseModel):
    """Validation boundary for one raw record in ``open-eqa-v0.json``.

    Extra keys are ignored (the upstream file occasionally carries optional
    fields such as ``question_type`` / ``scene_id`` that this task does not use).
    """

    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    question_id: str = ""
    category: str = "unknown"
    episode_history: str = ""
    extra_answers: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class OpenEqaQuestion:
    """One OpenEQA question with its ground-truth answer and scene reference.

    Attributes:
        question_id: Stable identifier (a UUID in the official file).
        question: The natural-language question shown to the agent.
        answer: The reference answer used by the LLM-as-judge.
        category: OpenEQA category, e.g. ``"object recognition"``.
        episode_history: Raw ``<dataset-v0>/<clip_id>`` reference.
        clip_id: On-disk clip directory name (``<data_root>/<clip_id>``).
        scene_id: ScanNet scene id derived from ``clip_id`` (for reporting).
        extra_answers: Additional acceptable answers, when the file provides them.
    """

    question_id: str
    question: str
    answer: str
    category: str
    episode_history: str
    clip_id: str
    scene_id: str
    extra_answers: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_raw(
        cls,
        record: _RawOpenEqaRecord,
        *,
        index: int,
        dataset_prefix: str,
    ) -> OpenEqaQuestion:
        """Build a question from a validated raw record.

        Args:
            record: A validated raw record.
            index: Position in the file, used to synthesize a ``question_id``
                when the record omits one (mirrors the upstream loader).
            dataset_prefix: Episode prefix that was used to select this record;
                stripped from ``episode_history`` to derive ``clip_id``.

        Raises:
            OpenEqaDataError: If ``episode_history`` does not start with
                ``dataset_prefix`` (i.e. the record was selected incorrectly).
        """
        if not record.episode_history.startswith(dataset_prefix):
            raise OpenEqaDataError(
                f"episode_history={record.episode_history!r} does not start with "
                f"dataset_prefix={dataset_prefix!r}"
            )
        clip_id = record.episode_history[len(dataset_prefix) :]
        if not clip_id:
            raise OpenEqaDataError(
                f"episode_history={record.episode_history!r} has an empty clip id"
            )
        return cls(
            question_id=record.question_id or str(index),
            question=record.question,
            answer=record.answer,
            category=record.category or "unknown",
            episode_history=record.episode_history,
            clip_id=clip_id,
            scene_id=_derive_scene_id(clip_id),
            extra_answers=tuple(record.extra_answers),
        )


def load_questions(
    questions_path: Path, *, dataset_prefix: str = SCANNET_EPISODE_PREFIX
) -> tuple[OpenEqaQuestion, ...]:
    """Load and validate questions from an ``open-eqa-v0.json`` file.

    Only records whose ``episode_history`` starts with ``dataset_prefix`` are
    returned (default: the locally-runnable ScanNet split).

    Raises:
        OpenEqaDataError: If the file is missing, is not a JSON array, or any
            selected record fails validation.
    """
    if not questions_path.exists():
        raise OpenEqaDataError(f"questions file is missing: {questions_path}")
    payload = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise OpenEqaDataError(
            f"{questions_path}: expected a JSON array, got {type(payload).__name__}"
        )
    questions: list[OpenEqaQuestion] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise OpenEqaDataError(
                f"{questions_path}: items[{index}] must be a JSON object"
            )
        episode_history = item.get("episode_history", "")
        if not isinstance(episode_history, str) or not episode_history.startswith(
            dataset_prefix
        ):
            continue
        questions.append(
            OpenEqaQuestion.from_raw(
                _validate_record(item, index=index, source=questions_path),
                index=index,
                dataset_prefix=dataset_prefix,
            )
        )
    return tuple(questions)


def index_by_question_id(
    questions: Sequence[OpenEqaQuestion],
) -> dict[str, OpenEqaQuestion]:
    """Index questions by ``question_id``.

    Raises:
        OpenEqaDataError: If two questions share a ``question_id`` (the runner
            checkpoints per id, so ids must be unique).
    """
    by_id: dict[str, OpenEqaQuestion] = {}
    for question in questions:
        if question.question_id in by_id:
            raise OpenEqaDataError(
                f"duplicate question_id in question set: {question.question_id!r}"
            )
        by_id[question.question_id] = question
    return by_id


def select_questions(
    questions: Sequence[OpenEqaQuestion], question_ids: Sequence[str]
) -> tuple[OpenEqaQuestion, ...]:
    """Return the questions named by ``question_ids``, preserving that order.

    Raises:
        OpenEqaDataError: If any requested id is absent from ``questions``.
    """
    by_id = index_by_question_id(questions)
    missing = [qid for qid in question_ids if qid not in by_id]
    if missing:
        raise OpenEqaDataError(
            f"requested question_ids not found in the question set: {missing[:10]}"
        )
    return tuple(by_id[qid] for qid in question_ids)


def _validate_record(
    item: Mapping[str, Any], *, index: int, source: Path
) -> _RawOpenEqaRecord:
    try:
        return _RawOpenEqaRecord.model_validate(dict(item))
    except ValidationError as exc:
        raise OpenEqaDataError(
            f"{source}: items[{index}] failed validation: {exc}"
        ) from exc


def _derive_scene_id(clip_id: str) -> str:
    """Derive the ScanNet scene id from a clip id.

    ``002-scannet-scene0709_00`` -> ``scene0709_00``; falls back to the clip id
    when the infix is absent.
    """
    if _SCANNET_INFIX in clip_id:
        return clip_id.split(_SCANNET_INFIX, 1)[1]
    return clip_id


__all__ = [
    "SCANNET_EPISODE_PREFIX",
    "OpenEqaQuestion",
    "load_questions",
    "index_by_question_id",
    "select_questions",
]
