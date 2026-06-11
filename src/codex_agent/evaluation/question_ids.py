"""Load OpenEQA question-id folds from JSON.

A fold file is either a list of ``question_id`` strings or a list of objects that
each carry a ``question_id`` field, mirroring the NR3D sample-id fold format.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..errors import OpenEqaDataError


def load_question_ids(path: Path) -> list[str]:
    """Return the ordered list of question ids defined in ``path``.

    Raises:
        OpenEqaDataError: If the file is not a list of strings or of objects with
            a non-empty ``question_id`` field.
    """
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise OpenEqaDataError(
            f"{path}: question-ids JSON must be a list, got {type(items).__name__}"
        )
    if all(isinstance(item, str) and item for item in items):
        return list(items)
    if all(isinstance(item, dict) for item in items):
        return [
            _question_id_from_object(item, index, path)
            for index, item in enumerate(items)
        ]
    raise OpenEqaDataError(
        f"{path}: question-ids JSON must be a list of strings or a list of objects "
        "with a 'question_id' field"
    )


def _question_id_from_object(item: dict[str, object], index: int, path: Path) -> str:
    question_id = item.get("question_id")
    if not isinstance(question_id, str) or not question_id:
        raise OpenEqaDataError(
            f"{path}: items[{index}] must include a non-empty 'question_id'"
        )
    return question_id


__all__ = ["load_question_ids"]
