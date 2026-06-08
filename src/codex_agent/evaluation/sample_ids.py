"""Load NR3D sample-id folds from JSON.

A fold file is either a list of sample-id strings or a list of objects that each
carry a ``sample_id`` field (the canonical strat-600 fold format).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..errors import Nr3dDataError


def load_sample_ids(path: Path) -> list[str]:
    """Return the ordered list of sample ids defined in ``path``.

    Raises:
        Nr3dDataError: If the file is not a list of strings or of objects with a
            non-empty ``sample_id`` field.
    """
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise Nr3dDataError(
            f"{path}: sample-ids JSON must be a list, got {type(items).__name__}"
        )
    if all(isinstance(item, str) and item for item in items):
        return list(items)
    if all(isinstance(item, dict) for item in items):
        return [
            _sample_id_from_object(item, index, path)
            for index, item in enumerate(items)
        ]
    raise Nr3dDataError(
        f"{path}: sample-ids JSON must be a list of strings or a list of objects "
        "with a 'sample_id' field"
    )


def _sample_id_from_object(item: dict[str, object], index: int, path: Path) -> str:
    sample_id = item.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise Nr3dDataError(
            f"{path}: items[{index}] must include a non-empty 'sample_id'"
        )
    return sample_id


__all__ = ["load_sample_ids"]
