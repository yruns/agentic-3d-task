"""Weighted selection applies only to creation of a durable session binding."""

from __future__ import annotations

import secrets
from collections.abc import Callable

from .errors import AdapterError
from .models import Credential, RegisteredModel


def choose_credential(
    model: RegisteredModel, *, draw: Callable[[int], int] = secrets.randbelow
) -> Credential:
    """Choose a new-session credential using integer weights; never rebalance."""
    candidates = tuple(
        key for key in model.credentials if key.status == "active" and key.weight > 0
    )
    total = sum(key.weight for key in candidates)
    if total <= 0:
        raise AdapterError(
            "model_unavailable",
            "Model has no credentials accepting new sessions",
            status_code=503,
        )
    slot = draw(total)
    if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < total:
        raise AdapterError(
            "invalid_selection",
            "Credential selector returned an invalid slot",
            status_code=500,
        )
    cumulative = 0
    for key in candidates:
        cumulative += key.weight
        if slot < cumulative:
            return key
    raise AdapterError(
        "invalid_selection", "Credential selection failed", status_code=500
    )
