"""Weighted selection applies only to creation of a durable session binding."""

from __future__ import annotations

import secrets
from collections.abc import Callable

from .errors import AdapterError
from .models import RegisteredModel, Upstream


def choose_upstream(
    model: RegisteredModel, *, draw: Callable[[int], int] = secrets.randbelow
) -> Upstream:
    """Choose one new-session upstream using integer weights; never rebalance."""
    candidates = tuple(
        item for item in model.upstreams if item.status == "active" and item.weight > 0
    )
    total = sum(item.weight for item in candidates)
    if total <= 0:
        raise AdapterError(
            "model_unavailable",
            "Model has no upstreams accepting new sessions",
            status_code=503,
        )
    slot = draw(total)
    if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < total:
        raise AdapterError(
            "invalid_selection",
            "Upstream selector returned an invalid slot",
            status_code=500,
        )
    cumulative = 0
    for item in candidates:
        cumulative += item.weight
        if slot < cumulative:
            return item
    raise AdapterError(
        "invalid_selection", "Upstream selection failed", status_code=500
    )
