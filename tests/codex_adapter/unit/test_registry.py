"""Contracts for explicit registration and weighted credential selection."""

from dataclasses import FrozenInstanceError, replace
from functools import partial

import pytest

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.models import (
    Credential,
    ModelCapabilities,
    RegisteredModel,
    Registry,
)
from codex_adapter.domain.routing import choose_credential


def model(*, weights: tuple[int, ...] = (3, 1)) -> RegisteredModel:
    credentials = tuple(
        Credential(
            id=f"key-{index}",
            version="v1",
            secret=f"test-secret-{index}",
            weight=weight,
        )
        for index, weight in enumerate(weights)
    )
    return RegisteredModel(
        id="gpt-test",
        protocol="responses",
        endpoint="https://upstream.example/responses",
        credentials=credentials,
        capabilities=ModelCapabilities(),
    )


def test_unknown_model_fails_without_fallback() -> None:
    with pytest.raises(AdapterError, match="not registered") as raised:
        Registry(models=(model(),)).get("gpt-other")
    assert raised.value.status_code == 404
    assert raised.value.code == "model_not_registered"


def _draw_slot(slot: int, total: int) -> int:
    assert slot < total
    return slot


def test_weighted_slots_match_configured_weights() -> None:
    selected = tuple(
        choose_credential(model(), draw=partial(_draw_slot, slot)).id
        for slot in range(4)
    )
    assert selected == ("key-0", "key-0", "key-0", "key-1")


def test_zero_weight_never_receives_new_sessions() -> None:
    assert choose_credential(model(weights=(0, 1)), draw=lambda total: 0).id == "key-1"


def test_draining_credential_is_not_selected() -> None:
    original = model()
    draining = replace(original.credentials[0], status="draining")
    selected = choose_credential(
        replace(original, credentials=(draining, original.credentials[1])),
        draw=lambda total: 0,
    )
    assert selected.id == "key-1"


def test_registry_rejects_duplicate_models() -> None:
    with pytest.raises(AdapterError, match="Duplicate"):
        Registry(models=(model(), model()))


def test_gpt_cannot_use_a_chat_codec() -> None:
    with pytest.raises(AdapterError, match="Responses"):
        replace(model(), protocol="gemini_chat")


def test_credentials_are_immutable_and_redacted() -> None:
    credential = model().credentials[0]
    assert credential.secret not in repr(credential)
    with pytest.raises(FrozenInstanceError):
        credential.weight = 2  # type: ignore[misc]


@pytest.mark.parametrize("weights", [(-1, 1), (0, 0)])
def test_invalid_weights_fail(weights: tuple[int, ...]) -> None:
    with pytest.raises(AdapterError):
        choose_credential(model(weights=weights), draw=lambda total: 0)
