"""Fixed perception vocabulary cannot depend on the requested task."""

from codex_agent.alfred.vocabulary import (
    ALFRED_CATEGORIES,
    EXCLUDED_BACKGROUND_CATEGORIES,
    category_prompt,
    detection_batches,
)


def test_full_vocabulary_includes_control_objects_and_transformed_food() -> None:
    assert len(ALFRED_CATEGORIES) == len(set(ALFRED_CATEGORIES)) == 85
    assert len(EXCLUDED_BACKGROUND_CATEGORIES) == 29
    assert not (set(ALFRED_CATEGORIES) & set(EXCLUDED_BACKGROUND_CATEGORIES))
    assert {
        "Faucet",
        "AppleSliced",
        "BreadSliced",
        "LettuceSliced",
        "PotatoSliced",
        "TomatoSliced",
    }.issubset(ALFRED_CATEGORIES)
    assert category_prompt("AppleSliced") == "sliced apple"
    assert category_prompt("CounterTop") == "counter top"


def test_fixed_batches_cover_every_category_once() -> None:
    batches = detection_batches(29)
    assert len(batches) == 3
    assert (
        tuple(category for batch in batches for category in batch) == ALFRED_CATEGORIES
    )
    assert all(len(batch) <= 29 for batch in batches)
