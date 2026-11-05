"""Frozen ALFRED object vocabulary, independent of episode goals or metadata.

Source: ALFRED gen/constants.py OBJECTS (108 classes), its five sliced-food
OBJ_PARENTS entries, and the Faucet control category. These are dataset-level
schema labels; no scene-specific category list is read at runtime.
"""

from __future__ import annotations

import hashlib
import json
import re

ALFRED_CATEGORIES: tuple[str, ...] = (
    "AlarmClock",
    "Apple",
    "AppleSliced",
    "ArmChair",
    "BaseballBat",
    "BasketBall",
    "Bathtub",
    "BathtubBasin",
    "Bed",
    "Blinds",
    "Book",
    "Boots",
    "Bowl",
    "Box",
    "Bread",
    "BreadSliced",
    "ButterKnife",
    "CD",
    "Cabinet",
    "Candle",
    "Cart",
    "CellPhone",
    "Chair",
    "Cloth",
    "CoffeeMachine",
    "CoffeeTable",
    "CounterTop",
    "CreditCard",
    "Cup",
    "Curtains",
    "Desk",
    "DeskLamp",
    "DiningTable",
    "DishSponge",
    "Drawer",
    "Dresser",
    "Egg",
    "Faucet",
    "FloorLamp",
    "Footstool",
    "Fork",
    "Fridge",
    "GarbageCan",
    "Glassbottle",
    "HandTowel",
    "HandTowelHolder",
    "HousePlant",
    "Kettle",
    "KeyChain",
    "Knife",
    "Ladle",
    "Laptop",
    "LaundryHamper",
    "LaundryHamperLid",
    "Lettuce",
    "LettuceSliced",
    "LightSwitch",
    "Microwave",
    "Mirror",
    "Mug",
    "Newspaper",
    "Ottoman",
    "Painting",
    "Pan",
    "PaperTowel",
    "PaperTowelRoll",
    "Pen",
    "Pencil",
    "PepperShaker",
    "Pillow",
    "Plate",
    "Plunger",
    "Poster",
    "Pot",
    "Potato",
    "PotatoSliced",
    "RemoteControl",
    "Safe",
    "SaltShaker",
    "ScrubBrush",
    "Shelf",
    "ShowerDoor",
    "ShowerGlass",
    "SideTable",
    "Sink",
    "SinkBasin",
    "SoapBar",
    "SoapBottle",
    "Sofa",
    "Spatula",
    "Spoon",
    "SprayBottle",
    "Statue",
    "StoveBurner",
    "StoveKnob",
    "TVStand",
    "TeddyBear",
    "Television",
    "TennisRacket",
    "TissueBox",
    "Toaster",
    "Toilet",
    "ToiletPaper",
    "ToiletPaperHanger",
    "ToiletPaperRoll",
    "Tomato",
    "TomatoSliced",
    "Towel",
    "TowelHolder",
    "Vase",
    "Watch",
    "WateringCan",
    "Window",
    "WineBottle",
)

_PROMPT_OVERRIDES = {
    "CD": "compact disc",
    "TVStand": "television stand",
    "Glassbottle": "glass bottle",
    "BasketBall": "basketball",
    "Footstool": "foot stool",
}


def category_prompt(category: str) -> str:
    """Translate a fixed simulator schema label to an unambiguous detection phrase."""
    if category not in ALFRED_CATEGORIES:
        raise ValueError(f"category={category!r} is not in the fixed ALFRED vocabulary")
    if category.endswith("Sliced"):
        return "sliced " + category[:-6].lower()
    return _PROMPT_OVERRIDES.get(
        category, re.sub(r"(?<=[a-z])(?=[A-Z])", " ", category).lower()
    )


def detection_batches(batch_size: int) -> tuple[tuple[str, ...], ...]:
    """Return deterministic full-vocabulary batches without task-driven filtering."""
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError(
            f"caption_batch_size={batch_size!r} must be an integer in [1, 32]"
        )
    return tuple(
        ALFRED_CATEGORIES[start : start + batch_size]
        for start in range(0, len(ALFRED_CATEGORIES), batch_size)
    )


def vocabulary_sha256() -> str:
    """Fingerprint canonical labels and actual detector prompts together."""
    payload = [(category, category_prompt(category)) for category in ALFRED_CATEGORIES]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()
