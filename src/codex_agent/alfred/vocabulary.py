"""Frozen ALFRED task-relevant object vocabulary (85 categories).

Derived from an exhaustive audit of all 7,080 unique trajectories (22,664
language annotations) across ``train`` (6,574), ``valid_seen`` (251), and
``valid_unseen`` (255) in ALFRED ``json_2.1.0``, plus ``gen/constants.py`` and
FILM's ``alfworld_constants.py`` detector pruning rules:
* 53 manipulable ``object_target`` base classes + 5 ``*Sliced`` forms = 58 classes
  (including the 7 ``mrecep_target`` movable receptacles),
* 24 stationary ``parent_target`` receptacles/surfaces,
* 2 ``toggle_target`` lamps (``DeskLamp``, ``FloorLamp``) + ``Faucet`` = 3 control classes.

The 29 non-task structural/background/duplicate-shell classes from the raw
AI2-THOR 108-object modeling list (e.g. ``Window``, ``Blinds``, ``Curtains``,
``Mirror``, ``Painting``, ``Poster``, ``ShowerDoor``, ``ShowerGlass``,
``LightSwitch``, ``StoveKnob``, ``Chair``, ``Footstool``, ``TVStand``,
``Television``, ``Sink`` [duplicate of ``SinkBasin``], ``Bathtub`` [duplicate
of ``BathtubBasin``], etc.) never appear in any PDDL goal or expert interaction
and are excluded so open-vocabulary 2D detection does not hallucinate background
surfaces or hollow out container masks during contained-mask subtraction.
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
    "BathtubBasin",
    "Bed",
    "Book",
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
    "Cloth",
    "CoffeeMachine",
    "CoffeeTable",
    "CounterTop",
    "CreditCard",
    "Cup",
    "Desk",
    "DeskLamp",
    "DiningTable",
    "DishSponge",
    "Drawer",
    "Dresser",
    "Egg",
    "Faucet",
    "FloorLamp",
    "Fork",
    "Fridge",
    "GarbageCan",
    "Glassbottle",
    "HandTowel",
    "Kettle",
    "KeyChain",
    "Knife",
    "Ladle",
    "Laptop",
    "Lettuce",
    "LettuceSliced",
    "Microwave",
    "Mug",
    "Newspaper",
    "Ottoman",
    "Pan",
    "Pen",
    "Pencil",
    "PepperShaker",
    "Pillow",
    "Plate",
    "Plunger",
    "Pot",
    "Potato",
    "PotatoSliced",
    "RemoteControl",
    "Safe",
    "SaltShaker",
    "Shelf",
    "SideTable",
    "SinkBasin",
    "SoapBar",
    "SoapBottle",
    "Sofa",
    "Spatula",
    "Spoon",
    "SprayBottle",
    "Statue",
    "StoveBurner",
    "TennisRacket",
    "TissueBox",
    "Toilet",
    "ToiletPaper",
    "ToiletPaperHanger",
    "Tomato",
    "TomatoSliced",
    "Vase",
    "Watch",
    "WateringCan",
    "WineBottle",
)

EXCLUDED_BACKGROUND_CATEGORIES: tuple[str, ...] = (
    "Bathtub",
    "Blinds",
    "Boots",
    "Chair",
    "Curtains",
    "Footstool",
    "HandTowelHolder",
    "HousePlant",
    "LaundryHamper",
    "LaundryHamperLid",
    "LightSwitch",
    "Mirror",
    "Painting",
    "PaperTowel",
    "PaperTowelRoll",
    "Poster",
    "ScrubBrush",
    "ShowerDoor",
    "ShowerGlass",
    "Sink",
    "StoveKnob",
    "TVStand",
    "TeddyBear",
    "Television",
    "Toaster",
    "ToiletPaperRoll",
    "Towel",
    "TowelHolder",
    "Window",
)

_PROMPT_OVERRIDES = {
    "CD": "compact disc",
    "Glassbottle": "glass bottle",
    "BasketBall": "basketball",
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
