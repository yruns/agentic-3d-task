"""Image-attached, structured ALFRED decisions using the shared Codex runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from codex_agent.errors import CodexResponseError
from codex_agent.json_extraction import extract_json_object
from codex_agent.models import CodexTurnRequest

from .actor_runtime import AlfredPolicyRuntime as AlfredPolicyRuntime
from .dataset import AlfredInstruction

PrimitiveAction = Literal[
    "MoveAhead",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
]
NAVIGATION_ACTIONS = frozenset(
    {"MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"}
)


class AlfredDecision(BaseModel):
    """One bounded action, map-navigation request, evidence query, or stop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["act", "navigate", "retrieve", "stop"]
    action: PrimitiveAction | None
    object_id: int | None = Field(ge=0)
    cell: list[int] | None = Field(min_length=2, max_length=2)
    repeat: int = Field(ge=1, le=8)
    rationale: str = Field(min_length=1, max_length=1600)
    memory_note: str = Field(max_length=2400)

    @model_validator(mode="after")
    def validate_request(self) -> AlfredDecision:
        if self.kind == "act":
            if self.action is None or self.cell is not None:
                raise ValueError("act requires action and cell=null")
            if self.action in NAVIGATION_ACTIONS:
                if self.object_id is not None:
                    raise ValueError("navigation primitives require object_id=null")
            elif self.object_id is None or self.repeat != 1:
                raise ValueError("interaction requires object_id and repeat=1")
        else:
            if self.action is not None or self.repeat != 1:
                raise ValueError("non-act decisions require action=null and repeat=1")
            if self.kind in {"navigate", "retrieve"}:
                if (self.object_id is None) == (self.cell is None):
                    raise ValueError(
                        "navigate/retrieve requires exactly one object_id or cell"
                    )
            elif self.object_id is not None or self.cell is not None:
                raise ValueError("stop requires object_id=null and cell=null")
        return self


_PLAYBOOK = """You control an embodied ALFRED agent. Use only the supplied language,
images, observed scene memory and action feedback. Detection labels and depth are
predictions, and instance matching can be wrong. The map starts empty. Unknown
space is unexplored, not an obstacle and not evidence that a target is absent.
Images: current labeled RGB, observed BEV, then any explicitly retrieved past
views (with frame identifiers). Historical images cannot supply a current mask.
MoveAhead moves 0.25 m. RotateLeft/Right turn 90 degrees. LookUp/Down adjust pitch
15 degrees. Positive camera horizon looks down. Repeat allows up to 8 navigation
primitives; execution stops early on failure and updates memory at EVERY step.
For interactions use object_id from the CURRENT visible detections. A predicted
mask from this exact pose is supplied to the official interaction interface.
PutObject targets the destination receptacle, not the held item. Pick up a knife
before slicing; put it down before picking up the sliced food. Heat: put inside
microwave, close, toggle on, toggle off, open, retrieve. Cool: put inside fridge,
close, open, retrieve. Clean: place in sink basin, toggle faucet on then off,
retrieve. Hold the item and switch on the lamp for a look-at-in-light task.
Navigate requests a route through observed free cells, to a grid cell or a known
object's vicinity. It does not ensure interaction readiness; inspect the returned
RGB and adjust position/pitch. If no observed route exists, explore with short
primitive moves. Retrieve asks for complementary past evidence for an object or
grid cell, without moving. Do not repeatedly retrieve the same evidence.
Stop when visual/action evidence indicates all requested state changes are done.
You cannot inspect the hidden scorer. Never infer success from your intention.
memory_note carries your concise plan, completed steps and remaining uncertainty
to the next independent decision. Return exactly the requested JSON schema.
"""


@dataclass(frozen=True)
class AlfredPolicyTask:
    """Actor input deliberately excludes trajectory paths and evaluator labels."""

    instruction: AlfredInstruction
    context: dict[str, object]
    image_paths: tuple[Path, ...]
    use_step_instructions: bool = False

    @property
    def task_name(self) -> str:
        return "alfred_incremental_execution"

    def build_turn_request(self) -> CodexTurnRequest:
        language: dict[str, object] = {"goal": self.instruction.goal}
        if self.use_step_instructions:
            language["step_instructions"] = self.instruction.steps
        return CodexTurnRequest(
            prompt=_PLAYBOOK
            + "\n"
            + json.dumps(
                {"instruction": language, "observations": self.context},
                ensure_ascii=False,
                allow_nan=False,
            ),
            output_schema=AlfredDecision.model_json_schema(),
            image_paths=self.image_paths,
        )

    def parse_response(self, response_text: str) -> AlfredDecision:
        try:
            return AlfredDecision.model_validate(extract_json_object(response_text))
        except (ValueError, ValidationError) as exc:
            raise CodexResponseError(f"invalid ALFRED decision: {exc}") from exc

    def is_valid_response(self, response_text: str) -> bool:
        try:
            self.parse_response(response_text)
        except CodexResponseError:
            return False
        return True
