"""Image-attached, structured ALFRED decisions using the shared Codex runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from codex_agent.errors import CodexResponseError
from codex_agent.json_extraction import extract_json_object
from codex_agent.models import CodexTurnRequest

from .actor_runtime import AlfredPolicyRuntime as AlfredPolicyRuntime
from .dataset import AlfredInstruction

MotionAction = Literal[
    "MoveAhead",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
]
InteractionAction = Literal[
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
]
PrimitiveAction = MotionAction | InteractionAction
NAVIGATION_ACTIONS = frozenset(
    {"MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"}
)


class _DecisionModel(BaseModel):
    """Forbid additional fields and coercion at every policy-command boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ObjectTarget(_DecisionModel):
    """Reference exactly one observed memory instance."""

    type: Literal["object"]
    object_id: int = Field(ge=0)


class CellTarget(_DecisionModel):
    """Reference exactly one stable world-grid cell."""

    type: Literal["cell"]
    cell: list[int] = Field(min_length=2, max_length=2)


class MotionCommand(_DecisionModel):
    """Execute a bounded sequence of one navigation primitive."""

    kind: Literal["motion"]
    action: MotionAction
    repeat: int = Field(ge=1, le=8)


class InteractionCommand(_DecisionModel):
    """Interact once using the target instance's current image mask."""

    kind: Literal["interact"]
    action: InteractionAction
    target: ObjectTarget


class NavigateCommand(_DecisionModel):
    """Approach an object or a grid cell through currently observed free space."""

    kind: Literal["navigate"]
    target: ObjectTarget | CellTarget


class RetrieveCommand(_DecisionModel):
    """Retrieve historical observations for exactly one object or grid cell."""

    kind: Literal["retrieve"]
    target: ObjectTarget | CellTarget


class StopCommand(_DecisionModel):
    """End actor execution without issuing a simulator action."""

    kind: Literal["stop"]


class AlfredDecision(_DecisionModel):
    """A command whose valid combinations are expressed in its wire JSON schema.

    Nested unions keep the schema root an object and expose all action/target
    constraints to structured generation. The executor properties preserve its
    existing dispatch interface without accepting or prioritizing contradictory
    fields from the previous flat response format.
    """

    command: (
        MotionCommand
        | InteractionCommand
        | NavigateCommand
        | RetrieveCommand
        | StopCommand
    )
    rationale: str = Field(min_length=1, max_length=1600)
    memory_note: str = Field(max_length=2400)

    @property
    def kind(self) -> Literal["act", "navigate", "retrieve", "stop"]:
        """Return the executor's dispatch family."""
        if isinstance(self.command, (MotionCommand, InteractionCommand)):
            return "act"
        return self.command.kind

    @property
    def action(self) -> PrimitiveAction | None:
        """Return a simulator primitive only for motion/interaction commands."""
        if isinstance(self.command, (MotionCommand, InteractionCommand)):
            return self.command.action
        return None

    @property
    def object_id(self) -> int | None:
        """Return the selected instance, if the command has an object target."""
        if isinstance(
            self.command, (InteractionCommand, NavigateCommand, RetrieveCommand)
        ):
            if isinstance(self.command.target, ObjectTarget):
                return self.command.target.object_id
        return None

    @property
    def cell(self) -> list[int] | None:
        """Return a copy of the selected grid cell, if present."""
        if isinstance(self.command, (NavigateCommand, RetrieveCommand)):
            if isinstance(self.command.target, CellTarget):
                return list(self.command.target.cell)
        return None

    @property
    def repeat(self) -> int:
        """Only motion commands can request repeated primitives."""
        return self.command.repeat if isinstance(self.command, MotionCommand) else 1


_PLAYBOOK = """You control an embodied ALFRED agent. Use only the supplied language,
images, observed scene memory and action feedback. Detection labels and depth are
predictions, and instance matching can be wrong. The map starts empty. Unknown
space is unexplored, not an obstacle and not evidence that a target is absent.
Images: current labeled RGB, observed BEV, then any explicitly retrieved past
views (with frame identifiers). Historical images cannot supply a current mask.
MoveAhead moves 0.25 m. RotateLeft/Right turn 90 degrees. LookUp/Down adjust pitch
15 degrees. Positive camera horizon looks down. Repeat allows up to 8 navigation
primitives; execution stops early on failure and updates memory at EVERY step.
Choose exactly one command: motion (action, repeat), interact (action, object
target), navigate (one target), retrieve (one target), or stop. A target is either
{type:"object",object_id:N} or {type:"cell",cell:[x,y]}, never both. Only motion
has repeat; interaction executes once. Do not include unused or null fields.
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
After a full scan from one position, prefer navigating to an observed free
approach or moving to obtain a new view over another identical scan or retrieval.
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
