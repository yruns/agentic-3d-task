"""Image-attached, structured ALFRED decisions using the shared Codex runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

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
WireOperation = Literal[
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
    "NavigateObject",
    "NavigateCell",
    "RetrieveObject",
    "RetrieveCell",
    "Explore",
    "Stop",
]
NAVIGATION_ACTIONS = frozenset(
    {"MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"}
)
INTERACTION_ACTIONS = frozenset(
    {
        "PickupObject",
        "PutObject",
        "OpenObject",
        "CloseObject",
        "ToggleObjectOn",
        "ToggleObjectOff",
        "SliceObject",
    }
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
    motion_action: MotionAction
    repeat: int = Field(ge=1, le=8)


class InteractionCommand(_DecisionModel):
    """Interact once using the target instance's current image mask."""

    kind: Literal["interact"]
    interaction_action: InteractionAction
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


class ExploreCommand(_DecisionModel):
    """Observe a reachable frontier using the live occupancy and coverage maps."""

    kind: Literal["explore"]


class AlfredDecision(_DecisionModel):
    """Internal typed command used by execution; never the upstream wire schema.

    Conversion from :class:`AlfredWireDecision` selects one command and target
    explicitly. These nested domain types protect the executor from invalid
    combinations without depending on upstream support for schema unions.
    """

    command: (
        MotionCommand
        | InteractionCommand
        | NavigateCommand
        | RetrieveCommand
        | ExploreCommand
        | StopCommand
    )
    rationale: str = Field(min_length=1, max_length=1600)
    memory_note: str = Field(max_length=2400)

    @property
    def kind(self) -> Literal["act", "navigate", "retrieve", "explore", "stop"]:
        """Return the executor's dispatch family."""
        if isinstance(self.command, (MotionCommand, InteractionCommand)):
            return "act"
        return self.command.kind

    @property
    def action(self) -> PrimitiveAction | None:
        """Return a simulator primitive only for motion/interaction commands."""
        if isinstance(self.command, MotionCommand):
            return self.command.motion_action
        if isinstance(self.command, InteractionCommand):
            return self.command.interaction_action
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

    def to_wire(self) -> AlfredWireDecision:
        """Encode one command with canonical zeros for unused scalar parameters."""
        operation: WireOperation
        if isinstance(self.command, MotionCommand):
            operation = self.command.motion_action
        elif isinstance(self.command, InteractionCommand):
            operation = self.command.interaction_action
        elif isinstance(self.command, NavigateCommand):
            operation = (
                "NavigateObject"
                if isinstance(self.command.target, ObjectTarget)
                else "NavigateCell"
            )
        elif isinstance(self.command, RetrieveCommand):
            operation = (
                "RetrieveObject"
                if isinstance(self.command.target, ObjectTarget)
                else "RetrieveCell"
            )
        elif isinstance(self.command, ExploreCommand):
            operation = "Explore"
        else:
            operation = "Stop"
        cell = self.cell
        return AlfredWireDecision(
            operation=operation,
            object_id=self.object_id if self.object_id is not None else 0,
            cell_x=cell[0] if cell is not None else 0,
            cell_y=cell[1] if cell is not None else 0,
            repeat=self.repeat,
            rationale=self.rationale,
            memory_note=self.memory_note,
        )


class AlfredWireDecision(_DecisionModel):
    """Flat scalar wire contract for the registered provider's constrained decoder.

    ``operation`` alone selects the applicable parameters. Object operations
    consume object_id; cell operations consume cell_x/cell_y; motion operations
    consume repeat. Other parameter values have no effect by this contract.
    There are no nullable values, unions, references or const schema keywords.
    """

    operation: WireOperation
    object_id: int = Field(ge=0)
    cell_x: int
    cell_y: int
    repeat: int = Field(ge=1, le=8)
    rationale: str = Field(min_length=1, max_length=1600)
    memory_note: str = Field(max_length=2400)

    def to_decision(self) -> AlfredDecision:
        """Select exactly one domain command according to the declared operation."""
        command: (
            MotionCommand
            | InteractionCommand
            | NavigateCommand
            | RetrieveCommand
            | ExploreCommand
            | StopCommand
        )
        if self.operation in NAVIGATION_ACTIONS:
            command = MotionCommand(
                kind="motion",
                motion_action=cast(MotionAction, self.operation),
                repeat=self.repeat,
            )
        elif self.operation in INTERACTION_ACTIONS:
            command = InteractionCommand(
                kind="interact",
                interaction_action=cast(InteractionAction, self.operation),
                target=ObjectTarget(type="object", object_id=self.object_id),
            )
        elif self.operation == "NavigateObject":
            command = NavigateCommand(
                kind="navigate",
                target=ObjectTarget(type="object", object_id=self.object_id),
            )
        elif self.operation == "NavigateCell":
            command = NavigateCommand(
                kind="navigate",
                target=CellTarget(type="cell", cell=[self.cell_x, self.cell_y]),
            )
        elif self.operation == "RetrieveObject":
            command = RetrieveCommand(
                kind="retrieve",
                target=ObjectTarget(type="object", object_id=self.object_id),
            )
        elif self.operation == "RetrieveCell":
            command = RetrieveCommand(
                kind="retrieve",
                target=CellTarget(type="cell", cell=[self.cell_x, self.cell_y]),
            )
        elif self.operation == "Stop":
            command = StopCommand(kind="stop")
        elif self.operation == "Explore":
            command = ExploreCommand(kind="explore")
        else:
            raise ValueError(f"unsupported ALFRED wire operation: {self.operation!r}")
        return AlfredDecision(
            command=command, rationale=self.rationale, memory_note=self.memory_note
        )


_PLAYBOOK = """You control an embodied ALFRED agent. Use only the supplied language,
images, observed scene memory and action feedback. Detection labels and depth are
predictions, and instance matching can be wrong. The map starts empty. Unknown
space is unexplored, not an obstacle and not evidence that a target is absent.
Images: current labeled RGB, observed BEV, then any explicitly retrieved past
views (with frame identifiers). Historical images cannot supply a current mask.
MoveAhead moves 0.25 m. RotateLeft/Right turn 90 degrees. LookUp/Down adjust pitch
15 degrees. Positive camera horizon looks down. Repeat allows up to 8 navigation
primitives; execution stops early on failure and updates memory at EVERY step.
Return flat JSON with operation, object_id, cell_x, cell_y, repeat, rationale,
and memory_note. operation is one native action name, NavigateObject, NavigateCell,
RetrieveObject, RetrieveCell, Explore, or Stop. Native interactions and *Object operations
use object_id; *Cell operations use cell_x/cell_y. Motion actions alone use repeat;
every other operation executes once. Set unused object/cell parameters to 0 and
unused repeat to 1. The operation explicitly selects which parameters apply.
Example: {"operation":"RotateLeft","object_id":0,"cell_x":0,"cell_y":0,
"repeat":1,"rationale":"Inspect the left side","memory_note":"Continue search"}.
For interactions use object_id from the CURRENT visible detections. A predicted
mask from this exact pose is supplied to the official interaction interface.
PutObject targets the destination receptacle, not the held item. Pick up a knife
before slicing; put it down before picking up the sliced food. Heat: put inside
microwave, close, toggle on, toggle off, open, retrieve. Cool: put inside fridge,
close, open, retrieve. Clean: place in sink basin, toggle faucet on then off,
retrieve. Hold the item and switch on the lamp for a look-at-in-light task.
Object IDs remain stable; merged aliases resolve to the surviving instance.
Provisional objects have limited observations, not confirmed absence or certainty.
Action-derived states record successful feedback, not an independent visual check.
Navigate requests a route through observed free cells, to a coarse map region or a known
object's vicinity. It does not ensure interaction readiness; inspect the returned
RGB and adjust position/pitch. If no observed route exists, use Explore.
Explore commits to a reachable frontier, follows its route, then observes new
directions. Coverage and navigation are separate: seen need not mean free.
Do not repeatedly select an unreachable object or completed inspection position.
Retrieve asks for complementary past evidence for an object or
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
            output_schema=AlfredWireDecision.model_json_schema(),
            image_paths=self.image_paths,
        )

    def parse_response(self, response_text: str) -> AlfredDecision:
        try:
            return AlfredWireDecision.model_validate(
                extract_json_object(response_text)
            ).to_decision()
        except (ValueError, ValidationError) as exc:
            raise CodexResponseError(f"invalid ALFRED decision: {exc}") from exc

    def is_valid_response(self, response_text: str) -> bool:
        try:
            self.parse_response(response_text)
        except CodexResponseError:
            return False
        return True
