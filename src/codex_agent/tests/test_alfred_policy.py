"""Actor boundary and action-odometry regression tests."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from codex_agent.alfred.dataset import AlfredInstruction
from codex_agent.alfred.odometry import ActionOdometry, camera_intrinsics
from codex_agent.alfred.policy import (
    AlfredDecision,
    AlfredPolicyTask,
    AlfredWireDecision,
)


def _decision(command: dict[str, object]) -> AlfredDecision:
    return AlfredDecision.model_validate(
        {"command": command, "rationale": "Use observed evidence", "memory_note": ""}
    )


@pytest.mark.parametrize(
    "command,expected",
    [
        (
            {"kind": "motion", "motion_action": "MoveAhead", "repeat": 3},
            ("act", "MoveAhead", None, None, 3),
        ),
        (
            {
                "kind": "interact",
                "interaction_action": "PickupObject",
                "target": {"type": "object", "object_id": 2},
            },
            ("act", "PickupObject", 2, None, 1),
        ),
        (
            {"kind": "navigate", "target": {"type": "object", "object_id": 22}},
            ("navigate", None, 22, None, 1),
        ),
        (
            {"kind": "navigate", "target": {"type": "cell", "cell": [5, -5]}},
            ("navigate", None, None, [5, -5], 1),
        ),
        (
            {"kind": "retrieve", "target": {"type": "object", "object_id": 3}},
            ("retrieve", None, 3, None, 1),
        ),
        (
            {"kind": "retrieve", "target": {"type": "cell", "cell": [1, 2]}},
            ("retrieve", None, None, [1, 2], 1),
        ),
        ({"kind": "stop"}, ("stop", None, None, None, 1)),
    ],
)
def test_commands_preserve_executor_interface(
    command: dict[str, object], expected: tuple[object, ...]
) -> None:
    decision = _decision(command)
    assert (
        decision.kind,
        decision.action,
        decision.object_id,
        decision.cell,
        decision.repeat,
    ) == expected
    assert decision.to_wire().to_decision() == decision


@pytest.mark.parametrize(
    "command",
    [
        {
            "kind": "interact",
            "interaction_action": "PickupObject",
            "target": {"type": "object", "object_id": 2},
            "repeat": 2,
        },
        {
            "kind": "motion",
            "motion_action": "MoveAhead",
            "target": {"type": "object", "object_id": 2},
            "repeat": 1,
        },
        {
            "kind": "interact",
            "interaction_action": "MoveAhead",
            "target": {"type": "object", "object_id": 2},
        },
        {
            "kind": "interact",
            "interaction_action": "PickupObject",
            "target": {"type": "cell", "cell": [1, 2]},
        },
        {
            "kind": "navigate",
            "target": {"type": "object", "object_id": 22, "cell": [5, -5]},
        },
        {
            "kind": "navigate",
            "target": {"type": "cell", "object_id": 22, "cell": [5, -5]},
        },
        {"kind": "navigate", "target": None},
        {"kind": "stop", "target": {"type": "object", "object_id": 22}},
    ],
)
def test_schema_commands_reject_ambiguous_or_inapplicable_fields(
    command: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _decision(command)


def test_real_dual_target_navigation_failure_is_not_silently_reinterpreted() -> None:
    failed_response = {
        "kind": "navigate",
        "action": None,
        "object_id": 22,
        "cell": [5, -5],
        "repeat": 1,
        "rationale": "Approach the target",
        "memory_note": "",
    }
    with pytest.raises(ValidationError):
        AlfredDecision.model_validate(failed_response)


def test_domain_schema_encodes_mutual_exclusion_without_hidden_cross_field_rules() -> (
    None
):
    schema = AlfredDecision.model_json_schema()
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"command", "rationale", "memory_note"}
    assert "anyOf" in schema["properties"]["command"]
    definitions = schema["$defs"]
    assert "cell" not in definitions["ObjectTarget"]["properties"]
    assert "object_id" not in definitions["CellTarget"]["properties"]
    for model in definitions.values():
        assert model["additionalProperties"] is False
        assert set(model["required"]) == set(model["properties"])
    encoded = json.dumps(schema)
    for unsupported in ("prefixItems", "discriminator", "oneOf", "if", "then"):
        assert f'"{unsupported}":' not in encoded


def test_action_field_names_do_not_collide_across_union_branches() -> None:
    definitions = AlfredDecision.model_json_schema()["$defs"]
    motion = definitions["MotionCommand"]["properties"]
    interaction = definitions["InteractionCommand"]["properties"]
    assert "action" not in motion and "action" not in interaction
    assert "interaction_action" not in motion
    assert "motion_action" not in interaction
    assert set(motion["motion_action"]["enum"]) == {
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
    }
    assert set(motion["motion_action"]["enum"]).isdisjoint(
        interaction["interaction_action"]["enum"]
    )


def test_legacy_shared_action_field_is_not_silently_translated() -> None:
    with pytest.raises(ValidationError):
        _decision({"kind": "motion", "action": "RotateLeft", "repeat": 1})


def _wire(**updates: object) -> AlfredWireDecision:
    return AlfredWireDecision.model_validate(
        {
            "operation": "Stop",
            "object_id": 0,
            "cell_x": 0,
            "cell_y": 0,
            "repeat": 1,
            "rationale": "Protocol test",
            "memory_note": "Test",
            **updates,
        }
    )


def test_actor_wire_schema_is_flat_and_has_no_union_or_reference_keywords(
    tmp_path: Path,
) -> None:
    task = AlfredPolicyTask(AlfredInstruction("Test", ()), {}, ())
    schema = dict(task.build_turn_request().output_schema)
    assert schema == AlfredWireDecision.model_json_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert len(schema["properties"]["operation"]["enum"]) == 17
    for field in schema["properties"].values():
        assert field["type"] in {"string", "integer"}
    encoded = json.dumps(schema)
    for unsupported in ("anyOf", "oneOf", "$ref", "$defs", "const"):
        assert f'"{unsupported}":' not in encoded


@pytest.mark.parametrize(
    "operation",
    [
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
        "Stop",
    ],
)
def test_all_wire_operations_round_trip_without_changing_selected_command(
    operation: str,
) -> None:
    wire = _wire(operation=operation, object_id=22, cell_x=5, cell_y=-5, repeat=3)
    decision = wire.to_decision()
    canonical = decision.to_wire()
    assert canonical.operation == operation
    assert canonical.to_decision() == decision
    assert canonical.rationale == wire.rationale
    assert canonical.memory_note == wire.memory_note


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("NavigateObject", ("navigate", None, 22, None, 1)),
        ("NavigateCell", ("navigate", None, None, [5, -5], 1)),
        ("RetrieveObject", ("retrieve", None, 22, None, 1)),
        ("RetrieveCell", ("retrieve", None, None, [5, -5], 1)),
        ("PickupObject", ("act", "PickupObject", 22, None, 1)),
        ("RotateLeft", ("act", "RotateLeft", None, None, 7)),
        ("Stop", ("stop", None, None, None, 1)),
    ],
)
def test_operation_explicitly_selects_parameters_and_ignores_unused_fields(
    operation: str, expected: tuple[object, ...]
) -> None:
    decision = _wire(
        operation=operation, object_id=22, cell_x=5, cell_y=-5, repeat=7
    ).to_decision()
    assert (
        decision.kind,
        decision.action,
        decision.object_id,
        decision.cell,
        decision.repeat,
    ) == expected
    canonical = decision.to_wire()
    if decision.object_id is None:
        assert canonical.object_id == 0
    if decision.cell is None:
        assert (canonical.cell_x, canonical.cell_y) == (0, 0)


@pytest.mark.parametrize(
    "updates",
    [
        {"operation": "act"},
        {"object_id": True},
        {"cell_x": 1.5},
        {"repeat": "2"},
        {"repeat": 0},
        {"repeat": 9},
    ],
)
def test_flat_wire_rejects_unknown_operations_and_value_coercion(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _wire(**updates)


@pytest.mark.parametrize(
    "target",
    [
        {"type": "object", "object_id": True},
        {"type": "cell", "cell": [1.5, 2]},
        {"type": "cell", "cell": [1, 2, 3]},
    ],
)
def test_targets_reject_coercion_and_invalid_dimensions(
    target: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _decision({"kind": "navigate", "target": target})


def test_policy_uses_only_language_and_supplied_observations(tmp_path: Path) -> None:
    task = AlfredPolicyTask(
        AlfredInstruction("Put a mug away", ("Private step",)),
        {"frame": 1},
        (tmp_path / "rgb.png",),
    )
    request = task.build_turn_request()
    assert "Put a mug away" in request.prompt
    assert "Private step" not in request.prompt
    assert "traj_data" not in request.prompt
    assert task.is_valid_response(_wire().model_dump_json())
    assert not task.is_valid_response(_decision({"kind": "stop"}).model_dump_json())
    assert not task.is_valid_response('{"kind":"stop"}')


def test_odometry_does_not_move_on_collision_and_pitch_projects_down() -> None:
    odometry = ActionOdometry()
    odometry.update("MoveAhead", success=False, yaw_degrees=90)
    assert odometry.agent_position == (0.0, 0.0, 0.0)
    odometry.update("MoveAhead", success=True, yaw_degrees=90)
    assert odometry.x == pytest.approx(0.25)
    transform = odometry.camera_to_world(yaw_degrees=0, horizon_degrees=45)
    assert transform[2, 2] < 0
    np.testing.assert_allclose(
        transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12
    )
    assert camera_intrinsics()[0, 0] == pytest.approx(150 / np.tan(np.pi / 6))


def test_cardinal_odometry_has_no_signed_epsilon_cell_drift() -> None:
    odometry = ActionOdometry()
    odometry.update("MoveAhead", success=True, yaw_degrees=270)
    assert odometry.agent_position == (-0.25, 0.0, 0.0)
    for heading in (90, 0, 180):
        odometry.update("MoveAhead", success=True, yaw_degrees=heading)
    assert odometry.agent_position == (0.0, 0.0, 0.0)
