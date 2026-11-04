"""Actor boundary and action-odometry regression tests."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from codex_agent.alfred.dataset import AlfredInstruction
from codex_agent.alfred.odometry import ActionOdometry, camera_intrinsics
from codex_agent.alfred.policy import AlfredDecision, AlfredPolicyTask


def test_interaction_cannot_repeat_or_use_navigation_target() -> None:
    decision = {
        "kind": "act",
        "action": "PickupObject",
        "object_id": 2,
        "cell": None,
        "repeat": 2,
        "rationale": "pick up",
        "memory_note": "",
    }
    with pytest.raises(ValidationError, match="repeat=1"):
        AlfredDecision.model_validate(decision)
    decision.update(action="MoveAhead", repeat=1)
    with pytest.raises(ValidationError, match="object_id=null"):
        AlfredDecision.model_validate(decision)


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
    assert task.is_valid_response(
        json.dumps(
            {
                "kind": "stop",
                "action": None,
                "object_id": None,
                "cell": None,
                "repeat": 1,
                "rationale": "done",
                "memory_note": "",
            }
        )
    )
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
