"""Tests for ALFRED single-turn policy task, playbook, and action odometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codex_agent.alfred.dataset import AlfredInstruction
from codex_agent.alfred.odometry import ActionOdometry, camera_intrinsics
from codex_agent.alfred.playbook import ALFRED_TOOL_NAMES, ALFRED_TOOLS_PLAYBOOK
from codex_agent.alfred.policy import (
    AlfredEpisodeTurnOutcome,
    AlfredPolicyTask,
)
from codex_agent.errors import CodexResponseError


def test_playbook_documents_all_registered_alfred_tools() -> None:
    for tool_name in ALFRED_TOOL_NAMES:
        assert tool_name in ALFRED_TOOLS_PLAYBOOK


def test_policy_task_builds_map_first_prompt_and_parses_outcome(
    tmp_path: Path,
) -> None:
    rgb = tmp_path / "initial_rgb.png"
    bev = tmp_path / "initial_bev.png"
    rgb.write_bytes(b"rgb")
    bev.write_bytes(b"bev")
    task = AlfredPolicyTask(
        instruction=AlfredInstruction(
            goal="Rinse off a ladle and move it to the table.",
            steps=(
                "Pick up the ladle.",
                "Clean it in the sink.",
                "Put it on the table.",
            ),
        ),
        episode_dir=tmp_path,
        initial_context={
            "agent_state": {"position_xy": [0.0, 0.0], "region": [0, 0]},
            "object_catalog": [{"object_id": 0, "category": "Ladle"}],
        },
        image_paths=(rgb, bev),
        use_step_instructions=True,
    )
    req = task.build_turn_request()
    assert "Rinse off a ladle and move it to the table." in req.prompt
    assert "Clean it in the sink." in req.prompt
    assert "codex_agent.alfred.tools" in req.prompt
    assert req.image_paths == (rgb, bev)

    raw_response = (
        '{"status": "completed", "summary": "Cleaned ladle #0 and placed on table #1", '
        '"completed_subgoals": ["pickup ladle", "clean in sink", "place on table"]}'
    )
    assert task.is_valid_response(raw_response)
    outcome = task.parse_response(raw_response)
    assert isinstance(outcome, AlfredEpisodeTurnOutcome)
    assert outcome.status == "completed"
    assert len(outcome.completed_subgoals) == 3

    assert not task.is_valid_response('{"status": "invalid"}')
    with pytest.raises(CodexResponseError):
        task.parse_response('{"status": "invalid"}')


def test_action_odometry_integrates_only_successful_forward_moves() -> None:
    odom = ActionOdometry()
    odom.update("MoveAhead", success=True, yaw_degrees=0.0)
    assert odom.agent_position == (0.0, 0.25, 0.0)
    odom.update("MoveAhead", success=False, yaw_degrees=0.0)
    assert odom.agent_position == (0.0, 0.25, 0.0)
    odom.update("RotateRight", success=True, yaw_degrees=90.0)
    odom.update("MoveAhead", success=True, yaw_degrees=90.0)
    assert odom.agent_position == (0.25, 0.25, 0.0)


def test_camera_intrinsics_and_extrinsics_remain_orthonormal() -> None:
    intrinsics = camera_intrinsics(width=300, height=300, vertical_fov_degrees=60.0)
    assert intrinsics.shape == (3, 3)
    assert intrinsics[0, 0] == pytest.approx(intrinsics[1, 1])
    odom = ActionOdometry()
    c2w = odom.camera_to_world(yaw_degrees=90.0, horizon_degrees=30.0)
    assert c2w.shape == (4, 4)
    np.testing.assert_allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-6)
