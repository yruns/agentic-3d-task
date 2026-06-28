"""Tests for SceneFunc3D approval state transitions."""

from __future__ import annotations

import pytest

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.task import (
    ApprovalAction,
    SceneFunc3dStage,
    SceneFunc3dState,
)


def test_valid_approval_flow() -> None:
    state = SceneFunc3dState.initial("421254::desc-a")
    state = state.transition(ApprovalAction.SELECT_EVIDENCE)
    state = state.transition(ApprovalAction.PROPOSE_MOLMO_POINT)
    state = state.transition(ApprovalAction.APPROVE_MOLMO_POINT)
    state = state.transition(ApprovalAction.PROPOSE_SAM_CANDIDATES)
    state = state.transition(ApprovalAction.APPROVE_SAM_CANDIDATE)
    state = state.transition(ApprovalAction.CREATE_FIRST_LIFT)
    state = state.transition(ApprovalAction.APPROVE_FIRST_LIFT)
    assert state.stage is SceneFunc3dStage.FIRST_LIFT_AGENT_APPROVED


def test_sam_cannot_run_before_point_approval() -> None:
    state = SceneFunc3dState.initial("421254::desc-a")
    state = state.transition(ApprovalAction.SELECT_EVIDENCE)
    state = state.transition(ApprovalAction.PROPOSE_MOLMO_POINT)

    with pytest.raises(SceneFunc3dDataError, match="invalid transition"):
        state.transition(ApprovalAction.PROPOSE_SAM_CANDIDATES)


def test_lift_cannot_run_before_sam_approval() -> None:
    state = SceneFunc3dState.initial("421254::desc-a")
    state = state.transition(ApprovalAction.SELECT_EVIDENCE)
    state = state.transition(ApprovalAction.PROPOSE_MOLMO_POINT)
    state = state.transition(ApprovalAction.APPROVE_MOLMO_POINT)
    state = state.transition(ApprovalAction.PROPOSE_SAM_CANDIDATES)

    with pytest.raises(SceneFunc3dDataError, match="invalid transition"):
        state.transition(ApprovalAction.CREATE_FIRST_LIFT)
