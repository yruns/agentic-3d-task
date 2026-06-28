"""Tests for SceneFunc3D approval state transitions."""

from __future__ import annotations

import pytest

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.task import (
    ApprovalAction,
    SceneFunc3dStage,
    SceneFunc3dState,
)


def _first_lift_created_state() -> SceneFunc3dState:
    state = SceneFunc3dState.initial("421254::desc-a")
    state = state.transition(ApprovalAction.SELECT_EVIDENCE)
    state = state.transition(ApprovalAction.PROPOSE_MOLMO_POINT)
    state = state.transition(ApprovalAction.APPROVE_MOLMO_POINT)
    state = state.transition(ApprovalAction.PROPOSE_SAM_CANDIDATES)
    state = state.transition(ApprovalAction.APPROVE_SAM_CANDIDATE)
    return state.transition(ApprovalAction.CREATE_FIRST_LIFT)


def _first_lift_approved_state() -> SceneFunc3dState:
    return _first_lift_created_state().transition(ApprovalAction.APPROVE_FIRST_LIFT)


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


def test_first_lift_approval_can_create_fused_mask_and_emit_final_answer() -> None:
    state = _first_lift_approved_state()

    state = state.transition(ApprovalAction.CREATE_FUSED_MASK)
    state = state.transition(ApprovalAction.EMIT_FINAL_ANSWER)

    assert state.stage is SceneFunc3dStage.FINAL_ANSWER


def test_multiview_expansion_can_create_fused_mask_and_emit_final_answer() -> None:
    state = _first_lift_approved_state()

    state = state.transition(ApprovalAction.START_MULTIVIEW_EXPANSION)
    state = state.transition(ApprovalAction.CREATE_FUSED_MASK)
    state = state.transition(ApprovalAction.EMIT_FINAL_ANSWER)

    assert state.stage is SceneFunc3dStage.FINAL_ANSWER


def test_fused_mask_cannot_be_created_before_first_lift_approval() -> None:
    state = _first_lift_created_state()

    with pytest.raises(SceneFunc3dDataError, match="invalid transition"):
        state.transition(ApprovalAction.CREATE_FUSED_MASK)


def test_final_answer_cannot_emit_before_fused_mask() -> None:
    state = _first_lift_approved_state()

    with pytest.raises(SceneFunc3dDataError, match="invalid transition"):
        state.transition(ApprovalAction.EMIT_FINAL_ANSWER)
