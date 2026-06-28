"""SceneFunc3D agent approval state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SceneFunc3dDataError


class SceneFunc3dStage(str, Enum):
    """Ordered stages in the SceneFunc3D mask-generation workflow."""

    INITIAL = "initial"
    EVIDENCE_SELECTED = "evidence_selected"
    MOLMO_POINT_PROPOSED = "molmo_point_proposed"
    MOLMO_POINT_AGENT_APPROVED = "molmo_point_agent_approved"
    SAM_CANDIDATES_PROPOSED = "sam_candidates_proposed"
    SAM_CANDIDATE_AGENT_APPROVED = "sam_candidate_agent_approved"
    FIRST_LIFT_CREATED = "first_lift_created"
    FIRST_LIFT_AGENT_APPROVED = "first_lift_agent_approved"
    OPTIONAL_MULTIVIEW_EXPANSION = "optional_multiview_expansion"
    FUSED_MASK_CREATED = "fused_mask_created"
    FINAL_ANSWER = "final_answer"


class ApprovalAction(str, Enum):
    """Agent or tool actions that move the SceneFunc3D workflow forward."""

    SELECT_EVIDENCE = "select_evidence"
    PROPOSE_MOLMO_POINT = "propose_molmo_point"
    APPROVE_MOLMO_POINT = "approve_molmo_point"
    PROPOSE_SAM_CANDIDATES = "propose_sam_candidates"
    APPROVE_SAM_CANDIDATE = "approve_sam_candidate"
    CREATE_FIRST_LIFT = "create_first_lift"
    APPROVE_FIRST_LIFT = "approve_first_lift"
    START_MULTIVIEW_EXPANSION = "start_multiview_expansion"
    CREATE_FUSED_MASK = "create_fused_mask"
    EMIT_FINAL_ANSWER = "emit_final_answer"


_TRANSITIONS: dict[tuple[SceneFunc3dStage, ApprovalAction], SceneFunc3dStage] = {
    (SceneFunc3dStage.INITIAL, ApprovalAction.SELECT_EVIDENCE): (
        SceneFunc3dStage.EVIDENCE_SELECTED
    ),
    (SceneFunc3dStage.EVIDENCE_SELECTED, ApprovalAction.PROPOSE_MOLMO_POINT): (
        SceneFunc3dStage.MOLMO_POINT_PROPOSED
    ),
    (SceneFunc3dStage.MOLMO_POINT_PROPOSED, ApprovalAction.APPROVE_MOLMO_POINT): (
        SceneFunc3dStage.MOLMO_POINT_AGENT_APPROVED
    ),
    (
        SceneFunc3dStage.MOLMO_POINT_AGENT_APPROVED,
        ApprovalAction.PROPOSE_SAM_CANDIDATES,
    ): SceneFunc3dStage.SAM_CANDIDATES_PROPOSED,
    (
        SceneFunc3dStage.SAM_CANDIDATES_PROPOSED,
        ApprovalAction.APPROVE_SAM_CANDIDATE,
    ): SceneFunc3dStage.SAM_CANDIDATE_AGENT_APPROVED,
    (
        SceneFunc3dStage.SAM_CANDIDATE_AGENT_APPROVED,
        ApprovalAction.CREATE_FIRST_LIFT,
    ): SceneFunc3dStage.FIRST_LIFT_CREATED,
    (SceneFunc3dStage.FIRST_LIFT_CREATED, ApprovalAction.APPROVE_FIRST_LIFT): (
        SceneFunc3dStage.FIRST_LIFT_AGENT_APPROVED
    ),
    (
        SceneFunc3dStage.FIRST_LIFT_AGENT_APPROVED,
        ApprovalAction.START_MULTIVIEW_EXPANSION,
    ): SceneFunc3dStage.OPTIONAL_MULTIVIEW_EXPANSION,
    (
        SceneFunc3dStage.FIRST_LIFT_AGENT_APPROVED,
        ApprovalAction.CREATE_FUSED_MASK,
    ): SceneFunc3dStage.FUSED_MASK_CREATED,
    (
        SceneFunc3dStage.OPTIONAL_MULTIVIEW_EXPANSION,
        ApprovalAction.CREATE_FUSED_MASK,
    ): SceneFunc3dStage.FUSED_MASK_CREATED,
    (SceneFunc3dStage.FUSED_MASK_CREATED, ApprovalAction.EMIT_FINAL_ANSWER): (
        SceneFunc3dStage.FINAL_ANSWER
    ),
}


@dataclass(frozen=True)
class SceneFunc3dState:
    """Current state for one SceneFunc3D sample run."""

    sample_id: str
    stage: SceneFunc3dStage

    @classmethod
    def initial(cls, sample_id: str) -> SceneFunc3dState:
        """Return the initial state for one sample run."""
        return cls(sample_id=sample_id, stage=SceneFunc3dStage.INITIAL)

    def transition(self, action: ApprovalAction) -> SceneFunc3dState:
        """Return the next state for a valid approval action."""
        next_stage = _TRANSITIONS.get((self.stage, action))
        if next_stage is None:
            raise SceneFunc3dDataError(
                f"invalid transition from {self.stage.value} via {action.value}"
            )
        return SceneFunc3dState(sample_id=self.sample_id, stage=next_stage)
