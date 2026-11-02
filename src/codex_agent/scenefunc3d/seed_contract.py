"""The SceneFun3D Phase-2c agent output contract: a single 2D seed confirmation.

The LLM turn's only job is to establish and confirm one high-quality seed at the
2D level: an evidence frame, an approved SAM candidate (verified on the original
RGB overlay), and a short affordance concept. The agent does no 3D lifting; the
deterministic anchor multi-view pipeline (Phase 2b) turns that seed into the
final fused mask, so the agent no longer emits the full final-artifact JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic.types import StringConstraints

from ..errors import CodexResponseError, SceneFunc3dDataError
from ..json_extraction import extract_json_object
from .task import ApprovalAction, validate_seed_approval_actions

NonEmptyString: TypeAlias = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class SeedConfirmation(BaseModel):
    """Strict final JSON contract for one SceneFun3D 2D seed-confirmation turn."""

    model_config = ConfigDict(extra="forbid")

    seed_frame_id: NonEmptyString = Field(
        description="Frame id of the confirmed seed evidence frame."
    )
    seed_candidate_id: NonEmptyString = Field(
        description="Approved SAM candidate id from sam_mask on the seed frame."
    )
    affordance_concept: NonEmptyString = Field(
        description=(
            "Short affordance noun phrase the seed targets (e.g. 'the drawer "
            "handle'); used verbatim as the deterministic Molmo prompt subject."
        )
    )
    approval_actions: tuple[ApprovalAction, ...] = Field(
        min_length=1,
        description=(
            "The seed's 2D approval gate replay, in this exact order: "
            "select_evidence, propose_molmo_point, approve_molmo_point, "
            "propose_sam_candidates, approve_sam_candidate."
        ),
    )
    uncertainties: tuple[str, ...] = Field(
        description=(
            "Short notes on anything ambiguous about the seed. Emit an empty "
            "list when there is nothing to flag; this field is required so the "
            "strict Codex structured-output schema stays valid (every property "
            "must appear in 'required')."
        )
    )

    @property
    def seed_fragment_id(self) -> str:
        """Return the canonical ``<frame>_<candidate>`` seed fragment id."""
        return f"{self.seed_frame_id}_{self.seed_candidate_id}"

    @model_validator(mode="after")
    def _validate_seed_approval_actions(self) -> SeedConfirmation:
        try:
            validate_seed_approval_actions(self.seed_fragment_id, self.approval_actions)
        except SceneFunc3dDataError as exc:
            raise ValueError(str(exc)) from exc
        return self


@dataclass(frozen=True)
class SceneFunc3dSeedOutcome:
    """Parsed 2D seed confirmation joined with the recovered SAM candidate mask."""

    seed_frame_id: str
    seed_candidate_id: str
    seed_fragment_id: str
    affordance_concept: str
    approval_actions: tuple[ApprovalAction, ...]
    uncertainties: tuple[str, ...]
    seed_sam_mask_npz_path: Path


def parse_seed_confirmation(response_text: str) -> SeedConfirmation:
    """Parse and validate one seed-confirmation response.

    Raises:
        CodexResponseError: If the text is not JSON, has extra/missing fields,
            or the approval actions do not replay the 2D seed gates.
    """
    payload = extract_json_object(response_text)
    try:
        return SeedConfirmation.model_validate(payload)
    except ValidationError as exc:
        raise CodexResponseError(
            f"Codex response does not match the SceneFun3D SeedConfirmation "
            f"schema: {exc}"
        ) from exc


__all__ = [
    "SceneFunc3dSeedOutcome",
    "SeedConfirmation",
    "parse_seed_confirmation",
]
