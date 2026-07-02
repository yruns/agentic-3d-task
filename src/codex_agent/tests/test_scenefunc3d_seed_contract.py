"""Tests for the SceneFun3D SeedConfirmation agent contract (2D seed)."""

from __future__ import annotations

import json

import pytest

from codex_agent.errors import CodexResponseError
from codex_agent.scenefunc3d.seed_contract import (
    SeedConfirmation,
    parse_seed_confirmation,
)
from codex_agent.scenefunc3d.task import SEED_APPROVAL_ACTIONS

_APPROVAL_VALUES = [action.value for action in SEED_APPROVAL_ACTIONS]


def _valid_payload() -> dict[str, object]:
    return {
        "seed_frame_id": "000010",
        "seed_candidate_id": "cand-a",
        "affordance_concept": "the drawer handle",
        "approval_actions": list(_APPROVAL_VALUES),
        "uncertainties": ["partial occlusion"],
    }


def test_parse_seed_confirmation_accepts_valid_payload() -> None:
    confirmation = parse_seed_confirmation(json.dumps(_valid_payload()))
    assert isinstance(confirmation, SeedConfirmation)
    assert confirmation.seed_frame_id == "000010"
    assert confirmation.seed_candidate_id == "cand-a"
    assert confirmation.seed_fragment_id == "000010_cand-a"
    assert confirmation.affordance_concept == "the drawer handle"
    assert confirmation.approval_actions == SEED_APPROVAL_ACTIONS
    assert confirmation.uncertainties == ("partial occlusion",)


def test_parse_seed_confirmation_defaults_uncertainties_to_empty() -> None:
    payload = _valid_payload()
    payload.pop("uncertainties")
    assert parse_seed_confirmation(json.dumps(payload)).uncertainties == ()


def test_parse_seed_confirmation_rejects_first_lift_gate_sequence() -> None:
    from codex_agent.scenefunc3d.task import FRAGMENT_APPROVAL_ACTIONS

    payload = _valid_payload()
    payload["approval_actions"] = [a.value for a in FRAGMENT_APPROVAL_ACTIONS]
    with pytest.raises(CodexResponseError, match="approval_actions"):
        parse_seed_confirmation(json.dumps(payload))


def test_parse_seed_confirmation_rejects_out_of_order_approvals() -> None:
    payload = _valid_payload()
    payload["approval_actions"] = list(reversed(_APPROVAL_VALUES))
    with pytest.raises(CodexResponseError, match="approval_actions"):
        parse_seed_confirmation(json.dumps(payload))


def test_parse_seed_confirmation_rejects_extra_fields() -> None:
    payload = _valid_payload()
    payload["mask_npz_path"] = "/tmp/x.npz"
    with pytest.raises(CodexResponseError):
        parse_seed_confirmation(json.dumps(payload))


def test_parse_seed_confirmation_rejects_non_json() -> None:
    with pytest.raises(CodexResponseError):
        parse_seed_confirmation("not json at all")
