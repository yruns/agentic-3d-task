# SceneFun3D Seed-Confirmation Contract + Pipeline Wiring (Phase 2c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the SceneFun3D agent contract so the single LLM turn only establishes and confirms **one high-quality seed fragment at the 2D level** (a small `SeedConfirmation` JSON: frame + approved SAM candidate + affordance concept), then have the deterministic Phase-2b `run_anchor_multiview_pipeline` produce the final fused mask — replacing the fragile free-form "agent fuses everything and self-reports a big final JSON" loop (P0-B) and the geometric over-selection it caused (P1).

**Architecture:** The agent turn's output schema shrinks from `SceneFunc3dMaskDecision` (7-field final-artifact JSON validated by the ~1000-line `final_mask_artifacts` + fuse-provenance machinery) to `SeedConfirmation` (`seed_frame_id`, `seed_candidate_id`, `affordance_concept`, `approval_actions`, `uncertainties`). **The agent does no 3D work**: it verifies the SAM candidate mask drawn on the *original RGB* and stops at "approve SAM candidate" (approval gate ends at `SAM_CANDIDATE_AGENT_APPROVED`, not `FIRST_LIFT_AGENT_APPROVED`). `SceneFunc3dMaskTask.parse_response` parses that schema, replays the SAM-candidate approval gates (new `task.validate_seed_approval_actions`), and recovers the confirmed seed's **2D SAM candidate mask NPZ** from `events.jsonl`. `run_single_sample` then **deterministically lifts that SAM mask to 3D** (moving `lift_mask_to_3d` out of the agent loop and into the runner), freezes a `TargetAnchor` from the lifted points, and runs the deterministic anchor multi-view pipeline through the real Molmo/SAM sidecars — writing a **new, pipeline-shaped `result.json`** (`result_kind="anchor_multiview_pipeline"`) plus the fused `mask_data.npz`/`lifted_points.ply`. A new `score_pipeline_result_file` scores the fused NPZ via the unchanged `score_mask_npz`/`metrics.py`. The old free-form contract, fuse-tool self-report validators, and their tests are removed once the new path is green.

**Tech Stack:** Python 3.11+, Pydantic v2 (agent-output + event boundaries only), numpy, pytest. Reuses `pipeline.run_anchor_multiview_pipeline`, `pipeline_backends.SidecarFrameProposer`, `tools/mask_lifting.{LiftMaskArgs,lift_mask_to_3d}`, `backends/{anchor,fusion,lift_3d,frame_assets}.py`, `task.{ApprovalAction,SceneFunc3dStage,SceneFunc3dState}`, `tools/mask_artifacts.artifact_paths_for`, `evaluation/scorer.score_mask_npz`. Must satisfy `docs/python_code_agent_quality_guide.md` (§5 typing, §6 boundary validation, §7 exceptions, §13 ban list), `AGENTS.md` §1, and the repo **no-fallback** rule (empty fused mask or unliftable seed → fail-closed, never emit a low-quality mask).

Reference spec: `docs/superpowers/specs/2026-07-02-scenefunc3d-anchor-multiview-fusion-design.md` (Stage A agent seed; boundary "small seed-confirmation JSON"; §"模块/文件划分" runner/playbook/task edits). Phase 2b engine is committed on `feat/scenefunc3d-agent-tools` (`83ff2cd`…`5ae92ef`).

Environment (macOS): `source .venv/bin/activate`. Tests: `PYTHONPATH=src pytest <path> -v`.

---

## Decisions (made — flag before execution if you disagree)

1. **Pure 2D seed confirmation.** The agent verifies only the SAM candidate mask overlaid on the *original RGB* (`sam_mask` already emits `{candidate_id}_overlay.jpg`). It does **not** call `lift_mask_to_3d` or `inspect_mask_artifact`. The seed approval sequence ends at `approve_sam_candidate`. The **3D lift moves into the runner** (deterministic) — the anchor is still built from lifted points, exactly as before; only *who reviews it* changes. Trade-off: the agent-side "3D bbox too big" reject signal is gone, but it is covered downstream by the `motion_type` radius cap + Stage C anchor gate + Stage D DBSCAN/size cap; and a 2D mask that visibly covers a whole panel is rejectable from the original-image overlay. Residual risk: a seed whose masked region has bad depth now fails **closed** in the runner (`failure.json`) rather than being agent-corrected — acceptable under the no-fallback rule.
2. **Scoring shape.** The deterministic fused mask is scored via a **new `result_kind="anchor_multiview_pipeline"` result.json** whose fused `mask_data.npz` is scored by the existing `score_mask_npz` (needs only `point_indices` + raw-mesh range). We do **not** synthesize a `FinalMaskArtifactDocument`: that schema requires per-fragment review artifacts on disk **and** `sum(fragment.point_count) == fused_point_count`, structurally impossible for a multi-view *fused* mask. `metrics.py` math is unchanged.
3. **`score_result_file` stays a single entry point** but dispatches on `result_kind` (absent/`"agent_freeform"` → legacy validation; `"anchor_multiview_pipeline"` → fused-NPZ scoring), so `_run_batch_sample` and `evaluation/__main__.py` also score old result files.
4. **Remove the dead free-form/fuse machinery in this plan (Task 7)**, gated on grep-for-references + green quality gate.
5. **The seed frame is re-processed by the pipeline** (Stage C runs Molmo→SAM→lift on every selected frame, including the seed frame). Accepted for Phase 2c; reusing the confirmed seed lift for its own frame is a follow-up.
6. **The "≤3 seed correction" bound** lives in the trimmed playbook (agent-internal retries) plus the runtime's existing single finalization re-ask (`runtime.CodexAgentRuntime`, `max_finalization_retries=1`). No new runner retry loop.

---

## Real contract state (verified 2026-07-02 on `feat/scenefunc3d-agent-tools`)

- **Runtime executor is generic** (`runtime.py:195` `execute`): `request = task.build_turn_request()` → `run_turn(request, response_validator=task.is_valid_response)` → `outcome = task.parse_response(turn.final_response)`. **No `runtime.py` change is needed.**
- **Task class** `SceneFunc3dMaskTask` (`runner.py:691`): `__init__(*, sample, scene_root, output_dir, backend_config_path, tool_context_path, tool_cli_module=DEFAULT_TOOL_CLI_MODULE)`; `build_turn_request` sets `output_schema=SceneFunc3dMaskDecision.model_json_schema()`; `parse_response` → `SceneFunc3dMaskOutcome`.
- **Approval FSM** (`task.py`): stages `INITIAL → EVIDENCE_SELECTED → MOLMO_POINT_PROPOSED → MOLMO_POINT_AGENT_APPROVED → SAM_CANDIDATES_PROPOSED → SAM_CANDIDATE_AGENT_APPROVED → FIRST_LIFT_CREATED → FIRST_LIFT_AGENT_APPROVED → …`. Actions to reach `SAM_CANDIDATE_AGENT_APPROVED` are `FRAGMENT_APPROVAL_ACTIONS[:5]` (`SELECT_EVIDENCE, PROPOSE_MOLMO_POINT, APPROVE_MOLMO_POINT, PROPOSE_SAM_CANDIDATES, APPROVE_SAM_CANDIDATE`). `SceneFunc3dState.initial(id).transition(action)` + `_approval_action_values` are reused. `task.py` has no `__all__`.
- **SAM event shape** (`tools/mask_artifacts.SceneFunc3dToolInvocationEventPayload` + `runner._SamMaskToolResult`/`_SamCandidateToolResult`): a successful `sam_mask` is one `events.jsonl` line `{"event_type":"tool_completed","tool_name":"sam_mask","status":"success","args":{"frame_id":..,"image_path":..,"points":[...]},"result":{"frame_id":..,"candidates":[{"candidate_id":..,"mask_npz_path":..,"overlay_path":..}, ...],"contact_sheet_path":..},"error":""}`. The candidate `mask_npz_path` is the 2D SAM mask the lift tool consumes.
- **Lift tool** (`tools/mask_lifting.py`): `LiftMaskArgs(frame_id, candidate_id, mask_path: FilePath (alias mask_npz_path), depth_path: FilePath, intrinsics_path: FilePath, pose_path: FilePath)`; `lift_mask_to_3d(args, *, out_dir, raw_mesh_path) -> LiftMaskResult` writes `<out_dir>/fragments/<frame>_<candidate>/{mask_data.npz,lifted_points.ply}`; `LiftMaskResult` = `frame_id, candidate_id, lifted_point_count, mask_npz_path, mask_ply_path, overlay_path`. The fragment NPZ holds `points_world` + `point_indices`. `backends/frame_assets.resolve_frame_geometry_assets(scene, frame_id)` returns `.depth_path/.intrinsics_path/.pose_path`.
- **Artifact layout** (`tools/mask_artifacts.artifact_paths_for(output_dir, visit_id, desc_id)`): `root=<output_dir>/<visit_id>/<desc_id>`, `root/events.jsonl`, `root/result.json`, `root/failure.json`, `root/fragments/`, `root/fused/`. `SceneFunc3dRunCompletionEvent(sample_id, result_path, summary_path, mask_artifact_path)`.
- **Pipeline entry** (`pipeline.py:203`): `run_anchor_multiview_pipeline(*, anchor, scene, scene_vertices, proposer, params, frame_cap=30, depth_tolerance=0.25, gate_radius_scale=1.0) -> AnchorMultiViewResult` where `AnchorMultiViewResult` = `fused: FusedMask`, `selected_frames`, `per_frame: tuple[PerFrameOutcome,...]`; `FusedMask` = `point_indices, confidence, instances, params`; `PerFrameOutcome` = `frame_id, point_indices, centroid, accepted, reject_reason, molmo_fallback_used`.
- **Proposer** (`pipeline_backends.SidecarFrameProposer`, frozen dataclass): `SidecarFrameProposer(scene, out_dir, backend_config_path, affordance_concept, task_description)`; sidecars untouched until `.propose()`.
- **Anchor / writers** (`backends`): `build_anchor(points_world, *, motion_type, seed_frame_id, radius_percentile=95.0)`; `motion_priors.motion_size_prior` has a `_DEFAULT_PRIOR` for unknown/empty motion; `lift_3d.{write_lift_npz(path, points_world, *, point_indices=None), write_lift_ply(path, points_world), load_scene_mesh_vertices(path)}`; `final_mask_artifacts.load_points_world_npz(path) -> FloatArray`.
- **Scorer** (`evaluation/scorer.py`): `score_mask_npz(*, data_root, sample_id, mask_npz_path, failure_type="")`; `score_result_file(*, data_root, result_path, failure_type="")` currently forces `validate_final_mask_artifact`. `SceneFunc3dScore` = `sample_id, metrics: MaskMetrics, failure_type`.
- **Exceptions** (`errors.py`): `CodexResponseError`, `CodexTurnError`, `SceneFunc3dDataError` all subclass `CodexAgentError`. `run_single_sample` handlers are ordered `CodexTurnError` / `CodexResponseError` / `CodexAgentError`, so a recovery `CodexResponseError` → `failure.json` (stage `response_validation`) and an empty-fused `SceneFunc3dDataError` → `failure.json` (stage `run`).
- **Test scaffolding** (`tests/test_scenefunc3d_runner.py`): `_sample()` (visit `421254`, desc `desc-a`, motion `trans`), `_mask_task(...)`, `_write_scene(data_root)` → `_write_scene_root` writes a **binary 100-vertex** `raw/mesh.ply` (`_write_raw_mesh`, line 4821) + `{frame}-rgb.png` for `000010`/`000020` (no depth/intrinsic/pose). `_FakeExecutor(outcome, on_execute=...)` builds the turn, runs `on_execute`, returns a `CodexTaskResult`. The importable `tests/scenefunc3d_synthetic_scene.py` writes a full binary-PLY scene.

---

## File Structure (Phase 2c)

- Modify `src/codex_agent/scenefunc3d/task.py` — add `SEED_APPROVAL_ACTIONS` + `validate_seed_approval_actions` (ends at `SAM_CANDIDATE_AGENT_APPROVED`).
- Create `src/codex_agent/scenefunc3d/seed_contract.py` — `SeedConfirmation` (Pydantic strict), `SceneFunc3dSeedOutcome` (frozen dataclass, carries the 2D SAM mask path), `parse_seed_confirmation`.
- Create `src/codex_agent/scenefunc3d/seed_recovery.py` — `SeedSamCandidate` + `recover_seed_sam_candidate(events_path, *, seed_frame_id, seed_candidate_id)` (standalone `events.jsonl` reader for the confirmed 2D SAM candidate; no `runner` import → no cycle).
- Create `src/codex_agent/scenefunc3d/pipeline_result.py` — pipeline result `TypedDict`s, `SceneFunc3dPipelineOutcome` (carries `seed_point_count` from the runner lift), `write_pipeline_fused_mask`, `write_pipeline_result`.
- Modify `src/codex_agent/scenefunc3d/evaluation/scorer.py` — add `score_pipeline_result_file`; dispatch `score_result_file` on `result_kind`.
- Modify `src/codex_agent/scenefunc3d/runner.py` — repoint `SceneFunc3dMaskTask` to `SeedConfirmation`; add `_lift_seed_fragment` + `_run_pipeline_for_seed`; rewrite `run_single_sample`; remove dead free-form/fuse machinery (Task 7); update `__all__`.
- Modify `src/codex_agent/scenefunc3d/playbook.py` — trim `SCENEFUNC3D_TOOL_NAMES` to the 2D seed tools (drop `lift_mask_to_3d`, `inspect_mask_artifact`, `suggest_additional_views`, `fuse_accepted_masks`); rewrite the final-JSON instructions for `SeedConfirmation`.
- Tests: create `tests/test_scenefunc3d_seed_contract.py`, `tests/test_scenefunc3d_seed_recovery.py`, `tests/test_scenefunc3d_pipeline_result.py`, `tests/test_scenefunc3d_scorer_pipeline.py`, `src/codex_agent/scenefunc3d/tests_support.py`, `tests/scenefunc3d_scene_fixtures.py`; rework `tests/test_scenefunc3d_runner.py`; update `tests/test_scenefunc3d_playbook.py` and `tests/test_scenefunc3d_state.py`.
- Docs: `docs/benchmark/scenefunc_molmo_sam3d/v8_seed_confirmation_online_<YYYYMMDD>.md` + README; Phase-2c pointer in the Phase-2b plan.

---

## Task 1: Seed approval gate + confirmation contract

**Files:**
- Modify: `src/codex_agent/scenefunc3d/task.py`
- Create: `src/codex_agent/scenefunc3d/seed_contract.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_seed_contract.py`, extend `src/codex_agent/tests/test_scenefunc3d_state.py`

- [ ] **Step 1: Add `SEED_APPROVAL_ACTIONS` + validator to `task.py`**

Insert after `validate_fragment_approval_actions` (line 138), before `_approval_action_values`:

```python
SEED_APPROVAL_ACTIONS: tuple[ApprovalAction, ...] = (
    ApprovalAction.SELECT_EVIDENCE,
    ApprovalAction.PROPOSE_MOLMO_POINT,
    ApprovalAction.APPROVE_MOLMO_POINT,
    ApprovalAction.PROPOSE_SAM_CANDIDATES,
    ApprovalAction.APPROVE_SAM_CANDIDATE,
)


def validate_seed_approval_actions(
    seed_fragment_id: str, approval_actions: tuple[ApprovalAction, ...]
) -> None:
    """Validate one confirmed 2D seed's approval action sequence."""
    if approval_actions != SEED_APPROVAL_ACTIONS:
        raise SceneFunc3dDataError(
            "seed approval_actions must exactly replay the SceneFun3D point/SAM "
            "approval gates ending at approve_sam_candidate: "
            f"seed_fragment_id={seed_fragment_id}; "
            f"expected={_approval_action_values(SEED_APPROVAL_ACTIONS)}; "
            f"actual={_approval_action_values(approval_actions)}"
        )
    state = SceneFunc3dState.initial(seed_fragment_id)
    for action in approval_actions:
        state = state.transition(action)
    if state.stage is not SceneFunc3dStage.SAM_CANDIDATE_AGENT_APPROVED:
        raise SceneFunc3dDataError(
            "seed approval_actions must end at "
            f"{SceneFunc3dStage.SAM_CANDIDATE_AGENT_APPROVED.value}: "
            f"seed_fragment_id={seed_fragment_id}; stage={state.stage.value}"
        )
```

Add a regression test to `test_scenefunc3d_state.py`:

```python
def test_validate_seed_approval_actions_accepts_five_gate_sequence() -> None:
    from codex_agent.scenefunc3d.task import (
        SEED_APPROVAL_ACTIONS,
        validate_seed_approval_actions,
    )

    validate_seed_approval_actions("000010_cand-a", SEED_APPROVAL_ACTIONS)


def test_validate_seed_approval_actions_rejects_first_lift_sequence() -> None:
    import pytest

    from codex_agent.errors import SceneFunc3dDataError
    from codex_agent.scenefunc3d.task import (
        FRAGMENT_APPROVAL_ACTIONS,
        validate_seed_approval_actions,
    )

    with pytest.raises(SceneFunc3dDataError, match="approve_sam_candidate"):
        validate_seed_approval_actions("000010_cand-a", FRAGMENT_APPROVAL_ACTIONS)
```

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_state.py -v` (Expected: PASS).

- [ ] **Step 2: Write the failing seed-contract test**

```python
# src/codex_agent/tests/test_scenefunc3d_seed_contract.py
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_seed_contract.py -v`
Expected: FAIL with `ModuleNotFoundError: codex_agent.scenefunc3d.seed_contract`.

- [ ] **Step 4: Write minimal implementation**

```python
# src/codex_agent/scenefunc3d/seed_contract.py
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
    uncertainties: tuple[str, ...] = ()

    @property
    def seed_fragment_id(self) -> str:
        """Return the canonical ``<frame>_<candidate>`` seed fragment id."""
        return f"{self.seed_frame_id}_{self.seed_candidate_id}"

    @model_validator(mode="after")
    def _validate_seed_approval_actions(self) -> SeedConfirmation:
        try:
            validate_seed_approval_actions(
                self.seed_fragment_id, self.approval_actions
            )
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_seed_contract.py src/codex_agent/tests/test_scenefunc3d_state.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/codex_agent/scenefunc3d/task.py \
        src/codex_agent/scenefunc3d/seed_contract.py \
        src/codex_agent/tests/test_scenefunc3d_seed_contract.py \
        src/codex_agent/tests/test_scenefunc3d_state.py
git commit -m "feat(scenefunc3d): add 2D SeedConfirmation contract + seed approval gate"
```

---

## Task 2: Recover the confirmed 2D SAM candidate from `events.jsonl` (`seed_recovery.py`)

Reads the run's `events.jsonl`, finds the **last successful** `sam_mask` event on the confirmed `seed_frame_id` that contains the confirmed `seed_candidate_id`, and returns that candidate's 2D mask NPZ path. Fails closed (`CodexResponseError`) when missing, so `is_valid_response` triggers the runtime re-ask. It does **not** stat/load the NPZ (that is the runner's deterministic lift's concern).

**Files:**
- Create: `src/codex_agent/scenefunc3d/seed_recovery.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_seed_recovery.py`

- [ ] **Step 1: Write the failing test**

```python
# src/codex_agent/tests/test_scenefunc3d_seed_recovery.py
"""Tests for recovering a confirmed 2D SAM candidate from events.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_agent.errors import CodexResponseError
from codex_agent.scenefunc3d.seed_recovery import (
    SeedSamCandidate,
    recover_seed_sam_candidate,
)


def _sam_event(*, frame_id: str, candidates: list[tuple[str, str]]) -> str:
    return json.dumps(
        {
            "event_type": "tool_completed",
            "tool_name": "sam_mask",
            "status": "success",
            "args": {"frame_id": frame_id, "image_path": f"{frame_id}-rgb.jpg"},
            "result": {
                "frame_id": frame_id,
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "mask_npz_path": mask_npz_path,
                        "overlay_path": f"{candidate_id}_overlay.jpg",
                    }
                    for candidate_id, mask_npz_path in candidates
                ],
                "contact_sheet_path": "contact_sheet.jpg",
            },
            "error": "",
        }
    )


def test_recover_seed_sam_candidate_returns_matching_candidate(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        _sam_event(
            frame_id="000010",
            candidates=[("cand-a", "/run/sam/000010/cand-a.npz"), ("cand-b", "b.npz")],
        )
        + "\n",
        encoding="utf-8",
    )
    seed = recover_seed_sam_candidate(
        events, seed_frame_id="000010", seed_candidate_id="cand-a"
    )
    assert isinstance(seed, SeedSamCandidate)
    assert seed.mask_npz_path == Path("/run/sam/000010/cand-a.npz")


def test_recover_seed_sam_candidate_uses_last_matching_event(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        _sam_event(frame_id="000010", candidates=[("cand-a", "/run/stale.npz")])
        + "\n"
        + _sam_event(frame_id="000010", candidates=[("cand-a", "/run/fresh.npz")])
        + "\n",
        encoding="utf-8",
    )
    seed = recover_seed_sam_candidate(
        events, seed_frame_id="000010", seed_candidate_id="cand-a"
    )
    assert seed.mask_npz_path == Path("/run/fresh.npz")


def test_recover_seed_sam_candidate_rejects_missing_candidate(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        _sam_event(frame_id="000010", candidates=[("cand-b", "b.npz")]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CodexResponseError, match="no successful sam_mask"):
        recover_seed_sam_candidate(
            events, seed_frame_id="000010", seed_candidate_id="cand-a"
        )


def test_recover_seed_sam_candidate_rejects_missing_events_file(tmp_path: Path) -> None:
    with pytest.raises(CodexResponseError, match="event log is missing"):
        recover_seed_sam_candidate(
            tmp_path / "absent.jsonl",
            seed_frame_id="000010",
            seed_candidate_id="cand-a",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_seed_recovery.py -v`
Expected: FAIL with `ModuleNotFoundError: codex_agent.scenefunc3d.seed_recovery`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/codex_agent/scenefunc3d/seed_recovery.py
"""Recover a confirmed 2D SAM candidate mask from a run's ``events.jsonl``.

The Phase-2c agent turn confirms a seed ``(frame_id, candidate_id)`` after
reviewing the SAM candidate mask on the original RGB. The run's tool event log
already recorded the successful ``sam_mask`` that produced that candidate. This
module resolves the candidate's 2D mask NPZ so ``run_single_sample`` can
deterministically lift it and freeze a ``TargetAnchor``. It reads the log
standalone (no ``runner`` imports) to avoid an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from json import JSONDecodeError, loads
from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.types import StringConstraints

from ..errors import CodexResponseError

NonEmptyString: TypeAlias = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]

_SAM_TOOL_NAME = "sam_mask"


class _SamCandidateEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: NonEmptyString
    mask_npz_path: NonEmptyString


class _SamEventResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    frame_id: NonEmptyString
    candidates: tuple[_SamCandidateEvent, ...] = Field(min_length=1)


@dataclass(frozen=True)
class SeedSamCandidate:
    """The confirmed seed's 2D SAM candidate mask NPZ."""

    frame_id: str
    candidate_id: str
    mask_npz_path: Path


def recover_seed_sam_candidate(
    events_path: Path, *, seed_frame_id: str, seed_candidate_id: str
) -> SeedSamCandidate:
    """Return the confirmed seed's last successful SAM candidate mask.

    Raises:
        CodexResponseError: If the event log is missing/unreadable, or no
            successful ``sam_mask`` event on ``seed_frame_id`` contains
            ``seed_candidate_id``.
    """
    if not events_path.is_file():
        raise CodexResponseError(
            "SceneFun3D tool event log is missing while recovering the confirmed "
            f"seed SAM candidate: events_path={events_path}; "
            f"seed_frame_id={seed_frame_id!r}; seed_candidate_id={seed_candidate_id!r}"
        )
    try:
        event_lines = events_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise CodexResponseError(
            "could not read SceneFun3D tool event log while recovering the seed "
            f"SAM candidate: events_path={events_path}; "
            f"error_type={exc.__class__.__name__}"
        ) from exc

    matched_npz_path: Path | None = None
    for line_number, raw_line in enumerate(event_lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        candidate_path = _match_seed_sam_line(
            line,
            events_path=events_path,
            line_number=line_number,
            seed_frame_id=seed_frame_id,
            seed_candidate_id=seed_candidate_id,
        )
        if candidate_path is not None:
            matched_npz_path = candidate_path

    if matched_npz_path is None:
        raise CodexResponseError(
            "no successful sam_mask event on the confirmed seed frame contains the "
            f"confirmed candidate: events_path={events_path}; "
            f"seed_frame_id={seed_frame_id!r}; seed_candidate_id={seed_candidate_id!r}"
        )
    return SeedSamCandidate(
        frame_id=seed_frame_id,
        candidate_id=seed_candidate_id,
        mask_npz_path=matched_npz_path,
    )


def _match_seed_sam_line(
    line: str,
    *,
    events_path: Path,
    line_number: int,
    seed_frame_id: str,
    seed_candidate_id: str,
) -> Path | None:
    try:
        event: object = loads(line)
    except JSONDecodeError as exc:
        raise CodexResponseError(
            "SceneFun3D tool event line is not valid JSON while recovering the "
            f"seed SAM candidate: events_path={events_path}; line={line_number}; "
            f"error={exc}"
        ) from exc
    if not isinstance(event, dict):
        return None
    if event.get("tool_name") != _SAM_TOOL_NAME or event.get("status") != "success":
        return None
    try:
        result = _SamEventResult.model_validate(event.get("result"))
    except ValidationError as exc:
        raise CodexResponseError(
            "successful sam_mask event does not match the expected result schema: "
            f"events_path={events_path}; line={line_number}; error={exc}"
        ) from exc
    if result.frame_id != seed_frame_id:
        return None
    for candidate in result.candidates:
        if candidate.candidate_id == seed_candidate_id:
            return Path(candidate.mask_npz_path).expanduser().resolve()
    return None


__all__ = ["SeedSamCandidate", "recover_seed_sam_candidate"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_seed_recovery.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/seed_recovery.py \
        src/codex_agent/tests/test_scenefunc3d_seed_recovery.py
git commit -m "feat(scenefunc3d): recover confirmed 2D SAM candidate from events log"
```

---

## Task 3: Pipeline result payload + writers (`pipeline_result.py`)

The durable Phase-2c `result.json` shape + fused NPZ/PLY writer. `seed_point_count` comes from the runner's deterministic seed lift (Task 5), not the parse. Fails closed on an empty fused mask.

**Files:**
- Create: `src/codex_agent/scenefunc3d/pipeline_result.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_pipeline_result.py`

- [ ] **Step 1: Write the failing test**

```python
# src/codex_agent/tests/test_scenefunc3d_pipeline_result.py
"""Tests for the Phase-2c pipeline result payload and writers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.models import CodexTurnMetadata
from codex_agent.scenefunc3d.backends.fusion import FusedMask, FusionParams
from codex_agent.scenefunc3d.pipeline import AnchorMultiViewResult, PerFrameOutcome
from codex_agent.scenefunc3d.pipeline_result import (
    SceneFunc3dPipelineOutcome,
    write_pipeline_fused_mask,
    write_pipeline_result,
)
from codex_agent.scenefunc3d.seed_contract import SceneFunc3dSeedOutcome
from codex_agent.scenefunc3d.task import SEED_APPROVAL_ACTIONS

_SCENE_VERTICES = np.array(
    [[0.0, 0.0, 2.0], [0.01, 0.0, 2.0], [0.0, 0.01, 2.0], [9.0, 9.0, 9.0]],
    dtype=np.float64,
)


def _seed_outcome() -> SceneFunc3dSeedOutcome:
    return SceneFunc3dSeedOutcome(
        seed_frame_id="000010",
        seed_candidate_id="cand-a",
        seed_fragment_id="000010_cand-a",
        affordance_concept="the drawer handle",
        approval_actions=SEED_APPROVAL_ACTIONS,
        uncertainties=("partial occlusion",),
        seed_sam_mask_npz_path=Path("/run/sam/000010/cand-a.npz"),
    )


def _pipeline_result(point_indices: tuple[int, ...]) -> AnchorMultiViewResult:
    return AnchorMultiViewResult(
        fused=FusedMask(
            point_indices=point_indices,
            confidence=0.75,
            instances=(),
            params=FusionParams(),
        ),
        selected_frames=(),
        per_frame=(
            PerFrameOutcome(
                frame_id="000010",
                point_indices=point_indices,
                centroid=(0.0, 0.0, 2.0),
                accepted=True,
                reject_reason="",
                molmo_fallback_used=False,
            ),
        ),
    )


def test_write_pipeline_fused_mask_writes_npz_and_ply(tmp_path: Path) -> None:
    npz_path, ply_path = write_pipeline_fused_mask(
        tmp_path / "fused", point_indices=(0, 1, 2), scene_vertices=_SCENE_VERTICES
    )
    assert npz_path.is_file() and ply_path.is_file()
    with np.load(npz_path) as archive:
        assert sorted(int(v) for v in archive["point_indices"]) == [0, 1, 2]
        assert archive["points_world"].shape == (3, 3)


def test_write_pipeline_fused_mask_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(SceneFunc3dDataError, match="empty fused mask"):
        write_pipeline_fused_mask(
            tmp_path / "fused", point_indices=(), scene_vertices=_SCENE_VERTICES
        )


def test_write_pipeline_result_writes_scorable_payload(tmp_path: Path) -> None:
    npz_path, ply_path = write_pipeline_fused_mask(
        tmp_path / "fused", point_indices=(0, 1, 2), scene_vertices=_SCENE_VERTICES
    )
    outcome = SceneFunc3dPipelineOutcome(
        seed=_seed_outcome(),
        seed_point_count=2,
        mask_npz_path=npz_path,
        mask_ply_path=ply_path,
        result=_pipeline_result((0, 1, 2)),
    )
    result_path = write_pipeline_result(
        tmp_path / "result.json",
        outcome,
        sample_id="421254::desc-a",
        turn_metadata=CodexTurnMetadata(turn_id="fake-turn", status="completed"),
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["task_name"] == "scenefunc3d_mask_generation"
    assert payload["result_kind"] == "anchor_multiview_pipeline"
    assert payload["outcome"]["mask_npz_path"] == str(npz_path)
    assert payload["outcome"]["confidence"] == pytest.approx(0.75)
    assert payload["outcome"]["selected_frame_ids"] == ["000010"]
    assert payload["outcome"]["seed"]["seed_fragment_id"] == "000010_cand-a"
    assert payload["outcome"]["seed"]["seed_point_count"] == 2
    assert payload["outcome"]["per_frame"][0]["frame_id"] == "000010"
    assert payload["turn"]["turn_id"] == "fake-turn"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_pipeline_result.py -v`
Expected: FAIL with `ModuleNotFoundError: codex_agent.scenefunc3d.pipeline_result`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/codex_agent/scenefunc3d/pipeline_result.py
"""Durable Phase-2c pipeline result payload + fused-mask writers.

``run_single_sample`` writes a ``result.json`` tagged
``result_kind="anchor_multiview_pipeline"`` plus the fused ``mask_data.npz`` /
``lifted_points.ply``. The fused NPZ carries ``point_indices`` so the existing
``score_mask_npz`` can score it directly (no ``FinalMaskArtifactDocument``).
``seed_point_count`` is the point count of the runner's deterministic seed lift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np

from ..errors import SceneFunc3dDataError
from .backends.lift_3d import FloatArray, write_lift_npz, write_lift_ply

if TYPE_CHECKING:
    from ..models import CodexTurnMetadata
    from .pipeline import AnchorMultiViewResult
    from .seed_contract import SceneFunc3dSeedOutcome

TASK_NAME = "scenefunc3d_mask_generation"
PIPELINE_RESULT_KIND: Literal["anchor_multiview_pipeline"] = "anchor_multiview_pipeline"


class SceneFunc3dSeedProvenancePayload(TypedDict):
    """JSON-ready confirmed-seed provenance for a pipeline result."""

    seed_frame_id: str
    seed_candidate_id: str
    seed_fragment_id: str
    affordance_concept: str
    seed_point_count: int


class SceneFunc3dFusionParamsPayload(TypedDict):
    """JSON-ready fusion parameters used for one pipeline result."""

    agreement_tau: float
    radius_scale: float
    cluster_link_eps_m: float
    min_cluster_points: int


class SceneFunc3dPerFramePayload(TypedDict):
    """JSON-ready per-frame provenance row for a pipeline result."""

    frame_id: str
    point_count: int
    accepted: bool
    reject_reason: str
    molmo_fallback_used: bool


class SceneFunc3dPipelineOutcomePayload(TypedDict):
    """JSON-ready outcome of one deterministic anchor multi-view run."""

    result_kind: Literal["anchor_multiview_pipeline"]
    mask_npz_path: str
    mask_ply_path: str
    confidence: float
    selected_frame_ids: list[str]
    seed: SceneFunc3dSeedProvenancePayload
    per_frame: list[SceneFunc3dPerFramePayload]
    fusion_params: SceneFunc3dFusionParamsPayload
    uncertainties: list[str]


class SceneFunc3dPipelineTurnPayload(TypedDict):
    """JSON-ready turn metadata for a pipeline result."""

    turn_id: str | None
    status: str | None
    duration_ms: int | None
    reasoning_summary: str | None
    run_home: str | None


class SceneFunc3dPipelineResultPayload(TypedDict):
    """JSON-ready durable ``result.json`` for one Phase-2c pipeline run."""

    task_name: str
    sample_id: str
    result_kind: Literal["anchor_multiview_pipeline"]
    outcome: SceneFunc3dPipelineOutcomePayload
    turn: SceneFunc3dPipelineTurnPayload


@dataclass(frozen=True)
class SceneFunc3dPipelineOutcome:
    """Deterministic pipeline outcome for one sample (durable result source)."""

    seed: SceneFunc3dSeedOutcome
    seed_point_count: int
    mask_npz_path: Path
    mask_ply_path: Path
    result: AnchorMultiViewResult

    def to_payload(self) -> SceneFunc3dPipelineOutcomePayload:
        """Return the JSON-serializable outcome payload."""
        fused = self.result.fused
        return {
            "result_kind": PIPELINE_RESULT_KIND,
            "mask_npz_path": str(self.mask_npz_path),
            "mask_ply_path": str(self.mask_ply_path),
            "confidence": fused.confidence,
            "selected_frame_ids": [
                outcome.frame_id
                for outcome in self.result.per_frame
                if outcome.accepted
            ],
            "seed": {
                "seed_frame_id": self.seed.seed_frame_id,
                "seed_candidate_id": self.seed.seed_candidate_id,
                "seed_fragment_id": self.seed.seed_fragment_id,
                "affordance_concept": self.seed.affordance_concept,
                "seed_point_count": self.seed_point_count,
            },
            "per_frame": [
                {
                    "frame_id": outcome.frame_id,
                    "point_count": len(outcome.point_indices),
                    "accepted": outcome.accepted,
                    "reject_reason": outcome.reject_reason,
                    "molmo_fallback_used": outcome.molmo_fallback_used,
                }
                for outcome in self.result.per_frame
            ],
            "fusion_params": {
                "agreement_tau": fused.params.agreement_tau,
                "radius_scale": fused.params.radius_scale,
                "cluster_link_eps_m": fused.params.cluster_link_eps_m,
                "min_cluster_points": fused.params.min_cluster_points,
            },
            "uncertainties": list(self.seed.uncertainties),
        }


def write_pipeline_fused_mask(
    fused_dir: Path, *, point_indices: tuple[int, ...], scene_vertices: FloatArray
) -> tuple[Path, Path]:
    """Write the fused mask NPZ (points + ids) and ASCII PLY; fail on empty."""
    if not point_indices:
        raise SceneFunc3dDataError(
            "anchor multi-view pipeline produced an empty fused mask; failing "
            f"closed instead of writing a zero-point artifact: fused_dir={fused_dir}"
        )
    index_array = np.asarray(point_indices, dtype=np.int64)
    points_world = np.asarray(scene_vertices, dtype=np.float64)[index_array]
    fused_dir.mkdir(parents=True, exist_ok=True)
    mask_npz_path = write_lift_npz(
        fused_dir / "mask_data.npz", points_world, point_indices=index_array
    )
    mask_ply_path = write_lift_ply(fused_dir / "lifted_points.ply", points_world)
    return mask_npz_path, mask_ply_path


def write_pipeline_result(
    result_path: Path,
    outcome: SceneFunc3dPipelineOutcome,
    *,
    sample_id: str,
    turn_metadata: CodexTurnMetadata,
) -> Path:
    """Write the durable pipeline ``result.json`` and return its path."""
    payload: SceneFunc3dPipelineResultPayload = {
        "task_name": TASK_NAME,
        "sample_id": sample_id,
        "result_kind": PIPELINE_RESULT_KIND,
        "outcome": outcome.to_payload(),
        "turn": {
            "turn_id": turn_metadata.turn_id,
            "status": turn_metadata.status,
            "duration_ms": turn_metadata.duration_ms,
            "reasoning_summary": turn_metadata.reasoning_summary,
            "run_home": turn_metadata.run_home,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result_path


__all__ = [
    "PIPELINE_RESULT_KIND",
    "SceneFunc3dFusionParamsPayload",
    "SceneFunc3dPerFramePayload",
    "SceneFunc3dPipelineOutcome",
    "SceneFunc3dPipelineOutcomePayload",
    "SceneFunc3dPipelineResultPayload",
    "SceneFunc3dPipelineTurnPayload",
    "SceneFunc3dSeedProvenancePayload",
    "write_pipeline_fused_mask",
    "write_pipeline_result",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_pipeline_result.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/codex_agent/scenefunc3d/pipeline_result.py \
        src/codex_agent/tests/test_scenefunc3d_pipeline_result.py
git commit -m "feat(scenefunc3d): add pipeline result payload and fused-mask writer"
```

---

## Task 4: Pipeline-result scorer + `result_kind` dispatch (`scorer.py`)

**Files:**
- Create: `src/codex_agent/scenefunc3d/tests_support.py`, `src/codex_agent/tests/scenefunc3d_scene_fixtures.py`
- Modify: `src/codex_agent/scenefunc3d/evaluation/scorer.py`, `src/codex_agent/tests/scenefunc3d_synthetic_scene.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_scorer_pipeline.py`

- [ ] **Step 1: Extract a shared binary-PLY writer + scoring-scene fixture**

Create the shared mesh writer (reused by the synthetic scene helper + the scoring fixture):

```python
# src/codex_agent/scenefunc3d/tests_support.py
"""Importable helpers for building SceneFun3D test artifacts on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .backends.lift_3d import FloatArray


def write_binary_ply_mesh(path: Path, vertices_world: FloatArray) -> Path:
    """Write ``vertices_world`` as a binary-little-endian XYZ PLY mesh."""
    vertices = np.asarray(vertices_world, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices_world must have shape (N, 3): {vertices.shape}")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes(order="C"))
    return path


__all__ = ["write_binary_ply_mesh"]
```

Update `src/codex_agent/tests/scenefunc3d_synthetic_scene.py` to delete its private `_write_binary_ply` and call `write_binary_ply_mesh` (change `_write_binary_ply(raw_dir / "mesh.ply", vertices_world)` → `write_binary_ply_mesh(raw_dir / "mesh.ply", vertices_world)`; import from `codex_agent.scenefunc3d.tests_support`). Run `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_frame_loader.py -q` (Expected: PASS).

Create the scoring-scene fixture (`score_mask_npz` needs a real prepared scene: `raw/mesh.ply` for the raw-id range check + descriptions/motions/annotations with hidden GT `indices`):

```python
# src/codex_agent/tests/scenefunc3d_scene_fixtures.py
"""Write a minimal scored SceneFun3D dataset root for scorer tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from codex_agent.scenefunc3d.tests_support import write_binary_ply_mesh


def write_scoring_scene(
    data_root: Path,
    *,
    visit_id: str = "421254",
    desc_id: str = "desc-a",
    annot_id: str = "annot-a",
    gt_point_ids: tuple[int, ...] = (0, 1, 2),
    vertex_count: int = 16,
) -> Path:
    """Write descriptions/motions/annotations + a raw binary mesh PLY."""
    scene_dir = data_root / visit_id
    (scene_dir / "raw").mkdir(parents=True, exist_ok=True)
    (scene_dir / f"{visit_id}_descriptions.json").write_text(
        json.dumps(
            {
                "visit_id": visit_id,
                "descriptions": [
                    {
                        "desc_id": desc_id,
                        "annot_id": annot_id,
                        "description": "Open the lower drawer.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / f"{visit_id}_motions.json").write_text(
        json.dumps(
            {
                "visit_id": visit_id,
                "motions": [
                    {
                        "motion_id": "motion-a",
                        "annot_id": annot_id,
                        "motion_type": "pinch_pull",
                        "motion_dir": [1.0, 0.0, 0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / f"{visit_id}_annotations.json").write_text(
        json.dumps(
            {
                "visit_id": visit_id,
                "annotations": [{"annot_id": annot_id, "indices": list(gt_point_ids)}],
            }
        ),
        encoding="utf-8",
    )
    write_binary_ply_mesh(
        scene_dir / "raw" / "mesh.ply",
        np.random.default_rng(0).random((vertex_count, 3)).astype(np.float32),
    )
    return scene_dir
```

- [ ] **Step 2: Write the failing scorer test**

```python
# src/codex_agent/tests/test_scenefunc3d_scorer_pipeline.py
"""Tests for scoring Phase-2c pipeline result files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from codex_agent.scenefunc3d.backends.lift_3d import write_lift_npz
from codex_agent.scenefunc3d.evaluation.scorer import (
    score_pipeline_result_file,
    score_result_file,
)
from codex_agent.tests.scenefunc3d_scene_fixtures import write_scoring_scene


def _pipeline_result_json(result_path: Path, *, mask_npz_path: Path) -> None:
    result_path.write_text(
        json.dumps(
            {
                "task_name": "scenefunc3d_mask_generation",
                "sample_id": "421254::desc-a",
                "result_kind": "anchor_multiview_pipeline",
                "outcome": {
                    "result_kind": "anchor_multiview_pipeline",
                    "mask_npz_path": str(mask_npz_path),
                    "mask_ply_path": str(mask_npz_path.with_name("lifted_points.ply")),
                    "confidence": 0.5,
                    "selected_frame_ids": ["000010"],
                    "seed": {
                        "seed_frame_id": "000010",
                        "seed_candidate_id": "cand-a",
                        "seed_fragment_id": "000010_cand-a",
                        "affordance_concept": "the handle",
                        "seed_point_count": 2,
                    },
                    "per_frame": [],
                    "fusion_params": {
                        "agreement_tau": 0.5,
                        "radius_scale": 1.0,
                        "cluster_link_eps_m": 0.02,
                        "min_cluster_points": 10,
                    },
                    "uncertainties": [],
                },
                "turn": {
                    "turn_id": "t",
                    "status": "completed",
                    "duration_ms": None,
                    "reasoning_summary": None,
                    "run_home": None,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_fused(npz_path: Path, ids: tuple[int, ...]) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    write_lift_npz(
        npz_path,
        np.zeros((len(ids), 3), dtype=np.float64),
        point_indices=np.array(ids, dtype=np.int64),
    )


def test_score_pipeline_result_file_scores_fused_npz(tmp_path: Path) -> None:
    write_scoring_scene(tmp_path / "data", gt_point_ids=(0, 1, 2))
    fused = tmp_path / "run" / "fused" / "mask_data.npz"
    _write_fused(fused, (0, 1, 2))
    result_path = tmp_path / "run" / "result.json"
    _pipeline_result_json(result_path, mask_npz_path=fused)
    score = score_pipeline_result_file(data_root=tmp_path / "data", result_path=result_path)
    assert score.metrics.iou == pytest.approx(1.0)


def test_score_result_file_dispatches_on_result_kind(tmp_path: Path) -> None:
    write_scoring_scene(tmp_path / "data", gt_point_ids=(0, 1, 2))
    fused = tmp_path / "run" / "fused" / "mask_data.npz"
    _write_fused(fused, (0, 1))
    result_path = tmp_path / "run" / "result.json"
    _pipeline_result_json(result_path, mask_npz_path=fused)
    score = score_result_file(data_root=tmp_path / "data", result_path=result_path)
    assert score.metrics.iou == pytest.approx(2.0 / 3.0)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_scorer_pipeline.py -v`
Expected: FAIL with `ImportError: cannot import name 'score_pipeline_result_file'`.

- [ ] **Step 4: Add the pipeline scorer + dispatch to `scorer.py`**

Add after `_ScoringResultFile` (line 73):

```python
class _PipelineScoringOutcome(BaseModel):
    """Subset of a pipeline result outcome needed by the scorer."""

    model_config = ConfigDict(extra="ignore")

    mask_npz_path: NonEmptyString


class _PipelineScoringResultFile(BaseModel):
    """Subset of a Phase-2c pipeline result JSON needed by the scorer."""

    model_config = ConfigDict(extra="ignore")

    task_name: NonEmptyString
    sample_id: NonEmptyString
    result_kind: NonEmptyString
    outcome: _PipelineScoringOutcome


_PIPELINE_RESULT_KIND = "anchor_multiview_pipeline"


def score_pipeline_result_file(
    *, data_root: Path, result_path: Path, failure_type: str = ""
) -> SceneFunc3dScore:
    """Score a Phase-2c anchor multi-view pipeline result against hidden GT."""
    result_file = _load_pipeline_result_file(result_path)
    if result_file.task_name != _SCENEFUNC3D_TASK_NAME:
        raise SceneFunc3dDataError(
            "runner result task_name is not a SceneFun3D mask task: "
            f"task_name={result_file.task_name!r}; result_path={result_path}"
        )
    mask_npz_path = _resolve_artifact_member_path(
        artifact_path=result_path,
        raw_member_path=result_file.outcome.mask_npz_path,
    )
    return score_mask_npz(
        data_root=data_root,
        sample_id=result_file.sample_id,
        mask_npz_path=mask_npz_path,
        failure_type=failure_type,
    )


def _load_pipeline_result_file(result_path: Path) -> _PipelineScoringResultFile:
    if not result_path.is_file():
        raise SceneFunc3dDataError(f"runner result JSON is missing: {result_path}")
    try:
        payload: object = json.loads(result_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise SceneFunc3dDataError(
            f"runner result is not valid JSON: {result_path}"
        ) from exc
    try:
        return _PipelineScoringResultFile.model_validate(payload)
    except ValidationError as exc:
        raise SceneFunc3dDataError(
            "pipeline runner result failed scorer validation: "
            f"result_path={result_path}; error={exc}"
        ) from exc


def _result_kind(result_path: Path) -> str:
    if not result_path.is_file():
        raise SceneFunc3dDataError(f"runner result JSON is missing: {result_path}")
    try:
        payload: object = json.loads(result_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise SceneFunc3dDataError(
            f"runner result is not valid JSON: {result_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise SceneFunc3dDataError(f"runner result must be a JSON object: {result_path}")
    raw_kind = payload.get("result_kind", "")
    return raw_kind if isinstance(raw_kind, str) else ""
```

Rename the existing `score_result_file` body to `_score_legacy_result_file` (byte-for-byte, only the `def` line changes), then add the dispatcher:

```python
def score_result_file(
    *, data_root: Path, result_path: Path, failure_type: str = ""
) -> SceneFunc3dScore:
    """Score a SceneFun3D runner result JSON (pipeline or legacy) against GT."""
    if _result_kind(result_path) == _PIPELINE_RESULT_KIND:
        return score_pipeline_result_file(
            data_root=data_root, result_path=result_path, failure_type=failure_type
        )
    return _score_legacy_result_file(
        data_root=data_root, result_path=result_path, failure_type=failure_type
    )
```

Add `"score_pipeline_result_file"` to `scorer.__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_scorer_pipeline.py -v`
Expected: PASS (2 passed). Then regression: `PYTHONPATH=src pytest src/codex_agent/tests/ -k "scorer or evaluation or offline_fusion or frame_loader" -q` (Expected: PASS).

- [ ] **Step 6: Commit**

```bash
git add src/codex_agent/scenefunc3d/evaluation/scorer.py \
        src/codex_agent/scenefunc3d/tests_support.py \
        src/codex_agent/tests/scenefunc3d_scene_fixtures.py \
        src/codex_agent/tests/scenefunc3d_synthetic_scene.py \
        src/codex_agent/tests/test_scenefunc3d_scorer_pipeline.py
git commit -m "feat(scenefunc3d): score pipeline result files via fused NPZ"
```

---

## Task 5: Flip the runner to the 2D-seed + pipeline contract (`runner.py`)

The crux: repoint `SceneFunc3dMaskTask` to `SeedConfirmation`, add `_lift_seed_fragment` + `_run_pipeline_for_seed`, rewrite `run_single_sample`, and swap the old runner tests. Dead free-form/fuse types are removed in Task 7.

**Files:**
- Modify: `src/codex_agent/scenefunc3d/runner.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_runner.py`

- [ ] **Step 1: Add imports + constants + the two seams to `runner.py`**

At module scope, add near `from .sample import ...` (line ~55):

```python
from .backends.anchor import build_anchor
from .backends.frame_assets import resolve_frame_geometry_assets
from .backends.fusion import FusionParams
from .backends.lift_3d import load_scene_mesh_vertices
from .final_mask_artifacts import load_points_world_npz
from .pipeline import run_anchor_multiview_pipeline
from .pipeline_backends import SidecarFrameProposer
from .pipeline_result import (
    SceneFunc3dPipelineOutcome,
    write_pipeline_fused_mask,
    write_pipeline_result,
)
from .seed_contract import (
    SceneFunc3dSeedOutcome,
    SeedConfirmation,
    parse_seed_confirmation,
)
from .seed_recovery import recover_seed_sam_candidate
from .tools.mask_lifting import LiftMaskArgs, LiftMaskResult, lift_mask_to_3d
```

Add `MaskArtifactPaths` to the existing `from .tools.mask_artifacts import (...)` block (line 58). Add constants near `TASK_NAME` (line ~68):

```python
_PIPELINE_FRAME_CAP = 30
_PIPELINE_DEPTH_TOLERANCE = 0.25
```

- [ ] **Step 2: Rewrite the task's turn/parse methods**

Replace `build_turn_request` (lines 715–722):

```python
    def build_turn_request(self) -> CodexTurnRequest:
        self._require_tool_context_file()
        return CodexTurnRequest(
            prompt=self._build_prompt(),
            output_schema=SeedConfirmation.model_json_schema(),
            skills=(),
            image_paths=(),
        )
```

Replace `parse_response`/`validate_outcome`/`_parse_decision` (lines 739–795) and delete `_validate_artifact_path` (797–826) + `_validate_selected_frame_ids` (828–846):

```python
    def is_valid_response(self, response_text: str) -> bool:
        try:
            self.parse_response(response_text)
        except CodexResponseError:
            return False
        return True

    def parse_response(self, response_text: str) -> SceneFunc3dSeedOutcome:
        confirmation = parse_seed_confirmation(response_text)
        seed_candidate = recover_seed_sam_candidate(
            self.output_dir / "events.jsonl",
            seed_frame_id=confirmation.seed_frame_id,
            seed_candidate_id=confirmation.seed_candidate_id,
        )
        return SceneFunc3dSeedOutcome(
            seed_frame_id=confirmation.seed_frame_id,
            seed_candidate_id=confirmation.seed_candidate_id,
            seed_fragment_id=confirmation.seed_fragment_id,
            affordance_concept=confirmation.affordance_concept,
            approval_actions=confirmation.approval_actions,
            uncertainties=confirmation.uncertainties,
            seed_sam_mask_npz_path=seed_candidate.mask_npz_path,
        )
```

Replace `_build_prompt` (line 848) to reference the seed schema and the 2D-only tool list:

```python
    def _build_prompt(self) -> str:
        schema = SeedConfirmation.model_json_schema()
        return (
            build_prompt(self.sample) + "\n\nRuntime paths:\n"
            f"- tool_context: {self.tool_context_path}\n"
            "\nHow to run a tool (in the shell):\n"
            f"- invoke: python -m {self.tool_cli_module} <tool> "
            f"--context {self.tool_context_path} --args '<json>'\n"
            "- For molmo_point and sam_mask, set yield_time_ms=30000 on the shell "
            "call so the command can finish and return JSON. If the shell reports "
            "Process running with session ID, wait on that same session until it "
            "exits and returns JSON; do not start another SceneFun3D tool while a "
            "previous tool command is still running.\n"
            "- Tool outputs are JSON. Open any returned image_path with view_image "
            "before relying on visual evidence.\n"
            "- Use the Molmo point and SAM candidate approval gates in the playbook "
            "before confirming the seed on the original-image SAM overlay.\n"
            "\n" + _hard_limits_section() + "\n"
            "\nFinal JSON schema:\n" + json.dumps(schema, ensure_ascii=False)
        )
```

- [ ] **Step 3: Rewrite `run_single_sample`'s execute→write body + add the seams**

Replace the try/except execute block + result writing (lines 963–1031). Keep setup (937–962) and the three failure handlers (they cover `CodexTurnError`/`CodexResponseError`/`CodexAgentError`; `SceneFunc3dDataError` subclasses `CodexAgentError`):

```python
    turn_metadata: CodexTurnMetadata | None = None
    try:
        result = executor.execute(task)
        turn_metadata = result.turn.metadata
        pipeline_outcome = _run_pipeline_for_seed(
            sample=sample,
            scene_root=scene_dir_for(config.dataset_root, sample.visit_id),
            output_dir=sample_output_dir,
            backend_config_path=config.backend_config_path,
            seed_outcome=result.outcome,
            artifact_paths=artifact_paths,
        )
    except CodexTurnError as exc:
        _write_sample_failure_artifact(
            sample=sample,
            failure_path=sample_output_dir / "failure.json",
            failure_stage="codex_turn",
            exc=exc,
            events_path=artifact_paths.events_jsonl,
            tool_context_path=tool_context_path,
            turn_metadata=turn_metadata,
        )
        raise
    except CodexResponseError as exc:
        _write_sample_failure_artifact(
            sample=sample,
            failure_path=sample_output_dir / "failure.json",
            failure_stage="response_validation",
            exc=exc,
            events_path=artifact_paths.events_jsonl,
            tool_context_path=tool_context_path,
            turn_metadata=turn_metadata,
        )
        raise
    except CodexAgentError as exc:
        _write_sample_failure_artifact(
            sample=sample,
            failure_path=sample_output_dir / "failure.json",
            failure_stage="run",
            exc=exc,
            events_path=artifact_paths.events_jsonl,
            tool_context_path=tool_context_path,
            turn_metadata=turn_metadata,
        )
        raise
    result_path = write_pipeline_result(
        sample_output_dir / "result.json",
        pipeline_outcome,
        sample_id=sample_id,
        turn_metadata=result.turn.metadata,
    )
    append_run_completion_event(
        artifact_paths,
        SceneFunc3dRunCompletionEvent(
            sample_id=sample_id,
            result_path=result_path,
            summary_path=result_path,
            mask_artifact_path=pipeline_outcome.mask_npz_path,
        ),
    )
    return result_path
```

Add these functions immediately after `run_single_sample` (before `run_samples`, line ~1033):

```python
def _run_pipeline_for_seed(
    *,
    sample: SceneFunc3dSample,
    scene_root: Path,
    output_dir: Path,
    backend_config_path: Path,
    seed_outcome: SceneFunc3dSeedOutcome,
    artifact_paths: MaskArtifactPaths,
) -> SceneFunc3dPipelineOutcome:
    """Deterministically lift the confirmed 2D seed, then run the multi-view pipeline."""
    scene = SceneFunc3dToolScene.load(scene_root)
    scene_vertices = load_scene_mesh_vertices(scene.raw_mesh_path)
    seed_lift = _lift_seed_fragment(
        scene=scene, seed_outcome=seed_outcome, out_dir=output_dir
    )
    seed_points = load_points_world_npz(seed_lift.mask_npz_path)
    anchor = build_anchor(
        seed_points,
        motion_type=_anchor_motion_type(sample),
        seed_frame_id=seed_outcome.seed_frame_id,
    )
    proposer = SidecarFrameProposer(
        scene=scene,
        out_dir=output_dir,
        backend_config_path=backend_config_path,
        affordance_concept=seed_outcome.affordance_concept,
        task_description=sample.task_description,
    )
    result = run_anchor_multiview_pipeline(
        anchor=anchor,
        scene=scene,
        scene_vertices=scene_vertices,
        proposer=proposer,
        params=FusionParams(),
        frame_cap=_PIPELINE_FRAME_CAP,
        depth_tolerance=_PIPELINE_DEPTH_TOLERANCE,
    )
    mask_npz_path, mask_ply_path = write_pipeline_fused_mask(
        artifact_paths.fused_dir,
        point_indices=result.fused.point_indices,
        scene_vertices=scene_vertices,
    )
    return SceneFunc3dPipelineOutcome(
        seed=seed_outcome,
        seed_point_count=seed_lift.lifted_point_count,
        mask_npz_path=mask_npz_path,
        mask_ply_path=mask_ply_path,
        result=result,
    )


def _lift_seed_fragment(
    *,
    scene: SceneFunc3dToolScene,
    seed_outcome: SceneFunc3dSeedOutcome,
    out_dir: Path,
) -> LiftMaskResult:
    """Lift the confirmed 2D SAM mask to a raw-mesh fragment (deterministic)."""
    assets = resolve_frame_geometry_assets(scene, seed_outcome.seed_frame_id)
    return lift_mask_to_3d(
        LiftMaskArgs(
            frame_id=seed_outcome.seed_frame_id,
            candidate_id=seed_outcome.seed_candidate_id,
            mask_path=seed_outcome.seed_sam_mask_npz_path,
            depth_path=assets.depth_path,
            intrinsics_path=assets.intrinsics_path,
            pose_path=assets.pose_path,
        ),
        out_dir=out_dir,
        raw_mesh_path=scene.raw_mesh_path,
    )


def _anchor_motion_type(sample: SceneFunc3dSample) -> str:
    """Return the seed anchor's motion type (first motion hint, else empty)."""
    if sample.motion_hints:
        return sample.motion_hints[0].motion_type
    return ""
```

- [ ] **Step 4: Swap the runner tests to the seed + pipeline contract**

In `src/codex_agent/tests/test_scenefunc3d_runner.py`:

**Delete** the free-form + fuse-provenance tests (removed contract): every test from `test_mask_task_parses_strict_final_json` (line 528) through `test_run_single_sample_accepts_standard_fragment_when_later_fuse_has_ordered_chain` (line 2245) inclusive. Also delete their now-unused helpers (`_write_outcome_artifacts`, `_write_standard_outcome_artifacts`, `_outcome_payload`, `_write_review_artifacts`, `_test_lift_geometry_payload`, `_write_fuse_accepted_masks_event`, `_write_suggest_additional_views_event`, `_DynamicSceneFuncExecutor`, `_FakeExecutor`) — but only after `rg`-confirming no kept test references them. **Keep** the arg-parser, `main`, `run_from_args`, sidecar-health, and `run_samples`-worker tests; the `run_samples` worker tests that use `_DynamicSceneFuncExecutor` must switch to `_SeedDynamicExecutor` below.

Add near the top (after the imports, replacing `_APPROVED_FRAGMENT_ACTIONS`):

```python
from codex_agent.scenefunc3d.task import SEED_APPROVAL_ACTIONS

_SEED_APPROVAL_VALUES = tuple(action.value for action in SEED_APPROVAL_ACTIONS)
```

Add the seed executor + helpers + fake seams at the bottom:

```python
class _SeedFakeExecutor:
    """Fake executor returning a canned SeedConfirmation final response."""

    def __init__(
        self, response_payload: Mapping[str, object], *, on_execute: Callable[[], None]
    ) -> None:
        self._response_payload = dict(response_payload)
        self._on_execute = on_execute
        self.task_name = ""
        self.prompt = ""

    def execute(self, task: CodexTask[ResultT]) -> CodexTaskResult[ResultT]:
        request = task.build_turn_request()
        self.task_name = task.task_name
        self.prompt = request.prompt
        self._on_execute()
        outcome = task.parse_response(json.dumps(self._response_payload))
        return CodexTaskResult(
            task_name=task.task_name,
            outcome=cast(ResultT, outcome),
            turn=CodexTurnResult(
                final_response=json.dumps(self._response_payload),
                metadata=CodexTurnMetadata(turn_id="fake-turn", status="completed"),
            ),
        )


def _seed_confirmation_payload(
    *, seed_frame_id: str = "000010", seed_candidate_id: str = "cand-a"
) -> dict[str, object]:
    return {
        "seed_frame_id": seed_frame_id,
        "seed_candidate_id": seed_candidate_id,
        "affordance_concept": "the drawer handle",
        "approval_actions": list(_SEED_APPROVAL_VALUES),
        "uncertainties": ["partial occlusion"],
    }


def _write_seed_sam_event(
    output_dir: Path, *, frame_id: str = "000010", candidate_id: str = "cand-a"
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event_type": "tool_completed",
                    "tool_name": "sam_mask",
                    "status": "success",
                    "args": {"frame_id": frame_id, "image_path": f"{frame_id}-rgb.png"},
                    "result": {
                        "frame_id": frame_id,
                        "candidates": [
                            {
                                "candidate_id": candidate_id,
                                "mask_npz_path": str(
                                    output_dir / "sam" / frame_id / f"{candidate_id}.npz"
                                ),
                                "overlay_path": f"{candidate_id}_overlay.jpg",
                            }
                        ],
                        "contact_sheet_path": "contact_sheet.jpg",
                    },
                    "error": "",
                }
            )
            + "\n"
        )


def _fake_lift_seed_fragment(fragment_npz: Path) -> Callable[..., object]:
    from codex_agent.scenefunc3d.tools.mask_lifting import LiftMaskResult

    def _fake(*, scene: object, seed_outcome: object, out_dir: Path) -> LiftMaskResult:
        return LiftMaskResult(
            frame_id="000010",
            candidate_id="cand-a",
            lifted_point_count=2,
            mask_npz_path=fragment_npz,
            mask_ply_path=fragment_npz.with_name("lifted_points.ply"),
            overlay_path=fragment_npz.with_name("lift_overlay.txt"),
        )

    return _fake


def _fake_pipeline_result(point_indices: tuple[int, ...]) -> object:
    from codex_agent.scenefunc3d.backends.fusion import FusedMask, FusionParams
    from codex_agent.scenefunc3d.pipeline import (
        AnchorMultiViewResult,
        PerFrameOutcome,
    )

    return AnchorMultiViewResult(
        fused=FusedMask(
            point_indices=point_indices,
            confidence=0.66,
            instances=(),
            params=FusionParams(),
        ),
        selected_frames=(),
        per_frame=(
            PerFrameOutcome(
                frame_id="000010",
                point_indices=point_indices,
                centroid=(0.0, 0.0, 2.0),
                accepted=True,
                reject_reason="",
                molmo_fallback_used=False,
            ),
        ),
    )


def _write_seed_fragment_npz(path: Path) -> Path:
    from codex_agent.scenefunc3d.backends.lift_3d import write_lift_npz

    path.parent.mkdir(parents=True, exist_ok=True)
    return write_lift_npz(
        path,
        np.array([[0.0, 0.0, 2.0], [0.01, 0.0, 2.0]], dtype=np.float64),
        point_indices=np.array([0, 1], dtype=np.int64),
    )
```

Add the new tests (place after the kept `run_samples` tests):

```python
def test_run_single_sample_writes_pipeline_result_from_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_scene(tmp_path / "data")
    sample_output_dir = tmp_path / "out" / "421254" / "desc-a"
    fragment_npz = _write_seed_fragment_npz(tmp_path / "seed_fragment.npz")

    monkeypatch.setattr(runner, "_lift_seed_fragment", _fake_lift_seed_fragment(fragment_npz))
    monkeypatch.setattr(
        runner, "run_anchor_multiview_pipeline", lambda **_: _fake_pipeline_result((0, 1))
    )
    executor = _SeedFakeExecutor(
        _seed_confirmation_payload(),
        on_execute=lambda: _write_seed_sam_event(sample_output_dir),
    )
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    result_path = run_single_sample(
        config, sample_id="421254::desc-a", executor=executor, check_sidecars=False
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["result_kind"] == "anchor_multiview_pipeline"
    assert payload["outcome"]["seed"]["seed_fragment_id"] == "000010_cand-a"
    assert payload["outcome"]["seed"]["seed_point_count"] == 2
    assert payload["outcome"]["selected_frame_ids"] == ["000010"]
    assert payload["outcome"]["confidence"] == pytest.approx(0.66)
    fused_npz = sample_output_dir / "fused" / "mask_data.npz"
    assert fused_npz.is_file()
    assert payload["outcome"]["mask_npz_path"] == str(fused_npz)


def test_run_single_sample_fails_closed_on_missing_seed_sam_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_scene(tmp_path / "data")
    monkeypatch.setattr(
        runner,
        "_lift_seed_fragment",
        lambda **_: (_ for _ in ()).throw(AssertionError("must not lift")),
    )
    executor = _SeedFakeExecutor(_seed_confirmation_payload(), on_execute=lambda: None)
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )
    with pytest.raises(CodexResponseError, match="no successful sam_mask"):
        run_single_sample(
            config, sample_id="421254::desc-a", executor=executor, check_sidecars=False
        )
    assert (tmp_path / "out" / "421254" / "desc-a" / "failure.json").is_file()


def test_run_single_sample_fails_closed_on_empty_fused_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_scene(tmp_path / "data")
    sample_output_dir = tmp_path / "out" / "421254" / "desc-a"
    fragment_npz = _write_seed_fragment_npz(tmp_path / "seed_fragment.npz")
    monkeypatch.setattr(runner, "_lift_seed_fragment", _fake_lift_seed_fragment(fragment_npz))
    monkeypatch.setattr(
        runner, "run_anchor_multiview_pipeline", lambda **_: _fake_pipeline_result(())
    )
    executor = _SeedFakeExecutor(
        _seed_confirmation_payload(),
        on_execute=lambda: _write_seed_sam_event(sample_output_dir),
    )
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )
    with pytest.raises(SceneFunc3dDataError, match="empty fused mask"):
        run_single_sample(
            config, sample_id="421254::desc-a", executor=executor, check_sidecars=False
        )
```

For the kept `run_samples` worker tests, replace `_DynamicSceneFuncExecutor` with `_SeedDynamicExecutor` (writes the SAM event into `task.output_dir`, returns the seed outcome via `task.parse_response`) and monkeypatch `runner._lift_seed_fragment` + `runner.run_anchor_multiview_pipeline` in those tests (or, simpler, monkeypatch `runner.run_single_sample` where those tests only assert worker scheduling — check each test's intent and keep the lightest change).

The seam: `_run_pipeline_for_seed` calls the module-globals `_lift_seed_fragment` and `run_anchor_multiview_pipeline`, so `monkeypatch.setattr(runner, ...)` intercepts both; the real `SidecarFrameProposer` is constructed but never `.propose()`-d, and no depth/intrinsic/pose files are needed. `build_anchor` runs for real on the canned 2-point fragment; `write_pipeline_fused_mask` runs for real against the binary 100-vertex `raw/mesh.ply`.

- [ ] **Step 5: Run the runner tests**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_runner.py -v`
Expected: PASS (kept CLI/health/batch tests + 3 new pipeline tests). Fix any kept test still referencing a deleted helper.

- [ ] **Step 6: Commit**

```bash
git add src/codex_agent/scenefunc3d/runner.py \
        src/codex_agent/tests/test_scenefunc3d_runner.py
git commit -m "feat(scenefunc3d): lift confirmed 2D seed in runner and drive pipeline"
```

---

## Task 6: Trim the playbook to the 2D seed stage (`playbook.py`)

**Files:**
- Modify: `src/codex_agent/scenefunc3d/playbook.py`
- Modify: `src/codex_agent/tests/test_scenefunc3d_playbook.py`

- [ ] **Step 1: Update the playbook test first**

Add to `test_scenefunc3d_playbook.py` (and remove/adjust any assertion requiring `fuse_accepted_masks`/`suggest_additional_views`/`lift_mask_to_3d`/`inspect_mask_artifact` wording):

```python
def test_playbook_tool_names_are_2d_seed_only() -> None:
    from codex_agent.scenefunc3d.playbook import SCENEFUNC3D_TOOL_NAMES

    for dropped in (
        "lift_mask_to_3d",
        "inspect_mask_artifact",
        "suggest_additional_views",
        "fuse_accepted_masks",
    ):
        assert dropped not in SCENEFUNC3D_TOOL_NAMES
    assert SCENEFUNC3D_TOOL_NAMES[-1] == "sam_mask"


def test_playbook_ends_at_seed_confirmation() -> None:
    from codex_agent.scenefunc3d.playbook import SCENEFUNC3D_TOOLS_PLAYBOOK

    assert "seed_frame_id" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "seed_candidate_id" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "affordance_concept" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "lift_mask_to_3d" not in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "fuse_accepted_masks" not in SCENEFUNC3D_TOOLS_PLAYBOOK
```

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_playbook.py -v` (Expected: FAIL until Step 2).

- [ ] **Step 2: Trim `SCENEFUNC3D_TOOL_NAMES` and rewrite the playbook**

```python
SCENEFUNC3D_TOOL_NAMES: tuple[str, ...] = (
    "scene_summary",
    "keyframe_selector",
    "view_frame",
    "view_bev",
    "frame_objects",
    "molmo_point",
    "sam_mask",
)
```

Rewrite `SCENEFUNC3D_TOOLS_PLAYBOOK`: keep the intro + tool catalog (only the 7 kept tools), the `molmo_point`/`sam_mask` arg specs, and approval gates 1–2 (Molmo point → SAM candidate) with the small-affordance + "≤3 Molmo attempts / change prompt or frame" guidance. Remove gate 3 (3D lift), gate 4 (multi-view), `suggest_additional_views`, `fuse_accepted_masks`, and the whole final-artifact tail. End with:

```
After you approve one compact SAM candidate whose mask (drawn on the original
image) covers only the target operable part, STOP. Do not lift to 3D, do not
inspect 3D geometry, do not search more frames, and do not fuse. The runner
takes your confirmed 2D seed, lifts it to 3D deterministically, and completes
the multi-view 3D mask.

Judge the SAM candidate on the original-image overlay: reject a broad panel,
body, cabinet face, wall patch, pipe run, or shadow even if its SAM score is
higher; prefer a compact mask that tightly covers the knob/handle/dial/switch/
button/valve.

Final answer must be compact JSON matching the SeedConfirmation schema:
seed_frame_id, seed_candidate_id, affordance_concept, approval_actions,
uncertainties.
- seed_frame_id / seed_candidate_id: the frame_id you worked on and the
  approved SAM candidate_id from sam_mask.
- affordance_concept: the short affordance noun phrase you targeted (e.g. "the
  drawer handle"); the runner reuses it verbatim as the Molmo prompt subject on
  every additional view, so make it specific and free of task verbs.
- approval_actions: exactly, in order: select_evidence, propose_molmo_point,
  approve_molmo_point, propose_sam_candidates, approve_sam_candidate.
- uncertainties: short notes on anything ambiguous about the seed.
You must have approved the SAM candidate on seed_frame_id before emitting this
JSON; the runner verifies that sam_mask event exists.
```

- [ ] **Step 3: Run the playbook + prompt tests**

Run: `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_playbook.py src/codex_agent/tests/test_scenefunc3d_runner.py -k "playbook or prompt" -v`
Expected: PASS. Update the kept `test_mask_task_prompt_inlines_tools_without_attachments` if it hard-codes old tool names (it iterates `SCENEFUNC3D_ALLOWED_TOOL_NAMES`, which now excludes the dropped tools; the `"SAM candidates"`/`"Molmo point"` prompt asserts still hold).

- [ ] **Step 4: Commit**

```bash
git add src/codex_agent/scenefunc3d/playbook.py \
        src/codex_agent/tests/test_scenefunc3d_playbook.py \
        src/codex_agent/tests/test_scenefunc3d_runner.py
git commit -m "feat(scenefunc3d): trim playbook to 2D seed-confirmation stage"
```

---

## Task 7: Remove the dead free-form / fuse machinery (`runner.py`)

Now that the task + `run_single_sample` no longer touch the free-form contract, delete the unreachable code. **Gate every deletion on `rg`** (runner, tools, scorer, tests).

**Files:**
- Modify: `src/codex_agent/scenefunc3d/runner.py`

- [ ] **Step 1: Enumerate dead symbols** — for each, `rg -n "<symbol>" src/` and confirm the only hit is its own definition:

- Free-form decision/outcome: `SceneFunc3dMaskDecision`, `SceneFunc3dMaskOutcome`, `SceneFunc3dMaskOutcomePayload`, `SceneFunc3dRunResultPayload`, `SceneFunc3dRunArtifactPayload`, `_artifact_payload`, `_completed_run_summary`, `_validate_outcome_artifact`, `_require_non_empty_string_tuple`.
- Fuse self-report: `_FuseToolResult`, `_FuseToolEvent`, `_ParsedFuseToolEvent`, `_accepted_frame_ids_from_fuse_fragments`, `_validate_outcome_against_fuse_tool_event`, `_last_matching_fuse_tool_event`, `_infer_output_root_from_outcome`, `_load_mask_artifact_document_for_provenance`, `_successful_fuse_tool_events`, `_parse_fuse_tool_event`, `_fuse_event_matches_outcome`, `_fuse_event_matches_artifact_document`, `_validate_standard_fragment_upstream_tool_events`, `_seed_standard_fragment_tool_chain`, `_reject_seed_suggest_events_before_inspection`, `_require_matching_standard_fragment_tool_chain`, `_standard_fragment_candidate_id`, `_standard_fragment_mask_npz_path`, `_standard_fragment_mask_ply_path`, `_matching_molmo_point_events`, `_raise_missing_molmo_point_event`, `_event_path_matches`, `_validate_multiview_decision_against_tool_events`.
- Tool-event models + parsed dataclasses (lines 260–549): `_ToolPointAliasPayload`, `_ToolPointPayload`, `_EvidenceImageFrameResult`, `_EvidenceImageToolResult`, `_MolmoPointToolArgs`, `_MolmoPointToolResult`, `_SamCandidateToolResult`, `_SamMaskToolResult`, `_SamMaskToolArgs`, `_LiftMaskToolResult`, `_LiftMaskToolArgs`, `_InspectMaskToolArgs`, `_InspectMaskToolResult`, `_SuggestAdditionalViewsToolArgs`, `_infer_frame_id_from_seed_fragment`, and the `_Parsed*ToolEvent` / `_Matched*` / `_ValidatedStandardFragmentToolChain` / `_SuccessfulToolResultEvent` dataclasses.
- Generic event readers `_load_event_payload`, `_require_mapping_field`, `_require_json_mapping`, `_tool_point_alias_payload_from_xy` — `rg` and delete only if unreferenced after the above go.

**Keep**: `_metadata_payload` (used by `_failure_turn_payload`), `_initialize_run_events`, `SceneFunc3dRunnerConfig`, `SceneFunc3dBatchSampleResult`/`Summary`, batch functions, CLI (`build_arg_parser`/`main`/`run_from_args`), sidecar health.

- [ ] **Step 2: Prune imports + `__all__`**

Drop from the `from .final_mask_artifacts import (...)` block everything except `load_points_world_npz`. Drop `SceneFunc3dCompletedRunSummary`, `SceneFunc3dRunMultiViewDecisionPayload`, `write_completed_run_summary` from the mask_artifacts import if now unused (`rg`). Drop `AliasChoices`, `StringConstraints`, `model_validator`, `Annotated`, `NonEmptyString`, `_POINT_MATCH_ABS_TOLERANCE_PX` if unreferenced. Update `runner.__all__`: remove `SceneFunc3dMaskDecision`, `SceneFunc3dMaskOutcome`, `SceneFunc3dMaskOutcomePayload`, `SceneFunc3dRunArtifactPayload`, `SceneFunc3dRunResultPayload`; add `SceneFunc3dSeedOutcome`.

- [ ] **Step 3: Verify**

`PYTHONPATH=src python -c "import codex_agent.scenefunc3d.runner"` (no ImportError) → `ruff check src/codex_agent/scenefunc3d/runner.py` (no F401/F811) → `PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_runner.py -q` (PASS).

- [ ] **Step 4: Commit**

```bash
git add src/codex_agent/scenefunc3d/runner.py
git commit -m "refactor(scenefunc3d): remove dead free-form/fuse self-report machinery"
```

---

## Task 8: Quality gate, online smoke, and docs

**Files:** verification + docs only.

- [ ] **Step 1: Full quality gate**

```bash
source .venv/bin/activate
ruff check src/
black --check src/codex_agent/scenefunc3d/ src/codex_agent/tests/
mypy src/codex_agent/scenefunc3d/
PYTHONPATH=src pytest src/codex_agent/tests/ -k "scenefunc3d" -q
```
Expected: all clean/pass. Fix + commit any gate failures: `git commit -m "fix(scenefunc3d): quality gate for phase 2c contract"`.

- [ ] **Step 2 (optional, needs sidecars + ModelHub adapter — run in tmux): one-sample online smoke**

Per `AGENTS.md`: confirm the adapter (`curl -sf http://127.0.0.1:8787/health`) + Molmo/SAM sidecars are healthy, then run one sample end-to-end on `421254`:

```bash
tmux new-session -d -s sf3d_2c "source .venv/bin/activate && \
  PYTHONPATH=src python -m codex_agent.scenefunc3d.runner \
    --dataset-root data/SceneFun3D \
    --sample-id 421254::<desc_id> \
    --output-dir tmp/scenefunc3d/artifacts/sf3d_phase2c_$(date +%Y%m%d)/ \
    --backend-config <backend.toml> 2>&1 | tee /tmp/sf3d_2c.log"
```
Monitor: `tmux capture-pane -t sf3d_2c -p -S -30`. If unavailable, **skip** and say so explicitly — the unit path proves the contract + wiring. Do not fabricate numbers.

- [ ] **Step 3: Write the benchmark doc + pointers**

Create `docs/benchmark/scenefunc_molmo_sam3d/v8_seed_confirmation_online_<YYYYMMDD>.md` (branch + tip commit at launch, exact CLI, raw artifact dir, sample used, what changed vs v7 — "agent confirms a 2D seed only; runner lifts it + runs the deterministic anchor multi-view pipeline; scoring via `result_kind=anchor_multiview_pipeline` + `score_mask_npz`", headline IoU/precision if Step 2 ran, caveats). Update `docs/benchmark/scenefunc_molmo_sam3d/README.md`. Add to the Phase-2b plan a line: "Implemented in `docs/superpowers/plans/2026-07-02-scenefunc3d-seed-confirmation-contract-phase2c.md`."

```bash
git add docs/benchmark/scenefunc_molmo_sam3d/ \
        docs/superpowers/plans/2026-07-02-scenefunc3d-anchor-multiview-pipeline-phase2b.md
git commit -m "docs(scenefunc3d): record v8 2D seed-confirmation contract"
```

---

## Self-Review

**1. Spec coverage** (spec §"边界" small seed-confirmation JSON; §"模块/文件划分"; §"错误处理" fail-closed):
- Boundary shrinks to a small 2D seed JSON: Task 1. ✓
- Runner turn contract emits `SeedConfirmation`; `parse_response` verifies the confirmed SAM candidate exists in `events.jsonl`: Tasks 5 + 2. ✓
- Agent does no 3D work; the deterministic lift moves into the runner, builds the anchor, runs B/C/D, writes the final artifact + `result.json`; scorer keeps working (metrics unchanged): Tasks 5 + 3 + 4. ✓
- `playbook.py` trimmed to the 2D seed stage (drops lift/inspect/multiview/fuse): Task 6. ✓
- FSM reused via new `validate_seed_approval_actions` ending at `SAM_CANDIDATE_AGENT_APPROVED`: Task 1. ✓
- Fail-closed: missing SAM seed → `CodexResponseError` → `failure.json`; empty fused mask → `SceneFunc3dDataError` → `failure.json`; unliftable seed → lift raises → `failure.json`. Never an empty/low-quality mask (Task 3 + Task 5 tests). ✓
- Non-goals respected: no ranked-AP wiring, no training head, no adapter work, no `metrics.py` change, no SAM/Molmo protocol change. ✓

**2. Placeholder scan:** No "TBD"/"add validation"/"similar to Task N". Every code step ships full code; every command lists an expected result. The judgement calls (which legacy runner tests to delete; which `run_samples` worker tests to re-point) are pinned to exact names/line ranges + `rg` gates. ✓

**3. Type consistency:**
- `SeedConfirmation(seed_frame_id, seed_candidate_id, affordance_concept, approval_actions: tuple[ApprovalAction,...], uncertainties)` + `.seed_fragment_id` property; validated by `validate_seed_approval_actions` (Task 1) using `SEED_APPROVAL_ACTIONS = FRAGMENT_APPROVAL_ACTIONS[:5]` ending at `SAM_CANDIDATE_AGENT_APPROVED`. ✓
- `SceneFunc3dSeedOutcome(seed_frame_id, seed_candidate_id, seed_fragment_id, affordance_concept, approval_actions, uncertainties, seed_sam_mask_npz_path: Path)` — Task 1; built in Task 5 `parse_response`; consumed by `pipeline_result` (`.seed_frame_id/.seed_candidate_id/.seed_fragment_id/.affordance_concept/.uncertainties`) and `_lift_seed_fragment` (`.seed_sam_mask_npz_path`). No `seed_point_count` on the outcome — it lives on `SceneFunc3dPipelineOutcome.seed_point_count` from the runner lift. ✓
- `SeedSamCandidate(frame_id, candidate_id, mask_npz_path: Path)` + `recover_seed_sam_candidate(events_path, *, seed_frame_id, seed_candidate_id) -> SeedSamCandidate` — Task 2; used in Task 5 (`seed_candidate.mask_npz_path`). ✓
- `SceneFunc3dPipelineOutcome(seed, seed_point_count: int, mask_npz_path: Path, mask_ply_path: Path, result: AnchorMultiViewResult)` + `write_pipeline_fused_mask(fused_dir, *, point_indices, scene_vertices) -> (Path, Path)` + `write_pipeline_result(result_path, outcome, *, sample_id, turn_metadata) -> Path` — Task 3; called identically in Task 5. ✓
- Runner seams: `_run_pipeline_for_seed(*, sample, scene_root, output_dir, backend_config_path, seed_outcome, artifact_paths) -> SceneFunc3dPipelineOutcome`; `_lift_seed_fragment(*, scene, seed_outcome, out_dir) -> LiftMaskResult` (module globals → monkeypatchable). ✓
- Reused verified signatures: `lift_mask_to_3d(LiftMaskArgs(frame_id, candidate_id, mask_path, depth_path, intrinsics_path, pose_path), *, out_dir, raw_mesh_path) -> LiftMaskResult(...,lifted_point_count, mask_npz_path,...)`; `resolve_frame_geometry_assets(scene, frame_id).{depth_path,intrinsics_path,pose_path}`; `build_anchor(points_world, *, motion_type, seed_frame_id)`; `run_anchor_multiview_pipeline(*, anchor, scene, scene_vertices, proposer, params, frame_cap, depth_tolerance)`; `SidecarFrameProposer(scene, out_dir, backend_config_path, affordance_concept, task_description)`; `load_points_world_npz(path)`; `load_scene_mesh_vertices(path)`; `score_mask_npz(*, data_root, sample_id, mask_npz_path, failure_type)`. ✓
- `score_result_file` keeps its public signature; adds `_score_legacy_result_file` (renamed old body) + `score_pipeline_result_file` + `_result_kind` dispatch (Task 4); `_run_batch_sample` + `evaluation/__main__.py` unchanged. ✓

**Known follow-ups (not blockers):** (a) The pipeline re-runs Molmo/SAM/lift on the seed frame too. (b) `motion_type` uses the sample's first motion hint. (c) Legacy `result.json` scoring preserved via `result_kind` dispatch. (d) No `summary.json` for pipeline runs (the `result.json` + completion event are the durable record); add a minimal pipeline summary later if downstream tooling globs `summary.json`. (e) A seed with bad depth in its masked region fails closed in the runner rather than being agent-corrected (Decision 1 trade-off).

---

## Execution Handoff

Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (spec-compliance then code-quality) between tasks. REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
2. **Inline Execution** — execute here with checkpoints. REQUIRED SUB-SKILL: superpowers:executing-plans.

Tasks 1–4 are independent and land green on their own. **Task 5** (task + `run_single_sample` flip + test swap) is the largest and must land green before **Task 7** (dead-code removal).
