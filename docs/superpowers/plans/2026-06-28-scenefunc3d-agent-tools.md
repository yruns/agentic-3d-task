# SceneFunc3D Agent Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate SceneFunc3D Codex Agent workflow that generates auditable 3D mask artifacts through evidence viewing, Molmo point approval, SAM candidate approval, 3D lifting, optional multi-view expansion, and scoring hooks.

**Architecture:** Add a new `codex_agent.scenefunc3d` package parallel to `nr3d` and `openeqa`. Keep the task entrypoint, sample schema, tool dispatcher, state machine, artifact metadata, and evaluator separate from NR3D proposal-id semantics while reusing the repository's existing Codex Agent CLI-tool patterns. Heavy Molmo/SAM/image dependencies stay behind lazy-import tool modules, and all external JSON inputs are validated at CLI boundaries.

**Tech Stack:** Python 3.11+, dataclasses, Pydantic v2, pytest, ruff, black, mypy, existing `codex_agent` runtime/tool patterns, optional Pillow/torch/SAM/Molmo dependencies loaded only inside heavy tool functions.

---

## Pre-Execution Notes

- The repository may already contain unrelated uncommitted model-gateway changes. Do not stage or modify those files while executing this plan.
- If execution starts from a dirty checkout, create an isolated worktree or branch before editing code.
- Read `AGENTS.md` and `docs/python_code_agent_quality_guide.md` before coding.
- Use `rg` and existing `src/codex_agent/openeqa/` and `src/codex_agent/nr3d/` modules as local style references.
- Keep every commit scoped to the task named in that commit command.

## File Structure

Create these SceneFunc3D files:

- `src/codex_agent/scenefunc3d/__init__.py`: package marker and public exports.
- `src/codex_agent/scenefunc3d/sample.py`: visit/sample ids, JSON loaders, hidden-GT separation.
- `src/codex_agent/scenefunc3d/task.py`: approval state machine and transition validation.
- `src/codex_agent/scenefunc3d/playbook.py`: inline agent tool guidance and approval gates.
- `src/codex_agent/scenefunc3d/runner.py`: single-sample runner skeleton that assembles prompt/playbook and validates final JSON.
- `src/codex_agent/scenefunc3d/tools/__init__.py`: tool package marker.
- `src/codex_agent/scenefunc3d/tools/__main__.py`: CLI entrypoint for SceneFunc3D tools.
- `src/codex_agent/scenefunc3d/tools/dispatch.py`: tool name validation and lazy routing.
- `src/codex_agent/scenefunc3d/tools/models.py`: shared tool payload protocol and recoverable input error.
- `src/codex_agent/scenefunc3d/tools/scene_context.py`: prepared scene loader for raw frames and ConceptGraph indices.
- `src/codex_agent/scenefunc3d/tools/frame_views.py`: `scene_summary`, `view_frame`, `view_crop`, and lightweight frame rendering.
- `src/codex_agent/scenefunc3d/tools/keyframe_retrieval.py`: typed `keyframe_selector` wrapper.
- `src/codex_agent/scenefunc3d/tools/molmo_pointing.py`: Molmo raw-output parsing plus configured backend interface.
- `src/codex_agent/scenefunc3d/tools/sam_masking.py`: SAM candidate metadata models and candidate contact-sheet contract.
- `src/codex_agent/scenefunc3d/tools/mask_lifting.py`: 2D mask to 3D lifting contract and artifact paths.
- `src/codex_agent/scenefunc3d/tools/mask_artifacts.py`: artifact layout, metadata models, event log writer.
- `src/codex_agent/scenefunc3d/tools/mask_inspection.py`: artifact summary for agent review.
- `src/codex_agent/scenefunc3d/evaluation/__init__.py`: evaluator package marker.
- `src/codex_agent/scenefunc3d/evaluation/metrics.py`: IoU/precision/recall/F1 math.
- `src/codex_agent/scenefunc3d/evaluation/scorer.py`: load predicted mask ids and hidden GT ids, compute metrics.

Modify this shared file:

- `src/codex_agent/errors.py`: add `SceneFunc3dDataError`.

Add tests:

- `src/codex_agent/tests/test_scenefunc3d_sample.py`
- `src/codex_agent/tests/test_scenefunc3d_state.py`
- `src/codex_agent/tests/test_scenefunc3d_artifacts.py`
- `src/codex_agent/tests/test_scenefunc3d_tools_cli.py`
- `src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py`
- `src/codex_agent/tests/test_scenefunc3d_metrics.py`
- `src/codex_agent/tests/test_scenefunc3d_playbook.py`

## Task 1: Sample Loader And SceneFunc Error Type

**Files:**
- Modify: `src/codex_agent/errors.py`
- Create: `src/codex_agent/scenefunc3d/__init__.py`
- Create: `src/codex_agent/scenefunc3d/sample.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_sample.py`

- [ ] **Step 1: Write failing tests for sample id parsing and hidden GT separation**

Create `src/codex_agent/tests/test_scenefunc3d_sample.py`:

```python
"""Tests for SceneFunc3D sample loading and hidden-GT separation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.sample import (
    SceneFunc3dSampleId,
    load_sample,
    safe_sample_id,
    scene_dir_for,
)


def _write_scene(root: Path) -> None:
    scene_dir = root / "421254"
    scene_dir.mkdir(parents=True)
    (scene_dir / "conceptgraph").mkdir()
    (scene_dir / "421254_descriptions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "descriptions": [
                    {
                        "desc_id": "desc-a",
                        "annot_id": ["annot-a"],
                        "description": "Open the lower drawer.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_motions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "motions": [
                    {
                        "motion_id": "motion-a",
                        "annot_id": "annot-a",
                        "motion_type": "trans",
                        "motion_dir": [1.0, 0.0, 0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_annotations.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "annotations": [
                    {
                        "annot_id": "annot-a",
                        "label": "pinch_pull",
                        "indices": [3, 5, 8],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_sample_id_parse() -> None:
    parsed = SceneFunc3dSampleId.parse("421254::desc-a")
    assert parsed.visit_id == "421254"
    assert parsed.desc_id == "desc-a"
    assert parsed.raw == "421254::desc-a"


def test_sample_id_rejects_bad_form() -> None:
    with pytest.raises(SceneFunc3dDataError, match="must have 2"):
        SceneFunc3dSampleId.parse("421254")


def test_scene_dir_for() -> None:
    assert scene_dir_for(Path("/data"), "421254") == Path("/data/421254")


def test_safe_sample_id() -> None:
    assert safe_sample_id("421254::desc-a") == "421254__desc-a"


def test_load_sample_hides_gt_indices(tmp_path: Path) -> None:
    _write_scene(tmp_path)
    sample = load_sample(tmp_path, "421254::desc-a")

    assert sample.sample_id == "421254::desc-a"
    assert sample.visit_id == "421254"
    assert sample.desc_id == "desc-a"
    assert sample.task_description == "Open the lower drawer."
    assert sample.annotation_ids == ("annot-a",)
    assert sample.motion_hints[0].motion_type == "trans"
    assert sample.agent_context == {
        "visit_id": "421254",
        "desc_id": "desc-a",
        "task_description": "Open the lower drawer.",
        "annotation_ids": ["annot-a"],
        "motion_hints": [
            {
                "motion_id": "motion-a",
                "annotation_id": "annot-a",
                "motion_type": "trans",
                "motion_dir": [1.0, 0.0, 0.0],
            }
        ],
    }
    assert "indices" not in json.dumps(sample.agent_context)


def test_load_sample_requires_matching_visit(tmp_path: Path) -> None:
    _write_scene(tmp_path)
    path = tmp_path / "421254" / "421254_descriptions.json"
    path.write_text(json.dumps({"visit_id": "wrong", "descriptions": []}), encoding="utf-8")

    with pytest.raises(SceneFunc3dDataError, match="visit_id"):
        load_sample(tmp_path, "421254::desc-a")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_sample.py -q
```

Expected: fail with `ImportError` for `SceneFunc3dDataError` or `codex_agent.scenefunc3d`.

- [ ] **Step 3: Add `SceneFunc3dDataError`**

Modify `src/codex_agent/errors.py`:

```python
class SceneFunc3dDataError(CodexAgentError):
    """Raised when SceneFunc3D scene/sample assets are missing or malformed."""
```

Add `"SceneFunc3dDataError"` to `__all__`.

- [ ] **Step 4: Add package marker**

Create `src/codex_agent/scenefunc3d/__init__.py`:

```python
"""SceneFunc3D agent task package."""

from __future__ import annotations

__all__: list[str] = []
```

- [ ] **Step 5: Implement sample loader**

Create `src/codex_agent/scenefunc3d/sample.py`:

```python
"""Load SceneFunc3D samples from prepared SceneFuncVal-CG scenes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import SceneFunc3dDataError


@dataclass(frozen=True)
class SceneFunc3dSampleId:
    """A parsed SceneFunc3D ``<visit_id>::<desc_id>`` identifier."""

    raw: str
    visit_id: str
    desc_id: str

    @classmethod
    def parse(cls, sample_id: str) -> SceneFunc3dSampleId:
        if not isinstance(sample_id, str):
            raise SceneFunc3dDataError(
                f"sample_id must be a string, got {sample_id!r}"
            )
        parts = sample_id.split("::")
        if len(parts) != 2:
            raise SceneFunc3dDataError(
                f"sample_id must have 2 '::'-separated parts, got {sample_id!r}"
            )
        visit_id, desc_id = parts
        if not visit_id or not desc_id:
            raise SceneFunc3dDataError(f"invalid SceneFunc3D sample_id: {sample_id!r}")
        return cls(raw=sample_id, visit_id=visit_id, desc_id=desc_id)


@dataclass(frozen=True)
class SceneFuncMotionHint:
    """Agent-visible motion metadata linked to one annotation id."""

    motion_id: str
    annotation_id: str
    motion_type: str
    motion_dir: tuple[float, ...]

    def to_agent_payload(self) -> dict[str, object]:
        return {
            "motion_id": self.motion_id,
            "annotation_id": self.annotation_id,
            "motion_type": self.motion_type,
            "motion_dir": list(self.motion_dir),
        }


@dataclass(frozen=True)
class SceneFunc3dSample:
    """One SceneFunc3D language-conditioned mask-generation sample."""

    sample_id: str
    visit_id: str
    desc_id: str
    task_description: str
    annotation_ids: tuple[str, ...]
    motion_hints: tuple[SceneFuncMotionHint, ...]

    @property
    def agent_context(self) -> dict[str, object]:
        return {
            "visit_id": self.visit_id,
            "desc_id": self.desc_id,
            "task_description": self.task_description,
            "annotation_ids": list(self.annotation_ids),
            "motion_hints": [hint.to_agent_payload() for hint in self.motion_hints],
        }


def scene_dir_for(data_root: Path, visit_id: str) -> Path:
    """Return the canonical ``<data_root>/<visit_id>`` scene directory."""
    return data_root / visit_id


def safe_sample_id(sample_id: str) -> str:
    """Return a filesystem-safe sample id."""
    return sample_id.replace("::", "__").replace("/", "__")


def load_sample(data_root: Path, sample_id: str) -> SceneFunc3dSample:
    parsed = SceneFunc3dSampleId.parse(sample_id)
    scene_dir = scene_dir_for(data_root, parsed.visit_id)
    descriptions = _load_scene_object(scene_dir / f"{parsed.visit_id}_descriptions.json")
    motions = _load_scene_object(scene_dir / f"{parsed.visit_id}_motions.json")
    annotations = _load_scene_object(scene_dir / f"{parsed.visit_id}_annotations.json")
    _validate_visit_id(descriptions, parsed.visit_id, "descriptions")
    _validate_visit_id(motions, parsed.visit_id, "motions")
    _validate_visit_id(annotations, parsed.visit_id, "annotations")

    description = _find_description(descriptions, parsed.desc_id)
    annotation_ids = _coerce_string_tuple(description.get("annot_id"), "annot_id")
    return SceneFunc3dSample(
        sample_id=sample_id,
        visit_id=parsed.visit_id,
        desc_id=parsed.desc_id,
        task_description=_coerce_non_empty_string(
            description.get("description"), "description"
        ),
        annotation_ids=annotation_ids,
        motion_hints=_motion_hints_for(motions, annotation_ids),
    )


def _load_scene_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SceneFunc3dDataError(f"SceneFunc3D JSON is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SceneFunc3dDataError(f"{path}: expected a JSON object")
    return payload


def _validate_visit_id(payload: dict[str, Any], visit_id: str, name: str) -> None:
    actual = payload.get("visit_id")
    if actual != visit_id:
        raise SceneFunc3dDataError(f"{name} visit_id={actual!r}, expected {visit_id!r}")


def _find_description(payload: dict[str, Any], desc_id: str) -> dict[str, Any]:
    descriptions = payload.get("descriptions")
    if not isinstance(descriptions, list):
        raise SceneFunc3dDataError("descriptions must be a list")
    for item in descriptions:
        if isinstance(item, dict) and item.get("desc_id") == desc_id:
            return item
    raise SceneFunc3dDataError(f"description id not found: {desc_id}")


def _motion_hints_for(
    payload: dict[str, Any], annotation_ids: tuple[str, ...]
) -> tuple[SceneFuncMotionHint, ...]:
    motions = payload.get("motions")
    if not isinstance(motions, list):
        raise SceneFunc3dDataError("motions must be a list")
    annotation_set = set(annotation_ids)
    hints: list[SceneFuncMotionHint] = []
    for item in motions:
        if not isinstance(item, dict):
            continue
        annotation_id = item.get("annot_id")
        if not isinstance(annotation_id, str) or annotation_id not in annotation_set:
            continue
        hints.append(
            SceneFuncMotionHint(
                motion_id=_coerce_non_empty_string(item.get("motion_id"), "motion_id"),
                annotation_id=annotation_id,
                motion_type=_coerce_non_empty_string(
                    item.get("motion_type"), "motion_type"
                ),
                motion_dir=_coerce_float_tuple(item.get("motion_dir"), "motion_dir"),
            )
        )
    return tuple(hints)


def _coerce_string_tuple(raw: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,)
    if not isinstance(raw, list):
        raise SceneFunc3dDataError(f"{field_name} must be a string or list")
    values = tuple(item for item in raw if isinstance(item, str) and item)
    if len(values) != len(raw):
        raise SceneFunc3dDataError(f"{field_name} contains non-string values")
    if not values:
        raise SceneFunc3dDataError(f"{field_name} must not be empty")
    return values


def _coerce_non_empty_string(raw: Any, field_name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise SceneFunc3dDataError(f"{field_name} must be a non-empty string")
    return raw


def _coerce_float_tuple(raw: Any, field_name: str) -> tuple[float, ...]:
    if not isinstance(raw, list):
        raise SceneFunc3dDataError(f"{field_name} must be a list")
    return tuple(float(value) for value in raw)
```

- [ ] **Step 6: Run tests**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_sample.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/codex_agent/errors.py src/codex_agent/scenefunc3d/__init__.py src/codex_agent/scenefunc3d/sample.py src/codex_agent/tests/test_scenefunc3d_sample.py
git commit -m "feat: add scenefunc3d sample loader"
```

## Task 2: Artifact Models And Approval State Machine

**Files:**
- Create: `src/codex_agent/scenefunc3d/task.py`
- Create: `src/codex_agent/scenefunc3d/tools/__init__.py`
- Create: `src/codex_agent/scenefunc3d/tools/mask_artifacts.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_state.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_artifacts.py`

- [ ] **Step 1: Write failing state-machine tests**

Create `src/codex_agent/tests/test_scenefunc3d_state.py`:

```python
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
```

- [ ] **Step 2: Write failing artifact metadata tests**

Create `src/codex_agent/tests/test_scenefunc3d_artifacts.py`:

```python
"""Tests for SceneFunc3D artifact metadata."""

from __future__ import annotations

import json
from pathlib import Path

from codex_agent.scenefunc3d.tools.mask_artifacts import (
    ArtifactStatus,
    MaskArtifactPaths,
    SceneFunc3dRunSummary,
    artifact_paths_for,
    write_run_summary,
)


def test_artifact_paths_for(tmp_path: Path) -> None:
    paths = artifact_paths_for(tmp_path, "421254", "desc-a")
    assert paths.root == tmp_path / "421254" / "desc-a"
    assert paths.summary_json == paths.root / "summary.json"
    assert paths.events_jsonl == paths.root / "events.jsonl"
    assert paths.overlays_dir == paths.root / "overlays"
    assert paths.fragments_dir == paths.root / "fragments"
    assert paths.fused_dir == paths.root / "fused"


def test_write_run_summary(tmp_path: Path) -> None:
    paths = artifact_paths_for(tmp_path, "421254", "desc-a")
    summary = SceneFunc3dRunSummary(
        sample_id="421254::desc-a",
        visit_id="421254",
        desc_id="desc-a",
        task_description="Open the drawer.",
        status=ArtifactStatus.IN_PROGRESS,
        selected_frame_ids=("000050",),
        accepted_fragment_ids=(),
        failure_type="",
        stop_reason="",
    )

    write_run_summary(paths, summary)

    payload = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert payload["sample_id"] == "421254::desc-a"
    assert payload["selected_frame_ids"] == ["000050"]
    assert payload["status"] == "in_progress"
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_state.py src/codex_agent/tests/test_scenefunc3d_artifacts.py -q
```

Expected: fail with import errors for `scenefunc3d.task` and `mask_artifacts`.

- [ ] **Step 4: Implement state machine**

Create `src/codex_agent/scenefunc3d/task.py`:

```python
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


_TRANSITIONS: dict[
    tuple[SceneFunc3dStage, ApprovalAction], SceneFunc3dStage
] = {
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
        return cls(sample_id=sample_id, stage=SceneFunc3dStage.INITIAL)

    def transition(self, action: ApprovalAction) -> SceneFunc3dState:
        next_stage = _TRANSITIONS.get((self.stage, action))
        if next_stage is None:
            raise SceneFunc3dDataError(
                f"invalid transition from {self.stage.value} via {action.value}"
            )
        return SceneFunc3dState(sample_id=self.sample_id, stage=next_stage)
```

- [ ] **Step 5: Add tools package marker**

Create `src/codex_agent/scenefunc3d/tools/__init__.py`:

```python
"""SceneFunc3D agent tool package."""

from __future__ import annotations

__all__: list[str] = []
```

- [ ] **Step 6: Implement artifact metadata**

Create `src/codex_agent/scenefunc3d/tools/mask_artifacts.py`:

```python
"""Artifact layout and metadata for SceneFunc3D mask-generation runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class ArtifactStatus(str, Enum):
    """Status stored in SceneFunc3D run summaries."""

    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class MaskArtifactPaths:
    """Filesystem paths for one SceneFunc3D sample artifact."""

    root: Path
    summary_json: Path
    events_jsonl: Path
    raw_outputs_dir: Path
    overlays_dir: Path
    fragments_dir: Path
    fused_dir: Path

    def create_dirs(self) -> None:
        self.raw_outputs_dir.mkdir(parents=True, exist_ok=True)
        self.overlays_dir.mkdir(parents=True, exist_ok=True)
        self.fragments_dir.mkdir(parents=True, exist_ok=True)
        self.fused_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SceneFunc3dRunSummary:
    """JSON-serializable run summary for one SceneFunc3D sample."""

    sample_id: str
    visit_id: str
    desc_id: str
    task_description: str
    status: ArtifactStatus
    selected_frame_ids: tuple[str, ...]
    accepted_fragment_ids: tuple[str, ...]
    failure_type: str
    stop_reason: str

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["selected_frame_ids"] = list(self.selected_frame_ids)
        payload["accepted_fragment_ids"] = list(self.accepted_fragment_ids)
        return payload


def artifact_paths_for(output_dir: Path, visit_id: str, desc_id: str) -> MaskArtifactPaths:
    root = output_dir / visit_id / desc_id
    return MaskArtifactPaths(
        root=root,
        summary_json=root / "summary.json",
        events_jsonl=root / "events.jsonl",
        raw_outputs_dir=root / "raw_outputs",
        overlays_dir=root / "overlays",
        fragments_dir=root / "fragments",
        fused_dir=root / "fused",
    )


def write_run_summary(paths: MaskArtifactPaths, summary: SceneFunc3dRunSummary) -> Path:
    paths.create_dirs()
    paths.summary_json.write_text(
        json.dumps(summary.to_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths.summary_json
```

- [ ] **Step 7: Run tests**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_state.py src/codex_agent/tests/test_scenefunc3d_artifacts.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/codex_agent/scenefunc3d/task.py src/codex_agent/scenefunc3d/tools/__init__.py src/codex_agent/scenefunc3d/tools/mask_artifacts.py src/codex_agent/tests/test_scenefunc3d_state.py src/codex_agent/tests/test_scenefunc3d_artifacts.py
git commit -m "feat: add scenefunc3d approval state and artifacts"
```

## Task 3: Tool CLI Skeleton And Scene Summary

**Files:**
- Create: `src/codex_agent/scenefunc3d/tools/models.py`
- Create: `src/codex_agent/scenefunc3d/tools/scene_context.py`
- Create: `src/codex_agent/scenefunc3d/tools/frame_views.py`
- Create: `src/codex_agent/scenefunc3d/tools/dispatch.py`
- Create: `src/codex_agent/scenefunc3d/tools/__main__.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_tools_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `src/codex_agent/tests/test_scenefunc3d_tools_cli.py`:

```python
"""Tests for the SceneFunc3D tools CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_agent.scenefunc3d.tools.__main__ import main


def _write_scene(root: Path) -> Path:
    scene_dir = root / "421254"
    raw_dir = scene_dir / "raw"
    conceptgraph_dir = scene_dir / "conceptgraph"
    raw_dir.mkdir(parents=True)
    conceptgraph_dir.mkdir()
    (raw_dir / "000000-rgb.png").write_bytes(b"not-a-real-image")
    (raw_dir / "000010-rgb.png").write_bytes(b"not-a-real-image")
    (conceptgraph_dir / "indices").mkdir()
    return scene_dir


def test_cli_scene_summary_prints_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scene_dir = _write_scene(tmp_path)
    code = main(["scene_summary", "--scene-root", str(scene_dir), "--args", "{}"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["visit_id"] == "421254"
    assert payload["total_rgb_frames"] == 2
    assert payload["rgb_frame_ids"] == ["000000", "000010"]
    assert payload["has_conceptgraph"] is True


def test_cli_bad_args_json_is_recoverable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_scene(tmp_path)
    code = main(["scene_summary", "--scene-root", str(scene_dir), "--args", "not-json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "must be valid JSON" in payload["error"]


def test_cli_missing_scene_root_exits_nonzero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["scene_summary", "--scene-root", str(tmp_path / "missing"), "--args", "{}"])
    assert excinfo.value.code == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_tools_cli.py -q
```

Expected: fail with import error for `codex_agent.scenefunc3d.tools.__main__`.

- [ ] **Step 3: Implement shared tool models**

Create `src/codex_agent/scenefunc3d/tools/models.py`:

```python
"""Shared types for the SceneFunc3D agent tool layer."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ...errors import CodexAgentError


class ToolInputError(CodexAgentError):
    """Raised when SceneFunc3D tool arguments are invalid but recoverable."""


@runtime_checkable
class ToolPayload(Protocol):
    """A tool result that can render itself as a JSON-serializable mapping."""

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-ready dict at the CLI boundary."""
        ...
```

- [ ] **Step 4: Implement scene context**

Create `src/codex_agent/scenefunc3d/tools/scene_context.py`:

```python
"""Prepared SceneFunc3D scene context for CLI tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ...errors import SceneFunc3dDataError

_RAW_DIRNAME = "raw"
_CONCEPTGRAPH_DIRNAME = "conceptgraph"
_RGB_FRAME_RE = re.compile(r"^(\d{6})-rgb\.(png|jpg|jpeg)$")


@dataclass(frozen=True)
class SceneFunc3dToolScene:
    """Filesystem context for one prepared SceneFuncVal-CG scene."""

    visit_id: str
    scene_root: Path
    rgb_frame_ids: tuple[str, ...]

    @property
    def raw_dir(self) -> Path:
        return self.scene_root / _RAW_DIRNAME

    @property
    def conceptgraph_dir(self) -> Path:
        return self.scene_root / _CONCEPTGRAPH_DIRNAME

    @classmethod
    def load(cls, scene_root: Path) -> SceneFunc3dToolScene:
        if not scene_root.is_dir():
            raise SceneFunc3dDataError(f"SceneFunc3D scene root is missing: {scene_root}")
        raw_dir = scene_root / _RAW_DIRNAME
        if not raw_dir.is_dir():
            raise SceneFunc3dDataError(f"SceneFunc3D raw frame directory is missing: {raw_dir}")
        frame_ids = sorted(
            match.group(1)
            for match in (_RGB_FRAME_RE.match(path.name) for path in raw_dir.iterdir())
            if match is not None
        )
        if not frame_ids:
            raise SceneFunc3dDataError(f"no RGB frames found under {raw_dir}")
        return cls(
            visit_id=scene_root.name,
            scene_root=scene_root,
            rgb_frame_ids=tuple(frame_ids),
        )
```

- [ ] **Step 5: Implement `scene_summary`**

Create `src/codex_agent/scenefunc3d/tools/frame_views.py`:

```python
"""Lightweight SceneFunc3D evidence-view tools."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from .scene_context import SceneFunc3dToolScene


class SceneSummaryArgs(BaseModel):
    """Arguments for ``scene_summary``."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class SceneSummaryResult:
    """Summary payload for one prepared SceneFunc3D scene."""

    visit_id: str
    total_rgb_frames: int
    rgb_frame_ids: tuple[str, ...]
    has_conceptgraph: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "visit_id": self.visit_id,
            "total_rgb_frames": self.total_rgb_frames,
            "rgb_frame_ids": list(self.rgb_frame_ids),
            "has_conceptgraph": self.has_conceptgraph,
        }


def scene_summary(
    tool_scene: SceneFunc3dToolScene, args: SceneSummaryArgs
) -> SceneSummaryResult:
    _ = args
    return SceneSummaryResult(
        visit_id=tool_scene.visit_id,
        total_rgb_frames=len(tool_scene.rgb_frame_ids),
        rgb_frame_ids=tool_scene.rgb_frame_ids,
        has_conceptgraph=tool_scene.conceptgraph_dir.is_dir(),
    )
```

- [ ] **Step 6: Implement dispatcher**

Create `src/codex_agent/scenefunc3d/tools/dispatch.py`:

```python
"""SceneFunc3D tool dispatcher with typed argument validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .models import ToolInputError, ToolPayload
from .scene_context import SceneFunc3dToolScene

TOOL_NAMES: tuple[str, ...] = ("scene_summary",)

_ArgsT = TypeVar("_ArgsT", bound=BaseModel)


def run_tool(
    tool_scene: SceneFunc3dToolScene,
    name: str,
    raw_args: Mapping[str, Any],
    *,
    out_dir: Path,
) -> ToolPayload:
    _ = out_dir
    if name == "scene_summary":
        from .frame_views import SceneSummaryArgs, scene_summary

        return scene_summary(tool_scene, _parse(SceneSummaryArgs, raw_args))
    raise ToolInputError(f"unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}")


def _parse(model: type[_ArgsT], raw_args: Mapping[str, Any]) -> _ArgsT:
    try:
        return model.model_validate(dict(raw_args))
    except ValidationError as exc:
        raise ToolInputError(
            f"invalid arguments for {model.__name__}: {_format_validation_error(exc)}"
        ) from exc


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ())) or "(root)"
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return "; ".join(parts)
```

- [ ] **Step 7: Implement CLI entrypoint**

Create `src/codex_agent/scenefunc3d/tools/__main__.py`:

```python
"""CLI entry point for SceneFunc3D agent tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...errors import CodexAgentError
from .dispatch import TOOL_NAMES, run_tool
from .models import ToolInputError
from .scene_context import SceneFunc3dToolScene

_DEFAULT_OUT_DIR = Path("tmp") / "scenefunc3d_tool_scratch"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codex_agent.scenefunc3d.tools", description=__doc__
    )
    parser.add_argument("tool", choices=list(TOOL_NAMES), help="Tool to run.")
    parser.add_argument("--scene-root", required=True, type=Path, help="Scene root.")
    parser.add_argument(
        "--args",
        default="{}",
        help="Tool arguments as a single JSON object.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Writable scratch directory for rendered images.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    out_dir = args.out_dir if args.out_dir is not None else _DEFAULT_OUT_DIR
    try:
        raw_args = _parse_args_json(args.args)
        tool_scene = SceneFunc3dToolScene.load(args.scene_root)
        payload = run_tool(tool_scene, args.tool, raw_args, out_dir=out_dir)
    except ToolInputError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 0
    except CodexAgentError as exc:
        parser.exit(status=1, message=f"ERROR: {exc}\n")
    print(json.dumps(payload.to_payload(), ensure_ascii=False))
    return 0


def _parse_args_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolInputError(f"--args must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ToolInputError("--args must be a JSON object")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Run tests**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_tools_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/codex_agent/scenefunc3d/tools/models.py src/codex_agent/scenefunc3d/tools/scene_context.py src/codex_agent/scenefunc3d/tools/frame_views.py src/codex_agent/scenefunc3d/tools/dispatch.py src/codex_agent/scenefunc3d/tools/__main__.py src/codex_agent/tests/test_scenefunc3d_tools_cli.py
git commit -m "feat: add scenefunc3d tool cli skeleton"
```

## Task 4: Evidence Image Tools

**Files:**
- Modify: `src/codex_agent/scenefunc3d/tools/frame_views.py`
- Create: `src/codex_agent/scenefunc3d/tools/keyframe_retrieval.py`
- Modify: `src/codex_agent/scenefunc3d/tools/dispatch.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_tools_cli.py`

- [ ] **Step 1: Add failing tests for `view_frame` and `keyframe_selector`**

Append to `src/codex_agent/tests/test_scenefunc3d_tools_cli.py`:

```python
def test_cli_view_frame_returns_image_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    scene_dir = _write_scene(tmp_path)
    Image.new("RGB", (12, 10), color=(10, 20, 30)).save(scene_dir / "raw" / "000000-rgb.png")
    code = main(
        [
            "view_frame",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_ids": ["000000"]}),
            "--out-dir",
            str(tmp_path / "scratch"),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["frames"][0]["frame_id"] == "000000"
    assert Path(payload["frames"][0]["image_path"]).exists()


def test_cli_keyframe_selector_returns_first_k_frames(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_scene(tmp_path)
    code = main(
        [
            "keyframe_selector",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"query": "drawer handle", "k": 1}),
            "--out-dir",
            str(tmp_path / "scratch"),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["query"] == "drawer handle"
    assert payload["frames"] == [{"frame_id": "000000", "rank": 1}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_tools_cli.py -q
```

Expected: fail because `view_frame` and `keyframe_selector` are not registered.

- [ ] **Step 3: Add `view_frame` implementation**

Extend `src/codex_agent/scenefunc3d/tools/frame_views.py` with:

```python
from pathlib import Path

from .models import ToolInputError


class ViewFrameArgs(BaseModel):
    """Arguments for ``view_frame``."""

    model_config = ConfigDict(extra="forbid")

    frame_ids: tuple[str, ...]


@dataclass(frozen=True)
class FrameViewPayload:
    """One rendered frame returned to the agent."""

    frame_id: str
    image_path: Path

    def to_payload(self) -> dict[str, object]:
        return {"frame_id": self.frame_id, "image_path": str(self.image_path)}


@dataclass(frozen=True)
class ViewFrameResult:
    """Rendered frame result."""

    frames: tuple[FrameViewPayload, ...]

    def to_payload(self) -> dict[str, object]:
        return {"frames": [frame.to_payload() for frame in self.frames]}


def view_frame(
    tool_scene: SceneFunc3dToolScene, args: ViewFrameArgs, *, out_dir: Path
) -> ViewFrameResult:
    rendered: list[FrameViewPayload] = []
    for frame_id in args.frame_ids:
        if frame_id not in tool_scene.rgb_frame_ids:
            raise ToolInputError(f"unknown frame_id {frame_id!r}")
        source = tool_scene.raw_dir / f"{frame_id}-rgb.png"
        destination = out_dir / tool_scene.visit_id / f"{frame_id}.jpg"
        rendered.append(
            FrameViewPayload(
                frame_id=frame_id,
                image_path=_downsize_rgb_for_view(source, destination),
            )
        )
    return ViewFrameResult(frames=tuple(rendered))


def _downsize_rgb_for_view(source: Path, destination: Path) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ToolInputError("Pillow is required for view_frame") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, format="JPEG", quality=85)
    return destination
```

- [ ] **Step 4: Add simple keyframe selector**

Create `src/codex_agent/scenefunc3d/tools/keyframe_retrieval.py`:

```python
"""SceneFunc3D keyframe retrieval tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .scene_context import SceneFunc3dToolScene


class KeyframeSelectorArgs(BaseModel):
    """Arguments for ``keyframe_selector``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    k: int = Field(default=4, ge=1, le=8)


@dataclass(frozen=True)
class KeyframePayload:
    """One selected frame."""

    frame_id: str
    rank: int

    def to_payload(self) -> dict[str, object]:
        return {"frame_id": self.frame_id, "rank": self.rank}


@dataclass(frozen=True)
class KeyframeSelectorResult:
    """Keyframe selector result."""

    query: str
    frames: tuple[KeyframePayload, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "query": self.query,
            "frames": [frame.to_payload() for frame in self.frames],
        }


def keyframe_selector(
    tool_scene: SceneFunc3dToolScene, args: KeyframeSelectorArgs, *, out_dir: Path
) -> KeyframeSelectorResult:
    _ = out_dir
    selected = tool_scene.rgb_frame_ids[: args.k]
    return KeyframeSelectorResult(
        query=args.query,
        frames=tuple(
            KeyframePayload(frame_id=frame_id, rank=index + 1)
            for index, frame_id in enumerate(selected)
        ),
    )
```

- [ ] **Step 5: Register tools in dispatcher**

Modify `src/codex_agent/scenefunc3d/tools/dispatch.py`:

```python
TOOL_NAMES: tuple[str, ...] = ("scene_summary", "view_frame", "keyframe_selector")
```

Add branches inside `run_tool`:

```python
    if name == "view_frame":
        from .frame_views import ViewFrameArgs, view_frame

        return view_frame(tool_scene, _parse(ViewFrameArgs, raw_args), out_dir=out_dir)
    if name == "keyframe_selector":
        from .keyframe_retrieval import KeyframeSelectorArgs, keyframe_selector

        return keyframe_selector(
            tool_scene, _parse(KeyframeSelectorArgs, raw_args), out_dir=out_dir
        )
```

- [ ] **Step 6: Run tests**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_tools_cli.py -q
```

Expected: all tests pass. If Pillow is not installed, the `view_frame` test is skipped by `pytest.importorskip("PIL")`.

- [ ] **Step 7: Commit**

```bash
git add src/codex_agent/scenefunc3d/tools/frame_views.py src/codex_agent/scenefunc3d/tools/keyframe_retrieval.py src/codex_agent/scenefunc3d/tools/dispatch.py src/codex_agent/tests/test_scenefunc3d_tools_cli.py
git commit -m "feat: add scenefunc3d evidence tools"
```

## Task 5: Molmo And SAM Contract Models

**Files:**
- Create: `src/codex_agent/scenefunc3d/tools/molmo_pointing.py`
- Create: `src/codex_agent/scenefunc3d/tools/sam_masking.py`
- Modify: `src/codex_agent/scenefunc3d/tools/dispatch.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py`

- [ ] **Step 1: Write failing tests for Molmo point parsing and SAM candidate serialization**

Create `src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py`:

```python
"""Tests for SceneFunc3D Molmo and SAM tool contracts."""

from __future__ import annotations

from pathlib import Path

from codex_agent.scenefunc3d.tools.molmo_pointing import parse_molmo_points
from codex_agent.scenefunc3d.tools.sam_masking import SamCandidate, SamMaskResult


def test_parse_molmo_percent_point() -> None:
    points = parse_molmo_points(
        '<point x="81.0" y="61.9" alt="drawer knob">drawer knob</point>',
        image_width=1440,
        image_height=1920,
    )
    assert len(points) == 1
    assert points[0].x_px == 1166.4
    assert points[0].y_px == 1188.48
    assert points[0].label == "drawer knob"


def test_sam_result_payload() -> None:
    result = SamMaskResult(
        frame_id="000050",
        candidates=(
            SamCandidate(
                candidate_id="mask_00",
                score=0.82,
                pixel_count=1119,
                coverage_percent=0.0405,
                overlay_path=Path("/tmp/mask_00.jpg"),
            ),
        ),
        contact_sheet_path=Path("/tmp/sam_candidates.jpg"),
    )
    assert result.to_payload()["candidates"][0]["candidate_id"] == "mask_00"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py -q
```

Expected: fail with import errors.

- [ ] **Step 3: Implement Molmo parsing contract**

Create `src/codex_agent/scenefunc3d/tools/molmo_pointing.py`:

```python
"""Molmo point parsing and tool contracts for SceneFunc3D."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_POINT_RE = re.compile(
    r'<point\s+x="(?P<x>[0-9.]+)"\s+y="(?P<y>[0-9.]+)"(?:\s+alt="(?P<alt>[^"]*)")?'
)


@dataclass(frozen=True)
class MolmoPoint:
    """One parsed Molmo point in pixel coordinates."""

    x_px: float
    y_px: float
    source: str
    label: str

    def to_payload(self) -> dict[str, object]:
        return {
            "x_px": self.x_px,
            "y_px": self.y_px,
            "source": self.source,
            "label": self.label,
        }


class MolmoPointArgs(BaseModel):
    """Arguments for ``molmo_point``."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(min_length=1)
    image_path: Path
    prompt: str = Field(min_length=1)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)


@dataclass(frozen=True)
class MolmoPointResult:
    """Molmo point tool result."""

    frame_id: str
    prompt: str
    points: tuple[MolmoPoint, ...]
    raw_text_path: Path
    overlay_path: Path

    def to_payload(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "prompt": self.prompt,
            "points": [point.to_payload() for point in self.points],
            "raw_text_path": str(self.raw_text_path),
            "overlay_path": str(self.overlay_path),
        }


def parse_molmo_points(
    raw_text: str, *, image_width: int, image_height: int
) -> tuple[MolmoPoint, ...]:
    points: list[MolmoPoint] = []
    for match in _POINT_RE.finditer(raw_text):
        x_percent = float(match.group("x"))
        y_percent = float(match.group("y"))
        label = match.group("alt") or ""
        points.append(
            MolmoPoint(
                x_px=image_width * x_percent / 100.0,
                y_px=image_height * y_percent / 100.0,
                source="molmo_percent",
                label=label,
            )
        )
    return tuple(points)
```

- [ ] **Step 4: Implement SAM candidate contract**

Create `src/codex_agent/scenefunc3d/tools/sam_masking.py`:

```python
"""SAM candidate mask contracts for SceneFunc3D."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SamMaskArgs(BaseModel):
    """Arguments for ``sam_mask``."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(min_length=1)
    image_path: Path
    points: tuple[tuple[float, float], ...] = Field(min_length=1)


@dataclass(frozen=True)
class SamCandidate:
    """One SAM candidate mask."""

    candidate_id: str
    score: float
    pixel_count: int
    coverage_percent: float
    overlay_path: Path

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "pixel_count": self.pixel_count,
            "coverage_percent": self.coverage_percent,
            "overlay_path": str(self.overlay_path),
        }


@dataclass(frozen=True)
class SamMaskResult:
    """SAM mask tool result."""

    frame_id: str
    candidates: tuple[SamCandidate, ...]
    contact_sheet_path: Path

    def to_payload(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "candidates": [candidate.to_payload() for candidate in self.candidates],
            "contact_sheet_path": str(self.contact_sheet_path),
        }
```

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/codex_agent/scenefunc3d/tools/molmo_pointing.py src/codex_agent/scenefunc3d/tools/sam_masking.py src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py
git commit -m "feat: add scenefunc3d molmo sam contracts"
```

## Task 6: Mask Lifting Contract And Metrics

**Files:**
- Create: `src/codex_agent/scenefunc3d/tools/mask_lifting.py`
- Create: `src/codex_agent/scenefunc3d/evaluation/__init__.py`
- Create: `src/codex_agent/scenefunc3d/evaluation/metrics.py`
- Create: `src/codex_agent/scenefunc3d/evaluation/scorer.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Create `src/codex_agent/tests/test_scenefunc3d_metrics.py`:

```python
"""Tests for SceneFunc3D mask metrics."""

from __future__ import annotations

from codex_agent.scenefunc3d.evaluation.metrics import MaskMetrics, compute_mask_metrics


def test_compute_mask_metrics() -> None:
    metrics = compute_mask_metrics(predicted_ids={1, 2, 3}, gt_ids={2, 3, 4, 5})
    assert metrics == MaskMetrics(
        iou=0.4,
        precision=2 / 3,
        recall=0.5,
        f1=4 / 7,
        predicted_count=3,
        gt_count=4,
    )


def test_compute_empty_metrics() -> None:
    metrics = compute_mask_metrics(predicted_ids=set(), gt_ids={1, 2})
    assert metrics.iou == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_metrics.py -q
```

Expected: fail with import error for `codex_agent.scenefunc3d.evaluation`.

- [ ] **Step 3: Add evaluation package marker**

Create `src/codex_agent/scenefunc3d/evaluation/__init__.py`:

```python
"""SceneFunc3D evaluation package."""

from __future__ import annotations

__all__: list[str] = []
```

- [ ] **Step 4: Implement metric math**

Create `src/codex_agent/scenefunc3d/evaluation/metrics.py`:

```python
"""Metric math for SceneFunc3D mask predictions."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Set


@dataclass(frozen=True)
class MaskMetrics:
    """Point-index mask metrics."""

    iou: float
    precision: float
    recall: float
    f1: float
    predicted_count: int
    gt_count: int


def compute_mask_metrics(predicted_ids: Set[int], gt_ids: Set[int]) -> MaskMetrics:
    intersection = len(predicted_ids & gt_ids)
    union = len(predicted_ids | gt_ids)
    precision = intersection / len(predicted_ids) if predicted_ids else 0.0
    recall = intersection / len(gt_ids) if gt_ids else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return MaskMetrics(
        iou=intersection / union if union else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        predicted_count=len(predicted_ids),
        gt_count=len(gt_ids),
    )
```

- [ ] **Step 5: Add lifting and scorer contracts**

Create `src/codex_agent/scenefunc3d/tools/mask_lifting.py`:

```python
"""2D mask to 3D lifting contracts for SceneFunc3D."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class LiftMaskArgs(BaseModel):
    """Arguments for ``lift_mask_to_3d``."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    mask_path: Path
    depth_path: Path
    intrinsics_path: Path
    pose_path: Path


@dataclass(frozen=True)
class LiftMaskResult:
    """3D lifting result for one accepted SAM candidate."""

    frame_id: str
    candidate_id: str
    lifted_point_count: int
    mask_npz_path: Path
    mask_ply_path: Path
    overlay_path: Path

    def to_payload(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "candidate_id": self.candidate_id,
            "lifted_point_count": self.lifted_point_count,
            "mask_npz_path": str(self.mask_npz_path),
            "mask_ply_path": str(self.mask_ply_path),
            "overlay_path": str(self.overlay_path),
        }
```

Create `src/codex_agent/scenefunc3d/evaluation/scorer.py`:

```python
"""SceneFunc3D artifact scoring contracts."""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import MaskMetrics, compute_mask_metrics


@dataclass(frozen=True)
class SceneFunc3dScore:
    """Score for one SceneFunc3D sample."""

    sample_id: str
    metrics: MaskMetrics
    failure_type: str


def score_point_ids(
    *, sample_id: str, predicted_ids: set[int], gt_ids: set[int], failure_type: str = ""
) -> SceneFunc3dScore:
    return SceneFunc3dScore(
        sample_id=sample_id,
        metrics=compute_mask_metrics(predicted_ids, gt_ids),
        failure_type=failure_type,
    )
```

- [ ] **Step 6: Run tests**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/codex_agent/scenefunc3d/tools/mask_lifting.py src/codex_agent/scenefunc3d/evaluation/__init__.py src/codex_agent/scenefunc3d/evaluation/metrics.py src/codex_agent/scenefunc3d/evaluation/scorer.py src/codex_agent/tests/test_scenefunc3d_metrics.py
git commit -m "feat: add scenefunc3d lifting and metrics contracts"
```

## Task 7: Playbook And Runner Skeleton

**Files:**
- Create: `src/codex_agent/scenefunc3d/playbook.py`
- Create: `src/codex_agent/scenefunc3d/runner.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_playbook.py`

- [ ] **Step 1: Write failing playbook tests**

Create `src/codex_agent/tests/test_scenefunc3d_playbook.py`:

```python
"""Tests for the SceneFunc3D inline playbook."""

from __future__ import annotations

from codex_agent.scenefunc3d.playbook import (
    SCENEFUNC3D_TOOL_NAMES,
    SCENEFUNC3D_TOOLS_PLAYBOOK,
)


def test_playbook_mentions_every_tool() -> None:
    for tool_name in SCENEFUNC3D_TOOL_NAMES:
        assert tool_name in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_requires_agent_approval_gates() -> None:
    assert "Molmo point" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "SAM candidates" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "3D lift" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "must approve" in SCENEFUNC3D_TOOLS_PLAYBOOK
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_playbook.py -q
```

Expected: fail with import error for `scenefunc3d.playbook`.

- [ ] **Step 3: Implement playbook**

Create `src/codex_agent/scenefunc3d/playbook.py`:

```python
"""Inline SceneFunc3D tool-using playbook."""

from __future__ import annotations

SCENEFUNC3D_TOOL_NAMES: tuple[str, ...] = (
    "scene_summary",
    "keyframe_selector",
    "view_frame",
    "view_crop",
    "view_bev",
    "frame_objects",
    "molmo_point",
    "sam_mask",
    "lift_mask_to_3d",
    "inspect_mask_artifact",
    "suggest_additional_views",
    "fuse_accepted_masks",
)

SCENEFUNC3D_TOOLS_PLAYBOOK = """\
Playbook for SceneFunc3D mask generation.

No image is evidence until you open it with view_image. First use scene_summary,
keyframe_selector, view_frame, view_crop, view_bev, or frame_objects to find
visual evidence for the task.

Approval gates are mandatory:
1. After molmo_point, inspect the Molmo point overlay. You must approve the
Molmo point before calling sam_mask.
2. After sam_mask, inspect the SAM candidates contact sheet. You must approve
one SAM candidate before calling lift_mask_to_3d.
3. After lift_mask_to_3d, inspect the selected mask overlay and artifact
summary. You must approve the first 3D lift before any multi-view expansion.
4. Every additional view repeats Molmo point approval, SAM candidates approval,
and 3D lift approval before fusion.

For small knobs, handles, dials, switches, and buttons, first identify the
affordance concept, then use a complete task-constrained Molmo point prompt.
Never repeat an expensive Molmo or SAM call with identical arguments after a
model-quality failure. Change crop, prompt, or frame.

Final answer must be compact JSON with mask_artifact_path, mask_npz_path,
mask_ply_path, selected_frame_ids, accepted_fragment_ids, confidence, and
uncertainties.
"""
```

- [ ] **Step 4: Add runner skeleton**

Create `src/codex_agent/scenefunc3d/runner.py`:

```python
"""SceneFunc3D runner skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .playbook import SCENEFUNC3D_TOOLS_PLAYBOOK
from .sample import SceneFunc3dSample, load_sample


@dataclass(frozen=True)
class SceneFunc3dRunnerConfig:
    """Configuration for a SceneFunc3D single-sample run."""

    dataset_root: Path
    output_dir: Path


def build_prompt(sample: SceneFunc3dSample) -> str:
    """Build the prompt prefix for one SceneFunc3D sample."""
    return (
        f"{SCENEFUNC3D_TOOLS_PLAYBOOK}\n\n"
        "Task context:\n"
        f"{sample.agent_context}\n\n"
        "Generate a SceneFunc3D 3D mask artifact for this task."
    )


def load_runner_sample(config: SceneFunc3dRunnerConfig, sample_id: str) -> SceneFunc3dSample:
    """Load the sample a future runtime turn will solve."""
    _ = config.output_dir
    return load_sample(config.dataset_root, sample_id)
```

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_playbook.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/codex_agent/scenefunc3d/playbook.py src/codex_agent/scenefunc3d/runner.py src/codex_agent/tests/test_scenefunc3d_playbook.py
git commit -m "feat: add scenefunc3d playbook and runner skeleton"
```

## Task 8: Real Heavy-Tool Adapters Behind Explicit Backends

**Files:**
- Modify: `src/codex_agent/scenefunc3d/tools/molmo_pointing.py`
- Modify: `src/codex_agent/scenefunc3d/tools/sam_masking.py`
- Modify: `src/codex_agent/scenefunc3d/tools/mask_lifting.py`
- Modify: `src/codex_agent/scenefunc3d/tools/dispatch.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py`

- [ ] **Step 1: Add backend-unavailable tests**

Append to `src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py`:

```python
import pytest

from codex_agent.scenefunc3d.tools.models import ToolInputError
from codex_agent.scenefunc3d.tools.molmo_pointing import MolmoBackendConfig, run_molmo_backend
from codex_agent.scenefunc3d.tools.sam_masking import SamBackendConfig, run_sam_backend


def test_molmo_backend_missing_path_fails(tmp_path: Path) -> None:
    config = MolmoBackendConfig(model_name="MolmoPoint-8B", model_path=tmp_path / "missing")
    with pytest.raises(ToolInputError, match="Molmo backend unavailable"):
        run_molmo_backend(config)


def test_sam_backend_missing_path_fails(tmp_path: Path) -> None:
    config = SamBackendConfig(model_name="SAM2.1-Hiera-L", checkpoint_path=tmp_path / "missing.pt")
    with pytest.raises(ToolInputError, match="SAM backend unavailable"):
        run_sam_backend(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py -q
```

Expected: fail because backend config classes do not exist.

- [ ] **Step 3: Add explicit backend configs**

Add to `src/codex_agent/scenefunc3d/tools/molmo_pointing.py`:

```python
@dataclass(frozen=True)
class MolmoBackendConfig:
    """Explicit Molmo backend selection."""

    model_name: str
    model_path: Path


def run_molmo_backend(config: MolmoBackendConfig) -> None:
    """Validate configured Molmo backend before heavy model loading."""
    if not config.model_path.exists():
        from .models import ToolInputError

        raise ToolInputError(
            f"Molmo backend unavailable: {config.model_name} at {config.model_path}"
        )
```

Add to `src/codex_agent/scenefunc3d/tools/sam_masking.py`:

```python
@dataclass(frozen=True)
class SamBackendConfig:
    """Explicit SAM backend selection."""

    model_name: str
    checkpoint_path: Path


def run_sam_backend(config: SamBackendConfig) -> None:
    """Validate configured SAM backend before heavy model loading."""
    if not config.checkpoint_path.exists():
        from .models import ToolInputError

        raise ToolInputError(
            f"SAM backend unavailable: {config.model_name} at {config.checkpoint_path}"
        )
```

- [ ] **Step 4: Register backend-gated heavy tools as recoverable errors**

Update `TOOL_NAMES` in `src/codex_agent/scenefunc3d/tools/dispatch.py`:

```python
TOOL_NAMES: tuple[str, ...] = (
    "scene_summary",
    "view_frame",
    "keyframe_selector",
    "molmo_point",
    "sam_mask",
    "lift_mask_to_3d",
)
```

Add branches that parse typed args and return recoverable errors until the full model execution is wired:

```python
    if name == "molmo_point":
        from .molmo_pointing import MolmoPointArgs

        _parse(MolmoPointArgs, raw_args)
        raise ToolInputError("molmo_point backend execution is not configured")
    if name == "sam_mask":
        from .sam_masking import SamMaskArgs

        _parse(SamMaskArgs, raw_args)
        raise ToolInputError("sam_mask backend execution is not configured")
    if name == "lift_mask_to_3d":
        from .mask_lifting import LiftMaskArgs

        _parse(LiftMaskArgs, raw_args)
        raise ToolInputError("lift_mask_to_3d backend execution is not configured")
```

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py src/codex_agent/tests/test_scenefunc3d_tools_cli.py -q
```

Expected: all tests pass, with heavy model execution returning recoverable JSON errors when called without configured backends.

- [ ] **Step 6: Commit**

```bash
git add src/codex_agent/scenefunc3d/tools/molmo_pointing.py src/codex_agent/scenefunc3d/tools/sam_masking.py src/codex_agent/scenefunc3d/tools/mask_lifting.py src/codex_agent/scenefunc3d/tools/dispatch.py src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py
git commit -m "feat: add scenefunc3d explicit backend contracts"
```

## Task 9: Inspection, Multi-View, Fusion, And Remaining Evidence Contracts

**Files:**
- Modify: `src/codex_agent/scenefunc3d/tools/frame_views.py`
- Create: `src/codex_agent/scenefunc3d/tools/mask_inspection.py`
- Modify: `src/codex_agent/scenefunc3d/tools/dispatch.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_tools_cli.py`
- Test: `src/codex_agent/tests/test_scenefunc3d_artifacts.py`

- [ ] **Step 1: Add failing tests for remaining evidence tools**

Append to `src/codex_agent/tests/test_scenefunc3d_tools_cli.py`:

```python
def test_cli_frame_objects_returns_empty_visible_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_scene(tmp_path)
    code = main(
        [
            "frame_objects",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"frame_id": "000000"}),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload == {"frame_id": "000000", "objects": []}


def test_cli_view_crop_requires_frame_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_scene(tmp_path)
    code = main(
        [
            "view_crop",
            "--scene-root",
            str(scene_dir),
            "--args",
            json.dumps({"bbox": [0.1, 0.1, 0.5, 0.5]}),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "frame_id" in payload["error"]


def test_cli_view_bev_reports_unavailable_without_bev_asset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scene_dir = _write_scene(tmp_path)
    code = main(["view_bev", "--scene-root", str(scene_dir), "--args", "{}"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "BEV asset is not available" in payload["error"]
```

- [ ] **Step 2: Add failing tests for inspection, multi-view, and fusion contracts**

Append to `src/codex_agent/tests/test_scenefunc3d_artifacts.py`:

```python
from codex_agent.scenefunc3d.tools.mask_inspection import (
    AcceptedFragment,
    FusedMaskResult,
    MaskInspectionResult,
    SuggestedView,
    SuggestedViewsResult,
)


def test_mask_inspection_payload() -> None:
    result = MaskInspectionResult(
        artifact_path=Path("/tmp/summary.json"),
        overlay_paths=(Path("/tmp/overlay.jpg"),),
        lifted_point_count=42,
        status="success",
    )
    payload = result.to_payload()
    assert payload["artifact_path"] == "/tmp/summary.json"
    assert payload["overlay_paths"] == ["/tmp/overlay.jpg"]


def test_suggested_views_payload() -> None:
    result = SuggestedViewsResult(
        seed_fragment_id="000050_mask_00",
        views=(SuggestedView(frame_id="000060", reason="different view angle", rank=1),),
    )
    assert result.to_payload()["views"][0]["reason"] == "different view angle"


def test_fused_mask_payload() -> None:
    result = FusedMaskResult(
        accepted_fragments=(
            AcceptedFragment(fragment_id="000050_mask_00", point_count=1119),
        ),
        mask_npz_path=Path("/tmp/fused/mask_data.npz"),
        mask_ply_path=Path("/tmp/fused/lifted_points.ply"),
    )
    payload = result.to_payload()
    assert payload["accepted_fragments"][0]["fragment_id"] == "000050_mask_00"
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_tools_cli.py src/codex_agent/tests/test_scenefunc3d_artifacts.py -q
```

Expected: fail because `frame_objects`, `view_crop`, `view_bev`, and `mask_inspection` contracts are not implemented.

- [ ] **Step 4: Add remaining evidence tool contracts**

Extend `src/codex_agent/scenefunc3d/tools/frame_views.py`:

```python
class FrameObjectsArgs(BaseModel):
    """Arguments for ``frame_objects``."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(min_length=1)


@dataclass(frozen=True)
class FrameObjectsResult:
    """Visible object summary for one frame."""

    frame_id: str
    objects: tuple[dict[str, object], ...]

    def to_payload(self) -> dict[str, object]:
        return {"frame_id": self.frame_id, "objects": list(self.objects)}


class ViewCropArgs(BaseModel):
    """Arguments for ``view_crop``."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(min_length=1)
    bbox: tuple[float, float, float, float]


class ViewBevArgs(BaseModel):
    """Arguments for ``view_bev``."""

    model_config = ConfigDict(extra="forbid")


def frame_objects(
    tool_scene: SceneFunc3dToolScene, args: FrameObjectsArgs
) -> FrameObjectsResult:
    if args.frame_id not in tool_scene.rgb_frame_ids:
        raise ToolInputError(f"unknown frame_id {args.frame_id!r}")
    return FrameObjectsResult(frame_id=args.frame_id, objects=())


def view_crop(
    tool_scene: SceneFunc3dToolScene, args: ViewCropArgs, *, out_dir: Path
) -> ViewFrameResult:
    if args.frame_id not in tool_scene.rgb_frame_ids:
        raise ToolInputError(f"unknown frame_id {args.frame_id!r}")
    _ = args.bbox
    return view_frame(tool_scene, ViewFrameArgs(frame_ids=(args.frame_id,)), out_dir=out_dir)


def view_bev(
    tool_scene: SceneFunc3dToolScene, args: ViewBevArgs, *, out_dir: Path
) -> ViewFrameResult:
    _ = args
    _ = out_dir
    bev_path = tool_scene.conceptgraph_dir / "bev" / "scene_bev.png"
    if not bev_path.exists():
        raise ToolInputError(f"BEV asset is not available: {bev_path}")
    return ViewFrameResult(
        frames=(FrameViewPayload(frame_id="bev", image_path=bev_path),)
    )
```

Add imports near the top of `frame_views.py`:

```python
from pydantic import Field
```

- [ ] **Step 5: Add inspection and fusion contracts**

Create `src/codex_agent/scenefunc3d/tools/mask_inspection.py`:

```python
"""Mask inspection, multi-view suggestion, and fusion contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaskInspectionResult:
    """Artifact inspection result for agent review."""

    artifact_path: Path
    overlay_paths: tuple[Path, ...]
    lifted_point_count: int
    status: str

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_path": str(self.artifact_path),
            "overlay_paths": [str(path) for path in self.overlay_paths],
            "lifted_point_count": self.lifted_point_count,
            "status": self.status,
        }


@dataclass(frozen=True)
class SuggestedView:
    """One additional view suggested from an accepted 3D seed."""

    frame_id: str
    reason: str
    rank: int

    def to_payload(self) -> dict[str, object]:
        return {"frame_id": self.frame_id, "reason": self.reason, "rank": self.rank}


@dataclass(frozen=True)
class SuggestedViewsResult:
    """Additional views suggested for multi-view expansion."""

    seed_fragment_id: str
    views: tuple[SuggestedView, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "seed_fragment_id": self.seed_fragment_id,
            "views": [view.to_payload() for view in self.views],
        }


@dataclass(frozen=True)
class AcceptedFragment:
    """One accepted 3D mask fragment."""

    fragment_id: str
    point_count: int

    def to_payload(self) -> dict[str, object]:
        return {"fragment_id": self.fragment_id, "point_count": self.point_count}


@dataclass(frozen=True)
class FusedMaskResult:
    """Fused mask artifact result."""

    accepted_fragments: tuple[AcceptedFragment, ...]
    mask_npz_path: Path
    mask_ply_path: Path

    def to_payload(self) -> dict[str, object]:
        return {
            "accepted_fragments": [
                fragment.to_payload() for fragment in self.accepted_fragments
            ],
            "mask_npz_path": str(self.mask_npz_path),
            "mask_ply_path": str(self.mask_ply_path),
        }
```

- [ ] **Step 6: Register remaining tool names**

Update `TOOL_NAMES` in `src/codex_agent/scenefunc3d/tools/dispatch.py`:

```python
TOOL_NAMES: tuple[str, ...] = (
    "scene_summary",
    "view_frame",
    "view_crop",
    "view_bev",
    "frame_objects",
    "keyframe_selector",
    "molmo_point",
    "sam_mask",
    "lift_mask_to_3d",
    "inspect_mask_artifact",
    "suggest_additional_views",
    "fuse_accepted_masks",
)
```

Add branches inside `run_tool`:

```python
    if name == "view_crop":
        from .frame_views import ViewCropArgs, view_crop

        return view_crop(tool_scene, _parse(ViewCropArgs, raw_args), out_dir=out_dir)
    if name == "view_bev":
        from .frame_views import ViewBevArgs, view_bev

        return view_bev(tool_scene, _parse(ViewBevArgs, raw_args), out_dir=out_dir)
    if name == "frame_objects":
        from .frame_views import FrameObjectsArgs, frame_objects

        return frame_objects(tool_scene, _parse(FrameObjectsArgs, raw_args))
    if name == "inspect_mask_artifact":
        raise ToolInputError("inspect_mask_artifact requires an existing artifact path")
    if name == "suggest_additional_views":
        raise ToolInputError("suggest_additional_views requires an accepted seed fragment")
    if name == "fuse_accepted_masks":
        raise ToolInputError("fuse_accepted_masks requires accepted fragment paths")
```

- [ ] **Step 7: Run tests**

Run:

```bash
PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_tools_cli.py src/codex_agent/tests/test_scenefunc3d_artifacts.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/codex_agent/scenefunc3d/tools/frame_views.py src/codex_agent/scenefunc3d/tools/mask_inspection.py src/codex_agent/scenefunc3d/tools/dispatch.py src/codex_agent/tests/test_scenefunc3d_tools_cli.py src/codex_agent/tests/test_scenefunc3d_artifacts.py
git commit -m "feat: add scenefunc3d inspection and multiview contracts"
```

## Task 10: Final Verification

**Files:**
- Verify all files touched in Tasks 1-9.

- [ ] **Step 1: Run focused SceneFunc3D tests**

Run:

```bash
PYTHONPATH=src pytest \
  src/codex_agent/tests/test_scenefunc3d_sample.py \
  src/codex_agent/tests/test_scenefunc3d_state.py \
  src/codex_agent/tests/test_scenefunc3d_artifacts.py \
  src/codex_agent/tests/test_scenefunc3d_tools_cli.py \
  src/codex_agent/tests/test_scenefunc3d_molmo_sam_contracts.py \
  src/codex_agent/tests/test_scenefunc3d_metrics.py \
  src/codex_agent/tests/test_scenefunc3d_playbook.py \
  -q
```

Expected: all focused SceneFunc3D tests pass.

- [ ] **Step 2: Run repository quality gate**

Run:

```bash
ruff check src/
black src/
mypy src/
PYTHONPATH=src pytest src/keyframe/tests -q
```

Expected:

- `ruff check src/` exits 0.
- `black src/` exits 0 after formatting, or reports changed files that are then committed.
- `mypy src/` exits 0.
- `PYTHONPATH=src pytest src/keyframe/tests -q` exits 0.

- [ ] **Step 3: Inspect git status**

Run:

```bash
git status --short
```

Expected: only intentional SceneFunc3D files are modified or staged. Existing unrelated model-gateway files are not staged by this plan.

- [ ] **Step 4: Commit formatting-only changes if black changed files**

Run only if `black src/` modified files:

```bash
git add src/codex_agent/scenefunc3d src/codex_agent/tests/test_scenefunc3d_*.py src/codex_agent/errors.py
git commit -m "style: format scenefunc3d implementation"
```

Expected: commit contains only formatting changes for SceneFunc3D implementation files.

- [ ] **Step 5: Write implementation summary**

Add a final note in the handoff message with:

- commits created
- verification commands run and results
- heavy-tool backend status
- remaining work for full Molmo/SAM execution and real multi-view fusion

Do not mark the task complete if any mandatory quality gate command fails.
