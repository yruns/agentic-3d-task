"""Artifact layout and metadata for SceneFunc3D mask-generation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypedDict

from ...errors import SceneFunc3dDataError
from ..final_mask_artifacts import FinalMaskMultiViewDecision


class ArtifactStatus(str, Enum):
    """Status stored in SceneFunc3D run summaries."""

    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


class SceneFunc3dRunSummaryPayload(TypedDict):
    """JSON-ready payload for a SceneFunc3D run summary."""

    sample_id: str
    visit_id: str
    desc_id: str
    task_description: str
    status: str
    selected_frame_ids: list[str]
    accepted_fragment_ids: list[str]
    failure_type: str
    stop_reason: str


class SceneFunc3dRunMultiViewDecisionPayload(TypedDict):
    """JSON-ready first-lift multi-view decision stored in run summary."""

    seed_fragment_id: str
    action: str
    reason: str
    suggested_frame_ids: list[str]


class SceneFunc3dCompletedRunSummaryPayload(TypedDict):
    """JSON-ready payload for a successful SceneFunc3D run summary."""

    sample_id: str
    visit_id: str
    desc_id: str
    task_description: str
    status: str
    selected_frame_ids: list[str]
    accepted_fragment_ids: list[str]
    failure_type: str
    stop_reason: str
    mask_artifact_path: str
    mask_npz_path: str
    mask_ply_path: str
    confidence: float
    uncertainties: list[str]
    multi_view_decision: SceneFunc3dRunMultiViewDecisionPayload
    final_point_count: int


class SceneFunc3dRunCompletionEventPayload(TypedDict):
    """JSONL event written after a successful SceneFunc3D sample run."""

    event_type: str
    sample_id: str
    status: str
    result_path: str
    summary_path: str
    mask_artifact_path: str


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
        """Create all artifact directories for a sample run."""
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

    def to_payload(self) -> SceneFunc3dRunSummaryPayload:
        """Return the JSON-ready representation stored in ``summary.json``."""
        return {
            "sample_id": self.sample_id,
            "visit_id": self.visit_id,
            "desc_id": self.desc_id,
            "task_description": self.task_description,
            "status": self.status.value,
            "selected_frame_ids": list(self.selected_frame_ids),
            "accepted_fragment_ids": list(self.accepted_fragment_ids),
            "failure_type": self.failure_type,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class SceneFunc3dCompletedRunSummary:
    """Durable run summary after a valid final mask artifact is produced."""

    sample_id: str
    visit_id: str
    desc_id: str
    task_description: str
    selected_frame_ids: tuple[str, ...]
    accepted_fragment_ids: tuple[str, ...]
    mask_artifact_path: Path
    mask_npz_path: Path
    mask_ply_path: Path
    confidence: float
    uncertainties: tuple[str, ...]
    multi_view_decision: FinalMaskMultiViewDecision
    final_point_count: int

    def to_payload(self) -> SceneFunc3dCompletedRunSummaryPayload:
        """Return the JSON-ready representation stored in ``summary.json``."""
        return {
            "sample_id": self.sample_id,
            "visit_id": self.visit_id,
            "desc_id": self.desc_id,
            "task_description": self.task_description,
            "status": ArtifactStatus.SUCCESS.value,
            "selected_frame_ids": list(self.selected_frame_ids),
            "accepted_fragment_ids": list(self.accepted_fragment_ids),
            "failure_type": "",
            "stop_reason": self._stop_reason(),
            "mask_artifact_path": str(self.mask_artifact_path),
            "mask_npz_path": str(self.mask_npz_path),
            "mask_ply_path": str(self.mask_ply_path),
            "confidence": self.confidence,
            "uncertainties": list(self.uncertainties),
            "multi_view_decision": _multi_view_decision_payload(
                self.multi_view_decision
            ),
            "final_point_count": self.final_point_count,
        }

    def _stop_reason(self) -> str:
        if self.multi_view_decision.action.value == "stop":
            return "single_view_complete"
        return "multiview_expanded"


@dataclass(frozen=True)
class SceneFunc3dRunCompletionEvent:
    """One JSONL event marking a validated SceneFunc3D run completion."""

    sample_id: str
    result_path: Path
    summary_path: Path
    mask_artifact_path: Path

    def to_payload(self) -> SceneFunc3dRunCompletionEventPayload:
        """Return the JSON-ready completion event."""
        return {
            "event_type": "run_completed",
            "sample_id": self.sample_id,
            "status": ArtifactStatus.SUCCESS.value,
            "result_path": str(self.result_path),
            "summary_path": str(self.summary_path),
            "mask_artifact_path": str(self.mask_artifact_path),
        }


def _validate_artifact_path_component(component_name: str, component_value: str) -> str:
    """Return a safe single path component or raise a data error."""
    is_empty_or_whitespace = component_value.strip() == ""
    is_current_or_parent_reference = component_value in {".", ".."}
    has_path_separator = "/" in component_value or "\\" in component_value
    if (
        is_empty_or_whitespace
        or is_current_or_parent_reference
        or has_path_separator
        or Path(component_value).is_absolute()
    ):
        raise SceneFunc3dDataError(
            f"invalid artifact path component {component_name}={component_value!r}"
        )
    return component_value


def artifact_paths_for(
    output_dir: Path, visit_id: str, desc_id: str
) -> MaskArtifactPaths:
    """Return canonical artifact paths for one visit-description pair."""
    safe_visit_id = _validate_artifact_path_component("visit_id", visit_id)
    safe_desc_id = _validate_artifact_path_component("desc_id", desc_id)
    root = output_dir / safe_visit_id / safe_desc_id
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
    """Write one run summary and return the written JSON path."""
    paths.create_dirs()
    paths.summary_json.write_text(
        json.dumps(summary.to_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths.summary_json


def write_completed_run_summary(
    paths: MaskArtifactPaths, summary: SceneFunc3dCompletedRunSummary
) -> Path:
    """Write the successful run summary and return the written JSON path."""
    paths.create_dirs()
    paths.summary_json.write_text(
        json.dumps(summary.to_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths.summary_json


def append_run_completion_event(
    paths: MaskArtifactPaths, event: SceneFunc3dRunCompletionEvent
) -> Path:
    """Append one successful-run event to ``events.jsonl``."""
    paths.create_dirs()
    with paths.events_jsonl.open("a", encoding="utf-8") as event_file:
        event_file.write(json.dumps(event.to_payload(), ensure_ascii=False) + "\n")
    return paths.events_jsonl


def _multi_view_decision_payload(
    decision: FinalMaskMultiViewDecision,
) -> SceneFunc3dRunMultiViewDecisionPayload:
    return {
        "seed_fragment_id": decision.seed_fragment_id,
        "action": decision.action.value,
        "reason": decision.reason,
        "suggested_frame_ids": list(decision.suggested_frame_ids),
    }
