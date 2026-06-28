"""Artifact layout and metadata for SceneFunc3D mask-generation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypedDict


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


def artifact_paths_for(
    output_dir: Path, visit_id: str, desc_id: str
) -> MaskArtifactPaths:
    """Return canonical artifact paths for one visit-description pair."""
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
    """Write one run summary and return the written JSON path."""
    paths.create_dirs()
    paths.summary_json.write_text(
        json.dumps(summary.to_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths.summary_json
