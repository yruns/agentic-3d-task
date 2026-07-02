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
