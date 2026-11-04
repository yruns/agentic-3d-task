"""ALFRED fold provenance and separation of actor instructions from labels."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from codex_agent.alfred.dataset import (
    TASK_TYPES,
    create_pilot_manifest,
    load_episode,
    load_manifest,
    load_split,
)


def _write_dataset(root: Path) -> Path:
    splits: dict[str, list[dict[str, object]]] = {}
    for split in ("valid_seen", "valid_unseen"):
        entries: list[dict[str, object]] = []
        for kind in TASK_TYPES:
            for trial in range(4):
                task = f"{kind}-Apple-None-Table-1/trial_{split}_{trial}"
                directory = root / split / task
                directory.mkdir(parents=True)
                trajectory = {
                    "task_id": f"trial_{split}_{trial}",
                    "task_type": kind,
                    "scene": {"floor_plan": "FloorPlan1"},
                    "pddl_params": {"object_target": "SECRET"},
                    "plan": {"low_actions": [{"action": "SECRET"}] * 10},
                    "turk_annotations": {
                        "anns": [
                            {
                                "task_desc": "Put the apple on the table.",
                                "high_descs": ["Pick up apple."],
                            },
                            {
                                "task_desc": "Move an apple to a table.",
                                "high_descs": ["Find the apple."],
                            },
                        ]
                    },
                }
                (directory / "traj_data.json").write_text(json.dumps(trajectory))
                entries.extend(
                    {"task": task, "repeat_idx": repeat} for repeat in range(2)
                )
        splits[split] = entries
    path = root / "splits.json"
    path.write_text(json.dumps(splits))
    return path


def test_episode_only_exposes_language_to_actor(tmp_path: Path) -> None:
    splits_path = _write_dataset(tmp_path)
    episode = load_split(tmp_path, splits_path, split="valid_seen")[0]
    assert asdict(episode.instruction) == {
        "goal": "Put the apple on the table.",
        "steps": ("Pick up apple.",),
    }
    assert episode.expert_actions == 10
    assert len(episode.trajectory_sha256) == 64
    assert episode.episode_id.endswith("#0")


def test_frozen_fold_is_balanced_deterministic_and_has_unique_trajectories(
    tmp_path: Path,
) -> None:
    splits_path = _write_dataset(tmp_path)
    manifest = create_pilot_manifest(tmp_path, splits_path)
    assert manifest == create_pilot_manifest(tmp_path, splits_path)
    assert len(manifest.samples) == 50
    assert len({sample.task for sample in manifest.samples}) == 50
    for split in ("valid_seen", "valid_unseen"):
        samples = [sample for sample in manifest.samples if sample.split == split]
        assert len(samples) == 25
        assert {sample.task_type for sample in samples} == set(TASK_TYPES)
    manifest_path = tmp_path / "pilot.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    assert len(load_manifest(manifest_path, tmp_path, splits_path)) == 50
    first = manifest.samples[0]
    trajectory_path = tmp_path / first.split / first.task / "traj_data.json"
    trajectory_path.write_text(trajectory_path.read_text() + "\n")
    with pytest.raises(ValueError, match="trajectory_sha256"):
        load_manifest(manifest_path, tmp_path, splits_path)


def test_fold_rejects_modified_split_and_nonmember_sample(tmp_path: Path) -> None:
    splits_path = _write_dataset(tmp_path)
    manifest = create_pilot_manifest(tmp_path, splits_path)
    manifest_path = tmp_path / "pilot.json"
    manifest_path.write_text(manifest.model_dump_json())
    splits_path.write_text(splits_path.read_text() + "\n")
    with pytest.raises(ValueError, match="splits_sha256"):
        load_manifest(manifest_path, tmp_path, splits_path)
    splits_path.write_text(splits_path.read_text()[:-1])
    payload = json.loads(manifest.model_dump_json())
    payload["samples"][0]["repeat_idx"] = 99
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="official split"):
        load_manifest(manifest_path, tmp_path, splits_path)


@pytest.mark.parametrize(
    "task", ["../outside/trial", "/absolute/trial", "type/../trial", "type/trial/extra"]
)
def test_rejects_unsafe_task_paths(tmp_path: Path, task: str) -> None:
    with pytest.raises(ValueError, match="task"):
        load_episode(tmp_path, split="valid_seen", task=task, repeat_idx=0)


def test_rejects_missing_annotation_and_duplicate_split_entry(tmp_path: Path) -> None:
    splits_path = _write_dataset(tmp_path)
    split_payload = json.loads(splits_path.read_text())
    first = split_payload["valid_seen"][0]
    with pytest.raises(ValueError, match="repeat_idx"):
        load_episode(tmp_path, split="valid_seen", task=first["task"], repeat_idx=2)
    split_payload["valid_seen"].append(first)
    splits_path.write_text(json.dumps(split_payload))
    with pytest.raises(ValueError, match="duplicate"):
        load_split(tmp_path, splits_path, split="valid_seen")
