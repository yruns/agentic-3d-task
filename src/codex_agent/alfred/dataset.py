"""Validated ALFRED validation episodes and reproducible pilot folds.

Only :class:`AlfredInstruction` is an actor input. Task labels, expert lengths,
and trajectory paths belong to the simulator/evaluator boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ValidationSplit = Literal["valid_seen", "valid_unseen"]
TaskType = Literal[
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_and_place_with_movable_recep",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
]
TASK_TYPES: tuple[TaskType, ...] = (
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_and_place_with_movable_recep",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
)
VALIDATION_SPLITS: tuple[ValidationSplit, ...] = ("valid_seen", "valid_unseen")


class _SplitEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    task: str = Field(min_length=1)
    repeat_idx: int = Field(ge=0)

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or len(path.parts) != 2
            or any(part in (".", "..") for part in path.parts)
            or "\\" in value
            or "\x00" in value
            or str(path) != value
        ):
            raise ValueError(f"task={value!r} must be a relative '<type>/trial' path")
        return value


class _RawAnnotation(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    task_desc: str = Field(min_length=1)
    high_descs: list[str]

    @field_validator("task_desc")
    @classmethod
    def validate_goal(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task_desc must contain non-whitespace text")
        return value


class _RawAnnotations(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    anns: list[_RawAnnotation] = Field(min_length=1)


class _RawPlan(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    low_actions: list[dict[str, object]] = Field(min_length=1)


class _RawTrajectory(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    task_id: str = Field(min_length=1)
    task_type: TaskType
    plan: _RawPlan
    turk_annotations: _RawAnnotations


@dataclass(frozen=True)
class AlfredInstruction:
    """Language-only input; contains neither expert actions nor simulator labels."""

    goal: str
    steps: tuple[str, ...]


@dataclass(frozen=True)
class AlfredEpisode:
    """Evaluator-owned episode identity and the separately scoped actor input."""

    split: ValidationSplit
    task: str
    repeat_idx: int
    task_id: str
    task_type: TaskType
    trajectory_path: Path
    trajectory_sha256: str
    expert_actions: int
    instruction: AlfredInstruction

    @property
    def episode_id(self) -> str:
        """Return an identity unique across splits and language annotations."""
        return f"{self.split}/{self.task}#{self.repeat_idx}"


class FoldSample(_SplitEntry):
    """Portable frozen episode reference, including its exact trajectory hash."""

    split: ValidationSplit
    task_type: TaskType
    trajectory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PilotManifest(BaseModel):
    """Versioned, immutable selection protocol and selected validation entries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    selection: Literal["stratified_round_robin_sha256_v1"] = (
        "stratified_round_robin_sha256_v1"
    )
    seed: int = Field(strict=True, ge=0)
    per_split: int = Field(strict=True, ge=len(TASK_TYPES))
    splits_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    samples: tuple[FoldSample, ...]

    @model_validator(mode="after")
    def validate_samples(self) -> PilotManifest:
        tasks = [sample.task for sample in self.samples]
        if len(tasks) != len(set(tasks)):
            raise ValueError("samples contain duplicate trajectories")
        for split in VALIDATION_SPLITS:
            selected = [sample for sample in self.samples if sample.split == split]
            if len(selected) != self.per_split:
                raise ValueError(f"samples for {split} must number {self.per_split}")
            if {sample.task_type for sample in selected} != set(TASK_TYPES):
                raise ValueError(f"samples for {split} must cover all seven task types")
        return self


class PilotSubsetManifest(BaseModel):
    """Outcome-independent pilot15 derived from the immutable original pilot50."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[2] = 2
    selection: Literal["parent_stratified_pilot15_sha256_v1"] = (
        "parent_stratified_pilot15_sha256_v1"
    )
    seed: int = Field(ge=0)
    parent_filename: str = Field(pattern=r"^[A-Za-z0-9_-]+\.json$")
    parent_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    splits_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_counts: dict[ValidationSplit, int]
    samples: tuple[FoldSample, ...]

    @model_validator(mode="after")
    def validate_samples(self) -> PilotSubsetManifest:
        if self.split_counts != {"valid_seen": 7, "valid_unseen": 8}:
            raise ValueError("pilot15 split_counts must be 7 seen and 8 unseen")
        if len(self.samples) != 15 or len({s.task for s in self.samples}) != 15:
            raise ValueError("pilot15 requires exactly 15 distinct trajectories")
        for split, count in self.split_counts.items():
            selected = [s for s in self.samples if s.split == split]
            if len(selected) != count or {s.task_type for s in selected} != set(
                TASK_TYPES
            ):
                raise ValueError(f"pilot15 {split} must cover all seven task types")
        return self


class FullValidationManifest(BaseModel):
    """Official full validation splits (valid_seen + valid_unseen) in oct21 order."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[3] = 3
    selection: Literal["official_validation_splits_v1"] = (
        "official_validation_splits_v1"
    )
    splits_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_counts: dict[ValidationSplit, int]
    samples: tuple[FoldSample, ...]

    @model_validator(mode="after")
    def validate_samples(self) -> FullValidationManifest:
        if not self.samples:
            raise ValueError("full validation manifest must not be empty")
        identities = [(s.split, s.task, s.repeat_idx) for s in self.samples]
        if len(identities) != len(set(identities)):
            raise ValueError("full validation manifest contains duplicate episodes")
        for split in VALIDATION_SPLITS:
            selected = [s for s in self.samples if s.split == split]
            if len(selected) != self.split_counts.get(split, 0):
                raise ValueError(
                    f"full validation {split} count={len(selected)} differs from {self.split_counts}"
                )
        return self


def create_full_validation_manifest(
    data_root: Path, splits_path: Path
) -> FullValidationManifest:
    """Create the frozen manifest for all official valid_seen and valid_unseen entries."""
    samples: list[FoldSample] = []
    split_counts: dict[ValidationSplit, int] = {}
    for split in VALIDATION_SPLITS:
        episodes = load_split(data_root, splits_path, split=split)
        split_counts[split] = len(episodes)
        for episode in episodes:
            samples.append(
                FoldSample(
                    split=episode.split,
                    task=episode.task,
                    repeat_idx=episode.repeat_idx,
                    task_type=episode.task_type,
                    trajectory_sha256=episode.trajectory_sha256,
                )
            )
    return FullValidationManifest(
        splits_sha256=hashlib.sha256(splits_path.read_bytes()).hexdigest(),
        split_counts=split_counts,
        samples=tuple(samples),
    )


def create_pilot15_manifest(parent_path: Path, *, seed: int = 0) -> PilotSubsetManifest:
    """Select one parent sample per split/type plus one additional unseen sample.

    SHA-256 of ``seed:split/task#repeat`` ranks candidates without reading scores,
    instructions or execution traces. Output order follows the original parent.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError(f"seed={seed!r} must be a nonnegative integer")
    parent_bytes = parent_path.read_bytes()
    parent = PilotManifest.model_validate_json(parent_bytes)
    if parent.per_split != 25:
        raise ValueError("pilot15 parent must be the original balanced pilot50")

    def identity(sample: FoldSample) -> str:
        return f"{sample.split}/{sample.task}#{sample.repeat_idx}"

    def rank(sample: FoldSample) -> str:
        return hashlib.sha256(f"{seed}:{identity(sample)}".encode()).hexdigest()

    chosen: set[str] = set()
    for split in VALIDATION_SPLITS:
        for task_type in TASK_TYPES:
            candidates = [
                sample
                for sample in parent.samples
                if sample.split == split and sample.task_type == task_type
            ]
            chosen.add(identity(min(candidates, key=rank)))
    extra = min(
        (
            sample
            for sample in parent.samples
            if sample.split == "valid_unseen" and identity(sample) not in chosen
        ),
        key=rank,
    )
    chosen.add(identity(extra))
    return PilotSubsetManifest(
        seed=seed,
        parent_filename=parent_path.name,
        parent_sha256=hashlib.sha256(parent_bytes).hexdigest(),
        splits_sha256=parent.splits_sha256,
        split_counts={"valid_seen": 7, "valid_unseen": 8},
        samples=tuple(s for s in parent.samples if identity(s) in chosen),
    )


def load_episode(
    data_root: Path, *, split: ValidationSplit, task: str, repeat_idx: int
) -> AlfredEpisode:
    """Read one raw ``<split>/<task>/traj_data.json`` without FILM preprocessing.

    Raises ValueError for unsafe paths, unknown splits, invalid annotations, or
    malformed trajectory fields; missing files retain their FileNotFoundError.
    """
    entry = _SplitEntry(task=task, repeat_idx=repeat_idx)
    if split not in VALIDATION_SPLITS:
        raise ValueError(f"split={split!r} must be one of {VALIDATION_SPLITS}")
    root = data_root.resolve()
    path = (root / split / entry.task / "traj_data.json").resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"task={task!r} resolves outside data_root={root}")
    content = path.read_bytes()
    trajectory = _RawTrajectory.model_validate_json(content)
    if trajectory.task_id != PurePosixPath(entry.task).name:
        raise ValueError(
            f"{path}: task_id={trajectory.task_id!r} does not match task path"
        )
    annotations = trajectory.turk_annotations.anns
    if repeat_idx >= len(annotations):
        raise ValueError(
            f"{path}: repeat_idx={repeat_idx} must be in [0, {len(annotations)})"
        )
    annotation = annotations[repeat_idx]
    return AlfredEpisode(
        split=split,
        task=entry.task,
        repeat_idx=repeat_idx,
        task_id=trajectory.task_id,
        task_type=trajectory.task_type,
        trajectory_path=path,
        trajectory_sha256=hashlib.sha256(content).hexdigest(),
        expert_actions=len(trajectory.plan.low_actions),
        instruction=AlfredInstruction(
            annotation.task_desc, tuple(annotation.high_descs)
        ),
    )


def _read_split_entries(
    splits_path: Path, split: ValidationSplit
) -> tuple[_SplitEntry, ...]:
    payload = json.loads(splits_path.read_bytes())
    if not isinstance(payload, dict) or not isinstance(payload.get(split), list):
        raise ValueError(f"{splits_path}: split={split!r} must be a JSON array")
    entries = tuple(_SplitEntry.model_validate(item) for item in payload[split])
    identities = [(entry.task, entry.repeat_idx) for entry in entries]
    if len(identities) != len(set(identities)):
        raise ValueError(
            f"{splits_path}: split={split!r} has duplicate task/repeat_idx"
        )
    if not entries:
        raise ValueError(f"{splits_path}: split={split!r} is empty")
    return entries


def load_split(
    data_root: Path, splits_path: Path, *, split: ValidationSplit
) -> tuple[AlfredEpisode, ...]:
    """Load every official validation entry in its recorded order."""
    return tuple(
        load_episode(
            data_root, split=split, task=entry.task, repeat_idx=entry.repeat_idx
        )
        for entry in _read_split_entries(splits_path, split)
    )


def create_pilot_manifest(
    data_root: Path, splits_path: Path, *, per_split: int = 25, seed: int = 0
) -> PilotManifest:
    """Select balanced task-type strata, with one annotation per trajectory.

    Within each type, a SHA-256 rank over seed and episode identity fixes the
    order. Round-robin selection across the seven sorted types fixes quotas.
    This defines the first pilot protocol; it is not an existing ALFRED fold.
    """
    if type(per_split) is not int or per_split < len(TASK_TYPES):
        raise ValueError(f"per_split={per_split!r} must be an integer >= 7")
    if type(seed) is not int or seed < 0:
        raise ValueError(f"seed={seed!r} must be a nonnegative integer")
    samples: list[FoldSample] = []
    used_tasks: set[str] = set()
    for split in VALIDATION_SPLITS:
        candidates: dict[TaskType, list[AlfredEpisode]] = defaultdict(list)
        ranked = sorted(
            load_split(data_root, splits_path, split=split),
            key=lambda episode: hashlib.sha256(
                f"{seed}:{episode.episode_id}".encode()
            ).hexdigest(),
        )
        split_tasks: set[str] = set()
        for episode in ranked:
            if episode.task not in used_tasks and episode.task not in split_tasks:
                candidates[episode.task_type].append(episode)
                split_tasks.add(episode.task)
        if set(candidates) != set(TASK_TYPES):
            raise ValueError(f"split={split!r} must contain all seven task types")
        selected = 0
        round_index = 0
        while selected < per_split:
            before = selected
            for task_type in TASK_TYPES:
                if round_index < len(candidates[task_type]):
                    episode = candidates[task_type][round_index]
                    samples.append(
                        FoldSample(
                            split=episode.split,
                            task=episode.task,
                            repeat_idx=episode.repeat_idx,
                            task_type=episode.task_type,
                            trajectory_sha256=episode.trajectory_sha256,
                        )
                    )
                    used_tasks.add(episode.task)
                    selected += 1
                    if selected == per_split:
                        break
            if selected == before:
                raise ValueError(
                    f"split={split!r} has fewer than {per_split} unique trajectories"
                )
            round_index += 1
    return PilotManifest(
        seed=seed,
        per_split=per_split,
        splits_sha256=hashlib.sha256(splits_path.read_bytes()).hexdigest(),
        samples=tuple(samples),
    )


def load_manifest(
    path: Path, data_root: Path, splits_path: Path
) -> tuple[AlfredEpisode, ...]:
    """Verify frozen selection, official membership, and every trajectory hash."""
    content = path.read_bytes()
    payload = json.loads(content)
    manifest: PilotManifest | PilotSubsetManifest | FullValidationManifest
    if isinstance(payload, dict) and payload.get("schema_version") == 2:
        subset = PilotSubsetManifest.model_validate_json(content)
        parent_path = path.parent / subset.parent_filename
        if hashlib.sha256(parent_path.read_bytes()).hexdigest() != subset.parent_sha256:
            raise ValueError(f"{path}: parent_sha256 differs from {parent_path}")
        if subset != create_pilot15_manifest(parent_path, seed=subset.seed):
            raise ValueError(
                f"{path}: samples differ from deterministic parent selection"
            )
        manifest = subset
    elif isinstance(payload, dict) and payload.get("schema_version") == 3:
        manifest = FullValidationManifest.model_validate_json(content)
    else:
        manifest = PilotManifest.model_validate_json(content)
    if manifest.splits_sha256 != hashlib.sha256(splits_path.read_bytes()).hexdigest():
        raise ValueError(f"{path}: splits_sha256 differs from {splits_path}")
    allowed = {
        split: {
            (entry.task, entry.repeat_idx)
            for entry in _read_split_entries(splits_path, split)
        }
        for split in VALIDATION_SPLITS
    }
    episodes: list[AlfredEpisode] = []
    for sample in manifest.samples:
        if (sample.task, sample.repeat_idx) not in allowed[sample.split]:
            raise ValueError(f"{path}: sample={sample.task!r} is not in official split")
        episode = load_episode(
            data_root,
            split=sample.split,
            task=sample.task,
            repeat_idx=sample.repeat_idx,
        )
        if episode.trajectory_sha256 != sample.trajectory_sha256:
            raise ValueError(
                f"{path}: trajectory_sha256 differs for {episode.episode_id}"
            )
        if episode.task_type != sample.task_type:
            raise ValueError(f"{path}: task_type differs for {episode.episode_id}")
        episodes.append(episode)
    return tuple(episodes)
