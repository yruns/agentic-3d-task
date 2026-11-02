#!/usr/bin/env bash
# Freeze six-GPU stage storage while preserving the two running split-1 shards.
set -euo pipefail
scope_prepare_python=${SCOPE_PREPARE_PYTHON:-python3}
"$scope_prepare_python" - "$@" <<'PY'
"""Prepare isolated SCOPE stage storage from existing immutable run manifests."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

_SPLITS = tuple(range(1, 11))
_GPU_PAIRS = ((0, 2), (3, 4), (5, 6))
_EXPECTED_SUBTASKS = 2669


@dataclass(frozen=True)
class FrozenWorker:
    """Original scene half and its ordered subtask identities for every split."""

    worker: int
    physical_gpu: int
    start_ratio: float
    end_ratio: float
    expected_ids_by_split: dict[int, tuple[str, ...]]
    logger_suffix_by_split: dict[int, str]


@dataclass(frozen=True)
class FrozenExperiment:
    """Validated original experiment paths and unchanged task partition."""

    experiment: str
    output_dir: Path
    config_path: Path
    workers: tuple[FrozenWorker, ...]


@dataclass(frozen=True)
class OriginalStage:
    """Existing coverage gate and source-file identity, retained in plan v2."""

    split: int
    worker: int
    gate_dir: Path
    expected_subtasks: int


@dataclass(frozen=True)
class TaskIdentity:
    """Validated identity fields used to check the untouched metadata payload."""

    subtask_id: str
    scene_id: str
    episode_id: int
    subtask_index: int
    split: int


@dataclass(frozen=True)
class Artifact:
    """One immutable file to publish after all inputs and conflicts are checked."""

    path: Path
    payload: bytes


class PlanStage(TypedDict):
    split: int
    worker: int
    physical_gpu: int
    cfg_file: str
    source_output_dir: str
    logger_suffix: str
    gate_dir: str
    expected_subtasks: int


class StoragePlan(TypedDict):
    schema_version: Literal[2]
    experiment: str
    central_output_dir: str
    stages: list[PlanStage]


def require_mapping(value: object, location: str) -> dict[str, object]:
    """Validate a JSON object at the deserialization boundary."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{location}: expected a JSON object with string keys")
    return dict(value)


def require_sequence(value: object, location: str) -> list[object]:
    """Validate a JSON array before converting its elements to domain types."""
    if not isinstance(value, list):
        raise ValueError(f"{location}: expected a JSON array")
    return list(value)


def require_text(value: object, location: str) -> str:
    """Reject absent or empty textual identities."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location}: expected a nonempty string")
    return value


def require_integer(value: object, location: str) -> int:
    """Reject booleans, fractional values and negative indices."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{location}={value!r}: expected a nonnegative integer")
    return value


def require_ratio(value: object, location: str) -> float:
    """Validate an original scene-sharding ratio without coercing strings."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{location}={value!r}: expected a ratio in [0, 1]")
    ratio = float(value)
    if not 0 <= ratio <= 1:
        raise ValueError(f"{location}={value!r}: expected a ratio in [0, 1]")
    return ratio


def parse_ids(value: object, location: str) -> tuple[str, ...]:
    """Validate ordered, nonempty and unique subtask identifiers."""
    ids = tuple(
        require_text(item, location) for item in require_sequence(value, location)
    )
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{location}: expected unique nonempty subtask IDs")
    return ids


def parse_worker(value: object, worker: int) -> FrozenWorker:
    """Project one frozen scene half into strongly typed split identities."""
    location = f"frozen.workers[{worker}]"
    raw = require_mapping(value, location)
    gpu = require_integer(raw["physical_gpu"], f"{location}.physical_gpu")
    start = require_ratio(raw["start_ratio"], f"{location}.start_ratio")
    end = require_ratio(raw["end_ratio"], f"{location}.end_ratio")
    if (gpu, start, end) != ((0, 0.0, 0.5), (2, 0.5, 1.0))[worker]:
        raise ValueError(f"{location}: original GPU and scene-half identity differ")
    raw_ids = require_mapping(raw["expected_ids_by_split"], location)
    raw_suffixes = require_mapping(raw["logger_suffix_by_split"], location)
    split_keys = {str(split) for split in _SPLITS}
    if set(raw_ids) != split_keys or set(raw_suffixes) != split_keys:
        raise ValueError(f"{location}: expected exactly splits 1..10")
    ids: dict[int, tuple[str, ...]] = {}
    suffixes: dict[int, str] = {}
    for split in _SPLITS:
        ids[split] = parse_ids(raw_ids[str(split)], f"{location}.split{split}")
        suffix = require_text(raw_suffixes[str(split)], location)
        if suffix != f"{start}_{end}_{split}":
            raise ValueError(f"{location}.split{split}: logger suffix differs")
        suffixes[split] = suffix
    return FrozenWorker(worker, gpu, start, end, ids, suffixes)


def load_experiment(assets_root: Path) -> FrozenExperiment:
    """Validate original paths and all 2,669 disjoint frozen task identities."""
    path = assets_root / "manifests/full2669_frozen_manifest.json"
    decoded: object = json.loads(path.read_bytes())
    raw = require_mapping(decoded, str(path))
    experiment = require_text(raw["experiment"], "frozen.experiment")
    if Path(experiment).name != experiment or experiment in {".", ".."}:
        raise ValueError("frozen.experiment: expected one directory name")
    output_dir = Path(require_text(raw["output_dir"], "frozen.output_dir"))
    config_path = Path(require_text(raw["config_path"], "frozen.config_path"))
    if output_dir != assets_root / "outputs" / experiment:
        raise ValueError("frozen.output_dir: must remain under assets_root/outputs")
    if config_path != assets_root / "manifests/eval_full2669.yaml":
        raise ValueError("frozen.config_path: unexpected original configuration path")
    raw_workers = require_sequence(raw["workers"], "frozen.workers")
    if len(raw_workers) != 2:
        raise ValueError("frozen.workers: expected exactly two original scene halves")
    workers = tuple(parse_worker(item, index) for index, item in enumerate(raw_workers))
    all_ids = tuple(
        subtask_id
        for worker in workers
        for split in _SPLITS
        for subtask_id in worker.expected_ids_by_split[split]
    )
    declared_count = require_integer(raw["all_subtasks"], "frozen.all_subtasks")
    if (
        declared_count != _EXPECTED_SUBTASKS
        or len(all_ids) != _EXPECTED_SUBTASKS
        or len(set(all_ids)) != _EXPECTED_SUBTASKS
    ):
        raise ValueError("frozen.workers: coverage must be 2,669 disjoint subtasks")
    return FrozenExperiment(experiment, output_dir, config_path, workers)


def validate_old_worker(
    value: object, worker: FrozenWorker, assets_root: Path, experiment: str
) -> None:
    """Ensure the existing v1 worker paths and scene halves have not drifted."""
    location = f"storage_plan.workers[{worker.worker}]"
    raw = require_mapping(value, location)
    index = require_integer(raw["worker"], f"{location}.worker")
    gpu = require_integer(raw["physical_gpu"], f"{location}.physical_gpu")
    start = require_ratio(raw["start_ratio"], f"{location}.start_ratio")
    end = require_ratio(raw["end_ratio"], f"{location}.end_ratio")
    cfg_file = Path(require_text(raw["cfg_file"], f"{location}.cfg_file"))
    output_dir = Path(require_text(raw["isolated_output_dir"], location))
    if (
        (index, gpu, start, end)
        != (worker.worker, worker.physical_gpu, worker.start_ratio, worker.end_ratio)
        or cfg_file
        != assets_root / "manifests" / f"eval_full2669_worker{worker.worker}.yaml"
        or output_dir != assets_root / "outputs" / f"{experiment}_worker{worker.worker}"
    ):
        raise ValueError(f"{location}: differs from the original frozen worker")


def parse_old_stage(
    value: object, sources: FrozenExperiment, assets_root: Path
) -> OriginalStage:
    """Validate one v1 stage, including its exact suffix, gate and source path."""
    raw = require_mapping(value, "storage_plan.stages")
    split = require_integer(raw["split"], "storage_plan.stage.split")
    worker_index = require_integer(raw["worker"], "storage_plan.stage.worker")
    if split not in _SPLITS or worker_index not in (0, 1):
        raise ValueError("storage_plan.stage: split must be 1..10 and worker 0 or 1")
    worker = sources.workers[worker_index]
    source = Path(require_text(raw["source_output_dir"], "storage_plan.stage.source"))
    suffix = require_text(raw["logger_suffix"], "storage_plan.stage.logger_suffix")
    gate = Path(require_text(raw["gate_dir"], "storage_plan.stage.gate_dir"))
    count = require_integer(raw["expected_subtasks"], "storage_plan.stage.count")
    original_source = (
        sources.output_dir
        if split == 1
        else assets_root / "outputs" / f"{sources.experiment}_worker{worker_index}"
    )
    original_gate = (
        assets_root
        / "outputs"
        / f"{sources.experiment}_gates"
        / f"split{split}_worker{worker_index}"
    )
    if (
        source != original_source
        or gate != original_gate
        or suffix != worker.logger_suffix_by_split[split]
        or count != len(worker.expected_ids_by_split[split])
    ):
        raise ValueError(
            f"storage_plan.stage split={split}, worker={worker_index}: drift"
        )
    return OriginalStage(split, worker_index, gate, count)


def load_original_stages(
    assets_root: Path, sources: FrozenExperiment
) -> tuple[OriginalStage, ...]:
    """Require the complete immutable v1 plan before generating its successor."""
    path = assets_root / "manifests/storage_plan.json"
    decoded: object = json.loads(path.read_bytes())
    raw = require_mapping(decoded, str(path))
    version = require_integer(raw["schema_version"], "storage_plan.schema_version")
    experiment = require_text(raw["experiment"], "storage_plan.experiment")
    central = Path(require_text(raw["central_output_dir"], "storage_plan.central"))
    if (
        version != 1
        or experiment != sources.experiment
        or central != sources.output_dir
    ):
        raise ValueError("storage_plan: schema, experiment or central output differs")
    workers = require_sequence(raw["workers"], "storage_plan.workers")
    if len(workers) != 2:
        raise ValueError("storage_plan.workers: expected exactly two original workers")
    for worker, value in zip(sources.workers, workers):
        validate_old_worker(value, worker, assets_root, sources.experiment)
    stages = tuple(
        parse_old_stage(value, sources, assets_root)
        for value in require_sequence(raw["stages"], "storage_plan.stages")
    )
    keys = {(stage.split, stage.worker) for stage in stages}
    if len(stages) != 20 or keys != {
        (split, worker) for split in _SPLITS for worker in (0, 1)
    }:
        raise ValueError("storage_plan.stages: expected each of the 20 stages once")
    return tuple(sorted(stages, key=lambda stage: (stage.split, stage.worker)))


def parse_task_identity(subtask_id: str, value: object) -> TaskIdentity:
    """Check externally supplied metadata identity without rewriting any bytes."""
    raw = require_mapping(value, f"metadata.{subtask_id}")
    scene = require_text(raw["scene_id"], f"{subtask_id}.scene_id")
    episode = require_integer(raw["episode_id"], f"{subtask_id}.episode_id")
    index = require_integer(raw["subtask_index"], f"{subtask_id}.subtask_index")
    split = require_integer(raw["split"], f"{subtask_id}.split")
    modality = require_text(raw["task_type"], f"{subtask_id}.task_type")
    if modality not in {"object", "description", "image"}:
        raise ValueError(f"{subtask_id}.task_type: unknown navigation modality")
    if subtask_id != f"{scene}_{episode}_{index}" or split not in _SPLITS:
        raise ValueError(f"{subtask_id}: metadata identity or split differs")
    return TaskIdentity(subtask_id, scene, episode, index, split)


def validate_metadata(payload: bytes, sources: FrozenExperiment) -> None:
    """Require exact frozen task coverage and each task's original split."""
    decoded: object = json.loads(payload)
    raw = require_mapping(decoded, "subtask_metadata")
    tasks = {
        subtask_id: parse_task_identity(subtask_id, value)
        for subtask_id, value in raw.items()
    }
    expected = {
        subtask_id
        for worker in sources.workers
        for split in _SPLITS
        for subtask_id in worker.expected_ids_by_split[split]
    }
    if set(tasks) != expected:
        raise ValueError("subtask_metadata: IDs differ from original frozen coverage")
    for worker in sources.workers:
        for split in _SPLITS:
            if any(
                tasks[task_id].split != split
                for task_id in worker.expected_ids_by_split[split]
            ):
                raise ValueError(
                    f"subtask_metadata: worker={worker.worker}, split={split} drift"
                )


def json_bytes(payload: object) -> bytes:
    """Encode deterministic UTF-8 artifacts using the existing storage convention."""
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def stage_configuration(
    configuration: bytes, original_name: str, stage_name: str
) -> bytes:
    """Replace only exp_name, retaining every other byte and original line endings."""
    lines = configuration.decode("utf-8").splitlines(keepends=True)
    matches = [
        index for index, line in enumerate(lines) if line.startswith("exp_name:")
    ]
    if len(matches) != 1:
        raise ValueError("configuration: expected exactly one exp_name line")
    index = matches[0]
    declared: object = json.loads(lines[index].split(":", 1)[1].strip())
    if declared != original_name:
        raise ValueError("configuration.exp_name: differs from the frozen experiment")
    ending = (
        "\r\n"
        if lines[index].endswith("\r\n")
        else "\n" if lines[index].endswith("\n") else ""
    )
    lines[index] = f"exp_name: {json.dumps(stage_name)}{ending}"
    return "".join(lines).encode("utf-8")


def ensure_compatible(artifact: Artifact) -> None:
    """Refuse to overwrite any pre-existing immutable artifact with other bytes."""
    exists = artifact.path.exists() or artifact.path.is_symlink()
    if exists and artifact.path.read_bytes() != artifact.payload:
        raise ValueError(f"frozen artifact differs: {artifact.path}")


def persist(artifact: Artifact) -> None:
    """Publish atomically using an exclusive hard link; concurrent retries are safe."""
    artifact.path.parent.mkdir(parents=True, exist_ok=True)
    if artifact.path.exists() or artifact.path.is_symlink():
        ensure_compatible(artifact)
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".scope-parallel-", dir=artifact.path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(artifact.payload)
        try:
            os.link(temporary_path, artifact.path)
        except FileExistsError:
            if artifact.path.read_bytes() != artifact.payload:
                raise ValueError(f"frozen artifact differs: {artifact.path}") from None
    finally:
        temporary_path.unlink()


def prepare_parallel_outputs(assets_root: Path) -> Path:
    """Freeze stage storage while keeping split-1 paths and every old gate intact."""
    assets_root = assets_root.absolute()
    sources = load_experiment(assets_root)
    original_stages = load_original_stages(assets_root, sources)
    metadata = (sources.output_dir / "subtask_metadata.json").read_bytes()
    validate_metadata(metadata, sources)
    configuration = sources.config_path.read_bytes()
    stage_configuration(configuration, sources.experiment, sources.experiment)
    stages: list[PlanStage] = []
    artifacts: list[Artifact] = []
    for original in original_stages:
        split, worker_index = original.split, original.worker
        worker = sources.workers[worker_index]
        expected_ids = worker.expected_ids_by_split[split]
        if (original.gate_dir / "subtask_metadata.json").read_bytes() != metadata:
            raise ValueError(f"original gate metadata differs: {original.gate_dir}")
        gate_ids: object = json.loads(
            (original.gate_dir / "expected_subtask_ids.json").read_bytes()
        )
        if parse_ids(gate_ids, str(original.gate_dir)) != expected_ids:
            raise ValueError(f"original gate identities differ: {original.gate_dir}")
        cfg_file, source_dir, gpu = (
            sources.config_path,
            sources.output_dir,
            worker.physical_gpu,
        )
        if split > 1:
            stage_name = f"{sources.experiment}_split{split}_worker{worker_index}"
            cfg_file = (
                assets_root
                / "manifests"
                / f"eval_full2669_split{split}_worker{worker_index}.yaml"
            )
            source_dir = assets_root / "outputs" / stage_name
            gpu = _GPU_PAIRS[(split - 2) % len(_GPU_PAIRS)][worker_index]
            artifacts.extend(
                (
                    Artifact(
                        cfg_file,
                        stage_configuration(
                            configuration, sources.experiment, stage_name
                        ),
                    ),
                    Artifact(source_dir / "subtask_metadata.json", metadata),
                    Artifact(
                        source_dir / "expected_subtask_ids.json",
                        json_bytes(expected_ids),
                    ),
                )
            )
        stages.append(
            PlanStage(
                split=split,
                worker=worker_index,
                physical_gpu=gpu,
                cfg_file=str(cfg_file),
                source_output_dir=str(source_dir),
                logger_suffix=worker.logger_suffix_by_split[split],
                gate_dir=str(original.gate_dir),
                expected_subtasks=original.expected_subtasks,
            )
        )
    plan = StoragePlan(
        schema_version=2,
        experiment=sources.experiment,
        central_output_dir=str(sources.output_dir),
        stages=stages,
    )
    plan_path = assets_root / "manifests/storage_plan_parallel_v2.json"
    artifacts.append(Artifact(plan_path, json_bytes(plan)))
    for artifact in artifacts:
        ensure_compatible(artifact)
    for artifact in artifacts:
        persist(artifact)
    return plan_path


def main() -> None:
    """Parse the asset location and delegate immutable preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-root", type=Path, default=Path("/home/ysh/Datasets/SCOPE")
    )
    options = parser.parse_args()
    print(prepare_parallel_outputs(options.assets_root))


if __name__ == "__main__":
    main()
PY
