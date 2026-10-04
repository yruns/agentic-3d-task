#!/usr/bin/env bash
# Freeze isolated future worker outputs and exact stage coverage gates.
set -euo pipefail
scope_prepare_python=${SCOPE_PREPARE_PYTHON:-python3}
"$scope_prepare_python" - "$@" <<'PY'
"""Prepare SCOPE output storage without changing navigation configuration."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

_SPLITS = tuple(range(1, 11))


@dataclass(frozen=True)
class FrozenWorker:
    """Validated scene-shard identity and requested subtask coverage."""

    worker: int
    physical_gpu: int
    start_ratio: float
    end_ratio: float
    expected_ids_by_split: dict[int, tuple[str, ...]]
    logger_suffix_by_split: dict[int, str]


@dataclass(frozen=True)
class FrozenSources:
    """Inputs required to prepare future worker storage."""

    experiment: str
    output_dir: Path
    config_path: Path
    workers: tuple[FrozenWorker, ...]


class PlanWorker(TypedDict):
    worker: int
    physical_gpu: int
    cfg_file: str
    isolated_output_dir: str
    start_ratio: float
    end_ratio: float


class PlanStage(TypedDict):
    split: int
    worker: int
    source_output_dir: str
    logger_suffix: str
    gate_dir: str
    expected_subtasks: int


class StoragePlan(TypedDict):
    schema_version: Literal[1]
    experiment: str
    central_output_dir: str
    workers: list[PlanWorker]
    stages: list[PlanStage]


def require_mapping(value: object, location: str) -> dict[str, object]:
    """Validate a JSON object before projecting it into concrete types."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{location}: expected a JSON object with string keys")
    return dict(value)


def require_sequence(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{location}: expected a JSON array")
    return list(value)


def require_text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location}: expected a nonempty string")
    return value


def require_nonnegative_integer(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{location}={value!r}: expected a nonnegative integer")
    return value


def require_ratio(value: object, location: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{location}={value!r}: expected a ratio in [0, 1]")
    ratio = float(value)
    if not 0 <= ratio <= 1:
        raise ValueError(f"{location}={value!r}: expected a ratio in [0, 1]")
    return ratio


def require_split_mapping(value: object, location: str) -> dict[str, object]:
    splits = require_mapping(value, location)
    if set(splits) != {str(split) for split in _SPLITS}:
        raise ValueError(f"{location}: expected exactly splits 1..10")
    return splits


def parse_worker(value: object, worker_index: int) -> FrozenWorker:
    """Validate one frozen worker and its logger's exact filename suffixes."""
    location = f"workers[{worker_index}]"
    raw = require_mapping(value, location)
    gpu = require_nonnegative_integer(raw["physical_gpu"], f"{location}.physical_gpu")
    if gpu == 1:
        raise ValueError(f"{location}.physical_gpu=1: GPU 1 is excluded by CLAUDE.md")
    start = require_ratio(raw["start_ratio"], f"{location}.start_ratio")
    end = require_ratio(raw["end_ratio"], f"{location}.end_ratio")
    if start >= end:
        raise ValueError(f"{location}: start_ratio must be less than end_ratio")
    raw_ids = require_split_mapping(raw["expected_ids_by_split"], location)
    raw_suffixes = require_split_mapping(raw["logger_suffix_by_split"], location)
    expected_ids: dict[int, tuple[str, ...]] = {}
    suffixes: dict[int, str] = {}
    for split in _SPLITS:
        ids = tuple(
            require_text(subtask_id, f"{location}.split{split}.subtask_id")
            for subtask_id in require_sequence(raw_ids[str(split)], location)
        )
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{location}.split{split}: expected unique nonempty IDs")
        suffix = require_text(raw_suffixes[str(split)], location)
        if suffix != f"{start}_{end}_{split}":
            raise ValueError(
                f"{location}.split{split}: logger suffix does not match shard"
            )
        expected_ids[split] = ids
        suffixes[split] = suffix
    return FrozenWorker(worker_index, gpu, start, end, expected_ids, suffixes)


def load_sources(assets_root: Path) -> FrozenSources:
    """Validate frozen paths and disjoint worker coverage before any writes."""
    manifest_path = assets_root / "manifests/full2669_frozen_manifest.json"
    payload: object = json.loads(manifest_path.read_bytes())
    raw = require_mapping(payload, str(manifest_path))
    experiment = require_text(raw["experiment"], "experiment")
    if Path(experiment).name != experiment or experiment in {".", ".."}:
        raise ValueError("experiment: expected a single directory name")
    output_dir = Path(require_text(raw["output_dir"], "output_dir"))
    config_path = Path(require_text(raw["config_path"], "config_path"))
    if output_dir != assets_root / "outputs" / experiment:
        raise ValueError("output_dir: frozen path must be under assets_root/outputs")
    if config_path != assets_root / "manifests/eval_full2669.yaml":
        raise ValueError(
            "config_path: expected assets_root/manifests/eval_full2669.yaml"
        )
    workers = tuple(
        parse_worker(worker, index)
        for index, worker in enumerate(require_sequence(raw["workers"], "workers"))
    )
    if len(workers) != 2 or len({worker.physical_gpu for worker in workers}) != 2:
        raise ValueError("workers: expected two workers on distinct allowed GPUs")
    all_ids = tuple(
        subtask_id
        for worker in workers
        for split in _SPLITS
        for subtask_id in worker.expected_ids_by_split[split]
    )
    requested_count = require_nonnegative_integer(raw["all_subtasks"], "all_subtasks")
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != requested_count:
        raise ValueError("workers: coverage must be disjoint and match all_subtasks")
    return FrozenSources(experiment, output_dir, config_path, workers)


def validate_metadata(payload: bytes, sources: FrozenSources) -> None:
    """Check original metadata identities while retaining its exact bytes."""
    decoded: object = json.loads(payload)
    metadata = require_mapping(decoded, "subtask_metadata")
    expected_ids = {
        subtask_id
        for worker in sources.workers
        for split in _SPLITS
        for subtask_id in worker.expected_ids_by_split[split]
    }
    if set(metadata) != expected_ids:
        raise ValueError("subtask_metadata: IDs do not match frozen worker coverage")
    for worker in sources.workers:
        for split in _SPLITS:
            for subtask_id in worker.expected_ids_by_split[split]:
                task = require_mapping(metadata[subtask_id], subtask_id)
                scene = require_text(task["scene_id"], f"{subtask_id}.scene_id")
                episode = require_nonnegative_integer(
                    task["episode_id"], f"{subtask_id}.episode_id"
                )
                index = require_nonnegative_integer(
                    task["subtask_index"], f"{subtask_id}.subtask_index"
                )
                task_split = require_nonnegative_integer(
                    task["split"], f"{subtask_id}.split"
                )
                if subtask_id != f"{scene}_{episode}_{index}" or task_split != split:
                    raise ValueError(
                        f"{subtask_id}: metadata identity or split differs"
                    )
                task_type = require_text(task["task_type"], f"{subtask_id}.task_type")
                if task_type not in {"object", "description", "image"}:
                    raise ValueError(
                        f"{subtask_id}.task_type: unexpected navigation modality"
                    )


def isolated_configuration(configuration: bytes, experiment: str, worker: int) -> bytes:
    """Change only the exp_name line, preserving every other configuration byte."""
    lines = configuration.decode("utf-8").splitlines(keepends=True)
    matches = [
        index for index, line in enumerate(lines) if line.startswith("exp_name:")
    ]
    if len(matches) != 1:
        raise ValueError("configuration: expected exactly one exp_name line")
    index = matches[0]
    declared: object = json.loads(lines[index].split(":", 1)[1].strip())
    if declared != experiment:
        raise ValueError("configuration.exp_name: differs from frozen experiment")
    ending = (
        "\r\n"
        if lines[index].endswith("\r\n")
        else "\n" if lines[index].endswith("\n") else ""
    )
    lines[index] = f"exp_name: {json.dumps(f'{experiment}_worker{worker}')}" + ending
    return "".join(lines).encode("utf-8")


def persist(path: Path, payload: bytes) -> None:
    """Publish each immutable artifact atomically; refuse differing existing bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"frozen artifact differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".scope-prepare-", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"frozen artifact differs: {path}") from None
    finally:
        temporary_path.unlink()


def json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def prepare_worker_outputs(assets_root: Path) -> Path:
    """Create exact coverage gates and a storage plan for the resumable controller."""
    assets_root = assets_root.absolute()
    sources = load_sources(assets_root)
    metadata = (sources.output_dir / "subtask_metadata.json").read_bytes()
    validate_metadata(metadata, sources)
    configuration = sources.config_path.read_bytes()
    workers: list[PlanWorker] = []
    stages: list[PlanStage] = []
    for worker in sources.workers:
        cfg_file = (
            assets_root / "manifests" / f"eval_full2669_worker{worker.worker}.yaml"
        )
        isolated_dir = (
            assets_root / "outputs" / f"{sources.experiment}_worker{worker.worker}"
        )
        persist(
            cfg_file,
            isolated_configuration(configuration, sources.experiment, worker.worker),
        )
        persist(isolated_dir / "subtask_metadata.json", metadata)
        all_worker_ids = tuple(
            subtask_id
            for split in _SPLITS
            for subtask_id in worker.expected_ids_by_split[split]
        )
        persist(isolated_dir / "expected_subtask_ids.json", json_bytes(all_worker_ids))
        workers.append(
            PlanWorker(
                worker=worker.worker,
                physical_gpu=worker.physical_gpu,
                cfg_file=str(cfg_file),
                isolated_output_dir=str(isolated_dir),
                start_ratio=worker.start_ratio,
                end_ratio=worker.end_ratio,
            )
        )
        for split in _SPLITS:
            gate_dir = (
                assets_root
                / "outputs"
                / f"{sources.experiment}_gates"
                / f"split{split}_worker{worker.worker}"
            )
            persist(gate_dir / "subtask_metadata.json", metadata)
            persist(
                gate_dir / "expected_subtask_ids.json",
                json_bytes(worker.expected_ids_by_split[split]),
            )
            stages.append(
                PlanStage(
                    split=split,
                    worker=worker.worker,
                    source_output_dir=str(
                        sources.output_dir if split == 1 else isolated_dir
                    ),
                    logger_suffix=worker.logger_suffix_by_split[split],
                    gate_dir=str(gate_dir),
                    expected_subtasks=len(worker.expected_ids_by_split[split]),
                )
            )
    plan = StoragePlan(
        schema_version=1,
        experiment=sources.experiment,
        central_output_dir=str(sources.output_dir),
        workers=workers,
        stages=stages,
    )
    plan_path = assets_root / "manifests/storage_plan.json"
    persist(plan_path, json_bytes(plan))
    return plan_path


def main() -> None:
    """Parse artifact location and delegate to the preparation service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-root", type=Path, default=Path("/home/ysh/Datasets/SCOPE")
    )
    options = parser.parse_args()
    print(prepare_worker_outputs(options.assets_root))


if __name__ == "__main__":
    main()
PY
