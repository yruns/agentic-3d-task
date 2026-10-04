"""Exercise completed-stage SCOPE recovery without starting navigation workers."""

from __future__ import annotations

import json
import os
import pickle
import shlex
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import pytest
from pydantic import BaseModel, ConfigDict

from scripts.ingest_goat_bench_run import ExpectedIds, load_samples

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENT = "REPRO_20261005_SCOPE_Gemini38Flash_full2669"
_PUBLIC_RUN = "REPRO_20261005_SCOPE_Gemini38Flash_public_example_ep0"
_CANONICAL_RUN = "REPRO_20261005_SCOPE_Gemini38Flash_canonical278"


class FrozenTask(TypedDict):
    task_type: Literal["object"]
    answer: str
    scene_id: str
    episode_id: int
    subtask_index: int
    split: int


class FrozenWorker(TypedDict):
    physical_gpu: int
    start_ratio: float
    end_ratio: float
    expected_ids_by_split: dict[str, list[str]]
    logger_suffix_by_split: dict[str, str]


class FrozenManifest(TypedDict):
    experiment: str
    output_dir: str
    config_path: str
    workers: list[FrozenWorker]
    all_subtasks: int


class WorkerStage(BaseModel):
    """Typed storage-plan boundary used by the integration fixture."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    split: int
    worker: int
    source_output_dir: Path
    logger_suffix: str
    gate_dir: Path
    expected_subtasks: int


class FixturePlan(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    stages: tuple[WorkerStage, ...]


@dataclass(frozen=True)
class ControllerFixture:
    """Local artifacts and fake CLI boundaries for the real controller script."""

    assets_root: Path
    controller: Path
    tmux_calls: Path
    environment: dict[str, str]
    stages: tuple[WorkerStage, ...]
    canonical_ids: tuple[str, ...]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_primary_metrics(
    output_dir: Path, suffix: str, subtask_ids: tuple[str, ...]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, value in (("success_by_distance", 1.0), ("spl_by_distance", 0.5)):
        (output_dir / f"{metric}_{suffix}.pkl").write_bytes(
            pickle.dumps(dict.fromkeys(subtask_ids, value))
        )


def _frozen_tasks() -> tuple[list[FrozenWorker], dict[str, FrozenTask]]:
    """Create 278 first-split subtasks and 2,391 later subtasks."""
    workers: list[FrozenWorker] = []
    metadata: dict[str, FrozenTask] = {}
    for worker, (gpu, start, end) in enumerate(((0, 0.0, 0.5), (2, 0.5, 1.0))):
        expected: dict[str, list[str]] = {}
        suffixes: dict[str, str] = {}
        for split in range(1, 11):
            count = 139 if split == 1 else 133 if split <= 7 or worker == 0 else 132
            ids: list[str] = []
            for index in range(count):
                scene_id = f"fixture_scene{worker}"
                subtask_id = f"{scene_id}_{split - 1}_{index}"
                ids.append(subtask_id)
                metadata[subtask_id] = FrozenTask(
                    task_type="object",
                    answer="chair",
                    scene_id=scene_id,
                    episode_id=split - 1,
                    subtask_index=index,
                    split=split,
                )
            expected[str(split)] = ids
            suffixes[str(split)] = f"{start}_{end}_{split}"
        workers.append(
            FrozenWorker(
                physical_gpu=gpu,
                start_ratio=start,
                end_ratio=end,
                expected_ids_by_split=expected,
                logger_suffix_by_split=suffixes,
            )
        )
    assert len(metadata) == 2669
    return workers, metadata


def _fake_commands(fakebin: Path, tmux_calls: Path) -> None:
    fakebin.mkdir()
    tmux = fakebin / "tmux"
    tmux.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(tmux_calls))}\n"
        'if [[ "$1" == new-session ]]; then\n'
        "  printf 'Test forbids starting a worker\\n' >&2\n"
        "  exit 91\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    if sys.platform == "darwin":
        flock = fakebin / "flock"
        flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        flock.chmod(0o755)


@pytest.fixture
def controller_fixture(tmp_path: Path) -> ControllerFixture:
    """Prepare all terminal stages using production storage preparation."""
    assets_root = tmp_path / "assets"
    manifests = assets_root / "manifests"
    scripts = assets_root / "scripts"
    logs = assets_root / "logs"
    full_output = assets_root / "outputs" / _EXPERIMENT
    for directory in (manifests, scripts, logs, full_output):
        directory.mkdir(parents=True, exist_ok=True)
    workers, metadata = _frozen_tasks()
    config_path = manifests / "eval_full2669.yaml"
    config_path.write_text(
        f'exp_name: "{_EXPERIMENT}"\n'
        f'output_parent_dir: "{assets_root / "outputs"}"\n'
        "seed: 3407\n",
        encoding="utf-8",
    )
    manifest = FrozenManifest(
        experiment=_EXPERIMENT,
        output_dir=str(full_output),
        config_path=str(config_path),
        workers=workers,
        all_subtasks=2669,
    )
    _write_json(manifests / "full2669_frozen_manifest.json", manifest)
    _write_json(full_output / "subtask_metadata.json", metadata)
    _write_json(full_output / "expected_subtask_ids.json", tuple(metadata))
    canonical_ids = tuple(
        subtask_id
        for worker in workers
        for subtask_id in worker["expected_ids_by_split"]["1"]
    )
    _write_json(manifests / "canonical278_ids.json", canonical_ids)
    for name in ("full_project_commit.txt", "full_controller_commit.txt"):
        (manifests / name).write_text("0123456789abcdef\n", encoding="utf-8")
    subprocess.run(
        [
            "bash",
            str(_PROJECT_ROOT / "scripts/scope/prepare_worker_outputs.sh"),
            "--assets-root",
            str(assets_root),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    stages = FixturePlan.model_validate_json(
        (manifests / "storage_plan.json").read_text(encoding="utf-8")
    ).stages
    for stage in stages:
        ids = ExpectedIds.model_validate_json(
            (stage.gate_dir / "expected_subtask_ids.json").read_bytes()
        ).root
        _write_primary_metrics(stage.source_output_dir, stage.logger_suffix, ids)
        _write_primary_metrics(full_output, stage.logger_suffix, ids)
        (logs / f"full_split{stage.split}_worker{stage.worker}.exit").write_text(
            "0\n", encoding="utf-8"
        )
    public_output = assets_root / "outputs" / _PUBLIC_RUN
    public_ids = tuple(f"public_scene_0_{index}" for index in range(5))
    _write_primary_metrics(public_output, "0.0_1.0_1", public_ids)
    _write_json(public_output / "expected_subtask_ids.json", public_ids)
    _write_json(
        public_output / "subtask_metadata.json",
        {
            subtask_id: {
                "task_type": "object",
                "scene_id": "public_scene",
                "episode_id": 0,
            }
            for subtask_id in public_ids
        },
    )
    _write_json(
        public_output / "run_manifest.json", {"head_commit_at_launch": "abcdef"}
    )
    (logs / "public_episode.exit").write_text("0\n", encoding="utf-8")
    shutil.copy2(
        _PROJECT_ROOT / "scripts/ingest_goat_bench_run.py",
        scripts / "ingest_goat_bench_run.py",
    )
    controller = scripts / "run_full.sh"
    controller.write_text(
        (_PROJECT_ROOT / "scripts/scope/run_full.sh")
        .read_text(encoding="utf-8")
        .replace(
            "scope_assets=/home/ysh/Datasets/SCOPE",
            f"scope_assets={shlex.quote(str(assets_root))}",
            1,
        )
        .replace(
            "scope_gate_python=/home/ysh/miniconda3/envs/conceptgraph/bin/python",
            f"scope_gate_python={shlex.quote(sys.executable)}",
            1,
        ),
        encoding="utf-8",
    )
    tmux_calls = tmp_path / "tmux_calls.log"
    fakebin = tmp_path / "fakebin"
    _fake_commands(fakebin, tmux_calls)
    environment = dict(os.environ)
    environment["PATH"] = str(fakebin) + os.pathsep + environment["PATH"]
    return ControllerFixture(
        assets_root, controller, tmux_calls, environment, stages, canonical_ids
    )


def _run_controller(fixture: ControllerFixture) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(fixture.controller)],
        env=fixture.environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _assert_no_worker_started(fixture: ControllerFixture) -> None:
    calls = fixture.tmux_calls.read_text() if fixture.tmux_calls.is_file() else ""
    assert "new-session" not in calls


def test_completed_stages_resume_without_polluting_canonical_subset(
    controller_fixture: ControllerFixture,
) -> None:
    fixture = controller_fixture
    for _ in range(2):
        result = _run_controller(fixture)
        assert result.returncode == 0, result.stderr[-2000:]
        assert (fixture.assets_root / "logs/full_run.exit").read_text().strip() == "0"
    _assert_no_worker_started(fixture)
    canonical = fixture.assets_root / "outputs" / _CANONICAL_RUN
    samples = load_samples(canonical)
    assert len(samples) == 278
    assert {sample.question_id for sample in samples} == set(fixture.canonical_ids)
    with sqlite3.connect(fixture.assets_root / "outputs/runs.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone() == (23,)
        assert connection.execute(
            "SELECT n, coverage_complete FROM runs WHERE run_id = ?", (_EXPERIMENT,)
        ).fetchone() == (2669, 1)
        assert connection.execute(
            "SELECT n, coverage_complete FROM runs WHERE run_id = ?", (_CANONICAL_RUN,)
        ).fetchone() == (278, 1)


def test_missing_worker_sample_fails_exact_coverage(
    controller_fixture: ControllerFixture,
) -> None:
    fixture = controller_fixture
    stage = next(
        stage for stage in fixture.stages if stage.split == 2 and stage.worker == 0
    )
    expected = ExpectedIds.model_validate_json(
        (stage.gate_dir / "expected_subtask_ids.json").read_bytes()
    ).root
    _write_primary_metrics(stage.source_output_dir, stage.logger_suffix, expected[:-1])
    result = _run_controller(fixture)
    assert result.returncode != 0
    assert "Result coverage or modality validation failed" in result.stderr
    _assert_no_worker_started(fixture)


def test_nonzero_terminal_worker_is_not_treated_as_success(
    controller_fixture: ControllerFixture,
) -> None:
    fixture = controller_fixture
    (fixture.assets_root / "logs/full_split1_worker0.exit").write_text("1\n")
    result = _run_controller(fixture)
    assert result.returncode != 0
    assert "Failed session" in result.stderr
    _assert_no_worker_started(fixture)


def test_conflicting_immutable_primary_copy_is_rejected(
    controller_fixture: ControllerFixture,
) -> None:
    fixture = controller_fixture
    stage = next(
        stage for stage in fixture.stages if stage.split == 2 and stage.worker == 0
    )
    conflicting = (
        fixture.assets_root
        / "outputs"
        / _EXPERIMENT
        / f"success_by_distance_{stage.logger_suffix}.pkl"
    )
    original = conflicting.read_bytes()
    expected = ExpectedIds.model_validate_json(
        (stage.gate_dir / "expected_subtask_ids.json").read_bytes()
    ).root
    altered = pickle.dumps(dict.fromkeys(expected, 0.0))
    assert altered != original
    conflicting.write_bytes(altered)
    result = _run_controller(fixture)
    assert result.returncode != 0
    assert "Conflicting primary shard" in result.stderr
    assert conflicting.read_bytes() == altered
    _assert_no_worker_started(fixture)
