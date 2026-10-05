"""Exercise completed-stage SCOPE recovery without starting navigation workers."""

from __future__ import annotations

import hashlib
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
from scripts.scope.runtime_lineage import (
    LineagePlan,
    LineageReport,
    PreservedEpoch,
    RuntimeSource,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENT = "REPRO_20261005_SCOPE_Gemini38Flash_full2669"
_PUBLIC_RUN = "REPRO_20261005_SCOPE_Gemini38Flash_public_example_ep0"
_CANONICAL_RUN = "REPRO_20261005_SCOPE_Gemini38Flash_canonical278"
_ORIGINAL_RUNTIME = "46f1e1f78077219c30e35440540e4ef22995194c"
_EMPTY_PREFILTER_RUNTIME = "bfffc2be995a7223c8a0e9c86c18064dbe878838"
_PARALLEL_RUNTIME = "606d29ca19901882015cdd432017b4a867cef978"
_CONTROLLER_LAUNCH_HEAD = "4" * 40
_MIXED_RUNTIME = "mixed:" + "+".join(
    sorted((_ORIGINAL_RUNTIME, _EMPTY_PREFILTER_RUNTIME, _PARALLEL_RUNTIME))
)


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
    physical_gpu: int
    cfg_file: Path
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


def _write_lineage_plan(manifests: Path, expected_ids: tuple[str, ...]) -> None:
    """Freeze two preserved source epochs plus one active attempt source."""
    source_map = LineagePlan(
        expected_subtask_ids=expected_ids,
        active_source=RuntimeSource(
            runtime_commit_sha=_PARALLEL_RUNTIME,
            project_launch_head="3" * 40,
            evidence_records=("fixture_parallel_recovery.json",),
        ),
        preserved_epochs=(
            PreservedEpoch(
                source=RuntimeSource(
                    runtime_commit_sha=_ORIGINAL_RUNTIME,
                    project_launch_head="1" * 40,
                    evidence_records=("fixture_original_bootstrap.json",),
                ),
                completed_subtask_ids=expected_ids[:2],
            ),
            PreservedEpoch(
                source=RuntimeSource(
                    runtime_commit_sha=_EMPTY_PREFILTER_RUNTIME,
                    project_launch_head=None,
                    evidence_records=("fixture_empty_prefilter_recovery.json",),
                ),
                completed_subtask_ids=expected_ids[2:4],
            ),
        ),
    )
    (manifests / "frozen_result_lineage.json").write_text(
        source_map.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


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
    _write_lineage_plan(manifests, tuple(metadata))
    (manifests / "lineage_controller_commit.txt").write_text(
        _CONTROLLER_LAUNCH_HEAD + "\n", encoding="utf-8"
    )
    canonical_ids = tuple(
        subtask_id
        for worker in workers
        for subtask_id in worker["expected_ids_by_split"]["1"]
    )
    _write_json(manifests / "canonical278_ids.json", canonical_ids)
    for name in ("full_project_commit.txt", "full_controller_commit.txt"):
        (manifests / name).write_text("0123456789abcdef\n", encoding="utf-8")
    for preparation in ("prepare_worker_outputs.sh", "prepare_parallel_outputs.sh"):
        subprocess.run(
            [
                "bash",
                str(_PROJECT_ROOT / "scripts/scope" / preparation),
                "--assets-root",
                str(assets_root),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    stages = FixturePlan.model_validate_json(
        (manifests / "storage_plan_parallel_v2.json").read_text(encoding="utf-8")
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
    shutil.copy2(
        _PROJECT_ROOT / "scripts/scope/runtime_lineage.py",
        scripts / "runtime_lineage.py",
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


def _assert_sample_lineage(
    fixture: ControllerFixture, output_dir: Path, expected_ids: tuple[str, ...]
) -> None:
    source_path = fixture.assets_root / "manifests/frozen_result_lineage.json"
    source_bytes = source_path.read_bytes()
    plan = LineagePlan.model_validate_json(source_bytes)
    report = LineageReport.model_validate_json(
        (output_dir / "runtime_lineage.json").read_bytes()
    )
    preserved = {
        subtask_id: epoch.source
        for epoch in plan.preserved_epochs
        for subtask_id in epoch.completed_subtask_ids
    }
    observed = {sample.question_id: sample for sample in report.samples}
    assert set(observed) == set(expected_ids)
    assert report.observed_subtasks == len(expected_ids)
    assert report.source_map_sha256 == hashlib.sha256(source_bytes).hexdigest()
    for subtask_id in expected_ids:
        expected_source = preserved.get(subtask_id, plan.active_source)
        assert observed[subtask_id].source == expected_source
        assert observed[subtask_id].disposition == (
            "preserved" if subtask_id in preserved else "active_attempt"
        )


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
    full_output = fixture.assets_root / "outputs" / _EXPERIMENT
    full_ids = ExpectedIds.model_validate_json(
        (full_output / "expected_subtask_ids.json").read_bytes()
    ).root
    _assert_sample_lineage(fixture, full_output, full_ids)
    _assert_sample_lineage(fixture, canonical, fixture.canonical_ids)
    for stage in fixture.stages:
        expected_ids = ExpectedIds.model_validate_json(
            (stage.gate_dir / "expected_subtask_ids.json").read_bytes()
        ).root
        _assert_sample_lineage(fixture, stage.gate_dir, expected_ids)
    with sqlite3.connect(fixture.assets_root / "outputs/runs.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone() == (23,)
        assert connection.execute(
            "SELECT n, coverage_complete FROM runs WHERE run_id = ?", (_EXPERIMENT,)
        ).fetchone() == (2669, 1)
        assert connection.execute(
            "SELECT n, coverage_complete FROM runs WHERE run_id = ?", (_CANONICAL_RUN,)
        ).fetchone() == (278, 1)
        for run_id in (_EXPERIMENT, _CANONICAL_RUN):
            assert connection.execute(
                "SELECT commit_sha, runtime_commit_sha FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone() == (_CONTROLLER_LAUNCH_HEAD, _MIXED_RUNTIME)
        for stage in fixture.stages:
            expected_runtime = (
                _MIXED_RUNTIME
                if (stage.split, stage.worker) == (1, 0)
                else _PARALLEL_RUNTIME
            )
            run_id = (
                "REPRO_20261005_SCOPE_Gemini38Flash_"
                f"split{stage.split}_worker{stage.worker}"
            )
            assert connection.execute(
                "SELECT commit_sha, runtime_commit_sha FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone() == (_CONTROLLER_LAUNCH_HEAD, expected_runtime)
        assert connection.execute(
            "SELECT commit_sha, runtime_commit_sha FROM runs WHERE run_id = ?",
            (_PUBLIC_RUN,),
        ).fetchone() == ("abcdef", _ORIGINAL_RUNTIME)
    assert not (
        fixture.assets_root / "outputs" / _PUBLIC_RUN / "runtime_lineage.json"
    ).exists()


def test_lineage_helper_failure_is_not_swallowed(
    controller_fixture: ControllerFixture,
) -> None:
    fixture = controller_fixture
    (fixture.assets_root / "scripts/runtime_lineage.py").write_text(
        "raise SystemExit(71)\n", encoding="utf-8"
    )

    result = _run_controller(fixture)

    assert result.returncode != 0
    assert "Runtime lineage validation failed" in result.stderr
    assert (fixture.assets_root / "logs/full_run.exit").read_text().strip() != "0"
    _assert_no_worker_started(fixture)


def test_observed_id_absent_from_frozen_lineage_fails_before_stage_ingest(
    controller_fixture: ControllerFixture,
) -> None:
    fixture = controller_fixture
    source_path = fixture.assets_root / "manifests/frozen_result_lineage.json"
    plan = LineagePlan.model_validate_json(source_path.read_bytes())
    missing_id = plan.expected_subtask_ids[4]
    changed = LineagePlan(
        expected_subtask_ids=tuple(
            subtask_id
            for subtask_id in plan.expected_subtask_ids
            if subtask_id != missing_id
        ),
        active_source=plan.active_source,
        preserved_epochs=plan.preserved_epochs,
    )
    source_path.write_text(changed.model_dump_json() + "\n", encoding="utf-8")

    result = _run_controller(fixture)

    assert result.returncode != 0
    _assert_no_worker_started(fixture)
    with sqlite3.connect(fixture.assets_root / "outputs/runs.sqlite") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM runs WHERE run_id = ?",
            ("REPRO_20261005_SCOPE_Gemini38Flash_split1_worker0",),
        ).fetchone() == (0,)


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


def test_future_batches_launch_six_disjoint_gpu_workers_before_collection(
    controller_fixture: ControllerFixture,
) -> None:
    """Exercise tmux dispatch with synthetic terminal workers, without GPUs."""
    fixture = controller_fixture
    for stage in fixture.stages:
        if stage.split != 1:
            (
                fixture.assets_root
                / f"logs/full_split{stage.split}_worker{stage.worker}.exit"
            ).unlink()
    fakebin = Path(fixture.environment["PATH"].split(os.pathsep)[0])
    (fakebin / "tmux").write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(fixture.tmux_calls))}\n"
        'if [[ "$1" == new-session && "$*" =~ scope-s([0-9]+)-w([01])- ]]; then\n'
        f"  printf '0\\n' > {shlex.quote(str(fixture.assets_root / 'logs'))}"
        "/full_split${BASH_REMATCH[1]}_worker${BASH_REMATCH[2]}.exit\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    result = _run_controller(fixture)
    assert result.returncode == 0, result.stderr[-2000:]
    events = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(("Started worker:", "Finished split"))
    ]
    expected_events = ["Finished split 1"]
    for first_split in (2, 5, 8):
        batch = tuple(range(first_split, first_split + 3))
        stages = [stage for stage in fixture.stages if stage.split in batch]
        assert {stage.physical_gpu for stage in stages} == {0, 2, 3, 4, 5, 6}
        assert len({stage.source_output_dir for stage in stages}) == 6
        assert len({stage.cfg_file for stage in stages}) == 6
        for split in batch:
            for worker in (0, 1):
                expected_events.append(
                    f"Started worker: scope-s{split}-w{worker}-20261005"
                )
        expected_events.extend(f"Finished split {split}" for split in batch)
    assert events == expected_events
    launches = [
        line
        for line in fixture.tmux_calls.read_text().splitlines()
        if line.startswith("new-session")
    ]
    assert len(launches) == 18
    assert not any("scope-s1-w" in line for line in launches)
