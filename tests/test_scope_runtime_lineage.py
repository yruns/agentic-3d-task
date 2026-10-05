"""Attribute saved metrics to actual runtime epochs without rewriting metrics."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.scope.runtime_lineage import (
    LineagePlan,
    PreservedEpoch,
    RuntimeSource,
    write_lineage_report,
)

_BASE = "1" * 40
_VISUALIZATION = "2" * 40
_PARALLEL = "3" * 40
_IDS = ("scene_0_0", "scene_0_1", "scene_1_0")


def _source(commit: str) -> RuntimeSource:
    return RuntimeSource(runtime_commit_sha=commit, evidence_records=("launch.json",))


def _plan() -> LineagePlan:
    return LineagePlan(
        expected_subtask_ids=_IDS,
        active_source=_source(_PARALLEL),
        preserved_epochs=(
            PreservedEpoch(source=_source(_BASE), completed_subtask_ids=(_IDS[0],)),
            PreservedEpoch(
                source=_source(_VISUALIZATION), completed_subtask_ids=(_IDS[1],)
            ),
        ),
    )


def _write_run(root: Path, observed: tuple[str, ...] = _IDS) -> tuple[Path, Path]:
    output = root / "output with spaces"
    output.mkdir()
    for metric, value in (("success_by_distance", 1.0), ("spl_by_distance", 0.5)):
        (output / f"{metric}_0.0_1.0_1.json").write_text(
            json.dumps(dict.fromkeys(observed, value)), encoding="utf-8"
        )
    (output / "expected_subtask_ids.json").write_text(
        json.dumps(_IDS), encoding="utf-8"
    )
    source_map = root / "lineage.json"
    source_map.write_text(_plan().model_dump_json(), encoding="utf-8")
    return output, source_map


def test_mixed_saved_epochs_leave_primary_metrics_unchanged(tmp_path: Path) -> None:
    output, source_map = _write_run(tmp_path)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    report = write_lineage_report(output_dir=output, source_map=source_map)
    assert report.runtime_commit_sha == f"mixed:{_BASE}+{_VISUALIZATION}+{_PARALLEL}"
    assert report.observed_subtasks == 3
    assert report.expected_subtasks == 3
    assert [
        (row.question_id, row.source.runtime_commit_sha) for row in report.samples
    ] == [
        (_IDS[0], _BASE),
        (_IDS[1], _VISUALIZATION),
        (_IDS[2], _PARALLEL),
    ]
    assert [row.disposition for row in report.samples] == [
        "preserved",
        "preserved",
        "active_attempt",
    ]
    assert all(row.source.project_launch_head is None for row in report.samples)
    assert {
        path.name: path.read_bytes() for path in output.iterdir() if path.name in before
    } == before
    assert (output / "runtime_lineage.json").is_file()


def test_unobserved_epochs_do_not_enter_runtime_label(tmp_path: Path) -> None:
    output, source_map = _write_run(tmp_path, observed=(_IDS[0],))
    report = write_lineage_report(output_dir=output, source_map=source_map)
    assert report.runtime_commit_sha == _BASE
    assert report.observed_subtasks == 1


@pytest.mark.parametrize("invalid_ids", [("unknown_0_0",), ()])
def test_unknown_or_empty_observation_fails_before_report(
    tmp_path: Path, invalid_ids: tuple[str, ...]
) -> None:
    output, source_map = _write_run(tmp_path, observed=invalid_ids)
    with pytest.raises(ValueError):
        write_lineage_report(output_dir=output, source_map=source_map)
    assert not (output / "runtime_lineage.json").exists()


def test_observed_id_outside_selected_fold_fails(tmp_path: Path) -> None:
    output, source_map = _write_run(tmp_path)
    (output / "expected_subtask_ids.json").write_text(
        json.dumps((_IDS[0],)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="selected"):
        write_lineage_report(output_dir=output, source_map=source_map)
    assert not (output / "runtime_lineage.json").exists()


def test_unpaired_primary_fails_before_report(tmp_path: Path) -> None:
    output, source_map = _write_run(tmp_path)
    (output / "spl_by_distance_0.0_1.0_1.json").unlink()
    with pytest.raises(FileNotFoundError):
        write_lineage_report(output_dir=output, source_map=source_map)
    assert not (output / "runtime_lineage.json").exists()


@pytest.mark.parametrize("preserved_id", [_IDS[0], "unknown_0_0"])
def test_conflicting_or_unknown_preserved_identity_is_rejected(
    preserved_id: str,
) -> None:
    with pytest.raises(ValidationError):
        LineagePlan(
            expected_subtask_ids=_IDS,
            active_source=_source(_PARALLEL),
            preserved_epochs=(
                PreservedEpoch(source=_source(_BASE), completed_subtask_ids=(_IDS[0],)),
                PreservedEpoch(
                    source=_source(_VISUALIZATION),
                    completed_subtask_ids=(preserved_id,),
                ),
            ),
        )


def test_direct_cli_prints_only_the_observed_runtime_label(tmp_path: Path) -> None:
    output, source_map = _write_run(tmp_path, observed=(_IDS[0],))
    project = Path(__file__).parents[1]
    scripts = tmp_path / "deployed scripts"
    scripts.mkdir()
    for source in (
        project / "scripts/ingest_goat_bench_run.py",
        project / "scripts/scope/runtime_lineage.py",
    ):
        shutil.copy2(source, scripts / source.name)
    result = subprocess.run(
        [
            sys.executable,
            str(scripts / "runtime_lineage.py"),
            "--output-dir",
            str(output),
            "--source-map",
            str(source_map),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == _BASE + "\n"


@pytest.mark.parametrize("commit", ["not-a-commit", "F" * 40])
def test_runtime_source_validates_commit_hash(commit: str) -> None:
    with pytest.raises(ValidationError):
        _source(commit)
