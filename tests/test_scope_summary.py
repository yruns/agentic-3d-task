"""Require complete, attributable SCOPE fold summaries from local artifacts."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import pytest

from scripts.ingest_goat_bench_run import GoalType
from scripts.scope.summarize_metrics import FoldReport, write_summary_report

_PRIMARY_SUFFIX = "0.0_1.0_1"
_METRIC_NAMES = ("success_by_distance", "spl_by_distance")


@dataclass(frozen=True)
class SampleFixture:
    """An original task and its terminal distance metrics."""

    scene_id: str
    episode_id: int
    subtask_index: int
    task_type: GoalType
    success: float
    spl: float

    @property
    def question_id(self) -> str:
        return f"{self.scene_id}_{self.episode_id}_{self.subtask_index}"


class TaskFactsFixture(TypedDict):
    task_type: GoalType
    answer: str
    scene_id: str
    episode_id: int


class MetadataFixture(TaskFactsFixture, total=False):
    """Include intentionally malformed index values at the JSON boundary."""

    subtask_index: object


_SAMPLES = (
    SampleFixture("scene_a", 0, 0, "object", 1.0, 0.5),
    SampleFixture("scene_a", 0, 1, "description", 0.0, 0.0),
    SampleFixture("scene_b", 1, 0, "image", 1.0, 0.25),
    SampleFixture("scene_b", 1, 1, "object", 1.0, 1.0),
    SampleFixture("scene_c", 2, 0, "description", 1.0, 0.75),
    SampleFixture("scene_c", 2, 1, "image", 0.0, 0.0),
)
_EXPECTED_IDS = tuple(sample.question_id for sample in _SAMPLES)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _task_metadata(
    samples: Sequence[SampleFixture] = _SAMPLES,
) -> dict[str, MetadataFixture]:
    return {
        sample.question_id: MetadataFixture(
            task_type=sample.task_type,
            answer="chair",
            scene_id=sample.scene_id,
            episode_id=sample.episode_id,
            subtask_index=sample.subtask_index,
        )
        for sample in samples
    }


def _write_run(
    output_dir: Path,
    *,
    samples: Sequence[SampleFixture] = _SAMPLES,
    expected_ids: tuple[str, ...] = _EXPECTED_IDS,
) -> None:
    _write_json(
        output_dir / f"success_by_distance_{_PRIMARY_SUFFIX}.json",
        {sample.question_id: sample.success for sample in samples},
    )
    _write_json(
        output_dir / f"spl_by_distance_{_PRIMARY_SUFFIX}.json",
        {sample.question_id: sample.spl for sample in samples},
    )
    _write_json(output_dir / "expected_subtask_ids.json", expected_ids)
    _write_json(output_dir / "subtask_metadata.json", _task_metadata())


def _summarize(output_dir: Path, report_file: Path) -> FoldReport:
    return write_summary_report(
        output_dir,
        expected_subtasks=len(_EXPECTED_IDS),
        report_file=report_file,
    )


@pytest.mark.parametrize("primary_format", ["json", "pickle"])
def test_complete_fold_reports_percentages_and_original_subtask_indices(
    tmp_path: Path, primary_format: Literal["json", "pickle"]
) -> None:
    output_dir = tmp_path / "run"
    report_file = tmp_path / "summary.json"
    _write_run(output_dir)
    if primary_format == "pickle":
        for metric_name in _METRIC_NAMES:
            values = {
                sample.question_id: (
                    sample.success
                    if metric_name == "success_by_distance"
                    else sample.spl
                )
                for sample in _SAMPLES
            }
            (output_dir / f"{metric_name}_{_PRIMARY_SUFFIX}.pkl").write_bytes(
                pickle.dumps(values)
            )
            (output_dir / f"{metric_name}_{_PRIMARY_SUFFIX}.json").unlink()

    report = _summarize(output_dir, report_file)

    assert report.schema_version == 1
    assert report.complete is True
    assert report.expected_subtasks == 6
    assert report.output_dir == output_dir
    assert report.overall.n == 6
    assert report.overall.sr == pytest.approx(200 / 3)
    assert report.overall.spl == pytest.approx(250 / 6)
    expected_modality_metrics: dict[GoalType, tuple[int, float, float]] = {
        "object": (2, 100.0, 75.0),
        "description": (2, 50.0, 37.5),
        "image": (2, 50.0, 12.5),
    }
    assert set(report.per_modality) == set(expected_modality_metrics)
    for modality, (count, sr, spl) in expected_modality_metrics.items():
        metrics = report.per_modality[modality]
        assert (metrics.n, metrics.sr, metrics.spl) == (count, sr, spl)
    assert set(report.per_subtask_index) == {0, 1}
    for index, sr, spl in ((0, 100.0, 50.0), (1, 100 / 3, 100 / 3)):
        metrics = report.per_subtask_index[index]
        assert metrics.subtask_index == index
        assert metrics.ordinal1based == index + 1
        assert metrics.n == 3
        assert metrics.sr == pytest.approx(sr)
        assert metrics.spl == pytest.approx(spl)
    assert FoldReport.model_validate_json(report_file.read_bytes()) == report


@pytest.mark.parametrize(
    "coverage_error",
    ["partial", "unexpected", "count", "missing_expected", "duplicate_expected"],
)
def test_inexact_frozen_coverage_never_creates_report(
    tmp_path: Path,
    coverage_error: Literal[
        "partial", "unexpected", "count", "missing_expected", "duplicate_expected"
    ],
) -> None:
    output_dir = tmp_path / "run"
    report_file = tmp_path / "summary.json"
    _write_run(output_dir)
    expected_subtasks = len(_EXPECTED_IDS)
    if coverage_error == "partial":
        _write_run(output_dir, samples=_SAMPLES[:-1])
    elif coverage_error == "unexpected":
        unexpected = SampleFixture("unexpected_scene", 0, 1, "image", 0.0, 0.0)
        _write_run(output_dir, samples=(*_SAMPLES[:-1], unexpected))
    elif coverage_error == "count":
        expected_subtasks += 1
    elif coverage_error == "missing_expected":
        (output_dir / "expected_subtask_ids.json").unlink()
    else:
        _write_json(
            output_dir / "expected_subtask_ids.json",
            (*_EXPECTED_IDS[:-1], _EXPECTED_IDS[0]),
        )

    with pytest.raises((ValueError, FileNotFoundError)):
        write_summary_report(
            output_dir,
            expected_subtasks=expected_subtasks,
            report_file=report_file,
        )

    assert not report_file.exists()


def test_partial_fold_preserves_an_existing_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    report_file = tmp_path / "summary.json"
    _write_run(output_dir, samples=_SAMPLES[:-1])
    original_report = b"previous validated report\n"
    report_file.write_bytes(original_report)

    with pytest.raises(ValueError):
        _summarize(output_dir, report_file)

    assert report_file.read_bytes() == original_report


@pytest.mark.parametrize("invalid_index", ["missing", True, -1, 1, "0"])
def test_metadata_requires_an_explicit_matching_strict_nonnegative_index(
    tmp_path: Path, invalid_index: object
) -> None:
    output_dir = tmp_path / "run"
    report_file = tmp_path / "summary.json"
    _write_run(output_dir)
    metadata = _task_metadata()
    first_task = metadata[_EXPECTED_IDS[0]]
    if invalid_index == "missing":
        del first_task["subtask_index"]
    else:
        first_task["subtask_index"] = invalid_index
    _write_json(output_dir / "subtask_metadata.json", metadata)

    with pytest.raises(ValueError, match="subtask_index"):
        _summarize(output_dir, report_file)

    assert not report_file.exists()


def test_canonical_subset_accepts_superset_original_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    report_file = tmp_path / "summary.json"
    subset = _SAMPLES[:3]
    _write_run(
        output_dir,
        samples=subset,
        expected_ids=tuple(sample.question_id for sample in subset),
    )

    report = write_summary_report(
        output_dir, expected_subtasks=3, report_file=report_file
    )

    assert report.complete
    assert report.overall.n == 3
    assert report.overall.sr == pytest.approx(200 / 3)
    assert report.overall.spl == 25.0
    assert report.per_subtask_index[0].n == 2
    assert report.per_subtask_index[1].n == 1


def test_plain_aggregate_metrics_cannot_prove_primary_completion(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    report_file = tmp_path / "summary.json"
    _write_run(output_dir)
    for metric_name in _METRIC_NAMES:
        (output_dir / f"{metric_name}_{_PRIMARY_SUFFIX}.json").rename(
            output_dir / f"{metric_name}.json"
        )

    with pytest.raises((ValueError, FileNotFoundError)):
        _summarize(output_dir, report_file)

    assert not report_file.exists()


def test_fingerprints_cover_only_selected_json_primaries_and_frozen_inputs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    report_file = tmp_path / "summary.json"
    _write_run(output_dir)
    for metric_name in _METRIC_NAMES:
        (output_dir / f"{metric_name}_{_PRIMARY_SUFFIX}.pkl").write_bytes(
            b"invalid superseded pickle"
        )
        (output_dir / f"{metric_name}.pkl").write_bytes(b"invalid legacy aggregate")
        (output_dir / f"{metric_name}.json").write_bytes(b"invalid legacy JSON")
    (output_dir / "success_by_task.json").write_bytes(b"invalid grouped aggregate")
    (output_dir / "run_manifest.json").write_bytes(b"not a metric input")

    report = _summarize(output_dir, report_file)

    expected_roles = {
        f"success_by_distance_{_PRIMARY_SUFFIX}.json": "success_by_distance",
        f"spl_by_distance_{_PRIMARY_SUFFIX}.json": "spl_by_distance",
        "expected_subtask_ids.json": "expected_ids",
        "subtask_metadata.json": "task_metadata",
    }
    assert len(report.inputs) == len(expected_roles)
    assert {str(fingerprint.path) for fingerprint in report.inputs} == set(
        expected_roles
    )
    for fingerprint in report.inputs:
        filename = str(fingerprint.path)
        content = (output_dir / filename).read_bytes()
        assert fingerprint.role == expected_roles[filename]
        assert fingerprint.size_bytes == len(content)
        assert fingerprint.sha256 == hashlib.sha256(content).hexdigest()
    assert report.overall.spl == pytest.approx(250 / 6)


@pytest.mark.parametrize("metric_name", _METRIC_NAMES)
def test_conflicting_primary_shards_never_create_report(
    tmp_path: Path, metric_name: str
) -> None:
    output_dir = tmp_path / "run"
    report_file = tmp_path / "summary.json"
    _write_run(output_dir)
    _write_json(output_dir / f"{metric_name}_0.5_1.0_1.json", {_EXPECTED_IDS[0]: 0.0})
    companion_metric = (
        "spl_by_distance"
        if metric_name == "success_by_distance"
        else "success_by_distance"
    )
    companion_value = (
        _SAMPLES[0].spl
        if companion_metric == "spl_by_distance"
        else _SAMPLES[0].success
    )
    _write_json(
        output_dir / f"{companion_metric}_0.5_1.0_1.json",
        {_EXPECTED_IDS[0]: companion_value},
    )

    with pytest.raises(ValueError, match="conflicting"):
        _summarize(output_dir, report_file)

    assert not report_file.exists()


@pytest.mark.parametrize(
    ("metric_name", "invalid_value"),
    [("success_by_distance", 0.5), ("spl_by_distance", 1.1)],
)
def test_ingester_metric_validation_precedes_report_write(
    tmp_path: Path, metric_name: str, invalid_value: float
) -> None:
    output_dir = tmp_path / "run"
    report_file = tmp_path / "summary.json"
    _write_run(output_dir)
    values = {
        sample.question_id: (
            sample.success if metric_name == "success_by_distance" else sample.spl
        )
        for sample in _SAMPLES
    }
    values[_EXPECTED_IDS[0]] = invalid_value
    _write_json(output_dir / f"{metric_name}_{_PRIMARY_SUFFIX}.json", values)

    with pytest.raises(ValueError):
        _summarize(output_dir, report_file)

    assert not report_file.exists()


def test_mismatched_success_and_spl_primary_suffixes_never_create_report(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    report_file = tmp_path / "summary.json"
    _write_run(output_dir)
    (output_dir / f"spl_by_distance_{_PRIMARY_SUFFIX}.json").rename(
        output_dir / "spl_by_distance_0.5_1.0_1.json"
    )

    with pytest.raises(ValueError, match="suffix"):
        _summarize(output_dir, report_file)

    assert not report_file.exists()


@pytest.mark.parametrize(
    ("report_name", "existing_report"),
    [
        ("success_by_distance.json", True),
        ("spl_by_distance.json", True),
        ("success_by_distance_summary.json", False),
        ("spl_by_distance_summary.json", False),
        ("success_by_distancesummary.txt", False),
    ],
)
def test_report_cannot_overwrite_or_create_reserved_metric_namespace(
    tmp_path: Path, report_name: str, existing_report: bool
) -> None:
    output_dir = tmp_path / "run"
    _write_run(output_dir)
    report_file = output_dir / report_name
    original_content = b"legacy derived aggregate must remain intact\n"
    if existing_report:
        report_file.write_bytes(original_content)

    with pytest.raises(ValueError, match="namespace"):
        _summarize(output_dir, report_file)

    if existing_report:
        assert report_file.read_bytes() == original_content
    else:
        assert not report_file.exists()
