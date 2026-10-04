"""Regression checks for trustworthy GOAT-Bench result ingestion."""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from scripts.ingest_goat_bench_run import (
    RunMetadata,
    TaskMetadata,
    ingest_run,
    load_samples,
)

_FIRST_ID = "00861-GLAQ4DNUx5U_0_0"
_SECOND_ID = "00861-GLAQ4DNUx5U_0_1"
_THIRD_ID = "00861-GLAQ4DNUx5U_0_2"


def _write_run(
    output_dir: Path,
    success: dict[str, float],
    spl: dict[str, float],
    *,
    expected_ids: tuple[str, ...] = (_FIRST_ID, _SECOND_ID, _THIRD_ID),
    episode_id: str | int = "0",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "success_by_distance.json").write_text(json.dumps(success))
    (output_dir / "spl_by_distance.json").write_text(json.dumps(spl))
    (output_dir / "expected_subtask_ids.json").write_text(json.dumps(expected_ids))
    (output_dir / "subtask_metadata.json").write_text(
        json.dumps(
            {
                subtask_id: {
                    "task_type": "object" if index == 0 else "description",
                    "answer": "chair",
                    "scene_id": "00861-GLAQ4DNUx5U",
                    "episode_id": episode_id,
                }
                for index, subtask_id in enumerate(expected_ids)
            }
        )
    )


def _metadata(expected_subtasks: int = 3) -> RunMetadata:
    return RunMetadata(
        run_id="scope-test",
        branch="codex/scope",
        commit_sha="01234567",
        runtime_commit_sha="46f1e1f7",
        model="test-model",
        expected_subtasks=expected_subtasks,
    )


@pytest.mark.parametrize("bad_spl", [-0.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_spl_before_database_write(
    tmp_path: Path, bad_spl: float
) -> None:
    output_dir = tmp_path / "run"
    _write_run(output_dir, {_FIRST_ID: 1.0}, {_FIRST_ID: bad_spl})
    db_path = tmp_path / "runs.sqlite"
    with pytest.raises(ValueError):
        ingest_run(output_dir=output_dir, db_path=db_path, metadata=_metadata())
    assert not db_path.exists()


def test_rejects_nonbinary_success_and_failed_subtask_with_positive_spl(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    _write_run(output_dir, {_FIRST_ID: 0.5}, {_FIRST_ID: 0.2})
    with pytest.raises(ValueError, match="binary"):
        load_samples(output_dir)
    _write_run(output_dir, {_FIRST_ID: 0.0}, {_FIRST_ID: 0.2})
    with pytest.raises(ValueError, match="failed"):
        load_samples(output_dir)


def test_partial_counts_do_not_invent_results_or_calls(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    _write_run(
        output_dir,
        {_FIRST_ID: 1.0, _SECOND_ID: 0.0},
        {_FIRST_ID: 0.5, _SECOND_ID: 0.0},
    )
    db_path = tmp_path / "runs.sqlite"
    summary = ingest_run(output_dir=output_dir, db_path=db_path, metadata=_metadata())
    assert summary.n == 2
    assert summary.expected_subtasks == 3
    assert summary.sr == 50.0
    assert summary.spl == 25.0
    assert summary.completion_rate == pytest.approx(2 / 3)
    assert not summary.complete
    assert summary.expected_ids_verified
    assert summary.modality_complete
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM samples").fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM tool_calls").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM llm_calls").fetchone() == (0,)
        assert connection.execute(
            "SELECT tool_count, input_tokens FROM samples LIMIT 1"
        ).fetchone() == (None, None)


def test_ingest_is_idempotent_and_replaces_stale_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    _write_run(
        output_dir,
        {_FIRST_ID: 1.0, _SECOND_ID: 0.0},
        {_FIRST_ID: 0.5, _SECOND_ID: 0.0},
    )
    db_path = tmp_path / "runs.sqlite"
    for _ in range(2):
        ingest_run(output_dir=output_dir, db_path=db_path, metadata=_metadata())
    _write_run(output_dir, {_FIRST_ID: 0.0}, {_FIRST_ID: 0.0})
    summary = ingest_run(output_dir=output_dir, db_path=db_path, metadata=_metadata())
    assert summary.n == 1
    assert summary.sr == 0.0
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM samples").fetchone() == (1,)


def test_rejects_unexpected_ids_even_when_counts_match(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    _write_run(output_dir, {"wrong_0_0": 1.0}, {"wrong_0_0": 0.5})
    with pytest.raises(ValueError, match="unexpected"):
        ingest_run(
            output_dir=output_dir,
            db_path=tmp_path / "runs.sqlite",
            metadata=_metadata(),
        )


def test_missing_modality_is_explicit_and_not_guessed_from_grouped_metrics(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    _write_run(output_dir, {_FIRST_ID: 1.0}, {_FIRST_ID: 0.5})
    (output_dir / "subtask_metadata.json").unlink()
    (output_dir / "success_by_task.json").write_text('{"image":[1.0]}')
    summary = ingest_run(
        output_dir=output_dir,
        db_path=tmp_path / "runs.sqlite",
        metadata=_metadata(),
    )
    assert not summary.modality_complete
    assert summary.per_modality["unknown"].n == 1


def test_restricted_pickle_supports_real_numpy_scalar_metrics(tmp_path: Path) -> None:
    with (tmp_path / "success_by_distance_0.0_1.0_1.pkl").open("wb") as stream:
        pickle.dump({_FIRST_ID: 1.0}, stream)
    with (tmp_path / "spl_by_distance_0.0_1.0_1.pkl").open("wb") as stream:
        pickle.dump({_FIRST_ID: np.float64(0.25)}, stream)
    samples = load_samples(tmp_path)
    assert samples[0].success == 1.0
    assert samples[0].spl == 0.25


def test_restricted_pickle_rejects_executable_globals(tmp_path: Path) -> None:
    path = tmp_path / "success_by_distance.pkl"
    path.write_bytes(b"cbuiltins\neval\n(S'1+1'\ntR.")
    (tmp_path / "spl_by_distance.json").write_text("{}")
    with pytest.raises(ValueError, match="forbidden"):
        load_samples(tmp_path)


def test_empty_artifacts_report_null_metrics(tmp_path: Path) -> None:
    _write_run(tmp_path, {}, {})
    summary = ingest_run(
        output_dir=tmp_path,
        db_path=tmp_path / "runs.sqlite",
        metadata=_metadata(),
    )
    assert summary.n == 0
    assert summary.sr is None and summary.spl is None
    assert not summary.complete


def test_rejects_misaligned_metric_keys(tmp_path: Path) -> None:
    _write_run(tmp_path, {_FIRST_ID: 1.0}, {_SECOND_ID: 0.5})
    with pytest.raises(ValueError, match="same subtask IDs"):
        load_samples(tmp_path)


def test_complete_run_requires_expected_count_and_ids(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        {_FIRST_ID: 1.0, _SECOND_ID: 0.0, _THIRD_ID: 1.0},
        {_FIRST_ID: 0.5, _SECOND_ID: 0.0, _THIRD_ID: 0.25},
    )
    summary = ingest_run(
        output_dir=tmp_path,
        db_path=tmp_path / "runs.sqlite",
        metadata=_metadata(),
    )
    assert summary.complete and summary.expected_ids_verified
    assert summary.sr == pytest.approx(200 / 3)
    assert summary.spl == 25.0
    assert summary.completion_rate == 1.0


def test_rejects_expected_count_mismatch_before_write(tmp_path: Path) -> None:
    _write_run(tmp_path, {_FIRST_ID: 1.0}, {_FIRST_ID: 0.5})
    with pytest.raises(ValueError, match="expected ID count"):
        ingest_run(
            output_dir=tmp_path,
            db_path=tmp_path / "runs.sqlite",
            metadata=_metadata(expected_subtasks=2),
        )


def test_json_metrics_take_precedence_over_same_named_pickle(tmp_path: Path) -> None:
    _write_run(tmp_path, {_FIRST_ID: 1.0}, {_FIRST_ID: 0.5})
    (tmp_path / "success_by_distance.pkl").write_bytes(b"cbuiltins\neval\n.")
    assert load_samples(tmp_path)[0].success == 1.0


def test_rejects_conflicting_shards(tmp_path: Path) -> None:
    _write_run(tmp_path, {_FIRST_ID: 1.0}, {_FIRST_ID: 0.5})
    (tmp_path / "success_by_distance.json").rename(
        tmp_path / "success_by_distance_0.0_0.5_1.json"
    )
    (tmp_path / "success_by_distance_0.0_1.0_1.json").write_text(
        json.dumps({_FIRST_ID: 0.0})
    )
    with pytest.raises(ValueError, match="conflicting"):
        load_samples(tmp_path)


@pytest.mark.parametrize(
    ("episode_id", "expected"),
    [(0, "0"), (27, "27"), ("0", "0"), ("27", "27"), ("", "")],
)
def test_normalizes_valid_episode_ids(episode_id: object, expected: str) -> None:
    metadata = TaskMetadata.model_validate(
        {"task_type": "object", "episode_id": episode_id}
    )
    assert metadata.episode_id == expected


def test_absent_episode_id_remains_empty() -> None:
    assert TaskMetadata.model_validate({"task_type": "object"}).episode_id == ""


@pytest.mark.parametrize(
    "episode_id", [True, False, -1, "-1", None, 1.5, {}, [], "abc", " 0"]
)
def test_rejects_invalid_episode_ids(episode_id: object) -> None:
    with pytest.raises(ValueError, match="episode_id"):
        TaskMetadata.model_validate({"task_type": "object", "episode_id": episode_id})


def test_ingests_integer_episode_metadata_from_frozen_full_run(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        {_FIRST_ID: 1.0},
        {_FIRST_ID: 0.5},
        expected_ids=(_FIRST_ID,),
        episode_id=0,
    )
    db_path = tmp_path / "runs.sqlite"
    summary = ingest_run(
        output_dir=tmp_path, db_path=db_path, metadata=_metadata(expected_subtasks=1)
    )
    assert (
        summary.complete and summary.expected_ids_verified and summary.modality_complete
    )
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT episode_id FROM samples").fetchone() == ("0",)


@pytest.mark.parametrize(
    "aggregate_content",
    [pickle.dumps({"wrong_0_0": 0.0}), b"\x80\x04}"],
    ids=("stale", "truncated"),
)
def test_primary_shards_ignore_stale_or_truncated_aggregates(
    tmp_path: Path, aggregate_content: bytes
) -> None:
    _write_run(tmp_path, {_FIRST_ID: 1.0}, {_FIRST_ID: 0.5})
    for name in ("success_by_distance", "spl_by_distance"):
        (tmp_path / f"{name}.json").rename(tmp_path / f"{name}_0.0_0.5_1.json")
        (tmp_path / f"{name}.pkl").write_bytes(aggregate_content)
    samples = load_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].question_id == _FIRST_ID
    assert samples[0].success == 1.0 and samples[0].spl == 0.5


def test_primary_shard_json_precedes_same_named_pickle(tmp_path: Path) -> None:
    _write_run(tmp_path, {_FIRST_ID: 1.0}, {_FIRST_ID: 0.5})
    for name in ("success_by_distance", "spl_by_distance"):
        (tmp_path / f"{name}.json").rename(tmp_path / f"{name}_0.0_0.5_1.json")
        (tmp_path / f"{name}_0.0_0.5_1.pkl").write_bytes(b"\x80\x04}")
    assert load_samples(tmp_path)[0].spl == 0.5
