"""Tests for the opt-in local processed_v1 dataset path helper."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "datasets" / "processed_v1.sh"
)
_INPUT_FILES = (
    "nr3d/processed_v1/folds/full_test_sample_ids.json",
    "nr3d/processed_v1/folds/strat600_sample_ids.json",
    "open-eqa/processed_v1/open-eqa-v0.json",
    "open-eqa/processed_v1/folds/v5_pilot_fold_300.json",
)


def _create_inputs(root: Path) -> None:
    for benchmark in ("nr3d", "open-eqa"):
        (root / benchmark / "processed_v1" / "scannet").mkdir(parents=True)
    for relative in _INPUT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]\n", encoding="utf-8")


def test_processed_paths_use_explicit_shared_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "shared datasets"
    _create_inputs(root)
    monkeypatch.setenv("AGENTIC3D_DATASETS_ROOT", str(root))
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -e; source "$1"; printf "%s\\n" "$NR3D_DATA_ROOT" '
            '"$NR3D_FULL_SAMPLE_IDS" "$NR3D_PILOT_SAMPLE_IDS" "$NR3D_PACK_NAME" '
            '"$OPENEQA_DATA_ROOT" "$OPENEQA_QUESTIONS" "$OPENEQA_PILOT_QUESTION_IDS"',
            "processed-path-test",
            str(_SCRIPT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [
        str(root / "nr3d/processed_v1/scannet"),
        str(root / _INPUT_FILES[0]),
        str(root / _INPUT_FILES[1]),
        "pack_nr3d_v9_catalog_first",
        str(root / "open-eqa/processed_v1/scannet"),
        str(root / _INPUT_FILES[2]),
        str(root / _INPUT_FILES[3]),
    ]


@pytest.mark.parametrize("missing", _INPUT_FILES)
def test_processed_paths_reject_missing_input_without_partial_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    _create_inputs(tmp_path)
    (tmp_path / missing).unlink()
    monkeypatch.setenv("AGENTIC3D_DATASETS_ROOT", str(tmp_path))
    result = subprocess.run(
        ["bash", str(_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    assert "input file is missing" in result.stderr
    assert str(tmp_path / missing) in result.stderr


def test_processed_paths_reject_absent_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTIC3D_DATASETS_ROOT", str(tmp_path))
    result = subprocess.run(
        ["bash", str(_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    assert "directory is missing" in result.stderr
