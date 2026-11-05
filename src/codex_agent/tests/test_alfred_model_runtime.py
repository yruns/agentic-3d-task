"""Cooperative GPU reservation never requires a GPU or another user's process."""

import subprocess
from pathlib import Path
from typing import cast

import pytest

from codex_agent.alfred.model_runtime import gpu_file_lock, physical_gpu_uuid

GPU_ZERO = "GPU-00000000-1111-2222-3333-444444444444"
GPU_FOUR = "GPU-44444444-1111-2222-3333-444444444444"


def test_gpu_file_lock_excludes_other_owners_and_releases_on_error(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "gpu.lock"
    with pytest.raises(RuntimeError, match="failure"):
        with gpu_file_lock(lock, timeout_seconds=0.5) as waited:
            assert waited >= 0
            with pytest.raises(TimeoutError, match="GPU lock"):
                with gpu_file_lock(lock, timeout_seconds=0.05):
                    pytest.fail("second owner acquired held lock")
            raise RuntimeError("inference failure")
    assert lock.is_file()
    with gpu_file_lock(lock, timeout_seconds=0.1):
        pass


def test_physical_gpu_uuid_ignores_inherited_cuda_remapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def query(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "check": True,
            "timeout": 10,
        }
        return subprocess.CompletedProcess(
            command, 0, f"4, {GPU_FOUR}\n0, {GPU_ZERO}\n", ""
        )

    monkeypatch.setattr(subprocess, "run", query)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    assert physical_gpu_uuid(4) == GPU_FOUR
    assert calls == [
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    ]


@pytest.mark.parametrize("index", [-1, 1, True, 4.0, "4"])
def test_physical_gpu_rejects_forbidden_or_noninteger_index_without_query(
    index: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_query(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid physical GPU index reached nvidia-smi")

    monkeypatch.setattr(subprocess, "run", unexpected_query)
    with pytest.raises(ValueError, match="physical_gpu_index"):
        physical_gpu_uuid(cast(int, index))


@pytest.mark.parametrize(
    "output",
    [
        f"0, {GPU_ZERO}\n",
        "4, not-a-gpu-uuid\n",
        f"4, {GPU_FOUR}\n4, {GPU_ZERO}\n",
        f"0, {GPU_ZERO}\n4, {GPU_ZERO}\n",
        "unparseable response",
    ],
)
def test_physical_gpu_query_requires_unambiguous_real_uuid(
    output: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def query(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(subprocess, "run", query)
    with pytest.raises(ValueError, match="nvidia-smi"):
        physical_gpu_uuid(4)


def test_physical_gpu_query_failure_cannot_fall_back_to_logical_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def query(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 10)

    monkeypatch.setattr(subprocess, "run", query)
    with pytest.raises(RuntimeError, match="physical_gpu_index=4"):
        physical_gpu_uuid(4)
