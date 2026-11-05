"""Lazy framework boundary for bounded sequential model residency."""

from __future__ import annotations

import importlib
import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def physical_gpu_uuid(index: int) -> str:
    """Resolve an explicit physical index without trusting CUDA's logical order.

    Physical GPU 1 is prohibited. A UUID pins the selected board even when the
    caller inherited a different CUDA_VISIBLE_DEVICES or CUDA_DEVICE_ORDER.
    This read-only query does not initialize CUDA or load any model.
    """
    if type(index) is not int or index < 0 or index == 1:
        raise ValueError(
            f"physical_gpu_index={index!r} must be a nonnegative integer other than 1"
        )
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"nvidia-smi failed while resolving physical_gpu_index={index}"
        ) from exc
    devices: dict[int, str] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if (
            len(fields) != 2
            or not fields[0].isascii()
            or not fields[0].isdigit()
            or re.fullmatch(
                r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                fields[1],
            )
            is None
        ):
            raise ValueError(f"nvidia-smi returned malformed index/UUID row={line!r}")
        physical_index, uuid = int(fields[0]), fields[1]
        if physical_index in devices or uuid.lower() in {
            value.lower() for value in devices.values()
        }:
            raise ValueError("nvidia-smi returned duplicate physical indices or UUIDs")
        devices[physical_index] = uuid
    if index not in devices:
        raise ValueError(
            f"nvidia-smi did not report physical_gpu_index={index}; available={sorted(devices)}"
        )
    return devices[index]


@contextmanager
def active_model(model: Any, device: str) -> Iterator[Any]:
    """Keep the neural model resident on the configured target device.

    On 24 GB RTX 4090 GPUs, all four perception models (DA3Metric-Large,
    GroundingDINO SwinT, SAM1 ViT-H, OpenCLIP ViT-H-14) occupy ~9.2 GB in total.
    Keeping weights resident on the target device avoids transferring 17 GB
    across PCIe on every single 300x300 frame (~2.5 s/frame overhead). If
    another worker already occupies half of the 24 GB GPU, ``active_model``
    falls back cleanly to staging weights back to CPU.
    """
    torch = importlib.import_module("torch")
    try:
        first_param = next(model.parameters(), None)
    except Exception:  # noqa: BLE001
        first_param = None
    already_on_device = first_param is not None and str(first_param.device) == device
    staged_for_low_vram = False
    if not already_on_device:
        if device.startswith("cuda") and torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            if free_bytes < 3 * 1024**3:
                staged_for_low_vram = True
        model.to(device)
    try:
        with torch.inference_mode():
            yield model
    finally:
        if staged_for_low_vram and device.startswith("cuda"):
            model.to("cpu")
            torch.cuda.empty_cache()


@contextmanager
def gpu_file_lock(path: Path, *, timeout_seconds: float) -> Iterator[float]:
    """Serialize our complete inference frames without touching other GPU users.

    The stable lock file is intentionally retained: unlinking it could create
    two lock inodes and break mutual exclusion across existing workers.
    """
    import fcntl
    import math
    import time

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("GPU lock timeout_seconds must be finite and positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with path.open("a", encoding="utf-8") as stream:
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= timeout_seconds:
                    raise TimeoutError(
                        f"timed out waiting for ALFRED GPU lock: {path}"
                    ) from None
                time.sleep(0.05)
        try:
            yield time.monotonic() - started
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
