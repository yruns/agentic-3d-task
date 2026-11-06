"""Multi-GPU local HTTP perception server with session-fair work-stealing scheduling.

Decouples GPU model residency from evaluation client concurrency:
* Exactly one persistent ``perception_worker`` process is spawned per physical
  GPU in the server pool, keeping VRAM constant (~3.5 GB static weights per GPU)
  regardless of how many evaluation shards run concurrently.
* Concurrent client requests are scheduled across all GPUs in the pool via a
  round-robin session-fair work-stealing queue so a 12-frame initial scan from
  one episode never starves single-step action frames from other episodes.
* Request-level validation/file errors are isolated to the calling client and
  never terminate the shared GPU workers.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Any

from .model_runtime import physical_gpu_uuid
from .perception import PERCEPTION_PROFILE, ScenePerceptionConfig
from .provenance import PERCEPTION_PACKAGES, RuntimeProvenance, alfred_source_sha256
from .vocabulary import vocabulary_sha256
from .worker import JsonWorker, WorkerError

logger = logging.getLogger(__name__)


def validate_gpu_pool_indices(gpu_indices: Sequence[int]) -> tuple[int, ...]:
    """Require at least one unique nonnegative physical GPU index, forbidding GPU 1."""
    if not gpu_indices:
        raise ValueError("gpu_indices must contain at least one physical GPU index")
    normalized: list[int] = []
    for index in gpu_indices:
        if type(index) is not int or index < 0 or index == 1:
            raise ValueError(
                f"physical_gpu_index={index!r} must be a nonnegative integer other than forbidden GPU 1"
            )
        if index not in normalized:
            normalized.append(index)
    return tuple(normalized)


def canonical_model_config_signature(
    config: ScenePerceptionConfig,
) -> dict[str, object]:
    """Return the model-defining parameters that must match between server and clients."""
    return {
        "dino_config": str(config.dino_config),
        "dino_checkpoint": str(config.dino_checkpoint),
        "dino_text_encoder_dir": str(config.dino_text_encoder_dir),
        "sam_checkpoint": str(config.sam_checkpoint),
        "clip_checkpoint": str(config.clip_checkpoint),
        "depth_model_dir": str(config.depth_model_dir),
        "device": config.device,
        "sam_model_type": config.sam_model_type,
        "clip_model_name": config.clip_model_name,
        "box_threshold": config.box_threshold,
        "text_threshold": config.text_threshold,
        "nms_iou_threshold": config.nms_iou_threshold,
        "caption_batch_size": config.caption_batch_size,
        "max_detections": config.max_detections,
        "depth_source": config.depth_source,
    }


class _GpuWorkerSlot:
    """Own one persistent perception_worker subprocess bound to a single GPU."""

    def __init__(
        self,
        slot_id: int,
        config: ScenePerceptionConfig,
        *,
        log_path: Path,
        worker_factory: Callable[..., Any] = JsonWorker,
    ) -> None:
        self.slot_id = slot_id
        self.gpu_index = config.physical_gpu_index
        self.config = replace(config, server_url=None)
        self.log_path = log_path
        self._worker_factory = worker_factory
        self._worker: Any = None
        self._runtime_provenance: RuntimeProvenance | None = None
        self.restarts = 0
        self.completed_frames = 0
        self._start_worker()

    def _start_worker(self) -> None:
        environment = dict(os.environ)
        visible_uuid = (
            physical_gpu_uuid(self.config.physical_gpu_index)
            if self.config.device == "cuda:0"
            else ""
        )
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": visible_uuid,
                "NVIDIA_TF32_OVERRIDE": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        worker = self._worker_factory(
            command=[
                str(self.config.python_executable),
                "-u",
                "-m",
                "codex_agent.alfred.perception_worker",
            ],
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            log_path=self.log_path,
            timeout_seconds=self.config.timeout_seconds,
        )
        try:
            initialized = worker.request(
                {"command": "initialize", "config": self.config.to_payload()}
            )
            if initialized.get("backend") != PERCEPTION_PROFILE:
                raise ValueError("perception worker initialized an unexpected backend")
            provenance = RuntimeProvenance.model_validate(
                initialized.get("runtime_provenance")
            )
            if set(provenance.packages) != set(PERCEPTION_PACKAGES):
                raise ValueError("perception worker package provenance is incomplete")
            if provenance.package_source_sha256 != alfred_source_sha256():
                raise ValueError(
                    "perception worker imported different project source bytes"
                )
            if provenance.visible_devices != visible_uuid:
                raise ValueError(
                    "perception worker accelerator identity differs from configured UUID"
                )
        except BaseException:
            worker.close()
            raise
        self._worker = worker
        self._runtime_provenance = provenance

    @property
    def runtime_provenance(self) -> RuntimeProvenance:
        assert self._runtime_provenance is not None
        return self._runtime_provenance

    def _ensure_alive(self) -> None:
        if self._worker is None or getattr(self._worker, "is_closed", False):
            logger.warning(
                "Respawning perception worker slot=%d on physical_gpu=%d",
                self.slot_id,
                self.gpu_index,
            )
            self.restarts += 1
            self._start_worker()

    def predict_with_stats(
        self,
        *,
        rgb_path: str,
        frame_id: str,
        intrinsics: list[list[float]],
        output_dir: str,
    ) -> dict[str, object]:
        self._ensure_alive()
        response = self._worker.request(
            {
                "command": "predict_with_stats",
                "rgb_path": rgb_path,
                "frame_id": frame_id,
                "intrinsics": intrinsics,
                "output_dir": output_dir,
            }
        )
        self.completed_frames += 1
        if "observation" in response and "frame_stats" in response:
            return dict(response)
        # Support test doubles that return a bare PerceptionObservation dict.
        return {
            "observation": response,
            "frame_stats": {
                "load_seconds": 0.0,
                "inference_seconds": 0.0,
                "depth_seconds": 0.0,
                "detection_seconds": 0.0,
                "segmentation_seconds": 0.0,
                "feature_seconds": 0.0,
                "gpu_wait_seconds": 0.0,
                "peak_cuda_memory_bytes": 0,
                "device": self.config.device,
                "capped_detections": 0,
                "nms_suppressed": 0,
                "empty_masks": 0,
            },
        }

    def close(self) -> None:
        if self._worker is not None:
            try:
                self._worker.close()
            finally:
                self._worker = None


@dataclass
class _WaitTicket:
    client_id: str
    slot: _GpuWorkerSlot | None = None
    cancelled: bool = False


class _FairPerceptionPool:
    """Round-robin session-fair work-stealing scheduler across GPU worker slots."""

    def __init__(self, slots: Sequence[_GpuWorkerSlot]) -> None:
        if not slots:
            raise ValueError("perception pool requires at least one GPU worker slot")
        self._slots = tuple(slots)
        self._condition = threading.Condition()
        self._idle_slots: deque[_GpuWorkerSlot] = deque(self._slots)
        self._client_queues: dict[str, deque[_WaitTicket]] = {}
        self._client_order: deque[str] = deque()
        self._closed = False
        self.total_requests = 0

    @property
    def slots(self) -> tuple[_GpuWorkerSlot, ...]:
        return self._slots

    def acquire(
        self, client_id: str, *, timeout_seconds: float
    ) -> tuple[_GpuWorkerSlot, float]:
        started = time.monotonic()
        deadline = started + timeout_seconds
        with self._condition:
            if self._closed:
                raise RuntimeError("perception server pool is closed")
            if self._idle_slots and not self._client_order:
                slot = self._idle_slots.popleft()
                self.total_requests += 1
                return slot, 0.0
            ticket = _WaitTicket(client_id=client_id)
            queue = self._client_queues.get(client_id)
            if queue is None:
                queue = deque()
                self._client_queues[client_id] = queue
                self._client_order.append(client_id)
            queue.append(ticket)
            while ticket.slot is None:
                if self._closed:
                    ticket.cancelled = True
                    raise RuntimeError("perception server pool closed while waiting")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    ticket.cancelled = True
                    self._remove_cancelled_ticket(client_id, ticket)
                    raise TimeoutError(
                        f"timed out after {timeout_seconds}s waiting for a free GPU worker"
                    )
                self._condition.wait(timeout=remaining)
            self.total_requests += 1
            return ticket.slot, time.monotonic() - started

    def _remove_cancelled_ticket(self, client_id: str, ticket: _WaitTicket) -> None:
        queue = self._client_queues.get(client_id)
        if queue is not None:
            try:
                queue.remove(ticket)
            except ValueError:
                pass
            if not queue:
                del self._client_queues[client_id]
                try:
                    self._client_order.remove(client_id)
                except ValueError:
                    pass

    def release(self, slot: _GpuWorkerSlot) -> None:
        with self._condition:
            if self._closed:
                slot.close()
                return
            while self._client_order:
                client_id = self._client_order.popleft()
                queue = self._client_queues.get(client_id)
                if not queue:
                    self._client_queues.pop(client_id, None)
                    continue
                while queue:
                    ticket = queue.popleft()
                    if not ticket.cancelled:
                        if queue:
                            self._client_order.append(client_id)
                        else:
                            del self._client_queues[client_id]
                        ticket.slot = slot
                        self._condition.notify_all()
                        return
                self._client_queues.pop(client_id, None)
            self._idle_slots.append(slot)
            self._condition.notify_all()

    def status_snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "num_workers": len(self._slots),
                "idle_workers": len(self._idle_slots),
                "waiting_clients": len(self._client_order),
                "waiting_requests": sum(len(q) for q in self._client_queues.values()),
                "total_requests": self.total_requests,
                "workers": [
                    {
                        "slot_id": s.slot_id,
                        "gpu_index": s.gpu_index,
                        "completed_frames": s.completed_frames,
                        "restarts": s.restarts,
                    }
                    for s in self._slots
                ],
            }

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        for slot in self._slots:
            slot.close()


class ScenePerceptionServer:
    """Local loopback HTTP perception server backed by a multi-GPU worker pool."""

    def __init__(
        self,
        config: ScenePerceptionConfig,
        *,
        log_dir: Path,
        host: str = "127.0.0.1",
        port: int = 0,
        gpu_indices: Sequence[int] | None = None,
        worker_factory: Callable[..., Any] = JsonWorker,
    ) -> None:
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("ScenePerceptionServer must bind to loopback (127.0.0.1)")
        self.config = replace(config, server_url=None)
        self.log_dir = log_dir.absolute()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        resolved_gpus = (
            validate_gpu_pool_indices(gpu_indices)
            if gpu_indices is not None
            else (self.config.physical_gpu_index,)
        )
        self.gpu_indices = resolved_gpus
        slots: list[_GpuWorkerSlot] = []
        try:
            for slot_id, gpu_idx in enumerate(resolved_gpus):
                slot_cfg = replace(
                    self.config,
                    physical_gpu_index=gpu_idx,
                    gpu_lock_path=(
                        Path(f"/tmp/agentic3d-alfred-gpu{gpu_idx}.lock")
                        if self.config.device == "cuda:0"
                        else self.config.gpu_lock_path
                    ),
                )
                slots.append(
                    _GpuWorkerSlot(
                        slot_id,
                        slot_cfg,
                        log_path=self.log_dir
                        / f"perception_worker_slot{slot_id}_gpu{gpu_idx}.stderr.log",
                        worker_factory=worker_factory,
                    )
                )
        except BaseException:
            for s in slots:
                s.close()
            raise
        self._pool = _FairPerceptionPool(slots)
        self._server = self._build_http_server(host, port)
        self._thread: threading.Thread | None = None

    def _build_http_server(self, host: str, port: int) -> ThreadingHTTPServer:
        parent = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._send_json(200, parent.health_payload())
                elif self.path == "/statistics":
                    self._send_json(
                        200, {"ok": True, "result": parent._pool.status_snapshot()}
                    )
                else:
                    self._send_json(
                        404,
                        {
                            "ok": False,
                            "error_type": "NotFound",
                            "message": f"unknown path: {self.path}",
                        },
                    )

            def do_POST(self) -> None:
                if self.path != "/predict":
                    self._send_json(
                        404,
                        {
                            "ok": False,
                            "error_type": "NotFound",
                            "message": f"unknown path: {self.path}",
                        },
                    )
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("predict payload must be a JSON object")
                    result = parent.handle_predict(payload)
                    self._send_json(200, {"ok": True, "result": result})
                except WorkerError as exc:
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error_type": "WorkerError",
                            "message": str(exc),
                        },
                    )
                except (
                    ValueError,
                    FileNotFoundError,
                    FileExistsError,
                    PermissionError,
                ) as exc:
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Perception server error on /predict")
                    self._send_json(
                        500,
                        {
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )

            def _send_json(self, status: int, body: dict[str, object]) -> None:
                encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode(
                    "utf-8"
                )
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        return ThreadingHTTPServer((host, port), _Handler)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    @property
    def url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}"

    def health_payload(self) -> dict[str, object]:
        first_prov = self._pool.slots[0].runtime_provenance
        return {
            "ok": True,
            "status": "ready",
            "backend": PERCEPTION_PROFILE,
            "vocabulary_sha256": vocabulary_sha256(),
            "package_source_sha256": alfred_source_sha256(),
            "gpu_indices": list(self.gpu_indices),
            "num_workers": len(self._pool.slots),
            "configuration": canonical_model_config_signature(self.config),
            "runtime_provenance": first_prov.model_dump(mode="json"),
        }

    def handle_predict(self, payload: dict[str, object]) -> dict[str, object]:
        allowed_keys = {
            "client_id",
            "rgb_path",
            "frame_id",
            "intrinsics",
            "output_dir",
        }
        unexpected = set(payload) - allowed_keys
        if unexpected:
            raise ValueError(f"Extra inputs are not permitted: {sorted(unexpected)}")
        for required in ("rgb_path", "frame_id", "intrinsics", "output_dir"):
            if required not in payload:
                raise ValueError(f"missing required field: {required}")
        client_id = str(payload.get("client_id") or "default")
        rgb_path = str(payload["rgb_path"])
        frame_id = str(payload["frame_id"])
        intrinsics = payload["intrinsics"]
        output_dir = str(payload["output_dir"])
        if not isinstance(intrinsics, list):
            raise ValueError("intrinsics must be a 3x3 list")
        slot, pool_wait_s = self._pool.acquire(
            client_id, timeout_seconds=self.config.timeout_seconds
        )
        try:
            res = slot.predict_with_stats(
                rgb_path=rgb_path,
                frame_id=frame_id,
                intrinsics=intrinsics,
                output_dir=output_dir,
            )
            raw_stats = res.get("frame_stats")
            frame_stats = dict(raw_stats) if isinstance(raw_stats, dict) else {}
            raw_lock_wait = frame_stats.get("gpu_wait_seconds", 0.0)
            lock_wait = (
                float(raw_lock_wait) if isinstance(raw_lock_wait, (int, float)) else 0.0
            )
            frame_stats["gpu_wait_seconds"] = lock_wait + pool_wait_s
            return {
                "observation": res["observation"],
                "frame_stats": frame_stats,
            }
        finally:
            self._pool.release(slot)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="alfred-perception-server",
            daemon=True,
        )
        self._thread.start()

    def serve_forever(self) -> None:
        try:
            self._server.serve_forever()
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:  # noqa: BLE001
            pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._pool.close()

    def __enter__(self) -> ScenePerceptionServer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
