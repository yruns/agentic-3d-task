"""Native single-process SceneFunc3D Molmo and SAM2 sidecar server."""

from __future__ import annotations

import argparse
import socket
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import TypedDict, TypeVar, cast, overload

from pydantic import ValidationError

from codex_agent.scenefunc3d.servers.http_json import (
    JsonHttpError,
    JsonObject,
    JsonRoute,
    make_json_handler,
)
from codex_agent.scenefunc3d.servers.molmo_point_server import (
    DEFAULT_DEVICE,
    MolmoRunner,
    MolmoRunnerPointResult,
    TransformersMolmoRunner,
    run_molmo_point_request,
)
from codex_agent.scenefunc3d.servers.molmo_point_server import (
    DEFAULT_MODEL_NAME as DEFAULT_MOLMO_MODEL_NAME,
)
from codex_agent.scenefunc3d.servers.sam2_mask_server import (
    DEFAULT_MODEL_NAME as DEFAULT_SAM_MODEL_NAME,
)
from codex_agent.scenefunc3d.servers.sam2_mask_server import (
    OfficialSam2Runner,
    Sam2Backend,
    Sam2Runner,
    SamArtifactWriteError,
    SamImageLoadError,
    SamInvalidOutputError,
    SamStagingPathError,
    TransformersSam2Runner,
    run_sam_mask_request,
)
from codex_agent.scenefunc3d.servers.schemas import (
    HealthResponse,
    MolmoPointRequest,
    MolmoPointResponse,
    SamMaskRequest,
    SamMaskResponse,
)

DEFAULT_HOST = "::"
DEFAULT_PORT = 9001
_ArgparseNamespace = TypeVar("_ArgparseNamespace")
_GPU_RESOURCE_ERROR_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cuda error",
    "resource exhausted",
)


class ValidationIssuePayload(TypedDict):
    """JSON-ready details for one request validation issue."""

    field: str
    message: str


@dataclass(frozen=True)
class SceneFuncSidecarRunners:
    """Loaded inference runners hosted by the native SceneFunc3D sidecar."""

    molmo_runner: MolmoRunner
    sam_runner: Sam2Runner


class _NativeSidecarArgumentParser(argparse.ArgumentParser):
    @overload
    def parse_args(
        self,
        args: Iterable[str] | None = None,
        namespace: None = None,
    ) -> argparse.Namespace: ...

    @overload
    def parse_args(
        self,
        args: Iterable[str] | None,
        namespace: _ArgparseNamespace,
    ) -> _ArgparseNamespace: ...

    @overload
    def parse_args(
        self,
        *,
        namespace: _ArgparseNamespace,
    ) -> _ArgparseNamespace: ...

    def parse_args(
        self,
        args: Iterable[str] | None = None,
        namespace: _ArgparseNamespace | None = None,
    ) -> argparse.Namespace | _ArgparseNamespace:
        parsed_args = super().parse_args(args, namespace)
        _validate_cli_args(self, cast(argparse.Namespace, parsed_args))
        return parsed_args


class _DualStackThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    daemon_threads = True


def build_routes(
    runners: SceneFuncSidecarRunners,
    *,
    base_paths: Sequence[str] = (),
) -> dict[str, JsonRoute]:
    """Build HTTP JSON routes for one native Molmo+SAM sidecar process."""

    routes: dict[str, JsonRoute] = {
        "/health": lambda payload: _handle_aggregate_health(runners),
        "/molmo/health": lambda payload: _handle_component_health(
            runners.molmo_runner.model_name
        ),
        "/sam/health": lambda payload: _handle_component_health(
            runners.sam_runner.model_name
        ),
        "/molmo/v1/point": lambda payload: _handle_molmo_point(
            runners.molmo_runner,
            payload,
        ),
        "/sam/v1/masks": lambda payload: _handle_sam_masks(
            runners.sam_runner,
            payload,
        ),
    }
    return _with_base_path_routes(routes, base_paths=base_paths)


def serve(
    runners: SceneFuncSidecarRunners,
    *,
    host: str,
    port: int,
    base_paths: Sequence[str] = (),
) -> None:
    """Run the blocking native SceneFunc3D sidecar HTTP server."""

    handler = make_json_handler(build_routes(runners, base_paths=base_paths))
    server = _build_http_server(host=host, port=port, handler=handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the native SceneFunc3D sidecar command line parser."""

    parser = _NativeSidecarArgumentParser(
        description="Serve MolmoPoint and SAM2 from one SceneFunc3D sidecar process."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--molmo-model-name", default=DEFAULT_MOLMO_MODEL_NAME)
    parser.add_argument("--molmo-model-path", type=Path, required=True)
    parser.add_argument("--sam-model-name", default=DEFAULT_SAM_MODEL_NAME)
    parser.add_argument(
        "--sam-backend",
        choices=("official", "transformers"),
        default="transformers",
    )
    parser.add_argument("--sam-checkpoint-path", type=Path)
    parser.add_argument("--sam-config-path", type=Path)
    parser.add_argument("--sam-model-path", type=Path)
    parser.add_argument("--sam-model-id", default="")
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument(
        "--base-path",
        action="append",
        default=[],
        help="Optional reverse-proxy mount path such as /s/<export-id>.",
    )
    return parser


def build_runners_from_args(args: argparse.Namespace) -> SceneFuncSidecarRunners:
    """Construct MolmoPoint and SAM2 runners from validated CLI arguments."""

    device = _namespace_str(args, "device")
    molmo_runner = TransformersMolmoRunner(
        model_name=_namespace_str(args, "molmo_model_name"),
        model_path=_namespace_required_path(args, "molmo_model_path"),
        device=device,
    )
    sam_runner = _build_sam_runner_from_args(args, device=device)
    return SceneFuncSidecarRunners(
        molmo_runner=molmo_runner,
        sam_runner=sam_runner,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the native SceneFunc3D sidecar server."""

    args = build_arg_parser().parse_args(argv)
    runners = build_runners_from_args(args)
    try:
        serve(
            runners,
            host=_namespace_str(args, "host"),
            port=_namespace_int(args, "port"),
            base_paths=_namespace_str_sequence(args, "base_path"),
        )
    except KeyboardInterrupt:
        return 130
    return 0


def _handle_aggregate_health(runners: SceneFuncSidecarRunners) -> JsonObject:
    return {
        "status": "ok",
        "molmo": _component_health_payload(runners.molmo_runner.model_name),
        "sam": _component_health_payload(runners.sam_runner.model_name),
    }


def _handle_component_health(model_name: str) -> JsonObject:
    return _component_health_payload(model_name)


def _component_health_payload(model_name: str) -> JsonObject:
    response = HealthResponse(
        status="ok",
        model_name=model_name,
        model_loaded=True,
    )
    return response.model_dump(mode="json")


def _handle_molmo_point(
    runner: MolmoRunner,
    payload: JsonObject,
) -> JsonObject:
    try:
        request = MolmoPointRequest.model_validate(payload)
    except ValidationError as exc:
        raise JsonHttpError(
            400,
            {
                "error": "invalid_request",
                "details": _validation_error_details(exc),
            },
        ) from exc
    start_time = time.perf_counter()
    try:
        point_result = run_molmo_point_request(runner, request)
    except RuntimeError as exc:
        if _is_gpu_resource_error(exc):
            raise JsonHttpError(503, {"error": "molmo_resource_unavailable"}) from exc
        raise
    return _molmo_point_response(
        request=request,
        point_result=point_result,
        model_name=runner.model_name,
        latency_ms=(time.perf_counter() - start_time) * 1000,
    )


def _handle_sam_masks(
    runner: Sam2Runner,
    payload: JsonObject,
) -> JsonObject:
    try:
        request = SamMaskRequest.model_validate(payload)
    except ValidationError as exc:
        raise JsonHttpError(
            400,
            {
                "error": "invalid_request",
                "details": _validation_error_details(exc),
            },
        ) from exc
    start_time = time.perf_counter()
    try:
        candidates = run_sam_mask_request(runner, request)
    except SamStagingPathError as exc:
        raise JsonHttpError(
            400,
            {"error": "invalid_staging_dir", "request_id": request.request_id},
        ) from exc
    except SamImageLoadError as exc:
        raise JsonHttpError(
            400,
            {"error": "sam_image_load_failed", "request_id": request.request_id},
        ) from exc
    except SamInvalidOutputError as exc:
        raise JsonHttpError(
            500,
            {"error": "sam_invalid_output", "request_id": request.request_id},
        ) from exc
    except SamArtifactWriteError as exc:
        raise JsonHttpError(
            500,
            {"error": "sam_artifact_write_failed", "request_id": request.request_id},
        ) from exc
    except RuntimeError as exc:
        if _is_gpu_resource_error(exc):
            raise JsonHttpError(503, {"error": "sam_resource_unavailable"}) from exc
        raise
    response = SamMaskResponse(
        request_id=request.request_id,
        model_name=runner.model_name,
        candidates=candidates,
        latency_ms=(time.perf_counter() - start_time) * 1000,
    )
    return cast(JsonObject, response.model_dump(mode="json", exclude_none=True))


def _molmo_point_response(
    *,
    request: MolmoPointRequest,
    point_result: MolmoRunnerPointResult,
    model_name: str,
    latency_ms: float,
) -> JsonObject:
    response = MolmoPointResponse(
        request_id=request.request_id,
        model_name=model_name,
        raw_text=point_result.raw_text,
        image_points=point_result.image_points,
        latency_ms=latency_ms,
    )
    return response.model_dump(mode="json")


def _build_sam_runner_from_args(
    args: argparse.Namespace,
    *,
    device: str,
) -> Sam2Runner:
    staging_root = _namespace_required_path(args, "staging_root")
    backend = _namespace_sam_backend(args)
    if backend == "official":
        return OfficialSam2Runner(
            model_name=_namespace_str(args, "sam_model_name"),
            checkpoint_path=_namespace_required_path(args, "sam_checkpoint_path"),
            config_path=_namespace_required_path(args, "sam_config_path"),
            staging_root=staging_root,
            device=device,
        )
    return TransformersSam2Runner(
        model_name=_namespace_str(args, "sam_model_name"),
        model_reference=_sam_transformers_model_reference(args),
        staging_root=staging_root,
        device=device,
    )


def _build_http_server(
    *,
    host: str,
    port: int,
    handler: type,
) -> ThreadingHTTPServer:
    if ":" in host:
        return _DualStackThreadingHTTPServer((host, port), handler)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def _with_base_path_routes(
    routes: dict[str, JsonRoute],
    *,
    base_paths: Sequence[str],
) -> dict[str, JsonRoute]:
    expanded_routes = dict(routes)
    for raw_base_path in base_paths:
        base_path = _normalize_base_path(raw_base_path)
        for route_path, route in routes.items():
            expanded_routes[base_path + route_path] = route
    return expanded_routes


def _normalize_base_path(base_path: str) -> str:
    normalized_base_path = "/" + base_path.strip("/")
    if normalized_base_path == "/":
        raise RuntimeError("base_path must not be empty")
    return normalized_base_path


def _validate_cli_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    backend = _namespace_sam_backend(args)
    if backend == "official":
        if (
            _namespace_optional_path(args, "sam_checkpoint_path") is None
            or _namespace_optional_path(args, "sam_config_path") is None
        ):
            parser.error(
                "--sam-backend official requires "
                "--sam-checkpoint-path and --sam-config-path"
            )
        return
    has_model_path = _namespace_optional_path(args, "sam_model_path") is not None
    has_model_id = bool(_namespace_trimmed_str(args, "sam_model_id"))
    if has_model_path == has_model_id:
        parser.error(
            "--sam-backend transformers requires exactly one of "
            "--sam-model-path or --sam-model-id"
        )


def _namespace_sam_backend(args: argparse.Namespace) -> Sam2Backend:
    backend: object = args.sam_backend
    if backend == "official" or backend == "transformers":
        return backend
    raise RuntimeError(f"unsupported SAM2 backend: {backend!r}")


def _namespace_str(args: argparse.Namespace, name: str) -> str:
    value: object = getattr(args, name)
    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be a string: {value!r}")
    return value


def _namespace_trimmed_str(args: argparse.Namespace, name: str) -> str:
    return _namespace_str(args, name).strip()


def _namespace_int(args: argparse.Namespace, name: str) -> int:
    value: object = getattr(args, name)
    if not isinstance(value, int):
        raise RuntimeError(f"{name} must be an int: {value!r}")
    return value


def _namespace_str_sequence(args: argparse.Namespace, name: str) -> tuple[str, ...]:
    value: object = getattr(args, name)
    if not isinstance(value, list):
        raise RuntimeError(f"{name} must be a list of strings: {value!r}")
    values: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise RuntimeError(f"{name} item must be a string: {item!r}")
        values.append(item)
    return tuple(values)


def _namespace_required_path(args: argparse.Namespace, name: str) -> Path:
    value = _namespace_optional_path(args, name)
    if value is None:
        raise RuntimeError(f"{name} is required")
    return value


def _namespace_optional_path(args: argparse.Namespace, name: str) -> Path | None:
    value: object = getattr(args, name)
    if value is None:
        return None
    if not isinstance(value, Path):
        raise RuntimeError(f"{name} must be a path: {value!r}")
    return value


def _sam_transformers_model_reference(args: argparse.Namespace) -> str:
    model_path = _namespace_optional_path(args, "sam_model_path")
    if model_path is not None:
        return _resolve_transformers_model_path(model_path)
    return _namespace_trimmed_str(args, "sam_model_id")


def _resolve_transformers_model_path(path: Path) -> str:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise RuntimeError(f"sam_model_path must exist: {resolved_path}")
    if resolved_path.is_file():
        if resolved_path.name != "model.safetensors":
            raise RuntimeError(
                "sam_model_path file must be model.safetensors or a "
                f"HF snapshot directory: {resolved_path}"
            )
        return str(resolved_path.parent)
    if not resolved_path.is_dir():
        raise RuntimeError(
            f"sam_model_path must be a file or directory: {resolved_path}"
        )
    return str(resolved_path)


def _validation_error_details(exc: ValidationError) -> list[ValidationIssuePayload]:
    details: list[ValidationIssuePayload] = []
    for error in exc.errors():
        field_name = ".".join(str(part) for part in error.get("loc", ())) or "(root)"
        details.append(
            {
                "field": field_name,
                "message": str(error.get("msg", "invalid")),
            }
        )
    return details


def _is_gpu_resource_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _GPU_RESOURCE_ERROR_MARKERS)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SceneFuncSidecarRunners",
    "build_arg_parser",
    "build_routes",
    "build_runners_from_args",
    "main",
    "serve",
]
