"""Run pinned upstream SCOPE with an explicit, fail-closed model transport.

The upstream algorithm and prompts remain untouched. Its two global model
clients are replaced after import so transport failures cannot be swallowed by
the upstream fallback paths. No upstream API secret is copied to Linux.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import logging
import os
import runpy
import sys
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Final, Protocol, cast, runtime_checkable

import httpx
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
)
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, SecretStr
from typing_extensions import Unpack

if TYPE_CHECKING:
    from scripts.scope.image_cache_transport import ImageCachingTransport
    from scripts.scope.prefilter_response_contract import (
        EMPTY_PREFILTER_HEADER,
        allows_empty_object_prefilter_response,
    )
elif __package__:
    from .image_cache_transport import ImageCachingTransport
    from .prefilter_response_contract import (
        EMPTY_PREFILTER_HEADER,
        allows_empty_object_prefilter_response,
    )
else:
    from image_cache_transport import ImageCachingTransport
    from prefilter_response_contract import (
        EMPTY_PREFILTER_HEADER,
        allows_empty_object_prefilter_response,
    )

logger = logging.getLogger(__name__)

_UPSTREAM_API_MODULES = (
    "src.eval_utils_gpt_goatbench",
    "src.potential_estimation_gpt_goal",
)
# Permit two SDK transport retries, then surface the final error to FailClosed.
_MODEL_TRANSPORT_MAX_RETRIES: Final[int] = 2
# Leave a response margin above the Mac bridge's 180-second upstream timeout.
_MODEL_TRANSPORT_TIMEOUT_SECONDS: Final[float] = 240.0
_CAPTURE_ENTRYPOINT_SHA256: Final[str] = (
    "ea2189b53f36f5cc7537a55c190ad0770d27cea0b56f8956dbd589a37203f3f9"
)
_CAPTURE_UPSTREAM_ENVIRONMENT: Final[tuple[tuple[str, str], ...]] = (
    ("TRANSFORMERS_VERBOSITY", "error"),
    ("TOKENIZERS_PARALLELISM", "false"),
    ("HABITAT_SIM_LOG", "quiet"),
    ("MAGNUM_LOG", "quiet"),
)
_DENOISE_ARGUMENT_NAMES: Final[tuple[str, ...]] = (
    "downsample_voxel_size",
    "dbscan_remove_noise",
    "dbscan_eps",
    "dbscan_min_points",
    "spatial_sim_type",
    "device",
    "objects",
)


class BridgeConnection(BaseModel):
    """Validated private connection settings for the Linux loopback bridge."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    endpoint: AnyHttpUrl
    bridge_token: SecretStr
    expected_model: str = Field(min_length=1)


class CompletionPort(Protocol):
    """The non-streaming completion operation used by upstream SCOPE."""

    def create(
        self, **parameters: Unpack[CompletionCreateParamsNonStreaming]
    ) -> ChatCompletion: ...


@dataclass(frozen=True)
class FailClosedCompletions:
    """Abort the CLI on infrastructure failure rather than inventing a score."""

    delegate: CompletionPort
    expected_model: str

    def create(
        self, **parameters: Unpack[CompletionCreateParamsNonStreaming]
    ) -> ChatCompletion:
        try:
            completion = self.delegate.create(**parameters)
        except openai.APIError as error:
            status = getattr(error, "status_code", None)
            description = (
                f"HTTP {status}" if status is not None else type(error).__name__
            )
            raise SystemExit(f"SCOPE model transport failed: {description}") from None
        if completion.model != self.expected_model:
            raise SystemExit("SCOPE model transport returned an unexpected model")
        if not completion.choices:
            raise SystemExit("SCOPE model transport returned empty assistant content")
        choice = completion.choices[0]
        content = choice.message.content
        if content is None:
            raise SystemExit("SCOPE model transport returned empty assistant content")
        if not content.strip():
            if not (
                choice.finish_reason == "stop"
                and allows_empty_object_prefilter_response(parameters.get("messages"))
            ):
                raise SystemExit(
                    "SCOPE model transport returned empty assistant content"
                )
            logger.info("Accepted empty answer for the original empty object prefilter")
        return completion


@dataclass(frozen=True)
class ChatEndpoint:
    """Expose the upstream client's chat.completions access path."""

    completions: FailClosedCompletions


@dataclass(frozen=True)
class ScopeClient:
    """Minimal injected client, without implicit initialization."""

    chat: ChatEndpoint


class ScopeApiModule(Protocol):
    """Concrete client attribute at the external Python-module boundary."""

    client: OpenAI | ScopeClient


class DenoiseOperation(Protocol):
    """Opaque upstream operation whose arguments and return identity are retained."""

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class ScopeSceneModule(Protocol):
    denoise_objects: DenoiseOperation


@runtime_checkable
class PointCloudPort(Protocol):
    """The point-cloud fields consumed by the pinned denoise implementation."""

    points: object
    colors: object

    def has_normals(self) -> bool: ...

    def has_covariances(self) -> bool: ...


@dataclass(frozen=True)
class _ObservedPointCloud:
    object_id: int
    point_cloud: PointCloudPort


def _prepare_capture_environment(upstream_root: Path) -> None:
    """Replay the verified entrypoint's environment before importing its Scene."""
    entrypoint = upstream_root / "run_goatbench_evaluation.py"
    source = entrypoint.read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    if source_sha != _CAPTURE_ENTRYPOINT_SHA256:
        raise ValueError(
            f"CPU capture entrypoint SHA={source_sha}; "
            f"requires {_CAPTURE_ENTRYPOINT_SHA256}"
        )
    prefix = ast.parse(source, filename=str(entrypoint)).body
    if not prefix or ast.dump(prefix[0]) != ast.dump(ast.parse("import os").body[0]):
        raise ValueError("CPU capture entrypoint environment prefix is incompatible")
    for position, (name, value) in enumerate(_CAPTURE_UPSTREAM_ENVIRONMENT, start=1):
        expected = ast.parse(f"os.environ[{name!r}] = {value!r}").body[0]
        if position >= len(prefix) or ast.dump(prefix[position]) != ast.dump(expected):
            raise ValueError(
                f"CPU capture entrypoint environment prefix is incompatible at {name}"
            )
    os.environ.update(_CAPTURE_UPSTREAM_ENVIRONMENT)


def _observe_point_clouds(objects: object) -> tuple[_ObservedPointCloud, ...]:
    """Validate original integer IDs and their point-cloud interface at the boundary."""
    if not isinstance(objects, Mapping):
        raise TypeError("CPU capture objects must be an ID-to-object mapping")
    clouds: list[_ObservedPointCloud] = []
    for object_id, original in cast(Mapping[object, object], objects).items():
        if type(object_id) is not int:
            raise TypeError("CPU capture object IDs must be integers")
        if not isinstance(original, Mapping):
            raise TypeError("CPU capture objects must provide a pcd field")
        point_cloud = cast(Mapping[str, object], original).get("pcd")
        if not isinstance(point_cloud, PointCloudPort):
            raise TypeError(
                "CPU capture pcd must provide points, colors, has_normals and has_covariances"
            )
        if not isinstance(point_cloud.points, Sized):
            raise TypeError("CPU capture pcd.points must have a point count")
        clouds.append(_ObservedPointCloud(object_id, point_cloud))
    return tuple(clouds)


def _capture_denoise_call(
    *,
    clouds: tuple[_ObservedPointCloud, ...],
    arguments: Mapping[str, object],
    capture_dir: Path,
    upstream_root: Path,
) -> None:
    """Copy diagnostic inputs lazily, without mutating the original objects."""
    if TYPE_CHECKING or __package__:
        from .cpu_pointcloud_capture import (
            DenoiseParameters,
            PointCloudInput,
            capture_denoise_inputs,
        )
    else:
        from cpu_pointcloud_capture import (
            DenoiseParameters,
            PointCloudInput,
            capture_denoise_inputs,
        )
    for cloud in clouds:
        if cloud.point_cloud.has_normals():
            raise ValueError("CPU capture refuses point clouds with unrecorded normals")
        if cloud.point_cloud.has_covariances():
            raise ValueError(
                "CPU capture refuses point clouds with unrecorded covariances"
            )
    parameters = DenoiseParameters.model_validate(
        {
            **{name: arguments[name] for name in _DENOISE_ARGUMENT_NAMES[:-1]},
            "device": str(arguments["device"]),
        }
    )
    inputs = tuple(
        PointCloudInput.from_arrays(
            cloud.object_id, cloud.point_cloud.points, cloud.point_cloud.colors
        )
        for cloud in clouds
    )
    metadata = capture_denoise_inputs(
        inputs,
        parameters,
        capture_dir=capture_dir,
        upstream_root=upstream_root,
        enabled=True,
    )
    if metadata is None:
        raise RuntimeError("Enabled CPU capture did not produce capture metadata")


class _FirstDenoiseCapture:
    """Capture the first nontrivial call once per actor, including concurrent callers."""

    def __init__(
        self,
        delegate: DenoiseOperation,
        *,
        capture_dir: Path,
        upstream_root: Path,
    ) -> None:
        self._delegate = delegate
        self._signature = inspect.signature(delegate)
        if set(self._signature.parameters) != set(_DENOISE_ARGUMENT_NAMES):
            raise ValueError(
                "CPU capture requires the original denoise parameter names"
            )
        self._capture_dir = capture_dir
        self._upstream_root = upstream_root
        self._captured = False
        self._lock = Lock()

    def __call__(self, *args: object, **kwargs: object) -> object:
        with self._lock:
            if not self._captured:
                bound = self._signature.bind(*args, **kwargs)
                arguments: Mapping[str, object] = bound.arguments
                if set(_DENOISE_ARGUMENT_NAMES) <= arguments.keys():
                    clouds = _observe_point_clouds(arguments["objects"])
                    if any(
                        len(cast(Sized, cloud.point_cloud.points)) > 1
                        for cloud in clouds
                    ):
                        _capture_denoise_call(
                            clouds=clouds,
                            arguments=arguments,
                            capture_dir=self._capture_dir,
                            upstream_root=self._upstream_root,
                        )
                        self._captured = True
        return self._delegate(*args, **kwargs)


@dataclass(frozen=True)
class RunSettings:
    """CLI paths and scene/episode selection for one upstream launch."""

    upstream_root: Path
    cfg_file: Path
    bridge_config: Path
    split: int
    start_ratio: float
    end_ratio: float
    cpu_capture_dir: Path | None = None

    def __post_init__(self) -> None:
        for path in (self.cfg_file, self.bridge_config):
            if not path.is_file():
                raise ValueError(f"required configuration file does not exist: {path}")
        entrypoint = self.upstream_root / "run_goatbench_evaluation.py"
        if not entrypoint.is_file():
            raise ValueError(f"upstream SCOPE entrypoint does not exist: {entrypoint}")
        if not 1 <= self.split <= 10:
            raise ValueError(f"split={self.split}; allowed range is 1..10")
        if not 0 <= self.start_ratio < self.end_ratio <= 1:
            raise ValueError(
                "scene ratios must satisfy 0 <= start_ratio < end_ratio <= 1"
            )
        if self.cpu_capture_dir is not None:
            if not self.cpu_capture_dir.is_absolute():
                raise ValueError(
                    f"cpu_capture_dir={self.cpu_capture_dir}; "
                    "must be an absolute directory path"
                )
            if self.cpu_capture_dir.exists() and not self.cpu_capture_dir.is_dir():
                raise ValueError(
                    f"cpu_capture_dir={self.cpu_capture_dir}; must refer to a directory"
                )


def _install_cpu_capture(settings: RunSettings) -> None:
    """Patch the Scene module's already-bound function only for an explicit capture."""
    if settings.cpu_capture_dir is None:
        return
    module = cast(ScopeSceneModule, importlib.import_module("src.scene_goatbench"))
    if not callable(module.denoise_objects):
        raise TypeError(
            "CPU capture requires callable src.scene_goatbench.denoise_objects"
        )
    actor_dir = settings.cpu_capture_dir / (
        f"split{settings.split}_{settings.start_ratio:.8f}_{settings.end_ratio:.8f}"
        f"_pid{os.getpid()}"
    )
    module.denoise_objects = _FirstDenoiseCapture(
        module.denoise_objects,
        capture_dir=actor_dir,
        upstream_root=settings.upstream_root,
    )


def run_upstream(settings: RunSettings) -> None:
    """Inject the validated bridge client and execute the original entrypoint."""
    connection = BridgeConnection.model_validate_json(
        settings.bridge_config.read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            {
                "requested_upstream_model": "gpt-4o-2024-11-20",
                "actual_model": connection.expected_model,
                "split": settings.split,
                "source": str(settings.upstream_root),
            }
        ),
        flush=True,
    )
    sys.path.insert(0, str(settings.upstream_root))
    os.chdir(settings.upstream_root)
    if settings.cpu_capture_dir is not None:
        _prepare_capture_environment(settings.upstream_root)
    with OpenAI(
        base_url=str(connection.endpoint),
        api_key=connection.bridge_token.get_secret_value(),
        max_retries=_MODEL_TRANSPORT_MAX_RETRIES,
        timeout=_MODEL_TRANSPORT_TIMEOUT_SECONDS,
        default_headers={EMPTY_PREFILTER_HEADER: "1"},
        http_client=httpx.Client(
            transport=ImageCachingTransport(httpx.HTTPTransport()),
            timeout=_MODEL_TRANSPORT_TIMEOUT_SECONDS,
            trust_env=False,
        ),
    ) as client:
        injected = ScopeClient(
            ChatEndpoint(
                FailClosedCompletions(
                    client.chat.completions, expected_model=connection.expected_model
                )
            )
        )
        for module_name in _UPSTREAM_API_MODULES:
            module = cast(ScopeApiModule, importlib.import_module(module_name))
            original_client = module.client
            if not isinstance(original_client, OpenAI):
                raise TypeError(f"unexpected upstream client type in {module_name}")
            original_client.close()
            module.client = injected
        _install_cpu_capture(settings)
        sys.argv = [
            str(settings.upstream_root / "run_goatbench_evaluation.py"),
            "-cf",
            str(settings.cfg_file),
            "--split",
            str(settings.split),
            "--start_ratio",
            str(settings.start_ratio),
            "--end_ratio",
            str(settings.end_ratio),
        ]
        runpy.run_path(sys.argv[0], run_name="__main__")


def main() -> None:
    """Parse the CLI and delegate to the upstream execution service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--bridge-config", type=Path, required=True)
    parser.add_argument("--split", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--start-ratio", type=float, default=0)
    parser.add_argument("--end-ratio", type=float, default=1)
    parser.add_argument("--cpu-capture-dir", type=Path, default=None)
    args = parser.parse_args()
    settings = RunSettings(
        upstream_root=args.upstream_root.resolve(),
        cfg_file=args.cfg_file.resolve(),
        bridge_config=args.bridge_config.resolve(),
        split=args.split,
        start_ratio=args.start_ratio,
        end_ratio=args.end_ratio,
        cpu_capture_dir=(
            args.cpu_capture_dir.resolve() if args.cpu_capture_dir is not None else None
        ),
    )
    run_upstream(settings)


if __name__ == "__main__":
    main()
