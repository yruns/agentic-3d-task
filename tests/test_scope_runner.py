"""Regression checks for the SCOPE transport boundary."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TypedDict, cast

import httpx
import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion

from scripts.scope import run_scope as runner
from scripts.scope.run_scope import (
    FailClosedCompletions,
    RunSettings,
    ScopeClient,
)

_ENTRYPOINT_PREFIX = """import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["MAGNUM_LOG"] = "quiet"
"""


@dataclass
class _ApiModule:
    client: OpenAI | ScopeClient


@dataclass(frozen=True)
class _ApiImports:
    modules: dict[str, _ApiModule]
    scene_module: _SceneModule | None = None

    def import_module(self, name: str) -> _ApiModule | _SceneModule:
        if name == "src.scene_goatbench" and self.scene_module is not None:
            return self.scene_module
        return self.modules[name]


@dataclass
class _SceneModule:
    denoise_objects: Callable[..., object]


@dataclass(frozen=True)
class _PointCloud:
    points: tuple[tuple[float, float, float], ...]
    colors: tuple[tuple[float, float, float], ...]
    normals_present: bool = False
    covariances_present: bool = False

    def has_normals(self) -> bool:
        return self.normals_present

    def has_covariances(self) -> bool:
        return self.covariances_present


class _PointCloudObject(TypedDict):
    pcd: _PointCloud


@dataclass(frozen=True)
class _SourceVersion:
    def check_output(self, command: list[str], *, text: bool) -> str:
        assert command[:2] == ["git", "-C"]
        assert command[-2:] == ["rev-parse", "HEAD"]
        assert text
        return "606d29ca19901882015cdd432017b4a867cef978\n"


@dataclass(frozen=True)
class _OriginalDevice:
    def __str__(self) -> str:
        return "cuda:0"


@dataclass(frozen=True)
class _DenoiseDelegate:
    calls: list[object]
    result: object
    expected_device: object
    failure: RuntimeError | None = None

    def __call__(
        self,
        downsample_voxel_size: float,
        dbscan_remove_noise: bool,
        dbscan_eps: float,
        dbscan_min_points: int,
        spatial_sim_type: str,
        device: object,
        objects: object,
    ) -> object:
        assert (downsample_voxel_size, dbscan_remove_noise, dbscan_eps) == (
            0.025,
            True,
            0.12,
        )
        assert (dbscan_min_points, spatial_sim_type) == (5, "overlap")
        assert device is self.expected_device
        self.calls.append(objects)
        if self.failure is not None:
            raise self.failure
        return self.result


def _denoise(module: _SceneModule, objects: object, device: object) -> object:
    return module.denoise_objects(
        downsample_voxel_size=0.025,
        dbscan_remove_noise=True,
        dbscan_eps=0.12,
        dbscan_min_points=5,
        spatial_sim_type="overlap",
        device=device,
        objects=objects,
    )


def _execute_runner_with_mock_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    respond: Callable[[httpx.Request], httpx.Response],
    *,
    cpu_capture_dir: Path | None = None,
    scene_module: _SceneModule | None = None,
    execute_navigation: Callable[[], object] | None = None,
) -> ChatCompletion:
    """Exercise the production client's actual timeout/retry configuration."""
    configured: list[tuple[int, float]] = []

    class MockTransportClient(OpenAI):
        def __init__(
            self,
            *,
            base_url: str,
            api_key: str,
            max_retries: int,
            timeout: float,
            default_headers: Mapping[str, str],
            http_client: httpx.Client,
        ) -> None:
            configured.append((max_retries, timeout))
            super().__init__(
                base_url=base_url,
                api_key=api_key,
                max_retries=max_retries,
                timeout=timeout,
                http_client=http_client,
                default_headers=default_headers,
            )

    modules = {
        name: _ApiModule(
            MockTransportClient(
                base_url="http://test-bridge.invalid/v1",
                api_key="test-token",
                max_retries=0,
                timeout=1.0,
                default_headers={},
                http_client=httpx.Client(transport=httpx.MockTransport(respond)),
            )
        )
        for name in (
            "src.eval_utils_gpt_goatbench",
            "src.potential_estimation_gpt_goal",
        )
    }
    completions: list[ChatCompletion] = []

    def execute(path_name: str, *, run_name: str) -> dict[str, object]:
        assert run_name == "__main__"
        assert "--cpu-capture-dir" not in sys.argv
        if execute_navigation is not None:
            execute_navigation()
        client = modules["src.eval_utils_gpt_goatbench"].client
        assert isinstance(client, ScopeClient)
        completions.append(
            client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[{"role": "user", "content": "test"}],
            )
        )
        return {}

    upstream_root = tmp_path / "upstream"
    upstream_root.mkdir()
    entrypoint = upstream_root / "run_goatbench_evaluation.py"
    entrypoint.write_text(_ENTRYPOINT_PREFIX)
    cfg_file = tmp_path / "evaluation.yaml"
    cfg_file.write_text("seed: 3407\n")
    bridge_config = tmp_path / "test_bridge.json"
    bridge_config.write_text(
        json.dumps(
            {
                "endpoint": "http://test-bridge.invalid/v1",
                "bridge_token": "test-token",
                "expected_model": "gemini-3.8-flash",
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(runner, "OpenAI", MockTransportClient)

    def mock_http_transport() -> httpx.MockTransport:
        return httpx.MockTransport(respond)

    monkeypatch.setattr(httpx, "HTTPTransport", mock_http_transport)
    monkeypatch.setattr(runner, "importlib", _ApiImports(modules, scene_module))
    monkeypatch.setattr(runpy, "run_path", execute)
    if cpu_capture_dir is None:
        settings = RunSettings(upstream_root, cfg_file, bridge_config, 1, 0.0, 1.0)
    else:
        source_file = upstream_root / "src/conceptgraph/slam/utils.py"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("# source version fixture\n")
        monkeypatch.setattr(
            runner,
            "_CAPTURE_ENTRYPOINT_SHA256",
            hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
            raising=False,
        )
        settings = RunSettings(
            upstream_root,
            cfg_file,
            bridge_config,
            1,
            0.0,
            1.0,
            cpu_capture_dir=cpu_capture_dir,
        )
    runner.run_upstream(settings)
    assert configured[-1] == (2, 240.0)
    assert len(completions) == 1
    return completions[0]


def test_api_failure_exits_instead_of_entering_upstream_fallback() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"error": {"message": "failed"}})

    with OpenAI(
        api_key="test-token",
        base_url="http://localhost/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        completions = FailClosedCompletions(
            client.chat.completions, expected_model="gemini-3.8-flash"
        )
        with pytest.raises(SystemExit, match="HTTP 502"):
            completions.create(
                model="gpt-4o-2024-11-20",
                messages=[{"role": "user", "content": "test"}],
            )


@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_completion_exits(content: str | None) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "test",
                "created": 1,
                "object": "chat.completion",
                "model": "gemini-3.8-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            },
        )

    with OpenAI(
        api_key="test-token",
        base_url="http://localhost/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        completions = FailClosedCompletions(
            client.chat.completions, expected_model="gemini-3.8-flash"
        )
        with pytest.raises(SystemExit, match="empty assistant content"):
            completions.create(
                model="gpt-4o-2024-11-20",
                messages=[{"role": "user", "content": "test"}],
            )


def test_runner_recovers_from_one_transient_gateway_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                504,
                json={"error": {"message": "temporary gateway timeout"}},
                headers={"retry-after-ms": "1"},
            )
        return httpx.Response(
            200,
            json={
                "id": "valid-retry",
                "created": 1,
                "object": "chat.completion",
                "model": "gemini-3.8-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "frontier 0"},
                    }
                ],
            },
        )

    completion = _execute_runner_with_mock_transport(tmp_path, monkeypatch, respond)
    assert len(requests) == 2
    assert completion.id == "valid-retry"
    assert completion.choices[0].message.content == "frontier 0"


def test_runner_still_exits_after_two_failed_transport_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            504,
            json={"error": {"message": "persistent gateway timeout"}},
            headers={"retry-after-ms": "1"},
        )

    with pytest.raises(SystemExit, match="SCOPE model transport failed: HTTP 504"):
        _execute_runner_with_mock_transport(tmp_path, monkeypatch, respond)
    assert len(requests) == 3


def test_capture_settings_reject_a_file_as_the_capture_directory(
    tmp_path: Path,
) -> None:
    upstream_root = tmp_path / "upstream"
    upstream_root.mkdir()
    (upstream_root / "run_goatbench_evaluation.py").write_text("# fixture\n")
    cfg_file = tmp_path / "evaluation.yaml"
    cfg_file.write_text("seed: 3407\n")
    bridge_config = tmp_path / "test_bridge.json"
    bridge_config.write_text("{}\n")
    capture_file = tmp_path / "capture"
    capture_file.write_text("existing unrelated data\n")

    with pytest.raises(ValueError, match="cpu_capture_dir"):
        RunSettings(
            upstream_root,
            cfg_file,
            bridge_config,
            1,
            0.0,
            0.5,
            cpu_capture_dir=capture_file,
        )


def test_capture_settings_reject_a_relative_directory(tmp_path: Path) -> None:
    upstream_root = tmp_path / "upstream"
    upstream_root.mkdir()
    (upstream_root / "run_goatbench_evaluation.py").write_text("# fixture\n")
    cfg_file = tmp_path / "evaluation.yaml"
    cfg_file.write_text("seed: 3407\n")
    bridge_config = tmp_path / "test_bridge.json"
    bridge_config.write_text("{}\n")

    with pytest.raises(ValueError, match="cpu_capture_dir.*absolute"):
        RunSettings(
            upstream_root,
            cfg_file,
            bridge_config,
            1,
            0.0,
            0.5,
            cpu_capture_dir=Path("shared-capture"),
        )


def _successful_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "valid-response",
            "created": 1,
            "object": "chat.completion",
            "model": "gemini-3.8-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "frontier 0"},
                }
            ],
        },
    )


def test_runner_default_does_not_change_environment_or_import_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = {
        name: os.environ.get(name) for name, _ in runner._CAPTURE_UPSTREAM_ENVIRONMENT
    }
    _execute_runner_with_mock_transport(tmp_path, monkeypatch, _successful_response)
    assert {name: os.environ.get(name) for name in environment} == environment


def test_enabled_capture_rejects_an_unknown_entrypoint_before_environment_change(
    tmp_path: Path,
) -> None:
    upstream_root = tmp_path / "upstream"
    upstream_root.mkdir()
    (upstream_root / "run_goatbench_evaluation.py").write_text(_ENTRYPOINT_PREFIX)
    environment = {
        name: os.environ.get(name) for name, _ in runner._CAPTURE_UPSTREAM_ENVIRONMENT
    }
    with pytest.raises(ValueError, match="entrypoint.*SHA"):
        runner._prepare_capture_environment(upstream_root)
    assert {name: os.environ.get(name) for name in environment} == environment


def test_runner_captures_first_actual_call_and_preserves_delegate_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.scope import cpu_pointcloud_capture as capture

    monkeypatch.setattr(capture, "subprocess", _SourceVersion())
    for name, _ in runner._CAPTURE_UPSTREAM_ENVIRONMENT:
        monkeypatch.setenv(name, "before-upstream-import")
    device = _OriginalDevice()
    returned = object()
    calls: list[object] = []
    scene = _SceneModule(_DenoiseDelegate(calls, returned, device))
    capture_root = tmp_path / "captures"
    one_point: dict[int, _PointCloudObject] = {
        4: {"pcd": _PointCloud(((0.0, 0.0, 0.0),), ((0.5, 0.5, 0.5),))}
    }
    actual: dict[int, _PointCloudObject] = {
        11: {
            "pcd": _PointCloud(
                ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                ((0.5, 0.5, 0.5), (0.1, 0.2, 0.3)),
            )
        }
    }

    def execute_navigation() -> None:
        assert all(
            os.environ[name] == value
            for name, value in runner._CAPTURE_UPSTREAM_ENVIRONMENT
        )
        assert _denoise(scene, {}, device) is returned
        assert _denoise(scene, one_point, device) is returned
        assert not capture_root.exists()
        assert _denoise(scene, actual, device) is returned
        assert _denoise(scene, actual, device) is returned

    _execute_runner_with_mock_transport(
        tmp_path,
        monkeypatch,
        _successful_response,
        cpu_capture_dir=capture_root,
        scene_module=scene,
        execute_navigation=execute_navigation,
    )
    assert len(calls) == 4
    assert calls[1] is one_point and calls[2] is actual and calls[3] is actual
    captures = tuple(capture_root.iterdir())
    assert len(captures) == 1
    assert captures[0].name.endswith(f"_pid{os.getpid()}")
    stored = capture.load_capture(captures[0])
    assert tuple(item.object_id for item in stored.inputs) == (11,)
    assert not stored.inputs[0].points.flags.writeable
    assert not stored.inputs[0].colors.flags.writeable
    assert stored.metadata.parameters == capture.DenoiseParameters(
        downsample_voxel_size=0.025,
        dbscan_remove_noise=True,
        dbscan_eps=0.12,
        dbscan_min_points=5,
        spatial_sim_type="overlap",
        device="cuda:0",
    )


@pytest.mark.parametrize("attribute", ["normals", "covariances"])
def test_capture_refuses_unrecorded_point_cloud_attributes(
    tmp_path: Path, attribute: str
) -> None:
    calls: list[object] = []
    device = _OriginalDevice()
    original = _DenoiseDelegate(calls, object(), device)
    wrapped = runner._FirstDenoiseCapture(
        cast(runner.DenoiseOperation, original),
        capture_dir=tmp_path / "capture",
        upstream_root=tmp_path,
    )
    scene = _SceneModule(wrapped)
    objects: dict[int, _PointCloudObject] = {
        11: {
            "pcd": _PointCloud(
                ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                ((0.5, 0.5, 0.5), (0.1, 0.2, 0.3)),
                normals_present=attribute == "normals",
                covariances_present=attribute == "covariances",
            )
        }
    }
    with pytest.raises(ValueError, match=f"unrecorded {attribute}"):
        _denoise(scene, objects, device)
    assert not calls and not (tmp_path / "capture").exists()


def test_capture_keeps_the_original_denoise_exception(tmp_path: Path) -> None:
    failure = RuntimeError("original denoise failure")
    calls: list[object] = []
    device = _OriginalDevice()
    wrapped = runner._FirstDenoiseCapture(
        cast(
            runner.DenoiseOperation, _DenoiseDelegate(calls, object(), device, failure)
        ),
        capture_dir=tmp_path / "capture",
        upstream_root=tmp_path,
    )
    with pytest.raises(RuntimeError) as caught:
        _denoise(_SceneModule(wrapped), {}, device)
    assert caught.value is failure and len(calls) == 1


@pytest.mark.parametrize("object_id", ["11", True])
def test_capture_does_not_coerce_original_object_ids(
    tmp_path: Path, object_id: object
) -> None:
    calls: list[object] = []
    device = _OriginalDevice()
    wrapped = runner._FirstDenoiseCapture(
        cast(runner.DenoiseOperation, _DenoiseDelegate(calls, object(), device)),
        capture_dir=tmp_path / "capture",
        upstream_root=tmp_path,
    )
    objects: dict[object, _PointCloudObject] = {
        object_id: {
            "pcd": _PointCloud(
                ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                ((0.5, 0.5, 0.5), (0.1, 0.2, 0.3)),
            )
        }
    }
    with pytest.raises(TypeError, match="object IDs must be integers"):
        _denoise(_SceneModule(wrapped), objects, device)
    assert not calls and not (tmp_path / "capture").exists()


def _constant_pid(pid: int) -> int:
    return pid


def test_two_actors_publish_separate_captures_under_a_shared_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.scope import cpu_pointcloud_capture as capture

    monkeypatch.setattr(capture, "subprocess", _SourceVersion())
    for name, _ in runner._CAPTURE_UPSTREAM_ENVIRONMENT:
        monkeypatch.setenv(name, "before-upstream-import")
    capture_root = tmp_path / "shared-captures"
    device = _OriginalDevice()
    objects: dict[int, _PointCloudObject] = {
        11: {
            "pcd": _PointCloud(
                ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                ((0.5, 0.5, 0.5), (0.1, 0.2, 0.3)),
            )
        }
    }
    for pid in (301, 302):
        actor_root = tmp_path / f"actor-{pid}"
        actor_root.mkdir()
        scene = _SceneModule(_DenoiseDelegate([], object(), device))
        monkeypatch.setattr(os, "getpid", partial(_constant_pid, pid))
        _execute_runner_with_mock_transport(
            actor_root,
            monkeypatch,
            _successful_response,
            cpu_capture_dir=capture_root,
            scene_module=scene,
            execute_navigation=partial(_denoise, scene, objects, device),
        )
    captures = tuple(capture_root.iterdir())
    assert len(captures) == 2
    assert {path.name.rsplit("_pid", 1)[1] for path in captures} == {"301", "302"}
    assert all(
        capture.load_capture(path).inputs[0].object_id == 11 for path in captures
    )


def test_runner_cli_routes_the_capture_directory_to_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream_root = tmp_path / "upstream"
    upstream_root.mkdir()
    (upstream_root / "run_goatbench_evaluation.py").write_text(_ENTRYPOINT_PREFIX)
    cfg_file = tmp_path / "evaluation.yaml"
    cfg_file.write_text("seed: 3407\n")
    bridge_config = tmp_path / "test_bridge.json"
    bridge_config.write_text("{}\n")
    received: list[RunSettings] = []

    def record_settings(settings: RunSettings) -> None:
        received.append(settings)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "run_upstream", record_settings)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scope.py",
            "--upstream-root",
            str(upstream_root),
            "--cfg-file",
            str(cfg_file),
            "--bridge-config",
            str(bridge_config),
            "--cpu-capture-dir",
            "captured-inputs",
        ],
    )
    runner.main()
    assert len(received) == 1
    assert received[0].cpu_capture_dir == (tmp_path / "captured-inputs").resolve()
