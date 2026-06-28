"""Tests for the SceneFunc3D single-case runtime runner."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import TypeVar, cast

import pytest

from codex_agent.errors import CodexResponseError
from codex_agent.models import CodexTaskResult, CodexTurnMetadata, CodexTurnResult
from codex_agent.scenefunc3d.runner import (
    SceneFunc3dMaskOutcome,
    SceneFunc3dMaskTask,
    SceneFunc3dRunnerConfig,
    build_arg_parser,
    check_sidecar_health,
    run_single_sample,
)
from codex_agent.scenefunc3d.sample import SceneFunc3dSample, SceneFuncMotionHint
from codex_agent.scenefunc3d.servers.http_json import (
    JsonObject,
    make_json_handler,
)
from codex_agent.tasks.base import CodexTask

ResultT = TypeVar("ResultT")


def test_build_arg_parser_accepts_single_case_runtime_options(
    tmp_path: Path,
) -> None:
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "--dataset-root",
            str(tmp_path / "data"),
            "--sample-id",
            "421254::desc-a",
            "--backend-config",
            str(tmp_path / "backends.toml"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert args.dataset_root == tmp_path / "data"
    assert args.sample_id == "421254::desc-a"
    assert args.backend_config == tmp_path / "backends.toml"
    assert args.output_dir == tmp_path / "out"


def test_mask_task_prompt_inlines_tools_without_attachments(
    tmp_path: Path,
) -> None:
    scene_root = tmp_path / "421254"
    backend_config_path = tmp_path / "backends.toml"
    output_dir = tmp_path / "out"
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=scene_root,
        output_dir=output_dir,
        backend_config_path=backend_config_path,
    )

    request = task.build_turn_request()

    assert request.skills == ()
    assert request.image_paths == ()
    assert "Molmo point" in request.prompt
    assert "SAM candidates" in request.prompt
    assert "--backend-config" in request.prompt
    assert (
        f"python -m codex_agent.scenefunc3d.tools <tool> "
        f"--scene-root {scene_root} "
        f"--backend-config {backend_config_path} "
        f"--out-dir {output_dir} --args '<json>'"
    ) in request.prompt


def test_mask_task_parses_strict_final_json(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    _write_outcome_artifacts(output_dir)
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=tmp_path / "421254",
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )
    payload = _outcome_payload(output_dir)

    outcome = task.parse_response(json.dumps(payload))

    assert task.is_valid_response(json.dumps(payload)) is True
    assert isinstance(outcome, SceneFunc3dMaskOutcome)
    assert outcome.mask_artifact_path == output_dir / "mask_artifact.json"
    assert outcome.mask_npz_path == output_dir / "mask.npz"
    assert outcome.mask_ply_path == output_dir / "mask.ply"
    assert outcome.selected_frame_ids == ("000010", "000020")
    assert outcome.accepted_fragment_ids == ("frag-a",)
    assert outcome.confidence == 0.87
    assert outcome.uncertainties == ("partial occlusion",)
    assert outcome.to_payload() == payload


def test_mask_task_rejects_nonexistent_final_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    task = SceneFunc3dMaskTask(
        sample=_sample(),
        scene_root=tmp_path / "421254",
        output_dir=output_dir,
        backend_config_path=tmp_path / "backends.toml",
    )
    response_text = json.dumps(_outcome_payload(output_dir))

    assert task.is_valid_response(response_text) is False
    with pytest.raises(CodexResponseError, match="does not exist"):
        task.parse_response(response_text)


def test_check_sidecar_health_passes_for_healthy_fake_servers(
    tmp_path: Path,
) -> None:
    molmo_server = _start_health_server(model_name="fake-molmo")
    sam_server = _start_health_server(model_name="fake-sam")
    backend_config_path = tmp_path / "backends.toml"
    _write_backend_config(
        backend_config_path,
        molmo_url=molmo_server.base_url,
        sam_url=sam_server.base_url,
        root=tmp_path,
    )

    try:
        check_sidecar_health(backend_config_path)
    finally:
        molmo_server.close()
        sam_server.close()


def test_check_sidecar_health_rejects_unloaded_sidecar(
    tmp_path: Path,
) -> None:
    molmo_server = _start_health_server(model_name="fake-molmo")
    sam_server = _start_health_server_with_state(
        model_name="fake-sam",
        model_loaded=False,
    )
    backend_config_path = tmp_path / "backends.toml"
    _write_backend_config(
        backend_config_path,
        molmo_url=molmo_server.base_url,
        sam_url=sam_server.base_url,
        root=tmp_path,
    )

    try:
        with pytest.raises(RuntimeError, match="model_loaded=false"):
            check_sidecar_health(backend_config_path)
    finally:
        molmo_server.close()
        sam_server.close()


def test_check_sidecars_script_uses_backend_config(
    tmp_path: Path,
) -> None:
    molmo_server = _start_health_server(model_name="fake-molmo")
    sam_server = _start_health_server(model_name="fake-sam")
    backend_config_path = tmp_path / "backends.toml"
    _write_backend_config(
        backend_config_path,
        molmo_url=molmo_server.base_url,
        sam_url=sam_server.base_url,
        root=tmp_path,
    )
    script_path = Path("scripts/scenefunc3d/check_sidecars.sh").resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    try:
        result = subprocess.run(
            [str(script_path), str(backend_config_path)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    finally:
        molmo_server.close()
        sam_server.close()

    assert "sidecars ok" in result.stdout


def test_run_single_sample_writes_result_json_with_outcome_payload(
    tmp_path: Path,
) -> None:
    _write_scene(tmp_path / "data")
    sample_output_dir = tmp_path / "out" / "421254__desc-a"
    _write_outcome_artifacts(sample_output_dir)
    outcome = SceneFunc3dMaskOutcome(
        mask_artifact_path=sample_output_dir / "mask_artifact.json",
        mask_npz_path=sample_output_dir / "mask.npz",
        mask_ply_path=sample_output_dir / "mask.ply",
        selected_frame_ids=("000010", "000020"),
        accepted_fragment_ids=("frag-a",),
        confidence=0.87,
        uncertainties=("partial occlusion",),
    )
    executor = _FakeExecutor(outcome)
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path / "data",
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    result_path = run_single_sample(
        config,
        sample_id="421254::desc-a",
        executor=executor,
        check_sidecars=False,
    )

    assert result_path == tmp_path / "out" / "421254__desc-a" / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["task_name"] == "scenefunc3d_mask_generation"
    assert payload["sample_id"] == "421254::desc-a"
    assert payload["outcome"] == _outcome_payload(sample_output_dir)
    assert payload["turn"] == {
        "turn_id": "fake-turn",
        "status": "completed",
        "duration_ms": None,
        "usage": None,
        "input_tokens": None,
        "cached_input_tokens": None,
        "reasoning_summary": None,
        "run_home": None,
        "attempts": [],
    }
    assert executor.task_name == "scenefunc3d_mask_generation"
    assert str(tmp_path / "data" / "421254") in executor.prompt


class _FakeExecutor:
    def __init__(self, outcome: SceneFunc3dMaskOutcome) -> None:
        self._outcome = outcome
        self.task_name = ""
        self.prompt = ""

    def execute(self, task: CodexTask[ResultT]) -> CodexTaskResult[ResultT]:
        request = task.build_turn_request()
        self.task_name = task.task_name
        self.prompt = request.prompt
        return CodexTaskResult(
            task_name=task.task_name,
            outcome=cast(ResultT, self._outcome),
            turn=CodexTurnResult(
                final_response=json.dumps(self._outcome.to_payload()),
                metadata=CodexTurnMetadata(turn_id="fake-turn", status="completed"),
            ),
        )


@dataclass(frozen=True)
class _HealthServer:
    base_url: str
    server: ThreadingHTTPServer
    thread: threading.Thread

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def _start_health_server(*, model_name: str) -> _HealthServer:
    return _start_health_server_with_state(model_name=model_name, model_loaded=True)


def _start_health_server_with_state(
    *,
    model_name: str,
    model_loaded: bool,
) -> _HealthServer:
    def health_route(_payload: JsonObject) -> JsonObject:
        return {
            "status": "ok",
            "model_name": model_name,
            "model_loaded": model_loaded,
        }

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_json_handler({"/health": health_route})
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    return _HealthServer(
        base_url=f"http://127.0.0.1:{port}",
        server=server,
        thread=thread,
    )


def _sample() -> SceneFunc3dSample:
    return SceneFunc3dSample(
        sample_id="421254::desc-a",
        visit_id="421254",
        desc_id="desc-a",
        task_description="Open the lower drawer.",
        annotation_ids=("annot-a",),
        motion_hints=(
            SceneFuncMotionHint(
                motion_id="motion-a",
                annotation_id="annot-a",
                motion_type="trans",
                motion_dir=(1.0, 0.0, 0.0),
            ),
        ),
    )


def _outcome_payload(root: Path) -> dict[str, object]:
    return {
        "mask_artifact_path": str(root / "mask_artifact.json"),
        "mask_npz_path": str(root / "mask.npz"),
        "mask_ply_path": str(root / "mask.ply"),
        "selected_frame_ids": ["000010", "000020"],
        "accepted_fragment_ids": ["frag-a"],
        "confidence": 0.87,
        "uncertainties": ["partial occlusion"],
    }


def _write_outcome_artifacts(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mask_artifact.json").write_text("{}", encoding="utf-8")
    (root / "mask.npz").write_bytes(b"npz")
    (root / "mask.ply").write_text("ply\n", encoding="utf-8")


def _write_backend_config(
    path: Path,
    *,
    molmo_url: str,
    sam_url: str,
    root: Path,
) -> None:
    path.write_text(
        "\n".join(
            (
                f"molmo_url = {json.dumps(molmo_url)}",
                f"sam_url = {json.dumps(sam_url)}",
                "request_timeout_seconds = 2.0",
                f"artifact_staging_root = {json.dumps(str(root / 'stage'))}",
                f"allowed_image_roots = [{json.dumps(str(root))}]",
                f"allowed_output_roots = [{json.dumps(str(root / 'out'))}]",
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_scene(root: Path) -> None:
    scene_dir = root / "421254"
    scene_dir.mkdir(parents=True)
    (scene_dir / "421254_descriptions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "descriptions": [
                    {
                        "desc_id": "desc-a",
                        "annot_id": ["annot-a"],
                        "description": "Open the lower drawer.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_motions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "motions": [
                    {
                        "motion_id": "motion-a",
                        "annot_id": "annot-a",
                        "motion_type": "trans",
                        "motion_dir": [1.0, 0.0, 0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_annotations.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "annotations": [
                    {
                        "annot_id": "annot-a",
                        "label": "pinch_pull",
                        "indices": [3, 5, 8],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
