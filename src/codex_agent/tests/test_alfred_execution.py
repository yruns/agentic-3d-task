"""Closed-loop execution tests for single-turn tool-driven ALFRED episodes."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from codex_agent.alfred.dataset import AlfredInstruction
from codex_agent.alfred.execution import (
    EpisodeExecutor,
    ExecutionConfig,
)
from codex_agent.alfred.perception import (
    PerceptionDetection,
    PerceptionDiagnostics,
    PerceptionObservation,
)
from codex_agent.alfred.policy import AlfredEpisodeTurnOutcome, AlfredPolicyTask
from codex_agent.alfred.simulator import SimulatorObservation
from codex_agent.alfred.tools.__main__ import call_episode_server
from codex_agent.alfred.tools.__main__ import main as tools_cli_main
from codex_agent.alfred.vocabulary import ALFRED_CATEGORIES
from codex_agent.models import CodexTaskResult, CodexTurnMetadata, CodexTurnResult
from codex_agent.tasks.base import CodexTask

OutcomeT = TypeVar("OutcomeT")


class FakeEnvironment:
    """Produce RGB/proprioception while recording primitive actions and masks."""

    def __init__(
        self,
        root: Path,
        *,
        successes: Sequence[bool] = (),
        yaw_degrees: float = 0.0,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.successes = successes
        self.yaw_degrees = yaw_degrees
        self.horizon_degrees = 0.0
        self.frame_id = 0
        self.calls: list[tuple[str, Path | None]] = []

    def observe(self, *, success: bool = True) -> SimulatorObservation:
        path = self.root / f"{self.frame_id}.png"
        Image.new("RGB", (300, 300), (200, 120, 80)).save(path)
        return SimulatorObservation(
            frame_id=self.frame_id,
            rgb_path=path,
            yaw_degrees=self.yaw_degrees,
            horizon_degrees=self.horizon_degrees,
            last_action_success=success,
            last_action_error="" if success else "Action blocked",
        )

    def step(
        self, action: str, *, mask_path: Path | None = None
    ) -> SimulatorObservation:
        index = len(self.calls)
        success = self.successes[index] if index < len(self.successes) else True
        self.calls.append((action, mask_path))
        if success and action == "RotateRight":
            self.yaw_degrees = (self.yaw_degrees + 90) % 360
        if success and action == "RotateLeft":
            self.yaw_degrees = (self.yaw_degrees - 90) % 360
        if success and action == "LookDown":
            self.horizon_degrees = min(60.0, self.horizon_degrees + 15)
        if success and action == "LookUp":
            self.horizon_degrees = max(-30.0, self.horizon_degrees - 15)
        self.frame_id += 1
        return self.observe(success=success)


class FakePerception:
    """Supply synthetic depth, masks, and normalized CLIP features per frame."""

    def __init__(self, *, include_sink_and_faucet: bool = False) -> None:
        self.include_sink_and_faucet = include_sink_and_faucet
        self.frames: list[int] = []

    def predict(
        self,
        rgb_path: Path,
        *,
        frame_id: str,
        intrinsics: NDArray[np.float64],
        output_dir: Path,
    ) -> PerceptionObservation:
        assert rgb_path.is_file()
        assert intrinsics.shape == (3, 3)
        index = int(frame_id)
        self.frames.append(index)
        output_dir.mkdir(parents=True, exist_ok=True)
        depth_path = output_dir / f"depth_{index}.npy"
        # Provide floor and object depth at 0.65m so objects are within interaction reach.
        np.save(
            depth_path,
            np.full((300, 300), 0.65, dtype=np.float32),
            allow_pickle=False,
        )

        detections: list[PerceptionDetection] = []
        img_feats: list[list[float]] = []
        txt_feats: list[list[float]] = []

        # Detection 0: parent container (Cabinet or SinkBasin)
        parent_mask = np.zeros((300, 300), dtype=np.uint8)
        parent_mask[40:260, 40:260] = 1
        parent_mask_path = output_dir / f"mask_{index}_0.npy"
        np.save(parent_mask_path, parent_mask, allow_pickle=False)
        parent_cat = "SinkBasin" if self.include_sink_and_faucet else "Cabinet"
        detections.append(
            PerceptionDetection(
                category=parent_cat,
                mask_path=parent_mask_path,
                bbox_xyxy=(40.0, 40.0, 260.0, 260.0),
                confidence=0.95,
            )
        )
        img_feats.append([1.0, 0.0, 0.0])
        txt_feats.append([1.0, 0.0, 0.0])

        # Detection 1: child object inside parent box (Mug)
        child_mask = np.zeros((300, 300), dtype=np.uint8)
        child_mask[80:220, 80:220] = 1
        child_mask_path = output_dir / f"mask_{index}_1.npy"
        np.save(child_mask_path, child_mask, allow_pickle=False)
        detections.append(
            PerceptionDetection(
                category="Mug",
                mask_path=child_mask_path,
                bbox_xyxy=(80.0, 80.0, 220.0, 220.0),
                confidence=0.92,
            )
        )
        img_feats.append([0.0, 1.0, 0.0])
        txt_feats.append([0.0, 1.0, 0.0])

        if self.include_sink_and_faucet:
            faucet_mask = np.zeros((300, 300), dtype=np.uint8)
            faucet_mask[10:38, 120:180] = 1
            faucet_mask_path = output_dir / f"mask_{index}_2.npy"
            np.save(faucet_mask_path, faucet_mask, allow_pickle=False)
            detections.append(
                PerceptionDetection(
                    category="Faucet",
                    mask_path=faucet_mask_path,
                    bbox_xyxy=(120.0, 10.0, 180.0, 38.0),
                    confidence=0.90,
                )
            )
            img_feats.append([0.0, 0.0, 1.0])
            txt_feats.append([0.0, 0.0, 1.0])

        features_path = output_dir / f"features_{index}.npz"
        np.savez(
            features_path,
            image_features=np.asarray(img_feats, dtype=np.float32),
            text_features=np.asarray(txt_feats, dtype=np.float32),
            frame_feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        )
        return PerceptionObservation(
            frame_id=frame_id,
            depth_path=depth_path,
            depth_model="da3metric-large",
            image_width=300,
            image_height=300,
            features_path=features_path,
            detections=tuple(detections),
            diagnostics=PerceptionDiagnostics(
                vocabulary_categories=len(ALFRED_CATEGORIES),
                caption_batches=5,
                candidate_boxes=len(detections),
                nms_suppressed=0,
                capped_detections=0,
                empty_masks=0,
            ),
        )


class ScriptedTurnRuntime:
    """Simulate a Codex turn that invokes ALFRED CLI tools against EpisodeToolServer."""

    def __init__(self, callback: Callable[[Path], AlfredEpisodeTurnOutcome]) -> None:
        self.callback = callback
        self.turns = 0

    def execute(self, task: CodexTask[OutcomeT]) -> CodexTaskResult[OutcomeT]:
        self.turns += 1
        assert isinstance(task, AlfredPolicyTask)
        outcome = self.callback(task.episode_dir)
        return CodexTaskResult(
            task_name=task.task_name,
            outcome=cast(OutcomeT, outcome),
            turn=CodexTurnResult(
                final_response=outcome.model_dump_json(),
                metadata=CodexTurnMetadata(
                    usage={"total": {"input_tokens": 1200, "output_tokens": 180}}
                ),
            ),
        )


def test_single_turn_episode_executes_high_level_tools_and_returns_scene_delta(
    tmp_path: Path,
) -> None:
    env = FakeEnvironment(tmp_path / "sim")
    perception = FakePerception(include_sink_and_faucet=True)
    ep_dir = tmp_path / "episode"

    def run_agent_tools(episode_dir: Path) -> AlfredEpisodeTurnOutcome:
        # 1. Query objects in memory
        listed = call_episode_server(episode_dir, "list_objects", {})
        assert listed["count"] >= 2
        objs_by_cat = {o["category"]: o["object_id"] for o in listed["objects"]}
        assert "Mug" in objs_by_cat
        assert "SinkBasin" in objs_by_cat

        # 2. Inspect high-res crop and keyframes (0 simulator steps)
        crop_res = call_episode_server(
            episode_dir, "view_crop", {"object_id": objs_by_cat["Mug"]}
        )
        assert Path(crop_res["image_path"]).is_file()

        kf_res = call_episode_server(
            episode_dir, "keyframe_selector", {"query": "Mug in sink", "k": 2}
        )
        assert "frames" in kf_res

        vf_res = call_episode_server(
            episode_dir, "view_frame", {"frame_ids": ["current"]}
        )
        assert Path(vf_res["image_path"]).is_file()

        bev_res = call_episode_server(episode_dir, "view_bev", {})
        assert Path(bev_res["image_path"]).is_file()

        # 3. Pick up the Mug
        pickup_delta = call_episode_server(
            episode_dir,
            "interact_object",
            {"action": "pickup", "object_id": objs_by_cat["Mug"]},
        )
        assert pickup_delta["status"] == "completed"
        assert pickup_delta["agent_state"]["held_object"]["category"] == "Mug"

        # 4. Run compound macro: clean_in_sink
        clean_delta = call_episode_server(
            episode_dir,
            "interact_object",
            {"action": "clean_in_sink", "object_id": objs_by_cat["SinkBasin"]},
        )
        assert clean_delta["status"] == "completed"
        assert clean_delta["agent_state"]["held_object"]["category"] == "Mug"

        # 5. Finish episode via CLI main entry point
        rc = tools_cli_main(
            [
                "finish_episode",
                "--episode-dir",
                str(episode_dir),
                "--args",
                '{"summary": "Cleaned the mug"}',
            ]
        )
        assert rc == 0
        return AlfredEpisodeTurnOutcome(
            status="completed",
            summary="Picked up mug and cleaned it in sink.",
            completed_subgoals=["pickup mug", "clean mug in sink"],
        )

    runtime = ScriptedTurnRuntime(run_agent_tools)
    executor = EpisodeExecutor(
        environment=env,
        perception=perception,
        runtime=runtime,
        output_dir=ep_dir,
        config=ExecutionConfig(max_actions=50, max_failures=10, initial_scan=False),
    )
    outcome = executor.run(
        AlfredInstruction("Clean the mug in the sink.", ()),
        env.observe(),
    )
    assert runtime.turns == 1
    assert outcome.stop_reason == "policy_stop"
    assert outcome.input_tokens == 1200
    assert outcome.output_tokens == 180
    assert (ep_dir / "memory.json").is_file()
    assert (ep_dir / "final_bev.png").is_file()
    assert (ep_dir / "tool_calls.jsonl").is_file()
    tool_records = [
        json.loads(line)
        for line in (ep_dir / "tool_calls.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [r["tool"] for r in tool_records] == [
        "list_objects",
        "view_crop",
        "keyframe_selector",
        "view_frame",
        "view_bev",
        "interact_object",
        "interact_object",
        "finish_episode",
    ]


def test_open_container_uses_unsubtracted_raw_sam_mask(tmp_path: Path) -> None:
    env = FakeEnvironment(tmp_path / "sim")
    perception = FakePerception(include_sink_and_faucet=False)
    ep_dir = tmp_path / "episode"

    def run_open_cabinet(episode_dir: Path) -> AlfredEpisodeTurnOutcome:
        listed = call_episode_server(
            episode_dir, "list_objects", {"categories": ["Cabinet"]}
        )
        cab_id = listed["objects"][0]["object_id"]
        delta = call_episode_server(
            episode_dir,
            "interact_object",
            {"action": "open", "object_id": cab_id},
        )
        assert delta["status"] == "completed"
        return AlfredEpisodeTurnOutcome(
            status="completed",
            summary="Opened cabinet",
            completed_subgoals=["open cabinet"],
        )

    runtime = ScriptedTurnRuntime(run_open_cabinet)
    executor = EpisodeExecutor(
        environment=env,
        perception=perception,
        runtime=runtime,
        output_dir=ep_dir,
        config=ExecutionConfig(max_actions=30, initial_scan=False),
    )
    executor.run(AlfredInstruction("Open the cabinet.", ()), env.observe())
    actions = [
        json.loads(line)
        for line in (ep_dir / "actions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    open_records = [a for a in actions if a["action"] == "OpenObject"]
    assert len(open_records) == 1
    assert open_records[0]["interaction_mask"]["source"] == "current_raw_sam_mask"
    saved_mask = np.load(open_records[0]["interaction_mask"]["path"])
    # Full raw parent mask is 220x220 = 48400 pixels (not subtracted by child 140x140 = 19600).
    assert int(saved_mask.sum()) == 220 * 220


def test_navigate_to_and_explore_frontier_with_early_stop(tmp_path: Path) -> None:
    env = FakeEnvironment(tmp_path / "sim")
    perception = FakePerception(include_sink_and_faucet=False)
    ep_dir = tmp_path / "episode"

    def run_nav_and_explore(episode_dir: Path) -> AlfredEpisodeTurnOutcome:
        nav_obj = call_episode_server(episode_dir, "navigate_to", {"object_id": 0})
        assert nav_obj["status"] in {"arrived_and_framed", "arrived_unverified_mask"}
        assert "scene_delta" in nav_obj

        nav_reg = call_episode_server(episode_dir, "navigate_to", {"region": [0, 1]})
        assert "scene_delta" in nav_reg

        exp_stop = call_episode_server(
            episode_dir,
            "explore_frontier",
            {"stop_on_categories": ["Mug"], "max_steps": 4},
        )
        assert exp_stop["status"] == "target_discovered"

        bad_tool = call_episode_server(episode_dir, "unknown_tool_xyz", {})
        assert "error" in bad_tool

        return AlfredEpisodeTurnOutcome(
            status="completed",
            summary="Navigated and explored",
            completed_subgoals=["navigate", "explore"],
        )

    runtime = ScriptedTurnRuntime(run_nav_and_explore)
    executor = EpisodeExecutor(
        environment=env,
        perception=perception,
        runtime=runtime,
        output_dir=ep_dir,
        config=ExecutionConfig(max_actions=40, initial_scan=True),
    )
    outcome = executor.run(
        AlfredInstruction("Find the mug.", ()),
        env.observe(),
    )
    assert outcome.stop_reason == "policy_stop"
    assert outcome.decisions >= 4


def test_episode_executor_resumes_after_mid_turn_409_interruption(
    tmp_path: Path, monkeypatch: object
) -> None:
    import codex_agent.alfred.execution as exec_mod
    from codex_agent.errors import CodexTurnError

    monkeypatch.setattr(exec_mod.time, "sleep", lambda _: None)  # type: ignore[attr-defined]
    env = FakeEnvironment(tmp_path / "sim")
    perception = FakePerception(include_sink_and_faucet=True)
    ep_dir = tmp_path / "episode"
    attempt_counter = 0

    def run_with_409_on_first_turn(episode_dir: Path) -> AlfredEpisodeTurnOutcome:
        nonlocal attempt_counter
        attempt_counter += 1
        if attempt_counter == 1:
            listed = call_episode_server(
                episode_dir, "list_objects", {"query": "mug sink garbage"}
            )
            assert listed["count"] >= 2
            mug_id = next(
                o["object_id"] for o in listed["objects"] if o["category"] == "Mug"
            )
            pickup_delta = call_episode_server(
                episode_dir,
                "interact_object",
                {"action": "pickup", "object_id": mug_id},
            )
            assert pickup_delta["status"] == "completed"
            raise CodexTurnError(
                "unexpected status 409 Conflict: Session has an unfinished or indeterminate request"
            )

        # Second turn (continuation): verify held_object is preserved in memory
        listed2 = call_episode_server(episode_dir, "list_objects", {})
        assert listed2["held_object_id"] is not None
        sink_id = next(
            o["object_id"] for o in listed2["objects"] if o["category"] == "SinkBasin"
        )
        put_delta = call_episode_server(
            episode_dir,
            "interact_object",
            {"action": "put", "object_id": sink_id},
        )
        assert put_delta["status"] == "completed"
        call_episode_server(
            episode_dir, "finish_episode", {"summary": "Resumed and placed mug"}
        )
        return AlfredEpisodeTurnOutcome(
            status="completed",
            summary="Resumed after 409 and placed mug in sink.",
            completed_subgoals=["pickup mug", "place mug in sink"],
        )

    runtime = ScriptedTurnRuntime(run_with_409_on_first_turn)
    executor = EpisodeExecutor(
        environment=env,
        perception=perception,
        runtime=runtime,
        output_dir=ep_dir,
        config=ExecutionConfig(max_actions=40, initial_scan=False),
    )
    outcome = executor.run(
        AlfredInstruction("Put the mug in the sink.", ()),
        env.observe(),
    )
    assert runtime.turns == 2
    assert outcome.stop_reason == "policy_stop"
    assert outcome.decisions == 5
