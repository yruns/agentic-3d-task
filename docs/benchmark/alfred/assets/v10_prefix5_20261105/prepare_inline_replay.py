"""Build compact replay data from saved actor evidence, without model inference."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path("/home/ysh/codebase/agentic-3d-task")
OUTPUT = ROOT / "outputs/alfred/visualizations_prefix5_20261105"


def encode_image(path: Path, *, object_count: int | None = None) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if object_count is not None:
        legend_height = 48 + 20 * (min(24, object_count) + 2)
        image = image.crop((0, 64, image.width, image.height - legend_height + 8))
    image.thumbnail((420, 450), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=60, method=6)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode()


def command_label(command: dict[str, object]) -> str:
    kind = str(command["kind"])
    action = command.get("interaction_action", command.get("motion_action", kind))
    target = command.get("target")
    suffix = " " + json.dumps(target, ensure_ascii=False) if target else ""
    return str(action) + suffix


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, object]] = []
    provenance: list[dict[str, str]] = []
    for index, decisions, title in (
        (2, [1, 8, 15, 21, 29, 38], "i02 · 寻找高脚杯与叉子"),
        (3, [1, 4, 5, 15, 35, 60], "i03 · 清洁与放置长柄勺"),
    ):
        directory = (
            ROOT
            / f"outputs/alfred/ALFRED_V9_20261105_pilot15_i{index:02d}/episodes/0001"
        )
        attempt = json.loads((directory / "attempt.json").read_text())
        snapshots = []
        for decision in decisions:
            folder = directory / "decisions" / f"{decision:04d}"
            request_path = folder / "request.json"
            request = json.loads(request_path.read_text())
            payload = json.loads(request["prompt"].splitlines()[-1])
            observation = payload["observations"]
            memory = observation["scene_memory"]
            response = json.loads((folder / "response.json").read_text())
            count = len(memory["objects"])
            snapshots.append(
                {
                    "label": f"决策 {decision}",
                    "frame": memory["latest_frame_id"],
                    "decision": decision,
                    "nodes": count,
                    "held": memory["held_object_id"],
                    "regions": memory["observed_region_count"],
                    "action": command_label(response["decision"]["command"]),
                    "feedback": str(observation["feedback"]),
                    "rgb": encode_image(folder / "current.png"),
                    "bev": encode_image(folder / "bev.png", object_count=count),
                }
            )
            for path in [
                request_path,
                folder / "response.json",
                folder / "current.png",
                folder / "bev.png",
            ]:
                provenance.append(
                    {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        memory = json.loads((directory / "memory.json").read_text())
        last = json.loads(
            (directory / "observations.jsonl").read_text().splitlines()[-1]
        )
        score = attempt["score"]
        snapshots.append(
            {
                "label": "终态",
                "decision": None,
                "frame": memory["latest_frame_id"],
                "nodes": len(memory["objects"]),
                "held": memory["held_object_id"],
                "regions": memory["observed_region_count"],
                "action": f"官方目标条件 {score['completed_goal_conditions']}/{score['total_goal_conditions']}",
                "feedback": json.loads((directory / "execution.json").read_text())["stop_reason"],
                "rgb": encode_image(Path(last["observation"]["rgb_path"])),
                "bev": encode_image(
                    directory / "final_bev.png", object_count=len(memory["objects"])
                ),
            }
        )
        trajectory = [
            {
                "frame": row["observation"]["frame_id"],
                "x": round(row["agent_position"][0], 3),
                "y": round(row["agent_position"][1], 3),
            }
            for row in map(
                json.loads, (directory / "observations.jsonl").read_text().splitlines()
            )
        ]
        episodes.append(
            {
                "id": f"i{index:02d}",
                "title": title,
                "snapshots": snapshots,
                "trajectory": trajectory,
            }
        )
    output = OUTPUT / "inline_replay_data.json"
    output.write_text(json.dumps(episodes, ensure_ascii=False, separators=(",", ":")))
    (OUTPUT / "inline_replay_sources.json").write_text(
        json.dumps(
            {
                "scope": "Saved actor decision snapshots and final evidence, no recomputation or model call",
                "selection": "i02 container search and i03 observed pickup/state update",
                "image_transform": "BEV catalog cropped using the recorded object count and frozen rendering layout; proportional downsampling and WEBP compression only",
                "sources": provenance,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"bytes": output.stat().st_size, "episodes": len(episodes)}))


if __name__ == "__main__":
    main()
