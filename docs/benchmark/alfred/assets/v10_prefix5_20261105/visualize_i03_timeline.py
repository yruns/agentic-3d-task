"""Visualize a completed episode's real pickup and held-state updates from disk."""

import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

root = Path(__file__).resolve().parents[2]
episode = root / "outputs/alfred/ALFRED_V9_20261105_pilot15_i03/episodes/0001"
attempt = json.loads((episode / "attempt.json").read_text())
assert attempt["status"] == "completed"
execution = json.loads((episode / "execution.json").read_text())
final_memory = json.loads((episode / "memory.json").read_text())
actions = [json.loads(line) for line in (episode / "actions.jsonl").read_text().splitlines()]
frames = {row["observation"]["frame_id"]: row for row in (json.loads(line) for line in (episode / "observations.jsonl").read_text().splitlines())}


def artifact(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def actor_context(decision):
    path = episode / "decisions" / f"{decision:04d}" / "request.json"
    prompt = json.loads(path.read_text())["prompt"]
    return json.loads(prompt[prompt.index('\n\n{"instruction"') + 2:])["observations"]


before, after = actor_context(4), actor_context(5)
before_object = next(item for item in before["scene_memory"]["objects"] if item["object_id"] == 48)
after_object = next(item for item in after["scene_memory"]["objects"] if item["object_id"] == 48)
assert before["scene_memory"]["held_object_id"] is None
assert actions[14]["action"] == "PickupObject" and actions[14]["success"] is True
assert after["scene_memory"]["held_object_id"] == final_memory["held_object_id"] == 48
assert after_object["state"] == "held" and after_object["geometry_valid"] is False
stage_definitions = [
    ("Before successful pickup", 4, 14, episode / "decisions/0004/current.png", episode / "decisions/0004/bev.png", [
        "Decision 4 / action 15: PickupObject 48",
        "Current Ladle detection: confidence 0.486",
        "Held object before: none",
        "Result: success; next observed frame 15",
    ]),
    ("Held-state update", 5, 15, episode / "decisions/0005/current.png", episode / "decisions/0005/bev.png", [
        "Next decision receives held_object_id=48",
        "Object 48 state=held; geometry_valid=false",
        "Old location retained only as last-known",
        "Next action 16: PutObject 36 -> bad mask",
    ]),
    ("Placement remains unresolved", 16, 22, episode / "decisions/0016/current.png", episode / "decisions/0016/bev.png", [
        "Decision 16 / action 23: PutObject 94",
        "Current label: SinkBasin, confidence 0.301",
        "Detection is a 9.5 px-wide right-edge region",
        "Result: target is NOT a receptacle",
        "Held object remains 48",
    ]),
    ("Terminal observation", None, 74, episode / "simulator/frames/000074.png", episode / "final_bev.png", [
        "After decision 60 / 74 total actions",
        "Held object remains 48 (Ladle)",
        "6 PutObject attempts, all failed",
        "No ToggleObjectOn/Off action occurred",
        "Stop: decision budget; score 1/3 GC",
    ]),
]
fig, axes = plt.subplots(3, 4, figsize=(18, 11.5), gridspec_kw={"height_ratios": [1, 1, .62]})
fig.patch.set_facecolor("#f7f8fa")
stages = []
for column, (title, decision, frame, rgb_path, bev_path, notes) in enumerate(stage_definitions):
    for row_index, path in enumerate((rgb_path, bev_path)):
        axes[row_index, column].imshow(Image.open(path))
        axes[row_index, column].axis("off")
    axes[0, column].set_title(f"{title}\nFrame {frame}", fontsize=13, weight="bold")
    axes[1, column].set_title("Saved BEV" if decision is not None else "Saved final BEV", fontsize=10, color="#555555")
    axes[2, column].text(0, 1, "\n".join(textwrap.fill(line, 43) for line in notes), va="top", transform=axes[2, column].transAxes, fontsize=10, linespacing=1.4)
    axes[2, column].axis("off")
    stages.append({"title": title, "decision": decision, "frame": frame, "rgb": artifact(rgb_path), "bev": artifact(bev_path), "notes": notes})
fig.suptitle("ALFRED i03: successful pickup updates memory, but rinsing and placement remain incomplete", fontsize=17, weight="bold", y=.99)
fig.text(.5, .942, "Goal (original wording): Rinse off a laddle and move it to the table.", ha="center", fontsize=12)
fig.text(.5, .027, "Terminal: 1/3 goal conditions; 74 actions, 60 decisions.  Pickup succeeded; the ladle remains held in the final observation.", ha="center", fontsize=11)
fig.text(.5, .009, "Saved actor-visible frames and memory plus final RGB/BEV only. No hidden object metadata, replay, new perception or mapping was used.", ha="center", fontsize=9, color="#555555")
fig.subplots_adjust(top=.90, bottom=.075, left=.025, right=.99, hspace=.18, wspace=.10)
output = root / "tmp/alfred_v9/i03_visual_timeline.png"
fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
report = {
    "schema_version": 1,
    "scope": "Post-terminal visualization from saved RGB/BEV, action logs and actor memory; no simulator terminal metadata objects read, no model/GPU/replay/mapping calls",
    "run_id": "ALFRED_V9_20261105_pilot15_i03", "episode_id": attempt["episode_id"],
    "score": attempt["score"], "execution": execution,
    "output": artifact(output), "stages": stages,
    "pickup_transition": {
        "action": actions[14], "before_frame": 14, "after_frame": 15,
        "before_held_object_id": before["scene_memory"]["held_object_id"],
        "after_held_object_id": after["scene_memory"]["held_object_id"],
        "before_object": before_object, "after_object": after_object,
        "request_before": artifact(episode / "decisions/0004/request.json"),
        "request_after": artifact(episode / "decisions/0005/request.json"),
        "interaction_mask": artifact(actions[14]["interaction_mask"]["path"]),
    },
    "placement_actions": [item for item in actions if item["action"] == "PutObject"],
    "final_held_object": next(item for item in final_memory["objects"] if item["object_id"] == 48),
    "artifacts": [artifact(episode / name) for name in ("attempt.json", "execution.json", "actions.jsonl", "observations.jsonl", "memory.json")],
    "interpretation_limits": [
        "Pickup success and action-derived memory update are directly recorded; no claim that geometry independently verified the held identity",
        "Failed placement plus lack of faucet toggles does not diagnose a unique perception/navigation/policy cause",
        "The official 1/3 score is copied from the completed attempt, not recomputed by this visualization",
    ],
}
with (root / "tmp/alfred_v9/i03_visual_timeline_sources.json").open("x") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
print(output)
