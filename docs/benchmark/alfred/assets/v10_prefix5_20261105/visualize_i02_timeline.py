"""Shareable storyboard from saved current RGB/BEV; no mapping or model calls."""

import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

root = Path(__file__).resolve().parents[2]
episode = root / "outputs/alfred/ALFRED_V9_20261105_pilot15_i02/episodes/0001"
rows = [json.loads(line) for line in (episode / "observations.jsonl").read_text().splitlines()]
actions = [json.loads(line) for line in (episode / "actions.jsonl").read_text().splitlines()]
stages = ((8, 11), (21, 20), (29, 27), (38, 33))
fig, axes = plt.subplots(3, 4, figsize=(18, 11.5), gridspec_kw={"height_ratios": [1, 1, .62]})
fig.patch.set_facecolor("#f7f8fa")
sources = []
for column, (decision_index, action_index) in enumerate(stages):
    decision_dir = episode / "decisions" / f"{decision_index:04d}"
    prompt = json.loads((decision_dir / "request.json").read_text())["prompt"]
    current = json.loads(prompt[prompt.index('\n\n{"instruction"') + 2:])["observations"]
    action = actions[action_index - 1]
    frame = rows[action["source_frame"]]
    det_index = action["interaction_mask"]["source_detection_index"]
    detection = frame["prediction"]["detections"][det_index]
    memory = next(item for item in current["scene_memory"]["objects"] if item["object_id"] == action["object_id"])
    for row_index, name in enumerate(("current.png", "bev.png")):
        path = decision_dir / name
        axes[row_index, column].imshow(Image.open(path))
        axes[row_index, column].axis("off")
        sources.append({"decision": decision_index, "action": action_index, "source_frame": action["source_frame"], "kind": name, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    axes[0, column].set_title(f"Decision {decision_index} / action {action_index}\nCurrent frame {action['source_frame']}", fontsize=13, weight="bold")
    axes[1, column].set_title("Saved BEV at this decision", fontsize=10, color="#555555")
    prior = current["feedback"]
    if prior.startswith("{"):
        prior_json = json.loads(prior)
        prior = f"{prior_json['action']}: success={prior_json['success']}"
    lines = [
        f"OpenObject  |  memory ID {action['object_id']}",
        f"Memory label: {memory['category']}",
        f"Current detector: {detection['category']} ({detection['confidence']:.2f})",
        f"Base position: {tuple(frame['agent_position'][:2])} m",
        f"Objects in memory: {len(current['scene_memory']['objects'])}",
        "Previous feedback: " + prior,
        "Result: " + action["error"],
    ]
    wrapped = "\n".join(textwrap.fill(line, 43) for line in lines)
    axes[2, column].text(0, 1, wrapped, va="top", transform=axes[2, column].transAxes, fontsize=10, linespacing=1.3)
    axes[2, column].axis("off")
fig.suptitle("ALFRED i02: repeated opening attempts while approach remained unverified", fontsize=18, weight="bold", y=.99)
fig.text(.5, .942, "Goal: place a chilled martini glass containing a fork on the table.", ha="center", fontsize=12)
fig.text(.5, .027, "Terminal: 0/3 goal conditions; failure budget after 33 actions and 38 decisions.  Only 2 MoveAhead actions; 3 distinct base positions.", ha="center", fontsize=11)
fig.text(.5, .009, "All panels are saved actor inputs. No replay, new perception or mapping. Historical evaluator-only identity checks are kept separately in the case JSON.", ha="center", fontsize=9, color="#555555")
fig.subplots_adjust(top=.90, bottom=.075, left=.025, right=.99, hspace=.18, wspace=.10)
output = root / "tmp/alfred_v9/i02_visual_timeline.png"
fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
manifest = {
    "scope": "Visualization from completed episode actor-visible saved RGB/BEV and logs only",
    "run_id": "ALFRED_V9_20261105_pilot15_i02",
    "output": str(output),
    "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "sources": sources,
    "no_inference_or_mapping_recomputation": True,
}
(root / "tmp/alfred_v9/i02_visual_timeline_sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(output)
