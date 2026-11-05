"""CPU-only decision replays from immutable saved actor artifacts."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/alfred/visualizations_prefix5_20261105"
OUTPUT.mkdir(parents=True, exist_ok=True)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SIZE = (1600, 1080)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence(path):
    return {"path": str(path), "sha256": digest(path)}


def text(draw, position, value, size=22, bold=False, fill="#162535"):
    draw.text(position, value, font=ImageFont.truetype(BOLD if bold else FONT, size), fill=fill)


def wrapped(draw, xy, value, width, size=21, line_height=28):
    font = ImageFont.truetype(FONT, size)
    x, y = xy
    line = ""
    for word in value.split():
        candidate = word if not line else line + " " + word
        if draw.textlength(candidate, font=font) > width and line:
            draw.text((x, y), line, font=font, fill="#263445")
            y += line_height
            line = word
        else:
            line = candidate
    if line:
        draw.text((x, y), line, font=font, fill="#263445")
    return y + line_height


def image_panel(canvas, path, bounds, crop=None):
    with Image.open(path) as source:
        image = source.convert("RGB")
        if crop is not None:
            image = image.crop(crop)
        x, y, width, height = bounds
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (width-image.width)//2, y + (height-image.height)//2))


def crop_bev(path, object_count):
    with Image.open(path) as image:
        width, height = image.size
    # In the saved renderer, H = 80 + map_H + 48 + 20*(legend_rows+2).
    # Keep through map bottom+8; the catalog starts at map bottom+16.
    bottom = height - 40 - 20 * (min(24, object_count) + 2)
    if bottom <= 80 or bottom > height:
        raise ValueError(f"Unexpected saved BEV shape/count: {path}")
    return (0, 0, width, bottom)


def compact_feedback(value):
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return str(value)
    if isinstance(parsed, dict) and "action" in parsed:
        return (
            f"Action {parsed.get('index')}: {parsed['action']}; object_id={parsed.get('object_id')}; "
            f"success={parsed.get('success')}; error={parsed.get('error') or '(none)'}"
        )
    return json.dumps(parsed, ensure_ascii=False)


def render_frame(path, *, title, subtitle, rgb_path, bev_path, objects, command, feedback, terminal=False):
    canvas = Image.new("RGB", SIZE, "#f6f8fb")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, SIZE[0], 112), fill="#152b45")
    text(draw, (28, 16), title, 28, True, "white")
    text(draw, (28, 57), subtitle, 22, False, "#dce7f3")
    text(draw, (28, 88), "decision replay, not real-time  |  1 second per decision  |  saved RGB / BEV only", 16, False, "#afc9e6")
    text(draw, (30, 120), "Final RGB" if terminal else "Current labeled RGB supplied to actor", 20, True)
    text(draw, (810, 120), "Saved final BEV (catalog removed)" if terminal else "Saved decision BEV (catalog removed)", 20, True)
    crop = crop_bev(bev_path, objects)
    image_panel(canvas, rgb_path, (30, 155, 730, 620))
    image_panel(canvas, bev_path, (810, 155, 760, 620), crop)
    draw.line((30, 796, 1570, 796), fill="#cbd6e0", width=2)
    text(draw, (30, 811), "TERMINAL SUMMARY — no next command" if terminal else "UPCOMING COMMAND — result is not shown here", 21, True, "#175e93")
    wrapped(draw, (30, 845), command, 1520, 23, 31)
    text(draw, (30, 922), "FINAL STATUS" if terminal else "PREVIOUS FEEDBACK — available before the upcoming command", 20, True, "#815526")
    wrapped(draw, (30, 955), feedback, 1520, 20, 27)
    canvas.save(path)
    return crop


manifest = {
    "schema_version": 1,
    "scope": "Post-terminal decision replay of saved artifacts; no new inference, simulation or mapping",
    "timing": "1 second per decision plus 1 terminal frame; output 24 fps by frame duplication; not real-time",
    "bev_crop_rule": "PIL crop (0,0,width,height-40-20*(min(24,object_count)+2)); derived from saved renderer dimensions; keeps header/map+8px and removes catalog beginning map+16px",
    "crop_reference": evidence(ROOT / "src/codex_agent/alfred/rendering.py"),
    "frame_size": list(SIZE), "runs": [], "static_copies": [],
}
for suffix in ("i02", "i03"):
    episode = ROOT / f"outputs/alfred/ALFRED_V9_20261105_pilot15_{suffix}/episodes/0001"
    attempt = json.loads((episode / "attempt.json").read_text())
    assert attempt["status"] == "completed"
    execution = json.loads((episode / "execution.json").read_text())
    decisions = sorted((episode / "decisions").glob("*/response.json"))
    assert len(decisions) == execution["decisions"]
    frames_dir = ROOT / f"tmp/alfred_v9/replay_frames_{suffix}"
    frames_dir.mkdir(exist_ok=False)
    run = {"run_id": f"ALFRED_V9_20261105_pilot15_{suffix}", "episode_id": attempt["episode_id"], "decision_count": len(decisions), "frames": []}
    for ordinal, response_path in enumerate(decisions):
        directory = response_path.parent
        request_path = directory / "request.json"
        prompt = json.loads(request_path.read_text())["prompt"]
        request = json.loads(prompt[prompt.index('\n\n{"instruction"') + 2:])
        observations = request["observations"]
        memory = observations["scene_memory"]
        decision = json.loads(response_path.read_text())["decision"]
        command = json.dumps(decision["command"], ensure_ascii=False, separators=(", ", ": "))
        feedback = compact_feedback(observations["feedback"])
        rgb_path, bev_path = directory / "current.png", directory / "bev.png"
        frame_path = frames_dir / f"{ordinal:04d}.png"
        count = len(memory["objects"])
        crop = render_frame(
            frame_path,
            title=f"ALFRED {suffix} | decision {ordinal + 1}/{len(decisions)}",
            subtitle=f"Observation frame {memory['latest_frame_id']}  |  held ID: {memory['held_object_id']}  |  object nodes: {count}",
            rgb_path=rgb_path, bev_path=bev_path, objects=count,
            command=command, feedback=feedback,
        )
        run["frames"].append({"ordinal": ordinal, "decision": ordinal + 1, "observation_frame": memory["latest_frame_id"], "held_object_id": memory["held_object_id"], "object_nodes": count, "upcoming_command": decision["command"], "previous_feedback": feedback, "rgb": evidence(rgb_path), "bev": evidence(bev_path), "bev_crop_xyxy": crop, "request": evidence(request_path), "response": evidence(response_path), "rendered_frame_sha256": digest(frame_path)})
    final_memory = json.loads((episode / "memory.json").read_text())
    observations = [json.loads(line) for line in (episode / "observations.jsonl").read_text().splitlines()]
    final_rgb = Path(observations[-1]["observation"]["rgb_path"])
    final_bev = episode / "final_bev.png"
    score = attempt["score"]
    count = len(final_memory["objects"])
    final_path = frames_dir / f"{len(decisions):04d}.png"
    crop = render_frame(
        final_path, title=f"ALFRED {suffix} | terminal observation",
        subtitle=f"Observation frame {final_memory['latest_frame_id']}  |  held ID: {final_memory['held_object_id']}  |  object nodes: {count}",
        rgb_path=final_rgb, bev_path=final_bev, objects=count,
        command=f"Success: {score['success']}  |  Goal conditions: {score['completed_goal_conditions']}/{score['total_goal_conditions']}  |  Stop: {execution['stop_reason']}",
        feedback=f"{execution['actions']} actions; {execution['failures']} failed actions; {execution['decisions']} decisions. This is the saved final observation and official recorded score.", terminal=True,
    )
    run["frames"].append({"ordinal": len(decisions), "decision": None, "observation_frame": final_memory["latest_frame_id"], "held_object_id": final_memory["held_object_id"], "object_nodes": count, "upcoming_command": None, "rgb": evidence(final_rgb), "bev": evidence(final_bev), "bev_crop_xyxy": crop, "attempt": evidence(episode / "attempt.json"), "execution": evidence(episode / "execution.json"), "rendered_frame_sha256": digest(final_path)})
    video = OUTPUT / f"{suffix}_decision_replay.mp4"
    if video.exists():
        raise FileExistsError(video)
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-n", "-framerate", "1", "-i", str(frames_dir / "%04d.png"), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "2", "-pix_fmt", "yuv420p", "-r", "24", "-movflags", "+faststart", str(video)]
    with (ROOT / f"tmp/alfred_v9/{suffix}_replay_ffmpeg.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,width,height,nb_frames,r_frame_rate", "-of", "json", str(video)], text=True))
    assert abs(float(probe["format"]["duration"]) - len(run["frames"])) < .001
    run.update(video=evidence(video), duration_seconds=len(run["frames"]), encoding_command=command, ffprobe=probe)
    manifest["runs"].append(run)
    print(suffix, "complete", video, flush=True)
for name in ("i02_visual_timeline.png", "i03_visual_timeline.png", "i02_saved_mask_comparison.png", "i02_visual_timeline_sources.json", "i03_visual_timeline_sources.json", "i02_terminal_failure_cases.json", "i02_terminal_failure_cases.md"):
    source, target = ROOT / "tmp/alfred_v9" / name, OUTPUT / name
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(source, target)
    assert digest(source) == digest(target)
    manifest["static_copies"].append({"source": evidence(source), "copy": evidence(target)})
with (OUTPUT / "replay_manifest.json").open("x") as stream:
    json.dump(manifest, stream, indent=2)
    stream.write("\n")
print(OUTPUT / "replay_manifest.json", flush=True)
