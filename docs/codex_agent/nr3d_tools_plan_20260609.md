# NR3D agent tooling — design & implementation plan

Status: **design agreed**, implementation pending. 2026-06-09.

Builds on:
- `view_image_adapter_fix_20260609.md` (mid-turn image viewing now works)
- the prompt-only baseline in `src/codex_agent/nr3d/grounding.py`

## Architecture: agent-driven CLI tools + `view_image`

The agent runs an evidence loop **inside one `thread.run` (one runtime
`execute`)**:

1. shell out to NR3D CLI tools in the sandbox;
2. text tools print JSON to stdout; frame tools also write an annotated PNG into
   the run's scratch dir and print its path;
3. the agent calls the built-in `view_image` on that path to see the pixels;
4. it iterates until it emits the structured `proposal_id`.

This keeps the original "equip the agent with tools, let it decide" intent,
gives it real first-person pixels, and stays cheap (one execute; prompt prefix
stays cacheable).

Runtime changes required (`src/codex_agent`):
- sandbox `read_only → workspace_write` for the grounding turn (tools write
  annotated frames + the agent shells out);
- the NR3D prompt/skill must teach the CLI entrypoint + the
  "always `view_image` a frame before citing it" rule;
- tools must be importable/runnable inside the sandbox (CLI module on PYTHONPATH).

## Confirmed local data assets (per prepared scene `pack_nr3d_v9_catalog_first/`)

- `scene_catalog.json`: proposals with `frame_views` → per-frame `bbox_2d` +
  `raw_rgb_path`; plus `enrichment` (description/color/location/nearby_objects).
- `camera_trajectory.json`: frame_id → `[x, y, yaw]`.
- `proposals.jsonl`, `visibility.json`, `bev/scene_bev_nr3d.png` (+ `.view.json`),
  `raw/*-rgb.png` (+ depth).

Everything the tools need already exists; tools are wrappers over existing
`src/keyframe` capabilities (`select_keyframes_v2`, `spatial/checker.py`,
`bev/`, `parsing/parser.py`).

## Tool set (core 6)

| Tool | Purpose | Returns | Reuses |
| --- | --- | --- | --- |
| `summarize_query` | structure the referring expression (target/anchor/relation/attrs) | JSON | `parsing/parser.py` (`parse_query`) |
| `keyframe_selector` | fetch ≤3 first-person frames for the query | frame paths + visible ids | `select_keyframes_v2` |
| `select_by_proposal` | fetch frames showing given candidate ids (esp. **co-visible** target+anchor) | frame paths + visible ids | `visibility.json` |
| `mark_frame_with_bbox` | render a frame with high-contrast boxes on named ids; **emit PNG for `view_image`** + 2D left→right text | PNG path + 2D layout | `frame_views.bbox_2d` |
| `compare_proposals_spatial` | deterministic 3D ranking + co-viewed 2D left/right votes | ranked JSON | `spatial/checker.py` |
| `inspect_proposal` | full enrichment (color/description/nearby) beyond the truncated note | JSON | `scene_catalog` |

Backlog (add if core plateaus): `list_frame_proposals`,
`compare_candidates_to_anchors`, `view_bev(highlight)`.

`select_by_text` (heavy Stage-1 CLIP/KeyframeSelector retrieval) is folded into
`keyframe_selector`; no separate heavyweight tool.

## CLI shape

Single dispatcher, runnable inside the sandbox:

```
python -m codex_agent.nr3d.tools <tool> --scene-dir <pack_dir> [--json '<args>']
```

- text tools → compact JSON on stdout;
- frame tools → write `*.png` into a per-run scratch dir under the workspace and
  print the absolute path (so the agent can `view_image` it);
- reconstructs state from the prepared artifacts via the existing `Nr3dScene`
  loader (extended to read `frame_views` + `camera_trajectory.json`); no
  serialized "state blob".

## Key design points (carried over from the original, validated here)

- **Fetch must pair with annotate.** A raw RGB frame doesn't tell the agent
  which object is `#7`; `keyframe_selector`/`select_by_proposal` (fetch) must be
  paired with `mark_frame_with_bbox` (label) + `view_image` (see).
- **Co-visible frames are the spatial workhorse.** Left/right is viewpoint-
  dependent; resolve it on a frame that shows target+anchor together
  (`select_by_proposal(require_all=True)`), not from 3D coords alone.
- **Spatial = 3D + 2D.** Metric relations (near/far/above/below/between) from 3D
  centers; view-dependent (left/right/front/back) from co-viewed 2D centers.
- **Anti-hallucination guard.** A frame counts as evidence only after the agent
  has `view_image`-d it; the skill must forbid citing an un-viewed frame.

## Phased implementation

- **P0 — text tools + wiring** (lowest risk, no image path): `summarize_query`,
  `inspect_proposal`, `compare_proposals_spatial`; CLI dispatcher; extend
  `Nr3dScene`; flip sandbox to `workspace_write`; rewrite the skill to teach the
  CLI + loop; unit tests. A/B vs prompt-only to isolate the lift from exact
  geometry.
- **P1 — visual loop**: `keyframe_selector`, `select_by_proposal`,
  `mark_frame_with_bbox`; prompt/skill guidance to `view_image` the marked
  frame; scratch-dir management + cleanup.
- **P2 — eval**: run the canonical strat600 fold, ingest to SQLite, write a
  `docs/benchmark/nr3d/` version doc; compare against the prompt-only baseline.

## Risks

- **Selector recall** caps single-pass quality (~80% target hit@3); mitigate
  with co-visible `select_by_proposal` frames, not just text retrieval.
- **Turn/cost** grows with the agent loop; prefix cache covers the stable
  prefix, and the skill should route tools by query type rather than always-on.
- **Hallucination** if the agent cites un-viewed frames — enforced by the
  `view_image` guard above.

## Open question for the operator

Whether some always-useful text tools (`summarize_query`,
`compare_proposals_spatial`) should be **pre-computed by the runtime into the
prompt** (saves the agent a tool round-trip) vs. left as agent-called CLI tools.
Default plan: agent-called, but cheap to pre-compute later if turn budget bites.
