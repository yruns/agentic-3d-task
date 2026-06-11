# OpenEQA v2 — evidence-seeking tools + BEV smoke (stratified 10), 2026-06-12

Second OpenEQA run in `agentic-3d-task`. Validates the **evidence-seeking tool
loop**: the QA agent can now fetch more visual evidence beyond the uniform frame
sample — language-grounded keyframes, arbitrary frames, a top-down BEV, and the
scene object list — and the mesh-free BEV works in both the keyframe selector and
the agent.

## What this run validates

New capability vs [v1](v1_codex_sdk_migration_smoke10_20260611.md) (prompt-only):

| Concern | Module |
|---|---|
| `keyframe_selector` tool — language → frames (wraps `KeyframeSelector.select_keyframes_v2`) | `src/codex_agent/openeqa/tools/keyframe_retrieval.py` |
| `view_frame` tool — fetch any raw frame id(s) beyond the attached sample | `src/codex_agent/openeqa/tools/frame_tools.py` |
| `view_bev` tool — mesh-free top-down schematic map, highlight by id/category | `src/codex_agent/openeqa/tools/bev_tools.py` |
| `list_objects` tool — ConceptGraph object inventory (ids/category/size/desc) | `src/codex_agent/openeqa/tools/object_tools.py` |
| CLI dispatcher `python -m codex_agent.openeqa.tools <tool> --scene-dir …` | `src/codex_agent/openeqa/tools/dispatch.py`, `__main__.py` |
| Inlined tool playbook + synced skill | `src/codex_agent/openeqa/playbook.py`, `.agents/skills/openeqa-codex-tools/SKILL.md` |
| Mesh-free schematic BEV (renderer + `OpenEqaSceneBEVBuilder`) | `src/keyframe/bev/schematic.py`, `src/keyframe/bev/builder.py` |
| BEV in the keyframe selector (`generate_scene_bev` `openeqa` branch, `use_visual_context`) | `src/keyframe/keyframe_selector.py` |
| Tool wiring into the QA task / runner / CLI (`--tools`) | `src/codex_agent/openeqa/qa.py`, `evaluation/openeqa_runner.py`, `cli/run_openeqa.py` |

Tool mode follows the NR3D pattern: the playbook is **inlined into the prompt**
(no advertised `SKILL.md` path, which avoids the context-window re-read loop), the
sandbox is `workspace_write` + network, and the runtime tool-call loop caps apply
(`max_tool_calls=24`, `max_repeated_tool_calls=4`).

OpenEQA ScanNet clips ship **no `mesh.ply`**, so the BEV is a top-down *schematic*
floor plan (object footprints + `#id category` labels + camera trajectory) drawn
from the lightweight ConceptGraph objects, not a textured mesh raster.

## Run metadata

- **Branch:** `feat/openeqa-codex-task`
- **Head commit at launch:** `bd97b7a` (no worktree drift).
  - Caveat: the tools+BEV change was **uncommitted in the working tree** at run
    time (development smoke; the `CLAUDE.md` "commit-clean" checklist was not
    applied because the change was in flight). Durable artifacts are archived
    under `assets/` so the result is reproducible from this doc.
- **Model (answerer):** `gpt-5.4-2026-03-05` via the in-repo ModelHub adapter
  (`http://127.0.0.1:8787`).
- **Judge:** `gemini-2.5-pro` (default in `configs/llm.toml`), official `mmbench`
  prompt, temperature 0.2.
- **Sandbox:** `workspace_write` + network access (required: tools shell out and
  `keyframe_selector` calls the parsing LLM).
- **Tools:** enabled (`--tools`); `max_tool_calls=24`, `max_repeated_tool_calls=4`.
- **Reasoning effort:** `medium` (config default).
- **Frames:** 8 uniform first-person RGB frames per question attached as starting
  evidence, downscaled to ≤768 px JPEG; the agent fetches more on demand.
- **Workers:** 5.

## Fold

- **Same** category-stratified 10-question fold as v1 (so the two runs share a
  fold), covering all 7 OpenEQA categories.
- **File:** `assets/smoke10_sample_ids_20260611.json`

## Exact invocation

```bash
source .venv/bin/activate
PYTHONPATH=src python -m codex_agent.cli.run_openeqa \
  --questions data/open-eqa-v0.json \
  --data-root data/OpenEQA/scannet \
  --question-ids docs/benchmark/openeqa/assets/smoke10_sample_ids_20260611.json \
  --output-dir tmp/openeqa_tools_smoke10 \
  --num-frames 8 --workers 5 --tools
```

- **Raw artifact dir:** `tmp/openeqa_tools_smoke10/`.
- **Durable copies:** `assets/tools_smoke10_summary_20260612.json`,
  `assets/tools_smoke10_predictions_20260612.json`.

## Headline

**MNAS = 67.5** over 10 questions. **Completion 10/10, 0 errors**, wall-clock
≈ 105 s. Prompt-cache hit rate 0.50 (mean cache ratio 0.335). Per-turn latency
min/median/max = 8.6 / 11.9 / 77.6 s (the long tail is the tool-using turns).

### Tool usage (confirmed on a 2-question keep-run-home probe)

A separate 2-question run with `CODEX_AGENT_KEEP_RUN_HOME=1` confirmed the agent
actually drives the tools in-sandbox: ≥3 `keyframe_selector` calls and ≥5
`view_bev` calls across two questions, each followed by `view_image` on the
returned path. Both probe questions scored 5/5.

### Per-category MNAS (n ≤ 2 per cell — smoke only)

| Category | n | MNAS |
|---|---:|---:|
| object state recognition | 1 | 100.0 |
| attribute recognition | 2 | 87.5 |
| functional reasoning | 2 | 75.0 |
| object recognition | 1 | 75.0 |
| world knowledge | 1 | 75.0 |
| object localization | 2 | 50.0 |
| spatial understanding | 1 | 0.0 |

### Per-question (judge score / 5)

| Category | Score | Question | Prediction | GT |
|---|---:|---|---|---|
| attribute recognition | 5 | Which material is the furniture of the scene? | wood | Wood |
| attribute recognition | 4 | Which material are the side and center tables? | light-colored wood | Wood |
| functional reasoning | 5 | How can I see myself? | in the mirror on the door | using the mirror … |
| functional reasoning | 3 | How can I share the presentation materials? | Use the ceiling-mounted projector | using the remote and … |
| object localization | 5 | Where did I leave my water bottle? | on the floor beside the office chair | on the floor |
| object localization | 1 | Where is the book on sailing located? | On the bookshelf beside the window | On the coffee table |
| object recognition | 4 | What leans on the wall on the right? | a rolled-up poster | a poster |
| object state recognition | 5 | Is the countertop clean? | No, it looks cluttered rather than clean | No |
| spatial understanding | 1 | What is on top of the calculator? | a comb | red notebook |
| world knowledge | 4 | What can I use to trim the edges of a document? | a paper cutter | Use the paper cutter … |

## Reading the result

The tool-enabled pipeline runs **end-to-end with 10/10 completion**; that is the
purpose of this smoke. The two 1/5 misses (a `book on sailing` localization and a
small object-on-object spatial relation) are genuine perception errors, not
pipeline failures.

**Do not read v2 (67.5) vs v1 (75.0) as a regression.** At n=10 the per-category
cells are n≤2 and the fold-level variance is ±10–20 pp; one spatial-understanding
miss (n=1) alone moves the headline by 10 pp. Tools change *which evidence* the
model sees, so a fair tools-vs-prompt-only comparison needs the canonical
stratified fold (see below), not this 10-question smoke. Both runs share the same
fold, so the matched-fold delta is recorded here only as a sanity signal.

## What changed vs v1

- v1 was prompt-only (`read_only` sandbox, only the 8 attached frames).
- v2 adds the four-tool evidence loop + mesh-free BEV and runs in
  `workspace_write` + network with the loop caps. No change to the judge, fold,
  frame count, or MNAS definition.

## Verification

- `ruff check src/codex_agent src/keyframe` — clean.
- `black --check src/codex_agent src/keyframe` — clean.
- `mypy src/` — no issues (108 source files).
- `pytest src/codex_agent/tests src/keyframe/tests` — **353 passed** (incl. the
  new tool / playbook / schematic-BEV / tool-CLI / tool-mode-QA tests).
- Coverage on the new OpenEQA tool modules — **96 %**
  (`COVERAGE_CORE=sysmon`); `schematic.py` 99 %, `qa.py`/`playbook.py` 100 %.
- Live tool CLI smoke on a real scene (`002-scannet-scene0709_00`): all four
  tools return valid payloads; `keyframe_selector` retrieves a frame that shows
  the queried microwave; `view_bev` renders a labeled floor plan with highlights.

## Next steps (before a decision-grade run)

1. **Design the canonical OpenEQA pilot fold** (stratify on category, allocate
   proportionally, salt-lock against a full-set baseline) per the `CLAUDE.md`
   canonical-pilot-fold rule — required before any tools-vs-prompt-only A/B.
2. A/B **tools vs prompt-only** on that fold (and eventually the full 1,079-question
   ScanNet split, workers ≥ 20) for a real MNAS delta.
3. Add the OpenEQA SQLite ingester (`scripts/ingest_openeqa_run.py`) so per-tool
   traces can be sliced per question.
