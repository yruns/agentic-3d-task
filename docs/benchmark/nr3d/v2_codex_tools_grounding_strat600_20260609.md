# NR3D v2 — `CodexAgentRuntime` tool-using visual grounding, strat600 (PARTIAL 169/600)

Second **canonical strat600** evaluation of the Codex Agent SDK runtime
(`src/codex_agent/`). This adds the **in-turn evidence tools** that the
[v1 prompt-only run](v1_codex_runtime_grounding_strat600_20260608.md) explicitly
left as future work. The agent now actively fetches first-person frames, ranks
spatial relations, and views annotated frames/BEV maps before committing to one
`proposal_id`, instead of reasoning purely from the printed catalog + a single
BEV image.

> **Status: PARTIAL — run halted at 169/600.** The run was stopped manually
> after throughput collapsed (see [Why the run was halted](#why-the-run-was-halted-throughput-collapse)).
> The headline below is computed on the **169 completed** samples, which happen
> to track the fold's tier composition within ~2 pp (so the partial numbers are
> indicative, not just survivorship-skewed). Durable per-tier asset:
> `assets/codex_tools_grounding_strat600_PARTIAL169_gpt54_20260609.json`.
>
> **One-line takeaway:** on this partial set the in-turn tools land **at the v1
> prompt-only floor on every slice** (Overall 63.3% vs 63.83%; View-Dep 50.0%
> vs 51.66%) — i.e. the tools did **not** lift accuracy, despite ~2–10× more
> wall time per case. **Trace analysis explains why the extra time was wasted:**
> in the slow tail, **91% of the agent's shell calls were `sed`/`cat` re-reads of
> its own `SKILL.md`** (median 58×/case), not tool use — a re-read loop the
> `project_doc_max_bytes=0` fix did not stop. The spatial/co-visible tools *are*
> invoked (19/22 View-Dep used a spatial tool), they are just drowned by the
> loop. See [Trace analysis](#trace-analysis-the-skillmd-re-read-loop-answers-open-question-1).
> This is a negative result with a clear, fixable cause.

## Pre-run

| Item | Value |
|---|---|
| Repo / branch | `agentic-3d-task` / `master` |
| Head commit at launch / halt | `004e22e` / `004e22e` (no commits during run; tool layer uncommitted both times) |
| Completed / halted | **169 / 600** (manually stopped — throughput collapse) |
| Wall time before halt | ~1 h 50 min (13:49 → 15:39) |
| Working tree | tool layer **uncommitted** at run time (`src/codex_agent/nr3d/tools/`, `scene_assets.py`, loader/runtime/grounding/CLI edits, `.agents/skills/nr3d-codex-tools/`); no worktree drift |
| Fold | canonical `v9_3_strat600` (`tmp/nr3d_case600/sample_ids.json`) |
| Fold size | 600 (Easy 290 / Hard 310 ; View-Dep 211 / View-Indep 389) |
| Data root | `3DVLMReasoning/data/nr3d/scannet` (shared prepared packs) |
| Pack | `pack_nr3d_v9_catalog_first` |
| Backend | `gpt-5.4-2026-03-05` via Codex `modelhub_adapter` (`127.0.0.1:8787`, `wire_api=responses`, chat-completions routing for `gpt-5.4*`) |
| AK pool | `configs/llm.toml` weighted pool (`gpt54_a:gpt54_b:gpt54_c = 5:1:5`) |
| Sandbox | `workspace_write` + `network_access=true` (tools shell out, write annotated PNGs; `keyframe_selector` needs the parsing LLM) |
| Skill | `.agents/skills/nr3d-codex-tools/SKILL.md` |
| Turn budget | none (`--turn-timeout 0`); the loop fixes below bound turns naturally |
| Reasoning effort | model default (low effort was tested and **hurt** spatial accuracy) |
| Concurrency | **40 workers**, `sample_retries = 2` |
| Judge | none — deterministic oriented 3D IoU vs GT 9-DOF box |
| Date | 2026-06-09 |

### CLI

```bash
PYTHONPATH=src python -m codex_agent.cli.run_nr3d \
  --sample-ids tmp/nr3d_case600/sample_ids.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
  --output-dir tmp/nr3d_tools_case600 \
  --pack-name pack_nr3d_v9_catalog_first \
  --workers 40 --sample-retries 2 --tools --turn-timeout 0
```

## What changed vs v1 (prompt-only)

The agent gets nine deterministic CLI tools plus the built-in `view_image`,
wired so the whole evidence loop runs **inside one `thread.run`** (one runtime
`execute`, prompt prefix stays cacheable):

| Group | Tool | Purpose |
|---|---|---|
| narrow (text) | `inspect_proposal` | full enrichment (color/desc/nearby/frames) for one id |
| narrow (text) | `list_scene_proposals` | filter the pool by category / BEV region |
| fetch (image) | `keyframe_selector` | language → ≤3 first-person frames (reuses `select_keyframes_v2`) |
| fetch (image) | `select_by_proposal` | frames showing given ids (esp. **co-visible** target+anchor) |
| read (image) | `mark_frame_with_bbox` | labeled boxes on a frame → PNG for `view_image` + 2D left→right |
| read (text) | `list_frame_proposals` | which ids are in a frame, ordered left→right |
| spatial (text) | `compare_proposals_spatial` | 3D metric ranking + co-viewed 2D left/right votes |
| spatial (text) | `compare_candidates_to_anchors` | multi-anchor consistency check |
| align (image) | `view_bev` | proposal-highlighted top-down map |

Supporting infra changes required to make the loop actually work:

1. **Adapter: forward structured output.** The ModelHub adapter
   (`build_chat_completions_body`) now maps the Responses `text.format`
   (json_schema) to a Chat `response_format`. Strict structured output makes the
   model emit a complete valid decision object as its final answer, eliminating
   the "terse final / finalization re-ask" overhead seen without it. The
   decision schema was made strict-compatible (`additionalProperties:false`, all
   fields required) since the upstream rejects non-strict schemas.
2. **Adapter: surface `view_image` images upstream.** (Already landed
   2026-06-09 — see `docs/codex_agent/view_image_adapter_fix_20260609.md`.)
3. **Disable repo `AGENTS.md` for the focused agent** (`project_doc_max_bytes=0`
   in the runtime's config overrides) and add hard anti-distraction rules to the
   prompt/skill. Without this the model inherited the repo's coding-agent rules
   and looped re-reading `SKILL.md`/`AGENTS.md` (~250 tool calls, 41 min on one
   sample). After the fix the same hard case solves in ~100 s with ~7 focused
   tool calls.

## Run health (observed, halted at 169/600)

- **0 pipeline errors / 169.** No `status=failed`, no `selected=-1` (absent)
  among completed — strict structured output + the `AGENTS.md`/anti-distraction
  loop fixes held: every completed case produced a valid decision object.
- Adapter healthy throughout: `HTTP 200` on health probe at halt time, `429 = 0`
  (never rate-limited at 40 workers); transient 5xx recovered by retries.
- **Prompt caching confirmed live:** 62.9% of input tokens served from cache
  (1.87 M / 2.97 M) — the ModelHub `session_id` prefix cache is working.
- **But throughput collapsed** (see next section). Per-case duration is
  bimodal: median **157 s** (healthy), but a heavy tail — p90 **607 s**,
  p95 703 s, **max 5190 s (86 min)**; 18 turns > 600 s, 6 > 1200 s, **4 > 3000 s**.

## Why the run was halted (throughput collapse)

The run was launched with **no turn budget** (`--turn-timeout 0`) on the bet
that the loop fixes alone would bound turns. They bound the *median* (157 s) but
**not the tail**: a small fraction of cases (4 over 50 min, up to 86 min) ran
away. With no hard cap, each runaway turn pins one of the 40 workers for over an
hour. Because fast cases recycle their worker quickly while slow ones do not,
the worker pool is progressively occupied by long in-flight turns
(head-of-line blocking), and aggregate throughput decays:

| Window | completed | rate |
|---|---:|---:|
| First ~36 min | ~130 | **~3.6 /min** |
| Last 60 min before halt | 21 | **~0.35 /min** |
| (stalled 25 min at done=160 with 0 completions) | | |

Projected ETA at the halt-time rate was **~20–24 h** for the remaining 431
cases — not the ~1.5–2 h the median implies. Root cause is structural (no turn
cap), **not** the adapter (HTTP 200, 0×429) and **not** a pipeline bug (0
errors). This is exactly the risk flagged in the v1→v2 caveat ("a reliable
hard-cap … is noted as follow-up if any case is observed to stall"); it
materialized at full-fold scale.

**Required follow-up before the next full run:** a turn hard-cap that actually
stops a busy turn. The SDK's `turn_interrupt` only sets a flag and does not
free the worker, so the fix is to **kill the per-turn Codex app-server
subprocess** on budget exceed (e.g. cap = p95 ≈ 700 s, leaving 95% of cases
untouched) and record a best-effort / declined decision for the capped case
rather than letting it run unbounded. With the tail capped, the median-implied
~1.5–2 h ETA is recoverable at almost no accuracy cost (the runaway turns'
accuracy was already mixed: IoU 1.0 / 0 / 0).

## Validated smoke evidence (pre-launch)

- **End-to-end loop works.** On the hard view-dependent sample
  `scene0221_00::47` ("the square table close to the door"), the agent runs
  `compare_proposals_spatial` → `select_by_proposal` → `mark_frame_with_bbox` →
  `view_image`, and correctly selects proposal **47** (IoU ≈ 1.0). `view_image`
  receives the annotated pixels (adapter image-injection fix confirmed).
- **Tools verified individually** against real scenes (annotated frame + BEV
  highlight render correctly; spatial ranking, co-visible frame selection,
  enrichment lookup all produce the expected JSON).
- **10-case canonical subset** (prompt-only baseline 7/10): the tool pipeline
  reproduces correct picks on the solvable cases (lamp/pillow/desk/table all
  IoU 1.0 when not artificially time-capped).

## Baseline for comparison (v1 prompt-only, same fold)

| Slice | n | Acc (= Acc@0.25 = Acc@0.50) |
|---|---:|---:|
| Overall | 600 | 63.83% |
| Easy | 290 | 73.10% |
| Hard | 310 | 55.16% |
| View-Dep | 211 | 51.66% |
| View-Indep | 389 | 70.44% |

The tools specifically target the **View-Dep < View-Indep** gap (co-visible
frames + 2D left/right voting + first-person `view_image`), so that slice is the
primary thing to watch when v2 numbers land.

## Headline (strat600 PARTIAL, n=169 completed)

Acc = Acc@0.25 = Acc@0.50 (selection task, `source=gt` pool). v1 column is the
**full-600** prompt-only baseline; v2 column is the **169 completed** here.

| Slice | v2 n | v2 Acc | mean IoU | v1 (full-600) | Δ vs v1 |
|---|---:|---:|---:|---:|---:|
| **Overall** | 169 | **63.31%** | 0.636 | 63.83% | **−0.5** |
| Easy | 85 | 70.59% | 0.709 | 73.10% | −2.5 |
| Hard | 84 | 55.95% | 0.561 | 55.16% | +0.8 |
| **View-Dep** | 62 | **50.00%** | 0.503 | 51.66% | **−1.7** |
| View-Indep | 107 | 71.03% | 0.712 | 70.44% | +0.6 |

**Survivorship-bias check (why the partial is trustworthy):** the 169 completed
samples track the fold's tier mix closely — Easy share 50.3% (fold 48.3%),
View-Dep share 36.7% (fold 35.2%). The skew is ~1–2 pp, far inside the noise
band, so the partial is **not** a cherry-picked easy subset. (If anything
View-Dep is slightly *over*-represented, which would bias the headline *down*,
not up.)

## Partial conclusion

**The in-turn evidence tools did not improve grounding accuracy on this fold.**
Every slice lands within the strat600 90% variance band of the v1 prompt-only
floor (band ≈ ±2.3 pp Overall, ±4.5 pp View-Dep; even wider at n=169), so all
Δ here are statistically indistinguishable from zero. Most importantly, the
**View-Dep slice — the explicit target of co-visible frames + 2D left/right
voting + first-person `view_image` — is flat (50.0% vs 51.66%)**, i.e. the
tools that were designed to close the View-Dep gap did not close it.

This holds *despite* the tools costing materially more per case (median 157 s vs
v1's ~65 s per-turn, plus the multi-call tool loop) and the image-injection /
structured-output infra all working as designed. The bottleneck is therefore
**not** plumbing — it is that the agent, given the tools, is not converting the
extra evidence into better picks on the hard/view-dependent cases.

Open questions for the next iteration (before re-running the full fold):

1. **Is the agent actually using the spatial/co-visible tools on View-Dep cases,
   or defaulting to the catalog?** Needs per-case tool-trace analysis (ingest to
   SQLite, slice tool-call mix by tier) — this partial run's traces are in
   `tmp/nr3d_tools_case600/` if `CODEX_AGENT_KEEP_RUN_HOME` was set (it was not,
   so only the per-sample summaries survive; future runs should keep traces).
2. **Does `view_image` evidence change the decision, or does the model commit
   from text first?** A/B: tools-text-only vs tools+`view_image`.
3. **Land the turn hard-cap** (above) so a full 600 actually finishes, then
   re-measure on the full fold before drawing a firm conclusion — n=169 is
   suggestive, not decisive.

## Trace analysis: the SKILL.md re-read loop (answers open question #1)

**Source.** 57 Codex session rollouts (`.codex-home/runs/*/sessions/.../
rollout-*.jsonl`) survived the kill — these are the run homes that were *in
flight* when the run was stopped (completed turns delete their run home), so
this set is the **slow tail**, not a random sample. The completed-169's traces
are gone (`CODEX_AGENT_KEEP_RUN_HOME` was off). Each NR3D tool runs via Codex
`exec_command` (`python -m codex_agent.nr3d.tools <tool> …`); `view_image` is its
own function call. Analyzer (durable):
`assets/analyze_nr3d_traces_20260609.py` (maps each rollout to a fold tier by
matching the prompt's `query:` line to the prepared sample artifacts).

**Finding 1 — the agent *does* use the spatial / co-visible tools.** The
hypothesis "the agent ignores the spatial tools and defaults to the catalog" is
**false**. Among the 22 View-Dep rollouts: **19/22 called a spatial tool**
(`compare_proposals_spatial` / `compare_candidates_to_anchors`), **16/22 fetched
co-visible frames** (`select_by_proposal require_all=true`), 17/22 used
`view_image`; median **6** real NR3D tool calls. View-Indep is similar. So the
tools are wired correctly and the agent reaches for them.

**Finding 2 — but the turn is drowned by a re-read loop.** Across the 57
rollouts, **4462 `exec_command` calls — only 357 (8%) are NR3D tools; 4073 (91%)
are `sed`/`cat` re-reads of Markdown skill files.** Breakdown of the .md reads:

| Re-read target | reads | share |
|---|---:|---:|
| **PROJECT `nr3d-codex-tools/SKILL.md`** | 3553 | **87.2%** |
| ambient `.system/*` (skill-creator, etc.) | 208 | 5.1% |
| ambient `using-superpowers/SKILL.md` | 200 | 4.9% |
| other `SKILL.md` variants | 112 | 2.7% |

Median **58** skill re-reads per case (max **194**). The failure mode is sharp:

| Outcome | n | median .md reads | median NR3D calls |
|---|---:|---:|---:|
| **killed mid-turn** (in loop) | 49 | **74** (max 194) | 6 |
| empty/just-started stub | 8 | 0 | 0 |

i.e. **every genuinely-running slow case did ~6 real tool calls and then spiraled
into re-reading its own skill file dozens of times until killed.** This is the
mechanism behind the throughput collapse, and it almost certainly degrades the
*decision* too: the model's context gets flooded with the same ~220-line skill
dozens of times (median 58×, up to 194×), which is exactly the kind of context
bloat that hurts reasoning even on the cases that do eventually finalize.

**Root cause.** `project_doc_max_bytes=0` stopped the `AGENTS.md` injection but
**not** the model's compulsion to re-read the *skill* it was given — it just
switched from reading `AGENTS.md` to `sed`-ing `nr3d-codex-tools/SKILL.md`. The
isolated `CODEX_HOME` also still carries ambient global skills
(`using-superpowers`, `.system/skill-creator`) that leak in (~13% of reads). The
earlier "fixed in ~100 s / ~7 tool calls" smoke evidence was a single fast case
that never entered the loop — it did not generalize to the tail.

**Fix targets for v3 (in priority order).**

1. **Stop the skill being a re-readable file.** Inline the playbook into the
   turn's developer/system prompt so it is always in context and there is nothing
   to `sed`; drop it as a discoverable on-disk skill (or strip the skill file
   from the sandbox after injection). This removes the 87% term directly.
2. **Strip ambient global skills** from the per-run `CODEX_HOME`
   (`using-superpowers`, `.system/*`) so they cannot leak (~13%).
3. **Hard rule**: "the playbook is already in your context; never run
   `sed`/`cat`/`grep` on any `*.md`." Cheap backstop for 1–2.
4. **Turn hard-cap** (kill the per-turn app-server) as the final safety net so a
   single looping case can never again pin a worker for 80 min.

Only after the loop is removed is the v2-vs-v1 accuracy question answerable: with
the turn no longer 91%-wasted, we can see whether the (already-invoked) spatial
tools actually change the pick on View-Dep cases, or whether the model invokes
them but ignores their verdict.

## Caveats

- `Acc@0.25 == Acc@0.50` for a `source=gt` pool: a correct proposal pick scores
  IoU ≈ 1.0, a wrong/declined pick ≈ 0, so IoU collapses to NR3D **selection
  accuracy**.
- **This run halted at 169/600** — partial, not full. n=169 per-tier counts are
  small (View-Dep n=62), so treat all Δ vs v1 as within-noise until a full fold
  lands. The completed set is composition-representative (see survivorship check)
  but excludes the slowest cases (which were the runaway-turn tail).
- The turn budget caveat **materialized**: `--turn-timeout 0` + unreliable
  `turn_interrupt` → runaway turns → throughput collapse → manual halt. See
  [Why the run was halted](#why-the-run-was-halted-throughput-collapse). The
  reliable hard-cap (kill the per-turn app-server) is now a **blocking
  follow-up**, not an optional one.
- `keyframe_selector` is a fallback (it reaches the parsing LLM over the
  network); the catalog-first tools carry the main load.
