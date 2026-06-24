# OpenEQA v4 — failure trace analysis: "why so many tool calls, still no useful frame?"

Companion to [`v4_noframes_full1079_20260612.md`](v4_noframes_full1079_20260612.md).
v4 (no seed frames, MNAS 64.09) regressed −9.87 vs v3 and **8.3 % of questions
hit the `max_tool_calls=24` ceiling**. This doc digs into *why* — from real
Codex rollouts, not guesses.

## TL;DR

The premise "many tool calls and still can't find a useful frame" is true for
**only ~2 of the 7 worst cases**. The fuller picture, from traces:

1. **`keyframe_selector` almost always returns frames** — "can't find a frame"
   is rarely literal. The failures are about *which* frame and *what's legible in
   it*, not retrieval returning nothing.
2. **Retrieval-pool collapse** is the real "many calls" mechanism: re-phrasing
   the query returns the **same small overlapping set of frames**, so extra
   rounds add almost no new visual coverage (shampoo: 7 queries → 23 frames but
   only **13 distinct**; book: 5 queries → 19 → **16 distinct**; printer: 4 → 4 →
   **2 distinct**).
3. **The 24-action budget is split with `view_image`** (each returned frame costs
   one `view_image` action to actually see), so 2–3 unproductive retrieval
   batches exhaust it → forced finalize → low-confidence guess.
4. **Resolution ceiling**: frames are ≤768 px JPEG with no zoom/crop, so
   fine-grained reads (a shampoo *brand*, a *book color*, phone-vs-photo) are
   physically unresolvable. More rounds cannot fix this.
5. Most score-1s are actually **confident early commits to the wrong
   object/region** at modest tool counts (6–17 actions), *not* budget blowouts.

Net: raising `max_tool_calls` alone will **not** move the needle much — the
binding constraints are retrieval diversity, image resolution, and the agent's
over-confidence, not the call cap.

## Method

Traces are not kept for the full run (keeping every per-question Codex sandbox
for 1,079 questions is infeasible). So I **re-ran 7 worst-case questions one at a
time** with the per-turn `CODEX_HOME` preserved and the reasoning summary on:

```bash
CODEX_AGENT_KEEP_RUN_HOME=1 PYTHONPATH=src python -m codex_agent.cli.run_openeqa \
  --questions data/open-eqa-v0.json --data-root data/OpenEQA/scannet \
  --output-dir <out> --question-ids <single-qid fold> \
  --no-judge --workers 1 --sample-retries 0 --reasoning-summary detailed
```

This preserves `<run_home>/sessions/**/rollout-*.jsonl` — the full ordered
tool-call / image-view / output stream. Parser + aggregator + the 7 rendered
traces are durable under
[`assets/v4_failure_traces/`](assets/v4_failure_traces/). Driver:
`tmp/trace_v4_failures.sh` (ephemeral). The 7 qids are the longest-duration
score-1 cases (a proxy for heavy tool use), spread across categories. Re-runs are
non-deterministic, so individual answers differ from the full run, but the
**failure mechanisms reproduce**.

## The 7 traces

`cnt` = guard-counted actions (exec + view_image; the 24 cap); `kf` =
keyframe_selector calls; `uq` = distinct kf queries; `fA/fD` = frames returned
total / **distinct**; `poll` = `write_stdin` waits (NOT counted — see mechanics).

| qid | category | cnt | hit cap | kf | uq | fA | **fD** | poll | GT → prediction |
|---|---|--:|:--:|--:|--:|--:|--:|--:|---|
| 52ee7f3e | world knowledge | 25 | **YES** | 7 | 7 | 23 | **13** | 25 | dove → *Head&Shoulders/Herbal Essences* |
| 6ad7b496 | attribute recog | 25 | **YES** | 5 | 5 | 19 | **16** | 20 | red → white |
| 5fa829d4 | object localization | 17 | no | 3 | 3 | 12 | 9 | 9 | basket on cart → bin on cart |
| 0b172e4a | spatial | 11 | no | 3 | 3 | 5 | 5 | 1 | mobile phone → b/w photo |
| 199fc658 | spatial | 10 | no | 4 | 4 | 4 | **2** | 12 | recycling tray → a desk |
| 2f7ae8d8 | object recog | 10 | no | 3 | 3 | **0** | **0** | 12 | charger → hair straightener |
| 962b8661 | object recog | 6 | no | 1 | 1 | 2 | 2 | 4 | lamp → desk chair (conf 0.77) |

## Five failure modes (each grounded in a trace)

### A. Retrieval-pool collapse (the real "many rounds" case)
`52ee7f3e` (shampoo brand). The agent issues **7 different queries** —
`"shampoo bottle"`, `"… near mouthwash"`, `"dresser top with shampoo"`,
`"mouthwash bottle"`, `"chest of drawers"`, `"wall shelf"`, `"white shampoo
bottle on desk shelf"` — but they keep grounding to the **same ~13 frames**
(240, 225, 455, 155 recur). Re-phrasing does not widen coverage because the
underlying CLIP+geometry grounding maps near-synonyms to the same objects/views.
The agent burns 7 kf + 13 view_image + 5 other = 25 actions, **hits the cap**,
and is force-finalized into a guess. `list_objects` even self-reports *"Due to
the low resolution…"*.

### B. Grounding miss — retrieval returns nothing
`2f7ae8d8` (item next to the fan). `keyframe_selector` returns
**`"no frames grounded for this query; try view_frame or rephrase"`** for
`"box fan"`, `"box fan and wall heater"`, **and** `"wall heater"` — 3/3 empty.
The query terms don't match the scene's detected vocabulary, so geometry
grounding yields nothing. The agent falls back to `list_objects` + `view_bev`
and guesses "hair straightener" (GT charger). This is a **keyframe_selector
recall failure**, the closest to the literal "can't find a frame."

### C. Confident early commit (the majority of score-1s)
`962b8661` (left of the office table). **6 actions total**: list_objects →
view_bev → 1 keyframe_selector (`"office desk"` → frames 595, 560) → view 2
frames → done. Final answer "a desk chair" at **confidence 0.77** with a crisp
rationale ("BEV places #22 desk chair left of #20 desk"). GT is *lamp*. The agent
found *a* object to the left and committed — it never hit the cap and never
doubted itself. Under-search + over-confidence, not budget exhaustion.

### D. Budget exhaustion via view_image tax
`6ad7b496` (color of the thickest book). 5 queries → 16 distinct frames, **16
view_image actions** eat the budget; hits the cap; force-finalized to "white"
(GT red). Because *each returned frame needs its own `view_image` to be seen*, a
high-recall retrieval batch (k≤4) can cost 4–5 budget units, so even diverse
retrieval drains the 24 cap in ~5 batches.

### E. Fine-grained misread / resolution ceiling
`0b172e4a` (object between pencil and red notebook): correct desk, correct
region, but a **phone** read as a "black-and-white photo". `52ee7f3e`: brand text
unreadable at ≤768 px. These are pixel-limited; the agent's own claims say "too
blurry to read", "appears flatter, like a photo". No tool loop fixes a
resolution wall.

## Tool-budget mechanics (important + non-obvious)

- The loop guard (`runtime._ToolCallLoopGuard`) counts **CommandExecution** and
  **ImageView** thread items. It does **NOT** count `write_stdin`. Verified:
  shampoo reran to 47 SDK function-calls (11 exec + 13 view_image + 23
  write_stdin) and the guard only tripped at **exec+view_image = 25 > 24**.
- **`write_stdin` is the model polling a still-running PTY exec.**
  `keyframe_selector` is slow (~25 s: loads the point cloud + visibility index +
  an LLM query-parse per call), and Codex hands control back every ~1 s, so the
  model issues 2–5 empty `write_stdin` waits per kf call. These don't cost
  budget, **but** they cost wall-clock (turns run 250–500 s) and **flood the
  context** with poll round-trips. Combined with base64 images entering context,
  Codex windows the context (drops older images/outputs); the "amnesiac" model
  then re-issues a query it already tried — **feeding mode A**.
- So the 24 cap effectively allows only ~5–6 retrieval batches. For pixel-limited
  or pool-collapsed questions that's not enough signal, and the cap converts an
  unanswerable-from-evidence question into a forced guess.

## Population-level cross-check (all 249 score-1, from `supporting_claims`)

Per-question tool traces aren't stored for the full run, but the agents'
`supporting_claims` are. Keyword scan over the 249 score-1 samples:

- **27 %** explicitly cite a visual-evidence limitation (legibility / blur /
  not-visible / occlusion).
- Only **3–4 %** explicitly blame legibility/resolution; **14 %** say
  "not visible/not found", **11 %** "occluded".
- "Not visible" language appears in **14 % of score-5** answers too — so it is
  **not discriminative**; the agent says it whether it succeeds or fails.

Read with the traces: the agent is usually **not aware** it failed — most score-1s
are confidently-wrong (mode C/E), which is why explicit "I couldn't see it"
citations are a minority. This matters: the model won't self-rescue by trying
harder, because it doesn't think it's wrong.

## Recommendations (ranked by expected MNAS leverage)

1. **Diversify retrieval, not just cap size.** Make `keyframe_selector` return
   *spatially de-duplicated* frames (different camera positions / yaw), and have
   the tool reject/annotate near-duplicate views across successive calls. Pool
   collapse (mode A) is the #1 "wasted rounds" cause; a bigger cap without
   diversity just buys more duplicate looks.
2. **Add zoom/crop.** A `view_frame`/`view_crop` that returns a high-res crop
   around an object id or bbox would directly attack the resolution ceiling
   (modes A-tail, E) — the shampoo/book/phone cases are unanswerable at 768 px.
3. **Batch viewing to cut the view_image tax.** Let one action view all k frames
   a `keyframe_selector` call returned (a contact sheet / multi-image view), so a
   retrieval batch costs ~2 budget units instead of ~5. Frees the cap for more
   distinct retrievals.
4. **Calibrate / penalize over-confidence (mode C/E).** Require a second
   independent view before committing to fine-grained object identity, or have
   the finalizer down-weight high-confidence answers backed by a single frame.
5. **Fix grounding recall (mode B).** When `keyframe_selector` grounds nothing,
   auto-fall back to a category/BEV-guided `view_frame` sweep instead of leaving
   the agent to re-query empty.
6. **Speed up `keyframe_selector`** (cache the per-scene PCD/visibility index
   across calls in one turn) to cut the write_stdin polling and context bloat
   that feeds the re-query loop.
7. Only **then** consider raising `max_tool_calls` (e.g. 32) — and measure it
   against a canonical pilot fold, not the full set.

## Caveats

- 7 traces (longest score-1 cases) — a deliberately biased, small sample for
  *mechanism discovery*, not a rate estimate. Rates come from the 249-sample
  `supporting_claims` scan, which is coarse (keyword-based, agent self-report).
- Re-runs are non-deterministic; predictions differ from the full run, but the
  per-mode mechanics (pool collapse, grounding miss, early commit, budget tax)
  reproduce across the set.
- No code was changed for this analysis; it characterizes the **shipped `bfb00c2`
  pipeline**.
