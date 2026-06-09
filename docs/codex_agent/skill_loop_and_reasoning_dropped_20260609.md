# Codex agent infinite SKILL.md loop + dropped reasoning — root-cause log (2026-06-09)

Investigation of why the NR3D tool-using Codex agent (`CodexAgentRuntime`) falls
into an unbounded loop re-reading `SKILL.md`, which caused the v2 `case600`
throughput collapse (see
[`docs/benchmark/nr3d/v2_codex_tools_grounding_strat600_20260609.md`](../benchmark/nr3d/v2_codex_tools_grounding_strat600_20260609.md)).

This is a process record. It documents three linked findings, the fixes landed,
and — importantly — a hypothesis the experiment **disproved**. The ModelHub
adapter lives in a **separate, non-git** directory
(`/Users/bytedance/aispace/codex_modelhub_adapter`), so its patches are
reproduced verbatim here so they survive.

## TL;DR

1. **Skill-catalog flood.** The agent's skill catalog was ~50 skills (user/global
   `~/.agents/skills` + `~/.codex/superpowers` + bundled `.system`), almost all
   ambient. Restricting to project-only skills (HOME isolation + disabling the
   bundled `.system` skills) cut it to the 2 project NR3D skills and fixed
   **8/10** pilot cases — but 2/10 hard cases still looped.
2. **Reasoning was silently OFF.** `reasoning_output_tokens = 0` on every call.
   Cause: codex sends `reasoning:{effort:"medium"}` but the adapter's
   Responses→ChatCompletions conversion **dropped the field**. Fixed in the
   adapter (forward `reasoning_effort` + surface `reasoning_tokens`). Verified the
   upstream honors it (medium→83, high→94 reasoning tokens) and accuracy rose
   62%→75% on the pilot.
3. **Reasoning was NOT the loop's cause (hypothesis disproved).** With reasoning
   ON the loop got *worse* (SKILL.md re-reads 20%→66% of shell calls; per-case
   wall time 40–97 s → 750–800 s). Reason: the Chat Completions path must strip
   `encrypted_content`, so per-step reasoning has **no continuity** across tool
   calls — the model still "forgets" mid-turn and degenerates after the
   `view_image` step.

## Method

All evidence came from the surviving Codex session rollouts
(`.codex-home/runs/*/sessions/.../rollout-*.jsonl`, kept via
`CODEX_AGENT_KEEP_RUN_HOME=1`), a `skills/list` RPC probe, a captured
codex→adapter request, and a direct upstream probe. Analyzers:
`docs/benchmark/nr3d/assets/analyze_nr3d_traces_20260609.py` and the throwaway
`tmp/probe_skills.py` / `tmp/probe_reasoning.py`.

## Finding 1 — skill-catalog flood

`skills/list` against a fresh run home returned **~50 enabled skills**:

| Scope | Source | Count |
|---|---|---:|
| USER | `~/.agents/skills/` (lark-*, bytedance-*, cmux, …) | ~30 |
| GLOBAL | `~/.codex/superpowers/skills/` (superpowers:*, incl. `using-superpowers`) | ~15 |
| SYSTEM | `<run_home>/skills/.system/` (skill-creator, plugin-creator, imagegen, openai-docs, skill-installer) | 5 |
| REPO | `.agents/skills/` (nr3d-codex-sdk, nr3d-codex-tools) | 2 |

The `using-superpowers` meta-skill instructs "invoke skills before acting",
priming compulsive skill reading.

**Fix (this repo, `CodexAgentRuntime`, gated by `restrict_skills_to_project`,
default `True`):**

- **HOME isolation** — run the app-server with `HOME=<run_home>/.home` so
  `~/.agents/skills` and `~/.codex/superpowers` resolve to an empty dir → removes
  the ~45 user/global skills.
- **Disable bundled `.system` skills** via the official `skills/config/write`
  RPC (`enabled=false`) per turn.

Probe after the fix: catalog → **only the 2 project NR3D skills enabled**.

Pilot effect (10-case, restrict ON, reasoning still OFF): **8/10 clean** (2–4
tool calls, 0 skill re-reads, 40–97 s); **2/10 still looped** on the *project*
`SKILL.md` (15–22 reads). So ambient removal was necessary but not sufficient.

## Finding 2 — reasoning silently dropped by the adapter

Every rollout token-count event showed `reasoning_output_tokens = 0` across all
cases. gpt-5.4 is a reasoning model, so this was wrong.

**Captured codex→adapter request (`/v1/responses`) vs the adapter's upstream
body:**

| | reasoning field |
|---|---|
| incoming (codex→adapter) | `reasoning: {"effort": "medium"}` ✅ |
| outgoing (adapter→upstream) | dropped — only `model, messages, max_tokens, tools, response_format, …` ❌ |

`build_chat_completions_body` never mapped the Responses `reasoning` block to a
Chat `reasoning_effort`. `_usage()` also never surfaced upstream
`completion_tokens_details.reasoning_tokens`.

**Direct upstream probe** (`tmp/probe_reasoning.py`, a 9-balls puzzle): the
ModelHub gpt-5.4 Chat Completions endpoint **does** honor `reasoning_effort` —
`medium`→`reasoning_tokens=83`, `high`→`94`. So the fix is to forward it.

### Adapter patches (reproduce — adapter dir is NOT git-tracked)

`adapter/mapping.py`, in `build_chat_completions_body`, before the final return:

```python
reasoning = responses_body.get("reasoning")
if isinstance(reasoning, dict):
    effort = reasoning.get("effort")
    if isinstance(effort, str) and effort:
        chat_body["reasoning_effort"] = effort
```

`adapter/mapping.py`, in `_usage`, after the cached-tokens block:

```python
out_details = usage.get("completion_tokens_details") or usage.get(
    "output_tokens_details"
)
if isinstance(out_details, dict) and out_details.get("reasoning_tokens") is not None:
    result["output_tokens_details"] = {
        "reasoning_tokens": out_details.get("reasoning_tokens")
    }
```

After the patches, a real run showed `reasoning_output_tokens` 0 → **45,610** and
pilot accuracy **62%→75%**; one prior looper ("closet doors") now solves cleanly.

## Finding 3 — reasoning is NOT the loop cause (DISPROVED)

The hypothesis "zero reasoning → no planning → loop" was tested by re-running the
same 10-case pilot with reasoning ON. Result: **the loop persisted and got
worse.**

| metric | reasoning OFF | reasoning ON |
|---|---|---|
| reasoning_output_tokens | 0 | 45,610 |
| accuracy (completed) | 62% (5/8) | **75% (6/8)** |
| SKILL.md reads (share of shell calls) | 20% | **66%** (338 exec / 224 md) |
| per-case wall time | 40–97 s | 71–165 s; loopers 750–800 s |
| looping cases | 2/10 | ~4/10 (different cases) |

Per-case trace of a reasoning-ON looper ("middle window", 104 exec / 88 md):
productive phase (`compare → view_bev → view_image → inspect → select →
mark_frame → view_image`) then a pure `sed SKILL.md ×88` tail — the **same
degenerate pattern, triggered right after the `view_image` step**, now with
reasoning on.

**Why reasoning didn't help the loop:** the Chat Completions path forces the
adapter to strip `encrypted_content` from reasoning items (`mapping.py` ~L1199),
the carrier of cross-turn reasoning. So the model reasons *within* a step but the
reasoning is discarded for the next step — **no continuity**. Per-step thinking
improves single decisions (accuracy ↑) but cannot sustain a multi-step plan to
convergence, so after the heavy-image step the model loses the thread and falls
back to its default "lost" action: re-reading `SKILL.md`.

### Corrected root-cause model of the loop

The loop triggers **after the `view_image` step** and is driven by the
combination of:

1. **No reasoning continuity** across tool calls (encrypted_content stripped —
   inherent to the Chat Completions adapter path; not fixable in the adapter).
2. **Large images** (~17 K tokens each) destabilising/bloating context (the turn
   also blew past codex's reported 258 K window — itself a misconfig for a model
   whose real window is ~1 M; codex falls back to a conservative default for the
   unrecognised custom model name).
3. **A re-readable on-disk skill file** as the default action when "lost".
4. **No no-progress guard / turn cap** to break a repeated-identical-call rut.

Note: codex's reported `model_context_window=258400` is a conservative default
for the unknown custom model `gpt-5.4-2026-03-05`; the real window is ~1 M. This
is a separate config hygiene issue but **not** the loop trigger (the loop ignites
at ~83 K tokens, far below any limit).

## Recommended fixes (LANDED + verified — see Resolution below)

1. **Inline the skill/playbook into the turn prompt and drop the on-disk skill
   file** — removes the specific degenerate action (no file to re-read). Highest
   leverage for the loop.
2. **No-progress / duplicate-tool-call guard + reliable turn hard-cap** (kill the
   per-turn app-server) — backstop any rut; the SDK `turn_interrupt` is
   unreliable.
3. **Downscale annotated frames / BEV before `view_image`** (e.g. ≤768 px) and
   cap images per turn — reduce the post-image destabilisation.
4. **Set `model_context_window` to the real value** in the codex config — hygiene.
5. **Keep reasoning ON** — it is a real accuracy win and a real bug fix, even
   though it is not the loop cure and roughly doubles latency.

All five landed; see **Resolution** for the code, the knobs, and the 10-case
verification that the loop is gone (0 real `SKILL.md` reads, bounded tool calls,
accuracy ↑ to 80 %).

## What landed in this repo

- `src/codex_agent/config.py` — `restrict_skills_to_project: bool = True` (+ env
  `CODEX_AGENT_RESTRICT_SKILLS`).
- `src/codex_agent/runtime.py` — HOME isolation (`_ISOLATED_HOME_DIRNAME`),
  `_disable_ambient_system_skills` via `skills/config/write`,
  `_BUNDLED_SYSTEM_SKILLS`.
- `src/codex_agent/tests/test_runtime.py` — regression tests for both.

Adapter patches (Finding 2) are **not** in this repo — reproduce from the blocks
above against `/Users/bytedance/aispace/codex_modelhub_adapter/adapter/mapping.py`.

## Reproduce

```bash
# skill catalog probe (which skills the agent sees)
PYTHONPATH=src python tmp/probe_skills.py base   # ~50 skills
PYTHONPATH=src python tmp/probe_skills.py rpc    # restricted -> 2 project skills

# upstream reasoning probe (does gpt-5.4 honor reasoning_effort?)
cd /Users/bytedance/aispace/codex_modelhub_adapter && uv run python tmp_probe_reasoning.py

# 10-case A/B (keep run homes, then analyze md-read share)
CODEX_AGENT_KEEP_RUN_HOME=1 PYTHONPATH=src python -m codex_agent.cli.run_nr3d \
  --sample-ids tmp/nr3d_case10/sample_ids.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
  --output-dir tmp/nr3d_skill10 --limit 10 --workers 10 --tools --turn-timeout 0
```

## Resolution (landed + verified — 2026-06-09)

All five recommended fixes landed in this repo. The loop is a *defence in
depth* problem: the inline playbook removes the specific bait, the guard caps any
residual rut, and the smaller images + bigger window reduce the trigger.

### What changed

| # | Fix | Where |
|---|---|---|
| 1 | **Inline playbook, no on-disk skill in tool mode.** The whole tool catalog + loop + budget guidance is a prompt string; tool mode attaches **no** `SkillInput`, so there is no advertised `SKILL.md` path to re-read. | `src/codex_agent/nr3d/playbook.py` (new), `nr3d/grounding.py` (`build_turn_request` skips the skill in tool mode; `_tools_section` inlines `NR3D_TOOLS_PLAYBOOK`), `cli/run_nr3d.py` (`_resolve_skill` → `None` in tool mode unless `--skill-path`). |
| 2 | **In-turn loop guard (reliable hard-cap).** The runtime *streams* the turn (`thread.turn(...).stream()`), counts tool actions, and `interrupt()`s when a total or repeated-action cap trips. If the server ignores the interrupt and keeps emitting tools, a post-interrupt grace abandons the turn → the `with Codex(...)` exit tears the app-server down. A finalization re-ask then collects an answer from the evidence already gathered. | `runtime.py` (`_run_turn_bounded`, `_consume_guarded_turn`, `_ToolCallLoopGuard`, `_tool_action_signature`, `_final_response_from_items`), `config.py` (`max_tool_calls`, `max_repeated_tool_calls`). Defaults when `--tools`: **30 / 4**. |
| 3 | **Downscale viewed images to ≤768 px.** `mark_frame_with_bbox` / `view_bev` shrink the PNG before writing (boxes are reported in the written image's coords). | `src/codex_agent/nr3d/tools/image_io.py` (new, `MAX_VIEW_IMAGE_DIM=768`), used by `frame_annotation.py` + `bev_tools.py`. |
| 4 | **`model_context_window` override.** Stops codex falling back to the conservative 258 400 default for the unknown custom model. | `config.py` (`model_context_window`, env `CODEX_AGENT_MODEL_CONTEXT_WINDOW`), `runtime.py` (`_build_config_overrides`). |
| 5 | **Reasoning stays ON** (model default effort) — unchanged from Finding 2. | adapter forward (Finding 2). |

### Anti-exploration prompt hardening

`_tools_section` now states the agent is *"NOT exploring or editing a codebase"*
and hard-forbids `read/cat/sed/head/grep/rg/open` of any `SKILL.md`, `AGENTS.md`,
`README`, docs, or source — directly countering the injected coding-agent
persona that biased the model toward file exploration.

### Verification (10-case pilot, `tmp/nr3d_case10`, tools mode, all fixes on)

Run: `--tools --workers 10`, `CODEX_AGENT_MODEL_CONTEXT_WINDOW=900000`,
`CODEX_AGENT_KEEP_RUN_HOME=1`; whole run ≈ 3.5 min (no single case stalled).

| metric | reasoning OFF (F1) | reasoning ON, pre-fix (F3) | **fixed (this)** |
|---|---|---|---|
| Acc@0.25 / Acc@0.50 | 0.62 / — | 0.75 / — | **0.80 / 0.80** |
| mean IoU | — | — | **0.80** |
| looping cases | 2/10 | ~4/10 (750–800 s each) | **0/10** |
| guard interrupts | n/a | n/a | **0** (no case reached the cap) |
| real `SKILL.md` / doc reads | 20 % of shell calls | 66 % (338 exec / 224 md) | **0** |
| tool calls / case | — | 80–224 (loopers) | **2–14** (median ~4) |
| `view_image` / case | — | — | **1–2** |
| reasoning_output_tokens | 0 | 45 610 | **>0** (e.g. 512/turn) |
| `model_context_window` | 258 400 | 258 400 | **~855 000** |
| cache hit rate | — | — | **0.90** |

The two "doc reads" my first grep flagged were **false positives** — the
anti-exploration rule literally contains the strings `SKILL.md` / `AGENTS.md` /
`README`, so rollout lines echoing the prompt matched. A structural walk of the
`commandExecution` items found **zero** doc-file reads. The loop is gone because
there is no longer a file to re-read, not merely because the guard cut it off
(the guard never fired).

### Knobs added (all default-safe)

- `--max-tool-calls` / `CODEX_AGENT_MAX_TOOL_CALLS` (tools default 30; 0 = off)
- `--max-repeated-tool-calls` / `CODEX_AGENT_MAX_REPEATED_TOOL_CALLS` (tools default 4)
- `CODEX_AGENT_MODEL_CONTEXT_WINDOW` (0 = leave codex default)
- `--skill-path` still forces the legacy on-disk tools skill for A/B.

### Gate + tests

`ruff`, `black --check`, `mypy` clean on `src/codex_agent/`; `pytest
src/codex_agent/tests` = **156 passed**. New tests: `test_runtime_guard.py`
(guard logic, stream consume/interrupt/abandon, signature drift vs the real
`openai_codex` models), plus inline-playbook (`test_grounding.py`), caps
(`test_run_nr3d_cli.py` / `test_config.py`), and downscale (`test_nr3d_tools.py`).

### Next step

Re-run the canonical **strat600** fold with this config and write a new
`docs/benchmark/nr3d/vN_…` version doc + leaderboard update; this 10-case pilot
is a bug-fix smoke check, not a decision-grade benchmark.
