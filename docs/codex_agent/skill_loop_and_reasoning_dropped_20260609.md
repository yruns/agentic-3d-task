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

## Recommended fixes (not yet landed)

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
