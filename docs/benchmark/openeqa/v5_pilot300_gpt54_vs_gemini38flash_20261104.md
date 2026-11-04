# OpenEQA v5 — canonical pilot300 (`gpt-5.4` vs `gemini-3.8-flash`), 2026-11-04

First evaluation on the **canonical 300-question OpenEQA pilot fold**
(`assets/v5_pilot_fold_300_20260612.json`, stratified across all 7 categories over
85 ScanNet clips), comparing **`gpt-5.4-2026-03-05`** (native `/v1/responses`) and
**`gemini-3.8-flash`** (stateful `gemini_chat` codec) side by side through the
upgraded `codex_adapter`.

This run also validates the post-v4 evidence-loop and adapter improvements:

1. **Spatially-diverse keyframe retrieval + contact sheets + `view_crop` zoom**
   (`max_tool_calls=32`, `max_repeated_tool_calls=4`).
2. **Unique contact-sheet and crop filenames + atomic image writes** (`cbc1445`,
   `0635b65`), eliminating same-turn path collisions (`<clip_id>_kf_sheet_<ids>.jpg`
   and `<clip_id>_crop_<frame>_<box>.jpg`) that previously blocked `view_image` on
   second retrieval batches.
3. **Multimodal tool-output splitting in `gemini_replay.py`** (`f680dec`), moving
   `view_image` data-URL images out of `role="tool"` into an immediate `role="user"`
   part so ModelHub does not tokenize Base64 strings as text.

---

## Headline

Both models completed **300 / 300 questions (100% completion, 0 unresolved
failures)** on the canonical `pilot300` fold:

- **`gemini-3.8-flash`:** **MNAS = 78.92** (+14.00 pp over the v4 baseline on the
  same 300 questions).
- **`gpt-5.4-2026-03-05`:** **MNAS = 69.25** (+4.33 pp over the v4 baseline on the
  same 300 questions).

| Metric | v4 baseline (same 300 IDs, `gpt-5.4`) | **v5 `gpt-5.4-2026-03-05`** | **v5 `gemini-3.8-flash`** | Δ (`gemini` vs `gpt-5.4`) |
|---|---:|---:|---:|---:|
| **Overall MNAS** | 64.92 | **69.25** | **78.92** | **+9.67** |
| Questions (`n_judged / n`) | 300 / 300 | 300 / 300 | 300 / 300 | — |
| Completion rate | 100.0% | 100.0% | 100.0% | 0.0% |
| Prompt cache hit rate | 97.7% | 97.0% | 85.0% | −12.0% |
| Mean cache ratio | 46.6% | **84.3%** | **59.3%** | −25.0% |
| Median turn latency (s) | ~114 s | 189.3 s | **165.0 s** | −24.3 s |
| Judge score = 5 (perfect) | 135 (45.0%) | 149 (49.7%) | **180 (60.0%)** | +31 (+10.3 pp) |
| Judge score = 1 (miss) | 66 (22.0%) | 51 (17.0%) | **29 (9.7%)** | −22 (−7.3 pp) |

Judge-score histograms (1–5):

- **`gpt-5.4-2026-03-05`:** `1: 51, 2: 13, 3: 39, 4: 48, 5: 149`
- **`gemini-3.8-flash`:** `1: 29, 2: 3, 3: 40, 4: 48, 5: 180`

---

## Per-category MNAS (`pilot300`)

| Category | n | v4 baseline (same 300 IDs) | **v5 `gpt-5.4`** | Δ (`v5 gpt` − `v4`) | **v5 `gemini-3.8-flash`** | Δ (`gemini` − `v5 gpt`) |
|---|---:|---:|---:|---:|---:|---:|
| object state recognition | 45 | 77.78 | 87.22 | +9.44 | **90.56** | +3.34 |
| attribute recognition | 45 | 72.78 | 79.44 | +6.66 | **85.56** | +6.12 |
| object localization | 45 | 67.22 | 62.78 | −4.44 | **82.22** | +19.44 |
| world knowledge | 39 | 65.38 | 66.67 | +1.29 | **80.13** | +13.46 |
| object recognition | 43 | 52.33 | 59.88 | +7.55 | **75.58** | +15.70 |
| spatial understanding | 42 | 51.79 | 60.71 | +8.92 | **74.40** | +13.69 |
| functional reasoning | 41 | 65.85 | **66.46** | +0.61 | 62.20 | −4.26 |
| **Overall** | **300** | **64.92** | **69.25** | **+4.33** | **78.92** | **+9.67** |

### Key findings

1. **Post-v4 tool improvements lift `gpt-5.4` by +4.33 MNAS on the matched 300-question
   subset (`64.92 → 69.25`)**, with the largest gains in **object state recognition
   (+9.44)**, **spatial understanding (+8.92)**, **object recognition (+7.55)**, and
   **attribute recognition (+6.66)** — the exact fine-grained categories targeted by
   `view_crop` and diverse retrieval.
2. **`gemini-3.8-flash` reaches 78.92 MNAS (+9.67 over `gpt-5.4`)** and exceeds even
   the historical v3 8-seed-frame baseline (`73.96`), driven by large gains on visual
   grounding and perception categories: **object localization (82.22 vs 62.78)**,
   **object recognition (75.58 vs 59.88)**, **spatial understanding (74.40 vs 60.71)**,
   and **world knowledge (80.13 vs 66.67)**.

---

## Run metadata

- **Branch:** `feat/upgrade_codex_adapter`
- **Head commit at launch:** `0635b65` (`fix(tools): make NR3D and OpenEQA BEV/frame
  image writes atomic`). Working tree clean (`head == executed code`).
- **Fold:** `docs/benchmark/openeqa/assets/v5_pilot_fold_300_20260612.json` (300
  questions across 85 ScanNet clips).
- **Data root / questions:**
  - `--questions /Users/bytedance/project/3DVLMReasoning/data/OpenEQA/open-eqa-v0.json`
  - `--data-root /Users/bytedance/project/3DVLMReasoning/data/OpenEQA/scannet`
- **Adapter:** `codex_adapter` (`http://127.0.0.1:8787`), plaintext JSON SQLite WAL
  state store with concurrent reader pool and in-memory turn cache.
- **Judge:** `gemini-2.5-pro` via `configs/llm.toml` (dead `SpXKxu...` key disabled
  prior to run; 5 active keys with 2,500 QPM total).
- **Sandbox & loop caps:** `workspace_write` + network; `max_tool_calls=32`,
  `max_repeated_tool_calls=4`, `turn_timeout=900`.
- **Concurrency & rate-limit handling:** launched concurrently in `tmux` session
  `bench_20261104` at `--workers 10`.
  - `gemini-3.8-flash` ran at `workers=10` end-to-end (~2.0 h), followed by a
    low-concurrency recovery pass (`workers=2`) for 4 transient 409/interrupted
    samples → **300/300 completed**.
  - `gpt-5.4-2026-03-05` stepped down from `workers=10` to `workers=6` via the
    adaptive supervisor when `keyframe` parser calls rotated keys under concurrent
    load, followed by a recovery pass (`workers=3`) for 6 transient upstream
    `invalid_encrypted_content` / 403 errors → **300/300 completed**.

---

## Exact commands

```bash
# GPT-5.4-2026-03-05
uv run --no-sync python -m codex_agent.cli.run_openeqa \
  --questions /Users/bytedance/project/3DVLMReasoning/data/OpenEQA/open-eqa-v0.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/OpenEQA/scannet \
  --question-ids docs/benchmark/openeqa/assets/v5_pilot_fold_300_20260612.json \
  --output-dir tmp/openeqa_pilot300_gpt54_20261104_0635b65 \
  --model gpt-5.4-2026-03-05 \
  --reasoning-summary auto \
  --turn-timeout 900 \
  --sample-retries 4 \
  --workers 10

# Gemini-3.8-Flash
uv run --no-sync python -m codex_agent.cli.run_openeqa \
  --questions /Users/bytedance/project/3DVLMReasoning/data/OpenEQA/open-eqa-v0.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/OpenEQA/scannet \
  --question-ids docs/benchmark/openeqa/assets/v5_pilot_fold_300_20260612.json \
  --output-dir tmp/openeqa_pilot300_gemini38flash_20261104_0635b65 \
  --model gemini-3.8-flash \
  --reasoning-effort provider-default \
  --turn-timeout 900 \
  --sample-retries 4 \
  --workers 10
```

---

## Durable artifacts & SQLite ingestion

- **Committed assets:**
  - `docs/benchmark/openeqa/assets/v5_pilot300_gpt54_20261104_summary.json`
  - `docs/benchmark/openeqa/assets/v5_pilot300_gpt54_20261104_predictions.json`
  - `docs/benchmark/openeqa/assets/v5_pilot300_gemini38flash_20261104_summary.json`
  - `docs/benchmark/openeqa/assets/v5_pilot300_gemini38flash_20261104_predictions.json`
- **SQLite (`docs/benchmark/openeqa/runs.sqlite`):**
  - Run IDs: `v5_pilot300_gpt54_20261104` and `v5_pilot300_gemini38flash_20261104`.

---

## Caveats

- **Judge sensitivity:** Both runs are scored by `gemini-2.5-pro` (`configs/llm.toml`).
  Cross-model comparisons should note potential same-family judge affinity, though
  `gemini-3.8-flash`'s advantage is also consistent on deterministic 3D IoU in NR3D
  `strat600` (`91.50%` vs `84.83%`, where no LLM judge is involved).
- **Pose non-finiteness:** 13 ScanNet clips in OpenEQA contain non-finite poses in
  `traj.txt`; the tool layer's diversity/fallback guards handle missing pose features
  gracefully without crashing.
