# NR3D v14 — canonical `strat600` (`gpt-5.4` vs `gemini-3.8-flash`), 2026-11-04

Full **canonical `strat600` (600 samples across 119 ScanNet scenes)** evaluation
comparing **`gpt-5.4-2026-03-05`** (native `/v1/responses`) and
**`gemini-3.8-flash`** (stateful `gemini_chat` codec with multimodal tool-output
splitting) side by side on the upgraded `codex_adapter`.

Both runs use the self-contained inline tool playbook (`--tools`), `workspace_write`
sandbox + network access, and the `pack_nr3d_v9_catalog_first` proposal pool, scored
by deterministic oriented 3D IoU against the ground-truth 9-DOF box (`Acc@0.25 ==
Acc@0.50`).

---

## Headline

- **`gemini-3.8-flash`:** **Acc@0.25 = 91.50% (549 / 600)**, **View-Dep = 89.57%
  (189 / 211)**, **Hard = 89.68% (278 / 310)** — **600 / 600 completed (0 infra
  failures)**, setting a new high on `strat600` (**+5.50 pp Overall, +7.11 pp
  View-Dep** over the previous best `v7` at 86.00%).
- **`gpt-5.4-2026-03-05`:** **Acc@0.25 = 84.83% (509 / 600)**, **View-Dep = 80.09%
  (169 / 211)** — **599 / 600 completed (0 infra failures, 1 semantic target-absent
  refusal)**, matching `v8` (`84.50%`) and `v11` (`84.50%`) within +0.33 pp and
  confirming no regression from the `codex_adapter` SQLite WAL / plaintext state
  upgrade.

| Slice | n | v7 (`gpt-5.4`) | v8 (`gpt-5.4`) | v11 (`gpt-5.4` Skill) | **v14 `gpt-5.4`** | **v14 `gemini-3.8-flash`** | Δ (`gemini` − `v14 gpt`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 600 | 86.00% (516) | 84.50% (507) | 84.50% (507) | **84.83%** (509) | **91.50%** (549) | **+6.67 pp** (+40) |
| Easy | 290 | 90.34% (262) | 88.62% (257) | 89.31% (259) | **88.28%** (256) | **93.45%** (271) | **+5.17 pp** (+15) |
| Hard | 310 | 81.94% (254) | 80.65% (250) | 80.00% (248) | **81.61%** (253) | **89.68%** (278) | **+8.07 pp** (+25) |
| **View-Dep** | 211 | 82.46% (174) | 77.25% (163) | 79.15% (167) | **80.09%** (169) | **89.57%** (189) | **+9.48 pp** (+20) |
| View-Indep | 389 | 87.92% (342) | 88.43% (344) | 87.40% (340) | **87.40%** (340) | **92.54%** (360) | **+5.14 pp** (+20) |

---

## Run health & adapter telemetry

| Check | **v14 `gpt-5.4-2026-03-05`** | **v14 `gemini-3.8-flash`** |
|---|---:|---:|
| Sample records | 600 / 600 | 600 / 600 |
| Final status | 599 completed, 1 semantic target-absent | **600 completed, 0 failed** |
| Mean 3D IoU | 0.8503 | **0.9159** |
| Prompt cache hit rate | 0.9783 | 0.3133 |
| Mean cache ratio | 0.8794 | 0.2259 |
| Reasoning summaries captured | 538 / 600 (89.7%) | 0 / 600 (not supported on `gemini_chat`) |
| Turn latency (median / p90 / max) | 89.0 s / 172.0 s / 583.4 s | **79.0 s** / 175.6 s / 768.0 s |

### Key findings

1. **Decisive accuracy gain on `Hard` (+8.07 pp) and `View-Dep` (+9.48 pp) with
   `gemini-3.8-flash`:** Once the `gemini_replay.py` multimodal tool-output fix
   (`f680dec`) enabled `gemini-3.8-flash` to inspect `mark_frame_with_bbox` and
   `view_bev` images without token explosion, it achieved nearly 90% accuracy on both
   Hard and View-Dependent referring expressions.
2. **Adapter stability under 4-way concurrent load:** Running alongside the two
   `pilot300` OpenEQA jobs (40 total workers across 4 processes sharing one
   `codex_adapter` instance), SQLite WAL reader-pool separation and in-memory turn
   caching handled all multi-turn tool loops with zero database lock errors. The 3
   transient upstream `invalid_encrypted_content` errors on `gpt-5.4` and 2 transient
   `409` retries on `gemini-3.8-flash` were cleanly recovered in the low-concurrency
   tail pass (`workers=3..4`).

---

## Run metadata

- **Branch:** `feat/upgrade_codex_adapter`
- **Head commit at launch:** `0635b65` (`fix(tools): make NR3D and OpenEQA BEV/frame
  image writes atomic`). Working tree clean (`head == executed code`).
- **Fold:** `/Users/bytedance/project/3DVLMReasoning/tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json`
  (canonical `strat600`: 600 samples across 119 scenes; Easy 290 / Hard 310;
  View-Dep 211 / View-Indep 389).
- **Data root:** `/Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet`
- **Pack:** `pack_nr3d_v9_catalog_first`
- **Sandbox & loop caps:** `workspace_write` + network (`--tools`);
  `max_tool_calls=30`, `max_repeated_tool_calls=4`, `turn_timeout=900`.
- **Concurrency:** `--workers 10` in `tmux` session `bench_20261104` (both runs
  sustained `workers=10` for the entire main pass without stepping down), followed by
  a tail recovery pass at `workers=3..4`.

---

## Exact commands

```bash
# GPT-5.4-2026-03-05
CODEX_AGENT_MODEL_CONTEXT_WINDOW=900000 \
uv run --no-sync python -m codex_agent.cli.run_nr3d \
  --sample-ids /Users/bytedance/project/3DVLMReasoning/tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_strat600_gpt54_20261104_0635b65 \
  --model gpt-5.4-2026-03-05 \
  --tools \
  --reasoning-summary auto \
  --turn-timeout 900 \
  --sample-retries 4 \
  --workers 10

# Gemini-3.8-Flash (no CODEX_AGENT_MODEL_CONTEXT_WINDOW override, since gemini_chat does not support /compact)
uv run --no-sync python -m codex_agent.cli.run_nr3d \
  --sample-ids /Users/bytedance/project/3DVLMReasoning/tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_strat600_gemini38flash_20261104_0635b65 \
  --model gemini-3.8-flash \
  --reasoning-effort provider-default \
  --tools \
  --turn-timeout 900 \
  --sample-retries 4 \
  --workers 10
```

---

## Durable artifacts & SQLite ingestion

- **Committed leaderboard summaries:**
  - `docs/benchmark/nr3d/assets/v14_strat600_gpt54_20261104_leaderboard.json`
  - `docs/benchmark/nr3d/assets/v14_strat600_gemini38flash_20261104_leaderboard.json`
- **SQLite (`docs/benchmark/nr3d/runs.sqlite`):**
  - Run IDs: `v14_strat600_gpt54_20261104` and `v14_strat600_gemini38flash_20261104`.
