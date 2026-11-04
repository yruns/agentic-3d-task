# v13: office Responses smoke10, 2026-11-03

## Scope and result

**10/10 cases completed, 8/10 correct, mean oriented IoU 0.8000,
Acc@0.25 = Acc@0.50 = 80.00%.** No permanent API/sample failures.

This is an explicitly requested **connectivity/compatibility diagnostic**, using
the transferred frozen `strat10_smoke_sample_ids.json`. It is **catalog + attached
BEV, prompt-only** (`tools_enabled=false`), not the current tool-enabled agent
baseline, not a new random subset, and not a decision-grade strat600 comparison.
No claims of improvement over earlier versions are made.

```text
Branch:                 feat/nr3d-office-responses-smoke-20261103
Head commit at launch:  8ef0f7d1eca536857ac355a9b75490c091bd5461
Run-time code commit:   same (no worktree drift)
Run ID:                 v13_office_responses_smoke10_20261103
Model:                  gpt-5.4-2026-03-05
SDK / CLI runtime:      0.163.0 / 0.163.0
Reasoning:              medium; summary auto
Workers:                2
Sample retries:         0
Per-turn guard:         180 s; 12 tool actions; 3 identical actions
NR3D tools / Skill:     disabled / none
Judge model:            none (geometric 3D IoU)
```

Tracked working tree was clean at evaluation launch. Documentation and ingestion
code were added after completion and were not runtime changes to this run.

## Exact inputs and command

- Data: `/home/ysh/Datasets/nr3d/processed_v1/scannet/`.
- Pack: `pack_nr3d_v9_catalog_first`.
- Frozen IDs: `/home/ysh/Datasets/nr3d/processed_v1/folds/strat10_smoke_sample_ids.json`.
- Fold SHA256: `8525d9a82151638061a1589b3a29c97e2182e5474134a162dfcf229842f1b0bc`.
- Raw outputs: `tmp/nr3d_eval_v13_office_responses_smoke10_20261103/`.
- Raw launch/probe diagnostics: `tmp/nr3d_office_smoke_20261103/`.
- Durable results, probes, launch metadata, frozen IDs, and compressed original
  SDK rollouts: [assets/v13_office_responses_smoke10_20261103](assets/v13_office_responses_smoke10_20261103/summary.json).
- Launcher: [run_smoke10.sh](assets/v13_office_responses_smoke10_20261103/run_smoke10.sh).

Run in tmux from the project root, after activating `agentic3d`:

```bash
source scripts/datasets/processed_v1.sh
export CODEX_AGENT_KEEP_RUN_HOME=1
export CODEX_AGENT_PREFIX_CACHE_SESSION_ID=v13_office_responses_smoke10_20261103
python -m codex_agent.cli.run_nr3d \
  --sample-ids "$NR3D_DATA_ROOT/../folds/strat10_smoke_sample_ids.json" \
  --data-root "$NR3D_DATA_ROOT" --pack-name "$NR3D_PACK_NAME" \
  --output-dir tmp/nr3d_eval_v13_office_responses_smoke10_20261103 \
  --workers 2 --sample-retries 0 \
  --reasoning-effort medium --reasoning-summary auto \
  --turn-timeout 180 --max-tool-calls 12 --max-repeated-tool-calls 3
```

The runner reads those ten IDs directly; no `head -N`, fresh sampling, or implicit
full-fold limit was used. Future accuracy decisions should use canonical
strat600 and matched model/tool settings.

## Office endpoint and transparent networking

The SDK provider still points at `http://127.0.0.1:8787/v1`. The local adapter
passes Responses requests through to:

```text
https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/unified/v1/responses
```

The real AK is stored only in the ignored, mode-0600 adapter `.env`; no AK is
archived here. Relevant non-secret settings are:

```text
AIDP_CODEX_PROXY_UPSTREAM_ENV=office
AIDP_CODEX_PROXY_RESPONSES_PATH=/unified/v1/responses
AIDP_CODEX_PROXY_TIMEOUT_SECONDS=300
AIDP_CODEX_PROXY_MAX_429_RETRIES=1
```

Early connectivity probes failed because the long-running tmux server inherited
older networking variables and lacked the user's transparent `LD_PRELOAD`
forwarder. Merely propagating that library still left stale routing variables.
Starting the experiment adapter through **`zsh -ic`**, which loads the user's
existing `~/.zshrc`, made its exported environment match the current terminal.
No new HTTP proxy or `NO_PROXY` override was needed or retained.

- Direct curl in the corrected shell: **HTTP 200**, about **2.30 s**.
- Responses through the adapter: **HTTP 200**, about **1.99 s**.
- Actual SDK streaming, BEV image input, and structured grounding answers were
  exercised by the ten-case run, beyond these simple text connectivity probes.

This does **not** establish that a separately configured Codex CLI can directly
reach the office URL without the adapter. The SDK runtime, adapter, and curl are
distinct clients. The tested route remained SDK → local adapter → office API.

## Metrics and cases

| Metric | Value |
|---|---:|
| Cases | 10 |
| Completed / permanent failures | 10 / 0 |
| Correct at IoU ≥ 0.25 | 8 |
| Correct at IoU ≥ 0.50 | 8 |
| Mean IoU | 0.8000 |
| Run-summary cache-hit fraction | 0.60 |
| Run-summary mean cached/input ratio | 0.4571539051 |
| Actual model calls from rollout token events | 15 |
| Actual prompt / cached / output tokens | 296,022 / 140,160 / 8,705 |
| SDK-native `view_image` calls | 5 |

`tools_enabled=false` disables the benchmark's additional NR3D CLI tools, not
all SDK-native tools. Five cases reopened the already-attached BEV using
`view_image`; these yielded five additional model continuations. No first-person
frame retrieval, keyframe-parser LLM, or extra NR3D scene tool was exercised.

| Sample ID | Selected proposal | IoU |
|---|---:|---:|
| `scannet/scene0221_00::47::4274` | 47 | 1.0 |
| `scannet/scene0699_00::26::40486` | 26 | 1.0 |
| `scannet/scene0500_00::25::13471` | 26 | 0.0 |
| `scannet/scene0426_00::22::9868` | 22 | 1.0 |
| `scannet/scene0653_00::18::29686` | 18 | 1.0 |
| `scannet/scene0462_00::6::40854` | 6 | 1.0 |
| `scannet/scene0207_00::20::12189` | 20 | 1.0 |
| `scannet/scene0246_00::5::12312` | 5 | 1.0 |
| `scannet/scene0591_00::14::19171` | 16 | 0.0 |
| `scannet/scene0608_00::9::17679` | 9 | 1.0 |

The misses were a relative window position (“middle rectangular window ... left
of the chalkboard”) and a pillow appearance/location (“darker pillow ... against
the armrest”). Prompt-only evidence was not augmented with first-person tools;
this run cannot establish whether tool use would repair either error.

## Compact observation, separate from benchmark scoring

The user initially requested exploration, then asked not to pursue compact
further. These already-completed synthetic probes are retained for completeness;
no compact setting was changed for the NR3D run.

1. Direct `/unified/v1/responses/compact`: **HTTP 404**.
2. A real SDK thread with custom provider `modelhub_adapter` completed
   `thread.compact()` in about **8.09 s** and exposed a completed
   `contextCompaction` history item.
3. A subsequent turn correctly recalled the exact synthetic marker.

These results are consistent, not contradictory. Codex 0.163.0 classifies this
custom provider as lacking remote compaction and takes its designed **local
compaction** path: the client organizes an ordinary Responses request that asks
the remote model for a summary, then installs that summary into local history.
“Local” does not mean a model runs on this machine. It is not a fallback invented
by the adapter after a 404. Source references:

- `external/third_party/codex/codex-rs/model-provider/src/capabilities.rs`.
- `external/third_party/codex/codex-rs/core/src/tasks/compact.rs`.

The synthetic compact probe is not evidence that image-heavy NR3D automatic
compaction preserves every visual detail; no such benchmark was performed.

## Durable traces, SQLite, and caveats

Each of the ten original SDK rollouts is preserved compressed, mapped to the
sample by exact query text. `trace_index.json` records its path and uncompressed
SHA256. The trace input contains no target-ID ground truth; evaluation GT remains
in the checkpoint/sample records, not the model prompt.

This run has been ingested into `docs/benchmark/nr3d/runs.sqlite`: **10 samples,
15 per-call token rows, 5 SDK-native image-view tool rows**. The ingestion is
idempotent; reproduce it using:

```bash
python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v13_office_responses_smoke10_20261103 \
  --trace-index docs/benchmark/nr3d/assets/v13_office_responses_smoke10_20261103/trace_index.json \
  --run-id v13_office_responses_smoke10_20261103 \
  --branch feat/nr3d-office-responses-smoke-20261103 \
  --commit 8ef0f7d1eca536857ac355a9b75490c091bd5461 \
  --model gpt-5.4-2026-03-05 \
  --notes 'Office unified Responses; frozen smoke10; prompt-only catalog+BEV; SDK/CLI 0.163.0' \
  --db docs/benchmark/nr3d/runs.sqlite
```

Reproduce the headline from SQLite:

```sql
SELECT COUNT(*) AS n,
       AVG(iou) AS mean_iou,
       100.0 * AVG(iou >= 0.25) AS acc_025_pct,
       100.0 * AVG(iou >= 0.50) AS acc_050_pct
FROM samples
WHERE run_id = 'v13_office_responses_smoke10_20261103';
```

- The smoke10 is not directly comparable to strat600 or full 8,584/filtered
  7,805 results. It is deliberately not inserted into the strat600 accuracy table.
- Query-parser LLM configuration was not exercised: no extra NR3D tools were
  enabled, and no `configs/llm.toml` was needed. A tool-enabled validation is a
  separate experiment with additional prerequisites.
- SDK app-server rollouts contain per-call token events. Do not equate the
  checkpoint's last-request token counts with all calls across tool continuations
  or finalization re-asks; use the ingested trace rows for run-wide call totals.
  In this run, five native BEV image-view calls account for the extra five model
  continuations. Cache presence does not certify billing discounts or latency gains.
- Model credentials and private transport environment files remain ignored and
  are excluded from durable assets. No keys were committed or pasted into logs.
