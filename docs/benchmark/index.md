# Benchmark Archive

This chapter is the durable archive for benchmark runs. Each benchmark keeps a
timeline page plus dated version records. Raw summaries, predictions, traces,
and other assets stay beside the records that cite them.

## Benchmarks

- [NR3D Timeline](nr3d/README.md): visual grounding evaluations, stage-1
  keyframe coverage, tool-loop runs, and reasoning-path ablations.
- [OpenEQA Timeline](openeqa/README.md): embodied QA evaluations, no-frame
  redesign runs, judge notes, and failure trace analysis.
- [SceneFunc Molmo SAM3D Timeline](scenefunc_molmo_sam3d/README.md):
  SceneFuncVal-CG small-object Molmo/SAM/3D-lifting feasibility checks.
- [Adapter Protocol Diagnostics](adapter/README.md): direct HTTP tool/history
  and opaque-state checks, archived separately from task accuracy leaderboards.

## Process Rules

When adding a new evaluation record:

1. Add a dated version file under the relevant benchmark directory.
2. Update that benchmark's `README.md` timeline.
3. Keep raw artifacts under the benchmark's `assets/` directory when they must
   be preserved with the result.
4. Update `mkdocs.yml` so the new record appears in the web navigation.

Per-run data is ingested into the benchmark's local SQLite database:
`openeqa/runs.sqlite` via `scripts/ingest_openeqa_run.py`, and `nr3d/runs.sqlite`
via `scripts/ingest_nr3d_run.py`. Keep durable summaries/traces alongside the
version records so the ignored databases can be rebuilt.
Adapter protocol probes use `adapter/runs.sqlite`, rebuilt from their durable
assets with the version's `ingest.py`.

Latest SDK/protocol diagnostic:
[v3 Codex–Gemini state/cache](adapter/v3_codex_gemini_cache_20261103.md)
— 65 upstream requests, six dynamic tools, 34 completed SDK turns. The isolated
SDK bridge preserved native signed state and observed cache hits; an exact
accounting control supports a 3,315-token historical reasoning/state cache lower
bound. Failed and zero-cache runs remain archived. Production adapter source is
unchanged; GPT remains native Responses-only.

Latest canonical pilot evaluations (2026-11-04):
- [NR3D Stage-1 Keyframe Coverage v2 (`baseline` vs `current` on `strat600`)](nr3d/v2_keyframe_grid_upgrade_coverage_strat600_20261104.md)
  — `hit@3` improves from **79.17% (475/600)** to **89.50% (537/600) (+10.33 pp)**,
  `hit@1` from **65.67% to 77.00% (+11.33 pp)**, **Hard `hit@3`** from **74.84% to 89.03% (+14.19 pp)**,
  **View-Dep `hit@3`** from **81.99% to 91.47% (+9.48 pp)**, and empty `no_evidence` returns drop from **40 to 0**.
- [NR3D v14 `strat600` (`gpt-5.4` vs `gemini-3.8-flash`)](nr3d/v14_strat600_gpt54_vs_gemini38flash_20261104.md)
  — `gemini-3.8-flash` reaches **91.50% Acc@0.25 (549/600)** and **89.57% View-Dep**;
  `gpt-5.4-2026-03-05` reaches **84.83% (509/600)** and **80.09% View-Dep**.
- [OpenEQA v6 `pilot300` Map-First & Tool Suite Overhaul (`gemini-3.8-flash`)](openeqa/v6_mapfirst_tools_overhaul_pilot300_20261104.md)
  — `gemini-3.8-flash` reaches **MNAS = 83.08 (300/300)** (**+4.16** over v5
  `gemini-3.8-flash`, **+13.83** over v5 `gpt-5.4`, **+18.16** over v4 on the
  matched 300-question fold), while cutting median turn latency **4.0× (`165.1 s → 41.0 s`)**.
- [OpenEQA v6 `pilot300` Map-First & Tool Suite Overhaul (`gemini-3.8-flash`)](openeqa/v6_mapfirst_tools_overhaul_pilot300_20261104.md)
  — `gemini-3.8-flash` reaches **MNAS = 83.08 (300/300)** (**+4.16** over v5
  `gemini-3.8-flash`, **+13.83** over v5 `gpt-5.4`, **+18.16** over v4 on the
  matched 300-question fold), while cutting median turn latency **4.0× (`165.1 s → 41.0 s`)**.
- [OpenEQA v5 `pilot300` (`gpt-5.4` vs `gemini-3.8-flash`)](openeqa/v5_pilot300_gpt54_vs_gemini38flash_20261104.md)
  — `gemini-3.8-flash` reaches **MNAS = 78.92 (300/300)**; `gpt-5.4-2026-03-05`
  reaches **MNAS = 69.25 (300/300)** (+4.33 over v4 on the matched 300-question
  fold).

Latest local API validation:
[v13 office Responses NR3D smoke10](nr3d/v13_office_responses_smoke10_20261103.md)
— 10/10 completed, 8/10 correct, catalog + BEV without extra NR3D tools;
a connectivity diagnostic, not a strat600 leaderboard result.
