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

Latest local API validation:
[v13 office Responses NR3D smoke10](nr3d/v13_office_responses_smoke10_20261103.md)
— 10/10 completed, 8/10 correct, catalog + BEV without extra NR3D tools;
a connectivity diagnostic, not a strat600 leaderboard result.
