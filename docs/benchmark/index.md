# Benchmark Archive

This chapter is the durable archive for benchmark runs. Each benchmark keeps a
timeline page plus dated version records. Raw summaries, predictions, traces,
and other assets stay beside the records that cite them.

## Benchmarks

- [NR3D Timeline](nr3d/README.md): visual grounding evaluations, stage-1
  keyframe coverage, tool-loop runs, and reasoning-path ablations.
- [OpenEQA Timeline](openeqa/README.md): embodied QA evaluations, no-frame
  redesign runs, judge notes, and failure trace analysis.

## Process Rules

When adding a new evaluation record:

1. Add a dated version file under the relevant benchmark directory.
2. Update that benchmark's `README.md` timeline.
3. Keep raw artifacts under the benchmark's `assets/` directory when they must
   be preserved with the result.
4. Update `mkdocs.yml` so the new record appears in the web navigation.

For OpenEQA, per-run data is also ingested into `openeqa/runs.sqlite` when the
run produces durable sample/tool-call records.
