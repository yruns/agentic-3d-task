# SCOPE v1 — public example episode, 2026-10-05

The first SCOPE navigation episode completed end to end on Linux with the
user-authorized `gemini-3.8-flash` replacement. The launcher exited **0**;
all **5/5** frozen subtasks have matching distance-based success and SPL
records. **SR is 80.00% and SPL is 68.83971360148118%.**

This is an engineering check on the official public HM3D example scene. It
does not measure the paper's 36-scene / 278-subtask protocol or full validation
set, and it does not reproduce the paper's GPT-4o model configuration. Those
larger evaluations remain incomplete.

## Provenance and selection

```text
Run ID:                 REPRO_20261005_SCOPE_Gemini38Flash_public_example_ep0
Branch:                 feat/realtime-model
Head commit at launch:  d5c24a5e676d8729fd1f6d0f890d5bf5760b5827
Adapter code commit:    d5c24a5e676d8729fd1f6d0f890d5bf5760b5827
Run-time code commit:   46f1e1f78077219c30e35440540e4ef22995194c
Run-time source:        /home/ysh/codecase/SCOPE-repro/SCOPE
Environment:            /home/ysh/miniconda3/envs/scope-repro
GPU:                    CUDA_VISIBLE_DEVICES=0
Launcher:               scripts/scope/run_public_episode.sh
Start UTC:              2026-10-04T18:49:32.721419+00:00
Harvest date:           2026-10-05, Asia/Shanghai
Terminal exit status:   0
```

The runtime SHA belongs to the pinned official SCOPE repository; the project
SHA identifies the bridge and bootstrap. The original SCOPE checkout was
clean at launch. This is a separate upstream runtime, not a project worktree
running at an older project revision.

The selection contains scene `00861-GLAQ4DNUx5U`, original episode ID `0`,
selected with `--split 1` from the unchanged official task JSON. Its five
original subtasks were frozen in
[expected_subtask_ids.json](assets/public_episode_20261005/expected_subtask_ids.json).
The official public example supplies this scene's geometry, navmesh, semantic
mesh and semantic annotations. There are no substitute tasks or labels.

This was the first infrastructure check while the other scenes' semantics
were being prepared. It is not the canonical 278-ID selection. All 36 scenes
are now ready using existing geometry and the verified public semantic mirror;
the broader evaluation is described in the
[setup record](scope_reproduction_setup_20261005.md).

## Model, observations and execution

The original paper model is `gpt-4o-2024-11-20`; this run uses
`gemini-3.8-flash`, requesting `include_thoughts=true` and
`budget_tokens=2000`, through the Mac bridge and SSH reverse tunnel. No LLM
judge is used. The bridge rejects failed or empty API responses, and the
bootstrap terminates on infrastructure API errors before upstream API-failure
defaults can be accepted. Nonempty semantic-format fallbacks remain part of
the pinned implementation; their scope is described in the
[effective protocol record](scope_reproduction_setup_20261005.md#effective-protocol-and-fidelity-boundaries).

The original SCOPE prompts and navigation algorithm are retained, including
potential estimation and self-refinement. Memory persists across subtasks
within an episode; scene memory, TSDF and potential graph are initialized
again for each new episode. Requested sampling settings are temperature 0.7,
`top_p=0.95`, maximum output tokens 4,096 and frequency/presence penalties 0.
The shared fields follow [paper Appendix A, Table 3](https://arxiv.org/html/2511.08935v1);
all settings are explicit in the
[pinned API request](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/src/eval_utils_gpt_goatbench.py#L49-L57).
RGB-D observations are 1,280 × 1,280 with HFOV 120 degrees. The
[per-subtask observation loop](https://github.com/mrwangyou/SCOPE/blob/46f1e1f78077219c30e35440540e4ef22995194c/run_goatbench_evaluation.py#L207-L255)
captures seven views at each subtask's first step and three at subsequent
steps. Execution uses TSDF waypoints and the Habitat pathfinder. Its macro-step
budget follows
the original scene-size setting; it is not the original GOAT-Bench budget of
500 discrete movement actions.

The primary metrics are upstream `success_by_distance` and
`spl_by_distance`: success uses geodesic distance to annotated goal viewpoints
with a 1 m threshold. The separate snapshot metric in the upstream log is
40.00% SR / 40.00% SPL and is not substituted for the distance metric here.
The original run log reached `All scenes finish` after 38 min 25 s and recorded
162 API requests, all with HTTP 200.

## Results

SR and SPL below are percentages. Coverage is computed against the frozen
IDs, separately from navigation success.

| Modality | Recorded / expected | Successful | SR | SPL |
| --- | ---: | ---: | ---: | ---: |
| Description | 1 / 1 | 1 | 100.00 | 100.00 |
| Image | 2 / 2 | 2 | 100.00 | 72.09928400370295 |
| Object | 2 / 2 | 1 | 50.00 | 50.00 |
| All | 5 / 5 | 4 | 80.00 | 68.83971360148118 |

| Frozen subtask ID | Modality | Goal | Success | SPL |
| --- | --- | --- | ---: | ---: |
| `00861-GLAQ4DNUx5U_0_0` | Image | Dresser | 1 | 49.759576599460803 |
| `00861-GLAQ4DNUx5U_0_1` | Description | Mirror | 1 | 100.00 |
| `00861-GLAQ4DNUx5U_0_2` | Object | Mirror | 1 | 100.00 |
| `00861-GLAQ4DNUx5U_0_3` | Image | Picture | 1 | 94.43899140794508 |
| `00861-GLAQ4DNUx5U_0_4` | Object | Book | 0 | 0.00 |

The book task finished with a logged final viewpoint distance of
6.323014259338379 m and failed the 1 m success threshold. It is an observed
navigation failure within a completed episode. No missing subtask was replaced
with a fabricated metric.

The ingested summary verifies complete coverage, exact expected-ID membership
and complete modality metadata. SQLite contains five `completed` sample rows;
this status means the subtask produced a terminal result, including failures.
LLM calls are available only as a run-level bridge ledger. The ingester leaves
per-subtask token counts and the `llm_calls` table empty because the upstream
artifacts do not identify which call belongs to which subtask.

The [transport summary](assets/public_episode_20261005/transport_summary.json)
and [sanitized ledger](assets/public_episode_20261005/api_ledger.jsonl) contain
exactly this episode's 162 calls, excluding the two earlier preflight calls:

| API usage, run level | Reported tokens |
| --- | ---: |
| Prompt | 1,875,211 |
| Completion | 7,207 |
| Reasoning | 51,804 |
| Total | 1,934,222 |
| Cached prompt | 0 |

The image-rich navigation observations produced substantial prompt usage even
in one episode: about 97% of reported total tokens are prompt tokens. This
matters when planning the larger evaluation. The ledger does not separate
image tokens from text tokens, and this record makes no price assumption.
The figures are run-level usage, not a per-subtask cost breakdown.

## Reproduction and durable evidence

Remote output directory:

```text
/home/ysh/Datasets/SCOPE/outputs/REPRO_20261005_SCOPE_Gemini38Flash_public_example_ep0
```

Remote launch command, inside the original Linux `tmux` session
`scope-public-ep0-20261005`:

```bash
bash /home/ysh/Datasets/SCOPE/scripts/run_public_episode.sh \
  > /home/ysh/Datasets/SCOPE/logs/public_episode.log 2>&1
```

The launcher activates `scope-repro`, validates bridge health and the clean
upstream checkout, then invokes `run_scope.py --split 1` with the frozen
public-example config. A repeat needs a new output/run ID and separate exit
record to preserve this run's artifacts.

Tracked artifacts are in
[assets/public_episode_20261005](assets/public_episode_20261005/ingested_summary.json):

- [Terminal exit](assets/public_episode_20261005/public_episode.exit).
- [Run manifest](assets/public_episode_20261005/run_manifest.json), including
  runner and bridge SHA256 values.
- [Exact config](assets/public_episode_20261005/eval_public_example.yaml).
- [Task metadata](assets/public_episode_20261005/subtask_metadata.json) and
  [frozen IDs](assets/public_episode_20261005/expected_subtask_ids.json).
- [Success shard](assets/public_episode_20261005/success_by_distance_0.0_1.0_1.pkl)
  and [SPL shard](assets/public_episode_20261005/spl_by_distance_0.0_1.0_1.pkl).
- [Validated summary](assets/public_episode_20261005/ingested_summary.json).
- [Transport summary](assets/public_episode_20261005/transport_summary.json)
  and [sanitized run-level API ledger](assets/public_episode_20261005/api_ledger.jsonl).

The canonical database is `docs/benchmark/goat_bench/runs.sqlite`. The tracked
raw artifacts can reconstruct its run row without trusting a printed summary:

```bash
source .venv/bin/activate
python scripts/ingest_goat_bench_run.py \
  --output-dir docs/benchmark/goat_bench/assets/public_episode_20261005 \
  --run-id REPRO_20261005_SCOPE_Gemini38Flash_public_example_ep0 \
  --branch feat/realtime-model \
  --commit d5c24a5e676d8729fd1f6d0f890d5bf5760b5827 \
  --runtime-commit 46f1e1f78077219c30e35440540e4ef22995194c \
  --model gemini-3.8-flash --expected-subtasks 5 \
  --db docs/benchmark/goat_bench/runs.sqlite
```

SQL recomputation from sample rows:

```sql
SELECT COUNT(*) AS n, SUM(success) AS successes,
       100.0 * AVG(success) AS sr, 100.0 * AVG(spl) AS spl
FROM samples
WHERE run_id = 'REPRO_20261005_SCOPE_Gemini38Flash_public_example_ep0';
-- 5 | 4 | 80.0 | 68.83971360148118

SELECT task_type, COUNT(*) AS n,
       100.0 * AVG(success) AS sr, 100.0 * AVG(spl) AS spl
FROM samples
WHERE run_id = 'REPRO_20261005_SCOPE_Gemini38Flash_public_example_ep0'
GROUP BY task_type ORDER BY task_type;
-- description | 1 | 100.0 | 100.0
-- image       | 2 | 100.0 | 72.09928400370295
-- object      | 2 |  50.0 | 50.0

SELECT n, expected_subtasks, completion_rate, coverage_complete,
       expected_ids_verified, modality_complete
FROM runs
WHERE run_id = 'REPRO_20261005_SCOPE_Gemini38Flash_public_example_ep0';
-- 5 | 5 | 1.0 | 1 | 1 | 1
```

These queries were executed against the harvested local database and matched
the validated artifact summary. Larger-fold results require their own terminal
proof, exact frozen-ID coverage and version record.
