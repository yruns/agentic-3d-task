# ALFRED v6 — upstream outage after 41 scored tasks, 2026-11-05

The four direct-route pilot runs produced **41/50 official terminal scores**
before three runs stopped on native Responses HTTP 502 errors. One shard
completed; three episodes have explicit infrastructure errors and six were
never attempted. **This is a partial result, not a completed pilot50 result.**
The nine missing scores are not imputed as task failures.

The run-local loopback exclusions from [v5](v5_loopback_proxy_20261105.md) were
already applied. These failures occurred at `/v1/responses`, with the adapter
reporting `Unable to reach upstream; outcome may be indeterminate`. A separate
synthetic native Responses check also failed with HTTP 502. The evidence
establishes an unavailable model route at those times, but does not identify a
specific remote provider or VPN failure as the cause. Subsequent checks found
that the existing adapter loads `libaidp_proxy.so`, which redirects this host
through a local SOCKS forward at `127.0.0.1:10800`. Its SOCKS greeting stopped
responding; later checks received connection refusal. The adapter itself
continued to report `ready`. No shared adapter, proxy, or other benchmark
service was restarted. Restoring this existing forward, then verifying a real
native Responses turn, is the remaining external prerequisite.
[Read-only forwarding checks](assets/v6_upstream_20261105/network/forwarding_probe_01.jsonl)
and [diagnostic scope](assets/v6_upstream_20261105/network/diagnostic_context.json)
preserve the observed boundary without storing private configuration or keys.

## Fixed configuration and coverage

| Configuration | Value |
|---|---|
| Branch / clean HEAD | `feat/alfred-benchmark` / `fb408c34ab85284b32387002e92745b92983a173` |
| Executed-source SHA-256 | `24f41534d1c3929a2ef65eb0f6626dedebafe6131eb78f2342c933eb19ea7370` |
| Fold SHA-256 | `1d22afc51da128395a6d65a250d9dc603c8f50f0252942ca28b335568f646d4e` |
| Model / reasoning | `gpt-5.4-2026-03-05` / medium; native Responses throughout |
| Input / perception | H; FILM predicted depth/masks; action odometry and yaw/pitch proprioception |
| Action / failure / decision budgets | 1000 / 10 / 60 per episode |
| Worker / model-turn deadlines | 180 s / 300 s |
| Sharding | Original frozen order `[shard_index::4]`; no prefix limit |

The first frozen sample had already been used for the v2–v5 infrastructure
diagnostics. This is a development pilot, not an untouched independent test
set. Execution code, model, and budgets stayed fixed throughout these four
runs. The [end-of-run source audit](assets/v6_upstream_20261105/source_end_audit.json)
also confirms unchanged external sources and no modifications to existing
NR3D, OpenEQA, SceneFunc3D, keyframe, or adapter implementation files.

All four run IDs begin with
`REPRO_20261105_ALFRED_V3_20261105_pilot50_` and end in `sN_direct`.

| Shard | Selected | Scored | Error attempts without score | Unattempted | Run status |
|---|---:|---:|---:|---:|---|
| s0 | 13 | 8 | 1 | 4 | error |
| s1 | 13 | 11 | 1 | 1 | error |
| s2 | 12 | 12 | 0 | 0 | completed |
| s3 | 12 | 10 | 1 | 1 | error |
| Total | 50 | 41 | 3 | 6 | partial |

The complete run configurations, checkpoint/source hashes, exact argument
vectors, and summaries are preserved for
[s0](assets/v6_upstream_20261105/REPRO_20261105_ALFRED_V3_20261105_pilot50_s0_direct/run.json),
[s1](assets/v6_upstream_20261105/REPRO_20261105_ALFRED_V3_20261105_pilot50_s1_direct/run.json),
[s2](assets/v6_upstream_20261105/REPRO_20261105_ALFRED_V3_20261105_pilot50_s2_direct/run.json), and
[s3](assets/v6_upstream_20261105/REPRO_20261105_ALFRED_V3_20261105_pilot50_s3_direct/run.json).
Each directory also contains its unchanged `summary.json` and shell-quoted
`command.txt`. Full original artifacts remain in the corresponding
`outputs/alfred/<run-id>/` directory.

## Scores of the completed tasks only

The table below uses saved official terminal counts and
`codex_agent.alfred.metrics.aggregate_scores`. Each saved score was checked
against its terminal snapshot and frozen trajectory's expert length. GC uses
summed goal-condition counts; path-weighted metrics use expert-length weights.
No full-split or completed-prefix percentage was extrapolated to missing tasks.

| Subset | Scored / selected | Successes / SR | Conditions / GC | PLWSR | PLWGC |
|---|---:|---|---|---:|---:|
| Seen pilot | 25 / 25 | 3/25 = 12.00% | 9/63 = 14.29% | 7.78% | 12.16% |
| Unseen pilot, incomplete | 16 / 25 | 2/16 = 12.50% | 11/39 = 28.21% | 8.11% | 23.91% |
| Completed subset, descriptive | 41 / 50 | 5/41 = 12.20% | 20/102 = 19.61% | 7.89% | 16.08% |

All nine missing entries are unseen tasks. The combined 41-task subset therefore
has a different composition from the frozen 25-seen/25-unseen pilot. Its rates
must not be presented as pilot50 or full ALFRED validation accuracy. The 41
scored records contain 1,481 attempted actions and 1,813 expert actions; these
totals exclude the three unfinished error prefixes.

The [machine-readable partial metrics](assets/v6_upstream_20261105/partial_metrics.json)
retain all 41 identities/scores, the three error identities, and the six pending
identities. The [SQLite audit](assets/v6_upstream_20261105/sqlite_audit.json)
verified exact run metadata and all 44 attempted-episode records: 41 completed
scores and three errors with null scores. Pending entries are not fabricated
attempt records.

GC describes terminal condition satisfaction, not necessarily new progress
relative to initialization. For example, a SoapBar task received 1/3 despite
never completing a pickup. No initial-state score was supplied to the actor.

## Verified behavior, costs, and limits

The RGB → predicted depth/masks → incremental object/view/grid memory → action
→ new observation loop reached official scoring on all 41 completed episodes.
Five tasks succeeded: three seen and two unseen, including placement, cleaning
followed by placement, and holding an object under a light. The successful
unseen Cloth task used 29 attempted actions, 34 decisions, and eight action
failures before reaching 3/3 conditions. Its
[initial BEV](assets/v6_upstream_20261105/cases/clean_cloth_initial_bev.png),
[final BEV](assets/v6_upstream_20261105/cases/clean_cloth_final_bev.png), and
[official score](assets/v6_upstream_20261105/cases/clean_cloth_attempt.json)
are retained. These are estimated maps, not ground-truth geometry.

| Accounting for the 41 scored episodes | Value |
|---|---:|
| Policy decisions | 1,825 |
| Attempted simulator actions / failed actions | 1,481 / 267 |
| Observations, including each initial frame | 1,522 |
| Decision image attachments / historical attachments | 3,687 / 37 |
| Reported input / output tokens | 16,824,132 / 2,105,734 |
| Cached input tokens (subset of input) | 605,824 |
| Cumulative policy-call / perception-call wall time | 42,995.65 s / 719.79 s |
| Decision-budget / failure-budget / policy-Stop terminations | 17 / 13 / 11 |

Image counts are decision attachments, not an assertion about every underlying
wire retransmission. Times sum episode calls across concurrent shards; they
are not overall elapsed time or pure model-kernel latency. Output tokens
already include reasoning tokens. The three unscored prefixes add **at least
439,747 input and 75,784 output tokens**. Their unanswered final decisions have
unknown usage, so these are lower bounds, not complete outage costs.
The [accounting report](assets/v6_upstream_20261105/audits/pilot41_accounting.json)
includes per-episode, per-type, action, stop-reason, and SQLite checks.

An independent [official-goal recheck](assets/v6_upstream_20261105/audits/pilot41_terminal_rescore.json)
recomputed all **41/41** terminal scores from saved evaluator-only metadata and
cleaned/cooled/heated histories with hash-pinned official goal methods. It made
no model calls and did not replay the simulator. The
[actor-boundary audit](assets/v6_upstream_20261105/audits/pilot41_actor_boundary.json)
checked 1,825 requests/responses/catalogs/traces, 3,687 image references, and 314
interactions with zero recorded violations. Available checks on the three error
prefixes are separate: 59 requests, 56 responses, and three unanswered final
requests. Their coverage is explicitly incomplete. Scope limits include no
saved original mask-RPC arguments and no full on-wire tool capture for every
request; declared catalogs and preserved SDK traces showed no host-tool calls.
The [lossless full audit](assets/v6_upstream_20261105/audits/pilot41_actor_boundary.full.json.gz)
retains the 5,962-entry file manifest; the readable JSON above omits only that
large manifest and records the full report's original and compressed hashes.

The actor uses yaw/pitch proprioception, motion odometry, and native action
feedback. Simulator IDs are redacted from errors, but obstacle names/categories
can remain. This is not a strict RGB-and-language-only leaderboard setting.
Perception is ALFRED-trained FILM depth and Mask R-CNN, with 0.5 detection/mask
thresholds; association uses geometry and RGB histograms, not CLIP embeddings.
Generated local Python policies, a complete interaction-state verifier, and
controlled memory/retrieval ablations remain unimplemented.

| Confirmed limitation | Post-hoc evidence |
|---|---|
| Interaction readiness | Targets can be detected yet outside usable interaction range; a predicted Sink mask also selected a Bowl in five failed placements. [Audit](assets/v6_upstream_20261105/cases/interaction_readiness_audit.json) |
| Identity after manipulation | One CD was picked twice under memory IDs 11/21; a ToiletPaper instance likewise reappeared as IDs 1/44. Both two-object goals ended 1/2. [CD association](assets/v6_upstream_20261105/cases/placement_identity_audit.json), [ToiletPaper case](assets/v6_upstream_20261105/cases/interaction_readiness_audit.json) |
| Reobserving a remembered target | A Mug was detected early, but 45 of 50 actions were rotations; 18 approach-reached messages did not restore a usable current mask. [Audit](assets/v6_upstream_20261105/cases/exploration_failure_audit.json) |
| Unsupported completion claims | A Bowl/light task stopped without a successful pickup or toggle and scored 0/2. [Audit](assets/v6_upstream_20261105/cases/premature_stop_audit.json) |
| Native simulator exceptions | Eight ToggleObjectOn failures across three scored episodes contained Unity NullReferenceException errors; one of those episodes still ultimately succeeded. These remain failed actions, not unscored infrastructure errors. [Full cohort scan](assets/v6_upstream_20261105/cases/simulator_exception_cohort_scan.json), [two detailed cases](assets/v6_upstream_20261105/cases/simulator_action_exception_audit.json) |

These analyses used terminal artifacts after actor execution and supplied no
ground truth back to policies. They identify concrete follow-up work without
establishing module-level causal gains. The evaluated implementation passed
Ruff, Black, mypy, **1,284 source tests**, and **440 adapter tests with 91.34%
branch-inclusive coverage**. [Verification commands/results](assets/v6_upstream_20261105/verification/results.json)
and original check output are preserved.

## Error prefixes and independent connectivity check

| Original global index | Shard | Attempted decisions | Physical actions | Action failures | Terminal score |
|---|---|---:|---:|---:|---|
| 32 | s0 | 39 | 23 | 2 | None |
| 45 | s1 | 8 | 5 | 0 | None |
| 43 | s3 | 12 | 10 | 1 | None |

These counters include the decision attempt that encountered the upstream
error; they are not counts of valid model responses. Each error directory
preserves `attempt.json`, `partial_execution.json`, and `actions.jsonl`, and
partial token counters remain explicitly named lower bounds. No terminal
scorer result was manufactured after the failure.

The [synthetic probe 01 report](assets/v6_upstream_20261105/synthetic_probe_01/report.json)
records the same clean code and model, medium reasoning, a 90-second turn limit,
and zero finalization retries. It submitted a synthetic image and an exact Stop
JSON request without starting Unity or an embodied task. The invocation failed
after 10.45 seconds with HTTP 502 and no final response. Its
[source](assets/v6_upstream_20261105/synthetic_probe_01/upstream_recovery_probe.py),
[output](assets/v6_upstream_20261105/synthetic_probe_01/launcher.txt), synthetic
input, and reproduction command are archived. No provider configuration,
credentials, or SDK home contents are included. This probe contributes no
ALFRED task score.

## Prepared recovery, not executed

The [cohort manifest](assets/v6_upstream_20261105/unlaunched_recovery_plan/pilot50_cohort.json)
selects the original 41 completed results and plans recovery of exactly global
indices **32, 36, 40, 43, 44, 45, 47, 48, 49**. Indices 32/43/45 come from the
three error attempts; the other six were pending. Each planned invocation uses
the unchanged CLI with `--num-shards 50 --shard-index <original-index>`, selecting
one entry from the original fold. No new fold or resampling is introduced.

Recovery IDs follow
`REPRO_20261105_ALFRED_V3_20261105_pilot50_iNN_recovery1`. The three prepared
queues are `[32,43,47]`, `[36,44,48]`, and `[40,45,49]`. They retain the same
source, model, perception, language, budgets, and timeouts; each queue stops on
its first failed invocation and never automatically retries. Original scored
episodes will not be replaced, and old error evidence remains archived.

At the [saved plan check](assets/v6_upstream_20261105/unlaunched_recovery_plan/not_launched_snapshot.json),
none of the nine recovery output/status paths or three queue status files
existed. Recovery is awaiting restored upstream connectivity. This record does
not claim recovery has started or the 50-task cohort has completed.

The [current recovery manifest](assets/v6_upstream_20261105/recovery_tools/pilot50_cohort.json)
and [queue driver](assets/v6_upstream_20261105/recovery_tools/run_pilot50_recovery_queue.py)
also accept a later documentation-only HEAD while requiring the identical full
executed-source digest, branch, clean state, and actor configuration. Each run
retains its actual HEAD; the original run/summary hashes remain immutable.
Boundary regressions verify that source, branch, dirty state, model, input, or
budget changes are rejected. The earlier unlaunched-plan snapshot above remains
unchanged. Neither version has launched a recovery task.

Exact source snapshots for the completed audits are retained in
[audit_scripts](assets/v6_upstream_20261105/audit_scripts/executed_versions.json);
their hashes match the reports, including the original cohort manifest. The
separate [historical FILM reference](assets/v6_upstream_20261105/reference/film_matched_reference.json)
covers the full frozen50 under different complete-policy/perception settings;
its full-fold rates must not be directly compared with this partial41 subset.

All preserved files are covered by the
[SHA-256 manifest](assets/v6_upstream_20261105/sha256.json). This record adds new
evidence only; v1–v5 and the evaluated source remain unchanged.
