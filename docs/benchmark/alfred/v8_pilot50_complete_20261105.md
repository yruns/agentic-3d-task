# ALFRED v8 — complete frozen pilot50, 2026-11-05

The incremental scene-memory agent completed official terminal evaluation on
**all 50 frozen pilot identities**, with **5/50 successes (10.00% SR)** and
**20/126 satisfied goal conditions (15.87% GC)**. Independent official-goal
recomputation matched **50/50** saved scores, and the final accounting and
actor-boundary audits passed. The 41 scores reported in
[v6](v6_upstream_outage_20261105.md) remain unchanged; nine previously unscored
identities supplied the remaining terminal results. All nine recovered tasks
produced valid task-failure scores without new infrastructure errors.
[Final audit results](assets/v8_recovery_complete_20261105/final_audit_results.json) and
[complete metrics/accounting](assets/v8_recovery_complete_20261105/audits/pilot50_analysis.json).

## Fixed setting and source identity

| Setting | Evaluated value |
|---|---|
| Fold | Deterministic stratified pilot: 25 `valid_seen` and 25 `valid_unseen`, seven task types |
| Language | H-only goal instruction; no step-by-step instructions |
| Model | `gpt-5.4-2026-03-05`, medium reasoning, native Responses throughout |
| Perception | `film_predicted_rgb_v1`: ALFRED-trained FILM depth and Mask R-CNN; detection/mask thresholds 0.5 |
| Pose | Successful-action odometry plus yaw/pitch proprioception; 0.25 m forward step |
| Camera | Height 1.576 m, vertical FOV 60°, 300×300 input |
| Per-episode action / failure / decision budgets | 1000 / 10 / 60 |
| Navigation chunk / retrieved-history-image budget | 12 actions / 3 images |
| Worker / model-turn timeout | 180 s / 300 s |
| Simulator | AI2-THOR 2.1.0, build `201909061227` |
| Branch | `feat/alfred-benchmark` |

The frozen [pilot file](folds/pilot50.json) has SHA-256
`1d22afc51da128395a6d65a250d9dc603c8f50f0252942ca28b335568f646d4e`.
The initial four direct runs used clean HEAD
`fb408c34ab85284b32387002e92745b92983a173`; all recovery runs used clean HEAD
`7010ba8f54088bdad50ad30ba8ac709765c0d1c6`. All 19 actual runs have identical
executed-source SHA-256:
`24f41534d1c3929a2ef65eb0f6626dedebafe6131eb78f2342c933eb19ea7370`.
Only documentation changed between these HEADs. Each run preserves its actual
HEAD rather than rewriting it to the baseline. Model, prompts, perception,
memory, navigation, budget, and scoring settings were held fixed.
[End-of-run source audit](assets/v8_recovery_complete_20261105/source_end_audit.json).

The recorded external sources are ALFRED commit
`f91f4c0c96c7a29f33d0557f86b0a21035379b3b` and FILM commit
`63abf9599bbcfae0b2ab91695da86fcca751f044`. FILM's existing local reproduction
changes were explicitly fingerprinted and stayed unchanged; this is not a
claim that the external FILM checkout was clean. Exact external source hashes,
checkpoint hashes, simulator configuration, and CLI arguments are preserved in
the [cohort record](assets/v8_recovery_complete_20261105/cohort.redacted.json) and each linked recovery run.
The [ALFRED implementation guide](../../codex_agent/alfred.md) describes the
incremental object/view/grid memory and subprocess interfaces.

## Official terminal results

| Subset | Scored | Successes / SR | Conditions / GC | PLWSR | PLWGC |
|---|---:|---|---|---:|---:|
| Full frozen pilot | 50 | 5/50 = 10.00% | 20/126 = 15.87% | 6.19% | 12.61% |
| Seen | 25 | 3/25 = 12.00% | 9/63 = 14.29% | 7.78% | 12.16% |
| Unseen | 25 | 2/25 = 8.00% | 11/63 = 17.46% | 4.45% | 13.10% |

| Task type | Scored | Successes / SR | Conditions / GC | PLWSR | PLWGC |
|---|---:|---|---|---:|---:|
| Look at object in light | 8 | 1/8 = 12.50% | 4/16 = 25.00% | 9.87% | 15.83% |
| Pick and place | 8 | 2/8 = 25.00% | 2/8 = 25.00% | 19.41% | 19.41% |
| Place with movable receptacle | 8 | 0/8 = 0.00% | 0/25 = 0.00% | 0.00% | 0.00% |
| Clean and place | 8 | 2/8 = 25.00% | 9/24 = 37.50% | 23.76% | 38.07% |
| Cool and place | 6 | 0/6 = 0.00% | 2/20 = 10.00% | 0.00% | 10.82% |
| Heat and place | 6 | 0/6 = 0.00% | 1/21 = 4.76% | 0.00% | 3.19% |
| Place two objects | 6 | 0/6 = 0.00% | 2/12 = 16.67% | 0.00% | 11.03% |

SR is the fraction of episodes satisfying the complete task goal. GC is the micro-average of satisfied
and total goal-condition counts. The path-weighted metrics use the official
per-episode factor `expert_actions / max(expert_actions, executed_actions)` and
then weight episodes by their expert lengths. The selected cohort has **2,311
expert actions and 1,955 attempted actions**; failed actions remain in attempted
path lengths. These are cohort aggregations, not averages of shard percentages.
There is no LLM judge. Terminal GC can include conditions already satisfied at
initialization; it does not establish net new progress during execution.
[Per-episode scores and denominators](assets/v8_recovery_complete_20261105/audits/pilot50_analysis.json).

The unchanged five successes cover placement, cleaning followed by placement,
and holding an object under a light. The 41-task descriptive rates in v6
(12.20% SR, 19.61% GC) are superseded for this pilot by the complete 50-task
rates above. The reduction reflects the addition of nine unseen failures;
none of the original successful or failed scored episodes was rerun.

A previously audited historical FILM H reference matched to these same 50
identities reports **11/50 successes (22.00%)**, **47/126 conditions (37.30%)**,
**8.31% PLWSR**, and **12.06% PLWGC**. Its trained policy, perception treatment,
navigation, and decision schedule differ from this agent. The reference is a
descriptive comparison: it is neither a controlled memory ablation nor an
experiment with matched inference budgets. FILM was not rerun in this version.
[Matched reference and provenance](assets/v6_upstream_20261105/reference/film_matched_reference.json).

## Recovery selection and all nine outcomes

The hash-pinned local selection manifest, preserved as a
[redacted archival copy](assets/v8_recovery_complete_20261105/cohort.redacted.json), binds one
terminal score to every frozen `(split, task, repeat_idx)` identity: 41 original
completed records and nine recovery records. The missing zero-based indices
were `32,36,40,43,44,45,47,48,49`, all unseen. Recovery selected each identity
using the same fold with `--num-shards 50 --shard-index INDEX`; this partitioning
change did not change actor settings. Failed unscored attempts restarted from
the original initial state. They were not continued physical trajectories.
Completed task failures were never eligible for retries.

All run IDs below have prefix
`REPRO_20261105_ALFRED_V3_20261105_pilot50_`. The links contain the exact recorded
argument vectors; neighboring `run.json`, `summary.json`, `attempt.json`, and
`job_status.json` preserve configuration and results. Every row has SR 0.

| Fold index | Task / object identifiers | Selected run suffix and exact command | Actions / failures | Decisions | GC count | Termination |
|---:|---|---|---:|---:|---:|---|
| 32 | Look at object in light: Box/FloorLamp | [i32_recovery3](assets/v8_recovery_complete_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i32_recovery3/command.txt) | 38 / 0 | 60 | 0/2 | decision_budget |
| 36 | Cool and place: LettuceSliced/GarbageCan | [i36_recovery3](assets/v8_recovery_complete_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i36_recovery3/command.txt) | 37 / 3 | 60 | 0/4 | decision_budget |
| 40 | Pick and place: SaltShaker/Drawer | [i40_recovery3](assets/v8_recovery_complete_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i40_recovery3/command.txt) | 39 / 9 | 60 | 0/1 | decision_budget |
| 43 | Cool and place: BreadSliced/Microwave | [i43_recovery1](assets/v8_recovery_complete_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i43_recovery1/command.txt) | 45 / 1 | 60 | 0/4 | decision_budget |
| 44 | Heat and place: AppleSliced/CounterTop | [i44_recovery1](assets/v8_recovery_complete_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i44_recovery1/command.txt) | 80 / 5 | 60 | 0/4 | decision_budget |
| 45 | Place two objects: PepperShaker/Drawer | [i45_recovery1](assets/v8_recovery_complete_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i45_recovery1/command.txt) | 47 / 10 | 44 | 0/2 | failure_budget |
| 47 | Pick and place: SaltShaker/Cabinet | [i47_recovery1](assets/v8_recovery_complete_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i47_recovery1/command.txt) | 70 / 2 | 60 | 0/1 | decision_budget |
| 48 | Place with movable receptacle: Pencil/Bowl/Shelf | [i48_recovery1](assets/v8_recovery_complete_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i48_recovery1/command.txt) | 76 / 4 | 60 | 0/3 | decision_budget |
| 49 | Clean and place: Pan/CounterTop | [i49_recovery1](assets/v8_recovery_complete_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i49_recovery1/command.txt) | 42 / 2 | 60 | 0/3 | decision_budget |

The successful third dispatch used queues `[43,47,32]`, `[44,48,36]`, and
`[45,49,40]`. Indices 32/36/40 use their third recovery run IDs; the other six
retain first-recovery IDs because their earlier queued jobs had never started.
Queue order is scheduling metadata, not a change to the frozen selection.
All nine selected jobs exited 0 and produced scores; eight reached the decision
budget and i45 reached the failure budget.
[Queue 0](assets/v8_recovery_complete_20261105/queues/q0.json),
[queue 1](assets/v8_recovery_complete_20261105/queues/q1.json),
[queue 2](assets/v8_recovery_complete_20261105/queues/q2.json).

The [archive manifest](assets/v8_recovery_complete_20261105/manifest.json)
records raw-source hashes and transformations; the
[asset checksums](assets/v8_recovery_complete_20261105/sha256.json) identify the
archived bytes. Redacted copies preserve selection and numeric evidence but
are not byte-identical replacements for the local originals.

## Cost and behavior of the 50 scored episodes

| Accounting measure | Selected 50 only |
|---|---:|
| Policy decisions / saved responses / observed SDK attempts | 2,349 / 2,349 / 2,349 |
| Attempted simulator actions / failed actions | 1,955 / 303 |
| Observations, including each episode's initial frame | 2,005 |
| Decision image attachments / historical-image attachments | 4,748 / 50 |
| Reported input tokens | 21,855,851 |
| Reported output tokens | 2,762,524 |
| Reported input + output tokens | 24,618,375 |
| Cached input tokens, included in input | 653,568 |
| Reasoning output tokens, included in output | 2,437,282 |
| Cumulative policy-call wall time | 60,805.61 s |
| Cumulative perception-call wall time | 767.13 s |
| Decision-budget / failure-budget / policy-Stop terminations | 25 / 14 / 11 |

These are sums of persisted SDK usage for all completed decisions; all 2,349
responses provide usage. Output already includes reasoning tokens, and cached
input is a subset of input. Image counts are decision attachments, not counts
of every possible wire transmission. Reported policy time includes request and
runtime waiting; summed call times across concurrent jobs are neither elapsed
runtime nor pure model inference latency. Diagnostic network/model probes are
outside these episode totals.
[Detailed usage, actions, and image accounting](assets/v8_recovery_complete_20261105/audits/pilot50_analysis.json).

For elapsed time, the original four-run stage spans **12,297.40 s**, failed
recovery1 **1,062.11 s**, failed recovery2 **1,162.66 s**, and the successful
nine-job dispatch **6,723.57 s**. Each span runs from its first job start to its
last job finish and includes waits inside that stage. Calendar elapsed from
the earliest original start (`2026-11-04 18:40:11 UTC`) to the final recovery
finish (`2026-11-05 05:14:28 UTC`) is **38,056.98 s**, including the gaps spent
waiting for upstream recovery. It should not be used as a throughput benchmark.
The final three audits completed at `2026-11-05 05:14:47 UTC`.

The 1,955 actions include 949 rotations and 344 object interactions. Of those
interactions, 273 failed; the other 30 action failures were forward movements.
Recorded model requests contain 508 no-route feedback messages and 337
approach-reached messages. These are feedback occurrences, not counts of unique
navigation failures or proof that interaction conditions were satisfied.
Only 42 decisions requested object-history retrieval, and 50 historical image
attachments were actually supplied. The trace therefore establishes that
retrieval is callable, but provides no controlled evidence that its present
usage improves task success or token efficiency.

## Nine historical infrastructure errors remain separate

There are **19 actual runs and 59 attempted episodes** associated with this
cohort: 50 scored completions and nine unscored error prefixes across five
identities. All nine error attempts remain in SQLite with null scores and
separate artifact directories. The 50-score denominator contains none of
those prefixes; no missing score is replaced by zero.

| Historical attempts | Fold indices | Attempts | Actions | Requests / completed responses | Input-token lower bound | Output-token lower bound |
|---|---|---:|---:|---:|---:|---:|
| Original direct / HTTP502 | 32, 43, 45 | 3 | 38 | 59/56 | 439,747 | 75,784 |
| Recovery1 / HTTP502 | 32, 36, 40 | 3 | 57 | 86/83 | 684,565 | 93,001 |
| Recovery2 / HTTP403 | 32, 36, 40 | 3 | 68 | 97/94 | 770,091 | 111,527 |
| Total | Five unique identities | 9 | 163 | 242/233 | 1,894,403 | 280,312 |

Together, the prefixes contain **163 actions, 25 action failures, 172
observations, 242 started decisions, and 233 completed responses**. Their
persisted SDK metadata exposes at least **234 SDK attempts** and at least
**2,174,715 input-plus-output tokens**. The final nine unanswered decisions
have unknown usage; exact consumed usage remains null. These lower bounds
must not be presented as complete outage costs. Scored usage plus these known
prefixes totals at least **26,793,090 tokens**, still excluding unreported final
request consumption and separate diagnostic probes.
[Separate prefix accounting](assets/v8_recovery_complete_20261105/audits/pilot50_analysis.json).

The first six errors were HTTP502 failures. The next three were upstream
HTTP403/ROW5004 system-processing errors with an affiliation-detection dependency
timeout. Existing forwarding and native Responses checks were used to determine
when dispatch could resume; no credential, access-control, model-route, or
benchmark-setting change was made to obtain scores. The new public records use
safe classifications and exact original-error SHA-256 fingerprints. The new archive copies redact raw error strings; their unmodified originals
remain in local output and SQLite records. Old run records are not overwritten.
[Launch readiness snapshot](assets/v8_recovery_complete_20261105/network/recovery3_launch_readiness.safe.json).
That snapshot establishes readiness at launch, not uninterrupted availability.

## Independent verification and audit scope

The [official terminal recheck](assets/v8_recovery_complete_20261105/audits/pilot50_terminal_rescore.json)
recomputed all **50/50** scores from saved evaluator-only terminal metadata and
cleaned/cooled/heated histories. It executed the hash-pinned official ALFRED goal
and helper method bodies without constructing a simulator, replaying episodes,
or calling a model. The goal-source hashes are recorded for `env/tasks.py` and
`gen/utils/game_util.py`. This is independent recomputation of terminal goal
conditions, rather than merely comparing two copies of saved scalar scores.

The accounting audit checked exact metadata and attempt equality against
the local `runs.sqlite` database for **all 19 runs / 59 attempts**, including nine
null-score errors. The hash-locked manifest chain preserves the original four
runs and all 41 original scored records. The original source identity, model,
input, budgets, checkpoint/source hashes, and permitted documentation-only
HEAD difference were checked for every actual run.
[Accounting and SQLite evidence](assets/v8_recovery_complete_20261105/audits/pilot50_analysis.json).

The [actor-boundary audit](assets/v8_recovery_complete_20261105/audits/pilot50_actor_boundary.json) found
**zero recorded violations** across the selected 50 episodes: 2,349
requests/responses/catalogs/traces, 4,748 image references, 2,005 observations,
and 344 interactions. Available prefix evidence was checked separately: 242
requests, 233 responses/catalogs/traces, 487 image references, 172 observations,
and 23 interactions. Its nine unanswered final requests remain explicitly
incomplete. The [lossless full audit](assets/v8_recovery_complete_20261105/audits/pilot50_actor_boundary.full.json.gz)
retains the 8,122-entry artifact manifest.

These checks establish the saved actor-input and action-evidence boundary,
with specific limits: current-mask association is reconstructed from action
source frames, memory IDs, predictions, and saved memory state; original mask
RPC arguments were not separately persisted. SDK catalogs and traces show no
host-tool executions, while full on-wire tool lists were not captured for every
request. The failed final requests lack response metadata needed to join their
SDK run homes. The boundary audit does not semantically judge image pixels or
establish that predicted masks identify the intended simulator object.

A separate [completion validation](assets/v8_completion_validation_20261105.json)
independently joined the 50 selected scores and nine null-score errors to SQLite,
verified unchanged protected NR3D/OpenEQA/SceneFunc3D/keyframe/runtime/adapter
paths, and checked process cleanup. No matching ALFRED runner, project
perception/simulator worker, or Unity executable remained active at the check.
All 15 recorded recovery runner PIDs had exited; the four original direct runs
had no saved PID records, so their current absence was checked by exact argv.
No unrelated process was stopped. Image-reference counts include repeated BEV
and current views and are not counts of distinct image files.

The unchanged executable snapshot had passed Ruff, Black, mypy, **1,284 source
tests**, and **440 adapter tests with 91.34% branch-inclusive coverage** before
these evaluations. These existing results remain applicable to the unchanged
code; they are not claimed as newly rerun tests in this record.
[Recorded verification](assets/v6_upstream_20261105/verification/results.json).

Documentation links, archived asset checksums, privacy checks, and the MkDocs
build passed. The build retained pre-existing warnings in unrelated benchmark
documentation; no ALFRED link or navigation error was reported.
[Publication checks](assets/v8_publication_20261105/checks.json).

## Observed limitations and research interpretation

The completed run demonstrates RGB observation → predicted depth/masks →
incremental object/view/grid memory → action → updated observation → official
terminal evaluation on the frozen pilot. It also exposes substantial limits
in exploration, interaction readiness, and decision efficiency. This balanced
pilot is a development subset, not the full official validation split; its
first identity had already supported earlier infrastructure diagnostics.
No generalization claim follows from five successes on these 50 tasks.

The actor receives H-only language, RGB-derived predictions, motion odometry,
yaw/pitch proprioception, and native action feedback. It receives no evaluator
PDDL goal, expert plan, or task-type label. Coordinate-bearing ALFRED `objectId`
strings are redacted from native action-error feedback; obstacle categories
and GameObject names, including suffixes such as `Chair_...`, can remain. This
is therefore not a strict RGB-and-language-only leaderboard setting. The
perception backend is ALFRED-trained; object association uses geometry and RGB
histograms, not CLIP embeddings. Learned depth uses the documented MLE baseline
and finite range rather than reproducing every FILM confidence-filtering rule.
The methods and preprocessing assumptions remain those documented in
[the implementation guide](../../codex_agent/alfred.md).

The new [i45 interaction case](assets/v8_recovery_complete_20261105/cases/recovery3_i45_interaction_case.json)
records the unseen two-PepperShaker-to-Drawer task: 44 decisions, 47 actions,
37 successful movement/view actions, and 10 failed `OpenObject` attempts,
ending at the failure budget with 0/2 conditions. Four failures report that
the interaction mask could not locate a target; six report object-not-found.
All ten actions link to masks from their immediately preceding observation
through the proposed memory ID: eight masks were predicted as Drawer and two
as Cabinet. The artifact includes RGB/mask hashes, boxes, confidence, and mask
pixel counts. Same-frame association is observable; correct instance alignment,
reachability, and readiness are not established by that association alone.
This analysis used only completed artifacts and supplied no ground truth or
advice back to any running actor.

Earlier posthoc cases remain relevant: fragmented object identities after manipulation,
revisiting a detected target without obtaining a usable mask, premature Stop,
and native simulator interaction exceptions are retained in
[v6's failure analysis](v6_upstream_outage_20261105.md#verified-behavior-costs-and-limits).
Eight Unity `NullReferenceException` toggle failures across three scored
episodes remain attempted failed actions; they are distinct from the nine
unscored upstream-request errors.

Generated local Python policies, a complete interaction-state verifier, and
controlled memory/BEV/retrieval ablations are subsequent research work. The
present data support prioritizing those changes, but do not isolate a causal
benefit of any individual memory or retrieval component. Future comparisons
should use a new versioned configuration and the same frozen fold while
preserving this cohort and its complete failure history.
