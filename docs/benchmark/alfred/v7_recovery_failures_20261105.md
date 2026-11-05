# ALFRED v7 — two unsuccessful recovery rounds, 2026-11-05

Recovery1 and recovery2 each started three of the nine missing pilot tasks.
All six invocations ended with an upstream request error and **no terminal
score**. The original **41/50 scores remain unchanged**. Together with the
three original direct-run errors in [v6](v6_upstream_outage_20261105.md), this
historical snapshot contains **10 actual runs and 50 episode attempts: 41
scored attempts plus nine unscored errors across five task identities**.
It is not a completed pilot50 result, and error attempts are not counted as
task failures with score zero.

This record ends after recovery2. Recovery3 outcomes belong in a subsequent
record. The six recovery jobs that had not started at the end of recovery2
retained their reserved IDs; those IDs do not imply that the jobs ran during
recovery1.

## Configuration and immutable identities

| Setting | Value |
|---|---|
| Branch | `feat/alfred-benchmark` |
| Original direct-run HEAD | `fb408c34ab85284b32387002e92745b92983a173` |
| Recovery1/2 HEAD | `7010ba8f54088bdad50ad30ba8ac709765c0d1c6` |
| Identical executed-source SHA-256 | `24f41534d1c3929a2ef65eb0f6626dedebafe6131eb78f2342c933eb19ea7370` |
| Original frozen-fold SHA-256 | `1d22afc51da128395a6d65a250d9dc603c8f50f0252942ca28b335568f646d4e` |
| Model / reasoning | `gpt-5.4-2026-03-05` / medium; native Responses |
| Language / perception / pose | H; FILM predicted depth and masks; action odometry with yaw/pitch proprioception |
| Action / failure / decision budgets | 1000 / 10 / 60 per episode |
| Worker / model-turn deadlines | 180 s / 300 s |
| Recovery selection | Original fold; `--num-shards 50 --shard-index <original-index>` |

The HEAD difference is a documentation-only change. The launch gate requires
the same branch, clean tracked tree and identical complete executed-source
digest; every run preserves its actual HEAD. Original run and summary hashes
remain locked. No original scored task was rerun, no fold was resampled, and
no actor configuration, perception threshold, budget, or evaluation method was
changed. The existing three benchmark implementations remained isolated as
recorded in v6.

Both rounds used queues `[32,43,47]`, `[36,44,48]`, and `[40,45,49]`; each queue
stopped on its first failed invocation, so only 32, 36 and 40 started. Failed
invocations received new output/run IDs for the next round. The six other
recovery entries had not started at this snapshot. The nine missing frozen
indices remained **32, 36, 40, 43, 44, 45, 47, 48, 49**, all unseen tasks.

The exact per-run argument vectors are archived as `command.txt` beside each
run below. [Recovery1](assets/v7_recovery_failures_20261105/plans/recovery1.redacted.json)
and [recovery2](assets/v7_recovery_failures_20261105/plans/recovery2.redacted.json)
preserve the planned identities, queue order, configuration and source locks.
These are historical plan snapshots; queue status files establish what actually
ran. Launchers retain the run-local `NO_PROXY`/`no_proxy` loopback exclusions
and reuse the same queue driver documented in v6.

## Actual error prefixes

All run IDs share prefix `REPRO_20261105_ALFRED_V3_20261105_pilot50_`.
Decision counts include the final request that did not return a valid response.

| Run suffix | Actions | Failed actions | Decisions / valid responses | Observed input / output tokens | Score |
|---|---:|---:|---:|---:|---|
| [i32_recovery1](assets/v7_recovery_failures_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i32_recovery1/run.redacted.json) | 16 | 4 | 26 / 25 | 210,256 / 33,245 | None |
| [i36_recovery1](assets/v7_recovery_failures_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i36_recovery1/run.redacted.json) | 19 | 2 | 30 / 29 | 221,761 / 29,208 | None |
| [i40_recovery1](assets/v7_recovery_failures_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i40_recovery1/run.redacted.json) | 22 | 4 | 30 / 29 | 252,548 / 30,548 | None |
| [i32_recovery2](assets/v7_recovery_failures_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i32_recovery2/run.redacted.json) | 13 | 1 | 25 / 24 | 182,685 / 30,973 | None |
| [i36_recovery2](assets/v7_recovery_failures_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i36_recovery2/run.redacted.json) | 34 | 6 | 42 / 41 | 341,735 / 41,232 | None |
| [i40_recovery2](assets/v7_recovery_failures_20261105/runs/REPRO_20261105_ALFRED_V3_20261105_pilot50_i40_recovery2/run.redacted.json) | 21 | 5 | 30 / 29 | 245,671 / 39,322 | None |
| Six recovery error prefixes | **125** | **22** | **183 / 177** | **1,454,656 / 204,528** | None |
| Three original direct-run error prefixes, from v6 | 38 | 3 | 59 / 56 | 439,747 / 75,784 | None |
| All nine error prefixes | **163** | **25** | **242 / 233** | **1,894,403 / 280,312** | None |

Token values are **lower bounds from returned responses**, not exact total
outage costs. The final failed request in each of the nine error attempts has
unknown usage. These counters are separate from the 41 scored episodes and
must not be added to a task-success denominator. The raw terminal-score field
is null in the run artifacts and SQLite; no terminal evaluation was invented
after the request error.

The [failure accounting](assets/v7_recovery_failures_20261105/failure_accounting.json)
contains per-attempt source identity, error classification/hash and token
lower bounds. The [read-only SQLite check](assets/v7_recovery_failures_20261105/sqlite_audit.json)
matched metadata and every attempted-episode record for all ten historical
runs: 41 scored and nine null-score errors. The official scores and matched
41-task metrics remain those in v6; they are not extrapolated to 50 tasks.

## Connectivity evidence and its limits

Recovery1 followed a successful synthetic native Responses probe at
01:57:42 UTC; that probe returned in 4.85 seconds. The three task runs started
at 01:58:37 UTC and later failed with HTTP502 between 02:15:51 and 02:16:19 UTC.
The live operator observation at approximately 02:14 UTC was that TCP remained
reachable while the local SOCKS greeting timed out. That observation is
minute-resolution context rather than a saved continuous trace.
[Probe02 metadata](assets/v7_recovery_failures_20261105/network/upstream_recovery_probe_02.safe.json)
records only the successful transport check, not sustained availability.

Before recovery2, a second synthetic native Responses probe succeeded at
02:28:47 UTC in 4.84 seconds. The three runs started at 02:32:07 UTC and failed
between 02:45:17 and 02:51:29 UTC with **HTTP403 / ROW5004 / dependency timeout /
processing_error**. All **17/17 sampled TCP and SOCKS greetings succeeded**
from 02:39:29 to 02:51:31 UTC. This distinguishes the observed error from the
previous SOCKS interruption; successful sampled greetings do not prove
continuous end-to-end service availability or identify the remote cause.
[Probe03 metadata](assets/v7_recovery_failures_20261105/network/upstream_recovery_probe_03.safe.json)
and the [forwarding observations](assets/v7_recovery_failures_20261105/network/recovery2_forward_health.jsonl)
preserve these separate measurements.

A single diagnostic at 03:06:46 UTC resubmitted the saved failing i32 prompt
and images through the same authenticated native Responses route, model,
medium reasoning setting and 300-second turn deadline. The diagnostic set
`max_finalization_retries=0`; it did not reproduce the actor's full retry
policy. It succeeded once in 31.65 seconds. Its returned action was **not
executed**; no simulator action or ALFRED score resulted. This check
shows that the saved request could complete later; it neither proves a remote
implementation cause nor establishes reliable future service.
[Allowlisted diagnostic metadata](assets/v7_recovery_failures_20261105/network/full_request_probe_04.safe.json)
excludes raw prompts, response text, SDK metadata and configuration.

## Archive safety and comparison limits

Published `*.redacted.json` and `actions.redacted.jsonl` files are derived
copies, not byte-identical original artifacts. Error text is replaced with a
safe category and SHA-256 of the exact UTF-8 source error string. Absolute URLs
and sensitive named fields are removed if encountered. The
[archive manifest](assets/v7_recovery_failures_20261105/manifest.json) retains
raw-source hashes, archive hashes and each transformation. Original complete
logs remain under their immutable `outputs/alfred/<run-id>/` paths. Gateway
error plaintext, credentials, provider configuration, SDK home files and
raw console logs are not copied into this archive.

Neither failed recovery round is an algorithm ablation or a new accuracy
estimate. They preserve the v6 implementation and its known perception,
interaction, memory-association, exploration and stopping limitations. The
historical FILM H reference uses different full-policy/training/perception and
budget settings; no new controlled comparison follows from these transport
failures. Recovery3 changes queue ordering only, putting previously unstarted
entries first, while keeping per-episode settings frozen. Its completed scores
must be verified separately before the cohort can be called pilot50.
