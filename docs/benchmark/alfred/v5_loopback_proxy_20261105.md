# ALFRED v5 — loopback discovery failure and direct-route recovery, 2026-11-05

The scalar policy wire restored real simulator motion and masked interactions.
The standalone run then stopped on a transient HTTP 502 while checking the
local adapter's health. The four concurrent pilot shards were deliberately
interrupted before their first terminal scores so their loopback routing could
be corrected consistently. **None of these five attempts has a task score.**
Their errors and interruptions remain separate from scored task failures.

| Shared configuration | Value |
|---|---|
| Branch / clean HEAD | `feat/alfred-benchmark` / `f7c46cdbeb70070f1bcedd2dd4dabf398d2a1903` |
| Executed-source SHA-256 | `24f41534d1c3929a2ef65eb0f6626dedebafe6131eb78f2342c933eb19ea7370` |
| Model / reasoning | `gpt-5.4-2026-03-05` / medium |
| Input / perception | High-level language; FILM predicted depth and instance masks |
| Fold SHA-256 | `1d22afc51da128395a6d65a250d9dc603c8f50f0252942ca28b335568f646d4e` |
| Action / failure budgets | 1000 / 10 |
| Decision budget | Standalone 120; each pilot shard 60 |

## Preserved attempts

| Run | Selected | Terminal status | Valid decisions | Physical actions | Action failures | Score |
|---|---:|---|---:|---:|---:|---|
| `ALFRED_V3_20261105_smoke1` | 1 | Infrastructure error | 25 | 19 | 2 | None |
| `ALFRED_V3_20261105_pilot50_s0` | 13 | Interrupted | 13 | 10 | 2 | None |
| `ALFRED_V3_20261105_pilot50_s1` | 13 | Interrupted | 14 | 9 | 1 | None |
| `ALFRED_V3_20261105_pilot50_s2` | 12 | Interrupted | 13 | 7 | 2 | None |
| `ALFRED_V3_20261105_pilot50_s3` | 12 | Interrupted | 16 | 15 | 2 | None |

The standalone run attempted decision 26 after 25 valid model responses. Its
two `PickupObject` actions used the Pillow's current predicted masks from
frames 14 and 17; both returned `object is not visible.` The detection occupied
only a small left-edge region, so detection visibility did not establish
interaction readiness. These were genuine simulator action failures. The
subsequent exception was different: `CodexConfigError` from
`http://127.0.0.1:8787/health`, HTTP 502, before the next model turn.

Each pilot shard retained one interrupted first-episode attempt; together they
have four interrupted attempts, 46 unattempted entries, and zero terminal
scores. They must not be counted as SR 0 or combined into a scored 50-task
result. The [attempt and SQLite audit](assets/v5_loopback_20261105/attempt_status.json)
confirms one `error` and four `interrupted` records, all with `score_json=NULL`.
Their simulator, perception, and Unity workers were closed after termination.

Every run's exact command, configuration, source/checkpoint hashes, and selected
IDs are preserved in its metadata:
[standalone](assets/v5_loopback_20261105/ALFRED_V3_20261105_smoke1/run.json),
[shard 0](assets/v5_loopback_20261105/ALFRED_V3_20261105_pilot50_s0/run.json),
[shard 1](assets/v5_loopback_20261105/ALFRED_V3_20261105_pilot50_s1/run.json),
[shard 2](assets/v5_loopback_20261105/ALFRED_V3_20261105_pilot50_s2/run.json), and
[shard 3](assets/v5_loopback_20261105/ALFRED_V3_20261105_pilot50_s3/run.json).
Each corresponding asset directory also contains `command.txt`, `summary.json`,
`attempt.json`, `partial_execution.json`, and `actions.jsonl`. Token counts in
partial execution records are explicitly lower bounds. Full image and model
traces remain under the matching `outputs/alfred/<run-id>/` directory.

## Observed routing and launch-only repair

The [environment probe](assets/v5_loopback_20261105/proxy_environment.json)
found that `urllib.request.getproxies()` selected `127.0.0.1:17890` for HTTP and
HTTPS, while `proxy_bypass('127.0.0.1')` was false. The configured `NO_PROXY` and
`no_proxy` values did not exempt loopback. Provider discovery uses urllib's
default proxy-aware opener, so local health requests were eligible for that
proxy route. A subsequent curl health request returned HTTP 200 and `ready`.
This makes a transient proxy failure plausible; it does not establish the
internal cause of the earlier 502.

Recovery changes only each new launch's shell environment, preserving existing
exclusions:

```bash
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,$no_proxy}"
```

The [direct-route verification](assets/v5_loopback_20261105/loopback_probe.json)
confirmed bypass changed from false to true and all 20 subsequent local health
checks returned HTTP 200 with `status=ready`. This verifies the corrected
route, not the remote/internal cause of the original failure. No Python source,
adapter implementation, shared benchmark implementation, model, or frozen fold
is changed by this repair.

The planned replacement runs are
`REPRO_20261105_ALFRED_V3_20261105_pilot50_s0_direct` through
`REPRO_20261105_ALFRED_V3_20261105_pilot50_s3_direct`. They retain the same
executed-source hash, model, medium effort, fold, strided shard assignment, and
60-decision cap. Their new metadata and outcomes must be archived independently;
this record makes no claim that the replacement runs have completed.
The [checksum manifest](assets/v5_loopback_20261105/sha256.json) covers all
preserved evidence, including the [launch environment](assets/v5_loopback_20261105/launch_environment.txt).
