# GOAT-Bench reproduction timeline

This archive records navigation runs, their exact episode selection, runtime
source, model, observation budget, execution interface, and raw artifacts.
Paper numbers and completed local runs are reported separately.

| Date | Record | Status |
| --- | --- | --- |
| 2026-10-05 | [SCOPE reproduction setup](scope_reproduction_setup_20261005.md) | All 36 scenes and model assets ready; public mirrors removed the dataset blocker |
| 2026-10-05 | [v1 SCOPE public episode](v1_public_episode_20261005.md) | Completed engineering check: 5/5 subtasks recorded, SR 80.00%, SPL 68.84%; Gemini 3.8 Flash |

The first-episode-per-scene run (36 episodes / 278 subtasks) is running in
Linux `tmux` with two workers on GPUs 0 and 2. Four additional workers process
later episodes on GPUs 3, 4, 5 and 6. All six resumed at 06:14 after SSH tunnel
recovery, retaining 12 saved subtasks. The full 360-episode / 2,669-subtask
evaluation remains incomplete. The five-subtask public episode is a separate
engineering check and is not ranked against either paper fold.

At 07:06, split 5 worker 1 started early in its isolated directory, sharing
GPU 2 with the first-split worker. Seven navigation workers are now active;
the existing controller will reuse the additional worker.

At 07:34, the complementary split 5 worker started on GPU 0 after resource
verification. Eight workers now process splits 1, 3, 4 and 5; all output
directories retain the frozen storage plan.

The 07:42 [progress observation](assets/full2669_progress_20261005_0742/ingested_summary.json)
records 17/2,669 subtasks in three complete episodes, observed SR 76.47% and SPL
52.21%. It remains an incomplete observation of the full requested set.

The 08:00 snapshots record **38/2,669** full-set IDs in seven complete episodes
(observed SR 63.16%, SPL 42.08%) and **12/278** canonical IDs in two complete
episodes (observed SR 66.67%, SPL 40.79%). Both remain incomplete; raw artifacts
and independently verified SQLite records are archived in the setup record.

At 08:21, the two frozen split-6 workers started early on GPUs 3 and 4 after
exact-ID, configuration and resource checks. Ten workers are now active;
the same controller will reuse these stages.

At 08:34, split 7 also started early after the same checks. Twelve workers now
process frozen splits 1, 3, 4, 5, 6 and 7, with two processes per healthy GPU;
future dispatch must wait for a validated slot to be released.

The 08:55 immutable snapshots record **79/2,669** full-set IDs in 13 complete
episodes (observed SR 72.15%, SPL 47.57%) and **20/278** canonical IDs in three
complete episodes (observed SR 75.00%, SPL 45.55%). Raw paired shards, frozen
IDs, launch provenance and verified SQLite records are retained in the
[setup record](scope_reproduction_setup_20261005.md). Both selections remain
incomplete and these observed metrics are not completed benchmark results.

The next [09:05 full-set observation](assets/full2669_progress_20261005_0905/ingested_summary.json)
records **95/2,669** IDs in 15 complete episodes (observed SR 67.37%, SPL
45.50%). Canonical coverage remains 20/278. All 12 workers had fresh model
responses and navigation activity at 09:05; no worker had a terminal status.

The 09:15 snapshots record **106/2,669** full-set IDs in 17 complete episodes
(observed SR 67.92%, SPL 46.43%) and **25/278** canonical IDs in four complete
episodes (observed SR 76.00%, SPL 50.20%). Raw metrics, exact IDs and SQLite
were independently checked; both requested selections remain incomplete.

The [global API audit](assets/scope_20261005/global_api_usage_20261005_0928.json)
retains transport and usage evidence for the ongoing run. Its interleaved
calls include unfinished episodes and earlier attempts, so usage cannot be
attributed to the saved subtasks or canonical fold.

The [12-worker semantic log audit](assets/scope_20261005/semantic_fallback_audit_20261005_0930.json)
records original-policy retries and candidate acceptance, with independently
verified log-prefix hashes. Silent neutral-score defaults remain unobservable;
HTTP success and syntactic recovery do not establish navigation success.

The next immutable observation, captured at 09:47:10 under the `0950` archive
tag, records **122/2,669** full-set IDs in 20 complete episodes (observed SR
66.39%, SPL 46.00%). Canonical coverage remains 25/278. Exact raw/SQLite rows,
original episode IDs and indices, and overall/per-modality metrics passed
independent checks; the requested full set remains incomplete.

The next observation, captured at 10:11:24 under archive tag `1010`, records
**132/2,669** full-set IDs in 21 complete episodes (observed SR 62.88%, SPL
43.75%). It adds all ten subtasks of scene `00890` episode 4, including two
successes. Canonical coverage remains 25/278; this remains an incomplete
observation of the original requested set.

The `1020` observations, captured at 10:21:42–43, record **137/2,669** full-set
IDs in 22 complete episodes (observed SR 64.23%, SPL 44.58%) and **30/278**
canonical IDs in five complete episodes (observed SR 80.00%, SPL 52.93%).
Both add the same complete first episode of scene `00839`, with five successes.
All raw/SQLite and frozen-ID checks passed; both requested selections remain
incomplete.

The `1100` snapshots, captured at 11:01:20, record **164/2,669** full-set IDs
in 26 complete episodes (observed SR 64.63%, SPL 45.10%) and **39/278**
canonical IDs in six complete episodes (observed SR 76.92%, SPL 52.81%).
Both first-split workers have now saved an episode. The full observation also
contains the first split-6 and split-7 checkpoints; their launch provenance
is included. Both requested selections remain incomplete.

The `1120` observation, captured at 11:15:08, records **177/2,669** full-set
IDs in 28 complete episodes (observed SR 66.10%, SPL 46.17%). It adds 13
subtasks from scene `00810` episodes 2 and 5, including 11 successes.
Canonical coverage remains 39/278. Exact raw/SQLite, original ID/index and
complete-episode checks passed; full coverage remains incomplete.

The `1145` observation, captured at 11:43:13, records **185/2,669** full-set
IDs in 29 complete episodes (observed SR 67.03%, SPL 46.57%). It adds all eight
subtasks of scene `00877` episode 3, including seven successes. Canonical
coverage remains 39/278; both requested selections remain incomplete.

The `1150` observation, captured at 11:48:28, records **196/2,669** full-set
IDs in 31 complete episodes (observed SR 65.31%, SPL 45.37%). It adds scene
`00823` episode 4 (five subtasks, zero successes) and scene `00824` episode 5
(six subtasks, four successes). Exact raw/SQLite and complete-episode checks
passed; canonical coverage remains 39/278 and full coverage is incomplete.

The `1155` observation, captured at 11:55:35, records **206/2,669** full-set
IDs in 32 complete episodes (observed SR 66.02%, SPL 46.45%). It adds all ten
subtasks of scene `00829` episode 2, including eight successes. Canonical
coverage remains 39/278; both requested selections remain incomplete.

The `1200` observation, captured at 11:58:22, records **213/2,669** full-set
IDs in 33 complete episodes (observed SR 66.67%, SPL 47.40%). It adds all seven
subtasks of scene `00829` episode 5, including six successes. Exact raw/SQLite
and complete-episode checks passed; canonical coverage remains 39/278 and
full coverage is incomplete.

The `1205` observation, captured at 12:00:40, records **220/2,669** full-set
IDs in 34 complete episodes (observed SR 67.27%, SPL 47.62%). It adds all seven
subtasks of scene `00815` episode 4, including six successes. Exact raw/SQLite
and complete-episode checks passed; canonical coverage remains 39/278 and
full coverage is incomplete.

The `1210` observation, captured at 12:05:35, records **226/2,669** full-set
IDs in 35 complete episodes (observed SR 66.37%, SPL 47.24%). It adds all six
subtasks of scene `00890` episode 3, including two successes. Exact raw/SQLite
and complete-episode checks passed; canonical coverage remains 39/278 and
full coverage is incomplete.

The `1215` observations, captured at 12:09:23, record **240/2,669** full-set
IDs in 37 complete episodes (observed SR 66.67%, SPL 47.78%) and **48/278**
canonical IDs in seven complete episodes (observed SR 79.17%, SPL 57.27%).
Both add the same nine-subtask first episode of scene `00810`, with eight
successes. The full observation also adds scene `00803` episode 4, with five
subtasks and two successes. Both requested selections remain incomplete.

The `1220` observation, captured at 12:19:17, records **249/2,669** full-set
IDs in 38 complete episodes (observed SR 67.87%, SPL 47.55%). It adds all nine
subtasks of scene `00877` episode 2, all successful. Exact raw/SQLite and
complete-episode checks passed; canonical coverage remains 48/278 and full
coverage is incomplete.

The `1235` observation, captured at 12:31:22, records **256/2,669** full-set
IDs in 39 complete episodes (observed SR 67.58%, SPL 47.22%). It adds all seven
subtasks of scene `00827` episode 4, including four successes. Exact raw/SQLite
and complete-episode checks passed; canonical coverage remains 48/278 and
full coverage is incomplete.

The `1240` observation, captured at 12:33:39, records **266/2,669** full-set
IDs in 40 complete episodes (observed SR 66.54%, SPL 46.34%). It adds all ten
subtasks of scene `00853` episode 3, including four successes. Exact raw/SQLite
and complete-episode checks passed; canonical coverage remains 48/278 and
full coverage is incomplete.

The `1245` observation, captured at 12:35:54, records **282/2,669** full-set
IDs in 42 complete episodes (observed SR 66.67%, SPL 46.35%). It adds scene
`00800` episode 4 (seven subtasks, five successes) and scene `00876` episode 6
(nine subtasks, six successes). Exact raw/SQLite and complete-episode checks
passed; canonical coverage remains 48/278 and full coverage is incomplete.

The `1250` observation, captured at 12:38:33, records **292/2,669** full-set
IDs in 43 complete episodes (observed SR 67.12%, SPL 47.20%). It adds all ten
subtasks of scene `00808` episode 5, including eight successes. Exact raw/SQLite
and complete-episode checks passed; canonical coverage remains 48/278 and
full coverage is incomplete.

The `1255` observation, captured at 12:44:23, records **297/2,669** full-set
IDs in 44 complete episodes (observed SR 67.68%, SPL 48.01%). It adds all five
subtasks of scene `00869` episode 3, all successful. Exact raw/SQLite and
complete-episode checks passed; canonical coverage remains 48/278 and full
coverage is incomplete.

The `1300` observation, captured at 12:54:34, records **307/2,669** full-set
IDs in 45 complete episodes (observed SR 68.40%, SPL 48.87%). It adds all ten
subtasks of scene `00862` episode 4, including nine successes. Exact raw/SQLite
and complete-episode checks passed; canonical coverage remains 48/278 and
full coverage is incomplete.

The `1305` observation, captured at 12:57:06, records **317/2,669** full-set
IDs in 46 complete episodes (observed SR 69.09%, SPL 49.68%). It adds all ten
subtasks of scene `00839` episode 4, including nine successes. Exact raw/SQLite
and complete-episode checks passed; canonical coverage remains 48/278 and
full coverage is incomplete.

The `1310` observations, captured at 13:00:22, record **324/2,669** full-set
IDs in 47 complete episodes (observed SR 68.52%, SPL 49.21%) and **55/278**
canonical IDs in eight complete episodes (observed SR 74.55%, SPL 53.55%).
Both add scene `00853` episode 0, with seven subtasks and three successes.
Canonical artifacts are copied from the same immutable full snapshot;
both requested selections remain incomplete.

The `1315` observations, captured at 13:07:10, record **336/2,669** full-set
IDs in 49 complete episodes (observed SR 68.45%, SPL 48.73%) and **62/278**
canonical IDs in nine complete episodes (observed SR 74.19%, SPL 51.57%).
Both add scene `00829` episode 0 (seven subtasks, five successes); full also
adds scene `00823` episode 2 (five subtasks, three successes). Both requested
selections remain incomplete.

The `1320` observation, captured at 13:17:55, records **359/2,669** full-set
IDs in 52 complete episodes (observed SR 69.64%, SPL 49.50%). It adds scene
`00810` episode 6 (nine subtasks, nine successes), scene `00815` episode 3
(nine subtasks, six successes) and scene `00832` episode 4 (five subtasks,
five successes). Canonical coverage remains 62/278; both requested
selections remain incomplete.

The `1335` observation, captured at 13:30:39, records **375/2,669** full-set
IDs in 54 complete episodes (observed SR 70.40%, SPL 49.83%). It adds scene
`00808` episode 6 (ten subtasks, eight successes) and scene `00890` episode 2
(six subtasks, six successes). Exact raw/SQLite and complete-episode checks
passed; canonical coverage remains 62/278 and full coverage is incomplete.

## Completed local runs

SR and SPL use SCOPE's distance-based metrics and are percentages.

| Record | Selection | Model | Recorded / expected | SR | SPL |
| --- | --- | --- | ---: | ---: | ---: |
| [v1](v1_public_episode_20261005.md) | Official public scene `00861-GLAQ4DNUx5U`, original episode `0` | `gemini-3.8-flash`, thinking budget 2,000 | 5 / 5 | 80.00 | 68.84 |

## References

- [GOAT-Bench official implementation](https://github.com/Ram81/goat-bench).
- [SCOPE official implementation](https://github.com/mrwangyou/SCOPE).
- [Paper references and protocol boundaries](leaderboard.md).
