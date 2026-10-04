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

## Completed local runs

SR and SPL use SCOPE's distance-based metrics and are percentages.

| Record | Selection | Model | Recorded / expected | SR | SPL |
| --- | --- | --- | ---: | ---: | ---: |
| [v1](v1_public_episode_20261005.md) | Official public scene `00861-GLAQ4DNUx5U`, original episode `0` | `gemini-3.8-flash`, thinking budget 2,000 | 5 / 5 | 80.00 | 68.84 |

## References

- [GOAT-Bench official implementation](https://github.com/Ram81/goat-bench).
- [SCOPE official implementation](https://github.com/mrwangyou/SCOPE).
- [Paper references and protocol boundaries](leaderboard.md).
