# ALFRED local leaderboard

Full milestone releases are certified on `valid_full1641` (`820 valid_seen + 821 valid_unseen = 1,641`), while rapid architectural iterations and ablations are evaluated on the canonical 150-trajectory stratified subset [`valid_strat150`](valid_strat150_protocol_20261108.md) (`75 valid_seen + 75 valid_unseen`). Earlier development milestones (`v10`–`v15`) were evaluated on the 5-episode `prefix5` diagnostic cohort.

| Version | Fold | Observation setting | Scored episodes | SR / GC / PLWSR / PLWGC |
|---|---|---|---:|---|
| [v16 full seen](v16_full_valid1641_20261107.md) | Official `valid_seen` (full 820) | H; simulator GT depth + GroundingDINO/SAM/CLIP, online 3D memory/FMM + macro tools | 820/820 | **30.37% / 36.94% / 15.99% / 23.00%** |
| [v16 full unseen](v16_full_valid1641_20261107.md) | Official `valid_unseen` (full 821) | Same setting | 821/821 | **18.64% / 30.47% / 9.38% / 17.54%** |
| [v16 full combined](v16_full_valid1641_20261107.md) | Official `valid_full1641` (1,641) | Same setting | 1641/1641 | **24.50% / 33.70% / 12.83% / 20.39%** |
| [v16 matched strat150 seen](valid_strat150_protocol_20261108.md) | Frozen `valid_strat150` seen75 | Same setting | 75/75 | **30.67% / 36.98% / 16.04% / 22.99%** |
| [v16 matched strat150 unseen](valid_strat150_protocol_20261108.md) | Frozen `valid_strat150` unseen75 | Same setting | 75/75 | **18.67% / 30.26% / 9.38% / 17.56%** |
| [v16 matched strat150 combined](valid_strat150_protocol_20261108.md) | Frozen `valid_strat150` (150) | Same setting | 150/150 | **24.67% / 33.59% / 12.84% / 20.38%** |
| [v15 root-cause fixes](v15_root_cause_fixes_prefix5_20261106.md) | Original pilot15 prefix5, all seen | Same setting | 5/5 | **60.00% / 58.33% / 44.70% / 44.11%** |
| [v14 simulator GT depth](v14_simulator_gt_depth_prefix5_20261105.md) | Original pilot15 prefix5, all seen | Same setting | 5/5 | **20.00% / 25.00% / 8.89% / 17.13%** |
| [v12 macro tools + 85 categories](v12_macro_tools_85cats_prefix5_20261105.md) | Original pilot15 prefix5, all seen | H; DA3Metric + 85-category GroundingDINO/SAM/CLIP, online 3D memory/FMM + macro tools | 5/5 | **20.00% / 25.00% / 7.68% / 14.34%** |
| [v10 online mapping](v10_prefix5_20261105.md) | Original pilot15 prefix5, all seen | H; DA3Metric + fixed-vocabulary GroundingDINO/SAM/CLIP, online memory/FMM | 5/5 | 0.00% / 8.33% / 0.00% / 3.78% |

Future rows must report seen and unseen separately and use the frozen fold
(`valid_strat150.json` for rapid iteration/ablation or `valid_full1641.json` for
full validation), matching language inputs, perception models, action interface
and budgets. See the [protocol and metric definitions](README.md).
