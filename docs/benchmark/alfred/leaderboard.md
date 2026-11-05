# ALFRED local leaderboard

The frozen pilot is complete. v8 preserves all 41 earlier scores and adds the
nine previously unscored identities under the same configuration. Historical
v6 rows remain partial snapshots and are not matched full-cohort comparisons.

| Version | Fold | Observation setting | Scored episodes | SR / GC / PLWSR / PLWGC |
|---|---|---|---:|---|
| [v8 seen](v8_pilot50_complete_20261105.md) | Frozen pilot seen25 | H; predicted FILM depth/masks, odometry and proprioception | 25/25 | 12.00% / 14.29% / 7.78% / 12.16% |
| [v8 unseen](v8_pilot50_complete_20261105.md) | Frozen pilot unseen25 | Same setting | 25/25 | 8.00% / 17.46% / 4.45% / 13.10% |
| [v8 combined](v8_pilot50_complete_20261105.md) | Frozen pilot50 | Same setting | 50/50 | 10.00% / 15.87% / 6.19% / 12.61% |
| [v6 seen](v6_upstream_outage_20261105.md) | Frozen pilot seen25 | H; predicted FILM depth/masks, odometry and proprioception | 25/25 | 12.00% / 14.29% / 7.78% / 12.16% |
| [v6 unseen, partial](v6_upstream_outage_20261105.md) | Frozen pilot unseen25 | Same setting | 16/25 | 12.50% / 28.21% / 8.11% / 23.91% |
| [v6 combined, partial](v6_upstream_outage_20261105.md) | Frozen pilot50 | Same setting | 41/50 | 12.20% / 19.61% / 7.89% / 16.08% |
| [v3 early Stop](v3_smoke1_20261105.md) | pilot50 prefix1, seen | H; predicted depth/masks, odometry and proprioception | 1 | 0 / 0 / 0 / 0 |
| [v1 setup](v1_setup_20261105.md) | pilot50 protocol: 25 seen + 25 unseen | Predicted depth/masks + action odometry and proprioception | 0 | Not evaluated |

Future rows must report seen and unseen separately and use the frozen fold,
matching language inputs, perception models, action interface and budgets.
See the [protocol and metric definitions](README.md).

The [v9 rewrite](v9_online_mapping_20261105.md) uses frozen pilot15 and has no
scored results yet. A future comparison must restrict v8 to those same identities.

## Diagnostic attempts

The [two failed recovery rounds](v7_recovery_failures_20261105.md) added six
unscored errors. All nine historical error attempts, including the three from
v6, remain outside the 50-score denominator and retain separate cost records.

The [first frozen-prefix rollout](v2_smoke1_20261105.md) ended with a decision
contract error after 10 successful simulator actions. It has 0/1 scored coverage
and is excluded from task-success metrics; its failure and logs remain archived.
