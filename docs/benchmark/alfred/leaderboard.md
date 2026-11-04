# ALFRED local leaderboard

The current pilot is incomplete after an upstream outage. Seen coverage is
complete; the unseen and combined rows describe only scored tasks and do not
estimate the missing nine outcomes.

| Version | Fold | Observation setting | Scored episodes | SR / GC / PLWSR / PLWGC |
|---|---|---|---:|---|
| [v6 seen](v6_upstream_outage_20261105.md) | Frozen pilot seen25 | H; predicted FILM depth/masks, odometry and proprioception | 25/25 | 12.00% / 14.29% / 7.78% / 12.16% |
| [v6 unseen, partial](v6_upstream_outage_20261105.md) | Frozen pilot unseen25 | Same setting | 16/25 | 12.50% / 28.21% / 8.11% / 23.91% |
| [v6 combined, partial](v6_upstream_outage_20261105.md) | Frozen pilot50 | Same setting | 41/50 | 12.20% / 19.61% / 7.89% / 16.08% |
| [v3 early Stop](v3_smoke1_20261105.md) | pilot50 prefix1, seen | H; predicted depth/masks, odometry and proprioception | 1 | 0 / 0 / 0 / 0 |
| [v1 setup](v1_setup_20261105.md) | pilot50 protocol: 25 seen + 25 unseen | Predicted depth/masks + action odometry and proprioception | 0 | Not evaluated |

Future rows must report seen and unseen separately and use the frozen fold,
matching language inputs, perception models, action interface and budgets.
See the [protocol and metric definitions](README.md).

## Diagnostic attempts

The [first frozen-prefix rollout](v2_smoke1_20261105.md) ended with a decision
contract error after 10 successful simulator actions. It has 0/1 scored coverage
and is excluded from task-success metrics; its failure and logs remain archived.
