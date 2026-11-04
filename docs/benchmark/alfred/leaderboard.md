# ALFRED local leaderboard

Only a one-task diagnostic is available; it does not estimate full-fold accuracy.

| Version | Fold | Observation setting | Scored episodes | SR / GC / PLWSR / PLWGC |
|---|---|---|---:|---|
| [v3 early Stop](v3_smoke1_20261105.md) | pilot50 prefix1, seen | H; predicted depth/masks, odometry and proprioception | 1 | 0 / 0 / 0 / 0 |
| [v1 setup](v1_setup_20261105.md) | pilot50 protocol: 25 seen + 25 unseen | Predicted depth/masks + action odometry and proprioception | 0 | Not evaluated |

Future rows must report seen and unseen separately and use the frozen fold,
matching language inputs, perception models, action interface and budgets.
See the [protocol and metric definitions](README.md).

## Diagnostic attempts

The [first frozen-prefix rollout](v2_smoke1_20261105.md) ended with a decision
contract error after 10 successful simulator actions. It has 0/1 scored coverage
and is excluded from task-success metrics; its failure and logs remain archived.
