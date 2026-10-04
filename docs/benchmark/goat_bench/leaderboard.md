# GOAT-Bench reference results

These are **author-reported results**, not reproductions performed by this
project. SR and SPL are percentages. Do not rank rows across different episode
selections, models, sensors, or movement interfaces.

| Method | Episode selection | Model | SR | SPL | Source |
| --- | --- | --- | ---: | ---: | --- |
| SCOPE | First episode per validation scene, 36 episodes / 278 subtasks | GPT-4o-2024-11-20 | 73.7 | 53.5 | [Paper](https://arxiv.org/html/2511.08935v1) |
| SCOPE | Full val_unseen, 360 episodes | GPT-4o-2024-11-20 | 66.8 | 46.5 | [Paper Appendix B](https://arxiv.org/html/2511.08935v1) |

The SCOPE checkout used here contains 2,669 subtasks across its complete
val_unseen task files. Its execution interface follows TSDF waypoints and a
Habitat pathfinder, with a scene-size-dependent macro-step budget; it does not
implement the original GOAT-Bench budget as 500 discrete movement actions.

The requested Gemini replacement is a separate model configuration. Results
from that configuration must identify `gemini-3.8-flash` and must not be
presented as recovery of the paper's GPT-4o numbers.

## Completed local engineering checks

These runs verify the implementation and infrastructure. Their selection is
listed explicitly; they do not establish a rank on the paper's 278-subtask or
full validation protocol.

| Record | Episode selection | Model | Recorded / expected | Distance SR | Distance SPL |
| --- | --- | --- | ---: | ---: | ---: |
| [v1 public episode](v1_public_episode_20261005.md) | Official public HM3D scene `00861-GLAQ4DNUx5U`, original episode `0` | `gemini-3.8-flash`, thinking budget 2,000 | 5 / 5 | 80.00 | 68.84 |

The v1 launcher exited 0 and all five frozen IDs have matching success/SPL
records. Four tasks succeeded; the book task failed the distance threshold.
The 36-scene / 278-subtask run is in progress, and the full 2,669-subtask run
has no completed result yet.
