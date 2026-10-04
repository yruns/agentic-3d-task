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
