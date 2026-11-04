# ALFRED: incremental scene memory and task execution

ALFRED adds interactive task execution to the prepared-scene tasks in this
repository. Each action produces a new RGB observation; predicted depth and
instance masks update an episode-local object/view/grid memory. The policy uses
the current image, a gridded BEV, and retrieved historical views to choose its
next action. The first implementation is an integration baseline, not a claim
that the proposed program-generation method is complete.

**Status on 2026-11-05:** 41/50 frozen validation tasks reached official scoring,
with five successes. Three upstream transport failures and six unattempted
tasks remain; the exact nine-task recovery plan is prepared. All 41 scores
passed an independent official-goal recheck. This is an integration baseline
with partial pilot coverage, not completed pilot50 accuracy.
See the [results and failure evidence](../benchmark/alfred/v6_upstream_outage_20261105.md) and
[benchmark protocol](../benchmark/alfred/README.md).

## Inputs and information boundary

The default actor input is the high-level `task_desc` (H). Optional H+L adds the
selected annotation's `high_descs`. Evaluator-only information includes the
trajectory's task type, PDDL parameters, expert plan, initial scene restoration,
and terminal goal conditions. The actor never receives those fields, simulator
object identities, world positions, reachable-position lists, or true masks and
depth. Simulator mask-to-object dispatch and official scoring remain inside the
simulator worker.

Native action-error feedback can still contain obstacle names/categories after
simulator IDs are redacted; it is part of the declared action-feedback input.

The first policy runtime explicitly supports registered **native Responses**
models through `codex_adapter`. Its per-turn SDK catalog removes all host tools;
the real outgoing request was checked to contain **zero tools**. Shell, patch,
image-file browsing, MCP, plugins and subagents cannot provide a route to hidden
trajectory data. Unsupported provider protocols fail explicitly. Image evidence
and actions remain available through the typed decisions executed by the runner.

This baseline **does use yaw/pitch proprioception, action-success feedback, and
action odometry**. Translation is integrated from successful 0.25 m forward
moves. It is therefore a validation research setting, not the strict RGB-and-
language-only test setting described by the
[official leaderboard rules](https://askforalfred.com/leaderboard/leaderboard.html).
Sensor assumptions must remain fixed in comparisons.

## Implementation

| Component | Responsibility |
|---|---|
| `codex_agent.alfred.dataset` | Validate raw trajectories, select language annotations, verify frozen fold membership and hashes |
| `simulator` / `simulator_worker` | Own an AI2-THOR process, restore episodes, execute bounded actions, score terminal state |
| `perception` / `perception_worker` | Persistent RGB-only neural inference in the existing FILM interpreter |
| `geometry`, `memory`, `rendering` | Backproject masks; associate and fuse instances; retain source observations; maintain grid evidence; render BEV |
| `odometry`, `navigation` | Episode-relative camera transforms and local navigation over the observed map |
| `policy`, `execution` | Structured Codex decisions, bounded action sequences, reobservation and memory updates |
| `actor_runtime` | Task-scoped tool-free SDK catalog, native Responses transport and actor isolation |
| `metrics`, `results`, `provenance` | Official metric aggregation, SQLite attempt archive, clean-source checks |
| `codex_agent.cli.run_alfred` | Evaluation CLI |

Instance association currently combines geometric proximity and RGB histograms;
it does not use CLIP embeddings. Unknown, occupied, and observed-free cells are
separate from observation coverage. Interaction updates invalidate affected
historical evidence, while retaining its provenance. Retrieval can select
complementary object/grid views and explicitly request stale history. These
mechanisms require validation under predicted depth, missed detections, object
motion, and repeated views; infrastructure smoke checks do not establish their
task-level effectiveness.

Actor retrieval excludes the attached current frame and historical frames
already returned in the episode. Exhausted evidence produces an explicit
request to obtain a new observation, without substituting an automatic action.
The internal command selects one object or grid target. Its provider wire
format uses one explicit operation enum and required scalar parameters; the
operation determines which parameters apply. This avoids the configured
provider's demonstrated problems with nested schema alternatives, while
retaining typed internal commands and the original raw response for audit.

Perception imports the published [FILM architectures](https://github.com/soyeonm/FILM)
from the external checkout, rather than vendoring their implementation. It uses
the released ALFRED-trained object/receptacle Mask R-CNN models and two depth
models. The nearest 0°/45° depth checkpoint is selected by camera pitch. Depth is
the maximum-likelihood bin in metres: 0.1 m quantization and a maximum predicted
value of 4.9 m. Other camera pitches are not independently calibrated models.
Detection and mask thresholds default to 0.5. RGB input must be 300×300.

The depth checkpoint receives BGR channels, while Mask R-CNN and archived
images retain RGB. This channel distinction follows the released model's input
contract. Unlike FILM's complete perception pipeline, this baseline retains
unfiltered maximum-likelihood depth and uses nearest-checkpoint pitch routing;
it is not an exact reproduction of FILM. Later ground observations can clear
earlier perceptual occupancy errors. Physical collision evidence is retained
separately until traversal or a relevant interaction invalidates it.

The external segmentation loader requests COCO initialization before loading its
complete checkpoint. A temporary constructor override disables that redundant
download (`weights=None`, `weights_backbone=None`); the published model topology
and trained checkpoint are retained. This is an ALFRED-trained starting point,
not a claim that these are the newest or strongest perception models.

Camera intrinsics use the measured 60° vertical FOV and a 1.576 m standing camera
height. An isolated calibration checked these constants; it does not provide
per-episode world geometry to the actor. The legacy worker stays in FILM's
Python 3.8 environment, while the main runner uses `agentic3d`. GPU inference is
pinned to **physical GPU 0**; physical GPU 1 is never used.

## Local prerequisites

Verified paths on this machine:

| Resource | Path |
|---|---|
| Raw trajectory root | `/home/ysh/Datasets/FILM/alfred/data/json_2.1.0` |
| Official splits | `/home/ysh/codebase/FILM/alfred_data_small/splits/oct21.json` |
| Official simulator/scorer source | `/home/ysh/Datasets/FILM/alfred` |
| FILM source | `/home/ysh/codebase/FILM` |
| Legacy interpreter | `/home/ysh/codebase/FILM/.venv/bin/python` |
| Model root | `/home/ysh/Datasets/FILM/Pretrained_Models_FILM` |

Keep AI2-THOR **2.1.0**, Linux build **201909061227**. Do not upgrade it to a
current AI2-THOR release: scene restoration and action behavior are benchmark
dependencies. No offline RGB/ResNet archive is required for online simulation.
FILM's local `/home/ysh/codebase/FILM/reproduction/QUICKSTART_ZH.md` describes
the existing assets. Check the display server's rendering GPU separately from
the predictor's physical GPU 0; this machine's verified Xorg configuration uses
physical GPU 6 for rendering.

Before launching, activate `agentic3d`, inspect `nvidia-smi`, verify the configured
X display and its authentication, and check the adapter:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
curl -fsS http://127.0.0.1:8787/health
python -m codex_agent.cli.run_alfred --help
```

An adapter health response alone does not verify upstream model credentials.
GPT-family requests must remain on Responses end to end. Run evaluations in
tmux, using a unique output directory. The source-provenance gate rejects dirty
tracked files and untracked executable source/configuration. Commit authorization
is required before the first evaluation; leaving new files untracked does not
satisfy the clean-code requirement.

## Run the frozen validation fold

Validate paths and provider readiness with this command; `--preflight-only`
does not start the simulator, predictor or a model turn:

```bash
NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}" \
no_proxy="127.0.0.1,localhost,::1${no_proxy:+,$no_proxy}" \
python -m codex_agent.cli.run_alfred \
  --data-root /home/ysh/Datasets/FILM/alfred/data/json_2.1.0 \
  --splits /home/ysh/Datasets/FILM/alfred_data_small/splits/oct21.json \
  --fold docs/benchmark/alfred/folds/pilot50.json \
  --output-dir outputs/alfred/pilot50_20261105 \
  --simulator-python /home/ysh/codebase/FILM/.venv/bin/python \
  --alfred-root /home/ysh/Datasets/FILM/alfred \
  --film-root /home/ysh/codebase/FILM \
  --weights-root /home/ysh/Datasets/FILM/Pretrained_Models_FILM \
  --x-display 5 --device cuda:0 \
  --model gpt-5.4-2026-03-05 --reasoning-effort medium \
  --max-actions 1000 --max-failures 10 --max-decisions 60 \
  --preflight-only
```

The command-local proxy exclusions keep loopback health/model traffic direct
while preserving existing exclusions. They do not change other running
processes or the adapter's upstream route. The initial launch's missing loopback
exclusion and interrupted attempts are retained in the
[network diagnostic](../benchmark/alfred/v5_loopback_proxy_20261105.md).

After the reviewed source is committed, remove `--preflight-only` to evaluate
in tmux. For the first diagnostic add `--limit 1` and use a different output
directory, such as `outputs/alfred/smoke1_20261105`; this runs a prefix of the
same frozen fold. The complete pilot uses all 50 entries. Defaults are 1,000
attempted actions, 10 failed simulator actions, and 120 model decisions per
episode. The evaluated pilot explicitly used 60 decisions, as shown above.
No output directory or archived run ID may be reused.

The active display created for this setup uses `DISPLAY=:5` and
`XAUTHORITY=/home/ysh/codebase/agentic-3d-task/tmp/alfred_setup/Xauthority`.
That authentication file belongs to this display instance; on another machine
configure an accessible display before launching. Raw artifacts go under the
chosen output directory, including `run.json`, `summary.json`, per-episode
`actions.jsonl`, `observations.jsonl`, decision requests/responses, memory
geometry and BEV, and the separate official terminal snapshot. The runner
automatically archives completed and failed attempts into
`docs/benchmark/alfred/runs.sqlite`.

Independent processes can partition the same frozen fold with `--num-shards 4`
and `--shard-index 0`, `1`, `2`, or `3`, each using a distinct run ID and output
directory. Selection is exactly `manifest[index::num_shards]`, producing
13/13/12/12 entries without duplication. An optional `--limit` is applied after
sharding. Aggregate only completed shards with identical executed-code, fold,
model, perception and action budgets; verify all 50 episode IDs occur once.
Each process owns its own predictor and simulator lifecycle.

After the v6 transport outage, only original global indices
`32, 36, 40, 43, 44, 45, 47, 48, 49` lack scores. Using `--num-shards 50` and
`--shard-index <index>` selects exactly one of these from the unchanged fold;
new run IDs retain the failed attempts separately. Do not rerun already scored
tasks or substitute another model/protocol to complete this cohort. Recovery
can follow a documentation-only commit if the executed-source digest, branch,
clean state, and all actor settings remain identical; record each actual HEAD.

The shared adapter currently relies on an existing local SOCKS forward at
`127.0.0.1:10800` via its network shim. A healthy `/health` endpoint does not
verify that forward. At the outage, SOCKS negotiation stopped and then the
port refused connections. Restore the established forwarding path and verify
one native Responses turn before launching the prepared recovery jobs;
there is no need to restart unrelated benchmark services.

## Related implementation references

The integration used focused references, not an exhaustive survey:

- [ALFRED](https://github.com/askforalfred/alfred): scene restoration, mask-based actions, terminal goals and metrics.
- [FILM](https://github.com/soyeonm/FILM): domain-trained perception, semantic mapping and modular navigation.
- [HLSM](https://github.com/valtsblukis/hlsm): persistent spatial state for long-horizon instruction execution.
- [LLM-Planner](https://github.com/OSU-NLP-Group/LLM-Planner): language planning updated from observations and completed subgoals.
- [CAPEAM](https://github.com/snumprlab/capeam): memory of changed object positions and states.
- [Prompter](https://github.com/hitachi-rd-cv/prompter-alfred): object-location priors and local geometric adjustment.
- [OPEx](https://aclanthology.org/2024.acl-long.37/): exploration and execution using observed semantic maps and approach goals.

ReALFRED and TEACH are related tasks with different data/protocols; their results
are not ALFRED validation results.
