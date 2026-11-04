# ALFRED v4 — motion-schema compatibility diagnosis, 2026-11-05

Increasing reasoning effort to medium did not resolve the zero-action behavior
from [v3](v3_smoke1_20261105.md). The same task ended after two unsuccessful
navigation requests and a Stop labeled “Schema placeholder.” A subsequent
synthetic test explicitly requested `RotateLeft` and still received Stop.
Controlled schema variants isolated an observed compatibility defect involving
same-named `action` fields with disjoint enums in different command branches.
Giving those fields distinct names allowed the requested motion command.

These are protocol diagnostics. The retained task scores are official scored
failures, but zero-action runs with unresolved command compatibility do not
measure the proposed memory method's navigation or manipulation quality.

## Same-code medium-effort reproduction

| Configuration / result | Value |
|---|---|
| Run | `REPRO_20261105_ALFRED_V2_smoke1_medium` |
| Branch / clean HEAD | `feat/alfred-benchmark` / `f3f9da0f0dae8ecc7ff35f9910c9ef7aaf3ead2e` |
| Executed-source SHA-256 | `b78c8427a3329ecd1d76b28327de2d73b5c79fefefc4ff12150baefefe3a2069` |
| Experimental change from v3 | Reasoning effort `low` → `medium`; executable source unchanged from `def5a1f` |
| Model / input | `gpt-5.4-2026-03-05`; high-level language; FILM predicted depth and masks |
| Fold | Frozen pilot50 prefix 1; SHA-256 `1d22afc51da128395a6d65a250d9dc603c8f50f0252942ca28b335568f646d4e` |
| Task | `valid_seen/look_at_obj_in_light-Pillow-None-DeskLamp-302/trial_T20190909_150916_682139#0` |
| Action / failure / decision budgets | 1000 / 10 / 120 |
| Decisions / physical actions / action failures | 3 / 0 / 0 |
| Official SR / GC | 0 / 0 of 2 conditions |
| PLWSR / PLWGC | 0 / 0; expert length 35 actions |
| Input / output tokens | 13,545 / 1,647 |
| Policy / perception elapsed | 43.23 s / 2.86 s |

The two navigation requests targeted cells `[-2,-3]` and `[-1,-1]`; neither
had a route through observed free cells, so neither issued a simulator action.
The third decision returned Stop with rationale `Schema placeholder.` and
memory note `Done? No—this is placeholder due to schema check.` This was the
first model response, with no finalization retry or code-generated fallback.
Official terminal scoring then returned failure and GC 0/2. The run was
ingested into `docs/benchmark/alfred/runs.sqlite` as a completed scored failure.

The [exact command](assets/v4_schema_20261105/medium/command.txt) and
[run metadata](assets/v4_schema_20261105/medium/run.json) retain all arguments,
checkpoint hashes, external source revisions, and geometry settings. Preserved
evidence includes the [summary](assets/v4_schema_20261105/medium/summary.json),
[execution accounting](assets/v4_schema_20261105/medium/execution.json), and
decisions [1](assets/v4_schema_20261105/medium/decision_0001_response.json),
[2](assets/v4_schema_20261105/medium/decision_0002_response.json), and
[3](assets/v4_schema_20261105/medium/decision_0003_response.json), with their
corresponding request files. Full observations and simulator terminal state
remain under `outputs/alfred/REPRO_20261105_ALFRED_V2_smoke1_medium/`.

## Unscored controlled schema checks

All probes used `gpt-5.4-2026-03-05`, low reasoning, synthetic images, and the
isolated policy runtime. They did not control Unity or score an ALFRED episode.
The requested output was a single motion command with `RotateLeft`, repeat 1.
Each row below represents one recorded response, not a reliability estimate.

| Schema / request path | Returned command | Matched requested motion |
|---|---|---|
| [Original full schema](assets/v4_schema_20261105/protocol/original_report.json) | Stop | No |
| [Original full schema, audited local relay](assets/v4_schema_20261105/protocol/original_wire_report.json) | Stop | No |
| [Full schema with references inlined](assets/v4_schema_20261105/protocol/inline_refs.json) | Stop | No |
| [Only the MotionCommand branch](assets/v4_schema_20261105/protocol/sole_motion.json) | `motion`, `action=RotateLeft`, `repeat=1` | Yes |
| [Full schema with distinct action field names](assets/v4_schema_20261105/protocol/distinct_action_names.json) | `motion`, `motion_action=RotateLeft`, `repeat=1` | Yes |

The original full schema permits motion and interaction branches with different
enums under the same property name, `action`. Inlining references did not
restore the requested motion. Keeping only the motion branch succeeded;
retaining all branches while naming the fields `motion_action` and
`interaction_action` also succeeded. This supports a schema compatibility
explanation rather than an unavailable simulator action. The negative outputs
were valid Stop objects under the supplied schema, but contradicted the
explicit requested command.

The [local wire audit](assets/v4_schema_20261105/protocol/actor_tool_audit.jsonl)
contained zero tools and the complete `text.format.schema`. An independent
[comparison](assets/v4_schema_20261105/protocol/wire_schema_comparison.json)
against the Pydantic schema loaded from the recorded `f3f9da0` source found
exact structural equality; both canonical schema hashes are
`53b0a9383168505d6544388d27314e0eccdbc30c0f0efbd9269eb166c0dc08d6`.
The local relay retained request bytes. Inspection of the adapter's native
Responses branch confirmed forwarding of `request.envelope.raw` as the HTTP
body, without Chat translation or schema rewriting. Retained SDK traces for
all five probes contained only message/reasoning items and zero tool calls.

The observed defect is specific to the configured structured-output route and
these controlled requests. The remote implementation was not inspected; these
results do not establish which remote component or internal algorithm caused
the behavior.

Expanded [seven-command coverage](assets/v4_schema_20261105/protocol/all_command_probe/result.txt)
showed that distinct action fields alone were insufficient: motion, navigation
to a cell, retrieval by object/cell, and Stop matched their requested commands;
navigation to an object and object interaction did not. Thus that proposed
repair was not used for a new task evaluation. The final local transport uses
a single explicit operation enum and required scalar arguments, with no unions
or references in the wire schema. Internal domain commands remain typed and
select exactly one target. This avoids relying on the observed problematic
schema features without changing the model, protocol or action set. Every
command branch and a physical rollout must pass before pilot expansion.

The [flat-schema coverage probe](assets/v4_schema_20261105/protocol/flat_command_probe/result.txt)
matched **7/7** requested command families on the actual provider: motion,
interaction, navigation by object/cell, retrieval by object/cell, and Stop.
The complete scalar schema and all responses are archived with that log.
Each request had one attempt; this establishes compatibility for these cases,
not a statistical reliability guarantee or a physical task-success result.

The [diagnostic manifest](assets/v4_schema_20261105/protocol/diagnostic_manifest.json)
records exact historical script commands and trace checks. Original scripts,
the original full schema, and each variant's schema are archived beside it.
The probes were development checks, and their reports do not record an
independent clean executed-code fingerprint; they must not be relabeled as
clean benchmark evaluations. Their scripts imported the then-current policy
schema, so later reproductions must use the archived schema or corresponding
source revision rather than assume the latest import is identical. The
[artifact checksums](assets/v4_schema_20261105/sha256.json) preserve all positive
and negative evidence in this record.
