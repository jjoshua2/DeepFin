# Project guidance

Shared guidance for every agent. `AGENTS.md` points here; keep one copy.
This guide holds durable project constraints. Task-specific procedures live in
Skills and the linked documentation. Explicit user direction sets the task scope;
historical plans and experiment entries are evidence, not new assignments.
When documentation disagrees with the running system, check the relevant source,
config and artifacts, and describe the discrepancy.

## Purpose and navigation

This project trains chess networks primarily against Stockfish, including research
into its blind spots. The distributed loop is selfplay → shard upload → disk-backed
replay → training → checkpoint → model publication to workers.

- [Development](docs/development.md): setup, commands, validation and code map.
- [Experiment navigation](docs/experiments/README.md): find prior decisions and artifacts.
- [Evaluation](docs/eval_protocol.md): decide what a measurement can establish.
- [Model heads](docs/model_heads.md): head/target/loss semantics; read before changing them.
- [Operations](docs/operations.md): training, pause/recovery, config reload and deployment.
- [Loop audit](docs/rl_loop_audit.md): measurement traps and stage invariants; read the
  relevant method and stage sections when diagnosing the pipeline.

`configs/pbt2_small.yaml` is the production configuration template;
`configs/default.yaml` is a reference model. The live checkout can differ from
`main`, and a checkpoint carries its own architecture. Read the relevant config or
checkpoint rather than copying sizes, paths or tuning values from prose.
Research configs and dated plans do not establish what is currently enabled.

## Preserve running work

This machine often has long-running training, generation and monitoring jobs.
Before an action that affects them, identify the owning checkout, process, config,
artifacts and resource use. Continue from existing records and banked data before
starting replacement work. Preserve unrelated dirty files and detached jobs.

Use a separate worktree for branch work when a checkout serves a live run. Never
switch or reset that checkout underneath it: the live YAML is re-read during training.
A merged PR does not update Python already loaded by a process or its native extension
image. Plan adoption and any restart separately from merging.

A live YAML edit is a production change. Trace schema → validation → consumer →
reload behavior for the affected key; acceptance does not prove it takes effect.
Validate a copy first and verify the realized value after adoption. Some keys are
restart-only, and removing an override may leave the old value in memory. The live
run Skill and operations document cover the mechanics.

Respect intentional-stop and pause markers. Before a change whose effects survive a
config revert, preserve enough state to recover: weights, optimizer, PID and replay,
plus their provenance. Ray can prune checkpoints, so copy long-lived evaluation
baselines outside its managed tune directory. Keep run output and large artifacts
out of commits.

Budget CPU, GPU memory and disk alongside existing jobs. No simulation count or GPU
allocator fraction guarantees an arena is safe next to training. Pytest defaults to
two torch threads for this reason; keep that default on a shared machine.

## Experiments and evidence

The [ledger](docs/experiment_ledger.md) records hypotheses, readouts and revert
points. Search relevant entries and the “Protocol gotchas” before an experiment;
there is no need to load the whole ledger for routine code work.

Before committing training compute or changing the live distribution, record the
hypothesis, baseline/control, deciding metric, success/kill rule, compute budget,
readout horizon and recovery plan. Choose these for the question using the evaluation
protocol; a screen does not automatically require a training run or a fixed-size arena.
Existing authorization covers work within that scope and budget. Resolve ordinary
implementation choices without a new approval gate.

Keep one data-affecting intervention per readout window where practical; record
confounds when overlap is necessary. Bank raw observations and their source, shard,
game and position identities alongside engine/build/search settings. Reuse those
observations when correcting an estimator. Cluster correlated measurements at their
sampling unit; game IDs are not necessarily unique across shards or corpora.

Verify the producing code, effective settings and artifact lineage before a verdict.
Record the readout against the precommitted rule, including uncertainty or an unread
status when evidence is insufficient. Negative results are conditional on the model,
data and horizon tested. Calibration, teacher agreement and throughput do not by
themselves establish playing strength.

## Semantics worth knowing

- The SF component of the WDL blend has been load-bearing in prior runs. Consult the
  ledger before removing it; a sharper teacher-fit score is not sufficient evidence.
- `policy_sf` predicts the opponent's reply at P1 after the network's move. It is not
  a teacher distribution over the network's current moves.
- MCTS uses the `wdl` value head; `sf_eval` and `categorical` are auxiliary.
- Higher PID `wdl_regret` permits worse Stockfish moves and makes the opponent weaker.
  Best-move-based labels are distinct from the handicapped move actually played.
- Search action IDs and network policy indices are different spaces. Use shared
  mappings in `moves/torch_maps.py`; preserve encoding and history metadata across
  checkpoints, replay and evaluation.
- Count tied parameters once. A naive `state_dict` element sum counts shared Smolgen
  weights repeatedly; architecture regression tests measure the actual model.
- Production uses `train_views_per_ingested_position`; the old
  `train_views_per_position` name is rejected and used a different denominator.

## Implementation and review

Follow nearby code and use tests that demonstrate the affected behavior, especially
configuration propagation through the real worker path. This project's recurring
failure is a value that is accepted but silently ignored. Ask what observable result
proves the intended effect. [Development](docs/development.md) defines validation by
change scope; avoid repeated full suites after checks pass without a new reason.

Delegate bounded independent work when useful, with clear ownership and enough
context to judge it. Inherit the selected model unless the task warrants another;
there is no fixed worker count or requirement to predesign every implementation.

Open PRs ready for review unless requested otherwise. Use a separate reviewer for
consequential changes; the author is not an independent review. If independent review
is unavailable, label the self-review and its limits. Address findings or explain why
they do not warrant a change. Review evidence can be in comments, review bodies and
inline threads; `reviewDecision` alone cannot establish review status. Paid or external
review services are optional. For requested Grok reviews, use an available Grok
review Skill and its disposable-snapshot wrapper, preserve the raw result, and verify
its findings. Report checks and material limitations accurately.

## Optional workflows

Repository Skills live in `.agents/skills/`; `.claude/skills` links to the same files.
Load only the workflow relevant to the task:

- `experiment-readout`: banked-data analysis, preregistration and readouts.
- `live-run-change`: changing or recovering an existing long-running job.
- `independent-review`: reviewing a diff or reconciling PR review findings.

These Skills describe reusable methods. Repository-specific constraints stay here;
commands and operating details stay in the linked documentation.
