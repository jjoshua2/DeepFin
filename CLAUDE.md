# DeepFin project guidance

Shared constraints for all agents; `AGENTS.md` points here. Keep one policy copy.
DeepFin trains chess networks primarily against Stockfish, through offline bootstrap
and distributed selfplay/training. Verify the phase relevant to the task rather than
assuming either workflow is currently running.

## Scope and completion

The user request defines the deliverable; historical plans are evidence, not new
assignments. For implementation, continue through the requested working result, fix
introduced failures and complete applicable checks, not just a plan or first patch.
Respect analysis-only and review-only scope.

Choose implementation, isolated edits, bounded checks and useful delegation (normally
inheriting the selected model) within the agreed scope/budget without repeated approval.
Pause for consequential ambiguity that evidence cannot resolve, or actions needing
new authorization for live changes, destructive effects or additional resources.
Existing authorization covers its stated scope; stop at the deliverable or agreed limit.

## Protect running work

`main` owns development; live branches pin deployments. Merging does not update running
Python/native code. Isolate branch/build work from live checkouts and environments;
never switch/reset beneath jobs or rebuild their in-use extensions. Preserve unrelated
dirty files, detached jobs and intentional-stop/pause markers.

Before affecting a job, establish its owner checkout, process, effective config,
artifacts and resource use. Live YAML edits are production changes: validate a copy,
trace the key through validation, consumer and reload behavior, then verify adoption.
Preserve recovery state before effects that outlive a config revert; keep durable
baselines outside Ray-pruned directories. Budget CPU/GPU/disk beside existing jobs;
keep the default two-thread pytest cap on shared machines. Do not commit bulk
artifacts or transient run output.

## Context by task

Read relevant sections, not a prerequisite stack of documents. Skills live in
`.agents/skills/`; `.claude/skills` links there. Load only the applicable workflow.

| Task | Reference |
| --- | --- |
| Setup, commands, code navigation, validation | [Development](docs/development.md); [toolchains](docs/toolchains.md) for corpus, offline-training and match entry points. |
| Heads, labels, search values, encoding, training-view accounting | [Model contracts](docs/model_contracts.md); [model heads](docs/model_heads.md) for target/loss wiring. |
| Pipeline diagnosis or measurement traps | Relevant method/stage sections of the [loop audit](docs/rl_loop_audit.md). |
| Experiment planning, banked-data analysis, readouts | [Experiment index](docs/experiments/README.md), [evaluation](docs/eval_protocol.md), `experiment-readout`. Search relevant historical ledger entries; it is frozen. |
| Live changes, deployment, pause, recovery | [Operations](docs/operations.md), [branch lifecycle](docs/branch_lifecycle.md), `live-run-change`; live-first fixes need same-session main PRs linked in the run record. |
| Independent review or PR findings | `independent-review`; `deepfin-grok-review` only for requested Grok review. External/paid review is not a routine gate. |

Use effective config and checkpoint architecture, not values copied from prose.
`configs/pbt2_small.yaml` is the production template; `configs/default.yaml` is a
reference. Neither proves live state. Resolve discrepancies against relevant source,
effective settings and artifact lineage, and report them.

## Evidence and delivery

Before training compute or live-distribution changes, preregister the hypothesis,
control, deciding metric, success/kill rule, budget, horizon and recovery plan. Publish
indexed `docs/experiments/YYYY-MM-DD-slug.md` registrations/readouts on `main`, with
compact evidence and external artifact identities/locations. Reuse banked observations;
keep interventions separate where practical, record confounds and source-qualified
sampling identities, and judge results against the registered rule with uncertainty.
Teacher fit, calibration and throughput are not playing-strength results; negatives
are conditional on the model, data and horizon tested.

Use change-scoped validation from the development guide. Documentation needs
link/discovery/consistency checks, not automatic training, arenas or full suites.
Behavior changes need an observable effect and meaningful failure case; accepted
config is not proof it reaches the executing worker. Expand checks for affected
boundaries, not ritual repetition after a passing final candidate.

Deliver PRs ready for review unless requested otherwise. Use independent review for
consequential changes; when unavailable, label self-review and its limits. Address
findings or explain their disposition. Report actual checks and remaining blockers,
distinguishing implementation, validation, deployment and research results.
