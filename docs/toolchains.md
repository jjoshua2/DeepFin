# Toolchain entry points

Install the full development environment in [development](development.md) for these
tools. Corpus repair and production `.zst` files require `zstandard`, included in
`.[dev]`. Fresh worktrees also need their own native extension build; importing the
corpus tools uses `CBoard` even for CLI help. Normal package installation builds portable
extensions; production host builds follow the separate operations procedure.

Run scripts from the repository root with an installed package or `PYTHONPATH=.`.
Use each command's `--help` for its current arguments; examples below inspect interfaces
and do not launch experiments.

## Corpus to offline training

| Stage | Entry point | Contract to preserve |
| --- | --- | --- |
| Generate rooted Stockfish corpus | `scripts/gen_sf_rooted_corpus.py` | Frozen worker/search/resume settings, engine identity, row history and source-qualified game identity |
| Audit possible teacher labels | `scripts/audit_label_candidates.py` | Fixed banked observations and comparable teacher/search settings |
| Repair an existing corpus's history | `scripts/repair_corpus_history.py` | Distinct output, inventory coverage, unchanged labels unless explicitly relabeling, truthful completion stamp |
| Derive replay targets | [`scripts/derive_corpus_targets.py`](corpus_derivation.md) | Input corpus identity, target recipe, history/encoding stamps and completed-shard inventory |
| Train a controlled LC0-shaped comparison | `scripts/lc0_control_train.py` | Declared architecture, data lineage, objective, view/step budget and checkpoint identity |
| Prepare selected common-input batches | [scripts/common_input_batch.py](common_input_batch.md) | Frozen closed-shard manifest; one/two bounded source lanes; independent survivor and BT4/rank qualification |
| Registered BT4 exact-epoch screen | [scripts/bt4_one_epoch_screen.py](experiments/2026-09-07-bt4-hybrid-endpoints.md) | One explicitly selected profile; qualified corpus, completed training and fixed comparisons |
| Offline target/loss comparison | `scripts/retarget_retrain.py` | Same-seed controls and a frozen readout appropriate to the hypothesis |
| Replay epoch reference runner | `scripts/offline_replay_epoch.py` | Distinguish its sampling contract from the game-aware exact-epoch mode |

```bash
PYTHONPATH=. python scripts/gen_sf_rooted_corpus.py --help
PYTHONPATH=. python scripts/derive_corpus_targets.py --help
PYTHONPATH=. python scripts/repair_corpus_history.py --help
PYTHONPATH=. python scripts/lc0_control_train.py --help
```

The generator also supports the validated G10 staircase policy, with its decision
rule stamped through generation, derivation and repair. A policy flag must not change
an existing corpus identity during resume.

Repair is conditional on a known corpus defect; it is not an obligatory stage for a
new healthy corpus. Never mutate a populated corpus to reuse its identity for different
worker or teacher settings. Keep a companion corpus separate when scaling changes the
run's interpretation.

Game-aware exact epochs are implemented in `chess_anti_engine/replay/game_epoch.py`
and consumed by the replay/training paths. They freeze a corpus census, choose rows
per source-qualified game, preflight memory and target masks, and account for complete
epoch consumption without wrapping or silently truncating. Follow
[target rebuildability](target_rebuildability.md) for retained observations and identity.

## Uninterrupted offline game epochs

`scripts/lc0_control_train.py --sampling-mode game_epoch --steps 0 --epochs 2`
trains two complete passes through one frozen corpus using one freshly initialized
trainer. `--epochs` defaults to 1, preserving the existing one-epoch path; values
above one require `--steps 0`. This interface does not resume old checkpoints.
Each epoch uses sampling seed `--seed + epoch_index - 1`. The optimizer, scheduler,
Torch RNG and augmentation RNG continue across boundaries. The usual 88-step
windows restart at each epoch, including a separate short final window per pass.
The corpus is revalidated and replanned at each boundary; budget that CPU and disk
work as well as training. Corpus fingerprint or planned batch-count changes abort.

Successful runs publish `checkpoint_epoch1.pt` and final `checkpoint.pt`, with
both identities in `summary.json`. The existing `checkpoint_mid.pt` remains a
run-budget snapshot snapped to an actual interior window endpoint, including
ragged epoch boundaries (equal-distance ties choose the earlier endpoint). For
two epochs at the default half-budget fraction it coincides with epoch one; it
need not do so for other epoch counts or fractions. Until all epochs and realized
loss guards pass, the first-epoch snapshot is named `checkpoint_epoch1.pending.pt`.
A failed run can retain that diagnostic artifact, but emits no completed summary
or published epoch-one checkpoint, and output reuse is refused. Multi-epoch
`sampling.mode` is `game_epochs`: its `epochs` list records each sampling seed,
planned/realized schedule hashes, rows, batches, window count and cumulative step
boundaries. Window records also identify their epoch. Readers that require one
`game_epoch` must explicitly support this different receipt before using it.

These runs retain the historical `valid_control: false` limitations of exact
sampling. Current main additionally uses corpus fingerprints and corpus-wide
objective-mask loss normalization absent from the frozen wise-cloud BT4 runtime.
A horizon comparison must train both target families freshly under the same
qualified implementation and compare each trajectory's own epoch-one/epoch-two
checkpoints. Comparing a new two-epoch run with an old frozen-runtime one-epoch
checkpoint would confound training duration with objective/backend changes.
This interface does not select an experiment or alter the active H20 protocol.


### Qualified two-epoch training coordinator

[`scripts/bt4_two_epoch_train.py`](../scripts/bt4_two_epoch_train.py) is a separate
training-only coordinator for H20, B100, G50 or genuine SoftSF10 on the qualified
original 18,910,484-row corpus. It fixes two uninterrupted epochs, sampler seeds
0/1, batch size 512 and 88-step windows. It consumes preparation evidence; it does
not create prospective plans, select a family or launch a match. Preview with
`--manifest FILE`; execution additionally requires `--execute`.

Its manifest has `schema: 1`, `scope: original_corpus_two_epoch_training_only`,
`profile`, fresh absolute `state` and `run` paths, explicit `training_seconds`
(at most 32,400 including kill grace), `plan_workers`, `load_workers` and
`max_working_set_bytes`. It requires `{path, sha256}` pins for
`runtime_qualification`, `preparation` and `preregistration`, plus
`launcher_sha256`, `stage_helper_sha256` and `recipe_helper_sha256` for the
coordinator, `bt4_direct_screen.py` supervision and `bt4_one_epoch_screen.py`
recipe admission. Extra manifest fields are refused.

The runtime qualification declares `status: PASS_TWO_EPOCH_TRAINING_RUNTIME`,
`root`, the pinned training `head` (`0ff96f006`), `runtime`, `environment_pins`,
`cpu_qualification`, `cuda_qualification` and `resources`. `runtime` retains the
usual Python/executable/Torch/CUDA/NumPy values, with `native_extensions` and
`native_extension_sha256` covering features, LC0 encoding and NNUE. Environment
pins must include the actual interpreter binary. `resources` binds the manifest's
four explicit worker/memory/time allocations; the tiny execution probe does not
qualify full-corpus memory. CPU evidence must be the actual
`PASS_CPU_TRAINING_IMPORTS` receipt; CUDA evidence must be the completed
`PASS_COMPILED_CUDA_TWO_EPOCH_PROBE` receipt, with matching runtime identity,
observed graph capture and the complete 1,024-row, two-pass model probe. Pending,
CPU-only or eager-only evidence cannot qualify a training launch.

Preparation declares `schema: 1`, `status: PASS_TWO_EPOCH_PREPARATION`, `profile`,
`corpus`, the same `runtime_qualification` and `preregistration` pins, and the
three worker/memory values. It pins `source` (the original derive summary),
`data_qualification`, `derive_summary`, `recipe_summary` and the qualifying
`producer` source. Its two `epochs` entries contain `epoch_index`,
`source_logical_order_sha256`, `corpus_logical_order_sha256`,
`source_plan_sha256` and the complete candidate `GameEpochPlan.as_dict()` as
`plan`. The producer must establish source-qualified logical row-order equality
under each seed, independent of the deliberately different policy/content hashes.
The coordinator compares those logical witnesses and later checks every candidate
plan field against the realized receipt; it does not independently reconstruct
that witness. Old one-epoch `dc687...` evidence is insufficient. Genuine SoftSF
rewrite metadata uses the existing producer admission, never a renamed BT4 mix.

One child runs the unchanged pinned trainer via its normal imported `main()` and
argparse, retaining its file identity. The small coordinator bootstrap records
actual before/after Torch, Blosc and compiler thread counts, sets each to two, and
uses private compile-cache directories. Inherited compiler-error suppression is
removed and actual Dynamo `suppress_errors` is set to false; ordinary graph breaks
remain allowed. The qualified loader uses thread pools.
The shared GPU lease and owned process-group deadline cover the entire training
child, including staging and both plans; STOP and failures preserve partial
artifacts. Checkpoint hashing and completion validation run after lease release.

Only two complete finite passes, matching planned/realized identities and memory
bounds, exact per-epoch update/sample totals and both published checkpoint hashes
produce `two_epoch_training.complete.json`. That receipt has
`two_epoch_training_complete: true` and separate `epoch1`/`last` checkpoints;
it does not implement the old single-epoch completion interface. Historical
control limitations remain in the receipt. Future matches need separate
checkpoint/reader admission and registration; this tool does not require replacing
a compatible qualified search backend.

## Playing and measurement

| Question | Entry point |
| --- | --- |
| Paired checkpoint comparison, resume and optional SPRT | `scripts/arena_standard.py` |
| Registered BT4 checkpoint screen | [scripts/bt4_direct_screen.py](experiments/2026-09-06-bt4-direct-close-global.md) |
| Fresh-seed BT4 confirmation with two newly trained roles | [`scripts/bt4_confirmation.py`](bt4_confirmation.md), `scripts/bt4_joint_readout.py --profile confirmation` |
| Match against another UCI engine | `scripts/match_vs_uci.py` |
| Fixed handicapped-Stockfish opponent | `scripts/match_vs_handicapped_sf.py` |
| Joint PGN rating estimate | `scripts/ordo_pooled_fit.py` |
| Bank or inspect foreign BT4 policy/history behavior | `scripts/bt4_policy_dump.py`, `scripts/bt4_history_sensitivity.py` |
| Bank and analyze varying-horizon continuation trajectories | `scripts/collect_varying_budget_trajectories.py`, `scripts/analyze_varying_budget_controller.py` |
| Observe fenlist and SF-refute outcomes without inferring playing strength | `scripts/monitor_sf_refute_outcomes.py` |
| Relabel/reconstruct RVG targets and shadow readout | `scripts/rvg_label_pass.py`, `scripts/nnue_shadow_label_readout.py` |

The legacy BT4 checkpoint and sharpened-tie screens retain their pinned host-local reader at
`/tmp/deepfin-bt4-prior-one/scripts/bt4_joint_readout.py`, plus its frozen runtime
and development opening book. Updating the repository reader does not migrate
that executed protocol; the linked registration records its identities. Set
`CHESS_EXPERIMENT_ROOT` to the data-owning repository when it differs from
`~/projects/chess`; the launcher checkout can remain separate.

The [adaptive H20 investigation](experiments/2026-09-07-bt4-hybrid-endpoints.md)
adds explicit H20, B100 and G50 profiles to the same launchers. A manifest selects
one profile; available profiles are not a queue. These profiles pin the adjacent
extended reader, require a successful identity-bound corpus qualification, verify
the complete seed-zero epoch and run only that profile's fixed comparisons. The
500-game higher-search probe is tied to the completed same-candidate C100 bank's
first 250 opening pairs. Old profile defaults retain their original reader.

For a separately registered training stage, `bt4_one_epoch_screen.py` also accepts
`schema: 3`, `mode: "training_only"` and one of the same H20/B100/G50 profiles.
Use the ordinary `--manifest FILE` preview and add `--execute` only to run it.
The manifest retains `state`, `run`, `training_seconds: 16200`, `runtime_manifest`,
`preregistration`, `prospective_schedule`, `launcher_sha256`, `input_pins` and
`data_qualification`. It pins the shared supervision module with
`stage_helper_sha256`; omit `arena_launcher_sha256`, `reader`, `comparisons`,
`arena_seconds` and `total_seconds`. Extra arena fields are rejected.

`SoftSF10` is an additional **schema3-only** profile for the original-source
raw effective-cp10 SF policy rewrite. It requires the actual
`sf_policy_rewrite_summary.json` and a `rewrite_summary` pin in the corpus
qualification; it never adopts a renamed BT4 mix receipt. The final rewrite must
match the reviewed producer, original raw/source identities, all 2,309 shards,
unchanged 16 non-policy columns, finite stored-policy mass and positive changed
rows. The derived summary must preserve the original source selectors, value and
history metadata, with only its documented postprocess projection added. A
`.writing` directory or failed/incomplete publication is refused. Qualification
reuses the completed producer proof and checks current metadata; this launcher
does not claim a second full payload verification or fresh history encoding.
SoftSF10 has no schema2 comparison schedule. After its qualified epoch, the
separate `matched_original_epoch` arena profile can bind it to a completed
reference such as B100 under a new match registration.

Training-only preserves the qualified old training runtime, exact 18,910,484-row
seed-zero epoch, 512-row batches, complete finite-window checks and prospective /
realized canonical schedule checks. The GPU stage keeps its 4.5-hour cap; the
separate CPU schedule verifier keeps its 1,800-second cap. C's completed schedule
summary remains a witness, but no opening book, arena reader or control-checkpoint
payload is required. The CPU runtime probe checks actual package/native identities
without resolving arena search settings.

The existing `training.complete.json` checkpoint/schedule interface is unchanged.
The top-level `complete.json` instead declares `scope: "training_only"` and
`training_only_complete: true`, with its training receipt hash and GPU charge;
it intentionally has no generic `complete: true` package flag. No arena is
dispatched or implied. A later match needs its own registration, runtime and
reader qualification. Schema 1/2 still run their original fixed comparisons.

For H20 target construction, `scripts/bt4_policy_mix.py mix --scope c20-global`
keeps `--shards` pointed at original SF. Supply the actual stored C corpus through
`--c20-parent`, pin it with `--expected-c20-summary-sha256` and
`--expected-c20-mix-sha256`, and retain original BT4/rank sidecars. The fixed recipe
uses `--alpha 0.2 --bt4-temperature 0.5 --sf-rank-cap 3 --sf-cp-window 20`.
Both audit and materialization use descriptive SF admission with the same immutable
experiment record. The mixer verifies C's recipe, exact stored policy and all other
fields before composing the new target, and records the parent lineage explicitly.
This is specific C20T05 composition support, not arbitrary nested-source admission.

The varying-horizon tools follow their [staged protocol](experiments/varying_horizon_online_controller.md),
including grouped source identities and limits on interpretation.

Use [evaluation](eval_protocol.md) to select the deciding measure and budget. A frozen
handicapped-opponent match is different from a production curriculum winrate whose
opponent changes under PID control. Resume only with matching schedule/settings and
preserve per-game records; an interrupted pair is not an observed draw. SPRT is an
optional precommitted stopping rule, not permission to keep extending any weak result.

### Bounded speculative paired play

For a separately registered rolling matched-simulation SPRT, the arena CLI accepts
`--sprt-lookahead-pairs 64`. New opening pair IDs must be below the next declared
SPRT sample size plus 64, capped by the total registered pairs. For example,
`first_pairs=128,step_pairs=64` initially admits pairs 0 through 191. Once the first
look is reached without a decision, the window advances to pairs 0 through 255.
Every declared statistical look is still consumed in canonical order; the first
crossing still decides. Zero allowance is supported. Omitting the flag preserves
existing admission and fixed-N behavior; chunked and fixed-N runs reject the flag.

Both colors are admitted together. This requires at least two pool slots and may
leave one slot unused in an odd-sized pool. A delayed early game can shrink the
active pool while later admission waits. The allowance is recorded in the result
and game-log settings fingerprint; resuming with a changed or removed allowance
is refused. Complete previously banked suffix pairs remain usable on resume.

This bounds speculative admission, not elapsed time. Lower occupancy can hurt
throughput, while fewer speculative games can reduce work. Batching and the shared
RNG's consumption change, so identical gameplay trajectories or a speedup are not
promised. The two-depth recipe launcher accepts optional manifest `sprt_lookahead_pairs`
(an integer from 0 through 500; the proposed allowance is 64). Explicit nulls,
booleans and fractional values are rejected. It applies only to the rolling
100-simulation SPRT; the fixed 400-simulation probe has no flag. The prepared
settings, command, game-bank fingerprint and terminal result must all agree. The
reader permits precisely that declared low/high execution difference alongside
the existing game-count and simulation differences. It retains canonical looks,
first-crossing decisions and the fixed 128-pair cross-budget comparison.

Omission keeps the historical runtime, settings and command unchanged. An enabled
option selects a separately frozen two-file overlay on the qualified Python
3.10/Torch 2.11 arena stack; model, search and native inputs remain unchanged.
Runtime qualification, preparation and registration must belong to this fresh
comparison. Existing B100/H20 or G50 banks are not retroactively reclassified.
The change does not broaden the original-corpus one-epoch training admission.

For a future bounded telemetry observation, `chess_anti_engine.mcts.gumbel_c` already
emits a DEBUG record with board count, evaluator call/position counts, and coarse
stage times. A small dedicated logging handler can aggregate those records in memory
and emit one summary per minute and at shutdown, with propagation disabled for that
logger. Enable only that logger, and aggregate counts to compute weighted occupancy;
do not print every call or enable global DEBUG. This is a proposed observation hook,
not enabled here. Its host-side evaluator time includes transport/wait and is not
isolated CUDA kernel time. Startup, steady play and admission-window drain should
be reported separately before attributing a whole-match bottleneck.

## Agent tooling

`scripts/grok_review.sh` provides independently authored reviews from a disposable
snapshot; the `deepfin-grok-review` Skill documents invocation and interpretation.
`scripts/grok_fix.sh` prepares a separate worktree for explicitly requested Grok
implementation and leaves its diff for inspection. Neither workflow deploys or merges
its result. Useful changes from either route still target main under the
[branch lifecycle](branch_lifecycle.md).

Dated `scratchpad/` drivers preserve experiment history and may contain old absolute
paths or run selections. Inspect and adapt those before use; the supported entry points
above expose their current interfaces through `--help`.
