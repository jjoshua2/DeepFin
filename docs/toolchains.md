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

For lower resident memory, keep the original logical `--workers` and add
`--worker-concurrency 2` (or another positive process limit). All original worker
IDs, game partitions, seeds and dedup capacities remain unchanged; queued workers
start as slots free. The default runs all logical workers concurrently. The
effective limit is recorded separately in `execution_invocations.jsonl` and
`summary.json` under `execution`, so it can change on resume without changing the
scientific manifest. It bounds active workers, not RAM bytes; a worker can still
hold its full dedup cache. Uneven progress across workers is expected until all
finish. Preserve unlisted tails before recovery: existing resume cleanup removes
them. Merging this option does not update a running generator.

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
training-only coordinator for H20, B100, B100T1, G50 or genuine SoftSF10 on the qualified
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

`B100T1` is pure global BT4 policy at target temperature 1.0, with the original
SF value and other non-policy arrays retained. Its corpus is the original source
directory with suffix `_bt4_global_B100T1`; `B100` continues to mean the separate
temperature-0.5 corpus. The existing mixer produces T1 using `--scope global
--alpha 1 --bt4-temperature 1` and the retained unsharpened BT4 sidecars. Do not
reconstruct it from stored sharpened targets or substitute the SF-mixed G20T1
or top-tie corpora. The new profile is only admitted by this two-epoch coordinator.

T1 preparation needs its own pinned `PASS_REGISTERED_CORPUS_QUALIFICATION` with
`profile: B100T1`, genuine derive/mix summaries, and both full prospective plans
with source-qualified logical-order witnesses. It uses the same runtime,
resource, completion and checkpoint requirements below. Adding the profile does
not qualify a corpus or change the trainer, schedule, search settings or value
targets; it does not admit historical B100 checkpoints as an epoch-one control.

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

For one separately registered within-checkpoint prior comparison, use
`python scripts/bt4_joint_readout.py --calibration-contract FILE.json`.
This reads completed banks; it does not launch a match or choose a temperature.
The original `--profile calibration` remains the fixed S0/E0, 1.0-versus-1.5,
100-simulation, 500-pair chunked protocol.

The explicit contract has `schema: 1` and these fields:

- `checkpoint`, `bank`, `opening_panel`: objects with absolute `path` and `sha256`.
  The checkpoint must identify both arena sides. Its bytes are hashed in a stream;
  no model is loaded. The panel uses the existing recipe format: one
  `{root_fen, moves, fen}` record per pair, with 16 legal history moves and unique,
  nonterminal endpoints in canonical pair order.
- `candidate_prior_temperature`, `reference_prior_temperature`: finite positive
  numbers; `sims`: a positive integer; `expected_pairs`: an integer of at least two;
  `seed`: an integer; `loop`: `rolling` or `chunked`.
- `expected_settings`: the complete dictionary produced by
  `arena_game_log_settings`, including each `SideSearch.as_record()` result.
  Both sides must use the same checkpoint and training search, differing only in
  prior temperature and its recorded source string. The candidate-only volatility
  override must be absent (`volatility_candidate: null`). Other inherited book/search
  protocol requirements remain enforced.
- `expected_execution`: the compile/evaluator tags, for example `["on", "4096"]`.
  Every game must match; compile remains on.

The reader requires exactly all registered color-swapped pairs, with no missing,
replayed, duplicate, orphan or torn records and no SPRT. An exit-zero arena that
stopped early at its time limit cannot produce `bank_complete: true`. Bank hashes,
full settings, current checkpoint bytes, legal panel history and logged endpoints
are checked. `launch_qualification_verified` remains false: effective launch argv,
runtime, original checkpoint/book bytes and actual history consumed still require
the existing external launch evidence. The output is one nominal paired contrast,
not a grid selection, training improvement or optimal-prior claim.

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

`bt4_value_rewrite.py --alpha 0.5` produces the B100V50 value challenger:
50% normalized stored SF WDL plus 50% normalized native BT4 WDL, computed in
float64 and stored in float16. The reusable `--alpha` accepts finite weights in
[0, 1] and defaults to 0.1. The default retains the exact historical V10
algorithm, value-scheme and teacher/head source stamps; other doses carry an
explicit alpha in their scheme and source identity. Shard and recipe receipts
record both weights. Only `search_wdl` changes; B100 policy and the other 16
compressed arrays remain byte-identical, under the existing source/sidecar join
and final stability checks. This preserves inherited source history and its
historical control-validity limitations; it is not a new history reconstruction.

The one-epoch launcher admits `B100V10` and `B100V50` only under schema3
`training_only`, with the genuine `bt4_value_rewrite_summary.json` and a
`rewrite_summary` qualification pin. Each profile fixes its dose, corpus path,
algorithm, teacher/head, ancestry and producing-code identities; relabeling a
V10 receipt as V50 fails admission. Historical V10 producer receipts remain
accepted, while V50 requires the alpha-capable producer. Other alpha values are
producer capabilities, not additional registered training profiles. Both reuse
the original training command and separate matched-recipe arena workflow. No
corpus rewrite, training run or scientific result is implied by this tooling.

`B100Tactical100` is a separate schema3-only policy profile. It requires the genuine
`bt4_sf_tactical_policy_summary.json`, its `rewrite_summary` qualification pin,
and the reviewed gap100/decay100/floor0.1 categorical-mate recipe on stored B100.
The producer proof must bind original SF/raw and qualified B100 summaries, all
2,309 output shards, the consumed/output policy hashes, unchanged SF value and all
16 nonpolicy columns. Mate-group counts and stored-support diagnostics remain
visible, including winning-mate groups with zero original B100 mass; attenuation
cannot invent that missing support. Its output path is the original B100 path
with suffix `_tactical100`. Renaming another policy/value recipe is rejected.
This is an admission path for a completed, independently qualified corpus, not a
claim that the full rewrite or training has run. History/control limitations
remain inherited. The existing `matched_original_epoch` arena route can consume
its eventual completed training receipt without changing training or search rules.

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

## Stored features to Ceres byte inputs

`chess_anti_engine.encoding.ceres_tpg.stored_x_to_ceres_tpg_bytes` converts original
float16 `(175,8,8)` features, singly or in a batch, under explicit
`input_history_encoding="lc0_root_legacy_meta"` and `history_rep_fix=True`. It
preserves the eight recorded root-relative piece/repetition slots and fills
missing trailing slots from the oldest real position. It retains the original EP
file, recovers the integral rule-50 counter and emits default uint8 TPG records
with symmetric Q=.03 and zero ply features. Unsupported encodings, nonfinite or
malformed writer-domain inputs fail. The existing Board API is unchanged.

This is a feature conversion, not raw-history or float32 input-key reconstruction.
Callers must bind the qualified source lineage; structurally valid history slots
do not prove a legal move sequence. Exact comparison with the local Board encoder
and saved source-qualified inputs does not establish a native Ceres oracle,
model quality, runtime qualification or a complete policy/value collector. Keep
original stored features: `x_to_lc0_planes` removes the EP metadata needed here.


### Compact Ceres collection options

`scripts/ceres_derived_sidecar.py` keeps fixed physical batches of 32. By default,
it retains raw primary `value` logits and refuses nondivisible shards, preserving
the original primary-only profile. `--pad-final-batch` explicitly repeats the last
real TPG feed row to complete a batch, validates the full returned batch, and trims
all outputs before gathering or storage. Source hashes, row identities and feed
digests cover only real rows. Shard and completion receipts distinguish real rows,
padding rows, physical input rows and calls; completion also records newly executed
counts separately from verified cache coverage.

`--retain-value2` requests `policy`, `value` and `value2` in the same session call.
It retains secondary raw float16 `(N,3)` W/D/L logits as `value2_logits`, alongside
unchanged primary `value_logits`. There is no softmax, temperature or native blend.
Missing, malformed or nonfinite outputs fail, including outputs for padded rows.
Secondary storage adds six raw bytes per real row, before container overhead.

Either option selects the extended profile; requested heads and remainder policy
are bound in the namespace. Primary-only banks remain verifiable under their
original profile, including older receipts without explicit counts. They cannot
satisfy secondary coverage or be silently backfilled. Producer source pins remain
exact: a new code revision does not automatically adopt an older output namespace,
even with default flags. Use a distinct qualified output lineage for new options.

These are collection capabilities, not operational or numerical qualification of
a full corpus. The earlier fixed32 approximation and native-order/history limits
remain. A future launch needs its own exact source selection, call budget and
runtime evidence; an old fixed-count pilot wrapper is not a full-collection plan.

### Offline BT4–Ceres policy mixtures

`scripts/ceres_target_mix.py` consumes completed teacher sidecars; it performs no
inference. A SHA-pinned JSON manifest names the original derived corpus, its summary,
the two teachers, and one full Ceres/BT4 sidecar entry per source shard. It validates
coverage, legal indices, row identities and available input provenance. Selected-row
banks cannot stand in for full-shard coverage.

Each teacher is independently normalized and temperature-scaled before probability
mixing. Defaults are 50% BT4 / 50% Ceres, both at temperature 0.5. Feed raw BT4
probabilities, not already sharpened B100 targets. Only `policy_target` changes;
16 other compressed arrays remain byte-identical. The output uses ordinary copied
shards and becomes complete through an atomic rename. Run with a disk reserve and
external hard timeout; internal STOP/deadline checks are cooperative.

Legacy BT4 collection uses an explicit pinned lineage mode: current-position and
history-regime checks do not newly prove historical input frames. The summary records
that inherited limitation. Full-input hashes are checked whenever supplied; stored-x
mode requires them. Ceres alignment additionally checks actual input arrays and TPG
feeds. These checks establish data construction, not playing strength.

The script validates by default; `--execute` performs the rewrite. Its versioned
`ceres_target_mix_summary.json` is distinct from BT4-only summaries. The registered
`CeresB50` profile in `bt4_one_epoch_screen.py` admits the fixed 50/50 T0.5 recipe in
schema3 training-only mode, with an explicit producer freeze, corpus qualification
and matched schedule. This adds no new trainer or value mixture. A producer test on
the development runtime does not qualify the historical training runtime.

### Bounded Ceres collection batches

`scripts/ceres_collection_batches.py` runs an ordered JSON plan of prepared
collector commands. Validate first with `--plan PLAN --expected-plan-sha256 SHA`;
add `--execute` to run it. Each chunk needs a fresh output directory, explicit
source range and expected row/padding counts, argv array, working directory,
completion mode and timeout. The plan pins its dependencies and sets a fresh
state directory, disk reserve, whole-operation budget and inter-chunk pause.

Use `completion_mode: "ceres_invocations"` for the collector's timestamped
`invocations/<id>/completed.json`. Exactly one genuine parent completion must
match the current invocation and all expected counts; a child receipt alone does
not qualify. `fixed_path` instead requires `expected_completion` within that
chunk's output directory. Existing output is refused, not counted as new work.

The limits are 30 minutes per chunk, 30 hours overall, at least 150 GiB free and
at least 30 seconds between chunks. `state_directory/STOP` or a termination
signal stops the queue and cleans up its owned process group. Failures retain
completed-chunk receipts and do not retry. Cleanup cannot recover a kernel-stuck
process or descendants that deliberately leave the group.

The prepared collector command must acquire the shared GPU lease and implement
its own inference/resource checks. This driver does not acquire that lease or
validate the meaning of arbitrary argv. Its process tests do not qualify a new
backend, dataset, teacher recipe or full-corpus collection; register and verify
those separately before launching collection.

### Qualified selected-bank Ceres collection

`--selected-bank-qualification /bank/complete.json` together with
`--expected-selected-bank-qualification-sha256` opts into the immutable
`soft_sf_qualified_sample_v1` selection (4,096 rows in 64 saved NPZ fragments).
Use `--source /bank --max-shards 64 --pad-final-batch --retain-value2`; omit the
whole-corpus summary and G10 qualification arguments. Partial shard ranges are
rejected. This profile has its own namespace and `selection_*.zarr` outputs; its
completion counts fragments, never full source shards. It does not qualify a
training corpus or expand the sample.

Admission binds the exact completion receipt, original ordered selection, inclusion
weights, saved raw-history records and fragment bytes. Collection uses the shared
stored-x converter, fixed32 provider proof, lease, resource guards and compact
writer, with one GPU session across all uncached fragments. It retains compact legal
policy logits and both raw WDL heads. Original derived-row indices remain
`row_index`; `selection_index` records their position in the frozen selection.
Bindings retain original source/raw-shard/physical-row/game/ply/worker identities,
input keys, selection strata/weights and raw-record/history hashes. Per-row stored-x
and TPG-feed hashes bind the actual inputs. History and original float32 input-key
correctness are inherited from the qualified bank; collection does not replay the
original corpus or establish a native repetition oracle. Numerical, library and
operational qualification for an actual selected-bank launch remains separate.

### Optional exact-epoch host overlap

`lc0_control_train.py --sampling-mode game_epoch --epoch-host-batch-overlap`
opts a future run into one host-preparation worker with at most one pending batch.
Omitting the flag preserves synchronous exact replay. The option is rejected for
replacement sampling and is recorded in each epoch's physical plan and receipt.
It does not change targets, sampling seeds, game/row order or optimizer settings.

The same `--epoch-max-working-set-gib` limit includes an explicit reservation of
16 maximum-schema persisted batch payloads for the retained generation. This is a
conservative allowance, not a claim that 16 copies are allocated. Each prepared
batch must also fit its exact retained bound: its array bytes plus eight bytes per
element, covering either converted CPU tensors or CUDA pinned sources, including
derived fields. Current collation uses each source at most once and no output
dtype wider than eight bytes. The planner and runtime include the reservation in
validated loads, materialization and optional-compaction decisions. New physical
plans are required; this is a payload working-set bound, not an RSS or GPU cap.

CUDA collation remains on the caller. An event retires the preceding H2D transfer
before another pinned generation can accumulate. The iterator joins its sole
producer at window/epoch boundaries and on close or failure; running I/O remains
subject to the existing outer process deadline. Prepared-ahead rows do not become
successful optimizer updates, and exact-mode errors remain terminal. The default
historical plan identity is unchanged; an enabled plan carries the reservation
alongside its source/order digest.

CPU tests cover actual emitted tensors and two-epoch final weights, RNG/order,
reservation refusal, ragged batches, one-future cleanup, and transfer-event order
with a fake CUDA boundary. They do not establish CUDA allocator behavior or speed.
This opt-in requires separately budgeted CUDA qualification and a matched timing
comparison before adoption; no existing training runtime is changed by adding it.

### Immutable policy overlays (explicit exact-epoch opt-in)

`bt4_policy_mix.py mix --output-storage immutable-overlay` writes a fresh global
policy recipe directly as replacement `policy_target` chunks and local metadata.
It inherits every other array from one ordinary, sealed base. It does **not**
create a full intermediate recipe copy. The existing copy mode remains the
default and retains its original audit, target arithmetic and output behavior.

This path requires three separate steps:

```bash
python scripts/target_overlay_storage.py seal-base \
  --shards BASE --output BASE_STORAGE_SEAL.json
# Add these options to the existing, fully specified global mix command:
# --output-storage immutable-overlay --base-storage-seal BASE_STORAGE_SEAL.json
# --expected-base-storage-seal-sha256 SHA256
python scripts/target_overlay_storage.py qualify-overlay \
  --shards OVERLAY --output OVERLAY_STORAGE_QUALIFICATION.json
# Add these options to an exact-epoch lc0_control_train.py command:
# --sampling-mode game_epoch --overlay-storage-qualification OVERLAY_STORAGE_QUALIFICATION.json
# --expected-overlay-storage-qualification-sha256 SHA256
```

The base seal validates actual ordinary shard content and row/history declarations,
streams the existing exact-epoch byte digest, and anchors it to every file's
membership, device, inode, size, modification time and change time. It is a new
storage proof, **not** a scientific recipe or held-out qualification. The overlay
manifest binds that seal, exact source shard/rows/history, replacement layout and
target bytes. The storage qualifier validates the actual composed arrays and
completed global-producer summaries. Training verifies the pinned qualification
before coverage checks and records its identity. Each epoch also rechecks the
actual qualification and exact ordered resolved staging paths; exact planning and eager reads
include both inherited and replacement bytes in their content identities and
reject changed dependencies. Each producer, qualifier, preflight and epoch owns
one validated seal index: repeated shard access checks its anchored receipt identity
instead of reparsing the complete corpus seal. This is operation-scoped metadata,
not a permanent unchecked cache. Ordinary shard digests remain unchanged.

The base and seal must remain at their recorded locations for the lifetime of
**every** dependent overlay. No base files are linked or modified by this producer.
Changing files, creating hard links, moving/restoring the base, or replacing an
identical file changes admission identities; a newly qualified lineage is needed.
Treat the base as retained training input when planning archival or reclamation.
Existing archival operators and frozen experiment coordinators have no new overlay
admission in this change: do not use their old receipts to authorize it.

Only fresh **global-policy** output and a single overlay corpus in exact-epoch
`lc0_control_train.py` are supported here. C20/ranked/recovery producer modes,
value-only overlays, overlay chains and mutable/replacement replay are unsupported;
the relevant opt-in paths refuse them. Generic readers retain their default refusal.
No existing runtime or registered experiment adopts this storage mode automatically.
The explicit exact-host-overlap option remains separate and defaults off.

Creation needs the retained base, original policy sidecars, new policy chunks and
small manifests, plus the mixer's existing bounded chunk buffers. Sealing and
qualification read real bytes; this is not a metadata-only shortcut. The composed
training arrays occupy the same decoded RAM as an ordinary corpus, so the existing
working-set limit still counts inherited arrays in full. CPU/file-system costs and
large-corpus throughput have not been measured for this path. The earlier estimate
of roughly 42 GiB avoided per additional 100M-row policy recipe remains conditional
on that sampled storage mix; it is neither measured exclusive allocation nor proof
that a complete 100M experiment fits on the SSD.

### Separate Ceres value mixture

`scripts/ceres_value_mix.py` consumes original SF values, historical BT4 WDL
sidecars and qualified Ceres dual-head sidecars. It copies the B100 policy corpus,
changes only `search_wdl`, and atomically publishes a complete manifest. The
`B100CeresV25` one-epoch profile keeps B100 policy and historical training settings.
Weights and Ceres head conversions are defined in the
[weighted-bootstrap record](experiments/2026-09-11-ceres-weighted-bootstrap.md).
This is a separate value intervention from the equal-weight policy mixture.

### Ceres corpus publication and admission

`scripts/ceres_materialize.py` runs either `CeresB50` or `B100CeresV25` from a
pinned, complete teacher manifest. It constructs the registered producer command,
defaults to the shared preparation lock and CPUs 0–1, keeps the GPU hidden and preserves the existing
STOP, disk, process-cleanup and eight-hour enclosing bounds. Run it with the exact
planned interpreter and an external timeout; a merged tool does not start a rewrite.

The plan records `schema`, `profile`, frozen `cwd`/`commit`, `python`, fresh `state`
and `corpus`, `producer_manifest` (path/hash), `producer_sha256`, `pins`,
`supervisor_sha256` and `stop_paths`. Default execution validates only. Add
`--execute --deadline UNIX_SECONDS` to the `--plan PATH
--expected-plan-sha256 SHA` invocation only after the concrete allocation qualifies.
The deadline includes preflight, waiting, publication checks and cleanup; at the
maximum eight-hour allocation, use an outer timeout of 28,770 seconds followed by
30 seconds of kill grace.

A reviewed independent allocation may optionally set `cpu_affinity` (a nonempty
list of unique available CPU IDs) and `preparation_lock`. The lock must be either
the existing `hybrid_endpoint_run01/preparation.lock` or the fixed
`hybrid_endpoint_run01/{profile}.preparation.lock`; arbitrary lock paths and
symlinks are rejected. This permits disjoint policy/value outputs to use separate
CPU allocations while reading shared immutable teacher data. It does not schedule
resources automatically or make overlapping writes safe. Duplicate attempts on
the same selected lock still contend, and the child inherits its lock FD and CPU
affinity. The receipt records the realized affinity and lock. Numeric thread
limits, batch 128, recipe/input pins, STOP, 150 GiB reserve, sampled 32 GiB output
cap and deadline cleanup stay unchanged. There is no implicit memory-limit guard.

Never edit a supervisor or producer checkout serving an active materialization.
Prepare a separately pinned runtime and plan; preserve producer bytes when only
the allocation changes. Concurrent SSD traffic can slow both jobs despite distinct
cores. Choose allocations against current memory/disk use and the existing run's
deadline, and stop only the newly owned job if it threatens that allocation.

Both Ceres producers bind each completed shard's storage identity into the summary
and recheck it before publication. `scripts/ceres_corpus_qualification.py` consumes
that proof together with the successful materialization receipt. It checks actual
array layouts, recipe attributes, full coverage and unchanged source/teacher/output
identities, then uses the training coordinator's recipe admission. It does not
reread all payloads or re-encode histories already verified by the producer.

Qualification accepts `--plan PATH --expected-plan-sha256 SHA --out FRESH_JSON`,
with `--execute` publishing the receipt. Its plan pins the producer manifest,
final derive/rewrite summaries, successful materialization receipt and producer
sources, plus the profile, corpus, deadline, disk reserve and STOP paths. Run with
two CPU threads, GPU hidden and a separately bounded enclosing process. The emitted
`PASS_REGISTERED_CORPUS_QUALIFICATION` receipt still precedes the frozen prospective
schedule check and exact training manifest; it does not launch training.

### Combined original/G10 value comparison

`scripts/combined_corpus_train.py --manifest PLAN.json` inspects the explicit
`combined35m_value_seed101` training plan; `--execute` performs one fresh arm.
This is the registered 21-cohort comparison, not a replacement for the historical
single-source coordinator. `Combined35M_SF100` uses the admission report's ordered
**B100** roots, and `Combined35M_V50` uses its V50 roots. The `source` arm exists
only for identity proof. Both use seed 101, batch 512 and two planner/loader
workers under the unchanged frozen trainer. Its history/value/encoding gates
remain in force.

The plan binds `corpus_manifest`, actually passed `prospective`, `runtime_manifest`,
`preregistration`, `selected_subset_qualification`, the pretraining `opening_panel`, absolute `state`/`run` and
`stop_paths`, plus exact `code_pins` for the coordinator and its schedule, stage,
original-coordinator and memory helpers. It selects `schema: 1`, the profile and
role, `training_seconds: 21600` and `coordinator_seconds: 27000`. V50 additionally
requires the same comparison's completed SF100 `previous_training` receipt.
No existing output is adopted or resumed.

The selected-subset qualification binds the same manifest, prospective schedule,
runtime/config and ordered roots, plus the frozen preflight's exact per-arm
`partial_corpus` records. The explicit `--allow-partial-corpus` flag permits the
complete selected G10 products whose global raw producers are still running.
Every flagged derivation must be finalized, and no original-corpus or unselected
shard may appear. Completed training must reproduce those exact records. This
does not enable `--allow-leak` or `--allow-mixed-history`, or admit unfinished
selected products. The frozen trainer still runs its preflights before staging
or constructing a model.

The GPU lease wait counts toward the inclusive coordinator allowance; the full
training stage must still fit before it can start. The existing owned-stage
wait loop applies STOP, disk, deadline and host-memory checks, retaining its
failure cleanup. Startup requires 48 GiB available RAM, running work 32 GiB and
150 GiB free SSD. There is no CUDA address-space cap. An outer bounded operator
still owns terminal evidence and the absolute launch deadline.

After training, the lease is released and a 1,800-second CPU stage checks actual
staged link order, full game columns against the prospective witnesses, summary
window totals and the actual planned/realized physical hash. It does not rerun
all prospective planners, open feature/target arrays or load a model. Canonical
row-order equivalence inherits the frozen sampler and matching game-column proof;
it is not a newly recorded per-row training trace. Failed completion preserves
the run and cannot produce a valid training receipt.

The fixed `bt4_package_readout.py` accepts this explicit combined profile with
`training: {candidate_training: PIN, reference_training: PIN}`. It requires V50
versus SF100, matching admitted corpus/schedule/panel and completed process
receipts, 256 pairs, 400 simulations, arena seed 20260913 and both priors 1.0.
The same seed regenerates the pretraining opening panel from the pinned book. Its generic
`launch_qualification_verified: false` remains honest: actual arena launch
qualification is still separate from completed training and bank validation.
The historical generic package schema and two-depth recipe launcher are unchanged.

`combined_corpus_arena.py` prepares and runs the registered Combined35M V50/SF100
value comparison after both actual training receipts qualify. It uses the pinned
original combined-training verifier, the existing arena runtime qualification,
and the existing owned-stage supervisor. Its separate CPU probe checks the actual
checkpoint pair and regenerates the frozen 256-pair panel at seed 20260913.
The explicit command fixes 512 games, 400 simulations, priors 1.0, rolling 128
and batch 4096; legacy direct-screen defaults do not select this experiment.
Preparation and execution require pinned manifests and absolute owned deadlines;
no-flag inspection is static only. The active trainer and arena implementation are
unchanged. Final host adoption still binds genuine completion, runtime and source
pins; metadata drafts do not authorize a match.
