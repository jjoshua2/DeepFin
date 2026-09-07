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
| Derive replay targets | `scripts/derive_corpus_targets.py` | Input corpus identity, target recipe, history/encoding stamps and completed-shard inventory |
| Train a controlled LC0-shaped comparison | `scripts/lc0_control_train.py` | Declared architecture, data lineage, objective, view/step budget and checkpoint identity |
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
