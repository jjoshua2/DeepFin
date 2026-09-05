# TailRL-inspired policy coverage screen

## Status and scope

Preregistered 2026-09-05. **No real replay readout, training arm, or arena has run.**
This change adds a bounded checkpoint diagnostic and an isolated differentiable
auxiliary-loss primitive. It does not change any production loss, YAML, search,
replay schema, checkpoint, running process, or model publication path. There is
no new live configuration key. Merging does not enable the auxiliary objective.

Motivation: [Ramasubramanian et al., *Tail-Likelihood Reinforcement Learning*,
arXiv:2609.02987v1](https://arxiv.org/abs/2609.02987v1), submitted September 2, 2026.
The paper optimizes the log-probability of exceeding reward thresholds rather
than just expected rollout reward. The helper here shares that event-log-mass
form, **not** its finite-rollout gradient estimator or experimental protocol.

## Hypothesis and frozen ruler

Improvement in ordinary primary-policy target CE can coexist with loss of
probability on reference-rare moves promoted by a recorded search target.
The initial test is whether this failure mode is present, not whether a different
loss already improves chess strength.

For each held-out position, define a fixed set C of legal moves satisfying:

- Reference T=1 legal-renormalized prior <= 0.01.
- Recorded `policy_target` mass >= 0.02.
- Recorded target / reference prior >= 4.

These are initial diagnostic thresholds, not paper hyperparameters. Freeze them
before looking at candidate outcomes. Bank the actual boolean membership mask;
never reselect it using candidate priors. Numerical boundaries allow only floating
point roundoff. An empty cohort is an unread instrument, not a zero-effect result.

**Label contract:** use the primary `policy_own` output (`policy` is its TinyNet
alias), current-position `policy_target`, and `legal_mask`. Respect `has_policy`
and `has_legal_mask`. Do not use `sf_policy_target`, `sf_multipv_raw`, or
`policy_sf` as current-position move rewards: those describe P1/opponent replies.
Nor can `prior_top1_index/prob` reconstruct the generating model's entire prior.

The reported cohort is therefore **reference-rare, historical-search-target-
promoted**, not proven same-model search rescue or objectively exceptional play.
The historical target may come from another checkpoint and includes the prior
and completed-Q shaping. Promotion alone does not demonstrate tactical quality.
A genuine rescue study must bank same-checkpoint priors and fresh search traces,
then establish move quality independently (deeper search, tactics, tablebases,
or another preregistered adjudicator appropriate to the claim).

## Run the diagnostic

Use immutable checkpoint copies and quarantined replay that was not used to train
the compared checkpoints. Install the repository's locked development environment
first; see [development](../development.md). No background jobs are launched.

```bash
PYTHONPATH=. python scripts/policy_tail_diagnostic.py \
  --reference /absolute/snapshots/before/trainer.pt \
  --candidate /absolute/snapshots/after/trainer.pt \
  --replay-dir /absolute/quarantined_replay \
  --output-dir runs/tailrl-screen-20260905 \
  --max-positions 2048 --max-shards 4 --batch-size 64 --threads 2
```

The output directory must be new. CPU is the default. Budget additional GPU memory
explicitly before using `--device cuda`: both models are resident. The default
horizon is one pass over at most 2,048 eligible rows in the first four sorted
shards; this is a bounded convenience sample, **not** a representative random
sample. Zarr is read lazily in batches through the shared guarded loader. Legacy
NPZ shards are eagerly loaded by that loader and can use substantially more RAM.

The collector requires compact lc0_1858 checkpoints with embedded architecture,
matching stored input-history/repetition-fix identities, and exact input widths.
It uses the shared history selector without lossy remapping. It refuses implicit
policy-index conversions, padding/threat upgrades, and dynamic-relation models.
These are explicit first-version limitations, not silently defaulted features.

Outputs: `observations.npz` retains inputs, both logits, targets, legal masks,
frozen cohort, source-qualified row/game keys, and a JSON provenance manifest.
`per_row.npz` retains paired CE and log-tail-mass observations. `report.json` is
the completion marker and records checkpoint/implementation/bank SHA256 hashes,
architecture, source metadata, bounds and presence coverage. Failed collection
leaves no completion report; use a new directory for a retry.

Recompute a readout without inference, changing the checkpoint files, or moving
the cohort definition:

```bash
PYTHONPATH=. python scripts/policy_tail_diagnostic.py \
  --bank runs/tailrl-screen-20260905/observations.npz \
  --output-dir runs/tailrl-screen-20260905-reread
```

The bank reader rejects checkpoint/cohort overrides. Keep banks outside git.
Retained encoded inputs preserve the actual observations even when source replay
is later unavailable; do not feed these held-out inputs back into a training arm.

## Deciding measurements and initial rule

Read overall paired `target_ce_delta` alongside `rare_log_mass_delta`, the mean
change in log probability of hitting **any** cohort move. Also read
`rare_action_logp_delta` and the fractions of individual cohort moves losing
10x/100x probability. Union mass can improve while a particular good move vanishes;
the individual-action metrics intentionally expose this failure of the union loss.

The IID curves compute the position-wise `1 - (1 - p(C))**N` and then average.
They are explicitly **independent samples with replacement**, not MCTS simulation
budgets, Gumbel top-k inclusion, or a predicted search-strength improvement.
They use T=1 policy, not the search's tempered/noisy prior. Log-sum-exp keeps
log-retention measurable even when exponentiating a tiny prior underflows.

Initial flag for further investigation: CE improves by at least 0.5% relative to
the reference AND the geometric-mean union mass falls by at least 2x
(`rare_log_mass_delta <= -log(2)`). Require at least 100 cohort moves across 50
verified source-qualified games before treating this as more than a small-sample
example. Confirm the paired signs using a game-cluster bootstrap on retained
observations before a statistical conclusion. These gates are **not** automatic
strength or causality tests, and no automatic verdict is emitted by the CLI.

The CLI deliberately provides point estimates, not an IID-position confidence
interval. Stored `(shard path, game_id)` keys avoid collisions but do not prove
that split games, duplicated positions, or shared seed families are independent.
Verify producer lineage and consolidate such clusters first; missing game IDs
are counted, never invented. If cohort/provenance coverage is insufficient, the
readout is **unread** at the declared budget, not permission for an automatic
larger run. A negative finding is conditional on this sample/checkpoint pair.

## Optional next stage: controlled loss ablation

`chess_anti_engine.policy_tail.tail_event_loss(logits, legal, events)` accepts
fixed boolean `[batch, thresholds, actions]` events. It returns mean negative
log probability mass, averaging nonempty thresholds within each eligible row.
Empty events/rows are excluded; an all-empty batch has an exact, connected zero.
FP16/BF16 accumulation is promoted to float32; no probability floor erases rare
move gradients. This is a **single-move proxy**, not adversarial rollout RL.

A future bounded training comparison can use the existing full loss unchanged
as the control, and `existing_loss + lambda * tail_event_loss(...)` as the arm.
For a one-tail search-promotion experiment, `events = frozen_train_cohort[:, None, :]`.
For a closer threshold-mixture experiment, construct nested events from independently
measured **current-position** mover-quality labels, not search probabilities
misnamed as rewards. Fix quality thresholds, coverage and aggregation rules first.

Use identical initialization, optimizer state, seeds, minibatch order and update
budget, with a **separate training cohort** and the untouched held-out ruler.
Preregister lambda, the gradient-share cap/abort rule and exact compute budget
before launching; none is authorized or selected by this diagnostic. Include a
simpler broad-softening/matched-entropy control before attributing preservation to
upper-tail targeting. A union event loss does not preserve diversity within its
set, guarantee good moves outside the measured set, or prove search quality.

Only a subsequent paired playing evaluation at declared search/time budgets can
support a strength claim; follow [evaluation](../eval_protocol.md). Recovery for
this stage is simply stopping the isolated command; it has no model/YAML writes.
Any later training arm must use a separate run directory and preserve its donor.

## Validation/readout log

Local partial-workspace validation: 41 focused numerical/bank/CLI tests passed,
including an explicit CE-improves/rare-move-loses-50x fixture, within-tail collapse,
extreme-logit gradients, gradcheck, empty events, legal masks, head selection,
missing flags, provenance roundtrips and overwrite refusal. These are synthetic
mechanism tests, not observed chess improvements.

The added real TinyNet-checkpoint + lazy-Zarr collector test requires the installed
repository/native dependencies and was not run in this container. Full repository
lint/CI and real replay comparisons were also not run here. The container could
read/write GitHub through the connector, but could not clone the full repository
or install its locked dependencies. Self-review only; no independent reviewer.

```bash
# In a complete, installed checkout:
python -m pytest tests/test_policy_tail.py -q
./scripts/lint.sh
```
