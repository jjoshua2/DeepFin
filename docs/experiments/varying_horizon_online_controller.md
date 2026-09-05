# Varying-horizon online controller screen

Status: preregistered successor to closed PR #505. This branch starts from `main` and must remain a small diagnostic-only experiment.

## Decision question

Does independently varied **remaining total search budget** add predictive value beyond current search state and search age for an online match-search continuation decision?

The primary comparison is `M_budget` versus the age-controlled `M_age`. A state-only model is retained to measure the separate value of ordinary search age:

- `M_state`: current search-state features only.
- `M_age`: `M_state` plus `log1p(cumulative_nodes)`.
- `M_budget`: `M_age` plus total/remaining-budget features and a small fixed interaction set.

This experiment must not repeat PR #505's confounding of age and remaining budget, and it must not rank an entire held-out bank to fill a quota. Every held-out decision is made independently with a fixed online threshold.

## Scope constraints

Allowed implementation:

- one trajectory collector;
- one analyzer;
- focused tests;
- this protocol document.

Not allowed:

- engine, search, native-extension, model-loader, or global import changes;
- a bank-wide quota allocator;
- stopped-position re-entry;
- a new provenance or filesystem-hardening framework;
- boosted trees, neural predictors, feature searches, or post-hoc interactions;
- deployment from this experiment alone.

A passing result authorizes only a subsequent real-clock/history bank.

## Collector

Run the current production-shaped root-log walker search on a stable checkpoint:

- two walkers;
- 2,048 simulations per chunk;
- eight accumulated chunks per position (16,384 simulations maximum);
- production evaluator factory and warmup;
- production terminal/root shortcuts;
- production Syzygy path when available;
- deterministic phase/source-balanced position ordering.

Collect one uninterrupted eight-chunk trajectory. Reuse its prefixes as paired counterfactual total horizons. Search itself must not be told a shorter horizon, so the state at chunk `t` is identical for `H=4`, `H=6`, and `H=8`.

Every completed row must contain at least:

- position key, FEN, phase, and source;
- source-game group when a canonical matched index is available, otherwise an explicit position-only smoke group;
- chunk number and cumulative nodes;
- chosen action/UCI and expected-score regret;
- root visit gap and entropy;
- root Q and emitted-action Q gap;
- best-move flip and stability count;
- Q drift and visit churn;
- the current clock-free complexity-predicate decision;
- raw root actions, visits, and aligned child Q values.

Only complete eight-row trajectories enter the bank. Resume must remove incomplete or duplicated tail groups before appending. Terminal-shortcut/incomplete positions are excluded and replaced until the requested number of complete trajectories is reached.

## Paired total-budget variation

Evaluate total horizons:

- `H=4` chunks = 8,192 simulations;
- `H=6` chunks = 12,288 simulations;
- `H=8` chunks = 16,384 simulations.

All horizon versions of a position and all of its chunks remain in one outer source-game fold.

## Target

Let `R_t` be expected-score regret after chunk `t`. For compute price `lambda` per additional 2,048-simulation chunk, the continuation advantage at state `t` under total horizon `H` is:

```text
A(t, H, lambda) = max over d in {t+1, ..., H}
                  [R_t - R_d - lambda * (d - t)]
```

This is a finite-horizon value-to-go target. It permits a negative target: continuing must buy at least one chunk, while stopping now has value zero.

Predeclared price grid:

```text
0, 0.000025, 0.00005, 0.0001 expected-score units per chunk
```

Primary price: `0.00005`. Other prices are Pareto diagnostics and cannot replace the primary result after seeing the data.

## Fixed feature ablation

`M_state` uses only:

- visit gap and visit entropy;
- emitted-action Q gap plus a missing indicator;
- root Q;
- best-move flip and stability count;
- Q drift and visit churn, each with a missing indicator;
- piece count, legal-move count, and phase indicators.

`M_age` adds only:

- `log1p(cumulative_nodes)`.

`M_budget` adds only:

- `log1p(total_node_horizon)`;
- remaining chunks;
- remaining fraction;
- remaining-fraction interactions with visit gap, entropy, flip, Q drift, and visit churn.

Use deterministic ridge regression. Ridge strength is selected by grouped inner CV using training groups only. Outer predictions are grouped out of fold.

## Deployable online policy

For a held-out state, continue independently iff:

```text
predicted A(t, H, lambda) > 0
```

After continuing, observe the next real state and decide again. Once stopped, a position never re-enters. Test-set spend emerges from the fixed threshold; no held-out rank, percentile, quota, or future-bank information may set it.

Compare:

- fixed full search at each horizon;
- the current clock-free complexity predicate;
- `M_state`;
- `M_age`;
- `M_budget`;
- per-position hindsight best stopping depth as an upper-bound diagnostic only.

## Sample plan and expected cost

Stage 1: 512 complete positions, eight chunks each. Based on the 24-position pilot rate, expected cost is roughly 8.5 GPU-hours, but actual runtime must be reported.

Stage 2: expand the same deterministic bank to 1,024 complete positions only if every pilot gate passes. Expected total cost is roughly 17 GPU-hours.

Canonical source-game groups are preferred for Stage 1 and required for Stage 2. A position-grouped pilot can only produce `POSITIVE_SMOKE_REQUIRES_SOURCE_GAME_REPLICATION`, never authorization to expand directly.

## Precommitted 512-position gate

Expand only if all applicable gates pass:

1. At least 512 complete positions and at least 256 evaluation groups.
2. At least 30 positions have an attainable later stopping depth improving expected-score regret by at least 0.001.
3. The hindsight stopping upper bound shows practical headroom at the primary price: either at least 10% compute savings with mean expected-score degradation no worse than 0.0001, or at least 0.00025 mean expected-score improvement at no greater compute.
4. Out-of-fold `M_budget - M_age` mean net utility is positive for at least two of the three total horizons.
5. `M_budget` p95 and p99 regret are no worse than `M_age` by more than 0.0005.
6. Horizons and the primary price exactly match this protocol.

Failure means `STOP_NO_EXPANSION`.

## Precommitted 1,024-position gate

Advance only if every gate passes:

1. At least 1,024 complete positions, at least 512 source-game groups, and canonical source-game grouping throughout.
2. A paired source-game cluster bootstrap with at least 1,000 resamples has a strictly positive lower 95% bound for held-out `M_budget - M_age` net utility.
3. Practical value: either at least 8% fewer simulations with mean expected-score degradation no worse than 0.0001, or at least 0.00025 mean expected-score improvement at equal or lower compute.
4. `M_budget` beats the clock-free complexity predicate at the primary price.
5. `M_budget - M_age` is positive for at least two of the three horizons.
6. `M_budget` p95 and p99 regret are no worse than `M_age` by more than 0.0005.
7. The result is not driven entirely by one phase or one source class.
8. Horizons, price grid, and primary price exactly match this protocol.

Passing authorizes `ADVANCE_TO_REAL_CLOCK_BANK`, not deployment.

## Permanent stop rule

Stop work on remaining-budget-conditioned root-feature match controllers for the current root-log walker/checkpoint family if:

- the 512-position gate fails;
- the full `M_budget` versus `M_age` gate fails; or
- no online model beats the existing complexity predicate at a practically meaningful Pareto point.

After failure, do not rescue the result with a larger bank, a new model family, additional interactions, post-hoc thresholds, or more provenance hardening. Reopen only after an exogenous change materially increases meaningful late-search corrections or real match logs demonstrate repeated time-allocation misses.

This experiment is separate from the already-successful G10 NNUE staircase and does not retest or block G10.