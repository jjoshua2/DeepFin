# Value Head Architecture Experiments (2026-04)

## Motivation

Activation probe on iter 167 of `trial_3cef0` (2026-04-20) showed 86-91% of the
128-channel hidden layer in each value head (`value_wdl`, `value_sf_eval`,
`value_categorical`) was dead or always-near-zero. Dead biases were inherited
from the salvage pool and hadn't moved in 15 hours of training, gradient flow
through the dead-Mish zone too small to revive them. All three core losses
(`wdl_loss`, `policy_loss`, `test_policy_loss`) had been flat for ~15 hours.

Hypothesis: the `Linear(8192 → 128) → Mish` squeeze in `ValueHead` is fragile
at our scale (15M params, ~15M replay positions). LC0 BT4 uses a gentler
bottleneck (`Linear(2048 → 128)` with token_dim=32 instead of our 128).

## Experiment protocol

- **Shared start**: all experiments salvage-restart from pool
  `data/best_pools/value_reinit_src` (regret 0.0810, 221 replay shards, value
  heads freshly reinited via `scripts/reinit_value_heads.py`).
- **Iter budget**: 60 iters (~5h) per experiment.
- **Exit criteria**: plateau (regret flat for 15+ iters) or budget exhausted.
- **Rename trial dir after each run** to `exp_<letter>_<label>` so TensorBoard
  overlay reads cleanly:
  ```
  mv runs/pbt2_small/tune/train_trial_<hash>_... runs/pbt2_small/tune/exp_A_plan_a_reinit
  ```
- **Compare in TB**: `tensorboard --logdir runs/pbt2_small/tune`
  — overlay scalars `train/wdl_loss`, `train/policy_loss`, `test/*`,
  `eval/wdl_regret`, `pid/ema_winrate`.
- **Dead-neuron probe** at end of each run: rerun the ad-hoc script from
  Apr 20 session (hook `nn.Mish`/`nn.SiLU` outputs on 512-sample batch, report
  per-module dead fraction).

## Reference baseline (pre-reinit, iter 167 of trial_3cef0)

| metric | value |
|---|---|
| wdl_loss | 0.82 (flat 15h) |
| policy_loss | 2.51 (flat 15h) |
| test_policy_loss | 2.57 |
| sf_eval_loss | 0.35 |
| categorical_loss | 2.70 |
| regret | 0.087 (plateau after reaching 0.081 at iter 7) |
| value_wdl dead fraction | 88% |
| value_sf_eval dead fraction | 91% |
| value_categorical dead fraction | 86% |

## Exp A — Plan A: value head reinit (no arch change)

**Config diff**: none (identical arch to baseline). Ran
`scripts/reinit_value_heads.py` on the pool's `trainer.pt` to Xavier(0.1) the
value heads' weights, zero their biases, zero their AdamW exp_avg /
exp_avg_sq.

**Motivation**: does reinit alone suffice, or do we need arch change? If Exp A
hits a new low regret, the ceiling was the dead biases. If Exp A plateaus
again near 0.087, the architecture itself is the limit.

**Start**: 2026-04-20 11:17 (trial `ff3ba`, pool `value_reinit_src`).
**Stop**: 2026-04-20 14:24 (51 iters, ~3h).

**Observations**:
- iter 1: wdl_loss 1.009, cat_loss 2.98, sf_eval_loss 0.94 (expected spike from reinit)
- iter 2: wdl_loss 0.855 (near baseline already), cat_loss 1.59, sf_eval_loss 0.48 (big wins)
- iter 5: regret 0.078 — best of run, below the inherited 0.081 floor
- iter 11-51: regret monotone regression: 0.087 → 0.10 → 0.13 → **0.136** stuck
- policy_loss briefly 2.32 at iter 20 (below baseline 2.51), then drifted back to 2.50
- Value losses stayed improved throughout: categorical 1.35 vs 2.70 baseline; sf_eval 0.36 vs 0.35.

**Result**: FAILED. Regret regressed from 0.087 baseline to 0.136 (55% worse) over 51 iters.

**Verdict**: Reinit rescued the value-head internal losses but broke value/policy coordination. The old policy head had learned to interpret the old (dead-biased) value features; fresh value features confused it, and the policy head didn't adapt in time. Net play quality worsened despite better value supervision.

**Lesson for Exp B/C/D**: When we change value-head topology, expect a similar policy/value decoupling. Consider reinitializing the policy head at the same time (via existing `bootstrap_zero_policy_heads: True` on restart), or using a lower LR warmup so policy can re-tune more gradually.

**Recovery**: restored pre-reinit state to new pool `data/best_pools/value_baseline` (from the `.bak` taken by `scripts/reinit_value_heads.py`). Training continues from this to reach baseline 0.087 plateau, at which point Exp B can start.

---

## Exp B — BT4 shrink: `token_dim` 128 → 32

**Config diff**: `ValueHead(embed_dim, out_dim, token_dim=32)` per-head
(currently hard-coded default; plumb through `TransformerConfig` or change the
default). Bottleneck input becomes 64×32 = 2048 instead of 8192. Matches LC0
BT4 sizing.

**Motivation**: LC0's value head has been stable over billions of positions at
this sizing. Smaller bottleneck input means less pre-activation saturation
capacity per channel, should have fewer dead channels at equilibrium.

**Param delta**: token_proj shrinks (384×32 vs 384×128, saves ~150k per head)
and the bottleneck Linear shrinks (2048×128 vs 8192×128, saves ~760k per
head). ~2.7M fewer value-head params across three heads.

**Start**: _tbd, after Exp A plateaus_

**Result**: _tbd_

**Verdict**: _tbd_

---

## Exp C — BT4 shrink + shared value trunk

**Config diff**: introduce `SharedValueTrunk` that does the
`Linear(embed_dim → 32)` per-square + `Linear(2048 → 128) → Mish` once, then
three small output taps (`Linear(128, 3)`, `Linear(128, 3)`, `Linear(128, 32)`)
for wdl / sf_eval / categorical respectively.

**Motivation**: WDL, sf_eval, and categorical are highly correlated targets
(all are "who is winning"). Sharing the trunk gives the backbone 3× more
gradient signal per parameter. Should improve sample efficiency at our
modest training-data budget.

**Risk**: if the three heads *do* need meaningfully different features, sharing
harms them. Watch `sf_eval_loss` vs `wdl_loss` separately to detect this.

**Param delta**: ~2× savings over Exp B (bottleneck now shared across 3 heads).

**Start**: _tbd_

**Result**: _tbd_

**Verdict**: _tbd_

---

## Exp D — Shared trunk + SwiGLU activation

**Config diff**: in the shared trunk, replace
`Linear(2048, 128) → Mish` with
`Linear(2048, 256) → split → gate * silu(value)` (SwiGLU).

**Motivation**: SwiGLU is the default in modern LLMs (GPT, Llama) for good
reason — multiplicative gate has no dead zone, smoother optimization
landscape. We accept ~2× the first-Linear params to get activation resilience.

**Param delta**: doubles the shared trunk's first Linear. Still net smaller
than the original (un-shared, un-shrunk) value heads.

**Start**: _tbd_

**Result**: _tbd_

**Verdict**: _tbd_

---

## Summary

| Exp | wdl_loss final | regret final | dead % (max across heads) | iters to plateau | verdict |
|---|---|---|---|---|---|
| baseline | 0.82 | 0.087 | 91% | n/a (inherited) | - |
| A reinit | tbd | tbd | tbd | tbd | tbd |
| B bt4 shrink | tbd | tbd | tbd | tbd | tbd |
| C + shared | tbd | tbd | tbd | tbd | tbd |
| D + swiglu | tbd | tbd | tbd | tbd | tbd |
