# Training Log

Tracks experiments, what was tried, why, and results. Prevents duplicate effort.

---

## Run 1: First PB2 with DiskReplayBuffer (2026-02-28, ~7 hours)

**Config**: `configs/pbt2_small.yaml` (original wide bounds)
- 10 trials, pb2_perturbation_interval=20, shuffle_buffer_size=20000
- gate_games=100, gate_threshold=0.50, gate_mcts_sims=1
- games_per_iter=1000, selfplay_batch=10, mcts_simulations=64
- Shared iter-0 data: 99 shards, 98,921 positions from bootstrap net

**Result**: All 10 trials ran ~7 iterations each. All ended with ERROR (SIGTERM on driver exit, not real errors). opponent_strength peaked at ~15.5.

**Issues found**:
- PB2 perturbation_interval=20 too slow — never perturbed in 7 iters
- Bottom 4 trials were declining (bad hyperparams), wasting compute
- Top 6 clustered at 54.8-55.6% winrate, bottom 4 at 52.1-53.4%

**Learnings**:
- Top performers had diff_focus_q_weight 4-7, pol_scale 1.5-4.6, slope 2.7-5.5
- Bottom had very high q_weight (7-8) or extreme pol_scale
- Tightened bounds based on this data

---

## Run 2: Resumed with --resume (2026-02-28, ~2 hours)

**Config**: Changed pb2_perturbation_interval to 8, added pb2_bounds_* to YAML

**Result**: --resume loaded old tuner.pkl, ignoring YAML changes. Still using interval=20 and wide bounds. Killed after realizing this.

**Bug found**: pb2_bounds_* weren't flowing through config pipeline:
1. config_yaml.py wasn't passing pb2_bounds_* from tune: section
2. run.py wasn't forwarding them to base_config dict

**Fix**: Added dynamic pb2_bounds_* passthrough in both files.

---

## Run 3: Fresh start with tightened bounds (2026-02-28 22:52, ~10 hours)

**Config**: pb2_perturbation_interval=8, tightened bounds:
- diff_focus_q_weight: [3.0, 8.0] (was [2.0, 10.0])
- diff_focus_pol_scale: [1.0, 5.0] (was [1.0, 6.0])
- diff_focus_slope: [2.0, 6.0] (was [1.0, 6.0])
- diff_focus_min: [0.02, 0.15]
- lr: [1e-4, 3e-3] (was [1e-5, 3e-3])
- w_sf_eval: [0.05, 0.30], w_sf_move: [0.05, 0.35], w_categorical: [0.05, 0.22]

**Result**: 10 trials ran 12-15 iterations. PB2 perturbed at iteration 8.

**Issues found**:
1. **Gate check broken**: Using 1-sim MCTS for gating while selfplay uses 64-sim. Raw policy much weaker than MCTS-guided policy. Gate systematically rejected every training step after iteration 2-3. Model was FROZEN — no learning happening for 10+ iterations. All compute wasted on selfplay + gate games with no model updates.
2. **PB2 cloned 5 trials to identical config**: All 5 bottom performers got Trial 00007's config (lr=0.000179). PB2 explore step didn't differentiate them. 50% of compute running duplicates.
3. **Cloned lr too low**: lr=0.000179 caused all clones to slowly decline.

**Gate failure analysis**:
- Gate plays post-training model vs Stockfish with 1-sim MCTS
- Should have been: new model vs old model (KataGo-style)
- Even with correct comparison, 1-sim vs 64-sim mismatch would cause failures
- Trial 00000: gate=0 on 8 of 13 iterations
- Trial 00002: gate=0 on 7 of 15 iterations
- After iter 3, ALL trials failed almost every gate check

---

## Run 4: Gating disabled (2026-03-01 08:33, ~5 hours)

**Config**: gate_games=0 (disabled gating entirely)
- All other settings same as Run 3 (tightened bounds, interval=8)
- Searched 14 params: lr, diff_focus_*, temperature, playout_cap_fraction, sf_policy_temp, sf_policy_label_smooth, feature_dropout_p, w_volatility, w_sf_move, w_sf_eval, w_categorical

**Result**: All 10 trials ran 7-9 iterations. **Value head death spiral** — all trials declining.

**Key metrics at final iteration** (all trials similar):
- Winrate: 45.6-47.1% against FULLY RANDOM Stockfish (should be >50%)
- WDL loss: ~0.95 CE — FLAT, not improving (near random 3-class entropy)
- Policy loss: 3.60-4.15 — improving steadily
- Total loss: 7.7-9.6 — going down, but driven by policy, not value
- PID stuck: random_move_prob=1.0, can't make SF weaker

**Root cause: gradient allocation imbalance**:
```
Weighted loss budget (iter 9, Trial 00000):
  policy (w=1.0):     3.60  37%  ← improving
  soft_policy (w=0.5): 1.97  20%  ← improving
  sf_wdl (w=1.0):     1.57  16%  ← slight improvement
  wdl (w=1.0):        0.95  10%  ← FLAT (the thing MCTS uses!)
  sf_move (w=0.1):    0.64   7%  ← slight improvement
  future (w=0.1):     0.63   7%  ← flat
  other:              0.25   3%
```
The WDL head (what MCTS uses to evaluate positions) only gets ~10% of total gradient. Model optimizes policy prediction (64% of gradient) but value head stays stuck → MCTS can't evaluate positions → move selection degrades → winrate drops.

**Top 5 vs Bottom 5 analysis** (narrow spread, 0.466-0.471 vs 0.456-0.465):
- lr: Top 0.0013, Bottom 0.0019 — lower LR slightly better
- temperature: Top 1.21, Bottom 1.45 — lower temp better
- dropout: Top 0.26, Bottom 0.37 — less dropout better
- diff_focus params: well-converged across all trials, narrow useful ranges

**Learnings**:
- diff_focus and selfplay params are well-converged — ready to pin
- The loss weights that control gradient allocation (w_soft, w_future, w_sf_wdl) are the knobs that matter now but were FIXED during this run
- 14 search params too many for PB2's GP — need to reduce to 5-7

---

## Run 5: Loss weight search + overfitting fix + SF quality (2026-03-01 16:46)

**Config changes**:
- **Pinned** (from Run 1-4 top-5 medians): diff_focus_q_weight=4.8, pol_scale=3.8, slope=4.0, min=0.09, temperature=1.1, playout_cap_fraction=0.30, sf_policy_temp=0.35, sf_policy_label_smooth=0.10, feature_dropout_p=0.25, w_sf_eval=0.10, w_categorical=0.10, w_volatility=0.10
- **Searching** (5 params via PB2): lr [5e-4, 3e-3], w_sf_wdl [0.0, 2.0], w_soft [0.0, 1.0], w_future [0.0, 0.3], w_sf_move [0.0, 0.3]
- **Code changes**: harness.py now builds search space entirely from pb2_bounds_* config keys (data-driven, no hardcoded params). trainable.py re-reads loss weights from config each iteration so PB2 perturbations take effect.
- Cosine annealing aligned: lr_T0=680 (8 iters × ~85 steps/iter), lr_T_mult=1

**Critical fix: overfitting on iter-1 shared data**:
- trainable.py computed `steps = total_positions // batch_size` → 99k/256 = 386 steps on iter 1
- 386 steps on 99k positions = nearly 1 full epoch → model memorized training data
- Bootstrap model predicted Q≈0 (flat), but after training predicted Q≈-0.3 for ALL positions
- MCTS with confident-but-wrong value head is WORSE than MCTS with flat value head
- Direct test: bootstrap net 65% WR, trained net 40% WR (same SF settings)
- Fix: `steps = min(max(1, total_positions // batch_size), train_steps_cap)`
- train_steps=100: caps iter 1 at 100 steps (26% of data), normal iters ~97 (uncapped)

**SF analysis quality**:
- SF nodes increased 250→1000 for better analysis targets (sf_wdl, sf_move)
- Skill Level was never an issue in tune path — trainable doesn't pass it, SF defaults to full strength
- Difficulty controlled entirely by random_move_prob (already the case)
- pid_min_nodes=pid_max_nodes=1000 (nodes fixed, PID only adjusts random_move_prob)

**Result**: All trials still declined (58.8%→19.8% over 7 iters in Run 5c), even with train_steps=100 cap and SF nodes=1000. w_sf_wdl ranged 0.02-1.95 across trials — ALL declined regardless.

**Root causes found** (Run 5 post-mortem):

1. **Bootstrap checkpoint poisons optimizer/scheduler/step** (THE MAIN BUG):
   - Bootstrap trains for 13,323 steps → checkpoint saves step=13323, optimizer momentum, scheduler state
   - trainable.py loaded ALL of it via `trainer.load(bp)` → step=13323
   - With warmup_steps=400, step 13323 >> 400 → warmup completely skipped
   - Scheduler at position 582 of 680-step cosine cycle → lr ≈ 0.0000252 (near eta_min)
   - First ~98 steps of iter 0: lr ≈ 0.00002 (no learning). Then cosine restart → lr SPIKES to 0.0003 (15x jump)
   - PB2's lr perturbation had ZERO effect: scheduler's base_lr was locked to bootstrap's 0.0003, never updated
   - Optimizer AdamW momentum buffers from 14.2M bootstrap positions → wrong gradient directions on selfplay data

2. **w_sf_wdl corrupts main WDL head** (secondary issue):
   - `soft_cross_entropy(outputs["wdl"], sf_wdl_probs)` trains the MAIN WDL head (what MCTS uses)
   - SF correctly evaluates positions as bad for network (bootstrap plays weakly)
   - For DRAW games: sf_wdl mean = [W=0.361, D=0.154, L=0.486] — 48.5% predict Loss
   - Trains value head "draws = losing" → MCTS gets negative Q → worse moves → spiral
   - But this alone didn't explain the decline (trials with w_sf_wdl≈0.02 also declined)

**Fixes applied for Run 6**:
- trainable.py + run.py: bootstrap loading now only restores MODEL WEIGHTS — fresh optimizer, scheduler, step=0
- w_sf_wdl set to 0.0 (disabled). sf_eval head (via w_sf_eval) still learns SF evals independently
- Removed w_sf_wdl from PB2 search. Now 4 params: lr, w_soft, w_future, w_sf_move

---

## Run 6: Fresh optimizer + disabled sf_wdl (2026-03-01)

**Config changes**:
- Bootstrap loading: model weights only (no optimizer/scheduler/step)
- w_sf_wdl: 0.0 (disabled — was corrupting main WDL head)
- PB2 searches 4 params: lr [5e-4, 3e-3], w_soft [0.0, 1.0], w_future [0.0, 0.3], w_sf_move [0.0, 0.3]
- All other settings unchanged from Run 5

**Rationale**: With fresh optimizer state, the warmup (400 steps) will ramp lr from 0 to PB2's assigned peak, and cosine annealing starts from cycle 0. PB2's lr perturbation will actually take effect since the scheduler is initialized with the trainable's config lr (not bootstrap's). Disabling w_sf_wdl removes the contradictory value-head signal.

---

## Known Issues / TODO

- **Gate check design flaw**: Plays model vs Stockfish instead of new model vs old model. Also uses 1-sim while selfplay uses 64-sim. Needs redesign before re-enabling.
- **PB2 clone diversity**: When multiple trials are exploited simultaneously, they can all get identical configs. May need custom explore_fn to add noise.
- **PID state on clone**: Mechanically transfers via pid_state.json in checkpoint. Verified it IS saved/restored correctly.
- **opponent_strength metric**: Heavily dependent on PID dynamics. When PID is stuck at random_move_prob=1.0, opponent_strength just decays with EMA — doesn't differentiate trials well.
