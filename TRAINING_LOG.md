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

## Run 7: Narrowed bounds + PB2 checkpoint fix (2026-03-02 09:31, ~10 hours)

**Config changes**:
- PB2 searches 5 params: lr [8e-4, 2.5e-3], w_soft [0.5, 1.0], w_future [0.0, 0.3], w_sf_move [0.0, 0.05], soft_policy_temp [2.0, 4.0]
- trainable.py: work_dir = trial_dir (fix PB2 checkpoint cloning)
- soft_policy_temp made configurable (was hardcoded 2.0)

**Result**: 10 trials ran 13 iterations. 4/9 original trials OK (>50% WR), 3 BAD (<35%), 2 MID. T09 was cloned from T02 at iter 8 (only perturbation that fired).

**PB2 perturbation analysis**:
- PB2 DID fire at iter 8, but only T09 was exploited (async race: lower-quantile trials that reported before upper-quantile trials had saved checkpoints got silently skipped)
- Explore step produced ZERO perturbation: GP returned identical config, no parameter changes
- T09's dramatic recovery (WR 0%→78%) was just T02's checkpoint restarting with fresh replay

**Winners vs Losers** (excluding T09 clone):
- OK (T02,T07,T08): wdl_loss stays >0.86, w_sf_move 0.024-0.050
- BAD (T00,T01,T04): wdl_loss drops <0.81 (overconfident value head causes regression)
- LR doesn't explain regression: T01 BAD at lr=0.0023, T02 OK at same lr=0.0023
- No single parameter cleanly separates winners from losers — failure is stochastic

**Learnings**:
- w_sf_move in [0.02, 0.05] range works (was thought to want 0)
- Value head overconfidence spiral still the failure mode but not param-driven
- 1000 games/iter too slow — PB2 needs more frequent perturbation to accumulate GP data
- Async PBT unreliable for cloning (checkpoint race condition), but synch would waste compute

---

## Run 8: 4x shorter iterations for faster PB2 (2026-03-02 20:17)

**Config changes**:
- games_per_iter: 1000 → 250 (4x shorter)
- train_steps: 100 → 25 (4x shorter)
- warmup_steps: 400 → 100 (proportional)
- lr_T0: 680 → 200 (aligned with 8 iters × 25 steps)
- w_sf_move bounds: [0.0, 0.05] → [0.02, 0.05] (narrowed from Run 7 winners)
- pid_min_games_between_adjust: 100 → 250 (1 full iter)
- replay_window_growth: 1000 → 250 (proportional)

**Rationale**: 4x shorter iterations means PB2 reaches perturbation_interval=8 in 1/4 the wall time. More frequent perturbation → more GP data points → better explore, and more chances for async cloning to succeed.

---

## Known Issues / TODO

- **Gate check design flaw**: Plays model vs Stockfish instead of new model vs old model. Also uses 1-sim while selfplay uses 64-sim. Needs redesign before re-enabling.
- **PB2 clone diversity**: When multiple trials are exploited simultaneously, they can all get identical configs. May need custom explore_fn to add noise.
- **PID state on clone**: Mechanically transfers via pid_state.json in checkpoint. Verified it IS saved/restored correctly.
- **opponent_strength metric**: Heavily dependent on PID dynamics. When PID is stuck at random_move_prob=1.0, opponent_strength just decays with EMA — doesn't differentiate trials well.

---

## Run 9: PB2 exploit replay/RNG fixes + no iter-0 skip (2026-03-03 10:38)

**Code changes applied**:
- `trainable.py`: removed iter-0 selfplay skip entirely. Every iteration now plays fresh selfplay games.
- `trainable.py`: checkpoint now saves `rng_state.json` and `trial_meta.json`.
- `trainable.py`: restore behavior now branches:
  - same-trial resume: restore RNG state from checkpoint
  - cross-trial PB2 exploit: fork RNG seed using recipient trial id + restore salt to avoid replaying donor RNG stream/opening sequence
- `trainable.py`: on cross-trial exploit, refresh replay shards before buffer init:
  - keep newest 60% of local shards (drop oldest 40%)
  - copy 6 recent donor shards (skip donor newest 1 for write-safety)
- `config_yaml.py` + `run.py`: added YAML/CLI plumbing for exploit replay refresh knobs so they are explicit run config, not hardcoded-only defaults.
- `configs/pbt2_small.yaml`: explicit tune knobs added:
  - `exploit_replay_refresh_enabled: true`
  - `exploit_replay_keep_fraction: 0.60`
  - `exploit_replay_donor_shards: 6`
  - `exploit_replay_skip_newest: 1`

**Restart**:
- Stopped prior run process.
- Started a **fresh non-resume** tune run at `2026-03-03 10:38`.
- New trial prefix: `0402b`.

**Expected effects to validate**:
- No synthetic `win=draw=loss=0` rows at perturb boundaries.
- Post-exploit recipients should train on mixed recent+donor data, not stale local-only data.
- Less opening-sequence lockstep across exploited clones.

---

## Run 10: Value-head weight range retune + fresh restart (2026-03-03 21:33)

**Config changes**:
- `pb2_bounds_w_wdl`: `[0.5, 1.0]` → `[0.3, 0.8]`
- `pb2_bounds_w_sf_wdl`: `[0.0, 0.05]` (unchanged, kept low)
- Iteration pacing remains short/frequent:
  - `games_per_iter: 150`
  - `train_steps: 15`
  - `replay_window_growth: 150`
  - `sf_pid_min_games_between_adjust: 150`
- `w_future` remains pinned at `0.05` (not searched)

**Restart**:
- Stopped the prior tune run.
- Started a **fresh non-resume** run at `2026-03-03 21:33`.
- New trial prefix: `7773d`.

**Intent**:
- Keep SF-WDL auxiliary pressure minimal.
- Search main WDL-head weight in a lower band to reduce value-head overconfidence collapse while preserving enough value signal.

---

## Run 11: Salvage restart integrity fix (2026-03-05 10:14)

**Issue observed**:
- Salvage warmstarts loaded top checkpoints, but most trials still started with `random_move_prob=1.0` due stale `pid_state.json` in seed slots.
- TensorBoard confusion came from using stale path: `/tmp/ray/session_latest` still pointed to older session.
- Also detected two concurrent tune processes competing for resources (`1877824`, `2399046`).

**Code changes applied**:
- `trainable.py`: warmstart now reads salvage `manifest.json` entry for selected slot and merges PID fields from `result_row` (`random_move_prob_next`, `sf_nodes_next`, `skill_level_next`, `pid_ema_winrate`) as authoritative fallback.
- `trainable.py`: added TB scalars for `difficulty/opponent_strength`, `difficulty/random_move_prob`, `difficulty/random_move_prob_next`, `meta/salvage_warmstart_used`, `meta/salvage_warmstart_slot`.

**Restart actions**:
- Force-stopped both concurrent runs and Ray workers.
- Started a single fresh run:
  - `python3 -m chess_anti_engine.run --config configs/pbt2_small.yaml --salvage-seed-pool-dir /home/josh/projects/chess/runs/pbt2_small/salvage/7773d_20260305_083208`
  - New trial prefix: `09543` (8 trials).
  - Active TB root: `/tmp/ray/session_2026-03-05_10-14-39_577500_2427274/.../driver_artifacts`

**Verification**:
- All 8 workers logged:
  - `salvage warmstart loaded slot=...`
  - `salvage PID overrides from manifest row: random_move_prob<-random_move_prob_next, ...`
- Confirms patched difficulty carryover is active on this run.

---

## Run 12: Fresh 384d GPBT restarts and early-collapse diagnosis (2026-03-06 to 2026-03-07)

**Goal**:
- Start a fresh 384d run (not salvage) and understand why trials often opened around 55-60% raw winrate, then collapsed below 40% by iteration 2-3 against nearly random-move Stockfish.

**What was tried**:
- Switched the small-model tune path from PB2 to `gpbt_pl`.
- Narrowed the active search to:
  - `lr: [1.0e-3, 1.6e-3]`
  - `w_wdl: [0.45, 0.65]`
- Tightened PID to slow difficulty movement:
  - `pid_max_rand_step: 0.005`
  - `pid_ema_alpha: 0.20`
  - `pid_random_move_prob_start: 0.995`
- Increased effective batch gently:
  - `batch_size=256`, `accum_steps=2`, later `accum_steps=4`
- Disabled `w_sf_wdl` again to test whether SF-WDL auxiliary was the main failure source.

**Important code/logging fixes found along the way**:
- `run.py`: Tune config was not forwarding `sf_pid_random_move_prob_start`, `sf_pid_random_move_prob_min/max`, `sf_pid_random_move_stage_end`, or `sf_pid_max_rand_step`, so fresh tune runs silently fell back to `random_move_prob=1.0`.
- `trainable.py`: Tune TensorBoard difficulty metrics were logging on mixed x-axes (`trainer_step_now` vs Tune iteration). Patched custom difficulty/Tune metrics to use iteration consistently.
- `trainable.py`: step-count math now uses `batch_size * accum_steps` when estimating one pass over fresh data, so increasing accumulation no longer silently increases total data passes.

**Findings**:
- PID was **not** the main reason for the collapse. Early bad trials often fell below 40% raw winrate while still at `random_move_prob=1.0` or `0.995`.
- Warmup length also was **not** the main problem. Even at only ~3 optimizer steps/iteration (`accum_steps=4`), some trials still collapsed early.
- Disabling `w_sf_wdl` did **not** fix the core failure pattern.
- The most likely cause was bootstrap initialization mismatch:
  - bootstrap training had effectively learned **value/trunk only**
  - policy heads (`policy_own`, `policy_soft`, `policy_sf`, `policy_future`) were left at random initialization
  - random policy logits plus a trained trunk created accidental, inconsistent search priors
  - some trials benefited from that random prior at iteration 1, then destabilized once early learning started

**Conclusion**:
- The early 384d failure mode was not mainly “difficulty moved too fast”.
- The stronger hypothesis was: **bootstrap trunk/value + random policy heads** gives unstable early search/training behavior.

---

## Run 13: Bootstrap policy-head reset fixed the early collapse (2026-03-07)

**Code changes applied**:
- Added a helper to explicitly zero policy-head parameters after bootstrap load:
  - `policy_own`
  - `policy_soft`
  - `policy_sf`
  - `policy_future`
- Added config flag:
  - `bootstrap_zero_policy_heads: true`
- Applied this in both the tune startup/bootstrap load path and the single-run path.

**Rationale**:
- Bootstrap data had not meaningfully trained the policy heads.
- Zeroing these heads makes the initial legal-move prior close to uniform instead of arbitrary random logits from an untrained readout on a trained trunk.

**Observed result**:
- Iteration 1 looked slightly weaker than some earlier fresh runs (expected: the accidental random prior was removed).
- But iteration 2-3 behavior improved dramatically:
  - previous runs often dropped to ~0.40 raw WR or worse by iteration 2-3
  - with zeroed policy heads, the floor stayed around ~0.49-0.54 instead of collapsing
  - population median stayed healthy (~0.56-0.58), and trials were broadly learning instead of diverging immediately

**Interpretation**:
- This was the first change that clearly attacked the real problem.
- The bootstrap checkpoint should be treated as:
  - useful trunk/value initialization
  - **not** trustworthy policy initialization

**Operational note**:
- After this fix, GPBT perturb/exploit behavior started looking reasonable instead of just recycling already-broken trials.

---

## Run 14: Central-server distributed Tune selfplay (2026-03-07)

**Goal**:
- Replace the in-process Tune selfplay loop with real multiprocess worker fanout so CPU can scale past what a single trial actor can drive.

**Architecture changes**:
- Added trial-aware namespaces to the distributed stack:
  - `server/app.py`: `/v1/trials/<trial_id>/...`
  - `worker.py`: `--trial-id`
  - `learner.py`: `--trial-id`
- `tune/harness.py` now auto-starts one central server for Tune distributed runs and provisions worker auth.
- `tune/trainable.py` can now:
  - publish trial-scoped manifests/models
  - launch real `worker.py` subprocesses per trial
  - ingest uploaded shards from the trial inbox
  - track stale positions/games from older model SHAs
- Added tune-side distributed config knobs:
  - `distributed_workers_per_trial`
  - `distributed_worker_sf_workers`
  - `distributed_worker_poll_seconds`
  - `distributed_worker_device`
  - `distributed_worker_auto_tune`
  - `distributed_worker_target_batch_seconds`
  - `distributed_worker_min_games_per_batch`
  - `distributed_worker_max_games_per_batch`
  - `distributed_server_port`

**Validation**:
- 1-trial smoke test succeeded: worker downloaded manifest/model/book, uploaded shard, and trial reported `distributed_selfplay=1`.
- This replaced the earlier failed threaded `selfplay_pipelines` experiment, which worked functionally but did not improve throughput.

**Throughput experiments**:
- Fixed layout first tested:
  - `8` trials
  - `3` workers/trial
  - `1` SF process/worker
- Measured by fresh `positions/sec`, total `positions/sec`, and stale spillover.

**Batch-size results**:
- `selfplay_batch=8`
  - fresher data
  - too slow (`~20.8 fresh pos/s`, `~25.2 total pos/s`)
- `selfplay_batch=12`
  - better balance (`~29.9 fresh pos/s`, `~37.5 total pos/s`)
- `selfplay_batch=16`
  - fastest of the three (`~33.3 fresh pos/s`, `~42.9 total pos/s`)
  - but more stale spillover (~22.5% stale vs ~20.0% at batch 12)

**Important caveat found**:
- Distributed workers were defaulting to CUDA when `distributed_worker_device` was unset.
- So `selfplay_batch` affects **VRAM**, not just CPU RAM, because:
  - each worker keeps a model on GPU
  - larger selfplay batches increase worker-side inference/MCTS memory
  - training actors also share the same GPU
- With `8` concurrent trials and CUDA workers, batch-16 runs started hitting CUDA OOM in `trainer.train_steps`.

**Current direction**:
- Keep the central-server distributed Tune path.
- Reduce total concurrent training pressure rather than only shrinking per-worker batch.
- Latest live experiment:
  - `6` GPBT trials
  - `4` workers/trial
  - `selfplay_batch=16`
  - `optimizer=muon`

---

## Run 15: Muon optimizer experiment start (2026-03-07)

**Why**:
- Once early-collapse was fixed by zeroing bootstrap policy heads, the next question became training speed/quality, not just stability.
- Wanted to test whether Muon learns faster than the existing `nadamw` setup.

**Code changes applied**:
- Added internal plain Muon implementation:
  - Muon on 2D trunk/hidden weights (`embed.weight`, transformer blocks)
  - AdamW fallback on heads, norms, and biases
- Wired `muon` through:
  - `train/trainer.py`
  - `run.py`
  - `learner.py`
  - config parsing / YAML usage
- Added regression test:
  - `tests/test_muon_optimizer.py`

**Why plain Muon first**:
- `MuonClip` and `NorMuon` exist, but they are newer/specialized variants.
- Plain Muon is the lowest-risk baseline for this codebase:
  - simpler to integrate
  - good enough to answer “does Muon help here at all?”

**Status**:
- Muon integration compiles and unit test passes.
- Live distributed Tune runs are now using `optimizer: muon`.
- Need more data before concluding whether Muon is actually better than `nadamw` for this chess setup.

---

## Run 16: Gumbel/search debugging and curriculum fixes (2026-03-10)

**Main finding**:
- The old low-sim Gumbel path was materially underusing the configured search budget.
- At `64` "simulations", the previous code typically:
  - selected a top-k root set with Gumbel noise
  - evaluated each candidate child once
  - then repeatedly halved cached Q values
- So the run was not getting anything close to a true 64-search root policy improvement signal.

**Confirmed bug fixed**:
- `run_gumbel_root_many()` had already been patched to stop returning stale root Q as `values_out`.
- Before that fix, `manager.py` saw:
  - `orig_q ~= best_q`
  - `q_surprise ~= 0`
- This effectively broke the Q branch of diff-focus and made `search_wdl_est` use stale root value instead of searched child value.

**Additional Gumbel fixes implemented**:
- Reworked `chess_anti_engine/mcts/gumbel.py` so `simulations` now drives actual subtree search under root sequential halving instead of only affecting candidate count.
- Added a full-tree Gumbel selector below the root (default on via `GumbelConfig.full_tree=True`) instead of falling back to PUCT after the forced root action.
- Fixed completed-Q perspective in the Gumbel path:
  - visited child action values must be interpreted from the parent/root perspective (`-child.Q`), not `child.Q`.
- Root policy improvement now uses completed-Q style logits over searched legal actions rather than the earlier one-pass child-eval shortcut.

**Validation**:
- `python3 -m py_compile chess_anti_engine/mcts/gumbel.py`
- `python3 -m pytest tests/test_gumbel_mcts_smoke.py tests/test_gumbel_root_many_edge_cases.py tests/test_gumbel_budget_usage.py -q`
- `python3 -m pytest tests/test_e2e_smoke.py -q -k gumbel_selfplay_smoke`
- Added regression tests for:
  - budget usage / extra forward passes with higher Gumbel simulations
  - completed-Q sign / parent-perspective action value

**Policy-target investigation**:
- Probed recent replay targets directly.
- `policy_target` looked sane but very diffuse:
  - top-1 mass around `0.074`
  - top-5 mass around `0.332`
  - support around `16` moves
- `policy_soft_target` was almost identical to `policy_target`, suggesting that `w_soft` may be mostly redundant in this regime.
- SF policy targets were much sharper and very different from the main search-improved policy.

**Curriculum fixes / clarifications**:
- Fixed the intended early opponent semantics:
  - `random_move_prob` now means true random legal moves
  - the non-random branch samples from SF `MultiPV`
- Restored strong SF labels / teacher quality:
  - `Skill Level 20`
  - `MultiPV 12`
- Mixed self-play support added:
  - `selfplay_fraction` config knob
  - self-play games train both sides
  - curriculum games still drive PID
- Added selfplay diagnostics for future restarts:
  - `avg_game_plies`
  - `timeout_rate`
  - `game_draw_rate`

**Replay / data-flow cleanup**:
- Fixed replay-window startup behavior so fresh runs honor `replay_window_start` instead of always expanding to `len(buf)`.
- Still preserve expansion on true resume or intentionally seeded replay starts (salvage/shared shards).

**Interpretation**:
- Several earlier conclusions about:
  - diff-focus Q tuning
  - low-sim Gumbel quality
  - policy-target usefulness
  should be treated cautiously because the search path itself was weaker than intended.
- After the Gumbel fixes, `diff_focus_q_weight` became meaningful again and was re-added to the GPBT search surface.
