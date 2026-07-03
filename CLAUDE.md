# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Chess anti-engine training framework — trains a transformer neural network to exploit Stockfish weaknesses (fortress blindness, horizon effects, closed-position overconfidence). Targets CUDA GPUs (primarily RTX 5090, but supports any CUDA device).

## Commands

```bash
pip install -e ".[dev]"     # Full server/test environment — includes the
                             # server, train, tune, and onnx extras; the test
                             # suite hard-requires all of them (no skips)
pip install -e ".[worker]"  # Lite selfplay-client install (core deps + requests)

python -m pytest            # Run all tests
python -m pytest tests/test_transformer_forward.py  # Run single test file

# Training (distributed selfplay with PBT hyperparameter search)
PYTHONPATH=. python3 -m chess_anti_engine.run --config configs/pbt2_small.yaml --mode tune
PYTHONPATH=. python3 -m chess_anti_engine.run --config configs/pbt2_small.yaml --mode tune --resume

# Single distributed trial (no PBT; still starts local server + worker)
PYTHONPATH=. python3 -m chess_anti_engine.run --config configs/default.yaml --mode train
```

Current CLI modes are `train`, `tune`, and `salvage`; there is no `single` mode.

## Operations

Use `scripts/train.sh` to drive training; it manages the PID file, log, and Ray cleanup.

```bash
./scripts/train.sh start                   # auto-resumes if $WORK_DIR/tune state exists; else fresh
./scripts/train.sh start --fresh           # force a fresh run (ignore prior tune state)
./scripts/train.sh stop                    # SIGTERM + ray stop + orphan worker sweep
./scripts/train.sh restart                 # stop + start (auto-resume same as start)
./scripts/train.sh status | log            # status / tail -f log
```

`start` auto-passes `--resume` when `$WORK_DIR/tune/experiment_state-*.json` exists. Without that behavior, restarting after a stop silently drops the running trial and spawns a random-init one. If you want to abandon the current trial's state, either pass `--fresh` or use `salvage-restart` from a good pool; never `rm` the tune dir while a run is live.

**Graceful pause before killing PBT**: `python3 scripts/graceful_restart.py` creates `pause.txt` in the tune dir; active trials finish the current iteration, hold, and then the script restarts cleanly. Useful before a restart that would otherwise orphan a mid-iteration trial.

**Salvage** (warm-start fresh trials from past checkpoints + replay):

```bash
./scripts/train.sh salvage-export --top-n 3 [--out DIR] [--metric KEY]
#   → data/salvage/<run-id>_<ts>/{manifest.json, seeds/slot_NNN/{trainer.pt,pid_state.json,replay_shards/}}

./scripts/train.sh salvage-restart data/salvage_iter37
#   stops, then starts with the pool activated via CLI flags.
#   Defaults: restore PID + full trainer state, keep GPBT-sampled LR, don't reinit volatility.
#   Toggles: --no-pid, --no-optimizer, --reinit-volatility, --donor-config.
```

Salvage is driven entirely by CLI flags (`--salvage-seed-pool-dir`, `--salvage-restore-*`), so you don't need to edit `configs/pbt2_small.yaml` to activate or disable it. When to salvage: after a bad exploit, a training run that regressed, or to rebase onto a better-regret checkpoint. A pool is a one-shot seed — once trials are past startup it plays no further role.

**Operational gotchas:**

- After pulling changes to `.c`/`.h` files, rebuild in place: `python3 setup.py build_ext --inplace` (NOT `pip install -e .` — the .venv setuptools lacks PEP 660).
- Scripts that import the C extensions run with `PYTHONPATH=.` (either python works since the 2026-07-02 numpy-2 rebuild — extensions built with numpy-2 headers import under both `/usr/bin/python3` and `.venv`). After a numpy upgrade, `build_ext --inplace` silently reuses stale cached `.so`s — use `--force`.
- NEVER run a 256+ sim arena concurrent with training (GPU OOM crashed the live run 2026-06-18). sims-1/32 arenas and `audit_targets`/`value_regret` at small batch + `--gpu-mem-fraction` are safe concurrent.
- The live YAML is re-read every iteration, and the strict validator rejects the WHOLE reload if it contains a key the running code doesn't know — add new config keys only after restarting onto code that defines them.
- Live checkpoints get pruned by Ray; before using one as a long-lived reference (arena/audit baseline), copy it out of the tune dir first.
- **Never `git checkout` in this working tree while a run is live** — the live YAML is part of the tree and is re-read every iteration, so a branch switch can silently revert live experiments (2026-07-02: a checkout rolled back the label uncap, fast-ply revert, and rung-1 blend for 3 iterations). Do all branch work in `git worktree` checkouts, and merge PRs that touch the live yaml promptly.

## Configs

- `configs/pbt2_small.yaml` — **Production config.** 384-dim, 12-layer model (~46M params — per-layer Smolgen + 4 policy heads dominate the count). Distributed selfplay with shared inference broker, PID difficulty controller, PBT/GPBT hyperparameter search. All active training uses this.
- `configs/default.yaml` — Reference config with BT3-scale model (768-dim, 15-layer, ~105M params). For future larger-model training.

## Experiment protocol — MANDATORY steps

**`docs/experiment_ledger.md` is the canonical experiment record** (verdicts:
WORKED / FAILED / LIVE-UNREAD, yardstick anchors, revert points). Read it before
proposing, launching, judging, or reverting ANY experiment. These steps are not
optional and apply to every assistant/model working in this repo:

1. **Before a training-affecting change goes live** (config key, loss weight,
   data-pipeline or selfplay change, PR merge that alters training): add a
   ledger entry with the hypothesis, ONE deciding yardstick (exact command),
   and a pre-committed kill/success threshold. No entry → don't launch.
2. **Before big changes**: snapshot weights + optimizer + PID + replay window:
   `./scripts/train.sh salvage-export --top-n 1 --metric training_iteration --out data/salvage/<label>`
   (safe while training runs, ~2.3G; `--metric training_iteration` is REQUIRED —
   the default metric picks the best-metric row, not the current state) and
   record it in the ledger's Revert points table. A yaml revert alone is NOT a rollback — the replay window
   keeps ~a day of data made under the old settings.
3. **After a readout**: record the verdict in the ledger the same session,
   judged by the pre-committed rule (not post-hoc reading). "Deferred" is not
   a verdict.
4. **One data-affecting change per readout window.** Unavoidable overlaps go
   in each entry's Confounds line.
5. **Before running any yardstick**: read the ledger's "Protocol gotchas"
   (e.g. `audit_targets` needs `--max-positions 2000`; live dashboard signals
   are flat by design; the PID winrate sample is congestion-biased).

## Evaluation & experiments

`docs/eval_protocol.md` is the decision protocol. The audit-first rule: every
training-target candidate is scored against the frozen deep-SF audit set
BEFORE any training compute — a candidate that loses the direct audit is
killed without training. Tooling:

- `scripts/arena_standard.py` — standardized paired-opening arena
  (pentanomial Elo + 95% CI, JSONL log with git SHA). `matched_sims` for
  search/target changes; `matched_time` only when the change ships in the
  fast C path (Python-path features would be under-credited).
- `scripts/build_audit_set.py` / `scripts/audit_targets.py` — frozen
  deep-SF audit set (>=1M nodes, MultiPV >=10, side-to-move canonical) and
  the direct scorer (expected/top-1 deep-SF regret per phase+source; WDL
  Brier/ECE). The set FREEZES after generation; new sampling = new version.
- `scripts/value_regret.py` — value-head 1-ply deep-SF regret: the VALUE
  yardstick (ranking strength, directly comparable to the policy regret;
  Brier/ECE are fooled by calibration and must not be used to judge value
  strength).
- `scripts/probe_policy_targets.py` — policy vs soft-policy divergence
  probe. `scripts/retarget_retrain.py` — offline SF-target retuning from
  the sparse MultiPV shard labels, no live run needed.
- `scripts/convert_shards_v2_threats.py` — offline v1→v2_threats shard
  converter (recomputes the 29 threat planes from stored input planes;
  idempotent, atomic per shard). On-the-fly twin:
  `train.replay_upgrade_v1_planes`.
- Flag-gated research bets live in `configs/exp_*.yaml` (each header has
  the sweep plan + kill threshold); ALL flags default off and only enter
  `configs/pbt2_small.yaml` once promoted. Promoted to production:
  v2_threats planes (2026-06-17), `train_views_per_position` (2026-07-01 — the
  one surviving leg of the throughput triple; `sf_label_nodes_cap` and
  `record_fast_ply_value` were promoted with it and KILLED 2026-07-02, see
  docs/experiment_ledger.md; `configs/exp_throughput_views.yaml` holds the
  isolated variant + kill thresholds). Open bets: sparse SF-policy CE (`train.sf_policy_sparse_ce`),
  soft-policy ablation (`train.soft_policy_min_tv`), volatility-aware Gumbel
  search (`selfplay.volatility_*`, Python search path only). Shelved:
  dynamic board-relation bias (offline ≡ threat planes).

## Architecture

**Data flow per iteration:** distributed selfplay (MCTS games vs Stockfish) → shard upload → ingest into disk-backed replay buffer → training step → checkpoint → publish model to workers.

### Input Encoding (`encoding/`)
Production input is **175 planes** (`input_extra_features: v2_threats`): 112 LC0 history planes + 34 classical feature planes (king safety, pins/xrays, pawn structure, mobility, outposts) + 29 threat planes. Legacy `v1` = 146 planes (no threat planes). `encode_position()` is the main entry point. C extension `_lc0_ext` provides `CBoard` for fast board operations (push, encode, legal moves).

### Model (`model/`)
Transformer encoder-only backbone (`ChessNet` in `transformer.py`). BT4-aligned architecture with Smolgen attention bias, gating, configurable embed dim/layers/heads. Multi-task output heads — each head and its training target (see `train/losses.py`):

Policy heads emit policy-size logits — **lc0_1858 compact in production**, legacy az_4672 (see Move Encoding).

| Head output | Shape | Training target | Target source | Loss | Weight knob |
|---|---|---|---|---|---|
| `policy` / `policy_own` | policy logits | `policy_t` (soft) | Gumbel completed-Q **improved policy** over all legal moves (`rec.policy_probs` = softmax(log prior + σ(completed Q)) at the searched root — the paper's recommended target, NOT raw visit counts). Move-selection temperature affects only the played move, not this target. | CE, legal-masked | `w_policy` |
| `policy_soft` | policy logits | `policy_soft_t` (soft) | Same improved policy as `policy_t`, retempered via `apply_policy_temperature(soft_policy_temp)` (typically softer) | CE, legal-masked | `w_soft` |
| `policy_future` | policy logits | `future_policy_t` (soft) | The t+2 record's `policy_probs` — improved policy at position t+2 (predict-own-reply) | CE, **no** mask | `w_future` |
| `policy_sf` | policy logits | `sf_policy_t` (soft) | Softmax over SF's MultiPV candidate WDL scores + label smoothing. SF labels are queried at **P1** (after the net's move), so this is the **opponent's reply distribution**, NOT a move-teacher for the sample's own position — the `sf_p0_*` fields (one-ply shift, selfplay rows) provide that. `sf_move_index` is the stored bestmove pointer, used only by the `sf_move_acc` metric. | CE (soft), no mask | `w_sf_move` |
| `wdl` | 3 logits | three-way soft blend | `game_frac`·game outcome (one-hot) + `sf_wdl_frac`·SF eval + `search_wdl_frac`·own MCTS root WDL (fracs live-tunable; ≈0.30/0.35/0.35 at low regret). `sf_wdl` is a **cp-logistic** label (`sf_wdl_use_cp_logistic`, slope/draw-width in config), soft by construction, not SF's native near-one-hot WDL. See the blend note below | soft CE on the blend | `w_wdl` + the three `*_frac` knobs |
| `sf_eval` | 3 logits | `sf_wdl` (soft) | SF's WDL eval only (auxiliary, **not** used in MCTS) | Soft CE | `w_sf_eval` |
| `categorical` | 32 logits | `categorical_t` (HL-Gauss) | Game outcome as 32-bin Gaussian distribution (distributional value) | CE | `w_categorical` |
| `volatility` | N scalars | `volatility_t` | Net-derived position volatility signal | Huber (δ=0.1) | `w_volatility` |
| `sf_volatility` | N scalars | `sf_volatility_t` | SF-derived position volatility signal | Huber (δ=0.1) | `w_sf_volatility` |
| `moves_left` | 1 scalar | `moves_left` | Plies remaining in the game | smooth L1 | `w_moves_left` |

Implementation details:
- Each of the 4 policy heads is a separate `AttentionPolicyHead` (Q/K/underpromo projections) that shares the trunk. Uses `Q@K^T` with `1/√d` scaling plus a learnable `log_temp` scalar per head (added Apr 2026 — the `1/√d` scale alone squashed output sharpness below what MCTS targets required).
- `wdl` is the ONLY value head used in MCTS search. `sf_eval` and `categorical` are auxiliary supervision signals that share the trunk but don't feed the search.
- Rows are recorded per net turn with `has_policy` ⇔ full-sim ply. Only the MAIN policy CE masks by `has_policy`; the aux policy heads mask by their own presence flags, so value-only rows must have those targets cleared at finalize (they are — see `_build_replay_samples`).

`TinyNet` in `tiny.py` is a small reference model for testing.

### Move Encoding (`moves/`)
4672-plane LC0 policy encoding mapping (square, direction) pairs to policy indices. Production uses the compact `lc0_1858` policy encoding; conversion maps live in `moves/encode.py` and the shared device-cached torch lookup tensors in `moves/torch_maps.py` (use these — don't add per-module `lru_cache` copies).

### MCTS (`mcts/`)
Gumbel MCTS with sequential halving (primary) and PUCT (legacy). C-accelerated tree operations in `_mcts_tree.c` — fused tree traversal + CBoard replay + encoding. `gumbel_c.py` orchestrates the simulation loop with GPU inference pipelining.

### Selfplay (`selfplay/`)
`play_batch()` orchestrates games (mostly net-vs-net "selfplay" slots plus a curriculum fraction vs the PID-handicapped SF opponent). Network turns use MCTS + temperature sampling; ~75% of plies are playout-capped "fast" plies (`playout_cap_fraction`/`fast_simulations`), the rest full-sim. `has_policy` ⇔ full ply; `selfplay.record_fast_ply_value` can keep fast plies as **value-only rows** (KataGo playout-cap harvest, ~4× rows/game) — tried in production 2026-07-01→02 and REVERTED (trunk dilution + value regression; see docs/experiment_ledger.md). Off in production. SF labels (`sf_wdl`/`sf_policy`) attach only to full plies, are queried at P1 (after the net's move) and POV-flipped to the sample; `stockfish.sf_label_nodes_cap` caps label-query nodes independently of the PID opponent budget.

### Distributed Selfplay (`tune/distributed_runtime.py`, `server/`)
Workers run as separate processes, each playing game batches via shared inference broker. Broker (`inference.py: SlotBroker/SharedSlotBroker`) uses pre-allocated shared memory slots with pinned CPU buffers for zero-copy GPU transfer. Workers upload shard files to server inbox; trainable ingests them into the replay buffer each iteration.

### PID Difficulty Controller (`stockfish/pid.py`)
Adaptive opponent strength via WDL regret-based difficulty. PID controller targets `sf_pid_target_winrate` (see `configs/pbt2_small.yaml`) by adjusting `wdl_regret` and SF node count.

**How regret works:** SF selects randomly among all moves whose WDL loss vs the best move is within `wdl_regret`. Higher regret = wider pool of acceptable moves including bad ones = SF plays weaker = model wins more easily. Lower regret = SF constrained to near-optimal moves = harder. So the controller LOWERS regret to increase difficulty and RAISES it when winrate is too low (airbag). The training target is always best-move based — `policy_sf` trains on the soft distribution over SF's MultiPV candidates by WDL eval, and `sf_wdl` reflects the objective position eval — neither depends on which handicapped move SF actually chose.

### Training (`train/`)
`Trainer` class runs training steps with `torch.amp` (BF16 on CUDA). Multi-component loss computed in `losses.py`. Optimizer is configurable (`aurora` / `nadamw` / `adamw` / `cosmos` / ...); production uses **`aurora`** with `matrix_optimizer_scope: mlp_out`. Gradient clipping via z-clip (`zclip_max_norm` hard cap + z-score outlier clip). Per-iteration step budget: production uses **views-targeting** (`train_views_per_position`: trained-samples per ingested position held fixed, so steps scale with ingest volume); `train_window_fraction` is the legacy mode and the ingest-drought floor.

**The WDL blend's SF component is load-bearing — do not zero it.** The main `value_wdl` head trains on one soft-CE against a three-way blend built in `losses.py`: `game_frac`·outcome + `sf_wdl_frac`·SF cp-logistic label + `search_wdl_frac`·own search-root WDL (no separate `w_sf_wdl` knob exists — the blend replaced the old dual loss). Removing SF value supervision crashed winrate 0.64 → 0.40 in 4 iters (2026-04-17, reverted in 52ab9c0). The cp-logistic label is deliberately soft and ~calibrated to actual selfplay outcomes (verified 2026-07-01), so the head's "hedged" predictions are an accurate fit to its target, not under-fitting — the head sharpens only as the net's real conversion ability improves. Don't chase value-head sharpness against a deep-SF ruler; that comparison is a category error. `value_sf_eval` is a weak auxiliary channel, not a substitute.

### Replay Buffer (`replay/`)
Disk-backed replay buffer (`DiskReplayBuffer`) with zarr shard storage. Growing sliding window: starts small, expands as training progresses. KataGo-style surprise weighting for sampling.

### Stockfish Interface (`stockfish/`)
`StockfishUCI` for single-threaded UCI communication; `StockfishPool` for multi-worker parallel analysis.

### Configuration
YAML config provides all defaults. `utils/config_yaml.py` flattens nested YAML into a flat dict. Live YAML reload each iteration (non-topology keys only). PBT-searched keys are preserved across reloads.

### ONNX Export (`onnx/`)
Export for Ceres chess engine compatibility.

### Hyperparameter Tuning (`tune/`)
Ray Tune with GPBT (Gaussian Process Bandit PBT) scheduler. Pairwise velocity-based parameter exploration. Current production config pins everything (LR bounds collapsed to a single value, `search_optimizer/smolgen/nla: false`) — search is wired up but effectively off. Exploit copies model + optimizer + replay from donor trial.

## Code Conventions

- Python 3.10+, uses `from __future__ import annotations` throughout
- C extensions in `encoding/_lc0_ext.c` and `mcts/_mcts_tree.c`
- Type hints on functions and dataclasses
- Tests in `tests/`
- PYTHONPATH=. required for scripts

### Static analysis

Run `./scripts/lint.sh <paths>` after editing. The default GATE is **ruff + basedpyright + vulture** (wall time ≈ basedpyright, a few seconds) and is kept at **zero findings repo-wide — no baseline** (the old `.basedpyright/baseline.json` was burned down to zero and deleted 2026-07-02; CI gates basedpyright on the whole repo). pylint was removed 2026-07-02: its checks migrated to ruff rules (`ARG`/`B006`/`G004`/`SIM115`, see `[tool.ruff.lint]`) and basedpyright's flow checks.

```bash
./scripts/lint.sh chess_anti_engine/train/trainer.py   # specific files
./scripts/lint.sh --changed                            # changed and untracked .py files
./scripts/lint.sh --deep [paths...]  # + skylos + ruff cleanup report (advisory, ~40s —
                                     #   run before a cleanup pass, not per edit)
```

The `--deep` ruff report sweeps the not-yet-gated groups (B, NPY) as a cleanup shopping list — after the 2026-07-02 burn-down only the needs-judgment tail remains there (B905 `zip strict=`, B008 call-in-default, NPY002 legacy-RNG — never autofix; NPY002 changes RNG streams). Promote a rule into the pyproject gate once it reaches zero findings.

Configs:
- `pyproject.toml`: `[tool.ruff.lint]` (gate rule set + rationale), `[tool.vulture]`
- `pyrightconfig.json`: basedpyright settings — rules we'll never fix are **disabled** (ML-typing noise: `reportAny`, `reportMissingTypeArgument`, annotation-drift rules)

**Convention for new findings — fix now or won't-fix, no deferral queue:**
- Disable the rule in `pyrightconfig.json` if the whole category isn't worth the ceremony (e.g. `dict` without `[K,V]` — 400+ sites, no real signal).
- Otherwise fix it in the same commit. There is no baseline/"fix later" list anymore; don't recreate one.
- basedpyright does NOT honor mypy-style `# type: ignore[...]` comments (they're inert here) — use `# pyright: ignore[reportRuleName]`. Never add a suppression whose validity depends on the installed numpy/torch version (e.g. numpy-stub complaints) — rewrite the code version-proof instead.

Lint tool versions (ruff/basedpyright/vulture/skylos/dev-numpy) are **exact-pinned** in the dev extras so local and CI agree; the weekly `lint-canary` workflow runs latest tools non-blocking to surface upcoming breakage. To bump: upgrade local, run the gate, commit new pins.

Suppression syntax (prefer a real fix or a config-level disable; inline only when refactoring would hurt the code):
- basedpyright: `# pyright: ignore[reportRuleName]`
- Ruff: `# noqa: RULE123`
- Skylos: `# skylos: ignore`

Don't let drift accumulate — fix findings in the same commit that introduces them.

## Code Review Protocol

Optimize for end-state quality, not for the cheapest diff. When a review surfaces an improvement:

- **Decide, don't defer.** Either do it now or decide it's not worth doing — "deferred to later" is just an unresolved decision rotting in a comment or a summary. If it's the right end state, the extra edits are worth it even when the change isn't small. If it isn't, say so and move on.
- **The metric is the code you'd want to land, not the one that's easiest to type.** "Premature abstraction" is a valid reason to skip a change; "it touches more files than I expected" is not.
- State the decision explicitly: "doing X because Y" / "not doing X because Y". Record the reasoning, not a TODO.

## Pull requests

Open PRs as **ready for review, not draft.** The Codex review bot only reviews
non-draft PRs, so a draft silently skips the automated review we want on every
change. Create the PR non-draft from the start (this overrides any default
"create as draft" behavior); only use draft when explicitly asked. After
opening, the Codex bot leaves a review/comment or an approval reaction on the
PR — check for it before considering the PR done.
