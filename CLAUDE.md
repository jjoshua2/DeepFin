# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Chess anti-engine training framework — trains a transformer neural network to exploit Stockfish weaknesses (fortress blindness, horizon effects, closed-position overconfidence). Targets CUDA GPUs (primarily RTX 5090, but supports any CUDA device).

## Commands

```bash
pip install -e ".[dev]"     # Install package with test dependencies
                             # (.[tune] for Ray Tune, .[onnx] for ONNX export)

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

**Graceful pause before killing PBT**: `python3 scripts/graceful_restart.py --wait N` creates `pause.txt` in the tune dir; trials finish the current iteration then hold. Useful before a restart that would otherwise orphan a mid-iteration trial.

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

## Configs

- `configs/pbt2_small.yaml` — **Production config.** 384-dim, 9-layer model (~15M params). Distributed selfplay with shared inference broker, PID difficulty controller, PBT/GPBT hyperparameter search. All active training uses this.
- `configs/default.yaml` — Reference config with BT3-scale model (768-dim, 15-layer, ~105M params). For future larger-model training.

## Architecture

**Data flow per iteration:** distributed selfplay (MCTS games vs Stockfish) → shard upload → ingest into disk-backed replay buffer → training step → checkpoint → publish model to workers.

### Input Encoding (`encoding/`)
146-plane 8x8 input: 112 LC0 history planes + 34 classical feature planes (king safety, pins/xrays, pawn structure, mobility, outposts). `encode_position()` is the main entry point. C extension `_lc0_ext` provides `CBoard` for fast board operations (push, encode, legal moves).

### Model (`model/`)
Transformer encoder-only backbone (`ChessNet` in `transformer.py`). BT4-aligned architecture with Smolgen attention bias, gating, configurable embed dim/layers/heads. Multi-task output heads — each head and its training target (see `train/losses.py`):

| Head output | Shape | Training target | Target source | Loss | Weight knob |
|---|---|---|---|---|---|
| `policy` / `policy_own` | 4672 logits | `policy_t` (soft) | MCTS visit-count distribution at the move-selection temperature (`rec.policy_probs` from gumbel) | CE, legal-masked | `w_policy` |
| `policy_soft` | 4672 logits | `policy_soft_t` (soft) | Same visit distribution as `policy_t`, retempered via `apply_policy_temperature(soft_policy_temp)` (typically softer) | CE, legal-masked | `w_soft` |
| `policy_future` | 4672 logits | `future_policy_t` (soft) | The t+2 record's `policy_probs` — visit distribution at position t+2 (predict-own-reply) | CE, **no** mask | `w_future` |
| `policy_sf` | 4672 logits | `sf_policy_t` (soft) | Softmax over SF's MultiPV candidate WDL scores + label smoothing. `sf_move_index` is stored as the bestmove index but **not** used as a target — `policy_sf` trains on the soft distribution. **WDL saturation in decided positions can flatten this target**. | CE (soft), no mask | `w_sf_move` |
| `wdl` | 3 logits | `wdl_t` (hard 0=W/1=D/2=L) + `sf_wdl` (soft SF eval) | Game outcome (hard) blended with SF's WDL eval. Both target the **same** head — load-bearing, see w_sf_wdl note below | CE + soft CE | `w_wdl`, `w_sf_wdl` |
| `sf_eval` | 3 logits | `sf_wdl` (soft) | SF's WDL eval only (auxiliary, **not** used in MCTS) | Soft CE | `w_sf_eval` |
| `categorical` | 32 logits | `categorical_t` (HL-Gauss) | Game outcome as 32-bin Gaussian distribution (distributional value) | CE | `w_categorical` |
| `volatility` | N scalars | `volatility_t` | Net-derived position volatility signal | Huber (δ=0.1) | `w_volatility` |
| `sf_volatility` | N scalars | `sf_volatility_t` | SF-derived position volatility signal | Huber (δ=0.1) | `w_sf_volatility` |
| `moves_left` | 1 scalar | `moves_left` | Plies remaining in the game | smooth L1 | `w_moves_left` |

Implementation details:
- Each of the 4 policy heads is a separate `AttentionPolicyHead` (Q/K/underpromo projections) that shares the trunk. Uses `Q@K^T` with `1/√d` scaling plus a learnable `log_temp` scalar per head (added Apr 2026 — the `1/√d` scale alone squashed output sharpness below what MCTS targets required).
- `wdl` is the ONLY value head used in MCTS search. `sf_eval` and `categorical` are auxiliary supervision signals that share the trunk but don't feed the search.
- `sf_move_index` in the shard is the stored "SF's best move" pointer; training does not use it as a 1-hot target — `policy_sf` trains on the soft `sf_policy_t` distribution instead. But it IS used by the `sf_move_acc` metric (top-1 accuracy: `argmax(policy_sf) == sf_move_index`), reported on TensorBoard as `sf_move_acc` / `test_sf_move_acc`. So: train-on-soft, evaluate-top-1-against-bestmove.

`TinyNet` in `tiny.py` is a small reference model for testing.

### Move Encoding (`moves/`)
4672-plane LC0 policy encoding mapping (square, direction) pairs to policy indices.

### MCTS (`mcts/`)
Gumbel MCTS with sequential halving (primary) and PUCT (legacy). C-accelerated tree operations in `_mcts_tree.c` — fused tree traversal + CBoard replay + encoding. `gumbel_c.py` orchestrates the simulation loop with GPU inference pipelining.

### Selfplay (`selfplay/`)
`play_batch()` orchestrates games. Network turns use MCTS + temperature sampling; Stockfish turns query engine with MultiPV for soft policy targets.

### Distributed Selfplay (`tune/distributed_runtime.py`, `server/`)
Workers run as separate processes, each playing game batches via shared inference broker. Broker (`inference.py: SlotBroker/SharedSlotBroker`) uses pre-allocated shared memory slots with pinned CPU buffers for zero-copy GPU transfer. Workers upload shard files to server inbox; trainable ingests them into the replay buffer each iteration.

### PID Difficulty Controller (`stockfish/pid.py`)
Adaptive opponent strength via WDL regret-based difficulty. PID controller targets `sf_pid_target_winrate` (see `configs/pbt2_small.yaml`) by adjusting `wdl_regret` and SF node count.

**How regret works:** SF selects randomly among all moves whose WDL loss vs the best move is within `wdl_regret`. Higher regret = wider pool of acceptable moves including bad ones = SF plays weaker = model wins more easily. Lower regret = SF constrained to near-optimal moves = harder. So the controller LOWERS regret to increase difficulty and RAISES it when winrate is too low (airbag). The training target is always best-move based — `policy_sf` trains on the soft distribution over SF's MultiPV candidates by WDL eval, and `sf_wdl` reflects the objective position eval — neither depends on which handicapped move SF actually chose.

### Training (`train/`)
`Trainer` class runs training steps with `torch.amp` (BF16 on CUDA). Multi-component loss computed in `losses.py`. Optimizer is configurable (`nadamw` / `adamw` / `cosmos` / `cosmos_fast`); current production config uses `adamw`. Gradient clipping via z-clip (`zclip_max_norm` hard cap + z-score outlier clip).

**`w_sf_wdl` is load-bearing, not a conflicting dual target.** The main `value_wdl` head is trained on both the hard game-outcome CE (`w_wdl`) and soft CE against Stockfish's WDL eval (`w_sf_wdl`). The two targets disagree often (~63% of samples) because selfplay game outcomes reflect the opponent's handicap level (PID-controlled SF regret), while sf_wdl is SF's objective eval at 5k nodes. That disagreement is the *point*: sf_wdl injects search-horizon bootstrap that pure selfplay outcomes cannot carry. Zeroing `w_sf_wdl` was tried 2026-04-17 (reverted in commit 52ab9c0) — winrate crashed 0.64 → 0.40 in 4 iters. The separate `value_sf_eval` head exists as a weak auxiliary channel (`w_sf_eval`) but does not substitute for main-head SF supervision.

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

Run `./scripts/lint.sh <paths>` after editing. Default is all five tools — **ruff + basedpyright + pylint + vulture + skylos** — and the repo is kept at zero findings for ruff/pylint/vulture/skylos. basedpyright uses a baseline file (`.basedpyright/baseline.json`) to suppress the ~1200 pre-existing warnings from its stricter-than-pyright defaults; new code must be clean.

```bash
./scripts/lint.sh chess_anti_engine/train/trainer.py   # specific files
./scripts/lint.sh --changed                            # git-changed .py files since HEAD
./scripts/lint.sh --fast [paths...]                    # skip vulture + skylos (the slower ones)
basedpyright --writebaseline                           # refresh baseline after fixing a batch
```

Configs:
- `pyproject.toml`: `[tool.ruff.lint]`, `[tool.pylint.main]`, `[tool.vulture]`
- `pyrightconfig.json`: basedpyright settings — rules we'll never fix are **disabled** (ML-typing noise: `reportAny`, `reportMissingTypeArgument`, annotation-drift rules), so they don't pollute the baseline
- `.basedpyright/baseline.json`: machine-generated frozen "fix later" queue. Each item is a real signal we just haven't addressed yet

**Convention for splitting "won't fix" from "fix later":**
- Disable the rule in `pyrightconfig.json` if we've decided the whole category isn't worth the ceremony (e.g. `dict` without `[K,V]` — 400+ sites, no real signal).
- Keep the rule enabled + let baseline suppress current instances if it's a real signal and we plan to fix it eventually (e.g. `reportOptionalMemberAccess` — crash risk). New instances in edited code fail, baseline items are the todo list. Run `basedpyright --writebaseline` after a cleanup pass to shrink it.

Suppression syntax (prefer config/baseline over inline; inline only when refactoring would hurt the code):
- basedpyright: `# pyright: ignore[reportRuleName]`
- Ruff: `# noqa: E741`
- Skylos: `# skylos: ignore`

Baseline cleanup sequence used to get here: ruff (commits `b1a7cca`), pyright (`55d0dd5`), pylint (`b490310`), skylos (`9220460`), plus the basedpyright migration. Don't let drift accumulate — fix findings in the same commit that introduces them.

## Code Review Protocol

Optimize for end-state quality, not for the cheapest diff. When a review surfaces an improvement:

- **Decide, don't defer.** Either do it now or decide it's not worth doing — "deferred to later" is just an unresolved decision rotting in a comment or a summary. If it's the right end state, the extra edits are worth it even when the change isn't small. If it isn't, say so and move on.
- **The metric is the code you'd want to land, not the one that's easiest to type.** "Premature abstraction" is a valid reason to skip a change; "it touches more files than I expected" is not.
- State the decision explicitly: "doing X because Y" / "not doing X because Y". Record the reasoning, not a TODO.
