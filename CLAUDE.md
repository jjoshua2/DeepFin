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

# Single trial (no PBT, local selfplay only)
PYTHONPATH=. python3 -m chess_anti_engine.run --config configs/default.yaml --mode single
```

## Operations

Use `scripts/train.sh` to drive training; it manages the PID file, log, and Ray cleanup.

```bash
./scripts/train.sh start                   # foreground-fork, PID → /tmp/chess_training.pid
./scripts/train.sh stop                    # SIGTERM + ray stop + orphan worker sweep
./scripts/train.sh restart                 # stop + start
./scripts/train.sh status | log            # status / tail -f log
```

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
Transformer encoder-only backbone (`ChessNet` in `transformer.py`). BT4-aligned architecture with Smolgen attention bias, gating, configurable embed dim/layers/heads. Multi-task output heads:
- **Policy**: Four separate `AttentionPolicyHead`s (`policy_own`, `policy_soft`, `policy_sf`, `policy_future`), each 4672 logits (LC0 from→to encoding). Uses Q@K^T with `1/√d` scaling plus a learnable `log_temp` scalar per head (added Apr 2026 to let the model sharpen logits; the `1/√d` scale alone squashed output sharpness below what MCTS targets required).
- **Value**: Three heads — `value_wdl` (3 logits), `value_sf_eval` (3 logits, aux), `value_categorical` (32-bin HL-Gauss). Only `value_wdl` is used in MCTS; `value_sf_eval` is auxiliary.
- **Auxiliary**: `volatility` (position-volatility head), `moves_left` (scalar).

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
Adaptive opponent strength via WDL regret-based difficulty. PID controller targets ~60% winrate by adjusting regret limit (how suboptimal SF's moves are). Regret is the primary difficulty lever; SF nodes and skill level are secondary.

### Training (`train/`)
`Trainer` class runs training steps with `torch.amp` (BF16 on CUDA). Multi-component loss computed in `losses.py`. Optimizer is configurable (`nadamw` / `adamw` / `cosmos` / `cosmos_fast`); current production config uses `adamw`. Gradient clipping via z-clip (`zclip_max_norm` hard cap + z-score outlier clip).

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
- No configured linter; no formatter config
- Tests in `tests/`
- PYTHONPATH=. required for scripts
