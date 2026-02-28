# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Chess anti-engine training framework — trains a transformer neural network to exploit Stockfish weaknesses (fortress blindness, horizon effects, closed-position overconfidence). Phase 1: all-Python pipeline targeting consumer GPUs (RTX 3090/4090/5090).

## Commands

```bash
pip install -e .            # Install package
pip install -e ".[dev]"     # Install with test dependencies
pip install -e ".[tune]"    # Install Ray Tune for hyperparameter search
pip install -e ".[onnx]"    # Install ONNX export support

python -m pytest            # Run all tests (quiet mode by default)
python -m pytest tests/test_transformer_forward.py  # Run single test file

# Training
python -m chess_anti_engine.run --config configs/default.yaml --mode single
python -m chess_anti_engine.run --config configs/default.yaml --mode tune
```

## Architecture

**Data flow per iteration:** selfplay (MCTS games vs Stockfish) → replay buffer → training step → checkpoint.

### Input Encoding (`encoding/`)
146-plane 8x8 input: 112 LC0 history planes + 34 classical feature planes (king safety, pins/xrays, pawn structure, mobility, outposts). `encode_position()` is the main entry point.

### Model (`model/`)
Transformer encoder-only backbone (`ChessNet` in `transformer.py`). Configurable embed dim (256–1024), layers (6–15), heads (8–24). Optional Smolgen attention bias and Non-Linear Attention (NLA). Multi-task output heads:
- **Policy**: Attention-based, 4672 logits (LC0 from→to encoding)
- **Value**: WDL (win/draw/loss, 3 logits)
- **Optional**: SF move prediction, volatility, moves_left

`TinyNet` in `tiny.py` is a small reference model for testing.

### Move Encoding (`moves/`)
4672-plane LC0 policy encoding mapping (square, direction) pairs to policy indices.

### MCTS (`mcts/`)
Two variants: PUCT (standard AlphaZero-style) and Gumbel-max trick. Used during selfplay for move selection.

### Selfplay (`selfplay/`)
`play_batch()` orchestrates games. Network turns use MCTS + temperature sampling; Stockfish turns query engine with MultiPV for soft policy targets.

### Training (`train/`)
`Trainer` class runs training steps with `torch.amp` (BF16 on CUDA). Multi-component loss computed in `losses.py`. Target computation (WDL, volatility) in `targets.py`. AdamW optimizer with gradient clipping.

### Replay Buffer (`replay/`)
Circular buffer with KataGo-style surprise weighting (50% uniform + 50% priority sampling). `collate()` in `dataset.py` prepares batches.

### Stockfish Interface (`stockfish/`)
`StockfishUCI` for single-threaded UCI communication; `StockfishPool` for multi-worker parallel analysis.

### Configuration (`config.py`, `utils/config_yaml.py`)
YAML config provides defaults, CLI args override. Two-pass argparse: first pass loads YAML, second pass applies CLI overrides. Config dataclasses in `config.py`.

### ONNX Export (`onnx/`)
Export for Ceres chess engine compatibility.

### Hyperparameter Tuning (`tune/`)
Ray Tune integration with ASHA early stopping.

## Code Conventions

- Python 3.10+, uses `from __future__ import annotations` throughout
- Type hints on functions and dataclasses
- No configured linter; no formatter config
- Tests in `tests/` directory, pytest with `-q` flag
