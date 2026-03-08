# Repository Guidelines

## Project Structure & Module Organization
- `chess_anti_engine/` is the main package. Core areas: `encoding/` + `moves/` (input/policy encoding), `model/` + `mcts/` + `selfplay/` + `train/` + `replay/` (training loop), and `stockfish/` + `server/` + `worker.py` + `learner.py` (distributed pipeline).
- `tests/` contains pytest coverage for units and smoke flows (`test_*.py`).
- `configs/` stores YAML presets (for example, `default.yaml`, `pbt2_small.yaml`).
- `scripts/` contains automation for smoke/e2e runs and monitoring.
- Generated data (`runs/`, `tb/`, `server/`, `data/`, large model/book artifacts) is local runtime output and should generally stay uncommitted.

## Build, Test, and Development Commands
```bash
pip install -e .
pip install -e ".[dev]"      # pytest extras
pip install -e ".[tune]"     # Ray Tune + Optuna
pip install -e ".[onnx]"     # ONNX export/runtime support
```
```bash
python -m pytest
python -m pytest tests/test_transformer_forward.py
python -m chess_anti_engine.run --config configs/default.yaml --mode single
python -m chess_anti_engine.run --config configs/default.yaml --mode tune
./scripts/e2e_distributed_smoke_gumbel.sh
```
- Use `--mode single` for local training loops, `--mode tune` for hyperparameter search, and the smoke script for server/learner/worker integration checks.

## Coding Style & Naming Conventions
- Target Python 3.10+, 4-space indentation, and explicit type hints.
- Match existing files by keeping `from __future__ import annotations` at module top.
- Naming: functions/modules `snake_case`, classes `PascalCase`, tests `test_*`.
- No enforced formatter/linter config exists; follow local style and keep imports grouped `stdlib` / `third-party` / `local`.

## Testing Guidelines
- Test framework is `pytest` (configured in `pyproject.toml` with `testpaths = ["tests"]` and quiet output).
- Add or update tests with every behavior change; prefer deterministic unit tests around encoding, replay, MCTS, and training targets.
- For distributed or networking changes, run `tests/test_e2e_smoke.py` and/or `./scripts/e2e_distributed_smoke_gumbel.sh`.
- There is no hard coverage threshold; reviewers expect regression-focused coverage for touched code.

## Commit & Pull Request Guidelines
- Follow the existing commit style: short, imperative subjects (for example, `Fix network-turn alignment...`, `Lazy-load Trainer...`).
- Keep commits focused by subsystem and avoid mixing refactors with logic changes.
- PRs should include: what changed, why, config/CLI impacts, and test evidence (commands run + brief results).
- Link related issues when available; include training metric screenshots only when they clarify behavior changes.
