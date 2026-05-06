# Repository Guidelines

## Project Structure & Module Organization
- This is a chess anti-engine training framework: it trains a transformer network to exploit Stockfish weaknesses through distributed selfplay, search, and replay-buffer training.
- `chess_anti_engine/` is the main package. Core areas: `encoding/` + `moves/` (input/policy encoding), `model/` + `mcts/` + `selfplay/` + `train/` + `replay/` (training loop), and `stockfish/` + `server/` + `worker.py` + `tune/` (distributed pipeline).
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
PYTHONPATH=. python3 -m chess_anti_engine.run --config configs/pbt2_small.yaml --mode tune
PYTHONPATH=. python3 -m chess_anti_engine.run --config configs/pbt2_small.yaml --mode tune --resume
python -m chess_anti_engine.run --config configs/default.yaml --mode train
```
- Use `--mode train` for a single distributed trial (no PBT); `--mode tune` adds population-based search on top of the same pipeline. There is no non-distributed selfplay path — `--mode train` still boots the local server + at least one worker.
- Current CLI modes are `train`, `tune`, and `salvage`; there is no `single` mode.
- Some scripts assume repo-root imports; use `PYTHONPATH=.` when invoking module or script entrypoints directly.

## Training Operations
- Use `./scripts/train.sh` to drive live training. It manages the PID file, log, Ray cleanup, and orphan worker sweep.
```bash
./scripts/train.sh start
./scripts/train.sh start --fresh
./scripts/train.sh stop
./scripts/train.sh restart
./scripts/train.sh status
./scripts/train.sh log
```
- `./scripts/train.sh start` auto-resumes when `$WORK_DIR/tune/experiment_state-*.json` exists. Without that resume path, a restart can silently drop the running trial and spawn a random-init one. Use `--fresh` only when intentionally abandoning the prior Tune state; do not remove the Tune directory while a run is live.
- Before stopping or restarting PBT, prefer `python3 scripts/graceful_restart.py --wait N` so trials finish the current iteration and pause cleanly.
- Salvage workflows are CLI-driven: `./scripts/train.sh salvage-export ...` and `./scripts/train.sh salvage-restart <pool_dir>`. Do not edit `configs/pbt2_small.yaml` just to activate or disable salvage.
- `configs/pbt2_small.yaml` is the production config for active training. `configs/default.yaml` is the larger reference config.

## Coding Style & Naming Conventions
- Target Python 3.10+, 4-space indentation, and explicit type hints.
- Match existing files by keeping `from __future__ import annotations` at module top.
- Naming: functions/modules `snake_case`, classes `PascalCase`, tests `test_*`.
- Keep imports grouped `stdlib` / `third-party` / `local`.
- Run `./scripts/lint.sh <paths>` after editing Python when practical. `./scripts/lint.sh --changed` checks changed and untracked Python files; `--fast` skips the slower vulture and skylos passes.
- New code should be clean for ruff, basedpyright, pylint, vulture, and skylos. Prefer fixing findings over adding inline suppressions; when suppression is necessary, use the repo's existing syntax (`# pyright: ignore[rule]`, `# noqa: ...`, `# skylos: ignore`).

## Testing Guidelines
- Test framework is `pytest` (configured in `pyproject.toml` with `testpaths = ["tests"]` and quiet output).
- Add or update tests with every behavior change; prefer deterministic unit tests around encoding, replay, MCTS, and training targets.
- For distributed or networking changes, run `tests/test_e2e_smoke.py` and/or `./scripts/e2e_distributed_smoke_gumbel.sh`.
- There is no hard coverage threshold; reviewers expect regression-focused coverage for touched code.

## Architecture Guardrails
- Data flow per iteration is distributed selfplay -> shard upload -> disk-backed replay ingest -> training step -> checkpoint -> publish model to workers.
- Input encoding is 146 planes: 112 LC0 history planes plus 34 classical feature planes. The C extensions in `encoding/_lc0_ext.c` and `mcts/_mcts_tree.c` are performance-critical.
- Gumbel MCTS with sequential halving is the primary search path; PUCT is legacy.
- Distributed workers use the shared inference broker (`SlotBroker` / `SharedSlotBroker`) with pinned shared-memory slots. Be careful with buffer shape, lifetime, and model hot-swap behavior.
- YAML config is flattened by `utils/config_yaml.py`. Live YAML reload happens each iteration for non-topology keys, while PBT-searched keys are preserved across reloads.
- PID regret direction is intentional: higher `wdl_regret` gives Stockfish a wider pool of acceptable moves and makes the opponent weaker; lower regret makes it harder. Training targets remain best-move/objective-eval based, not based on the handicapped move actually selected.
- The `wdl` head is the only value head used by MCTS. `sf_eval` and `categorical` are auxiliary and should not be substituted into search casually.
- `policy_sf` trains on a soft distribution over Stockfish MultiPV candidate WDL scores. `sf_move_index` is for top-1 accuracy metrics, not a one-hot training target.
- `w_sf_wdl` is load-bearing supervision on the main WDL head, not an accidental conflicting target. Do not remove or zero it as a cleanup without training evidence.

## Commit & Pull Request Guidelines
- Follow the existing commit style: short, imperative subjects (for example, `Fix network-turn alignment...`, `Lazy-load Trainer...`).
- Keep commits focused by subsystem and avoid mixing refactors with logic changes.
- PRs should include: what changed, why, config/CLI impacts, and test evidence (commands run + brief results).
- Link related issues when available; include training metric screenshots only when they clarify behavior changes.

## Review Protocol
- Optimize for the end state rather than the smallest diff. When review surfaces an improvement, either make it or explicitly decide it is not worth doing.
- State decisions clearly in review responses and summaries; avoid leaving vague TODOs for choices that should be settled now.
