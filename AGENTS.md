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
pip install -e ".[dev]"      # full server/test env (server+train+tune+onnx);
                             # the suite hard-requires all of them (no skips)
pip install -e ".[worker]"   # lite selfplay-client install
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
- Before stopping or restarting PBT, prefer `python3 scripts/graceful_restart.py` so active trials finish the current iteration and pause cleanly before the restart.
- Salvage workflows are CLI-driven: `./scripts/train.sh salvage-export ...` and `./scripts/train.sh salvage-restart <pool_dir>`. Do not edit `configs/pbt2_small.yaml` just to activate or disable salvage.
- `configs/pbt2_small.yaml` is the production config for active training (384-dim, 12-layer, ~46M params). `configs/default.yaml` is the larger reference config (768-dim, 15-layer, ~105M params).
- After pulling changes to `.c`/`.h` files in the local production checkout, rebuild with `python3 scripts/build_production_extensions.py` (validated GCC15 + native + LTO, always a forced rebuild; NOT `pip install -e .` — the .venv setuptools lacks PEP 660). Keep distributed wheels portable unless every target CPU matches.
- NEVER run a 256+ sim arena concurrently with training (GPU OOM crashed the live run 2026-06-18). sims-1/32 arenas and `audit_targets`/`value_regret` at small batch + `--gpu-mem-fraction` are safe concurrent.
- The live YAML is re-read every iteration and the strict validator rejects the WHOLE reload if it contains a key the running code does not know — add new config keys only after restarting onto code that defines them.
- Ray prunes live checkpoints; copy one out of the tune dir before using it as a long-lived arena/audit baseline.
- Production Syzygy uses the same paths as `configs/pbt2_small.yaml`: `/home/josh/projects/chess/data/syzygy_3-4-5:/mnt/e/chess/syzygy_6_dtz`. Despite the local directory name, it contains 3-6 man WDL (`.rtbw`) plus 3-5 man DTZ (`.rtbz`); the 6-man DTZ tables are on the external 8 TB drive mounted at `/mnt/e`.
- For engine matches or search debugging, pass that full colon-separated path to `SyzygyPath` for both engines when you want production-equivalent tablebase behavior. Do not fall back to `data/syzygy_3-4man` unless the task explicitly wants a small-tablebase smoke test.

## Coding Style & Naming Conventions
- Target Python 3.10+, 4-space indentation, and explicit type hints.
- Match existing files by keeping `from __future__ import annotations` at module top.
- Naming: functions/modules `snake_case`, classes `PascalCase`, tests `test_*`.
- Keep imports grouped `stdlib` / `third-party` / `local`.
- Run `./scripts/lint.sh <paths>` after editing Python when practical. `./scripts/lint.sh --changed` checks changed and untracked Python files; `--deep` adds the slower advisory passes (skylos + ruff cleanup report, ~40s) — run it before a cleanup pass, not per edit.
- The lint gate is ruff + basedpyright + vulture at **zero findings repo-wide, no baseline** (pylint was removed 2026-07-02; its checks migrated into the ruff rule set). Fix new findings in the same commit or disable the whole rule in config — there is no "fix later" list; don't recreate one.
- Prefer fixing findings over adding inline suppressions; when suppression is necessary, use the repo's existing syntax (`# pyright: ignore[rule]`, `# noqa: ...`, `# skylos: ignore`). basedpyright does NOT honor mypy-style `# type: ignore[...]` comments — they are inert here.

## Testing Guidelines
- Test framework is `pytest` (configured in `pyproject.toml` with `testpaths = ["tests"]` and quiet output).
- Add or update tests with every behavior change; prefer deterministic unit tests around encoding, replay, MCTS, and training targets.
- For distributed or networking changes, run `tests/test_e2e_smoke.py` and/or `./scripts/e2e_distributed_smoke_gumbel.sh`.
- There is no hard coverage threshold; reviewers expect regression-focused coverage for touched code.

## Architecture Guardrails
- Data flow per iteration is distributed selfplay -> shard upload -> disk-backed replay ingest -> training step -> checkpoint -> publish model to workers.
- Production input encoding is **175 planes** (`input_extra_features: v2_threats`): 112 LC0 history planes + 34 classical feature planes + 29 threat planes. Legacy `v1` = 146 planes (no threat planes). The C extensions in `encoding/_lc0_ext.c` and `mcts/_mcts_tree.c` are performance-critical.
- Gumbel MCTS with sequential halving is the primary search path; PUCT is legacy.
- Distributed workers use the shared inference broker (`SlotBroker` / `SharedSlotBroker`) with pinned shared-memory slots. Be careful with buffer shape, lifetime, and model hot-swap behavior.
- YAML config is flattened by `utils/config_yaml.py`. Live YAML reload happens each iteration for non-topology keys, while PBT-searched keys are preserved across reloads.
- PID regret direction is intentional: higher `wdl_regret` gives Stockfish a wider pool of acceptable moves and makes the opponent weaker; lower regret makes it harder. Training targets remain best-move/objective-eval based, not based on the handicapped move actually selected.
- The `wdl` head is the only value head used by MCTS. `sf_eval` and `categorical` are auxiliary and should not be substituted into search casually.
- `policy_sf` trains on a soft distribution over Stockfish MultiPV candidate WDL scores. SF labels are queried at **P1** (after the net's move), so `policy_sf` is the opponent-reply distribution, not a move teacher for the sample's own position (the `sf_p0_*` fields provide that). `sf_move_index` is for top-1 accuracy metrics, not a one-hot training target.
- The main `wdl` head trains on a three-way soft blend: `game_frac`·outcome + `sf_wdl_frac`·SF cp-logistic label + `search_wdl_frac`·own search-root WDL (no separate `w_sf_wdl` knob exists — the blend replaced the old dual loss). The SF component is load-bearing supervision: do not zero `sf_wdl_frac` as a cleanup without training evidence (doing so crashed winrate 0.64→0.40 in 4 iterations, 2026-04-17).
- Evaluation protocol: `docs/eval_protocol.md`. Standardized arenas run through `scripts/arena_standard.py` (paired openings, pentanomial Elo+CI); training-target candidates are scored against the frozen deep-SF audit set (`scripts/audit_targets.py`) BEFORE training compute is spent.
- Flag-gated experiments live in `configs/exp_*.yaml` with all flags default-off; `configs/pbt2_small.yaml` is never edited to run an experiment. Open flag families: `train.soft_policy_min_tv`, `train.sf_policy_sparse_ce` + `selfplay.record_dense_sf_policy` (config load refuses the dense-off/sparse-off combination), and `selfplay.volatility_*` (forces the Python Gumbel path with a logged warning — the C fast path does not implement it). Promoted to production: v2_threats planes (2026-06-17) and the throughput triple `sf_label_nodes_cap` + `record_fast_ply_value` + `train_views_per_position` (2026-07-01). Shelved: `model.use_dynamic_relations` (offline ≡ threat planes).
- Shared device-cached policy lookup tensors live in `chess_anti_engine/moves/torch_maps.py`; do not re-add per-module `lru_cache` copies of `COMPACT_TO_FULL_POLICY`/`FULL_TO_COMPACT_POLICY`.

## Commit & Pull Request Guidelines
- Follow the existing commit style: short, imperative subjects (for example, `Fix network-turn alignment...`, `Lazy-load Trainer...`).
- Keep commits focused by subsystem and avoid mixing refactors with logic changes.
- PRs should include: what changed, why, config/CLI impacts, and test evidence (commands run + brief results).
- Link related issues when available; include training metric screenshots only when they clarify behavior changes.
- Open Codex-authored PRs as ready for review by default so review automation runs. Use draft only when explicitly requested or when the branch is intentionally incomplete.
- After opening a PR, check back after roughly 30 minutes for automated code-review comments and resolve actionable feedback promptly when it is safe to do so.
- Do not let local changes accumulate across unrelated tasks. If the working tree contains more than one focused change, split work into separate branches/PRs before continuing.
- Prefer a temporary worktree for PR preparation when the live training checkout has local config edits, so production tuning changes do not leak into unrelated branches.

## Review Protocol
- Optimize for the end state rather than the smallest diff. When review surfaces an improvement, either make it or explicitly decide it is not worth doing.
- State decisions clearly in review responses and summaries; avoid leaving vague TODOs for choices that should be settled now.

## Experiment Protocol (MANDATORY — all agents)

`docs/experiment_ledger.md` is the canonical experiment record (verdicts, yardstick
anchors, revert points, decision queue). Read it before proposing, launching,
judging, or reverting ANY experiment, and follow its rules:

1. Before a training-affecting change goes live: add a ledger entry with the
   hypothesis, ONE deciding yardstick (exact runnable command), and a
   pre-committed kill/success threshold. No entry → don't launch.
2. Before big changes: snapshot weights + optimizer + PID + replay window
   (`./scripts/train.sh salvage-export --top-n 1 --metric training_iteration
   --out data/salvage/<label>`) and record it in the ledger's Revert points.
   A yaml revert alone is NOT a rollback — the replay window keeps ~a day of
   data made under the old settings.
3. After a readout: record the verdict in the ledger the same session, judged
   by the pre-committed rule. "Deferred" is not a verdict.
4. One data-affecting change per readout window; unavoidable overlaps go in
   each entry's Confounds.
5. Before running any yardstick: read the ledger's "Protocol gotchas".
