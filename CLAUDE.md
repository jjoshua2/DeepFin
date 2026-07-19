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

Salvage is driven entirely by CLI flags (`--salvage-seed-pool-dir`, `--salvage-restore-*`), so you don't need to edit `configs/pbt2_small.yaml` to activate or disable it. When to salvage: after a bad exploit, a training run that regressed, or to rebase onto a better-regret checkpoint. A pool is a one-shot seed — once trials are past startup it plays no further role. `./scripts/train.sh best-save | best-list` manages named good-checkpoint pools under `data/best_pools/`.

**Observers** (auto-started by `train.sh start`, one instance each): `scripts/watchdog_loop.sh` — stall detection with auto-recovery (confirmed stall → `scripts/recover_stall.sh`; log `scratchpad/watchdog.log`), and `scripts/monitor_fen.sh` — cadenced FEN-panel reads + the seed retire/probation step (log `scratchpad/live_read/monitor/`).

**Blind-spot FEN seeding** (the active data lever — seeds selfplay openings
from positions the net misplays):

- The active list is `selfplay.opening_fen_list_path` in the live yaml,
  delivered via the server dole (`opening_fen_dole_per_iter`), selfplay-only,
  PID-safe. The path is live-reloaded — no restart to change lists.
- **Removal is automatic**: `scripts/blindspot_retire_step.py` (run on cadence
  by the monitor_fen loop) retires seeds after 2 consecutive AWARE reads, runs probation re-feed on retirees that regress, writes a new
  versioned list, and repoints the yaml itself. Don't hand-edit the active
  list for removals.
- **Feeding new seeds is manual and ledger-gated** (it's a data-affecting
  change — one per readout window, needs a ledger entry). Mechanics:
  `PYTHONPATH=. python3 scripts/blindspot_feed_step.py --batch <vetted-file> --tag <label>`
  — dedupes against the active pool and the retired store, writes a fresh
  versioned file (loader caches are path-keyed; never edit a list in place),
  validates, and repoints the yaml. `--dry-run` first. Per-seed dose control:
  `# weight=N` line markers.
- New seed material comes from mining external-match losses
  (`scripts/mine_blindspot_seeds.py --pgn`; default bakes blunder+deep-SF
  first-refute ply into the seed — `--no-append-refute-ply` for bare
  blind-spot terminals). `--existing` is for incremental *new-hole* growth
  only (dedup keys the pre-blunder blind-spot for bare and post-refute
  lines, so a bare list there will not re-emit upgraded seeds for those
  holes). To upgrade known bare holes: re-mine from PGN without them in
  `--existing`, write a new versioned list, feed as replacement — then
  vetting/gating before feed.

**Operational gotchas:**

- After pulling changes to `.c`/`.h` files in the local production checkout, rebuild with `python3 scripts/build_production_extensions.py` (validated GCC15 + native + LTO, always a forced rebuild; NOT `pip install -e .` — the .venv setuptools lacks PEP 660). Keep distributed wheels portable unless every target CPU matches.
- Scripts that import the C extensions run with `PYTHONPATH=.` (either python works since the 2026-07-02 numpy-2 rebuild — extensions built with numpy-2 headers import under both `/usr/bin/python3` and `.venv`). The production builder always forces recompilation, including after numpy or header changes.
- NEVER run a 256+ sim arena concurrent with training (GPU OOM crashed the live run 2026-06-18). sims-1/32 arenas and `audit_targets`/`value_regret` at small batch + `--gpu-mem-fraction` are safe concurrent.
- The live YAML is re-read every iteration, and the strict validator rejects the WHOLE reload if it contains a key the running code doesn't know — add new config keys only after restarting onto code that defines them.
- Live checkpoints get pruned by Ray; before using one as a long-lived reference (arena/audit baseline), copy it out of the tune dir first.
- **Never `git checkout` in this working tree while a run is live** — the live YAML is part of the tree and is re-read every iteration, so a branch switch can silently revert live experiments (2026-07-02: a checkout rolled back the label uncap, fast-ply revert, and rung-1 blend for 3 iterations). Do all branch work in `git worktree` checkouts, and merge PRs that touch the live yaml promptly.

## Configs

- `configs/pbt2_small.yaml` — **Production config.** 512-dim, 16-layer model (~63M params — per-layer Smolgen dominates the count; swapped in 2026-07-11 from the previous 384×12/46M net via offline bootstrap distillation). Distributed selfplay with shared inference broker, PID difficulty controller, PBT/GPBT hyperparameter search. All active training uses this.
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
   a verdict. Readout-window sizing: stability/throughput changes (crashes,
   wedge cadence, games/h) can be judged in ~5 iterations (~3h; span >=2 of
   the failure's cadence periods); learning-quality changes need day-plus
   windows and paired CIs.
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
  `configs/pbt2_small.yaml` once promoted. Which bets are promoted, open,
  killed, or shelved is tracked in `docs/experiment_ledger.md` — do not rely
  on this file for experiment status.

## Architecture

**Data flow per iteration:** distributed selfplay (MCTS games vs Stockfish) → shard upload → ingest into disk-backed replay buffer → training step → checkpoint → publish model to workers.

### Input Encoding (`encoding/`)
Production input is **175 planes** (`input_extra_features: v2_threats`): 112 LC0 history planes + 34 classical feature planes (king safety, pins/xrays, pawn structure, mobility, outposts) + 29 threat planes. Legacy `v1` = 146 planes (no threat planes). `encode_position()` is the main entry point. C extension `_lc0_ext` provides `CBoard` for fast board operations (push, encode, legal moves).

### Model (`model/`)
Transformer encoder-only backbone (`ChessNet` in `transformer.py`). BT4-aligned architecture with Smolgen attention bias, gating, configurable embed dim/layers/heads. Multi-task output heads: 4 policy heads (`policy`, `policy_soft`, `policy_future`, `policy_sf` — separate `AttentionPolicyHead`s sharing the trunk, lc0_1858 compact logits in production), value heads (`wdl` — the ONLY head used in MCTS; `sf_eval` and `categorical` are auxiliary), plus `volatility`/`sf_volatility`/`moves_left`. **The full head/target/loss table and its gotchas live in `docs/model_heads.md`** — read it before touching `train/losses.py`, loss weights, or replay-sample target building. Two non-negotiables from that doc: the WDL blend's SF component is load-bearing (zeroing it crashed winrate 0.64→0.40; the cp-logistic label is deliberately soft — don't chase value sharpness vs a deep-SF ruler), and `policy_sf` trains on the OPPONENT's reply distribution (labels queried at P1), not a move-teacher.

`TinyNet` in `tiny.py` is a small reference model for testing.

### Move Encoding (`moves/`)
4672-plane LC0 policy encoding mapping (square, direction) pairs to policy indices. Production uses the compact `lc0_1858` policy encoding; conversion maps live in `moves/encode.py` and the shared device-cached torch lookup tensors in `moves/torch_maps.py` (use these — don't add per-module `lru_cache` copies).

### MCTS (`mcts/`)
Gumbel MCTS with sequential halving (primary) and PUCT (legacy). C-accelerated tree operations in `_mcts_tree.c` — fused tree traversal + CBoard replay + encoding. `gumbel_c.py` orchestrates the simulation loop with GPU inference pipelining.

### Selfplay (`selfplay/`)
`play_batch()` orchestrates games (mostly net-vs-net "selfplay" slots plus a curriculum fraction vs the PID-handicapped SF opponent). Network turns use MCTS + temperature sampling; ~75% of plies are playout-capped "fast" plies (`playout_cap_fraction`/`fast_simulations`), the rest full-sim. `has_policy` ⇔ full ply; `selfplay.record_fast_ply_value` can keep fast plies as **value-only rows** — OFF in production (tried and REVERTED, trunk dilution; see the ledger before re-enabling). SF labels (`sf_wdl`/`sf_policy`) attach only to full plies, are queried at P1 (after the net's move) and POV-flipped to the sample; `stockfish.sf_label_nodes_cap` caps label-query nodes independently of the PID opponent budget.

### Distributed Selfplay (`tune/distributed_runtime.py`, `server/`)
Workers run as separate processes, each playing game batches via shared inference broker. Broker (`inference.py: SlotBroker/SharedSlotBroker`) uses pre-allocated shared memory slots with pinned CPU buffers for zero-copy GPU transfer. Workers upload shard files to server inbox; trainable ingests them into the replay buffer each iteration.

### PID Difficulty Controller (`stockfish/pid.py`)
Adaptive opponent strength via WDL regret-based difficulty. PID controller targets `sf_pid_target_winrate` (see `configs/pbt2_small.yaml`) by adjusting `wdl_regret` and SF node count.

**How regret works:** SF selects randomly among all moves whose WDL loss vs the best move is within `wdl_regret`. Higher regret = wider pool of acceptable moves including bad ones = SF plays weaker = model wins more easily. Lower regret = SF constrained to near-optimal moves = harder. So the controller LOWERS regret to increase difficulty and RAISES it when winrate is too low (airbag). The training target is always best-move based — `policy_sf` trains on the soft distribution over SF's MultiPV candidates by WDL eval, and `sf_wdl` reflects the objective position eval — neither depends on which handicapped move SF actually chose.

### Training (`train/`)
`Trainer` class runs training steps with `torch.amp` (BF16 on CUDA). Multi-component loss computed in `losses.py`. Optimizer is configurable (`aurora` / `nadamw` / `adamw` / `cosmos` / ...); production uses **`aurora`** with `matrix_optimizer_scope: mlp_out`. Gradient clipping via z-clip (`zclip_max_norm` hard cap + z-score outlier clip). Per-iteration step budget: production uses **views-targeting** (`train_views_per_position`: trained-samples per ingested position held fixed, so steps scale with ingest volume); `train_window_fraction` is the legacy mode and the ingest-drought floor.

**The WDL blend's SF component is load-bearing — do not zero it** (crashed winrate 0.64→0.40 when removed; full rationale and the blend spec in `docs/model_heads.md`).

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

Run `./scripts/lint.sh <paths>` after editing. The default GATE is **ruff + basedpyright + vulture** (wall time ≈ basedpyright, a few seconds) and is kept at **zero findings repo-wide — no baseline**; CI gates basedpyright on the whole repo.

```bash
./scripts/lint.sh chess_anti_engine/train/trainer.py   # specific files
./scripts/lint.sh --changed                            # changed and untracked .py files
./scripts/lint.sh --deep [paths...]  # + skylos + ruff cleanup report (advisory, ~40s —
                                     #   run before a cleanup pass, not per edit)
```

The `--deep` ruff report sweeps the not-yet-gated groups (B, NPY) as a cleanup shopping list — only the needs-judgment tail remains there (B905 `zip strict=`, B008 call-in-default, NPY002 legacy-RNG — never autofix; NPY002 changes RNG streams). Promote a rule into the pyproject gate once it reaches zero findings.

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

Open PRs as **ready for review, not draft** (only use draft when explicitly
asked). The Codex review bot was DISABLED 2026-07-11 — do not wait for or
expect a bot review. Every PR gets a manual correctness review instead (the
assistant or a review subagent) before it is considered done; record the
verdict in the PR conversation or the session summary.
