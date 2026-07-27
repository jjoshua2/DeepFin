# CLAUDE.md

Guidance for Claude Code working in this repository. Everything here is either a hard
rule or a fact you cannot get by reading the code. Module-level detail is deliberately
absent — read the source.

## Project

Chess anti-engine training framework — trains a transformer to exploit Stockfish
weaknesses (fortress blindness, horizon effects, closed-position overconfidence).
CUDA, primarily one RTX 5090.

Per-iteration data flow: distributed selfplay (MCTS vs Stockfish) → shard upload →
ingest into disk-backed replay buffer → training step → checkpoint → publish model to
workers.

## Commands

```bash
pip install -e ".[dev]"     # full env; the test suite hard-requires every extra (no skips)
pip install -e ".[worker]"  # lite selfplay-client install
python -m pytest

PYTHONPATH=. python3 -m chess_anti_engine.run --config configs/pbt2_small.yaml --mode tune [--resume]
```

Scripts need `PYTHONPATH=.`. CLI modes are `train`, `tune`, `salvage` — there is no
`single` mode. Drive live training with `scripts/train.sh`, not the module directly;
see `docs/operations.md` for that and for salvage, blind-spot seeding, and lint detail.

## Configs

- `configs/pbt2_small.yaml` — **production**, and the only config active training uses.
  512-dim × 16-layer × 16-head, **63.08M trainable params** (63,084,128, counted
  2026-07-26 by unique storage on `checkpoint_000042`). **An earlier revision of
  this line "corrected" 63M up to 78.8M. That was the error, not the fix.**
  78.81M is the sum of `numel()` over the 496 `state_dict` entries, which
  double-counts weight tying: the 16 `layer_smolgens.N.gen_weight.weight` keys
  are **one shared tensor** (16 keys, 1 distinct storage, 1,048,576 params), so
  15 × 1,048,576 = 15,728,640 is counted 15 times too many — exactly the gap.
  Count unique `v.untyped_storage().data_ptr()`, never `sum(v.numel())`.
  Per-layer Smolgen still dominates, but by less than advertised:
  `layer_smolgens` **26.7M of 63.08M (42.3%)**, not 42.4M of 78.8M (54%).
  `ffn_mult` is per-layer and non-uniform (1.5 rising to ~1.9 in the upper
  blocks), so the count is not reproducible by assuming a flat multiplier.
  Related: only 28.6% of trainable params are in the Aurora matrix group
  (`matrix_optimizer_scope: mlp_out`) — see `docs/rl_loop_audit.md` I12.
  `tests/test_param_count.py` rebuilds both configs and fails if any number in
  this section drifts from the measurement.
- `configs/default.yaml` — reference BT3-scale model (768-dim × 15-layer × 24-head,
  **73,700,885 params**), unused. The "~105M" this line used to claim was never
  measured; the config has no tied tensors, so its `state_dict` sum agrees.
- `configs/exp_*.yaml` — flag-gated research bets, ALL default off. A flag enters the
  production config only once promoted; promotion status lives in the ledger, not here.

All tunables live in the yaml — grep it rather than assuming a value.

## Working on a live run

The run is usually live. These break production:

- **Never `git checkout` in this working tree while a run is live.** The live yaml is
  part of the tree and is re-read every iteration, so a branch switch silently reverts
  live experiments (2026-07-02: rolled back three experiments for 3 iterations). Use
  `git worktree` for all branch work, and merge PRs touching the live yaml promptly.
- **The live-yaml validator is all-or-nothing**: an unknown key rejects the WHOLE
  reload. Add config keys only after restarting onto code that defines them.
- **Never run a 256+ sim arena concurrent with training** — GPU OOM crashed the run
  2026-06-18. sims-1/32 arenas and `audit_targets`/`value_regret` at small batch with
  `--gpu-mem-fraction` are safe.
- Ray prunes live checkpoints. Copy one out of the tune dir before using it as a
  long-lived arena/audit baseline.
- After pulling `.c`/`.h` changes, rebuild with
  `python3 scripts/build_production_extensions.py` (GCC15 + native + LTO, forced
  rebuild) — NOT `pip install -e .`, the .venv setuptools lacks PEP 660. Keep
  distributed wheels portable unless every target CPU matches.

## Experiment protocol — MANDATORY

**`docs/experiment_ledger.md` is the canonical experiment record** (WORKED / FAILED /
LIVE-UNREAD verdicts, yardstick anchors, revert points). Read it before proposing,
launching, judging, or reverting any experiment.

**`docs/rl_loop_audit.md` is the companion invariant record** — per-stage checks with
the exact instrument and current status. The ledger says whether an experiment worked;
the audit says whether the pipeline that produced the number was sound. Read its
"Method rules" before deriving any metric by hand: several confident findings there
turned out to be artifacts of the measurement, not the loop. A verdict read off a stage
that FAILS its invariant is not a verdict.

1. **Before a training-affecting change goes live** (config key, loss weight,
   data-pipeline or selfplay change, PR merge that alters training): add a ledger entry
   with the hypothesis, ONE deciding yardstick as an exact command, and a pre-committed
   kill/success threshold. No entry → don't launch.
2. **Before big changes**, snapshot weights + optimizer + PID + replay window and record
   it in the ledger's Revert points table:
   `./scripts/train.sh salvage-export --top-n 1 --metric training_iteration --out data/salvage/<label>`
   (safe while training runs, ~2.3G; `--metric training_iteration` is REQUIRED — the
   default picks the best-metric row, not current state). A yaml revert is NOT a
   rollback: the replay window holds ~a day of data made under the old settings.
3. **After a readout**, record the verdict the same session, judged by the pre-committed
   rule rather than post-hoc reading. "Deferred" is not a verdict. Stability/throughput
   changes read out in ~5 iterations (~3h, spanning ≥2 of the failure's cadence periods);
   learning-quality changes need day-plus windows and paired CIs.
4. **One data-affecting change per readout window.** Unavoidable overlaps go in each
   entry's Confounds line.
5. **Before running any yardstick**, read the ledger's "Protocol gotchas".

## Evaluation

`docs/eval_protocol.md` is the decision protocol. The audit-first rule: every
training-target candidate is scored against the frozen deep-SF audit set BEFORE any
training compute, and one that loses the direct audit is killed without training.

- `scripts/arena_standard.py` — paired-opening arena, pentanomial Elo + 95% CI.
  `matched_sims` for search/target changes; `matched_time` only when the change ships in
  the fast C path, since Python-path features are under-credited.
- `scripts/build_audit_set.py` / `scripts/audit_targets.py` — the frozen audit set
  (≥1M nodes, MultiPV ≥10, side-to-move canonical) and its scorer. The set FREEZES after
  generation; new sampling = new version.
- `scripts/value_regret.py` — value-head 1-ply deep-SF regret, the VALUE yardstick.
  Brier/ECE are fooled by calibration and must not be used to judge value strength.
- `scripts/probe_policy_targets.py`, `scripts/retarget_retrain.py`,
  `scripts/convert_shards_v2_threats.py` — policy/soft-policy divergence, offline
  SF-target retuning, offline v1→v2_threats shard conversion.

## Non-obvious training facts

Consequential and not apparent from the code:

- **The WDL blend's SF component is load-bearing — do not zero it.** Removing it crashed
  winrate 0.64 → 0.40. The cp-logistic label is deliberately soft; don't chase value
  sharpness against a deep-SF ruler. Blend spec and the full head/target/loss table are
  in `docs/model_heads.md` — read it before touching `train/losses.py`, loss weights, or
  replay-sample target building.
- **`policy_sf` trains on the OPPONENT's reply distribution**, labels queried at P1 after
  the net's move and POV-flipped. It is not a move-teacher, which is why upweighting it
  hurt.
- **`wdl` is the only value head used in MCTS**; `sf_eval` and `categorical` are
  auxiliary.
- **PID regret runs backwards from intuition**: SF picks randomly among moves within
  `wdl_regret` of best, so higher regret = weaker SF = model wins more. The controller
  LOWERS regret to raise difficulty and RAISES it as an airbag on low winrate. Training
  targets are always best-move based and never depend on which handicapped move SF
  actually played.
- `selfplay.record_fast_ply_value` is OFF in production — tried and REVERTED for trunk
  dilution. Check the ledger before re-enabling.
- Production input is 175 planes (`v2_threats`); `v1`'s 146 planes are legacy. Production
  policy output is the compact `lc0_1858` encoding, though search still uses 4672 action
  ids. Use the shared device-cached lookups in `moves/torch_maps.py` — don't add
  per-module `lru_cache` copies.
- Production optimizer is `aurora` with `matrix_optimizer_scope: mlp_out`.
- Step budget uses views-targeting (`train_views_per_position`), so steps scale with
  ingest volume; `train_window_fraction` is the legacy mode and the ingest-drought floor.
- GPBT is wired up but effectively off — the production config pins everything.

## Code conventions

Python 3.10+ with `from __future__ import annotations`; type hints on functions and
dataclasses; tests in `tests/`. Write code that reads like the code around it.

Run `./scripts/lint.sh <paths>` after editing; the gate is kept at zero findings
repo-wide with no baseline. Fix a new finding in the same commit or disable the rule in
`pyrightconfig.json` if the whole category isn't worth the ceremony — there is no
deferral queue, don't recreate one. basedpyright ignores mypy-style
`# type: ignore[...]`; use `# pyright: ignore[reportRuleName]`. Never write a suppression
whose validity depends on the installed numpy/torch version — rewrite the code
version-proof instead.

## Reviews and pull requests

Optimize for end-state quality, not the cheapest diff. When a review surfaces an
improvement, decide it now: either make the change or say why it isn't worth making.
"Deferred to later" is an unresolved decision rotting in a comment. "Premature
abstraction" is a valid reason to skip; "it touches more files than I expected" is not.
State the call and the reasoning explicitly.

Open PRs ready for review, not draft, unless asked. The Codex review bot was disabled
2026-07-11 — don't wait for a bot review; every PR gets a manual correctness review (you
or a review subagent) before it counts as done, with the verdict recorded in the PR
conversation or session summary.
