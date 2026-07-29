# Evaluation protocol for architecture / target candidates

Every architecture or training-target candidate PR is judged with the same
measurement, so results stay comparable across months of experiments. The
production policy path is the compact `lc0_1858` encoding
(`configs/pbt2_small.yaml: model.policy_encoding`); all tooling below follows
the encoding stored in the checkpoint itself.

## Required measurements per candidate

1. **Frozen-holdout loss delta.** Evaluate candidate and reference on the same
   frozen holdout replay set at a matched training-step count, and record the
   deltas for at least `test_policy_loss` and `test_wdl_loss`. The holdout
   must be frozen — never the live (still-growing) replay window, which shifts
   under PID difficulty and window growth. **`test_wdl_loss` changed meaning
   on 2026-07-26**: it now reports the blended loss the optimizer sees rather
   than a hard one-hot diagnostic (docs/rl_loop_audit.md I7), so it steps by
   ~+0.026 at that boundary. Candidate and reference must be measured with the
   same code; never compare a delta across it. The old quantity survives as
   `test_wdl_onehot_loss`.

2. **1000-game paired arena, BOTH modes.**

   ```bash
   PYTHONPATH=. python3 scripts/arena_standard.py \
       --candidate <ckpt> --reference <ckpt> --games 1000 \
       --mode matched_sims --search-shape training --sims 64

   PYTHONPATH=. python3 scripts/arena_standard.py \
       --candidate <ckpt> --reference <ckpt> --games 1000 \
       --mode matched_time --ms-per-move 100
   ```

   - **`--search-shape` is REQUIRED for `matched_sims` and has no default.**
     `training` is what production selfplay runs (`c_scale` 0.1, `topk` 16,
     linear root, `gumbel_vloss_weight` from the yaml); `play` is the tuned
     UCI/match shape (`c_scale` 0.025, `topk` 32, log root, `vloss_weight` 3).
     Judge the training loop with `training`; judge the shipped engine with
     `play`. Before 2026-07-29 the script silently used the play shape and
     passed no `vloss_weight` at all, so **every arena Elo recorded before
     that date was measured on a third configuration neither flag reproduces**
     (the play shape at `vloss_weight=0`) and is not comparable with anything
     measured after — see the ledger's 2026-07-28 findings. The realized knobs
     are printed at startup and stored per row under `search_candidate` /
     `search_reference`.
   - `matched_time` rejects `--search-shape` and the other in-process search
     flags: it plays through UCI subprocesses that build their own search from
     their own flags. Pass those via `--uci-args`.
   - Openings come from the production 8-move UHO book
     (`opening_book_path_2` in `configs/pbt2_small.yaml`); each opening is
     played twice with colors swapped, and the pair is the unit of analysis.
   - The Elo point estimate and 95% CI come from the pentanomial pair-score
     variance, not the trinomial W/D/L counts — paired games are correlated
     and trinomial CIs are misleadingly tight.
   - `matched_sims` isolates net quality at a fixed search budget.
     `matched_time` runs each side as a real UCI engine
     (`python -m chess_anti_engine.uci`) at fixed wall clock per move, so a
     slower architecture pays for its latency. A candidate must be judged on
     both: the gap between the two numbers is its throughput cost in Elo.
   - Every run appends one JSON line to `runs/arena_results.jsonl` with git
     SHA, production-config hash, mode, and both checkpoint paths.

3. **Record the (holdout delta, arena Elo) pair** in the tracking table below.
   Accumulating these pairs builds the loss→Elo exchange rate, which is what
   lets us judge future candidates from cheap holdout numbers before paying
   for a full arena.

## Exchange-rate curves

- **Search exchange rate (sims → Elo).** How much strength one doubling of
  search buys; equivalently, what a throughput regression costs:

  ```bash
  PYTHONPATH=. python3 scripts/elo_vs_sims.py \
      --checkpoint <ckpt> --games-per-rung 400
  ```

  Arenas the checkpoint against itself at sims {32, 64, 128, 256, 512},
  adjacent rungs paired (~200 pairs per rung), and prints Elo-vs-previous-rung
  with CI. Re-run when the architecture or search changes materially; the
  curve is checkpoint-specific.

- **Loss exchange rate (holdout delta → Elo).** Not a script — it is the
  tracking table below, accumulated one candidate at a time.

## Variance notes

- Defaults are temperature 0.1 with root Gumbel noise in `matched_sims`
  (mirrors `scripts/match_checkpoints.py`). Fully deterministic play
  (`--temperature 0 --no-gumbel-noise`) makes self-play pairs carry zero
  information — both games of a pair mirror each other exactly — so keep the
  noise on for self-referenced runs like `elo_vs_sims`.
- 500 pairs (1000 games) gives a CI of roughly ±10-15 Elo at typical draw
  rates; if a candidate's effect is smaller than that, more games — not a
  tighter prior — is the only honest fix.

## Tracking table: holdout delta vs arena Elo

| date | candidate | reference | Δ test_policy_loss | Δ test_wdl_loss | Elo (matched_sims) | Elo (matched_time) | notes |
|---|---|---|---|---|---|---|---|
| _(append one row per candidate)_ | | | | | | | |

## Target audit: every training-target candidate is audited here FIRST

The arena gates above price a candidate AFTER spending training compute. The
target audit prices it BEFORE: a frozen, deeply-labeled position set
(`data/audit_set_v1.jsonl`, built once by `scripts/build_audit_set.py` with
unhandicapped Stockfish at >=1M nodes, MultiPV >= 10) and a scorer
(`scripts/audit_targets.py`) that evaluates any policy or value target
DIRECTLY against it.

**The rule: a target that loses the direct audit — higher expected deep-SF
regret, or worse Brier/ECE calibration than the incumbent — is killed
without training.** No fixed-budget run, no arena, no tracking-table row.
Only candidates that win or tie the audit graduate to the
training-then-arena pipeline above.

Mechanics:

- The audit set is FROZEN after generation (the build script refuses to
  overwrite; new sampling = new version `audit_set_v2...`). All candidates,
  forever, are scored against the same positions — phase-stratified
  (endgame/middlegame/opening by piece count), source-stratified
  (selfplay/curriculum), deduplicated, side-to-move canonical.
- Policy candidates are scored as expected deep-SF regret in cp of a move
  sampled from the distribution (plus top-1 regret), per phase and per
  source. Value candidates are scored as Brier + ECE against the deep-SF
  WDL, and separately against full-strength game outcomes where available.
- `scripts/audit_targets.py` reports the current production lineup as the
  incumbent baseline in every run: net raw policy, net+search at production
  sims, the SF MultiPV soft target at production label settings, the actual
  blended training target, and all four WDL components/blends. Reports land
  in `runs/target_audit_<sha>.md`.
- Two standing numbers to watch on each report: (search-policy regret) vs
  (SF-soft-target regret) per phase — when search wins everywhere, the
  50k-node MultiPV-40 labeling is no longer worth its CPU bill — and
  (production WDL blend) vs its best single component — when a single
  component matches the blend, the blend weights are dead complexity.
