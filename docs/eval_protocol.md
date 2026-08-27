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
   - **Cap a run with `--max-seconds`, never with an external `timeout` alone.**
     A SIGKILLed arena loses whatever is still in its block-buffered stdout,
     which is always exactly the LAST block printed — each flushed line pushes
     everything written before it. A run that printed several blocks keeps all
     but the last; a run slow enough to print only ONE loses everything, and
     its log ends mid-report at `[arena] RUNNING Elo after 6 complete pairs:`
     with no `[arena] Elo:` line for any parser to read. That is why the daily
     ratchet wrote no CSV row on 2026-07-30 and 07-31.
     `--max-seconds` stops the play loop on the arena's own clock, scores the
     opening pairs that FINISHED (half-played games are dropped, never imputed
     as draws), prints the summary and appends the record. A capped run is then
     a small sample rather than no sample, and `games`/`pairs` in the record
     mean what they say; `games_requested` and `truncated` say what was asked
     for. With nothing complete it prints `NO COMPLETE PAIRS` and exits 3.
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
      --checkpoint <ckpt> --games-per-rung 400 --search-shape training
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

## A crashed match is resumable — never replay a finished game

Both drivers (`scripts/arena_standard.py`, `scripts/match_vs_uci.py`) append one
flushed JSONL record per FINISHED game to a **game log**, so a run that dies
mid-match leaves every game that ended. 2026-08-21 is why: a 128-game compiled
arena OOMed at ply 20 with ZERO games persisted, and its relaunch lost its first
minutes the same way.

- **Where**: `--games-out <path>`. Default for the arena is
  `runs/arena_games/<label>.<settings-fingerprint>.games.jsonl`; for
  `match_vs_uci` it rides `--pgn-out` when that is given, else
  `runs/match_games/`. It is deliberately NOT derived from the arena's `--out`,
  which is the shared append-only aggregate every arena in history writes to.
- **Resume**: `--resume` keeps what finished and plays only the remainder, then
  scores everything as one match. The arena keeps COMPLETE PAIRS only — a pair
  with one coloring played is discarded and replayed, because the pentanomial's
  sampling unit is the pair. The result record carries `resumed_pairs`,
  `resumed_orphan_pairs` and `game_log`, so a row that is a splice of two
  processes says so.
- **It refuses to mix.** The log header stores a fingerprint of every setting
  that defines the population and the ruler (nets/engines, seed, games,
  openings, mode, sims or budget, search shape, syzygy). `--resume` onto a
  changed one names the difference and exits. Execution knobs are deliberately
  NOT in the fingerprint — resuming an OOMed arena at a lower
  `--max-concurrent-games` is the main use.
- **Without `--resume`, an existing log for the same settings is an error.**
  Appending would silently pool two runs. `scripts/daily_gate_ratchet.sh` passes
  `--resume` for exactly this reason: its same-day retry of a series is the same
  invocation, and it now continues that attempt instead of replaying it.
- ⚑ A resumed PGN keeps the discarded orphan game (append-only). Its
  replacements carry `ResumeReplay "1"`, so for any `PairId` that has the tag,
  the games WITHOUT it are the stale ones — drop those before a pooled fit.

## Pooled multi-player ratings (Ordo) — when 3+ arms play a round robin

A paired arena answers ONE A/B. For three arms (Tier-13 A/B/C) it answers three
of them separately and cannot use the indirect comparisons. Ordo fits all
players jointly from pooled PGN, models white advantage and draw rate, and
tolerates unequal pair counts per matchup.

**Produce the PGN**: `scripts/arena_standard.py --pgn-out <file>` (default OFF;
the run is unchanged when it is unset). **Fit it**:
`PYTHONPATH=. python3 scripts/ordo_pooled_fit.py <pgn>... --anchor <name>`.
Ordo lives at `~/local_engines/ordo/ordo` (built from source, v1.2.6,
commit `17eec774`); it is NOT in this repo.

What was measured, on synthetic paired-opening data at known ratings:

- **Ordo's point estimate matches ours.** On independently simulated
  two-player games, Ordo's rating difference and `arena_standard`'s pentanomial
  Elo agree to a mean of +0.4 Elo (max 1.7 over 30 reps at 400 games). Both
  estimators are unbiased, so a MATERIAL gap between them is a bug, not a
  modelling choice. This is the real agreement test — the games are fresh
  draws, not derived from either estimator's output.
- **⚑ The BANKED-row comparison is a check on `-z`, NOT an agreement test.**
  Reconstructing three banked arena rows from their pentanomial counts
  reproduces the banked Elo to 0.04 Elo (+49.00 vs +48.96, +21.60 vs +21.57,
  −3.50 vs −3.47) — but that is close to tautological and must not be quoted as
  independent corroboration. **Measured:** for two players with balanced
  colours, Ordo's rating is a function of TOTAL SCORE alone — holding the score
  at 0.57 and sweeping the draw fraction from 0.10 to 0.75 returns **exactly
  49.00 every time**. Pentanomial Elo is likewise a function of total score, and
  a pentanomial reconstruction preserves total score exactly, so the two must
  agree whenever the SCALE is right. What the check therefore falsifies is the
  scale constant, and it does that with ~10× margin: the same data reads
  **+49.00 at `-z 200.2409` and +49.40 at Ordo's default `-z 202`**, against a
  banked +48.96. It also means the DD/WL reconstruction ambiguity cannot move
  the point estimate — the earlier caveat about it was over-cautious.
- **Ordo cannot exploit our mirrored pairs.** Its entire per-game state is
  `{whiteplayer, blackplayer, score}` (`mytypes.h:90`), and `-s` is a
  PARAMETRIC Monte Carlo that regenerates every game independently from the
  fitted model (`sim.c: simulate_scores`) — not a bootstrap. BayesElo is worse
  still: it condenses games to per-opponent colour-split counts
  (`CCondensedResults.h`). Neither can see an opening.
- **The cost of that is small for a balanced book and large for an unbalanced
  one.** 95% CI width, independent-games vs paired, as per-opening colour bias
  grows: 1.00 (0 Elo), 1.02 (60), 1.07 (120), 1.17 (200), 1.31 (300), 1.53
  (450). Production plays a UHO book, so this is not negligible.
- **⚑ Read the PAIRWISE CONTRAST row, not the two per-player rows.** "Is A
  better than B" is a contrast, and its interval is NOT recoverable from two
  per-player intervals — the players' errors are correlated because they are
  fitted jointly from overlapping games, and the sign of that correlation
  depends on the anchoring. Measured on the same 1000-game 3-arm file:
  under `--anchor armC` the per-player halfwidths are 25.5 / 0 / 23.6, under
  pool-average anchoring they are 14.7 / 15.2 / 12.3 — **while the A−B contrast
  is 20.3 and 20.4 respectively.** The contrast is invariant to the anchoring;
  the per-player numbers are an artifact of a reporting choice, and under
  pool-average anchoring they understate the contrast by ~2×. `ordo_pooled_fit`
  prints the contrasts because that is the number a reader acts on.
- **The fix is a pair-level block bootstrap**, which `ordo_pooled_fit.py` does:
  Ordo's joint fit for the point estimates, then resampling whole PAIRS,
  stratified by matchup and preserving each matchup's own pair count.
  Resampling pairs pooled across matchups would perturb the design, not the
  data. Coverage of the true value measured at 36/40 = 0.90 ± 0.05 (nominal
  0.95, n=40 — reassuring, not conclusive).
- **Pooling BUYS precision when you need all three answers.** In a 3-arm round
  robin, Ordo's A−C estimate (using the A−B and B−C games too) had true SD
  10.57 vs 13.27 for pentanomial on the A−C games alone — same precision for
  ~37% fewer games. ⚑ But if A−C is the ONLY question, spending the whole
  budget on A vs C directly still beats pooling.

**⚑ Ordo flags that are not optional.**

- `-M` (force maximum likelihood) is **required** whenever `-D` (auto draw
  rate) is combined with `-s`. Without it Ordo can fail to converge on an
  unlucky SIMULATED replication and spin forever: measured **1/30 datasets hung
  at `-s 1000`**, 0/30 with `-M`, 0/30 with a fixed `-d`. The sim RNG is
  fixed-seeded (`main.c:981`), so a hang is deterministic per dataset — rerunning
  will not clear it.
- `-z 200.2409`, not the default 202. Ordo's default puts it on
  invbeta = 202 / −ln(1/0.76−1) = 175.2447 while our Elo is
  −400·log10(1/p−1), i.e. 400/ln(10) = 173.7178 — Ordo's number is **0.88%
  larger** for the same score unless rescaled.
- Ordo's `ERROR` column is `sdev × confidence2x(0.95)` ≈ **1.96 σ**
  (`report.c:214`), i.e. already a 95% margin. Comparing it to a 1-σ standard
  error overstates its width by ~2× — verified by scaling `-F`.

**⚑⚑ Incremental fitting is VISIBILITY, not a verdict.** "Use all the games
without each pair being finished" is operationally right and the bootstrap
gives correct intervals *conditional on the pair counts in hand* — but reading
the verdict at whichever refit looks good is optional stopping, which has
already produced a +112 Elo read on a null and inflated a banked +239.5 to
+245.1 here. So: **pre-commit the total pair count per matchup and the read
point before launching**; look at partial fits freely for *operational* checks
(is an arm crashing, are counts advancing, has the missingness flag tripped)
and never for the sign or size of the effect.

**⚑ A bootstrap cannot fix informative missingness.** If pairs complete faster
in one matchup for a reason correlated with the arms, the pairs you HAVE are
that matchup's FAST pairs — systematically more decisive — and resampling them
returns a tight interval around a biased number. `ordo_pooled_fit.py` therefore
ALWAYS runs a completion-order-vs-outcome check and flags it (exit code 2);
`--pgn-out` records `Plies` and `GameDurationSec` per game so the check has
something to test. A promotion gate's apparent −26 Elo per publish was once
exactly this artifact, worth ~+50 Elo in the other direction.

**The guard's false-positive rate is corrected and MEASURED.** It runs two
tests per matchup and ORs across matchups, so uncorrected it is a family of
2×M tests at α=0.05. Measured on 400 true-null 3-arm datasets (outcomes drawn
independently of completion order): **0.273 ± 0.022 uncorrected** — better than
one clean fit in four told "do not read a verdict", which trains the operator
to ignore it. With the Holm step-down correction, applied ONCE over the whole
family rather than per matchup: **0.052 ± 0.011** against a nominal 0.05.
Reproduce with `scratchpad/fpr_measure.py <n_datasets> <n_perm>`, which calls
the shipped `completion_bias_report` rather than a copy of it.

⚑ **That rate is GRID-DEPENDENT — quote it with its `n_perm`.** The same
harness measures **0.033 ± 0.009 at n_perm=300**. Holm's strictest threshold is
α/m = 0.05/6 = 0.00833, while attainable permutation p-values are k/(n_perm+1):
at n_perm=1000 the largest usable k is 8 (region 0.0080), at n_perm=300 it is 2
(region 0.0066). The shipped default n_perm=2000 gives 0.0080, indistinguishable
from 1000, so **0.052 is the rate the tool actually ships with**. A coarse grid
can only lower the rate, never inflate it — it costs power, not safety. Two
independent measurements of this number disagreed until the grid was pinned
down, which is exactly why it is stated with its grid rather than as a bare
number.

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
- **⚑⚑ A RULER CHANGE INVALIDATES ITS RECORDS: rows (d) and (e) MOVED on
  2026-08-16 and are not comparable with any earlier reading.** The training
  rows were built from a hand-list of search fields that silently omitted three
  the production selfplay builder sets from the live yaml —
  `gumbel_policy_temp` (live **1.5**, audit used the `--policy-temp` default
  **1.0**), `gumbel_target_max_visit_cap` (live **5**, audit used **0**) and
  `gumbel_target_untempered_prior` (live **true**, audit used **False**). The
  last two are the ONLY two adjustments that separate the stored target
  `imp_store` from the play distribution `imp_all` in `mcts/gumbel.py`, so with
  both at their defaults the rows headed "production training target" were
  scoring the PLAY distribution. Any (d)/(e) number banked before that date
  describes a target production does not store; do not put one in a table,
  trend or threshold with a number from after it. The PLAY row (b) and the SF
  soft-target row (c) are unaffected.
- The training rows are now derived by calling production's own
  `build_selfplay_search_shape` rather than by re-listing its fields, and the
  realized-vs-production table is printed on every run under `[shape]`. A field
  the audit overrides without declaring it in `TRAIN_SHAPE_DEVIATIONS` stops
  the run. `--config` is checked FIELD-COMPLETE against the live yaml — both
  are pushed through production's builder and every field is diffed — so a
  search knob added to production later is covered the day it is added rather
  than when someone remembers to extend a list.
- ⚑ **The compared object is the RUNNER's argument set, not `GumbelConfig`.**
  "Field-complete over `GumbelConfig`" was itself a hand-drawn boundary:
  `gumbel_vloss_weight` and `gumbel_target_batch` reach the C search as
  keyword arguments with no `GumbelConfig` field, so a `--config` differing
  from live on `gumbel_vloss_weight` alone was not refused and was stamped
  authoritative — while `--vloss-weight`'s CLI default of `0` searched the
  duplicate-leaf shape against production's `1` on EVERY ordinary invocation.
  Both now default from the resolved production config and are part of the
  diff. `--vloss-weight` / `--target-batch` still take an explicit value for
  the C17 separating arm; an explicit value prints as a DELIBERATE deviation
  and lands on the ruler stamp.
  **Not covered, and said so on every run:** the per-ply root-noise schedule
  (`per_game_add_noise` / `per_game_gumbel_scale`) is a function of the move
  number and has no field to compare. `--search-shape training` arenas print
  what they do about it instead of implying coverage.
- **`config_authority.authoritative` means one checkable thing**, and
  `covers` on the same stamp says which: every value the script consumes from
  the config — the complete selfplay runner argument set, plus every key
  `audit_targets` reads directly (`AUDIT_DIRECT_CONFIG_KEYS`, whose
  completeness is regenerated from the module's AST by a test) — agreed with
  the live file by VALUE. It previously covered the `GumbelConfig` fields plus
  two sim budgets, so a stale `sf_policy_temp` produced a non-live SF soft
  target for row (c) under a stamp saying the config was proved production's.
- **Point `$CHESS_ANTI_ENGINE_LIVE_CONFIG` at the live yaml, or the audit
  refuses.** `scripts/train.sh` exports it; see `docs/operations.md` for the
  by-hand form. Without an authoritative reference the script will NOT emit
  rows (d)/(e), because the fallback it used to take — compare against the
  in-tree copy, warn, continue — reproduced the exact defect above from any
  worktree and printed "matches the live config" while doing it. The escape is
  `--allow-stale-config`, and it stamps `config_authority.authoritative=false`
  onto every row of `--dump-per-position` so the artifact carries the caveat.
- **`--dump-per-position` rows carry a `search_shape` stamp** — the COMPLETE
  realized training-search shape, read off the object the runner was handed
  (so it includes CLI overrides), and `scripts/paired_compare.py` lists it in
  `RULER_FIELDS`. Complete rather than "the three fields that were wrong":
  a three-field stamp only catches the ruler change that already happened, and
  moving `topk` or the sim budget afterwards would let two dumps pass their own
  live-config checks while emitting identical stamps. Fields read off the
  CHECKPOINT (the input/policy encodings) are excluded — `input_encoding` is
  already its own `RULER_FIELD` and stamping them would refuse legitimate
  cross-net joins.
  An unstamped dump predates 2026-08-16 and declares **UNKNOWN**, not a guessed
  shape: pre-fix, `--policy-temp` was an operator-settable flag that fed the
  training rows, so a legacy dump made at `--policy-temp 2.2` is not deducible
  as `1.0`. Unknown compares equal only to another unknown, so pre-fix-vs-
  post-fix is still refused and two pre-fix dumps still compare.
  The stamp is checked only for metrics it governs (`cand.train*`); comparing
  `cand.raw.exp` or `cand.sf_soft.exp` across a legacy and a current dump is
  not refused over a training-row ruler that cannot touch either number.
- **Scoring a FOREIGN net (LC0/BT4, Ceres) on the same ruler:** pass `--onnx
  <net>.onnx` instead of `--checkpoint` to either `scripts/audit_targets.py` or
  `scripts/value_regret.py`. The two flags are mutually exclusive and exactly
  one is required — there is no default and no fallback, so a run cannot be
  ambiguous about the weights behind its number, and the resolved net is
  echoed on the report header and into `--dump-per-position`. The foreign net
  goes through `chess_anti_engine/onnx/load.py`, which slices our planes to
  LC0's 112, fills the LC0 history, and remaps Leela's 1858 policy ordering
  into ours PER POSITION (`moves/leela_index.py`) — our `lc0_1858` and Leela's
  agree on only 46 of 1858 slots, and castling/promotion cannot be mapped by
  any static table. ⚑ An LC0 net declares `lc0_root`/`v1` planes, so
  `--input-encoding stored` (audit-v2, 175-plane production rows) refuses it;
  compare foreign and own nets under the DEFAULT `fen_only` with a pinned
  `--batch-size`, and remember `--device cpu` is the only setting that
  structurally cannot allocate on the training GPU.
- **`--gpu-mem-fraction` on `--onnx` caps onnxruntime, not torch.** Both
  rulers apply the fraction to two SEPARATE allocators, because a foreign net
  is not a torch module: `torch.cuda.set_per_process_memory_fraction` bounds
  the torch caching allocator, and ORT's CUDA arena is bounded only by
  `gpu_mem_limit` in the CUDA provider options (the shape
  `scripts/foreign_net_audit.py` has always used). The log says which is
  which — `TORCH GPU allocator capped ...` at parse time, and
  `onnxruntime session on [...]; CUDA arena capped at N bytes` once the
  session exists — and a fraction on `--device cpu` prints `IGNORED` rather
  than being dropped in silence. ⚑ A run on `--device cuda...` whose ORT
  session comes up without `CUDAExecutionProvider` ABORTS: ORT drops an
  unusable provider with a warning and runs on CPU, so the number would be a
  CUDA-labelled CPU number. **`onnxruntime.get_available_providers()` cannot
  see that happen** — it is the wheel's COMPILE-TIME list, and it can name a
  provider that does not start (observed here: it reports CUDA while every ORT
  session seen on this box has come back CPU-only; note there are TWO ORT
  installs, a CPU-only wheel in the project `.venv` and a GPU wheel under
  `/usr/bin/python3`, so which one a bare `python3` gets depends on venv
  activation). The verdict is read off `session.get_providers()`, twice: a
  76-byte throwaway probe session at PARSE TIME — before the audit set and
  before Stockfish labels anything — and again on the real scoring session.
  The probe costs one onnxruntime init (150-240ms cold, 20-50ms after) and is
  paid ONLY by `--onnx --device cuda...`; `--device cpu` and every
  `--checkpoint` run never build it.
- Two standing numbers to watch on each report: (search-policy regret) vs
  (SF-soft-target regret) per phase — when search wins everywhere, the
  50k-node MultiPV-40 labeling is no longer worth its CPU bill — and
  (production WDL blend) vs its best single component — when a single
  component matches the blend, the blend weights are dead complexity.
