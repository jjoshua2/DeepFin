# 2026-11-21: RL run 1 — full-width root round (`--root-full-width`): cost, target audit, search strength, training test (pre-registration)

**Status: registered, not launched.**
- **Stages 0–2** need two things before they start: this record is reviewed and merged, and the GPU queue has
  released the card. Every launcher refuses until both queue markers exist (see [the GPU guard](#the-gpu-guard)).
- **Stage 3 is NOT AUTHORISED.** It needs the owner's go-ahead. The coordinator records it in the bank as
  `stage3/OWNER_GO`, and one command then runs the whole stage.

**Queue position:** after the value-anchor screen (#1475) and the sibling-ordering screen (#1481), unless the owner
reorders.

**Recovery:** nothing live changes. No yaml, no live tree and no deployed code is involved. Every output goes to a
new bank.

## Why

The owner's hypothesis, verbatim:
- "perhaps if we force our gumbel to eval all nodes at root that we can focus on correct ordering?"
- "if we just do all it makes batching way easier and gpu will load better so we could also increase node count
  from 64 to like 74 and probably still come out ahead"
- "just this change should help some i believe with value head"

The knob exists on main since #1477 (merge `fd8c0ce08`) as `GumbelConfig.root_full_width`, exposed as the collect
flag `--root-full-width`. It is default off.

**What it does:**
- **Round 0** visits every searched root child that has 0 visits, once each, before sequential halving starts.
  - The top-k candidates are then re-chosen by gumbel + log prior + σ(completed q).
  - The halving budget is unchanged.
  - Round 0 is skipped on fast plies.
- **Accounting.** The round-0 simulations are spent **on top of** `--sims`.
  - They are stored per row as `root_prelude_visits`.
  - The stats field is `mean_root_prelude_visits`.

**Spec deviation, stated once and carried everywhere.** The owner pictured round 0 as one wide batch. The
implementation drives the unchanged C `start_gumbel_sims` state machine instead. It packs round 0 first-fit into
calls of at most the existing leaf buffer (`len(enc_buf)`, 4,096 rows at 3 × 128) and at most `GSS_MAX_CANDS` = 64
children per board per call.

**Is the 4,096-row buffer binding?** Measured on CPU from the pilot's own roots (`sizing/results/round0_fit.json`):
- The searched support is 27.43 children per root (median 29, p90 43, max 68). Only 12 of 752,885 rows exceed 64.
- **A typical-call estimate under independent sampling**: sum the supports of 128 rows drawn at random from the
  whole collect, ignoring carried roots. The mean is 3,510, SD 141 and p99.9 3,939. P(> 4,096) is 2e-5.
- **This is not an upper bound.** All 128 slots start together from 16-ply openings, so early in a run the boards
  are in phase, at middlegame plies where the support is widest. The review's computation (not banked) found:
  - with 128 boards at ply 30 and full supports, the sum averages 4,985 and exceeds 4,096 every time;
  - with the same boards but carried roots (only unvisited children are evaluated), the mean is 3,692 and
    P(> 4,096) = 0.0004.
- **So the buffer splits a round-0 call rarely: mostly in the synchronised opening phase, and even there in about 1
  call in 2,500 once carried roots are counted.** A split costs one extra GPU call on that ply. Enlarging the buffer
  is not part of this test.
- **Stage 0 sees more of the synchronised phase than a real collect does.** A run is 384 games per process, about
  three waves of 128, against about 21 waves in an 8,000-game collect.

**What the CPU smoke already shows, so the owner's cost guess is checked rather than assumed.** On 32 cold rl-audit
roots (`stage1/smoke_cpu/`; a smoke, not a result):
- Knob on at 64 sims spends **94.1** root visits per position, against 64.0 with it off: round 0 visited 29.2
  children.
- In the collect, carried roots need fewer round-0 visits.
- "64 → 74" therefore cannot be assumed to cost about the same. **The equal-cost sims count `S_eq` may be below 64.**
  Stage 0 measures it.

## Prior evidence (grepped first; [evidence log](2026-11-21-rl-loop-evidence-log.md), bank `INDEX.md`)

- **The pilot's searches leave about half the root unvisited.**
  - At 64 sims, 14.53 of 27.43 searched children are visited.
  - The prior mass on unvisited children is small: mean 0.0041, median 0.00021. It exceeds 0.05 on 2.1% of rows.
  - So round 0 mostly buys q-values for moves the prior already rates near zero. Its candidate re-selection matters
    only where one of those moves is actually good.
- **Search-ceiling and search-gain banks.**
  - Search gains from the SL base are large: 64 vs 1 is +849, 256 vs 64 is +272.
  - **A net's own deeper search is NOT an admissible ruler for its target.** In
    `self_search_ruler_inverts_vs_independent_refs`, the stored 64-sim target looked +8.4 pp better by the net's own
    R800 but only +0.7 by BT4.
  - So Stage 1 uses deep SF 8M and the searched Ceres ruler, never the SL base's own deeper search.
- **Policy-target zeros (`policy-target-zeros/`).**
  - The stored target's near-zeros are mostly right on deep SF.
  - Search does push some SF near-equal alternatives below 1e-4: 9.2% [7.5, 10.8] of them, vs 1.6% for the prior.
  - The full-width round could change exactly this. It is a Stage 1 secondary measure.
- **Searched ordering vs deep SF (`sibling-target-scoping/`).**
  - On clear pairs (gap ≥ 50 cp), R64n q orders children better than the raw net: +0.0274 [+0.0180, +0.0365].
  - On near-ties it is a coin flip.
  - The knob gives every child at least one visit, which widens the set this applies to. Stage 1 measures that.
- **Collect throughput (`collect-throughput/`, [simple_rl_loop.md](../simple_rl_loop.md)).**
  - 3 × 128 at 64 sims: 5,021 games/h, 9,262 forwarded positions/s, GPU 87%, 21.9 GB.
  - The repeat agreed within 0.5% (4,993.6 vs 5,021.4).
- **Pilot collect timing.**
  - The pilot's `stats.json` wall fields are broken: `wall_seconds = active_seconds = 2.33e-06`. That attempt was
    a merge-only reseal after the seal step was OOM-killed.
  - Collect time is therefore taken from the per-process walls (5,512 / 5,534 / 5,471 s) and the stamps:
    - `collect.started` 06:41:24;
    - first attempt exit 137 at 08:26:46;
    - reseal 09:51:00 → 09:54:42.
  - **Play time is 1.54 h (slowest process), 5,204 games/h.**
  - The stats bug still existed on main and is fixed separately in #1483, with a test.

## The GPU guard

Every launcher in the bank sources `gpu_guard.sh` v2 (sha256 `997969c87bf3…`; v1 `a0a2f869a719…`). It refuses (exit 8) unless **all** of
these hold:
- **Memory.** `nvidia-smi --query-gpu=memory.used` is below 1,024 MiB.
  - **Not `--query-compute-apps`.** Under WSL2 that list cannot be relied on to see another process tree. During
    the incident below it listed this bank's three collects in 62 of 64 samples, but it never listed match 8's
    PID 2337486, which was using the card the whole time.
- **No running GPU work.** An exact `ps -u <me> -o args` scan finds no `arena_standard.py`, `simple_rl_loop.py` or
  `match_vs_uci.py`. The match is on the script's basename as an argv token; `pgrep -f` is never used.
- **The queue has released the card.** Both markers exist: `rl-run1-followup/chain.exit` and
  `rl-run1-value-anchor/queue2.exit`.
- **The operator released THIS stage.** The bank holds a file `GPU_RELEASED_BY_OPERATOR` with a line whose first
  word is the stage the launcher requests (`stage0` … `stage3`), followed by a date, e.g.
  `stage0 2026-11-22T09:00`.
  - **Queue order is enforced by that file, which the operator writes. It is never inferred.**
  - The markers cannot do this job. The sibling screen (#1481) runs after queue 2 and has no marker this guard
    could know about. And between a chain's steps, memory drops below 1 GiB.

It then takes a non-blocking `flock` (exit 9 if held) on the bank's own `.gpu_launch.lock`, so two of this bank's
launches cannot overlap. It never opens `~/projects/chess/scratchpad/gpu0_experiment.lock`.

**Tested** (`guard_test/guard_test.sh` v2 `730d515388e3…`, output `guard_test/guard_test.txt`; v1 and its output are kept as
`.v1`). Nothing in the test can reach the GPU: a fake `nvidia-smi` and `ps` sit first on PATH, HOME is a fake, and the
launchers' python is replaced by a stub. For a collect call the stub records its arguments and fabricates a tiny
finished collect; for the launcher's summary step it runs the real python. All 18 cases pass:
- memory 10,971 MiB, with every other check clear → refused for memory (the test greps "GPU memory in use");
- memory unreadable → refused for memory;
- a missing queue marker → 8;
- each of the three script names running → 8;
- the name present only as a substring → passes;
- all clear → passes;
- no operator release file → 8;
- a release file without the requested stage → 8;
- a release file with the requested stage → passes;
- a stage name outside stage0–3 → 8;
- the lock held → 9;
- `run_stage0.sh` end to end with the stub:
  - three collect calls, each carrying `--root-full-width`;
  - exit 0;
  - a summary row whose steady-state rows/s comes from the games-progress log;
- `run_stage0.sh` with memory busy → 8, and the stub was never called.

**Mutants** (`guard_test/guard_mutants.txt`; the restore was checked by md5):
- with the memory check disabled, the two memory cases fail;
- with the release-file stage check disabled, the "without the stage" case fails.

## Stage 0 — cost (GPU)

**Question.** What sims count `S_eq` makes knob-on cost the same wall per stored position as knob-off at 64?

**Design.**
- The pilot's collect flags (`--full-prob 1.0 --topk 16 --c-scale 0.1 --c-visit 50 --max-plies 450`, production
  Syzygy pair, the pilot's SF and openings) and the SL base (`a39f900c…`).
- Main `fd8c0ce08`, in the detached worktree `rl-run1-rfw-arena-fd8c0ce08`.
- **One run** is three independent collects × 128 games in flight × 384 games each. This is the collect-throughput
  bench's method (`mp_sims.sh`), and the doc's table says it is what `--collect-procs 3` launches.
  - Forward rows are counted by `count_wrap.py`, a byte copy of that bench's counter (sha256 `fcb2861aa1a9…`).
  - nvidia-smi samples every 500 ms.
- **Every run uses the same seeds (2026112121/22/23) and the same openings.** Only `--sims` and the knob change.
- **Measured per run** (`stage0/summary.jsonl`):
  - **steady-state stored positions/s, the registered cost quantity.** For each process: the rows of the games
    that finish between its 10% and 90% game counts, divided by the time between those two counts. The counts come
    from `logs/games_progress.txt`, which samples each process's `games.jsonl` line count every second. The three
    processes are summed.
    - This excludes model load and the low-concurrency tail. Rows over the whole wall would include them and pull
      the ratio toward 1, by about 1–2%, which is worth about 1 sim in the knob's favour.
    - Rows over the whole wall are reported beside it.
  - forwarded positions/s;
  - forwarded evaluations per stored position;
  - evaluations per position computed as sims + mean prelude, from the search's own `mean_root_prelude_visits`;
  - mean eval batch size;
  - mean and peak GPU utilisation and memory;
  - `buffer_full_flushes`.

**Take-effect checks.** Every run must pass all three, or it fails:
- the search's own `mean_root_prelude_visits` is > 0 when the knob is on and absent when it is off (read from what
  the consumer recorded, not from the flag passed);
- the manifest reads back `root_full_width`;
- `buffer_full_flushes` = 0 and `eval_max_batch_realised` = 4,096 on every process.

**The rule** (`stage0_decide.py` v2 `4c02fc7565dd…`, applied mechanically; v1 `fb099bed9a35…` is kept as `.v1`). Cost per stored position is
t = 1 / steady-state rows/s.
1. **Step A.** Run off@64, on@64, on@64, off@64, in that ABBA order to cancel drift.
2. **Repeat gate.** Two runs of one configuration that differ by > 2% get exactly one more run, and the
   configuration's t is then the mean of three.
   - Two runs within 2% put each mean within about ±1%, so the ratio is known to about ±3%. The bench's own repeat
     agreed within 0.55%.
3. **Step B.** Compute S1 = round((64 + p) · t_off / t_on64 − p), where p is the mean prelude at on@64, clamped to
   [16, 128]. That is where knob-on would cost t_off if a position's cost were proportional to its evaluations.
   - Run on@S1 once. If |t_onS1 / t_off − 1| ≤ 0.015 after that one run, then **S_eq = S1**.
   - Otherwise run on@S1 a second time. If the mean of the two runs is within ±0.03, then **S_eq = S1**.
4. **Step C**, only if step B misses. One step: S2 is the point where the straight line through (64, t_on64) and
   (S1, t_onS1) reaches t_off. Run on@S2 twice. **S_eq = S2**, final, with its measured ratio reported.
5. If S1 = 64, then S_eq = 64.

**Why ±3% is precise enough.** The SL base gains ≈ 136 Elo per doubling of search: 256 vs 64 is +272 over two
doublings.
- A 3% cost error is therefore worth ≈ 136 × log2(1.03) ≈ 5.8 Elo of search.
- A 6% error is worth ≈ 11 Elo, about the size of Stage 2's −10 kill margin.
- The measured ratio at `S_eq` (`ratio_at_S_eq`) is reported next to Stage 2 (b).

**Tests.** `stage0/fab_test/fab_test.txt` (8/8 pass) covers each branch of the rule, including step B's one-run early
stop, and the flush gate. `driver_test.txt` runs the full driver against a fake `run_stage0.sh` and stops after one
step-B run. The v1 outputs are kept as `.v1.txt`.

**Report-only.** Mean eval batch, GPU utilisation and evaluations per position, knob on vs off. These answer the
owner's batching claim directly.

**Cost.** One run is ≈ 15–18 min: the bench's 3 × 512 took 18 min, and the knob adds round 0. Five to eight runs
make **≈ 1.3–2.4 GPU-h**.
- **This deviates from the coordinator's ≈ 15 min.** One run per arm cannot show that the ratio is known to ±3%;
  the repeat gate needs at least two runs per configuration.

## Stage 1 — audit first (CLAUDE.md rule 5; GPU ≈ 0.15 h, then CPU)

**Rows.** The 2,433 leak-free rl-audit rows from 190 games, scored. The search runs on all 3,080.

**Arms.** All are searched with the SL base at main `fd8c0ce08` by `stage1_search.py` (`b928707dee79…`), which is
adapted from the rl-audit `search_bank.py` that produced R64n. These are kept verbatim from it:
- positions;
- the rl-audit protocol's search/game/serving config;
- root logits pre-computed from the stored x;
- the strict rule50 TB probe;
- 16 boards per call on a cold tree, noise on, seed 1.

The arms:

| Arm | Sims | Knob | Role |
|---|---|---|---|
| `M64off` | 64 | off | control (same code and seed as the knob arms) |
| `M64on` | 64 | on | report-only |
| `MSEQon` | `S_eq` | on | **the registered arm** (the Stage 3 arm) |
| `R64n_banked` | 64 | off | the banked rl-audit target; the reproduction gate |

Main's mapping reads two fields that the rl-audit protocol predates: c_visit 50 and q rescale on. These are the
values the GumbelConfig defaults gave R64n, and the script exits unless they took. The knob goes through the
production `SearchConfig` field. The script logs each arm's `GumbelConfig`, which is the consumer's own object. It
exits if knob-on records no prelude visits, or if knob-off records any.

**CPU smoke** (32 positions, `stage1/smoke_cpu/smoke32_check.txt`):
- `M64off` agrees with banked R64n on top-1 for 32 of 32 positions; L1 0.026, from fp32 vs bf16.
- Knob on visits 30.1 children per row vs 14.8.

**Scoring.** `stage1_score.py` (`b1e62b3ab729…`), on CPU. Every row-level measure is banked per row with its row
index and game id (rule 4). Paired differences use a game-clustered bootstrap (2,000 resamples, seed 20261121).
- **Deep SF 8M.** `plan-review-1465/sf8m_table.npz`, with `policy-target-zeros/code/part23_audit.py`'s definitions
  exec'd verbatim:
  - **expected cp loss, lower bound (`exp_cp_lb`): the PRIMARY measure**;
  - SF-best probability;
  - top-1 SF-best;
  - cross-entropy to SF-best;
  - rows giving the SF-best set < 1e-4;
  - mass on unscored moves;
  - the share of SF near-equal alternatives (≤ 10 cp, excluding the target's top-1) pushed below 1e-4.
- **Independent net ruler.** "Ceres C3 (Mish-rewritten net, 98.5% best-move agreement with the original)" @ 256,
  from `ceres-ruler/raw`, with `wide_table_v3.py`'s formulas: top-1, argmax loss, and expected loss over covered
  mass.
- **Root-child VALUE quality.**
  - Mean |q_child − q_SF(child)| over SF-scored children, where q_SF is W − L of the production cp-logistic (slope
    0.006, draw width 120). It is reported over each arm's own visited set and over the set every arm visited.
  - Clear-pair ordering (SF gap ≥ 50 cp, not both mate scores), as in `searched_order_vs_sf.py`:
    - (a) on the pairs `M64off` visited, with every arm scored on the same pairs;
    - (b) on **all** SF-scored legal pairs, each arm on the pairs it visited.
  - A knob-on child with one visit has the raw net's value. So (b) mixes search quality and coverage, and the pair
    counts are reported beside it.

**Scorer validation.** `stage1/dryrun_banked/` scores banked R64n as "off" and R64n_seed2 as "on". It reproduces
the seed-noise numbers below exactly, and the reproduction gate reads top-1 1.000 with paired 0.0. It is a
validation, not a result, and the decision script refuses it.

**The rule** (`stage1_decide.py` `baafe67e25f6…`; `stage1/fab_test/fab_test.txt` 7/7 branches pass):
1. **Code-base control.** `M64off` must reproduce banked R64n: top-1 agreement ≥ 0.95, and a paired
   `exp_cp_lb` CI that contains 0. Otherwise **UNREAD**, stop.
2. **Coverage.** The paired mean of mass on moves deep SF did not score, `MSEQon − M64off`, must be ≤ +0.01.
   Otherwise **UNREAD** until an SF top-up amendment. The reason: `exp_cp_lb` gives unscored moves the 10th-best
   regret as a lower bound, so extra unscored mass would flatter the knob.
   - The banked level is 0.018 for R64n and 0.051 for the prior. The review estimates ≈ 0.0084 for both arms, so the
     gate is not expected to fire.
   - **If it does fire, the top-up procedure is this.** Every move with ≥ 1e-3 mass in any Stage 1 arm that is
     missing from `sf_rows.json` is labelled with `search-compose/scripts/sf_label.py`'s protocol: 8M nodes, cold TT,
     production Syzygy, `searchmoves`. The result goes into a copy of the SF table, and the arms are rescored. It is
     a new amendment, never a silent rerun.
3. **Kill: STOP iff** the paired `MSEQon − M64off` on `exp_cp_lb` has LB95 > 0 **and** its estimate is above
   +1.2 cp.
   - 1.2 cp is the 95% half-width of the paired difference between two 64-sim targets that differ only in the
     search seed: R64n − R64n_seed2 = +0.05 [−1.10, +1.20] (`sizing/results/seed_noise.json`).
4. Otherwise **PROCEED**.

**What the verdicts license.** "Worse" licenses stopping. "Better" licenses only proceeding, never an Elo claim:
there are three fit→Elo transfer failures on record (v020, ft595, F-only) and the ladder's sharp-target inversion.
`M64on` and every secondary measure are report-only.

## Stage 2 — search strength (GPU ≈ 3.0–3.6 h; report-only for the training question, one kill)

**Arena commit.**
- The knob is not at the pilot's PIN, so Stage 2 runs at **main `fd8c0ce08`**, from the detached worktree
  `rl-run1-rfw-arena-fd8c0ce08`, through the follow-up launcher's main-pinned twin `run_rfw_match.sh`
  (v2 `0487d857e379…`, which adds the stage-2 release line; v1 `0c192e052082…`).
- The twin keeps the follow-up launcher's flags byte for byte, and its guards:
  - refusal to write into a frozen bank, exit 94 (now the pilot bank **and** the follow-up bank);
  - never resume a look whose game log exists, exit 95;
  - worktree at the pinned commit with no tracked edits, exit 96;
  - `cd` failure, exit 97;
  - a `.exit` stamp;
  - plus the GPU guard.
- Exits 94, 95 and 97 were exercised (`fab_test/launcher_guards.txt`).
- **Main's training shape is the PIN's.** A CPU dry resolution at `fd8c0ce08` prints the same realised gumbel
  fields as match 5's banked row, plus `root_full_width`, and the in-tree yaml has the same sha256 (`8275f2b84d0e`)
  at both commits.

**Design.**
- SL base vs itself, knob on vs off.
- `--search-shape training`, no noise, `--compile off`, the production Syzygy pair for both sides, the pilot's
  `eval.pgn` at 16 opening plies, `--max-plies 600 --on-max-plies abort`.
- Seat-balanced: the knob seat is the candidate in run A (`--cand-gumbel root_full_width=1`) and the reference in
  run B (`--ref-gumbel root_full_width=1`). 3,200 games per run, 6,400 per match.
- The two matches:
  - **(a) `rfw_matched`**, matched sims 64 vs 64, seed **2026112152**.
    - **This match is NOT at equal cost.** On the arena's cold trees, the knob seat gets ≈ 46% more root
      evaluations: 94 vs 64 root visits per position in the CPU smoke.
  - **(b) `rfw_eqtime`**, equal GPU time, `S_eq` vs 64, seed **2026112153**.
    - The equal-cost point is the one measured in the collect, where carried roots are reused. On the arena's cold
      trees the knob seat still evaluates somewhat more, so (b) leans slightly toward the knob.
    - Stage 0's `ratio_at_S_eq` is reported beside this match.
  - Neither seed appears under `docs/` or in any bank.

**Seat proof.**
- `check_shape.py` (`95ee428f58f9…`) reads each run's result row and requires:
  - the knob seat realised `root_full_width=True` and the other seat `False`;
  - every other realised gumbel field, plus `vloss_weight`, `target_batch`, `tree_reuse` and shape, on both seats
    equal to match 5's banked row (which ran at the PIN);
  - the registered sims on each seat;
  - no value hybrid, and no noise.
- Fabricated-row tests pass 6/6 (`fab_test/fab_test.txt`): knob on the wrong seat, a changed c_scale, wrong sims,
  a missing knob and a warm tree all fail.
- **Take effect, observed in the search** (`stage2/run_stage2.sh` v2 `c07fe428aa39…`). gumbel_c logs once per process "disabling
  the 2-group eval pipeline for this call because root_full_width is on". It does so on a call of ≥ 64 boards, and
  only the knob seat's config can trigger it.
  - After each real run, the knob run's log is grepped for that line, and `TAKE_EFFECT_{A,B}.txt` records OBSERVED
    or UNOBSERVED.
  - A real run without it makes Stage 2 **"unread: take effect unobserved"**. No kill fires, and Stage 3 refuses.
  - The arena does not expose per-seat forward counts. Banking them would need a code change, so this record uses
    the log line.
  - The 4-game smoke has too few boards per call to show the line, so it is not read there.
- A 4-game smoke pass of both matches, with the pool, runs first. The real pass runs only if every smoke step exits
  0. That includes the pool's seat-swap check on a pair whose two seats are the same file.

**Rules.**
- Kill/void as the follow-up record's "Common" section: the launch record's max-plies rule, any non-zero arena
  exit, and any failed readback.
- Read on `pool.json` → `result.pooled.elo_ci95`, with the arm being the knob seat.
- **Kill: STOP iff match (b)'s UB95 < −10.** At equal GPU time the knob would then make the agent clearly weaker:
  by more than the −10 adoption margin every match here uses, at 95%.
  - Why −10 and not 0: an arena measures the PLAY regime on a cold tree (the
    `gumbel_cscale_miscalibrated` trap), and the training temperature exceeds the strength-optimal one
    (`training_temp_exceeds_strength_optimal_temp`). So a small loss in play does not speak to target quality.
  - Only a clear loss at equal cost means the selfplay games themselves come from a weaker player for the same
    compute.
- Match (a) and every other reading are **report-only**. This search is not tuned toward the play optimum
  (production c_scale 0.1 is chosen for target quality).

**Power.** At the measured match-scale SE (≈ 3.3 per 6,400 games), the 95% interval is ≈ ±6.5 Elo.

**Cost.** Match 5 took 78 min (38.5 + 39.5) at 64 vs 64 with the 2-group pipeline on both seats. In these matches
the knob seat loses the pipeline and evaluates more: (a) ≈ +46% at the root, (b) less.
- Estimate: ≈ 1.6–1.9 GPU-h per match, ≈ 3.0–3.6 GPU-h with the smokes.
- The arena's own timings are reported.

## Stage 3 — the training test (NOT AUTHORISED; needs the owner's go)

**One command:** `stage3/launch_stage3.sh` (v2 `a519b39f2200…`; v1 `95c8a66dbc59…`).
- It refuses without `stage3/OWNER_GO` (exit 90, tested), a Stage 1 PROCEED, Stage 2 (b) UB95 ≥ −10, Stage 2's
  take effect observed on both (b) runs, and the operator's `stage3` release line.
- It checks every instrument's sha, sources the GPU guard and holds the bank lock to the end.

**The registered arm: knob on at `S_eq`.** The owner's claim is that the knob comes out ahead at equal GPU cost.
Knob-on at 64 would confound the knob with ≈ 1.3–1.5× the evaluations.
- If `S_eq` ≠ 64, the arm changes the root's width and its sims together. That pairing is the owner's proposal
  itself ("increase node count … and still come out ahead"), registered as ONE data-affecting change; see Confounds.

**Steps**, each stopping the run on failure:
1. **Collect at `fd8c0ce08`**, identical to the pilot's command (same openings, seed 2026112100, 8,000 games, 3 ×
   128, same flags) except `--root-full-width --sims S_eq`. Take effect: the collect's `mean_root_prelude_visits`
   > 0 and 0 buffer flushes.
2. **Train at `fd8c0ce08`.** The pilot's `train_cell.sh` flags verbatim, plus `--w-sibling 0` (main's default, made
   explicit). Three cells:
   - `P0_rfw`: `--w-wdl 0`, on the new collect;
   - `Zq_rfw`: 0.5 outcome / 0.5 stored search WDL, on the new collect;
   - `P0_ctl`: P0's flags on the **pilot's** collect, the code-base control.
   - Between the sibling screen's training commit `febed2042` and `fd8c0ce08`, the trainer path differs only in the
     new optional field `root_prelude_visits` (checked with `git diff`).
3. **Guard (d).** Each main-written checkpoint loads and evaluates identically at the PIN, checked with the sibling
   screen's `check_load.py` (`e8a870764bb5…`).
4. **Stage A instruments by sha**, at the PIN: the value-anchor screen's banked `run_fits.sh` (`5017cea31e6e…`),
   `collect_logits.py` (`956c238f88c7…`), `analyze_holdout.py` (`3291449af64b…`), `supp_outcome_distance.py`
   (`2ad7f272d00a…`) and `sibling_order.py` (`17eeb159f691…`). The bank's `SHA256SUMS` is checked first.
   - The hold-out rows are the pilot collect's hold-out games. Neither knob-on cell trained on them, so for those
     cells this is a check on a different collection.
   - **Veto,** as the anchor screen: a cell whose clear-pair sibling-ordering UB95 (cell − base) is < 0 gets no
     match.
   - **Code-base control** (the sibling screen's guard (c) rule): |P0_ctl − banked P0| ≤ √2 × the banked P0's 95%
     CI half-width on the clear-pair difference, `KL_base_net_raw` and `share_of_gap_closed`. A failure makes every
     Stage 3 match **unread**.
5. **Sibling sidecar re-materialised** on the new collect (a separate CPU command, below; ≈ 30 min, ≈ 63 GB) with
   `scripts/simple_rl_sibling_pairs.py` at defaults (K 2, min visits 2). It is for later screens only; no cell here
   trains on it. See Confounds.
6. **Matches vs the SL base at the PIN**, using the follow-up launcher (`7c8fd065…`) with output root `stage3/`.
   - Plain seats, seat-balanced, 6,400 games each.
   - Seeds **2026112162** (`P0_rfw`) and **2026112163** (`Zq_rfw`).
   - The smoke pass runs first.
   - Match 5's rules: damage iff UB95 < 0; adoptable iff LB95 > −10; "improves on the base" iff LB95 > 0.
   - The seat proof is the pool's row-sha256 cross-check, as for match 5's plain seats.

**Dose, reported.** Stage 3 fixes 8,000 games, as the pilot did, not rows or GPU time. If the knob changes game
length, the arm gets more or fewer rows and GPU-h than the pilot's 761,979 rows and 1.54 h of play.
- Both are reported: rows and rows per game, played GPU-h, and the training updates they yield.
- See Confounds.

**Comparison with the pilot's banked arms.**
- Banked P0 as a single net (match 5): +2.2 [−4.2, +8.6], SE ≈ 3.3.
- Banked Zq (match 1, the gate): −10.0 [−23.2, +3.2], SE 6.74 at 1,600 games.
- Recalibrated Zq per match 8, when it is read.

**Power.** The SE of the difference of two independent 6,400-game matches is √2 × 3.3 ≈ 4.7 Elo.
- The 95% half-width of the difference is ≈ 9.2 Elo.
- The difference detectable at 80% power (two-sided 5%) is ≈ 13 Elo.
- Against the 1,600-game Zq gate the difference SE is ≈ 7.5, so detectable at 80% is ≈ 21 Elo.
- **Promise:** "the knob helps training" only if `P0_rfw` − banked P0 has LB95 > 0. A true effect below ≈ 13 Elo is
  likely to read as unread. Nothing smaller is claimed.

## Confounds

- **Code base.**
  - Stages 0–2 and Stage 3's collect and training run at main; every banked comparator ran at the PIN.
  - Bridging Stage 1: `M64off`'s reproduction of R64n.
  - Bridging Stage 2: the realised-shape readback.
  - Bridging Stage 3: `P0_ctl` and guard (d).
- **Width and sims move together** in the registered arm when `S_eq` ≠ 64. This is unavoidable at equal cost. The
  matched-sims report (`M64on`, Stage 2 (a)) separates them descriptively.
- **Sibling sidecar min-visits.** With the knob on, a candidate eliminated after the first halving round has 2
  visits (round 0 + round 1). So `min_visits=2` admits children that the knob-off collect would never pair: pairs
  per row, the children's visit mix and dq noise all shift.
  - A sibling cell trained on this sidecar is not comparable to one trained on the pilot's sidecar without
    re-deriving the strengths.
- **Cold trees in Stages 1–2 vs tree reuse in Stage 3** (reused-root share 0.646). Round 0 evaluates fewer children
  on carried roots, so Stages 1–2 overstate the per-position round-0 work. Stage 0 measures the collect's real
  share.
- **Dose.** Stage 0's unit is cost per stored position. Stage 3's dose is 8,000 games. If the knob changes game
  length or the decisive/draw mix, rows, training updates and GPU-h all differ from the pilot's, and the training
  comparison carries that difference. Both are reported; neither is equalised after the fact.
- **Stage 1 search config** is the rl-audit protocol's (gumbel scale 0.75, serving max batch 512), chosen to match
  banked R64n. Stage 3's is the pilot's.

## Cost and order

| Item | GPU-h |
|---|---|
| Stage 0: 5–8 runs × ≈ 15–18 min | ≈ 1.3–2.4 |
| Stage 1: three searches on 3,080 rows (R64n took 91 s; the knob adds ≈ 40%) + model load | ≈ 0.15 |
| Stage 2: two 6,400-game matches (match 5: 78 min; the knob seat loses the pipeline and evaluates more) + smokes | ≈ 3.0–3.6 |
| Stage 3: collect (equal cost to the pilot's 1.54 h play + seal) | ≈ 1.6 |
| Stage 3: training P0 / Zq / P0_ctl (pilot: 11 / 14 / 11 min) | ≈ 0.6 |
| Stage 3: Stage A, 3 nets (fits + logits + sibling order) | ≈ 0.3 |
| Stage 3: two 6,400-game matches + smokes (no knob in these matches) | ≈ 2.7 |
| **Total** | **≈ 4.5–6.2 (Stages 0–2) + ≈ 5.2 (Stage 3)** |

CPU: Stage 1 scoring takes ≈ 10 min, the Stage 3 sidecar ≈ 30 min. Disk: ≈ 63 GB for the sidecar plus ≈ 30 GB for
the collect; ≈ 800 GB is free.

## Option, not registered and not built: a budget-neutral round 0

This is the owner's idea in a form that adds no evaluations.
- **What it is.** Round 0 spends the first halving round's budget on every legal child, instead of adding a
  prelude, so the total stays at `--sims`. With k children and a first-round budget below k, the remaining rounds
  are shortened to fit.
- **Code it would take.** `_full_width_round0` in `mcts/gumbel_c.py` would charge its visits against the halving
  schedule instead of adding to it. The C `start_gumbel_sims` state machine would then need to start from a reduced
  budget with a pre-chosen candidate set, or take a per-board budget offset. Tests would need to pin the total root
  visits at `--sims`.
- **How Stage 0 decides whether it is worth building.** If `S_eq` comes out well below 64 (say ≤ 48), the additive
  form pays for its width with real search depth, about 0.4 doublings ≈ 55 Elo of search at 136 Elo per doubling.
  The budget-neutral form then becomes the version worth registering.
  - If `S_eq` is near 64, the additive form costs little and this option is moot.

## Commands

```bash
B=$HOME/chess-artifacts/operations/rl-run1-root-full-width
AW=$HOME/projects/chess-worktrees/rl-run1-rfw-arena-fd8c0ce08; PY=$HOME/projects/chess/.venv/bin/python
(cd $B && sha256sum -c --quiet SHA256SUMS) || exit 93
cd $AW && [ "$(git rev-parse HEAD)" = fd8c0ce08e1a0829e051cf1a6fef07fd6e83c244 ] || exit 96
$PY scripts/build_production_extensions.py || exit 1          # the worktree's .so files, at this commit
# Every GPU stage first needs the OPERATOR's line in $B/GPU_RELEASED_BY_OPERATOR, e.g. "stage0 2026-11-22T09:00".
# Stage 0 (writes stage0/decision.json)
$B/stage0/run_stage0_all.sh || exit 1
# Stage 1 (GPU search, then CPU score and verdict)
$B/stage1/run_stage1.sh || exit 1
(cd $AW && PYTHONPATH=. $PY $B/stage1/stage1_score.py $B/stage1/stage1_bank.npz $B/stage1/scores) || exit 1
$PY $B/stage1/stage1_decide.py $B/stage1/scores/scores.json $B/stage1/verdict.json || exit 1
# Stage 2 (refuses unless Stage 1 is PROCEED)
$B/stage2/run_stage2.sh || exit 1
# Stage 3: NOT AUTHORISED. After the owner's yes, the coordinator writes $B/stage3/OWNER_GO, then:
$B/stage3/launch_stage3.sh
```

The sidecar step, after Stage 3's collect (CPU):

```bash
(cd $AW && PYTHONPATH=. /usr/bin/time -v $PY scripts/simple_rl_sibling_pairs.py \
  --collect $B/stage3/collect/gen_0001/collect --out $B/stage3/sibling/gen_0001) > $B/stage3/sibling.log 2>&1
```

## Provenance

- **Bank:** `~/chess-artifacts/operations/rl-run1-root-full-width/`. `README.md` lists every file; `SHA256SUMS`
  covers every script, and the short shas above are its first 12 hex digits.
- **Sizing measurements:** `sizing/results/`, code head `fd8c0ce08` in `code_head.txt`. They were added as two
  rows to Table 2 of the evidence log.
- **Incident, 2026-11-21 21:38:44–21:40:56 EST.**
  - What happened: a guard test of v1 of `run_stage0.sh` passed its GPU-idle check. v1 trusted
    `nvidia-smi --query-compute-apps`, which never listed match 8's PID 2337486, although it did list this bank's
    own collects in 62 of 64 samples. Three collect processes (SL base, 64 sims, 128 in flight each) then ran for
    ≈ 126 s on the GPU beside the registered match 8 (`zq_recal_vs_base_1_A`).
  - Peak memory was 30,406 MiB, at about 21:40:49.
  - They were killed by exact PID. The match survived: no OOM, no CUDA error. It is matched-sims, so its games are
    not clock-dependent. A note on it is being added to match 8's results in PR #1482 (not merged at the time of
    writing).
  - Their partial outputs, all under this bank, were deleted. `stage0/INCIDENT_guardtest_20261121/` keeps only the
    logs and `INCIDENT.txt`; it is not a measurement.
  - The guard was rewritten as above. From now on, launchers are tested only with the GPU command stubbed and a
    fake `nvidia-smi`.
- **Versions.** Scripts changed after the first review are kept beside their v2 as `.v1`, and `SHA256SUMS.v1` is the
  first sum file.
