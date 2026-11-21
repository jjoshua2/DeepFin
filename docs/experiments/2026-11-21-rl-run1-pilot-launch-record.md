# RL run 1 pilot: launch record (NOT LAUNCHED)

This record makes the pilot of
[the games-not-rows amendment](2026-11-21-rl-run1-amendment-games-not-rows.md)
(Step 1) executable. It gives the exact commands, output directories, stopping rules
and prerequisites. The hypotheses, cells, thresholds and the simulated reading
table are the amendment's and are not repeated here. Where the two differ, this
record wins, and each change is listed under [Changes from the amendment](#changes-from-the-amendment).

**Nothing launches from this record until the coordinator says so** and the owner
block at the end is filled in.

## Prerequisites (all merged before the pinned commit is chosen)

| PR | What the pilot needs from it | State |
|---|---|---|
| #1441 | multi-process collect (`--collect-procs`), the PAUSE marker | merged `b43ef912a` |
| #1445 | `--optimizer aurora --optimizer-hparams-source init-checkpoint`: the SL base's Aurora split and end rates on a fresh optimizer | merged `c2eabd4a2` |
| #1455 | loader RAM: the training window streams into preallocated arrays | merged `40e0a9b4e` |
| #1457 | `--candidate/--reference-value-from`, `--candidate/--reference-value-recal`; `scripts/fit_value_recal.py` | merged `9b74b1294` |
| #1459 | the #1457 review's follow-ups: the seat-balanced pool re-hashes value-from files and compares the seats' value settings; the fit reads the run's pinned `holdout_frac`; `logits_dtype` recorded | open |
| #1462 | `--on-max-plies {abort,draw}`: a game that reaches `--max-plies` unresolved is banked (moves, `*`, ply count) before the abort; the choice stamped and counted | open |
| #1463 | `arena_seat_balanced_pool.py` pools several A/B run pairs (looks) into one cumulative estimate, clustering by opening across looks, and refuses looks that are not one match | open, stacked on #1459 (match 4 and the S extension need cumulative looks) |
| #1448 | the scorer, for the report-only secondaries | merged `727f7a6c6` |

**#1444 is NOT a prerequisite**, and the pilot does not use it. Its review verdict
was FIX: a collect resumed mid-process keeps the games that finished and replays
the unfinished ones, so long games are under-represented. That is a biased sample.
The pilot collect instead runs uninterrupted: about 1.6 h, scheduled when the owner
is not gaming, with no PAUSE. If it is killed, main's collect (#1441) redoes every
unfinished process whole from its seed. That selects nothing (losing up to about
1.45 h at a kill near the end). See the pre-training check below.

### The pinned commit

Collect, training, the recalibration fit and **all four matches run from ONE
detached worktree at ONE commit**:

```bash
PIN=<main's sha after the last prerequisite merges; written into the owner block before the collect starts>
git -C ~/projects/chess worktree add --detach ~/projects/chess-worktrees/rl-run1-pilot-$PIN $PIN
cd ~/projects/chess-worktrees/rl-run1-pilot-$PIN && python3 scripts/build_production_extensions.py
```

**Why one commit.** On 2026-11-21 the 800-game head-swap arms no longer reproduced the
2026-11-19 bank game for game. This was true of the PLAIN arms too: plain Q50 matched
on 30 of 424 games and plain F12 on 63 of 799. 24 commits since the bank's
`df55413a6` touch the play path, CUDA graphs among them
(`~/chess-artifacts/operations/rl-claude-20261119/arena-hybrid-option/README.md`).
So every pilot contrast is **within-run, on one commit**:
- every arena result row's `git_sha` must equal `$PIN`;
- the bank's absolute numbers are context only, never a control.

## Directories

```bash
R=$HOME/chess-artifacts/operations/rl-run1-pilot
TB="$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6"
BOOKS=$HOME/projects/chess/data/opening_books
SF=<stockfish binary; required by play_batch, its labels are unused>
```

| Path | Contents |
|---|---|
| `$R/sl_base.pt` | the SL base, sha256 `a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`, read-only |
| `$R/openings/` | the TRAIN / EVAL split (checked against the banked split's sha256s) |
| `$R/collect/` | the one collection (source run dir, frozen after sealing) |
| `$R/arm_Zq/`, `$R/arm_S/`, `$R/arm_P0/` | the three training cells (`--collect-from $R/collect`) |
| `$R/recal/` | the recalibration fit |
| `$R/arenas/<match>/` | game logs, PGNs, result rows, pool reports, unresolved-game sidecars |

## Step 0: inputs

```bash
mkdir -p $R && cp -n <shared SL base file> $R/sl_base.pt
echo "a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0  $R/sl_base.pt" | sha256sum -c - || exit 1
chmod a-w $R/sl_base.pt; SL=$R/sl_base.pt

PYTHONPATH=. python -u scripts/simple_rl_loop.py split-openings --out $R/openings \
  --book $BOOKS/8moves_v3_plus_policybeam_final145cp_plus_uho2024_060_110_plus_2move_thinbeam_dedup.pgn.zip:16 \
  --book $BOOKS/2moves_v2_no8overlap_uho060110.pgn.zip:4 --eval-lines 4000 --salt simple-rl-v1
```

Then run the banked-split check from
[loop exp1](2026-11-20-loop-exp1-one-generation-from-sl-base.md) (step 0b). Its
expected values:

| Field | Expected |
|---|---|
| `train.jsonl` sha256 | `1a98416d…` |
| `eval.jsonl` sha256 | `1d3d13ed…` |
| `eval.pgn` sha256 | `067675c2…` |
| counts (train, eval) | 153,842 and 4,000 |
| `eval_max_plies` | 16 |

The check exits non-zero on any difference.

## Step 1: the collection (about 1.6 GPU-h, uninterrupted)

```bash
COLLECT="--openings-dir $R/openings --init-checkpoint $SL --device cuda --seed 2026112100 \
  --games 8000 --collect-procs 3 --concurrency 128 --sims 64 --full-prob 1.0 \
  --topk 16 --c-scale 0.1 --c-visit 50 --max-plies 450 --syzygy $TB --sf-path $SF"
PYTHONPATH=. python -u scripts/simple_rl_loop.py collect --run-dir $R/collect --generation 1 $COLLECT \
  > $R/collect.log 2>&1
chmod -R a-w $R/collect
```

- **Collect shape.** `--collect-procs 3 --concurrency 128` is the default shape at 64 sims. It is spelled out because it is pinned and it sets the RNG streams.
- **Hold-out split.** The source run pins `holdout_frac` 0.05 (the default) in `$R/collect/run_config.json`. The arms use the same value, and `fit_value_recal.py` reads it from there (#1459).
- **Do not PAUSE.** A PAUSE keeps about 23 GB of VRAM allocated.
- **If the collect is killed,** rerun the same command. Unfinished processes are redone whole from their seeds; finished processes are kept. There is no game-level resume.

**Pre-training check (all must hold, or nothing trains):**

```bash
python3 - $R/collect/gen_0001/collect <<'EOF'
import json, sys
d = sys.argv[1]; m = json.load(open(d + "/manifest.json"))
games = sum(1 for _ in open(d + "/games.jsonl"))
resumes = m.get("game_resumes", m.get("stats", {}).get("game_resumes"))
ok = games == 8000 and not resumes
print(json.dumps({"games": games, "game_resumes": resumes, "ok": ok})); sys.exit(0 if ok else 3)
EOF
```

`game_resumes` exists only under #1444, which the pilot does not use. On `$PIN` the
key is absent, and the check requires it to be absent or empty, so a collect that
#1444 resumed cannot pass. The collect gates
(`scripts/simple_rl_collect_gates.py $R/collect/gen_0001/collect`) are also recorded.

## Step 2: three training cells, ONE AT A TIME, nothing else on the box

```bash
TRAIN="--presentations 1 --mirror-prob 0.5 --batch-size 512 --lr 3e-6 --lr-floor 3e-6 --lr-schedule flat \
  --warmup-steps 0 --optimizer aurora --optimizer-hparams-source init-checkpoint \
  --optimizer-state fresh --init-optimizer-state fresh --train-target-temp 1.0 \
  --kl-policy 0 --kl-wdl 0 --sl-mix-frac 0 --holdout-frac 0.05 --probe-rows 2048 --no-eval"
declare -A CELL=(
  [Zq]="--value-outcome-weight 0.5 --value-search-weight 0.5 --value-search-source stored_search_wdl"
  [S]="--value-outcome-weight 0 --value-search-weight 1 --value-search-source stored_search_wdl"
  [P0]="--w-wdl 0")
for c in Zq S P0; do
  PYTHONPATH=. python -u scripts/simple_rl_loop.py run --run-dir $R/arm_$c --collect-from $R/collect \
    --generations 1 $COLLECT $TRAIN ${CELL[$c]} > $R/arm_$c.log 2>&1 || break
done
```

- **Arm requirements.** Each arm takes the source's collect flags (including `--seed 2026112100`; an arm with another seed is refused) and the same `--init-checkpoint $SL`.
- **Recipe.** As pinned by the amendment:
  - SL mix off;
  - train temperature 1.0;
  - `stored_search_wdl` on both cells that have a search share;
  - KL 0 / 0;
  - exact-once with mirror 0.5;
  - batch 512, flat.
- **Learning rates.** `--lr 3e-6` is the AdamW groups' rate. The matrix group runs at 3e-6 × the checkpoint's multiplier (20), so 6e-5. Both are recorded in `config.realised.optimizer`.
- **RAM.** With #1455, about 31 KiB per row on average, so 20–23 GiB at the collection's ≈ 758k rows. The actor prior mix would take this to 36–45 GiB, but the pilot does not use it. Each cell runs alone, never beside an arena or another cell.
- **Time.** About 16 min per cell at ≥ 770 rows/s (a floor measured at batch 256).

## Step 3: the recalibration fit (before match 2 is played)

```bash
PYTHONPATH=. python scripts/fit_value_recal.py --net $R/arm_Zq/gen_0001/train/model.pt --reference $SL \
  --collect-dir $R/collect/gen_0001/collect --device cuda --out $R/recal/Zq_recal.json
RECAL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cli"])' $R/recal/Zq_recal.json)
```

- **Rows.** The fit uses the collect's **hold-out games** only (`is_holdout_game` at the run's pinned 0.05), which no cell trained on. They are TRAIN-split openings, so they are disjoint from the matches' `eval.pgn` by the split's orbit rule.
- **Output record.** `Zq_recal.json` records the rows' digest, both nets' sha256 and the logits dtype (bf16 on CUDA, what the arena plays).
- **Pre-registration.** The fit is pre-registered here and is not refitted after any match game is seen.

**Where a match game starts, and what `--exclude-openings` keys** (read from the code
on main 6c72b821b).
- **Where the arena starts.** `arena_standard.load_paired_openings` →
  `selfplay.opening._sample_from_pgn`:
  - It starts from `chess.Board()`. It ignores any `FEN` header; `eval.pgn` has none.
  - It replays each game's mainline for at most `--opening-plies` plies (`_load_pgn_opening_sequences` truncates at `max_plies`).
  - It deduplicates positions by `board.epd()`.
- **What the fit keys.** `fit_value_recal.opening_keys` (#1459) keys each `eval.pgn`
  game's **full mainline final position**. It uses `game.board()`, which is the
  standard start when there is no FEN header, and it compares by `board.epd()`.
- **They agree exactly iff `--opening-plies` ≥ the longest `eval.pgn` line.**
  - The split's `summary.json` records `eval_max_plies` 16, and every match here passes `--opening-plies 16`.
  - simple-RL's evaluate refuses an `--eval-opening-plies` below the split's longest line.
  - With a smaller value, the arena would start at a PREFIX of the line. `--exclude-openings` would then miss rows that pass through the prefix but not through the full line, which voids the guarantee. So `--opening-plies` is pinned at 16 and is checked in every result row (`opening_plies` is in the settings fingerprint).
- **The pilot's fit does not use this guard.** It uses `--collect-dir`, which is
  disjoint from the EVAL lines by the split's orbit rule.
  `--exclude-openings` matters only for a `--rows` fit.

## Step 4: the four matches

Every match is **seat-balanced**. Run A has the arm as candidate, and run B has the
arm as reference, with the same seed, so both runs play the same opening pairs.
"N games" means N in total, N/2 per run. Common flags:

```bash
ARENA="--mode matched_sims --sims 64 --search-shape training --no-gumbel-noise --compile off \
  --openings $R/openings/eval.pgn --opening-plies 16 --max-plies 600 --on-max-plies abort \
  --syzygy $TB --max-concurrent-games 128"
run() {  # run <match> <look> <A|B> <seed> <games-per-run> <seat args: --candidate ... --reference ... [value flags]>
  local m=$1 k=$2 s=$3 seed=$4 n=$5; shift 5; local d=$R/arenas/$m; mkdir -p $d
  PYTHONPATH=. python -u scripts/arena_standard.py $ARENA --seed $seed --games $n "$@" \
    --label ${m}_${k}_$s --games-out $d/${m}_${k}_$s.games.jsonl --pgn-out $d/${m}_${k}_$s.pgn \
    --out $d/results.jsonl > $d/${m}_${k}_$s.log 2>&1
}
```

(`run` passes its trailing arguments to the arena, so the seat flags below go after
`<games>`.)

**On GPU:**
- **CUDA graphs.** CUDA graphs capture the whole forward of a hybrid seat, both nets included.
- **Compile.** `--compile on` is refused with a hybrid seat, so every match runs with `--compile off`.
- **Measured wall** on the RTX 5090 at 64 sims, per 800 games:

  | Seat type | Wall | Ratio to plain | GPU-h per 800 games |
  |---|---|---|---|
  | plain | 698 s | 1.0 | 0.19 |
  | `--*-value-recal` only (one forward) | 657 s | about 0.94 | 0.19 |
  | `--*-value-from` hybrid (two forwards) | 974 s | about 1.4 | 0.27 |

### The matches

Seeds are fixed here. A void look's backup seed is the look's seed + 100 (see the
max-plies rule).

| # | Match | A: candidate | A: reference | B: candidate | B: reference | Games | Seed | Looks |
|---|---|---|---|---|---|---|---|---|
| 1 | gate: Zq vs base | Zq | SL | SL | Zq | 1,600 | 2026112101 | one |
| 2 | head-to-head: Zq + recal vs Zq | Zq `--candidate-value-recal $RECAL` | Zq | Zq | Zq `--reference-value-recal $RECAL` | 1,600 | 2026112102 | one |
| 3 | S vs base | S | SL | SL | S | 800 | 2026112103 | one, plus the extension |
| 4 | P0 policy + base value vs base | P0 `--candidate-value-from $SL` | SL | SL | P0 `--reference-value-from $SL` | 1,600 / 3,200 / 4,800 | 2026112141 / 42 / 43 | three |

For example, match 1:

```bash
Zq=$R/arm_Zq/gen_0001/train/model.pt
run gate 1 A 2026112101 800 --candidate $Zq --reference $SL
run gate 1 B 2026112101 800 --candidate $SL --reference $Zq
PYTHONPATH=. python scripts/arena_seat_balanced_pool.py --arm $Zq \
  --a-games $R/arenas/gate/gate_1_A.games.jsonl --b-games $R/arenas/gate/gate_1_B.games.jsonl \
  --a-result $R/arenas/gate/results.jsonl --b-result $R/arenas/gate/results.jsonl \
  --a-label gate_1_A --b-label gate_1_B --out $R/arenas/gate/pool.json
```

Matches 2 and 3 follow the same pattern with the seat arguments in the table. In
match 2 both seats are Zq's file, and the pool (with #1459) refuses unless A's
`candidate_value` equals B's `reference_value`. The engine names carry the
`+recal(T,b)` suffix.

**Match 4 runs in looks.** Each look is its own A/B pair of 800 games per run, with
the look's own seed (2026112141, 42, 43). After look k, the multi-look pool
combines looks 1..k and clusters by opening across looks (an opening drawn in two
looks is one cluster). The pool gives the estimate and its SE, and they are read
against z = 3.471 / 2.454 / 2.004:
- stop with "the policy pays" iff est − z·SE > 0;
- stop with "does not pay +10" iff est + z·SE < +10;
- otherwise run the next look;
- after look 3 without a stop, the reading is "unread".

There is no other stopping point.

The pool after look k (k = 2 shown) repeats the run flags once per look, in look
order, and reads the look's z:

```bash
P0=$R/arm_P0/gen_0001/train/model.pt; d=$R/arenas/graft
PYTHONPATH=. python scripts/arena_seat_balanced_pool.py --arm $P0 --z 2.454 \
  --a-games $d/graft_1_A.games.jsonl --b-games $d/graft_1_B.games.jsonl \
  --a-result $d/results.jsonl --b-result $d/results.jsonl --a-label graft_1_A --b-label graft_1_B \
  --a-games $d/graft_2_A.games.jsonl --b-games $d/graft_2_B.games.jsonl \
  --a-result $d/results.jsonl --b-result $d/results.jsonl --a-label graft_2_A --b-label graft_2_B \
  --out $d/pool_look2.json
```

**The pool's checks across looks.** The pool refuses looks that are not one match:
- a repeated seed or game log;
- a header setting other than `seed` and `games` that differs between looks;
- a result-row field in `ROW_MATCH_FIELDS` that differs (`git_sha` among them),
  except `games_requested`;
- different nets or seat values.

**The decision rule.** The decision reads `result.pooled`: `excess` ± z·`se` on the
score scale, with the +10 boundary converted by Elo(x) = −400·log₁₀(1/(0.5 + x) − 1).

**The S extension (reviewer item b).** Match 3 is read at 800 games with the
amendment's rules: damage iff UB95 < 0; adoptable iff LB95 > −10. **If S's estimate
is > 0 but LB95 ≤ −10**, S is extended with match 4's look schedule:
- looks at 1,600 / 3,200 / 4,800 cumulative games;
- the added looks use seeds 2026112131 / 32 / 33;
- the 800-game first read is pooled into the first look.

At each look, S is adoptable iff est − z·SE > −10. The extension stops early iff
est ≤ 0 or est + z·SE < −10. After the third look without adoption, S is not
adopted and the order falls through to Zq (Step 2 of the amendment).

Pre-registered caveat: the 800-game read already had one chance to adopt at
z = 1.96. The extension adds three more chances, so S's false-adoption rate is
somewhat above a single look's. The extension runs only when the point estimate is
positive, which bounds that rate.

### Adoption is measured against the SL base (reviewer item a)

- **Adoption.** A generation's value target is adopted only if its match **against
  the SL base** has LB95 > −10 (at the look's z for a sequential match).
- **Drift alarm.** The match against the previous generation is kept as a drift
  alarm only. It never adopts and never vetoes adoption on its own; a reading of
  damage there is reported to the owner.
- In the pilot, the previous generation IS the SL base, so the two coincide. The
  distinction applies from Stage 2's second generation on.

### The max-plies rule (registered)

**What happened.** The 2026-11-21 regression bank plus the head-swap bank hold 13,707
completed games:

| Statistic | Plies |
|---|---|
| median | 95 |
| p99 | 191 |
| p99.9 | 262 |
| longest completed game | 298 |

24 games reached ≥ 250 plies and 4 reached ≥ 280. Four arms aborted at the default
`--max-plies 300` with "strict Syzygy arena reached max_plies with unresolved game …
refusing a fabricated draw": two head-swap arms (f12_A_full, f12_C_Pbase_Vf12) and two
regression arms (plain q50_A_repro and plain f12_A_repro). That is about 3e-4 per game.
- **Plain arms abort too.** The refusal is #854's strict-Syzygy rule (2026-09-26), so it applies to every `--syzygy` arena, plain included. It was not introduced by #1457.
- **The offending games were lost.** None was banked: the process died with the moves. #1462 fixes this.

**The registered setting is `--max-plies 600 --on-max-plies abort`.** The completed-game
tail falls by about 6× per 30 plies past 250. Extrapolated, an unresolved game is
about 4e-8 per game at 450 plies and about 5e-12 at 600. **That is an extrapolation,
not a measurement.** The limit costs nothing: a game runs longer only if it would
otherwise have aborted. The 50-move rule, repetition and 6-man adjudication end
practically every game long before 600.

**If a registered look still aborts:**
1. #1462 has written the unresolved game (moves, `*`, ply count,
   final FEN) to `<games-out>.unresolved.jsonl` before raising. The look's finished
   games stay in its game log and PGN.
2. **Tripwire: STOP and report to the owner. Do not rerun anything automatically.**
   An abort at 600 means the extrapolation above is wrong.
3. Only with the owner's decision is the look **voided** and rerun from scratch on
   its backup seed (seed + 100). The void keeps its files, renamed `*.void`, and the
   record lists it.
   - Voiding a look is itself a selection: it discards a look *because* it contained
     a very long game.
   - That is acceptable ONLY because the abort probability at 600 is negligible by
     the extrapolation. The tripwire exists because a second abort would show the
     selection is not negligible.
4. `--on-max-plies draw` is never used for a registered match.

**Never resume a registered look.**
- **Why resume changes games.** Arena play uses ONE `np.random.default_rng(seed)`, shared by every active board's move sampling on every ply (`pick_moves_for_boards`, temperature 0.1). So the games a resumed run replays are new games, not the old ones.
- **The bank shows it.** In the head-swap bank's f12_A resume, the offending pair 352 half 0 (unresolved at ≥ 300 plies) was replayed as a 109-ply Syzygy win. 44 in-flight games were replayed with it.
- **Same selection as the collector's.** That is the collector's survivorship selection in miniature: a long game is replaced by a fresh one.
- **The rule.** The looks are the only stopping points. A look that is killed (OOM, CUDA error, power) is rerun whole on the same seed. A fresh run on the same seed and commit is deterministic: the 40-game identity runs and the D-arm repeats matched exactly. A finished look is never re-rolled.
- **PAUSE is fine.** The arena's PAUSE marker holds state, so a paused look continues the identical games.
- **`--resume` is for non-registered runs only.**

**Caveat on the existing head-swap bank.** Three of its arms were resumed:

| Arm | Cause | `ResumeReplay` games |
|---|---|---|
| f12_A_full | max-plies abort | 44 |
| f12_C_Pbase_Vf12 | max-plies abort | 2 |
| q50_A_full | CUDA "unknown error" | 42 |

Each carries a small survivorship selection, of at most that many games out of 800.
Its paired head-swap contrasts stand. The caveat applies to the absolute numbers.

## DRAFT for the owner: value anchoring, and the owner's playout-cap design as an equal-time arm

Nothing in this section is part of the pilot until the owner decides (L4–L6 below).

**The owner, verbatim (2026-11-21):**

> "So my plan with only a few value samples per game wad gonna use like 16 modes on the other moves and not train and do 64 on the few we do sample so we wouldnt oversample value and get big speed increase. Your plan to only sample search seems likely to work in short term but without a way to anchor it seems it won't work ling term many cycles but perhaps we can make if work by playing stockfish like we originally did in my loop tuning its strength it works punish holes and we can train on those spegicially somehow but still need some z grounding or opponent search score when it agrees with z"

### 1. S is a one-generation diagnostic of the shape hypothesis, nothing more

- **What S is for.** Cell S (value trained on the stored search WDL alone) exists to
  test one thing: is the value loss a shape loss that a soft target avoids?
- **No multi-generation recipe will train the value on search WDL alone. z stays in every recipe.** That holds whatever S reads. Step 2's adoption order is amended
  accordingly: an S reading of "adoptable" licenses a search-heavy *blend* with z, never
  s = 1.0 across generations. Without an outside anchor, the value then has nothing
  to pull it back once it drifts.
- **The banked facts already point that way** (amendment, "the leading hypothesis
  is WDL shape"):
  - **The stored 64-sim search WDL is flatter than the base,** in the damage direction:

    | Statistic | Base | Stored search WDL | Change |
    |---|---|---|---|
    | entropy | 0.416 | 0.450 | +0.034 |
    | \|W − L\| | 0.440 | 0.402 | −0.038 |

    Only its draw mass matches the base.
  - **Q50, with half its value target from search WDL, lost more than F12, with a
    tenth.** Both learners' boundaries were identical except `outcome_wdl_frac`
    0.5 vs 0.9:

    | Measure | Q50 | F12 |
    |---|---|---|
    | value half | −55.2 | −18.3 |
    | full net | −63.2 | −34.9 |

### 2. The owner's design as an equal-GPU-time arm

**The standing rule:** sims per move is an equal-GPU-time arm, never presumed.

**Collection B** is a second collection from the SL base with the same GPU time as
collection A (about 1.6 h):
- `--full-prob p --fast-sims 16` at `--sims 64`;
- fast plies store **no row** (the default, without `--fast-value-rows`);
- everything else, including the seed family, as A;
- its own source run dir `$R/collect_B`.

**Prerequisites before B:**
- `--value-rows-per-game r`, which the engine agent is building;
- the playout-cap throughput, measured in `collect-throughput/playout-cap/`;
- the fast-ply root-noise check: KataGo disables noise on fast searches. This was "to check" in the amendment.

**p is chosen from the measurement, not the guess.** Until the throughput run
lands, the table below uses the amendment's eval-count model,
games/h ≈ 1 / (p / 5,021 + (1 − p) / 19,300), which is a **guess**. Policy rows are
near-independent (residual ICC 0.03), so the policy's effective rows are about
rows / (1 + (m − 1)·0.03) for m rows per game. Value information scales with games
(outcome ICC 0.6).

| Collection (1.6 GPU-h) | Games | Full-search (policy) rows | Rows per game m | Policy effective rows | Value: games |
|---|---|---|---|---|---|
| A: p = 1 (the pilot) | 8,000 (measured rate) | ≈ 758k | 95 | ≈ 198k | 8,000 |
| B: p = 0.25 | ≈ 18k (guess) | ≈ 434k | ≈ 24 | ≈ 257k | ≈ 18k |
| B: p ≈ 0.04 (≈ 4 full plies per game) | ≈ 28k (guess) | ≈ 106k | ≈ 3.8 | ≈ 98k | ≈ 28k |

At p = 0.25, B gives more games AND, by this model, more effective policy rows. At
p ≈ 0.04, games rise further, but the policy is starved to about half of A's
effective rows. Fill in from `collect-throughput/playout-cap/` before choosing.

**Cells from B:**

| Cell | Training | Contrast | Numbers |
|---|---|---|---|
| **K-blend** | Zq's exact recipe (outcome 0.5 / search 0.5, `stored_search_wdl`) on B's rows | vs Zq on A = "more games, fewer stored rows per game, at equal collection time" | rows to fill from `collect-throughput/playout-cap/` |
| **K-z2** | value loss on r = 2 rows per game, z-anchored (outcome only), via `--value-rows-per-game 2`; policy on every full-search row | the owner's design: little value per game, from z | needs `--value-rows-per-game`; numbers to fill from `collect-throughput/playout-cap/` |

**Added matches** follow the same rule as matches 1–3: seat-balanced, plain speed,
SE = 8.9 × √(800 / games).

| # | Match | Games | SE | Reads | Cannot show |
|---|---|---|---|---|---|
| K1 | K-blend vs base | 1,600 | 6.3 | damage iff UB95 < 0; adoptable iff LB95 > −10 | whether K-blend beats Zq: the difference of two vs-base results has SE ≈ 8.9, so only gaps above about 17 Elo read |
| K2 | K-z2 vs base | 1,600 | 6.3 | as K1 | why it reads as it does: it changes the value dose, the source (z only) and the games at once |
| K3 | K-blend vs Zq, head-to-head | 1,600 | 6.3 | "B's shape helps at equal time" iff LB95 > 0 | anything about the base: it is a within-pair contrast only |

K3 exists because K1 − match 1 is underpowered. Both nets are plain, so K3 plays
at plain speed.

**Added cost:**

| Item | GPU-h |
|---|---|
| collect B | 1.6 |
| two cells (B has fewer stored rows than A) | ≈ 0.3–0.5 |
| three 1,600-game plain matches (3 × 0.38) | 1.14 |
| **Total** | **≈ 3.0–3.3**, on top of the pilot's 4.0–6.0 |

### 3. Parked, with launch conditions (not queued)

- **Stockfish-opponent games at tuned strength, as an outside anchor.** This is the
  owner's original loop: tune SF's strength so the net's holes are punished, then
  train on those games.
  - **Launch condition:** EITHER a self-play generation has shown a measured step
    (a match vs the base with LB95 > 0), OR a multi-generation run gains in
    self-play but not against the outside yardsticks (BT4 node-odds, SF).
  - The design must keep the training target best-move based, never the handicapped
    move SF played.
- **Outcome-conditioned value source in games against strength-limited Stockfish (DESIGNED arm, not queued).**

  The owner clarified this item (verbatim, 2026-11-21):

  > "I meant we could train on self vs limited stockfish and use our search only on games we win and on games we losd use our search until sf thinks its winning and then use its search values etc"

  So the item is NOT "filter rows where search agrees with z". (Such a filter would
  be a selection: it drops exactly the rows where the head was wrong.)

  **Definition.** Every row is relabelled, and the result z picks which source labels it:

  | Game result | Value target |
  |---|---|
  | **Won** | our stored search WDL on every row |
  | **Lost** | our stored search WDL up to ply t*; from t* to the end, SF's evaluation converted to WDL by the project's cp-logistic (`sf_wdl_use_cp_logistic`, `sf_wdl_cp_slope` / `sf_wdl_cp_draw_width` from the run's config) and flipped to our POV |
  | **Drawn** | our stored search WDL |

  t\* is "SF thinks it is winning": the LAST ply after which SF's full-strength
  evaluation stays at or above a threshold until the game ends.

  **Defaults, each owner-changeable:**
  1. **Threshold:** SF's win probability (W of the cp-logistic WDL, SF's POV) ≥ 0.80. Using the last ply from which SF *stays* above it means the relabelled segment is consistent with the result. A transient swing earlier in the game is not relabelled.
  2. **Drawn games:** our search WDL.
  3. **Share:** the arm is a share of the generation's games, mixed with self-play. The share is an arm parameter (no default proposed until the gift-win measurement below).
  4. **Policy target:** unchanged (the stored search target).

  **Properties:**
  - **Grounded in results:** z chooses the source.
  - **Keeps soft targets:** in line with the pilot's shape hypothesis.
  - **Disagreement rows are relabelled, not dropped:** a row where our search and the result disagree gets the winner's evaluation.
  - **One-sided gap:** a win against a deliberately weakened opponent does not show our evaluation was right, so won games get no correction. These are positions where full-strength SF already saw us losing and the limited SF let us off.

  **The cheap measurement that sizes the gap, before any training (report only).**
  Play a small collection, SL base vs limited SF, and report two things:
  - the share of won-game rows where SF's full-strength evaluation had us below the same threshold (our win probability ≤ 0.20: "gift wins");
  - the resulting target's shape statistics (entropy, |W − L|) against the base's.

  **What the simple loop lacks, from the code (estimate only; nothing is built):**
  - **SF opponents.** `simple_rl/collect.py` forces self-play (`selfplay_fraction=1.0`, a default `OpponentConfig()`, `sf_nodes` 1, with SF labels unused). The play path itself supports SF opponents:
    - `selfplay_fraction < 1` gives curriculum games;
    - the opponent's move is drawn from `stockfish_turn.within_regret_candidates`, the candidates within `regret_limit` of SF's best, with the limit driven by the live loop's PID on `wdl_regret`.

    So the strength limit is a choice among SF's MultiPV candidates, not a node cut. A fixed regret limit (no PID) is the simpler arm.
  - **Per-ply SF evaluation on rows.** The live play path attaches a SF label to each net-turn row (`rec.sf_wdl`, cp-logistic, at the label node budget, which is separate from the handicap).
    - It is computed AFTER the net's move has been pushed: the position P1, SF to move, flipped to our POV (`stockfish_turn._sf_result_wdl_for_record`).
    - So it evaluates the row's position *after our chosen move*. That is the P0/P1 alignment trap (`sf_multipv_raw` is the previous ply, so use `sf_p0_regret`).
    - That is the right quantity for "SF thinks it is winning" after our move, but it is one ply off the row's own position.
    - Simple-RL's `TRAIN_KEYS` (`simple_rl/learn.py`) do not include `sf_wdl`, so the shards would need it stored.
  - **A new value option.** It needs a `--value-search-source`-style option that takes z and the per-ply SF evaluation and builds the per-row target above (t\* computed per game from the stored evaluations).

  **Launch condition:** the pilot has read, so we know whether value shape is the damage and whether the policy pays.

## Cost (GPU-h, measured rates on `$PIN`'s predecessor code)

| Item | Games | GPU-h |
|---|---|---|
| collect | 8,000 | 1.6 |
| three training cells, in sequence | — | about 0.8 |
| recalibration fit | — | < 0.1 |
| 1: gate (plain) | 1,600 | 0.38 |
| 2: head-to-head (recal-only seats, plain speed) | 1,600 | 0.38 |
| 3: S (plain) | 800, up to 4,800 with the extension | 0.19, up to 1.14 |
| 4: P0 graft (value-from hybrid) | 1,600 to 4,800, expected about 4,070 | 0.54 to 1.62, expected about 1.37 |
| **Total** | | **4.0 to 6.0; expected about 4.8 without the S extension** |

The amendment's match costs (0.45–0.7 GPU-h per 800 hybrid games) came from the
head-swap harness without CUDA graphs and are superseded by this table.

## Changes from the amendment

1. #1444 is removed from the prerequisites. The collect runs uninterrupted, and a
   killed collect is redone whole per process. Consequently `--rows-per-shard 5000`
   is dropped, since it only shortened #1444's resume window.
2. Adoption is measured against the SL base; the previous-generation match is a drift
   alarm (reviewer item a).
3. The S extension (reviewer item b).
4. `--max-plies 600 --on-max-plies abort`, the void-and-tripwire rule, and no resume
   of a registered look.
5. Match costs are re-measured with CUDA graphs.
6. All contrasts are within-run on one pinned commit. The bank does not reproduce
   game for game on current code.
7. Two prerequisites are added: #1462 (`--on-max-plies`) and #1463 (the multi-look pool)
   (match 4 as registered needs cumulative pooling across looks).

## Owner decisions

| # | Decision | Proposal | Owner's decision |
|---|---|---|---|
| L1 | Launch the pilot as recorded here | yes, once every prerequisite above is merged and `PIN` is written below | |
| L2 | Budget | 4.0–6.0 GPU-h (expected about 4.8), plus the collect window of about 1.6 h with no gaming | |
| L3 | Cell changes | none (Zq, S, P0 as in the amendment) | |
| L4 | S's role | a one-generation shape diagnostic only; z in every multi-generation recipe | |
| L5 | Add collection B and cells K-blend / K-z2 with matches K1–K3 (≈ 3.0–3.3 GPU-h more) | yes, once `--value-rows-per-game` and the playout-cap throughput are banked; p from that measurement | |
| L6 | The parked items (SF opponent as an outside anchor; the outcome-conditioned SF value arm with its defaults) | stay parked with the launch conditions above; run the gift-win measurement when the second unparks | |
| — | `PIN` (the commit every stage runs from) | | |
