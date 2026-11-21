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

### The pinned commits: C0 for the collection, `$PIN` for everything after it

**The collection runs at C0 = `6c72b821b`.** Step 0 and Step 1 run from the detached
worktree `~/projects/chess-worktrees/rl-run1-pilot-6c72b821b`. They started before
#1459, #1462 and #1463 merged, on the owner's go for Step 1 (owner block).

**Training, the fit and all four matches run at `$PIN`.** That is main after the
last prerequisite merges, in its own detached worktree. Every command here runs
with the live venv's interpreter, `PY=$HOME/projects/chess/.venv/bin/python`
(torch 2.10.0+cu128); `python` / `python3` below mean `$PY`.

```bash
PIN=<main's sha after the last prerequisite merges; written into the owner block before any training>
git -C ~/projects/chess merge-base --is-ancestor 6c72b821b $PIN || exit 1     # $PIN must descend from C0
git -C ~/projects/chess diff --name-only 6c72b821b $PIN                        # -> the owner block
git -C ~/projects/chess worktree add --detach ~/projects/chess-worktrees/rl-run1-pilot-$PIN $PIN
cd ~/projects/chess-worktrees/rl-run1-pilot-$PIN && $PY scripts/build_production_extensions.py
```

- The collection is the shared input of every cell, so `$PIN` must be a descendant
  of C0.
- The list `git diff --name-only C0 $PIN` goes into the owner block.

**The C0 → `$PIN` guard is mechanical, not a path rule** (statistics review of #1464).
Shards store the ENCODED input `x`. The cells therefore learn from C0-encoded
inputs, while every match encodes at `$PIN`. An encoder change between the two
would put the cells out of distribution, but not the SL base, and would bias every
vs-base match against the cells.

**Tripwire.** The checks below are MANDATORY if the diff touches any of:
- the encode path:
  - `chess_anti_engine/encoding/**`, which includes the `rep_fix` / history regime, `_lc0_ext.c`, `_features_ext.c`, `_cboard_impl.h`, `_features_impl.h`, `_bitboard_planes_impl.h` and `_slider_attacks_impl.h`;
  - `chess_anti_engine/mcts/_mcts_tree.c`, which includes the cboard and features headers;
  - the encoder selection in `selfplay/network_turn.py` (`_batch_encoder_pair`);
  - `x_hash` in `simple_rl/collect.py`;
- the `moves/` policy maps;
- shard I/O (`replay/shard.py`, `replay/packed_zarr.py`);
- `simple_rl/learn.py`'s `prepare_targets` or the loader (`load_window`, `gather_rows`).

They are cheap, so run them anyway. A diff in `simple_rl/learn.py` alone, for example
adding `--value-rows-per-game`, is not a redo if (a) passes with that option off.

**(a) The decoded training input is identical.** Run this at C0 (from its worktree)
and again at `$PIN`, once per cell with the cell's own flags. The output must be
byte-equal, key by key. It digests every train row the cell reads, after
`prepare_targets`, plus the row ids. This is the content the batches are drawn from.
The sampler's order is not digested: it is fixed by `--seed` and is a learn.py
property that does not change what the inputs are.

```bash
for c in S Zq P0; do
  PYTHONPATH=. $PY - $R/collect $COLLECT $TRAIN ${CELL[$c]} > $R/guard_a_${c}_$(git rev-parse --short HEAD).json <<'EOF'
import hashlib, json, sys
from pathlib import Path
import numpy as np
from scripts.simple_rl_loop import parser, train_settings
from chess_anti_engine.simple_rl.learn import load_window, prepare_targets
run = Path(sys.argv[1])
a = parser().parse_args(["run", "--run-dir", str(run), "--generations", "1", *sys.argv[2:]])
s = train_settings(a, run)
train, _held, _gens, ids = load_window(run, 1, s)
rows, _counts = prepare_targets(train, s)
out = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in sorted(rows.items())}
out.update(_row_ids=hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest(), _rows=int(ids.shape[0]))
print(json.dumps(out, sort_keys=True))
EOF
done
```

This was smoke-tested on a banked small collect (11,274 rows): two runs gave
identical digests. It has not yet been run on the pilot's collect.

**(b) The encoder that wrote the stored `x` is identical.** The collector does not
store `encode_position` output. It stores the C batch encoder that
`selfplay/network_turn.py:_batch_encoder_pair` picks for `lc0_root` history:
`batch_encode_146_lc0_root`, or its `_bf16` twin when the evaluator takes bf16
input. (b) runs both of those, on CBoards, over every position of 300 banked FULL
arena games, with the `rep_fix` regime set explicitly. It runs at C0 and at `$PIN`.
Both digests must be equal.

```bash
B=~/chess-artifacts/operations/rl-claude-20261119
PYTHONPATH=. $PY - <history_rep_fix of $SL: 0|1> 300 \
  $B/search-compose/pgn/t0s1_F12_A.pgn $B/search-compose/pgn/t0s1_Q50_A.pgn $B/arena-hybrid-option/pgn/*.pgn <<'EOF'
import hashlib, json, sys
import chess, chess.pgn, numpy as np
from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import batch_encode_146_lc0_root, batch_encode_146_lc0_root_bf16
from chess_anti_engine.moves.encode import move_to_index
rep_fix.apply(sys.argv[1] == "1")                 # the regime is process-global: set it explicitly
max_games, planes = int(sys.argv[2]), 175
h32, h16, n, games = hashlib.sha256(), hashlib.sha256(), 0, 0
def flush(cbs):
    o32 = np.zeros((len(cbs), planes, 8, 8), dtype=np.float32); batch_encode_146_lc0_root(cbs, o32); h32.update(o32.tobytes())
    o16 = np.zeros((len(cbs), planes, 8, 8), dtype=np.uint16); batch_encode_146_lc0_root_bf16(cbs, o16); h16.update(o16.tobytes())
for path in sys.argv[3:]:
    with open(path) as fh:
        while games < max_games and (g := chess.pgn.read_game(fh)) is not None:
            board = g.board(); cb = CBoard.from_board(board); cbs = []
            for mv in g.mainline_moves():
                cb.push_index(int(move_to_index(mv, board))); board.push(mv); cbs.append(cb.copy())
            if cbs:
                flush(cbs)
            n += len(cbs); games += 1
print(json.dumps({"games": games, "positions": n, "x_fp32_sha256": h32.hexdigest(), "x_bf16_sha256": h16.hexdigest()}))
EOF
```

- **Coverage.** The banks hold ≈ 13.7k full games (the 11-21 regression and head-swap banks), including repetitions, 250–298-ply games and Syzygy endgames. The `history_rep_fix` value comes from `$SL`'s `arch`.
- **Smoke test.** On 300 of these games (24,961 positions) two runs gave identical digests, and flipping `rep_fix` changed both digests. So the check is deterministic and sees a regime change.
- **The shard regime stamps** in each shard's meta (`input_history_encoding`, `history_rep_fix`, `policy_encoding`) must also equal `$PIN`'s resolved collect config.
- **Rule: the collect is redone at `$PIN` iff (a) or (b) fails.**

**Why one commit.** On 2026-11-21 the 800-game head-swap arms no longer reproduced the
2026-11-19 bank game for game. This was true of the PLAIN arms too: plain Q50 matched
on 30 of 424 games and plain F12 on 63 of 799. 24 commits since the bank's
`df55413a6` touch the play path, CUDA graphs among them
(`~/chess-artifacts/operations/rl-claude-20261119/arena-hybrid-option/README.md`).
So every pilot contrast is **within-run, on one commit**:
- every arena result row's `git_sha` must equal `$PIN` (the one-commit rule for the matches is unchanged by C0);
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

`game_resumes` exists only under #1444, which the pilot does not use. On C0 and on
`$PIN` the key is absent, and the check requires it to be absent or empty, so a
collect that #1444 resumed cannot pass. **On this code the only clause that can
fail is `games == 8000`.** The collect gates
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
for c in S Zq P0; do      # S first: see 'Order of work' below
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

### Order of work: S first

The owner, verbatim (2026-11-21):

> "So how long until we have results of this experiment? Im pretttybsure its going to show that training on the search does best so im wondering if maybe we should just skip to that? Is it really important to know by how much?"

**The order.** The order is changed; the content is not:
1. train S, then play match 3 (S vs base, 800 games, plain arena);
2. train Zq, then play match 1 (the gate), then match 2, which stays conditional on the gate as before;
3. train P0, then play match 4 (the graft, in looks).

**Why the order is free.** The four matches are separate arena invocations with
their own seeds and opening blocks. Only match 2 is conditional (on the gate), and
nothing conditions on S.

**S's rules are unchanged.** Its reading rule and its extension rule stay exactly as
registered. The owner's expectation is a prior, not a change to the thresholds.

**Steps 2 and 3 run whatever S reads.** The Step 2 loop may train all three cells
back to back, or interleave each with its matches. The S read's real path also
waits for `$PIN`, which needs #1462 and #1463 merged and this record merged.

**What an early S read licenses:** nothing beyond the registered reading. In
particular, a second generation from S is its own record. The owner's own concern
about an unanchored target over many cycles is quoted in the DRAFT section below.

**Wall-clock path to the S read** (measured rates where they exist):

| Step | Time | Basis |
|---|---|---|
| collection | ≈ 1.6 h | 5,021 games/h measured |
| S training | ≈ 16 min | ≈ 758k rows minus the 5% hold-out ≈ 720k trainable rows, one presentation, ≈ 1,406 optimizer updates at batch 512. Updates/s at batch 512 is **unmeasured**; 16 min is the floor rate of ≥ 770 rows/s measured at batch 256 |
| match 3 | ≈ 0.19 GPU-h | 800 games, plain, measured with CUDA graphs |
| **Total** | **≈ 2.1 h** of compute | from the collection's start to the S read, plus the wait for `$PIN` |

### The matches

Seeds are fixed here. A void look's backup seed is the look's seed + 100 (see the
max-plies rule).

| # | Match | A: candidate | A: reference | B: candidate | B: reference | Games | Seed | Looks |
|---|---|---|---|---|---|---|---|---|
| 1 | gate: Zq vs base | Zq | SL | SL | Zq | 1,600 | 2026112101 | one |
| 2 | head-to-head: Zq + recal vs Zq (**skipped unless match 1 reads UB95 < 0**) | Zq `--candidate-value-recal $RECAL` | Zq | Zq | Zq `--reference-value-recal $RECAL` | 1,600 | 2026112102 | one |
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
the look's own seed (2026112141, 42, 43).
- **Sizing correction (#1463 review).** A 1,600-game look is **400 openings**. Each opening is played four times: two colours in run A and two in run B.
- **Recurrence across looks.** It is light at the pilot's book: 759 / 800 and 1,084 / 1,200 distinct openings after looks 2 and 3. The SE inflation is 1.03 at look 3, and the information fractions are 0.35 / 0.68 / 1.
- **Alpha check.** A null simulation through `pool_looks` at 3.471 / 2.454 / 2.004 gives P("pays" | 0) = 0.023–0.027, which is nominal (`plan-review-1463/`). After look k, the multi-look pool
combines looks 1..k and clusters by opening across looks (an opening drawn in two
looks is one cluster). The pool gives the estimate and its SE, and they are read
against z = 3.471 / 2.454 / 2.004:
- stop with "the policy pays" iff est − z·SE > 0;
- stop with "does not pay +10" iff est + z·SE < +10;
- otherwise run the next look;
- after look 3 without a stop, the reading is "unread".

There is no other stopping point.

The pool after look k (k = 2 shown) repeats the run flags once per look, in look
order. `--boundaries` with `--look-seeds` picks the z by look count and refuses a missing, extra or reordered look:

```bash
P0=$R/arm_P0/gen_0001/train/model.pt; d=$R/arenas/graft
PYTHONPATH=. python scripts/arena_seat_balanced_pool.py --arm $P0 --boundaries 3.471,2.454,2.004 --look-seeds 2026112141,2026112142,2026112143 \
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

As run pairs, those are:

| Run pair | Seed | Games per run | Cumulative games |
|---|---|---|---|
| the first read | 2026112103 | 400 | 800 |
| 2026112131 | 2026112131 | 400 | 1,600 |
| 2026112132 | 2026112132 | 800 | 3,200 |
| 2026112133 | 2026112133 | 800 | 4,800 |

The pool takes them in that order with
`--boundaries 1.96,3.471,2.454,2.004 --look-seeds 2026112103,2026112131,2026112132,2026112133`.
It reads z = 1.96 on the first read alone, then 3.471 / 2.454 / 2.004 at the three
extension looks, and it refuses any pair that is missing or out of order.

**False adoption, simulated** (statistics review of #1464,
`plan-review-1464/s_extension_sim.txt`):

| True S | With the extension | Single 800-game read |
|---|---|---|
| −10 | 0.0277 | 0.0251 |
| 0 | 0.34 | 0.20 |

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

**p is chosen from the measurement, by a rule fixed now, before the throughput is read.**
p = the smallest value in {0.5, 0.25, 0.125} whose MEASURED effective policy rows
at 1.6 GPU-h are ≥ collection A's 198k. B's game count is then fixed from the
measured games/h × 1.6 h, not from wall time. **If no p in {0.5, 0.25, 0.125}
reaches 198k effective policy rows, collection B is not run.** Until the throughput run
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
effective rows. The numbers are filled in from `collect-throughput/playout-cap/`, and the rule above then picks p.

**Collection B changes more than the game count** (engine agent, read from main 6c72b821b):
- 75% of plies are played at 16 sims;
- fast plies use the same temperature schedule as full plies;
- root noise is OFF on fast plies;
- tree reuse is carried across fast and full plies, so a stored full-ply target includes inherited visits;
- there is no per-row capped flag.

**Optimizer steps.** The K cells take fewer optimizer steps than A's cells, because
B stores fewer rows.

**Cells from B:**

| Cell | Training | Contrast | Numbers |
|---|---|---|---|
| **K-blend** | Zq's exact recipe (outcome 0.5 / search 0.5, `stored_search_wdl`) on B's rows | vs Zq on A = "more games, fewer stored rows per game, at equal collection time" | rows to fill from `collect-throughput/playout-cap/` |
| **K-z2** | value loss on r = 2 rows per game, z-anchored (outcome only), via `--value-rows-per-game 2`; policy on every full-search row | the owner's design: little value per game, from z | needs `--value-rows-per-game`; numbers to fill from `collect-throughput/playout-cap/` |

**Added matches** follow the same rule as matches 1–3: seat-balanced, plain speed,
SE = 8.9 × √(800 / games).

| # | Match | Games | SE | Reads | Cannot show |
|---|---|---|---|---|---|
| K1 | K-blend vs base | 1,600 | 6.3 | damage iff UB95 < 0; adoptable iff LB95 > −10. At 80% power (one-sided 0.025, 2.80 × SE): adoption reads at true ≥ +7.6, damage at true ≤ −17.6 | whether K-blend beats Zq: the difference of two vs-base results (K1 − match 1) has SE ≈ 8.9 and reaches 80% power only at ≈ 25 Elo |
| K2 | K-z2 vs base | 1,600 | 6.3 | as K1, as a **package verdict on the owner's recipe** only | why it reads as it does. It changes rows per game AND the value source (z only) AND the games. Isolating them needs cells this record does not add: **K-z-all** (z on every full-ply row of B) and **K-blend2** (the blend on 2 rows per game). Its value dose is ≈ 36k–56k value rows vs K-blend's ≈ 434k, so it also reports a value-drift readout (value KL vs the base on hold-out rows, or a value-half head-swap); otherwise "adoptable" could mean only that the value barely trained |
| K3 | K-blend vs Zq, head-to-head | 1,600 | 6.3 | "collection B's package beats A's at equal collection time" iff LB95 > 0; 80% power at a true difference ≥ 17.6 (19.0 with the pool's measured SE) | anything about the base. It is a **package contrast for collection B**: games, rows per game AND the game distribution all change (the fast-ply list above) |

K3 exists because K1 − match 1 is underpowered (80% power only at ≈ 25 Elo). Both nets are plain, so K3 plays
at plain speed.

**Added cost:**

| Item | GPU-h |
|---|---|
| collect B | 1.6 |
| two cells (B has fewer stored rows than A) | ≈ 0.3–0.5 |
| three 1,600-game plain matches (3 × 0.38) | 1.14 |
| **Total** | **≈ 3.0–3.3**, on top of the pilot's 4.0–6.0 |

### 3. Parked, with launch conditions (not queued): the anchor already exists in the original DeepFin loop

The owner, verbatim (2026-11-21):

> "Have you seen how we set this up in original deepfin? We had a complicated setup that addresses a lot of ur concerns. Like we ran full strength sf and limited by forcing random moves within its top choices etc and we had multiple output heads to predict our search and its and we learned on a combo of it our and game result and also we did some self play and some sf so they all balance each other with different bias but still anchored to rreality"

He is right: the production loop already has this design. The simple loop strips it.

**What the original loop does.** Read from `docs/model_heads.md`, the live
`configs/pbt2_small.yaml` and the code on main 6c72b821b; values are the live
yaml's, not main's.

- **Main `wdl` head: one soft CE against a three-way blend.** The blend is
  `game_frac`·outcome + `sf_wdl_frac`·SF cp-logistic label +
  `search_wdl_frac`·own search-root WDL (`train/losses.py::normalize_value_blend_fracs`).
  - Live shares: `sf_wdl_frac` 0.69 (`sf_wdl_frac_floor` 0.69) and `search_wdl_frac` 0.31, so the outcome share of the MAIN head is 0 there.
  - The yaml comment records an earlier realised blend of 0.35 outcome / 0.45 SF / 0.20 search (iter 478).
  - In the live config the game result enters through the trained `categorical` head (`w_categorical` 1.0, blended into the search value at `categorical_blend_frac` 0.69).
- **SF games.** `selfplay_fraction` is 0.8. The other games are played against SF:
  - SF's strength is limited by choosing uniformly among its MultiPV candidates within a PID-controlled regret limit (`stockfish_turn.within_regret_candidates`; `sf_pid_target_winrate` 0.5; `sf_nodes` 75,000 for play).
  - Labels are always from full-strength SF's search, never from the handicapped move: `sf_label_nodes_floor` 150,000, `sf_label_nodes_cap` 200,000, `sf_multipv` 6, `sf_workers` 4.
- **Auxiliary heads:**
  - `sf_eval`: `w_sf_eval` is 0.0 live.
  - `policy_sf`: the opponent's reply.
  - `policy_soft`: `w_soft` 1.0.
  - `policy_future`: `w_future` 0.01.
  - volatility: 0.1, and SF volatility 0.05.
  - moves-left: 0.02.
- **What the simple loop strips:**
  - `simple_rl/learn.py` hard-sets every auxiliary weight to 0 (`w_soft`, `w_future`, `w_sf_*`, `w_categorical`, `w_volatility`, `w_moves_left`, ...) and trains only `w_policy` + `w_wdl`.
  - `simple_rl/collect.py` forces `selfplay_fraction` 1.0, with `sf_nodes` 1 and no SF label stored.
  - Its `TRAIN_KEYS` carry no `sf_wdl`.

**Existing blend-weight mechanisms (do not rebuild).**

The owner, verbatim (2026-11-21):

> "We had some experiments tying the weight of our search to opponent search based on distance from z or how much ti weight it based on how many moves left and other things"

Each mechanism below is in `chess_anti_engine/train/losses.py` on main. Live values
are from the live `configs/pbt2_small.yaml`; verdicts are from `docs/experiment_ledger.md`
(line numbers on main 6c72b821b).

| # | Knob | What it does | Live value | Recorded verdict |
|---|---|---|---|---|
| 1 | `wdl_terminal_outcome_plies` / `_full_plies` / `_sf_frac` (`terminal_outcome_transfer_taper`) | Moves the SEARCH share of the blend to the game result near the end of the game: full weight within `_full_plies`, tapered to zero by `_plies` via `moves_left`. `_sf_frac` lets the outcome also take part of the SF share | unset, so off (defaults 0 / 2 / 0.0) | **Biased by construction** (the weight depends on plies-to-end, i.e. on the future; see below). **Calibration** (ledger L38651, 2026-08-07, `data/terminal_outcome_calibration_20260807/`, 149k rows): the result's noise as a value label is sd ≈ 0.10 at 1–2 plies from the end, rising to 0.77 deep. It crosses the search component's noise at d ≈ 6–7, and d ≤ 6 touches 4.7% of rows. **Stage-1 screen: KILL, both arms** (L38726): the endgame CI included 0 for both, e.g. conservative +1.24 [−2.93, +5.53] cp. Stage 2 never launched |
| 2 | `use_adjusted_wdl_target` (+ `adjusted_wdl_regret_source` / `_scale` / `_cap`) | Discounts the game result by the opponent's later regret, so a result owed to SF's later mistakes counts for less. **This is the "gift win" correction** | `false` (the three sub-keys are deleted from the yaml as unreachable, L57920–57926) | **Biased by construction** (the weight depends on future regret). **2026-07-20/21 multi-seed NULL** (L3775): its 2× dose won only at seed 0, and "at s0 EVERY value-pressure lever won" |
| 3 | `sf_search_dampen_sf_low` / `_high` | Lowers the SF share on rows where SF and our search disagree in sign, separately for each direction | 0.0 / 0.0 | **Biased by construction** (the weight depends on the labels). **No screening arm found.** A grep of the ledger for `sf_search_dampen` finds only "inert" and "restored to the yaml" entries (L7059, L25754, L41637, L46472, L61867). It was restored because `scripts/value_optimism.py` requires the keys to be present (L41637) |
| 4 | `sf_wdl_frac` → `sf_wdl_frac_floor` interpolation by `wdl_regret` | The SF share rises with difficulty past a 0.10 regret knee | pinned: frac = floor = 0.69 | **Never fired in the current run:** realised `wdl_regret` never reached 0.10 (max 0.0984) in 478/478 iterations, and it was pinned to the realised value (live yaml comment; `rl_loop_audit` I5). An earlier run crossed the knee on 15.3% of 646 iterations (`docs/rl_loop_audit.md` item L15, ≈ L820) |
| 5 | `sf_wdl_conf_power` / `sf_wdl_draw_scale` | **Affect ONLY the auxiliary `sf_eval` mask, never the value target** (`losses.py` docstring at L1916; pinned by `tests/test_sf_wdl_conf_knobs_are_aux_only.py`) | as live | Listed so nobody tunes them expecting a value effect |

**How to read these verdicts.**
- **They are leads, not closures, for the SL base.** They came from offline `value_regret` screens on the OLD net at small dose. This lane has since found that no offline value metric predicts the SL base's 64-sim strength.
- **The calibration in (1) bears directly on the pilot.** A one-hot result far from the end is the noisiest of the three labels. That is the pilot's damage hypothesis seen from the other side.
- **Design option for the arm after the pilot** (not a change to the pilot's cells): wherever the result enters (K-z2, or any z share), its weight would fall with plies-to-end, using a taper like (1)'s. **Marked biased:** like (1)–(3), it is a weight that depends on the future, which the statistics review shows reintroduces bias. It needs the calibration table below before any use.
- **What is and is not calibrated** (statistics review of #1464, `plan-review-1464/outcome_conditioned_sim.txt`, `outcome_conditioned_variants.txt`):
  - FIXED-weight blends are calibrated. This supports the primary design.
  - ANY weight that depends on labels or on the future reintroduces bias, even with perfectly calibrated evaluators. In the model, a threshold rule was biased by ± 0.065.

**Primary design: restore the three-way value blend and an SF-opponent share of
games in the simple loop.**
- **Unconditional blend.** It has no result-conditioned switch between sources, so the bias question raised by the alternative below does not arise.
- **Gift wins are corrected.** Full-strength SF labels every position, including positions the limited opponent let us off.

**Staging: one data-affecting change per readout.**

| Stage | Change |
|---|---|
| Stage | Change | Control, and what is held equal |
|---|---|---|
| (a) | SF labels plus the three-way blend, on self-play games only | Relabel collection A's positions OFFLINE with SF and compare against Zq **on the same rows**, stating the three shares. A freshly labelled collection would also resample the games |
| (b) | add the SF-opponent game share | This changes games, rows per game and opponent difficulty. Use a FIXED regret limit, not the PID, inside a one-generation readout, and state equal games or equal time |
| (c) | add the auxiliary heads | Not a pure training change: `categorical_blend_frac` feeds the categorical head into the MCTS value, which is a search-config change. Hold `categorical_blend_frac` at 0 in (c), or read (c) as a search change |

**Before the primary design is trusted.** Each item has its status on main 6c72b821b,
from the code and the audits. "Unknown" means not re-verified in this session.

| Item | What must hold | Status |
|---|---|---|
| One-ply alignment of the SF label | `rec.sf_wdl` is SF's search AFTER our move (P1, SF to move, POV-flipped; `stockfish_turn._sf_result_wdl_for_record`; `docs/model_heads.md` `policy_sf` row). As a value target for the row's position, it is the value after our chosen move, one ply off. **Direction:** a label taken after our move is at most the P0 value, lower by our move's regret, so it carries a systematic pessimistic offset equal to the mean regret. The P0 fields (`sf_p0_*`) are built from the PARENT row's MultiPV (`finalize._build_sf_p0_regret_vector`). | **open, by design.** Decide whether the value target takes P1 as-is (the value under our move, pessimistic by the mean regret) or a P0 eval; no P0 value field exists today |
| Label coverage (the signature defect) | A row with no usable `sf_wdl` silently falls back to the one-hot result for its SF share (`train/losses.py` ≈ L1891). If the labeller cannot keep up, the blend turns z-heavy on a non-random subset with no error | **open.** Require 100% label coverage, or make `train/value_blend_guard.py` fail the run instead of reporting |
| Dirty-TT label regime | Labels should be searched from a known TT state | **open.** Default label queries run with `fresh=False` (warm TT carried across positions; `stockfish/pool.py`). Only escalation re-queries run cold (`fresh=True`, `stockfish_turn.py` ~L745–753, L1365) |
| MultiPV-6 regret-tail fabrication | No fabricated values in the stored targets | **open, for the POLICY fields.** `finalize._build_sf_p0_regret_vector` imputes a regret for about 68% of legal moves at MultiPV 6, and `record_sf_p0_regret: true` banks it into every shard (`docs/experiment_ledger.md` ~L980–1016). It does not touch `sf_wdl`, the value label, which is from SF's best line |
| Unconverged labels (about 3%) | A label must be a converged search | **open.** About 2.92% of labels were found unconverged, and escalation is OFF: `sf_label_escalate_q_gap` defaults to 0.0 (`selfplay/config.py` ≈ L286) and is unset in the live yaml |
| cp-logistic slope and draw width | The cp → WDL map fits the data | **open.** Live values are 0.006 and 120 cp. The replay-fit notes (about 0.0045–0.0055 and 130–145 cp) are comments in MAIN's `configs/pbt2_small.yaml` (≈ L500–501), not the live one. Not refitted |
| `search_wdl_draw_mode` | The search-WDL component's draw channel is the intended one | **unknown for the simple loop.** The live yaml does not set it, so the runtime default `net_raw` applies (`stockfish/wdl.py` `SEARCH_WDL_DRAW_NET_RAW`) |
| "The SF share is load-bearing" | NOT to be cited | **operator's rule** (`docs/model_heads.md`; ledger, 2026-08-23): the 0.64 → 0.40 winrate fall is an old, uncontrolled result and must not be cited as a mechanism without a fresh preregistered re-ablation |

**Cost: SF label CPU against our collect rate.**
- The pilot collects ≈ 5,000 games/h × ≈ 95 rows ≈ 475k rows/h.
- **Measured nps on this box:** 290–370k nodes/s per engine with 32 engines (`docs/sf_cpu_cost_split.md`, ≈ L164).
- At the production label budget (150k–200k nodes, MultiPV 6), a label costs ≈ 0.4–0.7 core-seconds. 475k rows/h then needs ≈ 53–91 cores, **more than the machine has**.
- (A first draft of this record assumed ≈ 1M nodes/s per core and underestimated the cost about 3×.)
- **So at 64 sims on every ply, SF labelling, not the GPU, is the limit.** Fast-move collection makes it worse, because games per hour rise while every stored row still needs a label.
- **The measurement that settles it:** the production label path (MultiPV 6, warm TT, history moves, the Syzygy pair), measured beside the collect's own CPU load.

**Launch condition (unchanged):** the pilot has read.

**Design option for the multi-generation run (not the pilot): an outside gauge per generation.**

The owner, verbatim (2026-11-21):

> "In self plah only games you also get 50% winrate"

**The problem.** Self-play has no built-in progress gauge. The original loop's
regret window is one, and it is measured against an independent engine.

**The option.** Give each generation of the small loop one outside gauge, reported
report-only beside the match against the SL base. Either:
- **a fixed-handicap match against limited Stockfish:**
  - the same `within_regret_candidates` handicap, at a pinned regret limit and a pinned node budget;
  - the production Syzygy pair;
  - banked finals of a fixed size;
- **or the regret level at which the generation scores 50%.**

**Caveat (CLAUDE.md).** Regret measures net plus search, never the net alone. It is
comparable across generations only while the search settings and SF nodes are
frozen. Difficulty is regret AND nodes.

**Cost:** SF CPU per game at the play node budget, to measure.

#### Alternative: the owner's outcome-conditioned value source (designed, not queued)

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

  **Bias, from the statistics review** (`plan-review-1464/outcome_conditioned_sim.txt`).
  This alternative is biased even with perfectly calibrated evaluators: E[target | s] − s
  reaches −0.024 at a 2-ply lag and −0.061 at 8 plies, because t\* looks ahead to the
  end of the game.

  The definition above is **variant A** (the source is chosen by the result, with t\*
  found by looking ahead).

  **Variant B: trust SF only on a confident jump, with no look-ahead.**

  The owner, verbatim (2026-11-21):

  > "I was thinking id fix that bias by only trusting sf when search found something iys confident in and its winrate jumped. It's very strong its basically always right in that case especially when it wins right after"

  **Definition.** On each row, SF's value is used when a trigger fires; otherwise
  the row keeps our search WDL (or the fixed blend). The trigger depends only on SF's
  evaluations up to that ply. It fires when both hold:
  - SF's win probability moved by at least Δ from the previous ply;
  - it is now at or beyond a confidence threshold.

  It fires in EITHER direction, for a jump against us or in our favour. "Wins right
  after" is the future and is NOT part of the trigger. It is reported only as a check
  statistic: the share of triggered rows whose game ends as SF said.

  **Properties:**
  - **No look-ahead.** It removes the look-ahead that biased variant A.
  - **Selection on the label remains.** Rows that look alike to our net split into
    triggered rows (SF's value written in) and untriggered rows (left at our
    search's average). So a one-sided trigger is pessimistic by about
    P(trigger among look-alikes) × the jump size, concentrated in sharp positions.
    A two-sided trigger reduces this but need not cancel it.
  - **Quiet positions keep our search's value.** That avoids importing SF's
    cp-logistic calibration in quiet positions, which is the cost of the primary
    (SF-everywhere) design.
  - **Model bias** (statistics review, `plan-review-1464/variant_b_sim.txt`,
    `variant_b_per_trigger.txt`; settings (Δ, threshold) = (0.10, 0.80) and
    (0.20, 0.90); smooth and tactical-jump models):
    - The bias in an s-bin = trigger share × E[q − s | triggered, bin].
    - Triggers fire on ≤ 0.4% of rows in the model, so |E[target | s] − s| ≤ 0.008
      (usually 0.001–0.006). That is an order of magnitude below variant A's
      −0.024 to −0.061.
    - Per triggered row, one-sided is −0.23 to −0.34 (to −0.6 / −0.8 in the high-s
      deciles): pessimistic.
    - Two-sided is ≈ 0 overall, but ± 0.07–0.15 by bin (sharpening, i.e.
      overconfident). **Two-sided trades pessimism for overconfidence; it does not
      remove the bias.**
    - **The launch table MUST report the trigger share per bin.** The bias is small
      only if the real share is small.

  **Qualification: some of this bias is correctable after training.**

  The owner, verbatim (2026-11-21):

  > "It's easy to return search to be more optimistic or less sharp and confident."

  What the bank supports:
  - **Correctable kind.** A bias that is a function of the value alone (a level
    shift, or sharpening / flattening) is correctable after training by the
    two-parameter recalibration (T, draw bias).
  - **Evidence.** Banked: F12 −34.9 → −0.9 and Q50 −63.2 → −14.3 (head-swap bank).
    The 11-21 reproduction R3 at −31.8 [−50.0, −13.7] makes the size of the
    recovery uncertain. The pilot's match 2 is the registered test.
  - **Consequences.**
    - Variant B's two-sided residual (sharpening by value bin) is of the
      correctable kind.
    - What recalibration cannot fix is error that differs by position type at
      equal value, for example one-sided pessimism concentrated in sharp
      positions.
  - **Design option for the multi-generation run (an option, not a plan): a
    per-generation recalibration against the game results.**
    - Fit (T, b) on held-out GAMES against their results and fold it into the
      checkpoint's WDL head. With 2 parameters, thousands of games give a tight
      fit.
    - It uses z as an anchor for level and sharpness, without putting the noisy
      one-hot result into the training target.
    - Today `scripts/fit_value_recal.py` fits against a REFERENCE NET's WDL
      (cross-entropy to `p_ref`). A result-referenced fit (cross-entropy to the
      one-hot z of each row's game) would be a small addition.

  **The launch measurement, the same for every candidate on the SAME games.**
  The candidates are:
  1. our search WDL alone;
  2. variant B at two (Δ, threshold) settings fixed in advance, (0.10, 0.80) and (0.20, 0.90);
  3. the fixed three-way blend.

  - **Decision quantity: the PAIRED column** E[t_X − t_search | bin] on the same rows.
    At 500 games its SD is ≤ 0.001 (statistics review).
  - **The z column,** E[t − z | bin], is kept only for the search's own absolute
    calibration. 500 games cannot resolve 0.02 absolute calibration per bin: the SD
    is 0.028–0.039 in the model, so ≈ 0.05–0.1 is detectable.
  - **Binning.** All candidates are binned on ONE common variable: the stored search
    WDL (primary), or the base's raw value. Bins on each candidate's own value are
    secondary.
  - **Also reported:** target sharpness, and variant B's trigger share per bin.

  The SF values come from an **offline SF relabel of a game sample of collection A**:
  - whole games, every ply;
  - the production label path: MultiPV 6, the label node budget (`sf_label_nodes_floor` 150k), the Syzygy pair, a warm TT as in production, and each game's moves sent as history (`position <root> moves ...`).
  - **Gap: the move record.** Collection A does not store its games' moves. Shard rows hold `x` plus `game_id` / `ply_index`, and `games.jsonl` has no move list. So the relabel needs a collect that writes each game's move list (a small addition to `simple_rl/collect.py`'s `on_game`), or a collection made with it. A deterministic replay of A's games is not a substitute, because GPU play is not bit-reproducible (see "Why one commit").
  - **P0-aligned.** Collection A stores no SF label, so the relabel queries each trained position itself (P0, our side to move), not the position after our move. This differs from the live loop's stored label, which is P1. The P1 value (after the stored move) can be added for comparison at about twice the cost.

  **Sample size and cost** (corrected; the first version overstated the size 2×).
  - **Size.** 500 games ≈ 47.5k positions at ≈ 95 rows per game.
  - **Cost.** At 0.4–0.7 core-s per label (`docs/sf_cpu_cost_split.md` ≈ L164: 290–370k nps per engine), that is ≈ 19k–33k core-s ≈ 5.3–9.2 core-h, about 0.3–0.6 h on 16 cores. It needs no GPU and is run outside training.
  - **Which games.** The hold-out has only ≈ 400 games (5% of 8,000). A target-only table does not need a hold-out, because no net is scored. A cannot be relabelled at all, because it stores no move lists (above). So the sample is a **SEPARATE ≈ 500-game collection** (≈ 6 GPU-min). It is acceptable under these conditions:
    - the same SL base, the same collect flags and the same TRAIN-split openings as A;
    - only the seed changed;
    - the move-list addition only writes, and does not change play.

    Under those conditions its games are exchangeable with A's, and the table measures the target rules on that distribution.

  The gift-win share (won-game rows where full-strength SF had us at win probability
  ≤ 0.20) and the shape statistics (entropy, |W − L|) stay report-only.

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
| L1 | Launch the pilot as recorded here | yes, once every prerequisite above is merged and `PIN` is written below | **Step 1 (the collection) only: GO.** Owner, 2026-11-21, in reply to "say go and I'll start the pilot collection as soon as the Ceres run finishes": "I guess? I dont see downsides?" Run at C0 = `6c72b821b`. Training, the fit and the matches wait for this record's merge |
| L2 | Budget | 4.0–6.0 GPU-h (expected about 4.8), plus the collect window of about 1.6 h with no gaming | |
| L3 | Cell changes | none (Zq, S, P0 as in the amendment) | |
| L4 | S's role | a one-generation shape diagnostic only; z in every multi-generation recipe | |
| L5 | Add collection B and cells K-blend / K-z2 with matches K1–K3 (≈ 3.0–3.3 GPU-h more) | yes, once `--value-rows-per-game` and the playout-cap throughput are banked; p from that measurement | |
| L6 | The parked anchor: restore the original loop's three-way value blend plus an SF-opponent share in the simple loop (primary, staged a → b → c), or the outcome-conditioned alternative | stay parked until the pilot reads; then the primary design, after the trust checks and the label-throughput measurement | |
| — | C0 (the collection's commit) | `6c72b821b` | |
| — | `PIN` (training, fit and matches; must descend from C0) | | |
| — | `git diff --name-only C0 $PIN`: the tripwire list; guard checks (a) and (b) are run, and the collect is redone at `$PIN` iff (a) or (b) fails | | |
| — | Step 1 launch | **2026-11-21T06:41:24-05:00** at C0 = `6c72b821b` from `~/projects/chess-worktrees/rl-run1-pilot-6c72b821b`. Launcher `~/chess-artifacts/operations/rl-run1-pilot/launch_collect.sh` runs the verbatim Step 1 command with the live `.venv` python. The Step 0 split check passed (sha256 match; 153,842 / 4,000 / 16). Its exit code and time go to `collect.exit` | |
