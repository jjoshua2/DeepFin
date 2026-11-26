# 2026-11-26: RL — deep wide search on a quarter of moves, fast on the rest, and an ordering term on its ≥ 2-visit children (pre-registration)

**Status: design merged; launch addendum below (2026-11-26).** No GPU work has run under this record. Launch needs this record merged, the engine item below
(only for the ≥ 3-visit variant), a launch addendum that pins the commit, the bank scripts' sha256 and the Stage 0
numbers, and a free GPU.

## Why: the owner's ask, and what the full-width kill did and did not close

The owner, 2026-11-26: "for the wide test i thought we were going to test widening it to like 80 so we'd have room
for each move to get 2 visits since that is what it took to get better value head then base? 1 didn't seem to help
but 2 did. and we could even experiment giving each 3 with like 200 nodes etc along with doing like 75% of the
moves fast?"

- **What the kill covers.** The full-width record was killed at Stage 1
  ([record](2026-11-21-rl-run1-root-full-width.md#results-and-verdict-2026-11-26-killed-at-stage-1-the-record-stops)).
  - It covers one claim: with the SL base as actor, a wide search on **every** ply at equal cost (33 sims) gives a
    **policy** target worse than stock 64 by deep SF 8M: +2.15 cp [+0.13, +4.13], a fragile, tail-driven kill.
  - Wide at matched sims (64, 1.54× the cost) was null: +0.65 [−0.89, +2.20]
    (`RFW/stage1/scores/scores.json` [`9969fa7d`]).
  - Nothing has ever trained on the ≥ 2-visit child values, and no deep/fast allocation has been tested.
- **"1 didn't help, 2 did".** This is the two-visit scoping (`RC/two-visit-children-scoping/README.md`
  [`7e5a9693`]).
  - The searched order of exactly-2-visit children beats the net's own prior against deep SF: B +0.0267
    [+0.0086, +0.0444], C +0.0242 [+0.0069, +0.0430].
  - 1-visit children were not shown informative: a 1-visit q is the collecting net's own value.
  - That is an ordering measurement on one net with an SF reference. It is not a value-head result.
- **More sims per root did not buy more on the proxies** (`RC/visits-per-child-scoping/README.md` [`a77b9936`],
  `RC/visits-per-child-scoping/out/analysis.json` [`f15c8937`]).
  - All-children ordering: W80 −0.0018, W128 −0.0039 [−0.0075, −0.0001], W270 −0.0039, each against wide@64.
  - BT4 CE of the policy target rises with sims: 2.390 at wide@64, 2.427 at W80, 2.532 at W128, 2.810 at W270.
  - These are one net and proxy rulers. The owner's question is whether **training** on the deep moves helps, so it
    is tested by training.
- **What changed today: generation size matters.** The data-volume run trained `gen1_baked.pt` once over 2.77M
  rows (kC):
  - kC vs the parent: +5.08 [+0.48, +9.67];
  - kC vs the SL base: +8.52 [+4.00, +13.05];
  - the 1× bank vs the parent: −5.13 [−9.47, −0.79];
  - kC minus 1× passes the registered one-sided contrast
  (`~/chess-artifacts/operations/rl-run1-volume/verdicts.json` [`fab91794`]).
  - So a deep-move design is costed in **policy rows per GPU-hour** and judged at **equal GPU-hours**. That is the
    test it must pass.

**Evidence log first.** I grepped `2026-11-21-rl-loop-evidence-log.md` for playout cap, full-prob, fast,
value-only, root_full_width, two-visit and visits-per-child. Everything this record cites is banked:
- in the log: the playout-cap throughput at 64 / 16 sims (11-21) and the two scoping banks (11-22);
- in their banks, cited directly because their log rows are not on main yet: the full-width Stage 0 / 1 (11-26) and
  the data-volume matches (11-26, `verdicts.json` [`fab91794`]).

No row measures training on deep/fast data or on an ordering term over ≥ 2-visit children.

## Two ideas, two readouts

| Idea | Arm | Compared with | Readout |
|---|---|---|---|
| **A. Allocation:** deep wide search on a share p of plies, fast elsewhere, only deep plies stored | **D** (deep/fast collection, stock training recipe) | **C** (stock 64 on every ply), equal collection GPU-hours | Readout A: d̄_D − d̄_C |
| **B. Ordering term** over root children with ≥ 2 visits | **D+O** (the same D rows, plus the sibling screen's term) | **D** | Readout B: d̄_{D+O} − d̄_D |

The arms are separated so that each idea gets its own readout. B is read on D's collection, so its only difference
from D is the term.

## Collection (idea A)

**The collector already does this, on the C path, today.** No engine work is needed for D.
- `--full-prob p` draws full plies per game from the game's own RNG (`selfplay/network_turn.py` `_draw_is_full`).
- A full ply searches `--sims`, a fast ply `--fast-sims`. The per-game budgets go into one C runner call
  (`combined_sims`).
- `--root-full-width` runs round 0 **only on full plies**: `per_game_root_full_width` is the `is_full` mask, which
  reaches `mcts/gumbel_c.py` `_full_width_round0(board_mask=…)`.
- **topk changes only the deep plies.** Width is min(topk, L, (budget + 1) // 2) (`_realized_candidate_width`), so a
  16-sim fast ply ranks at most 8 candidates whatever the topk. One topk does both jobs.
- **Fast plies store no row.** `fast_value_rows` defaults to False (`CollectSettings`; the comment there, and
  `finalize`).

**D's settings:**
- p = 0.25, deep plies at 80 sims with `--root-full-width` and topk 40, which is (80 + 1) // 2, the smallest topk
  giving the full realised width at 80 sims;
- fast plies at 16 sims, storing no row;
- everything else the pilot's flags, as C.
- Why 80: W80 visits 22.95 of 23.62 legal children ≥ 2 times on the audit rows, against 21.42 for wide@64
  (scoping table, "children ≥ 1 / ≥ 2").
- Why 16 fast sims and p = 0.25: that is the measured playout cap (below), so the fast half is banked, not a guess.

**One variable at a time: fast plies store nothing (`--fast-value-rows` off).**
- A fast-ply value-only row would carry a 16-sim root q into the Zq value target. That is a second, different
  value target. In the RL pilot, value-target choices alone cost 10–20 Elo (raw Zq −10, S −20; evidence log, pilot
  rows).
- So D's stored rows are deep plies only, with the same value target as C's rows.
- A value-only-rows variant is a later record, after Readout A.

**Behaviour of fast plies, from the code and banked** (evidence log, caveat 11 and the 11-21 playout-cap row):
- no root noise;
- the same move-temperature schedule as full plies;
- the tree is reused across fast and full plies (carried visits on full rows: 8.5 at p = 0.25 against 12.1);
- games are more decisive: 0.606 decisive at p = 0.25 against 0.510.

These are part of what "deep/fast" means, so they are confounds of arm D, not removed (see Confounds).

**The ≥ 3-visit variant (D3, conditional; this one needs engine work).**
- Round 0 gives each unvisited child **one** visit (`_full_width_round0`), and sequential halving then concentrates
  the rest. So more sims do not give every child 3 visits: W270 has only 16.47 of 23.62 children with ≥ 3.
- D3 is p = 0.25, with a **k = 3 visit round 0** plus 128 halving sims (about 3 × 23.6 + 128 ≈ 200 leaves per deep
  ply), so every legal child gets ≥ 3.
- **Engine work, about one day:**
  - Add `root_full_width_visits` (default 1) to `GumbelConfig`.
  - Round 0 repeats its forced pass k times. The docstring says the unchanged C state machine already runs "a board
    whose candidate list is its k unvisited children and whose budget is k", so no C or ABI change is expected.
  - Thread it through `CollectSettings` → `SearchConfig` → `GumbelConfig` → `run_gumbel_root_many_c`.
  - It is orchestrated in `gumbel_c.py`, the C runner's Python driver, so it is not one of the GumbelConfig fields
    the C path drops. The Python reference search already refuses `root_full_width`, and must refuse k > 1 too.
- **Take-effect proof:**
  - read from the **stored rows**, the consumer: on every deep row of a fresh root, the minimum root-child visit
    count over legal children is ≥ k, and `root_prelude_visits` = k × the unvisited legal count;
  - a CPU unit test with a mutant (k ignored ⇒ the test fails);
  - Stage 0's realised gate.
- D3 runs Stages 0–1 only if the engine item is merged. It enters Stage 2 only if it passes Stage 1 and Readout A
  reads "D beats C". It is never a substitute for D.

**Projected cost: an estimate, which Stage 0 replaces.**
- Measured stock 64 on the full-width record's Stage 0 (3 × 128, SL base, pin `0d4d11572`): 152 rows/s steady and
  69.4 forwarded rows per stored row (`RFW/stage0/summary.jsonl` [`cebc9f1a`]).
- That table's on@64 legs (103.0 forwarded / row, 99.1 rows/s) scale with forwarded rows per stored row to within 3%.
- Leaves per ply from the scoping bank: stock 63.4, W80 105.0, fast 16 ≈ 16.
- Forwarded rows per stored row for D ≈ 1.1 × (0.25 × 105.0 + 0.75 × 16) / 0.25 ≈ 168. That is about 63 stored
  rows/s, **≈ 0.41× C's policy rows per GPU-hour**, with about 1.65× the games.
- D3 ≈ 1.1 × (0.25 × 199 + 12) / 0.25 ≈ 272 forwarded per row. That is ≈ 39 rows/s, ≈ 0.26×.
- **Effective rows go the other way.** Within-game correlation (ICC 0.03) discounts 95 rows per game more than 24.
  At p = 0.25 the measured 64 / 16 cap gave 283k effective policy rows per 1.6 GPU-h, against 207k at p = 1 (11-21
  playout-cap row, `RC/collect-throughput/playout-cap/README.md` [`5e0f2666`]). D's deeper plies cost more than
  that cap's, so D's effective rate is expected near C's, not above it.
  - The decision is not taken on this arithmetic. It is the reason the comparison is at equal GPU-hours.

## Training use

**Policy target: deep plies only, by construction.** D stores nothing else. Training is the kC recipe unchanged:
- one presentation per row, mirror 0.5, batch 512;
- LR 3e-6 flat, Aurora, Zq flags;
- the same recalibration fit and bake (data-volume record, "Training, checkpoints and recalibration").

Exactly-once means that D, with fewer rows, takes fewer updates than C at equal GPU-hours (Confounds).

**The ordering term (idea B) is the sibling screen's term, reused and not redesigned**
([sibling screen rev 7](2026-11-21-rl-run1-sibling-value-screen.md)):
- the root-list unit over pairs of eligible children, eligibility being **≥ 2 visits**;
- pair weights with ρ = 0.08 (`n_eff = n / (1 + (n − 1)·0.08)`);
- the forward chunked at 64 pairs, with rev 7's fallback;
- its strength sealed by rev 7's cost probe.

On D's collection the term is restricted to deep rows, which are the only rows. The root-list sidecar comes from the
collector's `record_root_stats` (always on in the collect).

**Memory gate: mandatory, before registration of any D+O number.**
- Revision 6 died on update 2 with an OOM. Peak reserved ≈ 16.95 + 0.0574 × pairs GiB: 31.6 GiB at 255 pairs, about
  20.6 at a 64-pair chunk (`~/chess-artifacts/operations/rl-run1-sibling/oprun/diag_mem/RESULT.md` [`0468d345`]).
- D's deep roots have about 23 eligible children, so up to about 250 pairs per root; chunking is required.
- The gate is rev 7's **3-update GPU smoke** on D's own pool, with peak memory written after each update:
  - abort if peak reserved > 0.85 × the device total, if any update dies, or if memory grows by more than 0.10 GiB
    from update 2 to 3;
  - no retry into a pass.
- **Sequencing:** D+O trains only after the sibling screen's Part 1 has a Stage A reading on the pilot's frozen data.
  The term must first be shown not to damage, under its own record.

## Stages

### Stage 0: cost (GPU ≈ 1.0 h; ≈ 1.5 h with D3)

- **Legs:** C, D, D, C, at 3 × 128 and 1,152 games each, at the launch pin (main containing #1511), with the actor
  `kC_baked.pt` (`c48a7b2c1cc7…`). D3 adds two legs if it is built.
- **Method:** the full-width record's Stage 0 scripts (`run_stage0.sh` v10 with its sealed-games read).
- **Reported per leg:**
  - steady stored rows/s, and forwarded rows per stored row;
  - stored rows per game, and the effective rows per game (ICC 0.03);
  - games/h;
  - peak MiB, with the 83% usable limit;
  - the realised gates: leaf buffer 128 × 2 × topk, chunk rows 512, 0 eager forwards, 0 flushes;
  - **on D legs** (each a gate):
    - **no fast row stored:** every stored policy row that carries `has_search_budget` has `search_budget` == 80 (the
      budget the ply was given, recorded per row), and `policy_rows == rows` in the collect stats. Rows without the
      field (search shortcuts) are counted and reported, not gated;
    - **full width took effect:** `has_root_prelude_visits` on every stored row that carries root stats, and the
      leg-mean prelude > 0.
    - A row with 0 prelude is valid: under tree reuse a root whose legal children are all already visited gets
      none, and solved roots skip round 0. Shortcut rows are counted and reported;
    - **stored share:** stored rows ÷ (games × `mean_plies_after_book`), both from the collect's own `stats.json`,
      |share − 0.25| ≤ 0.02, applied to each process's collect (so the pooled share passes too). The denominator
      counts every ply after the book of every sealed game, so finalize's drops (single-legal-move plies, unresolved
      games) sit inside it; the expected share is ≈ 0.247. Pinned in the launch addendum (`stage0/row_gates.py`).
- **Output:** the GPU-hours G_C that C needs for 2.8M stored rows (the kC dose; ≈ 5.1 GPU-h at 152 rows/s), and
  D's steady games per hour. Neither sets D's final dose: Stage 2's dose rule (below) sizes D from C's realised
  hours.
- **Kill (D):** any realised gate fails, or `check_vram` refuses at topk 40.
  - If `check_vram` refuses, topk 32 is the registered fallback: same realised width on positions with ≤ 32 legal
    moves, stated as a deviation.
  - If a gate fails, the arm is unread until a fix is reviewed.

### Stage 1: target quality (CPU; GPU ≈ 0.1 h only for D3)

**Arms, from banked searches.**
- W80 is already banked on the 2,433 leak-free audit rows: `RC/visits-per-child-scoping/out/arm_W80.npz`
  [`5e32883d`].
- A CPU adapter scatters its per-child arrays into the full-width Stage 1 bank's format. It maps no moves: the
  banked action ids are already `lc0_1858`, mapped upstream by the scoping bank's `alloc_bank.py`
  (`move_to_index_for_encoding`).
- W80 was searched at `fd8c0ce08` with the evaluator capped at 512. It flushed and resumed 13 times (ABI-11
  semantics, no starved leaves), so its targets are valid.
- The banked R64n is the reference, and the full-width Stage 1 showed that M64off reproduces it: top-1 0.9737, paired
  −0.075 [−0.286, +0.145].
- D3 needs one new GPU search (about 0.1 h) once the engine item exists.

**Scorer:** a revision of the full-width record's `stage1_score.py` (`RFW/stage1/stage1_score.py` [`b1e62b3a`])
whose only change is the arm list and the R64n reference: it reads the arms from the adapted bank and writes one
`<arm>_minus_R64n` pair per arm. Pinned by sha256 in the launch addendum, with a fixture showing the W80 − R64n pair
for `exp_cp_lb` and `mass_unscored`. Its scores:
- deep SF 8M `exp_cp_lb` as the primary;
- Ceres expected loss;
- child-value error and clear-pair ordering.

The net's own deeper search is never a reference.

**Rule (per deep arm, paired against R64n), applied in order:**
1. **Coverage.** The paired mean of mass on moves deep SF did not score (arm − R64n) must be ≤ +0.01. Otherwise the
   arm is **UNREAD** until an SF top-up amendment. `exp_cp_lb` gives unscored moves only the 10th-best regret, so
   extra unscored mass would flatter the arm.
   - The top-up is the full-width record's procedure: every move with ≥ 1e-3 mass missing from the SF table is
     labelled at 8M nodes, cold TT, production Syzygy, `searchmoves`, into a copy of the table, and the arm is
     rescored. It is a new amendment, never a silent rerun.
2. **Kill.** STOP that arm iff the paired `exp_cp_lb` has LB95 > 0 and the estimate is > +1.2 cp (the seed-noise
   half-width).
3. Otherwise it proceeds to Stage 2.

**Report-only:** leave-top-game-out and drop-two-games intervals, since the full-width kill was tail-driven.

Stage 1 judges only the deep ply's policy target. It cannot see the allocation, which only Stage 2 tests.

### Stage 2: the generation at equal GPU-hours (GPU ≈ 22 h)

**Collections.** From `kC_baked.pt`, at the launch pin, one seed per arm:
- C: stock 64 on every ply, ≈ 2.8M rows in G_C;
- D: deep/fast, in the same G_C (≈ 1.15M rows projected), sized by the dose rule below.
- **Equal GPU-hours is checked, not assumed.** The clock is each collect's own `active_seconds` (`stats.json`: the
  orchestrator's wall minus `paused_seconds`, so a pause window never counts), at the shape Stage 0 measured
  (3 × 128, 3 processes, the same pin). **Readout A is read only if the realised D / C active hours are in
  [0.9, 1.1]**, and only if both clocks are complete (no game-level resume); otherwise it is "unread: unequal cost"
  or "unread: cost unmeasured".
- **Dose rule (launch addendum).** C runs first, with round(2.8M ÷ its Stage 0 rows per game) games; H_C is its
  realised active hours. D's main collect gets round(0.85 × H_C × D's Stage 0 steady games per hour) games. After it
  seals, if its hours are < 0.9 × H_C, one top-up collect (same flags, its own run dir, seed base + 100 + k for process k: disjoint from every
  main-collect seed) gets
  ceil((H_C − H_D,main) × the main collect's realised games per active hour) games, and D's hours are the sum.

**Training.** Both from `kC_baked.pt` with the kC recipe, then recal fit and bake. D+O is the D rows plus the term,
after the memory gate passes.

**Matches.** Each arm plays 12,800 games against the parent `kC_baked.pt`, with the data-volume record's rules
unchanged:
- `arena_standard.py` matched sims, 64 v 64, plain seats, baked files, `rl-run1-pilot/openings/eval.pgn`;
- per-seat resume; a damage-only 800-game screen;
- `arena_seat_balanced_pool.py`'s normal interval over opening keys.

All arms share one seed, so they play the same book draw. No pairing is claimed.

**The deciding yardstick, exactly:**
```bash
# from the launch pin's worktree; DF = the new bank (rl-deep-fast/), created at launch
PYTHONPATH=. $PY $DF/prelaunch/verdicts3.py final $DF
```
- **`verdicts3.py` is a new three-arm script for this bank.** The launch addendum pins it by sha256. It is written
  from the data-volume bank's `verdicts.py` (`~/chess-artifacts/operations/rl-run1-volume/prelaunch/verdicts.py`
  [`f487d849`]) for its per-match pooling and bootstrap only. That file cannot be reused as is: it has one two-arm
  contrast with no "loses" branch, and it refuses to finish when the first arm improves without a base match, which
  this record does not register. The new script's rule:
  - **Per match** (C, D, D+O, each vs the parent `kC_baked.pt`, from its `pool.json`): improves iff LB95 > 0, damage
    iff UB95 < 0, adoptable iff LB95 > −10 Elo, else unread. An unavailable interval reads unread.
  - **Readout A:** Δ̂_DC = d̄_D − d̄_C on the score scale, SE = √(SE_D² + SE_C²). It reads
    **"deep/fast beats stock" iff Δ̂_DC − 1.645·SE > 0**, **"deep/fast loses" iff Δ̂_DC + 1.645·SE < 0**, both
    strict. Otherwise it is unread, and it is also unread if the cost check above fails. MDE80 ≈ 8.1 Elo
    (SE ≈ 3.24 Elo-equivalent; data-volume record, power).
  - **Readout B:** Δ̂_OD = d̄_{D+O} − d̄_D, the same SE rule. It reads **"the term helps" iff Δ̂_OD − 1.645·SE > 0**,
    **"the term hurts" iff Δ̂_OD + 1.645·SE < 0**, otherwise unread. If D+O did not play, it reads "unread" with
    its reason.
  - **No base match is required** or read. A missing arm makes only its own readings unread; it never makes the
    script fail.
  - **Bootstrap, a check only:** B = 20,000, the opening key as the unit, arms resampled independently. It never
    decides.
  - The launch addendum shows from fixture `pool.json` files that a correct run reaches every reading: each match
    reading, both branches and unread for A and for B, the cost-check unread, and a missing D+O.
- D+O plays only if D's own match does not read damage.
- No threshold changes in flight.

| Outcome | Licenses |
|---|---|
| A: D beats C | at the kC dose and this actor, the deep/fast collection beats stock 64 per GPU-hour. D3 (if built and Stage 1-clean) is registered next, against D |
| A: D loses | deep/fast at p = 0.25 / 80 / 16 is worse per GPU-hour than stock 64 here. Says nothing about value-only fast rows or other p |
| A unread | no claim. Each arm's own vs-parent reading is reported |
| B: the term helps | the ordering term over ≥ 2-visit children adds strength on deep/fast data |
| B: the term hurts | the term costs strength on deep/fast data at its sealed strength |
| B unread | no claim. **If the sibling screen has no Stage A reading by 2027-01-31, Readout B is recorded as "unread, insufficient evidence"** and this record closes on Readout A alone |
| either arm vs parent | reported as in the data-volume record. C is also the line's next stock generation, so its reading is useful whatever A says |

## Confounds (named, not removed)

1. **Rows and updates.**
   - At equal GPU-hours, D trains on about 0.4× C's rows, and under exactly-once on proportionally fewer updates.
   - Readout A compares two collection policies at equal cost, and does not separate row count from row quality. That
     is the question asked.
   - The data-volume result (more rows helped) is the prior against D.
2. **Game distribution.** D's fast plies have no root noise and its games are more decisive (0.606 against 0.510 at
   p = 0.25). Value targets come from a different outcome mix.
3. **Two changes in "deep".** Deep plies change both width (round 0, topk 40) and budget (80). The full-width Stage 1
   already showed that width alone at 64 sims is a null on the policy target.
4. **Tree reuse** carries fast-ply visits into deep roots: 8.5 carried visits against 12.1.
5. **Actor and parent** are `kC_baked.pt`, not the SL base. Every scoping number cited is the SL base's.
6. **The actor's own step is fragile.** kC's "improves" over its parent is +5.08 [+0.48, +9.67], a lower bound
   near 0 (`~/chess-artifacts/operations/rl-run1-volume/verdicts.json` [`fab91794`]).
7. **Stage 1 screens a different actor without reuse.** It scores the SL base's W80 searches on fresh audit roots, not
   kC's searches with tree reuse. It screens the deep target's quality; it does not measure D's stored rows.
8. **Fast plies choose the positions.** 75% of D's moves are chosen by a 16-sim search, so the positions the deep
   plies are stored at come from a different distribution than C's, not only a more decisive one.
9. **The pin** must contain #1511. If #1522 merges first, its `check_vram` applies, and it marks topk 40 / 80 sims as
   extrapolated.
10. **A D top-up draws its own openings.** If the dose rule needs a top-up collect, its run dir starts a fresh
    opening cursor, so some of its openings can repeat the main collect's. The count of top-up starts present in the
    main collect's `starts.txt` is reported beside Readout A.
11. **C and D draw different opening permutations.** Their base seeds differ (2026112701 and 2026112702), so the two
    arms train on different openings from the same book (added in the Stage 2 phase B addendum).

## GPU-hours

| Stage | Item | GPU-h |
|---|---|---:|
| 0 | four legs (C, D, D, C), 1,152 games each | ≈ 1.0 |
| 0 | D3 two legs (if built) | ≈ +0.5 |
| 1 | CPU scoring; D3 search (if built) | ≈ 0 / +0.1 |
| 2 | collections C and D, G_C each | ≈ 10.2 |
| 2 | training: C (≈ 5,500 updates), D, D+O, plus 3-update smoke | ≈ 1.2 |
| 2 | recal fits + bake, 3 arms | ≈ 1.2 |
| 2 | matches C, D, D+O vs parent, 12,800 each | ≈ 8.7 |
| | **total (no D3)** | **≈ 22** |

Rates: 0.3164 s/update and 1.45 GPU-h per 6,400 games (data-volume record); 152 rows/s stock (full-width Stage 0).

## Recovery

- Nothing live changes: no yaml, no live tree, no deployed code.
- Every output goes to a new bank (`rl-deep-fast/`).
- `kC_baked.pt` stays the line's head unless an arm improves by its own registered reading, and adoption is a later
  record.
- A failed stage is moved aside by the bank's `clear_step` and rerun from its inputs. Matches resume per seat.

## What the code cannot do today

- **D3's every-child-≥ 3 visits:** needs `root_full_width_visits` (above). Everything D needs exists.
- **Per-ply topk:** not needed. Width is capped by the budget on fast plies.
- **A per-row "fast" flag:** not needed. Every stored row carries `search_budget`, the budget its ply was given
  (`selfplay/network_turn.py`, `replay/shard.py`). That is the per-row take-effect observation for `--full-prob` /
  `--fast-sims`; the evidence log's caveat 11 ("manifest only") is out of date on this point.
- **The ordering term on D:** exists (`--w-sibling`, rev 7). Its strength and chunk are sealed by the sibling
  screen's addendum, which has not launched. Readout B waits for it, until the date above.

## Launch addendum (2026-11-26)

Bank: `~/chess-artifacts/operations/rl-deep-fast/` (`DF`). Every file below is listed in `~/chess-artifacts/operations/rl-deep-fast/SHA256SUMS`
[`7945de9c`], which `stage0/run_stage0_all.sh` verifies before any leg. Launch pin: `b0541c15a` (main with #1511
and #1522; #1531 is docs only), in a detached worktree `rl-deep-fast-b0541c15a`.

**1. Stage 1 scorer.** `~/chess-artifacts/operations/rl-deep-fast/stage1/stage1_score.py` [`708bf777`] is the full-width scorer [`b1e62b3a`, kept beside it
as `stage1_score.v0_rfw.py`] with only the arm list and the R64n reference changed (47 changed lines: the dry-run and
bank branches read the arms from the adapted bank; the paired loop writes `<arm>_minus_R64n` for every arm except
the reference; the full-width reproduction block is dropped). W80 enters through `~/chess-artifacts/operations/rl-deep-fast/stage1/adapt_scoping_arm.py`
[`9708383b`], which scatters `RC/visits-per-child-scoping/out/arm_W80.npz` [`5e32883d`] into the bank format with no move mapping
(its action ids are already `lc0_1858`, from the scoping bank's `alloc_bank.py`) (`~/chess-artifacts/operations/rl-deep-fast/stage1/stage1_bank_W80.npz` [`96ebcf0e`]; duplicate (row, move) entries and target rows not summing to 1 are
refused).
- **Fixture:** `--dry-run-banked` (W80's slot filled with the banked R64n seed-2 search, so it validates the
  scorer and is not a Stage 1 result) writes `paired.W80_minus_R64n` with `exp_cp_lb` −0.046 [−1.203, +1.096] and
  `mass_unscored` −0.0005 [−0.0015, +0.0005], den 2,433 (`~/chess-artifacts/operations/rl-deep-fast/stage1/fixture_dry_run/scores.json` [`6c124fd2`]).
  Seed noise reads inside the +1.2 cp kill, as it should.
- The deciding read waits for this addendum's merge.

**2. Coverage pre-check (banked; not the deciding read).** `~/chess-artifacts/operations/rl-deep-fast/stage1/coverage_precheck.py` [`2452e4c3`] on the
2,433 rows: W80's mass on moves deep SF did not score is 0.0220 against R64n's 0.0183, **paired +0.0036
[+0.0019, +0.0055]**, under the +0.01 gate (`~/chess-artifacts/operations/rl-deep-fast/stage1/coverage_precheck_W80.json` [`a81b50b1`]). **No SF top-up is needed.**
Had it tripped, the top-up would have been 3,278 moves on 798 positions (moves with ≥ 1e-3 W80 mass and no SF
score) at 8M nodes, cold TT, `searchmoves`, production Syzygy, on CPU.

**3. Cost check.** The clock is each collect's `active_seconds` (not `wall_seconds`, never including
`paused_seconds`); `~/chess-artifacts/operations/rl-deep-fast/prelaunch/cost_from_stats.py` [`ba02611e`] writes the bank's `cost.json` (written at Stage 2) from the sealed collects' own
`stats.json`, with `clock_complete` false if any process resumed at game level. The dose rule is the registered
top-up (Stage 2, above): D's main collect aims at 0.85 × H_C from D's Stage 0 rate, and a top-up sized from the
main collect's own realised rate brings it to H_C. Why this and not the alternatives:
- **Stopping D by hours is not available.** `<run>/STOP` drains to an unsealed collect, and an unsealed collect
  cannot be trained on; there is no time-stop that seals.
- **A dose fixed from the Stage 0 rate** voids Readout A on any rate miss above 10%.
- **Splitting D into many chunks** gives each run dir a fresh opening cursor; one top-up of ≈ 15% bounds that to the
  top-up's games (Confound 10).
- Readout A is then voided only if D's realised rate is below 0.773 × its Stage 0 rate (0.85 ÷ 1.1) at the same
  shape, or a clock is incomplete. The top-up's two run dirs must both reach D's training pool; the Stage 2
  launcher's addendum shows this.
- **Top-up seeds are disjoint from every main-collect seed** (base + 100 + k). With single-process collects the game
  RNG and the opening permutation follow the seed alone, so "seed + 1" for process k would replay process k + 1's
  games (non-author review of #1535). Confound 10's repeat count is kept.
- **Also due in the Stage 2 launcher's addendum:** a scan for other GPU processes during each collect (or its SMI
  samples), banked beside `cost.json`, since the guard checks only at start; and `verdicts3.py`'s bootstrap check
  made unable to abort before `verdicts.json` is written (its d̄ reproduction is an exact float comparison that
  exits).

**4. Stored-share denominator.** games × `mean_plies_after_book`, both from each process's `stats.json` (Stage 0,
above), applied per process by `~/chess-artifacts/operations/rl-deep-fast/stage0/row_gates.py` [`ba165cfe`].

**5. `~/chess-artifacts/operations/rl-deep-fast/prelaunch/verdicts3.py`** [`2ca64d73`], as registered in Stage 2, reading the bank's `cost.json` (written at Stage 2) for the cost check (`D/C` of
`active_seconds` in [0.9, 1.1], both `clock_complete`). `~/chess-artifacts/operations/rl-deep-fast/test/verdicts3_fixtures.py` builds synthetic `pool.json`
banks and reaches every reading: per match improves / damage / adoptable / unread; Readout A beats / loses / unread,
plus unread on unequal cost (D/C 1.2), on an incomplete clock and on a missing `cost.json`, and a read at D/C exactly
0.9; Readout B helps / hurts / unread; D+O missing (B unread with its reason, A still reads, D+O's match "not
played"); C missing (A unread, B still reads); D stopped by its screen (its match "damage (screen)", A and B
unread). **23 of 23 pass.** `cost_from_stats.py` run on the full-width bank's real Stage 0 `stats.json` writes a
cost file `verdicts3.py` reads.

**6. Stage 0 launcher.** `~/chess-artifacts/operations/rl-deep-fast/stage0/run_stage0.sh` [`30b545b7`] is the full-width v10 launcher with the actor
`kC_baked.pt` (sha256 `c48a7b2c1cc734a75b9673e2bfbe20233bc5cfb9fe7c622a1290259a5b2b6cc5`, refused otherwise, exit 92), the
arm table (C: 64 sims, topk 16, `--full-prob 1.0`; D: `--full-prob 0.25 --fast-sims 16 --sims 80 --topk 40
--root-full-width`), the pin above, and the D-leg gates read from each collect's shards and `stats.json` before the
shard stores are dropped. `~/chess-artifacts/operations/rl-deep-fast/stage0/run_stage0_all.sh` [`8d076663`] runs r01_C, r02_D, r03_D, r04_C (resume and clear
rules as the full-width bank), then `~/chess-artifacts/operations/rl-deep-fast/stage0/stage0_report.py` [`97defcd3`]. The guard `~/chess-artifacts/operations/rl-deep-fast/gpu_guard.sh` [`e2c4a5aa`] is the
data-volume guard with the stage name `stage0` and the full-width scan v7 (`~/chess-artifacts/operations/rl-deep-fast/gpu_scan.sh` [`d1aee351`], zombie skip).
- **`check_vram` at topk 40, settled on CPU:** the check is per process. Stage 0 runs three `--collect-procs 1`
  collects, and each refuses on its own need: `vram_settings_detail` at the pin gives 3.22 GiB per process. The
  9.65 GiB total for 3 processes is never evaluated by a collect, and three concurrent starts each read free memory
  before the others allocate. The figures are marked extrapolated (80 sims, a 10,240-row leaf buffer, topk 40), against ≈ 26.4 GiB usable
  (`~/chess-artifacts/operations/rl-deep-fast/prelaunch/check_vram_cpu.json` [`d45da28a`]). The topk-32 fallback is not needed; Stage 0's peak MiB is the
  measurement.
- **Gate proof:** `~/chess-artifacts/operations/rl-deep-fast/test/row_gates_fixtures.py` writes 300 real stored rows with the repo's zarr writer: a correct
  D and a correct C collect pass; zero mean prelude, leaked fast rows (budget 16), a root-stat row without the
  prelude field, a share of 0.30, and C rows gated as D are each refused; a share of 0.23 passes. **8 of 8 pass.**
- **Guard tests** (`~/chess-artifacts/operations/rl-deep-fast/test/guard_tests.sh`, CUDA_VISIBLE_DEVICES empty, fake HOME; none reaches nvidia-smi): stage
  unset or wrong, no release file, a guard refusal leaves no run dir, a tampered bank file (SHA256SUMS, 93), a bad
  `DF_REGUARD_WAIT_S`, and `clear_step.sh`'s five cases (bad tag, nothing to clear, a finished leg refused, an
  unfinished leg moved aside, refused while the bank lock is held). **13 of 13 pass.**
- **The one unmutated harness pass** (`~/chess-artifacts/operations/rl-deep-fast/test/harness_correct_run.sh`: fake HOME, fake nvidia-smi, `fake_collect.py`
  in place of the network; the real scan, launchers, gates, `steady_rate` and report) runs once the GPU is free of
  the data-volume control; its result is reported on the PR.

### Stage 1 readout (2026-11-26): W80 PROCEED

**Disclosure:** this read ran after a non-author GO on this addendum (PR #1535 at `5463f6f7e`, bank
`~/chess-artifacts/operations/rl-deep-fast/SHA256SUMS` [`7945de9c`]) but before the addendum merged. It ran on CPU only (numpy; torch imports blocked; nice 15, ≤ 4
threads) with the reviewed scorer `~/chess-artifacts/operations/rl-deep-fast/stage1/stage1_score.py` [`708bf777`]
on `~/chess-artifacts/operations/rl-deep-fast/stage1/stage1_bank_W80.npz` [`96ebcf0e`], unchanged.
Outputs: `~/chess-artifacts/operations/rl-deep-fast/stage1/scores/scores.json` [`6122f85a`], `.../per_row.npz`
[`c7ba0366`], the mechanical verdict `.../verdict.json` [`1bb4d461`] from
`~/chess-artifacts/operations/rl-deep-fast/stage1/stage1_decide.py` [`f0b300f2`], and the report-only intervals
`~/chess-artifacts/operations/rl-deep-fast/stage1/scores/fragility.out` [`13b9348f`].

| Step | W80 − R64n, paired over 2,433 rows (190 games) | Rule | Result |
|---|---|---|---|
| 1. Coverage | mass on moves deep SF did not score: +0.0036 [+0.0019, +0.0055] | ≤ +0.01 | passes |
| 2. Kill | `exp_cp_lb`: +0.315 cp [−1.502, +2.246] | STOP iff LB95 > 0 and est > +1.2 | not fired |
| Verdict | | | **PROCEED** to Stage 2 |

- **Report-only tail:** 28 rows with |d| > 100 cp carry +780.9 of the +765.5 cp-row total; the median per-row
  difference is −0.0009 cp. Leaving out the single largest game gives −0.138 [−1.853, +1.673]; dropping the two
  largest gives −0.578 [−2.114, +0.881]. The estimate is 0.885 cp below the threshold.
- **Report-only secondaries (paired):** the Ceres secondaries use 2,422 rows (11 rows have no Ceres value), not
  the table's 2,433. Ceres expected loss −0.0013 [−0.0028, +0.0001]; Ceres top-1 +0.0058
  [−0.0052, +0.0166]; p(best) −0.0036 [−0.0084, +0.0014]; top-1 best +0.0016 [−0.0082, +0.0113].
- It licenses only this: the registered kill did not fire (estimate +0.315 cp, below +1.2; LB95 −1.50 ≤ 0); the
  interval [−1.50, +2.25] spans zero and does not exclude a deficit larger than seed noise; tail-driven; not evidence
  W80's target is as good or better; says nothing about allocation or strength.

## Stage 2 phase A addendum: collection only (2026-11-26)

Stage 2 is split into two launchers, so the GPU does not wait on the training steps being proven. **Phase A**
collects and banks; **phase B** (pool, train, fits, bounds, bake, matches, `verdicts3.py` v2) is a separate addendum,
built while phase A collects. Phase A writes only run dirs, their sealed manifests, per-collect GPU-holder scans and
the cost files; nothing in it fixes a choice phase B makes.

Bank: `~/chess-artifacts/operations/rl-deep-fast/stage2/`, listed in its own
`~/chess-artifacts/operations/rl-deep-fast/stage2/SHA256SUMS` [`5bdcd64a`] (the bank's root
`~/chess-artifacts/operations/rl-deep-fast/SHA256SUMS` [`7945de9c`] is unchanged and is verified too). Launch pin `d08b03adb` (the Stage 0 pin
`b0541c15a` plus the collector memory fix; re-pinned after the first `collect_C` was OOM-killed, see the collector
OOM addendum below), worktree `rl-deep-fast-d08b03adb`.

**Launch condition.** Stage 0 finished (all four legs in `stage0/summary.jsonl`, `stage0/report.json` written) and
read within the record's cost assumptions, and the coordinator wrote `stage2/PHASE_A_GO`. If Stage 0's report puts D
outside those assumptions, phase A does not launch and the dose rule is not adjusted; the owner decides.

**Steps** (`~/chess-artifacts/operations/rl-deep-fast/stage2/launch_phase_a.sh` [`2df225b4`]), each its own process group:
1. `collect_C`: stock 64 (`--sims 64 --topk 16 --full-prob 1.0`), seed 2026112701, games = round(2.8M ÷ C's Stage 0
   rows per game).
2. `collect_D`: `--sims 80 --topk 40 --full-prob 0.25 --fast-sims 16 --root-full-width`, seed 2026112702, games =
   round(0.85 × H_C × D's Stage 0 steady games per hour), where H_C is `collect_C`'s realised `active_seconds`.
3. `collect_Dtopup`: if H_D,main < 0.9 × H_C, ceil((H_C − H_D,main) × `collect_D`'s realised games per active hour)
   games with `collect_D`'s flags, in its own run dir, **seed 2026112802 (2026112702 + 100), disjoint from every
   main-collect seed**; otherwise it records "not needed" and plays nothing.
4. `cost`: `prelaunch/cost_from_stats.py` writes `stage2/cost.json` (`active_seconds`, D summed over its collects,
   `clock_complete`), and `stage2/cost_scan.json` banks every collect's GPU-holder check beside it.

All collects: `--collect-procs 3 --concurrency 128 --eval-chunk-rows 512 --c-scale 0.1 --c-visit 50 --max-plies 450`,
the production Syzygy pair, actor `kC_baked.pt` (`c48a7b2c…`, refused otherwise). The doses come from
`~/chess-artifacts/operations/rl-deep-fast/stage2/dose.py` [`ccd39e16`], which reads Stage 0's numbers from
`stage0/report.json` and the sealed collects' `stats.json`; no Stage 0 number is a constant in the launcher. A collect
with an incomplete clock (a game-level resume or a kept process) makes `dose.py` refuse.

**Completion check per collect:** the sealed `stats.json` has games equal to the dose and a complete clock, and
`stage0/row_gates.py` passes on the collect (the arm's budget on every stored row, no fast row stored, and for D the
prelude and the stored share 0.25 ± 0.02).

**GPU-holder scan.** `~/chess-artifacts/operations/rl-deep-fast/stage2/gpu_holders.py` [`9991af8d`] samples `/proc` every 15 s for the whole collect:
processes of this user holding the GPU device (matched by rdev, as the guard's scan does), and which of them are
outside the collect's own session. A collect is **clean** iff no two consecutive samples show a foreign holder (a
single ~0.1 s `nvidia-smi` probe can land in one sample; two in a row is a co-resident job). **Registered now for
phase B:** Readout A also reads "unread: cost unmeasured (a co-resident GPU job)" if any of the arms' collects is not
clean. The scan never fails a collect; it is banked and read at the verdict.

**Resume.** A step is done iff its exit is 0, its check passes and its recorded inputs (launcher, `dose.py`, row gates,
`stage0/report.json`, actor, pin, its dose file) equal the inputs now; a relaunch skips it. A done step with other
inputs refuses. An interrupted or failed collect is never resumed in place, because a game-level resume leaves part
of its clock behind and the cost check would read unread: it refuses until
`~/chess-artifacts/operations/rl-deep-fast/stage2/clear_phase_a.sh` [`cfce5e18`] moves its run dir and markers aside (never deletes them); the rerun
plays the same dose and seed.

**Deviations, stated.**
- **Collect shape:** each Stage 2 collect is one run dir with K = 3 processes (as the generation-2 and data-volume
  collects), not Stage 0's three K = 1 collects, because phase B trains from one run dir per collect. Stage 0's rate
  therefore only *estimates* the K = 3 rate. It sizes D's main collect, and any error is absorbed by the registered
  top-up, sized from the main collect's own realised rate. The equal-cost check itself uses only the real collects'
  `active_seconds`, so H_C is measured, not projected. The shape is the same for both arms.
- **Training pin (phase B):** both arms train at `b0541c15a` (re-pinned to `d08b03adb`, see the collector OOM addendum), one pin for collect and train, so the C-vs-D comparison
  uses one trainer. `kC_baked.pt` was trained at `0c795fb7a`; absolute numbers here are therefore not comparable with
  the data-volume run's.
- **The top-up's openings:** a separate run dir starts a fresh opening permutation, so some top-up openings repeat
  the main collect's (Confound 10). The phase B pool tool allows and counts these repeats (reported in the pool
  manifest) and still refuses a repeated game id.

**Fixtures (CPU).** `~/chess-artifacts/operations/rl-deep-fast/stage2/test/dose_fixtures.py` [`ec8f2db2`]: the three dose rules, the inclusive 0.9 band, refusal of
an incomplete clock (a resume, a kept process), no overwrite, and the scan's clean / probe / co-resident / empty
cases. **12 of 12 pass.**
**Harness (one unmutated pass, run when the GPU is free):** `~/chess-artifacts/operations/rl-deep-fast/stage2/test/harness_phase_a.sh` [`2285a5a4`], with a fake
HOME, a fake `nvidia-smi`, and `test/fake_collect_a.py` in place of the collect. It proves a correct run (C, D main,
top-up, cost; doses from the report and the realised clocks; seeds and flags reaching the collect; D/C in
[0.9, 1.1]; scans banked), a relaunch that skips every done step, and the refusals: no GO (90), `CUDA_VISIBLE_DEVICES`
empty (94), Stage 0 unfinished (91), a tampered stage-2 file (93), an interrupted step (87), and the clear (finished
step refused, unfinished moved aside, rerun). Its result is posted on the PR: **11 of 11 pass** (#1539).

## Stage 2 phase B addendum: pool, train, fits, bake, matches, verdicts (2026-11-26)

Phase B reads only what phase A banked (its run dirs and sealed manifests, `stage2/cost.json`,
`stage2/cost_scan.json`, the dose files) and runs arms C and D from `kC_baked.pt` to the registered readings. Bank:
`~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/`, listed in
`~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/SHA256SUMS` [`f4f722c9`], whose sha256 the launcher
`~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/launch_phase_b.sh` [`ca2db464`] pins.
**Launcher change after the harness (2026-11-26).** The 13-of-13 harness pass below ran on `launch_phase_b.sh`
`72647431`; after it, two lines were added and nothing else: a usage comment, and, right after the launch line in
`resume.log` (past every refusal), `echo "$$" > $PB/phase_b.pid`, so the launcher records its own pid. Why: the
launch form `setsid nohup bash … & echo $! > pid` records the wrapper's pid, which exits at once (the coordinator had
to correct phase A's pid file by hand). That 13-of-13 pass predates this change. The phase B harness was rerun once on
the final launcher (re-pinned, `ca2db464`) and passed 13 of 13; see the collector OOM addendum. The launcher is not listed in `SHA256SUMS` (it pins that file's sha256), so the operator checks
it directly before launch: `sha256sum ~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/launch_phase_b.sh`
must print `ca2db464588f393a8bfcabed19cdee27b1630f3465ca9586c28be7d6fef99dd8` (the re-pinned launcher, collector OOM
addendum); any other value means do not launch.
Launch: `PB=~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb; setsid nohup bash $PB/launch_phase_b.sh >
$PB/phase_b.log 2>&1 < /dev/null &` (no `echo $!`); the pid is in `$PB/phase_b.pid` once the refusals pass. Stop:
`kill -TERM $(cat $PB/phase_b.pid)`, then check `ps` for leftover Stockfish groups. **Phase A**: this record never
carried phase A's launch line; the instructions sent with it used the `& echo $!` form, and the live run's
`stage2/phase_a.pid` was corrected by the coordinator to 3116424. The re-pinned `launch_phase_a.sh` (collector OOM
addendum) now writes its own pid to `stage2/phase_a.pid` past its refusals, as phase B does. Pin `d08b03adb` (re-pinned
from `b0541c15a`) for
every step (collect, pool, train, fits, bake, play). Launch needs phase A finished (every phase-A step exited 0,
`cost.json` and `cost_scan.json` present), `stage2/PHASE_B_GO`, and a `stage2b` release line.

**It is the data-volume control's launcher, ported.** `launch_volume.sh` (control bank) with its step machinery,
resume rule, markers, host-RAM floors and watchdog, per-seat match resume and damage-only screen kept; the changes:

| Item | Control | Phase B |
|---|---|---|
| Pin | `0c795fb7a` (pilot) | `d08b03adb` (re-pinned from `b0541c15a`) for everything; matches through `run_df_match.sh` (the follow-up launcher with only the commit changed) |
| Arms | one cell (c2x) | C and D, each from `kC_baked.pt` |
| Pool | `pool_collect.py`, admission record | `pool_df.py`: sources = phase A's run dirs, admitted iff their step exited 0; D = `run_D` + `run_Dtopup` iff its dose > 0 (**both D run dirs reach the pool, or the tool refuses**); openings repeated across sources are kept and counted in the pool manifest (Confound 10), a repeated game id still refuses |
| Flags | `volume_flags.sh` | `TRAIN` and `ZQ` byte-identical; each arm's collect flags = its phase-A flags with the pool's seed (2026112703, also the training seed) and games, so the trainer's own precheck accepts the pool |
| Snapshots | 1478, 2956 | (U + 1) // 2 and U, U = the PIN's `planned_phase_updates` on that arm's pool (R2 compares the final with the half-way net, as the control's 1478-of-2956) |
| Take-effect | `opt_check.py` vs generation 2's arm, gen1's saved lr | `opt_check.py` unchanged, each arm's train dir the other's non-LR reference (both arms must realise the same optimizer), `kC_baked.pt`'s saved group lr (6.0e-05, 3e-06; read with `torch.load`) |
| Fits | generation 2's collect, reference gen1 | **both arms on C's pool collect** (one common calibration set; hold-out 0.05), reference `kC_baked.pt`; `fit_value_recal.py` is the same file at both pins (sha256 `37e27420…`) |
| Bounds, bake | `recal_bounds.py`, `bake_recal.py` | unchanged, per arm |
| Matches | one match | C vs `kC_baked.pt` and D vs `kC_baked.pt`, 12,800 games each (6,400 per seat), seed 2026112711, a 4-game smoke per seat; the damage-only screen (200 pairs = 800 games) inside seat B **for each arm**, per match: a screen-stopped match is complete as "damage (screen)" (`stage2/arenas/<L>/STOP_SCREEN`) and the next match still plays |
| Verdicts | `verdicts.py` | `verdicts3_v2.py final stage2/` |
| Host-RAM floor | 58 GiB | 86 GiB before the smoke and `train_C` (the volume bank's floor for its 2.77M-row pool; C is that size); 58 GiB before `train_D` |

Steps: `pool_C pool_D smoke train_C train_D optcheck fits bounds bake_C bake_D m_C m_D`, then the verdicts. A failed or
interrupted non-match, non-train step moves aside with `~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/move_aside_b.sh` (the control's mover, ported; it renames, never deletes).

**`verdicts3_v2.py`** (listed in phase B's `SHA256SUMS`):
`~/chess-artifacts/operations/rl-deep-fast/prelaunch/verdicts3.py` [`2ca64d73`] with only (1) the bootstrap
check made unable to abort before `verdicts.json` is written (a tolerance on the d(o) reproduction; a mismatch, a
missing or broken log, or any error inside the check is recorded, never raised), (2) the cost check also reading
`cost_scan.json` (Readout A "unread: cost unmeasured (a co-resident GPU job)" unless every collect is clean, as
phase A registered), (3) its bank argument being the Stage 2 directory. D+O is not played by this launcher; Readout B
reads unread with its reason until D+O's own addendum.

**Deviation, stated:** the training pin is `b0541c15a` (now `d08b03adb`), not kC's `0c795fb7a` (phase A addendum). The fits use C's pool
for both arms, a choice the record did not make; it keeps the calibration data identical between the arms.

**Proofs (CPU).**
- `~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/test/pool_df_fixtures.py` [`e7a41e00`]: a C pool; a D pool from both run dirs with 3
  repeated openings kept and counted; a D pool from `run_D` alone when the top-up's dose is 0; refusals for a top-up
  step that did not exit 0, a missing top-up run dir, a game id in both sources, an opening repeated within a source,
  a settings difference, another actor, and a tampered shard. **11 of 11 pass.** `stage_checks_df.py`'s pool gate
  passes on each of the three pools written.
- `~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/test/verdicts3_v2_fixtures.py` [`d75c6520`]: v1's 23 readings plus a co-resident GPU job, a missing
  `cost_scan.json`, and a broken seat log that neither aborts nor changes Readout A and is recorded in the check.
  **27 of 27 pass.**
- **The one unmutated harness pass** (`~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/test/harness_phase_b.sh` [`503e7e64`], run when the GPU is free): a
  fake HOME with a finished fake phase A (sealed run dirs at the pin, the real `kC_baked.pt`), a fake `nvidia-smi`,
  and `test/fake_phase_b.py` in place of the GPU and heavy steps only; the pool tool, the completion checks, the
  seat checker, `opt_check.py`, `recal_bounds.py`, the guard and `verdicts3_v2.py` are real. It proves a correct run
  through all twelve steps to `verdicts.json`, a relaunch that skips every step, a run whose D screen reads damage
  (D "damage (screen)", C still read, Readout A unread), and the refusals: no GO (90), `CUDA_VISIBLE_DEVICES` empty
  (94), phase A unfinished (91), a tampered phase B file (93), an interrupted step (87), and the mover's refusals. Its
  result is posted on the PR: **13 of 13 pass** (#1540). The first attempt broke on an unquoted `PATH`
  holding a space; only the harness script was fixed (it is not in `SHA256SUMS`; its sha256 moved from `d76ea3ef`).

**Registered decisions** (the coordinator accepted each; one line of reasoning apiece):
- **Snapshots at (U + 1) // 2 and U:** R2 then compares the final net with the half-way net, which is the control's
  1478-of-2956 rule, sized from each arm's own plan rather than from a constant.
- **Both arms' fits on C's pool collect:** the recalibration is then fitted on identical data for C and D, so a
  difference between the arms' baked nets is not a difference in their calibration sets.
- **`opt_check.py` cross-reference (each arm's train dir the other's non-LR reference):** the comparison needs the two
  arms to realise the same optimizer, and no earlier train dir shares this pin and this init file.
- **A damage screen per arm, per match:** the data-volume record screens every match vs the parent; a damage stop on
  one arm must not stop the other arm's match, so the stop file is per match.
- **Host-RAM floors (86 GiB before the smoke and `train_C`, 58 GiB before `train_D`):** the measured basis is kC's own
  training on its 2.77M-row pool, 60.46 GiB peak RSS (72.14 GiB of loaded arrays), and C's pool is that size; D's
  pool (~1.2M rows) is under the control's 1.59M-row pool, whose floor was 58 GiB. A refusal at the floor (exit 89,
  checked before the step's `.started`, so no step is marked failed) is a **wait-and-retry**: quiet the host, move
  `HOLD_HOST_RAM` aside, relaunch.

**Confound added (from #1539's review):** Confound 11 in the list above. C and D draw different opening permutations
(base seeds 2026112701 and 2026112702), so the arms train on different openings from the same book. This adds sampling
variance to Readout A; it is not a systematic advantage to either arm.

**Notes from #1539's review (non-blocking), recorded:**
- `stage0/stage0_report.py`'s projection still prints the superseded "D aimed at 0.95 x C's hours" wording and a
  `D_games_at_0_95_C_hours` field. That block is report-only (`decides: false`), and the bank file stays unchanged
  because Stage 0 runs with it pinned. Phase A sizes D with `dose.py` at 0.85 × H_C plus the registered top-up.
- **Stopping either phase:** after any stop, check `ps` for leftover Stockfish process groups from the collect or match
  (the TERM goes to the step's process group; a grandchild that called `setsid` itself would leave it; this is the
  control's known gap) and stop only those whose command line names this bank's paths.

## Collector OOM and re-pin addendum (2026-11-26)

**What happened.** Phase A's first `collect_C` (pin `b0541c15a`, launched 15:08:54 EST) was killed by the kernel's
global OOM killer: `stage2/collect_C.exit` reads `1 2026-11-26T17:16:45-05:00`, after 10,860 of its 27,955 games.
dmesg showed three collect processes at about 32 GB anonymous memory each. The cause: in finite mode, `play_batch`
(`selfplay/manager.py`) appended every `ReplaySample` of the call to `all_samples` and kept them until it returned
(`selfplay/finalize.py` clears that list only in continuous mode). `simple_rl/collect.py` throws the returned list away
(`_, batch_stats = play_batch(...)`) and takes its rows from `on_game_complete`, so every row was held twice over for
nothing. That is about 8.9 MB per game (100 rows × 175×8×8 × 8 B). C needs 9,318 games per process, about 83 GB per
process. D (24 rows per game, about 13k games per process) would need about 28 GB per process, so it would also have
OOMed at three processes.

**The fix** (pin commit `d08b03adbd34ffc42b679ec5f3fa7e878da3eab6`, branch `exp/rl-deep-fast-pin`, which is
`b0541c15a` plus that one commit; the same change is in this PR for main). `play_batch` takes a new
`retain_samples: bool = True`, and `collect.py` passes `False`. With `False`, each game's rows are dropped from
`all_samples` once its `on_game_complete` and `on_game_provenance` callbacks have run, and are counted so that
`BatchStats.positions` stays correct. The returned list is empty. `retain_samples=False` without `on_game_complete` is
refused. The default is unchanged for every other caller, and game generation is unchanged.
`tests/test_play_batch_retain_samples.py` checks that the returned list is empty, that every row reached the callback,
and that `positions` equals the rows delivered. It also checks that the rows are byte-identical to the default path,
in the same order and with the same seed (`x` and `policy_target` per row; the same seed reproduces the same rows and a
different seed changes them). Mutant: with the clear removed, both row tests fail; restored, all three pass. The
collector's other per-game structures are bounded: `pending` is popped per game in `on_game`, `buffered` is flushed at
`rows_per_shard`, the journal is compacted at every flush, and the provenance objects are consumed per game and not
kept.

**One pin for collection and phase B training: `d08b03adb`**, worktree
`~/projects/chess-worktrees/rl-deep-fast-d08b03adb`. Both arms collect and train at it. The bank files that named
`b0541c15a` are re-stamped with only the pin changed: `launch_phase_a.sh`, `phaseb/launch_phase_b.sh`,
`phaseb/run_df_match.sh`, `phaseb/move_aside_b.sh`, `phaseb/pool_df.py`, and the harnesses' fixtures.
- `~/chess-artifacts/operations/rl-deep-fast/stage2/launch_phase_a.sh` [`2df225b4`] also writes its own pid to
  `stage2/phase_a.pid` past its refusals. Every `DF_RSS_INTERVAL_S` seconds (default 600) of a collect, it logs one
  `rss <step> session_total=<MiB> <pid>:<comm>:<MiB> …` line to `phase_a.log`. The line covers the collect's process
  session (processes at or above 256 MiB are listed) and is the observation that proves the fix took effect.
- Hashes: `~/chess-artifacts/operations/rl-deep-fast/stage2/SHA256SUMS` [`5bdcd64a`],
  `~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/SHA256SUMS` [`f4f722c9`], and
  `~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/launch_phase_b.sh` [`ca2db464`], which pins
  `phaseb/SHA256SUMS`.

**`collect_C` restarts whole, under the new pin, with the same seed (2026112701) and the same dose (27,955 games, from
the kept `collect_C.dose.json`).** The collect's `plan.json` code check refuses to resume a run dir written at another
commit, so a resume is not possible. `clear_phase_a.sh collect_C` moves the partial `run_C` and its markers into
`stage2/aborted_*`. Nothing is deleted. That partial run is a duplicate prefix of the rerun (same seed, same doled
openings, same code path for game generation) and **must never be admitted alongside it**. `pool_df.py` reads only
`run_C`, `run_D` and `run_Dtopup`, never `aborted_*`.

**Harness on the final launchers** (fake HOME, fake `nvidia-smi`, the guard's `CUDA_VISIBLE_DEVICES` refusal tested as
before; GPU idle; no mutant series): phase A **11 of 11 pass**
(`~/chess-artifacts/operations/rl-deep-fast/stage2/test/harness_phase_a.repin.out` [`e8767a56`]) and phase B **13 of 13 pass**
(`~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/test/harness_phase_b.repin.out` [`c49930ef`]), on `launch_phase_a.sh` `2df225b4` and
`launch_phase_b.sh` `ca2db464` at pin `d08b03adb`. Two earlier attempts were refused by the guard (exit 8, in every
step that guards) because a CPU-only pytest held `/dev/dxg` beside them: first my own test run, then another
session's. That is the guard behaviour described in the confound below.

**Confound (first attempt only).** A CPU-only test audit (pytest, 2 threads, `CUDA_VISIBLE_DEVICES` empty) ran beside
the first `collect_C` from 15:57 to 17:16. `gpu_holders.py` listed its pids as foreign holders in 302 of 509 samples
(at most 121 consecutive), so that attempt's scan is not clean. It is aborted and never pooled, so no reading uses it.
**Would the phase A/B guards hold on such pids? Yes.** The pinned `gpu_scan.sh` exempts a process whose
`CUDA_VISIBLE_DEVICES` is empty only from its argv-name clause. Its fd clause refuses any process of this user that
holds the `/dev/dxg` device, whatever its environment, and torch's CUDA probe opens `/dev/dxg` even with CUDA hidden.
A step's first guard therefore refuses at once (exit 8). The between-step re-guard waits up to `DF_REGUARD_WAIT_S`
(900 s) and then refuses. The harness rerun above hit exactly this. A long CPU test suite beside the collect therefore
halts phase A at its next step boundary: it does not corrupt the run, it stops it. Run such audits in a pause window or
between phases.

**Launch** (the coordinator, after non-author review and merge):
`S2=~/chess-artifacts/operations/rl-deep-fast/stage2; bash $S2/clear_phase_a.sh collect_C`, then
`setsid nohup bash $S2/launch_phase_a.sh >> $S2/phase_a.log 2>&1 < /dev/null &`
(no `echo $!`; the launcher writes `$S2/phase_a.pid`). Stop: `kill -TERM $(cat $S2/phase_a.pid)`, then check `ps` for
leftover Stockfish groups.
