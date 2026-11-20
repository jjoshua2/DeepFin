# Twofold repetition as a proven draw: semantics, footprint, defects, plan

**Status:** investigation and prospective plan only. **No code change is made by this
record. None will be made, apart from the engine-only fix in PR #1421 (option C, in
progress), until the owner agrees to an option below.**

- CPU only; no GPU and no training compute.
- Code read at `origin/main` a593e59ab. Every cited line was re-verified at that sha;
  `mcts/gumbel_c.py` and `selfplay/network_turn.py` citations moved with #1416.
- The banked games were produced at `source_revision` e633212d. It is not an ancestor of
  main. I checked it in a separate worktree: `_search_terminal.h` and `bt4_outcome.py`
  are byte-identical to main's, and `cboard_is_repetition`,
  `cboard_insufficient_material`, `tree_resolve_from_children`,
  `tree_mark_solved_and_propagate`, `gss_begin_round` and the `gumbel_c.py` reuse gate
  are textually identical. The gate sits at lines 1381-1383 there and 1484-1486 on main.
  The semantics below are therefore the ones that produced the banked rows.
- "SL base" means the actor whose sha256 is `a39f900c…` (model-state `9796079c…`, step
  680,276). The banked directory names call it A39.
- Scripts, raw outputs and per-row data are banked under
  `~/chess-artifacts/operations/rl-claude-20261119/twofold-audit/`, written `<bank>/`
  below. An independent reviewer's scripts are in `<bank>/review-census/`. They
  reproduce the census exactly; the reviewer checked the code claims by reading.

## Summary

- **The rule.** The compiled search scores a position as an **exact, proven draw**
  (`Q = 0`, solved status `DRAW`) the first time it repeats. It counts occurrences
  **before the search root (game history) as well as inside the tree**. lc0 counts only
  occurrences at or below the root and never treats a twofold as a proof. Stockfish
  needs a threefold when the earlier occurrence is before the root.
- **The mark is a proof.** It ends the search early at a proven root, it makes the
  policy target one-hot, and it survives tree reuse and compaction.
- **Training targets** (measured on the banked `raw-targets.npz` of the production
  net, every cycle of both arms):
  - Rows whose root has a child repeating a game-history position are 3.90% (F12) and
    3.94% (Q50) of all training rows.
  - **24.4–25.9% of those rows store q exactly 0**, against 0.26–0.28% of other rows.
    That is roughly 100× enriched, and it amounts to about 1% of all training rows.
  - When the played move is the repeat, 79–80% of rows store q exactly 0.
  - On average, 24–25% of the stored policy target mass sits on the repeating move(s).
  - 10.7% (F12) and 18.8% (Q50) of the exact-zero rows come from games that ended
    decisively.
  - **On outcome evidence these exact zeros look accurate on average.** The mean
    outcome on exact-zero rep rows is +0.006 (F12) and +0.015 (Q50), with W and L
    symmetric. The non-zero rep rows are calibrated too (stored +0.25 against outcome
    +0.26).
  - That comparison is partly self-fulfilling: 86.9% / 86.8% of the exact-zero rows
    are rows where the repeat was played. Outcomes therefore neither confirm nor refute
    the zero target. Whether a better target exists is a question for the deep-SF
    audit in §4.
- **Self-play is protected from the stale-root defect** by the reuse gate in
  `gumbel_c.py`. **The UCI engine is not protected** on its PUCT pool and walker paths,
  where a reused root keeps `DRAW` with unproven children. The walker path reproduces
  this deterministically. PR #1421 fixes it on the engine side and is in progress.
- **The repeated position is never searched,** so an unsearched exact 0 beats searched
  values below 0. In a constructed defender position the search plays the repeat in 5 of
  8 seeds with a stored target of 0.999–1.000; the history-free control never plays it.
  In the 2 seeds where the control search actually proves the repeat loses to a mate in
  1, the same seed with history plays it.
- **Banked games** (4,608 games, 497,027 searched plies):
  - 1.11% of searched plies play into a twofold, and 31.2% of games contain one.
  - 18.2% of games end on a claimable threefold. By construction of the claim rule, no
    game reaches an actual third occurrence.
  - For 39.2% of twofold events, an irreversible move followed later. Those games were
    still 66% drawn.
  - 13.6% of twofold events are in games that ended decisively.
  - A twofold terminal is a claim about value, not a prediction about play. These game
    statistics bound how often "twofold = draw" is wrong only loosely.
- **Recommendation:**
  - Option C now (PR #1421, engine only, identity on self-play).
  - Option B1 (lc0's root boundary: only occurrences at or below the root count) as
    a registered experiment, then B2 (an in-tree twofold is not a proof).
  - Each is decided by a **target-side audit** on a frozen set of banked repeat-child
    rows; an arena serves only as a pre-committed non-inferiority guard.
  - The case for B rests on path-dependent proofs, the unsearched repeat, departure
    from lc0/Stockfish practice, and what the audit finds. If the audit finds no
    measurable target benefit, only C is warranted.

## 1. Semantics in code

### Where repetition is detected and against what

| Fact | Code |
|---|---|
| Search terminal test: true game over (mate, stalemate, **strict** threefold, 50-move, insufficient material), else **any prior occurrence** gives `Q = 0.0`, `SOLVED_DRAW` | `mcts/_search_terminal.h:59-72` (`cboard_search_terminal`) |
| "Any prior occurrence" scans the board's whole `hash_stack` | `encoding/_cboard_impl.h:1416-1422` (`cboard_is_repetition`) |
| `hash_stack` holds every position since the last irreversible move: it is cleared on a zeroing move and the pre-move key is pushed otherwise | `_cboard_impl.h:1201-1211` (inside `cboard_push`) |
| The root board imported from python-chess fills `hash_stack` from `board._stack`, which is **game history before the root**, back to the halfmove clock (plus one unreachable pre-zeroing key) | `encoding/_lc0_ext.c:448-477` |
| C game over needs a strict third occurrence | `_cboard_impl.h:1424-1445` |
| Banked collection: the game ends at the root, before inference, when `board.outcome(claim_draw=True)` fires, i.e. when the side to move **can claim** threefold or 50-move. This applies in both outcome modes (`claim_draw=True` is unconditional; the banked games used `rule50_match_v1`) | `selfplay/bt4_outcome.py:101-105` |
| Every search entry point uses the same test: Gumbel collect, forced-move collapse, PUCT batch descent, walker descent | `_mcts_tree.c:1943` (`gss_prepare_batch`), `:1159` (`try_forced_collapse`), `:3110` (`batch_descend_puct`), `:2720` (`walker_descend_puct`) |

So the first repeat is a terminal, and it counts **both** the in-tree path and the game
history before the root. The comment at `_search_terminal.h:52` says "inside the search
tree"; the code does not do that. `tests/test_mcts_root_history_lifecycle.py:99-109`
pins the current behaviour: a real twofold that depends on root history is solved, and
its history-free twin is not.

### What the DRAW mark does

- **The leaf is never expanded.** A terminal is backed up and never queued for an NN
  evaluation (`_mcts_tree.c:1943-1947`), so nothing below the repeated position is ever
  searched.
- **It propagates as a proof.** `tree_mark_solved_and_propagate`
  (`_mcts_tree.c:1180-1213`) writes the status and walks up. `tree_resolve_from_children`
  (`:1071-1088`) makes a parent `DRAW` once every child is solved and none is a win for
  the parent. A twofold `DRAW` can therefore turn a would-be `LOSS` into a proven
  `DRAW`. It can also hide a `WIN` for the side to move at the repeated node, which is
  never searched. A WIN or LOSS proof never rests on a draw, as `uci/search.py:3607-3625`
  already notes.
- **Selection uses it.** A `DRAW` child's Q is forced to exactly 0, in
  `tree_select_child` (`_mcts_tree.c:1009-1013`) and in `tree_gumbel_select_child`
  (`:3509-3511`). Public Q queries return the proof value (`tree_child_q`, `:351-357`).
- **Early stops use it.** At the start of each Gumbel halving round, a root that is
  already solved `WIN`/`DRAW` collapses to one proving child with zero remaining budget
  (`gss_begin_round`, `_mcts_tree.c:1599-1644`). The Python side then returns a one-hot
  policy and `value = node_q(root)` (`mcts/gumbel_c.py:2040-2070`).
- **It is monotone.** A node already solved is never revisited
  (`_mcts_tree.c:1188-1193`, `:1204`). On the root-parallel UCI path,
  `_resolve_root_solved` returns early on any nonzero status
  (`uci/root_parallel_gumbel.py:1374-1376`).

### Path-dependent fact, node-local storage

| Mechanism | Treatment of `solved` | Code |
|---|---|---|
| Transposition table | Copies priors and backs up donor W/N only, after a full history and repetition-stack context match. It **does not copy `solved`**. Each node has one path | `_mcts_tree.c:1963-2020`, `:599-667` |
| Compaction | Copies `solved` and `has_solved_child` verbatim | `_mcts_tree.c:4106-4107` |
| Self-play reuse | The root becomes the played child (`network_turn.py:1322-1327`). It is reused **only if `is_expanded(rid)` and `root_context_matches`** (`gumbel_c.py:1483-1499`). A twofold leaf is never expanded, so it is never reused | as cited |
| UCI `advance_root` | Repoints `_root_id` with **no expanded check and no solved reset** | `uci/search.py:2390-2427` |
| UCI classic Gumbel path | Goes through the `gumbel_c.py` gate above, so it is safe | |
| UCI PUCT pool / walker paths | `_drop_pool_root_if_support_differs` **returns early on an unexpanded root** and keeps it, `solved` and all | `uci/search.py:3036-3071` |
| Root-parallel path | `advance_root` refuses reuse (`search.py:2405-2408`) | |

### Where self-play stores the value

- `values_out` is the survivor child's completed Q, or `node_q(root)` for a solved root
  (`gumbel_c.py:2183-2190`, `:2069`). It enters the row through `c_process_ply`
  (`selfplay/network_turn.py:718-727`).
- A twofold survivor, or a root proven `DRAW`, gives **scalar q exactly 0**.
- The stored `search_wdl`, under production `search_wdl_draw_mode: net_raw`
  (`_mcts_tree.c:236-249`), is then a symmetric triple: W = L, with D the net's raw draw
  output. That is the signature the target measurement in §2b counts.
- The winning-root draw prune (`gumbel_c.py:1372-1392`) removes only threefold and
  50-move claim draws and on-board draws, never twofold children
  (`encoding/_lc0_ext.c:698-760`).

### lc0 and Stockfish (read from source, master on 2026-11-19)

**lc0** (`LeelaChessZero/lc0` at `bf15d558`, `src/search/classic/`):

- `search.cc` (twofold block, around l.1948-1961) marks a twofold terminal only if
  `repetitions == 1 && depth-1 >= 4 && two_fold_draws &&
  depth-1 >= plies_since_prev_repetition`. The earlier occurrence must lie at or below
  the root on the current path.
- Because a repetition cycle is at least 4 plies, the `depth-1 >= 4` clause is implied
  by the last clause.
- `repetitions >= 2` is a draw regardless of where the occurrences are.
- These draw-by-rule shortcuts are skipped at the root.
- `node.cc` `MakeTerminal` skips `SetBounds` for `Terminal::TwoFold`, so **a twofold is
  not a proof**.
- `NodeTree::MakeMove` calls `MakeNotTerminal()` on a terminal new head.
- `EnsureNodeTwoFoldCorrectForDepth` (`search.cc` around l.1520-1555) clears a reused
  twofold terminal whose first occurrence is now before the root and reverts its visits.

All of this was verified in the fetched source. That this matches the shipped binary in
`local_engines/` is unverified.

**Stockfish** (`official-stockfish/Stockfish` at `5d03b6e5`):

- `position.cpp` `is_repetition(ply)` is `st->repetition && st->repetition < ply`. The
  comment reads "repeats once earlier but strictly after the root, or repeats twice
  before or at the root".
- `search.cpp` returns `value_draw(nodes)` (a small randomised draw score) at a
  non-root node, before any TT probe, so the repeated node writes no TT entry. Ancestors'
  values that depend on such draws **are** stored (accepted graph-history interaction).
- `search.cpp:756` adds `upcoming_repetition` (cuckoo tables). If the side to move can
  play into a repetition, alpha is raised to a draw score. This is a **lower bound**,
  with an early cutoff only if it reaches beta; it is not an exact score.

## 2. Measurements on banked self-play

Source: `~/chess-artifacts/operations/rl-week-one-20261112-v1/fresh-current-policy-A39-{F12,Q50}-cycle-NN-collect-results-v1/game-*/`.
The opening is book; "searched" means ply ≥ `start_ply`.

### 2a. Game census (python-chess replay)

Script `<bank>/measure_repetition.py`; rows `<bank>/games_{F12,Q50}.jsonl`; tables
`<bank>/tables.md`, `<bank>/trend.out`, `<bank>/event_outcomes.out`.

| arm | cyc | searched plies/game | draw | TB adjudication | mate | threefold claim | 50-move claim |
|---|---|---|---|---|---|---|---|
| F12 | 1 | 92.9 | 44.3 | 51.0 | 33.9 | 15.1 | 0.0 |
| F12 | 6 | 102.9 | 46.9 | 56.8 | 25.5 | 17.7 | 0.0 |
| F12 | 12 | 106.4 | 56.2 | 54.7 | 17.2 | 27.6 | 0.5 |
| Q50 | 1 | 94.1 | 46.4 | 55.7 | 27.6 | 16.7 | 0.0 |
| Q50 | 6 | 110.3 | 45.8 | 61.5 | 22.9 | 15.1 | 0.5 |
| Q50 | 12 | 116.2 | 56.8 | 67.7 | 12.5 | 18.8 | 1.0 |

- Pooled, 18.2% of games end by threefold claim and 28 by 50-move claim.
- Of the 866 natural draws that are not mate, stalemate or TB, 838 are a claimable
  threefold and 28 are a 50-move claim (`<bank>/classify_other.out`).
- The claim rule ends the game when the side to move *can* play into a third
  occurrence, so no game reaches one.

| arm | plies into a twofold | games with a twofold | twofold events | irreversible move followed later | events in games that ended decisively |
|---|---|---|---|---|---|
| F12 | 1.15% | 31.8% | 2,804 | 33.8% (games with one: 70.1% drawn) | 9.5% |
| Q50 | 1.07% | 30.6% | 2,723 | 44.8% (games with one: 63.0% drawn) | 17.8% |

- All games: 50.8% decisive. Games in which an irreversible move followed a twofold
  were **less** often decisive (66.3% drawn).
- Leaving the cycle is therefore not a non-draw. The game statistic that bears on the
  value claim is the 13.6% of twofold events in games that ended decisively.

**Trend, cycle 1 to cycle 12** (OLS slope on per-game values; games independent):

| indicator | F12 change (z) | Q50 change (z) |
|---|---|---|
| searched plies / game | +11.4 (5.2) | +17.8 (7.9) |
| draw rate | +6.7 pp (2.0) | +1.2 pp (0.4) |
| threefold-claim ending | +7.6 pp (2.8) | +0.6 pp (0.3) |
| game contains a twofold | +10.0 pp (3.3) | +4.3 pp (1.4) |
| roots with a game-history-repeat child | +1.2 pp (4.6) | +0.8 pp (3.1) |
| mean halfmove clock at searched plies | +1.04 (4.7) | +1.10 (4.6) |

Causation is not established: every cycle retrains the net.

### 2b. Production-net stored targets at repeat-child roots (no GPU needed)

- Script: `<bank>/twofold_targets.py`, which re-derives and extends
  `<bank>/review-census/zeros.py`. Rows: `<bank>/targets_{F12,Q50}.jsonl`, all 24
  arm-cycles, 0 length mismatches.
- Aggregation: `<bank>/agg_targets.py` writes `<bank>/targets_tables.md`. CIs are 95%
  bootstraps that resample **games** within an arm-cycle (B = 1,000).
- **Definitions.**
  - "Rep root": a searched row where some non-zeroing legal move reaches a position
    already seen in the game since the last zeroing move. These are exactly the
    children the C search marks `DRAW`.
  - "q0": the stored `search_wdl_raw` has W == L exactly.
  - Policy mass uses the stored target's compact-1858 slots. I checked on a full game
    that legal-move mass sums to 1.000 in that convention, against 0.206 in Leela
    order.

| arm | cyc | rep rows | rep rows / all rows | **q0 \| rep** | q0 \| other | rep & q0 / all rows | target mass on repeat move(s) | one-hot on a repeat move | played the repeat |
|---|---|---|---|---|---|---|---|---|---|
| F12 | 1 | 489 | 2.77 [2.31, 3.28] | **32.5 [26.2, 38.3]** | 0.24 [0.18, 0.31] | 0.90 [0.66, 1.17] | 31.4 | 1.6 | 34.6 |
| F12 | 6 | 778 | 3.98 [3.24, 4.74] | **27.8 [22.7, 33.2]** | 0.30 [0.23, 0.38] | 1.10 [0.81, 1.40] | 27.2 | 0.8 | 31.8 |
| F12 | 12 | 972 | 4.80 [4.00, 5.65] | **28.1 [23.3, 33.5]** | 0.29 [0.23, 0.36] | 1.35 [1.05, 1.63] | 27.4 | 1.3 | 30.1 |
| Q50 | 1 | 564 | 3.15 [2.62, 3.77] | **33.7 [28.6, 38.9]** | 0.31 [0.24, 0.38] | 1.06 [0.80, 1.32] | 31.1 | 1.4 | 35.8 |
| Q50 | 6 | 751 | 3.58 [2.89, 4.28] | **25.3 [20.5, 30.1]** | 0.27 [0.21, 0.32] | 0.91 [0.69, 1.13] | 24.7 | 0.4 | 27.0 |
| Q50 | 12 | 943 | 4.27 [3.57, 4.95] | **24.3 [19.8, 29.1]** | 0.34 [0.27, 0.41] | 1.04 [0.80, 1.29] | 23.9 | 1.5 | 25.9 |
| **F12** | all | 9,400 | 3.90 [3.68, 4.12] | **25.9 [24.3, 27.3]** | 0.26 [0.24, 0.28] | **1.01 [0.93, 1.08]** | 25.2 [24.0, 26.8] | 1.10 [0.88, 1.33] | 28.3 [26.7, 30.0] |
| **Q50** | all | 9,895 | 3.94 [3.72, 4.16] | **24.4 [23.0, 25.8]** | 0.28 [0.26, 0.30] | **0.96 [0.90, 1.03]** | 23.6 [22.3, 25.0] | 1.00 [0.78, 1.25] | 26.3 [24.8, 27.8] |

All 24 rows, with CIs on every cell, are in `<bank>/targets_tables.md`. The enrichment
of q0 at rep roots over other roots ranges from 68× to 136× across arm-cycles.

**Further readings:**

- q0 given rep root **and the played move is the repeat**: 79.4% [77.3, 81.3] (F12)
  and 80.4% [78.7, 82.2] (Q50). In other words, the stored scalar is the repeat child's
  exact 0.
- **Trend, slope per cycle** (game bootstrap):

  | indicator | F12 | Q50 |
  |---|---|---|
  | rep rows / all rows | +0.155 pp [+0.094, +0.213] | +0.105 pp [+0.045, +0.163] |
  | q0 \| rep | −0.31 pp [−0.73, +0.15] | −0.55 pp [−0.95, −0.11] |
  | rep & q0 share of all rows | +0.031 pp [+0.010, +0.053] (0.90% → 1.35%) | +0.007 pp [−0.013, +0.028] |
  | one-hot on a repeat move | flat | flat |

  Exposure grows. The per-row exact-zero rate does not grow (in Q50 it falls), so the
  share of exact-zero training rows grows only in F12.
- **Stored q against the eventual outcome:**

  | rep rows | n | outcome W / D / L | mean q | decisive |
  |---|---|---|---|---|
  | F12, q0 | 2,433 | 5.6 / 89.3 / 5.1 | | 10.7% |
  | F12, q ≠ 0 | 6,967 | 29.6 / 67.3 / 3.1 | +0.252 | |
  | Q50, q0 | 2,411 | 10.1 / 81.2 / 8.7 | | 18.8% |
  | Q50, q ≠ 0 | 7,484 | 31.4 / 62.6 / 6.0 | +0.257 | |

  Most exact-zero rows do end drawn. About 1 in 9 (F12) and 1 in 5 (Q50) do not.
- **Calibration** (`<bank>/calib.py`, `.out`):
  - The mean outcome (W = +1, D = 0, L = −1) on exact-zero rep rows is **+0.006**
    (F12) and **+0.015** (Q50), with W and L symmetric, so the zeros are accurate on
    average.
  - The non-zero rep rows store a mean q of +0.252 / +0.257 against a mean outcome of
    +0.265 / +0.254, so they are calibrated too.
  - This is partly self-fulfilling. **86.9% (F12) and 86.8% (Q50) of the exact-zero
    rows are rows where the repeat was played**, and playing the repeat steers the game
    toward the draw the target claims. Outcomes therefore neither confirm nor refute
    the zero target.
  - Not demonstrated: that the zeros are wrong. Demonstrated: they are path-dependent
    exact proofs produced without searching the repeated position.
- **Syzygy: no coverage.** No rep-root searched row has 6 men or fewer. The collection
  adjudicates such positions at the root before any search (`bt4_outcome.py`), so
  Syzygy cannot price these rows. Deep SF is needed (§4 B).

## 3. Repros (tiny random nets, CPU)

### (i) Stale DRAW root after UCI reroot

`<bank>/repro_i_uci_reroot.py`, outputs `<bank>/repro_i_uci_reroot_{classic,walkers,pool}.out`.

The script searches a position with the `tests/test_uci_solved_root_stops` worker. It
then finds the shallowest node that is a natural twofold `DRAW` (no hand-marking),
calls `advance_root` onto it, and searches again.

| path | case | root before run | after `go nodes 2000` | children proven | `go depth 4` |
|---|---|---|---|---|---|
| classic Gumbel | KR vs K, KQ vs KR | `DRAW`, unexpanded | 0 (fresh root via the gate) | 2/15, 1/35 | stops like the fresh control |
| walker | KR vs K | `DRAW`, unexpanded | **`DRAW`** | **1/14** | runs to the 30 s cap (32,816 nodes); the control stops after 1,984 |
| walker | KQ vs KR | `DRAW`, unexpanded | **`DRAW`** | 1/7 | to the cap (as the control) |
| PUCT pool | opening (Ne5 Ng8 Nf3), KR vs K | `DRAW`, unexpanded | **`DRAW`** | 1/20, 1/14 | KR vs K to the cap (46,208); the control stops at 30,272 |

- **The pool-path rows are a single nondeterministic draw.** The pool's first search
  is not deterministic. In the independent reviewer's rerun the pool path reached a
  natural twofold `DRAW` root 0 of 3 times, while the walker path reproduced fully.
- The reviewer's 64-node `go depth` stop came from hand-marking an *expanded*
  grandchild. On main the natural mark sits on an unexpanded leaf.
- With a random net `score cp 0` is uninformative (the controls also print 0).
- **Requirement on option C's test (PR #1421).** It must assert its precondition, i.e.
  that the root status is `DRAW` and the root is unexpanded *before* the post-reroot
  run. It must also build that state deterministically: either via the walker path,
  which reproduces, or by marking an unexpanded twofold leaf by hand. Otherwise the test
  can pass vacuously when the nondeterministic search never creates the stale root.

### (ii) Self-play reuse

`<bank>/probe_ii_selfplay_reuse.py`, `<bank>/probe_ii_summary.out`.

The probe calls `run_gumbel_root_many_c` the way `network_turn.py` does (persistent
tree, `root_node_ids` = the child of the banked move). It ran over the 16 cycle-12
games with the most twofold plies, 2,179 roots, 64 sims.

- 91 offered roots carried `DRAW`. All were unexpanded twofold leaves and **all were
  rejected** by the `is_expanded` gate. **0 of 1,324 reused roots carried a status.**
- Self-play stores exact zeros through the twofold rule itself: 37 rows. Of these, 33
  are twofold survivors, and 4 are roots proven `DRAW` in-search because every legal
  move repeats a game-history position (one-hot target, value 0).
- §2b gives the production-net rate of the same signature.

### (iii) An unsearched exact 0 beats searched values: the repeated position is never searched

`<bank>/repro_iii_twofold_hides_mate.py`, outputs `<bank>/repro_iii_twofold_hides_mate_{200,800}sims.out`.

- Repeated position P = `6k1/r4ppp/8/8/8/8/2P2PPP/4R1K1 w`: white, a pawn up, has Re8#.
- History P, 1.Re2 Ra8 2.Re1 gives root `r5k1/5ppp/8/8/8/8/2P2PPP/4R1K1 b`. Black's
  Ra7?? returns to P.
- Evaluator: "the side with more pawns wins". That puts black in the defender's regime:
  every line that does not repeat evaluates below 0.
- A second variant adds a large prior on Re1-e8, so a search that expands P can find
  the mate.
- 8 seeds per cell.

| evaluator, sims | root | chose Ra7?? | Ra7 node | stored target on Ra7 | stored value |
|---|---|---|---|---|---|
| pawn count, 200 | with history | **5/8** | `DRAW`, never expanded | 0.999 | 0.000 |
| pawn count, 200 | FEN only | 0/8 | expanded, unproven (solved 0, valued −1 like every move) | ≤0.09 | −1.000 |
| + Re8 prior, 200 / 800 | with history | **5/8 / 5/8** | `DRAW`, never expanded | 0.999 / 1.000 | 0.000 |
| + Re8 prior, 200 / 800 | FEN only | 0/8 / 0/8 | **proven lost in seeds 6 and 7** (solved = WIN for white); unproven in 6 seeds | 0.000 in seeds 6, 7 | −1.000 |

- **What this shows in general:** an unsearched exact 0 beats searched values below
  0, because the repeated position is never searched. The mate is irrelevant to that
  conclusion.
- **Where the mate claim holds:** only in the paired seeds 6 and 7. There the control
  search proves Ra7 loses to the mate in 1, and the same seed with history plays Ra7
  with target 0.999–1.000.
- In the other 3 history seeds that did not pick it, Ra7 was outside the 16 Gumbel
  candidates.
- Random or "optimistic" values never prefer the repeat (0/8 in both arms; 1/8 for the
  FEN-only random net at 800 sims). The defect needs the defender's regime.
- lc0 would search P here, because the earlier occurrence is before the root.

## 4. Options (prospective)

Every option is weighed against these rules:

- `CLAUDE.md` "Experiments", rules 1 and 5: a record with hypothesis, control, one
  yardstick as an exact command, a pre-committed threshold, and a recovery plan; and an
  audit against a frozen deep-SF set before any training compute.
- The search config must be frozen while `wdl_regret` is read as a net signal.
- A Python-only `GumbelConfig` field is dropped by the C path. A new knob must be
  plumbed into C, with a take-effect proof logged from the C consumer.

### Option C: engine-side reuse fix (in progress: PR #1421)

**Change.** UCI reroot refuses a root that is unexpanded, or that has a nonzero
`solved` status with an unproven child, and rebuilds a fresh root. This is what
`gumbel_c.py:1483-1499` already does for self-play.

- **Self-play footprint:** none (§3 ii). The change is in `uci/` only.
- **Identity test:** a fixed-seed `tests/test_e2e_smoke.py -k gumbel_selfplay_smoke`
  produces byte-identical output, and `tests/test_mcts_*` / `tests/test_gumbel_*` are
  unchanged.
- **Yardstick:** the PR's reroot test, with the precondition assertion and the
  deterministic construction required in §3 (i). **Threshold:** 0 stale roots, and the
  test fails on the parent commit (mutant run recorded).
- **Recovery:** revert.

### Option A: keep `Q = 0` for twofold, but never as a proof

**Change.** No `SOLVED_DRAW` mark for a twofold, no `tree_resolve_from_children` proof,
no solved-root stop, and the mark never survives a reroot. A is option B2 below without
B1.

- **Leaves the dominant exact-zero source in place.** 86.9% (F12) and 86.8% (Q50) of
  the exact-zero rep rows are rows where the repeat was played, and that child is still
  an exact-0 terminal under A.
- Does not fix (iii).
- It changes only the one-hot or zero-budget proven roots, and their ancestors' proofs.

### Option B: lc0-style twofold, split into B1 and B2 (recommended experiment)

B is two separable changes. They have different identity properties, so they are
specified, tested and audited separately.

**What the case for B rests on.** The production targets at repeat-child rows are not
shown to be wrong (§2b: the exact zeros are calibrated against outcomes on average, and
that comparison is partly self-fulfilling). The case rests on:

- **(a)** path-dependent proofs and stale marks (§1, §3 i);
- **(b)** the repeated position never being searched (§3 iii);
- **(c)** departure from lc0 and Stockfish practice (§1);
- **(d)** whatever the frozen audit below finds.

It is entirely possible that the audit says "no measurable target benefit". In that
case only the engine-side guard (#1421) is warranted.

#### B1: root boundary only

**Change.**

- A first repeat is a terminal draw only if the earlier occurrence is at or after the
  root on the current path. Implement this by storing the root's `hash_stack_len`
  boundary and matching only indices at or above it.
- lc0's `depth-1 >= 4` clause is implied, since a cycle is at least 4 plies; it will be
  written explicitly anyway.
- A threefold still counts pre-root occurrences.
- **Proof semantics are unchanged:** an in-tree twofold is still `SOLVED_DRAW` and still
  propagates.
- On reroot, any twofold leaf whose earlier occurrence moves before the new root becomes
  invalid, so reuse must clear it or refuse to reuse.
- `tests/test_mcts_root_history_lifecycle.py:99-109` changes, with the reason stated.

**Fixes:**

- (b) for the case that matters in self-play, a repeat of a *game* position: the
  repeated position is now searched.
- (c) in part: lc0's occurrence rule.
- Not (a)'s proof semantics.

**Footprint.**

- Direct: the 3.90–3.94% rep-root rows, of which about 25% currently store q exactly 0.
  That is about 1% of all rows.
- About 1.1% of played moves are candidates to change.
- Not measured: repeats of pre-root positions deeper below the root. Measuring it needs
  searches with the real net.

**Identity: both claims hold because only pre-root matches change, under the
conditions below.**

1. A C counter of twofold hits whose match index lies below the root boundary. On a
   fixed seed and position set, every row with counter 0 produces a bit-identical
   policy, value and move, under the conditions stated here:

   **Conditions for both counter-based identity tests (B1 and B2).** "Counter 0 ⇒
   bit-identical" holds only when each row's search is independent: a fresh tree (no
   subtree carried from an earlier ply, and a fresh transposition table) and
   batch-invariant evaluation (CPU fp32, or one row per call). Otherwise a row with zero
   hits in its own search can still differ through:
   - (i) the subtree carried from an earlier ply (`network_turn.py:1322-1327`, gate
     `gumbel_c.py:1483-1486`), whose N/W/solved marks differ between arms, and B1's own
     reroot clearing;
   - (ii) TT donors built in earlier, divergent searches (`_mcts_tree.c:1995-2017`);
   - (iii) cross-game batch composition under bf16/TF32 GPU kernels (a stated risk, not
     measured);
   - (iv) trajectory divergence once any played move differs.

   The alternative is a counter cumulative over the whole game and the tree's lifetime,
   with one game at a time; then only games whose cumulative counter is 0 are claimed
   identical. Without one of these setups the gate raises false "leak" alarms in
   production-style runs.
2. On `data/audit_set_v1.jsonl`, targets are bit-identical under both rules. Its rows
   are FEN-only, with no move history, so the repetition stack starts empty and no
   pre-root match can exist. In-tree twofolds still fire there, identically under both
   rules.

**Audit B1** (deciding yardstick): the frozen target-side audit below, with
`--rule any_prior,root_boundary`.

#### B2: in-tree twofold not stored as a proof; reverted on reroot

**Change.**

- In-tree twofold leaves keep value 0 but are not `SOLVED_DRAW`. They are excluded from
  `tree_resolve_from_children` and from the solved-root stop.
- On reroot they are cleared and their visits reverted, as lc0's
  `EnsureNodeTwoFoldCorrectForDepth` does.

**Fixes:** (a) and the rest of (c), i.e. lc0's "not a bound" rule.

**Identity.**

- A C counter of **every** twofold hit, in-tree and pre-root alike. Rows whose searches
  hit no twofold at all are bit-identical, under the same conditions as B1 (independent
  per-row search with a fresh tree and fresh TT, batch-invariant evaluation; or a
  counter cumulative over the game with one game at a time).
- **No audit-set identity claim.** In-tree twofolds fire on `audit_set_v1` too, and B2
  changes how they resolve, propagate and stop a solved root. Its targets can
  legitimately differ there.

**Audit B2:** the same frozen audit procedure, run as a separate readout with
`--rule root_boundary,root_boundary+no_proof`. That way B2 is judged on top of B1.

The `v1` frozen rows were selected for pre-root repeats, so they are a weak sample for B2,
whose effect is on in-tree proofs anywhere. B2's audit therefore needs its own selection:
rows whose search had any twofold hit, frozen the same way as `v1` (written as
`twofold_audit_rows_b2_v1.jsonl`, with its sha256 recorded before either arm runs).

**Order.** B1 first: it is the minimal step, cleanly testable, and covers the game-history
case that produces the self-play rows measured in §2b. Then B2, only if B1 passes, or as
an engine-correctness change that is decided separately.

#### Frozen target-side audit (for B1, and then for B2)

**Question** (not an asserted hypothesis): do B's stored q and stored policy target agree
better with deep SF than the current rule's, on rows where the rule matters?

**Control:** the current rule (`any_prior`): same checkpoint, rows, seeds and sims.

**Prerequisites:**

- `scripts/audit_targets.py` has **no twofold-rule knob** today.
- The audit script below does not exist yet.
- The rule must be a C-plumbed knob, with a take-effect proof logged from the C counter.

**Why the standing frozen audit set cannot be the yardstick.** `data/audit_set_v1.jsonl`
rows carry no move history, so they contain no pre-root repetition: B1 is an identity on
them by construction. They still serve as B1's identity control.

1. **Freeze the rows before either arm runs.**
   - From `<bank>/targets_{F12,Q50}.jsonl`, take a game-clustered random sample of 2,000
     rep-root rows, plus a 2,000-row continuation for the one permitted extension. Use
     seed 20261119, stratify by arm and cycle, and take at most 2 rows per game.
   - Write `<bank>/twofold_audit_rows_v1.jsonl`. Each line holds the arm, cycle, game,
     ply, the full UCI move list from `root_fen`, the move-list sha256, and the banked
     outcome.
   - Record the file's sha256 in this record first. The set never changes; a new sample
     means `v2`.
2. **Label once with deep SF**, with the full move history passed as
   `position fen <root_fen> moves …` (MultiPV 10, ≥ 1M nodes, production Syzygy pair).
   Bank per-move cp/WDL and PVs keyed by row and engine settings, plus `game_id`.

   ```bash
   PYTHONPATH=. python3 scripts/twofold_target_audit.py label \
     --rows <bank>/twofold_audit_rows_v1.jsonl --rows-sha256 <recorded> \
     --multipv 10 --nodes 1000000 --syzygy <production syzygy pair> \
     --out <bank>/twofold_audit_sf_v1.jsonl
   ```

3. **Run both arms** with the SL base in a GPU pause window, at the collection's
   self-play search profile, read from its `resolved-profile.json`.
   - **Fresh tree per row.** The deciding comparison searches each row from a fresh tree
     (fresh TT, no carried subtree), with the full move history, identically in both
     arms.
   - **Consequence.** The stored production targets came from warm trees, carried across
     plies. This measures the rule's effect at a cold root, not the carried-subtree
     effect.
   - **Warm-tree variant (secondary, non-deciding).** Replay each row's game from the
     opening with tree reuse, as `network_turn.py` does, and score it the same way.

   ```bash
   PYTHONPATH=. python3 scripts/twofold_target_audit.py search \
     --rows <bank>/twofold_audit_rows_v1.jsonl --rows-sha256 <recorded> \
     --checkpoint $SL_BASE --profile <collection resolved-profile.json> \
     --rule any_prior,root_boundary --out <bank>/twofold_audit_search_b1_v1.jsonl
   PYTHONPATH=. python3 scripts/twofold_target_audit.py score \
     --sf <bank>/twofold_audit_sf_v1.jsonl --search <bank>/twofold_audit_search_b1_v1.jsonl \
     --out <bank>/twofold_audit_readout_b1_v1.json
   ```

4. **Scoring** (paired per row, CIs clustered by game):
   - **Value:** |q − q_SF|, with q_SF = W − L of the deep-SF WDL at the root.
   - **Policy:** E[regret] of the stored target in cp against SF's MultiPV, plus its
     mass on moves SF scores ≥ 50 cp below best.
5. **The ruler's bias, stated.** Stockfish, given the full history, does not score a
   single pre-root repeat as a draw; it searches the position. On exactly these rows
   |q − q_SF| therefore leans structurally toward B1, which does the same. The PASS
   margin below (−0.005) is small next to that.
   - Deep SF stays the teacher.
   - As a **secondary, non-deciding** reading, report |q − z| against the banked game
     outcome z on the same frozen rows. This is partly self-fulfilling where the repeat
     was played (§2b).
6. **Pre-committed outcomes:**
   - **PASS** (the variant proceeds to a training-arm record): the value-error
     difference (variant − control) has a 95% upper bound below −0.005, **and** the
     policy E[regret] difference has a 95% upper bound ≤ +2 cp.
   - **KILL:** the value-error difference has a 95% lower bound ≥ 0, or the policy
     E[regret] difference has a 95% lower bound > +2 cp.
   - **Anything else is UNREAD.** Extend once to the frozen 4,000 rows and apply the
     same rule. Still unread means the variant is not adopted.

**Guard: arena non-inferiority. It does not decide B.**

- **Why `matched_time`, not `matched_sims`.** `matched_sims` passes no tree
  (`selfplay/match.py`), so every move is a cold tree and the reroot handling is never
  exercised.
- **Search settings.** The guard plays the **UCI play-search settings**, not the
  self-play ones, so it tests the engine and not the target generator.
- **Prerequisites:** a per-side UCI option and per-side arena arguments, neither of
  which exists yet.
- **Command:**

  ```bash
  PYTHONPATH=. python3 scripts/arena_standard.py --candidate $SL_BASE --reference $SL_BASE \
    --mode matched_time --ms-per-move 200 --games 2000 \
    --cand-uci-args 'TwofoldRule=root_boundary' --ref-uci-args 'TwofoldRule=any_prior' \
    --label twofold-guard-b1-v1
  ```

- **Power:** about ±11 Elo 95% CI at 2,000 games (about ±15 at 1,000). A change
  touching about 1.1% of moves is probably < 10 Elo, so the guard catches only a gross
  regression.
- **Outcomes:**
  - **PASS:** the lower bound is ≥ −15.
  - **FAIL:** the upper bound is < 0.
  - If both hold (an estimate between −15 and 0 with a tight CI), **FAIL wins**.
  - If neither holds, extend once to 4,000 games; still neither, then FAIL.
  - A guard FAIL blocks the variant even after an audit PASS.

**Other terms:**

- **Cost:** for B1, the C root boundary, reroot clearing and ABI bump (about 1 day);
  B2 is about 1 more day. Then CPU SF labelling of 4,000 rows, two GPU search passes per
  readout, and a 2,000-game guard, all in pause windows.
- **Confounds:** one data-affecting change per readout window, so do not overlap #1410
  or claim-rule changes, and do not overlap B1 with B2. Each is a search-config change
  for `wdl_regret`.
- **Recovery:** the rule defaults to `any_prior`, which is bit-identical to today.

### Ranking and recommendation

1. **C now (PR #1421):** the engine-side guard.
2. **B1 as the registered experiment,** decided by the frozen target audit, with the
   arena as a guard only.
3. **B2** after B1.
4. **A** only as a fallback.

If B1's audit finds no measurable target benefit, stop at C.

**No self-play-affecting code change is made until the owner chooses.**

## 5. Same-coloured bishops: insufficient material

- **C rule:** with no pawns, rooks or queens, the C board calls material insufficient
  only for K vs K, K+minor vs K, or **exactly** K+B vs K+B on one colour, i.e. 4 men
  (`encoding/_cboard_impl.h:1394-1408`).
- **python-chess rule:** material is insufficient whenever each side's only non-king
  pieces are bishops, all on one square colour (and no knights).
- **Check** (`<bank>/check_bishops.py`, `.out`): after
  `4b3/8/8/8/6bk/1B6/2r5/2K5 w - - 97 1` Bxc2, python-chess reports a draw by
  insufficient material and the C board reports the game not over.
- **Banked footprint:** 0 positions in 4,608 games, all plies.
- **Why it is zero:** the game end uses python-chess, and positions with 6 men or fewer
  are TB-adjudicated. A mismatch needs 7 or more men made only of kings and
  same-coloured bishops.
- **Where it can still matter:** in-search nodes without a TB probe (via bishop
  promotions), and the C `is_game_over` used for forced moves (`network_turn.py:482`).
- **Proposed fix:** mirror python-chess's per-colour rule, with a differential test.
- **Footprint:** expected 0 rows. Identity: the census reports 0 changed positions, and
  the self-play smoke output is byte-identical.

## What this record could not determine

- How often, below the root, a repeat of a pre-root position occurs (B's indirect
  footprint). It needs real-net searches.
- Syzygy truth for rep-root rows: there are none with 6 men or fewer (§2b). Deep-SF
  labels are part of B's audit.
- Whether the twofold rule causes the cycle trend in shuffling.
- lc0 binary parity with the source I read.
