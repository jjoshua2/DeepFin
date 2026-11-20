# Twofold repetition as a proven draw: semantics, footprint, defects, plan

**Status:** investigation and prospective plan only. **No code change is made by this
record, and none will be made until the owner agrees to one of the options below.**
CPU only (no GPU, no training compute). Code read at `origin/main` 5c9b44722.
Scripts, raw outputs and per-game rows are banked under
`~/chess-artifacts/operations/rl-claude-20261119/twofold-audit/` (referred to below as
`<bank>/`).

## Summary

- The compiled search scores a position as an **exact, proven draw** (`Q = 0`, solved
  status `DRAW`) the first time it repeats, and it counts occurrences **before the search
  root (game history) as well as inside the tree**. lc0 counts only occurrences at or
  below the root, and does not treat the twofold as a proof. Stockfish needs a threefold
  when the earlier occurrence is before the root.
- The `DRAW` mark is used as a proof. It ends the search early at a proven root, it makes
  the policy target one-hot, and it is kept through tree reuse and compaction.
- **Self-play is protected from the stale-root defect** by the reuse gate in
  `gumbel_c.py`. Self-play still stores a value of exactly 0, with a one-hot target, on
  roots that are "proven drawn" only because every legal move repeats a position from
  the game history.
- **The UCI engine is not protected on its PUCT pool and walker paths.** After
  `advance_root` onto a twofold leaf, the root keeps `DRAW` while 13 of its 14 children
  are unproven, and it still reports `DRAW` after 2,000 more nodes.
- **The twofold leaf is never expanded, so tactics inside the repeated position are
  invisible.** In a constructed position the defender walks into a mate in 1 because
  the repeat scores 0. This happened in 5 of 8 seeds; the history-free control did it
  in 0 of 8.
- **Banked games** (A39 F12/Q50, 12 cycles each, 4,608 games, 497,027 searched plies):
  - 3.93% of searched roots have at least one child that repeats a game-history
    position. Today each such child is an exact-draw terminal.
  - 1.11% of searched plies play into a twofold, and 31.2% of games contain one.
  - Games end on a **claimable** threefold, one ply before a third occurrence. This
    happens in 18.2% of games; no game reaches an actual third occurrence.
  - Of 5,527 twofold events, **39.2% were followed by play leaving the cycle**
    (an irreversible move later), so the "draw" did not come true. 29.9% (F12) and
    37.0% (Q50) of the games where that happened ended decisively.
  - Repetition and shuffling rise across cycles: roots with a repeat child, and the
    halfmove clock, rise in both arms (z 3–5). Threefold-claim endings rise in F12 only
    (+7.6 pp, z = 2.8).
- **Recommendation:** fix the engine now with option C, which has no self-play footprint
  and merges on an identity test. Then run option B (lc0-style: in-tree occurrences
  only, never a proof, recomputed on reroot) as a registered experiment, scored
  audit-first and decided by one arena command.

## 1. Semantics in code

### Where repetition is detected and against what

| Fact | Code |
|---|---|
| Search terminal test: true game over (mate, stalemate, **strict** threefold, 50-move, insufficient material), else **any prior occurrence** gives `Q = 0.0`, `SOLVED_DRAW` | `mcts/_search_terminal.h:59-72` (`cboard_search_terminal`) |
| "Any prior occurrence" scans the board's whole `hash_stack` | `encoding/_cboard_impl.h:1416-1422` (`cboard_is_repetition`) |
| `hash_stack` holds every position since the last irreversible move: it is cleared on a zeroing move and the pre-move key is pushed otherwise | `_cboard_impl.h:1201-1211` (inside `cboard_push`) |
| The root board imported from python-chess fills `hash_stack` from `board._stack`, which is **game history before the root**, back to the halfmove clock (plus one unreachable pre-zeroing key) | `encoding/_lc0_ext.c:448-477` |
| The game ends only on a third occurrence (strict, in C) | `_cboard_impl.h:1424-1445` (`cboard_is_threefold_repetition`, `cboard_is_game_over`) |
| Banked collection: the game ends at the root if `board.outcome(claim_draw=True)` fires, i.e. when the side to move **can claim** threefold or 50-move | `selfplay/bt4_outcome.py:101-105` (mode `rule50_match_v1`, as stamped in every banked `provenance.json`) |
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
  the parent. A twofold `DRAW` can therefore turn a would-be `LOSS` into a proven `DRAW`.
  It can also hide a `WIN` for the side to move at the repeated node, because that node
  is never searched. A WIN or LOSS proof never rests on a draw, as `uci/search.py:3607-3625`
  already notes.
- **Selection uses it.** In `tree_select_child` a `DRAW` child's Q is forced to exactly 0
  (`_mcts_tree.c:1009-1013`, and `:3509-3511` on the PUCT path). Public Q queries return
  the proof value (`tree_child_q`, `:351-357`).
- **Early stops use it.** At the start of each Gumbel halving round, a root that is
  already solved `WIN`/`DRAW` collapses to one proving child with zero remaining budget
  (`gss_begin_round`, `_mcts_tree.c:1599-1644`). The Python side then returns a one-hot
  policy and `value = node_q(root)`, which is exactly 0 for a draw
  (`mcts/gumbel_c.py:1952-1982`).
- **It is monotone.** A node already solved is never revisited
  (`_mcts_tree.c:1188-1193`, `:1204`). On the root-parallel UCI path,
  `_resolve_root_solved` returns early on any nonzero status
  (`uci/root_parallel_gumbel.py:1374-1376`).

### Path-dependent fact, node-local storage

| Mechanism | Treatment of `solved` | Code |
|---|---|---|
| Transposition table | Copies priors and backs up donor W/N only, after a full history and repetition-stack context match. It **does not copy `solved`**. Each node has one path, so the TT does not leak a path-dependent status | `_mcts_tree.c:1963-2020`, `:599-667` |
| Compaction | Copies `solved` and `has_solved_child` verbatim | `_mcts_tree.c:4106-4107` (`compact_to_subtree`) |
| Self-play reuse | The root becomes the played child (`network_turn.py:1286-1290`). It is reused **only if `is_expanded(rid)` and `root_context_matches`**; otherwise a fresh root is built (`gumbel_c.py:1397-1413`). A twofold leaf is terminal and unexpanded, so it is never reused | as cited |
| UCI `advance_root` | Repoints `_root_id` to the descendant with **no expanded check and no solved reset** | `uci/search.py:2390-2427` |
| UCI classic Gumbel path | Goes through the `gumbel_c.py` gate above, so it is safe | |
| UCI PUCT pool / walker paths | `_drop_pool_root_if_support_differs` **returns early on an unexpanded root** and keeps it, `solved` and all | `uci/search.py:3036-3071` |
| Root-parallel path | `advance_root` refuses reuse (`search.py:2405-2408`), so a stale mark cannot arrive by reroot | |

### Where self-play stores the value

- `values_out` is the survivor child's completed Q, or `node_q(root)` for a solved root
  (`gumbel_c.py:2082-2089`, `:1981`).
- It enters the row through `c_process_ply` (`selfplay/network_turn.py:700-709`), which
  builds the stored `search_wdl` target (`_mcts_tree.c:236-249`).
- A survivor that is a twofold child therefore stores an exact 0. A root proven `DRAW`
  stores a one-hot policy and an exact 0.
- The winning-root draw prune (`gumbel_c.py:1285-1305`) removes only threefold and
  50-move claim draws and on-board draws, never twofold children
  (`encoding/_lc0_ext.c:698-760`).

### lc0 and Stockfish (read from source, master on 2026-11-19)

**lc0** (`LeelaChessZero/lc0` at `bf15d558`, `src/search/classic/`):

- `search.cc` (twofold block, around l.1948-1961, in `ExtendNode`) marks a twofold
  terminal only if `repetitions == 1 && depth-1 >= 4 && two_fold_draws &&
  depth-1 >= plies_since_prev_repetition`. That means the earlier occurrence must lie
  **at or below the root on the current path**. Game history before the root does not
  count. A position repeated twice or more (`repetitions >= 2`) is a draw regardless.
- These draw-by-rule shortcuts are skipped at the root itself.
- `node.cc` `MakeTerminal`: `if (type != Terminal::TwoFold) SetBounds(...)`, so **a
  twofold is not a bound (proof)**.
- `NodeTree::MakeMove` calls `MakeNotTerminal()` on a terminal new head.
- `SearchWorker::EnsureNodeTwoFoldCorrectForDepth` (`search.cc` around l.1520-1555)
  clears a reused twofold terminal whose first occurrence is now before the root and
  reverts its visits up the path.

All of this was verified in the fetched source. That this matches the shipped binary in
`local_engines/` is unverified.

**Stockfish** (`official-stockfish/Stockfish` at `5d03b6e5`):

- `position.cpp` `is_repetition(ply)` is `st->repetition && st->repetition < ply`. The
  comment reads "draw score if a position repeats once earlier but strictly after the
  root, or repeats twice before or at the root".
- `search.cpp` returns `value_draw(nodes)` (a small randomised draw score) at a
  non-root node before any TT probe, so the repeated node itself writes no TT entry.
- Ancestors' values that depend on such draws **are** stored in the TT (accepted
  graph-history interaction). The common statement that "it is never stored" is
  therefore only true of the repeated node.

## 2. Banked self-play census

Source: `~/chess-artifacts/operations/rl-week-one-20261112-v1/fresh-current-policy-A39-{F12,Q50}-cycle-NN-collect-results-v1/game-*/provenance.json`.

- Each game's full move list (`tablebase_witness.moves` from `root_fen`, opening
  included) is replayed with python-chess.
- "Searched" means plies at or after `start_ply`. The opening is book.
- Script: `<bank>/measure_repetition.py`. Rows: `<bank>/games_{F12,Q50}.jsonl`. Tables:
  `<bank>/aggregate.py` writes `<bank>/tables.md`, and `<bank>/trend.py` writes
  `<bank>/trend.out`. The full per-cycle tables are in `<bank>/tables.md`.

**How games end (% of games, 192 per cycle):**

| arm | cyc | searched plies/game | draw | TB adjudication | mate | threefold **claim** | 50-move claim | actual 3rd occurrence |
|---|---|---|---|---|---|---|---|---|
| F12 | 1 | 92.9 | 44.3 | 51.0 | 33.9 | 15.1 | 0.0 | 0 |
| F12 | 6 | 102.9 | 46.9 | 56.8 | 25.5 | 17.7 | 0.0 | 0 |
| F12 | 12 | 106.4 | 56.2 | 54.7 | 17.2 | 27.6 | 0.5 | 0 |
| Q50 | 1 | 94.1 | 46.4 | 55.7 | 27.6 | 16.7 | 0.0 | 0 |
| Q50 | 6 | 110.3 | 45.8 | 61.5 | 22.9 | 15.1 | 0.5 | 0 |
| Q50 | 12 | 116.2 | 56.8 | 67.7 | 12.5 | 18.8 | 1.0 | 0 |

Pooled over 4,608 games, 18.2% end by threefold claim and 28 by 50-move claim. Of the
866 natural draws that are not mate, stalemate or TB adjudication, 838 are a claimable
threefold: in 635 the final position is itself a twofold, and in 203 a move into a
twice-seen position is available. The other 28 are 50-move claims
(`<bank>/classify_other.out`).

**Repetition exposure (pooled per arm):**

| arm | searched plies | plies into a twofold | games with a twofold | roots with ≥1 child repeating a game-history position (exact DRAW today) | twofold events | ended by repetition claim, no irreversible move after | play left the cycle |
|---|---|---|---|---|---|---|---|
| F12 | 243,453 | 1.15% | 31.8% | 3.92% (9,540) | 2,804 | 65.5% | 33.8% |
| Q50 | 253,574 | 1.07% | 30.6% | 3.95% (10,014) | 2,723 | 54.0% | 44.8% |

- "Ended by repetition claim" is a game-level attribution: the game ended by a threefold
  claim with no zeroing move after the event, and the claim may involve a different key
  from the same cycle.
- The 0.7–1.2% remainder of events ended by TB adjudication or mate without leaving the
  cycle.
- In the games where play left a twofold cycle, 29.9% (F12) and 37.0% (Q50) were
  decisive.
- So treating a twofold as a draw was false for roughly 2 in 5 events in these games.

**Trend, cycle 1 to cycle 12** (OLS slope per cycle on per-game values, games treated as
independent):

| indicator | F12 change (z) | Q50 change (z) |
|---|---|---|
| searched plies / game | +11.4 (5.2) | +17.8 (7.9) |
| draw rate | +6.7 pp (2.0) | +1.2 pp (0.4) |
| threefold-claim ending | +7.6 pp (2.8) | +0.6 pp (0.3) |
| game contains a twofold | +10.0 pp (3.3) | +4.3 pp (1.4) |
| roots with a game-history-repeat child | +1.2 pp (4.6) | +0.8 pp (3.1) |
| mean halfmove clock at searched plies | +1.04 (4.7) | +1.10 (4.6) |

- **Shuffling and repetition exposure grow in both arms.** Repetition-claim endings and
  draws grow in F12 only.
- Q50's length growth comes with a rising share of TB adjudication (55.7% to 67.7%),
  not with repetition.
- Causation is not established. Every cycle retrains the net, so the trend is confounded
  with everything else that changed.

## 3. Repros (tiny random nets, CPU)

**(i) Stale DRAW root after UCI reroot** (`<bank>/repro_i_uci_reroot.py`, outputs
`<bank>/repro_i_uci_reroot_{classic,walkers,pool}.out`).

The script searches a position with the `tests/test_uci_solved_root_stops` worker. It
then finds the shallowest node that is a natural twofold `DRAW` (no hand-marking),
calls `advance_root` onto it, and searches again.

| path | case | root before run | after `go nodes 2000` | children proven | `go depth 4` |
|---|---|---|---|---|---|
| classic Gumbel | KR vs K, KQ vs KR | `DRAW`, unexpanded | **0** (fresh root via the `gumbel_c.py` gate) | 2/15, 1/35 | stops at depth like the fresh control (64 nodes) |
| walker | KR vs K | `DRAW`, unexpanded | **`DRAW`** | **1/14** | runs to the 30 s cap (32,816 nodes); the fresh control stops at depth after 1,984 |
| walker | KQ vs KR | `DRAW`, unexpanded | **`DRAW`** | 1/7 | runs to the cap (the fresh control does too) |
| PUCT pool | after 1.Nf3 Nf6: Ne5 Ng8 Nf3 | `DRAW`, unexpanded | **`DRAW`** | **1/20** | |
| PUCT pool | KR vs K | `DRAW`, unexpanded | **`DRAW`** | 1/14 | runs to the cap (46,208 nodes); the fresh control stops at 30,272 |

- The reviewer's script (`go depth` stopping after 64 nodes) hand-marked an
  *expanded* grandchild. On main the natural mark sits on an unexpanded leaf, so the
  classic path discards it.
- The PUCT pool and walker paths keep it, and the root then reports a proven draw
  that no child supports.
- With a random net `score cp 0` is uninformative (the controls also print 0), so this
  repro does not show the score claim. It does show the status.
- PR #1418 changes the `go depth` stop. It does not stop `advance_root` from carrying
  the mark.

**(ii) Self-play reuse** (`<bank>/probe_ii_selfplay_reuse.py`, rows in
`<bank>/probe_ii_selfplay_reuse.jsonl`, summary in `<bank>/probe_ii_summary.out`).

The probe calls `run_gumbel_root_many_c` the way `network_turn.py` does: a persistent
tree, and `root_node_ids` set to the child of the banked move. It ran over the 16
cycle-12 games with the most twofold plies (8 per arm), 2,179 searched roots, 64 sims.

- 2,163 roots were offered a carried root, and 1,324 of them were reused.
- 91 offered roots carried `DRAW`. All 91 were unexpanded twofold leaves, **all 91 were
  rejected** by the `is_expanded` gate, and a fresh root was built.
- **No reused root ever had a nonzero solved status** (0 of 1,324).
- Self-play does store exact zeros from the twofold rule:
  - 37 rows have value exactly 0.0. 33 of them are survivors that were a twofold child
    of game history.
  - 4 rows are roots proven `DRAW` within the same search, because each of their 2–3
    legal moves repeats a game-history position. These get a **one-hot policy and value
    0**. lc0 would not mark any of those children, since the earlier occurrence is
    before the root.
- 10.9% of probed roots had at least one root child marked `DRAW` against game history.
  This sample was selected for repetition; the population rate from the census is
  3.93%.

**(iii) A twofold hides a mate in 1** (`<bank>/repro_iii_twofold_hides_mate.py`, output
`<bank>/repro_iii_twofold_hides_mate.out`).

- The repeated position is P = `6k1/r4ppp/8/8/8/8/2P2PPP/4R1K1 w`. White is a pawn up
  and has Re8#.
- The history is P, 1.Re2 Ra8 2.Re1, giving root `r5k1/5ppp/8/8/8/8/2P2PPP/4R1K1 b`.
- Black's Ra7?? returns to P and is mated. Every other move survives.
- The evaluator uses tiny-net priors, and its value is "the side with more pawns wins".
  That puts black in the defender's regime: every line that does not repeat evaluates
  below 0.

| root | chose Ra7?? | Ra7 node | stored target on Ra7 | stored value |
|---|---|---|---|---|
| with game history | **5/8 seeds** | solved `DRAW`, never expanded, 48 visits | **0.999** | **0.000** |
| FEN only (control) | 0/8 | expanded, normal child | ≤0.09 | −1.000 |

- In the other 3 seeds Ra7 was not among the 16 Gumbel candidates.
- With a purely random value, or the "optimistic" value, the repeat is never preferred
  (0/8 in both arms). The defect needs the defender's regime, which is exactly when a
  repeat looks like salvation.
- lc0 would search P here, because the earlier occurrence is before the root.

**Reported today and not re-measured here:** in the PR #1410 draw-pruning work, 26 of
the 58 changed self-play moves went to a twofold-repeat child scored as an exact draw.

## 4. Options (prospective; none is implemented)

Every option is weighed against these rules:

- `CLAUDE.md` "Experiments": record, hypothesis, control, one yardstick, pre-committed
  threshold, and recovery plan.
- The search config must be frozen while `wdl_regret` is read as a net signal. Any
  change to search semantics injects a step into PID-driven numbers.
- A Python-only `GumbelConfig` field is dropped by the C path. A new knob must be
  plumbed into C, and its take-effect proof must log from the C consumer.

`$SL_BASE` below is the A39 checkpoint used by the banked runs: model-state sha256
`9796079c…`, actor sha256 `a39f900c…`, step 680,276. Pin it by path and step at launch.

### Option C: engine-side, no change to self-play semantics (smallest)

**Change.** `advance_root` (or `_drop_pool_root_if_support_differs`) refuses to reuse a
root that is unexpanded, or that has a nonzero `solved` status with an unproven child.
It rebuilds a fresh root, exactly as `gumbel_c.py:1397-1413` already does for self-play
and the classic UCI path. Optionally, `compact_to_subtree` clears the new root's
`solved` unless its children prove it.

- **Self-play footprint:** none. Self-play never takes this path, as shown in (ii): 0 of
  1,324 reused roots carried a status.
- **Identity test:** `tests/test_e2e_smoke.py -k gumbel_selfplay_smoke` produces
  byte-identical shards and targets before and after, with a fixed seed. All
  `tests/test_mcts_*` and `tests/test_gumbel_*` pass unchanged.
- **Audit-first:** not applicable; it is not a training target.
- **Deciding yardstick:** `python -m pytest tests/test_uci_reroot_twofold.py`. This is a
  new test that ports repro (i) to the walker and pool paths and asserts that after
  `advance_root` onto a natural twofold leaf the root status is 0 and `go depth 4` stops
  like the fresh control. **Threshold:** 0 stale roots across the three cases on both
  paths. The test must first fail on the parent commit; that is the mutant.
- **Cost:** a few hours. **Confounds:** none. **Recovery:** revert the PR.

### Option A: keep `Q = 0` for twofold, but never as a proof

**Change.** The twofold leaf stays a terminal with value 0 but is written as
`SOLVED_UNKNOWN`, or as a separate non-proof flag. It is excluded from
`tree_resolve_from_children` and from the solved-root stop, and it never survives a
reroot.

- **Self-play footprint:** roots that today become proven `DRAW` from twofold children
  stop getting a one-hot policy and exact 0. In the probe that was 4 of 2,179 roots,
  0.18% of a repetition-heavy sample. Deeper in the tree, more ancestors stay unproven.
  The population rate is unmeasured, with an upper bound of 3.93% of roots (roots that
  have any repeat child).
- **Does not fix (iii).** The leaf is still terminal, so the mate is still invisible.
- **Identity test:** on a position suite whose trees contain no twofold terminal
  (assert a C counter of twofold hits is 0), targets and moves are bit-identical.
- **Audit-first and yardstick:** as for B.
- A is dominated by B. Use it only if B's reroot bookkeeping proves too invasive.

### Option B: lc0-style (recommended experiment)

**Change.** A first repeat is a terminal draw only if the earlier occurrence lies **at
or after the root on the current path**. Implement this by storing the root's
`hash_stack_len` boundary and matching only indices at or above it. A repeat of a
pre-root position is searched normally; threefold still counts pre-root occurrences.
The twofold draw is never a proof (A's rule). On reroot, twofold leaves whose earlier
occurrence is now before the new root are cleared and their visits reverted, as lc0's
`EnsureNodeTwoFoldCorrectForDepth` does. The simpler alternative is to refuse reuse when
the kept subtree holds any.

**Self-play footprint:**

- Direct: the 3.93% of searched roots with a child repeating a game-history position.
  Those children become ordinary searched children.
- Played moves into a repeat: 1.11% of plies are candidates to change.
- Indirect: deeper in-tree nodes that repeat a pre-root position. Unmeasured; the
  repetition-selected probe saw 10.9% at the root level.
- Game level: 31% of games contain a twofold, and 18% end by threefold claim. Both may
  shift.
- `tests/test_mcts_root_history_lifecycle.py:99-109` pins the current behaviour and
  would have to change, with the reason stated.

**Identity test.** A C counter of "twofold hits whose match index is below the root
boundary". On a fixed seed and position suite, every root where the counter is 0 must
produce a bit-identical policy, value and move. Only rows with a nonzero counter may
differ.

**Audit-first** (before any training compute): take the 19,554 banked roots that have a
game-history-repeat child, or a stratified 2,000 of them, frozen by `game_id` and ply.

- Run the current and the candidate search with `$SL_BASE` in a pause window.
- Score both chosen moves against Syzygy (≤6 men) and deep SF (MultiPV, the frozen
  deep-SF audit protocol).
- Bank per-move scores and PVs keyed by position, arm and engine settings, plus
  `game_id`.
- **Proceed** only if the candidate's mean regret against deep SF is not worse
  (one-sided 95% bound ≤ +2 cp). Otherwise stop: there is no gain to train for.

**Deciding yardstick (play side, one command):**

```bash
PYTHONPATH=. python3 scripts/arena_standard.py --candidate $SL_BASE --reference $SL_BASE \
  --mode matched_sims --sims 400 --cand-gumbel twofold=in_tree --games 1000 \
  --syzygy <production syzygy pair> --pgn-out <out>.pgn \
  --sprt 'elo0=-10,elo1=5,alpha=0.05,beta=0.05' --label twofold-in-tree-vs-current
```

- `twofold` is a new per-side knob. It must be plumbed into C, added to the
  `--cand-gumbel` allow-list, and proven to take effect from the C consumer's own
  counter.
- **Pre-committed rule:** adopt on SPRT H1 or INCONCLUSIVE with the 95% lower bound
  ≥ −10 Elo. **Kill** on H0.
- This decides the engine. A self-play training arm is a separate record, judged by
  `scripts/arena_standard.py` on the trained checkpoint against `$SL_BASE`.

**Other terms:**

- **Cost:** C work on the terminal test, the reroot bookkeeping and the ABI bump
  (about 1–2 days), plus a 1,000-game arena in a pause window.
- **Confounds:** one data-affecting change per readout window, so do not overlap the
  draw-pruning (#1410) or claim-rule changes. It changes `wdl_regret` as a search-config
  change.
- **Recovery:** the knob defaults to the current behaviour, and a revert to
  `twofold=any_prior` restores it bit-identically.

### Ranking and recommendation

1. **C now.** It is a correctness bug in the engine with zero self-play footprint, and
   it merges on identity.
2. **B as the registered experiment.** It is the only option that fixes (iii) and
   matches lc0's convention. Gate it audit-first, then use the arena command above.
3. **A** only as a fallback, if B's reroot bookkeeping is rejected.

**No code change is made until the owner chooses.**

## 5. Same-coloured bishops: insufficient material

- **C rule:** with no pawns, rooks or queens, the C board calls material insufficient
  for K vs K, K+minor vs K, or **exactly** K+B vs K+B on the same colour, i.e. 4 men
  (`encoding/_cboard_impl.h:1394-1408`).
- **python-chess rule:** material is insufficient whenever each side's only non-king
  pieces are bishops, all on one square colour (and no knights).
- **Check** (`<bank>/check_bishops.py`, `.out`): after
  `4b3/8/8/8/6bk/1B6/2r5/2K5 w - - 97 1` Bxc2, python-chess reports a draw by
  insufficient material and the C board reports the game not over.
- **Banked footprint:** 0 positions in 4,608 games, all plies including the opening.
- **Why the footprint is zero:** the banked game end uses python-chess (`claim_draw=True`),
  and positions with 6 men or fewer are TB-adjudicated. A mismatch needs 7 or more men
  made only of kings and same-coloured bishops.
- **Where it can still matter:** in-search terminal detection, for nodes without a TB
  probe, which requires promotions to bishops; and the C `is_game_over` used for forced
  moves (`network_turn.py:464`).
- **Proposed fix:** mirror python-chess's per-colour rule in
  `cboard_insufficient_material`, with a differential test against python-chess on
  random and constructed positions.
- **Self-play footprint:** expected 0 rows in the banked distribution. The identity
  test is the census script, which must report 0 changed positions, plus the
  self-play smoke test producing byte-identical output.
- This fix can ship with option C.

## What this record could not determine

- The population rate of deeper in-tree nodes that repeat a pre-root position (option B's
  indirect footprint). That needs instrumented searches with the real net on GPU.
- Whether production-net self-play prefers twofold children more or less often than the
  random-net probe. The probe's 37 exact-zero rows are a mechanism demonstration, not a
  rate.
- Whether the cycle-over-cycle growth in shuffling is caused by the twofold rule.
- The banked collection's `source_revision` (`e633212d…`) is not in this repository's
  history. Its outcome rule is inferred from the matching `rule50_match_v1` mode in
  `selfplay/bt4_outcome.py` and from the census itself (838 claimable-threefold endings,
  0 third occurrences).
- That lc0 behaves identically in the shipped binary under `local_engines/`. Only the
  source was read.
