# Winning-root draw pruning: no fake forced moves, and a below-zero draw fallback

**Status:** search-correctness change with a measured self-play footprint. No
training compute is spent and none is proposed (reasoning in
[Decision](#decision)). Code: PR #1410 (`chess_anti_engine/mcts/gumbel_c.py`,
`mcts/gumbel.py`, `mcts/root_tactics.py`, `uci/root_parallel_gumbel.py`,
`uci/search.py`). No C source or ABI change.

## What the search did before

When the network's root value is positive, every Gumbel root (C runner, Python
reference, root-parallel UCI pool) removes the moves that draw at once from the
search: stalemate, insufficient material, 50-move clock at 100, a third
occurrence, and also moves after which the opponent could claim a 50-move or
threefold draw on its reply. Two shapes lost a drawn position through it:

1. **Pruned to one move.** If exactly one move survived, the runner's one-legal-
   move shortcut returned it with zero simulations and a one-hot policy, even
   when the pruned move was better. `4K2k/3q1P2/6Q1/8/8/8/8/8 w`: Kf8 is mated by
   Qd8, Kxd7 stalemates, and the classic search played Kf8 unsearched.
2. **Every survivor refuted.** If several moves survived and every one searched
   below 0, the search still played one of them. `8/q7/1P6/8/8/3Q4/8/K1k5 w`:
   everything but bxa7 (stalemate) is mated next move. Only the UCI engine had a
   rescue, and only after a full proof (PR #1406).

## The change

1. Pruning happens only while **at least two moves survive**. Otherwise the
   draws stay in the search, so the root is searched exactly as a root with
   root value at or below 0 always is (drawing moves are terminal children
   worth exactly 0). Same rule in the C runner, the Python reference and the
   root-parallel UCI pool.
2. **Below-zero fallback** (C runner and Python reference). When pruning did
   happen and removed at least one move that ends the game drawn on the board
   (not a mere reply-claim), and every move the search visited comes back
   below 0, the runner plays that draw: one-hot policy target, value 0, the
   lowest-index draw if several. The UCI classic path honours this choice
   (`_drawing_escape_action`), with `score cp 0`.

Rejected alternative: never pruning (draws always in the search). Pruning is
what keeps a high-prior stalemate out of a winning root's played move and
stored target (`test_gumbel_c_avoids_high_prior_stalemate_when_root_is_winning`),
and removing it would change the 3,858 draw-move roots per 497,027 plies
measured below instead of the few hundred this change touches.

## Measured self-play footprint

Banked self-play, read-only: all 24 `fresh-current-policy-A39-{F12,Q50}-cycle-*-collect-results-v1`
directories (both twelve-cycle loops) under `operations/rl-week-one-20261112-v1/`
in the artifact store, 4,608
games, **497,027 searched plies**, positions rebuilt from each game's move list
(history included, so repetition draws are exact) and scanned with
`CBoard.root_terminal_actions`, the runner's own draw scan.

| Root class | Plies | Per 10,000 searched plies |
|---|---:|---:|
| At least one immediate-draw move (prunable set, before the root-value test) | 3,858 | 77.6 |
| ... of which one ends the game drawn on the board | 315 | 6.3 |
| Pruning would leave exactly one move | 256 | 5.2 |
| ... and the root value is positive, so the old code returned it unsearched | 156 | **3.1** |
| ... and the played move changes (A/B below) | 58 | **1.2** |
| Several survivors and an on-board draw (fallback candidates) | 295 | 5.9 |
| ... where the fallback fires (A/B below) | 1 | **0.02** |

The "root value positive" row is recomputed with the actual self-play net
(checkpoint sha256 `a39f900c…`, CPU fp32): 156 roots. The banked data agrees: 157
of the 256 stored a one-hot policy target with a positive stored root value,
the signature of the zero-simulation shortcut.

**Before/after A/B on the same roots.** Each root was searched twice with the
same seed, the production self-play search shape (64 simulations, top-k 16,
`c_scale` 0.1, Gumbel noise scale 0.75, `vloss_weight` 1) and the A39 net on CPU:
once with `origin/main`'s `gumbel_c.py`, once with this branch. Tablebases were
not attached; every root in the singleton set has more than six pieces.

- **Singleton roots (256).** The 100 with a non-positive root value: bit-identical
  move, value and stored target. The 156 with a positive root value: now
  searched (64 visits instead of 0), so the stored target changes from one-hot to
  the searched distribution in all 156. The search's chosen move (the survivor at
  temperature 0; self-play then samples the stored target at its temperature,
  0.1 in these runs, so its realized count can differ slightly) changes in
  **58** (56 games, 1.2% of games), always to a move after which the opponent can claim a
  repetition or 50-move draw; never to a stalemate-type draw. In those 58 the
  old root value was small (median +0.014) and the searched value of the new
  move about 0. How the real games ended, from the mover's side: 34 draws, **23
  losses**, 1 win; the change sits in the last few plies (median 5.5 plies left).
- **Several survivors with an on-board draw (295, 293 with a positive root
  value).** 294 bit-identical. The fallback fired once:
  `3Q4/4q2p/7k/5p1P/5P1K/4P3/8/8 w` (Black's queen checks; Qxe7 stalemates),
  where the old search played Kh3 at a searched value of -0.94 and the new one
  plays Qxe7 (value 0). That game did end drawn on its next ply.
- **Control: roots with no immediate-draw move** (125 roots from a uniform random
  sample of the same plies, 63 of them with a positive root value). **125 of 125
  bit-identical** (move, value and stored target).

Every other root takes the old code path unchanged: the new code runs only when
the root value is positive, a draw-move was found, and either one move would
survive or an on-board draw was pruned. The repository's identity pins
(`tests/test_uci_search_identity_pin.py`, the Gumbel C/Python parity and
self-play tests, `tests/test_e2e_smoke.py -k gumbel_selfplay_smoke`) pass
unchanged.

Banked beside the evidence: `~/chess-artifacts/operations/rl-claude-20261119/root-draw-pruning/`
holds the scan and A/B scripts (`scan.py`, `ab.py`, `summ.py`), the case lists
keyed by game directory and ply, every A/B row (`ab_*.log`: FEN, game, ply,
move, value, visits and support before and after), the exact baseline
`gumbel_c.py` and the md5 of the branch files the A/B imported.

## Decision

No training-strength experiment. The per-ply footprint is 3.1 stored targets
and 1.2 played moves per 10,000 searched plies, concentrated in the final plies
of near-drawn endgames, and the change makes the search use its own values
where it previously skipped them: a root that is not forced is searched, and a
draw is no longer discarded for moves the search scores below it. The game
outcomes of the affected plies (23 of 58 were losses for the side whose move
changes) point the same way. The change is search-path only and touches no
loss, target definition or training knob.

Confounds for any readout window that spans the deployment of this change:
self-play draw rate can rise slightly (the 58 changed moves are draw-seeking),
and the stored policy target changes on about 3 plies per 10,000.

## Addendum 2026-11-19: the reply-forced fifty-move draw (PR #1418)

Added rule (outside-model review lead R1): `forced_draw_actions` and
`forced_draw_actions_cboard` (`mcts/root_tactics.py`, one definition used by
the C runner's and the Python reference's below-zero fallback, root-parallel
Gumbel's root resolver and the UCI drawing escape) also count a move that
leaves the fifty-move clock at 99 when every opponent reply ends the game drawn
and none mates, i.e. an opponent with only reversible moves cannot decline the
fifty-move draw. Before it, such a move was pruned from a winning root as a
reply-claim and then counted as no draw, so a root whose searched moves all
lose was played as lost (e.g. `2K5/8/1k6/1q6/8/8/P7/8 w - - 98 1`: Kd8 draws,
a2a3 / a2a4 are mated next move).

Scope: the rule is examined only when a move leaves the clock at exactly 99,
and only adds moves to the pruned-draw set. Every other root is unchanged.

Evidence (`tests/test_root_reply_forced_draw.py`):

- On ~4,000 seeded random roots (two seeds, clocks 0-100, repetition-bearing
  histories) with no reply-forced move, both helpers return exactly what the
  pre-rule helpers returned.
- The C runner (board and CBoard entries) and the Python reference give
  bit-identical policy targets, moves and values with the old and new helper
  on 7 roots x 2 leaf evaluators, including the pruned-stalemate fallback root.
- A seeded port of the review's differential test against a python-chess
  oracle (100-ply adjudication, checkmate first): no false draw on any of
  >20,000 moves, and no miss where a move leaves the clock at 99 for the
  python helper. The CBoard twin can still miss there when a reply ends in the
  same-coloured-bishops dead position it does not recognise (below); it then
  keeps a move it could have pruned, the cautious side. The review's
  own unbounded run found 0 false draws in ~360k moves.
- `tests/test_uci_search_identity_pin.py`, the Gumbel parity tests and
  `tests/test_e2e_smoke.py -k gumbel_selfplay_smoke` pass unchanged.

Not covered by design: a move after which every reply draws some other way
(for example the only reply captures into insufficient material) at a clock
other than 99. Found while porting the test, and older than this rule: CBoard
does not treat same-coloured bishops as a dead position where python-chess
does (`4b3/8/8/8/6bk/1B6/2r5/2K5 w - - 97 1`, Bxc2), so the two helpers' base
sets can differ there.
