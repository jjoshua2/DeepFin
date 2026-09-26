# BT4 generation outcome policy — September 22, 2026

The BT4 actor needs a result for every replay row. The current generation-zero
generator maps a game stopped at `max_plies` from `"*"` to a draw target. That
fallback is unsuitable for a BT4-labelled corpus: an unfinished game has no
known winner. This record covers a standalone outcome helper and small CPU
tests. It does not wire the helper into generation, change a default, or admit
any real games.

`preflight_six_man_tablebases(path)` is intended to run once per worker. It
requires two distinct resolved directories with tablebase files, opens each
component, and checks that the combined handle contains WDL and DTZ tables.
Directory checks cannot establish that
every six-piece material position is covered. Each missing per-position probe
therefore remains an explicit discard decision.

Before each BT4 root evaluation, `decide_bt4_outcome(board, plies, max_plies,
syzygy_path)` returns `None` to keep playing or a decision with `result`,
`termination`, and `detail`:

1. `board.outcome(claim_draw=True)` handles natural and claimable endings first,
   matching the SF-rooted corpus generator. This includes a possible 50-move
   claim at clock 99 and a claim for an impending threefold repetition.
2. At six pieces or fewer, the shared `tb_adjudicate_result` gives the
   adjudicated result. Castling rights or missing material coverage instead
   return `result=None`; the future caller must discard the whole game and
   count its attempted rows and reason. This check precedes the ply cap.
3. Above six pieces, reaching `max_plies` returns `result=None` with
   `max_plies_unresolved`. The future caller must not pass that result to
   `_result_to_wdl`, append its rows, or count it as a draw.

The Syzygy outcome uses the repository's existing **theoretical training-label
convention**: raw WDL `+2` and cursed `+1` are a win for the side to move; `0`
is a draw; blessed `-1` and `-2` are a loss. This preserves the SF-rooted
generator's `tb_adjudicate_result` mapping. It is not a claim-aware outcome at
the board's current halfmove clock. [python-chess documents](https://python-chess.readthedocs.io/en/latest/syzygy.html)
that WDL assumes the position was reached just after a capture or pawn move;
DTZ and the current clock would be needed for a different rule-aware policy.
Natural claims still take priority before the tablebase probe.

The helper is deliberately unwired. A later opt-in BT4 generator integration
must call preflight once per worker, call the decision before root inference,
drop `result=None` games before row conversion and shard tallying, and publish
separate counts for missing coverage and capped unresolved games. Existing
generation-zero behavior remains unchanged until that integration is reviewed.

The 18 fake-tablebase CPU cases cover both colors, all five raw WDL categories,
clock-99/100 claims, impending threefold and checkmate priority, missing
material and castling coverage, cap precedence, and pair preflight. They passed
on two capped CPU cores with CUDA hidden. Scoped Ruff, Basedpyright, and Vulture
passed. No real generation, full CPU stream, or GPU run was started.
