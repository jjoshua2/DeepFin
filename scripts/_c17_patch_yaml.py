"""One-shot: add `gumbel_vloss_weight: 1` to the live yaml. Idempotent.

Kept as a file rather than an inline heredoc so the exact text that lands in the
production config is reviewable before the pause window, and so re-running the
deploy script cannot double-insert.
"""
from __future__ import annotations

import pathlib
import sys

_ANCHOR = "  gumbel_topk: 16\n"

_BLOCK = """  # 2026-07-28 C17 DEPLOY (ledger: "READOUT 2 — SETTLED"). LEGACY virtual loss
  # in the selfplay Gumbel descent. Within one sequential-halving round the tree
  # cannot update, so every visit allocated to a candidate descends to the SAME
  # leaf. At the realized live shape -- n_boards ~= 1, because 23.5 of every 24
  # games per thread are blocked on a pending Stockfish move -- 88.3% of rows at
  # 256 sims are byte-identical duplicates; 65.9% mix-weighted, and 97% of that
  # waste is on full plies.
  #
  # A 256-sim ply spends 256 forwards to visit 30 distinct positions. At
  # vloss=1 it visits 242: 8.1x more distinct search information on exactly the
  # 25% of plies that produce every training target. Measured -7.4 cp on the
  # target (PR #278). Costs ~26% more GPU calls, which are FREE here because
  # selfplay is Stockfish-bound, not GPU-bound (sf_block_starved 2699-2986% of
  # 3200% thread-time).
  #
  # LEGACY (the only mode selfplay can select) values the pending visit as a
  # loss. That pessimism IS the spreading mechanism -- VLOSS_MODE_VIRTUAL_MEAN
  # is theoretically cleaner but cannot fill a batch, so it is deliberately not
  # reachable from here. 0 restores the pre-2026-07-28 behaviour.
  # _RECO_RESTART_KEY: read into SearchConfig once at worker-session start.
  gumbel_vloss_weight: 1
"""


def main() -> int:
    path = pathlib.Path("configs/pbt2_small.yaml")
    text = path.read_text(encoding="utf-8")
    if "gumbel_vloss_weight" in text:
        print("already present — nothing to do")
        return 0
    if text.count(_ANCHOR) != 1:
        print(f"ABORT: anchor {_ANCHOR!r} found {text.count(_ANCHOR)} times")
        return 1
    path.write_text(text.replace(_ANCHOR, _ANCHOR + _BLOCK, 1), encoding="utf-8")
    print("yaml patched: gumbel_vloss_weight = 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
