"""Syzygy probe wrapper for the UCI MCTS search.

Runs on the batch-of-leaves boundary between ``start/continue_gumbel_sims``
and the caller's GPU eval: for each leaf with few enough pieces and no
castling rights (via :func:`is_tb_eligible`), we probe the TB and overwrite
the corresponding row of the wdl logits array with a saturated decisive
distribution, so MCTS backprop uses the TB truth instead of the NN estimate.

Policy logits are left untouched — MCTS still needs NN priors for expansion,
and the TB Q dominates via value backup anyway.

Cursed-win / blessed-loss (±1) are treated as *draws* here, not wins/losses.
That diverges from the training-label version in
:mod:`chess_anti_engine.tablebase` because UCI plays under the 50-move rule,
where a cursed win really does draw. Training labels the theoretical result
regardless — that's the intended divergence, not a bug.
"""
from __future__ import annotations

import logging

import chess
import chess.syzygy
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.tablebase import get_tablebase


_log = logging.getLogger(__name__)


# Large-magnitude logits so softmax collapses to one-hot. 12.0 against zeros
# gives >0.999999 on the target bin — well inside what MCTS needs to pin Q.
_WIN_LOGITS = np.array([12.0, -12.0, -12.0], dtype=np.float32)
_LOSS_LOGITS = np.array([-12.0, -12.0, 12.0], dtype=np.float32)
_DRAW_LOGITS = np.array([-12.0, 12.0, -12.0], dtype=np.float32)


class SyzygyProbe:
    """Overrides NN wdl logits with TB truth for TB-eligible leaves.

    Holds a ``syzygy_path`` string rather than an opened Tablebase; the
    actual handle is fetched (and cached) via
    :func:`chess_anti_engine.tablebase.get_tablebase`, so training and UCI
    share one open instance per path.
    """

    def __init__(self, syzygy_path: str, max_pieces: int | None = None) -> None:
        """``max_pieces`` caps the filter piece-count; if None (the default),
        we detect it from the opened tables' material keys, so a user with
        only 3–5-man files gets 5 and we don't waste probes on 6–7-piece
        leaves that would miss. Pass an explicit value to clamp lower."""
        self._path = syzygy_path
        tb = get_tablebase(syzygy_path)
        # Tablebase keys are material strings like "KQvK" / "KRPvKP". Piece
        # letters are uppercase; 'v' is lowercase — count uppercase only.
        if tb is not None:
            self.n_wdl = len(tb.wdl)
            self.n_dtz = len(tb.dtz)
            available = max(
                (sum(c.isupper() for c in k) for k in tb.wdl.keys()),
                default=0,
            )
        else:
            self.n_wdl = 0
            self.n_dtz = 0
            available = 0
        self.max_pieces = (
            min(int(max_pieces), available) if max_pieces is not None else available
        )
        self.hits = 0
        self.probes = 0

    def reset_counts(self) -> None:
        self.hits = 0
        self.probes = 0

    def apply(
        self,
        leaf_cboards: list[CBoard],
        wdl: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> int:
        """Probe each leaf and overwrite its wdl row on a TB hit.

        When ``indices`` is provided, ``leaf_cboards`` are assumed to already
        satisfy ``is_tb_eligible`` (the C-side ``get_pending_tb_leaves`` path)
        and ``indices[j]`` is the row of ``wdl`` to write. When ``indices`` is
        None, each leaf is checked here — the root-probe path, where leaves
        are not pre-filtered.
        """
        if not leaf_cboards:
            return 0
        tb = get_tablebase(self._path)
        if tb is None:
            return 0
        hits = 0
        max_pieces = self.max_pieces
        for j, cb in enumerate(leaf_cboards):
            if indices is None:
                occ = int(cb.occ_white) | int(cb.occ_black)
                if occ.bit_count() > max_pieces or int(cb.castling) != 0:
                    continue
                i = j
            else:
                i = int(indices[j])
            self.probes += 1
            try:
                board = chess.Board(cb.fen())
                wdl_val = tb.probe_wdl(board)
            except (KeyError, chess.syzygy.MissingTableError, IndexError):
                continue
            except Exception as exc:
                _log.debug("syzygy probe failed on %s: %r", cb.fen(), exc)
                continue
            if wdl_val >= 2:
                wdl[i] = _WIN_LOGITS
            elif wdl_val <= -2:
                wdl[i] = _LOSS_LOGITS
            else:
                # Draw, cursed-win (+1), or blessed-loss (-1).
                wdl[i] = _DRAW_LOGITS
            hits += 1
        self.hits += hits
        return hits
