"""Stockfish annotation + curriculum-opponent move selection.

* ``_eff_sf_nodes`` — adaptive per-slot SF node budget (scales down when
  the previous net decision used fast sims).
* ``submit_sf_queries`` — dispatch async queries to ``StockfishPool``
  (no-op for synchronous stockfish).
* ``finish_sf_annotation_and_moves`` — resolve futures (or run sync),
  then ``_process_sf_results`` per slot: build the softmax SF policy
  target, attach (with per-head legal mask) to the last ``_NetRecord``
  for that slot, and play the curriculum opponent's move for non-
  selfplay games.

All entry points expect slot-index iterables disjoint from the network-
turn indices — the driver in ``manager.play_batch`` enforces this
partitioning via ``classify_active_slots``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.selfplay.state import SelfplayState
from chess_anti_engine.stockfish.pool import StockfishPool


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over a 1D numpy array."""
    z = x.astype(np.float64, copy=False)
    z = z - float(np.max(z))
    e = np.exp(z)
    s = float(e.sum())
    if s <= 0:
        return np.full_like(z, 1.0 / float(z.size))
    return e / s


def _flip_wdl_pov(wdl: np.ndarray) -> np.ndarray:
    """Flip a ``[W, D, L]`` vector to the opposite POV (swaps W and L)."""
    wdl = np.asarray(wdl, dtype=np.float32)
    if wdl.shape != (3,):
        return wdl.astype(np.float32, copy=False)
    return np.array(
        [float(wdl[2]), float(wdl[1]), float(wdl[0])], dtype=np.float32,
    )


def _choose_curriculum_opponent_move(
    *,
    rng: np.random.Generator,
    legal_indices: np.ndarray,
    cand_indices: list[int],
    cand_scores: list[float],
    regret_limit: float,
) -> int:
    """Pick a curriculum-opponent move index from SF's candidate list.

    * Empty candidate list -> uniform random legal move.
    * ``regret_limit == inf`` -> take SF's top choice verbatim (used by
      eval/gate matches where we want full-strength SF).
    * Otherwise -> uniform random among moves within ``regret_limit``
      score of the best candidate.
    """
    if not cand_indices:
        return int(legal_indices[int(rng.integers(len(legal_indices)))])

    if not math.isfinite(float(regret_limit)):
        # MultiPV lists PVs in rank order so cand_indices[0] is SF's best.
        return cand_indices[0]

    best_score = max(float(s) for s in cand_scores)
    acceptable = [
        idx
        for idx, score in zip(cand_indices, cand_scores, strict=False)
        if (best_score - float(score)) <= float(regret_limit) + 1e-12
    ]
    if not acceptable:
        acceptable = [cand_indices[0]]
    return acceptable[int(rng.integers(len(acceptable)))]


def _eff_sf_nodes(state: SelfplayState, idx: int) -> int | None:
    """Return the per-slot SF node budget, scaled down after fast-sim plies."""
    if state.base_nodes <= 0:
        return None
    fast_scale = 1.0 if bool(state.last_net_full[idx]) else 0.25
    return max(1, int(round(float(state.base_nodes) * float(fast_scale))))


def submit_sf_queries(
    state: SelfplayState, idxs: list[int],
) -> dict[int, Any]:
    """Submit SF queries to the pool without blocking; return futures dict.

    Only valid when ``state.stockfish`` is a ``StockfishPool``.  The
    caller guards with ``isinstance`` (same pattern as the original
    nested closure).
    """
    assert isinstance(state.stockfish, StockfishPool)
    return {
        idx: state.stockfish.submit(
            state.cboards[idx].fen(), nodes=_eff_sf_nodes(state, idx),
        )
        for idx in idxs
    }


def finish_sf_annotation_and_moves(
    state: SelfplayState,
    idxs: list[int],
    *,
    play_curriculum_moves: bool,
    futures: dict[int, Any] | None = None,
) -> None:
    """Collect SF results (from futures or synchronously), then process."""
    if not idxs:
        return
    if futures is not None:
        results = {idx: futures[idx].result() for idx in idxs if idx in futures}
    elif isinstance(state.stockfish, StockfishPool):
        futs = {
            idx: state.stockfish.submit(
                state.cboards[idx].fen(), nodes=_eff_sf_nodes(state, idx),
            )
            for idx in idxs
        }
        results = {idx: fut.result() for idx, fut in futs.items()}
    else:
        results = {
            idx: state.stockfish.search(
                state.cboards[idx].fen(), nodes=_eff_sf_nodes(state, idx),
            )
            for idx in idxs
        }
    _process_sf_results(
        state, idxs, results=results, play_curriculum_moves=play_curriculum_moves,
    )


def _process_sf_results(
    state: SelfplayState,
    idxs: list[int],
    *,
    results: dict,
    play_curriculum_moves: bool,
) -> None:
    """Attach SF policy target (+legal mask) to the last _NetRecord per slot
    and, for curriculum games, push the SF-chosen move onto the board."""
    if not idxs:
        return

    sf_policy_temp_local = float(state.game.sf_policy_temp)
    sf_policy_label_smooth_local = float(state.game.sf_policy_label_smooth)

    regret_limit = (
        float(state.opponent.wdl_regret_limit)
        if state.opponent.wdl_regret_limit is not None
        else float("inf")
    )

    for idx in idxs:
        res = results[idx]
        legal_indices = state.cboards[idx].legal_move_indices()
        if legal_indices.size == 0:
            state.done_arr[idx] = 1
            continue

        _turn = bool(state.cboards[idx].turn)
        legal_set = {int(x) for x in legal_indices}

        a_idx = uci_to_policy_index(res.bestmove_uci, _turn)
        if a_idx < 0 or a_idx not in legal_set:
            a_idx = int(legal_indices[0])

        cand_idxs: list[int] = []
        cand_scores: list[float] = []
        if getattr(res, "pvs", None):
            for pv in res.pvs:
                if pv.wdl is None:
                    continue
                a = uci_to_policy_index(pv.move_uci, _turn)
                if a < 0 or a not in legal_set:
                    continue
                w_sf, d_sf = float(pv.wdl[0]), float(pv.wdl[1])
                cand_idxs.append(a)
                cand_scores.append(w_sf + 0.5 * d_sf)

        if not cand_idxs:
            cand_idxs = [a_idx]
            cand_scores = [0.0]

        scores = np.array(cand_scores, dtype=np.float64) / max(
            1e-6, sf_policy_temp_local,
        )
        p_top = _softmax_np(scores).astype(np.float32, copy=False)

        p_sf = np.zeros((POLICY_SIZE,), dtype=np.float32)
        for a, p in zip(cand_idxs, p_top, strict=False):
            p_sf[int(a)] += float(p)

        if sf_policy_label_smooth_local > 0.0:
            n_legal = legal_indices.size
            if n_legal > 0:
                p_sf *= 1.0 - sf_policy_label_smooth_local
                p_sf[legal_indices] += sf_policy_label_smooth_local / float(n_legal)

        ps = float(p_sf.sum())
        if ps > 0:
            p_sf /= ps

        if state.samples_per_game[idx]:
            rec = state.samples_per_game[idx][-1]
            if rec.sf_policy_target is None and rec.sf_move_index is None:
                rec.sf_policy_target = p_sf
                rec.sf_move_index = a_idx
                if res.wdl is not None:
                    rec.sf_wdl = _flip_wdl_pov(res.wdl)
                _sf_mask = np.zeros((POLICY_SIZE,), dtype=np.uint8)
                _sf_mask[legal_indices] = 1
                rec.sf_legal_mask = _sf_mask

        if not play_curriculum_moves or state.selfplay_arr[idx]:
            continue

        _opp_move_idx = _choose_curriculum_opponent_move(
            rng=state.rng,
            legal_indices=legal_indices,
            cand_indices=cand_idxs,
            cand_scores=cand_scores,
            regret_limit=regret_limit,
        )

        state.cboards[idx].push_index(_opp_move_idx)
        state.move_idx_history[idx].append(_opp_move_idx)
        # Advance tree root through opponent's move for next-ply reuse.
        if state.mcts_tree is not None and state.root_ids[idx] >= 0:
            state.root_ids[idx] = state.mcts_tree.find_child(
                state.root_ids[idx], _opp_move_idx,
            )
        if state.cboards[idx].is_game_over():
            state.done_arr[idx] = 1


__all__ = [
    "finish_sf_annotation_and_moves",
    "submit_sf_queries",
]
