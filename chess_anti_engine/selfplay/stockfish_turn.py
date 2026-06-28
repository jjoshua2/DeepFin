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

import logging
import math
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass
from typing import Any

import numpy as np

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.selfplay.state import SelfplayState, _NetRecord
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.wdl import cp_to_wdl as _cp_to_wdl
from chess_anti_engine.replay.shard import SF_CP_SENTINEL, SF_MULTIPV_RAW_MAX


from chess_anti_engine.utils.numpy_helpers import softmax_1d as _softmax_np  # noqa: E402


_LOG = logging.getLogger("chess_anti_engine.selfplay")

_multipv_truncation_warned = False


def _warn_multipv_truncated() -> None:
    """Warn once: sf_multipv > SF_MULTIPV_RAW_MAX breaks the rebuild parity contract.

    The live target uses ALL legal MultiPV candidates while the stored sparse
    rows are capped, so train-time rebuilds (train/target_builder.py) would
    diverge from the stored dense targets for these records.
    """
    global _multipv_truncation_warned
    if not _multipv_truncation_warned:
        _multipv_truncation_warned = True
        _LOG.warning(
            "sparse MultiPV capture truncated at %d rows; live targets use all "
            "candidates, so rebuilt SF targets will diverge from stored ones. "
            "Raise SF_MULTIPV_RAW_MAX (replay/shard.py) or lower sf_multipv.",
            SF_MULTIPV_RAW_MAX,
        )


@dataclass
class _PendingSfLabel:
    future: Any
    record: _NetRecord
    turn: bool
    legal_indices: np.ndarray


def flip_wdl_pov(wdl: np.ndarray) -> np.ndarray:
    """Flip a ``[W, D, L]`` vector to the opposite POV (swaps W and L)."""
    wdl = np.asarray(wdl, dtype=np.float32)
    if wdl.shape != (3,):
        return wdl.astype(np.float32, copy=False)
    return np.array(
        [float(wdl[2]), float(wdl[1]), float(wdl[0])], dtype=np.float32,
    )


def _sf_result_wdl_for_record(
    res,
    *,
    sf_wdl_use_cp_logistic: bool,
    sf_wdl_cp_slope: float,
    sf_wdl_cp_draw_width: float,
) -> np.ndarray | None:
    """Return SF WDL in the sampled net-record POV.

    SF label searches are run after the network move has been pushed, both for
    curriculum opponent replies and for async selfplay annotation. Therefore
    Stockfish's WDL is from the opponent side-to-move and must be flipped back
    before attaching it to the previous network-turn record.
    """
    if sf_wdl_use_cp_logistic and (res.cp is not None or res.mate is not None):
        wdl_stm = _cp_to_wdl(
            res.cp,
            res.mate,
            slope=sf_wdl_cp_slope,
            draw_width_cp=sf_wdl_cp_draw_width,
        )
        return flip_wdl_pov(wdl_stm)
    if res.wdl is not None:
        return flip_wdl_pov(res.wdl)
    return None


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


def _sf_played_move_diagnostics(
    played_idx: int,
    cand_idxs: list[int],
    cand_scores: list[float],
) -> tuple[int | None, float | None]:
    if not cand_idxs or not cand_scores:
        return None, None
    played_score = None
    for idx, score in zip(cand_idxs, cand_scores, strict=False):
        if int(idx) == int(played_idx):
            played_score = float(score)
            break
    if played_score is None:
        # Regret is measured only against the sampled MultiPV set; outside-set moves are missing diagnostics.
        return None, None
    best_score = max(float(score) for score in cand_scores)
    rank = 1 + sum(1 for score in cand_scores if float(score) > played_score + 1e-12)
    return int(rank), max(0.0, best_score - played_score)


def _eff_sf_nodes(state: SelfplayState, idx: int, *, for_move: bool = False) -> int | None:
    """Return the per-slot SF node budget, scaled down on fast-sim plies.

    On fast (playout-capped) plies the SF budget is scaled by
    ``game.sf_fast_ply_node_scale`` (full-sim plies always use 1.0). This is an
    intentional compute optimization, NOT a weakened training target: SF labels
    only attach to full plies (``has_policy`` <=> ``is_full`` <=>
    ``last_net_full``; see network_turn.py and the label-reuse guard in
    ``_slot_latest_record_needs_sf_label``), so every label already runs at full
    nodes. The scale only makes the opponent play cheaply on the ~75% of fast
    plies that are not training targets. Default 0.25 (see GameConfig); raise
    toward 1.0 only for a more consistently strong opponent at extra SF cost.
    """
    base_nodes = int(state.base_nodes)
    if for_move:
        move_nodes = int(getattr(state.game, "sf_move_nodes", 0) or 0)
        if move_nodes > 0:
            base_nodes = move_nodes
    if base_nodes <= 0:
        return None
    if bool(state.last_net_full[idx]):
        fast_scale = 1.0
    else:
        fast_scale = float(getattr(state.game, "sf_fast_ply_node_scale", 0.25))
    return max(1, int(round(float(base_nodes) * fast_scale)))


def _sf_syzygy_path_for_slot(state: SelfplayState, idx: int) -> str | None:
    """Pick the low-IO or DTZ-capable Stockfish tablebase path for one root.

    Most SF calls are high-volume curriculum labeling where SSD-only WDL/DTZ is
    enough. For the explicit non-adjudicated play-through tail, switch that SF
    process to the full tablebase path once the root itself is <=6 pieces so
    Stockfish can choose conversion-correct DTZ moves.
    """
    normal_path = state.game.stockfish_syzygy_path or state.game.syzygy_path
    full_path = state.game.syzygy_path
    if not full_path or str(normal_path or "") == str(full_path):
        return normal_path
    cb = state.cboards[idx]
    occ = int(cb.occ_white) | int(cb.occ_black)
    if occ.bit_count() > 6 or int(cb.castling) != 0:
        return normal_path
    playthrough = (not state.game.syzygy_adjudicate) or (not bool(state.tb_adj_roll_arr[idx]))
    return full_path if playthrough else normal_path


def _search_stockfish_sync(
    stockfish: Any,
    fen: str,
    *,
    nodes: int | None,
    syzygy_path: str | None,
) -> Any:
    if syzygy_path is None:
        return stockfish.search(fen, nodes=nodes)
    try:
        return stockfish.search(fen, nodes=nodes, syzygy_path=syzygy_path)
    except TypeError as exc:
        if "syzygy_path" not in str(exc):
            raise
        return stockfish.search(fen, nodes=nodes)


def _slot_latest_record_needs_sf_label(state: SelfplayState, idx: int) -> bool:
    if not state.samples_per_game[idx]:
        return False
    rec = state.samples_per_game[idx][-1]
    if not bool(rec.has_policy):
        return False
    return rec.sf_policy_target is None and rec.sf_move_index is None


def submit_sf_queries(
    state: SelfplayState, idxs: list[int], *, for_move: bool = False,
) -> dict[int, Any]:
    """Submit SF queries to the pool without blocking; return futures dict.

    Only valid when ``state.stockfish`` is a ``StockfishPool``.  The
    caller guards with ``isinstance`` (same pattern as the original
    nested closure).
    """
    assert isinstance(state.stockfish, StockfishPool)
    return {
        idx: state.stockfish.submit(
            state.cboards[idx].fen(),
            nodes=_eff_sf_nodes(state, idx, for_move=for_move),
            syzygy_path=_sf_syzygy_path_for_slot(state, idx),
        )
        for idx in idxs
    }


def submit_async_curriculum_move_queries(state: SelfplayState, idxs: list[int]) -> int:
    """Submit curriculum SF move queries and leave them pending per slot."""
    if not idxs:
        return 0
    assert isinstance(state.stockfish, StockfishPool)
    submitted = 0
    for idx in idxs:
        if idx in state.pending_sf_moves:
            continue
        state.pending_sf_moves[idx] = state.stockfish.submit(
            state.cboards[idx].fen(),
            nodes=_eff_sf_nodes(state, idx, for_move=True),
            syzygy_path=_sf_syzygy_path_for_slot(state, idx),
        )
        submitted += 1
    return submitted


def submit_async_sf_labels_from_curriculum_moves(state: SelfplayState, idxs: list[int]) -> int:
    """Reuse full-strength curriculum move futures as labels when possible."""
    if int(getattr(state.game, "sf_move_nodes", 0) or 0) > 0:
        return submit_async_sf_label_queries(state, idxs)
    submitted = 0
    max_pending = max(1, int(state.batch_size) * 8)
    for idx in idxs:
        if len(state.pending_sf_labels) >= max_pending:
            break
        fut = state.pending_sf_moves.get(idx)
        if fut is None or not _slot_latest_record_needs_sf_label(state, idx):
            continue
        legal_indices = state.cboards[idx].legal_move_indices()
        if legal_indices.size == 0:
            continue
        state.pending_sf_labels.append(
            _PendingSfLabel(
                future=fut,
                record=state.samples_per_game[idx][-1],
                turn=bool(state.cboards[idx].turn),
                legal_indices=np.asarray(legal_indices, dtype=np.int64).copy(),
            ),
        )
        submitted += 1
    return submitted


def finish_pending_curriculum_moves(
    state: SelfplayState, *, block: bool = False,
) -> int:
    """Apply completed pending curriculum moves.

    If ``block`` is true and no move is ready, wait only until the first
    Stockfish move completes. This avoids per-step head-of-line blocking while
    still making progress when all runnable slots are waiting on Stockfish.
    """
    if not state.pending_sf_moves:
        return 0
    if block and not any(fut.done() for fut in state.pending_sf_moves.values()):
        wait(tuple(state.pending_sf_moves.values()), return_when=FIRST_COMPLETED)

    completed: list[tuple[int, Any]] = []
    for idx, fut in list(state.pending_sf_moves.items()):
        if not fut.done():
            continue
        completed.append((idx, fut.result()))
        del state.pending_sf_moves[idx]

    for idx, res in completed:
        if state.finalized_arr[idx] or state.done_arr[idx]:
            continue
        _process_sf_results(
            state,
            [idx],
            results={idx: res},
            play_curriculum_moves=True,
            attach_labels=False,
        )
    return len(completed)


def finish_sf_annotation_and_moves(
    state: SelfplayState,
    idxs: list[int],
    *,
    play_curriculum_moves: bool,
    attach_labels: bool = True,
    for_move: bool = False,
    futures: dict[int, Any] | None = None,
) -> None:
    """Collect SF results (from futures or synchronously), then process."""
    if not idxs:
        return
    if attach_labels and not play_curriculum_moves:
        idxs = [idx for idx in idxs if _slot_latest_record_needs_sf_label(state, idx)]
        if not idxs:
            return
    if futures is not None:
        results = {idx: futures[idx].result() for idx in idxs if idx in futures}
    elif isinstance(state.stockfish, StockfishPool):
        futs = {
            idx: state.stockfish.submit(
                state.cboards[idx].fen(),
                nodes=_eff_sf_nodes(state, idx, for_move=for_move),
                syzygy_path=_sf_syzygy_path_for_slot(state, idx),
            )
            for idx in idxs
        }
        results = {idx: fut.result() for idx, fut in futs.items()}
    else:
        results = {
            idx: _search_stockfish_sync(
                state.stockfish,
                state.cboards[idx].fen(),
                nodes=_eff_sf_nodes(state, idx, for_move=for_move),
                syzygy_path=_sf_syzygy_path_for_slot(state, idx),
            )
            for idx in idxs
        }
    _process_sf_results(
        state, idxs, results=results, play_curriculum_moves=play_curriculum_moves,
        attach_labels=attach_labels,
    )


def _pv_wdl_score(
    pv,
    *,
    sf_wdl_use_cp_logistic: bool,
    sf_wdl_cp_slope: float,
    sf_wdl_cp_draw_width: float,
) -> float | None:
    if sf_wdl_use_cp_logistic and (pv.cp is not None or pv.mate is not None):
        wdl = _cp_to_wdl(
            pv.cp,
            pv.mate,
            slope=sf_wdl_cp_slope,
            draw_width_cp=sf_wdl_cp_draw_width,
        )
    else:
        wdl = pv.wdl
    if wdl is None:
        return None
    w_sf, d_sf = float(wdl[0]), float(wdl[1])
    return w_sf + 0.5 * d_sf


def _collect_sparse_pv_rows(res, *, turn: bool, legal_set: set[int]) -> np.ndarray | None:
    """Raw MultiPV rows (K, 5) int16 in FULL policy space, rank order.

    Same legality filter + ordering as ``_collect_sf_pv_candidates`` so
    train-time target rebuilds (train/target_builder.py) see exactly the
    candidate set the live targets were built from. Columns documented at
    replay/shard.py::SF_MULTIPV_RAW_MAX. Indices are converted to the shard's
    policy encoding in finalize, like ``sf_move_index``.
    """
    rows: list[tuple[int, int, int, int, int]] = []
    truncated = False
    for pv in getattr(res, "pvs", None) or []:
        a = uci_to_policy_index(pv.move_uci, turn)
        if a < 0 or a not in legal_set:
            continue
        if len(rows) >= SF_MULTIPV_RAW_MAX:
            truncated = True
            break
        cp = SF_CP_SENTINEL if pv.cp is None else int(np.clip(int(pv.cp), -32000, 32000))
        mate = 0 if pv.mate is None else int(np.clip(int(pv.mate), -127, 127))
        if pv.wdl is None:
            w = d = -1
        else:
            w, d = int(round(float(pv.wdl[0]))), int(round(float(pv.wdl[1])))
        rows.append((a, cp, mate, w, d))
    if truncated:
        _warn_multipv_truncated()
    if not rows:
        return None
    return np.array(rows, dtype=np.int16)


def _collect_sf_label_meta(res) -> np.ndarray:
    """Record-level SF eval metadata (6,) int32; layout in replay/shard.py."""
    nodes = getattr(res, "nodes", None)
    depth = getattr(res, "depth", None)
    cp = SF_CP_SENTINEL if res.cp is None else int(np.clip(int(res.cp), -32000, 32000))
    mate = 0 if res.mate is None else int(np.clip(int(res.mate), -127, 127))
    if res.wdl is None:
        w = d = -1
    else:
        w, d = int(round(float(res.wdl[0]))), int(round(float(res.wdl[1])))
    # nodes/depth are config-driven; clamp so a huge budget can't overflow the
    # int32 field and raise out of _process_sf_results mid-selfplay.
    return np.array(
        [min(int(nodes), 2**31 - 1) if nodes is not None else -1,
         min(int(depth), 2**31 - 1) if depth is not None else -1,
         cp, mate, w, d],
        dtype=np.int32,
    )


def _stamp_sparse_sf_labels(rec, res, *, turn: bool, legal_set: set[int]) -> None:
    rec.sf_multipv_raw = _collect_sparse_pv_rows(res, turn=turn, legal_set=legal_set)
    rec.sf_label_meta = _collect_sf_label_meta(res)


def _collect_sf_pv_candidates(
    res,
    *,
    _turn: bool,
    legal_set: set[int],
    sf_wdl_use_cp_logistic: bool = False,
    sf_wdl_cp_slope: float = 0.010,
    sf_wdl_cp_draw_width: float = 60.0,
) -> tuple[list[int], list[float]]:
    """Extract (action_idx, w + 0.5*d) per legal SF MultiPV candidate."""
    cand_idxs: list[int] = []
    cand_scores: list[float] = []
    for pv in getattr(res, "pvs", None) or []:
        a = uci_to_policy_index(pv.move_uci, _turn)
        if a < 0 or a not in legal_set:
            continue
        score = _pv_wdl_score(
            pv,
            sf_wdl_use_cp_logistic=sf_wdl_use_cp_logistic,
            sf_wdl_cp_slope=sf_wdl_cp_slope,
            sf_wdl_cp_draw_width=sf_wdl_cp_draw_width,
        )
        if score is None:
            continue
        cand_idxs.append(a)
        cand_scores.append(score)
    return cand_idxs, cand_scores


def _build_sf_policy_target(
    cand_idxs: list[int], cand_scores: list[float],
    *, legal_indices: np.ndarray,
    sf_policy_temp: float, sf_policy_label_smooth: float,
) -> np.ndarray:
    """Softmax over MultiPV candidates → POLICY_SIZE vector with optional
    legal-set label smoothing. Final vector is renormalized."""
    scores = np.array(cand_scores, dtype=np.float64) / max(1e-6, sf_policy_temp)
    p_top = _softmax_np(scores).astype(np.float32, copy=False)
    p_sf = np.zeros((POLICY_SIZE,), dtype=np.float32)
    for a, p in zip(cand_idxs, p_top, strict=False):
        p_sf[int(a)] += float(p)

    # Only smooth when SF's candidates don't already cover every legal move. When
    # fully covered (the common case — multipv=40 ≥ legal count for ~83% of
    # positions) the softmax is already a complete distribution and the uniform
    # floor would just flatten it. When uncovered legal moves exist, the floor
    # gives them mass strictly below every covered move (covered = floor + share).
    n_covered = int(np.isin(legal_indices, cand_idxs).sum())  # legal moves SF scored
    has_uncovered = n_covered < int(legal_indices.size)
    if sf_policy_label_smooth > 0.0 and legal_indices.size > 0 and has_uncovered:
        p_sf *= 1.0 - sf_policy_label_smooth
        p_sf[legal_indices] += sf_policy_label_smooth / float(legal_indices.size)

    ps = float(p_sf.sum())
    if ps > 0:
        p_sf /= ps
    return p_sf


def _attach_sf_target_to_last_record(
    state: SelfplayState, idx: int,
    *, p_sf: np.ndarray, a_idx: int, res, legal_indices: np.ndarray,
    turn: bool | None = None,
    sf_wdl_use_cp_logistic: bool = False,
    sf_wdl_cp_slope: float = 0.010,
    sf_wdl_cp_draw_width: float = 60.0,
) -> None:
    """Stamp SF policy target / move idx / wdl / legal_mask onto the latest
    _NetRecord (idempotent: skipped if already populated)."""
    if not state.samples_per_game[idx]:
        return
    rec = state.samples_per_game[idx][-1]
    if rec.sf_policy_target is not None or rec.sf_move_index is not None:
        return
    rec.sf_policy_target = p_sf
    rec.sf_move_index = a_idx
    rec.sf_wdl = _sf_result_wdl_for_record(
        res,
        sf_wdl_use_cp_logistic=sf_wdl_use_cp_logistic,
        sf_wdl_cp_slope=sf_wdl_cp_slope,
        sf_wdl_cp_draw_width=sf_wdl_cp_draw_width,
    )
    if turn is not None:
        _stamp_sparse_sf_labels(
            rec, res, turn=bool(turn), legal_set={int(x) for x in legal_indices},
        )
    _sf_mask = np.zeros((POLICY_SIZE,), dtype=np.uint8)
    _sf_mask[legal_indices] = 1
    rec.sf_legal_mask = _sf_mask


def _attach_sf_target_to_record(
    rec: _NetRecord,
    *,
    p_sf: np.ndarray,
    a_idx: int,
    res,
    legal_indices: np.ndarray,
    turn: bool | None = None,
    sf_wdl_use_cp_logistic: bool = False,
    sf_wdl_cp_slope: float = 0.010,
    sf_wdl_cp_draw_width: float = 60.0,
) -> None:
    if rec.sf_policy_target is not None or rec.sf_move_index is not None:
        return
    rec.sf_policy_target = p_sf
    rec.sf_move_index = a_idx
    rec.sf_wdl = _sf_result_wdl_for_record(
        res,
        sf_wdl_use_cp_logistic=sf_wdl_use_cp_logistic,
        sf_wdl_cp_slope=sf_wdl_cp_slope,
        sf_wdl_cp_draw_width=sf_wdl_cp_draw_width,
    )
    if turn is not None:
        _stamp_sparse_sf_labels(
            rec, res, turn=bool(turn), legal_set={int(x) for x in legal_indices},
        )
    _sf_mask = np.zeros((POLICY_SIZE,), dtype=np.uint8)
    _sf_mask[legal_indices] = 1
    rec.sf_legal_mask = _sf_mask


def _process_sf_label_result_for_record(
    state: SelfplayState,
    *,
    rec: _NetRecord,
    res,
    turn: bool,
    legal_indices: np.ndarray,
) -> None:
    if legal_indices.size == 0:
        return
    sf_policy_temp = float(state.game.sf_policy_temp)
    sf_policy_label_smooth = float(state.game.sf_policy_label_smooth)
    legal_set = {int(x) for x in legal_indices}

    a_idx = uci_to_policy_index(res.bestmove_uci, bool(turn))
    if a_idx < 0 or a_idx not in legal_set:
        a_idx = int(legal_indices[0])

    cand_idxs, cand_scores = _collect_sf_pv_candidates(
        res, _turn=bool(turn), legal_set=legal_set,
        sf_wdl_use_cp_logistic=bool(state.game.sf_wdl_use_cp_logistic),
        sf_wdl_cp_slope=float(state.game.sf_wdl_cp_slope),
        sf_wdl_cp_draw_width=float(state.game.sf_wdl_cp_draw_width),
    )
    if not cand_idxs:
        cand_idxs = [a_idx]
        cand_scores = [0.0]

    p_sf = _build_sf_policy_target(
        cand_idxs, cand_scores,
        legal_indices=legal_indices,
        sf_policy_temp=sf_policy_temp,
        sf_policy_label_smooth=sf_policy_label_smooth,
    )
    _attach_sf_target_to_record(
        rec, p_sf=p_sf, a_idx=a_idx, res=res, legal_indices=legal_indices,
        turn=bool(turn),
        sf_wdl_use_cp_logistic=bool(state.game.sf_wdl_use_cp_logistic),
        sf_wdl_cp_slope=float(state.game.sf_wdl_cp_slope),
        sf_wdl_cp_draw_width=float(state.game.sf_wdl_cp_draw_width),
    )


def submit_async_sf_label_queries(state: SelfplayState, idxs: list[int]) -> int:
    """Submit selfplay SF label queries without blocking.

    The pending task captures the exact record and legal move set from query
    time, so later game advancement or slot recycling cannot stamp the result
    onto the wrong sample.
    """
    if not idxs:
        return 0
    assert isinstance(state.stockfish, StockfishPool)
    max_pending = max(1, int(state.batch_size) * 8)
    submitted = 0
    for idx in idxs:
        if len(state.pending_sf_labels) >= max_pending:
            break
        if not _slot_latest_record_needs_sf_label(state, idx):
            continue
        legal_indices = state.cboards[idx].legal_move_indices()
        if legal_indices.size == 0:
            continue
        fut = state.stockfish.submit(
            state.cboards[idx].fen(),
            nodes=_eff_sf_nodes(state, idx, for_move=False),
            syzygy_path=_sf_syzygy_path_for_slot(state, idx),
        )
        state.pending_sf_labels.append(
            _PendingSfLabel(
                future=fut,
                record=state.samples_per_game[idx][-1],
                turn=bool(state.cboards[idx].turn),
                legal_indices=np.asarray(legal_indices, dtype=np.int64).copy(),
            ),
        )
        submitted += 1
    return submitted


def poll_async_sf_labels(state: SelfplayState) -> tuple[int, int]:
    """Attach completed async SF labels. Returns ``(attached, failed)``."""
    if not state.pending_sf_labels:
        return 0, 0
    still_pending: list[_PendingSfLabel] = []
    attached = 0
    failed = 0
    for pending in state.pending_sf_labels:
        if not pending.future.done():
            still_pending.append(pending)
            continue
        try:
            res = pending.future.result()
            _process_sf_label_result_for_record(
                state,
                rec=pending.record,
                res=res,
                turn=pending.turn,
                legal_indices=pending.legal_indices,
            )
            attached += 1
        except Exception as exc:  # pragma: no cover - defensive drop on SF failure.
            failed += 1
            _LOG.debug("async SF label failed: %s", exc, exc_info=True)
    state.pending_sf_labels = still_pending
    return attached, failed


def flush_async_sf_labels_for_records(
    state: SelfplayState, records: list[_NetRecord],
) -> tuple[int, int]:
    """Wait for pending labels attached to these records before replay emit."""
    if not state.pending_sf_labels or not records:
        return 0, 0
    target_records = set(records)
    still_pending: list[_PendingSfLabel] = []
    attached = 0
    failed = 0
    for pending in state.pending_sf_labels:
        if pending.record not in target_records:
            still_pending.append(pending)
            continue
        try:
            res = pending.future.result()
            _process_sf_label_result_for_record(
                state,
                rec=pending.record,
                res=res,
                turn=pending.turn,
                legal_indices=pending.legal_indices,
            )
            attached += 1
        except Exception as exc:  # pragma: no cover - defensive drop on SF failure.
            failed += 1
            _LOG.debug("async SF label failed during finalize: %s", exc, exc_info=True)
    state.pending_sf_labels = still_pending
    return attached, failed


def _push_curriculum_opponent_move(
    state: SelfplayState, idx: int,
    *, legal_indices: np.ndarray,
    cand_idxs: list[int], cand_scores: list[float], regret_limit: float,
) -> None:
    """Pick a curriculum-strength opponent move + push it on the board + advance
    the tree root for next-ply reuse. Marks the slot done if the push terminates."""
    opp_move_idx = _choose_curriculum_opponent_move(
        rng=state.rng,
        legal_indices=legal_indices,
        cand_indices=cand_idxs,
        cand_scores=cand_scores,
        regret_limit=regret_limit,
    )
    if state.samples_per_game[idx]:
        rank, regret = _sf_played_move_diagnostics(opp_move_idx, cand_idxs, cand_scores)
        rec = state.samples_per_game[idx][-1]
        rec.sf_played_move_index = int(opp_move_idx)
        rec.sf_played_rank = rank
        rec.sf_played_regret = regret
    state.cboards[idx].push_index(opp_move_idx)
    state.move_idx_history[idx].append(opp_move_idx)
    if state.mcts_tree is not None and state.root_ids[idx] >= 0:
        state.root_ids[idx] = state.mcts_tree.find_child(
            state.root_ids[idx], opp_move_idx,
        )
    if state.cboards[idx].is_game_over():
        state.done_arr[idx] = 1


def _process_sf_results(
    state: SelfplayState,
    idxs: list[int],
    *,
    results: dict,
    play_curriculum_moves: bool,
    attach_labels: bool = True,
) -> None:
    """Attach SF policy target (+legal mask) to the last _NetRecord per slot
    and, for curriculum games, push the SF-chosen move onto the board."""
    if not idxs:
        return

    sf_policy_temp = float(state.game.sf_policy_temp)
    sf_policy_label_smooth = float(state.game.sf_policy_label_smooth)
    sf_wdl_use_cp_logistic = bool(state.game.sf_wdl_use_cp_logistic)
    sf_wdl_cp_slope = float(state.game.sf_wdl_cp_slope)
    sf_wdl_cp_draw_width = float(state.game.sf_wdl_cp_draw_width)
    regret_limit = (
        float(state.opponent.wdl_regret_limit)
        if state.opponent.wdl_regret_limit is not None else float("inf")
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

        cand_idxs, cand_scores = _collect_sf_pv_candidates(
            res, _turn=_turn, legal_set=legal_set,
        )
        if not cand_idxs:
            cand_idxs = [a_idx]
            cand_scores = [0.0]

        if attach_labels and _slot_latest_record_needs_sf_label(state, idx):
            label_cand_idxs, label_cand_scores = _collect_sf_pv_candidates(
                res, _turn=_turn, legal_set=legal_set,
                sf_wdl_use_cp_logistic=sf_wdl_use_cp_logistic,
                sf_wdl_cp_slope=sf_wdl_cp_slope,
                sf_wdl_cp_draw_width=sf_wdl_cp_draw_width,
            )
            if not label_cand_idxs:
                label_cand_idxs = [a_idx]
                label_cand_scores = [0.0]
            p_sf = _build_sf_policy_target(
                label_cand_idxs, label_cand_scores,
                legal_indices=legal_indices,
                sf_policy_temp=sf_policy_temp,
                sf_policy_label_smooth=sf_policy_label_smooth,
            )
            _attach_sf_target_to_last_record(
                state, idx, p_sf=p_sf, a_idx=a_idx, res=res, legal_indices=legal_indices,
                turn=_turn,
                sf_wdl_use_cp_logistic=sf_wdl_use_cp_logistic,
                sf_wdl_cp_slope=sf_wdl_cp_slope,
                sf_wdl_cp_draw_width=sf_wdl_cp_draw_width,
            )

        if play_curriculum_moves and not state.selfplay_arr[idx]:
            _push_curriculum_opponent_move(
                state, idx, legal_indices=legal_indices,
                cand_idxs=cand_idxs, cand_scores=cand_scores,
                regret_limit=regret_limit,
            )


__all__ = [
    "finish_sf_annotation_and_moves",
    "finish_pending_curriculum_moves",
    "flush_async_sf_labels_for_records",
    "poll_async_sf_labels",
    "submit_async_curriculum_move_queries",
    "submit_async_sf_labels_from_curriculum_moves",
    "submit_async_sf_label_queries",
    "submit_sf_queries",
]
