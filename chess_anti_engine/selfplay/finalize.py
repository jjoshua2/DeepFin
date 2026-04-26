"""Finalize a completed game into ``ReplaySample`` objects + stats updates.

``finalize_game`` is the orchestrator; the underscore-prefixed helpers
break it into stages: terminal-result resolution, optional syzygy rescore,
volatility / SF-delta-6 computation, aggregate-stats update, and per-game
``ReplaySample`` build (which writes per-head legal masks + is_selfplay
tagging on each row).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
import numpy as np

from chess_anti_engine.moves import POLICY_SIZE, move_to_index
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.selfplay.state import (
    CompletedGameBatch,
    SelfplayState,
    _NetRecord,
)
from chess_anti_engine.selfplay.stockfish_turn import flip_wdl_pov
from chess_anti_engine.selfplay.temperature import apply_policy_temperature
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishResult
from chess_anti_engine.tablebase import (
    is_tb_eligible,
    probe_best_move,
    rescore_game_samples,
)
from chess_anti_engine.train.targets import hlgauss_target

if TYPE_CHECKING:
    from chess_anti_engine.encoding._lc0_ext import CBoard


_LOG = logging.getLogger("chess_anti_engine.selfplay")


def _sf_terminal_result(
    *,
    turn_is_white: bool,
    sf_res: StockfishResult | None,
    adjudication_threshold: float,
) -> str:
    """Map SF's evaluation of a timed-out position to a result string."""
    if sf_res is None or sf_res.wdl is None:
        return "1/2-1/2"
    wdl_stm = sf_res.wdl
    wdl_white = (
        np.asarray(wdl_stm, dtype=np.float32) if turn_is_white
        else flip_wdl_pov(wdl_stm)
    )
    if float(wdl_white[0]) > float(adjudication_threshold):
        return "1-0"
    if float(wdl_white[2]) > float(adjudication_threshold):
        return "0-1"
    return "1/2-1/2"




def _compute_terminal_result(
    state: SelfplayState, cb: CBoard,
) -> tuple[str, bool, StockfishResult | None]:
    """Determine the final result string for a finalized slot.

    Returns ``(result, was_adjudicated, sf_res)``.  When the game ended
    naturally (checkmate/stalemate/threefold/fifty-move) the CBoard
    already reports a non-"*" result; otherwise we adjudicate with a
    high-nodes SF eval.  ``sf_res`` is returned so the caller can log the
    final-position WDL on max-ply games.
    """
    result = cb.result()
    if result != "*":
        return result, False, None

    sf_res: StockfishResult | None = None
    try:
        if isinstance(state.stockfish, StockfishPool):
            sf_res = state.stockfish.submit(
                cb.fen(), nodes=int(state.terminal_eval_nodes),
            ).result()
        else:
            sf_res = state.stockfish.search(
                cb.fen(), nodes=int(state.terminal_eval_nodes),
            )
    except (OSError, ValueError, RuntimeError):
        sf_res = None

    result = _sf_terminal_result(
        turn_is_white=bool(cb.turn),
        sf_res=sf_res,
        adjudication_threshold=float(state.game.timeout_adjudication_threshold),
    )
    return result, True, sf_res


def _rescore_with_syzygy(
    state: SelfplayState,
    i: int,
    b: chess.Board,
    records: list[_NetRecord],
    result: str,
) -> tuple[str, dict[int, np.ndarray]]:
    """Apply syzygy-tablebase rescoring + per-sample policy overrides.

    Returns ``(result, tb_policy_overrides)``.  ``result`` may be
    overridden when the TB proves the outcome; ``tb_policy_overrides``
    is populated only when ``game.syzygy_rescore_policy`` is enabled.
    """
    tb_policy_overrides: dict[int, np.ndarray] = {}
    if not state.game.syzygy_path or state.starting_boards is None:
        return result, tb_policy_overrides

    starting = state.starting_boards[i]
    replay_board = starting.copy()
    replay_boards: list[chess.Board] = []
    move_stack = list(b.move_stack)
    # With ``state.has_c_ply``, ``b`` was rebuilt via ``copy(stack=False)``
    # + replayed network moves, so ``b.move_stack`` contains only network
    # moves. In the Python path ``b`` was pushed in-place, so its stack
    # contains opening + network moves.
    opening_len = 0 if state.has_c_ply else len(starting.move_stack)

    for mv in move_stack[opening_len:]:
        replay_board.push(mv)
        replay_boards.append(replay_board.copy())

    tb_result = rescore_game_samples(replay_boards, state.game.syzygy_path)
    if tb_result is not None:
        result = tb_result

    if not state.game.syzygy_rescore_policy:
        return result, tb_policy_overrides

    # Map each record's ply_index to its position in the records list. The
    # replay walks every ply in ``move_stack`` (including forced-move plies
    # that the 1-legal shortcut in ``run_network_turn`` push but skip
    # recording), so a naive ``sample_idx`` counter would overshoot ``t``
    # by the number of skipped forced plies — stamping TB overrides from
    # K+P endgame positions onto the wrong records.
    record_at_ply = {int(rec.ply_index): t for t, rec in enumerate(records)}

    replay_board = starting.copy()
    for mv in move_stack[opening_len:]:
        cur_ply = len(replay_board.move_stack)
        t = record_at_ply.get(cur_ply)
        if t is not None and is_tb_eligible(replay_board):
            best = probe_best_move(replay_board, state.game.syzygy_path)
            if best is not None:
                try:
                    a = int(move_to_index(best, replay_board))
                except (ValueError, KeyError):
                    a = -1
                if a >= 0:
                    p = np.zeros((POLICY_SIZE,), dtype=np.float32)
                    p[a] = 1.0
                    tb_policy_overrides[t] = p
        replay_board.push(mv)

    return result, tb_policy_overrides


@dataclass
class _PerGameCounters:
    """Per-game counters emitted on the ``on_game_complete`` callback."""

    w: int = 0
    d: int = 0
    l: int = 0
    total_draw_games: int = 0
    selfplay_games: int = 0
    selfplay_adjudicated_games: int = 0
    selfplay_draw_games: int = 0
    curriculum_games: int = 0
    curriculum_adjudicated_games: int = 0
    curriculum_draw_games: int = 0


def _update_aggregate_stats(
    state: SelfplayState,
    i: int,
    *,
    result: str,
    was_adjudicated: bool,
    game_plies: int,
) -> _PerGameCounters:
    """Fold one game's outcome into ``state.stats``; return its counters."""
    c = _PerGameCounters()
    is_sp = bool(state.selfplay_arr[i])

    if is_sp:
        state.stats.selfplay_games += 1
        c.selfplay_games = 1
        if was_adjudicated:
            state.stats.selfplay_adjudicated_games += 1
            c.selfplay_adjudicated_games = 1
    else:
        state.stats.curriculum_games += 1
        c.curriculum_games = 1
        if was_adjudicated:
            state.stats.curriculum_adjudicated_games += 1
            c.curriculum_adjudicated_games = 1

    if result == "1/2-1/2":
        state.stats.total_draw_games += 1
        c.total_draw_games = 1
        if is_sp:
            state.stats.selfplay_draw_games += 1
            c.selfplay_draw_games = 1
        else:
            state.stats.curriculum_draw_games += 1
            c.curriculum_draw_games = 1

    if not is_sp:
        net_col = state.net_color(i)
        if result == "1/2-1/2":
            state.stats.d += 1
            c.d = 1
        elif (result == "1-0" and net_col == chess.WHITE) or (
            result == "0-1" and net_col == chess.BLACK
        ):
            state.stats.w += 1
            c.w = 1
        else:
            state.stats.l += 1
            c.l = 1

        if c.w:
            state.stats.plies_win += game_plies
        elif c.d:
            state.stats.plies_draw += game_plies
        elif c.l:
            state.stats.plies_loss += game_plies

    return c


def _compute_volatility_and_sf_delta(
    state: SelfplayState,
    records: list[_NetRecord],
    ply_to_index: dict[int, int],
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    """Compute per-sample volatility targets + log SF eval delta6.

    Single pass over ``records``.  Updates ``state.stats.sf_d6_sum`` and
    ``state.stats.sf_d6_n`` as a side-effect.
    """
    n = len(records)
    vol_targets: list[np.ndarray | None] = [None] * n
    sf_vol_targets: list[np.ndarray | None] = [None] * n
    use_search = state.volatility_source == "search"

    for t in range(n):
        th = ply_to_index.get(int(records[t].ply_index) + 6)
        if th is None:
            continue
        if use_search:
            w0 = records[t].search_wdl_est
            w6 = records[th].search_wdl_est
        else:
            w0 = records[t].net_wdl_est
            w6 = records[th].net_wdl_est
        vol_targets[t] = np.abs(w6 - w0).astype(np.float32, copy=False)

        sf0 = records[t].sf_wdl
        sf6 = records[th].sf_wdl
        if (sf0 is not None) and (sf6 is not None):
            sf_vol_targets[t] = np.abs(sf6 - sf0).astype(np.float32, copy=False)
            wr0 = float(sf0[0]) + 0.5 * float(sf0[1])
            wr6 = float(sf6[0]) + 0.5 * float(sf6[1])
            state.stats.sf_d6_sum += abs(wr6 - wr0)
            state.stats.sf_d6_n += 1

    return vol_targets, sf_vol_targets


def _build_replay_samples(
    state: SelfplayState,
    i: int,
    records: list[_NetRecord],
    *,
    result: str,
    tb_policy_overrides: dict[int, np.ndarray],
    vol_targets: list[np.ndarray | None],
    sf_vol_targets: list[np.ndarray | None],
    total_plies_played: int,
    ply_to_index: dict[int, int],
) -> list[ReplaySample]:
    """Materialize one game's per-ply records into ``ReplaySample`` objects.

    Applies sample-weight and keep-prob subsampling, computes WDL /
    categorical / moves-left / future-policy targets, and substitutes
    TB policy overrides where available.  ``i`` is the slot index, used
    only to read ``state.selfplay_arr[i]`` for the per-sample
    ``is_selfplay`` tag.
    """
    out: list[ReplaySample] = []
    is_selfplay_slot = bool(state.selfplay_arr[i])

    for t, rec in enumerate(records):
        if float(rec.sample_weight) < 1.0 and state.rng.random() > float(rec.sample_weight):
            continue
        if float(rec.keep_prob) < 1.0 and state.rng.random() > float(rec.keep_prob):
            continue
        if not bool(rec.has_policy):
            continue

        if result == "1/2-1/2":
            wdl = 1
        elif (result == "1-0" and rec.pov_color == chess.WHITE) or (
            result == "0-1" and rec.pov_color == chess.BLACK
        ):
            wdl = 0
        else:
            wdl = 2

        total = max(1, int(total_plies_played))
        moves_left = float(max(0, total - int(rec.ply_index))) / max(
            1.0, float(state.game.max_plies),
        )

        scalar_v = 1.0 if wdl == 0 else (0.0 if wdl == 1 else -1.0)
        cat = hlgauss_target(
            scalar_v,
            num_bins=state.game.categorical_bins,
            sigma=state.game.hlgauss_sigma,
        )

        eff_probs = tb_policy_overrides.get(t, rec.policy_probs)
        soft = apply_policy_temperature(eff_probs, state.game.soft_policy_temp)

        future = None
        future_lmask = None
        future_idx = ply_to_index.get(int(rec.ply_index) + 2)
        # Don't gate on records[future_idx].has_policy — the future record's
        # MCTS distribution is valid supervision regardless of sim count
        # (fast-sim is noisier than full but still ground truth for "what the
        # network played"). Gating dropped 75% of targets.
        if future_idx is not None:
            future = records[future_idx].policy_probs
            future_lmask = records[future_idx].legal_mask

        vol = vol_targets[t]
        sf_vol = sf_vol_targets[t]

        out.append(
            ReplaySample(
                x=rec.x,
                policy_target=eff_probs,
                wdl_target=int(wdl),
                priority=float(rec.priority),
                has_policy=bool(rec.has_policy),
                sf_wdl=rec.sf_wdl,
                sf_move_index=rec.sf_move_index,
                sf_policy_target=rec.sf_policy_target,
                moves_left=moves_left,
                is_network_turn=True,
                categorical_target=cat,
                policy_soft_target=soft,
                future_policy_target=future,
                has_future=(future is not None),
                volatility_target=vol,
                has_volatility=(vol is not None),
                sf_volatility_target=sf_vol,
                has_sf_volatility=(sf_vol is not None),
                legal_mask=rec.legal_mask,
                sf_legal_mask=rec.sf_legal_mask,
                future_legal_mask=future_lmask,
                is_selfplay=is_selfplay_slot,
            ),
        )

    return out


def finalize_game(
    state: SelfplayState,
    i: int,
    all_samples: list[ReplaySample],
    on_game_complete: Callable[[CompletedGameBatch], None] | None,
) -> None:
    """Finalize a completed game: compute labels, build samples, update stats.

    Side effects:
    * Updates ``state.stats`` (aggregate counters for the whole batch).
    * Appends new ``ReplaySample`` instances to ``all_samples``.
    * Invokes ``on_game_complete`` with a ``CompletedGameBatch`` when it
      is provided and the game produced samples.
    * Clears ``all_samples`` in continuous mode (samples are delivered
      through the callback; retaining them would cause unbounded memory
      growth for long-running workers).
    """
    cb = state.cboards[i]
    b = state.replay_board(i)

    # TB adjudication (if enabled) short-circuits cb.result() — this game
    # was ended early at a known-result endgame position. Skip the SF
    # terminal eval branch and use the stashed TB result directly.
    tb_stash = state.tb_result_arr[i]
    was_tb_adjudicated = tb_stash is not None

    game_plies = int(cb.ply)
    state.stats.total_game_plies += game_plies

    is_cm = cb.is_checkmate()
    is_sm = cb.is_stalemate()
    if is_cm:
        state.stats.checkmate_games += 1
    elif is_sm:
        state.stats.stalemate_games += 1

    if was_tb_adjudicated:
        state.stats.tb_adjudicated_games += 1
        result: str = tb_stash  # type: ignore[assignment]
        was_adjudicated = False
        sf_res: StockfishResult | None = None
    else:
        result, was_adjudicated, sf_res = _compute_terminal_result(state, cb)
        if was_adjudicated:
            state.stats.adjudicated_games += 1

    # Debug: log unexpectedly short wins.  Skips selfplay (which is noisier).
    if (
        not state.selfplay_arr[i]
        and result in ("1-0", "0-1")
        and game_plies < int(state.game.max_plies) - 10
    ):
        outcome = b.outcome(claim_draw=True)
        _term = outcome.termination.name if outcome else "adjudicated"
        _LOG.warning(
            "Short win: %s at %d plies (max=%d), term=%s, adj=%s, is_over=%s",
            result, game_plies, int(state.game.max_plies), _term,
            was_adjudicated, cb.is_game_over(),
        )

    # Log final FEN + SF WDL for games that reached max_plies.
    # These are rare (~0.03% of games); SF eval was already paid for during adjudication.
    if was_adjudicated and game_plies >= int(state.game.max_plies):
        _wdl_str = (
            f"{float(sf_res.wdl[0]):.3f}/{float(sf_res.wdl[1]):.3f}/{float(sf_res.wdl[2]):.3f}"
            if (sf_res is not None and sf_res.wdl is not None) else "none"
        )
        _LOG.warning(
            "MAX_PLY_GAME plies=%d result=%s sf_wdl(stm)=%s fen=%s",
            game_plies, result, _wdl_str, cb.fen(),
        )

    records = state.samples_per_game[i]
    result, tb_policy_overrides = _rescore_with_syzygy(state, i, b, records, result)

    counters = _update_aggregate_stats(
        state,
        i,
        result=result,
        was_adjudicated=was_adjudicated,
        game_plies=game_plies,
    )

    ply_to_index = {int(rec.ply_index): idx for idx, rec in enumerate(records)}
    _sf_d6_sum_before = state.stats.sf_d6_sum
    _sf_d6_n_before = state.stats.sf_d6_n
    vol_targets, sf_vol_targets = _compute_volatility_and_sf_delta(
        state, records, ply_to_index,
    )
    _game_sf_d6_sum = state.stats.sf_d6_sum - _sf_d6_sum_before
    _game_sf_d6_n = state.stats.sf_d6_n - _sf_d6_n_before

    sample_start = len(all_samples)
    new_samples = _build_replay_samples(
        state,
        i,
        records,
        result=result,
        tb_policy_overrides=tb_policy_overrides,
        vol_targets=vol_targets,
        sf_vol_targets=sf_vol_targets,
        total_plies_played=int(cb.ply),
        ply_to_index=ply_to_index,
    )
    all_samples.extend(new_samples)

    if on_game_complete is not None:
        game_samples = list(all_samples[sample_start:])
        if game_samples:
            on_game_complete(
                CompletedGameBatch(
                    samples=game_samples,
                    positions=len(game_samples),
                    w=counters.w,
                    d=counters.d,
                    l=counters.l,
                    total_game_plies=game_plies,
                    adjudicated_games=1 if was_adjudicated else 0,
                    tb_adjudicated_games=1 if was_tb_adjudicated else 0,
                    total_draw_games=counters.total_draw_games,
                    selfplay_games=counters.selfplay_games,
                    selfplay_adjudicated_games=counters.selfplay_adjudicated_games,
                    selfplay_draw_games=counters.selfplay_draw_games,
                    curriculum_games=counters.curriculum_games,
                    curriculum_adjudicated_games=counters.curriculum_adjudicated_games,
                    curriculum_draw_games=counters.curriculum_draw_games,
                    checkmate_games=1 if is_cm else 0,
                    stalemate_games=1 if is_sm else 0,
                    plies_win=game_plies if counters.w else 0,
                    plies_draw=game_plies if counters.d else 0,
                    plies_loss=game_plies if counters.l else 0,
                    sf_d6_sum=float(_game_sf_d6_sum),
                    sf_d6_n=int(_game_sf_d6_n),
                ),
            )
    # In continuous mode, samples flow through ``on_game_complete`` and
    # ``all_samples`` would otherwise grow without bound.
    if state.continuous:
        all_samples.clear()


__all__ = ["finalize_game"]
