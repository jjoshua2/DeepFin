"""Finalize a completed game into ``ReplaySample`` objects + stats updates.

``finalize_game`` is the orchestrator; the underscore-prefixed helpers
break it into stages: terminal-result resolution, optional syzygy rescore,
volatility / SF-delta-6 computation, aggregate-stats update, and per-game
``ReplaySample`` build (which writes per-head legal masks + is_selfplay
tagging on each row).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
import numpy as np

from chess_anti_engine.moves import (
    POLICY_SIZE,
    move_to_index_for_encoding,
    policy_index_for_encoding,
    policy_mask_to_encoding,
    policy_size_for_encoding,
    policy_vector_to_encoding,
)
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.selfplay.game import _result_to_wdl
from chess_anti_engine.selfplay.state import (
    CompletedGameBatch,
    SelfplayState,
    _NetRecord,
)
from chess_anti_engine.selfplay.stockfish_turn import (
    flip_wdl_pov,
    flush_async_sf_labels_for_records,
)
from chess_anti_engine.selfplay.temperature import apply_policy_temperature
from chess_anti_engine.replay.shard import SF_CP_SENTINEL, SF_MULTIPV_PAD_ROW, SF_MULTIPV_RAW_MAX
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishResult
from chess_anti_engine.tablebase import (
    is_tb_eligible,
    probe_best_move,
    rescore_game_samples,
)
from chess_anti_engine.train.targets import categorical_target_value, hlgauss_target

if TYPE_CHECKING:
    from chess_anti_engine.encoding._lc0_ext import CBoard


_LOG = logging.getLogger("chess_anti_engine.selfplay")

# Max cp-regret used to normalize the per-move SF regret vector (sf_p0_regret).
# A move 1000+ cp worse than SF's best becomes 1.0 (maximally bad); the best
# move is 0.0. Legal moves absent from the MultiPV default to 1.0.
SF_OWN_REGRET_CAP_CP = 1000.0


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
                    a = int(move_to_index_for_encoding(
                        best,
                        replay_board,
                        policy_encoding=state.game.policy_encoding,
                    ))
                except (ValueError, KeyError):
                    a = -1
                if a >= 0:
                    p = np.zeros(
                        (policy_size_for_encoding(state.game.policy_encoding),),
                        dtype=np.float32,
                    )
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
    outcome_stats: dict[str, int] | None = None


@dataclass(frozen=True)
class _DiffFocusGameStats:
    records: int = 0
    kept: int = 0
    keep_prob_sum: float = 0.0
    keep_limited: int = 0
    sample_weight_sum: float = 0.0
    sample_weight_limited: int = 0
    priority_sum: float = 0.0
    priority_sq_sum: float = 0.0
    priority_min: float = 0.0
    priority_max: float = 0.0


@dataclass(frozen=True)
class _GumbelPolicyGameStats:
    records: int = 0
    top_prob_sum: float = 0.0
    action_prob_sum: float = 0.0
    entropy_sum: float = 0.0
    eff_moves_sum: float = 0.0
    candidate_mass_sum: float = 0.0
    non_candidate_top_prob_sum: float = 0.0
    argmax_is_candidate_sum: int = 0
    argmax_is_action_sum: int = 0
    legal_count_sum: int = 0
    candidate_count_sum: int = 0


_OPENING_SOURCES = {"book1", "book2", "random", "start", "fenlist"}


def _opening_source_for_stats(state: SelfplayState, i: int) -> str:
    try:
        source = str(state.opening_source_arr[i])
    except (AttributeError, IndexError):
        return "unknown"
    return source if source in _OPENING_SOURCES else "unknown"


def _inc_outcome(stats: dict[str, int], key: str, amount: int = 1) -> None:
    stats[str(key)] = int(stats.get(str(key), 0)) + int(amount)


def _merge_outcome_stats(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, val in src.items():
        _inc_outcome(dst, str(key), int(val))


def _compute_diff_focus_game_stats(
    records: list[_NetRecord],
    samples: list[ReplaySample],
) -> _DiffFocusGameStats:
    eligible = [rec for rec in records if bool(rec.has_policy)]
    n = len(eligible)
    if n <= 0:
        return _DiffFocusGameStats()

    keep_sum = 0.0
    keep_limited = 0
    weight_sum = 0.0
    weight_limited = 0
    priority_sum = 0.0
    priority_sq_sum = 0.0
    priority_min = float("inf")
    priority_max = float("-inf")

    for rec in eligible:
        keep_prob = float(rec.keep_prob)
        sample_weight = float(rec.sample_weight)
        priority = float(rec.priority)
        keep_sum += keep_prob
        weight_sum += sample_weight
        keep_limited += 1 if keep_prob < 1.0 else 0
        weight_limited += 1 if sample_weight < 1.0 else 0
        priority_sum += priority
        priority_sq_sum += priority * priority
        priority_min = min(priority_min, priority)
        priority_max = max(priority_max, priority)

    return _DiffFocusGameStats(
        records=n,
        # Count only policy-bearing samples: with record_fast_ply_value the
        # samples list also carries value-only fast rows (has_policy=0), which
        # are not in the diff-focus-eligible denominator above — counting them
        # would report keep rates > 1.0 and corrupt the replay-filter telemetry.
        kept=sum(1 for s in samples if bool(s.has_policy)),
        keep_prob_sum=keep_sum,
        keep_limited=keep_limited,
        sample_weight_sum=weight_sum,
        sample_weight_limited=weight_limited,
        priority_sum=priority_sum,
        priority_sq_sum=priority_sq_sum,
        priority_min=priority_min,
        priority_max=priority_max,
    )


def _compute_gumbel_policy_game_stats(records: list[_NetRecord]) -> _GumbelPolicyGameStats:
    n = 0
    top_prob_sum = 0.0
    action_prob_sum = 0.0
    entropy_sum = 0.0
    eff_moves_sum = 0.0
    candidate_mass_sum = 0.0
    non_candidate_top_prob_sum = 0.0
    argmax_is_candidate_sum = 0
    argmax_is_action_sum = 0
    legal_count_sum = 0
    candidate_count_sum = 0

    for rec in records:
        if not bool(rec.has_policy):
            continue
        diag = rec.gumbel_policy_diag
        if not diag:
            continue
        n += 1
        top_prob_sum += float(diag.get("top_prob", 0.0))
        action_prob_sum += float(diag.get("action_prob", 0.0))
        entropy_sum += float(diag.get("entropy", 0.0))
        eff_moves_sum += float(diag.get("eff_moves", 0.0))
        candidate_mass_sum += float(diag.get("candidate_mass", 0.0))
        non_candidate_top_prob_sum += float(diag.get("non_candidate_top_prob", 0.0))
        argmax_is_candidate_sum += int(float(diag.get("argmax_is_candidate", 0.0)) > 0.5)
        argmax_is_action_sum += int(float(diag.get("argmax_is_action", 0.0)) > 0.5)
        legal_count_sum += round(float(diag.get("legal_count", 0.0)))
        candidate_count_sum += round(float(diag.get("candidate_count", 0.0)))

    return _GumbelPolicyGameStats(
        records=n,
        top_prob_sum=top_prob_sum,
        action_prob_sum=action_prob_sum,
        entropy_sum=entropy_sum,
        eff_moves_sum=eff_moves_sum,
        candidate_mass_sum=candidate_mass_sum,
        non_candidate_top_prob_sum=non_candidate_top_prob_sum,
        argmax_is_candidate_sum=argmax_is_candidate_sum,
        argmax_is_action_sum=argmax_is_action_sum,
        legal_count_sum=legal_count_sum,
        candidate_count_sum=candidate_count_sum,
    )


def _update_gumbel_policy_stats(state: SelfplayState, stats: _GumbelPolicyGameStats) -> None:
    if stats.records <= 0:
        return
    state.stats.gumbel_policy_diag_n += int(stats.records)
    state.stats.gumbel_policy_top_prob_sum += float(stats.top_prob_sum)
    state.stats.gumbel_policy_action_prob_sum += float(stats.action_prob_sum)
    state.stats.gumbel_policy_entropy_sum += float(stats.entropy_sum)
    state.stats.gumbel_policy_eff_moves_sum += float(stats.eff_moves_sum)
    state.stats.gumbel_policy_candidate_mass_sum += float(stats.candidate_mass_sum)
    state.stats.gumbel_policy_non_candidate_top_prob_sum += float(
        stats.non_candidate_top_prob_sum,
    )
    state.stats.gumbel_policy_argmax_is_candidate_sum += int(stats.argmax_is_candidate_sum)
    state.stats.gumbel_policy_argmax_is_action_sum += int(stats.argmax_is_action_sum)
    state.stats.gumbel_policy_legal_count_sum += int(stats.legal_count_sum)
    state.stats.gumbel_policy_candidate_count_sum += int(stats.candidate_count_sum)


def _update_diff_focus_stats(state: SelfplayState, stats: _DiffFocusGameStats) -> None:
    if stats.records <= 0:
        return
    old_n = int(state.stats.diff_focus_records)
    state.stats.diff_focus_records += int(stats.records)
    state.stats.diff_focus_kept += int(stats.kept)
    state.stats.diff_focus_keep_prob_sum += float(stats.keep_prob_sum)
    state.stats.diff_focus_keep_limited += int(stats.keep_limited)
    state.stats.diff_focus_sample_weight_sum += float(stats.sample_weight_sum)
    state.stats.diff_focus_sample_weight_limited += int(stats.sample_weight_limited)
    state.stats.diff_focus_priority_sum += float(stats.priority_sum)
    state.stats.diff_focus_priority_sq_sum += float(stats.priority_sq_sum)
    if old_n <= 0:
        state.stats.diff_focus_priority_min = float(stats.priority_min)
        state.stats.diff_focus_priority_max = float(stats.priority_max)
    else:
        state.stats.diff_focus_priority_min = min(
            float(state.stats.diff_focus_priority_min), float(stats.priority_min),
        )
        state.stats.diff_focus_priority_max = max(
            float(state.stats.diff_focus_priority_max), float(stats.priority_max),
        )


def _update_aggregate_stats(
    state: SelfplayState,
    i: int,
    *,
    result: str,
    was_adjudicated: bool,
    game_plies: int,
) -> _PerGameCounters:
    """Fold one game's outcome into ``state.stats``; return its counters."""
    c = _PerGameCounters(outcome_stats={})
    is_sp = bool(state.selfplay_arr[i])
    source = _opening_source_for_stats(state, i)
    assert c.outcome_stats is not None
    outcome_stats = c.outcome_stats
    _inc_outcome(outcome_stats, f"opening_{source}_games")

    if is_sp:
        state.stats.selfplay_games += 1
        c.selfplay_games = 1
        _inc_outcome(outcome_stats, f"selfplay_{source}_games")
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
        _inc_outcome(outcome_stats, f"opening_{source}_draws")
        if is_sp:
            state.stats.selfplay_draw_games += 1
            c.selfplay_draw_games = 1
            _inc_outcome(outcome_stats, f"selfplay_{source}_draws")
        else:
            state.stats.curriculum_draw_games += 1
            c.curriculum_draw_games = 1

    if not is_sp:
        net_col = state.net_color(i)
        side = "net_white" if net_col == chess.WHITE else "net_black"
        if result == "1/2-1/2":
            outcome = "d"
        elif (result == "1-0" and net_col == chess.WHITE) or (
            result == "0-1" and net_col == chess.BLACK
        ):
            outcome = "w"
        else:
            outcome = "l"
        # Blind-spot FEN seeds force the net onto the historically-losing seat, so
        # they are systematic losses. They MUST NOT enter the PID winrate sample
        # (state.stats.w/d/l is reported to the controller, which would otherwise
        # ease SF across the whole batch) — exclude them exactly like selfplay
        # games, but still record per-source telemetry so seed performance is
        # visible via the curriculum_fenlist_* outcome keys.
        count_in_pid = source != "fenlist"
        if count_in_pid:
            if outcome == "d":
                state.stats.d += 1
                c.d = 1
            elif outcome == "w":
                state.stats.w += 1
                c.w = 1
            else:
                state.stats.l += 1
                c.l = 1
        _inc_outcome(outcome_stats, f"curriculum_{side}_{outcome}")
        _inc_outcome(outcome_stats, f"curriculum_{source}_{outcome}")
        _inc_outcome(outcome_stats, f"curriculum_{source}_{side}_{outcome}")

        if count_in_pid:
            if c.w:
                state.stats.plies_win += game_plies
            elif c.d:
                state.stats.plies_draw += game_plies
            elif c.l:
                state.stats.plies_loss += game_plies

    _merge_outcome_stats(state.stats.outcome_stats, outcome_stats)
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


@dataclass(frozen=True, slots=True)
class _SfRegretSuffix:
    total: float
    d95: float
    d98: float
    max_single: float
    h4: float
    h6: float
    h12: float
    h24: float
    h50: float
    count: int


def _suffix_sf_regret_features(
    records: list[_NetRecord],
    *,
    is_selfplay: bool,
) -> list[_SfRegretSuffix]:
    """Future opponent-SF loss summaries from each network-turn sample to game end.

    Discounted d95/d98 suffixes advance once per network record; h4..h50
    sources use ply-index windows.
    """
    zero = _SfRegretSuffix(
        total=0.0,
        d95=0.0,
        d98=0.0,
        max_single=0.0,
        h4=0.0,
        h6=0.0,
        h12=0.0,
        h24=0.0,
        h50=0.0,
        count=0,
    )
    if is_selfplay:
        return [zero for _ in records]
    out = [zero for _ in records]
    regrets: list[float] = []
    has_regret: list[bool] = []
    for rec in records:
        raw = rec.sf_played_regret
        if raw is not None and np.isfinite(float(raw)):
            regrets.append(max(0.0, float(raw)))
            has_regret.append(True)
        else:
            regrets.append(0.0)
            has_regret.append(False)

    def _horizon_sum(start: int, horizon: int) -> float:
        end_ply = int(records[start].ply_index) + int(horizon)
        total = 0.0
        for j in range(start, len(records)):
            if int(records[j].ply_index) >= end_ply:
                break
            total += regrets[j]
        return total

    running_sum = 0.0
    running_d95 = 0.0
    running_d98 = 0.0
    running_max = 0.0
    running_count = 0
    for t in range(len(records) - 1, -1, -1):
        value = regrets[t]
        if has_regret[t]:
            running_sum += value
            running_d95 = value + 0.95 * running_d95
            running_d98 = value + 0.98 * running_d98
            running_max = max(running_max, value)
            running_count += 1
        else:
            running_d95 *= 0.95
            running_d98 *= 0.98
        out[t] = _SfRegretSuffix(
            total=running_sum,
            d95=running_d95,
            d98=running_d98,
            max_single=running_max,
            h4=_horizon_sum(t, 4),
            h6=_horizon_sum(t, 6),
            h12=_horizon_sum(t, 12),
            h24=_horizon_sum(t, 24),
            h50=_horizon_sum(t, 50),
            count=running_count,
        )
    return out


def _stable_game_id(
    *,
    start_fen: str,
    opening_source: str,
    move_trace: str,
    result: str,
    total_plies_played: int,
) -> int:
    game_key = (
        f"{opening_source}|{start_fen}|{move_trace}|{result}|"
        f"{int(total_plies_played)}"
    )
    return (
        int.from_bytes(hashlib.blake2b(game_key.encode("utf-8"), digest_size=8).digest(), "big")
        & ((1 << 63) - 1)
    )


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
    starting_boards = state.starting_boards
    start_board = None if starting_boards is None else starting_boards[i]
    start_fen = start_board.fen() if start_board is not None else ""
    opening_source = str(state.opening_source_arr[i]) if i < len(state.opening_source_arr) else ""
    move_trace = ",".join(str(int(m)) for m in state.move_idx_history[i])
    game_id = _stable_game_id(
        start_fen=start_fen,
        opening_source=opening_source,
        move_trace=move_trace,
        result=result,
        total_plies_played=total_plies_played,
    )
    suffix_sf_regret = _suffix_sf_regret_features(records, is_selfplay=is_selfplay_slot)

    keep_fast_plies = bool(getattr(state.game, "record_fast_ply_value", False))
    for t, rec in enumerate(records):
        row_has_policy = bool(rec.has_policy)
        if row_has_policy:
            # sample_weight / diff-focus keep_prob are POLICY-difficulty
            # subsampling signals (q-surprise/KL of the search vs the prior),
            # so they only gate policy rows. Value-only fast rows bypass them:
            # subsampling value rows by policy difficulty would both shrink
            # the fast-ply value multiplier and bias the kept rows toward
            # positions where the low-sim search disagreed with the prior.
            if float(rec.sample_weight) < 1.0 and state.rng.random() > float(rec.sample_weight):
                continue
            if float(rec.keep_prob) < 1.0 and state.rng.random() > float(rec.keep_prob):
                continue
        if not row_has_policy and not keep_fast_plies:
            # Fast-ply (playout-capped) records are dropped by default. With
            # record_fast_ply_value they become value-only rows (has_policy=0):
            # outcome/search-WDL/moves_left are valid targets regardless of sim
            # count — the KataGo playout-cap design trains value on ALL plies
            # for exactly this reason. losses.py masks only the MAIN policy
            # head by has_policy; the aux heads use their own flags
            # (has_policy_soft/has_future/has_sf_p0*), so every policy-head
            # target is cleared below for fast rows to keep them value-only.
            # Fast plies never get SF label queries either way
            # (_slot_latest_record_needs_sf_label gates on has_policy).
            continue

        wdl = _result_to_wdl(result, pov_white=rec.pov_color == chess.WHITE)

        total = max(1, int(total_plies_played))
        moves_left = float(max(0, total - int(rec.ply_index))) / max(
            1.0, float(state.game.max_plies),
        )

        scalar_v = 1.0 if wdl == 0 else (0.0 if wdl == 1 else -1.0)
        suffix_regret = suffix_sf_regret[t]
        cat = hlgauss_target(
            categorical_target_value(
                scalar_v, rec.sf_wdl,
                blend_frac=getattr(state.game, "categorical_blend_frac", 0.0),
                search_wdl=getattr(rec, "search_wdl_est", None),
                search_blend_frac=getattr(state.game, "categorical_search_blend_frac", 0.0),
            ),
            num_bins=state.game.categorical_bins,
            sigma=state.game.hlgauss_sigma,
        )

        eff_probs = tb_policy_overrides.get(
            t,
            policy_vector_to_encoding(rec.policy_probs, policy_encoding=state.game.policy_encoding),
        )
        # Value-only fast rows (has_policy=0) must not carry aux policy-head
        # targets: has_policy_soft/has_future are derived from target presence
        # in the shard, so populating them here would train policy_soft /
        # policy_future on low-sim visit distributions despite the row being
        # value-only. Gate every policy-head target on THIS row's has_policy.
        soft = (
            apply_policy_temperature(eff_probs, state.game.soft_policy_temp)
            if row_has_policy else None
        )

        future = None
        future_lmask = None
        future_idx = (
            ply_to_index.get(int(rec.ply_index) + 2) if row_has_policy else None
        )
        # Don't gate on records[future_idx].has_policy — the future record's
        # MCTS distribution is valid supervision regardless of sim count
        # (fast-sim is noisier than full but still ground truth for "what the
        # network played"). Gating dropped 75% of targets.
        if future_idx is not None:
            future = policy_vector_to_encoding(
                records[future_idx].policy_probs,
                policy_encoding=state.game.policy_encoding,
            )
            future_legal_mask = records[future_idx].legal_mask
            if future_legal_mask is not None:
                future_lmask = policy_mask_to_encoding(
                    future_legal_mask,
                    policy_encoding=state.game.policy_encoding,
                ).astype(np.uint8, copy=False)

        vol = vol_targets[t]
        sf_vol = sf_vol_targets[t]
        sf_search_gap = None
        if rec.sf_wdl is not None:
            sf_arr = np.asarray(rec.sf_wdl, dtype=np.float32)
            search_arr = np.asarray(rec.search_wdl_est, dtype=np.float32)
            if sf_arr.shape == (3,) and search_arr.shape == (3,):
                sf_sig = float(sf_arr[0] - sf_arr[2])
                search_sig = float(search_arr[0] - search_arr[2])
                sf_search_gap = abs(sf_sig - search_sig)
        sf_policy_target = None
        if rec.sf_policy_target is not None and state.game.record_dense_sf_policy:
            sf_policy_target = policy_vector_to_encoding(
                rec.sf_policy_target,
                policy_encoding=state.game.policy_encoding,
            )

        # P0 own-move teacher for policy_own: SF's analysis of THIS position is
        # the *prior* ply's sf_policy (SF labels run at P1 = after a move). Only
        # valid when the immediately-preceding ply is a stored full record with
        # an SF label — i.e. two consecutive full plies, which only happens in
        # selfplay (the net plays every ply). See replay/buffer.py.
        sf_p0_policy_target = None
        if (
            row_has_policy
            and getattr(state.game, "record_sf_p0_policy", False)
            and is_selfplay_slot
        ):
            prev_idx = ply_to_index.get(int(rec.ply_index) - 1)
            prev_sf = (
                records[prev_idx].sf_policy_target
                if prev_idx is not None and records[prev_idx].has_policy
                else None
            )
            if prev_sf is not None:
                sf_p0_policy_target = policy_vector_to_encoding(
                    prev_sf, policy_encoding=state.game.policy_encoding,
                )

        # P0 own-move SF cp-regret vector for policy_own (regret-weighted SF
        # teacher). Uses the SAME one-ply shift as sf_p0_policy_target: the prior
        # full ply's raw MultiPV is SF's read of THIS position. Per-move values in
        # [0,1] (best move -> 0.0), NOT a distribution. See replay/buffer.py.
        sf_p0_regret = None
        if (
            row_has_policy
            and getattr(state.game, "record_sf_p0_regret", False)
            and is_selfplay_slot
        ):
            prev_idx = ply_to_index.get(int(rec.ply_index) - 1)
            prev_raw = (
                getattr(records[prev_idx], "sf_multipv_raw", None)
                if prev_idx is not None and records[prev_idx].has_policy
                else None
            )
            if prev_raw is not None:
                sf_p0_regret = _build_sf_p0_regret_vector(
                    np.asarray(prev_raw),
                    policy_encoding=state.game.policy_encoding,
                )

        out.append(
            ReplaySample(
                x=rec.x,
                policy_target=eff_probs,
                wdl_target=int(wdl),
                x_lc0_root=rec.x_lc0_root,
                relations=getattr(rec, "relations", None),
                input_history_encoding=state.game.input_history_encoding,
                history_rep_fix=bool(state.game.history_rep_fix),
                # Value-only fast rows get a neutral priority: rec.priority is a
                # difficulty score from the playout-capped (low-sim) search and
                # is not calibrated against full-ply priorities, so letting it
                # into the surprise-weighted sampler would systematically skew
                # the policy/value row mix.
                priority=float(rec.priority) if row_has_policy else 1.0,
                priority_policy_kl=(
                    None if rec.priority_policy_kl is None else float(rec.priority_policy_kl)
                ),
                priority_q_delta=(
                    None if rec.priority_q_delta is None else float(rec.priority_q_delta)
                ),
                priority_sf_search_gap=sf_search_gap,
                game_id=game_id,
                ply_index=int(rec.ply_index),
                has_policy=row_has_policy,
                sf_wdl=rec.sf_wdl,
                sf_multipv_raw=_padded_sparse_multipv(
                    getattr(rec, "sf_multipv_raw", None),
                    policy_encoding=state.game.policy_encoding,
                ),
                sf_label_meta=getattr(rec, "sf_label_meta", None),
                sf_move_index=(
                    None if rec.sf_move_index is None else
                    policy_index_for_encoding(
                        int(rec.sf_move_index),
                        policy_encoding=state.game.policy_encoding,
                    )
                ),
                sf_played_move_index=(
                    None if rec.sf_played_move_index is None else
                    policy_index_for_encoding(
                        int(rec.sf_played_move_index),
                        policy_encoding=state.game.policy_encoding,
                    )
                ),
                sf_played_rank=(
                    None if rec.sf_played_rank is None else int(rec.sf_played_rank)
                ),
                sf_played_regret=(
                    None if rec.sf_played_regret is None else float(rec.sf_played_regret)
                ),
                future_sf_regret_sum=suffix_regret.total,
                future_sf_regret_d95=suffix_regret.d95,
                future_sf_regret_d98=suffix_regret.d98,
                future_sf_regret_max=suffix_regret.max_single,
                future_sf_regret_h4=suffix_regret.h4,
                future_sf_regret_h6=suffix_regret.h6,
                future_sf_regret_h12=suffix_regret.h12,
                future_sf_regret_h24=suffix_regret.h24,
                future_sf_regret_h50=suffix_regret.h50,
                future_sf_regret_count=suffix_regret.count,
                sf_policy_target=sf_policy_target,
                sf_p0_policy_target=sf_p0_policy_target,
                has_sf_p0=(sf_p0_policy_target is not None),
                sf_p0_regret=sf_p0_regret,
                has_sf_p0_regret=(sf_p0_regret is not None),
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
                search_wdl=rec.search_wdl_est,
                legal_mask=(
                    None if rec.legal_mask is None else
                    policy_mask_to_encoding(
                        rec.legal_mask,
                        policy_encoding=state.game.policy_encoding,
                    ).astype(np.uint8, copy=False)
                ),
                sf_legal_mask=(
                    None if rec.sf_legal_mask is None else
                    policy_mask_to_encoding(
                        rec.sf_legal_mask,
                        policy_encoding=state.game.policy_encoding,
                    ).astype(np.uint8, copy=False)
                ),
                future_legal_mask=future_lmask,
                is_selfplay=is_selfplay_slot,
            ),
        )

    return out


def _resolve_terminal_result(
    state: SelfplayState, i: int,
) -> tuple[str, bool, bool, StockfishResult | None]:
    """Resolve the game's terminal result. Returns (result, was_adjudicated,
    was_tb_adjudicated, sf_res). Updates per-bucket counters in state.stats."""
    cb = state.cboards[i]
    tb_stash = state.tb_result_arr[i]
    was_tb_adjudicated = tb_stash is not None

    if was_tb_adjudicated:
        state.stats.tb_adjudicated_games += 1
        assert tb_stash is not None
        return tb_stash, False, True, None

    result, was_adjudicated, sf_res = _compute_terminal_result(state, cb)
    if was_adjudicated:
        state.stats.adjudicated_games += 1
    return result, was_adjudicated, False, sf_res


def _log_terminal_anomalies(
    state: SelfplayState, i: int, *,
    result: str, game_plies: int, was_adjudicated: bool,
    sf_res: StockfishResult | None,
) -> None:
    """Log unexpectedly short curriculum wins + dump max_plies-game FENs."""
    cb = state.cboards[i]
    b = state.replay_board(i)
    max_plies = int(state.game.max_plies)

    if (
        not state.selfplay_arr[i]
        and result in ("1-0", "0-1")
        and game_plies < max_plies - 10
    ):
        outcome = b.outcome(claim_draw=True)
        term = outcome.termination.name if outcome else "adjudicated"
        _LOG.debug(
            "Short win: %s at %d plies (max=%d), term=%s, adj=%s, is_over=%s",
            result, game_plies, max_plies, term,
            was_adjudicated, cb.is_game_over(),
        )

    if was_adjudicated and game_plies >= max_plies:
        wdl_str = (
            f"{float(sf_res.wdl[0]):.3f}/{float(sf_res.wdl[1]):.3f}/{float(sf_res.wdl[2]):.3f}"
            if (sf_res is not None and sf_res.wdl is not None) else "none"
        )
        _LOG.warning(
            "MAX_PLY_GAME plies=%d result=%s sf_wdl(stm)=%s fen=%s",
            game_plies, result, wdl_str, cb.fen(),
        )


def _emit_completed_game_batch(
    *, samples: list[ReplaySample], counters,
    game_plies: int, is_cm: bool, is_sm: bool,
    was_adjudicated: bool, was_tb_adjudicated: bool,
    sf_d6_sum: float, sf_d6_n: int,
    diff_focus: _DiffFocusGameStats,
    gumbel_policy: _GumbelPolicyGameStats,
    input_history_encoding: str,
    history_rep_fix: bool,
    on_game_complete: Callable[[CompletedGameBatch], None],
) -> None:
    """Build + dispatch CompletedGameBatch for one finalized game."""
    on_game_complete(
        CompletedGameBatch(
            samples=samples,
            input_history_encoding=input_history_encoding,
            history_rep_fix=bool(history_rep_fix),
            positions=len(samples),
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
            sf_d6_sum=float(sf_d6_sum),
            sf_d6_n=int(sf_d6_n),
            diff_focus_records=int(diff_focus.records),
            diff_focus_kept=int(diff_focus.kept),
            diff_focus_keep_prob_sum=float(diff_focus.keep_prob_sum),
            diff_focus_keep_limited=int(diff_focus.keep_limited),
            diff_focus_sample_weight_sum=float(diff_focus.sample_weight_sum),
            diff_focus_sample_weight_limited=int(diff_focus.sample_weight_limited),
            diff_focus_priority_sum=float(diff_focus.priority_sum),
            diff_focus_priority_sq_sum=float(diff_focus.priority_sq_sum),
            diff_focus_priority_min=float(diff_focus.priority_min),
            diff_focus_priority_max=float(diff_focus.priority_max),
            gumbel_policy_diag_n=int(gumbel_policy.records),
            gumbel_policy_top_prob_sum=float(gumbel_policy.top_prob_sum),
            gumbel_policy_action_prob_sum=float(gumbel_policy.action_prob_sum),
            gumbel_policy_entropy_sum=float(gumbel_policy.entropy_sum),
            gumbel_policy_eff_moves_sum=float(gumbel_policy.eff_moves_sum),
            gumbel_policy_candidate_mass_sum=float(gumbel_policy.candidate_mass_sum),
            gumbel_policy_non_candidate_top_prob_sum=float(
                gumbel_policy.non_candidate_top_prob_sum,
            ),
            gumbel_policy_argmax_is_candidate_sum=int(gumbel_policy.argmax_is_candidate_sum),
            gumbel_policy_argmax_is_action_sum=int(gumbel_policy.argmax_is_action_sum),
            gumbel_policy_legal_count_sum=int(gumbel_policy.legal_count_sum),
            gumbel_policy_candidate_count_sum=int(gumbel_policy.candidate_count_sum),
            outcome_stats=dict(counters.outcome_stats or {}),
        ),
    )


def _sf_move_score(cp: int, mate: int) -> float | None:
    """Per-move ordering score for SF regret (higher = better for the mover).

    Mate dominates cp: a forced mate in N is ``100000 - N*100`` (quicker mate
    scores higher), a mate being delivered against the mover is
    ``-100000 - N*100``. Otherwise the cp score is used directly; a SENTINEL cp
    with no mate means the row carries no score and is skipped.
    """
    if mate > 0:
        return 100000.0 - float(mate) * 100.0
    if mate < 0:
        return -100000.0 - float(mate) * 100.0
    if cp == SF_CP_SENTINEL:
        return None
    return float(cp)


def _build_sf_p0_regret_vector(
    rows: np.ndarray | None, *, policy_encoding: str,
) -> np.ndarray | None:
    """Per-move normalized cp-regret vector from raw MultiPV rows (P0 position).

    ``rows`` are ``(K, 5)`` ``[move_idx, cp, mate, w, d]`` in FULL policy space
    (the same layout as ``sf_multipv_raw``). Returns a dense value vector in the
    shard's policy encoding where each present move's entry is
    ``clamp(best_score - score, 0, CAP) / CAP`` (best move -> 0.0). NOT a
    distribution: values are independent per-move regrets in [0, 1], never
    renormalized.

    Legal moves SF surfaced no PV for (legal count > multipv) are worse than
    every scored move but their true regret is unknown, so they default to the
    midpoint between the worst scored regret and the max (1.0) — adaptive to the
    position, parameter-free, and >= every covered move (strictly above it
    unless a covered move already hit the 1.0 cap, in which case both are 1.0).
    When all legal moves are scored, only illegal indices carry this default and
    they are masked out by the legal-masked policy at loss time. (Raise multipv
    if the uncovered tail ever matters.)
    """
    if rows is None:
        return None
    scored: list[tuple[int, float]] = []
    for r in rows.tolist():
        move_idx = int(r[0])
        if move_idx < 0:
            continue
        score = _sf_move_score(int(r[1]), int(r[2]))
        if score is None:
            continue
        scored.append((move_idx, score))
    if not scored:
        return None
    best = max(score for _, score in scored)
    # MultiPV move indices are distinct, so a plain list (not a dedup dict) holds
    # the per-move regrets while we track the worst for the uncovered default.
    covered: list[tuple[int, np.float32]] = []
    worst_regret = 0.0
    for move_idx, score in scored:
        regret_cp = min(max(best - score, 0.0), SF_OWN_REGRET_CAP_CP)
        r = regret_cp / SF_OWN_REGRET_CAP_CP
        covered.append((move_idx, np.float32(r)))
        worst_regret = max(worst_regret, r)
    default_regret = np.float32((worst_regret + 1.0) / 2.0)
    reg_full = np.full((POLICY_SIZE,), default_regret, dtype=np.float32)
    for move_idx, r in covered:
        reg_full[move_idx] = r
    # Remap full -> shard encoding by GATHER only (value vector, never
    # renormalized — policy_vector_to_encoding would renormalize to sum 1).
    return policy_mask_to_encoding(reg_full, policy_encoding=policy_encoding)


def _padded_sparse_multipv(
    rows: np.ndarray | None, *, policy_encoding: str,
) -> np.ndarray | None:
    """Pad raw MultiPV rows to the shard's fixed width and convert the move
    column from full policy space to the shard's policy encoding (same
    convention as ``sf_move_index``)."""
    if rows is None:
        return None
    out = np.empty((SF_MULTIPV_RAW_MAX, rows.shape[1]), dtype=np.int16)
    out[:] = np.asarray(SF_MULTIPV_PAD_ROW, dtype=np.int16)
    k = min(int(rows.shape[0]), SF_MULTIPV_RAW_MAX)
    out[:k] = rows[:k]
    for j in range(k):
        out[j, 0] = policy_index_for_encoding(
            int(rows[j, 0]), policy_encoding=policy_encoding,
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
    game_plies = int(cb.ply)
    state.stats.total_game_plies += game_plies

    is_cm = cb.is_checkmate()
    is_sm = cb.is_stalemate()
    if is_cm:
        state.stats.checkmate_games += 1
    elif is_sm:
        state.stats.stalemate_games += 1

    result, was_adjudicated, was_tb_adjudicated, sf_res = _resolve_terminal_result(
        state, i,
    )
    _log_terminal_anomalies(
        state, i,
        result=result, game_plies=game_plies,
        was_adjudicated=was_adjudicated, sf_res=sf_res,
    )

    records = state.samples_per_game[i]
    flush_async_sf_labels_for_records(state, records)
    result, tb_policy_overrides = _rescore_with_syzygy(state, i, b, records, result)
    counters = _update_aggregate_stats(
        state, i, result=result,
        was_adjudicated=was_adjudicated, game_plies=game_plies,
    )

    ply_to_index = {int(rec.ply_index): idx for idx, rec in enumerate(records)}
    sf_d6_sum_before = state.stats.sf_d6_sum
    sf_d6_n_before = state.stats.sf_d6_n
    vol_targets, sf_vol_targets = _compute_volatility_and_sf_delta(
        state, records, ply_to_index,
    )
    game_sf_d6_sum = state.stats.sf_d6_sum - sf_d6_sum_before
    game_sf_d6_n = state.stats.sf_d6_n - sf_d6_n_before

    sample_start = len(all_samples)
    all_samples.extend(_build_replay_samples(
        state, i, records,
        result=result,
        tb_policy_overrides=tb_policy_overrides,
        vol_targets=vol_targets,
        sf_vol_targets=sf_vol_targets,
        total_plies_played=int(cb.ply),
        ply_to_index=ply_to_index,
    ))

    if on_game_complete is not None:
        game_samples = list(all_samples[sample_start:])
        diff_focus_stats = _compute_diff_focus_game_stats(records, game_samples)
        gumbel_policy_stats = _compute_gumbel_policy_game_stats(records)
        _update_diff_focus_stats(state, diff_focus_stats)
        _update_gumbel_policy_stats(state, gumbel_policy_stats)
        if game_samples:
            _emit_completed_game_batch(
                samples=game_samples, counters=counters,
                game_plies=game_plies, is_cm=is_cm, is_sm=is_sm,
                was_adjudicated=was_adjudicated,
                was_tb_adjudicated=was_tb_adjudicated,
                sf_d6_sum=game_sf_d6_sum, sf_d6_n=game_sf_d6_n,
                diff_focus=diff_focus_stats,
                gumbel_policy=gumbel_policy_stats,
                input_history_encoding=state.game.input_history_encoding,
                history_rep_fix=bool(state.game.history_rep_fix),
                on_game_complete=on_game_complete,
            )
    else:
        game_samples = list(all_samples[sample_start:])
        _update_diff_focus_stats(
            state, _compute_diff_focus_game_stats(records, game_samples),
        )
        _update_gumbel_policy_stats(
            state, _compute_gumbel_policy_game_stats(records),
        )

    # In continuous mode, samples flow through ``on_game_complete`` and
    # ``all_samples`` would otherwise grow without bound.
    if state.continuous:
        all_samples.clear()


__all__ = ["finalize_game"]
