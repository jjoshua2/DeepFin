from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from chess_anti_engine.inference import BatchEvaluator
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishUCI

try:
    from chess_anti_engine.mcts.puct_c import run_mcts_many_c as _run_mcts_many_c
    _HAS_C_TREE = True
except ImportError:
    _HAS_C_TREE = False

try:
    from chess_anti_engine.mcts.gumbel_c import (
        run_gumbel_root_many_c as _run_gumbel_root_many_c,
    )
    _HAS_GUMBEL_C = True
except ImportError:
    _HAS_GUMBEL_C = False

if TYPE_CHECKING:
    from chess_anti_engine.mcts.gumbel_c import (
        run_gumbel_root_many_c as _run_gumbel_root_many_c,  # noqa: F401,F811
    )
    from chess_anti_engine.mcts.puct_c import (
        run_mcts_many_c as _run_mcts_many_c,  # noqa: F401,F811
    )
import chess

from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.finalize import finalize_game
from chess_anti_engine.selfplay.network_turn import run_network_turn
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.state import (
    BatchStats,
    CompletedGameBatch,
    SelfplayState,
)
from chess_anti_engine.selfplay.stockfish_turn import (
    finish_sf_annotation_and_moves,
    submit_sf_queries,
)
from chess_anti_engine.selfplay.temperature import (
    apply_policy_temperature,
)
from chess_anti_engine.tablebase import tb_adjudicate_result


# Backward-compatible alias: tests and bench scripts import this name directly.
# The implementation lives in ``selfplay.temperature`` as ``apply_policy_temperature``.
_apply_temperature = apply_policy_temperature


def play_batch(
    model: torch.nn.Module | None,
    *,
    device: str,
    rng: np.random.Generator,
    stockfish: StockfishUCI | StockfishPool,
    evaluator: BatchEvaluator | None = None,
    games: int,
    target_games: int = 0,
    on_game_complete: Callable[[CompletedGameBatch], None] | None = None,
    on_step: Callable[[], None] | None = None,
    stop_fn: Callable[[], bool] | None = None,
    # Config groups (frozen dataclasses with sensible defaults).
    opponent: OpponentConfig = OpponentConfig(),
    temp: TemperatureConfig = TemperatureConfig(),
    search: SearchConfig = SearchConfig(),
    opening: OpeningConfig = OpeningConfig(),
    diff_focus: DiffFocusConfig = DiffFocusConfig(),
    game: GameConfig = GameConfig(),
) -> tuple[list[ReplaySample], BatchStats]:
    """Play a batch of games.

    Design goals:
    - keep GPU busy via batched inference
    - keep SF queries minimal (one per SF ply)
    - compute volatility targets from a consistent network-side WDL series without per-ply overhead

    Continuous mode (stop_fn provided, target_games=0): runs forever with all
    finished slots recycled immediately, until stop_fn() returns True.  Samples
    are delivered incrementally via on_game_complete; the returned sample list
    is empty to avoid unbounded memory growth.

    Finite mode (target_games > 0 or stop_fn is None): plays exactly
    target_games (or `games` if target_games=0) then returns all samples.
    """

    requested_batch = int(games)
    continuous = stop_fn is not None and int(target_games) <= 0
    target = int(target_games) if int(target_games) > 0 else requested_batch
    batch_size = min(requested_batch, target)
    if batch_size <= 0:
        raise ValueError("play_batch requires at least one game")

    state = SelfplayState.create(
        model=model,
        device=device,
        rng=rng,
        stockfish=stockfish,
        evaluator=evaluator,
        batch_size=batch_size,
        continuous=continuous,
        target=target,
        opponent=opponent,
        temp=temp,
        search=search,
        opening=opening,
        diff_focus=diff_focus,
        game=game,
    )

    all_samples: list[ReplaySample] = []

    # ── Main game loop (rolling batch) ────────────────────────────────────────
    if continuous:
        max_steps = 2**62  # effectively infinite; stop_fn controls exit
    else:
        max_steps = int(target) * (int(game.max_plies) // 2 + 2)  # safety bound

    def _tb_adjudicate_active_games() -> int:
        """Scan active games; if any current position is TB-eligible and
        probable AND this game's per-game roll said "adjudicate", mark the
        game done and stash the TB-proven result. Runs once per step;
        per-active-game cost is ~10µs after the popcount prefilter rejects
        non-endgame positions.

        Games whose roll landed on "play through" (controlled by
        ``syzygy_adjudicate_fraction``) are skipped here and finish
        naturally, so the NN continues training on endgame positions with
        its own labels rather than losing endgame skill entirely.
        """
        assert state.tb_probe is not None and game.syzygy_path is not None
        max_p = state.tb_probe.max_pieces
        adjudicated = 0
        for i in range(batch_size):
            if state.done_arr[i] or state.finalized_arr[i] or state.tb_result_arr[i] is not None:
                continue
            if not state.tb_adj_roll_arr[i]:
                continue
            cb = state.cboards[i]
            occ = int(cb.occ_white) | int(cb.occ_black)
            if occ.bit_count() > max_p or int(cb.castling) != 0:
                continue
            board = chess.Board(cb.fen())
            result = tb_adjudicate_result(board, game.syzygy_path)
            if result is not None:
                state.tb_result_arr[i] = result
                state.done_arr[i] = 1
                adjudicated += 1
        return adjudicated
    _t_net = 0.0
    _t_sf = 0.0

    for _step in range(max_steps):  # skylos: ignore (_step loop var unused by convention)
        # Allow caller to update model/evaluator between moves.
        if on_step is not None:
            on_step()
        if stop_fn is not None and stop_fn():
            break

        # Tablebase adjudication before the classify pass. Any game that's
        # now TB-eligible gets marked done; the classify will then skip it
        # and the finalize path uses the stashed TB result. Runs at most
        # once per game — the ``state.tb_result_arr`` stash is the idempotency key.
        if state.tb_probe is not None and game.syzygy_adjudicate:
            _tb_adjudicate_active_games()

        net_idxs, selfplay_opp_idxs, curriculum_opp_idxs, all_done = state.classify_active_slots()
        if all_done:
            break

        # Submit SF queries for curriculum games FIRST — curriculum boards
        # are disjoint from net/selfplay boards, so we can overlap SF I/O
        # with the full combined network turn below.
        _sf_futures: dict[int, Any] | None = None
        if curriculum_opp_idxs and isinstance(state.stockfish, StockfishPool):
            _t0 = time.time()
            _sf_futures = submit_sf_queries(state, curriculum_opp_idxs)
            _t_sf += time.time() - _t0

        # Combined network turn: merge net_idxs + selfplay_opp_idxs into a
        # single MCTS call.  This doubles the GPU batch size (256 vs 128),
        # giving much better GPU utilization and pipeline overlap.  Both
        # sets are disjoint and independent — same MCTS treatment, same
        # per-ply processing.
        _combined_net_idxs = net_idxs + selfplay_opp_idxs
        if _combined_net_idxs:
            _t0 = time.time()
            run_network_turn(state, _combined_net_idxs)
            _t_net += time.time() - _t0

        # Submit selfplay SF queries immediately after network turn
        # (boards now have the move pushed).  These run in the SF pool
        # while we collect curriculum results below.
        _sf_sp_futures: dict[int, Any] | None = None
        if selfplay_opp_idxs and isinstance(state.stockfish, StockfishPool):
            _t0 = time.time()
            _sf_sp_futures = submit_sf_queries(state, selfplay_opp_idxs)
            _t_sf += time.time() - _t0

        # Collect curriculum SF results (overlapped with combined net turn)
        if curriculum_opp_idxs:
            _t0 = time.time()
            finish_sf_annotation_and_moves(state, curriculum_opp_idxs, play_curriculum_moves=True, futures=_sf_futures)
            _t_sf += time.time() - _t0

        # Collect selfplay SF move results (submitted above, overlapped with curriculum)
        if selfplay_opp_idxs:
            _t0 = time.time()
            finish_sf_annotation_and_moves(state, selfplay_opp_idxs, play_curriculum_moves=True, futures=_sf_sp_futures)
            _t_sf += time.time() - _t0
            # Submit label queries immediately; collect after (overlaps with
            # finalization below for negligible extra latency).
            selfplay_label_idxs = [i for i in selfplay_opp_idxs if not state.done_arr[i]]
            _sf_label_futures: dict[int, Any] | None = None
            if selfplay_label_idxs and isinstance(state.stockfish, StockfishPool):
                _t0 = time.time()
                _sf_label_futures = submit_sf_queries(state, selfplay_label_idxs)
                _t_sf += time.time() - _t0
            if selfplay_label_idxs:
                _t0 = time.time()
                finish_sf_annotation_and_moves(state, selfplay_label_idxs, play_curriculum_moves=False, futures=_sf_label_futures)
                _t_sf += time.time() - _t0

        # Finalize completed games and optionally recycle slots
        for i in range(batch_size):
            if state.done_arr[i] and not state.finalized_arr[i]:
                finalize_game(state, i, all_samples, on_game_complete)
                state.finalized_arr[i] = 1
                state.games_completed += 1
                if continuous or state.games_started < target:
                    state.recycle_slot(i)

        # Reset tree when it gets too large and no roots reference old nodes
        if state.mcts_tree is not None and state.mcts_tree.node_count() > 500_000:
            if all(rid < 0 for rid in state.root_ids):
                state.mcts_tree.reset()

    # ── Timing summary ─────────────────────────────────────────────────────────
    logging.getLogger("chess_anti_engine.worker").info(
        "play_batch timing: net=%.1fs sf=%.1fs (net %.0f%%, sf %.0f%%)",
        _t_net, _t_sf,
        _t_net / max(0.001, _t_net + _t_sf) * 100,
        _t_sf / max(0.001, _t_net + _t_sf) * 100,
    )
    if state.nn_cache is not None:
        _nc_stats = state.nn_cache.stats()
        _nc_total = _nc_stats["hits"] + _nc_stats["misses"]
        _nc_hit_rate = _nc_stats["hits"] / _nc_total if _nc_total > 0 else 0.0
        logging.getLogger("chess_anti_engine.worker").info(
            "nncache: hits=%d misses=%d (hit_rate=%.1f%%)  "
            "inserts=%d collisions=%d  count=%d/%d (%.1f%% full)",
            _nc_stats["hits"], _nc_stats["misses"], 100.0 * _nc_hit_rate,
            _nc_stats["inserts"], _nc_stats["insert_collisions"],
            _nc_stats["count"], _nc_stats["cap"],
            100.0 * _nc_stats["count"] / max(1, _nc_stats["cap"]),
        )

    # ── Return results ────────────────────────────────────────────────────────
    sf_nodes = int(getattr(stockfish, "nodes", 0) or 0)
    return all_samples, state.stats.to_batch_stats(
        games=state.games_completed,
        positions=len(all_samples),
        sf_nodes=sf_nodes if sf_nodes > 0 else None,
    )
