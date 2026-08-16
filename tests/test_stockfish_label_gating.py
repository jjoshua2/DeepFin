from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.selfplay.config import GameConfig
from chess_anti_engine.selfplay.config import OpponentConfig
from chess_anti_engine.selfplay.state import _NetRecord
from chess_anti_engine.selfplay.stockfish_turn import (
    _collect_sf_pv_candidates,
    _eff_sf_nodes,
    _process_sf_results,
    flush_async_sf_labels_for_records,
    finish_pending_curriculum_moves,
    submit_async_curriculum_move_queries,
    submit_async_sf_labels_from_curriculum_moves,
    submit_async_sf_label_queries,
    submit_sf_queries,
)
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishPV
from chess_anti_engine.stockfish.uci import StockfishResult
from chess_anti_engine.stockfish.wdl import cp_to_wdl


class _FakeCBoard:
    turn = True
    occ_white = 0
    occ_black = 0
    castling = 0

    def __init__(self, legal_indices: np.ndarray | None = None) -> None:
        self.pushed: list[int] = []
        self._legal_indices = (
            legal_indices if legal_indices is not None
            else np.array([0], dtype=np.int64)
        )

    def fen(self) -> str:
        return chess.STARTING_FEN

    def legal_move_indices(self) -> np.ndarray:
        return self._legal_indices

    def push_index(self, idx: int) -> None:
        self.pushed.append(int(idx))
        self.turn = not self.turn

    def is_game_over(self) -> bool:
        return False


class _FakePool(StockfishPool):
    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        self.calls: list[dict] = []

    def submit(
        self, fen: str, *, nodes=None, syzygy_path=None, fresh: bool = False,
        searchmoves=None,
    ):
        # `searchmoves` is RECORDED and HONOURED, not accepted and dropped. A
        # fake that swallowed it would let a caller lose the restriction while
        # every test here still passed — the exact defect the real pool's
        # end-to-end test exists to catch, re-introduced one layer down.
        self.calls.append({
            "fen": fen, "nodes": nodes, "syzygy_path": syzygy_path,
            "fresh": fresh, "searchmoves": None if searchmoves is None else list(searchmoves),
        })
        fut: Future = Future()
        best = list(searchmoves)[0] if searchmoves else "a2a3"
        fut.set_result(StockfishResult(
            bestmove_uci=best, wdl=np.array([0.0, 1.0, 0.0]), pvs=[],
        ))
        return fut


def _record(*, has_policy: bool) -> _NetRecord:
    return _NetRecord(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_probs=np.zeros((POLICY_SIZE,), dtype=np.float32),
        net_wdl_est=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        search_wdl_est=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        pov_color=chess.WHITE,
        ply_index=0,
        has_policy=has_policy,
        priority=1.0,
        sample_weight=1.0,
        keep_prob=1.0,
        legal_mask=np.zeros((POLICY_SIZE,), dtype=np.uint8),
    )


def _state(
    *,
    has_policy: bool,
    sf_move_nodes: int = 0,
    legal_indices: np.ndarray | None = None,
    game: GameConfig | None = None,
    opponent: OpponentConfig | None = None,
) -> Any:
    cboard = _FakeCBoard(legal_indices=legal_indices)
    return SimpleNamespace(
        batch_size=1,
        pending_sf_labels=[],
        pending_sf_moves={},
        stockfish=_FakePool(),
        samples_per_game=[[_record(has_policy=has_policy)]],
        cboards=[cboard],
        base_nodes=100,
        last_net_full=[True],
        tb_adj_roll_arr=[True],
        game=game if game is not None else GameConfig(sf_move_nodes=sf_move_nodes),
        rng=np.random.default_rng(0),
        opponent=opponent if opponent is not None else OpponentConfig(),
        move_idx_history=[[]],
        mcts_tree=None,
        root_ids=[-1],
        done_arr=np.zeros((1,), dtype=np.int8),
        finalized_arr=np.zeros((1,), dtype=np.int8),
        selfplay_arr=np.zeros((1,), dtype=np.int8),
        # Production SelfplayState always has this; zeros = not in SF-refute phase.
        sf_refute_opp_plies_left=np.zeros((1,), dtype=np.int32),
    )


def test_async_sf_labels_skip_records_discarded_by_finalize() -> None:
    state = _state(has_policy=False)

    submitted = submit_async_sf_label_queries(state, [0])

    assert submitted == 0
    assert state.pending_sf_labels == []
    assert state.stockfish.calls == []


def test_async_sf_labels_submit_for_replay_kept_records() -> None:
    state = _state(has_policy=True)

    submitted = submit_async_sf_label_queries(state, [0])

    assert submitted == 1
    assert len(state.pending_sf_labels) == 1
    assert state.stockfish.calls[0]["nodes"] == 100


def test_finalize_flush_waits_for_pending_replay_kept_labels() -> None:
    state = _state(has_policy=True)
    rec = state.samples_per_game[0][0]
    submitted = submit_async_sf_label_queries(state, [0])

    attached, failed = flush_async_sf_labels_for_records(state, [rec])

    assert submitted == 1
    assert (attached, failed) == (1, 0)
    assert state.pending_sf_labels == []
    assert rec.sf_policy_target is not None
    assert rec.sf_move_index is not None
    assert rec.sf_wdl is not None


def test_label_nodes_cap_applies_to_selfplay_label_queries() -> None:
    state = _state(has_policy=True, game=GameConfig(sf_label_nodes_cap=40))

    submitted = submit_async_sf_label_queries(state, [0])

    assert submitted == 1
    assert state.stockfish.calls[0]["nodes"] == 40


def test_label_nodes_cap_never_raises_the_budget() -> None:
    state = _state(has_policy=True, game=GameConfig(sf_label_nodes_cap=10_000))

    submit_async_sf_label_queries(state, [0])

    assert state.stockfish.calls[0]["nodes"] == 100  # min(base, cap)


def test_label_nodes_cap_leaves_curriculum_moves_at_full_budget() -> None:
    state = _state(has_policy=True, game=GameConfig(sf_label_nodes_cap=40))

    submit_async_curriculum_move_queries(state, [0])

    assert state.stockfish.calls[0]["nodes"] == 100


def test_label_nodes_cap_off_by_default() -> None:
    state = _state(has_policy=True)

    submit_async_sf_label_queries(state, [0])

    assert state.stockfish.calls[0]["nodes"] == 100


def test_curriculum_move_queries_can_use_separate_low_node_budget() -> None:
    state = _state(has_policy=True, sf_move_nodes=10)

    submit_sf_queries(state, [0], for_move=True)

    assert state.stockfish.calls[0]["nodes"] == 10


def test_pending_curriculum_move_applies_without_stamping_low_node_label() -> None:
    state = _state(has_policy=True, sf_move_nodes=10)
    rec = state.samples_per_game[0][0]

    submitted = submit_async_curriculum_move_queries(state, [0])
    completed = finish_pending_curriculum_moves(state, block=False)

    assert submitted == 1
    assert completed == 1
    assert state.pending_sf_moves == {}
    assert state.cboards[0].pushed == [0]
    assert state.move_idx_history[0] == [0]
    assert rec.sf_policy_target is None
    assert rec.sf_move_index is None


def test_curriculum_move_future_reused_for_label_when_nodes_are_shared() -> None:
    state = _state(has_policy=True, sf_move_nodes=0)
    rec = state.samples_per_game[0][0]

    submitted_moves = submit_async_curriculum_move_queries(state, [0])
    submitted_labels = submit_async_sf_labels_from_curriculum_moves(state, [0])
    attached, failed = flush_async_sf_labels_for_records(state, [rec])

    assert submitted_moves == 1
    assert submitted_labels == 1
    assert (attached, failed) == (1, 0)
    assert len(state.stockfish.calls) == 1
    assert rec.sf_policy_target is not None


def test_curriculum_label_reuse_refused_below_the_floor() -> None:
    """PR #354 review H1: with sf_move_nodes=0 (production) the curriculum
    label REUSES the move future, which was submitted for_move and takes
    neither cap nor floor — so on the production config the floor would
    silently miss the curriculum half of the fleet's labels. Below the floor
    the reuse must be refused and a fresh floored label query paid for."""
    state = _state(
        has_policy=True,
        game=GameConfig(sf_move_nodes=0, sf_label_nodes_floor=700_000),
    )
    rec = state.samples_per_game[0][0]

    submitted_moves = submit_async_curriculum_move_queries(state, [0])
    submitted_labels = submit_async_sf_labels_from_curriculum_moves(state, [0])
    attached, failed = flush_async_sf_labels_for_records(state, [rec])

    assert submitted_moves == 1
    assert submitted_labels == 1
    assert (attached, failed) == (1, 0)
    # Move at the PID budget, label at the floor: two engine calls, no reuse.
    assert [call["nodes"] for call in state.stockfish.calls] == [100, 700_000]
    assert rec.sf_policy_target is not None


def test_curriculum_label_reuse_kept_when_budget_meets_the_floor() -> None:
    """The reuse stays free when the PID budget already satisfies the floor —
    pinned AT the boundary (base == floor), so the guard's `<` cannot drift to
    `<=` and silently start paying for a query the reuse already covers."""
    state = _state(
        has_policy=True,
        game=GameConfig(sf_move_nodes=0, sf_label_nodes_floor=100),
    )
    rec = state.samples_per_game[0][0]

    submit_async_curriculum_move_queries(state, [0])
    submit_async_sf_labels_from_curriculum_moves(state, [0])
    attached, failed = flush_async_sf_labels_for_records(state, [rec])

    assert (attached, failed) == (1, 0)
    assert len(state.stockfish.calls) == 1  # reused, base 100 == floor 100


def test_eff_sf_nodes_label_floor_is_absolute_even_on_fast_plies() -> None:
    """PR #354 review L1: the floor is applied AFTER the fast-ply scale, so
    the >= floor guarantee cannot be silently multiplied down to floor*0.25.
    Unreachable in production today (labels only attach to full plies), but
    the guarantee must not depend on an invariant maintained two modules away
    in network_turn.py."""
    state = _eff_state(500_000, last_full=False)
    state.game.sf_label_nodes_floor = 700_000
    assert _eff_sf_nodes(state, 0, for_label=True) == 700_000


def test_curriculum_label_uses_separate_query_when_move_nodes_are_low() -> None:
    state = _state(has_policy=True, sf_move_nodes=10)
    rec = state.samples_per_game[0][0]

    submitted_moves = submit_async_curriculum_move_queries(state, [0])
    submitted_labels = submit_async_sf_labels_from_curriculum_moves(state, [0])
    attached, failed = flush_async_sf_labels_for_records(state, [rec])

    assert submitted_moves == 1
    assert submitted_labels == 1
    assert (attached, failed) == (1, 0)
    assert [call["nodes"] for call in state.stockfish.calls] == [10, 100]


def test_sf_pv_candidates_use_native_wdl_by_default() -> None:
    move = "a2a3"
    move_idx = uci_to_policy_index(move, True)
    res = SimpleNamespace(
        pvs=[
            StockfishPV(
                move_uci=move,
                wdl=np.array([0.80, 0.10, 0.10], dtype=np.float32),
                cp=0,
            )
        ]
    )

    cand_idxs, cand_scores = _collect_sf_pv_candidates(
        res,
        _turn=True,
        legal_set={move_idx},
    )

    assert cand_idxs == [move_idx]
    assert cand_scores == pytest.approx([0.85])


def test_sf_pv_candidates_use_cp_logistic_when_enabled() -> None:
    move = "a2a3"
    move_idx = uci_to_policy_index(move, True)
    res = SimpleNamespace(
        pvs=[
            StockfishPV(
                move_uci=move,
                wdl=np.array([0.80, 0.10, 0.10], dtype=np.float32),
                cp=0,
            )
        ]
    )

    cand_idxs, cand_scores = _collect_sf_pv_candidates(
        res,
        _turn=True,
        legal_set={move_idx},
        sf_wdl_use_cp_logistic=True,
        sf_wdl_cp_slope=0.00875,
        sf_wdl_cp_draw_width=75.0,
    )

    wdl = cp_to_wdl(0, None, slope=0.00875, draw_width_cp=75.0)
    expected = float(wdl[0]) + 0.5 * float(wdl[1])
    assert cand_idxs == [move_idx]
    assert cand_scores == pytest.approx([expected])
    assert cand_scores != pytest.approx([0.85])


def test_cp_logistic_policy_scores_do_not_change_curriculum_regret_choice() -> None:
    native_best = "a2a3"
    logistic_best = "a2a4"
    native_best_idx = uci_to_policy_index(native_best, True)
    logistic_best_idx = uci_to_policy_index(logistic_best, True)
    state = _state(
        has_policy=True,
        legal_indices=np.array([native_best_idx, logistic_best_idx], dtype=np.int64),
        game=GameConfig(
            sf_wdl_use_cp_logistic=True,
            sf_wdl_cp_slope=0.00875,
            sf_wdl_cp_draw_width=75.0,
            sf_policy_temp=0.1,
            sf_policy_label_smooth=0.0,
        ),
        opponent=OpponentConfig(wdl_regret_limit=0.0),
    )
    rec = state.samples_per_game[0][0]
    res = StockfishResult(
        bestmove_uci=native_best,
        wdl=np.array([0.80, 0.10, 0.10], dtype=np.float32),
        pvs=[
            StockfishPV(
                move_uci=native_best,
                wdl=np.array([0.80, 0.10, 0.10], dtype=np.float32),
                cp=-200,
            ),
            StockfishPV(
                move_uci=logistic_best,
                wdl=np.array([0.10, 0.10, 0.80], dtype=np.float32),
                cp=200,
            ),
        ],
    )

    _process_sf_results(
        state,
        [0],
        results={0: res},
        play_curriculum_moves=True,
        attach_labels=True,
    )

    assert state.cboards[0].pushed == [native_best_idx]
    assert rec.sf_policy_target is not None
    assert rec.sf_policy_target[logistic_best_idx] > rec.sf_policy_target[native_best_idx]


def _eff_state(
    base_nodes: int, *, last_full: bool, sf_move_nodes: int = 0, fast_scale: float = 0.25,
) -> Any:
    """Minimal stand-in exposing just what _eff_sf_nodes reads."""
    return SimpleNamespace(
        base_nodes=base_nodes,
        game=SimpleNamespace(
            sf_move_nodes=sf_move_nodes, sf_fast_ply_node_scale=fast_scale,
        ),
        last_net_full=[last_full],
    )


def test_eff_sf_nodes_full_ply_uses_full_budget():
    """Full-sim plies (last_net_full True) always use the configured budget,
    for both the played move and labels. Labels only ever attach to full plies,
    so this is the path every training label takes."""
    assert _eff_sf_nodes(_eff_state(500_000, last_full=True), 0, for_move=True) == 500_000
    assert _eff_sf_nodes(_eff_state(500_000, last_full=True), 0, for_move=False) == 500_000


def test_eff_sf_nodes_fast_ply_scaled_by_knob():
    """Fast-sim plies scale by sf_fast_ply_node_scale. Default 0.25 is the
    long-standing intended value (opponent plays cheaply on throwaway plies);
    the knob lets it be raised toward 1.0 for a more consistent opponent."""
    # Default 0.25 -> 125000.
    assert _eff_sf_nodes(_eff_state(500_000, last_full=False), 0, for_move=True) == 125_000
    assert _eff_sf_nodes(_eff_state(500_000, last_full=False), 0, for_move=False) == 125_000
    # Knob honored: 0.5 doubles the fast-ply budget...
    assert _eff_sf_nodes(
        _eff_state(500_000, last_full=False, fast_scale=0.5), 0, for_move=True
    ) == 250_000
    # ...and 1.0 fully decouples (full strength even on fast plies).
    assert _eff_sf_nodes(
        _eff_state(500_000, last_full=False, fast_scale=1.0), 0, for_move=False
    ) == 500_000


def test_eff_sf_nodes_move_node_override_then_scaled():
    """sf_move_nodes overrides the base for moves, then the fast-ply scale still
    applies (the move budget is the thing being scaled)."""
    assert _eff_sf_nodes(
        _eff_state(500_000, last_full=False, sf_move_nodes=20_000), 0, for_move=True
    ) == 5_000  # 20000 * 0.25
    assert _eff_sf_nodes(
        _eff_state(500_000, last_full=True, sf_move_nodes=20_000), 0, for_move=True
    ) == 20_000


def test_eff_sf_nodes_zero_budget_returns_none():
    assert _eff_sf_nodes(_eff_state(0, last_full=True), 0, for_move=True) is None
    assert _eff_sf_nodes(_eff_state(0, last_full=False), 0, for_move=False) is None


def test_eff_sf_nodes_rejects_move_and_label_combination() -> None:
    state = _state(has_policy=True)
    with pytest.raises(ValueError, match="cannot be both a move and a label"):
        _eff_sf_nodes(state, 0, for_move=True, for_label=True)


def test_generic_submit_threads_label_cap() -> None:
    state = _state(has_policy=True, game=GameConfig(sf_label_nodes_cap=40))

    submit_sf_queries(state, [0], for_label=True)

    assert state.stockfish.calls[0]["nodes"] == 40


def test_generic_submit_threads_label_floor() -> None:
    """The floor must reach the actual engine query, not just the config: a
    label query at a PID base budget below the floor (base_nodes=100 here) is
    raised to the floor. This is the production defect the knob exists for —
    the 2026-08-04 restart's labels silently rode the 50k opponent budget."""
    state = _state(has_policy=True, game=GameConfig(sf_label_nodes_floor=700_000))

    submit_sf_queries(state, [0], for_label=True)

    assert state.stockfish.calls[0]["nodes"] == 700_000


def test_label_floor_ignores_moves_and_wins_over_cap() -> None:
    """The floor applies to labels only (curriculum moves keep the PID budget),
    and on a conflicting config (floor > cap > 0) the floor wins because it is
    applied after the cap."""
    game = GameConfig(sf_label_nodes_floor=700_000)
    state = _state(has_policy=True, game=game)
    submit_sf_queries(state, [0], for_move=True)
    assert state.stockfish.calls[0]["nodes"] == 100  # base budget, not floored

    conflicted = GameConfig(sf_label_nodes_cap=40, sf_label_nodes_floor=700_000)
    state = _state(has_policy=True, game=conflicted)
    submit_sf_queries(state, [0], for_label=True)
    assert state.stockfish.calls[0]["nodes"] == 700_000


def test_label_cap_warns_when_native_wdl_labels(caplog) -> None:
    """The cap's cost-free rationale assumes cp-logistic labels; the config
    warns when the cap is combined with SF-native WDL labels."""
    import logging

    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.selfplay.config"):
        GameConfig(sf_label_nodes_cap=150_000, sf_wdl_use_cp_logistic=False)
    assert any("sf_label_nodes_cap" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.selfplay.config"):
        GameConfig(sf_label_nodes_cap=150_000, sf_wdl_use_cp_logistic=True)
    assert not caplog.records


def test_label_floor_over_cap_warns(caplog) -> None:
    """floor > cap > 0 makes the cap dead (floor is applied after); the config
    warns so the conflict is visible at construction."""
    import logging

    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.selfplay.config"):
        GameConfig(sf_label_nodes_cap=100_000, sf_label_nodes_floor=700_000)
    assert any("sf_label_nodes_floor" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.selfplay.config"):
        GameConfig(sf_label_nodes_floor=700_000)
    assert not caplog.records
