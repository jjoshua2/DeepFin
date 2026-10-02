"""Native suspend/resume/submit/poll/flush/replay lifecycle, with controlled SF."""
from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path
from typing import Any, cast

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig, GameConfig, OpponentConfig, SearchConfig, TemperatureConfig,
)
from chess_anti_engine.selfplay.finalize import _build_replay_samples
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.resume import (
    RESUME_FILE_SUFFIX, resume_inflight_games, suspend_inflight_games,
)
from chess_anti_engine.selfplay.state import SelfplayState, _NetRecord
from chess_anti_engine.selfplay.stockfish_turn import (
    flush_async_sf_labels_for_records, has_pending_sf_labels_for_records,
    poll_async_sf_labels, submit_async_sf_label_queries,
    submit_async_sf_labels_from_curriculum_moves,
)
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishPV, StockfishResult


class _ManualPool(StockfishPool):
    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        self.calls: list[tuple[str, Future, int | None]] = []
        self.nodes = 100
        self.reject = False
        self.immediate = False
        self.fresh_calls: list[bool] = []

    def submit(self, fen: str, *, nodes=None, syzygy_path=None,
               fresh=False, searchmoves=None):
        del syzygy_path, searchmoves
        self.fresh_calls.append(bool(fresh))
        if self.reject:
            raise RuntimeError("controlled submit failure")
        fut: Future = Future()
        self.calls.append((fen, fut, nodes))
        if self.immediate:
            fut.set_result(_result(fen))
        return fut

    def complete(self) -> None:
        for fen, fut, _nodes in self.calls:
            if not fut.done():
                fut.set_result(_result(fen))


def _result(fen: str) -> StockfishResult:
    board = chess.Board(fen)
    move = next(iter(board.legal_moves)).uci()
    # Different POVs and positions cannot accidentally compare equal.
    wdl = np.array([0.1, 0.2, 0.7] if board.turn else [0.6, 0.3, 0.1],
                   dtype=np.float32)
    return StockfishResult(bestmove_uci=move, wdl=wdl,
                           pvs=[StockfishPV(move, wdl, cp=17)], cp=17,
                           nodes=100, depth=4)


def _state(pool: _ManualPool, *, batch_size: int = 1) -> SelfplayState:
    return SelfplayState.create(
        model=None, evaluator=cast(Any, object()), device="cpu",
        rng=np.random.default_rng(0), stockfish=pool, batch_size=batch_size,
        continuous=False, target=1, opponent=OpponentConfig(),
        temp=TemperatureConfig(), search=SearchConfig(simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(min_keep=1.0),
        game=GameConfig(selfplay_fraction=1.0, syzygy_adjudicate=False),
    )


def _move(state: SelfplayState, uci: str, *, slot: int = 0, has_policy: bool = True,
          refute: bool = False) -> _NetRecord:
    cb = state.cboards[slot]
    idx = int(uci_to_policy_index(uci, bool(cb.turn)))
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    policy[idx] = 1
    rec = _NetRecord(
        x=encode_cboard(cb, input_history_encoding=state.game.input_history_encoding,
                        input_extra_features=state.game.input_extra_features),
        policy_probs=policy, net_wdl_est=np.array([0., 1., 0.], dtype=np.float32),
        search_wdl_est=np.array([0., 1., 0.], dtype=np.float32),
        pov_color=bool(cb.turn), ply_index=int(cb.ply), has_policy=has_policy,
        priority=1., sample_weight=1., keep_prob=1.,
        move_offset=len(state.move_idx_history[slot]), pos_hash=int(cb.zobrist_hash),
    )
    rec.is_sf_refute_opp = refute
    state.samples_per_game[slot].append(rec)
    cb.push_index(idx)
    state.move_idx_history[slot].append(idx)
    return rec


def _suspend(state: SelfplayState, path: Path) -> Path:
    report = suspend_inflight_games(
        state, out_dir=path, compat_fingerprint="labels", trial_id="trial",
        model_sha="model", model_step=1,
    )
    assert report.persisted == 1
    return next(path.glob(f"*{RESUME_FILE_SUFFIX}"))


def _edit_meta(path: Path, **updates: Any) -> None:
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    meta = json.loads(arrays["meta_json"].tobytes())
    if updates.pop("legacy", False):
        meta.pop("pending_sf_label_offsets", None)
    meta.update(updates)
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    with path.open("wb") as out:
        np.savez(out, **arrays)


def _resume(pool: _ManualPool, path: Path) -> SelfplayState:
    state = _state(pool)
    report = resume_inflight_games(state, in_dir=path,
                                  compat_fingerprint="labels", trial_id="trial")
    assert report.resumed == 1
    assert report.discarded == 0
    return state


def _rows(state: SelfplayState) -> list[Any]:
    records = state.samples_per_game[0]
    return _build_replay_samples(
        state, 0, records, result="1-0", tb_policy_overrides={},
        vol_targets=[None] * len(records), sf_vol_targets=[None] * len(records),
        total_plies_played=len(state.move_idx_history[0]),
        ply_to_index={rec.ply_index: i for i, rec in enumerate(records)},
    )


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("flush", [False, True])
def test_resumed_coverage_matches_uninterrupted_after_tail_becomes_interior(
    tmp_path: Path, legacy: bool, flush: bool,
) -> None:
    control_pool = _ManualPool()
    control = _state(control_pool)
    _move(control, "e2e4")
    submit_async_sf_label_queries(control, [0])
    control_pool.complete()
    assert poll_async_sf_labels(control) == (1, 0)
    head = control.samples_per_game[0][0].sf_policy_target
    assert head is not None
    head = head.copy()
    _move(control, "e7e5")
    assert submit_async_sf_label_queries(control, [0]) == 1
    original_fen = control_pool.calls[-1][0]
    path = _suspend(control, tmp_path)
    if legacy:
        _edit_meta(path, legacy=True)
    pool = _ManualPool()
    resumed = _resume(pool, tmp_path)
    assert pool.calls == []  # resume never blocks on teacher work
    assert has_pending_sf_labels_for_records(resumed, resumed.samples_per_game[0])
    assert submit_async_sf_label_queries(resumed, [0]) == 0  # no duplicate tail
    _move(control, "g1f3")
    _move(resumed, "g1f3")
    submit_async_sf_label_queries(control, [0])
    submit_async_sf_label_queries(resumed, [0])
    assert poll_async_sf_labels(resumed) == (0, 0)
    assert original_fen in [fen for fen, _fut, _nodes in pool.calls]
    assert len(pool.calls) == 2
    pool.complete()
    control_pool.complete()
    assert poll_async_sf_labels(control) == (2, 0)
    if flush:
        assert flush_async_sf_labels_for_records(resumed, resumed.samples_per_game[0]) == (2, 0)
    else:
        assert poll_async_sf_labels(resumed) == (2, 0)
    assert poll_async_sf_labels(resumed) == (0, 0)
    assert not has_pending_sf_labels_for_records(resumed, resumed.samples_per_game[0])
    np.testing.assert_array_equal(resumed.samples_per_game[0][0].sf_policy_target, head)
    for expected, actual in zip(_rows(control), _rows(resumed), strict=True):
        assert actual.sf_wdl is not None
        assert actual.sf_policy_target is not None
        np.testing.assert_array_equal(actual.sf_wdl, expected.sf_wdl)
        np.testing.assert_array_equal(actual.sf_policy_target, expected.sf_policy_target)
        np.testing.assert_array_equal(actual.sf_legal_mask, expected.sf_legal_mask)
        np.testing.assert_array_equal(actual.x, expected.x)


@pytest.mark.parametrize("failure", ["cancel", "result", "submit"])
def test_failed_recovery_drops_once_and_keeps_masking(tmp_path: Path, failure: str) -> None:
    live = _state(_ManualPool())
    _move(live, "e2e4")
    submit_async_sf_label_queries(live, [0])
    _suspend(live, tmp_path)
    pool = _ManualPool()
    resumed = _resume(pool, tmp_path)
    pool.reject = failure == "submit"
    if failure == "submit":
        assert poll_async_sf_labels(resumed) == (0, 1)
    else:
        assert poll_async_sf_labels(resumed) == (0, 0)
        fut = pool.calls[0][1]
        if failure == "cancel":
            fut.cancel()
        else:
            fut.set_exception(RuntimeError("controlled result failure"))
        assert flush_async_sf_labels_for_records(resumed, resumed.samples_per_game[0]) == (0, 1)
    assert resumed.pending_sf_labels == []
    assert poll_async_sf_labels(resumed) == (0, 0)
    rows = _rows(resumed)
    assert rows[0].sf_policy_target is None
    assert rows[0].sf_wdl is None
    assert rows[0].wdl_target == 0
    # Explicit empty provenance avoids restarting failed work indefinitely.
    _suspend(resumed, tmp_path)
    again = _resume(_ManualPool(), tmp_path)
    assert again.pending_sf_labels == []


def test_repeated_suspend_recovers_multiple_original_positions(tmp_path: Path) -> None:
    live = _state(_ManualPool())
    for move in ["e2e4", "e7e5"]:
        _move(live, move)
        submit_async_sf_label_queries(live, [0])
    original_fens = [fen for fen, _fut, _nodes in cast(_ManualPool, live.stockfish).calls]
    _suspend(live, tmp_path)
    resumed = _resume(_ManualPool(), tmp_path)
    _move(resumed, "g1f3")
    _suspend(resumed, tmp_path)  # recovery still queued, old tail now interior
    pool = _ManualPool()
    again = _resume(pool, tmp_path)
    assert poll_async_sf_labels(again) == (0, 0)
    assert [fen for fen, _fut, _nodes in pool.calls] == original_fens
    pool.complete()
    assert poll_async_sf_labels(again) == (2, 0)
    assert len(pool.calls) == 2


def test_recycled_slot_drops_old_descriptors_without_escalation_credit(tmp_path: Path) -> None:
    live = _state(_ManualPool())
    _move(live, "e2e4")
    submit_async_sf_label_queries(live, [0])
    _suspend(live, tmp_path)
    pool = _ManualPool()
    resumed = _resume(pool, tmp_path)
    old_rec = resumed.samples_per_game[0][0]
    poll_async_sf_labels(resumed)
    resumed.recycle_slot(0)
    pool.complete()
    assert poll_async_sf_labels(resumed) == (0, 0)
    assert old_rec.sf_wdl is None
    assert resumed.sf_label_escalations[0] == 0
    assert resumed.pending_sf_labels == []


@pytest.mark.parametrize(("has_policy", "refute", "move"), [
    (False, False, "e2e4"), (True, True, "e2e4"), (True, False, "f7g7"),
])
def test_legacy_fast_refute_and_terminal_rows_do_not_requeue(
    tmp_path: Path, has_policy: bool, refute: bool, move: str,
) -> None:
    live = _state(_ManualPool())
    if move == "f7g7":
        from chess_anti_engine.encoding._lc0_ext import CBoard

        opening = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        live.starting_boards = [opening]
        live.cboards[0] = CBoard.from_board(opening)
    _move(live, move, has_policy=has_policy, refute=refute)
    _edit_meta(_suspend(live, tmp_path), legacy=True)
    resumed = _resume(_ManualPool(), tmp_path)
    assert resumed.pending_sf_labels == []


def test_curriculum_reuse_does_not_duplicate_resumed_label(tmp_path: Path) -> None:
    live = _state(_ManualPool())
    live.selfplay_arr[0] = 0
    live.net_color_arr[0] = 1
    _move(live, "e2e4")
    submit_async_sf_label_queries(live, [0])
    _suspend(live, tmp_path)
    resumed = _resume(_ManualPool(), tmp_path)
    resumed.pending_sf_moves[0] = Future()
    assert submit_async_sf_labels_from_curriculum_moves(resumed, [0]) == 0
    assert len(resumed.pending_sf_labels) == 1


def test_invalid_recovery_offsets_and_wrong_trial_are_rejected(tmp_path: Path) -> None:
    live = _state(_ManualPool())
    _move(live, "e2e4")
    path = _suspend(live, tmp_path)
    _edit_meta(path, pending_sf_label_offsets=[5])
    state = _state(_ManualPool())
    report = resume_inflight_games(state, in_dir=tmp_path,
                                  compat_fingerprint="labels", trial_id="trial")
    assert report.reasons == {"bad_pending_labels": 1}
    assert state.pending_sf_labels == []
    _edit_meta(_suspend(live, tmp_path), trial_id="other-trial")
    report = resume_inflight_games(state, in_dir=tmp_path,
                                  compat_fingerprint="labels", trial_id="trial")
    assert report.resumed == 0
    assert state.pending_sf_labels == []


def test_finalize_flush_can_dispatch_queued_recovery(tmp_path: Path) -> None:
    live = _state(_ManualPool())
    _move(live, "e2e4")
    submit_async_sf_label_queries(live, [0])
    _suspend(live, tmp_path)
    pool = _ManualPool()
    pool.immediate = True
    resumed = _resume(pool, tmp_path)
    assert flush_async_sf_labels_for_records(resumed, resumed.samples_per_game[0]) == (1, 0)
    assert len(pool.calls) == 1
    assert _rows(resumed)[0].sf_policy_target is not None


def test_recovery_respects_pool_cap_and_full_label_budget(tmp_path: Path) -> None:
    live = _state(_ManualPool())
    for _ in range(10):
        board = chess.Board(live.cboards[0].fen())
        _move(live, next(iter(board.legal_moves)).uci())
    path = _suspend(live, tmp_path)
    _edit_meta(path, pending_sf_label_offsets=list(range(10)))
    pool = _ManualPool()
    resumed = _resume(pool, tmp_path)
    assert poll_async_sf_labels(resumed) == (0, 0)
    assert len(pool.calls) == 8
    assert len(resumed.pending_sf_labels) == 10
    pool.complete()
    assert poll_async_sf_labels(resumed) == (8, 0)
    assert len(pool.calls) == 10
    assert all(nodes == 100 for _fen, _fut, nodes in pool.calls)
    pool.complete()
    assert poll_async_sf_labels(resumed) == (2, 0)


def test_real_manager_dispatches_recovery_before_network_advances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from chess_anti_engine.selfplay import manager

    live = _state(_ManualPool())
    _move(live, "e2e4")
    submit_async_sf_label_queries(live, [0])
    _suspend(live, tmp_path)
    pool = _ManualPool()
    pool.immediate = True
    events: list[str] = []
    captured: list[SelfplayState] = []
    completed: list[Any] = []

    def ready(state: SelfplayState) -> None:
        captured.append(state)
        report = resume_inflight_games(state, in_dir=tmp_path,
                                      compat_fingerprint="labels", trial_id="trial")
        assert report.resumed == 1

    def network(state: SelfplayState, _idxs: list[int]) -> None:
        assert state.samples_per_game[0][0].sf_wdl is not None
        events.append("recovered-before-network")
        _move(state, "e7e5")
        state.done_arr[0] = 1
        state.tb_result_arr[0] = "1/2-1/2"  # deterministic finalization, no engine

    monkeypatch.setattr(manager, "run_network_turn", network)
    manager.play_batch(
        model=None, evaluator=cast(Any, object()), device="cpu",
        rng=np.random.default_rng(0), stockfish=pool, games=1, target_games=1,
        on_state_ready=ready, on_game_complete=completed.append,
        opponent=live.opponent, temp=live.temp, search=live.search,
        opening=live.opening, diff_focus=live.diff_focus,
        game=replace(live.game, max_plies=2),
    )
    assert events == ["recovered-before-network"]
    assert len(completed) == 1
    assert captured[0].games_completed == 1
    assert captured[0].pending_sf_labels == []
    assert len(pool.calls) == 1
    assert completed[0].samples[0].sf_wdl is not None


@pytest.mark.parametrize("failure", ["none", "cancel", "result", "submit"])
def test_interrupted_escalation_preserves_fallback_budget_and_credit(
    tmp_path: Path, failure: str,
) -> None:
    from dataclasses import replace

    live_pool = _ManualPool()
    live = _state(live_pool)
    live.game = replace(live.game, sf_label_escalate_q_gap=0.01,
                        sf_label_escalate_max_per_game=2)
    rec = _move(live, "e2e4")
    submit_async_sf_label_queries(live, [0])
    live_pool.complete()
    assert poll_async_sf_labels(live) == (0, 0)  # deep query now unresolved
    original = live.pending_sf_labels[0].escalated_from_res
    assert original is not None
    assert live.sf_label_escalations[0] == 1
    original_wdl = original.wdl[::-1].copy()
    query_fen = live_pool.calls[-1][0]
    _suspend(live, tmp_path)
    pool = _ManualPool()
    resumed = _state(pool)
    resumed.game = live.game
    report = resume_inflight_games(resumed, in_dir=tmp_path,
                                  compat_fingerprint="labels", trial_id="trial")
    assert report.resumed == 1
    pool.reject = failure == "submit"
    if failure == "submit":
        assert poll_async_sf_labels(resumed) == (1, 0)
    else:
        assert poll_async_sf_labels(resumed) == (0, 0)
        assert pool.calls[0][0] == query_fen
        assert pool.calls[0][2] == live.game.sf_label_escalate_nodes
        assert pool.fresh_calls == [True]
        fut = pool.calls[0][1]
        if failure == "cancel":
            fut.cancel()
        elif failure == "result":
            fut.set_exception(RuntimeError("controlled deep failure"))
        else:
            pool.complete()
        assert poll_async_sf_labels(resumed) == (1, 0)
    restored = resumed.samples_per_game[0][0]
    np.testing.assert_array_equal(restored.sf_wdl, original_wdl)
    assert restored.sf_multipv_raw is not None
    if failure == "none":
        np.testing.assert_array_equal(restored.sf_wdl_original, original_wdl)
    else:
        assert restored.sf_wdl_original is None
    assert resumed.sf_label_escalations[0] == 1
    assert resumed.pending_sf_labels == []
    assert poll_async_sf_labels(resumed) == (0, 0)
    assert rec.sf_wdl is None  # old session cannot gain a resumed label


def test_synchronous_recovery_uses_original_p1(tmp_path: Path) -> None:
    live_pool = _ManualPool()
    live = _state(live_pool)
    _move(live, "e2e4")
    submit_async_sf_label_queries(live, [0])
    _suspend(live, tmp_path)
    calls: list[str] = []

    class SyncTeacher:
        def search(self, fen: str, *, nodes=None):
            del nodes
            calls.append(fen)
            return _result(fen)

    resumed = _resume(_ManualPool(), tmp_path)
    resumed.stockfish = cast(Any, SyncTeacher())
    assert poll_async_sf_labels(resumed) == (1, 0)
    assert calls == [live_pool.calls[0][0]]
    assert _rows(resumed)[0].sf_wdl is not None


def test_manager_backlog_does_not_lose_next_selfplay_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from chess_anti_engine.selfplay import manager

    live = _state(_ManualPool())
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
    for move in moves:
        _move(live, move)
        assert submit_async_sf_label_queries(live, [0]) == 1
    _suspend(live, tmp_path)
    pool = _ManualPool()
    captured: list[SelfplayState] = []
    completed: list[Any] = []
    advances: list[int] = []

    def ready(state: SelfplayState) -> None:
        captured.append(state)
        report = resume_inflight_games(state, in_dir=tmp_path,
                                      compat_fingerprint="labels", trial_id="trial")
        assert report.resumed == 1

    def network(state: SelfplayState, _idxs: list[int]) -> None:
        assert all(rec.sf_wdl is not None for rec in state.samples_per_game[0])
        advances.append(len(state.samples_per_game[0]))
        _move(state, "e1g1" if len(advances) == 1 else "f8e7")
        if len(advances) == 2:
            state.done_arr[0] = 1
            state.tb_result_arr[0] = "1/2-1/2"

    def wait_for_teacher(_state: SelfplayState, *, control_poll: bool) -> None:
        del control_poll
        pool.complete()

    monkeypatch.setattr(manager, "run_network_turn", network)
    monkeypatch.setattr(manager, "_wait_for_starved_sf", wait_for_teacher)
    manager.play_batch(
        model=None, evaluator=cast(Any, object()), device="cpu",
        rng=np.random.default_rng(0), stockfish=pool, games=1, target_games=1,
        on_state_ready=ready, on_game_complete=completed.append,
        on_step=pool.complete,
        opponent=live.opponent, temp=live.temp, search=live.search,
        opening=live.opening, diff_focus=live.diff_focus,
        game=replace(live.game, max_plies=10),
    )
    assert advances == [8, 9]
    assert len(pool.calls) == 9
    assert len(completed) == 1
    assert all(s.sf_wdl is not None for s in completed[0].samples[:9])
    assert captured[0].pending_sf_labels == []


@pytest.mark.parametrize(("batch_size", "pending"), [(2, 16), (3, 23)])
def test_recovery_saturation_preserves_other_slots_next_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int, pending: int,
) -> None:
    from chess_anti_engine.selfplay import manager

    live = _state(_ManualPool(), batch_size=batch_size)
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
             "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6", "c2c3", "e8g8",
             "h2h3", "c8b7", "d2d3", "c6b8", "c1e3", "b8d7", "b1d2"]
    for move in moves[:pending]:
        _move(live, move)
        assert submit_async_sf_label_queries(live, [0]) == 1
    _suspend(live, tmp_path)
    pool = _ManualPool()
    captured: list[SelfplayState] = []
    waits: list[int] = []
    network_calls: list[int] = []

    def ready(state: SelfplayState) -> None:
        captured.append(state)
        report = resume_inflight_games(state, in_dir=tmp_path,
                                      compat_fingerprint="labels", trial_id="trial")
        assert report.resumed == 1

    def network(state: SelfplayState, idxs: list[int]) -> None:
        assert waits == [pending]
        assert all(rec.sf_wdl is not None for rec in state.samples_per_game[0])
        _move(state, "e2e4", slot=1)
        # Only the other slot creates a row in this synthetic network turn.
        for slot in range(batch_size):
            if slot != 1:
                state.selfplay_arr[slot] = 0
        network_calls.extend(idxs)
        pool.immediate = True

    def wait_for_teacher(_state: SelfplayState, *, control_poll: bool) -> None:
        del control_poll
        waits.append(len(pool.calls))
        pool.complete()

    def stop() -> bool:
        return bool(network_calls)

    monkeypatch.setattr(manager, "run_network_turn", network)
    monkeypatch.setattr(manager, "_wait_for_starved_sf", wait_for_teacher)
    manager.play_batch(
        model=None, evaluator=cast(Any, object()), device="cpu",
        rng=np.random.default_rng(0), stockfish=pool, games=batch_size, target_games=0,
        on_state_ready=ready, stop_fn=stop,
        opponent=live.opponent, temp=live.temp, search=live.search,
        opening=live.opening, diff_focus=live.diff_focus, game=live.game,
    )
    assert network_calls
    assert len(pool.calls) == pending + 1
    assert captured[0].samples_per_game[1][0].sf_wdl is not None
    assert captured[0].pending_sf_labels == []
