"""Finite collection observations; no checkpoints, real engines or GPU work."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace
import json
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.moves import POLICY_SIZE, index_to_move_strict, move_to_index
from chess_anti_engine.replay.shard import samples_to_arrays
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay import finalize as finalize_module
from chess_anti_engine.selfplay import manager as manager_module
from chess_anti_engine.selfplay.config import DiffFocusConfig, GameConfig, SearchConfig
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.provenance import GameProvenance, capture_game_start
from chess_anti_engine.selfplay.state import SelfplayState, _NetRecord
from tests.selfplay_helpers import FakeStockfish, UniformPolicyValueModel


def _collect(**overrides: Any):
    args: dict[str, Any] = {
        "model": UniformPolicyValueModel().eval(), "device": "cpu",
        "rng": np.random.default_rng(7), "stockfish": FakeStockfish([0.0, 1.0, 0.0]),
        "games": 2, "target_games": 3,
        "search": SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        "opening": OpeningConfig(random_start_plies=0),
        "diff_focus": DiffFocusConfig(enabled=False),
        "game": GameConfig(max_plies=2, selfplay_fraction=1.0),
    }
    args.update(overrides)
    return play_batch(**args)


def test_zero_retained_games_emit_provenance_without_changing_sample_callback(monkeypatch):
    monkeypatch.setattr(finalize_module, "_build_replay_samples", lambda *_a, **_kw: [])
    observed, completed = [], []
    samples, stats = _collect(
        on_game_provenance=observed.append, on_game_complete=completed.append,
    )
    assert samples == completed == []
    assert stats.games == len(observed) == 3
    assert sorted(x.start.start_ordinal for x in observed) == [0, 1, 2]
    assert all(x.retained_samples == 0 for x in observed)
    assert observed[0].start.slot == observed[2].start.slot == 0
    # Recycled-slot identity does not mutate an earlier frozen observation.
    assert observed[0].start.start_ordinal == 0
    json.dumps([asdict(x) for x in observed])
    with pytest.raises(FrozenInstanceError):
        setattr(observed[0], "result", "1-0")


def test_default_samples_and_existing_callback_are_unchanged():
    original, enabled, observed = [], [], []
    old_samples, old_stats = _collect(on_game_complete=original.append)
    new_samples, new_stats = _collect(
        on_game_complete=enabled.append, on_game_provenance=observed.append,
    )
    assert asdict(old_stats) == asdict(new_stats)
    old_arrays, new_arrays = samples_to_arrays(old_samples), samples_to_arrays(new_samples)
    assert old_arrays.keys() == new_arrays.keys()
    for key in old_arrays:
        np.testing.assert_array_equal(old_arrays[key], new_arrays[key])
    assert len(original) == len(enabled) == len(observed) == 3
    for old, new, audit in zip(original, enabled, observed, strict=True):
        assert old.positions == new.positions == audit.retained_samples
        assert all(s.game_id == audit.game_id for s in new.samples)


def test_opening_history_and_network_record_offsets_replay_legally(monkeypatch):
    start = chess.Board()
    for uci in ["g1f3", "d7d5"]:
        start.push_uci(uci)
    monkeypatch.setattr(
        "chess_anti_engine.selfplay.state.resolve_slot_opening",
        lambda **_kw: SimpleNamespace(board=start.copy(), source="test_history"),
    )
    observed: list[GameProvenance] = []
    states: list[SelfplayState] = []
    _collect(
        games=1, target_games=1, game=GameConfig(max_plies=2, selfplay_fraction=1.0),
        on_game_provenance=observed.append, on_state_ready=states.append,
    )
    audit = observed[0]
    assert audit.start.root_fen == chess.STARTING_FEN
    assert audit.start.opening_moves == ("g1f3", "d7d5")
    board = chess.Board(audit.start.root_fen)
    for uci in audit.start.opening_moves:
        board.push_uci(uci)
    assert board.fen() == audit.start.start_fen
    positions = []
    for action in audit.played_move_indices:
        positions.append((board.turn, board.ply()))
        move = index_to_move_strict(action, board)
        assert move in board.legal_moves
        board.push(move)
    assert board.ply() == audit.total_plies
    assert board.fen() == audit.final_fen == states[0].cboards[0].fen()
    assert audit.records
    for rec in audit.records:
        assert positions[rec.move_offset] == (rec.pov_color, rec.ply_index)
        assert rec.has_policy
        assert not rec.is_sf_refute_opp


def test_capped_sf_default_draw_is_not_natural():
    class UnavailableSF(FakeStockfish):
        def search(self, *_args, **_kwargs):
            raise OSError("synthetic missing terminal evaluation")

    observed = []
    _collect(
        games=1, target_games=1, game=GameConfig(max_plies=0),
        stockfish=UnavailableSF([0.0, 1.0, 0.0]), on_game_provenance=observed.append,
    )
    audit = observed[0]
    assert audit.result == audit.resolved_result == "1/2-1/2"
    assert audit.natural_result == "*"
    assert audit.resolution == "sf_default_draw"
    assert audit.cap_reached


def test_sf_adjudication_and_natural_mate_have_distinct_provenance(monkeypatch):
    capped = []
    _collect(games=1, target_games=1, game=GameConfig(max_plies=0), on_game_provenance=capped.append)
    assert capped[0].resolution == "sf_adjudication"
    assert capped[0].cap_reached
    assert capped[0].natural_result == "*"
    mate = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    monkeypatch.setattr(
        "chess_anti_engine.selfplay.state.resolve_slot_opening",
        lambda **_kw: SimpleNamespace(board=mate.copy(), source="test_terminal"),
    )
    observed = []
    _collect(
        games=1, target_games=1, on_game_provenance=observed.append,
        on_state_ready=lambda state: state.done_arr.fill(1),
    )
    assert observed[0].result == observed[0].natural_result == "1-0"
    assert observed[0].resolution == "natural"
    assert not observed[0].cap_reached
    assert observed[0].records == ()
    assert observed[0].retained_samples == 0


def test_stashed_tablebase_is_identified_without_a_real_probe():
    def stash(state):
        state.tb_result_arr[0] = "1-0"
        state.done_arr[0] = 1

    observed = []
    _collect(games=1, target_games=1, on_state_ready=stash, on_game_provenance=observed.append)
    assert observed[0].resolution == "tablebase_adjudication"
    assert observed[0].natural_result == "*"
    assert observed[0].result == "1-0"


@pytest.mark.parametrize("tb_result", ["1/2-1/2", "1-0"])
def test_tablebase_rescore_is_recorded_even_when_result_does_not_change(monkeypatch, tb_result):
    monkeypatch.setattr(finalize_module, "rescore_game_samples", lambda *_a: tb_result)

    def enable_after_init(state):
        state.game = replace(state.game, syzygy_path="synthetic_not_opened")

    observed = []
    _collect(
        games=1, target_games=1, game=GameConfig(max_plies=0),
        on_state_ready=enable_after_init, on_game_provenance=observed.append,
    )
    audit = observed[0]
    assert audit.resolution == "sf_adjudication"
    assert audit.tablebase_rescore_result == audit.result == tb_result
    assert audit.resolved_result == "1/2-1/2"
    assert audit.natural_result == "*"


def test_raw_sf_refute_record_is_not_hidden_by_replay_network_flag(monkeypatch):
    monkeypatch.setattr(finalize_module, "_build_replay_samples", lambda *_a, **_kw: [])

    def inject(state):
        board = chess.Board()
        for offset, uci in enumerate(["e2e4", "e7e5"]):
            rec = _NetRecord(
                x=np.zeros((175, 8, 8), np.float32),
                policy_probs=np.full(POLICY_SIZE, 1 / POLICY_SIZE, np.float32),
                net_wdl_est=np.array([0.0, 1.0, 0.0], np.float32),
                search_wdl_est=np.array([0.0, 1.0, 0.0], np.float32),
                pov_color=board.turn, ply_index=board.ply(), has_policy=True,
                priority=1.0, sample_weight=1.0, keep_prob=1.0, move_offset=offset,
            )
            rec.is_sf_refute_opp = offset == 1
            state.samples_per_game[0].append(rec)
            move = chess.Move.from_uci(uci)
            action = move_to_index(move, board)
            state.move_idx_history[0].append(action)
            state.cboards[0].push_index(action)
            board.push(move)
        state.done_arr[0] = 1

    observed = []
    _collect(games=1, target_games=1, on_state_ready=inject, on_game_provenance=observed.append)
    first, second = observed[0].records
    assert (first.move_offset, first.pov_color, first.is_sf_refute_opp) == (0, True, False)
    assert (second.move_offset, second.pov_color, second.is_sf_refute_opp) == (1, False, True)


def test_continuous_observation_refuses_before_state_or_engine_work(monkeypatch):
    monkeypatch.setattr(SelfplayState, "create", lambda **_kw: pytest.fail("state must not be created"))
    with pytest.raises(ValueError, match="requires finite"):
        _collect(target_games=0, stop_fn=lambda: False, on_game_provenance=lambda _x: None)


def test_ordinal_is_assigned_at_start_even_when_completions_are_out_of_order(monkeypatch):
    state: Any = SimpleNamespace(
        done_arr=[0, 1], finalized_arr=[0, 0], games_started=2, games_completed=0,
        starting_boards=[chess.Board(), chess.Board()],
        opening_source_arr=["test", "test"], net_color_arr=[1, 0], selfplay_arr=[1, 1],
    )
    starts = [capture_game_start(state, i, i) for i in range(2)]
    finalized_starts = []

    def finalize(*_args, **kwargs):
        finalized_starts.append(kwargs["game_start"])

    def recycle(slot):
        state.done_arr[slot] = state.finalized_arr[slot] = 0
        state.games_started += 1

    state.recycle_slot = recycle
    monkeypatch.setattr(manager_module, "finalize_game", finalize)
    for slot in [1, 0, 1]:
        state.done_arr[slot] = 1
        manager_module._finalize_completed_slots(
            state, all_samples=[], on_game_complete=None,
            on_game_provenance=lambda _x: None, game_starts=starts,
            batch_size=2, continuous=False, target=6,
        )
    assert [x.start_ordinal for x in finalized_starts] == [1, 0, 2]
    assert [x.slot for x in finalized_starts] == [1, 0, 1]


def test_early_tb_at_large_absolute_ply_does_not_claim_cap(monkeypatch, caplog):
    board = chess.Board(chess.STARTING_FEN.replace("0 1", "0 69"))
    monkeypatch.setattr(
        "chess_anti_engine.selfplay.state.resolve_slot_opening",
        lambda **_kw: SimpleNamespace(board=board.copy(), source="test_late_seed"),
    )

    def stash(state):
        state.tb_result_arr[0] = "1/2-1/2"
        state.done_arr[0] = 1

    observed = []
    _collect(games=1, target_games=1, on_state_ready=stash, on_game_provenance=observed.append)
    assert observed[0].start.start_ply == observed[0].total_plies == 136
    assert not observed[0].cap_reached
    assert observed[0].played_move_indices == ()
    assert not any("MAX_PLY_GAME" in record.message for record in caplog.records)


def test_actual_cap_at_large_opening_ply_is_still_reported(monkeypatch, caplog):
    board = chess.Board(chess.STARTING_FEN.replace("0 1", "0 69"))
    monkeypatch.setattr(
        "chess_anti_engine.selfplay.state.resolve_slot_opening",
        lambda **_kw: SimpleNamespace(board=board.copy(), source="test_late_seed"),
    )
    observed = []
    _collect(
        games=1, target_games=1,
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
        on_game_provenance=observed.append,
    )
    assert observed[0].total_plies - observed[0].start.start_ply == 2
    assert observed[0].cap_reached
    assert sum("MAX_PLY_GAME" in record.message for record in caplog.records) == 1


def test_pre_cap_sf_adjudication_at_large_opening_ply_is_not_a_cap(monkeypatch, caplog):
    board = chess.Board(chess.STARTING_FEN.replace("0 1", "0 69"))
    monkeypatch.setattr(
        "chess_anti_engine.selfplay.state.resolve_slot_opening",
        lambda **_kw: SimpleNamespace(board=board.copy(), source="test_late_seed"),
    )
    observed = []
    _collect(
        games=1, target_games=1,
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
        on_state_ready=lambda state: state.done_arr.fill(1),
        on_game_provenance=observed.append,
    )
    assert observed[0].total_plies == observed[0].start.start_ply == 136
    assert observed[0].resolution == "sf_adjudication"
    assert not observed[0].cap_reached
    assert not any("MAX_PLY_GAME" in record.message for record in caplog.records)


def test_raw_observation_is_snapshotted_before_existing_sample_callback():
    states, observed = [], []

    def mutate_raw_after_samples(_batch):
        states[0].samples_per_game[0].clear()

    _collect(
        games=1, target_games=1, on_state_ready=states.append,
        on_game_complete=mutate_raw_after_samples, on_game_provenance=observed.append,
    )
    assert observed[0].records
    assert observed[0].retained_samples > 0
    assert states[0].samples_per_game[0] == []


def test_default_off_does_not_construct_provenance(monkeypatch):
    monkeypatch.setattr(
        manager_module, "capture_game_start",
        lambda *_a, **_kw: pytest.fail("default-off must not build observations"),
    )
    _collect(games=1, target_games=1)


def test_stale_opening_state_is_observable_even_with_same_legal_trace_length():
    from chess_anti_engine.encoding._lc0_ext import CBoard

    actual = chess.Board()
    actual.castling_rights = 0
    states, observed = [], []

    def mismatch(state):
        # Explicitly model a caller restoring a different native board while
        # leaving the captured starting history stale. The observer must expose
        # the disagreement, not repair either producer board or history.
        state.cboards[0] = CBoard.from_board(actual)
        states.append(state)

    _collect(
        games=1, target_games=1, game=GameConfig(max_plies=0),
        on_state_ready=mismatch, on_game_provenance=observed.append,
    )
    audit = observed[0]
    replayed = chess.Board(audit.start.root_fen)
    assert audit.start.opening_moves == audit.played_move_indices == ()
    assert replayed.ply() == audit.total_plies == 0
    assert audit.final_fen == actual.fen() == states[0].cboards[0].fen()
    assert replayed.fen() != audit.final_fen
