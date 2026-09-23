"""CPU fake-batch contracts for buffered BT4 root-policy games."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import chess
import chess.syzygy
import numpy as np
import pytest

from chess_anti_engine import tablebase
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.eval.rvg_surgery import position_fingerprints
from chess_anti_engine.moves.leela_index import compact_index_for_move
from chess_anti_engine.selfplay.bt4_outcome import BT4OutcomeDecision
from scripts import bt4_root_policy_stepper as policy
from scripts.bt4_generation_evaluator import BT4RootOutput
from scripts.gen_sf_rooted_corpus import input_tensor_key

HISTORY = "lc0_root_legacy_meta"
FEATURES = "v2_threats"
MODEL_SHA = "a" * 64
SEVEN_MAN = "7k/8/8/8/8/8/2p5/KQBNR3 w - - 0 1"


@pytest.fixture(autouse=True)
def checked_history_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the guard without changing a shared process's native encoder mode.
    monkeypatch.setattr(policy.rep_fix, "current", lambda: False)


class FakeBatchEvaluator:
    """One synthetic teacher call per prepared batch, with no ONNX or search."""

    def __init__(self, preferred_uci: str | None = None) -> None:
        self.preferred_uci = preferred_uci
        self.calls = 0
        self.rows = 0

    def evaluate_roots(
        self, boards: list[chess.Board], x_batch: np.ndarray,
    ) -> list[BT4RootOutput]:
        assert len(boards) == len(x_batch)
        self.calls += 1
        self.rows += len(boards)
        source_keys = position_fingerprints(
            x_batch, input_history_encoding=HISTORY,
        )
        outputs: list[BT4RootOutput] = []
        for board, x, source_key in zip(boards, x_batch, source_keys):
            moves = list(board.legal_moves)
            dense = np.zeros((1858,), dtype=np.float32)
            if self.preferred_uci is not None and any(
                move.uci() == self.preferred_uci for move in moves
            ):
                preferred = chess.Move.from_uci(self.preferred_uci)
                for move in moves:
                    dense[compact_index_for_move(board, move)] = np.float32(
                        0.25 / (len(moves) - 1) if len(moves) > 1 else 0.0
                    )
                dense[compact_index_for_move(board, preferred)] = np.float32(1.0 if len(moves) == 1 else 0.75)
            else:
                for move in moves:
                    dense[compact_index_for_move(board, move)] = np.float32(
                        1.0 / len(moves)
                    )
            # The native teacher predicts LOSS for the mover. Final game
            # outcomes in these cases differ and must never overwrite it.
            raw = np.array([0.05, 0.10, 0.85], dtype=np.float64)
            for arr in (dense, raw):
                arr.flags.writeable = False
            outputs.append(BT4RootOutput(
                fen=board.fen(), input_key=input_tensor_key(x),
                source_key=source_key, policy_t1=dense, wdl_raw=raw,
                policy_output="native_policy", wdl_output="native_wdl",
                wdl_kind="probabilities", model_sha256=MODEL_SHA,
                input_name="planes", input_dtype="float32",
                input_history_encoding=HISTORY,
                input_extra_features=FEATURES,
                history_rep_fix=False,
            ))
        return outputs


class FakeMatchTablebase:
    def __init__(self, wdl: int | None, dtz: int | None) -> None:
        self.raw_wdl = wdl
        self.raw_dtz = dtz
        self.wdl = {"KQBNRvK": object()}
        self.dtz = {"KQBNRvK": object()}
        self.probes: list[str] = []

    def probe_wdl(self, board: chess.Board) -> int:
        self.probes.append("wdl")
        if self.raw_wdl is None:
            raise chess.syzygy.MissingTableError(board.fen())
        return self.raw_wdl

    def probe_dtz(self, board: chess.Board) -> int:
        self.probes.append("dtz")
        if self.raw_dtz is None:
            raise chess.syzygy.MissingTableError(board.fen())
        return self.raw_dtz


def make_theoretical_stepper(
    boards: dict[int, chess.Board], *, max_plies: int,
) -> tuple[policy.BT4RootPolicyStepper, dict[int, np.random.Generator]]:
    rngs = {slot: np.random.default_rng(100 + slot) for slot in boards}
    stepper = policy.BT4RootPolicyStepper(
        boards, rngs, max_plies=max_plies, syzygy_path="fake:pair",
        input_history_encoding=HISTORY, input_extra_features=FEATURES,
        history_rep_fix=False,
        model_sha256=MODEL_SHA,
        outcome_mode="theoretical_wdl",
    )
    return stepper, rngs


def run_fake_batch(
    stepper: policy.BT4RootPolicyStepper, evaluator: FakeBatchEvaluator,
    *, temperature: float,
) -> tuple[policy.PreparedBatch, tuple[policy.BT4SelectedMove, ...]]:
    batch, finalized = stepper.prepare_roots()
    assert batch is not None
    assert finalized == ()
    boards, x_batch = batch.inference_inputs()
    outputs = evaluator.evaluate_roots(boards, x_batch)
    choices = stepper.apply_root_outputs(
        batch, outputs,
        temperatures={root.slot_id: temperature for root in batch.roots},
    )
    return batch, choices


def test_seven_to_six_capture_adjudicates_before_next_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTablebase:
        probes = 0

        def probe_wdl(self, _board: chess.Board) -> int:
            self.probes += 1
            return -2  # Black to move loses after White's capture.

    tb = FakeTablebase()
    monkeypatch.setattr(tablebase, "get_tablebase", lambda _path: tb)
    stepper, _rngs = make_theoretical_stepper({0: chess.Board(SEVEN_MAN)}, max_plies=8)
    evaluator = FakeBatchEvaluator(preferred_uci="b1c2")

    batch, choices = run_fake_batch(stepper, evaluator, temperature=0.0)
    assert [choice.move.uci() for choice in choices] == ["b1c2"]
    next_batch, finalized = stepper.prepare_roots()
    assert next_batch is None
    assert len(finalized) == 1
    game = finalized[0]
    assert isinstance(game, policy.BT4CompletedGame)
    assert game.result == "1-0"
    assert game.termination == "syzygy"
    assert len(game.records) == 1
    record = game.records[0]
    assert record.wdl_target == 0  # White won.
    assert record.played.x.dtype == np.dtype(np.float32)
    assert np.array_equal(record.played.x, batch.roots[0].x)
    assert input_tensor_key(record.played.x) == record.played.teacher.input_key
    assert not record.played.x.flags.writeable
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        record.played.x.flags.writeable = True
    assert record.played.teacher.wdl_raw.tolist() == [0.05, 0.10, 0.85]
    assert np.count_nonzero(record.played.teacher.policy_t1) > 1
    assert float(record.played.teacher.policy_t1.max()) == 0.75
    assert not record.played.teacher.wdl_raw.flags.writeable
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        record.played.teacher.wdl_raw.flags.writeable = True
    assert evaluator.calls == evaluator.rows == tb.probes == 1
    assert stepper.counts.rows_emitted == 1
    assert stepper.counts.rows_discarded == 0


def test_capped_game_drops_every_buffered_row() -> None:
    stepper, _rngs = make_theoretical_stepper({0: chess.Board()}, max_plies=1)
    evaluator = FakeBatchEvaluator(preferred_uci="e2e4")
    _batch, choices = run_fake_batch(stepper, evaluator, temperature=0.0)
    assert choices[0].move.uci() == "e2e4"

    next_batch, finalized = stepper.prepare_roots()
    assert next_batch is None
    assert finalized == (
        policy.BT4DiscardedGame(
            0, "max_plies_unresolved", "above_six_man", 1, 1,
            stepper.outcome_provenance,
        ),
    )
    assert stepper.counts.games_discarded == 1
    assert stepper.counts.rows_discarded == 1
    assert stepper.counts.rows_emitted == 0
    assert stepper.counts.discarded_by_termination == {"max_plies_unresolved": 1}


def test_natural_claim_backfills_draw_separately_from_teacher() -> None:
    board = chess.Board()
    board.halfmove_clock = 98
    stepper, _rngs = make_theoretical_stepper({0: board}, max_plies=5)
    evaluator = FakeBatchEvaluator(preferred_uci="g1f3")
    _batch, choices = run_fake_batch(stepper, evaluator, temperature=0.0)
    assert choices[0].move.uci() == "g1f3"

    next_batch, finalized = stepper.prepare_roots()
    assert next_batch is None
    game = finalized[0]
    assert isinstance(game, policy.BT4CompletedGame)
    assert game.termination == "natural"
    assert game.detail == "fifty_moves"
    assert game.result == "1/2-1/2"
    assert [row.wdl_target for row in game.records] == [1]
    assert game.records[0].played.teacher.wdl_raw[2] == 0.85
    assert evaluator.calls == 1


def test_both_colors_receive_their_own_outcome_sign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def two_ply_result(
        _board: chess.Board, *, plies: int, max_plies: int, syzygy_path: str,
        outcome_mode: policy.BT4OutcomeMode,
        match_tablebase: chess.syzygy.Tablebase | None,
    ) -> BT4OutcomeDecision | None:
        assert max_plies == 4
        assert syzygy_path == "fake:pair"
        assert outcome_mode == "theoretical_wdl"
        assert match_tablebase is None
        return (
            BT4OutcomeDecision("1-0", "syzygy", "fake_two_ply")
            if plies == 2 else None
        )

    monkeypatch.setattr(policy, "decide_bt4_outcome", two_ply_result)
    stepper, _rngs = make_theoretical_stepper({0: chess.Board()}, max_plies=4)
    for uci in ("e2e4", "e7e5"):
        _batch, choices = run_fake_batch(
            stepper, FakeBatchEvaluator(preferred_uci=uci), temperature=0.0,
        )
        assert choices[0].move.uci() == uci
    next_batch, finalized = stepper.prepare_roots()
    assert next_batch is None
    game = finalized[0]
    assert isinstance(game, policy.BT4CompletedGame)
    assert [row.wdl_target for row in game.records] == [0, 2]
    assert [row.played.pov_white for row in game.records] == [True, False]
    assert all(row.played.teacher.wdl_raw[2] == 0.85 for row in game.records)


def test_claimable_six_man_natural_terminal_skips_inference() -> None:
    board = chess.Board("7k/8/8/8/8/8/8/KQBNR3 w - - 99 1")
    stepper, _rngs = make_theoretical_stepper({0: board}, max_plies=8)
    batch, finalized = stepper.prepare_roots()
    assert batch is None
    assert finalized == (
        policy.BT4CompletedGame(
            0, "1/2-1/2", "natural", "fifty_moves", (),
            stepper.outcome_provenance,
        ),
    )
    assert stepper.counts.games_completed == 1
    assert stepper.counts.rows_emitted == 0


def test_wrong_order_rejects_entire_batch_before_rng_or_board_mutation() -> None:
    other = chess.Board()
    other.push_uci("d2d4")
    stepper, rngs = make_theoretical_stepper({0: chess.Board(), 1: other}, max_plies=3)
    evaluator = FakeBatchEvaluator()
    batch, finalized = stepper.prepare_roots()
    assert batch is not None
    assert finalized == ()
    before = {slot: deepcopy(rng.bit_generator.state) for slot, rng in rngs.items()}
    boards, x_batch = batch.inference_inputs()
    outputs = evaluator.evaluate_roots(boards, x_batch)

    with pytest.raises(ValueError, match="another root"):
        stepper.apply_root_outputs(
            batch, [outputs[0], replace(outputs[1], input_key="wrong")],
            temperatures={0: 1.0, 1: 1.0},
        )
    assert {slot: rng.bit_generator.state for slot, rng in rngs.items()} == before

    with pytest.raises(ValueError, match="another root"):
        stepper.apply_root_outputs(
            batch, list(reversed(outputs)), temperatures={0: 1.0, 1: 1.0},
        )
    assert {slot: rng.bit_generator.state for slot, rng in rngs.items()} == before
    assert stepper.counts.rows_emitted == stepper.counts.rows_discarded == 0
    with pytest.raises(RuntimeError, match="awaiting outputs"):
        stepper.prepare_roots()

    with pytest.raises(ValueError, match="temperature"):
        stepper.apply_root_outputs(
            batch, outputs, temperatures={0: 1.0, 1: float("nan")},
        )
    assert {slot: rng.bit_generator.state for slot, rng in rngs.items()} == before

    selected = stepper.apply_root_outputs(
        batch, outputs, temperatures={0: 1.0, 1: 1.0},
    )
    assert len(selected) == 2
    with pytest.raises(RuntimeError, match="stale or already consumed"):
        stepper.apply_root_outputs(
            batch, outputs, temperatures={0: 1.0, 1: 1.0},
        )


def test_returned_root_copies_cannot_mutate_owned_game() -> None:
    stepper, rngs = make_theoretical_stepper({0: chess.Board()}, max_plies=1)
    evaluator = FakeBatchEvaluator(preferred_uci="e2e4")
    batch, _finalized = stepper.prepare_roots()
    assert batch is not None
    boards, x_batch = batch.inference_inputs()
    outputs = evaluator.evaluate_roots(boards, x_batch)
    before = deepcopy(rngs[0].bit_generator.state)
    batch.roots[0].board.push_uci("d2d4")
    with pytest.raises(ValueError, match="prepared root changed"):
        stepper.apply_root_outputs(batch, outputs, temperatures={0: 0.0})
    assert rngs[0].bit_generator.state == before
    assert stepper.counts.rows_emitted == stepper.counts.rows_discarded == 0


def test_returned_input_copies_cannot_mutate_owned_game() -> None:
    stepper, rngs = make_theoretical_stepper({0: chess.Board()}, max_plies=1)
    evaluator = FakeBatchEvaluator(preferred_uci="e2e4")
    batch, finalized = stepper.prepare_roots()
    assert batch is not None
    assert finalized == ()
    boards, x_batch = batch.inference_inputs()
    outputs = evaluator.evaluate_roots(boards, x_batch)
    before = deepcopy(rngs[0].bit_generator.state)
    batch.roots[0].x[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="prepared root changed"):
        stepper.apply_root_outputs(batch, outputs, temperatures={0: 0.0})
    assert rngs[0].bit_generator.state == before
    assert stepper.counts.rows_emitted == 0
    assert stepper.counts.rows_discarded == 0


def test_history_mode_is_bound_before_prepare_and_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boards = {0: chess.Board()}
    rngs = {0: np.random.default_rng(100)}
    for invalid, observed in ((None, None), (0, False), (1, True)):
        monkeypatch.setattr(policy.rep_fix, "current", lambda observed=observed: observed)
        with pytest.raises(TypeError, match="explicit bool"):
            policy.BT4RootPolicyStepper(
                boards, rngs, max_plies=1, syzygy_path="fake:pair",
                input_history_encoding=HISTORY, input_extra_features=FEATURES,
                history_rep_fix=cast(bool, invalid), model_sha256=MODEL_SHA,
                outcome_mode="theoretical_wdl",
            )
    for observed in (None, True):
        monkeypatch.setattr(policy.rep_fix, "current", lambda observed=observed: observed)
        with pytest.raises(RuntimeError, match="history_rep_fix"):
            policy.BT4RootPolicyStepper(
                boards, rngs, max_plies=1, syzygy_path="fake:pair",
                input_history_encoding=HISTORY, input_extra_features=FEATURES,
                history_rep_fix=False, model_sha256=MODEL_SHA,
                outcome_mode="theoretical_wdl",
            )

    monkeypatch.setattr(policy.rep_fix, "current", lambda: False)
    stepper, rngs = make_theoretical_stepper(boards, max_plies=1)
    monkeypatch.setattr(policy.rep_fix, "current", lambda: True)
    with pytest.raises(RuntimeError, match="history_rep_fix"):
        stepper.prepare_roots()

    monkeypatch.setattr(policy.rep_fix, "current", lambda: False)
    batch, finalized = stepper.prepare_roots()
    assert batch is not None
    assert finalized == ()
    evaluator = FakeBatchEvaluator()
    boards_for_eval, x_batch = batch.inference_inputs()
    outputs = evaluator.evaluate_roots(boards_for_eval, x_batch)
    before = deepcopy(rngs[0].bit_generator.state)

    monkeypatch.setattr(policy.rep_fix, "current", lambda: True)
    with pytest.raises(RuntimeError, match="history_rep_fix"):
        stepper.apply_root_outputs(batch, outputs, temperatures={0: 1.0})
    assert rngs[0].bit_generator.state == before

    monkeypatch.setattr(policy.rep_fix, "current", lambda: False)
    with pytest.raises(ValueError, match="another root"):
        stepper.apply_root_outputs(
            batch, [replace(outputs[0], history_rep_fix=True)],
            temperatures={0: 1.0},
        )
    assert rngs[0].bit_generator.state == before
    assert len(stepper.apply_root_outputs(batch, outputs, temperatures={0: 1.0})) == 1


def test_native_teacher_head_contract_cannot_change_between_plies() -> None:
    stepper, rngs = make_theoretical_stepper({0: chess.Board()}, max_plies=2)
    evaluator = FakeBatchEvaluator(preferred_uci="e2e4")
    run_fake_batch(stepper, evaluator, temperature=0.0)
    batch, finalized = stepper.prepare_roots()
    assert batch is not None
    assert finalized == ()
    boards, x_batch = batch.inference_inputs()
    outputs = evaluator.evaluate_roots(boards, x_batch)
    before = deepcopy(rngs[0].bit_generator.state)
    with pytest.raises(ValueError, match="mixes output contracts"):
        stepper.apply_root_outputs(
            batch, [replace(outputs[0], wdl_output="other_wdl")],
            temperatures={0: 1.0},
        )
    assert rngs[0].bit_generator.state == before
    assert len(stepper.apply_root_outputs(batch, outputs, temperatures={0: 1.0})) == 1


def test_prepare_failure_retains_earlier_terminal_event_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkmated = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    stepper, _rngs = make_theoretical_stepper({0: checkmated, 1: chess.Board()}, max_plies=1)
    original_encode = policy.encode_cboard
    calls = 0

    def fail_once(
        cb: CBoard, *, input_history_encoding: str | None = None,
        input_extra_features: str | None = None,
    ) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected later encode failure")
        return original_encode(
            cb, input_history_encoding=input_history_encoding,
            input_extra_features=input_extra_features,
        )

    monkeypatch.setattr(policy, "encode_cboard", fail_once)
    with pytest.raises(RuntimeError, match="injected later encode failure"):
        stepper.prepare_roots()
    assert stepper.counts.games_completed == 0
    assert stepper.counts.rows_emitted == 0

    batch, finalized = stepper.prepare_roots()
    assert batch is not None
    assert [root.slot_id for root in batch.roots] == [1]
    assert finalized == (
        policy.BT4CompletedGame(
            0, "1-0", "natural", "checkmate", (), stepper.outcome_provenance,
        ),
    )
    assert stepper.counts.games_completed == 1
    boards, x_batch = batch.inference_inputs()
    outputs = FakeBatchEvaluator().evaluate_roots(boards, x_batch)
    stepper.apply_root_outputs(batch, outputs, temperatures={1: 0.0})
    next_batch, next_finalized = stepper.prepare_roots()
    assert next_batch is None
    assert [event.slot_id for event in next_finalized] == [1]
    assert stepper.counts.games_completed == 1


def test_equal_policy_probabilities_remain_valid() -> None:
    stepper, _rngs = make_theoretical_stepper({0: chess.Board()}, max_plies=1)
    evaluator = FakeBatchEvaluator()  # uniform legal policy with many equal weights
    _batch, selected = run_fake_batch(stepper, evaluator, temperature=0.0)
    assert selected[0].move in chess.Board().legal_moves


def _match_stepper(
    boards: dict[int, chess.Board], fake: FakeMatchTablebase, *, max_plies: int,
) -> policy.BT4RootPolicyStepper:
    return policy.BT4RootPolicyStepper(
        boards, {slot: np.random.default_rng(100 + slot) for slot in boards},
        max_plies=max_plies, syzygy_path="fake:pair",
        input_history_encoding=HISTORY, input_extra_features=FEATURES,
        history_rep_fix=False, model_sha256=MODEL_SHA,
        outcome_mode="rule50_match_v1",
        match_tablebase=cast(chess.syzygy.Tablebase, cast(object, fake)),
    )


def test_match_capture_resolves_before_second_inference_and_stamps_mode() -> None:
    fake = FakeMatchTablebase(-2, -1)
    stepper = _match_stepper({0: chess.Board(SEVEN_MAN)}, fake, max_plies=8)
    evaluator = FakeBatchEvaluator(preferred_uci="b1c2")
    _batch, choices = run_fake_batch(stepper, evaluator, temperature=0.0)
    assert choices[0].move.uci() == "b1c2"

    next_batch, finalized = stepper.prepare_roots()
    assert next_batch is None
    assert len(finalized) == 1
    game = finalized[0]
    assert isinstance(game, policy.BT4CompletedGame)
    assert (game.result, game.termination, game.detail) == (
        "1-0", "syzygy", "rule50_match_wdl_dtz",
    )
    assert game.outcome_provenance == stepper.outcome_provenance
    assert game.outcome_provenance.mode == "rule50_match_v1"
    assert game.outcome_provenance.max_pieces == 6
    assert game.outcome_provenance.wdl_table_count == 1
    assert game.outcome_provenance.dtz_table_count == 1
    assert game.records[0].played.teacher.wdl_raw.tolist() == [0.05, 0.10, 0.85]
    assert game.records[0].wdl_target == 0
    assert evaluator.calls == evaluator.rows == 1
    assert fake.probes == ["wdl", "dtz"]


def test_match_positive_clock_decisive_discard_is_not_draw_or_inferred() -> None:
    board = chess.Board("7k/8/8/8/8/8/8/KQBNR3 w - - 1 1")
    fake = FakeMatchTablebase(2, 1)
    stepper = _match_stepper({0: board}, fake, max_plies=8)
    batch, finalized = stepper.prepare_roots()
    assert batch is None
    assert finalized == (
        policy.BT4DiscardedGame(
            0, "rule50_unresolved", "positive_clock_decisive_wdl", 0, 0,
            stepper.outcome_provenance,
        ),
    )
    assert stepper.counts.discarded_by_termination == {"rule50_unresolved": 1}
    assert stepper.counts.rows_emitted == 0
    assert fake.probes == ["wdl", "dtz"]


def test_match_missing_probe_fails_before_other_slot_finalizes_and_can_retry() -> None:
    checkmated = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    six_man = chess.Board("7k/8/8/8/8/8/8/KQBNR3 w - - 0 1")
    fake = FakeMatchTablebase(None, 1)
    stepper = _match_stepper({0: checkmated, 1: six_man}, fake, max_plies=8)
    with pytest.raises(tablebase.MatchTablebaseError, match="missing eligible"):
        stepper.prepare_roots()
    assert stepper.counts.games_completed == 0
    assert stepper.counts.games_discarded == 0
    assert stepper.counts.rows_emitted == stepper.counts.rows_discarded == 0
    fake.raw_wdl = 2
    batch, finalized = stepper.prepare_roots()
    assert batch is None
    assert len(finalized) == 2
    assert all(isinstance(event, policy.BT4CompletedGame) for event in finalized)
    assert [
        event.result for event in finalized
        if isinstance(event, policy.BT4CompletedGame)
    ] == ["1-0", "1-0"]
    assert stepper.counts.games_completed == 2
    assert stepper.counts.games_discarded == 0


def test_match_requires_explicit_mode_and_covered_six_man_capacity() -> None:
    fake = FakeMatchTablebase(2, 1)
    with pytest.raises(TypeError, match="outcome_mode"):
        cast(Any, policy.BT4RootPolicyStepper)(
            {0: chess.Board()}, {0: np.random.default_rng(100)},
            max_plies=8, syzygy_path="fake:pair",
            input_history_encoding=HISTORY, input_extra_features=FEATURES,
            history_rep_fix=False, model_sha256=MODEL_SHA,
        )
    with pytest.raises(ValueError, match="opened strict tablebase"):
        policy.BT4RootPolicyStepper(
            {0: chess.Board()}, {0: np.random.default_rng(100)},
            max_plies=8, syzygy_path="fake:pair",
            input_history_encoding=HISTORY, input_extra_features=FEATURES,
            history_rep_fix=False, model_sha256=MODEL_SHA,
            outcome_mode="rule50_match_v1",
        )
    fake.dtz = {"KQvK": object()}
    with pytest.raises(tablebase.MatchTablebaseError, match="capacity"):
        _match_stepper({0: chess.Board()}, fake, max_plies=8)


def test_mixed_outcome_mode_rejects_all_staged_events_before_state_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkmated = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    six_man = chess.Board("7k/8/8/8/8/8/8/KQBNR3 w - - 0 1")
    stepper = _match_stepper(
        {0: checkmated, 1: six_man}, FakeMatchTablebase(2, 1), max_plies=8,
    )
    original = policy.decide_bt4_outcome

    def wrong_second(
        board: chess.Board, *, plies: int, max_plies: int,
        syzygy_path: str, outcome_mode: policy.BT4OutcomeMode,
        match_tablebase: chess.syzygy.Tablebase | None,
    ) -> BT4OutcomeDecision | None:
        if board.turn == chess.WHITE:
            return BT4OutcomeDecision("1-0", "syzygy", "wrong_mode")
        return original(
            board, plies=plies, max_plies=max_plies,
            syzygy_path=syzygy_path, outcome_mode=outcome_mode,
            match_tablebase=match_tablebase,
        )

    monkeypatch.setattr(policy, "decide_bt4_outcome", wrong_second)
    with pytest.raises(ValueError, match="outcome mode differs"):
        stepper.prepare_roots()
    assert stepper.counts.games_completed == 0
    assert stepper.counts.games_discarded == 0
    assert stepper.counts.rows_emitted == stepper.counts.rows_discarded == 0
