"""Actual native stepper controls, independent of pytest and model startup.

Synthetic teacher values exercise real output validation and sampling; board,
history encoding, paired apply and RNG operations are actual production code.
Every fixture stays above six pieces, so real Syzygy decoding is not covered.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import sys
import threading
import time
import unittest

import chess
import numpy as np

from chess_anti_engine.encoding import _lc0_ext, rep_fix
from chess_anti_engine.encoding.cboard_encode import encode_cboard_batch
from chess_anti_engine.generation_runtime import ExclusiveLaneRegistry
from chess_anti_engine.mcts import _mcts_tree
from chess_anti_engine.producer_executor import ProducerExecutor, ProducerJoinTimeout
from scripts import bt4_root_policy_stepper as subject
from scripts.bt4_root_contracts import BT4RootOutput, input_tensor_key, validate_wdl_values


HISTORY, FEATURES, MODEL = "lc0_root_legacy_meta", "v2_threats", "a" * 64
HISTORIES = (
    "g1f3 g8f6 f3g1 f6g8 e2e4",
    "e2e4 a7a6 e4e5 d7d5",
    "g1f3 g8f6 e2e3 e7e6 f1e2 f8e7 e1g1 e8g8",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
)
Prepared = tuple[tuple[subject.PreparedBatch | None, tuple[subject.BT4FinalizedGame, ...]], ...]
Moves = tuple[tuple[subject.BT4SelectedMove, ...], ...]
Request = tuple[subject.BT4RootPolicyStepper, subject.PreparedBatch, Sequence[BT4RootOutput], Mapping[int, float]]
_RETAINED_COMPONENT_FAILURES: list[tuple[object, ...]] = []

def raises(expected: type[Exception], operation: Callable[[], object], text: str = "") -> None:
    caught: Exception | None = None
    try:
        operation()
    except expected as exc:
        caught = exc
    assert caught is not None
    assert text in str(caught)


def construct() -> tuple[list[subject.BT4RootPolicyStepper], list[np.random.Generator]]:
    steppers: list[subject.BT4RootPolicyStepper] = []
    rngs: list[np.random.Generator] = []
    for game_id, history in enumerate(HISTORIES):
        board = chess.Board()
        for move in history.split():
            board.push_uci(move)
        rng = np.random.default_rng(np.random.SeedSequence([123456789, game_id]))
        steppers.append(subject.BT4RootPolicyStepper(
            {game_id: board}, {game_id: rng}, max_plies=400, syzygy_path="not-probed",
            input_history_encoding=HISTORY, input_extra_features=FEATURES,
            history_rep_fix=True, model_sha256=MODEL, outcome_mode="theoretical_wdl",
        ))
        rngs.append(rng)
    return steppers, rngs


def outputs(batch: subject.PreparedBatch) -> tuple[BT4RootOutput, ...]:
    result: list[BT4RootOutput] = []
    for root in batch.dispatch_inputs():
        dense = np.zeros(1858, dtype=np.float32)
        legal = root.legal.compact_indices
        dense[legal] = np.float32(1 / len(legal))
        dense.flags.writeable = False
        raw = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        raw.flags.writeable = False
        result.append(BT4RootOutput(
            fen=root.fen, input_key=root.input_key, source_key=root.source_key,
            policy_t1=dense, wdl_raw=raw, policy_output="synthetic_policy",
            wdl_output="synthetic_wdl", wdl_kind="probabilities", model_sha256=MODEL,
            input_name="planes", input_dtype="float32", input_history_encoding=HISTORY,
            input_extra_features=FEATURES, history_rep_fix=True,
        ))
    return tuple(result)


def encoded_state(steppers: list[subject.BT4RootPolicyStepper]) -> tuple[object, ...]:
    result = []
    for stepper in steppers:
        game_id, game = next(iter(stepper._games.items()))
        reconstructed = _lc0_ext.CBoard.from_board(game.board)
        both = encode_cboard_batch([game.cboard, reconstructed],
                                  input_history_encoding=HISTORY, input_extra_features=FEATURES)
        if both[0].tobytes() != both[1].tobytes():
            raise AssertionError("incremental and reconstructed native histories differ")
        result.append((game_id, game.board.fen(), tuple(move.uci() for move in game.board.move_stack),
                       game.cboard.fen(), game.cboard.zobrist_hash, game.cboard.hist_len,
                       game.cboard.hash_stack_len, game.cboard.ply, both[0].tobytes(), len(game.records)))
    return tuple(result)


def prepared_state(prepared: Prepared) -> tuple[object, ...]:
    result = []
    for batch, events in prepared:
        if events:
            raise AssertionError("short above-six-piece fixtures unexpectedly finalized")
        if batch is None:
            raise AssertionError("expected actual prepared roots")
        result.append(tuple((root.slot_id, root.generation, root.fen, root.x.shape,
                             root.x.dtype.str, root.x.tobytes(), root.input_key, root.source_key,
                             tuple(move.uci() for move in root.board.move_stack),
                             root.legal.compact_indices.tobytes()) for root in batch.dispatch_inputs()))
    return tuple(result)


def requests_for(steppers: Sequence[subject.BT4RootPolicyStepper], prepared: Prepared) -> list[Request]:
    result: list[Request] = []
    for stepper, (batch, events) in zip(steppers, prepared, strict=True):
        if batch is None or events:
            raise AssertionError("actual prepared batch required")
        temperatures = {root.slot_id: 1.0 for root in batch.dispatch_inputs()}
        result.append((stepper, batch, outputs(batch), temperatures))
    return result


def transition(steppers: list[subject.BT4RootPolicyStepper], prepared: Prepared,
               executor: ProducerExecutor | None = None) -> tuple[Moves, Prepared]:
    def admit() -> bool:
        if executor is None:
            return True
        with executor.admission() as accepted:
            return accepted
    selected = subject.BT4RootPolicyStepper.apply_many(requests_for(steppers, prepared), admit=admit)
    if selected is None:
        raise AssertionError("unexpected rejected transition")
    return selected, subject.BT4RootPolicyStepper.prepare_many(steppers)


class NativeStepperComponentControls(unittest.TestCase):
    def setUp(self) -> None:
        if _RETAINED_COMPONENT_FAILURES:
            raise RuntimeError("unresolved producer ownership retained; no further native work")

    @classmethod
    def setUpClass(cls) -> None:
        if "torch" in sys.modules or "onnxruntime" in sys.modules:
            raise AssertionError("native component must not import model runtimes")
        if not callable(getattr(_lc0_ext, "encode_full_batch", None)):
            raise AssertionError("actual native batch encoder required; no fallback")
        if not callable(getattr(_lc0_ext, "set_history_rep_fix", None)):
            raise AssertionError("actual native history setter required")
        if not callable(getattr(_mcts_tree, "set_history_rep_fix", None)):
            raise AssertionError("actual secondary native history setter required")
        # Dedicated bounded component process; no prior live board survives.
        rep_fix.apply(True, boards_discarded=True)
        if rep_fix.current() is not True:
            raise AssertionError("actual process history mode not True")

    def test_serial_equals_two_disjoint_producers_native_history_apply_and_rng(self) -> None:
        parallel, parallel_rng = construct()
        serial, serial_rng = construct()
        lanes = ExclusiveLaneRegistry()
        executor = ProducerExecutor(max_workers=2, capacity=2)
        parallel_prepared: list[tuple[subject.PreparedBatch | None, tuple[subject.BT4FinalizedGame, ...]]] = []
        primary: BaseException | None = None
        cleanup: BaseException | None = None
        try:
            prepare_tasks = []
            for start in (0, 2):
                keys = tuple(("private", game) for game in range(start, start + 2))
                lease = lanes.admit(keys)
                future = executor.submit(subject.BT4RootPolicyStepper.prepare_many, parallel[start:start + 2])
                prepare_tasks.append((future, lease))
            for future, lease in prepare_tasks:
                parallel_prepared.extend(future.result(timeout=5))
                lanes.complete(lease)
            serial_prepared = subject.BT4RootPolicyStepper.prepare_many(serial)
            assert prepared_state(tuple(parallel_prepared)) == prepared_state(serial_prepared)
            for _round in range(3):
                transition_tasks = []
                for start in (0, 2):
                    lease = lanes.admit(tuple(("private", i) for i in range(start, start + 2)))
                    job = executor.submit(transition, parallel[start:start + 2],
                                          tuple(parallel_prepared[start:start + 2]), executor)
                    transition_tasks.append((job, lease))
                parallel_prepared = []
                parallel_moves: list[tuple[subject.BT4SelectedMove, ...]] = []
                for future, lease in transition_tasks:
                    moves, pending = future.result(timeout=5)
                    parallel_moves.extend(moves)
                    parallel_prepared.extend(pending)
                    lanes.complete(lease)
                serial_moves, serial_prepared = transition(serial, serial_prepared)
                assert tuple(parallel_moves) == serial_moves
                assert prepared_state(tuple(parallel_prepared)) == prepared_state(serial_prepared)
                assert encoded_state(parallel) == encoded_state(serial)
                assert ([json.dumps(r.bit_generator.state, sort_keys=True) for r in parallel_rng]
                        == [json.dumps(r.bit_generator.state, sort_keys=True) for r in serial_rng])
        except BaseException as exc:
            primary = exc
        finally:
            try:
                executor.finish(timeout=5)
            except BaseException as exc:
                cleanup = exc
            finally:
                join_deadline = time.monotonic() + 5
                join_errors: list[BaseException] = []
                try:
                    threads = tuple(threading.enumerate())
                except BaseException as exc:
                    join_errors.append(exc)
                    threads = ()
                for thread in threads:
                    if thread.name.startswith("GameProducer"):
                        try:
                            thread.join(max(0.0, join_deadline - time.monotonic()))
                        except BaseException as exc:
                            join_errors.append(exc)
                try:
                    live: tuple[threading.Thread, ...] | None = tuple(
                        thread for thread in threading.enumerate()
                        if thread.name.startswith("GameProducer"))
                except BaseException as exc:
                    join_errors.append(exc)
                    live = None
                if live or live is None or join_errors:
                    unresolved = ProducerJoinTimeout(
                        "component producer join failed or unresolved; ownership retained",
                        executor, lanes, parallel, parallel_rng, serial, serial_rng, live, join_errors,
                    )
                    _RETAINED_COMPONENT_FAILURES.append((unresolved, primary, cleanup))
                    cleanup = unresolved
                elif cleanup is not None:
                    _RETAINED_COMPONENT_FAILURES.append(
                        (executor, lanes, parallel, parallel_rng, serial, serial_rng, primary, cleanup),
                    )
        if primary is not None:
            if cleanup is not None and cleanup is not primary:
                raise primary from cleanup
            raise primary
        if cleanup is not None:
            raise cleanup
        assert not live
        assert rep_fix.current() is True
        assert "torch" not in sys.modules
        assert "onnxruntime" not in sys.modules

    def test_rejected_transition_and_duplicate_lane_do_not_consume_rng(self) -> None:
        steppers, rngs = construct()
        pending = subject.BT4RootPolicyStepper.prepare_many(steppers)
        before = [deepcopy(r.bit_generator.state) for r in rngs]
        state = encoded_state(steppers)
        requests = requests_for(steppers, pending)
        assert subject.BT4RootPolicyStepper.apply_many(requests, admit=lambda: False) is None
        assert before == [r.bit_generator.state for r in rngs]
        assert state == encoded_state(steppers)
        raises(ValueError, lambda: subject.BT4RootPolicyStepper.apply_many((requests[0], requests[0])),
               "distinct stepper identities")
        assert before == [r.bit_generator.state for r in rngs]

    def test_live_native_mode_flip_is_rejected_without_changing_mode(self) -> None:
        steppers, _rngs = construct()
        before = encoded_state(steppers)
        raises(rep_fix.RepFixFlipError, lambda: rep_fix.apply(False))
        assert rep_fix.current() is True
        assert before == encoded_state(steppers)

    def test_pure_hash_and_wdl_negative_contracts(self) -> None:
        values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert input_tensor_key(values) == input_tensor_key(values.astype(np.float32))
        assert (input_tensor_key(values[:, :, ::-1])
                == input_tensor_key(np.ascontiguousarray(values[:, :, ::-1], dtype=np.float32)))
        contract = {"dtype": "float32", "kind": "probabilities"}
        valid = np.array([[0.25, 0.5, 0.25]], dtype=np.float32)
        validate_wdl_values(valid, 1, contract)
        for bad in (np.array([[0.25, 0.5, 0.2501]], dtype=np.float32),
                    np.array([[np.nan, 0.5, 0.5]], dtype=np.float32),
                    np.array([[-0.25, 0.75, 0.5]], dtype=np.float32)):
            raises(ValueError, lambda bad=bad: validate_wdl_values(bad, 1, contract))
        raises(ValueError, lambda: validate_wdl_values(valid.astype(np.float64), 1, contract),
               "shape/native dtype")


if __name__ == "__main__":
    unittest.main()
