"""In-memory, batch-friendly BT4 root-policy games; no generator or writer.

Call ``prepare_roots`` before inference, evaluate its copied boards and inputs
with an external batched evaluator, then call ``apply_root_outputs`` exactly
once. Natural and six-man outcomes are decided before a root is prepared.
Only completed games expose labeled records; unresolved games discard the
entire buffered history and report its size.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import chess
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.encoding.encode import input_plane_count
from chess_anti_engine.encoding.lc0 import normalize_lc0_history_encoding
from chess_anti_engine.eval.rvg_surgery import position_fingerprints
from chess_anti_engine.mcts.sampling import sample_action_with_temperature
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.leela_index import compact_index_for_move
from chess_anti_engine.selfplay.bt4_outcome import (
    BT4OutcomeDecision,
    decide_bt4_outcome,
)
from chess_anti_engine.selfplay.game import _result_to_wdl
from scripts.bt4_generation_evaluator import BT4RootOutput
from scripts.bt4_raw_corpus_sidecar import validate_wdl_values
from scripts.gen_sf_rooted_corpus import input_tensor_key


@dataclass(frozen=True)
class PreparedRoot:
    slot_id: int
    generation: int
    board: chess.Board
    x: np.ndarray
    fen: str
    input_key: str
    source_key: bytes


@dataclass(frozen=True)
class PreparedBatch:
    generation: int
    roots: tuple[PreparedRoot, ...]

    def inference_inputs(self) -> tuple[list[chess.Board], np.ndarray]:
        """Fresh copies for ``BT4OnnxEvaluator.evaluate_roots``."""
        return (
            [root.board.copy(stack=True) for root in self.roots],
            np.stack([root.x.copy() for root in self.roots]),
        )


@dataclass(frozen=True)
class BT4PlayedPly:
    """The retained teacher is separate from the sampled actor move."""

    x: np.ndarray  # exact float32 root input, including encoded history
    teacher: BT4RootOutput
    move: chess.Move
    pov_white: bool
    ply_index: int
    temperature: float


@dataclass(frozen=True)
class BT4LabeledPly:
    played: BT4PlayedPly
    wdl_target: int  # 0=W, 1=D, 2=L from this ply's side to move


@dataclass(frozen=True)
class BT4CompletedGame:
    slot_id: int
    result: str
    termination: str
    detail: str
    records: tuple[BT4LabeledPly, ...]


@dataclass(frozen=True)
class BT4DiscardedGame:
    slot_id: int
    termination: str
    detail: str
    attempted_plies: int
    discarded_rows: int


BT4FinalizedGame = BT4CompletedGame | BT4DiscardedGame


@dataclass(frozen=True)
class BT4SelectedMove:
    slot_id: int
    move: chess.Move
    temperature: float


@dataclass(frozen=True)
class BT4GameCounts:
    games_started: int
    games_completed: int
    games_discarded: int
    rows_emitted: int
    rows_discarded: int
    discarded_by_termination: dict[str, int]


@dataclass
class _Game:
    board: chess.Board
    rng: np.random.Generator
    records: list[BT4PlayedPly] = field(default_factory=list)
    done: bool = False


@dataclass(frozen=True)
class _ExpectedRoot:
    slot_id: int
    generation: int
    fen: str
    move_stack: tuple[chess.Move, ...]
    x_shape: tuple[int, ...]
    x_bytes: bytes
    input_key: str
    source_key: bytes


def _immutable_array(array: np.ndarray) -> np.ndarray:
    """Copy into a bytes-backed ndarray whose write flag cannot be re-enabled."""
    arr = np.asarray(array)
    return np.frombuffer(arr.tobytes(), dtype=arr.dtype).reshape(arr.shape)


def _immutable_teacher(root: BT4RootOutput) -> BT4RootOutput:
    return replace(
        root,
        policy_t1=_immutable_array(root.policy_t1),
        wdl_raw=_immutable_array(root.wdl_raw),
    )


class BT4RootPolicyStepper:
    """Own several board histories until each resolves or is discarded.

    The caller must preflight the Syzygy pair once per worker. This class does
    not invoke a model, choose a hidden temperature, or write replay shards.
    """

    def __init__(
        self,
        boards: Mapping[int, chess.Board],
        rngs: Mapping[int, np.random.Generator],
        *,
        max_plies: int,
        syzygy_path: str,
        input_history_encoding: str,
        input_extra_features: str,
        history_rep_fix: bool,
        model_sha256: str,
    ) -> None:
        if not boards or set(boards) != set(rngs):
            raise ValueError("BT4 games need matching nonempty board/RNG slots")
        if max_plies <= 0 or not syzygy_path:
            raise ValueError("BT4 games need positive max_plies and a Syzygy path")
        if len(model_sha256) != 64 or any(c not in "0123456789abcdef" for c in model_sha256):
            raise ValueError("BT4 games need a lowercase model SHA-256")
        self.input_history_encoding = normalize_lc0_history_encoding(input_history_encoding)
        self.input_extra_features = input_extra_features
        input_plane_count(input_extra_features)
        if type(history_rep_fix) is not bool:
            raise TypeError("BT4 history_rep_fix must be an explicit bool")  # pyright: ignore[reportUnreachable]
        self.history_rep_fix = history_rep_fix
        self._require_history_rep_fix()
        self.model_sha256 = model_sha256
        self.max_plies = int(max_plies)
        self.syzygy_path = syzygy_path
        self._games: dict[int, _Game] = {}
        for slot_id, board in boards.items():
            if not isinstance(slot_id, int) or not isinstance(board, chess.Board):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError("BT4 slots need integer IDs and chess boards")
            rng = rngs[slot_id]
            if not isinstance(rng, np.random.Generator):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError("BT4 slots need explicit NumPy Generators")
            self._games[slot_id] = _Game(board=board.copy(stack=True), rng=rng)
        self._generation = 0
        self._pending: PreparedBatch | None = None
        self._expected: tuple[_ExpectedRoot, ...] = ()
        self._output_contract: tuple[str, ...] | None = None
        self._completed = 0
        self._discarded = 0
        self._rows_emitted = 0
        self._rows_discarded = 0
        self._discarded_by_termination: dict[str, int] = {}

    @property
    def counts(self) -> BT4GameCounts:
        return BT4GameCounts(
            games_started=len(self._games), games_completed=self._completed,
            games_discarded=self._discarded, rows_emitted=self._rows_emitted,
            rows_discarded=self._rows_discarded,
            discarded_by_termination=dict(self._discarded_by_termination),
        )

    def _require_history_rep_fix(self) -> None:
        if rep_fix.current() is not self.history_rep_fix:
            raise RuntimeError(
                "BT4 stepper history_rep_fix differs from the process encoder mode"
            )

    def _finalize(self, slot_id: int, decision: BT4OutcomeDecision) -> BT4FinalizedGame:
        if decision.result not in (None, "1-0", "0-1", "1/2-1/2"):
            raise ValueError(f"BT4 outcome has unsupported result {decision.result!r}")
        game = self._games[slot_id]
        game.done = True
        attempted = len(game.records)
        if decision.result is None:
            self._discarded += 1
            self._rows_discarded += attempted
            self._discarded_by_termination[decision.termination] = (
                self._discarded_by_termination.get(decision.termination, 0) + 1
            )
            game.records.clear()
            return BT4DiscardedGame(
                slot_id, decision.termination, decision.detail, attempted, attempted,
            )
        labeled = tuple(
            BT4LabeledPly(
                rec, int(_result_to_wdl(decision.result, pov_white=rec.pov_white)),
            )
            for rec in game.records
        )
        game.records.clear()
        self._completed += 1
        self._rows_emitted += len(labeled)
        return BT4CompletedGame(
            slot_id, decision.result, decision.termination, decision.detail, labeled,
        )

    def prepare_roots(self) -> tuple[PreparedBatch | None, tuple[BT4FinalizedGame, ...]]:
        """Decide terminal/TB/cap first, then expose only playable root copies."""
        if self._pending is not None:
            raise RuntimeError("BT4 prepared roots are still awaiting outputs")
        self._require_history_rep_fix()
        generation = self._generation + 1
        roots: list[PreparedRoot] = []
        expected: list[_ExpectedRoot] = []
        decisions: list[tuple[int, BT4OutcomeDecision]] = []
        for slot_id, game in sorted(self._games.items()):
            if game.done:
                continue
            decision = decide_bt4_outcome(
                game.board, plies=len(game.records), max_plies=self.max_plies,
                syzygy_path=self.syzygy_path,
            )
            if decision is not None:
                if decision.result not in (None, "1-0", "0-1", "1/2-1/2"):
                    raise ValueError(f"BT4 outcome has unsupported result {decision.result!r}")
                decisions.append((slot_id, decision))
                continue
            x = np.asarray(encode_cboard(
                CBoard.from_board(game.board),
                input_history_encoding=self.input_history_encoding,
                input_extra_features=self.input_extra_features,
            ), dtype=np.float32)
            key = input_tensor_key(x)
            source_key = position_fingerprints(
                x[None], input_history_encoding=self.input_history_encoding,
            )[0]
            fen = game.board.fen()
            roots.append(PreparedRoot(
                slot_id, generation, game.board.copy(stack=True), x.copy(),
                fen, key, source_key,
            ))
            expected.append(_ExpectedRoot(
                slot_id, generation, fen, tuple(game.board.move_stack),
                x.shape, x.tobytes(), key, source_key,
            ))
        self._generation = generation
        finalized = tuple(self._finalize(slot_id, decision) for slot_id, decision in decisions)
        if not roots:
            return None, finalized
        batch = PreparedBatch(generation, tuple(roots))
        self._pending = batch
        self._expected = tuple(expected)
        return batch, finalized

    def _validate_root(
        self, prepared: PreparedRoot, expected: _ExpectedRoot,
        output: BT4RootOutput,
    ) -> tuple[list[chess.Move], np.ndarray, BT4RootOutput, np.ndarray]:
        if (
            prepared.slot_id != expected.slot_id
            or prepared.generation != expected.generation
            or prepared.fen != expected.fen
            or prepared.board.fen() != expected.fen
            or tuple(prepared.board.move_stack) != expected.move_stack
            or prepared.x.dtype != np.dtype(np.float32)
            or prepared.x.shape != expected.x_shape
            or not prepared.x.flags.c_contiguous
            or prepared.x.tobytes() != expected.x_bytes
            or prepared.input_key != expected.input_key
            or prepared.source_key != expected.source_key
        ):
            raise ValueError("BT4 prepared root changed or is out of order")
        if (
            not isinstance(output, BT4RootOutput)  # pyright: ignore[reportUnnecessaryIsInstance]
            or output.fen != expected.fen
            or output.input_key != expected.input_key
            or output.source_key != expected.source_key
            or output.input_history_encoding != self.input_history_encoding
            or output.input_extra_features != self.input_extra_features
            or output.history_rep_fix is not self.history_rep_fix
            or output.model_sha256 != self.model_sha256
        ):
            raise ValueError("BT4 output belongs to another root or model")
        policy = output.policy_t1
        raw = output.wdl_raw
        if (
            policy.shape != (COMPACT_POLICY_SIZE,)
            or policy.dtype != np.dtype(np.float32)
            or policy.flags.writeable
            or not np.isfinite(policy).all()
            or bool(np.any(policy < 0))
            or raw.shape != (3,)
            or raw.dtype not in (
                np.dtype("float16"), np.dtype("float32"), np.dtype("float64")
            )
            or raw.flags.writeable
            or not np.isfinite(raw).all()
        ):
            raise ValueError("BT4 root teacher tensors violate the output contract")
        if output.wdl_kind not in ("probabilities", "logits"):
            raise ValueError("BT4 root WDL kind is unsupported")
        validate_wdl_values(
            raw[None], 1, {"dtype": raw.dtype.name, "kind": output.wdl_kind},
        )
        moves = list(prepared.board.legal_moves)
        indices = np.asarray(
            [compact_index_for_move(prepared.board, move) for move in moves],
            dtype=np.int64,
        )
        if len(moves) == 0 or len(set(indices.tolist())) != len(moves):
            raise ValueError("BT4 root has no distinct legal policy slots")
        legal_mask = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.bool_)
        legal_mask[indices] = True
        weights = policy[indices].astype(np.float64, copy=True)
        if (
            bool(np.any(policy[~legal_mask] != 0.0))
            or not np.isclose(float(weights.sum()), 1.0, rtol=0, atol=2e-6)
        ):
            raise ValueError("BT4 root policy is not normalized on legal moves")
        return moves, weights, _immutable_teacher(output), _immutable_array(prepared.x)

    def apply_root_outputs(
        self, batch: PreparedBatch, outputs: Sequence[BT4RootOutput],
        *, temperatures: Mapping[int, float],
    ) -> tuple[BT4SelectedMove, ...]:
        """Validate the whole batch before any RNG draw or board mutation."""
        if self._pending is not batch or batch.generation != self._generation:
            raise RuntimeError("BT4 prepared batch is stale or already consumed")
        self._require_history_rep_fix()
        if len(outputs) != len(batch.roots) or len(outputs) != len(self._expected):
            raise ValueError("BT4 output count disagrees with prepared roots")
        slots = {root.slot_id for root in batch.roots}
        if set(temperatures) != slots:
            raise ValueError("BT4 needs one explicit temperature per prepared slot")
        validated: list[
            tuple[int, float, list[chess.Move], np.ndarray, BT4RootOutput, np.ndarray]
        ] = []
        output_contract = self._output_contract
        for prepared, expected, output in zip(batch.roots, self._expected, outputs):
            temperature = float(temperatures[prepared.slot_id])
            if not math.isfinite(temperature) or temperature < 0:
                raise ValueError("BT4 temperature must be finite and nonnegative")
            moves, weights, teacher, x = self._validate_root(prepared, expected, output)
            contract = (
                output.policy_output, output.wdl_output, output.wdl_kind,
                output.input_name, output.input_dtype, output.wdl_raw.dtype.name,
            )
            if output_contract is not None and contract != output_contract:
                raise ValueError("BT4 batch mixes output contracts")
            output_contract = contract
            validated.append((prepared.slot_id, temperature, moves, weights, teacher, x))

        selections: list[BT4SelectedMove] = []
        for slot_id, temperature, moves, weights, _teacher, _x in validated:
            actions = np.arange(len(moves), dtype=np.int32)
            index = sample_action_with_temperature(
                self._games[slot_id].rng, actions, weights, temperature,
                argmax_idx=int(np.argmax(weights)),
            )
            selections.append(BT4SelectedMove(slot_id, moves[index], temperature))

        for selected, (_slot_id, _temperature, _moves, _weights, teacher, x) in zip(
            selections, validated,
        ):
            game = self._games[selected.slot_id]
            game.records.append(BT4PlayedPly(
                x=x, teacher=teacher, move=selected.move,
                pov_white=bool(game.board.turn), ply_index=game.board.ply(),
                temperature=selected.temperature,
            ))
            game.board.push(selected.move)
        self._pending = None
        self._expected = ()
        self._output_contract = output_contract
        return tuple(selections)
