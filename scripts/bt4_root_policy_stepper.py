"""In-memory, batch-friendly BT4 root-policy games; no generator or writer.

Call ``prepare_roots`` (or ``prepare_many`` across steppers) before inference,
evaluate the copied boards and inputs
with an external batched evaluator, then call ``apply_root_outputs`` exactly
once. Natural and six-man outcomes are decided before a root is prepared.
Only completed games expose labeled records; unresolved games discard the
entire buffered history and report its size.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from typing import cast

import chess
import chess.syzygy
import numpy as np

from chess_anti_engine import tablebase
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding.cboard_encode import encode_cboard_batch
from chess_anti_engine.encoding.encode import input_plane_count
from chess_anti_engine.encoding.lc0 import normalize_lc0_history_encoding
from chess_anti_engine.eval.rvg_surgery import position_fingerprints
from chess_anti_engine.mcts.sampling import sample_action_with_temperature
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.selfplay.bt4_outcome import (
    BT4OutcomeDecision,
    BT4OutcomeMode,
    SIX_MAN_MAX_PIECES,
    decide_bt4_outcome,
)
from chess_anti_engine.selfplay.game import _result_to_wdl
from scripts.bt4_generation_evaluator import BT4RootOutput
from scripts.bt4_model_arithmetic import validate_model_wdl
from scripts.bt4_prepared_root import (
    BT4LegalSnapshot,
    BT4PreparedInput,
    copy_history_board,
    immutable_array,
)
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


@dataclass(frozen=True, init=False, eq=False)
class PreparedBatch:
    """Canonical owned rows with a lazy, defensive public compatibility view."""

    generation: int
    __inputs: tuple[BT4PreparedInput, ...]
    __roots: tuple[PreparedRoot, ...] | None

    def __init__(self, generation: int, roots: tuple[PreparedRoot, ...]) -> None:
        # Keep the public constructor, but arbitrary public roots cannot issue
        # an internal transfer or masquerade as the stepper's pending batch.
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "_PreparedBatch__inputs", ())
        object.__setattr__(self, "_PreparedBatch__roots", roots)

    @classmethod
    def _from_inputs(
        cls, generation: int, inputs: tuple[BT4PreparedInput, ...],
    ) -> PreparedBatch:
        batch = cls(generation, ())
        object.__setattr__(batch, "_PreparedBatch__inputs", inputs)
        object.__setattr__(batch, "_PreparedBatch__roots", None)
        return batch

    @property
    def roots(self) -> tuple[PreparedRoot, ...]:
        roots = self.__roots
        if roots is None:
            roots = tuple(PreparedRoot(
                root.slot_id, root.generation, root.board, root.x.copy(),
                root.fen, root.input_key, root.source_key,
            ) for root in self.__inputs)
            object.__setattr__(self, "_PreparedBatch__roots", roots)
        return roots

    def dispatch_inputs(self) -> tuple[BT4PreparedInput, ...]:
        """Stable private transfers; never materialize the public root copies."""
        if not self.__inputs:
            raise ValueError("BT4 prepared inputs require canonical preparation")
        self._validate_public_roots()
        return self.__inputs

    def _validate_public_roots(self) -> None:
        roots = self.__roots
        if roots is None:
            return
        if len(roots) != len(self.__inputs):
            raise ValueError("BT4 prepared root changed or is out of order")
        for root, owned in zip(roots, self.__inputs):
            if (
                root.slot_id != owned.slot_id or root.generation != owned.generation
                or root.fen != owned.fen or not owned._matches_board(root.board)
                or root.x.dtype != np.dtype(np.float32)
                or root.x.shape != owned.x.shape or not root.x.flags.c_contiguous
                or root.x.tobytes() != owned._input_bytes()
                or root.input_key != owned.input_key or root.source_key != owned.source_key
            ):
                raise ValueError("BT4 prepared root changed or is out of order")

    def inference_inputs(self) -> tuple[list[chess.Board], np.ndarray]:
        """Fresh mutable copies for strict public ``evaluate_roots`` callers."""
        return (
            [copy_history_board(root.board) for root in self.roots],
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
class BT4OutcomeProvenance:
    """In-memory outcome contract; a writer must separately pin actual TB files."""

    mode: BT4OutcomeMode
    syzygy_path: str
    max_pieces: int
    wdl_table_count: int | None
    dtz_table_count: int | None
    handle_contract: str


@dataclass(frozen=True)
class BT4CompletedGame:
    slot_id: int
    result: str
    termination: str
    detail: str
    records: tuple[BT4LabeledPly, ...]
    outcome_provenance: BT4OutcomeProvenance


@dataclass(frozen=True)
class BT4DiscardedGame:
    slot_id: int
    termination: str
    detail: str
    attempted_plies: int
    discarded_rows: int
    outcome_provenance: BT4OutcomeProvenance


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
    cboard: CBoard
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
    return immutable_array(array)


def _immutable_teacher(root: BT4RootOutput) -> BT4RootOutput:
    return replace(
        root,
        policy_t1=_immutable_array(root.policy_t1),
        wdl_raw=_immutable_array(root.wdl_raw),
    )


class BT4RootPolicyStepper:
    """Own several board histories until each resolves or is discarded.

    The caller must explicitly select the outcome mode. Match mode requires a
    caller-owned handle from ``open_strict_match_tablebase``; this class checks
    WDL/DTZ capacity but does not independently attest file identity. A future
    writer must retain that separate strict-open provenance. This class does
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
        outcome_mode: BT4OutcomeMode,
        match_tablebase: chess.syzygy.Tablebase | None = None,
    ) -> None:
        if not boards or set(boards) != set(rngs):
            raise ValueError("BT4 games need matching nonempty board/RNG slots")
        if max_plies <= 0 or not syzygy_path:
            raise ValueError("BT4 games need positive max_plies and a Syzygy path")
        if len(model_sha256) != 64 or any(c not in "0123456789abcdef" for c in model_sha256):
            raise ValueError("BT4 games need a lowercase model SHA-256")
        if cast(str, outcome_mode) not in ("theoretical_wdl", "rule50_match_v1"):
            raise ValueError("BT4 games need an explicit supported outcome mode")
        if outcome_mode == "rule50_match_v1":
            if match_tablebase is None:
                raise ValueError("BT4 rule50 match mode needs an opened strict tablebase")
            tablebase.SyzygyProbe(
                syzygy_path, max_pieces=SIX_MAN_MAX_PIECES,
                rule50_aware=True, tablebase=match_tablebase,
            )
            wdl_tables, dtz_tables = len(match_tablebase.wdl), len(match_tablebase.dtz)
            handle_contract = "caller_owned_capacity_checked"
        else:
            if match_tablebase is not None:
                raise ValueError("BT4 theoretical mode cannot ignore a match tablebase")
            wdl_tables = dtz_tables = None
            handle_contract = "historical_cached_path"
        self.outcome_mode: BT4OutcomeMode = outcome_mode
        self._match_tablebase = match_tablebase
        self.outcome_provenance = BT4OutcomeProvenance(
            mode=outcome_mode, syzygy_path=syzygy_path,
            max_pieces=SIX_MAN_MAX_PIECES,
            wdl_table_count=wdl_tables, dtz_table_count=dtz_tables,
            handle_contract=handle_contract,
        )
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
            owned = copy_history_board(board)
            self._games[slot_id] = _Game(
                board=owned, cboard=CBoard.from_board(owned), rng=rng,
            )
        self._poisoned: BaseException | None = None
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

    def _require_healthy(self) -> None:
        if self._poisoned is not None:
            raise RuntimeError("BT4 stepper is poisoned after a sampled or partial commit") from self._poisoned

    def _require_history_rep_fix(self) -> None:
        if rep_fix.current() is not self.history_rep_fix:
            raise RuntimeError(
                "BT4 stepper history_rep_fix differs from the process encoder mode"
            )

    def _prepare_finalized(
        self, slot_id: int, decision: BT4OutcomeDecision,
    ) -> BT4FinalizedGame:
        """Build an outcome event without consuming its buffered game history."""
        self._require_healthy()
        if decision.result not in (None, "1-0", "0-1", "1/2-1/2"):
            raise ValueError(f"BT4 outcome has unsupported result {decision.result!r}")
        if decision.outcome_mode != self.outcome_mode:
            raise ValueError("BT4 outcome mode differs from the stepper contract")
        game = self._games[slot_id]
        attempted = len(game.records)
        if decision.result is None:
            return BT4DiscardedGame(
                slot_id, decision.termination, decision.detail, attempted, attempted,
                self.outcome_provenance,
            )
        labeled = tuple(
            BT4LabeledPly(
                rec, int(_result_to_wdl(decision.result, pov_white=rec.pov_white)),
            )
            for rec in game.records
        )
        return BT4CompletedGame(
            slot_id, decision.result, decision.termination, decision.detail, labeled,
            self.outcome_provenance,
        )

    def _commit_finalized(self, event: BT4FinalizedGame) -> None:
        self._require_healthy()
        game = self._games[event.slot_id]
        game.done = True
        game.records.clear()
        if isinstance(event, BT4DiscardedGame):
            self._discarded += 1
            self._rows_discarded += event.discarded_rows
            self._discarded_by_termination[event.termination] = (
                self._discarded_by_termination.get(event.termination, 0) + 1
            )
        else:
            self._completed += 1
            self._rows_emitted += len(event.records)

    def prepare_roots(self) -> tuple[PreparedBatch | None, tuple[BT4FinalizedGame, ...]]:
        """Use the same transactional native preparation as a multi-stepper call."""
        return self.prepare_many((self,))[0]

    @staticmethod
    def prepare_many(
        steppers: Sequence[BT4RootPolicyStepper],
    ) -> tuple[tuple[PreparedBatch | None, tuple[BT4FinalizedGame, ...]], ...]:
        """Prepare independent steppers with one native encoding batch.

        Results retain the supplied stepper order and each batch's sorted slot
        order. Encoding settings must agree, but model, generation and outcome
        contracts remain independent. No stepper state changes until all outcome,
        encoding and root identity preparation has succeeded.
        """
        steppers = tuple(steppers)
        if not steppers:
            return ()
        if len({id(stepper) for stepper in steppers}) != len(steppers):
            raise ValueError("BT4 prepare_many needs distinct stepper identities")
        first = steppers[0]
        encoding_contract = (
            first.input_history_encoding, first.input_extra_features,
            first.history_rep_fix,
        )
        for stepper in steppers:
            stepper._require_healthy()
            if stepper._pending is not None:
                raise RuntimeError("BT4 prepared roots are still awaiting outputs")
            if (
                stepper.input_history_encoding, stepper.input_extra_features,
                stepper.history_rep_fix,
            ) != encoding_contract:
                raise ValueError("BT4 prepare_many needs identical encoding settings")
            stepper._require_history_rep_fix()

        boards_by_stepper: list[list[tuple[int, chess.Board]]] = []
        finalized_by_stepper: list[tuple[BT4FinalizedGame, ...]] = []
        cboards: list[CBoard] = []
        for stepper in steppers:
            boards: list[tuple[int, chess.Board]] = []
            finalized: list[BT4FinalizedGame] = []
            for slot_id, game in sorted(stepper._games.items()):
                if game.done:
                    continue
                decision = decide_bt4_outcome(
                    game.board, plies=len(game.records), max_plies=stepper.max_plies,
                    syzygy_path=stepper.syzygy_path, outcome_mode=stepper.outcome_mode,
                    match_tablebase=stepper._match_tablebase,
                )
                if decision is not None:
                    finalized.append(stepper._prepare_finalized(slot_id, decision))
                else:
                    boards.append((slot_id, game.board))
                    # The actor owns this paired state exclusively until prepare
                    # returns. The backend verifies it independently from Python.
                    cboards.append(game.cboard)
            boards_by_stepper.append(boards)
            finalized_by_stepper.append(tuple(finalized))

        x_batch = np.asarray(encode_cboard_batch(
            cboards,
            input_history_encoding=first.input_history_encoding,
            input_extra_features=first.input_extra_features,
        ), dtype=np.float32) if cboards else np.empty(
            (0, input_plane_count(first.input_extra_features), 8, 8), dtype=np.float32,
        )
        if x_batch.shape != (
            len(cboards), input_plane_count(first.input_extra_features), 8, 8,
        ):
            raise ValueError("BT4 native encoding batch has the wrong shape")
        source_keys = position_fingerprints(
            x_batch, input_history_encoding=first.input_history_encoding,
        )
        pending: list[PreparedBatch | None] = []
        expected_by_stepper: list[tuple[_ExpectedRoot, ...]] = []
        generations = [stepper._generation + 1 for stepper in steppers]
        row = 0
        for stepper, generation, boards in zip(steppers, generations, boards_by_stepper):
            inputs: list[BT4PreparedInput] = []
            expected: list[_ExpectedRoot] = []
            for slot_id, board in boards:
                x = x_batch[row]
                key = input_tensor_key(x)
                source_key = source_keys[row]
                owned = BT4PreparedInput._issue(
                    slot_id=slot_id, generation=generation, board=board, x=x,
                    input_key=key, source_key=source_key, model_sha256=stepper.model_sha256,
                    input_history_encoding=stepper.input_history_encoding,
                    input_extra_features=stepper.input_extra_features,
                    history_rep_fix=stepper.history_rep_fix,
                )
                inputs.append(owned)
                expected.append(_ExpectedRoot(
                    slot_id, generation, owned.fen,
                    tuple(copy(move) for move in board.move_stack),
                    x.shape, owned._input_bytes(), key, source_key,
                ))
                row += 1
            pending.append(PreparedBatch._from_inputs(generation, tuple(inputs)) if inputs else None)
            expected_by_stepper.append(tuple(expected))

        for stepper, generation, batch, expected_roots, events in zip(
            steppers, generations, pending, expected_by_stepper, finalized_by_stepper,
        ):
            stepper._generation = generation
            for event in events:
                stepper._commit_finalized(event)
            stepper._pending = batch
            stepper._expected = expected_roots
        return tuple(zip(pending, finalized_by_stepper))

    def _validate_root(
        self, prepared: BT4PreparedInput, expected: _ExpectedRoot,
        output: BT4RootOutput,
    ) -> tuple[BT4LegalSnapshot, np.ndarray, BT4RootOutput, np.ndarray]:
        if (
            prepared.slot_id != expected.slot_id
            or prepared.generation != expected.generation
            or prepared.fen != expected.fen
            or not prepared._matches_board(self._games[prepared.slot_id].board)
            or prepared.x.dtype != np.dtype(np.float32)
            or prepared.x.shape != expected.x_shape
            or not prepared.x.flags.c_contiguous
            or prepared._input_bytes() != expected.x_bytes
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
        if sum(len(value.encode("utf-8")) for value in (
            output.fen, output.policy_output, output.wdl_output, output.wdl_kind,
            output.input_name, output.input_dtype, output.input_history_encoding,
            output.input_extra_features, output.model_sha256, output.onnx_feed_sha256,
        )) > 4096:
            raise ValueError("BT4 retained teacher metadata exceeds bounded row schema")
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
        validate_model_wdl(
            raw[None], 1, {"dtype": raw.dtype.name, "kind": output.wdl_kind}, output.model_sha256,
        )
        legal = prepared.legal
        weights = policy[legal.compact_indices].astype(np.float64, copy=True)
        if (
            bool(np.any(policy[~legal.legal_mask] != 0.0))
            or not np.isclose(float(weights.sum()), 1.0, rtol=0, atol=2e-6)
        ):
            raise ValueError("BT4 root policy is not normalized on legal moves")
        return legal, weights, _immutable_teacher(output), prepared.x

    @staticmethod
    def _stage_native_move(game: _Game, az_index: int) -> CBoard:
        staged = game.cboard.copy()
        staged.push_index(az_index)
        return staged

    @staticmethod
    def _commit_paired_move(game: _Game, move: chess.Move, staged: CBoard) -> None:
        # A failure after the Python push is fatal. Never rebuild/rollback and
        # continue after an uncertain half-commit.
        game.board.push(copy(move))
        game.cboard = staged

    def _validate_outputs(
        self, batch: PreparedBatch, outputs: Sequence[BT4RootOutput],
        *, temperatures: Mapping[int, float],
    ) -> tuple[list[tuple[int, float, BT4LegalSnapshot, np.ndarray, BT4RootOutput, np.ndarray]], tuple[str, ...] | None]:
        """Validate without consuming any game RNG or changing board state."""
        self._require_healthy()
        if self._pending is not batch or batch.generation != self._generation:
            raise RuntimeError("BT4 prepared batch is stale or already consumed")
        self._require_history_rep_fix()
        inputs = batch.dispatch_inputs()
        if len(outputs) != len(inputs) or len(outputs) != len(self._expected):
            raise ValueError("BT4 output count disagrees with prepared roots")
        slots = {root.slot_id for root in inputs}
        if set(temperatures) != slots:
            raise ValueError("BT4 needs one explicit temperature per prepared slot")
        validated: list[
            tuple[int, float, BT4LegalSnapshot, np.ndarray, BT4RootOutput, np.ndarray]
        ] = []
        output_contract = self._output_contract
        for prepared, expected, output in zip(inputs, self._expected, outputs):
            temperature = float(temperatures[prepared.slot_id])
            if not math.isfinite(temperature) or temperature < 0:
                raise ValueError("BT4 temperature must be finite and nonnegative")
            legal, weights, teacher, x = self._validate_root(prepared, expected, output)
            contract = (
                output.policy_output, output.wdl_output, output.wdl_kind,
                output.input_name, output.input_dtype, output.wdl_raw.dtype.name,
            )
            if output_contract is not None and contract != output_contract:
                raise ValueError("BT4 batch mixes output contracts")
            output_contract = contract
            validated.append((prepared.slot_id, temperature, legal, weights, teacher, x))

        return validated, output_contract

    def apply_root_outputs(
        self, batch: PreparedBatch, outputs: Sequence[BT4RootOutput],
        *, temperatures: Mapping[int, float],
    ) -> tuple[BT4SelectedMove, ...]:
        """Validate the whole batch before any RNG draw or board mutation."""
        return self._apply_validated(*self._validate_outputs(batch, outputs, temperatures=temperatures))

    @staticmethod
    def apply_many(
        requests: Sequence[tuple[BT4RootPolicyStepper, PreparedBatch, Sequence[BT4RootOutput], Mapping[int, float]]],
        *, admit: Callable[[], bool] = lambda: True,
    ) -> tuple[tuple[BT4SelectedMove, ...], ...] | None:
        """Validate every independent lane in a CPU microbatch before any draw.

        Scheduling units are not sampling units: each lane keeps its own RNG,
        output contract and exactly-once pending generation. A post-draw failure
        poisons that lane; already committed peers are never sampled again.
        """
        if len({id(stepper) for stepper, *_ in requests}) != len(requests):
            raise ValueError("BT4 apply_many needs distinct stepper identities")
        validated = [stepper._validate_outputs(batch, outputs, temperatures=temperatures)
                     for stepper, batch, outputs, temperatures in requests]
        if not admit():
            return None
        return tuple(stepper._apply_validated(*values)
                     for (stepper, *_), values in zip(requests, validated, strict=True))

    def _apply_validated(
        self, validated: list[tuple[int, float, BT4LegalSnapshot, np.ndarray, BT4RootOutput, np.ndarray]],
        output_contract: tuple[str, ...] | None,
    ) -> tuple[BT4SelectedMove, ...]:
        selections: list[BT4SelectedMove] = []
        selected_indices: list[int] = []
        # From the first attempted sample onward, any exception makes this
        # stepper unusable. Validation errors above remain safely retryable.
        try:
            for slot_id, temperature, legal, weights, _teacher, _x in validated:
                actions = np.arange(len(legal.az_indices), dtype=np.int32)
                index = sample_action_with_temperature(
                    self._games[slot_id].rng, actions, weights, temperature,
                    argmax_idx=int(np.argmax(weights)),
                )
                selected_indices.append(index)
                selections.append(BT4SelectedMove(slot_id, legal._move(index), temperature))

            # Stage every fallible native push before changing either live half.
            staged = [self._stage_native_move(self._games[slot_id], legal.az_indices[index])
                      for (slot_id, _, legal, _, _, _), index in zip(validated, selected_indices)]
            for selected, native, (_slot_id, _temperature, _legal, _weights, teacher, x) in zip(
                selections, staged, validated,
            ):
                game = self._games[selected.slot_id]
                record = BT4PlayedPly(
                    x=x, teacher=teacher, move=copy(selected.move),
                    pov_white=bool(game.board.turn), ply_index=game.board.ply(),
                    temperature=selected.temperature,
                )
                self._commit_paired_move(game, selected.move, native)
                game.records.append(record)
            self._pending = None
            self._expected = ()
            self._output_contract = output_contract
            return tuple(selections)
        except BaseException as exc:
            self._poisoned = exc
            raise
