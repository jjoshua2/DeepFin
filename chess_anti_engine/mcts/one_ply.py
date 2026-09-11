from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import chess
import numpy as np

from chess_anti_engine.encoding import encode_positions_batch
from chess_anti_engine.encoding.features import relation_matrices
from chess_anti_engine.inference import BatchEvaluator


@dataclass(frozen=True)
class OnePlyBackup:
    """Best successor value returned in the parent position's W/D/L perspective."""

    move: chess.Move
    wdl: np.ndarray
    q: float
    legal_moves: int
    terminal_children: int
    evaluated_children: int
    selected_terminal: bool


@dataclass
class _Candidate:
    move: chess.Move
    wdl: np.ndarray | None
    terminal: bool


def wdl_probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    """Stable softmax for a batch of W/D/L logits."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"expected WDL logits shaped [N,3], got {values.shape!r}")
    if not bool(np.isfinite(values).all()):
        raise ValueError("non-finite WDL logits")
    shifted = values - values.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    mass = weights.sum(axis=1, keepdims=True)
    if not bool(np.isfinite(mass).all() and (mass > 0).all()):
        raise ValueError("invalid WDL softmax mass")
    return np.asarray(weights / mass, dtype=np.float32)


def _relation_batch(boards: Sequence[chess.Board]) -> np.ndarray:
    return np.stack([np.asarray(relation_matrices(board)) for board in boards], axis=0)


def evaluate_wdl_probabilities(
    boards: Sequence[chess.Board],
    evaluator: BatchEvaluator,
    *,
    batch_size: int = 4096,
    input_history_encoding: str | None = None,
    input_extra_features: str | None = None,
    compute_relations: bool = False,
) -> np.ndarray:
    """Evaluate boards with one frozen network and return W/D/L probabilities."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    board_list = list(boards)
    if not board_list:
        return np.zeros((0, 3), dtype=np.float32)
    outputs: list[np.ndarray] = []
    for start in range(0, len(board_list), batch_size):
        chunk = board_list[start : start + batch_size]
        encoded = encode_positions_batch(
            chunk,
            input_history_encoding=input_history_encoding,
            input_extra_features=input_extra_features,
        )
        relations = _relation_batch(chunk) if compute_relations else None
        _policy, logits = evaluator.evaluate_encoded(encoded, relations=relations)
        logits = np.asarray(logits)
        if logits.shape != (len(chunk), 3):
            raise ValueError(
                f"evaluator returned WDL shape {logits.shape!r} for {len(chunk)} boards"
            )
        outputs.append(wdl_probabilities_from_logits(logits))
    return np.concatenate(outputs, axis=0)


def terminal_parent_wdl(
    child: chess.Board,
    *,
    parent_turn: chess.Color,
    claim_draws: bool = True,
) -> np.ndarray | None:
    """Exact terminal WDL after a move, from the mover/parent perspective.

    ``claim_draws`` follows python-chess claimable threefold/50-move semantics.
    A mere twofold repetition is not terminal here. This is intentionally distinct
    from DeepFin search's LC0-style twofold-as-draw pruning convention.
    """
    outcome = child.outcome(claim_draw=claim_draws)
    if outcome is None:
        return None
    if outcome.winner is None:
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    if outcome.winner == parent_turn:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


def one_ply_value_backups(
    boards: Sequence[chess.Board],
    evaluator: BatchEvaluator,
    *,
    batch_size: int = 4096,
    input_history_encoding: str | None = None,
    input_extra_features: str | None = None,
    compute_relations: bool = False,
    claim_draws: bool = True,
) -> list[OnePlyBackup]:
    """Evaluate every legal successor and back up the best child value.

    Non-terminal child WDL is predicted from the child's side-to-move perspective,
    then flipped to the parent perspective before maximizing q = P(win)-P(loss).
    Terminal children use their exact result and never consume network inference.
    Ties keep python-chess legal-move iteration order, making the choice deterministic.
    """
    board_list = list(boards)
    candidates: list[list[_Candidate]] = []
    pending_boards: list[chess.Board] = []
    pending_refs: list[tuple[int, int]] = []

    for parent_index, board in enumerate(board_list):
        if board.outcome(claim_draw=claim_draws) is not None:
            raise ValueError(f"parent board {parent_index} is already terminal")
        parent_turn = board.turn
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError(f"parent board {parent_index} has no legal moves")
        row: list[_Candidate] = []
        for move in legal_moves:
            child = board.copy(stack=True)
            child.push(move)
            exact = terminal_parent_wdl(
                child, parent_turn=parent_turn, claim_draws=claim_draws
            )
            row.append(_Candidate(move=move, wdl=exact, terminal=exact is not None))
            if exact is None:
                pending_refs.append((parent_index, len(row) - 1))
                pending_boards.append(child)
        candidates.append(row)

    child_wdl = evaluate_wdl_probabilities(
        pending_boards,
        evaluator,
        batch_size=batch_size,
        input_history_encoding=input_history_encoding,
        input_extra_features=input_extra_features,
        compute_relations=compute_relations,
    )
    parent_wdl = child_wdl[:, [2, 1, 0]]
    for (parent_index, child_index), wdl in zip(pending_refs, parent_wdl, strict=True):
        candidates[parent_index][child_index].wdl = np.asarray(wdl, dtype=np.float32)

    result: list[OnePlyBackup] = []
    for row in candidates:
        if any(candidate.wdl is None for candidate in row):
            raise RuntimeError("one-ply backup left an unevaluated legal child")

        def candidate_q(candidate: _Candidate) -> float:
            if candidate.wdl is None:
                raise RuntimeError("one-ply backup left an unevaluated legal child")
            return float(candidate.wdl[0] - candidate.wdl[2])

        best_index = max(range(len(row)), key=lambda i: candidate_q(row[i]))
        best = row[best_index]
        assert best.wdl is not None
        wdl = np.asarray(best.wdl, dtype=np.float32).copy()
        result.append(
            OnePlyBackup(
                move=best.move,
                wdl=wdl,
                q=float(wdl[0] - wdl[2]),
                legal_moves=len(row),
                terminal_children=sum(candidate.terminal for candidate in row),
                evaluated_children=sum(not candidate.terminal for candidate in row),
                selected_terminal=best.terminal,
            )
        )
    return result


def _normalized_wdl(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != 3:
        raise ValueError(f"expected final WDL dimension of 3, got {array.shape!r}")
    if not bool(np.isfinite(array).all() and (array >= 0).all()):
        raise ValueError("WDL probabilities must be finite and non-negative")
    mass = array.sum(axis=-1, keepdims=True)
    if not bool((mass > 0).all()):
        raise ValueError("WDL probability mass must be positive")
    return array / mass


def blend_wdl_targets(anchor: np.ndarray, neural: np.ndarray, *, alpha: float) -> np.ndarray:
    """Blend an anchored target with root-distillation or one-ply neural WDL."""
    if not np.isfinite(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    anchor_n = _normalized_wdl(anchor)
    neural_n = _normalized_wdl(neural)
    if anchor_n.shape != neural_n.shape:
        raise ValueError(
            f"anchor/neural WDL shapes differ: {anchor_n.shape!r} vs {neural_n.shape!r}"
        )
    mixed = (1.0 - float(alpha)) * anchor_n + float(alpha) * neural_n
    return np.asarray(mixed, dtype=np.float32)
