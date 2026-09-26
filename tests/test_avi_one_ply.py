from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts import one_ply


class FakeEvaluator:
    def __init__(self, logits: np.ndarray | None = None) -> None:
        self.logits = None if logits is None else np.asarray(logits, dtype=np.float32)
        self.offset = 0
        self.rows = 0

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        n = int(x.shape[0])
        self.rows += n
        if self.logits is None:
            wdl = np.zeros((n, 3), dtype=np.float32)
        else:
            wdl = self.logits[self.offset : self.offset + n]
            self.offset += n
            assert len(wdl) == n
        return np.zeros((n, 4672), dtype=np.float32), wdl


def fake_encoder(boards: list[chess.Board], **_kwargs: object) -> np.ndarray:
    return np.zeros((len(boards), 1, 8, 8), dtype=np.float32)


def test_one_ply_flips_child_perspective_and_covers_every_legal_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(one_ply, "encode_positions_batch", fake_encoder)
    board = chess.Board()
    legal = list(board.legal_moves)
    logits = np.zeros((len(legal), 3), dtype=np.float32)
    logits[0] = np.asarray([8.0, 0.0, -8.0])
    logits[1] = np.asarray([-8.0, 0.0, 8.0])
    evaluator = FakeEvaluator(logits)

    backup = one_ply.one_ply_value_backups(
        [board], evaluator, batch_size=3, claim_draws=True
    )[0]

    assert backup.move == legal[1]
    assert backup.q > 0.99
    assert backup.legal_moves == len(legal)
    assert backup.evaluated_children == len(legal)
    assert backup.terminal_children == 0
    assert evaluator.rows == len(legal)


def test_exact_mate_child_beats_network_children(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(one_ply, "encode_positions_batch", fake_encoder)
    board = chess.Board("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1")
    evaluator = FakeEvaluator()

    backup = one_ply.one_ply_value_backups([board], evaluator, batch_size=2)[0]
    child = board.copy(stack=True)
    child.push(backup.move)

    assert backup.selected_terminal
    assert backup.q == pytest.approx(1.0)
    assert child.is_checkmate()
    assert evaluator.rows == backup.evaluated_children
    assert backup.terminal_children >= 1


def test_twofold_repetition_is_not_promoted_to_exact_draw() -> None:
    board = chess.Board()
    parent_turn = chess.BLACK
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    assert board.is_repetition(2)
    assert not board.can_claim_threefold_repetition()
    assert one_ply.terminal_parent_wdl(
        board, parent_turn=parent_turn, claim_draws=True
    ) is None


def test_successor_encoding_preserves_full_move_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_stack_lengths: list[int] = []

    def capture_encoder(boards: list[chess.Board], **_kwargs: object) -> np.ndarray:
        seen_stack_lengths.extend(len(board.move_stack) for board in boards)
        return np.zeros((len(boards), 1, 8, 8), dtype=np.float32)

    monkeypatch.setattr(one_ply, "encode_positions_batch", capture_encoder)
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        board.push_uci(uci)
    parent_history = len(board.move_stack)
    backup = one_ply.one_ply_value_backups(
        [board], FakeEvaluator(), batch_size=4
    )[0]

    assert backup.evaluated_children == len(seen_stack_lengths)
    assert seen_stack_lengths
    assert set(seen_stack_lengths) == {parent_history + 1}


def test_blend_wdl_targets_normalizes_and_keeps_distribution_shape() -> None:
    anchor = np.asarray([[6.0, 2.0, 2.0]], dtype=np.float32)
    neural = np.asarray([[2.0, 2.0, 6.0]], dtype=np.float32)
    mixed = one_ply.blend_wdl_targets(anchor, neural, alpha=0.25)
    np.testing.assert_allclose(mixed, [[0.5, 0.2, 0.3]], atol=1e-7)
    np.testing.assert_allclose(mixed.sum(axis=1), 1.0, atol=1e-7)


@pytest.mark.parametrize("alpha", [-0.01, 1.01, float("nan")])
def test_blend_rejects_invalid_alpha(alpha: float) -> None:
    wdl = np.asarray([[0.5, 0.25, 0.25]], dtype=np.float32)
    with pytest.raises(ValueError, match="alpha"):
        one_ply.blend_wdl_targets(wdl, wdl, alpha=alpha)
