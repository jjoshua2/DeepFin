"""Dense evaluator policy selection must use full search action ids."""
import chess
import numpy as np

from scripts.blindspot_netside_vet import _full_policy_top_move


def test_full_logits_do_not_use_compact_indices():
    board = chess.Board()
    policy = np.zeros((1, 4672))
    policy[0, 877] = 100  # e2e4 full action id
    policy[0, 168] = 90  # g1h3 compact id, NOT its full action id (494)
    assert _full_policy_top_move(board, list(board.legal_moves), policy) == chess.Move.from_uci("e2e4")


def test_agreeing_control_and_legal_order_ties():
    board = chess.Board()
    legal = list(board.legal_moves)
    policy = np.zeros((1, 4672))
    assert _full_policy_top_move(board, legal, policy) == legal[0]
    policy[0, 494] = 10
    policy[0, 168] = 10
    assert _full_policy_top_move(board, legal, policy) == chess.Move.from_uci("g1h3")
    policy[0, 877] = 10
    assert _full_policy_top_move(board, legal, policy) == legal[0]


def test_orientation_castling_and_promotions():
    cases = [
        (chess.STARTING_FEN, "g1h3", 494),
        (chess.STARTING_FEN.replace(" w ", " b "), "g8h6", 494),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", 307),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", 335),
    ]
    for fen, prefix in [
        ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8"),
        ("7k/8/8/8/8/8/p7/7K b - - 0 1", "a2a1"),
    ]:
        for piece, index in [("q", 3504), ("n", 3569), ("b", 3572), ("r", 3575)]:
            cases.append((fen, prefix + piece, index))
    for fen, uci, index in cases:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        legal = list(board.legal_moves)
        assert move in legal
        policy = np.zeros((1, 4672))
        policy[0, index] = 20
        assert _full_policy_top_move(board, legal, policy) == move


def test_no_legal_moves():
    for fen in ["7k/6Q1/5K2/8/8/8/8/8 b - - 0 1", "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"]:
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        assert not legal
        assert _full_policy_top_move(board, legal, np.zeros((1, 4672))) is None


def test_reject_malformed_evaluator_shape():
    board = chess.Board()
    legal = list(board.legal_moves)
    for shape in [(1, 1858), (1, 4671), (4672,), (2, 4672), (1, 1, 4672)]:
        message = ""
        try:
            _full_policy_top_move(board, legal, np.zeros(shape))
        except ValueError as exc:
            message = str(exc)
        assert "expected evaluator policy shape (1, 4672)" in message
