"""``OnnxChessNet`` must read a Leela policy through the board-aware remap.

The remap itself is covered by ``test_leela_index``; what this file pins is
that the ONNX wrapper — the only production-reachable consumer — actually goes
through it. The failure this guards against is the one the module used to have:
a static 4672->1858 table that returned a plausible logit for the wrong move,
silently, for every castle and every back-rank slide.

The session is faked so the test needs no 700MB net: the "policy" is
``arange(1858)`` in LEELA order, so the value that lands in a 4672 slot IS the
Leela index the wrapper chose for it, and a mis-map is directly readable.
"""
from __future__ import annotations

from typing import Any

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE, move_to_index
from chess_anti_engine.moves.lc0_1858_movestrs import LC0_1858_MOVE_STRS
from chess_anti_engine.moves.leela_index import leela_index_for_move
from chess_anti_engine.onnx.load import OnnxChessNet


class _FakeSession:
    """Returns the identity policy so the output reveals the chosen index."""

    def __init__(self) -> None:
        self.seen: np.ndarray | None = None

    def run(self, _names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        planes = next(iter(feeds.values()))
        self.seen = planes
        batch = planes.shape[0]
        policy = np.tile(
            np.arange(COMPACT_POLICY_SIZE, dtype=np.float32), (batch, 1),
        )
        wdl = np.tile(np.array([0.2, 0.5, 0.3], dtype=np.float32), (batch, 1))
        return [policy, wdl]


def _wrapper() -> tuple[OnnxChessNet, _FakeSession]:
    net = object.__new__(OnnxChessNet)
    torch.nn.Module.__init__(net)
    session = _FakeSession()
    # Injected through an Any handle: the wrapper takes its session in __init__
    # from a file path, and widening that signature purely for a test would put
    # test scaffolding on the production constructor.
    inject: Any = net
    inject._session = session
    inject._input_name = "planes"
    inject._policy_out = "policy"
    inject._wdl_out = "wdl"
    inject._plane_count = 112
    net.input_history_encoding = "lc0_root"
    net.input_extra_features = "v1"
    net.use_dynamic_relations = False
    net.policy_encoding = "az_4672"
    return net, session


def _forward(board: chess.Board) -> np.ndarray:
    net, _session = _wrapper()
    planes = encode_position(
        board, add_features=False, input_history_encoding="lc0_root",
    )
    out = net.forward(torch.from_numpy(planes)[None, ...])
    return out["policy_own"][0].numpy()


@pytest.mark.parametrize(
    "fen",
    [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
        "7k/R7/8/8/8/8/8/7K w - - 0 1",
        "7k/P7/8/8/8/8/8/7K w - - 0 1",
        "6k1/3Q4/8/8/8/8/8/6K1 w - - 0 1",
        "r1bqkb1r/pp2pppp/2n2n2/2pp4/3P4/2N1PN2/PPP2PPP/R1BQKB1R b KQkq - 0 5",
    ],
)
def test_every_legal_move_reads_its_own_leela_logit(fen: str) -> None:
    board = chess.Board(fen)
    policy = _forward(board)
    for move in board.legal_moves:
        want = leela_index_for_move(board, move)
        got = round(float(policy[move_to_index(move, board)]))
        assert got == want, (
            f"{fen} {move.uci()}: wrapper read Leela slot "
            f"{LC0_1858_MOVE_STRS[got]}, should be {LC0_1858_MOVE_STRS[want]}"
        )


def test_castling_is_not_read_as_the_plain_slide() -> None:
    """The single most damaging case, and the one a static table cannot see."""
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    policy = _forward(board)
    for uci, leela in (("e1g1", "e1h1"), ("e1c1", "e1a1")):
        slot = move_to_index(chess.Move.from_uci(uci), board)
        got = round(float(policy[slot]))
        assert LC0_1858_MOVE_STRS[got] == leela
        assert LC0_1858_MOVE_STRS[got] != uci


def test_geometrically_impossible_slots_are_masked_not_garbage() -> None:
    """4672 has ~2814 slots that name no move at all (a1 sliding south, etc.).
    They must come back masked, or the legal filter downstream has nothing to
    filter against."""
    from chess_anti_engine.moves.encode import FULL_TO_COMPACT_POLICY

    board = chess.Board()
    policy = _forward(board)
    impossible = np.flatnonzero(np.asarray(FULL_TO_COMPACT_POLICY) < 0)
    assert impossible.size > 0
    assert float(np.max(policy[impossible])) < -1e8
    legal = sorted({int(move_to_index(m, board)) for m in board.legal_moves})
    assert float(np.min(policy[legal])) > -1e8


def test_history_fill_reaches_the_session() -> None:
    """The wrapper repeat-fills empty history before the net sees it; if that
    stopped happening the fake session would receive zeroed frames."""
    net, session = _wrapper()
    planes = encode_position(
        chess.Board(), add_features=False, input_history_encoding="lc0_root",
    )
    assert not planes[13:26].any(), "startpos should start with an empty second frame"
    net.forward(torch.from_numpy(planes)[None, ...])
    assert session.seen is not None
    np.testing.assert_array_equal(session.seen[0, 13:25], session.seen[0, 0:12])
