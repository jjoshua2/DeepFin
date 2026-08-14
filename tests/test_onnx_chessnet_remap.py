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
from chess_anti_engine.encoding.ceres_tpg import encode_ceres_tpg
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE, move_to_index
from chess_anti_engine.moves.lc0_1858_movestrs import LC0_1858_MOVE_STRS
from chess_anti_engine.moves.leela_index import leela_index_for_move
from chess_anti_engine.onnx.load import (
    INPUT_FORMAT_CERES_TPG,
    INPUT_FORMAT_LC0_PLANES,
    WDL_OUTPUT_LOGITS,
    WDL_OUTPUT_PROBABILITIES,
    OnnxChessNet,
    declared_input_contract,
)


class _FakeSession:
    """Returns the identity policy so the output reveals the chosen index."""

    def __init__(self) -> None:
        self.seen: np.ndarray | None = None
        # Overridable so a test can hand the wrapper a row that the old
        # per-batch heuristic would have classified the other way.
        self.wdl: np.ndarray | None = None

    def run(self, _names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        planes = next(iter(feeds.values()))
        self.seen = planes
        batch = planes.shape[0]
        policy = np.tile(
            np.arange(COMPACT_POLICY_SIZE, dtype=np.float32), (batch, 1),
        )
        row = np.array([0.2, 0.5, 0.3], dtype=np.float32) if self.wdl is None else self.wdl[0]
        wdl = np.tile(row, (batch, 1))
        return [policy, wdl]


def _wrapper(
    input_format: str = INPUT_FORMAT_LC0_PLANES,
    *,
    input_dtype: type[np.floating] = np.float32,
) -> tuple[OnnxChessNet, _FakeSession]:
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
    inject._input_dtype = input_dtype
    net.input_format = input_format
    net.wdl_output_kind = WDL_OUTPUT_PROBABILITIES
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


def _forward_ceres(board: chess.Board) -> np.ndarray:
    net, _session = _wrapper(INPUT_FORMAT_CERES_TPG, input_dtype=np.float16)
    squares = encode_ceres_tpg(board)
    out = net.forward(torch.from_numpy(squares)[None, ...])
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


# ---------------------------------------------------------------------------
# The Ceres TPG input format. Same wrapper, same remap, different input: the
# board context now has to be recovered from the 64x137 square records rather
# than from LC0 planes. If that recovery is wrong the policy is a permutation
# of itself, which is exactly the failure that does not crash.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen",
    [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
        "7k/R7/8/8/8/8/8/7K w - - 0 1",
        "7k/P7/8/8/8/8/8/7K w - - 0 1",
        "6k1/3Q4/8/8/8/8/8/6K1 w - - 0 1",
        "k7/8/7K/8/8/8/pppppppp/8 b - - 0 1",
        "r1bqkb1r/pp2pppp/2n2n2/2pp4/3P4/2N1PN2/PPP2PPP/R1BQKB1R b KQkq - 0 5",
    ],
)
def test_ceres_every_legal_move_reads_its_own_leela_logit(fen: str) -> None:
    board = chess.Board(fen)
    policy = _forward_ceres(board)
    for move in board.legal_moves:
        want = leela_index_for_move(board, move)
        got = round(float(policy[move_to_index(move, board)]))
        assert got == want, (
            f"{fen} {move.uci()}: wrapper read Leela slot "
            f"{LC0_1858_MOVE_STRS[got]}, should be {LC0_1858_MOVE_STRS[want]}"
        )


def test_ceres_castling_is_not_read_as_the_plain_slide() -> None:
    """Ceres uses Leela's 1858 table verbatim, so it inherits the king-takes-rook
    castling spelling and the 49x-120x under-read a static map produced."""
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    policy = _forward_ceres(board)
    for uci, leela in (("e1g1", "e1h1"), ("e1c1", "e1a1")):
        slot = move_to_index(chess.Move.from_uci(uci), board)
        assert LC0_1858_MOVE_STRS[round(float(policy[slot]))] == leela


def test_ceres_back_rank_slot_follows_the_piece_on_it() -> None:
    """The 22 shared back-rank slots mean =Q for a pawn and a plain slide for
    anything else; the pawn mask that decides comes out of the square records."""
    rook = chess.Board("7k/R7/8/8/8/8/8/7K w - - 0 1")
    pawn = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    slot = move_to_index(chess.Move.from_uci("a7a8"), rook)
    assert LC0_1858_MOVE_STRS[round(float(_forward_ceres(rook)[slot]))] == "a7a8"
    promo = chess.Move.from_uci("a7a8q")
    pawn_slot = move_to_index(promo, pawn)
    assert LC0_1858_MOVE_STRS[round(float(_forward_ceres(pawn)[pawn_slot]))] == "a7a8q"


def test_ceres_input_reaches_the_session_unchanged_in_the_declared_dtype() -> None:
    """The C1 graphs declare tensor(float16); feeding float32 is an ORT hard
    error, and feeding a silently-wrong tensor is the failure this file exists
    for. Assert on what the session actually received."""
    net, session = _wrapper(INPUT_FORMAT_CERES_TPG, input_dtype=np.float16)
    squares = encode_ceres_tpg(chess.Board())
    net.forward(torch.from_numpy(squares)[None, ...])
    assert session.seen is not None
    assert session.seen.dtype == np.float16
    assert session.seen.shape == (1, 64, 137)
    np.testing.assert_allclose(session.seen[0], squares, rtol=0, atol=1e-3)


def test_ceres_format_rejects_plane_shaped_input() -> None:
    net, _session = _wrapper(INPUT_FORMAT_CERES_TPG, input_dtype=np.float16)
    planes = encode_position(
        chess.Board(), add_features=False, input_history_encoding="lc0_root",
    )
    with pytest.raises(ValueError, match="square records"):
        net.forward(torch.from_numpy(planes)[None, ...])


def test_lc0_format_rejects_square_shaped_input() -> None:
    net, _session = _wrapper()
    squares = encode_ceres_tpg(chess.Board())
    with pytest.raises(ValueError, match=r"\(B, planes, 8, 8\)"):
        net.forward(torch.from_numpy(squares)[None, ...])


def test_unknown_input_format_is_rejected_before_the_session_opens() -> None:
    with pytest.raises(ValueError, match="input_format"):
        OnnxChessNet(
            "/nonexistent.onnx",
            input_name="squares",
            policy_output_name="policy",
            wdl_output_name="value",
            input_format="tpg",
        )


def test_wrapper_matches_the_direct_1858_gather() -> None:
    """The audit script gathers in 1858 space itself and never touches this
    wrapper, so a number from there vouches for the encoder and the table, not
    for the class we ship. Pin the two together.

    `_leela_idxs` reaches the slot by mirroring the UCI STRING for a
    black-to-move board; the wrapper reaches it through the board-aware compact
    map. Two independent routes to the same index — agreement is a real
    cross-check, not a restatement.
    """
    import scripts.foreign_net_audit as fna

    for fen in (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "7k/P7/8/8/8/8/8/7K w - - 0 1",
        "k7/8/7K/8/8/8/pppppppp/8 b - - 0 1",
    ):
        board = chess.Board(fen)
        legal = [m.uci() for m in board.legal_moves]
        # The script has no public API for this; it is a script.
        direct = fna._leela_idxs(board, legal)
        through_wrapper = _forward_ceres(board)
        for uci, want in zip(legal, direct, strict=True):
            move = chess.Move.from_uci(uci)
            got = round(float(through_wrapper[move_to_index(move, board)]))
            assert got == want, (
                f"{fen} {uci}: wrapper read Leela slot {LC0_1858_MOVE_STRS[got]}, "
                f"the audit script's direct gather reads {LC0_1858_MOVE_STRS[want]}"
            )


def test_wdl_kind_is_frozen_not_re_decided_per_batch() -> None:
    """A declared `probabilities` head must stay `probabilities` even when a
    batch's values happen to look like logits, and vice versa. The old code
    re-ran the heuristic every call, so batch content could flip the meaning of
    the value head silently."""
    net, session = _wrapper(INPUT_FORMAT_CERES_TPG, input_dtype=np.float16)
    squares = torch.from_numpy(encode_ceres_tpg(chess.Board()))[None, ...]

    net.wdl_output_kind = WDL_OUTPUT_PROBABILITIES
    session.wdl = np.array([[0.2, 0.5, 0.3]], dtype=np.float32)
    np.testing.assert_allclose(
        net.forward(squares)["wdl"].numpy(), np.log([[0.2, 0.5, 0.3]]), rtol=1e-5,
    )
    # Same object, a row that a per-batch heuristic would call logits.
    session.wdl = np.array([[-1.0, 2.0, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(
        net.forward(squares)["wdl"].numpy(),
        np.log(np.clip([[-1.0, 2.0, 0.5]], 1e-9, None)), rtol=1e-5,
    )

    net.wdl_output_kind = WDL_OUTPUT_LOGITS
    session.wdl = np.array([[0.2, 0.5, 0.3]], dtype=np.float32)
    np.testing.assert_allclose(
        net.forward(squares)["wdl"].numpy(), [[0.2, 0.5, 0.3]], rtol=1e-6,
    )


def test_unknown_wdl_output_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="wdl_output_kind"):
        OnnxChessNet(
            "/nonexistent.onnx",
            input_name="squares",
            policy_output_name="policy",
            wdl_output_name="value",
            wdl_output_kind="softmaxed",
        )


class _KindProbeSession:
    """Returns a fixed WDL row so the probe's classification is checkable."""

    def __init__(self, row: list[float]) -> None:
        self.row = np.array([row], dtype=np.float32)
        self.calls = 0
        self.fed: np.ndarray = np.zeros((0,), dtype=np.float32)

    def run(self, _names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls += 1
        self.fed = next(iter(feeds.values()))
        return [self.row]


@pytest.mark.parametrize(
    ("input_format", "row", "expected"),
    [
        # BT4: a softmaxed head.
        (INPUT_FORMAT_LC0_PLANES, [0.2, 0.5, 0.3], WDL_OUTPUT_PROBABILITIES),
        # Ceres `value`: logits, with negatives and a row sum far from 1.
        (INPUT_FORMAT_CERES_TPG, [-0.511, 0.757, -0.809], WDL_OUTPUT_LOGITS),
        # Non-negative but nowhere near summing to 1 — still logits.
        (INPUT_FORMAT_CERES_TPG, [3.0, 1.0, 2.0], WDL_OUTPUT_LOGITS),
        # An fp16/quantised probability row that sums to 0.98 must still count.
        (INPUT_FORMAT_LC0_PLANES, [0.19, 0.49, 0.30], WDL_OUTPUT_PROBABILITIES),
    ],
)
def test_wdl_probe_classifies_and_feeds_the_right_input_shape(
    input_format: str, row: list[float], expected: str,
) -> None:
    """The probe runs ONE forward on a start position in the net's own format.
    If it fed the wrong shape the real session would reject it, so assert on the
    shape the session actually received as well as on the verdict."""
    net = object.__new__(OnnxChessNet)
    torch.nn.Module.__init__(net)
    session = _KindProbeSession(row)
    inject: Any = net
    inject._session = session
    inject._input_name = "in"
    inject._wdl_out = "wdl"
    inject._plane_count = 112
    inject._input_dtype = np.float32
    net.input_format = input_format
    net.input_history_encoding = (
        "lc0_root" if input_format == INPUT_FORMAT_LC0_PLANES else input_format
    )

    assert net._probe_wdl_kind() == expected
    assert session.calls == 1
    want = (1, 64, 137) if input_format == INPUT_FORMAT_CERES_TPG else (1, 112, 8, 8)
    assert session.fed.shape == want


def test_ceres_format_declares_an_encoding_no_plane_encoder_accepts() -> None:
    """A helper that tries to encode positions FOR a Ceres net must fail loudly.
    Declaring `lc0_root` would let it build 112 LC0 planes for a net that reads
    64x137 records — accepted, silently meaningless.

    Reads what the constructor WOULD set (`declared_input_contract`), not a value
    the test itself assigned to a hand-built object: a double that is handed the
    answer cannot test the answer.
    """
    from chess_anti_engine.encoding.lc0 import normalize_lc0_history_encoding

    hist, extra = declared_input_contract(INPUT_FORMAT_CERES_TPG)
    assert (hist, extra) == (INPUT_FORMAT_CERES_TPG, INPUT_FORMAT_CERES_TPG)
    with pytest.raises(ValueError, match="history"):
        normalize_lc0_history_encoding(hist)
    # ...while the lc0 path still declares something the encoders accept.
    lc0_hist, lc0_extra = declared_input_contract(INPUT_FORMAT_LC0_PLANES)
    assert normalize_lc0_history_encoding(lc0_hist) == "lc0_root"
    assert lc0_extra == "v1"
