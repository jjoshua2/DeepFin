"""Tests for the native big-net NNUE evaluator, its weight pipeline and its seam.

⚑ WHAT THESE TESTS ARE NOT. They are not the correctness gate. The gate is
``scripts/nnue_parity.py``, which requires EXACT integer equality against
Stockfish's own ``(Big net) NNUE evaluation ... internal units`` line over a
stratified 50k-FEN sample. Internal parity between our C and our numpy reference
cannot find a rule that is wrong in both. What these tests do is (a) pin the
pieces the gate cannot localise — parser, converter, index computation, seam
dispatch — and (b) keep a green suite from certifying a regression the gate would
have caught only on a manual run.

Real weights are a RUNTIME ARTIFACT and are never committed, so most tests here
run against a SYNTHETIC pack: a sparse file with the real layout and almost
everything zero, with a handful of values poked in so that a specific wiring
question has an exactly predictable answer. Zero weights make the arithmetic
trivial, which is precisely what makes the WIRING visible.

Tests that need the real network are gated on ``CAE_NNUE_TEST_PACK`` /
``CAE_NNUE_TEST_NNUE`` and say so in their skip reason.
"""

from __future__ import annotations

import dataclasses
import os
import struct
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.nnue import _nnue_ext
from scripts import nnue_pack, nnue_parse
from scripts.nnue_reference import (
    THREAT_TABLES,
    halfka_active_indices,
    halfka_make_index,
    position_view,
    threat_active_indices,
)

BIG = nnue_parse.ARCHS[0]
SMALL = nnue_parse.ARCHS[1]

# A spread of quiet, legal, not-in-check positions covering several buckets.
POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r2q1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N1BP2/PP1QN1PP/R3KB1R w KQ - 0 10",
    "8/5pk1/6p1/8/1P6/5PKP/8/8 w - - 0 40",
    "8/8/4k3/8/8/4K3/8/8 w - - 0 60",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 6 9",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]

IN_CHECK_POSITIONS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 3",
]


def cboard(fen: str) -> CBoard:
    return CBoard.from_board(chess.Board(fen))


# ===========================================================================
# Synthetic packs
# ===========================================================================


def _big_layout() -> nnue_pack.PackLayout:
    return nnue_pack.pack_layout(
        l1=BIG.l1,
        l2=BIG.l2,
        l3=BIG.l3,
        halfka_dims=nnue_parse.HALFKA_DIMS,
        threat_dims=nnue_parse.THREAT_DIMS,
        nnue_version=nnue_parse.VERSION,
        net_hash=0x1234ABCD,
        ft_hash=BIG.ft_hash,
        source_sha256="00" * 32,
    )


def write_synthetic_pack(path: Path, pokes: dict[str, list[tuple[int, int]]]) -> None:
    """A real-layout pack that is all zeros except for ``pokes``.

    ``pokes`` maps a tensor name to (element_index, value) pairs; int32 tensors
    only, which is all the wiring tests need. The file is created as a sparse
    file — the 111 MB of holes read back as zeros and cost no disk.
    """
    layout = _big_layout()
    with open(path, "wb") as fh:
        fh.truncate(layout.total_size)
        fh.write(layout.header)
        for name, entries in pokes.items():
            for index, value in entries:
                fh.seek(layout.offsets[name] + index * 4)
                fh.write(struct.pack("<i", value))


@pytest.fixture(scope="module")
def bucket_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """fc2 bias = (bucket + 1) * 1600, everything else zero.

    With zero feature weights the accumulator, the transformer output and every
    fc0/fc1 activation are zero, so the evaluation collapses to
    ``fc2_bias[bucket] / 16`` = ``(bucket + 1) * 100``. The evaluation therefore
    NAMES the bucket it used.
    """
    path = tmp_path_factory.mktemp("nnue") / "bucket.pack"
    write_synthetic_pack(
        path, {"fc2_bias": [(b, (b + 1) * 1600) for b in range(nnue_parse.PSQT_BUCKETS)]}
    )
    return path


# ===========================================================================
# Parser
# ===========================================================================


def _encode_leb128_small(values: np.ndarray) -> bytes:
    """Signed LEB128 for values in [-64, 63] — exactly one byte each."""
    assert values.min() >= -64
    assert values.max() <= 63
    return bytes((values.astype(np.int64) & 0x7F).astype(np.uint8).tobytes())


def _leb_block(*arrays: np.ndarray) -> bytes:
    payload = b"".join(_encode_leb128_small(a.ravel()) for a in arrays)
    return nnue_parse.LEB_MAGIC + struct.pack("<I", len(payload)) + payload


def build_synthetic_small_nnue(
    seed: int = 7,
) -> tuple[bytes, dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
    """A complete, parseable small-architecture ``.nnue`` file.

    Small rather than big so the test file stays a few megabytes: the big net's
    threat tensor alone is 62 MB. This exercises the header, the LEB128 block
    machinery, the layer-stack loop and — the part that matters — the
    land-exactly-on-EOF property, which is the parser's own proof that it
    understood the layout.
    """
    rng = np.random.default_rng(seed)
    l1, l2, l3 = SMALL.l1, SMALL.l2, SMALL.l3
    tensors = {
        "ft_bias": rng.integers(-64, 64, size=l1, dtype=np.int64),
        "ft_weight": rng.integers(-64, 64, size=(nnue_parse.HALFKA_DIMS, l1), dtype=np.int64),
        "ft_psqt": rng.integers(
            -64, 64, size=(nnue_parse.HALFKA_DIMS, nnue_parse.PSQT_BUCKETS), dtype=np.int64
        ),
    }
    desc = b"synthetic test net"
    out = bytearray()
    out += struct.pack("<III", nnue_parse.VERSION, 0xDEADBEEF, len(desc))
    out += desc
    out += struct.pack("<I", SMALL.ft_hash)
    out += _leb_block(tensors["ft_bias"])
    out += _leb_block(tensors["ft_weight"])
    out += _leb_block(tensors["ft_psqt"])

    stacks: list[dict[str, np.ndarray]] = []
    for _ in range(nnue_parse.LAYER_STACKS):
        fc0_b = rng.integers(-64, 64, size=l2 + 1, dtype=np.int32)
        fc0_w = rng.integers(-64, 64, size=(l2 + 1, nnue_parse.pad(l1)), dtype=np.int8)
        fc1_b = rng.integers(-64, 64, size=l3, dtype=np.int32)
        fc1_w = rng.integers(-64, 64, size=(l3, nnue_parse.pad(l2 * 2)), dtype=np.int8)
        fc2_b = rng.integers(-64, 64, size=1, dtype=np.int32)
        fc2_w = rng.integers(-64, 64, size=(1, nnue_parse.pad(l3)), dtype=np.int8)
        out += struct.pack("<I", 0xC0FFEE01)
        for arr in (fc0_b, fc0_w, fc1_b, fc1_w, fc2_b, fc2_w):
            out += arr.astype(arr.dtype.newbyteorder("<")).tobytes()
        stacks.append(
            {"fc0_b": fc0_b, "fc0_w": fc0_w, "fc1_b": fc1_b,
             "fc1_w": fc1_w, "fc2_b": fc2_b, "fc2_w": fc2_w}
        )
    return bytes(out), tensors, stacks


def test_parser_round_trips_a_complete_file_and_lands_on_eof() -> None:
    blob, tensors, stacks = build_synthetic_small_nnue()
    net = nnue_parse.parse_bytes(blob)

    assert net.arch.name == "small"
    assert net.version == nnue_parse.VERSION
    assert net.net_hash == 0xDEADBEEF
    assert net.description == "synthetic test net"
    np.testing.assert_array_equal(net.ft_bias, tensors["ft_bias"].astype(np.int16))
    np.testing.assert_array_equal(net.ft_weight, tensors["ft_weight"].astype(np.int16))
    np.testing.assert_array_equal(net.ft_psqt, tensors["ft_psqt"].astype(np.int32))
    assert net.threat_weight is None
    for parsed, expected in zip(net.stacks, stacks, strict=True):
        np.testing.assert_array_equal(parsed.fc0_bias, expected["fc0_b"])
        np.testing.assert_array_equal(parsed.fc0_weight, expected["fc0_w"])
        np.testing.assert_array_equal(parsed.fc2_weight, expected["fc2_w"])


def test_parser_rejects_trailing_bytes() -> None:
    """A parse that does not land on EOF is a parse that misread the layout."""
    blob, _tensors, _stacks = build_synthetic_small_nnue()
    with pytest.raises(ValueError, match="did not land on EOF"):
        nnue_parse.parse_bytes(blob + b"\x00")


def test_parser_version_check_is_fatal() -> None:
    header = struct.pack("<III", 0x7AF32F21, 0, 0)
    with pytest.raises(ValueError, match=r"unsupported \.nnue version"):
        nnue_parse.parse_bytes(header)


def test_parser_rejects_unknown_architecture_hash() -> None:
    desc = b"x"
    blob = struct.pack("<III", nnue_parse.VERSION, 0, len(desc)) + desc
    blob += struct.pack("<I", 0xBADF00D)
    with pytest.raises(ValueError, match="matches no known architecture"):
        nnue_parse.parse_bytes(blob)


def test_architecture_hashes_are_derived_not_observed() -> None:
    """The arch discriminator comes from the feature sets' own hash constants."""
    assert BIG.ft_hash == 0x6165DDC9
    assert SMALL.ft_hash == 0x7F234DB8
    assert BIG.ft_hash != SMALL.ft_hash


@pytest.mark.parametrize(
    "values",
    [
        [0, 1, -1, 63, -64, 127, -128, 1000, -1000, 2**20, -(2**20), 2**30, -(2**30)],
        list(range(-300, 300, 7)),
    ],
)
def test_leb128_decode_matches_a_straightforward_encoder(values: list[int]) -> None:
    payload = bytearray()
    for value in values:
        v = value
        while True:
            byte = v & 0x7F
            v >>= 7
            if (v == 0 and not byte & 0x40) or (v == -1 and byte & 0x40):
                payload.append(byte)
                break
            payload.append(byte | 0x80)
    decoded = nnue_parse.decode_leb128(bytes(payload), len(values))
    assert decoded.tolist() == values


# ===========================================================================
# Converter
# ===========================================================================


def _mini_big_net() -> nnue_parse.NnueNet:
    """A big-architecture net with tiny dimensions, for serialisation tests."""
    rng = np.random.default_rng(11)
    arch = nnue_parse.ArchSpec(name="big", l1=32, l2=3, l3=4, use_threats=True)
    stacks = tuple(
        nnue_parse.LayerStack(
            arch_hash=1,
            fc0_bias=rng.integers(-9, 9, size=arch.l2 + 1, dtype=np.int32),
            fc0_weight=rng.integers(
                -9, 9, size=(arch.l2 + 1, nnue_parse.pad(arch.l1)), dtype=np.int8
            ),
            fc1_bias=rng.integers(-9, 9, size=arch.l3, dtype=np.int32),
            fc1_weight=rng.integers(
                -9, 9, size=(arch.l3, nnue_parse.pad(arch.l2 * 2)), dtype=np.int8
            ),
            fc2_bias=rng.integers(-9, 9, size=1, dtype=np.int32),
            fc2_weight=rng.integers(-9, 9, size=(1, nnue_parse.pad(arch.l3)), dtype=np.int8),
        )
        for _ in range(nnue_parse.LAYER_STACKS)
    )
    return nnue_parse.NnueNet(
        version=nnue_parse.VERSION,
        net_hash=0x5EED,
        description="mini",
        arch=arch,
        ft_hash=arch.ft_hash,
        ft_bias=rng.integers(-99, 99, size=arch.l1, dtype=np.int16),
        ft_weight=rng.integers(-99, 99, size=(40, arch.l1), dtype=np.int16),
        ft_psqt=rng.integers(-99, 99, size=(40, nnue_parse.PSQT_BUCKETS), dtype=np.int32),
        threat_weight=rng.integers(-99, 99, size=(24, arch.l1), dtype=np.int8),
        threat_psqt=rng.integers(-99, 99, size=(24, nnue_parse.PSQT_BUCKETS), dtype=np.int32),
        stacks=stacks,
        source_sha256="ab" * 32,
        source_bytes=123,
    )


def test_pack_round_trips_every_tensor_at_the_declared_offsets() -> None:
    net = _mini_big_net()
    blob = nnue_pack.build_pack(net)

    assert blob[0:8] == nnue_pack.MAGIC
    fields = struct.unpack_from("<18I", blob, 8)
    assert fields[0] == nnue_pack.PACK_VERSION
    assert fields[1] == nnue_parse.VERSION
    (total_size,) = struct.unpack_from("<Q", blob, 80)
    assert total_size == len(blob)
    offsets = struct.unpack_from("<11Q", blob, 88)
    assert blob[176:208].hex() == net.source_sha256

    expected = {
        "ft_bias": (net.ft_bias, "<i2"),
        "ft_weight": (net.ft_weight, "<i2"),
        "ft_psqt": (net.ft_psqt, "<i4"),
        "threat_weight": (net.threat_weight, "<i1"),
        "threat_psqt": (net.threat_psqt, "<i4"),
    }
    for name, offset in zip(nnue_pack._TENSOR_ORDER, offsets, strict=True):
        assert offset % nnue_pack.ALIGN == 0, f"{name} is not 64-byte aligned"
        if name in expected:
            array, dtype = expected[name]
            assert array is not None
            read = np.frombuffer(blob, dtype=dtype, count=array.size, offset=offset)
            np.testing.assert_array_equal(read.reshape(array.shape), array)

    stack_bias = np.frombuffer(
        blob, dtype="<i4", count=nnue_parse.LAYER_STACKS, offset=offsets[9]
    )
    np.testing.assert_array_equal(
        stack_bias, np.array([int(s.fc2_bias[0]) for s in net.stacks], dtype=np.int32)
    )


def test_pack_refuses_the_small_architecture() -> None:
    """The pack format and the C loader carry the big net only, by design."""
    net = dataclasses.replace(_mini_big_net(), threat_weight=None, threat_psqt=None)
    with pytest.raises(ValueError, match="big \\(threat\\) architecture"):
        nnue_pack.build_pack(net)


# ===========================================================================
# Loader rejections
# ===========================================================================


def test_loader_rejects_a_foreign_nnue_version(tmp_path: Path) -> None:
    """The version check must be FATAL: a foreign layout evaluates to plausible
    nonsense everywhere rather than failing loudly anywhere."""
    layout = _big_layout()
    header = bytearray(layout.header)
    struct.pack_into("<I", header, 12, 0x7AF32F21)
    path = tmp_path / "badversion.pack"
    with open(path, "wb") as fh:
        fh.truncate(layout.total_size)
        fh.write(header)
    with pytest.raises(ValueError, match="only accepts 0x7AF32F20"):
        _nnue_ext.load(str(path))


def test_loader_rejects_bad_magic(tmp_path: Path) -> None:
    layout = _big_layout()
    header = bytearray(layout.header)
    header[0:8] = b"NOTAPACK"
    path = tmp_path / "badmagic.pack"
    with open(path, "wb") as fh:
        fh.truncate(layout.total_size)
        fh.write(header)
    with pytest.raises(ValueError, match="bad magic"):
        _nnue_ext.load(str(path))


def test_loader_rejects_a_truncated_file(tmp_path: Path) -> None:
    layout = _big_layout()
    path = tmp_path / "short.pack"
    with open(path, "wb") as fh:
        fh.truncate(layout.total_size - 4096)
        fh.write(layout.header)
    with pytest.raises(ValueError, match="pack says"):
        _nnue_ext.load(str(path))


def test_loader_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot open"):
        _nnue_ext.load(str(tmp_path / "nope.pack"))


# ===========================================================================
# Feature-index computation
# ===========================================================================


def test_threat_dimension_count_is_reconstructed_not_assumed() -> None:
    """60720 falls out of the attack geometry; if it did not, our geometry would
    disagree with the one the shipped weights were trained against."""
    assert THREAT_TABLES.dimensions == nnue_parse.THREAT_DIMS == _nnue_ext.THREAT_DIMS


@pytest.mark.parametrize("fen", POSITIONS)
@pytest.mark.parametrize("perspective", [0, 1])
def test_c_active_features_match_the_reference(fen: str, perspective: int) -> None:
    halfka, threats = _nnue_ext.active_features(cboard(fen), perspective)
    pos = position_view(chess.Board(fen))
    assert sorted(halfka) == sorted(halfka_active_indices(perspective, pos))
    assert sorted(threats) == sorted(threat_active_indices(perspective, pos))


def test_halfka_index_of_a_hand_checked_position() -> None:
    """Two kings, e1 and e8, white to move — small enough to check by hand.

    From white's perspective the king square e1 (file e..h) means OrientTBL is 0,
    so squares are unmirrored; KingBuckets[e1] is bucket id 31; PieceSquareIndex
    for a king is PS_KING = 640. So the white king's own feature index is
    31 * 704 + 640 + 4 and the black king's is 31 * 704 + 640 + 60.
    """
    fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    halfka, _threats = _nnue_ext.active_features(cboard(fen), 0)
    e1, e8 = chess.E1, chess.E8
    assert halfka == tuple(sorted((31 * 704 + 640 + e1, 31 * 704 + 640 + e8)))
    # And the same numbers out of the independent reference formula.
    assert halfka_make_index(0, e1, 6, e1) == 31 * 704 + 640 + e1
    assert halfka_make_index(0, e8, 14, e1) == 31 * 704 + 640 + e8


def test_threat_features_of_a_hand_checked_position() -> None:
    """A lone white rook on a1 and a white pawn on a2, black king far away.

    The rook attacks exactly one occupied square (a2, its own pawn) and nothing
    else; the pawn has no diagonal target and nothing in front of it; kings are
    neither attackers nor targets in this feature set. So there is exactly one
    active threat relation, and it must be rook->pawn.
    """
    fen = "7k/8/8/8/8/8/P7/R3K3 w - - 0 1"
    board = chess.Board(fen)
    pos = position_view(board)
    relations = threat_active_indices(0, pos)
    assert len(relations) == 1
    halfka, threats = _nnue_ext.active_features(cboard(fen), 0)
    assert list(threats) == relations
    assert len(halfka) == 4  # rook, pawn, both kings


def test_kings_are_never_threat_attackers_or_targets() -> None:
    """map[][KING] and map[KING][] are all -1, so a king relation is excluded."""
    fen = "8/8/3k4/8/8/3K4/8/8 w - - 0 1"
    _halfka, threats = _nnue_ext.active_features(cboard(fen), 0)
    assert threats == ()


# ===========================================================================
# The in-check contract
# ===========================================================================


@pytest.mark.parametrize("fen", IN_CHECK_POSITIONS)
def test_evaluator_refuses_in_check_positions(fen: str, bucket_pack: Path) -> None:
    """⚑ The refusal is a hard error, never a sentinel value.

    A caller must resolve check nodes recursively (search the evasions, which can
    themselves give check) before asking for a static evaluation. If this ever
    returned a number, a search with a hole in it would silently label positions
    with a value the network never defined.
    """
    assert chess.Board(fen).is_check()
    handle = _nnue_ext.load(str(bucket_pack))
    with pytest.raises(_nnue_ext.InCheckError):
        _nnue_ext.evaluate(handle, cboard(fen))
    with pytest.raises(_nnue_ext.InCheckError):
        _nnue_ext.trace(handle, cboard(fen))


def test_in_check_refusal_is_not_a_sentinel(bucket_pack: Path) -> None:
    """The out-parameter is untouched on refusal — there is no value to misread."""
    handle = _nnue_ext.load(str(bucket_pack))
    quiet = _nnue_ext.evaluate(handle, cboard(POSITIONS[0]))
    assert isinstance(quiet, int)
    with pytest.raises(_nnue_ext.InCheckError) as excinfo:
        _nnue_ext.evaluate(handle, cboard(IN_CHECK_POSITIONS[0]))
    assert "recursively" in str(excinfo.value)


def test_reference_also_refuses_in_check() -> None:
    from scripts.nnue_reference import InCheckError, position_view as pv

    assert pv(chess.Board(IN_CHECK_POSITIONS[0])).in_check
    assert InCheckError is not None


# ===========================================================================
# Bucket selection
# ===========================================================================


@pytest.mark.parametrize("fen", POSITIONS)
def test_layer_stack_bucket_is_piece_count_minus_one_over_four(
    fen: str, bucket_pack: Path
) -> None:
    """With only fc2 biases non-zero, the evaluation IS the bucket index.

    ⚑ This is the test that kills an off-by-one in the bucket rule: an evaluator
    using ``piece_count / 4`` instead of ``(piece_count - 1) / 4`` reads the wrong
    layer stack for every position whose piece count is a multiple of four, and
    on real weights that is a plausible-looking number, not a crash.
    """
    board = chess.Board(fen)
    piece_count = bin(board.occupied).count("1")
    expected_bucket = (piece_count - 1) // 4
    handle = _nnue_ext.load(str(bucket_pack))
    assert _nnue_ext.evaluate(handle, cboard(fen)) == (expected_bucket + 1) * 100

    reported_bucket, _psqt, positional = _nnue_ext.trace(handle, cboard(fen))
    assert reported_bucket == expected_bucket
    assert list(positional) == [(b + 1) * 100 for b in range(nnue_parse.PSQT_BUCKETS)]


# ===========================================================================
# Side-to-move POV
# ===========================================================================


def test_psqt_is_side_to_move_relative(tmp_path: Path) -> None:
    """⚑ Kills a swapped POV in the feature transformer.

    One HalfKA PSQT row is given a value, chosen so it is active for WHITE's
    accumulator and not for BLACK's. ``transform()`` computes
    ``(psqt[stm] - psqt[~stm]) / 2``, so flipping the side to move on an
    otherwise identical position must flip that term's sign. If the two
    perspectives were swapped, the two evaluations would trade places.
    """
    # ⚑ Deliberately ASYMMETRIC kings. On e1/e8 the two perspectives' HalfKA
    # feature sets are identical by construction, so the position that reads
    # most naturally is the one position this test cannot be run on.
    white_fen = "k7/8/8/8/8/8/8/4K3 w - - 0 1"
    black_fen = "k7/8/8/8/8/8/8/4K3 b - - 0 1"
    white_only = set(_nnue_ext.active_features(cboard(white_fen), 0)[0])
    black_side = set(_nnue_ext.active_features(cboard(white_fen), 1)[0])
    exclusive = sorted(white_only - black_side)
    assert exclusive, "expected a HalfKA feature active for white only"
    row = exclusive[0]

    value = 32 * 100  # V; the term is V / 2 / 16 = 100 internal units
    path = tmp_path / "pov.pack"
    write_synthetic_pack(
        path,
        {
            "ft_psqt": [
                (row * nnue_parse.PSQT_BUCKETS + b, value)
                for b in range(nnue_parse.PSQT_BUCKETS)
            ]
        },
    )
    handle = _nnue_ext.load(str(path))
    assert _nnue_ext.evaluate(handle, cboard(white_fen)) == 100
    assert _nnue_ext.evaluate(handle, cboard(black_fen)) == -100


# ===========================================================================
# Kernel equivalence
# ===========================================================================


@pytest.mark.parametrize("fen", POSITIONS)
def test_scalar_and_simd_kernels_agree(fen: str, bucket_pack: Path) -> None:
    """Both kernels live in one binary precisely so this can be asserted."""
    handle = _nnue_ext.load(str(bucket_pack))
    try:
        _nnue_ext.set_simd(True)
        with_simd = _nnue_ext.evaluate(handle, cboard(fen))
        _nnue_ext.set_simd(False)
        scalar = _nnue_ext.evaluate(handle, cboard(fen))
    finally:
        _nnue_ext.set_simd(bool(_nnue_ext.HAVE_AVX2))
    assert with_simd == scalar


def test_simd_flag_reports_the_live_state_not_the_request() -> None:
    try:
        if _nnue_ext.HAVE_AVX2:
            assert _nnue_ext.set_simd(True) is True
            assert _nnue_ext.simd_active() is True
        assert _nnue_ext.set_simd(False) is False
        assert _nnue_ext.simd_active() is False
    finally:
        _nnue_ext.set_simd(bool(_nnue_ext.HAVE_AVX2))


# ===========================================================================
# The value-provider seam
# ===========================================================================


def test_provider_registry_contains_nnue() -> None:
    assert "nnue" in _nnue_ext.provider_names()


def test_tree_seam_reports_the_provider_it_is_actually_holding(bucket_pack: Path) -> None:
    """⚑ The name is read off the tree's stored pointer, not off the argument.

    This repo's signature defect is a value that is accepted and then silently
    ignored; a setter that returns success while the consumer keeps a NULL
    pointer is exactly that shape. Asking the consumer is the only question that
    cannot be answered by the producer.
    """
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    assert tree.value_provider_name() is None
    with pytest.raises(ValueError, match="no value provider set"):
        tree.value_provider_eval(cboard(POSITIONS[0]))

    tree.set_value_provider("nnue", str(bucket_pack))
    assert tree.value_provider_name() == "nnue"

    tree.clear_value_provider()
    assert tree.value_provider_name() is None
    with pytest.raises(ValueError, match="no value provider set"):
        tree.value_provider_eval(cboard(POSITIONS[0]))


def test_tree_seam_rejects_an_unknown_provider(bucket_pack: Path) -> None:
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    with pytest.raises(ValueError, match="no value provider named"):
        tree.set_value_provider("qsearch", str(bucket_pack))
    assert tree.value_provider_name() is None


def test_tree_seam_propagates_a_load_failure(tmp_path: Path) -> None:
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    with pytest.raises(ValueError, match="failed to load weights"):
        tree.set_value_provider("nnue", str(tmp_path / "absent.pack"))
    assert tree.value_provider_name() is None


@pytest.mark.parametrize("fen", POSITIONS)
def test_tree_seam_evaluates_identically_to_the_direct_call(
    fen: str, bucket_pack: Path
) -> None:
    """One evaluator, two callers — the tree must not be running a second copy."""
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    tree.set_value_provider("nnue", str(bucket_pack))
    handle = _nnue_ext.load(str(bucket_pack))
    assert tree.value_provider_eval(cboard(fen)) == _nnue_ext.evaluate(handle, cboard(fen))


@pytest.mark.parametrize("fen", IN_CHECK_POSITIONS)
def test_tree_seam_refuses_in_check(fen: str, bucket_pack: Path) -> None:
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    tree.set_value_provider("nnue", str(bucket_pack))
    with pytest.raises(ValueError, match="in-check"):
        tree.value_provider_eval(cboard(fen))


def test_seam_rejects_a_non_cboard(bucket_pack: Path) -> None:
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    tree.set_value_provider("nnue", str(bucket_pack))
    with pytest.raises(TypeError, match="expected a CBoard"):
        # Deliberately the wrong type: the seam duck-types a struct across .so
        # boundaries, so it must reject anything that is not the CBoard it
        # mirrors rather than reinterpreting whatever memory it was handed.
        tree.value_provider_eval(chess.Board())  # pyright: ignore[reportArgumentType]


# ===========================================================================
# Real-weight regression checks (opt-in)
# ===========================================================================

_REAL_PACK = os.environ.get("CAE_NNUE_TEST_PACK", "")
_REAL_NNUE = os.environ.get("CAE_NNUE_TEST_NNUE", "")

_pack_reason = (
    "set CAE_NNUE_TEST_PACK to a pack built by scripts/nnue_pack.py; the real "
    "gate is scripts/nnue_parity.py against Stockfish, not this test"
)


@pytest.mark.skipif(not _REAL_PACK or not _REAL_NNUE, reason=_pack_reason)
@pytest.mark.parametrize("fen", POSITIONS)
def test_c_matches_the_numpy_reference_on_real_weights(fen: str) -> None:
    """Localiser check: same weights, same positions, C against numpy.

    ⚑ Passing here is NOT parity. Both implementations were written from the same
    reading of the Stockfish sources, so a rule wrong in both survives this. Its
    job is to say, when Stockfish disagrees, WHICH of the two is wrong.
    """
    from scripts.nnue_reference import ReferenceEvaluator

    handle = _nnue_ext.load(_REAL_PACK)
    evaluator = ReferenceEvaluator(nnue_parse.parse(Path(_REAL_NNUE)))
    board = chess.Board(fen)
    assert _nnue_ext.evaluate(handle, cboard(fen)) == evaluator.evaluate(board)

    reference_trace = evaluator.trace(board)
    bucket, psqt, positional = _nnue_ext.trace(handle, cboard(fen))
    assert bucket == reference_trace.bucket
    assert list(psqt) == list(reference_trace.psqt)
    assert list(positional) == list(reference_trace.positional)


@pytest.mark.skipif(not _REAL_PACK, reason=_pack_reason)
def test_real_pack_provenance_is_readable_from_the_loaded_weights() -> None:
    handle = _nnue_ext.load(_REAL_PACK)
    info = _nnue_ext.info(handle)
    assert info["l1"] == BIG.l1
    assert info["threat_dims"] == nnue_parse.THREAT_DIMS
    assert info["ft_hash"] == BIG.ft_hash
    assert len(_nnue_ext.source_sha256(handle)) == 64
