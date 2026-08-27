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
import gzip
import json
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


#: Element type of each packed tensor, so a test can poke any of them and not
#: just the int32 ones. Mirrors the CaeNnueWeights pointer types.
TENSOR_DTYPE: dict[str, type[np.number]] = {
    "ft_bias": np.int16,
    "ft_weight": np.int16,
    "ft_psqt": np.int32,
    "threat_weight": np.int8,
    "threat_psqt": np.int32,
    "fc0_bias": np.int32,
    "fc0_weight": np.int8,
    "fc1_bias": np.int32,
    "fc1_weight": np.int8,
    "fc2_bias": np.int32,
    "fc2_weight": np.int8,
}


def write_synthetic_pack(
    path: Path,
    pokes: dict[str, list[tuple[int, int]]] | None = None,
    blobs: dict[str, list[tuple[int, np.ndarray]]] | None = None,
    header: bytes | None = None,
) -> None:
    """A real-layout pack that is all zeros except for ``pokes`` and ``blobs``.

    ``pokes`` maps a tensor name to (element_index, value) pairs; ``blobs`` maps
    one to (element_index, array) pairs written contiguously from that element.
    Element size comes from TENSOR_DTYPE, so int8/int16 tensors are writable too.
    ``header`` replaces the layout's own header, for tests that need a pack whose
    declared dimensions are deliberately inconsistent. The file is created sparse
    — the 111 MB of holes read back as zeros and cost no disk.
    """
    layout = _big_layout()
    with open(path, "wb") as fh:
        fh.truncate(layout.total_size)
        fh.write(header if header is not None else layout.header)
        for name, entries in (pokes or {}).items():
            dtype = np.dtype(TENSOR_DTYPE[name])
            for index, value in entries:
                fh.seek(layout.offsets[name] + index * dtype.itemsize)
                fh.write(np.asarray(value, dtype=dtype).tobytes())
        for name, chunks in (blobs or {}).items():
            dtype = np.dtype(TENSOR_DTYPE[name])
            for index, array in chunks:
                fh.seek(layout.offsets[name] + index * dtype.itemsize)
                fh.write(np.ascontiguousarray(array, dtype=dtype).tobytes())


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
    # ⚑ A VALID file: the header hash and the per-stack architecture hash are
    # the real ones for this architecture. They used to be 0xDEADBEEF and
    # 0xC0FFEE01, which the parser accepted because it only checked that the
    # stacks agreed with each other.
    out += struct.pack("<III", nnue_parse.VERSION, SMALL.net_hash, len(desc))
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
        out += struct.pack("<I", SMALL.layer_hash)
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
    assert net.net_hash == SMALL.net_hash
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
    arch = nnue_parse.ArchSpec(
        name="big", l1=32, l2=3, l3=4, use_threats=True, layer_hash=BIG.layer_hash
    )
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


def _full_size_zero_big_net() -> nnue_parse.NnueNet:
    """A big-architecture net at REAL dimensions with all-zero weights.

    Full size because the reference evaluator indexes ft_weight/threat_weight by
    the true feature indices, so the mini net used for the serialisation tests
    would raise IndexError rather than evaluate. Zeros because the point here is
    reaching the code path, not the number at the end of it; np.zeros pages are
    never touched, so the 108 MB is address space rather than memory.
    """
    arch = nnue_parse.ARCHS[0]
    pad = nnue_parse.pad
    stacks = tuple(
        nnue_parse.LayerStack(
            arch_hash=1,
            fc0_bias=np.zeros(arch.l2 + 1, dtype=np.int32),
            fc0_weight=np.zeros((arch.l2 + 1, pad(arch.l1)), dtype=np.int8),
            fc1_bias=np.zeros(arch.l3, dtype=np.int32),
            fc1_weight=np.zeros((arch.l3, pad(arch.l2 * 2)), dtype=np.int8),
            fc2_bias=np.zeros(1, dtype=np.int32),
            fc2_weight=np.zeros((1, pad(arch.l3)), dtype=np.int8),
        )
        for _ in range(nnue_parse.LAYER_STACKS)
    )
    return nnue_parse.NnueNet(
        version=nnue_parse.VERSION,
        net_hash=0x5EED,
        description="zero",
        arch=arch,
        ft_hash=arch.ft_hash,
        ft_bias=np.zeros(arch.l1, dtype=np.int16),
        ft_weight=np.zeros((nnue_parse.HALFKA_DIMS, arch.l1), dtype=np.int16),
        ft_psqt=np.zeros((nnue_parse.HALFKA_DIMS, nnue_parse.PSQT_BUCKETS), dtype=np.int32),
        threat_weight=np.zeros((nnue_parse.THREAT_DIMS, arch.l1), dtype=np.int8),
        threat_psqt=np.zeros((nnue_parse.THREAT_DIMS, nnue_parse.PSQT_BUCKETS), dtype=np.int32),
        stacks=stacks,
        source_sha256="cd" * 32,
        source_bytes=1,
    )


def test_reference_also_refuses_in_check() -> None:
    """The numpy reference refuses in check too — asserted by CALLING it.

    ⚑ This test used to check that ``position_view(...).in_check`` was True and
    that ``InCheckError`` was importable. Both were true no matter what
    ``ReferenceEvaluator.evaluate`` did: an import is not a behaviour and a flag
    on a different object is not the refusal. It would have passed unchanged if
    the reference had happily returned a number for a king in check — which is
    the one thing it exists here to rule out, because the reference is the
    bisector a parity failure gets localised with.
    """
    from scripts.nnue_reference import InCheckError, ReferenceEvaluator

    evaluator = ReferenceEvaluator(_full_size_zero_big_net())

    # Positive half: the same evaluator DOES return a value for a quiet
    # position, so the refusal below is about the check and not about the
    # evaluator being broken for everything.
    assert isinstance(evaluator.evaluate(chess.Board(POSITIONS[0])), int)

    for fen in IN_CHECK_POSITIONS:
        with pytest.raises(InCheckError):
            evaluator.evaluate(chess.Board(fen))
        with pytest.raises(InCheckError):
            evaluator.trace(chess.Board(fen))


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


@pytest.fixture(scope="module")
def dense_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A pack with NONZERO ft/threat/psqt rows for every feature POSITIONS uses.

    ⚑ THE ALL-ZERO FIXTURE CANNOT TELL THE TWO KERNELS APART. With zero feature
    weights the accumulator is zero everywhere, and the AVX2 transform's
    clamp/mulhi/packus/permute chain maps zeros to zeros whatever order its lanes
    are in — so a wrong ``_mm256_permute4x64_epi64`` selector, the single most
    likely SIMD defect in this file, agrees with the scalar loop perfectly. The
    weights below are varied ACROSS j precisely so lane order is observable.
    """
    rng = np.random.default_rng(4242)
    active_halfka: set[int] = set()
    active_threats: set[int] = set()
    for fen in POSITIONS:
        for perspective in (0, 1):
            ka, th = _nnue_ext.active_features(cboard(fen), perspective)
            active_halfka.update(ka)
            active_threats.update(th)

    l1 = BIG.l1
    nb = nnue_parse.PSQT_BUCKETS
    blobs: dict[str, list[tuple[int, np.ndarray]]] = {
        "ft_bias": [(0, rng.integers(-120, 120, size=l1, dtype=np.int16))],
        "ft_weight": [
            (row * l1, rng.integers(-40, 40, size=l1, dtype=np.int16))
            for row in sorted(active_halfka)
        ],
        "ft_psqt": [
            (row * nb, rng.integers(-300, 300, size=nb, dtype=np.int32))
            for row in sorted(active_halfka)
        ],
        "threat_weight": [
            (row * l1, rng.integers(-30, 30, size=l1, dtype=np.int8))
            for row in sorted(active_threats)
        ],
        "threat_psqt": [
            (row * nb, rng.integers(-300, 300, size=nb, dtype=np.int32))
            for row in sorted(active_threats)
        ],
        "fc0_bias": [(0, rng.integers(-500, 500, size=nnue_parse.LAYER_STACKS * (BIG.l2 + 1),
                                      dtype=np.int32))],
        "fc0_weight": [
            (0, rng.integers(-20, 20,
                             size=nnue_parse.LAYER_STACKS * (BIG.l2 + 1) * nnue_parse.pad(l1),
                             dtype=np.int8))
        ],
        "fc1_bias": [(0, rng.integers(-500, 500, size=nnue_parse.LAYER_STACKS * BIG.l3,
                                      dtype=np.int32))],
        "fc1_weight": [
            (0, rng.integers(-20, 20,
                             size=nnue_parse.LAYER_STACKS * BIG.l3 * nnue_parse.pad(BIG.l2 * 2),
                             dtype=np.int8))
        ],
        "fc2_bias": [(0, rng.integers(-500, 500, size=nnue_parse.LAYER_STACKS, dtype=np.int32))],
        "fc2_weight": [
            (0, rng.integers(-20, 20, size=nnue_parse.LAYER_STACKS * nnue_parse.pad(BIG.l3),
                             dtype=np.int8))
        ],
    }
    path = tmp_path_factory.mktemp("nnue") / "dense.pack"
    write_synthetic_pack(path, blobs=blobs)
    return path


requires_avx2 = pytest.mark.skipif(
    not _nnue_ext.HAVE_AVX2,
    reason="portable build: no AVX2 kernels compiled in, so there is no second "
    "kernel to compare against (CAE_EXT_NATIVE unset, as in CI)",
)


def test_the_dense_pack_actually_produces_varied_evaluations(dense_pack: Path) -> None:
    """Guards the guard: a degenerate dense pack would make agreement vacuous."""
    handle = _nnue_ext.load(str(dense_pack))
    values = [_nnue_ext.evaluate(handle, cboard(fen)) for fen in POSITIONS]
    assert any(v != 0 for v in values), "dense pack evaluates to all zeros"
    assert len(set(values)) > 1, "dense pack gives every position the same value"


@requires_avx2
@pytest.mark.parametrize("fen", POSITIONS)
@pytest.mark.parametrize("pack_name", ["bucket_pack", "dense_pack"])
def test_scalar_and_simd_kernels_agree(
    fen: str, pack_name: str, request: pytest.FixtureRequest
) -> None:
    """Both kernels live in one binary precisely so this can be asserted.

    Run against BOTH the all-zero bucket pack and the dense one: the zero pack
    proves the plumbing, and only the dense pack can actually see a lane-order
    defect.
    """
    pack: Path = request.getfixturevalue(pack_name)
    handle = _nnue_ext.load(str(pack))
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
    # InCheckError subclasses ValueError, so this stays a ValueError check AND
    # pins the wording of the contract the caller has to satisfy.
    with pytest.raises(ValueError, match="resolve check nodes recursively"):
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


# ===========================================================================
# Pack internal consistency — the RELATIONS between fields, not their ranges
# ===========================================================================

#: Byte offset of each u32 header field, from the "<18I" pack at offset 8.
_HDR_U32 = {
    name: 8 + 4 * i
    for i, name in enumerate(
        [
            "pack_version", "nnue_version", "net_hash", "ft_hash",
            "l1", "l2", "l3", "psqt_buckets", "layer_stacks",
            "halfka_dims", "threat_dims", "use_threats",
            "fc0_outputs", "fc0_padded_in", "fc1_outputs", "fc1_padded_in",
            "fc2_padded_in", "reserved0",
        ]
    )
}


def _pack_with_header_field(tmp_path: Path, name: str, value: int, filename: str) -> Path:
    """A layout-correct pack with ONE header field overwritten."""
    layout = _big_layout()
    header = bytearray(layout.header)
    struct.pack_into("<I", header, _HDR_U32[name], value)
    path = tmp_path / filename
    with open(path, "wb") as fh:
        fh.truncate(layout.total_size)
        fh.write(bytes(header))
    return path


def test_header_field_offsets_match_the_packer() -> None:
    """The offset table above is derived from the packer, not from memory."""
    layout = _big_layout()
    assert struct.unpack_from("<I", layout.header, _HDR_U32["l1"])[0] == BIG.l1
    assert struct.unpack_from("<I", layout.header, _HDR_U32["l2"])[0] == BIG.l2
    assert struct.unpack_from("<I", layout.header, _HDR_U32["l3"])[0] == BIG.l3
    assert struct.unpack_from("<I", layout.header, _HDR_U32["fc0_outputs"])[0] == BIG.l2 + 1
    assert struct.unpack_from("<I", layout.header, _HDR_U32["fc1_outputs"])[0] == BIG.l3
    assert (
        struct.unpack_from("<I", layout.header, _HDR_U32["fc0_padded_in"])[0]
        == nnue_parse.pad(BIG.l1)
    )
    assert (
        struct.unpack_from("<I", layout.header, _HDR_U32["fc2_padded_in"])[0]
        == nnue_parse.pad(BIG.l3)
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        # fc0_outputs == l2 + 1. With l2 outputs, propagate() reads fc0[l2] —
        # one past what the loop initialised — as the forward-skip term, i.e.
        # uninitialised stack returned as an evaluation.
        ("fc0_outputs", BIG.l2, "needs exactly l2 \\+ 1"),
        ("fc0_outputs", BIG.l2 + 2, "needs exactly l2 \\+ 1"),
        # fc1_outputs == l3.
        ("fc1_outputs", BIG.l3 - 1, "fc1_outputs"),
        # Padded widths must cover their consumers, or the layer loops walk off
        # the end of a row into the next one.
        ("fc0_padded_in", BIG.l1 - 64, "narrower than its l1"),
        ("fc1_padded_in", 2 * BIG.l2 - 2, "narrower than its 2\\*l2"),
        ("fc2_padded_in", BIG.l3 - 1, "fc2_padded_in"),
        # Degenerate layers.
        ("l2", 0, "empty layer"),
        ("l3", 0, "empty layer"),
    ],
)
def test_loader_rejects_internally_inconsistent_dimensions(
    tmp_path: Path, field: str, value: int, message: str
) -> None:
    """⚑ Every one of these passes the per-field RANGE checks.

    The defect class is relational: fields that are each individually plausible
    but cannot be true together. Our own packer cannot emit any of them, which is
    exactly why bind() has to reject them — it is the only thing between a
    hand-made or corrupted pack and an evaluation that is silently wrong rather
    than absent.
    """
    path = _pack_with_header_field(tmp_path, field, value, f"bad_{field}_{value}.pack")
    with pytest.raises(ValueError, match=message):
        _nnue_ext.load(str(path))


def test_loader_rejects_an_l1_the_kernels_cannot_step(tmp_path: Path) -> None:
    """l1 must suit the compiled kernels, or the two of them silently diverge.

    The AVX2 transform steps 32 int16 lanes over each half of the accumulator, so
    it needs l1 % 64 == 0 where the scalar loop needs only 32. On a build with
    AVX2 compiled in, accepting l1 % 64 == 32 would mean the two kernels compute
    different numbers from the same weights — and "the kernels agree" is the
    whole reason for shipping both.
    """
    required = 64 if _nnue_ext.HAVE_AVX2 else 32
    bad = required + 32 if required == 64 else required + 16
    path = _pack_with_header_field(tmp_path, "l1", bad, "bad_l1.pack")
    with pytest.raises(ValueError, match="not a multiple of"):
        _nnue_ext.load(str(path))


def test_loader_rejects_a_tensor_offset_that_wraps_uint64(tmp_path: Path) -> None:
    """⚑ The bounds check is a subtraction because addition wraps.

    `off + size > map_size` is the natural phrasing and a crafted off near
    2**64 makes the sum small, so the tensor passes the check and then points
    wherever it likes. Writing it as `size > map_size - off` cannot wrap.
    """
    layout = _big_layout()
    header = bytearray(layout.header)
    # off[1] is ft_weight, 46 MB; this value plus that size wraps to a small
    # number in uint64 arithmetic.
    struct.pack_into("<Q", header, 88 + 8 * 1, (1 << 64) - 1024)
    path = tmp_path / "wrap.pack"
    with open(path, "wb") as fh:
        fh.truncate(layout.total_size)
        fh.write(bytes(header))
    with pytest.raises(ValueError, match="outside the file"):
        _nnue_ext.load(str(path))


# ===========================================================================
# Malformed positions are rejected, not evaluated
# ===========================================================================


def _raw_cboard(
    *,
    pawns: int = 0,
    knights: int = 0,
    bishops: int = 0,
    rooks: int = 0,
    queens: int = 0,
    kings: int = 0,
    white_occ: int = 0,
    black_occ: int = 0,
    turn: int = 1,
) -> CBoard:
    """A CBoard straight from raw bitboards, bypassing python-chess validation.

    from_raw takes positional integers only; the keyword wrapper is here so the
    tests below read as the boards they describe.
    """
    return CBoard.from_raw(
        pawns, knights, bishops, rooks, queens, kings,
        white_occ, black_occ, turn, 0, -1, 0,
    )


def test_evaluator_rejects_overlapping_piece_bitboards(bucket_pack: Path) -> None:
    """⚑ A CBoard is not the same thing as a chess position.

    from_raw() does not require the piece-type masks to be disjoint, and a square
    claimed by several types is emitted as several attackers — which multiplies
    the threat-relation count far past anything a legal board reaches. The
    relation buffer is a fixed array on the caller's stack, so the alternative to
    rejecting this is a stack smash reached from Python-supplied integers.
    """
    handle = _nnue_ext.load(str(bucket_pack))
    # ⚑ 30 occupied squares, deliberately UNDER the 32-piece limit. An earlier
    # version of this test filled the board, and every piece-count check fired
    # before the disjointness check was reached — so it passed with the
    # disjointness check deleted, which is precisely the mutant it exists to
    # kill. A rejection test has to fail for the reason it names.
    kings = (1 << 4) | (1 << 60)
    overlap = ((1 << 28) - 1) << 16   # 28 squares on ranks 3-6, no back rank
    board = _raw_cboard(
        knights=overlap,
        bishops=overlap,   # same squares claimed by a second piece type
        rooks=overlap,
        queens=overlap,
        kings=kings,
        white_occ=overlap | (1 << 4),
        black_occ=1 << 60,
        turn=1,
    )
    assert bin(overlap | kings).count("1") == 30
    with pytest.raises(ValueError, match="malformed position"):
        _nnue_ext.evaluate(handle, board)


def test_evaluator_rejects_a_pawn_on_the_back_rank(bucket_pack: Path) -> None:
    """Pawns on rank 1/8 produce in-range but meaningless threat indices."""
    handle = _nnue_ext.load(str(bucket_pack))
    board = _raw_cboard(
        pawns=1 << 0,
        knights=0,
        bishops=0,
        rooks=0,
        queens=0,
        kings=(1 << 4) | (1 << 60),
        white_occ=(1 << 0) | (1 << 4),
        black_occ=1 << 60,
        turn=1,
    )
    with pytest.raises(ValueError, match="malformed position"):
        _nnue_ext.evaluate(handle, board)


def test_evaluator_rejects_a_board_with_two_kings_of_one_colour(bucket_pack: Path) -> None:
    handle = _nnue_ext.load(str(bucket_pack))
    board = _raw_cboard(
        pawns=0, knights=0, bishops=0, rooks=0, queens=0,
        kings=(1 << 4) | (1 << 6) | (1 << 60),
        white_occ=(1 << 4) | (1 << 6),
        black_occ=1 << 60,
        turn=1,
    )
    with pytest.raises(ValueError, match="malformed position"):
        _nnue_ext.evaluate(handle, board)


def test_a_legal_position_still_evaluates_after_the_validators(bucket_pack: Path) -> None:
    """The positive control: the validators reject malformed boards ONLY.

    Without this, every rejection test above would still pass if the validator
    rejected everything.
    """
    handle = _nnue_ext.load(str(bucket_pack))
    for fen in POSITIONS:
        assert isinstance(_nnue_ext.evaluate(handle, cboard(fen)), int)


# ===========================================================================
# The provider capsule: ONE copy of the evaluator, shared across extensions
# ===========================================================================


def test_the_tree_uses_the_evaluator_modules_own_weight_cache(
    bucket_pack: Path, tmp_path: Path
) -> None:
    """⚑ THE OBSERVATION THAT PROVES THE TREE IS NOT RUNNING A SECOND COPY.

    The weight cache is a static of the evaluator, so each extension that
    COMPILED the evaluator in would have its own. This loads a pack through the
    TREE only and then reads the cache count from _nnue_ext: if the tree carried
    its own copy of the evaluator, the file would be mapped into that copy's
    cache and this count would stay where it started.

    That was the real state of this seam before the capsule: the tree included
    _nnue_provider.h, so set_simd() on _nnue_ext did not govern what the tree
    ran, and a shared weight file was mapped twice at 62 MB each — both of them
    silent, because the tree kept returning perfectly plausible evaluations.
    """
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    # A path this process has not loaded, so the count change is attributable.
    unique = tmp_path / "capsule_probe.pack"
    unique.write_bytes(bucket_pack.read_bytes())

    before = _nnue_ext.weight_cache_size()
    tree = MCTSTree()
    tree.set_value_provider("nnue", str(unique))
    try:
        assert _nnue_ext.weight_cache_size() == before + 1
    finally:
        tree.clear_value_provider()
    assert _nnue_ext.weight_cache_size() == before


def test_loading_one_path_twice_maps_it_once(bucket_pack: Path, tmp_path: Path) -> None:
    """The refcounted cache keeps one mapping per path, across both surfaces."""
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    unique = tmp_path / "shared.pack"
    unique.write_bytes(bucket_pack.read_bytes())

    before = _nnue_ext.weight_cache_size()
    handle = _nnue_ext.load(str(unique))
    tree = MCTSTree()
    tree.set_value_provider("nnue", str(unique))
    try:
        assert _nnue_ext.weight_cache_size() == before + 1
        assert _nnue_ext.evaluate(handle, cboard(POSITIONS[0])) == tree.value_provider_eval(
            cboard(POSITIONS[0])
        )
    finally:
        tree.clear_value_provider()


@requires_avx2
def test_set_simd_on_the_evaluator_module_governs_the_tree_path(bucket_pack: Path) -> None:
    """⚑ ANNOUNCED FROM THE CONSUMER'S OWN POINTER.

    The tree reports the kernel by asking the vtable it is holding, so this
    asserts that a switch thrown on _nnue_ext reaches the code the TREE will run
    — not that two independent flags happen to agree. With the evaluator
    compiled into both extensions this fails: the tree's copy never hears about
    _nnue_ext.set_simd() and keeps reporting (and running) the other kernel.
    """
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    tree.set_value_provider("nnue", str(bucket_pack))
    try:
        _nnue_ext.set_simd(True)
        assert tree.value_provider_kernel() == "avx2"
        assert _nnue_ext.simd_active() is True

        _nnue_ext.set_simd(False)
        assert tree.value_provider_kernel() == "scalar"
        assert _nnue_ext.simd_active() is False
    finally:
        _nnue_ext.set_simd(bool(_nnue_ext.HAVE_AVX2))
        tree.clear_value_provider()


def test_a_provider_can_be_installed_from_its_capsule_directly(bucket_pack: Path) -> None:
    """Future providers need no edit to the tree — that is the point of the shape."""
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    tree.set_value_provider(_nnue_ext.value_provider_capsule, str(bucket_pack))
    try:
        assert tree.value_provider_name() == "nnue"
        assert tree.value_provider_eval(cboard(POSITIONS[0])) == _nnue_ext.evaluate(
            _nnue_ext.load(str(bucket_pack)), cboard(POSITIONS[0])
        )
    finally:
        tree.clear_value_provider()


def test_the_tree_rejects_a_capsule_that_is_not_a_value_provider(bucket_pack: Path) -> None:
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    weights_capsule = _nnue_ext.load(str(bucket_pack))   # a capsule, wrong name
    with pytest.raises(ValueError, match=r"cae\.value_provider"):
        tree.set_value_provider(weights_capsule, str(bucket_pack))
    assert tree.value_provider_name() is None


def test_clearing_a_provider_during_an_evaluation_is_safe(bucket_pack: Path) -> None:
    """⚑ The evaluation holds its OWN reference across the GIL release.

    Without it, a thread clearing the provider unmaps 62 MB of weights while
    another thread is reading them — and a read of a freed read-only mapping
    usually returns data rather than crashing, so the symptom is a wrong
    evaluation, not a signal.
    """
    import threading

    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    tree.set_value_provider("nnue", str(bucket_pack))
    boards = [cboard(fen) for fen in POSITIONS]
    expected = {id(b): tree.value_provider_eval(b) for b in boards}

    errors: list[BaseException] = []
    stop = threading.Event()

    def hammer() -> None:
        try:
            while not stop.is_set():
                for b in boards:
                    assert tree.value_provider_eval(b) == expected[id(b)]
        except BaseException as exc:  # reported below, never swallowed
            errors.append(exc)

    workers = [threading.Thread(target=hammer) for _ in range(2)]
    for w in workers:
        w.start()
    try:
        for _ in range(30):
            tree.set_value_provider("nnue", str(bucket_pack))
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=30)
    tree.clear_value_provider()

    # A racing eval may legitimately observe a cleared provider; it may never
    # observe a WRONG number or a crash.
    assert [e for e in errors if not isinstance(e, ValueError)] == []


def test_reinitialising_a_tree_releases_its_provider(bucket_pack: Path, tmp_path: Path) -> None:
    """__init__ can be called twice; the second call must not orphan a mapping."""
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    unique = tmp_path / "reinit.pack"
    unique.write_bytes(bucket_pack.read_bytes())

    before = _nnue_ext.weight_cache_size()
    tree = MCTSTree()
    tree.set_value_provider("nnue", str(unique))
    assert _nnue_ext.weight_cache_size() == before + 1
    # Re-running __init__ on a live object is the point of this test, so the
    # explicit call is deliberate.
    tree.__init__()
    assert tree.value_provider_name() is None
    assert _nnue_ext.weight_cache_size() == before


def test_the_seam_rejects_an_object_that_merely_calls_itself_cboard(bucket_pack: Path) -> None:
    """⚑ A NAME IS NOT A TYPE.

    Both surfaces used to accept anything whose tp_name ended in "CBoard" and
    then reinterpret its storage as a much larger struct — an out-of-bounds read
    behind a promised TypeError, performed after the GIL is released.
    """
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    class CBoard:  # deliberately collides with the real class's name
        pass

    impostor = CBoard()
    handle = _nnue_ext.load(str(bucket_pack))
    with pytest.raises(TypeError):
        _nnue_ext.evaluate(handle, impostor)  # pyright: ignore[reportArgumentType]

    tree = MCTSTree()
    tree.set_value_provider("nnue", str(bucket_pack))
    try:
        with pytest.raises(TypeError):
            tree.value_provider_eval(impostor)  # pyright: ignore[reportArgumentType]
    finally:
        tree.clear_value_provider()


@pytest.mark.parametrize("fen", IN_CHECK_POSITIONS)
def test_the_tree_raises_the_providers_own_in_check_type(fen: str, bucket_pack: Path) -> None:
    """⚑ Caught by TYPE, not by message text.

    The recursive check resolver lands in a third module and will catch this
    exception. A ValueError carrying an explanatory string would force it to
    match on wording, which nothing keeps stable.
    """
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    tree.set_value_provider("nnue", str(bucket_pack))
    try:
        with pytest.raises(_nnue_ext.InCheckError):
            tree.value_provider_eval(cboard(fen))
    finally:
        tree.clear_value_provider()


# ===========================================================================
# The parity gate's own failure paths
# ===========================================================================


class _FakeStockfish:
    """Stands in for the engine so the gate's control flow can be tested.

    ``scores`` maps FEN -> value, or the sentinel "check" to refuse the way the
    engine refuses an in-check position.
    """

    def __init__(self, scores: dict[str, object], eval_file: str) -> None:
        self.eval_file = eval_file
        self.scores = scores
        self.asked: list[str] = []

    def evaluate(self, fen: str) -> int:
        from scripts.nnue_parity import InCheckRefused

        self.asked.append(fen)
        value = self.scores[fen]
        if value == "check":
            raise InCheckRefused(fen)
        assert isinstance(value, int)
        return value

    def close(self) -> None:
        pass

    def __enter__(self) -> _FakeStockfish:
        return self

    def __exit__(self, *_exc: object) -> None:
        pass


def _agreeing_scores(bucket_pack: Path, fens: list[str]) -> dict[str, object]:
    """What a correct engine would print for these FENs under this pack.

    ⚑ The default for every gate test, so that a gate under test is the ONLY
    thing that can fail the run. Feeding arbitrary numbers instead makes the
    harness exit non-zero because of a MISMATCH, and a test asserting only
    "non-zero" then passes with the gate deleted — which is how the first
    version of these tests passed against a deliberately reintroduced bug.
    """
    handle = _nnue_ext.load(str(bucket_pack))
    return {fen: _nnue_ext.evaluate(handle, cboard(fen)) for fen in fens}


def _run_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bucket_pack: Path,
    fens: list[str],
    scores: dict[str, object] | None = None,
    eval_file: str | None = None,
    extra_args: list[str] | None = None,
) -> tuple[int, _FakeStockfish]:
    """Drive nnue_parity.main() against a stubbed engine."""
    if scores is None:
        scores = _agreeing_scores(bucket_pack, fens)
    import scripts.nnue_parity as parity

    handle = _nnue_ext.load(str(bucket_pack))
    sha = _nnue_ext.source_sha256(handle)
    fake = _FakeStockfish(scores, f"nn-{sha[:12]}.nnue" if eval_file is None else eval_file)
    monkeypatch.setattr(parity, "StockfishDriver", lambda _path: fake)

    fens_file = tmp_path / "fens.txt"
    fens_file.write_text("\n".join(fens) + ("\n" if fens else ""))
    args = [
        "--pack", str(bucket_pack),
        "--stockfish", "/nonexistent-engine",
        "--fens-in", str(fens_file),
        "--observations", str(tmp_path / "obs.jsonl.gz"),
        *(extra_args or []),
    ]
    return parity.main(args), fake


def test_parity_gate_fails_when_it_checked_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bucket_pack: Path
) -> None:
    """⚑ AN EMPTY SAMPLE IS NOT A PASS.

    An empty --fens-in, --n 0, or a sampler regression returning [] used to print
    "PARITY PASSED" and exit 0 — the same words, and the same exit code, that
    fifty thousand exact matches produce. This is the gate that decides whether a
    hundred-million-row corpus gets labelled, so "we compared nothing" must not
    be reportable as "we agree".
    """
    code, fake = _run_gate(monkeypatch, tmp_path, bucket_pack, [], {})
    assert code == 2, "an empty sample must be INCONCLUSIVE (2), not a pass or a mismatch"
    assert fake.asked == []


def test_parity_gate_fails_when_every_position_was_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bucket_pack: Path
) -> None:
    """All-refused is zero comparisons, however many FENs went in."""
    fens = list(IN_CHECK_POSITIONS)
    code, _ = _run_gate(
        monkeypatch, tmp_path, bucket_pack, fens, dict.fromkeys(fens, "check")
    )
    assert code == 2


def test_parity_gate_fails_when_the_engines_eval_file_is_unreadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bucket_pack: Path
) -> None:
    """⚑ A PROVENANCE GATE THAT SKIPS WHEN IT CANNOT READ IS NOT A GATE.

    The check was written as `if sf.eval_file and sf.eval_file != expected`, so
    an engine whose option line the regex could not parse — a build change, a
    renamed option — produced an empty string, skipped the comparison, and the
    gate reported parity against an oracle nobody had verified. That is exactly
    the drifted case the check exists for, and the one case it could not fire on.
    """
    # ⚑ The engine AGREES on every position here. So the only thing that can
    # fail this run is the provenance gate — an earlier version passed numbers
    # the evaluator disagreed with, and then "exit != 0" was satisfied by the
    # mismatch while the provenance gate was deleted.
    fens = [POSITIONS[0]]
    code, _ = _run_gate(monkeypatch, tmp_path, bucket_pack, fens, eval_file="")
    assert code == 2, "an unreadable EvalFile must refuse the run, not report parity"


def test_parity_gate_fails_when_the_engine_runs_a_different_net(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bucket_pack: Path
) -> None:
    fens = [POSITIONS[0]]
    code, _ = _run_gate(
        monkeypatch, tmp_path, bucket_pack, fens, eval_file="nn-ffffffffffff.nnue"
    )
    assert code == 2


def test_parity_gate_passes_only_when_it_really_compared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bucket_pack: Path
) -> None:
    """The positive control for all four gates above.

    Without it every one of them would still pass if main() had simply been made
    to always fail.
    """
    code, fake = _run_gate(monkeypatch, tmp_path, bucket_pack, list(POSITIONS))
    assert code == 0
    assert len(fake.asked) == len(POSITIONS)


def test_parity_gate_reports_a_real_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bucket_pack: Path
) -> None:
    scores = _agreeing_scores(bucket_pack, list(POSITIONS))
    first = POSITIONS[0]
    scores[first] = int(scores[first]) + 1  # pyright: ignore[reportArgumentType]
    code, _ = _run_gate(monkeypatch, tmp_path, bucket_pack, list(POSITIONS), scores)
    assert code == 1, "a real disagreement is a FAILURE (1), distinct from inconclusive (2)"


def test_parity_gate_banks_every_observation_not_only_mismatches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bucket_pack: Path
) -> None:
    """⚑ BANK THE DUMP, NOT JUST THE NUMBER.

    Keeping only the disagreements makes any later re-analysis — re-stratifying,
    estimating on a subset, checking a newer engine build — need a fresh engine
    run, which re-rolls the sample and the engine version along with whatever was
    being changed. The equal rows are the expensive part to reproduce and the
    cheap part to store.
    """
    scores = _agreeing_scores(bucket_pack, list(POSITIONS))
    fens = [*POSITIONS, *IN_CHECK_POSITIONS]
    for fen in IN_CHECK_POSITIONS:
        scores[fen] = "check"

    code, _ = _run_gate(monkeypatch, tmp_path, bucket_pack, fens, scores)
    assert code == 0

    with gzip.open(tmp_path / "obs.jsonl.gz", "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    header = [r for r in rows if r.get("record") == "run"]
    assert len(header) == 1
    assert header[0]["fens_delivered"] == len(fens)

    compared = [r for r in rows if "ours" in r]
    assert len(compared) == len(POSITIONS)
    assert {r["fen"] for r in compared} == set(POSITIONS)
    for row in compared:
        assert row["ours"] == row["sf"] == scores[row["fen"]]

    refused = [r for r in rows if r.get("refused") == "in_check"]
    assert {r["fen"] for r in refused} == set(IN_CHECK_POSITIONS)


# ===========================================================================
# The sampler reports the sample it delivered
# ===========================================================================


def test_pooled_cell_counts_describe_the_fens_actually_returned() -> None:
    """⚑ A COVERAGE TABLE MUST COUNT THE FILE THE GATE READS.

    The pooled sampler dedupes across seeds and truncates to `count`, so adding
    up the sub-samples' own cell counts described a larger, differently
    stratified population than the FENs it returned — a report claiming strata
    the parity gate never saw.
    """
    from scripts.nnue_fens import sample_fens, sample_fens_pooled

    count, seeds, base = 120, 3, 5150
    fens, stats = sample_fens_pooled(count, seeds=seeds, base_seed=base)

    assert len(fens) == len(set(fens)), "pooled sample contains duplicates"
    assert len(fens) <= count
    assert stats.accepted == len(fens)
    assert sum(stats.cell_counts.values()) == len(fens)
    assert set(stats.cell_of) == set(fens)

    # ⚑ NON-VACUITY: what the OLD code reported, reconstructed. Summing the
    # sub-samples' own counts overstates the pooled sample, so the equalities
    # above are not equalities that hold by construction — they hold because the
    # counts are now derived from the delivered list.
    naive_total = sum(
        sum(sample_fens(-(-count // seeds), seed=base + i)[1].cell_counts.values())
        for i in range(seeds)
    )
    assert naive_total > len(fens), (
        "expected dedup/truncation to make the naive sum overstate the sample; "
        "if this ever ties, the test below it proves nothing"
    )


def test_pooled_sampling_is_reproducible_from_its_seed() -> None:
    from scripts.nnue_fens import sample_fens_pooled

    a, _ = sample_fens_pooled(60, seeds=2, base_seed=99)
    b, _ = sample_fens_pooled(60, seeds=2, base_seed=99)
    assert a == b


# ===========================================================================
# Round 2: architecture-hash validation
# ===========================================================================


def _synthetic_small_with(net_hash: int | None = None, layer_hash: int | None = None) -> bytes:
    """The valid synthetic small net, with one hash field overridden."""
    blob = bytearray(build_synthetic_small_nnue()[0])
    if net_hash is not None:
        struct.pack_into("<I", blob, 4, net_hash)
    if layer_hash is not None:
        # Every layer stack's arch hash; find them by their known value.
        old = struct.pack("<I", SMALL.layer_hash)
        new = struct.pack("<I", layer_hash)
        assert blob.count(old) == nnue_parse.LAYER_STACKS
        blob = bytearray(bytes(blob).replace(old, new))
    return bytes(blob)


def test_parser_rejects_stacks_that_agree_on_the_wrong_architecture() -> None:
    """⚑ MUTUAL AGREEMENT IS NOT VALIDATION.

    Eight layer stacks agreeing with each other says only that the file is
    internally consistent. A foreign network whose stacks all carry the same
    unexpected hash describes a topology this parser does not implement — and
    the parser would read it with our hard-coded affine/activation chain and
    convert it into a pack of plausible numbers.
    """
    blob = _synthetic_small_with(layer_hash=0xC0FFEE01)
    with pytest.raises(ValueError, match="is not the 0x6333712A this parser implements"):
        nnue_parse.parse_bytes(blob)


def test_parser_rejects_a_header_hash_that_contradicts_the_file() -> None:
    """`net_hash` is now RELATED to the architecture instead of merely read."""
    blob = _synthetic_small_with(net_hash=0xDEADBEEF)
    with pytest.raises(ValueError, match="does not match this file's own feature"):
        nnue_parse.parse_bytes(blob)


def test_the_expected_net_hash_is_derived_from_the_two_component_hashes() -> None:
    """Measured on both shipped nets: net_hash == ft_hash ^ layer_hash."""
    for arch in nnue_parse.ARCHS:
        assert arch.net_hash == (arch.ft_hash ^ arch.layer_hash) & 0xFFFFFFFF
    assert BIG.layer_hash == 0x63337116
    assert SMALL.layer_hash == 0x6333712A


# ===========================================================================
# Round 2: the weight cache notices a pack replaced at the same path
# ===========================================================================


def _bucket_style_pack(path: Path, scale: int) -> None:
    """A pack whose evaluation is (bucket + 1) * scale // 16."""
    write_synthetic_pack(
        path, {"fc2_bias": [(b, (b + 1) * scale) for b in range(nnue_parse.PSQT_BUCKETS)]}
    )


def _atomic_replace(src: Path, dst: Path) -> None:
    """Publish src over dst the way nnue_pack.convert() does — new inode."""
    os.replace(src, dst)


def test_replacing_a_pack_at_the_same_path_is_observed_by_a_new_load(tmp_path: Path) -> None:
    """⚑ CACHE HITS ARE KEYED ON FILE IDENTITY, NOT ON THE PATHNAME.

    `nnue_pack.convert()` publishes atomically, so a repacked net arrives at the
    SAME pathname as a different file. Keyed by path, a long-lived process keeps
    serving the mapping it already had: `set_value_provider(..., same_path)`
    returns success and the tree goes on evaluating with the previous weights.
    No error, no log line — just the wrong numbers, which is this repo's
    signature defect with the ignored value being a whole network.
    """
    pack = tmp_path / "live.pack"
    _bucket_style_pack(pack, 1600)          # eval == (bucket + 1) * 100
    first = _nnue_ext.load(str(pack))
    board = cboard(POSITIONS[0])
    before = _nnue_ext.evaluate(first, board)

    staged = tmp_path / "staged.pack"
    _bucket_style_pack(staged, 3200)        # eval == (bucket + 1) * 200
    _atomic_replace(staged, pack)

    second = _nnue_ext.load(str(pack))
    after = _nnue_ext.evaluate(second, board)

    assert after == before * 2, (
        "a new load of a replaced pack still returned the old weights' evaluation"
    )
    # The handle taken before the swap keeps its own mapping — that is what makes
    # the replacement safe rather than a use-after-free for existing users.
    assert _nnue_ext.evaluate(first, board) == before


def test_the_tree_seam_sees_a_replaced_pack_too(tmp_path: Path) -> None:
    """The path Codex named: set_value_provider on an already-loaded pathname."""
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    pack = tmp_path / "tree.pack"
    _bucket_style_pack(pack, 1600)
    tree = MCTSTree()
    tree.set_value_provider("nnue", str(pack))
    board = cboard(POSITIONS[0])
    try:
        before = tree.value_provider_eval(board)
        staged = tmp_path / "tree_staged.pack"
        _bucket_style_pack(staged, 3200)
        _atomic_replace(staged, pack)

        tree.set_value_provider("nnue", str(pack))
        assert tree.value_provider_eval(board) == before * 2
    finally:
        tree.clear_value_provider()


def test_the_same_unchanged_pack_is_still_mapped_once(tmp_path: Path) -> None:
    """The positive control: identity keying must not defeat sharing.

    Without this, the replacement test above would pass just as well if the
    cache had been deleted outright.
    """
    pack = tmp_path / "shared_identity.pack"
    _bucket_style_pack(pack, 1600)
    before = _nnue_ext.weight_cache_size()
    a = _nnue_ext.load(str(pack))
    b = _nnue_ext.load(str(pack))
    assert _nnue_ext.weight_cache_size() == before + 1
    board = cboard(POSITIONS[0])
    assert _nnue_ext.evaluate(a, board) == _nnue_ext.evaluate(b, board)


# ===========================================================================
# Round 2: what the evaluator can actually see
# ===========================================================================


@pytest.mark.parametrize(
    ("base", "variant"),
    [
        # halfmove / fullmove clocks
        (
            "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 6 9",
            "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 41 77",
        ),
        # castling rights
        (
            "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 6 9",
            "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w - - 6 9",
        ),
        # en-passant square
        (
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        ),
    ],
    ids=["clocks", "castling", "ep"],
)
def test_ep_castling_and_clocks_do_not_change_the_evaluation(
    base: str, variant: str, dense_pack: Path
) -> None:
    """⚑ ESTABLISHES what `nnue_state_key` is allowed to drop, by MEASURING it.

    The parity sample dedupes on placement + side to move. That is only correct
    if nothing else in a FEN reaches the feature computation, and "I read the
    code and it doesn't" is the kind of claim that silently stops being true.
    Asserted here against the compiled evaluator, on a pack with nonzero feature
    weights so a difference would actually show.
    """
    handle = _nnue_ext.load(str(dense_pack))
    assert _nnue_ext.evaluate(handle, cboard(base)) == _nnue_ext.evaluate(
        handle, cboard(variant)
    )
    for perspective in (0, 1):
        assert _nnue_ext.active_features(cboard(base), perspective) == _nnue_ext.active_features(
            cboard(variant), perspective
        )


def test_placement_or_side_to_move_DOES_change_the_evaluation(dense_pack: Path) -> None:
    """The other half: the key's own fields are not inert either."""
    handle = _nnue_ext.load(str(dense_pack))
    # ⚑ ASYMMETRIC on purpose. The position used in the tests above is a perfect
    # mirror, so flipping the side to move genuinely changes nothing there — the
    # same trap as the POV fixture, and it makes this assertion unfalsifiable
    # rather than false.
    white = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3"
    black = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    assert _nnue_ext.evaluate(handle, cboard(white)) != _nnue_ext.evaluate(
        handle, cboard(black)
    )


def test_sampler_dedupes_on_the_nnue_visible_state() -> None:
    """Positions differing only in clocks are one sample, not several."""
    from scripts.nnue_fens import nnue_state_key, state_key_of_fen

    a = chess.Board("8/5pk1/6p1/8/1P6/5PKP/8/8 w - - 0 40")
    b = chess.Board("8/5pk1/6p1/8/1P6/5PKP/8/8 w - - 17 61")
    assert nnue_state_key(a) == nnue_state_key(b)
    assert state_key_of_fen(a.fen()) == nnue_state_key(a)
    assert nnue_state_key(a) != nnue_state_key(chess.Board("8/5pk1/6p1/8/1P6/5PKP/8/8 b - - 0 40"))


def test_sampled_positions_carry_their_playout_key() -> None:
    """⚑ THE RESAMPLING UNIT SURVIVES INTO THE SAMPLE.

    `_playout()` walks a whole random game, so its positions are correlated. A
    sample that keeps only FENs cannot be re-analysed as clustered data, and the
    repo protocol keys positions by game ID for exactly that reason.
    """
    from scripts.nnue_fens import sample_fens

    fens, stats = sample_fens(80, seed=4242)
    assert fens
    assert set(stats.origin) >= set(fens)
    playouts = {stats.origin[f][0] for f in fens}
    assert len(playouts) > 1, "expected the sample to span several playouts"
    assert all(stats.origin[f][1] >= 0 for f in fens)
    # Positions really do cluster: fewer playouts than positions.
    assert len(playouts) < len(fens)


def test_sample_file_round_trips_the_playout_key(tmp_path: Path) -> None:
    from scripts.nnue_fens import read_sample, sample_fens, write_sample

    fens, stats = sample_fens(40, seed=99)
    path = tmp_path / "sample.tsv"
    write_sample(path, fens, stats)
    back = read_sample(path)

    assert [s.fen for s in back] == fens
    for s in back:
        assert (s.playout, s.ply) == stats.origin[s.fen]

    # A bare FEN file still loads, with no cluster key rather than a fake one.
    plain = tmp_path / "plain.txt"
    plain.write_text("\n".join(fens) + "\n")
    bare = read_sample(plain)
    assert [s.fen for s in bare] == fens
    assert all(s.playout is None and s.ply is None for s in bare)


# ===========================================================================
# Round 2: the observation bank is never destroyed by a failed run
# ===========================================================================


def test_a_failed_run_does_not_truncate_the_previous_bank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bucket_pack: Path
) -> None:
    """⚑⚑ BANK THE DUMP — INCLUDING THE ONE THAT IS ALREADY THERE.

    The destination was opened "wt" before the engine's provenance was checked,
    so a run that then refused (unreadable EvalFile, wrong net) had already
    truncated an expensive previous bank on its way to exiting 2. The failure
    mode is silent and total: the artifact is gone and the run that destroyed it
    reports only that it declined to produce a new one.
    """
    bank = tmp_path / "obs.jsonl.gz"
    with gzip.open(bank, "wt", encoding="utf-8") as fh:
        fh.write('{"record": "previous run"}\n')
    original = bank.read_bytes()

    code, _ = _run_gate(
        monkeypatch, tmp_path, bucket_pack, [POSITIONS[0]],
        eval_file="", extra_args=["--overwrite"],
    )
    assert code == 2
    assert bank.read_bytes() == original, "a refused run destroyed the previous bank"


def test_an_existing_bank_is_not_overwritten_without_the_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bucket_pack: Path
) -> None:
    bank = tmp_path / "obs.jsonl.gz"
    with gzip.open(bank, "wt", encoding="utf-8") as fh:
        fh.write('{"record": "previous run"}\n')
    original = bank.read_bytes()

    code, fake = _run_gate(monkeypatch, tmp_path, bucket_pack, list(POSITIONS))
    assert code == 2
    assert fake.asked == [], "the engine was driven before the bank collision was noticed"
    assert bank.read_bytes() == original


def test_overwrite_publishes_the_new_bank_on_a_successful_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bucket_pack: Path
) -> None:
    """The positive control: --overwrite really does replace it."""
    bank = tmp_path / "obs.jsonl.gz"
    with gzip.open(bank, "wt", encoding="utf-8") as fh:
        fh.write('{"record": "previous run"}\n')

    code, _ = _run_gate(
        monkeypatch, tmp_path, bucket_pack, list(POSITIONS), extra_args=["--overwrite"]
    )
    assert code == 0
    with gzip.open(bank, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    assert not any(r.get("record") == "previous run" for r in rows)
    assert len([r for r in rows if "ours" in r]) == len(POSITIONS)


def test_no_partial_file_is_left_behind_by_a_refused_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bucket_pack: Path
) -> None:
    code, _ = _run_gate(monkeypatch, tmp_path, bucket_pack, [POSITIONS[0]], eval_file="")
    assert code == 2
    leftovers = list(tmp_path.glob("*.partial-*"))
    assert leftovers == [], f"staging file not cleaned up: {leftovers}"


def test_banked_rows_carry_the_resampling_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bucket_pack: Path
) -> None:
    """Every banked row names the playout it came from and its ply within it."""
    import scripts.nnue_parity as parity

    from scripts.nnue_fens import SampledPosition

    fens = [SampledPosition(fen, f"s1p{i // 3}", i) for i, fen in enumerate(POSITIONS)]
    scores = _agreeing_scores(bucket_pack, list(POSITIONS))
    fake = _FakeStockfish(scores, "nn-000000000000.nnue")
    monkeypatch.setattr(parity, "StockfishDriver", lambda _p: fake)

    bank = tmp_path / "cluster.jsonl.gz"
    backend = parity.NativeBackend(bucket_pack)
    with gzip.open(bank, "wt", encoding="utf-8") as fh:
        result = parity.run_parity(backend, fens, fake, None, observations=fh)  # pyright: ignore[reportArgumentType]

    assert result.checked == len(POSITIONS)
    with gzip.open(bank, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    assert len(rows) == len(POSITIONS)
    for row, item in zip(rows, fens, strict=True):
        assert row["playout"] == item.playout
        assert row["ply"] == item.ply
    # The clustering is recoverable from the artifact alone.
    assert len({r["playout"] for r in rows}) < len(rows)


# ===========================================================================
# Round 2: the benchmark rejects an empty sample
# ===========================================================================


def test_benchmark_rejects_an_empty_sample(tmp_path: Path, bucket_pack: Path) -> None:
    """An empty input is an input error, not a ZeroDivisionError in the report."""
    import scripts.nnue_bench as bench

    empty = tmp_path / "empty.txt"
    empty.write_text("")
    code = bench.main(
        ["--pack", str(bucket_pack), "--fens-in", str(empty), "--n", "0", "--repeats", "1"]
    )
    assert code == 2


def test_the_delivered_sample_holds_no_duplicate_evaluator_inputs() -> None:
    """⚑ The sample's own claim, checked on the sample it delivers.

    Deduping on the full FEN lets the same placement+STM through several times
    with different clocks, and every copy increments the delivered count and its
    cell count — so the gate's "N positions covered" counts the same evaluator
    input more than once.
    """
    from scripts.nnue_fens import sample_fens, state_key_of_fen

    # ⚑ 800, not 200. Measured: with the full-FEN key restored, a 200-position
    # sample happens to deliver no colliding pair, so the assertion below would
    # hold with the fix reverted — the test would exist and prove nothing. At
    # 800 the reverted sampler delivers two.
    fens, stats = sample_fens(800, seed=31337)
    states = [state_key_of_fen(f) for f in fens]
    assert len(set(states)) == len(fens), "the delivered sample repeats an evaluator input"
    assert stats.accepted == len(fens)
    # And the sampler saw duplicates and rejected them, so this is not vacuous.
    assert stats.duplicate_states > 0
