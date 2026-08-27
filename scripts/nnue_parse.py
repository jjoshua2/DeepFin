"""Parser for the Stockfish ``.nnue`` weight-file format (version 0x7AF32F20).

Written from the Stockfish sources as a FORMAT SPECIFICATION (``nnue/network.cpp``
header layout, ``nnue_feature_transformer.h``, ``nnue_architecture.h``,
``layers/*.h``) — no Stockfish code is imported, linked or copied. This module
reads the file, extracts every weight tensor, and asserts it lands exactly on
EOF: consuming the file to its last byte is the strongest available proof that
the layout was understood correctly.

Layout (big net, ``UseThreats = true``)::

    HEADER
      u32   version           == 0x7AF32F20
      u32   network hash
      u32   desc_size
      char[desc_size]         description

    FEATURE TRANSFORMER
      u32   ft hash
      leb128 block            biases            [L1]              int16
      raw le                  threatWeights     [60720 * L1]      int8
      leb128 block            weights           [22528 * L1]      int16
      leb128 block            threatPsqtWeights [60720 * 8]       int32
                              psqtWeights       [22528 * 8]       int32  (same block)

    LAYER STACKS  (x8, bucket = (piece_count - 1) / 4)
      u32   arch hash
      fc_0  biases [L2+1] int32 ; weights [(L2+1) * pad(L1)]   int8
      fc_1  biases [L3]   int32 ; weights [L3     * pad(L2*2)] int8
      fc_2  biases [1]    int32 ; weights [1      * pad(L3)]   int8
      (ClippedReLU / SqrClippedReLU carry no parameters)

A leb128 block is ``b"COMPRESSED_LEB128"`` + u32 byte_count + byte_count bytes of
signed LEB128 varints; one block may carry several arrays back to back.

⚑ Weight tensors are returned in FILE order, which is the canonical
``[output][padded_input]`` row-major order. Stockfish permutes them in memory for
its SIMD kernels (``FeatureTransformer::permute_weights``,
``AffineTransform::get_weight_index``); those permutations are identity for the
scalar reference and are deliberately NOT applied here.

Getting a net file: any Stockfish build can write its embedded net out with the
UCI ``export_net`` command, which reproduces the canonical file byte for byte
(its SHA-256 prefix is the ``nn-<hex>.nnue`` filename)::

    printf 'export_net big.nnue\\nquit\\n' | stockfish

Alternatively download ``nn-<hex>.nnue`` from
``https://tests.stockfishchess.org/api/nn/<filename>``.

Usage::

    python3 scripts/nnue_parse.py <file.nnue> [--dump-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

#: The only weight-file version this project accepts. A mismatch is FATAL: a
#: silently-accepted foreign layout would produce plausible-looking evaluations
#: that are wrong everywhere, which is exactly the defect class that poisons a
#: whole corpus without ever raising.
VERSION: Final = 0x7AF32F20

LEB_MAGIC: Final = b"COMPRESSED_LEB128"

# --- architecture constants, read off nnue_architecture.h / the feature sets ---
PS_NB: Final = 11 * 64
HALFKA_DIMS: Final = 64 * PS_NB // 2  # 22528
THREAT_DIMS: Final = 60720
PSQT_BUCKETS: Final = 8
LAYER_STACKS: Final = 8
SIMD_WIDTH: Final = 32  # ceil_to_multiple padding unit (MaxSimdWidth)

#: ``HalfKAv2_hm::HashValue`` and ``FullThreats::HashValue``.
HALFKA_HASH: Final = 0x7F234CB8
THREATS_HASH: Final = 0x8F234CB8


@dataclass(frozen=True)
class ArchSpec:
    """One of the two architectures Stockfish ships."""

    name: str
    l1: int
    l2: int
    l3: int
    use_threats: bool

    #: ``Network::get_hash_value()`` for this architecture's layer stack.
    #:
    #: ⚑ MEASURED FROM THE SHIPPED NETS, NOT DERIVED — and said so rather than
    #: dressed up as a derivation. Stockfish's documented layer-hash algebra
    #: (0xCC03DAE4 per affine, 0x538D24C7 per activation, seeded with
    #: 0xEC42E90D ^ 2*L1) does NOT reproduce these values for either shipped
    #: architecture, so writing that chain out would give a constant that is
    #: wrong in a way whose only symptom is REJECTING VALID FILES. Pinning is
    #: safe here because the file version is pinned too: this parser accepts
    #: only 0x7AF32F20, and a topology change necessarily changes that.
    #:
    #: big:   nn-f68ec79f0fe3.nnue  layer hash 0x63337116
    #: small: nn-47fc8b7fff06.nnue  layer hash 0x6333712A
    layer_hash: int

    @property
    def ft_hash(self) -> int:
        """``FeatureTransformer::get_hash_value()`` for this architecture."""
        base = _combine_hash(THREATS_HASH, HALFKA_HASH) if self.use_threats else HALFKA_HASH
        return (base ^ (self.l1 * 2)) & 0xFFFFFFFF

    @property
    def net_hash(self) -> int:
        """The header hash this architecture must declare.

        ⚑ DERIVED, and verified against both shipped nets: the file's top-level
        hash is exactly ``ft_hash ^ layer_hash``. The parser previously read this
        field and never related it to anything, so a header claiming one
        architecture while carrying another's layers was accepted.
        """
        return (self.ft_hash ^ self.layer_hash) & 0xFFFFFFFF


ARCHS: Final = (
    ArchSpec(name="big", l1=1024, l2=31, l3=32, use_threats=True, layer_hash=0x63337116),
    ArchSpec(name="small", l1=128, l2=15, l3=32, use_threats=False, layer_hash=0x6333712A),
)


def _combine_hash(*hashes: int) -> int:
    """``FeatureTransformer::combine_hash`` — rotate-left-1 then xor, in order."""
    out = 0
    for component in hashes:
        out = ((out << 1) | (out >> 31)) & 0xFFFFFFFF
        out ^= component
    return out & 0xFFFFFFFF


def pad(n: int, mult: int = SIMD_WIDTH) -> int:
    """``ceil_to_multiple`` — the input padding every affine layer is stored with."""
    return ((n + mult - 1) // mult) * mult


@dataclass(frozen=True)
class LayerStack:
    """One of the eight per-bucket layer stacks."""

    arch_hash: int
    fc0_bias: npt.NDArray[np.int32]
    fc0_weight: npt.NDArray[np.int8]
    fc1_bias: npt.NDArray[np.int32]
    fc1_weight: npt.NDArray[np.int8]
    fc2_bias: npt.NDArray[np.int32]
    fc2_weight: npt.NDArray[np.int8]


@dataclass(frozen=True)
class NnueNet:
    """A fully parsed Stockfish network."""

    version: int
    net_hash: int
    description: str
    arch: ArchSpec
    ft_hash: int
    ft_bias: npt.NDArray[np.int16]
    ft_weight: npt.NDArray[np.int16]
    ft_psqt: npt.NDArray[np.int32]
    threat_weight: npt.NDArray[np.int8] | None
    threat_psqt: npt.NDArray[np.int32] | None
    stacks: tuple[LayerStack, ...]
    source_sha256: str
    source_bytes: int


class Reader:
    """A cursor over the weight file."""

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0

    def u32(self) -> int:
        value: int = struct.unpack_from("<I", self.buf, self.pos)[0]
        self.pos += 4
        return value

    def raw(self, n: int) -> bytes:
        chunk = self.buf[self.pos : self.pos + n]
        if len(chunk) != n:
            raise EOFError(f"wanted {n} bytes at {self.pos}, got {len(chunk)}")
        self.pos += n
        return chunk

    def raw_array(self, count: int, dtype: str) -> npt.NDArray[np.generic]:
        itemsize = np.dtype(dtype).itemsize
        return np.frombuffer(self.raw(count * itemsize), dtype=dtype)

    def leb128_block(self, *counts: int) -> list[npt.NDArray[np.int64]]:
        """Decode one COMPRESSED_LEB128 block holding ``len(counts)`` arrays."""
        magic = self.raw(len(LEB_MAGIC))
        if magic != LEB_MAGIC:
            raise ValueError(f"bad LEB128 magic at {self.pos - len(LEB_MAGIC)}: {magic!r}")
        byte_count = self.u32()
        payload = self.raw(byte_count)
        values = decode_leb128(payload, sum(counts))
        out: list[npt.NDArray[np.int64]] = []
        off = 0
        for count in counts:
            out.append(values[off : off + count])
            off += count
        return out


def decode_leb128(payload: bytes, total: int) -> npt.NDArray[np.int64]:
    """Signed LEB128, vectorised. Mirrors ``read_leb_128_detail`` exactly.

    ⚑ Two details of the reference decoder are load-bearing and are easy to
    "clean up" into something that decodes the shipped nets correctly and a
    hypothetical wider one wrongly:

    * the accumulator is a **32-bit register**, so a value whose varint carries
      bits past bit 31 WRAPS rather than growing;
    * the per-byte shift is ``shift % 32``, so the sixth and later bytes of a
      varint fold back over the low bits instead of shifting off the top.

    Together they mean a 5-byte varint like ``-2**30`` decodes only if the
    accumulation is done in 32 bits: in arbitrary precision it comes out as
    ``33285996544``. No tensor in the shipped nets needs more than five bytes,
    so this never fires in production — which is exactly why it has to be pinned
    by a test rather than by observation.
    """
    raw = np.frombuffer(payload, dtype=np.uint8)
    cont = (raw & 0x80) != 0
    # Each value ends at the first byte without the continuation bit.
    ends = np.flatnonzero(~cont)
    # ⚑ Exact, not "at least". Too FEW values is an obvious truncation; too MANY
    # means the block boundary we computed is wrong, and silently decoding the
    # first `total` of them yields a full-size tensor of plausible weights that
    # happens to start in the wrong place. Both directions are corruption.
    if ends.size != total:
        raise ValueError(f"LEB128 stream holds {ends.size} values, expected exactly {total}")
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1

    acc = np.zeros(total, dtype=np.uint64)  # a 32-bit register, held in uint64
    lengths = ends - starts + 1
    maxlen = int(lengths.max())
    payload7 = (raw & 0x7F).astype(np.uint64)
    for k in range(maxlen):
        idx = starts[:total] + k
        active = k < lengths[:total]
        safe = np.where(active, idx, 0)
        contribution = (payload7[safe] << np.uint64((7 * k) % 32)) & np.uint64(0xFFFFFFFF)
        acc |= np.where(active, contribution, np.uint64(0))
        acc &= np.uint64(0xFFFFFFFF)

    # Sign-extend iff the 0x40 bit of the LAST byte is set and the varint did not
    # already fill 32 bits.
    shift = 7 * lengths[:total]
    last = raw[ends[:total]]
    neg = ((last & 0x40) != 0) & (shift < 32)
    fill = (np.uint64(0xFFFFFFFF) << shift[neg].astype(np.uint64)) & np.uint64(0xFFFFFFFF)
    acc[neg] |= fill

    signed = acc.astype(np.int64)
    signed[signed >= 0x80000000] -= 0x100000000
    return signed


def _parse_layer_stacks(reader: Reader, arch: ArchSpec) -> tuple[LayerStack, ...]:
    fc0_out = arch.l2 + 1
    fc1_in = arch.l2 * 2
    stacks: list[LayerStack] = []
    for _ in range(LAYER_STACKS):
        arch_hash = reader.u32()
        fc0_b = reader.raw_array(fc0_out, "<i4")
        fc0_w = reader.raw_array(fc0_out * pad(arch.l1), "<i1").reshape(fc0_out, pad(arch.l1))
        fc1_b = reader.raw_array(arch.l3, "<i4")
        fc1_w = reader.raw_array(arch.l3 * pad(fc1_in), "<i1").reshape(arch.l3, pad(fc1_in))
        fc2_b = reader.raw_array(1, "<i4")
        fc2_w = reader.raw_array(pad(arch.l3), "<i1").reshape(1, pad(arch.l3))
        stacks.append(
            LayerStack(
                arch_hash=arch_hash,
                fc0_bias=fc0_b.astype(np.int32),
                fc0_weight=fc0_w.astype(np.int8),
                fc1_bias=fc1_b.astype(np.int32),
                fc1_weight=fc1_w.astype(np.int8),
                fc2_bias=fc2_b.astype(np.int32),
                fc2_weight=fc2_w.astype(np.int8),
            )
        )
    return tuple(stacks)


def _arch_for_ft_hash(ft_hash: int) -> ArchSpec:
    for arch in ARCHS:
        if arch.ft_hash == ft_hash:
            return arch
    known = ", ".join(f"{a.name}={a.ft_hash:#010x}" for a in ARCHS)
    raise ValueError(
        f"feature-transformer hash {ft_hash:#010x} matches no known architecture ({known})"
    )


def parse_bytes(data: bytes) -> NnueNet:
    """Parse an in-memory ``.nnue`` file. Raises unless it lands exactly on EOF."""
    reader = Reader(data)

    version = reader.u32()
    if version != VERSION:
        raise ValueError(
            f"unsupported .nnue version {version:#010x}; this project only accepts "
            f"{VERSION:#010x}"
        )
    net_hash = reader.u32()
    desc_size = reader.u32()
    description = reader.raw(desc_size).decode("utf-8", "replace")

    ft_hash = reader.u32()
    arch = _arch_for_ft_hash(ft_hash)

    (bias,) = reader.leb128_block(arch.l1)
    threat_weight: npt.NDArray[np.int8] | None = None
    threat_psqt: npt.NDArray[np.int32] | None = None
    if arch.use_threats:
        threat_weight = (
            reader.raw_array(THREAT_DIMS * arch.l1, "<i1")
            .reshape(THREAT_DIMS, arch.l1)
            .astype(np.int8)
        )
        (weight,) = reader.leb128_block(HALFKA_DIMS * arch.l1)
        threat_psqt_flat, psqt_flat = reader.leb128_block(
            THREAT_DIMS * PSQT_BUCKETS, HALFKA_DIMS * PSQT_BUCKETS
        )
        threat_psqt = threat_psqt_flat.astype(np.int32).reshape(THREAT_DIMS, PSQT_BUCKETS)
    else:
        (weight,) = reader.leb128_block(HALFKA_DIMS * arch.l1)
        (psqt_flat,) = reader.leb128_block(HALFKA_DIMS * PSQT_BUCKETS)

    stacks = _parse_layer_stacks(reader, arch)

    leftover = len(data) - reader.pos
    if leftover != 0:
        raise ValueError(f"parse did not land on EOF: {leftover} bytes left over")

    # ⚑ MUTUAL AGREEMENT IS NOT VALIDATION. Eight stacks that agree with each
    # other but not with the architecture we selected describe a network this
    # parser does not implement — and it would go on to read them with our
    # hard-coded affine/activation topology and emit a pack full of plausible
    # numbers. The expected hash is what makes this a check rather than a
    # consistency observation.
    stack_hashes = {stack.arch_hash for stack in stacks}
    if len(stack_hashes) != 1:
        raise ValueError(f"layer stacks disagree on architecture hash: {sorted(stack_hashes)}")
    got_layer_hash = stack_hashes.pop()
    if got_layer_hash != arch.layer_hash:
        raise ValueError(
            f"layer-stack architecture hash 0x{got_layer_hash:08X} is not the "
            f"0x{arch.layer_hash:08X} this parser implements for the {arch.name} "
            "architecture; refusing to reinterpret a foreign layer topology"
        )
    if net_hash != arch.net_hash:
        raise ValueError(
            f"network hash 0x{net_hash:08X} does not match this file's own feature "
            f"transformer and layers (0x{arch.ft_hash:08X} ^ 0x{arch.layer_hash:08X} = "
            f"0x{arch.net_hash:08X})"
        )

    return NnueNet(
        version=version,
        net_hash=net_hash,
        description=description,
        arch=arch,
        ft_hash=ft_hash,
        ft_bias=bias.astype(np.int16),
        ft_weight=weight.astype(np.int16).reshape(HALFKA_DIMS, arch.l1),
        ft_psqt=psqt_flat.astype(np.int32).reshape(HALFKA_DIMS, PSQT_BUCKETS),
        threat_weight=threat_weight,
        threat_psqt=threat_psqt,
        stacks=stacks,
        source_sha256=hashlib.sha256(data).hexdigest(),
        source_bytes=len(data),
    )


def parse(path: Path | str) -> NnueNet:
    """Parse a ``.nnue`` file from disk."""
    return parse_bytes(Path(path).read_bytes())


def describe(net: NnueNet, name: str) -> str:
    """A human-readable summary, for the CLI and for provenance logging."""
    lines = [
        f"file            : {name}  ({net.source_bytes:,} bytes)",
        f"sha256          : {net.source_sha256}",
        f"version         : {net.version:#010x}  OK",
        f"network hash    : {net.net_hash:#010x}",
        f"description     : {net.description.strip()!r}",
        f"architecture    : {net.arch.name}  L1={net.arch.l1} L2={net.arch.l2} "
        f"L3={net.arch.l3} threats={net.arch.use_threats}",
        f"ft hash         : {net.ft_hash:#010x}  (derived, matches {net.arch.name})",
        f"layer-stack hash: {net.stacks[0].arch_hash:#010x}  (all {LAYER_STACKS} agree)",
        "tensors (file order, pre-SIMD-permutation):",
    ]
    tensors: list[tuple[str, npt.NDArray[np.generic] | None]] = [
        ("ft_bias", net.ft_bias),
        ("ft_weight", net.ft_weight),
        ("ft_psqt", net.ft_psqt),
        ("threat_weight", net.threat_weight),
        ("threat_psqt", net.threat_psqt),
        ("stack0.fc0_bias", net.stacks[0].fc0_bias),
        ("stack0.fc0_weight", net.stacks[0].fc0_weight),
        ("stack0.fc1_bias", net.stacks[0].fc1_bias),
        ("stack0.fc1_weight", net.stacks[0].fc1_weight),
        ("stack0.fc2_bias", net.stacks[0].fc2_bias),
        ("stack0.fc2_weight", net.stacks[0].fc2_weight),
    ]
    for label, tensor in tensors:
        if tensor is None:
            lines.append(f"  {label:<18} (absent)")
            continue
        lines.append(
            f"  {label:<18} {tensor.dtype!s:<6} {tensor.shape!s:<16} "
            f"min={tensor.min():>8} max={tensor.max():>8}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="path to a .nnue weight file")
    ap.add_argument("--dump-dir", help="also write each tensor as a .npy into this directory")
    args = ap.parse_args(argv)

    path = Path(args.path)
    net = parse(path)
    print(describe(net, path.name))

    if args.dump_dir:
        out = Path(args.dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        dump: dict[str, npt.NDArray[np.generic] | None] = {
            "ft_bias": net.ft_bias,
            "ft_weight": net.ft_weight,
            "ft_psqt": net.ft_psqt,
            "threat_weight": net.threat_weight,
            "threat_psqt": net.threat_psqt,
        }
        for key, tensor in dump.items():
            if tensor is not None:
                np.save(out / f"{key}.npy", tensor)
        print(f"dumped feature-transformer tensors to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
