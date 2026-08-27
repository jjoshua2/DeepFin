"""Convert a Stockfish ``.nnue`` file into an mmap-able little-endian weight pack.

The ``.nnue`` format stores most tensors as LEB128 varints, which a C loader
would have to decode on every process start (and which cannot be mmap'd at all).
This converter does that decode once, offline, and emits a flat file whose
tensors are 64-byte-aligned raw little-endian arrays in exactly the layout the
evaluator indexes them by — so the runtime loader is an ``mmap`` and a header
check, and the weights are shared read-only across every worker on the box
through the page cache.

⚑ BOTH THE ``.nnue`` AND THE ``.pack`` ARE RUNTIME ARTIFACTS AND ARE NEVER
COMMITTED. The loader takes a path; nothing in the repo hardcodes one.

Getting the weights (no local paths — this works from any Stockfish build)::

    printf 'export_net big.nnue\\nquit\\n' | stockfish
    PYTHONPATH=. python3 scripts/nnue_pack.py big.nnue big.pack

``export_net`` writes the engine's embedded big net back out byte for byte: the
file's SHA-256 prefix reproduces its canonical ``nn-<hex>.nnue`` name, and this
converter records that SHA-256 in the pack header so the parity harness can
prove it is measuring the same network the engine is running. The net can also
be downloaded from ``https://tests.stockfishchess.org/api/nn/<filename>`` for the
filename Stockfish reports as its ``EvalFile`` UCI option default.

Pack layout (version 1, little-endian throughout)::

    0x000  char[8]   magic "CAENNUE1"
           u32       pack_version, nnue_version, net_hash, ft_hash
           u32       l1, l2, l3, psqt_buckets, layer_stacks
           u32       halfka_dims, threat_dims, use_threats
           u32       fc0_outputs, fc0_padded_in
           u32       fc1_outputs, fc1_padded_in, fc2_padded_in, reserved
           u64       total_size
           u64       off_ft_bias, off_ft_weight, off_ft_psqt
           u64       off_threat_weight, off_threat_psqt
           u64       off_fc0_bias, off_fc0_weight
           u64       off_fc1_bias, off_fc1_weight
           u64       off_fc2_bias, off_fc2_weight
           u8[32]    source_sha256
    0x100  tensors, each padded to a 64-byte boundary, in the order above

Tensor shapes (C order, row-major)::

    ft_bias        int16 [l1]
    ft_weight      int16 [halfka_dims][l1]
    ft_psqt        int32 [halfka_dims][psqt_buckets]
    threat_weight  int8  [threat_dims][l1]
    threat_psqt    int32 [threat_dims][psqt_buckets]
    fc0_bias       int32 [layer_stacks][fc0_outputs]
    fc0_weight     int8  [layer_stacks][fc0_outputs][fc0_padded_in]
    fc1_bias       int32 [layer_stacks][fc1_outputs]
    fc1_weight     int8  [layer_stacks][fc1_outputs][fc1_padded_in]
    fc2_bias       int32 [layer_stacks][1]
    fc2_weight     int8  [layer_stacks][1][fc2_padded_in]
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from scripts.nnue_parse import LAYER_STACKS, PSQT_BUCKETS, NnueNet, pad, parse

MAGIC = b"CAENNUE1"
PACK_VERSION = 1
HEADER_BYTES = 256
ALIGN = 64

_TENSOR_ORDER = (
    "ft_bias",
    "ft_weight",
    "ft_psqt",
    "threat_weight",
    "threat_psqt",
    "fc0_bias",
    "fc0_weight",
    "fc1_bias",
    "fc1_weight",
    "fc2_bias",
    "fc2_weight",
)


def _tensors(net: NnueNet) -> dict[str, npt.NDArray[np.generic]]:
    """Every tensor in pack layout, dtypes and shapes already final."""
    if net.threat_weight is None or net.threat_psqt is None:
        raise ValueError(
            f"the pack format carries the big (threat) architecture; got {net.arch.name}"
        )
    stacks = net.stacks
    return {
        "ft_bias": np.ascontiguousarray(net.ft_bias, dtype="<i2"),
        "ft_weight": np.ascontiguousarray(net.ft_weight, dtype="<i2"),
        "ft_psqt": np.ascontiguousarray(net.ft_psqt, dtype="<i4"),
        "threat_weight": np.ascontiguousarray(net.threat_weight, dtype="<i1"),
        "threat_psqt": np.ascontiguousarray(net.threat_psqt, dtype="<i4"),
        "fc0_bias": np.stack([s.fc0_bias for s in stacks]).astype("<i4"),
        "fc0_weight": np.stack([s.fc0_weight for s in stacks]).astype("<i1"),
        "fc1_bias": np.stack([s.fc1_bias for s in stacks]).astype("<i4"),
        "fc1_weight": np.stack([s.fc1_weight for s in stacks]).astype("<i1"),
        "fc2_bias": np.stack([s.fc2_bias for s in stacks]).astype("<i4"),
        "fc2_weight": np.stack([s.fc2_weight for s in stacks]).astype("<i1"),
    }


def _align(offset: int) -> int:
    return (offset + ALIGN - 1) // ALIGN * ALIGN


@dataclass(frozen=True)
class PackLayout:
    """Where every tensor lands, and the header that says so.

    Exposed (rather than kept inside ``build_pack``) so tests can lay out a
    synthetic pack — a sparse file with a handful of poked-in values — without
    restating the byte layout. A second copy of the layout is exactly the kind of
    thing that drifts out of step with the C loader.
    """

    header: bytes
    offsets: dict[str, int]
    sizes: dict[str, int]
    total_size: int


def pack_layout(
    *,
    l1: int,
    l2: int,
    l3: int,
    halfka_dims: int,
    threat_dims: int,
    nnue_version: int,
    net_hash: int,
    ft_hash: int,
    source_sha256: str,
) -> PackLayout:
    """Compute the pack header and tensor offsets for one architecture."""
    sizes = {
        "ft_bias": l1 * 2,
        "ft_weight": halfka_dims * l1 * 2,
        "ft_psqt": halfka_dims * PSQT_BUCKETS * 4,
        "threat_weight": threat_dims * l1,
        "threat_psqt": threat_dims * PSQT_BUCKETS * 4,
        "fc0_bias": LAYER_STACKS * (l2 + 1) * 4,
        "fc0_weight": LAYER_STACKS * (l2 + 1) * pad(l1),
        "fc1_bias": LAYER_STACKS * l3 * 4,
        "fc1_weight": LAYER_STACKS * l3 * pad(l2 * 2),
        "fc2_bias": LAYER_STACKS * 4,
        "fc2_weight": LAYER_STACKS * pad(l3),
    }
    offsets: dict[str, int] = {}
    cursor = HEADER_BYTES
    for name in _TENSOR_ORDER:
        cursor = _align(cursor)
        offsets[name] = cursor
        cursor += sizes[name]
    total_size = _align(cursor)

    header = bytearray(HEADER_BYTES)
    header[0:8] = MAGIC
    struct.pack_into(
        "<18I", header, 8,
        PACK_VERSION, nnue_version, net_hash, ft_hash,
        l1, l2, l3, PSQT_BUCKETS, LAYER_STACKS,
        halfka_dims, threat_dims, 1 if threat_dims else 0,
        l2 + 1, pad(l1), l3, pad(l2 * 2), pad(l3), 0,
    )
    struct.pack_into("<Q", header, 80, total_size)
    struct.pack_into("<11Q", header, 88, *(offsets[name] for name in _TENSOR_ORDER))
    header[176:208] = bytes.fromhex(source_sha256)
    return PackLayout(
        header=bytes(header), offsets=offsets, sizes=sizes, total_size=total_size
    )


def build_pack(net: NnueNet) -> bytes:
    """Serialise a parsed net into the pack byte string."""
    tensors = _tensors(net)
    arch = net.arch
    assert net.threat_weight is not None
    layout = pack_layout(
        l1=arch.l1,
        l2=arch.l2,
        l3=arch.l3,
        halfka_dims=int(net.ft_weight.shape[0]),
        threat_dims=int(net.threat_weight.shape[0]),
        nnue_version=net.version,
        net_hash=net.net_hash,
        ft_hash=net.ft_hash,
        source_sha256=net.source_sha256,
    )
    for name in _TENSOR_ORDER:
        if tensors[name].nbytes != layout.sizes[name]:
            raise ValueError(
                f"tensor {name} is {tensors[name].nbytes} bytes, layout says "
                f"{layout.sizes[name]}"
            )

    out = bytearray(layout.total_size)
    out[0:HEADER_BYTES] = layout.header
    for name in _TENSOR_ORDER:
        raw = tensors[name].tobytes()
        out[layout.offsets[name] : layout.offsets[name] + len(raw)] = raw
    return bytes(out)


def convert(nnue_path: Path, pack_path: Path) -> tuple[NnueNet, int]:
    """Parse ``nnue_path`` and write the pack to ``pack_path``. Returns (net, bytes)."""
    net = parse(nnue_path)
    blob = build_pack(net)
    tmp = pack_path.with_suffix(pack_path.suffix + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(pack_path)
    return net, len(blob)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("nnue", type=Path, help="input .nnue weight file")
    ap.add_argument("pack", type=Path, help="output .pack file (a runtime artifact)")
    args = ap.parse_args(argv)

    net, size = convert(args.nnue, args.pack)
    print(f"source     : {args.nnue}  ({net.source_bytes:,} bytes)")
    print(f"sha256     : {net.source_sha256}")
    print(f"canonical  : nn-{net.source_sha256[:12]}.nnue")
    print(f"arch       : {net.arch.name}  L1={net.arch.l1} L2={net.arch.l2} L3={net.arch.l3}")
    print(f"wrote      : {args.pack}  ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
