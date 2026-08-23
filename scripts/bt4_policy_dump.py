"""Batched BT4 (LC0) policy dump, as JSONL, from FENs or from replay shards.

For each input position this runs the BT4 ONNX policy head, gathers the logits
of the LEGAL moves through the board-aware LC0-1858 mapping
(:mod:`chess_anti_engine.moves.leela_index`), softmaxes over the legal moves
ONLY, and writes one record per position.

Two input sources, and they are NOT equivalent:

``--input-source fens`` (default)
    Positions come from a FEN row list. The 112-plane LC0 input is built from
    the FEN alone, with the 7 history frames filled by repeating the current
    position (``fill_lc0_history_repeat``).
    ⚑ HISTORY-LESS — A KNOWN, MEASURED DEGRADATION. BT4 was trained on real
    move history. Measured against lc0's own v6 training records, n=1000,
    by::

        PYTHONPATH=. python3 scripts/bt4_history_sensitivity.py \
            --tar data/lc0_training/training-run2-test91-20260810-1417.tar \
            --positions 1000 --validate 400 --roundtrip 400 --threads 16

    which reported, for the ``repeat_norep`` arm (the faithful emulation of
    what this path feeds): top-1 flip rate **0.0610 (61/1000)**, median
    KL(true‖fen-only) **0.0043** nats, mean 0.0197, Spearman(log p) 0.9728.
    Re-run the command to reproduce; those numbers are from that n, not a
    banked artifact in the tree. En passant is conveyed through the history
    frames rather than a plane, so a FEN-built input also cannot show BT4 the
    double-push behind an en-passant capture. This is the same convention every
    prior BT4 instrument here used (``scripts/foreign_net_audit.py``,
    ``scripts/lc0_adapter_probe.py``, the banked ``bt4_audit_cache``), so those
    numbers stay comparable.

``--input-source shards:<dir>[,<dir>]``
    Positions come from stored replay ``x`` tensors, whose first 112 planes are
    the LC0 block and carry TRUE 8-slot history. No FEN rebuild happens, so the
    degradation above does not apply. Prefer this whenever the rows exist.

CONSUMER CONTRACT — read before joining a dump to anything
----------------------------------------------------------
(a) ⚑⚑ THE TWO MODES EMIT DIFFERENT KEYS AND DIFFERENT COORDINATE FRAMES. This
    is deliberate and is the single most important thing to get right before
    joining.

    ``shards`` ⇒ **RIG KEY + CANONICAL frame.** Replay planes are stored
    POV-oriented, so a decoded row is ALWAYS White-to-move. ``key`` is the
    32-hex PLANE FINGERPRINT — the very digest
    :func:`chess_anti_engine.eval.rvg_surgery.position_fingerprints` computes,
    imported and called here rather than re-derived, because that function IS
    the R/V/G rig's join key and a second copy of it would drift and silently
    unjoin every row. The mirrored, side-to-move-canonical FEN is still carried,
    in the ``fen`` field, for humans and for debugging — it is NOT the join key.
    The ``policy`` keys are canonical UCIs: for a black-to-move row a kingside
    castle is ``e1g1``, NEVER ``e8g8``, and a black pawn push reads ``e2e4``,
    not ``e7e5``.
    ``fens`` ⇒ **TRUE frame.** ``key`` is the caller's own FEN, verbatim (there
    is no ``fen`` field), and the UCIs are in real board coordinates.

    So the same position dumped through the two modes does NOT produce the same
    record, and a consumer holding true-frame FENs gets an EMPTY join against a
    shard dump. This is not a bug to "fix" by flipping the shard path: the
    downstream rig builds its labels on the same white-to-move canonical board
    reconstructed from the same planes, so canonical↔canonical is the coherent
    join and a true-frame shard path would BREAK it. The frame is recorded per
    dump in the header's ``contract.frame``, and
    ``tests/test_bt4_policy_dump.py`` pins it end to end.
    To join a true-frame position against a shard dump, decode it through
    :func:`board_from_stored_x` (or mirror the board yourself when Black is to
    move) so both sides speak the canonical frame.

(a2) ⚑ The ``fen`` field is IDENTITY-BY-PLANES, not a true game state. It is
    rebuilt from ``chess.Board(None)``, so its fullmove number is always 1, and
    its halfmove clock saturates at 100 because rule50 arrives through a plane.
    Never parse a move number out of it, and never join on it — the fingerprint
    in ``key`` is the join, and it hashes the PLANES (the clock at full 0..150
    resolution included), not this string.

(a3) **A SHARD DUMP IS DIRECTLY LOADABLE BY THE RIG.** The header line is a
    ``{"record": "provenance", "v": 1, "policy_encoding": "lc0_1858", ...}``
    record, which is what
    :meth:`chess_anti_engine.eval.rvg_surgery.RvgExternalPolicyIndex.load`
    requires — a missing or wrong ``v`` is a REFUSAL there, not a warning, and
    so are duplicate keys. ``--restrict-keys <file>`` (shard mode only) narrows
    a dump to an enumerated key set: one 32-hex key per line, lines beginning
    with ``{`` skipped so a label file's own provenance header can be fed
    straight back in. The restrict file's path/size/mtime and key count go in
    the header.
(b) ⚑ The gather index space is LC0's 1858 ORDER, which is not ours. Measured:
    ``compact_index_for_move`` and ``leela_index_for_move`` agree on only
    88 of 5,243 moves. **UCI STRINGS ARE THE INTERCHANGE FORMAT — never join on
    a policy index.** The dict is keyed by UCI precisely so no index crosses
    the boundary.
(c) ⚑ Transpositions are FIRST-WINS SKIP: a key already emitted is not emitted
    again. Measured on 113k corpus rows: 20 keys / 40 rows (0.035%), 3 of them
    reached with different history. Because every record carries ``shard`` and
    ``row`` (plus ``game_id`` / ``ply_index``), which occurrence was kept is
    observable, and a consumer that cares can re-derive the others instead of
    silently inheriting our choice. In shard mode the dedup key is the emitted
    FINGERPRINT, which is what the rig loader refuses duplicates of — deduping
    on anything else (the FEN, say) would let two rows that share a fingerprint
    both through and make the file unloadable.

⚑ The gather is board-aware for a reason. LC0's 1858 head spells castling
king-takes-rook (``e1h1``) and ALSO has an ordinary ``e1g1`` slide slot, and it
puts KNIGHT promotion (not queen) on the bare 7th->8th slot. A static
from/to-string remap reads a real but unrelated logit for both families rather
than failing, which is how castling priors came out 49x-120x too small. The
header record's ``remap`` block pins the exact code that produced a dump.

Example
-------
    PYTHONPATH=. python3 scripts/bt4_policy_dump.py \
        --input-source shards:data/salvage/<label>/seeds/slot_000/replay_shards \
        --out data/lc0/bt4_policy_dump.jsonl --batch-size 128 --threads 16 \
        --restrict-keys data/rvg/labels_keys.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import (
    LC0_FULL,
    fill_lc0_history_repeat,
    normalize_lc0_history_encoding,
    x_to_lc0_planes,
)
from chess_anti_engine.encoding.plane_decode import (
    decode_ep_square,
    decode_step0_bitboards,
)
from chess_anti_engine.eval.rvg_surgery import (
    FINGERPRINT_BYTES,
    RVG_EXTERNAL_POLICY_SCHEMA_VERSION,
    UNFINGERPRINTABLE,
    file_provenance,
    position_fingerprints,
)
from chess_anti_engine.moves.encode import (
    COMPACT_POLICY_SIZE,
    POLICY_ENCODING_LC0_1858,
)
from chess_anti_engine.moves.leela_index import (
    compact_index_for_move,
    leela_index_for_move,
)

DEFAULT_ONNX = "data/lc0/onnx/BT4-it332-vanilla-winner.onnx"
# The 112-plane LC0-canonical layout used for FEN-built inputs.
HISTORY_ENCODING = "lc0_root"
# Layouts whose planes 0..103 ARE LC0's interleaved history block, so
# `x_to_lc0_planes` can convert them. `legacy` APPENDS its repetition planes
# instead, which puts a repetition plane where LC0 reads rule50 -- a shard read
# under the wrong name yields a valid tensor and a silently wrong rule50.
SUPPORTED_SHARD_ENCODINGS = ("lc0_root_legacy_meta", "lc0_root")
# Files whose content defines the remap; pinned per-dump so a cache can never
# be silently attributed to the wrong revision of it.
REMAP_SOURCES = (
    "chess_anti_engine/moves/leela_index.py",
    "chess_anti_engine/moves/lc0_1858_movestrs.py",
    "chess_anti_engine/encoding/lc0.py",
)


@dataclass
class Row:
    """One position on its way to the net, with the identity it will be dumped under.

    ``key`` is what the dump is KEYED BY, deduped on and resumed off: the 32-hex
    plane fingerprint in shard mode, the caller's own FEN string in FEN mode.
    ``fen`` is set on the shard path only and is descriptive, never a join key.
    """

    key: str
    board: chess.Board
    fen: str | None = None
    #: Ready-made LC0 planes (shard path, TRUE history) or ``None`` to build
    #: them from the FEN (history-less path).
    planes: np.ndarray | None = None
    shard: str | None = None
    row_index: int | None = None
    game_id: int | None = None
    ply_index: int | None = None
    legal_mask: np.ndarray | None = None

    def identity(self) -> dict[str, Any]:
        """The join fields, omitted entirely for FEN input where they are absent."""
        if self.shard is None:
            return {}
        return {
            "fen": self.fen,
            "shard": self.shard,
            "row": self.row_index,
            "game_id": self.game_id,
            "ply_index": self.ply_index,
        }


# --------------------------------------------------------------- input rows
def iter_fen_rows(path: Path) -> Iterator[Row]:
    """Rows from a JSONL row list (``key``/``fen`` field) or one FEN per line.

    ⚑ The audit set's 4-field ``key`` has NO halfmove clock, so a board built
    from it carries rule50 = 0 — and rule50 is plane 109 of the LC0 input, a
    plane BT4 reads. Measured on 500 audit rows: keying off ``key`` instead of
    ``fen`` flips BT4's top-1 on 5 of them (1.0%), every one a position with a
    large halfmove clock (23, 26, 32, 89, 92) where BT4 correctly plays
    differently near the 50-move boundary. So the FULL fen builds the board
    whenever the row carries one, while the output stays keyed by ``key`` for
    joinability with the audit set. On rows where both describe the same clock
    this script and ``scripts/foreign_net_audit.py`` agree exactly.
    """
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("{"):
                rec = json.loads(line)
                key = rec.get("key") or rec.get("fen")
                if key is None:
                    raise ValueError(f"row has neither 'key' nor 'fen': {line[:120]}")
                yield Row(key=str(key), board=chess.Board(str(rec.get("fen") or key)))
            else:
                yield Row(key=line, board=chess.Board(line))


def shard_castling_rights(planes: np.ndarray, board: chess.Board) -> int:
    """Castling rights from the LC0 metadata planes, for a decoded us=White board.

    Plane order is ``(us_Q, us_K, them_Q, them_K)`` -- LC0's
    ``us_ooo, us_oo, them_ooo, them_oo`` -- verified directly against
    ``encode_lc0_full_root``'s writer, not read off its docstring. The decoded
    board is always us=White, so "us" maps to the first rank.
    """
    base = LC0_FULL.root_metadata_base
    rights = 0
    if planes[base + 0][0, 0] > 0.5:
        rights |= chess.BB_A1
    if planes[base + 1][0, 0] > 0.5:
        rights |= chess.BB_H1
    if planes[base + 2][0, 0] > 0.5:
        rights |= chess.BB_A8
    if planes[base + 3][0, 0] > 0.5:
        rights |= chess.BB_H8
    return rights & board.rooks


def board_from_stored_x(
    x_row: np.ndarray, planes: np.ndarray, *, input_history_encoding: str,
) -> chess.Board:
    """Rebuild the position from one stored replay row.

    Replay shards hold encoded planes, NOT FENs, so the position has to come
    back out of the tensor. Decoding is side-to-move canonical: the result is
    always White to move, which is the frame the planes were written in and the
    frame BT4's policy head speaks. ``input_history_encoding`` must be the
    shard's OWN declared layout -- the en-passant square lives in a different
    place under each one.
    """
    bbs = decode_step0_bitboards(x_row[None])[0]
    board = chess.Board(None)
    for color, offset in ((chess.WHITE, 0), (chess.BLACK, 6)):
        for pt_idx in range(6):
            piece = chess.Piece(pt_idx + 1, color)  # PAWN..KING == 1..6
            for sq in chess.scan_forward(int(bbs[offset + pt_idx])):
                board.set_piece_at(sq, piece)
    board.turn = chess.WHITE
    ep = decode_ep_square(x_row, input_history_encoding)
    board.ep_square = ep if ep >= 0 else None
    board.castling_rights = shard_castling_rights(planes, board)
    board.halfmove_clock = int(
        min(100.0, float(planes[LC0_FULL.root_metadata_base + 5][0, 0])),
    )
    return board


def shard_encoding(group: Any, shard: Path) -> str:
    """The shard's OWN declared input layout, or a loud failure.

    ⚑ Never assume this. Every shard records ``input_history_encoding``, and
    reading one layout as another fails SILENTLY: an ``lc0_root`` shard read as
    ``lc0_root_legacy_meta`` has its raw rule50 multiplied by 100 (clamped to
    100, i.e. "50-move rule about to fire" on every row), and a ``legacy``
    shard has a REPETITION plane sitting where LC0 reads rule50, so every row
    reports rule50 = 0. ``--check-legal-mask`` cannot catch either, because the
    step-0 piece planes are identical across all three layouts.
    """
    declared = group.attrs.get("input_history_encoding")
    if declared is None:
        raise SystemExit(
            f"{shard} has no 'input_history_encoding' attribute; refusing to "
            f"guess a layout. Expected one of {list(SUPPORTED_SHARD_ENCODINGS)}.",
        )
    name = str(declared)
    try:
        canonical = normalize_lc0_history_encoding(name)
    except ValueError as exc:
        raise SystemExit(f"{shard}: unknown input_history_encoding {name!r}") from exc
    if canonical not in SUPPORTED_SHARD_ENCODINGS:
        raise SystemExit(
            f"{shard}: input_history_encoding {name!r} (canonical {canonical!r}) is "
            f"not an LC0-root layout. Supported: {list(SUPPORTED_SHARD_ENCODINGS)}. "
            "Converting it would produce a plausible tensor with a wrong rule50.",
        )
    return canonical


def iter_shard_rows(dirs: Sequence[str]) -> Iterator[Row]:
    """Stream ``Row``s from replay shards, one shard resident at a time.

    ⚑ THE KEY COMES FROM :func:`position_fingerprints`, IMPORTED. That function
    is the R/V/G rig's join key and the label pass's row key; a local
    re-derivation of "the same" digest is the one change that would unjoin every
    row in a 31-hour dump without raising anything.

    ⚑ Each row's planes are ``.copy()``d out of the shard's converted block.
    Handing out a VIEW would make one retained row pin the whole
    ``(2000, 112, 8, 8)`` float32 array (~460 MB per 12k rows retained), which
    is what turned a 3.1M-row corpus dump into a ~121 GB resident set.

    ⚑ "Streams" means the PLANES stream; it is not O(1) overall. The ``seen``
    and ``done`` key sets in :func:`source_rows` and :func:`load_done_keys`
    still grow with distinct keys — measured at ~0.60 KB/row back when the key
    was a FEN, i.e. about **1.9 GB at 3.1M rows, ~3.7 GB for a resumed run**
    that also holds ``done``. Those figures now bound rather than describe it:
    a 32-hex key is a smaller string than a FEN (81 vs ~99 bytes) in the same
    set. ``--restrict-keys`` adds a third set of the same shape, held for the
    whole run. That is the price of first-wins dedup and resumability; budget
    for it, or shard the run by directory and merge.
    """
    import zarr

    for directory in dirs:
        for shard in sorted(Path(directory).glob("*.zarr")):
            group = zarr.open(str(shard), mode="r")
            encoding = shard_encoding(group, shard)
            missing = [
                name for name in ("x", "legal_mask", "game_id", "ply_index")
                if name not in group
            ]
            if missing:
                raise SystemExit(
                    f"{shard} is missing {missing}; the arm-B join needs "
                    "game_id/ply_index on every record. Re-export the shard or "
                    "point --input-source at a directory that has them.",
                )
            xs = np.asarray(group["x"][:])
            masks = np.asarray(group["legal_mask"][:])
            game_ids = np.asarray(group["game_id"][:])
            plies = np.asarray(group["ply_index"][:])
            # The rig's key, off the SAME planes the label pass hashes -- note
            # this reads the STORED `xs`, not the LC0-converted block, because
            # that is what `position_fingerprints` is defined over. Taken BEFORE
            # the conversion so a shard too narrow to fingerprint is named by
            # that fact rather than by `x_to_lc0_planes`'s 112-plane complaint.
            keys = position_fingerprints(xs, input_history_encoding=encoding)
            # ⚑ REFUSED, NOT SKIPPED, and it is a whole-shard property (the
            # sentinel is returned for the entire batch or for none of it). An
            # unfingerprintable row cannot join, and a dump that exists only to
            # be joined must not quietly emit rows keyed by the empty string --
            # they would collide with each other and the rig would refuse the
            # whole file for "duplicate key ''", hours later and far from here.
            if keys and keys[0] == UNFINGERPRINTABLE:
                raise SystemExit(
                    f"{shard}: x is {xs.shape[1]} planes, too narrow to carry the "
                    f"metadata position_fingerprints hashes under "
                    f"{encoding!r}; this shard cannot produce a rig join key.",
                )
            planes_all = x_to_lc0_planes(xs, input_history_encoding=encoding)
            for row in range(xs.shape[0]):
                key = keys[row]
                board = board_from_stored_x(
                    xs[row], planes_all[row], input_history_encoding=encoding,
                )
                yield Row(
                    key=key.hex(),
                    board=board,
                    fen=board.fen(),
                    planes=planes_all[row].copy(),
                    shard=shard.name,
                    row_index=row,
                    game_id=int(game_ids[row]),
                    ply_index=int(plies[row]),
                    legal_mask=masks[row],
                )
            del xs, planes_all, masks


def batched(rows: Iterable[Row], size: int) -> Iterator[list[Row]]:
    """Fixed-size batches from a stream, so nothing accumulates across batches."""
    batch: list[Row] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_restrict_keys(path: str | Path) -> set[str]:
    """The enumerated key set for ``--restrict-keys``, as canonical lowercase hex.

    Tolerant in exactly one way, matching ``scripts/rvg_label_pass.py``'s
    ``--restrict-to``: a line starting with ``{`` is a JSONL provenance header
    and is skipped, so a label file can be fed back in whole. Everything else
    must be a real ``FINGERPRINT_BYTES``-wide key.

    ⚑ A wrong-width or non-hex key is a REFUSAL, not a skip. Dropping it would
    leave a restriction that quietly matches fewer rows than the caller
    enumerated, and "the dump came out short" is not a symptom anyone traces
    back to a typo in the key file.
    """
    wanted: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("{"):
                continue
            try:
                blob = bytes.fromhex(line)
            except ValueError as exc:
                raise SystemExit(
                    f"{path}:{lineno}: {line[:80]!r} is not a hex key ({exc})",
                ) from exc
            if len(blob) != FINGERPRINT_BYTES:
                raise SystemExit(
                    f"{path}:{lineno}: key {line[:80]!r} is {len(blob)} bytes, but "
                    f"the join key is {FINGERPRINT_BYTES} ({FINGERPRINT_BYTES * 2} "
                    "hex chars); this file does not hold rig keys.",
                )
            wanted.add(blob.hex())
    return wanted


def source_rows(
    args: argparse.Namespace, done: set[str], stats: dict[str, int] | None = None,
    restrict: set[str] | None = None,
) -> Iterator[Row]:
    """The de-duplicated, limited row stream for whichever input source is set.

    ``stats`` accumulates ``seen_rows`` / ``distinct_keys`` / ``collisions`` so
    the first-wins transposition skip is COUNTED rather than silent.

    ``restrict`` is the ``--restrict-keys`` set (hex keys), applied FIRST so a
    row the caller never asked for is reported as ``restrict_skipped`` rather
    than as ``resume_skipped`` or ``terminal_skipped``. ⚑ It is NOT applied
    first in order to protect ``collisions``: ``seen`` only ever receives keys
    that were KEPT, so a restricted-away row cannot reach that counter from
    either position — a mutation pass that moved the clause below the dedup
    check changed no observable, and the docstring used to claim otherwise.
    """
    counts = stats if stats is not None else {}
    for field in ("seen_rows", "distinct_keys", "collisions", "resume_skipped",
                  "terminal_skipped", "restrict_skipped"):
        counts.setdefault(field, 0)
    shard_dirs = shard_source_dirs(args.input_source)
    raw: Iterator[Row] = (
        iter_shard_rows(shard_dirs) if shard_dirs else iter_fen_rows(Path(args.rows))
    )
    seen: set[str] = set()
    emitted = 0
    for row in raw:
        counts["seen_rows"] += 1
        if restrict is not None and row.key not in restrict:
            counts["restrict_skipped"] += 1
            continue
        if row.key in seen:
            counts["collisions"] += 1  # first-wins skip; see the module contract
            continue
        if row.key in done:
            counts["resume_skipped"] += 1
            continue
        if not any(row.board.legal_moves):
            counts["terminal_skipped"] += 1
            continue
        if row.legal_mask is not None and args.check_legal_mask:
            ours = {compact_index_for_move(row.board, m) for m in row.board.legal_moves}
            if ours != set(np.flatnonzero(row.legal_mask > 0).tolist()):
                raise SystemExit(
                    f"decoded legal moves disagree with the shard's stored "
                    f"legal_mask at {row.shard}:{row.row_index} ({row.key}); "
                    "the plane decode is wrong",
                )
        seen.add(row.key)
        counts["distinct_keys"] += 1
        row.legal_mask = None  # not needed downstream; do not keep it alive
        yield row
        emitted += 1
        if args.limit and emitted >= args.limit:
            return


# ---------------------------------------------------------------- dump file
def load_done_keys(out_path: Path) -> set[str]:
    """Keys already dumped, for --resume.

    ⚑ A record only counts as done if it actually carries a ``policy``. A write
    torn at just the right byte can leave STRUCTURALLY VALID JSON that stops
    before the policy field — it would parse, mark the key done, and that
    position would then be skipped forever with no policy in the dump. Requiring
    the payload makes an incomplete record get retried instead of trusted.
    """
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final write from a killed run
            key = rec.get("key")
            if key is not None and isinstance(rec.get("policy"), dict):
                done.add(str(key))
    return done


#: Record types the first line of a dump may carry. ``provenance`` is what this
#: script writes now (the rig's external-policy loader keys off exactly that
#: string); ``header`` is what it wrote before, and dumps in that shape still
#: exist on disk, so a resume must still be able to read their provenance.
HEADER_RECORD_TYPES = ("provenance", "header")


def read_header(out_path: Path) -> dict[str, Any] | None:
    """The header record of an existing dump, if it has one."""
    if not out_path.exists() or not out_path.stat().st_size:
        return None
    with out_path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return None
            return rec if rec.get("record") in HEADER_RECORD_TYPES else None
    return None


def truncate_torn_tail(out_path: Path) -> int:
    """Drop an unterminated final line before appending. Returns bytes removed.

    ⚑ Opening in ``"a"`` after a killed run appends straight onto a half-written
    record, welding it to the first new record to form ONE invalid line — so the
    torn row AND the first good row are both lost, and ``load_done_keys`` cannot
    see it because it only skips lines it cannot parse.
    """
    if not out_path.exists():
        return 0
    size = out_path.stat().st_size
    if size == 0:
        return 0
    with out_path.open("rb+") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) == b"\n":
            return 0
        # Walk back to the last newline and cut there.
        chunk = 1 << 16
        pos = size
        while pos > 0:
            step = min(chunk, pos)
            pos -= step
            fh.seek(pos)
            buf = fh.read(step)
            idx = buf.rfind(b"\n")
            if idx >= 0:
                keep = pos + idx + 1
                fh.truncate(keep)
                return size - keep
        fh.truncate(0)
        return size


# ------------------------------------------------------------------ session
def open_session(
    onnx: str, *, gpu_mem_gb: float, threads: int,
) -> tuple[Any, str, np.dtype[Any], list[str]]:
    import onnxruntime as ort

    providers: list[Any] = []
    if gpu_mem_gb > 0:
        if "CUDAExecutionProvider" in ort.get_available_providers():
            # ⚑ A bare provider NAME cannot carry gpu_mem_limit, and ORT
            # allocates through its own CUDA arena that torch's memory fraction
            # does not bound — so an uncapped session would sit on the
            # trainer's GPU.
            providers.append((
                "CUDAExecutionProvider",
                {"device_id": 0, "gpu_mem_limit": int(gpu_mem_gb * 1024 ** 3)},
            ))
        else:
            print(
                f"[dump] WARNING: --gpu-mem-gb {gpu_mem_gb} requested but this "
                f"onnxruntime has no CUDAExecutionProvider "
                f"(available: {ort.get_available_providers()}); running on CPU.",
                file=sys.stderr, flush=True,
            )
    providers.append("CPUExecutionProvider")

    options = None
    if threads > 0:
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
    sess = ort.InferenceSession(onnx, options, providers=providers)
    in_name = sess.get_inputs()[0].name
    in_type = next(i.type for i in sess.get_inputs() if i.name == in_name)
    dtype = np.dtype(np.float16 if in_type == "tensor(float16)" else np.float32)
    return sess, in_name, dtype, list(sess.get_providers())


def resolve_policy_output(sess: Any, policy_output: str | None) -> int:
    """Index of the policy tensor among the graph outputs.

    Picked by WIDTH, not position: some LC0/BT4 graphs emit the 3-wide WDL
    before the 1858-wide policy, so ``out[0]`` is not reliably the policy.
    An AMBIGUOUS graph raises rather than taking the first match, matching
    ``scripts/net_source.py`` — picking one of two policy heads silently is
    exactly the class of defect this file exists to avoid.
    """
    names = [o.name for o in sess.get_outputs()]
    if policy_output:
        if policy_output not in names:
            raise SystemExit(
                f"--policy-output {policy_output!r} is not an output of this graph. "
                f"Available: {names}",
            )
        return names.index(policy_output)
    widths = [
        (o.shape[-1] if isinstance(o.shape[-1], int) else -1) for o in sess.get_outputs()
    ]
    exact = [i for i, w in enumerate(widths) if w == COMPACT_POLICY_SIZE]
    if len(exact) > 1:
        raise SystemExit(
            f"graph has {len(exact)} {COMPACT_POLICY_SIZE}-wide outputs "
            f"({[names[i] for i in exact]}); pass --policy-output to say which.",
        )
    if exact:
        return exact[0]
    raise SystemExit(
        f"no {COMPACT_POLICY_SIZE}-wide policy output; widths={widths}, names={names}. "
        "Pass --policy-output explicitly.",
    )


# ------------------------------------------------------------------- scoring
def legal_move_policy(
    board: chess.Board, policy_row: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """(legal UCIs, probabilities) — softmax over the LEGAL moves only.

    The gather is :func:`leela_index_for_move`, which orients the move for a
    black-to-move board and applies LC0's own spelling for castling and
    promotions. Renormalising over legal moves (rather than softmaxing all
    1858 and slicing) is what makes the returned dict a distribution.
    """
    ucis = [m.uci() for m in board.legal_moves]
    if not ucis:
        return [], np.zeros((0,), dtype=np.float64)
    idx = np.array(
        [leela_index_for_move(board, chess.Move.from_uci(u)) for u in ucis],
        dtype=np.int64,
    )
    if int((idx < 0).sum()):
        missing = [u for u, i in zip(ucis, idx.tolist(), strict=True) if i < 0]
        raise RuntimeError(f"{board.fen()}: no LC0 policy slot for {missing}")
    logits = policy_row[idx].astype(np.float64)
    # BT4 emits -inf-ish fills for slots it masks; keep the softmax finite.
    logits = np.where(np.isfinite(logits), logits, -1e9)
    probs = np.exp(logits - logits.max())
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError(f"{board.fen()}: degenerate policy row (sum={total})")
    return ucis, probs / total


def entropy_nats(probs: np.ndarray) -> float:
    p = probs[probs > 0.0]
    return float(-(p * np.log(p)).sum())


def castling_probability(
    board: chess.Board, ucis: Sequence[str], probs: np.ndarray,
) -> dict[str, float]:
    """Prior on each legal castling move — the fingerprint of the fixed remap.

    Under the old static remap these read ~0 because the logit came from LC0's
    unrelated ``e1g1`` SLIDE slot rather than its ``e1h1`` castling slot.
    """
    out: dict[str, float] = {}
    for uci, p in zip(ucis, probs.tolist(), strict=True):
        if board.is_castling(chess.Move.from_uci(uci)):
            out[uci] = float(p)
    return out


# -------------------------------------------------------------------- header
def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except OSError:
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def remap_provenance() -> dict[str, Any]:
    """Which revision of the remap produced a dump.

    Records the last COMMIT touching the remap sources and each file's current
    blob hash. The blob hashes are the load-bearing half: a commit sha alone
    says nothing when the tree is dirty, which is exactly the state a dump run
    during development is made in.
    """
    return {
        "git_head": _git(["rev-parse", "HEAD"]),
        "commit": _git(["log", "-1", "--format=%H", "--", *REMAP_SOURCES]),
        "dirty": bool(_git(["status", "--porcelain", "--", *REMAP_SOURCES])),
        "blobs": {
            src: (_git(["hash-object", src]) or "unavailable") for src in REMAP_SOURCES
        },
    }


def build_header(args: argparse.Namespace, shard_dirs: list[str], pol_name: str,
                 providers: list[str], restrict: set[str] | None = None) -> dict[str, Any]:
    """The dump's provenance line.

    ⚑ ``record``/``v``/``policy_encoding`` are not decoration: they are the
    three fields ``RvgExternalPolicyIndex.load`` reads. ``v`` and
    ``policy_encoding`` are written on the shard path only — that is the path
    whose ``key`` is a rig join key; a FEN-mode dump is keyed by FEN strings
    and is not a rig file, so claiming the rig schema for it would be a lie.
    (The loader rejects a FEN dump either way, but note the mechanism: its row
    loop hits ``bytes.fromhex`` on the first FEN key BEFORE the after-loop
    missing-``v`` check ever runs, so the clean "declares no schema version"
    refusal only fires for a header-only file.)
    """
    header: dict[str, Any] = {
        "record": "provenance",
        "onnx": {"path": str(args.onnx), "sha256": file_sha256(args.onnx)},
        "policy_output": pol_name,
        "providers": providers,
        "batch_size": int(args.batch_size),
        "threads": int(args.threads),
        "input": {
            "planes": 112,
            "source": "shards" if shard_dirs else "fens",
            "shard_dirs": list(shard_dirs),
            "history_encoding": None if shard_dirs else HISTORY_ENCODING,
            "history_fill": None if shard_dirs else "repeat",
            "history_less": not shard_dirs,
        },
        "contract": {
            "frame": "canonical" if shard_dirs else "true",
            "key": (
                f"{FINGERPRINT_BYTES * 2}-hex plane fingerprint, from "
                "chess_anti_engine.eval.rvg_surgery.position_fingerprints -- the "
                "R/V/G rig's join key. The side-to-move canonical FEN is in the "
                "'fen' field (always White to move, fullmove always 1, halfmove "
                "saturating at 100); it is descriptive, never the join."
                if shard_dirs else
                "the caller's own FEN, verbatim, in true board coordinates"
            ),
            "policy_keys": (
                "canonical UCI (black O-O is e1g1, never e8g8)" if shard_dirs
                else "UCI in true board coordinates"
            ),
            "index_space": (
                "LC0 1858 order internally; UCI strings are the interchange format"
            ),
            "duplicate_keys": (
                "first-wins skip on the emitted key (the fingerprint); "
                "fen/shard/row/game_id/ply_index identify which row supplied it"
                if shard_dirs else
                "first-wins skip; shard/row/game_id/ply_index identify which"
            ),
        },
        "remap": remap_provenance(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if shard_dirs:
        header["v"] = RVG_EXTERNAL_POLICY_SCHEMA_VERSION
        header["policy_encoding"] = POLICY_ENCODING_LC0_1858
    header["restrict_keys"] = (
        None if args.restrict_keys is None
        else {**file_provenance(args.restrict_keys), "keys": len(restrict or ())}
    )
    return header


# ---------------------------------------------------------------------- main
def shard_source_dirs(spec: str) -> list[str]:
    """Directories named by ``--input-source shards:<dir>[,<dir>]``, else []."""
    if not spec or spec == "fens":
        return []
    if not spec.startswith("shards:"):
        raise SystemExit(f"--input-source must be 'fens' or 'shards:<dirs>'; got {spec!r}")
    dirs = [d for d in spec[len("shards:"):].split(",") if d]
    if not dirs:
        raise SystemExit("--input-source shards: needs at least one directory")
    return dirs


def _resume_guard(
    prior_header: dict[str, Any], header: dict[str, Any], pol_name: str, out_path: Path,
) -> dict[str, Any]:
    """Refuse to append rows that a dump's own header would misdescribe.

    ⚑ Without this, resuming with a different ``--onnx`` writes rows under the
    ORIGINAL net's provenance record and nothing ever says so.
    """
    old_sha = (prior_header.get("onnx") or {}).get("sha256")
    new_sha = header["onnx"]["sha256"]
    if old_sha and old_sha != new_sha:
        raise SystemExit(
            f"{out_path} was written with onnx sha256 {old_sha}, but --onnx is "
            f"{new_sha}. Resuming would mix two nets in one dump; use a different "
            "--out or pass the original net.",
        )
    old_source = (prior_header.get("input") or {}).get("source")
    if prior_header.get("policy_output") != pol_name or (
        old_source != header["input"]["source"]
    ):
        raise SystemExit(
            f"{out_path} was written with policy_output="
            f"{prior_header.get('policy_output')!r} / source={old_source!r}, which "
            f"does not match this run ({pol_name!r} / "
            f"{header['input']['source']!r}).",
        )
    # ⚑ KEY SCHEMA. A shard dump written before the rig-key change passes all
    # three checks above (same net, same output, source "shards") but keys its
    # rows by FEN and declares no ``v`` — so ``load_done_keys`` would fill
    # ``done`` with FEN strings that can never match a fingerprint: the run
    # prints a nonzero "resume: N keys already present" and then silently
    # re-scans and re-emits EVERY row, and the merged file still opens with a
    # ``header`` record the rig loader cannot classify. Refuse instead.
    old_v = prior_header.get("v")
    if old_v != header.get("v"):
        raise SystemExit(
            f"{out_path} declares key-schema version {old_v!r} "
            f"(record={prior_header.get('record')!r}), but this run writes "
            f"v={header.get('v')!r}. Their keys do not join, so its rows cannot "
            "be skipped and resuming would redo the whole scan while appending "
            "under an incompatible header. Use a fresh --out and keep the old "
            "dump as its own artifact.",
        )
    # ⚑ STILL A `provenance` RECORD. The rig loader classifies every line by
    # `record`, so a `"resume"` line would fall through to the row branch and
    # die on a missing `key` -- a resumed dump would be unloadable purely
    # because it was resumed. `resumed: true` carries the distinction instead,
    # and the loader's last-provenance-wins read then describes the rows this
    # run appended.
    return dict(header, record="provenance", resumed=True)


def run(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size and not args.resume:
        # Appending a second header + a second copy of every row to an existing
        # dump is the silent-corruption failure; make the caller say which.
        raise SystemExit(
            f"{out_path} already exists and is non-empty. Pass --resume to "
            "continue it, or remove it to start over.",
        )

    shard_dirs = shard_source_dirs(args.input_source)
    restrict: set[str] | None = None
    if args.restrict_keys is not None:
        # FEN mode is keyed by FEN strings, so a hex key set could only ever
        # match zero rows there -- an empty dump and no error is precisely the
        # silent-wrongness this refusal exists to prevent.
        if not shard_dirs:
            raise SystemExit(
                "--restrict-keys is shard mode only: it enumerates 32-hex plane "
                "fingerprints, and FEN mode keys its records by the caller's own "
                "FEN string, so the restriction would match nothing. Use "
                "--input-source shards:<dirs>, or filter the --rows file itself.",
            )
        restrict = load_restrict_keys(args.restrict_keys)
        if not restrict:
            raise SystemExit(
                f"--restrict-keys {args.restrict_keys} enumerates no keys; that "
                "would dump nothing.",
            )
        print(f"[dump] --restrict-keys {args.restrict_keys}: {len(restrict)} keys")

    prior_header = read_header(out_path) if args.resume else None
    done: set[str] = set()
    if args.resume:
        # ⚑ N5: a file whose header cannot be read is exactly the damaged case
        # the mismatch guard exists for -- so refusing here, rather than
        # appending provenance-less rows, is the whole point.
        if out_path.exists() and out_path.stat().st_size and prior_header is None:
            raise SystemExit(
                f"{out_path} has no readable header record, so this run cannot "
                "check that it was produced by the same net and source. Refusing "
                "to append; use a fresh --out.",
            )
    sess, in_name, in_dtype, providers = open_session(
        args.onnx, gpu_mem_gb=args.gpu_mem_gb, threads=args.threads,
    )
    pol_idx = resolve_policy_output(sess, args.policy_output)
    pol_name = sess.get_outputs()[pol_idx].name
    print(f"[dump] providers={providers} input={in_name}({in_dtype}) policy={pol_name}")
    print(f"[dump] source: {'shards ' + ','.join(shard_dirs) if shard_dirs else args.rows}"
          f", history="
          f"{'TRUE (from stored x)' if shard_dirs else 'repeat-filled (FEN-only)'}")

    header = build_header(args, shard_dirs, pol_name, providers, restrict)
    resume_note = (
        _resume_guard(prior_header, header, pol_name, out_path)
        if prior_header is not None else None
    )
    if args.resume:
        # Only after the guard: printing "N keys already present" and then
        # refusing would report a resume that never happens.
        done = load_done_keys(out_path)
        print(f"[dump] resume: {len(done)} keys already present")
    # ⚑ N4: only NOW is the file touched. Truncating before the guard meant a
    # refused resume had already mutated the file it refused to write to.
    if args.resume:
        dropped = truncate_torn_tail(out_path)
        if dropped:
            print(f"[dump] resume: dropped {dropped} bytes of a torn final record")

    stats: dict[str, int] = {}
    n_written = 0
    ent_sum = 0.0
    top1_sum = 0.0
    castle_examples: list[dict[str, Any]] = []
    t0 = time.time()

    needs_header = not (out_path.exists() and out_path.stat().st_size)
    with out_path.open("a", encoding="utf-8") as fh:
        if needs_header:
            fh.write(json.dumps(header) + "\n")
        elif resume_note is not None:
            # Provenance for the rows appended by THIS run, so a dump built over
            # several sessions still says what produced each stretch of it.
            fh.write(json.dumps(resume_note) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

        for batch in batched(
            source_rows(args, done, stats, restrict), int(args.batch_size),
        ):
            feats = np.stack([
                row.planes if row.planes is not None else fill_lc0_history_repeat(
                    encode_position(row.board, add_features=False,
                                    input_history_encoding=HISTORY_ENCODING),
                )
                for row in batch
            ]).astype(in_dtype, copy=False)
            raw = sess.run([pol_name], {in_name: feats})[0]
            policy = np.asarray(raw, dtype=np.float32)[:, :COMPACT_POLICY_SIZE]

            lines: list[str] = []
            for i, row in enumerate(batch):
                ucis, probs = legal_move_policy(row.board, policy[i])
                ent_sum += entropy_nats(probs)
                top1_sum += float(probs.max())
                best = ucis[int(np.argmax(probs))]
                castles = castling_probability(row.board, ucis, probs)
                if castles and len(castle_examples) < args.castle_examples:
                    castle_examples.append({
                        "where": row.fen or row.key, "castling": castles,
                        "best": best,
                        "best_p": round(float(probs.max()), 4),
                    })
                lines.append(json.dumps({
                    "key": row.key,
                    "n_legal": len(ucis),
                    **row.identity(),
                    "policy": {u: round(float(p), 6)
                               for u, p in zip(ucis, probs.tolist(), strict=True)},
                }))
                n_written += 1
            fh.write("".join(line + "\n" for line in lines))
            fh.flush()
            os.fsync(fh.fileno())

            elapsed = time.time() - t0
            rate = n_written / elapsed if elapsed > 0 else 0.0
            print(f"[dump] {n_written} rows {rate:.1f} pos/s", flush=True)

    elapsed = time.time() - t0
    rate = n_written / elapsed if elapsed > 0 else 0.0
    print(f"[dump] wrote {n_written} rows in {elapsed:.1f}s = {rate:.1f} pos/s "
          f"-> {out_path}")
    print(f"[dump] rows read {stats.get('seen_rows', 0)}, distinct keys "
          f"{stats.get('distinct_keys', 0)}, transposition collisions skipped "
          f"{stats.get('collisions', 0)}, terminal positions skipped "
          f"{stats.get('terminal_skipped', 0)}, already-done skipped "
          f"{stats.get('resume_skipped', 0)}")
    if restrict is not None:
        print(f"[dump] --restrict-keys {args.restrict_keys}: {len(restrict)} keys "
              f"enumerated -> kept {stats.get('distinct_keys', 0)}, skipped "
              f"{stats.get('restrict_skipped', 0)} rows not in the set "
              f"(of {stats.get('seen_rows', 0)} read)")
    if n_written:
        # ⚑ NOT comparable to the banked 0.970: that figure is BT4-100, i.e.
        # after 100 nodes of SEARCH and matched to our branching factor. This is
        # the RAW PRIOR, whose documented top-1 is 0.476 against 0.693
        # post-search. Judge a wiring change by the delta against a previous run
        # of THIS script on THE SAME rows, never against 0.970.
        print(f"[sanity] mean legal-move entropy: {ent_sum / n_written:.4f} nats "
              "(raw prior; NOT the banked 0.970, which is BT4-100 post-search)")
        print(f"[sanity] mean top-1 probability: {top1_sum / n_written:.4f} "
              "(BT4 raw-prior reference ~0.476)")
    for ex in castle_examples:
        print(f"[sanity] castling prior {ex['castling']} "
              f"(best {ex['best']} p={ex['best_p']}) {ex['where']}")
    if not castle_examples:
        print("[sanity] no position with a legal castling move in this slice")
    if rate > 0:
        print(f"[project] 3.1M rows at {rate:.1f} pos/s = "
              f"{3_100_000 / rate / 3600:.1f} h")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__ or "",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default=None,
                    help="JSONL with a 'key'/'fen' field, or one FEN per line "
                         "(FEN-only mode; ignored with --input-source shards:)")
    ap.add_argument("--input-source", default="fens",
                    help="'fens' (default, history-less, for the audit set) or "
                         "'shards:<dir>[,<dir>]' to read stored replay x tensors, "
                         "which carry TRUE history")
    ap.add_argument("--no-check-legal-mask", dest="check_legal_mask",
                    action="store_false",
                    help="shard mode: skip asserting that the decoded legal moves "
                         "match the shard's own legal_mask (the check is ON by "
                         "default and is the plane decode's only cross-check)")
    ap.set_defaults(check_legal_mask=True)
    ap.add_argument("--out", required=True, help="JSONL dump path (appended to)")
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threads", type=int, default=16,
                    help="ORT intra-op threads for the CPU provider (0 = ORT default)")
    ap.add_argument("--gpu-mem-gb", type=float, default=0.0,
                    help=">0 tries CUDAExecutionProvider with this hard memory cap; "
                         "warns and uses CPU if the provider is unavailable")
    ap.add_argument("--policy-output", default=None,
                    help="ONNX policy output name (default: the 1858-wide one; "
                         "required if the graph has more than one)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N emitted rows (0 = all). Every skip is "
                         "applied before counting, so with --resume this means "
                         "N NEW rows, with --restrict-keys N RESTRICTED rows, "
                         "and with both it counts only rows that pass BOTH "
                         "filters")
    ap.add_argument("--restrict-keys", type=Path, default=None,
                    help="shard mode only: dump ONLY rows whose 32-hex plane "
                         "fingerprint appears in this file (one key per line; "
                         "lines starting with '{' are skipped, so a label file's "
                         "own provenance header can be fed back in)")
    ap.add_argument("--resume", action="store_true",
                    help="skip keys already present in --out")
    ap.add_argument("--castle-examples", type=int, default=3,
                    help="how many legal-castling positions to print priors for")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.rows and not shard_source_dirs(args.input_source):
        raise SystemExit("need --rows, or --input-source shards:<dirs>")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
