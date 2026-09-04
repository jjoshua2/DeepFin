"""How much does BT4's policy move when you take its move history away?

Every BT4 instrument in this repo feeds the net a position built from a FEN
alone, with the 7 older history frames faked. BT4 was trained on real history,
so that is a known degradation -- but its SIZE has never been measured here,
and a policy dump used as a training target is only as good as that bound.

This probe measures it against ground truth. lc0's public v6 training records
carry the REAL 112-plane input BT4-class nets were trained on, history frames
included. So for each sampled record we run the same net twice: once on the
planes exactly as stored, and once on the same planes with the history
replaced by what a FEN-only build produces. Everything else -- castling, side
to move, rule50, the ones plane -- is held identical, so the delta is history
and nothing else.

Two history-less arms are reported, because they are not the same thing:

``repeat``
    frames 1..7 replaced by a copy of frame 0, which is what
    ``fill_lc0_history_repeat`` does. ⚑ It still inherits frame 0's REPETITION
    plane from the lc0 record, and a FEN-built board can never set that plane
    (it has no move stack), so this arm slightly UNDERSTATES the degradation on
    repeated positions.
``repeat_norep``
    ``repeat`` plus the repetition planes cleared -- the faithful emulation of
    what ``scripts/bt4_policy_dump.py``, ``scripts/foreign_net_audit.py`` and
    ``OnnxChessNet`` actually feed from a FEN. **This is the decision arm.**
``zero``
    frames 1..7 zeroed. Included because it is the intuitive reading of
    "no history", and because the repo's own encoder notes claim zeros break
    the LC0 value head -- worth having the number rather than the claim.

Also validates the plane construction itself: for a subset it rebuilds the
input from the reconstructed position through the ordinary FEN path and
requires the CURRENT-position piece planes to match the raw record bit for
bit. A mismatch there means every other number on the page is measuring our
own encoder, so it is checked first and reported as a hard PASS/FAIL.

Example
-------
    PYTHONPATH=. python3 scripts/bt4_history_sensitivity.py \
        --tar data/lc0_training/training-run2-test91-20260810-1417.tar \
        --positions 2000 --threads 16
"""
from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import (
    LC0_FULL,
    fill_lc0_history_repeat,
    x_to_lc0_planes,
)
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.leela_index import leela_index_for_move
from scripts.bt4_policy_dump import (
    HISTORY_ENCODING,
    entropy_nats,
    file_sha256,
    open_session,
    remap_provenance,
    resolve_policy_output,
)

# --- v6 training record, by fixed offset (lczero-training tf/chunkparser.py) --
V6_RECORD_BYTES = 8356
V6_OFF_VERSION = 0            # int32, == 6
V6_OFF_INPUT_FORMAT = 4       # int32, == 1 for the classical 112-plane input
V6_OFF_PROBS = 8              # 1858 x float32 (the net's search policy target)
V6_OFF_PLANES = 7440          # 104 x uint64 bitboards: 8 frames x 13 planes
V6_OFF_CASTLING = 8272        # us_ooo, us_oo, them_ooo, them_oo (4 x uint8)
V6_OFF_STM = 8276             # side_to_move (classical) / en passant (later)
V6_OFF_RULE50 = 8277          # uint8, RAW ply count -- lc0 plane 109 is raw
V6_INPUT_CLASSICAL = 1

LC0_FRAMES = 8
LC0_PLANES_PER_FRAME = 13     # 12 piece bitboards + 1 repetition plane
CURRENT_FRAME_PLANES = LC0_PLANES_PER_FRAME
PIECE_PLANES_PER_FRAME = 12
HISTORY_START = LC0_PLANES_PER_FRAME          # 13
HISTORY_END = LC0_FRAMES * LC0_PLANES_PER_FRAME  # 104

PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING)
#: Fraction of rows allowed to disagree on a repetition plane. lc0 counts
#: repetitions over the whole game; our encoder sees only the history window it
#: is handed, so a repetition older than 7 plies is invisible to us. Measured
#: 7/400 = 0.0175; the budget is set with headroom, not to the measurement.
REPETITION_MISMATCH_BUDGET = 0.05


@dataclass
class Record:
    planes: np.ndarray          # (112, 8, 8) float32, exactly as lc0 stored it
    board: chess.Board          # side-to-move oriented, so always White to move
    masks: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="<u8"))
    legal: list[str] = field(default_factory=list)
    gather: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def with_history_stack(self) -> chess.Board:
        """``board`` carrying the record's 7 older frames as a python-chess stack.

        Our encoder reads history from ``board._stack`` back-to-front
        (``stack[len - hist_idx]`` for ``hist_idx`` 1..7), so the stack must run
        oldest-first. Frames are already ROOT-POV in both conventions, so "us"
        is White in the reconstruction for every frame, not just frame 0.
        """
        board = self.board.copy(stack=False)
        stack = []
        for f in range(LC0_FRAMES - 1, 0, -1):
            frame = self.masks[f * LC0_PLANES_PER_FRAME:
                               f * LC0_PLANES_PER_FRAME + PIECE_PLANES_PER_FRAME]
            if not any(int(v) for v in frame):
                continue  # unplayed frames at the start of a game
            tmp = frame_board(frame, board.castling_rights, 0)
            stack.append(chess._BoardState(tmp))
        board._stack = stack
        return board


def plane_from_mask(mask: int) -> np.ndarray:
    """One 8x8 plane from an lc0 bitboard.

    ⚑ lc0's bit order runs h..a WITHIN a rank, so the file axis is reversed
    relative to a naive little-endian unpack; skipping the flip silently puts
    kings on d1 and every downstream number is then measuring nothing.
    """
    bits = np.unpackbits(
        np.array([mask], dtype="<u8").view(np.uint8), bitorder="little",
    ).astype(np.float32).reshape(8, 8)
    return bits[:, ::-1]


def frame_board(frame_masks: np.ndarray, castling: int, rule50: int) -> chess.Board:
    """Reconstruct the position from one 13-plane frame's bitboards."""
    board = chess.Board(None)
    board.turn = chess.WHITE  # records are already side-to-move oriented
    for i, piece_type in enumerate(PIECE_ORDER):
        for bit in chess.scan_forward(int(frame_masks[i])):
            board.set_piece_at(int(_BIT_TO_SQUARE[bit]), chess.Piece(piece_type, chess.WHITE))
        for bit in chess.scan_forward(int(frame_masks[6 + i])):
            board.set_piece_at(int(_BIT_TO_SQUARE[bit]), chess.Piece(piece_type, chess.BLACK))
    board.castling_rights = castling & board.rooks
    board.halfmove_clock = int(rule50)
    return board


# lc0 bit j -> rank j//8, file 7-(j%8)
_BIT_TO_SQUARE = np.array(
    [chess.square(7 - (j % 8), j // 8) for j in range(64)], dtype=np.int64,
)


def parse_record(buf: bytes) -> Record | None:
    version, input_format = struct.unpack_from("<ii", buf, V6_OFF_VERSION)
    if version != 6 or input_format != V6_INPUT_CLASSICAL:
        return None
    masks = np.frombuffer(buf, dtype="<u8", count=104, offset=V6_OFF_PLANES)
    castle = np.frombuffer(buf, dtype=np.uint8, count=4, offset=V6_OFF_CASTLING)
    stm = buf[V6_OFF_STM]
    rule50 = buf[V6_OFF_RULE50]

    planes = np.zeros((LC0_FULL.num_planes, 8, 8), dtype=np.float32)
    for p in range(HISTORY_END):
        planes[p] = plane_from_mask(int(masks[p]))
    base = LC0_FULL.root_metadata_base
    for i in range(4):
        planes[base + i] = 1.0 if castle[i] else 0.0
    planes[base + 4] = float(stm)
    planes[base + 5] = float(rule50)   # RAW, matching lc0 and our `lc0_root`
    planes[LC0_FULL.ones_plane] = 1.0

    rights = 0
    if castle[0]:
        rights |= chess.BB_A1
    if castle[1]:
        rights |= chess.BB_H1
    if castle[2]:
        rights |= chess.BB_A8
    if castle[3]:
        rights |= chess.BB_H8
    board = frame_board(masks[:PIECE_PLANES_PER_FRAME], rights, int(rule50))
    if not board.is_valid():
        return None
    legal = [m.uci() for m in board.legal_moves]
    if len(legal) < 2:  # a forced move carries no policy signal
        return None
    gather = np.array(
        [leela_index_for_move(board, chess.Move.from_uci(u)) for u in legal],
        dtype=np.int64,
    )
    if int((gather < 0).sum()):
        return None
    return Record(planes=planes, board=board, masks=masks, legal=legal, gather=gather)


def roundtrip_check(records: list[Record], limit: int) -> dict[str, Any]:
    """Our tensor -> LC0 planes, diffed against LC0's own bits.

    Rebuilds each record's position AND history through our production encoder
    (``lc0_root_legacy_meta`` + ``v2_threats``, i.e. exactly what the replay
    shards store), converts with :func:`x_to_lc0_planes`, and compares against
    the raw record. Piece planes must match EXACTLY. The repetition planes are
    reported separately and are the one expected disagreement: lc0 counts
    repetitions over the whole game, our encoder only sees the 7 plies it was
    handed, so a repetition older than the history window is invisible to us.
    """
    piece_rows = rep_rows = meta_rows = stm_rows = 0
    rep_planes_differing = 0
    checked = 0
    examples: list[str] = []
    piece_idx = np.concatenate([
        np.arange(f * LC0_PLANES_PER_FRAME, f * LC0_PLANES_PER_FRAME + PIECE_PLANES_PER_FRAME)
        for f in range(LC0_FRAMES)
    ])
    rep_idx = np.array(
        [f * LC0_PLANES_PER_FRAME + PIECE_PLANES_PER_FRAME for f in range(LC0_FRAMES)],
    )
    # ⚑ Plane 108 (side to move) is scored SEPARATELY and is expected to differ
    # on ~half the rows. A v6 record is POV-canonical -- it stores the position
    # already flipped -- so the reconstruction is always White-to-move and our
    # encoder honestly writes stm = 0, while the record remembers that the real
    # position was Black-to-move. That is the RECONSTRUCTION losing the original
    # colour, not the converter: given a real board our encoder does set plane
    # 108 to 1.0 for Black (verified directly). Shard rows are encoded from the
    # true board, so they carry the right value.
    stm_plane = LC0_FULL.root_metadata_base + 4
    meta_idx = np.array([p for p in range(LC0_FULL.root_metadata_base, LC0_FULL.num_planes)
                         if p != stm_plane])
    for rec in records[:limit]:
        board = rec.with_history_stack()
        ours = encode_position(board, add_features=True,
                               input_history_encoding="lc0_root_legacy_meta")
        got = x_to_lc0_planes(ours, input_history_encoding="lc0_root_legacy_meta")
        checked += 1
        if not np.array_equal(got[piece_idx], rec.planes[piece_idx]):
            piece_rows += 1
            if len(examples) < 3:
                bad = [int(p) for p in piece_idx
                       if not np.array_equal(got[p], rec.planes[p])]
                examples.append(f"{rec.board.fen()} piece planes {bad[:8]}")
        if not np.array_equal(got[rep_idx], rec.planes[rep_idx]):
            rep_rows += 1
            rep_planes_differing += int(sum(
                not np.array_equal(got[p], rec.planes[p]) for p in rep_idx
            ))
        if not np.array_equal(got[stm_plane], rec.planes[stm_plane]):
            stm_rows += 1
        if not np.array_equal(got[meta_idx], rec.planes[meta_idx]):
            meta_rows += 1
            if len(examples) < 6:
                bad = [int(p) for p in meta_idx
                       if not np.array_equal(got[p], rec.planes[p])]
                examples.append(
                    f"{rec.board.fen()} meta planes {bad} "
                    f"ours={[float(got[p][0, 0]) for p in bad]} "
                    f"lc0={[float(rec.planes[p][0, 0]) for p in bad]}",
                )
    return {
        "checked": checked,
        "piece_plane_mismatch_rows": piece_rows,
        "repetition_mismatch_rows": rep_rows,
        "repetition_planes_differing": rep_planes_differing,
        "metadata_mismatch_rows": meta_rows,
        "stm_plane_mismatch_rows_expected": stm_rows,
        "examples": examples,
    }


def sample_records(tar_path: Path, want: int, per_game: int, seed: int) -> list[Record]:
    """Records spread across many games, so the sample is not one game's opening."""
    rng = np.random.default_rng(seed)
    out: list[Record] = []
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if len(out) >= want:
                break
            if not member.isfile() or not member.name.endswith(".gz"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            try:
                blob = gzip.decompress(handle.read())
            except (OSError, EOFError):
                continue
            n = len(blob) // V6_RECORD_BYTES
            if n == 0:
                continue
            picks = rng.choice(n, size=min(per_game, n), replace=False)
            for i in sorted(int(p) for p in picks):
                rec = parse_record(blob[i * V6_RECORD_BYTES:(i + 1) * V6_RECORD_BYTES])
                if rec is not None:
                    out.append(rec)
                if len(out) >= want:
                    break
    return out


# ------------------------------------------------------------------- arms
#: Offset of a frame's repetition plane (12 piece planes, then 1 repetition).
FRAME_REPETITION = PIECE_PLANES_PER_FRAME


def history_stripped(planes: np.ndarray, mode: str) -> np.ndarray:
    """A copy of ``planes`` with the 7 older history frames replaced.

    ``repeat_norep`` additionally clears the repetition planes, because a board
    rebuilt from a FEN has no move stack and therefore cannot know it is a
    repetition -- leaving lc0's bit set would flatter the FEN-only path.
    """
    out = planes.copy()
    out[HISTORY_START:HISTORY_END] = 0.0
    if mode in ("repeat", "repeat_norep"):
        fill_lc0_history_repeat(out)
        if mode == "repeat_norep":
            for slot in range(LC0_FRAMES):
                out[slot * LC0_PLANES_PER_FRAME + FRAME_REPETITION] = 0.0
    elif mode != "zero":
        raise ValueError(f"unknown history mode {mode!r}")
    return out


def softmax_legal(policy_row: np.ndarray, gather: np.ndarray) -> np.ndarray:
    logits = policy_row[gather].astype(np.float64)
    logits = np.where(np.isfinite(logits), logits, -1e9)
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, average ties. Written out to avoid a scipy import."""
    if a.size < 2:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    # average ties so a flat tail cannot fake a perfect correlation
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size, dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def top_gaps(logp: np.ndarray, order: np.ndarray, k: int) -> np.ndarray:
    """log-probability gaps between adjacent moves in a fixed ranking."""
    idx = order[:k]
    return logp[idx[:-1]] - logp[idx[1:]]


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    records = sample_records(
        Path(args.tar), args.positions, args.per_game, args.seed,
    )
    print(f"[probe] {len(records)} usable records from {Path(args.tar).name} "
          f"in {time.time() - t0:.1f}s")
    if not records:
        raise SystemExit("no usable v6 records parsed")

    # ---- (3) plane-construction validation, BEFORE any policy number ----
    checked = mismatched = 0
    examples: list[str] = []
    for rec in records[:args.validate]:
        ours = encode_position(rec.board, add_features=False,
                               input_history_encoding=HISTORY_ENCODING)
        theirs = rec.planes
        if not np.array_equal(ours[:PIECE_PLANES_PER_FRAME],
                              theirs[:PIECE_PLANES_PER_FRAME]):
            mismatched += 1
            if len(examples) < 3:
                bad = [p for p in range(PIECE_PLANES_PER_FRAME)
                       if not np.array_equal(ours[p], theirs[p])]
                examples.append(f"{rec.board.fen()} planes {bad}")
        checked += 1
    print(f"[validate] current-position piece planes: {checked - mismatched}/{checked} "
          f"exact vs the raw record")
    for e in examples:
        print(f"[validate]   MISMATCH {e}")
    if mismatched:
        print("[validate] FAIL — our FEN->planes path disagrees with lc0's own "
              "records, so every policy number below would be measuring our "
              "encoder. Fix this first.")
        return 1
    print("[validate] PASS")

    rt = roundtrip_check(records, args.roundtrip)
    print(f"[roundtrip] our tensor -> x_to_lc0_planes vs lc0's own bits, "
          f"{rt['checked']} records")
    print(f"[roundtrip]   piece/history planes mismatching rows: "
          f"{rt['piece_plane_mismatch_rows']}/{rt['checked']}")
    print(f"[roundtrip]   repetition-plane mismatching rows:     "
          f"{rt['repetition_mismatch_rows']}/{rt['checked']} "
          f"({rt['repetition_planes_differing']} planes of "
          f"{rt['checked'] * LC0_FRAMES})")
    print(f"[roundtrip]   metadata (excl. stm) mismatching rows:  "
          f"{rt['metadata_mismatch_rows']}/{rt['checked']}")
    print(f"[roundtrip]   stm plane differing (EXPECTED, the record is "
          f"POV-canonical so the rebuild loses the original colour): "
          f"{rt['stm_plane_mismatch_rows_expected']}/{rt['checked']}")
    for e in rt["examples"]:
        print(f"[roundtrip]   {e}")
    # ⚑ The equivalence this script exists to prove must FAIL THE PROCESS, not
    # merely print. Piece/history and metadata planes must match exactly; the
    # repetition planes are allowed a small budget because lc0 counts
    # repetitions over the whole game and our encoder only sees the 7 plies it
    # was handed (measured: 7/400 rows, 8 of 3200 planes).
    rep_rate = rt["repetition_mismatch_rows"] / max(1, rt["checked"])
    if (rt["piece_plane_mismatch_rows"] or rt["metadata_mismatch_rows"]
            or rep_rate > REPETITION_MISMATCH_BUDGET):
        print(f"[roundtrip] FAIL — piece={rt['piece_plane_mismatch_rows']} "
              f"metadata={rt['metadata_mismatch_rows']} "
              f"repetition={rep_rate:.4f} (budget {REPETITION_MISMATCH_BUDGET})")
        return 1
    print("[roundtrip] PASS")

    # ---- inference: the true planes and both history-less arms ----
    sess, in_name, in_dtype, providers = open_session(
        args.onnx, gpu_mem_gb=args.gpu_mem_gb, threads=args.threads,
    )
    pol_idx = resolve_policy_output(sess, None)
    pol_name = sess.get_outputs()[pol_idx].name
    print(f"[probe] providers={providers} policy={pol_name}")

    arms = {
        "true": [r.planes for r in records],
        "repeat": [history_stripped(r.planes, "repeat") for r in records],
        "repeat_norep": [history_stripped(r.planes, "repeat_norep") for r in records],
        "zero": [history_stripped(r.planes, "zero") for r in records],
    }
    scored: dict[str, np.ndarray] = {}
    for name, stack in arms.items():
        feats = np.stack(stack).astype(in_dtype, copy=False)
        chunks = []
        for s in range(0, len(feats), args.batch_size):
            chunks.append(np.asarray(
                sess.run([pol_name], {in_name: feats[s:s + args.batch_size]})[0],
                dtype=np.float32,
            )[:, :COMPACT_POLICY_SIZE])
            print(f"[probe] {name} {min(s + args.batch_size, len(feats))}/{len(feats)}",
                  flush=True)
        scored[name] = np.concatenate(chunks)

    report: dict[str, Any] = {
        "tar": Path(args.tar).name,
        "positions": len(records),
        "onnx": {"path": str(args.onnx), "sha256": file_sha256(args.onnx)},
        "providers": providers,
        "remap": remap_provenance(),
        "plane_validation": {"checked": checked, "mismatched": mismatched},
        "roundtrip": rt,
        "arms": {},
    }
    for arm in ("repeat", "repeat_norep", "zero"):
        report["arms"][arm] = compare(records, scored["true"], scored[arm], args.topk)

    print_report(report, args)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[probe] report -> {args.out}")
    return 0


def compare(
    records: list[Record], pol_true: np.ndarray, pol_alt: np.ndarray, topk: int,
) -> dict[str, Any]:
    flips = 0
    kls: list[float] = []
    dent: list[float] = []
    rhos: list[float] = []
    gaps: list[float] = []
    for i, rec in enumerate(records):
        p = softmax_legal(pol_true[i], rec.gather)
        q = softmax_legal(pol_alt[i], rec.gather)
        if int(np.argmax(p)) != int(np.argmax(q)):
            flips += 1
        kls.append(float((p * (np.log(p) - np.log(q))).sum()))
        dent.append(entropy_nats(q) - entropy_nats(p))
        lp, lq = np.log(p), np.log(q)
        rhos.append(spearman(lp, lq))
        order = np.argsort(-p)
        if order.size >= 2:
            k = min(topk, order.size)
            gaps.append(float(np.abs(top_gaps(lp, order, k) - top_gaps(lq, order, k)).mean()))
    rho = np.array([r for r in rhos if np.isfinite(r)], dtype=np.float64)
    return {
        "n": len(records),
        "top1_flip_rate": flips / len(records),
        "top1_flips": flips,
        "kl_median": float(np.median(kls)),
        "kl_mean": float(np.mean(kls)),
        "kl_p90": float(np.percentile(kls, 90)),
        "entropy_shift_mean": float(np.mean(dent)),
        "entropy_shift_median": float(np.median(dent)),
        "spearman_mean": float(rho.mean()) if rho.size else float("nan"),
        "spearman_median": float(np.median(rho)) if rho.size else float("nan"),
        "spearman_p10": float(np.percentile(rho, 10)) if rho.size else float("nan"),
        "top_gap_abs_delta_mean": float(np.mean(gaps)) if gaps else float("nan"),
        "top_gap_abs_delta_median": float(np.median(gaps)) if gaps else float("nan"),
    }


def verdict(flip: float, kl_median: float) -> str:
    """The PRE-COMMITTED rule. Stated once, applied without softening."""
    if flip <= 0.03 and kl_median <= 0.05:
        return "FEN-ONLY LICENSED"
    if flip > 0.08 or kl_median > 0.15:
        return "HISTORY REQUIRED"
    return "OPERATOR CALL"


def print_report(report: dict[str, Any], args: argparse.Namespace) -> None:
    print(f"\n=== BT4 history sensitivity ({report['positions']} positions, "
          f"{report['tar']}) ===")
    for arm, m in report["arms"].items():
        tag = (" (DECISION ARM - what a FEN-built input really is)"
               if arm == "repeat_norep" else "")
        print(f"\n-- history '{arm}'{tag}")
        print(f"   top-1 flip rate        {m['top1_flip_rate']:.4f} "
              f"({m['top1_flips']}/{m['n']})")
        print(f"   KL(true||alt) median   {m['kl_median']:.4f} nats")
        print(f"   KL(true||alt) mean     {m['kl_mean']:.4f} nats  "
              f"(p90 {m['kl_p90']:.4f})")
        print(f"   entropy shift mean     {m['entropy_shift_mean']:+.4f} nats  "
              f"(median {m['entropy_shift_median']:+.4f})")
        print(f"   Spearman(log p) mean   {m['spearman_mean']:.4f}  "
              f"(median {m['spearman_median']:.4f}, p10 {m['spearman_p10']:.4f})")
        print(f"   |d log-gap| top-{args.topk} mean  "
              f"{m['top_gap_abs_delta_mean']:.4f}  "
              f"(median {m['top_gap_abs_delta_median']:.4f})")
        print(f"   VERDICT: {verdict(m['top1_flip_rate'], m['kl_median'])}")
    decision = report["arms"]["repeat_norep"]
    print(f"\n>>> PRE-COMMITTED DECISION (repeat_norep arm): "
          f"{verdict(decision['top1_flip_rate'], decision['kl_median'])}")
    print("    rule: flip<=3% AND medKL<=0.05 => licensed; "
          "flip>8% OR medKL>0.15 => history required; else operator call")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__ or "",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", required=True, help="one lc0 v6 training tarball")
    ap.add_argument("--onnx", default="data/lc0/onnx/BT4-it332-vanilla-winner.onnx")
    ap.add_argument("--positions", type=int, default=2000)
    ap.add_argument("--per-game", type=int, default=3,
                    help="records sampled per game, to decorrelate the sample")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--gpu-mem-gb", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--validate", type=int, default=300,
                    help="records to plane-check against the raw v6 bits")
    ap.add_argument("--roundtrip", type=int, default=200,
                    help="records for the our-tensor -> lc0-planes round trip")
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--out", default=None, help="write the report as JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
