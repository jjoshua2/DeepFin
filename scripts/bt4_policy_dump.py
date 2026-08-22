"""Batched BT4 (LC0) policy dump over a FEN row list, as JSONL.

For each input position this runs the BT4 ONNX policy head, gathers the logits
of the LEGAL moves through the board-aware LC0-1858 mapping
(:mod:`chess_anti_engine.moves.leela_index`), softmaxes over the legal moves
ONLY, and writes ``{key, n_legal, policy: {uci: prob}}``.

⚑ HISTORY-LESS INPUT — A KNOWN, DOCUMENTED DEGRADATION. The 112-plane LC0
input is built from the FEN alone, with the 7 history frames filled by
repeating the current position (``fill_lc0_history_repeat``). BT4 was trained
on real move history, so this asks it a question it never saw in training. Two
consequences worth naming: repetition/50-move context is faked, and en passant
is conveyed in the T-format through the history frames rather than a plane, so
on the ~1% of positions whose key move is an en-passant capture BT4 cannot see
the double-push that made it available. This is the SAME convention every
prior BT4 instrument here used (``scripts/foreign_net_audit.py``,
``scripts/lc0_adapter_probe.py``, and the banked ``bt4_audit_cache``), so the
numbers stay comparable with those; it is not a defect introduced here, and it
is not a reason to compare these dumps against a history-carrying BT4 run.

⚑ The gather is board-aware for a reason. LC0's 1858 head spells castling
king-takes-rook (``e1h1``) and ALSO has an ordinary ``e1g1`` slide slot, and it
puts KNIGHT promotion (not queen) on the bare 7th->8th slot. A static
from/to-string remap reads a real but unrelated logit for both families rather
than failing, which is how castling priors came out 49x-120x too small. See
``chess_anti_engine/onnx/load.py`` and the header record's ``remap`` block,
which pins the exact code that produced a dump.

Example
-------
    PYTHONPATH=. python3 scripts/bt4_policy_dump.py \
        --rows data/audit_set_v1.jsonl --out data/lc0/bt4_policy_dump.jsonl \
        --batch-size 128 --threads 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
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
from chess_anti_engine.encoding.plane_decode import (
    decode_ep_square,
    decode_step0_bitboards,
)
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.leela_index import (
    compact_index_for_move,
    leela_index_for_move,
)

DEFAULT_ONNX = "data/lc0/onnx/BT4-it332-vanilla-winner.onnx"
# The 112-plane LC0-canonical layout. `lc0_root_legacy_meta` packs an
# en-passant file into plane 110 and is NOT what an LC0 net reads.
HISTORY_ENCODING = "lc0_root"
# What production actually stores (configs/pbt2_small.yaml: input_history_encoding
# + input_extra_features: v2_threats). Its first 112 planes are the LC0 block and
# carry TRUE history, which is the whole point of the shard path.
SHARD_HISTORY_ENCODING = "lc0_root_legacy_meta"
# Files whose content defines the remap; pinned per-dump so a cache can never
# be silently attributed to the wrong revision of it.
REMAP_SOURCES = (
    "chess_anti_engine/moves/leela_index.py",
    "chess_anti_engine/moves/lc0_1858_movestrs.py",
    "chess_anti_engine/encoding/lc0.py",
)


# --------------------------------------------------------------- input rows
def iter_rows(path: Path) -> Iterator[tuple[str, str]]:
    """``(key, fen)`` pairs from a JSONL row list or a plain one-FEN-per-line file.

    ``key`` is the dump's identity (echoed to the output); ``fen`` is what the
    board is built from. They differ deliberately.

    ⚑ The audit set's 4-field ``key`` has NO halfmove clock, so a board built
    from it carries rule50 = 0 — and rule50 is plane 109 of the LC0 input, a
    plane BT4 reads. Measured on 500 audit rows: keying off ``key`` instead of
    ``fen`` flips BT4's top-1 on 5 of them (1.0%), every one a position with a
    large halfmove clock (23, 26, 32, 89, 92) where BT4 correctly plays
    differently near the 50-move boundary. So the FULL fen is preferred for the
    board whenever the row carries one, while the output stays keyed by ``key``
    for joinability with the audit set. On rows where both describe the same
    clock this script and ``scripts/foreign_net_audit.py`` agree exactly.
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
                yield str(key), str(rec.get("fen") or key)
            else:
                yield line, line


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


def board_from_stored_x(x_row: np.ndarray, planes: np.ndarray) -> chess.Board:
    """Rebuild the position from one stored replay row.

    Replay shards hold encoded planes, NOT FENs, so the position has to come
    back out of the tensor. Decoding is side-to-move canonical: the result is
    always White to move, which is the frame the planes were written in and the
    frame BT4's policy head speaks.
    """
    bbs = decode_step0_bitboards(x_row[None])[0]
    board = chess.Board(None)
    for color, offset in ((chess.WHITE, 0), (chess.BLACK, 6)):
        for pt_idx in range(6):
            piece = chess.Piece(pt_idx + 1, color)  # PAWN..KING == 1..6
            for sq in chess.scan_forward(int(bbs[offset + pt_idx])):
                board.set_piece_at(sq, piece)
    board.turn = chess.WHITE
    ep = decode_ep_square(x_row, SHARD_HISTORY_ENCODING)
    board.ep_square = ep if ep >= 0 else None
    board.castling_rights = shard_castling_rights(planes, board)
    board.halfmove_clock = int(min(100.0, float(planes[LC0_FULL.root_metadata_base + 5][0, 0])))
    return board


def iter_shard_rows(
    dirs: Sequence[str], limit: int,
) -> Iterator[tuple[str, chess.Board, np.ndarray, np.ndarray]]:
    """``(key, board, lc0_planes, legal_mask)`` for each stored replay row.

    The key is the decoded side-to-move-canonical FEN, which is the only
    joinable identity a shard row has -- the shards carry ``game_id`` and
    ``ply_index`` but no position string.
    """
    import zarr

    n = 0
    for directory in dirs:
        for shard in sorted(Path(directory).glob("*.zarr")):
            group = zarr.open(str(shard), mode="r")
            xs = np.asarray(group["x"][:])
            masks = np.asarray(group["legal_mask"][:])
            planes_all = x_to_lc0_planes(
                xs, input_history_encoding=SHARD_HISTORY_ENCODING,
            )
            for row in range(xs.shape[0]):
                board = board_from_stored_x(xs[row], planes_all[row])
                yield board.fen(), board, planes_all[row], masks[row]
                n += 1
                if limit and n >= limit:
                    return


def load_done_keys(out_path: Path) -> set[str]:
    """Keys already dumped, for --resume. Ignores a torn trailing line."""
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
            if key is not None:
                done.add(str(key))
    return done


# ------------------------------------------------------------------ session
def open_session(
    onnx: str, *, gpu_mem_gb: float, threads: int,
) -> tuple[Any, str, np.dtype[Any], list[str]]:
    import onnxruntime as ort

    providers: list[Any] = []
    if gpu_mem_gb > 0 and "CUDAExecutionProvider" in ort.get_available_providers():
        # ⚑ A bare provider NAME cannot carry gpu_mem_limit, and ORT allocates
        # through its own CUDA arena that torch's memory fraction does not
        # bound — so an uncapped session here would sit on the trainer's GPU.
        providers.append((
            "CUDAExecutionProvider",
            {"device_id": 0, "gpu_mem_limit": int(gpu_mem_gb * 1024 ** 3)},
        ))
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
    """
    names = [o.name for o in sess.get_outputs()]
    if policy_output:
        return names.index(policy_output)
    widths = [
        (o.shape[-1] if isinstance(o.shape[-1], int) else -1) for o in sess.get_outputs()
    ]
    exact = [i for i, w in enumerate(widths) if w == COMPACT_POLICY_SIZE]
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

    done = load_done_keys(out_path) if args.resume else set()
    shard_dirs = shard_source_dirs(args.input_source)
    # `items` is (key, board, lc0 planes or None). None means "build the planes
    # from the FEN", which is the history-less path; the shard path supplies
    # planes that carry TRUE history and are never rebuilt.
    items: list[tuple[str, chess.Board, np.ndarray | None]] = []
    seen: set[str] = set()
    if shard_dirs:
        for key, board, planes, mask in iter_shard_rows(shard_dirs, 0):
            if key in seen or key in done:
                continue
            if not any(board.legal_moves):
                continue
            if args.check_legal_mask:
                ours = {compact_index_for_move(board, m) for m in board.legal_moves}
                if ours != set(np.flatnonzero(mask > 0).tolist()):
                    raise SystemExit(
                        f"decoded legal moves disagree with the shard's stored "
                        f"legal_mask at {key}; the plane decode is wrong",
                    )
            seen.add(key)
            items.append((key, board, planes))
            if args.limit and len(items) >= args.limit:
                break
    else:
        for key, fen in iter_rows(Path(args.rows)):
            if key in seen or key in done:
                continue
            seen.add(key)
            items.append((key, chess.Board(fen), None))
            if args.limit and len(items) >= args.limit:
                break
    if done:
        print(f"[dump] resume: {len(done)} keys already present, {len(items)} to do")
    if not items:
        print("[dump] nothing to do")
        return 0
    print(f"[dump] source: {'shards ' + ','.join(shard_dirs) if shard_dirs else args.rows}"
          f" -> {len(items)} rows, history="
          f"{'TRUE (from stored x)' if shard_dirs else 'repeat-filled (FEN-only)'}")

    sess, in_name, in_dtype, providers = open_session(
        args.onnx, gpu_mem_gb=args.gpu_mem_gb, threads=args.threads,
    )
    pol_idx = resolve_policy_output(sess, args.policy_output)
    pol_name = sess.get_outputs()[pol_idx].name
    print(f"[dump] providers={providers} input={in_name}({in_dtype}) policy={pol_name}")

    header = {
        "record": "header",
        "onnx": {"path": str(args.onnx), "sha256": file_sha256(args.onnx)},
        "policy_output": pol_name,
        "providers": providers,
        "batch_size": int(args.batch_size),
        "threads": int(args.threads),
        "input": {
            "planes": 112,
            "source": "shards" if shard_dirs else "fens",
            "shard_dirs": list(shard_dirs),
            "history_encoding": (
                SHARD_HISTORY_ENCODING if shard_dirs else HISTORY_ENCODING
            ),
            "history_fill": None if shard_dirs else "repeat",
            "history_less": not shard_dirs,
        },
        "remap": remap_provenance(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    n_written = 0
    n_skipped = 0
    ent_sum = 0.0
    argmax_legal = 0
    castle_examples: list[dict[str, Any]] = []
    t0 = time.time()
    bs = int(args.batch_size)

    # Size, not `done`: a resumed file holding only a header has no keys yet,
    # and re-emitting the header there would put two of them in one dump.
    needs_header = not (out_path.exists() and out_path.stat().st_size)
    with out_path.open("a", encoding="utf-8") as fh:
        if needs_header:
            fh.write(json.dumps(header) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        for start in range(0, len(items), bs):
            chunk = items[start:start + bs]
            usable = [(k, b, pl) for k, b, pl in chunk if any(b.legal_moves)]
            n_skipped += len(chunk) - len(usable)
            if not usable:
                continue
            feats = np.stack([
                planes if planes is not None else fill_lc0_history_repeat(
                    encode_position(b, add_features=False,
                                    input_history_encoding=HISTORY_ENCODING),
                )
                for _k, b, planes in usable
            ]).astype(in_dtype, copy=False)
            raw = sess.run([pol_name], {in_name: feats})[0]
            policy = np.asarray(raw, dtype=np.float32)[:, :COMPACT_POLICY_SIZE]

            lines: list[str] = []
            for row, (key, board, _planes) in enumerate(usable):
                ucis, probs = legal_move_policy(board, policy[row])
                ent_sum += entropy_nats(probs)
                best = ucis[int(np.argmax(probs))]
                # By construction the argmax is drawn from board.legal_moves,
                # so this cannot fail unless the gather stopped being legal-only.
                assert chess.Move.from_uci(best) in board.legal_moves, (
                    f"{board.fen()}: argmax {best} is not legal"
                )
                argmax_legal += 1
                castles = castling_probability(board, ucis, probs)
                if castles and len(castle_examples) < args.castle_examples:
                    castle_examples.append({
                        "key": key, "castling": castles, "best": best,
                        "best_p": round(float(probs.max()), 4),
                    })
                lines.append(json.dumps({
                    "key": key,
                    "n_legal": len(ucis),
                    "policy": {u: round(float(p), 6)
                               for u, p in zip(ucis, probs.tolist(), strict=True)},
                }))
                n_written += 1
            fh.write("".join(line + "\n" for line in lines))
            fh.flush()
            os.fsync(fh.fileno())

            done_n = start + len(chunk)
            elapsed = time.time() - t0
            rate = done_n / elapsed if elapsed > 0 else 0.0
            eta = f" eta {(len(items) - done_n) / rate / 60:.1f} min" if rate > 0 else ""
            print(f"[dump] {done_n}/{len(items)} {rate:.1f} pos/s{eta}", flush=True)

    elapsed = time.time() - t0
    rate = n_written / elapsed if elapsed > 0 else 0.0
    print(f"[dump] wrote {n_written} rows ({n_skipped} terminal positions skipped) "
          f"in {elapsed:.1f}s = {rate:.1f} pos/s -> {out_path}")
    if n_written:
        print(f"[sanity] mean legal-move entropy: {ent_sum / n_written:.4f} nats "
              f"(banked BT4 figure ~0.970)")
        print(f"[sanity] argmax-legal fraction: {argmax_legal / n_written:.4f} "
              "(1.0 by construction)")
    for ex in castle_examples:
        print(f"[sanity] castling prior {ex['castling']} "
              f"(best {ex['best']} p={ex['best_p']}) {ex['key']}")
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
    ap.add_argument("--check-legal-mask", action="store_true", default=True,
                    help="shard mode: assert the decoded legal moves match the "
                         "shard's own legal_mask (default on)")
    ap.add_argument("--no-check-legal-mask", dest="check_legal_mask",
                    action="store_false")
    ap.add_argument("--out", required=True, help="JSONL dump path (appended to)")
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threads", type=int, default=16,
                    help="ORT intra-op threads for the CPU provider (0 = ORT default)")
    ap.add_argument("--gpu-mem-gb", type=float, default=0.0,
                    help=">0 tries CUDAExecutionProvider with this hard memory cap")
    ap.add_argument("--policy-output", default=None,
                    help="ONNX policy output name (default: the 1858-wide one)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
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
