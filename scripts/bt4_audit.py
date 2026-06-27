#!/usr/bin/env python3
"""Score LC0 BT4 on the frozen deep-SF audit set and cache its outputs.

BT4 is a ~191M-param Leela net (far stronger than our ~46M model). Running it
on the same audit positions gives two things:

  1. A strong-reference regret curve vs the frozen deep-SF labels — in
     particular, how much regret BT4 *itself* bleeds in quiet positions. If a
     3500-Elo net also bleeds ~70cp where SF's "best" is near-arbitrary, that
     regret is a label-noise floor, not our model's weakness.
  2. A persistent per-position cache (bestmove / top-k policy / WDL) so we can
     compare our net to BT4 DIRECTLY (move agreement, policy divergence),
     not only through SF regret. Reusable for later analyses.

Input encoding is canonical-LC0 (`--history lc0_root`, validated: startpos →
sensible openings + correct WDL). Audit FENs are side-to-move canonical
(white to move), so no mirroring is needed. BT4's 1858 policy is gathered
directly in 1858 space at the legal moves via the canonical Leela index map
(``LC0_1858_UCI_TO_IDX`` / ``_leela_idxs``) — NOT through the project's compact
1858 ordering, which differs from Leela's on nearly every index.

Usage:
  PYTHONPATH=. python3 scripts/bt4_audit.py \
    --onnx data/lc0/onnx/BT4-it332-vanilla-winner.onnx \
    --cache-out data/lc0/bt4_audit_cache.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import fill_lc0_history_repeat
from chess_anti_engine.eval.audit import (
    CRITICALITY_BUCKET_NAMES,
    PHASE_NAMES,
    SOURCE_NAMES,
    criticality_bucket,
    criticality_gap,
    expected_and_top1_regret,
    legal_full_indices,
    move_regrets,
    parse_audit_record,
)
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.lc0_1858_movestrs import LC0_1858_UCI_TO_IDX


def _mirror_uci(uci: str) -> str:
    """Vertically flip a UCI move (rank 1<->8). LC0 policy is White-POV, so a
    black-to-move position must mirror its moves before the 1858 lookup."""
    def flip(sq: str) -> str:
        return sq[0] + str(9 - int(sq[1]))
    return flip(uci[:2]) + flip(uci[2:4]) + uci[4:]


def _leela_idxs(board: chess.Board, ucis: list[str]) -> np.ndarray:
    """Map each legal UCI to its canonical LC0 1858 policy slot (-1 if absent).
    Mirrors moves for black-to-move boards (audit boards are white-canonical)."""
    flip = not board.turn  # board.turn True == white
    out = np.empty(len(ucis), dtype=np.int64)
    for i, u in enumerate(ucis):
        key = _mirror_uci(u) if flip else u
        out[i] = LC0_1858_UCI_TO_IDX.get(key, -1)
    return out


def _session(onnx: str, gpu_mem_gb: float) -> tuple[Any, str]:
    import onnxruntime as ort

    providers: list = []
    if gpu_mem_gb > 0:
        providers.append(
            ("CUDAExecutionProvider",
             {"device_id": 0, "gpu_mem_limit": int(gpu_mem_gb * 1024 ** 3)}),
        )
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(onnx, providers=providers)
    return sess, sess.get_inputs()[0].name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"))
    ap.add_argument("--onnx", type=str,
                    default="data/lc0/onnx/BT4-it332-vanilla-winner.onnx")
    ap.add_argument("--history", default="lc0_root",
                    help="LC0 input history layout (lc0_root = canonical, validated)")
    ap.add_argument("--history-fill", choices=("repeat", "zero"), default="repeat",
                    help="empty-history fill: 'repeat' the current position into "
                         "all 8 slots (lc0's actual behavior, required for a sane "
                         "value head) or 'zero' (legacy, breaks BT4's WDL)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--gpu-mem-gb", type=float, default=4.0,
                    help="hard cap on the CUDA arena so a concurrent trainer "
                         "is not faulted; 0 forces CPU")
    ap.add_argument("--topk", type=int, default=5, help="policy moves cached per position")
    ap.add_argument("--cache-out", type=Path, default=Path("data/lc0/bt4_audit_cache.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs/bt4_audit.md"))
    ap.add_argument("--max-positions", type=int, default=0)
    args = ap.parse_args()

    positions = [parse_audit_record(ln) for ln in
                 args.audit_set.read_text().splitlines() if ln.strip()]
    if args.max_positions > 0:
        positions = positions[: args.max_positions]
    print(f"[bt4] {len(positions)} positions from {args.audit_set}")

    sess, in_name = _session(args.onnx, args.gpu_mem_gb)
    print(f"[bt4] providers: {sess.get_providers()}")

    boards = [chess.Board(p.fen) for p in positions]

    def _enc(b: chess.Board) -> np.ndarray:
        # `lc0_root` emits LC0-canonical metadata in planes 104..111 — verified
        # directly against LC0's layout: side-relative castling (us-Q, us-K,
        # them-Q, them-K, flipped for black-to-move), side-to-move flag (108),
        # raw rule50 count (109), zeros (110), ones (111). It writes NO
        # en-passant plane, which is correct: BT4's T-format has no EP plane and
        # conveys en passant through the move history. The single caveat is that
        # `_fill_history_repeat` fakes the 7 empty history frames, so for the
        # ~1% of audit FENs whose key move is an en-passant capture, BT4 can't
        # see the double-push that would make it available (a single-FEN
        # limitation, not a metadata bug). The `*_legacy_meta` variant DOES pack
        # an EP file into plane 110 and must NOT be used here.
        e = encode_position(b, add_features=False, input_history_encoding=args.history)
        return fill_lc0_history_repeat(e) if args.history_fill == "repeat" else e

    feats = np.stack([_enc(b) for b in boards]).astype(np.float32)

    pol = np.empty((len(boards), COMPACT_POLICY_SIZE), dtype=np.float32)
    wdl = np.empty((len(boards), 3), dtype=np.float32)
    bs = int(args.batch_size)
    pol_idx = wdl_idx = -1
    for s in range(0, len(boards), bs):
        out = [np.asarray(o) for o in sess.run(None, {in_name: feats[s:s + bs]})]
        if pol_idx < 0:
            # Pick outputs by width, not position: some LC0/BT4 ONNX graphs emit
            # WDL (3-wide) before policy (1858-wide), so out[0] is not always
            # the policy tensor.
            widths = [a.shape[-1] for a in out]
            pol_idx = int(np.argmax(widths))
            wdl_idx = next((i for i, w in enumerate(widths) if w == 3), -1)
            if wdl_idx < 0:
                raise SystemExit(
                    "no 3-wide WDL output found; ONNX outputs have widths "
                    f"{widths}. Pass an LC0/Ceres net with a WDL value head."
                )
        pol[s:s + bs] = out[pol_idx][:, :COMPACT_POLICY_SIZE]
        wdl[s:s + bs] = out[wdl_idx]
        if s % (bs * 8) == 0:
            print(f"[bt4] {s + min(bs, len(boards) - s)}/{len(boards)}")

    # Aggregate by group; also keep per-criticality-bucket regret (shared edges).
    bucket_names = list(CRITICALITY_BUCKET_NAMES)

    sums: dict[str, list[float]] = {}
    cnts: dict[str, int] = {}

    def add(group: str, exp_r: float, top1_r: float) -> None:
        s = sums.setdefault(group, [0.0, 0.0])
        s[0] += exp_r
        s[1] += top1_r
        cnts[group] = cnts.get(group, 0) + 1

    cache_rows: list[dict] = []
    for i, (pos, board) in enumerate(zip(positions, boards, strict=True)):
        legal_ucis, _ = legal_full_indices(board)
        if not legal_ucis:
            continue
        regrets = move_regrets(pos, legal_ucis)
        leela = _leela_idxs(board, legal_ucis)
        lg = np.where(leela >= 0, pol[i][leela.clip(0)], -1e9).astype(np.float64)
        lg = lg - lg.max()
        p = np.exp(lg)
        p = p / p.sum()
        exp_r, top1_r = expected_and_top1_regret(p, regrets)

        gap = criticality_gap(pos.move_cp)
        bname = bucket_names[criticality_bucket(gap)]
        for g in ("overall", PHASE_NAMES[pos.phase], SOURCE_NAMES[pos.source], bname):
            add(g, exp_r, top1_r)

        order = np.argsort(-p)[: int(args.topk)]
        cache_rows.append({
            "key": pos.key, "phase": pos.phase, "source": pos.source,
            "gap_cp": gap, "best_move": legal_ucis[int(np.argmax(p))],
            "wdl": [round(float(v), 4) for v in wdl[i]],
            "exp_regret": exp_r, "top1_regret": top1_r,
            "topk": [[legal_ucis[int(o)], round(float(p[int(o)]), 4)] for o in order],
        })

    args.cache_out.parent.mkdir(parents=True, exist_ok=True)
    with args.cache_out.open("w") as fh:
        for r in cache_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[bt4] cache → {args.cache_out} ({len(cache_rows)} rows)")

    groups = (["overall", *PHASE_NAMES, *SOURCE_NAMES, *bucket_names])
    lines = [f"# BT4 audit @ {args.onnx}", "",
             f"- audit set: {args.audit_set} ({len(cache_rows)} scored)",
             f"- input history: {args.history}; policy mapped 1858→legal",
             "", "| group | E[regret] cp | top-1 cp | n |", "|---|---|---|---|"]
    for g in groups:
        if g not in cnts:
            continue
        n = cnts[g]
        lines.append(f"| {g} | {sums[g][0] / n:.1f} | {sums[g][1] / n:.1f} | {n} |")
    report = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
