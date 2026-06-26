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
(white to move), so no mirroring is needed. BT4's 1858 policy is mapped to the
4672 full space via COMPACT_TO_FULL_POLICY and gathered at the legal indices —
the same move space the net audit uses.

Usage:
  PYTHONPATH=. python3 scripts/bt4_audit.py \
    --onnx data/lc0/onnx/BT4-it332-vanilla-winner.onnx \
    --cache-out data/lc0/bt4_audit_cache.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.eval.audit import (
    expected_and_top1_regret,
    legal_full_indices,
    move_regrets,
    parse_audit_record,
)
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.lc0_1858_movestrs import LC0_1858_UCI_TO_IDX

PHASE_NAMES = {0: "endgame", 1: "middlegame", 2: "opening"}
SOURCE_NAMES = {0: "selfplay", 1: "curriculum"}


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


def _session(onnx: str, gpu_mem_gb: float):
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


def _criticality(move_cp: dict[str, float]) -> float:
    listed = sorted(move_cp.values(), reverse=True)
    return float(listed[0] - listed[1]) if len(listed) >= 2 else float("inf")


def _fill_history_repeat(enc: np.ndarray) -> np.ndarray:
    """Replicate lc0's empty-history fill (encoder.cc: ``history[idx<0?0:idx]``).

    For a single FEN with no move history, lc0 repeats the CURRENT position into
    every empty history slot — identically, NOT with an alternating per-ply
    perspective flip (verified: identical-repeat → WDL Brier 0.026 vs deep-SF;
    zeroed → 1.08; alternating-flip → 0.83). The value head is history-sensitive
    and reads garbage from a zeroed history; policy is more robust but uses the
    same input. Planes 0..103 are 8 slots × 13 (12 piece + 1 repetition); slot 0
    is the live position. Metadata planes 104..111 are untouched.
    """
    out = enc.copy()
    for s in range(1, 8):
        out[13 * s:13 * s + 13] = enc[0:13]
    return out


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
        e = encode_position(b, add_features=False, input_history_encoding=args.history)
        return _fill_history_repeat(e) if args.history_fill == "repeat" else e

    feats = np.stack([_enc(b) for b in boards]).astype(np.float32)

    pol = np.empty((len(boards), COMPACT_POLICY_SIZE), dtype=np.float32)
    wdl = np.empty((len(boards), 3), dtype=np.float32)
    bs = int(args.batch_size)
    for s in range(0, len(boards), bs):
        out = sess.run(None, {in_name: feats[s:s + bs]})
        pol[s:s + bs] = np.asarray(out[0])[:, :COMPACT_POLICY_SIZE]
        wdl[s:s + bs] = np.asarray(out[1])
        if s % (bs * 8) == 0:
            print(f"[bt4] {s + min(bs, len(boards) - s)}/{len(boards)}")

    # Aggregate by group; also keep per-criticality-bucket regret.
    GAP_EDGES = [20.0, 50.0, 100.0]  # cp: quiet < 20 <= soft < 50 <= sharp < 100 <= decisive
    bucket_names = ["quiet(<20)", "soft(20-50)", "sharp(50-100)", "decisive(>=100)"]

    def bucket(gap: float) -> int:
        for i, e in enumerate(GAP_EDGES):
            if gap < e:
                return i
        return len(GAP_EDGES)

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

        gap = _criticality(pos.move_cp)
        bname = bucket_names[bucket(gap)]
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

    groups = (["overall", *PHASE_NAMES.values(), *SOURCE_NAMES.values(), *bucket_names])
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
