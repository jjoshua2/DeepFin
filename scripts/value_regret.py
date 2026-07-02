#!/usr/bin/env python3
"""Value-head STRENGTH via 1-ply deep-SF regret (lc0-style value match, offline).

Brier scores value-head CALIBRATION and is fooled by chronic under-confidence —
a head can be badly hedged yet still RANK moves correctly, and ranking is what
matters for play. This scores the value head as a 1-ply player: for every frozen
audit position, evaluate the value head on EVERY legal child one ply deep, pick
the move that maximises the parent-mover's value (the move lc0's value-match
would play), and score THAT move's deep-SF regret in cp. The number is directly
comparable to the policy top-1 regret in ``scripts/audit_targets.py``.

Parent value of a move = child P(loss) - child P(win) from the CHILD's
side-to-move POV (the child is the opponent to move, so flip). Terminal children:
checkmate-by-the-move = best (parent wins); any other game-over = draw.

Forward-only, GPU-light — safe to run concurrent with training. Re-run on later
checkpoints to track whether value RANKING improves with training (which Brier
cannot show). See docs/eval_protocol.md and scripts/audit_targets.py.
"""
from __future__ import annotations

import argparse

import chess
import numpy as np
import torch

from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
from chess_anti_engine.eval.audit import PHASE_NAMES, load_audit_set, move_regrets
from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


def _softmax_rows(a: np.ndarray) -> np.ndarray:
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def value_1ply_regret(
    *, checkpoint: str, positions, device: str, batch_size: int, pos_chunk: int,
) -> tuple[float, dict[int, float]]:
    """Return (overall mean value-1ply regret cp, {phase: mean cp})."""
    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    # Dynamic-relation checkpoints apply their attention bias only when the
    # relation tensor is passed; without it we'd silently score a relation-less
    # model. Carry relations exactly like scripts/audit_targets.py does.
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    ev = LocalModelEvaluator(model, device=device)

    top1 = np.full(len(positions), np.nan, dtype=np.float64)
    phases = np.array([p.phase for p in positions], dtype=np.int64)

    for cs in range(0, len(positions), pos_chunk):
        chunk = positions[cs:cs + pos_chunk]
        encs: list[np.ndarray] = []
        rels: list[np.ndarray] = []
        owner: list[tuple[int, int]] = []          # (local_pos_idx, move_idx)
        recs: list[tuple[np.ndarray, np.ndarray]] = []  # (regrets, parent_val)
        for lpi, pos in enumerate(chunk):
            board = chess.Board(pos.fen)
            legal = list(board.legal_moves)
            regrets = move_regrets(pos, [m.uci() for m in legal])
            pv = np.full(len(legal), -np.inf, dtype=np.float64)
            for mi, m in enumerate(legal):
                board.push(m)
                if board.is_checkmate():
                    pv[mi] = 2.0                    # parent mates -> best
                elif board.is_game_over():
                    pv[mi] = 0.0                    # stalemate / draw
                else:
                    cb = CBoard.from_board(board)
                    encs.append(encode_cboard(
                        cb, input_history_encoding=hist, input_extra_features=extra))
                    if use_rel:
                        rels.append(cb.compute_relations())
                    owner.append((lpi, mi))
                board.pop()
            recs.append((regrets, pv))
        for s in range(0, len(encs), batch_size):
            xs = np.stack(encs[s:s + batch_size])
            rel_batch = np.stack(rels[s:s + batch_size]) if use_rel else None
            with torch.no_grad():
                _, wdl = ev.evaluate_encoded(xs, relations=rel_batch)
            wdl = np.asarray(wdl, dtype=np.float64)
            if not np.allclose(wdl.sum(axis=1), 1.0, atol=1e-3):
                wdl = _softmax_rows(wdl)
            for j in range(len(xs)):
                lpi, mi = owner[s + j]
                recs[lpi][1][mi] = wdl[j, 2] - wdl[j, 0]   # P(loss)-P(win), child POV
        for lpi, (regrets, pv) in enumerate(recs):
            if pv.size == 0:
                continue  # terminal root (no legal moves) — leave top1 as NaN
            top1[cs + lpi] = float(regrets[int(np.argmax(pv))])

    overall = float(np.nanmean(top1))
    per_phase = {
        ph: float(np.nanmean(top1[phases == ph]))
        for ph in range(3) if (phases == ph).any()
    }
    return overall, per_phase


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="trainer.pt or checkpoint dir")
    ap.add_argument("--audit-set", default="data/audit_set_v1.jsonl")
    ap.add_argument("--max-positions", type=int, default=0,
                    help="0 (default) = score ALL rows. The audit set is written in "
                         "sorted phase/source strata, so a prefix slice is biased "
                         "(over-weights early strata, can omit openings) and not "
                         "comparable to audit_targets.py — only limit for a quick check.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--pos-chunk", type=int, default=128,
                    help="positions buffered per forward pass group (bounds RAM)")
    ap.add_argument("--gpu-mem-fraction", type=float, default=None,
                    help="cap this process's CUDA memory fraction (coexist with training)")
    args = ap.parse_args()

    if args.gpu_mem_fraction is not None and str(args.device).startswith("cuda"):
        # Cap the SELECTED device (e.g. cuda:1), not just the current default one.
        # torch rejects an index-less device, so resolve "cuda" -> current index.
        _dev = torch.device(args.device)
        _idx = (  # torch stubs type .index as int, but it IS None for a bare "cuda"
            _dev.index if _dev.index is not None  # pyright: ignore[reportUnnecessaryComparison]
            else torch.cuda.current_device()
        )
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_fraction), device=_idx)
        print(f"[value-regret] GPU memory capped at fraction {args.gpu_mem_fraction} "
              f"on cuda:{_idx}")

    positions = load_audit_set(args.audit_set)
    if args.max_positions > 0:
        positions = positions[: args.max_positions]
    print(f"[value-regret] {len(positions)} positions from {args.audit_set}")
    overall, per_phase = value_1ply_regret(
        checkpoint=args.checkpoint, positions=positions, device=args.device,
        batch_size=args.batch_size, pos_chunk=args.pos_chunk,
    )
    print(f"\n=== value-head 1-ply deep-SF regret @ {args.checkpoint} ===")
    print(f"  OVERALL {overall:.1f} cp (n={len(positions)})")
    for ph in sorted(per_phase):
        print(f"  {PHASE_NAMES[ph]:11s} {per_phase[ph]:.1f} cp")


if __name__ == "__main__":
    main()
