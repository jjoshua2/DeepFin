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


def _piece_count(fen: str) -> int:
    """Total pieces on the board (both kings included) from a FEN.

    Used to exclude Syzygy-range positions from the ruler: with <=7 pieces the
    engine plays via tablebase (selfplay TB-optimal moves, UCI root probe), so
    the net's value there never decides a real move — scoring it measures value
    quality we do not use. See chess_anti_engine/tablebase.py (<=7, no castling).
    """
    return sum(1 for c in fen.split(" ", 1)[0] if c.isalpha())


def value_1ply_regret(
    *, checkpoint: str, positions, device: str, batch_size: int, pos_chunk: int,
) -> tuple[float, dict[int, float], np.ndarray]:
    """Return (overall mean regret cp, {phase: mean cp}, per-position regrets).

    The third element is the per-position top-1 regret array (NaN for
    terminal roots), aligned with ``positions`` — dump it for paired
    checkpoint comparisons (scripts/paired_compare.py).
    """
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
    return overall, per_phase, top1


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
    ap.add_argument("--dump-per-position", default=None, metavar="PATH",
                    help="write per-position JSONL (fen, phase, top1 regret cp) for "
                         "paired checkpoint comparison via scripts/paired_compare.py")
    ap.add_argument("--min-pieces", type=int, default=8,
                    help="Minimum total pieces to score. DEFAULT 8 EXCLUDES "
                         "Syzygy-range (<=7-man) positions: the engine plays those via "
                         "tablebase (see tablebase.py), so the net's value there never "
                         "decides a real move and scoring it measures value quality we "
                         "never use — the default ruler now tracks PLAY-RELEVANT value. "
                         "Pass --min-pieces 0 to score ALL positions (pre-2026-07-20 "
                         "behavior), required to reproduce historical full-set numbers "
                         "(e.g. the 70-76cp band / BT4=43cp ref were full-set).")
    args = ap.parse_args()

    if args.gpu_mem_fraction is not None and str(args.device).startswith("cuda"):
        # Cap the SELECTED device (e.g. cuda:1), not just the current default one.
        # torch rejects an index-less device, so resolve bare "cuda" -> current index.
        dev_idx = (int(args.device.split(":", 1)[1]) if ":" in args.device
                   else torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_fraction), device=dev_idx)
        print(f"[value-regret] GPU memory capped at fraction {args.gpu_mem_fraction} "
              f"on cuda:{dev_idx}")

    positions = load_audit_set(args.audit_set)
    # Slice the canonical max_positions subset (e.g. v1-2k = first 2000) FIRST,
    # then drop TB-range — so --min-pieces filters WITHIN the standard subset
    # (2000 -> 1723) rather than pulling later strata forward from the full set.
    if args.max_positions > 0:
        positions = positions[: args.max_positions]
    if args.min_pieces > 0:
        n_before = len(positions)
        positions = [p for p in positions if _piece_count(p.fen) >= args.min_pieces]
        print(f"[value-regret] min-pieces>={args.min_pieces}: dropped "
              f"{n_before - len(positions)} TB-range positions ({len(positions)} kept)")
    print(f"[value-regret] {len(positions)} positions from {args.audit_set}")
    overall, per_phase, per_position = value_1ply_regret(
        checkpoint=args.checkpoint, positions=positions, device=args.device,
        batch_size=args.batch_size, pos_chunk=args.pos_chunk,
    )
    if args.dump_per_position:
        import json
        with open(args.dump_per_position, "w", encoding="utf-8") as f:
            for pos, r in zip(positions, per_position, strict=True):
                f.write(json.dumps({
                    "fen": pos.fen, "phase": int(pos.phase),
                    "value": None if np.isnan(r) else float(r),
                }) + "\n")
        print(f"[value-regret] per-position dump -> {args.dump_per_position}")

    ruler = "TB-excluded" if args.min_pieces > 0 else "FULL-SET (incl. TB)"
    print(f"\n=== value-head 1-ply deep-SF regret @ {args.checkpoint} ===")
    print(f"  ruler: {ruler}"
          + (f" (>={args.min_pieces}-man)" if args.min_pieces > 0 else "")
          + f"  |  OVERALL {overall:.1f} cp (n={len(positions)})")
    for ph in sorted(per_phase):
        print(f"  {PHASE_NAMES[ph]:11s} {per_phase[ph]:.1f} cp")
    # Tail view: the audit mean is tail-dominated (median ~10cp vs mean ~75) and
    # the >300cp blowups are the Cheese single-collapse failure mode, so the
    # tail is reported alongside the mean rather than hidden inside it.
    finite = per_position[~np.isnan(per_position)]
    if finite.size:
        print(f"  TAIL {'all':11s} med={float(np.median(finite)):6.1f} "
              f"P90={float(np.percentile(finite, 90)):7.1f} "
              f">100cp={100 * float((finite > 100).mean()):5.1f}% "
              f">300cp={100 * float((finite > 300).mean()):5.1f}%")
        pos_phases = np.array([int(p.phase) for p in positions])
        for ph in sorted(per_phase):
            v = per_position[pos_phases == ph]
            v = v[~np.isnan(v)]
            if v.size:
                print(f"  TAIL {PHASE_NAMES[ph]:11s} med={float(np.median(v)):6.1f} "
                      f"P90={float(np.percentile(v, 90)):7.1f} "
                      f">100cp={100 * float((v > 100).mean()):5.1f}% "
                      f">300cp={100 * float((v > 300).mean()):5.1f}%")


if __name__ == "__main__":
    main()
