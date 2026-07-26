#!/usr/bin/env python3
"""Blind-spot resolution / stability tracker (Level-2, offline).

Scores a blind-spot SEED set (harvester output, or any seed_board_from_line
file) with one or more checkpoints and classifies each seed as still-BLIND (the
net's value still thinks the position is fine) or RESOLVED (the net now reads it
as losing). It is two things at once:

  * the REMOVAL policy — a seed RESOLVED across enough checkpoints has done its
    job (the value head is fixed there) and can be retired from the seed list;
  * the STABILITY read — the still-BLIND count over a checkpoint sequence is the
    flywheel health metric: shrinking = the net is fixing them (winning), flat =
    whack-a-mole (capacity-bound), growing = harvest too loose.

Each seed's terminal is the position the net faced (DeepFin to move), so the
value ``net_q = W-L`` is DeepFin's own expected value there. RESOLVED when
``net_q < --resolved-below`` (net no longer thinks it is fine). Seeds are scored
WITH their real history (via seed_board_from_line), matching how they train.

Usage:
  PYTHONPATH=. python3 scripts/blindspot_resolution.py --seeds harvest.txt \
      --checkpoint A.pt B.pt C.pt          # trajectory across checkpoints
  PYTHONPATH=. python3 scripts/blindspot_resolution.py --seeds harvest.txt \
      --checkpoint cur.pt --retire-below -0.3 --out retire.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResolutionSummary:
    n: int
    blind: int          # net still thinks fine (net_q > resolved_below)
    resolved: int       # net now reads losing (net_q <= resolved_below)

    @property
    def resolved_frac(self) -> float:
        return self.resolved / self.n if self.n else 0.0


def classify(net_q: np.ndarray, *, resolved_below: float) -> ResolutionSummary:
    """Pure: split scored seeds into still-BLIND vs RESOLVED at a threshold."""
    q = np.asarray(net_q, dtype=np.float64)
    resolved = int((q <= resolved_below).sum())
    return ResolutionSummary(n=int(q.size), blind=int(q.size) - resolved, resolved=resolved)


def score_seeds(checkpoint: str, seed_lines: list[str], *, device: str,
                gpu_mem_fraction: float | None, batch_size: int = 256) -> np.ndarray:
    """DeepFin-POV value ``net_q = W-L`` per seed terminal, scored WITH history."""
    import torch

    from chess_anti_engine.encoding import model_encoding_kwargs
    from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
    from chess_anti_engine.inference import LocalModelEvaluator
    from chess_anti_engine.selfplay.opening import seed_board_from_line
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    if device.startswith("cuda") and gpu_mem_fraction:
        idx = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(float(gpu_mem_fraction), idx)

    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    enc_kwargs = model_encoding_kwargs(model)
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    ev = LocalModelEvaluator(model, device=device)

    encs, rels = [], []
    for line in seed_lines:
        # Score the LOST terminal (blame-dropped plies re-appended), with the
        # seed's real move history.
        board = seed_board_from_line(reconstruct_lost_line(line))
        cb = CBoard.from_board(board)
        encs.append(encode_cboard(cb, **enc_kwargs))
        if use_rel:
            rels.append(cb.compute_relations())
    enc_arr = np.stack(encs)
    rel_arr = np.stack(rels) if use_rel else None
    # Forward in chunks so a few-thousand-seed set does not OOM one batch at the
    # default 0.15 GPU fraction.
    out = np.empty(len(encs), dtype=np.float64)
    for s in range(0, len(encs), batch_size):
        e = slice(s, s + batch_size)
        with torch.no_grad():
            _, wdl = ev.evaluate_encoded(
                enc_arr[e], relations=rel_arr[e] if rel_arr is not None else None)
        # evaluate_encoded returns raw wdl LOGITS; softmax unconditionally (a
        # conditional guard can skip it when logits happen to ~sum to 1).
        wdl = np.asarray(wdl, dtype=np.float64)
        w = wdl - wdl.max(axis=1, keepdims=True)
        wdl = np.exp(w) / np.exp(w).sum(axis=1, keepdims=True)
        out[e] = wdl[:, 0] - wdl[:, 2]  # DeepFin (side-to-move) expected value
    return out


_DROPPED_RE = re.compile(r"\bdropped=([a-h1-8nbrqNBRQ,]+)")


def reconstruct_lost_line(raw: str) -> str:
    """Scoring line whose terminal is the ORIGINAL deep-LOST position.

    Blame-backed gate emissions (scripts/blindspot_deepsf_gate.py) start
    games at the earlier DECISION POINT and stash the trimmed plies in the
    line comment (``dropped=uci,uci,...``). Scoring/retirement must evaluate
    the LOST terminal — a correct net reads the decision point itself as
    ~holdable forever, so scoring the emitted terminal would report backed
    seeds as permanently BLIND and never retire them (Codex #134). Re-append
    the dropped plies (real history preserved); comment-less or un-backed
    lines pass through unchanged (minus the comment).
    """
    body, _, comment = raw.partition("#")
    body = body.strip()
    if not body:
        return ""
    m = _DROPPED_RE.search(comment)
    if m is None:
        return body
    moves = " ".join(m.group(1).split(","))
    if "|" in body:
        return f"{body} {moves}"
    return f"{body} | {moves}"


def load_seed_lines(path: str) -> list[str]:
    """Seed lines VERBATIM (comments kept — they carry blame-backup metadata
    that must survive the retire step's list rewrites), blanks and pure
    comment lines skipped. Unusable lines are SKIPPED with a warning exactly
    like the live loader (_load_fen_list) — one bad line must not abort a
    resolution read or make the retire step fail closed; the RECONSTRUCTED
    form is what gets scored, so it is what gets validated. Raises if the
    file yields zero usable seeds (mirrors the live fail-fast contract)."""
    from chess_anti_engine.selfplay.opening import _fen_reject_reason, seed_board_from_line

    lines: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.rstrip("\n")
            body = raw.partition("#")[0].strip()
            if not body:
                continue
            # The BODY must pass the LIVE loader's reject predicate — a line
            # workers skip must not be scored/streaked here (the monitor would
            # otherwise retire a seed that was never trained). The
            # RECONSTRUCTED form (lost terminal) additionally has to parse,
            # since that is the position score_seeds evaluates.
            reason = _fen_reject_reason(body)
            if reason is not None:
                print(f"[seeds] skipping unusable line ({reason}): {raw[:80]}",
                      file=sys.stderr)
                continue
            try:
                seed_board_from_line(reconstruct_lost_line(raw))
            except ValueError as e:
                print(f"[seeds] skipping line with bad dropped= history ({e}): {raw[:80]}",
                      file=sys.stderr)
                continue
            lines.append(raw)
    if not lines:
        raise ValueError(f"seed list {path!r} contains no usable seeds")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", required=True, help="seed file (harvester output / any seed list)")
    ap.add_argument("--checkpoint", nargs="+", required=True,
                    help="one or more checkpoints (trajectory in order given)")
    ap.add_argument("--resolved-below", type=float, default=0.0,
                    help="net_q at/below which a seed is RESOLVED (default 0.0: net no "
                         "longer thinks it is ahead; -0.4 ~ the panel AWARE bar)")
    ap.add_argument("--retire-below", type=float, default=None,
                    help="write seeds with net_q <= this (fully fixed) to --out for removal")
    ap.add_argument("--out", default=None, help="retire list output (with --retire-below)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=256, help="forward-pass chunk (bounds GPU mem)")
    args = ap.parse_args()

    seeds = load_seed_lines(args.seeds)
    print(f"[resolution] {len(seeds)} seeds from {args.seeds}; "
          f"RESOLVED when net_q <= {args.resolved_below}\n")
    print(f"{'checkpoint':<40} {'blind':>6} {'resolved':>9} {'resolved%':>10}")
    print("-" * 68)
    last_q: np.ndarray | None = None
    for ckpt in args.checkpoint:
        q = score_seeds(ckpt, seeds, device=args.device,
                        gpu_mem_fraction=args.gpu_mem_fraction, batch_size=args.batch_size)
        s = classify(q, resolved_below=args.resolved_below)
        print(f"{ckpt.split('/')[-1]:<40} {s.blind:>6} {s.resolved:>9} "
              f"{100 * s.resolved_frac:>9.1f}%")
        last_q = q

    if args.retire_below is not None and args.out and last_q is not None:
        retire = [seeds[i] for i in range(len(seeds)) if last_q[i] <= args.retire_below]
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("# seeds the value head now reads correctly (retire from the list)\n")
            for line in retire:
                fh.write(line + "\n")
        print(f"\n[resolution] {len(retire)} seeds fixed (net_q <= {args.retire_below}) "
              f"-> {args.out} (retire from the seed list)")
    print("\nblind trajectory over the checkpoints = the flywheel health read: "
          "shrinking = fixing them, flat = whack-a-mole (capacity-bound), growing "
          "= harvest too loose.")


if __name__ == "__main__":
    main()
