#!/usr/bin/env python3
"""Blind-spot panel: value-head TAIL yardstick over frozen collapse positions.

The panel (``data/blindspot_panel_v1.jsonl``, FROZEN 2026-07-02) is the first
decisive collapse position from each of the 35 genuine on-board losses to
full-strength Cheese 3.2.1 (June 21 matches): DeepFin's move took the position
from ``sf_before`` to ``sf_after`` centipawns (DeepFin POV, deep-SF 300k-node
annotation), with ``sf_after < -150`` in every row. The 2026-07-02 autopsy
showed 80% of those losses hinged on a single such move, and the then-current
net still read 21/35 of these positions as fine — tail errors the mean-based
``value_regret`` cannot see and the RL loop does not self-correct.

For each row the checkpoint's raw WDL evaluates the position AFTER the played
move (DeepFin-POV score = P(loss)-P(win) of the side to move, flipped):
  BLIND  — net score > -0.2 while deep SF says clearly lost
  AWARE  — net score < -0.4
Baseline: 21/35 BLIND at ckpt477/478. Success for a tail-targeting change is
this count falling; the mean value_regret is judged separately.

GPU-light, safe concurrent with training. See docs/experiment_ledger.md.
"""
from __future__ import annotations

import argparse
import json
import os

import chess
import numpy as np
import torch

from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

BLIND_ABOVE = -0.2
AWARE_BELOW = -0.4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="trainer.pt or checkpoint dir")
    ap.add_argument("--panel", default="data/blindspot_panel_v1.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.15)
    ap.add_argument(
        "--dump-per-position", default=None,
        help="write {fen, phase, value} per row (value = net_after collapse "
        "severity, higher=blinder) for paired_compare.py — mirrors value_regret",
    )
    args = ap.parse_args()

    if args.device.startswith("cuda") and args.gpu_mem_fraction:
        # Cap the SELECTED device, not the current one — running the panel on
        # cuda:1 must not leave that GPU uncapped next to a live trainer.
        idx = (int(args.device.split(":", 1)[1]) if ":" in args.device
               else torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_fraction), idx)

    with open(args.panel, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if not rows:
        raise SystemExit(f"panel {args.panel} has no rows (empty or truncated file)")
    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    # Dynamic-relation checkpoints apply their attention bias only when the
    # relation tensor is passed — carry it exactly like scripts/value_regret.py.
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    ev = LocalModelEvaluator(model, device=args.device)

    encs: list[np.ndarray] = []
    rels: list[np.ndarray] = []
    for r in rows:
        cb = CBoard.from_board(chess.Board(r["fen_after"]))
        encs.append(encode_cboard(
            cb, input_history_encoding=hist, input_extra_features=extra))
        if use_rel:
            rels.append(cb.compute_relations())
    with torch.no_grad():
        _, wdl = ev.evaluate_encoded(
            np.stack(encs), relations=np.stack(rels) if use_rel else None)
    wdl = np.asarray(wdl, dtype=np.float64)
    if not np.allclose(wdl.sum(axis=1), 1.0, atol=1e-3):
        w = wdl - wdl.max(axis=1, keepdims=True)
        e = np.exp(w)
        wdl = e / e.sum(axis=1, keepdims=True)

    print(f"{'src':<12} {'rd':>3} {'ply':>4} {'sf_after':>8} {'net_after':>9}")
    blind = aware = 0
    severities: list[float] = []
    dump_recs: list[dict[str, object]] = []
    for r, w in zip(rows, wdl, strict=True):
        # side to move in fen_after is the opponent -> flip to DeepFin POV
        net_after = float(w[2] - w[0])
        blind += net_after > BLIND_ABOVE
        aware += net_after < AWARE_BELOW
        severities.append(net_after)
        # value = net_after collapse severity (higher = blinder = worse), so
        # paired_compare's "A better = lower value" convention holds; join on
        # fen_after (the scored position). phase carried when the row has it.
        rec: dict[str, object] = {"fen": r["fen_after"], "value": net_after}
        if "phase" in r:
            rec["phase"] = r["phase"]
        dump_recs.append(rec)
        tag = r["src"].split("cheeseFULL_")[-1][:10]
        print(f"{tag:<12} {r['round']:>3} {r['ply']:>4} {r['sf_after']:>8} {net_after:>9.2f}")

    sev = np.asarray(severities, dtype=np.float64)

    n = len(rows)
    # Baseline is panel-specific (v1: 21/35 @ckpt477, v2: 54/113 @ckpt500) — a
    # hardcoded label misannotates whichever panel isn't v1 (Codex review). Show
    # the matching baseline by panel filename; omit it for an unknown panel
    # rather than print a wrong anchor.
    _panel_baselines = {
        "blindspot_panel_v1.jsonl": "baseline 21/35 @ckpt477",
        "blindspot_panel_v2.jsonl": "baseline 54/113 @ckpt500",
    }
    base = _panel_baselines.get(os.path.basename(args.panel), "")
    ann = f"   [{base}]" if base else ""
    print(f"\npanel {os.path.basename(args.panel)} n={n} (deep SF says all lost after our move)")
    print(f"BLIND (net > {BLIND_ABOVE}): {blind}/{n}{ann}")
    print(f"AWARE (net < {AWARE_BELOW}): {aware}/{n}")
    print(f"in between:        {n - blind - aware}/{n}")
    # Continuous collapse severity (net_after in DeepFin POV; every position is
    # lost so lower/more-negative = the net correctly sees it). Less noisy than
    # the binary BLIND count and paired-comparable across checkpoints via
    # scripts/paired_compare.py on the --dump-per-position files.
    print(f"SEVERITY net_after: mean {sev.mean():+.3f}  median {np.median(sev):+.3f}  "
          f"P90 {np.percentile(sev, 90):+.3f}  (lower=better)")

    if args.dump_per_position:
        with open(args.dump_per_position, "w", encoding="utf-8") as fout:
            for rec in dump_recs:
                fout.write(json.dumps(rec) + "\n")
        print(f"per-position dump -> {args.dump_per_position}")


if __name__ == "__main__":
    main()
