#!/usr/bin/env python3
"""Per-chunk search trajectory on the REAL accumulating tree, for the time-management
predictor: at chunk k, can we predict whether more chunks change/improve the move?

Runs the production SearchWorker (chunked Gumbel, abort disabled) one position at a
time and snapshots root state after every chunk via the run() on_chunk hook — the exact
states the early-abort/extend decision sees. Each row carries the chosen move + its
deep-SF regret, plus the STATIC settledness (visit-gap, entropy, q-gap) and the DYNAMIC
"is the search still moving" signals (bestmove_flip / q_drift / visit_churn vs the
previous chunk). Downstream: label each chunk by whether the move changes / regret drops
in later chunks, and learn which features predict it — the abort/extend rule.

Unlike backtest_time_value.py (independent one-shot Gumbel per budget), this is the
faithful chunk-accumulating path, so the predictor learned here applies directly to the
gate. Slower (single board), so run a few hundred positions.

Usage:
  PYTHONPATH=. python3 scripts/backtest_chunk_trajectory.py --checkpoint <ckpt> \\
    --device cuda --chunk-sims 2048 --max-chunks 8 --max-positions 200
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import chess
import numpy as np

from chess_anti_engine.eval.audit import legal_full_indices, load_audit_set, move_regrets
from chess_anti_engine.mcts.gumbel import PLAY_SEARCH_DEFAULTS
from chess_anti_engine.moves import index_to_move
from scripts.backtest_time_value import _stratified


def _entropy(shares: np.ndarray) -> float:
    p = shares[shares > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(prog="backtest_chunk_trajectory")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"))
    ap.add_argument("--chunk-sims", type=int, default=2048)
    ap.add_argument("--max-chunks", type=int, default=8, help="chunks per position (8 -> up to 16k sims)")
    ap.add_argument("--max-positions", type=int, default=200)
    ap.add_argument("--c-scale", type=float, default=0.025)
    ap.add_argument("--c-visit", type=float, default=50.0)
    ap.add_argument("--c-visit-root", type=float, default=900.0)
    ap.add_argument("--c-scale-root", type=float, default=PLAY_SEARCH_DEFAULTS["c_scale_root"],
                    help="ROOT-ONLY c_scale (descent keeps --c-scale). Default matches the "
                         "production play search (root-log); <0 = legacy linear root.")
    ap.add_argument("--q-visit-exp-root", type=float, default=PLAY_SEARCH_DEFAULTS["q_visit_exp_root"],
                    help="ROOT-ONLY value-transform exponent. Default matches the production play "
                         "search (root-log); >=90 = legacy linear root.")
    ap.add_argument("--gumbel-topk", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("runs/backtest/chunk_trajectory.jsonl"))
    args = ap.parse_args()

    import torch

    from chess_anti_engine.inference import LocalModelEvaluator
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint
    from chess_anti_engine.uci.search import SearchWorker
    from chess_anti_engine.uci.time_manager import Deadline

    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    pol_enc = str(getattr(model, "policy_encoding", "lc0_1858"))
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    cfg = GumbelConfig(
        simulations=int(args.chunk_sims), add_noise=False, temperature=0.0,
        input_history_encoding=hist, input_extra_features=extra, policy_encoding=pol_enc,
        compute_relations=use_rel, topk=int(args.gumbel_topk),
        c_scale=float(args.c_scale), c_visit=float(args.c_visit),
        c_visit_root=float(args.c_visit_root),
        c_scale_root=float(args.c_scale_root), q_visit_exp_root=float(args.q_visit_exp_root),
    )
    worker = SearchWorker(
        LocalModelEvaluator(model, device=args.device), device=args.device,
        gumbel_cfg=cfg, chunk_sims=int(args.chunk_sims),
    )

    positions = _stratified(load_audit_set(args.audit_set), int(args.max_positions))
    print(f"[traj] {len(positions)} positions x {args.max_chunks} chunks of {args.chunk_sims} "
          f"(c_scale={args.c_scale} c_visit_root={args.c_visit_root} topk={args.gumbel_topk})",
          flush=True)
    import threading

    def action_to_uci(act: int, board: chess.Board, ucis: list[str]) -> tuple[str | None, int]:
        # The worker's own action->move decode (handles policy encoding + orientation);
        # then locate it in the position's legal-move order for the regret lookup.
        try:
            uci = index_to_move(int(act), board).uci()
        except Exception:
            return None, -1
        return (uci, ucis.index(uci)) if uci in ucis else (uci, -1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with args.out.open("w") as fh:
        for pi, pos in enumerate(positions):
            board = chess.Board(pos.fen)
            ucis, _ = legal_full_indices(board)
            regrets = move_regrets(pos, ucis)
            worker.reset_tree()
            snaps: list[dict] = []

            # Default-arg binding captures this iteration's loop vars (avoids
            # cell-var-from-loop); the list defaults are intentional, not a bug.
            def on_chunk(
                total_nodes: int, _b=board, _u=ucis, _r=regrets, _s=snaps,
            ) -> None:
                actions, visits = worker._filtered_root_visits(None)
                if actions.size == 0:
                    return
                best = worker._emitted_action(actions, visits, None)
                uci, li = action_to_uci(int(best), _b, _u)
                tot = float(visits.sum())
                shares = visits.astype(np.float64) / tot if tot > 0 else visits.astype(np.float64)
                srt = np.sort(shares)[::-1]
                ngap = float(srt[0] - srt[1]) if srt.size >= 2 else float(srt[0] if srt.size else 0)
                rq, qg = 0.0, float("nan")
                tree, rid = worker._tree, worker._root_id
                if tree is not None and rid is not None:
                    rq = float(tree.node_q(rid))
                    ca, _cv, cq = tree.get_children_q(rid, rq)
                    if ca.size >= 2 and best in ca.tolist():
                        bm = ca == best
                        oth = cq[~bm]
                        qg = float(cq[bm].max() - oth.max()) if oth.size else float("inf")
                _s.append({
                    "nodes": total_nodes, "uci": uci,
                    "regret_cp": float(_r[li]) if li >= 0 else float(_r.max()),
                    "visit_gap": ngap, "visit_entropy": _entropy(shares),
                    "q_gap": qg, "root_q": rq,
                    "shares": {int(a): float(s) for a, s in zip(actions, shares)},
                })

            worker.run(
                board, stop_event=threading.Event(), deadline=Deadline(deadline_ms=None),
                max_nodes=int(args.max_chunks) * int(args.chunk_sims), optimum_ms=None,
                allow_terminal_shortcuts=False, on_chunk=on_chunk,
            )
            # Dynamic features (vs previous chunk) + final-move/regret targets.
            final_uci = snaps[-1]["uci"] if snaps else None
            final_regret = snaps[-1]["regret_cp"] if snaps else None
            for k, s in enumerate(snaps):
                prev = snaps[k - 1] if k > 0 else None
                flip = bool(prev is not None and s["uci"] != prev["uci"])
                qdrift = abs(s["root_q"] - prev["root_q"]) if prev else None
                churn = None
                if prev is not None:
                    keys = set(s["shares"]) | set(prev["shares"])
                    churn = 0.5 * sum(abs(s["shares"].get(a, 0.0) - prev["shares"].get(a, 0.0)) for a in keys)
                row = {
                    "key": pos.key, "phase": pos.phase, "source": pos.source,
                    "piece_count": chess.popcount(board.occupied), "chunk": k + 1,
                    "nodes": s["nodes"], "uci": s["uci"], "regret_cp": s["regret_cp"],
                    "visit_gap": s["visit_gap"], "visit_entropy": s["visit_entropy"],
                    "q_gap": (None if isinstance(s["q_gap"], float) and math.isnan(s["q_gap"]) else s["q_gap"]),
                    "bestmove_flip": flip, "q_drift": qdrift, "visit_churn": churn,
                    # targets: does THIS chunk's move/regret differ from the final (settled) one?
                    "changes_to_final": bool(final_uci is not None and s["uci"] != final_uci),
                    "regret_vs_final_cp": (s["regret_cp"] - final_regret) if final_regret is not None else None,
                }
                fh.write(json.dumps(row) + "\n")
                n_rows += 1
            if (pi + 1) % 25 == 0:
                print(f"[traj] {pi + 1}/{len(positions)}", flush=True)
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()

    print(f"[traj] wrote {n_rows} rows -> {args.out}")


if __name__ == "__main__":
    main()
