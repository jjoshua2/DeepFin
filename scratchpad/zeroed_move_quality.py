"""How good are the legal moves the policy target trains to ~zero?

At 256 sims `m = min(topk=32, n_legal, m_cap=128)`, so topk binds and every legal
move outside the net's top-32 PRIOR is a Gumbel non-candidate. Those moves are never
visited, all receive `completed_q = root_q` -- one identical constant -- so the
target ranks them by prior alone and hands them prior-shaped, near-zero mass.

The question is not whether one of them is secretly BEST (measured: essentially
never). It is whether they are all worthless. If the median non-candidate loses
30cp it is a playable move being trained to impossible; if it loses 300cp the
target is right.

Ruler: `data/audit_set_v1.jsonl.shallow_sf.jsonl` -- MultiPV 40 @ 50k nodes. This is
a WIDER but SHALLOWER ruler than the frozen audit set (MultiPV 10 @ >=1M), and its
numbers are NOT comparable to any frozen-set record. It is used here only to
separate "loses a little" from "loses a lot", which 50k nodes resolves fine.

Restricted to positions where MultiPV 40 covers EVERY legal move, so no move is
scored by omission. Mate-containing positions are dropped rather than mapped to a
cp: the mate<->cp fold is a known contradiction in this repo and would dominate
exactly the tail being measured.

The candidate set is approximated by top-32 prior. Live selfplay adds Gumbel noise
(`gumbel_scale`), which swaps a few borderline moves in and out; that blurs the
boundary slightly and cannot manufacture the effect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shallow", default="data/audit_set_v1.jsonl.shallow_sf.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--max-positions", type=int, default=1200)
    ap.add_argument("--out", default="scratchpad/zeroed_move_quality.json")
    args = ap.parse_args()

    import torch

    from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
    from chess_anti_engine.inference import LocalModelEvaluator
    from chess_anti_engine.eval.audit import legal_full_indices
    from chess_anti_engine.moves import POLICY_SIZE
    from chess_anti_engine.moves.encode import policy_batch_to_full_if_needed
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    rows: list[tuple[chess.Board, dict[str, int]]] = []
    with Path(args.shallow).open() as f:
        for line in f:
            r = json.loads(line)
            board = chess.Board(r["key"] + " 0 1")
            n = board.legal_moves.count()
            if n <= args.topk or len(r["pvs"]) < n:
                continue
            if r.get("mate") is not None or any(p.get("mate") is not None for p in r["pvs"]):
                continue
            cps = {str(p["move"]): int(p["cp"]) for p in r["pvs"] if p.get("cp") is not None}
            if len(cps) < n:
                continue
            rows.append((board, cps))
            if len(rows) >= args.max_positions:
                break
    print(f"positions usable: {len(rows)}")

    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    pol_enc = str(getattr(model, "policy_encoding", "lc0_1858"))
    print(f"checkpoint encoding: {hist}/{extra}/{pol_enc}")
    evaluator = LocalModelEvaluator(model, device=args.device)

    inv_rate: list[float] = []          # P(tail move BETTER than a candidate), within position
    n_cand_worse: list[int] = []         # candidates worse than the BEST tail move
    best_tail_rank: list[float] = []     # rank of best tail move among all legal (1=best)
    best_tail_rank_floor: list[float] = []  # rank it would have if the tail were the worst moves
    pos_any_inversion: list[bool] = []
    conc_all: list[float] = []
    conc_tail: list[float] = []
    conc_cand: list[float] = []
    conc_cross: list[float] = []
    inv_topk: dict[int, list[float]] = {4: [], 8: [], 16: []}
    nworse_topk: dict[int, list[int]] = {4: [], 8: [], 16: []}
    good_counts: dict[int, list[int]] = {25: [], 50: [], 100: []}
    good_prior_mass: dict[int, list[float]] = {25: [], 50: [], 100: []}
    top1_prior: list[float] = []
    tail_loss: list[float] = []
    cand_loss: list[float] = []
    tail_prior_mass: list[float] = []
    per_pos_tail_best: list[float] = []
    n_tail = 0
    n_cand = 0

    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        xs = np.stack([
            encode_cboard(
                CBoard.from_board(b), input_history_encoding=hist, input_extra_features=extra,
            ) for b, _ in chunk
        ]).astype(np.float32)
        with torch.no_grad():
            pol_logits, _ = evaluator.evaluate_encoded(xs)
        pol_logits = np.asarray(pol_logits, dtype=np.float32)
        if pol_logits.shape[1] != POLICY_SIZE:
            pol_logits = policy_batch_to_full_if_needed(
                pol_logits, policy_encoding=pol_enc, fill_value=-1e9,
            )
        for j, (board, cps) in enumerate(chunk):
            ucis, idxs = legal_full_indices(board)
            logits = pol_logits[j, idxs].astype(np.float64)
            logits -= logits.max()
            prior = np.exp(logits)
            prior /= prior.sum()
            order = np.argsort(-prior)
            best_cp = max(cps[u] for u in ucis)
            tail_idx = order[args.topk:]
            cand_idx = order[:args.topk]
            tail_prior_mass.append(float(prior[tail_idx].sum()))
            losses = [float(best_cp - cps[ucis[int(i)]]) for i in tail_idx]
            tail_loss.extend(losses)
            per_pos_tail_best.append(min(losses))
            cand_loss.extend(float(best_cp - cps[ucis[int(i)]]) for i in cand_idx)
            all_loss = np.array([best_cp - cps[u] for u in ucis], dtype=np.float64)
            tl = all_loss[tail_idx]
            cl = all_loss[cand_idx]
            # Strictly better = lower cp loss. Ties are counted as non-inversions.
            inv_rate.append(float((tl[:, None] < cl[None, :]).mean()))
            n_cand_worse.append(int((cl > tl.min()).sum()))
            pos_any_inversion.append(bool((cl > tl.min()).any()))
            best_tail_rank.append(float((all_loss < tl.min()).sum() + 1))
            best_tail_rank_floor.append(float(len(cand_idx) + 1))
            # RANKING ACCURACY: pairwise concordance of the net prior against SF's
            # cp ordering. 0.5 == coin flip. In the TAIL the target's order IS the
            # prior's order (all tail moves share completed_q), so conc_tail is
            # literally the ranking quality of the signal the target teaches there.
            def _conc(sel: np.ndarray) -> float | None:
                L = all_loss[sel]
                P = prior[sel]
                if L.size < 2:
                    return None
                di = L[:, None] - L[None, :]
                dp = P[:, None] - P[None, :]
                m = (di != 0) & (dp != 0)
                if not m.any():
                    return None
                # prior says i better (dp>0) and SF says i better (di<0) => concordant
                return float((((dp > 0) & (di < 0)) | ((dp < 0) & (di > 0)))[m].mean())
            allsel = np.arange(all_loss.size)
            for acc, sel in ((conc_all, allsel), (conc_tail, tail_idx), (conc_cand, cand_idx)):
                v = _conc(np.asarray(sel))
                if v is not None:
                    acc.append(v)
            # cross-boundary pairs only: does the prior at least know tail < candidate?
            Lt, Lc = all_loss[tail_idx], all_loss[cand_idx]
            mm = (Lt[:, None] - Lc[None, :]) != 0
            if mm.any():
                conc_cross.append(float(((Lt[:, None] > Lc[None, :]))[mm].mean()))
            # "Not trained to zero" is NARROWER than "candidate": many candidates also
            # end at ~0 target mass. Proxy the mass-carrying set by prior top-K and
            # recompute -- if the inversions live only against the bad candidates, the
            # rate collapses here and the mis-ordering never touches a move that matters.
            for K in (4, 8, 16):
                sub = all_loss[order[:K]]
                inv_topk[K].append(float((tl[:, None] < sub[None, :]).mean()))
                nworse_topk[K].append(int((sub > tl.min()).sum()))
            top1_prior.append(float(prior.max()))
            for thr in (25, 50, 100):
                sel = all_loss <= thr
                good_counts[thr].append(int(sel.sum()))
                good_prior_mass[thr].append(float(prior[sel].sum()))
            n_tail += len(tail_idx)
            n_cand += len(cand_idx)
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[net] {min(start + args.batch_size, len(rows))}/{len(rows)}")

    t = np.asarray(tail_loss)
    c = np.asarray(cand_loss)
    pm = np.asarray(tail_prior_mass)
    pb = np.asarray(per_pos_tail_best)

    def q(a: np.ndarray, p: float) -> float:
        return float(np.percentile(a, p))

    out = {
        "positions": len(rows),
        "topk": args.topk,
        "checkpoint": args.checkpoint,
        "ruler": "shallow_sf multipv40 @50k nodes (NOT the frozen >=1M/mpv10 ruler)",
        "n_tail_moves": n_tail,
        "n_candidate_moves": n_cand,
        "tail_cp_loss": {
            "mean": float(t.mean()), "p10": q(t, 10), "p25": q(t, 25),
            "median": q(t, 50), "p75": q(t, 75), "p90": q(t, 90),
        },
        "candidate_cp_loss": {
            "mean": float(c.mean()), "median": q(c, 50), "p90": q(c, 90),
        },
        "tail_frac_within": {
            "25cp": float((t <= 25).mean()), "50cp": float((t <= 50).mean()),
            "100cp": float((t <= 100).mean()), "200cp": float((t <= 200).mean()),
        },
        "best_tail_move_per_position_cp_loss": {
            "median": q(pb, 50), "p25": q(pb, 25), "p75": q(pb, 75),
            "frac_positions_with_a_tail_move_within_50cp": float((pb <= 50).mean()),
        },
        "separation": {
            "pairwise_inversion_rate_tail_better_than_candidate": float(np.mean(inv_rate)),
            "frac_positions_with_ANY_inversion": float(np.mean(pos_any_inversion)),
            "candidates_worse_than_best_tail_move": {
                "mean": float(np.mean(n_cand_worse)),
                "median": float(np.median(n_cand_worse)),
                "p90": float(np.percentile(n_cand_worse, 90)),
            },
            "rank_of_best_tail_move": {
                "mean": float(np.mean(best_tail_rank)),
                "median": float(np.median(best_tail_rank)),
                "mean_if_tail_were_strictly_worst": float(np.mean(best_tail_rank_floor)),
            },
        },
        "ranking_accuracy_prior_vs_sf": {
            "concordance_all_legal": float(np.mean(conc_all)),
            "concordance_within_tail": float(np.mean(conc_tail)),
            "concordance_within_candidates": float(np.mean(conc_cand)),
            "concordance_cross_boundary": float(np.mean(conc_cross)),
            "chance": 0.5,
            "n_positions_tail": len(conc_tail),
        },
        "separation_vs_prior_topK": {
            str(K): {
                "pairwise_inversion_rate": float(np.mean(inv_topk[K])),
                "mean_topK_moves_worse_than_best_tail": float(np.mean(nworse_topk[K])),
                "frac_positions_with_any": float(np.mean([n > 0 for n in nworse_topk[K]])),
            } for K in (4, 8, 16)
        },
        "moves_within_X_cp_of_best": {
            str(thr): {
                "mean_count": float(np.mean(good_counts[thr])),
                "median_count": float(np.median(good_counts[thr])),
                "mean_net_prior_mass_on_them": float(np.mean(good_prior_mass[thr])),
                "median_net_prior_mass_on_them": float(np.median(good_prior_mass[thr])),
            } for thr in (25, 50, 100)
        },
        "net_top1_prior": {
            "mean": float(np.mean(top1_prior)), "median": float(np.median(top1_prior)),
        },
        "tail_prior_mass": {
            "mean": float(pm.mean()), "median": q(pm, 50), "p90": q(pm, 90),
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
