"""Does the 256-sim training search rank moves better than the raw net prior?

The loop's whole engine is `target = search(net)`. If the search's ORDERING of moves
is no better than the net's own prior ordering, the target carries no information the
net does not already have, and CE against it is a fixed point -- which is what the
plateau looks like.

Metric: pairwise concordance against SF's cp ordering (0.5 = coin flip), scored over
the same positions for both arms so the comparison is PAIRED. Reported unweighted and
mass-weighted (pairs weighted by p_i+p_j under the arm's own distribution), because an
unweighted score lets thousands of irrelevant bad-vs-bad pairs drown out the handful
of pairs that decide which move gets played.

Ruler: banked SF MultiPV 40 @ 50k nodes, restricted to positions where it covers EVERY
legal move. Wider but shallower than the frozen audit ruler; not comparable to frozen
records. Mate positions dropped (mate<->cp fold is a known contradiction).

Search shape read off the LIVE yaml: sims 256, topk 32, c_scale 0.025, vloss_weight 1.
`add_noise=False` matches scripts/audit_targets.py -- root Gumbel noise perturbs the
stored visit distribution, and a non-deterministic ruler is worse than a slightly
idealised one. This is the one axis on which this differs from live selfplay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np


def concordance(loss: np.ndarray, score: np.ndarray, w: np.ndarray | None = None) -> float | None:
    """Fraction of comparable pairs where a HIGHER score means a LOWER cp loss."""
    if loss.size < 2:
        return None
    dl = loss[:, None] - loss[None, :]
    ds = score[:, None] - score[None, :]
    m = (dl != 0) & (ds != 0)
    if not m.any():
        return None
    agree = ((ds > 0) & (dl < 0)) | ((ds < 0) & (dl > 0))
    if w is None:
        return float(agree[m].mean())
    pw = w[:, None] + w[None, :]
    tot = float(pw[m].sum())
    if tot <= 0:
        return None
    return float((agree & m).astype(np.float64).ravel() @ (pw * m).ravel() / tot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shallow", default="data/audit_set_v1.jsonl.shallow_sf.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-positions", type=int, default=400)
    ap.add_argument("--sims", type=int, default=256)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--c-scale", type=float, default=0.025)
    ap.add_argument("--vloss-weight", type=int, default=1)
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.25)
    ap.add_argument("--out", default="scratchpad/search_vs_prior_ranking.json")
    args = ap.parse_args()

    import torch

    from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
    from chess_anti_engine.eval.audit import legal_full_indices
    from chess_anti_engine.inference import LocalModelEvaluator
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
    from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY, POLICY_SIZE
    from chess_anti_engine.moves.encode import policy_batch_to_full_if_needed
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    if str(args.device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(
            float(args.gpu_mem_fraction), torch.device(args.device).index or 0)
        print(f"[rank] GPU memory capped at fraction {args.gpu_mem_fraction}")

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
    print(f"positions: {len(rows)}")

    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    pol_enc = str(getattr(model, "policy_encoding", "lc0_1858"))
    evaluator = LocalModelEvaluator(model, device=args.device)
    cfg = GumbelConfig(
        simulations=int(args.sims), topk=int(args.topk), c_scale=float(args.c_scale),
        add_noise=False, temperature=0.0,
        input_history_encoding=hist, input_extra_features=extra,
        policy_encoding=pol_enc,
    )
    rng = np.random.default_rng(0)

    acc: dict[str, list[float]] = {k: [] for k in (
        "prior_all", "search_all", "prior_top32", "search_top32",
        "prior_all_w", "search_all_w", "prior_top32_w", "search_top32_w",
    )}
    paired_delta_top32: list[float] = []
    search_entropy: list[float] = []
    prior_entropy: list[float] = []
    top1_agree = 0
    n_pos = 0

    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        boards = [b for b, _ in chunk]
        xs = np.stack([
            encode_cboard(CBoard.from_board(b),
                          input_history_encoding=hist, input_extra_features=extra)
            for b in boards
        ]).astype(np.float32)
        with torch.no_grad():
            pol_logits, _ = evaluator.evaluate_encoded(xs)
        pol_logits = np.asarray(pol_logits, dtype=np.float32)
        if pol_logits.shape[1] != POLICY_SIZE:
            pol_logits = policy_batch_to_full_if_needed(
                pol_logits, policy_encoding=pol_enc, fill_value=-1e9)
        result = run_gumbel_root_many_c(
            None, [b.copy() for b in boards], device=args.device, rng=rng, cfg=cfg,
            evaluator=evaluator, vloss_weight=int(args.vloss_weight), target_batch=0,
        )
        probs_b = result[0]

        for j, (board, cps) in enumerate(chunk):
            ucis, idxs = legal_full_indices(board)
            lg = pol_logits[j, idxs].astype(np.float64)
            lg -= lg.max()
            prior = np.exp(lg)
            prior /= prior.sum()
            visit = np.asarray(probs_b[j], dtype=np.float64)
            if visit.shape[0] != POLICY_SIZE:
                full = np.zeros(POLICY_SIZE, dtype=np.float64)
                full[COMPACT_TO_FULL_POLICY] = visit
                visit = full
            tgt = visit[idxs]
            if tgt.sum() <= 0:
                continue
            tgt = tgt / tgt.sum()
            best_cp = max(cps[u] for u in ucis)
            loss = np.array([best_cp - cps[u] for u in ucis], dtype=np.float64)
            top32 = np.argsort(-prior)[:args.topk]

            for name, sc in (("prior", prior), ("search", tgt)):
                v = concordance(loss, sc)
                if v is not None:
                    acc[f"{name}_all"].append(v)
                vw = concordance(loss, sc, w=sc)
                if vw is not None:
                    acc[f"{name}_all_w"].append(vw)
                v32 = concordance(loss[top32], sc[top32])
                if v32 is not None:
                    acc[f"{name}_top32"].append(v32)
                v32w = concordance(loss[top32], sc[top32], w=sc[top32])
                if v32w is not None:
                    acc[f"{name}_top32_w"].append(v32w)
            pv = concordance(loss[top32], prior[top32])
            sv = concordance(loss[top32], tgt[top32])
            if pv is not None and sv is not None:
                paired_delta_top32.append(sv - pv)
            prior_entropy.append(float(-(prior * np.log(prior + 1e-300)).sum()))
            search_entropy.append(float(-(tgt * np.log(tgt + 1e-300)).sum()))
            top1_agree += int(int(np.argmax(prior)) == int(np.argmax(tgt)))
            n_pos += 1
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[rank] {min(start + args.batch_size, len(rows))}/{len(rows)}")

    d = np.asarray(paired_delta_top32)
    boot = np.array([
        np.random.default_rng(s).choice(d, d.size, replace=True).mean()
        for s in range(2000)
    ])
    out = {
        "positions_scored": n_pos,
        "checkpoint": args.checkpoint,
        "shape": {"sims": args.sims, "topk": args.topk, "c_scale": args.c_scale,
                  "vloss_weight": args.vloss_weight, "add_noise": False},
        "ruler": "shallow_sf multipv40 @50k nodes (NOT the frozen >=1M/mpv10 ruler)",
        "concordance": {k: float(np.mean(v)) for k, v in acc.items() if v},
        "paired_delta_top32_search_minus_prior": {
            "mean": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "frac_positions_search_better": float((d > 0).mean()),
            "frac_positions_tied": float((d == 0).mean()),
        },
        "entropy_nats": {"prior": float(np.mean(prior_entropy)),
                         "search": float(np.mean(search_entropy))},
        "top1_agreement_prior_vs_search": top1_agree / max(1, n_pos),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
