#!/usr/bin/env python3
"""Log the marginal value of extra search per position, for time-management tuning.

For each frozen deep-SF audit position, run the net+Gumbel search at a grid of
sim budgets and snapshot, per budget, the move the search would play plus the
"complexity / search-frozen-vs-changing" features the abort gate keys on. The
deep-SF labels give the regret of the chosen move directly, so each row carries
``regret(budget)`` and the analysis can build per-position regret-vs-sims curves.

This is the WHOLE-SET stage of the backtest ladder: the audit set is a
position-keyed (decorrelated) sample, so it's the right data to learn which
features rank positions by the value of one more chunk — NOT to read off an
absolute per-game time budget (that needs game traces with tree reuse, a later
stage). Budgets here are INDEPENDENT Gumbel runs (not the accumulating-chunk
production path); that's faithful enough for feature ranking and avoids the
Gumbel sequential-halving non-monotonicity trap of snapshotting one long search.
Objective is WDL expected-score regret (flat in already-decided positions, so the
allocator learns NOT to waste time there); cp regret is logged alongside.

Output: one tidy JSONL row per (position, budget). Downstream (pandas):
  - regret(B) - regret(1.25B) = marginal value of ~25% more search;
  - oracle (greedy on marginal value at fixed total budget) vs uniform vs a
    feature policy -> fraction of avoidable regret captured per feature;
  - the budget-matched threshold lambda = the continue/stop cutoff the gate wants.

Usage:
  PYTHONPATH=. python3 scripts/backtest_time_value.py \\
      --checkpoint <ckpt> --device cuda \\
      --budgets 256,512,1024,2048,4096 --max-positions 1500 \\
      --out runs/backtest/time_value.jsonl
Run on CPU (--device cpu) for a smoke test so it doesn't fight a live trainer.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import chess
import numpy as np

from chess_anti_engine.eval.audit import (
    AuditPosition,
    criticality_gap,
    legal_full_indices,
    load_audit_set,
    move_regrets,
)
from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY, POLICY_SIZE

# cp -> expected score (logistic). C in cp; ~300 keeps it flat once a side is
# clearly winning, so regret in score units stops rewarding spend on decided
# positions. Only used for the score-regret column; cp regret is also logged.
_CP_TO_SCORE_C = 300.0


def _score(cp: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-cp / _CP_TO_SCORE_C))


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


class _Searcher:
    """Loads the model + evaluator once; runs batched Gumbel at any sim budget,
    returning per-board (visit probs over legal, Q over legal, root Q). Mirrors
    ``audit_targets._net_candidates`` and reuses its legal-move mapping, but also
    extracts per-child Q (for the top-2 Q gap the abort gate keys on)."""

    def __init__(
        self, checkpoint: str, device: str, *, topk: int, policy_temp: float,
        c_scale: float, c_visit: float, c_visit_root: float,
        q_visit_exp: float = 1.0, q_global_scale: bool = False, q_visit_floor: float = -1.0,
        compile_mode: str | None = None,
    ) -> None:
        import torch

        from chess_anti_engine.inference import LocalModelEvaluator
        from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

        self._torch = torch
        self._device = device
        self._topk = int(topk)
        self._policy_temp = float(policy_temp)
        self._c_scale = float(c_scale)
        self._c_visit = float(c_visit)
        self._c_visit_root = float(c_visit_root)
        self._q_visit_exp = float(q_visit_exp)
        self._q_global_scale = bool(q_global_scale)
        self._q_visit_floor = float(q_visit_floor)
        model = load_model_from_checkpoint(checkpoint, device=device)
        model.eval()
        self._hist = str(getattr(model, "input_history_encoding", "legacy"))
        self._extra = str(getattr(model, "input_extra_features", "v1"))
        self._pol_enc = str(getattr(model, "policy_encoding", "az_4672"))
        self._use_rel = bool(getattr(model, "use_dynamic_relations", False))
        if compile_mode:
  # Same shared TorchInductor/Triton cache as the engine + training, so repeat
  # invocations reuse compiled kernels (first config warms, the rest are fast).
            from pathlib import Path

            from chess_anti_engine.worker import _configure_shared_compile_cache
            _configure_shared_compile_cache(
                cache_dir=Path("~/.cache/deepfin/worker_cache").expanduser(),
            )
            model = torch.compile(model, mode=compile_mode)
        self._evaluator = LocalModelEvaluator(model, device=device)  # type: ignore[arg-type]

    def run(
        self, boards: list[chess.Board], sims: int, *, seed: int,
    ) -> list[dict]:
        from chess_anti_engine.mcts.gumbel import GumbelConfig
        from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c

        cfg = GumbelConfig(
            simulations=int(sims), add_noise=False, temperature=0.0,
            input_history_encoding=self._hist, input_extra_features=self._extra,
            policy_encoding=self._pol_enc, compute_relations=self._use_rel,
            policy_temp=self._policy_temp, topk=self._topk,
            c_scale=self._c_scale, c_visit=self._c_visit,
            c_visit_root=self._c_visit_root, q_visit_exp=self._q_visit_exp,
            q_global_scale=self._q_global_scale, q_visit_floor=self._q_visit_floor,
        )
        rng = np.random.default_rng(seed)
  # The Gumbel runner encodes the boards itself (relations included, via cfg).
        probs_b, _actions, values, _masks, tree, ids = run_gumbel_root_many_c(
            model=None, boards=list(boards), device=self._device, rng=rng, cfg=cfg,
            evaluator=self._evaluator,
        )
        out: list[dict] = []
        for j, board in enumerate(boards):
            _, idxs = legal_full_indices(board)
            visit = np.asarray(probs_b[j], dtype=np.float64)
            is_compact = visit.shape[0] != POLICY_SIZE
            if is_compact:
                full = np.zeros(POLICY_SIZE, dtype=np.float64)
                full[COMPACT_TO_FULL_POLICY] = visit
                visit = full
            root_q = float(values[j])
  # Per-child Q -> full policy -> legal order (default = root Q for unexplored).
            ca, cv, cq = tree.get_children_q(int(ids[j]), root_q)
            ca_full = COMPACT_TO_FULL_POLICY[ca] if is_compact else ca
            q_full = np.full(POLICY_SIZE, root_q, dtype=np.float64)
            v_full = np.zeros(POLICY_SIZE, dtype=np.float64)
            if ca_full.size:
                q_full[ca_full] = cq
                v_full[ca_full] = cv
            out.append({
                "visit": visit[idxs],          # visit prob per legal move
                "q": q_full[idxs],             # root Q per legal move
                "child_visits": v_full[idxs],  # raw child visit counts per legal
                "root_q": np.float64(root_q),
            })
        if str(self._device).startswith("cuda"):
            self._torch.cuda.empty_cache()
        return out


def _features(
    snap: dict, legal_ucis: list[str], regrets: np.ndarray, best_cp: float,
) -> dict:
    """Per-budget complexity/search features + the chosen move's regret."""
    visit = snap["visit"]
    q = snap["q"]
    explored = snap["child_visits"] > 0
    chosen = int(np.argmax(visit))
    chosen_uci = legal_ucis[chosen]
    regret_cp = float(regrets[chosen])
    # Score regret: best move vs chosen, in expected score on the cp->score curve
    # (cp(chosen) = best_cp - regret_cp). Flat once a side is clearly winning.
    regret_score = _score(best_cp) - _score(best_cp - regret_cp)
    # top-2 visit gap.
    vs = np.sort(visit)[::-1]
    n_gap = float(vs[0] - vs[1]) if vs.size >= 2 else float(vs[0] if vs.size else 0.0)
    # top-2 Q gap over EXPLORED moves: chosen Q minus best other explored Q
    # (matches the abort gate's _value_is_decisive). NaN-safe when <2 explored.
    q_gap = float("nan")
    if explored.any():
        q_exp = q[explored]
        if q_exp.size >= 2:
            others = q[explored & (np.arange(q.size) != chosen)]
            q_gap = float(q[chosen] - others.max()) if others.size else float("inf")
    return {
        "chosen_uci": chosen_uci,
        "regret_cp": regret_cp,
        "regret_score": regret_score,
        "visit_entropy": _entropy(visit),
        "n_gap_top2": n_gap,
        "q_gap_top2": q_gap,
        "root_q": float(snap["root_q"]),
        "n_explored": int(explored.sum()),
    }


def _stratified(positions: list[AuditPosition], limit: int) -> list[AuditPosition]:
    """Round-robin across piece-count buckets so the thin middlegame buckets are
    represented, not just whatever the file happens to front-load."""
    if limit <= 0 or limit >= len(positions):
        return positions
    buckets: dict[int, list[AuditPosition]] = {}
    for p in positions:
        pc = chess.popcount(chess.Board(p.fen).occupied)
        buckets.setdefault(min(32, max(2, pc)) // 4, []).append(p)
    order = sorted(buckets)
    out: list[AuditPosition] = []
    i = 0
    while len(out) < limit and any(buckets[b] for b in order):
        b = order[i % len(order)]
        if buckets[b]:
            out.append(buckets[b].pop())
        i += 1
    return out[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(prog="backtest_time_value")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"))
    ap.add_argument("--budgets", default="256,512,1024,2048,4096",
                    help="comma-separated sim budgets (independent Gumbel runs)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-positions", type=int, default=0, help="0 = all")
    ap.add_argument("--gumbel-topk", type=int, default=32)
    ap.add_argument("--policy-temp", type=float, default=1.0)
    ap.add_argument("--c-scale", type=float, default=0.025,
                    help="Gumbel sigma(q) value-transform scale. PRODUCTION UCI uses 0.025; the "
                         "GumbelConfig default 0.1 over-trusts the value head at high sims (search "
                         "collapses -> flat scaling). Sweep this to find what makes search scale.")
    ap.add_argument("--c-visit", type=float, default=50.0)
    ap.add_argument("--c-visit-root", type=float, default=900.0,
                    help="root-halving c_visit floor (the root/descent c_scale SPLIT that fixes "
                         "scaling): a large root floor (~900) makes one c_scale fit every sim count "
                         "at the root while descent keeps c_visit~50. -1 = legacy (no split).")
    ap.add_argument("--q-visit-exp", type=float, default=1.0,
                    help="descent value-transform exponent on max_visit. 1.0=linear (q_scale grows "
                         "with visits -> over-trusts the value head at high sims); <1=sublinear "
                         "(sim-invariant). Test for high-sim scaling past the plateau.")
    ap.add_argument("--q-global-scale", action="store_true",
                    help="descent scales q_scale by the ROOT max_visit (uniform across the tree); "
                         "pairs with --q-visit-exp<1 for sim-invariance.")
    ap.add_argument("--q-visit-floor", type=float, default=-1.0,
                    help="decoupled additive floor: q_scale=floor+c_scale*max_visit^exp when >=0.")
    ap.add_argument("--compile", dest="compile_mode", nargs="?", const="max-autotune", default=None,
                    help="torch.compile the model (much faster forward; needs the GPU free). "
                         "Optional mode arg (default max-autotune). Uses the shared engine cache.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/backtest/time_value.jsonl"))
    args = ap.parse_args()

    budgets = sorted({int(b) for b in str(args.budgets).split(",") if b.strip()})
    if not budgets:
        raise SystemExit("--budgets is empty")
    positions = load_audit_set(args.audit_set)
    positions = _stratified(positions, int(args.max_positions))
    print(f"[backtest] {len(positions)} positions x {len(budgets)} budgets {budgets}")

    boards = [chess.Board(p.fen) for p in positions]
    legal = [legal_full_indices(b)[0] for b in boards]
    regrets = [move_regrets(p, lu) for p, lu in zip(positions, legal, strict=True)]
    meta = []
    for p, b in zip(positions, boards, strict=True):
        meta.append({
            "key": p.key, "phase": p.phase, "source": p.source,
            "piece_count": chess.popcount(b.occupied),
            "legal_moves": b.legal_moves.count(),
            "in_check": b.is_check(),
            "criticality_cp": (lambda g: g if math.isfinite(g) else None)(criticality_gap(p.move_cp)),
            "best_cp": p.best_cp,
        })

    searcher = _Searcher(
        args.checkpoint, args.device, topk=args.gumbel_topk, policy_temp=args.policy_temp,
        c_scale=args.c_scale, c_visit=args.c_visit, c_visit_root=args.c_visit_root,
        q_visit_exp=args.q_visit_exp, q_global_scale=bool(args.q_global_scale),
        q_visit_floor=args.q_visit_floor, compile_mode=args.compile_mode,
    )
    print(f"[backtest] c_scale={args.c_scale} c_visit_root={args.c_visit_root} "
          f"q_visit_exp={args.q_visit_exp} q_global_scale={args.q_global_scale} "
          f"topk={args.gumbel_topk} policy_temp={args.policy_temp}")
    # snapshot[pos_idx][budget] = features; fill by running each budget over all boards.
    per_pos: list[dict[int, dict]] = [{} for _ in positions]
    bs = int(args.batch_size)
    for sims in budgets:
        for s in range(0, len(boards), bs):
            sl = slice(s, s + bs)
            snaps = searcher.run(boards[sl], sims, seed=int(args.seed))
            for k, snap in enumerate(snaps):
                j = s + k
                per_pos[j][sims] = _features(snap, legal[j], regrets[j], positions[j].best_cp)
            print(f"[backtest] sims={sims} {min(s + bs, len(boards))}/{len(boards)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with args.out.open("w") as fh:
        for j, p in enumerate(positions):
            prev_uci: str | None = None
            prev_q: float | None = None
            for sims in budgets:
                f = per_pos[j].get(sims)
                if f is None:
                    continue
                row = {**meta[j], "sims": sims, **f}
                row["bestmove_flip"] = bool(prev_uci is not None and f["chosen_uci"] != prev_uci)
                row["q_drift"] = (
                    abs(f["root_q"] - prev_q) if prev_q is not None else None
                )
                prev_uci, prev_q = f["chosen_uci"], f["root_q"]
                fh.write(json.dumps(row) + "\n")
                n_rows += 1
    print(f"[backtest] wrote {n_rows} rows -> {args.out}")


if __name__ == "__main__":
    main()
