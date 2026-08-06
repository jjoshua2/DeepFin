#!/usr/bin/env python3
"""What late-game ``gumbel_scale`` restores candidate-support replenishment?

Production decays ``gumbel_scale`` to 0 after move ~15. ``gumbel_scale`` gates
CANDIDATE SELECTION (``gumbel.py::_select_top_m_with_gumbel``: the root score is
``scale * Gumbel + log_prior``), so a scale of 0 makes the candidate set the
deterministic top-m by prior — the support-replenishment valve is shut. This
probe measures, per phase and per scale, how much replenishment is bought and
what it costs in played-move quality.

Two definitions of "outside support" are reported, because the obvious one
saturates. Our net's prior support (p>1e-3) is ~9-15 moves while ``topk`` is 32,
so the deterministic top-32 ALREADY reaches outside the prior's support in
essentially every position at every scale, including 0. That metric therefore
cannot discriminate scales and must not be read as "the valve is open". The
discriminating quantity is replenishment relative to the NO-NOISE candidate set:
moves that would not have been candidates at all had the scale been 0.

Metrics 1, 2 and the no-noise baseline are pure root-prior computations (the
score above needs no tree), so they run over many draws for free. Metrics 3 and
4 need a real search per draw and run over a smaller draw count on a subset.

The played move is the sequential-halving survivor ``remaining[0]`` — the argmax
of ``g + logit + sigma(q)`` over candidates — read off the root, not a
visit-count proxy, which can differ.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess_anti_engine.mcts.gumbel as gumbel_mod
from chess_anti_engine.encoding import encode_positions_batch
from chess_anti_engine.mcts.gumbel import GumbelConfig, run_gumbel_root_many
from chess_anti_engine.moves import policy_batch_to_full_if_needed
from scripts.lc0_adapter_probe import OurNetEvaluator, Position, build_positions, decode_records

PHASES = ("opening", "middlegame", "endgame")


@dataclass
class Root:
    """Root state captured at improved-policy time.

    ``candidates`` is the selected set, which is NOT ``actions``: the improved
    policy is built over every legal move, and non-candidates still receive
    prior-shaped mass through the completed-Q fallback. Earned mass must be
    measured over candidates only, or it reads the prior tail and comes out
    near-identical at every scale.
    """
    actions: np.ndarray
    qvalues: np.ndarray
    played: int
    candidates: np.ndarray


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _with(cfg: GumbelConfig, **kw: Any) -> GumbelConfig:
    out = copy.copy(cfg)
    for k, v in kw.items():
        setattr(out, k, v)
    return out


def candidate_set(
    log_pri: np.ndarray, gumbel: np.ndarray, *, scale: float, topk: int, sim_budget: int,
) -> np.ndarray:
    """Replicate ``_select_top_m_with_gumbel``'s selection over local indices.

    Mirrors the production code exactly, including the sequential-halving cap
    ``m_cap = max(2, (sim_budget + 1) // 2)`` that makes ``topk`` above
    ceil(sims/2) inert.
    """
    g = scale * gumbel if scale > 0.0 else np.zeros_like(gumbel)
    score = g + log_pri
    n = log_pri.size
    if sim_budget <= 1:
        m = 1
    else:
        m_cap = max(2, (sim_budget + 1) // 2)
        m = max(2, int(min(int(topk), int(n), int(m_cap))))
    kth = min(m - 1, n - 1)
    return np.argpartition(-score, kth)[:m]


def search_root(pos: Position, ev: Any, cfg: GumbelConfig, seed: int) -> tuple[np.ndarray, Root | None]:
    """One search; returns the improved policy and the captured root."""
    sink: list[Root] = []
    original = gumbel_mod._build_improved_policy_for_board

    def wrapper(st: Any, *, root_q: float, cfg: GumbelConfig, rng: Any) -> Any:
        pri = st.priors
        legal = np.nonzero(pri > 0)[0].astype(np.int64)
        qvalues = np.zeros(legal.size, dtype=np.float64)
        for i, a in enumerate(legal):
            ch = st.root.children.get(int(a))
            n = 0 if ch is None else int(ch.N)
            qvalues[i] = (-ch.W / n) if (ch is not None and n > 0) else float(root_q)
        played = int(st.remaining[0]) if st.remaining else -1
        cands = np.array(sorted(st.candidates or []), dtype=np.int64)
        sink.append(Root(legal, qvalues, played, cands))
        return original(st, root_q=root_q, cfg=cfg, rng=rng)

    gumbel_mod._build_improved_policy_for_board = wrapper
    try:
        probs, _actions, _values, _masks = run_gumbel_root_many(
            None, [pos.board], device="cpu",
            rng=np.random.default_rng(seed), cfg=cfg, evaluator=ev,
        )
    finally:
        gumbel_mod._build_improved_policy_for_board = original
    return probs[0], (sink[0] if sink else None)


def prior_rows(ev: Any, positions: list[Position], cfg: GumbelConfig) -> list[np.ndarray]:
    """Masked prior probabilities over each position's legal moves."""
    xs = encode_positions_batch(
        [p.board for p in positions], add_features=True,
        input_history_encoding=cfg.input_history_encoding,
        input_extra_features=cfg.input_extra_features,
    )
    pol, _wdl = ev.evaluate_encoded(xs)
    pol_full = policy_batch_to_full_if_needed(
        np.asarray(pol, dtype=np.float32), fill_value=-1e9,
    )
    out: list[np.ndarray] = []
    for i, pos in enumerate(positions):
        lg = pol_full[i][pos.full_idx].astype(np.float64)
        p = np.exp(lg - lg.max())
        out.append(p / p.sum())
    return out


def selection_metrics(
    priors: list[np.ndarray], positions: list[Position], *,
    scales: list[float], topk: int, sims: int, draws: int, seed: int,
) -> dict[tuple[float, int], dict[str, float]]:
    """Metrics 1 and 2: candidate selection only, no search."""
    acc: dict[tuple[float, int], dict[str, list[float]]] = {}
    for pos_i, (p, pos) in enumerate(zip(priors, positions, strict=True)):
        log_pri = np.log(np.maximum(p, 1e-12))
        deficit = float(log_pri.max()) - log_pri  # nats below the top move
        in_support = p > 1e-3
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        in_99 = np.zeros(p.size, dtype=bool)
        in_99[order[: int(np.searchsorted(cum, 0.99) + 1)]] = True
        rng = np.random.default_rng(seed + 1000 * pos_i)
        gumbels = rng.gumbel(size=(draws, p.size))
        base = set(candidate_set(
            log_pri, gumbels[0], scale=0.0, topk=topk, sim_budget=sims,
        ).tolist())
        for scale in scales:
            slot = acc.setdefault((scale, pos.phase), {
                "outside_support": [], "outside_99": [], "new_rate": [],
                "n_new": [], "deepest": [], "deepest_new": [],
            })
            for d in range(draws):
                cands = candidate_set(
                    log_pri, gumbels[d], scale=scale, topk=topk, sim_budget=sims,
                )
                new = sorted(set(cands.tolist()) - base)
                slot["outside_support"].append(float(bool((~in_support[cands]).any())))
                slot["outside_99"].append(float(bool((~in_99[cands]).any())))
                slot["new_rate"].append(float(bool(new)))
                slot["n_new"].append(float(len(new)))
                slot["deepest"].append(float(deficit[cands].max()))
                if new:
                    slot["deepest_new"].append(float(deficit[new].max()))
    return {
        key: {
            "outside_support": _mean(s["outside_support"]),
            "outside_99": _mean(s["outside_99"]),
            "new_rate": _mean(s["new_rate"]),
            "n_new": _mean(s["n_new"]),
            "deepest_med": _pct(s["deepest"], 50),
            "deepest_p95": _pct(s["deepest"], 95),
            "deepest_new_med": _pct(s["deepest_new"], 50),
            "deepest_new_p95": _pct(s["deepest_new"], 95),
        }
        for key, s in acc.items()
    }


def search_metrics(
    ev: Any, priors: list[np.ndarray], positions: list[Position], cfg: GumbelConfig, *,
    scales: list[float], topk: int, sims: int, draws: int, seed: int, log: list[str],
) -> dict[tuple[float, int], dict[str, float]]:
    """Metrics 3 and 4: earned target mass and played-move cost. Needs trees."""
    acc: dict[tuple[float, int], dict[str, list[float]]] = {}
    t0 = time.time()
    zero = np.zeros(1)
    for pos_i, (p, pos) in enumerate(zip(priors, positions, strict=True)):
        log_pri = np.log(np.maximum(p, 1e-12))
        in_support = p > 1e-3
        full_of_local = pos.full_idx
        # No-noise reference: scale 0 is deterministic, so one run is the baseline.
        _probs0, root0 = search_root(pos, ev, _with(cfg, gumbel_scale=0.0), seed + pos_i)
        ref_action = root0.played if root0 is not None else -1
        ref_q = _q_of(root0, ref_action)
        base_full = {
            int(full_of_local[i])
            for i in candidate_set(
                log_pri, np.broadcast_to(zero, p.shape), scale=0.0, topk=topk,
                sim_budget=sims,
            ).tolist()
        }
        for scale in scales:
            slot = acc.setdefault((scale, pos.phase), {
                "earned_new": [], "earned_outside": [], "flip": [], "qdef": [],
            })
            for d in range(draws):
                probs, root = search_root(
                    pos, ev, _with(cfg, gumbel_scale=scale),
                    seed + 7919 * (d + 1) + pos_i,
                )
                if root is None or root.candidates.size == 0:
                    continue
                # Normalise over ALL legal moves (the improved policy's own
                # normalisation), then sum only the candidate slots of interest.
                local = probs[full_of_local].astype(np.float64)
                tot = local.sum()
                if tot > 0:
                    local = local / tot
                cand_local = np.array(
                    [int(np.flatnonzero(full_of_local == int(a))[0]) for a in root.candidates],
                    dtype=np.int64,
                )
                is_new = np.array(
                    [int(a) not in base_full for a in root.candidates], dtype=bool,
                )
                slot["earned_new"].append(float(local[cand_local[is_new]].sum()))
                out_cand = cand_local[~in_support[cand_local]]
                slot["earned_outside"].append(float(local[out_cand].sum()))
                flip = float(root.played != ref_action)
                slot["flip"].append(flip)
                q = _q_of(root, root.played)
                if flip and not np.isnan(q) and not np.isnan(ref_q):
                    slot["qdef"].append(float(ref_q - q))
        if (pos_i + 1) % 5 == 0:
            log.append(f"  [search] {pos_i + 1}/{len(positions)} positions "
                       f"({time.time() - t0:.0f}s, {ev.misses} misses)")
            print(log[-1], flush=True)
    return {
        key: {
            "earned_new": _mean(s["earned_new"]),
            "earned_outside": _mean(s["earned_outside"]),
            "flip": _mean(s["flip"]),
            "qdef": _mean(s["qdef"]),
            "n_flip": float(len(s["qdef"])),
        }
        for key, s in acc.items()
    }


def _q_of(root: Root | None, action: int) -> float:
    if root is None or action < 0:
        return float("nan")
    hit = np.flatnonzero(root.actions == action)
    return float(root.qvalues[hit[0]]) if hit.size else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="gumbel_scale replenishment probe")
    ap.add_argument("--our-checkpoint", required=True)
    ap.add_argument("--matched", default="tests/data/lc0_matched60.npz")
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--c-scale", type=float, default=0.025)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--sims", type=int, default=256)
    ap.add_argument("--select-draws", type=int, default=512)
    ap.add_argument("--search-draws", type=int, default=16)
    ap.add_argument("--search-positions-per-phase", type=int, default=10)
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    log: list[str] = []
    t_start = time.time()
    ev = OurNetEvaluator(args.our_checkpoint, device="cpu")
    cfg = GumbelConfig(
        simulations=args.sims, topk=args.topk, c_scale=args.c_scale,
        gumbel_scale=0.0, add_noise=True, temperature=0.0,
        input_history_encoding=str(ev.input_history_encoding),
        input_extra_features=str(ev.input_extra_features),
    )
    log.append("# gumbel_scale replenishment probe")
    log.append("")
    log.append(f"- net: `{ev.path}`")
    log.append(f"- fixed: c_scale {args.c_scale}, topk {args.topk}, {args.sims} sims, "
               f"temperature 0")
    log.append(f"- encoding read OFF the checkpoint: history "
               f"`{ev.input_history_encoding}`, extras `{ev.input_extra_features}`")
    log.append("- inputs are FEN-ONLY (stackless board), matching the frozen rulers")
    log.append("- played move = sequential-halving survivor `remaining[0]` "
               "(argmax of g + logit + sigma(q) over candidates), not argmax-visits")

    records = decode_records(Path(args.matched))
    positions = build_positions(records, log)
    priors = prior_rows(ev, positions, cfg)
    log.append("- prior support (p>1e-3), median by phase: " + ", ".join(
        f"{PHASES[ph]} "
        f"{np.median([float((priors[i] > 1e-3).sum()) for i in range(len(positions)) if positions[i].phase == ph]):.0f}"
        for ph in range(3)
    ))

    sel = selection_metrics(
        priors, positions, scales=args.scales, topk=args.topk, sims=args.sims,
        draws=args.select_draws, seed=args.seed,
    )

    sub_idx: list[int] = []
    for ph in range(3):
        sub_idx.extend(
            [i for i in range(len(positions)) if positions[i].phase == ph][
                : args.search_positions_per_phase
            ],
        )
    sub_pos = [positions[i] for i in sub_idx]
    sub_pri = [priors[i] for i in sub_idx]
    log.append(f"- selection metrics: all {len(positions)} positions x "
               f"{args.select_draws} draws (no search)")
    log.append(f"- search metrics: {len(sub_pos)} positions x {args.search_draws} draws "
               f"x {len(args.scales)} scales")

    if not args.no_warm:
        t0 = time.time()
        warm = _with(cfg, gumbel_scale=0.0, topk=10_000)
        for i, pos in enumerate(sub_pos):
            search_root(pos, ev, warm, args.seed + i)
        log.append(f"- warm pass: topk=all, {args.sims} sims, {len(sub_pos)} positions, "
                   f"{time.time() - t0:.1f}s, {ev.net_calls} net calls")
        print(log[-1], flush=True)

    srch = search_metrics(
        ev, sub_pri, sub_pos, cfg, scales=args.scales, topk=args.topk, sims=args.sims,
        draws=args.search_draws, seed=args.seed, log=log,
    )

    nan = float("nan")
    for ph, name in enumerate(PHASES):
        log.append("")
        log.append(f"## {name}")
        log.append("")
        log.append("| gumbel_scale | replenish rate (new vs no-noise) | mean #new cands | "
                   "deepest cand med/p95 (nats below top) | deepest NEW cand med/p95 | "
                   "earned mass on new cands | earned mass outside p>1e-3 | "
                   "played-move flip rate | mean Q deficit when flipped |")
        log.append("|---|---|---|---|---|---|---|---|---|")
        for scale in args.scales:
            s = sel.get((scale, ph), {})
            r = srch.get((scale, ph), {})
            log.append(
                f"| {scale} | {s.get('new_rate', nan):.3f} | {s.get('n_new', nan):.2f} | "
                f"{s.get('deepest_med', nan):.2f} / {s.get('deepest_p95', nan):.2f} | "
                f"{s.get('deepest_new_med', nan):.2f} / {s.get('deepest_new_p95', nan):.2f} | "
                f"{r.get('earned_new', nan):.3e} | {r.get('earned_outside', nan):.3e} | "
                f"{r.get('flip', nan):.3f} | "
                f"{r.get('qdef', nan):+.4f} (n={r.get('n_flip', 0):.0f}) |"
            )
        log.append("")
        log.append("- SATURATING control, do NOT read as the valve — "
                   "P(any candidate outside prior p>1e-3): " + ", ".join(
                       f"{sc}:{sel.get((sc, ph), {}).get('outside_support', nan):.3f}"
                       for sc in args.scales))
        log.append("- P(any candidate outside the prior's cumulative-99% set): " + ", ".join(
            f"{sc}:{sel.get((sc, ph), {}).get('outside_99', nan):.3f}"
            for sc in args.scales))

    log.append("")
    log.append(f"- total {time.time() - t_start:.0f}s; cache hits {ev.hits}, "
               f"misses {ev.misses}, net calls {ev.net_calls}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
