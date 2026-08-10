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
    log_pri: np.ndarray  # prior logits over `actions`
    qbar: np.ndarray  # min-max-normalized completed Q in [0,1], over `actions`
    sigma: float  # realized sigma from the MEASURED max per-move visits
    max_visit: float  # the measured max per-move visit count sigma was built from


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
        visits = np.zeros(legal.size, dtype=np.float64)
        for i, a in enumerate(legal):
            ch = st.root.children.get(int(a))
            n = 0 if ch is None else int(ch.N)
            visits[i] = float(n)
            qvalues[i] = (-ch.W / n) if (ch is not None and n > 0) else float(root_q)
        played = int(st.remaining[0]) if st.remaining else -1
        cands = np.array(sorted(st.candidates or []), dtype=np.int64)
  # `vol` is what production feeds the volatility factors with. `Node.vol`
  # always exists, so read it directly rather than through a `getattr(..., 0.0)`
  # default, which would silently substitute the neutral value -- the very
  # failure this argument exists to remove.
        qbar, sigma = _qbar_and_sigma(
            priors=pri[legal], visits=visits, qvalues=qvalues, root_q=float(root_q), cfg=cfg,
            vol=float(st.root.vol),
        )
        sink.append(Root(
            legal, qvalues, played, cands,
            np.log(np.maximum(pri[legal], 1e-12)), qbar, sigma,
            float(visits.max(initial=0.0)),
        ))
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


def _qbar_and_sigma(
    *, priors: np.ndarray, visits: np.ndarray, qvalues: np.ndarray,
    root_q: float, cfg: GumbelConfig, vol: float,
) -> tuple[np.ndarray, float]:
    """Split the root value transform into normalized Q-bar and the sigma scale.

    Replicates ``_completed_q_transform``'s min-max normalisation and
    ``_root_sigma_scale``, then ASSERTS that ``sigma * qbar`` reproduces
    production's transform output. Without that assert this is a re-derivation
    that could silently drift from the code it claims to measure.

    ``sigma`` uses the MEASURED max per-move visit count, never the sim budget:
    sequential halving concentrates visits, so max_visit is far below the budget
    and using the budget would overstate the barrier the Q term can clear.

    ⚑ THE VOLATILITY FACTORS ARE READ OFF ``cfg``, NOT HARDCODED. This function
    used to pass ``sigma_factor=1.0, fpu_penalty=0.0`` while production passes
    ``_volatility_sigma_factor(root.vol, cfg)`` / ``_volatility_fpu_penalty(...)``
    (gumbel.py's root call sites). That was latent only because
    ``volatility_q_scale``/``volatility_fpu`` both default to 0.0 and appear in
    no config -- the day either is armed, the probe would have measured a
    transform production does not use, and the assert below could not have
    noticed, because it passed the SAME neutral values to the reference call.
    A guard has to share the criterion's instrument. Both sides now take the
    realized factors, so the assert stays a real check at every config.
    """
    sigma_factor = float(gumbel_mod._volatility_sigma_factor(float(vol), cfg))
    fpu_penalty = float(gumbel_mod._volatility_fpu_penalty(float(vol), cfg))
    prior = np.maximum(np.asarray(priors, dtype=np.float64), np.finfo(np.float64).tiny)
    q = np.asarray(qvalues, dtype=np.float64)
    v = np.asarray(visits, dtype=np.float64)
    visited = v > 0.0
    sum_visits = float(v.sum())
    sum_probs = float(prior[visited].sum()) if visited.any() else 0.0
    weighted_q = (
        float((prior[visited] * q[visited] / sum_probs).sum())
        if sum_probs > 0.0 and np.isfinite(sum_probs) else float(root_q)
    )
    mixed = (float(root_q) + sum_visits * weighted_q) / (sum_visits + 1.0)
    completed = np.where(visited, q, mixed - fpu_penalty)
    lo, hi = float(completed.min()), float(completed.max())
    qbar = (completed - lo) / max(hi - lo, 1e-8)
    sigma = sigma_factor * float(
        gumbel_mod._root_sigma_scale(max_visit=int(v.max(initial=0.0)), cfg=cfg)
    )

    ref = gumbel_mod._completed_q_transform(
        actions=np.arange(prior.size, dtype=np.int64), priors=prior, visits=v,
        qvalues=q, raw_value=float(root_q), cfg=cfg,
        sigma_factor=sigma_factor, fpu_penalty=fpu_penalty, root=True,
    )
  # ⚑ WHAT THIS ASSERT DOES *NOT* COVER -- read it for exactly what it proves.
  # Both sides now derive `sigma_factor`/`fpu_penalty` from the SAME `vol`
  # argument, so the assert checks the ALGEBRA (min-max normalise, sigma scale,
  # fpu offset) against production's implementation of it. It cannot check that
  # `vol` is the right number. Pass the wrong one and both sides move together
  # and it stays silent: measured on one synthetic root with both volatility
  # knobs armed, `vol=0.0` against production's `vol=0.31` gives max|diff| 21.21
  # and `vol=1.0` gives 0.563, with no complaint either time (an independent
  # reviewer reproduced the same shape on their own inputs, 33.49 and 0.329).
  # The single caller reads `st.root.vol` -- the same attribute gumbel.py's root
  # call sites pass -- and that correspondence is by inspection, not by test.
  # Deliberately left there: it is a far smaller hole than the hardcoded
  # neutrals it replaced, and closing it needs a captured production transform
  # to compare against, which this probe has no way to obtain.
    if not np.allclose(sigma * qbar, ref, atol=1e-8, rtol=1e-6):
        raise AssertionError(
            "sigma*qbar does not reproduce _completed_q_transform: the residual "
            "barrier would be computed against a transform production does not use",
        )
    return qbar, sigma


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
    scales: list[float], c_scales: list[float], topk: int, sims: int, draws: int,
    seed: int, log: list[str],
) -> dict[tuple[float, float, int], dict[str, float]]:
    """Residual barrier, earned target mass and played-move cost. Needs trees.

    Keyed by (gumbel_scale, c_scale, phase). The candidate SET does not depend on
    c_scale (selection is prior + noise + topk only), so the no-noise baseline
    set is shared; the search dynamics and the transform do depend on it, so
    every c_scale gets its own searches.
    """
    acc: dict[tuple[float, float, int], dict[str, list[float]]] = {}
    t0 = time.time()
    zero = np.zeros(1)
    for pos_i, (p, pos) in enumerate(zip(priors, positions, strict=True)):
        log_pri = np.log(np.maximum(p, 1e-12))
        in_support = p > 1e-3
        full_of_local = pos.full_idx
        base_full = {
            int(full_of_local[i])
            for i in candidate_set(
                log_pri, np.broadcast_to(zero, p.shape), scale=0.0, topk=topk,
                sim_budget=sims,
            ).tolist()
        }
        for c_scale in c_scales:
            cfg_c = _with(cfg, c_scale=c_scale)
            # No-noise reference: scale 0 is deterministic, so one run per c_scale.
            _probs0, root0 = search_root(
                pos, ev, _with(cfg_c, gumbel_scale=0.0), seed + pos_i,
            )
            ref_action = root0.played if root0 is not None else -1
            ref_q = _q_of(root0, ref_action)
            for scale in scales:
                slot = acc.setdefault((scale, c_scale, pos.phase), {
                    "earned_new": [], "earned_outside": [], "flip": [], "qdef": [],
                    "R": [], "R_neg": [], "dq": [], "equalq": [],
                    "R_pos_adv": [], "R_pos_adv_neg": [], "sigma": [], "maxn": [],
                })
                # scale 0 is deterministic: one draw is the whole distribution.
                n_draws = 1 if scale <= 0.0 else draws
                for d in range(n_draws):
                    probs, root = search_root(
                        pos, ev, _with(cfg_c, gumbel_scale=scale),
                        seed + 7919 * (d + 1) + pos_i,
                    )
                    if root is None or root.candidates.size == 0:
                        continue
                    # Normalise over ALL legal moves (the improved policy's own
                    # normalisation), then sum only the candidate slots.
                    local = probs[full_of_local].astype(np.float64)
                    tot = local.sum()
                    if tot > 0:
                        local = local / tot
                    cand_local = np.array(
                        [int(np.flatnonzero(full_of_local == int(a))[0])
                         for a in root.candidates], dtype=np.int64,
                    )
                    is_new = np.array(
                        [int(a) not in base_full for a in root.candidates], dtype=bool,
                    )
                    slot["earned_new"].append(float(local[cand_local[is_new]].sum()))
                    out_cand = cand_local[~in_support[cand_local]]
                    slot["earned_outside"].append(float(local[out_cand].sum()))
                    slot["sigma"].append(root.sigma)
                    slot["maxn"].append(root.max_visit)

                    # Residual barrier for each noise-promoted candidate.
                    best = int(np.argmax(root.log_pri))
                    for a in root.candidates[is_new]:
                        hit = np.flatnonzero(root.actions == int(a))
                        if not hit.size:
                            continue
                        j = int(hit[0])
                        dq = float(root.qbar[j] - root.qbar[best])
                        barrier = float(root.log_pri[best] - root.log_pri[j])
                        r = barrier - root.sigma * dq
                        slot["R"].append(r)
                        slot["R_neg"].append(float(r < 0.0))
                        slot["dq"].append(dq)
                        slot["equalq"].append(float(abs(dq) < 0.05))
                        if dq > 0.05:  # positive-advantage discoveries only
                            slot["R_pos_adv"].append(r)
                            slot["R_pos_adv_neg"].append(float(r < 0.0))

                    flip = float(root.played != ref_action)
                    slot["flip"].append(flip)
                    q = _q_of(root, root.played)
                    if flip and not np.isnan(q) and not np.isnan(ref_q):
                        slot["qdef"].append(float(ref_q - q))
        if (pos_i + 1) % 3 == 0:
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
            "R_med": _pct(s["R"], 50), "R_p10": _pct(s["R"], 10),
            "R_p90": _pct(s["R"], 90), "R_neg": _mean(s["R_neg"]),
            "n_promoted": float(len(s["R"])),
            "equalq": _mean(s["equalq"]), "dq_med": _pct(s["dq"], 50),
            "R_adv_med": _pct(s["R_pos_adv"], 50),
            "R_adv_neg": _mean(s["R_pos_adv_neg"]),
            "n_adv": float(len(s["R_pos_adv"])),
            "sigma": _mean(s["sigma"]), "maxn": _mean(s["maxn"]),
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
    ap.add_argument("--c-scale", type=float, default=0.025,
                    help="c_scale for the selection-only metrics table")
    ap.add_argument("--c-scales", type=float, nargs="+", default=[0.025, 0.05, 0.1],
                    help="c_scales swept for the residual-barrier / search metrics")
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
        ev, sub_pri, sub_pos, cfg, scales=args.scales, c_scales=args.c_scales,
        topk=args.topk, sims=args.sims, draws=args.search_draws, seed=args.seed, log=log,
    )

    nan = float("nan")
    for ph, name in enumerate(PHASES):
        log.append("")
        log.append(f"## {name}")
        log.append("")
        log.append("Selection only (c_scale-independent — the candidate set is prior + "
                   "noise + topk).")
        log.append("")
        log.append("| gumbel_scale | replenish rate (new vs no-noise) | mean #new cands | "
                   "deepest cand med/p95 (nats below top) | deepest NEW cand med/p95 |")
        log.append("|---|---|---|---|---|")
        for scale in args.scales:
            s = sel.get((scale, ph), {})
            log.append(
                f"| {scale} | {s.get('new_rate', nan):.3f} | {s.get('n_new', nan):.2f} | "
                f"{s.get('deepest_med', nan):.2f} / {s.get('deepest_p95', nan):.2f} | "
                f"{s.get('deepest_new_med', nan):.2f} / {s.get('deepest_new_p95', nan):.2f} |"
            )
        log.append("")
        log.append("Residual barrier R = (z_best - z_a) - sigma*(Qbar_a - Qbar_best) over "
                   "noise-promoted candidates. R < 0 predicts the move can take the "
                   "improved-policy argmax from the prior leader. `equal-Q` = "
                   "|dQbar| < 0.05: lift is sigma*dQbar, so these get ~zero lift at ANY "
                   "sigma and are structurally unreachable by a one-parameter config.")
        log.append("")
        log.append("| c_scale | gumbel_scale | sigma (measured maxN) | R med (p10/p90) | "
                   "frac R<0 | frac equal-Q | R med, +adv only | frac R<0, +adv only | "
                   "n promoted | earned mass on new | flip rate | mean Q deficit when flipped |")
        log.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c_scale in args.c_scales:
            for scale in args.scales:
                r = srch.get((scale, c_scale, ph), {})
                log.append(
                    f"| {c_scale} | {scale} | {r.get('sigma', nan):.2f} "
                    f"(N={r.get('maxn', nan):.0f}) | "
                    f"{r.get('R_med', nan):+.2f} ({r.get('R_p10', nan):+.2f}/"
                    f"{r.get('R_p90', nan):+.2f}) | {r.get('R_neg', nan):.3f} | "
                    f"{r.get('equalq', nan):.3f} | {r.get('R_adv_med', nan):+.2f} | "
                    f"{r.get('R_adv_neg', nan):.3f} | "
                    f"{r.get('n_promoted', 0):.0f} (adv {r.get('n_adv', 0):.0f}) | "
                    f"{r.get('earned_new', nan):.3e} | {r.get('flip', nan):.3f} | "
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
