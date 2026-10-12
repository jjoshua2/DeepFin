#!/usr/bin/env python3
"""Score fixed-tree completed-Q and I-opt-proxy policy counterfactuals OFFLINE.

Input is scripts/search_gain_probe.py --operator-trace --out TRACE.jsonl.
No training, search reruns, teacher calls, checkpoint loads or GPU use here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.eval.iopt_operator import policy_metrics, reconstruct

ARMS = ("current", "unscaled", "iopt_proxy", "iopt_matched_kl")


def _bootstrap_delta(
    samples: list[tuple[str, float]],
    *,
    seed: int,
    n_boot: int = 2000,
) -> dict[str, float | int] | None:
    """Root-cluster bootstrap. One root may recur across simulation rungs."""
    by_root: dict[str, list[float]] = defaultdict(list)
    for key, delta in samples:
        by_root[key].append(delta)
    if not by_root:
        return None
    vals = np.array([np.mean(v) for v in by_root.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    if vals.size == 1:
        return {"n_roots": 1, "mean_cp": float(vals[0]), "ci_low": None,
                "ci_high": None}
    take = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = vals[take].mean(axis=1)
    return {
        "n_roots": int(vals.size),
        "mean_cp": float(vals.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def _aggregate(
    rows: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for name in ARMS:
        metrics = [r["policies"][name] for r in rows]
        top = [float(m["top1_regret_cp"]) for m in metrics if m["top1_regret_cp"] is not None]
        exp = [float(m["expected_regret_cp"]) for m in metrics if m["expected_regret_cp"] is not None]
        arms[name] = {
            "top1_regret_cp_mean_covered": float(np.mean(top)) if top else None,
            "n_top1_covered": len(top),
            "expected_regret_cp_mean_FULL_labels": float(np.mean(exp)) if exp else None,
            "n_FULL_legal_reference": len(exp),
            "mean_unknown_reference_mass": float(np.mean([m["unknown_mass"] for m in metrics])),
            "mean_kl_prior": float(np.mean([m["kl_prior"] for m in metrics])),
            "mean_entropy": float(np.mean([m["entropy"] for m in metrics])),
        }

    paired: dict[str, Any] = {}
    for name in ARMS[1:]:
        top_pairs: list[tuple[str, float]] = []
        exp_pairs: list[tuple[str, float]] = []
        for row in rows:
            cur = row["policies"]["current"]
            alt = row["policies"][name]
            if cur["top1_regret_cp"] is not None and alt["top1_regret_cp"] is not None:
                top_pairs.append((
                    row["root_key"],
                    float(cur["top1_regret_cp"]) - float(alt["top1_regret_cp"]),
                ))
            if cur["expected_regret_cp"] is not None and alt["expected_regret_cp"] is not None:
                exp_pairs.append((
                    row["root_key"],
                    float(cur["expected_regret_cp"]) - float(alt["expected_regret_cp"]),
                ))
        paired[name] = {
            "top1_regret_gain_cp": _bootstrap_delta(top_pairs, seed=seed),
            "FULL_reference_expected_regret_gain_cp": _bootstrap_delta(
                exp_pairs, seed=seed + 11
            ),
        }

    flags = Counter(k for row in rows for k, v in row["flags"].items() if v)
    matched = sum(bool(r["matched_kl_reachable"]) for r in rows)
    return {
        "qualified_roots": len(rows),
        "arms": arms,
        "paired_gain_positive_is_better": paired,
        "failure_flags": dict(flags),
        "n_iopt_KL_match_reachable": matched,
        "n_iopt_KL_match_unreachable": len(rows) - matched,
        "median_q_range": float(np.median([r["q_range"] for r in rows])),
        "max_baseline_native_parity_abs": float(
            max(r["parity_max_abs"] for r in rows)
        ),
    }


def run_screen(
    input_path: Path,
    *,
    beta: float = 0.1,
    tau2: float = 0.08,
    variance_scale: float = 0.04,
    unvisited_multiplier: float = 4.0,
    parity_tolerance: float = 3e-4,
    seed: int = 42,
    small_q_threshold: float = 0.03,
    bad_regret_cp: float = 50.0,
) -> dict[str, Any]:
    """Consume frozen raw observations, preserve unknown labels and parity gates."""
    metadata: dict[str, Any] | None = None
    accepted: dict[int, list[dict[str, Any]]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    sha = hashlib.sha256()
    with input_path.open("rb") as fh:
        for line_num, line in enumerate(fh, 1):
            sha.update(line)
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("_meta"):
                if metadata is not None or line_num != 1:
                    raise ValueError("expected exactly one _meta header on line 1")
                metadata = raw
                if not raw.get("operator_trace") or raw.get("operator_trace_format") != 1:
                    raise ValueError("source was not captured with --operator-trace")
                continue
            if metadata is None:
                raise ValueError("missing provenance header")
            trace = raw.get("operator_trace")
            if trace is None:
                skipped["trace_absent"] += 1
                continue
            if not isinstance(trace, dict):
                raise ValueError(f"line {line_num}: trace is not a JSON object")
            if trace.get("label_kind") not in (
                "native_deep_sf_multipv_partial", "unavailable"
            ):
                raise ValueError(f"line {line_num}: unsupported reference kind")
            try:
                result = reconstruct(
                    trace, beta=beta, tau2=tau2,
                    variance_scale=variance_scale,
                    unvisited_multiplier=unvisited_multiplier,
                    parity_tolerance=parity_tolerance,
                )
            except (ValueError, TypeError, KeyError) as exc:
                reason = ("native_parity" if "native root-policy" in str(exc)
                          else "invalid_or_unsupported_trace")
                skipped[reason] += 1
                continue
            sf = np.isfinite(result.reference_cp)
            n_known = int(sf.sum())
            vis = result.visits > 0
            known_visited = vis & sf
            best_known = int(np.nanargmax(result.reference_cp)) if n_known else None

            # A partial MultiPV list only supports best-*known* claims.
            value_misrank = False
            if known_visited.sum() >= 2:
                idx = np.flatnonzero(known_visited)
                q_winner = int(idx[np.argmax(result.completed_q[idx])])
                sf_winner = int(idx[np.argmax(result.reference_cp[idx])])
                value_misrank = (
                    result.reference_cp[sf_winner] -
                    result.reference_cp[q_winner] >= bad_regret_cp
                )
            measured = {
                name: policy_metrics(p, result.prior, result.reference_cp)
                for name, p in result.policies.items()
            }
            cur = measured["current"]
            flags = {
                "best_known_SF_move_unvisited": (
                    best_known is not None and not bool(vis[best_known])
                ),
                "visited_Q_misranks_known_moves_by_threshold": bool(value_misrank),
                "small_Q_range_high_KL_bad_current_top1": (
                    result.q_range <= small_q_threshold
                    and cur["kl_prior"] >= 0.3
                    and cur["top1_regret_cp"] is not None
                    and float(cur["top1_regret_cp"]) >= bad_regret_cp
                ),
                "missing_ANY_reference_move": n_known < result.prior.size,
            }
            accepted[int(raw["sims"])].append({
                "root_key": str(raw["fen"]),
                "source_key": str(raw.get("fen")),
                "phase": int(raw.get("phase", -1)),
                "sims": int(raw["sims"]),
                "q_range": result.q_range,
                "n_legal": int(result.prior.size),
                "n_visited": int(vis.sum()),
                "n_reference": n_known,
                "parity_max_abs": result.parity_max_abs,
                "matched_kl_reachable": result.matched_kl_reachable,
                "flags": flags,
                "policies": measured,
                "top1_uci": {
                    name: trace["legal_uci"][int(m["top1"])]
                    for name, m in measured.items()
                },
            })
    if metadata is None:
        raise ValueError("empty or missing trace header")
    by_sims = {
        str(sims): _aggregate(rows, seed=seed + sims)
        for sims, rows in sorted(accepted.items()) if rows
    }
    seen = sum(len(rows) for rows in accepted.values()) + sum(skipped.values())
    all_quality = sum(
        m["arms"]["current"]["n_top1_covered"]
        for m in by_sims.values()
    )
    fidelity_excluded = skipped.get("native_parity", 0)
    status = (
        "unqualified_no_valid_roots" if not accepted
        else "unqualified_native_fidelity" if (
            seen and fidelity_excluded / seen > 0.05
        ) else "quality_unread_no_direct_labels" if not all_quality
        else "diagnostic_only_no_training_or_arena_claim"
    )
    return {
        "status": status,
        "input_sha256": sha.hexdigest(),
        "producer_meta": metadata,
        "parameters": {
            "beta": beta, "tau2": tau2, "variance_scale": variance_scale,
            "unvisited_multiplier": unvisited_multiplier,
            "parity_tolerance": parity_tolerance, "seed": seed,
            "small_q_threshold": small_q_threshold,
            "bad_regret_cp": bad_regret_cp,
        },
        "counts": {
            "rows_seen": seen,
            "rows_qualified": sum(len(rows) for rows in accepted.values()),
            "skipped": dict(skipped),
        },
        "by_simulations": by_sims,
        "limitations": [
            "I-opt uses an uncalibrated visit-based independent-Q-variance proxy.",
            "A tree held fixed cannot establish the strength of changed halving.",
            "Unknown MultiPV tail is NOT assigned a centipawn value.",
            "Current probe rebuilds root positions from FEN; full training history may differ.",
            "Same-checkpoint target quality does not establish retrained policy Elo.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--tau2", type=float, default=0.08)
    ap.add_argument("--variance-scale", type=float, default=0.04)
    ap.add_argument("--unvisited-multiplier", type=float, default=4.0)
    ap.add_argument("--parity-tolerance", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.input.resolve() == args.out.resolve():
        ap.error("--out must differ from --input")
    report = run_screen(
        args.input, beta=args.beta, tau2=args.tau2,
        variance_scale=args.variance_scale,
        unvisited_multiplier=args.unvisited_multiplier,
        parity_tolerance=args.parity_tolerance, seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print(json.dumps({
        "status": report["status"],
        "counts": report["counts"],
        "by_simulations": {
            k: {
                "qualified_roots": v["qualified_roots"],
                "failure_flags": v["failure_flags"],
                "paired_gain_positive_is_better": v["paired_gain_positive_is_better"],
            }
            for k, v in report["by_simulations"].items()
        },
        "output": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
