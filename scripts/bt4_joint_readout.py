#!/usr/bin/env python3
"""Read the global BT4 screen against S0 across 25, 100 and 400 simulations.

Usage: PYTHONPATH=. python scripts/bt4_joint_readout.py --reference S0.pt \
    --cell G20T1:25=G20T1.25.games.jsonl --cell G20T1:400=G20T1.400.games.jsonl

Every supplied cell must contain 500 color-swapped opening pairs. Replayed
orphan halves use the producing arena's last-write rule; raw banks stay untouched.
Cross-cell comparisons resample aligned opening pairs against the common S0
opponent. These score differences are not direct head-to-head Elo estimates.
Missing cells remain unread. This exploratory screen never promotes a recipe.
Checkpoint/book content and build hashes require the external run manifest;
arena headers record paths rather than the bytes originally used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.utils.game_log import (
    latest_rows_by_key,
    read_game_log,
    settings_fingerprint,
)
from scripts.arena_standard import pentanomial_counts, summarize_pentanomial

ARMS = ("G20T1", "G20T05", "E0")
BUDGETS = (25, 100, 400)
PAIRS = 500


def read_arm(path: Path, *, reference: Path, seed: int, sims: int = 100) -> dict[str, Any]:
    """Refuse incomplete, conflicting or off-protocol banks before scoring."""
    log = read_game_log(path)
    settings = log.settings
    if log.truncated_tail:
        raise ValueError(f"{path}: require a complete game bank without a torn tail")
    if log.header.get("driver") != "arena_standard" or log.header.get("version") != 1:
        raise ValueError(f"{path}: unsupported arena header")
    if log.fingerprint != settings_fingerprint(settings):
        raise ValueError(f"{path}: header fingerprint disagrees with its settings")
    required = {
        "games": 1000, "mode": "matched_sims", "sims_candidate": sims,
        "sims_reference": sims, "seed": seed, "opening_plies": 16,
        "openings_kind": "book", "max_plies": 300, "temperature": 0.1,
        "gumbel_add_noise": True,
    }
    wrong = {k: settings.get(k) for k, v in required.items() if settings.get(k) != v}
    if wrong:
        raise ValueError(f"{path}: off-protocol settings {wrong}")
    if not settings.get("openings") or not settings.get("candidate"):
        raise ValueError(f"{path}: missing opening book or candidate identity")
    if Path(settings.get("reference", "")).resolve() != reference.resolve():
        raise ValueError(f"{path}: reference is not the requested control checkpoint")
    if Path(settings["candidate"]).resolve() == reference.resolve():
        raise ValueError(f"{path}: candidate is the control itself")
    search = settings.get("search_candidate")
    if not isinstance(search, dict) or search.get("shape") != "training":
        raise ValueError(f"{path}: candidate lacks realized training search settings")
    if search != settings.get("search_reference"):
        raise ValueError(f"{path}: candidate and reference search settings differ")
    if log.info.get("sprt") is not None:
        raise ValueError(f"{path}: sequential stopping is not the fixed-N preregistration")

    history: dict[int, list[dict[str, Any]]] = {}
    for row in log.games:
        pair, half = row.get("pair_id"), row.get("half")
        if type(pair) is not int or not 0 <= pair < PAIRS or type(half) is not int or half not in (0, 1):
            raise ValueError(f"{path}: invalid pair/half identity {(pair, half)}")
        key = (pair, half)
        if row.get("a_is_white") is not (half == 0) or row.get("opening_index") != pair:
            raise ValueError(f"{path}: inconsistent color/opening identity at {key}")
        if not row.get("opening_fen") or row.get("start_fen") != row["opening_fen"]:
            raise ValueError(f"{path}: missing or changed opening at {key}")
        result = row.get("result")
        if result not in ("1-0", "0-1", "1/2-1/2"):
            raise ValueError(f"{path}: unfinished/unknown result at {key}")
        white_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
        score = white_score if half == 0 else 1.0 - white_score
        if type(row.get("score_candidate")) not in (float, int) or row["score_candidate"] != score:
            raise ValueError(f"{path}: result and candidate score disagree at {key}")
        if row.get("seed") != seed or row.get("loop") != "chunked":
            raise ValueError(f"{path}: seed/loop differs from preregistration at {key}")
        history.setdefault(pair, []).append(row)

    for pair, games in history.items():
        # Resume replays BOTH halves of an orphan pair, in either finish order.
        # Earlier attempts can therefore have repeated one half, but cannot
        # contain a complete pair: the arena never replays completed pairs.
        if len(games) < 2 or {r["half"] for r in games[-2:]} != {0, 1}:
            raise ValueError(f"{path}: pair {pair} has no complete final attempt")
        if len({r["half"] for r in games[:-2]}) > 1:
            raise ValueError(f"{path}: pair {pair} replays an already complete pair")
        if len({r["opening_fen"] for r in games}) != 1:
            raise ValueError(f"{path}: pair {pair} changes opening across replay attempts")
    rows = latest_rows_by_key(log.games, key=lambda r: (r["pair_id"], r["half"]))
    if len(rows) != 2 * PAIRS:
        raise ValueError(f"{path}: require 1000 canonical games / 500 complete pairs")

    scores, openings = [], []
    for pair in range(PAIRS):
        white, black = rows[(pair, 0)], rows[(pair, 1)]
        if white["opening_fen"] != black["opening_fen"]:
            raise ValueError(f"{path}: colors did not share opening {pair}")
        scores.append((white["score_candidate"] + black["score_candidate"]) / 2)
        openings.append(white["opening_fen"])
    modes = {(r.get("compile"), r.get("eval_hoist")) for r in rows.values()}
    if len(modes) != 1 or any(v in (None, "", "unknown") for v in next(iter(modes))):
        raise ValueError(f"{path}: mixed or unknown compile/evaluator modes")
    summary = summarize_pentanomial(pentanomial_counts([2 * s for s in scores]))
    lo = summary.score - 1.96 * summary.score_se
    hi = summary.score + 1.96 * summary.score_se
    verdict = "SUCCESS" if lo > 0.5 else "KILL" if hi < 0.5 else "INCONCLUSIVE"
    return {
        "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "settings": settings, "execution": list(next(iter(modes))),
        "raw_game_rows": len(log.games), "superseded_orphan_rows": len(log.games) - len(rows),
        "openings": openings, "scores": np.asarray(scores, dtype=np.float64),
        "result": {
            "games": 1000, "pairs": PAIRS, "score": summary.score,
            "score_ci95": [lo, hi], "elo": summary.elo,
            "elo_ci95": list(summary.elo_ci95), "verdict": verdict,
            "pentanomial": dict(zip(("WW", "WD_DW", "DD_WL", "LD_DL", "LL"), summary.counts)),
        },
    }


def compare(a: np.ndarray, b: np.ndarray, *, seed: int, samples: int) -> dict[str, Any]:
    """Bootstrap aligned pair differences, retaining common-opponent covariance."""
    if a.shape != (PAIRS,) or b.shape != (PAIRS,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("comparisons require 500 finite aligned pair scores per cell")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    delta = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, 1000):
        stop = min(start + 1000, samples)
        idx = rng.integers(0, PAIRS, size=(stop - start, PAIRS))
        means[start:stop] = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "score_advantage": float(delta.mean()), "ci95": [float(lo), float(hi)],
        "pair_outcome_covariance": float(np.cov(a, b, ddof=1)[0, 1]),
        "discordant_pairs": int(np.count_nonzero(delta)),
    }


def build_report(
    cells: dict[tuple[str, int], dict[str, Any]], *, bootstrap_seed: int = 20260903,
    bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    """Validate alignment before computing any cross-budget or treatment effect."""
    expected = [(arm, budget) for arm in ARMS for budget in BUDGETS]
    if any(key not in expected for key in cells):
        raise ValueError("unexpected arm/search-budget cell")
    first = next(iter(cells.values()), None)
    candidate_by_arm: dict[str, str] = {}
    omitted = {"candidate", "sims_candidate", "sims_reference"}
    for (arm, budget), cell in cells.items():
        if cell["settings"].get("sims_candidate") != budget or cell["settings"].get("sims_reference") != budget:
            raise ValueError(f"{arm}:{budget}: cell label and realized simulation budget differ")
        if cell["execution"][0] != "on":
            raise ValueError(f"{arm}:{budget}: preregistration requires compile on")
        if first is not None:
            protocol = {k: v for k, v in cell["settings"].items() if k not in omitted}
            first_protocol = {k: v for k, v in first["settings"].items() if k not in omitted}
            if protocol != first_protocol or cell["execution"] != first["execution"]:
                raise ValueError(f"{arm}:{budget}: protocol or execution differs across cells")
            if cell["openings"] != first["openings"]:
                raise ValueError(f"{arm}:{budget}: pair IDs do not identify the same opening sequence")
        candidate = str(Path(cell["settings"]["candidate"]).resolve())
        if arm in candidate_by_arm and candidate_by_arm[arm] != candidate:
            raise ValueError(f"{arm}: candidate checkpoint differs across budgets")
        candidate_by_arm[arm] = candidate
    if len(set(candidate_by_arm.values())) != len(candidate_by_arm):
        raise ValueError("different arms identify the same candidate checkpoint")

    results = {}
    for key in expected:
        label = f"{key[0]}:{key[1]}"
        if key not in cells:
            results[label] = {"status": "UNREAD"}
            continue
        cell = cells[key]
        results[label] = {
            "status": "READ",
            **{k: v for k, v in cell.items() if k not in ("scores", "openings")},
            "strength_interpretation": {"SUCCESS": "PROMISING", "KILL": "LOSS",
                                        "INCONCLUSIVE": "INCONCLUSIVE"}[cell["result"]["verdict"]],
        }

    interactions = {}
    for arm in ARMS:
        low, high = (arm, 25), (arm, 400)
        missing = [f"{key[0]}:{key[1]}" for key in (low, high) if key not in cells]
        interactions[arm] = ({"status": "UNREAD", "missing_cells": missing} if missing else {
            "status": "READ", "contrast": "score400 - score25 against S0",
            **compare(cells[high]["scores"], cells[low]["scores"],
                      seed=bootstrap_seed, samples=bootstrap_samples),
        })

    comparisons = {}
    for budget in BUDGETS:
        left, right = ("G20T1", budget), ("G20T05", budget)
        missing = [f"{key[0]}:{key[1]}" for key in (left, right) if key not in cells]
        comparisons[str(budget)] = ({"status": "UNREAD", "missing_cells": missing} if missing else {
            "status": "READ", "contrast": "G20T1 - G20T05 score against S0",
            **compare(cells[left]["scores"], cells[right]["scores"],
                      seed=bootstrap_seed, samples=bootstrap_samples),
        })
    return {
        "cells": results, "search_interactions_25_to_400": interactions,
        "global_arm_comparisons_by_budget": comparisons,
        "missing_cells": [f"{arm}:{budget}" for arm, budget in expected if (arm, budget) not in cells],
        "screen_complete": len(cells) == len(expected),
        "bootstrap": {"unit": "aligned opening pair", "samples": bootstrap_samples,
                      "seed": bootstrap_seed, "method": "percentile"},
        "primary_strength_budget": 100, "deployment_relevant_budget": 400,
        "promotion": "NONE; prospective independent-seed/fresh-opening confirmation required",
        "limitations": [
            "Checkpoint/book contents and build provenance require the external run manifest.",
            "Score differences against S0 are not direct head-to-head Elo.",
            "Positive search interaction means increasing relative advantage against S0, not absolute improvement with search.",
            "Exploratory one-training-seed screen: nominal intervals without family-wise correction.",
            "No optimal recipe or promotion follows from this screen; report 400-simulation strength separately.",
            "E0 retains its historical runtime provenance limit; new arms require matched runtime evidence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", default=[], metavar="G20T1:25=GAMES.jsonl")
    parser.add_argument("--reference", type=Path, required=True, help="S0 checkpoint path recorded by arena")
    parser.add_argument("--seed", type=int, default=42, help="preregistered opening/search seed")
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        parser.error("--bootstrap-samples must be at least 1000")
    cells = {}
    try:
        for spec in args.cell:
            label, separator, filename = spec.partition("=")
            arm, colon, budget_text = label.partition(":")
            if not separator or not colon or arm not in ARMS or not filename:
                raise ValueError(f"invalid cell {spec!r}; use G20T1:25=path (or G20T05/E0;25/100/400)")
            budget = int(budget_text)
            key = (arm, budget)
            if budget not in BUDGETS or key in cells:
                raise ValueError(f"invalid budget or duplicate cell {label!r}")
            cells[key] = read_arm(Path(filename), reference=args.reference, seed=args.seed, sims=budget)
        report = build_report(cells, bootstrap_seed=args.bootstrap_seed,
                              bootstrap_samples=args.bootstrap_samples)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
