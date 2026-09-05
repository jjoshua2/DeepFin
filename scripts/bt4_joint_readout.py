#!/usr/bin/env python3
"""Read the fixed BT4 A-F arenas against E0 and compare successful arms.

Usage: PYTHONPATH=. python scripts/bt4_joint_readout.py --reference E0.pt \
    --arm A=A.games.jsonl --arm B=B.games.jsonl

Each supplied arm must contain all 500 color-swapped opening pairs. Replayed
orphan halves use the arena's last-write rule; the raw bank stays untouched.
Individual verdicts use arena_standard's pentanomial CI. Comparisons resample aligned
opening pairs, preserving outcome covariance against the common E0 opponent.
These are score advantages against E0, not direct head-to-head Elo estimates.
Omitted arms remain unread; selection is provisional until all admitted arms
are supplied. Checkpoint/book content hashes must be verified from the run
manifest: arena headers identify paths, not the bytes originally used.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
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

PREFERENCE = ("B", "A", "E", "D", "C", "F")
PAIRS = 500


def read_arm(path: Path, *, reference: Path, seed: int) -> dict[str, Any]:
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
        "games": 1000, "mode": "matched_sims", "sims_candidate": 100,
        "sims_reference": 100, "seed": seed, "opening_plies": 16,
        "openings_kind": "book", "max_plies": 300, "temperature": 0.1,
        "gumbel_add_noise": True,
    }
    wrong = {k: settings.get(k) for k, v in required.items() if settings.get(k) != v}
    if wrong:
        raise ValueError(f"{path}: off-protocol settings {wrong}")
    if not settings.get("openings") or not settings.get("candidate"):
        raise ValueError(f"{path}: missing opening book or candidate identity")
    if Path(settings.get("reference", "")).resolve() != reference.resolve():
        raise ValueError(f"{path}: reference is not the requested E0 checkpoint")
    if Path(settings["candidate"]).resolve() == reference.resolve():
        raise ValueError(f"{path}: candidate is E0 itself")
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
    """A-B score advantage against E0; bootstrap the matched pair differences."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="A=GAMES.jsonl")
    parser.add_argument("--reference", type=Path, required=True, help="E0 checkpoint path recorded by arena")
    parser.add_argument("--seed", type=int, default=42, help="preregistered opening/search seed")
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        parser.error("--bootstrap-samples must be at least 1000")
    arms = {}
    try:
        for spec in args.arm:
            label, separator, filename = spec.partition("=")
            if not separator or label not in PREFERENCE or label in arms:
                raise ValueError(f"invalid or duplicate arm {spec!r}; use A=path through F=path")
            arms[label] = read_arm(Path(filename), reference=args.reference, seed=args.seed)
        first = next(iter(arms.values()))
        for label, arm in arms.items():
            a = {k: v for k, v in arm["settings"].items() if k != "candidate"}
            b = {k: v for k, v in first["settings"].items() if k != "candidate"}
            if a != b or arm["execution"] != first["execution"]:
                raise ValueError(f"{label}: protocol or execution modes differ between arms")
            if arm["openings"] != first["openings"]:
                raise ValueError(f"{label}: pair IDs do not identify the same opening sequence")
        candidates = [str(Path(a["settings"]["candidate"]).resolve()) for a in arms.values()]
        if len(set(candidates)) != len(candidates):
            raise ValueError("multiple arms identify the same candidate checkpoint")
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))

    successful = [a for a in PREFERENCE if a in arms and arms[a]["result"]["verdict"] == "SUCCESS"]
    comparisons = {}
    for a, b in itertools.combinations(successful, 2):
        comparisons[f"{a}-{b}"] = compare(
            arms[a]["scores"], arms[b]["scores"], seed=args.bootstrap_seed,
            samples=args.bootstrap_samples,
        )
    leader = successful[0] if successful else "E0"
    selection = []
    for challenger in successful[1:]:
        comparison = comparisons[f"{leader}-{challenger}"]
        # Reverse the recorded leader-minus-challenger interval.
        lower = -comparison["ci95"][1]
        replace = lower > 0
        selection.append({"leader": leader, "challenger": challenger,
                          "challenger_advantage_lower": lower, "replaces": replace})
        if replace:
            leader = challenger
    report = {
        "arms": {k: {key: v for key, v in arm.items() if key not in ("scores", "openings")}
                 for k, arm in arms.items()},
        "successful_arm_comparisons": comparisons,
        "bootstrap": {"unit": "aligned opening pair", "samples": args.bootstrap_samples,
                      "seed": args.bootstrap_seed, "method": "percentile"},
        "selection_steps": selection, "preferred_among_supplied": leader,
        "omitted_arms": [a for a in PREFERENCE if a not in arms],
        "limitations": [
            "Checkpoint/book contents and build provenance require the external run manifest.",
            "Same-opening score differences against E0 are not direct head-to-head Elo.",
            "Nominal 95% intervals follow the preregistration; no family-wise correction or independent confirmation.",
            "Preference is provisional until every admitted arm has completed; omitted arms are unread.",
        ],
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
