#!/usr/bin/env python3
"""Read the global BT4 screen against S0 across 25, 100 and 400 simulations.

Usage: PYTHONPATH=. python scripts/bt4_joint_readout.py --reference S0.pt \
    --cell G20T1:25=G20T1.25.games.jsonl --cell G20T1:400=G20T1.400.games.jsonl

Every supplied cell must contain 500 color-swapped opening pairs. Replayed
orphan halves use the producing arena's last-write rule; raw banks stay untouched.
Cross-cell comparisons resample aligned opening pairs against the common S0
opponent. These score differences are not direct head-to-head Elo estimates.
Missing cells remain unread. This exploratory screen never promotes a recipe.
Use --profile sf-close with C20T05 cells and --reference E0.pt for the
separate direct comparison of sharpened SF-close targets against exact ties.
Both sides must realize the requested search-prior temperature (default 1.0).
Reading superseded softened-prior banks requires explicit --prior-temperature 1.5.
Separately registered confirmation banks require --confirmation-openings FILE
and --confirmation-sha256 SHA; game-log endpoints cannot prove launch-time history.
Use --profile confirmation for one direct match with explicit --candidate,
--candidate-role, --reference-role and --confirmation-sims. This does not certify
independent training or select a recipe; those require the prospective run record.
Checkpoint/book content and build hashes require the external run manifest;
arena headers record paths rather than the bytes originally used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.utils.game_log import (
    latest_rows_by_key,
    read_game_log,
    settings_fingerprint,
)
from scripts.arena_standard import load_fen_openings, pentanomial_counts, summarize_pentanomial

ARMS = ("G20T1", "G20T05", "E0")
PROFILE_ARMS = {"global": ARMS, "sf-close": ("C20T05",), "calibration": ("S0", "E0")}
CONFIRMATION_ROLES = ("S0", "E0", "E0T05", "G20T1", "G20T05", "C20T05")
PRIOR_TEMPERATURES = (1.0, 1.5)
BUDGETS = (25, 100, 400)
PAIRS = 500


def confirmation_input(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Validate a fixed history-bearing bank; launch provenance remains external."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(expected_sha256) != 64 or digest != expected_sha256:
        raise ValueError("confirmation opening file SHA256 differs from expected identity")
    lines = path.read_text().splitlines()
    if len(lines) != PAIRS:
        raise ValueError("confirmation requires exactly 500 history-bearing opening lines")
    expected = []
    for line in lines:
        start, separator, moves = line.partition("|")
        if not separator or len(moves.split()) != 16:
            raise ValueError("confirmation opening must contain exactly 16 history moves")
        board = chess.Board(start.strip())
        if not board.is_valid():
            raise ValueError("invalid confirmation root position")
        for token in moves.split():
            board.push_uci(token)
        if not board.is_valid() or board.is_game_over() or board.legal_moves.count() < 2:
            raise ValueError("unusable confirmation opening")
        expected.append(board)
    if len({board.epd() for board in expected}) != PAIRS:
        raise ValueError("confirmation terminal positions must be unique")
    loaded = load_fen_openings(path, n_pairs=PAIRS, rng=np.random.default_rng(0))
    if len(loaded) != PAIRS or any(
        actual.fen() != wanted.fen() or actual.root().fen() != wanted.root().fen()
        or actual.move_stack != wanted.move_stack
        for actual, wanted in zip(loaded, expected, strict=True)
    ):
        raise ValueError("arena loader changed confirmation opening order or history")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("confirmation opening file changed during validation")
    return {"path": str(path.resolve()), "sha256": digest,
            "fens": [board.fen() for board in expected], "history_plies": 16,
            "limitation": "File and loader history verified now; game logs record endpoints only. "
            "Launch-time file bytes, loader/build identity and realized history require an external launch receipt."}


def read_arm(
    path: Path, *, reference: Path, seed: int, sims: int = 100,
    prior_temperature: float = 1.0, calibration: bool = False,
    confirmation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refuse incomplete, conflicting or off-protocol banks before scoring."""
    if confirmation is not None and calibration:
        raise ValueError("confirmation input is not part of the fixed calibration protocol")
    log = read_game_log(path)
    settings = log.settings
    if log.truncated_tail:
        raise ValueError(f"{path}: require a complete game bank without a torn tail")
    if log.header.get("driver") != "arena_standard" or log.header.get("version") != 1:
        raise ValueError(f"{path}: unsupported arena header")
    if log.fingerprint != settings_fingerprint(settings):
        raise ValueError(f"{path}: header fingerprint disagrees with its settings")
    required: dict[str, Any] = {
        "games": 1000, "mode": "matched_sims", "sims_candidate": sims,
        "sims_reference": sims, "seed": seed, "opening_plies": 16,
        "openings_kind": "book", "max_plies": 300, "temperature": 0.1,
        "gumbel_add_noise": True,
    }
    if confirmation is not None:
        required.update(openings_kind="fen", opening_plies=None)
        if "opening_plies" not in settings:
            raise ValueError("confirmation header lacks explicit FEN opening protocol")
        if Path(settings.get("openings", "")).resolve() != Path(confirmation["path"]):
            raise ValueError("arena opening path differs from expected confirmation file")
    wrong = {k: settings.get(k) for k, v in required.items() if settings.get(k) != v}
    if wrong:
        raise ValueError(f"{path}: off-protocol settings {wrong}")
    if not settings.get("openings") or not settings.get("candidate"):
        raise ValueError(f"{path}: missing opening book or candidate identity")
    if Path(settings.get("reference", "")).resolve() != reference.resolve():
        raise ValueError(f"{path}: reference is not the requested control checkpoint")
    if calibration and Path(settings["candidate"]).resolve() != reference.resolve():
        raise ValueError(f"{path}: calibration candidate is not the expected checkpoint")
    if not calibration and Path(settings["candidate"]).resolve() == reference.resolve():
        raise ValueError(f"{path}: candidate is the control itself")
    search = settings.get("search_candidate")
    if not isinstance(search, dict) or search.get("shape") != "training":
        raise ValueError(f"{path}: candidate lacks realized training search settings")
    reference_search = settings.get("search_reference")
    if calibration:
        if sims != 100 or prior_temperature != 1.0:
            raise ValueError("calibration requires 100 simulations and candidate prior 1.0")
        if not isinstance(reference_search, dict):
            raise ValueError("calibration reference search missing")
        adjusted = {**reference_search, "gumbel": {**reference_search.get("gumbel", {}), "policy_temp": 1.0}}
        if (reference_search.get("gumbel", {}).get("policy_temp") != 1.5
                or {k: v for k, v in search.items() if k != "source"}
                != {k: v for k, v in adjusted.items() if k != "source"}):
            raise ValueError("calibration must differ only in candidate 1.0 versus reference 1.5 prior")
    elif search != reference_search:
        raise ValueError(f"{path}: candidate and reference search settings differ")
    if prior_temperature not in PRIOR_TEMPERATURES:
        raise ValueError(f"unsupported registered prior temperature {prior_temperature!r}")
    if search.get("gumbel", {}).get("policy_temp") != prior_temperature:
        raise ValueError(f"{path}: realized prior temperature differs from requested {prior_temperature}")
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
        if confirmation is not None and row["opening_fen"] != confirmation["fens"][pair]:
            raise ValueError(f"{path}: confirmation opening endpoint/order differs at {key}")
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
        "confirmation_input": None if confirmation is None else {
            k: v for k, v in confirmation.items() if k != "fens"
        },
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
    bootstrap_samples: int = 10000, profile: str = "global", prior_temperature: float = 1.0,
) -> dict[str, Any]:
    """Validate alignment before computing any cross-budget or treatment effect."""
    if profile not in PROFILE_ARMS or profile == "calibration":
        raise ValueError(f"use calibration_report for calibration; unknown strength profile {profile!r}")
    if prior_temperature not in PRIOR_TEMPERATURES:
        raise ValueError(f"unsupported registered prior temperature {prior_temperature!r}")
    arms = PROFILE_ARMS[profile]
    reference_label = "E0" if profile == "sf-close" else "S0"
    expected = [(arm, budget) for arm in arms for budget in BUDGETS]
    if any(key not in expected for key in cells):
        raise ValueError("unexpected arm/search-budget cell")
    first = next(iter(cells.values()), None)
    candidate_by_arm: dict[str, str] = {}
    omitted = {"candidate", "sims_candidate", "sims_reference"}
    for (arm, budget), cell in cells.items():
        for side in ("search_candidate", "search_reference"):
            if cell["settings"].get(side, {}).get("gumbel", {}).get("policy_temp") != prior_temperature:
                raise ValueError(f"{arm}:{budget}: {side} prior temperature differs from requested {prior_temperature}")
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
    for arm in arms:
        low, high = (arm, 25), (arm, 400)
        missing = [f"{key[0]}:{key[1]}" for key in (low, high) if key not in cells]
        interactions[arm] = ({"status": "UNREAD", "missing_cells": missing} if missing else {
            "status": "READ", "contrast": f"score400 - score25 against {reference_label}",
            **compare(cells[high]["scores"], cells[low]["scores"],
                      seed=bootstrap_seed, samples=bootstrap_samples),
        })

    comparisons = {}
    for budget in BUDGETS if profile == "global" else ():
        left, right = ("G20T1", budget), ("G20T05", budget)
        missing = [f"{key[0]}:{key[1]}" for key in (left, right) if key not in cells]
        comparisons[str(budget)] = ({"status": "UNREAD", "missing_cells": missing} if missing else {
            "status": "READ", "contrast": "G20T1 - G20T05 score against S0",
            **compare(cells[left]["scores"], cells[right]["scores"],
                      seed=bootstrap_seed, samples=bootstrap_samples),
        })
    return {
        "profile": profile, "reference_role": reference_label, "prior_temperature": prior_temperature,
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
            f"Score differences against {reference_label} are not direct head-to-head Elo between candidate recipes.",
            f"Positive search interaction means increasing relative advantage against {reference_label}, not absolute improvement with search.",
            "Exploratory one-training-seed screen: nominal intervals without family-wise correction.",
            "No optimal recipe or promotion follows from this screen; report 400-simulation strength separately.",
            "E0 retains its historical runtime provenance limit; new arms require matched runtime evidence.",
        ] + ([
            "C20T05 changes both support and BT4 temperature versus E0; this is a recipe comparison, not an isolated support effect.",
            "Direct-E0 Elo cannot be subtracted from the global screen's direct-S0 Elo to infer head-to-head strength.",
        ] if profile == "sf-close" else []),
    }


def calibration_report(cells: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    """Report two within-checkpoint temperature contrasts, without pooling them."""
    expected = [("S0", 100), ("E0", 100)]
    if any(key not in expected for key in cells):
        raise ValueError("calibration accepts only S0:100 and E0:100")
    first = next(iter(cells.values()), None)
    checkpoints = set()
    for cell in cells.values():
        settings = cell["settings"]
        if Path(settings["candidate"]).resolve() != Path(settings["reference"]).resolve():
            raise ValueError("calibration must use the same checkpoint on both sides")
        if settings["sims_candidate"] != 100 or settings["sims_reference"] != 100:
            raise ValueError("calibration requires 100 simulations on both sides")
        reference_search = settings["search_reference"]
        adjusted = {**reference_search, "gumbel": {**reference_search["gumbel"], "policy_temp": 1.0}}
        if ({k: v for k, v in settings["search_candidate"].items() if k != "source"}
                != {k: v for k, v in adjusted.items() if k != "source"}):
            raise ValueError("calibration search settings differ beyond temperature")
        checkpoint = str(Path(settings["candidate"]).resolve())
        if checkpoint in checkpoints:
            raise ValueError("calibration nets must identify distinct checkpoints")
        checkpoints.add(checkpoint)
        for side, temperature in (("candidate", 1.0), ("reference", 1.5)):
            if settings[f"search_{side}"]["gumbel"]["policy_temp"] != temperature:
                raise ValueError("calibration prior temperatures differ")
        if cell["execution"][0] != "on":
            raise ValueError("calibration requires compile on")
        if first is not None:
            omitted = {"candidate", "reference"}
            if ({k: v for k, v in settings.items() if k not in omitted}
                    != {k: v for k, v in first["settings"].items() if k not in omitted}
                    or cell["execution"] != first["execution"]
                    or cell["openings"] != first["openings"]):
                raise ValueError("calibration protocol/execution/opening alignment differs")
    return {
        "profile": "calibration", "candidate_prior_temperature": 1.0,
        "reference_prior_temperature": 1.5,
        "cells": {f"{arm}:{budget}": ({"status": "READ", **{
            k: v for k, v in cells[(arm, budget)].items() if k not in ("scores", "openings")
        }} if (arm, budget) in cells else {"status": "UNREAD"}) for arm, budget in expected},
        "screen_complete": len(cells) == 2,
        "contrast": "Within each fixed checkpoint: prior 1.0 versus prior 1.5 at 100 simulations",
        "limitations": [
            "Nominal paired 95% intervals; no pooling across checkpoints or family-wise correction.",
            "This calibrates inference on two checkpoints, not training-temperature transfer or an optimal prior.",
            "Checkpoint/book bytes and build provenance require the external launch manifest.",
        ],
    }


def confirmation_report(
    cells: dict[tuple[str, int], dict[str, Any]], *, candidate_role: str,
    reference_role: str, candidate: Path, reference: Path, sims: int,
) -> dict[str, Any]:
    """Report one explicitly identified direct match; training proof stays external."""
    if (candidate_role not in CONFIRMATION_ROLES or reference_role not in CONFIRMATION_ROLES
            or candidate_role == reference_role or sims not in BUDGETS):
        raise ValueError("confirmation requires distinct recipe roles and an explicit supported budget")
    key = (candidate_role, sims)
    if set(cells) != {key}:
        raise ValueError("confirmation requires exactly the selected arm/search-budget cell")
    cell = cells[key]
    if cell.get("confirmation_input") is None:
        raise ValueError("confirmation requires the pinned history-bearing opening bank")
    settings = cell["settings"]
    for side, expected in (("candidate", candidate), ("reference", reference)):
        if Path(settings[side]).resolve() != expected.resolve():
            raise ValueError(f"confirmation {side} differs from requested checkpoint")
        if settings[f"sims_{side}"] != sims:
            raise ValueError("confirmation cell and realized simulation budget differ")
        if settings[f"search_{side}"]["gumbel"]["policy_temp"] != 1.0:
            raise ValueError("confirmation requires common prior 1.0")
    if cell["execution"] != ["on", "4096"]:
        raise ValueError("confirmation requires compile on and evaluator-hoist tag 4096")
    return {
        "profile": "confirmation", "candidate_role": candidate_role,
        "reference_role": reference_role, "prior_temperature": 1.0,
        "primary_strength_budget": sims,
        "contrast": f"{candidate_role} directly against {reference_role}",
        "cells": {f"{candidate_role}:{sims}": {
            "status": "READ",
            **{k: v for k, v in cell.items() if k not in ("scores", "openings")},
        }},
        "match_complete": True,
        "training_provenance_verified": False,
        "promotion": "NONE; apply the prospective decision rule after external provenance verification",
        "limitations": [
            "Nominal paired 95% interval: mean pair score plus/minus 1.96 standard errors, "
            "with score bounds transformed to Elo; no bootstrap interval.",
            "Recipe labels and checkpoint paths are explicit inputs; checkpoint bytes, recipe identity, "
            "new matched training seed and realized schedules require the external run manifest.",
            "Opening history is validated now; launch/resume bytes, loader/runtime identity and "
            "configured evaluation batch require external receipts, as does non-overlap with screening openings.",
            "This report does not establish an optimal recipe, search scaling, or automatic promotion.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=(*PROFILE_ARMS, "confirmation"), default="global")
    parser.add_argument("--prior-temperature", type=float, choices=PRIOR_TEMPERATURES,
                        default=1.0, help="realized search prior; use 1.5 only for superseded banks")
    parser.add_argument("--cell", action="append", default=[], metavar="G20T1:25=GAMES.jsonl")
    parser.add_argument("--reference", type=Path,
                        help="control checkpoint recorded by arena: S0 for global, E0 for sf-close")
    parser.add_argument("--candidate", type=Path, help="confirmation only: expected candidate checkpoint")
    parser.add_argument("--candidate-role", choices=CONFIRMATION_ROLES, help="confirmation candidate recipe")
    parser.add_argument("--reference-role", choices=CONFIRMATION_ROLES, help="confirmation reference recipe")
    parser.add_argument("--confirmation-sims", type=int, choices=BUDGETS,
                        help="confirmation only: prospectively selected direct-match budget")
    parser.add_argument("--confirmation-openings", type=Path,
                        help="explicit fixed history-bearing FEN bank for a separately registered confirmation")
    parser.add_argument("--confirmation-sha256", help="expected immutable opening file SHA256")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="S0=PATH",
                        help="calibration only: expected S0/E0 checkpoint paths")
    parser.add_argument("--seed", type=int, default=42, help="preregistered opening/search seed")
    parser.add_argument("--bootstrap-seed", type=int, help="screen interactions only; default 20260903")
    parser.add_argument("--bootstrap-samples", type=int, help="screen interactions only; default 10000")
    args = parser.parse_args()
    if args.bootstrap_samples is not None and args.bootstrap_samples < 1000:
        parser.error("--bootstrap-samples must be at least 1000")
    cells = {}
    try:
        is_confirmation = args.profile == "confirmation"
        confirmation_options = (args.candidate, args.candidate_role, args.reference_role, args.confirmation_sims)
        if is_confirmation:
            if args.bootstrap_seed is not None or args.bootstrap_samples is not None:
                raise ValueError("confirmation uses an analytic paired interval; bootstrap options are unsupported")
            if not all(confirmation_options) or not args.confirmation_openings or args.prior_temperature != 1.0:
                raise ValueError("confirmation requires --candidate, both recipe roles, --confirmation-sims, "
                                 "pinned confirmation openings and prior 1.0")
        elif any(option is not None for option in confirmation_options):
            raise ValueError("candidate/recipe-role/confirmation-sims options require --profile confirmation")
        confirmation = None
        if bool(args.confirmation_openings) != bool(args.confirmation_sha256):
            raise ValueError("confirmation requires both --confirmation-openings and --confirmation-sha256")
        if args.confirmation_openings:
            if args.profile == "calibration":
                raise ValueError("confirmation input is not part of calibration")
            confirmation = confirmation_input(args.confirmation_openings, args.confirmation_sha256)
        checkpoints = {}
        for spec in args.checkpoint:
            name, sep, filename = spec.partition("=")
            if not sep or name not in ("S0", "E0") or not filename or name in checkpoints:
                raise ValueError("use unique --checkpoint S0=PATH and E0=PATH")
            checkpoints[name] = Path(filename)
        calibration = args.profile == "calibration"
        if calibration:
            if set(checkpoints) != {"S0", "E0"} or args.reference is not None or args.prior_temperature != 1.0:
                raise ValueError("calibration requires both --checkpoint identities, no --reference, and prior 1.0")
        elif args.reference is None or checkpoints:
            raise ValueError("strength profiles require --reference and no --checkpoint")
        allowed_arms = (args.candidate_role,) if is_confirmation else PROFILE_ARMS[args.profile]
        for spec in args.cell:
            label, separator, filename = spec.partition("=")
            arm, colon, budget_text = label.partition(":")
            if not separator or not colon or arm not in allowed_arms or not filename:
                raise ValueError(f"invalid cell {spec!r} for {args.profile}; use ARM:SIMS=path")
            budget = int(budget_text)
            key = (arm, budget)
            if budget not in ((100,) if calibration else BUDGETS) or key in cells:
                raise ValueError(f"invalid budget or duplicate cell {label!r}")
            reference = checkpoints[arm] if calibration else args.reference
            if not isinstance(reference, Path):
                raise ValueError("expected checkpoint path is missing")
            cells[key] = read_arm(Path(filename), reference=reference, seed=args.seed,
                                  sims=budget, prior_temperature=args.prior_temperature, calibration=calibration, confirmation=confirmation)
        if is_confirmation:
            if not isinstance(args.reference, Path):
                raise ValueError("confirmation requires an explicit reference checkpoint")
            report = confirmation_report(cells, candidate_role=args.candidate_role,
                                         reference_role=args.reference_role, candidate=args.candidate,
                                         reference=args.reference, sims=args.confirmation_sims)
        elif calibration:
            report = calibration_report(cells)
        else:
            report = build_report(cells, bootstrap_seed=args.bootstrap_seed if args.bootstrap_seed is not None else 20260903,
                                  bootstrap_samples=args.bootstrap_samples if args.bootstrap_samples is not None else 10000,
                                  profile=args.profile,
                                  prior_temperature=args.prior_temperature)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
