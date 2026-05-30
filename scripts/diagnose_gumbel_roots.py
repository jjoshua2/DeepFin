#!/usr/bin/env python3
"""Compare root move quality across checkpoints and Gumbel simulation budgets.

The search itself is run once per simulation budget.  Action temperature can
then be swept post-hoc from the returned improved policy, which isolates
"search got worse with depth" from "sampling temperature is too noisy".
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_anti_engine.inference import LocalModelEvaluator  # noqa: E402
from chess_anti_engine.mcts.gumbel import GumbelConfig  # noqa: E402
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c  # noqa: E402
from chess_anti_engine.moves import index_to_move  # noqa: E402
from chess_anti_engine.stockfish import StockfishUCI  # noqa: E402
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint  # noqa: E402


@dataclass
class ScoredPosition:
    board: chess.Board
    scores: dict[str, float]
    best_score: float
    best_move: str


def _walk_positions(
    *,
    n: int,
    min_ply: int,
    max_ply: int,
    seed: int,
) -> list[chess.Board]:
    rng = random.Random(seed)
    positions: list[chess.Board] = []
    seen: set[str] = set()
    attempts = 0
    while len(positions) < n and attempts < max(100, n * 30):
        attempts += 1
        board = chess.Board()
        target_ply = rng.randint(min_ply, max_ply)
        ok = True
        for _ in range(target_ply):
            if board.is_game_over(claim_draw=True):
                ok = False
                break
            moves = list(board.legal_moves)
            if not moves:
                ok = False
                break
            board.push(rng.choice(moves))
        if not ok or board.is_game_over(claim_draw=True):
            continue
        fen = board.fen()
        if fen in seen or board.legal_moves.count() <= 1:
            continue
        seen.add(fen)
        positions.append(board.copy(stack=False))
    if len(positions) < n:
        raise RuntimeError(f"only generated {len(positions)} positions after {attempts} attempts")
    return positions


def _wdl_score(wdl: np.ndarray) -> float:
    win, draw, _loss = [float(x) for x in wdl]
    return win + 0.5 * draw


def _score_positions(
    boards: list[chess.Board],
    *,
    stockfish_path: str,
    sf_nodes: int,
    multipv: int,
    syzygy_path: str | None,
) -> list[ScoredPosition]:
    sf = StockfishUCI(
        stockfish_path,
        nodes=int(sf_nodes),
        multipv=int(multipv),
        syzygy_path=syzygy_path,
    )
    scored: list[ScoredPosition] = []
    try:
        for idx, board in enumerate(boards, 1):
            res = sf.search(board.fen())
            scores: dict[str, float] = {}
            for pv in res.pvs:
                if pv.wdl is None:
                    continue
                scores[pv.move_uci] = _wdl_score(pv.wdl)
            if not scores:
                print(f"[sf] {idx}/{len(boards)} skipped: no WDL PVs")
                continue
            best_move, best_score = max(scores.items(), key=lambda item: item[1])
            scored.append(
                ScoredPosition(
                    board=board,
                    scores=scores,
                    best_score=float(best_score),
                    best_move=str(best_move),
                )
            )
            if idx % 8 == 0 or idx == len(boards):
                print(f"[sf] scored {idx}/{len(boards)} positions")
    finally:
        sf.close()
    return scored


def _summarize(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    regrets = np.array([float(r["regret"]) for r in rows if int(r["missing"]) == 0], dtype=np.float64)
    ranks = np.array([float(r["rank"]) for r in rows if int(r["missing"]) == 0], dtype=np.float64)
    expected_regrets = np.array(
        [float(r["expected_regret"]) for r in rows if int(r["expected_missing"]) == 0],
        dtype=np.float64,
    )
    catastrophic_probs = np.array(
        [float(r["catastrophic_prob"]) for r in rows if int(r["expected_missing"]) == 0],
        dtype=np.float64,
    )
    missing = sum(int(r["missing"]) for r in rows)
    expected_missing = sum(int(r["expected_missing"]) for r in rows)
    if regrets.size == 0 and expected_regrets.size == 0:
        return {"n": 0, "missing": missing, "expected_missing": expected_missing}
    return {
        "n": int(regrets.size),
        "missing": int(missing),
        "expected_n": int(expected_regrets.size),
        "expected_missing": int(expected_missing),
        "mean_regret": float(regrets.mean()) if regrets.size else math.nan,
        "median_regret": float(np.median(regrets)) if regrets.size else math.nan,
        "p90_regret": float(np.percentile(regrets, 90)) if regrets.size else math.nan,
        "bad_gt_005": int((regrets > 0.05).sum()) if regrets.size else 0,
        "bad_gt_010": int((regrets > 0.10).sum()) if regrets.size else 0,
        "expected_regret": float(expected_regrets.mean()) if expected_regrets.size else math.nan,
        "expected_p90_regret": (
            float(np.percentile(expected_regrets, 90)) if expected_regrets.size else math.nan
        ),
        "catastrophic_prob_gt_010": (
            float(catastrophic_probs.mean()) if catastrophic_probs.size else math.nan
        ),
        "mean_sf_rank": float(ranks.mean()) if ranks.size else math.nan,
    }


def _rank_of_move(scores: dict[str, float], move_uci: str) -> int:
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    for rank, (uci, _score) in enumerate(ordered, 1):
        if uci == move_uci:
            return rank
    return len(ordered) + 1


def _policy_stats_for_temperature(
    *,
    board: chess.Board,
    scores: dict[str, float],
    best_score: float,
    prob: np.ndarray,
    temperature: float,
    argmax_action: int,
) -> tuple[float, float, float, float, int]:
    legal_actions: list[int] = []
    weights: list[float] = []
    regrets: list[float] = []
    catastrophic: list[float] = []
    for action, raw_weight in enumerate(prob):
        weight = float(raw_weight)
        if weight <= 0.0:
            continue
        move = index_to_move(int(action), board)
        if move not in board.legal_moves:
            continue
        move_score = scores.get(move.uci())
        if move_score is None:
            continue
        regret = max(0.0, float(best_score) - float(move_score))
        legal_actions.append(int(action))
        weights.append(weight)
        regrets.append(regret)
        catastrophic.append(1.0 if regret > 0.10 else 0.0)
    if not legal_actions:
        return math.nan, math.nan, math.nan, math.nan, 1
    w = np.asarray(weights, dtype=np.float64)
    if float(temperature) <= 0.0:
        # Match live temperature_resample(): temperature <= 0 preserves the
        # Gumbel sequential-halving survivor instead of switching to the
        # improved-policy argmax.
        chosen_idx = legal_actions.index(int(argmax_action)) if int(argmax_action) in legal_actions else int(np.argmax(w))
        p = np.zeros_like(w)
        p[chosen_idx] = 1.0
    else:
        p = np.power(np.maximum(w, 0.0), 1.0 / float(temperature))
        ps = float(p.sum())
        if not np.isfinite(ps) or ps <= 0.0:
            chosen_idx = legal_actions.index(int(argmax_action)) if int(argmax_action) in legal_actions else int(np.argmax(w))
            p = np.zeros_like(w)
            p[chosen_idx] = 1.0
        else:
            p /= ps
    regret_arr = np.asarray(regrets, dtype=np.float64)
    cat_arr = np.asarray(catastrophic, dtype=np.float64)
    entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    top1 = float(p.max()) if p.size else math.nan
    return float((p * regret_arr).sum()), float((p * cat_arr).sum()), entropy, top1, 0


def _analyze_checkpoint(
    *,
    label: str,
    checkpoint: str,
    scored: list[ScoredPosition],
    sims_list: list[int],
    device: str,
    seed: int,
    search_temperature: float,
    temperatures: list[float],
    add_noise: bool,
    topks: list[int],
    gumbel_scale: float,
) -> dict[str, Any]:
    print(f"\n[model] loading {label}: {checkpoint}")
    t0 = time.time()
    model = load_model_from_checkpoint(checkpoint, device=device)
    evaluator = LocalModelEvaluator(model, device=device)
    input_history = str(getattr(model, "input_history_encoding", "legacy"))
    boards = [p.board for p in scored]
    print(f"[model] loaded {label} in {time.time() - t0:.1f}s; input_history={input_history}")

    out: dict[str, Any] = {}
    for topk in topks:
        topk_key = f"topk_{int(topk)}"
        out[topk_key] = {}
        for sims in sims_list:
            cfg = GumbelConfig(
                simulations=int(sims),
                topk=int(topk),
                temperature=float(search_temperature),
                add_noise=bool(add_noise),
                gumbel_scale=float(gumbel_scale),
                input_history_encoding=input_history,
            )
            rng = np.random.default_rng(seed + int(sims) * 1009 + int(topk) * 9176)
            t1 = time.time()
            probs, actions, values, _legal_masks, _tree, _root_ids = run_gumbel_root_many_c(
                None,
                boards,
                device=device,
                rng=rng,
                cfg=cfg,
                evaluator=evaluator,
            )
            sims_key = f"sims_{sims}"
            rows_by_temp: dict[float, list[dict[str, float | int | str]]] = {
                float(t): [] for t in temperatures
            }
            value_sum = 0.0
            for pos, action, prob, value in zip(scored, actions, probs, values):
                move = index_to_move(int(action), pos.board)
                move_uci = move.uci()
                selected = pos.scores.get(move_uci)
                missing = selected is None
                regret = float("nan") if missing else float(pos.best_score - float(selected))
                value_sum += float(value)
                for temp in temperatures:
                    exp_regret, cat_prob, entropy, top1, expected_missing = _policy_stats_for_temperature(
                        board=pos.board,
                        scores=pos.scores,
                        best_score=pos.best_score,
                        prob=prob,
                        temperature=float(temp),
                        argmax_action=int(action),
                    )
                    rows_by_temp[float(temp)].append(
                        {
                            "move": move_uci,
                            "best_move": pos.best_move,
                            "regret": regret,
                            "rank": _rank_of_move(pos.scores, move_uci),
                            "missing": int(missing),
                            "expected_regret": exp_regret,
                            "catastrophic_prob": cat_prob,
                            "expected_missing": int(expected_missing),
                            "entropy": entropy,
                            "top1_prob": top1,
                        }
                    )
            out[topk_key][sims_key] = {}
            seconds = float(time.time() - t1)
            last_summary: dict[str, float | int] = {"n": 0, "seconds": seconds}
            for temp, rows in rows_by_temp.items():
                summary = _summarize(rows)
                entropies = np.asarray(
                    [float(r["entropy"]) for r in rows if int(r["expected_missing"]) == 0],
                    dtype=np.float64,
                )
                top1s = np.asarray(
                    [float(r["top1_prob"]) for r in rows if int(r["expected_missing"]) == 0],
                    dtype=np.float64,
                )
                summary["entropy"] = float(entropies.mean()) if entropies.size else math.nan
                summary["top1_prob"] = float(top1s.mean()) if top1s.size else math.nan
                summary["root_value"] = float(value_sum / max(1, len(rows)))
                summary["seconds"] = seconds
                out[topk_key][sims_key][f"temp_{temp:g}"] = summary
                last_summary = summary
            temp0_summary = out[topk_key][sims_key].get("temp_0", {})
            print(
                f"[{label}] topk={int(topk):>2} sims={sims:>3} n={last_summary.get('n', 0):>3} "
                f"search_regret={temp0_summary.get('mean_regret', math.nan):.4f} "
                f"exp_regret@0.35={out[topk_key][sims_key].get('temp_0.35', {}).get('expected_regret', math.nan):.4f} "
                f"cat@0.35={out[topk_key][sims_key].get('temp_0.35', {}).get('catastrophic_prob_gt_010', math.nan):.3f} "
                f"dt={float(last_summary['seconds']):.1f}s"
            )
    del evaluator
    del model
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="checkpoint A")
    parser.add_argument("--b", default="", help="optional checkpoint B")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--positions", type=int, default=48)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=60)
    parser.add_argument("--sims", default="1,4,16,64,128")
    parser.add_argument(
        "--search-temperature",
        type=float,
        default=0.0,
        help="temperature used inside the MCTS call; keep 0 to audit deterministic search",
    )
    parser.add_argument(
        "--temperatures",
        default="0,0.03,0.05,0.1,0.15,0.25,0.35,0.5,0.75,1.0",
        help="post-hoc action temperatures to evaluate from the same improved policy",
    )
    parser.add_argument("--add-noise", action="store_true")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument(
        "--topks",
        default="",
        help="Comma-separated top-k values to sweep. Defaults to --topk.",
    )
    parser.add_argument(
        "--gumbel-scale",
        type=float,
        default=1.0,
        help="Root Gumbel noise scale when --add-noise is set.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--stockfish-path",
        default="/home/josh/projects/chess/e2e_server/publish/stockfish",
    )
    parser.add_argument("--sf-nodes", type=int, default=10000)
    parser.add_argument("--multipv", type=int, default=80)
    parser.add_argument("--syzygy-path", default="/home/josh/projects/chess/data/syzygy_3-4-5")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    sims = [int(x.strip()) for x in str(args.sims).split(",") if x.strip()]
    topks = (
        [int(x.strip()) for x in str(args.topks).split(",") if x.strip()]
        if str(args.topks).strip()
        else [int(args.topk)]
    )
    temperatures = [float(x.strip()) for x in str(args.temperatures).split(",") if x.strip()]
    if not sims or any(x <= 0 for x in sims):
        raise SystemExit("--sims must contain positive integers")
    if not topks or any(x <= 0 for x in topks):
        raise SystemExit("--topks must contain positive integers")
    if not temperatures or any(x < 0.0 for x in temperatures):
        raise SystemExit("--temperatures must contain non-negative floats")
    if not math.isfinite(float(args.gumbel_scale)) or float(args.gumbel_scale) < 0.0:
        raise SystemExit("--gumbel-scale must be finite and non-negative")
    if args.min_ply > args.max_ply:
        raise SystemExit("--min-ply must be <= --max-ply")

    print(
        f"[setup] generating {args.positions} positions, plies {args.min_ply}..{args.max_ply}, "
        f"seed={args.seed}"
    )
    boards = _walk_positions(
        n=int(args.positions),
        min_ply=int(args.min_ply),
        max_ply=int(args.max_ply),
        seed=int(args.seed),
    )
    print(
        f"[setup] scoring with SF nodes={args.sf_nodes}, multipv={args.multipv}, "
        f"stockfish={args.stockfish_path}"
    )
    scored = _score_positions(
        boards,
        stockfish_path=str(args.stockfish_path),
        sf_nodes=int(args.sf_nodes),
        multipv=int(args.multipv),
        syzygy_path=str(args.syzygy_path) if args.syzygy_path else None,
    )
    if not scored:
        raise SystemExit("no scored positions")

    result: dict[str, object] = {
        "positions": len(scored),
        "seed": int(args.seed),
        "sf_nodes": int(args.sf_nodes),
        "multipv": int(args.multipv),
        "sims": sims,
        "topks": topks,
        "search_temperature": float(args.search_temperature),
        "temperatures": temperatures,
        "add_noise": bool(args.add_noise),
        "gumbel_scale": float(args.gumbel_scale),
        "models": {},
    }
    models = {
        str(args.label_a): _analyze_checkpoint(
            label=str(args.label_a),
            checkpoint=str(args.a),
            scored=scored,
            sims_list=sims,
            device=str(args.device),
            seed=int(args.seed),
            search_temperature=float(args.search_temperature),
            temperatures=temperatures,
            add_noise=bool(args.add_noise),
            topks=topks,
            gumbel_scale=float(args.gumbel_scale),
        )
    }
    if args.b:
        models[str(args.label_b)] = _analyze_checkpoint(
                label=str(args.label_b),
                checkpoint=str(args.b),
                scored=scored,
                sims_list=sims,
                device=str(args.device),
                seed=int(args.seed) + 777_777,
                search_temperature=float(args.search_temperature),
                temperatures=temperatures,
                add_noise=bool(args.add_noise),
                topks=topks,
                gumbel_scale=float(args.gumbel_scale),
            )
    result["models"] = models

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
