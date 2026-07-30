#!/usr/bin/env python3
"""Net-side admission vet for blind-spot seeds — "does OUR net actually blunder here".

Why this exists, and why `blindspot_deepsf_gate.py` cannot do it
----------------------------------------------------------------
A harvested seed line is ``<start_fen> | <uci...>`` whose TERMINAL is the
position the net was about to move in. The blunder itself is **not stored**:
the harvest annotation's ``sq`` labels the position AFTER the net's move.

`blindspot_deepsf_gate.py` scores the terminal and keeps seeds deep SF calls
LOST. On current harvest output that keeps **nothing** — measured 2026-07-30,
0 of 80 — because the terminal is the position BEFORE the blunder and deep SF
correctly calls it winning (median q +1.000 on the newest files; several are
forced mates). The gate is not broken, it is one ply early: it asks "is this
position lost?" when the seed's whole point is "this position is FINE and the
net throws it away".

So the admission test has to be net-side, which is exactly the documented
criterion: **the seed is playable AND the net's own top move drops it to lost.**

    q_before = deep-SF eval of the terminal        (terminal mover's POV)
    q_after  = deep-SF eval after the NET's top move (SAME POV)
    keep if q_before >= --fine  AND  q_after <= --lost

Both evals are held to the terminal mover's POV, so a kept seed always reads
"was winning/OK, is now lost" in one consistent sign convention.

Two-stage by cost: every candidate gets a cheap screen, only survivors pay the
calibrated deep budget (4M nodes — `blindspot_deepsf_calibrate` found 4M=8M=16M).

NEGATIVE CONTROL (`--control random`): replaces the net's top move with a
uniformly random legal move. The keep rate MUST drop substantially. If a random
move keeps seeds at the same rate, this tool is measuring "positions where most
moves lose", not "positions the net misplays", and its output is worthless.
Run it before trusting a batch.

⚠ THE CONTROL ALREADY CAUGHT THIS TOOL ONCE — 2026-07-30, and the guard below
is the fix. With only the two-threshold rule (`before >= fine and after
<= lost`) the rates were **net 20/42 = 48% vs random 20/53 = 38%**: a random
legal move condemned the position almost as often as the net's own. Cause: the
current harvest pool is dominated by forced-mate-for-the-mover positions where
nearly EVERY non-mating move throws the game away, so "the net's move loses" is
close to vacuous.

`--min-safe-frac` closes it. For each seed we evaluate a SAMPLE of legal moves
(``--safe-frac-sample``, drawn from a position-seeded rng so both arms survey
the identical set) and compute ``safe_frac`` = the fraction that keep the mover
at >= ``--fine``. Surveying every legal move was tried first and cost ~2.1M
nodes/seed — one keep in 90 minutes — which is why this is a sample. A
seed is only admitted when the position was FORGIVING (``safe_frac >=
--min-safe-frac``) and the net still chose a losing move — i.e. the net erred
where most moves were fine. On a knife-edge mate-in-2, ``safe_frac`` is tiny
and the seed is rejected no matter what the net played, which is exactly what
makes the random control fail as it should.

⚠⚠ THE CONTROL CAUGHT IT A SECOND TIME — 2026-07-30, same day, and the cause
was the *cost fix above*. Rates were **net 6/80 = 7.5% vs random 4/80 = 5.0%**
(Fisher p≈0.75, indistinguishable). The keep rate had fallen a long way, but
the CONTROL FELL WITH IT: net:random only moved 1.26 → 1.50. Two positions were
deep-checked in both arms, with DIFFERENT moves, and both arms KEPT both.

Cause, and the lesson worth more than the tool: **the guard was measured at a
different search depth from the criterion it guards.** Keep required
``after <= --lost`` at 4,000,000 nodes; ``safe_frac`` counted a move "safe" at
``cq >= --fine`` on **30,000** nodes — 1/133rd the search. A move reading +0.3
at 30k reads −0.9 at 4M, so ``safe_frac`` was systematically INFLATED and the
guard was measuring *search depth*, not forgiveness. The tell was visible in
the output and is worth recognising on sight: kept rows with
``safe_frac=0.92`` — 92% of moves supposedly fine — on positions a RANDOM move
threw away. That is arithmetically absurd and was the signature of the bug.

Note what was NOT wrong: sampling instead of surveying every move is fine, and
the position-seeded rng is fine. Only the node budget was wrong. Hence
``--safe-frac-nodes`` now defaults to ``--screen-nodes`` rather than to a cheap
constant. **Any threshold compared against another threshold must be measured
with the same instrument at the same setting** — the cost of the guard is not a
free parameter, it is part of the guard's meaning.

History note: boards come from ``seed_board_from_line``, which replays the move
list, so the encoder sees real LC0 history. Do NOT swap in ``chess.Board(fen)``
— a bare FEN zeroes the history planes and silently changes the net's top move.
"""
from __future__ import annotations

import argparse
import json
import random

import chess
import chess.engine
import numpy as np
import torch

from chess_anti_engine.encoding import model_encoding_kwargs
from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.moves.encode import move_to_index_for_encoding
from chess_anti_engine.selfplay.opening import seed_board_from_line
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

_DEFAULT_SF = "/home/josh/projects/chess/e2e_server/publish/stockfish"


def _q(info: chess.engine.InfoDict, pov: chess.Color) -> float:
    """WDL win-minus-loss in [-1, 1] from ``pov``'s side."""
    score = info.get("score")
    if score is None:
        raise RuntimeError("stockfish returned no score for an analysed position")
    wdl = score.pov(pov).wdl()
    return (wdl.wins - wdl.losses) / 1000.0


def _seed_lines(path: str) -> list[str]:
    """Seed bodies, comments stripped (the feed grammar the worker parses)."""
    out: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            body = raw.split("#")[0].strip()
            if body:
                out.append(body)
    return out


def _net_top_move(
    board: chess.Board, ev: LocalModelEvaluator, enc_kwargs: dict,
    policy_encoding: str | None, use_rel: bool,
) -> chess.Move | None:
    """The net's raw policy argmax over LEGAL moves, or None if none exist.

    Argmax is taken over legal moves only, so an illegal index can never be
    returned; the net's mass on illegal moves is irrelevant to what it plays.
    """
    legal = list(board.legal_moves)
    if not legal:
        return None
    cb = CBoard.from_board(board)
    x = encode_cboard(cb, **enc_kwargs)
    rel = np.stack([cb.compute_relations()]) if use_rel else None
    with torch.no_grad():
        policy, _ = ev.evaluate_encoded(np.stack([x]), relations=rel)
    pol = np.asarray(policy, dtype=np.float64)[0]
    best, best_p = legal[0], -np.inf
    for mv in legal:
        idx = move_to_index_for_encoding(
            mv, board, policy_encoding=policy_encoding,
        )
        if idx < 0 or idx >= pol.size:
            continue
        if pol[idx] > best_p:
            best, best_p = mv, float(pol[idx])
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="harvest or feed-format seed file")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stockfish", default=_DEFAULT_SF)
    ap.add_argument("--screen-nodes", type=int, default=300_000)
    ap.add_argument("--deep-nodes", type=int, default=4_000_000)
    ap.add_argument("--hash-mb", type=int, default=512)
    ap.add_argument("--syzygy-path", default="data/syzygy_3-4-5-6")
    ap.add_argument("--fine", type=float, default=0.20,
                    help="terminal must be at least this good for the mover")
    ap.add_argument("--lost", type=float, default=-0.50,
                    help="after the net's move it must be at most this")
    ap.add_argument("--min-safe-frac", type=float, default=0.5,
                    help="fraction of LEGAL moves that must keep the mover >= --fine. "
                         "Rejects knife-edge positions where any move loses; without "
                         "it a random move scores like the net's (see module docstring)")
    ap.add_argument("--safe-frac-sample", type=int, default=12,
                    help="max legal moves surveyed per seed; sampled from a "
                         "POSITION-seeded rng so both control arms survey the same set")
    ap.add_argument("--safe-frac-nodes", type=int, default=0,
                    help="per-legal-move budget for the safe_frac survey. 0 (default) "
                         "means USE --screen-nodes, which is the only defensible "
                         "setting: safe_frac is compared against the screen's verdict "
                         "on the arm's move, so both must be measured at the SAME "
                         "depth. Overriding this to a cheaper budget silently breaks "
                         "the guard — see the docstring's second control failure")
    ap.add_argument("--max-keep", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="0 = all seeds")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--control", choices=["net", "random"], default="net",
                    help="'random' = negative control; keep rate must collapse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-list", required=True)
    ap.add_argument("--out-jsonl", default="")
    args = ap.parse_args()

    # The survey must run at the screen's depth: safe_frac is only meaningful
    # relative to the verdict the screen reached on the arm's own move, and a
    # cheaper budget makes "safe" mean "not yet refuted" instead of "fine".
    if args.safe_frac_nodes <= 0:
        args.safe_frac_nodes = args.screen_nodes

    rng = random.Random(args.seed)
    lines = _seed_lines(args.seeds)
    if args.limit:
        lines = lines[: args.limit]

    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    enc_kwargs = model_encoding_kwargs(model)
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    policy_encoding = getattr(model, "policy_encoding", None)
    ev = LocalModelEvaluator(model, device=args.device)

    eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    opts: dict[str, str | int] = {"Threads": 1, "Hash": int(args.hash_mb)}
    if args.syzygy_path:
        opts["SyzygyPath"] = str(args.syzygy_path)
    eng.configure(opts)

    kept: list[str] = []
    audit: list[dict] = []
    n_parse = n_noleg = n_screen_drop = n_knife_edge = 0

    try:
        for line in lines:
            if len(kept) >= args.max_keep:
                break
            try:
                board = seed_board_from_line(line)
            except Exception:
                n_parse += 1
                continue
            if board.is_game_over():
                n_noleg += 1
                continue
            mover = board.turn

            if args.control == "random":
                mv = rng.choice(list(board.legal_moves))
            else:
                mv = _net_top_move(board, ev, enc_kwargs, policy_encoding, use_rel)
            if mv is None:
                n_noleg += 1
                continue

            # Stage 1: cheap screen on both sides of the move.
            before = _q(eng.analyse(board, chess.engine.Limit(nodes=args.screen_nodes)), mover)
            board.push(mv)
            after = _q(eng.analyse(board, chess.engine.Limit(nodes=args.screen_nodes)), mover) \
                if not board.is_game_over() else (
                    -1.0 if board.is_checkmate() else 0.0)
            board.pop()
            if not (before >= args.fine and after <= args.lost):
                n_screen_drop += 1
                continue

            # Stage 1b: how FORGIVING is this position? Survey a SAMPLE of legal
            # moves. Without this the random control scores like the net (see
            # docstring). Surveying all ~35 legal moves cost ~2.1M nodes/seed and
            # made the tool unusable (1 keep in 90 min), so we estimate the
            # fraction from a sample instead — a coarse proportion is all the
            # threshold needs.
            #
            # The sample is drawn from a rng seeded by the POSITION, not by the
            # run: the `random` control arm consumes the main rng to pick its
            # move, so a shared stream would survey different moves in each arm
            # and make the two keep rates incomparable. Same position => same
            # surveyed moves in both arms, which is what the control requires.
            legal = list(board.legal_moves)
            survey_rng = random.Random(board.board_fen() + str(board.turn))
            survey = (
                legal if len(legal) <= args.safe_frac_sample
                else survey_rng.sample(legal, args.safe_frac_sample)
            )
            n_safe = 0
            for cand in survey:
                board.push(cand)
                if board.is_game_over():
                    cq = -1.0 if board.is_checkmate() else 0.0
                else:
                    cq = _q(eng.analyse(
                        board, chess.engine.Limit(nodes=args.safe_frac_nodes)), mover)
                board.pop()
                if cq >= args.fine:
                    n_safe += 1
            safe_frac = n_safe / len(survey) if survey else 0.0
            if safe_frac < args.min_safe_frac:
                n_knife_edge += 1
                continue

            # Stage 2: calibrated deep confirmation, survivors only.
            d_before = _q(eng.analyse(board, chess.engine.Limit(nodes=args.deep_nodes)), mover)
            board.push(mv)
            d_after = _q(eng.analyse(board, chess.engine.Limit(nodes=args.deep_nodes)), mover) \
                if not board.is_game_over() else (
                    -1.0 if board.is_checkmate() else 0.0)
            board.pop()

            rec = {
                "line": line, "move": mv.uci(), "mover": "W" if mover else "B",
                "screen_before": before, "screen_after": after,
                "deep_before": d_before, "deep_after": d_after,
                "gap": d_before - d_after, "control": args.control,
                "safe_frac": safe_frac, "n_legal": len(legal), "n_surveyed": len(survey),
                # Recorded so a later reader can PROVE the guard and the criterion
                # shared a depth, instead of inferring it from wall-clock. The
                # 30k-vs-4M mismatch that broke the control was invisible in the
                # output that existed at the time.
                "safe_frac_nodes": args.safe_frac_nodes,
                "screen_nodes": args.screen_nodes, "deep_nodes": args.deep_nodes,
            }
            keep = d_before >= args.fine and d_after <= args.lost
            rec["keep"] = keep
            audit.append(rec)
            if keep:
                kept.append(line)
                print(f"  KEEP {mv.uci()} before={d_before:+.3f} after={d_after:+.3f} "
                      f"gap={d_before - d_after:+.3f} safe_frac={safe_frac:.2f}", flush=True)

    finally:
        eng.quit()

    with open(args.out_list, "w", encoding="utf-8") as fh:
        fh.write(f"# net-side vetted ({args.control}); ckpt={args.checkpoint}; "
                 f"fine>={args.fine} lost<={args.lost} deep={args.deep_nodes}\n")
        for ln in kept:
            fh.write(ln + "\n")
    if args.out_jsonl:
        with open(args.out_jsonl, "w", encoding="utf-8") as fh:
            for r in audit:
                fh.write(json.dumps(r) + "\n")

    print(f"\n[netvet:{args.control}] seeds={len(lines)} parse_fail={n_parse} "
          f"no_legal={n_noleg} screen_dropped={n_screen_drop} "
          f"knife_edge_dropped={n_knife_edge} "
          f"deep_checked={len(audit)} KEPT={len(kept)} -> {args.out_list}")


if __name__ == "__main__":
    main()
