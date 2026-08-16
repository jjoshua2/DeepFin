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

A REAL defect was found while diagnosing this, and it is fixed, but **it was
not the cause** — recorded that way deliberately, because the tempting write-up
("found the bug, fixed it, moved on") would have been wrong. The defect: the
guard ran at a different search depth from the criterion it guards. Keep
required ``after <= --lost`` at 4,000,000 nodes while ``safe_frac`` counted a
move "safe" at ``cq >= --fine`` on **30,000** — 1/133rd the search. Hence
``--safe-frac-nodes`` now defaults to ``--screen-nodes``: **a threshold
compared against another threshold must be measured with the same instrument at
the same setting**; the guard's cost is not a free parameter, it is part of the
guard's meaning.

But re-running at 300k returned **identical** ``safe_frac`` values (c8c5 0.92,
g1g5 0.67 — the same numbers to 2dp). On these lopsided ``+1.000`` positions
the ``--fine`` bar of +0.20 is so low that a move keeping the mover clearly
winning at 30k also does at 300k; the dead band between −0.50 and +0.20 almost
never binds. So the depth fix is correct and was measuring the wrong thing by
luck, and it changed no verdict.

**The actual defect is statistical power, and it is a defect in the CONTROL'S
DESIGN, not the criterion.** Comparing two marginal keep rates over 80 seeds
cannot resolve anything: separating 7.5% from 5.0% needs ~1500 seeds per arm at
~8M nodes each. The arm was never capable of the job it was given.

So the primary evidence is now a PAIRED statistic that costs nothing extra,
because the survey already measures it. For each surveyed position
``blunder_frac`` is the fraction of sampled legal moves that would themselves
have dropped it to ``<= --lost`` — i.e. the random mover's blunder probability
ON THAT POSITION. Summed over the surveyed set that gives
``expected_if_random``, tested against this arm's observed blunder count on the
SAME positions at the SAME depth. The printed ``lift`` is the ratio.

    lift ≈ 1.0  ⇒ indistinguishable from random play: the seeds are "positions
                  where most moves lose", NOT blind spots. Do not admit them.
    lift >> 1.0 ⇒ the net errs where random play would not. Admit.

⚠⚠⚠ AND THE FIRST VERSION OF THAT STATISTIC WAS ALSO WRONG — third catch, same
day, this one found by reading the number rather than by a control. It printed:

    PAIRED surveyed=28 blunders=28 expected_if_random=18.35 lift=1.53x

**28 of 28 is 100%, and it was 100% BY CONSTRUCTION.** The survey ran only on
positions that had passed the screen, and the screen required
``after <= --lost`` — i.e. it required that the arm had ALREADY blundered. So
the numerator was fixed at 1.0 by the selection rule, and it was being compared
against the ``blunder_frac`` of exactly the positions selected for having a
high one. The 1.53x meant nothing whatsoever.

The fix is the ORDER of the stages, which is why the code below is split into
1a/1b and the ``arm_blundered`` flag is carried rather than used immediately:

    1a  ``before >= --fine``     position-level, arm-independent  → may gate
    1b  ``after  <= --lost``     the arm's outcome                → RECORDED ONLY
    1c  survey → blunder_frac    arm-independent                  → may gate
        ... accumulate the paired counters over EVERYTHING reaching here ...
    2   arm_blundered / knife-edge / deep confirmation            → candidate only

**Never gate the denominator of a control on the event the control measures.**
The general form of this mistake is conditioning on the outcome, and note that
it survived both a negative control and a lint gate — it produced a plausible
non-trivial number (1.53x, neither 1.0 nor absurd) and only the coincidence of
``surveyed`` and ``blunders`` being the same integer gave it away.

``--control random`` is retained as a cheap end-to-end sanity check — it does
exercise the whole path — but **it must not be used to decide whether the tool
works.** Note also for the record that ``blunder_frac`` is counted separately
from ``safe_frac`` rather than as ``1 - safe_frac``: "safe" (>= --fine) and
"blunder" (<= --lost) are not complements, and folding the dead band into the
baseline would have inflated it and understated the net's lift.

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
from chess_anti_engine.utils.engine_discovery import default_stockfish

# ⚑ DISCOVERED, not merely repo-relative. `e2e_server/publish/` is UNTRACKED
# runtime output, so it exists only in the checkout that published it — a
# checkout-relative default resolves to nothing in the `git worktree` CLAUDE.md
# mandates for branch work, which is where these tools are run. The shared
# lookup falls back through $CAE_STOCKFISH, this checkout, the MAIN checkout and
# PATH. (Codex inline review, #441.)
_DEFAULT_SF = default_stockfish()


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
    n_parse = n_noleg = n_screen_drop = n_knife_edge = n_no_blunder = 0
    n_surveyed_pos = n_screen_blunders = 0
    exp_random_blunders = 0.0

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

            # Stage 1a: is the position PLAYABLE at all? Position-level and
            # arm-independent, so it may gate the paired denominator.
            before = _q(eng.analyse(board, chess.engine.Limit(nodes=args.screen_nodes)), mover)
            if before < args.fine:
                n_screen_drop += 1
                continue

            # Stage 1b: did THIS arm's move throw it away? Recorded, NOT used to
            # gate the survey below.
            #
            # ⚑ THE ORDER HERE IS THE WHOLE POINT. This test used to be part of
            # the screen, so the survey only ever ran on positions where the arm
            # had ALREADY blundered — which made the paired baseline conditional
            # on its own outcome. The 2026-07-30 run printed
            # `surveyed=28 blunders=28`: 100% by construction, compared against
            # the blunder_frac of exactly those positions selected for having a
            # high one. It reported lift=1.53x, which meant nothing. Never gate
            # the denominator of a control on the event being measured.
            board.push(mv)
            after = _q(eng.analyse(board, chess.engine.Limit(nodes=args.screen_nodes)), mover) \
                if not board.is_game_over() else (
                    -1.0 if board.is_checkmate() else 0.0)
            board.pop()
            arm_blundered = after <= args.lost

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
            # Counted separately from n_safe because "safe" (>= --fine) and
            # "blunder" (<= --lost) are NOT complements — a move landing in the
            # dead band between them is neither. Using 1 - safe_frac as the
            # random blunder rate would inflate the baseline with dead-band
            # moves and understate the net's lift.
            n_blunder = 0
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
                if cq <= args.lost:
                    n_blunder += 1
            safe_frac = n_safe / len(survey) if survey else 0.0
            blunder_frac = n_blunder / len(survey) if survey else 0.0

            # PAIRED BASELINE — the reason every surveyed position is recorded,
            # not just the ones that reach the deep check.
            #
            # The survey IS the per-position random-move control: `blunder_frac`
            # is the measured fraction of sampled legal moves that a random
            # mover would have thrown the position away with. So the expected
            # number of blunders a random player commits over the surveyed set
            # is sum(blunder_frac), and the net's observed blunder count is
            # tested against it on the SAME positions at the SAME depth.
            #
            # This replaces `--control random` as the primary evidence. That arm
            # compares two MARGINAL rates and is hopelessly underpowered: the
            # 2026-07-30 run read net 6/80 vs random 4/80, and separating 7.5%
            # from 5.0% needs ~1500 seeds per arm at ~8M nodes each. The paired
            # form uses each position as its own control and needs no second
            # arm. Keep `--control random` as a cheap end-to-end sanity check;
            # do not use it to decide whether the tool works.
            n_surveyed_pos += 1
            exp_random_blunders += blunder_frac
            if arm_blundered:
                n_screen_blunders += 1
            audit.append({
                "line": line, "move": mv.uci(), "mover": "W" if mover else "B",
                "stage": "surveyed", "control": args.control,
                "screen_before": before, "screen_after": after,
                "screen_blunder": arm_blundered,
                "safe_frac": safe_frac, "blunder_frac": blunder_frac,
                "n_legal": len(legal),
                "n_surveyed": len(survey),
                "safe_frac_nodes": args.safe_frac_nodes,
                "screen_nodes": args.screen_nodes,
            })

            # Only now may the arm's own outcome gate anything: everything the
            # paired statistic needs has already been accumulated above.
            if not arm_blundered:
                n_no_blunder += 1
                continue
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
                "stage": "deep",
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

    n_deep = sum(1 for r in audit if r.get("stage") == "deep")
    print(f"\n[netvet:{args.control}] seeds={len(lines)} parse_fail={n_parse} "
          f"no_legal={n_noleg} unplayable_dropped={n_screen_drop} "
          f"no_blunder={n_no_blunder} knife_edge_dropped={n_knife_edge} "
          f"deep_checked={n_deep} KEPT={len(kept)} -> {args.out_list}")

    # PAIRED CONTROL — the number that actually decides whether this tool works.
    # Over the surveyed positions, a uniformly random mover would blunder
    # sum(1 - safe_frac) times; this arm blundered n_screen_blunders times.
    # Both are measured on the SAME positions at the SAME depth, so the ratio is
    # interpretable at n=80 where two marginal keep rates are not.
    if n_surveyed_pos:
        lift = (n_screen_blunders / exp_random_blunders
                if exp_random_blunders > 0 else float("inf"))
        print(f"[netvet:{args.control}] PAIRED surveyed={n_surveyed_pos} "
              f"blunders={n_screen_blunders} "
              f"expected_if_random={exp_random_blunders:.2f} lift={lift:.2f}x")
        # A GATE THAT CAN ACTUALLY FAIL. `blunders == surveyed` is the exact
        # fingerprint of the 2026-07-30 conditioning bug: it means every
        # position in the denominator was one where the arm blundered, so the
        # numerator is pinned at 1.0 by the selection rule and the lift is an
        # artifact. That run printed a plausible 1.53x and was believed. Refuse
        # to hand back a number instead of letting the next reader trust it.
        if n_screen_blunders == n_surveyed_pos:
            raise SystemExit(
                f"[netvet:{args.control}] REFUSING TO REPORT: blunders "
                f"({n_screen_blunders}) == surveyed ({n_surveyed_pos}). The "
                "paired denominator is conditioned on the arm having blundered, "
                "so `lift` is meaningless. The survey must run on every "
                "PLAYABLE position (stage 1a), not only on positions that "
                "failed stage 1b — see the module docstring."
            )
        print(f"[netvet:{args.control}] lift ~1.0 means this arm is "
              f"INDISTINGUISHABLE from random play on these positions; the "
              f"seeds are then 'positions where most moves lose', not blind "
              f"spots, and must NOT be admitted.")


if __name__ == "__main__":
    main()
