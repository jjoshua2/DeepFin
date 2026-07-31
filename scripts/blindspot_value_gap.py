#!/usr/bin/env python3
"""Type-A seed scorer — "does the net STILL misevaluate this position?".

Why this exists, and why `blindspot_netside_vet.py` cannot answer it
--------------------------------------------------------------------
The blind-spot pipeline produces TWO different kinds of seed, and they need
different instruments. Measured 2026-07-30 from
``scripts/harvest_gate_step.py``'s state (emitted 2,734 / rejected 8,963):

  Type A — what the gate KEEPS.  The net says the position is fine (>= net_ok)
           while deep SF says it is lost (<= --vet-lost-below, default -0.80).
           A VALUE error about the position itself.
  Type B — what the gate REJECTS. The position is genuinely fine (SF +1.000)
           and the net picks a losing MOVE. A policy/tactical error.

``blindspot_netside_vet.py`` asks the Type B question ("is the net's top move
worse than a random legal move here"). Pointing it at a Type A pool measures
nothing: the pool's positions are LOST, so its ``before >= --fine`` screen
rejects essentially all of them and the paired denominator collapses. That was
attempted on 2026-07-30 and abandoned mid-run for exactly this reason.

**Type B looks unpromising, but it is NOT settled — an earlier revision of this
docstring said "settled and NOT worth harvesting" and that was an overclaim.**
Stratifying the paired lift by how forgiving the position is showed no hidden
signal:

    ALL                     n=80  blunders=28  expected=25.60  lift=1.09x
    forgiving   >=0.50      n=49  blunders= 6  expected= 4.67  lift=1.29x
    very forgiving >=0.67   n=33  blunders= 1  expected= 1.08  lift=0.92x
    knife-edge   <0.50      n=31  blunders=22  expected=20.93  lift=1.05x

Read the POWER before reading the verdict. Only the ALL row is well powered:
sd ~ sqrt(25.60) = 5.06, so it excludes a lift above roughly 1.4x. The
very-forgiving row — the one that looks most damning at 0.92x — is **n=33 with
ONE observed event**; observing >=3 when expecting 1.08 has p~0.10, so it could
not detect a 3x lift. It is a weak null, not a refutation.

Second confound, and the more important one: **these seeds were harvested from
an OLDER net, and the score above is the CURRENT net's top move.** A null is
therefore consistent with "the current net has already learned them" — which is
what a working seeding loop should produce. The clean Type B experiment asks
whether the current net blunders on positions harvested from the CURRENT net,
and has not been run.

What genuinely favours the value channel is evidence INDEPENDENT of the above:
on 2,496 positions mined from real Cheese losses the net reads -13.7cp where SF
reads -300.9cp, and that was shown not to be a search problem — 200k nodes
changes nothing and the solution is already inside the root set in 98.8% of
cases at topk=16. When the right move is being considered and more search does
not help, the failure is in EVALUATION, not move generation.

Note also that Type B candidates are ~3.3x MORE NUMEROUS in the raw stream
(8,963 vs 2,734). Value errors are not more common; they are the ones the gate
keeps.

So the live pool is Type A, and this tool scores Type A on its own terms:

    net_q = the net's own wdl value at the seed FEN     (side-to-move POV)
    sf_q  = deep SF's value at the same FEN             (same POV)
    gap   = net_q - sf_q      > 0 means the net is OPTIMISTIC

A seed is LIVE (still worth feeding) when the net still says the position is
fine while SF says it is lost — ``net_q >= --live-net-ok`` AND
``sf_q <= --live-sf-lost``, defaulting to HarvestConfig's own net_ok/sf_lost so
a seed is re-tested against the rule that admitted it. It is LEARNED when the
net has come to agree with SF, and should be retired rather than doled.

``gap`` is reported for diagnostics but is deliberately NOT the criterion. The
net's value output is compressed toward zero — across a sanity slice
``|net_q| <= 0.456`` while ``|sf_q|`` reached 1.000 — so a raw-gap threshold
mostly measures SCALE, not blindness: any position with large ``|sf_q|``
produces a large opposite-signed gap automatically. This is the documented
trap (the value head is calibrated rather than broken; Brier/ECE are fooled by
exactly this). Judge sign and ranking, not magnitude.

⚠ HISTORY PLANES. Live-pool seeds are BARE FENs, so the board has no move
stack and 117/175 input planes are zero — see
[[frozen_rulers_score_fen_only_inputs]]. That is NOT corrected here on purpose:
production seeds selfplay from these same bare FENs, so this is what the net
actually sees at dole time. The measurement is faithful to deployment. Do not
"fix" it by synthesising history — that would measure a position production
never evaluates.

⚠ This scores the POOL, not the harvest stream. They are different files with
different grammars and on 2026-07-30 they had ZERO overlap; a verdict about one
says nothing about the other. Pass the path from the live yaml's
``opening_fen_list_path``, and print the overlap if you are unsure.
"""
from __future__ import annotations

import argparse
import json

import chess
import chess.engine
import numpy as np
import torch

from chess_anti_engine.encoding import model_encoding_kwargs
from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.mcts.puct import _value_scalar_from_wdl_logits
from chess_anti_engine.selfplay.opening import seed_board_from_line
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

_DEFAULT_SF = "/home/josh/projects/chess/e2e_server/publish/stockfish"


def _sf_q(info: chess.engine.InfoDict, pov: chess.Color) -> float:
    """SF win-minus-loss in [-1, 1] from ``pov``'s side."""
    score = info.get("score")
    if score is None:
        raise RuntimeError("stockfish returned no score for an analysed position")
    wdl = score.pov(pov).wdl()
    return (wdl.wins - wdl.losses) / 1000.0


def _net_q(
    board: chess.Board, ev: LocalModelEvaluator, enc_kwargs: dict, use_rel: bool,
) -> float:
    """The net's own wdl value, side-to-move POV, in [-1, 1].

    Uses the SAME head MCTS uses (`wdl`); `sf_eval` and `categorical` are
    auxiliary and would not describe what search believes.
    """
    cb = CBoard.from_board(board)
    x = encode_cboard(cb, **enc_kwargs)
    rel = np.stack([cb.compute_relations()]) if use_rel else None
    with torch.no_grad():
        _policy, wdl_logits = ev.evaluate_encoded(np.stack([x]), relations=rel)
    # `evaluate_encoded` returns out["wdl"] RAW — these are LOGITS, not
    # probabilities, and `_value_scalar_from_wdl_logits` is the exact function
    # the search value path applies to them. Reusing it rather than
    # reimplementing keeps this instrument on the PRODUCTION call path.
    #
    # ⚑ The first version of this function computed `v[0] - v[2]` on the raw
    # logits. It ran, produced plausible-looking per-seed numbers, and was only
    # caught because a sanity slice printed values OUTSIDE [-1, 1] (+1.316,
    # -1.291) — impossible for a win-minus-loss. Range-check any quantity with
    # known bounds before believing a single row of it.
    q = float(_value_scalar_from_wdl_logits(
        np.asarray(wdl_logits, dtype=np.float64)[0]))
    # The gate that would have caught the logits bug on row 1 instead of by eye.
    if not -1.0 <= q <= 1.0:
        raise RuntimeError(
            f"net_q={q} is outside [-1, 1]; a win-minus-loss cannot be. The wdl "
            "head's output contract has changed under this tool."
        )
    return q


def _seed_lines(path: str) -> list[str]:
    out: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            body = raw.split("#")[0].strip()
            if body:
                out.append(body)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True,
                    help="the LIVE pool (yaml opening_fen_list_path), not a harvest file")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stockfish", default=_DEFAULT_SF)
    ap.add_argument("--sf-nodes", type=int, default=2_000_000,
                    help="matches harvest_gate_step.py's budget, so LIVE/LEARNED here "
                         "is judged on the same ruler the gate admitted the seed with")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash-mb", type=int, default=512)
    ap.add_argument("--syzygy-path", default="data/syzygy_3-4-5-6")
    # LIVE is a SIGN DISAGREEMENT WITH MARGINS, deliberately not a raw-gap
    # threshold, and these defaults are HarvestConfig.net_ok / .sf_lost so a
    # seed is re-tested against the same rule that admitted it.
    #
    # ⚑ A raw `net_q - sf_q >= --min-gap` rule was tried first and is wrong.
    # The net's value output is COMPRESSED toward zero (|net_q| <= 0.456 across
    # a sanity slice where |sf_q| reached 1.000), so the gap is dominated by
    # scale, not by blindness: any position with large |sf_q| shows a large gap
    # of the opposite sign automatically. That is the documented trap — the
    # value head is calibrated rather than broken, and sharpness metrics
    # (Brier/ECE) are fooled by exactly this. Judge RANKING/sign, not magnitude.
    ap.add_argument("--live-net-ok", type=float, default=0.2,
                    help="LIVE requires net_q >= this (HarvestConfig.net_ok)")
    ap.add_argument("--live-sf-lost", type=float, default=-0.5,
                    help="LIVE requires sf_q <= this (HarvestConfig.sf_lost)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all seeds")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-jsonl", default="")
    args = ap.parse_args()

    lines = _seed_lines(args.seeds)
    if args.limit:
        lines = lines[: args.limit]

    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    enc_kwargs = model_encoding_kwargs(model)
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    ev = LocalModelEvaluator(model, device=args.device)

    eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    eng.configure({"Threads": args.threads, "Hash": args.hash_mb})
    try:
        eng.configure({"SyzygyPath": args.syzygy_path})
    except Exception:
        pass

    rows: list[dict] = []
    n_bad = 0
    try:
        for line in lines:
            try:
                board = seed_board_from_line(line)
            except Exception:
                n_bad += 1
                continue
            if board.is_game_over():
                n_bad += 1
                continue
            mover = board.turn
            nq = _net_q(board, ev, enc_kwargs, use_rel)
            sq = _sf_q(eng.analyse(board, chess.engine.Limit(nodes=args.sf_nodes)), mover)
            gap = nq - sq
            live = nq >= args.live_net_ok and sq <= args.live_sf_lost
            rows.append({
                "line": line, "mover": "W" if mover else "B",
                "net_q": nq, "sf_q": sq, "gap": gap,
                "live": live,
                "sf_nodes": args.sf_nodes,
                "live_net_ok": args.live_net_ok, "live_sf_lost": args.live_sf_lost,
            })
            print(f"  net={nq:+.3f} sf={sq:+.3f} gap={gap:+.3f} "
                  f"{'LIVE' if live else 'learned'}", flush=True)
    finally:
        eng.quit()

    if args.out_jsonl:
        with open(args.out_jsonl, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    if not rows:
        print(f"[valuegap] no scorable seeds (unparsable/terminal={n_bad})")
        return
    gaps = sorted(r["gap"] for r in rows)
    live = sum(1 for r in rows if r["live"])
    mid = gaps[len(gaps) // 2]
    print(f"\n[valuegap] seeds={len(rows)} unscorable={n_bad} "
          f"LIVE(net>={args.live_net_ok} and sf<={args.live_sf_lost})={live} "
          f"({live / len(rows):.1%}) "
          f"median_gap={mid:+.3f} "
          f"p10={gaps[len(gaps) // 10]:+.3f} p90={gaps[(9 * len(gaps)) // 10]:+.3f}")
    print("[valuegap] LIVE share is the number that decides whether this pool is "
          "worth doling. A pool the net has LEARNED teaches nothing no matter how "
          "efficiently it is harvested.")


if __name__ == "__main__":
    main()
