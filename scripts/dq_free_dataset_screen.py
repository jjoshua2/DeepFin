"""ΔQ phase 1, step 0: how much of the ΔQ dataset do we ALREADY OWN?

The ΔQ programme buys a targeted Stockfish comparison between the move the NET
wants and the move SEARCH wants. Buying is expensive, so the first question is
not "how big is ΔQ" but:

    On rows where the net and Stockfish disagree, is the NET'S preferred move
    already inside Stockfish's surfaced set?

If it is, we already hold SF evaluations for BOTH candidates and ΔQ is FREE —
it is a re-read of shards we have. If it is not, the net's move is unscored and
only a targeted `searchmoves` query can price it, which is exactly what the
plumbing buys. Either answer is decision-relevant, which is why this runs
BEFORE any loss design and before the query path is even merged.

⚑ POPULATION CAVEAT, stated up front. This runs on the wide MultiPV-40 era
shards with ranks > k HIDDEN to simulate production's narrow width. That is a
deliberate choice, not a shortcut: on real MultiPV-6 shards the hidden moves
have no scores at all, so "what would the query have bought" is unanswerable.
Here we can measure both the RATE (how often a query is needed) and its VALUE
(what the answer would have been), because the wide era already paid for it.
The rate transfers structurally; the era is not production's.

⚑ This is an OBSERVATIONAL screen. It changes nothing and trains nothing.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.replay.shard import load_shard_arrays
from scripts.tail_censor_screen import (
    SF_OWN_REGRET_CAP_CP,
    Row,
    attach_prior,
    check_invariants,
    collect,
    select_shards,
)


def classify(rows: list[Row]) -> dict[str, Any]:
    """Split disagreement rows by whether the net's move is already scored."""
    n_total = 0
    n_agree = 0
    free: list[float] = []      # ΔQ available from stored labels
    buyable: list[float] = []   # ΔQ obtainable only by a targeted query
    unscorable = 0              # SF never scored it even at width 40
    free_conf: list[float] = []
    buy_conf: list[float] = []

    for row in rows:
        if row.prior is None:
            continue
        n_total += 1
        j = int(np.argmax(row.prior))
        a = int(row.legal[j])            # the move the NET wants
        b = int(row.surfaced[0])         # the move STOCKFISH wants (rank 1)
        conf = float(row.prior[j])

        if a == b:
            n_agree += 1
            continue

        # regret is keyed by global move index and is 0 for SF's best, so the
        # ΔQ between the two candidates IS the net-move's regret.
        if a in row.surfaced:
            free.append(row.regret[a])
            free_conf.append(conf)
        elif a in row.regret:
            buyable.append(row.regret[a])
            buy_conf.append(conf)
        else:
            unscorable += 1

    n_disagree = len(free) + len(buyable) + unscorable
    return {
        "rows": n_total,
        "agree": n_agree,
        "disagree": n_disagree,
        "free": len(free),
        "buyable": len(buyable),
        "unscorable": unscorable,
        "free_regret": free,
        "buyable_regret": buyable,
        "free_conf": free_conf,
        "buy_conf": buy_conf,
    }


def recover_search_moves(
    shards: list[Path], rows: list[Row], planes: list[np.ndarray],
) -> tuple[list[int | None], dict[str, int]]:
    """The move SEARCH actually played, per row, via the child position.

    ⚑ This is the pairing the pre-registration actually specifies. Comparing the
    prior against SF's OWN best move is a DIFFERENT question: SF's best is rank 1
    by construction, so "both candidates surfaced" degenerates into "is the net's
    move surfaced" and the free fraction comes out too high.

    ⚑ `argmax(policy_target)` is NOT the played move under Gumbel at final
    temperature 0 — see `cast_probe.recover_played_move`, which fails CLOSED
    (returns None) when the child does not identify a unique move.
    """
    from scripts.cast_probe import recover_played_move

    want = {(r.game_id, r.ply + 1) for r in rows}
    children: dict[tuple[int, int], np.ndarray] = {}
    encodings: dict[str, str] = {}
    for shard in shards:
        try:
            arrs, _ = load_shard_arrays(shard)
        except (OSError, ValueError, KeyError):
            continue
        encodings.setdefault(
            "hist", str(np.asarray(arrs["_input_history_encoding"]).item()))
        encodings.setdefault(
            "pol", str(np.asarray(arrs["_policy_encoding"]).item()))
        gid = np.asarray(arrs["game_id"])
        ply = np.asarray(arrs["ply_index"])
        x = np.asarray(arrs["x"])
        for i in range(gid.shape[0]):
            key = (int(gid[i]), int(ply[i]))
            if key in want:
                children[key] = x[i]

    stats = {"no_child": 0, "ambiguous": 0, "recovered": 0}
    out: list[int | None] = []
    for row, parent_x in zip(rows, planes):
        child = children.get((row.game_id, row.ply + 1))
        if child is None:
            stats["no_child"] += 1
            out.append(None)
            continue
        mv = recover_played_move(
            parent_x, child,
            input_history_encoding=encodings["hist"],
            policy_encoding=encodings["pol"],
        )
        if mv is None:
            stats["ambiguous"] += 1
        else:
            stats["recovered"] += 1
        out.append(mv)
    return out, stats


def classify_pairs(
    rows: list[Row], search_moves: list[int | None],
) -> dict[str, int]:
    """The pre-registered 4-way split on (prior move, SEARCH move)."""
    out = {
        "usable": 0, "same_move": 0, "both_surfaced": 0,
        "one_outside": 0, "both_outside": 0,
    }
    for row, a_m in zip(rows, search_moves):
        if row.prior is None or a_m is None:
            continue
        out["usable"] += 1
        a_p = int(row.legal[int(np.argmax(row.prior))])
        if a_p == a_m:
            out["same_move"] += 1
            continue
        n_out = int(a_p not in row.surfaced) + int(a_m not in row.surfaced)
        if n_out == 0:
            out["both_surfaced"] += 1
        elif n_out == 1:
            out["one_outside"] += 1
        else:
            out["both_outside"] += 1
    return out


def _q(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean_cp": float(arr.mean() * SF_OWN_REGRET_CAP_CP),
        "median_cp": float(np.median(arr) * SF_OWN_REGRET_CAP_CP),
        "p90_cp": float(np.percentile(arr, 90) * SF_OWN_REGRET_CAP_CP),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay-dir", type=Path, required=True, action="append")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="the net whose PRIOR defines 'the move the net wants'")
    ap.add_argument("--k", type=int, default=6, help="simulated production MultiPV")
    ap.add_argument("--max-shards", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    shards: list[Path] = []
    for d in args.replay_dir:
        shards.extend(select_shards(Path(d), args.max_shards))
    scan: dict[str, Any] = {
        "shard_names": [str(p) for p in shards], "rows_scanned": 0,
        "rows_selfplay": 0, "skipped_not_selfplay": 0,
        "desync_checked": 0, "desync_orphaned": 0, "desync_rows_rejected": 0,
        "coverage_sum": 0.0, "coverage_n": 0, "unscored_mass_sum": 0.0,
        "surfaced_not_legal": 0, "skipped_shards": [], "skipped_shards_omitted": 0,
    }
    rows, planes = collect(shards, scan, args.k)
    if not rows:
        raise SystemExit("no rows with a parent MultiPV block wider than --k")
    check_invariants(scan, len(rows))
    device = attach_prior(rows, planes, args.checkpoint, args.batch)

    res = classify(rows)
    n_d = res["disagree"]
    print(f"shards {len(shards)}   analysed {res['rows']} rows   prior on {device}")
    print(f"  simulated production width k={args.k}\n")

    print("NET vs STOCKFISH top-1")
    print(f"  agree            {res['agree']:6d}   {res['agree']/max(res['rows'],1):.3f}")
    print(f"  DISAGREE         {n_d:6d}   {n_d/max(res['rows'],1):.3f}"
          "   <- the ΔQ candidate pool\n")

    if n_d:
        print("OF THE DISAGREEMENTS — is the net's move already scored?")
        print(f"  ALREADY SURFACED (ΔQ is FREE)   {res['free']:6d}   "
              f"{res['free']/n_d:.3f}")
        print(f"  hidden below k (NEEDS A QUERY)  {res['buyable']:6d}   "
              f"{res['buyable']/n_d:.3f}")
        print(f"  never scored even at width 40   {res['unscorable']:6d}   "
              f"{res['unscorable']/n_d:.3f}")
        print("    ⚑ the last row is the honest unknown: a targeted query would\n"
              "      price it, but nothing in this era's data can say what it costs.\n")

        print("ΔQ MAGNITUDE (net-move regret vs SF's best, cp)")
        print(f"  free rows    {_q(res['free_regret'])}")
        print(f"  buyable rows {_q(res['buyable_regret'])}")
        print("⚑⚑ DO NOT READ THE TWO ROWS ABOVE AS 'QUERIES BUY THE BIG ΔQ'.\n"
              "   The split IS the comparison: 'buyable' means SF ranked the net's\n"
              "   move 7th or worse, so its regret is >= this row's r_k BY\n"
              "   CONSTRUCTION, while every free row's is <= r_k. The ordering is\n"
              "   forced by the definition and carries no information. Only the\n"
              "   ABSOLUTE scale of the buyable row is a reading.\n")

        fc, bc = res["free_conf"], res["buy_conf"]
        if fc and bc:
            print("PRIOR CONFIDENCE on the net's move (a pre-query diagnostic)")
            print(f"  free    mean {np.mean(fc):.3f}   median {np.median(fc):.3f}")
            print(f"  buyable mean {np.mean(bc):.3f}   median {np.median(bc):.3f}")
            print("\nP(NEEDS A QUERY | prior confidence) — the query policy's input.\n"
                  "  Unlike the magnitude split above, this is NOT forced: nothing\n"
                  "  ties the net's confidence to SF's ranking of its move.")
            allc = np.concatenate([np.asarray(fc), np.asarray(bc)])
            need = np.concatenate([np.zeros(len(fc)), np.ones(len(bc))])
            edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            print(f"    {'confidence':<14}{'n':>7}{'P(query)':>11}")
            for lo, hi in itertools.pairwise(edges):
                sel = (allc >= lo) & (allc < hi if hi < 1.0 else allc <= 1.0)
                if int(sel.sum()) == 0:
                    continue
                print(f"    [{lo:.1f}, {hi:.1f})    {int(sel.sum()):>7}"
                      f"{float(need[sel].mean()):>11.3f}")

    print("\n" + "=" * 68)
    print("THE PRE-REGISTERED PAIRING — prior move vs SEARCH move, SF adjudicates")
    print("=" * 68)
    search_moves, rec = recover_search_moves(shards, rows, planes)
    print(f"  played-move recovery: {rec['recovered']} ok, "
          f"{rec['ambiguous']} ambiguous, {rec['no_child']} no stored child")
    pairs = classify_pairs(rows, search_moves)
    u = max(pairs["usable"], 1)
    dis = pairs["both_surfaced"] + pairs["one_outside"] + pairs["both_outside"]
    print(f"  usable rows {pairs['usable']}")
    print(f"    prior == search (nothing to adjudicate)  {pairs['same_move']:6d}"
          f"   {pairs['same_move']/u:.3f}")
    print(f"    DISAGREE                                 {dis:6d}   {dis/u:.3f}")
    if dis:
        print(f"      both already in SF6 (ΔQ FREE)          {pairs['both_surfaced']:6d}"
              f"   {pairs['both_surfaced']/dis:.3f}")
        print(f"      exactly one outside (buy it: S×T)      {pairs['one_outside']:6d}"
              f"   {pairs['one_outside']/dis:.3f}")
        print(f"      both outside (buy both: T×T)           {pairs['both_outside']:6d}"
              f"   {pairs['both_outside']/dis:.3f}")
        res["pairs"] = pairs

    if args.json_out:
        out = {k: v for k, v in res.items() if not isinstance(v, list)}
        out["free_regret_summary"] = _q(res["free_regret"])
        out["buyable_regret_summary"] = _q(res["buyable_regret"])
        out["k"] = args.k
        out["checkpoint"] = args.checkpoint
        out["shards"] = scan["shard_names"]
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"\nbanked -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
