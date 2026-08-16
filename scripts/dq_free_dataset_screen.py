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

⚑⚑ THIS IS A CHEAP AVAILABILITY *PROXY* SCREEN, NOT A ΔQ DATASET MEASUREMENT.
Three separate gaps sit between its number and the live question, and none of
them is closed by more rows:

1. **NOT SAME-MODEL.** The live question pairs `a_P = argmax π_θ(s)` against
   `a_M = MCTS_θ(s)` for ONE θ. Here `a_P` comes from re-running a chosen
   checkpoint over historical positions while `a_M` is the move the ORIGINAL
   net played when the game was generated. Different networks. The replay
   schema does not persist the generating prior — `_NetRecord.policy_probs`
   never reaches a shard, only the improved `policy_target` does — so the
   same-model pairing is NOT recoverable offline at all. It needs a
   prospective run.
2. **SIMULATED WIDTH IS NOT REAL WIDTH.** Ranks > k are hidden from MultiPV-40
   labels. PR #428 measured that changing MultiPV width at a fixed node budget
   changes the search itself (median depth 12 → 9 at width 6 → 64), so a
   truncated MPV40 block is NOT what a real MPV6 search would have produced.
   Do not claim the rate "transfers structurally".
3. **SELECTION.** ~75% of rows drop out because their child ply is not stored.
   The screen now reports pre-recovery observables for kept vs dropped rows;
   read that table before trusting the headline.

What it IS good for, and this is genuinely useful: establishing that a large
fraction of the candidate pairs are ALREADY COVERED by banked labels, so
"check what we hold before querying" is worth building. Treat the percentage as
an order of magnitude, not a calibration.

⚑ This is an OBSERVATIONAL screen. It changes nothing and trains nothing.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from dataclasses import dataclass
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


def coverage_of_prior_move(rows: list[Row]) -> dict[str, Any]:
    """PRIOR vs SF-BEST coverage. ⚑ A DIFFERENT POPULATION from the ΔQ pool.

    Deliberately named and typed apart from `PairObservation`, and it emits NO
    confidence curve: SF's best is rank 1 by construction, so "both surfaced"
    degenerates here into "is the prior's move surfaced". Binning this one and
    labelling it "P(needs a query)" is the exact bug this split prevents.
    """
    n_total = 0
    n_agree = 0
    free: list[float] = []      # ΔQ available from stored labels
    buyable: list[float] = []   # ΔQ obtainable only by a targeted query
    unscorable = 0              # SF never scored it even at width 40
    for row in rows:
        if row.prior is None:
            continue
        n_total += 1
        j = int(np.argmax(row.prior))
        a = int(row.legal[j])            # the move the NET wants
        b = int(row.surfaced[0])         # the move STOCKFISH wants (rank 1)

        if a == b:
            n_agree += 1
            continue

        # regret is keyed by global move index and is 0 for SF's best, so the
        # ΔQ between the two candidates IS the net-move's regret.
        if a in row.surfaced:
            free.append(row.regret[a])
        elif a in row.regret:
            buyable.append(row.regret[a])
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


@dataclass(frozen=True)
class PairObservation:
    """ONE ΔQ candidate pair: the prior's move vs the move SEARCH played.

    ⚑ THIS TYPE EXISTS BECAUSE THE POPULATION WAS THE BUG. An earlier revision
    computed the headline split from the (prior, search) pairing while a
    downstream confidence table was still binning a DIFFERENT function's
    (prior, SF-best) rows. Both printed as "P(needs a query | confidence)" and
    were visually indistinguishable. Making the population a value that every
    diagnostic must be handed means a consumer cannot silently stay on the old
    framing -- it would have to be wired to a different type by name.

    The prior-vs-SF-best analysis still exists, deliberately under a different
    name and return type (`coverage_of_prior_move`), and it does NOT emit a
    confidence curve.
    """

    game_id: int
    ply: int
    prior_move: int
    search_move: int
    prior_conf: float
    prior_entropy: float
    prior_in_sf6: bool
    search_in_sf6: bool

    @property
    def is_disagreement(self) -> bool:
        return self.prior_move != self.search_move

    @property
    def n_outside(self) -> int:
        return int(not self.prior_in_sf6) + int(not self.search_in_sf6)

    @property
    def pair_class(self) -> str:
        if not self.is_disagreement:
            return "same_move"
        return ("both_surfaced", "one_outside", "both_outside")[self.n_outside]


def build_pair_observations(
    rows: list[Row], search_moves: list[int | None],
) -> list[PairObservation]:
    """The ΔQ population. Every downstream diagnostic derives from this list."""
    obs: list[PairObservation] = []
    for row, a_m in zip(rows, search_moves):
        if row.prior is None or a_m is None:
            continue
        p = row.prior
        j = int(np.argmax(p))
        a_p = int(row.legal[j])
        obs.append(PairObservation(
            game_id=row.game_id, ply=row.ply,
            prior_move=a_p, search_move=int(a_m),
            prior_conf=float(p[j]),
            prior_entropy=float(-(p * np.log(np.clip(p, 1e-12, None))).sum()),
            prior_in_sf6=a_p in row.surfaced,
            search_in_sf6=int(a_m) in row.surfaced,
        ))
    return obs


def headline_split(obs: list[PairObservation]) -> dict[str, int]:
    counts = {
        "usable": len(obs), "same_move": 0, "both_surfaced": 0,
        "one_outside": 0, "both_outside": 0,
    }
    for o in obs:
        counts[o.pair_class] += 1
    return counts


def confidence_curves(
    obs: list[PairObservation],
) -> list[tuple[str, int, float, float, float]]:
    """P(any query), P(S×T), P(T×T) vs prior confidence, over DISAGREEMENTS."""
    pool = [o for o in obs if o.is_disagreement]
    if not pool:
        return []
    conf = np.asarray([o.prior_conf for o in pool], dtype=float)
    any_ = np.asarray([float(o.n_outside > 0) for o in pool])
    st = np.asarray([float(o.n_outside == 1) for o in pool])
    tt = np.asarray([float(o.n_outside == 2) for o in pool])
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    out: list[tuple[str, int, float, float, float]] = []
    for lo, hi in itertools.pairwise(edges):
        sel = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= 1.0)
        n = int(sel.sum())
        if n == 0:
            continue
        out.append((f"[{lo:.1f}, {hi:.1f})", n, float(any_[sel].mean()),
                    float(st[sel].mean()), float(tt[sel].mean())))
    return out


def attrition_bias(
    rows: list[Row], search_moves: list[int | None],
) -> list[tuple[str, float, float]]:
    """Compare recovered vs non-recovered rows on PRE-recovery observables.

    ⚑ 75% of rows drop out because their child ply is not stored. That is far
    too much attrition to assume is random, and recoverability is not something
    the analysis controls. These observables are all computable BEFORE recovery,
    so a difference here is a selection effect, not an outcome.
    """
    keep: dict[str, list[float]] = {}
    drop: dict[str, list[float]] = {}
    for row, a_m in zip(rows, search_moves):
        if row.prior is None:
            continue
        bucket = keep if a_m is not None else drop
        p = row.prior
        j = int(np.argmax(p))
        ent = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
        bucket.setdefault("prior_top1", []).append(float(p[j]))
        bucket.setdefault("prior_entropy", []).append(ent)
        bucket.setdefault("legal_moves", []).append(float(len(row.legal)))
        bucket.setdefault("top1_in_sf6", []).append(
            float(int(row.legal[j]) in row.surfaced))
        bucket.setdefault("r_k", []).append(float(row.r_k))
        bucket.setdefault("ply", []).append(float(row.ply))
    return [
        (name, float(np.mean(keep[name])), float(np.mean(drop[name])))
        for name in sorted(keep)
    ]


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

    res = coverage_of_prior_move(rows)
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

        print("⚑ The prior-vs-SF framing above does NOT get a confidence curve.\n"
              "  A query is needed when EITHER candidate is outside SF6, so the\n"
              "  curve belongs to the prior-vs-SEARCH pool below.")

    print("\n" + "=" * 68)
    print("THE PRE-REGISTERED PAIRING — prior move vs SEARCH move, SF adjudicates")
    print("=" * 68)
    search_moves, rec = recover_search_moves(shards, rows, planes)
    print(f"  played-move recovery: {rec['recovered']} ok, "
          f"{rec['ambiguous']} ambiguous, {rec['no_child']} no stored child")
    obs = build_pair_observations(rows, search_moves)
    pairs = headline_split(obs)
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
        print("    ⚑ 'buy both' is ONE root-restricted search, not two:\n"
              "      `searchmoves a_prior a_search` at MultiPV 2 returns both.\n"
              "      T×T is informationally harder, NOT 2x the query budget.\n")

        curves = confidence_curves(obs)
        if curves:
            # ⚑ The denominator is printed IN the title. The previous revision's
            # table summed to a different n than the headline split directly
            # above it, and nobody noticed because neither carried its own
            # population size.
            print(f"P(NEEDS A QUERY | prior confidence),  n={dis} "
                  "ACTUAL prior/search disagreements\n"
                  "  ⚑ Derived from PairObservation, the same objects the split\n"
                  "  above is counted from. An earlier revision binned the\n"
                  "  prior-vs-SF-best rows instead — a different question on a\n"
                  "  different population, visually indistinguishable here.")
            print(f"    {'confidence':<14}{'n':>6}{'P(any)':>9}{'P(S×T)':>9}{'P(T×T)':>9}")
            for label, n, p_any, p_st, p_tt in curves:
                print(f"    {label:<14}{n:>6}{p_any:>9.3f}{p_st:>9.3f}{p_tt:>9.3f}")
        res["pairs"] = {k: v for k, v in pairs.items() if not isinstance(v, list)}
        res["confidence_curves"] = curves

    bias = attrition_bias(rows, search_moves)
    if bias:
        print("\n⚑ STORED-CHILD SELECTION BIAS — recovered vs dropped rows.\n"
              "  All observables are computable BEFORE recovery, so a gap here is\n"
              "  a selection effect. 75% attrition is too large to assume random.")
        print(f"    {'observable':<16}{'recovered':>11}{'dropped':>10}{'ratio':>9}")
        for name, kept, dropped in bias:
            ratio = kept / dropped if dropped else float("nan")
            print(f"    {name:<16}{kept:>11.3f}{dropped:>10.3f}{ratio:>9.3f}")
        res["attrition_bias"] = [
            {"observable": n, "recovered": k, "dropped": d} for n, k, d in bias
        ]

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
