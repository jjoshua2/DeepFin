#!/usr/bin/env python3
"""§8's reference-arm harness: FastQ-4+ against qsearch-4 on the parity corpus.

Same population, same rows, each arm through its own provider. Reports mean |Δ|,
p95 |Δ|, identical fraction, the >50/100/250 buckets — and SIGN AGREEMENT, which
§8 names as primary because mean |Δ| alone is gameable by an arm that never
corrects anything: a FastQ that returned the static value everywhere would score
a small mean |Δ| against qsearch and be worthless.

⚑⚑ SIMILARITY TO QSEARCH IS NOT THE TARGET, AND A HIGH SCORE HERE PROVES NOTHING
ABOUT STRENGTH. §8 is explicit: the deciding readout for any production claim is
the downstream standardized primary against deep SF, not agreement with qsearch.
qsearch-4 is a cheap first-pass reference — it says "FastQ is resolving the same
tactics, not a different game" — and that is the whole of what it says.

The third arm (`nnue-qsearch-dag`) is reported alongside because it separates the
two independent changes: it is qsearch's move policy on FastQ's substrate, so the
gap qsearch → qsearch-dag is what the DAG bought and the gap qsearch-dag → fastq
is what the move policy and pruning bought. Reporting only the endpoints would
attribute all of it to whichever change was being argued for.

Usage:
    PYTHONPATH=. python3 scripts/fastq_reference_arm.py --pack <weights.pack>
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, TypeVar

import chess

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.nnue import _nnue_ext

#: ⚑ The corpus is imported, never re-listed. It is pinned in the parity test
#: because that test's bit-identity claim is only as strong as what the rows
#: provably span (promotions, en passant, quiet-evasion-only checks — all
#: asserted there). A second copy of 467 FENs in scripts/ would drift from it,
#: and then the two files would disagree about what "the corpus" means while both
#: kept passing.
from tests.test_qsearch_dag_parity import CORPUS

ARMS = ("nnue-qsearch", "nnue-qsearch-dag", "nnue-fastq")
REFERENCE = "nnue-qsearch"
SUBJECT = "nnue-fastq"


class ArmRun(NamedTuple):
    """One arm's sweep: per-row values and per-row cost, plus the live handle."""

    values: list[int]
    evals: list[int]
    wall_us: list[float]
    trips: list[int]
    see_prunes: list[int]
    delta_prunes: list[int]
    handle: object


#: FastQ counters recorded per row so the attribution in docs/fastq_search.md is
#: re-derivable from the dump instead of only from the prose that describes it.
_PER_ROW_COUNTERS = ("budget_trips", "see_prunes", "delta_prunes")


def _counters(arm: str, handle: object) -> dict[str, int]:
    """The counter block that actually moves for this arm.

    ⚑ arm_stats() AND fastq_stats() EACH REFUSE THE OTHER ARM'S HANDLE, and that
    refusal is why this function is short. The first version of this harness read
    arm_stats()["nnue_evals"] for every arm — which for FastQ was a permanent 0 —
    and printed `nnue-fastq mean 0.00 median 0 max 0` across all 467 rows,
    indistinguishable from a perfect result on a PR whose headline is an
    evaluation-count reduction. Both directions now raise rather than answer
    zeros, so this dispatch cannot silently pick the wrong block.
    """
    if arm == SUBJECT:
        return _nnue_ext.fastq_stats(handle)
    return _nnue_ext.arm_stats(handle)


def _per_call(arm: str, pack: Path, boards: Sequence[chess.Board]) -> ArmRun:
    """Evaluate row by row, recording each call's own cost.

    One handle for the whole sweep, so an arm that persists work across calls
    gets credit for it — that persistence is the point of the DAG substrate, and
    a fresh handle per row would measure a configuration nothing runs.

    ⚑ §7 ASKS FOR WALL p50/p99 AND IT IS MEASURED HERE, NOT IN THE C COUNTERS.
    Deliberate: a per-call clock_gettime in the hot path buys a number dominated
    by whatever else this machine is running (usually live training), and a
    p50/p99 needs per-call samples — a histogram or reservoir — inside a search
    whose whole point is to be cheap. The harness already loops per call, so it
    takes the same measurement for free and identically for every arm. Eval count
    stays the DECIDING instrument because it is deterministic; wall time is
    reported alongside it, never instead of it.
    """
    handle = _nnue_ext.arm_open(arm, str(pack))
    values: list[int] = []
    evals: list[int] = []
    wall_us: list[float] = []
    tracked: dict[str, list[int]] = {k: [] for k in _PER_ROW_COUNTERS}
    previous = _counters(arm, handle)
    for board in boards:
        cboard = CBoard.from_board(board)
        start = time.perf_counter()
        values.append(_nnue_ext.arm_handle_eval(handle, [cboard])[0])
        wall_us.append((time.perf_counter() - start) * 1e6)
        now = _counters(arm, handle)
        evals.append(now["nnue_evals"] - previous["nnue_evals"])
        for key in _PER_ROW_COUNTERS:
            tracked[key].append(now.get(key, 0) - previous.get(key, 0))
        previous = now
    if sum(evals) == 0:
        raise RuntimeError(f"{arm} reported zero NNUE evaluations; wrong counter")
    return ArmRun(
        values,
        evals,
        wall_us,
        tracked["budget_trips"],
        tracked["see_prunes"],
        tracked["delta_prunes"],
        handle,
    )


_Number = TypeVar("_Number", int, float)


def _quantile(sorted_values: Sequence[_Number], q: float) -> _Number:
    index = min(len(sorted_values) - 1, int(q * (len(sorted_values) - 1) + 0.5))
    return sorted_values[index]


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _report_values(subject: list[int], reference: list[int]) -> None:
    pairs = list(zip(subject, reference, strict=True))
    deltas = [abs(a - b) for a, b in pairs]
    ordered = sorted(deltas)
    n = len(deltas)

    agree = sum(_sign(a) == _sign(b) for a, b in pairs)
    both_nonzero = [(a, b) for a, b in pairs if _sign(a) != 0 and _sign(b) != 0]
    agree_nonzero = sum(_sign(a) == _sign(b) for a, b in both_nonzero)

    print(f"  rows                {n}")
    print(f"  SIGN AGREEMENT      {agree}/{n} = {agree / n:.4f}   <-- §8 primary")
    if both_nonzero:
        print(
            f"    excl. zero-valued {agree_nonzero}/{len(both_nonzero)} = "
            f"{agree_nonzero / len(both_nonzero):.4f}"
        )
    print(f"  identical           {ordered.count(0)}/{n} = {ordered.count(0) / n:.4f}")
    print(f"  mean |delta|        {statistics.fmean(deltas):.2f}")
    print(f"  median |delta|      {_quantile(ordered, 0.50)}")
    print(f"  p95 |delta|         {_quantile(ordered, 0.95)}")
    print(f"  max |delta|         {ordered[-1]}")
    for threshold in (50, 100, 250):
        over = sum(d > threshold for d in deltas)
        print(f"  |delta| > {threshold:<4}       {over}/{n} = {over / n:.4f}")

    # ⚑ A MATE SCORE IS NOT A LARGE CENTIPAWN NUMBER, AND AVERAGING IT AS ONE
    # MAKES THE MEAN REPORT THE MATE COUNT. Values above RESOLVER_EVAL_CLAMP are
    # mate-distance scores near ±100000, so a single row where one arm found a
    # mate and the other did not moves the mean by ~200 on a 467-row corpus —
    # swamping every genuine evaluation difference in the other 466. The two
    # populations answer different questions and are reported separately.
    mate = [
        (a, b)
        for a, b in pairs
        if max(abs(a), abs(b)) > _nnue_ext.RESOLVER_EVAL_CLAMP
    ]
    quiet_rows = [
        d
        for d, (a, b) in zip(deltas, pairs, strict=True)
        if max(abs(a), abs(b)) <= _nnue_ext.RESOLVER_EVAL_CLAMP
    ]
    mate_disagree = sum(_sign(a) != _sign(b) or abs(a - b) > 0 for a, b in mate)
    print(f"  mate-band rows      {len(mate)} ({mate_disagree} differing)")
    if quiet_rows:
        quiet_sorted = sorted(quiet_rows)
        print(
            f"  non-mate rows       {len(quiet_rows)}: mean |delta| "
            f"{statistics.fmean(quiet_rows):.2f}  p95 {_quantile(quiet_sorted, 0.95)}"
            f"  max {quiet_sorted[-1]}"
        )


def _report_evals(name: str, evals: list[int]) -> None:
    ordered = sorted(evals)
    n = len(ordered)
    in_band = sum(5 <= e <= 20 for e in evals)
    print(
        f"  {name:<18} mean {statistics.fmean(evals):7.2f}  median {_quantile(ordered, 0.50):5d}"
        f"  p90 {_quantile(ordered, 0.90):5d}  p99 {_quantile(ordered, 0.99):6d}"
        f"  max {ordered[-1]:6d}   <5: {sum(e < 5 for e in evals) / n:.3f}"
        f"  5-20: {in_band / n:.3f}  >20: {sum(e > 20 for e in evals) / n:.3f}"
    )


def _report_wall(name: str, wall_us: list[float]) -> None:
    ordered = sorted(wall_us)
    print(
        f"  {name:<18} p50 {_quantile(ordered, 0.50):9.1f}us  "
        f"p99 {_quantile(ordered, 0.99):9.1f}us  max {ordered[-1]:9.1f}us"
    )


def _write_dump(path: Path, results: dict[str, ArmRun]) -> None:
    """One JSON object per corpus row, so every claim in the summary is re-derivable.

    ⚑⚑ THE SUMMARY ALONE IS NOT EVIDENCE, WHICH IS WHY THIS EXISTS. Aggregates
    cannot be re-attributed after the fact: "the differing rows are the check
    policy, not the budget" is a claim about WHICH rows, and a printed mean can
    neither confirm nor refute it. Anyone can now re-derive the split — or find a
    different one — without rerunning the sweep or trusting the prose.
    """
    subject = results[SUBJECT]
    reference = results[REFERENCE]
    dag_values = results["nnue-qsearch-dag"].values
    clamp = _nnue_ext.RESOLVER_EVAL_CLAMP

    with path.open("w", encoding="utf-8") as out:
        for i, board in enumerate(CORPUS):
            fastq, qsearch = subject.values[i], reference.values[i]
            legal = list(board.legal_moves)
            row = {
                "fen": board.fen(),
                "fastq": fastq,
                "qsearch": qsearch,
                "qsearch_dag": dag_values[i],
                "delta": fastq - qsearch,
                "sign_fastq": _sign(fastq),
                "sign_qsearch": _sign(qsearch),
                "sign_agree": _sign(fastq) == _sign(qsearch),
                "mate_band": max(abs(fastq), abs(qsearch)) > clamp,
                "fastq_evals": subject.evals[i],
                "qsearch_evals": reference.evals[i],
                "fastq_wall_us": round(subject.wall_us[i], 2),
                "root_in_check": board.is_check(),
                "root_quiet_check_available": any(
                    not board.is_capture(m) and not m.promotion and board.gives_check(m)
                    for m in legal
                ),
                "cap_tripped": subject.trips[i] > 0,
                "see_prunes": subject.see_prunes[i],
                "delta_prunes": subject.delta_prunes[i],
            }
            out.write(json.dumps(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True, help="NNUE weights pack.")
    parser.add_argument(
        "--max-qply", type=int, default=_nnue_ext.FASTQ_MAX_QPLY,
        help="FastQ qply; the default matches QSEARCH_MAX_PLY so the arms are paired.",
    )
    parser.add_argument(
        "--node-cap", type=int, default=_nnue_ext.FASTQ_NODE_CAP,
        help="FastQ's §3.4 budget tripwire; 0 disables it.",
    )
    parser.add_argument(
        "--dump", type=Path, default=None,
        help="Write the per-row JSONL the summary is derived from.",
    )
    args = parser.parse_args()

    if args.max_qply != _nnue_ext.QSEARCH_MAX_PLY:
        print(
            f"⚑ qply {args.max_qply} != qsearch's compiled "
            f"{_nnue_ext.QSEARCH_MAX_PLY}; the arms are NOT depth-paired."
        )

    _nnue_ext.fastq_set_config(
        args.max_qply,
        args.node_cap,
        _nnue_ext.FASTQ_DELTA_MARGIN,
        _nnue_ext.FASTQ_RECAPTURE_EXEMPT,
    )

    results = {arm: _per_call(arm, args.pack, CORPUS) for arm in ARMS}
    _nnue_ext.fastq_set_config()

    print(f"pack: {args.pack.name}   rows: {len(CORPUS)}")
    print(
        f"qply: fastq {args.max_qply} / qsearch {_nnue_ext.QSEARCH_MAX_PLY}"
        f"   fastq node_cap {args.node_cap}\n"
    )

    print("NNUE evaluations per call")
    for arm in ARMS:
        _report_evals(arm, results[arm].evals)

    print("\nWall per call (§7; measured here, not in the C counters — see _per_call)")
    for arm in ARMS:
        _report_wall(arm, results[arm].wall_us)

    print(f"\n{SUBJECT} vs {REFERENCE}")
    _report_values(results[SUBJECT].values, results[REFERENCE].values)

    print(f"\nnnue-qsearch-dag vs {REFERENCE}  (substrate only; must be identical)")
    _report_values(results["nnue-qsearch-dag"].values, results[REFERENCE].values)

    fastq_counters = _nnue_ext.fastq_stats(results[SUBJECT].handle)
    print("\nFastQ counters (§7)")
    for key in sorted(fastq_counters):
        print(f"  {key:<24} {fastq_counters[key]}")
    identity = (
        fastq_counters["nnue_evals"] + fastq_counters["nodes_created_in_check"]
        == fastq_counters["nodes_created"]
    )
    print(f"  evaluate-once identity   {'HOLDS' if identity else '*** BROKEN ***'}")

    if args.dump:
        _write_dump(args.dump, results)
        print(f"\nper-row dump: {args.dump}  ({len(CORPUS)} rows)")
    return 0 if identity else 1


if __name__ == "__main__":
    raise SystemExit(main())
