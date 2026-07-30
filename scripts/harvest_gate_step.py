"""Live harvest → deep-SF gate → stage step (collect flywheel's missing middle).

The live harvester (#124) appends value-blind severe seeds to
``data/harvest/blindspot_live.severe.p*.txt``. This script is the gate that
consumes those append-only files:

  1. Read only NEW lines (per-file byte offsets in a state file).
  2. Dedup against the live pool, holdout panels, and already-emitted /
     already-rejected placement keys.
  3. Deep-SF-vet survivors (expected score ≤ ``--vet-lost-below``).
  4. APPEND winners to a staging file. The gate itself never edits the live pool;
     monitor_fen.sh promotes the cumulative staging file through the safe feed step.

Promotion remains a separate validated operation (mirrors the retire-step split),
but is automatic at the monitor boundary. Fail-soft on the CLI entrypoint: never
raises into a monitor.

Summary line (one, stdout):
  harvest_gate: new_lines=N unique_new=U after_dedup=D vetted_kept=K
                vetted_rejected=R capped=C staged_total=T
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Placement-key identity for dedup — the same helper mine_blindspot_seeds uses
# (chess.Board(fen).epd()). Re-exported so callers/tests import one place.
from scripts.mine_blindspot_seeds import (
    _sf_expected_score,
    load_existing_keys,
    load_holdout_keys,
    position_key,
)

placement_key = position_key

SfScore = Callable[[str], float | None]  # fen -> expected score [-1,1] or None


@dataclass
class GateState:
    """Persisted gate state: offsets + processed placement keys.

    ``pending`` holds (key, raw_line) candidates that passed dedup but were
    not sent to SF because of ``--max-vet-per-run`` — so a hard cap never
    silently drops them forever after the byte offset advances past them.
    """

    offsets: dict[str, int] = field(default_factory=dict)
    emitted: set[str] = field(default_factory=set)
    rejected: set[str] = field(default_factory=set)
    pending: list[tuple[str, str]] = field(default_factory=list)  # (key, raw_line)


@dataclass(frozen=True)
class GateCounts:
    new_lines: int
    unique_new: int
    after_dedup: int
    vetted_kept: int
    vetted_rejected: int
    sf_failed: int  # SF returned no score (crash/timeout) — an ERROR, not a legit reject
    capped: int
    staged_total: int


def load_gate_state(raw: dict[str, Any] | None) -> GateState:
    """Deserialize a state-file dict (missing/empty → fresh state)."""
    if not raw:
        return GateState()
    offsets = {str(k): int(v) for k, v in dict(raw.get("offsets") or {}).items()}
    emitted = {str(k) for k in (raw.get("emitted") or [])}
    rejected = {str(k) for k in (raw.get("rejected") or [])}
    pending = [
        (str(item["key"]), str(item["line"]))
        for item in (raw.get("pending") or [])
        if isinstance(item, dict) and "key" in item and "line" in item
    ]
    return GateState(offsets=offsets, emitted=emitted, rejected=rejected, pending=pending)


def dump_gate_state(state: GateState) -> dict[str, Any]:
    """Serialize GateState for the state file (stable key order for diffs)."""
    return {
        "offsets": {k: int(state.offsets[k]) for k in sorted(state.offsets)},
        "emitted": sorted(state.emitted),
        "rejected": sorted(state.rejected),
        "pending": [{"key": k, "line": ln} for k, ln in state.pending],
    }


def read_new_lines(
    paths: list[str],
    offsets: dict[str, int],
) -> tuple[list[str], dict[str, int], int]:
    """Tail complete new lines from each path; return (lines, new_offsets, n_raw).

    Only complete newline-terminated lines are returned — a trailing partial
    line leaves the offset at the start of that partial so the next run can
    re-read it once finished. Truncated/rotated files (size < offset) reset
    to 0 and re-read from the start.
    """
    lines: list[str] = []
    new_offsets = dict(offsets)
    n_raw = 0
    for path in paths:
        key = str(path)
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        last = int(new_offsets.get(key, 0))
        if size < last:
            last = 0  # rotated / truncated
        if size == last:
            new_offsets[key] = last
            continue
        try:
            with open(path, "rb") as fh:
                fh.seek(last)
                chunk = fh.read(size - last)
        except OSError:
            continue
        # Keep only complete lines; leave a trailing partial for next run.
        complete, sep, _remainder = chunk.rpartition(b"\n")
        if not sep:
            # No newline in the new bytes — wait for a full line.
            new_offsets[key] = last
            continue
        consumed = last + len(complete) + 1  # +1 for the final \n
        new_offsets[key] = consumed
        for raw in complete.split(b"\n"):
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                n_raw += 1
                lines.append("")  # empty → treated as blank/skip later
                continue
            n_raw += 1
            lines.append(text)
    return lines, new_offsets, n_raw


# Vet-order rank: curriculum first, untagged next, selfplay last.
SRC_CURRICULUM, SRC_UNKNOWN, SRC_SELFPLAY = 0, 1, 2

_SP_TAG = re.compile(r"(?:^|\s)sp=([01])(?:\s|$)")


def source_rank(raw: str) -> int:
    """Vet-order rank from the line's ``sp=`` provenance tag.

    Measured 2026-07-30 on 11,718 unique placement keys: curriculum-sourced
    captures (net vs Stockfish) are KEPT by the deep-SF vet at 44.9%, selfplay
    ones at 3.7% — 12.2x, and curriculum supplies 91.7% of every seed the gate
    has ever emitted. In selfplay both sides share the blind spot, so a capture
    is usually a Type B mutual illusion over a pre-move position that was
    actually fine, which is precisely what this gate exists to reject.

    An ABSENT tag ranks in the MIDDLE, not as selfplay. Every line written
    before this field existed is untagged, so ranking untagged as selfplay
    would shove the entire existing backlog behind every new capture on the
    strength of a format change rather than evidence. Middle rank plus a stable
    sort leaves the backlog's existing newest-first order exactly as it was.
    """
    m = _SP_TAG.search(raw)
    if m is None:
        return SRC_UNKNOWN
    return SRC_SELFPLAY if m.group(1) == "1" else SRC_CURRICULUM


def parse_seed_line(raw: str) -> tuple[str, str] | None:
    """Parse a harvest/seed line → (placement_key, body) or None if unusable.

    Body is the seed grammar without the inline ``#`` comment. Malformed /
    unparseable / reject-predicate lines return None (caller skips, no crash).
    """
    text = raw.strip()
    if not text or text.startswith("#"):
        return None
    body = text.split("#", 1)[0].strip()
    if not body:
        return None
    # Lazy: keep module importable for pure unit tests that inject everything.
    from chess_anti_engine.selfplay.opening import _fen_reject_reason, seed_board_from_line

    if _fen_reject_reason(body) is not None:
        return None
    try:
        fen = seed_board_from_line(body).fen()
    except ValueError:
        return None
    return position_key(fen), body


def current_pool_path(yaml_path: str) -> str:
    """Resolve the live ``opening_fen_list_path`` from yaml (read-only)."""
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults, load_yaml_file

    return str(flatten_run_config_defaults(load_yaml_file(yaml_path))["opening_fen_list_path"])


def count_staged(out_path: str) -> int:
    """Count non-comment non-blank lines currently in the staging file."""
    p = Path(out_path)
    if not p.is_file():
        return 0
    n = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            n += 1
    return n


def stage_line(body: str, *, stamp: str, score: float) -> str:
    """Seed-grammar line with a deterministic staged provenance tag."""
    return f"{body}  # staged {stamp} deep_sq={score:.3f}"


def run_gate(
    *,
    new_lines: list[str],
    state: GateState,
    exclude_keys: set[str],
    sf_score: SfScore | None,
    vet_lost_below: float,
    max_vet_per_run: int,
    dry_run: bool,
    stamp: str,
    out_path: str,
    new_offsets: dict[str, int] | None = None,
    pending_cap: int = 5000,
) -> tuple[GateCounts, list[str], GateState]:
    """Core gate: dedup → cap → (optional) SF vet → stage.

    ``sf_score`` maps a terminal FEN to a deep-SF expected score in [-1, 1]
    (side-to-move POV). Injected for tests; production wires StockfishPool.
    When ``dry_run`` is set, no SF calls and no mutations to ``state`` /
    ``out_path`` are committed (a copy of counts is still returned).

    Returns ``(counts, staged_lines_this_run, updated_state)``.
    """
    # Work on a mutable copy so dry-run leaves the caller's state intact.
    work = GateState(
        offsets=dict(state.offsets if new_offsets is None else new_offsets),
        emitted=set(state.emitted),
        rejected=set(state.rejected),
        pending=list(state.pending),
    )

    # ── collect unique new placement keys from this run's lines ──────────
    unique_new_keys: list[str] = []  # insertion order
    unique_new_body: dict[str, str] = {}
    unique_new_rank: dict[str, int] = {}
    seen_new: set[str] = set()
    n_raw = len(new_lines)
    for raw in new_lines:
        parsed = parse_seed_line(raw)
        if parsed is None:
            continue
        key, body = parsed
        if key in seen_new:
            continue
        seen_new.add(key)
        unique_new_keys.append(key)
        unique_new_body[key] = body
        # Rank comes from the RAW line: parse_seed_line strips the comment the
        # sp= tag lives in, so reading it off `body` would silently rank every
        # capture UNKNOWN and turn this whole ordering into a no-op.
        unique_new_rank[key] = source_rank(raw)

    # NEWEST-FIRST: vet this run's fresh captures before the pending backlog,
    # each group most-recent-first. The harvester emits captures from the
    # CURRENT net; a captured hole that still exists is re-emitted every time
    # the net blunders it again, so freshness is a good proxy for relevance.
    # Stale pending (from superseded, weaker nets) is mostly already-fixed
    # false positives — the old backlog verified 0/387 genuine — so the former
    # "pending first (FIFO)" order spent the whole SF budget re-litigating a
    # dead net's mistakes and never reached the ~35%-real recent captures.
    # SOURCE-FIRST, then newest-first. Measured 2026-07-30 over 11,718 unique
    # placement keys: curriculum captures are kept by the vet at 44.9% vs
    # selfplay's 3.7% (12.2x), and curriculum supplied 91.7% of every seed ever
    # emitted. Spending a 30-vet budget in blended order buys 23.3% yield when
    # the same budget spent curriculum-first buys ~44.9% — 1.9x seeds per SF
    # node. Nothing is discarded: overflow is DEFERRED to `pending`, so a
    # deprioritised selfplay capture is vetted on a later run, not dropped.
    # The sort is STABLE, so within each source the newest-first order below is
    # preserved exactly.
    candidates: list[tuple[str, str]] = []
    ranks: list[int] = []
    cand_seen: set[str] = set()
    blocked = exclude_keys | work.emitted | work.rejected

    def _consider(key: str, body: str, rank: int) -> None:
        if key in cand_seen or key in blocked:
            return
        cand_seen.add(key)
        candidates.append((key, body))
        ranks.append(rank)

    for key in reversed(unique_new_keys):
        _consider(key, unique_new_body[key], unique_new_rank[key])
    for key, line in reversed(work.pending):
        # pending stores raw lines; re-parse body (pending keys already known)
        parsed = parse_seed_line(line)
        body = parsed[1] if parsed is not None else line.split("#", 1)[0].strip()
        _consider(key, body, source_rank(line))

    # Stable sort on source rank alone: within a rank the insertion order above
    # (fresh captures newest-first, then the pending backlog) is untouched.
    order = sorted(range(len(candidates)), key=lambda i: ranks[i])
    candidates = [candidates[i] for i in order]

    after_dedup = len(candidates)
    if max_vet_per_run < 0:
        raise ValueError("max_vet_per_run must be >= 0")
    to_vet = candidates[:max_vet_per_run]
    overflow = candidates[max_vet_per_run:]
    capped = len(overflow)

    staged_this: list[str] = []
    kept = 0
    rejected_n = 0
    sf_failed_n = 0
    requeue: list[tuple[str, str]] = []  # SF-failed candidates to retry next run

    if dry_run:
        # Count only — no SF, no writes, no state mutation.
        staged_total = count_staged(out_path)
        counts = GateCounts(
            new_lines=n_raw,
            unique_new=len(unique_new_keys),
            after_dedup=after_dedup,
            vetted_kept=0,
            vetted_rejected=0,
            sf_failed=0,
            capped=capped,
            staged_total=staged_total,
        )
        return counts, [], state

    if to_vet and sf_score is None:
        raise RuntimeError("sf_score is required when there are candidates to vet")

    # Lazy import of opening only on the real vet path (terminal FEN for SF).
    from chess_anti_engine.selfplay.opening import seed_board_from_line

    for key, body in to_vet:
        fen = seed_board_from_line(body).fen()
        score = sf_score(fen) if sf_score is not None else None
        if score is None:
            # SF crashed/timed out — a real error, NOT evidence the position is
            # fine. Do NOT mark rejected (that would permanently discard a
            # genuine candidate on a transient outage); re-queue for next run.
            requeue.append((key, f"{body}  # pending"))
            sf_failed_n += 1
        elif score <= vet_lost_below:
            staged_this.append(stage_line(body, stamp=stamp, score=score))
            work.emitted.add(key)
            kept += 1
        else:
            # SF says above threshold → legit reject; never re-vet.
            work.rejected.add(key)
            rejected_n += 1

    # Rebuild pending oldest-first (newest last), so reversed() re-vets the
    # freshest first next run. Overflow is this run's un-vetted tail; SF-failed
    # re-queues are recent, so they sit at the newest end for prompt retry.
    # Bound the queue to the newest `pending_cap`: the oldest tail is stale
    # superseded-net material that would otherwise accumulate unboundedly (the
    # 17k backlog that starved the gate). Dropping it is self-healing — a hole
    # that still exists is re-emitted by the harvester on the next blunder.
    older_first = [(k, f"{body}  # pending") for k, body in reversed(overflow)]
    combined = older_first + requeue
    if pending_cap > 0 and len(combined) > pending_cap:
        # Trim to the newest `pending_cap` — but floor the kept count at
        # len(requeue) so the SF-failed re-queues (newest end) always survive.
        # Requeue is the transient-outage retry set; the "never permanently
        # discard on a transient SF failure" guarantee must hold even if the
        # re-queue alone exceeds the cap (else offsets have advanced and the
        # candidate is lost for good).
        combined = combined[-max(pending_cap, len(requeue)):]
    work.pending = combined
    if new_offsets is not None:
        work.offsets = dict(new_offsets)

    if staged_this:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "a", encoding="utf-8") as fh:
            for ln in staged_this:
                fh.write(ln + "\n")

    staged_total = count_staged(out_path)
    counts = GateCounts(
        new_lines=n_raw,
        unique_new=len(unique_new_keys),
        after_dedup=after_dedup,
        vetted_kept=kept,
        vetted_rejected=rejected_n,
        sf_failed=sf_failed_n,
        capped=capped,
        staged_total=staged_total,
    )
    return counts, staged_this, work


def _make_sf_score(
    sf_path: str,
    *,
    sf_nodes: int,
    multipv: int,
    syzygy_path: str | None,
    nice: int = 15,
) -> tuple[SfScore, Callable[[], None]]:
    """StockfishPool-backed expected-score function + close callback."""
    from chess_anti_engine.stockfish.pool import StockfishPool
    from chess_anti_engine.stockfish.wdl import mate_to_effective_cp

    pool = StockfishPool(
        path=sf_path,
        nodes=sf_nodes,
        num_workers=1,
        multipv=multipv,
        syzygy_path=syzygy_path,
        nice=nice,
    )

    def sf_score(fen: str) -> float | None:
        try:
            res = pool.submit(fen, nodes=sf_nodes, fresh=True).result()
        except Exception:
            return None
        if res.cp is not None:
            return _sf_expected_score(int(res.cp))
        if res.mate is not None:
            return _sf_expected_score(round(mate_to_effective_cp(res.mate)))
        if res.wdl is not None:
            return float(res.wdl[0] - res.wdl[2])
        return None

    return sf_score, pool.close


def format_summary(counts: GateCounts, *, dry_run: bool = False) -> str:
    """One monitor-log line prefixed ``harvest_gate:``."""
    dry = "DRY:" if dry_run else ""
    return (
        f"harvest_gate: new_lines={counts.new_lines} unique_new={counts.unique_new} "
        f"after_dedup={counts.after_dedup} vetted_kept={dry}{counts.vetted_kept} "
        f"vetted_rejected={dry}{counts.vetted_rejected} sf_failed={dry}{counts.sf_failed} "
        f"capped={counts.capped} staged_total={counts.staged_total}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--harvest-glob",
        default="data/harvest/blindspot_live.severe.p*.txt",
        help="glob of severe harvest files (append-only per-worker)",
    )
    ap.add_argument(
        "--yaml",
        default="configs/pbt2_small.yaml",
        help="live yaml (read-only) — resolves opening_fen_list_path for dedup",
    )
    ap.add_argument(
        "--holdout",
        nargs="*",
        default=["data/blindspot_panel_v1.jsonl", "data/blindspot_panel_v2.jsonl"],
        help="panel jsonl / seed files whose positions to EXCLUDE",
    )
    ap.add_argument(
        "--state",
        default="scratchpad/live_read/harvest_gate_state.json",
        help="persisted offsets + emitted/rejected keys + pending queue",
    )
    ap.add_argument("--sf-path", default=None, help="Stockfish binary (required unless --dry-run)")
    ap.add_argument("--sf-nodes", type=int, default=2_000_000,
                    help="deep-SF node budget per vet. MUST EXCEED THE TRAINING LABEL "
                         "BUDGET (~698k live: PID base_nodes with sf_label_nodes_cap=0). "
                         "Was 300_000 until 2026-07-26 — i.e. 2.3x SHALLOWER than the "
                         "labels, so the gate staged positions as deep-LOST that the "
                         "training target then rated EQUAL. Measured consequence: of 26 "
                         "long-stuck seeds, sf_wdl at the seed position averaged 0.485 "
                         "and ZERO were labelled lost, versus 0.397 / 20% for seeds the "
                         "net actually learned. A seed vetted shallower than it is "
                         "labelled is unlearnable by construction.")
    ap.add_argument("--multipv", type=int, default=10, help="Stockfish MultiPV for vetting")
    ap.add_argument("--syzygy-path", default=None,
                    help="Stockfish SyzygyPath for <=6-man endgame rows")
    ap.add_argument("--vet-lost-below", type=float, default=-0.80,
                    help="keep only if deep-SF expected score ≤ this (truly lost)")
    ap.add_argument("--max-vet-per-run", type=int, default=30,
                    help="hard cap on SF vets per run (overflow stays pending)")
    ap.add_argument("--pending-cap", type=int, default=5000,
                    help="max pending queue depth; newest kept, stale tail expired (0=unbounded)")
    ap.add_argument("--out", default="data/harvest/staged_candidates.txt",
                    help="staging queue (APPEND survivors; not the live pool)")
    ap.add_argument("--stamp", default="",
                    help="UTC stamp string embedded in staged provenance (tests: fixed)")
    ap.add_argument("--dry-run", action="store_true",
                    help="counts only — no SF, no state/out writes")
    ap.add_argument("--nice", type=int, default=15, help="Stockfish process nice (0..19)")
    args = ap.parse_args(argv)

    try:
        harvest_paths = sorted({str(Path(p)) for p in glob.glob(args.harvest_glob)})
        state = GateState()
        if os.path.exists(args.state):
            with open(args.state, encoding="utf-8") as fh:
                state = load_gate_state(json.load(fh))

        # Exclude = live pool + holdout panels (read-only).
        exclude: set[str] = set()
        try:
            pool = current_pool_path(args.yaml)
            exclude |= load_existing_keys([pool])
        except Exception as e:
            print(f"[harvest_gate] warn: live pool load failed ({type(e).__name__}: {e})",
                  file=sys.stderr)
        holdout_paths = [p for p in args.holdout if p and os.path.exists(p)]
        if holdout_paths:
            exclude |= load_holdout_keys(holdout_paths)

        new_lines, new_offsets, _n = read_new_lines(harvest_paths, state.offsets)

        sf_score: SfScore | None = None
        sf_close: Callable[[], None] | None = None
        # Build SF only when we might actually vet (not dry-run / not idle).
        need_sf = not args.dry_run and bool(new_lines or state.pending)
        if need_sf:
            if not args.sf_path:
                print("harvest_gate: FAILED (sf-path required unless --dry-run or idle)")
                return 2
            sf_score, sf_close = _make_sf_score(
                args.sf_path,
                sf_nodes=args.sf_nodes,
                multipv=args.multipv,
                syzygy_path=args.syzygy_path,
                nice=args.nice,
            )

        try:
            counts, _staged, new_state = run_gate(
                new_lines=new_lines,
                state=state,
                exclude_keys=exclude,
                sf_score=sf_score,
                vet_lost_below=args.vet_lost_below,
                max_vet_per_run=args.max_vet_per_run,
                dry_run=args.dry_run,
                stamp=args.stamp,
                out_path=args.out,
                new_offsets=new_offsets,
                pending_cap=args.pending_cap,
            )
        finally:
            if sf_close is not None:
                sf_close()

        if not args.dry_run:
            state_path = Path(args.state)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(state_path, "w", encoding="utf-8") as fh:
                json.dump(dump_gate_state(new_state), fh, indent=2)
                fh.write("\n")

        print(format_summary(counts, dry_run=args.dry_run))
        return 0
    except Exception as e:  # side tool — must never break the monitor
        print(f"harvest_gate: FAILED ({type(e).__name__}: {e})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
