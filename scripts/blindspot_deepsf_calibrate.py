"""Deep-SF calibration of the continuation classifier.

The continuation classifier (scripts/blindspot_continuation.py) decides "who was
right" from the played-out game — cheap, but confounded: a REFUTED seed (sf_q
recovered downstream) is ambiguous between SF being wrong at the seed (a real
false positive) and the handicapped opponent blundering the win away (a real
blind spot the continuation was fooled on). The only unconfounded arbiter is a
static, very deep, UNHANDICAPPED Stockfish eval of the SEED position itself —
deeper than we could ever afford in-loop. We run it on a sample and cross-tab it
against the continuation buckets:

  * CONFIRMED_LOST vs deep-SF: fraction deep still calls LOST  -> validates the
    continuation's cheap "real blind spot" calls.
  * REFUTED vs deep-SF: deep FINE = TRUE false positive; deep LOST = opponent
    rescue (continuation over-refuted).

deep_sq = wdl[W]-wdl[L] from side-to-move (= net) POV, directly comparable to the
seed's captured sq. Verdicts: LOST (<= --deep-lost), FINE (>= --deep-fine), MID.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from chess_anti_engine.selfplay.opening import seed_board_from_line
from chess_anti_engine.stockfish.uci import StockfishUCI

_TAG = re.compile(r"bucket=(\w+).*?sq=(-?[0-9.]+).*?outcome=(\w+).*?game=(\d+)")
# Repo-relative, derived from this file: the published engine lives in the
# checkout, so an absolute path only ever named one machine's copy of it.
_DEFAULT_SF = str(Path(__file__).resolve().parents[1] / "e2e_server" / "publish" / "stockfish")


def ensure_parent(path: str) -> None:
    """Create the output file's parent dir UP FRONT — else a multi-hour deep-SF
    run raises FileNotFoundError at the final write and discards results (Codex #125)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def resolve_syzygy(path: str) -> str | None:
    """Fail-fast on a missing TB dir: a non-empty syzygy path whose first
    component is absent would SILENTLY run TB-less and give unreliable endgame
    verdicts (Codex #125). Pass --syzygy-path '' to run without TB deliberately."""
    if not path:
        return None
    first = path.split(":")[0].split(";")[0]
    if not os.path.isdir(first):
        raise SystemExit(f"syzygy path not found: {first!r} — would run deep SF "
                         f"TB-less (unreliable endgames). Pass --syzygy-path '' to opt out.")
    return path


def parse_all(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.rstrip("\n")
            if "#" not in raw:
                continue
            body, _, comment = raw.partition("#")
            m = _TAG.search(comment)
            if m is None:
                continue
            rows.append({"line": body.strip(), "bucket": m[1],
                         "sq": float(m[2]), "outcome": m[3], "game": int(m[4])})
    return rows


def select(rows: list[dict], control_n: int) -> list[dict]:
    """All REFUTED + INCONCLUSIVE, plus an evenly-strided CONFIRMED control."""
    ref = [r for r in rows if r["bucket"] == "REFUTED"]
    inc = [r for r in rows if r["bucket"] == "INCONCLUSIVE"]
    conf = [r for r in rows if r["bucket"] == "CONFIRMED_LOST"]
    if control_n > 0 and len(conf) > control_n:
        stride = len(conf) / control_n
        conf = [conf[int(i * stride)] for i in range(control_n)]
    return ref + inc + conf


def deep_verdict(sq: float, lost: float, fine: float) -> str:
    if sq <= lost:
        return "LOST"
    if sq >= fine:
        return "FINE"
    return "MID"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", default="scratchpad/harvest_fp/all_severe.txt")
    ap.add_argument("--stockfish", default=_DEFAULT_SF)
    ap.add_argument("--nodes", type=int, default=8_000_000,
                    help="deep-SF node budget per position (>> the ~700k in-loop label)")
    ap.add_argument("--hash-mb", type=int, default=2048)
    ap.add_argument("--control-n", type=int, default=24,
                    help="CONFIRMED_LOST control-sample size (0 = all)")
    ap.add_argument("--deep-lost", type=float, default=-0.5)
    ap.add_argument("--deep-fine", type=float, default=-0.2)
    ap.add_argument("--out", default="scratchpad/harvest_fp/deepsf_calib.jsonl")
    ap.add_argument("--nice", type=int, default=15)
    ap.add_argument("--syzygy-path", default="data/syzygy_3-4-5-6",
                    help="syzygy TB dir(s) — exact WDL for endgames the in-loop label misjudges")
    args = ap.parse_args()
    ensure_parent(args.out)
    syzygy = resolve_syzygy(args.syzygy_path)  # fail-fast before the long search

    rows = parse_all(args.all)
    sel = select(rows, args.control_n)
    print(f"[deepsf] {len(rows)} located seeds; rechecking {len(sel)} "
          f"(all REFUTED+INCONCLUSIVE + {sum(r['bucket']=='CONFIRMED_LOST' for r in sel)} CONFIRMED control)")
    print(f"[deepsf] nodes={args.nodes:,} hash={args.hash_mb}MB syzygy={syzygy} "
          f"stockfish={args.stockfish}")

    def make_engine() -> StockfishUCI:
        return StockfishUCI(args.stockfish, nodes=int(args.nodes),
                            hash_mb=int(args.hash_mb), nice=int(args.nice),
                            syzygy_path=syzygy)

    eng = make_engine()
    results: list[dict] = []
    t0 = time.time()
    try:
        for i, r in enumerate(sel):
            try:
                board = seed_board_from_line(r["line"])
                fen = board.fen()
                eng.new_game()  # cold TT per seed (Codex #125)
                res = eng.search(fen, nodes=int(args.nodes))
                wdl = res.wdl
                if wdl is None:
                    r2 = {**r, "fen": fen, "deep_sq": None, "deep": "UNKNOWN"}
                else:
                    # res.wdl is already normalized to [0,1] (sums to 1) -> the
                    # W-L difference is the [-1,1] q, directly comparable to sq.
                    dsq = float(wdl[0] - wdl[2])
                    r2 = {**r, "fen": fen, "deep_sq": round(dsq, 3),
                          "deep": deep_verdict(dsq, args.deep_lost, args.deep_fine)}
            except Exception as e:  # one bad seed must not kill the run
                # A raised/timed-out search can leave SF calculating -> desyncs the
                # next seed; recreate the engine (Codex #125).
                r2 = {**r, "fen": "", "deep_sq": None, "deep": f"ERR:{type(e).__name__}"}
                try:
                    eng.close()
                except Exception:
                    pass
                eng = make_engine()
            results.append(r2)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(sel)}  ({time.time()-t0:.0f}s)")
    finally:
        eng.close()

    with open(args.out, "w", encoding="utf-8") as fh:
        for r2 in results:
            fh.write(json.dumps(r2) + "\n")

    # Cross-tab: continuation bucket x deep-SF verdict
    buckets = ["CONFIRMED_LOST", "REFUTED", "INCONCLUSIVE"]
    deeps = ["LOST", "MID", "FINE"]
    print("\n=== continuation bucket  x  deep-SF verdict ===")
    print(f"  {'bucket':16s} {'n':>3}  " + "  ".join(f"{d:>5}" for d in deeps))
    for b in buckets:
        sub = [r for r in results if r["bucket"] == b]
        if not sub:
            continue
        counts = {d: sum(1 for r in sub if r["deep"] == d) for d in deeps}
        print(f"  {b:16s} {len(sub):>3}  " + "  ".join(f"{counts[d]:>5}" for d in deeps))

    # Headline splits
    conf = [r for r in results if r["bucket"] == "CONFIRMED_LOST"]
    ref = [r for r in results if r["bucket"] == "REFUTED"]
    if conf:
        agree = sum(1 for r in conf if r["deep"] == "LOST")
        print(f"\n  CONFIRMED_LOST validated by deep SF : {agree}/{len(conf)} "
              f"= {100*agree/len(conf):.0f}% still LOST")
    if ref:
        true_fp = sum(1 for r in ref if r["deep"] == "FINE")
        rescue = sum(1 for r in ref if r["deep"] == "LOST")
        print(f"  REFUTED true false-positive (deep FINE): {true_fp}/{len(ref)} "
              f"= {100*true_fp/len(ref):.0f}%")
        print(f"  REFUTED opponent-rescue (deep LOST)    : {rescue}/{len(ref)} "
              f"= {100*rescue/len(ref):.0f}%  (continuation over-refuted)")
    graded = [r for r in results if r["deep"] in ("LOST", "MID", "FINE")]
    if graded:
        fp = sum(1 for r in graded if r["deep"] == "FINE")
        print(f"\n  deep-SF false-positive rate over rechecked seeds: "
              f"{fp}/{len(graded)} = {100*fp/len(graded):.0f}%")


if __name__ == "__main__":
    main()
