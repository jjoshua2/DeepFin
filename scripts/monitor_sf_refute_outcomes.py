#!/usr/bin/env python3
"""Track fenlist selfplay vs SF-refute channel outcomes (seed-STM POV).

These are descriptive channel outcomes, not a strength or promotion verdict.
For confirmed-lost fenlist seeds, increasing STM wins/draws can mean selfplay
is failing to preserve the known loss. Backed-up, holdable seeds have different
semantics and are not pooled into either channel here. Interpret changes using
the seed provenance and experiment's deciding measure.

Data source: one trial's result.json (JSONL), field ``outcome_stats``
(pipe-separated k=v). Keys from finalize.py:
  selfplay_fenlist_stm_{w,d,l}
  selfplay_fenlist_sf_refute_stm_{w,d,l}
  (+ games/draws counts). Missing or incomplete outcome triples have unknown
  rates; n counts observed outcomes, while games is the reported total.
  The automatically selected newest trial is pinned for
  this invocation; restart or pass --trial to select a different trial.

Usage:
  PYTHONPATH=. python3 scripts/monitor_sf_refute_outcomes.py
  PYTHONPATH=. python3 scripts/monitor_sf_refute_outcomes.py --last 30
  PYTHONPATH=. python3 scripts/monitor_sf_refute_outcomes.py --watch --every 120
  PYTHONPATH=. python3 scripts/monitor_sf_refute_outcomes.py --csv scratchpad/live_read/sf_refute_outcomes.csv
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import sys
import time
from pathlib import Path


def _newest_trial(work_dir: Path) -> Path | None:
    tune = work_dir / "tune"
    if not tune.is_dir():
        return None
    trials = sorted(
        (p for p in tune.glob("train_trial_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for t in trials:
        if (t / "result.json").is_file():
            return t
    return None


def _parse_outcome_stats(raw: str | dict | None) -> dict[str, int]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        out: dict[str, int] = {}
        for k, v in raw.items():
            try:
                out[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
        return out
    out = {}
    for part in str(raw).split("|"):
        if "=" not in part:
            continue
        k, _, v = part.partition("=")
        try:
            out[k.strip()] = int(v)
        except ValueError:
            continue
    return out


def _stm_rates(stats: dict[str, int], source: str) -> dict[str, float | int | None]:
    """source e.g. 'fenlist' or 'fenlist_sf_refute' → rates from seed STM POV."""
    prefix = f"selfplay_{source}_stm_"
    games = int(stats.get(f"selfplay_{source}_games", 0))
    if not any(prefix + outcome in stats for outcome in ("w", "d", "l")):
        return {
            "games": games, "w": None, "d": None, "l": None,
            "wr": None, "dr": None, "lr": None, "n": 0,
        }
    w = int(stats.get(prefix + "w", 0))
    d = int(stats.get(prefix + "d", 0))
    l = int(stats.get(prefix + "l", 0))
    n = w + d + l
    # Zero-valued outcome keys are omitted by the producer. Only observed STM
    # outcomes determine the rate denominator; a game count is not an outcome.
    if n <= 0 or n < games:
        return {
            "games": games, "w": w, "d": d, "l": l,
            "wr": None, "dr": None, "lr": None, "n": n,
        }
    return {
        "games": games if games else n,
        "w": w, "d": d, "l": l, "n": n,
        "wr": w / n, "dr": d / n, "lr": l / n,
    }


def _iter_rows(result_path: Path, *, last: int) -> list[dict]:
    with result_path.open(encoding="utf-8") as fh:
        lines = deque((ln for ln in fh if ln.strip()), maxlen=last if last > 0 else None)
    rows: list[dict] = []
    malformed = 0
    for ln in lines:
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(d, dict):
            malformed += 1
            continue
        stats = _parse_outcome_stats(d.get("outcome_stats"))
        # Skip iters with no fenlist signal (outcome_stats sometimes empty on partials).
        if not any(k.startswith("selfplay_fenlist") for k in stats):
            continue
        pure = _stm_rates(stats, "fenlist")
        refute = _stm_rates(stats, "fenlist_sf_refute")
        gap_lr = None
        if pure["lr"] is not None and refute["lr"] is not None:
            gap_lr = float(refute["lr"]) - float(pure["lr"])
        it = d.get("training_iteration")
        if it is None:
            it = d.get("iter")
        rows.append({
            "trial": str(result_path.parent.resolve()),
            "iter": it,
            "checkpoint": d.get("checkpoint_id") or d.get("training_iteration"),
            "pure_n": pure["n"],
            "pure_games": pure["games"],
            "pure_w": pure["w"],
            "pure_d": pure["d"],
            "pure_l": pure["l"],
            "pure_lr": pure["lr"],
            "pure_wr": pure["wr"],
            "refute_n": refute["n"],
            "refute_games": refute["games"],
            "refute_w": refute["w"],
            "refute_d": refute["d"],
            "refute_l": refute["l"],
            "refute_lr": refute["lr"],
            "refute_wr": refute["wr"],
            "gap_lr": gap_lr,  # refute_loss − pure_loss on observed channel outcomes
            "opening_fenlist_games": stats.get("opening_fenlist_games", 0),
            "opening_fenlist_sf_refute_games": stats.get("opening_fenlist_sf_refute_games", 0),
        })
    if malformed:
        print(f"[monitor] ignored {malformed} malformed/partial result row(s) in {result_path}",
              file=sys.stderr)
    return rows


def _fmt_rate(x: float | None) -> str:
    return "  n/a" if x is None else f"{100.0 * x:5.1f}%"


def _print_table(rows: list[dict], *, trial: Path) -> None:
    print(f"trial: {trial.resolve()}")
    print(
        f"{'iter':>6}  {'pure_n':>6}  {'pure_L%':>7}  {'pure_W%':>7}  "
        f"{'ref_n':>5}  {'ref_L%':>7}  {'ref_W%':>7}  {'gap_L':>7}  "
        f"{'open_f':>6}  {'open_r':>6}"
    )
    for r in rows:
        print(
            f"{str(r['iter'])[:6]:>6}  {r['pure_n']:6d}  {_fmt_rate(r['pure_lr']):>7}  "
            f"{_fmt_rate(r['pure_wr']):>7}  {r['refute_n']:5d}  {_fmt_rate(r['refute_lr']):>7}  "
            f"{_fmt_rate(r['refute_wr']):>7}  {_fmt_rate(r['gap_lr']):>7}  "
            f"{int(r['opening_fenlist_games']):6d}  {int(r['opening_fenlist_sf_refute_games']):6d}"
        )
    if not rows:
        print("(no fenlist outcome rows yet — need training with fenlist dole + channel live)")
        return
    # Rolling summary on last up-to-10 rows with both arms.
    both = [r for r in rows if r["pure_lr"] is not None and r["refute_lr"] is not None][-10:]
    if both:
        def _pool(key_w: str, key_d: str, key_l: str) -> tuple[int, int, int]:
            return (
                sum(int(r[key_w]) for r in both),
                sum(int(r[key_d]) for r in both),
                sum(int(r[key_l]) for r in both),
            )
        pw, pd, pl = _pool("pure_w", "pure_d", "pure_l")
        rw, rd, rl = _pool("refute_w", "refute_d", "refute_l")
        pn, rn = pw + pd + pl, rw + rd + rl
        print(f"--- last {len(both)} iters with both arms ---")
        if pn:
            print(
                f"  pure selfplay fenlist:   n={pn}  L={100*pl/pn:.1f}%  "
                f"D={100*pd/pn:.1f}%  W={100*pw/pn:.1f}%"
            )
        if rn:
            print(
                f"  live SF-refute channel:  n={rn}  L={100*rl/rn:.1f}%  "
                f"D={100*rd/rn:.1f}%  W={100*rw/rn:.1f}%"
            )
        if pn and rn:
            gap = (rl / rn) - (pl / pn)
            print(f"  gap_L (refute − pure) = {100*gap:+.1f} pp (descriptive; not a strength verdict)")


def _append_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trial", "iter", "pure_n", "pure_games", "pure_w", "pure_d", "pure_l", "pure_lr", "pure_wr",
        "refute_n", "refute_games", "refute_w", "refute_d", "refute_l", "refute_lr", "refute_wr",
        "gap_lr", "opening_fenlist_games", "opening_fenlist_sf_refute_games",
    ]
    existing: set[tuple[str, str]] = set()
    if path.is_file() and path.stat().st_size:
        with open(path, encoding="utf-8") as fh:
            if not fh.read().endswith("\n"):
                raise ValueError(f"{path}: unterminated CSV record; repair it or use a new CSV path")
            fh.seek(0)
            reader = csv.DictReader(fh, strict=True)
            if reader.fieldnames != fields:
                raise ValueError(f"{path}: CSV header does not match trial-qualified records; use a new CSV path")
            for row in reader:
                if None in row or any(value is None for value in row.values()) or not row["trial"] or not row["iter"]:
                    raise ValueError(f"{path}: incomplete CSV record; repair it or use a new CSV path")
                existing.add((str(row["trial"]), str(row["iter"])))
    new_rows = []
    for row in rows:
        if not row.get("trial") or row.get("iter") is None:
            raise ValueError("CSV rows require both trial and iteration identity")
        key = (str(row["trial"]), str(row["iter"]))
        if key not in existing:
            new_rows.append(row)
            existing.add(key)
    if not new_rows:
        return
    write_header = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in new_rows:
            w.writerow(r)
    print(f"[monitor] appended {len(new_rows)} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", type=Path, default=Path("runs/pbt2_small"))
    ap.add_argument("--trial", type=Path, default=None, help="trial dir (default: newest, pinned at startup)")
    ap.add_argument("--last", type=int, default=40, help="scan last N result lines")
    ap.add_argument("--csv", type=Path, default=None,
                    help="append new iters to this CSV (e.g. scratchpad/live_read/sf_refute_outcomes.csv)")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=int, default=180, help="watch sleep seconds")
    args = ap.parse_args()

    trial = args.trial or _newest_trial(args.work_dir)
    if trial is None:
        print("no trial with result.json under", args.work_dir / "tune", file=sys.stderr)
        sys.exit(1)
    trial = trial.resolve()

    def once() -> int:
        result = trial / "result.json"
        rows = _iter_rows(result, last=args.last)
        _print_table(rows, trial=trial)
        if args.csv is not None:
            _append_csv(args.csv, rows)
        return 0

    if not args.watch:
        sys.exit(once())
    while True:
        print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        once()
        time.sleep(max(30, int(args.every)))


if __name__ == "__main__":
    main()
