"""Fast selfplay-throughput readout from the broker's 10s stat lines.

Parses ``[broker] N batches in Ts | avg P pos/batch (...), S/16 slots/batch |
R pos/s | ...`` lines and reports loaded-window percentiles, so a throughput
change can be judged in ~2-3 minutes of wall time instead of games/h over
multiple iterations (game length drifts with the net; pos/s does not).

  PYTHONPATH=. python3 scripts/broker_pos_rate.py                  # last 30 min
  PYTHONPATH=. python3 scripts/broker_pos_rate.py --minutes 5
  PYTHONPATH=. python3 scripts/broker_pos_rate.py --follow 120     # sample 120s live

Loaded window = slots/batch >= --min-slots (default 8): excludes game-tail
trickle and idle phases so the statistic is comparable across runs. Judge on
the loaded median; p90 approximates burst capacity.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

_LINE = re.compile(
    r"\[broker\] (?P<batches>\d+) batches in (?P<secs>[\d.]+)s \| "
    r"avg (?P<avg>[\d.]+) pos/batch .*?(?P<slots>[\d.]+)/16 slots/batch \| "
    r"(?P<rate>\d+) pos/s"
)

_DEFAULT_LOG = (
    "runs/pbt2_small/tune/train_trial_4c17c_00000_0_lr=0.0003_2026-07-11_13-16-47/"
    "distributed_inference/broker.out"
)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def _parse(lines: list[str], min_slots: float) -> tuple[list[float], list[float], int]:
    loaded: list[float] = []
    all_rates: list[float] = []
    for line in lines:
        m = _LINE.search(line)
        if not m:
            continue
        rate = float(m.group("rate"))
        all_rates.append(rate)
        if float(m.group("slots")) >= min_slots:
            loaded.append(rate)
    return loaded, all_rates, len(all_rates)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=_DEFAULT_LOG)
    ap.add_argument("--minutes", type=float, default=30.0,
                    help="how far back to read (approx, by line count: 6 lines/min)")
    ap.add_argument("--min-slots", type=float, default=8.0,
                    help="loaded-window threshold (slots/batch)")
    ap.add_argument("--follow", type=float, default=0.0, metavar="SECONDS",
                    help="ignore history; sample the log live for this many seconds")
    args = ap.parse_args()

    log = Path(args.log)
    if not log.exists():
        print(f"no broker log at {log}")
        return 1

    if args.follow > 0:
        start_size = log.stat().st_size
        time.sleep(args.follow)
        with log.open() as f:
            f.seek(start_size)
            lines = f.readlines()
        span = f"live {args.follow:.0f}s"
    else:
        n_lines = max(10, int(args.minutes * 6))
        lines = log.read_text(errors="replace").splitlines()[-n_lines:]
        span = f"last ~{args.minutes:g} min"

    loaded, all_rates, total = _parse(lines, args.min_slots)
    loaded.sort()
    all_rates.sort()
    print(f"broker pos/s ({span}, {total} windows, "
          f"{len(loaded)} loaded @ >={args.min_slots:g} slots)")
    if not all_rates:
        print("  no broker stat lines found in range")
        return 1
    for label, vals in (("loaded", loaded), ("all", all_rates)):
        if not vals:
            print(f"  {label:>6}: none")
            continue
        print(f"  {label:>6}: median {_percentile(vals, 0.5):>8.0f}"
              f"  p90 {_percentile(vals, 0.9):>8.0f}"
              f"  max {vals[-1]:>8.0f}  n={len(vals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
