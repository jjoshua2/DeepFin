#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def _latest_prefix(tune_dir: Path) -> str | None:
    latest: tuple[float, str] | None = None
    for d in tune_dir.glob("train_trial_*"):
        if not d.is_dir():
            continue
        parts = d.name.split("_")
        if len(parts) < 4:
            continue
        prefix = parts[2]
        mt = d.stat().st_mtime
        if latest is None or mt > latest[0]:
            latest = (mt, prefix)
    return None if latest is None else latest[1]


def _read_last_json_row(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    last: dict | None = None
    try:
        with path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except Exception:
                    continue
                if isinstance(row, dict):
                    last = row
    except Exception:
        return None
    return last


def _safe_int(v: object, default: int = 0) -> int:
    try:
        return int(v)  # type: ignore[arg-type]
    except Exception:
        return default


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return default


def _raw_winrate(row: dict) -> float:
    w = _safe_float(row.get("win"), 0.0)
    d = _safe_float(row.get("draw"), 0.0)
    l = _safe_float(row.get("loss"), 0.0)
    n = max(1.0, w + d + l)
    return (w + 0.5 * d) / n


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def _fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}"


def run_once(*, root: Path, prefix: str | None) -> int:
    tune_dir = root / "runs" / "pbt2_small" / "tune"
    run_prefix = prefix or _latest_prefix(tune_dir)
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 88)
    print(f"30m PBT Poll @ {now}")
    print(f"tune_dir={tune_dir}")
    if not run_prefix:
        print("status=no_trials")
        return 1
    print(f"run_prefix={run_prefix}")

    trial_dirs = sorted(p for p in tune_dir.glob(f"train_trial_{run_prefix}_*") if p.is_dir())
    if not trial_dirs:
        print("status=no_matching_trials")
        return 1
    print(f"num_trials={len(trial_dirs)}")

    per_trial: list[dict] = []
    for td in trial_dirs:
        row = _read_last_json_row(td / "result.json")
        params = {}
        try:
            params = json.loads((td / "params.json").read_text(encoding="utf-8"))
        except Exception:
            params = {}

        share_expected = bool(params.get("exploit_replay_share_top_enabled", False))
        if row is None:
            per_trial.append(
                {
                    "trial": td.name,
                    "iter": -1,
                    "raw_wr": None,
                    "ema_wr": None,
                    "rand": None,
                    "opp": None,
                    "share_sel": 0,
                    "share_ing": 0,
                    "share_samp": 0,
                    "share_expected": share_expected,
                    "rows": 0,
                }
            )
            continue

        per_trial.append(
            {
                "trial": td.name,
                "iter": _safe_int(row.get("iter", row.get("training_iteration", -1)), -1),
                "raw_wr": _raw_winrate(row),
                "ema_wr": _safe_float(row.get("pid_ema_winrate"), 0.0),
                "rand": _safe_float(row.get("random_move_prob"), 0.0),
                "opp": _safe_float(row.get("opponent_strength"), 0.0),
                "share_sel": _safe_int(row.get("shared_trials_selected"), 0),
                "share_ing": _safe_int(row.get("shared_trials_ingested"), 0),
                "share_samp": _safe_int(row.get("shared_samples_ingested"), 0),
                "share_expected": share_expected,
                "rows": 1,
            }
        )

    ready = [r for r in per_trial if r["rows"] > 0]
    raws = [float(r["raw_wr"]) for r in ready if r["raw_wr"] is not None]
    emas = [float(r["ema_wr"]) for r in ready if r["ema_wr"] is not None]
    rands = [float(r["rand"]) for r in ready if r["rand"] is not None]
    opps = [float(r["opp"]) for r in ready if r["opp"] is not None]
    iters = [int(r["iter"]) for r in ready]

    exp_share_trials = [r for r in ready if r["share_expected"]]
    share_sel_nonzero = sum(1 for r in ready if int(r["share_sel"]) > 0)
    share_ing_nonzero = sum(1 for r in ready if int(r["share_ing"]) > 0)
    share_samp_total = sum(int(r["share_samp"]) for r in ready)

    print(
        "summary: "
        f"rows={len(ready)}/{len(per_trial)} "
        f"iter_minmax={min(iters) if iters else 'n/a'}..{max(iters) if iters else 'n/a'} "
        f"raw_wr[min/med/max]={_fmt(min(raws) if raws else None)}/{_fmt(_median(raws) if raws else None)}/{_fmt(max(raws) if raws else None)} "
        f"ema_wr[med]={_fmt(_median(emas) if emas else None)} "
        f"rand[min/max]={_fmt(min(rands) if rands else None)}/{_fmt(max(rands) if rands else None)} "
        f"opp[med]={_fmt(_median(opps) if opps else None)}"
    )
    print(
        "share: "
        f"expected_flag_trials={len(exp_share_trials)}/{len(ready)} "
        f"selected_nonzero={share_sel_nonzero}/{len(ready)} "
        f"ingested_nonzero={share_ing_nonzero}/{len(ready)} "
        f"samples_total={share_samp_total}"
    )
    if iters and max(iters) >= 1 and share_samp_total == 0:
        print("ALERT: iter>=1 but shared_samples_ingested is still zero across all trials.")

    # Print concise per-trial lines for quick scanning.
    for r in sorted(per_trial, key=lambda x: str(x["trial"])):
        print(
            f"trial={r['trial']} "
            f"iter={r['iter']} "
            f"raw_wr={_fmt(r['raw_wr']) if r['raw_wr'] is not None else 'n/a'} "
            f"ema_wr={_fmt(r['ema_wr']) if r['ema_wr'] is not None else 'n/a'} "
            f"rand={_fmt(r['rand']) if r['rand'] is not None else 'n/a'} "
            f"share_sel={r['share_sel']} share_ing={r['share_ing']} share_samp={r['share_samp']}"
        )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Poll latest PBT trial metrics once.")
    ap.add_argument("--root", type=Path, default=Path("/home/josh/projects/chess"))
    ap.add_argument("--prefix", type=str, default=None)
    args = ap.parse_args()
    raise SystemExit(run_once(root=args.root, prefix=args.prefix))


if __name__ == "__main__":
    main()
