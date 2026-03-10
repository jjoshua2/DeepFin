#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any


TRIAL_RE = re.compile(r"^train_trial_([a-z0-9]+)_(\d{5})_")


def parse_trial_dir_name(name: str) -> tuple[str, str] | None:
    m = TRIAL_RE.match(name)
    if not m:
        return None
    return m.group(1), m.group(2)


def score_from_row(row: dict[str, str]) -> tuple[float | None, int]:
    try:
        w = float(row.get("win", 0.0))
        d = float(row.get("draw", 0.0))
        l = float(row.get("loss", 0.0))
    except Exception:
        return None, 0
    n = int(w + d + l)
    if n <= 0:
        return None, 0
    return (w + 0.5 * d) / float(n), n


def pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def latest_prefix(tune_dir: Path) -> str | None:
    latest: tuple[float, str] | None = None
    for d in tune_dir.glob("train_trial_*"):
        if not d.is_dir():
            continue
        parsed = parse_trial_dir_name(d.name)
        if parsed is None:
            continue
        prefix, _ = parsed
        mtime = d.stat().st_mtime
        if latest is None or mtime > latest[0]:
            latest = (mtime, prefix)
    return None if latest is None else latest[1]


def read_progress_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def maybe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    root = Path("/home/josh/projects/chess")
    tune_dir = root / "runs" / "pbt2_small" / "tune"

    now = time.time()
    prefix = latest_prefix(tune_dir)
    print("=" * 72)
    print(f"Hourly PBT Audit @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"tune_dir={tune_dir}")
    if prefix is None:
        print("status=no_trial_dirs")
        return
    print(f"latest_prefix={prefix}")

    trial_dirs = sorted(d for d in tune_dir.glob(f"train_trial_{prefix}_*") if d.is_dir())
    print(f"num_trials={len(trial_dirs)}")
    if not trial_dirs:
        return

    rows: list[dict[str, Any]] = []
    zero_rows_total = 0
    exploit_like_rows_total = 0
    stale_trials = 0
    cross_clone_trials = 0
    rng_meta_present = 0

    for d in trial_dirs:
        parsed = parse_trial_dir_name(d.name)
        if parsed is None:
            continue
        _, trial_num = parsed
        trial_id = f"{prefix}_{trial_num}"
        progress_path = d / "progress.csv"
        progress_rows = read_progress_rows(progress_path)
        if not progress_rows:
            rows.append(
                {
                    "trial_id": trial_id,
                    "score": None,
                    "last5": None,
                    "rows": 0,
                    "age_s": None,
                    "zero_rows": 0,
                    "exploit_like_rows": 0,
                }
            )
            continue

        age_s = int(now - progress_path.stat().st_mtime)
        if age_s > 900:
            stale_trials += 1

        scores_nonzero: list[float] = []
        zero_rows = 0
        exploit_like_rows = 0
        last_row_score = None

        for r in progress_rows:
            s, n = score_from_row(r)
            if n == 0:
                zero_rows += 1
                try:
                    ti = int(float(r.get("training_iteration", 0)))
                    it = int(float(r.get("iter", 0)))
                except Exception:
                    ti = 0
                    it = 0
                if ti > 1 and it == 0:
                    exploit_like_rows += 1
                continue
            scores_nonzero.append(float(s))
            last_row_score = float(s)

        zero_rows_total += zero_rows
        exploit_like_rows_total += exploit_like_rows

        last5 = None
        if scores_nonzero:
            k = min(5, len(scores_nonzero))
            last5 = sum(scores_nonzero[-k:]) / float(k)

        # Prefer config from most recent result row because PB2 mutates hyperparams.
        cfg = None
        result_path = d / "result.json"
        if result_path.exists() and result_path.stat().st_size > 0:
            try:
                with result_path.open("r", encoding="utf-8") as f:
                    last_line = ""
                    for ln in f:
                        if ln.strip():
                            last_line = ln
                    if last_line:
                        j = json.loads(last_line)
                        cfg = j.get("config") if isinstance(j, dict) else None
            except Exception:
                cfg = None

        ckpt_meta = maybe_load_json(d / "ckpt" / "trial_meta.json")
        if ckpt_meta is not None:
            rng_meta_present += 1
            owner = str(ckpt_meta.get("owner_trial_id", ""))
            if owner and owner != trial_id:
                cross_clone_trials += 1

        rows.append(
            {
                "trial_id": trial_id,
                "score": last_row_score,
                "last5": last5,
                "rows": len(progress_rows),
                "age_s": age_s,
                "zero_rows": zero_rows,
                "exploit_like_rows": exploit_like_rows,
                "cfg": cfg if isinstance(cfg, dict) else {},
            }
        )

    valid = [r for r in rows if isinstance(r.get("score"), float)]
    valid5 = [r for r in rows if isinstance(r.get("last5"), float)]
    mean_score = sum(float(r["score"]) for r in valid) / len(valid) if valid else None
    mean_last5 = sum(float(r["last5"]) for r in valid5) / len(valid5) if valid5 else None

    print(
        f"health: stale_trials={stale_trials}/{len(trial_dirs)} "
        f"zero_rows_total={zero_rows_total} exploit_like_rows_total={exploit_like_rows_total} "
        f"rng_meta_present={rng_meta_present}/{len(trial_dirs)} cross_clone_trials={cross_clone_trials}"
    )
    print(
        f"performance: mean_score={mean_score if mean_score is not None else 'n/a'} "
        f"mean_last5={mean_last5 if mean_last5 is not None else 'n/a'}"
    )

    # Rank trials by last5 to inspect winner/loser split.
    ranked = sorted(valid5, key=lambda r: float(r["last5"]), reverse=True)
    print("top_trials(last5):")
    for r in ranked[:3]:
        print(
            f"  {r['trial_id']} last5={r['last5']:.3f} score={r['score']:.3f} "
            f"age_s={r['age_s']} zero_rows={r['zero_rows']} exploit_like={r['exploit_like_rows']}"
        )
    print("bottom_trials(last5):")
    for r in ranked[-3:]:
        print(
            f"  {r['trial_id']} last5={r['last5']:.3f} score={r['score']:.3f} "
            f"age_s={r['age_s']} zero_rows={r['zero_rows']} exploit_like={r['exploit_like_rows']}"
        )

    # Lightweight hyperparam correlation snapshot.
    hp_keys = ["lr", "w_soft", "w_future", "w_sf_move", "soft_policy_temp"]
    for key in hp_keys:
        xs: list[float] = []
        ys: list[float] = []
        for r in valid5:
            cfg = r.get("cfg") or {}
            if key not in cfg:
                continue
            try:
                x = float(cfg[key])
            except Exception:
                continue
            xs.append(x)
            ys.append(float(r["last5"]))
        c = pearson_corr(xs, ys)
        if c is None:
            print(f"corr(last5,{key})=n/a")
        else:
            print(f"corr(last5,{key})={c:+.3f} (n={len(xs)})")


if __name__ == "__main__":
    main()

