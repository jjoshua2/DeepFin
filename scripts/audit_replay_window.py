#!/usr/bin/env python3
"""G7/G8/G11/G13: what the replay window is actually made of.

Four invariants that every window-wide readout silently assumes, and that
nothing else measures:

  G7  no row appears twice in the window
  G8  the window's content age is what readouts assume
  G11 composition is stationary across the window's own span
  G13 the holdout is drawn from the same population as the training window

Read-only, and deliberately so: constructing a ``DiskReplayBuffer`` against the
live shard dir DELETES shards, because ``__init__`` enforces the window (audit
G12). This opens the zarr groups directly and reads only small per-row columns
-- never ``x`` -- so a full 1.5M-row window scans in ~20s of CPU and is safe to
run alongside training.

  PYTHONPATH=. python3 scripts/audit_replay_window.py
  PYTHONPATH=. python3 scripts/audit_replay_window.py --shard-dir <dir> --json out.json

Exits 1 if any invariant fails.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[1]

# Columns the audit needs. Everything here is O(rows) bytes; `x` is never read.
_COLS = (
    "game_id", "has_game_id", "ply_index", "has_ply_index",
    "wdl_target", "is_selfplay", "has_is_selfplay",
    "has_seed_id", "has_opening_source_code", "has_sf_wdl", "priority",
)

# A duplicated row is a real defect at any rate; this is the level above which
# it stops being a rounding error on window-wide statistics.
_DUP_FAIL_FRAC = 0.001
# Two content ages this far apart mean the window is a mixture of regimes
# rather than a sliding window over one.
_AGE_SPLIT_H = 48.0


def _scan(shard_dir: Path) -> dict[str, Any]:
    """Read per-row columns and per-shard provenance from every window shard."""
    acc: dict[str, list[np.ndarray]] = {c: [] for c in _COLS}
    row_shard: list[np.ndarray] = []
    sh_index: list[int] = []
    sh_n: list[int] = []
    sh_link: list[int] = []
    sh_ctime: list[float] = []
    unreadable = 0
    for path in sorted(shard_dir.glob("shard_*.zarr")):
        try:
            grp = zarr.open_group(str(path), mode="r")
            cols = {col: np.asarray(grp[col][:]) for col in _COLS if col in set(grp.array_keys())}
            n = int(cols["wdl_target"].shape[0])
            # The link's own mtime is when the window adopted the shard; the
            # target's is when the data was made. G8 is about the latter.
            content_mtime = float(os.stat(os.path.realpath(path)).st_mtime)
        except Exception:
            # Eviction races the scan constantly on a live run; a shard that
            # vanished mid-read is not a finding.
            unreadable += 1
            continue
        for col in _COLS:
            acc[col].append(cols.get(col, np.full(n, -1, dtype=np.int64)))
        row_shard.append(np.full(n, len(sh_index), dtype=np.int32))
        sh_index.append(int(path.stem.split("_")[1]))
        sh_n.append(n)
        sh_link.append(int(path.is_symlink()))
        sh_ctime.append(content_mtime)

    out: dict[str, Any] = {c: np.concatenate(acc[c]) if acc[c] else np.zeros(0) for c in _COLS}
    out["row_shard"] = np.concatenate(row_shard) if row_shard else np.zeros(0, dtype=np.int32)
    out["sh_index"] = np.asarray(sh_index, dtype=np.int64)
    out["sh_n"] = np.asarray(sh_n, dtype=np.int64)
    out["sh_link"] = np.asarray(sh_link, dtype=np.int8)
    out["sh_ctime"] = np.asarray(sh_ctime, dtype=np.float64)
    out["unreadable"] = unreadable
    out["now"] = time.time()
    return out


def _row_key(gid: np.ndarray, ply: np.ndarray) -> np.ndarray:
    """Identity of a training row: which game, which ply of it."""
    return (gid.astype(np.uint64) << np.uint64(12)) ^ (ply.astype(np.uint64) & np.uint64(0xFFF))


def _report_g7(scan: dict[str, Any], report: dict[str, Any]) -> bool:
    gid = np.asarray(scan["game_id"], dtype=np.int64)
    ply = np.asarray(scan["ply_index"], dtype=np.int64)
    keyed = (np.asarray(scan["has_game_id"]) == 1) & (np.asarray(scan["has_ply_index"]) == 1)
    if not keyed.any():
        print("G7  SKIP  no shard carries game_id/ply_index")
        return True

    rows = np.flatnonzero(keyed)
    key = _row_key(gid[rows], ply[rows])
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    extra = int(key.size - cnt.size)
    frac = extra / max(1, key.size)
    dup_rows = rows[(cnt > 1)[inv]]

    link = np.asarray(scan["sh_link"])
    rs = np.asarray(scan["row_shard"])
    in_link = int(link[rs[dup_rows]].sum())
    ok = frac <= _DUP_FAIL_FRAC
    print(f"G7  {'ok  ' if ok else 'FAIL'}  duplicated rows: {extra} extra "
          f"({100 * frac:.3f}% of {key.size} keyed rows), max multiplicity {int(cnt.max())}; "
          f"copies live in {in_link} symlinked and {dup_rows.size - in_link} local rows")

    # Attribute the duplication to shard PAIRS. A concurrent second writer
    # shows up as a constant index offset between the two streams, because
    # each writer allocates its own shard indices from its own counter.
    sh_index = np.asarray(scan["sh_index"])
    order = np.argsort(inv[np.searchsorted(rows, dup_rows)], kind="stable")
    grouped = dup_rows[order]
    gi = inv[np.searchsorted(rows, grouped)]
    bounds = np.flatnonzero(np.diff(gi)) + 1
    pairs: Counter[tuple[int, int]] = Counter()
    for start, end in zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [grouped.size]))):
        shards = sorted({int(sh_index[rs[i]]) for i in grouped[start:end]})
        for a in range(len(shards)):
            for b in range(a + 1, len(shards)):
                pairs[(shards[a], shards[b])] += 1
    deltas: Counter[int] = Counter()
    for (a, b), k in pairs.items():
        deltas[b - a] += k
    if pairs:
        print(f"        shard-index offsets between copies (row-weighted): {deltas.most_common(3)}")
        for (a, b), k in pairs.most_common(3):
            print(f"        shard_{a:06d} <-> shard_{b:06d}: {k} shared rows")

    report["g7"] = {"extra_rows": extra, "frac": frac, "max_multiplicity": int(cnt.max()),
                    "top_offsets": deltas.most_common(3)}
    return ok


def _report_g13(scan: dict[str, Any], holdout_fraction: float, report: dict[str, Any]) -> bool:
    """The holdout split is per-ROW, so its rows' game-mates are in training."""
    gid = np.asarray(scan["game_id"], dtype=np.int64)
    link = np.asarray(scan["sh_link"])
    rs = np.asarray(scan["row_shard"])
    local = link[rs] == 0
    if not local.any():
        print("G13 SKIP  no locally written shards")
        return True
    _, counts = np.unique(gid[local], return_counts=True)
    m = counts.astype(np.float64)
    weight = m / m.sum()
    p = float(holdout_fraction)
    p_orphan = float((weight * p ** np.maximum(m - 1.0, 0.0)).sum())
    siblings = float((weight * (m - 1.0) * (1.0 - p)).sum())
    ok = p_orphan > 0.5
    print(f"G13 {'ok  ' if ok else 'FAIL'}  holdout split is per-row over games of "
          f"{m.mean():.1f} recorded plies: a holdout row has {siblings:.1f} expected same-game "
          f"rows in TRAINING; P(no same-game sibling) = {p_orphan:.2e}")
    report["g13"] = {"rows_per_game": float(m.mean()), "expected_siblings": siblings,
                     "p_no_sibling": p_orphan}
    return ok


def _report_g8(scan: dict[str, Any], report: dict[str, Any]) -> bool:
    now = float(scan["now"])
    link = np.asarray(scan["sh_link"])
    ctime = np.asarray(scan["sh_ctime"])
    sh_n = np.asarray(scan["sh_n"])
    if ctime.size == 0:
        print("G8  SKIP  empty window")
        return True
    age_h = (now - ctime) / 3600.0
    total = int(sh_n.sum())
    print(f"G8  window: {ctime.size} shards, {total} rows, "
          f"{scan['unreadable']} unreadable/evicted during the scan")
    for label, mask in (("salvage(symlink)", link == 1), ("local", link == 0)):
        if not mask.any():
            continue
        rows = int(sh_n[mask].sum())
        print(f"      {label:17s} shards={int(mask.sum()):4d} rows={rows:8d} "
              f"({100 * rows / total:5.1f}%)  age {age_h[mask].min():7.2f}h .. "
              f"{age_h[mask].max():7.2f}h (median {np.median(age_h[mask]):7.2f}h)")

    # Drain projection from the locally written shards' own arrival rate: no
    # second snapshot needed, and it is the rate that actually evicts.
    spread = float(age_h.max() - age_h.min())
    local_n = int((link == 0).sum())
    local_span = float(age_h[link == 0].max() - age_h[link == 0].min()) if local_n > 1 else 0.0
    rate = local_n / local_span if local_span > 0 else 0.0
    if rate > 0 and (link == 1).any():
        drain_h = int((link == 1).sum()) / rate
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(now + 3600 * drain_h))
        print(f"      arrival {rate:.1f} shards/h -> the {int((link == 1).sum())} salvage links "
              f"finish evicting in {drain_h:.1f}h ({when}); every window-wide metric moves "
              f"across that boundary with nothing in the loop having changed")
    ok = spread <= _AGE_SPLIT_H
    print(f"G8  {'ok  ' if ok else 'FAIL'}  content-age spread {spread:.1f}h "
          f"(fails above {_AGE_SPLIT_H:.0f}h: the window is a mixture of regimes, not a "
          f"sliding window over one)")
    report["g8"] = {"shards": int(ctime.size), "rows": total, "age_spread_h": spread,
                    "salvage_shards": int((link == 1).sum()),
                    "salvage_row_frac": float(sh_n[link == 1].sum() / max(1, total)),
                    "arrival_shards_per_h": rate}
    return ok


def _report_g11(scan: dict[str, Any], deciles: int, report: dict[str, Any]) -> bool:
    now = float(scan["now"])
    rs = np.asarray(scan["row_shard"])
    if rs.size == 0:
        print("G11 SKIP  empty window")
        return True
    row_age = (now - np.asarray(scan["sh_ctime"])[rs]) / 3600.0
    wdl = np.asarray(scan["wdl_target"])
    has_sp = np.asarray(scan["has_is_selfplay"]) == 1
    cols: list[tuple[str, np.ndarray]] = [
        ("is_selfplay", np.where(has_sp, np.asarray(scan["is_selfplay"]), np.nan).astype(float)),
        ("wdl_draw", (wdl == 1).astype(float)),
        ("has_seed_id", (np.asarray(scan["has_seed_id"]) == 1).astype(float)),
        ("has_open_src", (np.asarray(scan["has_opening_source_code"]) == 1).astype(float)),
        ("priority", np.asarray(scan["priority"], dtype=float)),
    ]
    order = np.argsort(-row_age)  # oldest decile first
    header = f"      {'dec':>3} {'age_h':>8} {'salv%':>6}" + "".join(f"{c[0]:>14}" for c in cols)
    print(header)
    table: dict[str, list[float]] = {c[0]: [] for c in cols}
    link = np.asarray(scan["sh_link"])
    for i, idx in enumerate(np.array_split(order, deciles)):
        line = (f"      {i:>3} {np.median(row_age[idx]):>8.2f} "
                f"{100 * float((link[rs[idx]] == 1).mean()):>6.1f}")
        for name, values in cols:
            val = float(np.nanmean(values[idx])) if np.isfinite(values[idx]).any() else float("nan")
            table[name].append(val)
            line += f"{val:>14.4f}"
        print(line)

    failed = []
    for name, _ in cols:
        vals = np.asarray(table[name])
        spread = float(np.nanmax(vals) - np.nanmin(vals))
        scale = max(1e-9, abs(float(np.nanmean(vals))))
        if spread > 0.05 * scale + 0.01:
            failed.append(f"{name} {vals[0]:.3f}->{vals[-1]:.3f}")
    ok = not failed
    print(f"G11 {'ok  ' if ok else 'FAIL'}  non-stationary across the window's own span: "
          f"{', '.join(failed) if failed else 'none'}")
    report["g11"] = {"deciles": dict(table), "non_stationary": failed}
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default=None,
                    help="replay_shards dir; default = newest under --replay-root")
    ap.add_argument("--replay-root", default=str(REPO / "runs/pbt2_small/replay"))
    ap.add_argument("--holdout-fraction", type=float, default=0.02,
                    help="must match the live config; drives the G13 arithmetic")
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--json", default=None, help="also write the numbers here")
    args = ap.parse_args()

    if args.shard_dir:
        shard_dir = Path(args.shard_dir)
    else:
        dirs = sorted(Path(args.replay_root).glob("*/replay_shards"),
                      key=lambda p: p.stat().st_mtime)
        if not dirs:
            print(f"no replay_shards dirs under {args.replay_root}\n"
                  "  (a git worktree has no runs/; pass --replay-root pointing at the "
                  "live checkout)")
            return 2
        shard_dir = dirs[-1]
    print(f"window: {shard_dir}")

    t0 = time.time()
    scan = _scan(shard_dir)
    print(f"scanned {scan['sh_index'].size} shards in {time.time() - t0:.0f}s\n")

    report: dict[str, Any] = {"shard_dir": str(shard_dir), "scanned_at": scan["now"]}
    results = [
        _report_g8(scan, report),
        _report_g11(scan, args.deciles, report),
        _report_g7(scan, report),
        _report_g13(scan, args.holdout_fraction, report),
    ]
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=float))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
