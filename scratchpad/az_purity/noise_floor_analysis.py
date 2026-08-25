#!/usr/bin/env python3
"""Noise-floor / standardisation analysis over `score_shard_labels.py --dump-rows`.

Two readings of the same rows, both required by the #273 prereg:

* RAW mean top-1 regret -- the property of the CORPUS the arm produces. It
  includes the arm's position mix, because the mix is part of what training
  sees. This is the prereg's PRIMARY.
* PHASE-STANDARDISED mean -- the same per-phase means reweighted onto one fixed
  reference phase distribution, so two cells are compared at the same mix. This
  isolates LABELLING from composition. Required secondary: a raw/standardised
  disagreement is a finding about the arm's trajectories, not about its labels.

Both carry cluster bootstrap CIs over GAMES, and the between-cell difference of
two same-config cells is reported for each -- that difference IS the noise floor.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PHASES = (0, 1, 2)
PHASE_NAMES = ("endgame", "middlegame", "opening")


def load(path: Path) -> dict[str, list[dict]]:
    cells: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                cells[str(r["cell"])].append(r)
    return dict(cells)


def _cell_arrays(rows: list[dict], metric: str):
    v = np.array([float(r[metric]) for r in rows])
    g = np.array([int(r["game_id"]) for r in rows])
    p = np.array([int(r["phase"]) for r in rows])
    return v, g, p


def _boot_index(g: np.ndarray, n_boot: int, rng: np.random.Generator):
    uniq = np.unique(g)
    return uniq, rng.integers(0, len(uniq), size=(n_boot, len(uniq)))


def raw_boot(rows: list[dict], metric: str, n_boot: int, rng) -> tuple[float, np.ndarray]:
    v, g, _ = _cell_arrays(rows, metric)
    uniq, draws = _boot_index(g, n_boot, rng)
    s = np.array([float(v[g == u].sum()) for u in uniq])
    c = np.array([float((g == u).sum()) for u in uniq])
    return float(v.mean()), s[draws].sum(axis=1) / np.maximum(c[draws].sum(axis=1), 1e-9)


def std_boot(
    rows: list[dict], metric: str, weights: np.ndarray, n_boot: int, rng,
) -> tuple[float, np.ndarray]:
    """Phase-standardised mean and its cluster bootstrap distribution."""
    v, g, p = _cell_arrays(rows, metric)
    uniq, draws = _boot_index(g, n_boot, rng)
    point = float(sum(
        weights[ph] * v[p == ph].mean() for ph in PHASES if (p == ph).any()
    ))
    sums = np.zeros((len(uniq), len(PHASES)))
    cnts = np.zeros((len(uniq), len(PHASES)))
    for i, u in enumerate(uniq):
        sel = g == u
        for ph in PHASES:
            m = sel & (p == ph)
            sums[i, ph] = float(v[m].sum())
            cnts[i, ph] = float(m.sum())
    bs = sums[draws].sum(axis=1)          # (n_boot, 3)
    bc = cnts[draws].sum(axis=1)
    # A bootstrap draw with an empty phase falls back to that phase's overall
    # mean rather than contributing NaN: dropping the draw would condition the
    # resample on its own composition.
    fallback = np.array([
        float(v[p == ph].mean()) if (p == ph).any() else 0.0 for ph in PHASES
    ])
    means = np.where(bc > 0, bs / np.maximum(bc, 1e-9), fallback[None, :])
    return point, (means * weights[None, :]).sum(axis=1)


def ci(dist: np.ndarray) -> list[float]:
    return [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--metric", default="top1_regret_cp")
    ap.add_argument(
        "--reference-cell", required=True,
        help="cell whose phase distribution is the standardisation reference",
    )
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    cells = load(args.rows)
    if args.reference_cell not in cells:
        raise SystemExit(f"reference cell {args.reference_cell!r} not in dump")
    ref_p = np.array([int(r["phase"]) for r in cells[args.reference_cell]])
    weights = np.array([float((ref_p == ph).mean()) for ph in PHASES])
    print(
        f"[std] reference={args.reference_cell} phase weights "
        + " ".join(f"{PHASE_NAMES[ph]}={weights[ph]:.4f}" for ph in PHASES),
        flush=True,
    )

    out: dict = {
        "metric": args.metric, "reference_cell": args.reference_cell,
        "phase_weights": {PHASE_NAMES[ph]: float(weights[ph]) for ph in PHASES},
        "cells": {}, "deltas": {},
    }
    dists: dict[str, dict[str, np.ndarray]] = {}
    for label, rows in cells.items():
        rng = np.random.default_rng(args.seed)
        r_pt, r_bs = raw_boot(rows, args.metric, args.n_boot, rng)
        rng = np.random.default_rng(args.seed)
        s_pt, s_bs = std_boot(rows, args.metric, weights, args.n_boot, rng)
        dists[label] = {"raw": r_bs, "std": s_bs}
        p = np.array([int(x["phase"]) for x in rows])
        out["cells"][label] = {
            "n": len(rows),
            "phase_mix": {
                PHASE_NAMES[ph]: float((p == ph).mean()) for ph in PHASES
            },
            "raw": {"mean": r_pt, "ci95": ci(r_bs)},
            "standardised": {"mean": s_pt, "ci95": ci(s_bs)},
        }
        print(
            f"[cell {label}] n={len(rows)} raw {r_pt:8.2f} {ci(r_bs)}  "
            f"std {s_pt:8.2f} {ci(s_bs)}  mix "
            + "/".join(f"{float((p == ph).mean()):.3f}" for ph in PHASES),
            flush=True,
        )

    labels = list(cells)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            row = {}
            for kind in ("raw", "std"):
                da = dists[a][kind] - dists[b][kind]
                pa = out["cells"][a]["raw" if kind == "raw" else "standardised"]["mean"]
                pb = out["cells"][b]["raw" if kind == "raw" else "standardised"]["mean"]
                row[kind] = {"delta": pa - pb, "ci95": ci(da), "sd": float(da.std(ddof=1))}
            out["deltas"][f"{a}-minus-{b}"] = row
            print(
                f"[delta] {a} - {b}: raw {row['raw']['delta']:+7.2f} "
                f"{[round(x, 2) for x in row['raw']['ci95']]}   "
                f"std {row['std']['delta']:+7.2f} "
                f"{[round(x, 2) for x in row['std']['ci95']]}",
                flush=True,
            )

    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[out] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
