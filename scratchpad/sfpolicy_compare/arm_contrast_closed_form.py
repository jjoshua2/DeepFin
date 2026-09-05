#!/usr/bin/env python3
"""PHASE 1 of the w_sf_own_regret arm comparison: closed-form, CPU-only, zero GPU.

Josh asked for an offline contrast of the candidate arms -- how much they actually differ
from each other, and which best predicts a stronger reference -- BEFORE any training
compute. This is that, restricted to what is computable in closed form from banked shard
fields. It cannot rank the arms on strength; it can KILL an arm by showing it is
near-identical to another (nothing to learn from running both) or inert.

FOUR ARMS, all closed-form on stored fields:
  A  native magnitude      E_p[regret] over ALL entries  -- what `w_sf_own_regret` does today
  B  native, surfaced only  E_p[regret] over SURFACED entries -- PR #447's gate. B-A isolates
                            the FABRICATED-TAIL axis (the tail is ~74% invented on 16.3% of rows)
  C  one-sided floor        hinge on the target's mass on SF's surfaced top-2. C-B isolates the
                            FUNCTIONAL-FORM axis (threshold vs magnitude)
  D  look-ahead weight      `future_sf_regret_h4`/`h6` -- realized SF regret over the rest of the
                            game. A PER-ROW SCALAR, not a per-move term (see below)

⚑ A/B/C ARE PER-MOVE TERMS AND D IS A PER-ROW WEIGHT. They are not commensurable in one
metric, and pretending otherwise is the trap. So this reports (i) per-move disagreement
among A/B/C, and (ii) D's correlation with A/B/C's PER-ROW magnitudes. (ii) is the
decisive one: `sf_p0_regret` covers 24.6% of rows and `future_sf_regret_*` covers
100.00%, so if D tracks A's per-row magnitude we have the same signal at 4x coverage for
zero extra SF cost.

⚑ NO DIRECTION IS ASSUMED for the look-ahead fields. Their sign/scale convention is
checked against `wdl_target` first (higher realized future regret should mean a WORSE
outcome for the side to move), because this session already produced one INVERTED
collapse measurement from assuming a POV. If that check fails, the D columns are reported
as VOID rather than silently reinterpreted.

Surfaced/fill separation uses PR #447's arithmetic bound: `finalize.py` sets the constant
tail to `(worst_surfaced + 1.0) / 2.0` with `worst_surfaced` in [0,1], so the fill is
ALWAYS in [0.5, 1.0] and an entry strictly below 0.5 provably carries no fill. Exactly
representable in float16, which is what `sf_p0_regret` is stored as.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

SF_REGRET_MIN_FILL = 0.5


def _get(z, key: str) -> np.ndarray | None:
    try:
        return np.asarray(z[key][:])
    except (KeyError, TypeError):
        return None


def arms_for_shard(path: Path, *, floor: float) -> dict[str, np.ndarray]:
    try:
        z = zarr.open(str(path), mode="r")
    except Exception:
        return {}
    reg = _get(z, "sf_p0_regret")
    hreg = _get(z, "has_sf_p0_regret")
    pol = _get(z, "policy_target")
    legal = _get(z, "legal_mask")
    if reg is None or hreg is None or pol is None or legal is None:
        return {}

    out: dict[str, np.ndarray] = {}

    # ── D's population is EVERY row (100% coverage) and is collected separately from
    # A/B/C's, because restricting D to A/B/C's rows would hide exactly the coverage
    # advantage the comparison is about. The correlation block below re-restricts.
    wdl = _get(z, "wdl_target")
    for h in ("future_sf_regret_h4", "future_sf_regret_h6", "future_sf_regret_max"):
        v = _get(z, h)
        if v is not None:
            out[f"D_all::{h}"] = v.astype(np.float32)
    if wdl is not None:
        out["D_all::wdl_target"] = wdl.astype(np.float32)

    m = hreg.astype(bool)
    if not m.any():
        return out
    r = reg[m].astype(np.float32)
    p = pol[m].astype(np.float32)
    lg = legal[m].astype(bool)
    tot = p.sum(-1)
    ok = np.isfinite(tot) & (tot > 1e-3)
    if not ok.any():
        return out
    r, p, lg = r[ok], p[ok] / tot[ok][:, None], lg[ok]
    idx = np.where(m)[0][ok]

    # surfaced = provably-real entries (strictly below the fill's arithmetic floor)
    surf = lg & (r < SF_REGRET_MIN_FILL)
    n_surf = surf.sum(-1)

    # ── ARM A: native, all LEGAL entries (what the live term computes)
    out["A_expected_regret_all"] = (p * np.where(lg, r, 0.0)).sum(-1)
    # ── ARM B: native, surfaced only
    out["B_expected_regret_surfaced"] = (p * np.where(surf, r, 0.0)).sum(-1)
    # ── ARM C: one-sided floor on the target's mass over SF's surfaced top-2
    rr = np.where(surf, r, np.float32(np.inf))
    order = np.argsort(rr, axis=-1, kind="stable")
    rows = np.arange(len(p))
    top2 = p[rows, order[:, 0]] + np.where(
        n_surf >= 2, p[rows, order[:, 1]], np.float32(0.0),
    )
    out["C_floor_hinge"] = np.maximum(0.0, floor - top2).astype(np.float32)
    out["C_mass_on_top2"] = top2.astype(np.float32)
    out["n_surfaced"] = n_surf.astype(np.float32)

    # ── D restricted to A/B/C's rows, for the correlation block
    for h in ("future_sf_regret_h4", "future_sf_regret_h6", "future_sf_regret_max"):
        v = _get(z, h)
        if v is not None:
            out[f"D_paired::{h}"] = v[idx].astype(np.float32)
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, computed by hand so this file needs no scipy."""
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 100:
        return float("nan")
    a, b = a[keep], b[keep]
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--max-shards", type=int, default=200)
    ap.add_argument("--floor", type=float, default=0.10,
                    help="arm C's floor on surfaced-top-2 mass (0.10 => binds 23.0%%)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    paths = sorted(args.shard_dir.glob("shard_*.zarr"))[: args.max_shards]
    acc: dict[str, list[np.ndarray]] = {}
    bad = 0
    for pth in paths:
        got = arms_for_shard(pth, floor=args.floor)
        if not got:
            bad += 1
            continue
        for k, v in got.items():
            acc.setdefault(k, []).append(v)
    if not acc:
        raise SystemExit("no usable shards")
    cat = {k: np.concatenate(v) for k, v in acc.items()}
    rep: dict[str, object] = {"shards": len(paths), "unusable": bad, "floor": args.floor}
    print(f"shards={len(paths)} unusable={bad}")
    print(f"rows: D population (all) = {len(cat['D_all::wdl_target']):,}   "
          f"A/B/C population (has sf_p0_regret) = {len(cat['A_expected_regret_all']):,}")

    # ── ⚑ DIRECTION CHECK FIRST. If the look-ahead sign convention is not what the name
    # implies, every D number below is void. wdl_target: 0=W 1=D 2=L from the STM's POV,
    # so future regret should RISE from W to L.
    w = cat["D_all::wdl_target"]
    print("\nDIRECTION CHECK (wdl_target 0=W 1=D 2=L, STM POV) -- future regret should RISE W->L")
    d_ok = True
    for h in ("future_sf_regret_h4", "future_sf_regret_h6", "future_sf_regret_max"):
        k = f"D_all::{h}"
        if k not in cat:
            continue
        v = cat[k]
        means = [float(v[w == c].mean()) if (w == c).any() else float("nan") for c in (0, 1, 2)]
        mono = means[0] <= means[1] <= means[2]
        rep[f"direction::{h}"] = {"W": means[0], "D": means[1], "L": means[2], "monotone_W_to_L": mono}
        print(f"  {h:26s} W={means[0]:9.4f} D={means[1]:9.4f} L={means[2]:9.4f}"
              f"   {'OK (monotone)' if mono else '⚑ NOT monotone'}")
        d_ok = d_ok and mono
    if not d_ok:
        print("  ⚑⚑ at least one look-ahead field is NOT monotone in outcome -- treat its "
              "correlations below as UNINTERPRETED, not as a verdict on arm D.")
    rep["direction_check_passed"] = bool(d_ok)

    # ── A/B/C: magnitude, spread, and how often each is inert
    print(f"\n{'arm':28s} {'mean':>10s} {'p50':>10s} {'p90':>10s} {'frac==0':>9s}")
    for k in ("A_expected_regret_all", "B_expected_regret_surfaced", "C_floor_hinge",
              "C_mass_on_top2", "n_surfaced"):
        v = cat[k]
        rep[k] = {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
                  "p90": float(np.percentile(v, 90)), "frac_zero": float((v == 0).mean()),
                  "n": int(v.size)}
        print(f"{k:28s} {v.mean():10.5f} {np.percentile(v,50):10.5f} "
              f"{np.percentile(v,90):10.5f} {(v==0).mean()*100:8.2f}%")

    # ── THE DECIDING BLOCK 1: do A and B differ enough to be worth two arms?
    a, b = cat["A_expected_regret_all"], cat["B_expected_regret_surfaced"]
    ratio = np.divide(b, a, out=np.zeros_like(b), where=a > 1e-9)
    rep["A_vs_B"] = {"spearman": _spearman(a, b), "mean_ratio_B_over_A": float(ratio[a > 1e-9].mean()),
                     "frac_rows_B_lt_half_A": float((ratio[a > 1e-9] < 0.5).mean())}
    print(f"\nA vs B (does dropping the fabricated tail change the term?)")
    print(f"  spearman(A,B)          = {_spearman(a,b):+.4f}")
    print(f"  mean B/A               = {ratio[a>1e-9].mean():.4f}")
    print(f"  rows where B < 0.5*A   = {(ratio[a>1e-9]<0.5).mean()*100:.2f}%")

    # ── DECIDING BLOCK 2: is C a different signal from A/B, or a monotone restatement?
    c = cat["C_floor_hinge"]
    rep["C_vs_AB"] = {"spearman_C_A": _spearman(c, a), "spearman_C_B": _spearman(c, b)}
    print(f"\nC vs A/B (is the floor a different signal or a restatement?)")
    print(f"  spearman(C,A) = {_spearman(c,a):+.4f}    spearman(C,B) = {_spearman(c,b):+.4f}")

    # ── ⚑⚑ DECIDING BLOCK 3, THE ONE THAT MATTERS MOST: does the 100%-coverage
    # look-ahead scalar track the 24.6%-coverage own-move regret? If yes, the same
    # signal is available on 4x the rows at zero extra SF cost.
    print(f"\n⚑⚑ D vs A/B/C on the PAIRED rows (D covers 100% of rows, A/B/C only 24.6%)")
    rep["D_vs_ABC"] = {}
    for h in ("future_sf_regret_h4", "future_sf_regret_h6", "future_sf_regret_max"):
        k = f"D_paired::{h}"
        if k not in cat:
            continue
        d = cat[k]
        n = min(len(d), len(a))
        row = {"spearman_D_A": _spearman(d[:n], a[:n]),
               "spearman_D_B": _spearman(d[:n], b[:n]),
               "spearman_D_C": _spearman(d[:n], c[:n])}
        rep["D_vs_ABC"][h] = row
        print(f"  {h:26s} vs A {row['spearman_D_A']:+.4f}   vs B {row['spearman_D_B']:+.4f}"
              f"   vs C {row['spearman_D_C']:+.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
