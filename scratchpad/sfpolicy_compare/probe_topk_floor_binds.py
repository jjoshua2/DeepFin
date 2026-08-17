#!/usr/bin/env python3
"""STAGE-0 PRECONDITION for the SF-policy arm comparison: does a top-k floor BIND?

Josh's criterion, in his words: "make sure it isn't completely ignoring the main SF
move or two". This asks that of STORED data with ZERO GPU, because it can kill the arm
before any compute -- the same precondition shape that already saved a day on #447,
where `sf_own_regret_listed_mass_min: 0.10` turned out to gate ZERO rows.

⚑ WHAT THIS MEASURES, PRECISELY, AND WHAT IT DOES NOT.
It measures the mass the stored `policy_target` places on SF's top-1 / top-2. That is
the quantity that decides REDUNDANCY: `policy_target` is what the policy loss already
trains toward, so if it ALREADY concentrates on SF's top move then an SF top-k floor
carries no information the existing loss lacks, and the arm is a guaranteed null.
It does NOT measure the NET's mass -- the floor acts on the net's output, and the net
may be failing to fit its target. That residual needs a forward pass and is Stage 0b.
Reporting this as "the net ignores SF" would be the exact same-name-different-population
error the ledger keeps catching, so the columns below are named for the TARGET.

⚑⚑ THERE ARE TWO 1858-WIDE MOVE FRAMES IN THE SHARD AND THEY ARE NOT THE SAME ENCODING.
MEASURED here, 2026-08-17, over 36,324 rows / 20 production shards:
  * `sf_move_index` is legal under `sf_legal_mask` on 100.00% of rows and under
    `legal_mask` on only 7.36%.
  * `sf_p0_regret`'s SURFACED entries are legal under `legal_mask` on 100.00% (and every
    one of 7,970 rows has its whole surfaced set inside it) but under `sf_legal_mask` on
    10.01%.
  * `legal_mask` and `sf_legal_mask` are identical on 0.00% of rows -- popcounts 26.74
    vs 26.22, so they are the same SIZE and different CONTENT, which is why nothing
    downstream notices.
⇒ `sf_p0_regret` is in the NET's frame (so the live `(po_probs * reg_vec).sum(-1)` term
  and PR #447's gate are correctly indexed), while `sf_move_index` is in SF's frame.
⇒ **Indexing `policy_target` with `sf_move_index` returns a real-looking probability for
  the WRONG MOVE on ~93% of rows.** The first version of this probe did exactly that and
  read "the target puts 0.0038 mass on SF's best move" -- a fabricated crisis. Any top-k
  floor built on `sf_move_index` needs an explicit remap; the one built on
  `sf_p0_regret`'s surfaced ranking needs none. The `sf_move_index` leg below is retained
  ONLY as a live regression check on that frame split, and is NEVER read as a mass.

This is the classic shape from the ledger: same name (`*_mask`, 1858 uint8, ~26.5
popcount), different population. A presence check cannot see it; only a cross-frame
legality test can.

Surfaced-vs-fabricated uses the arithmetic bound from PR #447: `finalize.py` sets the
constant tail to `(worst_surfaced + 1.0) / 2.0` with `worst_surfaced` in [0, 1], so the
fill is ALWAYS in [0.5, 1.0]. An entry strictly below 0.5 therefore PROVABLY carries no
fill. Exactly representable in float16, so no tolerance constant is needed -- and
`sf_p0_regret` is stored float16, where a tolerance would rot silently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

SF_REGRET_MIN_FILL = 0.5


def _col(z, key: str) -> np.ndarray | None:
    try:
        return np.asarray(z[key][:])
    except (KeyError, TypeError):
        return None


class ShardUnreadable(Exception):
    """A shard that could not be opened or decoded.

    ⚑ COUNTED AND REPORTED, never silently skipped. A salvage bank can hold a shard that
    was still being written when the export ran, and `zarr.open` then raises a bare
    `JSONDecodeError` from deep inside its metadata parser. Swallowing that would shrink
    the population invisibly -- the ledger's own repeated finding -- so the run prints the
    unreadable count next to the shard count and FAILS if it is more than a few percent.
    """


def probe_shard(path: Path) -> dict[str, np.ndarray]:
    try:
        z = zarr.open(str(path), mode="r")
    except Exception as exc:  # zarr raises JSONDecodeError/KeyError/OSError variously
        raise ShardUnreadable(f"{path.name}: {type(exc).__name__}: {exc}") from exc
    pol = _col(z, "policy_target")
    has_pol = _col(z, "has_policy")
    sf_mi = _col(z, "sf_move_index")
    has_sf_mi = _col(z, "has_sf_move")
    reg = _col(z, "sf_p0_regret")
    has_reg = _col(z, "has_sf_p0_regret")
    legal = _col(z, "legal_mask")
    if pol is None or sf_mi is None:
        return {}

    pol = pol.astype(np.float32)
    # Renormalize defensively: a stored row that does not sum to 1 would make every
    # "mass" number below incomparable across rows. Rows that cannot be normalized
    # are dropped rather than divided by ~0.
    tot = pol.sum(-1)
    ok = (has_pol.astype(bool) if has_pol is not None else np.ones(len(pol), bool))
    ok &= np.isfinite(tot) & (tot > 1e-3)
    if has_sf_mi is not None:
        ok &= has_sf_mi.astype(bool)
    ok &= (sf_mi >= 0) & (sf_mi < pol.shape[1])
    if not ok.any():
        return {}
    pol = pol[ok] / tot[ok][:, None]
    sf_mi_ok = sf_mi[ok]
    rows = np.arange(len(pol))

    out: dict[str, np.ndarray] = {
        # Denominator context: how peaked is the target at all, and how many legal
        # moves does it choose among? A low mass on SF's top-2 in a 40-move position
        # means something different from the same number in a 3-move position.
        "target_top1_mass": pol.max(-1),
        "legal_count": (
            legal[ok].astype(bool).sum(-1).astype(np.float32)
            if legal is not None else np.full(len(pol), np.nan, np.float32)
        ),
    }
    # ⚑ FRAME REGRESSION CHECK, not a mass. `sf_move_index` is in SF's frame, so its
    # legality rate under the NET's `legal_mask` is expected to be LOW (~7%) and its
    # rate under `sf_legal_mask` ~100%. If those ever converge, the frames have been
    # unified upstream and the docstring above -- plus any consumer of `sf_move_index`
    # -- must be re-derived. Reported as rates so a change is visible, never indexed
    # into `policy_target`.
    sf_legal = _col(z, "sf_legal_mask")
    if legal is not None:
        out["sf_move_index_legal_under_NET_mask"] = (
            legal[ok][rows, sf_mi_ok].astype(bool).astype(np.float32)
        )
    if sf_legal is not None:
        out["sf_move_index_legal_under_SF_mask"] = (
            sf_legal[ok][rows, sf_mi_ok].astype(bool).astype(np.float32)
        )

    # ── the richer, narrower source: the surfaced regret ranking ──
    if reg is not None and has_reg is not None:
        rmask = has_reg.astype(bool)[ok]
        if rmask.any():
            r = reg[ok][rmask].astype(np.float32)
            p = pol[rmask]
            sfi = sf_mi_ok[rmask]
            surfaced = r < SF_REGRET_MIN_FILL
            if legal is not None:
                surfaced &= legal[ok][rmask].astype(bool)
            n_surf = surfaced.sum(-1)
            keep = n_surf >= 2  # need >=2 to speak about a top-2 at all
            if keep.any():
                r, p, sfi, surfaced = r[keep], p[keep], sfi[keep], surfaced[keep]
                # Rank surfaced entries by regret ascending; +inf parks the rest so
                # argsort can never select a fabricated or illegal action.
                rr = np.where(surfaced, r, np.float32(np.inf))
                order = np.argsort(rr, axis=-1, kind="stable")
                i1, i2 = order[:, 0], order[:, 1]
                rr2 = np.arange(len(p))
                # NOT a validity gate -- see the frame finding in the docstring. The two
                # indices live in different frames, so disagreement here is EXPECTED and
                # a HIGH rate would be the surprise. Kept as the paired half of the
                # frame regression check.
                out["reg_top1_equals_sf_move_index_EXPECTED_LOW"] = (
                    (i1 == sfi).astype(np.float32)
                )
                out["mass_on_regret_top1"] = p[rr2, i1]
                out["mass_on_regret_top2"] = p[rr2, i1] + p[rr2, i2]
                out["surfaced_count"] = surfaced.sum(-1).astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--max-shards", type=int, default=60)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    paths = sorted(args.shard_dir.glob("shard_*.zarr"))[: args.max_shards]
    if not paths:
        raise SystemExit(f"no shards under {args.shard_dir}")

    acc: dict[str, list[np.ndarray]] = {}
    unreadable: list[str] = []
    for pth in paths:
        try:
            fields = probe_shard(pth)
        except ShardUnreadable as exc:
            unreadable.append(str(exc))
            continue
        for k, v in fields.items():
            acc.setdefault(k, []).append(v)
    if not acc:
        raise SystemExit("every shard yielded zero usable rows")
    cat = {k: np.concatenate(v) for k, v in acc.items()}

    report: dict[str, object] = {
        "shards_listed": len(paths),
        "shards_read": len(paths) - len(unreadable),
        "shards_unreadable": len(unreadable),
        "unreadable_detail": unreadable[:10],
        "shard_dir": str(args.shard_dir),
    }
    if unreadable:
        print(f"⚑ {len(unreadable)} of {len(paths)} shards UNREADABLE "
              f"(first: {unreadable[0]})")
        if len(unreadable) > max(2, len(paths) // 20):
            raise SystemExit(
                f"{len(unreadable)}/{len(paths)} shards unreadable -- the population is "
                "not what the shard list says; fix the bank before reading a verdict",
            )
    report["rows_policy_leg"] = int(len(cat["target_top1_mass"]))
    print(f"shards={len(paths)}  rows with a usable policy_target="
          f"{len(cat['target_top1_mass']):,}")

    # ⚑ FRAME CHECK FIRST, because it decides which numbers below are about SF's move at
    # all. This is a REGRESSION check on a measured fact, not a pass/fail gate: the
    # expected pattern is SF-mask ~100% / NET-mask ~7%, and CONVERGENCE is the anomaly.
    fnet = cat.get("sf_move_index_legal_under_NET_mask")
    fsf = cat.get("sf_move_index_legal_under_SF_mask")
    if fnet is not None and fsf is not None:
        a, b = float(fnet.mean()), float(fsf.mean())
        report["frame_check"] = {
            "sf_move_index_legal_under_net_mask": a,
            "sf_move_index_legal_under_sf_mask": b,
            "n": int(fnet.size),
        }
        split = "SPLIT as expected" if (b > 0.95 and a < 0.30) else (
            "⚑ FRAMES MAY HAVE CONVERGED -- re-derive every sf_move_index consumer")
        print(f"\nFRAME CHECK over {fnet.size:,} rows: sf_move_index legal under "
              f"SF mask {b:.4f} / under NET mask {a:.4f}  [{split}]")
    agree = cat.get("reg_top1_equals_sf_move_index_EXPECTED_LOW")
    if agree is not None:
        report["reg_top1_equals_sf_move_index_expected_low"] = float(agree.mean())

    print(f"\n{'metric':38s} {'mean':>8s} {'p10':>8s} {'p25':>8s} "
          f"{'p50':>8s} {'p75':>8s} {'p90':>8s}")
    for k in ("mass_on_regret_top1", "mass_on_regret_top2",
              "target_top1_mass", "legal_count", "surfaced_count"):
        v = cat.get(k)
        if v is None:
            continue
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        q = np.percentile(v, [10, 25, 50, 75, 90])
        report[k] = {"mean": float(v.mean()), "n": int(v.size),
                     "p10": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
                     "p75": float(q[3]), "p90": float(q[4])}
        print(f"{k:38s} {v.mean():8.4f} {q[0]:8.4f} {q[1]:8.4f} "
              f"{q[2]:8.4f} {q[3]:8.4f} {q[4]:8.4f}")

    # ── THE DECIDING NUMBERS: at each candidate floor, what share of rows BINDS? ──
    # A floor at `f` binds on a row iff the target's mass on SF's top-1 is below `f`.
    # "Binds on ~0% of rows" is the #447 `listed_mass_min: 0.10` failure -- a
    # guaranteed-null day-plus window. "Binds on ~100%" is a near-total rewrite of the
    # policy target, which is not a floor, it is a replacement.
    m1 = cat.get("mass_on_regret_top1")
    m2 = cat.get("mass_on_regret_top2")
    if m1 is None or m2 is None:
        raise SystemExit("no usable sf_p0_regret rows -- nothing to decide from")
    print(f"\n{'floor':>6s} | {'% rows binding (top1)':>22s} | "
          f"{'% rows binding (top2)':>22s}")
    binds: dict[str, dict[str, float]] = {}
    for f in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50):
        a, b = float((m1 < f).mean()), float((m2 < f).mean())
        binds[f"{f:.2f}"] = {"regret_top1": a, "regret_top2": b}
        print(f"{f:6.2f} | {a*100:21.2f}% | {b*100:21.2f}%")
    report["bind_rate_by_floor"] = binds
    report["bind_population"] = (
        "rows carrying sf_p0_regret with >=2 SURFACED legal entries; masses are the "
        "stored policy_target's, NOT the net's output"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
