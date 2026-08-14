#!/usr/bin/env python3
"""Join the per-position net audit dump + BT4 cache + deep-SF labels and
compare by position criticality (deep-SF best-vs-2nd gap).

Answers two questions:
  1. Does slicing regret by criticality change the "net is flat" read? In quiet
     positions SF's designated best is near-arbitrary, so top-1 there is noisy;
     this buckets it out.
  2. How does our net compare DIRECTLY to LC0 BT4 (move agreement), not only via
     SF regret — including how much a 3500-Elo net's RAW POLICY itself bleeds in
     quiet positions (a label-noise floor we cannot beat).

Inputs (all keyed by position `key`):
  --net    data/audit_analysis/per_position_277.jsonl  (audit_targets --dump-per-position)
  --bt4    data/lc0/bt4_audit_cache.jsonl              (foreign_net_audit.py)
  --audit  data/audit_set_v1.jsonl                     (for the deep-SF bestmove)

⚑ ``--net`` AND ``--bt4`` must both be PROVENANCE-STAMPED caches, and this script
validates both before it opens any input. Their regret / bucket / move columns are
only meaningful under the policy map and audit ruler that produced them, and the
files record neither in their numbers. It is not a warning: BOTH banked defaults
were contaminated the same way and on the same day (2026-06-26) — the BT4 cache
made every BT4 row wrong, and ``per_position_277.jsonl`` moved mean ``gap_cp``
70.68 -> 1212.03 and the criticality bucket on 94/4000 rows, which re-labels the
BT4 rows too. Guarding one and not the other is worse than guarding neither: the
provenance banner would then vouch for a cross-ruler join.

``--audit`` is deliberately NOT stamped. It is the frozen deep-SF label set, and
it is re-parsed through the CURRENT ruler on every read, so it has no baked-in
ruler that can go stale — it is an input to the ruler, not an output of it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chess_anti_engine.eval.audit import (
    CRITICALITY_BUCKET_NAMES,
    PHASE_NAMES,
    criticality_bucket,
    parse_audit_record,
    wdl_brier,
    wdl_ece,
)
from chess_anti_engine.eval.audit_cache import (
    read_audit_cache_by_key,
    read_audit_cache_stamp,
    stamp_summary,
)

BUCKETS = list(CRITICALITY_BUCKET_NAMES)


def bucket(gap: float) -> str:
    """Bucket name for a precomputed best-vs-2nd cp gap (shared edges)."""
    return CRITICALITY_BUCKET_NAMES[criticality_bucket(gap)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", type=Path, default=Path("data/audit_analysis/per_position_277.jsonl"))
    ap.add_argument("--bt4", type=Path, default=Path("data/lc0/bt4_audit_cache.jsonl"))
    ap.add_argument("--audit", type=Path, default=Path("data/audit_set_v1.jsonl"))
    args = ap.parse_args()

    # FIRST, before any input is opened. BOTH derived caches are validated, and
    # both must be: `--bt4` supplies the BT4 regret and agreement columns, and
    # `--net` supplies six of the eight regret rows AND the criticality bucket
    # every row is filed under — including BT4's. Guarding only one of them is
    # worse than guarding neither, because the banner printed below would then
    # vouch for a cross-ruler join. (`--audit` is NOT stamped: it is the frozen
    # deep-SF label set, re-parsed through the CURRENT ruler on every read, so
    # it carries no baked-in ruler to go stale. It is an input to the ruler,
    # not an output of it.)
    #
    # Both stamps are checked against the CURRENT versions, so agreeing with
    # today implies agreeing with each other — no third cross-check needed.
    net_stamp = read_audit_cache_stamp(args.net)
    stamp = read_audit_cache_stamp(args.bt4)

    net = read_audit_cache_by_key(args.net)
    bt4 = read_audit_cache_by_key(args.bt4)
    deep_best: dict[str, str | None] = {}
    deep_wdl: dict[str, tuple[float, float, float]] = {}
    for ln in args.audit.read_text().splitlines():
        if ln.strip():
            d = json.loads(ln)
            deep_best[d["key"]] = d.get("bestmove")
            pos = parse_audit_record(ln)
            deep_wdl[pos.key] = pos.deep_wdl

    keys = [k for k in net if k in bt4]
    print(f"joined {len(keys)} positions (net∩bt4)")
    print(f"net cache: {args.net} [{stamp_summary(net_stamp, ('producer', 'rows'))}]")
    print(f"bt4 cache: {args.bt4} [{stamp_summary(stamp, ('net', 'rows'))}]")
    print(f"provenance (both): {stamp_summary(stamp)}\n")

    # Per bucket: accumulate regret + agreement counters.
    acc: dict[str, dict[str, float]] = {}

    def a(b: str, k: str, v: float) -> None:
        acc.setdefault(b, {}).__setitem__(k, acc.setdefault(b, {}).get(k, 0.0) + v)

    for key in keys:
        n, t = net[key], bt4[key]
        b = bucket(float(n["gap_cp"]))
        nraw, nsrch = n["cand"]["raw"], n["cand"]["search"]
        a(b, "cnt", 1)
        a(b, "net_raw_exp", nraw["exp"])
        a(b, "net_raw_top1", nraw["top1"])
        a(b, "net_srch_exp", nsrch["exp"])
        a(b, "net_srch_top1", nsrch["top1"])
        a(b, "bt4_exp", t["exp_regret"])
        a(b, "bt4_top1", t["top1_regret"])
        a(b, "sf_exp", n["cand"]["sf_soft"]["exp"])
        a(b, "sf_top1", n["cand"]["sf_soft"]["top1"])
        db = deep_best.get(key)
        a(b, "net_eq_bt4", float(nraw["move"] == t["best_move"]))
        # deep_cnt is the denominator for the *_eq_deep fractions: positions with
        # no deep-SF bestmove must not count as disagreements (that understates
        # agreement). net_eq_bt4 keeps the full cnt (BT4 best is always present).
        a(b, "deep_cnt", float(db is not None))
        if db is not None:
            a(b, "net_eq_deep", float(nraw["move"] == db))
            a(b, "bt4_eq_deep", float(t["best_move"] == db))
            a(b, "srch_eq_deep", float(nsrch["move"] == db))

    hdr = "| metric | " + " | ".join(f"{b} (n={int(acc.get(b, {}).get('cnt', 0))})" for b in BUCKETS) + " | overall |"
    print(hdr)
    print("|" + "---|" * (len(BUCKETS) + 2))

    def mean(b: str, k: str, denom: str = "cnt") -> float:
        d = acc.get(b, {})
        c = d.get(denom, 0)
        return d.get(k, 0.0) / c if c else float("nan")

    def omean(k: str, denom: str = "cnt") -> float:
        num = sum(acc[b].get(k, 0.0) for b in acc)
        den = sum(acc[b].get(denom, 0) for b in acc)
        return num / den if den else float("nan")

    rows = [
        ("net raw E[regret]", "net_raw_exp"), ("net raw top1", "net_raw_top1"),
        ("net+search E[regret]", "net_srch_exp"), ("net+search top1", "net_srch_top1"),
        ("BT4 raw E[regret]", "bt4_exp"), ("BT4 raw top1", "bt4_top1"),
        ("SF-soft E[regret]", "sf_exp"), ("SF-soft top1", "sf_top1"),
    ]
    for label, k in rows:
        cells = " | ".join(f"{mean(b, k):.1f}" for b in BUCKETS)
        print(f"| {label} | {cells} | {omean(k):.1f} |")

    print("\n### Top-1 move agreement (fraction)\n")
    print(hdr)
    print("|" + "---|" * (len(BUCKETS) + 2))
    for label, k, denom in [
        ("net argmax == deep-SF best", "net_eq_deep", "deep_cnt"),
        ("net+search == deep-SF best", "srch_eq_deep", "deep_cnt"),
        ("BT4 argmax == deep-SF best", "bt4_eq_deep", "deep_cnt"),
        ("net argmax == BT4 argmax", "net_eq_bt4", "cnt"),
    ]:
        cells = " | ".join(f"{mean(b, k, denom):.2f}" for b in BUCKETS)
        print(f"| {label} | {cells} | {omean(k, denom):.2f} |")

    # BT4 value-head (WDL) calibration vs deep-SF, overall + per phase. The net's
    # own value calibration lives in the audit_targets report (it isn't dumped
    # per-position); cite that report for the net/SF-native/blend rows.
    print("\n### BT4 WDL calibration vs deep-SF (lower = better)\n")
    by_phase: dict[int, list[str]] = {0: [], 1: [], 2: []}
    preds_all, tgts_all = [], []
    for key in keys:
        if "wdl" not in bt4[key] or key not in deep_wdl:
            continue
        by_phase[net[key]["phase"]].append(key)
        preds_all.append(np.asarray(bt4[key]["wdl"], dtype=np.float64))
        tgts_all.append(np.asarray(deep_wdl[key], dtype=np.float64))
    if not preds_all:
        print("(no positions carry both a BT4 'wdl' and a deep-SF WDL — "
              "skipping BT4 calibration; pass a --bt4 cache produced by "
              "foreign_net_audit.py)")
        return
    preds_all_a = np.stack(preds_all)
    tgts_all_a = np.stack(tgts_all)
    print("| group | BT4 Brier | BT4 ECE | n |")
    print("|---|---|---|---|")
    brier = float(np.mean([wdl_brier(p, t) for p, t in zip(preds_all_a, tgts_all_a, strict=True)]))
    print(f"| overall | {brier:.4f} | {wdl_ece(preds_all_a, tgts_all_a):.4f} | {len(preds_all_a)} |")
    for ph, name in enumerate(PHASE_NAMES):
        ks = [k for k in by_phase[ph] if "wdl" in bt4[k] and k in deep_wdl]
        if not ks:
            continue
        pr = np.stack([np.asarray(bt4[k]["wdl"], dtype=np.float64) for k in ks])
        tg = np.stack([np.asarray(deep_wdl[k], dtype=np.float64) for k in ks])
        b = float(np.mean([wdl_brier(p, t) for p, t in zip(pr, tg, strict=True)]))
        print(f"| {name} | {b:.4f} | {wdl_ece(pr, tg):.4f} | {len(ks)} |")


if __name__ == "__main__":
    main()
