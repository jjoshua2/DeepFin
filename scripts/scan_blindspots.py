#!/usr/bin/env python3
"""Blind-spot harvest calibration scanner (offline, read-only).

Sweeps value-blindness thresholds over existing replay shards and reports, per
``(net_ok, sf_lost)`` band, how many positions the inline harvester WOULD flag:
volume (count, %, est./iter), where they come from (selfplay vs curriculum),
how blind they are (severity gap), and how diverse (unique placements). This is
the Level-1 calibration for the harvester — it sets the severe/broad threshold
boundary and the sustainable seeding dose from real numbers, with zero training
and no live risk.

A position is value-blind at ``(net_ok, sf_lost)`` when the net's own search
value says fine (``net_q = search_wdl[W]-search_wdl[L] > net_ok``) while the
in-loop SF eval says lost (``sf_q = sf_wdl[W]-sf_wdl[L] < sf_lost``). Both are
side-to-move POV. This is the same signal the harvester will use at finalize;
here it only counts (no move history / seed emission — that is the worker-side
harvester's job).

Usage:
  PYTHONPATH=. python3 scripts/scan_blindspots.py            # newest trial, recent shards
  PYTHONPATH=. python3 scripts/scan_blindspots.py --shards 24 \
      --net-ok 0.2,0.35,0.5 --sf-lost -0.3,-0.5,-0.7
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chess_anti_engine.replay.shard import load_shard_arrays, shard_index
from scripts.diagnostic_replay_utils import select_shards

DEFAULT_POSITIONS_PER_ITER = 13500  # ~live ingest/iter; only scales the /iter estimate


@dataclass(frozen=True)
class BandStat:
    net_ok: float
    sf_lost: float
    count: int
    frac_of_evald: float
    curriculum_frac: float
    severity_median: float
    severity_p90: float
    unique_frac: float


def band_stat(
    net_q: np.ndarray, sf_q: np.ndarray, is_selfplay: np.ndarray,
    pos_key: np.ndarray | None, *, net_ok: float, sf_lost: float,
) -> BandStat:
    """Pure per-band statistics over already-masked (finite-eval) arrays."""
    blind = (net_q > net_ok) & (sf_q < sf_lost)
    n = int(blind.sum())
    total = int(net_q.size)
    if n == 0:
        return BandStat(net_ok, sf_lost, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sev = (net_q - sf_q)[blind]  # how far apart net and SF are (max 2.0)
    cur = float((~is_selfplay[blind]).mean())
    uniq = float(len(np.unique(pos_key[blind])) / n) if pos_key is not None else float("nan")
    return BandStat(
        net_ok=net_ok, sf_lost=sf_lost, count=n,
        frac_of_evald=n / max(1, total), curriculum_frac=cur,
        severity_median=float(np.median(sev)), severity_p90=float(np.percentile(sev, 90)),
        unique_frac=uniq,
    )


def _q(wdl: np.ndarray) -> np.ndarray:
    return wdl[:, 0] - wdl[:, 2]


def _position_key(x: np.ndarray) -> np.ndarray:
    """Cheap placement identity per position: hash the step-0 piece planes
    ([0:12] are the current board, identical across history encodings). Ignores
    castling/ep, so it slightly OVER-merges — a conservative diversity floor."""
    planes = np.ascontiguousarray(x[:, :12] > 0.5)
    flat = planes.reshape(planes.shape[0], -1)
    # 64-bit FNV-ish fold over packed bits, vectorized.
    packed = np.packbits(flat, axis=1).astype(np.uint64)
    key = np.zeros(packed.shape[0], dtype=np.uint64)
    for col in range(packed.shape[1]):
        key = (key * np.uint64(1099511628211)) ^ packed[:, col]
    return key


def load_scan_arrays(shards: list[str], *, want_keys: bool) -> tuple[dict[str, np.ndarray], int]:
    """Returns (eval'd-row arrays, total rows scanned). ``total`` counts ALL
    rows (not just eval'd), so the /iter estimate divides by true ingest."""
    nq, sq, isp, keys = [], [], [], []
    total_rows = 0
    for sh in shards:
        # lazy: read only the datasets we touch (x / policy-width targets are
        # multi-GB on live shards; --no-diversity then never loads x). Per-shard
        # try/except: the tool runs beside a live writer that rmtree+renames
        # shard dirs during compaction, so a mid-rename shard is skipped, not
        # fatal — matching the production ingest path.
        try:
            a, _ = load_shard_arrays(sh, lazy=True)
            has_search, has_sf = a.get("has_search_wdl"), a.get("has_sf_wdl")
            if has_search is None or has_sf is None:
                continue
            has_search = np.asarray(has_search).astype(bool)
            total_rows += int(has_search.size)
            has = has_search & np.asarray(has_sf).astype(bool)
            if not has.any():
                continue
            nq.append(_q(np.asarray(a["search_wdl"], np.float32))[has])
            sq.append(_q(np.asarray(a["sf_wdl"], np.float32))[has])
            # is_selfplay is optional (legacy shards may lack it, or tag only
            # some rows); trust it via has_is_selfplay, else default untagged/
            # absent to selfplay (~95% majority) rather than KeyError/miscount.
            n = int(has.sum())
            raw_isp = a.get("is_selfplay")
            raw_has_isp = a.get("has_is_selfplay")
            if raw_isp is None:
                isp.append(np.ones(n, dtype=bool))
            else:
                v = np.asarray(raw_isp).astype(bool)
                if raw_has_isp is not None:
                    v = v | ~np.asarray(raw_has_isp).astype(bool)  # untagged -> selfplay
                isp.append(v[has])
            if want_keys:
                # Pass the lazy zarr; _position_key slices [:, :12] so only 12 of
                # 175 planes are read (not the full float32-upcast tensor).
                keys.append(_position_key(a["x"])[has])
        except (FileNotFoundError, KeyError, OSError):
            continue
    if not nq:
        raise SystemExit("no positions with both search_wdl and sf_wdl in the scanned shards")
    out = {
        "net_q": np.concatenate(nq), "sf_q": np.concatenate(sq),
        "is_selfplay": np.concatenate(isp),
    }
    if want_keys:
        out["pos_key"] = np.concatenate(keys)
    return out, total_rows


def _max_shard_index(replay_dir: str) -> int:
    shards = glob.glob(os.path.join(replay_dir, "shard_*.zarr"))
    return max((shard_index(s) for s in shards), default=-1)


def default_replay_dir() -> str:
    work = os.environ.get("TRAIN_WORK_DIR", "runs/pbt2_small")
    # Two production layouts: the exchange root ($WORK/replay/train_trial_*) and
    # the trial-local dir ($WORK/tune/train_trial_*/replay_shards) used when
    # tune_replay_root_override is unset (e.g. configs/default.yaml).
    cands = (glob.glob(os.path.join(work, "replay", "train_trial_*", "replay_shards"))
             + glob.glob(os.path.join(work, "tune", "train_trial_*", "replay_shards")))
    cands = [d for d in cands if _max_shard_index(d) >= 0]
    if not cands:
        raise SystemExit(
            f"no replay_shards under {work}/replay/train_trial_*/ or "
            f"{work}/tune/train_trial_*/")
    # Pick by highest shard index (append order), not dir mtime — copy/restore/
    # touch scramble mtimes, the same reason main() selects shards by index.
    return max(cands, key=_max_shard_index)


def _floats(spec: str) -> list[float]:
    return [float(t) for t in spec.split(",") if t.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay-dir", default=None, help="default: newest live trial's replay_shards")
    ap.add_argument("--shards", type=int, default=16, help="most-recent N shards to scan (default 16)")
    ap.add_argument("--net-ok", default="0.2,0.35,0.5", help="net_q thresholds (net thinks fine above)")
    ap.add_argument("--sf-lost", default="-0.3,-0.5,-0.7", help="sf_q thresholds (SF says lost below)")
    ap.add_argument("--positions-per-iter", type=int, default=DEFAULT_POSITIONS_PER_ITER)
    ap.add_argument("--no-diversity", action="store_true", help="skip unique-placement (faster; no x load)")
    args = ap.parse_args()
    if args.shards < 1:
        raise SystemExit("--shards must be >= 1")

    rdir = args.replay_dir or default_replay_dir()
    # select_shards sorts by numeric shard index (append order), not dir mtime —
    # copy/restore/touch scramble mtimes; the buffer orders by shard_NNNNNN.
    shards = [str(p) for p in select_shards(Path(rdir), args.shards)]
    if not shards:
        raise SystemExit(f"no shards under {rdir}")
    arr, total_rows = load_scan_arrays(shards, want_keys=not args.no_diversity)
    nq, sq, isp = arr["net_q"], arr["sf_q"], arr["is_selfplay"]
    key = arr.get("pos_key")
    n_eval = int(nq.size)
    # /iter divides by TOTAL ingested rows (not eval-only) — else a window where
    # SF labels cover a fraction f of rows would inflate /iter by ~1/f.
    iters = total_rows / max(1, args.positions_per_iter)
    show_uniq = key is not None

    print(f"[scan] {len(shards)} shards, {total_rows} rows ({n_eval} with both evals) "
          f"(~{iters:.1f} iters @ {args.positions_per_iter}/iter); "
          f"selfplay {100 * isp.mean():.1f}% / curriculum {100 * (~isp).mean():.1f}%\n")
    header = (f"{'net_q>':>6} {'sf_q<':>6} {'count':>7} {'%eval':>6} {'/iter':>7} "
              f"{'cur%':>5} {'sevMed':>7} {'sevP90':>7}")
    print(header + (f" {'uniq%':>6}" if show_uniq else ""))
    print("-" * (len(header) + (7 if show_uniq else 0)))
    for net_ok in _floats(args.net_ok):
        for sf_lost in _floats(args.sf_lost):
            s = band_stat(nq, sq, isp, key, net_ok=net_ok, sf_lost=sf_lost)
            row = (f"{s.net_ok:>6.2f} {s.sf_lost:>6.2f} {s.count:>7d} "
                   f"{100 * s.frac_of_evald:>5.2f}% {s.count / max(iters, 1e-9):>7.1f} "
                   f"{100 * s.curriculum_frac:>4.0f}% {s.severity_median:>7.2f} "
                   f"{s.severity_p90:>7.2f}")
            print(row + (f" {100 * s.unique_frac:>5.0f}%" if show_uniq else ""))
    print("\nseverity = net_q - sf_q (0..2, higher = blinder); /iter estimate scales "
          "with --positions-per-iter" + ("; uniq% = distinct placements among the band "
          "(diversity for sustaining a high seeding dose)." if show_uniq else "."))


if __name__ == "__main__":
    main()
