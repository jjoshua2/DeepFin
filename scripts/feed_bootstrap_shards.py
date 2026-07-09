#!/usr/bin/env python3
"""Hardlink (or copy) new live replay shards into the 512x16 bootstrap pool.

The bootstrap live-follow sampler rescans its --replay-dir and credits
``target_reuse`` samples per *newly seen* shard path. Feeding new zarrs into
that directory mid-run extends credit and keeps the pool tracking the live
data distribution (the swap gate compares against the CURRENT live net).

Atomicity contract: the sampler rescans between training steps and caches a
shard's position count PERMANENTLY the first time it sees the path — a
partially-linked zarr directory would be cached at 0 positions forever. So
each shard is built under a ``._tmp_feed_`` name (which the ``shard_*.zarr``
glob never matches) and atomically renamed into place only when complete.

Source settling: a live shard the ingest process is still writing has the
mirror-image problem, so shards modified within --settle-seconds are skipped
(they're picked up on the next feed pass).

Default: hardlink every settled live shard with index > pool max (same
filesystem; ~zero disk). The default pool is the DEDICATED bootstrap pool —
NOT the salvage pool, which is a ledgered frozen revert point.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

_TMP_PREFIX = "._tmp_feed_"


def _idx(path: Path) -> int:
    m = re.search(r"shard_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def _tree_mtime(path: Path) -> float:
    """Newest mtime in a shard tree (zarr stores are directories)."""
    newest = path.stat().st_mtime
    if path.is_dir():
        for child in path.rglob("*"):
            try:
                newest = max(newest, child.stat().st_mtime)
            except OSError:
                continue
    return newest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--live-dir",
        type=Path,
        default=Path(
            "/home/josh/projects/chess/runs/pbt2_small/replay/"
            "train_trial_5fac4_00000_0_lr=0.0003_2026-06-17_22-42-40/replay_shards"
        ),
    )
    ap.add_argument(
        "--boot-dir",
        type=Path,
        default=Path("/home/josh/projects/chess/data/scaleup_pool_512x16/replay_shards"),
    )
    ap.add_argument(
        "--max-shards",
        type=int,
        default=0,
        help="If >0, only feed the newest N missing shards (0 = all missing).",
    )
    ap.add_argument(
        "--settle-seconds",
        type=float,
        default=120.0,
        help="Skip live shards whose tree was modified this recently (in-flight ingest writes).",
    )
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of hardlink (default hardlink when same device).",
    )
    args = ap.parse_args()

    live_dir: Path = args.live_dir
    boot_dir: Path = args.boot_dir
    if not live_dir.is_dir():
        print(f"[feed] ERROR: live dir missing: {live_dir}", file=sys.stderr)
        return 2
    if not boot_dir.is_dir():
        print(f"[feed] ERROR: boot dir missing: {boot_dir}", file=sys.stderr)
        return 2

    # Sweep temp debris from a crashed prior feed (never visible to the glob).
    for stale in boot_dir.glob(f"{_TMP_PREFIX}*"):
        shutil.rmtree(stale, ignore_errors=True) if stale.is_dir() else stale.unlink(missing_ok=True)

    boot_names = {p.name for p in boot_dir.glob("shard_*.zarr")}
    boot_idxs = [_idx(p) for p in boot_dir.glob("shard_*.zarr")]
    boot_max = max(boot_idxs) if boot_idxs else -1

    live = sorted(live_dir.glob("shard_*.zarr"), key=_idx)
    # Prefer strictly newer-than-boot-max; also any name not present (safety).
    missing = [p for p in live if p.name not in boot_names and _idx(p) > boot_max]
    if not missing:
        missing = [p for p in live if p.name not in boot_names]
    now = time.time()
    settle = max(0.0, float(args.settle_seconds))
    if settle > 0:
        settled = [p for p in missing if now - _tree_mtime(p) >= settle]
        skipped_unsettled = len(missing) - len(settled)
        missing = settled
    else:
        skipped_unsettled = 0
    if args.max_shards > 0 and len(missing) > args.max_shards:
        missing = missing[-int(args.max_shards) :]

    if not missing:
        print(
            f"[feed] nothing to feed (boot_max={boot_max} boot_n={len(boot_names)} "
            f"live_n={len(live)} unsettled_skipped={skipped_unsettled})"
        )
        return 0

    same_dev = os.stat(live_dir).st_dev == os.stat(boot_dir).st_dev
    use_hardlink = same_dev and not args.copy
    linked = 0
    copied = 0
    failed = 0

    def _link_tree(src: Path, dst: Path) -> None:
        """Recursively hardlink a file/dir tree (zarr shards are directories)."""
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for child in src.iterdir():
                _link_tree(child, dst / child.name)
        else:
            if dst.exists():
                return
            os.link(src, dst)

    for src in missing:
        dst = boot_dir / src.name
        tmp = boot_dir / f"{_TMP_PREFIX}{src.name}"
        if dst.exists():
            continue
        try:
            if use_hardlink:
                # os.link() on a directory fails (EPERM); zarr stores are dirs.
                if src.is_dir():
                    _link_tree(src, tmp)
                else:
                    os.link(src, tmp)
                linked += 1
            else:
                if src.is_dir():
                    shutil.copytree(src, tmp)
                else:
                    shutil.copy2(src, tmp)
                copied += 1
            # Atomic publish: the sampler either sees the complete shard or
            # nothing — never a partial tree it would cache at 0 positions.
            os.rename(tmp, dst)
        except OSError as exc:
            failed += 1
            for leftover in (tmp, dst):
                if leftover.exists():
                    if leftover.is_dir():
                        shutil.rmtree(leftover, ignore_errors=True)
                    else:
                        leftover.unlink(missing_ok=True)
            print(f"[feed] FAIL {src.name}: {exc}", file=sys.stderr)

    print(
        f"[feed] boot_max_was={boot_max} fed={linked + copied} "
        f"(hardlink={linked} copy={copied} fail={failed} unsettled_skipped={skipped_unsettled}) "
        f"range={_idx(missing[0])}..{_idx(missing[-1])} "
        f"boot_n_now={sum(1 for _ in boot_dir.glob('shard_*.zarr'))}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
