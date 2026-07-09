#!/usr/bin/env python3
"""Hardlink (or copy) new live replay shards into the 512x16 bootstrap pool.

The bootstrap live-follow sampler rescans its --replay-dir and credits
``target_reuse`` samples per *newly seen* shard path. Feeding new zarrs into
that directory mid-run extends phase-1 credit; re-feeding at phase 2/3 handoff
gives the LR-drop phases fresher data too.

Default: hardlink every live shard with index > bootstrap max (same filesystem).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path


def _idx(path: Path) -> int:
    m = re.search(r"shard_(\d+)", path.name)
    return int(m.group(1)) if m else -1


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
        default=Path(
            "/home/josh/projects/chess/data/salvage/scaleup_512x16_window_20260707/"
            "seeds/slot_000/replay_shards"
        ),
    )
    ap.add_argument(
        "--max-shards",
        type=int,
        default=0,
        help="If >0, only feed the newest N missing shards (0 = all missing).",
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

    boot_names = {p.name for p in boot_dir.glob("shard_*.zarr")}
    boot_idxs = [_idx(p) for p in boot_dir.glob("shard_*.zarr")]
    boot_max = max(boot_idxs) if boot_idxs else -1

    live = sorted(live_dir.glob("shard_*.zarr"), key=_idx)
    # Prefer strictly newer-than-boot-max; also any name not present (safety).
    missing = [p for p in live if p.name not in boot_names and _idx(p) > boot_max]
    if not missing:
        missing = [p for p in live if p.name not in boot_names]
    if args.max_shards > 0 and len(missing) > args.max_shards:
        missing = missing[-int(args.max_shards) :]

    if not missing:
        print(f"[feed] nothing to feed (boot_max={boot_max} boot_n={len(boot_names)} live_n={len(live)})")
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
        if dst.exists():
            continue
        try:
            if use_hardlink:
                # os.link() on a directory fails (EPERM); zarr stores are dirs.
                if src.is_dir():
                    _link_tree(src, dst)
                else:
                    os.link(src, dst)
                linked += 1
            else:
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
                copied += 1
        except OSError as exc:
            failed += 1
            # Clean partial dest so a retry is clean.
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst, ignore_errors=True)
                else:
                    dst.unlink(missing_ok=True)
            print(f"[feed] FAIL {src.name}: {exc}", file=sys.stderr)

    print(
        f"[feed] boot_max_was={boot_max} fed={linked + copied} "
        f"(hardlink={linked} copy={copied} fail={failed}) "
        f"range={_idx(missing[0])}..{_idx(missing[-1])} "
        f"boot_n_now={sum(1 for _ in boot_dir.glob('shard_*.zarr'))}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
