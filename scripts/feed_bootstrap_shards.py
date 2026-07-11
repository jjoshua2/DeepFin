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

Holdout quarantine: a deterministic 1-in-N slice of settled shards (by
source index, so the split is stable regardless of scan order or timing)
routes to --holdout-dir instead of --boot-dir. Those shards are NEVER fed
into the training pool, so live-follow eval can point at --holdout-dir (see
offline_replay_epoch.py --eval-replay-dir) and get a real held-out signal
instead of reading the same frozen oldest-shards slice of the training pool
forever (see docs/experiment_ledger.md, 512x16 bootstrap thread, 2026-07-10).
"""
from __future__ import annotations

import argparse
import errno
import os
import re
import shutil
import sys
import time
from pathlib import Path

_TMP_PREFIX = "._tmp_feed_"
# The trial whose shards entered the pool as BARE names before per-trial
# tagging existed (the seed window + the 2026-07-08 feeds all came from it).
_LEGACY_PRETAG_TRIAL = "train_trial_5fac4_00000_0_lr=0.0003_2026-06-17_22-42-40"


def _idx(path: Path) -> int:
    m = re.search(r"shard_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def _discover_live_dir() -> Path | None:
    """Most recently modified train_trial_*/replay_shards that HAS shards.

    A Tune resume/exploit can create (or touch) a newer empty replay dir
    before any shard lands; ranking by mtime alone would select it and feed
    nothing while the active trial's shards are ignored.
    """
    root = Path("/home/josh/projects/chess/runs/pbt2_small/replay")
    best: tuple[float, Path] | None = None
    for d in root.glob("train_trial_*/replay_shards"):
        if not d.is_dir():
            continue
        try:
            mt = d.stat().st_mtime
            if next(iter(d.glob("shard_*.zarr")), None) is None:
                continue
        except OSError:
            continue
        if best is None or mt > best[0]:
            best = (mt, d)
    return best[1] if best is not None else None


def _trial_tag(live_dir: Path) -> str:
    """Short stable tag identifying the source trial (its dir name hash).

    Published shard names embed it (``shard_NNNNNN.<tag>.zarr``) so shard
    names are unique ACROSS trials: a fresh exploited trial restarts
    numbering at shard_000000, and bare basenames as global IDs would both
    collide with already-fed names and be skipped as "already fed".
    """
    import hashlib

    trial = live_dir.parent.name
    return hashlib.sha1(trial.encode()).hexdigest()[:8]


def _tree_mtime(path: Path) -> float | None:
    """Newest mtime in a shard tree (zarr stores are directories).

    None when the shard vanished mid-scan (live eviction can prune a source
    shard between the glob and this stat) — callers treat that as "skip this
    shard, keep feeding the rest".
    """
    try:
        newest = path.stat().st_mtime
        if path.is_dir():
            for child in path.rglob("*"):
                try:
                    newest = max(newest, child.stat().st_mtime)
                except OSError:
                    continue
    except OSError:
        return None
    return newest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--live-dir",
        type=Path,
        default=None,
        help="Live trial replay_shards dir. Default: auto-discover the most "
             "recently modified runs/pbt2_small/replay/train_trial_*/replay_shards "
             "(a Tune resume/exploit can change the active trial dir).",
    )
    ap.add_argument(
        "--boot-dir",
        type=Path,
        default=Path("/home/josh/projects/chess/data/scaleup_pool_512x16/replay_shards"),
    )
    ap.add_argument(
        "--holdout-dir",
        type=Path,
        default=Path("/home/josh/projects/chess/data/scaleup_pool_512x16/holdout_shards"),
        help="Quarantined shards, never fed into --boot-dir, for live-follow eval "
             "(offline_replay_epoch.py --eval-replay-dir). Auto-created if missing.",
    )
    ap.add_argument(
        "--holdout-every-n",
        type=int,
        default=40,
        help="1-in-N settled shards (by source index, deterministic) route to "
             "--holdout-dir instead of --boot-dir (default 40 = 2.5%%). 0 "
             "disables the holdout split (all settled shards go to --boot-dir, "
             "legacy behavior).",
    )
    ap.add_argument(
        "--max-shards",
        type=int,
        default=0,
        help="If >0, only feed the newest N missing --boot-dir shards (0 = all "
             "missing). Does not limit --holdout-dir feeding.",
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

    live_dir: Path | None = args.live_dir
    if live_dir is None:
        live_dir = _discover_live_dir()
        if live_dir is None:
            print("[feed] ERROR: no train_trial_*/replay_shards found to auto-discover", file=sys.stderr)
            return 2
        print(f"[feed] live dir (auto): {live_dir}")
    boot_dir: Path = args.boot_dir
    holdout_dir: Path = args.holdout_dir
    if not live_dir.is_dir():
        print(f"[feed] ERROR: live dir missing: {live_dir}", file=sys.stderr)
        return 2
    if not boot_dir.is_dir():
        print(f"[feed] ERROR: boot dir missing: {boot_dir}", file=sys.stderr)
        return 2
    holdout_dir.mkdir(parents=True, exist_ok=True)

    # Sweep temp debris from a crashed prior feed (never visible to the glob).
    for d in (boot_dir, holdout_dir):
        for stale in d.glob(f"{_TMP_PREFIX}*"):
            shutil.rmtree(stale, ignore_errors=True) if stale.is_dir() else stale.unlink(missing_ok=True)

    tag = _trial_tag(live_dir)
    boot_names = {p.name for p in boot_dir.glob("shard_*.zarr")}
    boot_idxs = [_idx(p) for p in boot_dir.glob("shard_*.zarr")]
    boot_max = max(boot_idxs) if boot_idxs else -1
    holdout_names = {p.name for p in holdout_dir.glob("shard_*.zarr")}
    holdout_every_n = max(0, int(args.holdout_every_n))

    def _dst_name(src: Path) -> str:
        return f"{src.stem}.{tag}.zarr"

    def _is_holdout(src: Path) -> bool:
        # Deterministic on the SOURCE index so the split is stable regardless
        # of scan order/timing — EXCEPT a shard already sitting in boot_names
        # (fed under the pre-holdout code, or any other path) must never
        # become "holdout": it's already been trained on, so it can no
        # longer serve as held-out eval data no matter what its index is.
        if holdout_every_n <= 0:
            return False
        if _dst_name(src) in boot_names:
            return False
        return _idx(src) % holdout_every_n == 0

    live = sorted(live_dir.glob("shard_*.zarr"), key=_idx)
    # Fed check is by TAGGED name (unique per source trial). Bare-name presence
    # counts as fed ONLY for the legacy trial whose shards populated the pool
    # pre-tag (seed window + 07-08 feeds) — for any OTHER trial a bare
    # collision (fresh trials restart at shard_000000) must NOT mask the new
    # trial's shard, whose tagged destination is distinct. The holdout pool is
    # new (post-tagging), so it has no legacy bare-name history to match.
    bare_compat = live_dir.parent.name == _LEGACY_PRETAG_TRIAL

    def _already_fed(src: Path) -> bool:
        if _is_holdout(src):
            return _dst_name(src) in holdout_names
        return _dst_name(src) in boot_names or (bare_compat and src.name in boot_names)

    # Prefer strictly newer-than-boot-max; also any name not present (safety).
    # boot_max only rate-limits the training-pool watermark — holdout shards
    # are considered regardless of it so a quiet training pool never starves
    # holdout growth.
    missing = [p for p in live if not _already_fed(p) and (_is_holdout(p) or _idx(p) > boot_max)]
    if not missing:
        missing = [p for p in live if not _already_fed(p)]
    now = time.time()
    settle = max(0.0, float(args.settle_seconds))
    if settle > 0:
        settled = []
        for p_ in missing:
            mt = _tree_mtime(p_)
            if mt is not None and now - mt >= settle:
                settled.append(p_)
        skipped_unsettled = len(missing) - len(settled)
        missing = settled
    else:
        skipped_unsettled = 0

    missing_boot = [p for p in missing if not _is_holdout(p)]
    missing_holdout = [p for p in missing if _is_holdout(p)]
    if args.max_shards > 0 and len(missing_boot) > args.max_shards:
        missing_boot = missing_boot[-int(args.max_shards) :]

    if not missing_boot and not missing_holdout:
        print(
            f"[feed] nothing to feed (boot_max={boot_max} boot_n={len(boot_names)} "
            f"holdout_n={len(holdout_names)} live_n={len(live)} unsettled_skipped={skipped_unsettled})"
        )
        return 0

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

    def _is_race(e: OSError) -> bool:
        # Expected races are SKIPS, not failures — the driver runs this under
        # `set -euo pipefail`, and one pruned source shard must not cancel a
        # bootstrap launch that fed everything else:
        #   ENOENT  = live eviction pruned the source after the settle check
        #   EEXIST/ENOTEMPTY = a concurrent feeder won the publish race
        # shutil.copytree (the --copy / cross-device path) aggregates
        # per-child failures into shutil.Error with errno=None — unwrap it and
        # treat an all-ENOENT bundle as the same prune race.
        if isinstance(e, shutil.Error):
            entries = e.args[0] if e.args and isinstance(e.args[0], list) else []
            return bool(entries) and all("No such file" in str(why) for _s, _d, why in entries)
        return e.errno in (errno.ENOENT, errno.EEXIST, errno.ENOTEMPTY)

    def _publish(items: list[Path], dst_dir: Path) -> tuple[int, int, int, int]:
        linked = copied = failed = raced = 0
        same_dev = os.stat(live_dir).st_dev == os.stat(dst_dir).st_dev
        use_hardlink = same_dev and not args.copy
        for src in items:
            dst = dst_dir / _dst_name(src)
            tmp = dst_dir / f"{_TMP_PREFIX}{_dst_name(src)}"
            if dst.exists():
                continue
            try:
                if use_hardlink:
                    # os.link() on a directory fails (EPERM); zarr stores are dirs.
                    if src.is_dir():
                        _link_tree(src, tmp)
                    else:
                        os.link(src, tmp)
                else:
                    if src.is_dir():
                        shutil.copytree(src, tmp)
                    else:
                        shutil.copy2(src, tmp)
                # Atomic publish: the sampler either sees the complete shard or
                # nothing — never a partial tree it would cache at 0 positions.
                os.rename(tmp, dst)
                if use_hardlink:
                    linked += 1
                else:
                    copied += 1
            except OSError as exc:
                if _is_race(exc):
                    raced += 1
                else:
                    failed += 1
                # Clean OUR temp only. Never touch dst: if the rename raced a
                # concurrent feeder that already published this shard, dst is
                # a valid shard the sampler may be reading.
                if tmp.exists():
                    if tmp.is_dir():
                        shutil.rmtree(tmp, ignore_errors=True)
                    else:
                        tmp.unlink(missing_ok=True)
                print(f"[feed] {'RACE-SKIP' if _is_race(exc) else 'FAIL'} {src.name}: {exc}", file=sys.stderr)
        return linked, copied, failed, raced

    boot_failed = holdout_failed = 0
    if missing_boot:
        b_linked, b_copied, boot_failed, b_raced = _publish(missing_boot, boot_dir)
        print(
            f"[feed] boot_max_was={boot_max} fed={b_linked + b_copied} "
            f"(hardlink={b_linked} copy={b_copied} fail={boot_failed} raced={b_raced}) "
            f"range={_idx(missing_boot[0])}..{_idx(missing_boot[-1])} "
            f"boot_n_now={sum(1 for _ in boot_dir.glob('shard_*.zarr'))}"
        )
    if missing_holdout:
        h_linked, h_copied, holdout_failed, h_raced = _publish(missing_holdout, holdout_dir)
        print(
            f"[feed] holdout fed={h_linked + h_copied} "
            f"(hardlink={h_linked} copy={h_copied} fail={holdout_failed} raced={h_raced}) "
            f"range={_idx(missing_holdout[0])}..{_idx(missing_holdout[-1])} "
            f"holdout_n_now={sum(1 for _ in holdout_dir.glob('shard_*.zarr'))}"
        )
    print(f"[feed] unsettled_skipped={skipped_unsettled}")
    return 0 if boot_failed == 0 and holdout_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
