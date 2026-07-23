"""Safe FEN-seed feed step: merge a vetted batch into the live active list.

Companion to blindspot_retire_step.py (which handles removal + probation
automatically). It is called by monitor_fen.sh after the deep-SF harvest gate
and remains usable manually for ledgered one-off batches. It dedupes against
the active pool AND the retired store, writes a NEW versioned list (cache keys
are path-based, so in-place edits would be ignored), and repoints
opening_fen_list_path in the live yaml via the same validated-or-reverted path
the retire step uses. The path is live-reloaded, so no restart is needed.

Usage:
  PYTHONPATH=. python3 scripts/blindspot_feed_step.py --batch data/v4_batch.txt --tag fedv4_ck68
  PYTHONPATH=. python3 scripts/blindspot_feed_step.py --batch ... --dry-run

Retired seeds in the batch are SKIPPED by default (retirement said the net is
aware; re-feeding is probation's job) — pass --allow-retired to force them in,
which also clears them from the retired store so they get a fresh streak.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from scripts.blindspot_retire_step import (
    _current_seed_path,
    _placement,
    _repoint_yaml,
    _write_seed_file,
    dump_retire_state,
    load_retire_state,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="file of new seed lines (retire-list grammar)")
    ap.add_argument("--yaml", default="configs/pbt2_small.yaml")
    ap.add_argument("--state", default="scratchpad/live_read/retire_state.json")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--tag", required=True, help="filename tag, e.g. fedv4_ck68")
    ap.add_argument("--allow-retired", action="store_true",
                    help="feed seeds present in the retired store (clears their retired entry)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from scripts.blindspot_resolution import load_seed_lines

    seed_path = _current_seed_path(args.yaml)
    active = load_seed_lines(seed_path)
    active_keys = {_placement(ln) for ln in active}

    streaks: dict[str, int] = {}
    retired: dict[str, str] = {}
    deep_seen: dict[str, bool] = {}
    if os.path.exists(args.state):
        with open(args.state, encoding="utf-8") as fh:
            streaks, retired, deep_seen = load_retire_state(json.load(fh))

    batch = load_seed_lines(args.batch)
    added: list[str] = []
    skipped_active = skipped_retired = skipped_dup = 0
    seen_new: set[str] = set()
    unretired: list[str] = []
    for ln in batch:
        k = _placement(ln)
        if k in active_keys:
            skipped_active += 1
        elif k in seen_new:
            skipped_dup += 1
        elif k in retired and not args.allow_retired:
            skipped_retired += 1
        else:
            if k in retired:
                unretired.append(k)
            added.append(ln)
            seen_new.add(k)

    print(f"[feed] batch {os.path.basename(args.batch)}: {len(batch)} lines -> "
          f"{len(added)} new, {skipped_active} already active, "
          f"{skipped_retired} retired (skipped), {skipped_dup} intra-batch dups")
    if not added:
        print("[feed] nothing to feed; list unchanged")
        return

    new_lines = list(active) + added
    new_path = os.path.join(args.out_dir, f"blindspot_fens_{args.tag}.txt")
    if os.path.exists(new_path):
        print(f"[feed] ABORT: {new_path} already exists — pick a new --tag "
              "(path-based caches require a fresh filename)", file=sys.stderr)
        sys.exit(1)
    if args.dry_run:
        print(f"[feed] DRY RUN: would write {new_path} "
              f"({len(active)} active + {len(added)} fed) and repoint {args.yaml}")
        return

    note = f"fed {len(added)} seeds from {os.path.basename(args.batch)} onto {os.path.basename(seed_path)}"
    _write_seed_file(new_path, new_lines, note=note)
    from chess_anti_engine.selfplay.opening import _load_fen_list
    n_loaded = len(_load_fen_list(new_path))
    if n_loaded < len(active):
        print(f"[feed] ABORT: new list loads only {n_loaded} usable seeds "
              f"(< {len(active)} currently active) — not repointing", file=sys.stderr)
        sys.exit(1)
    if not _repoint_yaml(args.yaml, os.path.abspath(new_path)):
        sys.exit(1)
    if unretired:
        # Re-load just before writing: the retire step runs on cadence against
        # the same file, and a stale read-modify-write here would silently drop
        # any streaks/retirements it recorded since our load at startup.
        if os.path.exists(args.state):
            with open(args.state, encoding="utf-8") as fh:
                streaks, retired, deep_seen = load_retire_state(json.load(fh))
        for k in unretired:
            retired.pop(k, None)
            streaks[k] = 0
            deep_seen[k] = False
        os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)
        with open(args.state, "w", encoding="utf-8") as fh:
            json.dump(dump_retire_state(streaks, retired, deep_seen), fh)
    print(f"[feed] LIVE: {new_path} ({n_loaded} seeds), yaml repointed; "
          f"{len(unretired)} cleared from retired store")


if __name__ == "__main__":
    main()
