"""Offline v1 -> v2_threats replay-shard converter.

Rewrites 146-plane replay shards to 175 planes by recomputing the 29
threat planes from the stored input planes (no FENs needed — see
chess_anti_engine/encoding/plane_decode.py). Every shard is validated:
the 34 recomputed v1 extra planes must match the stored ones, so a
decode problem aborts that shard instead of writing corrupt planes.
Already-175-plane shards are skipped, so reruns are idempotent.

Usage:
  PYTHONPATH=. python3 scripts/convert_shards_v2_threats.py DIR [DIR|SHARD.zarr ...] \
      [--out OUTDIR] [--workers N] [--history-encoding ENC] [--dry-run]

Default is in-place conversion (atomic tmp+rename per shard, same as the
production writer) — run it against a replay dir while that run is
stopped, or against a salvage pool's seeds/slot_NNN/replay_shards dirs.
With --out, converted shards are written there and inputs are untouched.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

import zarr

from chess_anti_engine.replay.shard import (
    ShardMeta,
    iter_shard_paths,
    load_shard_arrays,
    save_local_shard_arrays,
)
from chess_anti_engine.replay.threat_upgrade import (
    V2_INPUT_PLANES,
    upgrade_arrays_to_v2_threats,
)

_META_FIELDS = frozenset(f.name for f in fields(ShardMeta))


def _collect_shards(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.suffix == ".zarr":
            if not p.exists():
                raise FileNotFoundError(p)
            paths.append(p)
        elif p.is_dir():
            paths.extend(iter_shard_paths(p))
        else:
            raise FileNotFoundError(f"{p} is not a directory or .zarr shard")
    # de-dup while preserving order
    seen: set[Path] = set()
    out = [p for p in paths if not (p in seen or seen.add(p))]
    if not out:
        raise FileNotFoundError(f"no shard_*.zarr found under {inputs}")
    return out


def _stored_planes(path: Path) -> int:
    g = zarr.open_group(str(path), mode="r")
    return int(g["x"].shape[1])  # pyright: ignore[reportArgumentType]


def convert_shard(
    path: Path,
    *,
    out_path: Path | None = None,
    history_encoding: str | None = None,
) -> dict[str, object]:
    """Convert one shard; returns a summary row for the run report."""
    dst = out_path if out_path is not None else path
    try:
        if _stored_planes(path) == V2_INPUT_PLANES:
            if out_path is not None and not out_path.exists():
                arrs, meta = load_shard_arrays(path)
                save_local_shard_arrays(dst, arrs=arrs, meta=_filter_meta(meta))
            return {"path": str(path), "status": "skipped", "rows": 0}
        arrs, meta = load_shard_arrays(path)
        upgraded, stats = upgrade_arrays_to_v2_threats(
            arrs, history_encoding=history_encoding,
        )
        save_local_shard_arrays(dst, arrs=upgraded, meta=_filter_meta(meta))
        return {
            "path": str(path),
            "status": "converted",
            "rows": stats.upgraded_rows,
            "dropout_rows": stats.dropout_rows,
        }
    except Exception as exc:  # surfaced per-shard in the summary
        return {"path": str(path), "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _filter_meta(meta: dict[str, object]) -> dict[str, object]:
    # _meta_dict round-trips attrs through ShardMeta(**meta); drop any
    # foreign attr keys so an annotated shard can't crash the rewrite.
    return {k: v for k, v in meta.items() if k in _META_FIELDS}


def _run_one(args: tuple[Path, Path | None, str | None]) -> dict[str, object]:
    path, out_path, history_encoding = args
    return convert_shard(path, out_path=out_path, history_encoding=history_encoding)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="replay shard dirs and/or .zarr shard paths")
    parser.add_argument("--out", type=Path, default=None,
                        help="write converted shards here instead of in place")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel converter processes (default 4)")
    parser.add_argument("--history-encoding", default=None,
                        help="override the shard's input_history_encoding attr "
                             "(legacy / lc0_root / lc0_root_legacy_meta)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be converted and exit")
    args = parser.parse_args(argv)

    shards = _collect_shards(args.inputs)
    if args.out is not None:
        # Distinct inputs (e.g. several replay dirs) reuse basenames like
        # shard_000000.zarr; flattening them into one --out dir would
        # silently overwrite earlier conversions. Refuse instead.
        dupes = [name for name, k in Counter(p.name for p in shards).items() if k > 1]
        if dupes:
            parser.error(
                f"--out would merge {len(shards)} shards with duplicate names "
                f"({', '.join(sorted(dupes)[:3])}{', ...' if len(dupes) > 3 else ''}); "
                f"convert one input directory per --out run"
            )
        args.out.mkdir(parents=True, exist_ok=True)
    tasks = [
        (p, (args.out / p.name) if args.out is not None else None, args.history_encoding)
        for p in shards
    ]
    if args.dry_run:
        for p in shards:
            planes = _stored_planes(p)
            verb = "skip (already v2)" if planes == V2_INPUT_PLANES else "convert"
            print(f"{verb}: {p} ({planes} planes)")
        return 0

    results: list[dict[str, object]] = []
    if args.workers <= 1:
        results = [_run_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_run_one, tasks))

    converted = [r for r in results if r["status"] == "converted"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]
    rows = sum(int(r["rows"]) for r in converted)  # pyright: ignore[reportArgumentType]
    dropout = sum(int(r.get("dropout_rows", 0)) for r in converted)  # pyright: ignore[reportArgumentType]
    print(
        f"[convert] {len(converted)} shard(s) converted ({rows} rows, "
        f"{dropout} dropout-zeroed), {len(skipped)} already v2, {len(errors)} failed"
    )
    for r in errors:
        print(f"[convert] FAILED {r['path']}: {r['error']}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
