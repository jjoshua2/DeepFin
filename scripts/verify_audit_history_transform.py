#!/usr/bin/env python3
"""Verify `pov_flip_slot` bit-exactly against REAL consecutive-ply stored rows.

The audit-v2 value ruler (`--input-encoding stored`) cannot read a stored row
for a 1-ply child — the child position was never a stored training row. It
DERIVES the child's history from the parent's by shifting every slot down one
and re-expressing it from the child's point of view (`pov_flip_slot`: swap the
us/them 6-plane halves, mirror the rank axis).

A derived input that is subtly wrong is exactly this codebase's signature
defect — a value accepted and then silently misused — so the transform is not
asserted, it is MEASURED. This script finds pairs of stored rows that are the
same game at consecutive plies, applies the transform to the parent, and
compares bit-for-bit with the child that production actually wrote:

    child slot k+1  ==  pov_flip_slot(parent slot k)   for k in 0..6
    child colour flag (plane 108)  ==  1 - parent colour flag

The 2026-08-02 full-snapshot run (all 1463 shards):

    pairs 396733 | piece-history slots 2777131/2777131 exact
                 | colour flags 396733/396733 exact
                 | repetition planes 2776608/2777131 (523 misses, 0.019% —
                   repetitions older than the parent's 8-slot window, which no
                   shift can know; this is the transform's one known residual
                   and it is measured rather than assumed away)

`--emit-fixture` writes a small sample of verified pairs for
`tests/test_audit_history_encoding.py`, so the invariant stays pinned in CI
without the test ever reading `runs/`.

Usage:

    PYTHONPATH=. python3 scripts/verify_audit_history_transform.py \\
        --snapshot runs/parallel_candidate_replay_snapshots/\\
current_live_20260602_202037_v2threats \\
        --emit-fixture tests/data/audit_history_pairs.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.encoding.lc0 import normalize_lc0_history_encoding
from chess_anti_engine.eval.audit_history import (
    STORED_HISTORY_ENCODING,
    STORED_PLANES,
    verify_child_transform,
)
from chess_anti_engine.replay.shard import load_shard_arrays


def find_consecutive_pairs(group: dict[str, Any]) -> list[tuple[int, int]]:
    """(parent_row, child_row) indices of same-game consecutive-ply rows.

    Within one shard only. A game split across shards contributes no pair at
    the seam, so this UNDER-counts — which is fine, because the output is a
    verification sample and not a census, and the sample is already ~400k pairs.
    Only exact ply+1 steps count: selfplay drops playout-capped plies, so a gap
    of 2 is a DIFFERENT position whose history does not shift by one.
    """
    if not all(k in group for k in ("x", "game_id", "ply_index")):
        return []
    game = np.asarray(group["game_id"][:]).reshape(-1)
    ply = np.asarray(group["ply_index"][:]).reshape(-1)
    by_game: dict[int, dict[int, int]] = {}
    for row, (g, p) in enumerate(zip(game.tolist(), ply.tolist(), strict=True)):
        by_game.setdefault(int(g), {})[int(p)] = row
    pairs: list[tuple[int, int]] = []
    for rows_by_ply in by_game.values():
        for p, row in sorted(rows_by_ply.items()):
            nxt = rows_by_ply.get(p + 1)
            if nxt is not None:
                pairs.append((row, nxt))
    return pairs


def _emit_fixture(
    refs: list[tuple[int, Path, int, int, int]],
    out: Path,
    n_pairs: int,
    snapshot: Path,
) -> None:
    """Write `n_pairs` verified pairs spread evenly over the ply distribution."""
    ordered = sorted(refs)
    take = min(n_pairs, len(ordered))
    picks = [ordered[round(i * (len(ordered) - 1) / max(1, take - 1))] for i in range(take)]
    parents: list[np.ndarray] = []
    children: list[np.ndarray] = []
    provenance: list[str] = []
    for path in sorted({p[1] for p in picks}):
        group, _meta = load_shard_arrays(path, lazy=True)
        x = np.asarray(group["x"][:], dtype=np.float16)
        for parent_ply, ref_path, parent_row, child_row, game_id in picks:
            if ref_path != path:
                continue
            parents.append(x[parent_row])
            children.append(x[child_row])
            provenance.append(
                f"{path.name}:{parent_row}->{child_row} game={game_id} ply={parent_ply}"
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        parent=np.stack(parents),
        child=np.stack(children),
        provenance=np.array(provenance),
        snapshot=np.array([str(snapshot)]),
        input_history_encoding=np.array([STORED_HISTORY_ENCODING]),
    )
    plies = sorted(p[0] for p in picks)
    print(f"[verify] fixture ({len(parents)} pairs, parent plies {plies}) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--max-shards", type=int, default=0,
                    help="0 (default) = every shard in the snapshot")
    ap.add_argument("--emit-fixture", type=Path, default=None,
                    help="write the first --fixture-pairs verified pairs for the test")
    ap.add_argument("--fixture-pairs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None,
                    help="write the verification report as JSON")
    args = ap.parse_args()

    shards = sorted(args.snapshot.glob("*.zarr"))
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    if not shards:
        raise SystemExit(f"no *.zarr shards under {args.snapshot}")

    # Counters only — a full-snapshot scan is hundreds of thousands of pairs
    # and holding their planes would be tens of GB. Verify per shard, sum.
    report: dict[str, int] = {}
    # (parent ply, shard, parent row, child row, game id) for every verified
    # pair, so the fixture can be sampled across the PLY distribution rather
    # than taking whatever the first shard happened to hold. Early plies carry
    # partially-empty history and are the case a repeat-style fill would pass
    # and a shift would not, so they must be represented.
    refs: list[tuple[int, Path, int, int, int]] = []
    skipped: dict[str, int] = {}
    for si, path in enumerate(shards):
        group, _meta = load_shard_arrays(path, lazy=True)
        if "x" not in group:
            skipped["no-x"] = skipped.get("no-x", 0) + 1
            continue
        hist = "legacy"
        if "_input_history_encoding" in group:
            hist = normalize_lc0_history_encoding(
                str(np.asarray(group["_input_history_encoding"]).reshape(-1)[0])
            )
        arr = group["x"]
        if hist != STORED_HISTORY_ENCODING or int(arr.shape[1]) != STORED_PLANES:
            tag = f"{hist}/{int(arr.shape[1])}p"
            skipped[tag] = skipped.get(tag, 0) + 1
            continue
        rows = find_consecutive_pairs(group)
        if not rows:
            continue
        x = np.asarray(arr[:], dtype=np.float32)
        game = np.asarray(group["game_id"][:]).reshape(-1)
        ply = np.asarray(group["ply_index"][:]).reshape(-1)
        for key, value in verify_child_transform(
            [(x[parent_row], x[child_row]) for parent_row, child_row in rows]
        ).items():
            report[key] = report.get(key, 0) + value
        refs.extend(
            (int(ply[parent_row]), path, parent_row, child_row, int(game[parent_row]))
            for parent_row, child_row in rows
        )
        if (si + 1) % 200 == 0 or si + 1 == len(shards):
            print(f"[scan] {si + 1}/{len(shards)} shards, {report.get('pairs', 0)} pairs",
                  flush=True)

    if not refs:
        raise SystemExit("no consecutive-ply pairs found — nothing to verify")

    report["shards_scanned"] = len(shards)
    exact = (
        report["piece_slots_exact"] == report["piece_slots_total"]
        and report["stm_flag_exact"] == report["stm_flag_total"]
    )
    print(json.dumps({**report, "shards_skipped_wrong_layout": skipped}, indent=1))
    print(f"[verify] piece-history slots and colour flags EXACT: {exact}")
    print(
        f"[verify] repetition planes "
        f"{report['repetition_planes_exact']}/{report['repetition_planes_total']} "
        "(misses are repetitions older than the parent's 8-slot window)"
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    if args.emit_fixture is not None:
        if not exact:
            raise SystemExit("refusing to emit a fixture from a FAILING verification")
        _emit_fixture(refs, args.emit_fixture, int(args.fixture_pairs), args.snapshot)

    if not exact:
        raise SystemExit("pov_flip_slot is NOT bit-exact on real stored pairs")


if __name__ == "__main__":
    main()
