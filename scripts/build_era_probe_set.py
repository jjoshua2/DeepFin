#!/usr/bin/env python3
"""Build a FROZEN row set for the per-iteration era-forgetting probe.

The 2026-07-31 run lost 48.6 Elo over three weeks behind flat columns. The
signal that WAS there is the forgetting hinge: a fixed set of old-era rows gets
steadily worse once its content leaves the effective replay window, while
in-window rows keep improving. ``chess_anti_engine/eval/era_probe.py`` scores
that pair every iteration; this script cuts the sets it scores.

    # the ERA leg: old rows, frozen for the life of the run
    PYTHONPATH=. python3 scripts/build_era_probe_set.py \
        --shard-dir runs/pbt2_small/tune/<trial>/replay_shards \
        --oldest 40 --rows 2048 --label era \
        --out data/era_probe/era_20260803.npz

    # the IN-WINDOW twin: newest shards, re-cut at each restart
    PYTHONPATH=. python3 scripts/build_era_probe_set.py \
        --shard-dir runs/pbt2_small/tune/<trial>/replay_shards \
        --newest 40 --rows 2048 --label inwindow \
        --out data/era_probe/inwindow_20260803.npz --force

**The set FREEZES after generation.** Same convention as
``scripts/build_audit_set.py``: new sampling is a NEW VERSION, at a new path,
because a column of ``progress.csv`` is only readable across iterations if the
rows behind it did not move. Writing over an existing set therefore needs
``--force``, and ``--force`` on the ERA leg mid-run silently re-rules a column
whose header was fixed on row 1. The in-window twin is the one that is MEANT to
rotate — it is re-cut at each restart by construction, and that is why its
readings are only ever compared within a segment.

**Desync screening is not optional.** Every candidate shard goes through the
shipped two-axis SF-desync gate (``eval/value_optimism.desync_reject_reason``,
the SAME predicate ``scripts/quarantine_desync_shards.py`` and
``scripts/value_optimism.py`` enforce — a guard must share the criterion's
instrument), and any shard named by a quarantine manifest is refused before it
is even read. The frozen holdout is the cautionary tale: it was cut from
poisoned shards, reads ``test_sf_labelled_no_multipv_frac`` 0.101305, and does
NOT age out, because the set is frozen. A poisoned probe set would be a
forgetting curve about detached labels.

**Game-clustered sampling.** Rows are drawn in whole GAMES, not
independently. Two reasons, and the second is the load-bearing one:

  * a within-game row draw is biased by ply — games contribute different
    numbers of eligible rows at different phases, and an iid draw over rows
    silently over-weights long games' middlegames;
  * the effective sample size of the set is its GAME count, not its row count.
    Rows inside a game are strongly correlated, so a 2048-row set drawn from 30
    games has the noise floor of ~30 samples. ``n_games`` is recorded in the
    provenance so nobody quotes the row count as the denominator.

**Eligible rows** are those carrying ``sf_p0_regret`` AND ``legal_mask``. The
policy ruler is the net's EXPECTED SF cp-regret under its own prior, and
``sf_p0_regret`` is the only stored per-move cp signal that describes THIS
position (a row's own ``sf_multipv_raw`` is SF's read of the NEXT position in
the opponent's perspective — labels are queried at P1). The filter is
POSITION-level and applied once at build time, identically for both legs, so it
cannot condition a denominator on any outcome. It DOES restrict both sets to
consecutive-full-ply selfplay rows (~24% of selfplay rows): the pair is
comparable to each other, and neither is a sample of the whole window.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.eval.era_probe import (
    PROBE_LABELS,
    PROBE_SET_FIELDS,
    probe_set_digest,
    provenance_path,
)
from chess_anti_engine.eval.value_optimism import (
    SF_LABEL_ATTACHMENT_MIN,
    SF_MULTIPV_MISS_MAX,
)
from chess_anti_engine.replay.shard import (
    LOCAL_SHARD_SUFFIX,
    iter_shard_paths,
    load_shard_arrays,
    save_npz_arrays,
)
from scripts.quarantine_desync_shards import judge

PROVENANCE_VERSION = 1

# Written alongside the value arrays so the npz is a valid shard by the
# schema's own rules (`prune_storage_arrays` requires them) even though the
# probe reads neither.
_SCHEMA_FILLER_FIELDS: tuple[str, ...] = ("has_policy",)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _quarantined_paths(manifests: list[str]) -> tuple[set[str], set[tuple[str, str]]]:
    """(resolved original paths, (shard_dir, name) pairs) named by the manifests.

    BOTH keys, because neither alone is sound. ``original_path`` is exact but
    absent from a hand-written manifest; ``name`` alone is the join this repo
    has already been bitten by — shard indices COLLIDE across lineages, so
    ``shard_033951.zarr`` names a different shard in every trial. The pair
    ``(shard_dir, name)`` is the safe fallback.
    """
    by_path: set[str] = set()
    by_dir_name: set[tuple[str, str]] = set()
    for raw in manifests:
        p = Path(raw).expanduser()
        data = json.loads(p.read_text(encoding="utf-8"))
        root = str(Path(str(data.get("shard_dir", ""))).expanduser().resolve())
        for rec in data.get("shards", []) or []:
            orig = rec.get("original_path")
            if orig:
                by_path.add(str(Path(str(orig)).expanduser().resolve()))
            name = rec.get("name")
            if name:
                by_dir_name.add((root, str(name)))
    return by_path, by_dir_name


def _candidate_shards(
    shard_dirs: list[str], *, newest: int, oldest: int,
) -> list[Path]:
    """Shards to consider, in the order they will be screened.

    ``--newest`` / ``--oldest`` require exactly ONE ``--shard-dir``: shard
    indices collide across lineages, so "the newest 40" over a pooled list of
    two trials' directories is not a well-defined set of rows. Refused rather
    than resolved by a tie-break nobody would remember.
    """
    if (newest > 0 or oldest > 0) and len(shard_dirs) != 1:
        raise SystemExit(
            "--newest/--oldest need exactly one --shard-dir: shard indices "
            "collide across lineages, so a recency order over pooled "
            "directories is undefined. Cut one set per lineage, or drop the "
            "recency selector and take the whole directory."
        )
    if newest > 0 and oldest > 0:
        raise SystemExit("--newest and --oldest are mutually exclusive")

    found: list[Path] = []
    for raw in shard_dirs:
        d = Path(raw).expanduser()
        if not d.is_dir():
            raise SystemExit(f"--shard-dir {d} is not a directory")
        paths = iter_shard_paths(d)
        if not paths:
            raise SystemExit(f"--shard-dir {d} holds no {LOCAL_SHARD_SUFFIX} shards")
        found.extend(paths)
    # iter_shard_paths already sorts by name, which for `shard_NNNNNN` is index
    # order within one directory.
    if newest > 0:
        return found[-int(newest):]
    if oldest > 0:
        return found[: int(oldest)]
    return found


def _eligible_rows(arrs: dict[str, Any]) -> np.ndarray:
    """Boolean mask of rows this probe can score.

    Both flags are required, not just the regret one. With no legal mask the
    policy softmax spreads onto illegal indices, and those are not zero-regret:
    ``_build_sf_p0_regret_vector`` pre-fills every uncovered index with
    ``(worst_regret + 1) / 2 >= 0.5`` and only then overwrites the listed moves,
    so illegal indices measure mean **0.8302** against **0.3272** on legal ones
    (2578 rows of ``data/c17_ab/pre``). A maskless row therefore reads HIGHER
    than the net earns, not lower — a large PESSIMISTIC bias. (This comment
    previously claimed the opposite sign and a zero-regret premise; corrected in
    PR #315 review by measuring the shards.)
    """
    has_reg = np.asarray(arrs.get("has_sf_p0_regret", ()), dtype=np.uint8).astype(bool)
    if has_reg.size == 0:
        return np.zeros(0, dtype=bool)
    has_mask = np.asarray(
        arrs.get("has_legal_mask", np.zeros_like(has_reg)), dtype=np.uint8,
    ).astype(bool)
    return has_reg & has_mask


def _game_clusters(
    arrs: dict[str, Any], eligible: np.ndarray, *, shard_key: str,
) -> list[tuple[tuple[str, int], np.ndarray]]:
    """Eligible row indices grouped into (shard, game) clusters.

    Keyed on ``(shard_key, game_id)`` and never on ``game_id`` alone: game ids
    are per-writer counters and collide across shards exactly as shard indices
    collide across lineages, so a bare ``game_id`` join would silently merge
    two unrelated games into one cluster. Rows with no ``game_id`` become
    singleton clusters — correct rather than convenient, since an unclustered
    row's correlation with its neighbours is unknown, and the count is
    reported so a set that is mostly singletons announces itself.
    """
    idx = np.flatnonzero(eligible)
    if idx.size == 0:
        return []
    gid = np.asarray(arrs.get("game_id", np.zeros(eligible.shape[0], dtype=np.int64)))
    has_gid = np.asarray(
        arrs.get("has_game_id", np.zeros(eligible.shape[0], dtype=np.uint8)),
        dtype=np.uint8,
    ).astype(bool)
    groups: dict[tuple[str, int], list[int]] = {}
    for i in idx.tolist():
        key = (shard_key, int(gid[i])) if bool(has_gid[i]) else (shard_key, -1 - int(i))
        groups.setdefault(key, []).append(i)
    return [(k, np.asarray(v, dtype=np.int64)) for k, v in groups.items()]


def _take_fields(arrs: dict[str, Any], idx: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in (*PROBE_SET_FIELDS, *_SCHEMA_FILLER_FIELDS):
        val = arrs.get(name)
        if val is None:
            continue
        out[name] = np.asarray(val)[idx]
    return out


def build(args: argparse.Namespace) -> int:
    out_path = Path(args.out).expanduser()
    if out_path.exists() and not args.force:
        print(
            f"REFUSING to overwrite {out_path}: a probe set FREEZES after "
            f"generation, and re-cutting one in place silently changes what an "
            f"already-published column means. Write a new version (a new path) "
            f"or pass --force if this is the rotating in-window twin.",
            file=sys.stderr,
        )
        return 2

    quarantined_paths, quarantined_dir_name = _quarantined_paths(
        list(args.quarantine_manifest or []),
    )
    candidates = _candidate_shards(
        list(args.shard_dir), newest=int(args.newest), oldest=int(args.oldest),
    )

    rng = np.random.default_rng(int(args.seed))
    clusters: list[tuple[tuple[str, int], Path, np.ndarray]] = []
    shard_records: list[dict[str, Any]] = []
    rejected: list[list[str]] = []
    skipped_quarantined: list[str] = []
    loaded: dict[Path, dict[str, Any]] = {}

    for path in candidates:
        resolved = str(path.resolve())
        parent = str(path.parent.resolve())
        if resolved in quarantined_paths or (parent, path.name) in quarantined_dir_name:
            skipped_quarantined.append(resolved)
            continue
        verdict = judge(path)
        if verdict.reject:
            rejected.append([resolved, verdict.reason])
            continue
        if verdict.is_marker:
            continue
        arrs, _meta = load_shard_arrays(path)
        eligible = _eligible_rows(arrs)
        if not eligible.any():
            shard_records.append({
                "path": resolved, "name": path.name, "index": verdict.index,
                "rows": int(verdict.rows), "eligible": 0, "taken": 0,
            })
            continue
        loaded[path] = arrs
        for key, rows in _game_clusters(arrs, eligible, shard_key=resolved):
            clusters.append((key, path, rows))
        shard_records.append({
            "path": resolved, "name": path.name, "index": verdict.index,
            "rows": int(verdict.rows), "eligible": int(eligible.sum()), "taken": 0,
        })

    if not clusters:
        print(
            "no eligible rows found. Every candidate shard was quarantined, "
            "rejected by the desync gate, or carries no rows with BOTH "
            "sf_p0_regret and legal_mask (selfplay.record_sf_p0_regret was off "
            "when these shards were written).",
            file=sys.stderr,
        )
        for name, why in rejected[:10]:
            print(f"  rejected {Path(name).name}: {why}", file=sys.stderr)
        return 3

    order = rng.permutation(len(clusters))
    target = int(args.rows)
    chosen_by_shard: dict[Path, list[np.ndarray]] = {}
    taken = 0
    n_games = 0
    n_singletons = 0
    for j in order.tolist():
        key, path, rows = clusters[j]
        # Whole clusters only. Truncating the last game to hit `--rows` exactly
        # would put a partial game in the set and make `n_games` a lie about
        # the effective sample size, which is the number the noise floor is
        # set by.
        if taken + int(rows.size) > target:
            continue
        chosen_by_shard.setdefault(path, []).append(rows)
        taken += int(rows.size)
        n_games += 1
        n_singletons += int(key[1] < 0)
        if taken >= target:
            break

    parts: list[dict[str, np.ndarray]] = []
    by_record = {rec["path"]: rec for rec in shard_records}
    for path in candidates:
        picks = chosen_by_shard.get(path)
        if not picks:
            continue
        # Sorted so the frozen set's row ORDER is a deterministic function of
        # (shard order, row index) and not of the shuffle. Two builds with the
        # same seed already agree; this makes the digest independent of the
        # permutation's internal ordering as well, so a set is identified by
        # WHICH rows it holds.
        idx = np.sort(np.concatenate(picks))
        parts.append(_take_fields(loaded[path], idx))
        by_record[str(path.resolve())]["taken"] = int(idx.size)

    keys = sorted({k for part in parts for k in part})
    merged: dict[str, np.ndarray] = {}
    for key in keys:
        present = [p[key] for p in parts if key in p]
        if len(present) != len(parts):
            raise SystemExit(
                f"field {key!r} is present on some selected shards and absent "
                f"on others; a probe set must not be a mixture of schemas "
                f"(zero-filling here would make an absent legal mask "
                f"indistinguishable from an all-illegal position)"
            )
        merged[key] = np.concatenate(present, axis=0)

    n_rows = int(np.asarray(merged["x"]).shape[0])
    frozen = {k: v for k, v in merged.items() if k in PROBE_SET_FIELDS}
    digest = probe_set_digest(frozen)
    n_policy = int(np.count_nonzero(
        np.asarray(merged["has_sf_p0_regret"]).astype(bool)))

    provenance: dict[str, Any] = {
        "version": PROVENANCE_VERSION,
        "label": str(args.label),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "built_by_git_sha": _git_sha(),
        "seed": int(args.seed),
        "lineage": str(args.lineage or Path(args.shard_dir[0]).expanduser().resolve()),
        "rows_requested": target,
        "rows": n_rows,
        "policy_rows": n_policy,
        "n_games": n_games,
        "n_singleton_clusters": n_singletons,
        "n_shards": sum(1 for r in shard_records if r["taken"] > 0),
        "n_shards_considered": len(candidates),
        "shards": [r for r in shard_records if r["taken"] > 0],
        "desync_screened": True,
        "desync_gate": {
            "sf_label_attachment_min": SF_LABEL_ATTACHMENT_MIN,
            "sf_multipv_miss_max": SF_MULTIPV_MISS_MAX,
            "predicate": "chess_anti_engine.eval.value_optimism.desync_reject_reason",
        },
        "shards_rejected": rejected,
        "quarantine_manifests": [
            str(Path(m).expanduser().resolve()) for m in (args.quarantine_manifest or [])
        ],
        "shards_skipped_quarantined": skipped_quarantined,
        "fields": list(PROBE_SET_FIELDS),
        "planes": int(np.asarray(merged["x"]).shape[1]),
        "policy_width": int(np.asarray(merged["policy_target"]).shape[1]),
        "digest": digest,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz_arrays(out_path, arrs=merged, meta=None, compress=True)
    # Provenance rides in a SIDECAR, not inside the npz. `ShardMeta` is a closed
    # dataclass and would refuse the block; more to the point, the shard list,
    # the reject log and the gate thresholds are what an operator reads with
    # `cat` when deciding whether to trust a curve, and a member buried in an
    # npz is unreadable without Python. The binding between the two is the
    # DIGEST, recorded here and RECHECKED by `load_probe_set` against the rows
    # it actually loaded — so a sidecar that has drifted from its set announces
    # itself instead of describing the wrong file.
    provenance_path(out_path).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    # The build-time half of the proof-of-effect pair: this digest and row
    # count must equal the `[probe] <label> set loaded:` line the trial prints
    # at startup. If they differ, the run is scoring a different set than the
    # one that was screened here.
    print(
        f"[era-probe] wrote {out_path}\n"
        f"  label={args.label} rows={n_rows} policy_rows={n_policy} "
        f"games={n_games} (singletons {n_singletons}) digest={digest}\n"
        f"  shards used={provenance['n_shards']}/{len(candidates)} "
        f"rejected={len(rejected)} quarantined-skipped={len(skipped_quarantined)}\n"
        f"  lineage={provenance['lineage']}\n"
        f"  planes={provenance['planes']} policy_width={provenance['policy_width']}"
    )
    if rejected:
        for name, why in rejected[:6]:
            print(f"  DESYNC-REJECTED {Path(name).name}: {why}")
        if len(rejected) > 6:
            print(f"  ... and {len(rejected) - 6} more")
    if n_rows < target:
        print(
            f"  NOTE: {n_rows} < --rows {target}. Whole game clusters only, so "
            f"the last cluster that would have overshot was skipped; widen the "
            f"shard selection if the shortfall is large."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "", formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", action="append", required=True,
                    help="replay shard directory to draw from (repeatable)")
    ap.add_argument("--out", required=True, help="output .npz path for the frozen set")
    ap.add_argument("--label", default="era", choices=list(PROBE_LABELS),
                    help="which leg of the pair this set is (recorded in provenance)")
    ap.add_argument("--rows", type=int, default=2048,
                    help="target row count; whole game clusters only, so the "
                         "realized count is <= this")
    ap.add_argument("--newest", type=int, default=0,
                    help="consider only the N highest-index shards (the "
                         "in-window twin); needs exactly one --shard-dir")
    ap.add_argument("--oldest", type=int, default=0,
                    help="consider only the N lowest-index shards (the era "
                         "leg); needs exactly one --shard-dir")
    ap.add_argument("--seed", type=int, default=0,
                    help="cluster-shuffle seed; recorded in the provenance")
    ap.add_argument("--lineage", default="",
                    help="human label for the run these shards came from; "
                         "defaults to the first --shard-dir's resolved path")
    ap.add_argument("--quarantine-manifest", action="append",
                    help="quarantine_manifest.json whose shards must be "
                         "refused (repeatable)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing --out. A frozen set is only "
                         "re-cut deliberately; the in-window twin is the leg "
                         "that is meant to rotate")
    return build(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
