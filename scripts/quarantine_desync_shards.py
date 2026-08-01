#!/usr/bin/env python3
"""Quarantine the replay shards the shipped SF-desync gate rejects.

Cause and fix of the desync belong to PR #297; this script only disposes of the
rows it left behind. The disposition, its pricing and its pre-committed
thresholds are the 2026-08-01 entry in ``docs/experiment_ledger.md``.

Reversible by construction: shards are MOVED, never deleted; the manifest is a
JOURNAL written BEFORE the first move and marked complete only at the end, so an
interrupted run is resumable rather than wedged; and every recorded digest is
CHECKED on restore rather than merely stored.

    # 1. see what would move (default; touches nothing)
    PYTHONPATH=. python3 scripts/quarantine_desync_shards.py --shard-dir DIR

    # 2. do it
    PYTHONPATH=. python3 scripts/quarantine_desync_shards.py --shard-dir DIR \
        --quarantine-dir OUT --apply

    # 3. the deciding yardstick (needs the manifest: an unchecked axis is not a pass)
    PYTHONPATH=. python3 scripts/quarantine_desync_shards.py --shard-dir DIR \
        --quarantine-dir OUT --verify

    # 4. undo; resumable, digest-checked
    PYTHONPATH=. python3 scripts/quarantine_desync_shards.py --shard-dir DIR \
        --quarantine-dir OUT --restore

⚑ THE SHARD-INDEX RESERVATION IS NOT OPTIONAL. ``DiskReplayBuffer`` never
persists its next shard id: ``_scan_existing_shards`` recomputes it as
``max(shard_index(p)) + 1`` over a glob of the shard dir, and
``_next_free_shard_path`` only ever increments from there. The reject set
contains the NEWEST shards, so a plain move drops that counter below ids that
now live in the quarantine dir and the next writes silently reuse them.
``_next_free_shard_path`` CANNOT warn about this: it only logs when it steps over
a file that is still present, which by construction never happens for an id that
was moved away. Reserving the single highest quarantined id with a valid
ZERO-ROW shard holds the counter above the whole set (it is monotone and keyed
off the maximum), contributes no rows to the window, and — unlike an empty
directory or a dangling symlink, both of which also reserve — loads without
error, so it does not feed ``_refresh_failed_total`` on every refresh draw.

⚑ THE RESERVATION IS THEREFORE A VERIFY AXIS, NOT A PRINTOUT. The state this
mechanism exists to prevent (shards moved, reservation absent) is invisible to
every other check, so ``--verify`` fails on it. A yardstick that passes on the
exact failure it was built to catch is worse than no yardstick.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from chess_anti_engine.eval.value_optimism import (
    SF_LABEL_ATTACHMENT_MIN,
    SF_MULTIPV_MISS_MAX,
    AxisReading,
    sf_label_attachment_corr,
    sf_multipv_missing_rate,
)
from chess_anti_engine.replay.shard import (
    LOCAL_SHARD_SUFFIX,
    iter_shard_paths,
    load_shard_arrays,
    save_local_shard_arrays,
    shard_index,
    shard_positions,
)

MANIFEST_NAME = "quarantine_manifest.json"
RESTORED_MANIFEST_NAME = "quarantine_manifest.restored.json"
RESERVATION_STAGING = "_reservation"
TRAIN_PIDFILE = Path("/tmp/chess_training.pid")
# ⚑ There is deliberately NO "clean maximum" constant here any more.
#
# One used to live at this line, set to 0.008032 — the `:.6f` rendering of
# 0.008032128514056224, the exact maximum it was then compared against (16/1992
# on shard_033643). So the comparison `max <= CONST` was a float against its own
# rounded display and could only ever be FALSE. On the first real run after the
# apply it failed by 1.285e-07, on the very window it was derived from.
#
# The rounding was the proximate bug; the design was the real one. A bar taken
# from the maximum of the sample it judges is either vacuous (constant >= max)
# or unpassable (constant < max) — it cannot carry information about that
# sample, whichever way the rounding falls. Worse as a forward monitor: the
# maximum of a fresh equal-sized draw from the SAME clean distribution exceeds a
# previous sample maximum about half the time, so it is a false-positive
# generator by construction.
#
# The drift question it was reaching for is real and is now asked
# non-circularly, on data that is not the sample being judged — see axis B.
#
# The measured value 0.008032128514056224 is a fact about this window and is
# recorded in the ledger, where a fact belongs. It is not a live threshold.


@dataclass(frozen=True)
class ShardVerdict:
    """One shard's gate reading. ``reject`` is the shipped gate's own verdict."""

    name: str
    index: int
    rows: int
    labelled: int
    multipv_miss: float
    multipv_status: str
    attachment: float
    attachment_status: str
    reject: bool
    reason: str
    is_marker: bool = False


class ManifestError(RuntimeError):
    """A manifest that is missing, unreadable, or not about this shard dir."""


def _training_is_up() -> bool:
    """True when scripts/train.sh's pidfile names a live process."""
    try:
        pid = int(TRAIN_PIDFILE.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _axis(reading: AxisReading) -> tuple[float, str]:
    return float(reading.value), str(reading.status)


def judge(path: Path) -> ShardVerdict:
    """Apply the shipped two-axis gate to one shard, read-only.

    A non-"ok" status on either axis is a REJECT, not a pass: a gate that could
    not evaluate a shard has not cleared it.

    ⚑ This rule is a THIRD copy. ``scripts/value_optimism.py`` applies the same
    reject rule inline, and ``tests/test_quarantine_desync_shards.py`` pins this
    copy against the axis functions and thresholds imported from
    ``chess_anti_engine.eval.value_optimism`` — **it does not import, execute or
    observe ``scripts/value_optimism.py`` at all, so a drift in THAT copy passes
    the whole suite.** Demonstrated in review: changing its
    ``multipv.value > multipv_miss_max`` to ``> 999.0`` disables the shipped
    scorer's multipv axis, flips 118 of the 834 live shards, and every test here
    still passes. The only real fix is one shared predicate next to the
    thresholds in ``eval/value_optimism.py``, which this change is barred from
    editing; it is the recorded follow-up, and until it lands the copies are
    genuinely unpinned against each other.
    """
    z = zarr.open_group(str(path), mode="r")
    keys = set(z.array_keys())
    labelled = (
        np.asarray(z["has_sf_wdl"][:]).astype(bool)
        if "has_sf_wdl" in keys
        else np.zeros(0, dtype=bool)
    )
    rows = int(np.asarray(z["priority"][:]).shape[0]) if "priority" in keys else 0

    if "has_sf_multipv_raw" in keys:
        miss, miss_st = _axis(
            sf_multipv_missing_rate(np.asarray(z["has_sf_multipv_raw"][:]), labelled),
        )
    else:
        miss, miss_st = float("nan"), "field_missing"

    if {"x", "sf_wdl"} <= keys:
        att, att_st = _axis(
            sf_label_attachment_corr(
                np.asarray(z["x"][:, 0:12]),
                np.asarray(z["sf_wdl"][:]).astype(np.float64),
                labelled,
            ),
        )
    else:
        att, att_st = float("nan"), "field_missing"

    # A zero-row shard is an index reservation this script installed, not data.
    # It carries no rows into training, so it cannot contaminate anything and
    # must not be re-quarantined or counted against the --verify bar. Without
    # this the yardstick fails forever on the marker the script itself left,
    # and a second --apply would move it into the quarantine dir. The exemption
    # is keyed on the row count read back off disk, so it cannot be claimed by
    # a shard that actually holds rows.
    if rows == 0:
        return ShardVerdict(
            name=path.name, index=shard_index(path), rows=0, labelled=0,
            multipv_miss=miss, multipv_status=miss_st,
            attachment=att, attachment_status=att_st,
            reject=False, reason="zero-row index reservation", is_marker=True,
        )

    reason = ""
    if att_st != "ok":
        reason = f"attachment {att_st}"
    elif att < SF_LABEL_ATTACHMENT_MIN:
        reason = f"attachment {att:+.4f} < {SF_LABEL_ATTACHMENT_MIN}"
    elif miss_st != "ok":
        reason = f"multipv-miss {miss_st}"
    elif miss > SF_MULTIPV_MISS_MAX:
        reason = f"multipv-miss {miss:.6f} > {SF_MULTIPV_MISS_MAX}"
    return ShardVerdict(
        name=path.name, index=shard_index(path), rows=rows,
        labelled=int(labelled.sum()), multipv_miss=miss, multipv_status=miss_st,
        attachment=att, attachment_status=att_st,
        reject=bool(reason), reason=reason,
    )


def digest(path: Path) -> str:
    """Content digest of a zarr shard: sorted relative paths and their bytes."""
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _empty_like(path: Path) -> dict[str, np.ndarray]:
    """A zero-row copy of one shard's arrays, for use as an index reservation.

    Row-dimensioned arrays are sliced to ``[:0]``; the scalar encoding-identity
    markers travel unchanged, because dropping one makes the shard reload with
    the flag defaulted off.
    """
    arrs, _meta = load_shard_arrays(path)
    n = int(np.asarray(arrs["x"]).shape[0])
    out: dict[str, np.ndarray] = {}
    for key, val in arrs.items():
        arr = np.asarray(val)
        out[key] = arr[:0] if arr.ndim >= 1 and arr.shape[0] == n else arr
    return out


def _reservation_index(rejects: list[ShardVerdict], all_paths: list[Path]) -> int | None:
    """The id to reserve, or None when the counter cannot drop.

    Only the MAXIMUM matters: the counter is derived from it and never
    decreases, so holding the top id also protects every lower one in the set.
    """
    if not rejects:
        return None
    top_all = max(shard_index(p) for p in all_paths)
    top_reject = max(v.index for v in rejects)
    return top_reject if top_reject == top_all else None


def _manifest_path(quar_dir: Path) -> Path:
    return quar_dir / MANIFEST_NAME


def read_manifest(quar_dir: Path, shard_dir: Path | None = None) -> dict[str, Any]:
    """Load a manifest, refusing clearly on every way it can be unusable.

    A corrupt manifest must not surface as a raw ``JSONDecodeError``, and a
    manifest describing a DIFFERENT window must not be applied to this one:
    several sibling ``replay_shards`` dirs on this box share the
    ``shard_NNNNNN.zarr`` namespace, so the names alone do not identify a window.
    """
    p = _manifest_path(quar_dir)
    if not p.exists():
        retired = quar_dir / RESTORED_MANIFEST_NAME
        if retired.exists():
            raise ManifestError(
                f"no active manifest at {p}, but {retired.name} is present: this "
                f"quarantine was ALREADY RESTORED "
                f"(completed {json.loads(retired.read_text()).get('completed_utc', '?')}). "
                f"Nothing further to undo. Re-check the window with --verify.",
            )
        raise ManifestError(
            f"no manifest at {p}. If an --apply was interrupted before it wrote "
            f"the journal then no shard had moved yet and there is nothing to "
            f"undo; confirm with --shard-dir alone (dry run).",
        )
    try:
        man = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ManifestError(
            f"manifest at {p} is unreadable: {type(e).__name__}: {e}",
        ) from e
    if not isinstance(man, dict) or "shards" not in man:
        raise ManifestError(f"manifest at {p} has no 'shards' list; refusing to guess")
    if shard_dir is not None:
        recorded = str(man.get("shard_dir", ""))
        if recorded != str(shard_dir.resolve()):
            raise ManifestError(
                f"manifest is for a DIFFERENT window:\n"
                f"  manifest shard_dir : {recorded}\n"
                f"  --shard-dir        : {shard_dir.resolve()}\n"
                f"Sibling replay dirs share the shard_NNNNNN.zarr namespace, so "
                f"restoring across them would silently mix two windows. NOTE the "
                f"binding is the RESOLVED ABSOLUTE PATH recorded at apply time, "
                f"so renaming or moving the run directory also trips this — that "
                f"is deliberate (refusing is the safe direction), but the fix in "
                f"that case is to edit 'shard_dir' in the manifest to the new "
                f"path, not to bypass the check.",
            )
    return man


def _report(verdicts: list[ShardVerdict], reserve: int | None) -> None:
    rejects = [v for v in verdicts if v.reject]
    kept = [v for v in verdicts if not v.reject and not v.is_marker]
    markers = [v for v in verdicts if v.is_marker]
    rows_all = sum(v.rows for v in verdicts)
    rows_rej = sum(v.rows for v in rejects)
    only_attach = [
        v for v in rejects
        if v.reason.startswith("attachment")
        and v.multipv_status == "ok" and v.multipv_miss <= SF_MULTIPV_MISS_MAX
    ]
    too_few = [v for v in rejects if "too_few_rows" in v.reason]
    print(f"shards scanned        : {len(verdicts)}  rows {rows_all:,}")
    print(f"REJECTED by the gate  : {len(rejects)}  rows {rows_rej:,} "
          f"({rows_rej / max(rows_all, 1) * 100:.2f}% of the window)")
    print(f"  over-threshold        : {len(rejects) - len(too_few)}")
    print(f"  unevaluable (too_few_rows): {len(too_few)} "
          f"{[f'{v.name}({v.labelled})' for v in too_few]}")
    print(f"kept                  : {len(kept)}  rows {rows_all - rows_rej:,}")
    if markers:
        print(f"zero-row reservations already present (left alone): "
              f"{[v.name for v in markers]}")
    print(f"rejected by ATTACHMENT and not by MULTIPV: {len(only_attach)}")
    if kept:
        ok = [v.multipv_miss for v in kept if v.multipv_status == "ok"]
        if ok:
            print(f"max multipv-miss among kept: {max(ok)!r}")
    held = (
        f"shard_{reserve:06d}{LOCAL_SHARD_SUFFIX}" if reserve is not None
        else "NOT NEEDED (counter cannot drop)"
    )
    print(f"index reservation     : {held}")
    if rejects:
        lo, hi = min(v.index for v in rejects), max(v.index for v in rejects)
        print(f"reject id range       : {lo}..{hi}")
        # Dedupe: with fewer than 10 rejects the head and tail slices overlap.
        shown = list(dict.fromkeys(rejects[:5] + rejects[-5:]))
        print(f"\n  sample of {len(shown)} rejects:")
        for v in shown:
            print(f"    {v.name}  rows={v.rows:6d} lab={v.labelled:6d} "
                  f"miss={v.multipv_miss:.6f} att={v.attachment:+.4f}  {v.reason}")


def do_apply(shard_dir: Path, quar_dir: Path, verdicts: list[ShardVerdict],
             all_paths: list[Path]) -> int:
    rejects = [v for v in verdicts if v.reject]
    kept = [v for v in verdicts if not v.reject and not v.is_marker]
    if not rejects:
        print("nothing to quarantine")
        return 0
    if quar_dir.exists() and any(quar_dir.iterdir()):
        has_manifest = _manifest_path(quar_dir).exists()
        print(f"ERROR: {quar_dir} exists and is not empty; refusing to mix two "
              f"quarantines.")
        if has_manifest:
            print("  A manifest is present, so an --apply got at least as far as "
                  "the journal: --restore first (that is safe and resumable), "
                  "then re-apply.")
        else:
            print("  There is NO manifest, so the previous run died before the "
                  "journal was written and therefore before any shard moved. "
                  "Nothing is quarantined and --restore has nothing to undo; the "
                  f"only leftover is the staged reservation. Remove it by hand "
                  f"(rm -rf {quar_dir / RESERVATION_STAGING}) and re-apply.")
        return 2
    quar_dir.mkdir(parents=True, exist_ok=True)

    reserve = _reservation_index(rejects, all_paths)
    staged: Path | None = None
    if reserve is not None:
        # Build the reservation BEFORE anything moves, so a failure here leaves
        # the window untouched rather than half-quarantined and collision-prone.
        src = shard_dir / f"shard_{reserve:06d}{LOCAL_SHARD_SUFFIX}"
        stage_dir = quar_dir / RESERVATION_STAGING
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged = save_local_shard_arrays(stage_dir / src.name, arrs=_empty_like(src))
        got = shard_positions(staged)
        if got != 0:
            print(f"ERROR: staged reservation has {got} rows, expected 0; aborting")
            return 2
        print(f"staged zero-row reservation for id {reserve} at {staged}")

    # JOURNAL FIRST. Digests are taken while the shards are still in the window,
    # and the manifest lands before the first move, so an interrupted run is a
    # resumable restore rather than an unidentifiable pile of shards.
    print(f"digesting {len(rejects)} shards...", flush=True)
    records: list[dict[str, Any]] = []
    for v in rejects:
        src = shard_dir / v.name
        rec = asdict(v)
        rec["original_path"] = str(src.resolve())
        rec["mtime"] = src.stat().st_mtime
        rec["sha256"] = digest(src)
        records.append(rec)

    manifest: dict[str, Any] = {
        "complete": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shard_dir": str(shard_dir.resolve()),
        "gate": {
            "sf_label_attachment_min": SF_LABEL_ATTACHMENT_MIN,
            "sf_multipv_miss_max": SF_MULTIPV_MISS_MAX,
        },
        "reservation_index": reserve,
        "reservation_name": (
            f"shard_{reserve:06d}{LOCAL_SHARD_SUFFIX}" if reserve is not None else None
        ),
        "shards": records,
        # The exact post-apply window, so --verify can detect OVER-removal and a
        # botched move, not only leftover dirt. Names AND rows, because a count
        # alone cannot tell "lost a clean shard" from "gained one".
        "kept": [{"name": v.name, "rows": v.rows} for v in kept],
        "totals": {
            "shards": len(records),
            "rows": sum(int(r["rows"]) for r in records),
            "labelled": sum(int(r["labelled"]) for r in records),
            "kept_shards": len(kept),
            "kept_rows": sum(v.rows for v in kept),
        },
    }
    _manifest_path(quar_dir).write_text(json.dumps(manifest, indent=2))
    print(f"journal written (complete=false): {_manifest_path(quar_dir)}")

    for rec in records:
        shutil.move(str(shard_dir / str(rec["name"])), str(quar_dir / str(rec["name"])))
    print(f"moved {len(records)} shards to {quar_dir}")

    if staged is not None and reserve is not None:
        dest = shard_dir / f"shard_{reserve:06d}{LOCAL_SHARD_SUFFIX}"
        if dest.exists():
            print(f"ERROR: {dest} reappeared; reservation NOT installed. The window "
                  f"is quarantined but UNPROTECTED — --restore immediately.")
            return 2
        shutil.move(str(staged), str(dest))
        shutil.rmtree(quar_dir / RESERVATION_STAGING, ignore_errors=True)
        print(f"installed reservation {dest.name} (rows={shard_positions(dest)})")

    manifest["complete"] = True
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _manifest_path(quar_dir).write_text(json.dumps(manifest, indent=2))
    print(f"manifest marked complete: {_manifest_path(quar_dir)}")
    print("\nnow run --verify (WITH --quarantine-dir) before restarting training.")
    return 0


def _move_back(rec: dict[str, Any], shard_dir: Path, quar_dir: Path) -> str:
    """Restore one shard. Returns 'moved', 'skip', or a 'REFUSE: ...' reason.

    Idempotent by digest, which is what makes --restore resumable: a shard
    already correctly in place is skipped, while one whose bytes do not match
    the manifest stops the run rather than being silently accepted.
    """
    name = str(rec["name"])
    src, dest = quar_dir / name, shard_dir / name
    want = str(rec.get("sha256", ""))
    if dest.exists():
        got = digest(dest)
        if want and got != want:
            return (f"REFUSE: {dest} exists but its digest {got[:12]} != manifest "
                    f"{want[:12]}. That is NOT the shard this manifest describes.")
        return "skip"
    if not src.exists():
        return f"REFUSE: {src} is missing and {dest} does not exist"
    got = digest(src)
    if want and got != want:
        return (f"REFUSE: {src} digest {got[:12]} != manifest {want[:12]} — the "
                f"quarantined shard was modified after it was banked.")
    shutil.move(str(src), str(dest))
    return "moved"


def do_restore(shard_dir: Path, quar_dir: Path) -> int:
    """Move every quarantined shard back. Resumable and digest-checked.

    Order matters: the reservation occupies the id its own shard reclaims, so it
    is removed only after every OTHER shard is back, and that shard moves last.
    A run interrupted anywhere can simply be re-run — at no point is the window
    left both un-restored and unprotected.
    """
    try:
        man = read_manifest(quar_dir, shard_dir)
    except ManifestError as e:
        print(f"ERROR: {e}")
        return 2
    reserve_name = man.get("reservation_name")
    records = list(man["shards"])
    tail = [r for r in records if r["name"] == reserve_name]
    head = [r for r in records if r["name"] != reserve_name]

    moved = skipped = 0
    for rec in head:
        res = _move_back(rec, shard_dir, quar_dir)
        if res.startswith("REFUSE"):
            print(f"ERROR: {res}")
            print("Nothing further moved. Restore is resumable: fix and re-run.")
            return 2
        moved += res == "moved"
        skipped += res == "skip"

    if reserve_name:
        dest = shard_dir / str(reserve_name)
        want = next((str(r.get("sha256", "")) for r in tail), "")
        if dest.exists():
            rows = shard_positions(dest)
            if rows == 0:
                shutil.rmtree(dest)
                print(f"removed zero-row reservation {dest.name}")
            elif want and digest(dest) == want:
                print(f"{dest.name} is already the restored shard; nothing to remove")
            else:
                print(f"ERROR: {dest} holds {rows} rows and is neither the zero-row "
                      f"reservation nor the shard this manifest banked. That is LIVE "
                      f"data written after the quarantine; refusing to restore over "
                      f"it. The window has already moved on.")
                return 2

    for rec in tail:
        res = _move_back(rec, shard_dir, quar_dir)
        if res.startswith("REFUSE"):
            print(f"ERROR: {res}")
            return 2
        moved += res == "moved"
        skipped += res == "skip"

    print(f"restored {moved} shards to {shard_dir}"
          f"{f' ({skipped} already in place)' if skipped else ''}")
    _manifest_path(quar_dir).replace(quar_dir / RESTORED_MANIFEST_NAME)
    print(f"manifest retired to {RESTORED_MANIFEST_NAME}")
    return 0


def _verify_checks(
    shard_dir: Path, quar_dir: Path | None, verdicts: list[ShardVerdict], top: int,
) -> list[tuple[str, str, str]]:
    """Every verify axis, as one of four explicit states.

    ``pass`` / ``fail`` / ``unchecked`` / ``n/a``. Only ``pass`` and ``n/a``
    clear the verdict.

    ``unchecked`` is a FAIL — the same rule the shard gate applies to a shard it
    cannot read, and the reason the yardstick needs the manifest: without it,
    axes B/C/D/E are silently absent and the tool reports PASS on states it
    exists to prevent (notably: shards moved, reservation never installed).

    ``n/a`` clears, and is therefore the dangerous state — it is the one an
    author reaches for to make a red light go green. It is used in exactly one
    place (axis B before any post-apply shard exists) and only when emptiness is
    PROVED against the manifest's kept-list, never merely assumed. If any kept
    shard is missing, that proof is unavailable and the axis reverts to
    ``unchecked``, i.e. to failing.
    """
    data = [v for v in verdicts if not v.is_marker]
    rejects = [v for v in verdicts if v.reject]
    checks: list[tuple[str, str, str]] = [
        ("A no shard rejected by the gate",
         "pass" if not rejects else "fail", f"{len(rejects)} rejected"),
    ]

    man: dict[str, Any] | None = None
    err = ""
    if quar_dir is not None:
        try:
            man = read_manifest(quar_dir, shard_dir)
        except ManifestError as e:
            err = str(e).splitlines()[0]

    if man is None:
        why = err or "no --quarantine-dir given"
        checks += [
            ("B no post-apply shard carries desync signal", "unchecked", why),
            ("C index reservation holds", "unchecked", why),
            ("D window matches the manifest exactly", "unchecked", why),
            ("E quarantined shards match their digests", "unchecked", why),
        ]
        return checks

    top_q = max(int(r["index"]) for r in man["shards"])
    want = {str(r["name"]): int(r["rows"]) for r in man.get("kept", [])}
    have = {v.name: v.rows for v in data}
    missing = sorted(set(want) - set(have))
    extra = sorted(n for n in set(have) - set(want) if shard_index(Path(n)) <= top_q)
    badrows = sorted(n for n in set(want) & set(have) if want[n] != have[n])

    # B -- drift watch on data written AFTER the apply, which is the only
    # population this tool can say anything non-circular about.
    #
    # The bar is EXACTLY 0.000000, and it is not fitted to this window: on a
    # schema-complete healthy window `sf_multipv_missing_rate` is exactly zero,
    # measured over 2,878,620 labelled rows across four SEPARATED quiet
    # stretches (07-06..07-09, 07-19, shards 33118-33387, 33721-33943) — none of
    # which is the kept set being judged. A desynced engine cannot produce a
    # near-zero rate: it takes its ~1/8 share of labels to ~59% no-PV, so a
    # single non-zero row in a 2000-row shard is ~5e-4, some 500x the benign
    # ceiling of <1e-6 those stretches establish.
    #
    # This is strictly tighter than the gate (0.01) and therefore catches a new
    # episode during its ramp, which is exactly where the 07-30 one was missed.
    # It says nothing about the kept shards — those already cleared the gate,
    # and the tapers deliberately left among them read non-zero by design.
    new_shards = [v for v in data if v.name not in want]
    if new_shards:
        dirty = [v for v in new_shards
                 if v.multipv_status != "ok" or v.multipv_miss > 0.0]
        b_state = "pass" if not dirty else "fail"
        b_detail = (f"{len(dirty)} of {len(new_shards)} new shards non-zero: "
                    f"{[f'{v.name}={v.multipv_miss:.6f}' for v in dirty[:3]]}")
    elif missing:
        # Cannot claim the population is empty when the window does not match
        # the manifest: a shard that vanished is indistinguishable from one that
        # was never written, and guessing here would make the axis unfalsifiable.
        b_state, b_detail = "unchecked", (
            f"cannot establish the post-apply population: {len(missing)} kept "
            f"shards missing, so 'no new shards' is not provable")
    else:
        # ⚑ WHY THIS IS N/A AND NOT UNCHECKED — the one place the
        # UNCHECKED-is-a-FAIL rule does not apply, so the argument is written
        # down rather than assumed.
        #
        # UNCHECKED means: the question has a truth value and this run cannot
        # see it. "Does the reservation hold?" without a manifest is unchecked —
        # there IS a fact, and failing is the honest answer to not knowing it.
        #
        # This axis asks "does every post-apply shard read zero?". Over an empty
        # population that is VACUOUSLY TRUE, not unknown: there is no shard that
        # could falsify it. Failing here would report a defect that cannot
        # exist, and a yardstick that is red in its own correct steady state
        # gets ignored within a day — which is how the previous axis B, red by
        # 1.3e-07 on the state it was derived from, would have ended up.
        #
        # The danger is claiming an empty population without proof, so emptiness
        # is PROVED, not assumed: every data shard in the window is accounted
        # for in the manifest's kept-list, so nothing has been written since the
        # apply. The `elif missing` branch above is what keeps that honest — the
        # moment the window stops matching, the proof is void and the axis
        # reverts to UNCHECKED, i.e. to failing. Pinned by
        # `test_axis_B_is_unchecked_when_emptiness_cannot_be_proved` and by
        # `test_a_dirty_post_apply_shard_cannot_reach_the_empty_branch`;
        # forcing this branch to fire unconditionally kills the suite.
        #
        # Residual limitation, stated: a post-apply shard that was written and
        # then deleted WITHOUT any kept shard also going missing would be
        # invisible here. Eviction cannot do that (it takes the oldest first, so
        # kept shards go before new ones), but an out-of-band delete could.
        b_state, b_detail = "n/a", (
            "no shard written since the apply; axis goes live on the first one")
    checks.append((
        "B no post-apply shard carries desync signal (bar: exactly 0.000000)",
        b_state, b_detail,
    ))

    # C -- the axis the mechanism exists for. `top` must sit at or above the
    # highest quarantined id, or the buffer silently reuses ids that now live in
    # the quarantine dir, and --restore will refuse them forever.
    checks.append((
        f"C index reservation holds (top {top} >= quarantined max {top_q})",
        "pass" if top >= top_q else "fail",
        f"top id {top} < quarantined max {top_q}: ids "
        f"{top + 1}..{top_q} will be SILENTLY REUSED",
    ))
    # D -- over-removal and botched moves, which axis A structurally cannot see.
    checks.append((
        f"D window matches the manifest ({len(want)} shards, {sum(want.values()):,} rows)",
        "pass" if not (missing or extra or badrows) else "fail",
        f"missing {len(missing)}{missing[:3]}  unexpected {len(extra)}{extra[:3]}  "
        f"row-count changed {len(badrows)}{badrows[:3]}",
    ))
    # E -- the digests are a control, not a decoration.
    bad: list[str] = []
    for r in man["shards"]:
        p = quar_dir / str(r["name"]) if quar_dir is not None else None
        if p is None or not p.exists():
            bad.append(f"{r['name']}:absent")
        elif str(r.get("sha256", "")) and digest(p) != str(r["sha256"]):
            bad.append(f"{r['name']}:digest")
    checks.append((
        f"E quarantined shards intact ({len(man['shards'])} digests)",
        "pass" if not bad else "fail", f"{len(bad)} bad: {bad[:3]}",
    ))
    if not man.get("complete", False):
        checks.append((
            "F apply completed", "fail",
            "manifest says complete=false — the apply was interrupted",
        ))
    return checks


def do_verify(shard_dir: Path, quar_dir: Path | None) -> int:
    """The deciding yardstick."""
    paths = iter_shard_paths(shard_dir)
    verdicts = [judge(p) for p in paths]
    data = [v for v in verdicts if not v.is_marker]
    markers = [v for v in verdicts if v.is_marker]
    ok = [v.multipv_miss for v in data if v.multipv_status == "ok"]
    att = [v.attachment for v in data if v.attachment_status == "ok"]
    top = max((shard_index(p) for p in paths), default=-1)

    print(f"shards in window          : {len(paths)}  rows {sum(v.rows for v in verdicts):,}")
    print(f"  data shards             : {len(data)}    reservations: {len(markers)}")
    # Reported at full precision and NOT gated on: rounding this to 6 places is
    # precisely how the old axis B became a threshold equal to its own display.
    print(f"max sf_multipv_missing_rate: {max(ok)!r} (diagnostic, not a bar)"
          if ok else "max multipv-miss: n/a")
    print(f"min sf_label_attachment    : {min(att):+.4f}" if att else "min attachment: n/a")
    print(f"highest shard id           : {top} -> next write would be {top + 1}")
    print(f"zero-row reservations held : {[v.name for v in markers] or 'none'}")

    checks = _verify_checks(shard_dir, quar_dir, verdicts, top)
    print()
    marks = {"pass": "PASS", "fail": "FAIL", "unchecked": "UNCHECKED", "n/a": "N/A"}
    for label, state, detail in checks:
        print(f"  [{marks[state]:9s}] {label}"
              + (f"   -- {detail}" if state != "pass" else ""))
    passed = all(s in ("pass", "n/a") for _, s, _ in checks)
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("  (an UNCHECKED axis is a FAIL: a gate that could not evaluate a "
              "state has not cleared it. Pass --quarantine-dir.)")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--quarantine-dir", type=Path, default=None)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="perform the move")
    mode.add_argument("--restore", action="store_true", help="undo a previous --apply")
    mode.add_argument("--verify", action="store_true", help="the deciding yardstick")
    ap.add_argument("--allow-live-training", action="store_true",
                    help="proceed even if scripts/train.sh reports a live run")
    args = ap.parse_args(argv)

    shard_dir = args.shard_dir
    if not shard_dir.is_dir():
        print(f"ERROR: no such shard dir: {shard_dir}")
        return 2
    if args.apply or args.restore:
        if args.quarantine_dir is None:
            print("ERROR: --quarantine-dir is required with --apply/--restore")
            return 2
        if _training_is_up() and not args.allow_live_training:
            print("ERROR: training appears to be RUNNING (scripts/train.sh pidfile is "
                  "live). Moving shards under a running buffer races its own eviction "
                  "and prefetch. Stop training first, or pass --allow-live-training.")
            return 2

    if args.verify:
        return do_verify(shard_dir, args.quarantine_dir)
    if args.restore:
        return do_restore(shard_dir, args.quarantine_dir)

    paths = iter_shard_paths(shard_dir)
    verdicts = [judge(p) for p in paths]
    _report(verdicts, _reservation_index([v for v in verdicts if v.reject], paths))
    if not args.apply:
        print("\nDRY RUN — nothing moved. Re-run with --apply --quarantine-dir DIR.")
        return 0
    return do_apply(shard_dir, args.quarantine_dir, verdicts, paths)


if __name__ == "__main__":
    sys.exit(main())
