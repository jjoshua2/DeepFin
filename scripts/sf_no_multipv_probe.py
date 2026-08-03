#!/usr/bin/env python3
"""Positive/negative control for the always-on SF-label contamination column.

    PYTHONPATH=. nice -n 19 python3 scripts/sf_no_multipv_probe.py \
      --poisoned-dir data/desync_quarantine_20260801 \
      --clean-dir '<trial>/replay_shards' --clean-index-range 33118:33387 \
      --limit 24 --batch-size 512

``TrainMetrics.sf_labelled_no_multipv_frac`` reads exactly 0.000000 on healthy
data, which is what makes "any non-zero is an incident" an honest alert rule and
not a threshold argument. A metric that is only ever exercised on healthy data
is indistinguishable from a constant, so this drives the SAME code path over
known-poisoned shards and requires it to fire.

It is not the offline gate re-implemented. ``scripts/quarantine_desync_shards.py``
and ``eval/value_optimism.py::sf_multipv_missing_rate`` read the zarr directly;
this runs the PRODUCTION path — shard on disk → ``Trainer._prepare_host_arrays``
(including the payload prune that would drop the presence flag) →
``collate_arrays`` → ``compute_loss`` → ``_extract_loss_scalars`` →
``_loss_sums_to_metric_kwargs`` → ``TrainMetrics`` — and reads the number off the
dataclass the reporter publishes. Agreement between the two is the point: it is
what licenses reading the live column against the gate's per-shard numbers.

Read-only. Shards are opened ``mode="r"`` through the normal loader and nothing
is written, moved or deleted.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.replay.shard import densify_chunk, load_shard_arrays, shard_index
from chess_anti_engine.train.trainer import Trainer

# Pre-registered in docs/experiment_ledger.md (2026-08-01 "an ALWAYS-ON
# no-MultiPV column"). Here rather than in the ledger alone so the run enforces
# the bar it was launched under instead of the reader eyeballing it afterwards.
POISONED_MIN = 0.19
POISONED_MAX = 0.23
CHECKED_MIN = 0.5

# The VALUE half (2026-08-03, `sf_eval_pv_orphan_frac`). Same construction, own
# band: the value check runs on the rows the POLICY check passes, so its
# poisoned reading is a different number, not a copy of the one above.
# Pre-registered off the direct zarr scan banked in the ledger — quarantined
# 0.119118, policy-clean post-quarantine 0.000032.
ORPHAN_POISONED_MIN = 0.10
ORPHAN_POISONED_MAX = 0.14
# ⚑ NOT "exactly zero", unlike the policy arm, and the difference is the
# finding rather than a slack bar. The clean arm here is a RANGE of live
# shards, and the value check sees desync pass-through that the policy check
# structurally cannot — so a clean arm that reads exactly 0.0 on the policy
# column can still hold burst-edge residue on this one (33 rows over the 640
# policy-clean window shards, 3.2e-5). The bar is set an order of magnitude
# under the residue-free expectation and two orders under the poisoned band; a
# reading between them is a real signal about the "clean" arm, not a failure of
# the detector, and must be investigated rather than absorbed by widening this.
ORPHAN_CLEAN_MAX = 0.0005


@dataclass(frozen=True)
class ArmReading:
    """Row-weighted pooling of per-shard ``TrainMetrics`` over one arm.

    Pooled from COUNTS reconstructed out of the two published columns rather
    than from a second pass over the zarr: ``checked = checked_frac * rows`` and
    ``no_pv = no_multipv_frac * checked``. A mean of per-shard rates would
    weight a 40-row shard like a 2000-row one, and re-reading the arrays would
    measure something other than what the metric published.
    """

    shards: int = 0
    rows: float = 0.0
    checked: float = 0.0
    no_pv: float = 0.0
  # VALUE half, reconstructed the same way from its OWN published pair
  # (`sf_eval_pv_orphan_frac` over `sf_eval_pv_checked_frac`). Its checked
  # count is a different population from the policy one — rows carrying all
  # three blocks, versus rows carrying an SF eval — so it cannot share the
  # field above without silently rescaling one of the two rates.
    orphan_checked: float = 0.0
    orphan: float = 0.0

    def add(
        self, *, rows: int, no_multipv_frac: float, checked_frac: float,
        orphan_frac: float = 0.0, orphan_checked_frac: float = 0.0,
    ) -> ArmReading:
        checked = checked_frac * float(rows)
        orphan_checked = orphan_checked_frac * float(rows)
        return ArmReading(
            shards=self.shards + 1,
            rows=self.rows + float(rows),
            checked=self.checked + checked,
            no_pv=self.no_pv + no_multipv_frac * checked,
            orphan_checked=self.orphan_checked + orphan_checked,
            orphan=self.orphan + orphan_frac * orphan_checked,
        )

    @property
    def no_multipv_frac(self) -> float:
        return self.no_pv / self.checked if self.checked > 0.0 else 0.0

    @property
    def checked_frac(self) -> float:
        return self.checked / self.rows if self.rows > 0.0 else 0.0

    @property
    def orphan_frac(self) -> float:
        return self.orphan / self.orphan_checked if self.orphan_checked > 0.0 else 0.0

    @property
    def orphan_checked_frac(self) -> float:
        return self.orphan_checked / self.rows if self.rows > 0.0 else 0.0


class _SliceBuf:
    """Minimal ReplayBuffer stand-in for ``Trainer.eval_full_pass``."""

    def __init__(self, arrs: dict[str, np.ndarray], n: int) -> None:
        self._arrs = arrs
        self._n = int(n)
        self.rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return self._n

    def batch_row_bounds(self, bs: int) -> list[tuple[int, int]]:
        return [(i, min(i + bs, self._n)) for i in range(0, self._n, bs)]

    def rows_slice_arrays(self, start: int, stop: int) -> dict[str, np.ndarray]:
        return {k: np.array(v[start:stop], copy=True) for k, v in self._arrs.items()}


def _select_shards(
    directory: Path, *, index_range: tuple[int, int] | None, limit: int,
) -> list[Path]:
    """Shard paths for one arm, chosen by DIRECTORY and ID RANGE only.

    Deliberately not by anything derived from the reading. Selecting shards on
    "no-PV count == 0" would condition the negative control on its own outcome,
    which is the failure the memory note about controls is written from. An ID
    band is fixed before the run and is a property of when the shard was
    written, not of what it says.
    """
    paths = sorted(p for p in directory.glob("shard_*.zarr"))
    if index_range is not None:
        lo, hi = index_range
        paths = [p for p in paths if lo <= shard_index(p) <= hi]
    if limit > 0:
        # Evenly spread across the band rather than the first N, so one
        # contiguous run of shards cannot stand in for the whole range.
        step = max(1, len(paths) // limit)
        paths = paths[::step][:limit]
    return paths


def _read_arm(
    trainer: Trainer, paths: list[Path], *, batch_size: int, verbose: bool,
) -> ArmReading:
    reading = ArmReading()
    for path in paths:
        arrs, _meta = load_shard_arrays(path)
        dense = densify_chunk({k: np.asarray(v) for k, v in arrs.items()})
        # A zero-row shard is an index reservation `quarantine_desync_shards.py`
        # left behind, not data — it carries nothing into training and must not
        # dilute either arm. `shard_033951.zarr` is the current one; the guard
        # is on the row count read off disk, so it cannot be claimed by a shard
        # that actually holds rows.
        x = dense.get("x")
        rows = 0 if x is None else int(x.shape[0])
        if rows == 0:
            continue
        # The loader attaches scalar provenance arrays (policy encoding, shard
        # identity) alongside the per-row ones; a row slice can only take the
        # per-row ones.
        dense = {k: v for k, v in dense.items() if v.ndim > 0 and v.shape[0] == rows}
        metrics = trainer.eval_full_pass(
            cast(Any, _SliceBuf(dense, rows)), batch_size=batch_size,
        )
        reading = reading.add(
            rows=int(metrics.eval_rows),
            no_multipv_frac=float(metrics.sf_labelled_no_multipv_frac),
            checked_frac=float(metrics.sf_multipv_checked_frac),
            orphan_frac=float(metrics.sf_eval_pv_orphan_frac),
            orphan_checked_frac=float(metrics.sf_eval_pv_checked_frac),
        )
        if verbose:
            print(
                f"  {path.name}  rows={metrics.eval_rows:5d}"
                f"  no_multipv={metrics.sf_labelled_no_multipv_frac:.6f}"
                f"  checked={metrics.sf_multipv_checked_frac:.6f}"
                f"  eval_pv_orphan={metrics.sf_eval_pv_orphan_frac:.6f}"
                f"  orphan_checked={metrics.sf_eval_pv_checked_frac:.6f}",
            )
    return reading


def _index_range(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    lo, _, hi = text.partition(":")
    return int(lo), int(hi)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poisoned-dir", type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--poisoned-index-range", default=None, help="LO:HI, inclusive")
    ap.add_argument("--clean-index-range", default=None, help="LO:HI, inclusive")
    # Defaults to the WHOLE arm. A stride subsample was tried first and read
    # 0.182063 at --limit 24 against a whole-set truth of 0.207461: the
    # quarantined shards are highly heterogeneous (per-shard rate 0.013 to
    # 0.31), so 24 of 122 does not estimate the set mean to the precision the
    # pre-registered band assumed. Judging the criterion on the population the
    # target number describes is the fix; widening the band post-hoc is not.
    ap.add_argument("--limit", type=int, default=0, help="shards per arm; 0 = all")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    # The model is irrelevant to the columns under test — they are built from
    # the batch's presence flags, not from any output — but `compute_loss`
    # needs logits, so the smallest trunk that accepts a production shard is
    # what runs. `input_extra_features` defaults to v2_threats (175 planes),
    # which is what production shards carry.
    model = build_model(ModelConfig(embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False))
    trainer = Trainer(model, device="cpu", lr=1e-3)

    poisoned_paths = _select_shards(
        args.poisoned_dir,
        index_range=_index_range(args.poisoned_index_range),
        limit=args.limit,
    )
    clean_paths = _select_shards(
        args.clean_dir,
        index_range=_index_range(args.clean_index_range),
        limit=args.limit,
    )
    if not poisoned_paths or not clean_paths:
        print(
            f"FAIL: empty arm (poisoned={len(poisoned_paths)} clean={len(clean_paths)}); "
            "an empty arm passes every bar below and proves nothing",
        )
        return 1

    print(f"poisoned arm: {len(poisoned_paths)} shards from {args.poisoned_dir}")
    poisoned = _read_arm(trainer, poisoned_paths, batch_size=args.batch_size, verbose=args.verbose)
    print(f"clean arm:    {len(clean_paths)} shards from {args.clean_dir}")
    clean = _read_arm(trainer, clean_paths, batch_size=args.batch_size, verbose=args.verbose)

    print("")
    for name, arm in (("poisoned", poisoned), ("clean", clean)):
        print(
            f"{name:9s} shards={arm.shards:4d} rows={arm.rows:9.0f}"
            f"  sf_labelled_no_multipv_frac={arm.no_multipv_frac:.6f}"
            f"  sf_multipv_checked_frac={arm.checked_frac:.6f}"
            f"  sf_eval_pv_orphan_frac={arm.orphan_frac:.6f}"
            f"  sf_eval_pv_checked_frac={arm.orphan_checked_frac:.6f}",
        )

    checks = [
        (
            "positive control fires",
            POISONED_MIN <= poisoned.no_multipv_frac <= POISONED_MAX,
            f"{poisoned.no_multipv_frac:.6f} in [{POISONED_MIN}, {POISONED_MAX}]",
        ),
        (
            "negative control is EXACTLY zero",
            clean.no_multipv_frac == 0.0,
            f"{clean.no_multipv_frac!r} == 0.0",
        ),
        (
            "poisoned arm was actually inspected",
            poisoned.checked_frac > CHECKED_MIN,
            f"checked {poisoned.checked_frac:.6f} > {CHECKED_MIN}",
        ),
        (
            "clean arm was actually inspected",
            clean.checked_frac > CHECKED_MIN,
            f"checked {clean.checked_frac:.6f} > {CHECKED_MIN}",
        ),
        (
            "VALUE-half positive control fires",
            ORPHAN_POISONED_MIN <= poisoned.orphan_frac <= ORPHAN_POISONED_MAX,
            f"{poisoned.orphan_frac:.6f} in "
            f"[{ORPHAN_POISONED_MIN}, {ORPHAN_POISONED_MAX}]",
        ),
        (
            "VALUE-half negative control is near zero",
            clean.orphan_frac <= ORPHAN_CLEAN_MAX,
            f"{clean.orphan_frac:.6f} <= {ORPHAN_CLEAN_MAX}",
        ),
        (
            "VALUE-half poisoned arm was actually inspected",
            poisoned.orphan_checked_frac > CHECKED_MIN,
            f"orphan_checked {poisoned.orphan_checked_frac:.6f} > {CHECKED_MIN}",
        ),
        (
            "VALUE-half clean arm was actually inspected",
            clean.orphan_checked_frac > CHECKED_MIN,
            f"orphan_checked {clean.orphan_checked_frac:.6f} > {CHECKED_MIN}",
        ),
    ]
    print("")
    for label, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
