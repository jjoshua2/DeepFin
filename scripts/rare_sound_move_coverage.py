#!/usr/bin/env python3
"""Rare-sound-move coverage: does the training target give mass to sound moves
the network's own prior neglects?

WHY THIS EXISTS. The 2026-08-09 pre-registration bundles two knobs of the SAME
improved-policy expression, ``softmax(log_prior/T + sigma*Qbar)`` with
``sigma = c_scale*(c_visit + max_visit)``:

  * ``gumbel_c_scale``   0.025 -> 0.1   (scales sigma)
  * ``gumbel_policy_temp`` 1.0 -> 1.5   (divides the prior logits)

The bundle is only attributable post-hoc if some statistic separates them. The
two the prereg already uses do not: ``KL(target||prior)`` rises under BOTH, and
``H(target)`` moves in opposite directions, so a flat entropy readout cannot
distinguish "neither knob fired" from "both fired and cancelled". This script
supplies a third axis, chosen because its MECHANISM keys off T alone.

WHAT IT MEASURES. Over rows where Stockfish scored the position:

    coverage(tau, rho, phi) =
        #{m : sound(m; tau) AND rare(m; rho) AND target(m) >= phi}
        ---------------------------------------------------------
        #{m : sound(m; tau) AND rare(m; rho)}

    sound(m; tau)  SF scored m and its cp-regret vs SF's best is <= tau.
    rare(m; rho)   the network's own prior gives m less than rho.
    phi            a floor of mass in the STORED policy target.

The denominator is a population of moves, not of positions, and it is reported
in every cell: a coverage number with a 12-move denominator is not a number.

THE REFERENCE PRIOR IS ALWAYS UNTEMPERED (T = 1). ``rare`` is defined against
the network's raw policy, never against the tempered prior the search used.
Defining rarity on the tempered prior would make the metric move under T by
construction -- the ruler would be made of the thing it measures.

⚑ COVERAGE IS **NOT** FLAT IN ``c_scale`` ANYWHERE ITS OWN CONTROL PASSES.
The prereg's table asserts flatness. That would hold if the stored target were
supported on the Gumbel CANDIDATE set, which only ``policy_temp`` moves -- but
``mcts/gumbel.py::_build_improved_policy_for_board`` (and the production C
driver) build the improved policy over EVERY legal move, handing unvisited
moves the root's completed-Q. So sigma reaches every entry of the target and no
mass floor ``phi`` is sigma-invariant.

⚑⚑ AND THE FLOOR THAT LOOKS FLATTEST IS THE FLOOR WHERE THE METRIC INVERTS.
Raising ``phi`` makes the ``c_scale`` term smaller, but past a crossing the
statistic stops measuring what its name says: the shuffle CONTROL beats the
real value, i.e. sound-and-rare moves clear the floor LESS often than a random
legal move in the same position. The two crossings are ordered against us --
the control inverts BELOW the ``c_scale`` zero-crossing -- so there is no cell
that is both honest and flat. This was found by a reviewer, on the cell the
ledger had pinned, and it is why every cell now carries its own control.

Measured on the LIVE rig (``--mode research``: 600 real shard rows re-searched
at 256 sims / topk 32 / gumbel_scale 0.5, position-paired, tau=25cp, rho=0.01;
95% cluster bootstrap over positions; control = 16-seed target shuffle):

    phi    cov   ctrl(SD)  verdict    c_scale 0.025->0.1   policy_temp 1.0->1.5
    1e-4  0.867   +13.98   PASS       -0.196 [-0.250,-0.147] +0.109 [+0.072,+0.145]
    1e-3  0.606    +9.04   PASS       -0.183 [-0.233,-0.135] +0.286 [+0.234,+0.338]
    3e-3  0.367    +1.37   PASS       -0.149 [-0.203,-0.098] +0.367 [+0.315,+0.422]
    5e-3  0.186    -6.02   INVERTED   -0.044 [-0.080,-0.006] +0.441 [+0.387,+0.500]
    7e-3  0.115    -8.33   INVERTED   -0.012 [-0.044,+0.018] +0.422 [+0.366,+0.479]
    1e-2  0.040    -9.53   INVERTED   +0.031 [+0.006,+0.058] +0.413 [+0.356,+0.469]

**phi = 1e-2 at rho = 0.01 is UNSAFE and is not a pin.** Its control runs -9.5
SD the wrong way and its ``c_scale`` term is POSITIVE, so a rise there is
produced by either knob and attributes to neither.

WHAT SURVIVES: A ONE-SIDED READING AT A CONTROL-PASSING CELL. Searching all 128
(tau, rho, phi) cells with >= 50 pairs on the live rig: 81 pass the control, 39
are c_scale-quiet, and **2 are both** -- ``rho = 0.05, phi = 2e-2`` at tau 25
and 50. The best of them (largest denominator, strongest control):

    tau=50, rho=0.05, phi=2e-2   pairs 928
      control    stored +5.88 SD / re-searched +2.28 SD          PASS
      c_scale    -0.0593 [-0.0833, -0.0350]     <- LOWERS coverage
      policy_temp +0.2381 [+0.2096, +0.2667]    <- RAISES it
      bundle      +0.0765 [+0.0469, +0.1058]

``c_scale`` is not flat there either (|d_c| is a quarter of |d_T|), but its SIGN
is protective: ``c_scale`` alone can only push coverage DOWN, so a POSITIVE
bundle delta is reachable only via ``policy_temp``. That asymmetry, not
flatness, is the whole attribution:

  * ``delta >= +0.05`` at that cell  ==>  ``policy_temp`` fired.
  * ``delta <= 0``  ==>  **INDETERMINATE, never "policy_temp did not fire".**
    The halves have opposite signs and partially cancel -- the measured bundle
    (+0.077) is only a third of the ``policy_temp`` half (+0.238).

Read it with these caveats or not at all: at rho = 0.05 "rare" is barely below
uniform (1/27 ~ 0.037), the two crossings move with sim count and position mix,
and the arms must be re-run at the deployed shape before the numbers are reused.

⚑ THE POPULATION MOVES WITH THE KNOB. ``diff_focus`` drops ~20% of policy rows
with a keep probability that rises in ``KL(prior||improved)`` -- exactly what
the bundle raises -- so the rows that stop being discarded are the lowest-KL,
lowest-coverage ones. ``--progress-csv`` records ``diff_focus_keep_rate`` and
``keep_limited_frac`` with every readout; a delta taken across a change in them
is confounded and must be reported as such. ``assert_population`` cannot see
this: the ``sf_p0_regret`` RATE is taken inside the same population and is
invariant to a uniform keep_prob shift.

THREE MODES, ONE METRIC.

``--mode shards`` (the production readout)
    Reads LIVE replay shards by absolute path, newest-first BY MTIME, and
    prints the newest shard's basename and mtime. There is a dead
    ``runs/pbt2_small/replay_shards/`` whose newest file is from 2026-04-14;
    selecting by sorted filename or by shard index lands in it and every
    conclusion drawn from it is wrong (ledger 2026-08-09, twice in one day).
    The prior comes from a forward pass on the shard's STORED ``x`` planes, so
    no board decoding and no history loss are involved. The SF view of the row
    is ``sf_p0_regret`` -- NOT ``sf_multipv_raw``, which describes a different
    position (verified here: see ``--check-alignment``).

``--mode research`` (the isolation arms, ON THE LIVE POPULATION)
    The same live rows, re-searched at an explicit ``--c-scale`` /
    ``--policy-temp`` / ``--gumbel-scale``. Coverage is a bounded, steeply
    non-linear function of phi, so derivatives measured on one rig do not
    transfer to another operating point -- the arms have to run where the
    readout runs. Fidelity against production's STORED target is printed every
    run and is the gate: measured 0.81 argmax agreement / 0.14 mean TV at the
    live shape, with the residual coming from the frozen reference checkpoint
    and from the history planes a decoded board cannot carry.

``--mode simulate`` (frozen deep-SF audit positions)
    Same metric against the audit set's >=1M-node MultiPV labels. Better
    soundness labels than production's MultiPV 6, different position
    distribution -- useful for mechanism, NOT for transferring a level or a
    derivative to the live rig.

``--compare-to <banked.json>`` differences two arms with an exact row-key
equality check first; ``--out`` banks the per-row vectors so any cell can be
re-scored later without re-searching.

CONTROLS ARE ATTACHED TO EVERY CELL, ALWAYS. ``--control-seeds`` (default 12)
permutes the target within each row's legal moves and reports the shuffle's own
mean and SD beside the cell, plus ``(coverage - shuffled)/sd`` and a
PASS/null/INVERTED verdict. The shuffle distribution is the null; ``base_rate``
pools differently and is printed as indicative only. ``--shuffle prior``
permutes the prior instead, so ``rare`` becomes a size-matched random subset.
``tests/test_rare_sound_move_coverage.py`` ships both shuffles, the inversion,
and the cluster bootstrap as assertions.

Usage::

    PYTHONPATH=. python3 scripts/rare_sound_move_coverage.py \\
        --mode shards --checkpoint data/ruler_reads_20260808/trainer.pt \\
        --replay-dir /abs/path/to/<trial>/replay_shards --shards 40

    PYTHONPATH=. python3 scripts/rare_sound_move_coverage.py \\
        --mode research --checkpoint data/ruler_reads_20260808/trainer.pt \\
        --replay-dir /abs/path/to/<trial>/replay_shards --shards 12 \\
        --max-rows 600 --sims 256 --topk 32 --gumbel-scale 0.5 \\
        --c-scale 0.1 --policy-temp 1.0 --out arm_B.json \\
        --compare-to arm_A.json

Banked evidence for every number above:
``data/rare_sound_move_coverage/live_arms_20260809.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# `selfplay/finalize.py::_build_sf_p0_regret_vector` divides by this before
# storing, so a stored `sf_p0_regret` entry is a FRACTION of it, not a cp value.
SF_OWN_REGRET_CAP_CP = 1000.0

# Default sweeps. Every one of the three axes is swept rather than pinned: a
# single hardcoded cutoff is how an instrument here ends up measuring its own
# tuning (ledger: three rulers set to thresholds their mechanism could not reach).
DEFAULT_TAUS_CP: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0)
DEFAULT_RHOS: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)
DEFAULT_PHIS: tuple[float, ...] = (1e-4, 1e-3, 1e-2)

# Fields the shard reader needs on every selected row. `sf_p0_regret` is
# deliberately NOT here: it is intrinsically sparse (it needs two consecutive
# full plies in one selfplay game) and has its own floor, `--min-sf-p0-rate`.
REQUIRED_SHARD_FIELDS: tuple[str, ...] = (
    "x", "policy_target", "legal_mask", "has_policy", "is_network_turn",
    "sf_p0_regret", "has_sf_p0_regret",
)

# Presence floor for REQUIRED_SHARD_FIELDS on the selected population. A field
# that silently goes missing changes the denominator instead of raising.
FIELD_PRESENCE_FLOOR = 0.99


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowVectors:
    """One position, reduced to per-LEGAL-MOVE vectors. Encoding-agnostic.

    ``scored`` marks the moves Stockfish actually put a score on. Everything
    else is unknown-quality, not bad-quality, and can never be ``sound``.
    """

    prior: np.ndarray       # (L,) float64, sums to 1 over legal moves, UNTEMPERED
    target: np.ndarray      # (L,) float64, the stored/searched improved policy
    regret_cp: np.ndarray   # (L,) float64, cp behind SF's best (0 = SF's best)
    scored: np.ndarray      # (L,) bool
    # Stable identity of the underlying POSITION, so two arms can be proved to
    # describe the same rows before anything is differenced. A paired delta
    # between arms that silently drew different positions measures the draw.
    key: str = ""

    def __post_init__(self) -> None:
        n = int(self.prior.shape[0])
        for name in ("target", "regret_cp", "scored"):
            arr = getattr(self, name)
            if int(np.asarray(arr).shape[0]) != n:
                raise ValueError(
                    f"RowVectors: {name} has {np.asarray(arr).shape[0]} entries, "
                    f"prior has {n}"
                )


@dataclass
class CoverageCell:
    """One (tau, rho, phi) cell of the sweep, with its own denominator."""

    tau_cp: float
    rho: float
    phi: float
    n_pairs: int          # sound AND rare moves -- the denominator
    n_covered: int        # ... of which the target funds at >= phi
    n_rows: int           # positions contributing at least one pair
    coverage: float
    # ⚑ INDICATIVE ONLY, never the null. `base_rate` is `#legal>=phi / #legal`
    # pooled over ALL rows, while coverage is restricted to rows that HAVE a
    # sound-and-rare move and is weighted by their pair counts. The two pool
    # differently, so a gap between them is a pooling artefact and not evidence.
    # THE NULL IS `shuffled_mean` -- the shuffle's own distribution.
    base_rate: float
    ci_lo: float = float("nan")
    ci_hi: float = float("nan")
    # The negative control, evaluated AT THIS CELL. Never at one cell and
    # assumed for the rest: at phi=1e-2 the control inverts on live data while
    # at phi=1e-3 it passes, and reading one and pinning the other is how the
    # 2026-08-09 phi=1e-2 pin came to be made against an inverted control.
    shuffled_mean: float = float("nan")
    shuffled_sd: float = float("nan")
    control_seeds: int = 0

    @property
    def control_margin(self) -> float:
        """``coverage - shuffled_mean`` in units of the shuffle's own SD.

        Positive means the metric beats its null in the direction its name
        claims. Negative means the association it is named for runs BACKWARDS
        at this cell and no delta measured here can be attributed to soundness.
        """
        if not (math.isfinite(self.shuffled_mean) and self.shuffled_sd > 0.0):
            return float("nan")
        return (self.coverage - self.shuffled_mean) / self.shuffled_sd

    def passes_control(self, min_sds: float = 1.0) -> bool:
        m = self.control_margin
        return bool(math.isfinite(m) and m >= min_sds)


def _row_pair_counts(
    row: RowVectors, tau_cp: float, rho: float, phi: float,
) -> tuple[int, int, int, int]:
    """(pairs, covered, legal, legal_over_phi) for one row and one cell."""
    sound = row.scored & (row.regret_cp <= tau_cp)
    rare = row.prior < rho
    sel = sound & rare
    n_pairs = int(sel.sum())
    n_covered = int((sel & (row.target >= phi)).sum())
    return n_pairs, n_covered, int(row.prior.shape[0]), int((row.target >= phi).sum())


def coverage_cells(
    rows: list[RowVectors],
    *,
    taus_cp: tuple[float, ...] = DEFAULT_TAUS_CP,
    rhos: tuple[float, ...] = DEFAULT_RHOS,
    phis: tuple[float, ...] = DEFAULT_PHIS,
) -> list[CoverageCell]:
    """Pooled coverage over ``rows`` for the full (tau, rho, phi) grid."""
    out: list[CoverageCell] = []
    for tau in taus_cp:
        for rho in rhos:
            for phi in phis:
                pairs = covered = contributing = legal = over = 0
                for row in rows:
                    p, c, nl, no = _row_pair_counts(row, tau, rho, phi)
                    pairs += p
                    covered += c
                    legal += nl
                    over += no
                    contributing += 1 if p > 0 else 0
                out.append(CoverageCell(
                    tau_cp=float(tau), rho=float(rho), phi=float(phi),
                    n_pairs=pairs, n_covered=covered, n_rows=contributing,
                    coverage=(covered / pairs) if pairs else float("nan"),
                    base_rate=(over / legal) if legal else float("nan"),
                ))
    return out


def attach_controls(
    rows: list[RowVectors], cells: list[CoverageCell], *, seeds: int, seed0: int,
    what: str = "target",
) -> None:
    """Fill every cell's ``shuffled_mean`` / ``shuffled_sd`` IN PLACE.

    Run over the whole grid, not one cell. The control is cell-local: the
    association coverage is named for holds at a small mass floor and INVERTS at
    a large one, so a control read at one cell says nothing about another. This
    function exists so a cell can never be reported without its own null.
    """
    if seeds <= 0:
        return
    per_seed = np.empty((len(cells), int(seeds)), dtype=np.float64)
    for s in range(int(seeds)):
        sh = shuffled_rows(rows, what=what, seed=seed0 + 1000 * (s + 1))
        for i, cell in enumerate(cells):
            pairs = covered = 0
            for row in sh:
                p, c, _nl, _no = _row_pair_counts(row, cell.tau_cp, cell.rho, cell.phi)
                pairs += p
                covered += c
            per_seed[i, s] = (covered / pairs) if pairs else np.nan
    for i, cell in enumerate(cells):
        vals = per_seed[i][np.isfinite(per_seed[i])]
        cell.control_seeds = int(vals.size)
        cell.shuffled_mean = float(vals.mean()) if vals.size else float("nan")
        cell.shuffled_sd = (
            float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
        )


def bootstrap_ci(
    rows: list[RowVectors],
    *,
    tau_cp: float,
    rho: float,
    phi: float,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """(lo, hi, sd) for one cell by CLUSTER bootstrap over POSITIONS.

    Positions are the resampling unit, never moves: several sound-and-rare moves
    in one position share that position's target, so a move-level bootstrap
    would treat correlated draws as independent and report a CI several times
    too narrow.
    """
    per_row = np.empty((len(rows), 2), dtype=np.float64)
    for i, row in enumerate(rows):
        p, c, _nl, _no = _row_pair_counts(row, tau_cp, rho, phi)
        per_row[i, 0] = float(p)
        per_row[i, 1] = float(c)
    live = per_row[per_row[:, 0] > 0.0]
    if live.shape[0] < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = live.shape[0]
    idx = rng.integers(0, n, size=(int(resamples), n))
    pairs = live[idx, 0].sum(axis=1)
    covered = live[idx, 1].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = np.where(pairs > 0.0, covered / np.maximum(pairs, 1e-12), np.nan)
    stat = stat[np.isfinite(stat)]
    if stat.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.percentile(stat, 2.5)),
        float(np.percentile(stat, 97.5)),
        float(stat.std(ddof=1)),
    )


def assert_paired(rows_a: list[RowVectors], rows_b: list[RowVectors]) -> None:
    """Both arms must describe the SAME positions in the SAME order.

    The docstring promise that two arms are "checked for identical ordering
    before any delta is taken" was, in the first version of this script, only a
    promise: nothing read the keys. Every paired path now goes through here.
    """
    if len(rows_a) != len(rows_b):
        raise ValueError(
            f"paired arms must be equal-length, got {len(rows_a)} and {len(rows_b)}"
        )
    keys_a = [r.key for r in rows_a]
    keys_b = [r.key for r in rows_b]
    if any(k == "" for k in keys_a) or any(k == "" for k in keys_b):
        raise ValueError(
            "paired arms carry unset row keys; refusing to difference two arms "
            "that cannot be proved to describe the same positions"
        )
    if keys_a != keys_b:
        first = next(
            i for i, (a, b) in enumerate(zip(keys_a, keys_b, strict=True)) if a != b
        )
        raise ValueError(
            f"paired arms describe different positions: row {first} is "
            f"{keys_a[first]!r} in arm A and {keys_b[first]!r} in arm B"
        )


def paired_delta_ci(
    rows_a: list[RowVectors],
    rows_b: list[RowVectors],
    *,
    tau_cp: float,
    rho: float,
    phi: float,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """(delta, lo, hi) for coverage(B) - coverage(A) on POSITION-PAIRED rows.

    Both lists must describe the same positions in the same order -- the only
    admissible comparison between two search shapes, since an unpaired contrast
    is dominated by which positions each arm happened to draw. That is not a
    convention the caller is asked to honour: ``assert_paired`` enforces it on
    the row keys before anything is differenced.
    """
    assert_paired(rows_a, rows_b)
    stats = np.empty((len(rows_a), 4), dtype=np.float64)
    for i, (ra, rb) in enumerate(zip(rows_a, rows_b, strict=True)):
        pa, ca, _n, _o = _row_pair_counts(ra, tau_cp, rho, phi)
        pb, cb, _n2, _o2 = _row_pair_counts(rb, tau_cp, rho, phi)
        stats[i] = (pa, ca, pb, cb)
    live = stats[(stats[:, 0] > 0.0) | (stats[:, 2] > 0.0)]
    if live.shape[0] < 2:
        return float("nan"), float("nan"), float("nan")

    def _delta(sel: np.ndarray) -> np.ndarray:
        pa, ca, pb, cb = sel[..., 0], sel[..., 1], sel[..., 2], sel[..., 3]
        spa, sca = pa.sum(axis=-1), ca.sum(axis=-1)
        spb, scb = pb.sum(axis=-1), cb.sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (
                np.where(spb > 0, scb / np.maximum(spb, 1e-12), np.nan)
                - np.where(spa > 0, sca / np.maximum(spa, 1e-12), np.nan)
            )

    point = float(_delta(live))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, live.shape[0], size=(int(resamples), live.shape[0]))
    draws = _delta(live[idx])
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def shuffled_rows(
    rows: list[RowVectors], *, what: str, seed: int,
) -> list[RowVectors]:
    """Negative control: permute one vector WITHIN each row's legal moves.

    Within-row is the right scope. Permuting across rows would also swap the
    legal-set size and the position's difficulty, so a collapse would be
    explained by the mismatch rather than by the destroyed association.

    ``what='target'`` breaks target<->(sound, rare); coverage must fall to the
    base rate ``P(target >= phi)``. ``what='prior'`` breaks prior<->target, so
    ``rare`` becomes a size-matched random subset of legal moves.
    """
    if what not in ("target", "prior"):
        raise ValueError(f"shuffled_rows: what must be 'target' or 'prior', got {what!r}")
    rng = np.random.default_rng(seed)
    out: list[RowVectors] = []
    for row in rows:
        perm = rng.permutation(int(row.prior.shape[0]))
        if what == "target":
            out.append(RowVectors(
                prior=row.prior, target=row.target[perm],
                regret_cp=row.regret_cp, scored=row.scored, key=row.key,
            ))
        else:
            out.append(RowVectors(
                prior=row.prior[perm], target=row.target,
                regret_cp=row.regret_cp, scored=row.scored, key=row.key,
            ))
    return out


# ---------------------------------------------------------------------------
# Shard reading
# ---------------------------------------------------------------------------


@dataclass
class ShardSelection:
    """Which shard files were read, and when they were written."""

    paths: list[str]
    newest_basename: str
    newest_mtime: float
    oldest_mtime: float

    def describe(self) -> str:
        newest = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.newest_mtime))
        oldest = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.oldest_mtime))
        age_h = (time.time() - self.newest_mtime) / 3600.0
        return (
            f"{len(self.paths)} shards; newest {self.newest_basename} mtime {newest} "
            f"(age {age_h:.2f} h); oldest selected mtime {oldest}"
        )


def select_shards(replay_dir: str, n_shards: int) -> ShardSelection:
    """The newest ``n_shards`` by MTIME, not by name and not by shard index.

    Shard INDICES collide across lineages and sorted filenames do not track the
    live trial, so both of the obvious selectors can silently return a dead
    directory. The mtimes come back with the selection so the caller can print
    them and let a dead directory announce itself.
    """
    paths = glob.glob(os.path.join(replay_dir, "shard_*.zarr"))
    if not paths:
        raise SystemExit(f"no shard_*.zarr under {replay_dir!r}")
    paths.sort(key=os.path.getmtime)
    sel = paths[-int(n_shards):] if n_shards > 0 else paths
    return ShardSelection(
        paths=sel,
        newest_basename=os.path.basename(sel[-1]),
        newest_mtime=os.path.getmtime(sel[-1]),
        oldest_mtime=os.path.getmtime(sel[0]),
    )


def _scored_mask_from_regret(reg: np.ndarray) -> np.ndarray:
    """Which entries of a stored ``sf_p0_regret`` vector carry a real SF score.

    ``_build_sf_p0_regret_vector`` fills the whole vector with
    ``(worst_covered + 1)/2`` and then overwrites the MultiPV moves, so the
    fill is >= every covered value and is therefore the vector's maximum. An
    entry strictly below the maximum is a scored move. When a covered move
    already sat at the 1.0 cap the fill equals it and that move is dropped --
    a 1000cp-behind move is never ``sound`` at any threshold this script
    sweeps, so the loss is confined to the population it could not affect.
    """
    if reg.size == 0:
        return np.zeros((0,), dtype=bool)
    return reg < float(reg.max())


@dataclass
class ShardReadStats:
    """Population accounting for the shard reader -- printed, never hidden."""

    rows_total: int = 0
    rows_net_policy: int = 0
    rows_with_sf_p0: int = 0
    rows_used: int = 0
    # Every reason a selected row can still fail to become a RowVectors. Counted
    # so `rows_used` is never a number whose shortfall has no explanation.
    dropped_no_legal: int = 0
    dropped_degenerate_prior: int = 0
    dropped_undecodable: int = 0
    dropped_legal_mismatch: int = 0
    field_present: dict[str, int] = field(default_factory=dict)
    # Identity of the net that PRODUCED the shards, distinct from the frozen
    # reference checkpoint that supplies the prior. Two readouts taken against
    # different producing nets are not the same measurement.
    model_steps: list[int] = field(default_factory=list)
    model_sha_prefixes: list[str] = field(default_factory=list)


def assert_required_fields(z: object, path: str) -> None:
    """Every field in ``REQUIRED_SHARD_FIELDS`` must exist in the shard group.

    Separate from the reader so it can be exercised without a checkpoint: the
    abort is the guard, and a guard nobody can fire is not a guard.
    """
    missing = [k for k in REQUIRED_SHARD_FIELDS if k not in z]  # pyright: ignore[reportOperatorIssue]
    if missing:
        raise SystemExit(
            f"{os.path.basename(path)} is missing required field(s) {missing}; "
            "a dropped field changes the denominator silently -- aborting"
        )


def assert_encodings(
    attrs: dict[str, object], *, ck_hist: str, ck_pol: str, path: str,
) -> None:
    """The checkpoint and the shard must agree on how the planes are laid out.

    A mismatch would feed the network planes it never saw and every prior in
    the readout would be a plausible-looking number about a different position.
    """
    sh_hist = str(attrs.get("input_history_encoding", ""))
    sh_pol = str(attrs.get("policy_encoding", ""))
    if ck_hist and sh_hist and ck_hist != sh_hist:
        raise SystemExit(
            f"encoding mismatch: checkpoint input_history_encoding={ck_hist!r} "
            f"but {os.path.basename(path)} stores {sh_hist!r}"
        )
    if ck_pol and sh_pol and ck_pol != sh_pol:
        raise SystemExit(
            f"encoding mismatch: checkpoint policy_encoding={ck_pol!r} but "
            f"{os.path.basename(path)} stores {sh_pol!r}"
        )


def assert_population(stats: ShardReadStats, *, min_sf_p0_rate: float) -> None:
    """Abort on a denominator that moved, rather than reporting over it.

    ``sf_p0_regret`` needs two consecutive full plies inside one selfplay game,
    so it is intrinsically sparse -- 22.8% of net-turn policy rows measured on
    the live trial 2026-08-09. It is the RATE that must not collapse; the other
    fields are required on essentially every selected row.
    """
    if stats.rows_net_policy > 0:
        rate = stats.rows_with_sf_p0 / stats.rows_net_policy
        if rate < min_sf_p0_rate:
            raise SystemExit(
                f"`sf_p0_regret` present on only {100 * rate:.2f}% of net-turn policy "
                f"rows (floor {100 * min_sf_p0_rate:.2f}%). Baseline measured "
                "2026-08-09 on the live trial: 22.8%. A drop this large is a "
                "plumbing change, not a result -- aborting rather than reporting "
                "a metric over a silently different denominator"
            )
    for key in ("x", "legal_mask", "policy_target", "sf_p0_regret"):
        seen = stats.field_present.get(key, 0)
        if stats.rows_with_sf_p0 and seen < FIELD_PRESENCE_FLOOR * stats.rows_with_sf_p0:
            raise SystemExit(
                f"field {key!r} is finite on only "
                f"{100 * seen / stats.rows_with_sf_p0:.2f}% of selected rows "
                f"(floor {100 * FIELD_PRESENCE_FLOOR:.0f}%) -- aborting"
            )


def _compact_logits(pol: np.ndarray, compact_to_full: np.ndarray) -> np.ndarray:
    """Model policy output -> the shard's compact (lc0_1858) column space."""
    if pol.shape[1] == compact_to_full.shape[0]:
        return pol
    return pol[:, compact_to_full]


@dataclass(frozen=True)
class ResearchSpec:
    """Re-search live shard positions at an explicit shape.

    WHY THIS EXISTS. The audit-set rig (``--mode simulate``) prices the knobs on
    a different position distribution and lands at a different point of a
    steeply non-linear, bounded curve: live coverage falls 0.82 -> 0.13 across
    phi 1e-4 -> 1e-2, so a +0.38 delta measured on the audit rig is not even
    arithmetically available live. Derivatives do not transfer between rigs, and
    the isolation arms therefore have to be run on the LIVE population.

    The root is exact: the network is evaluated on the shard's own stored ``x``
    planes and those logits are handed to the search as ``pre_pol_logits`` /
    ``pre_wdl_logits``. Only the CHILDREN are re-encoded from a board decoded
    out of the planes, which has an empty move stack -- so their history planes
    are lost. ``fidelity_*`` reports the agreement between the re-searched
    target and the STORED one at the live shape, which is the gate: a harness
    that cannot reproduce the stored target is not measuring production.
    """

    shape: SimShape
    sims: int
    syzygy_path: str | None = None


@dataclass
class ResearchFidelity:
    """How well the re-search reproduces production's stored target."""

    n: int = 0
    argmax_agree: int = 0
    tv_sum: float = 0.0

    def report(self) -> str:
        if self.n == 0:
            return "fidelity: no rows scored"
        return (
            f"fidelity vs STORED target: argmax agreement "
            f"{self.argmax_agree / self.n:.4f}, mean TV {self.tv_sum / self.n:.4f} "
            f"(n={self.n})"
        )


def read_shard_rows(
    sel: ShardSelection,
    *,
    checkpoint: str,
    device: str,
    batch_size: int,
    max_rows: int,
    min_sf_p0_rate: float,
    stats: ShardReadStats,
    research: ResearchSpec | None = None,
    fidelity: ResearchFidelity | None = None,
) -> list[RowVectors]:
    """Live shard rows -> RowVectors, with the prior from the stored ``x``.

    The prior is a forward pass on the planes production searched with, so it
    is exact up to reference-checkpoint drift: nothing is decoded back to a
    board and no history is lost. The checkpoint's declared encodings are
    checked against each shard's, because a mismatch would feed the net planes
    it never saw.

    With ``research`` set, the ``target`` is replaced by a FRESH Gumbel search
    at that shape instead of the stored one, which is how the isolation arms are
    run on the live population.
    """
    import torch
    import zarr

    from chess_anti_engine.inference import LocalModelEvaluator
    from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY, POLICY_SIZE
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    if bool(getattr(model, "use_dynamic_relations", False)):
        raise SystemExit(
            "checkpoint uses dynamic relations; the stored `x` planes alone are "
            "not a sufficient input and this script would score a wrong prior"
        )
    ck_hist = str(getattr(model, "input_history_encoding", ""))
    ck_pol = str(getattr(model, "policy_encoding", ""))
    evaluator = LocalModelEvaluator(model, device=device)
    compact_to_full = np.asarray(COMPACT_TO_FULL_POLICY, dtype=np.int64)

    searcher = _ResearchRunner(research, device=device, evaluator=evaluator,
                               hist=ck_hist) if research is not None else None

    rows: list[RowVectors] = []
    for path in reversed(sel.paths):
        z = zarr.open(path, mode="r")
        assert_required_fields(z, path)
        attrs = dict(z.attrs)
        assert_encodings(attrs, ck_hist=ck_hist, ck_pol=ck_pol, path=path)
        if attrs.get("model_step") is not None:
            stats.model_steps.append(int(attrs["model_step"]))
        if attrs.get("model_sha256"):
            stats.model_sha_prefixes.append(str(attrs["model_sha256"])[:12])
        base = os.path.basename(path)

        has_policy = np.asarray(z["has_policy"][:]).astype(bool)
        is_net = np.asarray(z["is_network_turn"][:]).astype(bool)
        has_reg = np.asarray(z["has_sf_p0_regret"][:]).astype(bool)
        stats.rows_total += int(has_policy.shape[0])
        net_policy = has_policy & is_net
        stats.rows_net_policy += int(net_policy.sum())
        keep = net_policy & has_reg
        stats.rows_with_sf_p0 += int(keep.sum())
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            continue

        legal = np.asarray(z["legal_mask"][:][idx]).astype(bool)
        target = np.asarray(z["policy_target"][:][idx], dtype=np.float64)
        regret = np.asarray(z["sf_p0_regret"][:][idx], dtype=np.float64)
        xs = np.asarray(z["x"][:][idx], dtype=np.float32)

        for key, arr in (("legal_mask", legal), ("policy_target", target),
                         ("sf_p0_regret", regret)):
            ok = int(np.isfinite(np.asarray(arr, dtype=np.float64)).all(axis=1).sum())
            stats.field_present[key] = stats.field_present.get(key, 0) + ok
        stats.field_present["x"] = stats.field_present.get("x", 0) + int(
            np.isfinite(xs).all(axis=(1, 2, 3)).sum()
        )

        for start in range(0, idx.size, batch_size):
            sl = slice(start, start + batch_size)
            with torch.no_grad():
                pol, wdl = evaluator.evaluate_encoded(xs[sl])
            pol = np.asarray(pol, dtype=np.float32)
            logits = _compact_logits(np.asarray(pol, dtype=np.float64), compact_to_full)
            searched: dict[int, np.ndarray] | None = None
            if searcher is not None:
                searched = searcher.run(
                    xs[sl], pol, np.asarray(wdl, dtype=np.float32),
                    legal[sl], compact_to_full, POLICY_SIZE, stats,
                )
            for j in range(logits.shape[0]):
                k = start + j
                lm = legal[k]
                if not lm.any():
                    stats.dropped_no_legal += 1
                    continue
                lg = logits[j][lm]
                lg = lg - lg.max()
                e = np.exp(lg)
                s = float(e.sum())
                if not math.isfinite(s) or s <= 0.0:
                    stats.dropped_degenerate_prior += 1
                    continue
                tgt_full = target[k]
                if searched is not None:
                    if j not in searched:
                        continue
                    fresh = searched[j]
                    if fidelity is not None:
                        fidelity.n += 1
                        a = tgt_full[lm]
                        b = fresh[lm]
                        if a.size and int(np.argmax(a)) == int(np.argmax(b)):
                            fidelity.argmax_agree += 1
                        fidelity.tv_sum += 0.5 * float(np.abs(a - b).sum())
                    tgt_full = fresh
                reg_row = regret[k]
                scored_full = _scored_mask_from_regret(reg_row)
                rows.append(RowVectors(
                    prior=e / s,
                    target=tgt_full[lm],
                    regret_cp=reg_row[lm] * SF_OWN_REGRET_CAP_CP,
                    scored=scored_full[lm],
                    key=f"{base}:{int(idx[k])}",
                ))
                if max_rows and len(rows) >= max_rows:
                    break
            if max_rows and len(rows) >= max_rows:
                break
        if max_rows and len(rows) >= max_rows:
            break

    stats.rows_used = len(rows)
    assert_population(stats, min_sf_p0_rate=min_sf_p0_rate)
    if not rows:
        raise SystemExit("no usable rows: every selected row lacked a legal mask")
    return rows


class _ResearchRunner:
    """Decodes shard rows back to boards and re-searches them at one shape."""

    def __init__(
        self, spec: ResearchSpec, *, device: str, evaluator: object, hist: str,
    ) -> None:
        from chess_anti_engine.mcts.gumbel import GumbelConfig

        self.spec = spec
        self.device = device
        self.evaluator = evaluator
        self.hist = hist
        self.rng = np.random.default_rng(20260809)
        self.tb_probe = None
        if spec.syzygy_path:
            from chess_anti_engine.tablebase import SyzygyProbe

            self.tb_probe = SyzygyProbe(spec.syzygy_path)
        sh = spec.shape
        self.cfg = GumbelConfig(
            simulations=int(spec.sims), topk=int(sh.topk), temperature=0.0,
            policy_temp=float(sh.policy_temp), c_scale=float(sh.c_scale),
            c_visit=float(sh.c_visit), c_visit_root=float(sh.c_visit_root),
            c_scale_root=float(sh.c_scale_root), q_visit_exp=float(sh.q_visit_exp),
            q_visit_exp_root=float(sh.q_visit_exp_root),
            halving_div=int(sh.halving_div), add_noise=bool(sh.add_noise),
            gumbel_scale=float(sh.gumbel_scale), input_history_encoding=hist,
        )

    def run(
        self,
        xs: np.ndarray,
        pol: np.ndarray,
        wdl: np.ndarray,
        legal: np.ndarray,
        compact_to_full: np.ndarray,
        policy_size: int,
        stats: ShardReadStats,
    ) -> dict[int, np.ndarray]:
        """{batch index -> compact improved policy} for the decodable rows.

        A row is skipped when the planes do not decode to a valid position, or
        when the decoded board's legal moves disagree with the shard's stored
        ``legal_mask``. That second check is the one that matters: it proves the
        board handed to the search is the position the row is about, instead of
        assuming the plane decoder and the stored mask agree.
        """
        from chess_anti_engine.eval.audit import decode_board_from_planes
        from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
        from chess_anti_engine.moves import policy_batch_to_full_if_needed
        from chess_anti_engine.moves.encode import legal_move_indices

        boards = []
        keep_j: list[int] = []
        for j in range(xs.shape[0]):
            board = decode_board_from_planes(
                np.asarray(xs[j], dtype=np.float32), input_history_encoding=self.hist,
            )
            if board is None or board.is_game_over():
                stats.dropped_undecodable += 1
                continue
            full_idx = np.asarray(legal_move_indices(board), dtype=np.int64)
            compact_mask = np.zeros((legal.shape[1],), dtype=bool)
            lookup = np.full((policy_size,), -1, dtype=np.int64)
            lookup[compact_to_full] = np.arange(compact_to_full.shape[0])
            hit = lookup[full_idx]
            compact_mask[hit[hit >= 0]] = True
            if not np.array_equal(compact_mask, legal[j]):
                stats.dropped_legal_mismatch += 1
                continue
            boards.append(board)
            keep_j.append(j)
        if not boards:
            return {}

        full_pol = pol
        if full_pol.shape[1] != policy_size:
            full_pol = policy_batch_to_full_if_needed(
                full_pol, policy_encoding="lc0_1858", fill_value=-1e9,
            )
        sel_pol = np.ascontiguousarray(full_pol[np.asarray(keep_j)], dtype=np.float32)
        sel_wdl = np.ascontiguousarray(wdl[np.asarray(keep_j)], dtype=np.float32)
        result = run_gumbel_root_many_c(
            model=None, boards=boards, device=self.device, rng=self.rng,
            cfg=self.cfg, evaluator=self.evaluator,  # pyright: ignore[reportArgumentType]
            pre_pol_logits=sel_pol, pre_wdl_logits=sel_wdl,
            tb_probe=self.tb_probe, vloss_weight=int(self.spec.shape.vloss_weight),
            target_batch=0,
        )
        probs_b = result[0]
        out: dict[int, np.ndarray] = {}
        for i, j in enumerate(keep_j):
            out[j] = np.asarray(probs_b[i], dtype=np.float64)[compact_to_full]
        return out


def check_alignment(sel: ShardSelection, max_rows: int) -> dict[str, float]:
    """Prove, on real data, WHICH stored field describes the row's own position.

    ``sf_multipv_raw`` is queried at P1 -- after the row's move -- so its move
    indices belong to a different position; ``sf_p0_regret`` is built from the
    PREVIOUS record's MultiPV, which is Stockfish's read of THIS position. The
    test does not trust either name: it asks how often each field's moves are
    LEGAL in this row. The correct field must be at 1.0.
    """
    import zarr

    n_p0 = n_p0_legal = n_raw = n_raw_legal = 0
    rows_seen = 0
    for path in reversed(sel.paths):
        z = zarr.open(path, mode="r")
        if "sf_multipv_raw" not in z:
            continue
        has_policy = np.asarray(z["has_policy"][:]).astype(bool)
        is_net = np.asarray(z["is_network_turn"][:]).astype(bool)
        has_reg = np.asarray(z["has_sf_p0_regret"][:]).astype(bool)
        has_raw = np.asarray(z["has_sf_multipv_raw"][:]).astype(bool)
        legal = np.asarray(z["legal_mask"][:]).astype(bool)
        reg = np.asarray(z["sf_p0_regret"][:], dtype=np.float64)
        raw = np.asarray(z["sf_multipv_raw"][:])
        for i in np.nonzero(has_policy & is_net)[0]:
            rows_seen += 1
            if has_reg[i]:
                cov = np.nonzero(_scored_mask_from_regret(reg[i]))[0]
                n_p0 += int(cov.size)
                n_p0_legal += int(legal[i][cov].sum())
            if has_raw[i]:
                mv = raw[i][:, 0]
                mv = mv[mv >= 0].astype(np.int64)
                mv = mv[mv < legal.shape[1]]
                n_raw += int(mv.size)
                n_raw_legal += int(legal[i][mv].sum())
            if max_rows and rows_seen >= max_rows:
                break
        if max_rows and rows_seen >= max_rows:
            break
    return {
        "rows": float(rows_seen),
        "sf_p0_regret_moves": float(n_p0),
        "sf_p0_regret_legal_frac": (n_p0_legal / n_p0) if n_p0 else float("nan"),
        "sf_multipv_raw_moves": float(n_raw),
        "sf_multipv_raw_legal_frac": (n_raw_legal / n_raw) if n_raw else float("nan"),
    }


# ---------------------------------------------------------------------------
# Simulate mode: run the real search at an explicit shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimShape:
    """Every Gumbel field this script sets, named explicitly.

    Nothing falls back to a dataclass default silently: ``c_scale_root`` and
    ``q_visit_exp_root`` are SENTINELS that select a different root transform,
    and a shape that inherits them without saying so measures a search nobody
    runs. The live selfplay root is LINEAR because the trio is unset.
    """

    c_scale: float
    policy_temp: float
    topk: int = 32
    c_visit: float = 50.0
    c_visit_root: float = -1.0
    c_scale_root: float = -1.0
    q_visit_exp: float = 1.0
    q_visit_exp_root: float = 99.0
    halving_div: int = 2
    vloss_weight: int = 1
    add_noise: bool = True
    gumbel_scale: float = 1.0

    def root_sigma_span(self, max_visit: float) -> float:
        """Realized root sigma span in nats, with the sentinels resolved.

        Mirrors ``gumbel._root_sigma_scale`` and ``_mcts_tree.c``'s
        ``gss_score_and_halve``.
        """
        cvr = self.c_visit_root if self.c_visit_root >= 0.0 else self.c_visit
        csr = self.c_scale_root if self.c_scale_root >= 0.0 else self.c_scale
        qer = self.q_visit_exp_root if self.q_visit_exp_root < 90.0 else self.q_visit_exp
        if qer < 0.0:
            return csr * math.log1p(cvr + max_visit)
        mv = max_visit if qer == 1.0 else max_visit**qer
        return csr * (cvr + mv)


def simulate_rows(
    *,
    checkpoint: str,
    audit_set: str,
    positions: int,
    sims: int,
    shape: SimShape,
    device: str,
    batch_size: int,
    seed: int,
    syzygy_path: str | None = None,
) -> tuple[list[RowVectors], list[str]]:
    """Search real positions at ``shape``; score coverage on what search returns.

    Returns (rows, fens). The fens are returned so two arms can be paired
    position-by-position and checked for identical ordering before any delta is
    taken -- an unpaired arm contrast measures the draw, not the knob.
    """
    import chess
    import torch

    from chess_anti_engine.eval.audit import (
        legal_full_indices,
        load_audit_set,
        move_regrets,
    )
    from chess_anti_engine.inference import LocalModelEvaluator
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
    from chess_anti_engine.moves import POLICY_SIZE, policy_batch_to_full_if_needed
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    all_pos = load_audit_set(audit_set)
    rng = np.random.default_rng(seed)
    if positions and positions < len(all_pos):
        take = sorted(rng.choice(len(all_pos), size=int(positions), replace=False).tolist())
        all_pos = [all_pos[int(i)] for i in take]

    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    if bool(getattr(model, "use_dynamic_relations", False)):
        raise SystemExit("dynamic-relation checkpoints are out of scope for this script")
    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    pol_enc = str(getattr(model, "policy_encoding", "lc0_1858"))
    evaluator = LocalModelEvaluator(model, device=device)

    cfg = GumbelConfig(
        simulations=int(sims), topk=int(shape.topk), temperature=0.0,
        policy_temp=float(shape.policy_temp), c_scale=float(shape.c_scale),
        c_visit=float(shape.c_visit), c_visit_root=float(shape.c_visit_root),
        c_scale_root=float(shape.c_scale_root), q_visit_exp=float(shape.q_visit_exp),
        q_visit_exp_root=float(shape.q_visit_exp_root),
        halving_div=int(shape.halving_div), add_noise=bool(shape.add_noise),
        gumbel_scale=float(shape.gumbel_scale), input_history_encoding=hist,
        input_extra_features=extra, policy_encoding=pol_enc,
    )

    from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard

    tb_probe = None
    if syzygy_path:
        from chess_anti_engine.tablebase import SyzygyProbe

        tb_probe = SyzygyProbe(syzygy_path)

    boards = [chess.Board(p.fen) for p in all_pos]
    rows: list[RowVectors] = []
    fens: list[str] = []
    search_rng = np.random.default_rng(seed)
    eff_batch = max(4, min(int(batch_size), 8192 // max(1, int(sims))))
    for start in range(0, len(boards), eff_batch):
        chunk = boards[start:start + eff_batch]
        chunk_pos = all_pos[start:start + eff_batch]
        xs = np.stack([
            encode_cboard(
                CBoard.from_board(b), input_history_encoding=hist,
                input_extra_features=extra,
            )
            for b in chunk
        ])
        with torch.no_grad():
            pol_logits, _wdl = evaluator.evaluate_encoded(xs)
        pol_logits = np.asarray(pol_logits, dtype=np.float32)
        if pol_logits.shape[1] != POLICY_SIZE:
            pol_logits = policy_batch_to_full_if_needed(
                pol_logits, policy_encoding=pol_enc, fill_value=-1e9,
            )
        result = run_gumbel_root_many_c(
            model=None, boards=list(chunk), device=device, rng=search_rng,
            cfg=cfg, evaluator=evaluator, tb_probe=tb_probe,
            vloss_weight=int(shape.vloss_weight), target_batch=0,
        )
        probs_b = result[0]
        for j, (pos, board) in enumerate(zip(chunk_pos, chunk, strict=True)):
            ucis, idxs = legal_full_indices(board)
            if not ucis:
                continue
            lg = np.asarray(pol_logits[j][idxs], dtype=np.float64)
            lg = lg - lg.max()
            e = np.exp(lg)
            s = float(e.sum())
            if not math.isfinite(s) or s <= 0.0:
                continue
            tgt = np.asarray(probs_b[j], dtype=np.float64)[idxs]
            rows.append(RowVectors(
                prior=e / s,
                target=tgt,
                regret_cp=move_regrets(pos, ucis),
                scored=np.array([u in pos.move_cp for u in ucis], dtype=bool),
                key=pos.key,
            ))
            fens.append(pos.fen)
    return rows, fens


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_cells(cells: list[CoverageCell], *, title: str) -> None:
    """Every cell with its OWN negative control beside it.

    ``CONTROL`` is ``(coverage - shuffled) / sd_shuffled``. ``PASS`` means the
    metric beats its null in the direction its name claims; ``INVERTED`` means
    the shuffle scores HIGHER, i.e. at that cell soundness enters with a
    negative sign and no delta measured there can be read as sound-move
    coverage. Cells are never reported without this column.
    """
    print(f"\n=== {title} ===")
    print(f"{'tau_cp':>7} {'rho':>7} {'phi':>8} {'pairs':>7} {'rows':>6} "
          f"{'coverage':>9} {'95% CI':>17} {'shuffled':>9} {'sd':>7} "
          f"{'CONTROL':>9} {'verdict':>9} {'chance*':>8}")
    for c in cells:
        ci = (
            f"[{c.ci_lo:.4f},{c.ci_hi:.4f}]"
            if math.isfinite(c.ci_lo) else "                 "
        )
        cov = f"{c.coverage:.4f}" if math.isfinite(c.coverage) else "     nan"
        sh = f"{c.shuffled_mean:.4f}" if math.isfinite(c.shuffled_mean) else "      --"
        sd = f"{c.shuffled_sd:.4f}" if math.isfinite(c.shuffled_sd) else "     --"
        margin = c.control_margin
        marg = f"{margin:+9.2f}" if math.isfinite(margin) else "       --"
        if not math.isfinite(margin):
            verdict = "no-ctrl"
        elif margin >= 1.0:
            verdict = "PASS"
        elif margin <= -1.0:
            verdict = "INVERTED"
        else:
            verdict = "null"
        print(f"{c.tau_cp:7.0f} {c.rho:7.3f} {c.phi:8.1e} {c.n_pairs:7d} "
              f"{c.n_rows:6d} {cov:>9} {ci:>17} {sh:>9} {sd:>7} {marg} "
              f"{verdict:>9} {c.base_rate:8.4f}")
    print("* `chance` is a POOLING-MISMATCHED reference and is indicative only; "
          "the null is `shuffled`.")


def print_paired(
    rows_a: list[RowVectors],
    rows_b: list[RowVectors],
    *,
    cells: list[CoverageCell],
    resamples: int,
    seed: int,
    label_a: str,
    label_b: str,
) -> list[dict[str, object]]:
    """Paired A->B deltas per cell, carrying each cell's control verdict along.

    A delta at a cell whose control is INVERTED is printed with its verdict
    rather than suppressed, because the number is still real -- it just cannot
    be attributed to sound-move coverage.
    """
    assert_paired(rows_a, rows_b)
    print(f"\n=== paired delta {label_a} -> {label_b} "
          f"({len(rows_a)} paired positions) ===")
    print(f"{'tau_cp':>7} {'rho':>7} {'phi':>8} {'cov_A':>8} {'cov_B':>8} "
          f"{'delta':>9} {'95% CI':>19} {'ctrl_A':>8}")
    out: list[dict[str, object]] = []
    for c in cells:
        a = coverage_cells(
            rows_a, taus_cp=(c.tau_cp,), rhos=(c.rho,), phis=(c.phi,),
        )[0].coverage
        b = coverage_cells(
            rows_b, taus_cp=(c.tau_cp,), rhos=(c.rho,), phis=(c.phi,),
        )[0].coverage
        d, lo, hi = paired_delta_ci(
            rows_a, rows_b, tau_cp=c.tau_cp, rho=c.rho, phi=c.phi,
            resamples=resamples, seed=seed,
        )
        m = c.control_margin
        ctrl = f"{m:+8.2f}" if math.isfinite(m) else "      --"
        print(f"{c.tau_cp:7.0f} {c.rho:7.3f} {c.phi:8.1e} {a:8.4f} {b:8.4f} "
              f"{d:+9.4f} [{lo:+.4f},{hi:+.4f}] {ctrl}")
        out.append({
            "tau_cp": c.tau_cp, "rho": c.rho, "phi": c.phi,
            "coverage_a": a, "coverage_b": b, "delta": d, "ci_lo": lo, "ci_hi": hi,
            "control_margin_a": m,
        })
    return out


def read_diff_focus(progress_csv: str) -> dict[str, float]:
    """Last row's `diff_focus_keep_rate` / `keep_limited_frac` from progress.csv.

    ⚑ THE POPULATION MOVES WITH THE KNOB. `diff_focus` drops policy rows with
    probability `max(df_min, min(1, difficulty*df_slope))` where `difficulty`
    includes `kl * df_pol_scale` and `kl` is `KL(prior||improved)` -- exactly
    what the bundle is expected to raise. So raising search authority ALSO
    re-admits the lowest-KL rows, which are the lowest-coverage rows. A coverage
    delta taken across a change in these two numbers is confounded by
    re-composition of the stored population, and `assert_population`'s
    `sf_p0_regret` RATE cannot see it: that ratio is taken inside the same
    population and is invariant to a uniform keep_prob shift.

    Returned so both readouts can record it and the delta can be refused when it
    moved. Empty dict when the file or the columns are absent -- reported as
    UNMEASURED rather than silently treated as unchanged.
    """
    import csv

    wanted = ("diff_focus_keep_rate", "diff_focus_keep_limited_frac",
              "diff_focus_keep_prob_mean", "training_iteration")
    try:
        with open(progress_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return {}
    for row in reversed(rows):
        vals: dict[str, float] = {}
        for k in wanted:
            raw = row.get(k, "")
            try:
                vals[k] = float(raw)
            except (TypeError, ValueError):
                continue
        if "diff_focus_keep_rate" in vals:
            return vals
    return {}


def _parse_floats(spec: str) -> tuple[float, ...]:
    return tuple(float(v) for v in spec.split(",") if v.strip())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", choices=("shards", "research", "simulate"),
                    required=True,
                    help="shards: score production's STORED target. research: "
                         "re-search the same live rows at an explicit shape "
                         "(the isolation arms, on the live population). "
                         "simulate: search frozen deep-SF audit positions.")
    ap.add_argument("--checkpoint", required=True,
                    help="trainer.pt / checkpoint dir supplying the REFERENCE PRIOR")
    ap.add_argument("--replay-dir", default=None,
                    help="ABSOLUTE path to the LIVE trial's replay_shards (mode=shards)")
    ap.add_argument("--shards", type=int, default=40,
                    help="how many shards to read, newest-by-MTIME (mode=shards)")
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--min-sf-p0-rate", type=float, default=0.05,
                    help="abort if sf_p0_regret is present on fewer than this "
                         "fraction of net-turn policy rows (baseline 0.228)")
    ap.add_argument("--progress-csv", default=None,
                    help="the trial's progress.csv; its diff_focus_keep_rate / "
                         "keep_limited_frac are recorded with the readout because "
                         "the stored population moves with the knob being measured")
    ap.add_argument("--audit-set", default="data/audit_set_v1.jsonl")
    ap.add_argument("--positions", type=int, default=400)
    ap.add_argument("--sims", type=int, default=256)
    ap.add_argument("--c-scale", type=float, default=0.025)
    ap.add_argument("--policy-temp", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=32)
    # Live realized values, not dataclass defaults: gumbel_scale decays
    # 1.0 -> 0.5 across moves 12-15 and 88.9% of stored rows are at ply >= 30,
    # so 1.0 is the OPENING regime and 0.5 is the regime the data comes from.
    ap.add_argument("--gumbel-scale", type=float, default=1.0,
                    help="root Gumbel perturbation scale; live decays to 0.5 from "
                         "move 15, which is the regime ~89%% of stored rows sit in")
    ap.add_argument("--c-visit", type=float, default=50.0)
    ap.add_argument("--halving-div", type=int, default=2)
    ap.add_argument("--vloss-weight", type=int, default=1)
    ap.add_argument("--syzygy-path", default=None,
                    help="tablebase dir; live runs syzygy_in_search: true")
    ap.add_argument("--no-noise", action="store_true",
                    help="disable the root Gumbel perturbation (selfplay runs it ON)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--taus", default=",".join(str(v) for v in DEFAULT_TAUS_CP))
    ap.add_argument("--rhos", default=",".join(str(v) for v in DEFAULT_RHOS))
    ap.add_argument("--phis", default=",".join(str(v) for v in DEFAULT_PHIS))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--control-seeds", type=int, default=12,
                    help="shuffle seeds per cell for the negative control; 0 "
                         "disables it, which means reporting a cell with no null")
    ap.add_argument("--shuffle", choices=("none", "target", "prior"), default="none",
                    help="score the SHUFFLED rows as the primary output (the "
                         "control is already attached to every cell without this)")
    ap.add_argument("--check-alignment", action="store_true",
                    help="mode=shards: prove which SF field describes this row")
    ap.add_argument("--compare-to", type=Path, default=None,
                    help="a banked --out JSON to treat as arm A; this run is arm "
                         "B. Row keys must match exactly or the run aborts.")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the cells + provenance + per-row vectors as JSON")
    return ap


def load_dump(path: Path) -> tuple[list[RowVectors], dict[str, object]]:
    """Re-materialise a banked ``--out`` dump as rows + its provenance."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [
        RowVectors(
            prior=np.asarray(r["prior"], dtype=np.float64),
            target=np.asarray(r["target"], dtype=np.float64),
            regret_cp=np.asarray(r["regret_cp"], dtype=np.float64),
            scored=np.asarray(r["scored"], dtype=bool),
            key=str(r.get("key", "")),
        )
        for r in payload["per_row"]
    ]
    return rows, dict(payload.get("provenance", {}))


def _build_shape(args: argparse.Namespace) -> SimShape:
    return SimShape(
        c_scale=float(args.c_scale), policy_temp=float(args.policy_temp),
        topk=int(args.topk), c_visit=float(args.c_visit),
        halving_div=int(args.halving_div), vloss_weight=int(args.vloss_weight),
        add_noise=not args.no_noise, gumbel_scale=float(args.gumbel_scale),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    taus = _parse_floats(args.taus)
    rhos = _parse_floats(args.rhos)
    phis = _parse_floats(args.phis)
    # `_scored_mask_from_regret` drops a covered move that already sat at the
    # 1000cp cap, which is only harmless while no threshold can reach it.
    bad = [t for t in taus if t >= SF_OWN_REGRET_CAP_CP]
    if bad:
        raise SystemExit(
            f"--taus {bad} is at or above SF_OWN_REGRET_CAP_CP="
            f"{SF_OWN_REGRET_CAP_CP:.0f}: at that threshold the `scored` mask is "
            "wrong, because capped moves are indistinguishable from the "
            "unscored fill. Use a threshold below the cap."
        )

    provenance: dict[str, object] = {
        "mode": args.mode, "checkpoint": args.checkpoint, "seed": args.seed,
        "shuffle": args.shuffle, "taus_cp": list(taus), "rhos": list(rhos),
        "phis": list(phis), "control_seeds": int(args.control_seeds),
    }
    if args.progress_csv:
        df = read_diff_focus(args.progress_csv)
        provenance["diff_focus"] = df
        if df:
            print(f"[diff_focus] iter {df.get('training_iteration', float('nan')):.0f} "
                  f"keep_rate {df.get('diff_focus_keep_rate', float('nan')):.4f} "
                  f"keep_limited_frac "
                  f"{df.get('diff_focus_keep_limited_frac', float('nan')):.4f} "
                  "-- a delta taken across a change in these is confounded by "
                  "population re-composition")
        else:
            print("[diff_focus] UNMEASURED (no readable progress.csv columns)")
    else:
        print("[diff_focus] UNMEASURED (--progress-csv not given); the stored "
              "population moves with the knob, so record it for any A/B")

    fens: list[str] = []
    if args.mode in ("shards", "research"):
        if not args.replay_dir:
            raise SystemExit(f"--mode {args.mode} requires --replay-dir (ABSOLUTE path)")
        sel = select_shards(args.replay_dir, args.shards)
        print(f"[shards] {args.replay_dir}")
        print(f"[shards] {sel.describe()}")
        provenance["replay_dir"] = args.replay_dir
        provenance["newest_shard"] = sel.newest_basename
        provenance["newest_shard_mtime"] = sel.newest_mtime
        if args.check_alignment:
            al = check_alignment(sel, max_rows=4000)
            print("[align] sf_p0_regret scored moves legal in THIS row: "
                  f"{al['sf_p0_regret_legal_frac']:.4f} (n={al['sf_p0_regret_moves']:.0f})")
            print("[align] sf_multipv_raw  moves  legal in THIS row: "
                  f"{al['sf_multipv_raw_legal_frac']:.4f} (n={al['sf_multipv_raw_moves']:.0f})")
            provenance["alignment"] = al
        research = None
        fid = None
        if args.mode == "research":
            shape = _build_shape(args)
            research = ResearchSpec(
                shape=shape, sims=int(args.sims), syzygy_path=args.syzygy_path,
            )
            fid = ResearchFidelity()
            provenance["shape"] = asdict(shape)
            provenance["sims"] = args.sims
            print(f"[research] re-searching LIVE rows at c_scale={shape.c_scale} "
                  f"T={shape.policy_temp} gumbel_scale={shape.gumbel_scale} "
                  f"topk={shape.topk} sims={args.sims} "
                  f"syzygy={'on' if args.syzygy_path else 'OFF'}")
            print(f"[research] root sigma span at max_visit=59: "
                  f"{shape.root_sigma_span(59.0):.3f} nats")
        stats = ShardReadStats()
        t0 = time.perf_counter()
        rows = read_shard_rows(
            sel, checkpoint=args.checkpoint, device=args.device,
            batch_size=args.batch_size, max_rows=args.max_rows,
            min_sf_p0_rate=args.min_sf_p0_rate, stats=stats,
            research=research, fidelity=fid,
        )
        print(f"[shards] rows total {stats.rows_total}, net-turn policy "
              f"{stats.rows_net_policy}, with sf_p0_regret {stats.rows_with_sf_p0} "
              f"({100 * stats.rows_with_sf_p0 / max(1, stats.rows_net_policy):.1f}%), "
              f"used {stats.rows_used}; dropped no-legal {stats.dropped_no_legal}, "
              f"degenerate-prior {stats.dropped_degenerate_prior}, "
              f"undecodable {stats.dropped_undecodable}, "
              f"legal-mismatch {stats.dropped_legal_mismatch}")
        if stats.model_steps:
            print(f"[shards] producing net model_step "
                  f"{min(stats.model_steps)}..{max(stats.model_steps)}; reference "
                  f"prior from {args.checkpoint}")
        if fid is not None:
            print(f"[research] {fid.report()} in {time.perf_counter() - t0:.1f}s")
            provenance["fidelity"] = asdict(fid)
        provenance["read_stats"] = asdict(stats)
        title = (f"coverage over {len(rows)} live shard rows"
                 + (" [RE-SEARCHED]" if research is not None else ""))
    else:
        shape = _build_shape(args)
        print(f"[sim] c_scale={shape.c_scale} policy_temp={shape.policy_temp} "
              f"topk={shape.topk} sims={args.sims} noise={shape.add_noise} "
              f"gumbel_scale={shape.gumbel_scale}")
        print(f"[sim] root sigma span at max_visit=59: "
              f"{shape.root_sigma_span(59.0):.3f} nats")
        t0 = time.perf_counter()
        rows, fens = simulate_rows(
            checkpoint=args.checkpoint, audit_set=args.audit_set,
            positions=args.positions, sims=args.sims, shape=shape,
            device=args.device, batch_size=args.batch_size, seed=args.seed,
            syzygy_path=args.syzygy_path,
        )
        print(f"[sim] {len(rows)} positions searched in {time.perf_counter() - t0:.1f}s")
        provenance["shape"] = asdict(shape)
        provenance["sims"] = args.sims
        provenance["fens"] = fens
        title = (f"coverage over {len(rows)} searched audit positions "
                 f"(c_scale={shape.c_scale}, T={shape.policy_temp})")

    if args.shuffle != "none":
        rows = shuffled_rows(rows, what=args.shuffle, seed=args.seed + 1)
        title += f"  [SHUFFLE {args.shuffle}]"

    cells = coverage_cells(rows, taus_cp=taus, rhos=rhos, phis=phis)
    for cell in cells:
        lo, hi, _sd = bootstrap_ci(
            rows, tau_cp=cell.tau_cp, rho=cell.rho, phi=cell.phi,
            resamples=int(args.bootstrap), seed=int(args.seed),
        )
        cell.ci_lo, cell.ci_hi = lo, hi
    attach_controls(rows, cells, seeds=int(args.control_seeds), seed0=int(args.seed))
    print_cells(cells, title=title)
    n_pass = sum(1 for c in cells if c.passes_control())
    n_inv = sum(1 for c in cells if math.isfinite(c.control_margin)
                and c.control_margin <= -1.0)
    print(f"[control] {n_pass}/{len(cells)} cells PASS, {n_inv} INVERTED")

    paired: list[dict[str, object]] = []
    if args.compare_to is not None:
        rows_a, prov_a = load_dump(args.compare_to)
        ck_a = str(prov_a.get("checkpoint", ""))
        if ck_a and ck_a != str(args.checkpoint):
            raise SystemExit(
                f"refusing to compare readouts taken against different reference "
                f"priors: arm A used {ck_a!r}, this run used {args.checkpoint!r}"
            )
        paired = print_paired(
            rows_a, rows, cells=cells, resamples=int(args.bootstrap),
            seed=int(args.seed), label_a=str(args.compare_to.name), label_b="this run",
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provenance": provenance,
            "n_rows": len(rows),
            "cells": [asdict(c) for c in cells],
            "paired": paired,
            "per_row": [
                {
                    "key": row.key,
                    "prior": row.prior.tolist(),
                    "target": row.target.tolist(),
                    "regret_cp": row.regret_cp.tolist(),
                    "scored": row.scored.astype(int).tolist(),
                }
                for row in rows
            ],
        }
        args.out.write_text(json.dumps(payload), encoding="utf-8")
        print(f"[out] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
