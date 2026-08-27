#!/usr/bin/env python3
"""Rare-sound-move coverage: does the training target give mass to sound moves
the network's own prior neglects?

⚑⚑⚑ READ THIS FIRST: THERE IS NO ATTRIBUTION RULE HERE, AND NO CELL OF THIS
SCRIPT MAY BE PINNED AS A DECISION RULE. This file was built to separate a
bundled ``gumbel_c_scale`` + ``gumbel_policy_temp`` change into per-knob
contributions. **IT CANNOT, AND THAT CLAIM IS RETRACTED** -- from this
docstring, from the PR body, and from the ledger (Amendment 10). The four
measurements that killed it are below under "WHY THE ATTRIBUTION DIED"; the
retraction is stated here, at the top, rather than after fifty lines that
presuppose it, because a reader who stops early must not leave with the
withdrawn claim.

⚑⚑ AND THE CELL THE LEDGER ONCE PINNED IS UNSAFE. At ``rho = 0.01``,
``phi = 1e-2`` this metric's OWN negative control INVERTS -- shuffling the
target RAISES coverage -- so at that cell sound-and-rare moves clear the floor
LESS often than a random legal move in the same position. **Do not restart onto
it and do not read a delta from it.** It was missed the first time because the
control was only ever evaluated at ``phi = 1e-3`` and assumed for the rest;
``attach_controls`` now fills a null and an interval for EVERY cell of the
swept grid, ``print_cells`` cannot emit a cell without them, and
``test_the_control_is_attached_to_every_cell_of_the_grid`` pins that.

WHAT IT IS FOR, THEN. Reporting coverage on a stated population with its own
null, its own interval, and its own bias printed beside every number -- so that
a future claim about the target's support can be judged.

WHY IT WAS BUILT. The 2026-08-09 pre-registration bundles two knobs of the SAME
improved-policy expression, ``softmax(log_prior/T + sigma*Qbar)`` with
``sigma = c_scale*(c_visit + max_visit)``:

  * ``gumbel_c_scale``   0.025 -> 0.1   (scales sigma)
  * ``gumbel_policy_temp`` 1.0 -> 1.5   (divides the prior logits)

The bundle would only be attributable post-hoc if some statistic separated
them. The two the prereg already uses do not: ``KL(target||prior)`` rises under
BOTH, and ``H(target)`` moves in opposite directions, so a flat entropy readout
cannot distinguish "neither knob fired" from "both fired and cancelled". This
script was written as a third axis on the theory that its MECHANISM keys off T
alone. ⚑ **That theory is the retracted one** -- see below; ``c_scale`` reaches
every entry of the target, so no mass floor is sigma-invariant.

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
mass floor ``phi`` is sigma-invariant. That mechanism finding stands and is the
reason the axis was worth building.

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

⚑⚑⚑ WHY THE ATTRIBUTION DIED (the retraction announced at the top of this
docstring, with its four measurements).

An earlier revision of this file recommended a ONE-SIDED rule at
``tau=50, rho=0.05, phi=2e-2``: ``c_scale`` lowers coverage there (-0.0593)
while ``policy_temp`` raises it (+0.2381), so a POSITIVE bundle delta was said
to be reachable only through ``policy_temp``. **That rule is RETRACTED** (PR
#382 re-review; ledger Amendment 10). Four measurements killed it, each
reproducible from this file:

1. **The instrument's bias is the size of the effect.** Arm A *is* production's
   shape, so ``stored - A`` is pure harness error. It is now computed and
   printed for every cell (``print_paired``'s ``biasA`` column). On 200 fresh
   live rows at production's shape it runs +0.1165 [+0.0485,+0.1887] at
   ``(25, 0.01, 1e-2)`` and +0.0100 [-0.0347,+0.0552] at the pinned cell --
   whose CI half-width alone (~0.045) is 76% of the ``|c_scale|`` effect
   (0.0593) whose SIGN was the entire attribution. A reviewer's independent
   200 rows put the pinned cell's bias at +0.0450 [+0.0036,+0.0886] and the
   600-row bank at +0.0571. Across the bank's 128 cells |bias| >= |c_scale
   effect| in 46.
2. **The two selection criteria are one variable.** ``attribution_scan``
   measures corr(control margin, ``|d_c|/|d_T|``) = **+0.785** over the bank
   and **+0.859** over the fresh 16-cell run. Soundness proxies Q and
   ``c_scale`` is the gain on Q, so a cell is quiet exactly where the control
   is weak. The two "survivors" sat at the 6th and 10th percentile of control
   strength among the 81 passers: the boundary, not a regime.
3. **The PASS had no resolution.** Row-resampling the pinned cell's control
   margin gives +2.06 +/- 1.39, 95% [-0.06, +5.18], with 22% of draws FAILING
   the >= +1.0 gate. The verdict is now read off that interval (see
   ``CoverageCell.verdict``), and on the fresh run the pinned cell reads
   **+3.15 [+0.41, +5.12] -> null**, not PASS. Under the shipped verdict
   ``attribution_scan`` finds **0** cells that are both control-PASS and
   ``c_scale``-quiet, against 2 under the old point-estimate gate.
4. **Every CI in this file is a POSITION-PAIRED CI.** They describe two search
   shapes scored on THE SAME ROWS. The live readout is pre-bundle shards versus
   post-bundle shards -- different positions, different games, and a population
   re-composed by ``diff_focus`` as a monotone function of the KL the bundle
   raises. None of these intervals transfer to that comparison; carrying one
   across is the forbidden between-rig transfer.

WHAT THE INSTRUMENT IS FOR, THEN. Reporting coverage on a stated population
with its own null, its own interval, and its own bias printed beside every
number -- so that a future claim about the target's support can be judged.
It does NOT decompose a bundle into per-knob contributions, and no cell of it
should be pinned as a decision rule without redoing all four checks above.

RESOLUTION AND BIAS, MEASURED BEFORE ANY THRESHOLD (200 live rows, production
shape, 2026-08-09; the full 16-cell table is in the PR):

    cell                        coverage   control margin      harness bias
    tau=50 rho=0.05 phi=1e-3      0.7324   +14.42 [+7.2,+16.6]  -0.0334
    tau=50 rho=0.05 phi=2e-2      0.2575    +3.15 [+0.4, +5.1]  +0.0100
    tau=50 rho=0.01 phi=1e-2      0.0245    -7.80 [-8.1, -3.3]  +0.1043

THE FIDELITY GATE ABORTS -- see ``FidelityTolerance``. Production tier
(argmax >= 0.75, TV <= 0.20) applies when the run's shape is
``PRODUCTION_SEARCH_SHAPE``; a floor (0.65 / 0.35) applies to every run. The
harness clears the production tier on three independent windows: 0.8133/0.1423
(600 rows, banked), 0.8550/0.1221 (a reviewer's 200), 0.8200/0.1500 (200 fresh
rows here). **The failure mode that voided the earlier c_scale entropy sweep is
genuinely absent** -- that sweep predicted entropy 1.40 where production stores
0.92, whereas a reviewer's independent 200-row check put this harness within
+0.0127 +/- 0.0154 SE (ns) of production's own stored entropy. Driven to ``sims 2 / topk 2 / c_scale 5.0 / T 8.0 /
gumbel_scale 4.0`` it now measures 0.4500 / 0.7586 and EXITS 1 with no cells
printed; the previous revision printed ``PASS +19.57`` and exited 0.

⚑ ``gumbel_scale`` IS MEASURABLY INERT HERE, AND THAT IS WORTH RECORDING so
nobody re-does it. On a FROZEN 8-shard window with ``--row-keys`` pinning the
identical 200 rows, 0.5 vs 1.0 moves coverage by at most **0.0123** across 16
cells and by **exactly 0.0000** at the cell the retracted rule pinned
(``tau=50, rho=0.05, phi=2e-2``); fidelity 0.8200/0.1500 vs 0.8200/0.1488.
Wiring it to the C path was correct; it corrected nothing.

⚑ PRODUCING-NET PROVENANCE IS ABSENT ON LIVE SHARDS, and the guard now says so
instead of passing. ``ShardMeta.model_step`` / ``model_sha256`` are None on
every shard the trial writes, because ``DiskReplayBuffer._flush_shard_arrays``
calls ``save_local_shard_arrays`` with no ``meta``. ``--compare-to`` REFUSES on
absent provenance unless ``--allow-missing-shard-provenance`` is passed, which
prints an UNVERIFIABLE banner and is banked.

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
    readout runs. Fidelity against production's STORED target is checked every
    run and ABORTS outside ``FidelityTolerance`` -- 0.8200 argmax / 0.1500 TV
    measured at the live shape on 200 fresh rows, with the residual coming from
    the frozen reference checkpoint and from the history planes a decoded board
    cannot carry. An OFF-production arm additionally needs a production-shape
    ``--compare-to`` or ``--calibration``.

``--mode simulate`` (frozen deep-SF audit positions)
    Same metric against the audit set's >=1M-node MultiPV labels. Better
    soundness labels than production's MultiPV 6, different position
    distribution -- useful for mechanism, NOT for transferring a level or a
    derivative to the live rig.

``--compare-to <banked.json>`` differences two arms with an exact row-key
equality check first, refuses across different (or UNKNOWN) producing nets, and
prints each cell's harness bias beside its delta; ``--out`` banks the per-row
vectors AND the row keys, and ``--row-keys`` re-pins a later run to exactly
those rows, so a banked arm stays recomputable while its shards are on disk.
``--scan-bank`` recomputes the control-PASS / c_scale-quiet / both counts over a
banked multi-arm file, by a criterion that lives in ``attribution_scan`` rather
than in prose.

CONTROLS ARE ATTACHED TO EVERY CELL, ALWAYS. ``--control-seeds`` (default 12)
permutes the target within each row's legal moves and reports the shuffle's own
mean and SD beside the cell, plus ``(coverage - shuffled)/sd`` AND that margin's
own 95% position-cluster interval. ⚑ The PASS/null/INVERTED verdict is read off
the INTERVAL, never the point: a margin of +3.15 whose lower bound is +0.41 is
``null``. With ``--control-bootstrap 0`` no cell can be stamped PASS at all --
it reads ``no-res``, because an unresolved gate is not a passed one. The shuffle
distribution is the null; ``base_rate`` pools differently and is indicative
only. ``--shuffle prior``
permutes the prior instead, so ``rare`` becomes a size-matched random subset.
``tests/test_rare_sound_move_coverage.py`` ships both shuffles, the inversion,
and the cluster bootstrap as assertions.

Usage::

    PYTHONPATH=. python3 scripts/rare_sound_move_coverage.py \\
        --mode shards --checkpoint data/ruler_reads_20260808/trainer.pt \\
        --replay-dir /abs/path/to/<trial>/replay_shards --shards 40

    # the CALIBRATION arm: production's shape, so its `stored - A` bias is
    # pure harness error and it certifies every arm compared against it.
    PYTHONPATH=. python3 scripts/rare_sound_move_coverage.py \\
        --mode research --checkpoint data/ruler_reads_20260808/trainer.pt \\
        --replay-dir /abs/path/to/<trial>/replay_shards --shards 8 \\
        --max-rows 200 --sims 256 --topk 32 --gumbel-scale 0.5 \\
        --c-scale 0.025 --policy-temp 1.0 --out arm_A.json

    # an arm, pinned to arm A's exact rows and certified by it
    PYTHONPATH=. python3 scripts/rare_sound_move_coverage.py \\
        --mode research --checkpoint data/ruler_reads_20260808/trainer.pt \\
        --replay-dir /abs/path/to/<trial>/replay_shards --shards 8 \\
        --row-keys arm_A.json --sims 256 --topk 32 --gumbel-scale 0.5 \\
        --c-scale 0.1 --policy-temp 1.0 --out arm_B.json \\
        --compare-to arm_A.json --allow-missing-shard-provenance

    # recompute the retracted headline from the artifact
    PYTHONPATH=. python3 scripts/rare_sound_move_coverage.py \\
        --scan-bank tests/data/rare_sound_move_coverage/live_arms_20260809.json

Banked evidence for the 600-row arm table above:
``tests/data/rare_sound_move_coverage/live_arms_20260809.json``. ⚑ It predates the
control interval and the bias column, so its cells carry NO resolution;
``--scan-bank`` reports 0 interval-based passes over it for that reason.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import dataclasses
from dataclasses import asdict, dataclass, field, fields
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
    # ⚑ THE VERDICT'S OWN RESOLUTION. `control_margin` is a point estimate and
    # the first version of this script stamped a categorical PASS/INVERTED off
    # it against a hard >= 1.0. A reviewer row-resampled the cell the ledger had
    # pinned and got margin +2.06 +/- 1.39, 95% [-0.06, +5.18] -- 22% of draws
    # FAIL the gate the cell was reported as passing. A threshold read off an
    # estimate whose own noise is 1.4 SD is not a gate, so the interval is now
    # computed by the same POSITION-cluster bootstrap as the coverage CI and the
    # verdict is taken from the INTERVAL, never from the point.
    margin_ci_lo: float = float("nan")
    margin_ci_hi: float = float("nan")

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

    def verdict(self, min_sds: float = 1.0) -> str:
        """PASS / INVERTED / null / no-res / no-ctrl, decided on the INTERVAL.

        ``no-ctrl``  the shuffle produced no usable null at this cell.
        ``no-res``   a null exists but no interval was computed for it, so the
                     cell has no resolution and CANNOT be stamped PASS. Absence
                     of an interval reads as "unknown", never as "fine".
        ``PASS``     the margin's 95% LOWER bound clears ``min_sds``.
        ``INVERTED`` its UPPER bound is below ``-min_sds``.
        ``null``     the interval straddles the thresholds: indeterminate.
        """
        if not math.isfinite(self.control_margin):
            return "no-ctrl"
        if not (math.isfinite(self.margin_ci_lo) and math.isfinite(self.margin_ci_hi)):
            return "no-res"
        if self.margin_ci_lo >= min_sds:
            return "PASS"
        if self.margin_ci_hi <= -min_sds:
            return "INVERTED"
        return "null"

    def passes_control(self, min_sds: float = 1.0) -> bool:
        return self.verdict(min_sds) == "PASS"


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


def _cell_row_counts(
    rows: list[RowVectors], cell: CoverageCell,
) -> np.ndarray:
    """(n, 2) array of per-row (pairs, covered) for one cell."""
    out = np.empty((len(rows), 2), dtype=np.float64)
    for i, row in enumerate(rows):
        p, c, _nl, _no = _row_pair_counts(row, cell.tau_cp, cell.rho, cell.phi)
        out[i, 0] = float(p)
        out[i, 1] = float(c)
    return out


def _margin_interval(
    real: np.ndarray, sh: np.ndarray, *, resamples: int, seed: int,
) -> tuple[float, float]:
    """95% POSITION-cluster bootstrap interval for the control margin.

    ``real`` is (n, 2) of per-row (pairs, covered); ``sh`` is (S, n, 2), the
    same for each shuffle seed. One bootstrap draw resamples POSITIONS once and
    re-scores the real arm and every shuffle seed on that SAME draw, so the
    margin's numerator and denominator move together exactly as they do in the
    reported number. Resampling them independently would understate the
    correlation and hand back an interval that is too narrow -- the specific
    error that made the point estimate look decisive.
    """
    n_seeds, n_rows, _ = sh.shape
    if n_seeds < 2 or n_rows < 2 or resamples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    # Multinomial resample WEIGHTS rather than index arrays: the weights
    # contract against (n, S) in one matmul, where fancy-indexing (S, B, n)
    # would allocate seeds x resamples x rows and blow up on the full grid.
    w = rng.multinomial(n_rows, np.full(n_rows, 1.0 / n_rows), size=int(resamples))
    w = w.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        real_cov = np.where(
            (w @ real[:, 0]) > 0.0, (w @ real[:, 1]) / np.maximum(w @ real[:, 0], 1e-12),
            np.nan,
        )
        sh_pairs = w @ sh[:, :, 0].T          # (B, S)
        sh_cov = np.where(
            sh_pairs > 0.0, (w @ sh[:, :, 1].T) / np.maximum(sh_pairs, 1e-12), np.nan,
        )
        mu = np.nanmean(sh_cov, axis=1)
        sd = np.nanstd(sh_cov, axis=1, ddof=1)
        margin = np.where(sd > 0.0, (real_cov - mu) / np.maximum(sd, 1e-12), np.nan)
    margin = margin[np.isfinite(margin)]
    if margin.size < 2:
        return float("nan"), float("nan")
    return float(np.percentile(margin, 2.5)), float(np.percentile(margin, 97.5))


def attach_controls(
    rows: list[RowVectors], cells: list[CoverageCell], *, seeds: int, seed0: int,
    what: str = "target", resamples: int = 0,
) -> None:
    """Fill every cell's null AND the null's own interval, IN PLACE.

    Run over the whole grid, not one cell. The control is cell-local: the
    association coverage is named for holds at a small mass floor and INVERTS at
    a large one, so a control read at one cell says nothing about another. This
    function exists so a cell can never be reported without its own null.

    ``resamples`` > 0 additionally bootstraps the margin over POSITIONS and
    fills ``margin_ci_lo/hi``. With it at 0 no cell can be stamped PASS: the
    verdict is taken from the interval and a missing interval is ``no-res``.
    """
    if seeds <= 0:
        return
    # Built once and reused for every cell. Only the permuted vector is a new
    # array -- the other three are shared references -- so S copies of the row
    # list cost S x n x L floats, not S x the whole population.
    sh_sets = [
        shuffled_rows(rows, what=what, seed=seed0 + 1000 * (s + 1))
        for s in range(int(seeds))
    ]
    for i, cell in enumerate(cells):
        real = _cell_row_counts(rows, cell)
        sh = np.stack([_cell_row_counts(rs, cell) for rs in sh_sets])
        with np.errstate(invalid="ignore", divide="ignore"):
            tot_pairs = sh[:, :, 0].sum(axis=1)
            per_seed = np.where(
                tot_pairs > 0.0,
                sh[:, :, 1].sum(axis=1) / np.maximum(tot_pairs, 1e-12),
                np.nan,
            )
        vals = per_seed[np.isfinite(per_seed)]
        cell.control_seeds = int(vals.size)
        cell.shuffled_mean = float(vals.mean()) if vals.size else float("nan")
        cell.shuffled_sd = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
        cell.margin_ci_lo, cell.margin_ci_hi = _margin_interval(
            real, sh, resamples=int(resamples), seed=seed0 + 7919 * (i + 1),
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
    # Rows dropped for a non-finite target/regret. Under `assert_population`'s
    # 1% tolerance these used to be scored as uncovered instead.
    dropped_non_finite: int = 0


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


def assert_same_producing_net(
    stats: ShardReadStats, prov_a: dict[str, object], *, allow_missing: bool,
) -> None:
    """Refuse to difference two arms produced by DIFFERENT nets -- or by UNKNOWN ones.

    ⚑ ABSENT PROVENANCE MUST NOT READ AS "FINE". The previous version recorded
    ``model_step`` / ``model_sha256`` and claimed ``--compare-to`` would refuse
    across producing nets. It cannot: the trial's ``replay_shards`` are written
    by ``DiskReplayBuffer._flush_shard_arrays``, which calls
    ``save_local_shard_arrays`` with no ``meta``, so every ``ShardMeta`` field
    is None (verified on the live shards 2026-08-09). ``model_steps`` came back
    ``[]`` and the check silently passed. That is the house defect exactly: a
    value accepted and then ignored.

    So the guard now REFUSES when the identity is unknown on either side, and
    the escape hatch is explicit, printed, and banked. Fixing the data instead
    means populating ``ShardMeta`` at ``disk_buffer.py:1580``, which is an
    ingest-path change and out of scope for a diagnostic script.
    """
    # ⚑ THE STEP IS NOT THE IDENTITY. Two PBT trials, or two lineages, reach the
    # same `model_step` with different weights -- and they reuse shard basenames
    # and indices, so nothing else in this comparison would notice. The sha was
    # already collected and simply never read: the house defect, in the guard
    # written to catch the house defect.
    steps_b = sorted(set(stats.model_steps))
    shas_b = sorted(set(stats.model_sha_prefixes))
    read_a = prov_a.get("read_stats")
    steps_a = sorted(set(read_a.get("model_steps", []))) if isinstance(read_a, dict) else []
    shas_a = (
        sorted(set(read_a.get("model_sha_prefixes", [])))
        if isinstance(read_a, dict) else []
    )
    if shas_a and shas_b and shas_a != shas_b:
        raise SystemExit(
            f"refusing to difference arms produced by different nets: arm A "
            f"model_sha={shas_a}, this run={shas_b} (steps {steps_a} vs {steps_b} "
            "-- equal steps do NOT imply equal weights across trials or lineages)"
        )
    # ⚑ COMPLETENESS, not mere non-contradiction. The SHA check above can only
    # fire when BOTH sides carry a SHA, and the step check below proves nothing
    # on its own: `model_step` is a per-trial counter, so equal steps across
    # trials or lineages are equal COUNTERS, not equal WEIGHTS -- that is the
    # whole reason the SHA comparison exists. An earlier revision required only
    # that steps be present and equal, which accepted "equal step, SHA ABSENT on
    # one side" as verified identity: exactly the condition the guard is for.
    # Missing SHA is therefore routed to the SAME unverifiable branch as missing
    # steps. Absent provenance is not evidence of a match.
    missing = [
        name for name, got in (
            ("arm A model_steps", steps_a), ("this run model_steps", steps_b),
            ("arm A model_sha_prefixes", shas_a), ("this run model_sha_prefixes", shas_b),
        ) if not got
    ]
    if missing:
        detail = (
            f"arm A model_steps={steps_a or 'ABSENT'}, this run="
            f"{steps_b or 'ABSENT'}; arm A model_sha={shas_a or 'ABSENT'}, "
            f"this run={shas_b or 'ABSENT'}"
        )
        if allow_missing:
            print(
                "[provenance] ⚑ UNVERIFIABLE: the shards do not fully identify the "
                f"producing net -- ABSENT: {', '.join(missing)} ({detail}). "
                "Proceeding only because --allow-missing-shard-provenance was "
                "given; this comparison is NOT proved to be against one "
                "producing net."
            )
            return
        raise SystemExit(
            "refusing to difference two arms whose producing net is UNKNOWN -- "
            f"ABSENT: {', '.join(missing)} ({detail}). The trial's replay shards "
            "are written by DiskReplayBuffer._flush_shard_arrays with no "
            "ShardMeta, so model_step and model_sha256 are None on every one of "
            "them. Absent provenance is not evidence of a match. Pass "
            "--allow-missing-shard-provenance to proceed on the record that this "
            "is unverified."
        )
    if steps_a != steps_b:
        raise SystemExit(
            f"refusing to difference arms produced by different nets: arm A "
            f"model_steps={steps_a}, this run={steps_b}"
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
    target and the STORED one at the live shape, and ``FidelityTolerance``
    turns it into a gate that ABORTS -- see that class for the two tiers and
    the exact numbers.
    """

    shape: SimShape
    sims: int
    syzygy_path: str | None = None
    # ⚑ RECORDED IS NOT USED. The runner hard-coded its Gumbel RNG seed while
    # provenance banked `args.seed`, so `--seed` moved the bootstraps and the
    # shuffles but NOT the search itself: the recorded seed did not reproduce
    # the arm it was recorded for.
    seed: int = 20260809
    # The producing checkpoint's own extra-feature encoding. Left at the
    # `GumbelConfig` default, a legacy `v1` (146-plane) checkpoint got its ROOT
    # from the shard's stored planes and its CHILDREN re-encoded to 175, which
    # the model cannot consume.
    input_extra_features: str = "v2_threats"


@dataclass(frozen=True)
class FidelityTolerance:
    """The fidelity gate's thresholds. ⚑ THIS ABORTS. It is not a printout.

    The first version of this script called fidelity "the gate" in a docstring
    and then only ``print``ed it: no threshold, no abort, no verdict read it.
    Driven to ``sims 2 / topk 2 / c_scale 5.0 / T 8.0 / gumbel_scale 4.0`` the
    harness cratered to argmax 0.54 / TV 0.72 and still printed ``PASS +19.57``
    and exited 0 -- the broken shape beat the honest one's +1.56, because a
    shuffle control cannot tell you whether you are searching production's
    search [[a_gate_that_cannot_fail]].

    TWO TIERS, because "disagrees with the stored target" means two different
    things depending on the shape being searched:

    * ``floor_*`` applies to EVERY research run. Below it the re-search is not
      a perturbation of production's search at all. Calibrated from the widest
      legitimate arm in ``live_arms_20260809.json`` -- the bundle arm E
      (c_scale 0.1, T 1.5) at argmax 0.7433 / TV 0.2094 -- with headroom, and
      well clear of the broken shape's 0.5400 / 0.7245.
    * ``prod_*`` applies only when the run's shape IS ``PRODUCTION_SEARCH_SHAPE``.
      That run is the CALIBRATION: it is the one asserting the harness
      reproduces production, so it is held to the measured value (arm A: 0.8133
      argmax / 0.1423 TV over 600 rows) rather than to the floor.

    An off-production arm additionally has to be PAIRED against a
    production-shape arm (``--compare-to``) or accompanied by
    ``--calibration``; see ``assert_calibrated``. An arm measured by a harness
    that was never shown to reproduce production is a number about the harness.
    """

    floor_min_argmax: float = 0.65
    floor_max_tv: float = 0.35
    prod_min_argmax: float = 0.75
    prod_max_tv: float = 0.20


@dataclass
class ResearchFidelity:
    """How well the re-search reproduces production's stored target."""

    n: int = 0
    argmax_agree: int = 0
    tv_sum: float = 0.0

    @property
    def argmax_rate(self) -> float:
        return (self.argmax_agree / self.n) if self.n else float("nan")

    @property
    def mean_tv(self) -> float:
        return (self.tv_sum / self.n) if self.n else float("nan")

    def report(self) -> str:
        if self.n == 0:
            return "fidelity: no rows scored"
        return (
            f"fidelity vs STORED target: argmax agreement "
            f"{self.argmax_rate:.4f}, mean TV {self.mean_tv:.4f} (n={self.n})"
        )

    def assert_within(
        self, tol: FidelityTolerance, *, is_production_shape: bool,
    ) -> None:
        """Abort unless the re-search is close enough to production's target.

        Raises ``SystemExit`` -- the run must not proceed to print cells,
        because every number after this point would be scored on a target the
        production loop never stored.
        """
        if self.n == 0:
            raise SystemExit(
                "fidelity gate: no rows were scored against the stored target, so "
                "the re-search was never checked against production -- aborting"
            )
        lo = tol.prod_min_argmax if is_production_shape else tol.floor_min_argmax
        hi = tol.prod_max_tv if is_production_shape else tol.floor_max_tv
        tier = "PRODUCTION-SHAPE" if is_production_shape else "floor"
        bad = []
        if not (self.argmax_rate >= lo):
            bad.append(f"argmax agreement {self.argmax_rate:.4f} < {lo:.4f}")
        if not (self.mean_tv <= hi):
            bad.append(f"mean TV {self.mean_tv:.4f} > {hi:.4f}")
        if bad:
            raise SystemExit(
                f"FIDELITY GATE FAILED ({tier} tier, n={self.n}): " + "; ".join(bad)
                + ". The re-searched target is not a perturbation of the target "
                "production stored on these rows, so a coverage number computed "
                "from it describes this harness and not the training loop. "
                "Aborting instead of printing a cell whose control can still say "
                "PASS -- the shuffle control is blind to this failure."
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
    want_keys: list[str] | None = None,
    stored_out: list[RowVectors] | None = None,
) -> list[RowVectors]:
    """Live shard rows -> RowVectors, with the prior from the stored ``x``.

    The prior is a forward pass on the planes production searched with, so it
    is exact up to reference-checkpoint drift: nothing is decoded back to a
    board and no history is lost. The checkpoint's declared encodings are
    checked against each shard's, because a mismatch would feed the net planes
    it never saw.

    With ``research`` set, the ``target`` is replaced by a FRESH Gumbel search
    at that shape instead of the stored one, which is how the isolation arms are
    run on the live population; ``stored_out`` then receives the SAME rows
    carrying production's STORED target, so the harness's own bias can be
    measured in coverage units on the identical positions.

    ``want_keys`` pins the read to an exact set of ``shard:index`` keys and
    aborts if any is missing, which is the only way a banked arm stays
    reproducible: the newest-by-mtime window rolls every few minutes and two
    runs a quarter of an hour apart already draw disjoint rows.
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

    if research is not None:
        research = dataclasses.replace(
            research, input_extra_features=str(
                getattr(model, "input_extra_features", "v2_threats")),
        )
    searcher = _ResearchRunner(research, device=device, evaluator=evaluator,
                               hist=ck_hist) if research is not None else None

    wanted: set[str] | None = set(want_keys) if want_keys is not None else None
    wanted_bases = (
        {k.split(":", 1)[0] for k in wanted} if wanted is not None else None
    )

    rows: list[RowVectors] = []
    for path in reversed(sel.paths):
        if wanted_bases is not None and os.path.basename(path) not in wanted_bases:
            continue
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
        if wanted is not None:
            idx = np.asarray(
                [i for i in idx.tolist() if f"{base}:{int(i)}" in wanted],
                dtype=idx.dtype,
            )
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
                # ⚑ TOLERATED IS NOT HARMLESS. `assert_population` permits up to
                # 1% non-finite `policy_target` / `sf_p0_regret` rows before it
                # aborts, and those rows were still appended: a NaN target reads
                # as UNCOVERED at every threshold and a NaN regret empties the
                # scored mask, so corruption UNDER the abort threshold moved both
                # the numerator and the denominator with nothing saying so.
                # Dropped and counted instead -- the count is printed and banked.
                if not (
                    bool(np.all(np.isfinite(tgt_full[lm])))
                    and bool(np.all(np.isfinite(reg_row[lm])))
                ):
                    stats.dropped_non_finite += 1
                    continue
                scored_full = _scored_mask_from_regret(reg_row)
                prior = e / s
                row_key = f"{base}:{int(idx[k])}"
                rows.append(RowVectors(
                    prior=prior,
                    target=tgt_full[lm],
                    regret_cp=reg_row[lm] * SF_OWN_REGRET_CAP_CP,
                    scored=scored_full[lm],
                    key=row_key,
                ))
                if stored_out is not None:
                    # The SAME row with production's own target. Only `target`
                    # differs, so `stored - research` is position-paired with
                    # nothing else varying: it is the harness's OWN bias, and
                    # arm A is production's shape, which makes it pure error.
                    stored_out.append(RowVectors(
                        prior=prior,
                        target=target[k][lm],
                        regret_cp=reg_row[lm] * SF_OWN_REGRET_CAP_CP,
                        scored=scored_full[lm],
                        key=row_key,
                    ))
                if max_rows and wanted is None and len(rows) >= max_rows:
                    break
            if max_rows and wanted is None and len(rows) >= max_rows:
                break
        if max_rows and wanted is None and len(rows) >= max_rows:
            break

    if wanted is not None:
        order = {k: i for i, k in enumerate(want_keys or [])}
        missing = sorted(wanted - {r.key for r in rows})
        if missing:
            raise SystemExit(
                f"{len(missing)} of {len(wanted)} pinned row keys are not in the "
                f"selected shards (first: {missing[0]!r}). The newest-by-mtime "
                "window rolls, so a banked arm becomes unreproducible once its "
                "shards age out; widen --shards or accept that the arm cannot be "
                "recomputed -- refusing to silently score a different population"
            )
        paired = sorted(zip(rows, stored_out or [None] * len(rows), strict=True),
                        key=lambda pair: order[pair[0].key])
        rows = [p[0] for p in paired]
        if stored_out is not None:
            stored_out[:] = [p[1] for p in paired if p[1] is not None]

    stats.rows_used = len(rows)
    if wanted is None:
        assert_population(stats, min_sf_p0_rate=min_sf_p0_rate)
    else:
        print(
            f"[shards] pinned to {len(wanted)} banked row keys; the population "
            "guard is not applicable (the denominator is fixed by the keys) and "
            "a missing key aborts instead"
        )
    if not rows:
        raise SystemExit("no usable rows: every selected row lacked a legal mask")
    return rows


def build_sim_gumbel_config(
    shape: SimShape,
    *,
    sims: int,
    hist: str = "legacy",
    extra: str = "v1",
    pol_enc: str | None = None,
):
    """The ONE place a ``SimShape`` becomes a search config, for BOTH runners.

    Module-level rather than inline in ``_ResearchRunner.__init__`` and
    ``simulate_rows`` for the reason ``audit_targets.build_profile_search_shape``
    is: inside those, the refusal was unreachable without a checkpoint and a
    shard bank, so nothing could prove the guard runs. ``main()`` calls it right
    after ``_build_shape`` -- before any load -- and
    ``tests/test_gumbel_config_validation.py`` drives it directly.

    ``--policy-temp`` / ``--c-scale`` / ``--halving-div`` / ``--topk`` reach
    ``SimShape`` through a bare ``float()``/``int()`` in ``_build_shape``, and
    the arm's label banks them, so a value the search would drop becomes a
    coverage number attributed to a shape nobody ran.

    ``pol_enc=None`` leaves the ``GumbelConfig`` default, which is what the
    shard-decode runner relied on; the encodings do not participate in
    validation.
    """
    from chess_anti_engine.mcts.gumbel import GumbelConfig, validate_gumbel_config

    cfg = GumbelConfig(
        simulations=int(sims), topk=int(shape.topk), temperature=0.0,
        policy_temp=float(shape.policy_temp), c_scale=float(shape.c_scale),
        c_visit=float(shape.c_visit), c_visit_root=float(shape.c_visit_root),
        c_scale_root=float(shape.c_scale_root),
        q_visit_exp_root=float(shape.q_visit_exp_root),
        halving_div=int(shape.halving_div), add_noise=bool(shape.add_noise),
        gumbel_scale=float(shape.gumbel_scale), input_history_encoding=hist,
        input_extra_features=extra,
    )
    if pol_enc is not None:
        cfg = dataclasses.replace(cfg, policy_encoding=str(pol_enc))
    try:
        validate_gumbel_config(cfg, where="rare_sound_move_coverage --shape flags")
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    return cfg


class _ResearchRunner:
    """Decodes shard rows back to boards and re-searches them at one shape."""

    def __init__(
        self, spec: ResearchSpec, *, device: str, evaluator: object, hist: str,
    ) -> None:
        self.spec = spec
        self.device = device
        self.evaluator = evaluator
        self.hist = hist
        self.rng = np.random.default_rng(int(spec.seed))
        self.tb_probe = None
        if spec.syzygy_path:
            from chess_anti_engine.tablebase import SyzygyProbe

            self.tb_probe = SyzygyProbe(spec.syzygy_path)
        self.cfg = build_sim_gumbel_config(
            spec.shape, sims=int(spec.sims), hist=hist,
            extra=str(spec.input_extra_features),
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
      # >= 90 = LINEAR root, exponent 1.0. Previously resolved to the descent's
      # `q_visit_exp`, a deleted GumbelConfig field whose default was 1.0.
        qer = self.q_visit_exp_root if self.q_visit_exp_root < 90.0 else 1.0
        if qer < 0.0:
            return csr * math.log1p(cvr + max_visit)
        mv = max_visit if qer == 1.0 else max_visit**qer
        return csr * (cvr + mv)


# ⚑ THE SHAPE THE READ SHARDS WERE PRODUCED AT, hardcoded because THE SHARDS DO
# NOT RECORD IT. `DiskReplayBuffer._flush_shard_arrays` calls
# `save_local_shard_arrays(path, arrs=arrs)` with no `meta`, so every field of
# `ShardMeta` -- `model_step`, `model_sha256`, `run_id` -- is None on the trial's
# `replay_shards` (verified 2026-08-09 on the 3 newest live shards: the only
# non-None attrs are version/positions/encodings/history_rep_fix). There is
# therefore nothing in the data to read production's search shape off, and the
# fidelity gate needs it to know whether a run is the CALIBRATION or an ARM.
#
# ⚑ SOURCED FROM THE **LIVE** YAML, WHICH IS NOT THIS REPO'S YAML, AND IT IS A
# DATED SNAPSHOT, NOT A LIVE READING. Read on **2026-08-09** off the running
# branch's `configs/pbt2_small.yaml`: `gumbel_c_scale 0.025`, `gumbel_topk 32`,
# `gumbel_scale_after 0.5`, `mcts_simulations 256`, `gumbel_policy_temp` at its
# no-op 1.0. The in-repo `configs/pbt2_small.yaml` said
# `gumbel_topk 16 / gumbel_c_scale 0.1 / gumbel_scale_after 0.0` -- the live
# yaml and main had diverged (608 of 968 keys), so a test that pinned this
# constant to the in-repo config would pin it to a search NOBODY IS RUNNING.
# `gumbel_scale` is the DECAYED 0.5, not the 1.0 opening value, because 88.9%
# of stored policy rows are at ply >= 30.
#
# ⚑⚑ IT HAS ALREADY MOVED. The search-authority bundle shipped
# `gumbel_c_scale` 0.025 -> 0.1 and `gumbel_policy_temp` 1.0 -> 1.5 on the live
# branch within a day of this constant being written, so as of 2026-08-10 this
# is the shape of the shards banked in `tests/data/`, NOT the shape production
# is writing today. Nothing here auto-detects that, by design (see below);
# what it means in practice is that a run against TODAY's shards must pass
# `--production-shape 0.1,1.5,32,0.5,256`, and a run left on this constant will
# ABORT at the fidelity gate rather than mis-tier. Both behaviours are what you
# want; neither is "it still works".
#
# ⚑ NOTHING IN-BAND CAN CATCH THIS GOING STALE, and that is a limitation, not
# an oversight: the shards record no shape (see `assert_same_producing_net`).
# What DOES catch it is the production tier of the fidelity gate. A
# bundle-sized shape mismatch between the harness and the stored target moves
# fidelity by more than the tier allows -- banked arm E (c_scale 0.1, T 1.5 vs
# a target stored at 0.025/1.0) lands at argmax 0.7433 / TV 0.2094, failing
# both 0.75 and 0.20. So if production moves and this constant does not, the
# CALIBRATION RUN FAILS rather than silently mis-tiering. Declare a moved
# production shape with `--production-shape` instead of editing this.
PRODUCTION_SEARCH_SHAPE: dict[str, float] = {
    "c_scale": 0.025,
    "policy_temp": 1.0,
    "topk": 32.0,
    "gumbel_scale": 0.5,
}
PRODUCTION_SIMS = 256

# ⚑ THE OTHER EIGHT KNOBS. Every one of these is a `SimShape` field, every one
# is passed into the `GumbelConfig` that drives the re-search, and four of them
# (`c_visit`, `halving_div`, `vloss_weight`, `add_noise`) are settable from the
# CLI. The earlier `is_production_shape` compared only the four headline knobs
# plus `sims`, so `--c-visit 200 --no-noise` was still classified as PRODUCTION
# and its `stored - re-searched` delta was printed as "PURE HARNESS ERROR" --
# then usable as the calibration that certifies every other arm. A gate that
# checks four of twelve inputs is not a gate.
PRODUCTION_SEARCH_SHAPE_REST: dict[str, float] = {
    "c_visit": 50.0,
    "c_visit_root": -1.0,
    "c_scale_root": -1.0,
    "q_visit_exp_root": 99.0,
    "halving_div": 2.0,
    "vloss_weight": 1.0,
    "add_noise": 1.0,
}

# ⚑ EXHAUSTIVE BY CONSTRUCTION, NOT BY REVIEW. Adding a field to `SimShape`
# without declaring its production value now fails at import rather than
# silently widening the set of shapes that pass as the calibration arm.
_SHAPE_FIELD_NAMES = {f.name for f in fields(SimShape)}
_DECLARED_SHAPE_KEYS = set(PRODUCTION_SEARCH_SHAPE) | set(PRODUCTION_SEARCH_SHAPE_REST)
if _SHAPE_FIELD_NAMES != _DECLARED_SHAPE_KEYS:
    raise AssertionError(
        "PRODUCTION_SEARCH_SHAPE + PRODUCTION_SEARCH_SHAPE_REST must name EVERY "
        "SimShape field, because every one of them reaches the re-search. "
        f"undeclared={sorted(_SHAPE_FIELD_NAMES - _DECLARED_SHAPE_KEYS)} "
        f"stale={sorted(_DECLARED_SHAPE_KEYS - _SHAPE_FIELD_NAMES)}"
    )


def production_shape_mismatches(
    shape: SimShape, sims: int, *, declared: dict[str, float] | None = None,
) -> list[str]:
    """Every field on which ``shape``/``sims`` differs from production.

    Empty means this run IS the calibration arm. The list is returned rather
    than a bool so the caller can say WHICH knob disqualified the run --
    "not production-shaped" with no field named is the kind of message that
    gets read as a formality.
    """
    ref = dict(PRODUCTION_SEARCH_SHAPE_REST)
    ref.update(declared or PRODUCTION_SEARCH_SHAPE)
    bad: list[str] = []
    for name in sorted(_SHAPE_FIELD_NAMES):
        want = ref.get(name)
        if want is None:
            continue
        got = float(getattr(shape, name))
        if not math.isclose(got, float(want)):
            bad.append(f"{name}={got!r} (production {float(want)!r})")
    want_sims = int(ref.get("sims", PRODUCTION_SIMS))
    if int(sims) != want_sims:
        bad.append(f"sims={int(sims)} (production {want_sims})")
    return bad


def is_production_shape(
    shape: SimShape, sims: int, *, declared: dict[str, float] | None = None,
) -> bool:
    """Is this run the calibration arm rather than an experimental arm?"""
    return not production_shape_mismatches(shape, sims, declared=declared)


def parse_production_shape(spec: str) -> dict[str, float]:
    """``c_scale,policy_temp,topk,gumbel_scale,sims`` -> the reference shape."""
    parts = _parse_floats(spec)
    if len(parts) != 5:
        raise SystemExit(
            "--production-shape takes exactly 5 numbers: "
            f"c_scale,policy_temp,topk,gumbel_scale,sims (got {len(parts)})"
        )
    return {
        "c_scale": parts[0], "policy_temp": parts[1], "topk": parts[2],
        "gumbel_scale": parts[3], "sims": parts[4],
    }


def assert_calibrated(
    *, this_shape: SimShape, this_sims: int, other: dict[str, object] | None,
    other_label: str, declared: dict[str, float] | None = None,
) -> None:
    """An off-production arm may only be reported next to a calibration arm.

    ``other`` is the provenance of the arm this run is being differenced
    against (or of an explicit ``--calibration`` dump). At least one of the two
    must be ``PRODUCTION_SEARCH_SHAPE`` with a fidelity that cleared the gate,
    otherwise the pair is two unvalidated searches and their difference is a
    property of the harness.
    """
    if is_production_shape(this_shape, this_sims, declared=declared):
        return
    if other is None:
        raise SystemExit(
            "refusing to report an off-production arm with no calibration: this "
            f"run's shape (c_scale={this_shape.c_scale}, T={this_shape.policy_temp}, "
            f"topk={this_shape.topk}, gumbel_scale={this_shape.gumbel_scale}, "
            f"sims={this_sims}) is not PRODUCTION_SEARCH_SHAPE, so its fidelity "
            "against the stored target measures the ARM, not the harness. Pair it "
            "with a production-shape arm via --compare-to, or name one with "
            "--calibration."
        )
    sh = other.get("shape")
    if not isinstance(sh, dict):
        raise SystemExit(
            f"{other_label} carries no search shape in its provenance, so it "
            "cannot serve as the calibration arm -- aborting"
        )
    other_shape = SimShape(
        c_scale=float(sh.get("c_scale", float("nan"))),
        policy_temp=float(sh.get("policy_temp", float("nan"))),
        topk=int(sh.get("topk", -1)),
        gumbel_scale=float(sh.get("gumbel_scale", float("nan"))),
    )
    other_sims = int(float(str(other.get("sims", -1))))
    if not is_production_shape(other_shape, other_sims, declared=declared):
        raise SystemExit(
            f"neither arm is the production shape: this run is "
            f"(c_scale={this_shape.c_scale}, T={this_shape.policy_temp}) and "
            f"{other_label} is (c_scale={other_shape.c_scale}, "
            f"T={other_shape.policy_temp}, sims={other_sims}). Their difference "
            "is not anchored to a search anyone runs -- aborting"
        )
    fid = other.get("fidelity")
    if not isinstance(fid, dict) or not int(fid.get("n", 0)):
        raise SystemExit(
            f"{other_label} is the production shape but banked no fidelity, so "
            "nothing shows the harness reproduced the stored target -- aborting"
        )
    ref = ResearchFidelity(
        n=int(fid.get("n", 0)), argmax_agree=int(fid.get("argmax_agree", 0)),
        tv_sum=float(fid.get("tv_sum", 0.0)),
    )
    ref.assert_within(FidelityTolerance(), is_production_shape=True)


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

    cfg = build_sim_gumbel_config(
        shape, sims=int(sims), hist=hist, extra=extra, pol_enc=pol_enc,
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
          f"{'CONTROL':>9} {'ctrl 95% CI':>17} {'verdict':>9} {'chance*':>8}")
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
        mci = (
            f"[{c.margin_ci_lo:+.2f},{c.margin_ci_hi:+.2f}]"
            if math.isfinite(c.margin_ci_lo) else "                 "
        )
        print(f"{c.tau_cp:7.0f} {c.rho:7.3f} {c.phi:8.1e} {c.n_pairs:7d} "
              f"{c.n_rows:6d} {cov:>9} {ci:>17} {sh:>9} {sd:>7} {marg} "
              f"{mci:>17} {c.verdict():>9} {c.base_rate:8.4f}")
    print("* `chance` is a POOLING-MISMATCHED reference and is indicative only; "
          "the null is `shuffled`.")
    print("  The verdict is read off the control's 95% INTERVAL, never its point "
          "estimate: `no-res` means a null exists but was not resolved, and it is "
          "not a PASS.")


def print_paired(
    rows_a: list[RowVectors],
    rows_b: list[RowVectors],
    *,
    cells: list[CoverageCell],
    resamples: int,
    seed: int,
    label_a: str,
    label_b: str,
    bias: dict[tuple[float, float, float], float] | None = None,
    control_a: dict[tuple[float, float, float], tuple[float, float, float]]
    | None = None,
) -> list[dict[str, object]]:
    """Paired A->B deltas per cell, carrying each cell's control verdict along.

    A delta at a cell whose control is INVERTED is printed with its verdict
    rather than suppressed, because the number is still real -- it just cannot
    be attributed to sound-move coverage.

    ⚑ ``bias`` IS THE INSTRUMENT'S RESOLUTION AND IT IS PRINTED BESIDE EVERY
    DELTA. It is ``coverage(stored) - coverage(arm at production's shape)`` on
    the SAME rows, i.e. pure harness error, and at the cell this script's
    earlier revision pinned it was **larger than the effect whose SIGN carried
    the whole attribution**. A cell where ``|bias| >= |delta|`` is marked
    ``UNRESOLVED``: the rig cannot resolve production's own target there to
    better than the thing it is being asked to measure. Computing the CI and
    not the bias is how an instrument gets declared sharp while being wrong
    [[compute_instrument_resolution_before_the_threshold]].
    """
    assert_paired(rows_a, rows_b)
    ctrl_a = control_a or {}
    print(f"\n=== paired delta {label_a} -> {label_b} "
          f"({len(rows_a)} paired positions) ===")
    if not ctrl_a:
        print("[control] arm A banked NO per-cell control (pre-dates the field or "
              "was summary-only): ctrl_A reads `--`, which means UNKNOWN. It does "
              "NOT mean the control passed.")
    print(f"{'tau_cp':>7} {'rho':>7} {'phi':>8} {'cov_A':>8} {'cov_B':>8} "
          f"{'delta':>9} {'95% CI':>19} {'ctrl_A':>8} {'ctrl_B':>8} {'biasA':>8} "
          f"{'resolved':>10}")
    out: list[dict[str, object]] = []
    n_unresolved = 0
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
        # ⚑ ctrl_A MUST COME FROM ARM A. `cells` is computed from rows_b, so
        # `c.control_margin` is arm B's control -- it was printed under the
        # header `ctrl_A` and serialized as `control_margin_a`. When the target
        # shape changes the control can change with it, so a reader validating
        # "arm A's control was clean" was reading arm B's number. Both are now
        # shown, each under its own name, and A's is read from A's own bank.
        mb = c.control_margin
        ma, ma_lo, ma_hi = ctrl_a.get(
            (c.tau_cp, c.rho, c.phi), (float("nan"),) * 3)
        ctrl_a_s = f"{ma:+8.2f}" if math.isfinite(ma) else "      --"
        ctrl_b_s = f"{mb:+8.2f}" if math.isfinite(mb) else "      --"
        bz = (bias or {}).get((c.tau_cp, c.rho, c.phi), float("nan"))
        bstr = f"{bz:+8.4f}" if math.isfinite(bz) else "      --"
        if math.isfinite(bz) and math.isfinite(d):
            resolved = "yes" if abs(d) > abs(bz) else "UNRESOLVED"
            n_unresolved += int(resolved != "yes")
        else:
            resolved = "no-bias"
        print(f"{c.tau_cp:7.0f} {c.rho:7.3f} {c.phi:8.1e} {a:8.4f} {b:8.4f} "
              f"{d:+9.4f} [{lo:+.4f},{hi:+.4f}] {ctrl_a_s} {ctrl_b_s} {bstr} "
              f"{resolved:>10}")
        out.append({
            "tau_cp": c.tau_cp, "rho": c.rho, "phi": c.phi,
            "coverage_a": a, "coverage_b": b, "delta": d, "ci_lo": lo, "ci_hi": hi,
            "control_margin_a": ma, "control_margin_a_ci_lo": ma_lo,
            "control_margin_a_ci_hi": ma_hi, "control_margin_b": mb,
            "harness_bias_a": bz, "resolved": resolved,
        })
    if bias:
        print(f"[bias] {n_unresolved}/{len(cells)} cells have |harness bias| >= "
              "|delta|: at those the rig cannot resolve production's own stored "
              "target to better than the effect being attributed")
    return out


@dataclass(frozen=True)
class AttributionScan:
    """Whether ANY cell is both honest (control passes) and quiet in ``c_scale``.

    ⚑ THIS EXISTS BECAUSE THE ANSWER IS NO, AND THE PR SAID SO IN PROSE.
    An earlier revision reported "81 pass the control, 39 are c_scale-quiet, 2
    are both" with the word "quiet" appearing exactly once in the repository --
    in a docstring. A reviewer had to guess the definition and found five
    plausible ones giving 9/37/39/64/94 quiet cells and 0/0/2/20/81 survivors:
    the whole conclusion turned on a choice the text never stated. A number no
    one can recompute is not a result, so the criterion is now a function with
    an explicit ``quiet_ratio`` and a test.

    ``corr_margin_ratio`` is the reason the survivors were never a regime:
    "passes the control" and "is quiet in c_scale" are the SAME VARIABLE.
    Soundness is a proxy for Q and ``c_scale`` is the gain on Q, so the control
    is strong exactly where ``c_scale`` bites. Taking the argmax over a grid of
    that tension is selecting on the hypothesis
    [[never_condition_a_control_on_its_own_outcome]], one level up from row
    selection.
    """

    quiet_ratio: float
    min_pairs: int
    n_cells: int
    n_pass_point: int      # the OLD criterion: point margin >= min_sds
    n_pass_ci: int         # the criterion now shipped: 95% LOWER bound >= min_sds
    n_quiet: int
    n_both_point: int
    n_both_ci: int
    corr_margin_ratio: float
    both_point: list[tuple[float, float, float]]

    def report(self) -> str:
        return (
            f"[scan] {self.n_cells} cells with >= {self.min_pairs} pairs; quiet "
            f"criterion |d_c| <= {self.quiet_ratio:g} * |d_T|\n"
            f"[scan] control PASS: {self.n_pass_point} by POINT margin, "
            f"{self.n_pass_ci} by 95% LOWER bound\n"
            f"[scan] c_scale-quiet: {self.n_quiet}\n"
            f"[scan] BOTH: {self.n_both_point} (point) / {self.n_both_ci} (lower "
            f"bound)\n"
            f"[scan] corr(control margin, |d_c|/|d_T|) = "
            f"{self.corr_margin_ratio:+.3f}  <- the two criteria are one variable\n"
            f"[scan] cells that are both (point): {self.both_point}"
        )


def attribution_scan(
    cells: list[CoverageCell],
    *,
    deltas_c: dict[tuple[float, float, float], float],
    deltas_t: dict[tuple[float, float, float], float],
    quiet_ratio: float = 0.25,
    min_pairs: int = 50,
    min_sds: float = 1.0,
) -> AttributionScan:
    """Count control-passing / c_scale-quiet / both, by a criterion IN CODE."""
    margins: list[float] = []
    ratios: list[float] = []
    n_pass_point = n_pass_ci = n_quiet = n_both_point = n_both_ci = 0
    both_point: list[tuple[float, float, float]] = []
    n_cells = 0
    for cell in cells:
        key = (cell.tau_cp, cell.rho, cell.phi)
        if cell.n_pairs < min_pairs or key not in deltas_c or key not in deltas_t:
            continue
        d_c, d_t = deltas_c[key], deltas_t[key]
        if not (math.isfinite(d_c) and math.isfinite(d_t)) or d_t == 0.0:
            continue
        n_cells += 1
        margin = cell.control_margin
        ratio = abs(d_c) / abs(d_t)
        quiet = ratio <= quiet_ratio
        passes_point = bool(math.isfinite(margin) and margin >= min_sds)
        passes_ci = cell.passes_control(min_sds)
        n_pass_point += int(passes_point)
        n_pass_ci += int(passes_ci)
        n_quiet += int(quiet)
        if quiet and passes_point:
            n_both_point += 1
            both_point.append(key)
        n_both_ci += int(quiet and passes_ci)
        if math.isfinite(margin):
            margins.append(margin)
            ratios.append(ratio)
    corr = float("nan")
    if len(margins) > 2:
        corr = float(np.corrcoef(np.asarray(margins), np.asarray(ratios))[0, 1])
    return AttributionScan(
        quiet_ratio=float(quiet_ratio), min_pairs=int(min_pairs), n_cells=n_cells,
        n_pass_point=n_pass_point, n_pass_ci=n_pass_ci, n_quiet=n_quiet,
        n_both_point=n_both_point, n_both_ci=n_both_ci, corr_margin_ratio=corr,
        both_point=both_point,
    )


def scan_bank(
    path: Path, *, ref_arm: str, c_arm: str, t_arm: str, quiet_ratio: float,
    min_pairs: int,
) -> AttributionScan:
    """Run ``attribution_scan`` over a banked multi-arm JSON.

    The bank is ``{arms: {label: {cells: [...]}}, paired_vs_A: {label: [...]}}``.
    Cells banked before the control interval existed carry no
    ``margin_ci_lo/hi``, so their interval-based verdict is ``no-res`` and
    ``n_pass_ci`` is 0 -- which is the point: THE BANKED VERDICTS HAD NO
    RESOLUTION, and that is why they are not reproduced here as passes.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    arms = payload.get("arms", {})
    # ⚑ THE BANK'S DELTAS ARE ALL `paired_vs_A`. With `--scan-ref-arm` set to
    # anything else the CELLS and CONTROLS come from that arm while the EFFECTS
    # still come from the A-referenced table, so the pass/quiet counts mix two
    # different references and mean nothing.
    if str(ref_arm) != "A":
        raise SystemExit(
            f"--scan-ref-arm={ref_arm!r}: this bank schema stores deltas only as "
            "`paired_vs_A`, so a non-A reference would pair that arm's control "
            "with effects measured against A. Re-bank the deltas against "
            f"{ref_arm!r}, or scan with --scan-ref-arm A."
        )
    paired = payload.get("paired_vs_A", {})
    for label, where in ((ref_arm, arms), (c_arm, paired), (t_arm, paired)):
        if label not in where:
            raise SystemExit(f"{path} has no arm {label!r} (have {sorted(where)})")
    cells = [
        CoverageCell(**{k: v for k, v in c.items() if k in CoverageCell.__annotations__})
        for c in arms[ref_arm]["cells"]
    ]

    def _deltas(label: str) -> dict[tuple[float, float, float], float]:
        return {
            (float(r["tau_cp"]), float(r["rho"]), float(r["phi"])): float(r["delta"])
            for r in paired[label]
        }

    return attribution_scan(
        cells, deltas_c=_deltas(c_arm), deltas_t=_deltas(t_arm),
        quiet_ratio=quiet_ratio, min_pairs=min_pairs,
    )


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


# The two knobs that decide WHICH rows enter the stored population. `keep_rate`
# is the realized fraction kept; `keep_limited_frac` is the fraction that hit the
# clamp. `keep_prob_mean` is the intended rate and moves with them, so it is
# compared too. `training_iteration` is provenance, not a population knob, and
# is deliberately NOT compared.
DIFF_FOCUS_POPULATION_KEYS = (
    "diff_focus_keep_rate",
    "diff_focus_keep_limited_frac",
    "diff_focus_keep_prob_mean",
)


def diff_focus_shift(
    a: dict[str, float] | None, b: dict[str, float] | None, *, tol: float,
) -> list[str]:
    """Population knobs that moved between two arms, beyond ``tol``.

    ⚑ UNMEASURED IS NOT UNCHANGED. If either arm has no diff_focus record the
    comparison is reported as UNVERIFIABLE and listed, because "we did not look"
    must not read the same as "it did not move" -- that equivalence is the
    house defect this whole module is instrumented against.
    """
    if not a or not b:
        which = "arm A" if not a else "this run"
        if not a and not b:
            which = "both arms"
        return [f"{which} recorded NO diff_focus (pass --progress-csv to measure it)"]
    out: list[str] = []
    for k in DIFF_FOCUS_POPULATION_KEYS:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            out.append(f"{k} missing on {'arm A' if va is None else 'this run'}")
            continue
        if abs(float(va) - float(vb)) > tol:
            out.append(f"{k} {float(va):.4f} -> {float(vb):.4f}")
    return out


def _parse_floats(spec: str) -> tuple[float, ...]:
    return tuple(float(v) for v in spec.split(",") if v.strip())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Not `required=True` only so `--scan-bank` can run without them; every
    # other path checks for them explicitly and aborts.
    ap.add_argument("--mode", choices=("shards", "research", "simulate"),
                    default=None,
                    help="shards: score production's STORED target. research: "
                         "re-search the same live rows at an explicit shape "
                         "(the isolation arms, on the live population). "
                         "simulate: search frozen deep-SF audit positions.")
    ap.add_argument("--checkpoint", default=None,
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
    ap.add_argument("--diff-focus-tolerance", type=float, default=0.02,
                    help="max absolute move in a diff_focus population knob "
                         "before a --compare-to delta is refused as confounded")
    ap.add_argument("--allow-diff-focus-shift", action="store_true",
                    help="proceed with a --compare-to delta whose diff_focus "
                         "population moved; the confound is printed and banked")
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
    ap.add_argument("--control-bootstrap", type=int, default=500,
                    help="POSITION-cluster resamples for the control MARGIN's own "
                         "95%% interval. The PASS/INVERTED verdict is read off "
                         "that interval; at 0 no cell can be stamped PASS")
    ap.add_argument("--fidelity-min-argmax", type=float,
                    default=FidelityTolerance().floor_min_argmax,
                    help="ABORT floor on argmax agreement with the STORED target, "
                         "applied to every research run")
    ap.add_argument("--fidelity-max-tv", type=float,
                    default=FidelityTolerance().floor_max_tv,
                    help="ABORT ceiling on mean TV against the STORED target")
    ap.add_argument("--fidelity-prod-min-argmax", type=float,
                    default=FidelityTolerance().prod_min_argmax,
                    help="tighter argmax floor applied when the run's shape IS "
                         "PRODUCTION_SEARCH_SHAPE, i.e. when it is the calibration")
    ap.add_argument("--fidelity-prod-max-tv", type=float,
                    default=FidelityTolerance().prod_max_tv,
                    help="tighter TV ceiling for the calibration run")
    ap.add_argument("--production-shape", default=None,
                    help="c_scale,policy_temp,topk,gumbel_scale,sims of the search "
                         "that PRODUCED the shards being read. Defaults to "
                         "PRODUCTION_SEARCH_SHAPE, read off the LIVE yaml (which "
                         "is not this repo's yaml). Declare it when production "
                         "moves; nothing in the shards records it")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="a banked production-shape dump that certifies this "
                         "harness; required for an off-production arm that is not "
                         "already paired against a production-shape --compare-to")
    ap.add_argument("--allow-missing-shard-provenance", action="store_true",
                    help="proceed with --compare-to even though the shards carry "
                         "no producing-net id. Prints an UNVERIFIABLE banner and "
                         "is recorded in the dump; absent provenance is never "
                         "treated as a match on its own")
    ap.add_argument("--row-keys", type=Path, default=None,
                    help="pin the read to the `shard:index` keys in a banked dump "
                         "(reads `row_keys`, or `per_row[].key`). Aborts if any "
                         "key has aged out of the window")
    ap.add_argument("--scan-bank", type=Path, default=None,
                    help="recompute the control-PASS / c_scale-quiet / both counts "
                         "over a banked multi-arm JSON and exit")
    ap.add_argument("--scan-ref-arm", default="A")
    ap.add_argument("--scan-c-arm", default="B")
    ap.add_argument("--scan-t-arm", default="D")
    ap.add_argument("--scan-quiet-ratio", type=float, default=0.25,
                    help="a cell is `c_scale-quiet` when |d_c| <= ratio * |d_T|")
    ap.add_argument("--scan-min-pairs", type=int, default=50)
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


def load_provenance(path: Path) -> dict[str, object]:
    """A banked dump's provenance, without requiring its per-row vectors.

    ``--calibration`` only needs the shape and the fidelity, so it must not
    fail on a summary-only bank the way ``load_dump`` would.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prov = payload.get("provenance")
    if not isinstance(prov, dict):
        raise SystemExit(f"{path} carries no `provenance` block")
    return dict(prov)


def load_row_keys(path: Path) -> list[str]:
    """Row keys from a banked dump: top-level ``row_keys`` or ``per_row[].key``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = payload.get("row_keys")
    if not keys:
        keys = [str(r.get("key", "")) for r in payload.get("per_row", [])]
    keys = [k for k in (str(k) for k in keys) if k]
    if not keys:
        raise SystemExit(
            f"{path} carries no row keys (neither `row_keys` nor `per_row[].key`), "
            "so the arm it describes cannot be reproduced"
        )
    return keys


def load_banked_cells(path: Path) -> list[dict[str, object]]:
    """A banked dump's top-level ``cells`` list, or [] if it has none.

    ``load_dump`` deliberately returns only rows + provenance, and the control
    lives on the CELLS. Older dumps predate the field, so absence is reported
    as unknown by the caller and never as "the control was fine".
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = payload.get("cells")
    return [c for c in cells if isinstance(c, dict)] if isinstance(cells, list) else []


def _control_map(
    cells: list[dict[str, object]],
) -> dict[tuple[float, float, float], tuple[float, float, float]]:
    """Cell -> (margin, ci_lo, ci_hi) reconstructed from a BANKED arm's cells.

    ``control_margin`` is a ``@property``, so ``asdict`` never wrote it; it is
    recomputed here from the stored ``coverage`` / ``shuffled_mean`` /
    ``shuffled_sd`` by the same formula ``CoverageCell.control_margin`` uses.
    """
    def num(value: object, default: float = float("nan")) -> float:
        """A JSON scalar as a float. Anything else is UNKNOWN, i.e. NaN.

        `json.loads` hands back `object`, and a silent `float(None) -> 0.0`
        here would turn "this arm banked no control" into "its control was
        exactly at the null" -- the same absent-reads-as-fine failure the rest
        of this module refuses.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    out: dict[tuple[float, float, float], tuple[float, float, float]] = {}
    for c in cells:
        key = (num(c.get("tau_cp")), num(c.get("rho")), num(c.get("phi")))
        if not all(math.isfinite(k) for k in key):
            continue
        cov = num(c.get("coverage"))
        mu = num(c.get("shuffled_mean"))
        sd = num(c.get("shuffled_sd"))
        margin = (
            (cov - mu) / sd
            if (math.isfinite(cov) and math.isfinite(mu) and sd > 0.0)
            else float("nan")
        )
        out[key] = (margin, num(c.get("margin_ci_lo")), num(c.get("margin_ci_hi")))
    return out


def _bias_map(prov: dict[str, object]) -> dict[tuple[float, float, float], float]:
    """The banked per-cell harness bias of an arm, keyed by cell."""
    rows = prov.get("harness_bias")
    if not isinstance(rows, list):
        return {}
    out: dict[tuple[float, float, float], float] = {}
    for r in rows:
        if isinstance(r, dict):
            out[(float(r["tau_cp"]), float(r["rho"]), float(r["phi"]))] = float(
                r["bias"])
    return out


def require_calibration_anchor(
    shape_gap: list[str], *, calibration: Path | None, compare_to: Path | None,
) -> None:
    """An off-production arm may only be RUN next to a calibration.

    ⚑ `assert_calibrated` STATED THIS REQUIREMENT AND COULD NOT ENFORCE IT.
    It was reached only via `--calibration`, or via `--compare-to` in research
    mode; with neither flag both branches were skipped and the command printed
    cells and exited 0 having asserted nothing. That is the default path, not a
    corner: `--gumbel-scale` defaults to 1.0 against production's 0.5, so a bare
    `--mode research` run is off-production before the operator types anything.

    Checked BEFORE shard selection so it costs a second rather than a full
    re-search, and so it is reachable without a shard bank.
    """
    if not shape_gap or calibration is not None or compare_to is not None:
        return
    raise SystemExit(
        "refusing to report an off-production arm with no calibration: "
        + "; ".join(shape_gap)
        + ". An arm's coverage is only interpretable next to a production-shaped "
        "arm whose stored-minus-re-searched delta bounds the harness error. "
        "Pass --calibration <dump> or --compare-to <dump>, or run the production "
        "shape itself."
    )


def _build_shape(args: argparse.Namespace) -> SimShape:
    shape = SimShape(
        c_scale=float(args.c_scale), policy_temp=float(args.policy_temp),
        topk=int(args.topk), c_visit=float(args.c_visit),
        halving_div=int(args.halving_div), vloss_weight=int(args.vloss_weight),
        add_noise=not args.no_noise, gumbel_scale=float(args.gumbel_scale),
    )
  # Refuse HERE: this is the one place the flags become a shape and every
  # subcommand goes through it, so a value the search would silently drop costs
  # a second instead of a checkpoint load plus a shard sweep. `simulations` is
  # not a validated field, so the sims passed in is immaterial to the check.
    build_sim_gumbel_config(shape, sims=int(getattr(args, "sims", 1) or 1))
    return shape


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scan_bank is not None:
        print(scan_bank(
            args.scan_bank, ref_arm=args.scan_ref_arm, c_arm=args.scan_c_arm,
            t_arm=args.scan_t_arm, quiet_ratio=float(args.scan_quiet_ratio),
            min_pairs=int(args.scan_min_pairs),
        ).report())
        return 0
    if not args.mode or not args.checkpoint:
        raise SystemExit("--mode and --checkpoint are required unless --scan-bank")
    # ⚑ A SHUFFLED RUN IS A NULL, NOT AN ARM, and it may not be differenced
    # against anything. `stored_rows` and any `--compare-to` bank are NOT
    # shuffled, so `stored - re-searched` would price the PERMUTATION and print
    # it under the heading "PURE HARNESS ERROR", and a paired delta would price
    # the permutation and call it a knob effect. Refused up front, before any
    # shard is read, rather than annotated: a mislabelled number in a banked
    # table outlives the caveat that qualified it.
    if args.shuffle != "none" and (
        args.compare_to is not None or args.calibration is not None
    ):
        raise SystemExit(
            f"--shuffle {args.shuffle} cannot be combined with --compare-to/"
            "--calibration: the other arm is NOT shuffled, so the difference "
            "would measure the permutation and be reported as a knob effect. "
            "Run the shuffled arm on its own -- its purpose is the per-cell "
            "control column, which is computed for it like any other run."
        )
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
    stored_rows: list[RowVectors] | None = None
    stats = ShardReadStats()
    is_prod = False
    declared_prod = (
        parse_production_shape(args.production_shape)
        if args.production_shape else None
    )
    if args.mode in ("shards", "research"):
        if not args.replay_dir:
            raise SystemExit(f"--mode {args.mode} requires --replay-dir (ABSOLUTE path)")
        # AFTER the required-argument check -- an operator who forgot
        # `--replay-dir` must be told THAT, not told about calibration -- and
        # BEFORE `select_shards`, so an unanchored arm costs a second rather
        # than a full re-search.
        if args.mode == "research":
            require_calibration_anchor(
                production_shape_mismatches(
                    _build_shape(args), int(args.sims), declared=declared_prod),
                calibration=args.calibration, compare_to=args.compare_to,
            )
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
        tol = FidelityTolerance(
            floor_min_argmax=float(args.fidelity_min_argmax),
            floor_max_tv=float(args.fidelity_max_tv),
            prod_min_argmax=float(args.fidelity_prod_min_argmax),
            prod_max_tv=float(args.fidelity_prod_max_tv),
        )
        if args.mode == "research":
            shape = _build_shape(args)
            research = ResearchSpec(
                shape=shape, sims=int(args.sims), syzygy_path=args.syzygy_path,
                # ⚑ `--seed` is a REAL knob here: ResearchSpec.seed drives the
                # re-search RNG, so omitting it left the dataclass default
                # (20260809) in force and every `--seed N` run was the same run
                # under a different label -- while the provenance block dutifully
                # banked `args.seed` as if it had been used. A value accepted and
                # then silently ignored, and the reason the banked seed and the
                # realized seed have to be the same object.
                seed=int(args.seed),
            )
            fid = ResearchFidelity()
            stored_rows = []
            shape_gap = production_shape_mismatches(
                shape, int(args.sims), declared=declared_prod)
            is_prod = not shape_gap
            provenance["shape_mismatches_vs_production"] = list(shape_gap)
            provenance["shape"] = asdict(shape)
            provenance["sims"] = args.sims
            provenance["is_production_shape"] = is_prod
            provenance["production_shape_assumed"] = (
                declared_prod or dict(
                    PRODUCTION_SEARCH_SHAPE, sims=float(PRODUCTION_SIMS))
            )
            ref = declared_prod or PRODUCTION_SEARCH_SHAPE
            print("[research] production shape ASSUMED c_scale="
                  f"{ref['c_scale']} T={ref['policy_temp']} topk={int(ref['topk'])} "
                  f"gumbel_scale={ref['gumbel_scale']} "
                  f"sims={int(ref.get('sims', PRODUCTION_SIMS))} -- the shards do "
                  "NOT record it; if the live yaml has moved, declare it with "
                  "--production-shape")
            provenance["fidelity_tolerance"] = asdict(tol)
            print(f"[research] re-searching LIVE rows at c_scale={shape.c_scale} "
                  f"T={shape.policy_temp} gumbel_scale={shape.gumbel_scale} "
                  f"topk={shape.topk} sims={args.sims} "
                  f"syzygy={'on' if args.syzygy_path else 'OFF'}")
            print(f"[research] root sigma span at max_visit=59: "
                  f"{shape.root_sigma_span(59.0):.3f} nats")
            print("[research] fidelity GATE: "
                  + (f"PRODUCTION-SHAPE tier -- argmax >= {tol.prod_min_argmax:.2f}, "
                     f"TV <= {tol.prod_max_tv:.2f} (this run IS the calibration)"
                     if is_prod else
                     f"floor tier -- argmax >= {tol.floor_min_argmax:.2f}, "
                     f"TV <= {tol.floor_max_tv:.2f} (off-production arm; it also "
                     "needs a production-shape --compare-to or --calibration)"))
        want_keys = load_row_keys(args.row_keys) if args.row_keys else None
        if want_keys is not None:
            provenance["row_keys_from"] = str(args.row_keys)
        t0 = time.perf_counter()
        rows = read_shard_rows(
            sel, checkpoint=args.checkpoint, device=args.device,
            batch_size=args.batch_size, max_rows=args.max_rows,
            min_sf_p0_rate=args.min_sf_p0_rate, stats=stats,
            research=research, fidelity=fid, want_keys=want_keys,
            stored_out=stored_rows,
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
        else:
            print("[provenance] producing-net id ABSENT on every selected shard "
                  "(ShardMeta.model_step / model_sha256 are None because "
                  "DiskReplayBuffer._flush_shard_arrays writes no meta). This "
                  "readout cannot be proved to come from one producing net.")
        if fid is not None:
            print(f"[research] {fid.report()} in {time.perf_counter() - t0:.1f}s")
            provenance["fidelity"] = asdict(fid)
            fid.assert_within(tol, is_production_shape=is_prod)
        if stats.dropped_non_finite:
            print(f"[rows] DROPPED {stats.dropped_non_finite} row(s) for a "
                  "non-finite policy_target/sf_p0_regret. They are under "
                  "assert_population's tolerance, so the run continues -- but "
                  "they are excluded from coverage rather than scored as "
                  "uncovered.")
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
        if stored_rows is not None:
            print(
                f"[shuffle] {args.shuffle} is ON, so the harness-bias table is "
                "SUPPRESSED: `stored` is not shuffled and the difference would "
                "be the permutation, not harness error"
            )
            stored_rows = None
        title += f"  [SHUFFLE {args.shuffle}]"

    cells = coverage_cells(rows, taus_cp=taus, rhos=rhos, phis=phis)
    for cell in cells:
        lo, hi, _sd = bootstrap_ci(
            rows, tau_cp=cell.tau_cp, rho=cell.rho, phi=cell.phi,
            resamples=int(args.bootstrap), seed=int(args.seed),
        )
        cell.ci_lo, cell.ci_hi = lo, hi
    attach_controls(rows, cells, seeds=int(args.control_seeds), seed0=int(args.seed),
                    resamples=int(args.control_bootstrap))
    print_cells(cells, title=title)
    counts: dict[str, int] = {}
    for c in cells:
        counts[c.verdict()] = counts.get(c.verdict(), 0) + 1
    print("[control] " + ", ".join(
        f"{k} {counts[k]}" for k in sorted(counts)) + f" (of {len(cells)} cells)")

    # ⚑ THE HARNESS'S OWN BIAS, IN THE UNITS OF THE THING IT MEASURES. Computed
    # here, before any threshold is applied to any delta, because the CI was
    # computed for the earlier revision and the bias was not -- and the bias
    # turned out to be the larger of the two at the cell that got pinned.
    bias_map: dict[tuple[float, float, float], float] = {}
    bias_rows: list[dict[str, object]] = []
    if stored_rows:
        kind = (
            "CALIBRATION: production shape, so this is PURE HARNESS ERROR"
            if is_prod else "off-production arm: this is arm effect + error"
        )
        print(f"\n=== harness bias: stored - re-searched, {len(rows)} paired rows "
              f"({kind}) ===")
        print(f"{'tau_cp':>7} {'rho':>7} {'phi':>8} {'bias':>9} {'95% CI':>19}")
        for c in cells:
            d, lo, hi = paired_delta_ci(
                rows, stored_rows, tau_cp=c.tau_cp, rho=c.rho, phi=c.phi,
                resamples=int(args.bootstrap), seed=int(args.seed),
            )
            bias_map[(c.tau_cp, c.rho, c.phi)] = d
            bias_rows.append({"tau_cp": c.tau_cp, "rho": c.rho, "phi": c.phi,
                              "bias": d, "ci_lo": lo, "ci_hi": hi})
            print(f"{c.tau_cp:7.0f} {c.rho:7.3f} {c.phi:8.1e} {d:+9.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]")
        provenance["harness_bias"] = bias_rows

    paired: list[dict[str, object]] = []
    if args.calibration is not None:
        prov_cal = load_provenance(args.calibration)
        assert_calibrated(
            this_shape=_build_shape(args), this_sims=int(args.sims),
            other=prov_cal, other_label=str(args.calibration.name),
            declared=declared_prod,
        )
    if args.compare_to is not None:
        rows_a, prov_a = load_dump(args.compare_to)
        ck_a = str(prov_a.get("checkpoint", ""))
        if ck_a and ck_a != str(args.checkpoint):
            raise SystemExit(
                f"refusing to compare readouts taken against different reference "
                f"priors: arm A used {ck_a!r}, this run used {args.checkpoint!r}"
            )
        # Simulated arms are paired by AUDIT-POSITION key and have no producing
        # shards at all, so the shard-provenance guard has nothing to check and
        # its UNKNOWN branch would fire unconditionally -- forcing the operator
        # to pass an unrelated `--allow-missing-shard-provenance`, which then
        # ALSO waives the guard for any shard-backed arm in the same session.
        # An escape hatch reached for the wrong reason stops being a record.
        if args.mode in ("shards", "research"):
            assert_same_producing_net(
                stats, prov_a,
                allow_missing=bool(args.allow_missing_shard_provenance),
            )
        else:
            print("[provenance] simulate mode: no producing shards on either "
                  "side, so the producing-net guard does not apply (arms are "
                  "paired by audit-position key)")
        provenance["provenance_unverified"] = bool(
            args.allow_missing_shard_provenance
        )
        # ⚑ THE REFUSAL `read_diff_focus`'s DOCSTRING PROMISES. It says the
        # value is "returned so both readouts can record it and the delta can be
        # refused when it moved" -- and nothing refused. `diff_focus` drops
        # policy rows as a function of the same KL the arms move, so the two
        # arms' STORED POPULATIONS differ and the coverage delta is confounded
        # by re-composition. `assert_population`'s rate check is invariant to a
        # uniform keep_prob shift and cannot see it.
        #
        # ⚑ AND IT IS SHARD-BACKED, so it carries the SAME simulate-mode
        # exemption as the producing-net guard directly above. `diff_focus`
        # provenance is read off the stored shards; in simulate mode NEITHER arm
        # has any, `diff_focus_shift(None, None)` reports a shift, and the run
        # dies unless the operator passes `--allow-diff-focus-shift` -- an
        # unrelated waiver, reached for the wrong reason, which then also waives
        # a REAL diff_focus shift for any shard-backed arm in the same session.
        # That is precisely the escape-hatch defect the block above fixes, so
        # fixing one and not the other left `--mode simulate --compare-to`
        # unreachable without it.
        if args.mode in ("shards", "research"):
            df_b = provenance.get("diff_focus")
            df_a = prov_a.get("diff_focus")
            shifted = diff_focus_shift(
                df_a if isinstance(df_a, dict) else None,
                df_b if isinstance(df_b, dict) else None,
                tol=float(args.diff_focus_tolerance),
            )
        else:
            shifted = []
            print("[diff_focus] simulate mode: neither arm is shard-backed, so "
                  "there is no stored population to re-compose and the "
                  "diff_focus gate does not apply")
        provenance["diff_focus_shift"] = shifted
        if shifted:
            msg = ("refusing to difference two arms whose diff_focus population "
                   "changed: " + "; ".join(shifted) + ". These knobs decide WHICH "
                   "rows are stored, so the delta mixes a coverage change with a "
                   "population change.")
            if args.allow_diff_focus_shift:
                print(f"[diff_focus] ⚑ CONFOUNDED, proceeding on the record: {msg}")
                provenance["diff_focus_shift_allowed"] = True
            else:
                raise SystemExit(
                    msg + " Pass --allow-diff-focus-shift to proceed on the "
                    "record that this comparison is confounded."
                )
        if args.mode == "research" and args.calibration is None:
            assert_calibrated(
                this_shape=_build_shape(args), this_sims=int(args.sims),
                other=prov_a, other_label=str(args.compare_to.name),
                declared=declared_prod,
            )
        paired = print_paired(
            rows_a, rows, cells=cells, resamples=int(args.bootstrap),
            seed=int(args.seed), label_a=str(args.compare_to.name), label_b="this run",
            bias=_bias_map(prov_a),
            control_a=_control_map(load_banked_cells(args.compare_to)),
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provenance": provenance,
            "n_rows": len(rows),
            "row_keys": [row.key for row in rows],
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
