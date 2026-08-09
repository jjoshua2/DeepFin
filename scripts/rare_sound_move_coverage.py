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

⚑ COVERAGE IS **NOT** FLAT IN ``c_scale``, AND THE FLOOR ``phi`` DECIDES THE
SIGN. The prereg's table asserts flatness. That would hold if the stored target
were supported on the Gumbel CANDIDATE set, which only ``policy_temp`` moves --
but ``mcts/gumbel.py::_build_improved_policy_for_board`` builds the improved
policy over EVERY legal move, handing unvisited moves the root's completed-Q.
So sigma reaches every entry of the target and ``phi`` is not sigma-invariant.

Measured through ``--mode simulate``, 400 frozen audit positions at 256 sims,
position-paired, tau=25cp, rho=0.01 (95% bootstrap CI over positions):

    phi     c_scale 0.025->0.1     policy_temp 1.0->1.5    bundle (both)
    1e-4    -0.138 [-0.183,-0.098] +0.112 [+0.076,+0.151]  -0.006 [-0.050,+0.039] ns
    1e-3    -0.182 [-0.234,-0.131] +0.256 [+0.208,+0.304]  +0.071 [+0.017,+0.124]
    7e-3    -0.062 [-0.096,-0.030] +0.421 [+0.365,+0.479]  +0.203 [+0.149,+0.262]
    1e-2    +0.032 [+0.015,+0.053] +0.382 [+0.326,+0.444]  +0.247 [+0.194,+0.303]

The ``c_scale`` term is strongly NEGATIVE at a small floor, crosses zero
between phi 7e-3 and 1e-2, and is mildly positive above. At **phi = 1e-2** the
``c_scale`` half moves coverage 12x less than the ``policy_temp`` half, so that
is the cell to read the bundle on: a bundle delta at or above ~+0.15 there is
not reachable by ``c_scale`` alone. At phi = 1e-4 the two halves CANCEL to a
null -- reading the bundle there would report "nothing happened" while both
knobs fired, which is the exact failure this axis was added to prevent.

That near-flat cell is a CANCELLATION, not a structural invariance. It depends
on the sim count and the position mix, so re-run the two isolation arms at the
deployed shape before trusting it -- do not carry the number across a config
change.

TWO MODES, ONE METRIC.

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

``--mode simulate`` (the discriminating experiment)
    Runs the real C Gumbel search over frozen deep-SF audit positions at an
    explicit ``--c-scale`` / ``--policy-temp`` and scores the SAME coverage on
    the improved policy the search returns. Two arms differing only in
    ``c_scale`` measure the flatness claim; two differing only in
    ``policy_temp`` measure the effect the live readout has to detect.

CONTROLS. ``--shuffle target`` permutes the stored target within each row's
legal moves, destroying its association with both soundness and rarity while
preserving the legal set and the target's own marginal. Coverage must collapse
to the base rate ``P(target >= phi)`` over legal moves. ``--shuffle prior``
permutes the prior instead, so ``rare`` becomes a size-matched random subset.
A metric that survives either shuffle is measuring something structural.
``tests/test_rare_sound_move_coverage.py`` ships both as assertions.

Usage::

    PYTHONPATH=. python3 scripts/rare_sound_move_coverage.py \\
        --mode shards --checkpoint data/ruler_reads_20260808/trainer.pt \\
        --replay-dir /abs/path/to/<trial>/replay_shards --shards 40

    PYTHONPATH=. python3 scripts/rare_sound_move_coverage.py \\
        --mode simulate --checkpoint data/ruler_reads_20260808/trainer.pt \\
        --audit-set data/audit_set_v1.jsonl --positions 400 \\
        --c-scale 0.1 --policy-temp 1.0 --sims 256
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
    base_rate: float      # P(target >= phi) over ALL legal moves, the chance level
    ci_lo: float = float("nan")
    ci_hi: float = float("nan")


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
    is dominated by which positions each arm happened to draw.
    """
    if len(rows_a) != len(rows_b):
        raise ValueError(
            f"paired_delta_ci needs equal-length arms, got {len(rows_a)} and {len(rows_b)}"
        )
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
                regret_cp=row.regret_cp, scored=row.scored,
            ))
        else:
            out.append(RowVectors(
                prior=row.prior[perm], target=row.target,
                regret_cp=row.regret_cp, scored=row.scored,
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
    field_present: dict[str, int] = field(default_factory=dict)


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


def read_shard_rows(
    sel: ShardSelection,
    *,
    checkpoint: str,
    device: str,
    batch_size: int,
    max_rows: int,
    min_sf_p0_rate: float,
    stats: ShardReadStats,
) -> list[RowVectors]:
    """Live shard rows -> RowVectors, with the prior from the stored ``x``.

    The prior is a forward pass on the planes production searched with, so it
    is exact up to checkpoint drift: nothing is decoded back to a board and no
    history is lost. The checkpoint's declared encodings are checked against
    each shard's, because a mismatch would feed the net planes it never saw.
    """
    import torch
    import zarr

    from chess_anti_engine.inference import LocalModelEvaluator
    from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY
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

    rows: list[RowVectors] = []
    for path in reversed(sel.paths):
        z = zarr.open(path, mode="r")
        assert_required_fields(z, path)
        assert_encodings(dict(z.attrs), ck_hist=ck_hist, ck_pol=ck_pol, path=path)

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
                pol, _wdl = evaluator.evaluate_encoded(xs[sl])
            logits = _compact_logits(np.asarray(pol, dtype=np.float64), compact_to_full)
            for j in range(logits.shape[0]):
                k = start + j
                lm = legal[k]
                if not lm.any():
                    continue
                lg = logits[j][lm]
                lg = lg - lg.max()
                e = np.exp(lg)
                s = float(e.sum())
                if not math.isfinite(s) or s <= 0.0:
                    continue
                reg_row = regret[k]
                scored_full = _scored_mask_from_regret(reg_row)
                rows.append(RowVectors(
                    prior=e / s,
                    target=target[k][lm],
                    regret_cp=reg_row[lm] * SF_OWN_REGRET_CAP_CP,
                    scored=scored_full[lm],
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
            cfg=cfg, evaluator=evaluator, tb_probe=None,
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
            ))
            fens.append(pos.fen)
    return rows, fens


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_cells(cells: list[CoverageCell], *, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"{'tau_cp':>7} {'rho':>7} {'phi':>8} {'pairs':>7} {'rows':>6} "
          f"{'coverage':>9} {'95% CI':>17} {'chance':>7}")
    for c in cells:
        ci = (
            f"[{c.ci_lo:.4f},{c.ci_hi:.4f}]"
            if math.isfinite(c.ci_lo) else "                 "
        )
        cov = f"{c.coverage:.4f}" if math.isfinite(c.coverage) else "     nan"
        print(f"{c.tau_cp:7.0f} {c.rho:7.3f} {c.phi:8.1e} {c.n_pairs:7d} "
              f"{c.n_rows:6d} {cov:>9} {ci:>17} {c.base_rate:7.4f}")


def _parse_floats(spec: str) -> tuple[float, ...]:
    return tuple(float(v) for v in spec.split(",") if v.strip())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", choices=("shards", "simulate"), required=True)
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
    ap.add_argument("--audit-set", default="data/audit_set_v1.jsonl")
    ap.add_argument("--positions", type=int, default=400)
    ap.add_argument("--sims", type=int, default=256)
    ap.add_argument("--c-scale", type=float, default=0.025)
    ap.add_argument("--policy-temp", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--no-noise", action="store_true",
                    help="disable the root Gumbel perturbation (selfplay runs it ON)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--taus", default=",".join(str(v) for v in DEFAULT_TAUS_CP))
    ap.add_argument("--rhos", default=",".join(str(v) for v in DEFAULT_RHOS))
    ap.add_argument("--phis", default=",".join(str(v) for v in DEFAULT_PHIS))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--shuffle", choices=("none", "target", "prior"), default="none",
                    help="negative control: permute one vector within each row")
    ap.add_argument("--check-alignment", action="store_true",
                    help="mode=shards: prove which SF field describes this row")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the cells + provenance as JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    taus = _parse_floats(args.taus)
    rhos = _parse_floats(args.rhos)
    phis = _parse_floats(args.phis)

    provenance: dict[str, object] = {
        "mode": args.mode, "checkpoint": args.checkpoint, "seed": args.seed,
        "shuffle": args.shuffle, "taus_cp": list(taus), "rhos": list(rhos),
        "phis": list(phis),
    }

    if args.mode == "shards":
        if not args.replay_dir:
            raise SystemExit("--mode shards requires --replay-dir (ABSOLUTE path)")
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
        stats = ShardReadStats()
        rows = read_shard_rows(
            sel, checkpoint=args.checkpoint, device=args.device,
            batch_size=args.batch_size, max_rows=args.max_rows,
            min_sf_p0_rate=args.min_sf_p0_rate, stats=stats,
        )
        print(f"[shards] rows total {stats.rows_total}, net-turn policy "
              f"{stats.rows_net_policy}, with sf_p0_regret {stats.rows_with_sf_p0} "
              f"({100 * stats.rows_with_sf_p0 / max(1, stats.rows_net_policy):.1f}%), "
              f"used {stats.rows_used}")
        provenance["read_stats"] = asdict(stats)
        title = f"coverage over {len(rows)} live shard rows"
    else:
        shape = SimShape(
            c_scale=float(args.c_scale), policy_temp=float(args.policy_temp),
            topk=int(args.topk), add_noise=not args.no_noise,
        )
        print(f"[sim] c_scale={shape.c_scale} policy_temp={shape.policy_temp} "
              f"topk={shape.topk} sims={args.sims} noise={shape.add_noise}")
        print(f"[sim] root sigma span at max_visit=59: "
              f"{shape.root_sigma_span(59.0):.3f} nats")
        t0 = time.perf_counter()
        rows, fens = simulate_rows(
            checkpoint=args.checkpoint, audit_set=args.audit_set,
            positions=args.positions, sims=args.sims, shape=shape,
            device=args.device, batch_size=args.batch_size, seed=args.seed,
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
    print_cells(cells, title=title)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provenance": provenance,
            "n_rows": len(rows),
            "cells": [asdict(c) for c in cells],
            "per_row": [
                {
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
