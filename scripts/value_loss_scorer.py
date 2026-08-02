#!/usr/bin/env python3
"""Value-head error where the ORACLE says the side to move is LOST.

The failure profile this exists to measure: 80% of losses to Cheese are one
collapse, and the value head is symmetrically under-confident by ~98 cp against
its OWN target. Both statements are about the losing tail specifically, and
until now no packaged tool scored that tail — every readout was a one-off in a
scratchpad, with its own POV convention and its own (usually row-level) CI.

WHAT PICKS THE SAMPLE, AND WHY IT MATTERS
    The stratum is chosen by the row's stored ``sf_wdl`` label — the Stockfish
    evaluation of the row's OWN position, in the row's own side-to-move POV,
    the same tensor ``train/losses.py`` blends into the WDL target. Nothing the
    net emits touches the selection. That is not a style preference: a filter
    keyed to the net's own output makes the denominator a function of the thing
    under test, and the resulting error rate cannot be compared to anything.
    Only POSITION-level filters may gate a denominator here.

    Consequence worth stating out loud: this is a filter, and a single-stratum
    error rate is uninterpretable on its own. So the ``rest`` stratum (every
    eligible row the oracle does NOT call lost) is scored by the identical
    instrument and printed beside it, and the number to read is the CONTRAST.
    A raw error rate that moves is not evidence; the contrast against the
    control is.

WHY NOT ``scripts/value_optimism.py``
    That script is the level instrument for the same head and shares this
    module's conventions (it exports ``expected_score``,
    ``expected_score_to_cp`` and ``cluster_bootstrap_ci``, all reused here so
    the two cannot drift). Three things it structurally cannot do:

    * Its head-vs-target arm needs two rows that are consecutive plies of one
      game, and curriculum games never produce such a pair — measured pairing
      rate: selfplay 24.2%, curriculum 0.00%. So that arm is 100% SELFPLAY by
      construction, and the PID-handicapped Stockfish only plays in CURRICULUM
      games. This script needs no pairing: it scores each row against its own
      stored label, so curriculum rows are in scope, and ``--split-selfplay``
      reports the two populations separately.
    * It stratifies and reports bucket MEANS. A collapse is a tail event; the
      quantiles of the per-row error are the shape that shows it, and a mean
      over a bucket hides it.
    * It banks no per-position dump, so a surprising number cannot be re-read
      without paying the forward pass again.

WHAT IT REPORTS, PER STRATUM
    ``e2_sf``    sum over (W,D,L) of (net - sf_wdl)^2, per row. Plain squared
                 distance between the two distributions; 0 is exact agreement,
                 2 is maximal disagreement.
    ``e2_out``   the same against the realized outcome's one-hot (the
                 multi-class Brier score), on the rows that carry an outcome.
    ``d_score``  signed expected-score error, net minus oracle, where
                 score = W + 0.5*D. POSITIVE MEANS THE NET IS MORE OPTIMISTIC
                 THAN THE ORACLE — in a lost-position stratum that is the
                 quantity the "under-confident by ~98 cp" claim is about.
    ``d_loss``   net P(loss) minus oracle P(loss). Negative = the net does not
                 call the loss as hard as the oracle does.
    ``d_cp``     ``d_score`` re-expressed in centipawns through the exact
                 inverse of the production cp-logistic. Reported with its clamp
                 rate, because a silently pinned tail is how an absolute cp bar
                 stops meaning what its name says.

    Each gets a mean, a 95% CI, and the p05/p25/p50/p75/p95 of its per-row
    distribution. Every CI resamples GAMES, not rows: rows inside a game are
    consecutive plies of one position sequence, and a row-level bootstrap
    reports a CI several times too tight.

NEGATIVE CONTROLS (both runnable, neither optional when quoting a number)
    ``--shuffle-rows``     permutes the net's predictions across the eligible
                           rows, destroying the position <-> prediction
                           association while leaving both marginals intact.
    ``--shuffle-weights``  permutes the elements inside every parameter tensor
                           of the loaded model, in memory. A genuinely shuffled
                           NET: same architecture, same parameter histogram, no
                           information about the position.

    ⚑ THE COLUMN THE CONTROL KILLS IS ``net_score``, NOT THE ERROR COLUMNS.
    ``net_score`` is the net's own level, so a stratum-blind head must read the
    same value in both strata and its contrast has a null of exactly zero.
    Every error column is a difference against a reference that itself varies
    by stratum — the oracle score is ~0.02 where it says lost and ~0.6
    elsewhere — so a shuffled head reads as hugely "optimistic" in the lost
    stratum: ``d_score``'s shuffle null is around +0.44, not 0. Judging an
    error column against a null of zero would call a DESTROYED association a
    finding. ``--self-test`` T3 pins both halves of that.

    A weight shuffle is also not an information-free head: it is still a
    deterministic function of the input, so its null is SMALL, not zero. Read
    the control's contrast as a FRACTION of the measured one.

    ``--self-test`` runs both controls as assertions, so they ship as tests
    rather than as things someone remembers to do.

READ-ONLY over ``data/``. CPU by default. Forward-only, no gradients.

    PYTHONPATH=. nice -n 19 python3 scripts/value_loss_scorer.py \
        --checkpoint data/ratchet/snapshots/ck_2026-08-01_iter478.pt \
        --shard-dir data/c17_ab/pre --max-rows 400 --dump /tmp/lost.npz

    PYTHONPATH=. python3 scripts/value_loss_scorer.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chess_anti_engine.eval.value_optimism import (
    CP_CLAMP,
    cluster_bootstrap_ci,
    expected_score,
    expected_score_to_cp,
)
from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.moves.encode import POLICY_SIZE
from chess_anti_engine.train.trainer import SfTargetParams
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

# Shard fields the scorer needs. ``x`` is the exact tensor training and
# selfplay saw, so absolute numbers here are not subject to the frozen-ruler
# defect where a FEN-only input leaves 117 of the 175 planes zero.
SHARD_FIELDS: tuple[str, ...] = (
    "x", "sf_wdl", "has_sf_wdl", "is_network_turn", "game_id", "ply_index",
)
# Fields used when present and defaulted (visibly) when absent.
SHARD_OPTIONAL: tuple[str, ...] = ("wdl_target", "is_selfplay")

# ``wdl_target`` is int8 0=W / 1=D / 2=L, side-to-move POV, matching sf_wdl's
# column order. -1 marks "this row carries no outcome" and is never scored.
NO_OUTCOME: int = -1

QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)


def _norm3(p: np.ndarray) -> np.ndarray:
    """clamp_min(0) then renormalise — mirrors losses._normalize_sf_wdl_probs.

    Copied from the verified extraction pattern rather than re-derived: the
    stored ``sf_wdl`` is float16 and can carry small negatives, and the loss
    clamps before it normalises. Doing it in the other order changes the
    label this script is scoring against.
    """
    q = np.clip(np.asarray(p, dtype=np.float64), 0.0, None)
    return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-30)


def _softmax3(a: np.ndarray) -> np.ndarray:
    z = np.asarray(a, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def to_probs(wdl: np.ndarray) -> np.ndarray:
    """Head output -> probabilities, whichever convention the head emits.

    ``ChessNet``'s wdl head has emitted logits in some revisions and a softmax
    in others. Assuming one gives a silently wrong number under the other, so
    the convention is DETECTED per batch and the choice is reported.
    """
    w = np.asarray(wdl, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != 3:
        raise ValueError(f"expected (N,3) wdl output, got {w.shape!r}")
    if np.all(w >= -1e-6) and np.allclose(w.sum(axis=1), 1.0, atol=1e-3):
        return _norm3(w)
    return _softmax3(w)


@dataclass(frozen=True)
class RowSet:
    """Eligible rows plus everything needed to select, score and cluster them.

    ``p_sf`` is already normalised. ``outcome`` is ``NO_OUTCOME`` on rows with
    no realized result. ``game_id`` is the loop's own content hash of the whole
    game (``selfplay/finalize._stable_game_id``), so it is unique across shards
    and constant across a game that spans two of them — it is the cluster.
    """

    x: np.ndarray
    p_sf: np.ndarray
    outcome: np.ndarray
    game_id: np.ndarray
    ply: np.ndarray
    is_selfplay: np.ndarray
    source: str
    # Whether ``is_selfplay`` was MEASURED or merely defaulted. A row source
    # that did not carry the field yields all-False, which is indistinguishable
    # from "every row is curriculum" — so ``--split-selfplay`` refuses rather
    # than reporting a population split it did not observe. ``game_id`` needs
    # no such flag: it is required at every entry point, because it is
    # load-bearing for EVERY CI rather than for one optional flag.
    population_known: bool = True

    def __len__(self) -> int:
        return int(self.p_sf.shape[0])

    def take(self, idx: np.ndarray) -> RowSet:
        return RowSet(
            x=self.x[idx], p_sf=self.p_sf[idx], outcome=self.outcome[idx],
            game_id=self.game_id[idx], ply=self.ply[idx],
            is_selfplay=self.is_selfplay[idx], source=self.source,
            population_known=self.population_known,
        )


def _shard_indices(shard_dir: str, max_index: int, stride: int) -> list[str]:
    names = [
        n for n in sorted(os.listdir(shard_dir))
        if (m := re.match(r"shard_(\d+)", n)) and int(m.group(1)) < max_index
    ]
    return names[::max(1, stride)]


def load_rows_from_shards(
    shard_dir: str, *, max_index: int = 10 ** 9, stride: int = 1,
    max_rows: int = 0,
) -> RowSet:
    """Eligible rows from zarr replay shards. Read-only.

    Eligibility is ``is_network_turn & has_sf_wdl``: a row with no oracle label
    cannot be placed in either stratum, and a non-network turn is not a
    position the value head was trained to score. Both are position-level.
    """
    import zarr  # imported here so --self-test needs no zarr

    cols: dict[str, list[np.ndarray]] = {k: [] for k in (
        "x", "p_sf", "outcome", "game_id", "ply", "is_selfplay",
    )}
    absent: set[str] = set()
    kept = 0
    for name in _shard_indices(shard_dir, max_index, stride):
        # ``zarr.open`` is typed ``Array | Group``; this handle is a group.
        g: Any = zarr.open(os.path.join(shard_dir, name), mode="r")
        keys = set(g.array_keys())
        if not set(SHARD_FIELDS) <= keys:
            print(f"skip {name}: missing {sorted(set(SHARD_FIELDS) - keys)}", flush=True)
            continue
        sel = (
            np.asarray(g["is_network_turn"][:]).astype(bool)
            & np.asarray(g["has_sf_wdl"][:]).astype(bool)
        )
        idx = np.where(sel)[0]
        if max_rows > 0:
            idx = idx[: max(0, max_rows - kept)]
        if idx.size == 0:
            if max_rows > 0 and kept >= max_rows:
                break
            continue
        cols["x"].append(np.asarray(g["x"].oindex[idx, :, :, :], dtype=np.float32))
        cols["p_sf"].append(_norm3(np.asarray(g["sf_wdl"].oindex[idx, :])))
        cols["game_id"].append(np.asarray(g["game_id"][:])[idx].astype(np.int64))
        cols["ply"].append(np.asarray(g["ply_index"][:])[idx].astype(np.int32))
        if "wdl_target" in keys:
            cols["outcome"].append(np.asarray(g["wdl_target"][:])[idx].astype(np.int16))
        else:
            cols["outcome"].append(np.full(idx.size, NO_OUTCOME, np.int16))
        if "is_selfplay" in keys:
            cols["is_selfplay"].append(np.asarray(g["is_selfplay"][:])[idx].astype(bool))
        else:
            cols["is_selfplay"].append(np.zeros(idx.size, bool))
        absent |= set(SHARD_OPTIONAL) - keys
        kept += int(idx.size)
        if max_rows > 0 and kept >= max_rows:
            break
    if kept == 0:
        raise SystemExit(f"no eligible rows under {shard_dir}")
    if absent:
        # Said out loud rather than silently defaulted: without ``wdl_target``
        # every outcome column reads NaN, and without ``is_selfplay`` the
        # curriculum/selfplay split would silently report everything as
        # curriculum.
        print(f"NOTE: optional fields absent in some shards: {sorted(absent)}", flush=True)
    return RowSet(
        x=np.concatenate(cols["x"]), p_sf=np.concatenate(cols["p_sf"]),
        outcome=np.concatenate(cols["outcome"]),
        game_id=np.concatenate(cols["game_id"]), ply=np.concatenate(cols["ply"]),
        is_selfplay=np.concatenate(cols["is_selfplay"]),
        source=f"shards:{shard_dir}",
        population_known="is_selfplay" not in absent,
    )


def load_rows_from_npz(path: str) -> RowSet:
    """Eligible rows from a banked npz.

    The contract is exactly this script's own ``--dump --dump-x`` output, so a
    dump is a valid row source and a run can be reproduced without re-reading
    the shards.

    THREE REQUIRED KEYS, each because defaulting it corrupts a number silently:

    * ``x`` — without the planes there is no forward pass, and a bank carrying
      only the previous run's predictions would let a stale net's numbers be
      re-reported under a new checkpoint's name.
    * ``p_sf`` — the oracle label IS the stratum; there is nothing to select on
      without it.
    * ``game_id`` — **the CLUSTER**. This one used to default to
      ``np.arange(n)``, i.e. one game per row, which does not fail, does not
      warn, and silently turns every game-clustered CI in the output into a
      ROW-level bootstrap. Rows inside a game are consecutive plies of one
      position sequence, so that CI is several times too tight — the
      anti-conservative direction, and exactly how a null gets published as a
      finding. ``test_ci_resamples_games_not_rows`` measures the gap at >3x on
      correlated rows. A bank with no cluster identity cannot be scored
      honestly by this tool, so it is refused rather than scored wrongly.

    ``outcome``, ``ply`` and ``is_selfplay`` may be absent. Their defaults
    degrade VISIBLY: a missing ``outcome`` shows as ``n_used == 0`` and NaN on
    the outcome columns, and a missing ``is_selfplay`` makes
    ``--split-selfplay`` refuse (see ``population_known``) instead of reporting
    everything as curriculum.
    """
    with np.load(path) as z:
        keys = set(z.files)
        missing = {"x", "p_sf", "game_id"} - keys
        if missing:
            raise SystemExit(
                f"{path}: missing {sorted(missing)}. A row source needs the input "
                "planes (x), the oracle label (p_sf) and the game cluster "
                "(game_id) — game_id is what every CI resamples, and defaulting "
                "it to one-game-per-row would silently report row-level "
                "bootstraps that are several times too tight. Re-dump with "
                "--dump-x; scorer-written banks always carry all three."
            )
        n = int(z["p_sf"].shape[0])
        game_id = np.asarray(z["game_id"], dtype=np.int64)
        if game_id.shape != (n,):
            raise SystemExit(
                f"{path}: game_id has shape {game_id.shape!r}, expected ({n},) "
                "— one cluster id per row"
            )
        return RowSet(
            x=np.asarray(z["x"], dtype=np.float32),
            p_sf=_norm3(np.asarray(z["p_sf"])),
            outcome=(
                np.asarray(z["outcome"], dtype=np.int16) if "outcome" in keys
                else np.full(n, NO_OUTCOME, np.int16)
            ),
            game_id=game_id,
            ply=(
                np.asarray(z["ply"], dtype=np.int32) if "ply" in keys
                else np.zeros(n, np.int32)
            ),
            is_selfplay=(
                np.asarray(z["is_selfplay"], dtype=bool) if "is_selfplay" in keys
                else np.zeros(n, bool)
            ),
            source=f"npz:{path}",
            population_known="is_selfplay" in keys,
        )


def select_oracle_lost(
    p_sf: np.ndarray, *, loss_prob_min: float, sf_cp_max: float | None,
    slope: float, draw_width_cp: float,
) -> np.ndarray:
    """THE stratum filter. A function of the ORACLE LABEL and nothing else.

    ``p_sf`` is the only argument that carries data; the net is not in scope
    here and cannot be. Both criteria are ANDed when both are given.

    ``loss_prob_min`` reads the label directly. ``sf_cp_max`` reads the same
    label through the inverse cp-logistic, for when a claim is phrased in
    centipawns; it is the same quantity in another unit, not a second opinion.
    """
    if not 0.0 < loss_prob_min <= 1.0:
        raise ValueError(f"loss_prob_min must be in (0, 1], got {loss_prob_min}")
    mask = np.asarray(p_sf)[:, 2] >= float(loss_prob_min)
    if sf_cp_max is not None:
        cp, _ = expected_score_to_cp(
            expected_score(p_sf), slope=slope, draw_width_cp=draw_width_cp,
        )
        mask &= cp <= float(sf_cp_max)
    return mask


def net_wdl_probs(
    evaluator: LocalModelEvaluator, x: np.ndarray, *, batch_size: int,
) -> np.ndarray:
    """Value-head probabilities for every row, in row order."""
    out: list[np.ndarray] = []
    for start in range(0, int(x.shape[0]), max(1, batch_size)):
        xb = np.ascontiguousarray(x[start: start + max(1, batch_size)])
        _, wdl = evaluator.evaluate_encoded(xb)
        out.append(np.asarray(wdl, dtype=np.float32))
    return to_probs(np.concatenate(out, axis=0))


def shuffle_model_weights(model: torch.nn.Module, *, seed: int) -> int:
    """Permute the elements INSIDE every parameter tensor. In memory only.

    Same architecture, same per-tensor value histogram, zero information about
    the input. That is a stronger control than reinitialising from a prior:
    a reinit changes the weight distribution too, so a collapsed effect could
    be blamed on the distribution rather than on the destroyed association.

    Returns the number of tensors permuted, so a caller can assert the control
    actually did something — a shuffle that silently permuted nothing is a
    control that cannot fail.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    touched = 0
    with torch.no_grad():
        for param in model.parameters():
            if param.numel() < 2:
                continue
            flat = param.detach().reshape(-1)
            perm = torch.randperm(flat.numel(), generator=gen)
            param.copy_(flat[perm].reshape(param.shape))
            touched += 1
    return touched


@dataclass(frozen=True)
class ErrorStat:
    """One stratum's value-error readout. Every CI is game-clustered."""

    name: str
    n: int
    n_games: int
    n_outcome: int
    oracle_loss_prob: float
    oracle_score: float
    net_score: float
    means: dict[str, float]
    cis: dict[str, tuple[float, float]]
    quantiles: dict[str, tuple[float, ...]]
    # PER-COLUMN denominator. A column computed on fewer rows than the stratum
    # holds is a different sample wearing the stratum's name, and the only
    # legitimate reason for that here is a POSITION-level fact (the row carries
    # no outcome). Reporting it per column is what makes an accidental shrink
    # visible instead of silent.
    n_used: dict[str, int]
    cp_clamped_frac: float


def _per_row_errors(
    rows: RowSet, q: np.ndarray, *, slope: float, draw_width_cp: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Per-row error columns, the outcome mask, and the cp clamp rate."""
    p_sf = rows.p_sf
    has_out = rows.outcome != NO_OUTCOME
    onehot = np.zeros_like(p_sf)
    onehot[np.arange(len(rows)), np.clip(rows.outcome, 0, 2)] = 1.0

    net_cp, net_clamped = expected_score_to_cp(
        expected_score(q), slope=slope, draw_width_cp=draw_width_cp,
    )
    sf_cp, sf_clamped = expected_score_to_cp(
        expected_score(p_sf), slope=slope, draw_width_cp=draw_width_cp,
    )
    cols = {
        # The net's own LEVEL, not an error. It is the column the negative
        # controls have to kill: under a shuffle the net's predictions carry
        # no information about the stratum, so its lost-minus-rest contrast
        # must collapse to 0. Every OTHER column below is a difference against
        # a reference that itself varies by stratum, so NONE of them has a
        # null of zero under the shuffle -- see the T3 assertion in self_test.
        "net_score": expected_score(q),
        "e2_sf": ((q - p_sf) ** 2).sum(axis=1),
        "e2_out": np.where(has_out, ((q - onehot) ** 2).sum(axis=1), np.nan),
        "d_score": expected_score(q) - expected_score(p_sf),
        "d_score_out": np.where(has_out, expected_score(q) - expected_score(onehot), np.nan),
        "d_loss": q[:, 2] - p_sf[:, 2],
        "d_cp": net_cp - sf_cp,
    }
    clamp_rate = float(np.mean(net_clamped | sf_clamped)) if len(rows) else float("nan")
    return cols, has_out, clamp_rate


def score_stratum(
    name: str, rows: RowSet, q: np.ndarray, *,
    slope: float, draw_width_cp: float, n_boot: int, seed: int,
) -> ErrorStat:
    """Mean, game-clustered 95% CI and per-row quantiles for one stratum."""
    rng = np.random.default_rng(seed)
    cols, has_out, clamp_rate = _per_row_errors(
        rows, q, slope=slope, draw_width_cp=draw_width_cp,
    )
    means: dict[str, float] = {}
    cis: dict[str, tuple[float, float]] = {}
    quants: dict[str, tuple[float, ...]] = {}
    used: dict[str, int] = {}
    for key, val in cols.items():
        # The outcome columns are NaN where no result was recorded. Dropping
        # them is a POSITION-level exclusion (the row has no outcome, whatever
        # the net says), so it cannot condition the denominator on the net.
        keep = np.isfinite(val)
        v, gid = val[keep], rows.game_id[keep]
        used[key] = int(v.size)
        if v.size == 0:
            means[key] = float("nan")
            cis[key] = (float("nan"), float("nan"))
            quants[key] = tuple(float("nan") for _ in QUANTILES)
            continue
        means[key] = float(v.mean())
        cis[key] = cluster_bootstrap_ci(v, gid, n_boot=n_boot, rng=rng)
        quants[key] = tuple(float(x) for x in np.quantile(v, QUANTILES))
    return ErrorStat(
        name=name, n=len(rows), n_games=int(np.unique(rows.game_id).size),
        n_outcome=int(has_out.sum()),
        oracle_loss_prob=float(rows.p_sf[:, 2].mean()) if len(rows) else float("nan"),
        oracle_score=float(expected_score(rows.p_sf).mean()) if len(rows) else float("nan"),
        net_score=float(expected_score(q).mean()) if len(rows) else float("nan"),
        means=means, cis=cis, quantiles=quants, n_used=used,
        cp_clamped_frac=clamp_rate,
    )


@dataclass(frozen=True)
class Contrast:
    """lost-minus-rest for one column, with a game-clustered CI.

    THE number to read. Each stratum's raw level is confounded by everything
    that differs between winning and losing positions; the difference of the
    two, resampled over games, is what a negative control has to kill.
    """

    column: str
    delta: float
    ci: tuple[float, float]

    @property
    def excludes_zero(self) -> bool:
        lo, hi = self.ci
        return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0))


def _per_game_sums(
    values: np.ndarray, game_id: np.ndarray, games: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(sum, count) of ``values`` per entry of ``games``, zero where absent.

    ``games`` is the SHARED pool across both strata, so the two returned
    vectors are index-aligned and a bootstrap draw can index them with one
    index array.
    """
    idx = np.searchsorted(games, game_id)
    sums = np.zeros(games.size, dtype=np.float64)
    counts = np.zeros(games.size, dtype=np.float64)
    np.add.at(sums, idx, np.asarray(values, dtype=np.float64))
    np.add.at(counts, idx, 1.0)
    return sums, counts


def contrast(
    lost: RowSet, rest: RowSet, q_lost: np.ndarray, q_rest: np.ndarray, *,
    column: str, slope: float, draw_width_cp: float, n_boot: int, seed: int,
) -> Contrast:
    """``mean(column | lost) - mean(column | rest)``, bootstrapped over GAMES.

    Both strata are resampled in the same bootstrap draw, so the CI is on the
    difference rather than on two independently-resampled means.
    """
    rng = np.random.default_rng(seed)
    a, _, _ = _per_row_errors(lost, q_lost, slope=slope, draw_width_cp=draw_width_cp)
    b, _, _ = _per_row_errors(rest, q_rest, slope=slope, draw_width_cp=draw_width_cp)
    va, vb = a[column], b[column]
    ka, kb = np.isfinite(va), np.isfinite(vb)
    va, ga = va[ka], lost.game_id[ka]
    vb, gb = vb[kb], rest.game_id[kb]
    if va.size == 0 or vb.size == 0:
        return Contrast(column=column, delta=float("nan"), ci=(float("nan"), float("nan")))
    delta = float(va.mean() - vb.mean())
    # Paired over the shared game pool: a game that contributes rows to BOTH
    # strata must enter or leave a bootstrap draw as one unit, or the CI
    # ignores the within-game correlation that spans the two strata. Drawing
    # the two strata independently would also make the difference's CI wrong
    # in the anti-conservative direction whenever a game straddles the bar.
    #
    # A resampled mean is (sum of the drawn games' sums) / (their row counts),
    # so per-game sums and counts are all that is needed and the whole
    # bootstrap is one matrix index. Grouping the rows per draw instead is
    # O(n_boot x games x rows) and turns a full-window read into minutes.
    games = np.unique(np.concatenate([ga, gb]))
    sum_a, cnt_a = _per_game_sums(va, ga, games)
    sum_b, cnt_b = _per_game_sums(vb, gb, games)
    draws = rng.integers(0, games.size, size=(max(1, n_boot), games.size))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a = sum_a[draws].sum(axis=1) / cnt_a[draws].sum(axis=1)
        mean_b = sum_b[draws].sum(axis=1) / cnt_b[draws].sum(axis=1)
    boot = mean_a - mean_b
    boot = boot[np.isfinite(boot)]
    if boot.size < 2:
        return Contrast(column=column, delta=delta, ci=(float("nan"), float("nan")))
    return Contrast(
        column=column, delta=delta,
        ci=(float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))),
    )


@dataclass(frozen=True)
class Report:
    """Everything a run produced, in the order it should be read."""

    lost: ErrorStat
    rest: ErrorStat
    contrasts: list[Contrast]
    meta: dict[str, object] = field(default_factory=dict)


# ``net_score`` first: it is the control column (null 0 under a shuffle).
# The rest are the substance, and their shuffle nulls are NOT zero.
CONTRAST_COLUMNS: tuple[str, ...] = ("net_score", "d_score", "e2_sf", "d_loss", "d_cp")


def build_report(
    rows: RowSet, q: np.ndarray, *, loss_prob_min: float, sf_cp_max: float | None,
    slope: float, draw_width_cp: float, n_boot: int, seed: int,
    meta: dict[str, object] | None = None,
) -> tuple[Report, np.ndarray]:
    """Select, score both strata, and contrast them. Returns the lost mask too."""
    mask = select_oracle_lost(
        rows.p_sf, loss_prob_min=loss_prob_min, sf_cp_max=sf_cp_max,
        slope=slope, draw_width_cp=draw_width_cp,
    )
    lost, rest = rows.take(mask), rows.take(~mask)
    q_lost, q_rest = q[mask], q[~mask]
    report = Report(
        lost=score_stratum(
            "lost", lost, q_lost, slope=slope, draw_width_cp=draw_width_cp,
            n_boot=n_boot, seed=seed,
        ),
        rest=score_stratum(
            "rest", rest, q_rest, slope=slope, draw_width_cp=draw_width_cp,
            n_boot=n_boot, seed=seed + 1,
        ),
        contrasts=[
            contrast(
                lost, rest, q_lost, q_rest, column=c, slope=slope,
                draw_width_cp=draw_width_cp, n_boot=n_boot, seed=seed + 2,
            )
            for c in CONTRAST_COLUMNS
        ],
        meta=dict(meta or {}),
    )
    return report, mask


def render(report: Report) -> str:
    """The whole readout as text. Every number printed is one this run made."""
    lines: list[str] = []
    for key, val in sorted(report.meta.items()):
        lines.append(f"# {key}: {val}")
    head = (
        f"{'stratum':>8} {'n':>7} {'games':>6} {'n_out':>6} "
        f"{'orcl_pL':>8} {'orcl_sc':>8} {'net_sc':>8}"
    )
    lines += ["", head, "-" * len(head)]
    for st in (report.lost, report.rest):
        lines.append(
            f"{st.name:>8} {st.n:>7} {st.n_games:>6} {st.n_outcome:>6} "
            f"{st.oracle_loss_prob:>8.4f} {st.oracle_score:>8.4f} {st.net_score:>8.4f}"
        )
    for st in (report.lost, report.rest):
        lines += ["", f"[{st.name}] n={st.n} games={st.n_games} "
                      f"cp_clamped={st.cp_clamped_frac:.3f}"]
        lines.append(
            f"{'column':>12} {'rows':>6} {'mean':>10} {'ci_lo':>10} {'ci_hi':>10} "
            + " ".join(f"{f'p{int(p * 100):02d}':>9}" for p in QUANTILES)
        )
        for col in st.means:
            lo, hi = st.cis[col]
            qs = " ".join(f"{x:>9.4f}" for x in st.quantiles[col])
            lines.append(
                f"{col:>12} {st.n_used[col]:>6} {st.means[col]:>10.4f} "
                f"{lo:>10.4f} {hi:>10.4f} {qs}"
            )
    lines += ["", "CONTRAST  lost - rest  (this is the number to read)"]
    for c in report.contrasts:
        flag = "excludes 0" if c.excludes_zero else "CONTAINS 0"
        lines.append(
            f"{c.column:>12} {c.delta:>+10.4f}  95% CI [{c.ci[0]:+.4f}, {c.ci[1]:+.4f}]  {flag}"
        )
    return "\n".join(lines)


def write_dump(
    path: str, rows: RowSet, q: np.ndarray, mask: np.ndarray, *,
    with_x: bool, meta: dict[str, object],
) -> None:
    """Bank the per-position rows, not just the aggregate.

    A banked number that cannot be re-read is a number that has to be trusted.
    With ``--dump-x`` the file is itself a valid ``--npz`` row source, so the
    same rows can be re-scored under another checkpoint with no shard access.
    """
    payload: dict[str, np.ndarray] = {
        "p_sf": rows.p_sf.astype(np.float32),
        "p_net": q.astype(np.float32),
        "oracle_lost": mask.astype(bool),
        "outcome": rows.outcome.astype(np.int16),
        "game_id": rows.game_id.astype(np.int64),
        "ply": rows.ply.astype(np.int32),
        "is_selfplay": rows.is_selfplay.astype(bool),
    }
    if with_x:
        payload["x"] = rows.x.astype(np.float32)
    saver: Callable[..., Any] = np.savez_compressed
    saver(path, **payload)
    Path(path + ".meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True, default=str), encoding="utf-8",
    )


# --------------------------------------------------------------------------
# self-test: the negative controls, shipped as assertions
# --------------------------------------------------------------------------
class _TinyWdlNet(torch.nn.Module):
    """A real ``nn.Module`` with the head shape ``LocalModelEvaluator`` expects.

    It exists so ``--shuffle-weights`` can be exercised on an actual parameter
    tensor through the actual evaluator path, rather than on a stub that fakes
    the outcome the control is supposed to demonstrate.
    """

    def __init__(self, planes: int) -> None:
        super().__init__()
        self.wdl = torch.nn.Linear(planes * 64, 3)
        self.policy = torch.nn.Linear(planes * 64, POLICY_SIZE)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = x.reshape(x.shape[0], -1)
        return {"wdl": self.wdl(flat), "policy": self.policy(flat)}


def _synthetic_rows(*, n_games: int, plies: int, seed: int) -> RowSet:
    """Rows whose oracle label is a smooth function of one input plane."""
    rng = np.random.default_rng(seed)
    n = n_games * plies
    planes, x = 4, np.zeros((n_games * plies, 4, 8, 8), np.float32)
    lost_frac = rng.random(n)
    x[:, 0, 0, 0] = lost_frac
    x[:, 1:, :, :] = rng.random((n, planes - 1, 8, 8)).astype(np.float32)
    p_loss = 0.05 + 0.94 * lost_frac
    p_sf = np.stack([1.0 - p_loss - 0.02, np.full(n, 0.02), p_loss], axis=1)
    outcome = np.where(p_loss > 0.5, 2, 0).astype(np.int16)
    return RowSet(
        x=x, p_sf=_norm3(p_sf), outcome=outcome,
        game_id=np.repeat(np.arange(n_games, dtype=np.int64), plies),
        ply=np.tile(np.arange(plies, dtype=np.int32), n_games),
        is_selfplay=np.zeros(n, bool), source="synthetic",
    )


def _column(report: Report, name: str) -> Contrast:
    return next(c for c in report.contrasts if c.column == name)


def _self_test_report(
    rows: RowSet, q: np.ndarray, *, seed: int = 0,
) -> Report:
    par = SfTargetParams()
    return build_report(
        rows, q, loss_prob_min=0.75, sf_cp_max=None,
        slope=par.sf_wdl_cp_slope, draw_width_cp=par.sf_wdl_cp_draw_width,
        n_boot=400, seed=seed,
    )[0]


def self_test() -> int:
    """Assert the instrument reads what it claims, and that the controls kill it.

    Six checks, each of which fails on a specific way the tool could be wrong:

    T1  a head that reproduces the oracle exactly reads ~0 error everywhere.
    T2  an injected optimism on lost rows is recovered at its true size, with
        the contrast excluding 0.
    T3  the ROW-shuffle control collapses that contrast onto 0.
    T4  a genuinely WEIGHT-SHUFFLED net, run through the real evaluator,
        collapses it too — and the shuffle is verified to have touched
        parameters, so the control cannot pass by doing nothing.
    T5  the selection is a function of the oracle alone: two different heads
        select the identical rows.
    T6  the banked dump round-trips to identical aggregates.
    """
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    rows = _synthetic_rows(n_games=40, plies=12, seed=7)
    lost_mask = select_oracle_lost(
        rows.p_sf, loss_prob_min=0.75, sf_cp_max=None,
        slope=SfTargetParams().sf_wdl_cp_slope,
        draw_width_cp=SfTargetParams().sf_wdl_cp_draw_width,
    )
    print(f"self-test: {len(rows)} rows, {int(lost_mask.sum())} oracle-lost, "
          f"{int(np.unique(rows.game_id).size)} games")

    # T1 — a perfect head.
    perfect = _self_test_report(rows, rows.p_sf.copy())
    check("T1 perfect head reads zero error",
          abs(perfect.lost.means["d_score"]) < 1e-9
          and abs(perfect.lost.means["e2_sf"]) < 1e-9
          and abs(perfect.rest.means["d_score"]) < 1e-9,
          f"lost d_score={perfect.lost.means['d_score']:.2e}")

    # T2 — inject optimism on the LOST rows only, plus per-row noise so the
    # CI is a real interval rather than a degenerate point.
    inject = 0.20
    noise = np.random.default_rng(21).normal(0.0, 0.01, size=len(rows))
    q = rows.p_sf.copy()
    q[lost_mask, 0] += inject
    q[lost_mask, 2] -= inject
    q[:, 0] = np.clip(q[:, 0] + noise, 1e-4, None)
    q = _norm3(q)
    real = _self_test_report(rows, q)
    d_score = _column(real, "d_score")
    net_lvl = _column(real, "net_score")
    check("T2 injected optimism recovered on the lost stratum",
          abs(real.lost.means["d_score"] - inject) < 0.02
          and abs(real.rest.means["d_score"]) < 0.02
          and d_score.excludes_zero and net_lvl.excludes_zero,
          f"lost={real.lost.means['d_score']:+.4f} rest={real.rest.means['d_score']:+.4f} "
          f"contrast={d_score.delta:+.4f} CI[{d_score.ci[0]:+.4f},{d_score.ci[1]:+.4f}]")

    # T3 — the ROW-shuffle control, read on the column whose null IS zero.
    #
    # ⚑ THE TRAP THIS TEST PINS. Under a shuffle only ``net_score`` has a null
    # of zero: it is the net's own level, and a shuffled net's level cannot
    # depend on the stratum. Every error column is a difference against a
    # reference that varies by stratum, so its shuffle null is LARGE and
    # POSITIVE by construction (the oracle score is ~0 in the lost stratum and
    # ~0.5 in the rest, so a stratum-blind head reads as hugely "optimistic").
    # Reading ``d_score`` against a null of zero would therefore call a
    # DESTROYED association a significant finding. The second assertion is the
    # one that would catch someone re-pointing this control at ``d_score``.
    perm = np.random.default_rng(3).permutation(len(rows))
    shuffled = _self_test_report(rows, q[perm])
    lvl_shuf = _column(shuffled, "net_score")
    d_shuf = _column(shuffled, "d_score")
    check("T3 row-shuffle collapses net_score, and d_score's null is NOT zero",
          (not lvl_shuf.excludes_zero) and d_shuf.excludes_zero,
          f"net_score={lvl_shuf.delta:+.4f} CI[{lvl_shuf.ci[0]:+.4f},{lvl_shuf.ci[1]:+.4f}] | "
          f"d_score={d_shuf.delta:+.4f} (nonzero null, as documented)")

    # T4 — a real weight-shuffled net through the real evaluator path. Seeded:
    # a self-test whose control number moves run to run cannot be quoted, and
    # the reader could not tell drift from noise.
    torch.manual_seed(101)
    net = _TinyWdlNet(planes=int(rows.x.shape[1])).eval()
    before = [p.detach().clone() for p in net.parameters()]
    touched = shuffle_model_weights(net, seed=11)
    moved = sum(
        1 for a, b in zip(before, net.parameters()) if not torch.equal(a, b.detach())
    )
    q_net = net_wdl_probs(
        LocalModelEvaluator(net, device="cpu", use_amp=False), rows.x, batch_size=64,
    )
    shuffled_net = _self_test_report(rows, q_net)
    lvl_net = _column(shuffled_net, "net_score")
    # ⚑ A SHUFFLED NET IS STILL A DETERMINISTIC FUNCTION OF THE INPUT, so its
    # null is SMALL, not exactly zero: a random map retains some correlation
    # with any input feature that happens to encode the stratum, and on this
    # synthetic set one plane does exactly that. The bar is therefore "the
    # control kills at least 90% of the signal", judged against the real arm's
    # own contrast rather than against zero. Quoting a shuffled-net reading as
    # "≈0, so the control passed" is the mistake this phrasing prevents.
    ratio = abs(lvl_net.delta) / max(1e-12, abs(net_lvl.delta))
    check("T4 weight-shuffled net kills >=90% of the net_score contrast",
          touched >= 2 and moved >= 2 and ratio < 0.10,
          f"tensors_permuted={touched} tensors_changed={moved} "
          f"net_score={lvl_net.delta:+.4f} vs real {net_lvl.delta:+.4f} "
          f"(ratio {ratio:.3f})")

    # T5 — the selection never sees the net.
    other = select_oracle_lost(
        rows.p_sf, loss_prob_min=0.75, sf_cp_max=None,
        slope=SfTargetParams().sf_wdl_cp_slope,
        draw_width_cp=SfTargetParams().sf_wdl_cp_draw_width,
    )
    check("T5 stratum is a function of the oracle alone",
          bool(np.array_equal(lost_mask, other)) and int(lost_mask.sum()) > 0,
          f"n_lost={int(lost_mask.sum())}")

    # T6 — the dump round-trips.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dump.npz")
        write_dump(path, rows, q, lost_mask, with_x=True, meta={"self_test": True})
        again = load_rows_from_npz(path)
        with np.load(path) as z:
            q_again = to_probs(np.asarray(z["p_net"], dtype=np.float64))
        round_trip = _self_test_report(again, q_again)
        check("T6 banked dump round-trips to identical aggregates",
              abs(round_trip.lost.means["d_score"] - real.lost.means["d_score"]) < 1e-9
              and round_trip.lost.n == real.lost.n,
              f"n={round_trip.lost.n} d_score={round_trip.lost.means['d_score']:+.6f}")

    print("self-test: " + ("ALL PASS" if not failures else f"FAILED {failures}"))
    return 0 if not failures else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    par = SfTargetParams()
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--checkpoint")
    src = ap.add_argument_group("row source (exactly one)")
    src.add_argument("--shard-dir")
    src.add_argument("--npz")
    ap.add_argument("--max-index", type=int, default=10 ** 9)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = no cap")
    ap.add_argument("--loss-prob-min", type=float, default=0.75,
                    help="oracle P(loss) at or above this = the LOST stratum")
    ap.add_argument("--sf-cp-max", type=float, default=None,
                    help="additionally require the oracle cp at or below this")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cp-slope", type=float, default=par.sf_wdl_cp_slope)
    ap.add_argument("--cp-draw-width", type=float, default=par.sf_wdl_cp_draw_width)
    ap.add_argument("--shuffle-rows", action="store_true",
                    help="negative control: permute predictions across rows")
    ap.add_argument("--shuffle-weights", action="store_true",
                    help="negative control: permute the model's own weights")
    ap.add_argument("--split-selfplay", action="store_true",
                    help="also report the selfplay and curriculum populations separately")
    ap.add_argument("--dump", help="bank the per-position rows to this npz")
    ap.add_argument("--dump-x", action="store_true",
                    help="include the input planes, making the dump a valid --npz source")
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        return self_test()
    if bool(args.shard_dir) == bool(args.npz):
        raise SystemExit("give exactly one of --shard-dir / --npz")
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required")

    rows = (
        load_rows_from_shards(
            args.shard_dir, max_index=args.max_index, stride=args.stride,
            max_rows=args.max_rows,
        )
        if args.shard_dir else load_rows_from_npz(args.npz)
    )
    # Checked HERE, before the checkpoint load and the forward pass: a run that
    # discovers its requested split is unmeasurable only after paying for
    # inference has wasted the whole run. Everything a flag needs is knowable
    # from the row source alone, so it is settled at the row source.
    if args.split_selfplay and not rows.population_known:
        raise SystemExit(
            f"--split-selfplay: {rows.source} carries no is_selfplay field, so "
            "every row would default to curriculum and the 'selfplay' half "
            "would print as empty — a population split nobody observed"
        )

    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    touched = shuffle_model_weights(model, seed=args.seed) if args.shuffle_weights else 0
    if args.shuffle_weights and touched < 2:
        raise SystemExit("--shuffle-weights permuted nothing; the control is void")
    q = net_wdl_probs(
        LocalModelEvaluator(model, device=args.device), rows.x, batch_size=args.batch_size,
    )
    if args.shuffle_rows:
        q = q[np.random.default_rng(args.seed).permutation(len(rows))]

    meta: dict[str, object] = {
        "checkpoint": args.checkpoint,
        "source": rows.source,
        "rows_eligible": len(rows),
        "loss_prob_min": args.loss_prob_min,
        "sf_cp_max": args.sf_cp_max,
        "cp_slope": args.cp_slope,
        "cp_draw_width": args.cp_draw_width,
        "cp_clamp": CP_CLAMP,
        "device": args.device,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "control_shuffle_rows": bool(args.shuffle_rows),
        "control_shuffle_weights": bool(args.shuffle_weights),
        "weight_tensors_permuted": touched,
        "torch": torch.__version__,
    }
    loss_prob_min, sf_cp_max = float(args.loss_prob_min), args.sf_cp_max
    slope, draw_width = float(args.cp_slope), float(args.cp_draw_width)
    n_boot, seed = int(args.n_boot), int(args.seed)
    report, mask = build_report(
        rows, q, loss_prob_min=loss_prob_min, sf_cp_max=sf_cp_max,
        slope=slope, draw_width_cp=draw_width, n_boot=n_boot, seed=seed, meta=meta,
    )
    print(render(report))
    if args.split_selfplay:
        # The two populations are not interchangeable: the PID-handicapped
        # Stockfish plays ONLY in curriculum games, so a claim about the
        # handicap read off the selfplay half is read off the population where
        # the mechanism cannot operate. Split, never pooled, when that is the
        # question being asked.
        for label, sub in (("selfplay", rows.is_selfplay), ("curriculum", ~rows.is_selfplay)):
            if not sub.any():
                print(f"\n===== {label}: no rows =====")
                continue
            part, _ = build_report(
                rows.take(sub), q[sub],
                loss_prob_min=loss_prob_min, sf_cp_max=sf_cp_max, slope=slope,
                draw_width_cp=draw_width, n_boot=n_boot, seed=seed,
                meta={**meta, "population": label, "rows_eligible": int(sub.sum())},
            )
            print(f"\n===== {label} =====")
            print(render(part))
    if args.dump:
        write_dump(args.dump, rows, q, mask, with_x=args.dump_x, meta=meta)
        print(f"\nbanked {len(rows)} rows -> {args.dump} (+ .meta.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
