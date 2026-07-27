#!/usr/bin/env python3
"""How much of ``test_wdl_loss`` is same-game memorisation? (audit G13)

The replay holdout is split per ROW: ``_ingest_train_arrays`` draws
``rng.random(shard_n) < holdout_fraction`` uniformly over every ply of every
game. Games contribute ~17.8 recorded plies to the window, so a holdout row has
~22.6 expected same-game rows in the TRAINING set and ``P(no same-game sibling)
= 7.2e-04``; ``wdl_target`` is constant within a game. Every holdout row's value
label is therefore already in the training set attached to ~23 near-identical
positions, and ``freeze_holdout_at`` makes that permanent rather than rotating
it out. That is a mechanism, not a magnitude. This probe measures the magnitude
so the decision to change the split is made on a number.

WHAT IT MEASURES. One checkpoint, scored twice over two row sets:

  A  LEAKY    — the per-row 2% holdout, i.e. today's construction. Rows the
                checkpoint did not train on whose GAMES it did.
  B  DISJOINT — a game-disjoint 2% draw: rows whose ``game_id`` contributed
                ZERO rows to that checkpoint's training window.

The reported number is ``mean_A - mean_B`` of the blended value loss
(``wdl_ce``, the tensor ``total`` is built from — the same quantity the
``test_wdl_loss`` column reports), with a CI. Negative = the per-row holdout
looks easier than genuinely unseen games = the leak is real and that big.

  gap < 0.05 nats  => G13 is measured-immaterial; close it, no ledger entry.
  gap >= 0.05 nats => open a ledger entry for a per-game split, with a revert
                      point, and decide it there.

HOW THE SETS ARE CARVED. The checkpoint's training window is every shard in the
replay dir older than the checkpoint (``--boundary``); shards newer than it hold
rows the checkpoint never saw. Set B is drawn from those newer shards, filtered
to games with zero rows in the window (asserted, not assumed). Set A comes from
``holdout.npz`` beside ``trainer.pt`` (PR #261) when the checkpoint carries one —
those are literally the rows ``test_wdl_loss`` was computed on. Without a
sidecar, ``--set-a reconstruct`` redraws the production per-row split over the
window; those rows were themselves trained on, so that variant reports an UPPER
bound on the gap (direct memorisation + the sibling leak), and says so on every
line of output.

The two sets are subsampled to identical row count and identical W/D/L
histogram. Subsampled rather than reweighted so that every number printed stays
an unweighted mean over real rows — the same estimator as ``test_wdl_loss`` —
instead of a weighted mean whose effective sample size has to be tracked
separately.

Deterministic full pass: every selected row is scored exactly once, in a fixed
order, at ``holdout_fraction`` with no priority weighting and no WDL
rebalancing. It never calls ``sample_batch_arrays``. That is deliberate — the
production holdout eval (G14) is 2560 draws WITH REPLACEMENT from <=2000 rows,
WDL-rebalanced and 50% priority-weighted (ESS 52.6%, noise floor sd 0.0522
nats), which is larger than the effect under test.

Read-only. It opens zarr groups directly and never constructs a
``DiskReplayBuffer`` — that would enforce the window and DELETE live shards
(G12).

WHAT IT DOES NOT DO. It does not change the split, and it is not evidence that
changing the split helps. A per-game split is a separate, ledger-gated
experiment whose entry this number decides whether to open.

Usage (a checkpoint a few iterations old, so newer shards exist to draw B from):

  PYTHONPATH=. python3 scripts/probe_holdout_split_leak.py \\
      --checkpoint runs/pbt2_small/tune/train_trial_*/checkpoint_000058 \\
      --dry-run                      # composition only, no GPU, no model load
"""
from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chess_anti_engine.replay.dataset import collate_arrays
from chess_anti_engine.replay.shard import (
    INPUT_HISTORY_ENCODING_ARRAY_KEY,
    iter_shard_paths,
    load_shard_arrays,
    shard_index,
)
from chess_anti_engine.train.losses import compute_loss
from chess_anti_engine.train.trainer import (
    select_input_history_arrays,
    trainer_kwargs_from_config,
)
from chess_anti_engine.tune._utils import SIDECAR_HOLDOUT_ROWS
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint
from scripts.paired_compare import paired_bootstrap_ci
from scripts.trial_paths import latest_result, latest_trial_dir

# Pre-committed decision threshold. Stated in the PR that added this probe and
# in docs/rl_loop_audit.md G13; do not tune it after reading a result.
LEAK_THRESHOLD_NATS = 0.05

WDL_NAMES = ("win", "draw", "loss")

# Cheap per-row columns. Everything the carving needs; none of them is `x`, so
# the whole window can be indexed without decoding a single position tensor.
INDEX_FIELDS = ("game_id", "has_game_id", "wdl_target", "ply_index", "is_selfplay")


# --------------------------------------------------------------------------
# shard index
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    index: int
    n_rows: int
    mtime: float
    is_link: bool

    @property
    def regime(self) -> str:
        """``salvage`` for a symlink into a salvage pool, else ``local``.

        G8: 54% of the live window is symlinks into
        ``data/salvage/swap_512x16_20260711`` whose content is 15-18 days old,
        and the two halves differ in ``is_selfplay`` (0.96 vs 0.70) and draw
        rate (0.25 vs 0.33). Mixing them across the two sets would put a regime
        difference inside the measured gap.
        """
        return "salvage" if self.is_link else "local"


@dataclass(frozen=True)
class RowIndex:
    """Per-row identity columns for a list of shards, in shard order."""

    shard_pos: np.ndarray  # index into the ShardInfo list
    row: np.ndarray  # row within that shard
    game_id: np.ndarray
    wdl: np.ndarray
    ply: np.ndarray
    is_selfplay: np.ndarray

    def __len__(self) -> int:
        return int(self.shard_pos.shape[0])

    def take(self, sel: np.ndarray) -> RowIndex:
        sel = np.asarray(sel, dtype=np.int64)
        return RowIndex(
            shard_pos=self.shard_pos[sel], row=self.row[sel], game_id=self.game_id[sel],
            wdl=self.wdl[sel], ply=self.ply[sel], is_selfplay=self.is_selfplay[sel],
        )


def scan_shards(replay_dir: Path, *, regime: str, shard_slice: str) -> list[ShardInfo]:
    """List the window's shards with age and provenance. Never writes."""
    paths = iter_shard_paths(replay_dir)
    if not paths:
        raise FileNotFoundError(f"no replay shards under {replay_dir}")
    lo, hi = _parse_slice(shard_slice, len(paths))
    out: list[ShardInfo] = []
    for path in paths[lo:hi]:
        try:
            arrs, _meta = load_shard_arrays(path, lazy=True)
            n_rows = int(np.asarray(arrs["x"].shape)[0])
        except (KeyError, OSError, ValueError):
            continue
        # stat() follows the symlink, so mtime is CONTENT age (what G8 read),
        # not the age of the link the resume created.
        out.append(ShardInfo(
            path=path, index=shard_index(path), n_rows=n_rows,
            mtime=path.stat().st_mtime, is_link=path.is_symlink(),
        ))
    if regime != "all":
        out = [s for s in out if s.regime == regime]
    if not out:
        raise FileNotFoundError(f"no shards left under {replay_dir} after regime={regime}")
    return out


def _parse_slice(spec: str, n: int) -> tuple[int, int]:
    if not spec or spec == ":":
        return 0, n
    head, _, tail = spec.partition(":")
    lo = int(head) if head else 0
    hi = int(tail) if tail else n
    return max(0, lo), min(n, hi)


def read_row_index(shards: list[ShardInfo]) -> RowIndex:
    """Read the identity columns of every row in *shards* (no ``x`` decode)."""
    parts: list[tuple[int, dict[str, np.ndarray]]] = []
    for pos, info in enumerate(shards):
        arrs, _meta = load_shard_arrays(info.path, lazy=True)
        cols = {name: np.asarray(arrs[name]) for name in INDEX_FIELDS if name in arrs}
        parts.append((pos, cols))

    def _col(cols: dict[str, np.ndarray], name: str, n: int, fill: int) -> np.ndarray:
        got = cols.get(name)
        if got is None:
            return np.full((n,), fill, dtype=np.int64)
        # A short column would concatenate into a RowIndex whose fields have
        # different lengths, i.e. silently misaligned identities rather than an
        # error. Refuse instead.
        if int(got.shape[0]) != n:
            raise ValueError(f"shard column {name!r} has {got.shape[0]} rows, expected {n}")
        return got.astype(np.int64)

    shard_pos, row, game_id, wdl, ply, selfplay = [], [], [], [], [], []
    for pos, cols in parts:
        n = int(shards[pos].n_rows)
        has_gid = cols.get("has_game_id")
        gid = _col(cols, "game_id", n, -1)
        if has_gid is not None:
            gid = np.where(np.asarray(has_gid) > 0, gid, -1)
        shard_pos.append(np.full((n,), pos, dtype=np.int64))
        row.append(np.arange(n, dtype=np.int64))
        game_id.append(gid)
        wdl.append(_col(cols, "wdl_target", n, 1))
        ply.append(_col(cols, "ply_index", n, -1))
        selfplay.append(_col(cols, "is_selfplay", n, 0))
    return RowIndex(
        shard_pos=np.concatenate(shard_pos), row=np.concatenate(row),
        game_id=np.concatenate(game_id), wdl=np.concatenate(wdl),
        ply=np.concatenate(ply), is_selfplay=np.concatenate(selfplay),
    )


# --------------------------------------------------------------------------
# carving
# --------------------------------------------------------------------------


def per_row_holdout_mask(
    counts: list[int], *, holdout_fraction: float, seed: int,
) -> np.ndarray:
    """Reproduce the production per-row split over shards of size *counts*.

    Mirrors ``_ingest_train_arrays``: one ``rng.random(shard_n) <
    holdout_frac`` draw per shard, from a single generator advanced in shard
    order. Same expression, same call shape, same sequence — so the mask this
    returns has the production split's statistics, not merely its mean.
    """
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.random(int(n)) < float(holdout_fraction) for n in counts
    ]) if counts else np.zeros((0,), dtype=bool)


def sibling_stats(game_id: np.ndarray, holdout_mask: np.ndarray) -> dict[str, float]:
    """The G13 numbers, recomputed on whatever rows were actually loaded.

    ``expected_siblings`` is the mean over holdout rows of how many rows of the
    same game stayed in training; ``p_no_sibling`` is the fraction of holdout
    rows with none. G13 quoted 22.6 and 7.2e-04 on live data — a run whose
    figures differ by a lot is looking at a different window, and the gap it
    reports means something different too.
    """
    valid = game_id >= 0
    train_games, train_counts = np.unique(game_id[valid & ~holdout_mask], return_counts=True)
    held = game_id[valid & holdout_mask]
    if held.size == 0:
        return {"n_holdout": 0.0, "expected_siblings": 0.0, "p_no_sibling": 0.0,
                "plies_per_game": 0.0}
    pos = np.searchsorted(train_games, held)
    pos = np.clip(pos, 0, max(0, train_games.size - 1))
    found = (train_games.size > 0) & (train_games[pos] == held)
    siblings = np.where(found, train_counts[pos] if train_games.size else 0, 0)
    all_games = np.unique(game_id[valid])
    return {
        "n_holdout": float(held.size),
        "expected_siblings": float(siblings.mean()),
        "p_no_sibling": float((siblings == 0).mean()),
        "plies_per_game": float(valid.sum() / max(1, all_games.size)),
    }


def assert_game_disjoint(game_id: np.ndarray, train_games: np.ndarray) -> None:
    """Raise unless *game_id* shares no game with the training window.

    The whole comparison rests on this, so it is checked rather than trusted:
    a single shared game means set B carries the same leak as set A and the
    measured gap is an underestimate of unknown size.
    """
    shared = np.intersect1d(np.unique(game_id[game_id >= 0]), train_games)
    if shared.size:
        raise AssertionError(
            f"game-disjoint set is contaminated: {shared.size} game_id(s) also "
            f"appear in the training window (first: {shared[:5].tolist()})",
        )


def carve_game_disjoint(
    index: RowIndex, *, train_games: np.ndarray, target_rows: int, rng: np.random.Generator,
) -> np.ndarray:
    """Pick whole games with zero training rows until *target_rows* is reached.

    Returns row positions into *index*. Games are taken whole — that is the
    point of the set — so the row count lands on or just past the target.
    """
    eligible = (index.game_id >= 0) & ~np.isin(index.game_id, train_games)
    games = np.unique(index.game_id[eligible])
    if games.size == 0:
        return np.zeros((0,), dtype=np.int64)
    rng.shuffle(games)
    order = np.argsort(index.game_id[eligible], kind="stable")
    eligible_pos = np.flatnonzero(eligible)[order]
    sorted_games = index.game_id[eligible_pos]
    picked: list[np.ndarray] = []
    total = 0
    for game in games:
        lo, hi = np.searchsorted(sorted_games, [game, game + 1])
        picked.append(eligible_pos[lo:hi])
        total += int(hi - lo)
        if total >= target_rows:
            break
    return np.sort(np.concatenate(picked)) if picked else np.zeros((0,), dtype=np.int64)


def match_wdl_mix(
    wdl_a: np.ndarray, wdl_b: np.ndarray, *, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample both sets to an identical W/D/L histogram (and so row count).

    Subsampling, not reweighting: a weighted mean would keep more rows but stop
    being the estimator ``test_wdl_loss`` uses, and its effective sample size
    would then have to be carried into every CI. Dropping rows keeps both
    reported means plain unweighted means over real rows, and the cost is
    visible in the printed retention.
    """
    sel_a: list[np.ndarray] = []
    sel_b: list[np.ndarray] = []
    for cls in range(3):
        idx_a = np.flatnonzero(wdl_a == cls)
        idx_b = np.flatnonzero(wdl_b == cls)
        keep = min(idx_a.size, idx_b.size)
        if keep == 0:
            continue
        sel_a.append(rng.choice(idx_a, size=keep, replace=False))
        sel_b.append(rng.choice(idx_b, size=keep, replace=False))
    if not sel_a:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.sort(np.concatenate(sel_a)), np.sort(np.concatenate(sel_b))


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def cluster_bootstrap_means(
    values: np.ndarray, game_id: np.ndarray, wdl: np.ndarray, *,
    stratum_weights: dict[int, float], n_boot: int, rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap the stratum-weighted mean, resampling GAMES not rows.

    Set B holds whole games (~18-22 rows each, and ``wdl_target`` is constant
    within a game), so its rows are not independent draws; an iid row bootstrap
    would divide by an n that does not exist and report a CI several times too
    tight. Resampling games inside each W/D/L stratum, with the stratum weights
    pinned to the matched mix, is the standard cluster bootstrap for that
    design. ``stratum_weights`` is shared by both sets, which is what makes the
    two bootstraps comparable.
    """
    out = np.empty((n_boot,), dtype=np.float64)
    per_stratum: list[tuple[float, list[np.ndarray]]] = []
    for cls, weight in sorted(stratum_weights.items()):
        rows = np.flatnonzero(wdl == cls)
        if rows.size == 0:
            continue
        groups = [rows[game_id[rows] == g] for g in np.unique(game_id[rows])]
        per_stratum.append((weight, groups))
    for b in range(n_boot):
        acc = 0.0
        wsum = 0.0
        for weight, groups in per_stratum:
            draw = rng.integers(0, len(groups), size=len(groups))
            pooled = np.concatenate([groups[int(j)] for j in draw])
            acc += weight * float(values[pooled].mean())
            wsum += weight
        out[b] = acc / max(1e-12, wsum)
    return out


def class_aligned_order(wdl_a: np.ndarray, wdl_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orders making position *i* of both sets the same W/D/L class.

    Rows are scored in shard order, so position *i* of one set is otherwise an
    arbitrary class in the other and a row-level "paired" delta would pair a
    win against a draw — inflating the variance the matched mix was supposed to
    remove. The per-class counts are equal after ``match_wdl_mix``, so a stable
    sort by class lines the two up exactly.
    """
    ord_a = np.argsort(wdl_a, kind="stable")
    ord_b = np.argsort(wdl_b, kind="stable")
    if not np.array_equal(wdl_a[ord_a], wdl_b[ord_b]):
        raise AssertionError("class-aligned pairing failed: W/D/L counts differ")
    return ord_a, ord_b


def stratified_mean(values: np.ndarray, wdl: np.ndarray, weights: dict[int, float]) -> float:
    acc = 0.0
    wsum = 0.0
    for cls, weight in weights.items():
        rows = np.flatnonzero(wdl == cls)
        if rows.size == 0:
            continue
        acc += weight * float(values[rows].mean())
        wsum += weight
    return acc / max(1e-12, wsum)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


@dataclass
class RowSource:
    """Where a set's rows live: shard-backed, or already dense in memory."""

    label: str
    index: RowIndex
    shards: list[ShardInfo]
    arrays: dict[str, np.ndarray] | None = None

    def gather(self, sel: np.ndarray) -> dict[str, np.ndarray]:
        if self.arrays is not None:
            return _take_rows(self.arrays, self.index.row[sel])
        parts: list[dict[str, np.ndarray]] = []
        for pos in np.unique(self.index.shard_pos[sel]):
            rows = self.index.row[sel][self.index.shard_pos[sel] == pos]
            arrs, _meta = load_shard_arrays(self.shards[int(pos)].path, lazy=True)
            parts.append(_take_rows(arrs, rows))
        return _concat_rows(parts)


def _take_rows(arrs: dict[str, Any], rows: np.ndarray) -> dict[str, np.ndarray]:
    """Row-slice a (possibly lazy zarr) array dict, carrying scalar markers."""
    order = np.argsort(rows, kind="stable")
    sorted_rows = np.asarray(rows)[order]
    out: dict[str, np.ndarray] = {}
    for key, value in arrs.items():
        if key == INPUT_HISTORY_ENCODING_ARRAY_KEY:
            marker = np.asarray(value)
            out[key] = marker if marker.ndim == 0 else np.asarray(marker[sorted_rows])
            continue
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) == 0 or int(shape[0]) <= int(sorted_rows.max(initial=0)):
            continue
        out[key] = np.asarray(value[sorted_rows])
    return out


def _concat_rows(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if len(parts) == 1:
        return parts[0]
    keys = set.intersection(*(set(p) for p in parts))
    out: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        if key == INPUT_HISTORY_ENCODING_ARRAY_KEY:
            out[key] = parts[0][key]
            continue
        out[key] = np.concatenate([p[key] for p in parts], axis=0)
    return out


def score_rows(
    model: torch.nn.Module, source: RowSource, sel: np.ndarray, *,
    loss_kwargs: dict[str, Any], device: str, batch_size: int, history_encoding: str,
) -> dict[str, np.ndarray]:
    """Per-row ``wdl_ce`` and ``total`` for the selected rows, in order.

    Per-row values come from ``compute_loss`` itself on one-row slices of the
    batched forward pass, so there is no second copy of the blend arithmetic to
    drift out of sync with ``train/losses.py``. Every chunk then re-runs
    ``compute_loss`` over the whole chunk and checks that the mask-weighted
    mean of the per-row values reproduces it — a per-chunk verification of the
    decomposition against the authority, on the real data being measured.
    """
    # `gather` reads rows shard-by-shard in ascending order, so the scored
    # sequence equals `sel` only while `sel` is sorted. Everything downstream
    # (the W/D/L vectors, the game ids the cluster bootstrap resamples, the
    # per-row dumps) is indexed by `sel`, so a silent reordering here would
    # misalign every one of them.
    if sel.size and not bool(np.all(np.diff(sel) > 0)):
        raise ValueError("score_rows requires a strictly increasing row selection")
    wdl_rows: list[float] = []
    total_rows: list[float] = []
    max_dev = 0.0
    for start in range(0, sel.shape[0], batch_size):
        chunk = sel[start:start + batch_size]
        arrs = select_input_history_arrays(
            source.gather(chunk), input_history_encoding=history_encoding,
        )
        batch = collate_arrays(arrs, device=device)
        with torch.no_grad():
            rel = batch.get("relations")
            out = model(batch["x"], relations=rel) if rel is not None else model(batch["x"])
            chunk_losses = compute_loss(out, batch, **loss_kwargs)
            n = int(batch["x"].shape[0])
            for i in range(n):
                row_out = {k: v[i:i + 1] for k, v in out.items() if isinstance(v, torch.Tensor)}
                row_batch = {k: v[i:i + 1] for k, v in batch.items()}
                row_losses = compute_loss(row_out, row_batch, **loss_kwargs)
                wdl_rows.append(float(row_losses["wdl_ce"]))
                total_rows.append(float(row_losses["total"]))
        net = batch.get("is_network_turn")
        weights = np.ones((n,), dtype=np.float64) if net is None else net.float().cpu().numpy()
        recon = float(np.dot(np.asarray(wdl_rows[-n:]), weights) / max(1.0, weights.sum()))
        max_dev = max(max_dev, abs(recon - float(chunk_losses["wdl_ce"])))
    return {
        "wdl_ce": np.asarray(wdl_rows, dtype=np.float64),
        "total": np.asarray(total_rows, dtype=np.float64),
        "max_reconstruction_dev": np.asarray([max_dev], dtype=np.float64),
    }


# --------------------------------------------------------------------------
# config resolution
# --------------------------------------------------------------------------


def resolve_loss_kwargs(latest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Build ``compute_loss`` kwargs, and say where each one came from.

    Enumerates ``compute_loss``'s OWN parameters rather than the fields of some
    dataclass that is assumed to cover them (method rule 14) and reports any it
    could not source, so a knob that silently falls back to a code default is
    visible instead of absent. Per-iteration knobs — ``sf_wdl_frac`` above all,
    which is resolved from ``wdl_regret`` each iteration — are taken from the
    realized ``result.json`` row, not from the launch config, which is an input
    to that resolution rather than its answer (method rule 12).
    """
    cfg: dict[str, Any] = dict(latest.get("config") or {})
    params = [
        name for name in inspect.signature(compute_loss).parameters
        if name not in ("outputs", "batch")
    ]
    provenance: dict[str, str] = {}
    for name in params:
        if name in latest:
            cfg[name] = latest[name]
            provenance[name] = "realized result.json row"
    kw = trainer_kwargs_from_config(cfg)
    loss_kwargs: dict[str, Any] = {}
    for name in params:
        if name == "sf_sparse_params":
            sparse = bool(cfg.get("sf_policy_sparse_ce", False))
            loss_kwargs[name] = kw["sf_target_params"] if sparse else None
            provenance.setdefault(name, f"sf_policy_sparse_ce={sparse}")
        elif name in kw:
            loss_kwargs[name] = kw[name]
            provenance.setdefault(name, "launch config" if name in cfg else "code default")
        else:
            provenance[name] = "UNSOURCED (compute_loss default)"
    return loss_kwargs, provenance


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _wdl_mix(wdl: np.ndarray) -> str:
    counts = np.bincount(np.clip(wdl, 0, 2), minlength=3)
    total = max(1, int(counts.sum()))
    return "/".join(f"{c / total:.3f}" for c in counts) + f" (n={total})"


def _age_span(shards: list[ShardInfo], now: float) -> str:
    if not shards:
        return "-"
    ages = [(now - s.mtime) / 3600.0 for s in shards]
    return f"{min(ages):.1f}-{max(ages):.1f}h"


def _describe(label: str, index: RowIndex, sel: np.ndarray) -> dict[str, Any]:
    sub = index.take(sel)
    games = np.unique(sub.game_id[sub.game_id >= 0])
    return {
        "label": label,
        "rows": int(sel.shape[0]),
        "games": int(games.size),
        "rows_per_game": float(sel.shape[0] / max(1, games.size)),
        "wdl_mix": _wdl_mix(sub.wdl),
        "is_selfplay": float((sub.is_selfplay > 0).mean()) if sel.size else 0.0,
        "ply_median": float(np.median(sub.ply[sub.ply >= 0])) if (sub.ply >= 0).any() else -1.0,
    }


def _dump_jsonl(path: Path, values: np.ndarray, wdl: np.ndarray, keys: list[str]) -> None:
    """Per-row dump joinable by scripts/paired_compare.py."""
    with path.open("w", encoding="utf-8") as handle:
        for key, value, cls in zip(keys, values.tolist(), wdl.tolist(), strict=True):
            handle.write(json.dumps({
                "key": key, "value": float(value), "phase": WDL_NAMES[int(cls)],
            }) + "\n")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="checkpoint dir or trainer.pt. Prefer one a few iterations "
                         "old: set B is drawn from shards NEWER than it.")
    ap.add_argument("--run-dir", type=Path, default=Path("runs/pbt2_small"))
    ap.add_argument("--trial-dir", type=Path, default=None)
    ap.add_argument("--replay-dir", type=Path, default=None)
    ap.add_argument("--set-a", choices=("auto", "sidecar", "reconstruct"), default="auto",
                    help="auto: the checkpoint's holdout.npz if present, else "
                         "reconstruct (which reports an UPPER bound — those rows "
                         "were themselves trained on)")
    ap.add_argument("--holdout-fraction", type=float, default=0.02,
                    help="must match the live holdout_fraction (configs/pbt2_small.yaml)")
    ap.add_argument("--regime", choices=("local", "salvage", "all"), default="local",
                    help="G8: the window mixes fresh local shards with 15-18-day-old "
                         "salvage symlinks that differ in composition; default keeps "
                         "both sets inside one regime")
    ap.add_argument("--shard-slice", default=":", help="index slice over the sorted shard list")
    ap.add_argument("--boundary", default="checkpoint",
                    help="'checkpoint' (trainer.pt mtime) or an ISO timestamp: shards "
                         "older than it are the training window, newer ones are unseen")
    ap.add_argument("--target-rows", type=int, default=2000,
                    help="rows to aim for per set before matching (holdout_capacity)")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="hard cap on scored rows per set; 0 = no cap. For smoke tests.")
    ap.add_argument("--dry-run", action="store_true",
                    help="carve and report composition only — no model load, no GPU, "
                         "no position tensors decoded")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--gpu-mem-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--dump-dir", type=Path, default=None,
                    help="write set_a.jsonl / set_b.jsonl for scripts/paired_compare.py")
    ap.add_argument("--json", dest="json_out", type=Path, default=None)
    return ap


def _resolve_boundary(spec: str, checkpoint: Path) -> float:
    if spec == "checkpoint":
        trainer_pt = checkpoint / "trainer.pt" if checkpoint.is_dir() else checkpoint
        return trainer_pt.stat().st_mtime
    return datetime.fromisoformat(spec).timestamp()


def _sidecar_path(checkpoint: Path) -> Path:
    base = checkpoint if checkpoint.is_dir() else checkpoint.parent
    return base / SIDECAR_HOLDOUT_ROWS


def sidecar_row_index(arrs: dict[str, Any]) -> RowIndex:
    """Identity columns for rows restored from a ``holdout.npz`` sidecar."""
    n = int(np.asarray(arrs["x"]).shape[0])
    gid = np.asarray(arrs.get("game_id", np.full((n,), -1))).astype(np.int64)
    if "has_game_id" in arrs:
        gid = np.where(np.asarray(arrs["has_game_id"]) > 0, gid, -1)
    return RowIndex(
        shard_pos=np.full((n,), -1, dtype=np.int64), row=np.arange(n, dtype=np.int64),
        game_id=gid, wdl=np.asarray(arrs["wdl_target"]).astype(np.int64),
        ply=np.asarray(arrs.get("ply_index", np.full((n,), -1))).astype(np.int64),
        is_selfplay=np.asarray(arrs.get("is_selfplay", np.zeros((n,)))).astype(np.int64),
    )


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    trial_dir = args.trial_dir or latest_trial_dir(args.run_dir, required=True)
    replay_dir = args.replay_dir or (args.run_dir / "replay" / trial_dir.name / "replay_shards")
    boundary = _resolve_boundary(args.boundary, args.checkpoint)
    now = max(boundary, max((p.stat().st_mtime for p in iter_shard_paths(replay_dir)), default=boundary))

    shards = scan_shards(replay_dir, regime=args.regime, shard_slice=args.shard_slice)
    window = [s for s in shards if s.mtime <= boundary]
    unseen = [s for s in shards if s.mtime > boundary]

    print("=== probe_holdout_split_leak — realized inputs ===")
    print(f"  checkpoint     {args.checkpoint}")
    print(f"  trial          {trial_dir.name}")
    print(f"  replay dir     {replay_dir}")
    print(f"  boundary       {datetime.fromtimestamp(boundary).isoformat()}  ({args.boundary})")
    print(f"  regime filter  {args.regime}  "
          f"(kept {len(shards)}/{len(iter_shard_paths(replay_dir))} shards)")
    print(f"  window shards  {len(window)}  idx "
          f"{window[0].index if window else '-'}..{window[-1].index if window else '-'}  "
          f"rows {sum(s.n_rows for s in window)}  age {_age_span(window, now)}")
    print(f"  unseen shards  {len(unseen)}  idx "
          f"{unseen[0].index if unseen else '-'}..{unseen[-1].index if unseen else '-'}  "
          f"rows {sum(s.n_rows for s in unseen)}  age {_age_span(unseen, now)}")

    if not window:
        raise SystemExit("no shards older than the boundary — nothing was trained on")
    if not unseen:
        raise SystemExit(
            "no shards newer than the boundary: this checkpoint's training window is "
            "the whole shard range, so no game-disjoint rows exist. Pass an OLDER "
            "--checkpoint (set B is drawn from shards written after it).",
        )

    window_index = read_row_index(window)
    unseen_index = read_row_index(unseen)
    train_games = np.unique(window_index.game_id[window_index.game_id >= 0])

    holdout_mask = per_row_holdout_mask(
        [s.n_rows for s in window], holdout_fraction=args.holdout_fraction, seed=args.seed,
    )
    stats = sibling_stats(window_index.game_id, holdout_mask)
    print("\n=== G13 mechanism, recomputed on these rows ===")
    print(f"  recorded plies per game        {stats['plies_per_game']:.1f}")
    print(f"  E[same-game rows in training]  {stats['expected_siblings']:.1f}")
    print(f"  P(no same-game sibling)        {stats['p_no_sibling']:.2e}")
    straddle = int(np.isin(unseen_index.game_id, train_games).sum())
    print(f"  rows in unseen shards whose game IS in the window (boundary straddle): {straddle}")

    # --- set A: the leaky per-row holdout -------------------------------
    sidecar = _sidecar_path(args.checkpoint)
    use_sidecar = args.set_a == "sidecar" or (args.set_a == "auto" and sidecar.exists())
    if use_sidecar and not sidecar.exists():
        raise SystemExit(f"--set-a sidecar but no {sidecar} (PR #261 writes it; a "
                         f"checkpoint written before that has none)")
    if use_sidecar:
        arrs_a, _meta_a = load_shard_arrays(sidecar)
        source_a = RowSource("A_leaky", sidecar_row_index(arrs_a), window, arrays=dict(arrs_a))
        a_kind = f"sidecar {sidecar} — the rows test_wdl_loss is computed on"
        a_bound = "exact: not trained on; same-game siblings were"
    else:
        pick = np.flatnonzero(holdout_mask)
        if args.target_rows > 0 and pick.size > args.target_rows:
            pick = np.sort(rng.choice(pick, size=args.target_rows, replace=False))
        source_a = RowSource("A_leaky", window_index.take(pick), window)
        a_kind = (f"reconstructed per-row {args.holdout_fraction:.0%} draw over the window "
                  f"(no {SIDECAR_HOLDOUT_ROWS} on this checkpoint)")
        a_bound = "UPPER BOUND: these rows were themselves in training"

    in_train = np.isin(source_a.index.game_id, train_games)
    print("\n=== set A (leaky, per-row holdout) ===")
    print(f"  source         {a_kind}")
    print(f"  interpretation {a_bound}")
    print(f"  rows whose game IS in the training window: {int(in_train.sum())}/{in_train.size} "
          f"({in_train.mean():.4f}) — the leak condition")

    # --- set B: game-disjoint -------------------------------------------
    target_b = min(len(source_a.index), args.target_rows) if args.target_rows > 0 else len(source_a.index)
    sel_b_all = carve_game_disjoint(
        unseen_index, train_games=train_games, target_rows=target_b, rng=rng,
    )
    if sel_b_all.size == 0:
        raise SystemExit("no game-disjoint rows in the unseen shards — every game there "
                         "also has rows in the training window")
    index_b = unseen_index.take(sel_b_all)
    assert_game_disjoint(index_b.game_id, train_games)
    source_b = RowSource("B_disjoint", index_b, unseen)
    print("\n=== set B (game-disjoint) ===")
    print(f"  drawn from {len(unseen)} unseen shards; every game_id verified absent "
          f"from the {train_games.size} training games")

    # --- match count + WDL mix ------------------------------------------
    keep_a, keep_b = match_wdl_mix(source_a.index.wdl, source_b.index.wdl, rng=rng)
    if args.max_rows > 0:
        # Cap first, then re-match: truncating a matched pair of sets breaks the
        # mix, so the second match restores it inside the cap.
        head_a, head_b = keep_a[:args.max_rows], keep_b[:args.max_rows]
        sub_a, sub_b = match_wdl_mix(
            source_a.index.wdl[head_a], source_b.index.wdl[head_b], rng=rng,
        )
        keep_a, keep_b = head_a[sub_a], head_b[sub_b]
    desc_a = _describe("A_leaky", source_a.index, keep_a)
    desc_b = _describe("B_disjoint", source_b.index, keep_b)
    print("\n=== matched sets (subsampled to identical count + W/D/L) ===")
    for desc, before in ((desc_a, len(source_a.index)), (desc_b, len(source_b.index))):
        print(f"  {desc['label']:12s} rows {desc['rows']} (of {before}) "
              f"games {desc['games']} rows/game {desc['rows_per_game']:.1f} "
              f"W/D/L {desc['wdl_mix']} selfplay {desc['is_selfplay']:.2f} "
              f"ply_med {desc['ply_median']:.0f}")
    if desc_a["wdl_mix"] != desc_b["wdl_mix"]:
        raise AssertionError(f"WDL mix mismatch after matching: {desc_a} vs {desc_b}")

    payload: dict[str, Any] = {
        "checkpoint": str(args.checkpoint), "replay_dir": str(replay_dir),
        "boundary": boundary, "regime": args.regime,
        "window_shards": [window[0].index, window[-1].index], "window_rows": sum(s.n_rows for s in window),
        "unseen_shards": [unseen[0].index, unseen[-1].index], "unseen_rows": sum(s.n_rows for s in unseen),
        "g13": stats, "set_a": desc_a, "set_b": desc_b,
        "set_a_source": a_kind, "set_a_interpretation": a_bound,
        "set_a_rows_before_match": len(source_a.index),
        "set_b_rows_before_match": len(source_b.index),
        "boundary_straddle_rows": straddle,
    }

    if args.dry_run:
        print("\n[dry-run] composition only — no model was loaded and no position "
              "tensor was decoded.")
        _emit_json(args.json_out, payload)
        return

    # --- score -----------------------------------------------------------
    if args.gpu_mem_fraction is not None and str(args.device).startswith("cuda"):
        dev_idx = (int(args.device.split(":", 1)[1]) if ":" in args.device
                   else torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_fraction), device=dev_idx)

    latest = latest_result(trial_dir)
    loss_kwargs, provenance = resolve_loss_kwargs(latest)
    print("\n=== resolved compute_loss kwargs ===")
    for name in sorted(loss_kwargs):
        print(f"  {name:28s} {loss_kwargs[name]!r:>10}   [{provenance[name]}]")
    for name, src in sorted(provenance.items()):
        if src.startswith("UNSOURCED"):
            print(f"  {name:28s} {'-':>10}   [{src}]")

    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    history_encoding = str(getattr(model, "input_history_encoding", "legacy"))
    scored_a = score_rows(model, source_a, keep_a, loss_kwargs=loss_kwargs,
                          device=args.device, batch_size=args.batch_size,
                          history_encoding=history_encoding)
    scored_b = score_rows(model, source_b, keep_b, loss_kwargs=loss_kwargs,
                          device=args.device, batch_size=args.batch_size,
                          history_encoding=history_encoding)

    wdl_a = source_a.index.wdl[keep_a]
    wdl_b = source_b.index.wdl[keep_b]
    counts = np.bincount(np.clip(wdl_a, 0, 2), minlength=3)
    weights = {c: float(counts[c]) for c in range(3) if counts[c] > 0}
    ord_a, ord_b = class_aligned_order(wdl_a, wdl_b)

    print("\n=== losses ===")
    print(f"  per-row/chunk reconstruction max |dev|: A "
          f"{scored_a['max_reconstruction_dev'][0]:.2e}  B "
          f"{scored_b['max_reconstruction_dev'][0]:.2e}   (per-row decomposition "
          f"vs compute_loss over the whole chunk)")
    results: dict[str, dict[str, float]] = {}
    for metric in ("wdl_ce", "total"):
        va, vb = scored_a[metric], scored_b[metric]
        mean_a = stratified_mean(va, wdl_a, weights)
        mean_b = stratified_mean(vb, wdl_b, weights)
        gap = mean_a - mean_b
        boot_a = cluster_bootstrap_means(
            va, source_a.index.game_id[keep_a], wdl_a,
            stratum_weights=weights, n_boot=args.n_boot,
            rng=np.random.default_rng(args.seed + 1),
        )
        boot_b = cluster_bootstrap_means(
            vb, source_b.index.game_id[keep_b], wdl_b,
            stratum_weights=weights, n_boot=args.n_boot,
            rng=np.random.default_rng(args.seed + 2),
        )
        lo, hi = np.percentile(boot_a - boot_b, [2.5, 97.5])
        ilo, ihi = paired_bootstrap_ci(va[ord_a] - vb[ord_b], n_boot=args.n_boot, seed=args.seed)
        results[metric] = {
            "mean_a": mean_a, "mean_b": mean_b, "gap": gap,
            "ci_lo": float(lo), "ci_hi": float(hi),
            "iid_ci_lo": float(ilo), "iid_ci_hi": float(ihi),
        }
        print(f"  {metric:8s} A {mean_a:.4f}  B {mean_b:.4f}  "
              f"gap {gap:+.4f} nats  [cluster 95% CI {lo:+.4f} .. {hi:+.4f}]  "
              f"(iid-row CI {ilo:+.4f} .. {ihi:+.4f}, design effect "
              f"{(hi - lo) / max(1e-9, ihi - ilo):.1f}x)")

    if args.dump_dir is not None:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        paired_wdl = wdl_a[ord_a]
        keys = [f"{WDL_NAMES[int(c)]}:{i}" for i, c in enumerate(paired_wdl)]
        _dump_jsonl(args.dump_dir / "set_a.jsonl",
                    scored_a["wdl_ce"][ord_a], paired_wdl, keys)
        _dump_jsonl(args.dump_dir / "set_b.jsonl",
                    scored_b["wdl_ce"][ord_b], paired_wdl, keys)
        print(f"\n  per-row dumps -> {args.dump_dir}; cross-check the iid CI with:\n"
              f"    PYTHONPATH=. python3 scripts/paired_compare.py "
              f"{args.dump_dir}/set_a.jsonl {args.dump_dir}/set_b.jsonl "
              f"--join-key key --field value")

    gap = results["wdl_ce"]["gap"]
    lo, hi = results["wdl_ce"]["ci_lo"], results["wdl_ce"]["ci_hi"]
    verdict = ("MEASURED-IMMATERIAL: close G13, no ledger entry"
               if abs(gap) < LEAK_THRESHOLD_NATS
               else "MATERIAL: open a ledger entry for a per-game split, with a revert point")
    print(f"\n=== verdict (pre-committed at |gap| < {LEAK_THRESHOLD_NATS} nats on wdl_ce) ===")
    print(f"  gap {gap:+.4f} nats  =>  {verdict}")
    if lo < LEAK_THRESHOLD_NATS < hi or lo < -LEAK_THRESHOLD_NATS < hi:
        print(f"  UNDERPOWERED: the CI [{lo:+.4f}, {hi:+.4f}] straddles the "
              f"{LEAK_THRESHOLD_NATS} threshold — raise --target-rows before deciding.")
    payload["results"] = results
    payload["verdict"] = verdict
    _emit_json(args.json_out, payload)


def _emit_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\n  json -> {path}")


if __name__ == "__main__":
    main()
