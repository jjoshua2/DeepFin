"""One training phase of the simple RL loop.

Default: a shuffled exact-once presentation schedule. Each phase is one pass
WITHOUT replacement over the rows scheduled for it (the new generation's, plus
the previous generation's under ``presentations=2``), served by
``ScheduledSampler`` in precomputed batches with a short final batch, so every
trainable row is presented exactly ``presentations`` times in its lifetime; a
per-row ledger derived from completed phases enforces it. The explicit
alternative ``sample_with_replacement`` draws uniformly WITH replacement from
the last ``window`` generations for ``updates`` (or ``passes`` x rows / batch)
updates. The repo's ``Trainer`` does the step: its
optimizer, LR warmup, AMP, mirror augmentation and ``compute_loss`` are used
unchanged, configured with the policy loss on the stored target plus (when
``w_wdl`` > 0) the WDL loss on ``outcome_weight * game outcome +
search_weight * stored search value`` (normalised to a mix, which is what the
loss's ``search_wdl_frac`` is). Every auxiliary head weight and every SF
target is 0.

Optional, each off by default: a KL anchor to a frozen reference net on the
policy and/or WDL heads (attached through the Trainer's single penalty hook,
the one ``configure_frozen_policy_anchor`` uses), and a fixed fraction of
every batch drawn from an external supervised corpus whose rows keep their
own labels.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.simple_rl.collect import concat_arrays, load_generation_rows
from chess_anti_engine.simple_rl.common import (
    input_ref, resolve_collect_dir, seed_for, sha256_file, write_json_atomic, write_manifest,
)

# Everything the configured loss reads, plus the flags that gate it. Other
# shard fields (SF labels, root statistics, ...) never reach the batch.
TRAIN_KEYS: tuple[str, ...] = (
    "x", "policy_target", "wdl_target", "priority", "has_policy",
    "legal_mask", "has_legal_mask", "search_wdl", "has_search_wdl",
    "is_network_turn", "has_is_network_turn", "is_selfplay", "has_is_selfplay",
)
REQUIRED_KEYS: tuple[str, ...] = ("x", "policy_target", "wdl_target", "legal_mask", "has_legal_mask")
# Root statistics read ONLY to build targets at load time (prepare_targets);
# dropped before a row reaches the sampler.
TARGET_SOURCE_KEYS: tuple[str, ...] = (
    "root_survivor_q", "has_root_survivor_q", "root_net_wdl", "has_root_net_wdl",
    "root_log_prior", "has_root_log_prior", "root_child_visits",
)
VALUE_SOURCES = ("survivor_q", "stored_search_wdl")
DRAW_SOURCES = ("net", "zero")
PRIOR_MIX_SOURCES = ("reference", "actor")
INIT_OPTIMIZER_STATES = ("fresh", "checkpoint")
# Canonical Trainer matrix_optimizer_scope names (train/trainer.py
# _matrix_optimizer_filter; the aliases it also accepts are not offered here).
MATRIX_OPTIMIZER_SCOPES = ("default", "block_all", "mlp_only", "mlp_out", "mlp_out_v", "mlp_attn_all")
# Optimizers whose construction reads matrix_optimizer_scope (the Aurora/Muon
# matrix group, SOAP's split); weight_decay_mode "soda" reads it too.
SCOPED_OPTIMIZERS = ("aurora", "muon", "soap")
MODEL_FILE = "model.pt"
# Where unset optimizer knobs come from: "trainer" = the Trainer's constructor
# defaults (the driver's historical behaviour); "init-checkpoint" = the
# run's init checkpoint's saved param groups (and zclip on/off), resolved once
# by resolve_optimizer_hparams and recorded.
OPTIMIZER_HPARAM_SOURCES = ("trainer", "init-checkpoint")
GRAD_CLIP_MODES = ("none", "zclip")
# "flat": the peak lr for the whole phase (historical: a cosine whose period is
# 1e9 updates). "cosine-phase": one cosine from the peak lr (after warmup) down
# to lr_floor over the phase's planned optimizer updates, restarted each phase.
LR_SCHEDULES = ("flat", "cosine-phase")
# The Aurora knobs the driver passes to the Trainer when set. Each is read by
# the Aurora optimizer only (matrix_lr_multiplier / the weight decays by the
# Aurora and Muon splits); validate() refuses one that would be ignored.
AURORA_KNOBS = ("aurora_pp_iterations", "aurora_pp_beta", "aurora_polar_steps", "aurora_polar_method",
                "aurora_polar_dtype")
SPLIT_KNOBS = ("matrix_lr_multiplier", "matrix_weight_decay", "aux_weight_decay")
_CHECKPOINT_CLIP_FIELD = "grad_clip_mode"


@dataclass(frozen=True)
class TrainSettings:
    # Presentation schedule (default): every trainable row is presented
    # exactly `presentations` times in its lifetime, by shuffled passes
    # WITHOUT replacement. 1 = once, in the phase after its generation is
    # collected; 2 (the cap) = once there and once in the next generation's
    # phase. The phase's rows are a consequence of that schedule.
    presentations: int = 1
    # Explicit non-default alternative: uniform WITH replacement over the last
    # `window` generations for `updates` (or passes x rows / batch) updates.
    sample_with_replacement: bool = False
    window: int | None = None
    batch_size: int = 256
    updates: int = 0
    passes: float | None = None
    lr: float = 3e-5
    lr_floor: float | None = None
    warmup_steps: int = 0
    warmup_start: float = 0.0
    optimizer: str = "nadamw"
    optimizer_state: str = "fresh"
    # Under optimizer_state=continue the checkpoint's optimizer hyperparameters
    # (peak lr, lr floor, per-group weight decay, ...) are restored with its
    # state; a requested value that differs is REFUSED unless this is set,
    # which re-applies the requested values onto the restored optimizer.
    reapply_optimizer_hparams: bool = False
    # The other resolution of a param-group hyperparameter difference on a
    # restore: keep the CHECKPOINT's group values (e.g. the SL base's Aurora
    # polar knobs) instead of this run's configured ones. Applies to every
    # restore this run makes (the init checkpoint at generation 1, the previous
    # phase under optimizer_state=continue) and never to a fresh optimizer.
    # Mutually exclusive with reapply_optimizer_hparams; peak lr / lr floor
    # differences are never adopted (they stay refuse-or-reapply).
    adopt_checkpoint_optimizer_hparams: bool = False
    # Generation 1 only (the phase that starts from the run's init model, where
    # optimizer_state=continue has no previous phase to continue): "checkpoint"
    # restores the init checkpoint's optimizer moments and per-parameter step
    # counters instead of a fresh optimizer. The checkpoint's scheduler is NOT
    # restored (it belongs to the run that wrote it); this phase's schedule at
    # --lr drives the rate. Same refusal / reapply rules as continue.
    init_optimizer_state: str = "fresh"
    # Which 2-D matrices the Aurora/Muon group owns (Trainer's matrix_optimizer_scope).
    # Only consulted by those optimizers; the SL base was trained with "mlp_out".
    matrix_optimizer_scope: str = "default"
    weight_decay_mode: str = "weight_decay"
    mirror_prob: float = 0.5
    w_policy: float = 1.0
    w_wdl: float = 1.0
    value_outcome_weight: float = 1.0
    value_search_weight: float = 0.0
    # What "stored search value" means when value_search_weight > 0:
    # survivor_q (default) = the halving survivor's completed Q turned into a
    # WDL (see search_value_wdl); stored_search_wdl = the shard's legacy
    # search_wdl field. Never the root-node visit average (biased low).
    value_search_source: str = "survivor_q"
    value_draw_source: str = "net"
    # Policy target = lam * stored target + (1 - lam) * prior, then
    # policy_target_temp (inside the loss). 1.0 = off.
    policy_prior_mix: float = 1.0
    prior_mix_source: str = "reference"
    prior_mix_reference: str | None = None
    policy_target_temp: float = 1.0
    kl_policy: float = 0.0
    kl_wdl: float = 0.0
    kl_reference: str | None = None
    sl_mix_dir: str | None = None
    sl_mix_frac: float = 0.0
    sl_mix_max_rows: int = 1_000_000
    holdout_frac: float = 0.05
    probe_rows: int = 2048
    device: str = "cpu"
    use_amp: bool = False
    seed: int = 0
    # Train on rows from collects whose evaluator cap could absorb leaves
    # (capped_collect_record). Refused by default; recorded when allowed.
    allow_capped_collect_rows: bool = False
    # Exact-once mode only: after each listed optimizer update count (strictly
    # increasing, each <= the phase's planned updates) the phase writes
    # train/snapshots/step_<k>/ (weights-only model.pt + metrics.json) without
    # changing the training it is part of. See write_snapshot.
    snapshot_updates: tuple[int, ...] = ()
    # Optimizer knobs (None = not set: from optimizer_hparams_source). The
    # realised values are read back off the optimizer into the manifest
    # (realised_optimizer: param_group_table, grad_clip).
    optimizer_hparams_source: str = "trainer"
    matrix_lr_multiplier: float | None = None
    matrix_weight_decay: float | None = None
    aux_weight_decay: float | None = None
    aurora_pp_iterations: int | None = None
    aurora_pp_beta: float | None = None
    aurora_polar_steps: int | None = None
    aurora_polar_method: str | None = None
    aurora_polar_dtype: str | None = None
    # Gradient clipping: None = "none" (the driver's historical setting) unless
    # resolved from the init checkpoint; "zclip" = the Trainer's adaptive
    # z-score clip (25-step EMA warmup), with an optional hard cap.
    grad_clip: str | None = None
    zclip_max_norm: float | None = None
    lr_schedule: str = "flat"
    # What resolve_optimizer_hparams filled in, and from where (recorded).
    optimizer_hparams_resolved: dict[str, Any] | None = None

    def search_wdl_frac(self) -> float:
        total = float(self.value_outcome_weight) + float(self.value_search_weight)
        if total <= 0:
            raise ValueError("value_outcome_weight + value_search_weight must be > 0")
        return float(self.value_search_weight) / total

    def validate(self) -> None:
        if self.presentations not in (1, 2):
            raise ValueError(f"presentations must be 1 or 2 (the cap is 2), got {self.presentations}")
        if not self.sample_with_replacement and (
                self.window is not None or int(self.updates) > 0 or self.passes is not None):
            raise ValueError("window / updates / passes apply only with sample_with_replacement; the default "
                             "schedule presents each row exactly `presentations` times")
        if self.value_search_source not in VALUE_SOURCES:
            raise ValueError(f"value_search_source must be one of {VALUE_SOURCES}")
        if self.value_draw_source not in DRAW_SOURCES:
            raise ValueError(f"value_draw_source must be one of {DRAW_SOURCES}")
        if self.prior_mix_source not in PRIOR_MIX_SOURCES:
            raise ValueError(f"prior_mix_source must be one of {PRIOR_MIX_SOURCES}")
        if not 0.0 <= float(self.policy_prior_mix) <= 1.0:
            raise ValueError("policy_prior_mix must be in [0, 1]")
        if self.init_optimizer_state not in INIT_OPTIMIZER_STATES:
            raise ValueError(f"init_optimizer_state must be one of {INIT_OPTIMIZER_STATES}")
        if self.reapply_optimizer_hparams and self.adopt_checkpoint_optimizer_hparams:
            raise ValueError("reapply_optimizer_hparams (use the configured group hyperparameters) and "
                             "adopt_checkpoint_optimizer_hparams (keep the checkpoint's) are mutually exclusive")
        if self.matrix_optimizer_scope not in MATRIX_OPTIMIZER_SCOPES:
            raise ValueError(f"matrix_optimizer_scope must be one of {MATRIX_OPTIMIZER_SCOPES}, "
                             f"got {self.matrix_optimizer_scope!r}")
        if (self.matrix_optimizer_scope != "default" and str(self.optimizer).lower() not in SCOPED_OPTIMIZERS
                and self.weight_decay_mode != "soda"):
            raise ValueError(f"matrix_optimizer_scope {self.matrix_optimizer_scope!r} is read only by the "
                             f"{SCOPED_OPTIMIZERS} optimizers (or weight_decay_mode soda); with optimizer "
                             f"{self.optimizer!r} it would be ignored")
        opt = str(self.optimizer).lower()
        if self.optimizer_hparams_source not in OPTIMIZER_HPARAM_SOURCES:
            raise ValueError(f"optimizer_hparams_source must be one of {OPTIMIZER_HPARAM_SOURCES}")
        if self.optimizer_hparams_source != "trainer" and opt != "aurora":
            raise ValueError("optimizer_hparams_source init-checkpoint reproduces an Aurora split; with optimizer "
                             f"{self.optimizer!r} it would be ignored")
        unread = [k for k in AURORA_KNOBS if getattr(self, k) is not None and opt != "aurora"]
        unread += [k for k in SPLIT_KNOBS if getattr(self, k) is not None and opt not in ("aurora", "muon")]
        if unread:
            raise ValueError(f"{unread} are read only by the Aurora split (matrix knobs also by Muon); with "
                             f"optimizer {self.optimizer!r} they would be ignored")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {self.lr_schedule!r}")
        if self.lr_schedule == "cosine-phase":
            if self.sample_with_replacement:
                raise ValueError("lr_schedule cosine-phase needs the exact-once schedule's planned updates")
            if self.optimizer_state == "continue":
                raise ValueError("lr_schedule cosine-phase restarts each phase's cosine from the Trainer's step, "
                                 "which optimizer_state continue carries across phases; use fresh state")
            if self.lr_floor is None or not float(self.lr_floor) < float(self.lr):
                raise ValueError("lr_schedule cosine-phase decays to lr_floor; give an lr_floor below lr (without "
                                 "one the schedule would be flat while recorded as a decay)")
        if self.grad_clip is not None and self.grad_clip not in GRAD_CLIP_MODES:
            raise ValueError(f"grad_clip must be one of {GRAD_CLIP_MODES}, got {self.grad_clip!r}")
        if self.zclip_max_norm is not None and self.realised_grad_clip() != "zclip":
            raise ValueError("zclip_max_norm is read only with grad_clip zclip; it would be ignored")
        if self.zclip_max_norm is not None and not float(self.zclip_max_norm) > 0:
            raise ValueError(f"zclip_max_norm must be > 0, got {self.zclip_max_norm}")
        snaps = [int(k) for k in self.snapshot_updates]
        if snaps:
            if self.sample_with_replacement:
                raise ValueError("snapshot_updates applies to the exact-once schedule only, not "
                                 "sample_with_replacement")
            if snaps[0] < 1 or any(b <= a for a, b in itertools.pairwise(snaps)):
                raise ValueError(f"snapshot_updates must be strictly increasing positive update counts, got {snaps}")

    def realised_grad_clip(self) -> str:
        return "none" if self.grad_clip is None else str(self.grad_clip)

    def own_rows_per_batch(self) -> int:
        """Self-play rows in a full exact-once batch (the rest is the SL share)."""
        n_sl = round(float(self.sl_mix_frac) * int(self.batch_size)) if self.sl_mix_frac > 0 else 0
        return int(self.batch_size) - n_sl

    def uses_survivor_value(self) -> bool:
        return (float(self.w_wdl) > 0 and self.search_wdl_frac() > 0
                and self.value_search_source == "survivor_q")


def search_value_wdl(q: np.ndarray, draw: np.ndarray) -> np.ndarray:
    """A WDL target from a scalar search value ``q`` (side to move, [-1, 1]).

    ``draw`` is the draw probability to keep (the raw net's root draw, or 0).
    It is clipped to ``1 - |q|`` so W and L stay non-negative, then
    ``W = (1 - d + q) / 2`` and ``L = (1 - d - q) / 2``: W - L == q exactly
    and W + D + L == 1.
    """
    q = np.clip(np.asarray(q, dtype=np.float64), -1.0, 1.0)
    d = np.minimum(np.clip(np.asarray(draw, dtype=np.float64), 0.0, 1.0), 1.0 - np.abs(q))
    return np.stack([(1.0 - d + q) / 2.0, d, (1.0 - d - q) / 2.0], axis=-1)


def prepare_targets(rows: dict[str, np.ndarray], settings: TrainSettings,
                    reference_prior: Any = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Rewrite ``search_wdl`` / ``policy_target`` per the value and prior-mix
    settings, then drop everything the loss does not read.

    Rows lacking what a transform needs keep their stored field; the counts
    are returned so the fallback share is on the record, not silent.
    ``reference_prior(x, legal_mask) -> probs`` supplies the reference prior.
    """
    out = dict(rows)
    n = int(out["x"].shape[0])
    info: dict[str, Any] = {"rows": n}
    if settings.uses_survivor_value() and n:
        has = np.asarray(out.get("has_root_survivor_q", np.zeros(n)), dtype=bool)
        q = np.asarray(out.get("root_survivor_q", np.zeros(n)), dtype=np.float64)
        has &= np.isfinite(q)
        if settings.value_draw_source == "net":
            has &= np.asarray(out.get("has_root_net_wdl", np.zeros(n)), dtype=bool)
            draw = np.asarray(out.get("root_net_wdl", np.zeros((n, 3))), dtype=np.float64)[:, 1]
        else:
            draw = np.zeros(n)
        wdl = np.asarray(out["search_wdl"], dtype=np.float32).copy() if "search_wdl" in out else np.zeros((n, 3), np.float32)
        wdl[has] = search_value_wdl(q[has], draw[has]).astype(np.float32)
        flag = np.asarray(out.get("has_search_wdl", np.zeros(n)), dtype=np.uint8).copy()
        flag[has] = 1
        out["search_wdl"], out["has_search_wdl"] = wdl, flag
        info["value_from_survivor_q"] = int(has.sum())
        info["value_fallback_stored_search_wdl"] = int((~has & (flag > 0)).sum())
        info["value_without_search_target"] = int((flag == 0).sum())
    lam = float(settings.policy_prior_mix)
    if lam < 1.0 and n:
        policy = np.asarray(out.get("has_policy", np.ones(n)), dtype=bool)
        legal = np.asarray(out["legal_mask"]) > 0
        if settings.prior_mix_source == "actor":
            ok = policy & np.asarray(out.get("has_root_log_prior", np.zeros(n)), dtype=bool)
            support = np.asarray(out["root_child_visits"])[ok] >= 0
            logp = np.where(support, np.asarray(out["root_log_prior"], dtype=np.float64)[ok], -np.inf)
            logp -= logp.max(axis=1, keepdims=True)
            prior = np.where(support, np.exp(logp), 0.0)
        else:
            if reference_prior is None:
                raise ValueError("prior_mix_source=reference needs a reference checkpoint")
            ok = policy
            prior = np.asarray(reference_prior(np.asarray(out["x"])[ok], legal[ok]), dtype=np.float64)
        prior /= np.maximum(prior.sum(axis=1, keepdims=True), 1e-30)
        target = np.asarray(out["policy_target"], dtype=np.float32).copy()
        t = target[ok].astype(np.float64)
        t /= np.maximum(t.sum(axis=1, keepdims=True), 1e-30)
        target[ok] = (lam * t + (1.0 - lam) * prior).astype(np.float32)
        out["policy_target"] = target
        info["prior_mixed_rows"] = int(ok.sum())
        info["prior_mix_skipped_policy_rows"] = int((policy & ~ok).sum())
    for k in TARGET_SOURCE_KEYS:
        out.pop(k, None)
    return out, info


def is_holdout_game(game_id: int, frac: float) -> bool:
    h = hashlib.sha256(f"holdout:{int(game_id)}".encode()).digest()
    return int.from_bytes(h[:8], "little") / float(1 << 64) < float(frac)


def _select(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: (v if str(k).startswith("_") else np.asarray(v)[mask]) for k, v in arrays.items()}


def phase_generations(generation: int, settings: TrainSettings) -> list[int]:
    """The generations whose rows this phase presents."""
    if settings.sample_with_replacement:
        k = int(settings.window) if settings.window is not None else 4
        return list(range(max(1, generation - k + 1), generation + 1))
    return list(range(max(1, generation - int(settings.presentations) + 1), generation + 1))


def capped_collects(run_dir: Path, generations: list[int]) -> dict[int, dict[str, Any]]:
    """The sealed collects among ``generations`` whose evaluator cap could absorb leaves."""
    from chess_anti_engine.simple_rl.collect import capped_collect_record

    out: dict[int, dict[str, Any]] = {}
    for g in generations:
        path = Path(run_dir) / f"gen_{g:04d}" / "collect" / "manifest.json"
        if path.is_file():
            rec = capped_collect_record(json.loads(path.read_text()))
            if rec is not None:
                out[g] = rec
    return out


def lineage_capped_collects(run_dir: Path, generation: int) -> dict[int, dict[str, Any]]:
    """Capped collects anywhere in the lineage a phase-``generation`` model descends from.

    Every collected generation 1..generation is trained on by some phase of the
    run (at least its own), so the model trained at ``generation`` inherits all
    of them, whatever this phase's window is. Read from the collect manifests,
    so it does not depend on earlier phases having recorded a taint field.
    """
    return capped_collects(run_dir, list(range(1, int(generation) + 1)))


def refuse_capped_collects(capped: dict[int, dict[str, Any]], settings: TrainSettings) -> None:
    if capped and not settings.allow_capped_collect_rows:
        raise ValueError(
            f"generation(s) {sorted(capped)} were collected at an evaluator cap below the leaves one search rep "
            f"requests ({json.dumps({str(k): v for k, v in capped.items()})}): those searches backed starved leaves "
            "with the root's Q instead of evaluating them, so their targets are corrupt. Re-collect them, or pass "
            "--allow-capped-collect-rows to train on them anyway (recorded as trained_on_capped_rows)")


def load_window(run_dir: Path, generation: int, settings: TrainSettings) -> tuple[
        dict[str, np.ndarray], dict[str, np.ndarray], list[int], np.ndarray]:
    """(train rows, held-out rows of the newest generation, gens, row ids).

    Row ids are ``(generation, row index within that generation's collect
    rows in load order)``, the key of the presentation ledger.
    """
    gens = phase_generations(generation, settings)
    train_parts: list[dict[str, np.ndarray]] = []
    ids: list[np.ndarray] = []
    holdout: dict[str, np.ndarray] | None = None
    for g in gens:
        if not (Path(run_dir) / f"gen_{g:04d}" / "collect" / "manifest.json").is_file():
            raise ValueError(f"generation {g} collect is not sealed; refusing to train on its rows")
        if resolve_collect_dir(Path(run_dir) / f"gen_{g:04d}" / "collect") != Path(run_dir) / f"gen_{g:04d}" / "collect":
            from chess_anti_engine.simple_rl.lineage import verify_stage

            verify_stage(run_dir, g, "collect")  # a reference: the source still matches what it sealed
        arrays, meta = load_generation_rows(Path(run_dir) / f"gen_{g:04d}" / "collect",
                                            keys=(*TRAIN_KEYS, *TARGET_SOURCE_KEYS))
        held = holdout_mask(meta, settings.holdout_frac)
        train_parts.append(_select(arrays, ~held))
        rows = np.flatnonzero(~held)
        ids.append(np.stack([np.full(rows.size, g, dtype=np.int64), rows.astype(np.int64)], axis=1))
        if g == generation:
            holdout = _select(arrays, held)
    assert holdout is not None
    return concat_arrays(train_parts), holdout, gens, np.concatenate(ids, axis=0)


def planned_phase_updates(run_dir: Path, generation: int, settings: TrainSettings) -> int | None:
    """The exact-once phase's update count, from the sealed collects' row
    metadata alone (no shard is loaded); None while any of them is unsealed."""
    if settings.sample_with_replacement:
        return None
    n = 0
    for g in phase_generations(generation, settings):
        if not (Path(run_dir) / f"gen_{g:04d}" / "collect" / "manifest.json").is_file():
            return None
        n += int((~holdout_mask(_generation_meta(run_dir, g), settings.holdout_frac)).sum())
    return math.ceil(n / settings.own_rows_per_batch())


def check_snapshot_plan(snapshots: tuple[int, ...] | list[int], updates: int) -> None:
    if snapshots and int(snapshots[-1]) > int(updates):
        raise ValueError(f"snapshot_updates {list(snapshots)} reach past the phase's {updates} planned updates")


def holdout_mask(meta: dict[str, np.ndarray], frac: float) -> np.ndarray:
    return np.asarray([is_holdout_game(int(gid), frac) for gid in meta["game_id"]], dtype=bool)


PRESENTED = "presented.npz"
LEDGER = "presentation_ledger.json"


def _generation_meta(run_dir: Path, g: int) -> dict[str, np.ndarray]:
    metas = []
    for m in sorted((resolve_collect_dir(Path(run_dir) / f"gen_{g:04d}" / "collect") / "shards").glob("shard_*.meta.npz")):
        with np.load(m) as z:
            metas.append(z["game_id"])
    return {"game_id": np.concatenate(metas) if metas else np.zeros(0, np.int64)}


def presentation_counts(run_dir: Path, *, before_generation: int) -> tuple[dict[int, np.ndarray], int]:
    """Per-row presentation counts from every COMPLETED train phase (one with a
    manifest) of a generation below ``before_generation``, plus the SL cursor.

    Derived from the phases' own records rather than kept as a running
    counter, so a phase killed before its manifest contributes nothing and a
    redo cannot double-present.
    """
    counts: dict[int, np.ndarray] = {}
    sl_cursor = 0
    for g in range(1, int(before_generation)):
        train_dir = Path(run_dir) / f"gen_{g:04d}" / "train"
        if not (train_dir / "manifest.json").exists():
            continue
        with np.load(train_dir / PRESENTED) as z:
            gen, row = z["gen"], z["row"]
            sl_cursor = max(sl_cursor, int(z["sl_end"]))
        for pg in np.unique(gen):
            pg = int(pg)
            if pg not in counts:
                counts[pg] = np.zeros(len(_generation_meta(run_dir, pg)["game_id"]), dtype=np.int64)
            np.add.at(counts[pg], row[gen == pg], 1)
    return counts, sl_cursor


def presentation_histograms(run_dir: Path, counts: dict[int, np.ndarray], *, holdout_frac: float,
                            through_generation: int, presentations: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for g in range(1, int(through_generation) + 1):
        n = len(_generation_meta(run_dir, g)["game_id"])
        c = counts.get(g, np.zeros(n, dtype=np.int64))
        held = holdout_mask(_generation_meta(run_dir, g), holdout_frac)
        values, freq = np.unique(c[~held], return_counts=True)
        out[str(g)] = {
            "trainable_rows": int((~held).sum()), "holdout_rows": int(held.sum()),
            "holdout_presented": int(c[held].sum()),
            "histogram": {str(int(v)): int(f) for v, f in zip(values, freq, strict=True)},
            "lifetime_complete": g <= int(through_generation) - int(presentations) + 1,
        }
    return out


class ScheduledSampler:
    """Serves a precomputed list of batches -- each row of a phase exactly
    once, in a shuffled order, WITHOUT replacement.

    The final batch is short (the remainder) rather than dropped or padded, so
    no row is skipped and none repeats. ``served`` records the row positions
    actually handed to the Trainer: the presentation record is the consumer's,
    not the plan's. Exposes ``rng`` for the Trainer's mirror draw (per batch,
    i.e. per presentation).

    ``stored_row_quota`` is the Trainer's contract for a consumed-once row
    stream (the same marker ``StoredRowOnceReplay`` sets): batches are served
    serially with no retry, a non-finite gradient RAISES instead of skipping
    the step after its rows were consumed, and the loss is scaled by
    ``rows / batch_size`` so a short final batch keeps the full batches'
    per-row coefficient of ``1 / batch_size``.
    """

    stored_row_quota = True

    def __init__(self, rows: dict[str, np.ndarray], batches: list[tuple[np.ndarray, np.ndarray]], *,
                 rng: np.random.Generator, sl_rows: dict[str, np.ndarray] | None = None) -> None:
        self.rows = rows
        self.batches = batches
        self.rng = rng
        self.sl_rows = sl_rows
        self.n = int(rows["x"].shape[0])
        self.cursor = 0
        self.served: list[np.ndarray] = []
        self.served_sl: list[np.ndarray] = []
        self.drawn_rows = 0
        self.realised_sl_rows = 0
        self.aborted = False

    def __len__(self) -> int:
        return self.n

    def abort_consumption(self) -> None:
        """Trainer fault: the phase is not retried; the stage fails and redoes whole."""
        self.aborted = True

    def sample_batch_arrays(self, batch_size: int) -> dict[str, np.ndarray]:
        del batch_size  # the plan fixes every batch, including the short last one
        if self.cursor >= len(self.batches):
            raise RuntimeError("ScheduledSampler exhausted: the Trainer asked for more batches than planned")
        own, sl_idx = self.batches[self.cursor]
        self.cursor += 1
        batch = {k: (v if str(k).startswith("_") else np.asarray(v)[own]) for k, v in self.rows.items()}
        if sl_idx.size:
            assert self.sl_rows is not None
            sl = {k: (v if str(k).startswith("_") else np.asarray(v)[sl_idx]) for k, v in self.sl_rows.items()}
            batch = concat_arrays([batch, sl])
        self.served.append(np.asarray(own))
        self.served_sl.append(np.asarray(sl_idx))
        self.drawn_rows += int(own.size + sl_idx.size)
        self.realised_sl_rows += int(sl_idx.size)
        return batch


def plan_batches(n_own: int, *, batch_size: int, sl_frac: float, sl_available: int,
                 rng: np.random.Generator, sl_start: int = 0) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Shuffled exact-once batches over ``n_own`` rows (+ an SL share).

    Each full batch holds ``round(sl_frac * batch_size)`` SL rows; the short
    final batch holds the same SL proportion of its own rows, rounded. SL rows
    are taken consecutively from ``sl_start`` in the run's fixed corpus order,
    so none is used twice in a run. Returns (batches, sl_end).
    """
    order = rng.permutation(int(n_own))
    n_sl = round(float(sl_frac) * int(batch_size)) if sl_frac > 0 else 0
    own_per = int(batch_size) - n_sl
    if own_per <= 0:
        raise ValueError("sl_frac leaves no room for self-play rows in a batch")
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    sl_pos = int(sl_start)
    for i in range(0, order.size, own_per):
        own = order[i:i + own_per]
        k = n_sl if own.size == own_per else round(own.size * n_sl / own_per)
        if sl_pos + k > int(sl_available):
            raise ValueError(f"SL corpus exhausted: need rows {sl_pos}..{sl_pos + k}, have {sl_available} "
                             "(each SL row is used at most once per run; raise --sl-mix-max-rows)")
        batches.append((own, np.arange(sl_pos, sl_pos + k, dtype=np.int64)))
        sl_pos += k
    return batches, sl_pos


def load_sl_corpus(path: Path, *, max_rows: int, seed: int, need_search: bool,
                   consumed: list[Path] | None = None) -> dict[str, np.ndarray]:
    from chess_anti_engine.replay.shard import load_shard_arrays

    shards = sorted(p for p in Path(path).rglob("*.zarr") if p.is_dir())
    if not shards:
        raise ValueError(f"--sl-mix-dir {path}: no .zarr shards found")
    order = np.random.default_rng(seed).permutation(len(shards))
    parts, rows = [], 0
    for i in order:
        arrays, _ = load_shard_arrays(shards[int(i)], lazy=False, validate=True)
        missing = [k for k in REQUIRED_KEYS if k not in arrays]
        if need_search and "search_wdl" not in arrays:
            missing.append("search_wdl")
        if missing:
            raise ValueError(f"SL corpus shard {shards[int(i)]} lacks {missing}, which the configured loss needs")
        if not np.all(np.asarray(arrays["has_legal_mask"]) == 1):
            raise ValueError(f"SL corpus shard {shards[int(i)]} has rows without a legal mask")
        parts.append({k: v for k, v in arrays.items() if k in TRAIN_KEYS or str(k).startswith("_")})
        if consumed is not None:
            consumed.append(shards[int(i)])  # provenance: the shards actually read
        rows += int(arrays["x"].shape[0])
        if rows >= max_rows:
            break
    out = concat_arrays(parts)
    if int(out["x"].shape[0]) > max_rows:
        out = {k: (v if str(k).startswith("_") else v[:max_rows]) for k, v in out.items()}
    return out


class WindowSampler:
    """Uniform-with-replacement batches over the window (+ an SL fraction).

    Exposes ``rng`` because the Trainer draws its mirror augmentation from the
    sampler's generator. With an SL corpus, exactly ``round(frac * batch)``
    rows of every batch come from it; ``realised_sl_rows`` counts them.
    """

    def __init__(self, rows: dict[str, np.ndarray], *, rng: np.random.Generator,
                 sl_rows: dict[str, np.ndarray] | None = None, sl_frac: float = 0.0) -> None:
        self.rows = rows
        self.rng = rng
        self.sl_rows = sl_rows
        self.sl_frac = float(sl_frac)
        if self.sl_frac > 0 and sl_rows is None:
            raise ValueError("sl_frac > 0 needs an SL corpus")
        if not 0.0 <= self.sl_frac < 1.0:
            raise ValueError("sl_frac must be in [0, 1)")
        self.n = int(rows["x"].shape[0])
        if self.n <= 0:
            raise ValueError("training window is empty")
        self.drawn_rows = 0
        self.realised_sl_rows = 0
        if sl_rows is not None:
            for k in ("_input_history_encoding", "_policy_encoding", "_history_rep_fix"):
                a, b = rows.get(k), sl_rows.get(k)
                if a is not None and b is not None and str(np.asarray(a)) != str(np.asarray(b)):
                    raise ValueError(f"SL corpus {k}={np.asarray(b)} differs from self-play {np.asarray(a)}")
                if (a is None) != (b is None):
                    raise ValueError(f"SL corpus and self-play disagree on whether {k} is recorded")

    def __len__(self) -> int:
        return self.n

    def sample_batch_arrays(self, batch_size: int) -> dict[str, np.ndarray]:
        n_sl = round(self.sl_frac * batch_size) if self.sl_rows is not None else 0
        own = self.rng.integers(0, self.n, size=batch_size - n_sl)
        batch = {k: (v if str(k).startswith("_") else np.asarray(v)[own]) for k, v in self.rows.items()}
        if n_sl:
            assert self.sl_rows is not None
            idx = self.rng.integers(0, int(self.sl_rows["x"].shape[0]), size=n_sl)
            sl = {k: (v if str(k).startswith("_") else np.asarray(v)[idx]) for k, v in self.sl_rows.items()}
            batch = concat_arrays([batch, sl])
        self.drawn_rows += batch_size
        self.realised_sl_rows += n_sl
        return batch


class ReferenceAnchor:
    """KL(reference || student) on the policy (legal moves) and/or WDL heads.

    Same interface as ``FrozenPolicyAnchor.penalty`` so the Trainer's penalty
    hook drives it; ``diagnostics()`` is read off THIS object after training,
    which is the proof the coefficients reached the loss.
    """

    def __init__(self, reference: Any, *, policy_coef: float, wdl_coef: float) -> None:
        self.model = reference.eval()
        self.model.requires_grad_(False)
        self.policy_coef = float(policy_coef)
        self.wdl_coef = float(wdl_coef)
        self.rows = 0
        self.forwards = 0
        self.policy_kl_sum = 0.0
        self.wdl_kl_sum = 0.0

    def penalty(self, outputs: dict[str, Any], batch: dict[str, Any]) -> Any:
        import torch

        from chess_anti_engine.train.frozen_policy_anchor import eligible_legal, forward_kl, primary_logits

        student = primary_logits(outputs)
        eligible, legal = eligible_legal(student, batch)
        if not bool(eligible.any()):
            return None
        with torch.no_grad():
            ref = self.model(batch["x"][eligible])
        self.forwards += 1
        total = student.new_zeros((), dtype=torch.float32)
        count = int(eligible.sum())
        self.rows += count
        if self.policy_coef:
            terms = forward_kl(student[eligible], primary_logits(ref), legal)
            self.policy_kl_sum += float(terms.detach().sum())
            total = total + self.policy_coef * terms.mean()
        if self.wdl_coef:
            log_p = torch.log_softmax(outputs["wdl"][eligible].float(), dim=-1)
            log_q = torch.log_softmax(ref["wdl"].detach().float(), dim=-1)
            terms = (log_q.exp() * (log_q - log_p)).sum(dim=-1)
            self.wdl_kl_sum += float(terms.detach().sum())
            total = total + self.wdl_coef * terms.mean()
        return total

    def diagnostics(self) -> dict[str, Any]:
        return {"rows": self.rows, "reference_forwards": self.forwards,
                "policy_coef": self.policy_coef, "wdl_coef": self.wdl_coef,
                "mean_policy_kl": self.policy_kl_sum / self.rows if self.rows else None,
                "mean_wdl_kl": self.wdl_kl_sum / self.rows if self.rows else None}


def attach_anchor(trainer: Any, anchor: ReferenceAnchor) -> None:
    """Install ``anchor`` on the Trainer's penalty hook, refusing a silent no-op."""
    if not hasattr(trainer, "_frozen_policy_anchor"):
        raise RuntimeError("Trainer has no _frozen_policy_anchor hook; the KL anchor would be ignored")
    if getattr(trainer, "_prefetch_batches", False) or trainer.accum_steps != 1 or hasattr(trainer.model, "_orig_mod"):
        raise ValueError("the KL anchor needs eager, unprefetched, single-microbatch training")
    trainer._frozen_policy_anchor = anchor


def schedule_period(settings: TrainSettings, phase_updates: int | None) -> int:
    """CosineAnnealingWarmRestarts T_0 for the requested schedule."""
    if settings.lr_schedule == "flat":
        return 10**9
    if phase_updates is None:
        raise ValueError("lr_schedule cosine-phase needs the phase's planned updates")
    return max(1, int(phase_updates) - int(settings.warmup_steps))


def trainer_kwargs(settings: TrainSettings, *, phase_updates: int | None = None) -> dict[str, Any]:
    from chess_anti_engine.train.losses import policy_target_temp_active

    policy_target_temp_active(settings.policy_target_temp)  # range check, raises on a typo
    floor = settings.lr if settings.lr_floor is None else settings.lr_floor
    return {
        "optimizer": settings.optimizer, "weight_decay_mode": settings.weight_decay_mode,
        "matrix_optimizer_scope": settings.matrix_optimizer_scope,
        "grad_clip_mode": settings.realised_grad_clip(), "zclip_max_norm": settings.zclip_max_norm,
        **{k: getattr(settings, k) for k in (*AURORA_KNOBS, *SPLIT_KNOBS) if getattr(settings, k) is not None},
        "feature_dropout_p": 0.0, "fdp_king_safety": 0.0, "fdp_pins": 0.0, "fdp_pawns": 0.0,
        "fdp_mobility": 0.0, "fdp_outposts": 0.0, "resid_channel_dropout": 0.0,
        "resid_channel_balance_weight": 0.0, "rebuild_sf_targets": False, "sf_policy_sparse_ce": False,
        "rebuild_categorical_target": False, "swa_start": -1, "accum_steps": 1,
        "w_policy": settings.w_policy, "w_wdl": settings.w_wdl,
        "w_soft": 0.0, "w_future": 0.0, "w_sf_own": 0.0, "w_sf_own_regret": 0.0,
        "w_sf_policy_floor": 0.0, "w_sf_shape": 0.0, "w_sf_move": 0.0, "w_sf_eval": 0.0,
        "w_categorical": 0.0, "w_volatility": 0.0, "w_sf_volatility": 0.0, "w_moves_left": 0.0,
        "policy_target_temp": settings.policy_target_temp, "soft_policy_min_tv": 0.0,
        "sf_wdl_frac": 0.0, "search_wdl_frac": settings.search_wdl_frac(),
        "sf_wdl_conf_power": 0.0, "sf_wdl_draw_scale": 1.0, "sf_wdl_temperature": 1.0,
        "sf_search_dampen_sf_low": 0.0, "sf_search_dampen_sf_high": 0.0,
        "use_adjusted_wdl_target": False, "wdl_terminal_outcome_plies": 0,
        "wdl_terminal_outcome_sf_frac": 0.0, "moves_left_max_plies": 450.0,
        "use_compile": False, "prefetch_batches": False, "prefetch_split_timing": False,
        "device": settings.device, "use_amp": settings.use_amp,
        "lr": settings.lr, "lr_eta_min": floor, "warmup_steps": settings.warmup_steps,
        "warmup_lr_start": settings.warmup_start, "lr_schedule": "cosine",
        "lr_T0": (schedule_period(settings, phase_updates)
                  if settings.lr_schedule == "flat" or phase_updates is not None else 10**9),
        "lr_T_mult": 1, "mirror_prob": settings.mirror_prob,
    }


def _plain(v: Any) -> Any:
    """A JSON-ready, comparable copy of one param-group value."""
    import torch

    if torch.is_tensor(v):
        return v.detach().cpu().tolist()
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return repr(v)


def group_hparams(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """EVERY param-group key a restore replaces, except the ones the checkpoint
    legitimately owns (params, lr, initial_lr, SODA's step: the Trainer's
    ``_CHECKPOINT_OWNED_GROUP_KEYS``). ``Optimizer.load_state_dict`` swaps
    each group dict wholesale, so anything narrower lets a key change unseen
    (the Aurora polar knobs did, 2026-11-20 review of #1426)."""
    from chess_anti_engine.train.trainer import _CHECKPOINT_OWNED_GROUP_KEYS

    return [{k: _plain(v) for k, v in sorted(g.items()) if k not in _CHECKPOINT_OWNED_GROUP_KEYS} for g in groups]


def optimizer_hparams(trainer: Any) -> dict[str, Any]:
    """The optimizer hyperparameters a restore can carry, read off the live
    trainer (scheduler + param groups), JSON-ready."""
    sched = trainer._scheduler
    return {
        "peak_lr": float(trainer._peak_lr),
        "base_lrs": [float(v) for v in getattr(sched, "base_lrs", [])],
        "lr_floor": float(getattr(sched, "eta_min", float("nan"))),
        "groups": group_hparams(trainer.opt.param_groups),
    }


def group_key_diff(configured: list[dict[str, Any]], checkpoint: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per (group, key) whose value differs or exists on one side only."""
    out: list[dict[str, Any]] = []
    for i, (want, have) in enumerate(zip(configured, checkpoint, strict=True)):
        for k in sorted(set(want) | set(have)):
            a, b = want.get(k, "<absent>"), have.get(k, "<absent>")
            if not _close(a, b):
                out.append({"group": i, "key": k, "configured": a, "checkpoint": b})
    return out


# The per-group table that says which optimizer setup actually ran. Read off
# the live param groups (never the flags); compared against a checkpoint's
# saved groups by param_group_table_of.
TABLE_KEYS = ("lr", "betas", "eps", "weight_decay", "aurora_momentum", "aurora_nesterov", "aurora_pp_iterations",
              "aurora_pp_beta", "aurora_polar_steps", "aurora_polar_method", "aurora_polar_dtype",
              "aurora_polar_safety", "aurora_uw_floor")


def _group_class(g: dict[str, Any], optimizer_class: str) -> str:
    if "use_aurora" in g:
        return "aurora" if g["use_aurora"] else "adamw"
    if "use_muon" in g:
        return "muon" if g["use_muon"] else "adamw"
    return optimizer_class


def param_group_table_of(groups: list[dict[str, Any]], optimizer_class: str = "unknown") -> list[dict[str, Any]]:
    """One row per param group: count, class and every TABLE_KEYS value present."""
    return [{"params": len(g["params"]), "class": _group_class(g, optimizer_class),
             **{k: _plain(g[k]) for k in TABLE_KEYS if k in g}} for g in groups]


def param_group_table(trainer: Any) -> list[dict[str, Any]]:
    return param_group_table_of(trainer.opt.param_groups, type(trainer.opt).__name__)


def tables_match(a: list[dict[str, Any]], b: list[dict[str, Any]], *, rel_lr: float = 1e-6) -> list[str]:
    """Differences between two param-group tables (lr compared at rel_lr)."""
    if len(a) != len(b):
        return [f"{len(a)} groups vs {len(b)}"]
    out = []
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        for k in sorted(set(x) | set(y)):
            u, v = x.get(k, "<absent>"), y.get(k, "<absent>")
            same = (math.isclose(float(u), float(v), rel_tol=rel_lr, abs_tol=0.0)
                    if k == "lr" and isinstance(u, float) and isinstance(v, float) else _close(u, v))
            if not same:
                out.append(f"group {i} {k}: {u!r} vs {v!r}")
    return out


def realised_optimizer(trainer: Any) -> dict[str, Any]:
    out = optimizer_hparams(trainer)
    out["param_group_lr"] = [float(g["lr"]) for g in trainer.opt.param_groups]
    out["optimizer_class"] = type(trainer.opt).__name__
    out["param_group_table"] = param_group_table(trainer)
    sched = trainer._scheduler
    out["lr_schedule"] = {"class": type(sched).__name__, "T_0": getattr(sched, "T_0", None),
                          "eta_min": getattr(sched, "eta_min", None),
                          "warmup_steps": int(getattr(trainer, "_warmup_steps", 0))}
    out["grad_clip"] = {"mode": str(trainer.grad_clip_mode),
                        "zclip_max_norm": getattr(trainer.zclip, "max_grad_norm", None),
                        "zclip_warmup_steps": int(getattr(trainer.zclip, "warmup_steps", 0))}
    return out


def resolve_optimizer_hparams(settings: TrainSettings, checkpoint: Path | None) -> TrainSettings:
    """Fill every unset Aurora knob (and grad_clip) from ``checkpoint``'s saved
    optimizer when ``optimizer_hparams_source`` is "init-checkpoint".

    The checkpoint must hold the Trainer's Aurora layout (four groups: matrix,
    matrix no-decay, aux decay, aux no-decay); its matrix-LR multiplier is the
    matrix group's lr over the aux decay group's. Keys the driver cannot set
    (AdamW betas / eps, Aurora momentum / Nesterov / safety / uw floor) must
    already equal what the Trainer builds, else the setup is not reproducible
    and this refuses. Explicitly set knobs win and are recorded as such.
    """
    import dataclasses

    import torch

    if settings.optimizer_hparams_source == "trainer":
        return settings
    if checkpoint is None:
        raise ValueError("optimizer_hparams_source init-checkpoint needs an init checkpoint (a random init has no "
                         "optimizer to reproduce)")
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False, mmap=True)
    groups = (payload.get("opt") or {}).get("param_groups") if isinstance(payload.get("opt"), dict) else None
    flags = [g.get("use_aurora") for g in groups or []]
    if not groups or flags[:4] != [True, False, False, False]:
        raise ValueError(f"{checkpoint}: optimizer param groups are not the Trainer's Aurora layout "
                         f"(use_aurora {flags}); nothing to reproduce from")
    fixed = {"betas": [0.9, 0.95], "eps": 1e-8, "aurora_momentum": 0.95, "aurora_nesterov": True,
             "aurora_polar_safety": 1.01, "aurora_uw_floor": 0.0}
    bad = {k: _plain(groups[0].get(k)) for k, v in fixed.items() if not _close(_plain(groups[0].get(k)), v)}
    if bad:
        raise ValueError(f"{checkpoint}: optimizer keys the driver cannot set differ from the Trainer's {fixed}: "
                         f"{bad}; the setup is not reproducible")
    mat, aux = groups[0], groups[2]
    found: dict[str, Any] = {k: _plain(mat[k]) for k in AURORA_KNOBS}
    # A keyword update, not subscript writes: tests/test_construction_only_config_keys.py reads
    # `x["<yaml key>"]` as a config read, and these are the checkpoint's group values, not the yaml's.
    found.update(matrix_lr_multiplier=float(mat["lr"]) / float(aux["lr"]),
                 matrix_weight_decay=float(mat["weight_decay"]), aux_weight_decay=float(aux["weight_decay"]))
    zc = payload.get("zclip")
    # The CHECKPOINT's clip-mode field (Trainer.save writes it), read through a constant so the
    # construction-only key scan does not take it for a yaml read of `grad_clip_mode`.
    found["grad_clip"] = str(payload.get(_CHECKPOINT_CLIP_FIELD) or ("zclip" if isinstance(zc, dict) else "none"))
    filled = {k: v for k, v in found.items() if getattr(settings, k) is None}
    record = {"source": str(checkpoint), "checkpoint_values": found, "filled": sorted(filled),
              "explicit": sorted(k for k in found if getattr(settings, k) is not None)}
    out = dataclasses.replace(settings, **filled, optimizer_hparams_resolved=record)
    out.validate()
    return out


def _close(a: Any, b: Any) -> bool:
    import math as _m

    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        return (_m.isnan(a) and _m.isnan(b)) or _m.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
    return type(a) is type(b) and a == b if isinstance(a, bool) or isinstance(b, bool) else a == b


def _hparam_diff(want: dict[str, Any], have: dict[str, Any]) -> dict[str, Any]:
    return {k: {"checkpoint": have.get(k), "requested": want[k]} for k in want if not _close(want[k], have.get(k))}


def _resolve_group_diff(trainer: Any, configured: list[dict[str, Any]], settings: TrainSettings,
                        what: str) -> dict[str, Any]:
    """Refuse, reapply (configured) or adopt (checkpoint) after a restore; verify the outcome."""
    diff = group_key_diff(configured, group_hparams(trainer.opt.param_groups))
    resolution = None
    if diff:
        if settings.reapply_optimizer_hparams:
            trainer._reapply_configured_param_group_hparams()
            resolution = "configured"
            left = group_key_diff(configured, group_hparams(trainer.opt.param_groups))
            if left:
                raise RuntimeError(f"re-applying the configured group hyperparameters left differences: {left}")
        elif settings.adopt_checkpoint_optimizer_hparams:
            resolution = "checkpoint"
        else:
            lines = "; ".join(f"group {d['group']} {d['key']}: configured {d['configured']!r} vs checkpoint "
                              f"{d['checkpoint']!r}" for d in diff)
            raise ValueError(f"{what}: the checkpoint's optimizer param-group hyperparameters differ from this "
                             f"run's configured ones ({lines}); restoring would silently replace them. Pass "
                             "--reapply-optimizer-hparams to use the configured values or "
                             "--adopt-checkpoint-optimizer-hparams to keep the checkpoint's")
    return {"group_differed": diff, "group_resolution": resolution,
            "group_hparams_realised": group_hparams(trainer.opt.param_groups)}


def continue_optimizer(trainer: Any, checkpoint: Path, settings: TrainSettings) -> dict[str, Any]:
    """``trainer.load(exact_resume=True)`` restores the checkpoint's optimizer
    hyperparameters along with its moments, so the requested ``--lr`` /
    ``--lr-floor`` / weight decay would be silently ignored. Refuse any
    difference, or with ``reapply_optimizer_hparams`` re-apply the requested
    values (schedule phase kept). Returns what was compared and done."""
    requested = optimizer_hparams(trainer)
    trainer.load(checkpoint, exact_resume=True)
    restored = optimizer_hparams(trainer)
    want = {k: requested[k] for k in ("peak_lr", "lr_floor")}
    diff = _hparam_diff(want, restored)
    if diff and not settings.reapply_optimizer_hparams:
        raise ValueError(f"--optimizer-state continue: the checkpoint's optimizer hyperparameters differ from the "
                         f"requested ones {json.dumps(diff, default=str)}; restoring would silently ignore them. "
                         "Pass --reapply-optimizer-hparams to apply the requested values, or request the "
                         "checkpoint's")
    if diff:
        trainer.set_peak_lr(float(settings.lr))  # base_lrs, active lr and _peak_lr, phase preserved
        if hasattr(trainer._scheduler, "eta_min"):
            trainer._scheduler.eta_min = float(requested["lr_floor"])
        left = _hparam_diff(want, optimizer_hparams(trainer))
        if left:
            raise RuntimeError(f"re-applying optimizer hyperparameters left differences: {left}")
    groups = _resolve_group_diff(trainer, requested["groups"], settings, "--optimizer-state continue")
    return {"differed": diff, "reapplied": bool(diff), **groups}


UNRECOGNISED_OPTIMIZER = "unrecognised"


def _optimizer_family(param_groups: list[dict[str, Any]]) -> str | None:
    """The optimizer a saved state was written by, from its param-group keys.

    One of the Trainer's optimizers (aurora, muon, nadamw, adamw, soap);
    ``UNRECOGNISED_OPTIMIZER`` for groups none of them writes; None only when
    the state has no groups to read (``_ChainedOptimizer`` omits them -- its
    layout is then compared by parameter names instead). Aurora/Muon are
    detected by a group that USES the matrix optimizer: their branch groups
    carry ``use_aurora: False`` / ``use_muon: False`` in either optimizer.
    """
    if not param_groups:
        return None
    keys = set().union(*(set(g) for g in param_groups))
    if any(g.get("use_aurora") for g in param_groups):
        return "aurora"
    if any(g.get("use_muon") for g in param_groups):
        return "muon"
    if "momentum_decay" in keys:
        return "nadamw"
    if "precondition_frequency" in keys or "shampoo_beta" in keys:
        return "soap"
    if "amsgrad" in keys:
        return "adamw"
    return UNRECOGNISED_OPTIMIZER


def check_init_optimizer_source(payload: dict[str, Any], checkpoint: Path, settings: TrainSettings) -> None:
    """The settings-only refusals of --init-optimizer-state checkpoint (no
    optimizer state; a different optimizer type than requested). Also run by
    the loop driver before pinning."""
    missing = [k for k in ("opt", "scheduler") if k not in payload]
    if missing or not isinstance(payload.get("opt"), dict) or not payload["opt"].get("state"):
        raise ValueError(f"--init-optimizer-state checkpoint: {checkpoint} has no optimizer state "
                         f"(missing {missing or ['opt.state (empty)']}); use --init-optimizer-state fresh")
    family = _optimizer_family(payload["opt"].get("param_groups") or [])
    want = str(settings.optimizer).lower()
    if family == UNRECOGNISED_OPTIMIZER:
        raise ValueError(f"--init-optimizer-state checkpoint: {checkpoint}'s optimizer state has param groups no "
                         "Trainer optimizer writes, so it cannot be matched to --optimizer "
                         f"{settings.optimizer}; use --init-optimizer-state fresh")
    if family is not None and family != ("nadamw" if want in ("nadam", "nadamw") else want):
        raise ValueError(f"--init-optimizer-state checkpoint: {checkpoint}'s optimizer state was written by "
                         f"{family}, not the requested --optimizer {settings.optimizer}")


def preflight_init_optimizer(checkpoint: Path, settings: TrainSettings) -> None:
    """Header read for the pre-pin check: mmap'd, nothing materialised."""
    import torch

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False, mmap=True)
    check_init_optimizer_source(payload, checkpoint, settings)


def init_optimizer_from_checkpoint(trainer: Any, checkpoint: Path, settings: TrainSettings) -> dict[str, Any]:
    """Load the init checkpoint's weights AND optimizer moments into a freshly
    built trainer (``--init-optimizer-state checkpoint``, generation 1).

    Refused loudly when the checkpoint carries no optimizer/scheduler state,
    when its parameter layout differs from this trainer's (optimizer type or
    ``--matrix-optimizer-scope``), or -- unless ``reapply_optimizer_hparams`` --
    when its per-group hyperparameters differ from the requested ones. The
    checkpoint's per-group lr and scheduler are its own run's schedule state and
    are not adopted: the requested schedule's lr is put back on every group.
    """
    import torch

    payload = torch.load(str(checkpoint), map_location=trainer.device, weights_only=False)
    check_init_optimizer_source(payload, checkpoint, settings)
    names = trainer._optimizer_param_names()
    ck_names = payload.get("opt_param_names")
    sizes = [len(g["params"]) for g in payload["opt"]["param_groups"]]
    mine = [len(g["params"]) for g in trainer.opt.param_groups]
    if ck_names is None or names is None or list(ck_names) != list(names) or sizes != mine:
        raise ValueError(f"--init-optimizer-state checkpoint: {checkpoint}'s optimizer parameter layout (group sizes "
                         f"{sizes}, names {'absent' if ck_names is None else 'compared'}) is not this trainer's "
                         f"(group sizes {mine}); match --optimizer and --matrix-optimizer-scope to the run that "
                         "wrote it")
    load_weights(trainer, checkpoint)
    configured = group_hparams(trainer._configured_param_group_hparams)
    want_lr = [float(g["lr"]) for g in trainer.opt.param_groups]  # the schedule's current (warmup) lr
    base_lrs = [float(v) for v in getattr(trainer._scheduler, "base_lrs", want_lr)]  # its peak per group
    try:
        trainer.opt.load_state_dict(payload["opt"])
    except (ValueError, KeyError) as exc:
        raise ValueError(f"--init-optimizer-state checkpoint: cannot load {checkpoint}'s optimizer state into "
                         f"{settings.optimizer}: {exc}") from exc
    checkpoint_lr = [float(g["lr"]) for g in trainer.opt.param_groups]
    # The clip's EMA is part of "continue the SL run's optimizer" only when the
    # clip is on; load_zclip_state refuses (False) a missing or poisoned state.
    zclip_restored = bool(trainer.grad_clip_mode == "zclip" and trainer.load_zclip_state(payload.get("zclip")))
    groups = _resolve_group_diff(trainer, configured, settings, "--init-optimizer-state checkpoint")
    for g, lr, base in zip(trainer.opt.param_groups, want_lr, base_lrs, strict=True):
        g["lr"] = lr
        if "initial_lr" in g:
            g["initial_lr"] = base  # scheduler bookkeeping: the peak, not the warmup-start lr
    steps = [int(v["step"].item()) if torch.is_tensor(v["step"]) else int(v["step"])
             for v in trainer.opt.state.values() if isinstance(v, dict) and "step" in v]
    sched = payload["scheduler"]
    return {"source": str(checkpoint), "state_entries": len(trainer.opt.state),
            "state_step_min": min(steps) if steps else None, "state_step_max": max(steps) if steps else None,
            "checkpoint_param_group_lr": checkpoint_lr, "param_group_lr_set": want_lr,
            **groups, "reapplied": groups["group_resolution"] == "configured",
            "not_adopted": ["scheduler", "per-group lr", "trainer step",
                            *(["zclip state"] if not zclip_restored else [])],
            "zclip_state_restored": zclip_restored,
            "scheduler_not_restored": {"checkpoint_peak_lr": payload.get("peak_lr"),
                                       "checkpoint_last_epoch": sched.get("last_epoch") if isinstance(sched, dict) else None}}


def build_trainer(model_config: Any, settings: TrainSettings, *, log_dir: Path | None,
                  phase_updates: int | None = None) -> Any:
    if settings.lr_schedule != "flat" and phase_updates is None:
        # trainer_kwargs alone (the driver's range check) tolerates it; a real Trainer must not run flat
        # while the settings record a decay.
        raise ValueError(f"lr_schedule {settings.lr_schedule} needs phase_updates to build a Trainer")
    from chess_anti_engine.model import build_model
    from chess_anti_engine.train import Trainer

    return Trainer(build_model(model_config), model_config=model_config, log_dir=log_dir,
                   **trainer_kwargs(settings, phase_updates=phase_updates))


def load_arch_config(checkpoint: Path) -> tuple[dict[str, Any], Any]:
    import torch

    from chess_anti_engine.uci.model_loader import model_config_from_arch

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    arch = dict(payload["arch"])
    return arch, model_config_from_arch(arch)


def load_weights(trainer: Any, checkpoint: Path) -> None:
    import torch

    from chess_anti_engine.train.trainer import align_compile_prefix

    payload = torch.load(str(checkpoint), map_location=trainer.device, weights_only=True)
    trainer.model.load_state_dict(align_compile_prefix(payload["model"], reference=trainer.model.state_dict()),
                                  strict=True)


def _forward(model: Any, x: np.ndarray, device: str, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from chess_anti_engine.train.frozen_policy_anchor import primary_logits

    pols, wdls = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out = model(torch.from_numpy(np.ascontiguousarray(x[i:i + batch])).float().to(device))
            pols.append(primary_logits(out).float().cpu().numpy())
            wdls.append(out["wdl"].float().cpu().numpy())
    if not pols:
        return np.zeros((0, 0), np.float32), np.zeros((0, 3), np.float32)
    return np.concatenate(pols), np.concatenate(wdls)


def _log_softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask, logits.astype(np.float64), -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def heldout_metrics(model: Any, rows: dict[str, np.ndarray], device: str) -> dict[str, float]:
    """Policy CE to the stored target and policy entropy over rows that HAVE a
    policy target (``has_policy``: fast playout-cap rows are value-only);
    value NLL to the outcome over every row."""
    n = int(rows["x"].shape[0])
    if n == 0:
        return {"rows": 0}
    pol, wdl = _forward(model, rows["x"], device)
    has = np.asarray(rows.get("has_policy", np.ones(n)), dtype=bool)
    mask = np.asarray(rows["legal_mask"])[has] > 0
    logp = _log_softmax_masked(pol[has], mask)
    p = np.where(mask, np.exp(logp), 0.0)
    target = np.asarray(rows["policy_target"], dtype=np.float64)[has]
    target /= target.sum(axis=1, keepdims=True)
    safe = np.where(mask, logp, 0.0)
    ce = -(target * safe).sum(axis=1)
    ent = -(p * safe).sum(axis=1)
    lw = wdl - wdl.max(axis=1, keepdims=True)
    lw = lw - np.log(np.exp(lw).sum(axis=1, keepdims=True))
    y = np.asarray(rows["wdl_target"]).astype(np.int64)
    nll = -lw[np.arange(n), y]
    nan = float("nan")
    return {"rows": n, "policy_rows": int(has.sum()),
            "policy_ce": float(ce.mean()) if ce.size else nan,
            "policy_entropy": float(ent.mean()) if ent.size else nan,
            "value_nll": float(nll.mean())}


def probe_drift(new: Any, old: Any, probe: dict[str, np.ndarray], device: str) -> dict[str, float]:
    """mean |Q_new - Q_old| (Q = W - L) and KL(old || new) on the fixed probe set."""
    if int(probe["x"].shape[0]) == 0:
        return {"probe_rows": 0}
    pol_n, wdl_n = _forward(new, probe["x"], device)
    pol_o, wdl_o = _forward(old, probe["x"], device)
    mask = np.asarray(probe["legal_mask"]) > 0

    def q(w: np.ndarray) -> np.ndarray:
        e = np.exp(w - w.max(axis=1, keepdims=True))
        e /= e.sum(axis=1, keepdims=True)
        return e[:, 0] - e[:, 2]

    lo, ln = _log_softmax_masked(pol_o, mask), _log_softmax_masked(pol_n, mask)
    kl = np.where(mask, np.exp(lo) * (np.where(mask, lo, 0.0) - np.where(mask, ln, 0.0)), 0.0).sum(axis=1)
    top1 = np.where(mask, pol_n, -np.inf).argmax(axis=1) == np.where(mask, pol_o, -np.inf).argmax(axis=1)
    return {"probe_rows": len(kl), "mean_abs_dq": float(np.abs(q(wdl_n) - q(wdl_o)).mean()),
            "policy_kl_prev_to_new": float(kl.mean()), "top1_agree_prev": float(top1.mean())}


SNAPSHOTS = "snapshots"
SNAPSHOT_RECORD = "snapshot.json"
RESUME_TRAINER = "resume_trainer.pt"
RESUME_STATE = "resume_state.pt"


def snapshot_dir(out_dir: Path, update: int) -> Path:
    return Path(out_dir) / SNAPSHOTS / f"step_{int(update)}"


def _rng_states(device: str) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {"numpy_global": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if str(device).startswith("cuda"):
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return out


def _set_rng_states(states: dict[str, Any]) -> None:
    import torch

    np.random.set_state(states["numpy_global"])
    torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states:
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def write_snapshot(*, out_dir: Path, update: int, trainer: Any, arch: dict[str, Any], sampler: ScheduledSampler,
                   anchor: ReferenceAnchor | None, chunks: list[dict[str, Any]], measure: Callable[[], dict[str, Any]],
                   resume_key: str) -> dict[str, Any]:
    """Write ``train/snapshots/step_<update>/`` mid-phase. Nothing here touches
    the training that continues afterwards (the RNG streams are saved first and
    the measurements run under ``fork_rng``).

    Kept: ``model.pt`` (weights + arch only, loadable by every checkpoint loader
    including the arena's) and ``metrics.json`` (``measure()``: held-out CE /
    entropy / value NLL, probe drift vs the phase's init model, realised lr).
    Transient, for resuming a killed phase from here: ``resume_trainer.pt``
    (``Trainer.save``: weights, optimizer, scheduler, step) and
    ``resume_state.pt`` (sampler cursor / served rows / RNG streams, anchor
    counters, the trainer metrics of the chunks so far). Only the newest
    snapshot keeps them, and sealing the stage removes them. ``snapshot.json``
    is written LAST: a step dir without it is incomplete and is redone.
    """
    import torch

    from chess_anti_engine.train.trainer import strip_compile_prefix

    d = snapshot_dir(out_dir, update)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    rng = _rng_states(str(trainer.device))
    weights = d / MODEL_FILE
    tmp = d / f".{MODEL_FILE}.tmp"
    torch.save({"model": strip_compile_prefix(trainer.model.state_dict()), "arch": arch}, str(tmp))
    tmp.replace(weights)
    devices = [torch.cuda.current_device()] if str(trainer.device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        metrics = {"update": int(update), "trainer_step": int(trainer.step), **measure()}
    write_json_atomic(d / "metrics.json", metrics)
    trainer.save(d / RESUME_TRAINER)
    torch.save({"resume_key": resume_key, "update": int(update), "cursor": int(sampler.cursor),
                "served": list(sampler.served), "served_sl": list(sampler.served_sl),
                "drawn_rows": int(sampler.drawn_rows), "realised_sl_rows": int(sampler.realised_sl_rows),
                "sampler_rng": sampler.rng.bit_generator.state, "rng": rng, "chunks": chunks,
                "anchor": None if anchor is None else {k: getattr(anchor, k) for k in _ANCHOR_COUNTERS}},
               str(d / RESUME_STATE))
    record = {"update": int(update), "resume_key": resume_key,
              "files": {name: input_ref(d / name) for name in (MODEL_FILE, "metrics.json")}}
    write_json_atomic(d / SNAPSHOT_RECORD, record)
    for other in Path(out_dir, SNAPSHOTS).glob("step_*"):
        if other != d:
            for name in (RESUME_TRAINER, RESUME_STATE):
                (other / name).unlink(missing_ok=True)
    return {**record, "metrics": metrics}


_ANCHOR_COUNTERS = ("rows", "forwards", "policy_kl_sum", "wdl_kl_sum")


def resumable_snapshot(out_dir: Path, resume_key: str) -> int | None:
    """The newest COMPLETE snapshot of an unsealed attempt with this exact
    resume key (same settings, inputs and batch plan) whose resume files are
    still present; None when the phase must start from its first update."""
    best = None
    for d in Path(out_dir, SNAPSHOTS).glob("step_*"):
        rec = d / SNAPSHOT_RECORD
        if not (rec.is_file() and (d / RESUME_TRAINER).is_file() and (d / RESUME_STATE).is_file()):
            continue
        data = json.loads(rec.read_text())
        if data.get("resume_key") == resume_key and (best is None or int(data["update"]) > best):
            best = int(data["update"])
    return best


def resume_from_snapshot(out_dir: Path, update: int, *, trainer: Any, sampler: ScheduledSampler,
                         anchor: ReferenceAnchor | None) -> list[dict[str, Any]]:
    """Restore the phase's whole state as it was when ``step_<update>`` was
    written; returns the trainer metrics of the chunks already trained."""
    import torch

    d = snapshot_dir(out_dir, update)
    trainer.load(d / RESUME_TRAINER, exact_resume=True)
    state = torch.load(str(d / RESUME_STATE), map_location="cpu", weights_only=False)
    sampler.cursor = int(state["cursor"])
    sampler.served = list(state["served"])
    sampler.served_sl = list(state["served_sl"])
    sampler.drawn_rows = int(state["drawn_rows"])
    sampler.realised_sl_rows = int(state["realised_sl_rows"])
    sampler.rng.bit_generator.state = state["sampler_rng"]
    if anchor is not None and state["anchor"] is not None:
        for k in _ANCHOR_COUNTERS:
            setattr(anchor, k, state["anchor"][k])
    _set_rng_states(state["rng"])
    return list(state["chunks"])


def merged_losses(chunks: list[dict[str, Any]]) -> dict[str, float]:
    """The phase's mean losses over its chunks, weighted by the rows each saw
    (the exact-once metrics are per-row means)."""
    rows = [float(c.get("train_samples_seen", 0) or 0) for c in chunks]
    total = sum(rows)
    out = {}
    for k in ("loss", "policy_loss", "wdl_loss", "blended_wdl_loss"):
        if all(k in c for c in chunks) and total > 0:
            out[k] = float(sum(float(c[k]) * r for c, r in zip(chunks, rows, strict=True)) / total)
    return out


def train_generation(*, run_dir: Path, generation: int, init_checkpoint: Path,
                     prev_trainer_checkpoint: Path | None, settings: TrainSettings) -> dict[str, Any]:
    import torch

    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    run_dir = Path(run_dir)
    settings.validate()
    planned = planned_phase_updates(run_dir, generation, settings)
    if settings.snapshot_updates:
        # Before ANY write (stage dir, run-level probe.npz): a refused plan leaves the run as it was.
        if planned is None:
            raise ValueError(f"generation {generation}: a phase collect is not sealed; refusing to train")
        check_snapshot_plan(settings.snapshot_updates, planned)
    out_dir = run_dir / f"gen_{generation:04d}" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").unlink(missing_ok=True)  # a redo is unsealed until its own manifest lands
    seed = seed_for(settings.seed, "train", generation)
    global_seed = seed & 0xFFFFFFFF  # numpy's legacy seed is 32-bit; record what was realised
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    arch, model_config = load_arch_config(init_checkpoint)
    capped = capped_collects(run_dir, phase_generations(generation, settings))
    lineage_capped = lineage_capped_collects(run_dir, generation)
    prev_train = Path(run_dir) / f"gen_{generation - 1:04d}" / "train" / "metrics.json"
    inherited_taint = bool(lineage_capped) or bool(prev_train.is_file() and json.loads(
        prev_train.read_text())["summary"].get("trained_on_capped_rows", False))
    refuse_capped_collects(lineage_capped, settings)  # direct OR inherited: the same refusal
    train_rows, holdout, gens, row_ids = load_window(run_dir, generation, settings)
    reference_prior: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    if float(settings.policy_prior_mix) < 1.0 and settings.prior_mix_source == "reference":
        if not settings.prior_mix_reference:
            raise ValueError("policy_prior_mix < 1 with prior_mix_source=reference needs prior_mix_reference")
        ref_model = load_model_from_checkpoint(Path(settings.prior_mix_reference), device=settings.device)

        def _reference_prior(x: np.ndarray, legal: np.ndarray) -> np.ndarray:
            pol, _ = _forward(ref_model, x, settings.device)
            return np.where(legal, np.exp(_log_softmax_masked(pol, legal)), 0.0)

        reference_prior = _reference_prior

    train_rows, target_info = prepare_targets(train_rows, settings, reference_prior)
    holdout = {k: v for k, v in holdout.items() if k not in TARGET_SOURCE_KEYS}
    if settings.optimizer_state not in ("fresh", "continue"):
        raise ValueError("optimizer_state must be fresh or continue")

    if settings.lr_schedule != "flat" and planned is None:
        raise ValueError(f"generation {generation}: lr_schedule {settings.lr_schedule} needs the phase's planned "
                         "updates, and a phase collect is not sealed")
    trainer = build_trainer(model_config, settings, log_dir=None,
                            phase_updates=planned if settings.lr_schedule != "flat" else None)
    continued = settings.optimizer_state == "continue" and prev_trainer_checkpoint is not None
    optimizer_check: dict[str, Any] | None = None
    init_optimizer: dict[str, Any] | None = None
    if continued:
        assert prev_trainer_checkpoint is not None
        optimizer_check = continue_optimizer(trainer, prev_trainer_checkpoint, settings)
    elif settings.init_optimizer_state == "checkpoint" and int(generation) == 1:
        init_optimizer = init_optimizer_from_checkpoint(trainer, init_checkpoint, settings)
    else:
        load_weights(trainer, init_checkpoint)

    probe_path = run_dir / "probe.npz"
    if not probe_path.exists():
        # Fixed for the whole run: the first trained generation's held-out rows
        # (its training rows only if nothing was held out).
        source = holdout if int(holdout["x"].shape[0]) > 0 else train_rows
        n = min(int(settings.probe_rows), int(source["x"].shape[0]))
        np.savez(probe_path, x=source["x"][:n], legal_mask=source["legal_mask"][:n],
                 source=np.asarray("holdout" if source is holdout else "train"))
    with np.load(probe_path) as z:
        probe = {k: z[k] for k in ("x", "legal_mask")}

    before = heldout_metrics(trainer.model, holdout, settings.device)
    counts_before, sl_cursor = presentation_counts(run_dir, before_generation=generation)
    sl_rows = None
    sl_shards: list[Path] = []
    if settings.sl_mix_frac > 0:
        if not settings.sl_mix_dir:
            raise ValueError("--sl-mix-frac > 0 needs --sl-mix-dir")
        # One corpus order for the whole run (not per generation), consumed
        # from the ledger's cursor: no SL row is used twice in a run.
        sl_rows = load_sl_corpus(Path(settings.sl_mix_dir), max_rows=settings.sl_mix_max_rows,
                                 seed=seed_for(settings.seed, "sl"),
                                 need_search=settings.w_wdl > 0 and settings.search_wdl_frac() > 0,
                                 consumed=sl_shards)
        perm = np.random.default_rng(seed_for(settings.seed, "sl-order")).permutation(int(sl_rows["x"].shape[0]))
        sl_rows = {k: (v if str(k).startswith("_") else np.asarray(v)[perm]) for k, v in sl_rows.items()}
    sampler: ScheduledSampler | WindowSampler
    if settings.sample_with_replacement:
        sampler = WindowSampler(train_rows, rng=np.random.default_rng(seed), sl_rows=sl_rows,
                                sl_frac=settings.sl_mix_frac)
        passes = 1.0 if settings.passes is None else float(settings.passes)
        updates = int(settings.updates) if settings.updates > 0 else max(
            1, math.ceil(passes * sampler.n / int(settings.batch_size)))
        sl_end = sl_cursor
    else:
        # The double-presentation guard: every row this phase presents must
        # have been presented exactly as often as the schedule says so far.
        for g in gens:
            expected = generation - g  # 0 for the new generation, 1 for the previous (P=2)
            sel = row_ids[:, 0] == g
            seen = counts_before.get(g, np.zeros(int(row_ids[sel, 1].max(initial=-1)) + 1, np.int64))
            got = seen[row_ids[sel, 1]] if seen.size else np.zeros(int(sel.sum()), np.int64)
            if not np.all(got == expected):
                raise RuntimeError(f"presentation ledger: generation {g} rows have counts "
                                   f"{sorted(set(got.tolist()))} before phase {generation}, expected {expected}")
        batches, sl_end = plan_batches(
            int(row_ids.shape[0]), batch_size=int(settings.batch_size), sl_frac=settings.sl_mix_frac,
            sl_available=0 if sl_rows is None else int(sl_rows["x"].shape[0]),
            rng=np.random.default_rng(seed), sl_start=sl_cursor)
        sampler = ScheduledSampler(train_rows, batches, rng=np.random.default_rng(seed ^ 0x5A5A), sl_rows=sl_rows)
        updates = len(batches)
        if planned is not None and planned != updates:
            raise RuntimeError(f"planned_phase_updates said {planned} updates, the batch plan has {updates}")

    anchor = None
    if settings.kl_policy > 0 or settings.kl_wdl > 0:
        if not settings.kl_reference:
            raise ValueError("a KL anchor needs a reference checkpoint")
        reference = load_model_from_checkpoint(Path(settings.kl_reference), device=settings.device)
        anchor = ReferenceAnchor(reference, policy_coef=settings.kl_policy, wdl_coef=settings.kl_wdl)
        attach_anchor(trainer, anchor)

    start_step = int(trainer.step)
    snapshots = [int(k) for k in settings.snapshot_updates]
    check_snapshot_plan(snapshots, updates)
    prev_model: Any = None

    def previous_model() -> Any:
        nonlocal prev_model
        if prev_model is None:
            prev_model = load_model_from_checkpoint(init_checkpoint, device=settings.device)
        return prev_model

    chunks: list[dict[str, Any]] = []
    snapshot_records: list[dict[str, Any]] = []
    resumed_from: int | None = None
    done = 0
    t0 = time.perf_counter()
    if not snapshots:
        chunks.append(asdict(trainer.train_steps(sampler, batch_size=int(settings.batch_size), steps=updates)))
        done = updates
    else:
        assert isinstance(sampler, ScheduledSampler)  # validate(): exact-once only
        resume_key = snapshot_resume_key(settings=settings, generation=generation, init_checkpoint=init_checkpoint,
                                         prev_trainer_checkpoint=prev_trainer_checkpoint if continued else None,
                                         run_dir=run_dir, gens=gens, probe_path=probe_path, batches=sampler.batches)
        resumed_from = resumable_snapshot(out_dir, resume_key)
        for d in Path(out_dir, SNAPSHOTS).glob("step_*"):
            # Stale (another attempt's plan) or past the resume point (never completed): redone.
            if resumed_from is None or int(d.name.split("_")[1]) > resumed_from:
                shutil.rmtree(d)
        if resumed_from is not None:
            chunks = resume_from_snapshot(out_dir, resumed_from, trainer=trainer, sampler=sampler, anchor=anchor)
            if int(trainer.step) != start_step + resumed_from:
                raise RuntimeError(f"snapshot step_{resumed_from} restored trainer step {int(trainer.step)}, "
                                   f"expected {start_step + resumed_from}")
            done = resumed_from
            snapshot_records = [json.loads((snapshot_dir(out_dir, k) / SNAPSHOT_RECORD).read_text())
                                for k in snapshots if k <= resumed_from]

        def measure() -> dict[str, Any]:
            return {"heldout": heldout_metrics(trainer.model, holdout, settings.device),
                    "probe_vs_previous": probe_drift(trainer.model, previous_model(), probe, settings.device),
                    "realised_lr_param_groups": [float(g["lr"]) for g in trainer.opt.param_groups],
                    "rows_presented": int(sum(int(a.size) for a in sampler.served)),
                    "losses_so_far": merged_losses(chunks)}

        for k in [*snapshots, updates]:
            if k <= done:
                continue
            chunks.append(asdict(trainer.train_steps(sampler, batch_size=int(settings.batch_size), steps=k - done)))
            done = k
            if k in snapshots:
                rec = write_snapshot(out_dir=out_dir, update=k, trainer=trainer, arch=arch, sampler=sampler,
                                     anchor=anchor, chunks=chunks, measure=measure, resume_key=resume_key)
                snapshot_records.append({key: rec[key] for key in ("update", "resume_key", "files")})
    wall = time.perf_counter() - t0
    if int(trainer.step) != start_step + updates:
        raise RuntimeError(f"Trainer applied {int(trainer.step) - start_step} of {updates} updates")
    if isinstance(sampler, ScheduledSampler):
        if sampler.cursor != len(sampler.batches):
            raise RuntimeError(f"Trainer consumed {sampler.cursor} of {len(sampler.batches)} planned batches")
        served = np.concatenate(sampler.served) if sampler.served else np.zeros(0, np.int64)
        sl_served = np.concatenate(sampler.served_sl) if sampler.served_sl else np.zeros(0, np.int64)
    else:
        served = np.zeros(0, np.int64)  # with replacement: rows are not tracked per presentation
        sl_served = np.zeros(0, np.int64)
    presented_ids = row_ids[served] if served.size else np.zeros((0, 2), np.int64)
    model_path = out_dir / MODEL_FILE
    trainer.save(model_path)
    np.savez(out_dir / PRESENTED, gen=presented_ids[:, 0].astype(np.int32), row=presented_ids[:, 1].astype(np.int32),
             sl_start=np.asarray(sl_cursor), sl_end=np.asarray(sl_end), sl_rows=sl_served.astype(np.int64))
    after = heldout_metrics(trainer.model, holdout, settings.device)
    drift = probe_drift(trainer.model, previous_model(), probe, settings.device)
    realised_lr = [float(g["lr"]) for g in trainer.opt.param_groups]
    realised_opt = realised_optimizer(trainer)  # read off the live optimizer after the phase's steps
    # One train_steps call without snapshots (its metrics verbatim); one per
    # snapshot interval with them (rows-weighted mean losses, chunks kept).
    m = chunks[0] if len(chunks) == 1 else {**merged_losses(chunks), "chunks": chunks}
    counts_after = {g: c.copy() for g, c in counts_before.items()}
    for g in np.unique(presented_ids[:, 0]) if presented_ids.size else []:
        g = int(g)
        if g not in counts_after:
            counts_after[g] = np.zeros(len(_generation_meta(run_dir, g)["game_id"]), dtype=np.int64)
        np.add.at(counts_after[g], presented_ids[presented_ids[:, 0] == g, 1], 1)
    histograms = presentation_histograms(run_dir, counts_after, holdout_frac=settings.holdout_frac,
                                         through_generation=generation, presentations=int(settings.presentations))
    if not settings.sample_with_replacement:
        for g, h in histograms.items():
            want = {str(int(settings.presentations)): h["trainable_rows"]} if h["trainable_rows"] else {}
            if h["lifetime_complete"] and h["histogram"] != want:
                raise RuntimeError(f"presentation ledger: generation {g} ended its lifetime with {h['histogram']}, "
                                   f"expected {want}")
    summary = {
        "updates": updates, "start_step": start_step, "end_step": int(trainer.step),
        "sampling": "with_replacement" if settings.sample_with_replacement else "scheduled_exact_once",
        "presentations": int(settings.presentations),
        "presentation_histograms": histograms,
        "sl_cursor": [sl_cursor, sl_end],
        "window_generations": gens, "window_rows": sampler.n, "holdout_rows": int(holdout["x"].shape[0]),
        "drawn_rows": sampler.drawn_rows, "sl_rows_drawn": sampler.realised_sl_rows,
        "sl_frac_realised": sampler.realised_sl_rows / max(1, sampler.drawn_rows),
        "capped_collect_rows": {str(k): v for k, v in capped.items()},
        "capped_lineage_collects": {str(k): v for k, v in lineage_capped.items()},
        # this phase's rows, or any earlier phase of the lineage it starts from
        "trained_on_capped_rows": bool(capped) or inherited_taint,
        "optimizer_continued": continued, "optimizer_from_init_checkpoint": init_optimizer,
        "realised_lr_param_groups": realised_lr,
        "realised_optimizer": realised_opt, "optimizer_continue_check": optimizer_check,
        "realised_search_wdl_frac": float(trainer._loss_kwargs["search_wdl_frac"]),
        "realised_w_wdl": float(trainer._loss_kwargs["w_wdl"]),
        "realised_policy_target_temp": float(trainer._loss_kwargs["policy_target_temp"]),
        "targets": target_info,
        "losses": {k: m[k] for k in ("loss", "policy_loss", "wdl_loss", "blended_wdl_loss") if k in m},
        "heldout_before": before, "heldout_after": after, "probe_vs_previous": drift,
        "anchor": None if anchor is None else anchor.diagnostics(), "wall_seconds": wall,
        "updates_per_second": (updates - (resumed_from or 0)) / max(wall, 1e-9),
        "snapshot_updates": snapshots, "resumed_from_snapshot": resumed_from,
        "snapshots": [{"update": r["update"], "model_sha256": r["files"][MODEL_FILE]["sha256"],
                       "metrics_sha256": r["files"]["metrics.json"]["sha256"]} for r in snapshot_records],
    }
    write_json_atomic(out_dir / "metrics.json", {"summary": summary, "trainer_metrics": m})
    inputs = {"init_checkpoint": input_ref(init_checkpoint),
              **{f"gen_{g:04d}": input_ref(run_dir / f"gen_{g:04d}" / "collect" / "manifest.json") for g in gens}}
    if continued and prev_trainer_checkpoint is not None:
        inputs["prev_trainer_checkpoint"] = input_ref(prev_trainer_checkpoint)
    if settings.kl_reference and anchor is not None:
        inputs["kl_reference"] = input_ref(Path(settings.kl_reference))
    if reference_prior is not None and settings.prior_mix_reference:
        inputs["prior_mix_reference"] = input_ref(Path(settings.prior_mix_reference))
    inputs["probe"] = input_ref(probe_path)
    for k, shard in enumerate(sl_shards):
        inputs[f"sl_shard_{k:04d}"] = input_ref(shard)
    # Each snapshot's files, re-hashed now: the seal vouches for what is on disk.
    snapshot_outputs = {f"snapshot_{r['update']}_{name.split('.')[0]}": input_ref(snapshot_dir(out_dir, r["update"]) / name)
                        for r in snapshot_records for name in (MODEL_FILE, "metrics.json")}
    for r in snapshot_records:
        for name in (MODEL_FILE, "metrics.json"):
            if snapshot_outputs[f"snapshot_{r['update']}_{name.split('.')[0]}"]["sha256"] != r["files"][name]["sha256"]:
                raise RuntimeError(f"snapshot step_{r['update']}/{name} changed after it was written")
    # Run-level view of the ledger: derived from the phases' presented.npz and
    # rewritten by every phase, so it is written BEFORE the manifest seals this one.
    write_json_atomic(run_dir / LEDGER, {"through_generation": generation, "sampling": summary["sampling"],
                                         "presentations": int(settings.presentations),
                                         "trained_on_capped_rows": summary["trained_on_capped_rows"],
                                         "capped_collect_rows": summary["capped_collect_rows"],
                                         "capped_lineage_collects": summary["capped_lineage_collects"],
                                         "generations": histograms, "sl_rows_used": sl_end})
    write_manifest(out_dir, stage="train",
                   config={"settings": asdict(settings), "arch": arch,
                           # realised, read off the objects that consumed them
                           "realised": {"optimizer": realised_opt, "amp": bool(trainer.use_amp),
                                        "device": str(trainer.device)}},
                   inputs=inputs, outputs={"model": input_ref(model_path), "presented": input_ref(out_dir / PRESENTED),
                                           **snapshot_outputs},
                   counts={"updates": updates, "window_rows": sampler.n},
                   seeds={"train": seed, "global_seed_realised": global_seed, "base": settings.seed},
                   extra={"generation": generation, "summary": summary})
    for r in snapshot_records:  # sealed: the resume files have no further use
        for name in (RESUME_TRAINER, RESUME_STATE):
            (snapshot_dir(out_dir, r["update"]) / name).unlink(missing_ok=True)
    return summary


def snapshot_resume_key(*, settings: TrainSettings, generation: int, init_checkpoint: Path,
                        prev_trainer_checkpoint: Path | None, run_dir: Path, gens: list[int], probe_path: Path,
                        batches: list[tuple[np.ndarray, np.ndarray]]) -> str:
    """Identity of one phase attempt: a snapshot resumes only an attempt with
    the same settings, inputs (by sha256) and batch plan."""
    h = hashlib.sha256()
    for own, sl in batches:
        h.update(np.ascontiguousarray(own, dtype=np.int64).tobytes())
        h.update(b"|")
        h.update(np.ascontiguousarray(sl, dtype=np.int64).tobytes())
        h.update(b"\n")
    ident = {"settings": asdict(settings), "generation": int(generation),
             "init": sha256_file(init_checkpoint),
             "prev": None if prev_trainer_checkpoint is None else sha256_file(prev_trainer_checkpoint),
             "collects": {g: sha256_file(Path(run_dir) / f"gen_{g:04d}" / "collect" / "manifest.json") for g in gens},
             "probe": sha256_file(probe_path), "plan": h.hexdigest()}
    return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()
