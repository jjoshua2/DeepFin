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
import json
import math
from collections.abc import Callable
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.simple_rl.collect import concat_arrays, load_generation_rows
from chess_anti_engine.simple_rl.common import input_ref, seed_for, write_json_atomic, write_manifest

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
MODEL_FILE = "model.pt"


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


def holdout_mask(meta: dict[str, np.ndarray], frac: float) -> np.ndarray:
    return np.asarray([is_holdout_game(int(gid), frac) for gid in meta["game_id"]], dtype=bool)


PRESENTED = "presented.npz"
LEDGER = "presentation_ledger.json"


def _generation_meta(run_dir: Path, g: int) -> dict[str, np.ndarray]:
    metas = []
    for m in sorted((Path(run_dir) / f"gen_{g:04d}" / "collect" / "shards").glob("shard_*.meta.npz")):
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


def trainer_kwargs(settings: TrainSettings) -> dict[str, Any]:
    from chess_anti_engine.train.losses import policy_target_temp_active

    policy_target_temp_active(settings.policy_target_temp)  # range check, raises on a typo
    floor = settings.lr if settings.lr_floor is None else settings.lr_floor
    return {
        "optimizer": settings.optimizer, "weight_decay_mode": settings.weight_decay_mode,
        "matrix_optimizer_scope": "default", "grad_clip_mode": "none", "zclip_max_norm": None,
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
        "lr_T0": 10**9, "lr_T_mult": 1, "mirror_prob": settings.mirror_prob,
    }


_GROUP_HPARAMS = ("weight_decay", "betas", "eps", "momentum_decay")


def optimizer_hparams(trainer: Any) -> dict[str, Any]:
    """The optimizer hyperparameters a restore can carry, read off the live
    trainer (scheduler + param groups), JSON-ready."""
    def plain(v: Any) -> Any:
        return [float(x) for x in v] if isinstance(v, (tuple, list)) else float(v)

    sched = trainer._scheduler
    return {
        "peak_lr": float(trainer._peak_lr),
        "base_lrs": [float(v) for v in getattr(sched, "base_lrs", [])],
        "lr_floor": float(getattr(sched, "eta_min", float("nan"))),
        "groups": [{k: plain(g[k]) for k in _GROUP_HPARAMS if k in g} for g in trainer.opt.param_groups],
    }


def realised_optimizer(trainer: Any) -> dict[str, Any]:
    out = optimizer_hparams(trainer)
    out["param_group_lr"] = [float(g["lr"]) for g in trainer.opt.param_groups]
    return out


def _hparam_diff(want: dict[str, Any], have: dict[str, Any]) -> dict[str, Any]:
    import math as _m

    def close(a: Any, b: Any) -> bool:
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b, strict=True))
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
        if isinstance(a, float) and isinstance(b, float):
            return (_m.isnan(a) and _m.isnan(b)) or _m.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
        return a == b

    return {k: {"checkpoint": have.get(k), "requested": want[k]} for k in want if not close(want[k], have.get(k))}


def continue_optimizer(trainer: Any, checkpoint: Path, settings: TrainSettings) -> dict[str, Any]:
    """``trainer.load(exact_resume=True)`` restores the checkpoint's optimizer
    hyperparameters along with its moments, so the requested ``--lr`` /
    ``--lr-floor`` / weight decay would be silently ignored. Refuse any
    difference, or with ``reapply_optimizer_hparams`` re-apply the requested
    values (schedule phase kept). Returns what was compared and done."""
    requested = optimizer_hparams(trainer)
    trainer.load(checkpoint, exact_resume=True)
    restored = optimizer_hparams(trainer)
    want = {k: requested[k] for k in ("peak_lr", "lr_floor", "groups")}
    diff = _hparam_diff(want, restored)
    if diff and not settings.reapply_optimizer_hparams:
        raise ValueError(f"--optimizer-state continue: the checkpoint's optimizer hyperparameters differ from the "
                         f"requested ones {json.dumps(diff, default=str)}; restoring would silently ignore them. "
                         "Pass --reapply-optimizer-hparams to apply the requested values, or request the "
                         "checkpoint's")
    if diff:
        trainer._reapply_configured_param_group_hparams()  # weight decay & co, from this trainer's config
        trainer.set_peak_lr(float(settings.lr))  # base_lrs, active lr and _peak_lr, phase preserved
        if hasattr(trainer._scheduler, "eta_min"):
            trainer._scheduler.eta_min = float(requested["lr_floor"])
        left = _hparam_diff(want, optimizer_hparams(trainer))
        if left:
            raise RuntimeError(f"re-applying optimizer hyperparameters left differences: {left}")
    return {"differed": diff, "reapplied": bool(diff)}


def build_trainer(model_config: Any, settings: TrainSettings, *, log_dir: Path | None) -> Any:
    from chess_anti_engine.model import build_model
    from chess_anti_engine.train import Trainer

    return Trainer(build_model(model_config), model_config=model_config, log_dir=log_dir,
                   **trainer_kwargs(settings))


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
    return {"probe_rows": len(kl), "mean_abs_dq": float(np.abs(q(wdl_n) - q(wdl_o)).mean()),
            "policy_kl_prev_to_new": float(kl.mean())}


def train_generation(*, run_dir: Path, generation: int, init_checkpoint: Path,
                     prev_trainer_checkpoint: Path | None, settings: TrainSettings) -> dict[str, Any]:
    import torch

    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    run_dir = Path(run_dir)
    out_dir = run_dir / f"gen_{generation:04d}" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").unlink(missing_ok=True)  # a redo is unsealed until its own manifest lands
    seed = seed_for(settings.seed, "train", generation)
    global_seed = seed & 0xFFFFFFFF  # numpy's legacy seed is 32-bit; record what was realised
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    settings.validate()
    arch, model_config = load_arch_config(init_checkpoint)
    capped = capped_collects(run_dir, phase_generations(generation, settings))
    refuse_capped_collects(capped, settings)
    prev_train = Path(run_dir) / f"gen_{generation - 1:04d}" / "train" / "metrics.json"
    inherited_taint = bool(prev_train.is_file() and json.loads(prev_train.read_text())["summary"].get(
        "trained_on_capped_rows", False))
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

    trainer = build_trainer(model_config, settings, log_dir=None)
    continued = settings.optimizer_state == "continue" and prev_trainer_checkpoint is not None
    optimizer_check: dict[str, Any] | None = None
    if continued:
        assert prev_trainer_checkpoint is not None
        optimizer_check = continue_optimizer(trainer, prev_trainer_checkpoint, settings)
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

    anchor = None
    if settings.kl_policy > 0 or settings.kl_wdl > 0:
        if not settings.kl_reference:
            raise ValueError("a KL anchor needs a reference checkpoint")
        reference = load_model_from_checkpoint(Path(settings.kl_reference), device=settings.device)
        anchor = ReferenceAnchor(reference, policy_coef=settings.kl_policy, wdl_coef=settings.kl_wdl)
        attach_anchor(trainer, anchor)

    start_step = int(trainer.step)
    t0 = time.perf_counter()
    metrics = trainer.train_steps(sampler, batch_size=int(settings.batch_size), steps=updates)
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
    prev_model = load_model_from_checkpoint(init_checkpoint, device=settings.device)
    drift = probe_drift(trainer.model, prev_model, probe, settings.device)
    realised_lr = [float(g["lr"]) for g in trainer.opt.param_groups]
    realised_opt = realised_optimizer(trainer)  # read off the live optimizer after the phase's steps
    m = asdict(metrics)
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
        # this phase's rows, or any earlier phase of the lineage it starts from
        "trained_on_capped_rows": bool(capped) or inherited_taint,
        "optimizer_continued": continued, "realised_lr_param_groups": realised_lr,
        "realised_optimizer": realised_opt, "optimizer_continue_check": optimizer_check,
        "realised_search_wdl_frac": float(trainer._loss_kwargs["search_wdl_frac"]),
        "realised_w_wdl": float(trainer._loss_kwargs["w_wdl"]),
        "realised_policy_target_temp": float(trainer._loss_kwargs["policy_target_temp"]),
        "targets": target_info,
        "losses": {k: m[k] for k in ("loss", "policy_loss", "wdl_loss", "blended_wdl_loss") if k in m},
        "heldout_before": before, "heldout_after": after, "probe_vs_previous": drift,
        "anchor": None if anchor is None else anchor.diagnostics(), "wall_seconds": wall,
        "updates_per_second": updates / max(wall, 1e-9),
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
    # Run-level view of the ledger: derived from the phases' presented.npz and
    # rewritten by every phase, so it is written BEFORE the manifest seals this one.
    write_json_atomic(run_dir / LEDGER, {"through_generation": generation, "sampling": summary["sampling"],
                                         "presentations": int(settings.presentations),
                                         "trained_on_capped_rows": summary["trained_on_capped_rows"],
                                         "capped_collect_rows": summary["capped_collect_rows"],
                                         "generations": histograms, "sl_rows_used": sl_end})
    write_manifest(out_dir, stage="train",
                   config={"settings": asdict(settings), "arch": arch,
                           # realised, read off the objects that consumed them
                           "realised": {"optimizer": realised_opt, "amp": bool(trainer.use_amp),
                                        "device": str(trainer.device)}},
                   inputs=inputs, outputs={"model": input_ref(model_path), "presented": input_ref(out_dir / PRESENTED)},
                   counts={"updates": updates, "window_rows": sampler.n},
                   seeds={"train": seed, "global_seed_realised": global_seed, "base": settings.seed},
                   extra={"generation": generation, "summary": summary})
    return summary
