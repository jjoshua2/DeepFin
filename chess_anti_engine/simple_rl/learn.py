"""One training phase over a sliding window of the last K generations.

Rows are drawn uniformly WITH replacement from the window on every update, so
each update sees fresh rows and ``--passes`` is only a dose (updates =
passes x window rows / batch). The repo's ``Trainer`` does the step: its
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
    window: int = 4
    batch_size: int = 256
    updates: int = 0
    passes: float = 1.0
    lr: float = 3e-5
    lr_floor: float | None = None
    warmup_steps: int = 0
    warmup_start: float = 0.0
    optimizer: str = "nadamw"
    optimizer_state: str = "fresh"
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

    def search_wdl_frac(self) -> float:
        total = float(self.value_outcome_weight) + float(self.value_search_weight)
        if total <= 0:
            raise ValueError("value_outcome_weight + value_search_weight must be > 0")
        return float(self.value_search_weight) / total

    def validate(self) -> None:
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


def load_window(run_dir: Path, generation: int, settings: TrainSettings) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[int]]:
    """(train rows over the window, held-out rows of the newest generation, gens)."""
    first = max(1, generation - int(settings.window) + 1)
    gens = list(range(first, generation + 1))
    train_parts: list[dict[str, np.ndarray]] = []
    holdout: dict[str, np.ndarray] | None = None
    for g in gens:
        arrays, meta = load_generation_rows(Path(run_dir) / f"gen_{g:04d}" / "collect",
                                            keys=(*TRAIN_KEYS, *TARGET_SOURCE_KEYS))
        held = np.asarray([is_holdout_game(int(gid), settings.holdout_frac) for gid in meta["game_id"]], dtype=bool)
        train_parts.append(_select(arrays, ~held))
        if g == generation:
            holdout = _select(arrays, held)
    assert holdout is not None
    return concat_arrays(train_parts), holdout, gens


def load_sl_corpus(path: Path, *, max_rows: int, seed: int, need_search: bool) -> dict[str, np.ndarray]:
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
    """Policy CE to the stored target, policy entropy, value NLL to the outcome."""
    n = int(rows["x"].shape[0])
    if n == 0:
        return {"rows": 0}
    pol, wdl = _forward(model, rows["x"], device)
    mask = np.asarray(rows["legal_mask"]) > 0
    logp = _log_softmax_masked(pol, mask)
    p = np.where(mask, np.exp(logp), 0.0)
    target = np.asarray(rows["policy_target"], dtype=np.float64)
    target /= target.sum(axis=1, keepdims=True)
    safe = np.where(mask, logp, 0.0)
    ce = -(target * safe).sum(axis=1)
    ent = -(p * safe).sum(axis=1)
    lw = wdl - wdl.max(axis=1, keepdims=True)
    lw = lw - np.log(np.exp(lw).sum(axis=1, keepdims=True))
    y = np.asarray(rows["wdl_target"]).astype(np.int64)
    nll = -lw[np.arange(n), y]
    return {"rows": n, "policy_ce": float(ce.mean()), "policy_entropy": float(ent.mean()),
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
    seed = seed_for(settings.seed, "train", generation)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed & 0xFFFFFFFF)
    settings.validate()
    arch, model_config = load_arch_config(init_checkpoint)
    train_rows, holdout, gens = load_window(run_dir, generation, settings)
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
    if continued:
        assert prev_trainer_checkpoint is not None
        trainer.load(prev_trainer_checkpoint, exact_resume=True)
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
    sl_rows = None
    if settings.sl_mix_frac > 0:
        if not settings.sl_mix_dir:
            raise ValueError("--sl-mix-frac > 0 needs --sl-mix-dir")
        sl_rows = load_sl_corpus(Path(settings.sl_mix_dir), max_rows=settings.sl_mix_max_rows,
                                 seed=seed_for(settings.seed, "sl", generation),
                                 need_search=settings.w_wdl > 0 and settings.search_wdl_frac() > 0)
    sampler = WindowSampler(train_rows, rng=np.random.default_rng(seed), sl_rows=sl_rows, sl_frac=settings.sl_mix_frac)
    updates = int(settings.updates) if settings.updates > 0 else max(
        1, math.ceil(float(settings.passes) * sampler.n / int(settings.batch_size)))

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
    model_path = out_dir / MODEL_FILE
    trainer.save(model_path)
    after = heldout_metrics(trainer.model, holdout, settings.device)
    prev_model = load_model_from_checkpoint(init_checkpoint, device=settings.device)
    drift = probe_drift(trainer.model, prev_model, probe, settings.device)
    realised_lr = [float(g["lr"]) for g in trainer.opt.param_groups]
    m = asdict(metrics)
    summary = {
        "updates": updates, "start_step": start_step, "end_step": int(trainer.step),
        "window_generations": gens, "window_rows": sampler.n, "holdout_rows": int(holdout["x"].shape[0]),
        "drawn_rows": sampler.drawn_rows, "sl_rows_drawn": sampler.realised_sl_rows,
        "sl_frac_realised": sampler.realised_sl_rows / max(1, sampler.drawn_rows),
        "optimizer_continued": continued, "realised_lr_param_groups": realised_lr,
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
    write_manifest(out_dir, stage="train", config={"settings": asdict(settings), "arch": arch},
                   inputs=inputs, outputs={"model": input_ref(model_path)},
                   counts={"updates": updates, "window_rows": sampler.n},
                   seeds={"train": seed, "base": settings.seed}, extra={"generation": generation, "summary": summary})
    return summary
