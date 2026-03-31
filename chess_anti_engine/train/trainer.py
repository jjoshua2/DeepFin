from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from zclip import ZClip

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def add_scalar(self, *args: Any, **kwargs: Any) -> None:
            pass

        def close(self) -> None:
            pass

from chess_anti_engine.encoding.lc0 import LC0_FULL
from chess_anti_engine.replay.buffer import ReplayBuffer
from chess_anti_engine.replay.dataset import collate, collate_arrays
from chess_anti_engine.replay.augment import maybe_mirror_batch_arrays, maybe_mirror_samples
from .muon import MuonWithAuxAdam
from .cosmos import COSMOS
from .cosmos_fast import COSMOSFast
from .losses import compute_loss


@dataclass
class TrainMetrics:
    loss: float
    policy_loss: float
    soft_policy_loss: float
    future_policy_loss: float
    wdl_loss: float
    sf_move_loss: float
    sf_move_acc: float
    sf_eval_loss: float
    categorical_loss: float
    volatility_loss: float
    sf_volatility_loss: float
    moves_left_loss: float
    train_time_s: float = 0.0
    opt_step_time_s: float = 0.0
    train_steps_done: int = 0
    train_samples_seen: int = 0


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str,
        lr: float,
        zclip_z_thresh: float = 2.5,
        zclip_alpha: float = 0.97,
        zclip_max_norm: float = 1.0,
        log_dir: Path | None = None,
        use_amp: bool = True,
        feature_dropout_p: float = 0.3,
        fdp_king_safety: float | None = None,
        fdp_pins: float | None = None,
        fdp_pawns: float | None = None,
        fdp_mobility: float | None = None,
        fdp_outposts: float | None = None,
        w_volatility: float = 0.05,
        accum_steps: int = 1,
        warmup_steps: int = 1500,
        warmup_lr_start: float | None = None,
        lr_eta_min: float = 1e-5,
        lr_T0: int = 5000,
        lr_T_mult: int = 2,
        use_compile: bool = False,
        optimizer: str = "nadamw",
        cosmos_rank: int = 64,
        cosmos_gamma: float = 0.2,
        swa_start: int = 0,
        swa_freq: int = 50,
        mirror_prob: float = 0.5,
        # Loss weights (all tunable for Ray Tune ablations)
        w_policy: float = 1.0,
        w_soft: float = 0.5,
        w_future: float = 0.15,
        w_wdl: float = 1.0,
        w_sf_move: float = 0.15,
        w_sf_eval: float = 0.15,
        w_categorical: float = 0.10,
        w_sf_volatility: float | None = None,
        w_moves_left: float = 0.02,
        w_sf_wdl: float = 0.0,
        sf_wdl_conf_power: float = 0.0,
        sf_wdl_draw_scale: float = 1.0,
        tb_log_interval: int = 10,
        prefetch_batches: bool = True,
    ):
        self.device = device
        self.model = model.to(device)

        optimizer = str(optimizer).lower()
        if optimizer == "muon":
            muon_decay_params = []
            muon_no_decay_params = []
            aux_decay_params = []
            aux_no_decay_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                is_no_decay = param.ndim <= 1 or name.endswith(".bias")
                is_muon_trunk = param.ndim >= 2 and (name == "embed.weight" or name.startswith("blocks."))
                if is_muon_trunk and not is_no_decay:
                    muon_decay_params.append(param)
                elif is_muon_trunk:
                    muon_no_decay_params.append(param)
                elif is_no_decay:
                    aux_no_decay_params.append(param)
                else:
                    aux_decay_params.append(param)

            # Muon is typically run with a larger LR on hidden weights than the
            # AdamW fallback uses on heads / norms / biases. Keep one Tune LR and
            # derive the trunk LR from it so search stays simple.
            muon_lr = float(lr) * 20.0
            param_groups = [
                {"params": muon_decay_params, "weight_decay": 1e-4, "use_muon": True, "lr": muon_lr},
                {"params": muon_no_decay_params, "weight_decay": 0.0, "use_muon": True, "lr": muon_lr},
                {"params": aux_decay_params, "weight_decay": 1e-4, "use_muon": False, "lr": float(lr)},
                {"params": aux_no_decay_params, "weight_decay": 0.0, "use_muon": False, "lr": float(lr)},
            ]
            self.opt = MuonWithAuxAdam(param_groups)
        elif optimizer == "cosmos_fast":
            cosmos_decay_params = []
            cosmos_no_decay_params = []
            aux_decay_params = []
            aux_no_decay_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                is_no_decay = param.ndim <= 1 or name.endswith(".bias")
                is_cosmos_hidden = param.ndim == 2 and name.startswith("blocks.")
                if is_cosmos_hidden and not is_no_decay:
                    cosmos_decay_params.append(param)
                elif is_cosmos_hidden:
                    cosmos_no_decay_params.append(param)
                elif is_no_decay:
                    aux_no_decay_params.append(param)
                else:
                    aux_decay_params.append(param)
            param_groups = [
                {"params": cosmos_decay_params, "weight_decay": 1e-4, "use_cosmos_fast": True},
                {"params": cosmos_no_decay_params, "weight_decay": 0.0, "use_cosmos_fast": True},
                {"params": aux_decay_params, "weight_decay": 1e-4, "use_cosmos_fast": False},
                {"params": aux_no_decay_params, "weight_decay": 0.0, "use_cosmos_fast": False},
            ]
            self.opt = COSMOSFast(
                param_groups,
                lr=lr,
                weight_decay=1e-4,
                rank=int(cosmos_rank),
                gamma=float(cosmos_gamma),
            )
        else:
            # Selective weight decay: apply only to non-bias, non-LayerNorm parameters.
            decay_params = []
            no_decay_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            param_groups = [
                {"params": decay_params, "weight_decay": 1e-4},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]

        if optimizer == "nadamw":
            # NAdam with decoupled weight decay (spec: β1=0.9, β2=0.98, ε=1e-7).
            # PyTorch NAdam supports decoupled_weight_decay since 2.x; param-group
            # weight_decay values are applied per-group.
            self.opt = torch.optim.NAdam(
                param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-7,
                decoupled_weight_decay=True,
            )
        elif optimizer == "adamw":
            self.opt = torch.optim.AdamW(param_groups, lr=lr)
        elif optimizer == "muon":
            pass
        elif optimizer == "cosmos":
            self.opt = COSMOS(param_groups, lr=lr, weight_decay=1e-4)
        elif optimizer == "cosmos_fast":
            pass
        elif optimizer == "soap":
            # SOAP: Shampoo-like second-order optimizer. Prefer a local
            # `soap.py`; otherwise fall back to pytorch-optimizer's SOAP.
            try:
                from soap import SOAP  # type: ignore[import]
            except ImportError as exc:
                try:
                    from pytorch_optimizer import SOAP  # type: ignore[import]
                except ImportError:
                    raise ImportError(
                        "SOAP optimizer requires either a local `soap.py` module "
                        "or the `pytorch-optimizer` package. "
                        "Install with: pip install pytorch-optimizer"
                    ) from exc
            try:
                self.opt = SOAP(param_groups, lr=lr)
            except TypeError:
                self.opt = SOAP(self.model.parameters(), lr=lr)
        else:
            raise ValueError(
                f"Unknown optimizer {optimizer!r}. Supported: nadamw, adamw, muon, cosmos, cosmos_fast, soap"
            )
        self.zclip = ZClip(mode="zscore", alpha=float(zclip_alpha), z_thresh=float(zclip_z_thresh), max_grad_norm=float(zclip_max_norm), warmup_steps=25)
        self.writer = SummaryWriter(log_dir=str(log_dir or "tb"))
        self.step = 0
        self._tb_log_interval = max(1, int(tb_log_interval))
        self._prefetch_batches = bool(prefetch_batches)

        self.use_amp = bool(use_amp)
        self._amp_dtype = torch.bfloat16 if device.startswith("cuda") else None
        # BF16 typically does not need GradScaler; keep scaler only for fp16 if added later.
        self._scaler = None

        self.feature_dropout_p = float(feature_dropout_p)
        self._base_input_planes = int(LC0_FULL.num_planes)
        # Per-group dropout: (start_offset_from_base, num_planes, dropout_prob)
        # Groups: king_safety(10), pins(6), pawns(8), mobility(6), outposts(4)
        _fdp = float(feature_dropout_p)
        self._feature_group_dropout = [
            (0, 10, float(fdp_king_safety) if fdp_king_safety is not None else _fdp),
            (10, 6, float(fdp_pins) if fdp_pins is not None else _fdp),
            (16, 8, float(fdp_pawns) if fdp_pawns is not None else _fdp),
            (24, 6, float(fdp_mobility) if fdp_mobility is not None else _fdp),
            (30, 4, float(fdp_outposts) if fdp_outposts is not None else _fdp),
        ]
        self.w_volatility = float(w_volatility)
        self.w_policy = float(w_policy)
        self.w_soft = float(w_soft)
        self.w_future = float(w_future)
        self.w_wdl = float(w_wdl)
        self.w_sf_move = float(w_sf_move)
        self.w_sf_eval = float(w_sf_eval)
        self.w_categorical = float(w_categorical)
        self.w_sf_volatility = float(w_sf_volatility) if w_sf_volatility is not None else float(w_volatility)
        self.w_moves_left = float(w_moves_left)
        self.w_sf_wdl = float(w_sf_wdl)
        self.sf_wdl_conf_power = float(sf_wdl_conf_power)
        self.sf_wdl_draw_scale = float(sf_wdl_draw_scale)

        # Data augmentation: mirror positions left-right (files) with given probability.
        self.mirror_prob = float(mirror_prob)

        # Optional torch.compile for training throughput.
        if use_compile and device.startswith("cuda"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                pass  # torch.compile may not be available on all platforms

        # Gradient accumulation
        self.accum_steps = max(1, int(accum_steps))

        # LR schedule: linear warmup then cosine annealing with warm restarts
        self._peak_lr = float(lr)
        self._warmup_steps = int(warmup_steps)
        if warmup_lr_start is None:
            self._warmup_lr_start = max(0.0, float(lr_eta_min))
        else:
            self._warmup_lr_start = max(0.0, float(warmup_lr_start))
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt, T_0=int(lr_T0), T_mult=int(lr_T_mult), eta_min=float(lr_eta_min),
        )
        self._set_initial_lrs()

        # Stochastic Weight Averaging (SWA): maintain a running average of model
        # weights for smoother, more generalizable exported networks.
        self._swa_start = int(swa_start)
        self._swa_freq = max(1, int(swa_freq))
        self._swa_model: torch.optim.swa_utils.AveragedModel | None = None
        if self._swa_start > 0:
            self._swa_model = torch.optim.swa_utils.AveragedModel(self.model)

    def _should_log_step_scalars(self) -> bool:
        return (self.step % self._tb_log_interval) == 0

    def _sample_batch_tensors(
        self,
        buf: ReplayBuffer,
        *,
        batch_size: int,
        mirror_prob: float,
    ) -> dict[str, torch.Tensor]:
        if hasattr(buf, "sample_batch_arrays"):
            arrs = getattr(buf, "sample_batch_arrays")(batch_size)
            arrs = maybe_mirror_batch_arrays(arrs, rng=buf.rng, prob=mirror_prob)
            return collate_arrays(arrs, device=self.device)

        samples = buf.sample_batch(batch_size)
        samples = maybe_mirror_samples(samples, rng=buf.rng, prob=mirror_prob)
        return collate(samples, device=self.device)

    def _sample_batch_host(
        self,
        buf: ReplayBuffer,
        *,
        batch_size: int,
        mirror_prob: float,
    ) -> dict[str, np.ndarray] | list:
        if hasattr(buf, "sample_batch_arrays"):
            arrs = getattr(buf, "sample_batch_arrays")(batch_size)
            return maybe_mirror_batch_arrays(arrs, rng=buf.rng, prob=mirror_prob)

        samples = buf.sample_batch(batch_size)
        return maybe_mirror_samples(samples, rng=buf.rng, prob=mirror_prob)

    def _host_batch_to_tensors(self, batch: dict[str, np.ndarray] | list) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            return collate_arrays(batch, device=self.device)
        return collate(batch, device=self.device)

    def _iter_prefetched_batches(
        self,
        buf: ReplayBuffer,
        *,
        batch_size: int,
        mirror_prob: float,
        count: int,
    ):
        n = int(count)
        if n <= 0:
            return
        if not self._prefetch_batches or n == 1:
            for _ in range(n):
                host_batch = self._sample_batch_host(buf, batch_size=batch_size, mirror_prob=mirror_prob)
                yield self._host_batch_to_tensors(host_batch)
            return

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                self._sample_batch_host,
                buf,
                batch_size=batch_size,
                mirror_prob=mirror_prob,
            )
            for idx in range(n):
                host_batch = future.result()
                if idx + 1 < n:
                    future = pool.submit(
                        self._sample_batch_host,
                        buf,
                        batch_size=batch_size,
                        mirror_prob=mirror_prob,
                    )
                yield self._host_batch_to_tensors(host_batch)

    def _base_lrs(self) -> list[float]:
        base_lrs = list(getattr(self._scheduler, "base_lrs", []))
        if base_lrs:
            return [float(v) for v in base_lrs]
        return [float(pg.get("lr", self._peak_lr)) for pg in self.opt.param_groups]

    def _reference_lr_from_bases(self, base_lrs: list[float] | None = None) -> float:
        vals = [float(v) for v in (base_lrs if base_lrs is not None else self._base_lrs()) if float(v) > 0.0]
        if vals:
            return min(vals)
        return max(float(self._peak_lr), 1e-12)

    def _warmup_start_lr_for(self, base_lr: float) -> float:
        peak = max(float(self._peak_lr), 1e-12)
        return float(self._warmup_lr_start) * float(base_lr) / peak

    def _set_initial_lrs(self) -> None:
        if self._warmup_steps <= 0:
            return
        for pg, base_lr in zip(self.opt.param_groups, self._base_lrs(), strict=True):
            pg["lr"] = self._warmup_start_lr_for(float(base_lr))

    def _update_lr(self) -> None:
        """Apply linear warmup, then hand off to cosine schedule."""
        if self.step < self._warmup_steps:
            # Called after optimizer.step(); set the LR for the *next* training step.
            next_frac = min(1.0, float(self.step + 1) / max(1, self._warmup_steps))
            for pg, base_lr in zip(self.opt.param_groups, self._base_lrs(), strict=True):
                start_lr = self._warmup_start_lr_for(float(base_lr))
                pg["lr"] = start_lr + (float(base_lr) - start_lr) * next_frac
        else:
            self._scheduler.step(self.step - self._warmup_steps)

    def set_peak_lr(self, lr: float, *, rescale_current: bool = True) -> None:
        """Rebase LR schedule to a new peak while preserving schedule phase.

        PB2 mutates `lr` in the trial config. When a trial restores from another
        trial's checkpoint, optimizer/scheduler state is cloned too, so we need
        to explicitly rebind the schedule to the mutated peak LR.
        """
        new_peak = float(lr)
        if new_peak <= 0.0:
            return

        n_groups = len(self.opt.param_groups)
        old_bases = []
        if hasattr(self._scheduler, "base_lrs"):
            old_bases = [float(v) for v in getattr(self._scheduler, "base_lrs")]
        if not old_bases:
            old_bases = [float(self._peak_lr)] * n_groups
        if len(old_bases) < n_groups:
            old_bases.extend([old_bases[-1]] * (n_groups - len(old_bases)))
        old_bases = old_bases[:n_groups]

        ref_old = max(float(self._peak_lr), self._reference_lr_from_bases(old_bases))
        scale = new_peak / ref_old

        new_bases: list[float] = []
        for ob in old_bases:
            if ob > 0.0:
                new_bases.append(ob * scale)
            else:
                new_bases.append(new_peak)

        self._peak_lr = new_peak

        # Keep scheduler phase but rebase amplitude.
        if hasattr(self._scheduler, "base_lrs"):
            self._scheduler.base_lrs = list(new_bases)

        # Keep the scheduler's cached current LR consistent with the rebase.
        if hasattr(self._scheduler, "_last_lr"):
            last_lrs = getattr(self._scheduler, "_last_lr")
            if isinstance(last_lrs, list) and last_lrs:
                self._scheduler._last_lr = [float(v) * scale for v in last_lrs]

        # Keep optimizer param-group metadata aligned.
        for i, pg in enumerate(self.opt.param_groups):
            ob = old_bases[i] if i < len(old_bases) else ref_old
            nb = new_bases[i] if i < len(new_bases) else new_peak
            if "initial_lr" in pg:
                if ob > 0.0:
                    pg["initial_lr"] = float(pg["initial_lr"]) * (nb / ob)
                else:
                    pg["initial_lr"] = nb

        if not rescale_current:
            return

        # Rebase currently active optimizer LR so training continues at same phase.
        for i, pg in enumerate(self.opt.param_groups):
            ob = old_bases[i] if i < len(old_bases) else ref_old
            nb = new_bases[i] if i < len(new_bases) else new_peak
            if ob > 0.0:
                pg["lr"] = float(pg.get("lr", 0.0)) * (nb / ob)
            else:
                if self.step < self._warmup_steps:
                    frac = self.step / max(1, self._warmup_steps)
                    warm_start = self._warmup_start_lr_for(nb)
                    pg["lr"] = warm_start + (nb - warm_start) * frac
                else:
                    last_lrs = getattr(self._scheduler, "_last_lr", None)
                    if isinstance(last_lrs, list) and i < len(last_lrs):
                        pg["lr"] = float(last_lrs[i])
                    else:
                        pg["lr"] = nb

    @torch.no_grad()
    def _compute_metrics(self, *, buf: ReplayBuffer, batch_size: int, steps: int, tag: str) -> TrainMetrics:
        loss_sum = 0.0
        pol_sum = 0.0
        soft_sum = 0.0
        fut_sum = 0.0
        wdl_sum = 0.0
        sf_move_sum = 0.0
        sf_move_acc_num = 0.0
        sf_move_acc_den = 0.0
        sf_sum = 0.0
        cat_sum = 0.0
        vol_sum = 0.0
        sf_vol_sum = 0.0
        ml_sum = 0.0

        mirror_p = self.mirror_prob if str(tag).startswith("train") else 0.0

        for batch in self._iter_prefetched_batches(
            buf,
            batch_size=batch_size,
            mirror_prob=mirror_p,
            count=int(steps),
        ):

            if self.use_amp and self.device.startswith("cuda"):
                with torch.amp.autocast("cuda", dtype=self._amp_dtype):
                    out = self.model(batch["x"])
                    losses = compute_loss(out, batch, w_policy=self.w_policy, w_soft=self.w_soft, w_future=self.w_future, w_wdl=self.w_wdl, w_sf_move=self.w_sf_move, w_sf_eval=self.w_sf_eval, w_categorical=self.w_categorical, w_volatility=self.w_volatility, w_sf_volatility=self.w_sf_volatility, w_moves_left=self.w_moves_left, w_sf_wdl=self.w_sf_wdl, sf_wdl_conf_power=self.sf_wdl_conf_power, sf_wdl_draw_scale=self.sf_wdl_draw_scale)
                    loss = losses["total"]
            else:
                out = self.model(batch["x"])
                losses = compute_loss(out, batch, w_policy=self.w_policy, w_soft=self.w_soft, w_future=self.w_future, w_wdl=self.w_wdl, w_sf_move=self.w_sf_move, w_sf_eval=self.w_sf_eval, w_categorical=self.w_categorical, w_volatility=self.w_volatility, w_sf_volatility=self.w_sf_volatility, w_moves_left=self.w_moves_left, w_sf_wdl=self.w_sf_wdl, sf_wdl_conf_power=self.sf_wdl_conf_power, sf_wdl_draw_scale=self.sf_wdl_draw_scale)
                loss = losses["total"]

            loss_sum += float(loss.item())
            pol_sum += float(losses["policy_ce"].detach().item())

            sp = losses.get("soft_policy_ce", None)
            fp = losses.get("future_policy_ce", None)
            cat = losses.get("categorical_ce", None)
            vol = losses.get("volatility", None)
            sf_vol = losses.get("sf_volatility", None)

            soft_sum += float(sp.detach().item()) if isinstance(sp, torch.Tensor) else float(sp or 0.0)
            fut_sum += float(fp.detach().item()) if isinstance(fp, torch.Tensor) else float(fp or 0.0)
            wdl_sum += float(losses["wdl_ce"].detach().item())
            sf_move_sum += float(losses["sf_move_ce"].detach().item())
            sf_sum += float(losses["sf_eval_ce"].detach().item())

            sf_mask = batch.get("has_sf_move")
            if sf_mask is not None:
                sf_mask = sf_mask.to(torch.float32)
            else:
                sf_mask = torch.zeros((batch["x"].shape[0],), device=batch["x"].device)

            sf_logits = out.get("policy_sf")
            if sf_logits is not None:
                pred = torch.argmax(sf_logits.detach(), dim=-1)
                correct = (pred == batch["sf_move_index"]).to(torch.float32)
                sf_move_acc_num += float((correct * sf_mask).sum().item())
                sf_move_acc_den += float(sf_mask.sum().item())

            cat_sum += float(cat.detach().item()) if isinstance(cat, torch.Tensor) else float(cat or 0.0)
            vol_sum += float(vol.detach().item()) if isinstance(vol, torch.Tensor) else float(vol or 0.0)
            sf_vol_sum += float(sf_vol.detach().item()) if isinstance(sf_vol, torch.Tensor) else float(sf_vol or 0.0)
            ml_sum += float(losses["moves_left"].detach().item())

        n = float(max(1, steps))

        # Log averaged metrics once per eval call (not per-batch, which would
        # overwrite the same self.step and keep only the last batch's value).
        self.writer.add_scalar(f"{tag}/loss", loss_sum / n, self.step)
        self.writer.add_scalar(f"{tag}/policy_loss", pol_sum / n, self.step)
        self.writer.add_scalar(f"{tag}/soft_policy_loss", soft_sum / n, self.step)
        self.writer.add_scalar(f"{tag}/future_policy_loss", fut_sum / n, self.step)
        self.writer.add_scalar(f"{tag}/wdl_loss", wdl_sum / n, self.step)
        self.writer.add_scalar(f"{tag}/sf_move_loss", sf_move_sum / n, self.step)
        self.writer.add_scalar(f"{tag}/sf_move_acc", float(sf_move_acc_num / max(1.0, sf_move_acc_den)), self.step)
        self.writer.add_scalar(f"{tag}/sf_eval_loss", sf_sum / n, self.step)
        self.writer.add_scalar(f"{tag}/categorical_loss", cat_sum / n, self.step)
        self.writer.add_scalar(f"{tag}/volatility_loss", vol_sum / n, self.step)
        self.writer.add_scalar(f"{tag}/sf_volatility_loss", sf_vol_sum / n, self.step)
        self.writer.add_scalar(f"{tag}/moves_left_loss", ml_sum / n, self.step)

        return TrainMetrics(
            loss=loss_sum / n,
            policy_loss=pol_sum / n,
            soft_policy_loss=soft_sum / n,
            future_policy_loss=fut_sum / n,
            wdl_loss=wdl_sum / n,
            sf_move_loss=sf_move_sum / n,
            sf_move_acc=float(sf_move_acc_num / max(1.0, sf_move_acc_den)),
            sf_eval_loss=sf_sum / n,
            categorical_loss=cat_sum / n,
            volatility_loss=vol_sum / n,
            sf_volatility_loss=sf_vol_sum / n,
            moves_left_loss=ml_sum / n,
            train_time_s=0.0,
            opt_step_time_s=0.0,
            train_steps_done=0,
            train_samples_seen=0,
        )

    def train_steps(self, buf: ReplayBuffer, *, batch_size: int, steps: int) -> TrainMetrics:
        self.model.train()
        train_wall_start = time.perf_counter()

        # Accumulators — collect metrics from the actual training batches to avoid
        # a redundant second forward pass through the buffer.
        loss_sum = 0.0
        pol_sum = 0.0
        soft_sum = 0.0
        fut_sum = 0.0
        wdl_sum = 0.0
        sf_move_sum = 0.0
        sf_move_acc_num = 0.0
        sf_move_acc_den = 0.0
        sf_sum = 0.0
        cat_sum = 0.0
        vol_sum = 0.0
        sf_vol_sum = 0.0
        ml_sum = 0.0
        n_micro = 0
        opt_step_time_s = 0.0
        train_steps_done = 0

        _log = logging.getLogger(__name__)

        for _ in range(int(steps)):
          for _attempt in range(3):
            try:
                self.opt.zero_grad(set_to_none=True)

                step_loss = 0.0
                step_pol = 0.0
                step_soft = 0.0
                step_fut = 0.0
                step_wdl = 0.0
                step_sf_move = 0.0
                step_sf = 0.0
                step_cat = 0.0
                step_vol = 0.0
                step_sf_vol = 0.0
                step_ml = 0.0
                step_sf_acc_num = 0.0
                step_sf_acc_den = 0.0
                step_n_micro = 0

                for batch in self._iter_prefetched_batches(
                    buf,
                    batch_size=batch_size,
                    mirror_prob=self.mirror_prob,
                    count=self.accum_steps,
                ):

                    # Per-group feature dropout: independently zero each classical feature group.
                    base = int(self._base_input_planes)
                    x = batch["x"]
                    if x.shape[1] > base:
                        for g_off, g_len, g_p in self._feature_group_dropout:
                            if g_p > 0.0:
                                drop = (torch.rand((x.shape[0], 1, 1, 1), device=x.device) < g_p).to(x.dtype)
                                x[:, base + g_off : base + g_off + g_len, :, :] *= (1.0 - drop)

                    if self.use_amp and self.device.startswith("cuda"):
                        with torch.amp.autocast("cuda", dtype=self._amp_dtype):
                            out = self.model(batch["x"])
                            losses = compute_loss(out, batch, w_policy=self.w_policy, w_soft=self.w_soft, w_future=self.w_future, w_wdl=self.w_wdl, w_sf_move=self.w_sf_move, w_sf_eval=self.w_sf_eval, w_categorical=self.w_categorical, w_volatility=self.w_volatility, w_sf_volatility=self.w_sf_volatility, w_moves_left=self.w_moves_left, w_sf_wdl=self.w_sf_wdl, sf_wdl_conf_power=self.sf_wdl_conf_power, sf_wdl_draw_scale=self.sf_wdl_draw_scale)
                            loss = losses["total"] / self.accum_steps
                        loss.backward()
                    else:
                        out = self.model(batch["x"])
                        losses = compute_loss(out, batch, w_policy=self.w_policy, w_soft=self.w_soft, w_future=self.w_future, w_wdl=self.w_wdl, w_sf_move=self.w_sf_move, w_sf_eval=self.w_sf_eval, w_categorical=self.w_categorical, w_volatility=self.w_volatility, w_sf_volatility=self.w_sf_volatility, w_moves_left=self.w_moves_left, w_sf_wdl=self.w_sf_wdl, sf_wdl_conf_power=self.sf_wdl_conf_power, sf_wdl_draw_scale=self.sf_wdl_draw_scale)
                        loss = losses["total"] / self.accum_steps
                        loss.backward()

                    # Accumulate metric scalars inline (no graph kept — just .item() values).
                    sp = losses.get("soft_policy_ce", None)
                    fp = losses.get("future_policy_ce", None)
                    cat = losses.get("categorical_ce", None)
                    vol = losses.get("volatility", None)
                    sf_vol = losses.get("sf_volatility", None)
                    step_loss += float(loss.item() * self.accum_steps)
                    step_pol += float(losses["policy_ce"].detach().item())
                    step_soft += float(sp.detach().item()) if isinstance(sp, torch.Tensor) else float(sp or 0.0)
                    step_fut += float(fp.detach().item()) if isinstance(fp, torch.Tensor) else float(fp or 0.0)
                    step_wdl += float(losses["wdl_ce"].detach().item())
                    step_sf_move += float(losses["sf_move_ce"].detach().item())
                    step_sf += float(losses["sf_eval_ce"].detach().item())
                    step_cat += float(cat.detach().item()) if isinstance(cat, torch.Tensor) else float(cat or 0.0)
                    step_vol += float(vol.detach().item()) if isinstance(vol, torch.Tensor) else float(vol or 0.0)
                    step_sf_vol += float(sf_vol.detach().item()) if isinstance(sf_vol, torch.Tensor) else float(sf_vol or 0.0)
                    step_ml += float(losses["moves_left"].detach().item())

                    with torch.no_grad():
                        sf_mask = batch.get("has_sf_move")
                        sf_logits = out.get("policy_sf")
                        if sf_mask is not None and sf_logits is not None:
                            sf_mask_f = sf_mask.to(torch.float32)
                            pred = torch.argmax(sf_logits.detach(), dim=-1)
                            correct = (pred == batch["sf_move_index"]).to(torch.float32)
                            step_sf_acc_num += float((correct * sf_mask_f).sum().item())
                            step_sf_acc_den += float(sf_mask_f.sum().item())

                    step_n_micro += 1

                grad_norm = self.zclip.step(self.model)
                if self._should_log_step_scalars():
                    self.writer.add_scalar("train/grad_norm", float(grad_norm) if grad_norm is not None else 0.0, self.step)
                opt_step_start = time.perf_counter()
                self.opt.step()
                opt_step_time_s += time.perf_counter() - opt_step_start
                self._update_lr()

            except RuntimeError as exc:
                if "CUDA" not in str(exc) or _attempt >= 2:
                    raise
                _log.warning("Transient CUDA error (attempt %d/3), retrying: %s", _attempt + 1, exc)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                time.sleep(0.5 * (_attempt + 1))
                self.opt.zero_grad(set_to_none=True)
                continue

            # Success — commit metrics from this step.
            loss_sum += step_loss
            pol_sum += step_pol
            soft_sum += step_soft
            fut_sum += step_fut
            wdl_sum += step_wdl
            sf_move_sum += step_sf_move
            sf_sum += step_sf
            cat_sum += step_cat
            vol_sum += step_vol
            sf_vol_sum += step_sf_vol
            ml_sum += step_ml
            sf_move_acc_num += step_sf_acc_num
            sf_move_acc_den += step_sf_acc_den
            n_micro += step_n_micro

            # SWA: update averaged model after swa_start steps, every swa_freq steps.
            if (
                self._swa_model is not None
                and self.step >= self._swa_start
                and self.step % self._swa_freq == 0
            ):
                self._swa_model.update_parameters(self.model)

            # Step-aligned train logging (log unscaled loss from last micro-batch)
            if self._should_log_step_scalars():
                self.writer.add_scalar("train/loss", float(step_loss / max(1, step_n_micro) if step_n_micro else 0.0), self.step)
                self.writer.add_scalar("train/lr", self.opt.param_groups[0]["lr"], self.step)
            self.step += 1
            train_steps_done += 1
            break

        train_time_s = time.perf_counter() - train_wall_start
        train_samples_seen = int(n_micro * batch_size)
        train_steps_per_s = float(train_steps_done / max(train_time_s, 1e-9))
        train_samples_per_s = float(train_samples_seen / max(train_time_s, 1e-9))
        opt_steps_per_s = float(train_steps_done / max(opt_step_time_s, 1e-9)) if opt_step_time_s > 0.0 else 0.0
        n = float(max(1, n_micro))
        metrics = TrainMetrics(
            loss=loss_sum / n,
            policy_loss=pol_sum / n,
            soft_policy_loss=soft_sum / n,
            future_policy_loss=fut_sum / n,
            wdl_loss=wdl_sum / n,
            sf_move_loss=sf_move_sum / n,
            sf_move_acc=float(sf_move_acc_num / max(1.0, sf_move_acc_den)),
            sf_eval_loss=sf_sum / n,
            categorical_loss=cat_sum / n,
            volatility_loss=vol_sum / n,
            sf_volatility_loss=sf_vol_sum / n,
            moves_left_loss=ml_sum / n,
            train_time_s=float(train_time_s),
            opt_step_time_s=float(opt_step_time_s),
            train_steps_done=int(train_steps_done),
            train_samples_seen=int(train_samples_seen),
        )
        # Log aggregated metrics once (mirrors what _compute_metrics wrote under "train_avg/")
        self.writer.add_scalar("train_avg/loss", metrics.loss, self.step)
        self.writer.add_scalar("train_avg/policy_loss", metrics.policy_loss, self.step)
        self.writer.add_scalar("train_avg/soft_policy_loss", metrics.soft_policy_loss, self.step)
        self.writer.add_scalar("train_avg/future_policy_loss", metrics.future_policy_loss, self.step)
        self.writer.add_scalar("train_avg/wdl_loss", metrics.wdl_loss, self.step)
        self.writer.add_scalar("train_avg/sf_move_loss", metrics.sf_move_loss, self.step)
        self.writer.add_scalar("train_avg/sf_move_acc", metrics.sf_move_acc, self.step)
        self.writer.add_scalar("train_avg/sf_eval_loss", metrics.sf_eval_loss, self.step)
        self.writer.add_scalar("train_avg/categorical_loss", metrics.categorical_loss, self.step)
        self.writer.add_scalar("train_avg/volatility_loss", metrics.volatility_loss, self.step)
        self.writer.add_scalar("train_avg/sf_volatility_loss", metrics.sf_volatility_loss, self.step)
        self.writer.add_scalar("train_avg/moves_left_loss", metrics.moves_left_loss, self.step)
        self.writer.add_scalar("train_avg/time_s", float(train_time_s), self.step)
        self.writer.add_scalar("train_avg/opt_step_time_s", float(opt_step_time_s), self.step)
        self.writer.add_scalar("train_avg/steps_done", float(train_steps_done), self.step)
        self.writer.add_scalar("train_avg/samples_seen", float(train_samples_seen), self.step)
        self.writer.add_scalar("train_avg/steps_per_s", float(train_steps_per_s), self.step)
        self.writer.add_scalar("train_avg/samples_per_s", float(train_samples_per_s), self.step)
        self.writer.add_scalar("train_avg/opt_steps_per_s", float(opt_steps_per_s), self.step)
        return metrics

    @torch.no_grad()
    def eval_steps(self, buf: ReplayBuffer, *, batch_size: int, steps: int) -> TrainMetrics:
        self.model.eval()
        return self._compute_metrics(buf=buf, batch_size=batch_size, steps=steps, tag="eval")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "step": self.step,
            "peak_lr": float(self._peak_lr),
        }
        if self._swa_model is not None:
            state["swa_model"] = self._swa_model.state_dict()
        torch.save(state, str(path))

    def load(self, path: Path) -> None:
        from chess_anti_engine.model import load_state_dict_tolerant

        ckpt = torch.load(str(path), map_location=self.device)
        load_state_dict_tolerant(self.model, ckpt["model"], label="resume")
        fresh_opt_state = self.opt.state_dict()
        try:
            self.opt.load_state_dict(ckpt["opt"])
        except (ValueError, KeyError, RuntimeError) as exc:
            logging.getLogger(__name__).warning(
                "Optimizer state incompatible with new model layout, "
                "reinitialising optimizer: %s", exc,
            )
            self.opt.load_state_dict(fresh_opt_state)
        if "scheduler" in ckpt:
            self._scheduler.load_state_dict(ckpt["scheduler"])
        if "peak_lr" in ckpt:
            self._peak_lr = float(ckpt["peak_lr"])
        else:
            self._peak_lr = self._reference_lr_from_bases()
        if "swa_model" in ckpt and self._swa_model is not None:
            try:
                self._swa_model.load_state_dict(ckpt["swa_model"])
            except (RuntimeError, KeyError) as exc:
                logging.getLogger(__name__).warning(
                    "SWA model state incompatible, reinitialising: %s", exc,
                )
        self.step = int(ckpt.get("step", 0))

    def export_swa(self, path: Path, dataloader: object = None) -> None:
        """Export the SWA-averaged model weights.

        If a dataloader is provided, batch normalization statistics are updated
        using ``torch.optim.swa_utils.update_bn``.

        This is written atomically (temp file + os.replace) to avoid race conditions
        with workers downloading the file while the learner is writing it.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            if self._swa_model is None:
                # No SWA model — just save the regular model.
                torch.save({"model": self.model.state_dict()}, str(tmp))
            else:
                if dataloader is not None:
                    torch.optim.swa_utils.update_bn(
                        dataloader,
                        self._swa_model,
                        device=torch.device(self.device),
                    )
                torch.save({"model": self._swa_model.module.state_dict()}, str(tmp))

            os.replace(str(tmp), str(path))
        finally:
            # Best-effort cleanup if torch.save failed before replace.
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
