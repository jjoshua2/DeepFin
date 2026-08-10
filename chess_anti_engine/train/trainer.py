from __future__ import annotations

import dataclasses
import hashlib
import logging
import math
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from zclip import ZClip

from chess_anti_engine.utils.amp import inference_autocast
from chess_anti_engine.utils.atomic import atomic_write

try:
    from torch.utils.tensorboard import SummaryWriter as _ImportedSummaryWriter

    _SummaryWriter: Any = _ImportedSummaryWriter
except ImportError:  # pragma: no cover
  # ⚑ ImportError ONLY, and the narrowing is the point. This used to be
  # `except Exception`, which caught not just the intended "tensorboard is not
  # installed" but ANY failure inside a tensorboard that IS installed -- a
  # version incompatibility, a broken transitive dep, a protobuf mismatch.
  # The fallback's methods are `pass`, so the run then produced no event files
  # and said nothing, and every `self.writer.add_scalar(...)` in this file was
  # a silent no-op. This repo has a documented incident where flat live
  # progress signals let a real degradation go unread; a metrics writer that
  # can quietly become a no-op is the same family. A genuine breakage now
  # propagates at import instead of being absorbed as "not installed", and the
  # intended case still works and now says so.
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "tensorboard is not installed: SummaryWriter falls back to a no-op and "
        "NO scalar metrics will be recorded for this run (install the `train` "
        "extra to restore them)",
    )

    class _FallbackSummaryWriter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:  # skylos: ignore (stub signature parity)
            pass

        def add_scalar(self, *_args: Any, **_kwargs: Any) -> None:  # skylos: ignore (stub signature parity)
            pass

        def close(self) -> None:
            pass

    _SummaryWriter = _FallbackSummaryWriter

from chess_anti_engine.encoding.lc0 import (
    LC0_FULL,
    normalize_lc0_history_encoding,
    uses_lc0_root_history,
    uses_lc0_root_legacy_meta,
)
from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig
from chess_anti_engine.train.eval_ruler import call_closure, eval_ruler_id
from chess_anti_engine.train.target_builder import (
    DEFAULT_CATEGORICAL_BINS,
    CategoricalTargetParams,
    SfRebuildCoverage,
    SfTargetParams,
    rebuild_categorical_target_in_arrays,
    rebuild_sf_targets_in_arrays,
)
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.moves import torch_maps
from chess_anti_engine.replay.augment import (
    maybe_mirror_batch_arrays,
    maybe_mirror_samples,
)
from chess_anti_engine.replay.buffer import ReplayBuffer, ReplaySample
from chess_anti_engine.replay.dataset import collate, collate_arrays
from chess_anti_engine.replay.shard import (
    INPUT_HISTORY_ENCODING_ARRAY_KEY,
    SF_EVAL_PV_CHECKED_FIELD,
    SF_EVAL_PV_ORPHAN_FIELD,
    sf_eval_pv_orphan_flags,
)

from .aurora import AuroraWithAuxAdam
from .compile_probe import CompileProbe, apply_compile
from .losses import (
    align_policy_target,
    apply_policy_mask_to_logits,
    compute_loss,
    policy_target_temp_active,
    retemper_main_policy_target,
    wdl_brier_ece_from_stats,
    wdl_calibration_stats,
)
from .muon import MuonWithAuxAdam
from .soda import SODA_STEP_KEY, SODAWeightDecayWrapper, mark_soda_weight_decay_groups

SummaryWriter = _SummaryWriter  # skylos: ignore (used via runtime fallback)


class _TrainBatchIterator:
    """Keep replay prefetch alive across optimizer steps and CUDA retries."""

    def __init__(
        self,
        factory: Callable[[int], Iterator[dict[str, torch.Tensor]]],
        count: int,
    ) -> None:
        self._factory = factory
        self._current = factory(max(0, int(count)))
        self._extra_count = 0
        self._closed = False
        self.consumed = 0

    def __iter__(self) -> _TrainBatchIterator:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self._closed:
            raise StopIteration
        while True:
            try:
                batch = next(self._current)
            except StopIteration:
                if self._extra_count <= 0:
                    raise
                count = self._extra_count
                self._extra_count = 0
                self._current = self._factory(count)
                continue
            self.consumed += 1
            return batch

    def add_retry_batches(self, count: int) -> None:
        if self._closed:
            raise RuntimeError("cannot extend a closed training batch iterator")
        self._extra_count += max(0, int(count))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._current, "close", None)
        if callable(close):
            close()

_LC0_HISTORY_STEPS = LC0_FULL.history_len
_LC0_PIECE_PLANES = LC0_FULL.piece_planes_per_history
_INPUT_HISTORY_SELECTED_KEY = "_input_history_encoding_selected"


class _SqrtReleaseLRScheduler:
    """Flat LR cycle with a configurable WSD release tail.

    This is a WSD-style release schedule for indefinite training: hold each
    param group's base LR for most of the cycle, then decay to ``min_scale`` in
    the final tail before restarting. ``cycle_steps <= 0`` means callers supply
    the effective cycle length at ``step()`` time. ``release_shape`` controls
    only the tail curve; ``sqrt`` keeps the original behavior.
    """

    def __init__(
        self,
        optimizer: Any,
        *,
        cycle_steps: int,
        release_start_frac: float,
        min_scale: float,
        release_shape: str = "sqrt",
    ) -> None:
        self.optimizer = optimizer
        self.cycle_steps = int(cycle_steps)
        self.release_start_frac = min(1.0, max(0.0, float(release_start_frac)))
        self.min_scale = min(1.0, max(0.0, float(min_scale)))
        self.release_shape = str(release_shape).lower()
        if self.release_shape not in ("sqrt", "cosine"):
            raise ValueError(f"Unsupported lr_release_shape {release_shape!r}; expected sqrt or cosine")
        self.base_lrs = [float(pg.get("lr", 0.0)) for pg in optimizer.param_groups]
        self._last_lr = list(self.base_lrs)
        self.last_epoch = -1

    def _scale_from_progress(self, progress: float) -> float:
        if self.release_shape == "cosine":
            tail_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            tail_scale = 1.0 - math.sqrt(progress)
        return self.min_scale + (1.0 - self.min_scale) * tail_scale

    def _scale_for_epoch(self, epoch: float, *, cycle_steps: int | None = None) -> float:
        effective_cycle_steps = max(1, int(cycle_steps if cycle_steps is not None else self.cycle_steps))
        phase = float(epoch) % float(effective_cycle_steps)
        release_start = float(effective_cycle_steps) * self.release_start_frac
        if phase <= release_start:
            return 1.0
        release_len = max(1e-12, float(effective_cycle_steps) - release_start)
        progress = min(1.0, max(0.0, (phase - release_start) / release_len))
        return self._scale_from_progress(progress)

    def _scale_for_window_step(self, local_step: int, *, cycle_steps: int) -> float:
        effective_cycle_steps = max(1, int(cycle_steps))
        last_phase = float(max(0, effective_cycle_steps - 1))
        phase = min(last_phase, max(0.0, float(local_step)))
        if last_phase <= 0.0 or self.release_start_frac >= 1.0:
            return 1.0
        if phase >= last_phase:
            return self._scale_from_progress(1.0)
        release_start = float(effective_cycle_steps) * self.release_start_frac
        if phase <= release_start:
            return 1.0
        release_len = max(1e-12, last_phase - release_start)
        progress = min(1.0, max(0.0, (phase - release_start) / release_len))
        return self._scale_from_progress(progress)

    def step(self, epoch: float | None = None, *, cycle_steps: int | None = None) -> None:
        if epoch is None:
            epoch = float(self.last_epoch + 1)
        self.last_epoch = int(epoch)
        scale = self._scale_for_epoch(float(epoch), cycle_steps=cycle_steps)
        self._last_lr = [float(base_lr) * scale for base_lr in self.base_lrs]
        for pg, lr in zip(self.optimizer.param_groups, self._last_lr, strict=True):
            pg["lr"] = lr

    def step_window(self, local_step: int, *, cycle_steps: int) -> None:
        self.last_epoch = int(local_step)
        scale = self._scale_for_window_step(int(local_step), cycle_steps=cycle_steps)
        self._last_lr = [float(base_lr) * scale for base_lr in self.base_lrs]
        for pg, lr in zip(self.optimizer.param_groups, self._last_lr, strict=True):
            pg["lr"] = lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_lrs": list(self.base_lrs),
            "_last_lr": list(self._last_lr),
            "last_epoch": int(self.last_epoch),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        base_lrs = [float(v) for v in state_dict.get("base_lrs", self.base_lrs)]
        if len(base_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                f"loaded scheduler has {len(base_lrs)} param groups, "
                f"expected {len(self.optimizer.param_groups)}"
            )
        self.base_lrs = base_lrs
        last_lr = [float(v) for v in state_dict.get("_last_lr", base_lrs)]
        self._last_lr = last_lr if len(last_lr) == len(base_lrs) else list(base_lrs)
        self.last_epoch = int(state_dict.get("last_epoch", self.last_epoch))


def _metadata_strings(
    arrs: dict[str, np.ndarray],
    key: str,
    *,
    n: int | None = None,
) -> np.ndarray | None:
    value = arrs.get(key)
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    flat = array.reshape(-1).astype(str)
    if n is not None:
        if flat.size == 1:
            return np.full((n,), str(flat[0]), dtype=object)
        if flat.size != n:
            raise ValueError(f"replay metadata {key} has {flat.size} values for {n} rows")
    return flat.astype(object, copy=False)


def _single_metadata_string(arrs: dict[str, np.ndarray], key: str) -> str | None:
    flat = _metadata_strings(arrs, key)
    if flat is None:
        return None
    first = str(flat[0])
    if any(str(item) != first for item in flat[1:]):
        raise ValueError(f"mixed replay metadata {key}: {sorted({str(item) for item in flat})}")
    return first


def _legacy_x_to_synthetic_lc0_root(x: np.ndarray) -> np.ndarray:
    """Best-effort LC0-root history remap for legacy replay tensors.

    LOSSY, and not fixable in place: the root layout's plane 108 is the
    absolute side-to-move flag (1.0 when the root side to move is black),
    while the legacy layout has no absolute colour anywhere — its plane 101
    is a constant 1.0 for every position (``_write_metadata_planes``). Every
    converted row therefore reads as white-to-move. Callers must opt in via
    ``allow_lossy_legacy_remap`` (docs/rl_loop_audit.md M12).
    """
    src = np.asarray(x)
    out = np.array(src, copy=True, order="C")
    out[:, :LC0_FULL.num_planes, :, :] = 0
    half = _LC0_PIECE_PLANES // 2  # per-color split within a piece-plane slot
    rep_base = LC0_FULL.legacy_repetition_base
    ones_plane = LC0_FULL.ones_plane
    for hist_idx in range(_LC0_HISTORY_STEPS):
        legacy_start = hist_idx * _LC0_PIECE_PLANES
        root_start = hist_idx * (_LC0_PIECE_PLANES + 1)
        planes = src[:, legacy_start:legacy_start + _LC0_PIECE_PLANES, :, :]
        if hist_idx % 2 == 0:
            out[:, root_start:root_start + _LC0_PIECE_PLANES, :, :] = planes
        else:
            out[:, root_start:root_start + half, :, :] = planes[:, half:_LC0_PIECE_PLANES, ::-1, :]
            out[:, root_start + half:root_start + _LC0_PIECE_PLANES, :, :] = planes[:, 0:half, ::-1, :]
        rep_plane = rep_base + hist_idx
        if rep_plane < ones_plane:
            out[:, root_start + _LC0_PIECE_PLANES, :, :] = src[:, rep_plane, :, :]
    # Per-board metadata: legacy castling order is (K,Q) per side, root is (Q,K).
    meta = LC0_FULL.metadata_base
    root_meta = LC0_FULL.root_metadata_base
    out[:, root_meta + 0, :, :] = src[:, meta + 1, :, :]  # castle Q us
    out[:, root_meta + 1, :, :] = src[:, meta + 0, :, :]  # castle K us
    out[:, root_meta + 2, :, :] = src[:, meta + 3, :, :]  # castle Q them
    out[:, root_meta + 3, :, :] = src[:, meta + 2, :, :]  # castle K them
    out[:, root_meta + 4, :, :] = 0                        # side-to-move flag
    out[:, root_meta + 5, :, :] = src[:, meta + 6, :, :]  # rule50
    out[:, root_meta + 6, :, :] = 0                        # legacy movecount (zero)
    out[:, ones_plane, :, :] = src[:, ones_plane, :, :]  # all-ones bias
    return out


def _apply_lc0_root_legacy_meta(root_x: np.ndarray, legacy_x: np.ndarray) -> np.ndarray:
    root = np.array(root_x, copy=True, order="C")
    legacy = np.asarray(legacy_x)
    root_rule50 = LC0_FULL.root_metadata_base + 5
    root_movecount = LC0_FULL.root_metadata_base + 6
    legacy_rule50 = LC0_FULL.metadata_base + 6
    legacy_ep = LC0_FULL.metadata_base + 4
    root[:, root_rule50, :, :] = legacy[:, legacy_rule50, :, :].astype(root.dtype, copy=False)
    root[:, root_movecount, :, :] = legacy[:, legacy_ep, :, :].astype(root.dtype, copy=False)
    return root


def select_input_history_arrays(
    arrs: dict[str, np.ndarray],
    *,
    input_history_encoding: str,
    allow_lossy_legacy_remap: bool = False,
) -> dict[str, np.ndarray]:
    """Select the replay input tensor matching the model's configured history layout.

    Rows already stored in the target layout, and legacy rows that carry a
    recorded ``x_lc0_root`` tensor, convert losslessly. Legacy rows with
    neither can only go through :func:`_legacy_x_to_synthetic_lc0_root`, which
    cannot recover side-to-move — every such row would train as white-to-move.
    That case raises unless the caller passes ``allow_lossy_legacy_remap``,
    which offline tooling may do knowingly; the training path never does.
    """
    hist_enc = normalize_lc0_history_encoding(input_history_encoding)
    n = int(np.asarray(arrs["x"]).shape[0])
    stored_value = arrs.get(INPUT_HISTORY_ENCODING_ARRAY_KEY)
    if stored_value is not None:
        stored_array = np.asarray(stored_value)
        flat_stored = stored_array.reshape(-1)
        if flat_stored.size == 1 or (
            flat_stored.size == n
            and bool(np.all(flat_stored == flat_stored[0]))
        ):
            text = str(flat_stored[0]).strip()
            scalar_enc = normalize_lc0_history_encoding(text) if text else ""
            if not uses_lc0_root_history(hist_enc):
                if scalar_enc and uses_lc0_root_history(scalar_enc):
                    raise ValueError(
                        "replay x contains rows stored as LC0-root, "
                        f"cannot train {hist_enc!r} input"
                    )
                return arrs
            selected_enc = _single_metadata_string(arrs, _INPUT_HISTORY_SELECTED_KEY)
            if selected_enc is not None:
                if selected_enc != hist_enc:
                    raise ValueError(
                        f"replay input history already selected as {selected_enc!r}, "
                        f"cannot reselect as {hist_enc!r}"
                    )
                return arrs
            if scalar_enc == hist_enc:
                out = dict(arrs)
                out[_INPUT_HISTORY_SELECTED_KEY] = np.asarray(hist_enc)
                return out
            if scalar_enc and uses_lc0_root_history(scalar_enc):
                raise ValueError(f"replay x contains incompatible stored history encodings: {[scalar_enc]}")
    elif not uses_lc0_root_history(hist_enc):
        return arrs

    stored_raw = _metadata_strings(arrs, INPUT_HISTORY_ENCODING_ARRAY_KEY, n=n)
    stored_enc = np.full((n,), "", dtype=object)
    if stored_raw is not None:
        for i, raw in enumerate(stored_raw):
            text = str(raw).strip()
            stored_enc[i] = normalize_lc0_history_encoding(text) if text else ""
    if not uses_lc0_root_history(hist_enc):
        root_rows = np.array(
            [uses_lc0_root_history(str(enc)) if str(enc) else False for enc in stored_enc],
            dtype=bool,
        )
        if bool(np.any(root_rows)):
            raise ValueError(
                f"replay x contains {int(root_rows.sum())} rows stored as LC0-root, "
                f"cannot train {hist_enc!r} input"
            )
        return arrs
    selected_enc = _single_metadata_string(arrs, _INPUT_HISTORY_SELECTED_KEY)
    if selected_enc is not None:
        if selected_enc != hist_enc:
            raise ValueError(
                f"replay input history already selected as {selected_enc!r}, "
                f"cannot reselect as {hist_enc!r}"
            )
        return arrs
    already_root = stored_enc == hist_enc
    mismatched_root = np.array(
        [
            bool(enc) and enc != hist_enc and uses_lc0_root_history(str(enc))
            for enc in stored_enc
        ],
        dtype=bool,
    )
    if bool(np.any(mismatched_root)):
        bad = sorted({str(enc) for enc in stored_enc[mismatched_root]})
        raise ValueError(f"replay x contains incompatible stored history encodings: {bad}")
    if bool(np.all(already_root)):
        out = dict(arrs)
        out[_INPUT_HISTORY_SELECTED_KEY] = np.asarray(hist_enc)
        return out
    out = dict(arrs)
    legacy_x = np.asarray(out["x"])
    convert_rows = ~already_root
    use_recorded = np.zeros_like(convert_rows)
    if "x_lc0_root" in out:
        recorded = np.asarray(out["x_lc0_root"])
        has = np.asarray(out.get("has_x_lc0_root", np.ones((legacy_x.shape[0],), dtype=np.uint8))) != 0
        if recorded.shape == legacy_x.shape and has.shape == (legacy_x.shape[0],):
            use_recorded = has & convert_rows
    needs_synthetic = convert_rows & ~use_recorded
    n_synthetic = int(needs_synthetic.sum())
    if n_synthetic and not allow_lossy_legacy_remap:
        raise ValueError(
            f"{n_synthetic} of {n} replay rows are stored as legacy history with no "
            f"recorded x_lc0_root, so selecting {hist_enc!r} would synthesize them via "
            "a remap that CANNOT recover side-to-move: every converted row reads as "
            "white-to-move (docs/rl_loop_audit.md M12). Re-encode the shards for the "
            "current model (scripts/convert_shards_v2_threats.py) or drop them from "
            "the window; pass allow_lossy_legacy_remap=True only for offline "
            "diagnostics that accept a wrong POV plane."
        )
    root_x = np.array(legacy_x, copy=True, order="C")
    if n_synthetic:
        root_x[needs_synthetic] = _legacy_x_to_synthetic_lc0_root(legacy_x[needs_synthetic])
    if bool(np.any(use_recorded)):
        root_x[use_recorded] = np.asarray(out["x_lc0_root"])[use_recorded]
    if uses_lc0_root_legacy_meta(hist_enc) and bool(np.any(convert_rows)):
        patched = _apply_lc0_root_legacy_meta(root_x[convert_rows], legacy_x[convert_rows])
        root_x[convert_rows] = patched
    out["x"] = root_x
    out[_INPUT_HISTORY_SELECTED_KEY] = np.asarray(hist_enc)
    out[INPUT_HISTORY_ENCODING_ARRAY_KEY] = np.full((n,), hist_enc, dtype=object)
    return out


def select_input_history_samples(
    samples: list[ReplaySample],
    *,
    input_history_encoding: str,
    allow_lossy_legacy_remap: bool = False,
) -> list[ReplaySample]:
    """Select configured input planes for in-memory replay samples.

    ``allow_lossy_legacy_remap`` has the same meaning as in
    :func:`select_input_history_arrays`.
    """
    hist_enc = normalize_lc0_history_encoding(input_history_encoding)
    if not samples:
        return samples
    if not uses_lc0_root_history(hist_enc):
        root_samples = [
            sample
            for sample in samples
            if sample.input_history_encoding
            and uses_lc0_root_history(sample.input_history_encoding)
        ]
        if root_samples:
            raise ValueError(
                f"replay samples contain {len(root_samples)} rows stored as LC0-root, "
                f"cannot train {hist_enc!r} input"
            )
        return samples

    legacy_x = np.stack([np.asarray(s.x) for s in samples], axis=0)
    arrs: dict[str, np.ndarray] = {
        "x": legacy_x,
        INPUT_HISTORY_ENCODING_ARRAY_KEY: np.asarray(
            [
                normalize_lc0_history_encoding(sample.input_history_encoding)
                if sample.input_history_encoding
                else ""
                for sample in samples
            ],
            dtype=object,
        ),
    }
    if any(s.x_lc0_root is not None for s in samples):
        recorded = np.zeros_like(legacy_x)
        has = np.zeros((len(samples),), dtype=np.uint8)
        for i, sample in enumerate(samples):
            if sample.x_lc0_root is not None:
                recorded[i] = np.asarray(sample.x_lc0_root)
                has[i] = 1
        arrs["x_lc0_root"] = recorded
        arrs["has_x_lc0_root"] = has

    selected = select_input_history_arrays(
        arrs,
        input_history_encoding=hist_enc,
        allow_lossy_legacy_remap=allow_lossy_legacy_remap,
    )["x"]
    return [
        dataclasses.replace(sample, x=selected[i], input_history_encoding=hist_enc)
        for i, sample in enumerate(samples)
    ]


class _ChainedOptimizer(torch.optim.Optimizer):
    """Expose multiple optimizers as one optimizer for trainer scheduling."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        params = [
            param
            for opt in optimizers
            for group in opt.param_groups
            for param in group["params"]
        ]
        self._initializing = True
        super().__init__(params, defaults={})
        self._initializing = False
        self.optimizers = optimizers
        self.param_groups = [
            group
            for opt in optimizers
            for group in opt.param_groups
        ]
        self.state.clear()

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if getattr(self, "_initializing", False):
            torch.optim.Optimizer.add_param_group(self, param_group)
            return
        del param_group
        raise NotImplementedError(
            "_ChainedOptimizer cannot route add_param_group() to a child optimizer; "
            "add new parameters to the intended child optimizer directly."
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], float] | None = None) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        loss = 0.0
        for i, opt in enumerate(self.optimizers):
            loss_i = opt.step(closure if i == 0 else None)
            if loss_i is not None:
                loss = float(loss_i)
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for opt, opt_state in zip(self.optimizers, state_dict["optimizers"], strict=True):
            opt.load_state_dict(opt_state)
        self.param_groups = [
            group
            for opt in self.optimizers
            for group in opt.param_groups
        ]


def _split_decay_groups(
    model: torch.nn.Module,
    *,
    hidden_filter: Callable[[str, torch.nn.Parameter], bool] | None = None,
) -> tuple[list, list, list, list]:
    """Bucket parameters into (hidden_decay, hidden_no_decay, aux_decay, aux_no_decay).

    ``no_decay`` are 1-D tensors and biases (norms, biases — selective weight
    decay convention). ``hidden_filter`` separates a "trunk" subset (hidden_*)
    from the rest (aux_*); without it, hidden_* are empty and all params go
    into aux_*.
    """
    hidden_decay: list = []
    hidden_no_decay: list = []
    aux_decay: list = []
    aux_no_decay: list = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = param.ndim <= 1 or name.endswith(".bias")
        is_hidden = bool(hidden_filter(name, param)) if hidden_filter else False
        if is_hidden and not is_no_decay:
            hidden_decay.append(param)
        elif is_hidden:
            hidden_no_decay.append(param)
        elif is_no_decay:
            aux_no_decay.append(param)
        else:
            aux_decay.append(param)
    return hidden_decay, hidden_no_decay, aux_decay, aux_no_decay


def _extract_named_params(
    groups: tuple[list, ...],
    model: torch.nn.Module,
    predicate: Callable[[str, torch.nn.Parameter], bool],
) -> list:
    """Remove matching parameters from ``groups`` and return them in model order."""
    wanted_ids = {
        id(param)
        for name, param in model.named_parameters()
        if param.requires_grad and predicate(name, param)
    }
    if not wanted_ids:
        return []
    extracted: list = []
    for group in groups:
        kept = []
        for param in group:
            if id(param) in wanted_ids:
                extracted.append(param)
            else:
                kept.append(param)
        group[:] = kept
    return extracted


def _matrix_optimizer_filter(
    scope: str,
    *,
    include_embed_default: bool,
) -> Callable[[str, torch.nn.Parameter], bool]:
    """Return the 2D matrix subset owned by Muon/Aurora-style optimizers."""
    scope = str(scope or "default").lower()

    def _is_block_matrix(name: str, param: torch.nn.Parameter) -> bool:
        return param.ndim >= 2 and name.startswith("blocks.")

    def _is_ffn(name: str) -> bool:
        return ".ffn." in name

    def _is_attn_out(name: str) -> bool:
        return ".out_proj." in name

    def _is_attn_input(name: str) -> bool:
        return any(part in name for part in (".qkv_proj.", ".q_proj.", ".k_proj.", ".v_proj."))

    def _is_attn_v(name: str) -> bool:
        return ".v_proj." in name

    if scope in ("default", "", "legacy"):
        return lambda name, p: p.ndim >= 2 and (
            (include_embed_default and name == "embed.weight") or name.startswith("blocks.")
        )
    if scope in ("blocks", "block_all", "all_blocks", "all_block"):
        return _is_block_matrix
    if scope in ("mlp", "mlp_only", "ffn", "ffn_only"):
        return lambda name, p: _is_block_matrix(name, p) and _is_ffn(name)
    if scope in ("mlp_out", "mlp_o", "mlp_attn_o", "mlp_attention_o"):
        return lambda name, p: _is_block_matrix(name, p) and (_is_ffn(name) or _is_attn_out(name))
    if scope in ("mlp_out_v", "mlp_v_out", "mlp_attn_ov", "mlp_attention_ov"):
        return lambda name, p: _is_block_matrix(name, p) and (
            _is_ffn(name) or _is_attn_out(name) or _is_attn_v(name)
        )
    if scope in ("mlp_attn_all", "mlp_attention_all", "attn_mlp", "all_attention_mlp"):
        return lambda name, p: _is_block_matrix(name, p) and (
            _is_ffn(name) or _is_attn_out(name) or _is_attn_input(name)
        )
    raise ValueError(
        f"Unknown matrix_optimizer_scope {scope!r}. Supported: default, block_all, "
        "mlp_only, mlp_out, mlp_out_v, mlp_attn_all"
    )


# Optimizers whose matrix group applies a SCALE-INVARIANT update, so a global
# gradient-norm clip provably cannot move it. Muon and Aurora both take the
# polar factor of the (momentum-smoothed) gradient via Newton-Schulz; measured
# on `_polar_factor`, max|polar(c*G) - polar(G)| is 1.6e-07 at c=0.90 and
# exactly 0.0 at c=0.50. Every other optimizer here keeps the historical
# whole-model clip, because none of them has been shown to have this property.
_SCALE_INVARIANT_MATRIX_OPTIMIZERS = frozenset({"muon", "aurora"})


class _GradClipScope:
    """A fixed parameter list wearing the exact surface `ZClip` consumes.

    `ZClip` reaches into the model it is handed through only two methods:
    `parameters()` (the norm it measures in `_compute_grad_norm`, the tensors
    it rescales in `apply_in_place_clipping`, and the warmup
    `clip_grad_norm_`) and `modules()` (its `is_fsdp_model` probe). Handing it
    one of these instead of the model therefore restricts BOTH what the clip
    measures and what it rescales, with no other behavioural difference.

    `modules()` is deliberately empty: training here is single-process, and an
    empty iterator keeps `is_fsdp_model` False so zclip stays on its local
    path rather than trying to `all_reduce` on an uninitialised process group.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter]) -> None:
        self._params: list[torch.nn.Parameter] = list(params)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter(self._params)

    def modules(self) -> Iterator[torch.nn.Module]:
        return iter(())


def split_matrix_and_clipped_params(
    model: torch.nn.Module,
    *,
    optimizer: str,
    matrix_optimizer_scope: str,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split trainable parameters into (scale-invariant matrix group, the rest).

    The predicate is `_matrix_optimizer_filter` — literally the one the
    optimizer used to build its param groups — so the clip scope and the
    optimizer cannot drift apart by someone editing one and forgetting the
    other. `include_embed_default` mirrors the optimizer branch exactly: under
    the legacy `default` scope Muon claims `embed.weight` and Aurora does not.

    The 1-D/bias carve-out mirrors `_split_decay_groups`, which routes those to
    an AdamW group (`use_muon`/`use_aurora` False) even when the filter matches
    them. `Trainer.__init__` cross-checks the result against the optimizer's
    own groups, so a future divergence is a hard failure rather than a silently
    mis-scoped clip.
    """
    if str(optimizer).lower() not in _SCALE_INVARIANT_MATRIX_OPTIMIZERS:
        return [], [p for p in model.parameters() if p.requires_grad]
    predicate = _matrix_optimizer_filter(
        matrix_optimizer_scope,
        include_embed_default=str(optimizer).lower() == "muon",
    )
    matrix: list[torch.nn.Parameter] = []
    clipped: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = param.ndim <= 1 or name.endswith(".bias")
        if predicate(name, param) and not is_no_decay:
            matrix.append(param)
        else:
            clipped.append(param)
    return matrix, clipped


def _optimizer_matrix_param_ids(opt: torch.optim.Optimizer) -> set[int]:
    """ids of the parameters the optimizer actually routes to its matrix path."""
    return {
        id(param)
        for group in getattr(opt, "param_groups", [])
        if bool(group.get("use_aurora", False)) or bool(group.get("use_muon", False))
        for param in group["params"]
    }


@dataclass
class TrainMetrics:
    loss: float
    policy_loss: float
    soft_policy_loss: float
    future_policy_loss: float
  # The value loss the optimizer sees (blended soft CE). ``blended_wdl_loss``
  # below carries the same number under its explicit name; ``wdl_onehot_loss``
  # is the hard one-hot diagnostic that used to be reported here.
    wdl_loss: float
    sf_move_loss: float
    sf_move_acc: float
    sf_eval_loss: float
    categorical_loss: float
    volatility_loss: float
    sf_volatility_loss: float
    moves_left_loss: float
    blended_wdl_loss: float = 0.0
  # Diagnostic: hard one-hot CE against the recorded result. No gradient.
    wdl_onehot_loss: float = 0.0
    channel_balance: float = 0.0
    sf_search_agree_frac: float = 0.0
    sf_search_disagree_sf_low_frac: float = 0.0
    sf_search_disagree_sf_high_frac: float = 0.0
  # Terminal-proximal outcome transfer (train/losses.py). `_frac` is the mean
  # weight moved onto the recorded game outcome over ALL batch rows (off the
  # SEARCH component, plus the SF component only when
  # `wdl_terminal_outcome_sf_frac` is raised off 0.0); `_rows` is how many rows
  # received any. Both are exactly 0.0
  # while `wdl_terminal_outcome_plies` is 0, so a non-zero `_frac` is the proof
  # the knob reached the trained target — the config value alone is not. Read
  # them together: `_frac == 0` with `_rows == 0` cannot distinguish the knob
  # being off from a batch holding no near-terminal rows.
    wdl_terminal_outcome_frac: float = 0.0
    wdl_terminal_outcome_rows: float = 0.0
    sf_move_acc_top5: float = 0.0
  # Policy move-ordering accuracy, and its DENOMINATOR. An accuracy divides by
  # the number of rows carrying the head's target (`has_policy` / `has_future`),
  # and on an empty denominator `_acc` returns 0.0 — which for an accuracy is
  # the WORST attainable value, not a null. Without `_rows` alongside it, "the
  # policy head ranks nothing correctly" and "no row in this batch had a policy
  # target" are the same published number. `policy_future` is the live risk:
  # `has_future` is false for rows near the end of a game, so a batch can
  # legitimately hold none.
    policy_own_acc_top1: float = 0.0
    policy_own_acc_top5: float = 0.0
    policy_own_acc_rows: float = 0.0
    policy_future_acc_top1: float = 0.0
    policy_future_acc_top5: float = 0.0
    policy_future_acc_rows: float = 0.0
    train_time_s: float = 0.0
    opt_step_time_s: float = 0.0
    train_steps_done: int = 0
    train_samples_seen: int = 0
    aurora_uw_floor: float = 0.0
  # The matrix-group LR the effective-ratio pair below was multiplied by.
  # Sampled at the sqrt_release sawtooth FLOOR (M4-2) -- ~10x under a typical
  # step -- so the pair is only readable against this column.
    aurora_uw_lr: float = 0.0
    aurora_uw_count: float = 0.0
    aurora_uw_ratio_min: float = 0.0
    aurora_uw_ratio_p10: float = 0.0
    aurora_uw_ratio_median: float = 0.0
    aurora_uw_ratio_p90: float = 0.0
    aurora_uw_scale_max: float = 0.0
    aurora_uw_floored_frac: float = 0.0
    aurora_uw_effective_ratio_min: float = 0.0
    aurora_uw_effective_ratio_median: float = 0.0
  # Polar residual of the update Aurora applied, sampled on ONE designated
  # tensor per shape class per iteration (`train.aurora.polar_convergence`).
  # `_sv_ratio_*` is sigma_min/sigma_max (1.0 = a true orthogonal step) and
  # `_orth_err_*` is ||QQ^T - I||_F/sqrt(n) (0.0 = converged); both are read
  # AGAINST `aurora_polar_steps_configured`, which is the step count that
  # produced them. `aurora_polar_sv_samples` is 2 when both shape classes
  # were sampled -- below 2, a 0.0 ratio means "not measured", not
  # "degenerate". M4-1 / audit I3.
    aurora_polar_steps_configured: float = 0.0
    aurora_polar_sv_samples: float = 0.0
    aurora_polar_sv_errors: float = 0.0
    aurora_polar_sv_ratio_square: float = 0.0
    aurora_polar_sv_ratio_rect: float = 0.0
    aurora_polar_orth_err_square: float = 0.0
    aurora_polar_orth_err_rect: float = 0.0
  # Per-source loss split (observation-only; only meaningful once shards carry is_selfplay).
    policy_loss_selfplay: float = 0.0
    policy_loss_curriculum: float = 0.0
    wdl_loss_selfplay: float = 0.0
    wdl_loss_curriculum: float = 0.0
    frac_is_selfplay: float = 0.0
    frac_tagged: float = 0.0
  # Fraction of soft-policy rows surviving the soft_policy_min_tv mask (1.0 when off).
    soft_mask_kept_frac: float = 1.0
  # sf_p0 policy teacher (w_sf_own / w_sf_own_regret on policy_own, the head
  # MCTS reads). `m_*` are the masked losses over ELIGIBLE rows only; the
  # `_frac` pair is the eligible fraction of the iteration's rows and is the
  # part that catches an outage — it goes to exactly 0.0 when the selfplay
  # workers stop recording sf_p0 labels, independently of the loss weights.
  # Reporting only the masked means would leave "no eligible rows" and
  # "eligible rows at zero loss" indistinguishable, which is what let the
  # teacher sit dead for a month unnoticed (docs/experiment_ledger.md,
  # 2026-07-27 sf_p0 restore).
    m_sf_own: float = 0.0
    m_sf_own_regret: float = 0.0
    has_sf_p0_frac: float = 0.0
    has_sf_p0_regret_frac: float = 0.0
  # ALWAYS-ON SF-label contamination detector. `sf_labelled_no_multipv_frac`
  # is the share of the iteration's SF-LABELLED rows that carry no
  # `sf_multipv_raw` block — the Stockfish UCI desync fingerprint, whose value
  # on healthy data is EXACTLY 0.000000 (11.05M labelled rows across every
  # clean stretch on disk, PR #302). Because the floor is exactly zero the
  # alert rule needs no threshold: any non-zero reading is an incident.
  # Denominator is stated once, in `losses.sf_multipv_presence_counts`, and it
  # is `has_sf_wdl` rows — the same population the offline gate
  # `eval/value_optimism.py::sf_multipv_missing_rate` divides by.
  #
  # It reads the batch's own `has_` vectors and consults NO flag, which is the
  # whole point: the pre-existing signal (`sf_rebuild_policy_frac` below
  # `sf_rebuild_wdl_frac`) is definitionally the same measurement but only
  # exists while `rebuild_sf_targets` is on, and that key defaults False and
  # is in no config file — so it read 0.0, indistinguishable from healthy,
  # through three separate desync episodes spanning 25 days.
  #
  # ⚑ NEVER READ THE RATE WITHOUT `sf_multipv_checked_frac`, which reports that
  # same denominator as a share of all batch rows. (The RATE's own denominator
  # is the SF-labelled rows, above — not all batch rows.)
  # `has_sf_multipv_raw` is an OPTIONAL shard field, so a batch that lost it
  # reports rate 0.0 — which is also what perfect health reports. checked_frac
  # 0.0 means UNMEASURED, and it reads 0.0 in two cases that are operationally
  # identical: the field was absent from the batch, OR the batch held no
  # SF-labelled rows at all. Either way nothing was inspected. On the
  # production window it sits at the SF-labelled share of the batch (~0.99 on
  # the 2026-08-01 live window).
  #
  # ⚑⚑ THE `test_` TWINS WILL NOT READ ZERO ON THE FIRST LIVE ROW, AND THAT IS
  # NOT A PLUMBING BUG. The frozen holdout bundled with the checkpoints is
  # itself desync-contaminated: 194 no-PV rows out of 1915 labelled over 2000
  # rows, byte-identical across checkpoint_000474/476/478, so
  # `test_sf_labelled_no_multipv_frac` = 0.101305 and
  # `test_sf_multipv_checked_frac` = 0.957500 — about 500x the train row's
  # ~2e-4 post-quarantine residue. Unlike the train row it does NOT age out:
  # the holdout is FROZEN, so it stays at 0.101305 until the set is re-cut.
  # Do not read it as a wiring fault and mute the column.
    sf_labelled_no_multipv_frac: float = 0.0
    sf_multipv_checked_frac: float = 0.0
  # ALWAYS-ON SF-label contamination detector, VALUE HALF (audit P2). The
  # column above reads only the POLICY half of the SF label block; `sf_wdl` —
  # realized `sf_wdl_frac` 0.45 of the trained value target — had no detector
  # in EITHER direction until 2026-08-03, and 99.99 % of the rows the policy
  # column flagged on the quarantined shards still carried one.
  #
  # `sf_eval_pv_orphan_frac` is the one with detection power, and it is the
  # first instrument that looks INSIDE the population the policy column
  # passes: it fires when the record-level SF eval that BECAME this row's
  # `sf_wdl` disagrees with the top surviving MultiPV line, which on a healthy
  # row is impossible — both are the same accumulator field. Measured
  # 2026-08-03 THROUGH THIS COLUMN: 0.117386 over the 122 quarantined shards
  # (19,468 orphans of 165,846 checked), 3.0e-5 over the 640 policy-clean
  # post-quarantine shards (34 rows, 1,128,248 labelled), and exactly 0.000000
  # over the 474,278 labelled rows of the pre-episode range 33118:33387, which
  # is the floor. Its denominator is rows carrying ALL THREE blocks, published as
  # `sf_eval_pv_checked_frac` — read them together, exactly as with the policy
  # pair, because a rate over zero checked rows is UNMEASURED, not clean.
  #
  # `sf_wdl_degenerate_frac` has NO power against desync and is not pretending
  # to: it reads exactly 0 on the quarantined shards too, because a desynced
  # engine's label is well-formed and wrong. It is a producer/parameter
  # tripwire with a floor proven over 1.47 M rows.
  #
  # `sf_wdl_orphaned_frac` is the P2 blind spot counted rather than described:
  # policy-flagged rows still carrying a well-formed value label. It is a
  # near-twin of `sf_labelled_no_multipv_frac` BY DESIGN (0.9999 of flagged
  # rows on the quarantined set) — the pair being equal is the finding. Never
  # sum it with the policy rate; they count the same rows.
    sf_wdl_degenerate_frac: float = 0.0
    sf_wdl_orphaned_frac: float = 0.0
    sf_eval_pv_orphan_frac: float = 0.0
    sf_eval_pv_checked_frac: float = 0.0
  # SF target rebuild coverage (train.rebuild_sf_targets). All 0.0 when the
  # flag is off, so a non-zero value IS the proof the flip reached the batch
  # pipeline — the transition log only proves the config push, and
  # has_sf_p0_frac -> 0 only proves it on a window that has p0 rows at all.
  # `policy_frac` BELOW `wdl_frac` is a CONTAMINATION SIGNAL, not a coverage
  # cost. Both fracs divide by ALL rows in the rebuilt batch, and every healthy
  # labelled row carries `sf_label_meta` AND `sf_multipv_raw`, so the two are
  # EQUAL on clean data and their difference is the count of rows that lost
  # their whole MultiPV block, over ALL BATCH ROWS. That is the desync
  # fingerprint (`selfplay/stockfish_turn.py::_SF_NO_LEGAL_PV_WARN_RATE`) and a
  # LOWER BOUND on contamination — a desynced engine strips the block on only
  # ~59% of the labels it poisons, so divide by ~0.59 for the true share.
  # A gap of 5.4% was once documented here as structural; it was a 07-27 desync
  # episode. Measured through this very accumulator: 0.000000 on clean live
  # shards, 0.192 over the 122 quarantined 2026-08-01 (0.207 of LABELLED rows
  # there; do not mix the two denominators). ⚑ Reads 0.0 when
  # `rebuild_sf_targets` is off, which is the default and is not in any config
  # — see target_builder's metric_kwargs before treating it as a live alarm.
  # `eval_full_pass` — the frozen ruler, and the only eval production runs
  # (tune/trainable_phases.py) — pins the rebuild off, so these stay 0.0 on
  # its `eval` row by construction and a non-zero value there means the ruler
  # moved. The SAMPLED `Trainer.eval` is explicitly not a ruler and does
  # rebuild, mirroring the training distribution; it has no production caller.
    sf_rebuild_policy_frac: float = 0.0
    sf_rebuild_wdl_frac: float = 0.0
    sf_rebuild_masked_frac: float = 0.0
  # Per-flag PRE-mask decomposition of `sf_rebuild_masked_frac`. The cross-ply
  # mask zeroes has_sf_p0 / has_sf_volatility indistinguishably from "never
  # recorded", which pins `has_sf_p0_frac` — the outage detector above — at
  # 0.0 for the whole of any rebuild experiment. These columns carry the
  # pre-mask presence fractions, so the detector keeps working while the flag
  # is on: `sf_rebuild_masked_p0_frac == 0.0` with the rebuild running means
  # the selfplay workers stopped recording sf_p0 rows.
    sf_rebuild_masked_p0_frac: float = 0.0
    sf_rebuild_masked_volatility_frac: float = 0.0
  # Per-game-phase loss split, bucketed by PIECE COUNT on the same constant
  # `eval/audit.py` uses, so these columns and the audit's per-phase deep-SF
  # regret name the same positions.
  #
  # ⚑ RENAMED FROM `*_loss_{open,mid,end}` IN THE SAME COMMIT THAT CHANGED THE
  # DEFINITION. Until 2026-08-02 the split bucketed on `moves_left` =
  # plies-remaining / `max_plies` (the CAP), which is not a board property at
  # all and put 96.37 % of the live window in `end`. The old and new columns
  # measure different sets of rows, so they deliberately do not share a name —
  # see train/losses.py for the measurement.
    policy_loss_phase_open: float = 0.0
    policy_loss_phase_mid: float = 0.0
    policy_loss_phase_end: float = 0.0
    wdl_loss_phase_open: float = 0.0
    wdl_loss_phase_mid: float = 0.0
    wdl_loss_phase_end: float = 0.0
  # Rows each `wdl_loss_phase_{open,mid,end}` was actually averaged over,
  # summed across the iteration's microbatches. Raw COUNTS, not rates: a rate
  # would need a denominator to divide by and the whole point is that the
  # denominator is the thing that goes missing. `masked_mean` clamps its
  # denominator to 1.0, so a bucket that collected NO rows publishes 0.0 — the
  # best possible loss — and no other column can contradict it.
    wdl_loss_phase_n_open: float = 0.0
    wdl_loss_phase_n_mid: float = 0.0
    wdl_loss_phase_n_end: float = 0.0
  # The policy head's own denominators. NOT redundant with the three above: the
  # policy mask carries `has_policy` as well as `net_mask`, so the two counts
  # diverge on any window holding value-only rows. `scripts/status.py` prints
  # the POLICY phase columns, so these are the ones an operator reads against.
    policy_loss_phase_n_open: float = 0.0
    policy_loss_phase_n_mid: float = 0.0
    policy_loss_phase_n_end: float = 0.0
  # Value-head calibration (populated on holdout eval).
    wdl_brier: float = 0.0
    wdl_ece: float = 0.0
  # Rows the evaluation actually scored, and the denominator every loss above
  # was divided by on the eval path. Reported rather than reconstructed from
  # steps x batch_size because a full pass ends on a RAGGED batch, so that
  # product is not the row count (docs/rl_loop_audit.md G6, G14). Zero on the
  # training path, which reports `train_samples_seen` instead.
    eval_rows: int = 0
  # Identity of the MEASUREMENT that produced the numbers above, set by
  # `_compute_metrics` from what actually ran (see train/eval_ruler). Empty on
  # the training path, which is not a ruler. It travels on the metrics rather
  # than being re-derived by the consumer so that the async holdout path --
  # which calls `_compute_metrics` directly, on its own thread, with its own
  # `full_pass` argument -- cannot report a number under a ruler it did not
  # use. `tune.trainable` turns a change in this string into a
  # `holdout_generation` bump, which is what stops a best-model promotion
  # across two different instruments.
    eval_ruler: str = ""
  # Gradient-norm / clipping, aggregated over EVERY optimizer step of the
  # iteration (not the tb_log_interval subsample). These used to exist only as
  # TensorBoard scalars, whose event files rotate per Ray session — so the
  # history died at every restart and no ledger yardstick could cite them
  # (docs/rl_loop_audit.md I9).
    grad_norm_mean: float = 0.0
    grad_norm_median: float = 0.0
    grad_norm_p95: float = 0.0
    grad_norm_max: float = 0.0
    grad_clip_rate: float = 0.0
    grad_adaptive_clip_rate: float = 0.0
    grad_hard_clip_rate: float = 0.0
    grad_norm_samples: int = 0
  # Per-optimizer-group grad norms, iteration means. The clip acts on the
  # AdamW group ALONE (the matrix group's update is scale-invariant, so the
  # clip cannot move it), which makes `grad_norm_adamw` equal to
  # `grad_norm_mean` by construction — kept as its own column so a reader does
  # not have to know the clip scope to know which group they are looking at,
  # and so a future scope change shows up as the two disagreeing rather than
  # as `grad_norm_mean` silently changing meaning. `grad_norm_aurora` is the
  # split that previously required a bespoke offline probe to establish.
    grad_norm_aurora: float = 0.0
    grad_norm_adamw: float = 0.0
  # Fraction of optimizer steps skipped because some gradient was inf/nan.
  # Expected to be exactly 0.0 in a healthy run.
    grad_nonfinite_skip_rate: float = 0.0
  # Iteration mean/max of param_groups[0] (the matrix/trunk group) LR. The
  # end-of-iteration LR reported elsewhere is the TROUGH of the sqrt_release
  # ramp, ~9x below the LR the trunk actually trains at (I19). Every group
  # shares the schedule's scale factor, so the aux groups' means are this one
  # divided by matrix_lr_multiplier.
    opt_lr_mean: float = 0.0
    opt_lr_max: float = 0.0
  # ⚑ DRAW-SEQUENCE PROVENANCE — the other half of "same seed, same rows".
  # `batches_drawn` is the total microbatches this call pulled from the buffer,
  # and `transient_cuda_retry_batches` is how many of them were REPLACEMENT
  # draws after a transient CUDA error (`train_steps`' retry path calls
  # `add_retry_batches`, which advances the buffer's RNG). A single retry in one
  # arm of a paired offline A/B desynchronises that arm's row sequence from the
  # other's PERMANENTLY, for every subsequent step, and it was previously
  # visible only as a `logging.warning` — nothing in a report or a metrics row
  # recorded it, so a silently de-paired sweep looked exactly like a clean one.
  # `scripts/retarget_retrain.py` asserts both are equal across arms.
  # Expected to be exactly 0.0 in a healthy run.
  #
  # ⚑ REVIEW FINDING N5, DECIDED — these two fields are ALWAYS ON, and the PR
  # that added them is titled "default-off". Both statements are true and the
  # gap is deliberate: "default-off" describes the temperature's effect on
  # TRAINING, and these are observations, not behaviour. `batches_drawn` reads
  # `batch_iter.consumed`, which the loop already maintained; the retry counter
  # sums `_retry_batches`, a value the retry path already computed and already
  # passed to `add_retry_batches`. No new call, no new branch, no reordering on
  # the training path.
  #
  # They are deliberately NOT gated on `policy_target_temp != 1.0`. The counter
  # exists to detect a de-paired A/B, and in that A/B the arm that runs at 1.0
  # is the CONTROL -- the one that defines the reference draw sequence. Gating
  # the instrument on the intervention would leave the reference arm as the only
  # one with no counter, which is conditioning a control on the thing it is
  # controlling for. A retry is also not caused by the temperature; it is caused
  # by the box.
  #
  # The cost is one more `TrainMetrics` field pair on the live path, both
  # additive with defaults, neither read by the best-model comparison or the
  # promotion gate. The benefit is that a transient CUDA retry -- previously a
  # `logging.warning` on a thousand-line console and nothing else -- becomes a
  # recorded fact. That is worth having live regardless of this PR's knob.
    batches_drawn: float = 0.0
    transient_cuda_retry_batches: float = 0.0


# Map compute_loss dict keys → TrainMetrics field names where they differ.
# Keys not listed pass through unchanged (e.g. the split losses are same-named).
_LOSS_KEY_TO_METRIC_FIELD = {
    "policy_ce": "policy_loss",
    "soft_policy_ce": "soft_policy_loss",
    "future_policy_ce": "future_policy_loss",
    "wdl_ce": "wdl_loss",
    "wdl_onehot_ce": "wdl_onehot_loss",
    "sf_move_ce": "sf_move_loss",
    "sf_eval_ce": "sf_eval_loss",
    "categorical_ce": "categorical_loss",
    "volatility": "volatility_loss",
    "sf_volatility": "sf_volatility_loss",
    "moves_left": "moves_left_loss",
    "blended_wdl_ce": "blended_wdl_loss",
}
_TRAIN_METRICS_FIELDS = frozenset(f.name for f in dataclasses.fields(TrainMetrics))


# TrainMetrics field -> (numerator key, denominator key) among the compute_loss
# scalars accumulated in ``sums``. These are RATIOS OF SUMS: the numerator and
# the denominator are each summed over every microbatch of the iteration and
# divided once, so the result is weighted by how many rows each batch actually
# contributed. The `sums`/`n` path above cannot express that — it averages
# per-batch values with equal weight, which silently assumes every batch
# contributed the same number of rows. That holds for the whole-batch losses
# (their denominator is `net_mask`, ~constant) and does NOT hold for the sf_p0
# terms, whose eligible count swings batch to batch.
_RATIO_METRIC_FIELDS: dict[str, tuple[str, str]] = {
    "m_sf_own": ("sf_own_ce_sum", "sf_own_rows"),
    "m_sf_own_regret": ("sf_own_regret_sum", "sf_own_regret_rows"),
    "has_sf_p0_frac": ("sf_own_rows", "net_rows"),
    "has_sf_p0_regret_frac": ("sf_own_regret_rows", "net_rows"),
  # Contamination detector. Row-weighted for the same reason: the SF-labelled
  # count varies batch to batch, so a mean of per-batch rates is the wrong
  # estimator. `sf_multipv_checked_rows` is BOTH the rate's denominator and the
  # checked-frac's numerator — one quantity, so the pair cannot disagree about
  # how many rows were actually inspected.
    "sf_labelled_no_multipv_frac": ("sf_no_multipv_rows", "sf_multipv_checked_rows"),
    "sf_multipv_checked_frac": ("sf_multipv_checked_rows", "batch_rows"),
  # Value half. The first two divide by `sf_wdl_rows` — their OWN denominator,
  # not the policy pair's `sf_multipv_checked_rows`, so a batch that lost
  # `has_sf_multipv_raw` still reports them over the rows that have a value
  # label. The eval-PV pair keeps its own `checked` count for the same reason
  # the policy pair does: it is the blind-instrument column, and a rate above
  # zero checked rows is unmeasured rather than clean.
    "sf_wdl_degenerate_frac": ("sf_wdl_degenerate_rows", "sf_wdl_rows"),
    "sf_wdl_orphaned_frac": ("sf_wdl_orphaned_rows", "sf_wdl_rows"),
    "sf_eval_pv_orphan_frac": ("sf_eval_pv_orphan_rows", "sf_eval_pv_checked_rows"),
    "sf_eval_pv_checked_frac": ("sf_eval_pv_checked_rows", "batch_rows"),
  # Terminal-proximal outcome transfer. Row-weighted like the pairs above: the
  # near-terminal share of a batch swings, so a mean of per-batch means would
  # be the wrong estimator. Denominator is ALL batch rows, which makes the
  # column read as "share of the value target moved onto the outcome" rather
  # than "average taper among the rows that got one".
    "wdl_terminal_outcome_frac": ("wdl_terminal_outcome_weight_sum", "batch_rows"),
}

# TrainMetrics field -> the compute_loss scalar reported VERBATIM, with no
# division at all. `_loss_sums_to_metric_kwargs` divides every other key by the
# step count and `_ratio_metric_kwargs` divides one sum by another; a row COUNT
# needs neither, and passing it through either path would turn "how many rows
# were in this bucket over the iteration" into rows-per-step or a ratio.
#
# These are the denominators of `wdl_loss_{open,mid,end}` (rl_loop_audit /
# backlog #124). `masked_mean` clamps its denominator to 1.0, so a bucket with
# zero rows publishes a loss of 0.0 — the best possible value — and no other
# column can contradict it. The counts make that state legible.
_RAW_COUNT_METRIC_FIELDS: dict[str, str] = {
    "wdl_loss_phase_n_open": "wdl_rows_phase_open",
    "wdl_loss_phase_n_mid": "wdl_rows_phase_mid",
    "wdl_loss_phase_n_end": "wdl_rows_phase_end",
    "policy_loss_phase_n_open": "policy_rows_phase_open",
    "policy_loss_phase_n_mid": "policy_rows_phase_mid",
    "policy_loss_phase_n_end": "policy_rows_phase_end",
  # How many rows the terminal-proximal transfer actually touched this
  # iteration — `wdl_terminal_outcome_frac`'s companion count, for the same
  # reason the phase losses carry theirs: a rate of 0.0 over 0 eligible rows
  # is unmeasured, not clean.
    "wdl_terminal_outcome_rows": "wdl_terminal_outcome_rows",
}

# The compute_loss scalars consumed by ``_ratio_metric_kwargs`` and
# ``_raw_count_metric_kwargs``. They are already SUMS over the batch's rows, so
# they accumulate unweighted; every other scalar is a per-batch MEAN and must be
# weighted by the batch's row count before it can be pooled across ragged
# batches (see ``Trainer._compute_metrics``). Weighting these would scale
# numerator and denominator by the same factor per batch and so silently change
# the ratio.
_RAW_SUM_LOSS_KEYS: frozenset[str] = frozenset(
    [key for pair in _RATIO_METRIC_FIELDS.values() for key in pair]
    + list(_RAW_COUNT_METRIC_FIELDS.values()),
)


# Param-group keys the CHECKPOINT legitimately owns on resume: the schedule's
# phase (`lr` is re-applied by set_peak_lr right after load, `initial_lr` is
# scheduler bookkeeping) and per-group counters (SODA's step). EVERY other
# param-group key is config-derived and must be re-applied from this run's
# config after a load — see Trainer._reapply_configured_param_group_hparams.
_CHECKPOINT_OWNED_GROUP_KEYS = frozenset({"params", "lr", "initial_lr", SODA_STEP_KEY})


def _scalar_hparam_differs(old: Any, new: Any) -> bool:
    """True when a plain-scalar param-group hyperparameter changed.

    Restricted to scalars so a non-comparable group value (tuple of tensors,
    dtype object, ...) can never make the diagnostic logging path raise.
    """
    if not isinstance(old, (int, float, bool, str)) or not isinstance(new, (int, float, bool, str)):
        return False
    return old != new


# Pre-committed watch threshold from docs/rl_loop_audit.md I11: once the
# windowed MEDIAN grad norm passes this (hard-clip rate ~10%), zclip_max_norm
# has stopped being a tail guard and has become an LR cap in disguise. Recorded
# here (and in the yaml comment on zclip_max_norm) so the call is not made
# post-hoc.
#
# The median it is compared against is the CLIPPED group's norm, which is what
# makes the comparison meaningful: since 2026-07-27 the clip scope excludes the
# scale-invariant matrix group, so both sides of this inequality are the same
# quantity. Threshold unchanged; the series it reads steps down ~4.4% at that
# deploy, which is the measurement narrowing onto the clip, not the run moving.
GRAD_NORM_MEDIAN_WATCH = 4.75


def _nearest_rank_quantile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank quantile of an already-sorted list (0.0 when empty)."""
    if not sorted_values:
        return 0.0
    rank = math.ceil(q * len(sorted_values)) - 1
    return float(sorted_values[min(len(sorted_values) - 1, max(0, rank))])


def _mean(values: list[float]) -> float:
    """Arithmetic mean, 0.0 when empty (matches the other aggregators here)."""
    return float(sum(values) / len(values)) if values else 0.0


def _median(sorted_values: list[float]) -> float:
    n = len(sorted_values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return float(sorted_values[mid])
    return 0.5 * (float(sorted_values[mid - 1]) + float(sorted_values[mid]))


def _grad_clip_metric_kwargs(
    grad_norms: list[float],
    clip_counts: dict[str, int],
    aurora_grad_norms: list[float] | None = None,
) -> dict[str, float | int]:
    """Aggregate per-step grad norms + clip flags into TrainMetrics kwargs.

    Aggregated over EVERY step of the iteration, not the tb_log_interval
    subsample — a 1-in-10 sample cannot rule out a burst pattern aligned to the
    stride (the caveat recorded against I10).

    `grad_norms` is the CLIPPED group's norm (see `Trainer._grad_clip_target`);
    `aurora_grad_norms` is the scale-invariant matrix group's, empty when the
    optimizer has no such group.

    A DISCARDED step (non-finite norm, see `Trainer._run_optimizer_step`) still
    counts toward the RATES — it is a step that happened — but is kept out of
    the order statistics. `sorted()` does not order nan, so one of them takes
    the mean, the median AND the max with it: `[12.0, 12.1, nan, 11.9]` reports
    a max of 11.9, the smallest value in the list.
    """
    n = len(grad_norms)
    if n == 0:
        return {}
    ordered = sorted(v for v in grad_norms if math.isfinite(v))
    out: dict[str, float | int] = {
        "grad_norm_mean": _mean(ordered),
        "grad_norm_median": _median(ordered),
        "grad_norm_p95": _nearest_rank_quantile(ordered, 0.95),
        "grad_norm_max": float(ordered[-1]) if ordered else 0.0,
        "grad_clip_rate": float(clip_counts.get("clipped", 0)) / n,
        "grad_adaptive_clip_rate": float(clip_counts.get("adaptive_clip", 0)) / n,
        "grad_hard_clip_rate": float(clip_counts.get("hard_clip", 0)) / n,
        "grad_nonfinite_skip_rate": float(clip_counts.get("nonfinite_grad", 0)) / n,
  # Steps, not finite readings: the rates above are over this denominator.
  # The two agree unless `grad_nonfinite_skip_rate` is non-zero.
        "grad_norm_samples": n,
        "grad_norm_adamw": _mean(ordered),
    }
    out["grad_norm_aurora"] = _mean([v for v in (aurora_grad_norms or []) if math.isfinite(v)])
    return out


def _loss_sums_to_metric_kwargs(sums: dict[str, float], n: float) -> dict[str, float]:
    """Convert accumulated per-batch loss sums into TrainMetrics kwargs.

    Keys that don't map to a TrainMetrics field are dropped silently so
    compute_loss can add experimental scalars before TrainMetrics catches up.
    That drop also disposes of the raw numerator/denominator pairs consumed by
    ``_ratio_metric_kwargs``: they are sums over the whole iteration and would
    be meaningless divided by the step count.
    """
    out: dict[str, float] = {}
    for k, v in sums.items():
        field = _LOSS_KEY_TO_METRIC_FIELD.get(k, k)
        if field in _TRAIN_METRICS_FIELDS:
            out[field] = v / n
    out.update(_ratio_metric_kwargs(sums))
    out.update(_raw_count_metric_kwargs(sums))
    return out


def _raw_count_metric_kwargs(sums: dict[str, float]) -> dict[str, float]:
    """Row-count metrics passed through with no division of any kind.

    An absent key is reported as 0.0 rather than omitted: these columns are read
    as "how many rows landed here", and a missing column and a zero column mean
    the same thing to the reader, while omitting it would let the field's
    dataclass default supply the same 0.0 anyway with no record of which path
    produced it.
    """
    return {
        field: float(sums.get(sum_key, 0.0))
        for field, sum_key in _RAW_COUNT_METRIC_FIELDS.items()
    }


def _ratio_metric_kwargs(sums: dict[str, float]) -> dict[str, float]:
    """Row-weighted ratio metrics from accumulated numerator/denominator sums.

    A zero denominator reports 0.0 rather than raising or emitting NaN: for the
    sf_p0 terms that is the no-eligible-rows case, and the companion `_frac`
    column is what says so unambiguously.
    """
    out: dict[str, float] = {}
    for field, (num_key, den_key) in _RATIO_METRIC_FIELDS.items():
        if num_key not in sums or den_key not in sums:
            continue
        den = float(sums[den_key])
        out[field] = float(sums[num_key]) / den if den > 0.0 else 0.0
    return out


def state_dict_unique_param_count(sd: Mapping[str, Any]) -> int:
    """Parameters in *sd*, counting each STORAGE once.

    ``sum(v.numel())`` over `state_dict` entries double-counts weight tying: the
    production net shares one smolgen generator across 16 keys, so the naive sum
    reads 78,812,768 against a true 63,084,128 (CLAUDE.md, rl_loop_audit J11).
    A published-model log line that reported the naive number would be a count
    that does not mean what its name says, in the very place this log exists to
    make unambiguous.
    """
    seen: set[int] = set()
    total = 0
    for value in sd.values():
        if not isinstance(value, torch.Tensor):
            continue
        ptr = value.untyped_storage().data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += int(value.numel())
    return total


def state_dict_digest(sd: Mapping[str, Any]) -> str:
    """Short content hash over every tensor in *sd*, key order independent.

    Hashes the VALUES (plus each key, dtype and shape), so it changes when the
    weights change and matches when two artifacts carry the same weights under
    different names. That is the property the publish log needs: it is what
    distinguishes ``self.model`` from ``self._swa_model.module`` — measured in
    repro, those differ in 86/86 tensors under SWA — and what lets a reader
    tie a published file to the checkpoint it claims to be.

    Not `sha256_file` on the written path: this must describe the object that
    was exported, not the bytes torch happened to serialize, so a future change
    to the container format cannot make the identity silently drift.
    """
    digest = hashlib.sha256()
    for key in sorted(sd):
        value = sd[key]
        if not isinstance(value, torch.Tensor):
            continue
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(
            value.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes(),
        )
    return digest.hexdigest()[:16]


def strip_compile_prefix(sd: Mapping[str, Any]) -> dict[str, Any]:
    """Drop ``torch.compile``'s ``_orig_mod.`` wrapper segment from state_dict keys.

    Every tensor this project writes to disk must be wrap-agnostic: whether the
    producing trainer ran under ``use_compile`` is an implementation detail of
    the producer, and a consumer that rebuilds an unwrapped model must not have
    to know. Without this, a save under ``use_compile: true`` emits keys like
    ``_orig_mod.embed.weight``; a plain ``load_state_dict(..., strict=False)``
    then reports every key as unexpected and leaves a fresh-init model behind
    with no error — the failure mode that destroyed the model on 2026-04-27.

    ``replace(..., 1)`` rather than ``removeprefix`` because the segment is not
    always leading: ``AveragedModel`` (SWA) nests the compiled module, so its
    keys read ``module._orig_mod.embed.weight``. ``removeprefix`` left those
    untouched, so ``save()``'s ``swa_model`` entry was never actually made
    wrap-agnostic despite the comment claiming it was. No real submodule is
    named ``_orig_mod``, so removing the first occurrence anywhere is safe.
    """
    return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}


def align_compile_prefix(
    sd: Mapping[str, Any], *, reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-key *sd* into whatever ``_orig_mod.`` convention *reference* uses.

    The load-side counterpart of :func:`strip_compile_prefix`. Checkpoints are
    written wrap-agnostic, but the live module may be compiled, so a stored key
    ``module.embed.weight`` has to become ``module._orig_mod.embed.weight``
    before it will load. Matching on the *stripped* key handles both directions
    and both nesting depths (bare model, and ``AveragedModel``'s extra
    ``module.``) without hard-coding where the segment sits. Keys the reference
    does not know are passed through unchanged so the caller's own
    missing/unexpected reporting still sees them.
    """
    ref_by_stripped = {k.replace("_orig_mod.", "", 1): k for k in reference}
    return {
        ref_by_stripped.get(k.replace("_orig_mod.", "", 1), k): v
        for k, v in sd.items()
    }


def resolve_zclip_max_norm(config: dict) -> float | None:
    """The one answer to "what fixed grad-norm cap does this config mean?".

    Two callers need it and they must not disagree: the constructor path
    (``trainer_kwargs_from_config``) and the per-iteration live push in
    ``tune/trainable.py``. An earlier revision of the push spelled this
    ``config.get("zclip_max_norm")``, which resolves an ABSENT key to ``None``
    -- i.e. "cap disabled" -- while the constructor resolved the same config to
    ``grad_clip`` or ``1.0``. Any config that sets only ``grad_clip``
    (``configs/default.yaml``, or a bare CLI run) would have had its hard cap
    silently switched off at iteration 1. Resolve in one place so the divergence
    cannot come back.

    ``grad_clip`` is the argparse spelling in run.py. An explicit ``None``
    disables the fixed cap while leaving adaptive z-score clipping active.
    """
    raw = config.get("zclip_max_norm", config.get("grad_clip", 1.0))
    return None if raw is None else float(raw)


class _SfRebuildCoverageAccumulator:
    """Thread-safe running total of what the SF target rebuild touched.

    The rebuild runs on the host PREFETCH thread and the metrics are built on
    the thread that consumes the batches, so the counters need a lock. Drained
    (read-and-reset) at each metrics boundary; a window in which the rebuild
    never ran drains to all-zero rather than to nothing, so "the rebuild
    stopped happening" is visible as 0.0 instead of as an absent column.

    ONE INSTANCE PER MEASUREMENT, not one per process. The trainer holds the
    training one; ``_compute_metrics`` makes a fresh one for each eval and
    passes it down through ``coverage=``. A shared instance is wrong because
    ``drain()`` RESETS: the async holdout eval calls ``_compute_metrics`` on
    the same Trainer from its own thread while the next iteration is training
    (``distributed_async_test_eval: true`` in the production config), so its
    drain would take counts the training path had accumulated, publish them on
    the ``eval`` row, and leave the ``train`` row short by an unknowable
    amount. Reasoning about which paths ACCUMULATE is not sufficient — the
    full pass accumulates nothing yet still drains.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = SfRebuildCoverage()

    def add(self, coverage: SfRebuildCoverage) -> None:
        with self._lock:
            self._total = self._total + coverage

    def drain(self) -> dict[str, float]:
        with self._lock:
            total = self._total
            self._total = SfRebuildCoverage()
        return total.metric_kwargs()


def resolve_sf_target_params(config: dict) -> SfTargetParams:
    """SfTargetParams from a flat config dict.

    Shared by `trainer_kwargs_from_config` (construction) and the per-iteration
    live push, so the two cannot read the same yaml keys differently — the
    defect shape where a live edit lands in a value the constructor never saw.

    Defaults come FROM THE DATACLASS, not from a re-hardcoded copy: the same
    seven yaml keys are also read by `tune/distributed_runtime.py` (worker
    manifest), `worker.py` (reco resolution) and `tune/trial_config.py`, all
    of which now derive from `SfTargetParams` too, and
    `test_sf_target_param_defaults_have_one_home` pins the `GameConfig` /
    `TrialConfig` dataclass defaults to it. Five independent copies of the
    same default is how a drifted reader ships a silent capture/rebuild
    mismatch.
    """
    d = SfTargetParams()
    return SfTargetParams(
        sf_policy_temp=float(config.get("sf_policy_temp", d.sf_policy_temp)),
        sf_policy_label_smooth=float(
            config.get("sf_policy_label_smooth", d.sf_policy_label_smooth)
        ),
        sf_wdl_use_cp_logistic=bool(
            config.get("sf_wdl_use_cp_logistic", d.sf_wdl_use_cp_logistic)
        ),
        sf_wdl_cp_slope=float(config.get("sf_wdl_cp_slope", d.sf_wdl_cp_slope)),
        sf_wdl_cp_draw_width=float(
            config.get("sf_wdl_cp_draw_width", d.sf_wdl_cp_draw_width)
        ),
        sf_policy_score_mode=str(
            config.get("sf_policy_score_mode", d.sf_policy_score_mode)
        ),
        sf_policy_cp_temp=float(
            config.get("sf_policy_cp_temp", d.sf_policy_cp_temp)
        ),
    )


def trainer_kwargs_from_config(config: dict, *, log_dir: Path | None = None) -> dict:
    """Extract Trainer constructor kwargs from a flat config dict.

    Single source of truth for config → Trainer mapping, used by both
    run.py (single mode) and tune/trainable.py.  Callers can override
    individual keys in the returned dict before passing to Trainer().

    Accepts ``grad_clip`` as an alias for ``zclip_max_norm`` (the argparse
    name used in run.py).
    """
    def _f(key: str, default: float, typ: type = float) -> Any:
        return typ(config.get(key, default))

    zclip_max_norm = resolve_zclip_max_norm(config)

  # w_sf_volatility falls back to w_volatility if not explicitly set
    w_volatility = _f("w_volatility", 0.05)
    w_sf_volatility_raw = config.get("w_sf_volatility")
    w_sf_volatility = float(w_sf_volatility_raw) if w_sf_volatility_raw is not None else w_volatility

    kw: dict[str, Any] = {
        "device": str(config.get("device", "cpu")),
        "lr": _f("lr", 3e-4),
        "zclip_z_thresh": _f("zclip_z_thresh", 2.5),
        "zclip_alpha": _f("zclip_alpha", 0.97),
        "zclip_clip_factor": _f("zclip_clip_factor", 1.0),
        "zclip_max_norm": zclip_max_norm,
        "use_amp": bool(config.get("use_amp", True)),
        "feature_dropout_p": _f("feature_dropout_p", 0.3),
        "rebuild_sf_targets": bool(config.get("rebuild_sf_targets", False)),
        "sf_policy_sparse_ce": bool(config.get("sf_policy_sparse_ce", False)),
        "sf_target_params": resolve_sf_target_params(config),
        "rebuild_categorical_target": bool(config.get("rebuild_categorical_target", False)),
        "categorical_target_params": CategoricalTargetParams(
            blend_frac=_f("categorical_blend_frac", 0.0),
            search_blend_frac=_f("categorical_search_blend_frac", 0.0),
            num_bins=int(config.get("categorical_bins", DEFAULT_CATEGORICAL_BINS)),
            sigma=_f("hlgauss_sigma", 0.04),
        ),
        "fdp_king_safety": config.get("fdp_king_safety"),
        "fdp_pins": config.get("fdp_pins"),
        "fdp_pawns": config.get("fdp_pawns"),
        "fdp_mobility": config.get("fdp_mobility"),
        "fdp_outposts": config.get("fdp_outposts"),
        "w_volatility": w_volatility,
        "resid_channel_dropout": _f("resid_channel_dropout", 0.0),
        "resid_channel_balance_weight": _f("resid_channel_balance_weight", 0.0),
        "accum_steps": _f("accum_steps", 1, int),
        "warmup_steps": _f("warmup_steps", 1500, int),
        "warmup_lr_start": config.get("warmup_lr_start"),
        "lr_eta_min": _f("lr_eta_min", 1e-5),
        "lr_T0": _f("lr_T0", 5000, int),
        "lr_T_mult": _f("lr_T_mult", 2, int),
        "lr_schedule": str(config.get("lr_schedule", "cosine")),
        "lr_release_cycle_steps": _f("lr_release_cycle_steps", 0, int),
        "lr_release_start_frac": _f("lr_release_start_frac", 0.85),
        "lr_release_min_scale": _f("lr_release_min_scale", 0.1),
        "lr_release_shape": str(config.get("lr_release_shape", "sqrt")),
        "use_compile": bool(config.get("use_compile", False)),
        "compile_mode": str(config.get("compile_mode", "reduce-overhead")),
        "optimizer": str(config.get("optimizer", "nadamw")),
        "matrix_optimizer_scope": str(config.get("matrix_optimizer_scope", "default")),
        "matrix_lr_multiplier": _f("matrix_lr_multiplier", 20.0),
        "matrix_weight_decay": _f("matrix_weight_decay", 1e-4),
        "aux_weight_decay": _f("aux_weight_decay", 1e-4),
        "global_board_preprocess_lr_multiplier": _f("global_board_preprocess_lr_multiplier", 1.0),
        "global_board_preprocess_weight_decay": _f("global_board_preprocess_weight_decay", 0.0),
        "global_board_adapter_lr_multiplier": _f("global_board_adapter_lr_multiplier", 1.0),
        "global_board_adapter_weight_decay": _f("global_board_adapter_weight_decay", 0.0),
        "weight_decay_mode": str(config.get("weight_decay_mode", "weight_decay")),
        "soda_scope": str(config.get("soda_scope", "decay")),
        "soda_start_step": _f("soda_start_step", 0, int),
        "aurora_uw_floor": _f("aurora_uw_floor", 0.0),
        "aurora_pp_iterations": _f("aurora_pp_iterations", 2, int),
        "aurora_pp_beta": _f("aurora_pp_beta", 0.5),
        "aurora_polar_steps": _f("aurora_polar_steps", 12, int),
        "aurora_polar_method": str(config.get("aurora_polar_method", "simple")),
        "aurora_polar_dtype": str(config.get("aurora_polar_dtype", "auto")),
        "aurora_polar_safety": _f("aurora_polar_safety", 1.01),
        "aurora_cuda_graphs": bool(config.get("aurora_cuda_graphs", True)),
        "swa_start": _f("swa_start", 0, int),
        "swa_freq": _f("swa_freq", 50, int),
        "w_policy": _f("w_policy", 1.0),
        "w_soft": _f("w_soft", 0.5),
        "soft_policy_min_tv": _f("soft_policy_min_tv", 0.0),
  # Spelled as a literal `config.get` rather than `_f(...)` DELIBERATELY:
  # tests/test_startup_only_config_keys.py derives the startup-only set from
  # literal config reads, and a key read through the `_f` helper is invisible
  # to it. Declaring this one startup-only while the instrument cannot see it
  # would be exactly the hand-override that file's docstring forbids.
        "policy_target_temp": float(config.get("policy_target_temp", 1.0)),
        "w_future": _f("w_future", 0.15),
        "w_sf_own": _f("w_sf_own", 0.0),
        "w_sf_own_regret": _f("w_sf_own_regret", 0.0),
        "w_wdl": _f("w_wdl", 1.0),
        "w_sf_move": _f("w_sf_move", 0.15),
        "w_sf_eval": _f("w_sf_eval", 0.15),
        "w_categorical": _f("w_categorical", 0.10),
        "w_sf_volatility": w_sf_volatility,
        "w_moves_left": _f("w_moves_left", 0.02),
        "sf_wdl_frac": _f("sf_wdl_frac", 0.0),
        "search_wdl_frac": _f("search_wdl_frac", 0.0),
        "sf_wdl_conf_power": _f("sf_wdl_conf_power", 0.0),
        "sf_wdl_draw_scale": _f("sf_wdl_draw_scale", 1.0),
        "sf_wdl_temperature": _f("sf_wdl_temperature", 1.0),
        "sf_search_dampen_sf_low": _f("sf_search_dampen_sf_low", 0.0),
        "sf_search_dampen_sf_high": _f("sf_search_dampen_sf_high", 0.0),
        "use_adjusted_wdl_target": bool(config.get("use_adjusted_wdl_target", False)),
        "adjusted_wdl_regret_source": str(config.get("adjusted_wdl_regret_source", "sum")),
        "adjusted_wdl_regret_scale": _f("adjusted_wdl_regret_scale", 1.0),
        "adjusted_wdl_regret_cap": _f("adjusted_wdl_regret_cap", 0.0),
        "wdl_terminal_outcome_plies": _f("wdl_terminal_outcome_plies", 0, int),
        "wdl_terminal_outcome_full_plies": _f("wdl_terminal_outcome_full_plies", 2, int),
        "wdl_terminal_outcome_sf_frac": _f("wdl_terminal_outcome_sf_frac", 0.0),
  # The SELFPLAY ply cap, not a training knob — `moves_left` is stored as
  # plies-remaining divided by it, so the terminal-proximal transfer cannot
  # recover a ply distance without the same number the writers used. Default
  # matches `TrialConfig.max_plies` / `GameConfig.max_plies` (240); production
  # sets 450.
        "moves_left_max_plies": _f("max_plies", 240, int),
    }
    if log_dir is not None:
        kw["log_dir"] = log_dir
    return kw


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str,
        lr: float,
        zclip_z_thresh: float = 2.5,
        zclip_alpha: float = 0.97,
        zclip_clip_factor: float = 1.0,
        zclip_max_norm: float | None = 1.0,
        log_dir: Path | None = None,
        use_amp: bool = True,
        feature_dropout_p: float = 0.3,
        rebuild_sf_targets: bool = False,
        sf_policy_sparse_ce: bool = False,
        sf_target_params: SfTargetParams | None = None,
        rebuild_categorical_target: bool = False,
        categorical_target_params: CategoricalTargetParams | None = None,
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
        lr_schedule: str = "cosine",
        lr_release_cycle_steps: int = 0,
        lr_release_start_frac: float = 0.85,
        lr_release_min_scale: float = 0.1,
        lr_release_shape: str = "sqrt",
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        optimizer: str = "nadamw",
        matrix_optimizer_scope: str = "default",
        matrix_lr_multiplier: float = 20.0,
        matrix_weight_decay: float = 1e-4,
        aux_weight_decay: float = 1e-4,
        global_board_preprocess_lr_multiplier: float = 1.0,
        global_board_preprocess_weight_decay: float = 0.0,
        global_board_adapter_lr_multiplier: float = 1.0,
        global_board_adapter_weight_decay: float = 0.0,
        weight_decay_mode: str = "weight_decay",
        soda_scope: str = "decay",
        soda_start_step: int = 0,
        aurora_uw_floor: float = 0.0,
        aurora_pp_iterations: int = 2,
        aurora_pp_beta: float = 0.5,
        aurora_polar_steps: int = 12,
        aurora_polar_method: str = "simple",
        aurora_polar_dtype: str = "auto",
        aurora_polar_safety: float = 1.01,
        aurora_cuda_graphs: bool = True,
        swa_start: int = 0,
        swa_freq: int = 50,
        mirror_prob: float = 0.5,
  # Loss weights (all tunable for Ray Tune ablations)
        w_policy: float = 1.0,
        w_soft: float = 0.5,
        soft_policy_min_tv: float = 0.0,
        policy_target_temp: float = 1.0,
        w_future: float = 0.15,
        w_sf_own: float = 0.0,
        w_sf_own_regret: float = 0.0,
        w_wdl: float = 1.0,
        w_sf_move: float = 0.15,
        w_sf_eval: float = 0.15,
        w_categorical: float = 0.10,
        resid_channel_dropout: float = 0.0,
        resid_channel_balance_weight: float = 0.0,
        w_sf_volatility: float | None = None,
        w_moves_left: float = 0.02,
        sf_wdl_frac: float = 0.0,
        search_wdl_frac: float = 0.0,
        sf_wdl_conf_power: float = 0.0,
        sf_wdl_draw_scale: float = 1.0,
        sf_wdl_temperature: float = 1.0,
        sf_search_dampen_sf_low: float = 0.0,
        sf_search_dampen_sf_high: float = 0.0,
        use_adjusted_wdl_target: bool = False,
        adjusted_wdl_regret_source: str = "sum",
        adjusted_wdl_regret_scale: float = 1.0,
        adjusted_wdl_regret_cap: float = 0.0,
        wdl_terminal_outcome_plies: int = 0,
        wdl_terminal_outcome_full_plies: int = 2,
        wdl_terminal_outcome_sf_frac: float = 0.0,
        moves_left_max_plies: float = 0.0,
        tb_log_interval: int = 10,
        prefetch_batches: bool = True,
        model_config: ModelConfig | None = None,
    ):
        self.device = device
  # Declared as nn.Module; torch.compile (below) wraps it in a Module
  # subclass at runtime, but its stub types return Callable — cast on
  # assignment there to keep attribute access (.train/.eval/.state_dict)
  # type-checked here.
        self.model: torch.nn.Module = model.to(device)
  # Optional — when provided, `save()` and the SWA export embed it
  # into the checkpoint so standalone loaders (UCI engine) don't need
  # a sibling params.json. Kept optional for backward compatibility
  # with direct Trainer() construction in tests.
        self._model_config = model_config
        self._input_history_encoding = normalize_lc0_history_encoding(
            model_config.input_history_encoding if model_config is not None else None
        )

        optimizer = str(optimizer).lower()
        matrix_optimizer_scope = str(matrix_optimizer_scope).lower()
        weight_decay_mode = str(weight_decay_mode).lower()
        if weight_decay_mode not in ("weight_decay", "soda"):
            raise ValueError(
                f"Unknown weight_decay_mode {weight_decay_mode!r}. "
                "Supported: weight_decay, soda"
            )
        soda_scope = str(soda_scope or "decay").lower()
        soda_scope_aliases = {
            "decay": "decay",
            "weight_decay": "decay",
            "nonzero_decay": "decay",
            "hidden_matrix_only": "hidden_matrix_only",
        }
        if soda_scope not in soda_scope_aliases:
            raise ValueError(
                f"Unknown soda_scope {soda_scope!r}. "
                "Supported: decay, hidden_matrix_only "
                "(aliases: weight_decay, nonzero_decay)"
            )
        soda_scope = soda_scope_aliases[soda_scope]
        soda_start_step = max(0, int(soda_start_step))
        use_soda_weight_decay = False

        def _mark_soda(param_groups: list[dict], *, hidden_group_indices: tuple[int, ...] = (0,)) -> bool:
            if weight_decay_mode != "soda":
                return False
            if soda_scope == "decay":
                return mark_soda_weight_decay_groups(param_groups, start_step=soda_start_step)
            for group in param_groups:
                group["weight_decay"] = 0.0
            marked = False
            for idx in hidden_group_indices:
                if 0 <= idx < len(param_groups):
                    marked = (
                        mark_soda_weight_decay_groups(
                            [param_groups[idx]],
                            start_step=soda_start_step,
                            force=True,
                        )
                        or marked
                    )
            return marked

        if optimizer in ("muon", "aurora"):
            if optimizer == "muon":
                hidden_filter = _matrix_optimizer_filter(matrix_optimizer_scope, include_embed_default=True)
            else:
                hidden_filter = _matrix_optimizer_filter(matrix_optimizer_scope, include_embed_default=False)
            hd, hnd, ad, and_ = _split_decay_groups(
                self.model,
                hidden_filter=hidden_filter,
            )
            global_board_preprocess_params = _extract_named_params(
                (hd, hnd, ad, and_),
                self.model,
                lambda name, _param: name.startswith("global_board_preprocess."),
            )
            global_board_adapter_params = _extract_named_params(
                (hd, hnd, ad, and_),
                self.model,
                lambda name, _param: name.startswith("global_board_adapter."),
            )
  # Muon/Aurora trunk gets a larger LR than the AdamW fallback for heads/norms.
  # Keep one Tune-search LR and derive trunk LR so search stays simple.
            matrix_lr = float(lr) * float(matrix_lr_multiplier)
            matrix_wd = float(matrix_weight_decay)
            aux_wd = float(aux_weight_decay)
            branch_groups = []
            if global_board_preprocess_params:
                branch_groups.append(
                    {
                        "params": global_board_preprocess_params,
                        "weight_decay": float(global_board_preprocess_weight_decay),
                        "use_muon": False,
                        "use_aurora": False,
                        "lr": float(lr) * float(global_board_preprocess_lr_multiplier),
                    }
                )
            if global_board_adapter_params:
                branch_groups.append(
                    {
                        "params": global_board_adapter_params,
                        "weight_decay": float(global_board_adapter_weight_decay),
                        "use_muon": False,
                        "use_aurora": False,
                        "lr": float(lr) * float(global_board_adapter_lr_multiplier),
                    }
                )
            if optimizer == "muon":
                param_groups = [
                    {"params": hd, "weight_decay": matrix_wd, "use_muon": True, "lr": matrix_lr},
                    {"params": hnd, "weight_decay": 0.0, "use_muon": True, "lr": matrix_lr},
                    {"params": ad, "weight_decay": aux_wd, "use_muon": False, "lr": float(lr)},
                    {"params": and_, "weight_decay": 0.0, "use_muon": False, "lr": float(lr)},
                    *branch_groups,
                ]
                use_soda_weight_decay = _mark_soda(param_groups)
                self.opt = MuonWithAuxAdam(param_groups)
            else:
                param_groups = [
                    {
                        "params": hd,
                        "weight_decay": matrix_wd,
                        "use_aurora": True,
                        "lr": matrix_lr,
                        "aurora_uw_floor": float(aurora_uw_floor),
                    },
                    {"params": hnd, "weight_decay": 0.0, "use_aurora": False, "lr": float(lr)},
                    {"params": ad, "weight_decay": aux_wd, "use_aurora": False, "lr": float(lr)},
                    {"params": and_, "weight_decay": 0.0, "use_aurora": False, "lr": float(lr)},
                    *branch_groups,
                ]
                use_soda_weight_decay = _mark_soda(param_groups)
                self.opt = AuroraWithAuxAdam(
                    param_groups,
                    aurora_pp_iterations=int(aurora_pp_iterations),
                    aurora_pp_beta=float(aurora_pp_beta),
                    aurora_polar_steps=int(aurora_polar_steps),
                    aurora_polar_method=str(aurora_polar_method),
                    aurora_polar_dtype=str(aurora_polar_dtype),
                    aurora_polar_safety=float(aurora_polar_safety),
                    aurora_cuda_graphs=bool(aurora_cuda_graphs),
                )
        else:
  # Selective weight decay: apply only to non-bias, non-LayerNorm parameters.
            if weight_decay_mode == "soda" and soda_scope == "hidden_matrix_only":
                hd, hnd, ad, and_ = _split_decay_groups(
                    self.model,
                    hidden_filter=_matrix_optimizer_filter(
                        matrix_optimizer_scope,
                        include_embed_default=False,
                    ),
                )
                param_groups = [
                    {"params": hd, "weight_decay": 0.0},
                    {"params": hnd, "weight_decay": 0.0},
                    {"params": ad, "weight_decay": 0.0},
                    {"params": and_, "weight_decay": 0.0},
                ]
                use_soda_weight_decay = _mark_soda(param_groups)
            else:
                _, _, decay_params, no_decay_params = _split_decay_groups(self.model)
                param_groups = [
                    {"params": decay_params, "weight_decay": 1e-4},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ]
                use_soda_weight_decay = _mark_soda(param_groups)

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
        elif optimizer == "muon" or optimizer == "aurora":
            pass
        elif optimizer == "soap":
  # SOAP: Shampoo-like second-order optimizer. Prefer a local
  # `soap.py`; otherwise fall back to pytorch-optimizer's SOAP.
            try:
                from soap import SOAP  # pyright: ignore[reportMissingImports] # optional local module
            except ImportError as exc:
                try:
                    from pytorch_optimizer import SOAP
                except ImportError:
                    raise ImportError(
                        "SOAP optimizer requires either a local `soap.py` module "
                        "or the `pytorch-optimizer` package (ships in the "
                        "`train` extra: pip install -e '.[train]')"
                    ) from exc
            if matrix_optimizer_scope in ("default", "", "legacy"):
                try:
                    self.opt = SOAP(param_groups, lr=lr)
                except TypeError as exc:
  # ⚑ LOG, do not silently degrade. On this branch every parameter
  # gets the same flat `lr` and no group-specific weight decay: the
  # per-group split the caller configured is DISCARDED. Nothing
  # recorded that, so a later analysis would cite a decay/grouping
  # setting that never applied -- the same shape as the
  # `matrix_weight_decay is decorative` finding.
  # The fallback is kept rather than made fatal because production
  # runs `aurora`, not `soap` (see configs/pbt2_small.yaml), so
  # raising here would convert a dormant path into a hard failure
  # for a research config nobody is running, for no benefit today.
  # If SOAP is ever promoted, this warning is the thing that stops
  # every group-wise conclusion drawn from it being unfalsifiable.
                    logging.getLogger(__name__).warning(
                        "SOAP rejected param_groups (%s): falling back to a "
                        "FLAT parameter list. Per-group lr and weight decay "
                        "are NOT in effect for this run -- %d group(s) were "
                        "discarded. Do not draw group-wise conclusions from "
                        "it without fixing the SOAP signature first.",
                        exc, len(param_groups),
                    )
                    self.opt = SOAP(self.model.parameters(), lr=lr)
            else:
                hd, hnd, ad, and_ = _split_decay_groups(
                    self.model,
                    hidden_filter=_matrix_optimizer_filter(matrix_optimizer_scope, include_embed_default=False),
                )
                matrix_wd = float(matrix_weight_decay)
                aux_wd = float(aux_weight_decay)
                soap_groups = [{"params": hd, "weight_decay": matrix_wd}]
                adam_groups = [
                    {"params": hnd, "weight_decay": 0.0, "lr": float(lr)},
                    {"params": ad, "weight_decay": aux_wd, "lr": float(lr)},
                    {"params": and_, "weight_decay": 0.0, "lr": float(lr)},
                ]
                if weight_decay_mode == "soda" and soda_scope == "hidden_matrix_only":
                    for group in adam_groups:
                        group["weight_decay"] = 0.0
                    use_soda_weight_decay = mark_soda_weight_decay_groups(
                        soap_groups,
                        start_step=soda_start_step,
                        force=True,
                    )
                else:
                    soap_soda = _mark_soda(soap_groups)
                    adam_soda = _mark_soda(adam_groups, hidden_group_indices=())
                    use_soda_weight_decay = soap_soda or adam_soda
                self.opt = _ChainedOptimizer(
                    [
                        SOAP(soap_groups, lr=lr, weight_decay=0.0),
                        torch.optim.AdamW(adam_groups, lr=lr),
                    ]
                )
        else:
            raise ValueError(
                f"Unknown optimizer {optimizer!r}. Supported: nadamw, adamw, muon, aurora, soap"
            )

        if use_soda_weight_decay:
            self.opt = SODAWeightDecayWrapper(self.opt)
  # Snapshot the CONFIGURED param-group hyperparameters before any checkpoint
  # restore can overwrite them. See _reapply_configured_param_group_hparams.
        self._configured_param_group_hparams = self._snapshot_param_group_hparams()
  # Gradient-clip scope. Muon/Aurora take the POLAR FACTOR of the gradient,
  # which is scale-invariant, so a global norm clip is exactly inert for that
  # group while still inflating the norm the clip decides on — measured on
  # checkpoint_000122 over 8 real batches: ||g|| aurora 3.648, adamw 12.137,
  # global 12.673, i.e. the AdamW group carries 91.7% of the global norm^2 on
  # 71.4% of the parameters and the global norm sits 4.4% above the only
  # quantity the clip can actually move. Clip on the AdamW group alone, for
  # both the fixed cap and the adaptive z-score EMA.
        self._matrix_clip_params, clipped_params = split_matrix_and_clipped_params(
            self.model,
            optimizer=str(optimizer),
            matrix_optimizer_scope=str(matrix_optimizer_scope),
        )
  # The optimizer's own groups are the ground truth for who gets the
  # scale-invariant update. The predicate above is the one that BUILT those
  # groups, so this can only fire if the two are edited apart — in which case
  # the clip would be silently mis-scoped, which is the exact defect class
  # this change exists to remove. Fail loudly at construction instead.
        expected_matrix_ids = _optimizer_matrix_param_ids(self.opt)
        if {id(p) for p in self._matrix_clip_params} != expected_matrix_ids:
            raise ValueError(
                "grad-clip scope disagrees with the optimizer's matrix groups: "
                f"{len(self._matrix_clip_params)} param(s) from "
                f"matrix_optimizer_scope={matrix_optimizer_scope!r} vs "
                f"{len(expected_matrix_ids)} flagged use_muon/use_aurora "
                f"(optimizer={optimizer!r})"
            )
  # None = nothing to exclude (or nothing left to clip), in which case zclip
  # keeps getting the model and behaviour is bit-identical to before. Also
  # keeps `_compute_grad_norm`'s `next(model.parameters())` off an empty
  # iterator. Stored as a sentinel rather than as `self.model` because
  # `apply_compile` REBINDS `self.model` further down this constructor.
        self._grad_clip_scope: _GradClipScope | None = (
            _GradClipScope(clipped_params)
            if self._matrix_clip_params and clipped_params
            else None
        )
        max_grad_norm = None if zclip_max_norm is None else float(zclip_max_norm)
        self.zclip = ZClip(
            mode="zscore",
            alpha=float(zclip_alpha),
            z_thresh=float(zclip_z_thresh),
            max_grad_norm=max_grad_norm,  # pyright: ignore[reportArgumentType] # zclip accepts None to disable the hard cap.
            clip_factor=float(zclip_clip_factor),
            warmup_steps=25,
        )
        self.writer = SummaryWriter(log_dir=str(log_dir or "tb"))
        self.step = 0
        self._tb_log_interval = max(1, int(tb_log_interval))
        self._prefetch_batches = bool(prefetch_batches)

        self.use_amp = bool(use_amp)
        self._amp_dtype = torch.bfloat16 if device.startswith("cuda") else None

        self.feature_dropout_p = float(feature_dropout_p)
  # Rebuild SF targets from sparse MultiPV labels at sample time, so an
  # SfTargetParams change applies to the SF-labelled rows already in the replay
  # window instead of waiting ~18h for it to turn over. On healthy data that is
  # ALL of them: every labelled row is written with sf_multipv_raw, so the
  # rebuild reaches 100% of the SF-labelled window and there is no mixture of
  # two target regimes. sf_rebuild_policy_frac reports the realized rate, and
  # any shortfall below sf_rebuild_wdl_frac is Stockfish-desync contamination
  # (a LOWER bound on it: docs/target_rebuildability.md), not a structural cost
  # of the rebuild.
  # False = use stored targets, bitwise identical to the pre-flag pipeline.
  # `set_sf_target_rebuild` flips it live.
        self.rebuild_sf_targets = bool(rebuild_sf_targets)
        self.sf_policy_sparse_ce = bool(sf_policy_sparse_ce)
        self.sf_target_params = sf_target_params or SfTargetParams()
  # Proof-of-effect for the rebuild: reported as sf_rebuild_*_frac.
        self._sf_rebuild_coverage = _SfRebuildCoverageAccumulator()
  # Offline-only: recompute categorical_target from stored outcome + sf_wdl so
  # categorical_blend_frac can be screened on existing shards (sidecar). Default
  # off = stored targets used unchanged.
        self.rebuild_categorical_target = bool(rebuild_categorical_target)
        self.categorical_target_params = categorical_target_params or CategoricalTargetParams()
        self._base_input_planes = int(LC0_FULL.num_planes)
  # Per-group dropout: (start_offset_from_base, num_planes, dropout_prob)
  # Groups: king_safety(10), pins(6), pawns(8), mobility(6), outposts(4).
  # Per-group overrides default to a -1 sentinel in TrialConfig (not None),
  # so treat any negative value as "fall back to global feature_dropout_p".
        _fdp = float(feature_dropout_p)

        def _resolve_fdp(v: float | None) -> float:
            return _fdp if v is None or v < 0 else float(v)

        self._feature_group_dropout = [
            (0, 10, _resolve_fdp(fdp_king_safety)),
            (10, 6, _resolve_fdp(fdp_pins)),
            (16, 8, _resolve_fdp(fdp_pawns)),
            (24, 6, _resolve_fdp(fdp_mobility)),
            (30, 4, _resolve_fdp(fdp_outposts)),
  # v2_threats block [34:63]. On v1 inputs (146 planes) the slice past
  # x.shape[1] is empty, so this entry is a no-op there.
            (34, 29, _fdp),
        ]
        self.w_policy = float(w_policy)
        self.w_soft = float(w_soft)
        self.soft_policy_min_tv = float(soft_policy_min_tv)
  # Validated HERE, at construction, so a bad yaml value fails the trial's
  # startup instead of raising (or silently inverting the target) inside the
  # first training step. `retemper_main_policy_target` re-checks it, because a
  # test or the offline rig can call the helper without a Trainer.
        self.policy_target_temp = float(policy_target_temp)
        retemper_main_policy_target(torch.ones(1, 2), temp=self.policy_target_temp)
        self.w_future = float(w_future)
        self.w_sf_own = float(w_sf_own)
        self.w_sf_own_regret = float(w_sf_own_regret)
        self.w_wdl = float(w_wdl)
        self.w_sf_move = float(w_sf_move)
        self.w_sf_eval = float(w_sf_eval)
        self.w_categorical = float(w_categorical)
        self.w_volatility = float(w_volatility)
        self.resid_channel_dropout = max(0.0, min(0.95, float(resid_channel_dropout)))
        if hasattr(self.model, "resid_channel_dropout"):
            setattr(self.model, "resid_channel_dropout", self.resid_channel_dropout)
        self.resid_channel_balance_weight = max(0.0, float(resid_channel_balance_weight))
        if hasattr(self.model, "resid_channel_balance_weight"):
            setattr(self.model, "resid_channel_balance_weight", self.resid_channel_balance_weight)
        self.w_sf_volatility = float(w_sf_volatility) if w_sf_volatility is not None else float(w_volatility)
        self.w_moves_left = float(w_moves_left)
        self.sf_wdl_frac = float(sf_wdl_frac)
        self.search_wdl_frac = float(search_wdl_frac)
        self.sf_wdl_conf_power = float(sf_wdl_conf_power)
        self.sf_wdl_draw_scale = float(sf_wdl_draw_scale)
        self.sf_wdl_temperature = float(sf_wdl_temperature)
        self.sf_search_dampen_sf_low = float(sf_search_dampen_sf_low)
        self.sf_search_dampen_sf_high = float(sf_search_dampen_sf_high)
        self.use_adjusted_wdl_target = bool(use_adjusted_wdl_target)
        self.adjusted_wdl_regret_source = str(adjusted_wdl_regret_source)
        self.adjusted_wdl_regret_scale = float(adjusted_wdl_regret_scale)
        self.adjusted_wdl_regret_cap = float(adjusted_wdl_regret_cap)
  # Terminal-proximal outcome share of the value target (train/losses.py).
  # `moves_left_max_plies` is NOT a knob of its own: it is the selfplay
  # `max_plies` cap the stored `moves_left` field was normalized by, and it is
  # captured at construction rather than pushed live because changing the cap
  # mid-run would retroactively re-interpret every row already in the replay
  # window. Do not move the cap while the transfer is on.
        self.wdl_terminal_outcome_plies = int(wdl_terminal_outcome_plies)
        self.wdl_terminal_outcome_full_plies = int(wdl_terminal_outcome_full_plies)
        self.wdl_terminal_outcome_sf_frac = float(wdl_terminal_outcome_sf_frac)
        self.moves_left_max_plies = float(moves_left_max_plies)

  # Data augmentation: mirror positions left-right (files) with given probability.
        self.mirror_prob = float(mirror_prob)

  # Optional torch.compile for training throughput. Failures and recompile
  # thrash are surfaced via apply_compile + CompileProbe instead of being
  # swallowed silently.
        self.model = apply_compile(
            self.model,
            mode=(compile_mode if use_compile else "off"),
            device=device,
        )
        self._compile_probe = CompileProbe()
        self._compile_probe.snapshot_baseline()
        self._compile_probe_steps_remaining = 10  # report after first 10 steps

  # Gradient accumulation
        self.accum_steps = max(1, int(accum_steps))

  # LR schedule: linear warmup, then either cosine restarts or a WSD-style
  # release cycle that stays flat until the final tail.
        self._peak_lr = float(lr)
        self._warmup_steps = int(warmup_steps)
        if warmup_lr_start is None:
            self._warmup_lr_start = max(0.0, float(lr_eta_min))
        else:
            self._warmup_lr_start = max(0.0, float(warmup_lr_start))
        self._lr_schedule = str(lr_schedule).lower()
        self._lr_release_cycle_steps = int(lr_release_cycle_steps)
        if self._lr_schedule in ("cosine", "cosine_warm_restarts", "warm_restarts"):
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.opt, T_0=int(lr_T0), T_mult=int(lr_T_mult), eta_min=float(lr_eta_min),
            )
        elif self._lr_schedule in ("sqrt_release", "wsd_sqrt", "wsd_sqrt_release"):
            self._scheduler = _SqrtReleaseLRScheduler(
                self.opt,
                cycle_steps=int(lr_release_cycle_steps),
                release_start_frac=float(lr_release_start_frac),
                min_scale=float(lr_release_min_scale),
                release_shape=str(lr_release_shape),
            )
        else:
            raise ValueError(
                f"Unknown lr_schedule {lr_schedule!r}. "
                "Supported: cosine, sqrt_release"
            )
        self._set_initial_lrs()

  # Stochastic Weight Averaging (SWA): maintain a running average of model
  # weights for smoother, more generalizable exported networks.
        self._swa_start = int(swa_start)
        self._swa_freq = max(1, int(swa_freq))
        self._swa_model: torch.optim.swa_utils.AveragedModel | None = None
        self._init_swa()

  # ⚑ THE ONLY ARTIFACT THAT NAMES THE REALIZED TARGET TEMPERATURE. Everything
  # else a reader might reach for reports a DIFFERENT number: `params.json` is
  # the LAUNCH config (and says nothing about what the constructor accepted),
  # `scripts/retarget_retrain.py` never writes one at all, and
  # `scripts/audit_targets.py`'s "production training target" row is rebuilt
  # from the flat `temperature` key -- the selfplay SAMPLING temperature, an
  # unrelated knob -- so it reads identically whether or not this one is live.
  # Without this line an arm whose value never reached `compute_loss` is
  # indistinguishable from one whose did reach it, which is this codebase's
  # signature defect (a value accepted and then silently ignored).
  #
  # ⚑ LAST STATEMENT OF `__init__`, and that placement is load-bearing. Review
  # found `eval_pinned_temp=1` written as a string LITERAL when the print sat
  # beside the assignment above -- a claim, not a measurement, in the one line
  # whose entire justification is that it reports realized values. It cannot be
  # a real read up there: `_loss_kwargs` touches `self.w_future` and every other
  # weight, so reading `_eval_loss_kwargs` mid-constructor raises
  # `AttributeError: 'Trainer' object has no attribute 'w_future'`. Down here all
  # three fields are reads of state the training step will actually use. Keep any
  # new attribute assignment ABOVE this print.
  #
  # UNCONDITIONAL on purpose: the control arm's `1.0` has to be a positive
  # record, not an absent line that could equally mean "old code".
  # `reshape_active` comes from `policy_target_temp_active`, the same predicate
  # `retemper_main_policy_target` gates its early return on, so the claim and
  # the arithmetic cannot drift apart.
  # `!r`, not `.6g`: review found `.6g` renders 1.0000001 as `1`, which is both
  # self-contradictory next to `reshape_active=True` and indistinguishable from
  # the control arm. `repr` is shortest-round-trip, so the printed text always
  # reconstructs the float that was installed.
  # print(), NOT logging.info() -- the trial actor installs no logging handler,
  # so an INFO record is discarded; see the export_swa comment below.
  # ⚑ NO `^` ANCHOR when grepping for this. Ray prefixes actor stdout with
  # `(train_trial pid=NNNN)` and an ANSI colour code, so an anchored grep
  # returns 0 matches on a real trial log and reads as "the arm never ran".
        print(
            f"[trainer] policy_target_temp={self.policy_target_temp!r} "
            f"reshape_active={policy_target_temp_active(self.policy_target_temp)} "
            f"eval_pinned_temp={self._eval_loss_kwargs['policy_target_temp']!r}",
            flush=True,
        )

    def _init_swa(self) -> None:
        """(Re)initialize SWA from current model weights.

        Must be called after any external weight load (bootstrap, salvage)
        because AveragedModel deep-copies at creation time.
        """
        if self._swa_start >= 0:  # 0 = start immediately, <0 = disabled
            self._swa_model = torch.optim.swa_utils.AveragedModel(self.model)
        else:
            self._swa_model = None

    def _should_log_step_scalars(self) -> bool:
        return (self.step % self._tb_log_interval) == 0

    def _warn_if_grad_norm_median_past_watch(self, metrics: TrainMetrics) -> None:
        """Fire the pre-committed I11 watch when the hard cap stops being a tail guard."""
        max_grad_norm = getattr(self.zclip, "max_grad_norm", None)
        if max_grad_norm is None or metrics.grad_norm_samples <= 0:
            return
        if metrics.grad_norm_median <= GRAD_NORM_MEDIAN_WATCH:
            return
        logging.getLogger(__name__).warning(
            "grad-norm median %.3f over %d step(s) is past the pre-committed "
            "watch threshold %.2f with zclip_max_norm=%.2f (clip rate %.1f%%, "
            "hard-clip rate %.1f%%): the hard cap is acting as an LR cap, not a "
            "tail guard — re-set it (docs/rl_loop_audit.md I11)",
            metrics.grad_norm_median, metrics.grad_norm_samples,
            GRAD_NORM_MEDIAN_WATCH, float(max_grad_norm),
            100.0 * metrics.grad_clip_rate, 100.0 * metrics.grad_hard_clip_rate,
        )

    @property
    def _grad_clip_target(self) -> Any:
        """Whatever `ZClip.step` is handed: the AdamW scope, or the whole model.

        Resolved on every call so it follows `self.model` through
        `apply_compile`'s rebinding, exactly as the pre-scope code did.
        """
        return self.model if self._grad_clip_scope is None else self._grad_clip_scope

    def _matrix_grad_norm(self) -> float:
        """2-norm of the scale-invariant matrix group's grads (0.0 when empty).

        Accumulated in float32 regardless of parameter dtype: the number has to
        be comparable with the AdamW group's, and it doubles as the non-finite
        guard in `_run_optimizer_step`, so it must not itself overflow. A
        2-norm is non-finite exactly when some element is, which is what makes
        that guard free — it rides the sync `_zclip_step` already pays for.
        """
        norms = [
            param.grad.detach().to(torch.float32).norm(2)
            for param in self._matrix_clip_params
            if param.grad is not None
        ]
        if not norms:
            return 0.0
        return float(torch.linalg.vector_norm(torch.stack(norms)))

    def _zclip_stats_snapshot(self) -> tuple[bool, Any, Any, list[float]]:
        """zclip's whole adaptive state: (initialized, mean, var, warmup buffer)."""
        return (
            bool(getattr(self.zclip, "initialized", False)),
            getattr(self.zclip, "mean", None),
            getattr(self.zclip, "var", None),
            list(getattr(self.zclip, "buffer", None) or []),
        )

    def _restore_zclip_stats(self, snapshot: tuple[bool, Any, Any, list[float]]) -> None:
        """Roll zclip's adaptive state back to a snapshot.

        `_run_optimizer_step` uses this when it discards a step, because
        `ZClip.step` folds whatever norm it computed into the EMA
        unconditionally and a single nan there is PERMANENT: `mean` becomes
        nan, every later `z = (norm - nan) / std` is nan, `nan > z_thresh` is
        False, so `_compute_clip_val` returns None for the rest of the run and
        the adaptive clipper is silently switched off while the fixed cap goes
        on reporting normally. During warmup the same value lands in `buffer`
        and `_initialize_ema` averages it, seeding the EMA nan from the start.
        A discarded step is not a gradient magnitude and must not shape the
        statistics.
        """
        self.zclip.initialized, self.zclip.mean, self.zclip.var, self.zclip.buffer = snapshot

    def zclip_state_dict(self) -> dict[str, Any]:
        """zclip's adaptive state, as plain JSON-able scalars for a checkpoint.

        Built from `_zclip_stats_snapshot` rather than reading `self.zclip`
        again, so "zclip's adaptive state" has ONE definition in this file. A
        second field list here would drift the moment the upstream class grows
        one, and the failure would be silent: the missing field simply resets
        every restart, which is the very bug this method exists to close.
        """
        initialized, mean, var, buffer = self._zclip_stats_snapshot()
        return {
            "initialized": bool(initialized),
            "mean": None if mean is None else float(mean),
            "var": None if var is None else float(var),
            "buffer": [float(v) for v in buffer],
            "warmup_steps": int(getattr(self.zclip, "warmup_steps", 0)),
        }

    def load_zclip_state(self, state: Any) -> bool:
        """Restore zclip's EMA from a checkpoint payload. True when it took.

        Returns False -- leaving the fresh warmup state exactly as the
        constructor built it -- for every reason a restore should not happen,
        and says which in the log. The caller reports the outcome either way;
        a restore that quietly does nothing is the failure mode this whole
        change is about.

        ⚑ A non-finite `mean` or `var` is REFUSED rather than restored. Per
        `_restore_zclip_stats`: one nan in the EMA is permanent, because every
        later `z = (norm - nan) / std` is nan, `nan > z_thresh` is False, and
        `_compute_clip_val` returns None for the rest of the run -- the
        adaptive clipper is off while the fixed cap goes on reporting
        normally. Persisting the state means a poisoned EMA would otherwise
        survive the restart that used to clear it, so this path can only be
        added together with the refusal.
        """
        log = logging.getLogger(__name__)
        if state is None:
            log.info(
                "zclip: checkpoint carries no adaptive state; starting a fresh "
                "%d-step EMA warmup (expected for checkpoints written before "
                "this key existed)",
                int(getattr(self.zclip, "warmup_steps", 0)),
            )
            return False
        if not isinstance(state, dict):
            log.warning(
                "zclip: checkpoint adaptive state is %s, not a dict; starting "
                "a fresh EMA warmup", type(state).__name__,
            )
            return False

        try:
            initialized = bool(state["initialized"])
            raw_mean = state.get("mean")
            raw_var = state.get("var")
            mean = None if raw_mean is None else float(raw_mean)
            var = None if raw_var is None else float(raw_var)
            buffer = [float(v) for v in (state.get("buffer") or [])]
        except (KeyError, TypeError, ValueError) as exc:
            log.warning(
                "zclip: checkpoint adaptive state is malformed (%s); starting "
                "a fresh EMA warmup", exc,
            )
            return False

        if initialized and (mean is None or var is None):
            log.warning(
                "zclip: checkpoint claims an initialized EMA but carries "
                "mean=%r var=%r; starting a fresh EMA warmup", mean, var,
            )
            return False
        finite = [v for v in (mean, var, *buffer) if v is not None]
        if not all(math.isfinite(v) for v in finite):
            log.warning(
                "zclip: REFUSING a non-finite checkpoint EMA (mean=%r var=%r); "
                "a nan here disables the adaptive clipper permanently while "
                "the fixed cap keeps reporting normally. Starting a fresh EMA "
                "warmup", mean, var,
            )
            return False

        self._restore_zclip_stats((initialized, mean, var, buffer))
        if initialized and mean is not None and var is not None:
            log.info(
                "zclip: restored adaptive EMA from checkpoint "
                "(mean=%.4f var=%.4f std=%.4f); the adaptive branch is live "
                "from the first step of this run instead of after a %d-step "
                "warmup", mean, var, var ** 0.5,
                int(getattr(self.zclip, "warmup_steps", 0)),
            )
        else:
            log.info(
                "zclip: restored a PARTIAL warmup buffer from checkpoint "
                "(%d/%d norms collected); the EMA was not yet initialized "
                "when the checkpoint was written",
                len(buffer), int(getattr(self.zclip, "warmup_steps", 0)),
            )
        return True

    def _zclip_step(self, *, collect_stats: bool) -> tuple[float, dict[str, float] | None]:
        if not collect_stats:
            return float(self.zclip.step(self._grad_clip_target)), None

        was_initialized = bool(getattr(self.zclip, "initialized", False))
        mean_raw = getattr(self.zclip, "mean", 0.0)
        var_raw = getattr(self.zclip, "var", 0.0)
        mean = float(mean_raw if mean_raw is not None else 0.0)
        var = float(var_raw if var_raw is not None else 0.0)
        total_norm = float(self.zclip.step(self._grad_clip_target))
        clip_val = None
        if was_initialized:
            std = var ** 0.5
            mode = str(getattr(self.zclip, "mode", "zscore"))
            z_thresh = float(getattr(self.zclip, "z_thresh", 0.0))
            if mode == "percentile":
                threshold = mean + z_thresh * std
                if total_norm > threshold:
                    clip_val = threshold
            elif mode == "zscore":
                z_score = (total_norm - mean) / (std + float(getattr(self.zclip, "eps", 1e-8)))
                if z_score > z_thresh:
                    clip_option = str(getattr(self.zclip, "clip_option", "adaptive_scaling"))
                    if clip_option == "adaptive_scaling":
                        eta = z_score / z_thresh
                        clip_val = (mean + (z_thresh * std) / eta) * float(getattr(self.zclip, "clip_factor", 1.0))
                    elif clip_option == "mean":
                        clip_val = mean
        adaptive_clip = float(clip_val) if clip_val is not None else total_norm
        max_grad_norm = getattr(self.zclip, "max_grad_norm", None)
        effective_clip = adaptive_clip
        if max_grad_norm is not None:
            effective_clip = min(effective_clip, float(max_grad_norm))

        clipped = effective_clip < total_norm
        stats = {
            "total_norm": total_norm,
            "effective_clip": float(effective_clip),
            "adaptive_clip": 1.0 if clip_val is not None and adaptive_clip < total_norm else 0.0,
            "hard_clip": 1.0 if max_grad_norm is not None and clipped and effective_clip == float(max_grad_norm) else 0.0,
            "clipped": 1.0 if clipped else 0.0,
        }
        return total_norm, stats

    @property
    def _loss_kwargs(self) -> dict[str, Any]:
        return {
            "w_policy": self.w_policy, "w_soft": self.w_soft, "w_future": self.w_future,
            "w_sf_own": self.w_sf_own, "w_sf_own_regret": self.w_sf_own_regret,
            "soft_policy_min_tv": self.soft_policy_min_tv,
            "policy_target_temp": self.policy_target_temp,
            "w_wdl": self.w_wdl, "w_sf_move": self.w_sf_move, "w_sf_eval": self.w_sf_eval,
            "w_categorical": self.w_categorical, "w_volatility": self.w_volatility,
            "w_sf_volatility": self.w_sf_volatility, "w_moves_left": self.w_moves_left,
            "sf_wdl_frac": self.sf_wdl_frac, "search_wdl_frac": self.search_wdl_frac,
            "sf_wdl_conf_power": self.sf_wdl_conf_power,
            "sf_wdl_draw_scale": self.sf_wdl_draw_scale,
            "sf_wdl_temperature": self.sf_wdl_temperature,
            "sf_search_dampen_sf_low": self.sf_search_dampen_sf_low,
            "sf_search_dampen_sf_high": self.sf_search_dampen_sf_high,
            "use_adjusted_wdl_target": self.use_adjusted_wdl_target,
            "adjusted_wdl_regret_source": self.adjusted_wdl_regret_source,
            "adjusted_wdl_regret_scale": self.adjusted_wdl_regret_scale,
            "adjusted_wdl_regret_cap": self.adjusted_wdl_regret_cap,
            "wdl_terminal_outcome_plies": self.wdl_terminal_outcome_plies,
            "wdl_terminal_outcome_full_plies": self.wdl_terminal_outcome_full_plies,
            "wdl_terminal_outcome_sf_frac": self.wdl_terminal_outcome_sf_frac,
            "moves_left_max_plies": self.moves_left_max_plies,
            "sf_sparse_params": self.sf_target_params if self.sf_policy_sparse_ce else None,
        }

    @property
    def _eval_loss_kwargs(self) -> dict[str, Any]:
        """``_loss_kwargs`` with the policy-target RESHAPE pinned off.

        ⚑ THE RULER MUST NOT MOVE WITH THE ARM. `policy_target_temp` reshapes
        the target `policy_ce` is measured against, and `CE = H(target) +
        KL(target||model)`, so an UNCHANGED model reads a different `policy_ce`
        purely from the reshape. The direction is model-dependent -- flattening
        raises `H` but can lower `KL` by more, and it has been measured falling
        on a random-logit fixture -- so no magnitude is quoted here; the pin
        needs only that the number MOVES. Sharing `_loss_kwargs` between the
        training step and the holdout/EMA eval would therefore make every
        arm-vs-baseline CE comparison a comparison of two different rulers --
        the exact defect docs/experiment_ledger.md logs as "a ruler change must
        invalidate its records". Eval scores the model against the target the
        shards ACTUALLY carry, identically in every arm.

        Only the target-SHAPE knob is pinned. Loss WEIGHTS stay as configured:
        they scale a term without redefining it, so they leave the per-term
        columns (`policy_ce`, `wdl_ce`, ...) comparable, and pinning them would
        make `total` stop matching the trained objective.
        """
        return {**self._loss_kwargs, "policy_target_temp": 1.0}

    def _amp_context(self):
        # Pinned to bf16: training has no GradScaler, so an FP16 fallback
        # would silently underflow gradients on non-BF16 CUDA cards. The
        # ``inference_autocast`` helper would auto-fallback to FP16 there.
        return inference_autocast(device=self.device, enabled=self.use_amp, dtype="bf16")

    @staticmethod
    def _extract_loss_scalars(
        losses: dict[str, torch.Tensor], *,
        total_override: torch.Tensor | None = None,
        total_scale: float = 1.0,
    ) -> dict[str, float]:
        """Extract loss scalars from compute_loss output in one host transfer.

        ``total_override`` preserves gradient-accumulation reporting semantics:
        materialize the divided loss, then apply ``total_scale`` on the host.
        """
        keys = list(losses)
        if not keys:
            return {}
        stacked = torch.stack([
            (total_override if k == "total" and total_override is not None else losses[k]).detach()
            for k in keys
        ])
        values = stacked.tolist()
        return {
            ("loss" if key == "total" else key): (
                float(value) * total_scale if key == "total" else float(value)
            )
            for key, value in zip(keys, values, strict=True)
        }

    def _log_metrics(self, metrics: TrainMetrics, tag: str) -> None:
        """Log every SCALAR TrainMetrics field to TensorBoard under ``tag``.

        `eval_ruler` is an identity, not a measurement: it names the
        instrument the rest of these numbers came off. TensorBoard takes
        floats, and the loop used to assume every field was one, so the
        non-numeric field is skipped here rather than left to raise
        mid-evaluation. It reaches an operator through the checkpoint's
        trial_meta.json, `best.json` and the handover log line instead.
        """
        for field_name, value in dataclasses.asdict(metrics).items():
            if isinstance(value, str):
                continue
            self.writer.add_scalar(f"{tag}/{field_name}", float(value), self.step)

    @staticmethod
    def _build_metrics(
        sums: dict[str, float],
        acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
        n: float,
        **extras: Any,
    ) -> TrainMetrics:
        """Common tail of ``_compute_metrics`` and ``train_steps`` — averages
        loss sums by ``n`` and computes per-head accuracy ratios from
        (numerator, denominator) GPU-tensor pairs in ``acc_sums``. The
        tensors get materialized to CPU floats here in a single sync per
        head rather than per-microbatch.
        """
        def _acc(name: str) -> float:
            val = acc_sums.get(name)
            if val is None:
                return 0.0
            num, den = val
            den_f = float(den.item())
            if den_f <= 0:
                return 0.0
            return float(num.item()) / den_f

  # The denominator `_acc` divided by, published so a 0.0 accuracy can be told
  # apart from an absent one. Same accumulated tensor `_acc` reads, so this is
  # one extra sync per head, not an extra reduction.
        def _den(name: str) -> float:
            val = acc_sums.get(name)
            return 0.0 if val is None else float(val[1].item())

        return TrainMetrics(
            **_loss_sums_to_metric_kwargs(sums, n),  # dict[str,float] splat covers int fields (step counters) at runtime
            sf_move_acc=_acc("sf_move_acc"),
            sf_move_acc_top5=_acc("sf_move_acc_top5"),
            policy_own_acc_top1=_acc("policy_own_acc_top1"),
            policy_own_acc_top5=_acc("policy_own_acc_top5"),
            policy_own_acc_rows=_den("policy_own_acc_top1"),
            policy_future_acc_rows=_den("policy_future_acc_top1"),
            policy_future_acc_top1=_acc("policy_future_acc_top1"),
            policy_future_acc_top5=_acc("policy_future_acc_top5"),
            **extras,
        )

    def _prepare_host_arrays(
        self,
        arrs: dict[str, np.ndarray],
        *,
        rng: np.random.Generator,
        mirror_prob: float,
        rebuild_sf_targets: bool = False,
        coverage: _SfRebuildCoverageAccumulator | None = None,
    ) -> dict[str, np.ndarray]:
        """Target rebuilds, payload pruning, history selection and mirroring.

        Everything the host-side batch pipeline does AFTER the rows have been
        chosen. Shared by the sampling path and the deterministic full-pass
        path so the two cannot drift into scoring differently-shaped batches.
        Every step here is a pure function of ``arrs`` except the mirror, which
        is a no-op at ``mirror_prob <= 0`` and does not touch ``rng`` there.

        ``rebuild_sf_targets`` gates the SF target rebuild and DEFAULTS OFF —
        fail-closed. Only ``_sample_batch_host``, the training path, opts in
        with True (and even then nothing rebuilds unless
        ``self.rebuild_sf_targets`` is on). The default used to be True with
        the ruler protected by a per-callsite ``False`` pin, which made every
        FUTURE producer a hazard: a new caller that took no position would
        silently rebuild — i.e. retarget its measurement whenever the flag
        was on. Defaulting False inverts that: forgetting the kwarg can only
        under-apply a training experiment (visible as ``sf_rebuild_*_frac``
        staying 0), never move a ruler
        (test_prepare_host_arrays_defaults_to_stored_targets).

        ``coverage`` is the sink the rebuild's row counts land in, defaulting
        to the trainer-wide one that the TRAINING metrics drain. It has to be
        selectable per measurement, not per thread: an eval reaches this method
        on its own thread AND on a prefetch thread it owns, while training is
        doing the same, and `_compute_metrics` drains at the end. Sharing one
        sink would let the async holdout eval — which in production runs
        concurrently with the next iteration's training (`async_eval.py`, and
        `distributed_async_test_eval: true`) — drain counts the training path
        produced, reporting them on the `eval` row and under-reporting the
        `train` row by an unknowable amount. That would break both things the
        metric exists for: the training row's proof-of-effect, and the rule
        that a non-zero value on the ruler's row means the RULER rebuilt.
        """
        sink = coverage if coverage is not None else self._sf_rebuild_coverage
        if rebuild_sf_targets and self.rebuild_sf_targets:
            arrs, rebuilt_coverage = rebuild_sf_targets_in_arrays(
                arrs, params=self.sf_target_params,
            )
            sink.add(rebuilt_coverage)
        if self.rebuild_categorical_target:
            arrs = rebuild_categorical_target_in_arrays(
                arrs, params=self.categorical_target_params,
            )
  # The VALUE-half desync check, derived HERE because its two inputs
  # (`sf_multipv_raw`, `sf_label_meta`) never reach the GPU: the first is
  # pruned three lines below in production and the second is not in the collate
  # spec at all. Deriving it before the prune turns 960 B/row + 24 B/row of raw
  # blocks into two (B,) float32 vectors — 4 KB at batch_size 512, 0.03 % of
  # the `x` tensor — so the check rides the same always-on channel as
  # `sf_labelled_no_multipv_frac` instead of being gated behind
  # `sf_policy_sparse_ce`, which defaults False and is in no config file. That
  # gating is exactly how `sf_rebuild_policy_frac` came to read 0.0 through
  # three desync episodes; see `replay/shard.py::sf_eval_pv_orphan_flags`.
  #
  # Placed before the mirror as well as before the prune: mirroring permutes
  # move indices, and the (cp, mate) columns this compares are mirror-invariant,
  # so the two orders agree — but only this one is also independent of the
  # mirror ever learning to touch a score column.
        orphaned, checked = sf_eval_pv_orphan_flags(arrs)
        arrs[SF_EVAL_PV_ORPHAN_FIELD] = orphaned
        arrs[SF_EVAL_PV_CHECKED_FIELD] = checked
        if not self.sf_policy_sparse_ce:
            # Keep the H2D payload small: the (B, 40, 4) int32 candidate block
            # only rides to the GPU when the sparse loss consumes it. (Dropped
            # AFTER the rebuild hook, which also reads it.)
            #
            # Its `has_` flag is NOT dropped with it, and that is deliberate.
            # It is a (B,) float32 vector — 2 KB at batch_size 512, 0.017 % of
            # the `x` tensor alone (512x175x8x8 float16 = 11.5 MB), and the
            # smallest payload any batch field has — and it is the whole input to
            # `sf_labelled_no_multipv_frac`, the always-on desync detector.
            # Pruning it here would gate that column behind
            # `sf_policy_sparse_ce`, which defaults False and is in no config
            # file: the column would read 0.0 in production, which is also what
            # a healthy window reads. That is precisely the defect
            # `sf_rebuild_policy_frac` already has and the reason a 25-day
            # desync went unseen in-loop. `sparse_sf_policy_ce` tolerates the
            # flag arriving without its block (it needs `sf_multipv_raw` too
            # and returns an all-zero eligibility mask when any input is
            # missing), so the loss path is unchanged either way.
            arrs.pop("sf_multipv_raw", None)
        arrs = select_input_history_arrays(
            arrs,
            input_history_encoding=self._input_history_encoding,
        )
        return maybe_mirror_batch_arrays(
            arrs,
            rng=rng,
            prob=mirror_prob,
            input_history_encoding=self._input_history_encoding,
        )

    def _sample_batch_host(
        self,
        buf: ReplayBuffer,
        *,
        batch_size: int,
        mirror_prob: float,
        coverage: _SfRebuildCoverageAccumulator | None = None,
    ) -> dict[str, np.ndarray] | list:
        if hasattr(buf, "sample_batch_arrays"):
            return self._prepare_host_arrays(
                buf.sample_batch_arrays(batch_size),
                rng=buf.rng,
                mirror_prob=mirror_prob,
  # The one explicit opt-in: sampled batches are the TRAINING
  # distribution, the thing the rebuild exists to retarget.
                rebuild_sf_targets=True,
                coverage=coverage,
            )

        samples = buf.sample_batch(batch_size)
        samples = select_input_history_samples(
            samples,
            input_history_encoding=self._input_history_encoding,
        )
        return maybe_mirror_samples(
            samples,
            rng=buf.rng,
            prob=mirror_prob,
            input_history_encoding=self._input_history_encoding,
        )

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
        coverage: _SfRebuildCoverageAccumulator | None = None,
    ) -> Iterator[dict[str, torch.Tensor]]:
        n = int(count)
        if n <= 0:
            return
        if not self._prefetch_batches or n == 1:
            for _ in range(n):
                host_batch = self._sample_batch_host(
                    buf, batch_size=batch_size, mirror_prob=mirror_prob, coverage=coverage,
                )
                yield self._host_batch_to_tensors(host_batch)
            return

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                self._sample_batch_host,
                buf,
                batch_size=batch_size,
                mirror_prob=mirror_prob,
                coverage=coverage,
            )
            for idx in range(n):
                host_batch = future.result()
                if idx + 1 < n:
                    future = pool.submit(
                        self._sample_batch_host,
                        buf,
                        batch_size=batch_size,
                        mirror_prob=mirror_prob,
                        coverage=coverage,
                    )
                yield self._host_batch_to_tensors(host_batch)

    def _full_pass_host_batch(
        self, buf: ReplayBuffer, *, start: int, stop: int,
        coverage: _SfRebuildCoverageAccumulator | None = None,
    ) -> dict[str, np.ndarray]:
        """One deterministic chunk of rows, host-side, ready to collate.

        ``mirror_prob=0.0``: mirroring is a random augmentation, and a ruler
        that flips a random half of its positions is not a fixed ruler. The
        eval path already passed 0.0 here; it is pinned rather than plumbed so
        the full pass cannot acquire an rng dependency by configuration.

        ``rebuild_sf_targets=False`` is also the DEFAULT (fail-closed since
        the PR #283 review follow-up), but stays pinned explicitly here for
        the same reason: the ruler must not depend on a default that a
        refactor could flip back. With the rebuild on, this path would (a) score the
        model against REBUILT `sf_policy_target` / `sf_wdl` and (b) drop
        `w_sf_own` and `w_sf_volatility` from `total`, because the rebuild
        masks the two cross-ply targets it cannot move. Both change what
        `test_loss` MEANS with no `holdout_generation` bump, and the second
        moves it DOWN — a definitional fall that reads as improvement and that
        `_update_best_model` would happily promote across (the G16 / PR #277
        shape, docs/rl_loop_audit.md).

        A ruler that re-parameterises itself with the training target also
        cannot measure that target's effect: the bump would fire exactly when
        the experiment starts, so the holdout could never read it. Pinning
        keeps the pre-flip and post-flip numbers on one instrument. The cost —
        the SF-derived legs then score against capture-time targets while
        training uses rebuilt ones — is stated in
        docs/target_rebuildability.md; those legs are contaminated FOR THE
        DURATION and the experiment's yardstick must be an external one.

        ``coverage`` is still forwarded even though the pin above means nothing
        can ever be added to it here. That is deliberate: it is what keeps
        "``sf_rebuild_*`` non-zero on the `eval` row ⇒ the ruler rebuilt" a
        statement that can actually FAIL. If a later edit removes the pin, the
        counts land on the eval's OWN sink and the alarm fires on the eval row;
        wired to the trainer-wide sink instead, the same edit would have shown
        up as training-row counts going missing, which reads as nothing at all.
        """
        return self._prepare_host_arrays(
            buf.rows_slice_arrays(start, stop), rng=buf.rng, mirror_prob=0.0,
            rebuild_sf_targets=False, coverage=coverage,
        )

    def _iter_full_pass_batches(
        self, buf: ReplayBuffer, *, batch_size: int,
        coverage: _SfRebuildCoverageAccumulator | None = None,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Every row of ``buf`` exactly once, oldest first, in fixed order.

        Deliberately does NOT go through ``buf.sample_batch_arrays`` -- that
        method draws with replacement, WDL-rebalances and priority-weights, and
        those are precisely the three things a frozen ruler must not do
        (docs/rl_loop_audit.md G14).

        The final batch is ragged unless the row count divides evenly; the
        consumer weights by ``x.shape[0]``. Under ``torch.compile`` that second
        shape costs one extra graph on the first pass and is cached after.

        The bounds are computed once, up front. On the async holdout path this
        generator runs while the next iteration ingests, so a buffer that is
        still GROWING can shift underneath it -- the same exposure the sampler
        had, and inert in production because a frozen holdout takes no new rows
        (`_ingest_train_arrays` skips it). ``rows_slice_arrays`` clamps, so the
        worst case is a short last batch rather than an IndexError.
        """
        bs = int(batch_size)
        if len(buf) <= 0 or bs <= 0:
            return
        if not hasattr(buf, "batch_row_bounds"):
  # Loud rather than a silent fall back to sampling: a ruler that quietly
  # stops being a full pass is the failure this method exists to remove.
            raise TypeError(
                f"{type(buf).__name__} cannot be read in a fixed row order "
                "(no batch_row_bounds); a full-pass eval needs one that can",
            )
        bounds = buf.batch_row_bounds(bs)

        if not self._prefetch_batches or len(bounds) == 1:
            for start, stop in bounds:
                yield self._host_batch_to_tensors(
                    self._full_pass_host_batch(buf, start=start, stop=stop, coverage=coverage),
                )
            return

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                self._full_pass_host_batch,
                buf, start=bounds[0][0], stop=bounds[0][1], coverage=coverage,
            )
            for idx in range(len(bounds)):
                host_batch = future.result()
                if idx + 1 < len(bounds):
                    nxt = bounds[idx + 1]
                    future = pool.submit(
                        self._full_pass_host_batch,
                        buf, start=nxt[0], stop=nxt[1], coverage=coverage,
                    )
                yield self._host_batch_to_tensors(host_batch)

    @classmethod
    def eval_ruler_id_for(
        cls, *, batch_size: int, steps: int, mirror_prob: float, full_pass: bool,
    ) -> str:
        """Identity of the measurement `_compute_metrics` performs.

        A CLASSMETHOD, and that is load-bearing rather than tidiness: the id
        depends on nothing but code and the four arguments, so an operator's
        deploy check (and every test here) can ask for it without building a
        Trainer, a model or a device. The earlier instance form forced callers
        to assemble a stand-in object whose attributes had to be kept in step
        with the covered set BY HAND -- the same class of hand-maintained list
        this method exists to stop relying on. `type(self)` at the call site
        keeps a subclass that overrides part of the measurement fingerprinted
        as itself.

        **Covered: derived, not listed.** ``call_closure`` walks the call
        graph from `_compute_metrics` and returns every frame defined in this
        module that the taken branch can reach -- 19 for the full pass, from row
        selection through host-array prep, target rebuilds, history selection
        and tensor collation to the metric-assembly tail
        (`_extract_loss_scalars`, `_build_metrics`,
        `_loss_sums_to_metric_kwargs`) where the pooling denominator is
        actually APPLIED. Two hand-written lists were wrong here before: the
        first let a pooling change through, the second let `test_loss` be
        DOUBLED with every test in the suite green. A list is a claim about
        the call graph; this reads the call graph.

        **NOT covered -- the module edge, and it is real.** Recursion stops at
        anything defined outside this module: `compute_loss`'s body, the
        encoders, torch/numpy. Module-level CONSTANTS are not hashed either
        (`_RAW_SUM_LOSS_KEYS` decides which loss keys are row-summed at all),
        because the ones on this path include `_TRAIN_METRICS_FIELDS`, derived
        from the `TrainMetrics` dataclass -- hashing it would bump the ruler
        every time an unrelated diagnostic FIELD is added, which provably
        cannot change `loss`. And the loss WEIGHTS are excluded for the reason
        in docs/rl_loop_audit.md L15: `_loss_kwargs` carries the PID's
        realized `sf_wdl_frac`, so hashing it cannot tell a 0.006 controller
        excursion from a 0.10 config step. All of these are named in the
        ledger's not-covered list; none of them is implied to be covered here.

        Two arguments are PINNED in the full-pass branch rather than passed
        through, because neither reaches that measurement: ``steps`` is a
        sampled-batch budget the pass ignores, and ``_full_pass_host_batch``
        hard-codes ``mirror_prob=0.0`` so a fixed ruler cannot flip a random
        half of its positions. A knob that cannot move the number must not be
        able to move its identity, or the handover fires on nothing.
        """
        measured_by = call_closure(
            cls._compute_metrics, owner=cls,
  # Prune the branch not taken, so the sampled path cannot move the
  # full-pass ruler (and vice versa). This is the only name the closure
  # is told about, and it is the one the `mode` field already declares.
            skip=(
                (cls._iter_prefetched_batches,) if full_pass
                else (cls._iter_full_pass_batches,)
            ),
        )
        if full_pass:
            return eval_ruler_id(
                mode="full_pass", batch_size=int(batch_size), steps=0,
                mirror_prob=0.0, measured_by=measured_by,
            )
        return eval_ruler_id(
            mode="sampled", batch_size=int(batch_size), steps=int(steps),
            mirror_prob=float(mirror_prob), measured_by=measured_by,
        )

    def reset_optimizer_reference_weights(self) -> None:
        """Refresh optimizer reference weights after model-only loads."""
        reset_anchors = getattr(self.opt, "reset_anchors", None)
        if callable(reset_anchors):
            reset_anchors()

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

    def _uses_train_window_release_cycle(self) -> bool:
        return (
            self._lr_schedule in ("sqrt_release", "wsd_sqrt", "wsd_sqrt_release")
            and self._lr_release_cycle_steps <= 0
            and self.step >= self._warmup_steps
        )

    def _set_train_window_release_lr(self, *, local_step: int, cycle_steps: int) -> None:
        scheduler: Any = self._scheduler
        scheduler.step_window(int(local_step), cycle_steps=max(1, int(cycle_steps)))

    def set_lr_release_config(
        self,
        *,
        cycle_steps: Any | None = None,
        release_start_frac: Any | None = None,
        min_scale: Any | None = None,
        release_shape: Any | None = None,
    ) -> None:
        """Live-update WSD release knobs without rebuilding optimizer state.

        Switching scheduler families still requires a trainer restart; this only
        mutates the release scheduler that is already active.
        """
        if not isinstance(self._scheduler, _SqrtReleaseLRScheduler):
            return
        if cycle_steps is not None:
            self._lr_release_cycle_steps = int(cycle_steps)
            self._scheduler.cycle_steps = int(cycle_steps)
        if release_start_frac is not None:
            self._scheduler.release_start_frac = min(1.0, max(0.0, float(release_start_frac)))
        if min_scale is not None:
            self._scheduler.min_scale = min(1.0, max(0.0, float(min_scale)))
        if release_shape is not None:
            shape = str(release_shape).lower()
            if shape not in ("sqrt", "cosine"):
                raise ValueError(f"Unsupported lr_release_shape {release_shape!r}; expected sqrt or cosine")
            self._scheduler.release_shape = shape

    def _update_lr(self) -> None:
        """Apply linear warmup, then hand off to the post-warmup scheduler."""
        if self.step < self._warmup_steps:
  # Called after optimizer.step(); set the LR for the *next* training step.
            next_frac = min(1.0, float(self.step + 1) / max(1, self._warmup_steps))
            for pg, base_lr in zip(self.opt.param_groups, self._base_lrs(), strict=True):
                start_lr = self._warmup_start_lr_for(float(base_lr))
                pg["lr"] = start_lr + (float(base_lr) - start_lr) * next_frac
        else:
            self._scheduler.step(self.step - self._warmup_steps)

    def _resolve_old_bases(self, n_groups: int) -> list[float]:
        """Per-group base LRs to scale from. Falls back to ``self._peak_lr`` if
        the scheduler hasn't recorded any, and pads/truncates to ``n_groups``."""
        old = (
            [float(v) for v in self._scheduler.base_lrs]
            if hasattr(self._scheduler, "base_lrs") else []
        )
        if not old:
            old = [float(self._peak_lr)] * n_groups
        if len(old) < n_groups:
            old.extend([old[-1]] * (n_groups - len(old)))
        return old[:n_groups]

    def _rescale_active_lr_for_group(
        self,
        pg: dict,
        *,
        old_base: float,
        new_base: float,
        scheduler_last_lr_for_group: float | None,
    ) -> None:
        """Rebase one optimizer group's active ``lr``. Cold-start cases (old_base==0)
        recreate the warmup-phase LR from ``self.step`` and ``self._warmup_steps``."""
        if old_base > 0.0:
            pg["lr"] = float(pg.get("lr", 0.0)) * (new_base / old_base)
            return
        if self.step < self._warmup_steps:
            frac = self.step / max(1, self._warmup_steps)
            warm_start = self._warmup_start_lr_for(new_base)
            pg["lr"] = warm_start + (new_base - warm_start) * frac
        else:
            pg["lr"] = float(scheduler_last_lr_for_group) if scheduler_last_lr_for_group is not None else new_base

    def set_grad_clip_max_norm(self, max_norm: float | None) -> bool:
        """Re-point zclip's fixed hard cap on a RUNNING trainer.

        ``ZClip`` reads ``max_grad_norm`` once in its constructor, so before
        this existed a live yaml edit to ``zclip_max_norm`` was a silent no-op:
        the key is in none of the restart-required sets, so ``_reload_yaml_into_config``
        overlaid it into the config every iteration, nothing pushed it at the
        optimizer, and the reloader did not even log the ``requires restart``
        warning it gives ``lr_schedule``. Same shape as the ``weight_decay``
        defect in ``_reapply_configured_param_group_hparams`` (rl_loop_audit
        I13): a control surface that reads as live and is not.

        The cap is a per-step threshold with no state behind it, so re-pointing
        it mid-run is safe and takes effect on the next optimizer step.

        Returns True when the value actually changed, so the caller can log the
        transition rather than every iteration.

        Rejects values that would corrupt training rather than clip it. zclip
        computes ``clip_coef = max_grad_norm / (global_norm + 1e-6)`` with no
        clamp, so a cap of ``0`` zeroes every gradient and a NEGATIVE cap flips
        their sign into gradient ASCENT. Those were restart-only typos before
        this became a live surface; now a slip in the live yaml would land
        within one iteration. A bad value is refused and the current cap kept,
        because erroring out would take down a running trial over a typo.
        """
        old = self.grad_clip_max_norm
        if max_norm is None:
            new = None
        else:
            new = float(max_norm)
            if not math.isfinite(new) or new <= 0.0:
                print(
                    f"[trainer] REFUSING zclip_max_norm={max_norm!r} "
                    f"(must be finite and > 0); keeping {old}",
                    flush=True,
                )
                return False
        if new == old:
            return False
        self.zclip.max_grad_norm = new  # pyright: ignore[reportAttributeAccessIssue]
        return True

    def set_sf_target_rebuild(
        self, *, enabled: bool, params: SfTargetParams,
    ) -> bool:
        """Re-point the SF target rebuild on a RUNNING trainer.

        ``rebuild_sf_targets`` and every ``SfTargetParams`` knob are read at
        Trainer construction, so without this a live yaml edit would sit in
        ``config`` doing nothing until the next restart — the
        "config_change_may_not_be_in_effect" shape. The rebuild is a pure
        function of the sampled batch with no optimizer or model state behind
        it, so re-pointing it mid-run is safe and takes effect on the next
        batch the prefetch thread builds.

        Turning it ON is the whole point of the flag: an ``SfTargetParams``
        change then applies to the SF-LABELLED rows already in the replay
        window on the next iteration, instead of only to data generated after
        the edit (~18h for a 1.5M-row window to turn over at the current ingest
        rate). On healthy data that is every one of them — a labelled row is
        written with ``sf_multipv_raw`` — so the window does NOT become a
        mixture of two target regimes. ``sf_rebuild_policy_frac`` reports the
        realized rate; it falling below ``sf_rebuild_wdl_frac`` means desynced
        Stockfish rows, not a bound on what the rebuild can reach — and it
        UNDERCOUNTS them, since only ~59% of poisoned rows lose the block.

        ``sf_target_params`` is written only when a CONSUMER is active — this
        rebuild, or ``sf_policy_sparse_ce``, which reads the same field as
        ``sf_sparse_params`` in `_loss_kwargs`. Writing it unconditionally
        would mean that the day `sf_policy_sparse_ce` is switched on, a live
        `sf_policy_temp` edit silently retargets the sparse-CE `w_sf_move`
        loss; that it is inert today rests only on a key being absent from the
        yaml, which is not a guarantee. With both consumers off the field keeps
        its construction-time value and nothing reads it.

        Returns True when something actually changed, so the caller logs the
        transition rather than every iteration.
        """
        consumer_active = bool(enabled) or self.sf_policy_sparse_ce
        changed = bool(enabled) != self.rebuild_sf_targets or (
            consumer_active and params != self.sf_target_params
        )
        self.rebuild_sf_targets = bool(enabled)
        if consumer_active:
            self.sf_target_params = params
        return changed

    @property
    def grad_clip_max_norm(self) -> float | None:
        """The fixed hard cap the LIVE zclip object is currently applying."""
        raw = getattr(self.zclip, "max_grad_norm", None)
        return None if raw is None else float(raw)

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
        old_bases = self._resolve_old_bases(n_groups)
        ref_old = max(float(self._peak_lr), self._reference_lr_from_bases(old_bases))
        scale = new_peak / ref_old

        new_bases = [(ob * scale if ob > 0.0 else new_peak) for ob in old_bases]
        self._peak_lr = new_peak

  # Keep scheduler phase but rebase amplitude.
        scheduler: Any = self._scheduler
        if hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = list(new_bases)
        if hasattr(scheduler, "_last_lr"):
            last_lrs = scheduler._last_lr
            if last_lrs:
                scheduler._last_lr = [float(v) * scale for v in last_lrs]

  # Keep optimizer param-group metadata aligned.
        for pg, ob, nb in zip(self.opt.param_groups, old_bases, new_bases, strict=True):
            if "initial_lr" in pg:
                pg["initial_lr"] = float(pg["initial_lr"]) * (nb / ob) if ob > 0.0 else nb

        if not rescale_current:
            return

  # Rebase currently active optimizer LR so training continues at same phase.
        last_lrs = getattr(self._scheduler, "_last_lr", None)
        for i, (pg, ob, nb) in enumerate(zip(self.opt.param_groups, old_bases, new_bases, strict=True)):
            sched_lr = (
                float(last_lrs[i]) if isinstance(last_lrs, list) and i < len(last_lrs) else None
            )
            self._rescale_active_lr_for_group(
                pg, old_base=ob, new_base=nb, scheduler_last_lr_for_group=sched_lr,
            )

    @staticmethod
    def _policy_accuracy_stats(
        out: dict, batch: dict,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Top-1/top-5 accuracy for policy_own, policy_sf, policy_future.

        Returns (numerator, denominator) GPU 0-d tensors per head — accumulated
        on-device by callers so per-microbatch ``.item()`` syncs stay out of
        the inner loop. Per-head legal masks (``legal_mask`` at t,
        ``sf_legal_mask`` at t+1 opp-POV, ``future_legal_mask`` at t+2
        net-POV) are gated on ``has_*`` flags so old shards without them
        fall through unmasked.
        """
        stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        def _topk(
            logits: torch.Tensor, target: torch.Tensor, mask_f: torch.Tensor,
            k_values: tuple[int, ...],
        ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
            total = mask_f.sum()
            max_k = max(k_values)
            _, top_idx = torch.topk(logits, k=max_k, dim=-1)
            match = (top_idx == target.unsqueeze(-1))
            return {
                k: ((match[:, :k].any(dim=-1).to(torch.float32) * mask_f).sum(), total)
                for k in k_values
            }

        def _align_index(
            target: torch.Tensor, *, source_width: int, dst_width: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
  # ONE dispatch, in torch_maps, which owns both the width->table binding and
  # the ONE device cache for these tables (CLAUDE.md: "Use the shared
  # device-cached lookups in moves/torch_maps.py -- don't add per-module
  # lru_cache copies"). This module used to carry a private lru_cache over
  # the same two arrays, keyed on target.device.index RAW, so
  # torch.device("cuda") and ("cuda", 0) allocated two separate copies of
  # both tables; torch_maps._device_key normalises that away.
  #
  # The width dispatch lives there too, deliberately: duplicating it here
  # meant the branch->direction binding could be swapped with both symbol
  # names still present, which no test in this repo could see (a reviewer
  # swapped the two branch bodies and every value-identity test passed --
  # only the ruler-id source hash noticed, and a source hash is a tripwire,
  # not a semantic control). With one dispatch there is no pair of branches
  # to swap, and the binding is asserted directly in
  # tests/test_trainer_policy_index_lut.py.
            lut = torch_maps.policy_index_remap_table(
                source_width, dst_width, target.device,
            )
            if lut is None:  # widths already agree
                return target, torch.ones_like(target, dtype=torch.bool)
            valid = (target >= 0) & (target < source_width)
            safe_target = target.clamp(0, source_width - 1).to(torch.long)
            mapped = lut.index_select(0, safe_target)
  # `full_to_compact` stores -1 for "no compact move"; `compact_to_full` is
  # total (min 0), so the extra term is a no-op on that direction rather
  # than a behaviour change -- asserted, not assumed.
            return mapped, valid & (mapped >= 0)

        def _policy_width_from_batch(*keys: str) -> int:
            for key in keys:
                value = batch.get(key)
                if value is not None and value.ndim >= 2:
                    return int(value.shape[-1])
            return POLICY_SIZE

        pol_logits = out.get("policy") if "policy" in out else out.get("policy_own")
        pol_target = batch.get("policy_t")
        has_policy = batch.get("has_policy")
        if pol_logits is not None and pol_target is not None and has_policy is not None:
            logits = apply_policy_mask_to_logits(pol_logits.detach(), batch, "legal_mask", "has_legal_mask")
            tgt = torch.argmax(align_policy_target(pol_target, int(pol_logits.shape[-1])), dim=-1)
            tk = _topk(logits, tgt, has_policy.to(torch.float32), (1, 5))
            stats["policy_own_acc_top1"] = tk[1]
            stats["policy_own_acc_top5"] = tk[5]

        sf_logits = out.get("policy_sf")
        has_sf_move = batch.get("has_sf_move")
        if sf_logits is not None and has_sf_move is not None and "sf_move_index" in batch:
            dst_width = int(sf_logits.shape[-1])
            logits = apply_policy_mask_to_logits(sf_logits.detach(), batch, "sf_legal_mask", "has_sf_legal_mask")
            source_width = _policy_width_from_batch("sf_legal_mask", "sf_policy_t", "policy_t")
            tgt, valid_idx = _align_index(batch["sf_move_index"], source_width=source_width, dst_width=dst_width)
            tk = _topk(logits, tgt, has_sf_move.to(torch.float32) * valid_idx.to(torch.float32), (1, 5))
            stats["sf_move_acc"] = tk[1]
            stats["sf_move_acc_top5"] = tk[5]

        fut_logits = out.get("policy_future")
        fut_target = batch.get("future_policy_t")
        has_future = batch.get("has_future")
        if fut_logits is not None and fut_target is not None and has_future is not None:
            logits = apply_policy_mask_to_logits(fut_logits.detach(), batch, "future_legal_mask", "has_future_legal_mask")
            tgt = torch.argmax(align_policy_target(fut_target, int(fut_logits.shape[-1])), dim=-1)
            tk = _topk(logits, tgt, has_future.to(torch.float32), (1, 5))
            stats["policy_future_acc_top1"] = tk[1]
            stats["policy_future_acc_top5"] = tk[5]

        return stats

    @torch.no_grad()
    def _compute_metrics(
        self, *, buf: ReplayBuffer, batch_size: int, steps: int, tag: str,
        model_override: torch.nn.Module | None = None,
        full_pass: bool = False,
    ) -> TrainMetrics:
        """Score ``buf`` and pool the per-batch results into one TrainMetrics.

        ``full_pass`` walks every row of ``buf`` exactly once in a fixed order
        and ignores ``steps``; otherwise ``steps`` batches are SAMPLED from it.

        Pooling is ROW-WEIGHTED in both modes: each per-batch mean is scaled by
        that batch's row count and the total is divided by the rows scored.
        Sampled batches are all ``batch_size`` rows, so this is numerically
        identical to the old ``sum(mean_i) / steps`` there. A full pass is not
        -- its last batch is ragged (2000 rows at 512 = 3 x 512 + 464), and
        unweighted averaging would count each of those 464 rows 1.10x.
        """
        sums: dict[str, float] = {}
        acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
  # Accumulate on-device so each eval batch adds ~0 host syncs; one .item()
  # at the end produces the global Brier + ECE.
        calib_accum: dict[str, torch.Tensor] = {}
        total_rows = 0

        mirror_p = self.mirror_prob if str(tag).startswith("train") else 0.0

  # ``model_override`` lets the async test-eval path drive the loop on a
  # snapshot model while ``self.model`` is being mutated by the next iter's
  # train phase. ``self.model`` is read elsewhere (notably by
  # _policy_accuracy_stats — but that takes ``out`` already, not ``self.model``).
        eval_model = model_override if model_override is not None else self.model

  # This eval's OWN rebuild-coverage sink, never the trainer-wide one. The
  # async holdout eval runs on its own thread while the next iteration
  # trains (`distributed_async_test_eval: true`), so draining the shared
  # accumulator here would move counts the TRAINING path produced onto this
  # `eval` row and silently under-report the train row.
        eval_coverage = _SfRebuildCoverageAccumulator()
        batches = (
            self._iter_full_pass_batches(
                buf, batch_size=batch_size, coverage=eval_coverage,
            )
            if full_pass else
            self._iter_prefetched_batches(
                buf, batch_size=batch_size, mirror_prob=mirror_p, count=int(steps),
                coverage=eval_coverage,
            )
        )
        ruler = type(self).eval_ruler_id_for(
            batch_size=int(batch_size), steps=int(steps),
            mirror_prob=float(mirror_p), full_pass=bool(full_pass),
        )
        for batch in batches:
            n_rows = int(batch["x"].shape[0])
            if n_rows <= 0:
                continue
            with self._amp_context():
                _rel = batch.get("relations")
                out = eval_model(batch["x"], relations=_rel) if _rel is not None else eval_model(batch["x"])
                losses = compute_loss(out, batch, **self._eval_loss_kwargs)

            scalars = self._extract_loss_scalars(losses)
            for k, v in scalars.items():
  # `_RAW_SUM_LOSS_KEYS` are already row sums; the rest are row means.
                sums[k] = sums.get(k, 0.0) + (v if k in _RAW_SUM_LOSS_KEYS else v * n_rows)
            total_rows += n_rows

            for name, (n_, d_) in self._policy_accuracy_stats(out, batch).items():
                prev = acc_sums.get(name)
                acc_sums[name] = (n_, d_) if prev is None else (prev[0] + n_, prev[1] + d_)

            wdl_logits = out.get("wdl")
            wdl_target = batch.get("wdl_t")
            if wdl_logits is not None and wdl_target is not None and wdl_target.numel() > 0:
                stats = wdl_calibration_stats(wdl_logits.detach(), wdl_target)
                for k, v in stats.items():
                    calib_accum[k] = calib_accum.get(k, torch.zeros_like(v)) + v

        wdl_brier, wdl_ece = wdl_brier_ece_from_stats(calib_accum) if calib_accum else (0.0, 0.0)
        metrics = self._build_metrics(
            sums, acc_sums, float(max(1, total_rows)),
            wdl_brier=wdl_brier, wdl_ece=wdl_ece, eval_rows=int(total_rows),
            eval_ruler=ruler,
            **eval_coverage.drain(),
        )
        self._log_metrics(metrics, tag)
        return metrics

    def _apply_feature_group_dropout(self, x: torch.Tensor) -> None:
        """Per-group classical-feature dropout applied in-place on x[:, base:, ...]."""
        base = int(self._base_input_planes)
        if x.shape[1] <= base:
            return
        for g_off, g_len, g_p in self._feature_group_dropout:
            if g_p > 0.0:
                drop = (torch.rand((x.shape[0], 1, 1, 1), device=x.device) < g_p).to(x.dtype)
                x[:, base + g_off : base + g_off + g_len, :, :] *= (1.0 - drop)

    def _run_optimizer_step(
        self, *,
        step_sums: dict[str, float],
        step_acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]],
        step_opt_stats: dict[str, float],
        buf: ReplayBuffer,
        batch_size: int,
        update_lr: bool = True,
        collect_optimizer_stats: bool = True,
        batch_iter: Iterator[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[int, float]:
        """Run accum_steps microbatches, do zclip + opt.step + lr update.

        Mutates step_sums/step_acc_sums/step_opt_stats in place. Returns
        (step_n_micro, opt_step_time_s).
        """
        self.opt.zero_grad(set_to_none=True)
        step_n_micro = 0
        batches = batch_iter
        if batches is None:
            batches = self._iter_prefetched_batches(
                buf,
                batch_size=batch_size,
                mirror_prob=self.mirror_prob,
                count=self.accum_steps,
            )
        for _ in range(self.accum_steps):
            batch = next(batches)
            self._apply_feature_group_dropout(batch["x"])
            with self._amp_context():
                _rel = batch.get("relations")
  # kwarg only when present: TinyNet's forward has no relations param.
                out = self.model(batch["x"], relations=_rel) if _rel is not None else self.model(batch["x"])
                losses = compute_loss(out, batch, **self._loss_kwargs)
                balance_loss = getattr(self.model, "_last_channel_balance_loss", None)
                if balance_loss is not None and self.resid_channel_balance_weight > 0.0:
                    losses["channel_balance"] = balance_loss
                    losses["total"] = losses["total"] + self.resid_channel_balance_weight * balance_loss
                loss = losses["total"] / self.accum_steps
            loss.backward()

            scalars = self._extract_loss_scalars(
                losses,
                total_override=loss,
                total_scale=float(self.accum_steps),
            )
            for k, v in scalars.items():
                step_sums[k] = step_sums.get(k, 0.0) + v

            with torch.no_grad():
                for name, (n_, d_) in self._policy_accuracy_stats(out, batch).items():
                    prev = step_acc_sums.get(name)
                    step_acc_sums[name] = (n_, d_) if prev is None else (prev[0] + n_, prev[1] + d_)

            step_n_micro += 1

  # Collect on EVERY step, not just tb_log_interval ones: the stats are
  # pure Python arithmetic over floats zclip already materialized, and a
  # 1-in-10 sample cannot be aggregated into an honest per-iteration clip
  # rate (rl_loop_audit I9/I10).
  # Measured BEFORE the clip, which is a distinction without a difference
  # while the matrix group is outside the clip scope, and the honest
  # pre-clip reading if it ever is not.
        zclip_stats_before = self._zclip_stats_snapshot()
        matrix_grad_norm = self._matrix_grad_norm()
        grad_norm, zclip_stats = self._zclip_step(collect_stats=True)
        if zclip_stats is not None:
            step_opt_stats.update(zclip_stats)
        step_opt_stats["grad_norm"] = float(grad_norm)
        step_opt_stats["grad_norm_aurora"] = float(matrix_grad_norm)
  # The matrix group no longer passes through the clip, and scale-invariance
  # covers a FINITE rescale only: an inf/nan gradient is not a rescale and
  # would reach `_polar_factor` unchecked, where it becomes a hard
  # RuntimeError mid-iteration. Guard the clipped group too — zclip never
  # protected it either (`clip_coef = cap / (nan + 1e-6)` leaves `nan > cap`
  # False, so the whole non-finite gradient passes through unscaled).
        if not (math.isfinite(matrix_grad_norm) and math.isfinite(grad_norm)):
            logging.getLogger(__name__).warning(
                "non-finite gradient at step %d (matrix ||g||=%r, clipped "
                "||g||=%r): skipping the optimizer step",
                self.step, matrix_grad_norm, grad_norm,
            )
  # zclip has already folded this step's norm into its EMA. The step is
  # being discarded, so un-fold it — see _restore_zclip_stats for why a
  # single nan there would be permanent, and why even a finite reading
  # from a discarded step should not shape the clipper's statistics.
            self._restore_zclip_stats(zclip_stats_before)
            step_opt_stats["nonfinite_grad"] = 1.0
            step_opt_stats["lr"] = float(self.opt.param_groups[0]["lr"])
            self.opt.zero_grad(set_to_none=True)
            if update_lr:
                self._update_lr()
            return step_n_micro, 0.0
        if self._should_log_step_scalars():
            self.writer.add_scalar("train/grad_norm", float(grad_norm), self.step)
            self.writer.add_scalar("train/grad_norm_aurora", float(matrix_grad_norm), self.step)
            if zclip_stats is not None:
                self.writer.add_scalar("zclip/total_norm", zclip_stats["total_norm"], self.step)
                self.writer.add_scalar("zclip/effective_clip", zclip_stats["effective_clip"], self.step)
                self.writer.add_scalar("zclip/adaptive_clipped", zclip_stats["adaptive_clip"], self.step)
                self.writer.add_scalar("zclip/hard_clipped", zclip_stats["hard_clip"], self.step)
                self.writer.add_scalar("zclip/clipped", zclip_stats["clipped"], self.step)
  # The LR in force for THIS step, sampled before opt.step() and before
  # _update_lr() moves it on. Averaging these is the only honest answer to
  # "what LR is the trunk training at" under sqrt_release (I19).
        step_opt_stats["lr"] = float(self.opt.param_groups[0]["lr"])
        opt_step_start = time.perf_counter()
        set_collect_uw_stats = getattr(self.opt, "set_collect_uw_stats", None)
        if callable(set_collect_uw_stats):
            set_collect_uw_stats(bool(collect_optimizer_stats))
  # Polar residual rides the same one-step-per-iteration gate. It is
  # scale-invariant and carries no `lr` factor, so unlike the uw-effective
  # pair (M4-2) sampling at the sawtooth floor does not bias it.
        set_collect_polar_stats = getattr(self.opt, "set_collect_polar_stats", None)
        if callable(set_collect_polar_stats):
            set_collect_polar_stats(bool(collect_optimizer_stats))
        self.opt.step()
        opt_step_time_s = time.perf_counter() - opt_step_start
        if update_lr:
            self._update_lr()
        return step_n_micro, opt_step_time_s

    def train_steps(self, buf: ReplayBuffer, *, batch_size: int, steps: int) -> TrainMetrics:
        self.model.train()
        train_wall_start = time.perf_counter()

        sums: dict[str, float] = {}
        acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        n_micro = 0
        opt_step_time_s = 0.0
        train_steps_done = 0
  # Committed only on a successful step, so a retried CUDA-error attempt
  # can't double-count (same discipline as step_sums).
        grad_norms: list[float] = []
        aurora_grad_norms: list[float] = []
        lr_samples: list[float] = []
        clip_counts: dict[str, int] = {
            "clipped": 0, "adaptive_clip": 0, "hard_clip": 0, "nonfinite_grad": 0,
        }
        transient_cuda_retry_batches = 0

        _log = logging.getLogger(__name__)

        requested_steps = int(steps)
        effective_cycle_steps = max(1, requested_steps)
        batch_iter = _TrainBatchIterator(
            lambda count: self._iter_prefetched_batches(
                buf,
                batch_size=batch_size,
                mirror_prob=self.mirror_prob,
                count=count,
            ),
            requested_steps * self.accum_steps,
        )

        try:
            for _ in range(requested_steps):
              for _attempt in range(3):
                step_sums: dict[str, float] = {}
                step_acc_sums: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                step_opt_stats: dict[str, float] = {}
                consumed_before_attempt = batch_iter.consumed
                try:
                    local_release_cycle = self._uses_train_window_release_cycle()
                    if local_release_cycle:
                        self._set_train_window_release_lr(
                            local_step=train_steps_done,
                            cycle_steps=effective_cycle_steps,
                        )
                    step_n_micro, this_opt_time = self._run_optimizer_step(
                        step_sums=step_sums, step_acc_sums=step_acc_sums,
                        step_opt_stats=step_opt_stats,
                        buf=buf, batch_size=batch_size,
                        update_lr=not local_release_cycle,
                        collect_optimizer_stats=train_steps_done + 1 >= requested_steps,
                        batch_iter=batch_iter,
                    )
                    opt_step_time_s += this_opt_time
                except RuntimeError as exc:
                    if "CUDA" not in str(exc) or _attempt >= 2:
                        raise
                    _retry_batches = batch_iter.consumed - consumed_before_attempt
                    batch_iter.add_retry_batches(_retry_batches)
  # Counted, not merely logged. These replacement draws advance the buffer's
  # RNG, so an arm that retried and an arm that did not are no longer sampling
  # the same rows — see the TrainMetrics field comment.
                    transient_cuda_retry_batches += int(_retry_batches)
                    _log.warning("Transient CUDA error (attempt %d/3), retrying: %s", _attempt + 1, exc)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    time.sleep(0.5 * (_attempt + 1))
                    self.opt.zero_grad(set_to_none=True)
                    continue

  # Success — commit metrics from this step.
                for k, v in step_sums.items():
                    sums[k] = sums.get(k, 0.0) + v
                for name, (n_, d_) in step_acc_sums.items():
                    prev = acc_sums.get(name)
                    acc_sums[name] = (n_, d_) if prev is None else (prev[0] + n_, prev[1] + d_)
                n_micro += step_n_micro
                if "grad_norm" in step_opt_stats:
                    grad_norms.append(step_opt_stats["grad_norm"])
                    for flag in clip_counts:
                        clip_counts[flag] += int(step_opt_stats.get(flag, 0.0) > 0.0)
                if "grad_norm_aurora" in step_opt_stats:
                    aurora_grad_norms.append(step_opt_stats["grad_norm_aurora"])
                if "lr" in step_opt_stats:
                    lr_samples.append(step_opt_stats["lr"])

                if (
                    self._swa_model is not None
                    and self.step >= self._swa_start
                    and self.step % self._swa_freq == 0
                ):
                    self._swa_model.update_parameters(self.model)

                if self._should_log_step_scalars():
                    self.writer.add_scalar("train/loss", float(step_sums.get("loss", 0.0) / max(1, step_n_micro)), self.step)
                    self.writer.add_scalar("train/lr", self.opt.param_groups[0]["lr"], self.step)
                self.step += 1
                train_steps_done += 1
                break
        finally:
            batch_iter.close()

        train_time_s = time.perf_counter() - train_wall_start
        train_samples_seen = int(n_micro * batch_size)
        metrics = self._build_metrics(
            sums, acc_sums, float(max(1, n_micro)),
            train_time_s=float(train_time_s),
            opt_step_time_s=float(opt_step_time_s),
            train_steps_done=int(train_steps_done),
            train_samples_seen=int(train_samples_seen),
            opt_lr_mean=float(sum(lr_samples) / len(lr_samples)) if lr_samples else 0.0,
            opt_lr_max=float(max(lr_samples)) if lr_samples else 0.0,
  # `batch_iter.consumed` survives `close()`, and it is the TOTAL microbatches
  # this call pulled from the buffer -- retries included. Two paired arms that
  # disagree on it did not sample the same rows, whatever their shard pools say.
            batches_drawn=float(batch_iter.consumed),
            transient_cuda_retry_batches=float(transient_cuda_retry_batches),
            **_grad_clip_metric_kwargs(grad_norms, clip_counts, aurora_grad_norms),
            **self._sf_rebuild_coverage.drain(),
            **getattr(self.opt, "last_uw_stats", {}),
            **getattr(self.opt, "last_polar_stats", {}),
        )
        self._warn_if_grad_norm_median_past_watch(metrics)
        self._log_metrics(metrics, "train_avg")

  # Compile probe: report once after the first batch of train steps that
  # reach the configured threshold. This catches: (a) compile not engaged,
  # (b) graphs failing to capture, (c) per-step recompile thrash.
        if self._compile_probe_steps_remaining > 0 and train_steps_done > 0:
            self._compile_probe_steps_remaining -= train_steps_done
            if self._compile_probe_steps_remaining <= 0:
                self._compile_probe.report(step_count=10)

  # Log throughput stats that aren't in TrainMetrics
        self.writer.add_scalar("train_avg/steps_per_s", float(train_steps_done / max(train_time_s, 1e-9)), self.step)
        self.writer.add_scalar("train_avg/samples_per_s", float(train_samples_seen / max(train_time_s, 1e-9)), self.step)
        self.writer.add_scalar("train_avg/opt_steps_per_s", float(train_steps_done / max(opt_step_time_s, 1e-9)) if opt_step_time_s > 0.0 else 0.0, self.step)
        return metrics

    @torch.no_grad()
    def eval_steps(self, buf: ReplayBuffer, *, batch_size: int, steps: int) -> TrainMetrics:
        """``steps`` SAMPLED batches. Not a ruler -- see :meth:`eval_full_pass`."""
        self.model.eval()
        return self._compute_metrics(buf=buf, batch_size=batch_size, steps=steps, tag="eval")

    @torch.no_grad()
    def eval_full_pass(self, buf: ReplayBuffer, *, batch_size: int) -> TrainMetrics:
        """Deterministic single pass over ``buf``: every row scored exactly once.

        Two evaluations of the same weights over the same rows return the same
        number, which ``eval_steps`` does not -- it resamples with replacement
        through the training sampler's rebalancing and priority weighting, and
        on the live 2000-row frozen holdout that gave `test_loss` a floor of
        sd 0.0522 nats with nothing to do with the model (docs/rl_loop_audit.md
        G14). ``steps`` has no counterpart here on purpose: the row count is
        the set's, not a budget.
        """
        self.model.eval()
        return self._compute_metrics(
            buf=buf, batch_size=batch_size, steps=0, tag="eval", full_pass=True,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Strip torch.compile's `_orig_mod.` segment before saving so the
        # checkpoint is wrap-agnostic; see strip_compile_prefix for why.
        state: dict[str, Any] = {
            "model": strip_compile_prefix(self.model.state_dict()),
            "opt": self.opt.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "step": self.step,
            "peak_lr": float(self._peak_lr),
  # zclip's EMA is trained state, not configuration: without it every
  # restart re-enters the 25-step warmup during which the adaptive branch
  # cannot fire at all (`ZClip.step` returns before `_compute_clip_val`
  # while `initialized` is False), so the run drops into a hard-cap-only
  # regime and then re-converges the EMA from wherever warmup put it.
            "zclip": self.zclip_state_dict(),
        }
        if self._model_config is not None:
            state["arch"] = {
                "_schema_version": ARCH_SCHEMA_VERSION,
                **dataclasses.asdict(self._model_config),
            }
        if self._swa_model is not None:
            state["swa_model"] = strip_compile_prefix(self._swa_model.state_dict())
  # Atomic write so workers polling for new checkpoints never see a partial
  # file (matches the export_swa path; previously diverged).
        atomic_write(path, lambda tmp: torch.save(state, str(tmp)))

    _FRESH_PARAM_NAME_SUFFIXES = ("dynamic_relation_weight", "policy_relation_weight")

    def _remap_optimizer_state_for_new_params(self, ckpt_opt: dict) -> dict | None:
        """Remap a donor optimizer state dict around newly added parameters.

        Warm-starting a dynamic-relations model from a checkpoint without
        those (zero-init) parameters leaves the donor param groups one or two
        entries short; ``Optimizer.load_state_dict`` then raises on the group
        size mismatch and the caller's fallback would reinitialize ALL
        moments + the scheduler. Instead, splice fresh (state-less) slots in
        at the new parameters' positions so every donor moment lands on the
        parameter it belongs to. Returns None when the mismatch isn't
        explained by exactly the known fresh parameters (caller falls back
        to the existing reinit path).

        Order assumption: within each param group, the donor's parameters
        appear in the SAME relative order as the new model's non-fresh
        parameters -- true as long as fresh params are only APPENDED to the
        module registration order (the relation weights register after the
        trunk). A refactor that moves them earlier would silently splice
        donor moments onto the wrong slots; the per-group length check
        below cannot catch reordering.
        """
        fresh_ids = {
            id(param)
            for name, param in self.model.named_parameters()
            if name.endswith(self._FRESH_PARAM_NAME_SUFFIXES)
        }
        groups_ckpt = ckpt_opt.get("param_groups", [])
        groups_new = self.opt.param_groups
        if not fresh_ids or len(groups_ckpt) != len(groups_new):
            return None
        state_remap: dict[int, int] = {}
        out_groups: list[dict] = []
        new_idx = 0
        for g_new, g_ckpt in zip(groups_new, groups_ckpt, strict=True):
            ckpt_params = list(g_ckpt.get("params", []))
            n_fresh = sum(1 for param in g_new["params"] if id(param) in fresh_ids)
            if len(g_new["params"]) != len(ckpt_params) + n_fresh:
                return None
            donor_iter = iter(ckpt_params)
            new_param_ids: list[int] = []
            for param in g_new["params"]:
                if id(param) not in fresh_ids:
                    state_remap[next(donor_iter)] = new_idx
                new_param_ids.append(new_idx)
                new_idx += 1
            out_group = {k: v for k, v in g_ckpt.items() if k != "params"}
            out_group["params"] = new_param_ids
            out_groups.append(out_group)
        return {
            "state": {
                state_remap[k]: v
                for k, v in ckpt_opt.get("state", {}).items()
                if k in state_remap
            },
            "param_groups": out_groups,
        }

    def _snapshot_param_group_hparams(self) -> list[dict[str, Any]]:
        """Copy every config-derived key out of the live optimizer groups."""
        return [
            {
                key: value
                for key, value in group.items()
                if key not in _CHECKPOINT_OWNED_GROUP_KEYS
            }
            for group in self.opt.param_groups
        ]

    def _reapply_configured_param_group_hparams(self) -> None:
        """Restore this run's configured param-group hyperparameters after a load.

        ``torch.optim.Optimizer.load_state_dict`` does not merge group
        hyperparameters — it REPLACES each live group dict with the
        checkpoint's, keeping only ``params``. So every construction-time group
        key is inherited from whatever checkpoint the run happens to resume
        from, and the config value is silently discarded.

        Two keys survived that by accident: ``lr`` (``set_peak_lr`` re-applies
        it right after ``load``) and the ``aurora_*`` polar knobs
        (``_apply_lr_gamma_weights`` re-pushes them every iteration).
        ``weight_decay`` had no such path, which is how ``matrix_weight_decay:
        0`` sat in the yaml for seven weeks while the live Aurora group ran at
        ``1e-4`` — a config key that could not be changed (rl_loop_audit I13).
        ``aurora_uw_floor`` and the SODA marker keys are in the same class.

        Re-applying the whole snapshot (rather than special-casing
        ``weight_decay``) means any future construction-time group key is
        covered automatically. ``_CHECKPOINT_OWNED_GROUP_KEYS`` is the
        deliberate exception list: schedule phase and per-group counters must
        come from the checkpoint, not from config.
        """
        configured = getattr(self, "_configured_param_group_hparams", None)
        if not configured:
            return
        log = logging.getLogger(__name__)
        groups = self.opt.param_groups
        if len(groups) != len(configured):
            log.warning(
                "Optimizer has %d param group(s) after restore but %d were "
                "configured — skipping the configured-hyperparameter "
                "re-application (rl_loop_audit I13)",
                len(groups), len(configured),
            )
            return
        for idx, (group, hparams) in enumerate(zip(groups, configured, strict=True)):
            for key, value in hparams.items():
                if _scalar_hparam_differs(group.get(key), value):
                    log.info(
                        "Restore overwrote optimizer group %d %s (%r); "
                        "re-applying the configured %r",
                        idx, key, group.get(key), value,
                    )
            group.update(hparams)

    def load(self, path: Path) -> None:
        from chess_anti_engine.model import (
            load_state_dict_tolerant,
            migrate_optimizer_input_plane_state,
        )

        # Trainer checkpoints include optimizer/scheduler/RNG pickles, so resume needs the
        # full trusted checkpoint payload rather than PyTorch's weights-only loader.
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        load_state_dict_tolerant(self.model, ckpt["model"], label="resume")
        fresh_opt_state = self.opt.state_dict()
        fresh_scheduler_state = self._scheduler.state_dict()
        optimizer_state_loaded = True
        try:
            opt_state = ckpt["opt"]
            n_ckpt_params = sum(len(g.get("params", ())) for g in opt_state.get("param_groups", []))
            n_model_params = sum(len(g["params"]) for g in self.opt.param_groups)
            if n_ckpt_params < n_model_params:
                remapped = self._remap_optimizer_state_for_new_params(opt_state)
                if remapped is not None:
                    logging.getLogger(__name__).info(
                        "Spliced %d fresh parameter slot(s) into the donor "
                        "optimizer state (zero-init warm start)",
                        n_model_params - n_ckpt_params,
                    )
                    opt_state = remapped
            self.opt.load_state_dict(opt_state)
  # torch restores moment tensors without shape validation, so a v1
  # checkpoint under a v2_threats model would crash at the first
  # opt.step() instead of here. Pad the input-plane columns to match.
            migrated = migrate_optimizer_input_plane_state(self.opt)
            if migrated:
                logging.getLogger(__name__).info(
                    "Zero-padded %d optimizer state tensor(s) for the "
                    "v1 -> v2_threats input-plane migration", migrated,
                )
        except (ValueError, KeyError, RuntimeError) as exc:
            optimizer_state_loaded = False
            logging.getLogger(__name__).warning(
                "Optimizer state incompatible with new model layout, "
                "reinitialising optimizer: %s", exc,
            )
            self.opt.load_state_dict(fresh_opt_state)
            self._scheduler.load_state_dict(fresh_scheduler_state)
            self.reset_optimizer_reference_weights()
  # Both branches above went through Optimizer.load_state_dict, which
  # replaces every group's hyperparameters wholesale. Put this run's
  # configured values back (rl_loop_audit I13).
        self._reapply_configured_param_group_hparams()
        if "scheduler" in ckpt and optimizer_state_loaded:
            try:
                self._scheduler.load_state_dict(ckpt["scheduler"])
            except (ValueError, KeyError, RuntimeError) as exc:
                logging.getLogger(__name__).warning(
                    "Scheduler state incompatible with current optimizer layout, "
                    "reinitialising scheduler: %s", exc,
                )
                self._scheduler.load_state_dict(fresh_scheduler_state)
        if "peak_lr" in ckpt:
            self._peak_lr = float(ckpt["peak_lr"])
        else:
            self._peak_lr = self._reference_lr_from_bases()
  # Gated on `optimizer_state_loaded` for the same reason the scheduler is:
  # that branch only fails when the model's parameter layout no longer
  # matches the donor's, and an EMA of gradient norms taken over a
  # DIFFERENT set of parameters is not a description of this run's
  # gradients. Fresh warmup is the honest state there.
        zclip_restored = False
        if optimizer_state_loaded:
            zclip_restored = self.load_zclip_state(ckpt.get("zclip"))
        else:
            logging.getLogger(__name__).info(
                "zclip: not restoring the adaptive EMA -- the optimizer state "
                "was reinitialised for a new parameter layout, so the "
                "checkpoint's gradient-norm statistics describe a different "
                "model. Starting a fresh EMA warmup",
            )
        if "swa_model" in ckpt and self._swa_model is not None:
            try:
                # Checkpoints store SWA weights wrap-agnostic (module.*), but a
                # compiled trainer's AveragedModel expects module._orig_mod.*.
                # Without the realignment a compile-on resume would silently
                # discard the running average and restart it from the current
                # weights via the reinit below.
                self._swa_model.load_state_dict(
                    align_compile_prefix(
                        ckpt["swa_model"], reference=self._swa_model.state_dict(),
                    )
                )
            except (RuntimeError, KeyError) as exc:
                logging.getLogger(__name__).warning(
                    "SWA model state incompatible, reinitialising: %s", exc,
                )
        self.step = int(ckpt.get("step", 0))
  # Emitted AFTER `self.step` is set so the point lands on the resumed step
  # rather than on 0. This is the observation that distinguishes "restored"
  # from "accepted and ignored" on a real restart without reading a log:
  # `zclip/restored` is 1.0 at the resume step, and `zclip/ema_mean` is the
  # previous run's EMA rather than whatever a fresh warmup would produce.
        self.writer.add_scalar("zclip/restored", 1.0 if zclip_restored else 0.0, self.step)
        if zclip_restored and self.zclip.initialized:
            self.writer.add_scalar("zclip/ema_mean", float(self.zclip.mean or 0.0), self.step)
            self.writer.add_scalar("zclip/ema_var", float(self.zclip.var or 0.0), self.step)

    def export_swa(self, path: Path, dataloader: Any = None) -> None:
        """Export the SWA-averaged model weights.

        If a dataloader is provided, batch normalization statistics are updated
        using ``torch.optim.swa_utils.update_bn``.

        Written atomically to avoid races with workers downloading the file
        while the learner is writing it.

        Keys go through ``strip_compile_prefix`` for the same reason
        ``save()`` does. This path used not to, so with ``use_compile: true``
        (production since 2026-04-27) every published ``latest_model.pt``
        carried ``_orig_mod.`` on all 496 keys while the sibling checkpoint
        carried it on none — two conventions for the same weights. Nothing
        broke only because every in-tree consumer routes through
        ``load_state_dict_tolerant``, which normalizes; a plain
        ``load_state_dict(..., strict=False)`` on the published file returns
        missing=86/unexpected=86 and a fresh-init model, with no exception
        (rl_loop_audit J9).

        WHEN SWA IS ON, THIS FILE IS NOT THE FILE THE ARENA MEASURES. ``save``
        writes the raw model under ``"model"`` — resume must continue the real
        training trajectory, not an average — and the ratchet arena reads
        exactly that key out of ``checkpoint_*/trainer.pt``, so the strength
        ruler would be scoring a different net than the workers play (measured
        in repro: all 86/86 tensors differ). The two artifacts genuinely cannot
        be reconciled; what is available is refusing to let the divergence be
        quiet, so the warning below fires on every publish while SWA is
        enabled. Production runs ``swa_start: -1``, so it never fires today
        (rl_loop_audit J10).
        """
        if self._swa_model is not None and dataloader is not None:
            torch.optim.swa_utils.update_bn(
                dataloader,
                self._swa_model,
                device=torch.device(self.device),
            )
        if self._swa_model is None:
            source = "model"
            raw_state = self.model.state_dict()
        else:
            source = "swa_model.module"
            raw_state = self._swa_model.module.state_dict()
            logging.getLogger(__name__).warning(
                "export_swa: SWA is ENABLED, so %s carries the SWA average while "
                "checkpoint trainer.pt['model'] carries the raw model. Every "
                "consumer that reads the checkpoint -- the ratchet arena, "
                "value_regret, audit_targets -- is measuring a DIFFERENT net "
                "than the selfplay workers play. Point those tools at the "
                "published file, or keep swa_start negative. (audit J10)",
                path,
            )
        state_dict = strip_compile_prefix(raw_state)
        export: dict[str, Any] = {"model": state_dict}
        if self._model_config is not None:
            export["arch"] = {
                "_schema_version": ARCH_SCHEMA_VERSION,
                **dataclasses.asdict(self._model_config),
            }
        atomic_write(path, lambda tmp: torch.save(export, str(tmp)))
  # ⚑ print(), NOT logging.info(). The trial actor installs NO logging handler:
  # `tune/trainable.py::_set_log_level` sets a LEVEL on the `chess_anti_engine`
  # logger and stops there, and nothing in this package or in Ray attaches one
  # for that process. So an INFO record falls through to `logging.lastResort`,
  # which is WARNING+, and is DISCARDED — verified directly: with no handler,
  # `logger.warning` reaches stderr and `logger.info` reaches nothing at all.
  # Every operator-visible line in this process is a print() for that reason
  # (`[trial]`, `[disk_buf]`, `[tune]`), and this line exists precisely so that
  # which weights shipped is OBSERVABLE rather than re-derived from the config.
  # A provenance line nobody can read would be this PR's own defect: a value
  # emitted and then silently dropped. Do not "fix" this back to a logger
  # without also installing a handler, and do not promote it to WARNING — it is
  # not a warning, it is the normal record of a normal publish.
        print(
            f"[trial] export_swa: wrote {path} source={source} "
            f"step={int(self.step)} tensors={len(state_dict)} "
            f"params={state_dict_unique_param_count(state_dict)} "
            f"digest={state_dict_digest(state_dict)} "
            f"swa_enabled={self._swa_model is not None}",
            flush=True,
        )
