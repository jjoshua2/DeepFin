from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import queue
import random
import struct
import threading
import time
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace as dataclass_replace
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch

from chess_anti_engine.broker_hang import (
    BOOT_HANG_ABORT_ENV as _BOOT_HANG_ABORT_ENV,
    DEFAULT_BOOT_HANG_ABORT_S as _DEFAULT_BOOT_HANG_ABORT_S,
    DEFAULT_HANG_ABORT_S as _DEFAULT_HANG_ABORT_S,
    HANG_ABORT_ENV as _HANG_ABORT_ENV,
    BrokerHangWatchdog,
    pin_nvml_cuda_check,
    resolve_boot_hang_abort_seconds,
    resolve_hang_abort_seconds,
)
from chess_anti_engine.encoding import input_plane_count, model_input_plane_count
from chess_anti_engine.model import (
    ModelConfig,
    build_model,
    load_state_dict_tolerant,
    model_config_from_manifest_dict,
)
from chess_anti_engine.moves import COMPACT_POLICY_SIZE, POLICY_SIZE
from chess_anti_engine.moves.torch_maps import compact_to_full_index_for as _compact_to_full_index_for
from chess_anti_engine.utils.amp import inference_autocast
import contextlib

log = logging.getLogger(__name__)

# Input/output tensor shape constants shared by evaluators, slot layouts and
# pinned buffers. _CHANNELS is the v1 default (146 = 112 LC0 + 34 extra);
# v2_threats paths pass their own channel count explicitly.
#
# A 175-plane production system that forgets one of the --input-planes /
# --inference-slot-input-planes flags silently gets this 146 instead (encoding
# audit E4). The default is kept — changing it would only move which
# configuration is silently wrong — and every path that later meets a real model
# calls require_model_planes, so the mismatch raises instead of running.
_CHANNELS = input_plane_count()  # 146 = 112 LC0 + 34 v1 extra
_BOARD_H = 8
_BOARD_W = 8
_POLICY_SIZE = POLICY_SIZE
_WDL_SIZE = 3
_F32 = np.float32
_F32_BYTES = 4

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class BatchEvaluator(Protocol):
    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


class AsyncBatchEvaluator(BatchEvaluator, Protocol):
    """Evaluators that also expose a non-blocking GPU path (for MCTS pipelining)."""

    def evaluate_encoded_async(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        ...


def _policy_output(out: dict[str, torch.Tensor]) -> torch.Tensor:
    """Extract policy tensor from model output (handles both key conventions)."""
    return out["policy"] if "policy" in out else out["policy_own"]


def require_model_planes(model: torch.nn.Module, channels: int, *, where: str) -> None:
    """Fail loudly when a buffer/slot width disagrees with the model's encoding.

    ``channels`` is a shared-memory or pinned-buffer width chosen before the
    model was known — from ``--input-planes`` / ``--inference-slot-input-planes``,
    whose default is the v1 146. The model declares its own encoding, so once it
    exists the two can be compared; without this the broker allocates a
    146-plane segment for a 175-plane net and the mismatch surfaces as garbage
    or as a shape error far from its cause (encoding audit E4/E5).
    """
    want = int(model_input_plane_count(model))
    if int(channels) != want:
        raise ValueError(
            f"{where}: buffer width is {int(channels)} input planes but the model "
            f"declares input_extra_features="
            f"{getattr(model, 'input_extra_features', None)!r} = {want} planes. "
            "Pass the matching --input-planes / --inference-slot-input-planes "
            "(the default is the v1 146)."
        )


def _slot_rows(slot: _InferenceSlot, max_batch: int) -> int:
    """Rows this broker will actually evaluate for ``slot``.

    ONE definition for ALL FOUR row-count sites on the ``SlotBroker`` path: the
    gather loop that evaluates the rows, the serve loop that counts them at
    dispatch, and the two ``_process_batch`` failure handlers that hand them
    back. They used to differ: the gather clamped to ``max_batch`` while the
    other three summed the raw ``slot.batch_size``, so a client writing a
    batch_size above the layout's capacity had the excess counted as served
    throughput and never evaluated.

    ⚑ The first version of this helper covered only the gather and the serve
    loop, and its docstring said "both" -- which is how the two handlers went on
    reading raw. That made the residual WORSE than before the helper existed:
    dispatch counted the clamped 8 while a handler subtracted the claimed 12, so
    a wholly failed batch reported -4 rows instead of 0. A partial application
    of "share the instrument" inverted the error it was meant to remove. Four
    sites, one definition, and the count is stated here so the next edit has to
    notice if it adds a fifth.
    """
    return max(0, min(int(slot.batch_size), int(max_batch)))


def _policy_output_full(out: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return dense 4672 policy logits for search-time consumers.

    Compact 1858 checkpoints are the storage/training encoding, but the CBoard
    and MCTS action id space remains the legacy 4672 ids. Invalid padded slots
    are filled with -1e9 so downstream legal masking behaves as before.
    """
    pol = _policy_output(out)
    if int(pol.shape[-1]) == int(_POLICY_SIZE):
        return pol
    if int(pol.shape[-1]) != int(COMPACT_POLICY_SIZE):
        raise ValueError(f"unexpected policy output width {int(pol.shape[-1])}")
    full = pol.new_full((*pol.shape[:-1], _POLICY_SIZE), -1e9)
    full.index_copy_(-1, _compact_to_full_index_for(pol), pol)
    return full


def _forward_no_grad(
    model: torch.nn.Module,
    xt: torch.Tensor,
    *,
    device: str,
    use_amp: bool = True,
    amp_dtype: str = "auto",
    relations_t: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Run a single forward pass under no_grad + inference_autocast.

    Centralizes the ``with torch.no_grad(): with inference_autocast(...): ...``
    pattern so every evaluator path uses the same autocast policy. Eleven
    sites used to reimplement this inline. ``relations_t`` is forwarded only
    when present so models without the kwarg (TinyNet, AOT nets) still work.
    """
    with torch.no_grad(), inference_autocast(device=device, enabled=use_amp, dtype=amp_dtype):
        if relations_t is not None:
            return model(xt, relations=relations_t)
        return model(xt)


def _relations_to_device(
    relations: np.ndarray | None, *, device: str, bsz: int | None = None,
) -> torch.Tensor | None:
    """(N, 5, 64, 64) uint8 numpy -> device tensor (or None)."""
    if relations is None:
        return None
    rel = np.ascontiguousarray(relations if bsz is None else relations[:bsz])
    return torch.from_numpy(rel).to(device)


def _supports_legal_policy_forward(model: torch.nn.Module) -> bool:
    if getattr(model, "_orig_mod", None) is not None and os.environ.get("CAE_COMPILED_LEGAL_POLICY", "0") != "1":
        return False
    if callable(getattr(model, "forward_legal_policy", None)):
        return True
    orig = getattr(model, "_orig_mod", None)
    return callable(getattr(orig, "forward_legal_policy", None))


def _supports_legal_policy_rows_forward(model: torch.nn.Module) -> bool:
    if getattr(model, "_orig_mod", None) is not None and os.environ.get("CAE_COMPILED_LEGAL_ROWS_POLICY", "0") != "1":
        return False
    if callable(getattr(model, "forward_legal_policy_rows", None)):
        return True
    orig = getattr(model, "_orig_mod", None)
    return callable(getattr(orig, "forward_legal_policy_rows", None))


def _forward_legal_no_grad(
    model: torch.nn.Module,
    xt: torch.Tensor,
    legal_flat: torch.Tensor,
    legal_counts: torch.Tensor,
    *,
    device: str,
    use_amp: bool = True,
    amp_dtype: str = "auto",
) -> dict[str, torch.Tensor]:
    with torch.no_grad(), inference_autocast(device=device, enabled=use_amp, dtype=amp_dtype):
        if callable(getattr(getattr(model, "_orig_mod", None), "forward_legal_policy", None)):
            return model(xt, legal_flat=legal_flat, legal_counts=legal_counts)
        forward_legal = getattr(model, "forward_legal_policy")
        return forward_legal(xt, legal_flat, legal_counts)


def _forward_legal_rows_no_grad(
    model: torch.nn.Module,
    xt: torch.Tensor,
    legal_flat: torch.Tensor,
    legal_rows: torch.Tensor,
    *,
    device: str,
    use_amp: bool = True,
    amp_dtype: str = "auto",
) -> dict[str, torch.Tensor]:
    with torch.no_grad(), inference_autocast(device=device, enabled=use_amp, dtype=amp_dtype):
        if callable(getattr(getattr(model, "_orig_mod", None), "forward_legal_policy_rows", None)):
            return model(xt, legal_flat=legal_flat, legal_rows=legal_rows)
        forward_legal = getattr(model, "forward_legal_policy_rows")
        return forward_legal(xt, legal_flat, legal_rows)


def _configure_compile_cache(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    compile_root = cache_root / "compile_cache"
    inductor_dir = compile_root / "torchinductor"
    triton_dir = compile_root / "triton"
    inductor_dir.mkdir(parents=True, exist_ok=True)
    triton_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_dir))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_dir))


def _coerce_input_batch(x: np.ndarray) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"expected encoded batch shape (B,C,H,W), got {x.shape!r}")
    if x.dtype is np.dtype(np.float32) and x.flags["C_CONTIGUOUS"]:
        return x
    return np.ascontiguousarray(x, dtype=np.float32)


def _coerce_bf16_bits_batch(x: np.ndarray) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"expected encoded batch shape (B,C,H,W), got {x.shape!r}")
    if x.dtype != np.uint16:
        raise TypeError(f"expected uint16 bf16 bits, got {x.dtype}")
    if x.flags["C_CONTIGUOUS"]:
        return x
    return np.ascontiguousarray(x, dtype=np.uint16)


def _bf16_bits_to_float32_np(x: np.ndarray) -> np.ndarray:
    """Decode a numpy uint16 array holding bfloat16 bits into float32."""
    if x.dtype != np.uint16:
        raise TypeError(f"expected uint16 bf16 bits, got {x.dtype}")
    return (np.ascontiguousarray(x).astype(np.uint32) << 16).view(np.float32)


def _detach_attached_shm_from_resource_tracker(shm: SharedMemory) -> None:
    """Prevent attach-only clients from unlinking broker-owned POSIX SHM.

    Python's resource tracker registers every SharedMemory handle, including
    create=False attachments. If a worker exits after only attaching to a
    broker-owned slot, the worker's resource tracker can incorrectly unlink
    that shared-memory name and wedge the live broker. The creating broker
    remains responsible for unlinking the slot.
    """
    name = str(getattr(shm, "_name", "") or getattr(shm, "name", "")).strip()
    if not name:
        return
    try:
        resource_tracker.unregister(name, "shared_memory")
    except KeyError:
        pass  # already unregistered (race between parent/child detach)


# ---------------------------------------------------------------------------
# Local (in-process) evaluator — used in tests and single-GPU mode
# ---------------------------------------------------------------------------


class LocalModelEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str,
        use_amp: bool = True,
        amp_dtype: str = "auto",
    ) -> None:
        self.model = model
        self.device = str(device)
        self._use_cuda = self.device.startswith("cuda")
        self._use_amp = bool(use_amp)
        self._amp_dtype = str(amp_dtype)
  # Lazy-initialized on first evaluate_encoded_async call on CUDA.
        self._stream: torch.cuda.Stream | None = None

    def _forward_encoded(
        self, x: np.ndarray, relations: np.ndarray | None,
    ) -> dict[str, torch.Tensor]:
        xb = _coerce_input_batch(x)
        xt = torch.from_numpy(xb).to(self.device)
        return _forward_no_grad(
            self.model, xt, device=self.device,
            use_amp=self._use_amp, amp_dtype=self._amp_dtype,
            relations_t=_relations_to_device(relations, device=self.device),
        )

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        out = self._forward_encoded(x, relations)
        pol = _policy_output_full(out).detach().to(dtype=torch.float32, device="cpu").numpy()
        wdl = out["wdl"].detach().to(dtype=torch.float32, device="cpu").numpy()
        return pol, wdl

    def evaluate_encoded_with_volatility(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pol, wdl, volatility) — volatility is a (B,) scalar summary.

        The volatility head emits 3 non-negative components per position
        (VolatilityHead in model/transformer.py); search consumes their mean.
        Used only by volatility-aware Gumbel search (Python path) — the head
        is computed by every forward anyway, so this just stops discarding
        it; the default-off search path never calls this method.
        """
        out = self._forward_encoded(x, relations)
        pol = _policy_output_full(out).detach().to(dtype=torch.float32, device="cpu").numpy()
        wdl = out["wdl"].detach().to(dtype=torch.float32, device="cpu").numpy()
        vol_out = out.get("volatility")
        if vol_out is None:
            vol = np.zeros((pol.shape[0],), dtype=np.float32)
        else:
            vol = (
                vol_out.detach().to(dtype=torch.float32, device="cpu").numpy()
                .reshape(pol.shape[0], -1).mean(axis=1)
            )
        return pol, wdl, vol

    def evaluate_encoded_async(
        self,
        x: np.ndarray,
        relations: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        """Launch GPU forward pass and non-blocking D2H transfer.

        Returns (pol_cpu_tensor, wdl_cpu_tensor, event).  The tensors
        live in pinned memory but their data is NOT ready until ``event``
        has been synchronized.  Call ``event.synchronize()`` before
        reading the tensors via ``.numpy()``.

        On CPU devices falls back to synchronous evaluation and returns
        event=None.
        """
        xb = _coerce_input_batch(x)
        if not self._use_cuda:
            xt = torch.from_numpy(xb)
            out = _forward_no_grad(
                self.model, xt, device=self.device,
                use_amp=self._use_amp, amp_dtype=self._amp_dtype,
                relations_t=_relations_to_device(relations, device=self.device),
            )
            policy_out = _policy_output_full(out)
            pol = policy_out.detach().float()
            wdl = out["wdl"].detach().float()
            return pol, wdl, None

        if self._stream is None:
            self._stream = torch.cuda.Stream(device=self.device)
        stream = self._stream

  # Default stream must finish any prior work before we branch
        event_default = torch.cuda.Event()
        event_default.record(torch.cuda.current_stream(self.device))

        with torch.cuda.stream(stream):
            stream.wait_event(cast(Any, event_default))
            xt = torch.from_numpy(xb).to(self.device, non_blocking=True)
            out = _forward_no_grad(
                self.model, xt, device=self.device,
                use_amp=self._use_amp, amp_dtype=self._amp_dtype,
                relations_t=_relations_to_device(relations, device=self.device),
            )
            policy_out = _policy_output_full(out)
            pol = policy_out.detach().to(dtype=torch.float32, device="cpu", non_blocking=True)
            wdl = out["wdl"].detach().to(dtype=torch.float32, device="cpu", non_blocking=True)
            done = torch.cuda.Event()
            done.record(stream)

        return pol, wdl, done


class DirectGPUEvaluator(LocalModelEvaluator):
    """LocalModelEvaluator with pre-allocated pinned buffers.

    Eliminates per-call allocations by reusing pinned host tensors for
    input gather and output scatter.  Inherits ``evaluate_encoded_async``
    from the base class.

    The sync path copies output by default (``copy_out=True``).  Pass
    ``copy_out=False`` to return views into pinned buffers — the caller
    must consume results before the next ``evaluate_encoded`` call.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str,
        max_batch: int = 512,
        use_amp: bool = True,
        amp_dtype: str = "auto",
        n_slots: int = 1,
        input_bf16: bool = False,
        legal_bf16: bool = True,
    ) -> None:
        super().__init__(model, device=device, use_amp=use_amp, amp_dtype=amp_dtype)
        self._max_batch = int(max_batch)
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}")
        self._n_slots = int(n_slots)

        _pin = self._use_cuda
        self._input_bf16 = bool(input_bf16 and self._use_cuda and use_amp)
        # Compact legal-policy path opt-out (CAE_UCI_COMPACT_BF16 gate in UCI):
        # search activation keys on supports_legal_bf16, so a bare hasattr on
        # evaluate_legal_bf16 would flip match play onto the compact path by
        # default, before its pre-committed yardstick. Default True preserves
        # the existing selfplay/broker behavior.
        self._legal_bf16 = bool(legal_bf16)
        _channels = input_plane_count(getattr(model, "input_extra_features", None))
        self._pinned_inputs: list[torch.Tensor] = [
            torch.empty(
                (self._max_batch, _channels, _BOARD_H, _BOARD_W),
                dtype=torch.float32, pin_memory=_pin,
            ) for _ in range(self._n_slots)
        ]
        self._pinned_inputs_bf16: list[torch.Tensor] | None = (
            [
                torch.empty(
                    (self._max_batch, _channels, _BOARD_H, _BOARD_W),
                    dtype=torch.bfloat16, pin_memory=True,
                ) for _ in range(self._n_slots)
            ]
            if self._input_bf16 else None
        )
        self._slot_input_bf16: list[bool] = [False for _ in range(self._n_slots)]
        self._pinned_pols: list[torch.Tensor] = [
            torch.empty(
                (self._max_batch, _POLICY_SIZE),
                dtype=torch.float32, pin_memory=_pin,
            ) for _ in range(self._n_slots)
        ]
        self._pinned_wdls: list[torch.Tensor] = [
            torch.empty(
                (self._max_batch, _WDL_SIZE),
                dtype=torch.float32, pin_memory=_pin,
            ) for _ in range(self._n_slots)
        ]
        self._pinned_legal_pols_bf16_bits: list[torch.Tensor] = [
            torch.empty(
                (self._max_batch * 256,),
                dtype=torch.uint16, pin_memory=_pin,
            ) for _ in range(self._n_slots)
        ]
        self._pinned_inputs_np: list[np.ndarray] = [t.numpy(force=True) for t in self._pinned_inputs]
        self._pinned_inputs_bf16_bits_np: list[np.ndarray] | None = (
            [t.view(torch.uint16).numpy(force=True) for t in self._pinned_inputs_bf16]
            if self._pinned_inputs_bf16 is not None else None
        )
        self._pinned_pols_np: list[np.ndarray] = [t.numpy(force=True) for t in self._pinned_pols]
        self._pinned_wdls_np: list[np.ndarray] = [t.numpy(force=True) for t in self._pinned_wdls]

    @property
    def n_slots(self) -> int:
        return self._n_slots

    @property
    def supports_input_bf16_bits(self) -> bool:
        return self._pinned_inputs_bf16_bits_np is not None

    def get_input_buffer(self, bsz: int, slot: int = 0) -> np.ndarray:
        """Return a writable (bsz, C, H, W) view into pinned input slot ``slot``.

        Caller can write encoded positions directly into this buffer,
        avoiding a separate allocation + copy.  The view is valid until
        the next ``evaluate_inplace`` / ``evaluate_inplace_async`` /
        ``evaluate_encoded`` call **on the same slot**.
        """
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")
        if not 0 <= slot < self._n_slots:
            raise ValueError(f"slot {slot} out of range [0, {self._n_slots})")
        self._slot_input_bf16[slot] = False
        return self._pinned_inputs_np[slot][:bsz]

    def get_input_buffer_bf16_bits(self, bsz: int, slot: int = 0) -> np.ndarray:
        """Return a writable uint16 view containing bfloat16 input bits."""
        if self._pinned_inputs_bf16_bits_np is None:
            raise RuntimeError("bf16 input buffer is unavailable")
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")
        if not 0 <= slot < self._n_slots:
            raise ValueError(f"slot {slot} out of range [0, {self._n_slots})")
        self._slot_input_bf16[slot] = True
        return self._pinned_inputs_bf16_bits_np[slot][:bsz]

    def _device_input(self, bsz: int, *, slot: int) -> torch.Tensor:
        pin_in = self._pinned_inputs[slot]
        if self._pinned_inputs_bf16 is not None and self._slot_input_bf16[slot]:
            return self._pinned_inputs_bf16[slot][:bsz].to(self.device, non_blocking=True)
        if self._pinned_inputs_bf16 is not None:
            pin_bf16 = self._pinned_inputs_bf16[slot]
            pin_bf16[:bsz].copy_(pin_in[:bsz])
            return pin_bf16[:bsz].to(self.device, non_blocking=True)
        return pin_in[:bsz].to(self.device, non_blocking=True)

    def _stage_encoded_input(self, x: np.ndarray, bsz: int, *, slot: int = 0) -> None:
        """Copy an encoded batch into pinned input ``slot``, tracking bf16 dtype.

        uint16 inputs are bf16-bit-packed: stored raw when a bf16-bits pinned
        buffer exists, else widened to float32. Sets the per-slot bf16 flag so
        the forward path interprets the pinned bytes correctly.
        """
        if x.dtype == np.uint16:
            if self._pinned_inputs_bf16_bits_np is not None:
                self._pinned_inputs_bf16_bits_np[slot][:bsz] = x
                self._slot_input_bf16[slot] = True
            else:
                self._pinned_inputs_np[slot][:bsz] = _bf16_bits_to_float32_np(x)
                self._slot_input_bf16[slot] = False
        else:
            self._pinned_inputs_np[slot][:bsz] = x
            self._slot_input_bf16[slot] = False

    def evaluate_inplace(
        self, bsz: int, *, copy_out: bool = True, slot: int = 0,
        relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on data already written to ``get_input_buffer(slot)``."""
        if bsz <= 0:
            return np.empty((0, _POLICY_SIZE), dtype=np.float32), np.empty((0, _WDL_SIZE), dtype=np.float32)
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")
        if not 0 <= slot < self._n_slots:
            raise ValueError(f"slot {slot} out of range [0, {self._n_slots})")
        return self._run_forward(bsz, copy_out=copy_out, slot=slot, relations=relations)

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None, *, copy_out: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if x.ndim != 4:
            raise ValueError(f"expected 4D input, got {x.ndim}D")
        bsz = x.shape[0]
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")

        self._stage_encoded_input(x, bsz)
        return self._run_forward(bsz, copy_out=copy_out, slot=0, relations=relations)

    def _run_forward(
        self, bsz: int, *, copy_out: bool = True, slot: int = 0,
        relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pin_in = self._pinned_inputs[slot]
        pin_pol = self._pinned_pols[slot]
        pin_wdl = self._pinned_wdls[slot]
        rel_t = _relations_to_device(relations, device=self.device, bsz=bsz)
        if not self._use_cuda:
            xt = pin_in[:bsz]
            out = _forward_no_grad(
                self.model, xt, device=self.device,
                use_amp=self._use_amp, amp_dtype=self._amp_dtype,
                relations_t=rel_t,
            )
            return _policy_output_full(out).detach().float().numpy(), out["wdl"].detach().float().numpy()

        xt = self._device_input(bsz, slot=slot)
        out = _forward_no_grad(
            self.model, xt, device=self.device,
            use_amp=self._use_amp, amp_dtype=self._amp_dtype,
            relations_t=rel_t,
        )
        pin_pol[:bsz].copy_(_policy_output_full(out).detach().float(), non_blocking=True)
        pin_wdl[:bsz].copy_(out["wdl"].detach().float(), non_blocking=True)
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(torch.device(self.device)))
        done.synchronize()

        pol_np = self._pinned_pols_np[slot][:bsz]
        wdl_np = self._pinned_wdls_np[slot][:bsz]
        if copy_out:
            return pol_np.copy(), wdl_np.copy()
        return pol_np, wdl_np

    def evaluate_encoded_async(
        self,
        x: np.ndarray,
        relations: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        """Pinned-memory async eval: H2D via DMA, non-blocking D2H."""
        if x.ndim != 4:
            raise ValueError(f"expected 4D input, got {x.ndim}D")
        bsz = x.shape[0]
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")

        if not self._use_cuda:
            return super().evaluate_encoded_async(x, relations=relations)

        self._stage_encoded_input(x, bsz)
        return self._async_forward(bsz, slot=0, relations=relations)

    def evaluate_inplace_async(
        self, bsz: int, *, slot: int = 0, relations: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        """Async forward on data already written to ``get_input_buffer(slot)``.

        Skips the input memcpy: caller wrote the encoded batch directly into
        the pinned input view, so H2D DMA can launch immediately. Output
        slots are independent across slot indices, so two concurrent
        in-flight inplace_async calls on different slots don't clobber each
        other's pinned outputs.
        """
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")
        if not 0 <= slot < self._n_slots:
            raise ValueError(f"slot {slot} out of range [0, {self._n_slots})")
        if not self._use_cuda:
            xb = self._pinned_inputs_np[slot][:bsz]
            return super().evaluate_encoded_async(xb, relations=relations)
        return self._async_forward(bsz, slot=slot, relations=relations)

    def evaluate_inplace_legal_bf16_async(
        self, bsz: int, legal_flat: np.ndarray, legal_counts: np.ndarray, *, slot: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        """Async eval returning compact legal policy logits as BF16 bit patterns.

        ``legal_flat`` is the concatenated policy-index list for each real row;
        ``legal_counts`` has length ``bsz``. The policy tensor returned is a
        pinned CPU ``uint16`` tensor containing bfloat16 bits in that compact
        order. WDL remains float32 because tablebase override mutates it before
        C integration, and it is only three logits per row.
        """
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")
        if not 0 <= slot < self._n_slots:
            raise ValueError(f"slot {slot} out of range [0, {self._n_slots})")
        counts = np.asarray(legal_counts, dtype=np.int32)
        if counts.ndim != 1 or counts.shape[0] != bsz:
            raise ValueError(f"legal_counts must be shape ({bsz},), got {counts.shape}")
        flat = np.asarray(legal_flat, dtype=np.int32)
        if flat.ndim != 1:
            raise ValueError(f"legal_flat must be 1D, got {flat.ndim}D")
        n_legal = int(counts.sum())
        if n_legal != int(flat.shape[0]):
            raise ValueError(f"legal_flat len {flat.shape[0]} != sum(legal_counts) {n_legal}")
        if n_legal > self._pinned_legal_pols_bf16_bits[slot].shape[0]:
            raise ValueError(
                f"compact legal policy has {n_legal} entries > "
                f"{self._pinned_legal_pols_bf16_bits[slot].shape[0]} capacity"
            )
        if not self._use_cuda:
            pol, wdl = self.evaluate_inplace(bsz, copy_out=True, slot=slot)
            rows = np.repeat(np.arange(bsz, dtype=np.int64), counts.astype(np.int64, copy=False))
            compact = pol[rows, flat.astype(np.int64, copy=False)].astype(np.float32, copy=False)
            bits = torch.from_numpy(compact).to(torch.bfloat16).view(torch.uint16)
            return bits, torch.from_numpy(wdl), None
        return self._async_forward_legal_bf16(bsz, flat, counts, n_legal, slot=slot)

    @property
    def supports_legal_bf16(self) -> bool:
        return self._legal_bf16

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synchronous compact-policy wrapper used by coalescing dispatchers.

        Pad only the board/count dimension to the compiled bucket. Legal logits
        remain unpadded, so transport stays proportional to actual legal moves.
        """
        if x.ndim != 4:
            raise ValueError(f"expected 4D input, got {x.ndim}D")
        real_n = int(x.shape[0])
        if real_n > self._max_batch:
            raise ValueError(f"batch {real_n} > max {self._max_batch}")
        counts = np.asarray(legal_counts, dtype=np.int32)
        if counts.ndim != 1 or counts.shape[0] != real_n:
            raise ValueError(
                f"legal_counts must be shape ({real_n},), got {counts.shape}"
            )
        flat = np.asarray(legal_flat, dtype=np.int32)
        if flat.ndim != 1:
            raise ValueError(f"legal_flat must be 1D, got {flat.ndim}D")
        self._stage_encoded_input(x, real_n, slot=0)
        padded = _compiled_padded_batch_size(real_n, capacity=self._max_batch)
        if padded > real_n:
            counts = np.pad(counts, (0, padded - real_n))
        pol_t, wdl_t, event = self.evaluate_inplace_legal_bf16_async(
            padded, flat, counts, slot=0,
        )
        if event is not None:
            event.synchronize()
        return pol_t.numpy().copy(), wdl_t[:real_n].numpy().copy()

    def _async_forward(
        self, bsz: int, *, slot: int, relations: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        pin_pol = self._pinned_pols[slot]
        pin_wdl = self._pinned_wdls[slot]

        if self._stream is None:
            self._stream = torch.cuda.Stream(device=self.device)
        stream = self._stream

        event_default = torch.cuda.Event()
        event_default.record(torch.cuda.current_stream(self.device))

        with torch.cuda.stream(stream):
            stream.wait_event(cast(Any, event_default))
            xt = self._device_input(bsz, slot=slot)
            out = _forward_no_grad(
                self.model, xt, device=self.device,
                use_amp=self._use_amp, amp_dtype=self._amp_dtype,
                relations_t=_relations_to_device(relations, device=self.device, bsz=bsz),
            )
            pin_pol[:bsz].copy_(_policy_output_full(out).detach().float(), non_blocking=True)
            pin_wdl[:bsz].copy_(out["wdl"].detach().float(), non_blocking=True)
            done = torch.cuda.Event()
            done.record(stream)

        return pin_pol[:bsz], pin_wdl[:bsz], done

    def _async_forward_legal_bf16(
        self, bsz: int, legal_flat: np.ndarray, legal_counts: np.ndarray, n_legal: int, *, slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        pin_pol = self._pinned_legal_pols_bf16_bits[slot]
        pin_wdl = self._pinned_wdls[slot]

        if self._stream is None:
            self._stream = torch.cuda.Stream(device=self.device)
        stream = self._stream

        event_default = torch.cuda.Event()
        event_default.record(torch.cuda.current_stream(self.device))

        with torch.cuda.stream(stream):
            stream.wait_event(cast(Any, event_default))
            xt = self._device_input(bsz, slot=slot)
            legal_counts_gpu = torch.as_tensor(legal_counts, dtype=torch.long, device=self.device)
            legal_flat_gpu = torch.as_tensor(legal_flat, dtype=torch.long, device=self.device)
            if _supports_legal_policy_forward(self.model):
                out = _forward_legal_no_grad(
                    self.model, xt, legal_flat_gpu, legal_counts_gpu,
                    device=self.device, use_amp=self._use_amp, amp_dtype=self._amp_dtype,
                )
            else:
                out = _forward_no_grad(
                    self.model, xt, device=self.device,
                    use_amp=self._use_amp, amp_dtype=self._amp_dtype,
                )
            if n_legal > 0:
                if _supports_legal_policy_forward(self.model):
                    legal_logits = _policy_output(out).to(torch.bfloat16)
                else:
                    policy = _policy_output_full(out)[:bsz]
                    rows = torch.repeat_interleave(torch.arange(bsz, device=self.device), legal_counts_gpu)
                    legal_logits = policy[rows, legal_flat_gpu].to(torch.bfloat16)
                pin_pol[:n_legal].copy_(legal_logits.view(torch.uint16), non_blocking=True)
            pin_wdl[:bsz].copy_(out["wdl"][:bsz].detach().float(), non_blocking=True)
            done = torch.cuda.Event()
            done.record(stream)

        return pin_pol[:n_legal], pin_wdl[:bsz], done


# Bucket ladder for compiled batched GPU forwards. Coarser than the per-batch
# AOT bucket list (_BATCH_BUCKETS) because torch.compile graphs are cheaper to
# recapture across mid-range sizes.
_COMPILED_BATCH_BUCKETS = (
    16, 32, 64, 96, 128, 170, 256, 340, 384, 512, 680, 768, 1020, 1024, 1190,
    1536, 1792, 2048, 2336, 2720, 4096,
)

_COMPILED_LEGAL_POLICY_BUCKETS = (
    32_768, 65_536, 131_072, 262_144, 524_288,
)


# --- Optional forward size histogram (bucket-design diagnostic) ------------
# Set CAE_BUCKET_HIST=/path/to/hist.json to record the RAW pre-pad forward
# sizes across whichever inference path is live (broker / dispatcher / threaded
# all pad through these two functions). Records two axes:
#   "batch"        — the forward batch total (sizes the _COMPILED_BATCH_BUCKETS
#                    / AOT _BATCH_BUCKETS ladder must cover)
#   "legal_policy" — the flat legal-move policy size (the second axis the
#                    compact-legal path buckets on, _COMPILED_LEGAL_POLICY_BUCKETS)
# Zero-cost when unset. Read the dumped {size: count} per axis to place buckets
# on the actual mass instead of the hand-picked ladders. See build_aot_packages.py.
_BUCKET_HIST_PATH = os.environ.get("CAE_BUCKET_HIST", "").strip()
_BUCKET_HIST_ENABLED = bool(_BUCKET_HIST_PATH)
_bucket_hist: dict[str, dict[int, int]] = {"batch": {}, "legal_policy": {}}
_bucket_hist_lock = threading.Lock()
_bucket_hist_records = 0
_BUCKET_HIST_DUMP_EVERY = int(os.environ.get("CAE_BUCKET_HIST_EVERY", "2000") or "2000")


def _dump_bucket_hist() -> None:
    """Atomically write both histograms to CAE_BUCKET_HIST (caller holds lock)."""
    payload = {
        "records": _bucket_hist_records,
        "compiled_batch_buckets": list(_COMPILED_BATCH_BUCKETS),
        "compiled_legal_policy_buckets": list(_COMPILED_LEGAL_POLICY_BUCKETS),
        "batch_counts": {str(k): v for k, v in sorted(_bucket_hist["batch"].items())},
        "legal_policy_counts": {
            str(k): v for k, v in sorted(_bucket_hist["legal_policy"].items())
        },
    }
    tmp = f"{_BUCKET_HIST_PATH}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, _BUCKET_HIST_PATH)
    except OSError:
        pass  # diagnostic only — never disturb the forward path


def _record_bucket_hist(axis: str, size: int) -> None:
    global _bucket_hist_records
    with _bucket_hist_lock:
        counts = _bucket_hist[axis]
        counts[size] = counts.get(size, 0) + 1
        _bucket_hist_records += 1
        if _bucket_hist_records % _BUCKET_HIST_DUMP_EVERY == 0:
            _dump_bucket_hist()


def _flush_bucket_hist_atexit() -> None:
    """Flush the tail (< dump cadence) on process exit so short windows count."""
    with _bucket_hist_lock:
        if _bucket_hist_records:
            _dump_bucket_hist()


if _BUCKET_HIST_ENABLED:
    import atexit

    atexit.register(_flush_bucket_hist_atexit)


# --- Optional arrival-trace JSONL (gather-policy diagnostic) ----------------
# Set CAE_ARRIVAL_TRACE=/path/to/trace.jsonl to record per-dispatched-batch
# arrival timestamps (perf_counter) as seen by SlotBroker.serve_forever.
# Offline replay: scripts/sim_gather_policy.py. Zero-cost when unset.
_ARRIVAL_TRACE_PATH = os.environ.get("CAE_ARRIVAL_TRACE", "").strip()
_ARRIVAL_TRACE_ENABLED = bool(_ARRIVAL_TRACE_PATH)
_arrival_trace_buf: list[str] = []
_arrival_trace_lock = threading.Lock()
_arrival_trace_last_flush = 0.0
_ARRIVAL_TRACE_FLUSH_EVERY_S = 5.0
_ARRIVAL_TRACE_FLUSH_EVERY_N = 200


def _flush_arrival_trace() -> None:
    """Append buffered JSONL lines to CAE_ARRIVAL_TRACE (caller holds lock)."""
    global _arrival_trace_last_flush
    if not _arrival_trace_buf:
        return
    try:
        with open(_ARRIVAL_TRACE_PATH, "a", encoding="utf-8") as fh:
            fh.writelines(_arrival_trace_buf)
    except OSError:
        pass  # diagnostic only — never disturb the forward path
    _arrival_trace_buf.clear()
    _arrival_trace_last_flush = time.perf_counter()


def _record_arrival_trace(record: dict[str, Any]) -> None:
    """Buffer one JSONL record; flush every ~5s or 200 batches."""
    line = json.dumps(record, separators=(",", ":")) + "\n"
    with _arrival_trace_lock:
        _arrival_trace_buf.append(line)
        now = time.perf_counter()
        if (
            len(_arrival_trace_buf) >= _ARRIVAL_TRACE_FLUSH_EVERY_N
            or (now - _arrival_trace_last_flush) >= _ARRIVAL_TRACE_FLUSH_EVERY_S
        ):
            _flush_arrival_trace()


def _flush_arrival_trace_atexit() -> None:
    """Flush the tail (< flush cadence) on process exit so short windows count."""
    with _arrival_trace_lock:
        _flush_arrival_trace()


if _ARRIVAL_TRACE_ENABLED:
    import atexit

    _arrival_trace_last_flush = time.perf_counter()
    atexit.register(_flush_arrival_trace_atexit)


def _compiled_padded_batch_size(total: int, *, capacity: int | None = None) -> int:
    if total <= 0:
        return 0
    if _BUCKET_HIST_ENABLED:
        _record_bucket_hist("batch", int(total))
    for bucket in _COMPILED_BATCH_BUCKETS:
        if bucket >= total:
            if capacity is not None and bucket > capacity:
                return total
            return bucket
    return total


def _compiled_padded_legal_policy_size(total: int, *, capacity: int | None = None) -> int:
    if total <= 0:
        return 0
    if _BUCKET_HIST_ENABLED:
        _record_bucket_hist("legal_policy", int(total))
    for bucket in _COMPILED_LEGAL_POLICY_BUCKETS:
        if bucket >= total:
            if capacity is not None and bucket > capacity:
                return total
            return bucket
    return total


# Batch size buckets for AOT-compiled inference models.
# Only sizes with a corresponding chess_b{N}.pt2 in the AOT dir are loaded.
_BATCH_BUCKETS = (
    1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64, 96,
    128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 170,
    192, 224, 228, 232, 236, 240, 244, 248, 256, 288, 340,
    384, 448, 512, 768, 1024, 1536, 2048, 3072, 4096,
)


def aot_package_filename(bucket: int) -> str:
    """Filename for a single batch-bucket AOT package."""
    return f"chess_b{int(bucket)}.pt2"


def select_aot_buckets(
    *,
    max_batch: int,
    buckets: Sequence[int],
) -> tuple[int, ...]:
    """Return bucket sizes ``<= max_batch`` from *buckets* (preserving order)."""
    mb = int(max_batch)
    if mb <= 0:
        raise ValueError(f"max_batch must be positive, got {max_batch}")
    return tuple(int(b) for b in buckets if int(b) <= mb)


def select_compiled_aot_buckets(*, max_batch: int) -> tuple[int, ...]:
    """Broker AOT ladder: ``_COMPILED_BATCH_BUCKETS`` filtered by *max_batch*."""
    return select_aot_buckets(max_batch=max_batch, buckets=_COMPILED_BATCH_BUCKETS)


def should_use_aot_forward(
    aot_models: Mapping[int, Any] | None,
    forward_total: int,
) -> bool:
    """True when AOT packages are loaded and *forward_total* has an exact package.

    Exact-key match only — the broker pads to ``_COMPILED_BATCH_BUCKETS`` values,
    so we must not re-pick from the finer ``_BATCH_BUCKETS`` ladder.
    """
    return bool(aot_models) and int(forward_total) in aot_models


def model_constant_source(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Complete constant source for AOT ``load_constants``.

    The AOT packages are built with ``package_constants_in_so=False`` +
    ``use_runtime_constant_folding=False``, which externalizes **every**
    constant — including non-persistent buffers (``persistent=False``) such as
    ``arc_pos_encoding``, the per-layer smolgen ``relation_basis_flat`` bases,
    and the policy-head lookup tables (``compact_to_full`` etc.). Those are
    absent from ``state_dict()``, so sourcing constants from a checkpoint
    state_dict alone leaves them unfilled and the first forward reads a null
    device pointer -> CUDA illegal memory access. Params + *all* buffers (this
    helper) covers the full ``get_constant_fqns()`` set.
    """
    # state_dict() = params + persistent buffers (+ any custom state_dict-hook
    # entries, e.g. smolgen gen_weight.weight, which are NOT plain named
    # parameters). named_buffers() adds the non-persistent buffers state_dict
    # omits. Their union covers the full get_constant_fqns() set.
    src: dict[str, torch.Tensor] = dict(model.state_dict())
    for name, buf in model.named_buffers():  # includes persistent=False buffers
        src.setdefault(name, buf)
    return src


def build_aot_constants(
    state_dict: Mapping[str, torch.Tensor],
    constant_fqns: Sequence[str],
    *,
    device: str,
) -> dict[str, torch.Tensor]:
    """Build ``load_constants`` payload from a checkpoint state_dict.

    Tensor prep: move to *device*, make contiguous, and cast **floating-point**
    constants to bf16 (the package's compiled dtype). Non-float constants —
    e.g. the integer policy lookup buffers ``to_sq``/``promo_from``/
    ``compact_to_full``/``to_valid`` — keep their source dtype; casting an int
    index buffer to bf16 corrupts it and mismatches the package's expected
    constant dtype (``load_constants`` -> CUDA "invalid argument"). Fails loud if
    any expected FQN is missing — a partial rebind must not silently leave
    packages on stale/random constants.
    """
    missing = [fqn for fqn in constant_fqns if fqn not in state_dict]
    if missing:
        preview = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise KeyError(
            f"checkpoint missing {len(missing)} AOT constant fqn(s): {preview}{more}"
        )

    def _prep(t: torch.Tensor) -> torch.Tensor:
        t = t.to(device=device)
        if t.is_floating_point():
            t = t.to(torch.bfloat16)
        return t.contiguous()

    return {fqn: _prep(state_dict[fqn]) for fqn in constant_fqns}


AOT_CHECK_FULL_UPDATE_ENV = "CAE_AOT_CHECK_FULL_UPDATE"


def aot_check_full_update_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether ``load_constants`` should validate full constant coverage.

    Default ON (audit I4). It used to be hardcoded ``False``, which explicitly
    tells AOTInductor not to complain about constants the payload does not
    cover — so a package built from a different architecture revision silently
    kept its build-time weights across every model publish, while the comment
    above the call promised the opposite.

    Kept behind an env kill switch because the flip could not be exercised on
    the GPU path here: ``aoti_load_package`` needs CUDA. What WAS verified,
    offline, is the precondition the check tests — all 21 packages in
    ``data/aot_models_512/`` declare the same 455 constant FQNs, unique and
    non-empty (read out of each ``.pt2``'s generated ``wrapper.cpp``), and the
    rebind payload is built from exactly that FQN list. Set
    ``CAE_AOT_CHECK_FULL_UPDATE=0`` to fall back if a real load disagrees.
    """
    env_map = os.environ if env is None else env
    raw = str(env_map.get(AOT_CHECK_FULL_UPDATE_ENV, "")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def assert_uniform_constant_fqns(models: Mapping[int, Any]) -> list[str]:
    """Return the shared constant-FQN list, refusing a heterogeneous package set.

    The rebind path builds ONE constants payload from ONE package's
    ``get_constant_fqns()`` and loads it into all of them. That is only sound if
    every package declares the same FQNs; nothing checked it, and
    ``check_full_update=False`` guaranteed the mismatch would be silent (audit
    I4). ``load_aot_packages`` skips missing files, so a partial rebuild — the
    exact output of a bucket-ladder change — is how this arms.
    """
    if not models:
        raise ValueError("assert_uniform_constant_fqns requires at least one package")
    items = sorted(models.items())
    ref_bucket, ref_model = items[0]
    ref_fqns = list(ref_model.get_constant_fqns())
    ref_set = set(ref_fqns)
    for bucket, model in items[1:]:
        fqns = set(model.get_constant_fqns())
        if fqns != ref_set:
            missing = sorted(ref_set - fqns)[:5]
            extra = sorted(fqns - ref_set)[:5]
            raise RuntimeError(
                f"AOT package set is not uniform: bucket {bucket} declares "
                f"{len(fqns)} constant fqns vs bucket {ref_bucket}'s {len(ref_set)} "
                f"(missing={missing}, extra={extra}). Rebinding one bucket's fqn "
                f"list into all of them would leave some packages on stale "
                f"build-time weights. Rebuild aot_dir from a single model."
            )
    return ref_fqns


def _aoti_load_package(path: str) -> Any:
    """Load one AOTInductor package; isolated for tests to monkeypatch."""
    # PyTorch 2.10's package loader accesses ``torch._inductor.codecache`` as
    # a parent-module attribute without importing that submodule first. A
    # fresh process can therefore fail before loading its first package.
    importlib.import_module("torch._inductor.codecache")
    return torch._inductor.aoti_load_package(path)


def load_aot_packages(
    aot_dir: Path | str,
    *,
    buckets: Sequence[int],
) -> dict[int, Any]:
    """Load ``chess_b{N}.pt2`` packages for exact bucket sizes; skip missing files.

    Raises ``FileNotFoundError`` if *aot_dir* is set but no package loads.
    """
    from concurrent.futures import ThreadPoolExecutor

    root = Path(aot_dir)
    pkgs: dict[int, Path] = {}
    for b in buckets:
        pkg = root / aot_package_filename(int(b))
        if pkg.exists():
            pkgs[int(b)] = pkg
    if not pkgs:
        raise FileNotFoundError(
            f"No .pt2 packages found in {root} for buckets {list(buckets)!r}"
        )

    def _load(item: tuple[int, Path]) -> tuple[int, Any]:
        return item[0], _aoti_load_package(str(item[1]))

    with ThreadPoolExecutor(max_workers=min(4, len(pkgs))) as pool:
        models = dict(pool.map(_load, list(pkgs.items())))
    log.info(
        "AOT packages loaded from %s: buckets=%s",
        root,
        sorted(models.keys()),
    )
    return models


class AOTEvaluator:
    """Evaluator using pre-compiled AOTInductor models for fixed bucket sizes.

    Each bucket size has a separate .pt2 package compiled with max-autotune.
    At inference time, the input is padded to the next bucket and dispatched
    to the matching pre-compiled model.  No torch.compile or CUDA graph
    capture at runtime.
    """

    def __init__(
        self,
        aot_dir: str | Path,
        *,
        device: str = "cuda",
        max_batch: int = 512,
        input_planes: int = _CHANNELS,
    ) -> None:
        self.device = str(device)
        self._max_batch = int(max_batch)
        self._input_planes = int(input_planes)
        # Load compiled models in parallel (CUDA driver is thread-safe for loading).
        buckets = select_aot_buckets(max_batch=self._max_batch, buckets=_BATCH_BUCKETS)
        self._models = load_aot_packages(aot_dir, buckets=buckets)
        self._sorted_buckets = sorted(self._models.keys())
        self._constant_fqns = assert_uniform_constant_fqns(self._models)

        # Pre-allocate pinned buffers
        _pin = self.device.startswith("cuda")
        self._n_slots = 2
        self._pinned_inputs = [
            torch.empty(
                (self._max_batch, self._input_planes, _BOARD_H, _BOARD_W),
                dtype=torch.float32, pin_memory=_pin,
            )
            for _ in range(self._n_slots)
        ]
        self._pinned_input = self._pinned_inputs[0]
        # NumPy writes directly into the tensor-owned pinned allocation. The
        # prior separate np.empty made non_blocking=True fall back to a
        # synchronous pageable transfer and left _pinned_input unused.
        self._pinned_inputs_np = [tensor.numpy() for tensor in self._pinned_inputs]
        self._pinned_input_np = self._pinned_inputs_np[0]
        self._pinned_inputs_bf16 = [
            torch.empty(
                (self._max_batch, self._input_planes, _BOARD_H, _BOARD_W),
                dtype=torch.bfloat16, pin_memory=_pin,
            )
            for _ in range(self._n_slots)
        ]
        self._pinned_input_bf16 = self._pinned_inputs_bf16[0]
        self._pinned_inputs_bf16_bits_np = [
            tensor.view(torch.uint16).numpy(force=True)
            for tensor in self._pinned_inputs_bf16
        ]
        self._pinned_input_bf16_bits_np = self._pinned_inputs_bf16_bits_np[0]
        self._slot_input_bf16 = [False] * self._n_slots

    def load_weights(self, source: Mapping[str, torch.Tensor]) -> None:
        """Update all bucket models with new weights.

        *source* must supply every ``get_constant_fqns()`` entry — pass
        :func:`model_constant_source` (params + all buffers), NOT a bare
        ``state_dict()``, which omits the non-persistent buffers the packages
        externalize (see that helper). Fails loud on a missing fqn: a silent
        drop would leave a package constant unfilled -> CUDA illegal memory
        access on the next forward.
        """
        constants = build_aot_constants(
            source, self._constant_fqns, device=self.device,
        )
        check_full = aot_check_full_update_enabled()
        for bucket, model in self._models.items():
            try:
                model.load_constants(constants, check_full_update=check_full)
            except Exception as exc:  # re-raised below with the bucket named
                raise RuntimeError(
                    f"AOT bucket {bucket} rejected the rebind of {len(constants)} "
                    f"constants (check_full_update={check_full}): {exc}. Rebuild "
                    f"the packages against this model rather than serving stale "
                    f"build-time weights."
                ) from exc

    def _pick_bucket(self, bsz: int) -> int:
        for b in self._sorted_buckets:
            if b >= bsz:
                return b
        return self._sorted_buckets[-1]

    @property
    def supports_input_bf16_bits(self) -> bool:
        return True

    @property
    def n_slots(self) -> int:
        return self._n_slots

    def get_input_buffer(self, bsz: int, slot: int = 0) -> np.ndarray:
        self._validate_input_slot(bsz, slot)
        self._slot_input_bf16[slot] = False
        return self._pinned_inputs_np[slot][:bsz]

    def get_input_buffer_bf16_bits(self, bsz: int, slot: int = 0) -> np.ndarray:
        self._validate_input_slot(bsz, slot)
        self._slot_input_bf16[slot] = True
        return self._pinned_inputs_bf16_bits_np[slot][:bsz]

    def _validate_input_slot(self, bsz: int, slot: int) -> None:
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")
        if not 0 <= slot < self._n_slots:
            raise ValueError(f"slot {slot} out of range [0, {self._n_slots})")

    def _device_input(self, x: np.ndarray, *, bucket: int, slot: int = 0) -> torch.Tensor:
        bsz = int(x.shape[0])
        if x.dtype == np.uint16:
            self._pinned_inputs_bf16_bits_np[slot][:bsz] = x
            self._slot_input_bf16[slot] = True
            return self._pinned_inputs_bf16[slot][:bucket].to(
                device=self.device, non_blocking=True,
            )
        self._pinned_inputs_np[slot][:bsz] = x
        self._slot_input_bf16[slot] = False
        return self._pinned_inputs[slot][:bucket].to(
            device=self.device, dtype=torch.bfloat16, non_blocking=True,
        )

    def _device_input_inplace(self, *, bucket: int, slot: int) -> torch.Tensor:
        if self._slot_input_bf16[slot]:
            return self._pinned_inputs_bf16[slot][:bucket].to(
                device=self.device, non_blocking=True,
            )
        return self._pinned_inputs[slot][:bucket].to(
            device=self.device, dtype=torch.bfloat16, non_blocking=True,
        )

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if relations is not None:
            raise NotImplementedError(
                "dynamic relations are not transported on the AOTEvaluator path; "
                "use worker-local direct inference (see "
                "check_dynamic_relations_transport)"
            )
        if x.ndim != 4:
            raise ValueError(f"expected 4D input, got {x.ndim}D")
        bsz = x.shape[0]
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")
        bucket = self._pick_bucket(bsz)
        model = self._models[bucket]

  # Native search can supply bit-packed BF16 directly; float32 callers retain
  # the compatibility path and convert during H2D.
        xt = self._device_input(x, bucket=bucket)

        with torch.no_grad():
            out = model(xt)

        pol = _policy_output_full(out)[:bsz].detach().float().cpu().numpy()
        wdl = out["wdl"][:bsz].detach().float().cpu().numpy()
        return pol, wdl

    def evaluate_encoded_async(
        self, x: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        if x.ndim != 4:
            raise ValueError(f"expected 4D input, got {x.ndim}D")
        bsz = x.shape[0]
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")
        bucket = self._pick_bucket(bsz)
        model = self._models[bucket]

        xt = self._device_input(x, bucket=bucket)

        with torch.no_grad():
            out = model(xt)

        pol = _policy_output_full(out)[:bsz].detach().to(
            dtype=torch.float32, device="cpu", non_blocking=True,
        )
        wdl = out["wdl"][:bsz].detach().to(
            dtype=torch.float32, device="cpu", non_blocking=True,
        )
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(torch.device(self.device)))
        return pol, wdl, done

    def evaluate_inplace_async(
        self,
        bsz: int,
        *,
        slot: int = 0,
        relations: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        if relations is not None:
            raise NotImplementedError(
                "dynamic relations are not transported on the AOTEvaluator path",
            )
        self._validate_input_slot(bsz, slot)
        bucket = self._pick_bucket(bsz)
        model = self._models[bucket]
        xt = self._device_input_inplace(bucket=bucket, slot=slot)
        with torch.no_grad():
            out = model(xt)
        pol = _policy_output_full(out)[:bsz].detach().to(
            dtype=torch.float32, device="cpu", non_blocking=True,
        )
        wdl = out["wdl"][:bsz].detach().to(
            dtype=torch.float32, device="cpu", non_blocking=True,
        )
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(torch.device(self.device)))
        return pol, wdl, done


# ---------------------------------------------------------------------------
# Slot-based shared memory inference
# ---------------------------------------------------------------------------
#
# Each worker owns one "slot" — a pre-allocated shared memory region with:
#
#   [0]        state      uint8   (see _STATE_* constants)
#   [1]        mode       uint8   (see _MODE_* constants)
#   [2:4]      pad
#   [4:8]      batch_size int32   (number of positions in this request)
#   [8:12]     request_id uint32  (client-stamped tag; broker echoes it — see below)
#   [12:16]    magic      uint32  (_SLOT_MAGIC; written once at slot creation)
#   [16:20]    layout_id  uint32  (_SlotLayout.identity(); written once at creation)
#   [20:24]    reserved
#   [24:...]   input      float32 or bf16-bits[max_batch, planes, 8, 8]
#   [after input]  policy/output float32[max_batch, 4672] or compact metadata/bf16 bits
#   [after policy] wdl    float32[max_batch, 3]
#
# Flow:
#   1. Worker writes input + batch_size + a FRESH request_id, sets state = REQUEST
#   2. Broker sees state == REQUEST, snapshots request_id FIRST, then reads input
#   3. Broker writes policy + wdl, writes back the snapshotted request_id, sets
#      state = RESPONSE
#   4. Worker reads output ONLY if the echoed request_id matches the one it
#      stamped; otherwise it discards the answer and re-submits.
#
# WHY the request_id exists (audit I1, 2026-08-03). The broker does not change
# slot state while it works: it leaves the slot in _STATE_REQUEST until it
# writes the answer. So when a client's request_timeout_s elapses mid-forward,
# the worker resets its client (worker.py `_reset_inference_client`) and
# re-submits into the SAME named slot; the broker then finishes the OLD request
# and marks the slot _STATE_RESPONSE. Without a tag the waiting client accepts
# that as the answer to its NEW request -- a policy/WDL belonging to a
# different position, fed to MCTS and recorded as training data, with nothing
# raised and no counter moved. On the compact-legal transport it is worse: the
# client slices policy_u16[:n_legal] with its OWN n_legal, so the tail of the
# row is re-interpreted bytes of its own request metadata.
#
# ORDERING is load-bearing on both sides and must not be "tidied":
#   * client writes input/metadata/batch_size/mode, THEN request_id, THEN state
#   * broker reads request_id BEFORE batch_size/metadata/input
# Together these make a mismatch conservative: the broker can only echo a tag
# whose payload it read at or after that tag was published, so "echoed ==
# pending" implies the payload is the pending request's. Reading the tag last
# would invert that and let a torn read pass as a match.
#
# WHAT MAKES THE ORDER SAFE, precisely: there is no fence here. Program order in
# CPython plus x86-64's store ordering (TSO -- stores are not reordered with
# other stores, loads not with other loads) is what carries it, and this stack
# is x86-64 only. The same source on a weakly-ordered target (arm64, POWER)
# would need explicit barriers; the guarantee above would silently stop holding
# rather than fail a test. Do not port the protocol without revisiting this.
#
# WHY magic/layout_id exist (audit I2). Both sides compute _SlotLayout.compute()
# independently and never exchanged the result. A client with a SMALLER plane
# count than the broker still fits every numpy view inside the larger segment,
# so the protocol completed normally and the client read its policy and WDL out
# of the middle of the broker's INPUT region -- all-zero policy + all-zero WDL,
# no exception. That is the exact zero-fill poisoning `_release_slots_for_retry`
# and tests/test_broker_no_zero_fill.py exist to make impossible, reached
# through a channel none of that work covers.
#
# No sockets, no per-request allocation, no connection setup/teardown.

_STATE_IDLE = 0
_STATE_REQUEST = 1
_STATE_RESPONSE = 2
_STATE_SHUTDOWN = 255
_MODE_DENSE_F32 = 0
_MODE_DENSE_BF16 = 1
_MODE_LEGAL_BF16 = 2

# Consecutive failed batches before the broker gives up and exits. High enough
# that a burst of client-timeout races rides through, low enough that a dead
# CUDA context does not livelock the fleet for an entire iteration.
_MAX_CONSECUTIVE_BATCH_FAILURES = 50

# Idle poll ladder for the serve loops: (idle_below_s, spin_polls, sleep_s).
# The first rung is the pre-2026-08-03 behaviour verbatim -- 200 unyielded polls
# then a 20µs sleep -- and it is what the loaded path uses, so batch latency and
# the gather-window regime are untouched. The rungs below it only engage after
# the broker has seen NOTHING for the given wall time, which under any real load
# never happens: a client that has just been answered re-submits within
# microseconds, and every arrival resets the ladder to rung 0. The cost this
# removes is the drought/pause/inter-iteration case, where the old loop burned
# ~83% of a core polling an empty slot set forever (audit I6).
#
# Worst case added latency is one rung's sleep, and a rung is only reachable
# after that rung's idle time has already elapsed: 5ms of sleep can only be
# paid by a request that arrives after >=500ms of total silence.
_IDLE_BACKOFF_LADDER: tuple[tuple[float, int, float], ...] = (
    (0.005, 200, 0.00002),
    (0.050, 200, 0.0002),
    (0.500, 32, 0.001),
    (float("inf"), 8, 0.005),
)


def _idle_backoff(idle_s: float) -> tuple[int, float]:
    """Spin count + sleep for a serve loop that has been idle for *idle_s*."""
    for limit, polls, sleep_s in _IDLE_BACKOFF_LADDER:
        if idle_s < limit:
            return polls, sleep_s
    last = _IDLE_BACKOFF_LADDER[-1]
    return last[1], last[2]


class BrokerModelUnavailable(RuntimeError):
    """No model is loaded, so the batch cannot be answered honestly.

    A distinct type because this is an EXPECTED, self-describing condition
    (the publish manifest is missing or has no sha), not a bug with a
    traceback worth reading. _process_batch logs it as a single throttled
    line instead of a full stack, since the released slots come straight back
    round the serve loop and an unthrottled log.exception would bury the real
    signal under identical tracebacks. It still counts toward the
    consecutive-failure ceiling.
    """


_HEADER_BYTES = 24  # see the header map above; 8-byte aligned so input views stay aligned
_OFF_STATE = 0
_OFF_MODE = 1
_OFF_BATCH_SIZE = 4
_OFF_REQUEST_ID = 8
_OFF_MAGIC = 12
_OFF_LAYOUT_ID = 16

# Bumped whenever the wire format changes. Broker and clients are launched
# together by distributed_runtime, so a format change is deployable as long as
# the whole fleet restarts.
#
# **The header guards protect only the NEW side, and the earlier version of this
# comment claimed otherwise.** A client from a previous generation runs OLD
# code, which has no size/magic/layout validation to fail; and because the v2
# header grew at the FRONT while state@0 / mode@1 / batch_size@4 kept their
# offsets, the state machine still completes with the two sides 16 bytes out of
# phase. Measured across two worktrees, an origin/main client on a v2 broker's
# segment returned an ALL-ZERO policy and an ALL-ZERO WDL with no exception --
# exactly the poisoning this protocol version exists to delete. A version bump
# can never make an old peer refuse; only a NEW peer can refuse.
#
# What actually makes the old-client direction safe is that the version is in
# the slot NAME as well as the header (distributed_runtime._trial_slot_prefix
# and SharedSlotBroker._register_new_trial). A stale worker then looks up a name
# that does not exist and gets FileNotFoundError -> TimeoutError instead of
# mapping a v2 segment. **Both must be bumped together**, and the name is the
# one that matters for backward safety.
SLOT_PROTOCOL_VERSION = 2
_SLOT_MAGIC = 0x43414532  # b"CAE2"


def trial_slot_prefix(*, trial_id: str) -> str:
    """The ONE derivation of a trial's shared-memory slot name prefix.

    The launcher tells workers a name and the shared broker creates one; they
    must be byte-identical or every worker times out against a segment nobody
    created — silent from the broker's side, since it simply never sees a
    request. They used to be two hand-written copies of the same formula, and
    the test that "checked they agree" was a third copy: mutating either
    implementation left it green (PR #322 review). One function, two
    delegating call sites, and a test that calls the real ones.

    The version belongs in the NAME, not only in the header: see the
    SLOT_PROTOCOL_VERSION comment above for why a header bump cannot make a
    stale client refuse, and the name can.
    """
    from chess_anti_engine.tune._utils import (
        stable_seed_u32,  # deferred: avoids a circular import
    )

    h = stable_seed_u32(f"slot-prefix-v{SLOT_PROTOCOL_VERSION}", trial_id)
    return f"cae{SLOT_PROTOCOL_VERSION}-{h:08x}"

# request_id 0 is reserved for "no request has been stamped" (a freshly created
# or zeroed segment), so a client never treats a virgin slot as an echo.
_REQUEST_ID_NONE = 0
_REQUEST_ID_MASK = 0xFFFFFFFF


class SlotProtocolMismatch(RuntimeError):
    """The attached shared-memory segment is not a slot this client can use.

    Raised by ``SlotInferenceClient._connect`` when the segment is too small for
    the client's layout, carries a wrong/absent magic, or was created for a
    different layout (different max_batch or plane count). Deliberately fatal
    rather than a retry: every read on a mismatched layout is silently wrong
    (see the I2 note on the header map), so there is nothing to wait for.
    """


@dataclass(frozen=True)
class _SlotLayout:
    max_batch: int
    channels: int
    input_offset: int
    input_bytes: int
    policy_offset: int
    policy_bytes: int
    wdl_offset: int
    wdl_bytes: int
    total_bytes: int

    def identity(self) -> int:
        """Stable 32-bit id of everything both sides must agree on.

        Written into the segment header at creation and compared on connect.
        Covers the protocol version and every field that moves an offset, so a
        plane-count or max_batch skew cannot present as a compatible slot.
        """
        payload = (
            f"cae-slot-v{SLOT_PROTOCOL_VERSION}"
            f"|mb={self.max_batch}|ch={self.channels}"
            f"|io={self.input_offset}|ib={self.input_bytes}"
            f"|po={self.policy_offset}|pb={self.policy_bytes}"
            f"|wo={self.wdl_offset}|wb={self.wdl_bytes}"
            f"|tb={self.total_bytes}"
        ).encode("ascii")
        return zlib.crc32(payload) & _REQUEST_ID_MASK

    @staticmethod
    def compute(max_batch: int, channels: int = _CHANNELS) -> _SlotLayout:
        ib = max_batch * channels * _BOARD_H * _BOARD_W * _F32_BYTES
        pb = max_batch * _POLICY_SIZE * _F32_BYTES
        wb = max_batch * _WDL_SIZE * _F32_BYTES
        io = _HEADER_BYTES
        po = io + ib
        wo = po + pb
        return _SlotLayout(
            max_batch=max_batch,
            channels=channels,
            input_offset=io,
            input_bytes=ib,
            policy_offset=po,
            policy_bytes=pb,
            wdl_offset=wo,
            wdl_bytes=wb,
            total_bytes=wo + wb,
        )


class _InferenceSlot:
    """Numpy-backed view into a pre-allocated shared memory slot."""

    __slots__ = (
        "_buf",
        "_layout",
        "_owns",
        "_shm",
        "input",
        "input_bf16_bits",
        "policy",
        "policy_i32",
        "policy_u16",
        "wdl",
    )
    _buf: memoryview

    def __init__(self, shm: SharedMemory, layout: _SlotLayout, *, owns: bool = False):
        self._shm = shm
        self._layout = layout
        self._owns = owns
        assert shm.buf is not None  # attached SharedMemory always has a buffer
        self._buf = shm.buf
        self.input: np.ndarray = np.ndarray(
            (layout.max_batch, layout.channels, _BOARD_H, _BOARD_W),
            dtype=_F32,
            buffer=self._buf,
            offset=layout.input_offset,
        )
        self.input_bf16_bits: np.ndarray = np.ndarray(
            (layout.max_batch, layout.channels, _BOARD_H, _BOARD_W),
            dtype=np.uint16,
            buffer=self._buf,
            offset=layout.input_offset,
        )
        self.policy: np.ndarray = np.ndarray(
            (layout.max_batch, _POLICY_SIZE),
            dtype=_F32,
            buffer=self._buf,
            offset=layout.policy_offset,
        )
        self.policy_i32: np.ndarray = np.ndarray(
            (layout.policy_bytes // 4,),
            dtype=np.int32,
            buffer=self._buf,
            offset=layout.policy_offset,
        )
        self.policy_u16: np.ndarray = np.ndarray(
            (layout.policy_bytes // 2,),
            dtype=np.uint16,
            buffer=self._buf,
            offset=layout.policy_offset,
        )
        self.wdl: np.ndarray = np.ndarray(
            (layout.max_batch, _WDL_SIZE),
            dtype=_F32,
            buffer=self._buf,
            offset=layout.wdl_offset,
        )

    @property
    def state(self) -> int:
        return int(self._buf[_OFF_STATE])

    @state.setter
    def state(self, v: int) -> None:
        self._buf[_OFF_STATE] = int(v) & 0xFF

    @property
    def batch_size(self) -> int:
        return struct.unpack_from("<i", self._buf, _OFF_BATCH_SIZE)[0]

    @batch_size.setter
    def batch_size(self, v: int) -> None:
        struct.pack_into("<i", self._buf, _OFF_BATCH_SIZE, int(v))

    @property
    def request_mode(self) -> int:
        return int(self._buf[_OFF_MODE])

    @request_mode.setter
    def request_mode(self, v: int) -> None:
        self._buf[_OFF_MODE] = int(v) & 0xFF

    @property
    def request_id(self) -> int:
        """Client-stamped tag for the request currently in the slot (audit I1).

        Read by the broker BEFORE the payload and echoed back on response; the
        client accepts a response only when this matches what it stamped. See
        the ordering note on the header map — this must not be read after the
        payload.
        """
        return struct.unpack_from("<I", self._buf, _OFF_REQUEST_ID)[0]

    @request_id.setter
    def request_id(self, v: int) -> None:
        struct.pack_into("<I", self._buf, _OFF_REQUEST_ID, int(v) & _REQUEST_ID_MASK)

    def write_protocol_header(self) -> None:
        """Stamp layout id + magic. Called once by whoever CREATES the segment.

        **Magic is written LAST and is the publish barrier.** A client can attach
        to a freshly created segment at any point, so anything it validates must
        already be in place by the time the magic it keys on becomes visible.
        Writing magic first would let a client read a valid magic beside an
        unwritten layout id and reject a perfectly good slot.
        """
        self.request_id = _REQUEST_ID_NONE
        struct.pack_into("<I", self._buf, _OFF_LAYOUT_ID, self._layout.identity())
        struct.pack_into("<I", self._buf, _OFF_MAGIC, _SLOT_MAGIC)

    @property
    def name(self) -> str:
        return self._shm.name

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._shm.close()
        if self._owns:
            with contextlib.suppress(Exception):
                self._shm.unlink()


# ---------------------------------------------------------------------------
# Broker (runs in its own process, one per trial)
# ---------------------------------------------------------------------------


class SlotBroker:
    """Per-trial inference broker using slot-based shared memory.

    Creates *num_slots* pre-allocated shared memory regions (one per worker).
    The main loop polls all slots, collects ready requests, batches them
    into a single GPU forward pass, and scatters results back.
    """

    def __init__(
        self,
        *,
        publish_dir: Path,
        num_slots: int,
        max_batch_per_slot: int,
        device: str,
        compile_inference: bool,
        batch_wait_ms: float,
        slot_prefix: str,
        compile_mode: str = "reduce-overhead",
        input_planes: int = _CHANNELS,
        adaptive_idle_ms: float = 0.0,
        hang_abort_seconds: float = _DEFAULT_HANG_ABORT_S,
        aot_dir: str | None = None,
    ) -> None:
        # Boot-timeline anchors for measuring cold start (broker.out).
        self._boot_t0 = time.perf_counter()
        self._boot_wall0 = time.time()
        self._first_batch_logged = False
        self.publish_dir = Path(publish_dir)
        self.device = str(device)
        self.compile_inference = bool(compile_inference)
        self.compile_mode = str(compile_mode or "reduce-overhead")
        self._first_inference_pending = False
        self.batch_wait_ms = float(batch_wait_ms)
        self.adaptive_idle_ms = float(adaptive_idle_ms)
        self._hang_watchdog = BrokerHangWatchdog(
            threshold_s=float(hang_abort_seconds),
            boot_threshold_s=resolve_boot_hang_abort_seconds(float(hang_abort_seconds)),
        )
        # Started HERE, not in serve_forever: everything below this line in
        # __init__ (AOT package load/replay, pinned-buffer allocation) is a CUDA
        # call on a possibly-wedged bridge, and MEMORY.md records exactly that
        # -- "loading/replaying AOT cudagraph packages at startup reliably trips
        # the flaky bridge". The old watchdog started after all of it and was
        # inert until the first successful forward, so the boot-into-wedged-dxg
        # scenario its own docstring names could never fire it (audit I3).
        self._hang_watchdog.start()
        self._consecutive_batch_failures = 0
      # Throttle for the no-model warning; see BrokerModelUnavailable.
        self._no_model_warned_at = 0.0
      # Malformed compact-legal metadata: lifetime count plus a warn throttle.
      # Lifetime rather than per-interval because the metrics line that reports
      # it only prints when a batch completed, and a slot rejected here never
      # reaches a batch -- a monotonic total still grows visibly across reports.
        self._malformed_legal_meta_total = 0
        self._malformed_legal_meta_warned_at = 0.0
        self._model: torch.nn.Module | None = None
        self._model_sha: str | None = None
        self._stop = False
        self._manifest_cache: dict | None = None
        self._manifest_cache_sig: tuple[int, int] | None = None
        self._timing_metrics: dict[str, float] | None = None
        # Optional AOTInductor packages keyed by exact compiled batch bucket.
        # None when aot_dir is unset/empty (zero behaviour change); non-empty
        # dict when packages loaded. Per-batch uncovered buckets fall back to
        # eager self._model — setup/rebind errors fail loud instead.
        self._aot_models: dict[int, Any] | None = None
        self._aot_constant_fqns: list[str] = []
        # Per-slot first-seen REQUEST times for CAE_ARRIVAL_TRACE (id(slot) ->
        # perf_counter). Only written when the env diagnostic is enabled.
        self._arrival_trace_seen: dict[int, float] = {}

        self._layout = _SlotLayout.compute(max_batch_per_slot, int(input_planes))
        self._slots: list[_InferenceSlot] = []
        self._slot_names: list[str] = []
        # Monotonic start of the current no-work stretch (None = work flowing).
        self._idle_since: float | None = None

        print(
            f"[broker] boot start wall={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self._boot_wall0))} "
            f"slots={num_slots} max_batch_per_slot={max_batch_per_slot} "
            f"compile={self.compile_inference} mode={self.compile_mode} "
            f"aot_dir={aot_dir or ''!r}",
            flush=True,
        )

        # Load AOT packages before allocating SHM so a missing/misconfigured
        # aot_dir fails without leaving orphan shared-memory segments.
        aot_raw = str(aot_dir or "").strip()
        if aot_raw:
            # Pad capacity matches pin-buffer total: num_slots * max_batch_per_slot.
            aot_max_batch = int(num_slots) * int(max_batch_per_slot)
            aot_buckets = select_compiled_aot_buckets(max_batch=aot_max_batch)
            if not aot_buckets:
                raise FileNotFoundError(
                    f"aot_dir set ({aot_raw!r}) but no _COMPILED_BATCH_BUCKETS "
                    f"<= max_batch={aot_max_batch}"
                )
            t_aot = time.perf_counter()
            with self._hang_watchdog.stage("aot_package_load"):
                self._aot_models = load_aot_packages(aot_raw, buckets=aot_buckets)
                self._aot_constant_fqns = assert_uniform_constant_fqns(self._aot_models)
            print(
                f"[broker] AOT packages loaded n={len(self._aot_models)} "
                f"elapsed_s={time.perf_counter() - t_aot:.2f} "
                f"since_boot_s={time.perf_counter() - self._boot_t0:.2f}",
                flush=True,
            )

        # Pre-allocated pinned buffers for zero-copy GPU transfer.
        # Under a watchdog stage: cudaHostAlloc is a driver call and blocks
        # forever on a wedged bridge exactly like a forward does.
        _total_cap = num_slots * max_batch_per_slot
        _pin = "cuda" in self.device and torch.cuda.is_available()
        _legal_metadata_cap = _total_cap * 256 if _pin else 0
        with self._hang_watchdog.stage("pinned_buffer_alloc"):
            self._pinned_input = torch.empty(
                (_total_cap, int(input_planes), _BOARD_H, _BOARD_W),
                dtype=torch.float32, pin_memory=_pin,
            )
            self._pinned_input_bf16 = torch.empty(
                (_total_cap, int(input_planes), _BOARD_H, _BOARD_W),
                dtype=torch.bfloat16, pin_memory=_pin,
            )
            self._pinned_pol = torch.empty(
                (_total_cap, _POLICY_SIZE), dtype=torch.float32, pin_memory=_pin,
            )
            self._pinned_wdl = torch.empty(
                (_total_cap, _WDL_SIZE), dtype=torch.float32, pin_memory=_pin,
            )
            self._pinned_legal_pol_bf16_bits = torch.empty(
                (_total_cap * 256,), dtype=torch.uint16, pin_memory=_pin,
            )
            self._pinned_legal_flat = torch.empty(
                (_legal_metadata_cap,), dtype=torch.long, pin_memory=_pin,
            )
            self._pinned_legal_rows = torch.empty(
                (_legal_metadata_cap,), dtype=torch.long, pin_memory=_pin,
            )
  # Pinned tensors need force=True for numpy conversion.
        self._pinned_input_np = self._pinned_input.numpy(force=True)
        self._pinned_input_bf16_bits_np = self._pinned_input_bf16.view(torch.uint16).numpy(force=True)
        self._pinned_pol_np = self._pinned_pol.numpy(force=True)
        self._pinned_wdl_np = self._pinned_wdl.numpy(force=True)
        self._pinned_legal_pol_bf16_bits_np = self._pinned_legal_pol_bf16_bits.numpy(force=True)
        self._pinned_legal_flat_np = self._pinned_legal_flat.numpy(force=True)
        self._pinned_legal_rows_np = self._pinned_legal_rows.numpy(force=True)

        for i in range(num_slots):
            name = f"{slot_prefix}-{i}"
  # Clean up stale shm with the same name
            try:
                old = SharedMemory(name=name, create=False)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            shm = SharedMemory(name=name, create=True, size=self._layout.total_bytes)
            slot = _InferenceSlot(shm, self._layout, owns=True)
            slot.state = _STATE_IDLE
            slot.batch_size = 0
            slot.request_mode = _MODE_DENSE_F32
            slot.write_protocol_header()
            self._slots.append(slot)
            self._slot_names.append(name)

        print(
            f"[broker] slots ready n={len(self._slots)} "
            f"planes={self._layout.channels} "
            f"layout_id={self._layout.identity():#010x} "
            f"proto_v={SLOT_PROTOCOL_VERSION} "
            f"since_boot_s={time.perf_counter() - self._boot_t0:.2f}",
            flush=True,
        )

    @property
    def slot_names(self) -> list[str]:
        return list(self._slot_names)

    def _since_boot_s(self) -> float:
        return float(time.perf_counter() - self._boot_t0)

  # -- model loading (same logic as before) --

    def _load_manifest_if_changed(self) -> dict:
        mf = self.publish_dir / "manifest.json"
        stat = mf.stat()
        sig = (int(stat.st_mtime_ns), int(stat.st_size))
        if self._manifest_cache is not None and self._manifest_cache_sig == sig:
            return self._manifest_cache
        m = dict(json.loads(mf.read_text(encoding="utf-8")))
        self._manifest_cache = m
        self._manifest_cache_sig = sig
        return m

    def _ensure_model(self) -> None:
        deadline = time.monotonic() + 30.0
        while True:
            try:
                manifest = self._load_manifest_if_changed()
                break
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    return
                time.sleep(0.5)
        model_info = manifest.get("model") or {}
        model_sha = str(model_info.get("sha256") or "")
        if not model_sha:
            return
        if self._model is not None and self._model_sha == model_sha:
            return

        # Force-off gradient checkpointing for inference: the manifest value
        # reflects the trainer's setting, but the broker is eval-only and
        # any grad-ckpt hooks would diverge from SharedSlotBroker (which
        # also hardcodes False) in compiled-graph shape.
        model_cfg = dataclass_replace(
            model_config_from_manifest_dict(manifest.get("model_config") or {}),
            use_gradient_checkpointing=False,
        )
        # Plane-count binding (audit I2/I4). The shm layout, the pinned buffers
        # and — when AOT is on — the compiled packages were all sized from the
        # broker's --input-planes; the model about to be served is sized by the
        # manifest. A disagreement means every request is read at the wrong
        # offsets and every forward is fed the wrong shape, so it must raise
        # rather than serve a silent all-zero policy.
        #
        # This is a PER-BATCH error, not a startup one: _ensure_model is only
        # reachable from _process_batch_mode, so the raise is caught by
        # _process_batch, logged with a traceback, the slots are released for
        # retry, and it takes _MAX_CONSECUTIVE_BATCH_FAILURES (50) consecutive
        # batches to exit the broker. Loud and self-limiting, but it is not a
        # refusal to boot -- do not read it as one.
        model_planes = int(input_plane_count(model_cfg.input_extra_features))
        if model_planes != int(self._layout.channels):
            raise SlotProtocolMismatch(
                f"published model wants {model_planes} input planes "
                f"(input_extra_features={model_cfg.input_extra_features!r}) but this "
                f"broker was launched with --input-planes {int(self._layout.channels)}; "
                "refusing to serve a model the slot layout cannot carry"
            )

        model_path = self.publish_dir / "latest_model.pt"
        ckpt = torch.load(str(model_path), map_location="cpu")
        sd = ckpt.get("model", ckpt)
        raw_model = build_model(model_cfg)
        require_model_planes(
            raw_model, self._layout.channels, where="SlotBroker slot layout",
        )
        load_state_dict_tolerant(raw_model, sd, label="broker-model")
        raw_model.to(self.device)
        raw_model.eval()
        if hasattr(raw_model, "_inference_only"):
            setattr(raw_model, "_inference_only", True)

        # Rebind AOT package constants BEFORE compiling — model_constant_source
        # must read the raw module. torch.compile wraps it in an OptimizedModule
        # whose named_parameters/buffers are all "_orig_mod."-prefixed, which
        # matches none of the package fqns -> "missing 455 constants". Fail loud
        # on any real miss — never leave packages on stale weights while the
        # eager model has been swapped.
        if self._aot_models:
            if not isinstance(sd, dict):
                raise TypeError(
                    f"broker model checkpoint is not a state_dict mapping "
                    f"(got {type(sd).__name__}); cannot rebind AOT packages"
                )
            # Source from the materialized raw model (params + ALL buffers) — the
            # checkpoint sd omits the non-persistent buffers the packages
            # externalize (arc_pos_encoding, smolgen relation bases, policy
            # lookup tables); using it alone -> IMA (see model_constant_source).
            constants = build_aot_constants(
                model_constant_source(raw_model),
                self._aot_constant_fqns,
                device=self.device,
            )
            check_full = aot_check_full_update_enabled()
            for bucket, aot_model in self._aot_models.items():
                try:
                    aot_model.load_constants(constants, check_full_update=check_full)
                except Exception as exc:  # re-raised below with the bucket named
                    raise RuntimeError(
                        f"AOT bucket {bucket} rejected the rebind of "
                        f"{len(constants)} constants (check_full_update={check_full}): "
                        f"{exc}. The package set in aot_dir does not match the "
                        f"published model — rebuild it (see "
                        f"scripts/build_aot_packages.py) rather than serving "
                        f"stale build-time weights."
                    ) from exc

        t_model = time.perf_counter()
        if self.compile_inference and self.device.startswith("cuda"):
            model = cast(
                "torch.nn.Module",
                torch.compile(raw_model, mode=self.compile_mode),
            )
        else:
            model = raw_model
        self._model = model
        self._model_sha = model_sha
        self._first_inference_pending = bool(self.compile_inference)
        print(
            f"[broker] model ready sha={model_sha[:8]} "
            f"compile={self.compile_inference} aot={bool(self._aot_models)} "
            f"model_step_s={time.perf_counter() - t_model:.2f} "
            f"since_boot_s={self._since_boot_s():.2f}",
            flush=True,
        )

  # -- batch processing --

    def _process_batch(self, ready: list[_InferenceSlot]) -> int:
        """Evaluate the ready slots, grouped by request_mode.

        Returns the number of ROWS handed back UNANSWERED. ``serve_forever``
        adds ``sum(s.batch_size for s in ready)`` to its ``positions`` total
        BEFORE calling this, and three paths in here release slots without an
        answer: ``BrokerModelUnavailable``, a generic batch failure, and
        malformed compact-legal metadata (the last one inside
        ``_process_batch_mode``, which returns its own count for the same
        reason). Without the return, a broker that answered nothing at all
        still reported full ``pos/s`` -- the failure reads as throughput.

        Third and last of the three known instances of that family: B6 fixed
        it in ``MultiSlotInferenceClient`` and B6(c) in
        ``SharedSlotBroker._process_parallel``; this is the production broker
        (task #142).
        """
        # The request tag is read HERE, before request_mode, because this is the
        # first field of the slot the broker touches. Reading mode first and the
        # tag later would leave a window in which a re-submitting client writes
        # mode -> batch_size -> tag between the two reads: the broker would then
        # decode the NEW payload in the OLD mode and echo the NEW tag, and the
        # client would accept a mis-decoded answer as its own. The whole
        # guarantee is "the tag is read before anything it vouches for" (audit
        # I1); the comprehension below evaluates left to right, so it holds.
        snapshot = [
            (slot, int(slot.request_id), int(slot.request_mode)) for slot in ready
        ]
        by_mode: dict[int, list[tuple[_InferenceSlot, int]]] = {}
        for slot, req_id, mode in snapshot:
            by_mode.setdefault(mode, []).append((slot, req_id))
        unanswered_rows = 0
        for mode, entries in by_mode.items():
            slots = [s for s, _ in entries]
            request_ids = [r for _, r in entries]
            try:
                mode_unanswered = self._process_batch_mode(
                    slots, mode=mode, request_ids=request_ids,
                )
            except BrokerModelUnavailable as exc:
                # Counted BEFORE the release, and this ordering is load-bearing:
                # _release_slots_for_retry sets _STATE_IDLE, after which the
                # client is free to re-submit into the same shared slot with a
                # different batch_size. Reading afterwards would subtract a row
                # count belonging to the NEXT request. Same rule below and in
                # _process_batch_mode's malformed-metadata branch.
                unanswered_rows += sum(
                    _slot_rows(s, self._layout.max_batch) for s in slots
                )
                # Same recovery as any other failed batch, but logged as one
                # throttled line rather than a traceback: this path repeats
                # every serve loop until a model appears, and 50 identical
                # stacks before the exit would bury the reason for the exit.
                _now_nm = time.monotonic()
                if _now_nm - self._no_model_warned_at >= 30.0:
                    self._no_model_warned_at = _now_nm
                    log.warning(
                        "broker cannot answer mode=%d slots=%d: %s; releasing "
                        "slots for retry (%d consecutive failures)",
                        mode, len(slots), exc, self._consecutive_batch_failures + 1,
                    )
                self._release_slots_for_retry(slots)
                self._consecutive_batch_failures += 1
                if self._consecutive_batch_failures >= _MAX_CONSECUTIVE_BATCH_FAILURES:
                    log.error(
                        "broker had no model for %d batches in a row; exiting so "
                        "the next selfplay phase relaunches a clean broker",
                        self._consecutive_batch_failures,
                    )
                    raise
            except Exception:
                # Before the release, for the reason given above. A partial
                # failure -- _process_batch_mode released some malformed slots
                # and then raised -- loses its own return value, so counting
                # the whole group here is what keeps those rows counted; the
                # group is unanswered either way.
                unanswered_rows += sum(
                    _slot_rows(s, self._layout.max_batch) for s in slots
                )
                # One bad batch must not take down the broker. Every worker in
                # the fleet depends on this process and nothing relaunches it
                # until the next selfplay phase begins, so a crash here costs a
                # whole iteration of selfplay -- on 2026-07-24 a single
                # ValueError cost ~50 minutes of zero games.
                log.exception(
                    "broker batch failed mode=%d slots=%d; releasing slots for retry",
                    mode, len(slots),
                )
                self._release_slots_for_retry(slots)
                self._consecutive_batch_failures += 1
                if self._consecutive_batch_failures >= _MAX_CONSECUTIVE_BATCH_FAILURES:
                    # Persistent failure is a dead CUDA context or a real bug,
                    # not a race. Keeping the process alive then would only
                    # livelock the fleet, so die and let the next phase start a
                    # broker with a clean context.
                    log.error(
                        "broker failed %d batches in a row; exiting so the next "
                        "selfplay phase relaunches a clean broker",
                        self._consecutive_batch_failures,
                    )
                    raise
            else:
                self._consecutive_batch_failures = 0
                # The rows _process_batch_mode itself declined (malformed
                # compact-legal metadata) while the group otherwise succeeded.
                # Accumulated HERE rather than at the call site inside the try:
                # there, a callee that returned None would raise TypeError into
                # the generic handler above, be swallowed as "one bad batch",
                # and silently walk _consecutive_batch_failures toward a broker
                # exit. Outside the try it is loud, which is what a wrong return
                # type deserves.
                unanswered_rows += mode_unanswered
        return unanswered_rows

    def _release_slots_for_retry(self, slots: list[_InferenceSlot]) -> None:
        """Hand slots back to their clients unanswered so they re-submit.

        Deliberately NOT a zero-filled response: the clients feed selfplay, and
        a fabricated all-zero policy would be recorded as training data rather
        than raising. _STATE_IDLE is the state the client's own recovery path
        already treats as "slot went away, re-submit" (see
        _submit_and_wait_locked), so a transient failure costs a retry and a
        persistent one surfaces as a loud client-side TimeoutError.
        """
        for slot in slots:
            try:
                slot.state = _STATE_IDLE
            except Exception:
                log.exception("failed to release broker slot after a batch error")

    def _stage_legal_metadata_cuda(
        self,
        legal_flat: np.ndarray,
        legal_rows: np.ndarray,
        *,
        transfer_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage compact gather metadata in reusable pinned host buffers."""
        legal_real = int(legal_flat.size)
        self._pinned_legal_flat_np[:legal_real] = legal_flat
        self._pinned_legal_rows_np[:legal_real] = legal_rows
        if transfer_size > legal_real:
            self._pinned_legal_flat_np[legal_real:transfer_size].fill(0)
            self._pinned_legal_rows_np[legal_real:transfer_size].fill(0)
        return (
            self._pinned_legal_flat[:transfer_size].to(self.device, non_blocking=True),
            self._pinned_legal_rows[:transfer_size].to(self.device, non_blocking=True),
        )

    def _process_batch_mode(
        self, ready: list[_InferenceSlot], *, mode: int, request_ids: list[int],
    ) -> int:
        """Evaluate one request_mode's slots.

        Returns the number of ROWS released unanswered for malformed
        compact-legal metadata, which ``_process_batch`` forwards to the serve
        loop's ``positions`` total. See that method's docstring for why.
        """
        _timing = getattr(self, "_timing_metrics", None)
        _t_pack0 = time.perf_counter()
        # Under a watchdog stage (audit I3): torch.load, .to(device), the AOT
        # load_constants loop and torch.compile all live in here, and they are
        # the CUDA calls most likely to block forever on a cold boot into a
        # wedged bridge. Before this the instrumented window started AFTER
        # _ensure_model returned.
        with self._hang_watchdog.stage("ensure_model"):
            self._ensure_model()
        if self._model is None:
            # Deliberately NOT a zero-filled response, for the reason spelled
            # out on _release_slots_for_retry: an all-zero policy+WDL marked
            # _STATE_RESPONSE is indistinguishable from a real answer, so it
            # is fed to MCTS and RECORDED AS TRAINING DATA instead of raising.
            # _ensure_model returns silently on two paths (manifest still
            # missing at its 30s deadline; manifest carries no model sha), and
            # the old zero-fill also returned normally -- so _process_batch
            # scored it a success and reset _consecutive_batch_failures,
            # meaning a broker that never got a model served zeros forever
            # with no log line and no escalation.
            #
            # Raising hands this to _process_batch's handler, which releases
            # the slots unanswered (clients re-submit), logs, and counts
            # toward _MAX_CONSECUTIVE_BATCH_FAILURES so a persistently
            # model-less broker exits and the next phase relaunches a clean
            # one. Loud and recoverable beats silent and poisonous.
            raise BrokerModelUnavailable(
                "broker has no model loaded (publish manifest missing past its "
                "deadline, or carries no model sha); refusing to answer with a "
                "fabricated all-zero policy"
            )

        use_bf16_input = mode in (_MODE_DENSE_BF16, _MODE_LEGAL_BF16)
        compact_legal = mode == _MODE_LEGAL_BF16

  # Gather inputs directly into pre-allocated pinned buffer (one memcpy
  # from shm → pinned, then async DMA to GPU — no intermediate allocs).
        active: list[_InferenceSlot] = []
        batch_sizes: list[int] = []
        active_request_ids: list[int] = []
        legal_counts_by_slot: list[np.ndarray] = []
        legal_flat_by_slot: list[np.ndarray] = []
        total = 0
        unanswered_rows = 0
        # ``request_ids`` was snapshotted by _process_batch before it read any
        # other field of these slots; see the ordering note on the header map.
        for slot, req_id in zip(ready, request_ids, strict=True):
            bsz = _slot_rows(slot, self._layout.max_batch)
            if compact_legal:
                meta = slot.policy_i32
                n_legal = max(0, int(meta[0]))
                # SNAPSHOT, never view. This is client-writable shared memory
                # and the client does rewrite it under us: after a request
                # timeout the worker resets its client and re-submits into the
                # same named slot while we are still mid-batch on the previous
                # request (worker.log: "inference broker timed out; resetting
                # client"). With views, the validation below checked one
                # request and the gather further down read another, so
                # counts.sum() no longer matched flat.size and the row index
                # came out longer than the column index -- an uncaught
                # ValueError in _stage_legal_metadata_cuda that killed this
                # process and left the whole fleet with no inference until the
                # next iteration boundary (2026-07-24, ~50min of zero games).
                # Validating a snapshot is the only way that check can mean
                # anything; a torn snapshot fails it and takes the fallback
                # path below. ~360KB per batch against a ~1.4s forward.
                counts = np.array(meta[1:1 + bsz], dtype=np.int32)
                flat = np.array(meta[1 + bsz:1 + bsz + n_legal], dtype=np.int32)
                # Named rather than one boolean chain so the rejection log can
                # say which invariant broke. A bare "malformed" line leaves the
                # reader unable to tell a torn snapshot (counts/header
                # disagreement) from a real client encoding bug (out-of-range
                # indices), which is the first thing worth knowing.
                counts_sum = int(counts.sum())
                if counts.shape[0] != bsz:
                    bad_reason = f"counts len {counts.shape[0]} != batch_size {bsz}"
                elif counts_sum != n_legal:
                    bad_reason = f"counts sum {counts_sum} != header n_legal {n_legal}"
                elif flat.shape[0] != n_legal:
                    bad_reason = f"flat len {flat.shape[0]} != header n_legal {n_legal}"
                elif flat.size > 0 and (
                    int(flat.min()) < 0 or int(flat.max()) >= _POLICY_SIZE
                ):
                    bad_reason = (
                        f"flat index range [{int(flat.min())}, {int(flat.max())}] "
                        f"outside [0, {_POLICY_SIZE})"
                    )
                else:
                    bad_reason = ""
                if bad_reason:
                    # Release the slot unanswered rather than fabricating a
                    # response. The old code here zeroed policy_u16[:1] and the
                    # WDL rows and marked the slot _STATE_RESPONSE, which the
                    # client cannot distinguish from a real answer -- it never
                    # reads request_mode back, and the success path sets
                    # _MODE_DENSE_F32 too. Worse than the all-zero policy that
                    # _release_slots_for_retry exists to prevent: the client
                    # reads slot.policy_u16[:n_legal] using its OWN n_legal, so
                    # only entry 0 was zeroed and entries 1..n_legal came back
                    # as whatever bf16 logits the previous request left in this
                    # shared buffer -- a plausible-looking policy belonging to a
                    # different position, paired with an all-zero WDL, fed
                    # straight into MCTS and recorded as training data.
                    #
                    # _STATE_IDLE instead makes the client re-submit (see
                    # _submit_and_wait_locked), which is exactly the recovery
                    # the dominant cause wants: the torn snapshot described
                    # above happens because the client rewrote the slot
                    # mid-read, so re-reading settled metadata succeeds. A
                    # client that is genuinely malformed retries every ~10ms
                    # and then raises its own TimeoutError -- loud, and scoped
                    # to the one broken client.
                    #
                    # Deliberately NOT counted toward
                    # _consecutive_batch_failures: that ceiling exits the
                    # broker, which would take inference away from the entire
                    # fleet to punish one client, and a broker restart cannot
                    # fix a client-side protocol bug, so it would just repeat.
                    self._malformed_legal_meta_total += 1
                    _now_mm = time.monotonic()
                    if _now_mm - self._malformed_legal_meta_warned_at >= 30.0:
                        self._malformed_legal_meta_warned_at = _now_mm
                        log.warning(
                            "broker rejected malformed compact-legal metadata "
                            "(%s); releasing the slot for retry rather than "
                            "answering with a stale policy (%d total)",
                            bad_reason, self._malformed_legal_meta_total,
                        )
                    # These rows were counted at dispatch by serve_forever and
                    # no model will evaluate them, so report them back (task
                    # #142). ``bsz`` rather than a re-read of slot.batch_size:
                    # the release below lets the client overwrite that field,
                    # and bsz is the value this batch actually worked from.
                    unanswered_rows += bsz
                    self._release_slots_for_retry([slot])
                    continue
                legal_counts_by_slot.append(counts)
                legal_flat_by_slot.append(flat)
            if use_bf16_input:
                self._pinned_input_bf16_bits_np[total:total + bsz] = slot.input_bf16_bits[:bsz]
            else:
                self._pinned_input_np[total:total + bsz] = slot.input[:bsz]
            active.append(slot)
            batch_sizes.append(bsz)
            active_request_ids.append(req_id)
            total += bsz

        assert total <= self._pinned_input.shape[0], (
            f"gather overflow: {total} > {self._pinned_input.shape[0]}"
        )
        if total <= 0:
            for slot, req_id in zip(active, active_request_ids, strict=True):
                slot.request_mode = _MODE_DENSE_F32
                slot.request_id = req_id
                slot.state = _STATE_RESPONSE
            return unanswered_rows
        forward_total = _compiled_padded_batch_size(total, capacity=self._pinned_input.shape[0])
        pin_input = self._pinned_input_bf16 if use_bf16_input else self._pinned_input
        compact_offsets: list[tuple[int, int]] = []
        legal_counts_all: np.ndarray | None = None
        legal_flat_all: np.ndarray | None = None
        legal_rows_all: np.ndarray | None = None
        if compact_legal:
            compact_row_parts: list[np.ndarray] = []
            row_base = 0
            pol_base = 0
            for bsz, counts, flat in zip(
                batch_sizes, legal_counts_by_slot, legal_flat_by_slot, strict=True,
            ):
                # Slot metadata validation above established that the counts
                # sum equals this flat array's length, and both are private
                # snapshots, so that guarantee still holds here.
                n_legal = int(flat.size)
                compact_offsets.append((pol_base, pol_base + n_legal))
                if n_legal > 0:
                    compact_row_parts.append(
                        np.repeat(
                            np.arange(row_base, row_base + bsz, dtype=np.int64),
                            counts.astype(np.int64, copy=False),
                        )
                    )
                row_base += bsz
                pol_base += n_legal
            if len(legal_counts_by_slot) == 1:
                legal_counts_all = legal_counts_by_slot[0].astype(np.int64, copy=False)
                legal_flat_all = legal_flat_by_slot[0].astype(np.int64, copy=False)
            else:
                legal_counts_all = (
                    np.concatenate(legal_counts_by_slot).astype(np.int64, copy=False)
                    if legal_counts_by_slot else np.empty((0,), dtype=np.int64)
                )
                legal_flat_all = (
                    np.concatenate(legal_flat_by_slot).astype(np.int64, copy=False)
                    if legal_flat_by_slot else np.empty((0,), dtype=np.int64)
                )
            if len(compact_row_parts) == 1:
                legal_rows_all = compact_row_parts[0]
            else:
                legal_rows_all = (
                    np.concatenate(compact_row_parts).astype(np.int64, copy=False)
                    if compact_row_parts else np.empty((0,), dtype=np.int64)
                )
        if _timing is not None:
            _timing["pack_s"] += time.perf_counter() - _t_pack0
        _t_forward0 = time.perf_counter()
        # Hang window covers H2D + forward + device sync — any of these can
        # block forever on a dead CUDA/WSL2 context without raising.
        # Keep the token. Without it mark_forward_done falls back to popping the
        # OLDEST forward, so a completing newer batch would delete a wedged older
        # one's start time and the abort log would name the wrong batch. Serial
        # today, but the fallback is the trap audit I3-H2 describes.
        hang_token = self._hang_watchdog.mark_forward_start(total)
        forward_ok = False
        try:
            xt = pin_input[:forward_total].to(self.device, non_blocking=True)

            first_inf = self._first_inference_pending
            inf_t0 = time.time() if first_inf else 0.0

            # AOT packages are dense-only (input -> full policy dict + wdl).
            # When the exact forward_total bucket is covered, force dense and
            # let the existing Python compact-gather path produce legal-mode
            # outputs. Uncovered buckets fall through to eager/legal paths.
            use_aot = should_use_aot_forward(self._aot_models, forward_total)
            use_legal_rows_forward = (
                (not use_aot)
                and compact_legal
                and _supports_legal_policy_rows_forward(self._model)
            )
            use_legal_forward = (
                (not use_aot)
                and compact_legal
                and not use_legal_rows_forward
                and _supports_legal_policy_forward(self._model)
            )
            if use_aot:
                assert self._aot_models is not None
                # AOT packages are statically bf16 (no autocast wrapper like the
                # eager _forward_no_grad path). F32-mode requests stage into the
                # f32 pinned buffer, so coerce here (no-op when already bf16).
                with torch.no_grad():
                    out = self._aot_models[forward_total](xt.to(torch.bfloat16))
            elif use_legal_rows_forward:
                assert legal_flat_all is not None
                assert legal_rows_all is not None
                legal_real = int(legal_flat_all.shape[0])
                legal_capacity = int(self._pinned_legal_pol_bf16_bits.shape[0])
                legal_forward_total = (
                    _compiled_padded_legal_policy_size(legal_real, capacity=legal_capacity)
                    if self.compile_inference else legal_real
                )
                if self.device.startswith("cuda"):
                    legal_flat_gpu, legal_rows_gpu = self._stage_legal_metadata_cuda(
                        legal_flat_all, legal_rows_all, transfer_size=legal_forward_total,
                    )
                else:
                    if legal_forward_total > legal_real:
                        legal_flat_padded = np.zeros((legal_forward_total,), dtype=np.int64)
                        legal_rows_padded = np.zeros((legal_forward_total,), dtype=np.int64)
                        legal_flat_padded[:legal_real] = legal_flat_all
                        legal_rows_padded[:legal_real] = legal_rows_all
                    else:
                        legal_flat_padded = legal_flat_all
                        legal_rows_padded = legal_rows_all
                    legal_flat_gpu = torch.as_tensor(
                        legal_flat_padded, dtype=torch.long, device=self.device,
                    )
                    legal_rows_gpu = torch.as_tensor(
                        legal_rows_padded, dtype=torch.long, device=self.device,
                    )
                out = _forward_legal_rows_no_grad(
                    self._model, xt, legal_flat_gpu, legal_rows_gpu, device=self.device,
                )
            elif use_legal_forward:
                assert legal_counts_all is not None
                assert legal_flat_all is not None
                legal_counts_gpu = torch.as_tensor(legal_counts_all, dtype=torch.long, device=self.device)
                legal_flat_gpu = torch.as_tensor(legal_flat_all, dtype=torch.long, device=self.device)
                out = _forward_legal_no_grad(
                    self._model, xt, legal_flat_gpu, legal_counts_gpu, device=self.device,
                )
            else:
                out = _forward_no_grad(self._model, xt, device=self.device)

            if first_inf:
                log.info("first inference (includes kernel compile) elapsed_s=%.2f batch=%d",
                         time.time() - inf_t0, xt.shape[0])
                print(
                    f"[broker] first_forward_done batch={int(xt.shape[0])} "
                    f"forward_s={time.time() - inf_t0:.2f} "
                    f"since_boot_s={self._since_boot_s():.2f}",
                    flush=True,
                )
                self._first_inference_pending = False

            if _timing is not None:
                _timing["forward_s"] += time.perf_counter() - _t_forward0
            _t_output0 = time.perf_counter()
            policy_gpu = (
                _policy_output(out).detach()
                if compact_legal and (use_legal_rows_forward or use_legal_forward)
                else _policy_output_full(out).detach()
            )
            wdl_gpu = out["wdl"].detach().float()
            self._pinned_wdl[:total].copy_(wdl_gpu[:total], non_blocking=True)
            compact_bits_np: np.ndarray | None = None
            if compact_legal:
                if use_legal_rows_forward or use_legal_forward:
                    assert legal_flat_all is not None
                    n_real_compact = int(legal_flat_all.shape[0])
                    compact_bits = policy_gpu[:n_real_compact].to(torch.bfloat16).view(torch.uint16)
                    n_compact = int(compact_bits.numel())
                    self._pinned_legal_pol_bf16_bits[:n_compact].copy_(compact_bits, non_blocking=True)
                    compact_bits_np = self._pinned_legal_pol_bf16_bits_np[:n_compact]
                elif legal_counts_by_slot:
                    assert legal_flat_all is not None
                    assert legal_rows_all is not None
                    if legal_flat_all.size > 0:
                        if self.device.startswith("cuda"):
                            cols, rows = self._stage_legal_metadata_cuda(
                                legal_flat_all,
                                legal_rows_all,
                                transfer_size=int(legal_flat_all.size),
                            )
                        else:
                            rows = torch.as_tensor(
                                legal_rows_all, dtype=torch.long, device=self.device,
                            )
                            cols = torch.as_tensor(
                                legal_flat_all, dtype=torch.long, device=self.device,
                            )
                        compact_bits = policy_gpu[:total][rows, cols].to(torch.bfloat16).view(torch.uint16)
                        n_compact = int(compact_bits.numel())
                        self._pinned_legal_pol_bf16_bits[:n_compact].copy_(compact_bits, non_blocking=True)
                        compact_bits_np = self._pinned_legal_pol_bf16_bits_np[:n_compact]
                    else:
                        compact_bits_np = np.empty((0,), dtype=np.uint16)
                else:
                    compact_bits_np = np.empty((0,), dtype=np.uint16)
            else:
                self._pinned_pol[:total].copy_(policy_gpu[:total].float(), non_blocking=True)
            if _timing is not None:
                _timing["output_s"] += time.perf_counter() - _t_output0
            _t_scatter0 = time.perf_counter()
            if self.device.startswith("cuda"):
                torch.cuda.current_stream(torch.device(self.device)).synchronize()

            # Scatter from pinned buffer to worker slots
            start = 0
            compact_idx = 0
            for slot, bsz, req_id in zip(active, batch_sizes, active_request_ids, strict=True):
                end = start + bsz
                if compact_legal:
                    assert compact_bits_np is not None
                    pol_start, pol_end = compact_offsets[compact_idx]
                    slot.policy_u16[:pol_end - pol_start] = compact_bits_np[pol_start:pol_end]
                    compact_idx += 1
                else:
                    slot.policy[:bsz] = self._pinned_pol_np[start:end]
                slot.wdl[:bsz] = self._pinned_wdl_np[start:end]
                slot.request_mode = _MODE_DENSE_F32
                # Echo the tag snapshotted at gather, NOT whatever the client
                # may have written since: that is what lets a client which
                # timed out and re-submitted reject this answer (audit I1).
                # Written before state so the client cannot observe RESPONSE
                # with the previous tag still in place.
                slot.request_id = req_id
                slot.state = _STATE_RESPONSE
                start = end
            if _timing is not None:
                _timing["scatter_s"] += time.perf_counter() - _t_scatter0
            forward_ok = True
        finally:
            self._hang_watchdog.mark_forward_done(success=forward_ok, token=hang_token)
        return unanswered_rows

  # -- main loop --

    def _wait_for_ready_slots(self) -> bool:
        """Poll for a ready slot; returns True as soon as one is in REQUEST.

        Hot while work is flowing (rung 0 of ``_IDLE_BACKOFF_LADDER`` is the
        original 200-poll spin + 20µs yield), backing off only once the whole
        fleet has been silent for a while. ``_idle_since`` is cleared by any
        arrival here and by ``serve_forever`` whenever it dispatches a batch, so
        a busy broker never leaves rung 0.
        """
        now = time.monotonic()
        idle_since = self._idle_since
        spin_polls, sleep_s = _idle_backoff(0.0 if idle_since is None else max(0.0, now - idle_since))
        for _ in range(spin_polls):
            if any(s.state == _STATE_REQUEST for s in self._slots):
                self._idle_since = None
                return True
        if idle_since is None:
            self._idle_since = now
        time.sleep(sleep_s)
        return False

    def _gather_more_within_window(self, ready: list) -> None:
        """Block up to ``batch_wait_ms`` for more slots to enter REQUEST state.

        Mutates ``ready`` in place. A RESPONSE slot is still eligible: its
        client can consume the result and submit a new request during this
        window, so only a full ready set or the timing deadline ends gather.

        With ``adaptive_idle_ms`` > 0 the fixed window becomes arrival-adaptive:
        each newly arrived REQUEST slot restarts an idle countdown, and the
        gather dispatches once no new slot has arrived for ``adaptive_idle_ms``.
        ``batch_wait_ms`` stays the hard cap on total wait, so a steady trickle
        of arrivals cannot hold the batch open indefinitely.
        """
        if self.batch_wait_ms <= 0:
            return
        adaptive = self.adaptive_idle_ms > 0
        now = time.monotonic()
        deadline = now + (self.batch_wait_ms / 1000.0)
        idle_deadline = now + (self.adaptive_idle_ms / 1000.0)
        while time.monotonic() < deadline:
            more = [s for s in self._slots if s.state == _STATE_REQUEST and s not in ready]
            if more:
                ready.extend(more)
                if _ARRIVAL_TRACE_ENABLED:
                    self._note_arrival_trace(more)
                idle_deadline = time.monotonic() + (self.adaptive_idle_ms / 1000.0)
            if len(ready) >= len(self._slots):
                return
            if adaptive and time.monotonic() >= idle_deadline:
                return
            time.sleep(0.0001)

    def _note_arrival_trace(self, slots: list[_InferenceSlot]) -> None:
        """Record first-seen REQUEST time for each slot (perf_counter)."""
        seen = self._arrival_trace_seen
        t = time.perf_counter()
        for slot in slots:
            key = id(slot)
            if key not in seen:
                seen[key] = t

    def _emit_arrival_trace(self, ready: list[_InferenceSlot]) -> None:
        """Buffer one JSONL dispatch record; pop used arrival times for reuse."""
        t_dispatch = time.perf_counter()
        seen = self._arrival_trace_seen
        arrivals: list[list[float | int]] = []
        for slot in ready:
            key = id(slot)
            t_arrival = seen.pop(key, t_dispatch)
            arrivals.append([t_arrival, int(slot.batch_size)])
        batch_rows = int(sum(int(a[1]) for a in arrivals))
        if arrivals:
            wait_ms = (t_dispatch - min(float(a[0]) for a in arrivals)) * 1000.0
        else:
            wait_ms = 0.0
        _record_arrival_trace(
            {
                "t_dispatch": t_dispatch,
                "arrivals": arrivals,
                "batch_rows": batch_rows,
                "wait_ms": wait_ms,
            }
        )

    def _maybe_print_broker_metrics(self, m: dict, now: float, report_interval: float) -> None:
        """Print broker throughput once per ``report_interval`` and reset counters."""
        if not (now - m["last_report"] >= report_interval and m["batches"] > 0):
            return
        avg_pos = m["positions"] / m["batches"]
        avg_slots = m["slots"] / m["batches"]
  # Capacity = num_slots × max_batch_per_slot. Fullness < ~50% = small batches
  # (dispatch-bound); > 80% = saturated.
        capacity_per_batch = len(self._slots) * self._layout.max_batch
        fullness = (100.0 * avg_pos / capacity_per_batch) if capacity_per_batch else 0.0
        print(
            f"[broker] {m['batches']} batches in {now - m['last_report']:.1f}s | "
            f"avg {avg_pos:.1f} pos/batch ({fullness:.0f}% of {capacity_per_batch}-cap), "
            f"{avg_slots:.1f}/{len(self._slots)} slots/batch | "
            f"{m['positions'] / (now - m['last_report']):.0f} pos/s | "
            f"pack={m['pack_s'] * 1000.0 / m['batches']:.2f}ms "
            f"fwd={m['forward_s'] * 1000.0 / m['batches']:.2f}ms "
            f"out={m['output_s'] * 1000.0 / m['batches']:.2f}ms "
            f"sync+scatter={m['scatter_s'] * 1000.0 / m['batches']:.2f}ms"
  # Only when non-zero, so the healthy line stays unchanged and any
  # non-zero value is a grep hit rather than something to eyeball.
            + (
                f" | malformed_legal_meta={self._malformed_legal_meta_total}"
                if self._malformed_legal_meta_total else ""
            ),
            flush=True,
        )
        m["batches"] = 0
        m["positions"] = 0
        m["slots"] = 0
        m["pack_s"] = 0.0
        m["forward_s"] = 0.0
        m["output_s"] = 0.0
        m["scatter_s"] = 0.0
        m["last_report"] = now

    def serve_forever(self) -> None:
        metrics = {
            "batches": 0,
            "positions": 0,
            "slots": 0,
            "pack_s": 0.0,
            "forward_s": 0.0,
            "output_s": 0.0,
            "scatter_s": 0.0,
            "last_report": time.monotonic(),
        }
        report_interval = 10.0  # seconds
        self._hang_watchdog.start()
        # Model may already be loaded (or loads on first request). "accepting"
        # means SHM slots exist and the poll loop is live — workers can connect.
        print(
            f"[broker] ACCEPTING_REQUESTS since_boot_s={self._since_boot_s():.2f} "
            f"model_loaded={self._model is not None}",
            flush=True,
        )

        while not self._stop:
            if any(s.state == _STATE_SHUTDOWN for s in self._slots):
                self._stop = True
                break

            ready = [s for s in self._slots if s.state == _STATE_REQUEST]
            if not ready:
                # Both branches continued anyway; the bool is kept for tests and
                # for _idle_backoff coverage, not as a control-flow signal.
                self._wait_for_ready_slots()
                continue
            self._idle_since = None

            if _ARRIVAL_TRACE_ENABLED:
                self._note_arrival_trace(ready)

            self._gather_more_within_window(ready)

  # Re-collect in case some slots changed during the wait window.
            ready = [s for s in self._slots if s.state == _STATE_REQUEST]
            if ready:
                if _ARRIVAL_TRACE_ENABLED:
                    self._note_arrival_trace(ready)
                metrics["batches"] += 1
  # Read ONCE. FIRST_BATCH_DONE below used to re-read slot.batch_size after
  # _process_batch, i.e. after any release, so its row count could describe
  # the client's NEXT request rather than this batch's.
                dispatched = sum(_slot_rows(s, self._layout.max_batch) for s in ready)
                metrics["positions"] += dispatched
                metrics["slots"] += len(ready)
                self._timing_metrics = metrics
                # Emit before _process_batch so wait_ms is gather wait only
                # (forward duration would otherwise skew the diagnostic).
                if _ARRIVAL_TRACE_ENABLED:
                    self._emit_arrival_trace(ready)
  # positions was added at DISPATCH above; _process_batch returns the rows
  # it handed back UNANSWERED (no model, batch failure, malformed legal
  # metadata), so subtracting them makes `pos/s` mean rows a model
  # evaluated. batches/slots stay dispatch counts deliberately: they answer
  # "how much did the loop try to batch", which a refusal does not change.
                unanswered = self._process_batch(ready)
                metrics["positions"] -= unanswered
  # The boot marker means "this broker has served a batch", and operators
  # read it as exactly that. It used to print the DISPATCHED row count
  # unconditionally and set the flag either way, so a broker whose first
  # batch was released unanswered -- the cold-boot no-model case, the one
  # this marker is watched for -- still announced FIRST_BATCH_DONE with a
  # full row count. Now it reports rows actually served, and a wholly
  # unanswered batch is not a first batch: the flag stays down and the
  # marker fires on the first batch that really is served. A modelless
  # broker never prints it, which is the truth and is already accompanied
  # by the throttled no-model WARNING.
                served = dispatched - unanswered
                if served > 0 and not self._first_batch_logged:
                    self._first_batch_logged = True
                    print(
                        f"[broker] FIRST_BATCH_DONE positions={served} "
                        f"slots={len(ready)} "
                        f"since_boot_s={self._since_boot_s():.2f}",
                        flush=True,
                    )

            self._maybe_print_broker_metrics(metrics, time.monotonic(), report_interval)

    def shutdown(self) -> None:
        self._stop = True
        self._hang_watchdog.stop()
        for slot in self._slots:
            slot.close()


# ---------------------------------------------------------------------------
# Client (used by worker processes)
# ---------------------------------------------------------------------------


class SlotInferenceClient:
    """Zero-allocation inference client backed by a pre-allocated shared memory slot.

    Implements the BatchEvaluator protocol.
    """

    def __init__(
        self,
        *,
        slot_name: str,
        max_batch: int,
        request_timeout_s: float = 30.0,
        input_planes: int = _CHANNELS,
    ) -> None:
        self._slot_name = str(slot_name)
        self._layout = _SlotLayout.compute(max_batch, int(input_planes))
        self._shm: SharedMemory | None = None
        self._slot: _InferenceSlot | None = None
        self._request_timeout_s = max(0.001, float(request_timeout_s))
        self._lock = threading.Lock()
        # Request tags must be unique across client INSTANCES on the same named
        # slot, not just within one instance: the failure this closes is a
        # worker that resets its client and builds a fresh one on the same slot
        # (worker.py `_reset_inference_client`). A per-instance counter starting
        # at 0 would hand the new client the same tag the timed-out one used and
        # reproduce the bug exactly. Random 32-bit seed, incremented per submit.
        self._next_request_id = random.getrandbits(32)
        # Observability for the thing this guard exists to catch. Without a
        # counter a fixed race is indistinguishable from a race that never
        # fired -- see the house rule about gates that cannot be observed.
        self.stale_responses_rejected = 0

    @property
    def input_planes(self) -> int:
        """Slot width in input planes — must match the broker's and the model's."""
        return int(self._layout.channels)

    def _disconnect(self) -> None:
        shm = self._shm
        self._slot = None
        self._shm = None
        if shm is not None:
            with contextlib.suppress(Exception):
                shm.close()

    def _alloc_request_id(self) -> int:
        rid = self._next_request_id & _REQUEST_ID_MASK
        if rid == _REQUEST_ID_NONE:
            rid = 1
        self._next_request_id = (rid + 1) & _REQUEST_ID_MASK
        return rid

    def _validate_segment(self, shm: SharedMemory) -> tuple[str, bool]:
        """Check the segment is this client's slot layout (audit I2).

        Returns ``("", False)`` when the segment is usable, else
        ``(reason, still_settling)``. The old code attached and built numpy
        views with no comparison at all: when the client's plane count was the
        SMALLER of the two (146 v1 vs the broker's 175 v2_threats -- and 146 is
        the argparse default on both sides) every view still fit inside the
        larger segment, so the protocol completed and the client read its policy
        and WDL out of the middle of the broker's input region: all-zero policy,
        all-zero WDL, no exception.

        ``still_settling`` marks a reason a client can lose to the broker's own
        slot creation rather than to a real mismatch: a creating process stamps
        the header only after the segment exists, so a client attaching inside
        that window legitimately sees an unstamped one -- the same class as "the
        slot does not exist yet", which ``_connect`` already retries. A layout-id
        mismatch is NOT in that class: magic is written last and is the publish
        barrier, so seeing valid magic means the layout id beside it is final.

        The size branch is marked settling too, but note it does NOT actually
        cover the ``ftruncate`` window: ``SharedMemory(name=..., create=False)``
        mmaps the fd and raises ``ValueError: cannot mmap an empty file`` before
        this function runs, and ``_connect`` only catches ``FileNotFoundError``
        (unchanged from before this guard existed). In practice the size branch
        fires only for a genuinely undersized segment -- an OLD broker serving a
        NEW client -- where retrying to the deadline is merely slow, not wrong.
        """
        want = self._layout.total_bytes
        if int(shm.size) < want:
            return (
                f"inference broker slot {self._slot_name!r} is {int(shm.size)} bytes "
                f"but this client's layout needs {want} "
                f"(max_batch={self._layout.max_batch}, planes={self._layout.channels})",
                True,
            )
        buf = shm.buf
        assert buf is not None  # attached SharedMemory always has a buffer
        magic = struct.unpack_from("<I", buf, _OFF_MAGIC)[0]
        if magic != _SLOT_MAGIC:
            return (
                f"inference broker slot {self._slot_name!r} has magic {magic:#010x}, "
                f"expected {_SLOT_MAGIC:#010x} (protocol v{SLOT_PROTOCOL_VERSION}); "
                "broker and workers must be launched from the same code generation",
                True,
            )
        layout_id = struct.unpack_from("<I", buf, _OFF_LAYOUT_ID)[0]
        want_id = self._layout.identity()
        if layout_id != want_id:
            return (
                f"inference broker slot {self._slot_name!r} was created for layout "
                f"{layout_id:#010x} but this client computes {want_id:#010x} "
                f"(max_batch={self._layout.max_batch}, planes={self._layout.channels}, "
                f"segment_bytes={int(shm.size)}, client_layout_bytes={want}); "
                "refusing to read policy/WDL out of the wrong offsets",
                False,
            )
        return "", False

    def _connect(self, *, deadline: float) -> _InferenceSlot:
        while True:
            slot = self._slot
            if slot is not None:
                return slot
            try:
                shm = SharedMemory(name=self._slot_name, create=False)
            except FileNotFoundError:
                self._disconnect()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"inference broker slot {self._slot_name!r} was not available "
                        f"after {self._request_timeout_s:.3f}s"
                    ) from None
                time.sleep(0.01)
                continue
            _detach_attached_shm_from_resource_tracker(shm)
            reason, still_settling = self._validate_segment(shm)
            if reason:
                with contextlib.suppress(Exception):
                    shm.close()
                if still_settling and time.monotonic() < deadline:
                    # Mid-creation, not a mismatch — retry like a missing slot.
                    time.sleep(0.01)
                    continue
                raise SlotProtocolMismatch(reason)
            self._shm = shm
            self._slot = _InferenceSlot(shm, self._layout, owns=False)
            return self._slot

    @property
    def supports_input_bf16_bits(self) -> bool:
        return True

    @property
    def supports_compact_root_policy(self) -> bool:
        """Roots may reuse the compact legal-policy broker protocol."""
        return True

    @property
    def pads_batches_internally(self) -> bool:
        """Callers must submit real rows only — the broker pads the total.

        ``SlotBroker`` coalesces every requesting slot and rounds the *summed*
        row count up to ``_COMPILED_BATCH_BUCKETS`` before the forward, so a
        client that pre-pads to its own bucket ladder gains no shape stability
        (it holds no compiled graph) and pays for the padding rows twice: once
        in the shared-memory write, once in the broker's H2D + forward.
        """
        return True

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if relations is not None:
            raise NotImplementedError(
                "dynamic relations are not transported on the SlotInferenceClient path; "
                "use worker-local direct inference (see "
                "check_dynamic_relations_transport)"
            )
        with self._lock:
            return self._evaluate_encoded_locked(x)

    def _evaluate_encoded_locked(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # skylos: ignore — called from evaluate_encoded above
        if x.dtype == np.uint16:
            xb = _coerce_bf16_bits_batch(x)
            request_mode = _MODE_DENSE_BF16
        else:
            xb = _coerce_input_batch(x)
            request_mode = _MODE_DENSE_F32
        bsz = xb.shape[0]
        if bsz > self._layout.max_batch:
            raise ValueError(
                f"batch size {bsz} exceeds slot max {self._layout.max_batch}"
            )

        def _submit(slot: _InferenceSlot, request_id: int) -> None:
  # Write input directly into shared memory (one memcpy).
            if request_mode == _MODE_DENSE_BF16:
                slot.input_bf16_bits[:bsz] = xb
            else:
                slot.input[:bsz] = xb
            slot.request_mode = request_mode
            slot.batch_size = bsz
            # Tag AFTER the payload, state last. See the ordering note on the
            # header map: the broker reads the tag first, so publishing the tag
            # before the payload would let it echo a tag for data it has not
            # written yet.
            slot.request_id = request_id
            slot.state = _STATE_REQUEST

        def _read(slot: _InferenceSlot) -> tuple[np.ndarray, np.ndarray]:
  # slot.policy / slot.wdl are C-contiguous numpy views over the
  # shared-memory buffer (constructed that way in _InferenceSlot);
  # .copy() is enough — np.array(..., copy=True, order="C") was an
  # extra contiguity check we don't need.
            return slot.policy[:bsz].copy(), slot.wdl[:bsz].copy()

        return self._submit_and_wait_locked(_submit, _read)

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            xb = _coerce_bf16_bits_batch(x)
            bsz = xb.shape[0]
            if bsz > self._layout.max_batch:
                raise ValueError(
                    f"batch size {bsz} exceeds slot max {self._layout.max_batch}"
                )
            counts = np.asarray(legal_counts, dtype=np.int32)
            if counts.ndim != 1 or counts.shape[0] != bsz:
                raise ValueError(f"legal_counts must be shape ({bsz},), got {counts.shape}")
            flat = np.asarray(legal_flat, dtype=np.int32)
            if flat.ndim != 1:
                raise ValueError(f"legal_flat must be 1D, got {flat.ndim}D")
            n_legal = int(counts.sum())
            if n_legal != int(flat.shape[0]):
                raise ValueError(f"legal_flat len {flat.shape[0]} != sum(legal_counts) {n_legal}")
            if flat.size > 0 and (int(flat.min()) < 0 or int(flat.max()) >= _POLICY_SIZE):
                raise ValueError("legal_flat contains out-of-range policy indices")
            meta_len = 1 + bsz + n_legal
            if meta_len > self._layout.policy_bytes // 4:
                raise ValueError(
                    f"compact legal metadata has {meta_len} int32 entries > "
                    f"{self._layout.policy_bytes // 4} capacity"
                )
            if n_legal > self._layout.policy_bytes // 2:
                raise ValueError(
                    f"compact legal policy has {n_legal} entries > "
                    f"{self._layout.policy_bytes // 2} capacity"
                )

            def _submit(slot: _InferenceSlot, request_id: int) -> None:
                slot.input_bf16_bits[:bsz] = xb
                slot.policy_i32[0] = n_legal
                slot.policy_i32[1:1 + bsz] = counts
                slot.policy_i32[1 + bsz:1 + bsz + n_legal] = flat
                slot.request_mode = _MODE_LEGAL_BF16
                slot.batch_size = bsz
                # Tag last-but-one, state last -- see _evaluate_encoded_locked.
                slot.request_id = request_id
                slot.state = _STATE_REQUEST

            def _read(slot: _InferenceSlot) -> tuple[np.ndarray, np.ndarray]:
                return slot.policy_u16[:n_legal].copy(), slot.wdl[:bsz].copy()

            return self._submit_and_wait_locked(_submit, _read)

    def _submit_and_wait_locked(
        self,
        submit: Any,
        read: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        deadline = time.monotonic() + self._request_timeout_s
        last_timeout = False
        last_stale = False
        while True:
            slot = self._connect(deadline=deadline)

            request_id = self._alloc_request_id()
            submit(slot, request_id)

  # Wait for response. Keep the fast spin path for short broker latency,
  # but recover if the broker went away and the slot had to be recreated.
            spins = 0
            retry = False
            last_stale = False
            while True:
  # slot.state is shared-memory; at runtime the broker writes
  # other state values concurrently, so we wrap in int() to keep
  # pyright from narrowing to the last literal we stored.
                state = int(slot.state)
                if state == _STATE_RESPONSE:
                    echoed = int(slot.request_id)
                    if echoed != request_id:
                        # The broker answered an EARLIER request into this slot
                        # -- the audit-I1 race. Reading it would hand MCTS (and
                        # the training shards) a policy/WDL for a different
                        # position. Drop it and re-submit; the payload we wrote
                        # has been overwritten by the answer, so the slot must
                        # be re-armed from scratch rather than re-waited.
                        self.stale_responses_rejected += 1
                        last_stale = True
                        log.warning(
                            "inference slot %s returned a stale response "
                            "(echoed request_id=%d, expected %d); discarding and "
                            "re-submitting (%d rejected on this client)",
                            self._slot_name, echoed, request_id,
                            self.stale_responses_rejected,
                        )
                        slot.state = _STATE_IDLE
                        retry = True
                        break
                    pol, wdl = read(slot)
                    slot.state = _STATE_IDLE
                    return pol, wdl
                if state in (_STATE_SHUTDOWN, _STATE_IDLE):
                    retry = True
                    break
                if state != _STATE_REQUEST:
                    retry = True
                    break
                if time.monotonic() >= deadline:
                    last_timeout = True
                    retry = True
                    break
                spins += 1
                if spins >= 1000:
                    time.sleep(0.0001)
                    spins = 0

            self._disconnect()
            if time.monotonic() >= deadline:
                if last_timeout:
                    raise TimeoutError(
                        f"inference broker timed out after {self._request_timeout_s:.3f}s"
                    )
                if last_stale:
                    # Distinct from the shutdown message below: recovery is the
                    # same, but this is the ONE log line describing the race
                    # this identity check exists for, and calling it a shutdown
                    # would send the next reader looking at the wrong thing.
                    raise TimeoutError(
                        f"inference broker returned a stale response and the retry "
                        f"did not fit in {self._request_timeout_s:.3f}s "
                        f"({self.stale_responses_rejected} rejected on this client)"
                    )
                raise RuntimeError("inference broker shut down while request was in flight")
            if retry:
                time.sleep(0.01)

    def close(self) -> None:
        self._disconnect()


class MultiSlotInferenceClient:
    """Thread-safe fan-out across multiple broker slots.

    Each underlying slot still serializes its own request/response protocol,
    but different selfplay threads can make progress through different slots.

    A slot that keeps failing is QUARANTINED rather than handed straight back
    out: production runs 4 slots at a 30 s request timeout, so one wedged slot
    used to cost every fourth request a full 30 s stall, forever, with nothing
    in the stats able to tell it from a healthy one (SHARED_BROKER_AUDIT B6).
    Quarantine is temporary and re-probed on an exponential backoff — a
    transient broker stall must not permanently shrink the worker's capacity.
    """

    def __init__(
        self,
        *,
        slot_names: list[str],
        max_batch: int,
        request_timeout_s: float = 30.0,
        input_planes: int = _CHANNELS,
        slot_failure_threshold: int = 2,
        slot_quarantine_s: float = 5.0,
        slot_quarantine_max_s: float = 60.0,
    ) -> None:
        names = [str(n).strip() for n in slot_names if str(n).strip()]
        if not names:
            raise ValueError("MultiSlotInferenceClient requires at least one slot")
        self._request_timeout_s = max(0.001, float(request_timeout_s))
        self._slot_failure_threshold = max(1, int(slot_failure_threshold))
        self._slot_quarantine_s = max(0.0, float(slot_quarantine_s))
        self._slot_quarantine_max_s = max(
            self._slot_quarantine_s, float(slot_quarantine_max_s),
        )
        self._clients = [
            SlotInferenceClient(
                slot_name=name,
                max_batch=max_batch,
                request_timeout_s=self._request_timeout_s,
                input_planes=input_planes,
            )
            for name in names
        ]
        self._input_planes = int(input_planes)
        self._available_clients: queue.Queue[tuple[int, SlotInferenceClient]] = queue.Queue()
        for idx, client in enumerate(self._clients):
            self._available_clients.put((idx, client))
        self._stats_lock = threading.Lock()
        self._lifetime_requests = 0
        self._lifetime_positions = 0
        self._lifetime_legal_requests = 0
        self._lifetime_legal_positions = 0
        self._lifetime_wait_s = 0.0
        self._lifetime_roundtrip_s = 0.0
        self._inflight = 0
        self._max_inflight = 0
        self._lifetime_failed_requests = 0
        self._lifetime_failed_positions = 0
        # slot_requests counts ATTEMPTS (unchanged meaning); slot_served and
        # slot_failures split it, because attempts alone cannot distinguish a
        # dead slot from a healthy one -- the audit measured [3, 3, 3, 3] with
        # slot 3 serving nothing.
        self._slot_requests = [0 for _ in self._clients]
        self._slot_served = [0 for _ in self._clients]
        self._slot_failures = [0 for _ in self._clients]
        self._slot_consecutive_failures = [0 for _ in self._clients]
        self._slot_quarantines = [0 for _ in self._clients]
        self._slot_positions = [0 for _ in self._clients]
        self._slot_wait_s = [0.0 for _ in self._clients]
        self._slot_roundtrip_s = [0.0 for _ in self._clients]
        # idx -> (client, monotonic re-probe deadline). Held here instead of on
        # _available_clients, so a wedged slot stops being handed out; the
        # deadline is what puts it back.
        self._quarantined: dict[int, tuple[SlotInferenceClient, float]] = {}
        self._slot_quarantine_backoff_s = [
            self._slot_quarantine_s for _ in self._clients
        ]

    @property
    def input_planes(self) -> int:
        """Slot width in input planes — must match the broker's and the model's."""
        return self._input_planes

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if relations is not None:
            raise NotImplementedError(
                "dynamic relations are not transported on the MultiSlotInferenceClient path; "
                "use worker-local direct inference (see "
                "check_dynamic_relations_transport)"
            )
        idx, client, wait_s = self._acquire_client()
        t0 = time.perf_counter()
        # Set BEFORE the call and cleared only on a normal return, so every
        # abnormal exit (exception, GeneratorExit) is accounted as a failure
        # rather than as served throughput.
        failed = True
        try:
            out = client.evaluate_encoded(x)
            failed = False
            return out
        finally:
            self._release_client(
                idx, client,
                positions=int(x.shape[0]),
                legal=False,
                wait_s=wait_s,
                roundtrip_s=time.perf_counter() - t0,
                failed=failed,
            )

    @property
    def supports_input_bf16_bits(self) -> bool:
        return True

    @property
    def supports_compact_root_policy(self) -> bool:
        return True

    @property
    def pads_batches_internally(self) -> bool:
        """See ``SlotInferenceClient.pads_batches_internally``."""
        return True

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx, client, wait_s = self._acquire_client()
        t0 = time.perf_counter()
        failed = True
        try:
            out = client.evaluate_legal_bf16(x, legal_flat, legal_counts)
            failed = False
            return out
        finally:
            self._release_client(
                idx, client,
                positions=int(x.shape[0]),
                legal=True,
                wait_s=wait_s,
                roundtrip_s=time.perf_counter() - t0,
                failed=failed,
            )

    def _acquire_client(self) -> tuple[int, SlotInferenceClient, float]:
        """Take a free slot, or fail if none free within the request timeout.

        Unbounded ``queue.get()`` is a process-level hang risk: if every slot
        is held by a stuck request (or a leak), all selfplay threads park here
        forever and the per-request broker timeout never runs. Cap wait at the
        same budget as a single request so the worker can raise, reset, or
        exit via the session liveness watchdog.

        Quarantined slots are re-probed here: the wait is capped at the next
        re-probe deadline so a pool that is entirely quarantined recovers as
        soon as the first backoff expires instead of burning the whole acquire
        budget waiting on a queue nothing will fill.
        """
        t0 = time.perf_counter()
        # Allow a little more than one request so a just-busy slot can free.
        acquire_timeout_s = max(0.05, float(self._request_timeout_s) * 2.0)
        deadline = t0 + acquire_timeout_s
        while True:
            self._reinstate_due_slots()
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                break
            probe_in = self._next_probe_wait_s()
            wait_for = remaining if probe_in is None else min(remaining, max(0.01, probe_in))
            try:
                idx, client = self._available_clients.get(timeout=wait_for)
            except queue.Empty:
                continue
            wait_s = time.perf_counter() - t0
            with self._stats_lock:
                self._inflight += 1
                self._max_inflight = max(self._max_inflight, self._inflight)
            return idx, client, wait_s

        with self._stats_lock:
            inflight = int(self._inflight)
            available = int(self._available_clients.qsize())
            quarantined = sorted(self._quarantined)
        # The periodic `broker client stats:` line CANNOT show this window: no
        # request completes, so its own delta gate returns before printing and
        # slots_quarantined never reaches an operator through that channel. The
        # count therefore has to leave here, on the one event that does happen.
        log.warning(
            "inference slot acquire failed: %d/%d slot(s) quarantined %s, "
            "%d inflight, %d available -- the periodic stats line cannot show "
            "this window because no request completed in it (audit B6)",
            len(quarantined), len(self._clients), quarantined, inflight, available,
        )
        raise TimeoutError(
            f"inference slot acquire timed out after {acquire_timeout_s:.1f}s "
            f"(slots={len(self._clients)} inflight={inflight} available={available} "
            f"quarantined={quarantined}); "
            "all slots held or quarantined — broker stall or slot leak"
        )

    def _next_probe_wait_s(self) -> float | None:
        """Seconds until the earliest quarantined slot is due, or None if none are."""
        with self._stats_lock:
            if not self._quarantined:
                return None
            soonest = min(until for _, until in self._quarantined.values())
        return max(0.0, soonest - time.monotonic())

    def _reinstate_due_slots(self) -> None:
        """Return quarantined slots to the pool once their backoff has expired.

        Quarantine must be a probe, not an execution: a broker restart or a
        transient stall would otherwise shrink the worker to the slots that
        happened to be idle at the time, permanently, and nothing in the loop
        ever re-widens it.
        """
        now = time.monotonic()
        with self._stats_lock:
            due = [i for i, (_, until) in self._quarantined.items() if until <= now]
            revived = [(i, self._quarantined.pop(i)[0]) for i in due]
        for idx, client in revived:
            self._available_clients.put((idx, client))

    def _release_client(
        self,
        idx: int,
        client: SlotInferenceClient,
        *,
        positions: int,
        legal: bool,
        wait_s: float,
        roundtrip_s: float,
        failed: bool = False,
    ) -> None:
        """Account the request and hand the slot back — unless it must be quarantined.

        A failed request evaluated no positions, so it must not land in the
        lifetime totals the worker reports as throughput: during a broker stall
        the 60-second `broker client stats:` line used to report full pos_s
        while nothing at all was being served (SHARED_BROKER_AUDIT B6).
        """
        backoff = 0.0
        quarantined_now = 0
        total = len(self._clients)
        with self._stats_lock:
            self._lifetime_requests += 1
            self._slot_requests[idx] += 1
            self._lifetime_wait_s += float(wait_s)
            self._lifetime_roundtrip_s += float(roundtrip_s)
            self._slot_wait_s[idx] += float(wait_s)
            self._slot_roundtrip_s[idx] += float(roundtrip_s)
            self._inflight = max(0, self._inflight - 1)
            quarantine = False
            if failed:
                self._lifetime_failed_requests += 1
                self._lifetime_failed_positions += int(positions)
                self._slot_failures[idx] += 1
                self._slot_consecutive_failures[idx] += 1
                if self._slot_consecutive_failures[idx] >= self._slot_failure_threshold:
                    quarantine = self._slot_quarantine_s > 0.0
            else:
                self._lifetime_positions += int(positions)
                if legal:
                    self._lifetime_legal_requests += 1
                    self._lifetime_legal_positions += int(positions)
                self._slot_served[idx] += 1
                self._slot_positions[idx] += int(positions)
                # A success clears the strike count AND the backoff: the next
                # wedge on this slot starts from the short quarantine again.
                self._slot_consecutive_failures[idx] = 0
                self._slot_quarantine_backoff_s[idx] = self._slot_quarantine_s
            if quarantine:
                backoff = min(
                    self._slot_quarantine_backoff_s[idx], self._slot_quarantine_max_s,
                )
                self._quarantined[idx] = (client, time.monotonic() + backoff)
                self._slot_quarantines[idx] += 1
                self._slot_consecutive_failures[idx] = 0
                self._slot_quarantine_backoff_s[idx] = min(
                    max(backoff, self._slot_quarantine_s) * 2.0,
                    self._slot_quarantine_max_s,
                )
                quarantined_now = len(self._quarantined)
        if quarantine:
            log.warning(
                "inference slot %d quarantined for %.1fs after %d consecutive "
                "failures (%d/%d slots now quarantined); it will be re-probed, "
                "not dropped",
                idx, backoff, self._slot_failure_threshold, quarantined_now, total,
            )
            return
        self._available_clients.put((idx, client))

    @property
    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            requests = int(self._lifetime_requests)
            positions = int(self._lifetime_positions)
            served = requests - int(self._lifetime_failed_requests)
            return {
                "slots": len(self._clients),
                "available_slots": int(self._available_clients.qsize()),
                "inflight": int(self._inflight),
                "max_inflight": int(self._max_inflight),
                # requests = ATTEMPTS; served/failed split it. positions counts
                # only rows a model actually evaluated -- a timed-out request
                # transported nothing and must never read as throughput (B6).
                "lifetime_requests": requests,
                "lifetime_served_requests": served,
                "lifetime_failed_requests": int(self._lifetime_failed_requests),
                "lifetime_failed_positions": int(self._lifetime_failed_positions),
                "lifetime_positions": positions,
                "lifetime_legal_requests": int(self._lifetime_legal_requests),
                "lifetime_legal_positions": int(self._lifetime_legal_positions),
                "lifetime_wait_s": float(self._lifetime_wait_s),
                "lifetime_roundtrip_s": float(self._lifetime_roundtrip_s),
                "avg_rows_per_request": positions / served if served else 0.0,
                "avg_wait_ms": 1000.0 * self._lifetime_wait_s / requests if requests else 0.0,
                "avg_roundtrip_ms": 1000.0 * self._lifetime_roundtrip_s / requests if requests else 0.0,
                "slot_requests": list(self._slot_requests),
                "slot_served": list(self._slot_served),
                "slot_failures": list(self._slot_failures),
                "slot_quarantines": list(self._slot_quarantines),
                "slots_quarantined": len(self._quarantined),
                "slot_positions": list(self._slot_positions),
                "slot_wait_s": list(self._slot_wait_s),
                "slot_roundtrip_s": list(self._slot_roundtrip_s),
                # Audit I1. A guard nobody can observe is indistinguishable from
                # a guard that never fires, and this one closes a silent
                # data-integrity hole -- so the count is reported, not just
                # logged. Non-zero means the timeout/re-submit race DID occur
                # and was caught; it should be zero on a healthy run.
                "stale_responses_rejected": sum(
                    int(c.stale_responses_rejected) for c in self._clients
                ),
            }

    def close(self) -> None:
        for client in self._clients:
            client.close()




# ---------------------------------------------------------------------------
# Shared broker: one process serves all trials, swaps weights per-trial
# ---------------------------------------------------------------------------


class SharedSlotBroker:
    """Multi-trial inference broker sharing one compiled model.

    Instead of N separate broker processes (each ~15GB for CUDA + torch.compile),
    one shared broker holds the compiled model once and swaps weights when
    serving different trials.  Saves ~(N-1)*14GB RAM.

    Watches a server_root/trials/ directory for trial publish dirs.
    Each trial gets `slots_per_trial` shared memory slots.
    """

    def __init__(
        self,
        *,
        server_root: Path,
        slots_per_trial: int,
        max_batch_per_slot: int,
        device: str,
        compile_inference: bool,
        batch_wait_ms: float,
        compile_mode: str = "reduce-overhead",
        input_planes: int = _CHANNELS,
        hang_abort_seconds: float = _DEFAULT_HANG_ABORT_S,
    ) -> None:
        self.server_root = Path(server_root)
        self.slots_per_trial = int(slots_per_trial)
        self.device = str(device)
        self.compile_inference = bool(compile_inference)
        self.compile_mode = str(compile_mode or "reduce-overhead")
        self.batch_wait_ms = float(batch_wait_ms)
        self._hang_watchdog = BrokerHangWatchdog(
            threshold_s=float(hang_abort_seconds),
            boot_threshold_s=resolve_boot_hang_abort_seconds(float(hang_abort_seconds)),
        )
        self._stop = False

        self._layout = _SlotLayout.compute(max_batch_per_slot, int(input_planes))

  # Per-trial model instances on separate CUDA streams for parallel execution.
  # All share one CUDA context + compiled kernel cache (~13GB total)
  # instead of N separate processes (~15GB each).
        self._model_config_key: tuple | None = None
        self._first_inference_done = False

  # Per-trial state
        self._trial_slots: dict[str, list[_InferenceSlot]] = {}
        self._trial_shas: dict[str, str] = {}
        self._trial_models: dict[str, torch.nn.Module] = {}  # per-trial model on GPU
        self._trial_streams: dict[str, torch.cuda.Stream] = {}  # per-trial CUDA stream
        self._trial_manifest_sigs: dict[str, tuple[int, int]] = {}
      # Last time we warned that a trial has no model, per trial. Slots
      # released for retry come straight back round the serve loop, so an
      # unthrottled warning here would write thousands of lines a second.
        self._trial_no_model_warned_at: dict[str, float] = {}
      # Same, for a request whose transport this broker has no dispatch for.
        self._trial_bad_mode_warned_at: dict[str, float] = {}
      # Lifetime count of requests refused for an unimplemented request_mode.
      # A refusal nobody can count is indistinguishable from one that never
      # fires, and this one is the difference between a loud timeout and a
      # mis-decoded policy reaching MCTS (SHARED_BROKER_AUDIT B1).
        self.unsupported_mode_requests = 0
        self._all_slots: list[tuple[str, _InferenceSlot]] = []
        # Monotonic start of the current no-work stretch (None = work flowing).
        self._idle_since: float | None = None

    def _register_new_trial(self, trial_id: str) -> None:
        """Allocate ``slots_per_trial`` shared-memory slots for a freshly seen trial."""
        # Version-carrying name, derived by the SAME function the launcher uses
        # to tell workers what to attach to. Never re-derive it here.
        slot_prefix = trial_slot_prefix(trial_id=trial_id)

        slots: list[_InferenceSlot] = []
        for i in range(self.slots_per_trial):
            name = f"{slot_prefix}-{i}"
            try:
                old = SharedMemory(name=name, create=False)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            shm = SharedMemory(name=name, create=True, size=self._layout.total_bytes)
            slot = _InferenceSlot(shm, self._layout, owns=True)
            slot.state = _STATE_IDLE
            slot.batch_size = 0
            slot.write_protocol_header()
            slots.append(slot)

        self._trial_slots[trial_id] = slots
        self._all_slots.extend((trial_id, slot) for slot in slots)
        log.info("shared broker: registered trial %s with %d slots (prefix=%s)",
                 trial_id, len(slots), slot_prefix)

    def _deregister_trial(self, trial_id: str) -> None:
        """Tear down slot/model bookkeeping for a trial that's gone away."""
        for slot in self._trial_slots[trial_id]:
            slot.close()
        self._all_slots = [(t, s) for t, s in self._all_slots if t != trial_id]
        del self._trial_slots[trial_id]
        self._trial_models.pop(trial_id, None)
        self._trial_streams.pop(trial_id, None)
        self._trial_shas.pop(trial_id, None)
        self._trial_manifest_sigs.pop(trial_id, None)
      # Also the warn throttle, or a trial id re-registered within 30s would
      # have its first no-model gap silently un-warned by the previous
      # incarnation's timestamp (and dead ids would accumulate forever).
        self._trial_no_model_warned_at.pop(trial_id, None)
        print(f"[shared-broker] deregistered stale trial {trial_id}", flush=True)

    def _scan_trials(self) -> None:
        """Discover new trials and create slots for them."""
        trials_root = self.server_root / "trials"
        if not trials_root.exists():
            return
        active_prefix_path = self.server_root / "active_run_prefix.txt"
        try:
            active_prefix = active_prefix_path.read_text(encoding="utf-8").strip()
        except Exception:
            active_prefix = ""

        for trial_dir in sorted(trials_root.iterdir()):
            if not trial_dir.is_dir():
                continue
            trial_id = trial_dir.name
            if active_prefix and not trial_id.startswith(active_prefix):
                continue
            if trial_id in self._trial_slots:
                continue
            if not (trial_dir / "publish" / "manifest.json").exists():
                continue
            self._register_new_trial(trial_id)

        stale = [
            tid for tid in self._trial_slots
            if not (trials_root / tid / "publish" / "manifest.json").exists()
            or (active_prefix and not tid.startswith(active_prefix))
        ]
        for tid in stale:
            self._deregister_trial(tid)

    def _build_trial_model(
        self, trial_id: str, *, model_cfg: ModelConfig, sd: dict, model_sha: str,
    ) -> None:
        """Construct + register a fresh model instance + CUDA stream for a new trial."""
        model = build_model(model_cfg)
        require_model_planes(
            model, self._layout.channels,
            where=f"SharedSlotBroker slot layout (trial {trial_id})",
        )
        model.to(self.device)
        model.eval()
        if hasattr(model, "_inference_only"):
            setattr(model, "_inference_only", True)
        load_state_dict_tolerant(model, sd, label=f"shared-broker-{trial_id}")
        if self.compile_inference and self.device.startswith("cuda"):
            model = cast("torch.nn.Module", torch.compile(model, mode=self.compile_mode))
        self._trial_models[trial_id] = model
        if self.device.startswith("cuda"):
            self._trial_streams[trial_id] = torch.cuda.Stream(device=self.device)
        print(
            f"[shared-broker] created model for trial {trial_id} (sha={model_sha[:8]})",
            flush=True,
        )

    def _update_trial_weights(self, trial_id: str, *, sd: dict, model_sha: str) -> None:
        """Hot-swap weights into the existing model instance for ``trial_id``."""
        model = self._trial_models[trial_id]
        target = getattr(model, "_orig_mod", model)
        load_state_dict_tolerant(target, sd, label=f"shared-broker-{trial_id}")
        print(
            f"[shared-broker] updated weights for trial {trial_id} (sha={model_sha[:8]})",
            flush=True,
        )

    def _load_trial_weights(self, trial_id: str) -> bool:
        """Load/refresh model for a trial. Each trial gets its own model instance + CUDA stream."""
        publish_dir = self.server_root / "trials" / trial_id / "publish"
        manifest_path = publish_dir / "manifest.json"
        try:
            stat = manifest_path.stat()
            sig = (int(stat.st_mtime_ns), int(stat.st_size))
        except FileNotFoundError:
            return False

        if self._trial_manifest_sigs.get(trial_id) == sig:
            return trial_id in self._trial_models

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False  # manifest missing or mid-write; try again next poll

        model_info = manifest.get("model") or {}
        model_sha = str(model_info.get("sha256") or "")
        if not model_sha or model_sha == self._trial_shas.get(trial_id):
            self._trial_manifest_sigs[trial_id] = sig
            return trial_id in self._trial_models

        mc = manifest.get("model_config") or {}
        model_cfg = dataclass_replace(
            model_config_from_manifest_dict(mc),
            use_gradient_checkpointing=False,
        )
        config_key = (
            model_cfg.kind,
            model_cfg.embed_dim,
            model_cfg.num_layers,
            model_cfg.num_heads,
            model_cfg.embed_dim_by_layer,
            model_cfg.ffn_mult,
            model_cfg.ffn_mult_by_layer,
            model_cfg.use_smolgen,
            model_cfg.use_nla,
            model_cfg.use_qk_rmsnorm,
            model_cfg.input_pos_encoding,
            model_cfg.use_deepnorm,
            model_cfg.policy_encoding,
            model_cfg.input_history_encoding,
            model_cfg.input_global_embedding,
            model_cfg.input_global_embedding_channels,
            model_cfg.input_square_embedding,
            model_cfg.qkv_projection,
            model_cfg.smolgen_mode,
            model_cfg.smolgen_pooling,
            model_cfg.smolgen_hidden_channels,
            model_cfg.smolgen_hidden_sz,
            model_cfg.smolgen_gen_sz,
            model_cfg.smolgen_bias_scale,
            model_cfg.smolgen_bias_norm,
            model_cfg.arc_attention_bias,
            model_cfg.smolgen_relation_basis,
            model_cfg.smolgen_relation_norm,
            model_cfg.smolgen_relation_coeff_norm,
            model_cfg.smolgen_relation_scale,
            model_cfg.phase_output_adapter,
            model_cfg.phase_output_adapter_dim,
            model_cfg.phase_smolgen,
            model_cfg.phase_piece_thresholds,
            model_cfg.categorical_head_coupled,
        )
        if self._model_config_key is not None and config_key != self._model_config_key:
            print(
                f"[shared-broker] WARNING: trial {trial_id} has different model config, skipping",
                flush=True,
            )
            return False
        if self._model_config_key is None:
            self._model_config_key = config_key

        try:
            ckpt = torch.load(str(publish_dir / "latest_model.pt"), map_location="cpu")
            sd = ckpt.get("model", ckpt)
        except (OSError, RuntimeError, EOFError):
            return False  # checkpoint missing, mid-write, or torch load failed

        if trial_id not in self._trial_models:
            self._build_trial_model(
                trial_id, model_cfg=model_cfg, sd=sd, model_sha=model_sha,
            )
        else:
            self._update_trial_weights(trial_id, sd=sd, model_sha=model_sha)

        self._trial_shas[trial_id] = model_sha
        self._trial_manifest_sigs[trial_id] = sig
        return True

    def _process_parallel(self, ready_by_trial: dict[str, list[_InferenceSlot]]) -> int:
        """Process all trials' batches in parallel using per-trial CUDA streams.

        Returns the number of ROWS released UNANSWERED, on either path that
        can release one: a trial with no model yet, and a request_mode this
        broker does not implement. The serve loop counts positions BEFORE
        calling this, so without the return its pos/s would count rows this
        broker declined to evaluate -- the same "a failure reads as throughput"
        defect B6 fixed on the client side, in the sibling path.

        The name is ``unanswered_rows`` rather than the original
        ``refused_rows`` because it stopped meaning "refused for an
        unimplemented mode" when the no-model path was added to it (task #142
        re-review): a counter whose name is narrower than its contents is how
        the missing no-model case survived the first enumeration.
        """
        use_cuda = self.device.startswith("cuda")
        unanswered_rows = 0

        # Host-side pack first; H2D + forward sit under the hang watchdog.
        packed: list[tuple[str, list[_InferenceSlot], list[int], list[int], np.ndarray]] = []
        for trial_id, ready in ready_by_trial.items():
            model = self._trial_models.get(trial_id)
            if model is None:
                # Release unanswered rather than fabricating zeros: an all-zero
                # policy+WDL marked _STATE_RESPONSE is indistinguishable from a
                # real answer, so it reaches MCTS and is recorded as training
                # data. See SlotBroker._release_slots_for_retry for the same
                # rule on the per-trial path. _STATE_IDLE is what the client's
                # own recovery treats as "slot went away, re-submit", so a
                # transient gap costs a retry and a persistent one surfaces as
                # a loud client-side TimeoutError instead of silent poison.
                # Only this trial is skipped; other trials still get served.
                _now_nm = time.monotonic()
                if _now_nm - self._trial_no_model_warned_at.get(trial_id, 0.0) >= 30.0:
                    self._trial_no_model_warned_at[trial_id] = _now_nm
                    log.warning(
                        "shared broker has no model for trial %s; releasing %d "
                        "slot(s) for retry (NOT answering with zeros)",
                        trial_id, len(ready),
                    )
                # Counted BEFORE the release, for the reason spelled out on
                # SlotBroker._process_batch: _STATE_IDLE frees the client to
                # re-submit into the same shared slot with a different
                # batch_size, so a read afterwards belongs to the NEXT request.
                #
                # This increment was MISSING until the task #142 re-review.
                # The serve loop counts these rows at dispatch and subtracts
                # only this return, so a trial whose model had not loaded yet
                # had every one of its rows reported as served pos/s -- the
                # fourth instance of the family, in the path that is hardest to
                # notice because it is the transient one.
                unanswered_rows += sum(int(s.batch_size) for s in ready)
                for slot in ready:
                    slot.state = _STATE_IDLE
                continue
            # Tags snapshotted BEFORE batch_size/input, and echoed on response.
            # Same contract and same reason as SlotBroker._process_batch_mode
            # (audit I1) -- this broker shares the slot protocol verbatim.
            request_ids = [int(slot.request_id) for slot in ready]
            request_modes = [int(slot.request_mode) for slot in ready]
            # The mode was never read here at all (SHARED_BROKER_AUDIT B1).
            # This broker only implements dense-f32; a bf16-bits or compact
            # request was decoded AS f32 (the slot regions alias) and answered
            # with a dense f32 policy the client then re-read as bf16 -- a
            # plausible in-range WDL and a garbage policy, no exception, no
            # counter. Both production transports (evaluate_encoded's bf16 path
            # and evaluate_legal_bf16) send exactly those modes, so this is the
            # only branch that ever worked. Refuse loudly instead of
            # implementing dispatch that production does not use: the shared
            # broker is OFF (distributed_inference_shared_broker: false) and a
            # silent wrong answer is the failure class this whole audit exists
            # to delete.
            unsupported = sorted({m for m in request_modes if m != _MODE_DENSE_F32})
            if unsupported:
                n_bad = sum(1 for m in request_modes if m != _MODE_DENSE_F32)
                for slot, mode in zip(ready, request_modes, strict=True):
                    if mode != _MODE_DENSE_F32:
                        unanswered_rows += int(slot.batch_size)
                        slot.state = _STATE_IDLE
                # Throttled for the same reason as the no-model branch above:
                # a released slot comes straight back round the serve loop, so
                # an unthrottled line here writes thousands a second.
                _now_nm = time.monotonic()
                if _now_nm - self._trial_bad_mode_warned_at.get(trial_id, 0.0) >= 30.0:
                    self._trial_bad_mode_warned_at[trial_id] = _now_nm
                    log.error(
                        "shared broker cannot serve request_mode(s) %s for trial "
                        "%s: SharedSlotBroker._process_parallel implements only "
                        "_MODE_DENSE_F32 (%d). _MODE_DENSE_BF16 (%d) and "
                        "_MODE_LEGAL_BF16 (%d) have NO dispatch here -- serving "
                        "them as dense f32 returns a mis-decoded policy with no "
                        "error (SHARED_BROKER_AUDIT B1). Releasing %d slot(s) "
                        "unanswered; the per-trial SlotBroker implements all "
                        "three, this broker does not.",
                        unsupported, trial_id, _MODE_DENSE_F32,
                        _MODE_DENSE_BF16, _MODE_LEGAL_BF16, n_bad,
                    )
                self.unsupported_mode_requests += n_bad
                kept = [
                    (slot, req)
                    for slot, req, mode in zip(
                        ready, request_ids, request_modes, strict=True,
                    )
                    if mode == _MODE_DENSE_F32
                ]
                if not kept:
                    continue
                ready = [slot for slot, _ in kept]
                request_ids = [req for _, req in kept]
            batch_sizes = [slot.batch_size for slot in ready]
            xs = [np.array(slot.input[:bsz], copy=True, order="C") for slot, bsz in zip(ready, batch_sizes, strict=True)]
            xb = np.concatenate(xs, axis=0)
            packed.append((trial_id, ready, batch_sizes, request_ids, xb))

        if not packed:
            return unanswered_rows

        hang_batch = sum(int(xb.shape[0]) for *_, xb in packed)
        # Token kept for the same reason as SlotBroker._process_batch_mode.
        hang_token = self._hang_watchdog.mark_forward_start(hang_batch)
        forward_ok = False
        try:
            trial_data: list[tuple[str, list[_InferenceSlot], list[int], list[int], torch.Tensor]] = [
                (trial_id, ready, batch_sizes, request_ids,
                 torch.from_numpy(xb).to(self.device, non_blocking=True))
                for trial_id, ready, batch_sizes, request_ids, xb in packed
            ]
            # Launch forward passes in parallel on separate streams
            results: list[
                tuple[str, list[_InferenceSlot], list[int], list[int], torch.Tensor, torch.Tensor]
            ] = []
            for trial_id, ready, batch_sizes, request_ids, xt in trial_data:
                model = self._trial_models[trial_id]
                stream = self._trial_streams.get(trial_id)

                if use_cuda and stream is not None:
                    with torch.cuda.stream(stream):
                        out = _forward_no_grad(model, xt, device=self.device)
                        pol = _policy_output_full(out).detach().float().to("cpu", non_blocking=True)
                        wdl = out["wdl"].detach().float().to("cpu", non_blocking=True)
                else:
                    out = _forward_no_grad(model, xt, device=self.device)
                    pol = _policy_output_full(out).detach().float().cpu()
                    wdl = out["wdl"].detach().float().cpu()

                results.append((trial_id, ready, batch_sizes, request_ids, pol, wdl))

            # Synchronize all streams
            if use_cuda:
                for trial_id in ready_by_trial:
                    stream = self._trial_streams.get(trial_id)
                    if stream is not None:
                        stream.synchronize()

            if not self._first_inference_done and results:
                self._first_inference_done = True
                log.info("shared broker: first parallel inference complete (%d trials)", len(results))

            # Scatter results back to slots
            for _trial_id, ready, batch_sizes, request_ids, pol, wdl in results:
                pol_np = pol.numpy()
                wdl_np = wdl.numpy()
                start = 0
                for slot, bsz, req_id in zip(ready, batch_sizes, request_ids, strict=True):
                    end = start + bsz
                    slot.policy[:bsz] = pol_np[start:end]
                    slot.wdl[:bsz] = wdl_np[start:end]
                    slot.request_id = req_id
                    slot.state = _STATE_RESPONSE
                    start = end
            forward_ok = True
        finally:
            self._hang_watchdog.mark_forward_done(success=forward_ok, token=hang_token)
        return unanswered_rows

    def serve_forever(self) -> None:
        _batch_count = 0
        _total_positions = 0
        _last_report = time.monotonic()
        _last_unsupported = 0
        _last_scan = 0.0
        _report_interval = 10.0
        _scan_interval = 5.0
        self._hang_watchdog.start()

        while not self._stop:
            now = time.monotonic()

  # Periodically scan for new trials
            if now - _last_scan >= _scan_interval:
                self._scan_trials()
  # Refresh weights for all known trials. Under a watchdog stage for the
  # same reason as SlotBroker._ensure_model (audit I3): torch.load,
  # .to(device) and torch.compile happen in here, outside any forward.
                with self._hang_watchdog.stage("load_trial_weights"):
                    for tid in list(self._trial_slots.keys()):
                        self._load_trial_weights(tid)
                _last_scan = now

            if not self._all_slots:
                time.sleep(0.5)
                continue

            if any(s.state == _STATE_SHUTDOWN for _, s in self._all_slots):
                self._stop = True
                break

  # Collect ready slots grouped by trial.
  # No batching window: cross-trial batching isn't possible (different weights),
  # so delaying fast trials to wait for slow ones only adds latency.
            ready_by_trial: dict[str, list[_InferenceSlot]] = {}
            for trial_id, slot in self._all_slots:
                if slot.state == _STATE_REQUEST:
                    ready_by_trial.setdefault(trial_id, []).append(slot)

            if not ready_by_trial:
                # Same idle ladder as SlotBroker._wait_for_ready_slots (audit
                # I6): rung 0 is the original 200-poll spin + 20µs yield, so a
                # loaded shared broker is unchanged; only a fleet-wide drought
                # backs off.
                idle_since = self._idle_since
                spin_polls, sleep_s = _idle_backoff(
                    0.0 if idle_since is None else max(0.0, now - idle_since)
                )
                for _ in range(spin_polls):
                    if any(s.state == _STATE_REQUEST for _, s in self._all_slots):
                        self._idle_since = None
                        break
                else:
                    if idle_since is None:
                        self._idle_since = now
                    time.sleep(sleep_s)
                continue
            self._idle_since = None

  # Process all trials' batches in parallel on separate CUDA streams
            for ready in ready_by_trial.values():
                _batch_count += 1
                _total_positions += sum(s.batch_size for s in ready)
  # Rows released UNANSWERED were counted above but never evaluated;
  # subtract them so pos/s means served rows. Both release paths in
  # _process_parallel are in that return: an unimplemented request_mode
  # AND a trial whose model has not loaded yet. The comment used to name
  # only the first, which is exactly how the second went unfixed.
            _total_positions -= self._process_parallel(ready_by_trial)

  # Periodic metrics
            now = time.monotonic()
            if now - _last_report >= _report_interval and _batch_count > 0:
                avg_pos = _total_positions / _batch_count
                n_trials = len(self._trial_slots)
  # The B1 refusal counter's ONE reader. Appended only when non-zero, so
  # the healthy line is unchanged and any occurrence is a grep hit; a
  # refusal nobody can count is indistinguishable from one that never
  # fires, and this counter had no reader when the guard was written.
                _unsupported = int(self.unsupported_mode_requests)
                _refused_delta = _unsupported - _last_unsupported
                _last_unsupported = _unsupported
                print(
                    f"[shared-broker] {_batch_count} batches in {now - _last_report:.1f}s | "
                    f"avg {avg_pos:.1f} pos/batch | "
                    f"{_total_positions / (now - _last_report):.0f} pos/s | "
                    f"{n_trials} trials"
                    + (
                        f" | REFUSED {_refused_delta} request(s) for an "
                        f"unimplemented request_mode ({_unsupported} total) -- "
                        "this broker serves only dense f32 (audit B1)"
                        if _refused_delta else ""
                    ),
                    flush=True,
                )
                _batch_count = 0
                _total_positions = 0
                _last_report = now

    def shutdown(self) -> None:
        self._stop = True
        self._hang_watchdog.stop()
        for _, slot in self._all_slots:
            slot.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _add_hang_abort_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--hang-abort-seconds",
        type=float,
        default=_DEFAULT_HANG_ABORT_S,
        help=(
            "Hard-exit (code 42) if GPU work stays in flight longer than this. "
            "Covers model load, AOT package load and pinned allocation as well as "
            "the forward. Before the first SUCCESSFUL forward a longer cold-start "
            f"window applies instead ({_BOOT_HANG_ABORT_ENV}, default "
            f"{_DEFAULT_BOOT_HANG_ABORT_S:g}s), so a slow first compile cannot "
            "false-fire. 0 disables the watchdog entirely, cold window included. "
            f"Env {_HANG_ABORT_ENV} overrides when set. "
            f"Default: {_DEFAULT_HANG_ABORT_S:g}."
        ),
    )


def main() -> int:
  # Detect mode from argv: "shared" subcommand or legacy per-trial flags
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "shared":
        ap = argparse.ArgumentParser(description="Shared inference broker for all trials")
        ap.add_argument("mode", choices=["shared"])
        ap.add_argument("--server-root", type=str, required=True)
        ap.add_argument("--device", type=str, default="cuda")
        ap.add_argument("--compile-inference", action="store_true")
        ap.add_argument("--compile-mode", type=str, default="reduce-overhead")
        ap.add_argument("--batch-wait-ms", type=float, default=0.0)
        ap.add_argument("--slots-per-trial", type=int, default=2)
        ap.add_argument("--max-batch-per-slot", type=int, default=256)
        ap.add_argument("--input-planes", type=int, default=_CHANNELS)
        ap.add_argument("--shared-cache-dir", type=str, default=None)
        _add_hang_abort_arg(ap)
        args = ap.parse_args()
    else:
        ap = argparse.ArgumentParser(description="Per-trial shared-memory inference broker")
        ap.add_argument("--publish-dir", type=str, required=True)
        ap.add_argument("--device", type=str, default="cuda")
        ap.add_argument("--compile-inference", action="store_true")
        ap.add_argument("--compile-mode", type=str, default="reduce-overhead")
        ap.add_argument("--batch-wait-ms", type=float, default=5.0)
        # > 0 switches _gather_more_within_window to arrival-adaptive mode:
        # dispatch when no new REQUEST slot arrives for this long, with
        # --batch-wait-ms as the hard cap on total wait.
        ap.add_argument("--adaptive-idle-ms", type=float, default=0.0)
        ap.add_argument("--num-slots", type=int, default=2)
        ap.add_argument("--max-batch-per-slot", type=int, default=256)
        ap.add_argument("--slot-prefix", type=str, required=True)
        ap.add_argument("--input-planes", type=int, default=_CHANNELS)
        ap.add_argument("--shared-cache-dir", type=str, default=None)
        # Optional pre-built AOTInductor package directory (chess_b{N}.pt2 at
        # _COMPILED_BATCH_BUCKETS). Empty/absent => eager/compile path only.
        ap.add_argument(
            "--aot-dir",
            type=str,
            default=None,
            help=(
                "Directory of pre-built AOTInductor packages (chess_b{N}.pt2) "
                "for dense SlotBroker forwards. Default: off."
            ),
        )
        _add_hang_abort_arg(ap)
        args = ap.parse_args()
        args.mode = "per-trial"

    # Keep the CUDA availability probe off the driver-init path; see
    # `pin_nvml_cuda_check`. Matters here for a broker launched directly rather
    # than through scripts/train.sh, which exports the same variable.
    pin_nvml_cuda_check()

    hang_abort_seconds = resolve_hang_abort_seconds(float(args.hang_abort_seconds))

    shared_cache_raw = str(getattr(args, "shared_cache_dir", "") or "").strip()
    if shared_cache_raw:
        _configure_compile_cache(Path(shared_cache_raw).expanduser())

    if args.mode == "shared":
        broker = SharedSlotBroker(
            server_root=Path(args.server_root).expanduser(),
            slots_per_trial=int(args.slots_per_trial),
            max_batch_per_slot=int(args.max_batch_per_slot),
            device=str(args.device),
            compile_inference=bool(args.compile_inference),
            batch_wait_ms=float(args.batch_wait_ms),
            compile_mode=str(args.compile_mode),
            input_planes=int(args.input_planes),
            hang_abort_seconds=hang_abort_seconds,
        )
        try:
            broker.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            broker.shutdown()
        return 0

    # Per-trial mode
    aot_dir_arg = str(getattr(args, "aot_dir", None) or "").strip() or None
    broker = SlotBroker(
        publish_dir=Path(args.publish_dir).expanduser(),
        num_slots=int(args.num_slots),
        max_batch_per_slot=int(args.max_batch_per_slot),
        device=str(args.device),
        compile_inference=bool(args.compile_inference),
        batch_wait_ms=float(args.batch_wait_ms),
        slot_prefix=str(args.slot_prefix),
        compile_mode=str(args.compile_mode),
        input_planes=int(args.input_planes),
        adaptive_idle_ms=float(args.adaptive_idle_ms),
        hang_abort_seconds=hang_abort_seconds,
        aot_dir=aot_dir_arg,
    )

    manifest_path = Path(args.publish_dir).expanduser() / "broker_slots.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "slot_names": broker.slot_names,
                "max_batch_per_slot": int(args.max_batch_per_slot),
            }
        ),
        encoding="utf-8",
    )

    try:
        broker.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        broker.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
