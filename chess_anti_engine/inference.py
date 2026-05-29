from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import struct
import threading
import time
from dataclasses import dataclass, replace as dataclass_replace
from functools import lru_cache
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch

from chess_anti_engine.model import (
    ModelConfig,
    build_model,
    load_state_dict_tolerant,
    model_config_from_manifest_dict,
)
from chess_anti_engine.moves import COMPACT_POLICY_SIZE, COMPACT_TO_FULL_POLICY
from chess_anti_engine.utils.amp import inference_autocast

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class BatchEvaluator(Protocol):
    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...


class AsyncBatchEvaluator(BatchEvaluator, Protocol):
    """Evaluators that also expose a non-blocking GPU path (for MCTS pipelining)."""

    def evaluate_encoded_async(
        self, x: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        ...


def _policy_output(out: dict[str, torch.Tensor]) -> torch.Tensor:
    """Extract policy tensor from model output (handles both key conventions)."""
    return out["policy"] if "policy" in out else out["policy_own"]


@lru_cache(maxsize=16)
def _compact_to_full_index(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    return torch.as_tensor(COMPACT_TO_FULL_POLICY, dtype=torch.long, device=device)


def _compact_to_full_index_for(tensor: torch.Tensor) -> torch.Tensor:
    device = tensor.device
    return _compact_to_full_index(device.type, device.index)


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
) -> dict[str, torch.Tensor]:
    """Run a single forward pass under no_grad + inference_autocast.

    Centralizes the ``with torch.no_grad(): with inference_autocast(...): ...``
    pattern so every evaluator path uses the same autocast policy. Eleven
    sites used to reimplement this inline.
    """
    with torch.no_grad():
        with inference_autocast(device=device, enabled=use_amp, dtype=amp_dtype):
            return model(xt)


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
    with torch.no_grad():
        with inference_autocast(device=device, enabled=use_amp, dtype=amp_dtype):
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
    with torch.no_grad():
        with inference_autocast(device=device, enabled=use_amp, dtype=amp_dtype):
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

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xb = _coerce_input_batch(x)
        xt = torch.from_numpy(xb).to(self.device)
        out = _forward_no_grad(
            self.model, xt, device=self.device,
            use_amp=self._use_amp, amp_dtype=self._amp_dtype,
        )
        policy_out = _policy_output_full(out)
        _cpu_f32 = torch.float32
        pol = policy_out.detach().to(dtype=_cpu_f32, device="cpu").numpy()
        wdl = out["wdl"].detach().to(dtype=_cpu_f32, device="cpu").numpy()
        return pol, wdl

    def evaluate_encoded_async(
        self,
        x: np.ndarray,
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
            stream.wait_event(event_default) # torch stubs have duplicate Event types
            xt = torch.from_numpy(xb).to(self.device, non_blocking=True)
            out = _forward_no_grad(
                self.model, xt, device=self.device,
                use_amp=self._use_amp, amp_dtype=self._amp_dtype,
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
    ) -> None:
        super().__init__(model, device=device, use_amp=use_amp, amp_dtype=amp_dtype)
        self._max_batch = int(max_batch)
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}")
        self._n_slots = int(n_slots)

        _pin = self._use_cuda
        self._input_bf16 = bool(input_bf16 and self._use_cuda and use_amp)
        self._pinned_inputs: list[torch.Tensor] = [
            torch.empty(
                (self._max_batch, _CHANNELS, _BOARD_H, _BOARD_W),
                dtype=torch.float32, pin_memory=_pin,
            ) for _ in range(self._n_slots)
        ]
        self._pinned_inputs_bf16: list[torch.Tensor] | None = (
            [
                torch.empty(
                    (self._max_batch, _CHANNELS, _BOARD_H, _BOARD_W),
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

    def evaluate_inplace(
        self, bsz: int, *, copy_out: bool = True, slot: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on data already written to ``get_input_buffer(slot)``."""
        if bsz <= 0:
            return np.empty((0, _POLICY_SIZE), dtype=np.float32), np.empty((0, _WDL_SIZE), dtype=np.float32)
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")
        if not 0 <= slot < self._n_slots:
            raise ValueError(f"slot {slot} out of range [0, {self._n_slots})")
        return self._run_forward(bsz, copy_out=copy_out, slot=slot)

    def evaluate_encoded(
        self, x: np.ndarray, *, copy_out: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if x.ndim != 4:
            raise ValueError(f"expected 4D input, got {x.ndim}D")
        bsz = x.shape[0]
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")

        if x.dtype == np.uint16:
            if self._pinned_inputs_bf16_bits_np is not None:
                self._pinned_inputs_bf16_bits_np[0][:bsz] = x
                self._slot_input_bf16[0] = True
            else:
                self._pinned_inputs_np[0][:bsz] = _bf16_bits_to_float32_np(x)
                self._slot_input_bf16[0] = False
        else:
            self._pinned_inputs_np[0][:bsz] = x
            self._slot_input_bf16[0] = False
        return self._run_forward(bsz, copy_out=copy_out, slot=0)

    def _run_forward(
        self, bsz: int, *, copy_out: bool = True, slot: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        pin_in = self._pinned_inputs[slot]
        pin_pol = self._pinned_pols[slot]
        pin_wdl = self._pinned_wdls[slot]
        if not self._use_cuda:
            xt = pin_in[:bsz]
            out = _forward_no_grad(
                self.model, xt, device=self.device,
                use_amp=self._use_amp, amp_dtype=self._amp_dtype,
            )
            return _policy_output_full(out).detach().float().numpy(), out["wdl"].detach().float().numpy()

        xt = self._device_input(bsz, slot=slot)
        out = _forward_no_grad(
            self.model, xt, device=self.device,
            use_amp=self._use_amp, amp_dtype=self._amp_dtype,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        """Pinned-memory async eval: H2D via DMA, non-blocking D2H."""
        if x.ndim != 4:
            raise ValueError(f"expected 4D input, got {x.ndim}D")
        bsz = x.shape[0]
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")

        if not self._use_cuda:
            return super().evaluate_encoded_async(x)

        if x.dtype == np.uint16:
            if self._pinned_inputs_bf16_bits_np is not None:
                self._pinned_inputs_bf16_bits_np[0][:bsz] = x
                self._slot_input_bf16[0] = True
            else:
                self._pinned_inputs_np[0][:bsz] = _bf16_bits_to_float32_np(x)
                self._slot_input_bf16[0] = False
        else:
            self._pinned_inputs_np[0][:bsz] = x
            self._slot_input_bf16[0] = False
        return self._async_forward(bsz, slot=0)

    def evaluate_inplace_async(
        self, bsz: int, *, slot: int = 0,
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
            return super().evaluate_encoded_async(xb)
        return self._async_forward(bsz, slot=slot)

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

    def _async_forward(
        self, bsz: int, *, slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        pin_pol = self._pinned_pols[slot]
        pin_wdl = self._pinned_wdls[slot]

        if self._stream is None:
            self._stream = torch.cuda.Stream(device=self.device)
        stream = self._stream

        event_default = torch.cuda.Event()
        event_default.record(torch.cuda.current_stream(self.device))

        with torch.cuda.stream(stream):
            stream.wait_event(event_default)  # pyright: ignore[reportArgumentType]  # torch stubs have duplicate Event types
            xt = self._device_input(bsz, slot=slot)
            out = _forward_no_grad(
                self.model, xt, device=self.device,
                use_amp=self._use_amp, amp_dtype=self._amp_dtype,
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
            stream.wait_event(event_default)  # torch stubs have duplicate Event types
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


# Bucket ladder for ThreadedBatchEvaluator's coalesced GPU forwards. Coarser
# than the per-batch AOT bucket list (_BATCH_BUCKETS) because torch.compile
# graphs are cheaper to recapture across mid-range sizes.
_COMPILED_BATCH_BUCKETS = (
    128, 170, 256, 340, 384, 512, 680, 768, 1020, 1024, 1190, 1536, 2048, 4096,
)

_COMPILED_LEGAL_POLICY_BUCKETS = (
    32_768, 65_536, 131_072, 262_144, 524_288,
)


def _compiled_padded_batch_size(total: int, *, capacity: int | None = None) -> int:
    if total <= 0:
        return 0
    for bucket in _COMPILED_BATCH_BUCKETS:
        if bucket >= total:
            if capacity is not None and bucket > capacity:
                return total
            return bucket
    return total


def _compiled_padded_legal_policy_size(total: int, *, capacity: int | None = None) -> int:
    if total <= 0:
        return 0
    for bucket in _COMPILED_LEGAL_POLICY_BUCKETS:
        if bucket >= total:
            if capacity is not None and bucket > capacity:
                return total
            return bucket
    return total


class ThreadedBatchEvaluator:
    """Thread-safe batched evaluator for multi-threaded selfplay.

    Multiple selfplay threads submit inference requests via evaluate_encoded().
    A dedicated GPU thread accumulates submissions into large batches and runs
    a single GPU forward pass, then scatters results back to waiting threads.

    Only the GPU thread ever touches the model or CUDA — safe with torch.compile.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str = "cuda",
        max_batch: int = 4096,
        min_batch: int = 256,
        accumulation_timeout_s: float = 0.001,
        use_amp: bool = True,
        amp_dtype: str = "auto",
    ) -> None:

        self.device = str(device)
        self._min_batch = int(min_batch)
        self._max_batch = int(max_batch)
        self._timeout = float(accumulation_timeout_s)
        self._queue: queue.Queue = queue.Queue()
        self._stop = False

  # GPU evaluator created on the GPU thread to ensure CUDA context ownership.
        self._model = model
        self._use_amp = bool(use_amp)
        self._amp_dtype = str(amp_dtype)
        self._gpu_eval: DirectGPUEvaluator | None = None
        self._gpu_ready = threading.Event()

        self._gpu_thread = threading.Thread(target=self._gpu_loop, daemon=True)
        self._gpu_thread.start()
        self._gpu_ready.wait()  # Block until GPU eval is initialized

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Submit a batch from a selfplay thread, block until GPU results ready."""
        if not self._gpu_thread.is_alive():
            raise RuntimeError("GPU thread died")
        event = threading.Event()
        result: dict = {}
        self._queue.put((x, event, result))
  # Wait with timeout and check GPU thread health
        while not event.wait(timeout=5.0):
            if not self._gpu_thread.is_alive():
                raise RuntimeError("GPU thread died while waiting for results")
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["pol"], result["wdl"]

    def update_model(self, model: torch.nn.Module) -> None:
        """Swap the model (called from main thread between iterations)."""
        if not self._gpu_thread.is_alive():
            raise RuntimeError("GPU thread died")
        event = threading.Event()
        self._queue.put(("_update_model", model, event))
        while not event.wait(timeout=5.0):
            if not self._gpu_thread.is_alive():
                raise RuntimeError("GPU thread died during model update")

    def shutdown(self) -> None:
        self._stop = True
        self._queue.put(None)
        self._gpu_thread.join(timeout=5.0)

    @staticmethod
    def _is_model_update(item: object) -> bool:
        """Sentinel detection: model-update items are ``(str, model, event)`` tuples
        instead of the normal ``(np.ndarray, event, result_dict)``."""
        return isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], str)

    def _accumulate_batch(self, first) -> list:
        """Pull items from the queue until min_batch is hit or timeout expires.

        Returns the list of pending items (always includes ``first``). Items
        that would overflow ``max_batch`` or model-update sentinels are pushed
        back onto the queue for the next loop iteration.
        """
        pending = [first]
        total = first[0].shape[0]
        deadline = time.monotonic() + self._timeout
        while total < self._min_batch and total < self._max_batch and time.monotonic() < deadline:
            try:
                item = self._queue.get(timeout=max(0.0001, deadline - time.monotonic()))
            except queue.Empty:
                break
            if item is None:
                break
            if self._is_model_update(item):
  # Defer model-swap until pending batch processes with current model.
                self._queue.put(item)
                break
            item_size = item[0].shape[0]
            if total + item_size > self._max_batch:
                self._queue.put(item)
                break
            pending.append(item)
            total += item_size
        return pending

    def _build_padded_batch(self, pending: list) -> tuple[np.ndarray, int]:
        """Concat + bucket-pad pending inputs. Returns ``(combined, real_total)``."""
        total = sum(p[0].shape[0] for p in pending)
        combined = pending[0][0] if len(pending) == 1 else np.concatenate([p[0] for p in pending], axis=0)
        padded_size = _compiled_padded_batch_size(total, capacity=self._max_batch)
        if padded_size > total:
            pad = np.zeros((padded_size - total, *combined.shape[1:]), dtype=combined.dtype)
            combined = np.concatenate([combined, pad], axis=0)
        return combined, total

    @staticmethod
    def _propagate_error_to_pending(pending: list, exc: BaseException) -> None:
        for _x, event, result in pending:  # skylos: ignore (_x unpacked but unused by convention)
            result["error"] = str(exc)
            event.set()

    @staticmethod
    def _scatter_results(pending: list, pol_all: np.ndarray, wdl_all: np.ndarray) -> None:
        offset = 0
        for x, event, result in pending:
            n = x.shape[0]
            result["pol"] = pol_all[offset:offset + n].copy()
            result["wdl"] = wdl_all[offset:offset + n].copy()
            offset += n
            event.set()

    def _gpu_loop(self) -> None:
        """Dedicated GPU thread: accumulate requests, run batched inference."""
        try:
            self._gpu_eval = DirectGPUEvaluator(
                self._model, device=self.device, max_batch=self._max_batch,
                use_amp=self._use_amp, amp_dtype=self._amp_dtype,
            )
        finally:
            self._gpu_ready.set()  # Unblock constructor even on failure

        while not self._stop:
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if first is None:
                break

            if self._is_model_update(first):
                _, new_model, event = first
                self._gpu_eval.model = new_model
                event.set()
                continue

            pending = self._accumulate_batch(first)
            if not pending:
                continue
            combined, total = self._build_padded_batch(pending)

            try:
                pol_all, wdl_all = self._gpu_eval.evaluate_encoded(combined, copy_out=False)
                pol_all = pol_all[:total]
                wdl_all = wdl_all[:total]
            except Exception as exc:
                self._propagate_error_to_pending(pending, exc)
                continue
            except BaseException as exc:
  # Fatal (KeyboardInterrupt, SystemExit) — unblock all threads first.
                self._propagate_error_to_pending(pending, exc)
                raise

            self._scatter_results(pending, pol_all, wdl_all)


# Batch size buckets for AOT-compiled inference models.
# Only sizes with a corresponding chess_b{N}.pt2 in the AOT dir are loaded.
_BATCH_BUCKETS = (
    1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64, 96,
    128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 170,
    192, 224, 228, 232, 236, 240, 244, 248, 256, 288, 340,
    384, 448, 512, 768, 1024, 1536, 2048, 3072, 4096,
)


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
    ) -> None:
        self.device = str(device)
        self._max_batch = int(max_batch)
        aot_dir = Path(aot_dir)

  # Load compiled models in parallel (CUDA driver is thread-safe for loading).
        from concurrent.futures import ThreadPoolExecutor
        pkgs: dict[int, Path] = {}
        for b in _BATCH_BUCKETS:
            if b > self._max_batch:
                break
            pkg = aot_dir / f"chess_b{b}.pt2"
            if pkg.exists():
                pkgs[b] = pkg
        if not pkgs:
            raise FileNotFoundError(f"No .pt2 packages found in {aot_dir}")

        def _load(item: tuple[int, Path]) -> tuple[int, Any]:
            return item[0], torch._inductor.aoti_load_package(str(item[1]))

        with ThreadPoolExecutor(max_workers=min(4, len(pkgs))) as pool:
            self._models = dict(pool.map(_load, pkgs.items()))
        self._sorted_buckets = sorted(self._models.keys())
        self._constant_fqns = list(next(iter(self._models.values())).get_constant_fqns())

  # Pre-allocate pinned buffers
        _pin = self.device.startswith("cuda")
        self._pinned_input = torch.empty(
            (self._max_batch, _CHANNELS, _BOARD_H, _BOARD_W),
            dtype=torch.bfloat16, pin_memory=_pin,
        )
        self._pinned_input_np = np.empty(
            (self._max_batch, _CHANNELS, _BOARD_H, _BOARD_W),
            dtype=np.float32,
        )

    def load_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Update all bucket models with new weights from a state_dict."""
        constants = {
            fqn: state_dict[fqn].to(device=self.device, dtype=torch.bfloat16).contiguous()
            for fqn in self._constant_fqns
            if fqn in state_dict
        }
        for model in self._models.values():
            model.load_constants(constants, check_full_update=False)

    def _pick_bucket(self, bsz: int) -> int:
        for b in self._sorted_buckets:
            if b >= bsz:
                return b
        return self._sorted_buckets[-1]

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x.ndim != 4:
            raise ValueError(f"expected 4D input, got {x.ndim}D")
        bsz = x.shape[0]
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")
        bucket = self._pick_bucket(bsz)
        model = self._models[bucket]

  # Copy into pinned buffer and convert to BF16 on GPU
        self._pinned_input_np[:bsz] = x
        xt = torch.from_numpy(self._pinned_input_np[:bucket]).to(
            device=self.device, dtype=torch.bfloat16, non_blocking=True,
        )

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

        self._pinned_input_np[:bsz] = x
        xt = torch.from_numpy(self._pinned_input_np[:bucket]).to(
            device=self.device, dtype=torch.bfloat16, non_blocking=True,
        )

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
#   [4:8]      batch_size int32   (number of positions in this request)
#   [8:...]    input      float32 or bf16-bits[max_batch, 146, 8, 8]
#   [after input]  policy/output float32[max_batch, 4672] or compact metadata/bf16 bits
#   [after policy] wdl    float32[max_batch, 3]
#
# Flow:
#   1. Worker writes input + batch_size, sets state = REQUEST
#   2. Broker sees state == REQUEST, reads input, runs GPU inference
#   3. Broker writes policy + wdl, sets state = RESPONSE
#   4. Worker reads output, sets state = IDLE
#
# No sockets, no per-request allocation, no connection setup/teardown.

_STATE_IDLE = 0
_STATE_REQUEST = 1
_STATE_RESPONSE = 2
_STATE_SHUTDOWN = 255
_MODE_DENSE_F32 = 0
_MODE_DENSE_BF16 = 1
_MODE_LEGAL_BF16 = 2

_CHANNELS = 146
_BOARD_H = 8
_BOARD_W = 8
_POLICY_SIZE = 4672
_WDL_SIZE = 3
_F32 = np.float32
_F32_BYTES = 4

_HEADER_BYTES = 8  # 1 byte state + 1 byte mode + 2 pad + 4 byte batch_size


@dataclass(frozen=True)
class _SlotLayout:
    max_batch: int
    input_offset: int
    input_bytes: int
    policy_offset: int
    policy_bytes: int
    wdl_offset: int
    wdl_bytes: int
    total_bytes: int

    @staticmethod
    def compute(max_batch: int) -> _SlotLayout:
        ib = max_batch * _CHANNELS * _BOARD_H * _BOARD_W * _F32_BYTES
        pb = max_batch * _POLICY_SIZE * _F32_BYTES
        wb = max_batch * _WDL_SIZE * _F32_BYTES
        io = _HEADER_BYTES
        po = io + ib
        wo = po + pb
        return _SlotLayout(
            max_batch=max_batch,
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
        "_shm", "_layout", "_owns", "_buf",
        "input", "input_bf16_bits", "policy", "policy_i32", "policy_u16", "wdl",
    )
    _buf: memoryview

    def __init__(self, shm: SharedMemory, layout: _SlotLayout, *, owns: bool = False):
        self._shm = shm
        self._layout = layout
        self._owns = owns
        assert shm.buf is not None  # attached SharedMemory always has a buffer
        self._buf = shm.buf
        self.input: np.ndarray = np.ndarray(
            (layout.max_batch, _CHANNELS, _BOARD_H, _BOARD_W),
            dtype=_F32,
            buffer=self._buf,
            offset=layout.input_offset,
        )
        self.input_bf16_bits: np.ndarray = np.ndarray(
            (layout.max_batch, _CHANNELS, _BOARD_H, _BOARD_W),
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
        return int(self._buf[0])

    @state.setter
    def state(self, v: int) -> None:
        self._buf[0] = int(v) & 0xFF

    @property
    def batch_size(self) -> int:
        return struct.unpack_from("<i", self._buf, 4)[0]

    @batch_size.setter
    def batch_size(self, v: int) -> None:
        struct.pack_into("<i", self._buf, 4, int(v))

    @property
    def request_mode(self) -> int:
        return int(self._buf[1])

    @request_mode.setter
    def request_mode(self, v: int) -> None:
        self._buf[1] = int(v) & 0xFF

    @property
    def name(self) -> str:
        return self._shm.name

    def close(self) -> None:
        try:
            self._shm.close()
        except Exception:
            pass
        if self._owns:
            try:
                self._shm.unlink()
            except Exception:
                pass


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
    ) -> None:
        self.publish_dir = Path(publish_dir)
        self.device = str(device)
        self.compile_inference = bool(compile_inference)
        self.compile_mode = str(compile_mode or "reduce-overhead")
        self._first_inference_pending = False
        self.batch_wait_ms = float(batch_wait_ms)
        self._model: torch.nn.Module | None = None
        self._model_sha: str | None = None
        self._stop = False
        self._manifest_cache: dict | None = None
        self._manifest_cache_sig: tuple[int, int] | None = None
        self._timing_metrics: dict[str, float] | None = None

        self._layout = _SlotLayout.compute(max_batch_per_slot)
        self._slots: list[_InferenceSlot] = []
        self._slot_names: list[str] = []

  # Pre-allocated pinned buffers for zero-copy GPU transfer.
        _total_cap = num_slots * max_batch_per_slot
        _pin = "cuda" in self.device and torch.cuda.is_available()
        self._pinned_input = torch.empty(
            (_total_cap, _CHANNELS, _BOARD_H, _BOARD_W),
            dtype=torch.float32, pin_memory=_pin,
        )
        self._pinned_input_bf16 = torch.empty(
            (_total_cap, _CHANNELS, _BOARD_H, _BOARD_W),
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
  # Pinned tensors need force=True for numpy conversion.
        self._pinned_input_np = self._pinned_input.numpy(force=True)
        self._pinned_input_bf16_bits_np = self._pinned_input_bf16.view(torch.uint16).numpy(force=True)
        self._pinned_pol_np = self._pinned_pol.numpy(force=True)
        self._pinned_wdl_np = self._pinned_wdl.numpy(force=True)
        self._pinned_legal_pol_bf16_bits_np = self._pinned_legal_pol_bf16_bits.numpy(force=True)

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
            self._slots.append(slot)
            self._slot_names.append(name)

    @property
    def slot_names(self) -> list[str]:
        return list(self._slot_names)

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
        model_path = self.publish_dir / "latest_model.pt"
        ckpt = torch.load(str(model_path), map_location="cpu")
        sd = ckpt.get("model", ckpt)
        model = build_model(model_cfg)
        load_state_dict_tolerant(model, sd, label="broker-model")
        model.to(self.device)
        model.eval()
        if hasattr(model, "_inference_only"):
            setattr(model, "_inference_only", True)
        if self.compile_inference and self.device.startswith("cuda"):
            model = cast("torch.nn.Module", torch.compile(model, mode=self.compile_mode))
        self._model = model
        self._model_sha = model_sha
        self._first_inference_pending = bool(self.compile_inference)

  # -- batch processing --

    def _process_batch(self, ready: list[_InferenceSlot]) -> None:
        by_mode: dict[int, list[_InferenceSlot]] = {}
        for slot in ready:
            by_mode.setdefault(int(slot.request_mode), []).append(slot)
        for mode, slots in by_mode.items():
            self._process_batch_mode(slots, mode=mode)

    def _process_batch_mode(self, ready: list[_InferenceSlot], *, mode: int) -> None:
        _timing = getattr(self, "_timing_metrics", None)
        _t_pack0 = time.perf_counter()
        self._ensure_model()
        if self._model is None:
            for slot in ready:
                bsz = max(0, min(int(slot.batch_size), self._layout.max_batch))
                slot.policy[:bsz].fill(0.0)
                slot.wdl[:bsz].fill(0.0)
                slot.state = _STATE_RESPONSE
            return

        use_bf16_input = mode in (_MODE_DENSE_BF16, _MODE_LEGAL_BF16)
        compact_legal = mode == _MODE_LEGAL_BF16

  # Gather inputs directly into pre-allocated pinned buffer (one memcpy
  # from shm → pinned, then async DMA to GPU — no intermediate allocs).
        active: list[_InferenceSlot] = []
        batch_sizes: list[int] = []
        legal_counts_by_slot: list[np.ndarray] = []
        legal_flat_by_slot: list[np.ndarray] = []
        total = 0
        for slot in ready:
            bsz = max(0, min(int(slot.batch_size), self._layout.max_batch))
            if compact_legal:
                meta = slot.policy_i32
                n_legal = max(0, int(meta[0]))
                counts = np.asarray(meta[1:1 + bsz], dtype=np.int32).copy()
                flat = np.asarray(meta[1 + bsz:1 + bsz + n_legal], dtype=np.int32).copy()
                if (
                    counts.shape[0] != bsz
                    or int(counts.sum()) != n_legal
                    or flat.shape[0] != n_legal
                    or (flat.size > 0 and (int(flat.min()) < 0 or int(flat.max()) >= _POLICY_SIZE))
                ):
                    slot.policy_u16[:1] = 0
                    slot.wdl[:bsz].fill(0.0)
                    slot.request_mode = _MODE_DENSE_F32
                    slot.state = _STATE_RESPONSE
                    continue
                legal_counts_by_slot.append(counts)
                legal_flat_by_slot.append(flat)
            if use_bf16_input:
                self._pinned_input_bf16_bits_np[total:total + bsz] = slot.input_bf16_bits[:bsz]
            else:
                self._pinned_input_np[total:total + bsz] = slot.input[:bsz]
            active.append(slot)
            batch_sizes.append(bsz)
            total += bsz

        assert total <= self._pinned_input.shape[0], (
            f"gather overflow: {total} > {self._pinned_input.shape[0]}"
        )
        if total <= 0:
            for slot in active:
                slot.request_mode = _MODE_DENSE_F32
                slot.state = _STATE_RESPONSE
            return
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
            for bsz, counts in zip(batch_sizes, legal_counts_by_slot, strict=True):
                n_legal = int(counts.sum())
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
            legal_counts_all = (
                np.concatenate(legal_counts_by_slot).astype(np.int64, copy=False)
                if legal_counts_by_slot else np.empty((0,), dtype=np.int64)
            )
            legal_flat_all = (
                np.concatenate(legal_flat_by_slot).astype(np.int64, copy=False)
                if legal_flat_by_slot else np.empty((0,), dtype=np.int64)
            )
            legal_rows_all = (
                np.concatenate(compact_row_parts).astype(np.int64, copy=False)
                if compact_row_parts else np.empty((0,), dtype=np.int64)
            )
        if _timing is not None:
            _timing["pack_s"] += time.perf_counter() - _t_pack0
        _t_forward0 = time.perf_counter()
        xt = pin_input[:forward_total].to(self.device, non_blocking=True)

        first_inf = self._first_inference_pending
        if first_inf:
            inf_t0 = time.time()

        use_legal_rows_forward = compact_legal and _supports_legal_policy_rows_forward(self._model)
        use_legal_forward = (
            compact_legal
            and not use_legal_rows_forward
            and _supports_legal_policy_forward(self._model)
        )
        if use_legal_rows_forward:
            assert legal_flat_all is not None
            assert legal_rows_all is not None
            legal_real = int(legal_flat_all.shape[0])
            legal_capacity = int(self._pinned_legal_pol_bf16_bits.shape[0])
            legal_forward_total = (
                _compiled_padded_legal_policy_size(legal_real, capacity=legal_capacity)
                if self.compile_inference else legal_real
            )
            if legal_forward_total > legal_real:
                legal_flat_padded = np.zeros((legal_forward_total,), dtype=np.int64)
                legal_rows_padded = np.zeros((legal_forward_total,), dtype=np.int64)
                legal_flat_padded[:legal_real] = legal_flat_all
                legal_rows_padded[:legal_real] = legal_rows_all
            else:
                legal_flat_padded = legal_flat_all
                legal_rows_padded = legal_rows_all
            legal_flat_gpu = torch.as_tensor(legal_flat_padded, dtype=torch.long, device=self.device)
            legal_rows_gpu = torch.as_tensor(legal_rows_padded, dtype=torch.long, device=self.device)
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
                rows_parts: list[np.ndarray] = []
                cols_parts: list[np.ndarray] = []
                row_base = 0
                for bsz, counts, flat in zip(batch_sizes, legal_counts_by_slot, legal_flat_by_slot, strict=True):
                    n_legal = int(counts.sum())
                    if n_legal > 0:
                        rows_parts.append(
                            np.repeat(
                                np.arange(row_base, row_base + bsz, dtype=np.int64),
                                counts.astype(np.int64, copy=False),
                            )
                        )
                        cols_parts.append(flat.astype(np.int64, copy=False))
                    row_base += bsz
                if rows_parts:
                    rows = torch.as_tensor(np.concatenate(rows_parts), dtype=torch.long, device=self.device)
                    cols = torch.as_tensor(np.concatenate(cols_parts), dtype=torch.long, device=self.device)
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
        for slot, bsz in zip(active, batch_sizes, strict=True):
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
            slot.state = _STATE_RESPONSE
            start = end
        if _timing is not None:
            _timing["scatter_s"] += time.perf_counter() - _t_scatter0

  # -- main loop --

    def _wait_for_ready_slots(self) -> bool:
        """Tight-spin with 20µs yield until at least one slot becomes ready.

        Returns False to skip the rest of the loop iteration (no slot ready
        within the spin budget).
        """
        for _ in range(200):
            if any(s.state == _STATE_REQUEST for s in self._slots):
                return True
        time.sleep(0.00002)
        return False

    def _gather_more_within_window(self, ready: list) -> None:
        """Block up to ``batch_wait_ms`` for more slots to enter REQUEST state.

        Mutates ``ready`` in place. Exits early if all slots have responded
        (request/response/shutdown — no more can arrive) or the window fills.
        """
        if self.batch_wait_ms <= 0:
            return
        deadline = time.monotonic() + (self.batch_wait_ms / 1000.0)
        while time.monotonic() < deadline:
            if all(
                s.state in (_STATE_REQUEST, _STATE_RESPONSE, _STATE_SHUTDOWN)
                for s in self._slots
            ):
                return
            more = [s for s in self._slots if s.state == _STATE_REQUEST and s not in ready]
            if more:
                ready.extend(more)
            if len(ready) >= len(self._slots):
                return
            time.sleep(0.0001)

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
            f"sync+scatter={m['scatter_s'] * 1000.0 / m['batches']:.2f}ms",
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

        while not self._stop:
            if any(s.state == _STATE_SHUTDOWN for s in self._slots):
                self._stop = True
                break

            ready = [s for s in self._slots if s.state == _STATE_REQUEST]
            if not ready:
                if not self._wait_for_ready_slots():
                    continue
                continue

            self._gather_more_within_window(ready)

  # Re-collect in case some slots changed during the wait window.
            ready = [s for s in self._slots if s.state == _STATE_REQUEST]
            if ready:
                metrics["batches"] += 1
                metrics["positions"] += sum(s.batch_size for s in ready)
                metrics["slots"] += len(ready)
                self._timing_metrics = metrics
                self._process_batch(ready)

            self._maybe_print_broker_metrics(metrics, time.monotonic(), report_interval)

    def shutdown(self) -> None:
        self._stop = True
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
    ) -> None:
        self._slot_name = str(slot_name)
        self._layout = _SlotLayout.compute(max_batch)
        self._shm: SharedMemory | None = None
        self._slot: _InferenceSlot | None = None
        self._request_timeout_s = max(0.001, float(request_timeout_s))
        self._lock = threading.Lock()

    def _disconnect(self) -> None:
        shm = self._shm
        self._slot = None
        self._shm = None
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass

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
                    )
                time.sleep(0.01)
                continue
            _detach_attached_shm_from_resource_tracker(shm)
            self._shm = shm
            self._slot = _InferenceSlot(shm, self._layout, owns=False)
            return self._slot

    @property
    def supports_input_bf16_bits(self) -> bool:
        return True

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._evaluate_encoded_locked(x)

    def _evaluate_encoded_locked(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

        def _submit(slot: _InferenceSlot) -> None:
  # Write input directly into shared memory (one memcpy).
            if request_mode == _MODE_DENSE_BF16:
                slot.input_bf16_bits[:bsz] = xb
            else:
                slot.input[:bsz] = xb
            slot.request_mode = request_mode
            slot.batch_size = bsz
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

            def _submit(slot: _InferenceSlot) -> None:
                slot.input_bf16_bits[:bsz] = xb
                slot.policy_i32[0] = n_legal
                slot.policy_i32[1:1 + bsz] = counts
                slot.policy_i32[1 + bsz:1 + bsz + n_legal] = flat
                slot.request_mode = _MODE_LEGAL_BF16
                slot.batch_size = bsz
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
        while True:
            slot = self._connect(deadline=deadline)

            submit(slot)

  # Wait for response. Keep the fast spin path for short broker latency,
  # but recover if the broker went away and the slot had to be recreated.
            spins = 0
            retry = False
            while True:
  # slot.state is shared-memory; at runtime the broker writes
  # other state values concurrently, so we wrap in int() to keep
  # pyright from narrowing to the last literal we stored.
                state = int(slot.state)
                if state == _STATE_RESPONSE:
                    pol, wdl = read(slot)
                    slot.state = _STATE_IDLE
                    return pol, wdl
                if state == _STATE_SHUTDOWN or state == _STATE_IDLE:
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
                raise RuntimeError("inference broker shut down while request was in flight")
            if retry:
                time.sleep(0.01)

    def close(self) -> None:
        self._disconnect()


class MultiSlotInferenceClient:
    """Thread-safe fan-out across multiple broker slots.

    Each underlying slot still serializes its own request/response protocol,
    but different selfplay threads can make progress through different slots.
    """

    def __init__(
        self,
        *,
        slot_names: list[str],
        max_batch: int,
        request_timeout_s: float = 30.0,
    ) -> None:
        names = [str(n).strip() for n in slot_names if str(n).strip()]
        if not names:
            raise ValueError("MultiSlotInferenceClient requires at least one slot")
        self._clients = [
            SlotInferenceClient(
                slot_name=name,
                max_batch=max_batch,
                request_timeout_s=request_timeout_s,
            )
            for name in names
        ]
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
        self._slot_requests = [0 for _ in self._clients]
        self._slot_positions = [0 for _ in self._clients]
        self._slot_wait_s = [0.0 for _ in self._clients]
        self._slot_roundtrip_s = [0.0 for _ in self._clients]

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx, client, wait_s = self._acquire_client()
        t0 = time.perf_counter()
        try:
            return client.evaluate_encoded(x)
        finally:
            self._release_client(
                idx, client,
                positions=int(x.shape[0]),
                legal=False,
                wait_s=wait_s,
                roundtrip_s=time.perf_counter() - t0,
            )

    @property
    def supports_input_bf16_bits(self) -> bool:
        return True

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx, client, wait_s = self._acquire_client()
        t0 = time.perf_counter()
        try:
            return client.evaluate_legal_bf16(x, legal_flat, legal_counts)
        finally:
            self._release_client(
                idx, client,
                positions=int(x.shape[0]),
                legal=True,
                wait_s=wait_s,
                roundtrip_s=time.perf_counter() - t0,
            )

    def _acquire_client(self) -> tuple[int, SlotInferenceClient, float]:
        t0 = time.perf_counter()
        idx, client = self._available_clients.get()
        wait_s = time.perf_counter() - t0
        with self._stats_lock:
            self._inflight += 1
            self._max_inflight = max(self._max_inflight, self._inflight)
        return idx, client, wait_s

    def _release_client(
        self,
        idx: int,
        client: SlotInferenceClient,
        *,
        positions: int,
        legal: bool,
        wait_s: float,
        roundtrip_s: float,
    ) -> None:
        with self._stats_lock:
            self._lifetime_requests += 1
            self._lifetime_positions += int(positions)
            if legal:
                self._lifetime_legal_requests += 1
                self._lifetime_legal_positions += int(positions)
            self._lifetime_wait_s += float(wait_s)
            self._lifetime_roundtrip_s += float(roundtrip_s)
            self._slot_requests[idx] += 1
            self._slot_positions[idx] += int(positions)
            self._slot_wait_s[idx] += float(wait_s)
            self._slot_roundtrip_s[idx] += float(roundtrip_s)
            self._inflight = max(0, self._inflight - 1)
        self._available_clients.put((idx, client))

    @property
    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            requests = int(self._lifetime_requests)
            positions = int(self._lifetime_positions)
            return {
                "slots": len(self._clients),
                "available_slots": int(self._available_clients.qsize()),
                "inflight": int(self._inflight),
                "max_inflight": int(self._max_inflight),
                "lifetime_requests": requests,
                "lifetime_positions": positions,
                "lifetime_legal_requests": int(self._lifetime_legal_requests),
                "lifetime_legal_positions": int(self._lifetime_legal_positions),
                "lifetime_wait_s": float(self._lifetime_wait_s),
                "lifetime_roundtrip_s": float(self._lifetime_roundtrip_s),
                "avg_rows_per_request": positions / requests if requests else 0.0,
                "avg_wait_ms": 1000.0 * self._lifetime_wait_s / requests if requests else 0.0,
                "avg_roundtrip_ms": 1000.0 * self._lifetime_roundtrip_s / requests if requests else 0.0,
                "slot_requests": list(self._slot_requests),
                "slot_positions": list(self._slot_positions),
                "slot_wait_s": list(self._slot_wait_s),
                "slot_roundtrip_s": list(self._slot_roundtrip_s),
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
    ) -> None:
        self.server_root = Path(server_root)
        self.slots_per_trial = int(slots_per_trial)
        self.device = str(device)
        self.compile_inference = bool(compile_inference)
        self.compile_mode = str(compile_mode or "reduce-overhead")
        self.batch_wait_ms = float(batch_wait_ms)
        self._stop = False

        self._layout = _SlotLayout.compute(max_batch_per_slot)

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
        self._all_slots: list[tuple[str, _InferenceSlot]] = []

    def _register_new_trial(self, trial_id: str) -> None:
        """Allocate ``slots_per_trial`` shared-memory slots for a freshly seen trial."""
        from chess_anti_engine.tune._utils import (
            stable_seed_u32,  # deferred: avoids circular import
        )
        h = stable_seed_u32("slot-prefix", trial_id)
        slot_prefix = f"cae-{h:08x}"

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
        config_key = (
            str(mc.get("kind", "transformer")),
            int(mc.get("embed_dim", 256)),
            int(mc.get("num_layers", 6)),
            int(mc.get("num_heads", 8)),
            float(mc.get("ffn_mult", 2)),
            bool(mc.get("use_smolgen", True)),
            bool(mc.get("use_nla", False)),
            bool(mc.get("use_qk_rmsnorm", False)),
            str(mc.get("input_pos_encoding", "none")),
            bool(mc.get("use_deepnorm", False)),
            str(mc.get("policy_encoding", "az_4672")),
            str(mc.get("input_history_encoding", "legacy")),
            str(mc.get("input_global_embedding", "none")),
            int(mc.get("input_global_embedding_channels", 0)),
            str(mc.get("input_square_embedding", "none")),
            str(mc.get("qkv_projection", "fused")),
            str(mc.get("smolgen_mode", "shared")),
            str(mc.get("smolgen_bias_scale", "none")),
            str(mc.get("smolgen_bias_norm", "none")),
            str(mc.get("arc_attention_bias", "none")),
            bool(mc.get("smolgen_relation_basis", False)),
            str(mc.get("smolgen_relation_norm", "none")),
            str(mc.get("smolgen_relation_coeff_norm", "none")),
            str(mc.get("smolgen_relation_scale", "none")),
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

        model_cfg = ModelConfig(
            kind=config_key[0], embed_dim=config_key[1],
            num_layers=config_key[2], num_heads=config_key[3],
            ffn_mult=config_key[4], use_smolgen=config_key[5],
            use_nla=config_key[6], use_qk_rmsnorm=config_key[7],
            input_pos_encoding=config_key[8], use_deepnorm=config_key[9],
            policy_encoding=config_key[10],
            input_history_encoding=config_key[11],
            input_global_embedding=config_key[12],
            input_global_embedding_channels=config_key[13],
            input_square_embedding=config_key[14],
            qkv_projection=config_key[15],
            smolgen_mode=config_key[16],
            smolgen_bias_scale=config_key[17],
            smolgen_bias_norm=config_key[18],
            arc_attention_bias=config_key[19],
            smolgen_relation_basis=config_key[20],
            smolgen_relation_norm=config_key[21],
            smolgen_relation_coeff_norm=config_key[22],
            smolgen_relation_scale=config_key[23],
            use_gradient_checkpointing=False,
        )

        if trial_id not in self._trial_models:
            self._build_trial_model(
                trial_id, model_cfg=model_cfg, sd=sd, model_sha=model_sha,
            )
        else:
            self._update_trial_weights(trial_id, sd=sd, model_sha=model_sha)

        self._trial_shas[trial_id] = model_sha
        self._trial_manifest_sigs[trial_id] = sig
        return True

    def _process_parallel(self, ready_by_trial: dict[str, list[_InferenceSlot]]) -> None:
        """Process all trials' batches in parallel using per-trial CUDA streams."""
        use_cuda = self.device.startswith("cuda")

  # Prepare inputs for each trial
        trial_data: list[tuple[str, list[_InferenceSlot], list[int], torch.Tensor]] = []
        for trial_id, ready in ready_by_trial.items():
            model = self._trial_models.get(trial_id)
            if model is None:
                for slot in ready:
                    bsz = max(0, min(int(slot.batch_size), self._layout.max_batch))
                    slot.policy[:bsz].fill(0.0)
                    slot.wdl[:bsz].fill(0.0)
                    slot.state = _STATE_RESPONSE
                continue
            batch_sizes = [slot.batch_size for slot in ready]
            xs = [np.array(slot.input[:bsz], copy=True, order="C") for slot, bsz in zip(ready, batch_sizes)]
            xb = np.concatenate(xs, axis=0)
            xt = torch.from_numpy(xb).to(self.device, non_blocking=True)
            trial_data.append((trial_id, ready, batch_sizes, xt))

        if not trial_data:
            return

  # Launch forward passes in parallel on separate streams
        results: list[tuple[str, list[_InferenceSlot], list[int], torch.Tensor, torch.Tensor]] = []
        for trial_id, ready, batch_sizes, xt in trial_data:
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

            results.append((trial_id, ready, batch_sizes, pol, wdl))

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
        for trial_id, ready, batch_sizes, pol, wdl in results:
            pol_np = pol.numpy()
            wdl_np = wdl.numpy()
            start = 0
            for slot, bsz in zip(ready, batch_sizes):
                end = start + bsz
                slot.policy[:bsz] = pol_np[start:end]
                slot.wdl[:bsz] = wdl_np[start:end]
                slot.state = _STATE_RESPONSE
                start = end

    def serve_forever(self) -> None:
        _batch_count = 0
        _total_positions = 0
        _last_report = time.monotonic()
        _last_scan = 0.0
        _report_interval = 10.0
        _scan_interval = 5.0

        while not self._stop:
            now = time.monotonic()

  # Periodically scan for new trials
            if now - _last_scan >= _scan_interval:
                self._scan_trials()
  # Refresh weights for all known trials
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
                for _ in range(200):
                    if any(s.state == _STATE_REQUEST for _, s in self._all_slots):
                        break
                else:
                    time.sleep(0.00002)
                continue

  # Process all trials' batches in parallel on separate CUDA streams
            for ready in ready_by_trial.values():
                _batch_count += 1
                _total_positions += sum(s.batch_size for s in ready)
            self._process_parallel(ready_by_trial)

  # Periodic metrics
            now = time.monotonic()
            if now - _last_report >= _report_interval and _batch_count > 0:
                avg_pos = _total_positions / _batch_count
                n_trials = len(self._trial_slots)
                print(
                    f"[shared-broker] {_batch_count} batches in {now - _last_report:.1f}s | "
                    f"avg {avg_pos:.1f} pos/batch | "
                    f"{_total_positions / (now - _last_report):.0f} pos/s | "
                    f"{n_trials} trials",
                    flush=True,
                )
                _batch_count = 0
                _total_positions = 0
                _last_report = now

    def shutdown(self) -> None:
        self._stop = True
        for _, slot in self._all_slots:
            slot.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


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
        ap.add_argument("--shared-cache-dir", type=str, default=None)
        args = ap.parse_args()
    else:
        ap = argparse.ArgumentParser(description="Per-trial shared-memory inference broker")
        ap.add_argument("--publish-dir", type=str, required=True)
        ap.add_argument("--device", type=str, default="cuda")
        ap.add_argument("--compile-inference", action="store_true")
        ap.add_argument("--compile-mode", type=str, default="reduce-overhead")
        ap.add_argument("--batch-wait-ms", type=float, default=5.0)
        ap.add_argument("--num-slots", type=int, default=2)
        ap.add_argument("--max-batch-per-slot", type=int, default=256)
        ap.add_argument("--slot-prefix", type=str, required=True)
        ap.add_argument("--shared-cache-dir", type=str, default=None)
        args = ap.parse_args()
        args.mode = "per-trial"

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
        )
        try:
            broker.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            broker.shutdown()
        return 0

  # Per-trial mode
    broker = SlotBroker(
        publish_dir=Path(args.publish_dir).expanduser(),
        num_slots=int(args.num_slots),
        max_batch_per_slot=int(args.max_batch_per_slot),
        device=str(args.device),
        compile_inference=bool(args.compile_inference),
        batch_wait_ms=float(args.batch_wait_ms),
        slot_prefix=str(args.slot_prefix),
        compile_mode=str(args.compile_mode),
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
