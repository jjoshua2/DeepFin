from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import queue
import struct
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace as dataclass_replace
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch

from chess_anti_engine.broker_hang import (
    DEFAULT_HANG_ABORT_S as _DEFAULT_HANG_ABORT_S,
    HANG_ABORT_ENV as _HANG_ABORT_ENV,
    BrokerHangWatchdog,
    resolve_hang_abort_seconds,
)
from chess_anti_engine.encoding import input_plane_count
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
        self._constant_fqns = list(next(iter(self._models.values())).get_constant_fqns())

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
        for model in self._models.values():
            model.load_constants(constants, check_full_update=False)

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

# Consecutive failed batches before the broker gives up and exits. High enough
# that a burst of client-timeout races rides through, low enough that a dead
# CUDA context does not livelock the fleet for an entire iteration.
_MAX_CONSECUTIVE_BATCH_FAILURES = 50


_HEADER_BYTES = 8  # 1 byte state + 1 byte mode + 2 pad + 4 byte batch_size


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
        self._hang_watchdog = BrokerHangWatchdog(threshold_s=float(hang_abort_seconds))
        self._consecutive_batch_failures = 0
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
            self._aot_models = load_aot_packages(aot_raw, buckets=aot_buckets)
            self._aot_constant_fqns = list(
                next(iter(self._aot_models.values())).get_constant_fqns()
            )
            print(
                f"[broker] AOT packages loaded n={len(self._aot_models)} "
                f"elapsed_s={time.perf_counter() - t_aot:.2f} "
                f"since_boot_s={time.perf_counter() - self._boot_t0:.2f}",
                flush=True,
            )

        # Pre-allocated pinned buffers for zero-copy GPU transfer.
        _total_cap = num_slots * max_batch_per_slot
        _pin = "cuda" in self.device and torch.cuda.is_available()
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
        _legal_metadata_cap = _total_cap * 256 if _pin else 0
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
            self._slots.append(slot)
            self._slot_names.append(name)

        print(
            f"[broker] slots ready n={len(self._slots)} "
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
        model_path = self.publish_dir / "latest_model.pt"
        ckpt = torch.load(str(model_path), map_location="cpu")
        sd = ckpt.get("model", ckpt)
        raw_model = build_model(model_cfg)
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
            for aot_model in self._aot_models.values():
                aot_model.load_constants(constants, check_full_update=False)

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

    def _process_batch(self, ready: list[_InferenceSlot]) -> None:
        by_mode: dict[int, list[_InferenceSlot]] = {}
        for slot in ready:
            by_mode.setdefault(int(slot.request_mode), []).append(slot)
        for mode, slots in by_mode.items():
            try:
                self._process_batch_mode(slots, mode=mode)
            except Exception:
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
        self._hang_watchdog.mark_forward_start(total)
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
            forward_ok = True
        finally:
            self._hang_watchdog.mark_forward_done(success=forward_ok)

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
                if not self._wait_for_ready_slots():
                    continue
                continue

            if _ARRIVAL_TRACE_ENABLED:
                self._note_arrival_trace(ready)

            self._gather_more_within_window(ready)

  # Re-collect in case some slots changed during the wait window.
            ready = [s for s in self._slots if s.state == _STATE_REQUEST]
            if ready:
                if _ARRIVAL_TRACE_ENABLED:
                    self._note_arrival_trace(ready)
                metrics["batches"] += 1
                metrics["positions"] += sum(s.batch_size for s in ready)
                metrics["slots"] += len(ready)
                self._timing_metrics = metrics
                # Emit before _process_batch so wait_ms is gather wait only
                # (forward duration would otherwise skew the diagnostic).
                if _ARRIVAL_TRACE_ENABLED:
                    self._emit_arrival_trace(ready)
                self._process_batch(ready)
                if not self._first_batch_logged:
                    self._first_batch_logged = True
                    print(
                        f"[broker] FIRST_BATCH_DONE positions="
                        f"{sum(s.batch_size for s in ready)} "
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

    def _disconnect(self) -> None:
        shm = self._shm
        self._slot = None
        self._shm = None
        if shm is not None:
            with contextlib.suppress(Exception):
                shm.close()

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
        input_planes: int = _CHANNELS,
    ) -> None:
        names = [str(n).strip() for n in slot_names if str(n).strip()]
        if not names:
            raise ValueError("MultiSlotInferenceClient requires at least one slot")
        self._request_timeout_s = max(0.001, float(request_timeout_s))
        self._clients = [
            SlotInferenceClient(
                slot_name=name,
                max_batch=max_batch,
                request_timeout_s=self._request_timeout_s,
                input_planes=input_planes,
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

    @property
    def supports_compact_root_policy(self) -> bool:
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
        """Take a free slot, or fail if none free within the request timeout.

        Unbounded ``queue.get()`` is a process-level hang risk: if every slot
        is held by a stuck request (or a leak), all selfplay threads park here
        forever and the per-request broker timeout never runs. Cap wait at the
        same budget as a single request so the worker can raise, reset, or
        exit via the session liveness watchdog.
        """
        t0 = time.perf_counter()
        # Allow a little more than one request so a just-busy slot can free.
        acquire_timeout_s = max(0.05, float(self._request_timeout_s) * 2.0)
        try:
            idx, client = self._available_clients.get(timeout=acquire_timeout_s)
        except queue.Empty as exc:
            with self._stats_lock:
                inflight = int(self._inflight)
                available = int(self._available_clients.qsize())
            raise TimeoutError(
                f"inference slot acquire timed out after {acquire_timeout_s:.1f}s "
                f"(slots={len(self._clients)} inflight={inflight} available={available}); "
                "all slots held — broker stall or slot leak"
            ) from exc
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
        input_planes: int = _CHANNELS,
        hang_abort_seconds: float = _DEFAULT_HANG_ABORT_S,
    ) -> None:
        self.server_root = Path(server_root)
        self.slots_per_trial = int(slots_per_trial)
        self.device = str(device)
        self.compile_inference = bool(compile_inference)
        self.compile_mode = str(compile_mode or "reduce-overhead")
        self.batch_wait_ms = float(batch_wait_ms)
        self._hang_watchdog = BrokerHangWatchdog(threshold_s=float(hang_abort_seconds))
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

    def _process_parallel(self, ready_by_trial: dict[str, list[_InferenceSlot]]) -> None:
        """Process all trials' batches in parallel using per-trial CUDA streams."""
        use_cuda = self.device.startswith("cuda")

        # Host-side pack first; H2D + forward sit under the hang watchdog.
        packed: list[tuple[str, list[_InferenceSlot], list[int], np.ndarray]] = []
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
            xs = [np.array(slot.input[:bsz], copy=True, order="C") for slot, bsz in zip(ready, batch_sizes, strict=True)]
            xb = np.concatenate(xs, axis=0)
            packed.append((trial_id, ready, batch_sizes, xb))

        if not packed:
            return

        hang_batch = sum(int(xb.shape[0]) for _, _, _, xb in packed)
        self._hang_watchdog.mark_forward_start(hang_batch)
        forward_ok = False
        try:
            trial_data: list[tuple[str, list[_InferenceSlot], list[int], torch.Tensor]] = [
                (trial_id, ready, batch_sizes, torch.from_numpy(xb).to(self.device, non_blocking=True))
                for trial_id, ready, batch_sizes, xb in packed
            ]
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
            for _trial_id, ready, batch_sizes, pol, wdl in results:
                pol_np = pol.numpy()
                wdl_np = wdl.numpy()
                start = 0
                for slot, bsz in zip(ready, batch_sizes, strict=True):
                    end = start + bsz
                    slot.policy[:bsz] = pol_np[start:end]
                    slot.wdl[:bsz] = wdl_np[start:end]
                    slot.state = _STATE_RESPONSE
                    start = end
            forward_ok = True
        finally:
            self._hang_watchdog.mark_forward_done(success=forward_ok)

    def serve_forever(self) -> None:
        _batch_count = 0
        _total_positions = 0
        _last_report = time.monotonic()
        _last_scan = 0.0
        _report_interval = 10.0
        _scan_interval = 5.0
        self._hang_watchdog.start()

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
            "Hard-exit (code 42) if a GPU forward stays in flight longer than this "
            "after the first successful batch. 0 disables. Env "
            f"{_HANG_ABORT_ENV} overrides when set. Default: {_DEFAULT_HANG_ABORT_S:g}."
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
