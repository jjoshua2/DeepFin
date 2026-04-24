from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import struct
import threading
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch

from chess_anti_engine.model import ModelConfig, build_model, load_state_dict_tolerant
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
    arr = np.ascontiguousarray(x, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"expected encoded batch shape (B,C,H,W), got {arr.shape!r}")
    return arr


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
        with torch.no_grad():
            with inference_autocast(device=self.device, enabled=self._use_amp, dtype=self._amp_dtype):
                out = self.model(xt)
        policy_out = _policy_output(out)
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
            with torch.no_grad():
                with inference_autocast(device=self.device, enabled=self._use_amp, dtype=self._amp_dtype):
                    out = self.model(xt)
            policy_out = _policy_output(out)
            pol = policy_out.detach().float()
            wdl = out["wdl"].detach().float()
            return pol, wdl, None

        stream = getattr(self, "_stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=self.device)
            self._stream = stream

  # Default stream must finish any prior work before we branch
        event_default = torch.cuda.Event()
        event_default.record(torch.cuda.current_stream(self.device))

        with torch.cuda.stream(stream):
            stream.wait_event(event_default) # torch stubs have duplicate Event types
            xt = torch.from_numpy(xb).to(self.device, non_blocking=True)
            with torch.no_grad():
                with inference_autocast(device=self.device, enabled=self._use_amp, dtype=self._amp_dtype):
                    out = self.model(xt)
            policy_out = _policy_output(out)
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
    ) -> None:
        super().__init__(model, device=device, use_amp=use_amp, amp_dtype=amp_dtype)
        self._max_batch = int(max_batch)

        _pin = self._use_cuda
        self._pinned_input = torch.empty(
            (self._max_batch, _CHANNELS, _BOARD_H, _BOARD_W),
            dtype=torch.float32, pin_memory=_pin,
        )
        self._pinned_pol = torch.empty(
            (self._max_batch, _POLICY_SIZE),
            dtype=torch.float32, pin_memory=_pin,
        )
        self._pinned_wdl = torch.empty(
            (self._max_batch, _WDL_SIZE),
            dtype=torch.float32, pin_memory=_pin,
        )
        self._pinned_input_np = self._pinned_input.numpy(force=True)
        self._pinned_pol_np = self._pinned_pol.numpy(force=True)
        self._pinned_wdl_np = self._pinned_wdl.numpy(force=True)

    def get_input_buffer(self, bsz: int) -> np.ndarray:
        """Return a writable (bsz, C, H, W) view into the pinned input buffer.

        Caller can write encoded positions directly into this buffer,
        avoiding a separate allocation + copy.  The view is valid until
        the next ``evaluate_encoded`` or ``get_input_buffer`` call.
        """
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")
        return self._pinned_input_np[:bsz]

    def evaluate_inplace(
        self, bsz: int, *, copy_out: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on data already written to ``get_input_buffer()``.

        Same as ``evaluate_encoded`` but skips the input copy — caller must
        have already written ``bsz`` positions into the pinned input buffer.
        """
        if bsz <= 0:
            return np.empty((0, _POLICY_SIZE), dtype=np.float32), np.empty((0, _WDL_SIZE), dtype=np.float32)
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")
        return self._run_forward(bsz, copy_out=copy_out)

    def evaluate_encoded(
        self, x: np.ndarray, *, copy_out: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if x.ndim != 4:
            raise ValueError(f"expected 4D input, got {x.ndim}D")
        bsz = x.shape[0]
        if bsz > self._max_batch:
            raise ValueError(f"batch {bsz} > max {self._max_batch}")

        self._pinned_input_np[:bsz] = x
        return self._run_forward(bsz, copy_out=copy_out)

    def _run_forward(
        self, bsz: int, *, copy_out: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._use_cuda:
            xt = self._pinned_input[:bsz]
            with torch.no_grad():
                with inference_autocast(device=self.device, enabled=self._use_amp, dtype=self._amp_dtype):
                    out = self.model(xt)
            return _policy_output(out).detach().float().numpy(), out["wdl"].detach().float().numpy()

        xt = self._pinned_input[:bsz].to(self.device, non_blocking=True)
        with torch.no_grad():
            with inference_autocast(device=self.device, enabled=self._use_amp, dtype=self._amp_dtype):
                out = self.model(xt)
        self._pinned_pol[:bsz].copy_(_policy_output(out).detach().float(), non_blocking=True)
        self._pinned_wdl[:bsz].copy_(out["wdl"].detach().float(), non_blocking=True)
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(torch.device(self.device)))
        done.synchronize()

        pol_np = self._pinned_pol_np[:bsz]
        wdl_np = self._pinned_wdl_np[:bsz]
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

        self._pinned_input_np[:bsz] = x

        stream = getattr(self, "_stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=self.device)
            self._stream = stream

        event_default = torch.cuda.Event()
        event_default.record(torch.cuda.current_stream(self.device))

        with torch.cuda.stream(stream):
            stream.wait_event(event_default) # torch stubs have duplicate Event types
            xt = self._pinned_input[:bsz].to(self.device, non_blocking=True)
            with torch.no_grad():
                with inference_autocast(device=self.device, enabled=self._use_amp, dtype=self._amp_dtype):
                    out = self.model(xt)
            self._pinned_pol[:bsz].copy_(_policy_output(out).detach().float(), non_blocking=True)
            self._pinned_wdl[:bsz].copy_(out["wdl"].detach().float(), non_blocking=True)
            done = torch.cuda.Event()
            done.record(stream)

        return self._pinned_pol[:bsz], self._pinned_wdl[:bsz], done


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
  # Wait for first request
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if first is None:
                break

  # Handle model update (sentinel: string tag, not numpy array)
            if isinstance(first, tuple) and len(first) == 3 and isinstance(first[0], str):
                _, new_model, event = first
                self._gpu_eval.model = new_model
                event.set()
                continue

            pending = [first]
            total = first[0].shape[0]

  # Accumulate more requests up to min_batch or timeout (cap at max_batch)
            deadline = time.monotonic() + self._timeout
            while total < self._min_batch and total < self._max_batch and time.monotonic() < deadline:
                try:
                    item = self._queue.get(timeout=max(0.0001, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if item is None:
                    break
                if isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], str):
  # Model update — put it back and break so pending requests
  # get processed with the current model first.
                    self._queue.put(item)
                    break
                item_size = item[0].shape[0]
                if total + item_size > self._max_batch:
  # Would exceed max — put back for next round
                    self._queue.put(item)
                    break
                pending.append(item)
                total += item_size

            if not pending:
                continue

  # Concatenate all inputs
            if len(pending) == 1:
                combined = pending[0][0]
            else:
                combined = np.concatenate([p[0] for p in pending], axis=0)

  # Bucket-pad to reduce CUDA graph count
            _BUCKETS = (128, 256, 384, 512, 768, 1024, 1536, 2048, 4096)
            padded_size = total
            for b in _BUCKETS:
                if b >= total:
                    padded_size = min(b, self._max_batch)
                    break
            if padded_size > total:
                pad = np.zeros((padded_size - total, *combined.shape[1:]), dtype=combined.dtype)
                combined = np.concatenate([combined, pad], axis=0)

  # Single GPU call — catch any error and propagate to all waiting threads
            try:
                pol_all, wdl_all = self._gpu_eval.evaluate_encoded(combined, copy_out=False)
                pol_all = pol_all[:total]
                wdl_all = wdl_all[:total]
            except Exception as exc:
                for _x, event, result in pending:  # skylos: ignore (_x unpacked but unused by convention)
                    result["error"] = str(exc)
                    event.set()
                continue
            except BaseException as exc:
  # Fatal (KeyboardInterrupt, SystemExit) — unblock all threads
                for _x, event, result in pending:  # skylos: ignore (_x unpacked but unused by convention)
                    result["error"] = str(exc)
                    event.set()
                raise

  # Scatter results back to each thread
            offset = 0
            for x, event, result in pending:
                n = x.shape[0]
                result["pol"] = pol_all[offset:offset + n].copy()
                result["wdl"] = wdl_all[offset:offset + n].copy()
                offset += n
                event.set()


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

        pol = _policy_output(out)[:bsz].detach().float().cpu().numpy()
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

        pol = _policy_output(out)[:bsz].detach().to(
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
#   [4:8]      batch_size int32   (number of positions in this request)
#   [8:...]    input      float32[max_batch, 146, 8, 8]
#   [after input]  policy float32[max_batch, 4672]
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

_CHANNELS = 146
_BOARD_H = 8
_BOARD_W = 8
_POLICY_SIZE = 4672
_WDL_SIZE = 3
_F32 = np.float32
_F32_BYTES = 4

_HEADER_BYTES = 8  # 1 byte state + 3 pad + 4 byte batch_size


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

    __slots__ = ("_shm", "_layout", "_owns", "_buf", "input", "policy", "wdl")
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
        self.policy: np.ndarray = np.ndarray(
            (layout.max_batch, _POLICY_SIZE),
            dtype=_F32,
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
    ) -> None:
        self.publish_dir = Path(publish_dir)
        self.device = str(device)
        self.compile_inference = bool(compile_inference)
        self._first_inference_pending = False
        self.batch_wait_ms = float(batch_wait_ms)
        self._model: torch.nn.Module | None = None
        self._model_sha: str | None = None
        self._stop = False
        self._manifest_cache: dict | None = None
        self._manifest_cache_sig: tuple[int, int] | None = None

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
        self._pinned_pol = torch.empty(
            (_total_cap, _POLICY_SIZE), dtype=torch.float32, pin_memory=_pin,
        )
        self._pinned_wdl = torch.empty(
            (_total_cap, _WDL_SIZE), dtype=torch.float32, pin_memory=_pin,
        )
  # Pinned tensors need force=True for numpy conversion.
        self._pinned_input_np = self._pinned_input.numpy(force=True)
        self._pinned_pol_np = self._pinned_pol.numpy(force=True)
        self._pinned_wdl_np = self._pinned_wdl.numpy(force=True)

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

        mc = manifest.get("model_config") or {}
        model_cfg = ModelConfig(
            kind=str(mc.get("kind", "transformer")),
            embed_dim=int(mc.get("embed_dim", 256)),
            num_layers=int(mc.get("num_layers", 6)),
            num_heads=int(mc.get("num_heads", 8)),
            ffn_mult=float(mc.get("ffn_mult", 2)),
            use_smolgen=bool(mc.get("use_smolgen", True)),
            use_nla=bool(mc.get("use_nla", False)),
            use_qk_rmsnorm=bool(mc.get("use_qk_rmsnorm", False)),
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
            model = cast("torch.nn.Module", torch.compile(model, mode="reduce-overhead"))
        self._model = model
        self._model_sha = model_sha
        self._first_inference_pending = bool(self.compile_inference)

  # -- batch processing --

    def _process_batch(self, ready: list[_InferenceSlot]) -> None:
        self._ensure_model()
        if self._model is None:
            for slot in ready:
                bsz = max(0, min(int(slot.batch_size), self._layout.max_batch))
                slot.policy[:bsz].fill(0.0)
                slot.wdl[:bsz].fill(0.0)
                slot.state = _STATE_RESPONSE
            return

  # Gather inputs directly into pre-allocated pinned buffer (one memcpy
  # from shm → pinned, then async DMA to GPU — no intermediate allocs).
        batch_sizes: list[int] = []
        total = 0
        for slot in ready:
            bsz = max(0, min(int(slot.batch_size), self._layout.max_batch))
            batch_sizes.append(bsz)
            self._pinned_input_np[total:total + bsz] = slot.input[:bsz]
            total += bsz

        assert total <= self._pinned_input.shape[0], (
            f"gather overflow: {total} > {self._pinned_input.shape[0]}"
        )
        xt = self._pinned_input[:total].to(self.device, non_blocking=True)

        first_inf = self._first_inference_pending
        if first_inf:
            inf_t0 = time.time()

        with torch.no_grad():
            with inference_autocast(device=self.device, enabled=True, dtype="auto"):
                out = self._model(xt)

        if first_inf:
            log.info("first inference (includes kernel compile) elapsed_s=%.2f batch=%d",
                     time.time() - inf_t0, xt.shape[0])
            self._first_inference_pending = False

  # Copy results to pinned buffer (async GPU→pinned), then to shm.
        pol_gpu = _policy_output(out).detach().float()
        wdl_gpu = out["wdl"].detach().float()
        self._pinned_pol[:total].copy_(pol_gpu, non_blocking=True)
        self._pinned_wdl[:total].copy_(wdl_gpu, non_blocking=True)
        if self.device.startswith("cuda"):
            torch.cuda.current_stream(torch.device(self.device)).synchronize()

  # Scatter from pinned buffer to worker slots
        start = 0
        for slot, bsz in zip(ready, batch_sizes):
            end = start + bsz
            slot.policy[:bsz] = self._pinned_pol_np[start:end]
            slot.wdl[:bsz] = self._pinned_wdl_np[start:end]
            slot.state = _STATE_RESPONSE
            start = end

  # -- main loop --

    def serve_forever(self) -> None:
  # Batch size metrics (printed periodically)
        _batch_count = 0
        _total_positions = 0
        _total_slots_used = 0
        _last_report = time.monotonic()
        _report_interval = 10.0  # seconds

        while not self._stop:
            if any(s.state == _STATE_SHUTDOWN for s in self._slots):
                self._stop = True
                break

  # Scan for ready slots
            ready = [s for s in self._slots if s.state == _STATE_REQUEST]

            if not ready:
  # Tight spin with occasional yield to avoid burning 100% of
  # one core while keeping latency minimal.
                for _ in range(200):
                    if any(s.state == _STATE_REQUEST for s in self._slots):
                        break
                else:
                    time.sleep(0.00002)  # 20µs yield
                continue

  # Batching window: wait briefly for more slots to become ready
            if self.batch_wait_ms > 0:
                deadline = time.monotonic() + (self.batch_wait_ms / 1000.0)
                while time.monotonic() < deadline:
                    if all(
                        s.state in (_STATE_REQUEST, _STATE_RESPONSE, _STATE_SHUTDOWN)
                        for s in self._slots
                    ):
                        break  # all slots submitted or already served
                    more = [
                        s
                        for s in self._slots
                        if s.state == _STATE_REQUEST and s not in ready
                    ]
                    if more:
                        ready.extend(more)
                    if len(ready) >= len(self._slots):
                        break
                    time.sleep(0.0001)

  # Re-collect in case some changed during the wait
            ready = [s for s in self._slots if s.state == _STATE_REQUEST]
            if ready:
                total_pos = sum(s.batch_size for s in ready)
                _batch_count += 1
                _total_positions += total_pos
                _total_slots_used += len(ready)
                self._process_batch(ready)

  # Periodic metrics
            now = time.monotonic()
            if now - _last_report >= _report_interval and _batch_count > 0:
                avg_pos = _total_positions / _batch_count
                avg_slots = _total_slots_used / _batch_count
                print(
                    f"[broker] {_batch_count} batches in {now - _last_report:.1f}s | "
                    f"avg {avg_pos:.1f} pos/batch, {avg_slots:.1f} slots/batch | "
                    f"{_total_positions / (now - _last_report):.0f} pos/s",
                    flush=True,
                )
                _batch_count = 0
                _total_positions = 0
                _total_slots_used = 0
                _last_report = now

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

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xb = _coerce_input_batch(x)
        bsz = xb.shape[0]
        if bsz > self._layout.max_batch:
            raise ValueError(
                f"batch size {bsz} exceeds slot max {self._layout.max_batch}"
            )

        deadline = time.monotonic() + self._request_timeout_s
        last_timeout = False
        while True:
            slot = self._connect(deadline=deadline)

  # Write input directly into shared memory (one memcpy)
            slot.input[:bsz] = xb
            slot.batch_size = bsz
            slot.state = _STATE_REQUEST

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
                    pol = np.array(slot.policy[:bsz], copy=True, order="C")
                    wdl = np.array(slot.wdl[:bsz], copy=True, order="C")
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
    ) -> None:
        self.server_root = Path(server_root)
        self.slots_per_trial = int(slots_per_trial)
        self.device = str(device)
        self.compile_inference = bool(compile_inference)
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
            publish_dir = trial_dir / "publish"
            manifest = publish_dir / "manifest.json"
            if not manifest.exists():
                continue

  # Create slots for this trial
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

  # Clean up trials that no longer have a publish dir
        stale = [tid for tid in self._trial_slots
                 if not (trials_root / tid / "publish" / "manifest.json").exists()
                 or (active_prefix and not tid.startswith(active_prefix))]
        for tid in stale:
            for slot in self._trial_slots[tid]:
                slot.close()
            self._all_slots = [(t, s) for t, s in self._all_slots if t != tid]
            del self._trial_slots[tid]
            self._trial_models.pop(tid, None)
            self._trial_streams.pop(tid, None)
            self._trial_shas.pop(tid, None)
            self._trial_manifest_sigs.pop(tid, None)
            print(f"[shared-broker] deregistered stale trial {tid}", flush=True)

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

  # Validate config matches
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
        )
        if self._model_config_key is not None and config_key != self._model_config_key:
            print(f"[shared-broker] WARNING: trial {trial_id} has different model config, skipping", flush=True)
            return False
        if self._model_config_key is None:
            self._model_config_key = config_key

  # Load checkpoint
        model_path = publish_dir / "latest_model.pt"
        try:
            ckpt = torch.load(str(model_path), map_location="cpu")
            sd = ckpt.get("model", ckpt)
        except (OSError, RuntimeError, EOFError):
            return False  # checkpoint missing, mid-write, or torch load failed

  # Build or update model instance for this trial
        model_cfg = ModelConfig(
            kind=config_key[0], embed_dim=config_key[1],
            num_layers=config_key[2], num_heads=config_key[3],
            ffn_mult=config_key[4], use_smolgen=config_key[5],
            use_nla=config_key[6], use_qk_rmsnorm=config_key[7],
            use_gradient_checkpointing=False,
        )

        if trial_id not in self._trial_models:
  # New trial: build fresh model instance
            model = build_model(model_cfg)
            model.to(self.device)
            model.eval()
            if hasattr(model, "_inference_only"):
                setattr(model, "_inference_only", True)
            load_state_dict_tolerant(model, sd, label=f"shared-broker-{trial_id}")
            if self.compile_inference and self.device.startswith("cuda"):
                model = cast("torch.nn.Module", torch.compile(model, mode="reduce-overhead"))
            self._trial_models[trial_id] = model
            if self.device.startswith("cuda"):
                self._trial_streams[trial_id] = torch.cuda.Stream(device=self.device)
            print(f"[shared-broker] created model for trial {trial_id} (sha={model_sha[:8]})", flush=True)
        else:
  # Existing trial: update weights
            model = self._trial_models[trial_id]
            target = getattr(model, "_orig_mod", model)
            load_state_dict_tolerant(target, sd, label=f"shared-broker-{trial_id}")
            print(f"[shared-broker] updated weights for trial {trial_id} (sha={model_sha[:8]})", flush=True)

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
                    with torch.no_grad():
                        with inference_autocast(device=self.device, enabled=True, dtype="auto"):
                            out = model(xt)
                    pol = _policy_output(out).detach().float().to("cpu", non_blocking=True)
                    wdl = out["wdl"].detach().float().to("cpu", non_blocking=True)
            else:
                with torch.no_grad():
                    with inference_autocast(device=self.device, enabled=True, dtype="auto"):
                        out = model(xt)
                pol = _policy_output(out).detach().float().cpu()
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
