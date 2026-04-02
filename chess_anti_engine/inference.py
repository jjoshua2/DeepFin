from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Protocol

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
    except Exception:
        pass


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
    ) -> tuple[torch.Tensor, torch.Tensor, "torch.cuda.Event | None"]:
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
            stream.wait_event(event_default)
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

    def evaluate_for_expand(
        self,
        x: np.ndarray,
        legal_indices: list[np.ndarray],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Forward pass that keeps policy on GPU and extracts priors there.

        Instead of transferring the full (batch, 4672) policy tensor to CPU,
        this gathers only the legal-move logits on GPU, computes softmax, and
        transfers the small prior vectors (~30 floats each) back.

        Parameters
        ----------
        x : encoded positions, shape (B, C, H, W)
        legal_indices : list of B int32 arrays, each containing legal move
            indices for the corresponding position.

        Returns
        -------
        priors_list : list of B float64 arrays, each softmaxed over legal moves
        wdl : (B, 3) float32 numpy array
        """
        xb = _coerce_input_batch(x)
        xt = torch.from_numpy(xb).to(self.device)
        with torch.no_grad():
            with inference_autocast(device=self.device, enabled=self._use_amp, dtype=self._amp_dtype):
                out = self.model(xt)
        policy_gpu = _policy_output(out)
        policy_gpu = policy_gpu.detach().float()
        wdl = out["wdl"].detach().to(dtype=torch.float32, device="cpu").numpy()

        # Extract priors on GPU: batched gather + softmax
        batch = xb.shape[0]
        if not self._use_cuda or batch == 0:
            # CPU fallback: just do it in numpy
            pol_cpu = policy_gpu.cpu().numpy()
            priors_list = []
            for i in range(batch):
                idx = legal_indices[i]
                if idx.size == 0:
                    priors_list.append(np.empty(0, dtype=np.float64))
                    continue
                ll = pol_cpu[i][idx].astype(np.float64)
                ll -= ll.max()
                e = np.exp(ll)
                s = float(e.sum())
                priors_list.append((e / s) if s > 0 else np.full_like(e, 1.0 / e.size))
            return priors_list, wdl

        # GPU path: pad legal indices, gather, softmax on GPU
        max_legal = max(idx.size for idx in legal_indices) if legal_indices else 0
        if max_legal == 0:
            return [np.empty(0, dtype=np.float64)] * batch, wdl

        # Build padded index tensor and mask on CPU, then send to GPU
        gather_idx = torch.zeros(batch, max_legal, dtype=torch.long)
        mask = torch.zeros(batch, max_legal, dtype=torch.bool)
        lengths = np.empty(batch, dtype=np.int32)
        for i in range(batch):
            idx = legal_indices[i]
            n = idx.size
            lengths[i] = n
            if n > 0:
                gather_idx[i, :n] = torch.from_numpy(idx.astype(np.int64))
                mask[i, :n] = True

        gather_idx = gather_idx.to(self.device)
        mask = mask.to(self.device)

        # Gather legal logits and apply masked softmax
        gathered = policy_gpu.gather(1, gather_idx)  # (batch, max_legal)
        gathered[~mask] = -1e30
        # Stable softmax
        gathered_max = gathered.max(dim=1, keepdim=True).values
        gathered = gathered - gathered_max
        gathered_exp = gathered.exp()
        gathered_exp[~mask] = 0.0
        gathered_sum = gathered_exp.sum(dim=1, keepdim=True).clamp(min=1e-12)
        priors_gpu = gathered_exp / gathered_sum  # (batch, max_legal)

        # Transfer only the small priors tensor to CPU
        priors_cpu = priors_gpu.to(dtype=torch.float64, device="cpu").numpy()

        # Split into per-position arrays (trimmed to actual legal count)
        priors_list = []
        for i in range(batch):
            n = int(lengths[i])
            if n > 0:
                priors_list.append(priors_cpu[i, :n].copy())
            else:
                priors_list.append(np.empty(0, dtype=np.float64))

        return priors_list, wdl


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

    def __init__(self, shm: SharedMemory, layout: _SlotLayout, *, owns: bool = False):
        self._shm = shm
        self._layout = layout
        self._owns = owns
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

    def _load_manifest(self) -> dict:
        mf = self.publish_dir / "manifest.json"
        return dict(json.loads(mf.read_text(encoding="utf-8")))

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
        if self.compile_inference and self.device.startswith("cuda"):
            model = torch.compile(model, mode="reduce-overhead")
        self._model = model
        self._model_sha = model_sha
        self._first_inference_pending = bool(self.compile_inference)

    # -- batch processing --

    def _process_batch(self, ready: list[_InferenceSlot]) -> None:
        self._ensure_model()
        if self._model is None:
            # No model yet — return explicit zero outputs so workers don't
            # consume stale shared-memory contents from a prior response.
            for slot in ready:
                bsz = max(0, min(int(slot.batch_size), self._layout.max_batch))
                slot.policy[:bsz].fill(0.0)
                slot.wdl[:bsz].fill(0.0)
                slot.state = _STATE_RESPONSE
            return

        # Gather inputs from all ready slots
        batch_sizes: list[int] = []
        xs: list[np.ndarray] = []
        for slot in ready:
            bsz = slot.batch_size
            batch_sizes.append(bsz)
            # Copy from shm to contiguous array (avoids stale-data issues
            # and ensures the tensor is backed by process-local memory).
            xs.append(np.array(slot.input[:bsz], copy=True, order="C"))

        xb = np.concatenate(xs, axis=0)
        xt = torch.from_numpy(xb).to(self.device)

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

        pol = _policy_output(out).detach().float().cpu().numpy()
        wdl = out["wdl"].detach().float().cpu().numpy()

        # Scatter outputs back to each slot
        start = 0
        for slot, bsz in zip(ready, batch_sizes):
            end = start + bsz
            slot.policy[:bsz] = pol[start:end]
            slot.wdl[:bsz] = wdl[start:end]
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
                state = slot.state
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
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-trial shared-memory inference broker"
    )
    ap.add_argument("--publish-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--compile-inference", action="store_true")
    ap.add_argument("--batch-wait-ms", type=float, default=5.0)
    ap.add_argument("--num-slots", type=int, default=2)
    ap.add_argument("--max-batch-per-slot", type=int, default=256)
    ap.add_argument("--slot-prefix", type=str, required=True)
    ap.add_argument("--shared-cache-dir", type=str, default=None)
    args = ap.parse_args()

    shared_cache_raw = str(args.shared_cache_dir or "").strip()
    if shared_cache_raw:
        _configure_compile_cache(Path(shared_cache_raw).expanduser())

    broker = SlotBroker(
        publish_dir=Path(args.publish_dir).expanduser(),
        num_slots=int(args.num_slots),
        max_batch_per_slot=int(args.max_batch_per_slot),
        device=str(args.device),
        compile_inference=bool(args.compile_inference),
        batch_wait_ms=float(args.batch_wait_ms),
        slot_prefix=str(args.slot_prefix),
    )

    # Write slot manifest so workers can discover slot names
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
