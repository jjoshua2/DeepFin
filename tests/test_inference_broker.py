from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import pytest
import torch

from chess_anti_engine.inference import (
    _MODE_DENSE_BF16,
    _MODE_LEGAL_BF16,
    _STATE_REQUEST,
    _STATE_RESPONSE,
    _STATE_SHUTDOWN,
    LocalModelEvaluator,
    SlotBroker,
    SlotInferenceClient,
    _InferenceSlot,
    _SlotLayout,
)
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.utils import sha256_file as _sha256_file


def test_slot_inference_client_sends_dense_bf16_bits() -> None:
    slot_name = f"cae-bf16-{uuid.uuid4().hex}"
    layout = _SlotLayout.compute(8)
    shm = SharedMemory(name=slot_name, create=True, size=layout.total_bytes)
    done = threading.Event()
    x = np.arange(2 * 146 * 8 * 8, dtype=np.uint16).reshape(2, 146, 8, 8)

    def _serve_once() -> None:
        try:
            slot = _InferenceSlot(shm, layout, owns=False)
            slot.state = 0
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if int(slot.state) == _STATE_REQUEST:
                    assert slot.request_mode == _MODE_DENSE_BF16
                    assert slot.batch_size == 2
                    assert np.array_equal(slot.input_bf16_bits[:2], x)
                    slot.policy[:2].fill(3.0)
                    slot.wdl[:2].fill(4.0)
                    slot.state = _STATE_RESPONSE
                    done.set()
                    return
                time.sleep(0.001)
        finally:
            shm.close()

    t = threading.Thread(target=_serve_once, daemon=True)
    t.start()
    client = SlotInferenceClient(slot_name=slot_name, max_batch=8, request_timeout_s=1.0)
    try:
        pol, wdl = client.evaluate_encoded(x)
    finally:
        client.close()
        t.join(timeout=1.0)
        shm.unlink()

    assert done.is_set()
    assert np.allclose(pol, 3.0)
    assert np.allclose(wdl, 4.0)


def test_slot_inference_client_sends_compact_legal_bf16_request() -> None:
    slot_name = f"cae-legal-bf16-{uuid.uuid4().hex}"
    layout = _SlotLayout.compute(8)
    shm = SharedMemory(name=slot_name, create=True, size=layout.total_bytes)
    done = threading.Event()
    x = np.arange(2 * 146 * 8 * 8, dtype=np.uint16).reshape(2, 146, 8, 8)
    legal_counts = np.array([2, 1], dtype=np.int32)
    legal_flat = np.array([0, 5, 4671], dtype=np.int32)
    compact_bits = np.array([11, 22, 33], dtype=np.uint16)

    def _serve_once() -> None:
        try:
            slot = _InferenceSlot(shm, layout, owns=False)
            slot.state = 0
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if int(slot.state) == _STATE_REQUEST:
                    assert slot.request_mode == _MODE_LEGAL_BF16
                    assert slot.batch_size == 2
                    assert np.array_equal(slot.input_bf16_bits[:2], x)
                    assert int(slot.policy_i32[0]) == int(legal_flat.shape[0])
                    assert np.array_equal(slot.policy_i32[1:3], legal_counts)
                    assert np.array_equal(slot.policy_i32[3:6], legal_flat)
                    slot.policy_u16[:3] = compact_bits
                    slot.wdl[:2].fill(7.0)
                    slot.state = _STATE_RESPONSE
                    done.set()
                    return
                time.sleep(0.001)
        finally:
            shm.close()

    t = threading.Thread(target=_serve_once, daemon=True)
    t.start()
    client = SlotInferenceClient(slot_name=slot_name, max_batch=8, request_timeout_s=1.0)
    try:
        pol, wdl = client.evaluate_legal_bf16(x, legal_flat, legal_counts)
    finally:
        client.close()
        t.join(timeout=1.0)
        shm.unlink()

    assert done.is_set()
    assert np.array_equal(pol, compact_bits)
    assert np.allclose(wdl, 7.0)


def test_slot_broker_returns_compact_legal_bf16_logits(tmp_path: Path) -> None:
    class _TinyPolicy(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            bsz = x.shape[0]
            row = torch.arange(bsz, dtype=torch.float32).unsqueeze(1) * 10000.0
            return {
                "policy": row + torch.arange(4672, dtype=torch.float32).unsqueeze(0),
                "wdl": torch.arange(bsz * 3, dtype=torch.float32).reshape(bsz, 3),
            }

    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    broker = SlotBroker(
        publish_dir=publish_dir,
        num_slots=1,
        max_batch_per_slot=8,
        device="cpu",
        compile_inference=False,
        batch_wait_ms=0.0,
        slot_prefix=f"cae-legal-broker-{uuid.uuid4().hex}",
    )
    try:
        broker._model = _TinyPolicy().eval()
        broker._model_sha = "test"
        slot = broker._slots[0]
        x = np.zeros((2, 146, 8, 8), dtype=np.uint16)
        legal_counts = np.array([2, 2], dtype=np.int32)
        legal_flat = np.array([0, 5, 7, 4671], dtype=np.int32)
        slot.input_bf16_bits[:2] = x
        slot.policy_i32[0] = int(legal_flat.shape[0])
        slot.policy_i32[1:3] = legal_counts
        slot.policy_i32[3:7] = legal_flat
        slot.batch_size = 2
        slot.request_mode = _MODE_LEGAL_BF16
        slot.state = _STATE_REQUEST

        broker._process_batch([slot])

        expected = torch.tensor([0.0, 5.0, 10007.0, 14671.0]).to(torch.bfloat16).view(torch.uint16).numpy()
        assert slot.state == _STATE_RESPONSE
        assert slot.request_mode == 0
        assert np.array_equal(slot.policy_u16[:4], expected)
        assert np.allclose(slot.wdl[:2], [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    finally:
        broker.shutdown()


def test_slot_broker_uses_model_legal_policy_forward(tmp_path: Path) -> None:
    class _TinyLegalPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.called_legal = False

        def forward(self, _x: torch.Tensor):
            raise AssertionError("dense policy forward should not be used")

        def forward_legal_policy(
            self,
            x: torch.Tensor,
            legal_flat: torch.Tensor,
            legal_counts: torch.Tensor,
        ):
            del legal_counts
            self.called_legal = True
            bsz = x.shape[0]
            return {
                "policy_own": legal_flat.to(torch.float32) + 100.0,
                "wdl": torch.arange(bsz * 3, dtype=torch.float32).reshape(bsz, 3),
            }

    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    broker = SlotBroker(
        publish_dir=publish_dir,
        num_slots=1,
        max_batch_per_slot=8,
        device="cpu",
        compile_inference=False,
        batch_wait_ms=0.0,
        slot_prefix=f"cae-legal-forward-{uuid.uuid4().hex}",
    )
    try:
        model = _TinyLegalPolicy().eval()
        broker._model = model
        broker._model_sha = "test"
        slot = broker._slots[0]
        legal_counts = np.array([2, 2], dtype=np.int32)
        legal_flat = np.array([0, 5, 7, 4671], dtype=np.int32)
        slot.input_bf16_bits[:2] = np.zeros((2, 146, 8, 8), dtype=np.uint16)
        slot.policy_i32[0] = int(legal_flat.shape[0])
        slot.policy_i32[1:3] = legal_counts
        slot.policy_i32[3:7] = legal_flat
        slot.batch_size = 2
        slot.request_mode = _MODE_LEGAL_BF16
        slot.state = _STATE_REQUEST

        broker._process_batch([slot])

        expected = torch.tensor([100.0, 105.0, 107.0, 4771.0]).to(torch.bfloat16).view(torch.uint16).numpy()
        assert model.called_legal
        assert slot.state == _STATE_RESPONSE
        assert np.array_equal(slot.policy_u16[:4], expected)
        assert np.allclose(slot.wdl[:2], [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    finally:
        broker.shutdown()


def test_slot_broker_uses_padded_legal_rows_forward(tmp_path: Path) -> None:
    class _TinyLegalRowsPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_legal_shape: tuple[int, int] | None = None

        def forward(self, _x: torch.Tensor):
            raise AssertionError("dense policy forward should not be used")

        def forward_legal_policy_rows(
            self,
            x: torch.Tensor,
            legal_flat: torch.Tensor,
            legal_rows: torch.Tensor,
        ):
            self.seen_legal_shape = (int(legal_flat.shape[0]), int(legal_rows.shape[0]))
            bsz = x.shape[0]
            return {
                "policy_own": legal_flat.to(torch.float32) + legal_rows.to(torch.float32) * 10000.0,
                "wdl": torch.arange(bsz * 3, dtype=torch.float32).reshape(bsz, 3),
            }

    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    broker = SlotBroker(
        publish_dir=publish_dir,
        num_slots=1,
        max_batch_per_slot=256,
        device="cpu",
        compile_inference=True,
        batch_wait_ms=0.0,
        slot_prefix=f"cae-legal-rows-{uuid.uuid4().hex}",
    )
    try:
        model = _TinyLegalRowsPolicy().eval()
        broker._model = model
        broker._model_sha = "test"
        slot = broker._slots[0]
        legal_counts = np.array([2, 2], dtype=np.int32)
        legal_flat = np.array([0, 5, 7, 4671], dtype=np.int32)
        slot.input_bf16_bits[:2] = np.zeros((2, 146, 8, 8), dtype=np.uint16)
        slot.policy_i32[0] = int(legal_flat.shape[0])
        slot.policy_i32[1:3] = legal_counts
        slot.policy_i32[3:7] = legal_flat
        slot.batch_size = 2
        slot.request_mode = _MODE_LEGAL_BF16
        slot.state = _STATE_REQUEST

        broker._process_batch([slot])

        expected = torch.tensor([0.0, 5.0, 10007.0, 14671.0]).to(torch.bfloat16).view(torch.uint16).numpy()
        assert model.seen_legal_shape == (32768, 32768)
        assert slot.state == _STATE_RESPONSE
        assert np.array_equal(slot.policy_u16[:4], expected)
        assert np.allclose(slot.wdl[:2], [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    finally:
        broker.shutdown()


def test_slot_inference_broker_roundtrip(tmp_path: Path) -> None:
    """Test that the slot-based broker produces the same results as local eval."""
    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    model_path = publish_dir / "latest_model.pt"

    model_cfg = ModelConfig(
        kind="tiny",
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        ffn_mult=2,
        use_smolgen=False,
        use_nla=False,
    )
    model = build_model(model_cfg).eval()
    torch.save({"model": model.state_dict()}, model_path)
    manifest = {
        "model": {
            "sha256": _sha256_file(model_path),
            "filename": "latest_model.pt",
        },
        "model_config": {
            "kind": model_cfg.kind,
            "embed_dim": model_cfg.embed_dim,
            "num_layers": model_cfg.num_layers,
            "num_heads": model_cfg.num_heads,
            "ffn_mult": model_cfg.ffn_mult,
            "use_smolgen": model_cfg.use_smolgen,
            "use_nla": model_cfg.use_nla,
            "use_qk_rmsnorm": model_cfg.use_qk_rmsnorm,
            "gradient_checkpointing": False,
        },
    }
    (publish_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    slot_prefix = "cae-test-slot"
    max_batch = 64

    proc = subprocess.Popen(  # long-lived broker subprocess, explicit terminate in finally
        [
            sys.executable,
            "-m",
            "chess_anti_engine.inference",
            "--publish-dir",
            str(publish_dir),
            "--slot-prefix",
            slot_prefix,
            "--num-slots",
            "1",
            "--max-batch-per-slot",
            str(max_batch),
            "--device",
            "cpu",
            "--batch-wait-ms",
            "1.0",
        ]
    )
    try:
        # Wait for broker to start and create shared memory slots
        from multiprocessing.shared_memory import SharedMemory
        slot_name = f"{slot_prefix}-0"
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"broker exited early with code {proc.returncode}")
            try:
                probe = SharedMemory(name=slot_name, create=False)
                probe.close()
                break
            except FileNotFoundError:
                time.sleep(0.1)
        else:
            raise RuntimeError("broker did not create shared memory in time")

        x = np.random.default_rng(0).normal(size=(3, 146, 8, 8)).astype(np.float32)
        local_pol, local_wdl = LocalModelEvaluator(model, device="cpu").evaluate_encoded(x)

        client = SlotInferenceClient(slot_name=slot_name, max_batch=max_batch)
        try:
            remote_pol, remote_wdl = client.evaluate_encoded(x)
        finally:
            client.close()

        assert np.allclose(remote_pol, local_pol, atol=1e-6)
        assert np.allclose(remote_wdl, local_wdl, atol=1e-6)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except Exception:
            proc.kill()


def test_slot_inference_client_times_out_if_broker_never_responds() -> None:
    slot_name = f"cae-timeout-{uuid.uuid4().hex}"
    shm = SharedMemory(name=slot_name, create=True, size=64 * 146 * 8 * 8 * 4 + 64 * 4672 * 4 + 64 * 3 * 4 + 8)
    try:
        from chess_anti_engine.inference import (
            _STATE_REQUEST,
            _InferenceSlot,
            _SlotLayout,
        )

        slot = _InferenceSlot(shm, _SlotLayout.compute(64), owns=False)
        slot.state = _STATE_REQUEST
        client = SlotInferenceClient(slot_name=slot_name, max_batch=64, request_timeout_s=0.01)
        try:
            x = np.zeros((1, 146, 8, 8), dtype=np.float32)
            with pytest.raises(TimeoutError):
                client.evaluate_encoded(x)
        finally:
            client.close()
    finally:
        shm.close()
        shm.unlink()


def test_slot_inference_client_serializes_threaded_requests() -> None:
    client = SlotInferenceClient(
        slot_name=f"cae-lock-{uuid.uuid4().hex}",
        max_batch=8,
        request_timeout_s=0.1,
    )
    active = 0
    max_active = 0
    calls = 0
    guard = threading.Lock()

    def _fake_eval(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal active, max_active, calls
        with guard:
            active += 1
            calls += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with guard:
            active -= 1
        return (
            np.zeros((x.shape[0], 4672), dtype=np.float32),
            np.zeros((x.shape[0], 3), dtype=np.float32),
        )

    client._evaluate_encoded_locked = _fake_eval  # type: ignore[method-assign]

    x = np.zeros((1, 146, 8, 8), dtype=np.float32)
    threads = [threading.Thread(target=client.evaluate_encoded, args=(x,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=1.0)

    assert calls == len(threads)
    assert max_active == 1


def test_multi_slot_inference_client_fans_out_requests() -> None:
    from chess_anti_engine.inference import MultiSlotInferenceClient

    client = MultiSlotInferenceClient(
        slot_names=[f"cae-multi-{uuid.uuid4().hex}-{i}" for i in range(3)],
        max_batch=8,
        request_timeout_s=0.1,
    )
    calls: list[int] = []

    for idx, slot_client in enumerate(client._clients):  # type: ignore[attr-defined]
        def _fake_eval(
            x: np.ndarray,
            relations: np.ndarray | None = None,
            *,
            _idx: int = idx,
        ) -> tuple[np.ndarray, np.ndarray]:
            del relations  # interface conformance for SlotInferenceClient.evaluate_encoded
            calls.append(_idx)
            return (
                np.zeros((x.shape[0], 4672), dtype=np.float32),
                np.zeros((x.shape[0], 3), dtype=np.float32),
            )

        slot_client.evaluate_encoded = _fake_eval  # type: ignore[method-assign]

    x = np.zeros((1, 146, 8, 8), dtype=np.float32)
    for _ in range(5):
        client.evaluate_encoded(x)

    assert calls == [0, 1, 2, 0, 1]
    assert client.stats["lifetime_requests"] == 5
    assert client.stats["lifetime_positions"] == 5
    assert client.stats["slot_requests"] == [2, 2, 1]


def test_multi_slot_inference_client_uses_first_free_slot() -> None:
    from chess_anti_engine.inference import MultiSlotInferenceClient

    client = MultiSlotInferenceClient(
        slot_names=[f"cae-free-{uuid.uuid4().hex}-{i}" for i in range(2)],
        max_batch=8,
        request_timeout_s=1.0,
    )
    calls: list[int] = []
    calls_lock = threading.Lock()
    slow_started = threading.Event()
    release_slow = threading.Event()

    def _record(idx: int) -> None:
        with calls_lock:
            calls.append(idx)

    def _slow_eval(
        x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations  # interface conformance for SlotInferenceClient.evaluate_encoded
        _record(0)
        slow_started.set()
        assert release_slow.wait(timeout=1.0)
        return (
            np.zeros((x.shape[0], 4672), dtype=np.float32),
            np.zeros((x.shape[0], 3), dtype=np.float32),
        )

    def _fast_eval(
        x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations  # interface conformance for SlotInferenceClient.evaluate_encoded
        _record(1)
        return (
            np.zeros((x.shape[0], 4672), dtype=np.float32),
            np.zeros((x.shape[0], 3), dtype=np.float32),
        )

    client._clients[0].evaluate_encoded = _slow_eval  # type: ignore[method-assign,attr-defined]
    client._clients[1].evaluate_encoded = _fast_eval  # type: ignore[method-assign,attr-defined]

    x = np.zeros((1, 146, 8, 8), dtype=np.float32)
    slow_thread = threading.Thread(target=client.evaluate_encoded, args=(x,))
    slow_thread.start()
    assert slow_started.wait(timeout=1.0)

    client.evaluate_encoded(x)
    client.evaluate_encoded(x)
    release_slow.set()
    slow_thread.join(timeout=1.0)

    assert calls == [0, 1, 1]
    assert client.stats["lifetime_requests"] == 3
    assert client.stats["lifetime_positions"] == 3
    assert client.stats["slot_requests"] == [1, 2]
    assert client.stats["max_inflight"] == 2


def test_slot_inference_client_waits_for_slot_creation() -> None:
    slot_name = f"cae-late-{uuid.uuid4().hex}"
    layout = None
    done = threading.Event()

    def _serve_once() -> None:
        nonlocal layout
        time.sleep(0.05)
        shm = SharedMemory(name=slot_name, create=True, size=_SlotLayout.compute(8).total_bytes)
        try:
            layout = _SlotLayout.compute(8)
            slot = _InferenceSlot(shm, layout, owns=False)
            slot.state = 0
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if int(slot.state) == 1:
                    bsz = slot.batch_size
                    slot.policy[:bsz].fill(3.0)
                    slot.wdl[:bsz].fill(4.0)
                    slot.state = 2
                    done.set()
                    return
                time.sleep(0.001)
        finally:
            shm.close()
            shm.unlink()

    t = threading.Thread(target=_serve_once, daemon=True)
    t.start()
    client = SlotInferenceClient(slot_name=slot_name, max_batch=8, request_timeout_s=1.0)
    try:
        pol, wdl = client.evaluate_encoded(np.zeros((2, 146, 8, 8), dtype=np.float32))
    finally:
        client.close()
    t.join(timeout=1.0)
    assert done.is_set()
    assert np.allclose(pol, 3.0)
    assert np.allclose(wdl, 4.0)


def test_slot_inference_client_reconnects_after_slot_recreation() -> None:
    slot_name = f"cae-reconnect-{uuid.uuid4().hex}"
    layout = _SlotLayout.compute(8)
    old_shm = SharedMemory(name=slot_name, create=True, size=layout.total_bytes)
    old_slot = _InferenceSlot(old_shm, layout, owns=False)
    old_slot.state = 0
    old_slot.batch_size = 0

    restarted = threading.Event()
    served = threading.Event()

    def _restart_and_serve() -> None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if int(old_slot.state) == 1:
                old_slot.state = _STATE_SHUTDOWN
                old_shm.close()
                old_shm.unlink()
                new_shm = SharedMemory(name=slot_name, create=True, size=layout.total_bytes)
                try:
                    new_slot = _InferenceSlot(new_shm, layout, owns=False)
                    new_slot.state = 0
                    new_slot.batch_size = 0
                    restarted.set()
                    deadline_new = time.monotonic() + 2.0
                    while time.monotonic() < deadline_new:
                        if int(new_slot.state) == 1:
                            bsz = new_slot.batch_size
                            new_slot.policy[:bsz].fill(5.0)
                            new_slot.wdl[:bsz].fill(6.0)
                            new_slot.state = 2
                            served.set()
                            return
                        time.sleep(0.001)
                finally:
                    new_shm.close()
                    new_shm.unlink()
                return
            time.sleep(0.001)

    t = threading.Thread(target=_restart_and_serve, daemon=True)
    t.start()
    client = SlotInferenceClient(slot_name=slot_name, max_batch=8, request_timeout_s=1.0)
    try:
        pol, wdl = client.evaluate_encoded(np.zeros((2, 146, 8, 8), dtype=np.float32))
    finally:
        client.close()
    t.join(timeout=1.0)
    assert restarted.is_set()
    assert served.is_set()
    assert np.allclose(pol, 5.0)
    assert np.allclose(wdl, 6.0)


def test_slot_inference_client_attach_does_not_unlink_broker_slot(tmp_path: Path) -> None:
    broker = SlotBroker(
        publish_dir=tmp_path / "publish",
        num_slots=1,
        max_batch_per_slot=8,
        device="cpu",
        compile_inference=False,
        batch_wait_ms=0.0,
        slot_prefix=f"cae-attach-{uuid.uuid4().hex}",
    )
    try:
        slot_name = broker.slot_names[0]
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import time\n"
                    "from chess_anti_engine.inference import SlotInferenceClient\n"
                    f"client = SlotInferenceClient(slot_name={slot_name!r}, max_batch=8, request_timeout_s=1.0)\n"
                    "try:\n"
                    "    client._connect(deadline=time.monotonic() + 1.0)\n"
                    "finally:\n"
                    "    client.close()\n"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert child.returncode == 0, child.stderr

        probe = SharedMemory(name=slot_name, create=False)
        probe.close()
    finally:
        broker.shutdown()


def test_slot_broker_zeroes_outputs_when_model_unavailable(tmp_path: Path) -> None:
    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    broker = SlotBroker(
        publish_dir=publish_dir,
        num_slots=1,
        max_batch_per_slot=8,
        device="cpu",
        compile_inference=False,
        batch_wait_ms=0.0,
        slot_prefix=f"cae-zero-{uuid.uuid4().hex}",
    )
    try:
        slot = broker._slots[0]
        slot.batch_size = 2
        slot.policy[:2].fill(7.0)
        slot.wdl[:2].fill(9.0)
        broker._ensure_model = lambda: None  # type: ignore[method-assign]
        broker._process_batch([slot])
        assert slot.state == 2
        assert np.allclose(slot.policy[:2], 0.0)
        assert np.allclose(slot.wdl[:2], 0.0)
    finally:
        broker.shutdown()


def test_local_model_evaluator_respects_amp_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool, str]] = []

    class _TinyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            b = x.shape[0]
            return {
                "policy": torch.zeros((b, 4672), dtype=torch.float32, device=x.device),
                "wdl": torch.zeros((b, 3), dtype=torch.float32, device=x.device),
            }

    class _AutocastRecorder:
        def __init__(self, *, device: str, enabled: bool = True, dtype: str = "auto"):
            calls.append((str(device), bool(enabled), str(dtype)))

        def __enter__(self):
            return None

        def __exit__(self, _exc_type, _exc, _tb):
            return False

    monkeypatch.setattr("chess_anti_engine.inference.inference_autocast", _AutocastRecorder)

    x = np.zeros((1, 146, 8, 8), dtype=np.float32)
    evaluator = LocalModelEvaluator(_TinyModel().eval(), device="cpu", use_amp=False, amp_dtype="off")
    evaluator.evaluate_encoded(x)

    assert calls == [("cpu", False, "off")]


def test_slot_broker_honors_shutdown_while_idle(tmp_path: Path) -> None:
    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    broker = SlotBroker(
        publish_dir=publish_dir,
        num_slots=1,
        max_batch_per_slot=8,
        device="cpu",
        compile_inference=False,
        batch_wait_ms=0.0,
        slot_prefix=f"cae-shutdown-{uuid.uuid4().hex}",
    )
    try:
        t = threading.Thread(target=broker.serve_forever, daemon=True)
        t.start()
        deadline = time.monotonic() + 1.0
        while broker._slots[0].state != 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        broker._slots[0].state = 255
        t.join(timeout=1.0)
        assert not t.is_alive()
    finally:
        broker.shutdown()


def test_slot_broker_reloads_model_immediately_after_manifest_change(tmp_path: Path) -> None:
    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    model_path = publish_dir / "latest_model.pt"

    model_cfg = ModelConfig(
        kind="tiny",
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        ffn_mult=2,
        use_smolgen=False,
        use_nla=False,
    )

    def _write_model_with_manifest(seed: int) -> str:
        torch.manual_seed(seed)
        model = build_model(model_cfg).eval()
        torch.save({"model": model.state_dict()}, model_path)
        model_sha = _sha256_file(model_path)
        manifest = {
            "model": {
                "sha256": model_sha,
                "filename": "latest_model.pt",
            },
            "model_config": {
                "kind": model_cfg.kind,
                "embed_dim": model_cfg.embed_dim,
                "num_layers": model_cfg.num_layers,
                "num_heads": model_cfg.num_heads,
                "ffn_mult": model_cfg.ffn_mult,
                "use_smolgen": model_cfg.use_smolgen,
                "use_nla": model_cfg.use_nla,
                "use_qk_rmsnorm": model_cfg.use_qk_rmsnorm,
                "gradient_checkpointing": False,
            },
        }
        manifest_path = publish_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        st = manifest_path.stat()
        os.utime(manifest_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
        return model_sha

    broker = SlotBroker(
        publish_dir=publish_dir,
        num_slots=1,
        max_batch_per_slot=8,
        device="cpu",
        compile_inference=False,
        batch_wait_ms=0.0,
        slot_prefix=f"cae-reload-{uuid.uuid4().hex}",
    )
    try:
        first_sha = _write_model_with_manifest(seed=0)
        broker._ensure_model()
        assert broker._model_sha == first_sha

        second_sha = _write_model_with_manifest(seed=1)
        broker._ensure_model()
        assert broker._model_sha == second_sha
    finally:
        broker.shutdown()
