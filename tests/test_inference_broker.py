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
    _STATE_SHUTDOWN,
    LocalModelEvaluator,
    SlotBroker,
    SlotInferenceClient,
    _InferenceSlot,
    _SlotLayout,
)
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.utils import sha256_file as _sha256_file


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

    proc = subprocess.Popen(  # pylint: disable=consider-using-with  # long-lived broker subprocess, explicit terminate in finally
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
                if slot.state == 1:
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
            if old_slot.state == 1:
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
                        if new_slot.state == 1:
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
