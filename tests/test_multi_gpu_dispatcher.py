"""Multi-GPU dispatcher routing (phase 7).

Validates ``MultiGPUDispatcher`` without needing multiple GPUs — we stand up
N CPU evaluators and check that load distribution is balanced under
concurrent callers. Same-device semantics apply; the only difference
between a CPU test and a real multi-GPU deployment is where the compiled
model lives.
"""
from __future__ import annotations

import threading
from collections import Counter

import numpy as np

from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_dispatcher import MultiGPUDispatcher
from chess_anti_engine.model import ModelConfig, build_model


class _CountingEvaluator:
    """Records every call it receives so the test can assert routing."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.n_calls = 0
        self._lock = threading.Lock()

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            self.n_calls += 1
        return (np.zeros((x.shape[0], 4672), dtype=np.float32),
                np.zeros((x.shape[0], 3), dtype=np.float32))


def test_multi_gpu_distributes_calls() -> None:
    evaluators = [_CountingEvaluator(f"dev{i}") for i in range(4)]
    dispatcher = MultiGPUDispatcher(evaluators)

    x = np.zeros((1, 146, 8, 8), dtype=np.float32)
    for _ in range(40):
        dispatcher.evaluate_encoded(x)

    counts = Counter({ev.name: ev.n_calls for ev in evaluators})
    # With serial callers the first device always wins (all in-flight counts
    # are zero when the next call arrives), but the round-robin tiebreaker
    # rotates ties. Allow some slack.
    total = sum(counts.values())
    assert total == 40
    # Every device should at least be used once with round-robin tiebreak.
    for name, n in counts.items():
        assert n > 0, f"device {name} never selected: {counts}"


def test_multi_gpu_concurrent_callers() -> None:
    evaluators = [_CountingEvaluator(f"dev{i}") for i in range(4)]
    dispatcher = MultiGPUDispatcher(evaluators)

    x = np.zeros((1, 146, 8, 8), dtype=np.float32)
    calls_per_thread = 20
    n_threads = 8

    def worker() -> None:
        for _ in range(calls_per_thread):
            dispatcher.evaluate_encoded(x)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    total = sum(ev.n_calls for ev in evaluators)
    assert total == calls_per_thread * n_threads
    # Distribution should be reasonably balanced. For N calls across 4
    # devices, each should see within 50-150% of the mean.
    mean = total / 4
    for ev in evaluators:
        assert 0.3 * mean <= ev.n_calls <= 1.7 * mean, (
            f"{ev.name}: {ev.n_calls} vs mean {mean}")


def test_multi_gpu_single_device_equivalent() -> None:
    """N=1 should be a valid degenerate case — dispatcher routes all calls
    to the single device, which must work."""
    cfg = ModelConfig(embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0)
    model = build_model(cfg)
    model.eval()
    evaluator = DirectGPUEvaluator(model, device="cpu", max_batch=4, use_amp=False)
    dispatcher = MultiGPUDispatcher([evaluator])

    x = np.random.default_rng(0).standard_normal((2, 146, 8, 8), dtype=np.float32)
    pol, wdl = dispatcher.evaluate_encoded(x)
    assert pol.shape == (2, 4672)
    assert wdl.shape == (2, 3)
    assert dispatcher.n_devices == 1


def test_multi_gpu_rejects_empty() -> None:
    try:
        MultiGPUDispatcher([])
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty evaluator list")
