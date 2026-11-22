"""CAE_BUCKET_HIST / CAE_BATCH_TRACE record on the evaluator the UCI engine uses.

`DirectGPUEvaluator` (the compiled engine's bucket-padding evaluator) used to
pad through `compiled_forward_rows`, which never reached the histogram hook, so
setting CAE_BUCKET_HIST on the engine recorded nothing and said nothing. These
tests drive that evaluator on CPU with the diagnostics switched on in-process
(the module reads the environment once, at import).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from chess_anti_engine import inference
from chess_anti_engine.inference import DirectGPUEvaluator, compiled_forward_rows
from chess_anti_engine.model import ModelConfig, build_model

_PLANES = 146


def _model() -> torch.nn.Module:
    m = build_model(ModelConfig(
        input_extra_features="v1", embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0,
    ))
    m.eval()
    return m


@pytest.fixture
def hist(monkeypatch, tmp_path):
    monkeypatch.setattr(inference, "_BUCKET_HIST_ENABLED", True)
    monkeypatch.setattr(inference, "_FORWARD_SIZE_TRACE", True)
    monkeypatch.setattr(inference, "_BUCKET_HIST_PATH", str(tmp_path / "hist.json"))
    monkeypatch.setattr(inference, "_bucket_hist", {"batch": {}, "legal_policy": {}})
    return inference._bucket_hist


@pytest.fixture
def trace(monkeypatch, tmp_path):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(inference, "_BATCH_TRACE_ENABLED", True)
    monkeypatch.setattr(inference, "_FORWARD_SIZE_TRACE", True)
    monkeypatch.setattr(inference, "_BATCH_TRACE_PATH", str(path))
    monkeypatch.setattr(inference, "_batch_trace_buf", [])
    return path


def _x(n: int) -> np.ndarray:
    return np.random.default_rng(n).random((n, _PLANES, 8, 8), dtype=np.float32)


def test_every_dense_entry_of_a_padding_evaluator_records_its_live_rows(hist) -> None:
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, n_slots=2,
                            use_amp=False, pad_to_compiled_buckets=True)
    ev.evaluate_encoded(_x(17))                     # sync
    ev.get_input_buffer(5, slot=1)[:] = _x(5)
    ev.evaluate_inplace_async(5, slot=1)            # in place, async
    ev.evaluate_encoded_async(_x(33))               # async
    ev.get_input_buffer(1, slot=0)[:] = _x(1)
    ev.evaluate_inplace(1, slot=0)                  # in place, sync
    # Raw live counts, not the padded 32 / 16 / 64 shapes.
    assert hist["batch"] == {17: 1, 5: 1, 33: 1, 1: 1}


def test_the_legal_entry_records_its_policy_size_and_one_batch(hist) -> None:
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64,
                            use_amp=False, pad_to_compiled_buckets=True)
    counts = np.array([3, 4, 5], dtype=np.int32)
    flat = np.concatenate([np.arange(c, dtype=np.int32) for c in counts])
    ev.evaluate_legal_bf16(_x(3), flat, counts)
    assert hist["legal_policy"] == {12: 1}
    assert hist["batch"] == {3: 1}


def test_a_non_padding_legal_call_records_one_batch_not_two(hist) -> None:
    # The sync wrapper pre-pads for a non-padding evaluator; it must not note
    # the batch AND let the forward entry note it again.
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False)
    counts = np.array([2, 2, 2], dtype=np.int32)
    flat = np.zeros(6, dtype=np.int32)
    ev.evaluate_legal_bf16(_x(3), flat, counts)
    assert sum(hist["batch"].values()) == 1


def test_the_batch_trace_writes_one_record_per_forward(trace) -> None:
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64,
                            use_amp=False, pad_to_compiled_buckets=True)
    for n in (1, 9, 40):
        ev.evaluate_encoded(_x(n))
    with inference._batch_trace_lock:
        inference._flush_batch_trace()
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [(r["graph"], r["rows"], r["padded"]) for r in rows] == [
        ("dense", n, compiled_forward_rows(n, 64)) for n in (1, 9, 40)
    ]
    assert rows[0]["t"] <= rows[1]["t"] <= rows[2]["t"]


def test_switched_off_the_forward_path_never_calls_the_recorder(monkeypatch) -> None:
    monkeypatch.setattr(inference, "_FORWARD_SIZE_TRACE", False)

    def boom(*_a, **_k):
        raise AssertionError("recorder called with the diagnostics off")

    monkeypatch.setattr(inference, "_note_forward", boom)
    monkeypatch.setattr(inference, "_note_legal_policy", boom)
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64,
                            use_amp=False, pad_to_compiled_buckets=True)
    ev.evaluate_encoded(_x(7))
    counts = np.array([3], dtype=np.int32)
    ev.evaluate_legal_bf16(_x(1), np.arange(3, dtype=np.int32), counts)


def test_the_environment_variables_reach_a_fresh_process(tmp_path) -> None:
    """The real switch: both variables set in a child's environment, read at
    import, and the files written by the child's own atexit flush."""
    import os
    import subprocess
    import sys

    from tests._child_env import gpu_free_env

    hist_path, trace_path = tmp_path / "hist.json", tmp_path / "trace.jsonl"
    code = (
        "import numpy as np, torch\n"
        "from chess_anti_engine.inference import DirectGPUEvaluator\n"
        "from chess_anti_engine.model import ModelConfig, build_model\n"
        "m = build_model(ModelConfig(input_extra_features='v1', embed_dim=16, num_layers=1,"
        " num_heads=2, ffn_mult=2.0)).eval()\n"
        "ev = DirectGPUEvaluator(m, device='cpu', max_batch=64, use_amp=False,"
        " pad_to_compiled_buckets=True)\n"
        "ev.evaluate_encoded(np.zeros((21, 146, 8, 8), dtype=np.float32))\n"
    )
    env = gpu_free_env({
        **os.environ, "CAE_BUCKET_HIST": str(hist_path), "CAE_BATCH_TRACE": str(trace_path),
    })
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(hist_path.read_text())["batch_counts"] == {"21": 1}
    (record,) = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert (record["graph"], record["rows"], record["padded"]) == ("dense", 21, 32)
