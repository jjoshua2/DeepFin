"""CAE_BUCKET_HIST / CAE_BATCH_TRACE record on the evaluator the UCI engine uses.

`DirectGPUEvaluator` (the compiled engine's bucket-padding evaluator) used to
pad through `compiled_forward_rows`, which never reached the histogram hook, so
setting CAE_BUCKET_HIST on the engine recorded nothing and said nothing. These
tests drive that evaluator on CPU with the diagnostics switched on in-process
(the module reads the environment once, at import).
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine import inference
from chess_anti_engine.inference import (
    _MODE_DENSE_F32,
    _POLICY_SIZE,
    _STATE_REQUEST,
    AOTEvaluator,
    DirectGPUEvaluator,
    SharedSlotBroker,
    SlotBroker,
    compiled_forward_rows,
)
from chess_anti_engine.inference_dispatcher import CUDAOwnerDispatcher, ThreadSafeGPUDispatcher
from chess_anti_engine.inference_threaded import ThreadedDispatcher
from chess_anti_engine.mcts import gumbel_c
from chess_anti_engine.mcts.gumbel import GumbelConfig
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
    fd = inference._open_batch_trace(str(path))
    monkeypatch.setattr(inference, "_BATCH_TRACE_ENABLED", True)
    monkeypatch.setattr(inference, "_FORWARD_SIZE_TRACE", True)
    monkeypatch.setattr(inference, "_BATCH_TRACE_PATH", str(path))
    monkeypatch.setattr(inference, "_batch_trace_fd", fd)
    monkeypatch.setattr(inference, "_batch_trace_buf", [])
    yield path
    os.close(fd)


def _records(path: Path) -> list[dict]:
    with inference._batch_trace_lock:
        inference._flush_batch_trace()
    return [json.loads(line) for line in path.read_text().splitlines()]


def _triples(path: Path) -> list[tuple[str, int, int]]:
    return [(r["graph"], r["rows"], r["padded"]) for r in _records(path)]


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


# --- Fix round (review of 897b34d22) ------------------------------------------


def test_every_record_carries_the_writing_pid(trace) -> None:
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False)
    ev.evaluate_encoded(_x(3))
    (record,) = _records(trace)
    assert record["pid"] == os.getpid()


def test_a_flush_empties_the_buffer_so_no_record_is_written_twice(trace) -> None:
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False)
    ev.evaluate_encoded(_x(2))
    with inference._batch_trace_lock:
        inference._flush_batch_trace()
    assert inference._batch_trace_buf == []
    ev.evaluate_encoded(_x(3))
    assert [r["rows"] for r in _records(trace)] == [2, 3]


def test_an_unwritable_trace_path_fails_loudly_at_enable_time(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="CAE_BATCH_TRACE"):
        inference._open_batch_trace(str(tmp_path / "no-such-dir" / "trace.jsonl"))


def test_an_unwritable_trace_path_stops_a_fresh_process(tmp_path) -> None:
    import subprocess
    import sys

    from tests._child_env import gpu_free_env

    env = gpu_free_env({**os.environ, "CAE_BATCH_TRACE": str(tmp_path / "missing" / "t.jsonl")})
    proc = subprocess.run([sys.executable, "-c", "import chess_anti_engine.inference"], env=env,
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode != 0
    assert "cannot be opened for append" in proc.stderr


def test_the_non_padding_async_entry_records_dense_rows(trace) -> None:
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False)
    ev.evaluate_encoded_async(_x(9))
    ev.evaluate_encoded_async(_x(16), live_rows=11)
    assert _triples(trace) == [("dense", 9, 9), ("dense", 11, 16)]


def test_live_rows_reach_every_padded_dense_entry(trace) -> None:
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, n_slots=2,
                            use_amp=False, pad_to_compiled_buckets=True)
    ev.get_input_buffer(40, slot=1)[:] = _x(40)
    ev.evaluate_inplace_async(40, slot=1, live_rows=33)
    ev.evaluate_encoded_async(_x(24), live_rows=20)
    ev.get_input_buffer(8, slot=0)[:] = _x(8)
    ev.evaluate_inplace(8, slot=0, live_rows=5)
    ev.evaluate_encoded(_x(16), live_rows=13)
    assert _triples(trace) == [
        ("dense", 33, compiled_forward_rows(40, 64)),
        ("dense", 20, compiled_forward_rows(24, 64)),
        ("dense", 5, compiled_forward_rows(8, 64)),
        ("dense", 13, compiled_forward_rows(16, 64)),
    ]


def _legal_args(n_live: int, bsz: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([2] * n_live + [0] * (bsz - n_live), dtype=np.int32)
    return np.zeros(2 * n_live, dtype=np.int32), counts


def test_the_cpu_legal_entry_records_the_callers_live_rows(trace) -> None:
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False,
                            pad_to_compiled_buckets=True)
    ev.get_input_buffer(16, slot=0)[:] = _x(16)
    flat, counts = _legal_args(3, 16)
    ev.evaluate_inplace_legal_bf16_async(16, flat, counts, slot=0, live_rows=3)
    assert [(g, r) for g, r, _ in _triples(trace)] == [("dense", 3)]


def test_the_cuda_legal_branch_records_live_rows(trace, monkeypatch) -> None:
    """The CUDA branch notes "legal" itself; reached on CPU by stubbing the
    device-only forward."""
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False,
                            pad_to_compiled_buckets=True)
    monkeypatch.setattr(ev, "_use_cuda", True)
    monkeypatch.setattr(
        ev, "_async_forward_legal_bf16",
        lambda rows, f, c, nl, slot=0: (torch.zeros(nl, dtype=torch.uint16), torch.zeros(rows, 3), None),
        raising=False,
    )
    flat, counts = _legal_args(5, 16)
    ev.evaluate_inplace_legal_bf16_async(16, flat, counts, slot=0, live_rows=5)
    ((graph, rows, padded),) = _triples(trace)
    assert (graph, rows) == ("legal", 5)
    assert padded >= 16


def _aot() -> AOTEvaluator:
    """An AOTEvaluator on CPU: packages stubbed by one eager net per bucket."""
    aot = object.__new__(AOTEvaluator)
    aot.device = "cpu"
    aot._max_batch = 64
    aot._sorted_buckets = [8, 16, 32, 64]
    aot._n_slots = 2
    aot._ownership = threading.RLock()
    net = _model()
    aot._models = dict.fromkeys(aot._sorted_buckets, net)
    bufs = [np.zeros((64, _PLANES, 8, 8), dtype=np.float32) for _ in range(2)]
    aot._pinned_inputs_np = bufs
    aot._slot_input_bf16 = [False, False]
    aot._device_input = lambda xx, bucket, slot=0: torch.from_numpy(  # pyright: ignore[reportAttributeAccessIssue]
        np.pad(xx, ((0, bucket - xx.shape[0]), (0, 0), (0, 0), (0, 0))))
    aot._device_input_inplace = lambda bucket, slot: torch.from_numpy(
        bufs[slot][:bucket].copy())
    return aot


def test_aot_records_once_per_forward_not_per_bucket_lookup(trace) -> None:
    """Native search reserves a CAPACITY-sized input buffer, then forwards a
    small batch from it: only the forward is a sample."""
    aot = _aot()
    aot.get_input_buffer(64, slot=0)[:5] = _x(5)     # capacity reservation, no forward
    aot.evaluate_inplace_async(5, slot=0)
    aot.evaluate_encoded_async(_x(12))
    aot.evaluate_encoded(_x(3))
    aot.get_input_buffer(64, slot=1)
    aot.evaluate_inplace_async(16, slot=1, live_rows=9)
    aot.evaluate_encoded_async(_x(32), live_rows=30)
    assert _triples(trace) == [
        ("aot", 5, 8), ("aot", 12, 16), ("aot", 3, 8), ("aot", 9, 16), ("aot", 30, 32),
    ]


def test_aot_switched_off_never_calls_the_recorder(monkeypatch) -> None:
    monkeypatch.setattr(inference, "_FORWARD_SIZE_TRACE", False)

    def boom(*_a, **_k):
        raise AssertionError("recorder called with the diagnostics off")

    monkeypatch.setattr(inference, "_note_forward", boom)
    aot = _aot()
    aot.get_input_buffer(64, slot=0)[:5] = _x(5)
    aot.evaluate_inplace_async(5, slot=0)
    aot.evaluate_encoded_async(_x(5))
    aot.evaluate_encoded(_x(5))


def test_the_wrappers_forward_live_rows_and_declare_it(trace) -> None:
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, n_slots=2,
                            use_amp=False, pad_to_compiled_buckets=True)
    chain = CUDAOwnerDispatcher(ThreadSafeGPUDispatcher(ev))
    try:
        assert inference.accepts_live_rows(chain)
        assert inference.forward_live_rows_kwargs(chain, 7) == {"live_rows": 7}
        chain.get_input_buffer(16, slot=0)[:] = _x(16)
        chain.evaluate_inplace_async(16, slot=0, live_rows=7)
        chain.evaluate_encoded_async(_x(32), live_rows=21)
    finally:
        chain.close()
    assert [(g, r) for g, r, _ in _triples(trace)] == [("dense", 7), ("dense", 21)]


def test_a_getattr_forwarder_cannot_claim_live_rows() -> None:
    from chess_anti_engine.inference_cache import EncodedEvalCache

    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False)
    assert inference.accepts_live_rows(ev)
    assert not inference.accepts_live_rows(EncodedEvalCache(ev, max_entries=8))


def test_switched_off_no_call_site_passes_live_rows(monkeypatch) -> None:
    monkeypatch.setattr(inference, "_FORWARD_SIZE_TRACE", False)
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False)
    assert inference.forward_live_rows_kwargs(ev, 3) == {}


def test_the_threaded_dispatcher_legal_path_records_its_real_rows(trace) -> None:
    disp = ThreadedDispatcher(_model(), device="cpu", max_batch=64, legal_bf16=True)
    try:
        counts = np.array([3, 4, 5], dtype=np.int32)
        flat = np.concatenate([np.arange(c, dtype=np.int32) for c in counts])
        disp.evaluate_legal_bf16(_x(3), flat, counts)
    finally:
        disp.shutdown()
    assert [r for _, r, _ in _triples(trace)] == [3]


def test_a_forked_child_drops_the_parents_buffer_and_stamps_its_own_pid(monkeypatch) -> None:
    monkeypatch.setattr(inference, "_batch_trace_buf", ["parent record\n"])
    monkeypatch.setattr(inference, "_batch_trace_pid", 1)
    monkeypatch.setattr(inference, "_batch_trace_lock", inference._batch_trace_lock)
    monkeypatch.setattr(inference.os, "getpid", lambda: 4242)
    inference._reset_batch_trace_in_child()
    assert inference._batch_trace_buf == []
    assert inference._batch_trace_pid == 4242


def test_the_threaded_dispatcher_records_its_real_rows_not_its_bucket(trace) -> None:
    disp = ThreadedDispatcher(_model(), device="cpu", max_batch=64, legal_bf16=False)
    try:
        disp.evaluate_encoded(_x(5))
    finally:
        disp.shutdown()
    ((graph, rows, padded),) = _triples(trace)
    assert (graph, rows) == ("dense", 5)
    assert padded >= 16


# --- the native Gumbel search, end to end ------------------------------------

_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "8/8/4k3/8/2P5/8/4K3/8 w - - 0 1",
    "r3k2r/ppp2ppp/2n5/3qp3/3P4/2P2N2/PP3PPP/R2QKB1R b KQkq - 0 9",
]


class _EncodedArmEvaluator(DirectGPUEvaluator):
    """Declines the pinned slot API, so the search's single loop takes its
    ``evaluate_encoded_async`` arm (the only way to reach that arm's
    relations call with an evaluator that records)."""

    @property
    def supports_inplace_api(self) -> bool:
        return False


# KQ v K with mate in one: a group of these finishes at once, so the other
# group of a 2-group pipeline drains alone through `_drain_sequential`.
_QUICK = "k7/8/1K6/8/8/8/8/7Q w - - 0 1"


def _search(ev, n_boards: int, monkeypatch, *, sims: int = 32, relations: bool = False,
            fens: list[str] | None = None):
    drains: list[tuple[int, int]] = []
    callers: list[str] = []
    orig = gumbel_c._record_batch_dup

    def spy(x, n, p):
        import sys

        drains.append((int(n), int(p)))
        callers.append(sys._getframe(1).f_code.co_name)
        return orig(x, n, p)

    monkeypatch.setattr(gumbel_c, "_record_batch_dup", spy)
    torch.manual_seed(1234)
    res = gumbel_c.run_gumbel_root_many_c(
        None, [chess.Board(f) for f in (fens or [_FENS[i % len(_FENS)] for i in range(n_boards)])],
        device="cpu", rng=np.random.default_rng(99),
        cfg=GumbelConfig(simulations=sims, topk=8, add_noise=True,
                         input_extra_features="v2_threats" if relations else "v1",
                         compute_relations=relations),
        evaluator=ev, target_batch=32,
    )
    return res, drains, callers


@pytest.mark.parametrize(
    ("n_boards", "pad", "n_slots", "arm"),
    [
        (20, True, 2, "single loop in-place (UCI default, legal_bf16 off)"),
        # 66 boards: >= 64 takes the 2-group pipeline, and 33 per group keeps
        # the wide drains off the bucket ladder (132 -> 170), so they pad.
        (66, True, 2, "pipelined in-place"),
        (66, False, 1, "pipelined encoded-async, non-padding (RL collect)"),
        (20, False, 1, "single loop in-place, non-padding"),
        (20, True, 0, "single loop encoded-async"),
        (66, True, 2, "pipeline group drained alone, in-place"),
        (66, False, 1, "pipeline group drained alone, encoded-async"),
    ],
)
def test_a_real_search_traces_its_own_leaf_counts(trace, monkeypatch, n_boards, pad, n_slots, arm) -> None:
    cls = _EncodedArmEvaluator if n_slots == 0 else DirectGPUEvaluator
    ev = cls(_model_seeded(), device="cpu", max_batch=512, n_slots=max(n_slots, 1),
             use_amp=False, pad_to_compiled_buckets=pad, legal_bf16=False)
    fens = [_FENS[0]] * 33 + [_QUICK] * 33 if "drained alone" in arm else None
    _, drains, callers = _search(ev, n_boards, monkeypatch, fens=fens)
    if "drained alone" in arm:
        assert "_drain_sequential" in callers, f"{arm}: the lone-group drain was not reached"
    recs = _triples(trace)
    assert drains, arm
    leaf = recs[-len(drains):]          # the first record(s) are the root eval
    assert [r for _, r, _ in leaf] == [n for n, _ in drains], arm
    assert any(n < p for n, p in drains), f"{arm}: no drain was padded upstream -- vacuous"
    assert all(r <= padded for _, r, padded in recs), arm


@pytest.mark.parametrize("cls", [DirectGPUEvaluator, _EncodedArmEvaluator],
                         ids=["in-place arm", "encoded-async arm"])
def test_a_real_relations_search_traces_its_own_leaf_counts(trace, monkeypatch, cls) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(6)
        model = build_model(ModelConfig(
            embed_dim=16, num_layers=1, num_heads=2, use_smolgen=False,
            input_extra_features="v2_threats", use_dynamic_relations=True,
        )).eval()
    ev = cls(model, device="cpu", max_batch=512, n_slots=2, use_amp=False,
             pad_to_compiled_buckets=True, legal_bf16=False)
    _, drains, _ = _search(ev, 20, monkeypatch, sims=24, relations=True)
    leaf = _triples(trace)[-len(drains):]
    assert [r for _, r, _ in leaf] == [n for n, _ in drains]
    assert any(n < p for n, p in drains), "no drain was padded upstream -- vacuous"


def test_the_live_rows_keyword_does_not_change_the_search(monkeypatch, tmp_path) -> None:
    def run(on: bool):
        if on:
            path = tmp_path / f"t{uuid.uuid4().hex}.jsonl"
            monkeypatch.setattr(inference, "_BATCH_TRACE_ENABLED", True)
            monkeypatch.setattr(inference, "_FORWARD_SIZE_TRACE", True)
            monkeypatch.setattr(inference, "_batch_trace_fd", inference._open_batch_trace(str(path)))
        else:
            monkeypatch.setattr(inference, "_BATCH_TRACE_ENABLED", False)
            monkeypatch.setattr(inference, "_FORWARD_SIZE_TRACE", False)
        monkeypatch.setattr(inference, "_batch_trace_buf", [])
        ev = DirectGPUEvaluator(_model_seeded(), device="cpu", max_batch=256, n_slots=2,
                                use_amp=False, pad_to_compiled_buckets=True, legal_bf16=False)
        res, _, _ = _search(ev, 20, monkeypatch, sims=24)
        return res

    off, on = _digest(run(False)), _digest(run(True))
    assert off == on


def _digest(obj: object) -> str:
    """sha256 over every array / scalar in a search result (trees by type name:
    their repr carries an address)."""
    import hashlib

    h = hashlib.sha256()

    def feed(o: object) -> None:
        if isinstance(o, np.ndarray):
            h.update(o.tobytes())
            h.update(str(o.dtype).encode())
        elif isinstance(o, (list, tuple)):
            for v in o:
                feed(v)
        elif isinstance(o, dict):
            for k in sorted(o):
                feed(o[k])
        elif isinstance(o, torch.Tensor):
            feed(o.numpy())
        elif o is None or isinstance(o, (int, float, str, bool, np.integer, np.floating)):
            h.update(repr(o).encode())
        else:
            h.update(type(o).__name__.encode())

    feed(obj)
    return h.hexdigest()


def _model_seeded() -> torch.nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(7)
        return _model()


# --- the slot brokers ---------------------------------------------------------


class _TinyPolicy(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz = int(x.shape[0])
        return {
            "policy": torch.zeros((bsz, _POLICY_SIZE), dtype=torch.float32),
            "wdl": torch.full((bsz, 3), 1.0 / 3.0, dtype=torch.float32),
        }


def test_the_slot_broker_records_its_gathered_live_rows(trace, tmp_path, monkeypatch) -> None:
    publish = tmp_path / "publish"
    publish.mkdir()
    broker = SlotBroker(publish_dir=publish, num_slots=2, max_batch_per_slot=8, device="cpu",
                        compile_inference=False, batch_wait_ms=0.0,
                        slot_prefix=f"cae-trace-{uuid.uuid4().hex}")
    try:
        monkeypatch.setattr(SlotBroker, "_ensure_model", lambda _self: None)
        broker._model = _TinyPolicy().eval()
        broker._model_sha = "test"
        slots = []
        for idx, bsz in enumerate((3, 2)):
            slot = broker._slots[idx]
            slot.batch_size = bsz
            slot.request_mode = _MODE_DENSE_F32
            slot.request_id = 40 + idx
            slot.state = _STATE_REQUEST
            slots.append(slot)
        assert broker._process_batch(slots) == 0
        cap = int(broker._pinned_input.shape[0])
    finally:
        broker.shutdown()
    assert _triples(trace) == [
        ("broker_dense", 5, inference._compiled_padded_batch_size(5, capacity=cap, record=False)),
    ]


def test_the_shared_broker_records_each_trial_forward(trace) -> None:
    class _Slot:
        def __init__(self, bsz: int) -> None:
            self.state = _STATE_REQUEST
            self.request_mode = _MODE_DENSE_F32
            self.request_id = 7
            self.batch_size = bsz
            self.input = np.zeros((bsz, 4), dtype=np.float32)
            self.policy = np.zeros((bsz, 8), dtype=np.float32)
            self.wdl = np.zeros((bsz, 3), dtype=np.float32)

    class _Reached(torch.nn.Module):
        def forward(self, _x: torch.Tensor) -> None:
            raise RuntimeError("forward reached")

    class _Watchdog:
        def mark_forward_start(self, _batch: int) -> int:
            return 1

        def mark_forward_done(self, *, success: bool, token: int) -> None:
            del success, token

    broker = object.__new__(SharedSlotBroker)
    broker._trial_models = {"t0": _Reached()}
    broker._trial_no_model_warned_at = {}
    broker._trial_bad_mode_warned_at = {}
    broker._trial_bad_size_warned_at = {}
    broker._trial_streams = {}
    broker._first_inference_done = False
    broker._hang_watchdog = _Watchdog()  # pyright: ignore[reportAttributeAccessIssue]
    broker.unsupported_mode_requests = 0
    broker.device = "cpu"
    with pytest.raises(RuntimeError, match="forward reached"):
        broker._process_parallel({"t0": [_Slot(3)]})  # pyright: ignore[reportArgumentType]
    assert _triples(trace) == [("shared_broker", 3, 3)]
