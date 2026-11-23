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
    _MODE_LEGAL_BF16,
    _POLICY_SIZE,
    _STATE_REQUEST,
    _STATE_RESPONSE,
    AOTEvaluator,
    DirectGPUEvaluator,
    SharedSlotBroker,
    SlotBroker,
    LocalModelEvaluator,
    MultiSlotInferenceClient,
    SlotInferenceClient,
    _InferenceSlot,
    _SlotLayout,
    compiled_forward_rows,
)
from chess_anti_engine.inference_dispatcher import (
    BatchCoalescingDispatcher,
    CUDAOwnerDispatcher,
    MultiGPUDispatcher,
    ThreadSafeGPUDispatcher,
)
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
    return [(r["graph"], r["live_rows"], r["call_rows"]) for r in _records(path)]


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
    assert [(r["graph"], r["live_rows"], r["call_rows"]) for r in rows] == [
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
    assert (record["graph"], record["live_rows"], record["call_rows"]) == ("dense", 21, 32)


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
    assert [r["live_rows"] for r in _records(trace)] == [2, 3]


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
    monkeypatch.setattr(ev, "_forward_chunk_rows", 4, raising=False)
    flat, counts = _legal_args(5, 16)
    ev.evaluate_inplace_legal_bf16_async(16, flat, counts, slot=0, live_rows=5)
    (rec,) = _records(trace)
    assert (rec["graph"], rec["live_rows"]) == ("legal", 5)
    assert rec["call_rows"] >= 16
    assert (rec["chunk_rows"], rec["chunks"]) == (4, -(-rec["call_rows"] // 4))


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
    aot.evaluate_encoded(_x(16), live_rows=11)
    assert _triples(trace) == [
        ("aot", 5, 8), ("aot", 12, 16), ("aot", 3, 8), ("aot", 9, 16), ("aot", 30, 32), ("aot", 11, 16),
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
        chain.evaluate_encoded(_x(16), live_rows=5)            # the sync arm's entry
    finally:
        chain.close()
    assert [(g, r) for g, r, _ in _triples(trace)] == [("dense", 7), ("dense", 21), ("dense", 5)]


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
            fens: list[str] | None = None, policy_temp: float = 1.0):
    drains: list[tuple[int, int]] = []
    callers: list[str] = []
    orig = gumbel_c._record_batch_dup

    def spy(x, n, p):
        import sys

        drains.append((int(n), int(p)))
        callers.append(sys._getframe(1).f_code.co_name)
        return orig(x, n, p)

    monkeypatch.setattr(gumbel_c, "_record_batch_dup", spy)
    # Pinned: other test files have emptied this ladder for the session before
    # (no drain pads then, and these tests' vacuity guard would fire).
    monkeypatch.setattr(gumbel_c, "_COMPILED_BATCH_BUCKETS", inference._COMPILED_BATCH_BUCKETS)
    torch.manual_seed(1234)
    res = gumbel_c.run_gumbel_root_many_c(
        None, [chess.Board(f) for f in (fens or [_FENS[i % len(_FENS)] for i in range(n_boards)])],
        device="cpu", rng=np.random.default_rng(99),
        cfg=GumbelConfig(simulations=sims, topk=8, add_noise=True,
                         input_extra_features="v2_threats" if relations else "v1",
                         compute_relations=relations, policy_temp=policy_temp),
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


def test_a_real_relations_search_through_the_sync_arm_traces_its_leaf_counts(trace, monkeypatch) -> None:
    """A sync-only evaluator (no async / in-place API) with relations: the drain's
    `evaluate_encoded(..., relations=...)` call carries the live count."""
    from chess_anti_engine.uci.device_guard import GuardedPrimaryEvaluator

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(6)
        model = build_model(ModelConfig(
            embed_dim=16, num_layers=1, num_heads=2, use_smolgen=False,
            input_extra_features="v2_threats", use_dynamic_relations=True,
        )).eval()
    inner = DirectGPUEvaluator(model, device="cpu", max_batch=512, use_amp=False,
                               pad_to_compiled_buckets=True, legal_bf16=False)
    _, drains, _ = _search(GuardedPrimaryEvaluator(inner, lambda f: f()), 20, monkeypatch,
                           sims=24, relations=True)
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
        def __init__(self, bsz: int, live: int = 0) -> None:
            self.live_rows = live
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
        broker._process_parallel({"t0": [_Slot(3), _Slot(4, live=2)]})  # pyright: ignore[reportArgumentType]
    assert _triples(trace) == [("shared_broker", 3 + 2, 7)]


# --- second fix round (review of 607503d5d) -----------------------------------


class _ClientShaped:
    """The live selfplay evaluator's surface (`MultiSlotInferenceClient`): sync
    `evaluate_encoded` + `evaluate_legal_bf16`, no async / in-place API, pads
    internally, takes ``live_rows=``. Records what each call SENDS."""

    supports_input_bf16_bits = False

    def __init__(self, *, pads_internally: bool = True) -> None:
        self.pads_batches_internally = pads_internally
        self.inner = DirectGPUEvaluator(_model_seeded(), device="cpu", max_batch=512, use_amp=False)
        self.calls: list[tuple[int, int | None]] = []

    @property
    def accepts_live_rows(self) -> bool:
        return True

    def evaluate_encoded(self, x, relations=None, *, live_rows=None):
        del relations
        self.calls.append((int(x.shape[0]), live_rows))
        return self.inner.evaluate_encoded(x)

    def evaluate_legal_bf16(self, x, flat, counts):
        self.calls.append((int(x.shape[0]), None))
        return self.inner.evaluate_legal_bf16(x, flat, counts)


@pytest.mark.usefixtures("trace")
def test_the_live_selfplay_sync_arm_reports_its_leaf_counts(monkeypatch) -> None:
    """policy_temp 1.5 (the live yaml) switches the legal-bf16 leaf arm off, so the
    drains take the sync arm. A sync evaluator that does NOT pad internally is
    SENT `padded` rows (unchanged); the live count rides along. (One that pads
    internally -- the slot clients -- is sent the live rows only:
    tests/test_drain_live_rows_to_padding_evaluators.py.)"""
    ev = _ClientShaped(pads_internally=False)
    _, drains, _ = _search(ev, 20, monkeypatch, sims=64, policy_temp=1.5)
    leaf = ev.calls[-len(drains):]
    assert [sent for sent, _ in leaf] == [p for _, p in drains]       # forward unchanged
    assert [live for _, live in leaf] == [n for n, _ in drains]       # live reported
    assert any(n < p for n, p in drains), "no drain was padded upstream -- vacuous"


def test_switched_off_the_sync_arm_sends_no_keyword(monkeypatch) -> None:
    monkeypatch.setattr(inference, "_FORWARD_SIZE_TRACE", False)
    ev = _ClientShaped()
    _, drains, _ = _search(ev, 20, monkeypatch, sims=32, policy_temp=1.5)
    assert drains
    assert all(live is None for _, live in ev.calls)


def _fake_broker(slot_name: str, layout: _SlotLayout, seen: list[tuple[int, int, int]], n_requests: int):
    from multiprocessing.shared_memory import SharedMemory

    shm = SharedMemory(name=slot_name, create=True, size=layout.total_bytes)
    _InferenceSlot(shm, layout, owns=False).write_protocol_header()

    def serve() -> None:
        import time

        slot = _InferenceSlot(shm, layout, owns=False)
        deadline = time.monotonic() + 10.0
        while len(seen) < n_requests and time.monotonic() < deadline:
            if int(slot.state) == _STATE_REQUEST:
                seen.append((slot.request_mode, slot.batch_size, slot.live_rows))
                slot.policy[: slot.batch_size].fill(0.0)
                slot.wdl[: slot.batch_size].fill(1.0 / 3.0)
                slot.state = _STATE_RESPONSE
            time.sleep(0.0005)

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    return shm, t


def test_the_slot_client_writes_the_live_count_and_clears_it() -> None:
    layout = _SlotLayout.compute(16)
    name = f"cae-live-{uuid.uuid4().hex}"
    seen: list[tuple[int, int, int]] = []
    shm, t = _fake_broker(name, layout, seen, 3)
    client = SlotInferenceClient(slot_name=name, max_batch=16, request_timeout_s=5.0)
    try:
        assert inference.accepts_live_rows(client)
        client.evaluate_encoded(_x(8), live_rows=5)
        counts = np.array([2, 2, 2], dtype=np.int32)
        client.evaluate_legal_bf16(np.zeros((3, _PLANES, 8, 8), np.uint16),
                                   np.zeros(6, np.int32), counts)
        client.evaluate_encoded(_x(4))
    finally:
        client.close()
        t.join(timeout=5.0)
        shm.close()
        shm.unlink()
    assert seen == [(_MODE_DENSE_F32, 8, 5), (_MODE_LEGAL_BF16, 3, 0), (_MODE_DENSE_F32, 4, 0)]


def test_the_multi_slot_client_forwards_the_live_count() -> None:
    layout = _SlotLayout.compute(16)
    prefix = f"cae-livem-{uuid.uuid4().hex}"
    seen: list[tuple[int, int, int]] = []
    shm, t = _fake_broker(f"{prefix}-0", layout, seen, 1)
    client = MultiSlotInferenceClient(slot_names=[f"{prefix}-0"], max_batch=16,
                                      request_timeout_s=5.0)
    try:
        assert inference.accepts_live_rows(client)
        client.evaluate_encoded(_x(8).astype(np.float32), live_rows=6)
    finally:
        client.close()
        t.join(timeout=5.0)
        shm.close()
        shm.unlink()
    assert seen == [(_MODE_DENSE_F32, 8, 6)]


def test_the_slot_broker_sums_the_slots_live_counts(trace, tmp_path, monkeypatch) -> None:
    publish = tmp_path / "publish"
    publish.mkdir()
    broker = SlotBroker(publish_dir=publish, num_slots=3, max_batch_per_slot=8, device="cpu",
                        compile_inference=False, batch_wait_ms=0.0,
                        slot_prefix=f"cae-trace-{uuid.uuid4().hex}")
    try:
        monkeypatch.setattr(SlotBroker, "_ensure_model", lambda _self: None)
        broker._model = _TinyPolicy().eval()
        broker._model_sha = "test"
        slots = []
        # live 3 of 8, all of 2 (old client / no padding: 0), torn 99 -> all 4
        for idx, (bsz, live) in enumerate(((8, 3), (2, 0), (4, 99))):
            slot = broker._slots[idx]
            slot.batch_size = bsz
            slot.live_rows = live
            slot.request_mode = _MODE_DENSE_F32
            slot.request_id = 50 + idx
            slot.state = _STATE_REQUEST
            slots.append(slot)
        assert broker._process_batch(slots) == 0
        cap = int(broker._pinned_input.shape[0])
    finally:
        broker.shutdown()
    assert _triples(trace) == [
        ("broker_dense", 3 + 2 + 4, inference._compiled_padded_batch_size(14, capacity=cap, record=False)),
    ]


# --- every wrapper a padding search can call declares live_rows, or is exempt ---

_SEARCH_ENTRIES = frozenset({"evaluate_encoded", "evaluate_encoded_async", "evaluate_inplace_async"})

# class qualname -> why it need not take ``live_rows``.
_LIVE_ROWS_EXEMPT: dict[str, str] = {
    "chess_anti_engine.inference.BatchEvaluator": "Protocol, not an evaluator",
    "chess_anti_engine.inference.AsyncBatchEvaluator": "Protocol, not an evaluator",
    "chess_anti_engine.uci.walker_pool._Evaluator": "Protocol, not an evaluator",
    "chess_anti_engine.inference.LocalModelEvaluator":
        "eager evaluator; records no forward-size trace, so there is no count to correct",
    "chess_anti_engine.inference_cache.EncodedEvalCache":
        "asks for unpadded leaf rows (wants_unpadded_leaf_rows), so it is handed live rows; "
        "what it forwards is its de-duplicated misses -- the trace is forward size there",
    "chess_anti_engine.uci.device_guard.PoolRowsEvaluator":
        "degraded searchmoves fallback after a primary failure; MultiGpuPucvPool.evaluate_rows "
        "re-splits rows onto pool workers -- its records count the rows it forwards",
}


def _evaluator_classes() -> list[tuple[str, type]]:
    """Every class under chess_anti_engine that defines a search entry (AST scan,
    not a hand-kept list)."""
    import ast
    import importlib

    root = Path(inference.__file__).resolve().parent
    found: list[tuple[str, type]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            names = {f.name for f in node.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if not names & _SEARCH_ENTRIES:
                continue
            mod = ".".join(path.relative_to(root.parent).with_suffix("").parts)
            found.append((f"{mod}.{node.name}", getattr(importlib.import_module(mod), node.name)))
    return found


def test_every_search_entry_class_takes_live_rows_or_is_exempt() -> None:
    classes = _evaluator_classes()
    names = {q for q, _ in classes}
    assert "chess_anti_engine.inference.DirectGPUEvaluator" in names       # the scan works
    missing = [q for q, cls in classes
               if q not in _LIVE_ROWS_EXEMPT
               and not isinstance(inspect_attr(cls, "accepts_live_rows"), property)]
    assert not missing, f"declare accepts_live_rows (and forward live_rows) or exempt with a reason: {missing}"
    stale = sorted(set(_LIVE_ROWS_EXEMPT) - names)
    assert not stale, f"exempt entries for classes that no longer define a search entry: {stale}"


def inspect_attr(cls: type, name: str) -> object:
    import inspect

    return inspect.getattr_static(cls, name, None)


def test_every_declaring_class_takes_the_keyword_on_each_search_entry() -> None:
    import inspect

    for q, cls in _evaluator_classes():
        if not isinstance(inspect_attr(cls, "accepts_live_rows"), property):
            continue
        for entry in _SEARCH_ENTRIES:
            fn = getattr(cls, entry, None)
            if fn is None:
                continue
            assert "live_rows" in inspect.signature(fn).parameters, f"{q}.{entry}"


def test_the_leaf_evaluators_declare_live_rows_true() -> None:
    assert inference.accepts_live_rows(_aot())
    assert inference.accepts_live_rows(
        DirectGPUEvaluator(_model(), device="cpu", max_batch=16, use_amp=False))
    disp = ThreadedDispatcher(_model(), device="cpu", max_batch=16, legal_bf16=False)
    try:
        assert inference.accepts_live_rows(disp)
    finally:
        disp.shutdown()


@pytest.mark.parametrize("wrap", [ThreadSafeGPUDispatcher, CUDAOwnerDispatcher],
                         ids=["ThreadSafeGPUDispatcher", "CUDAOwnerDispatcher"])
@pytest.mark.usefixtures("trace")
def test_a_wrapper_over_an_inner_without_the_keyword_does_not_claim_it(monkeypatch, wrap) -> None:
    """Traced search through a wrapper whose inner (an eager evaluator) does not
    take ``live_rows``: a wrapper that claimed it anyway raises TypeError here."""
    chain = wrap(LocalModelEvaluator(_model_seeded(), device="cpu"))
    try:
        assert not inference.accepts_live_rows(chain)
        _, drains, _ = _search(chain, 20, monkeypatch, sims=24)
        assert drains
    finally:
        if hasattr(chain, "close"):
            chain.close()


def test_the_multi_gpu_dispatcher_forwards_live_rows_to_the_picked_device(trace, monkeypatch) -> None:
    devs = [DirectGPUEvaluator(_model_seeded(), device="cpu", max_batch=512, use_amp=False,
                               pad_to_compiled_buckets=True) for _ in range(2)]
    chain = MultiGPUDispatcher(devs)
    assert inference.accepts_live_rows(chain)
    assert not inference.accepts_live_rows(MultiGPUDispatcher([devs[0], LocalModelEvaluator(_model(), device="cpu")]))
    chain.evaluate_encoded(_x(16), live_rows=9)
    chain.evaluate_encoded_async(_x(24), live_rows=7)
    assert [(g, r) for g, r, _ in _triples(trace)] == [("dense", 9), ("dense", 7)]
    # And a real search through it (its row cap sizes the leaf buffer, so its
    # drains arrive unpadded): rows still equal the search's own counts.
    _, drains, _ = _search(chain, 20, monkeypatch, sims=24)
    leaf = _triples(trace)[-len(drains):]
    assert [r for _, r, _ in leaf] == [n for n, _ in drains]


def test_the_degrading_dispatcher_forwards_live_rows(trace) -> None:
    from chess_anti_engine.uci.device_guard import DegradingMultiGPUDispatcher

    dev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False)
    chain = DegradingMultiGPUDispatcher([dev], labels=["cpu:0"])
    try:
        assert inference.accepts_live_rows(chain)
        chain.evaluate_encoded(_x(16), live_rows=9)
    finally:
        chain.close()
    assert [(g, r) for g, r, _ in _triples(trace)] == [("dense", 9)]


def test_the_guarded_primary_forwards_live_rows(trace) -> None:
    from chess_anti_engine.uci.device_guard import GuardedPrimaryEvaluator

    dev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False)
    guarded = GuardedPrimaryEvaluator(dev, lambda f: f())
    assert inference.accepts_live_rows(guarded)
    assert not inference.accepts_live_rows(
        GuardedPrimaryEvaluator(LocalModelEvaluator(_model(), device="cpu"), lambda f: f()))
    guarded.evaluate_encoded(_x(16), live_rows=9)
    assert [(g, r) for g, r, _ in _triples(trace)] == [("dense", 9)]


def test_the_coalescer_sums_its_requests_live_rows(trace) -> None:
    dev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False)
    co = BatchCoalescingDispatcher(dev, max_batch=64)
    try:
        assert inference.accepts_live_rows(co)
        co.evaluate_encoded(_x(16), live_rows=9)
        co.evaluate_encoded(_x(5))
    finally:
        co.close()
    assert not inference.accepts_live_rows(
        BatchCoalescingDispatcher(LocalModelEvaluator(_model(), device="cpu")))
    assert [(g, r) for g, r, _ in _triples(trace)] == [("dense", 9), ("dense", 5)]


def test_the_threaded_dispatcher_takes_a_callers_live_rows(trace) -> None:
    disp = ThreadedDispatcher(_model(), device="cpu", max_batch=64, legal_bf16=False)
    try:
        disp.evaluate_encoded(_x(16), live_rows=9)
    finally:
        disp.shutdown()
    assert [r for _, r, _ in _triples(trace)] == [9]


def test_the_cache_is_not_warned_about_and_an_undeclared_evaluator_is(monkeypatch, caplog) -> None:
    import logging

    from chess_anti_engine.inference_cache import EncodedEvalCache

    monkeypatch.setattr(inference, "_FORWARD_SIZE_TRACE", True)
    monkeypatch.setattr(inference, "_live_rows_unsupported_warned", set())
    cache = EncodedEvalCache(DirectGPUEvaluator(_model(), device="cpu", max_batch=16, use_amp=False),
                             max_entries=8)
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.inference"):
        assert inference.forward_live_rows_kwargs(cache, 5) == {}
        assert not caplog.records
        assert inference.forward_live_rows_kwargs(LocalModelEvaluator(_model(), device="cpu"), 5) == {}
    (record,) = caplog.records
    assert "padding as live rows" in record.getMessage()


# --- third fix round: one record per CALL, with the chunking that call ran as ---


def test_a_chunked_call_is_one_record_carrying_its_chunk_count(trace, monkeypatch) -> None:
    """A call above the evaluator's per-forward bound N runs as ceil(rows / N)
    network forwards (#1511's ``forward_chunk_rows``) and is still ONE record:
    live / call rows of the call, plus N and the chunk count."""
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False)
    # The attribute #1511's ``forward_chunk_rows`` constructor argument sets
    # (absent before it: raising=False).
    monkeypatch.setattr(ev, "_forward_chunk_rows", 16, raising=False)
    ev.evaluate_encoded(_x(40))                  # 40 rows / 16 -> 3 forwards
    ev.evaluate_encoded(_x(16))                  # at the bound -> 1
    ev.evaluate_encoded_async(_x(33), live_rows=30)
    recs = _records(trace)
    assert [(r["live_rows"], r["call_rows"], r["chunk_rows"], r["chunks"]) for r in recs] == [
        (40, 40, 16, 3), (16, 16, 16, 1), (30, 33, 16, 3),
    ]


def test_an_unchunked_evaluator_records_no_bound_and_one_forward(trace, monkeypatch) -> None:
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, use_amp=False)
    monkeypatch.setattr(ev, "_forward_chunk_rows", None, raising=False)
    ev.evaluate_encoded(_x(40))
    _aot().evaluate_encoded(_x(5))
    recs = _records(trace)
    assert [(r["chunk_rows"], r["chunks"]) for r in recs] == [(None, 1), (None, 1)]


def test_the_chunk_count_is_the_ceiling() -> None:
    assert [inference._trace_chunks(r, 16) for r in (1, 16, 17, 32, 33, 64)] == [1, 1, 2, 2, 3, 4]
    assert inference._trace_chunks(10_000, None) == 1


def test_switched_off_no_broker_reads_the_live_rows_field(tmp_path, monkeypatch) -> None:
    """With both diagnostics off nothing branches on the slot header's live_rows
    bytes: reading them raises here, and a SlotBroker batch still completes."""
    monkeypatch.setattr(inference, "_FORWARD_SIZE_TRACE", False)
    monkeypatch.setattr(inference, "_BATCH_TRACE_ENABLED", False)

    def boom(_self):
        raise AssertionError("slot live_rows read with the diagnostics off")

    monkeypatch.setattr(_InferenceSlot, "live_rows", property(boom))
    publish = tmp_path / "publish"
    publish.mkdir()
    broker = SlotBroker(publish_dir=publish, num_slots=1, max_batch_per_slot=8, device="cpu",
                        compile_inference=False, batch_wait_ms=0.0,
                        slot_prefix=f"cae-off-{uuid.uuid4().hex}")
    try:
        monkeypatch.setattr(SlotBroker, "_ensure_model", lambda _self: None)
        broker._model = _TinyPolicy().eval()
        broker._model_sha = "test"
        slot = broker._slots[0]
        slot.batch_size = 3
        slot.request_mode = _MODE_DENSE_F32
        slot.request_id = 60
        slot.state = _STATE_REQUEST
        assert broker._process_batch([slot]) == 0
        assert slot.state == _STATE_RESPONSE
    finally:
        broker.shutdown()


@pytest.mark.parametrize(
    ("q_scale", "fpu", "chunk_rows", "pad"),
    [(0.5, 0.0, 2, True), (0.0, 0.2, None, False)],
    ids=["q-scale-chunked", "fpu-unchunked"],
)
def test_python_gumbel_traces_volatility_root_and_leaf_calls(
    hist, trace, monkeypatch, q_scale: float, fpu: float,
    chunk_rows: int | None, pad: bool,
) -> None:
    from chess_anti_engine.mcts.gumbel import run_gumbel_root_many

    class VolatilitySpy(DirectGPUEvaluator):
        def evaluate_encoded_with_volatility(
            self, x: np.ndarray, relations: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            calls.append(int(x.shape[0]))
            return super().evaluate_encoded_with_volatility(x, relations)

    calls: list[int] = []
    forwards: list[int] = []
    model = _model_seeded()
    ev = VolatilitySpy(
        model, device="cpu", use_amp=False, max_batch=16,
        pad_to_compiled_buckets=pad, forward_chunk_rows=chunk_rows,
    )
    cfg = GumbelConfig(
        simulations=8, topk=4, add_noise=False,
        input_extra_features="v1", volatility_q_scale=q_scale, volatility_fpu=fpu,
    )

    def count_forward(_model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        forwards.append(int(inputs[0].shape[0]))

    def search() -> object:
        return run_gumbel_root_many(
            None, [chess.Board() for _ in range(3)], device="cpu",
            rng=np.random.default_rng(73), cfg=cfg, evaluator=ev, return_diagnostics=True,
        )

    hook = model.register_forward_pre_hook(count_forward)
    try:
        with monkeypatch.context() as off:
            off.setattr(inference, "_FORWARD_SIZE_TRACE", False)
            off.setattr(inference, "_BUCKET_HIST_ENABLED", False)
            off.setattr(inference, "_BATCH_TRACE_ENABLED", False)
            off_result = _digest(search())
        assert _records(trace) == []
        assert hist["batch"] == {}
        off_calls, off_forwards = calls[:], forwards[:]
        calls.clear()
        forwards.clear()
        assert _digest(search()) == off_result
    finally:
        hook.remove()

    assert calls == off_calls
    assert forwards == off_forwards
    assert len(calls) > 1  # one root call, then actual leaf calls
    assert calls[0] == 3
    bound = chunk_rows or max(calls)
    expected_forwards = [
        min(rows - lo, bound) for rows in calls for lo in range(0, rows, bound)
    ]
    assert forwards == expected_forwards
    recs = _records(trace)
    assert [
        (r["graph"], r["live_rows"], r["call_rows"], r["chunk_rows"], r["chunks"])
        for r in recs
    ] == [("dense", n, n, chunk_rows, (n + bound - 1) // bound) for n in calls]
    assert hist["batch"] == {n: calls.count(n) for n in set(calls)}
