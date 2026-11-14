"""Every batch shape a compiled UCI evaluator can run is warmed before readyok.

The production factories (`_make_evaluator_factory`, the multi-GPU factory
builder) are driven on CPU with `torch.compile` replaced by the identity, so
the bookkeeping -- bucket padding in `DirectGPUEvaluator`, the warm loops, the
strict completeness check and the post-warmup guard -- is the production code
path while no real compile happens. The reachable set is enumerated from
`compiled_forward_rows`, the same function the evaluator pads through.
"""
from __future__ import annotations

import threading
from typing import Any

import chess
import numpy as np
import pytest
import torch

import chess_anti_engine.uci.__main__ as uci_main
from chess_anti_engine import inference
from chess_anti_engine.inference import (
    _COMPILED_BATCH_BUCKETS,
    DirectGPUEvaluator,
    compiled_forward_rows,
    reachable_forward_rows,
)
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.time_manager import Deadline


def _model() -> torch.nn.Module:
    m = build_model(ModelConfig(
        input_extra_features="v1", embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0,
    ))
    m.eval()
    return m


@pytest.fixture
def no_compile(monkeypatch):
    """torch.compile -> identity: the compiled-path bookkeeping without a compile."""
    monkeypatch.setattr(torch, "compile", lambda module, **_kw: module)


@pytest.fixture
def cold(monkeypatch):
    events: list[tuple[str, str, int]] = []
    monkeypatch.setattr(inference, "_COLD_SHAPE_HOOK", lambda *a: events.append(a))
    return events


def test_padding_function_image_is_the_enumerated_ladder() -> None:
    """Runtime padding and the warm enumeration are one function: for every
    cap, every row count lands on a shape of `reachable_forward_rows`, and that
    set is exactly the buckets below the cap plus the cap (plus 1)."""
    for cap in (64, 100, 300, 512, 1024, 5000, 8192):
        ladder = set(reachable_forward_rows(cap, cap))
        expect = {1, cap} | {b for b in _COMPILED_BATCH_BUCKETS if b < cap}
        assert ladder == expect, cap
        for n in range(1, cap + 1):
            assert compiled_forward_rows(n, cap) in ladder
            assert compiled_forward_rows(n, cap) >= n


def test_padding_evaluator_runs_only_ladder_shapes_and_returns_live_rows() -> None:
    """Drive a padding evaluator through all four entries at every row count:
    each forward lands on a ladder shape and the caller gets exactly its rows,
    equal to an unpadded evaluator's answer for the same input."""
    model = _model()
    padded = DirectGPUEvaluator(model, device="cpu", max_batch=40, n_slots=2,
                                use_amp=False, pad_to_compiled_buckets=True)
    plain = DirectGPUEvaluator(model, device="cpu", max_batch=40, n_slots=2, use_amp=False)
    rng = np.random.default_rng(0)
    for n in (1, 2, 15, 16, 17, 33, 40):
        x = rng.random((n, 146, 8, 8), dtype=np.float32)
        p_pol, p_wdl = padded.evaluate_encoded(x)
        q_pol, q_wdl = plain.evaluate_encoded(x)
        assert p_pol.shape[0] == n
        assert p_wdl.shape[0] == n
        np.testing.assert_allclose(p_pol, q_pol, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(p_wdl, q_wdl, rtol=1e-5, atol=1e-5)
        padded.get_input_buffer(n, slot=1)[:] = x
        pol_t, _wdl_t, _ = padded.evaluate_inplace_async(n, slot=1)
        assert pol_t.shape[0] == n
        np.testing.assert_allclose(pol_t.numpy(), q_pol, rtol=1e-5, atol=1e-5)
        pol_t, _, _ = padded.evaluate_encoded_async(x)
        assert pol_t.shape[0] == n
    assert padded.shape_ledger.seen() <= set(reachable_forward_rows(40, 40))
    assert plain.shape_ledger.seen() == {1, 2, 15, 16, 17, 33, 40}


@pytest.mark.usefixtures("no_compile")
def test_primary_factory_warms_every_reachable_shape(cold) -> None:
    build = uci_main._make_evaluator_factory(
        [_model()], ["cpu"], True, 1, 1, compile_mode="max-autotune",
    )
    chain: Any = build(256)
    try:
        directs = uci_main.direct_evaluators(chain)
        assert len(directs) == 1
        ev = directs[0]
        assert ev.pads_to_compiled_buckets
        want = set(reachable_forward_rows(256, 256))
        assert want == {1, 16, 32, 64, 96, 128, 170, 256}
        assert want <= ev.shape_ledger.seen()
        assert ev.shape_ledger.armed
        assert cold == []
    finally:
        chain.close()


@pytest.mark.usefixtures("no_compile", "cold")
def test_multi_device_primary_chain_warms_every_device() -> None:
    """--no-multi-gpu-pucv routing chain (MultiGPUDispatcher): every shape on
    EVERY device, not just the ones the round-robin happened to visit."""
    build = uci_main._make_evaluator_factory(
        [_model(), _model(), _model()], ["cpu", "cpu", "cpu"], True, 1, 1,
        compile_mode="default",
    )
    chain: Any = build(128)
    try:
        directs = uci_main.direct_evaluators(chain)
        assert len(directs) == 3
        for ev in directs:
            assert set(reachable_forward_rows(128, 128)) <= ev.shape_ledger.seen()
            assert ev.shape_ledger.armed
    finally:
        chain.close()


@pytest.mark.usefixtures("no_compile")
def test_pool_factories_warm_every_reachable_shape_on_every_device(cold) -> None:
    build = uci_main._make_multi_gpu_pucv_factory_builder(
        [_model(), _model()], ["cpu", "cpu"], compile_mode="max-autotune",
    )
    evaluators = [make() for make in build(1024, 300)]
    for ev in evaluators:
        want = set(ev.reachable_forward_rows(300))
        # gather 300 under MaxBatch 1024: the remainders 1..300 pad to these.
        assert want == {1, 16, 32, 64, 96, 128, 170, 256, 340}
        assert want <= ev.shape_ledger.seen()
        assert ev.shape_ledger.armed
    assert cold == []


@pytest.mark.usefixtures("no_compile")
def test_incomplete_warmup_refuses_to_start(monkeypatch) -> None:
    """The strict completeness check is real: a warm loop that skips a shape
    raises instead of arming over a cold shape."""
    real = uci_main.DirectGPUEvaluator.reachable_forward_rows

    def one_more(self, max_rows=None):
        return (*real(self, max_rows), 7)  # a shape no warm loop runs

    monkeypatch.setattr(uci_main.DirectGPUEvaluator, "reachable_forward_rows", one_more)
    calls = {"n": 0}
    real_verify = uci_main._verify_and_arm

    def verify(directs, max_rows_by_eval):
        calls["n"] += 1
        real_verify(directs, max_rows_by_eval)

    monkeypatch.setattr(uci_main, "_verify_and_arm", verify)
    monkeypatch.setattr(uci_main, "_warmup_compiled_buckets", _warm_without_shape_7)
    build = uci_main._make_evaluator_factory([_model()], ["cpu"], True, 1, 1, compile_mode="default")
    with pytest.raises(RuntimeError, match="warmup incomplete"):
        build(64)
    assert calls["n"] == 1


def _warm_without_shape_7(_chain, directs, **_kwargs):
    for ev in directs:
        for rows in ev.reachable_forward_rows():
            if rows != 7:
                ev.evaluate_encoded(np.zeros((rows, 146, 8, 8), dtype=np.float32))
    uci_main._verify_and_arm(directs, lambda _ev: None)


def test_guard_reports_a_first_seen_shape_after_warmup_once(cold) -> None:
    ev = DirectGPUEvaluator(_model(), device="cpu", max_batch=64, n_slots=2,
                            use_amp=False, pad_to_compiled_buckets=True)
    for rows in (1, 16, 32):
        ev.evaluate_encoded(np.zeros((rows, 146, 8, 8), dtype=np.float32))
    ev.shape_ledger.arm()
    ev.evaluate_encoded(np.zeros((10, 146, 8, 8), dtype=np.float32))  # pads to 16: warm
    assert cold == []
    ev.evaluate_encoded(np.zeros((40, 146, 8, 8), dtype=np.float32))  # pads to 64: cold
    ev.get_input_buffer(50, slot=0)[:] = 0.0
    ev.evaluate_inplace_async(50, slot=0)  # 64 again: reported once only
    assert cold == [("cpu", "sync", 64)]


def test_guard_prints_an_info_string_through_the_engine_hook(capsys) -> None:
    from chess_anti_engine.uci import engine as engine_module

    engine_module._report_cold_shape("cuda:2", "async", 340)
    out = capsys.readouterr().out
    assert out.startswith("info string WARNING first-seen batch shape after warmup: device=cuda:2")
    assert "batch=340" in out


@pytest.mark.usefixtures("no_compile")
def test_real_searches_after_warmup_meet_no_cold_shape(cold) -> None:
    """End to end: after the production warmup, classic Gumbel and the
    multi-GPU pool search several positions without one first-seen shape
    (before this change the same searches met 32/64/96/170/384 on Gumbel and
    arbitrary budget remainders on the pool)."""
    primary = uci_main._make_evaluator_factory(
        [_model()], ["cpu"], True, 1, 1, compile_mode="max-autotune",
    )(512)
    worker = SearchWorker(
        primary, device="cpu",
        gumbel_cfg=GumbelConfig(input_extra_features="v1", simulations=512, add_noise=False),
        chunk_sims=512,
    )
    fens = [
        chess.STARTING_FEN,
        "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 4 8",
        "8/5pk1/6p1/8/3R4/6P1/5PK1/3r4 w - - 0 40",
    ]
    try:
        for fen in fens:
            worker.reset_tree()
            worker.run(chess.Board(fen), stop_event=threading.Event(),
                       deadline=Deadline(None), max_nodes=1500)
        pool_build = uci_main._make_multi_gpu_pucv_factory_builder(
            [_model(), _model()], ["cpu", "cpu"], compile_mode="max-autotune",
        )
        worker.install_multi_gpu_pucv(pool_build(512, 200), gather=200, as_factories=True)
        for fen in fens:
            worker.reset_tree()
            worker.run(chess.Board(fen), stop_event=threading.Event(),
                       deadline=Deadline(None), max_nodes=1777)
        assert cold == []
    finally:
        worker.close()
