"""PR1110's tri-state readiness must survive the AOT ownership receipt."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import pytest
import torch

import chess_anti_engine.inference as inf
from chess_anti_engine.async_safety import Readiness, completion_readiness
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.selfplay import network_turn
from chess_anti_engine.selfplay.config import GameConfig
from tests.test_aot_evaluator_contracts import Package, evaluator


@pytest.mark.parametrize("state", ["ready", "not_ready", "unknown", "query_failure"])
def test_aot_receipt_preserves_readiness(monkeypatch, tmp_path, state):
    ev, _ = evaluator(monkeypatch, tmp_path)

    class Event:
        calls = 0

        def synchronize(self):
            self.calls += 1

    event = Event()
    if state != "unknown":
        def query():
            if state == "query_failure":
                raise RuntimeError("readiness is unavailable")
            return state == "ready"
        monkeypatch.setattr(event, "query", query, raising=False)

    receipt = ev._completion(event, None)
    expected = {
        "ready": Readiness.READY,
        "not_ready": Readiness.NOT_READY,
        "unknown": Readiness.UNKNOWN,
        "query_failure": Readiness.UNKNOWN,
    }[state]
    assert completion_readiness(receipt) is expected
    assert event.calls == 0  # Querying never drains or releases the receipt.
    assert receipt in ev._pending
    receipt.synchronize()
    assert event.calls == 1
    assert receipt not in ev._pending


def test_failed_pending_drain_prevents_rebind_and_retains_receipt(monkeypatch, tmp_path):
    ev, packages = evaluator(monkeypatch, tmp_path)

    class Event:
        calls = 0

        def synchronize(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("pending drain failed")

    class Stream:
        calls = 0

        def synchronize(self):
            self.calls += 1

    event, stream = Event(), Stream()
    receipt = ev._completion(event, stream)
    with pytest.raises(RuntimeError, match="pending drain failed"):
        ev.load_weights({"weight": torch.tensor([2.0])})
    assert stream.calls == 1
    assert receipt in ev._pending
    assert packages[16].weight == 1.0
    ev.load_weights({"weight": torch.tensor([2.0])})
    assert event.calls == 2
    assert not ev._pending
    assert packages[16].weight == 2.0


def test_failed_cuda_rebind_sync_poisons_evaluator(monkeypatch, tmp_path):
    ev, packages = evaluator(monkeypatch, tmp_path)
    ev.device = "cuda"  # Device operations below are all inert CPU fakes.

    class Stream:
        def synchronize(self):
            raise RuntimeError("rebind copies did not drain")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream())
    monkeypatch.setattr("chess_anti_engine.inference.build_aot_constants", lambda source, *args, **kwargs: source)
    with pytest.raises(RuntimeError, match="rebind copies did not drain"):
        ev.load_weights({"weight": torch.tensor([2.0])})
    assert packages[16].weight == 2.0
    with pytest.raises(RuntimeError, match="poisoned"):
        ev.get_input_buffer(1)
    with pytest.raises(RuntimeError, match="poisoned"):
        ev.load_weights({"weight": torch.tensor([3.0])})


@pytest.mark.parametrize("bsz", [1, 16])
def test_real_selfplay_root_fallback_leaves_aot_padding_to_evaluator(monkeypatch, tmp_path, bsz):
    package = Package(16)
    monkeypatch.setattr(inf, "load_aot_packages", lambda *args, **kwargs: {16: package})
    ev = inf.AOTEvaluator(tmp_path, device="cpu", max_batch=512, input_planes=146)
    submitted = []
    original = ev.evaluate_encoded

    def evaluate(x, **kwargs):
        submitted.append(len(x))
        return original(x, **kwargs)

    monkeypatch.setattr(ev, "evaluate_encoded", evaluate)
    state: Any = SimpleNamespace(
        evaluator=ev,
        cboards=[CBoard.from_board(chess.Board()) for _ in range(bsz)],
        game=GameConfig(input_extra_features="v1"), has_c_ply=False,
        batch_enc_146=None, batch_enc_146_bf16=None,
        batch_enc_146_lc0_root=None, batch_enc_146_lc0_root_bf16=None,
        batch_enc_146_lc0_root_legacy_meta=None,
        batch_enc_146_lc0_root_legacy_meta_bf16=None,
    )
    encoded, policy, wdl, probabilities = network_turn._evaluate_root_batch(state, list(range(bsz)))
    assert submitted == [bsz]
    assert encoded is not None
    assert encoded.shape == (bsz, 146, 8, 8)
    assert policy.shape == (bsz, inf.POLICY_SIZE)
    assert wdl.shape == probabilities.shape == (bsz, 3)
    assert np.all(policy == 1.0)
    assert np.allclose(probabilities, 1.0 / 3.0)
    assert package.calls == 1
