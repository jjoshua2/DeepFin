"""The Gumbel sync drain sends an evaluator that pads internally its live rows only.

The broker slot clients (`SlotInferenceClient` / `MultiSlotInferenceClient`)
declare ``pads_batches_internally``: the broker re-pads the summed total of every
slot, so the rows the search padded each drain up to bought no shape stability
and cost the shared-memory write, the H2D copy and the forward. At the live
yaml's ``gumbel_policy_temp: 1.5`` the legal-bf16 leaf arm is off and every drain
takes the sync arm, so that padding reached the broker (24% of the rows sent in
the #1507 review's stub). These tests drive a client-shaped evaluator through a
real CPU search.
"""
from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine import inference
from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.mcts import gumbel_c
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.model import ModelConfig, build_model

_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "8/8/4k3/8/2P5/8/4K3/8 w - - 0 1",
    "r3k2r/ppp2ppp/2n5/3qp3/3P4/2P2N2/PP3PPP/R2QKB1R b KQkq - 0 9",
]


def _model() -> torch.nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(7)
        return build_model(ModelConfig(
            input_extra_features="v1", embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0,
        )).eval()


class _Client:
    """The slot clients' surface: sync dense + legal entries, no async / in-place."""

    supports_input_bf16_bits = False

    def __init__(self, *, pads_internally: bool) -> None:
        self.pads_batches_internally = pads_internally
        self.inner = DirectGPUEvaluator(_model(), device="cpu", max_batch=1024, use_amp=False)
        self.sent: list[int] = []

    def evaluate_encoded(self, x, relations=None):
        del relations
        self.sent.append(int(x.shape[0]))
        return self.inner.evaluate_encoded(x)

    def evaluate_legal_bf16(self, x, flat, counts):
        self.sent.append(int(x.shape[0]))
        return self.inner.evaluate_legal_bf16(x, flat, counts)


class _AsyncPaddingEvaluator(DirectGPUEvaluator):
    """Declares pads_batches_internally like AOTEvaluator, and has the async API."""

    pads_batches_internally = True

    @property
    def supports_inplace_api(self) -> bool:
        return False


_REAL_RECORD = gumbel_c._record_batch_dup
_SENT_PER_DUP_INSTRUMENT: list[int] = []  # the rows the dup instrument is told were sent


def _search(ev, monkeypatch, *, policy_temp: float, n_boards: int = 20):
    leaves: list[int] = []
    orig = _REAL_RECORD  # never a previous call's spy

    def spy(x, n, p):
        leaves.append(int(n))
        _SENT_PER_DUP_INSTRUMENT.append(int(p))
        return orig(x, n, p)

    monkeypatch.setattr(gumbel_c, "_record_batch_dup", spy)
    monkeypatch.setattr(gumbel_c, "_COMPILED_BATCH_BUCKETS", inference._COMPILED_BATCH_BUCKETS)
    torch.manual_seed(1234)
    res = gumbel_c.run_gumbel_root_many_c(
        None, [chess.Board(_FENS[i % len(_FENS)]) for i in range(n_boards)], device="cpu",
        rng=np.random.default_rng(99),
        cfg=GumbelConfig(simulations=64, topk=8, add_noise=True, input_extra_features="v1",
                         policy_temp=policy_temp),
        evaluator=ev, target_batch=32,
    )
    return res, leaves


@pytest.mark.parametrize("n_boards", [20, 66])
def test_a_padding_client_is_sent_live_rows_and_searches_identically(monkeypatch, n_boards) -> None:
    live, padded = _Client(pads_internally=True), _Client(pads_internally=False)
    res_live, leaves = _search(live, monkeypatch, policy_temp=1.5, n_boards=n_boards)
    res_pad, leaves_pad = _search(padded, monkeypatch, policy_temp=1.5, n_boards=n_boards)
    assert leaves == leaves_pad
    assert live.sent[-len(leaves):] == leaves                       # live rows only
    assert sum(padded.sent[-len(leaves):]) > sum(leaves)            # the old arm padded (non-vacuous)
    # Same search: the padding rows never fed back into the tree.
    probs_l, actions_l, values_l, *_ = res_live
    probs_p, actions_p, values_p, *_ = res_pad
    assert actions_l == actions_p
    assert values_l == values_p
    for a, b in zip(probs_l, probs_p, strict=True):
        np.testing.assert_array_equal(a, b)


def test_an_async_evaluator_that_pads_internally_keeps_its_padded_slice(monkeypatch) -> None:
    """AOTEvaluator declares pads_batches_internally too, but takes the async
    arm, whose compiled bucket shapes need the padded slice: unchanged."""
    ev = _AsyncPaddingEvaluator(_model(), device="cpu", max_batch=1024, n_slots=1, use_amp=False,
                                pad_to_compiled_buckets=True)
    sent: list[int] = []
    real = ev.evaluate_encoded_async

    def spy(x, *a, **k):
        sent.append(int(x.shape[0]))
        return real(x, *a, **k)

    monkeypatch.setattr(ev, "evaluate_encoded_async", spy)
    _SENT_PER_DUP_INSTRUMENT.clear()
    _, leaves = _search(ev, monkeypatch, policy_temp=1.5)
    leaf_sent = sent[-len(leaves):]
    assert any(s > n for s, n in zip(leaf_sent, leaves, strict=True))
    # ...and the duplicate-row instrument is told the padded count it really sent.
    assert _SENT_PER_DUP_INSTRUMENT[-len(leaves):] == leaf_sent


def test_policy_temp_one_is_untouched(monkeypatch) -> None:
    """At policy_temp 1.0 the legal-bf16 arm already sends live rows."""
    live = _Client(pads_internally=True)
    _, leaves = _search(live, monkeypatch, policy_temp=1.0)
    assert live.sent[-len(leaves):] == leaves
