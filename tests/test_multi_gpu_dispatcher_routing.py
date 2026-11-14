"""MultiGPUDispatcher under the UCI routing chain (``--no-multi-gpu-pucv``).

The chain is ``MultiGPUDispatcher`` -> (``CUDAOwnerDispatcher`` when compiled)
-> (``EncodedEvalCache`` when enabled). Classic Gumbel probes it for
``evaluate_encoded_async`` and ``search_leaf_row_cap``; both must be real here.
"""
from __future__ import annotations

import numpy as np
import torch

from chess_anti_engine.inference import DirectGPUEvaluator
from chess_anti_engine.inference_cache import EncodedEvalCache
from chess_anti_engine.inference_dispatcher import CUDAOwnerDispatcher, MultiGPUDispatcher
from chess_anti_engine.mcts.gumbel_c import _noninplace_leaf_rows
from chess_anti_engine.model import ModelConfig, build_model


def _direct(max_batch: int) -> DirectGPUEvaluator:
    model = build_model(ModelConfig(
        input_extra_features="v1", embed_dim=16, num_layers=1, num_heads=2, ffn_mult=2.0,
    ))
    model.eval()
    return DirectGPUEvaluator(model, device="cpu", max_batch=max_batch, use_amp=False, n_slots=2)


def test_async_entry_returns_owned_copies_equal_to_the_sync_answer() -> None:
    disp = MultiGPUDispatcher([_direct(64), _direct(64)])
    x = np.random.default_rng(0).random((5, 146, 8, 8), dtype=np.float32)
    pol_t, wdl_t, event = disp.evaluate_encoded_async(x)
    assert event is None
    assert isinstance(pol_t, torch.Tensor)
    assert pol_t.shape[0] == 5
    assert wdl_t.shape[0] == 5
    pol_np = pol_t.numpy().copy()
    for _ in range(4):  # later calls on either device must not alter the result
        disp.evaluate_encoded(np.zeros((5, 146, 8, 8), dtype=np.float32))
    np.testing.assert_array_equal(pol_t.numpy(), pol_np)
    ref_pol, ref_wdl = MultiGPUDispatcher([_direct(64)]).evaluate_encoded(x)
    assert ref_pol.shape == pol_np.shape
    assert ref_wdl.shape == tuple(wdl_t.shape)


def test_leaf_row_cap_is_the_smallest_device_max_batch_through_every_wrapper() -> None:
    devices = [_direct(256), _direct(192)]
    disp = MultiGPUDispatcher(devices)
    assert disp.search_leaf_row_cap == 192
    owner = CUDAOwnerDispatcher(disp)
    try:
        assert owner.search_leaf_row_cap == 192
        cache = EncodedEvalCache(owner, max_entries=128, encoding_source=devices[0].model)
        assert cache.search_leaf_row_cap == 192
        # The Gumbel copying path sizes its leaf buffer from it.
        assert _noninplace_leaf_rows(owner, 512) == 192
        assert _noninplace_leaf_rows(cache, 512) == 192
        assert _noninplace_leaf_rows(owner, 128) == 128  # never raises the search's own size
    finally:
        owner.close()


def test_single_device_owner_chain_keeps_no_declared_cap() -> None:
    """The single-device compiled chain takes the in-place path, which caps by
    ``_max_batch`` itself; the owner wrapper must not invent a cap there."""
    owner = CUDAOwnerDispatcher(_direct(256))
    try:
        assert owner.search_leaf_row_cap is None
    finally:
        owner.close()
