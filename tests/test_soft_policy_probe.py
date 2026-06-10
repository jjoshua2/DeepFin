"""Soft-policy probe metrics + soft_policy_min_tv loss mask.

The default (soft_policy_min_tv = 0.0) must be bitwise-identical to the
pre-flag loss; the threshold only ever drops near-duplicate soft targets.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from chess_anti_engine.train.losses import compute_loss
from scripts.probe_policy_targets import _row_metrics, _summarize


def test_row_metrics_goldens():
    p = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32)
    q = np.array([[1.0, 0.0, 0.0], [0.25, 0.25, 0.5]], dtype=np.float32)
    m = _row_metrics(p, q)
    assert m["tv"][0] == pytest.approx(0.0, abs=1e-9)
    assert m["tv"][1] == pytest.approx(0.5)          # 0.5*(0.25+0.25+0.5)
    assert m["kl_pq"][0] == pytest.approx(0.0, abs=1e-6)
    # KL(p||q) row1 = 0.5*ln(2) + 0.5*ln(2) = ln 2
    assert m["kl_pq"][1] == pytest.approx(np.log(2.0), rel=1e-5)
    assert m["argmax_agree"].tolist() == [1.0, 0.0]  # q row1 argmax moves to index 2
    assert m["valid"].all()


def test_summarize_fractions():
    data = {
        "tv": np.array([0.0, 0.005, 0.2, 0.5]),
        "kl_pq": np.zeros(4),
        "kl_qp": np.zeros(4),
        "argmax_agree": np.array([1.0, 1.0, 0.0, 1.0]),
    }
    s = _summarize(np.ones(4, dtype=bool), data)
    assert s is not None
    assert s["n"] == 4
    assert s["frac_identical_tv_lt_0.01"] == pytest.approx(0.5)
    assert s["frac_tv_gt_0.1"] == pytest.approx(0.5)
    assert s["argmax_agreement"] == pytest.approx(0.75)


def _loss_batch(*, divergent_soft_row1: bool) -> tuple[dict, dict]:
    torch.manual_seed(0)
    n, policy_size = 2, 4672
    outputs = {
        "policy_own": torch.randn(n, policy_size),
        "policy_soft": torch.randn(n, policy_size),
        "policy_sf": torch.randn(n, policy_size),
        "policy_future": torch.randn(n, policy_size),
        "wdl": torch.randn(n, 3),
        "sf_eval": torch.randn(n, 3),
        "categorical": torch.randn(n, 32),
        "volatility": torch.randn(n),
        "sf_volatility": torch.randn(n),
        "moves_left": torch.randn(n, 1),
    }
    pol = torch.zeros(n, policy_size)
    pol[0, 10] = 1.0
    pol[1, 20] = 0.6
    pol[1, 21] = 0.4
    soft = pol.clone()
    if divergent_soft_row1:
        # row 0 stays identical (TV = 0); row 1 diverges (TV = 0.2)
        soft[1, 20] = 0.4
        soft[1, 21] = 0.6
    batch = {
        "x": torch.zeros(n, 1),
        "policy_t": pol,
        "policy_soft_t": soft,
        "has_policy_soft": torch.ones(n),
        "wdl_t": torch.tensor([0, 2]),
        "categorical_t": torch.zeros(n, 32),
        "has_categorical": torch.ones(n),
    }
    return outputs, batch


def test_loss_bitwise_unchanged_at_default():
    outputs, batch = _loss_batch(divergent_soft_row1=True)
    base = compute_loss(outputs, batch)
    flagged = compute_loss(outputs, batch, soft_policy_min_tv=0.0)
    assert base.keys() == flagged.keys()
    for k in base:
        assert torch.equal(base[k], flagged[k]), f"{k} changed at default flag value"


def test_min_tv_masks_only_near_identical_rows():
    outputs, batch = _loss_batch(divergent_soft_row1=True)
    base = compute_loss(outputs, batch)
    masked = compute_loss(outputs, batch, soft_policy_min_tv=0.01)
    # row 0 (TV=0) is dropped, row 1 (TV=0.2) kept -> soft CE becomes the
    # row-1-only mean, which differs from the two-row mean.
    assert not torch.equal(base["soft_policy_ce"], masked["soft_policy_ce"])

    # with BOTH rows identical, the soft loss masks to zero entirely
    outputs2, batch2 = _loss_batch(divergent_soft_row1=False)
    fully_masked = compute_loss(outputs2, batch2, soft_policy_min_tv=0.01)
    assert float(fully_masked["soft_policy_ce"]) == pytest.approx(0.0, abs=1e-9)

    # and a threshold above the divergence masks everything
    all_masked = compute_loss(outputs, batch, soft_policy_min_tv=0.5)
    assert float(all_masked["soft_policy_ce"]) == pytest.approx(0.0, abs=1e-9)


def test_probe_end_to_end_on_synthetic_shard(tmp_path):
    from chess_anti_engine.replay.buffer import ReplaySample
    from chess_anti_engine.replay.shard import ShardMeta, samples_to_arrays, save_local_shard_arrays
    from scripts.probe_policy_targets import _collect

    rng = np.random.default_rng(0)
    samples = []
    for i in range(8):
        pol = np.zeros(4672, dtype=np.float32)
        pol[i] = 0.7
        pol[i + 1] = 0.3
        soft = pol.copy() if i % 2 == 0 else np.roll(pol, 1) * 0.5 + pol * 0.5
        samples.append(ReplaySample(
            x=rng.random((146, 8, 8)).astype(np.float32),
            policy_target=pol, wdl_target=1,
            policy_soft_target=soft.astype(np.float32),
            is_selfplay=bool(i % 2),
        ))
    arrs = samples_to_arrays(samples)
    save_local_shard_arrays(
        tmp_path / "shard_000000.zarr", arrs=arrs,
        meta=ShardMeta(positions=len(samples)),
    )
    data = _collect(tmp_path, positions=100)
    assert data["tv"].shape[0] == 8
    even_rows_tv = data["tv"][data["source"] == 1]  # curriculum == not selfplay
    assert np.all(even_rows_tv < 1e-6)              # identical soft targets
