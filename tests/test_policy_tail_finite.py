"""Finite-order objective against independent integration and rollout enumeration."""
from __future__ import annotations

import itertools

import pytest
import torch

from chess_anti_engine.policy_tail import finite_tail_loss


def _enumerated_loss(logits, rewards, order):
    """Direct Eq. 23: enumerate every ordered iid rollout, without a CDF."""
    p = logits.softmax(-1)[0]
    loss = logits.sum() * 0
    for k in range(1, order + 1):
        best = logits.sum() * 0
        for actions in itertools.product(range(logits.shape[-1]), repeat=k):
            best = best + p[list(actions)].prod() * rewards[0, list(actions)].max()
        loss = loss + (1 - best) / k
    return loss


def test_exact_iid_enumeration_matches_loss_and_gradient():
    logits = torch.tensor([[0.8, -1.4, 0.2]], dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([[0.65, 0.9, 0.1]], dtype=torch.float64)
    legal = torch.ones_like(logits, dtype=torch.bool)
    actual = finite_tail_loss(logits, legal, rewards, order=4)
    expected = _enumerated_loss(logits, rewards, 4)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    actual_grad, = torch.autograd.grad(actual, logits)
    expected_grad, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-13, rtol=1e-13)


def test_midpoint_quadrature_with_uneven_reward_gaps():
    logits = torch.tensor([[0.3, -1.0, 0.8, 0.2]], dtype=torch.float64)
    rewards = torch.tensor([[0.08, 0.31, 0.31, 0.91]], dtype=torch.float64)
    thresholds = (torch.arange(10000, dtype=torch.float64) + 0.5) / 10000
    success = ((rewards[0, None, :] > thresholds[:, None]) * logits.softmax(-1)).sum(-1)
    expected = torch.stack([(1 - success).pow(k) / k for k in range(1, 33)]).sum(0).mean()
    actual = finite_tail_loss(logits, torch.ones_like(logits, dtype=torch.bool), rewards)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_order_one_is_expected_reward_and_rewards_are_detached():
    logits = torch.tensor([[0.5, -0.8, 0.1], [1.2, 0.0, -0.1]],
                          dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([[0.35, 0.4, 0.55], [0.1, 0.9, 0.2]],
                           dtype=torch.float64, requires_grad=True)
    legal = torch.tensor([[True, True, True], [True, False, True]])
    actual = finite_tail_loss(logits, legal, rewards, order=1, reduction="none")
    expected = 1 - (logits.masked_fill(~legal, -torch.inf).softmax(-1) * rewards.detach()).sum(-1)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert rewards.grad is None
    expected_grad, = torch.autograd.grad(expected.sum(), logits)
    torch.testing.assert_close(logits.grad, expected_grad)
    torch.testing.assert_close(finite_tail_loss(logits, legal, rewards, order=1), expected.mean())
    torch.testing.assert_close(finite_tail_loss(logits, legal, rewards, order=1, reduction="sum"), expected.sum())


@pytest.mark.parametrize("order", [1, 4, 32])
def test_binary_best_is_cross_entropy_with_analytical_scalar(order):
    logits = torch.tensor([[0.5, -2.2, 1.4]], dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    loss = finite_tail_loss(logits, torch.ones_like(logits, dtype=torch.bool), rewards, order=order)
    ce = -logits.log_softmax(-1)[0, 1]
    p = logits.softmax(-1)[0, 1].detach()
    grad, = torch.autograd.grad(loss, logits)
    ce_grad, = torch.autograd.grad(ce, logits)
    torch.testing.assert_close(grad, (1 - (1 - p)**order) * ce_grad)


def test_ties_illegal_values_and_action_permutation_do_not_change_result():
    logits = torch.tensor([[0.1, 0.4, -1.0, float("nan"), float("inf")]],
                          dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([[0.2, 0.8, 0.8, float("nan"), -999.0]], dtype=torch.float64)
    legal = torch.tensor([[True, True, True, False, False]])
    actual = finite_tail_loss(logits, legal, rewards)
    compact = finite_tail_loss(logits[:, :3], legal[:, :3], rewards[:, :3])
    permutation = [4, 2, 0, 3, 1]
    permuted = finite_tail_loss(logits[:, permutation], legal[:, permutation], rewards[:, permutation])
    torch.testing.assert_close(actual, compact)
    torch.testing.assert_close(actual, permuted)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:, 3:]) == 0


@pytest.mark.parametrize("value", [0.0, 0.37, 1.0])
def test_equal_rewards_keep_unreachable_constant_and_exact_zero_gradient(value):
    logits = torch.tensor([[0.4, -7.0, 1.1]], dtype=torch.float64, requires_grad=True)
    legal = torch.tensor([[True, False, True]])
    rewards = torch.full_like(logits, value)
    loss = finite_tail_loss(logits, legal, rewards)
    assert loss.item() == pytest.approx((1 - value) * sum(1 / k for k in range(1, 33)))
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def test_graded_tail_direction_differs_from_expected_reward():
    # Medium reward is above the mean but below the rare best outcome. Higher
    # order reverses its gradient rather than merely multiplying the mean loss.
    logits = torch.tensor([[0.05, 0.94, 0.01]], dtype=torch.float64).log().requires_grad_()
    rewards = torch.tensor([[0.0, 0.6, 1.0]], dtype=torch.float64)
    legal = torch.ones_like(logits, dtype=torch.bool)
    mean_grad, = torch.autograd.grad(finite_tail_loss(logits, legal, rewards, order=1), logits)
    tail_grad, = torch.autograd.grad(finite_tail_loss(logits, legal, rewards, order=32), logits)
    assert mean_grad[0, 1] < 0 < tail_grad[0, 1]
    assert tail_grad[0, 2] < mean_grad[0, 2] < 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_mixed_precision_extreme_logits_are_finite(dtype):
    logits = torch.tensor([[0.0, -20.0, -1000.0, float("nan")]], dtype=dtype, requires_grad=True)
    rewards = torch.tensor([[0.1, 0.7, 1.0, float("nan")]], dtype=dtype)
    legal = torch.tensor([[True, True, True, False]])
    loss = finite_tail_loss(logits, legal, rewards)
    assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 3] == 0


@pytest.mark.parametrize("rewards", [[-0.1, 1.0], [0.0, 1.01], [float("nan"), 0.5], [0.0, float("inf")]])
def test_invalid_legal_reward_rejected(rewards):
    with pytest.raises(ValueError, match="legal rewards"):
        finite_tail_loss(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool), torch.tensor([rewards]))


@pytest.mark.parametrize("order", [0, -1, 1.5, True])
def test_invalid_order_rejected(order):
    with pytest.raises(ValueError, match="order"):
        finite_tail_loss(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2), order=order)


def test_gradcheck_uneven_rewards_with_ties():
    logits = torch.tensor([[0.2, -0.5, 1.1, 0.4]], dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([[0.4, 0.1, 0.4, 0.95]], dtype=torch.float64)
    legal = torch.ones_like(logits, dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda x: finite_tail_loss(x, legal, rewards), (logits,))
