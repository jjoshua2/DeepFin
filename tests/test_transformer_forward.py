from typing import Any, cast

import torch
from torch import nn

from chess_anti_engine.model import reinit_volatility_head_parameters_
from chess_anti_engine.model.transformer import (
    ChessNet,
    TransformerConfig,
    VolatilityHead,
)


def test_transformer_forward_shapes():
    cfg = TransformerConfig(in_planes=146, embed_dim=64, num_layers=2, num_heads=4, use_smolgen=True, use_nla=False)
    m = ChessNet(cfg)
    x = torch.randn(3, 146, 8, 8)
    out = m(x)
    assert out["policy_own"].shape == (3, 64 * 73)
    assert out["wdl"].shape == (3, 3)
    assert out["sf_eval"].shape == (3, 3)
    assert out["categorical"].shape == (3, 32)
    assert out["volatility"].shape == (3, 3)
    assert out["sf_volatility"].shape == (3, 3)
    assert out["moves_left"].shape == (3, 1)


def test_transformer_nla_toggle():
    cfg = TransformerConfig(in_planes=146, embed_dim=64, num_layers=1, num_heads=4, use_smolgen=False, use_nla=True)
    m = ChessNet(cfg)
    x = torch.randn(1, 146, 8, 8)
    out = m(x)
    assert out["policy_own"].shape[-1] == 64 * 73


def test_volatility_head_keeps_gradient_for_negative_preactivations():
    head = VolatilityHead(embed_dim=4)
    first = head.net[0]
    last = head.net[2]
    assert isinstance(first, nn.Linear)
    assert isinstance(last, nn.Linear)
    with torch.no_grad():
        first.weight.zero_()
        first.bias.zero_()
        last.weight.zero_()
        last.bias.fill_(-10.0)

    x = torch.randn(2, 64, 4)
    out = head(x)
    loss = out.sum()
    loss.backward()

    assert torch.all(out > 0.0)
    assert last.bias.grad is not None
    assert torch.count_nonzero(last.bias.grad).item() == last.bias.numel()


def test_reinit_volatility_heads_only_touches_volatility_modules():
    cfg = TransformerConfig(in_planes=146, embed_dim=64, num_layers=1, num_heads=4, use_smolgen=False, use_nla=False)
    m = ChessNet(cfg)
    policy_own = cast(Any, m.policy_own)
    value_wdl = cast(Any, m.value_wdl)
    volatility = m.volatility
    sf_volatility = m.sf_volatility
    assert isinstance(volatility, VolatilityHead)
    assert isinstance(sf_volatility, VolatilityHead)
    before_policy = policy_own.q.weight.detach().clone()
    before_value_layer = value_wdl.net[0]
    assert isinstance(before_value_layer, nn.Linear)
    before_value = before_value_layer.weight.detach().clone()
    with torch.no_grad():
        for p in volatility.parameters():
            p.fill_(3.0)
        for p in sf_volatility.parameters():
            p.fill_(4.0)

    changed = reinit_volatility_head_parameters_(m)

    assert changed == ["volatility", "sf_volatility"]
    assert torch.equal(before_policy, policy_own.q.weight)
    assert torch.equal(before_value, before_value_layer.weight)
    volatility_layer = volatility.net[0]
    sf_volatility_layer = sf_volatility.net[0]
    assert isinstance(volatility_layer, nn.Linear)
    assert isinstance(sf_volatility_layer, nn.Linear)
    assert not torch.all(volatility_layer.weight == 3.0)
    assert not torch.all(sf_volatility_layer.weight == 4.0)


def test_reinit_volatility_heads_restore_near_zero_baseline():
    cfg = TransformerConfig(in_planes=146, embed_dim=64, num_layers=1, num_heads=4, use_smolgen=False, use_nla=False)
    m = ChessNet(cfg)
    volatility = m.volatility
    sf_volatility = m.sf_volatility
    assert isinstance(volatility, VolatilityHead)
    assert isinstance(sf_volatility, VolatilityHead)

    with torch.no_grad():
        for p in volatility.parameters():
            p.fill_(3.0)
        for p in sf_volatility.parameters():
            p.fill_(4.0)

    reinit_volatility_head_parameters_(m)

    x = torch.zeros(2, 64, 64)
    out = volatility(x)
    sf_out = sf_volatility(x)

    assert torch.all(out >= 0.0)
    assert torch.all(sf_out >= 0.0)
    assert torch.max(out).item() < 0.05
    assert torch.max(sf_out).item() < 0.05
