import torch

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
    with torch.no_grad():
        head.net[0].weight.zero_()
        head.net[0].bias.zero_()
        head.net[2].weight.zero_()
        head.net[2].bias.fill_(-10.0)

    x = torch.randn(2, 64, 4)
    out = head(x)
    loss = out.sum()
    loss.backward()

    assert torch.all(out > 0.0)
    assert head.net[2].bias.grad is not None
    assert torch.count_nonzero(head.net[2].bias.grad).item() == head.net[2].bias.numel()


def test_reinit_volatility_heads_only_touches_volatility_modules():
    cfg = TransformerConfig(in_planes=146, embed_dim=64, num_layers=1, num_heads=4, use_smolgen=False, use_nla=False)
    m = ChessNet(cfg)
    before_policy = m.policy_own.q.weight.detach().clone()
    before_value = m.value_wdl.net[0].weight.detach().clone()
    with torch.no_grad():
        for p in m.volatility.parameters():
            p.fill_(3.0)
        for p in m.sf_volatility.parameters():
            p.fill_(4.0)

    changed = reinit_volatility_head_parameters_(m)

    assert changed == ["volatility", "sf_volatility"]
    assert torch.equal(before_policy, m.policy_own.q.weight)
    assert torch.equal(before_value, m.value_wdl.net[0].weight)
    assert not torch.all(m.volatility.net[0].weight == 3.0)
    assert not torch.all(m.sf_volatility.net[0].weight == 4.0)


def test_reinit_volatility_heads_restore_near_zero_baseline():
    cfg = TransformerConfig(in_planes=146, embed_dim=64, num_layers=1, num_heads=4, use_smolgen=False, use_nla=False)
    m = ChessNet(cfg)

    with torch.no_grad():
        for p in m.volatility.parameters():
            p.fill_(3.0)
        for p in m.sf_volatility.parameters():
            p.fill_(4.0)

    reinit_volatility_head_parameters_(m)

    x = torch.zeros(2, 64, 64)
    out = m.volatility(x)
    sf_out = m.sf_volatility(x)

    assert torch.all(out >= 0.0)
    assert torch.all(sf_out >= 0.0)
    assert torch.max(out).item() < 0.05
    assert torch.max(sf_out).item() < 0.05
