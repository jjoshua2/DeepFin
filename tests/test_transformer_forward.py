import torch

from chess_anti_engine.model.transformer import ChessNet, TransformerConfig


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
