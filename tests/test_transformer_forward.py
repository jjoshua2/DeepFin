import warnings
from typing import Any, cast

import pytest
import torch
from torch import nn

from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.model import (
    ModelConfig,
    build_model,
    load_state_dict_tolerant,
    model_config_from_manifest_dict,
    model_config_to_manifest_dict,
    reinit_volatility_head_parameters_,
)
from chess_anti_engine.model.transformer import (
    ChessNet,
    TransformerConfig,
    VolatilityHead,
    _rmsnorm,
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


def test_build_model_attaches_runtime_encoding_metadata():
    m = build_model(
        ModelConfig(
            kind="transformer",
            embed_dim=64,
            num_layers=1,
            num_heads=4,
            use_smolgen=False,
            policy_encoding="lc0_1858",
            input_history_encoding="root_meta",
        )
    )

    assert getattr(m, "policy_encoding") == "lc0_1858"
    assert getattr(m, "input_history_encoding") == "lc0_root_legacy_meta"


def test_transformer_uses_per_layer_ffn_multipliers():
    cfg = TransformerConfig(
        in_planes=146,
        embed_dim=32,
        num_layers=3,
        num_heads=4,
        ffn_mult=1.0,
        ffn_mult_by_layer=(1.0, 1.5, 2.0),
        use_smolgen=False,
    )
    m = ChessNet(cfg)

    hidden_dims = [
        cast(nn.Linear, cast(Any, block).ffn[0]).out_features
        for block in m.blocks
    ]
    assert hidden_dims == [32, 48, 64]
    assert m.ffn_mult_by_layer == (1.0, 1.5, 2.0)


def test_transformer_rejects_invalid_per_layer_ffn_multipliers():
    with pytest.raises(ValueError, match="length"):
        ChessNet(
            TransformerConfig(
                in_planes=146,
                embed_dim=32,
                num_layers=3,
                num_heads=4,
                ffn_mult_by_layer=(1.0, 1.5),
                use_smolgen=False,
            )
        )

    with pytest.raises(ValueError, match="positive finite"):
        ChessNet(
            TransformerConfig(
                in_planes=146,
                embed_dim=32,
                num_layers=2,
                num_heads=4,
                ffn_mult_by_layer=(1.0, 0.0),
                use_smolgen=False,
            )
        )


def test_model_config_manifest_round_trips_per_layer_ffn_multipliers():
    cfg = ModelConfig(
        embed_dim=32,
        num_layers=3,
        num_heads=4,
        ffn_mult=1.0,
        ffn_mult_by_layer=(1.0, 1.25, 1.5),
        use_smolgen=False,
    )

    manifest = model_config_to_manifest_dict(cfg)

    assert manifest["ffn_mult_by_layer"] == [1.0, 1.25, 1.5]
    assert model_config_from_manifest_dict(manifest).ffn_mult_by_layer == (1.0, 1.25, 1.5)


def test_transformer_per_layer_smolgen_shapes_and_shares_gen_projection():
    cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        use_smolgen=True,
        smolgen_mode="per_layer",
    )
    m = ChessNet(cfg)
    assert m.smolgen is None
    assert m.layer_smolgens is not None
    assert len(m.layer_smolgens) == 2
    assert m.layer_smolgens[0].gen_weight is m.layer_smolgens[1].gen_weight

    x = torch.randn(2, 146, 8, 8)
    out = m(x)
    assert out["policy_own"].shape == (2, 64 * 73)
    assert out["wdl"].shape == (2, 3)


def test_transformer_smolgen_relation_knobs_forward():
    cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        use_smolgen=True,
        smolgen_mode="per_layer",
        smolgen_bias_scale="layer_head",
        smolgen_bias_norm="center_rms",
        arc_attention_bias="basic",
        smolgen_relation_basis=True,
        smolgen_relation_norm="branch_center_rms",
        smolgen_relation_coeff_norm="rms",
        smolgen_relation_scale="layer_head",
    )
    m = ChessNet(cfg)
    assert m.smolgen_layer_head_scale is not None
    assert m.smolgen_layer_head_scale.shape == (2, 4)
    assert m.arc_attention_weight is not None
    assert torch.count_nonzero(m.arc_attention_weight) == 0
    assert m.layer_smolgens is not None
    assert m.layer_smolgens[0].relation_weight is not None
    assert m.layer_smolgens[0].relation_norm == "branch_center_rms"
    assert m.layer_smolgens[0].relation_coeff_norm == "rms"
    assert m.layer_smolgens[0].relation_scale_param is not None
    assert m.layer_smolgens[0].relation_scale_param.shape == (4,)

    x = torch.randn(2, 146, 8, 8)
    out = m(x)
    assert out["policy_own"].shape == (2, 64 * 73)
    assert out["wdl"].shape == (2, 3)


def test_transformer_bt4_global_embedding_forward():
    cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        input_global_embedding="bt4_board",
        input_global_embedding_channels=16,
        input_square_embedding="ma_gate",
        input_pos_encoding="arc_adapter",
    )
    m = ChessNet(cfg)
    assert m.global_board_preprocess is not None
    assert m.global_board_preprocess.weight.shape == (64 * 16, 12 * 64)
    assert m.square_embed_gate_mul_raw is not None
    assert m.square_embed_gate_mul_raw.shape == (64, 64)
    assert m.square_embed_gate_add is not None
    assert m.square_embed_gate_add.shape == (64, 64)

    x = torch.randn(2, 146, 8, 8)
    out = m(x)
    assert out["policy_own"].shape == (2, 64 * 73)
    assert out["wdl"].shape == (2, 3)


def test_transformer_bt4_global_adapter_starts_as_noop():
    torch.manual_seed(123)
    base_cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        input_global_embedding="none",
        input_pos_encoding="arc_adapter",
    )
    base = ChessNet(base_cfg)

    torch.manual_seed(123)
    adapter_cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        input_global_embedding="bt4_board_adapter",
        input_global_embedding_channels=16,
        input_pos_encoding="arc_adapter",
    )
    adapter = ChessNet(adapter_cfg)
    assert adapter.global_board_preprocess is not None
    assert adapter.global_board_adapter is not None
    assert adapter.global_board_preprocess.weight.shape == (64 * 16, 12 * 64)
    assert adapter.global_board_adapter.weight.shape == (64, 16)
    assert torch.count_nonzero(adapter.global_board_adapter.weight) == 0

    base_state = base.state_dict()
    adapter_state = adapter.state_dict()
    for name, param in base_state.items():
        assert name in adapter_state
        assert torch.equal(param, adapter_state[name]), name

    x = torch.randn(2, 146, 8, 8)
    base.eval()
    adapter.eval()
    with torch.no_grad():
        base_out = base(x)
        adapter_out = adapter(x)
    assert torch.equal(adapter_out["policy_own"], base_out["policy_own"])
    assert torch.equal(adapter_out["wdl"], base_out["wdl"])


def test_transformer_split_qkv_forward_and_deepnorm_init():
    cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        qkv_projection="split",
        use_deepnorm=True,
    )
    m = ChessNet(cfg)
    block = m.blocks[0]
    assert isinstance(block.q_proj, nn.Linear)
    assert isinstance(block.k_proj, nn.Linear)
    assert isinstance(block.v_proj, nn.Linear)
    assert not hasattr(block, "qkv_proj")

    x = torch.randn(1, 146, 8, 8)
    out = m(x)
    assert out["policy_own"].shape[-1] == 64 * 73


def test_qk_rmsnorm_matches_activation_dtype_under_autocast():
    norm = _rmsnorm(32)
    x = torch.randn(2, 4, 8, 32).to(torch.bfloat16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = norm(x)

    assert y.dtype == torch.bfloat16
    assert not any("Mismatch dtype between input and weight" in str(w.message) for w in caught)


def test_tolerant_load_migrates_fused_qkv_into_split_qkv():
    fused_cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        qkv_projection="fused",
    )
    split_cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        qkv_projection="split",
    )
    fused = ChessNet(fused_cfg).eval()
    split = ChessNet(split_cfg).eval()
    load_state_dict_tolerant(split, fused.state_dict(), label="test")

    x = torch.randn(2, 146, 8, 8)
    with torch.no_grad():
        fused_out = fused(x)
        split_out = split(x)
    assert torch.allclose(split_out["policy_own"], fused_out["policy_own"], atol=1e-6)
    assert torch.allclose(split_out["wdl"], fused_out["wdl"], atol=1e-6)


def test_tolerant_load_migrates_split_qkv_into_fused_qkv():
    split_cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        qkv_projection="split",
    )
    fused_cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        qkv_projection="fused",
    )
    split = ChessNet(split_cfg).eval()
    fused = ChessNet(fused_cfg).eval()
    load_state_dict_tolerant(fused, split.state_dict(), label="test")

    x = torch.randn(2, 146, 8, 8)
    with torch.no_grad():
        split_out = split(x)
        fused_out = fused(x)
    assert torch.allclose(fused_out["policy_own"], split_out["policy_own"], atol=1e-6)
    assert torch.allclose(fused_out["wdl"], split_out["wdl"], atol=1e-6)


def test_tolerant_load_split_qkv_matches_fused_input_gradients():
    fused_cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        qkv_projection="fused",
    )
    split_cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        qkv_projection="split",
    )
    fused = ChessNet(fused_cfg).eval()
    split = ChessNet(split_cfg).eval()
    load_state_dict_tolerant(split, fused.state_dict(), label="test")

    x_fused = torch.randn(2, 146, 8, 8, requires_grad=True)
    x_split = x_fused.detach().clone().requires_grad_(True)
    fused_out = fused(x_fused)
    split_out = split(x_split)
    fused_loss = fused_out["policy_own"].sum() + fused_out["wdl"].square().sum()
    split_loss = split_out["policy_own"].sum() + split_out["wdl"].square().sum()
    fused_loss.backward()
    split_loss.backward()

    assert x_fused.grad is not None
    assert x_split.grad is not None
    assert torch.allclose(x_split.grad, x_fused.grad, atol=1e-6)


def test_tolerant_load_migrates_fused_qkv_into_compiled_split_qkv():
    fused_cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        qkv_projection="fused",
    )
    split_cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        qkv_projection="split",
    )
    fused = ChessNet(fused_cfg).eval()
    split = ChessNet(split_cfg).eval()
    compiled_like = nn.Module()
    compiled_like._orig_mod = split  # type: ignore[attr-defined]

    load_state_dict_tolerant(compiled_like, fused.state_dict(), label="test")

    x = torch.randn(2, 146, 8, 8)
    with torch.no_grad():
        fused_out = fused(x)
        split_out = split(x)
    assert torch.allclose(split_out["policy_own"], fused_out["policy_own"], atol=1e-6)
    assert torch.allclose(split_out["wdl"], fused_out["wdl"], atol=1e-6)


def test_transformer_nla_toggle():
    cfg = TransformerConfig(in_planes=146, embed_dim=64, num_layers=1, num_heads=4, use_smolgen=False, use_nla=True)
    m = ChessNet(cfg)
    x = torch.randn(1, 146, 8, 8)
    out = m(x)
    assert out["policy_own"].shape[-1] == 64 * 73


def test_transformer_compact_policy_encoding_dense_and_legal_shapes():
    cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        use_nla=False,
        policy_encoding="lc0_1858",
    )
    m = ChessNet(cfg)
    x = torch.randn(2, 146, 8, 8)
    out = m(x)
    assert out["policy_own"].shape == (2, COMPACT_POLICY_SIZE)

    legal_flat = torch.tensor([877, 950, 951], dtype=torch.long)
    legal_counts = torch.tensor([2, 1], dtype=torch.long)
    legal = m(x, legal_flat=legal_flat, legal_counts=legal_counts)
    assert legal["policy_own"].shape == (3,)
    assert legal["wdl"].shape == (2, 3)


def test_transformer_optional_arc_pos_encoding_and_deepnorm():
    cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        use_smolgen=False,
        use_nla=False,
        input_pos_encoding="arc",
        use_deepnorm=True,
    )
    m = ChessNet(cfg)
    x = torch.randn(2, 146, 8, 8)
    out = m(x)
    assert out["policy_own"].shape == (2, 64 * 73)
    assert out["wdl"].shape == (2, 3)
    assert m.embed.in_features == 146 + 64
    assert hasattr(m, "arc_pos_encoding")
    assert "arc_pos_encoding" in m._buffers
    assert "arc_pos_encoding" not in m.state_dict()
    assert m.blocks[0].residual_branch_scale == (2.0 * cfg.num_layers) ** -0.25


def test_transformer_arc_adapter_starts_checkpoint_compatible():
    base_cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        use_nla=False,
    )
    adapter_cfg = TransformerConfig(
        in_planes=146,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        use_smolgen=False,
        use_nla=False,
        input_pos_encoding="arc_adapter",
    )
    base = ChessNet(base_cfg).eval()
    adapter = ChessNet(adapter_cfg).eval()
    missing, unexpected = adapter.load_state_dict(base.state_dict(), strict=False)

    assert missing == ["pos_adapter.weight"]
    assert unexpected == []
    assert adapter.embed.in_features == 146
    assert hasattr(adapter, "arc_pos_encoding")
    pos_adapter = adapter.pos_adapter
    assert pos_adapter is not None
    assert torch.count_nonzero(pos_adapter.weight) == 0

    x = torch.randn(2, 146, 8, 8)
    with torch.no_grad():
        base_out = base(x)
        adapter_out = adapter(x)

    assert torch.allclose(adapter_out["policy_own"], base_out["policy_own"])
    assert torch.allclose(adapter_out["wdl"], base_out["wdl"])


def test_transformer_rejects_unknown_pos_encoding():
    cfg = TransformerConfig(in_planes=146, input_pos_encoding="bogus")
    try:
        ChessNet(cfg)
    except ValueError as exc:
        assert "input_pos_encoding" in str(exc)
    else:
        raise AssertionError("expected invalid input_pos_encoding to fail")


def test_transformer_legal_policy_matches_dense_logits():
    cfg = TransformerConfig(in_planes=146, embed_dim=64, num_layers=1, num_heads=4, use_smolgen=False, use_nla=False)
    m = ChessNet(cfg).eval()
    x = torch.randn(2, 146, 8, 8)
    legal_counts = torch.tensor([5, 4], dtype=torch.long)
    legal_flat = torch.tensor(
        [
            0,
            5,
            63,
            48 * 73 + 64,
            48 * 73 + 72,
            7 * 73 + 12,
            16 * 73 + 21,
            63 * 73 + 72,
            4671,
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        dense = m(x)
        compact = m.forward_legal_policy(x, legal_flat, legal_counts)

    rows = torch.repeat_interleave(torch.arange(2), legal_counts)
    assert torch.allclose(compact["policy_own"], dense["policy_own"][rows, legal_flat])
    assert torch.allclose(compact["wdl"], dense["wdl"])


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
