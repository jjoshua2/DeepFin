from __future__ import annotations

import pytest
import torch

from chess_anti_engine.train.aurora import AuroraWithAuxAdam, _aurora_update, _polar_express


def test_aurora_with_aux_adam_updates_matrix_and_fallback_params():
    mat = torch.nn.Parameter(torch.randn(6, 3, dtype=torch.float32))
    vec = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))

    opt = AuroraWithAuxAdam(
        [
            {"params": [mat], "lr": 0.02, "weight_decay": 0.01, "use_aurora": True},
            {"params": [vec], "lr": 1e-3, "weight_decay": 0.01, "use_aurora": False},
        ],
        aurora_polar_steps=3,
    )

    mat_before = mat.detach().clone()
    vec_before = vec.detach().clone()

    mat.grad = torch.randn_like(mat)
    vec.grad = torch.tensor([0.1, -0.2], dtype=torch.float32)
    opt.step()

    assert not torch.allclose(mat, mat_before)
    assert not torch.allclose(vec, vec_before)
    assert "momentum_buffer" in opt.state[mat]
    assert "exp_avg" in opt.state[vec]
    assert "exp_avg_sq" in opt.state[vec]


def test_aurora_update_accepts_wide_tall_and_square_matrices():
    for shape in [(8, 3), (3, 8), (4, 4)]:
        update = _aurora_update(
            torch.randn(*shape, dtype=torch.float32),
            pp_iterations=2,
            polar_steps=3,
        )
        assert update.shape == shape
        assert torch.isfinite(update).all()


def test_aurora_update_accepts_polar_express_fp16_request():
    update = _aurora_update(
        torch.randn(8, 3, dtype=torch.float32),
        pp_iterations=3,
        polar_steps=8,
        polar_method="polar_express",
        polar_dtype="fp16",
    )
    assert update.shape == (8, 3)
    assert torch.isfinite(update).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 matmul path is CUDA-only")
def test_polar_express_uses_fp16_inner_matmuls_on_cuda():
    mat = torch.randn(8, 3, device="cuda", dtype=torch.float32)
    out = _polar_express(mat, steps=2, work_dtype=torch.float16)
    assert out.dtype == torch.float32
    assert out.shape == mat.shape
    assert torch.isfinite(out).all()


def test_aurora_uw_floor_scales_relative_update():
    mat = torch.nn.Parameter(torch.full((12, 3), 100.0, dtype=torch.float32))
    opt = AuroraWithAuxAdam(
        [
            {
                "params": [mat],
                "lr": 0.01,
                "weight_decay": 0.0,
                "use_aurora": True,
                "aurora_uw_floor": 0.35,
            },
        ],
        aurora_polar_steps=3,
    )

    before = mat.detach().clone()
    mat.grad = torch.randn_like(mat) * 1e-3
    opt.step()

    rel_step = (mat.detach() - before).float().norm() / before.float().norm()
    assert rel_step >= 0.01 * 0.35 * 0.99
    assert opt.last_uw_stats["aurora_uw_floor"] == 0.35
    assert opt.last_uw_stats["aurora_uw_floored_frac"] > 0.0
    assert opt.last_uw_stats["aurora_uw_scale_max"] > 1.0
    assert opt.last_uw_stats["aurora_uw_effective_ratio_min"] >= 0.01 * 0.35 * 0.99


def test_aurora_uw_floor_skips_nonfinite_ratio_scale():
    mat = torch.nn.Parameter(torch.full((12, 3), 1e38, dtype=torch.float32))
    opt = AuroraWithAuxAdam(
        [
            {
                "params": [mat],
                "lr": 0.01,
                "weight_decay": 0.0,
                "use_aurora": True,
                "aurora_uw_floor": 0.35,
            },
        ],
        aurora_polar_steps=3,
    )

    mat.grad = torch.randn_like(mat) * 1e-3
    opt.step()

    assert torch.isfinite(mat).all()
    assert opt.last_uw_stats["aurora_uw_scale_max"] == 1.0
