from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_onnx_export_and_parity_cpu(tmp_path: Path):
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")

    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.onnx import export_onnx

    model = build_model(
        ModelConfig(
            kind="transformer",
            embed_dim=64,
            num_layers=1,
            num_heads=4,
            ffn_mult=2,
            use_smolgen=False,
            use_nla=False,
        )
    ).cpu()
    model.eval()

    onnx_path = tmp_path / "model.onnx"
    export_onnx(model, out_path=onnx_path, device="cpu")
    assert onnx_path.exists()

    # Compare ORT vs torch on a deterministic input.
    x = torch.randn(2, 146, 8, 8, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
        policy = out.get("policy", out["policy_own"]).detach().cpu().numpy()
        wdl = out["wdl"].detach().cpu().numpy()
        moves_left = out.get("moves_left", torch.zeros((x.shape[0], 1))).detach().cpu().numpy()

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_policy, ort_wdl, ort_moves_left = sess.run(
        None,
        {"input_planes": x.cpu().numpy()},
    )

    np.testing.assert_allclose(ort_policy, policy, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(ort_wdl, wdl, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(ort_moves_left, moves_left, rtol=1e-3, atol=1e-3)
