from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_onnx_export_int8_smoke_cpu(tmp_path: Path):
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime.quantization")

    from chess_anti_engine.model import ModelConfig, build_model
    from chess_anti_engine.onnx import export_onnx, export_onnx_int8

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

    fp32_path = tmp_path / "model.fp32.onnx"
    int8_path = tmp_path / "model.int8.onnx"

    export_onnx(model, out_path=fp32_path, device="cpu")
    export_onnx_int8(model, out_path=int8_path, device="cpu")

    assert fp32_path.exists()
    assert int8_path.exists()

    x = torch.randn(2, 146, 8, 8, dtype=torch.float32)
    x_np = x.cpu().numpy()

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    out_fp32 = sess_fp32.run(None, {"input_planes": x_np})
    out_int8 = sess_int8.run(None, {"input_planes": x_np})

    assert len(out_fp32) == len(out_int8) == 3

    for a, b in zip(out_fp32, out_int8):
        assert a.shape == b.shape
        assert np.isfinite(b).all()

    # Basic sanity: don't allow totally-degenerate outputs.
    assert float(np.abs(out_int8[0]).mean()) > 1e-8
