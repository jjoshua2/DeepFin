from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.export import Dim

from chess_anti_engine.inference import _policy_output


class _OnnxWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        policy = _policy_output(out)
        wdl = out["wdl"]
        moves_left = out.get("moves_left", torch.zeros((x.shape[0], 1), device=x.device, dtype=wdl.dtype))
        return policy, wdl, moves_left


@dataclass
class OnnxExportConfig:
    opset: int = 17


@dataclass
class OnnxQuantizeConfig:
    """Post-export ONNX quantization settings.

    We start with ORT dynamic quantization (weights INT8, activations stay float).
    This is usually the easiest/most-stable INT8 deployment path.
    """

    mode: str = "dynamic"  # dynamic only for now
    per_channel: bool = False
    reduce_range: bool = False
    weight_type: str = "qint8"  # qint8 | quint8


def export_onnx(model: nn.Module, *, out_path: Path, device: str = "cpu", cfg: OnnxExportConfig | None = None) -> None:
    cfg = cfg or OnnxExportConfig()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = _OnnxWrapper(model).to(device)
    wrapper.eval()

    dummy = torch.randn(1, 146, 8, 8, device=device)

    batch = Dim("batch", min=1)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["input_planes"],
        output_names=["policy", "wdl", "moves_left"],
        dynamic_shapes={"x": {0: batch}},
        opset_version=int(cfg.opset),
    )


def _quantize_dynamic_ort(
    *,
    fp32_path: Path,
    int8_path: Path,
    cfg: OnnxQuantizeConfig,
) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "INT8 quantization requires onnxruntime with quantization support. "
            "Install with: pip install onnxruntime"
        ) from e

    wt = str(cfg.weight_type).lower()
    if wt == "quint8":
        weight_type = QuantType.QUInt8
    else:
        weight_type = QuantType.QInt8

    extra_options: dict[str, Any] = {"MatMulConstBOnly": True}

    # ORT quantization runs ONNX shape inference internally. Some PyTorch-exported models
    # include intermediate ValueInfo shapes that can conflict with ONNX's inferred shapes
    # (even though the model itself runs fine in ORT). If that happens, fall back to
    # stripping ValueInfo shapes before quantization.
    try:
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            per_channel=bool(cfg.per_channel),
            reduce_range=bool(cfg.reduce_range),
            weight_type=weight_type,
            extra_options=extra_options,
        )
        return
    except Exception:
        pass

    try:
        import onnx  # type: ignore

        model = onnx.load(str(fp32_path))

        # Strip only *shapes* (keep dtype/type info). This avoids ONNX shape inference
        # complaining about mismatches while still leaving enough type data for ORT's
        # quantizer.
        def _clear_shape(v) -> None:
            if v.type.HasField("tensor_type") and v.type.tensor_type.HasField("shape"):
                # Clearing only dims can accidentally turn an "unknown" shape into a scalar
                # (rank=0) and trigger shape-inference mismatches. Clear the whole shape
                # field instead.
                v.type.tensor_type.ClearField("shape")

        for v in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
            _clear_shape(v)

        extra_options2 = dict(extra_options)
        extra_options2.setdefault("DefaultTensorType", onnx.TensorProto.FLOAT)

        quantize_dynamic(
            model_input=model,
            model_output=str(int8_path),
            per_channel=bool(cfg.per_channel),
            reduce_range=bool(cfg.reduce_range),
            weight_type=weight_type,
            extra_options=extra_options2,
        )
    except Exception as e:
        raise RuntimeError(
            "ONNX INT8 quantization failed. If you see a ShapeInferenceError, "
            "try exporting with a different opset or upgrading onnx/onnxruntime."
        ) from e


def export_onnx_int8(
    model: nn.Module,
    *,
    out_path: Path,
    device: str = "cpu",
    export_cfg: OnnxExportConfig | None = None,
    quant_cfg: OnnxQuantizeConfig | None = None,
    keep_fp32: bool = False,
) -> None:
    """Export ONNX and post-quantize to INT8.

    Produces an INT8 model at `out_path`.

    Notes:
    - This uses ORT *dynamic* quantization, which does not require a calibration set.
    - For now, we keep export and quantization separate so fp32 ONNX can still be
      produced and used for parity checks.
    """

    export_cfg = export_cfg or OnnxExportConfig()
    quant_cfg = quant_cfg or OnnxQuantizeConfig()

    if str(quant_cfg.mode).lower() != "dynamic":
        raise ValueError(f"Unsupported quantization mode {quant_cfg.mode!r} (only 'dynamic' supported)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only the final suffix (".onnx" -> ".fp32.onnx").
    fp32_path = out_path.with_suffix(".fp32.onnx")

    export_onnx(model, out_path=fp32_path, device=device, cfg=export_cfg)
    _quantize_dynamic_ort(fp32_path=fp32_path, int8_path=out_path, cfg=quant_cfg)

    if not keep_fp32:
        fp32_path.unlink(missing_ok=True)
