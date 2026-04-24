from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch


def _auto_cuda_amp_dtype() -> torch.dtype:
  # Prefer BF16 when supported (Ampere+ with BF16 support, Hopper, Ada, Blackwell, etc.).
  # Fall back to FP16 otherwise.
    try:
        is_bf16 = bool(torch.cuda.is_bf16_supported())
    except Exception:
        is_bf16 = False
    return torch.bfloat16 if is_bf16 else torch.float16


def _parse_amp_dtype(dtype: str) -> torch.dtype | None:
    d = str(dtype).lower().strip()
    if d in {"", "none", "off", "false", "0", "fp32", "float32"}:
        return None
    if d in {"auto"}:
        return _auto_cuda_amp_dtype()
    if d in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if d in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unknown amp dtype {dtype!r} (expected auto|bf16|fp16|off)")


@contextmanager
def inference_autocast(*, device: str, enabled: bool = True, dtype: str = "auto"):
    """Autocast context for inference.

    - Only activates on CUDA devices.
    - dtype='auto' chooses BF16 when supported else FP16.
    """

    if not enabled or not str(device).startswith("cuda"):
        with nullcontext():
            yield
        return

    amp_dtype = _parse_amp_dtype(dtype)
    if amp_dtype is None:
        with nullcontext():
            yield
        return

    with torch.amp.autocast("cuda", dtype=amp_dtype):
        yield
