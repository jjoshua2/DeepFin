"""Closed raw-WDL arithmetic contract for the retained BT4 artifact.

Ordinary artifacts retain validate_wdl_values unchanged. The one exact model
below has FP16 internals and FP32 I/O: its already-qualified FP16 probability
mass rule must not be inferred from storage dtype or implemented by casting.

Algorithm provenance: retained bt4_provisional_model_binding_v1.py source
SHA256 6e3556527146d4492528b0d1facb06a342873d3cb38f49c48f6ce06c95394324,
validate_model_wdl; saved precision replay receipt
ba3f35e768a9e64c1417856b176526e463b466898b1d3b42f80734d150cc4af4.
This factors the existing artifact-specific rule, not a new tolerance option.
The admitted owner must still authenticate the actual model/session, precision
binding and authority before startup. A caller's hash is not model attestation.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from scripts.bt4_raw_corpus_sidecar import validate_wdl_values

_FP16_INTERNAL_FP32_IO_MODEL_SHA256 = "809db9c3c8063683bf8e7c08de8fec92ea4e37ad8c68c8b1befbde4bd1958961"


def validate_model_wdl(
    values: np.ndarray, rows: int, contract: Mapping[str, Any], model_sha256: str,
) -> None:
    """Validate unmodified native values under a closed model-bound rule."""
    if model_sha256 != _FP16_INTERNAL_FP32_IO_MODEL_SHA256:
        validate_wdl_values(values, rows, contract)
        return
    if contract.get("kind") != "probabilities" or contract.get("dtype") != "float32":
        raise ValueError("exact FP16-internal model requires FP32 probability storage")
    # Retain the runtime boundary even for untyped external session adapters.
    if not isinstance(values, np.ndarray):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("exact raw WDL shape/storage required")
    if values.dtype != np.dtype("float32") or values.shape != (rows, 3):
        raise ValueError("exact raw WDL shape/storage required")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("finite range0..1 WDL required")
    if not np.all(np.abs(values.sum(axis=1, dtype=np.float64) - 1.0) <= 2**-10):
        raise ValueError("FP16 arithmetic WDL probability mass")
