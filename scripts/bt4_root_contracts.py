"""Pure BT4 root values, tensor identity and native WDL validation.

Authoritative definitions factored without changing their bodies from
bt4_generation_evaluator, gen_sf_rooted_corpus and bt4_raw_corpus_sidecar.
Those modules retain aliases; no session, model, replay or writer is loaded here.
BT4RootOutput now has this defining __module__; old pickle module lookups still
resolve the identical class through bt4_generation_evaluator's re-export.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE

_ZERO_LOGIT = np.float32(-1e9)


def _search_policy_logits(dense_policy: np.ndarray) -> np.ndarray:
    """Compact search logits whose legal softmax retains exact zero mass."""
    logits = np.full((COMPACT_POLICY_SIZE,), _ZERO_LOGIT, dtype=np.float32)
    positive = dense_policy > 0.0
    logits[positive] = np.log(dense_policy[positive]).astype(np.float32)
    return logits


def _search_wdl_logits(raw: np.ndarray, kind: str) -> np.ndarray:
    """Stable search-only logits; never alter the native WDL observation."""
    values = np.asarray(raw, dtype=np.float64)
    if kind == "probabilities":
        logits = np.full((3,), float(_ZERO_LOGIT), dtype=np.float64)
        positive = values > 0.0
        logits[positive] = np.log(values[positive])
    elif kind == "logits":
        with np.errstate(over="ignore"):
            logits = values - values.max()
        logits = np.maximum(logits, float(_ZERO_LOGIT))
    else:
        raise ValueError(f"unsupported WDL output kind {kind!r}")
    result = logits.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("BT4 WDL cannot be represented as finite search logits")
    return result


@dataclass(frozen=True)
class BT4RootOutput:
    """In-memory root observation, not a complete corpus provenance record."""

    fen: str
    input_key: str
    source_key: bytes
    policy_t1: np.ndarray  # dense lc0_1858, float32; not search-improved policy
    wdl_raw: np.ndarray  # native dtype and named head; side-to-move POV
    policy_output: str
    wdl_output: str
    wdl_kind: str
    model_sha256: str
    input_name: str
    input_dtype: str
    input_history_encoding: str
    input_extra_features: str
    history_rep_fix: bool
    onnx_feed_sha256: str = ""
    verified_session_sha256: str | None = None

    def search_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Fresh batched logits for search; root-only actors retain no derived copy."""
        return (_search_policy_logits(self.policy_t1)[None],
                _search_wdl_logits(self.wdl_raw, self.wdl_kind)[None])


def input_tensor_key(planes: np.ndarray) -> str:
    """The identity of one encoded input: blake2b-128 over its float32 bytes.

    Spelled once and called from BOTH ends of the chain: the generator hashes
    the tensor live play would encode (:func:`row_key`) and the deriver hashes
    the tensor it reconstructed from the banked window, and a row whose two
    hashes differ is refused there.  ``float32`` and C order are forced so the
    bytes are the same whichever encoder produced the array.
    """
    return hashlib.blake2b(
        np.ascontiguousarray(planes, dtype=np.float32).tobytes(), digest_size=16,
    ).hexdigest()


def validate_wdl_values(values: np.ndarray, rows: int, contract: Mapping[str, Any]) -> None:
    if values.shape != (rows, 3) or values.dtype != np.dtype(contract["dtype"]):
        raise ValueError("WDL shape/native dtype differs")
    if not np.isfinite(values).all():
        raise ValueError("WDL values are nonfinite")
    if contract["kind"] == "probabilities":
        # Raw values stay untouched; tolerate only native rounding of unit mass.
        tolerance = 2**-10 if values.dtype == np.dtype("float16") else 2e-6
        if (np.any(values < 0) or np.any(values > 1)
                or not np.allclose(values.sum(axis=1, dtype=np.float64), 1,
                                   rtol=0, atol=tolerance)):
            raise ValueError("WDL probability range/mass invalid")


def resolve_wdl_output(sess: Any, requested: Mapping[str, str] | None,
                       policy_name: str) -> dict[str, str] | None:
    """Explicit named W/D/L STM contract, never guess a three-wide head."""
    if requested is None:
        return None
    matches = [out for out in sess.get_outputs() if out.name == requested["output"]]
    if len(matches) != 1 or requested["output"] == policy_name:
        raise ValueError("WDL output absent, ambiguous or aliases policy")
    out = matches[0]
    types = {"tensor(float16)": "float16", "tensor(float)": "float32", "tensor(double)": "float64"}
    if len(out.shape) != 2 or out.shape[1] != 3 or out.type not in types:
        raise ValueError("WDL output must be a native floating [batch,3] tensor")
    return {**requested, "dtype": types[out.type]}
