"""Verified BT4 session for future in-memory generation; no writer or game loop.

The returned provenance binds one opened ONNX artifact, its declared heads and
realized session providers to the root observations. It is not a corpus sidecar
receipt: a future writer must also verify physical source rows and serialized x.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding.lc0 import normalize_lc0_history_encoding
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from scripts.bt4_generation_evaluator import BT4OnnxEvaluator
from scripts.bt4_policy_dump import (
    file_sha256, open_session, remap_provenance, resolve_policy_output,
)
from scripts.bt4_policy_mix import functional_remap_identity
from scripts.bt4_raw_corpus_sidecar import resolve_wdl_output

_FLOAT_TYPES = {
    "tensor(float16)": np.dtype("float16"),
    "tensor(float)": np.dtype("float32"),
    "tensor(double)": np.dtype("float64"),
}
_PRODUCER_SOURCES = (
    "scripts/bt4_generation_evaluator.py",
    "scripts/bt4_generation_session.py",
    "scripts/bt4_policy_dump.py",
    "scripts/bt4_raw_corpus_sidecar.py",
    "chess_anti_engine/encoding/cboard_encode.py",
    "chess_anti_engine/encoding/lc0.py",
)


@dataclass(frozen=True)
class BT4SessionProvenance:
    """Immutable, canonicalizable session descriptors; no physical row identity."""

    onnx_path: str
    onnx_sha256: str
    input_name: str
    input_dtype: str
    input_shape: tuple[str | int | None, int, int, int]
    providers: tuple[str, ...]
    provider_options: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    policy_output: str
    policy_dtype: str
    policy_shape: tuple[str | int | None, int]
    wdl_output: str
    wdl_kind: str
    wdl_dtype: str
    wdl_shape: tuple[str | int | None, int]
    wdl_order: tuple[str, str, str]
    wdl_pov: str
    input_history_encoding: str
    input_extra_features: str
    history_rep_fix: bool
    remap_commit: str
    remap_dirty: bool
    remap_blobs: tuple[tuple[str, str], ...]
    producer_sha256: tuple[tuple[str, str], ...]

    @property
    def sha256(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VerifiedBT4Session:
    evaluator: BT4OnnxEvaluator
    provenance: BT4SessionProvenance


def _reject_external_data(path: Path) -> None:
    # Parsing does not load external tensor files; those require a separate
    # artifact-bundle hash contract before this factory can accept them.
    model = onnx.load_model(str(path), load_external_data=False)
    def tensors(message: Any) -> Any:
        if isinstance(message, onnx.TensorProto):
            yield message
        for field, value in message.ListFields():
            if field.type != field.TYPE_MESSAGE:
                continue
            if field.label == field.LABEL_REPEATED:
                for child in value:
                    yield from tensors(child)
            else:
                yield from tensors(value)

    if any(tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data
           for tensor in tensors(model)):
        raise ValueError("BT4 ONNX external tensor data is not pinned")


def _input_descriptor(
    sess: Any,
) -> tuple[str, np.dtype[Any], tuple[str | int | None, int, int, int]]:
    inputs = sess.get_inputs()
    if len(inputs) != 1:
        raise ValueError("BT4 session requires exactly one input")
    meta = inputs[0]
    dtype = _FLOAT_TYPES.get(meta.type)
    if dtype is None or dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("BT4 input must be tensor(float16) or tensor(float)")
    shape = meta.shape
    batch = shape[0] if isinstance(shape, (list, tuple)) and shape else None
    if (
        not isinstance(meta.name, str) or not meta.name
        or not isinstance(shape, (list, tuple)) or len(shape) != 4
        or list(shape[1:]) != [112, 8, 8]
        or not (batch is None or (type(batch) is str and bool(batch))
                or (type(batch) is int and batch == 1))
    ):
        raise ValueError("BT4 input must have a named [dynamic-or-1,112,8,8] shape")
    return meta.name, dtype, (batch, 112, 8, 8)


def _provider_descriptor(
    sess: Any, providers: list[str], gpu_mem_gb: float,
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    if not providers or providers != list(sess.get_providers()) or len(set(providers)) != len(providers):
        raise ValueError("BT4 realized session provider list is invalid")
    raw = sess.get_provider_options()
    if set(raw) != set(providers):
        raise ValueError("BT4 realized provider options disagree with providers")
    if gpu_mem_gb > 0:
        if providers[0] != "CUDAExecutionProvider":
            raise ValueError("BT4 CUDA provider is not first in execution order")
        expected_bytes = int(gpu_mem_gb * 1024 ** 3)
        try:
            realized_bytes = int(raw["CUDAExecutionProvider"]["gpu_mem_limit"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("BT4 CUDA memory limit is unavailable") from exc
        if realized_bytes != expected_bytes:
            raise ValueError("BT4 CUDA memory limit differs from requested cap")
        if raw["CUDAExecutionProvider"].get("device_id") != "0":
            raise ValueError("BT4 CUDA device differs from requested device 0")
    return tuple((provider, tuple(sorted((str(k), str(v)) for k, v in raw[provider].items())))
                 for provider in providers)


def _head_shape(
    meta: Any, width: int, input_batch: str | int | None,
) -> tuple[str | int | None, int]:
    shape = meta.shape
    if not isinstance(shape, (list, tuple)) or len(shape) != 2 or shape[1] != width:
        raise ValueError(f"BT4 {meta.name} must be a floating [batch,{width}] output")
    batch = shape[0]
    if not (batch is None or (type(batch) is str and bool(batch))
            or (type(batch) is int and batch == input_batch)):
        raise ValueError(f"BT4 {meta.name} fixed output batch conflicts with input")
    return batch, width


def open_verified_bt4_session(
    onnx_path: Path, *, expected_sha256: str, gpu_mem_gb: float, threads: int,
    policy_output: str | None, wdl_output: str, wdl_kind: str,
    input_history_encoding: str, input_extra_features: str,
    history_rep_fix: bool,
) -> VerifiedBT4Session:
    """Verify artifact/session identity, then construct the existing evaluator.

    The caller configures repetition mode once before any boards exist; this
    factory only asserts it. No games, inference, or sidecar writes occur here.
    """
    if type(history_rep_fix) is not bool or rep_fix.current() is not history_rep_fix:
        raise RuntimeError("BT4 repetition mode must be explicitly configured before session creation")
    if not np.isfinite(gpu_mem_gb) or gpu_mem_gb < 0 or (gpu_mem_gb > 0 and int(gpu_mem_gb * 1024 ** 3) < 1):
        raise ValueError("BT4 GPU memory budget must be finite and nonnegative")
    history_mode = normalize_lc0_history_encoding(input_history_encoding)
    if (len(expected_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_sha256)):
        raise ValueError("BT4 expected model SHA-256 must be lowercase hex")
    path = onnx_path.resolve(strict=True)
    before_sha = file_sha256(path)
    if before_sha != expected_sha256:
        raise ValueError("BT4 ONNX artifact SHA-256 does not match expected pin")
    _reject_external_data(path)
    sess, opened_name, opened_dtype, providers = open_session(
        str(path), gpu_mem_gb=gpu_mem_gb, threads=threads,
    )
    after_sha = file_sha256(path)
    if after_sha != before_sha:
        raise ValueError("BT4 ONNX artifact changed while session opened")
    if gpu_mem_gb > 0 and "CUDAExecutionProvider" not in providers:
        raise ValueError("BT4 CUDA requested but not realized by session")
    input_name, input_dtype, input_shape = _input_descriptor(sess)
    if opened_name != input_name or np.dtype(opened_dtype) != input_dtype:
        raise ValueError("BT4 opened input descriptor disagrees with session")
    provider_options = _provider_descriptor(sess, providers, gpu_mem_gb)
    try:
        policy_meta = sess.get_outputs()[resolve_policy_output(sess, policy_output)]
    except SystemExit as exc:
        raise ValueError(f"BT4 policy head cannot be resolved: {exc}") from exc
    if policy_meta.type not in _FLOAT_TYPES:
        raise ValueError("BT4 policy must be a floating [batch,1858] output")
    policy_shape = _head_shape(policy_meta, COMPACT_POLICY_SIZE, input_shape[0])
    wdl = resolve_wdl_output(
        sess, {"output": wdl_output, "kind": wdl_kind}, policy_meta.name,
    )
    if wdl is None or wdl_kind not in ("logits", "probabilities"):
        raise ValueError("BT4 requires a named native WDL contract")
    wdl_meta = next(out for out in sess.get_outputs() if out.name == wdl["output"])
    wdl_shape = _head_shape(wdl_meta, 3, input_shape[0])
    remap = functional_remap_identity(remap_provenance())
    blobs = remap["blobs"]
    if any(len(blob) != 40 or any(c not in "0123456789abcdef" for c in blob)
           for blob in blobs.values()):
        raise ValueError("BT4 remap source blob hash unavailable")
    root = Path(__file__).resolve().parents[1]
    provenance = BT4SessionProvenance(
        onnx_path=str(path), onnx_sha256=before_sha,
        input_name=input_name, input_dtype=input_dtype.name,
        input_shape=input_shape,
        providers=tuple(providers), provider_options=provider_options,
        policy_output=policy_meta.name, policy_dtype=_FLOAT_TYPES[policy_meta.type].name,
        policy_shape=policy_shape,
        wdl_output=wdl["output"], wdl_kind=wdl["kind"], wdl_dtype=wdl["dtype"],
        wdl_shape=wdl_shape,
        wdl_order=("win", "draw", "loss"), wdl_pov="side_to_move",
        input_history_encoding=history_mode,
        input_extra_features=input_extra_features, history_rep_fix=history_rep_fix,
        remap_commit=remap["commit"], remap_dirty=remap["dirty"],
        remap_blobs=tuple(sorted(blobs.items())),
        producer_sha256=tuple((source, file_sha256(root / source)) for source in _PRODUCER_SOURCES),
    )
    evaluator = BT4OnnxEvaluator(
        sess, input_name=input_name, input_dtype=input_dtype,
        policy_output=policy_meta.name, wdl_output=wdl["output"], wdl_kind=wdl_kind,
        input_history_encoding=history_mode,
        input_extra_features=input_extra_features,
        model_sha256=before_sha, history_rep_fix=history_rep_fix,
        verified_session_sha256=provenance.sha256,
        fixed_batch_size=1 if input_shape[0] == 1 else None,
    )
    return VerifiedBT4Session(evaluator=evaluator, provenance=provenance)
