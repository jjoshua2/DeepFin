"""Batch-invariant WDL reads in ``scripts/foreign_net_audit.py``.

The pre-fix classifier decided logits-versus-probabilities on each scoring
batch. A row ``[0.2, 0.3, 0.5]`` is a legal probability triple and a legal
logit triple; alone it was stored raw, and beside a negative logit it was
softmaxed. ``audit_compare_buckets`` feeds that cached triple to Brier/ECE.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chess
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import fill_lc0_history_repeat
from chess_anti_engine.eval.audit import wdl_brier
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.onnx.load import (
    WDL_OUTPUT_LOGITS,
    WDL_OUTPUT_PROBABILITIES,
)
from scripts.foreign_net_audit import (
    _checkpoint_wdl_kind,
    _score_onnx,
    _to_wdl_probs,
)

_SIMPLEX = np.array([0.2, 0.3, 0.5], dtype=np.float64)
_NEGATIVE = np.array([-1.0, 0.0, 2.0], dtype=np.float64)


def _legacy_batch_to_wdl_probs(raw: np.ndarray) -> np.ndarray:
    """The classifier this file replaces: one decision for the whole array."""
    w = np.asarray(raw, dtype=np.float64)
    is_probs = bool((w >= -1e-4).all()) and bool(
        np.allclose(w.sum(axis=1), 1.0, atol=0.1),
    )
    if is_probs:
        return w
    shifted = np.exp(w - w.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def _softmax_rows(raw: np.ndarray) -> np.ndarray:
    return _to_wdl_probs(raw, kind=WDL_OUTPUT_LOGITS)


def test_legacy_classifier_changes_the_simplex_row_when_a_negative_joins() -> None:
    alone = _legacy_batch_to_wdl_probs(_SIMPLEX[None, :])
    paired = _legacy_batch_to_wdl_probs(np.stack([_SIMPLEX, _NEGATIVE]))
    assert np.allclose(alone[0], _SIMPLEX)
    np.testing.assert_allclose(paired[0], _softmax_rows(_SIMPLEX[None, :])[0])
    assert not np.allclose(alone[0], paired[0])
    target = np.array([0.6, 0.3, 0.1])
    assert wdl_brier(alone[0], target) == pytest.approx(0.32)
    assert wdl_brier(paired[0], target) == pytest.approx(0.18134963598861104)


def test_checkpoint_wdl_is_logits_and_probabilities_are_refused() -> None:
    assert _checkpoint_wdl_kind("auto") == WDL_OUTPUT_LOGITS
    assert _checkpoint_wdl_kind("logits") == WDL_OUTPUT_LOGITS
    with pytest.raises(SystemExit, match="probabilities"):
        _checkpoint_wdl_kind("probabilities")
    np.testing.assert_allclose(
        _to_wdl_probs(_SIMPLEX[None, :], kind=WDL_OUTPUT_LOGITS)[0],
        _softmax_rows(_SIMPLEX[None, :])[0],
    )
    with pytest.raises(ValueError, match="wdl kind"):
        _to_wdl_probs(_SIMPLEX[None, :], kind="auto")


def _encode(board: chess.Board) -> np.ndarray:
    planes = encode_position(board, add_features=False, input_history_encoding="lc0_root")
    return fill_lc0_history_repeat(planes).astype(np.float32)


def _feature_index(start: np.ndarray, other: np.ndarray) -> tuple[int, int, int]:
    gap = np.argwhere((np.abs(start) < 1e-6) & (np.abs(other - 1.0) < 1e-6))
    if gap.size == 0:
        raise AssertionError("fixture boards do not differ by a 0/1 plane")
    plane, rank, file = (int(v) for v in gap[0])
    return plane, rank, file


def _constant_row_model(
    path: Path, planes: int, index: tuple[int, int, int],
    row_when_zero: np.ndarray, row_when_one: np.ndarray,
) -> None:
    """Emit ``row_when_zero`` where plane index is 0 and ``row_when_one`` where it is 1."""
    plane, rank, file = index
    nodes = [
        helper.make_node("Slice", ["planes", "starts", "ends", "axes", "steps"], ["feat4"]),
        helper.make_node("Squeeze", ["feat4", "sq_axes"], ["feat"]),
        helper.make_node("Unsqueeze", ["feat", "unsq_axes"], ["fcol"]),
        helper.make_node("Sub", ["one", "fcol"], ["one_minus"]),
        helper.make_node("Mul", ["one_minus", "row0"], ["part0"]),
        helper.make_node("Mul", ["fcol", "row1"], ["part1"]),
        helper.make_node("Add", ["part0", "part1"], ["wdl"]),
        helper.make_node("Shape", ["planes"], ["shp"]),
        helper.make_node("Gather", ["shp", "zero_i"], ["batch"], axis=0),
        helper.make_node("Concat", ["batch", "pol_w"], ["pol_shape"], axis=0),
        helper.make_node(
            "ConstantOfShape", ["pol_shape"], ["policy"],
            value=helper.make_tensor("zero_f", TensorProto.FLOAT, [1], [0.0]),
        ),
    ]
    inits = [
        helper.make_tensor("starts", TensorProto.INT64, [4], [0, plane, rank, file]),
        helper.make_tensor(
            "ends", TensorProto.INT64, [4], [2**31 - 1, plane + 1, rank + 1, file + 1],
        ),
        helper.make_tensor("axes", TensorProto.INT64, [4], [0, 1, 2, 3]),
        helper.make_tensor("steps", TensorProto.INT64, [4], [1, 1, 1, 1]),
        helper.make_tensor("sq_axes", TensorProto.INT64, [3], [1, 2, 3]),
        helper.make_tensor("unsq_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("zero_i", TensorProto.INT64, [1], [0]),
        helper.make_tensor("pol_w", TensorProto.INT64, [1], [COMPACT_POLICY_SIZE]),
        numpy_helper.from_array(row_when_zero.astype(np.float32), "row0"),
        numpy_helper.from_array(row_when_one.astype(np.float32), "row1"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one"),
    ]
    graph = helper.make_graph(
        nodes, "wdl_rows",
        [helper.make_tensor_value_info("planes", TensorProto.FLOAT, ["B", planes, 8, 8])],
        [
            helper.make_tensor_value_info(
                "policy", TensorProto.FLOAT, ["B", COMPACT_POLICY_SIZE],
            ),
            helper.make_tensor_value_info("wdl", TensorProto.FLOAT, ["B", 3]),
        ],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)


def _score(path: Path, boards: list[chess.Board], batch_size: int, kind: str) -> np.ndarray:
    args = argparse.Namespace(
        onnx=str(path), gpu_mem_gb=0.0, input_format="lc0_planes", ort_threads=2,
        history="lc0_root", history_fill="repeat", policy_output=None, wdl_output=None,
        batch_size=batch_size, wdl_output_kind=kind,
    )
    _policy, wdl, heads = _score_onnx(boards, args)
    if kind == "auto":
        assert heads["wdl_output_kind"] in (WDL_OUTPUT_LOGITS, WDL_OUTPUT_PROBABILITIES)
    else:
        assert heads["wdl_output_kind"] == kind
    return np.asarray(wdl, dtype=np.float64)


def test_score_onnx_logits_are_batch_invariant_and_softmax_the_simplex_row(
    tmp_path: Path,
) -> None:
    """Start position emits a negative logit; the other row is ``[0.2, 0.3, 0.5]``.

    Unpatched, batch 1 stores that row raw and batch 2 softmaxes it. Explicit
    logits and auto (the probe sees the negative start row) softmax it in both.
    """
    start = chess.Board()
    other = chess.Board("6k1/8/8/8/8/8/8/7K w - - 0 1")
    enc_start, enc_other = _encode(start), _encode(other)
    index = _feature_index(enc_start, enc_other)
    path = tmp_path / "logits_probe.onnx"
    _constant_row_model(path, int(enc_start.shape[0]), index, _NEGATIVE, _SIMPLEX)

    legacy_alone = _legacy_batch_to_wdl_probs(_SIMPLEX[None, :])
    legacy_pair = _legacy_batch_to_wdl_probs(np.stack([_NEGATIVE, _SIMPLEX]))
    assert not np.allclose(legacy_alone[0], legacy_pair[1])

    expected = _softmax_rows(np.stack([_NEGATIVE, _SIMPLEX]))
    for kind in ("logits", "auto"):
        alone_other = _score(path, [other], 1, kind)
        pair = _score(path, [start, other], 2, kind)
        perm = _score(path, [other, start], 2, kind)
        split = _score(path, [start, other], 1, kind)
        np.testing.assert_allclose(alone_other[0], expected[1], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(pair, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(perm, expected[::-1], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(split, expected, rtol=1e-5, atol=1e-6)
        assert not np.allclose(alone_other[0], legacy_alone[0])


def test_explicit_probabilities_do_not_softmax_a_neighbour_of_negative_logits(
    tmp_path: Path,
) -> None:
    start = chess.Board()
    other = chess.Board("6k1/8/8/8/8/8/8/7K w - - 0 1")
    enc_start, enc_other = _encode(start), _encode(other)
    path = tmp_path / "declared_probs.onnx"
    _constant_row_model(
        path, int(enc_start.shape[0]), _feature_index(enc_start, enc_other),
        _NEGATIVE, _SIMPLEX,
    )
    raw = np.stack([_NEGATIVE, _SIMPLEX])
    pair = _score(path, [start, other], 2, "probabilities")
    alone = _score(path, [other], 1, "probabilities")
    np.testing.assert_allclose(pair, raw, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(alone[0], _SIMPLEX, rtol=1e-5, atol=1e-6)


def test_auto_follows_the_start_position_when_that_row_looks_like_probabilities(
    tmp_path: Path,
) -> None:
    """Residual of auto: a simplex probe freezes probabilities for every row.

    The run is batch-invariant, and it is not the logits reading. Callers that
    know the head is logits pass ``--wdl-output-kind logits``.
    """
    start = chess.Board()
    other = chess.Board("6k1/8/8/8/8/8/8/7K w - - 0 1")
    enc_start, enc_other = _encode(start), _encode(other)
    path = tmp_path / "simplex_probe.onnx"
    _constant_row_model(
        path, int(enc_start.shape[0]), _feature_index(enc_start, enc_other),
        _SIMPLEX, _NEGATIVE,
    )
    raw = np.stack([_SIMPLEX, _NEGATIVE])
    pair = _score(path, [start, other], 2, "auto")
    split = _score(path, [start, other], 1, "auto")
    np.testing.assert_allclose(pair, raw, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(split, raw, rtol=1e-5, atol=1e-6)
    logits = _score(path, [start, other], 1, "logits")
    assert not np.allclose(logits[0], pair[0])
