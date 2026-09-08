"""Tiny real ORT graphs expose original-byte feed, probe and policy mapping."""
from __future__ import annotations

import argparse
from pathlib import Path

import chess
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from chess_anti_engine.encoding.ceres_tpg import (
    encode_ceres_tpg_batch,
    encode_ceres_tpg_bytes,
)
from chess_anti_engine.moves.encode import move_to_index
from chess_anti_engine.moves.leela_index import leela_index_for_move
from chess_anti_engine.onnx.load import (
    INPUT_FORMAT_CERES_TPG,
    INPUT_FORMAT_LC0_PLANES,
    OnnxChessNet,
    encode_ceres_onnx_input,
)
from scripts.foreign_net_audit import _score_onnx, _session


def _graph(path: Path, dtype: int, *, extra_input: bool = False) -> Path:
    shape: list[str | int] = ["batch", 64, 137]
    inputs = [helper.make_tensor_value_info("squares", dtype, shape)]
    if extra_input:
        inputs.append(helper.make_tensor_value_info("state", TensorProto.FLOAT, ["batch", 1]))
    nodes = [helper.make_node("Cast", ["squares"], ["cast"], to=TensorProto.FLOAT)]
    initializers = [
        numpy_helper.from_array(np.array(100, np.float32), "scale"),
        numpy_helper.from_array(np.array([1, 2], np.int64), "axes"),
        numpy_helper.from_array(np.array([-1, 1], np.int64), "reshape"),
        numpy_helper.from_array(np.array(0.01, np.float32), "small"),
        numpy_helper.from_array(np.array(0, np.float32), "zero"),
        numpy_helper.from_array(np.arange(1858, dtype=np.float32)[None, :], "indices"),
    ]
    nodes.append(helper.make_node("Div" if dtype == TensorProto.UINT8 else "Identity",
                                  ["cast", "scale"] if dtype == TensorProto.UINT8 else ["cast"],
                                  ["normalized"]))
    nodes.extend([
        helper.make_node("ReduceSum", ["normalized", "axes"], ["sum"], keepdims=0),
        helper.make_node("Reshape", ["sum", "reshape"], ["column"]),
        helper.make_node("Mul", ["column", "small"], ["offset"]),
        helper.make_node("Add", ["indices", "offset"], ["policy32"]),
        helper.make_node("Cast", ["policy32"], ["policy"], to=TensorProto.FLOAT16),
        helper.make_node("Neg", ["offset"], ["negative"]),
        helper.make_node("Mul", ["offset", "zero"], ["zeros"]),
        helper.make_node("Concat", ["offset", "negative", "zeros"], ["value"], axis=1),
        helper.make_node("Identity", ["squares"], ["echo"]),
    ])
    outputs = [helper.make_tensor_value_info("policy", TensorProto.FLOAT16, ["batch", 1858]),
               helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch", 3]),
               helper.make_tensor_value_info("echo", dtype, shape)]
    graph = helper.make_graph(nodes, "ceres_byte_contract", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def _net(path: Path) -> OnnxChessNet:
    return OnnxChessNet(path, input_name="squares", policy_output_name="policy",
                        wdl_output_name="value", input_format=INPUT_FORMAT_CERES_TPG,
                        providers=["CPUExecutionProvider"], intra_op_num_threads=2)


def _boards() -> list[chess.Board]:
    history = chess.Board()
    for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
        history.push_uci(uci)
    return [history, chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
            chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
            chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1"),
            chess.Board("k7/8/7K/8/8/8/pppppppp/8 b - - 0 1")]


def test_actual_ort_byte_and_float_policy_probe_remap_and_batches(tmp_path: Path) -> None:
    byte_path = _graph(tmp_path / "byte.onnx", TensorProto.UINT8)
    float_path = _graph(tmp_path / "float.onnx", TensorProto.FLOAT)
    byte_net, float_net = _net(byte_path), _net(float_path)
    assert byte_net.wdl_output_kind == float_net.wdl_output_kind == "logits"
    np.testing.assert_array_equal(byte_net._canonical_probe_input()[0], encode_ceres_tpg_bytes(chess.Board()))
    boards = _boards()
    raw = np.stack([encode_ceres_tpg_bytes(b) for b in boards])
    assert raw.dtype == np.uint8
    np.testing.assert_array_equal(encode_ceres_onnx_input(boards, np.dtype(np.uint8)), raw)
    floating = encode_ceres_tpg_batch(boards)
    assert not np.array_equal(raw, floating.astype(np.uint8))
    got = byte_net(torch.from_numpy(raw))
    expected = float_net(torch.from_numpy(floating))
    for key in ("policy_own", "wdl"):
        torch.testing.assert_close(got[key], expected[key], rtol=0, atol=0)
        serial = torch.cat([byte_net(torch.from_numpy(row[None]))[key] for row in raw])
        torch.testing.assert_close(got[key], serial, rtol=0, atol=0)
    # Feed echo proves no normalization/cast before the byte graph consumes it.
    sess, name, dtype = _session(str(byte_path), 0, input_format=INPUT_FORMAT_CERES_TPG, ort_threads=2)
    assert dtype == np.uint8
    pol, echo = sess.run(["policy", "echo"], {name: raw})
    np.testing.assert_array_equal(echo, raw)
    for i, board in enumerate(boards):
        for move in board.legal_moves:
            assert got["policy_own"][i, move_to_index(move, board)] == pol[i, leela_index_for_move(board, move)]
    with pytest.raises(ValueError, match=r"original torch\.uint8"):
        byte_net(torch.from_numpy(floating))


@pytest.mark.parametrize(("input_type", "dtype"), [(TensorProto.UINT8, np.uint8),
                                             (TensorProto.FLOAT, np.float32),
                                             (TensorProto.FLOAT16, np.float16)])
def test_actual_audit_feed_and_full_policy_match_declared_graph(
    tmp_path: Path, input_type: int, dtype: type,
) -> None:
    path = _graph(tmp_path / "audit.onnx", input_type)
    boards = _boards()
    args = argparse.Namespace(onnx=str(path), gpu_mem_gb=0, input_format=INPUT_FORMAT_CERES_TPG,
                              ort_threads=2, batch_size=2, policy_output="policy", wdl_output="value")
    policy, wdl, heads = _score_onnx(boards, args)
    feed = (np.stack([encode_ceres_tpg_bytes(b) for b in boards]) if dtype == np.uint8
            else encode_ceres_tpg_batch(boards).astype(dtype))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    expected_policy, logits, echo = session.run(None, {"squares": feed})
    np.testing.assert_array_equal(echo, feed)
    np.testing.assert_array_equal(policy, expected_policy)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    np.testing.assert_allclose(wdl, probs, rtol=1e-6, atol=1e-7)
    assert heads == {"policy_output": "policy", "wdl_output": "value"}


@pytest.mark.parametrize("mode", ["extra", "int8", "wrong_name", "byte_lc0"])
def test_unsupported_graph_contract_refused_by_both_consumers(tmp_path: Path, mode: str) -> None:
    path = _graph(tmp_path / "invalid.onnx", TensorProto.INT8 if mode == "int8" else TensorProto.UINT8,
                  extra_input=mode == "extra")
    if mode == "wrong_name":
        with pytest.raises(ValueError, match="not found"):
            OnnxChessNet(path, input_name="absent", policy_output_name="policy", wdl_output_name="value",
                         input_format=INPUT_FORMAT_CERES_TPG, providers=["CPUExecutionProvider"],
                         intra_op_num_threads=2)
        return
    fmt = INPUT_FORMAT_LC0_PLANES if mode == "byte_lc0" else INPUT_FORMAT_CERES_TPG
    with pytest.raises(ValueError, match=r"exactly one|unsupported ONNX input dtype"):
        OnnxChessNet(path, input_name="squares", policy_output_name="policy", wdl_output_name="value",
                     input_format=fmt, providers=["CPUExecutionProvider"], intra_op_num_threads=2)
    with pytest.raises(ValueError, match=r"exactly one|unsupported ONNX input dtype"):
        _session(str(path), 0, input_format=fmt, ort_threads=2)
