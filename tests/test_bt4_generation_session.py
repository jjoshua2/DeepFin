"""CPU-only verified-session and exact feed provenance contracts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np
import onnx
import pytest

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.encoding.lc0 import x_to_lc0_planes
from scripts import bt4_generation_evaluator as adapter
from scripts import bt4_generation_session as factory
from scripts.bt4_policy_dump import file_sha256
from scripts.gen_sf_rooted_corpus import input_tensor_key

HISTORY = "lc0_root_legacy_meta"
FEATURES = "v2_threats"


@dataclass
class Meta:
    name: str
    shape: list[int | str]
    type: str


class FakeSession:
    def __init__(self, *, input_type: str = "tensor(float16)") -> None:
        self.inputs = [Meta("planes", ["batch", 112, 8, 8], input_type)]
        self.outputs = [Meta("wdl", ["batch", 3], "tensor(double)"),
                        Meta("policy", ["batch", 1858], "tensor(float)")]
        self.providers = ["CPUExecutionProvider"]
        self.options: dict[str, dict[str, str]] = {"CPUExecutionProvider": {"arena": "1"}}
        self.calls: list[tuple[list[str], np.ndarray]] = []

    def get_inputs(self) -> list[Meta]:
        return self.inputs

    def get_outputs(self) -> list[Meta]:
        return self.outputs

    def get_providers(self) -> list[str]:
        return self.providers

    def get_provider_options(self) -> dict[str, dict[str, str]]:
        return self.options

    def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert names == ["policy", "wdl"]
        rows = feed["planes"].copy()
        self.calls.append((names, rows))
        policy = np.broadcast_to(np.linspace(-2, 2, 1858, dtype=np.float32),
                                 (len(rows), 1858)).copy()
        wdl = np.broadcast_to(np.array([0.5, 0.3, 0.2], dtype=np.float64),
                              (len(rows), 3)).copy()
        return [policy, wdl]


@pytest.fixture(autouse=True)
def configured_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test the read-only guard without flipping native state in this process.
    monkeypatch.setattr(rep_fix, "current", lambda: True)


def tiny_model(path: Path, *, external: bool = False, external_sparse: bool = False) -> str:
    x = onnx.helper.make_tensor_value_info("planes", onnx.TensorProto.FLOAT16,
                                           ["batch", 112, 8, 8])
    policy = onnx.helper.make_tensor_value_info("policy", onnx.TensorProto.FLOAT,
                                                ["batch", 1858])
    wdl = onnx.helper.make_tensor_value_info("wdl", onnx.TensorProto.DOUBLE,
                                             ["batch", 3])
    initializers: list[onnx.TensorProto] = []
    if external:
        tensor = onnx.TensorProto()
        tensor.name = "external_weights"
        tensor.data_type = onnx.TensorProto.FLOAT
        tensor.dims.extend([1])
        tensor.data_location = onnx.TensorProto.EXTERNAL
        entry = tensor.external_data.add()
        entry.key = "location"
        entry.value = "external.bin"
        initializers.append(tensor)
    graph = onnx.helper.make_graph([], "test", [x], [policy, wdl], initializers)
    if external_sparse:
        values = onnx.TensorProto()
        values.name = "sparse_external"
        values.data_type = onnx.TensorProto.FLOAT
        values.dims.extend([1])
        values.data_location = onnx.TensorProto.EXTERNAL
        entry = values.external_data.add()
        entry.key = "location"
        entry.value = "sparse_external.bin"
        sparse = onnx.SparseTensorProto()
        sparse.values.CopyFrom(values)
        sparse.indices.CopyFrom(onnx.helper.make_tensor("indices", onnx.TensorProto.INT64, [1], [0]))
        sparse.dims.extend([1])
        graph.sparse_initializer.append(sparse)
    onnx.save_model(onnx.helper.make_model(graph), str(path))
    return file_sha256(path)


def open_fake(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> None:
    def opener(_path: str, *, gpu_mem_gb: float, threads: int) -> tuple[Any, str, np.dtype[Any], list[str]]:
        del gpu_mem_gb, threads
        dtype = np.dtype("float16" if session.inputs[0].type == "tensor(float16)" else "float32")
        return session, session.inputs[0].name, dtype, session.providers
    monkeypatch.setattr(factory, "open_session", opener)
    monkeypatch.setattr(factory, "remap_provenance", lambda: {
        "commit": "a" * 40, "dirty": False,
        "blobs": {"chess_anti_engine/encoding/lc0.py": "b" * 40},
    })


def build(path: Path, sha: str) -> factory.VerifiedBT4Session:
    return factory.open_verified_bt4_session(
        path, expected_sha256=sha, gpu_mem_gb=0, threads=1,
        policy_output="policy", wdl_output="wdl", wdl_kind="probabilities",
        input_history_encoding=HISTORY, input_extra_features=FEATURES,
        history_rep_fix=True,
    )


def test_verified_session_stamps_exact_batched_feeds_and_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    sha = tiny_model(model)
    session = FakeSession()
    open_fake(monkeypatch, session)
    verified = build(model, sha)
    assert verified.provenance.onnx_sha256 == sha
    assert verified.provenance.providers == ("CPUExecutionProvider",)
    assert verified.provenance.input_shape == ("batch", 112, 8, 8)
    assert verified.provenance.policy_shape == ("batch", 1858)
    assert verified.provenance.wdl_shape == ("batch", 3)
    assert verified.provenance.provider_options == (("CPUExecutionProvider", (("arena", "1"),)),)
    assert verified.provenance.wdl_order == ("win", "draw", "loss")
    assert verified.provenance.wdl_pov == "side_to_move"
    assert verified.provenance.remap_blobs == (("chess_anti_engine/encoding/lc0.py", "b" * 40),)
    boards = [chess.Board(), chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")]
    xs = np.stack([encode_cboard(CBoard.from_board(board), input_history_encoding=HISTORY,
                                 input_extra_features=FEATURES) for board in boards])
    roots = verified.evaluator.evaluate_roots(boards, xs)
    assert len(session.calls) == 1
    assert session.calls[0][1].dtype == np.dtype("float16")
    assert [root.input_key for root in roots] == [input_tensor_key(x) for x in xs]
    assert [root.verified_session_sha256 for root in roots] == [verified.provenance.sha256] * 2
    actual_feed = session.calls[0][1]
    assert [root.onnx_feed_sha256 for root in roots] == [
        adapter._feed_row_sha256("planes", row) for row in actual_feed
    ]
    assert roots[0].onnx_feed_sha256 != roots[1].onnx_feed_sha256
    assert roots[0].wdl_raw.dtype == np.dtype("float64")
    np.testing.assert_array_equal(actual_feed, x_to_lc0_planes(xs,
        input_history_encoding=HISTORY).astype(np.float16))


@pytest.mark.parametrize(("fault", "match"), [
    ("extra_input", "exactly one input"),
    ("input_type", r"tensor\(float16\) or tensor\(float\)"),
    ("input_shape", r"\[dynamic-or-1,112,8,8\]"),
    ("fixed_batch", r"\[dynamic-or-1,112,8,8\]"),
    ("policy_shape", r"floating \[batch,1858\]"),
    ("policy_type", r"floating \[batch,1858\]"),
    ("policy_fixed_batch", "fixed output batch conflicts with input"),
    ("wdl_shape", r"native floating \[batch,3\]"),
    ("wdl_fixed_batch", "fixed output batch conflicts with input"),
    ("wdl_missing", "WDL output absent"),
    ("provider_options", "provider options disagree"),
])
def test_bad_session_contract_rejected_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, match: str,
) -> None:
    model = tmp_path / "model.onnx"
    sha = tiny_model(model)
    session = FakeSession()
    if fault == "extra_input":
        session.inputs.append(Meta("other", ["batch", 112, 8, 8], "tensor(float)"))
    elif fault == "input_type":
        session.inputs[0].type = "tensor(double)"
    elif fault == "input_shape":
        session.inputs[0].shape = ["batch", 111, 8, 8]
    elif fault == "fixed_batch":
        session.inputs[0].shape = [2, 112, 8, 8]
    elif fault == "policy_shape":
        session.outputs[1].shape = ["batch", 1857]
    elif fault == "policy_type":
        session.outputs[1].type = "tensor(int64)"
    elif fault == "policy_fixed_batch":
        session.outputs[1].shape = [1, 1858]
    elif fault == "wdl_shape":
        session.outputs[0].shape = ["batch", 4]
    elif fault == "wdl_fixed_batch":
        session.outputs[0].shape = [1, 3]
    elif fault == "wdl_missing":
        session.outputs[0].name = "wrong"
    else:
        session.options = {}
    open_fake(monkeypatch, session)
    with pytest.raises(ValueError, match=match):
        build(model, sha)
    assert session.calls == []


def test_ambiguous_policy_head_is_a_factory_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    sha = tiny_model(model)
    session = FakeSession()
    session.outputs.append(Meta("second_policy", ["batch", 1858], "tensor(float)"))
    open_fake(monkeypatch, session)
    with pytest.raises(ValueError, match="policy head cannot be resolved"):
        factory.open_verified_bt4_session(
            model, expected_sha256=sha, gpu_mem_gb=0, threads=1,
            policy_output=None, wdl_output="wdl", wdl_kind="probabilities",
            input_history_encoding=HISTORY, input_extra_features=FEATURES,
            history_rep_fix=True,
        )
    assert session.calls == []


def test_realized_provider_options_change_session_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    sha = tiny_model(model)
    session = FakeSession()
    open_fake(monkeypatch, session)
    first = build(model, sha).provenance
    session.options["CPUExecutionProvider"]["arena"] = "0"
    second = build(model, sha).provenance
    assert first.provider_options != second.provider_options
    assert first.sha256 != second.sha256


def test_fixed_singleton_batch_is_stamped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    sha = tiny_model(model)
    session = FakeSession()
    session.inputs[0].shape = [1, 112, 8, 8]
    open_fake(monkeypatch, session)
    fixed_session = build(model, sha)
    fixed = fixed_session.provenance
    boards = [chess.Board(), chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")]
    xs = np.stack([encode_cboard(CBoard.from_board(board), input_history_encoding=HISTORY,
                                 input_extra_features=FEATURES) for board in boards])
    with pytest.raises(ValueError, match="fixed ONNX batch size"):
        fixed_session.evaluator.evaluate_roots(boards, xs)
    assert session.calls == []
    assert fixed_session.evaluator.root_calls == 0
    session.inputs[0].shape = ["batch", 112, 8, 8]
    dynamic = build(model, sha).provenance
    assert fixed.input_shape == (1, 112, 8, 8)
    assert fixed.sha256 != dynamic.sha256


def test_fixed_output_shape_is_stamped_with_fixed_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    sha = tiny_model(model)
    session = FakeSession()
    session.inputs[0].shape = [1, 112, 8, 8]
    session.outputs[0].shape = [1, 3]
    session.outputs[1].shape = [1, 1858]
    open_fake(monkeypatch, session)
    fixed = build(model, sha).provenance
    assert fixed.policy_shape == (1, 1858)
    assert fixed.wdl_shape == (1, 3)


def test_artifact_hash_and_external_data_refused_before_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    sha = tiny_model(model)
    session = FakeSession()
    open_fake(monkeypatch, session)
    opens = 0
    def opener(_path: str, *, gpu_mem_gb: float, threads: int) -> Any:
        nonlocal opens
        del gpu_mem_gb, threads
        opens += 1
        raise AssertionError("must not open")
    monkeypatch.setattr(factory, "open_session", opener)
    with pytest.raises(ValueError, match="does not match expected pin"):
        build(model, "0" * 64)
    external_sha = tiny_model(model, external=True)
    with pytest.raises(ValueError, match="external tensor data is not pinned"):
        build(model, external_sha)
    sparse_sha = tiny_model(model, external_sparse=True)
    with pytest.raises(ValueError, match="external tensor data is not pinned"):
        build(model, sparse_sha)
    assert sha != external_sha
    assert sparse_sha != sha
    assert opens == 0


def test_artifact_change_and_cuda_fallback_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    sha = tiny_model(model)
    session = FakeSession()
    open_fake(monkeypatch, session)
    def changing(_path: str, *, gpu_mem_gb: float, threads: int) -> Any:
        del gpu_mem_gb, threads
        model.write_bytes(model.read_bytes() + b"changed")
        return session, "planes", np.dtype("float16"), session.providers
    monkeypatch.setattr(factory, "open_session", changing)
    with pytest.raises(ValueError, match="changed while session opened"):
        build(model, sha)
    sha = tiny_model(model)
    open_fake(monkeypatch, session)
    with pytest.raises(ValueError, match="CUDA requested but not realized"):
        factory.open_verified_bt4_session(
            model, expected_sha256=sha, gpu_mem_gb=0.25, threads=1,
            policy_output="policy", wdl_output="wdl", wdl_kind="probabilities",
            input_history_encoding=HISTORY, input_extra_features=FEATURES,
            history_rep_fix=True,
        )
    assert session.calls == []


def test_cuda_cap_and_device_are_verified_before_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    sha = tiny_model(model)
    session = FakeSession()
    session.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session.options["CUDAExecutionProvider"] = {}
    open_fake(monkeypatch, session)
    def open_gpu() -> factory.VerifiedBT4Session:
        return factory.open_verified_bt4_session(
            model, expected_sha256=sha, gpu_mem_gb=0.25, threads=1,
            policy_output="policy", wdl_output="wdl", wdl_kind="probabilities",
            input_history_encoding=HISTORY, input_extra_features=FEATURES,
            history_rep_fix=True,
        )
    with pytest.raises(ValueError, match="memory limit is unavailable"):
        open_gpu()
    session.options["CUDAExecutionProvider"] = {
        "gpu_mem_limit": "123", "device_id": "0",
    }
    with pytest.raises(ValueError, match="differs from requested cap"):
        open_gpu()
    session.options["CUDAExecutionProvider"] = {
        "gpu_mem_limit": str(256 * 1024**2), "device_id": "1",
    }
    with pytest.raises(ValueError, match="requested device 0"):
        open_gpu()
    session.options["CUDAExecutionProvider"]["device_id"] = "0"
    verified = open_gpu()
    assert verified.provenance.providers[0] == "CUDAExecutionProvider"
    assert session.calls == []


def test_bad_mode_refuses_before_artifact_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model.onnx"
    sha = tiny_model(model)
    monkeypatch.setattr(rep_fix, "current", lambda: None)
    with pytest.raises(RuntimeError, match="repetition mode must be explicitly configured"):
        build(model, sha)


@pytest.mark.parametrize("bad_budget", [float("nan"), float("inf"), -0.1])
def test_bad_gpu_budget_refuses_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_budget: float,
) -> None:
    model = tmp_path / "model.onnx"
    sha = tiny_model(model)
    session = FakeSession()
    open_fake(monkeypatch, session)
    with pytest.raises(ValueError, match="GPU memory budget must be finite"):
        factory.open_verified_bt4_session(
            model, expected_sha256=sha, gpu_mem_gb=bad_budget, threads=1,
            policy_output="policy", wdl_output="wdl", wdl_kind="probabilities",
            input_history_encoding=HISTORY, input_extra_features=FEATURES,
            history_rep_fix=True,
        )
    assert session.calls == []


def test_feed_digest_uses_cast_bytes_without_conflating_original_inputs() -> None:
    a = np.array([1.0], dtype=np.float32)
    b = np.array([1.0001], dtype=np.float32)
    assert input_tensor_key(a) != input_tensor_key(b)
    a_feed, b_feed = a.astype(np.float16), b.astype(np.float16)
    assert a_feed.tobytes() == b_feed.tobytes()
    assert adapter._feed_row_sha256("planes", a_feed) == adapter._feed_row_sha256("planes", b_feed)
    assert adapter._feed_row_sha256("different", a_feed) != adapter._feed_row_sha256("planes", a_feed)
    assert adapter._feed_row_sha256("planes", a) != adapter._feed_row_sha256("planes", a_feed)
