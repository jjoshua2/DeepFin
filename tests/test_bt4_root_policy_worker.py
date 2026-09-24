"""CPU checks for the opt-in BT4 game writer and strict worker loop."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import chess
import chess.syzygy
import numpy as np
import pytest

from chess_anti_engine import tablebase
from chess_anti_engine.eval.rvg_surgery import position_fingerprints
from chess_anti_engine.moves.leela_index import compact_index_for_move
from scripts import bt4_root_policy_worker as worker
from scripts.bt4_generation_evaluator import BT4RootOutput
from scripts.gen_sf_rooted_corpus import input_tensor_key


SEVEN = "7k/8/8/8/8/8/2p5/KQBNR3 w - - 0 1"
MODEL_SHA = "a" * 64


class FakeEvaluator:
    def __init__(self, preferred: str = "b1c2") -> None:
        self.preferred = preferred
        self.calls = 0

    def evaluate_roots(
        self, boards: list[chess.Board], x_batch: np.ndarray,
    ) -> list[BT4RootOutput]:
        self.calls += 1
        keys = position_fingerprints(
            x_batch, input_history_encoding=worker.INPUT_HISTORY_ENCODING,
        )
        outputs = []
        for board, x, key in zip(boards, x_batch, keys):
            policy = np.zeros((1858,), dtype=np.float32)
            policy[compact_index_for_move(board, chess.Move.from_uci(self.preferred))] = 1
            raw = np.array([0.05, 0.1, 0.85], dtype=np.float64)
            policy.flags.writeable = raw.flags.writeable = False
            outputs.append(BT4RootOutput(
                fen=board.fen(), input_key=input_tensor_key(x), source_key=key,
                policy_t1=policy, wdl_raw=raw, policy_output="policy",
                wdl_output="wdl", wdl_kind="probabilities", model_sha256=MODEL_SHA,
                input_name="planes", input_dtype="float32",
                input_history_encoding=worker.INPUT_HISTORY_ENCODING,
                input_extra_features=worker.INPUT_EXTRA_FEATURES, history_rep_fix=True,
            ))
        return outputs


class FakeTablebase:
    def __init__(self, *, missing: bool = False) -> None:
        self.wdl: dict[str, Any] = {"KQBNRvK": object()}
        self.dtz: dict[str, Any] = {"KQBNRvK": object()}
        self.missing = missing

    def probe_wdl(self, board: chess.Board) -> int:
        if self.missing:
            raise chess.syzygy.MissingTableError(board.fen())
        return -2  # Black to move loses after White's zeroing capture.

    def probe_dtz(self, board: chess.Board) -> int:
        if self.missing:
            raise chess.syzygy.MissingTableError(board.fen())
        return -1


def fake_tablebase(*, missing: bool = False) -> chess.syzygy.Tablebase:
    """Only this test seam presents the fake as an opened Syzygy handle."""
    return cast(chess.syzygy.Tablebase, cast(object, FakeTablebase(missing=missing)))


def spec(tmp_path: Path, *, fen: str = SEVEN, max_plies: int = 8) -> worker.WorkerSpec:
    wdl_dir = tmp_path / "wdl"
    dtz_dir = tmp_path / "dtz"
    wdl_dir.mkdir(exist_ok=True)
    dtz_dir.mkdir(exist_ok=True)
    (wdl_dir / "KQBNRvK.rtbw").write_bytes(b"fake WDL fixture")
    (dtz_dir / "KQBNRvK.rtbz").write_bytes(b"fake DTZ fixture")
    return worker.WorkerSpec(
        out=tmp_path / "run", games=2, seed=14, max_plies=max_plies,
        parallel_games=2, temperature=0, initial_fen=fen,
        syzygy_path=f"{wdl_dir}{os.pathsep}{dtz_dir}", model_sha256=MODEL_SHA,
        outcome_mode=worker.OUTCOME_MODE, model_path="fake.onnx",
        providers=("CPUExecutionProvider",),
    )


@pytest.fixture(autouse=True)
def history_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker.rep_fix, "current", lambda: True)


def read_game(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"metadata", "x", "policy_t1", "wdl_raw"}
        metadata = json.loads(archive["metadata"].tobytes().decode())
        arrays = {key: archive[key].copy() for key in ("x", "policy_t1", "wdl_raw")}
    return metadata, arrays


def test_capture_worker_banks_complete_raw_teacher_and_terminal_target(
    tmp_path: Path,
) -> None:
    evaluator = FakeEvaluator()
    summary = worker.run_worker(
        spec(tmp_path), evaluator, fake_tablebase(),
    )
    assert summary["status"] == "complete"
    assert summary["completed"] == 2
    assert summary["rows_emitted"] == 2
    assert evaluator.calls == 1
    assert len(summary["game_files"]) == 2
    for receipt in summary["game_files"]:
        meta, arrays = read_game(tmp_path / "run" / "games" / receipt["path"])
        assert meta["status"] == "completed"
        assert meta["result"] == "1-0"
        assert meta["rows"][0]["move_uci"] == "b1c2"
        assert meta["rows"][0]["wdl_target"] == 0
        assert meta["rows"][0]["teacher"]["kind"] == "root_inference_no_search"
        assert meta["rows"][0]["teacher"]["wdl_dtype"] == "float64"
        assert arrays["wdl_raw"].dtype == np.dtype("float64")
        assert arrays["wdl_raw"][0].tolist() == [0.05, 0.1, 0.85]
        assert arrays["x"].shape[0] == arrays["policy_t1"].shape[0] == 1
        assert np.count_nonzero(arrays["policy_t1"][0]) == 1
    launch = json.loads((tmp_path / "run" / "launch.json").read_text())
    assert launch["actor"] == "bt4_root_policy_no_search"
    assert launch["outcome_mode"] == worker.OUTCOME_MODE
    assert launch["minimum_emitted_root_pieces"] == 7
    assert not list((tmp_path / "run" / "games").glob("*.writing"))


def test_unresolved_game_discard_has_no_rows(tmp_path: Path) -> None:
    # White can play once, then the game reaches the explicit ply cap above TB.
    starting = chess.STARTING_FEN
    short = spec(tmp_path, fen=starting, max_plies=1)
    # Choose a legal opening move for this fixture.
    summary = worker.run_worker(
        short, FakeEvaluator("e2e4"), fake_tablebase(),
    )
    assert summary["rows_emitted"] == 0
    assert summary["rows_attempted"] == 2
    assert summary["discarded"] == {"max_plies_unresolved": 2}
    for receipt in summary["game_files"]:
        meta, arrays = read_game(tmp_path / "run" / "games" / receipt["path"])
        assert meta["status"] == "discarded"
        assert meta["discarded_rows"] == 1
        assert meta["rows"] == []
        assert arrays["x"].shape[0] == arrays["policy_t1"].shape[0] == 0


def test_missing_required_probe_aborts_without_game_or_completion(
    tmp_path: Path,
) -> None:
    with pytest.raises(tablebase.MatchTablebaseError, match="missing eligible"):
        worker.run_worker(
            spec(tmp_path), FakeEvaluator(), fake_tablebase(missing=True),
        )
    assert (tmp_path / "run" / "launch.json").exists()
    assert not (tmp_path / "run" / "summary.json").exists()
    assert not list((tmp_path / "run" / "games").iterdir())


def test_writer_failure_leaves_no_published_or_partial_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_after_partial_write(handle: Any, **_arrays: Any) -> None:
        handle.write(b"partial npz")
        raise OSError("simulated full disk")

    monkeypatch.setattr(worker.np, "savez_compressed", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated full disk"):
        worker.run_worker(
            spec(tmp_path), FakeEvaluator(), fake_tablebase(),
        )
    assert not list((tmp_path / "run" / "games").iterdir())
    assert not (tmp_path / "run" / "summary.json").exists()


def test_table_file_change_refuses_completion(tmp_path: Path) -> None:
    current = spec(tmp_path)

    class ChangingEvaluator(FakeEvaluator):
        def evaluate_roots(
            self, boards: list[chess.Board], x_batch: np.ndarray,
        ) -> list[BT4RootOutput]:
            rows = super().evaluate_roots(boards, x_batch)
            (tmp_path / "wdl" / "KQBNRvK.rtbw").write_bytes(b"changed fixture")
            return rows

    with pytest.raises(RuntimeError, match="inventory changed"):
        worker.run_worker(current, ChangingEvaluator(), fake_tablebase())
    assert len(list((tmp_path / "run" / "games").glob("*.npz"))) == 2
    assert not (tmp_path / "run" / "summary.json").exists()


def test_bounded_buffer_and_explicit_mode_before_output(tmp_path: Path) -> None:
    bad = replace(spec(tmp_path), outcome_mode="theoretical_wdl")
    with pytest.raises(ValueError, match="explicit rule50"):
        worker.run_worker(bad, FakeEvaluator(), fake_tablebase())
    assert not bad.out.exists()
    oversized = replace(spec(tmp_path), max_plies=3000)
    with pytest.raises(ValueError, match="4096"):
        oversized.validate()
    too_many = replace(spec(tmp_path), games=worker.MAX_GAMES + 1)
    with pytest.raises(ValueError, match="games <= 32"):
        too_many.validate()


def test_research_128x400_profile_is_exact_and_default_caps_remain(tmp_path: Path) -> None:
    base = spec(tmp_path)
    for parallel in (16, 32, 64):
        large = replace(base, games=128, max_plies=400, parallel_games=parallel)
        with pytest.raises(ValueError, match="games <= 32"):
            large.validate()
        replace(large, research_capacity_128x400=True).validate()
    for changes in ({"games": 127}, {"max_plies": 401}, {"parallel_games": 8},
                    {"parallel_games": 128}):
        kwargs = {"games": 128, "max_plies": 400, "parallel_games": 64,
                  "research_capacity_128x400": True, **changes}
        with pytest.raises(ValueError, match="research capacity profile"):
            replace(base, **kwargs).validate()
    with pytest.raises(TypeError, match="explicit bool"):
        replace(base, research_capacity_128x400="true").validate()


@pytest.mark.parametrize("enabled", [False, True])
def test_research_profile_cli_reaches_worker_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake model")
    seen: list[worker.WorkerSpec] = []

    def stop_at_validation(value: worker.WorkerSpec, **_kwargs: Any) -> None:
        seen.append(value)
        raise RuntimeError("stop before session")

    monkeypatch.setattr(worker.WorkerSpec, "validate", stop_at_validation)
    argv = [
        "bt4_root_policy_worker.py", "--out", str(tmp_path / "out"),
        "--onnx", str(model), "--syzygy-path", str(tmp_path),
        "--outcome-mode", worker.OUTCOME_MODE, "--wdl-output", "wdl",
        "--wdl-kind", "probabilities", "--policy-output", "policy",
        "--games", "128", "--seed", "1", "--max-plies", "400",
        "--parallel-games", "64", "--temperature", "0",
    ]
    if enabled:
        argv.append("--research-capacity-128x400")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="stop before session"):
        worker.main()
    assert len(seen) == 1
    assert seen[0].research_capacity_128x400 is enabled
    assert (seen[0].games, seen[0].parallel_games, seen[0].max_plies) == (128, 64, 400)


def test_effective_batch_histogram_records_underfilled_tail(tmp_path: Path) -> None:
    short = replace(spec(tmp_path), games=3, parallel_games=2, max_plies=1,
                    initial_fen=chess.STARTING_FEN)
    summary = worker.run_worker(short, FakeEvaluator("e2e4"), fake_tablebase())
    assert summary["requested_parallel_games"] == 2
    assert summary["effective_batch_size_histogram"] == {"1": 1, "2": 1}
    assert summary["inference_calls"] == 2
    assert summary["full_batch_calls"] == 1
    assert summary["underfilled_calls"] == 1
    assert summary["max_effective_batch_size"] == 2


def test_cuda_requires_realized_device_zero_and_bounded_arena(tmp_path: Path) -> None:
    base = spec(tmp_path)
    cuda = replace(
        base, requested_provider="cuda", gpu_mem_gb=1.0,
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        provider_options={"device_id": "0", "gpu_mem_limit": str(1024 ** 3)},
    )
    cuda.validate()
    assert Path.home() / "projects/chess/scratchpad/gpu0_experiment.lock" == worker.GPU_LOCK
    with pytest.raises(ValueError, match="realized CUDA"):
        replace(cuda, providers=("CPUExecutionProvider",)).validate()
    with pytest.raises(ValueError, match="memory cap"):
        replace(cuda, provider_options={"device_id": "0", "gpu_mem_limit": "0"}).validate()
    with pytest.raises(ValueError, match="8 GiB"):
        replace(cuda, gpu_mem_gb=9.0).validate()
    with pytest.raises(ValueError, match="zero GPU memory"):
        replace(base, gpu_mem_gb=1.0).validate()
    conservative = replace(
        cuda, cudnn_conv_algo_search="DEFAULT", cudnn_conv_use_max_workspace=0,
        provider_options={
            **cuda.provider_options,
            "cudnn_conv_algo_search": "DEFAULT", "cudnn_conv_use_max_workspace": "0",
        },
    )
    conservative.validate()
    with pytest.raises(ValueError, match="cuDNN algorithm search"):
        replace(conservative, provider_options=cuda.provider_options).validate()
    with pytest.raises(ValueError, match="cuDNN maximum workspace"):
        replace(
            conservative,
            provider_options={
                **cuda.provider_options, "cudnn_conv_algo_search": "DEFAULT",
            },
        ).validate()


def test_child_gpu_lease_fails_immediately_when_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker, "GPU_LOCK", tmp_path / "gpu0_experiment.lock")
    fd = worker.acquire_gpu_lock()
    try:
        with pytest.raises(RuntimeError, match="canonical GPU lock is busy"):
            worker.acquire_gpu_lock()
    finally:
        os.close(fd)


def test_cuda_session_readback_rejects_cpu_fallback_and_wrong_cap() -> None:
    class Session:
        def __init__(self, providers: list[str], limit: str) -> None:
            self.providers = providers
            self.limit = limit

        def get_providers(self) -> list[str]:
            return self.providers

        def get_provider_options(self) -> dict[str, dict[str, str]]:
            return {"CUDAExecutionProvider": {
                "device_id": "0", "gpu_mem_limit": self.limit,
            }}

    expected = str(2 * 1024 ** 3)
    assert worker.verify_cuda_session(
        Session(["CUDAExecutionProvider", "CPUExecutionProvider"], expected),
        gpu_mem_gb=2,
    ) == (("CUDAExecutionProvider", "CPUExecutionProvider"), {
        "device_id": "0", "gpu_mem_limit": expected,
    })
    with pytest.raises(RuntimeError, match="fell back"):
        worker.verify_cuda_session(Session(["CPUExecutionProvider"], expected), gpu_mem_gb=2)
    with pytest.raises(RuntimeError, match="memory cap"):
        worker.verify_cuda_session(Session(["CUDAExecutionProvider"], "0"), gpu_mem_gb=2)


@pytest.mark.parametrize(
    ("search", "workspace", "extra"),
    [
        (None, None, {}),
        ("DEFAULT", 0, {
            "cudnn_conv_algo_search": "DEFAULT", "cudnn_conv_use_max_workspace": "0",
        }),
    ],
)
def test_cuda_provider_controls_reach_real_session_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    search: str | None, workspace: int | None, extra: dict[str, str],
) -> None:
    import onnxruntime as ort

    seen: dict[str, Any] = {}

    class FakeOptions:
        intra_op_num_threads = 0
        enable_profiling = False
        profile_file_prefix = ""

    class FakeSession:
        def disable_fallback(self) -> None:
            pass

        def get_providers(self) -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def get_provider_options(self) -> dict[str, dict[str, str]]:
            cuda = seen["providers"][0][1]
            return {"CUDAExecutionProvider": {key: str(value) for key, value in cuda.items()}}

        def get_inputs(self) -> list[Any]:
            return [SimpleNamespace(name="planes", type="tensor(float)")]

    def fake_session(*_args: Any, **kwargs: Any) -> FakeSession:
        seen.update(kwargs)
        return FakeSession()

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(ort, "SessionOptions", FakeOptions)
    monkeypatch.setattr(ort, "InferenceSession", fake_session)
    _, _, _, providers, realized = worker.open_worker_session(
        "unused.onnx", requested_provider="cuda", gpu_mem_gb=2,
        threads=2, profile_prefix=tmp_path / "profile",
        cudnn_conv_algo_search=search,
        cudnn_conv_use_max_workspace=workspace,
    )
    expected = {"device_id": 0, "gpu_mem_limit": 2 * 1024 ** 3, **extra}
    assert seen["providers"] == [("CUDAExecutionProvider", expected), "CPUExecutionProvider"]
    assert seen["enable_fallback"] is False
    assert providers[0] == "CUDAExecutionProvider"
    assert realized == {key: str(value) for key, value in expected.items()}


def test_cuda_provider_controls_reject_invalid_or_ignored_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="algorithm search"):
        worker.open_worker_session(
            "unused.onnx", requested_provider="cuda", gpu_mem_gb=2,
            threads=2, profile_prefix=tmp_path / "profile",
            cudnn_conv_algo_search="FAST",
        )
    with pytest.raises(ValueError, match="maximum workspace"):
        worker.open_worker_session(
            "unused.onnx", requested_provider="cuda", gpu_mem_gb=2,
            threads=2, profile_prefix=tmp_path / "profile",
            cudnn_conv_use_max_workspace=2,
        )
    with pytest.raises(ValueError, match="CUDA settings"):
        worker.open_worker_session(
            "unused.onnx", requested_provider="cpu", gpu_mem_gb=0,
            threads=2, profile_prefix=None, cudnn_conv_algo_search="DEFAULT",
        )

    class IgnoringSession:
        def get_providers(self) -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def get_provider_options(self) -> dict[str, dict[str, str]]:
            return {"CUDAExecutionProvider": {
                "device_id": "0", "gpu_mem_limit": str(2 * 1024 ** 3),
            }}

    with pytest.raises(RuntimeError, match="algorithm search"):
        worker.verify_cuda_session(
            IgnoringSession(), gpu_mem_gb=2, cudnn_conv_algo_search="DEFAULT",
        )
    with pytest.raises(RuntimeError, match="maximum workspace"):
        worker.verify_cuda_session(
            IgnoringSession(), gpu_mem_gb=2, cudnn_conv_use_max_workspace=0,
        )


def test_cuda_cli_controls_reach_session_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake model")
    model_sha = worker.file_sha256(model)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(worker.rep_fix, "apply", lambda *args, **kwargs: None)

    class FakeTablebase:
        def close(self) -> None:
            pass

    monkeypatch.setattr(worker.tablebase, "open_strict_match_tablebase", lambda *args, **kwargs: FakeTablebase())
    monkeypatch.setattr(worker, "acquire_gpu_lock", lambda: 123)
    seen: dict[str, Any] = {}

    def stop_at_session(*_args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        raise RuntimeError("stop before model load")

    monkeypatch.setattr(worker, "open_worker_session", stop_at_session)
    monkeypatch.setattr(sys, "argv", [
        "bt4_root_policy_worker.py", "--out", str(tmp_path / "out"),
        "--onnx", str(model), "--syzygy-path", str(tmp_path),
        "--outcome-mode", worker.OUTCOME_MODE,
        "--wdl-output", "wdl", "--wdl-kind", "probabilities",
        "--policy-output", "policy", "--games", "2", "--seed", "1",
        "--max-plies", "8", "--temperature", "0",
        "--provider", "cuda", "--gpu-mem-gb", "2",
        "--expected-onnx-sha256", model_sha,
        "--expected-input-name", "planes", "--expected-input-dtype", "float32",
        "--cudnn-conv-algo-search", "DEFAULT",
        "--cudnn-conv-use-max-workspace", "0",
    ])
    with pytest.raises(RuntimeError, match="stop before model load"):
        worker.main()
    assert seen["cudnn_conv_algo_search"] == "DEFAULT"
    assert seen["cudnn_conv_use_max_workspace"] == 0


def test_cuda_model_schema_pins_named_float32_heads_and_planes() -> None:
    input_head = SimpleNamespace(name="/input/planes", type="tensor(float)", shape=["N", 112, 8, 8])
    policy_head = SimpleNamespace(name="/output/policy", type="tensor(float)", shape=["N", 1858])
    wdl_head = SimpleNamespace(name="/output/wdl", type="tensor(float)", shape=["N", 3])

    class Session:
        def get_inputs(self) -> list[Any]:
            return [input_head]

        def get_outputs(self) -> list[Any]:
            return [policy_head, wdl_head]

    def verify() -> None:
        worker.verify_cuda_model_schema(
            Session(), input_name="/input/planes", input_dtype="float32",
            policy_output="/output/policy", wdl_output="/output/wdl",
            wdl_kind="probabilities",
        )

    verify()
    policy_head.shape[-1] = 4672
    with pytest.raises(ValueError, match="named head"):
        verify()
    policy_head.shape[-1] = 1858
    input_head.shape[1] = 111
    with pytest.raises(ValueError, match="input name/type/planes"):
        verify()
    input_head.shape[1] = 112
    with pytest.raises(ValueError, match="probability WDL"):
        worker.verify_cuda_model_schema(
            Session(), input_name="/input/planes", input_dtype="float32",
            policy_output="/output/policy", wdl_output="/output/wdl",
            wdl_kind="logits",
        )


def test_cuda_open_refuses_missing_ep_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="no CUDAExecutionProvider"):
        worker.open_worker_session(
            "unused.onnx", requested_provider="cuda", gpu_mem_gb=1,
            threads=2, profile_prefix=tmp_path / "profile",
        )


def test_first_root_profile_requires_cuda_neural_compute(tmp_path: Path) -> None:
    profile_path = tmp_path / "ort-profile.json"
    events = [
        {"args": {"provider": "CUDAExecutionProvider", "op_name": "Conv"}},
        {"args": {"provider": "CPUExecutionProvider", "op_name": "Shape"}},
    ]
    profile_path.write_text(json.dumps(events))

    class Session:
        def end_profiling(self) -> str:
            return str(profile_path)

        def get_providers(self) -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    class Evaluator:
        def evaluate_roots(
            self, boards: list[chess.Board], x_batch: np.ndarray,
        ) -> list[BT4RootOutput]:
            _ = boards, x_batch
            return []

    out = tmp_path / "output"
    out.mkdir()
    qualified = worker.CudaQualifiedEvaluator(Evaluator(), Session(), out)
    assert qualified.evaluate_roots([], np.empty((0,), dtype=np.float32)) == []
    assert qualified.proof is not None
    assert qualified.proof["cuda_neural_nodes"] == 1
    assert qualified.proof["cpu_nodes"] == 1
    assert (out / "provider_profile.json").exists()
    assert (out / "provider_proof.json").exists()
    profile_path.write_text(json.dumps([
        {"args": {"provider": "CPUExecutionProvider", "op_name": "Conv"}},
    ]))
    bad_out = tmp_path / "bad_output"
    bad_out.mkdir()
    unqualified = worker.CudaQualifiedEvaluator(Evaluator(), Session(), bad_out)
    with pytest.raises(RuntimeError, match="no neural"):
        unqualified.evaluate_roots([], np.empty((0,), dtype=np.float32))
    assert not list(bad_out.iterdir())


def test_cuda_run_never_publishes_game_without_profile(tmp_path: Path) -> None:
    cuda = replace(
        spec(tmp_path), requested_provider="cuda", gpu_mem_gb=1.0,
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        provider_options={"device_id": "0", "gpu_mem_limit": str(1024 ** 3)},
    )
    profile = tmp_path / "cpu-only-profile.json"
    profile.write_text(json.dumps([
        {"args": {"provider": "CPUExecutionProvider", "op_name": "Conv"}},
    ]))

    class Session:
        def end_profiling(self) -> str:
            return str(profile)

        def get_providers(self) -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    qualified = worker.CudaQualifiedEvaluator(FakeEvaluator(), Session(), cuda.out)
    with pytest.raises(RuntimeError, match="no neural"):
        worker.run_worker(cuda, qualified, fake_tablebase())
    assert not list((tmp_path / "run" / "games").iterdir())
    assert not (tmp_path / "run" / "summary.json").exists()
