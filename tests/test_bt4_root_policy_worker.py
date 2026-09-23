"""CPU checks for the opt-in BT4 game writer and strict worker loop."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

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
        metadata = json.loads(archive["metadata"].tobytes().decode())
        arrays = {key: archive[key].copy() for key in ("x", "policy_t1", "wdl_raw")}
    return metadata, arrays


def test_capture_worker_banks_complete_raw_teacher_and_terminal_target(
    tmp_path: Path,
) -> None:
    evaluator = FakeEvaluator()
    summary = worker.run_worker(
        spec(tmp_path), evaluator, FakeTablebase(),  # type: ignore[arg-type]
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
        assert meta["rows"][0]["teacher"]["kind"] == "raw_root_inference"
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
        short, FakeEvaluator("e2e4"), FakeTablebase(),  # type: ignore[arg-type]
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
            spec(tmp_path), FakeEvaluator(), FakeTablebase(missing=True),  # type: ignore[arg-type]
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
            spec(tmp_path), FakeEvaluator(), FakeTablebase(),  # type: ignore[arg-type]
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
        worker.run_worker(current, ChangingEvaluator(), FakeTablebase())  # type: ignore[arg-type]
    assert len(list((tmp_path / "run" / "games").glob("*.npz"))) == 2
    assert not (tmp_path / "run" / "summary.json").exists()


def test_bounded_buffer_and_explicit_mode_before_output(tmp_path: Path) -> None:
    bad = replace(spec(tmp_path), outcome_mode="theoretical_wdl")
    with pytest.raises(ValueError, match="explicit rule50"):
        worker.run_worker(bad, FakeEvaluator(), FakeTablebase())  # type: ignore[arg-type]
    assert not bad.out.exists()
    oversized = replace(spec(tmp_path), max_plies=3000)
    with pytest.raises(ValueError, match="4096"):
        oversized.validate()
    too_many = replace(spec(tmp_path), games=worker.MAX_GAMES + 1)
    with pytest.raises(ValueError, match="games <= 32"):
        too_many.validate()
