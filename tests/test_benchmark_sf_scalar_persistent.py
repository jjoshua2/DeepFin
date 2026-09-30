"""CPU-only contract checks; no registered corpus or Stockfish is opened."""

from __future__ import annotations

import chess
import os
import pytest
import time
from types import SimpleNamespace

from scripts import benchmark_sf_scalar_persistent as bench
from scripts import gen_sf_rooted_corpus as corpus


def sample_row() -> dict:
    board = chess.Board()
    legal = [m.uci() for m in board.legal_moves]
    return {
        "schema": 3, "result": 1.0, "run": {"run_id": "throughput_d8_c4"},
        "fen": board.fen(), "history_root_fen": board.fen(), "history_uci": [],
        "history_root_reason": "game_start", "worker_id": 0, "game_id": 7,
        "ply": 0, "input_key": "key", "piece_count": 32, "game_phase": "opening",
        "phases": [{"index": 0, "depth_requested": 8, "searchmoves": None,
            "width_realized": len(legal), "per_depth": [{"depth": 8, "complete": True,
                "lines": [[i+1, move, float(i), 10] for i, move in enumerate(legal)]}]}],
    }


def test_reference_is_full_width_and_history_is_preserved() -> None:
    row = sample_row()
    item = bench.compact_row(row, "w00-00000.jsonl.zst", 1)
    assert item["banked_full_width_d8"]["move"] in {m.uci() for m in chess.Board().legal_moves}
    assert item["history_root_fen"] == row["history_root_fen"]
    assert item["history_uci"] == []
    assert bench.row_key(item) == bench.row_key(item)
    row["phases"][0]["per_depth"][0]["lines"].pop()
    with pytest.raises(ValueError, match="rank roster"):
        bench.compact_row(row, "w00-00000.jsonl.zst", 1)


def test_raw_cp_and_native_wdl_are_separate() -> None:
    lines = ["info depth 8 multipv 1 score cp 25 wdl 400 300 300 nodes 100 pv e2e4"]
    parsed = corpus.parse_depth_blocks(lines, expected_lines=1)
    result = bench.first_depth_score(lines, 8, parsed)
    assert result["cp"] == 25
    assert result["mate"] is None
    assert result["native_wdl_permille"] == (400, 300, 300)
    assert result["d_style_wdl"] != [0.4, 0.3, 0.3]
    with pytest.raises(ValueError, match="incomplete scalar"):
        bench.first_depth_score(lines, 10, parsed)


def test_cold_tt_per_depth() -> None:
    class FakeSearcher:
        resets = 0
        depths: list[int]

        def __init__(self) -> None:
            self.depths = []

        def new_game(self) -> None:
            self.resets += 1

        def stream(self, history: corpus.RowHistory, *, depth: int, multipv: int) -> list[str]:
            self.depths.append(depth)
            assert corpus.position_command(history).startswith("position fen ")
            assert multipv == 1
            return [f"info depth {depth} multipv 1 score cp 25 wdl 400 300 300 nodes 100 pv e2e4"]

    row = bench.compact_row(sample_row(), "w00-00000.jsonl.zst", 1)
    fake = FakeSearcher()
    records = [bench.search_one(fake, row, depth, 8) for depth in (8, 10, 6)]
    assert fake.resets == 3
    assert fake.depths == [8, 10, 6]
    assert [record["depth"] for record in records] == [8, 10, 6]


def test_fixed_depth_pass_times_inclusive_wall_and_rotates_order(tmp_path, monkeypatch) -> None:
    assert bench.depth_order(1) == (6, 8, 10)
    assert bench.depth_order(8) == (10, 8, 6)
    with pytest.raises(ValueError, match="unknown"):
        bench.depth_order(2)

    row = bench.compact_row(sample_row(), "w00-00000.jsonl.zst", 1)
    calls = []

    def fake_search(_searcher, item, depth, arm):
        calls.append((depth, arm, item["ply"]))
        return {"depth": depth, "arm_engines": arm,
                "reset_seconds": 0.001, "search_seconds": 0.002,
                "parse_seconds": 0.001,
                "score": {"move": item["banked_full_width_d8"]["move"],
                          "d_style_wdl": item["banked_full_width_d8"]["d_style_wdl"]},
                "banked_full_width_d8": item["banked_full_width_d8"]}

    monkeypatch.setattr(bench, "search_one", fake_search)
    searcher = SimpleNamespace(engine=SimpleNamespace(proc=SimpleNamespace(pid=os.getpid())))
    plan = {"rss_cap_bytes": 8 * 2**30, "output_bytes": 128 * 2**20,
            "read_timeout_seconds": 1, "search_timeout_seconds": 1}
    result = bench.run_depth_pass(plan, [row]*3, [searcher], 1, 6, tmp_path, time.monotonic()+10)
    assert calls == [(6, 1, 0)] * 3
    assert result["rows"] == 3
    assert result["depth_wall_seconds"] > 0
    assert result["observed_rows_per_depth_wall_second"] == pytest.approx(3 / result["depth_wall_seconds"])
    assert result["reset_seconds_sum_across_engines"] == pytest.approx(0.003)
    assert len((tmp_path / "arm1_d6.jsonl").read_text().splitlines()) == 3

    near = tmp_path / "near_deadline"
    near.mkdir()
    calls.clear()
    with pytest.raises(ValueError, match="insufficient run wall reserve"):
        bench.run_depth_pass(plan, [row], [searcher], 1, 6, near, time.monotonic()+2)
    assert calls == []


def test_history_mismatch_refused_before_search() -> None:
    row = bench.compact_row(sample_row(), "w00-00000.jsonl.zst", 1)
    row["fen"] = chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1").fen()
    with pytest.raises(ValueError, match="history does not reproduce"):
        bench.search_one(object(), row, 6, 1)


def test_search_sends_full_history() -> None:
    board = chess.Board()
    root = board.fen()
    for token in ("e2e4", "e7e5"):
        board.push_uci(token)
    raw = sample_row()
    raw["fen"] = board.fen()
    raw["history_root_fen"] = root
    raw["history_uci"] = ["e2e4", "e7e5"]
    raw["phases"][0]["width_realized"] = board.legal_moves.count()
    raw["phases"][0]["per_depth"][0]["lines"] = [
        [i+1, move.uci(), 0.0, 10] for i, move in enumerate(board.legal_moves)
    ]
    row = bench.compact_row(raw, "w00-00000.jsonl.zst", 1)

    class FakeSearcher:
        def new_game(self) -> None:
            pass

        def stream(self, history: corpus.RowHistory, *, depth: int, multipv: int) -> list[str]:
            assert corpus.position_command(history) == f"position fen {root} moves e2e4 e7e5"
            assert depth == 6
            assert multipv == 1
            return ["info depth 6 multipv 1 score cp 12 nodes 50 pv g1f3"]

    assert bench.search_one(FakeSearcher(), row, 6, 1)["score"]["move"] == "g1f3"


def test_existing_output_refuses_resume(tmp_path) -> None:
    out = tmp_path / "spent"
    out.mkdir()
    with pytest.raises(ValueError, match="resume refused"):
        bench.run({}, tmp_path / "unused", out)
