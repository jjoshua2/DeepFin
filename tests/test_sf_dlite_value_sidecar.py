"""Synthetic small-bank D-lite joins and an injected UCI engine; no SF run."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import chess
import numpy as np
import pytest

from scripts import sf_dlite_value_sidecar as dlite
from scripts import sf_dlite_smallbank as small


OPENING = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5",
           "c2c3", "g8f6", "d2d3", "d7d6", "e1g1", "e8g8",
           "f1e1", "f8e8", "b1d2", "c6b8"]


def fixture(*, row_index: int = 0, source: str = "Ceres-v8"):
    game = {"uid_prefix": ["a" * 64, source, "root", 7],
            "source_proof_sha256": "b" * 64,
            "uid_ply_offset": 16 if source == "BT4-v9" else 0,
            "root_start_fen": chess.STARTING_FEN,
            "opening_uci": OPENING,
            "played_uci": ["a2a4", "a7a5"]}
    board = chess.Board()
    for move in OPENING + game["played_uci"][:row_index]:
        board.push_uci(move)
    uid = [*game["uid_prefix"], row_index + game["uid_ply_offset"]]
    stack_sha = dlite.digest(dlite.canonical(OPENING + game["played_uci"][:row_index]))
    winner = {"uid": uid, "input_digest": "c" * 64,
              "context": [stack_sha, "0:0:0", board.halfmove_clock, "legal", "query"]}
    proof = {"uid": uid, "input_digest": winner["input_digest"],
             "source_proof_sha256": game["source_proof_sha256"],
             "row_index": row_index, "full_history_sha256": stack_sha,
             "pov_white": bool(board.turn), "rule50": board.halfmove_clock,
             "fen": board.fen()}
    row = dlite.prepare_row(winner, proof, game)
    return row, winner, proof, game


class FakeEngine:
    def __init__(self, *_args, **kwargs) -> None:
        if kwargs:
            assert kwargs["hash_mb"] == 8
            assert kwargs["threads"] == 1
            assert kwargs["syzygy_50_move_rule"] is True
            assert kwargs["syzygy_probe_limit"] == 6
            assert kwargs["retain_syzygy_on_new_game"] is True
        self.syzygy_ready_after_requests = True
        self.retain_syzygy_option_sent = True
        self.closed = False
        self.resets = 0

    def close(self) -> None:
        self.closed = True


class FakeSearcher:
    def __init__(self, *, engine: FakeEngine, **kwargs) -> None:
        if kwargs:
            assert kwargs["cp_slope"] == .006
            assert kwargs["cp_draw_width"] == 120
            assert kwargs["search_timeout_s"] == 15
        self.engine = engine
        self.fail = False

    def new_game(self) -> None:
        self.engine.resets += 1

    def stream(self, history, *, depth: int, multipv: int) -> list[str]:
        assert multipv == 1
        assert dlite.corpus.position_command(history).startswith("position fen ")
        if self.fail:
            return []
        board = chess.Board(history.fen)
        move = next(iter(board.legal_moves)).uci()
        return [f"info depth {depth} multipv 1 score cp 25 wdl 400 300 300 nodes 100 pv {move}"]


def selected(row: dlite.PreparedRow):
    policy = np.zeros(1858, dtype="<f2")
    policy[0] = 1
    wdl = np.asarray([.7, .2, .1], dtype="<f2")
    target = policy.tobytes() + wdl.tobytes()
    mask = np.zeros(1858, dtype=np.uint8)
    mask[0] = 1
    label = {"uid": list(row.uid), "input_digest": row.input_digest,
             "pov_white": row.pov_white, "target_sha256": dlite.digest(target)}
    return label, target, mask


def test_history_identity_orientation_and_game_offset() -> None:
    for source in ("BT4-v9", "Ceres-v8", "SF-d6"):
        row, winner, proof, game = fixture(row_index=1, source=source)
        assert row.uid[4] == (17 if source == "BT4-v9" else 1)
        assert row.pov_white is False
        bad = copy.deepcopy(proof)
        bad["pov_white"] = True
        with pytest.raises(dlite.Hold, match="orientation"):
            dlite.prepare_row(winner, bad, game)
        bad = copy.deepcopy(proof)
        bad["rule50"] += 1
        with pytest.raises(dlite.Hold, match="rule50"):
            dlite.prepare_row(winner, bad, game)
        bad = copy.deepcopy(proof)
        bad["row_index"] = 0
        with pytest.raises(dlite.Hold, match="ply"):
            dlite.prepare_row(winner, bad, game)


def test_history_proof_and_missing_moves_refuse() -> None:
    _row, winner, proof, game = fixture()
    bad = copy.deepcopy(proof)
    bad["input_digest"] = "d" * 64
    with pytest.raises(dlite.Hold, match="input digest"):
        dlite.prepare_row(winner, bad, game)
    bad = copy.deepcopy(proof)
    bad["full_history_sha256"] = "d" * 64
    with pytest.raises(dlite.Hold, match="history stack"):
        dlite.prepare_row(winner, bad, game)
    bad = copy.deepcopy(game)
    bad["opening_uci"][3] = "a1a8"
    with pytest.raises(dlite.Hold, match="history"):
        dlite.prepare_row(winner, proof, bad)
    bad = copy.deepcopy(game)
    bad["played_uci"] = []
    with pytest.raises(dlite.Hold, match="complete selected-game"):
        dlite.prepare_row(winner, proof, bad)


def test_v6_compact_chain_adapter_exact_row_and_duplicate_refusal() -> None:
    row, winner, proof, game = fixture(row_index=1, source="BT4-v9")
    winner["source"] = "BT4-v9"
    chain_entry = {"uid": proof["uid"],
                   "input_digest": proof["input_digest"],
                   "history_stack_sha256": proof["full_history_sha256"],
                   "pov_white": proof["pov_white"], "rule50": proof["rule50"]}
    report = {"source": "BT4-v9", "game_id": 7, "root_id": "root",
              "history_chain": {"schema": "tri_source_full_game_history_chain_v1",
                                "root_start_fen": game["root_start_fen"],
                                "opening_uci": game["opening_uci"],
                                "played_uci": game["played_uci"],
                                "row_index": [chain_entry]}}
    encoded = dlite.canonical(report) + b"\n"
    prepared = dlite.prepare_v6_winner(winner, encoded)
    assert prepared.history.fen == row.history.fen
    assert prepared.source_proof_sha256 == dlite.digest(encoded)
    report["history_chain"]["row_index"].append(chain_entry)
    with pytest.raises(dlite.Hold, match="unique"):
        dlite.prepare_v6_winner(winner, dlite.canonical(report))


def test_cold_scalar_and_main_wdl_only() -> None:
    row, _, _, _ = fixture()
    engine = FakeEngine()
    searcher = FakeSearcher(engine=engine)
    raw = dlite.label_one(row, searcher, depth=8)
    assert engine.resets == 1
    assert raw["score"]["cp"] == 25
    assert raw["score"]["native_wdl_permille"] == [400, 300, 300]
    sel, target, mask = selected(row)
    candidate, receipt = dlite.attach_main_wdl(row, raw, sel, target, mask)
    assert candidate[:1858 * 2] == target[:1858 * 2]
    chosen = np.frombuffer(target[1858 * 2:], dtype="<f2").astype(np.float32)
    sf = np.asarray(raw["score"]["d_style_wdl"])
    expected = ((sf + 2 * chosen) / 3).astype("<f2")
    assert candidate[1858 * 2:] == expected.tobytes()
    assert receipt["sf_auxiliary_targets"] is False


def test_missing_score_mismatch_and_illegal_mask_refuse() -> None:
    row, _, _, _ = fixture()
    engine = FakeEngine()
    searcher = FakeSearcher(engine=engine)
    raw = dlite.label_one(row, searcher, depth=8)
    sel, target, mask = selected(row)
    bad = copy.deepcopy(raw)
    bad["score"]["cp"] = None
    with pytest.raises(dlite.Hold, match="missing scalar"):
        dlite.attach_main_wdl(row, bad, sel, target, mask)
    bad = copy.deepcopy(raw)
    bad["pov_white"] = not row.pov_white
    with pytest.raises(dlite.Hold, match="orientation"):
        dlite.attach_main_wdl(row, bad, sel, target, mask)
    bad = copy.deepcopy(raw)
    bad["score"]["d_style_wdl"] = [.1, .2, .7]
    with pytest.raises(dlite.Hold, match="D-calibrated"):
        dlite.attach_main_wdl(row, bad, sel, target, mask)
    bad = copy.deepcopy(raw)
    bad["score"]["native_wdl_permille"] = None
    with pytest.raises(dlite.Hold, match="complete raw"):
        dlite.attach_main_wdl(row, bad, sel, target, mask)
    bad = copy.deepcopy(raw)
    bad["score"]["move"] = "a1a8"
    with pytest.raises(dlite.Hold, match="complete raw"):
        dlite.attach_main_wdl(row, bad, sel, target, mask)
    mask[0] = 0
    with pytest.raises(dlite.Hold, match="legal mask"):
        dlite.attach_main_wdl(row, raw, sel, target, mask)


def test_engine_ownership_and_no_partial_success(tmp_path: Path) -> None:
    row, _, _, _ = fixture()
    engine_file = tmp_path / "stockfish"
    engine_file.write_bytes(b"synthetic verified binary")
    sf_sha = hashlib.sha256(engine_file.read_bytes()).hexdigest()
    tb1, tb2 = tmp_path / "tb1", tmp_path / "tb2"
    tb1.mkdir()
    tb2.mkdir()
    engine = FakeEngine()
    good = dlite.label_bank([row], depth=8, stockfish=engine_file,
                            stockfish_sha256=sf_sha, syzygy_path=f"{tb1}:{tb2}",
                            engine_factory=lambda *_a, **_kw: engine,
                            searcher_factory=FakeSearcher)
    assert len(good) == 1
    assert engine.closed
    assert engine.resets == 1
    engine = FakeEngine()
    class FailingSearcher(FakeSearcher):
        def stream(self, _history, *, depth: int, multipv: int) -> list[str]:
            assert depth == 8
            assert multipv == 1
            return []
    with pytest.raises((dlite.Hold, RuntimeError), match=r"score|search"):
        dlite.label_bank([row], depth=8, stockfish=engine_file,
                         stockfish_sha256=sf_sha, syzygy_path=f"{tb1}:{tb2}",
                         engine_factory=lambda *_a, **_kw: engine,
                         searcher_factory=FailingSearcher)
    assert engine.closed
    with pytest.raises(dlite.Hold, match="binary bytes"):
        dlite.label_bank([row], depth=8, stockfish=engine_file,
                         stockfish_sha256="0" * 64, syzygy_path=f"{tb1}:{tb2}",
                         engine_factory=lambda *_a, **_kw: FakeEngine(),
                         searcher_factory=FakeSearcher)


def test_executable_small_bank_label_attach_and_resume_refusal(tmp_path: Path) -> None:
    _row, winner, proof, game = fixture(source="Ceres-v8")
    winner["source"] = "Ceres-v8"
    report = {"source": "Ceres-v8", "root_id": "root", "game_id": 7,
              "history_chain": {"schema": "tri_source_full_game_history_chain_v1",
                                "root_start_fen": game["root_start_fen"],
                                "opening_uci": game["opening_uci"],
                                "played_uci": game["played_uci"],
                                "row_index": [{"uid": proof["uid"],
                                               "input_digest": proof["input_digest"],
                                               "history_stack_sha256": proof["full_history_sha256"],
                                               "pov_white": proof["pov_white"],
                                               "rule50": proof["rule50"]}]}}
    winner_file, proof_file = tmp_path / "WINNERS.jsonl", tmp_path / "PROOFS.jsonl"
    winner_file.write_bytes(dlite.canonical(winner) + b"\n")
    proof_file.write_bytes(dlite.canonical(report) + b"\n")
    winner_sha = dlite.digest(winner_file.read_bytes())
    proof_sha = dlite.digest(proof_file.read_bytes())
    engine_file = tmp_path / "stockfish"
    engine_file.write_bytes(b"synthetic verified binary")
    tb1, tb2 = tmp_path / "tb1", tmp_path / "tb2"
    tb1.mkdir()
    tb2.mkdir()
    out = tmp_path / "labels"
    status = small.run_label(winners=winner_file, winners_sha256=winner_sha,
                             proofs=proof_file, proofs_sha256=proof_sha,
                             stockfish=engine_file,
                             stockfish_sha256=dlite.digest(engine_file.read_bytes()),
                             syzygy_path=f"{tb1}:{tb2}", depth=8, out=out,
                             engine_factory=FakeEngine, searcher_factory=FakeSearcher)
    assert status["rows"] == 1
    assert status["status"] == "COMPLETE_DIAGNOSTIC_ZERO_CORPUS_CREDIT"
    with pytest.raises(dlite.Hold, match="one-shot"):
        small.run_label(winners=winner_file, winners_sha256=winner_sha,
                         proofs=proof_file, proofs_sha256=proof_sha,
                         stockfish=engine_file,
                         stockfish_sha256=dlite.digest(engine_file.read_bytes()),
                         syzygy_path=f"{tb1}:{tb2}", depth=8, out=out,
                         engine_factory=FakeEngine, searcher_factory=FakeSearcher)
    prepared = small.load_small_bank(winner_file.read_bytes(), proof_file.read_bytes())[0]
    selected_meta, target, mask = selected(prepared)
    selected_meta.update({"teacher": "Ceres", "target_hex": target.hex(),
                          "legal_mask_hex": mask.tobytes().hex()})
    selected_file = tmp_path / "SELECTED.jsonl"
    selected_file.write_bytes(dlite.canonical(selected_meta) + b"\n")
    attached = small.run_attach(winners=winner_file, winners_sha256=winner_sha,
                                 proofs=proof_file, proofs_sha256=proof_sha,
                                 labels=out / "LABELS.jsonl",
                                 labels_sha256=status["labels_sha256"],
                                 selected=selected_file,
                                 selected_sha256=dlite.digest(selected_file.read_bytes()),
                                 out=tmp_path / "attached")
    assert attached["rows"] == 1
    candidate = (tmp_path / "attached" / "CANDIDATE_TARGETS.jsonl").read_bytes()
    assert dlite.digest(candidate) == attached["candidate_targets_sha256"]
    assert json.loads(candidate)["selected_teacher"] == "Ceres"
