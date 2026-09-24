"""CPU-only source adaptation; no ONNX session or GPU device."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chess
import numpy as np
import pytest
import zarr

from chess_anti_engine.eval.rvg_surgery import position_fingerprints
from chess_anti_engine.moves.leela_index import compact_index_for_move
from scripts.bt4_own_game_teacher_source import inspect_bank, sha_file, write_source


def _bank(root: Path, *, model: str, value: float,
          fullmove: int = 1) -> tuple[Path, str, Path, str]:
    root.mkdir()
    games = root / "games"
    games.mkdir()
    board = chess.Board()
    board.fullmove_number = fullmove
    x = np.zeros((1, 175, 8, 8), dtype=np.float32)
    x[0, 0, 0, 0] = value
    policy = np.zeros((1, 1858), dtype=np.float32)
    policy[0, compact_index_for_move(board, chess.Move.from_uci("e2e4"))] = 1
    wdl = np.asarray([[0.2, 0.3, 0.5]], dtype=np.float32)
    teacher = {
        "kind": "root_inference_no_search", "policy_encoding": "lc0_1858_compact",
        "policy_output": "/policy", "wdl_output": "/wdl", "wdl_kind": "probabilities",
        "wdl_dtype": "float32", "wdl_order": ["win", "draw", "loss"],
        "wdl_pov": "side_to_move", "input_name": "/x", "input_dtype": "float32",
        "input_history_encoding": "lc0_root_legacy_meta", "input_extra_features": "v2_threats",
        "history_rep_fix": True, "model_sha256": model,
    }
    row = {
        "index": 0, "ply_index": board.ply(), "fen": board.fen(), "pov_white": True,
        "move_uci": "e2e4", "temperature": 1.0, "wdl_target": 0,
        "input_key": hashlib.blake2b(x[0].tobytes(), digest_size=16).hexdigest(),
        "source_key": position_fingerprints(x, input_history_encoding="lc0_root_legacy_meta")[0].hex(),
        "teacher": teacher,
    }
    metadata = {
        "schema": "bt4_root_policy_games_v1", "game_id": 0, "initial_fen": board.fen(),
        "status": "completed", "result": "1-0", "rows": [row],
        "attempted_plies": 1, "discarded_rows": 0, "termination": "syzygy",
        "outcome_provenance": {"mode": "rule50_match_v1"},
    }
    game = games / "game_00000000.npz"
    np.savez_compressed(game, x=x, policy_t1=policy, wdl_raw=wdl,
                        metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
    launch = {
        "schema": "bt4_root_policy_games_v1", "outcome_mode": "rule50_match_v1",
        "games": 1, "initial_fen": board.fen(),
        "input_history_encoding": "lc0_root_legacy_meta", "input_extra_features": "v2_threats",
        "history_rep_fix": True,
        "teacher_observation": "root_inference_compact_t1_policy_and_native_wdl_unmodified_by_outcome",
        "model": {"sha256": model, "requested_provider": "cpu"},
    }
    (root / "launch.json").write_text(json.dumps(launch))
    summary = {
        "schema": "bt4_root_policy_games_v1", "status": "complete",
        "launch_sha256": sha_file(root / "launch.json"),
        "provider_proof_sha256": None, "rows_emitted": 1, "rows_attempted": 1,
        "games": 1, "completed": 1, "discarded": {},
        "game_files": [{"path": game.name, "sha256": sha_file(game),
                        "game_id": 0, "status": "completed", "rows": 1,
                        "discarded_rows": 0}],
    }
    (root / "summary.json").write_text(json.dumps(summary))
    pin = sha_file(root / "summary.json")
    audit = root / "test-audit.json"
    audit.write_text(json.dumps({
        "status": "PASS_INDEPENDENT_BT4_V4_READBACK", "new_summary_sha256": pin,
        "new_accepted_rows": 1,
        "overall_status": "PASS_BT4_ROOT_CUDA_7_TO_6_CONSERVATIVE_CONTINUATION",
        "source_origins_verified": ["synthetic-test-only"],
    }))
    return root, pin, audit, sha_file(audit)


def test_same_game_id_in_two_banks_has_distinct_source_ids(tmp_path: Path) -> None:
    a, a_pin, a_audit, a_audit_pin = _bank(tmp_path / "a", model="a" * 64, value=1.0)
    b, b_pin, b_audit, b_audit_pin = _bank(tmp_path / "b", model="a" * 64, value=2.0,
                                           fullmove=10)
    banks = [inspect_bank(a, a_pin, a_audit, a_audit_pin),
             inspect_bank(b, b_pin, b_audit, b_audit_pin)]
    out = tmp_path / "source"
    result = write_source(banks, out)
    assert result["rows"] == 2
    assert result["training_ready"] is False
    g0 = zarr.open_group(str(out / "bank_0000.zarr"), mode="r")
    g1 = zarr.open_group(str(out / "bank_0001.zarr"), mode="r")
    assert int(g0["game_id"][0]) == int(g1["game_id"][0]) == 0
    assert int(g0["ply_index"][0]) == 0
    assert int(g1["ply_index"][0]) == 18
    assert json.loads((out / "bank_0001.rows.jsonl").read_text())["ply_index"] == 18
    assert not np.array_equal(g0["row_uid"][0], g1["row_uid"][0])
    assert np.array_equal(g0["bt4_policy"][0],
                          np.load(a / "games" / "game_00000000.npz")["policy_t1"][0])
    assert np.array_equal(g1["bt4_wdl_raw"][0],
                          np.load(b / "games" / "game_00000000.npz")["wdl_raw"][0])
    assert (out / "manifest.json").is_file()
    assert not (out / "incomplete.json").exists()


def test_changed_game_refused_with_incomplete_stage(tmp_path: Path) -> None:
    path, pin, audit, audit_pin = _bank(tmp_path / "a", model="a" * 64, value=1.0)
    indexed = inspect_bank(path, pin, audit, audit_pin)
    with (path / "games" / "game_00000000.npz").open("ab") as stream:
        stream.write(b"changed")
    out = tmp_path / "source"
    with pytest.raises(ValueError, match="changed before materialization"):
        write_source([indexed], out)
    assert not out.exists()
    assert (tmp_path / "source.writing" / "incomplete.json").is_file()


def test_incompatible_teacher_and_summary_pin_refused(tmp_path: Path) -> None:
    a, a_pin, a_audit, a_audit_pin = _bank(tmp_path / "a", model="a" * 64, value=1.0)
    b, b_pin, b_audit, b_audit_pin = _bank(tmp_path / "b", model="b" * 64, value=2.0)
    with pytest.raises(ValueError, match="summary pin"):
        inspect_bank(a, "0" * 64, a_audit, a_audit_pin)
    with pytest.raises(ValueError, match="audit pin"):
        inspect_bank(a, a_pin, a_audit, "0" * 64)
    with pytest.raises(FileNotFoundError):
        inspect_bank(a, a_pin, tmp_path / "absent-audit.json", "0" * 64)
    with pytest.raises(ValueError, match="mix teacher"):
        write_source([inspect_bank(a, a_pin, a_audit, a_audit_pin),
                      inspect_bank(b, b_pin, b_audit, b_audit_pin)], tmp_path / "source")
    assert not (tmp_path / "source").exists()


@pytest.mark.parametrize("duplicate", [False, True])
def test_game_receipt_coverage_refused(tmp_path: Path, duplicate: bool) -> None:
    bank, _, audit, audit_pin = _bank(tmp_path / "a", model="a" * 64, value=1.0)
    summary_path = bank / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["game_files"] = summary["game_files"] * 2 if duplicate else []
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match=r"game receipt|game/row accounting"):
        inspect_bank(bank, sha_file(summary_path), audit, audit_pin)
