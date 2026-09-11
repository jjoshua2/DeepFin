from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

from scripts import avi_successor_sidecar as sidecar
from scripts import avi_value_rewrite as rewrite
from scripts import gen_sf_rooted_corpus as corpus


def _raw_row_and_ref() -> tuple[dict[str, Any], dict[str, Any], chess.Board]:
    corpus.apply_history_rep_fix()
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push_uci(uci)
    history = corpus.history_for(board)
    config = "a" * 64
    input_key = corpus.row_key(board)
    row: dict[str, Any] = {
        "schema": 3,
        "run": {"config_sha256": config},
        "worker_id": 2,
        "game_id": 7,
        "ply": 4,
        "input_key": input_key,
        "fen": board.fen(),
        **history.as_row_fields(),
    }
    ref: dict[str, Any] = {
        "source_config_sha256": config,
        "worker_id": 2,
        "game_id": 7,
        "ply": 4,
        "input_key": input_key,
    }
    return row, ref, board


def test_raw_history_replay_reconstructs_authenticated_board() -> None:
    row, ref, expected = _raw_row_and_ref()
    got = sidecar._raw_row_board(row, ref)  # noqa: SLF001 - behavioral seam under test
    assert got.fen() == expected.fen()
    assert tuple(move.uci() for move in got.move_stack) == tuple(
        move.uci() for move in expected.move_stack[-len(got.move_stack) :]
    )
    assert corpus.row_key(got) == ref["input_key"]


def test_raw_history_replay_rejects_tampered_window() -> None:
    row, ref, _expected = _raw_row_and_ref()
    moves = row["history_uci"]
    assert isinstance(moves, list)
    row["history_uci"] = [*moves[:-1], "b8a6"]
    with pytest.raises(ValueError, match="reproduce FEN|input_key"):
        sidecar._raw_row_board(row, ref)  # noqa: SLF001


def test_sidecar_names_are_canonical() -> None:
    assert sidecar.sidecar_name("shard_000123.zarr") == "shard_000123.avi_values.npz"
    with pytest.raises(ValueError, match="shard name"):
        sidecar.sidecar_name("../shard_000123.zarr")


def test_rewrite_identity_distinguishes_root_backup_and_dose() -> None:
    checkpoint = "b" * 64
    assert rewrite.value_scheme("root", 0.25) != rewrite.value_scheme("backup", 0.25)
    assert rewrite.value_scheme("backup", 0.25) != rewrite.value_scheme("backup", 0.5)
    source = rewrite.value_source("backup", 0.25, checkpoint)
    assert checkpoint in source
    assert "backup" in source


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_rewrite_alpha_rejects_invalid_values(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        rewrite.checked_alpha(alpha)


def test_complete_sidecar_manifest_is_required(tmp_path: Path) -> None:
    sf_root = tmp_path / "sf"
    sf_root.mkdir()
    payload = {
        "schema": sidecar.SCHEMA,
        "kind": sidecar.KIND,
        "status": "COMPLETE_SELECTION",
        "full_source_coverage": False,
        "source_dir": str(sf_root),
        "source_summary_sha256": "c" * 64,
        "selected_rows": 2,
        "source_rows": 3,
        "selected_shards": 1,
        "checkpoint_sha256": "d" * 64,
        "outputs": [],
    }
    path = tmp_path / sidecar.SUMMARY
    path.write_text(json.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="complete source collection"):
        rewrite._sidecar_manifest(  # noqa: SLF001
            path,
            digest,
            sf_root,
            "c" * 64,
            [{"path": "shard_000000.zarr", "rows": 3}],
        )


def test_blend_roundtrip_stays_simplex_before_float16_storage() -> None:
    anchor = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2]], dtype=np.float16)
    neural = np.asarray([[0.1, 0.1, 0.8], [0.4, 0.5, 0.1]], dtype=np.float32)
    mixed = rewrite.blend_wdl_targets(anchor, neural, alpha=0.25)
    assert np.isfinite(mixed).all()
    assert (mixed >= 0).all()
    np.testing.assert_allclose(mixed.sum(axis=1), 1.0, atol=1e-7)
