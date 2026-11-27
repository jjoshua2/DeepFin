"""Synthetic-only bounded descriptive row reconstruction contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import chess
import numpy as np
import pytest
import zarr

from scripts import match_audit_rows as match
from scripts import stored_wdl_rows as reader
from chess_anti_engine.replay.shard import POLICY_SIZE


def fixture(tmp_path: Path) -> tuple[bytes, tuple[Path, ...]]:
    boards = [chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1"),
              chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")]
    rows = np.stack([match.encode_cboard(
        match.CBoard.from_board(board), input_history_encoding=match.STORED_HISTORY_ENCODING,
        input_extra_features=match.STORED_EXTRA_FEATURES).astype(np.float16) for board in boards])
    path = tmp_path / "one.zarr"
    group = zarr.open_group(str(path), mode="w")
    group.create_dataset("x", data=rows, chunks=(1, 175, 8, 8))
    group.create_dataset("policy_target", data=np.zeros((2, POLICY_SIZE), dtype=np.float32))
    group.create_dataset("wdl_target", data=np.ones(2, dtype=np.int64))
    group.create_dataset("game_id", data=np.array([7, 9], dtype=np.int64), chunks=(1,))
    group.create_dataset("has_game_id", data=np.ones(2, dtype=np.uint8), chunks=(1,))
    group.attrs["history_rep_fix"] = False
    group.attrs["input_history_encoding"] = match.STORED_HISTORY_ENCODING
    records = [{"key": match.position_key(board), "fen": board.fen(),
                "wdl": target} for board, target in zip(
                    boards, ([0.6, 0.3, 0.1], [0.2, 0.5, 0.3]), strict=True)]
    return ("\n".join(json.dumps(r) for r in reversed(records))+"\n").encode(), (path,)


def run(data: bytes, paths: tuple[Path, ...]) -> dict:
    return reader.scan_prefix(data, hashlib.sha256(data).hexdigest(), paths, expected_history_rep_fix=False)


def test_alignment_and_partial_prefix(tmp_path: Path) -> None:
    data, paths = fixture(tmp_path)
    result = run(data, paths)
    records = {r["key"]: r for r in map(json.loads, data.decode().splitlines())}
    assert result["candidate_universe_complete"] is False
    assert result["holdout_claim"] is False
    assert result["x"].dtype == np.float32
    assert result["x"].shape == (2, 175, 8, 8)
    for key, row, target in zip(result["row_ids"], result["x"], result["targets"], strict=True):
        decoded = match.decode_board_from_planes(row, input_history_encoding=match.STORED_HISTORY_ENCODING)
        assert decoded is not None
        assert match.position_key(decoded) == key
        np.testing.assert_array_equal(target, records[key]["wdl"])
    assert result["source_rows_scanned"] == 2
    assert result["decoded_chunk_bytes_charged"] == 2 * (175 * 64 * 2 + 8 + 1)
    assert len(result["observed_candidate_games"]) == 2


def test_selection_does_not_depend_on_labels_or_order(tmp_path: Path) -> None:
    data, _ = fixture(tmp_path)
    selected = reader.select_records(data, hashlib.sha256(data).hexdigest(), 1)[0]["record"]["key"]
    records = list(map(json.loads, data.decode().splitlines()))[::-1]
    for r in records:
        r["wdl"] = [0.0, 1.0, 0.0]
    changed = "\n".join(map(json.dumps, records)).encode()
    assert reader.select_records(changed, hashlib.sha256(changed).hexdigest(), 1)[0]["record"]["key"] == selected


@pytest.mark.parametrize("mutation", ["key", "fen", "wdl", "duplicate"])
def test_audit_admission(tmp_path: Path, mutation: str) -> None:
    data, paths = fixture(tmp_path)
    records = list(map(json.loads, data.decode().splitlines()))
    if mutation == "key":
        records[0]["key"] = "wrong"
    elif mutation == "fen":
        records[0]["fen"] = records[0]["fen"].replace(" w ", " b ")
    elif mutation == "wdl":
        records[0]["wdl"] = [333, 334, 333]
    else:
        records.append(records[0])
    changed = "\n".join(map(json.dumps, records)).encode()
    with pytest.raises(ValueError, match=r"canonical|simplex|unique"):
        run(changed, paths)


@pytest.mark.parametrize("plane", [0, 104])
def test_exact_plane_checks(tmp_path: Path, plane: int) -> None:
    data, paths = fixture(tmp_path)
    group = zarr.open_group(str(paths[0]), mode="a")
    x = np.asarray(group["x"][:], dtype=np.float32)
    if plane == 0:
        x[:, :12][x[:, :12] > 0.5] = 0.75
    else:
        x[:, 104:108][x[:, 104:108] > 0.5] = 0.75
    group["x"][:] = x
    with pytest.raises(ValueError, match="exact current/castling"):
        run(data, paths)


def test_wrong_encoding_and_unknown_identity(tmp_path: Path) -> None:
    data, paths = fixture(tmp_path)
    group = zarr.open_group(str(paths[0]), mode="a")
    group.attrs["history_rep_fix"] = False
    group.attrs["input_history_encoding"] = "legacy"
    with pytest.raises(ValueError, match="layout"):
        run(data, paths)
    group.attrs["history_rep_fix"] = False
    group.attrs["input_history_encoding"] = match.STORED_HISTORY_ENCODING
    group["has_game_id"][0] = 0
    with pytest.raises(ValueError, match="source identity incomplete"):
        run(data, paths)


def test_chunk_cap_precedes_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data, paths = fixture(tmp_path)
    path = paths[0] / "x" / ".zarray"
    declaration = json.loads(path.read_text())
    declaration["chunks"] = [100000, 175, 8, 8]
    path.write_text(json.dumps(declaration))
    monkeypatch.setattr(reader, "load_shard_arrays", lambda *a, **k: pytest.fail("loader called"))
    with pytest.raises(ValueError, match="chunk exceeds cap"):
        run(data, paths)


@pytest.mark.parametrize("limit", ["MAX_SCAN_ROWS", "MAX_DECODED_BYTES", "MAX_SECONDS"])
def test_budget_failure_never_returns_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str,
) -> None:
    data, paths = fixture(tmp_path)
    monkeypatch.setattr(reader, limit, 0)
    with pytest.raises(ValueError, match="budget"):
        run(data, paths)


def test_interruption_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data, paths = fixture(tmp_path)
    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt
    monkeypatch.setattr(reader, "load_shard_arrays", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(data, paths)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["one.zarr"]


def test_order_identity_and_no_backfill(tmp_path: Path) -> None:
    data, paths = fixture(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        reader.scan_prefix(data, "0"*64, paths, expected_history_rep_fix=False)
    with pytest.raises(ValueError, match="prefix"):
        run(data, (paths[0], paths[0]))
    with pytest.raises(ValueError, match="fewer than two"):
        reader.scan_prefix(data, hashlib.sha256(data).hexdigest(), paths, expected_history_rep_fix=False, count=1)


@pytest.mark.parametrize("actual", [None, True, 0, "false"])
def test_history_contract_rejected_before_array_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, actual) -> None:
    data, paths = fixture(tmp_path)
    attrs = paths[0] / ".zattrs"
    values = json.loads(attrs.read_text())
    if actual is None:
        values.pop("history_rep_fix")
    else:
        values["history_rep_fix"] = actual
    attrs.write_text(json.dumps(values))
    def forbidden(*_args, **_kwargs):
        pytest.fail("arrays loaded despite mismatched metadata")
    monkeypatch.setattr(reader, "load_shard_arrays", forbidden)
    with pytest.raises(ValueError, match="history_rep_fix"):
        run(data, paths)


def test_expected_history_bool_and_attribute_cap(tmp_path: Path) -> None:
    data, paths = fixture(tmp_path)
    with pytest.raises(ValueError, match="boolean"):
        reader.scan_prefix(data, hashlib.sha256(data).hexdigest(), paths, expected_history_rep_fix=cast(Any, 1))
    (paths[0] / ".zattrs").write_text(" " * (reader.MAX_ATTR_BYTES + 1))
    with pytest.raises(ValueError, match="oversized"):
        run(data, paths)
