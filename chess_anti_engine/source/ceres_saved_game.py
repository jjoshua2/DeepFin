"""CPU Ceres saved-game archive fixture with physical and semantic readback.

This tracked producer slice replays one pinned completed game into a fresh
unqualified archive. Syzygy endings require an owned strict six-man WDL/DTZ
handle, including during independent readback. It does not run a teacher.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import stat
import zipfile
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any, cast

import chess
import chess.syzygy
import numpy as np
import zarr
from numcodecs import Blosc

from chess_anti_engine.encoding.ceres_tpg import stored_x_to_ceres_tpg_bytes
from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.moves.leela_index import compact_index_for_move, leela_index_for_move
from chess_anti_engine.selfplay.bt4_outcome import decide_bt4_outcome

ARRAY_NAMES = ("x", "value", "value2", "legal_offsets", "legal_compact",
               "legal_leela", "legal_logits")
SCHEMA = "ceres_tracked_saved_game_fixture_v1_unqualified"
SOURCE_SCHEMA = "ceres_whole_game_legal_raw_zip_shard_v1_unqualified"
MAX_ROWS = 512
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_ZIP_MEMBERS = 512  # 8,192-row source shards have multiple Zarr chunks per array.
CODEC = Blosc(cname="zstd", clevel=2, shuffle=Blosc.BITSHUFFLE)


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= MAX_ARCHIVE_BYTES,
              "archive must be a bounded regular file")
        while block := os.read(fd, 1024 * 1024):
            digest.update(block)
    finally:
        os.close(fd)
    return digest.hexdigest()


def _zip_json(path: Path, name: str, *, compressed: bool) -> Any:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [info.filename for info in members]
        _need(len(names) <= MAX_ZIP_MEMBERS and len(names) == len(set(names))
              and all(not n.startswith("/") and ".." not in Path(n).parts for n in names),
              "unsafe or oversized archive member table")
        info = archive.getinfo(name)
        _need(info.file_size <= MAX_METADATA_BYTES and info.compress_size <= MAX_METADATA_BYTES,
              "archive metadata member exceeds cap")
        payload = archive.read(name)
    if compressed:
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
            payload = stream.read(MAX_METADATA_BYTES + 1)
    _need(len(payload) <= MAX_METADATA_BYTES, "decoded archive metadata exceeds cap")
    return json.loads(payload)


def _arrays_from_zip(path: Path, start: int, end: int) -> dict[str, np.ndarray]:
    store = zarr.ZipStore(str(path), mode="r")
    try:
        group = zarr.open_group(store=store, mode="r")
        _need(set(group.array_keys()) == set(ARRAY_NAMES), "archive array names differ")
        source_rows = cast(int, group["x"].shape[0])
        _need(group["x"].ndim == 4 and group["x"].shape[1:] == (175, 8, 8)
              and group["x"].dtype == np.float16
              and 0 <= start < end <= source_rows <= 8192
              and end - start <= MAX_ROWS
              and group["legal_offsets"].shape == (source_rows + 1,)
              and group["legal_offsets"].dtype == np.uint32
              and all(group[key].shape == (source_rows, 3)
                      and group[key].dtype == np.float16 for key in ("value", "value2")),
              "archive declared stored row shape/count differs")
        offsets = np.asarray(group["legal_offsets"][start:end + 1], dtype=np.uint32)
        lo, hi = int(offsets[0]), int(offsets[-1])
        _need(0 <= lo < hi <= source_rows * 218,
              "archive legal span outside bounded row capacity")
        _need(all(group[key].ndim == 1 and cast(int, group[key].shape[0]) >= hi
                  for key in ("legal_compact", "legal_leela", "legal_logits")),
              "archive legal arrays shorter than selected game span")
        arrays = {name: np.asarray(group[name][start:end]) for name in
                  ("x", "value", "value2")}
        arrays["legal_offsets"] = offsets - lo
        for name in ("legal_compact", "legal_leela", "legal_logits"):
            arrays[name] = np.asarray(group[name][lo:hi])
        return arrays
    finally:
        store.close()


@dataclass(frozen=True)
class SavedCeresGame:
    arrays: dict[str, np.ndarray]
    rows: list[dict[str, Any]]
    game: dict[str, Any]
    model_sha256: str
    source_archive_sha256: str


def _check_game(
    game: SavedCeresGame, *, syzygy_path: str | None = None,
    match_tablebase: chess.syzygy.Tablebase | None = None,
) -> None:
    arrays, rows, meta = game.arrays, game.rows, game.game
    n = len(rows)
    _need(0 < n <= MAX_ROWS and set(arrays) == set(ARRAY_NAMES), "game rows/arrays outside cap")
    _need(arrays["x"].shape == (n, 175, 8, 8) and arrays["x"].dtype == np.float16
          and arrays["value"].shape == arrays["value2"].shape == (n, 3)
          and arrays["value"].dtype == arrays["value2"].dtype == np.float16
          and arrays["legal_offsets"].shape == (n + 1,)
          and arrays["legal_offsets"].dtype == np.uint32
          and all(arrays[key].dtype == np.uint16 for key in ("legal_compact", "legal_leela"))
          and arrays["legal_logits"].dtype == np.float16,
          "saved game array shape or dtype differs")
    offsets = arrays["legal_offsets"]
    legal_n = len(arrays["legal_compact"])
    _need(int(offsets[0]) == 0 and int(offsets[-1]) == legal_n
          and len(arrays["legal_leela"]) == len(arrays["legal_logits"]) == legal_n
          and bool(np.all(np.diff(offsets.astype(np.int64)) > 0))
          and legal_n <= n * 218 and all(np.isfinite(arrays[key]).all()
                                        for key in ("x", "value", "value2", "legal_logits")),
          "saved game legal offsets or finite values differ")
    _need(meta["row_start"] == 0 and meta["row_end"] == n
          and meta["termination"] in ("natural", "syzygy") and meta["result"] in
          ("1-0", "0-1", "1/2-1/2")
          and meta["outcome_provenance"]["mode"] == "rule50_match_v1",
          "fixture supports one completed rule50-aware game")
    is_syzygy = meta["termination"] == "syzygy"
    if is_syzygy:
        provenance = meta["outcome_provenance"]
        _need(match_tablebase is not None and bool(syzygy_path)
              and provenance["syzygy_path"] == syzygy_path
              and provenance["max_pieces"] == 6
              and provenance["handle_contract"] == "caller_owned_capacity_checked"
              and provenance["wdl_table_count"] == len(match_tablebase.wdl)
              and provenance["dtz_table_count"] == len(match_tablebase.dtz),
              "saved Syzygy game needs its exact strict six-man WDL/DTZ handle")
    else:
        _need(match_tablebase is None and syzygy_path is None,
              "natural fixture does not use a Syzygy handle")
    board = chess.Board(meta["initial_replay_root_fen"])
    history: list[str] = []
    for uci in meta["initial_history_uci"]:
        _need(chess.Move.from_uci(uci) in board.legal_moves, "invalid initial history")
        board.push_uci(uci)
        history.append(uci)
    _need(board.fen() == meta["initial_fen"], "initial history/FEN mismatch")
    feeds = stored_x_to_ceres_tpg_bytes(
        arrays["x"], input_history_encoding="lc0_root_legacy_meta", history_rep_fix=True)
    for i, row in enumerate(rows):
        if is_syzygy:
            assert syzygy_path is not None
            assert match_tablebase is not None
            _need(decide_bt4_outcome(
                board, plies=i, max_plies=n + 1, syzygy_path=syzygy_path,
                outcome_mode="rule50_match_v1", match_tablebase=match_tablebase,
            ) is None, f"saved Syzygy game adjudicated before row {i}")
        else:
            _need(board.outcome(claim_draw=True) is None,
                  f"natural game ended before row {i}")
        x = arrays["x"][i]
        expected_x = encode_position(
            board, input_history_encoding="lc0_root_legacy_meta",
            input_extra_features="v2_threats",
        ).astype(np.float16)
        _need(np.array_equal(x, expected_x)
              and row["row_index"] == i and row["ply_index"] == i
              and row["game_id"] == meta["game_id"]
              and row["source_namespace"] == meta["source_namespace"]
              and row["source_shard"] == meta["source_shard"]
              and row["root_fen"] == board.fen()
              and row["pov_white"] is bool(board.turn)
              and row["history_stack_sha256"] == _sha(_canonical(history))
              and row["x_stored_sha256"] == _sha(x.tobytes())
              and row["stored_input_key"] == blake2b(
                  np.ascontiguousarray(x, dtype=np.float32).tobytes(),
                  digest_size=16).hexdigest()
              and row["feed_sha256"] == _sha(feeds[i].tobytes()),
              f"saved game row/input/history/feed mismatch at {i}")
        lo, hi = map(int, offsets[i:i + 2])
        moves = row["legal_moves_uci_sorted"]
        _need(len(moves) == hi - lo and len(set(moves)) == len(moves)
              and set(moves) == {move.uci() for move in board.legal_moves}
              and 0 <= row["played_sorted_index"] < len(moves)
              and moves[row["played_sorted_index"]] == row["played_move_uci"],
              f"saved game legal moves differ at {i}")
        parsed = [chess.Move.from_uci(uci) for uci in moves]
        compact = np.asarray([compact_index_for_move(board, move) for move in parsed],
                             dtype=np.uint16)
        leela = np.asarray([leela_index_for_move(board, move) for move in parsed],
                           dtype=np.uint16)
        _need(np.array_equal(compact, arrays["legal_compact"][lo:hi])
              and np.array_equal(leela, arrays["legal_leela"][lo:hi])
              and bool(np.all(np.diff(compact.astype(np.int32)) > 0)),
              f"saved game legal index maps differ at {i}")
        board.push_uci(row["played_move_uci"])
        history.append(row["played_move_uci"])
    _need(board.fen() == meta["terminal_fen"], "saved game terminal FEN differs")
    if is_syzygy:
        assert syzygy_path is not None
        assert match_tablebase is not None
        decision = decide_bt4_outcome(
            board, plies=n, max_plies=n + 1, syzygy_path=syzygy_path,
            outcome_mode="rule50_match_v1", match_tablebase=match_tablebase,
        )
        _need(decision is not None and decision.termination == "syzygy"
              and decision.result == meta["result"] and decision.detail == meta["detail"],
              "strict Syzygy terminal/result/detail differs")
    else:
        outcome = board.outcome(claim_draw=True)
        _need(outcome is not None and outcome.result() == meta["result"]
              and ("detail" not in meta or outcome.termination.name.lower() == meta["detail"]),
              "natural game terminal/result differs")


def load_saved_game(
    path: Path, expected_sha256: str, game_id: int, *, syzygy_path: str | None = None,
    match_tablebase: chess.syzygy.Tablebase | None = None,
) -> SavedCeresGame:
    """Select one complete game from a pinned legacy ZIP without scanning other arrays."""
    path = Path(path)
    _need(_file_sha(path) == expected_sha256, "saved source archive SHA differs")
    manifest = _zip_json(path, "manifest.json", compressed=False)
    rows = _zip_json(path, "rows.json.gz", compressed=True)
    games = _zip_json(path, "games.json.gz", compressed=True)
    _need(manifest["schema"] == SOURCE_SCHEMA and manifest["training_ready"] is False
          and manifest["source_qualified"] is False and len(rows) == manifest["row_count"],
          "unsupported saved source archive")
    matched = [g for g in games if g["game_id"] == game_id]
    _need(len(matched) == 1, "saved source game absent or duplicate")
    meta = dict(matched[0])
    start, end = meta["row_start"], meta["row_end"]
    _need(type(start) is int and type(end) is int and 0 <= start < end <= len(rows)
          and end - start <= MAX_ROWS, "saved source game row span outside cap")
    arrays = _arrays_from_zip(path, start, end)
    meta["row_start"], meta["row_end"] = 0, end - start
    selected_rows = [{**row, "row_index": i} for i, row in enumerate(rows[start:end])]
    game = SavedCeresGame(arrays, selected_rows, meta, manifest["model_sha256"],
                          expected_sha256)
    _check_game(game, syzygy_path=syzygy_path, match_tablebase=match_tablebase)
    _need(_file_sha(path) == expected_sha256, "saved source archive changed during read")
    return game


def load_saved_natural_game(path: Path, expected_sha256: str, game_id: int) -> SavedCeresGame:
    """Keep the original natural-only API for existing fixture callers."""
    return load_saved_game(path, expected_sha256, game_id)


def _read_written(
    path: Path, receipt: dict[str, Any], *, syzygy_path: str | None = None,
    match_tablebase: chess.syzygy.Tablebase | None = None,
) -> SavedCeresGame:
    _need(_file_sha(path) == receipt["archive_sha256"], "published archive SHA differs")
    manifest = _zip_json(path, "manifest.json", compressed=False)
    rows = _zip_json(path, "rows.json.gz", compressed=True)
    game_meta = _zip_json(path, "game.json.gz", compressed=True)
    _need(manifest["schema"] == SCHEMA and manifest["training_ready"] is False
          and manifest["source_qualified"] is False
          and manifest["row_count"] == len(rows) == receipt["rows"]
          and manifest["source_archive_sha256"] == receipt["source_archive_sha256"]
          and manifest["model_sha256"] == receipt["model_sha256"]
          and manifest["game_id"] == game_meta["game_id"] == receipt["game_id"]
          and _sha(_canonical(rows)) == manifest["rows_sha256"]
          and _sha(_canonical(game_meta)) == manifest["game_sha256"],
          "published manifest/metadata receipt mismatch")
    arrays = _arrays_from_zip(path, 0, len(rows))
    _need({key: _sha(arrays[key].tobytes()) for key in ARRAY_NAMES}
          == manifest["array_sha256"], "published array digest mismatch")
    game = SavedCeresGame(arrays, rows, game_meta, manifest["model_sha256"],
                          manifest["source_archive_sha256"])
    _check_game(game, syzygy_path=syzygy_path, match_tablebase=match_tablebase)
    _need(_file_sha(path) == receipt["archive_sha256"],
          "published archive changed during readback")
    return game


def write_saved_game(
    game: SavedCeresGame, output: Path, *, syzygy_path: str | None = None,
    match_tablebase: chess.syzygy.Tablebase | None = None,
) -> dict[str, Any]:
    """Write one fresh ZIP, prove it, then publish without replacing a peer file."""
    _check_game(game, syzygy_path=syzygy_path, match_tablebase=match_tablebase)
    output = Path(output)
    _need(output.is_absolute() and not output.exists(), "fresh absolute output required")
    output.mkdir(parents=True)
    partial = output / "game.zarr.zip.writing"
    published = output / "game.zarr.zip"
    store = zarr.ZipStore(str(partial), mode="w", compression=zipfile.ZIP_STORED)
    try:
        group = zarr.group(store=store)
        for name, value in game.arrays.items():
            lead = 16384 if name in ("legal_compact", "legal_leela", "legal_logits") else 512
            chunks = (max(1, min(len(value), lead)), *value.shape[1:])
            group.create_dataset(name, data=value, chunks=chunks, compressor=CODEC)
        store["rows.json.gz"] = gzip.compress(_canonical(game.rows), mtime=0)
        store["game.json.gz"] = gzip.compress(_canonical(game.game), mtime=0)
    finally:
        store.close()
    manifest = {
        "schema": SCHEMA, "training_ready": False, "source_qualified": False,
        "game_id": game.game["game_id"], "row_count": len(game.rows),
        "model_sha256": game.model_sha256,
        "source_archive_sha256": game.source_archive_sha256,
        "rows_sha256": _sha(_canonical(game.rows)),
        "game_sha256": _sha(_canonical(game.game)),
        "array_sha256": {key: _sha(value.tobytes()) for key, value in game.arrays.items()},
    }
    with zipfile.ZipFile(partial, "a", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", _canonical(manifest))
    receipt = {"schema": SCHEMA, "game_id": game.game["game_id"],
               "rows": len(game.rows), "archive_sha256": _file_sha(partial),
               "model_sha256": game.model_sha256,
               "source_archive_sha256": game.source_archive_sha256}
    _read_written(partial, receipt, syzygy_path=syzygy_path,
                  match_tablebase=match_tablebase)
    os.link(partial, published, follow_symlinks=False)
    partial.unlink()
    readback = _read_written(published, receipt, syzygy_path=syzygy_path,
                             match_tablebase=match_tablebase)
    _need(all(np.array_equal(readback.arrays[key], game.arrays[key]) for key in ARRAY_NAMES)
          and readback.rows == game.rows and readback.game == game.game,
          "published decoded game differs from source")
    (output / "receipt.json").write_bytes(_canonical(receipt) + b"\n")
    return receipt


def read_saved_game_archive(
    output: Path, *, syzygy_path: str | None = None,
    match_tablebase: chess.syzygy.Tablebase | None = None,
) -> SavedCeresGame:
    """Independently reopen a published fixture through its physical receipt."""
    output = Path(output)
    _need(output.is_dir() and not output.is_symlink(), "published fixture directory missing")
    receipt_path = output / "receipt.json"
    fd = os.open(receipt_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode) and info.st_size <= 4096,
              "published fixture receipt missing or oversized")
        payload = os.read(fd, 4097)
        _need(len(payload) == info.st_size, "published fixture receipt short read")
        receipt = json.loads(payload)
    finally:
        os.close(fd)
    _need(receipt["schema"] == SCHEMA, "published fixture receipt schema differs")
    return _read_written(output / "game.zarr.zip", receipt, syzygy_path=syzygy_path,
                         match_tablebase=match_tablebase)
