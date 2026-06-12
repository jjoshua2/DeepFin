"""v1 -> v2_threats replay upgrade: plane decode, chunk upgrade, converter.

The upgrade path recomputes the 29 threat planes for stored 146-plane
shards from the encoded planes alone (no FENs); see
chess_anti_engine/encoding/plane_decode.py for the equivalence argument.
Tests check the recomputed block bit-matches a direct v2_threats encode
across all three history encodings, including black-to-move and
en-passant rows, and that validation rejects corrupted inputs.
"""
from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import (
    LC0_HISTORY_LEGACY,
    LC0_HISTORY_ROOT,
    LC0_HISTORY_ROOT_LEGACY_META,
)
from chess_anti_engine.encoding.plane_decode import (
    decode_ep_square,
    decode_step0_bitboards,
)
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import (
    load_shard_arrays,
    save_local_shard_arrays,
)
from chess_anti_engine.replay.threat_upgrade import (
    V1_INPUT_PLANES,
    V2_INPUT_PLANES,
    upgrade_arrays_to_v2_threats,
)
from tests.script_loading import load_script_module

ENCODINGS = (LC0_HISTORY_LEGACY, LC0_HISTORY_ROOT, LC0_HISTORY_ROOT_LEGACY_META)

# 1.e4 a6 2.e5 d5 leaves white an en-passant capture (e5xd6): the EP
# square is live at ply 4. The tail adds black-to-move and capture rows.
_EP_GAME = ("e2e4", "a7a6", "e4e5", "d7d5", "e5d6", "g8f6", "d6c7", "b8c6")


def _game_positions() -> list[chess.Board]:
    boards: list[chess.Board] = []
    b = chess.Board()
    for uci in _EP_GAME:
        boards.append(b.copy(stack=True))
        b.push(chess.Move.from_uci(uci))
    boards.append(b.copy(stack=True))
    return boards


def _random_positions(seed: int, n: int) -> list[chess.Board]:
    rng = random.Random(seed)
    out: list[chess.Board] = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(2, 80)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if not b.is_game_over():
            out.append(b)
    return out


def _encode_rows(boards: list[chess.Board], enc: str, version: str) -> np.ndarray:
    rows = [
        encode_position(b, input_history_encoding=enc, input_extra_features=version)
        for b in boards
    ]
    return np.stack(rows).astype(np.float16)


def _chunk(x: np.ndarray, enc: str) -> dict[str, np.ndarray]:
    n = x.shape[0]
    policy = np.zeros((n, POLICY_SIZE), dtype=np.float16)
    policy[:, :8] = 0.125
    return {
        "x": x,
        "policy_target": policy,
        "wdl_target": np.zeros((n,), dtype=np.int8),
        "priority": np.ones((n,), dtype=np.float32),
        "has_policy": np.ones((n,), dtype=np.uint8),
        "_input_history_encoding": np.asarray(enc),
    }


def test_decode_step0_bitboards_round_trip():
    boards = _game_positions() + _random_positions(11, 12)
    x = _encode_rows(boards, LC0_HISTORY_LEGACY, "v1")
    bbs = decode_step0_bitboards(x)
    for i, b in enumerate(boards):
        us, them = b.turn, not b.turn
        flip = 0 if b.turn == chess.WHITE else 56
        for col, (pt, color) in enumerate(
            [(pt, c) for c in (us, them) for pt in range(1, 7)]
        ):
            expect = 0
            for sq in chess.scan_forward(int(b.pieces_mask(pt, color))):
                expect |= 1 << (sq ^ flip)
            assert int(bbs[i, col]) == expect, (i, col, b.fen())


@pytest.mark.parametrize("enc", ENCODINGS)
def test_decode_ep_square(enc):
    boards = _game_positions()
    x = _encode_rows(boards, enc, "v1")
    for i, b in enumerate(boards):
        got = decode_ep_square(x[i], enc)
        if b.ep_square is None:
            assert got == -1, (i, b.fen())
        else:
            assert got == 40 + chess.square_file(b.ep_square), (i, b.fen())


@pytest.mark.parametrize("enc", ENCODINGS)
def test_upgrade_matches_direct_v2_encode(enc):
    boards = _game_positions() + _random_positions(23, 20)
    x_v1 = _encode_rows(boards, enc, "v1")
    x_v2 = _encode_rows(boards, enc, "v2_threats")
    out, stats = upgrade_arrays_to_v2_threats(_chunk(x_v1, enc))
    assert stats.upgraded_rows == len(boards)
    assert stats.dropout_rows == 0
    assert out["x"].shape[1] == V2_INPUT_PLANES
    np.testing.assert_array_equal(out["x"], x_v2)


def test_upgrade_handles_x_lc0_root_and_unflagged_rows():
    boards = _game_positions()
    x = _encode_rows(boards, LC0_HISTORY_LEGACY, "v1")
    x_root_v1 = _encode_rows(boards, LC0_HISTORY_ROOT, "v1")
    x_root_v2 = _encode_rows(boards, LC0_HISTORY_ROOT, "v2_threats")
    has = np.ones((len(boards),), dtype=np.uint8)
    has[2] = 0
    x_root_v1[2] = 0
    x_root_v2[2] = 0
    chunk = _chunk(x, LC0_HISTORY_LEGACY)
    chunk["x_lc0_root"] = x_root_v1
    chunk["has_x_lc0_root"] = has
    out, _ = upgrade_arrays_to_v2_threats(chunk)
    np.testing.assert_array_equal(out["x_lc0_root"], x_root_v2)


def test_upgrade_noop_on_v2_chunk_and_rejects_unknown_widths():
    boards = _game_positions()[:3]
    x_v2 = _encode_rows(boards, LC0_HISTORY_LEGACY, "v2_threats")
    chunk = _chunk(x_v2, LC0_HISTORY_LEGACY)
    out, stats = upgrade_arrays_to_v2_threats(chunk)
    assert out is chunk and stats.upgraded_rows == 0
    bad = _chunk(x_v2[:, :150], LC0_HISTORY_LEGACY)
    with pytest.raises(ValueError, match="150 input planes"):
        upgrade_arrays_to_v2_threats(bad)


def test_upgrade_handles_empty_chunk():
    x = np.zeros((0, V1_INPUT_PLANES, 8, 8), dtype=np.float16)
    out, stats = upgrade_arrays_to_v2_threats(
        {"x": x, "x_lc0_root": x, "_input_history_encoding": np.asarray(LC0_HISTORY_LEGACY)},
    )
    assert out["x"].shape == (0, V2_INPUT_PLANES, 8, 8)
    assert out["x_lc0_root"].shape == (0, V2_INPUT_PLANES, 8, 8)
    assert stats.upgraded_rows == 0 and stats.dropout_rows == 0


def test_upgrade_validation_rejects_corrupted_planes():
    boards = _game_positions()[:4]
    x = _encode_rows(boards, LC0_HISTORY_LEGACY, "v1")
    x[1, 0] = 0.0  # drop the us-pawns plane: recompute can't match stored v1 block
    with pytest.raises(ValueError, match="disagree with stored"):
        upgrade_arrays_to_v2_threats(_chunk(x, LC0_HISTORY_LEGACY))


def test_upgrade_keeps_dropout_zeroed_rows_zero():
    boards = _game_positions()[:4]
    x = _encode_rows(boards, LC0_HISTORY_LEGACY, "v1")
    x[2, 112:] = 0.0  # encode-time feature dropout zeroes the whole block
    out, stats = upgrade_arrays_to_v2_threats(_chunk(x, LC0_HISTORY_LEGACY))
    assert stats.dropout_rows == 1
    assert not out["x"][2, 112:].any()
    assert out["x"][0, V1_INPUT_PLANES:].any()


def test_disk_buffer_upgrades_v1_shards_on_load(tmp_path):
    boards = _game_positions()
    enc = LC0_HISTORY_ROOT_LEGACY_META
    x_v1 = _encode_rows(boards, enc, "v1")
    x_v2 = _encode_rows(boards, enc, "v2_threats")
    chunk = _chunk(x_v1, enc)
    save_local_shard_arrays(
        tmp_path / "shard_000000.zarr", arrs=chunk,
        meta={"input_history_encoding": enc},
    )
    by_upgrade: dict[bool, np.ndarray] = {}
    for upgrade in (False, True):
        buf = DiskReplayBuffer(
            10_000, shard_dir=tmp_path, rng=np.random.default_rng(0),
            input_planes=V2_INPUT_PLANES, upgrade_v1_planes=upgrade,
            refresh_interval=0, refresh_shards=1,
        )
        batch = buf.sample_batch_arrays(len(boards))
        assert batch["x"].shape[1] == V2_INPUT_PLANES
        by_upgrade[upgrade] = np.asarray(batch["x"])
    assert not by_upgrade[False][:, V1_INPUT_PLANES:].any()  # zero-pad baseline
    threat_rows = {bytes(r) for r in by_upgrade[True][:, V1_INPUT_PLANES:].astype(np.float16)}
    expect_rows = {bytes(r) for r in x_v2[:, V1_INPUT_PLANES:]}
    assert threat_rows  # sampled rows actually carry recomputed planes
    assert threat_rows <= expect_rows


def test_converter_script_in_place_and_idempotent(tmp_path):
    mod = load_script_module("convert_shards_v2_threats.py")
    boards = _game_positions()
    for enc, name in ((LC0_HISTORY_LEGACY, "shard_000000.zarr"),
                      (LC0_HISTORY_ROOT, "shard_000001.zarr")):
        x_v1 = _encode_rows(boards, enc, "v1")
        save_local_shard_arrays(
            tmp_path / name, arrs=_chunk(x_v1, enc),
            meta={"input_history_encoding": enc},
        )
    assert mod.main([str(tmp_path), "--workers", "1"]) == 0
    for enc, name in ((LC0_HISTORY_LEGACY, "shard_000000.zarr"),
                      (LC0_HISTORY_ROOT, "shard_000001.zarr")):
        arrs, meta = load_shard_arrays(tmp_path / name)
        x_v2 = _encode_rows(boards, enc, "v2_threats")
        np.testing.assert_array_equal(np.asarray(arrs["x"]), x_v2)
        assert meta["input_history_encoding"] == enc
    # second run: everything already v2 -> skipped, content unchanged
    assert mod.main([str(tmp_path), "--workers", "1"]) == 0
    arrs2, _ = load_shard_arrays(tmp_path / "shard_000000.zarr")
    np.testing.assert_array_equal(
        np.asarray(arrs2["x"]),
        _encode_rows(boards, LC0_HISTORY_LEGACY, "v2_threats"),
    )


def test_converter_script_rejects_out_name_collisions(tmp_path):
    mod = load_script_module("convert_shards_v2_threats.py")
    boards = _game_positions()[:3]
    x_v1 = _encode_rows(boards, LC0_HISTORY_LEGACY, "v1")
    for src in ("a", "b"):
        save_local_shard_arrays(
            tmp_path / src / "shard_000000.zarr",
            arrs=_chunk(x_v1, LC0_HISTORY_LEGACY),
            meta={"input_history_encoding": LC0_HISTORY_LEGACY},
        )
    with pytest.raises(SystemExit):
        mod.main([
            str(tmp_path / "a"), str(tmp_path / "b"),
            "--out", str(tmp_path / "dst"), "--workers", "1",
        ])
    assert not list((tmp_path / "dst").glob("*.zarr"))  # nothing written


def test_converter_script_out_dir(tmp_path):
    mod = load_script_module("convert_shards_v2_threats.py")
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    boards = _game_positions()[:5]
    x_v1 = _encode_rows(boards, LC0_HISTORY_LEGACY, "v1")
    save_local_shard_arrays(
        src / "shard_000000.zarr", arrs=_chunk(x_v1, LC0_HISTORY_LEGACY),
        meta={"input_history_encoding": LC0_HISTORY_LEGACY},
    )
    assert mod.main([str(src), "--out", str(dst), "--workers", "1"]) == 0
    arrs_src, _ = load_shard_arrays(src / "shard_000000.zarr")
    assert np.asarray(arrs_src["x"]).shape[1] == V1_INPUT_PLANES  # untouched
    arrs_dst, _ = load_shard_arrays(dst / "shard_000000.zarr")
    np.testing.assert_array_equal(
        np.asarray(arrs_dst["x"]),
        _encode_rows(boards, LC0_HISTORY_LEGACY, "v2_threats"),
    )
