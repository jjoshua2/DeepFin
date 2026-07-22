"""Seed-provenance: per-row seed_id / seed_family_id / opening_source_code.

Backward-compatible optional shard fields plus the manifest resolver that
finalize.py uses to stamp them.
"""
import json

import numpy as np

from chess_anti_engine.replay import ReplaySample
from chess_anti_engine.replay.shard import load_npz, save_npz
from chess_anti_engine.selfplay.seed_manifest import (
    _load_by_key,
    content_seed_id,
    opening_source_code,
    position_key,
    resolve_seed_ids,
)


def _sample(policy_size: int = 4672) -> ReplaySample:
    x = np.zeros((146, 8, 8), dtype=np.float32)
    pol = np.zeros((policy_size,), dtype=np.float32)
    pol[0] = 1.0
    return ReplaySample(x=x, policy_target=pol, wdl_target=1)


def test_seed_fields_roundtrip(tmp_path):
    s = _sample()
    s.seed_id = 7
    s.seed_family_id = 3
    s.opening_source_code = 2

    out, _meta = load_npz(save_npz(tmp_path / "shard.npz", samples=[s], meta={"positions": 1}))
    r = out[0]
    assert r.seed_id == 7
    assert r.seed_family_id == 3
    assert r.opening_source_code == 2


def test_seed_fields_absent_roundtrip_to_none(tmp_path):
    # Old-shard safety: None in -> None out (has_ flags stay false).
    s = _sample()
    assert s.seed_id is None
    assert s.seed_family_id is None
    assert s.opening_source_code is None

    out, _meta = load_npz(save_npz(tmp_path / "shard.npz", samples=[s], meta={"positions": 1}))
    r = out[0]
    assert r.seed_id is None
    assert r.seed_family_id is None
    assert r.opening_source_code is None


def test_opening_source_code():
    assert opening_source_code("fenlist") == 2
    assert opening_source_code("fenlist_backed") == 2
    assert opening_source_code("fenlist_sf_refute") == 3
    assert opening_source_code("start") == 0
    assert opening_source_code("book_xyz") == 1
    assert opening_source_code("random") == 5
    assert opening_source_code("weird") == 255


def test_position_key_ignores_clocks():
    # Two FENs differing only in halfmove/fullmove clocks share a key.
    a = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    b = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 5 12"
    assert position_key(a) == position_key(b)


def test_resolve_seed_ids_manifest_wins_for_seed_rows(tmp_path):
    fen_match = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fen_miss = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    key = position_key(fen_match)

    list_path = tmp_path / "seeds.txt"
    manifest_path = tmp_path / "seeds.txt.manifest.json"
    manifest = {
        "version": 1,
        "list_path": str(list_path),
        "n_seeds": 1,
        "seeds": [{"seed_id": 4, "seed_family_id": 2, "fen": fen_match, "position_key": key}],
        "by_key": {key: [4, 2]},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # lru_cache is keyed on (path, mtime); clear so a prior test's entry can't leak.
    _load_by_key.cache_clear()

    # Seed-origin rows (source_code 2/3) get curated ids when the manifest hits.
    assert resolve_seed_ids(fen_match, str(list_path), source_code=2) == (4, 2)
    # Differing only in clocks still resolves (same position key).
    fen_clock = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 40 99"
    assert resolve_seed_ids(fen_clock, str(list_path), source_code=2) == (4, 2)
    # Seed row NOT in the manifest -> content-hash fallback (never -1 for a seed).
    hm = content_seed_id(fen_miss)
    assert resolve_seed_ids(fen_miss, str(list_path), source_code=3) == (hm, hm)
    # Non-seed rows never get an id, even when the manifest would hit.
    assert resolve_seed_ids(fen_match, str(list_path), source_code=1) == (-1, -1)
    assert resolve_seed_ids(fen_match, str(list_path), source_code=255) == (-1, -1)


def test_manifest_out_of_range_ids_dropped(tmp_path):
    # A hand-built manifest with an id outside int32 must not reach the shard
    # writer (uint/int32 arrays) — the entry is dropped and the row falls back
    # to the content hash.
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    key = position_key(fen)
    list_path = tmp_path / "seeds.txt"
    (tmp_path / "seeds.txt.manifest.json").write_text(
        json.dumps({"by_key": {key: [2**31, -1]}}), encoding="utf-8"
    )
    _load_by_key.cache_clear()
    h = content_seed_id(fen)
    assert resolve_seed_ids(fen, str(list_path), source_code=2) == (h, h)


def test_manifest_keys_terminal_position(tmp_path):
    # Manifest identity = TERMINAL position of a `fen | moves` line, rendered
    # by Board.fen() — what finalize sees as start_fen (review finding #1/#2).
    import chess

    from scripts.build_seed_manifest import build_manifest

    base = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    line = base + " | e2e4 e7e5"
    lp = tmp_path / "seeds.txt"
    lp.write_text(line + "\n", encoding="utf-8")

    m = build_manifest(str(lp))
    board = chess.Board(base)
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    terminal = board.fen()

    assert m["by_key"] == {position_key(terminal): [0, 0]}
    assert m["by_content_id"] == {str(content_seed_id(terminal)): terminal}
    # Base and intermediate positions land in the derived map -> terminal.
    derived = m["by_derived_content_id"]
    assert isinstance(derived, dict)
    assert derived[str(content_seed_id(base))] == terminal
    assert str(content_seed_id(terminal)) in derived


def test_content_hash_fallback_distributed(tmp_path):
    # Distributed case: worker's list path is an ephemeral sha-named copy with
    # no manifest beside it, so seed rows resolve via the content hash.
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    h = content_seed_id(fen)
    assert 0 <= h <= 0x7FFFFFFF  # fits non-negative int32 (shard dtype)
    assert resolve_seed_ids(fen, str(tmp_path / "nope.txt"), source_code=2) == (h, h)
    assert resolve_seed_ids(fen, None, source_code=2) == (h, h)
    # Clock-invariant, and still guarded on seed source.
    fen_clock = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 40 99"
    assert content_seed_id(fen_clock) == h
    assert resolve_seed_ids(fen, None, source_code=0) == (-1, -1)
