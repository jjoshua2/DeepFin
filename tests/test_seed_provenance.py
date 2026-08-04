"""Seed-provenance: per-row seed_id / seed_family_id / opening_source_code.

Backward-compatible optional shard fields plus the manifest resolver that
finalize.py uses to stamp them.
"""
import json

import numpy as np
import pytest

from chess_anti_engine.replay import ReplaySample
from chess_anti_engine.replay.shard import load_npz, save_npz
from chess_anti_engine.selfplay.seed_manifest import (
    _load_by_key,
    content_seed_id,
    opening_source_code,
    position_key,
    resolve_seed_ids,
)


def _curated(fen, list_path):
    """`curated_seed_ids` does not exist on origin/main.

    Imported through a helper so the module still COLLECTS there and the red
    count stays a per-test measurement instead of one collection error
    standing in for the whole file.
    """
    from chess_anti_engine.selfplay.seed_manifest import curated_seed_ids

    return curated_seed_ids(fen, list_path)


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


def test_resolve_seed_ids_is_the_content_hash_even_when_a_manifest_hits(tmp_path):
    """⚑ REVERSED (audit A24). This case used to assert that a present manifest
    OVERRODE the content hash, which is precisely the defect: the same seed FEN
    got id 4 in a local run and its content hash in a distributed one, both
    valid non-negative int32s, silently splitting one seed in two in any
    seed_id-grouped analysis.

    The curated pair is not lost — it moved to ``curated_seed_ids`` as an
    alias. What changed is that it no longer decides the row's identity.
    """
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

    h = content_seed_id(fen_match)
    assert resolve_seed_ids(fen_match, source_code=2) == (h, h)
    # ...and the curated pair is still reachable, just not as the identity.
    assert _curated(fen_match, str(list_path)) == (4, 2)
    # Differing only in clocks still resolves to the same position identity.
    fen_clock = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 40 99"
    assert resolve_seed_ids(fen_clock, source_code=2) == (h, h)
    # A seed row not in the manifest was already the content hash; unchanged.
    hm = content_seed_id(fen_miss)
    assert resolve_seed_ids(fen_miss, source_code=3) == (hm, hm)
    # Non-seed rows never get an id.
    assert resolve_seed_ids(fen_match, source_code=1) == (-1, -1)
    assert resolve_seed_ids(fen_match, source_code=255) == (-1, -1)


def test_the_same_fen_is_JOINABLE_regardless_of_a_manifest_on_disk(tmp_path):
    """⚑ THE defect, stated as the property that must hold.

    On origin/main this FEN got `(4, 2)` with the manifest present and its
    content hash without -- and main's own suite asserted the `(4, 2)` as
    intended behaviour, which is how the split survived.

    The identity must now be invariant to the filesystem. Asserted by calling
    across a manifest being created and then deleted: the function no longer
    takes a path, so this is really a check that no hidden filesystem or
    global state leaked back in.
    """
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    key = position_key(fen)
    list_path = tmp_path / "seeds.txt"
    manifest = tmp_path / "seeds.txt.manifest.json"

    without = resolve_seed_ids(fen, source_code=2)

    manifest.write_text(json.dumps({"by_key": {key: [4, 2]}}), encoding="utf-8")
    _load_by_key.cache_clear()
    with_manifest = resolve_seed_ids(fen, source_code=2)

    manifest.unlink()
    _load_by_key.cache_clear()
    after = resolve_seed_ids(fen, source_code=2)

    assert without == with_manifest == after, (
        "the same seed FEN must get the same id however the run is set up, or "
        "every seed_id-grouped analysis silently splits it into two seeds"
    )
    # And the curated alias is genuinely a DIFFERENT number, so the test above
    # is not passing because the two schemes happen to agree here.
    manifest.write_text(json.dumps({"by_key": {key: [4, 2]}}), encoding="utf-8")
    _load_by_key.cache_clear()
    assert _curated(fen, str(list_path)) == (4, 2)
    assert with_manifest != (4, 2)


def test_path_independence_is_enforced_by_the_SIGNATURE(tmp_path):
    """The strongest form of the fix: the bug is unrepresentable.

    `resolve_seed_ids` cannot consult a path because it is not given one. That
    is a stronger guarantee than any assertion about its return value, and it
    is why the joinability test above can be short. Checked as an API contract
    -- if someone reintroduces the parameter, the identity can become
    mode-dependent again and this fails.
    """
    import inspect

    del tmp_path
    params = inspect.signature(resolve_seed_ids).parameters
    assert "list_path" not in params
    assert set(params) == {"fen", "source_code"}


def test_manifest_out_of_range_ids_are_dropped_AND_logged(tmp_path, caplog):
    """The drop was already correct; the silence was not.

    An out-of-range id must not reach the shard writer (int32 arrays), but a
    hand-built manifest with one bad row used to yield a silent mix of curated
    and absent aliases with nothing naming the offending entry.
    """
    import logging

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    key = position_key(fen)
    list_path = tmp_path / "seeds.txt"
    (tmp_path / "seeds.txt.manifest.json").write_text(
        json.dumps({"by_key": {key: [2**31, -1]}}), encoding="utf-8"
    )
    _load_by_key.cache_clear()

    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.selfplay.seed_manifest"):
        assert _curated(fen, str(list_path)) is None

    assert "dropped 1 entry" in caplog.text
    assert key in caplog.text, "the log must name the offending entry"
    # The row identity is unaffected either way -- that is the point of making
    # the hash primary.
    h = content_seed_id(fen)
    assert resolve_seed_ids(fen, source_code=2) == (h, h)


def test_a_corrupt_manifest_warns_instead_of_silently_switching_schemes(
    tmp_path, caplog,
):
    """RED on origin/main: a bare `except Exception` swallowed this entirely.

    On main a truncated manifest silently changed which id scheme every seed
    row got, with no warning anywhere. Now the identity does not depend on the
    manifest at all, and the unreadable file is announced.
    """
    import logging

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    list_path = tmp_path / "seeds.txt"
    (tmp_path / "seeds.txt.manifest.json").write_text(
        '{"by_key": {"rnbq', encoding="utf-8",
    )
    _load_by_key.cache_clear()

    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.selfplay.seed_manifest"):
        assert _curated(fen, str(list_path)) is None

    assert "unreadable" in caplog.text
    h = content_seed_id(fen)
    assert resolve_seed_ids(fen, source_code=2) == (h, h)


def test_the_narrowed_except_does_not_swallow_a_programming_error(tmp_path):
    """⚑ The point of narrowing: only the failures a malformed FILE produces
    are caught. A bug inside the loader must still propagate rather than be
    reported as "corrupt manifest", which is how a real defect hides for months.
    """
    import chess_anti_engine.selfplay.seed_manifest as sm

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    list_path = tmp_path / "seeds.txt"
    (tmp_path / "seeds.txt.manifest.json").write_text(
        json.dumps({"by_key": {}}), encoding="utf-8",
    )
    _load_by_key.cache_clear()

    original = sm._load_by_key
    try:
        def _boom(*_a, **_k):
            raise RuntimeError("bug inside the loader")
        sm._load_by_key = _boom
        with pytest.raises(RuntimeError, match="bug inside the loader"):
            sm.curated_seed_ids(fen, str(list_path))
    finally:
        sm._load_by_key = original


def test_curated_ids_are_absent_when_there_is_no_manifest(tmp_path):
    """NEGATIVE CONTROL: the alias is None in the production (distributed) mode,
    which is exactly why it must not be the identity."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert _curated(fen, str(tmp_path / "nope.txt")) is None
    assert _curated(fen, None) is None


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


def test_content_hash_id_fits_the_shard_dtype_and_ignores_clocks():
    # Was `test_content_hash_fallback_distributed`: with the manifest override
    # gone the hash is no longer a "fallback", it is the only scheme. What is
    # still worth pinning is the shard dtype bound and clock-invariance.
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    h = content_seed_id(fen)
    assert 0 <= h <= 0x7FFFFFFF  # fits non-negative int32 (shard dtype)
    assert resolve_seed_ids(fen, source_code=2) == (h, h)
    # Clock-invariant, and still guarded on seed source.
    fen_clock = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 40 99"
    assert content_seed_id(fen_clock) == h
    assert resolve_seed_ids(fen, source_code=0) == (-1, -1)
