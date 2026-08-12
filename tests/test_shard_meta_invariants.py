"""Internal-consistency checks on the shard counter channel.

⚑ THE RISK HERE RUNS THE OTHER WAY. Every other test in this repo asks "does
the guard catch the bad thing". For this guard the expensive failure is the
FALSE POSITIVE: a predicate that looks obvious, is wrong, and rejects real
selfplay -- ingest to zero, which is this project's recurring outage class. So
the false-positive controls below are the load-bearing half, and two of them
pin predicates that were considered and deliberately NOT written:

    wins + draws + losses == games      # wrong: w/d/l exclude selfplay + seeds
    games <= positions                  # wrong: diff-focus can drop a game's rows
"""
from __future__ import annotations

from chess_anti_engine.replay.shard import shard_meta_violations


def test_a_consistent_shard_has_no_violations() -> None:
    meta = {
        "games": 10, "wins": 3, "draws": 2, "losses": 1, "positions": 500,
        "selfplay_games": 4, "curriculum_games": 6,
        "adjudicated_games": 2, "checkmate_games": 1, "stalemate_games": 0,
    }
    assert shard_meta_violations(meta, positions=500) == []


def test_selfplay_games_make_the_wdl_sum_strictly_less_than_games() -> None:
    """⚑ FALSE-POSITIVE CONTROL for the predicate most likely to be "fixed".

    `selfplay/finalize.py` increments w/d/l only when `not is_sp`, so a shard
    that is mostly selfplay has w+d+l far below `games`. An `==` here would
    reject nearly every real shard.
    """
    meta = {"games": 100, "wins": 2, "draws": 1, "losses": 1, "selfplay_games": 96}
    assert shard_meta_violations(meta, positions=1) == []


def test_blind_spot_seed_games_are_excluded_from_wdl() -> None:
    """Also a false-positive control: `fenlist` games are curriculum games that
    are deliberately kept out of the PID sample, so even a pure-curriculum
    shard can have w+d+l < curriculum_games."""
    meta = {"games": 50, "curriculum_games": 50, "wins": 10, "draws": 5, "losses": 5}
    assert shard_meta_violations(meta, positions=1) == []


def test_more_games_than_positions_is_allowed() -> None:
    """⚑ FALSE-POSITIVE CONTROL. `games <= positions` looks self-evidently true
    and is not: diff-focus filtering can drop every position of a game."""
    assert shard_meta_violations({"games": 40}, positions=3) == []


def test_wdl_sum_above_games_is_caught() -> None:
    meta = {"games": 5, "wins": 4, "draws": 3, "losses": 2}
    v = shard_meta_violations(meta, positions=1)
    assert len(v) == 1
    assert "wins+draws+losses=9 exceeds games=5" in v[0]


def test_declared_positions_must_match_the_rows_actually_present() -> None:
    """The one predicate with a genuinely trusted right-hand side: the server
    counted the rows itself. This is the flipped-digit case."""
    v = shard_meta_violations({"games": 1, "positions": 5000}, positions=500)
    assert any("disagrees with the 500 rows actually present" in s for s in v)
    # And the matching case is silent.
    assert shard_meta_violations({"games": 1, "positions": 500}, positions=500) == []


def test_negative_counters_are_caught() -> None:
    v = shard_meta_violations({"games": -1, "wins": 0, "draws": 0, "losses": 0}, positions=1)
    assert any("games=-1 is negative" in s for s in v)


def test_subset_counters_cannot_exceed_their_parent() -> None:
    v = shard_meta_violations(
        {"games": 10, "selfplay_games": 8, "curriculum_games": 7}, positions=1,
    )
    assert any("exceeds games=10" in s for s in v)

    v = shard_meta_violations(
        {"games": 10, "curriculum_games": 4, "curriculum_draw_games": 9}, positions=1,
    )
    assert any("exceeds curriculum_games=4" in s for s in v)


def test_absent_and_unparseable_counters_are_ignored_not_rejected() -> None:
    """A shard from an older worker simply lacks these keys, and a
    non-numeric value is someone else's bug -- neither is evidence of
    corruption, and rejecting on absence would break rolling upgrades."""
    assert shard_meta_violations({}, positions=0) == []
    assert shard_meta_violations({"games": None, "wins": "x"}, positions=0) == []


def test_a_real_worker_built_meta_passes() -> None:
    """End-to-end against the production builder rather than a hand-written
    dict, so a future field whose semantics break an invariant fails HERE.

    ⚑ A hand-written fixture can only encode what I already believe about the
    counters; this encodes what `worker_buffer` actually emits.
    """
    from dataclasses import asdict

    from chess_anti_engine.replay.shard import ShardMeta

    meta = asdict(ShardMeta(
        username="u", games=8, positions=120, model_sha256="abc1234567", model_step=0,
        wins=2, draws=1, losses=1,
        selfplay_games=4, curriculum_games=4,
        total_game_plies=900, adjudicated_games=1, total_draw_games=1,
    ))
    assert shard_meta_violations(meta, positions=120) == []


# ---------------------------------------------------------------------------
# Wiring. The pure function above is worthless if the route never calls it.
# ---------------------------------------------------------------------------


def _corrupt_tar(tmp_path, **meta_over) -> bytes:
    """A shard whose `.zattrs` counters have been edited after the arrays were
    written -- i.e. exactly the flipped-digit shape, not a malformed store."""
    import json

    from chess_anti_engine.replay.shard import (
        ShardMeta,
        pack_shard_for_upload,
        samples_to_arrays,
        save_local_shard_arrays,
    )

    from .test_server_upload_security import _sample

    tmp_path.mkdir(parents=True, exist_ok=True)
    zp = tmp_path / "valid.zarr"
    save_local_shard_arrays(
        zp,
        arrs=samples_to_arrays([_sample(i) for i in range(2)]),
        meta=ShardMeta(
            username="u", games=1, positions=2,
            model_sha256="abc1234567", model_step=0,
        ),
    )
    attrs_path = zp / ".zattrs"
    attrs = json.loads(attrs_path.read_text())
    attrs.update(meta_over)
    attrs_path.write_text(json.dumps(attrs, indent=4, sort_keys=True))
    _, buf = pack_shard_for_upload(zp)
    return buf.getvalue()


def test_route_rejects_a_shard_whose_counters_contradict_the_arrays(tmp_path) -> None:
    """⚑ Terminal `rejected`, not 422. An arithmetic inconsistency is a
    permanent property of these bytes, so a retry resends the same bad shard
    forever -- the opposite call from the digest mismatch, which is transient.
    """
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    payload = _corrupt_tar(tmp_path / "bad", positions=99999)
    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", payload, "application/x-tar")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("stored") is False
    assert body.get("rejected") is True, body
    assert "inconsistent shard metadata" in str(body.get("reason")), body
    # Nothing reached the pending dir.
    assert not [p for p in server_root.rglob("*.zarr") if "_pending" in str(p)]


def test_route_accepts_a_shard_whose_counters_are_consistent(tmp_path) -> None:
    """⚑ NEGATIVE CONTROL, and the one that matters. A validator wired to
    reject everything would pass the test above while taking ingest to zero.
    """
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    # Untouched `.zattrs` straight from the production builder.
    payload = _corrupt_tar(tmp_path / "good")
    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", payload, "application/x-tar")},
    )
    assert r.status_code == 200, r.text
    assert r.json().get("stored") is True, r.json()


# ---------------------------------------------------------------------------
# [4] retention: the quarantine sink had no ceiling
# ---------------------------------------------------------------------------


def _mk(root, name: str, *, size: int, mtime: float, sidecar: bool = True):
    p = root / name
    p.write_bytes(b"x" * size)
    if sidecar:
        s = p.with_suffix(p.suffix + ".reason.txt")
        s.write_text("ValueError: bad", encoding="utf-8")
        import os
        os.utime(s, (mtime, mtime))
    import os
    os.utime(p, (mtime, mtime))
    return p


def test_prune_evicts_oldest_first_under_a_byte_budget(tmp_path) -> None:
    """A quarantined shard diagnoses a defect happening NOW, so the newest
    entries are the ones worth keeping."""
    from chess_anti_engine.server.app import prune_retained_dir

    q = tmp_path / "invalid"
    q.mkdir()
    _mk(q, "old.tar", size=1000, mtime=1_000_000)
    _mk(q, "mid.tar", size=1000, mtime=2_000_000)
    _mk(q, "new.tar", size=1000, mtime=3_000_000)

    freed_entries, freed_bytes = prune_retained_dir(q, max_bytes=2500, max_entries=0)
    assert freed_entries == 1
    assert freed_bytes >= 1000
    names = {p.name for p in q.iterdir()}
    assert "old.tar" not in names
    assert {"mid.tar", "new.tar"} <= names
    # ⚑ The sidecar goes with its shard. Orphaning it would leave a reason
    # file explaining a shard that is no longer there.
    assert "old.tar.reason.txt" not in names
    assert "new.tar.reason.txt" in names


def test_prune_enforces_the_entry_count_independently(tmp_path) -> None:
    """Either limit can bind: a flood of tiny invalid uploads burns inodes
    while declaring almost no bytes."""
    from chess_anti_engine.server.app import prune_retained_dir

    q = tmp_path / "invalid"
    q.mkdir()
    for i in range(10):
        _mk(q, f"s{i}.tar", size=1, mtime=1_000_000 + i)

    prune_retained_dir(q, max_bytes=0, max_entries=4)
    remaining = sorted(p.name for p in q.iterdir() if not p.name.endswith(".reason.txt"))
    assert len(remaining) == 4, remaining
    assert remaining == ["s6.tar", "s7.tar", "s8.tar", "s9.tar"]


def test_prune_keeps_everything_when_under_budget(tmp_path) -> None:
    """⚑ NEGATIVE CONTROL. A sweep that deleted unconditionally would pass both
    tests above while destroying the diagnostic value the directory exists for.
    """
    from chess_anti_engine.server.app import prune_retained_dir

    q = tmp_path / "invalid"
    q.mkdir()
    for i in range(3):
        _mk(q, f"s{i}.tar", size=10, mtime=1_000_000 + i)

    freed, _ = prune_retained_dir(q, max_bytes=10_000, max_entries=100)
    assert freed == 0
    assert len([p for p in q.iterdir() if not p.name.endswith(".reason.txt")]) == 3


def test_prune_on_a_missing_dir_is_a_no_op(tmp_path) -> None:
    from chess_anti_engine.server.app import prune_retained_dir

    assert prune_retained_dir(tmp_path / "nope", max_bytes=1, max_entries=1) == (0, 0)


def _arena_payload(**over):
    p = {
        "games": 4, "a_win": 2, "a_draw": 1, "a_loss": 1,
        "a_sha256": "a" * 64, "b_sha256": "b" * 64,
        "generated_at_unix": 1_700_000_000,
    }
    p.update(over)
    return p


def test_arena_route_rejects_an_oversized_body(tmp_path) -> None:
    """Each distinct body persists a NEW file named after its own sha, and
    nothing capped the size -- `max_upload_mb` is the SHARD route only."""
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, arena_max_body_bytes=2048)

    r = client.post(
        "/v1/upload_arena_result",
        auth=("u", "p"),
        json=_arena_payload(padding="z" * 10_000),
    )
    assert r.status_code == 413, r.text
    assert not list((server_root / "arena_inbox").rglob("*.json"))


def test_arena_route_still_accepts_a_normal_result(tmp_path) -> None:
    """⚑ NEGATIVE CONTROL. A cap set below a real payload would silently
    destroy every arena result the fleet produces."""
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    r = client.post("/v1/upload_arena_result", auth=("u", "p"), json=_arena_payload())
    assert r.status_code == 200, r.text
    assert r.json()["stored"] is True
    assert list((server_root / "arena_inbox").rglob("*.json"))


def test_arena_inbox_is_bounded_per_user(tmp_path) -> None:
    """The growth mode is many small UNIQUE bodies -- dedupe by sha does not
    help, because every distinct body is a new filename."""
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, arena_user_max_entries=5, arena_user_max_bytes=0)

    for i in range(12):
        r = client.post(
            "/v1/upload_arena_result",
            auth=("u", "p"),
            json=_arena_payload(generated_at_unix=1_700_000_000 + i),
        )
        assert r.status_code == 200, r.text

    kept = list((server_root / "arena_inbox").rglob("*.json"))
    assert len(kept) <= 5, f"arena inbox grew unbounded: {len(kept)} files"
