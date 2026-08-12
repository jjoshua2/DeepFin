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


def test_an_adjudicated_draw_is_not_a_violation() -> None:
    """⚑⚑ REGRESSION. The first version of this validator summed the
    adjudication and draw children against their parent, and an independent
    review caught that they OVERLAP rather than partition.

    `finalize.py:510-558` has `if was_adjudicated:` and `if result ==
    "1/2-1/2":` as INDEPENDENT blocks, so ONE adjudicated draw increments
    `selfplay_games`, `selfplay_adjudicated_games` AND `selfplay_draw_games`.
    The sum then read 1+1=2 > 1 and TERMINALLY rejected the shard. Adjudicated
    draws are ordinary, so that was mass rejection of real selfplay.

    A sum-of-children bound silently assumes a partition. Only
    selfplay/curriculum actually is one.
    """
    one_adjudicated_draw = {
        "games": 1, "positions": 10,
        "selfplay_games": 1,
        "selfplay_adjudicated_games": 1,
        "selfplay_draw_games": 1,
        "adjudicated_games": 1,
        "total_draw_games": 1,
    }
    assert shard_meta_violations(one_adjudicated_draw, positions=10) == []

    # Same shape on the curriculum side.
    assert shard_meta_violations({
        "games": 1, "curriculum_games": 1,
        "curriculum_adjudicated_games": 1, "curriculum_draw_games": 1,
        "adjudicated_games": 1, "total_draw_games": 1,
    }, positions=1) == []

    # A whole batch of them -- every game adjudicated AND drawn.
    assert shard_meta_violations({
        "games": 64, "selfplay_games": 64,
        "selfplay_adjudicated_games": 64, "selfplay_draw_games": 64,
        "adjudicated_games": 64, "total_draw_games": 64,
    }, positions=1) == []

    # But a child genuinely exceeding its parent is still caught.
    v = shard_meta_violations(
        {"games": 5, "selfplay_games": 2, "selfplay_draw_games": 3}, positions=1,
    )
    assert any("selfplay_draw_games=3 exceeds selfplay_games=2" in s for s in v)


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
    # ⚑ Terminal on a 200, not a 413. The worker keeps ANY non-accepted
    # response for retry and `break`s, and drains in sorted (timestamp) order,
    # so a permanently-rejected file at the head of the queue blocks every
    # later result forever. `rejected` is the protocol's terminal channel.
    assert r.status_code == 200, r.text
    assert r.json().get("rejected") is True, r.json()
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


def test_sidecar_bytes_count_toward_the_budget(tmp_path) -> None:
    """⚑ REGRESSION (review finding). The sidecar's size was accounted only
    AFTER its shard had been picked for eviction, so the loop could conclude
    the directory was under budget while retained `.reason.txt` files still put
    it over. The reason text is an exception message -- attacker-influenced and
    unbounded -- so the ceiling could be walked straight past.
    """
    from chess_anti_engine.server.app import prune_retained_dir

    q = tmp_path / "invalid"
    q.mkdir()
    # 100 bytes of shard each, but 5000 bytes of "reason" each.
    for i in range(3):
        p = q / f"s{i}.tar"
        p.write_bytes(b"x" * 100)
        p.with_suffix(p.suffix + ".reason.txt").write_text("E" * 5000)
        import os
        os.utime(p, (1_000_000 + i, 1_000_000 + i))

    # Shards alone are 300 bytes -- under budget. With sidecars it is 15300.
    prune_retained_dir(q, max_bytes=6000, max_entries=0)

    actual = sum(f.stat().st_size for f in q.rglob("*") if f.is_file())
    assert actual <= 6000, (
        f"retention left {actual} bytes on disk against a 6000 byte budget -- "
        "sidecars are not being counted"
    )


def test_arena_retention_cannot_be_bypassed_by_rotating_trial_ids(tmp_path) -> None:
    """⚑⚑ REGRESSION (review finding, P1). A quota rooted under a
    CALLER-SELECTED path is not a quota.

    `_normalize_trial_id` only syntax-checks the id and `_check_worker_compat`
    admits a trial with no published manifest (legitimate before the first
    publish), so a per-trial budget hands the client a fresh full allowance for
    every well-formed id it invents. The budget is the USER'S, across trials.
    """
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, arena_user_max_entries=4, arena_user_max_bytes=0)

    # Same account, a different invented trial id every time.
    for i in range(15):
        r = client.post(
            f"/v1/trials/rotate_{i}/upload_arena_result",
            auth=("u", "p"),
            json=_arena_payload(generated_at_unix=1_700_000_000 + i),
        )
        assert r.status_code == 200, r.text

    kept = list(server_root.rglob("arena_inbox/*/*.json"))
    assert len(kept) <= 4, (
        f"{len(kept)} arena results retained against a 4-entry per-user budget -- "
        "rotating trial ids bought a fresh allowance each time"
    )


def test_quarantine_sweep_is_wired_to_the_upload_route(tmp_path) -> None:
    """⚑⚑ REGRESSION (review finding, P2). Replacing the whole
    `prune_retained_dir(qdir, ...)` call in the upload route with `pass` left
    every one of the original tests GREEN: they all called the helper directly,
    and only the ARENA path had a route-level test.

    Quarantine growth is finding [4]'s headline sink, so nothing would have
    noticed if a later refactor dropped the call. That is this codebase's
    signature defect -- a value accepted and then silently ignored -- sitting
    inside the fix for it.
    """
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, quarantine_max_entries=3, quarantine_max_bytes=0)

    # Each POST is an unparseable "shard", so each lands in quarantine/invalid.
    for i in range(10):
        r = client.post(
            "/v1/upload_shard",
            auth=("u", "p"),
            files={"file": (f"shard{i}.zarr.tar", b"not a tar at all %d" % i,
                            "application/x-tar")},
        )
        assert r.status_code == 200, r.text
        assert r.json().get("rejected") is True

    kept = [
        p for p in (server_root / "quarantine" / "invalid").iterdir()
        if not p.name.endswith(".reason.txt")
    ]
    assert len(kept) <= 3, (
        f"quarantine holds {len(kept)} entries against a 3-entry budget -- "
        "the sweep is not wired to the upload route"
    )


def test_client_report_sink_is_bounded(tmp_path) -> None:
    """⚑ The cheapest sink to spam: a small authenticated JSON POST, no tar
    upload and no size cap. Bounding `quarantine/invalid` while leaving this
    unbounded would close the headline sink and not the easy one."""
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, quarantine_max_entries=3, quarantine_max_bytes=0)

    for i in range(12):
        client.post(
            "/v1/report_bad_shard",
            auth=("u", "p"),
            json={"shard_name": f"s{i}.zarr", "reason": "unreadable"},
        )

    qdir = server_root / "quarantine" / "client_reports"
    if qdir.is_dir():
        kept = [p for p in qdir.iterdir() if not p.name.endswith(".reason.txt")]
        assert len(kept) <= 3, f"client_reports grew unbounded: {len(kept)}"


def test_a_rejected_arena_result_does_not_block_the_queue(tmp_path) -> None:
    """⚑ REGRESSION (review). The arena drain had NO terminal channel: any
    non-accepted response was kept and `break`ed on, and files drain in sorted
    (timestamp) order -- so one permanently-rejected result at the head of the
    queue blocks every later one, forever.

    Invisible until the body-size cap introduced the first permanent rejection
    on this route.
    """
    import json as _json
    import logging
    import types

    from chess_anti_engine.worker import WorkerSession

    pending = tmp_path / "arena_pending"
    pending.mkdir(parents=True)
    rejected = tmp_path / "arena_rejected"
    rejected.mkdir(parents=True)
    # Oldest file is the poison one; a newer good one sits behind it.
    (pending / "1000_bad.json").write_text(_json.dumps({"games": 1, "padding": "z" * 50}))
    (pending / "2000_good.json").write_text(_json.dumps({"games": 1}))

    posted: list[str] = []

    class _Resp:
        status_code = 200

        def __init__(self, rejected_: bool) -> None:
            self._r = rejected_

        def json(self):
            return ({"stored": False, "rejected": True, "terminal": True, "reason": "too big"}
                    if self._r else {"stored": True})

    class _Requests:
        @staticmethod
        def post(_url, **kw):
            body = kw.get("json") or {}
            big = "padding" in body
            posted.append("bad" if big else "good")
            return _Resp(big)

    w = types.SimpleNamespace(
        arena_pending_dir=pending, arena_rejected_dir=rejected,
        leased_trial_id="", fixed_trial_id="", trial_api_prefix="/v1",
        log=logging.getLogger("test_arena_wedge"), _arena_rejected_count=0,
        _auth=("u", "p"), _requests=_Requests,
    )
    w._server_url_for = lambda p: "http://server" + p

    WorkerSession._upload_pending_arena_results(w)  # pyright: ignore[reportArgumentType]

    # ⚑ The load-bearing assertion: the GOOD result behind it was reached.
    assert posted == ["bad", "good"], (
        f"queue was head-of-line blocked by the rejected result: {posted}"
    )
    assert not (pending / "1000_bad.json").exists()
    assert (rejected / "1000_bad.json").exists()
    assert not (pending / "2000_good.json").exists()
