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

import logging
import os

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
    from chess_anti_engine.server.app import prune_retained_dirs

    q = tmp_path / "invalid"
    q.mkdir()
    _mk(q, "old.tar", size=1000, mtime=1_000_000)
    _mk(q, "mid.tar", size=1000, mtime=2_000_000)
    _mk(q, "new.tar", size=1000, mtime=3_000_000)

    freed_entries, freed_bytes = prune_retained_dirs([q], max_bytes=2500, max_entries=0)
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
    from chess_anti_engine.server.app import prune_retained_dirs

    q = tmp_path / "invalid"
    q.mkdir()
    for i in range(10):
        _mk(q, f"s{i}.tar", size=1, mtime=1_000_000 + i)

    prune_retained_dirs([q], max_bytes=0, max_entries=4)
    remaining = sorted(p.name for p in q.iterdir() if not p.name.endswith(".reason.txt"))
    assert len(remaining) == 4, remaining
    assert remaining == ["s6.tar", "s7.tar", "s8.tar", "s9.tar"]


def test_prune_keeps_everything_when_under_budget(tmp_path) -> None:
    """⚑ NEGATIVE CONTROL. A sweep that deleted unconditionally would pass both
    tests above while destroying the diagnostic value the directory exists for.
    """
    from chess_anti_engine.server.app import prune_retained_dirs

    q = tmp_path / "invalid"
    q.mkdir()
    for i in range(3):
        _mk(q, f"s{i}.tar", size=10, mtime=1_000_000 + i)

    freed, _ = prune_retained_dirs([q], max_bytes=10_000, max_entries=100)
    assert freed == 0
    assert len([p for p in q.iterdir() if not p.name.endswith(".reason.txt")]) == 3


def test_prune_on_a_missing_dir_is_a_no_op(tmp_path) -> None:
    from chess_anti_engine.server.app import prune_retained_dirs

    assert prune_retained_dirs([tmp_path / "nope"], max_bytes=1, max_entries=1) == (0, 0)


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
    client = _build_client(server_root, arena_max_entries=5, arena_max_bytes=0)

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
    from chess_anti_engine.server.app import prune_retained_dirs

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
    prune_retained_dirs([q], max_bytes=6000, max_entries=0)

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
    client = _build_client(server_root, arena_max_entries=4, arena_max_bytes=0)

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
    `prune_retained_dirs(...)` call in the upload route with `pass` left
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


# ---------------------------------------------------------------------------
# #406: the quarantine sinks kept the per-directory sweep that the arena path
# had already been fixed off, so trial-id rotation still bought a fresh budget.
# ---------------------------------------------------------------------------


def _retained_quarantine(server_root, subdir: str) -> list:
    """Every retained entry of one quarantine sink, across ALL trials.

    Deliberately a whole-tree walk rather than a lookup under the trial the
    request named: the defect is that entries pile up under trials the caller
    invented, so a test that only looked where it wrote could not see it.

    ⚑ RECURSES BELOW THE SINK AND COUNTS FILES ONLY. #407 put a per-user
    bucket between the sink and the entries. The previous `quarantine/<subdir>/*`
    glob would now match those BUCKET DIRECTORIES, so an entry-count assertion
    would read "1 entry" for a user holding a thousand and a byte-sum guarded by
    `is_file()` would read ZERO -- a gate that cannot fail, in the tests written
    to prove the budget holds.
    """
    return [
        p
        for sink in server_root.rglob(f"quarantine/{subdir}")
        for p in sink.rglob("*")
        if p.is_file() and not p.name.endswith(".reason.txt")
    ]


def test_quarantine_invalid_retention_survives_trial_id_rotation(tmp_path) -> None:
    """⚑⚑ REGRESSION (#406, P1). The same defect as
    `test_arena_retention_cannot_be_bypassed_by_rotating_trial_ids`, in the
    sibling sink that fix did not reach.

    `quarantine_root` comes from `_quarantine_root(trial_id)` and the sweep was
    the per-directory `prune_retained_dir`. `_normalize_trial_id` only
    syntax-checks the id, and with the default `require_worker_lease=False`
    `_check_worker_compat` admits a trial whose manifest is not published, so
    every invented id was handed a fresh full allowance.

    ⚑ Asserts on ENTRIES ON DISK, not on a response body: every one of these
    uploads is answered `rejected: True` whether the budget holds or not, so
    the response cannot tell the fix from the bug.
    """
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, quarantine_max_entries=3, quarantine_max_bytes=0)

    # 12 rejected uploads, each under a trial id the client invents.
    for i in range(12):
        r = client.post(
            f"/v1/trials/rotate_{i}/upload_shard",
            auth=("u", "p"),
            files={"file": (f"s{i}.zarr.tar", b"not a tar at all %d" % i,
                            "application/x-tar")},
        )
        assert r.status_code == 200, r.text
        assert r.json().get("rejected") is True, r.json()

    kept = _retained_quarantine(server_root, "invalid")
    assert len(kept) <= 3, (
        f"{len(kept)} quarantined shards retained against a 3-entry budget -- "
        "rotating trial ids bought a fresh allowance each time"
    )


def test_quarantine_invalid_holds_a_byte_budget_across_trials(tmp_path) -> None:
    """The byte half of the same budget, read as bytes actually on disk.

    An entry-count assertion alone would pass a sweep that kept three
    arbitrarily large uploads, and disk exhaustion is the failure this bound
    exists to stop -- so the bytes are the thing to measure.
    """
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    # 4 KiB across every trial; entry count unbounded so only bytes can bind.
    client = _build_client(
        server_root, quarantine_max_entries=0, quarantine_max_bytes=4096,
    )

    for i in range(12):
        r = client.post(
            f"/v1/trials/rot_{i}/upload_shard",
            auth=("u", "p"),
            files={"file": (f"s{i}.zarr.tar", b"x" * 1024 + b"%d" % i,
                            "application/x-tar")},
        )
        assert r.status_code == 200, r.text
        assert r.json().get("rejected") is True, r.json()

    # ⚑ Recurses past the per-user bucket #407 introduced. The old
    # `quarantine/invalid/*` + `is_file()` pair sums to ZERO under that layout,
    # so this assertion would have passed no matter how many bytes were on disk.
    # Sidecars are counted here because the budget counts them.
    on_disk = sum(
        f.stat().st_size
        for sink in server_root.rglob("quarantine/invalid")
        for f in sink.rglob("*")
        if f.is_file()
    )
    assert on_disk <= 4096, (
        f"quarantine holds {on_disk} bytes against a 4096 byte budget -- "
        "the budget is being applied per invented trial"
    )


def test_client_report_sink_retention_survives_trial_id_rotation(tmp_path) -> None:
    """The cheapest sink to spam, and the second per-trial call site.

    A small authenticated JSON POST with no tar and no size cap. Bounding
    `quarantine/invalid` across trials and leaving this one per-trial would
    close the expensive bypass and leave the cheap one open.
    """
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, quarantine_max_entries=3, quarantine_max_bytes=0)

    for i in range(12):
        r = client.post(
            f"/v1/trials/report_{i}/report_bad_shard",
            auth=("u", "p"),
            json={"shard_name": f"s{i}.zarr", "reason": "unreadable"},
        )
        assert r.status_code == 200, r.text

    kept = _retained_quarantine(server_root, "client_reports")
    assert len(kept) <= 3, (
        f"{len(kept)} client reports retained against a 3-entry budget -- "
        "rotating trial ids bought a fresh allowance each time"
    )


# ---------------------------------------------------------------------------
# #407 review, P1: `.` is a well-formed trial id that pathlib COLLAPSES, so the
# write landed outside every swept set and no budget was enforced at all.
# ---------------------------------------------------------------------------


DOT_TRIAL = "%2E"  # `.`, percent-encoded so the client sends it verbatim.


def test_a_dot_trial_id_is_refused_rather_than_collapsed(tmp_path) -> None:
    """⚑⚑ REGRESSION (#407 review, P1). The ROOT of the three bypasses below.

    `_trial_id_re` is a charset allowlist and `.` satisfies it, so the id was
    accepted and then JOINED onto a root -- and `trials/./quarantine/invalid`
    collapses to `quarantine/invalid`. The entries landed in a directory no
    cross-trial enumerator lists, so every budget below was silently unenforced.

    Asserts the id is refused at the DOOR, which is what makes one fix cover
    all three sinks. `..` is the control: it was already harmless (it resolves
    onto a swept sink) and must be refused by the same predicate anyway.
    """
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    for encoded in (DOT_TRIAL, "%2E%2E"):
        r = client.post(
            f"/v1/trials/{encoded}/report_bad_shard",
            auth=("u", "p"),
            json={"shard_name": "s.zarr", "reason": "unreadable"},
        )
        assert r.status_code == 400, (
            f"trial_id {encoded!r} was accepted with {r.status_code}: {r.text}"
        )

    # ⚑ NEGATIVE CONTROL: a `.` INSIDE an otherwise ordinary id is legitimate
    # (`trial.1` is a real Ray trial name shape) and must still be accepted.
    # A predicate that rejected every dot would break real workers.
    r = client.post(
        "/v1/trials/trial.1/report_bad_shard",
        auth=("u", "p"),
        json={"shard_name": "s.zarr", "reason": "unreadable"},
    )
    assert r.status_code == 200, f"a dotted-but-safe trial id was refused: {r.text}"


def test_quarantine_invalid_budget_cannot_be_bypassed_by_a_dot_trial_id(
    tmp_path,
) -> None:
    """Budget 1 of 3. Measured before the fix: 12 retained against a 3 budget.

    ⚑ Asserts on entries ON DISK across the WHOLE tree, not under the trial the
    request named: the defect is precisely that the write went somewhere the
    caller did not name and the sweep did not look.
    """
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, quarantine_max_entries=3, quarantine_max_bytes=0)

    for i in range(12):
        client.post(
            f"/v1/trials/{DOT_TRIAL}/upload_shard",
            auth=("u", "p"),
            files={"file": (f"s{i}.zarr.tar", b"not a tar at all %d" % i,
                            "application/x-tar")},
        )

    kept = _retained_quarantine(server_root, "invalid")
    assert len(kept) <= 3, (
        f"{len(kept)} quarantined shards retained against a 3-entry budget -- "
        "a `.` trial id writes outside every swept directory"
    )


def test_client_report_budget_cannot_be_bypassed_by_a_dot_trial_id(tmp_path) -> None:
    """Budget 2 of 3. Measured before the fix: 12 retained against a 3 budget."""
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, quarantine_max_entries=3, quarantine_max_bytes=0)

    for i in range(12):
        client.post(
            f"/v1/trials/{DOT_TRIAL}/report_bad_shard",
            auth=("u", "p"),
            json={"shard_name": f"s{i}.zarr", "reason": "unreadable"},
        )

    kept = _retained_quarantine(server_root, "client_reports")
    assert len(kept) <= 3, (
        f"{len(kept)} client reports retained against a 3-entry budget -- "
        "a `.` trial id writes outside every swept directory"
    )


def test_arena_budget_cannot_be_bypassed_by_a_dot_trial_id(tmp_path) -> None:
    """Budget 3 of 3, and this one is LIVE ON `main`, independent of #407.

    Measured on `main`: 15 arena results retained against a 4-entry per-user
    budget. `test_arena_retention_cannot_be_bypassed_by_rotating_trial_ids`
    (#402) rotates ids of the form `rotate_i` and therefore cannot see the
    strongest form of the bypass it is named for.
    """
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, arena_max_entries=4, arena_max_bytes=0)

    for i in range(15):
        client.post(
            f"/v1/trials/{DOT_TRIAL}/upload_arena_result",
            auth=("u", "p"),
            json=_arena_payload(generated_at_unix=1_700_000_000 + i),
        )

    kept = list(server_root.rglob("arena_inbox/*/*.json"))
    assert len(kept) <= 4, (
        f"{len(kept)} arena results retained against a 4-entry per-user budget -- "
        "a `.` trial id writes outside every swept directory"
    )


# ---------------------------------------------------------------------------
# #407 review, P1-2: the server-wide budget let any worker evict every other
# worker's retained diagnostics. The fairness key is the USER.
# ---------------------------------------------------------------------------


def _seed_two_users(server_root) -> None:
    from chess_anti_engine.server.auth import UserRecord, hash_password, save_users

    users = {}
    for name in ("victim", "attacker"):
        salt, hsh, iters = hash_password("p")
        users[name] = UserRecord(
            username=name, salt_b64=salt, hash_b64=hsh, iterations=iters,
        )
    save_users(server_root / "users.json", users)


def _bad_upload(client, *, user: str, trial: str, name: str, body: bytes):
    return client.post(
        f"/v1/trials/{trial}/upload_shard",
        auth=(user, "p"),
        files={"file": (name, body, "application/x-tar")},
    )


def test_one_worker_cannot_evict_another_workers_quarantined_evidence(
    tmp_path,
) -> None:
    """⚑⚑ REGRESSION (#407 review, P1-2). THE GUARANTEE.

    #406 made `quarantine/invalid` server-wide to close the id-rotation
    bypass, and thereby built the exact "diagnostic-destruction primitive"
    this PR's own body spends three paragraphs arguing against for the
    pooled-sinks case: the sweep is global and mtime-ordered, so ~200 junk
    uploads evict everyone else's retained diagnostics oldest-first.

    Measured before this fix, budget 3: victim's evidence -> `[]`, where
    `main` (per-trial budget) preserved it.

    The username is the only fairness key available: trial ids are
    caller-invented, so keying on them hands back the bypass.
    """
    from .test_server_upload_security import _build_client

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_two_users(server_root)
    client = _build_client(server_root, quarantine_max_entries=3, quarantine_max_bytes=0)

    r = _bad_upload(
        client, user="victim", trial="t_real", name="victim.zarr.tar",
        body=b"not a tar -- the evidence",
    )
    assert r.json().get("rejected") is True, r.json()
    # ⚑ The server renames an upload to its own `tmp_<pid>_<hex>` name, so the
    # evidence is identified by the PATH it landed on, not by what it was
    # called on the wire.
    evidence = _retained_quarantine(server_root, "invalid")
    assert len(evidence) == 1, evidence
    victim_entry = evidence[0]

    # The attacker floods, rotating trial ids as well, from its own account.
    for i in range(12):
        _bad_upload(
            client, user="attacker", trial=f"attacker_{i}",
            name=f"j{i}.zarr.tar", body=b"junk %d" % i,
        )

    assert victim_entry.exists(), (
        "the attacker's flood destroyed the victim's retained evidence at "
        f"{victim_entry} -- the budget is a diagnostic-destruction primitive "
        "across users"
    )
    # ⚑ AND THE BOUND STILL HOLDS. A fix that simply stopped evicting would
    # also keep the victim's entry, and would reopen finding [4]. The
    # attacker's own bucket must still be at budget.
    kept = _retained_quarantine(server_root, "invalid")
    attacker_kept = [p for p in kept if p != victim_entry]
    assert len(attacker_kept) <= 3, (
        f"{len(attacker_kept)} attacker entries retained against a 3-entry "
        "per-user budget -- per-user fairness was bought by dropping the bound"
    )
    # The buckets really are keyed by user, not by trial.
    assert victim_entry.parent.name == "victim", victim_entry
    assert {p.parent.name for p in attacker_kept} == {"attacker"}, attacker_kept


def test_one_users_own_flood_still_evicts_its_own_older_evidence(tmp_path) -> None:
    """⚑ THE LIMIT OF THE P1-2 FIX, PINNED RATHER THAN ASSUMED.

    The review's repro is single-user, cross-TRIAL. This fix does not restore
    `main`'s behaviour there, and that is deliberate: within one authenticated
    account the entries compete for that account's quota, oldest-first. Trial
    id cannot be the fairness key -- it is caller-invented, so a per-trial
    floor hands back the rotation bypass, and per-trial fair EVICTION does not
    help either, because one entry under each of many invented ids makes every
    contributor tie at one and the tie-break is the oldest entry, the victim.

    So: one worker cannot destroy ANOTHER worker's diagnostics (the test
    above), and one worker can still age out its OWN. This test exists so that
    limit is a recorded decision rather than an accident nobody measured.
    """
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, quarantine_max_entries=3, quarantine_max_bytes=0)

    r = _bad_upload(
        client, user="u", trial="t_real", name="old.zarr.tar", body=b"not a tar old",
    )
    assert r.json().get("rejected") is True, r.json()
    first = _retained_quarantine(server_root, "invalid")
    assert len(first) == 1, first
    oldest = first[0]

    for i in range(12):
        _bad_upload(
            client, user="u", trial=f"rot_{i}", name=f"j{i}.zarr.tar",
            body=b"junk %d" % i,
        )

    kept = _retained_quarantine(server_root, "invalid")
    assert len(kept) <= 3, (
        f"{len(kept)} retained against a 3-entry per-user budget -- the "
        "per-user bucketing lost the bound it was built on top of"
    )
    assert not oldest.exists(), (
        f"the oldest entry ({oldest}) survived a 12-upload flood from the SAME "
        "account -- if this now passes, the budget grew a per-trial floor, "
        "which is exactly the rotation bypass #406 closed"
    )


# ---------------------------------------------------------------------------
# #407 re-review, P2: `worker_self_register` makes the bucket count
# CALLER-controlled, so a per-user budget is not a bound. Ceiling is global;
# fairness is the victim choice under it.
# ---------------------------------------------------------------------------


def test_minted_accounts_cannot_raise_the_quarantine_ceiling(tmp_path) -> None:
    """⚑⚑ REGRESSION (#407 re-review, P2). The per-user budget's stated premise
    was "usernames are administrator-created, so the bucket count is bounded".

    `worker_self_register` is a shipped yaml run-config key
    (`config_yaml.py:318`, plumbed via `harness.py:495`) and the #400-#403
    volunteer-hardening series exists to enable it. With it on, the caller
    mints buckets, so a per-user allowance multiplies the ceiling by a number
    the ATTACKER picks.

    Measured before this fix: 3 minted accounts retained **9** entries against
    a 3-entry budget. At the 4 GiB default that is ~40 GiB in the first hour
    from one IP (registration throttles at 5/IP/hour with no global account
    cap), against the 8 GiB the global bound had.

    Nothing was live -- the flag defaults off and production's
    `pbt2_small.yaml` does not set it -- but a docstring claiming a bound that
    a config key silently falsifies is this repo's signature defect.
    """
    from .test_server_upload_security import _build_client

    server_root = tmp_path / "server"
    server_root.mkdir()
    client = _build_client(
        server_root, quarantine_max_entries=3, quarantine_max_bytes=0,
        worker_self_register=True,
    )

    for account in ("attacker_a", "attacker_b", "attacker_c"):
        for i in range(4):
            client.post(
                "/v1/upload_shard",
                auth=(account, "pw12345678"),
                files={"file": (f"s{i}.zarr.tar", b"not a tar %d" % i,
                                "application/x-tar")},
            )

    kept = _retained_quarantine(server_root, "invalid")
    buckets = sorted({p.parent.name for p in kept})
    assert len(kept) <= 3, (
        f"{len(kept)} entries retained across buckets {buckets} against a "
        "3-entry GLOBAL budget -- self-registration mints buckets, so a "
        "per-user allowance is a caller-controlled ceiling"
    )


def test_minted_accounts_cannot_raise_the_arena_ceiling(tmp_path) -> None:
    """The identical exposure in the arena sink -- PRE-EXISTING since #402.

    `_arena_user_dirs_all_trials` keyed the budget on username from #402, so
    `arena_user_max_*` was `users x 256 MiB` with a caller-controlled
    multiplier the moment self-registration is on. Measured before this fix:
    3 minted accounts retained **6** results against a 2-entry budget.

    This is a bug being closed, not precedent that made the quarantine design
    safe -- the quarantine sinks were given this shape earlier in #407 by
    copying it.
    """
    from .test_server_upload_security import _build_client

    server_root = tmp_path / "server"
    server_root.mkdir()
    client = _build_client(
        server_root, arena_max_entries=2, arena_max_bytes=0,
        worker_self_register=True,
    )

    for account in ("a_one", "a_two", "a_three"):
        for i in range(4):
            client.post(
                "/v1/upload_arena_result",
                auth=(account, "pw12345678"),
                json=_arena_payload(generated_at_unix=1_700_000_000 + i),
            )

    kept = list(server_root.rglob("arena_inbox/*/*.json"))
    buckets = sorted({p.parent.name for p in kept})
    assert len(kept) <= 2, (
        f"{len(kept)} arena results retained across buckets {buckets} against "
        "a 2-entry GLOBAL budget -- minted accounts bought ceiling"
    )


def test_the_ceiling_evicts_the_largest_bucket_not_the_oldest_entry(
    tmp_path,
) -> None:
    """⚑⚑ THE FAIRNESS POLICY ITSELF. "Which bucket loses" is exactly where a
    fairness fix silently becomes a fairness bug, so it gets a direct test
    rather than being inferred from the end-to-end ones.

    Policy: evict the OLDEST entry of the LARGEST-CONTRIBUTING bucket. Here the
    victim's single entry is the OLDEST thing in the sink, so a plain
    oldest-first sweep -- the #406 behaviour -- destroys it first. Largest-first
    puts the whole cost of the flood on the flooder.
    """
    from chess_anti_engine.server.app import prune_retained_dirs

    sink = tmp_path / "invalid"
    victim_dir = sink / "victim"
    hog_dir = sink / "hog"
    victim_dir.mkdir(parents=True)
    hog_dir.mkdir(parents=True)

    victim = _mk(victim_dir, "v0.tar", size=100, mtime=1_000_000, sidecar=False)
    hogs = [
        _mk(hog_dir, f"h{i}.tar", size=100, mtime=1_000_010 + i, sidecar=False)
        for i in range(6)
    ]

    prune_retained_dirs([victim_dir, hog_dir], max_bytes=0, max_entries=3)

    assert victim.exists(), (
        "the victim's single entry was evicted -- it is the OLDEST in the sink, "
        "so this is a global oldest-first sweep, not largest-bucket-first"
    )
    assert sum(1 for p in hogs if p.exists()) == 2, (
        f"hog kept {[p.name for p in hogs if p.exists()]}; the flood should "
        "absorb every eviction down to the 3-entry ceiling"
    )


def test_the_byte_ceiling_evicts_the_bucket_holding_the_bytes(tmp_path) -> None:
    """⚑ THE DIMENSION MATTERS: largest is measured in whichever ceiling BINDS.

    Here bytes bind and entries do not. The bucket with the most ENTRIES holds
    almost no bytes; the bucket with the most BYTES holds two entries. Choosing
    the victim by entry count would evict the small-bytes bucket and never get
    under the byte ceiling without destroying far more diagnostics than needed.
    """
    from chess_anti_engine.server.app import prune_retained_dirs

    sink = tmp_path / "invalid"
    many_small = sink / "many_small"
    few_big = sink / "few_big"
    many_small.mkdir(parents=True)
    few_big.mkdir(parents=True)

    smalls = [
        _mk(many_small, f"s{i}.tar", size=10, mtime=1_000_000 + i, sidecar=False)
        for i in range(5)
    ]
    bigs = [
        _mk(few_big, f"b{i}.tar", size=500, mtime=1_000_100 + i, sidecar=False)
        for i in range(2)
    ]

    # 1050 bytes total against a 600-byte ceiling; entry count unbounded.
    prune_retained_dirs([many_small, few_big], max_bytes=600, max_entries=0)

    assert all(p.exists() for p in smalls), (
        "the 5-entry / 50-byte bucket was evicted under a BYTE ceiling -- the "
        "victim is being chosen by entry count while bytes are what binds"
    )
    assert sum(1 for p in bigs if p.exists()) == 1, (
        f"few_big kept {[p.name for p in bigs if p.exists()]}; one 500-byte "
        "eviction is what brings 1050 under 600"
    )


def test_the_two_quarantine_sinks_do_not_share_one_budget(tmp_path) -> None:
    """⚑ NEGATIVE CONTROL on the SHAPE of the fix, not just its presence.

    Pooling both sinks into a single budget would also close the rotation
    bypass -- and would hand the CHEAPEST sink the power to evict the most
    EXPENSIVE diagnostics, because the sweep is global and mtime-ordered: a
    burst of tiny `report_bad_shard` posts would delete every retained bad
    shard. This pins that the shard sink survives a report flood.
    """
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, quarantine_max_entries=3, quarantine_max_bytes=0)

    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("keepme.zarr.tar", b"not a tar", "application/x-tar")},
    )
    assert r.json().get("rejected") is True, r.json()

    for i in range(12):
        client.post(
            "/v1/report_bad_shard",
            auth=("u", "p"),
            json={"shard_name": f"s{i}.zarr", "reason": "unreadable"},
        )

    assert _retained_quarantine(server_root, "invalid"), (
        "a flood of cheap client reports evicted the retained bad shard -- "
        "the two sinks are sharing one budget"
    )


def test_pre_bucketing_quarantine_entries_are_drained_by_the_sweep(
    tmp_path,
) -> None:
    """⚑ REGRESSION (#407 re-review, P3). `legacy_roots` had ZERO coverage --
    mutating it to never be walked left the whole suite green.

    #407 moved quarantine entries from `quarantine/<subdir>/<entry>` to
    `quarantine/<subdir>/<user>/<entry>`. Entries written by the OLD code sit
    at the flat level, and if the sweep does not reach them they are a
    permanently unswept residue: the disk bound quietly not applying to exactly
    the entries that predate the bound. That is the "accepted and then silently
    ignored" shape, so it needs a test rather than a hand-verification.

    Simulates a real upgrade: 10 pre-bucketing flat entries already on disk,
    then one new upload arrives and triggers the sweep.
    """
    import os

    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, quarantine_max_entries=3, quarantine_max_bytes=0)

    sink = server_root / "quarantine" / "invalid"
    sink.mkdir(parents=True, exist_ok=True)
    legacy = []
    for i in range(10):
        p = sink / f"old{i}.zarr.tar"
        p.write_bytes(b"x" * 100)
        os.utime(p, (1_000_000 + i, 1_000_000 + i))
        legacy.append(p)

    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("new.zarr.tar", b"not a tar", "application/x-tar")},
    )
    assert r.json().get("rejected") is True, r.json()

    kept = _retained_quarantine(server_root, "invalid")
    assert len(kept) <= 3, (
        f"{len(kept)} entries retained against a 3-entry budget -- "
        f"{sum(1 for p in legacy if p.exists())} pre-bucketing flat entries "
        "were never swept, so the bound does not apply to them"
    )
    # ⚑ And the drain is FAIR too: the unattributed bucket holds 10 of the 11,
    # so it absorbs the evictions and the new upload survives.
    assert any(p.parent.name == "u" for p in kept), (
        "the new upload was evicted while 10 legacy entries competed -- the "
        "legacy pool is not being treated as its own bucket"
    )


def test_a_sweep_does_not_claim_sidecar_bytes_another_sweep_removed(
    tmp_path,
) -> None:
    """⚑ REGRESSION (#407 review, P2). The HALF of the exactness fix that #406
    left inexact.

    #406 tracked the ENTRY's deletion but left the sidecar on
    `unlink(missing_ok=True)` -- untracked -- while crediting the whole `size`
    to `freed_bytes`, and `size` INCLUDES the sidecar's bytes. So under the
    very two-sweep race the fix was written for, a sweep whose peer had already
    taken the `.reason.txt` still reported those bytes as its own, and
    `freed_bytes` still did not mean "bytes this sweep removed".

    Driven through the real pair, with the concurrent deletion landing between
    the snapshot and the sweep exactly where a racing sweep's would -- and it
    removes ONLY the sidecar, which is the case the entry-level fix cannot see.
    """
    from chess_anti_engine.server.app import _evict_fairly, _retention_entries

    q = tmp_path / "invalid"
    q.mkdir()
    for i in range(3):
        _mk(q, f"s{i}.tar", size=1000, mtime=1_000_000 + i, sidecar=True)
    sidecar_bytes = (q / "s0.tar.reason.txt").stat().st_size
    assert sidecar_bytes > 0

    entries = _retention_entries(q)
    assert len(entries) == 3

    # A concurrent sweep took the oldest victim's SIDECAR only.
    (q / "s0.tar.reason.txt").unlink()

    freed_entries, freed_bytes = _evict_fairly(
        entries, max_bytes=0, max_entries=2, log=None, label="test",
    )

    assert freed_entries == 1, freed_entries
    assert freed_bytes == 1000, (
        f"reported {freed_bytes} bytes freed; this sweep removed the 1000-byte "
        f"entry but NOT its {sidecar_bytes}-byte sidecar, which another sweep "
        "had already taken"
    )


def test_a_sweep_reports_only_the_evictions_it_performed(tmp_path) -> None:
    """⚑ REGRESSION (#406, P2). `unlink(missing_ok=True)` made a file another
    concurrent sweep had already deleted indistinguishable from one this sweep
    deleted, so both counted it and the reported totals summed to more than the
    bytes that left the disk.

    Driven through the real production pair: `_retention_entries` builds the
    snapshot, a deletion lands between the two calls exactly where a racing
    sweep's would, and `_evict_fairly` then runs on that snapshot. No
    test double, and nothing pre-seeded -- the state asserted on is the state
    the production functions produce.
    """
    from chess_anti_engine.server.app import _evict_fairly, _retention_entries

    q = tmp_path / "invalid"
    q.mkdir()
    for i in range(4):
        _mk(q, f"s{i}.tar", size=1000, mtime=1_000_000 + i, sidecar=False)

    entries = _retention_entries(q)
    assert len(entries) == 4

    # A concurrent sweep gets to the oldest victim first.
    (q / "s0.tar").unlink()

    freed_entries, freed_bytes = _evict_fairly(
        entries, max_bytes=0, max_entries=2, log=None, label="test",
    )

    assert freed_entries == 1, (
        f"reported {freed_entries} evictions when only s1 was actually removed "
        "by this sweep -- a file the other sweep deleted is being counted"
    )
    assert freed_bytes == 1000, f"reported {freed_bytes} bytes freed, removed 1000"
    # ⚑ The other half, and the reason `over_count` still counts absences
    # rather than this sweep's own deletions: not crediting the phantom must
    # not make the sweep keep deleting. The budget is 2 and 2 must survive.
    remaining = sorted(p.name for p in q.iterdir())
    assert remaining == ["s2.tar", "s3.tar"], (
        f"sweep over-evicted to {remaining} -- it deleted past the budget "
        "because a concurrently-removed entry stopped counting toward it"
    )


# ---------------------------------------------------------------------------
# #406: the cross-trial arena walk ran on the event loop, over a directory set
# that eviction never shrank.
# ---------------------------------------------------------------------------


def test_the_cross_trial_arena_walk_does_not_run_on_the_event_loop(
    tmp_path, monkeypatch,
) -> None:
    """⚑⚑ REGRESSION (#406, P1). Only the SWEEP was hopped to a threadpool;
    `_arena_user_dirs_all_trials` -- the part whose cost grows with the attack
    -- was evaluated on the loop thread to build the argument for it.

    The instrument is `asyncio.get_running_loop()` at the moment of the call:
    it SUCCEEDS on the event loop thread and raises `RuntimeError` inside
    `run_in_threadpool`. That is production's own definition of "on the loop",
    not a proxy for it.

    Exactly ONE on-loop `resolve_user_dir` is legitimate -- the route
    resolving the directory it is about to write to. Every additional one is a
    per-trial `realpath` pair charged to the loop thread.
    """
    import asyncio

    from chess_anti_engine.server import app as app_mod

    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    # Five trials already on disk, so the walk has something to cost.
    for i in range(5):
        (server_root / "trials" / f"t{i}" / "arena_inbox" / "u").mkdir(parents=True)

    real_resolve = app_mod.resolve_user_dir
    on_loop: list[str] = []

    def _spy(arena_root, username):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # Off the loop: a threadpool worker, which is the point.
        else:
            on_loop.append(str(arena_root))
        return real_resolve(arena_root, username)

    monkeypatch.setattr(app_mod, "resolve_user_dir", _spy)

    client = _build_client(server_root)
    r = client.post("/v1/upload_arena_result", auth=("u", "p"), json=_arena_payload())
    assert r.status_code == 200, r.text

    assert len(on_loop) == 1, (
        f"{len(on_loop)} arena directory resolutions ran on the event loop "
        f"({on_loop}); only the route's own target directory may. The "
        "cross-trial walk is back on the loop thread."
    )


def test_eviction_removes_the_arena_directories_it_empties(tmp_path) -> None:
    """⚑ REGRESSION (#406, P1). Eviction deleted the JSON and left
    `trials/<id>/arena_inbox/<user>/` in place, so a rotation burst left a
    permanent residue that every later cross-trial walk had to stat.

    Asserts on the DIRECTORIES on disk after the sweep, not on a response.
    """
    from .test_server_upload_security import _build_client, _seed_user

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, arena_max_entries=1, arena_max_bytes=0)

    for i in range(5):
        r = client.post(
            f"/v1/trials/rot_{i}/upload_arena_result",
            auth=("u", "p"),
            json=_arena_payload(generated_at_unix=1_700_000_000 + i),
        )
        assert r.status_code == 200, r.text

    surviving = sorted(p.parent.name for p in server_root.glob("trials/*/arena_inbox"))
    assert surviving == ["rot_4"], (
        f"emptied arena directories left behind for {surviving} -- the scanned "
        "set grows with every invented trial id and never shrinks"
    )
    # ⚑ NEGATIVE CONTROL. The server-root `arena_inbox` is created at boot as
    # part of the layout; sweeping it away would leave the tree unlike a fresh
    # boot's, and it is a single directory, so it is not the growth.
    assert (server_root / "arena_inbox").is_dir(), (
        "the boot-time arena_inbox was removed by the empty-directory sweep"
    )
    # And the results themselves still obey the budget.
    assert len(list(server_root.rglob("arena_inbox/*/*.json"))) == 1


def test_an_arena_write_survives_a_concurrent_empty_dir_sweep(tmp_path) -> None:
    """The race `drop_empty_arena_dirs` opens, closed at the writer.

    A request for trial A can be suspended between its `mkdir` and its
    `write_bytes` while a request for trial B sweeps A's now-empty directory
    away; without the retry that is a 500 on an upload that did nothing wrong.
    Driven by deleting the directory for real, which is what the racing sweep
    does.
    """
    from chess_anti_engine.server.app import write_arena_result

    out = tmp_path / "arena_inbox" / "u" / "deadbeef.json"
    out.parent.mkdir(parents=True)
    out.parent.rmdir()  # The concurrent sweep, mid-request.

    write_arena_result(out, b'{"games": 1}')

    assert out.read_bytes() == b'{"games": 1}'


def test_the_empty_dir_sweep_keeps_directories_that_still_hold_results(
    tmp_path,
) -> None:
    """⚑ NEGATIVE CONTROL. A sweep that removed unconditionally would pass the
    residue test above while destroying live arena results."""
    from chess_anti_engine.server.app import drop_empty_arena_dirs

    root = tmp_path / "arena_inbox"
    empty = root / "trials" / "t0" / "arena_inbox" / "u"
    full = root / "trials" / "t1" / "arena_inbox" / "u"
    keep = root / "trials" / "t2" / "arena_inbox" / "u"
    for d in (empty, full, keep):
        d.mkdir(parents=True)
    (full / "r.json").write_text("{}", encoding="utf-8")

    removed = drop_empty_arena_dirs(
        [empty, full, keep], keep=keep, default_arena_root=root,
    )

    assert removed == [empty]
    assert not empty.exists()
    assert not empty.parent.exists(), "the emptied per-trial arena_inbox stayed"
    assert (full / "r.json").is_file(), "a directory holding a result was removed"
    assert keep.is_dir(), "the request's own target directory was removed"


# ---------------------------------------------------------------------------
# #419 F1: the budget subtracted bytes that were still on disk, and the
# sidecar was unlinked before the entry delete was known to have succeeded.
# ---------------------------------------------------------------------------


def _undeletable_dir_entry(bucket, name: str, *, inner_size: int, mtime: float):
    """A retained entry `_evict_fairly` will fail to remove.

    A directory entry whose own mode denies write, so `shutil.rmtree` cannot
    unlink the child and takes the partial-removal branch. Returns the entry
    path; the caller must restore the mode (see `_restore_mode`).

    ⚑ The payload sits DIRECTLY in `d`, not in a subdirectory: unlinking it
    then needs write on `d` itself, which is what is denied. With a `d/inner/`
    level, `rmtree` removes the leaf bytes before failing on the directory, and
    the fixture measures a partially-drained entry instead of an undeletable
    one.
    """
    import os

    d = bucket / name
    d.mkdir(parents=True)
    (d / "blob").write_bytes(b"x" * inner_size)
    s = bucket / f"{name}.reason.txt"
    s.write_text("ValueError: bad", encoding="utf-8")
    os.utime(s, (mtime, mtime))
    os.utime(d, (mtime, mtime))
    os.chmod(d, 0o500)
    return d


def _restore_mode(d) -> None:
    import contextlib
    import os

    with contextlib.suppress(OSError):
        os.chmod(d, 0o700)


def _bytes_on_disk(root) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _sweep_bounded(fn, *, seconds: float = 30.0) -> tuple[int, int]:
    """Run `fn()` on a daemon thread and FAIL rather than hang.

    The fix for F1 stops crediting bytes for an entry it could not delete, and
    the obvious wrong shape of that fix -- re-selecting the same victim -- is
    an infinite loop, not a wrong number. A bare call would hang the suite;
    the repo has no `pytest-timeout`.
    """
    import threading

    value: list[tuple[int, int]] = []
    error: list[BaseException] = []

    def _run() -> None:
        try:
            value.append(fn())
        except BaseException as exc:
            error.append(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), (
        f"the sweep did not finish within {seconds}s -- it is spinning on a "
        "victim it cannot delete"
    )
    if error:
        raise error[0]
    return value[0]


def test_a_sweep_drains_to_budget_when_every_victim_is_deletable(tmp_path) -> None:
    """⚑ SERIAL CONTROL for the two F1 tests below.

    Same shape, same ceiling, same entry sizes -- only the oldest entry's mode
    differs. Without this, "the sweep stopped over budget" cannot be told from
    "this setup was never going to reach budget".
    """
    from chess_anti_engine.server.app import prune_retained_dirs

    u = tmp_path / "invalid" / "alice"
    u.mkdir(parents=True)
    d = _undeletable_dir_entry(u, "a_bad_shard", inner_size=200, mtime=1000.0)
    _restore_mode(d)  # the control: it IS deletable
    for i in range(3):
        _mk(u, f"z{i}.tar", size=100, mtime=2000.0 + i)

    freed_entries, freed_bytes = prune_retained_dirs(
        [u], max_bytes=200, max_entries=0, log=None,
    )

    # 215 (the dir entry + its 15-byte sidecar) + 3 x 115. Evicting the three
    # oldest leaves 115, which is under the ceiling.
    assert (freed_entries, freed_bytes) == (3, 445), (freed_entries, freed_bytes)
    assert _bytes_on_disk(u) == 115, "the control did not drain to budget"
    assert sorted(p.name for p in u.iterdir()) == ["z2.tar", "z2.tar.reason.txt"]


def test_a_sweep_does_not_credit_itself_bytes_it_failed_to_delete(tmp_path) -> None:
    """⚑ REGRESSION (#419 F1). `_evict_fairly` ran `total_bytes -= size`,
    `live_bytes[victim] -= size` and `gone_entries += 1` UNCONDITIONALLY and
    used `removed` only for telemetry -- so on the failed-delete branch, one
    line after concluding it could not remove the entry, it credited itself the
    entry's bytes and stopped while still over the ceiling.

    Measured on `main` with this exact fixture: reported `(2, 230)`, and `z2`
    left on disk -- the sweep stopped 315 bytes over a 200-byte ceiling with a
    removable entry still sitting there. The control above shows the same
    fixture stops exactly AT the ceiling when the oldest victim can actually be
    deleted.

    The entry that cannot be deleted is 215 bytes, so 215 is the floor; the
    assertion is that everything ELSE went, not that the ceiling was met.
    """
    import os

    import pytest

    from chess_anti_engine.server.app import prune_retained_dirs

    if os.geteuid() == 0:
        pytest.skip("running as root: a 0o500 directory is still writable")

    u = tmp_path / "invalid" / "alice"
    u.mkdir(parents=True)
    d = _undeletable_dir_entry(u, "a_bad_shard", inner_size=200, mtime=1000.0)
    try:
        for i in range(3):
            _mk(u, f"z{i}.tar", size=100, mtime=2000.0 + i)

        freed_entries, freed_bytes = _sweep_bounded(
            lambda: prune_retained_dirs(
                [u], max_bytes=200, max_entries=0, log=None,
            ),
        )

        assert d.exists(), "the fixture's undeletable entry was deleted"
        left = sorted(p.name for p in u.iterdir())
        assert left == ["a_bad_shard", "a_bad_shard.reason.txt"], (
            f"sweep stopped at {left} with a 200-byte ceiling -- it credited "
            "itself the 215 bytes of the entry it FAILED to delete and "
            "believed it was under budget"
        )
        assert _bytes_on_disk(u) == 215, _bytes_on_disk(u)
        assert (freed_entries, freed_bytes) == (3, 345), (freed_entries, freed_bytes)
    finally:
        _restore_mode(d)


def test_a_kept_entry_keeps_its_reason_sidecar_when_its_delete_fails(
    tmp_path,
) -> None:
    """⚑ REGRESSION (#419 F1, second half). The sidecar was unlinked BEFORE the
    entry delete was known to succeed, so a quarantine entry the sweep could
    not remove survived with its explanation destroyed -- which is the entire
    point of retaining it.

    Measured on `main`: `a_bad_shard` kept, `a_bad_shard.reason.txt` gone.
    """
    import os

    import pytest

    from chess_anti_engine.server.app import prune_retained_dirs

    if os.geteuid() == 0:
        pytest.skip("running as root: a 0o500 directory is still writable")

    u = tmp_path / "invalid" / "alice"
    u.mkdir(parents=True)
    d = _undeletable_dir_entry(u, "a_bad_shard", inner_size=200, mtime=1000.0)
    try:
        for i in range(3):
            _mk(u, f"z{i}.tar", size=100, mtime=2000.0 + i)

        _sweep_bounded(
            lambda: prune_retained_dirs(
                [u], max_bytes=200, max_entries=0, log=None,
            ),
        )

        assert d.exists(), "the fixture's undeletable entry was deleted"
        assert (u / "a_bad_shard.reason.txt").is_file(), (
            "the retained entry's .reason.txt was destroyed while the entry it "
            "explains was kept"
        )
    finally:
        _restore_mode(d)


def test_a_sweep_still_cleans_the_sidecar_of_an_entry_a_peer_removed(
    tmp_path,
) -> None:
    """⚑ CONTROL, NOT A REGRESSION -- it passes on `main` too. It pins the
    other direction of the reorder, so "do the entry first" cannot be read as
    "skip the sidecar whenever the entry was already gone".

    A peer takes the entry between the walk and the sweep; its `.reason.txt` is
    still this sweep's to clean up, or the reorder would MANUFACTURE the
    orphaned sidecars of #419 F4.
    """
    from chess_anti_engine.server.app import _evict_fairly, _retention_entries

    q = tmp_path / "invalid"
    q.mkdir()
    for i in range(3):
        _mk(q, f"s{i}.tar", size=1000, mtime=1_000_000 + i, sidecar=True)

    entries = _retention_entries(q)
    (q / "s0.tar").unlink()  # the peer, between the walk and the sweep

    freed_entries, freed_bytes = _evict_fairly(
        entries, max_bytes=0, max_entries=2, log=None, label="test",
    )

    assert not (q / "s0.tar.reason.txt").exists(), (
        "the sidecar of an entry a peer removed was left orphaned"
    )
    # This sweep removed s0's sidecar (15 bytes) but not s0 itself.
    assert freed_entries == 0, freed_entries
    assert freed_bytes == 15, freed_bytes


# ---------------------------------------------------------------------------
# #419 F2: the empty-dir guard compared one resolved operand against one
# unresolved one, so a symlinked server root inverted it.
# ---------------------------------------------------------------------------


def _arena_tree(root):
    """The production shape: `<root>/arena_inbox/{alice,bob}`, both empty.

    Mirrors `create_app`'s layout so the test is about the guard rather than
    about a directory tree invented for it.
    """
    inbox = root / "arena_inbox"
    for u in ("alice", "bob"):
        (inbox / u).mkdir(parents=True)
    return inbox


def test_the_empty_dir_guard_holds_through_an_unsymlinked_root(tmp_path) -> None:
    """⚑ SERIAL CONTROL for the symlink test below (passes on `main` too).

    Same call, same wiring, one difference: the server root is reached by its
    real path. Without this, "the guard let the caller's own directory be
    deleted" cannot be told from "this fixture never protected it".
    """
    from chess_anti_engine.server.app import drop_empty_arena_dirs, resolve_user_dir

    real = tmp_path / "server"
    real.mkdir()
    inbox = _arena_tree(real)

    # Production wiring: `dirs` off create_app's root, `keep` from
    # `resolve_user_dir` (which resolves).
    dirs = sorted(d for d in inbox.iterdir() if d.is_dir())
    keep = resolve_user_dir(inbox, "bob")

    removed = drop_empty_arena_dirs(dirs, keep=keep, default_arena_root=inbox)

    assert removed == [inbox / "alice"], removed
    assert (inbox / "bob").is_dir(), "the calling request's own directory was removed"
    assert inbox.is_dir(), "the server-root arena_inbox was removed"


def test_the_empty_dir_guard_holds_through_a_symlinked_server_root(
    tmp_path,
) -> None:
    """⚑ REGRESSION (#419 F2). `create_app` builds `root = Path(server_root)`
    UNRESOLVED, so `_arena_all_user_dirs()` yields unresolved paths, while
    `resolve_user_dir` returns resolved ones -- and the call site resolved only
    `default_arena_root`. One resolved operand, one unresolved: under a
    symlinked server root every comparison in the guard is false and the guard
    silently stops guarding, which is what its own docstring warned about.

    Measured on `main` with this fixture: `removed` came back
    `[alice, bob]` -- including the calling request's own `bob` -- and the
    server-root `arena_inbox` itself was rmdir'd, which the docstring says must
    never happen. Not reachable on today's config (no symlink under
    `runs/pbt2_small/server`), but production moved its data root between
    drives in July and a symlinked root is how that gets done.
    """
    from chess_anti_engine.server.app import drop_empty_arena_dirs, resolve_user_dir

    # ⚑ BOTH SPELLINGS of `default_arena_root`, because the two guards in this
    # function fail for the same reason at different call-site spellings.
    # `main`'s call site passed `arena_inbox.resolve()` -- resolved, against
    # unresolved `dirs` -- which is the spelling that rmdir'd the server-root
    # `arena_inbox`; this branch's passes it unresolved. Neither may invert the
    # guard, so the function is exercised with each.
    for case, resolve_root in (("resolved_root", True), ("unresolved_root", False)):
        home = tmp_path / case
        home.mkdir()
        real = home / "real_server"
        real.mkdir()
        inbox = _arena_tree(real)
        link = home / "server"
        link.symlink_to(real, target_is_directory=True)

        # `dirs` reach the sweep through the SYMLINK, exactly as they do when
        # `server_root` is the link; `keep` is resolved, exactly as
        # `resolve_user_dir` returns it.
        link_inbox = link / "arena_inbox"
        dirs = sorted(d for d in link_inbox.iterdir() if d.is_dir())
        keep = resolve_user_dir(link_inbox, "bob")
        assert keep == real / "arena_inbox" / "bob", keep

        removed = drop_empty_arena_dirs(
            dirs,
            keep=keep,
            default_arena_root=inbox if resolve_root else link_inbox,
        )

        assert (real / "arena_inbox" / "bob").is_dir(), (
            f"[{case}] the calling request's own directory was removed -- the "
            "`keep` guard compared a symlinked path against a resolved one "
            "and lost"
        )
        assert removed == [link_inbox / "alice"], (case, removed)
        assert (real / "arena_inbox").is_dir(), (
            f"[{case}] the server-root arena_inbox was removed, which this "
            "function's docstring says must never happen"
        )


def test_the_server_root_arena_inbox_survives_emptying_every_user_dir(
    tmp_path,
) -> None:
    """⚑ REGRESSION (#419 F2, the second guard). The `keep` guard is not the
    only comparison that inverts: `d.parent != default_arena_root` decides
    whether the emptied PARENT is removed too, and it is the one thing standing
    between this sweep and the server-root `arena_inbox` that `create_app`
    mkdirs at boot.

    It only becomes observable once the inbox is genuinely empty, which is why
    the symlink test above cannot see it: there, `keep` keeps one user
    directory alive and the `rmdir` fails on a non-empty directory for the
    wrong reason. Here every user directory in the ROOT inbox is emptied while
    the calling request's `keep` lives under a per-trial inbox -- the ordinary
    cross-trial shape of this sweep -- so the guard is the only protection
    left.

    Measured on `main` with `default_arena_root=arena_inbox.resolve()` (its
    call-site spelling) and `dirs` reached through a symlinked root: the
    server-root `arena_inbox` was removed.
    """
    from chess_anti_engine.server.app import drop_empty_arena_dirs, resolve_user_dir

    for case, resolve_root in (("resolved_root", True), ("unresolved_root", False)):
        home = tmp_path / case
        home.mkdir()
        real = home / "real_server"
        real.mkdir()
        inbox = _arena_tree(real)
        trial_inbox = real / "trials" / "t0" / "arena_inbox" / "bob"
        trial_inbox.mkdir(parents=True)
        link = home / "server"
        link.symlink_to(real, target_is_directory=True)

        link_inbox = link / "arena_inbox"
        dirs = sorted(d for d in link_inbox.iterdir() if d.is_dir())
        # The calling request wrote under a TRIAL inbox, so nothing in the root
        # inbox is protected by `keep`.
        keep = resolve_user_dir(link / "trials" / "t0" / "arena_inbox", "bob")

        removed = drop_empty_arena_dirs(
            dirs,
            keep=keep,
            default_arena_root=inbox if resolve_root else link_inbox,
        )

        assert sorted(r.name for r in removed) == ["alice", "bob"], (case, removed)
        assert (real / "arena_inbox").is_dir(), (
            f"[{case}] the server-root arena_inbox was removed once its last "
            "user directory went -- the parent guard compared a symlinked "
            "path against a resolved one and lost"
        )


# ---------------------------------------------------------------------------
# #419 F4: an orphaned `.reason.txt` was counted by nothing and deleted by
# nothing.
# ---------------------------------------------------------------------------


def test_a_sidecar_with_its_shard_present_is_not_an_entry_of_its_own(
    tmp_path,
) -> None:
    """⚑ NEGATIVE CONTROL for the orphan test below, and for the obvious wrong
    fix to it.

    Counting every `.reason.txt` as an entry would close the orphan hole and
    halve the effective entry budget while doing it, and could evict a sidecar
    while keeping the shard it explains. A paired sidecar must still be
    accounted WITH its shard: one entry, both sizes.
    """
    from chess_anti_engine.server.app import _retention_entries

    q = tmp_path / "invalid"
    q.mkdir()
    _mk(q, "good.tar", size=100, mtime=1000.0, sidecar=True)

    entries = _retention_entries(q)

    assert len(entries) == 1, [e.path.name for e in entries]
    assert entries[0].path.name == "good.tar"
    assert entries[0].size == 115, entries[0]
    assert entries[0].sidecar_size == 15, entries[0]


def test_an_orphaned_reason_sidecar_is_counted_and_swept(tmp_path) -> None:
    """⚑ REGRESSION (#419 F4). `_retention_entries` skipped every
    `*.reason.txt` and counted sidecar bytes only VIA an existing entry, so a
    sidecar whose entry is gone was counted by nothing and deleted by nothing:
    invisible to the ceiling and unreachable by every sweep, forever.

    The interleaving that creates one is on the production quarantine path --
    writer `replace`s the shard, a peer sweep evicts it, writer then writes the
    sidecar.

    Measured on `main`: the sweep below reports `(1, 140)` and leaves the
    5000-byte orphan on disk after emptying the sink around it.
    """
    import os

    from chess_anti_engine.server.app import _retention_entries, prune_retained_dirs

    q = tmp_path / "invalid"
    q.mkdir()
    _mk(q, "good.tar", size=100, mtime=2000.0, sidecar=True)
    orphan = q / "gone.tar.reason.txt"
    orphan.write_bytes(b"x" * 5000)
    os.utime(orphan, (1000.0, 1000.0))

    accounted = {e.path.name: e.size for e in _retention_entries(q)}
    assert accounted == {"gone.tar.reason.txt": 5000, "good.tar": 115}, accounted

    freed_entries, freed_bytes = prune_retained_dirs(
        [q], max_bytes=1, max_entries=0, log=None,
    )

    assert not orphan.exists(), (
        "the orphaned .reason.txt survived a sweep that emptied the sink "
        "around it -- nothing counts it and nothing can delete it"
    )
    assert (freed_entries, freed_bytes) == (2, 5115), (freed_entries, freed_bytes)
    assert list(q.iterdir()) == [], sorted(p.name for p in q.iterdir())


# ---------------------------------------------------------------------------
# #419 F5: the walk sized files that were still being written, and counted
# another writer's atomic_write temp as a retained entry.
# ---------------------------------------------------------------------------


def _real_atomic_tmp(directory, final_name: str) -> str:
    """Create a REAL `atomic_write` temp in `directory` and leave it there.

    ⚑ GENERATED, NOT SPELLED OUT. The whole F5 trap is that the tmp name is
    easy to get wrong from memory -- `is_tmp_shard_name` matches `tmp_`/
    `._tmp_` PREFIXES and would not match this suffix-style name at all -- so
    the fixture makes `atomic_write` produce the name and captures it, rather
    than asserting against a hand-written string.
    """
    from chess_anti_engine.utils.atomic import atomic_write

    captured: list[str] = []

    def _writer(tmp) -> None:
        captured.append(tmp.name)
        tmp.write_bytes(b"y" * 4096)

    atomic_write(directory / final_name, _writer, durable=False)
    name = captured[0]
    # Keep only the tmp: the directory is now in the state a walk finds it in
    # while ANOTHER writer is mid-`atomic_write`.
    (directory / final_name).unlink()
    (directory / name).write_bytes(b"y" * 4096)
    return name


def test_a_real_atomic_write_temp_is_not_matched_by_is_tmp_shard_name() -> None:
    """⚑ The false fix, pinned. `is_tmp_shard_name` is the sibling helper any
    reader reaches for, and it matches NEITHER shape `_tmp_path_for` produces.
    If someone swaps `is_atomic_tmp_name` for it, this fails."""
    from chess_anti_engine.replay.shard import is_tmp_shard_name
    from chess_anti_engine.utils.atomic import is_atomic_tmp_name

    for name in (
        "deadbeef.json.tmp.4242.2553220387414c0fbddd018ba12597d9",
        "x.tmp.4242.5d5cba078f894887b8961da0ef459772.npz",
        # A shorter token, because the matcher deliberately does not pin
        # `uuid4().hex`'s width -- see `_ATOMIC_TMP_RE`.
        "9999_deadbeef.json.tmp.4242.abcdef0123456789",
    ):
        assert is_atomic_tmp_name(name), name
        assert not is_tmp_shard_name(name), name
    # And it does not swallow the real entry names these sinks hold.
    for name in ("deadbeef.json", "good.tar", "good.tar.reason.txt",
                 "tmp_4242_abcdef0123456789.tar", "1770000000_abcd1234.json"):
        assert not is_atomic_tmp_name(name), name


def test_the_walk_skips_a_real_atomic_write_temp(tmp_path) -> None:
    """⚑ REGRESSION (#419 F5, second half). The walk had no temp filter, so an
    `atomic_write` tmp -- the shape the `client_reports` sink writes -- was
    counted as a retained entry, sized while its bytes were still arriving, and
    eligible for eviction. Evicting it makes the writer's `os.replace` raise
    FileNotFoundError.

    Measured on `main`: the walk saw 6 entries where 5 exist, and a sweep at
    `max_entries=5` deleted... the oldest real entry, having counted the tmp.
    """
    from chess_anti_engine.server.app import _retention_entries, prune_retained_dirs

    q = tmp_path / "client_reports"
    q.mkdir()
    for i in range(5):
        _mk(q, f"e{i}.json", size=100, mtime=2000.0 + i, sidecar=False)
    tmp_name = _real_atomic_tmp(q, "9999_deadbeef.json")

    names = sorted(e.path.name for e in _retention_entries(q))

    assert tmp_name not in names, (
        f"the walk counted another writer's in-flight temp {tmp_name!r} as a "
        "retained entry"
    )
    assert names == ["e0.json", "e1.json", "e2.json", "e3.json", "e4.json"], names

    # And the sweep cannot delete it out from under the writer's os.replace.
    prune_retained_dirs([q], max_bytes=1, max_entries=0, log=None)
    assert (q / tmp_name).is_file(), (
        "the sweep deleted an in-flight atomic_write temp; the writer's "
        "os.replace would raise FileNotFoundError"
    )


def test_the_walk_never_sizes_a_partly_written_arena_result(tmp_path) -> None:
    """⚑ REGRESSION (#419 F5). `write_arena_result` wrote straight to the final
    `<sha>.json`, so the retention walk -- which runs on every arena upload --
    sized a file whose bytes were still arriving. Those are the bytes the
    ceiling is computed from; the direction is under-eviction.

    Measured on `main` with this fixture: short sizes observed mid-write,
    including 0. On this branch the walk sees the file either absent or whole,
    because the write lands by rename.

    ⚑ CONCURRENCY TEST, SO IT CARRIES ITS SERIAL CONTROL: the same write with
    no observer running must produce exactly the same final bytes, or a
    "no partial sizes" pass could just mean the write never happened. And the
    observer is bounded by wall clock -- the repo has no `pytest-timeout`, and
    a hang here would be a hung suite rather than a failed test.
    """
    import threading
    import time

    from chess_anti_engine.server.app import _retention_entries, write_arena_result

    body = b"z" * (24 * 1024 * 1024)

    # --- SERIAL CONTROL: no observer, nothing concurrent.
    serial = tmp_path / "serial"
    serial.mkdir()
    write_arena_result(serial / "deadbeef.json", body)
    assert (serial / "deadbeef.json").read_bytes() == body
    assert sorted(p.name for p in serial.iterdir()) == ["deadbeef.json"], (
        "the atomic write left its temp behind"
    )

    # --- CONCURRENT: walk while the write is in flight.
    live = tmp_path / "live"
    live.mkdir()
    out = live / "deadbeef.json"
    sizes: list[int] = []
    done = threading.Event()

    def _write() -> None:
        try:
            write_arena_result(out, body)
        finally:
            done.set()

    t = threading.Thread(target=_write, daemon=True)
    deadline = time.monotonic() + 60.0
    t.start()
    observations = 0
    while not done.is_set() and time.monotonic() < deadline:
        observations += 1
        sizes.extend(e.size for e in _retention_entries(live))
    t.join(max(0.0, deadline - time.monotonic()))
    assert not t.is_alive(), "the arena write did not finish within 60s"

    assert out.read_bytes() == body, "the concurrent write did not land intact"
  # ⚑ POWER CHECK, and it counts WALKS rather than sizes: once the write is
  # atomic the walk correctly sees no entry at all for most of the window, so
  # "few sizes" is the fix working and "few walks" is the test being blind.
    assert observations >= 20, (
        f"only {observations} walks completed during the write -- too few for "
        "this test to have had any power to see a partial size"
    )
    partial = sorted({s for s in sizes if s != len(body)})
    assert partial == [], (
        f"the walk sized a partly-written arena result: {partial[:6]} against "
        f"a final {len(body)} bytes -- these are the bytes the retention "
        "ceiling is computed from"
    )


# ---------------------------------------------------------------------------
# #419 review follow-ups (F-A / F-B / F-C).
# ---------------------------------------------------------------------------


class _LogSpy(logging.Handler):
    """Captures (levelname, message) off a REAL logger.

    ⚑ A real `logging.Logger` with a handler attached, not a stub with a
    `.warning` attribute: the production code calls `log.warning(fmt, *args)`
    and a stub that stores the arguments would pass even if the format string
    and the arguments disagreed. This renders the record, so the message has to
    actually come out.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelname, record.getMessage()))


def _spy_logger(name: str) -> tuple[logging.Logger, _LogSpy]:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers = []
    spy = _LogSpy()
    log.addHandler(spy)
    return log, spy


def _unlinkable_sidecar(bucket, entry_name: str, *, mtime: float):
    """A `.reason.txt` whose `unlink()` fails, without needing privileges.

    A DIRECTORY: `Path.unlink()` on one raises `IsADirectoryError` (EISDIR),
    which is an `OSError` and lands on exactly the branch a real EACCES/EIO
    would. The mechanism is synthetic; the branch is the production one.
    """
    import os

    sd = bucket / f"{entry_name}.reason.txt"
    sd.mkdir()
    (sd / "inner").write_bytes(b"y" * 50)
    os.utime(sd, (mtime, mtime))
    return sd


def test_a_stuck_sidecar_does_not_credit_the_entry_count(tmp_path) -> None:
    """⚑ REGRESSION AGAINST THIS PR'S OWN F1 COMMIT (#419 F-A, from review).

    When the shard unlinks but its `.reason.txt` does not, the BYTES were
    handled correctly -- `gone_bytes` excludes the sidecar -- while
    `live_count[victim] -= 1` and `gone_entries += 1` still ran. So the sweep
    ended over the ENTRY ceiling: post-F4 the orphaned sidecar is an entry in
    its own right, so one entry left and one entry arrived, and the count did
    not go down.

    ⚑ THIS TEST PASSES ON `origin/main` AND FAILS ON THIS PR'S F1 COMMIT. It is
    not a defect of the base -- `main`'s outer `except OSError: continue`
    happened to skip the whole entry, so it was accidentally safe here. This
    pins a regression the fix introduced, which is why the direction is
    reported the other way round from every other test in this file.

    Measured before the fix: report (1, 100), next walk counts 3 entries
    against `max_entries=2`, and NO warning -- silent.
    """
    import os

    from chess_anti_engine.server.app import _retention_entries, prune_retained_dirs

    u = tmp_path / "invalid" / "alice"
    u.mkdir(parents=True)
    for i, mt in ((0, 1000.0), (1, 2000.0), (2, 3000.0)):
        p = u / f"e{i}.tar"
        p.write_bytes(b"x" * 100)
        os.utime(p, (mt, mt))
    _unlinkable_sidecar(u, "e0.tar", mtime=1000.0)

    assert len(_retention_entries(u)) == 3
    log, spy = _spy_logger("test_retention_stuck_sidecar")

    prune_retained_dirs([u], max_bytes=0, max_entries=2, log=log)

    after = len(_retention_entries(u))
    assert after <= 2, (
        f"the sweep left {after} entries against max_entries=2 -- it credited "
        "itself an entry whose .reason.txt is still on disk, and post-F4 that "
        "orphan is an entry"
    )
    assert [m for lvl, m in spy.records if lvl == "WARNING"], (
        "the sweep ended over the entry ceiling and said nothing"
    )


def test_same_dir_sees_one_directory_reached_by_two_names(tmp_path) -> None:
    """⚑ PIN (#419 F-B, from review). `_same_dir` consults `st_ino`/`st_dev`
    BEFORE falling back to comparing resolved names, and the two disagree on a
    real deployment shape: under a bind mount, two paths reach one directory
    with no symlink to resolve, so `samefile` is True while the resolved
    compare is False.

    That makes "drop `samefile`, keep the resolved compare" a COVERAGE GAP, not
    an equivalent mutant -- dropping it re-opens F2 in full under a
    bind-mounted server root, which is exactly how a data-root move between
    drives gets done.

    ⚑ A bind mount needs privileges, so the fixture uses a HARD LINK: the same
    "one inode, two names, no symlink to resolve" shape, buildable as an
    ordinary user. The `samefile` result is what the guard consumes, and it is
    the half a resolved-name compare cannot reproduce.
    """
    from chess_anti_engine.server.app import _same_dir

    real = tmp_path / "real"
    real.write_bytes(b"x")
    alias = tmp_path / "alias"
    os.link(real, alias)

    # The premise, asserted rather than assumed: one inode, two names, and the
    # resolved-name compare cannot see it.
    assert real.samefile(alias), "fixture failed: the hard link is not the same inode"
    assert real.resolve() != alias.resolve(), "fixture failed: the names resolve equal"

    assert _same_dir(real, alias), (
        "_same_dir missed one directory reached by two names -- it is comparing "
        "resolved NAMES, which a bind mount defeats"
    )

    # Negative control: two genuinely different objects must stay different.
    other = tmp_path / "other"
    other.write_bytes(b"x")
    assert not _same_dir(real, other)
