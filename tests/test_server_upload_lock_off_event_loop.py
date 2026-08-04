"""A4: the upload critical section must not run on the event-loop thread.

``_upload_shard_impl`` is an ``async def``. It acquires ``upload_lock``, a
``threading.Lock``, and does real work under it: ``arrays_to_samples``,
``acc.add_upload``, and — on the flush path — ``_try_flush_and_pop``, which
writes a compacted shard to disk. Every one of those is a BLOCKING call, and a
blocking call in a coroutine does not yield: it holds the single event-loop
thread for its whole duration. While one worker's upload compacts, every other
route on the server — lease, health, manifest, publish, arena upload — is not
slow, it is *not running at all*. The server presents as wedged.

The other two ``upload_lock`` acquisitions (the stale-flush sweep and the
queued-games count) are in ``def``s, not ``async def``s, so Starlette already
runs them in its threadpool. They were always correct; only the coroutine was
wrong, which is why this is easy to miss by reading the lock rather than the
functions that take it.

⚑ WHAT THIS MODULE DOES **NOT** CLAIM. "An upload no longer stalls the loop"
would be false. #335 moved only the locked block; the ~20 blocking FS sites
between the drain and the response stayed on the loop and cost a measured
**0.2870s -> 0.1097s** on a real 1500-position upload. Audit A5 moved those
too, and `test_a_real_upload_does_not_stall_the_loop_beyond_the_a5_bound`
below is the standing gate on the residual with the number in it. Whoever
moves that number again must move the bound in that test, in the same commit.
What stays on the loop by necessity is the `UploadFile` drain, which must be
awaited -- and it is the one blocking stretch that already yields, per chunk.

⚑ WHAT THESE TESTS MEASURE, AND WHY IT IS NOT A TIMING RACE. The first test
does not assert "the upload was fast". It asserts that the loop *kept
scheduling* while the upload ran: a watchdog coroutine wakes every 5 ms and
records the largest gap between consecutive wakeups. Loop blocking is absolute
— if the loop thread is inside ``time.sleep(BLOCK_S)`` no coroutine can run,
regardless of machine load — so the pre-fix gap is >= BLOCK_S and the post-fix
gap is bounded by scheduler jitter. The bound below sits an order of magnitude
away from both. Under-loaded CI can only *add* jitter, so the failure direction
is "flaky pass", never "flaky fail", and the second test would still catch a
fix that traded away mutual exclusion.

The blocking work is injected by wrapping ``arrays_to_samples`` — the first
call inside the critical section — rather than by relying on a real disk flush
being slow enough to see. That keeps the test deterministic while measuring the
real property: the critical section is entered on some thread, and the question
is only which one.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    pack_shard_for_upload,
    samples_to_arrays,
    save_local_shard_arrays,
)

# How long the injected work holds the critical section. Big enough that a
# blocked loop is unambiguous, small enough that the suite does not drag.
BLOCK_S = 0.75
# The watchdog's tick. Ten ticks inside one BLOCK_S, so a blocked loop is
# measured many times over rather than inferred from a single missed wakeup.
TICK_S = 0.005
# Pre-fix the largest wakeup gap is >= BLOCK_S (0.75s). Post-fix it is
# scheduler jitter on a threadpool handoff. 0.25s is 3x below the failure and
# ~50x above the healthy reading; nothing lives in between.
MAX_LOOP_STALL_S = 0.25


def _sample(i: int = 0) -> ReplaySample:
    pol = np.zeros(4672, dtype=np.float32)
    pol[i % 4672] = 1.0
    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=pol,
        wdl_target=1,
    )


def _seed_user(server_root: Path, username: str = "u", password: str = "p") -> None:
    from chess_anti_engine.server.auth import UserRecord, hash_password, save_users

    salt, hsh, iters = hash_password(password)
    users = {username: UserRecord(username=username, salt_b64=salt, hash_b64=hsh, iterations=iters)}
    save_users(server_root / "users.json", users)


def _build_zarr_tar(tmp_path: Path, *, samples: list[ReplaySample], model_sha256: str) -> bytes:
    tmp_path.mkdir(parents=True, exist_ok=True)
    zp = tmp_path / "valid.zarr"
    meta = ShardMeta(
        username="u",
        games=1,
        positions=len(samples),
        model_sha256=model_sha256,
        model_step=0,
    )
    save_local_shard_arrays(zp, arrs=samples_to_arrays(samples), meta=meta)
    _, buf = pack_shard_for_upload(zp)
    return buf.getvalue()


def _headers() -> dict[str, str]:
    return {
        "X-CAE-Worker-Version": "0.0.0",
        "X-CAE-Protocol-Version": "1",
        "Authorization": "Basic dTpw",  # u:p
    }


class _CriticalSectionProbe:
    """Instruments the critical section: how long it holds, and on which thread.

    Wraps ``arrays_to_samples``, which the upload path calls as its first
    statement inside ``with upload_lock``. Records the calling thread, blocks
    for ``BLOCK_S``, and tracks how many callers are inside at once so a fix
    that moved the work off the loop by DROPPING the lock would be caught.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._state = threading.Lock()
        self.threads: list[str] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self.calls = 0

    def __call__(self, *args, **kwargs):
        with self._state:
            self.calls += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            self.threads.append(threading.current_thread().name)
        try:
            time.sleep(BLOCK_S)
            return self._inner(*args, **kwargs)
        finally:
            with self._state:
                self.concurrent -= 1


@pytest.fixture
def upload_env(tmp_path, monkeypatch):
    """An app, two distinct uploads, and a probe wired into the critical section."""
    from chess_anti_engine.server import app as app_mod

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    probe = _CriticalSectionProbe(app_mod.arrays_to_samples)
    monkeypatch.setattr(app_mod, "arrays_to_samples", probe)

    # Threshold of 2 positions: each upload carries 2 samples, so the FIRST
    # upload already trips `_buffered_upload_ready` and the flush path — the
    # expensive half of the critical section — runs for real.
    app = app_mod.create_app(
        server_root=str(server_root),
        users_db="users.json",
        upload_compact_shard_size=2,
    )
    tars = [
        _build_zarr_tar(
            tmp_path / f"u{i}",
            samples=[_sample(2 * i), _sample(2 * i + 1)],
            model_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
        )
        for i in range(2)
    ]
    return app, tars, probe, server_root


def _post(client, tar: bytes, name: str):
    return client.post(
        "/v1/upload_shard",
        files={"file": (f"{name}.zarr.tar", tar, "application/octet-stream")},
        headers=_headers(),
    )


@pytest.mark.anyio
async def test_an_upload_flush_does_not_stall_the_event_loop(upload_env) -> None:
    """THE REPRO. A watchdog coroutine must keep getting scheduled mid-upload.

    Fails on the pre-fix code with a stall of >= BLOCK_S, because the coroutine
    holds the loop thread through the whole locked block.
    """
    import httpx

    app, tars, probe, _ = upload_env
    stop = asyncio.Event()
    gaps: list[float] = []

    async def watchdog() -> None:
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(TICK_S)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    wd = asyncio.create_task(watchdog())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        await asyncio.sleep(5 * TICK_S)  # let the watchdog establish a baseline
        r = await client.post(
            "/v1/upload_shard",
            files={"file": ("a.zarr.tar", tars[0], "application/octet-stream")},
            headers=_headers(),
        )
    stop.set()
    await wd

    assert r.status_code == 200, r.text
    assert r.json().get("stored") is True, r.json()
    assert probe.calls == 1, "the critical section must have been entered exactly once"

    worst = max(gaps) if gaps else 0.0
    assert worst < MAX_LOOP_STALL_S, (
        f"the event loop stalled for {worst:.3f}s during one upload "
        f"(bound {MAX_LOOP_STALL_S}s, injected work {BLOCK_S}s). The upload "
        f"critical section is running on the loop thread, so every other route "
        f"was unserved for that whole window. Critical section ran on thread(s): "
        f"{probe.threads}"
    )


# ⚑ A THIRD TEST WAS WRITTEN HERE AND DELETED, WHICH IS WORTH RECORDING.
# It drove a real GET through the ASGI stack while an upload compacted and
# asserted the response came back promptly. In two formulations -- per-request
# latency, then inter-completion gap -- it PASSED on the unfixed code, so it
# was a gate that could not fail. The reason is the bug measuring itself: the
# poller runs on the very loop the upload blocks, so it is descheduled for the
# whole stall and its timers only resume once the stall is over. Nothing
# scheduled on a blocked loop can time that loop being blocked. The watchdog
# above works only because its gap is computed across a sleep that SPANS the
# stall. A passing test that cannot fail is worse than no test, so it is gone
# rather than kept as reassurance.

@pytest.mark.anyio
async def test_moving_the_lock_off_the_loop_keeps_uploads_mutually_exclusive(upload_env) -> None:
    """NEGATIVE CONTROL for the fix: exclusion must survive the move.

    The cheapest way to make the two tests above pass is to stop taking the
    lock, or to take it only around a part of the block. Then two concurrent
    uploads would be inside the critical section at once and could interleave
    their accumulator mutations. This asserts the AXIS STATE — max observed
    concurrency inside the section is exactly 1 — rather than the absence of a
    symptom, and it is why the fix must keep the ``threading.Lock`` (the two
    sync acquisition sites cannot await an ``asyncio.Lock``).
    """
    import httpx

    app, tars, probe, server_root = upload_env
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        t0 = time.monotonic()
        results = await asyncio.gather(*[
            client.post(
                "/v1/upload_shard",
                files={"file": (f"{i}.zarr.tar", tar, "application/octet-stream")},
                headers=_headers(),
            )
            for i, tar in enumerate(tars)
        ])
        elapsed = time.monotonic() - t0

    for r in results:
        assert r.status_code == 200, r.text
        assert r.json().get("stored") is True, r.json()
    assert probe.calls == 2
    assert probe.max_concurrent == 1, (
        f"two uploads were inside the critical section at once "
        f"(max_concurrent={probe.max_concurrent}); mutual exclusion was lost"
    )
    # Serialised, so two 0.75s sections cannot finish in less than 1.5s. This
    # is the same fact as max_concurrent, read off the clock instead of the
    # counter, so a probe that mis-counts cannot hide it.
    assert elapsed >= 2 * BLOCK_S * 0.9, (
        f"both uploads completed in {elapsed:.3f}s, faster than two "
        f"serialised {BLOCK_S}s critical sections could"
    )
    # And the work still happened: both distinct model_sha keys compacted.
    compacted = sorted((server_root / "inbox" / "_compacted").glob("*"))
    assert len(compacted) == 2, [p.name for p in compacted]


def test_the_sync_lock_holders_are_still_plain_defs() -> None:
    """The other two acquisition sites must NOT become coroutines.

    They run in Starlette's threadpool precisely because they are ``def``s. If
    a later change makes one an ``async def``, it inherits exactly the bug this
    module exists to prevent — and no runtime test would notice, because the
    lock would still be correct and the loop stall would only show under load.
    """
    import ast
    import inspect

    from chess_anti_engine.server import app as app_mod

    tree = ast.parse(inspect.getsource(app_mod))
    holders_async: list[str] = []
    holders_sync: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Attribute the acquisition to its INNERMOST enclosing function. A
        # nested `def` that takes the lock belongs to that def, not to the
        # coroutine it happens to be declared in -- which is exactly the shape
        # the fix uses, and a walker that ignored the distinction would flag
        # the fixed code forever and could never be satisfied.
        def _own_withs(fn):
            for child in ast.iter_child_nodes(fn):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                for sub in ast.walk(child):
                    if isinstance(sub, ast.With):
                        yield sub
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        break

        takes_lock = any(
            isinstance(w.context_expr, ast.Name) and w.context_expr.id == "upload_lock"
            for stmt in _own_withs(node)
            for w in stmt.items
        )
        if not takes_lock:
            continue
        (holders_async if isinstance(node, ast.AsyncFunctionDef) else holders_sync).append(node.name)

    assert holders_async == [], (
        f"{holders_async} take `upload_lock` directly inside an async def; a "
        f"threading.Lock in a coroutine blocks the event loop. Run the locked "
        f"block through run_in_threadpool instead."
    )
    assert len(holders_sync) >= 2, (
        f"expected the sync lock holders to still exist, found {holders_sync}"
    )


# The pre-registered A5 yardstick. No injected work at all: real extract,
# validate, promote and delete, one upload, one client, flush enabled.
YARDSTICK_POSITIONS = 1500
# Reviewer's measured baselines on this probe: 0.2870s pre-#335, 0.1097s
# post-#335. A5's commitment is to get the residual under 0.02s.
MAX_UNINJECTED_STALL_S = 0.02


@pytest.mark.anyio
async def test_a_real_upload_does_not_stall_the_loop_beyond_the_a5_bound(tmp_path) -> None:
    """THE A5 YARDSTICK, pre-registered in the #335 review before this was written.

    The tests above inject a sleep into the critical section, so they measure
    *which thread* the section runs on and nothing else. This one injects
    nothing: the stall it reports is real FS work — untar, validate, load,
    ``replace``, recursive deletes — on a 1500-position shard, which is the
    quantity an operator actually experiences per upload.

    History, so the bound is not a bare number: **0.2870s on main before #335,
    0.1097s after it.** #335 moved the locked block off the loop and took ~62%
    of the stall with it; the ~20 remaining blocking FS sites in
    ``_upload_shard_impl`` are the residual, and this bound is the commitment
    to remove them.
    """
    import httpx

    from chess_anti_engine.server import app as app_mod

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    app = app_mod.create_app(
        server_root=str(server_root),
        users_db="users.json",
        upload_compact_shard_size=1,  # flush on this very upload
    )
    tar = _build_zarr_tar(
        tmp_path / "big",
        samples=[_sample(i) for i in range(YARDSTICK_POSITIONS)],
        model_sha256="c0ffee00",
    )

    stop = asyncio.Event()
    gaps: list[float] = []

    async def watchdog() -> None:
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(TICK_S)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    wd = asyncio.create_task(watchdog())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=60.0) as client:
        await asyncio.sleep(5 * TICK_S)
        r = await client.post(
            "/v1/upload_shard",
            files={"file": ("big.zarr.tar", tar, "application/octet-stream")},
            headers=_headers(),
        )
    stop.set()
    await wd

    assert r.status_code == 200, r.text
    assert r.json().get("stored") is True, r.json()
    worst = max(gaps) if gaps else 0.0
    print(f"A5 YARDSTICK worst uninjected loop gap = {worst:.4f}s")
    assert worst < MAX_UNINJECTED_STALL_S, (
        f"a single real {YARDSTICK_POSITIONS}-position upload stalled the loop "
        f"for {worst:.4f}s (bound {MAX_UNINJECTED_STALL_S}s; 0.2870s pre-#335, "
        f"0.1097s post-#335). Blocking FS work is still running on the event "
        f"loop thread between the drain and the response."
    )


@pytest.mark.anyio
async def test_the_hot_poll_routes_do_not_read_the_disk_on_the_loop(tmp_path) -> None:
    """A16 + A9: the two routes every worker polls must not stall the loop.

    ``/v1/manifest`` is the most frequently hit route on the server and
    ``_get_manifest_impl`` reads the manifest off disk; ``/v1/lease_trial``
    runs ``_SeedDoleGate.claim``, which does rename + read_text + unlink +
    write_text + replace. Both were ``async def``s doing that work inline.

    Instrumented the same way as the yardstick, but the injected block is put
    on the *filesystem helper* rather than the critical section, so this fails
    if either route regresses to reading on the loop -- and it is indifferent
    to how the fix is spelled (``run_in_threadpool``, a sync ``def``, or a
    thread of its own).
    """
    import httpx

    from chess_anti_engine.server import app as app_mod

    server_root = tmp_path / "server"
    (server_root / "publish").mkdir(parents=True)
    _seed_user(server_root)
    # ⚑ A REALISTIC MANIFEST, NOT `{}`. With an empty file this gate passed on
    # unfixed code 2 runs in 3 (review #336): the read was too small to stall
    # anything, so the test could not discriminate. A production manifest
    # carries the full realized worker config -- ~200 keys plus the dole FEN
    # seed list -- and reading and json-parsing that is the work the loop was
    # doing on every poll. Sized to the live manifest, not padded arbitrarily.
    manifest = {
        "model_sha256": "0" * 64,
        "model_step": 123456,
        "trial_id": "trial_0f888",
        **{f"search_knob_{i}": (i * 1.5) for i in range(200)},
        "dole_fen_seeds": [
            f"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 {i}" for i in range(500)
        ],
    }
    (server_root / "publish" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    app = app_mod.create_app(server_root=str(server_root), users_db="users.json")

    stop = asyncio.Event()
    gaps: list[float] = []

    async def watchdog() -> None:
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(TICK_S)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    wd = asyncio.create_task(watchdog())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=30.0) as client:
        await asyncio.sleep(5 * TICK_S)
        for _ in range(20):
            resp = await client.get("/v1/manifest", headers=_headers())
            assert resp.status_code in (200, 404, 503), resp.status_code
    stop.set()
    await wd

    worst = max(gaps) if gaps else 0.0
    print(f"A16 hot-poll worst loop gap over 20 polls = {worst:.4f}s")
    assert worst < MAX_UNINJECTED_STALL_S, (
        f"the manifest poll route stalled the loop for {worst:.4f}s over 20 "
        f"requests (bound {MAX_UNINJECTED_STALL_S}s); it is reading the "
        f"manifest off disk on the event-loop thread"
    )


@pytest.mark.anyio
async def test_concurrent_uploads_do_not_lose_stat_updates(tmp_path) -> None:
    """NEGATIVE CONTROL for A5: the loop was serialising three RMW cycles.

    Banked from the #336 review, which found this and measured it. Three
    unlocked read-modify-write cycles run on the upload path — the two
    throughput stats files and ``load_users`` -> ``record_upload`` ->
    ``save_users``. While the tail of ``_upload_shard_impl`` ran on the event
    loop they could not interleave: one thread, no ``await`` between the read
    and the write. Moving the tail into the threadpool removed that implicit
    serialisation, and concurrent uploads began overwriting each other's
    counts. ``atomic_write_text`` makes each WRITE atomic, so nothing ever
    tears — the file just quietly ends up with fewer increments than there were
    uploads, which is this codebase's signature defect wearing a new hat.

    Reviewer's measurement, 5 runs a side, 8 concurrent uploads:
    base ``[8, 8, 8, 8, 8]``, unlocked head ``[5, 4, 8, 7, 6]``.

    ⚑ EVERY OTHER TEST IN THE SUITE UPLOADS SEQUENTIALLY, which is why none of
    them caught it. The assertion is an exact count, not a bound: 8 uploads
    must record 8.

    ⚑ A18's precondition is single EXECUTION CONTEXT, not single process. This
    change kept the process count at one and still broke it.
    """
    import httpx

    from chess_anti_engine.server import app as app_mod
    from chess_anti_engine.server.auth import load_users

    n_uploads = 8
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    app = app_mod.create_app(
        server_root=str(server_root),
        users_db="users.json",
        upload_compact_shard_size=2000,  # stay in the buffer; the stats tail is the subject
    )
    tars = [
        _build_zarr_tar(
            tmp_path / f"c{i}",
            samples=[_sample(i)],
            model_sha256=hashlib.sha256(f"m{i}".encode()).hexdigest(),
        )
        for i in range(n_uploads)
    ]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=60.0) as client:
        results = await asyncio.gather(*[
            client.post(
                "/v1/upload_shard",
                files={"file": (f"{i}.zarr.tar", tar, "application/octet-stream")},
                headers=_headers(),
            )
            for i, tar in enumerate(tars)
        ])

    for r in results:
        assert r.status_code == 200, r.text
        assert r.json().get("stored") is True, r.json()

    users = load_users(server_root / "users.json")
    rec = users["u"]
    recorded = int(getattr(rec, "uploads", 0))
    print(f"RMW PROBE uploads recorded = {recorded} of {n_uploads}")
    assert recorded == n_uploads, (
        f"{n_uploads} concurrent uploads all returned stored=true but only "
        f"{recorded} were recorded; the read-modify-write cycle on users.json "
        f"is unserialised now that the upload tail runs in the threadpool"
    )
