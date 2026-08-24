"""``DiskReplayBuffer``'s prefetch thread must not outlive the test that made it.

The buffer starts a daemon prefetch thread the first time a sample triggers a
shuffle refresh. ``close()`` stops it; nothing else does. Before the reaping
fixture in ``tests/conftest.py``, a test that skipped ``close()`` handed the
thread to the whole remaining pytest session — measured on ``main``
(2026-08-24), ``pytest tests/test_replay_disk_buffer.py`` alone finished with
**17** live ``replay-prefetch-*`` threads.

That is not just untidy. Once pytest removes the owning ``tmp_path``, the
orphan's shard paths are still TRACKED in ``_shard_paths``, so
``_note_shard_load_failure`` calls them REAL faults rather than the benign trim
race and prints ``[disk_buf] WARNING: shuffle refresh failed to load a TRACKED
shard ...`` into whatever unrelated test is running at the time.

⚑ ``__del__`` cannot be the safety net. It calls ``close()``, but the thread's
target is the bound method ``self._prefetch_loop`` and a running thread is an
external GC root, so a buffer with a live prefetch thread stays reachable from
``threading._active``. Dropping the last reference and collecting three times
leaves the thread running. The finaliser only ever fires in the case that does
not matter — a buffer whose thread never started.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer

from tests.conftest import PREFETCH_THREAD_NAME_PREFIX, reaped_prefetch_buffer_count

# Set by the parent process on the child run below, where the leaky file has
# definitely been executed and the reap counter therefore MUST have moved.
_EXPECT_REAPED_ENV = "CAE_EXPECT_REAPED_PREFETCH"


def _sample() -> ReplaySample:
    policy = np.zeros((4672,), dtype=np.float32)
    policy[0] = 1.0
    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=0,
        priority=1.0,
        has_policy=True,
    )


def _live_prefetch_threads() -> list[str]:
    """Names of live prefetch threads.

    Filtered by the thread's own name rather than counted as a
    ``threading.enumerate()`` delta: the suite runs torch, Ray and requests
    pools, so an unfiltered count measures those instead and a "no new threads"
    assertion over that set is dominated by padding.
    """
    return sorted(
        t.name for t in threading.enumerate()
        if t.name.startswith(PREFETCH_THREAD_NAME_PREFIX) and t.is_alive()
    )


def _buffer_with_started_thread(shard_dir: Path) -> DiskReplayBuffer:
    """Build a buffer and get its prefetch thread genuinely running."""
    buf = DiskReplayBuffer(
        12,
        shard_dir=shard_dir,
        rng=np.random.default_rng(0),
        read_only=False,
        shuffle_cap=12,
        shard_size=2,
        refresh_interval=2,
        refresh_shards=1,
    )
    buf.add_many([_sample() for _ in range(6)])
    buf.flush()
    buf._schedule_refresh_prefetch()
    return buf


def test_close_stops_the_prefetch_thread(tmp_path: Path) -> None:
    """``close()`` must actually join the thread, not just orphan a daemon.

    The pre-assert is load-bearing: without it a buffer that never started a
    thread would satisfy the post-assert trivially, and the test would pass on a
    build where prefetching was disabled outright.
    """
    buf = _buffer_with_started_thread(tmp_path / "replay")
    t = buf._prefetch_thread
    assert t is not None, "the prefetch thread never started; the rest of this test is vacuous"
    assert t.is_alive()

    buf.close()

    assert not t.is_alive(), (
        "close() returned while the prefetch thread was still running — it is a "
        "daemon, so nothing else will ever stop it"
    )
    assert buf._prefetch_thread is None
    assert t.name not in _live_prefetch_threads()


def test_unclosed_buffer_is_reaped_by_teardown(tmp_path: Path) -> None:
    """Build a buffer the way the leaky tests do — and deliberately never close it.

    This test asserts only that the scenario is REAL (a thread is running when
    the body ends). Proving it gets reaped cannot happen here: the reaping
    fixture's teardown runs after this function returns. That half is
    ``test_leaky_replay_file_leaves_no_prefetch_thread_behind`` below, which
    watches a whole child pytest process instead.
    """
    buf = _buffer_with_started_thread(tmp_path / "replay")
    assert buf._prefetch_thread is not None
    assert buf._prefetch_thread.is_alive()
    # No close() — on purpose. The autouse fixture in conftest owns it now.


def test_no_prefetch_thread_is_alive() -> None:
    """No ``replay-prefetch-*`` thread may be alive while any test is running.

    A session-wide invariant, not a statement about this file: tests run one at
    a time, so any live prefetch thread here belongs to a test that already
    finished. Passed as the second node-id of the child run below, it is the
    assertion that fails on the unreaped behaviour.
    """
    leaked = _live_prefetch_threads()
    assert not leaked, (
        f"{len(leaked)} prefetch thread(s) outlived the test that created them and are "
        f"running now: {leaked}. They poll on a 0.1s tick and print [disk_buf] WARNING "
        f"lines into unrelated tests' stdout once their tmp_path is gone."
    )
    if os.environ.get(_EXPECT_REAPED_ENV) == "1":
        # Guards the emptiness above against being vacuous: on this run the
        # leaky file HAS executed, so the reaper must have had work to do. If
        # the fixture's wrapper ever stops matching `_ensure_prefetch_thread`,
        # `leaked` stays empty for the wrong reason and only this fires.
        assert reaped_prefetch_buffer_count() > 0, (
            "no buffer was reaped on a run that executed tests/test_replay_disk_buffer.py "
            "— the conftest fixture is no longer wrapping the thread-creation site, so "
            "the empty leak list above proves nothing"
        )


def test_leaky_replay_file_leaves_no_prefetch_thread_behind() -> None:
    """The real bug, end to end: leaky file first, another file after it.

    Run as a child pytest rather than asserted in-process because the leak is a
    CROSS-TEST effect and the reaping happens in a teardown this process cannot
    observe from inside a test. Node-ids are passed explicitly and pytest honours
    their command-line order, so the reproduction does not depend on collection
    order, on file naming, or on a shuffling plugin being absent.

    On the pre-fix behaviour the child's second node-id fails with 17 live
    threads; the parent surfaces the child's own message rather than a bare
    return code.
    """
    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env[_EXPECT_REAPED_ENV] = "1"
    env["PYTHONPATH"] = str(repo_root)
    probe = f"{Path(__file__).name}::test_no_prefetch_thread_is_alive"
    out = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/test_replay_disk_buffer.py",
            f"tests/{probe}",
            "-q", "-p", "no:cacheprovider",
        ],
        capture_output=True, text=True, cwd=repo_root, env=env, timeout=600, check=False,
    )
    assert out.returncode == 0, (
        "a pytest run of the leaky replay file followed by the prefetch-thread probe "
        f"failed (rc={out.returncode}).\n--- stdout ---\n{out.stdout}\n"
        f"--- stderr ---\n{out.stderr}"
    )


def test_prefetch_thread_keeps_its_buffer_alive_so_del_cannot_close_it(tmp_path: Path) -> None:
    """Pin WHY the reaping has to be explicit, so nobody deletes it as redundant.

    ``DiskReplayBuffer.__del__`` calls ``close()``, which reads as a safety net
    that makes the fixture unnecessary. It is not one: the thread's target is a
    bound method, so a live prefetch thread keeps its buffer strongly reachable
    and the finaliser never runs. This test would fail — and the fixture could
    then genuinely be dropped — if that ever changed.
    """
    import gc

    buf = _buffer_with_started_thread(tmp_path / "replay")
    t = buf._prefetch_thread
    assert t is not None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not t.is_alive():
        time.sleep(0.01)
    assert t.is_alive()

    del buf
    for _ in range(3):
        gc.collect()

    assert t.is_alive(), (
        "the prefetch thread stopped after its buffer was dropped and collected — "
        "__del__ can now fire, so re-examine whether the conftest reaping fixture "
        "is still needed"
    )
    # Leave it to the reaping fixture: with the only reference gone, this test
    # has no way to close it, which is exactly the leak being guarded.
