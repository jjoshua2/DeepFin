"""#416: the boot-time compaction commit witness must not accept a partial write.

``_try_flush_and_pop`` moves the contributing ``_pending/*.zarr`` into
``_in_flight/<token>/`` -- which is then the ONLY durable copy -- and writes the
merged shard afterwards. ``save_local_shard_arrays`` is atomic: it writes
``._tmp_<pid>_<hex>_<final name>`` and commits with ``tmp.rename(final)``. The
temp name is therefore a PREFIXED COPY of the final name and ends with the same
``_<token>.zarr`` suffix the commit witness matches on.

⚑ THE STATE THAT MATTERS IS "CRASHED DURING THE MERGE WRITE", NOT "RESTARTED".
A test that restarts after a clean flush passes against the bug, and so does one
that crashes BEFORE the merge write -- both are included here precisely because
they PASS either way: they isolate the defect to the name match rather than to
the recovery machinery. Only the middle state, where an abandoned ``._tmp_*``
exists and the real shard does not, distinguishes fixed from broken.

The crash is a real one: a child process is spawned and calls ``os._exit(9)``
from inside ``zarr.open_group``, which is what a SIGKILL leaves behind. Nothing
is killed by name, and every child is bounded by a wall-clock timeout so a wedge
fails the test instead of hanging it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from chess_anti_engine.replay.shard import LOCAL_SHARD_SUFFIX, load_shard_arrays

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Wall-clock bound for every spawned child. Generous relative to the ~5s the
# upload actually takes, but finite: a hung child must FAIL the test, because a
# gate that hangs is a gate that cannot fail.
_CHILD_TIMEOUT_S = 180.0

_UPLOAD_DRIVER = '''
"""Drive one real upload that triggers a compaction flush, optionally dying
mid-write like a SIGKILL. argv: <server_root> <when: clean|pre|mid> <repo_root>
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[3])

import numpy as np

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    pack_shard_for_upload,
    samples_to_arrays,
    save_local_shard_arrays,
)


def sample(i):
    pol = np.zeros(4672, dtype=np.float32)
    pol[i % 4672] = 1.0
    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32), policy_target=pol, wdl_target=1,
    )


def seed_user(server_root):
    from chess_anti_engine.server.auth import UserRecord, hash_password, save_users

    salt, hsh, iters = hash_password("p")
    save_users(
        server_root / "users.json",
        {"u": UserRecord(username="u", salt_b64=salt, hash_b64=hsh, iterations=iters)},
    )


def build_tar(stage, samples):
    stage.mkdir(parents=True, exist_ok=True)
    zp = stage / "valid.zarr"
    save_local_shard_arrays(
        zp,
        arrs=samples_to_arrays(samples),
        meta=ShardMeta(
            username="u", games=1, positions=len(samples),
            model_sha256="aaaa1111", model_step=0,
        ),
    )
    _name, buf = pack_shard_for_upload(zp)
    return buf.getvalue()


def arm_crash(when):
    """Die like a SIGKILL inside the compacted-shard write."""
    import chess_anti_engine.replay.shard as shard_mod

    real_open = shard_mod.zarr.open_group

    class Shim:
        """Lets the tmp dir get created (.zgroup/.zattrs land on disk), then
        dies before any dataset is written -- a partial zarr."""

        def __init__(self, inner):
            self.attrs = inner.attrs

        def create_dataset(self, *_a, **_k):
            os._exit(9)

    def patched(path, mode="w", **kw):
        if "_compacted" in str(path):
            if when == "pre":
                os._exit(9)  # nothing written yet
            group = real_open(path, mode=mode, **kw)
            if when == "mid":
                return Shim(group)
            return group
        return real_open(path, mode=mode, **kw)

    shard_mod.zarr.open_group = patched


def main():
    root = Path(sys.argv[1])
    when = sys.argv[2]
    root.mkdir(parents=True, exist_ok=True)
    seed_user(root)
    if when in ("pre", "mid"):
        arm_crash(when)

    from fastapi.testclient import TestClient

    from chess_anti_engine.server.app import create_app

    app = create_app(
        server_root=str(root), users_db="users.json", upload_compact_shard_size=1,
    )
    payload = build_tar(root.parent / "stage", [sample(0), sample(1)])
    with TestClient(app) as client:
        resp = client.post(
            "/v1/upload_shard",
            auth=("u", "p"),
            files={"file": ("s.zarr.tar", payload, "application/x-tar")},
            headers={"X-CAE-Worker-Version": "0.0.0", "X-CAE-Protocol-Version": "1"},
        )
        print("UPLOAD_STATUS", resp.status_code)


main()
'''


def _run_upload(tmp_path: Path, root: Path, when: str) -> subprocess.CompletedProcess[str]:
    """Spawn a child that uploads (and for pre/mid, crashes mid-flush)."""
    driver = tmp_path / "upload_driver.py"
    driver.write_text(_UPLOAD_DRIVER, encoding="utf-8")
    try:
        return subprocess.run(
            [sys.executable, str(driver), str(root), when, str(_REPO_ROOT)],
            capture_output=True, text=True, timeout=_CHILD_TIMEOUT_S, check=False,
            cwd=str(_REPO_ROOT),
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - wedge guard
        pytest.fail(f"upload child ({when}) did not finish in {_CHILD_TIMEOUT_S}s: {exc}")


def _reboot(root: Path) -> None:
    """A fresh ``create_app`` over the same root, which runs startup recovery."""
    from chess_anti_engine.server.app import create_app

    create_app(server_root=str(root), users_db="users.json", upload_compact_shard_size=1)


def _surviving_positions(root: Path) -> int:
    """Positions readable from ANY loadable shard anywhere under the inbox.

    Deliberately not "is the compacted shard there" -- the question is whether
    the samples the worker was told ``stored: true`` for still exist at all, and
    a partial zarr that raises on load is not a copy of anything.
    """
    total = 0
    for path in sorted((root / "inbox").rglob(f"*{LOCAL_SHARD_SUFFIX}")):
        try:
            arrs, _meta = load_shard_arrays(path)
        except Exception:
            continue
        total += int(arrs["x"].shape[0])
    return total


def _compacted_names(root: Path) -> list[str]:
    compacted = root / "inbox" / "_compacted"
    return sorted(p.name for p in compacted.iterdir()) if compacted.is_dir() else []


def test_clean_flush_then_restart_keeps_the_compacted_shard(tmp_path: Path) -> None:
    """Control A: no crash. Passes with the bug present -- that is the point."""
    root = tmp_path / "srv_clean"
    proc = _run_upload(tmp_path, root, "clean")
    assert proc.returncode == 0, proc.stderr[-2000:]

    _reboot(root)

    assert _surviving_positions(root) == 2
    names = _compacted_names(root)
    assert len(names) == 1, names
    assert not names[0].startswith("._tmp_"), names


def test_crash_before_merge_write_restores_inputs_to_pending(tmp_path: Path) -> None:
    """Control B: crash BEFORE the merge write. Also passes with the bug present.

    Together with the clean control this proves the recovery machinery itself is
    sound, which is what isolates the defect below to the commit witness.
    """
    root = tmp_path / "srv_pre"
    proc = _run_upload(tmp_path, root, "pre")
    assert proc.returncode == 9, f"expected the armed crash, got {proc.returncode}: {proc.stderr[-2000:]}"
    assert not _compacted_names(root), "nothing should have been written yet"

    _reboot(root)

    assert _surviving_positions(root) == 2
    pending = root / "inbox" / "_pending"
    assert pending.is_dir(), "the _pending dir should exist"
    assert any(pending.iterdir()), "inputs should be back in _pending"


def test_crash_during_merge_write_does_not_delete_the_only_copy(tmp_path: Path) -> None:
    """#416: the state that matters. FAILS on origin/main (0 positions survive).

    The crash leaves ``_compacted/._tmp_<pid>_<hex>_..._<token>.zarr``, whose
    name ends with the same ``_<token>.zarr`` the witness matches on. Reading it
    as proof of commit makes recovery delete ``_in_flight/<token>/`` -- the only
    durable copy of samples the worker was already acknowledged for.
    """
    root = tmp_path / "srv_mid"
    proc = _run_upload(tmp_path, root, "mid")
    assert proc.returncode == 9, f"expected the armed crash, got {proc.returncode}: {proc.stderr[-2000:]}"

    # Precondition: we really are in the crash-during-write state.
    before = _compacted_names(root)
    assert before, "_compacted should hold the abandoned temp"
    assert all(n.startswith("._tmp_") for n in before), (
        f"expected only an abandoned temp in _compacted, got {before}"
    )
    in_flight = root / "inbox" / "_in_flight"
    assert in_flight.is_dir(), "the _in_flight dir must exist"
    assert any(in_flight.iterdir()), "inputs must be in _in_flight"

    _reboot(root)

    assert _surviving_positions(root) == 2, (
        "recovery destroyed the only copy of the merged inputs; "
        f"_compacted now holds {_compacted_names(root)}"
    )


def test_an_in_flight_temp_is_not_restored_into_pending(tmp_path: Path) -> None:
    """A partial write staged under `_in_flight/` must not become `_pending` litter.

    The restore loop only checked the `.zarr` suffix, which a `._tmp_*` name
    also has. `_scan_pending_dir` then filters it forever, so it would sit in
    `_pending` unread and unreclaimed -- the same "nothing ever cleans this up"
    complaint as the abandoned merge temp, one directory over.
    """
    root = tmp_path / "srv_inflight"
    proc = _run_upload(tmp_path, root, "pre")
    assert proc.returncode == 9, proc.stderr[-2000:]

    in_flight = root / "inbox" / "_in_flight"
    token_dir = next(d for d in in_flight.iterdir() if d.is_dir())
    partial = token_dir / f"._tmp_999_deadbeef_1_{'a' * 64}_tok{LOCAL_SHARD_SUFFIX}"
    partial.mkdir()
    (partial / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")

    _reboot(root)

    pending = root / "inbox" / "_pending"
    restored = sorted(p.name for p in pending.iterdir()) if pending.is_dir() else []
    assert not any(n.startswith("._tmp_") for n in restored), restored
    assert restored, "the real shard should still have been restored"


def test_crash_during_merge_write_cleans_up_the_abandoned_temp(tmp_path: Path) -> None:
    """The partial zarr is litter nothing else ever removes -- recovery drops it.

    It cannot be finished by anyone: this token's inputs are re-seeded under a
    NEW flush token, so leaving it means a permanent ``_compacted/`` entry that
    loads with ``KeyError``.
    """
    root = tmp_path / "srv_mid_litter"
    proc = _run_upload(tmp_path, root, "mid")
    assert proc.returncode == 9, proc.stderr[-2000:]
    assert any(n.startswith("._tmp_") for n in _compacted_names(root))

    _reboot(root)

    leftovers = [n for n in _compacted_names(root) if n.startswith("._tmp_")]
    assert not leftovers, f"abandoned compaction temp left behind: {leftovers}"
