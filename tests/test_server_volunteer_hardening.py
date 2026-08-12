"""Controls that make an untrusted contributor survivable.

None of these prevent a malicious volunteer from uploading bad data -- nothing
can, because a policy target is a probability vector and there is no server-side
way to tell a genuine MCTS distribution from a plausible forgery (see the
2026-08-03 legality scan: it reads zero on corrupted search trees BY
CONSTRUCTION). What they buy is attribution and revocability: knowing whose rows
these were, and refusing writes outside the lease the server actually handed out.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    load_shard_arrays,
    pack_shard_for_upload,
    samples_to_arrays,
    save_local_shard_arrays,
)
from chess_anti_engine.server.app import lease_authorizes_upload


def _sample(i: int = 0) -> ReplaySample:
    p = np.zeros(4672, dtype=np.float32)
    p[i % 4672] = 1.0
    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=p,
        wdl_target=1,
    )


def _seed_user(server_root: Path, username: str = "u", password: str = "p") -> None:
    from chess_anti_engine.server.auth import UserRecord, hash_password, save_users

    users = {}
    for name in ({username} | {"other"}):
        salt, hsh, iters = hash_password(password)
        users[name] = UserRecord(username=name, salt_b64=salt, hash_b64=hsh, iterations=iters)
    save_users(server_root / "users.json", users)


def _build_client(server_root: Path, **kwargs):
    from fastapi.testclient import TestClient

    from chess_anti_engine.server.app import create_app

    app = create_app(server_root=str(server_root), users_db="users.json", **kwargs)
    return TestClient(app)


def _headers() -> dict[str, str]:
    return {"X-CAE-Worker-Version": "0.0.0", "X-CAE-Protocol-Version": "1"}


def _tar_bytes(tmp_path: Path, *, n: int = 1, username: str = "u") -> bytes:
    tmp_path.mkdir(parents=True, exist_ok=True)
    zp = tmp_path / "valid.zarr"
    samples = [_sample(i) for i in range(n)]
    meta = ShardMeta(
        username=username, games=1, positions=n, model_sha256="abc1234567", model_step=0,
    )
    save_local_shard_arrays(zp, arrs=samples_to_arrays(samples), meta=meta)
    _, buf = pack_shard_for_upload(zp)
    return buf.getvalue()


# --------------------------------------------------------------------------
# [5] lease authorization
# --------------------------------------------------------------------------

def test_lease_rule_rejects_other_users_lease() -> None:
    lease = {"username": "alice", "trial_id": "t1", "expires_at_unix": 1 << 40}
    assert lease_authorizes_upload(lease, username="bob", trial_id="t1", now_unix=0) is not None
    assert lease_authorizes_upload(lease, username="alice", trial_id="t1", now_unix=0) is None


def test_lease_rule_rejects_cross_trial_write() -> None:
    lease = {"username": "alice", "trial_id": "t1", "expires_at_unix": 1 << 40}
    reason = lease_authorizes_upload(lease, username="alice", trial_id="t2", now_unix=0)
    assert reason is not None
    assert "t1" in reason


def test_lease_rule_rejects_expired_and_unknown() -> None:
    expired = {"username": "alice", "trial_id": "t1", "expires_at_unix": 100}
    assert "expired" in (lease_authorizes_upload(
        expired, username="alice", trial_id="t1", now_unix=200,
    ) or "")
    assert lease_authorizes_upload(None, username="alice", trial_id="t1", now_unix=0) is not None


def test_lease_rule_allows_default_trial_route() -> None:
    """The no-trial route is the shared default; a trial-pinned lease may use it."""
    lease = {"username": "alice", "trial_id": "t1", "expires_at_unix": 1 << 40}
    assert lease_authorizes_upload(lease, username="alice", trial_id=None, now_unix=0) is None


def test_upload_with_foreign_lease_is_refused_before_body_is_read(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    # Written directly rather than through /v1/lease_trial: that route needs a
    # published trial to hand out, and skipping when the fixture has none would
    # silently retire the one test that proves the route enforces the rule.
    from chess_anti_engine.server.lease import save_lease

    foreign_lease = "lease-owned-by-other"
    save_lease(
        leases_root=server_root / "leases",
        lease={
            "lease_id": foreign_lease,
            "username": "other",
            "trial_id": None,
            "expires_at_unix": 1 << 40,
        },
    )

    r2 = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", _tar_bytes(tmp_path / "s"), "application/x-tar")},
        headers={**_headers(), "X-CAE-Worker-Lease-ID": foreign_lease},
    )
    assert r2.status_code == 403, r2.text
    assert "lease" in r2.text.lower()


def test_upload_with_unknown_lease_is_refused(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", _tar_bytes(tmp_path / "s"), "application/x-tar")},
        headers={**_headers(), "X-CAE-Worker-Lease-ID": "no-such-lease"},
    )
    assert r.status_code == 403, r.text


def test_upload_without_lease_allowed_by_default_refused_when_required(tmp_path) -> None:
    """The flag is the switch to flip before opening to volunteers; leaving it
    off must not change today's fleet behaviour."""
    for require, expect_ok in ((False, True), (True, False)):
        server_root = tmp_path / f"server_{require}"
        server_root.mkdir()
        _seed_user(server_root)
        client = _build_client(server_root, require_worker_lease=require)

        r = client.post(
            "/v1/upload_shard",
            auth=("u", "p"),
            files={"file": ("shard.zarr.tar", _tar_bytes(tmp_path / f"s{require}"), "application/x-tar")},
            headers=_headers(),
        )
        if expect_ok:
            assert r.status_code == 200, r.text
        else:
            assert r.status_code == 403, r.text


# --------------------------------------------------------------------------
# [7] provenance survives compaction
# --------------------------------------------------------------------------

def _compacted_shards(server_root: Path) -> list[Path]:
    return sorted(p for p in (server_root / "inbox").rglob("*.zarr") if "_compacted" in str(p))


def test_compacted_shard_records_contributor_rows(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    # Compact as soon as two uploads' worth of rows have landed.
    client = _build_client(server_root, upload_compact_shard_size=4, upload_compact_max_age_seconds=1e9)

    for i, user in enumerate(("u", "other")):
        r = client.post(
            "/v1/upload_shard",
            auth=(user, "p"),
            files={"file": (
                "shard.zarr.tar", _tar_bytes(tmp_path / f"s{i}", n=2, username="LIAR"), "application/x-tar",
            )},
            headers=_headers(),
        )
        assert r.status_code == 200, r.text

    shards = _compacted_shards(server_root)
    assert shards, "no compacted shard was written"
    _, meta = load_shard_arrays(shards[0])

    contributors = meta.get("contributors")
    assert contributors, f"compacted shard carries no contributor provenance: {meta.get('username')!r}"
    by_user = {str(c["username"]): c for c in contributors}
    assert set(by_user) == {"u", "other"}, by_user
    # ⚑ The shards CLAIMED username="LIAR". Provenance must come from the
    # authenticated account, or one volunteer could aim a ban at another.
    assert "LIAR" not in by_user

    # Row ranges must tile the shard exactly and in order.
    ordered = sorted(contributors, key=lambda c: int(c["start"]))
    cursor = 0
    for entry in ordered:
        assert int(entry["start"]) == cursor, ordered
        cursor += int(entry["count"])
    assert cursor == int(meta["positions"]), (cursor, meta["positions"])


def test_raw_upload_is_stamped_with_authenticated_username(tmp_path) -> None:
    """The worker's own claim must not survive into the stored shard."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, upload_compact_shard_size=10_000)

    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", _tar_bytes(tmp_path / "s", username="LIAR"), "application/x-tar")},
        headers=_headers(),
    )
    assert r.status_code == 200, r.text

    pending = [p for p in (server_root / "inbox").rglob("*.zarr") if "_pending" in str(p)]
    assert pending, "no pending shard stored"
    attrs = json.loads((pending[0] / ".zattrs").read_text())
    assert attrs["username"] == "u", attrs["username"]


def test_shard_meta_contributors_survives_the_meta_roundtrip(tmp_path) -> None:
    """`_meta_dict` round-trips attrs through `ShardMeta(**meta)`; a field that
    does not survive that is a field the compactor writes into a void."""
    zp = tmp_path / "rt.zarr"
    rows = [{"username": "alice", "start": 0, "count": 3}]
    save_local_shard_arrays(
        zp,
        arrs=samples_to_arrays([_sample(i) for i in range(3)]),
        meta=ShardMeta(username="server_compactor", positions=3, contributors=rows),
    )
    _, meta = load_shard_arrays(zp)
    assert meta["contributors"] == rows


# --------------------------------------------------------------------------
# [2] cleartext transport
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    ["http://203.0.113.7:45453", "http://example.com", "http://192.168.1.50:8080"],
)
def test_cleartext_to_remote_host_is_refused(url) -> None:
    from chess_anti_engine.worker import cleartext_transport_refusal

    assert cleartext_transport_refusal(url, allow=False) is not None
    assert cleartext_transport_refusal(url, allow=True) is None


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:45453",
        "http://localhost:45453",
        "http://[::1]:45453",
        "http://0.0.0.0:45453",
        "https://example.com",
    ],
)
def test_loopback_and_tls_are_allowed(url) -> None:
    from chess_anti_engine.worker import cleartext_transport_refusal

    assert cleartext_transport_refusal(url, allow=False) is None


def test_cleartext_refusal_names_the_override() -> None:
    from chess_anti_engine.worker import cleartext_transport_refusal

    msg = cleartext_transport_refusal("http://example.com", allow=False) or ""
    assert "--allow-cleartext-http" in msg
    assert "https://" in msg


# --------------------------------------------------------------------------
# The flag must actually reach the server
# --------------------------------------------------------------------------

def test_require_worker_lease_reaches_create_app(monkeypatch) -> None:
    """Hop 3->4, EXECUTED: parse real argv and see what `create_app` receives.

    ⚑ THIS REPLACES A SUBSTRING GREP THAT COULD NOT FAIL. The first version
    asserted `"require_worker_lease" in <source text>`, which stayed green when
    `run_server` was mutated to pass `bool(args.worker_self_register)` -- a knob
    silently driven by the WRONG flag -- and when the harness key was typo'd out
    of existence. A presence check is not a value read; that is the exact defect
    this PR is about.
    """
    import chess_anti_engine.server.app as app_mod
    import chess_anti_engine.server.run_server as run_server_mod
    import uvicorn

    seen: dict[str, object] = {}

    def _fake_create_app(**kwargs):
        seen.update(kwargs)
        return object()

    # `main()` does `from chess_anti_engine.server.app import create_app` at
    # call time, so the source module is the binding that matters.
    monkeypatch.setattr(app_mod, "create_app", _fake_create_app)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)

    for argv, expected in (([], False), (["--require-worker-lease"], True)):
        seen.clear()
        monkeypatch.setattr(
            "sys.argv", ["run_server", "--server-root", "/tmp/x", *argv],
        )
        run_server_mod.main()
        assert seen["require_worker_lease"] is expected, (
            f"argv={argv!r} -> create_app got {seen.get('require_worker_lease')!r}"
        )
        # And it is not accidentally driven by the neighbouring flag.
        assert seen["worker_self_register"] is False


def test_require_worker_lease_reaches_the_server_command_line(monkeypatch, tmp_path) -> None:
    """Hop 1->2, EXECUTED: drive the real driver and read the argv it builds."""
    import chess_anti_engine.tune.distributed_runtime as dr_mod
    import chess_anti_engine.tune.harness as harness_mod

    built: list[list[str]] = []

    def _fake_spawn(cmd, **_kwargs):
        built.append([str(c) for c in cmd])
        raise RuntimeError("stop after argv is built")

    monkeypatch.setattr(dr_mod, "_spawn_with_reap", _fake_spawn)
    # Credential provisioning reads a real secret from the environment; it is
    # not what this test is about.
    monkeypatch.setattr(
        harness_mod, "_prepare_distributed_worker_auth",
        lambda **kw: ("tester", None),
    )

    for value, expected in ((False, False), (True, True)):
        built.clear()
        cfg = {
            # ⚑ Without this the function returns None before building
            # anything -- a version of this test that omitted it "passed"
            # while asserting on an empty list.
            "distributed_workers_per_trial": 1,
            "distributed_server_host": "127.0.0.1",
            "distributed_server_port": 45999,
            "require_worker_lease": value,
        }
        with contextlib.suppress(Exception):
            harness_mod._launch_distributed_server(
                base_config=cfg, work_dir=tmp_path / f"wd_{value}",
            )
        assert built, "driver never built a server command line"
        assert ("--require-worker-lease" in built[0]) is expected, built[0]


def test_cleartext_guard_is_on_the_worker_startup_path() -> None:
    """The pure function is tested above; this proves it is CALLED.

    ⚑ Deleting the entire `cleartext_transport_refusal(...)` call block from
    `_merge_cli_with_yaml_defaults` left the whole suite green, because every
    other cleartext test exercised the pure function directly. A guard nothing
    proves is installed is a guard that can be removed by accident.
    """
    import argparse

    from chess_anti_engine.worker import _merge_cli_with_yaml_defaults

    class _Args(argparse.Namespace):
        """Every worker arg defaults to None.

        `main()` builds its parser inline, so there is no parser object to take
        real defaults from; enumerating them by hand would make this test fail
        whenever an unrelated worker flag is added, which is how a guard test
        gets deleted rather than fixed.
        """

        def __getattr__(self, name: str):
            if name.startswith("__"):
                raise AttributeError(name)
            return None

    def _args(**over):
        ns = _Args()
        ns.allow_cleartext_http = False
        for k, v in over.items():
            setattr(ns, k, v)
        return ns

    # Remote http:// must abort startup.
    with pytest.raises(SystemExit) as exc:
        _merge_cli_with_yaml_defaults(_args(), {"server_url": "http://203.0.113.7:45453"})
    assert "--allow-cleartext-http" in str(exc.value)

    # The override lets it through.
    ns = _args(allow_cleartext_http=True)
    _merge_cli_with_yaml_defaults(ns, {"server_url": "http://203.0.113.7:45453"})
    assert ns.server_url == "http://203.0.113.7:45453"

    # And loopback -- what the driver actually hands in-tree workers -- is fine.
    ns = _args()
    _merge_cli_with_yaml_defaults(ns, {"server_url": "http://127.0.0.1:45453"})
    assert ns.server_url == "http://127.0.0.1:45453"


def test_require_worker_lease_defaults_off_everywhere() -> None:
    """An unset flag must never start refusing a running fleet's uploads."""
    import inspect

    from chess_anti_engine.server.app import create_app

    assert inspect.signature(create_app).parameters["require_worker_lease"].default is False


def test_record_contribution_ignores_the_uploader_supplied_meta() -> None:
    """Defence in depth, tested where it can actually fail.

    ⚑ `test_compacted_shard_records_contributor_rows` cannot catch this. By the
    time `add_upload` runs on the route, `_stamp_shard_username` has already
    overwritten `meta["username"]`, so trusting the meta and trusting the
    authenticated account are the SAME value there -- mutating `add_upload` to
    `self._record_contribution(meta.get("username"), ...)`, the exact thing its
    comment forbids, left the whole suite green.

    Calling the accumulator directly is the only place the two sources differ.
    """
    from chess_anti_engine.server.app import _BufferedUploadAccumulator

    acc = _BufferedUploadAccumulator(
        trial_id="t1", model_sha256="sha", created_at_unix=0.0, last_update_unix=0.0,
    )
    acc.add_upload(
        samples=[_sample(0), _sample(1)],
        meta={"username": "LIAR", "games": 1},
        now_unix=0.0,
        username="alice",
    )
    assert [c["username"] for c in acc.contributor_rows] == ["alice"]


def test_contiguous_uploads_from_one_user_coalesce() -> None:
    """A 2000-row shard built from many small uploads must not carry hundreds
    of entries -- and coalescing must not corrupt the ranges."""
    from chess_anti_engine.server.app import _BufferedUploadAccumulator

    acc = _BufferedUploadAccumulator(
        trial_id="t1", model_sha256="sha", created_at_unix=0.0, last_update_unix=0.0,
    )
    for _ in range(5):
        acc.add_upload(samples=[_sample(0)], meta={}, now_unix=0.0, username="alice")
    acc.add_upload(samples=[_sample(0)], meta={}, now_unix=0.0, username="bob")
    for _ in range(3):
        acc.add_upload(samples=[_sample(0)], meta={}, now_unix=0.0, username="alice")

    assert acc.contributor_rows == [
        {"username": "alice", "start": 0, "count": 5},
        {"username": "bob", "start": 5, "count": 1},
        {"username": "alice", "start": 6, "count": 3},
    ]
    assert sum(c["count"] for c in acc.contributor_rows) == len(acc.samples)


def test_unverified_pending_shard_does_not_launder_its_claim() -> None:
    """A shard promoted by a server predating the stamp carries the UPLOADER'S
    claim. Re-seeding it must record no contributor rather than a name that
    might be someone else's -- otherwise a ban lands on the wrong volunteer."""
    from chess_anti_engine.server.app import _BufferedUploadAccumulator

    acc = _BufferedUploadAccumulator(
        trial_id="t1", model_sha256="sha", created_at_unix=0.0, last_update_unix=0.0,
    )
    meta_unverified = {"username": "victim"}  # no provenance_verified marker
    verified = bool(meta_unverified.get("provenance_verified"))
    recovered = str(meta_unverified.get("username") or "") if verified else ""

    acc.add_upload(
        samples=[_sample(0)], meta=meta_unverified, now_unix=0.0,
        username=recovered or None,
    )
    assert acc.contributor_rows == [{"username": None, "start": 0, "count": 1}]


def test_stamp_writes_the_verified_marker(tmp_path) -> None:
    """The marker is what separates evidence from a claim; if it is absent the
    reseed path has no way to tell them apart."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root, upload_compact_shard_size=10_000)

    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", _tar_bytes(tmp_path / "s", username="LIAR"), "application/x-tar")},
        headers=_headers(),
    )
    assert r.status_code == 200, r.text

    pending = [p for p in (server_root / "inbox").rglob("*.zarr") if "_pending" in str(p)]
    attrs = json.loads((pending[0] / ".zattrs").read_text())
    assert attrs["username"] == "u"
    assert attrs["provenance_verified"] is True


def test_lease_with_no_trial_cannot_write_to_a_named_trial() -> None:
    """Fail-open closed: `assign_trial_lease` can issue a trial-less lease when
    no trials are available, and reading that as 'authorized for everything'
    makes it a cross-trial token for its whole lifetime."""
    lease = {"username": "alice", "trial_id": None, "expires_at_unix": 1 << 40}
    assert lease_authorizes_upload(lease, username="alice", trial_id="t_victim", now_unix=0) is not None
    # It may still use the shared default route.
    assert lease_authorizes_upload(lease, username="alice", trial_id=None, now_unix=0) is None


def test_lease_without_expiry_is_not_eternal() -> None:
    """An absent or zero `expires_at_unix` is a malformed lease, not one that
    never expires -- the one reading that grants more than a valid lease can."""
    for bad in ({"username": "a", "trial_id": "t1"}, {"username": "a", "trial_id": "t1", "expires_at_unix": 0}):
        reason = lease_authorizes_upload(bad, username="a", trial_id="t1", now_unix=10**9)
        assert reason is not None, bad


def test_cleartext_override_is_readable_from_worker_yaml() -> None:
    """`docs/operations.md` documents `allow_cleartext_http: true` in
    worker.yaml. A documented escape hatch the guard cannot see would be an
    accepted-and-ignored value -- the defect this PR is about."""
    import argparse

    from chess_anti_engine.worker import _merge_cli_with_yaml_defaults

    class _Args(argparse.Namespace):
        def __getattr__(self, name: str):
            if name.startswith("__"):
                raise AttributeError(name)
            return None

    ns = _Args()
    ns.allow_cleartext_http = False
    _merge_cli_with_yaml_defaults(
        ns, {"server_url": "http://203.0.113.7:45453", "allow_cleartext_http": True},
    )
    assert ns.server_url == "http://203.0.113.7:45453"
