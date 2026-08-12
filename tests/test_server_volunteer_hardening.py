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


def test_unverified_pending_shard_does_not_launder_its_claim(tmp_path) -> None:
    """A shard promoted by a server predating the stamp carries the UPLOADER'S
    claim. Re-seeding it must record no contributor rather than a name that
    might be someone else's -- otherwise a ban lands on the wrong volunteer.

    ⚑ ROUTE-LEVEL ON PURPOSE. The first version of this test recomputed the
    production expression in its own body::

        verified = bool(meta.get("provenance_verified"))
        recovered = str(meta.get("username") or "") if verified else ""

    ...and then asserted on the result. That asserts the test's arithmetic, not
    the server's: deleting `if verified else ""` from `_scan_pending_dir` left
    the whole suite GREEN. A test that restates the code it is checking is the
    same defect as a gate that cannot fail, and it was sitting in the fix for
    the finding about laundering.

    So this plants a genuinely unstamped pending shard on disk and boots the
    real app, which runs `_recover_pending_uploads` at startup.
    """
    from chess_anti_engine.replay.shard import PENDING_DIR_NAME

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    # A pre-upgrade pending shard: uploaded by mallory, CLAIMING to be victim,
    # with no `provenance_verified` marker because the old server never wrote one.
    pending = server_root / "inbox" / PENDING_DIR_NAME
    pending.mkdir(parents=True, exist_ok=True)
    zp = pending / "1000_deadbeef_aaaa.zarr"
    save_local_shard_arrays(
        zp,
        arrs=samples_to_arrays([_sample(i) for i in range(2)]),
        meta=ShardMeta(
            username="victim", games=1, positions=2,
            model_sha256="abc1234567", model_step=0,
        ),
    )
    # ⚑ Strip the key entirely rather than leaving it None. `asdict(ShardMeta)`
    # writes every field, so a shard written by TODAY's code carries
    # `provenance_verified: null`; a shard written by the OLD code has no such
    # key at all. Both must be treated as unverified, and the fixture has to be
    # the real pre-upgrade shape or it is not testing the upgrade case.
    attrs_path = zp / ".zattrs"
    attrs = json.loads(attrs_path.read_text())
    attrs.pop("provenance_verified", None)
    attrs_path.write_text(json.dumps(attrs, indent=4, sort_keys=True))
    assert "provenance_verified" not in json.loads(attrs_path.read_text())

    # Booting the app runs `_recover_pending_uploads`, which RE-SEEDS the
    # accumulator but does not flush it; a real upload pushes it over the
    # compaction threshold. The result is the mixed shard that makes the
    # failure legible: recovered rows beside a freshly authenticated one.
    client = _build_client(
        server_root, upload_compact_shard_size=3, upload_compact_max_age_seconds=1e9,
    )
    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", _tar_bytes(tmp_path / "fresh", n=1), "application/x-tar")},
        headers=_headers(),
    )
    assert r.status_code == 200, r.text

    shards = _compacted_shards(server_root)
    assert shards, "no compacted shard was produced"
    _, meta = load_shard_arrays(shards[0])
    names = [c["username"] for c in (meta.get("contributors") or [])]
    assert "victim" not in names, (
        f"an unverified claim was laundered into provenance: {names!r}"
    )
    assert names == [None, "u"], names


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


def test_a_real_issued_lease_still_authorizes_its_own_upload(tmp_path) -> None:
    """⚑ THE REGRESSION GUARD FOR THE FAIL-OPEN FIX.

    Every other lease test builds the lease dict BY HAND, so none of them can
    see a mismatch between what `assign_trial_lease` actually STORES and what
    `lease_authorizes_upload` reads -- and tightening two fail-open branches is
    exactly the change that could introduce one. A rule that rejects the
    issuer's own leases would break the volunteer deployment the flag exists
    for, while every hand-built test stayed green.

    So this asks the REAL issuer for a lease and feeds it to the REAL rule.
    """
    import time

    from chess_anti_engine.server.lease import assign_trial_lease

    leases_root = tmp_path / "leases"
    lease = assign_trial_lease(
        leases_root=leases_root,
        username="alice",
        worker_info={"worker_id": "w1"},
        available_trials=["trial_a"],
        manifest_loader=lambda _tid: {"model": {}},
    )

    # The issuer's own output must satisfy the rule, on its own trial...
    assert lease_authorizes_upload(
        lease, username="alice", trial_id=str(lease["trial_id"]), now_unix=int(time.time()),
    ) is None, f"the rule rejects a lease its own issuer just produced: {lease}"

    # ...and on the shared default route.
    assert lease_authorizes_upload(
        lease, username="alice", trial_id=None, now_unix=int(time.time()),
    ) is None

    # The fields the tightened branches depend on must actually be populated;
    # if the issuer stops setting either, the rule silently starts refusing.
    assert int(lease.get("expires_at_unix") or 0) > int(time.time()), (
        "issuer produced a lease with no future expiry -- the tightened expiry "
        "branch would refuse every upload"
    )
    assert lease.get("trial_id"), (
        "issuer produced a trial-less lease -- the tightened trial branch would "
        "refuse every named-trial upload"
    )


def test_require_worker_lease_is_accepted_by_the_yaml_schema() -> None:
    """Hop 0->1, the one hop the wiring tests still missed.

    ⚑ Commenting the key out of `config_yaml._TUNE_KEYS` left the whole suite
    green, yet it is load-bearing and FATAL: per CLAUDE.md an unknown key is
    category (a), and `flatten_run_config_defaults` runs before the argument
    parser and outside any try -- so the run does not start at all. Loud, but
    "crosses four hops with a test asserting each" was only true of three.
    """
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    flat = flatten_run_config_defaults({"tune": {"require_worker_lease": True}})
    assert flat["require_worker_lease"] is True

    # And the schema really is what admits it -- an unknown sibling is refused,
    # so this is not passing because the validator accepts anything.
    with pytest.raises(ValueError, match=r"[Uu]nknown"):
        flatten_run_config_defaults({"tune": {"requre_worker_lease": True}})


def test_worker_logs_a_warning_on_a_non_200_upload(tmp_path, caplog) -> None:
    """Finding [8]'s fix, pinned.

    Deleting the new non-200 branch left the suite green. It is observability
    only -- but this PR creates the first DESIGNED 403 on the upload route, so
    without a log line a lease misconfiguration is undiagnosable in the field,
    and the PR's own cleartext test argues that a guard nothing proves is
    installed is one that gets removed by accident.
    """
    import logging
    import types

    from chess_anti_engine.worker import WorkerSession

    pending = tmp_path / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    save_local_shard_arrays(
        pending / "shard_000001.zarr",
        arrs=samples_to_arrays([_sample(0)]),
        meta=ShardMeta(username="u", games=1, positions=1, model_sha256="abc", model_step=0),
    )

    class _Resp:
        status_code = 403

        @staticmethod
        def json():
            return {"detail": "lease not authorized: unknown or expired lease"}

    class _Requests:
        @staticmethod
        def post(*_a, **_k):
            return _Resp()

    w = types.SimpleNamespace(
        pending_dir=pending, leased_trial_id="", fixed_trial_id="",
        trial_api_prefix="/v1", lease_id="", machine_id="m1",
        log=logging.getLogger("test_worker_403"), last_successful_send_s=0.0,
        args=types.SimpleNamespace(username="u", password="p"), _requests=_Requests,
    )
    w._server_url_for = lambda p: "http://server" + p
    w._report_bad_pending_shard = lambda payload: None

    with caplog.at_level(logging.WARNING, logger="test_worker_403"):
        WorkerSession._upload_pending_shards_locked(w, default_elapsed_s=1.0)  # pyright: ignore[reportArgumentType]

    text = caplog.text
    assert "403" in text, f"no HTTP status in the worker log: {text!r}"
    assert "lease not authorized" in text, "the server's reason never reached the worker log"
    # And the shard is retained for retry, not quarantined.
    assert list(pending.glob("*.zarr")), "a non-200 must not delete the pending shard"


# ---------------------------------------------------------------------------
# Review round 2 (Codex, PR #400). Each test below names the finding it pins.
# ---------------------------------------------------------------------------


def test_strict_config_bool_refuses_the_quoted_string() -> None:
    """⚑ `bool("false") is True`. That is the whole finding.

    YAML makes `false` and `"false"` visually identical in a config file and
    opposite in Python, and for these two keys the typo fails toward the
    dangerous answer in BOTH directions -- ingest to zero, or credentials in
    the clear.
    """
    from chess_anti_engine.utils.config_yaml import strict_config_bool

    assert strict_config_bool(True, key="k", source="s") is True
    assert strict_config_bool(False, key="k", source="s") is False

    for bad in ("false", "true", "no", "", 0, 1, None, [], {}):
        with pytest.raises(ValueError, match="must be a boolean"):
            strict_config_bool(bad, key="k", source="s")


def test_quoted_false_does_not_enable_lease_enforcement(monkeypatch, tmp_path) -> None:
    """Hop 1->2 under the typo, EXECUTED against the real driver.

    Pre-fix this appended `--require-worker-lease`, and because driver-launched
    workers never negotiate a lease that is ingest-to-zero at the next restart
    -- caused by typing the word that means "off".
    """
    import chess_anti_engine.tune.distributed_runtime as dr_mod
    import chess_anti_engine.tune.harness as harness_mod

    built: list[list[str]] = []

    def _fake_spawn(cmd, **_kwargs):
        built.append([str(c) for c in cmd])
        raise RuntimeError("stop after argv is built")

    monkeypatch.setattr(dr_mod, "_spawn_with_reap", _fake_spawn)
    monkeypatch.setattr(
        harness_mod, "_prepare_distributed_worker_auth", lambda **kw: ("tester", None),
    )

    for key in ("require_worker_lease", "worker_self_register"):
        built.clear()
        cfg = {
            "distributed_workers_per_trial": 1,
            "distributed_server_host": "127.0.0.1",
            "distributed_server_port": 45999,
            key: "false",
        }
        with pytest.raises(ValueError, match="must be a boolean"):
            harness_mod._launch_distributed_server(
                base_config=cfg, work_dir=tmp_path / f"wd_{key}",
            )
        # ⚑ The load-bearing half: it must fail BEFORE spawning, not spawn a
        # misconfigured server and then complain.
        assert not built, f"a server was launched despite a bad {key!r}: {built}"


def test_quoted_false_does_not_enable_the_cleartext_override() -> None:
    """Same defect, worker side, worse consequence: it sends the password."""
    import argparse

    from chess_anti_engine.worker import _merge_cli_with_yaml_defaults

    class _Args(argparse.Namespace):
        def __getattr__(self, name: str):
            if name.startswith("__"):
                raise AttributeError(name)
            return None

    ns = _Args()
    ns.allow_cleartext_http = False
    with pytest.raises(ValueError, match="must be a boolean"):
        _merge_cli_with_yaml_defaults(
            ns,
            {"server_url": "http://203.0.113.7:45453", "allow_cleartext_http": "false"},
        )
    assert ns.allow_cleartext_http is False, "the string enabled the override"

    # A real boolean still works, in both directions.
    ns = _Args()
    ns.allow_cleartext_http = False
    _merge_cli_with_yaml_defaults(
        ns, {"server_url": "http://203.0.113.7:45453", "allow_cleartext_http": True},
    )
    assert ns.allow_cleartext_http is True


def test_loopback_exemption_covers_the_whole_127_range() -> None:
    """`127.0.0.0/8` is all loopback, not just `127.0.0.1`.

    Refusing `127.0.0.2` would push a user to set the security override on a
    host where nothing can observe the traffic -- which teaches them to set it
    where something can.
    """
    from chess_anti_engine.worker import cleartext_transport_refusal

    for host in ("127.0.0.1", "127.0.0.2", "127.1.2.3", "[::1]", "localhost", "0.0.0.0"):
        assert cleartext_transport_refusal(f"http://{host}:45453", allow=False) is None, host

    # Negative control: the exemption did not swallow everything.
    for host in ("203.0.113.7", "10.0.0.5", "example.com", "[2001:db8::1]"):
        assert cleartext_transport_refusal(f"http://{host}:45453", allow=False) is not None, host


def test_stamp_clears_uploader_supplied_contributors(tmp_path) -> None:
    """⚑ Stamping `verified` over a forged contributor list is WORSE than not
    stamping: the quarantine procedure reads `contributors` in preference to
    `username` exactly when the shard is marked verified.
    """
    import json as _json

    from chess_anti_engine.server.app import _stamp_shard_username

    zroot = tmp_path / "shard_000001.zarr"
    zroot.mkdir()
    (zroot / ".zattrs").write_text(_json.dumps({
        "username": "victim",
        "contributors": [{"username": "victim", "start": 0, "end": 999}],
        "games": 1,
    }), encoding="utf-8")

    _stamp_shard_username(zroot, "mallory")

    attrs = _json.loads((zroot / ".zattrs").read_text(encoding="utf-8"))
    assert attrs["username"] == "mallory"
    assert attrs["provenance_verified"] is True
    assert "contributors" not in attrs, (
        "uploader-supplied contributor list survived the stamp and is now "
        "carried under the server's own attestation"
    )
    # Untouched fields stay untouched.
    assert attrs["games"] == 1


def test_forged_verified_marker_on_a_legacy_shard_is_not_trusted(tmp_path) -> None:
    """⚑ THE MARKER IS INSIDE THE UPLOAD. That is the finding.

    `test_unverified_pending_shard_does_not_launder_its_claim` plants a shard
    with NO marker, which is the honest pre-upgrade shape -- and it passed
    against a guard that read nothing but the marker. But `.zattrs` ships
    inside the uploader's tarball and only `_stamp_shard_username` overwrites
    it, so a shard staged by a server that predates the stamp carries whatever
    the uploader wrote, INCLUDING `provenance_verified: true`. Adding one key
    to that fixture defeated the guard completely and recovered `victim`.

    Nothing inside the shard can settle this, because the forgeable case
    predates the code doing the checking. `_legacy_unstamped_shard_keys`
    records the pre-existing shards server-side on first boot; this asserts
    that witness is what decides.
    """
    from chess_anti_engine.replay.shard import PENDING_DIR_NAME

    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    pending = server_root / "inbox" / PENDING_DIR_NAME
    pending.mkdir(parents=True, exist_ok=True)
    zp = pending / "1000_deadbeef_aaaa.zarr"
    save_local_shard_arrays(
        zp,
        arrs=samples_to_arrays([_sample(i) for i in range(2)]),
        meta=ShardMeta(
            username="victim", games=1, positions=2,
            model_sha256="abc1234567", model_step=0,
        ),
    )
    # The forgery: mallory's tarball asserts the server's own attestation.
    attrs_path = zp / ".zattrs"
    attrs = json.loads(attrs_path.read_text())
    attrs["username"] = "victim"
    attrs["provenance_verified"] = True
    attrs_path.write_text(json.dumps(attrs, indent=4, sort_keys=True))

    client = _build_client(
        server_root, upload_compact_shard_size=3, upload_compact_max_age_seconds=1e9,
    )
    # The witness must exist on disk, outside anything an uploader can write.
    watermark = server_root / "provenance_migration.json"
    assert watermark.exists(), "no server-side provenance watermark was written"
    assert "1000_deadbeef_aaaa.zarr" in watermark.read_text()

    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", _tar_bytes(tmp_path / "fresh", n=1), "application/x-tar")},
        headers=_headers(),
    )
    assert r.status_code == 200, r.text

    shards = _compacted_shards(server_root)
    assert shards, "no compacted shard was produced"
    _, meta = load_shard_arrays(shards[0])
    names = [c["username"] for c in (meta.get("contributors") or [])]
    assert "victim" not in names, (
        f"a FORGED verification marker was trusted and laundered: {names!r}"
    )
    assert names == [None, "u"], names


def test_the_watermark_does_not_distrust_shards_the_server_stamped(tmp_path) -> None:
    """⚑ NEGATIVE CONTROL. A watermark that distrusts everything would pass the
    test above while destroying attribution for the entire fleet -- which is
    the failure mode of a guard tuned only against its positive case.

    Boot once to lay down the watermark, upload through the real route so the
    shard is stamped by THIS build, then boot again on the same root. The
    second boot's recovery must still attribute it.
    """
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    # Boot 1: empty server, so the watermark records nothing.
    first = _build_client(server_root, upload_compact_shard_size=10_000)
    assert json.loads((server_root / "provenance_migration.json").read_text())[
        "legacy_shards"
    ] == []

    r = first.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", _tar_bytes(tmp_path / "s", n=2), "application/x-tar")},
        headers=_headers(),
    )
    assert r.status_code == 200, r.text
    assert [p for p in (server_root / "inbox").rglob("*.zarr") if "_pending" in str(p)], (
        "the upload did not stay pending; this test needs it to survive to boot 2"
    )

    # Boot 2: same root, shard still pending. It was stamped by this build, so
    # it must NOT be swept up as legacy.
    second = _build_client(
        server_root, upload_compact_shard_size=3, upload_compact_max_age_seconds=1e9,
    )
    r = second.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", _tar_bytes(tmp_path / "s2", n=1), "application/x-tar")},
        headers=_headers(),
    )
    assert r.status_code == 200, r.text

    shards = _compacted_shards(server_root)
    assert shards, "no compacted shard was produced"
    _, meta = load_shard_arrays(shards[0])
    names = [c["username"] for c in (meta.get("contributors") or [])]
    assert None not in names, (
        f"the watermark distrusted a shard this build stamped: {names!r}"
    )
    assert set(names) == {"u"}, names
