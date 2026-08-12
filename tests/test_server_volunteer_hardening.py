"""Controls that make an untrusted contributor survivable.

None of these prevent a malicious volunteer from uploading bad data -- nothing
can, because a policy target is a probability vector and there is no server-side
way to tell a genuine MCTS distribution from a plausible forgery (see the
2026-08-03 legality scan: it reads zero on corrupted search trees BY
CONSTRUCTION). What they buy is attribution and revocability: knowing whose rows
these were, and refusing writes outside the lease the server actually handed out.
"""
from __future__ import annotations

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

def test_require_worker_lease_is_wired_end_to_end() -> None:
    """A knob that never reaches its consumer is this repo's signature defect.

    `require_worker_lease` has to survive four hops -- yaml allowlist, the
    driver's server command line, the `run_server` parser, and `create_app` --
    and a break in ANY of them leaves a config key that is accepted, logged,
    and inert. Source-level on purpose: `_launch_distributed_server` spawns a
    real uvicorn, so the alternative to reading the call site is booting a
    server in a unit test.
    """
    import chess_anti_engine.server.app as app_mod
    import chess_anti_engine.server.run_server as run_server_mod
    import chess_anti_engine.tune.harness as harness_mod
    import chess_anti_engine.tune.trainable_config_ops as cfg_ops_mod
    import chess_anti_engine.utils.config_yaml as config_yaml_mod

    hops = {
        # yaml key is accepted by the schema
        "config_yaml allowlist": (config_yaml_mod, '"require_worker_lease"'),
        # classified so the reload path knows it is launch-fixed
        "launch-fixed classification": (cfg_ops_mod, '"require_worker_lease"'),
        # driver bakes it into the server command line
        "driver command line": (harness_mod, "--require-worker-lease"),
        # the server's own parser accepts it
        "run_server parser": (run_server_mod, "--require-worker-lease"),
        # and hands it to the app
        "create_app pass-through": (run_server_mod, "require_worker_lease=bool("),
    }
    def _src(mod) -> str:
        path = getattr(mod, "__file__", None)
        assert path, f"{mod.__name__} has no source file"
        return Path(path).read_text(encoding="utf-8")

    missing = [
        name for name, (mod, needle) in hops.items() if needle not in _src(mod)
    ]
    assert not missing, f"require_worker_lease never reaches: {missing}"

    # And the consumer reads it, rather than only accepting it.
    assert "if lease_id_hdr or require_worker_lease:" in _src(app_mod)


def test_require_worker_lease_defaults_off_everywhere() -> None:
    """An unset flag must never start refusing a running fleet's uploads."""
    import inspect

    from chess_anti_engine.server.app import create_app

    assert inspect.signature(create_app).parameters["require_worker_lease"].default is False
