"""`/v1/worker_throughput` and `/v1/trial_throughput` took no credential and
returned the raw accumulator -- including each worker's hostname and CPU count.

⚑ The interesting half is what is NOT redacted. Blanking the throughput numbers
would push whoever monitors this toward handing the dashboard a real worker
credential, which is a worse outcome than the disclosure being fixed. So the
aggregate stays public and only the machine-identifying fields go.
"""
from __future__ import annotations

import json

from chess_anti_engine.server.app import (
    HOST_IDENTIFYING_TELEMETRY_KEYS,
    redact_host_telemetry,
)

from .test_server_upload_security import _build_client, _seed_user

_STATS = {
    "NVIDIA GeForce RTX 5090": {
        "gpu_model": "NVIDIA GeForce RTX 5090",
        "positions": 12345,
        "games": 678,
        "positions_per_s": 91.2,
        "last_hostname": "josh-desktop",
        "last_cpu_count": 32,
    },
}


def test_redaction_drops_host_fields_and_keeps_throughput() -> None:
    out = redact_host_telemetry(_STATS, authenticated=False)
    entry = out["NVIDIA GeForce RTX 5090"]
    assert "last_hostname" not in entry
    assert "last_cpu_count" not in entry
    # ⚑ The load-bearing half: the useful data survives.
    assert entry["positions"] == 12345
    assert entry["games"] == 678
    assert entry["positions_per_s"] == 91.2
    assert entry["gpu_model"] == "NVIDIA GeForce RTX 5090"


def test_authenticated_callers_still_see_everything() -> None:
    assert redact_host_telemetry(_STATS, authenticated=True) == _STATS


def test_redaction_does_not_mutate_the_input() -> None:
    """The stats dict is the server's own accumulator; redacting a response
    must not quietly delete fields from it."""
    original = json.loads(json.dumps(_STATS))
    redact_host_telemetry(_STATS, authenticated=False)
    assert original == _STATS


def test_non_dict_payloads_pass_through() -> None:
    assert redact_host_telemetry([], authenticated=False) == []
    assert redact_host_telemetry(None, authenticated=False) is None
    # A non-dict entry inside the dict is preserved as-is rather than dropped.
    assert redact_host_telemetry({"k": 5}, authenticated=False) == {"k": 5}


def test_no_new_host_identifying_field_leaks() -> None:
    """⚑ A DENYLIST FAILS OPEN, so pin the shape it is denying.

    An allowlist was rejected deliberately: the stats dict is an open-ended
    accumulator keyed by GPU model, and an allowlist would silently drop any
    throughput field added later -- turning a privacy control into a data-loss
    bug nobody notices. The cost is that a NEW host-identifying field would
    leak until someone adds it. This test makes that arrival visible: it fails
    when `_record_gpu_throughput` learns a field this set does not cover.
    """
    import inspect

    from chess_anti_engine.server import app as app_mod

    src = inspect.getsource(app_mod)
    start = src.index("def _record_gpu_throughput(")
    body = src[start:start + 3000]

    # Every `entry["..."] = ` key the recorder writes.
    import re
    written = set(re.findall(r'entry\[\"([a-z_0-9]+)\"\]\s*=', body))
    assert written, "could not parse the recorder's written keys — did it move?"

    suspicious = {
        k for k in written
        if any(t in k for t in ("host", "cpu", "user", "machine", "ip", "addr"))
    }
    unredacted = suspicious - set(HOST_IDENTIFYING_TELEMETRY_KEYS)
    assert not unredacted, (
        f"host-identifying telemetry field(s) {sorted(unredacted)} are written by "
        "_record_gpu_throughput but not in HOST_IDENTIFYING_TELEMETRY_KEYS, so "
        "they are served to unauthenticated callers"
    )


def test_route_redacts_for_an_anonymous_caller(tmp_path) -> None:
    """⚑ WIRING. The pure function is worthless if the route does not call it.

    Deleting the `redact_host_telemetry(...)` wrapper from the two routes must
    fail HERE -- every other test above would still pass.
    """
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    (server_root / "worker_throughput_by_gpu.json").write_text(json.dumps(_STATS))
    client = _build_client(server_root)

    r = client.get("/v1/worker_throughput")
    assert r.status_code == 200, r.text
    text = r.text
    assert "josh-desktop" not in text, f"hostname served to an anonymous caller: {text}"
    assert "last_cpu_count" not in text
    # Still useful without a credential.
    assert "positions" in text


def test_route_serves_full_telemetry_to_an_authenticated_caller(tmp_path) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    (server_root / "worker_throughput_by_gpu.json").write_text(json.dumps(_STATS))
    client = _build_client(server_root)

    r = client.get("/v1/worker_throughput", auth=("u", "p"))
    assert r.status_code == 200, r.text
    assert "josh-desktop" in r.text


def test_a_bad_credential_is_not_an_error_but_is_not_trusted(tmp_path) -> None:
    """The route stays reachable, and a wrong password does not buy the full
    view. It still costs the caller an entry in the failed-sign-in throttle,
    so this is not a free guessing oracle."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    (server_root / "worker_throughput_by_gpu.json").write_text(json.dumps(_STATS))
    client = _build_client(server_root)

    r = client.get("/v1/worker_throughput", auth=("u", "wrong-password"))
    assert r.status_code == 200, r.text
    assert "josh-desktop" not in r.text


def test_telemetry_get_never_registers_an_account(tmp_path) -> None:
    """⚑⚑ REGRESSION (review finding, P1). The first version delegated to
    `_auth_user`, which SELF-REGISTERS an unknown username when
    `worker_self_register` is on -- so a plain **GET** of a telemetry route
    created a user account and then served that invented account the
    unredacted view.

    MEASURED before the fix: `GET /v1/worker_throughput` as
    `attacker_invented` / a 10-char password added `attacker_invented` to
    `users.json` and returned `last_hostname`.

    That is the same defect shape as the finding this change exists to fix -- a
    read-style public endpoint performing a state transition -- reproduced one
    route over. It was inert (`worker_self_register` defaults off) and would
    have armed itself the day volunteer registration was enabled.

    ⚑ The password must be long enough to REGISTER. A first version of this
    probe used a 5-character password, registration failed the length rule, the
    `except HTTPException` swallowed it, and the test reported "no account
    created" -- a false negative that made the vulnerability look absent. The
    positive control below is what makes this test trustworthy.
    """
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    (server_root / "worker_throughput_by_gpu.json").write_text(json.dumps(_STATS))
    users = server_root / "users.json"
    before = set(json.loads(users.read_text()))

    client = _build_client(server_root, worker_self_register=True)
    r = client.get("/v1/worker_throughput", auth=("attacker_invented", "pw12345678"))

    assert r.status_code == 200, r.text
    after = set(json.loads(users.read_text()))
    assert after == before, f"a telemetry GET created account(s): {sorted(after - before)}"
    assert "josh-desktop" not in r.text, "an invented account was served host telemetry"


def test_positive_control_self_registration_really_is_enabled(tmp_path) -> None:
    """⚑ Proves the test above can FAIL. Without this, a flag that never
    reached `create_app` -- or a password rejected for length -- would make the
    regression test pass while measuring nothing at all.

    `report_bad_shard` runs the full `_auth_user` path, so TOFU must fire there
    under exactly the credentials the test above uses.
    """
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    users = server_root / "users.json"
    before = set(json.loads(users.read_text()))

    client = _build_client(server_root, worker_self_register=True)
    r = client.post(
        "/v1/report_bad_shard",
        auth=("attacker_invented", "pw12345678"),
        json={"shard_name": "x.zarr", "reason": "test"},
    )
    assert r.status_code == 200, r.text
    created = set(json.loads(users.read_text())) - before
    assert created == {"attacker_invented"}, (
        f"self-registration did not fire on a route where it should: {created}. "
        "The regression test above is therefore not measuring anything."
    )


def test_an_unknown_name_on_a_public_route_is_charged_to_the_throttle(tmp_path) -> None:
    """⚑⚑ THIS TEST REPLACES A SOURCE GREP THAT COULD NOT DO ITS JOB.

    The old version asserted `"_auth_user(" not in <function source>`. It
    passed only because the docstrings happened to write `_auth_user` without a
    paren, so a future docstring could fail it with no behaviour change while
    no behavioural regression could fail it at all -- wrong in both directions,
    which is the worst kind of guard: it generates work and catches nothing.

    What it should have been checking, and now does: the non-registering path
    must not become a FREE username-enumeration oracle. An earlier version
    returned early when the name was unknown, skipping the ban check, the
    per-IP throttle AND the KDF. That made an anonymous sweep of a name
    wordlist unthrottled and ~18x faster on a miss than a hit -- so an attacker
    could separate real accounts from invented ones at full speed, on a route
    reachable with no credential at all.

    ⚑ It does NOT assert constant time. A known name pays PBKDF2 and an unknown
    one does not; that is pre-existing behaviour on every route here. The
    property under test is that the ATTEMPT IS COUNTED, so the existing per-IP
    throttle applies to enumeration exactly as it applies to password guessing.
    """
    server_root = tmp_path
    _seed_user(server_root)
    users = server_root / "users.json"
    before = set(json.loads(users.read_text()))

    client = _build_client(server_root, worker_self_register=True)

    # Sweep invented names against the credential-free telemetry route.
    statuses = set()
    for i in range(40):
        r = client.get("/v1/worker_throughput", auth=(f"invented_{i}", "a-long-password"))
        statuses.add(r.status_code)

    # No account may be created by any of them.
    assert set(json.loads(users.read_text())) == before, (
        "a public GET self-registered an account"
    )

    assert statuses <= {200, 429}, f"unexpected statuses from the public route: {statuses}"
    # The throttle must have engaged: a real user's wrong password from the
    # same IP is now refused, which it would not be if misses were free.
    probe = client.post("/v1/report_bad_shard", json={"shard": "x"}, auth=("u", "wrong"))
    assert probe.status_code == 429, (
        "40 unknown-name probes did not charge the per-IP failed-sign-in throttle, "
        "so username enumeration on this route is unthrottled"
    )
