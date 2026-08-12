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
