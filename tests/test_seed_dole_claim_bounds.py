"""`POST /v1/seed_dole_claim` bounds its caller input -- and REJECTS, never truncates.

The coverage hole: every other authenticated POST on this server bounds what a
caller can hand it (`upload_shard` -> `max_upload_mb` -> 413,
`upload_arena_result` -> `arena_max_body_bytes` -> a terminal rejection,
`report_bad_shard` -> `_bounded_report_field` -> truncation). The newest of
them, the seed-dole claim, bounded nothing: `payload: dict = Body(...)` parsed
the whole body on every claim, and `claim_id` was persisted VERBATIM into
`seed_dole_gate.json.winners.json` -- a file `create_app` re-reads on EVERY
boot. An 8 MB `claim_id` was accepted (`200 granted=true`) and written as an
8,388,756-byte sidecar.

⚑⚑ THE PART THAT IS NOT A COPY OF `_bounded_report_field`. `claim_id` is an
IDENTITY TOKEN COMPARED BY EQUALITY, not diagnostic free text. Truncating it
would collapse distinct ids onto one value, so a genuinely new claim would
match a DIFFERENT worker's stored winner and be handed that worker's
`grant_token` while the one-shot per-iteration dose stayed burned -- a silent
seeding loss, strictly worse than the unbounded write being fixed.
`test_two_long_claim_ids_sharing_a_prefix_cannot_collapse` is the test that
fails if anyone ever "unifies" the two helpers.

⚑ The suite leads with POSITIVE CONTROLS on purpose. The failure mode of a
bounds change is refusing real traffic, and a suite made only of rejection
tests cannot see it -- this series has already shipped one finding whose
proposed fix would have refused every worker poll.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import httpx

from chess_anti_engine.server.app import (
    _MAX_SEED_DOLE_CLAIM_ID_CHARS,
    _MAX_SEED_DOLE_WINNERS_BYTES,
    _SeedDoleGate,
    create_app,
)
from chess_anti_engine.worker import _manifest_poll_headers
from tests.test_distributed_selfplay_backpressure import (
    _DOLE_SEED_FEN,
    _publish_dole_trial,
    _seed_dole_user,
)

MANIFEST = "/v1/trials/trial_00000/manifest"
CLAIM = "/v1/trials/trial_00000/seed_dole_claim"

# What `worker.py::_claim_seed_dole` actually sends: `uuid.uuid4().hex`.
REAL_WORKER_CLAIM_ID_CHARS = 32


def _setup(tmp_path: Path, *, iteration: int = 7, **app_kwargs: Any):
    tmp_path.mkdir(parents=True, exist_ok=True)
    fen_path = tmp_path / "blindspot.txt"
    fen_path.write_text(_DOLE_SEED_FEN + "\n", encoding="utf-8")
    _publish_dole_trial(tmp_path, training_iteration=iteration, dole=1, fen_path=fen_path)
    _seed_dole_user(tmp_path)
    return create_app(server_root=tmp_path, users_db="users.json", **app_kwargs)


def _gate_state(tmp_path: Path) -> str:
    """The DURABLE gate -- the only honest witness to 'was the dose burned'."""
    path = tmp_path / "seed_dole_gate.json"
    return path.read_text(encoding="utf-8") if path.exists() else "<absent>"


def _winners_path(tmp_path: Path) -> Path:
    return tmp_path / "seed_dole_gate.json.winners.json"


async def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver",
    )


async def _revision(client) -> str:
    headers = _manifest_poll_headers(worker_id="w")
    manifest = (await client.get(MANIFEST, headers=headers)).json()
    return str(manifest["manifest_revision"])


# --------------------------------------------------------------------------
# POSITIVE CONTROLS -- a real worker must still win.
# --------------------------------------------------------------------------


def test_a_real_worker_claim_still_succeeds(tmp_path: Path) -> None:
    """⚑⚑ THE POSITIVE CONTROL. The failure mode of this change is refusing
    real workers, and no number of rejection tests can detect that.

    This sends exactly what `worker.py` sends: a `uuid.uuid4().hex` claim_id
    against the revision the manifest just advertised.
    """
    app = _setup(tmp_path)

    async def _run() -> tuple[int, dict]:
        async with await _client(app) as client:
            rev = await _revision(client)
            claim_id = uuid.uuid4().hex
            assert len(claim_id) == REAL_WORKER_CLAIM_ID_CHARS
            r = await client.post(
                CLAIM,
                json={"claim_id": claim_id, "manifest_revision": rev},
                auth=("u", "p"),
                headers=_manifest_poll_headers(worker_id="w"),
            )
            return r.status_code, r.json()

    status, body = asyncio.run(_run())
    assert status == 200, f"a legitimate worker claim was refused with HTTP {status}"
    assert body["granted"] is True, f"a legitimate worker claim lost: {body}"
    assert body["grant_token"]
    assert body["seeds"] == 1
    assert _gate_state(tmp_path) != "<absent>", "the winning claim did not advance the gate"


def test_the_real_worker_claim_id_length_is_far_under_the_cap() -> None:
    """The bound is justified against what our own client sends, not guessed.

    A bound below what the fleet emits would refuse every worker -- the failure
    this series has already produced once (`require_worker_lease`: 821,818
    uploads, zero leases ever issued).
    """
    assert REAL_WORKER_CLAIM_ID_CHARS < _MAX_SEED_DOLE_CLAIM_ID_CHARS
    assert _MAX_SEED_DOLE_CLAIM_ID_CHARS >= 8 * REAL_WORKER_CLAIM_ID_CHARS


def test_a_padded_body_under_the_cap_still_wins(tmp_path: Path) -> None:
    """Headroom is real, not nominal.

    Also pins the decision on UNKNOWN payload fields: they are never read and
    never persisted, so they are bounded by the body cap alone and get no
    per-field guard. A large-but-legal body is accepted, exactly as before.
    """
    app = _setup(tmp_path)

    async def _run() -> tuple[int, dict]:
        async with await _client(app) as client:
            rev = await _revision(client)
            payload = {
                "claim_id": uuid.uuid4().hex,
                "manifest_revision": rev,
                # Comfortably inside the 64 KiB default cap.
                "unused_future_field": "p" * 8192,
            }
            r = await client.post(
                CLAIM, json=payload, auth=("u", "p"),
                headers=_manifest_poll_headers(worker_id="w"),
            )
            return r.status_code, r.json()

    status, body = asyncio.run(_run())
    assert status == 200, f"a legal padded body was refused with HTTP {status}"
    assert body["granted"] is True, f"a legal padded body lost the claim: {body}"


# --------------------------------------------------------------------------
# THE BODY CAP
# --------------------------------------------------------------------------


def test_an_oversized_claim_body_is_refused_and_never_burns_the_dose(tmp_path: Path) -> None:
    """The body is refused mid-stream, and the dole survives it.

    ⚑ Asserts on the DURABLE GATE, not the response: a refusal that had already
    consumed the iteration would read identically in the response body while
    silently costing the real worker its dose.
    """
    app = _setup(tmp_path)

    async def _run() -> tuple[int, bool]:
        async with await _client(app) as client:
            rev = await _revision(client)
            fat = await client.post(
                CLAIM,
                json={
                    "claim_id": uuid.uuid4().hex,
                    "manifest_revision": rev,
                    "pad": "z" * (256 * 1024),
                },
                auth=("u", "p"),
                headers=_manifest_poll_headers(worker_id="w"),
            )
            # The gate must still be winnable by a correct claim afterwards.
            good = await client.post(
                CLAIM,
                json={"claim_id": uuid.uuid4().hex, "manifest_revision": rev},
                auth=("u", "p"),
                headers=_manifest_poll_headers(worker_id="w"),
            )
            good.raise_for_status()
            return fat.status_code, bool(good.json()["granted"])

    status, granted_after = asyncio.run(_run())
    assert status == 413, f"an oversized claim body was accepted with HTTP {status}"
    assert granted_after is True, "an oversized body burned the claim"


def test_the_body_cap_is_actually_wired_to_the_route(tmp_path: Path) -> None:
    """⚑ A knob that never reaches its consumer is this codebase's signature
    defect, so prove the parameter MOVES the behaviour rather than merely
    existing: one body, two caps, two different answers.
    """
    body = {"claim_id": uuid.uuid4().hex, "manifest_revision": "x", "pad": "z" * 8192}

    async def _status(app) -> int:
        async with await _client(app) as client:
            r = await client.post(
                CLAIM, json=body, auth=("u", "p"),
                headers=_manifest_poll_headers(worker_id="w"),
            )
            return r.status_code

    tight = _setup(tmp_path / "tight", seed_dole_max_body_bytes=2048)
    loose = _setup(tmp_path / "loose", seed_dole_max_body_bytes=1024 * 1024)
    assert asyncio.run(_status(tight)) == 413, "the cap did not reach the route"
    # Same body, bigger cap: it gets past the size guard and is answered on the
    # route's normal path (a revision mismatch, since "x" is not the revision).
    assert asyncio.run(_status(loose)) == 200, "a large cap still refused a body under it"


# --------------------------------------------------------------------------
# claim_id: REJECTED, NEVER TRUNCATED
# --------------------------------------------------------------------------


def test_an_over_long_claim_id_is_rejected_without_burning_the_dose(tmp_path: Path) -> None:
    app = _setup(tmp_path)

    async def _run() -> tuple[int, dict, bool]:
        async with await _client(app) as client:
            rev = await _revision(client)
            bad = await client.post(
                CLAIM,
                json={
                    "claim_id": "q" * (_MAX_SEED_DOLE_CLAIM_ID_CHARS + 1),
                    "manifest_revision": rev,
                },
                auth=("u", "p"),
                headers=_manifest_poll_headers(worker_id="w"),
            )
            good = await client.post(
                CLAIM,
                json={"claim_id": uuid.uuid4().hex, "manifest_revision": rev},
                auth=("u", "p"),
                headers=_manifest_poll_headers(worker_id="w"),
            )
            good.raise_for_status()
            return bad.status_code, bad.json(), bool(good.json()["granted"])

    status, body, granted_after = asyncio.run(_run())
    assert status == 400, f"an over-long claim_id was accepted with HTTP {status}"
    assert body["reason_code"] == "claim_id_too_long"
    assert body["granted"] is False
    assert granted_after is True, "a rejected claim_id burned the dose"


def test_an_over_long_claim_id_never_reaches_the_winner_sidecar(tmp_path: Path) -> None:
    """The sidecar is re-read on every boot, so this is the durable half."""
    app = _setup(tmp_path)
    huge = "w" * (512 * 1024)

    async def _run() -> None:
        async with await _client(app) as client:
            rev = await _revision(client)
            await client.post(
                CLAIM,
                json={"claim_id": huge, "manifest_revision": rev},
                auth=("u", "p"),
                headers=_manifest_poll_headers(worker_id="w"),
            )

    asyncio.run(_run())
    wp = _winners_path(tmp_path)
    if wp.exists():
        assert huge not in wp.read_text(encoding="utf-8"), "an unbounded claim_id reached disk"
        assert wp.stat().st_size < 4096, f"winner sidecar grew to {wp.stat().st_size} bytes"


def test_two_long_claim_ids_sharing_a_prefix_cannot_collapse(tmp_path: Path) -> None:
    """⚑⚑ THE TEST THAT FAILS IF ANYONE TRUNCATES `claim_id` INSTEAD.

    Under a `_bounded_report_field`-style truncation these two ids -- distinct
    requests from distinct workers -- both cut down to the same value. Worker A
    would win and store it; worker B's genuinely NEW claim would then match A's
    stored winner and be answered `granted=true` WITH A'S `grant_token`, having
    issued no dose. B plays a batch it was never granted, and the one-shot
    per-iteration dose is spent. That is the silent seeding loss this endpoint
    exists to prevent, reintroduced by the fix for a different bug.

    Rejection makes the collapse unreachable: neither id is ever stored, so
    there is nothing for the other to match.
    """
    app = _setup(tmp_path)
    shared = "s" * _MAX_SEED_DOLE_CLAIM_ID_CHARS
    id_a, id_b = shared + "AAA", shared + "BBB"
    assert id_a[:_MAX_SEED_DOLE_CLAIM_ID_CHARS] == id_b[:_MAX_SEED_DOLE_CLAIM_ID_CHARS]

    async def _run() -> tuple[dict, dict]:
        async with await _client(app) as client:
            rev = await _revision(client)

            async def claim(cid: str) -> dict:
                r = await client.post(
                    CLAIM, json={"claim_id": cid, "manifest_revision": rev},
                    auth=("u", "p"), headers=_manifest_poll_headers(worker_id="w"),
                )
                return {"status": r.status_code, **r.json()}

            return await claim(id_a), await claim(id_b)

    a, b = asyncio.run(_run())
    assert a["status"] == 400, f"worker A's over-long claim_id was accepted: {a}"
    assert b["status"] == 400, f"worker B's over-long claim_id was accepted: {b}"
    assert a["granted"] is False
    assert b["granted"] is False
    # The harm the collapse would cause, asserted directly: B must not be
    # handed A's token, and no dose may have been issued at all.
    assert not a.get("grant_token"), "a dose was issued for an over-long claim_id"
    assert b.get("grant_token") != a.get("grant_token") or not b.get("grant_token"), (
        "worker B was handed worker A's grant_token -- the truncation collapse"
    )
    assert _gate_state(tmp_path) == "<absent>", "a truncated-id collapse burned the dose"


# --------------------------------------------------------------------------
# The winner sidecar, which is parsed at EVERY boot.
# --------------------------------------------------------------------------


def _write_gate(tmp_path: Path, winners: Any, *, last_iter: int = 7) -> Path:
    state = tmp_path / "seed_dole_gate.json"
    state.write_text(json.dumps({"trial_00000": last_iter}), encoding="utf-8")
    _winners_path(tmp_path).write_text(json.dumps(winners), encoding="utf-8")
    return state


def test_a_normal_winner_sidecar_is_still_loaded(tmp_path: Path) -> None:
    """Control for the size check below -- it must not refuse real sidecars."""
    state = _write_gate(tmp_path, {
        "trial_00000": {
            "iteration": 7,
            "claim_id": uuid.uuid4().hex,
            "revision": "abc123",
            "grant_token": uuid.uuid4().hex,
        },
    })
    gate = _SeedDoleGate(state_path=state)
    assert gate._winners["trial_00000"]["iteration"] == 7
    assert gate._last_iter["trial_00000"] == 7


def test_an_oversized_winner_sidecar_is_refused_at_boot(tmp_path: Path) -> None:
    """⚑ This guards the PAST, not the future.

    With `claim_id` bounded at the route no sidecar this code writes can get
    here. But a sidecar written by a PRE-FIX server survives the upgrade, and
    `create_app` re-parses it on EVERY boot forever; only a read-side check can
    retire it. Dropping the record costs at most one dose -- `_last_iter` comes
    from the separate gate file and still refuses a second grant for an
    iteration already handed out, so this is not a double-grant risk.
    """
    state = _write_gate(tmp_path, {
        "trial_00000": {
            "iteration": 7,
            "claim_id": "x" * (_MAX_SEED_DOLE_WINNERS_BYTES + 4096),
            "revision": "abc123",
            "grant_token": "tok",
        },
    })
    assert _winners_path(tmp_path).stat().st_size > _MAX_SEED_DOLE_WINNERS_BYTES

    gate = _SeedDoleGate(state_path=state)
    assert gate._winners == {}, "an oversized winner sidecar was parsed at boot"
    # The double-grant guard is untouched: the durable gate still holds.
    assert gate._last_iter["trial_00000"] == 7
