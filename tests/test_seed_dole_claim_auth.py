"""The seed-dole claim is authenticated and the manifest GET is side-effect-free.

The defect: `/v1/manifest` carried no auth dependency and its handler called
`seed_dole_gate.claim(...)`, so any unauthenticated caller who could reach the
port won the one-shot per-iteration grant and denied it to the real worker.
Blind-spot seeding then stopped SILENTLY -- a stolen grant is indistinguishable
from no grant in every signal the fleet emits.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import httpx

from chess_anti_engine.server.app import create_app
from chess_anti_engine.worker import _manifest_poll_headers
from tests.test_distributed_selfplay_backpressure import (
    _DOLE_SEED_FEN,
    _publish_dole_trial,
    _seed_dole_user,
)

MANIFEST = "/v1/trials/trial_00000/manifest"
CLAIM = "/v1/trials/trial_00000/seed_dole_claim"


def _setup(tmp_path: Path, *, iteration: int = 7):
    fen_path = tmp_path / "blindspot.txt"
    fen_path.write_text(_DOLE_SEED_FEN + "\n", encoding="utf-8")
    _publish_dole_trial(tmp_path, training_iteration=iteration, dole=1, fen_path=fen_path)
    _seed_dole_user(tmp_path)
    return create_app(server_root=tmp_path, users_db="users.json")


def _gate_state(tmp_path: Path) -> str:
    """The DURABLE gate, which is the only honest witness here."""
    path = tmp_path / "seed_dole_gate.json"
    return path.read_text(encoding="utf-8") if path.exists() else "<absent>"


async def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver",
    )


def test_anonymous_manifest_polls_cannot_move_the_durable_gate(tmp_path: Path) -> None:
    """⚑⚑ THE NEGATIVE CONTROL, AND IT ASSERTS ON THE GATE FILE ON PURPOSE.

    A response of `dole_fen_seeds: false` is perfectly consistent with the gate
    having been advanced -- that is exactly what the attacker sees on their
    SECOND anonymous poll under the old code. So a test that read the response
    body would pass while the vulnerability was fully present. Only the
    persisted gate distinguishes "nobody claimed" from "somebody already did".
    """
    app = _setup(tmp_path)
    before = _gate_state(tmp_path)

    async def _run() -> None:
        async with await _client(app) as client:
            for _ in range(25):
                r = await client.get(MANIFEST, headers=_manifest_poll_headers(worker_id="anon"))
                r.raise_for_status()
                assert r.json()["dole_fen_seeds"] is False

    asyncio.run(_run())
    assert _gate_state(tmp_path) == before, "an anonymous GET moved the seed-dole gate"


def test_anonymous_claim_is_refused(tmp_path: Path) -> None:
    app = _setup(tmp_path)

    async def _run() -> int:
        async with await _client(app) as client:
            r = await client.post(
                CLAIM, json={"claim_id": "x"}, headers=_manifest_poll_headers(worker_id="anon"),
            )
            return r.status_code

    assert asyncio.run(_run()) == 401
    assert _gate_state(tmp_path) == "<absent>"


def test_an_unknown_credential_is_refused_and_creates_no_account(tmp_path: Path) -> None:
    """⚑ REGRESSION GUARD. `_auth_user` SELF-REGISTERS an unknown username when
    `worker_self_register` is on -- measured on the telemetry route, where a
    plain GET created an account. A claim endpoint wired to it would enrol its
    own attacker, who would then be an "existing account" on every later
    request. This is why the route depends on `_auth_existing_user`.
    """
    app = _setup(tmp_path)
    users_before = json.loads((tmp_path / "users.json").read_text(encoding="utf-8"))

    async def _run() -> int:
        async with await _client(app) as client:
            r = await client.post(
                CLAIM,
                json={"claim_id": "x"},
                auth=("attacker", "a-long-enough-password"),
                headers=_manifest_poll_headers(worker_id="anon"),
            )
            return r.status_code

    assert asyncio.run(_run()) == 401
    users_after = json.loads((tmp_path / "users.json").read_text(encoding="utf-8"))
    assert set(users_after) == set(users_before), "the claim endpoint registered an account"
    assert _gate_state(tmp_path) == "<absent>"


def test_a_revision_mismatch_declines_without_burning_the_claim(tmp_path: Path) -> None:
    """⚑ A mismatch must NO-OP, leaving the gate unclaimed for a later correct
    poll. Consuming it would turn a benign race into precisely the silent
    seeding loss this endpoint exists to prevent.
    """
    app = _setup(tmp_path)

    async def _run() -> tuple[dict, bool]:
        async with await _client(app) as client:
            headers = _manifest_poll_headers(worker_id="w")
            stale = await client.post(
                CLAIM,
                json={"claim_id": uuid.uuid4().hex, "manifest_revision": "not-the-revision"},
                auth=("u", "p"), headers=headers,
            )
            stale.raise_for_status()
            # The gate must still be winnable by a correct claim afterwards.
            manifest = (await client.get(MANIFEST, headers=headers)).json()
            good = await client.post(
                CLAIM,
                json={
                    "claim_id": uuid.uuid4().hex,
                    "manifest_revision": manifest["manifest_revision"],
                },
                auth=("u", "p"), headers=headers,
            )
            good.raise_for_status()
            return stale.json(), bool(good.json()["granted"])

    stale_body, granted_after = asyncio.run(_run())
    assert stale_body["granted"] is False
    assert stale_body["reason_code"] == "revision_mismatch"
    assert granted_after is True, "a revision mismatch burned the claim"


def test_a_dropped_response_is_recovered_by_replaying_the_claim_id(tmp_path: Path) -> None:
    """End-to-end idempotency over HTTP, not just at the gate object."""
    app = _setup(tmp_path)

    async def _run() -> tuple[bool, bool, bool]:
        async with await _client(app) as client:
            headers = _manifest_poll_headers(worker_id="w")
            manifest = (await client.get(MANIFEST, headers=headers)).json()
            rev = manifest["manifest_revision"]
            mine = uuid.uuid4().hex

            async def claim(cid: str) -> bool:
                r = await client.post(
                    CLAIM, json={"claim_id": cid, "manifest_revision": rev},
                    auth=("u", "p"), headers=headers,
                )
                r.raise_for_status()
                return bool(r.json()["granted"])

            return await claim(mine), await claim(mine), await claim(uuid.uuid4().hex)

    first, retry, other = asyncio.run(_run())
    assert (first, retry) == (True, True), "the winner could not recover a dropped response"
    assert other is False, "idempotency became 'everyone wins'"


def test_the_manifest_advertises_the_endpoint_only_when_the_dole_is_live(tmp_path: Path) -> None:
    """The capability advertisement is what lets a new worker talk to an OLD
    server, so it must not appear when there is nothing to claim."""
    fen_path = tmp_path / "blindspot.txt"
    fen_path.write_text(_DOLE_SEED_FEN + "\n", encoding="utf-8")
    _publish_dole_trial(tmp_path, training_iteration=3, dole=0, fen_path=fen_path)  # dole off
    _seed_dole_user(tmp_path)
    app = create_app(server_root=tmp_path, users_db="users.json")

    async def _run() -> dict:
        async with await _client(app) as client:
            r = await client.get(MANIFEST, headers=_manifest_poll_headers(worker_id="w"))
            r.raise_for_status()
            return r.json()

    manifest = asyncio.run(_run())
    assert "seed_dole_claim_endpoint" not in manifest
    assert manifest["dole_fen_seeds"] is False
