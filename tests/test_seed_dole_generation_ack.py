"""Concurrency contract for seed-dole generations, ACKs and recovery leases."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import chess_anti_engine.server.app as server_app
from chess_anti_engine.server.app import (
    SEED_DOLE_PERSIST_FAILED,
    SEED_DOLE_REARM_FILENAME,
    _SeedDoleGate,
)
from chess_anti_engine.tune.distributed_runtime import (
    _seed_dole_generation_for_iteration,
    _seed_dole_rearm_payload,
)
from tests.test_worker_dole_applied_once import _FakeRequests, _manifest, _session


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _write_rearm(pub: Path, iteration: int, token: str) -> Path:
    pub.mkdir(parents=True, exist_ok=True)
    path = pub / SEED_DOLE_REARM_FILENAME
    path.write_text(
        json.dumps({"training_iteration": iteration, "grant_token": token}),
        encoding="utf-8",
    )
    return path


def test_live_legacy_owner_heartbeat_defers_same_iteration_redole(tmp_path: Path) -> None:
    clock = _Clock()
    pub = tmp_path / "publish"
    gate = _SeedDoleGate(tmp_path / "seed_dole_gate.json", lease_seconds=10, clock=clock)
    token = asyncio.run(gate.claim_token("t", 7, claim_id="A", manifest_revision="r"))
    assert token

    clock.advance(9)
    rearm = _write_rearm(pub, 7, token)
    assert (
        asyncio.run(
            gate.claim_token(
                "t", 7, publish_dir=pub, claim_id="A", manifest_revision="r",
            )
        )
        == token
    )
    assert rearm.exists(), "a live lease must defer rather than destroy recovery"

    clock.advance(2)  # Past the original expiry, but not the renewed lease.
    assert (
        asyncio.run(
            gate.claim_token(
                "t", 7, publish_dir=pub, claim_id="B", manifest_revision="r",
            )
        )
        == ""
    )
    assert rearm.exists()

    clock.advance(9)
    replacement = asyncio.run(
        gate.claim_token("t", 7, publish_dir=pub, claim_id="B", manifest_revision="r")
    )
    assert replacement
    assert replacement != token
    assert not rearm.exists()


def test_active_rearm_survives_nonowner_poll_and_recovers_after_expiry(tmp_path: Path) -> None:
    clock = _Clock()
    pub = tmp_path / "publish"
    gate = _SeedDoleGate(tmp_path / "seed_dole_gate.json", lease_seconds=10, clock=clock)
    first = asyncio.run(gate.claim_token("t", 7, claim_id="A", manifest_revision="r"))
    clock.advance(9)
    rearm = _write_rearm(pub, 7, first)

    before_expiry = asyncio.run(
        gate.claim_token("t", 7, publish_dir=pub, claim_id="B", manifest_revision="r")
    )
    assert before_expiry == ""
    assert rearm.exists(), "a nonowner poll destroyed the pending recovery request"

    clock.advance(2)
    after_expiry = asyncio.run(
        gate.claim_token("t", 7, publish_dir=pub, claim_id="C", manifest_revision="r")
    )
    assert after_expiry
    assert after_expiry != first
    assert not rearm.exists()
    assert asyncio.run(gate.claim_token("t", 7, claim_id="D", manifest_revision="r")) == ""


def test_rearm_survives_replacement_persist_failure_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    state = tmp_path / "seed_dole_gate.json"
    pub = tmp_path / "publish"
    gate = _SeedDoleGate(state, lease_seconds=10, clock=clock)
    first = asyncio.run(gate.claim_token("t", 7, claim_id="A", manifest_revision="r"))
    clock.advance(11)
    rearm = _write_rearm(pub, 7, first)

    monkeypatch.setattr(gate, "_persist_winner", lambda: False)
    failed = asyncio.run(
        gate.claim_token("t", 7, publish_dir=pub, claim_id="B", manifest_revision="r")
    )
    assert failed == SEED_DOLE_PERSIST_FAILED
    marker = rearm.with_suffix(rearm.suffix + ".consuming")
    assert marker.exists(), "failed persistence discarded the durable rearm intent"
    assert gate._winners["t"]["grant_token"] == first

    restarted = _SeedDoleGate(state, lease_seconds=10, clock=clock)
    replacement = asyncio.run(
        restarted.claim_token(
            "t", 7, publish_dir=pub, claim_id="C", manifest_revision="r",
        )
    )
    assert replacement
    assert replacement != first
    assert not marker.exists()


def test_interrupted_active_restore_recovers_consuming_marker_after_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    pub = tmp_path / "publish"
    gate = _SeedDoleGate(tmp_path / "seed_dole_gate.json", lease_seconds=10, clock=clock)
    first = asyncio.run(gate.claim_token("t", 7, claim_id="A", manifest_revision="r"))
    rearm = _write_rearm(pub, 7, first)

    def _fail_link(_source: object, _destination: object) -> None:
        raise OSError(5, "simulated interrupted restore")

    monkeypatch.setattr(server_app.os, "link", _fail_link)
    assert (
        asyncio.run(
            gate.claim_token(
                "t", 7, publish_dir=pub, claim_id="B", manifest_revision="r",
            )
        )
        == ""
    )
    marker = rearm.with_suffix(rearm.suffix + ".consuming")
    assert not rearm.exists()
    assert marker.exists(), "interrupted restore lost the durable rearm marker"

    clock.advance(11)
    replacement = asyncio.run(
        gate.claim_token("t", 7, publish_dir=pub, claim_id="C", manifest_revision="r")
    )
    assert replacement
    assert replacement != first
    assert not marker.exists()


def test_ack_capable_uninstalled_replay_does_not_renew_forever(tmp_path: Path) -> None:
    clock = _Clock()
    pub = tmp_path / "publish"
    gate = _SeedDoleGate(tmp_path / "seed_dole_gate.json", lease_seconds=10, clock=clock)
    first, issued = asyncio.run(
        gate.claim_result(
            "t", 7, claim_id="A", manifest_revision="r", ack_capable=True,
        )
    )
    assert first
    assert issued is True
    original_expiry = gate._winners["t"]["lease_expires_at_unix"]

    clock.advance(9)
    rearm = _write_rearm(pub, 7, first)
    replay = asyncio.run(
        gate.claim_result(
            "t",
            7,
            publish_dir=pub,
            claim_id="A",
            manifest_revision="r",
            ack_capable=True,
        )
    )
    assert replay == (first, False)
    assert gate._winners["t"]["lease_expires_at_unix"] == original_expiry
    assert rearm.exists()

    clock.advance(2)
    replacement, newly_issued = asyncio.run(
        gate.claim_result(
            "t",
            7,
            publish_dir=pub,
            claim_id="B",
            manifest_revision="r",
            ack_capable=True,
        )
    )
    assert replacement
    assert replacement != first
    assert newly_issued is True


def test_expired_generation_rearms_exactly_once(tmp_path: Path) -> None:
    clock = _Clock()
    pub = tmp_path / "publish"
    gate = _SeedDoleGate(tmp_path / "seed_dole_gate.json", lease_seconds=10, clock=clock)
    first = asyncio.run(gate.claim_token("t", 7, claim_id="A", manifest_revision="r"))
    clock.advance(11)
    _write_rearm(pub, 7, first)
    second = asyncio.run(
        gate.claim_token("t", 7, publish_dir=pub, claim_id="B", manifest_revision="r")
    )
    assert second
    assert second != first
    assert asyncio.run(gate.claim_token("t", 7, claim_id="C", manifest_revision="r")) == ""


def test_legacy_owner_replay_wins_expiry_boundary_race(tmp_path: Path) -> None:
    clock = _Clock()
    pub = tmp_path / "publish"
    gate = _SeedDoleGate(tmp_path / "seed_dole_gate.json", lease_seconds=10, clock=clock)
    first = asyncio.run(gate.claim_token("t", 7, claim_id="A", manifest_revision="r"))
    clock.advance(11)
    rearm = _write_rearm(pub, 7, first)
    assert (
        asyncio.run(
            gate.claim_token(
                "t", 7, publish_dir=pub, claim_id="A", manifest_revision="r",
            )
        )
        == first
    )
    assert rearm.exists()
    assert asyncio.run(gate.claim_token("t", 7, claim_id="B", manifest_revision="r")) == ""


def test_generation_ack_survives_same_iteration_manifest_republish(tmp_path: Path) -> None:
    clock = _Clock()
    pub = tmp_path / "publish"
    gate = _SeedDoleGate(tmp_path / "seed_dole_gate.json", lease_seconds=10, clock=clock)
    first = asyncio.run(
        gate.claim_token(
            "t", 7, claim_id="A", manifest_revision="r1", ack_capable=True,
        )
    )
    clock.advance(11)
    rearm = _write_rearm(pub, 7, first)
    # Republish rotates both revision and worker claim_id. Possession/ACK of G
    # must still prove this is the live owner of G and renew the lease before
    # the pending rearm is judged.
    result = asyncio.run(
        gate.claim_result(
            "t",
            7,
            publish_dir=pub,
            claim_id="B",
            manifest_revision="r2",
            ack_grant_token=first,
            ack_capable=True,
        )
    )
    assert result == (first, False)
    assert rearm.exists()
    assert asyncio.run(gate.claim_token("t", 7, claim_id="C", manifest_revision="r2")) == ""


def test_stale_generation_rearm_cannot_roll_back_new_winner(tmp_path: Path) -> None:
    clock = _Clock()
    pub = tmp_path / "publish"
    gate = _SeedDoleGate(tmp_path / "seed_dole_gate.json", lease_seconds=10, clock=clock)
    first = asyncio.run(gate.claim_token("t", 7, claim_id="A", manifest_revision="r"))
    second = asyncio.run(
        gate.claim_token(
            "t", 7, claim_id="B", manifest_revision="r", allow_rearm=True,
        )
    )
    assert second != first
    clock.advance(11)
    _write_rearm(pub, 7, first)
    assert (
        asyncio.run(
            gate.claim_token(
                "t", 7, publish_dir=pub, claim_id="C", manifest_revision="r",
            )
        )
        == ""
    )
    assert asyncio.run(gate.claim_token("t", 7, claim_id="B", manifest_revision="r")) == second


def test_ack_is_generation_bound_and_durable(tmp_path: Path) -> None:
    clock = _Clock()
    state = tmp_path / "seed_dole_gate.json"
    gate = _SeedDoleGate(state, lease_seconds=10, clock=clock)
    first = asyncio.run(
        gate.claim_token(
            "t", 7, claim_id="A", manifest_revision="r", ack_capable=True,
        )
    )
    clock.advance(1)
    result = asyncio.run(
        gate.claim_result(
            "t",
            7,
            claim_id="A",
            manifest_revision="r",
            ack_grant_token=first,
            ack_capable=True,
        )
    )
    assert result == (first, False)
    sidecar = state.with_suffix(state.suffix + ".winners.json")
    record = json.loads(sidecar.read_text(encoding="utf-8"))["t"]
    assert record["acknowledged_at_unix"] == clock.now
    assert record["lease_expires_at_unix"] == clock.now + 10

    second = asyncio.run(
        gate.claim_token(
            "t", 7, claim_id="B", manifest_revision="r", allow_rearm=True,
        )
    )
    assert second != first
    asyncio.run(
        gate.claim_result(
            "t",
            7,
            claim_id="A",
            manifest_revision="r",
            ack_grant_token=first,
            ack_capable=True,
        )
    )
    replacement = json.loads(sidecar.read_text(encoding="utf-8"))["t"]
    assert replacement["grant_token"] == second
    assert "acknowledged_at_unix" not in replacement


def test_ack_only_renews_an_installed_generation_but_cannot_issue_one(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    pub = tmp_path / "publish"
    gate = _SeedDoleGate(
        tmp_path / "seed_dole_gate.json", lease_seconds=10, clock=clock,
    )
    first = asyncio.run(
        gate.claim_token(
            "t", 7, claim_id="A", manifest_revision="r", ack_capable=True,
        )
    )
    original_expiry = gate._winners["t"]["lease_expires_at_unix"]
    clock.advance(11)
    rearm = _write_rearm(pub, 7, first)

    heartbeat = asyncio.run(
        gate.claim_result(
            "t",
            7,
            publish_dir=pub,
            claim_id="A",
            manifest_revision="r",
            ack_grant_token=first,
            ack_capable=True,
            renew_only=True,
        )
    )
    assert heartbeat == (first, False)
    assert gate._winners["t"]["lease_expires_at_unix"] > original_expiry
    assert rearm.exists(), "ACK-only renewal consumed the live generation's rearm"

    empty_gate = _SeedDoleGate(tmp_path / "empty.json", lease_seconds=10, clock=clock)
    refused = asyncio.run(
        empty_gate.claim_result(
            "t",
            7,
            claim_id="B",
            manifest_revision="r",
            ack_grant_token="not-a-generation",
            ack_capable=True,
            renew_only=True,
        )
    )
    assert refused == ("", False)
    assert empty_gate._winners == {}


def test_worker_piggybacks_applied_generation_ack(tmp_path: Path) -> None:
    fake = _FakeRequests()
    session = _session(tmp_path, fake)
    session._live_dole_queue = []
    session._maybe_ingest_dole_flag(_manifest())
    assert fake.posts[0]["supports_seed_dole_ack"] is True
    assert "ack_grant_token" not in fake.posts[0]
    session._maybe_ingest_dole_flag(_manifest())
    assert fake.posts[1]["supports_seed_dole_ack"] is True
    assert fake.posts[1]["ack_grant_token"] == "A-tok1"


def test_trainable_rearm_is_fenced_by_durable_generation(tmp_path: Path) -> None:
    gate = tmp_path / "seed_dole_gate.json"
    winners = gate.with_suffix(gate.suffix + ".winners.json")
    gate.write_text(json.dumps({"trial_00000": 12}), encoding="utf-8")
    winners.write_text(
        json.dumps(
            {
                "trial_00000": {
                    "iteration": 12,
                    "claim_id": "A",
                    "revision": "r",
                    "grant_token": "gen-12",
                    "lease_expires_at_unix": 9999.0,
                }
            }
        ),
        encoding="utf-8",
    )
    assert (
        _seed_dole_generation_for_iteration(
            server_root=tmp_path,
            trial_id="trial_00000",
            training_iteration=12,
        )
        == "gen-12"
    )
    assert _seed_dole_rearm_payload(
        server_root=tmp_path,
        trial_id="trial_00000",
        training_iteration=12,
    ) == {"training_iteration": 12, "grant_token": "gen-12"}

    winners.unlink()
    assert (
        _seed_dole_rearm_payload(
            server_root=tmp_path,
            trial_id="trial_00000",
            training_iteration=12,
        )
        is None
    )
