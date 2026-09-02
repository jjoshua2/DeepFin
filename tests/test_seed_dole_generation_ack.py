"""Concurrency contract for seed-dole generations, ACKs and recovery leases."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from chess_anti_engine.server.app import SEED_DOLE_REARM_FILENAME, _SeedDoleGate
from chess_anti_engine.tune.distributed_runtime import _seed_dole_generation_for_iteration, _seed_dole_rearm_payload
from tests.test_worker_dole_applied_once import _FakeRequests, _manifest, _session

class _Clock:
    def __init__(self, now: float = 1000.0) -> None: self.now = now
    def __call__(self) -> float: return self.now
    def advance(self, seconds: float) -> None: self.now += seconds

def _write_rearm(pub: Path, iteration: int, token: str) -> None:
    pub.mkdir(parents=True, exist_ok=True)
    (pub / SEED_DOLE_REARM_FILENAME).write_text(json.dumps({"training_iteration": iteration, "grant_token": token}), encoding="utf-8")

def test_live_owner_heartbeat_blocks_same_iteration_redole(tmp_path: Path) -> None:
    clock = _Clock(); pub = tmp_path / "publish"
    gate = _SeedDoleGate(tmp_path / "seed_dole_gate.json", lease_seconds=10, clock=clock)
    token = asyncio.run(gate.claim_token("t", 7, claim_id="A", manifest_revision="r")); assert token
    clock.advance(9); _write_rearm(pub, 7, token)
    assert asyncio.run(gate.claim_token("t", 7, publish_dir=pub, claim_id="B", manifest_revision="r")) == ""
    assert not (pub / SEED_DOLE_REARM_FILENAME).exists()
    assert asyncio.run(gate.claim_token("t", 7, claim_id="A", manifest_revision="r")) == token

def test_expired_generation_rearms_exactly_once(tmp_path: Path) -> None:
    clock = _Clock(); pub = tmp_path / "publish"
    gate = _SeedDoleGate(tmp_path / "seed_dole_gate.json", lease_seconds=10, clock=clock)
    first = asyncio.run(gate.claim_token("t", 7, claim_id="A", manifest_revision="r")); clock.advance(11); _write_rearm(pub, 7, first)
    second = asyncio.run(gate.claim_token("t", 7, publish_dir=pub, claim_id="B", manifest_revision="r"))
    assert second and second != first
    assert asyncio.run(gate.claim_token("t", 7, claim_id="C", manifest_revision="r")) == ""

def test_owner_replay_wins_expiry_boundary_race(tmp_path: Path) -> None:
    clock = _Clock(); pub = tmp_path / "publish"
    gate = _SeedDoleGate(tmp_path / "seed_dole_gate.json", lease_seconds=10, clock=clock)
    first = asyncio.run(gate.claim_token("t", 7, claim_id="A", manifest_revision="r")); clock.advance(11); _write_rearm(pub, 7, first)
    assert asyncio.run(gate.claim_token("t", 7, publish_dir=pub, claim_id="A", manifest_revision="r")) == first
    assert asyncio.run(gate.claim_token("t", 7, claim_id="B", manifest_revision="r")) == ""

def test_generation_ack_survives_same_iteration_manifest_republish(tmp_path: Path) -> None:
    clock = _Clock(); pub = tmp_path / "publish"
    gate = _SeedDoleGate(tmp_path / "seed_dole_gate.json", lease_seconds=10, clock=clock)
    first = asyncio.run(gate.claim_token("t", 7, claim_id="A", manifest_revision="r1")); clock.advance(11); _write_rearm(pub, 7, first)
    # Republish rotates both revision and worker claim_id. Possession/ACK of G
    # must still prove this is the live owner of G and renew the lease before
    # the pending rearm is judged. This is the production residual.
    assert asyncio.run(gate.claim_result("t", 7, publish_dir=pub, claim_id="B", manifest_revision="r2", ack_grant_token=first)) == (first, False)
    assert not (pub / SEED_DOLE_REARM_FILENAME).exists()
    assert asyncio.run(gate.claim_token("t", 7, claim_id="C", manifest_revision="r2")) == ""

def test_stale_generation_rearm_cannot_roll_back_new_winner(tmp_path: Path) -> None:
    clock = _Clock(); pub = tmp_path / "publish"
    gate = _SeedDoleGate(tmp_path / "seed_dole_gate.json", lease_seconds=10, clock=clock)
    first = asyncio.run(gate.claim_token("t", 7, claim_id="A", manifest_revision="r"))
    second = asyncio.run(gate.claim_token("t", 7, claim_id="B", manifest_revision="r", allow_rearm=True)); assert second != first
    clock.advance(11); _write_rearm(pub, 7, first)
    assert asyncio.run(gate.claim_token("t", 7, publish_dir=pub, claim_id="C", manifest_revision="r")) == ""
    assert asyncio.run(gate.claim_token("t", 7, claim_id="B", manifest_revision="r")) == second

def test_ack_is_generation_bound_and_durable(tmp_path: Path) -> None:
    clock = _Clock(); state = tmp_path / "seed_dole_gate.json"
    gate = _SeedDoleGate(state, lease_seconds=10, clock=clock)
    first = asyncio.run(gate.claim_token("t", 7, claim_id="A", manifest_revision="r")); clock.advance(1)
    assert asyncio.run(gate.claim_result("t", 7, claim_id="A", manifest_revision="r", ack_grant_token=first)) == (first, False)
    sidecar = state.with_suffix(state.suffix + ".winners.json"); rec = json.loads(sidecar.read_text())["t"]
    assert rec["acknowledged_at_unix"] == clock.now and rec["lease_expires_at_unix"] == clock.now + 10
    second = asyncio.run(gate.claim_token("t", 7, claim_id="B", manifest_revision="r", allow_rearm=True)); assert second != first
    asyncio.run(gate.claim_result("t", 7, claim_id="A", manifest_revision="r", ack_grant_token=first))
    rec2 = json.loads(sidecar.read_text())["t"]; assert rec2["grant_token"] == second and "acknowledged_at_unix" not in rec2

def test_worker_piggybacks_applied_generation_ack(tmp_path: Path) -> None:
    fake = _FakeRequests(); session = _session(tmp_path, fake); session._live_dole_queue = []
    session._maybe_ingest_dole_flag(_manifest()); assert "ack_grant_token" not in fake.posts[0]
    session._maybe_ingest_dole_flag(_manifest()); assert fake.posts[1]["ack_grant_token"] == "A-tok1"

def test_trainable_rearm_is_fenced_by_durable_generation(tmp_path: Path) -> None:
    gate = tmp_path / "seed_dole_gate.json"; winners = gate.with_suffix(gate.suffix + ".winners.json")
    gate.write_text(json.dumps({"trial_00000": 12}), encoding="utf-8")
    winners.write_text(json.dumps({"trial_00000": {"iteration": 12, "claim_id": "A", "revision": "r", "grant_token": "gen-12", "lease_expires_at_unix": 9999.0}}), encoding="utf-8")
    assert _seed_dole_generation_for_iteration(server_root=tmp_path, trial_id="trial_00000", training_iteration=12) == "gen-12"
    assert _seed_dole_rearm_payload(server_root=tmp_path, trial_id="trial_00000", training_iteration=12) == {"training_iteration": 12, "grant_token": "gen-12"}
    winners.unlink(); assert _seed_dole_rearm_payload(server_root=tmp_path, trial_id="trial_00000", training_iteration=12) is None
