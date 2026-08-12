"""The seed-dole grant must survive a lost HTTP response.

⚑ WHY THIS EXISTS. PR5 moves the claim out of the manifest GET into its own
POST. The grant is persisted server-side and only THEN sent, so a connection
that dies after the write leaves the server believing the dole was handed out
while the worker never learned it won. Without idempotency the retry hits the
monotonic gate, loses, and the single seed opportunity for that iteration is
gone -- which is precisely the "server thinks seeding happened and it didn't"
failure the change exists to fix, reintroduced by the extra round trip the
change itself adds.

The old GET+claim path had the same response-loss hole; it is fixed here rather
than inherited because this change is what makes the window a real one.
"""
from __future__ import annotations

import asyncio
import json

from chess_anti_engine.server.app import _SeedDoleGate

REV = "rev-abc"


def test_same_claim_id_retry_is_still_granted(tmp_path) -> None:
    """THE case: response dropped, worker retries with the same claim_id."""
    g = _SeedDoleGate(state_path=tmp_path / "seed_dole_gate.json")
    assert asyncio.run(g.claim("t", 10, claim_id="A", manifest_revision=REV)) is True
    assert asyncio.run(g.claim("t", 10, claim_id="A", manifest_revision=REV)) is True


def test_a_different_worker_still_loses(tmp_path) -> None:
    """⚑ NEGATIVE CONTROL. Idempotency must not become "everyone wins" -- that
    would hand the same seed batch to every worker that asks."""
    g = _SeedDoleGate(state_path=tmp_path / "seed_dole_gate.json")
    assert asyncio.run(g.claim("t", 10, claim_id="A", manifest_revision=REV)) is True
    assert asyncio.run(g.claim("t", 10, claim_id="B", manifest_revision=REV)) is False


def test_replay_against_a_different_revision_is_not_the_same_request(tmp_path) -> None:
    """The winner replaying with a DIFFERENT manifest revision is asking for
    something else and must not inherit its earlier win."""
    g = _SeedDoleGate(state_path=tmp_path / "seed_dole_gate.json")
    assert asyncio.run(g.claim("t", 10, claim_id="A", manifest_revision=REV)) is True
    assert asyncio.run(g.claim("t", 10, claim_id="A", manifest_revision="rev-other")) is False


def test_a_new_iteration_has_a_new_winner(tmp_path) -> None:
    g = _SeedDoleGate(state_path=tmp_path / "seed_dole_gate.json")
    assert asyncio.run(g.claim("t", 10, claim_id="A", manifest_revision=REV)) is True
    assert asyncio.run(g.claim("t", 11, claim_id="B", manifest_revision=REV)) is True
    # And the previous winner cannot replay a superseded iteration.
    assert asyncio.run(g.claim("t", 10, claim_id="A", manifest_revision=REV)) is False


def test_idempotency_survives_a_server_restart(tmp_path) -> None:
    """The dropped response and the restart can be the SAME incident, so the
    winner has to be durable, not in-memory."""
    path = tmp_path / "seed_dole_gate.json"
    g = _SeedDoleGate(state_path=path)
    assert asyncio.run(g.claim("t", 10, claim_id="A", manifest_revision=REV)) is True

    g2 = _SeedDoleGate(state_path=path)
    assert asyncio.run(g2.claim("t", 10, claim_id="A", manifest_revision=REV)) is True
    assert asyncio.run(g2.claim("t", 10, claim_id="B", manifest_revision=REV)) is False


def test_the_legacy_no_claim_id_path_is_unchanged(tmp_path) -> None:
    """⚑ Every existing caller passes no claim_id. Their behaviour must be
    byte-identical, or this "additive" change is a live behaviour change to the
    running dole."""
    g = _SeedDoleGate(state_path=tmp_path / "seed_dole_gate.json")
    assert asyncio.run(g.claim("t", 5)) is True
    assert asyncio.run(g.claim("t", 5)) is False
    assert asyncio.run(g.claim("t", 6)) is True


def test_winners_live_in_a_sidecar_not_the_state_file(tmp_path) -> None:
    """⚑⚑ REGRESSION GUARD FOR A TRAP. The state file's loader does `int(v)`
    over EVERY value, so a nested winner entry inside it would raise, the whole
    load would fall back to `{}`, and the gate would FORGET which iterations
    were claimed -- re-granting a dole already handed out. The winner record
    therefore lives in its own file and the state file stays flat.
    """
    path = tmp_path / "seed_dole_gate.json"
    g = _SeedDoleGate(state_path=path)
    assert asyncio.run(g.claim("t", 10, claim_id="A", manifest_revision=REV)) is True

    state = json.loads(path.read_text())
    assert state == {"t": 10}, f"state file is no longer a flat trial->int map: {state}"
    # Prove the flat map still loads under the original strict reader.
    assert {str(k): int(v) for k, v in state.items()} == {"t": 10}

    sidecar = path.with_suffix(path.suffix + ".winners.json")
    assert sidecar.exists(), "winner record was not persisted"
    assert json.loads(sidecar.read_text())["t"]["claim_id"] == "A"


def test_a_missing_sidecar_costs_idempotency_not_correctness(tmp_path) -> None:
    """Losing the sidecar must never RE-GRANT: a retry looks like a new claim
    and loses.

    ⚑ STATED HONESTLY, because an earlier version of this docstring oversold
    it as "costs idempotency, never correctness". That is true only if
    correctness means "never double-grant". Relative to THIS PR's actual
    invariant -- the seeds get played -- losing the sidecar while the gate is
    spent silently loses that iteration's dose, which is the very failure the
    idempotency exists to prevent.

    What makes the guarantee hold is the WRITE ORDER in `_persist`, not this
    fallback: the winner is written BEFORE the gate, so a crash between them
    leaves the gate unspent and the retry simply re-races. This test covers the
    residual case where the sidecar is lost independently (deleted, or its
    write failed while the gate's succeeded), and pins the safe direction for
    it. The restart guarantee is "survives a restart once the winner record is
    durable", which the ordering makes the common case."""
    path = tmp_path / "seed_dole_gate.json"
    g = _SeedDoleGate(state_path=path)
    assert asyncio.run(g.claim("t", 10, claim_id="A", manifest_revision=REV)) is True
    path.with_suffix(path.suffix + ".winners.json").unlink()

    g2 = _SeedDoleGate(state_path=path)
    assert asyncio.run(g2.claim("t", 10, claim_id="A", manifest_revision=REV)) is False
