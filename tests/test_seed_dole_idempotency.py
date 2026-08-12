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


def test_a_crash_between_the_two_files_cannot_double_grant(tmp_path) -> None:
    """⚑⚑ THE CRASH WINDOW, AND IT BIT THE FIRST FIX.

    `_persist` writes the winner sidecar before the gate, so the recoverable
    outcome sits in the window. But recovery is not automatic: on restart the
    gate said 9 while a durable winner for iteration 10 existed, so A's replay
    was granted AND B then passed `10 > 9` and was granted too -- MEASURED.
    Reversing the write order had converted a lost dose into a DOUBLE grant,
    which is worse.

    A durable winner record means that iteration WAS handed out, so the loader
    reconciles it into the gate.
    """
    path = tmp_path / "seed_dole_gate.json"
    path.write_text(json.dumps({"t": 9}), encoding="utf-8")
    path.with_suffix(path.suffix + ".winners.json").write_text(
        json.dumps({"t": {"iteration": 10, "claim_id": "A", "revision": REV, "grant_token": "tok-old"}}),
        encoding="utf-8",
    )

    g = _SeedDoleGate(state_path=path)
    assert asyncio.run(g.claim("t", 10, claim_id="A", manifest_revision=REV)) is True
    assert asyncio.run(g.claim("t", 10, claim_id="B", manifest_revision=REV)) is False, (
        "a crash between the winner and gate writes let a second worker claim the same dose"
    )


def test_a_replay_returns_the_same_grant_token(tmp_path) -> None:
    """The token is the worker's "is this a new dose" answer, so a replay must
    not look like a new one."""
    g = _SeedDoleGate(state_path=tmp_path / "seed_dole_gate.json")
    first = asyncio.run(g.claim_token("t", 10, claim_id="A", manifest_revision=REV))
    assert first
    assert asyncio.run(g.claim_token("t", 10, claim_id="A", manifest_revision=REV)) == first


def test_a_rearm_issues_a_new_grant_token(tmp_path) -> None:
    """⚑ And the rearm must retire the old winner, or its replay would
    short-circuit and hand back the OLD sequence -- so the worker would skip
    the rearmed dose entirely."""
    g = _SeedDoleGate(state_path=tmp_path / "seed_dole_gate.json")
    first = asyncio.run(g.claim_token("t", 20, claim_id="A", manifest_revision=REV))
    rearmed = asyncio.run(
        g.claim_token("t", 20, claim_id="A", manifest_revision=REV, allow_rearm=True),
    )
    assert rearmed, "the rearm did not grant at all"
    assert rearmed != first, (
        "a rearmed dose replayed the previous grant token, so the worker would "
        "treat the legitimately re-opened batch as one it had already applied"
    )


def test_a_grant_is_not_acknowledged_when_the_winner_cannot_be_persisted(tmp_path) -> None:
    """⚑⚑ WINNER DURABILITY IS A PRECONDITION FOR ACKNOWLEDGING, not merely
    for advancing the gate.

    An earlier version returned early from persistence on failure but still
    answered `granted=true, grant_seq=N`. The worker would then install and
    PLAY a dose the server had no durable record of; a crash left neither
    winner nor gate, and a second worker won the same dose. Refusing to
    acknowledge is what makes the failure a retry instead of a duplicate.
    """
    path = tmp_path / "seed_dole_gate.json"
    g = _SeedDoleGate(state_path=path)

    def _fail() -> bool:
        return False

    g._persist_winner = _fail  # type: ignore[method-assign]
    from chess_anti_engine.server.app import SEED_DOLE_PERSIST_FAILED

    token, new = asyncio.run(g.claim_result("t", 10, claim_id="A", manifest_revision=REV))
    assert new is False, "an undurable grant was acknowledged to the worker"
    # ⚑ A DISTINCT sentinel, not a bare "": the route turns this into a 503 so a
    # broken server root cannot masquerade as the ordinary "another worker won".
    assert token == SEED_DOLE_PERSIST_FAILED, token
    assert not path.exists(), "the gate was advanced for a grant that was never acknowledged"

    # ⚑ And the failed attempt must leave nothing behind that a replay could
    # shortcut on -- otherwise the retry inherits a win that was never durable.
    assert asyncio.run(
        g.claim_token("t", 10, claim_id="A", manifest_revision=REV),
    ) == SEED_DOLE_PERSIST_FAILED

    # Once persistence works again, exactly one winner emerges.
    del g._persist_winner  # type: ignore[attr-defined]
    assert asyncio.run(g.claim_token("t", 10, claim_id="A", manifest_revision=REV))
    assert asyncio.run(g.claim("t", 10, claim_id="B", manifest_revision=REV)) is False


def test_a_replay_is_not_reported_as_a_new_grant(tmp_path) -> None:
    """The winner re-POSTs every poll by design, so `newly_issued` is what the
    `seed dole GRANTED` log line must key on. Keyed on `granted`, the line
    would fire every ~30s and the ledger's "one line per iteration per trial"
    yardstick would be measuring something else entirely."""
    g = _SeedDoleGate(state_path=tmp_path / "seed_dole_gate.json")
    first = asyncio.run(g.claim_result("t", 10, claim_id="A", manifest_revision=REV))
    replay = asyncio.run(g.claim_result("t", 10, claim_id="A", manifest_revision=REV))
    assert first[1] is True
    assert first[0]
    assert replay == (first[0], False)


def test_losing_the_winner_sidecar_does_not_suppress_future_doses(tmp_path) -> None:
    """⚑⚑ SIDECAR LOSS MUST NOT POISON THE WORKER'S "ALREADY APPLIED" STATE.

    This was a real regression, and it is the reason grant identity is an
    OPAQUE TOKEN rather than a monotone counter.

    Sidecar loss is explicitly TOLERATED -- the gate keeps serving. But the
    per-trial counter was rebuilt ONLY from that sidecar, so losing it reset
    the numbering while the durable gate kept its iteration. MEASURED on the
    counter design: five grants for trial T, delete only `*.winners.json`,
    reload -- the gate still read iteration 10 and iteration 11 then issued
    seq 1. A long-running worker holding applied seq 5 would compare `1 <= 5`
    and SILENTLY SKIP the next five legitimate doses.

    The bug was in encoding an EQUALITY question ("is this the dose I already
    applied?") as an ORDERING one, which imported a dependency on the numbering
    never restarting. A token cannot regress: a fresh one is simply != the
    applied one, whatever the storage did.
    """
    path = tmp_path / "seed_dole_gate.json"
    g = _SeedDoleGate(state_path=path)
    tokens = [
        asyncio.run(g.claim_token("T", i, claim_id=f"c{i}", manifest_revision=REV))
        for i in range(6, 11)
    ]
    assert all(tokens), tokens
    assert len(set(tokens)) == 5, tokens

    # Lose ONLY the winner sidecar; the durable monotonic gate survives.
    path.with_suffix(path.suffix + ".winners.json").unlink()
    assert json.loads(path.read_text())["T"] == 10, "the gate itself was lost; wrong scenario"

    g2 = _SeedDoleGate(state_path=path)
    revived = asyncio.run(g2.claim_token("T", 11, claim_id="c11", manifest_revision=REV))
    assert revived, "the next iteration was not grantable at all after sidecar loss"
    assert revived not in tokens, (
        "a grant issued after sidecar loss reused an identity the worker has already "
        "applied, so a long-running worker would silently skip this dose"
    )


def test_the_winner_is_durable_before_the_gate_is(tmp_path) -> None:
    """⚑⚑ THE WRITE ORDER ITSELF, WHICH WAS COMMENT-ONLY UNTIL A REVIEW MUTATED
    IT AND NOTHING FAILED.

    Reverting `_persist` to gate-then-winner passed the whole suite, because the
    other durability tests exercise the RESULT of the ordering (reconciliation
    on load) and not the ordering. Gate-first puts the unrecoverable outcome in
    the crash window: the iteration is durably spent while the server no longer
    knows which claim_id won, so the retry is refused and the dose is lost.

    This observes the order directly, by recording when each file appears.
    """
    path = tmp_path / "seed_dole_gate.json"
    winners = path.with_suffix(path.suffix + ".winners.json")
    g = _SeedDoleGate(state_path=path)

    order: list[str] = []
    real_winner = g._persist_winner
    real_gate = g._persist_gate

    def _w() -> bool:
        order.append("winner")
        return real_winner()

    def _g() -> None:
        order.append("gate")
        real_gate()

    g._persist_winner = _w  # type: ignore[method-assign]
    g._persist_gate = _g  # type: ignore[method-assign]

    assert asyncio.run(g.claim_token("t", 10, claim_id="A", manifest_revision=REV))
    assert order == ["winner", "gate"], (
        f"persistence order is {order}; gate-first leaves a crash window in which the "
        "dose is spent with no recoverable winner record"
    )
    assert winners.exists()
    assert path.exists()


def test_an_id_less_claim_still_writes_a_winner_record(tmp_path) -> None:
    """⚑ The gate must have NO second code path. It used to skip the winner
    write entirely when `claim_id is None`, burning the iteration with nothing
    to replay -- and every pre-existing gate test ran down exactly that branch,
    so the one-winner invariant was only ever asserted on a path production
    never takes. The gate now synthesises an id instead of branching."""
    path = tmp_path / "seed_dole_gate.json"
    g = _SeedDoleGate(state_path=path)
    assert asyncio.run(g.claim_token("t", 10, claim_id=None, manifest_revision=REV))
    winners = json.loads(path.with_suffix(path.suffix + ".winners.json").read_text())
    assert winners["t"]["grant_token"], "no winner record was written for an id-less claim"
    assert winners["t"]["claim_id"].startswith("anon-")


def test_a_corrupt_winner_iteration_does_not_500_every_claim(tmp_path) -> None:
    """The replay check's `int()` was unguarded while the loader that wrote the
    value accepts anything. A non-parseable `iteration` would raise inside the
    handler and 500 EVERY claim for that trial forever, with the worker seeing
    only 'rejected with HTTP 500'."""
    path = tmp_path / "seed_dole_gate.json"
    path.write_text(json.dumps({"t": 9}), encoding="utf-8")
    path.with_suffix(path.suffix + ".winners.json").write_text(
        json.dumps({"t": {"iteration": "not-a-number", "claim_id": "A", "grant_token": "x"}}),
        encoding="utf-8",
    )
    g = _SeedDoleGate(state_path=path)
    assert asyncio.run(g.claim_token("t", 10, claim_id="A", manifest_revision=REV))
