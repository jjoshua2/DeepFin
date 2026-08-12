"""A won seed grant is APPLIED exactly once, however many times it is replayed.

⚑⚑ WHY THE SERVER-SIDE TESTS CANNOT COVER THIS. The server's gate deliberately
answers `granted=true` EVERY time for the winning (iteration, revision,
claim_id) -- that replay is what recovers a dropped response. The worker reuses
one claim_id for the whole key, and the mid-session poll path calls
`_maybe_ingest_dole_flag` every ~30s. Compose those two correct behaviours and
the winner re-wins on every poll, re-running an install that does
`live.clear()` then `extend`: an undrained queue is reset to the head of the
batch every poll so its tail is never reached, and a drained one replays the
same seeds all iteration.

Every server-side dole test passes in that state, because the server's
one-winner invariant is genuinely intact. Only a worker-level test sees it.
"""
from __future__ import annotations

from pathlib import Path

from tests.test_worker_model_update import (
    _DOLE_FEN_A,
    _DOLE_FEN_B,
    _dole_session_with_list,
)

REV = "rev-abc"


def _manifest(*, iteration: int = 7, revision: str = REV, per_iter: int = 1) -> dict:
    return {
        "training_iteration": iteration,
        "manifest_revision": revision,
        "seed_dole_claim_endpoint": "/v1/trials/trial_00000/seed_dole_claim",
        "recommended_worker": {"opening_fen_dole_per_iter": per_iter},
    }


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


class _FakeRequests:
    """Records claim POSTs and answers them the way the real gate would.

    The gate replays the winner, so this returns granted=True for the SAME
    claim_id every time -- reproducing precisely the server behaviour that
    makes over-application possible.
    """

    def __init__(self, *, fail_first: bool = False) -> None:
        self.posts: list[dict] = []
        # Keyed by revision, mirroring the real gate: a new opportunity
        # re-opens and can have a new winner. Keying a single winner globally
        # would make the fake refuse every later opportunity, which is a defect
        # in the double rather than in the worker.
        self.winners: dict[str, str] = {}
        self._seq: dict[str, int] = {}
        self._n = 0
        self._fail_first = fail_first

    def post(self, _url: str, *, json: dict, **_kwargs) -> _FakeResponse:
        self.posts.append(dict(json))
        cid = str(json.get("claim_id") or "")
        rev = str(json.get("manifest_revision") or "")
        granted = cid == self.winners.setdefault(rev, cid)
        if self._fail_first and len(self.posts) == 1:
            # The grant IS committed server-side; only the response is lost.
            raise ConnectionError("connection reset after the server committed")
        seq = self._seq.setdefault(rev, 0) or 0
        if granted and not seq:
            self._n += 1
            seq = self._n
            self._seq[rev] = seq
        return _FakeResponse({"granted": granted, "grant_seq": seq if granted else 0,
                              "reason_code": "granted"})


def _session(tmp_path: Path, requests):
    session = _dole_session_with_list(tmp_path)
    session._requests = requests
    session._auth = ("u", "p")
    session.server = "http://testserver"
    session._dole_claim_key = None
    session._dole_claim_id = ""
    session._applied_dole_seq = 0
    session._legacy_dole_seq = 0
    return session


def test_two_successful_polls_apply_the_seeds_exactly_once(tmp_path: Path) -> None:
    """THE regression. Both polls win at the server; only one may install."""
    fake = _FakeRequests()
    session = _session(tmp_path, fake)
    session._live_dole_queue = []

    session._maybe_ingest_dole_flag(_manifest())
    assert session._live_dole_queue == [_DOLE_FEN_A, _DOLE_FEN_B]

    # Drain, as the running selfplay would.
    session._live_dole_queue.clear()

    session._maybe_ingest_dole_flag(_manifest())
    assert session._live_dole_queue == [], (
        "the same grant was applied twice; the seed batch replays every poll"
    )
    # ⚑ It DOES re-POST, deliberately, and that is not the bug. Only the server
    # knows whether a rearm re-opened the gate, so the worker has to keep
    # asking; what must not repeat is the APPLICATION. An earlier design
    # suppressed the POST itself and thereby broke same-iteration rearm.
    assert len(fake.posts) == 2, f"the worker stopped asking: {fake.posts}"
    assert fake.posts[0]["claim_id"] == fake.posts[1]["claim_id"]


def test_an_undrained_queue_is_not_reset_to_the_head_of_the_batch(tmp_path: Path) -> None:
    """The nastier half: re-application does `clear()` before `extend()`, so a
    queue that has not drained within a poll interval would lose its progress
    and the tail of the batch would never be played."""
    fake = _FakeRequests()
    session = _session(tmp_path, fake)
    session._live_dole_queue = []

    session._maybe_ingest_dole_flag(_manifest())
    # Partially drained: the first seed has been consumed.
    session._live_dole_queue.pop(0)
    assert session._live_dole_queue == [_DOLE_FEN_B]

    session._maybe_ingest_dole_flag(_manifest())
    assert session._live_dole_queue == [_DOLE_FEN_B], (
        "an undrained queue was reset to the head of the batch"
    )


def test_a_dropped_response_retries_and_still_applies_exactly_once(tmp_path: Path) -> None:
    """⚑ The case idempotency exists for, checked end to end at the worker.

    The first POST commits server-side and the response is lost, so nothing is
    installed and the applied-key must stay UNSET. The retry must reuse the
    same claim_id (or the server would treat it as a different worker and
    refuse), win the replay, and install the batch exactly once.
    """
    fake = _FakeRequests(fail_first=True)
    session = _session(tmp_path, fake)
    session._live_dole_queue = []

    session._maybe_ingest_dole_flag(_manifest())  # response lost
    assert session._live_dole_queue == [], "installed despite never seeing a grant"
    assert session._applied_dole_seq == 0, (
        "marked applied on a claim whose outcome was never observed -- the "
        "grant would be spent and the seeds never played"
    )

    session._maybe_ingest_dole_flag(_manifest())  # retry
    assert session._live_dole_queue == [_DOLE_FEN_A, _DOLE_FEN_B]
    assert len(fake.posts) == 2
    assert fake.posts[0]["claim_id"] == fake.posts[1]["claim_id"], (
        "the retry used a fresh claim_id, which the server reads as a "
        "different worker; the dropped grant would be unrecoverable"
    )

    session._live_dole_queue.clear()
    session._maybe_ingest_dole_flag(_manifest())
    assert session._live_dole_queue == [], "applied a second time after recovery"


def test_a_failed_fen_load_does_not_mark_the_grant_applied(tmp_path: Path) -> None:
    """A LOCAL failure after a real grant must remain retryable, or the dose is
    spent server-side and silently never played."""
    fake = _FakeRequests()
    session = _session(tmp_path, fake)
    session._live_dole_queue = []
    session.opening_fen_list_path = str(tmp_path / "does-not-exist.txt")

    session._maybe_ingest_dole_flag(_manifest())
    assert session._applied_dole_seq == 0
    assert session._live_dole_queue == []


def test_a_new_iteration_applies_again(tmp_path: Path) -> None:
    """Applied-once is per opportunity, not forever."""
    fake = _FakeRequests()
    session = _session(tmp_path, fake)
    session._live_dole_queue = []

    session._maybe_ingest_dole_flag(_manifest(iteration=7))
    session._live_dole_queue.clear()
    session._maybe_ingest_dole_flag(_manifest(iteration=8, revision="rev-next"))
    assert session._live_dole_queue == [_DOLE_FEN_A, _DOLE_FEN_B]


def test_a_same_iteration_republish_is_a_new_opportunity(tmp_path: Path) -> None:
    """A changed revision within one iteration is a different manifest, so the
    rearm path can legitimately re-dole it."""
    fake = _FakeRequests()
    session = _session(tmp_path, fake)
    session._live_dole_queue = []

    session._maybe_ingest_dole_flag(_manifest(iteration=7, revision="rev-a"))
    session._live_dole_queue.clear()
    session._maybe_ingest_dole_flag(_manifest(iteration=7, revision="rev-b"))
    assert session._live_dole_queue == [_DOLE_FEN_A, _DOLE_FEN_B]
    assert fake.posts[0]["claim_id"] != fake.posts[1]["claim_id"], (
        "a new opportunity reused the old claim_id"
    )
