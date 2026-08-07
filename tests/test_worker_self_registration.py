"""Trust-on-first-use registration, bans, and the throttles that make an open
port survivable.

⚑ THE FLAG IS THE FIRST THING TESTED, AND IT IS TESTED BY BEHAVIOUR. Today's
deployment is a closed LAN fleet; opening registration by accident would be a
security regression delivered as a feature. So `test_..._flag_off_...` and
`test_..._flag_on_...` drive the SAME request against the SAME server root and
assert the outcomes differ — proving the flag reaches the auth path, not merely
that it parses. A test that only checked `create_app(worker_self_register=True)`
does not raise would pass on a flag wired to nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_anti_engine.server.access import (
    BANS_FILENAME,
    AccessGuard,
    BanList,
    BannedIdentity,
    RateLimited,
)
from chess_anti_engine.server.auth import UserRecord, hash_password, save_users, verify_password

_HEADERS = {"X-CAE-Worker-Version": "0.0.0", "X-CAE-Protocol-Version": "1"}


def _seed(root: Path, username: str = "known", password: str = "pw") -> None:
    salt, hsh, iters = hash_password(password)
    save_users(
        root / "users.json",
        {username: UserRecord(username=username, salt_b64=salt, hash_b64=hsh, iterations=iters)},
    )


def _client(root: Path, *, self_register: bool):
    from fastapi.testclient import TestClient

    from chess_anti_engine.server.app import create_app

    return TestClient(
        create_app(
            server_root=str(root), users_db="users.json",
            worker_self_register=self_register,
        ),
        # The ASGI transport supplies a client host, which the IP-scoped
        # throttles and bans need in order to be exercised at all.
        client=("203.0.113.7", 5555),
    )


def _ping(client, user: str, password: str):
    return client.post(
        "/v1/report_bad_shard",
        json={"sha256": "0" * 64, "reason": "auth-probe"},
        auth=(user, password),
        headers=_HEADERS,
    )


# ---------------------------------------------------------------------------
# The flag, proven by behaviour on both sides.
# ---------------------------------------------------------------------------


def test_an_unknown_user_is_refused_with_the_flag_off(tmp_path: Path) -> None:
    """The closed world, unchanged. This is today's production behaviour and
    the thing that must not move."""
    _seed(tmp_path)
    client = _client(tmp_path, self_register=False)

    assert _ping(client, "stranger", "anything").status_code == 401
    assert "stranger" not in json.loads(
        (tmp_path / "users.json").read_text(encoding="utf-8")
    ), "no account may be created with the flag off"


def test_an_unknown_user_is_registered_with_the_flag_on(tmp_path: Path) -> None:
    """Same request, same server root, opposite outcome — so the flag reaches
    the auth path rather than merely parsing."""
    _seed(tmp_path)
    client = _client(tmp_path, self_register=True)

    assert _ping(client, "stranger", "their-own-secret").status_code not in (401, 403)

    raw = json.loads((tmp_path / "users.json").read_text(encoding="utf-8"))
    assert "stranger" in raw
    assert "their-own-secret" not in json.dumps(raw), "plaintext must never be stored"
    record = UserRecord(username="stranger", **raw["stranger"])
    assert verify_password("their-own-secret", record)
    assert not verify_password("", record)


def test_a_registered_name_then_requires_that_password(tmp_path: Path) -> None:
    """The whole point of trust-on-FIRST-use: the second use is not trusted."""
    _seed(tmp_path)
    client = _client(tmp_path, self_register=True)

    assert _ping(client, "stranger", "first-secret").status_code not in (401, 403)
    assert _ping(client, "stranger", "someone-elses-guess").status_code == 401
    assert _ping(client, "stranger", "first-secret").status_code not in (401, 403)


def test_the_existing_credential_path_is_untouched_in_both_worlds(tmp_path: Path) -> None:
    """A known user with the right password works, and with the wrong one gets
    401, whether the flag is on or off. Registration must not become a way
    around a real password."""
    _seed(tmp_path)
    for flag in (False, True):
        client = _client(tmp_path, self_register=flag)
        assert _ping(client, "known", "pw").status_code not in (401, 403)
        assert _ping(client, "known", "wrong").status_code == 401


# ---------------------------------------------------------------------------
# Bans
# ---------------------------------------------------------------------------


def test_a_banned_username_loses_every_authenticated_route(tmp_path: Path) -> None:
    """403 and not 401: 401 reads as "wrong password" and invites a retry loop."""
    _seed(tmp_path)
    client = _client(tmp_path, self_register=False)
    assert _ping(client, "known", "pw").status_code not in (401, 403)

    bans = BanList.load(tmp_path / BANS_FILENAME)
    bans.usernames.add("known")
    bans.notes["known"] = "uploading garbage"
    bans.save()

    resp = _ping(client, "known", "pw")
    assert resp.status_code == 403
    assert "uploading garbage" in resp.text
    assert "banned" in resp.text


def test_a_banned_ip_is_refused_even_with_a_valid_password(tmp_path: Path) -> None:
    _seed(tmp_path)
    client = _client(tmp_path, self_register=True)

    bans = BanList.load(tmp_path / BANS_FILENAME)
    bans.ips.add("203.0.113.7")
    bans.save()

    assert _ping(client, "known", "pw").status_code == 403
    # ...and cannot register around it.
    assert _ping(client, "brand-new-name", "x").status_code == 403
    assert "brand-new-name" not in (tmp_path / "users.json").read_text(encoding="utf-8")


def test_a_ban_takes_effect_without_a_server_restart(tmp_path: Path) -> None:
    """The operator runs manage_users.py in ANOTHER process. If the ban only
    landed at the next restart it would be useless at the moment it is needed."""
    _seed(tmp_path)
    client = _client(tmp_path, self_register=False)
    assert _ping(client, "known", "pw").status_code not in (401, 403)

    bans = BanList.load(tmp_path / BANS_FILENAME)
    bans.usernames.add("known")
    bans.save()
    assert _ping(client, "known", "pw").status_code == 403

    bans = BanList.load(tmp_path / BANS_FILENAME)
    bans.usernames.discard("known")
    bans.save()
    assert _ping(client, "known", "pw").status_code not in (401, 403)


def test_a_ban_is_checked_before_any_password_work(tmp_path: Path) -> None:
    """A banned identity must not be able to spend the server's PBKDF2 budget."""
    import chess_anti_engine.server.auth as auth_mod

    _seed(tmp_path)
    bans = BanList.load(tmp_path / BANS_FILENAME)
    bans.usernames.add("known")
    bans.save()
    client = _client(tmp_path, self_register=False)

    calls = 0
    real = auth_mod._pbkdf2

    def counted(password: str, *, salt: bytes, iterations: int) -> bytes:
        nonlocal calls
        calls += 1
        return real(password, salt=salt, iterations=iterations)

    auth_mod._pbkdf2 = counted
    try:
        assert _ping(client, "known", "pw").status_code == 403
    finally:
        auth_mod._pbkdf2 = real
    assert calls == 0, f"a banned identity ran PBKDF2 {calls} times"


# ---------------------------------------------------------------------------
# Throttles
# ---------------------------------------------------------------------------


def test_registration_is_rate_limited_per_ip(tmp_path: Path) -> None:
    """An open port that creates accounts on demand is an account-spam target."""
    from fastapi.testclient import TestClient

    from chess_anti_engine.server.app import create_app

    _seed(tmp_path)
    app = create_app(
        server_root=str(tmp_path), users_db="users.json", worker_self_register=True,
    )
    client = TestClient(app, client=("198.51.100.9", 4444))

    accepted = 0
    limited = 0
    for i in range(9):
  # 8 characters: below the password minimum these would all be refused
  # 400 and the rate limit would never be reached, which is how a policy
  # change can silently hollow out a limit test.
        code = _ping(client, f"vol{i}", f"secret-{i}").status_code
        if code == 429:
            limited += 1
        elif code not in (400, 401, 403):
            accepted += 1
    assert accepted == 5, f"expected the default limit of 5, got {accepted}"
    assert limited == 4
    raw = json.loads((tmp_path / "users.json").read_text(encoding="utf-8"))
    assert len([k for k in raw if k.startswith("vol")]) == 5


def test_failed_signins_are_throttled_and_cleared_by_a_success() -> None:
    """Asserted on the guard, because the HTTP path cannot show the CLEAR.

    The clear matters: a worker that restarts with a stale secret and then gets
    the right one must not be held out for the rest of the window by its own
    earlier retries.
    """
    guard = AccessGuard(Path("/nonexistent/bans.json"), failed_auth_limit=3)
    for _ in range(3):
        guard.check_auth_attempt_allowed("10.0.0.1")
        guard.note_auth_failure("10.0.0.1")
    with pytest.raises(RateLimited):
        guard.check_auth_attempt_allowed("10.0.0.1")

    # A different address is unaffected — the limit is per-IP, not global.
    guard.check_auth_attempt_allowed("10.0.0.2")

    guard.note_auth_success("10.0.0.1")
    guard.check_auth_attempt_allowed("10.0.0.1")
    assert guard.failed_auth_count("10.0.0.1") == 0


def test_the_guard_raises_rather_than_returning_a_verdict() -> None:
    """A caller cannot accept the answer and forget to act on it."""
    guard = AccessGuard(Path("/nonexistent/bans.json"))
    guard._bans.usernames.add("bad")
    with pytest.raises(BannedIdentity):
        guard.check_not_banned(username="bad", ip=None)


def test_an_absent_or_corrupt_ban_file_means_nobody_is_banned(tmp_path: Path) -> None:
    """Fails OPEN, deliberately, and the choice is argued in `BanList.load`:
    a ban list that refuses to start the server is worse than one that is
    empty, and `list-bans` shows the operator the truth either way."""
    assert BanList.load(tmp_path / "absent.json").usernames == set()
    corrupt = tmp_path / "bans.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert BanList.load(corrupt).usernames == set()


def test_a_corrupt_ban_file_is_audible_and_an_absent_one_is_not(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """⚑ F2: "FAILS OPEN" AND "FAILS OPEN SILENTLY" ARE DIFFERENT PROMISES.

    Only the first was intended. A `bans.json` that exists and will not parse
    unbans everybody, and it used to be indistinguishable from no file at all —
    the operator's only signal was `list-bans` printing nothing, which is also
    what an empty list prints. The behaviour stays fail-open (a LAN fleet must
    not lose its server to a bad ban file); what changes is that it says so.

    The absent case must stay silent, or the warning is noise on every fresh
    server root and gets tuned out before it ever matters.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.server.access"):
        BanList.load(tmp_path / "absent.json")
    assert caplog.records == [], "an absent ban list is the normal case"

    corrupt = tmp_path / "bans.json"
    corrupt.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.server.access"):
        assert BanList.load(corrupt).usernames == set()
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "NOBODY BANNED" in message
    assert str(corrupt) in message

    caplog.clear()
    wrong_shape = tmp_path / "list.json"
    wrong_shape.write_text('["someone"]', encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.server.access"):
        assert BanList.load(wrong_shape).usernames == set()
    assert len(caplog.records) == 1, (
        "a ban list that parses but is the wrong SHAPE unbans everybody too — "
        "a JSON array of names is the obvious way to write one by hand"
    )


def test_the_ban_file_survives_a_round_trip(tmp_path: Path) -> None:
    path = tmp_path / BANS_FILENAME
    bans = BanList.load(path)
    bans.usernames.add("spammer")
    bans.ips.add("1.2.3.4")
    bans.notes["spammer"] = "why"
    bans.save()

    again = BanList.load(path)
    assert again.usernames == {"spammer"}
    assert again.ips == {"1.2.3.4"}
    assert again.reason(username="spammer") == "why"
    assert again.is_banned(username="spammer")
    assert again.is_banned(ip="1.2.3.4")
    assert not again.is_banned(username="someone-else", ip="5.6.7.8")


# ---------------------------------------------------------------------------
# Password policy on the registration route (user-specified: minimum 8).
# ---------------------------------------------------------------------------


def test_a_short_password_cannot_register(tmp_path: Path) -> None:
    """⚑ THE VOLUNTEER ROUTE IS WHERE A LENGTH RULE ACTUALLY HAS TO HOLD.

    The admin route is used by one person who can be told; this one is open to
    anyone who reaches the port with the flag on. 400, not 401: the credential
    is not wrong, it is unacceptable, and a 401 would send the client into a
    retry loop with the same password.
    """
    _seed(tmp_path)
    client = _client(tmp_path, self_register=True)

    response = _ping(client, "stranger", "abcdefg")
    assert response.status_code == 400
    assert "8" in response.json()["detail"]
    assert "stranger" not in json.loads(
        (tmp_path / "users.json").read_text(encoding="utf-8")
    ), "a refused registration must not have created the account"


def test_eight_characters_registers(tmp_path: Path) -> None:
    """The boundary from the other side, so the rule is `< 8`, not `<= 8`."""
    _seed(tmp_path)
    client = _client(tmp_path, self_register=True)

    assert _ping(client, "stranger", "abcdefgh").status_code not in (400, 401, 403)
    raw = json.loads((tmp_path / "users.json").read_text(encoding="utf-8"))
    assert verify_password("abcdefgh", UserRecord(username="stranger", **raw["stranger"]))


def test_an_existing_short_password_still_authenticates(tmp_path: Path) -> None:
    """⚑ ENFORCEMENT IS ON SETTING, NOT ON CHECKING.

    `_seed` deliberately provisions `pw` — two characters, below the policy. It
    must keep working, on both sides of the flag: the fleet's current
    credentials predate the rule, and a policy that logged them out would be a
    worse outage than the weak password. This is the rotation-window guarantee.
    """
    _seed(tmp_path)
    for flag in (False, True):
        client = _client(tmp_path, self_register=flag)
        assert _ping(client, "known", "pw").status_code not in (400, 401, 403)


def test_a_weak_registration_is_not_counted_as_a_failed_signin(tmp_path: Path) -> None:
    """A rejected-for-length attempt must not spend the IP's throttle budget.

    Otherwise a volunteer who types a short password a few times locks
    themselves out with a 429 and no idea why — the failure is theirs to fix in
    one step, and the throttle exists for guessing, not for typos.
    """
    _seed(tmp_path)
    client = _client(tmp_path, self_register=True)

    for _ in range(6):
        assert _ping(client, "stranger", "short").status_code == 400
    assert _ping(client, "stranger", "a-good-password").status_code not in (400, 429)


def test_the_registration_route_calls_the_shared_policy_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ R3: THE DOCS SAY SELF-REGISTRATION SHARES `check_new_password`. THIS
    IS THE ASSERTION THAT MAKES THE SENTENCE TRUE.

    Everything else here tests the *effect* of the policy (a 7-character
    password is refused), which a second, private copy of the rule inside
    `app.py` would satisfy just as well — and that copy would then drift from
    the CLI's the first time the minimum changed. So this test replaces the
    shared function and asserts the ROUTE picked up the replacement: it can
    only pass if the HTTP path calls that exact function.

    A password of 10 characters, refused only because the patched policy
    refuses everything, cannot be confused with the length rule doing it.
    """
    from chess_anti_engine.server import auth as auth_mod

    called: list[str] = []

    def _reject_everything(password: str) -> None:
        called.append(password)
        raise auth_mod.WeakPassword("policy hook reached")

    monkeypatch.setattr(auth_mod, "check_new_password", _reject_everything)

    _seed(tmp_path)
    client = _client(tmp_path, self_register=True)
    response = _ping(client, "stranger", "long-enough-password")

    assert called == ["long-enough-password"], (
        "the registration route did not call auth.check_new_password — the "
        "policy is documented as shared and must not be a second copy"
    )
    assert response.status_code == 400
    assert "policy hook reached" in response.json()["detail"]
    assert "stranger" not in json.loads(
        (tmp_path / "users.json").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# The registration RACE. This module had 391 lines and no concurrency test,
# which is exactly how a 9-of-10 authentication bypass shipped in it.
# ---------------------------------------------------------------------------


def _race(client, username: str, passwords: list[str]) -> list[tuple[str, int]]:
    """Fire one first-connect per password, all released from a barrier.

    The barrier is the point: staggered requests serialise and the second one
    takes the ordinary known-user path, which is not the code under test. These
    have to arrive while the registration is genuinely in flight.
    """
    import threading

    barrier = threading.Barrier(len(passwords))
    out: list[tuple[str, int]] = []
    lock = threading.Lock()

    def attempt(password: str) -> None:
        barrier.wait(timeout=30)
        code = _ping(client, username, password).status_code
        with lock:
            out.append((password, code))

    threads = [threading.Thread(target=attempt, args=(p,)) for p in passwords]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "a racing request never returned"
    assert len(out) == len(passwords)
    return out


def test_a_ten_way_first_connect_race_authenticates_exactly_one_password(
    tmp_path: Path,
) -> None:
    """⚑ THE BYPASS THIS FILE EXISTED WITHOUT NOTICING.

    Ten clients present the same unknown username with ten DIFFERENT passwords
    at once. One wins the race and its password is what gets stored. The other
    nine lost, and the old code returned them the winner's record straight from
    `_register_new_user` — so nine clients authenticated on passwords the server
    had rejected, and the server could not tell you which password any of them
    holds.

    The invariant is not "nine get 401": it is that the set of credentials that
    authenticate is exactly the one stored. Anything else means the account's
    password is not the account's password.
    """
    _seed(tmp_path)
    client = _client(tmp_path, self_register=True)
    passwords = [f"race-password-{i:02d}" for i in range(10)]

    results = _race(client, "racer", passwords)

    accepted = [pw for pw, code in results if code not in (400, 401, 403, 429)]
    assert len(accepted) == 1, (
        f"{len(accepted)} of 10 racing passwords authenticated: {accepted}. "
        "Only the one that was stored may."
    )
    raw = json.loads((tmp_path / "users.json").read_text(encoding="utf-8"))
    record = UserRecord(username="racer", **raw["racer"])
    assert verify_password(accepted[0], record), (
        "the password that authenticated is not the password that was stored"
    )
    for pw in passwords:
        if pw != accepted[0]:
            assert not verify_password(pw, record)


def test_the_loser_of_a_registration_race_can_still_get_in_with_the_right_password(
    tmp_path: Path,
) -> None:
    """Losing the race is not itself a rejection.

    Two workers deployed with the SAME shared credential first-connect at the
    same moment — the realistic fleet case. One registers, the other verifies
    against what was just stored and is let in. Fixing the bypass must not turn
    that into a coin flip.
    """
    _seed(tmp_path)
    client = _client(tmp_path, self_register=True)

    results = _race(client, "fleet", ["one-shared-secret"] * 4)

    assert all(code not in (400, 401, 403, 429) for _, code in results), results


def test_a_racing_wrong_password_is_charged_against_the_failed_signin_quota(
    tmp_path: Path,
) -> None:
    """⚑ THE CHARGE IS HALF THE FIX.

    A refusal that costs nothing makes racing a username a free guessing
    oracle: the attacker fires N attempts at a name being claimed, and the only
    quota touched is the registration one, which only the WINNER reaches. So the
    loser has to be charged exactly like any other bad password.

    Asserted through the budget rather than by reaching into the guard: after
    the race has charged one failure, exactly `limit - 1` further bad sign-ins
    fit before the 429. If the racing failure were uncounted, the last one here
    would still be allowed.
    """
    from chess_anti_engine.server.access import DEFAULT_FAILED_AUTH_MAX_PER_WINDOW as LIMIT

    _seed(tmp_path)
    client = _client(tmp_path, self_register=True)

    results = _race(client, "racer", ["winner-password", "loser-password"])
    losers = [pw for pw, code in results if code == 401]
    assert len(losers) == 1, f"expected exactly one refusal, got {results}"

    for i in range(LIMIT - 1):
        assert _ping(client, "known", "wrong").status_code == 401, (
            f"budget ran out after {i} further failures — the race charged more "
            "than one"
        )
    assert _ping(client, "known", "wrong").status_code == 429, (
        "the quota was not charged for the racing failure: the window still had "
        "room for a full LIMIT of failures after it"
    )


# ---------------------------------------------------------------------------
# F1: the client IP is the SOCKET's, never a header the client controls.
# ---------------------------------------------------------------------------


def test_x_forwarded_for_cannot_move_a_ban(tmp_path: Path) -> None:
    """⚑ A CLIENT-SUPPLIED HEADER MUST NOT BE AN IDENTITY.

    `_client_ip` reads `request.client.host` and deliberately does not consult
    X-Forwarded-For, because this server is reached directly: trusting the
    header would let a banned client walk past its ban by typing a different
    address, and let anyone attribute their traffic to someone else's bucket.

    Mutating `_client_ip` to prefer the header survived every other test in this
    file, because nothing sent one. Both directions are asserted here — the
    header cannot clear a ban on the real address, and it cannot forge one onto
    an address that is not banned.
    """
    _seed(tmp_path)
    bans = BanList(path=tmp_path / BANS_FILENAME, ips={"203.0.113.7"})
    bans.save()
    client = _client(tmp_path, self_register=False)

    spoofed = dict(_HEADERS, **{"X-Forwarded-For": "198.51.100.1"})
    banned = client.post(
        "/v1/report_bad_shard",
        json={"sha256": "0" * 64, "reason": "auth-probe"},
        auth=("known", "pw"), headers=spoofed,
    )
    assert banned.status_code == 403, (
        "the ban on the real socket address was evaded by sending an "
        "X-Forwarded-For header"
    )

    BanList(path=tmp_path / BANS_FILENAME, ips={"198.51.100.1"}).save()
    innocent = client.post(
        "/v1/report_bad_shard",
        json={"sha256": "0" * 64, "reason": "auth-probe"},
        auth=("known", "pw"), headers=spoofed,
    )
    assert innocent.status_code not in (401, 403), (
        "a ban on an address that appears ONLY in a client-supplied header was "
        "enforced — the header is being trusted as an identity"
    )


def test_x_forwarded_for_cannot_move_the_rate_limit_bucket(tmp_path: Path) -> None:
    """The same header must not reset a throttle either.

    Registration is capped per IP. If the bucket were keyed on X-Forwarded-For,
    a fresh header value would be a fresh quota, and the cap would mean nothing
    to anyone who could type one.
    """
    from chess_anti_engine.server.access import DEFAULT_REGISTER_MAX_PER_WINDOW as LIMIT

    _seed(tmp_path)
    client = _client(tmp_path, self_register=True)

    for i in range(LIMIT):
        assert _ping(client, f"vol{i}", f"secret-{i}").status_code not in (401, 429)

    fresh_header = dict(_HEADERS, **{"X-Forwarded-For": f"198.51.100.{LIMIT + 1}"})
    response = client.post(
        "/v1/report_bad_shard",
        json={"sha256": "0" * 64, "reason": "auth-probe"},
        auth=("one-more", "secret-more"), headers=fresh_header,
    )
    assert response.status_code == 429, (
        "a new X-Forwarded-For value bought a fresh registration quota"
    )
