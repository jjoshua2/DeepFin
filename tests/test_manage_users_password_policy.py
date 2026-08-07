"""Password policy on every route that SETS a password.

Two rules, one home (`auth.check_new_password`, called by
`manage_users._resolve_password`):

* a minimum length of 8 — user-specified policy, 2026-08-05;
* which subsumes the empty case, whose realistic shape is
  `--password "$WORKER_PW"` with the variable unset. The shell expands that to
  an empty string; before this, it hashed and stored fine and then
  authenticated a client that sent nothing.

⚑ THE POLICY IS ON SETTING, NOT ON VERIFYING. `verify_password` is untouched,
so accounts that predate the rule keep working — a length rule that logged the
fleet out mid-rotation would be a worse outage than the weak password.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from chess_anti_engine.server import manage_users
from chess_anti_engine.server.auth import (
    MIN_PASSWORD_LENGTH,
    WeakPassword,
    check_new_password,
    hash_password,
    load_users,
    verify_password,
)


def _run(args: list[str], db: Path) -> None:
    """Drive the CLI exactly as an operator does -- through argv and `main`."""
    saved = sys.argv
    sys.argv = ["manage_users", "--users-db", str(db), *args]
    try:
        manage_users.main()
    finally:
        sys.argv = saved


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "users.json"


def test_the_policy_minimum_is_eight() -> None:
    """Pinned as a value, so a silent relaxation is a diff on this line."""
    assert MIN_PASSWORD_LENGTH == 8


def test_seven_characters_is_refused_end_to_end(db: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(["add", "volunteer", "--password", "abcdefg"], db)
    assert "8" in str(exc.value)
    assert not db.exists(), "no account may be created by a refused password"


def test_eight_characters_is_accepted_end_to_end(db: Path) -> None:
    _run(["add", "volunteer", "--password", "abcdefgh"], db)
    record = load_users(db)["volunteer"]
    assert verify_password("abcdefgh", record)


def test_an_unset_shell_variable_is_refused_end_to_end(db: Path) -> None:
    """`--password "$WORKER_PW"` with WORKER_PW unset — the exact shape.

    The message must name the CAUSE. "at least 8 characters" is true but
    useless here: the operator typed a variable, not a password.
    """
    with pytest.raises(SystemExit) as exc:
        _run(["add", "volunteer", "--password", ""], db)
    message = str(exc.value)
    assert "empty" in message
    assert "VAR unset" in message or "unset" in message
    assert not db.exists()


def test_a_refused_empty_password_leaves_nothing_that_authenticates(db: Path) -> None:
    """The failure that matters is not the exit code — it is the account.

    Asserted separately from the message so a future message rewrite cannot
    quietly take this with it.
    """
    with pytest.raises(SystemExit):
        _run(["add", "volunteer", "--password", ""], db)
    assert not db.exists()


def test_the_env_route_is_covered_too(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--password-env used to bypass `_resolve_password` entirely.

    It reached `ensure_user` through its own branch, so a policy check written
    only into the `--password` path would have left the RECOMMENDED route
    unchecked — the accepted-then-ignored shape, inverted.
    """
    monkeypatch.setenv("VOL_PW", "short7x")
    with pytest.raises(SystemExit) as exc:
        _run(["add", "volunteer", "--password-env", "VOL_PW"], db)
    assert "8" in str(exc.value)

    monkeypatch.setenv("VOL_PW", "longenough")
    _run(["add", "volunteer", "--password-env", "VOL_PW"], db)
    assert verify_password("longenough", load_users(db)["volunteer"])


def test_the_prompt_route_is_covered_too(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(["short7x", "short7x"])
    monkeypatch.setattr(manage_users.getpass, "getpass", lambda _prompt: next(replies))
    with pytest.raises(SystemExit) as exc:
        _run(["add", "volunteer"], db)
    assert "8" in str(exc.value)


def test_set_password_is_covered_too(db: Path) -> None:
    """`add` and `set-password` are separate call sites; both must check."""
    _run(["add", "volunteer", "--password", "goodenough"], db)
    with pytest.raises(SystemExit) as exc:
        _run(["set-password", "volunteer", "--password", "abcdefg"], db)
    assert "8" in str(exc.value)
    assert verify_password("goodenough", load_users(db)["volunteer"]), (
        "a refused change must not have disturbed the existing credential"
    )


def test_an_existing_short_password_still_authenticates(db: Path) -> None:
    """⚑ THE ROTATION-WINDOW GUARANTEE, and the reason it is a test.

    Enforcement is on SETTING. An account created before the policy — or by
    hand, or by an older build — keeps verifying, so raising the bar cannot
    take the fleet down before the operator has rotated.
    """
    salt, digest, iterations = hash_password("old")
    db.write_text(json.dumps({
        "legacy": {
            "salt_b64": salt,
            "hash_b64": digest,
            "iterations": iterations,
            "disabled": False,
        },
    }), encoding="utf-8")
    assert verify_password("old", load_users(db)["legacy"])


def test_whitespace_is_not_a_password() -> None:
    """Eight spaces clear the length rule and are not a secret."""
    with pytest.raises(WeakPassword):
        check_new_password(" " * 8)


def test_both_password_flags_together_are_refused(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ R1: ONE OF THE TWO WAS SILENTLY DISCARDED.

    `--password X --password-env Y` exited 0 having used Y. The operator who
    typed X walks away believing they set it, and the account does not accept
    the password they wrote down — the accepted-then-ignored shape, with a
    credential on the other end of it.
    """
    monkeypatch.setenv("VOL_PW", "env-password")
    with pytest.raises(SystemExit) as exc:
        _run(["add", "volunteer", "--password", "flag-password",
              "--password-env", "VOL_PW"], db)
    assert "mutually exclusive" in str(exc.value)
    assert not db.exists(), "no account may be created from an ambiguous request"


def test_each_flag_alone_still_works(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal must be about the COMBINATION, not about either flag."""
    monkeypatch.setenv("VOL_PW", "env-password")
    _run(["add", "from-env", "--password-env", "VOL_PW"], db)
    _run(["add", "from-flag", "--password", "flag-password"], db)
    users = load_users(db)
    assert verify_password("env-password", users["from-env"])
    assert verify_password("flag-password", users["from-flag"])
