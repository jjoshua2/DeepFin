from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

from .auth import (
    UserStats,
    WeakPassword,
    check_new_password,
    ensure_user,
    load_user_stats,
    load_users,
    migrate_user_stats,
    set_disabled,
    upsert_user,
    user_stats_path_for,
)


def _resolve_password(arg: str | None, env_name: str | None = None) -> str:
    """A password from --password-env, the prompt, or --password (discouraged).

    THE one home for choosing a password on the CLI: all three sources land
    here, so the policy check below cannot be routed around by picking a
    different flag. `add` and `set-password` both call it exactly once.

    ⚑ `--password` PUTS THE SECRET IN THE PROCESS TITLE. Every account on the
    box can read it out of `ps auxww` for the lifetime of the call, and shells
    persist it in history besides. It is kept because scripts use it and
    breaking them would push people to worse workarounds, but it now warns, and
    the two safe routes are listed first in `--help`.
    """
    if env_name:
        password = _password_from_env(env_name)
    elif arg is not None:
        if not arg.strip():
  # ⚑ THE REALISTIC ROUTE IS `--password "$WORKER_PW"` WITH THE VARIABLE
  # UNSET, which the shell expands to an empty string -- so this is a
  # typo away, not a hypothetical. `check_new_password` would reject it
  # too; this branch exists only to name the cause, because "must be at
  # least 8 characters" does not tell you your variable was unset.
            raise SystemExit(
                "refusing to set an empty password. `--password \"$VAR\"` with "
                "VAR unset expands to an empty string; check the variable, or "
                "use --password-env VAR so an unset variable is an error."
            )
        print(
            "WARNING: --password puts the secret in the process title, where "
            "`ps auxww` exposes it to every user on this machine, and in your "
            "shell history. Prefer --password-env NAME, or omit it for a "
            "prompt.",
            file=sys.stderr,
        )
        password = arg
    else:
        password = _prompt_password()

    try:
        check_new_password(password)
    except WeakPassword as exc:
  # Policy lives in auth.py so self-registration shares it; the CLI just
  # translates the refusal into an exit.
        raise SystemExit(str(exc)) from exc
    return password


def _password_from_env(name: str) -> str:
    value = str(os.environ.get(name, "") or "")
    if not value.strip():
  # Hard error, not a prompt fallback: the operator named a source, and
  # quietly using a different one is how a script "succeeds" with the wrong
  # credential.
        raise SystemExit(
            f"--password-env {name}: that environment variable is empty or "
            f"unset. Export it in this shell first."
        )
    return value


def _prompt_password() -> str:
    """Read a password interactively. Confirms it; does NOT police it.

    The policy (non-empty, minimum length) is applied once by
    `_resolve_password`, which is the only caller. An earlier version checked
    for emptiness here and claimed this was "the only place it can be caught" --
    it was not, which is exactly how `--password ''` reached `ensure_user`.
    """
    pw = getpass.getpass("Password: ")
    pw2 = getpass.getpass("Confirm: ")
    if pw != pw2:
        raise SystemExit("passwords do not match")
    return pw


def main() -> None:
    ap = argparse.ArgumentParser(description="Manage server users.json (upload accounts)")
    ap.add_argument("--users-db", type=str, default="server/users.json")

    sub = ap.add_subparsers(dest="cmd", required=True)

    add = sub.add_parser("add", help="Add a new user")
    add.add_argument("username", type=str)
    add.add_argument(
        "--password-env", type=str, default=None,
        help="Name of an environment variable holding the password (preferred)",
    )
    add.add_argument(
        "--password", type=str, default=None,
        help="Password inline. DISCOURAGED: visible in `ps` and shell history",
    )

    sp = sub.add_parser("set-password", help="Change an existing user's password")
    sp.add_argument("username", type=str)
    sp.add_argument(
        "--password-env", type=str, default=None,
        help="Name of an environment variable holding the password (preferred)",
    )
    sp.add_argument(
        "--password", type=str, default=None,
        help="Password inline. DISCOURAGED: visible in `ps` and shell history",
    )

    dis = sub.add_parser("disable", help="Disable a user")
    dis.add_argument("username", type=str)

    en = sub.add_parser("enable", help="Enable a user")
    en.add_argument("username", type=str)

    sub.add_parser("list", help="List users")

    args = ap.parse_args()
    db = Path(args.users_db)

    if args.cmd == "add":
        pw = _resolve_password(args.password, args.password_env)
        ensure_user(db, username=str(args.username), password=pw)
        print(f"Added user {args.username!r}")
        return

    if args.cmd == "set-password":
        pw = _resolve_password(args.password, args.password_env)
        users = load_users(db)
        if str(args.username) not in users:
            raise SystemExit(f"user {args.username!r} not found")
        upsert_user(db, username=str(args.username), password=pw)
        print(f"Updated password for {args.username!r}")
        return

    if args.cmd in ("disable", "enable"):
        set_disabled(db, username=str(args.username), disabled=(args.cmd == "disable"))
        return

    if args.cmd == "list":
        users = load_users(db)
  # Counters live beside the credential file since the users.json split.
  # Migrating here too means `manage_users list` reports correctly even if it
  # is the first thing run against a pre-split DB, rather than showing zeros
  # until the server happens to start.
        stats_path = user_stats_path_for(db)
        migrate_user_stats(db, stats_path)
        stats = load_user_stats(stats_path)
        for u in sorted(users.keys()):
            rec = users[u]
            status = "disabled" if rec.disabled else "enabled"
            st = stats.get(u, UserStats())
            print(f"{u}\t{status}\tuploads={st.uploads}\tpositions={st.total_positions}")
            for machine, mstats in sorted(st.machines.items()):
                print(f"  {machine}\tuploads={mstats.get('uploads', 0)}\tpositions={mstats.get('positions', 0)}")
        return


if __name__ == "__main__":
    main()
