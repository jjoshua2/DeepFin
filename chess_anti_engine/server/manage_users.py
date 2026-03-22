from __future__ import annotations

import argparse
import getpass
from pathlib import Path

from .auth import ensure_user, load_users, save_users, set_disabled, upsert_user


def _prompt_password() -> str:
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
    add.add_argument("--password", type=str, default=None, help="Password (omit for interactive prompt)")

    sp = sub.add_parser("set-password", help="Change an existing user's password")
    sp.add_argument("username", type=str)
    sp.add_argument("--password", type=str, default=None, help="Password (omit for interactive prompt)")

    dis = sub.add_parser("disable", help="Disable a user")
    dis.add_argument("username", type=str)

    en = sub.add_parser("enable", help="Enable a user")
    en.add_argument("username", type=str)

    sub.add_parser("list", help="List users")

    args = ap.parse_args()
    db = Path(args.users_db)

    if args.cmd == "add":
        pw = args.password or _prompt_password()
        ensure_user(db, username=str(args.username), password=pw)
        print(f"Added user {args.username!r}")
        return

    if args.cmd == "set-password":
        pw = args.password or _prompt_password()
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
        for u in sorted(users.keys()):
            rec = users[u]
            status = "disabled" if rec.disabled else "enabled"
            print(f"{u}\t{status}\tuploads={rec.uploads}\tpositions={rec.total_positions}")
            for machine, stats in sorted(rec.machines.items()):
                print(f"  {machine}\tuploads={stats.get('uploads', 0)}\tpositions={stats.get('positions', 0)}")
        return


if __name__ == "__main__":
    main()
