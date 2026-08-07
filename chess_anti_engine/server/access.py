"""Bans and per-IP throttling for a server that faces volunteers.

Today the fleet is four workers on one LAN with one shared credential. The
volunteer deployment this prepares for is a different threat model on the same
code: an open port, unknown clients, and no way to vet anyone in advance. Two
mechanisms, both deliberately simple:

* **Bans**, by username or by IP, persisted next to `users.json` so they survive
  a restart. Enforced at authentication, so a banned identity loses every
  authenticated route at once — uploads, leases, arena results — rather than
  each route having to remember.
* **Per-IP throttling**, separately for registrations and for failed logins on
  existing accounts. An open port that will create an account on demand is an
  account-spam target, and one that will check a password on demand is a
  guessing target.

⚑ A18: uvicorn stays SINGLE-PROCESS, which is what makes the in-memory counters
here sound — a second worker process would give each its own counters and
silently multiply every limit by the worker count. That precondition is load
bearing; see the A18 note in `app.py`. The counters are guarded by a
`threading.Lock` rather than left bare, because the routes that touch them run
in Starlette's threadpool: single PROCESS is not single EXECUTION CONTEXT, and
the loop is not serialising these for us.

⚑ WHAT THIS IS NOT. It is not a defence against a determined attacker with many
IPs, and it is not rate limiting for the authenticated hot path. It raises the
cost of the two cheap attacks that an open registration endpoint invites, and
it gives an operator a ban button. Anything stronger belongs in front of the
server, not in it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_anti_engine.utils.atomic import atomic_write_text

BANS_FILENAME = "bans.json"

# Registration is the expensive, irreversible one: it creates state. Failed
# logins are cheap to retry legitimately (a worker restarting with a stale
# secret), so that window is shorter and more forgiving.
DEFAULT_REGISTER_MAX_PER_WINDOW = 5
DEFAULT_REGISTER_WINDOW_S = 3600.0
DEFAULT_FAILED_AUTH_MAX_PER_WINDOW = 20
DEFAULT_FAILED_AUTH_WINDOW_S = 300.0


class BannedIdentity(RuntimeError):
    """The caller is banned. Surfaces as 403, never as 401.

    The distinction matters operationally: 401 tells a volunteer "your password
    is wrong" and invites a retry loop; 403 tells them to stop.
    """


class RateLimited(RuntimeError):
    """Too many attempts from one address inside the window. Surfaces as 429."""


_log = logging.getLogger("chess_anti_engine.server.access")


@dataclass
class BanList:
    """Banned usernames and IPs, persisted as JSON next to `users.json`.

    Untracked by construction: it lives in the server root, which is under
    `runs/`. It carries no secret material — names and addresses only — but it
    is still operator state, so it is written atomically and read fresh whenever
    the file changes, the same discipline `users.json` gets.
    """

    path: Path
    usernames: set[str] = field(default_factory=set)
    ips: set[str] = field(default_factory=set)
    notes: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path) -> BanList:
        p = Path(path)
        out = BanList(path=p)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
  # Absent is the normal case and means "nobody is banned", so it stays
  # silent. A file that EXISTS and will not parse is a different event with
  # the same consequence -- everybody unbanned -- and it used to be
  # indistinguishable from absent, so the operator's only signal was
  # `list-bans` printing nothing, which is also what an empty list prints.
  #
  # ⚑ STILL FAILS OPEN, deliberately: a ban list that refuses to start the
  # server is worse than one that is briefly ignored on a LAN fleet. But
  # "fail open" and "fail open silently" are different promises, and only
  # the first one was intended. `manage_users.py` writes atomically, so a
  # torn file means a disk fault, not a race -- worth a WARNING either way.
            if p.exists():
                _log.warning(
                    "ban list %s exists but could not be read (%s: %s); "
                    "PROCEEDING WITH NOBODY BANNED. Fix or delete the file -- "
                    "until then every banned username and IP is accepted.",
                    p, type(exc).__name__, exc,
                )
            return out
        if not isinstance(raw, dict):
            _log.warning(
                "ban list %s parsed as %s, not an object; PROCEEDING WITH "
                "NOBODY BANNED. Expected {\"usernames\": [...], \"ips\": [...]}.",
                p, type(raw).__name__,
            )
            return out
        out.usernames = {str(u) for u in raw.get("usernames", []) if str(u).strip()}
        out.ips = {str(i) for i in raw.get("ips", []) if str(i).strip()}
        notes = raw.get("notes")
        out.notes = {str(k): str(v) for k, v in notes.items()} if isinstance(notes, dict) else {}
        return out

    def save(self) -> None:
        payload: dict[str, Any] = {
            "usernames": sorted(self.usernames),
            "ips": sorted(self.ips),
            "notes": dict(sorted(self.notes.items())),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=True))

    def is_banned(self, *, username: str | None = None, ip: str | None = None) -> bool:
        if username and str(username) in self.usernames:
            return True
        return bool(ip and str(ip) in self.ips)

    def reason(self, *, username: str | None = None, ip: str | None = None) -> str:
        for key in (str(username or ""), str(ip or "")):
            if key and key in self.notes:
                return self.notes[key]
        return ""


class _Window:
    """Per-key attempt counter over a sliding window. Not thread-safe alone."""

    def __init__(self, *, limit: int, window_s: float) -> None:
        self.limit = int(limit)
        self.window_s = float(window_s)
        self._hits: dict[str, list[float]] = {}

    def _prune(self, key: str, now: float) -> list[float]:
        kept = [t for t in self._hits.get(key, ()) if now - t < self.window_s]
        if kept:
            self._hits[key] = kept
        else:
            self._hits.pop(key, None)
        return kept

    def would_exceed(self, key: str, now: float) -> bool:
        return len(self._prune(key, now)) >= self.limit

    def record(self, key: str, now: float) -> None:
        self._hits.setdefault(key, []).append(now)

    def clear(self, key: str) -> None:
        self._hits.pop(key, None)

    def count(self, key: str, now: float) -> int:
        return len(self._prune(key, now))


class AccessGuard:
    """Ban enforcement plus per-IP throttling, reloaded when the file changes.

    Constructed once per app. `check_*` raise rather than return a verdict, so
    a caller cannot accept the guard's answer and then forget to act on it —
    which is the failure this codebase produces most reliably.
    """

    def __init__(
        self,
        bans_path: str | Path,
        *,
        register_limit: int = DEFAULT_REGISTER_MAX_PER_WINDOW,
        register_window_s: float = DEFAULT_REGISTER_WINDOW_S,
        failed_auth_limit: int = DEFAULT_FAILED_AUTH_MAX_PER_WINDOW,
        failed_auth_window_s: float = DEFAULT_FAILED_AUTH_WINDOW_S,
    ) -> None:
        self.path = Path(bans_path)
        self._lock = threading.Lock()
        self._bans = BanList.load(self.path)
        self._stamp = self._file_stamp()
        self._registrations = _Window(limit=register_limit, window_s=register_window_s)
        self._failed_auth = _Window(limit=failed_auth_limit, window_s=failed_auth_window_s)

    def _file_stamp(self) -> tuple[int, int, int]:
        try:
            st = self.path.stat()
        except OSError:
            return (0, 0, 0)
        return (int(st.st_mtime_ns), int(st.st_size), int(st.st_ino))

    def bans(self) -> BanList:
        """The current ban list, re-read only when the file has changed.

        Same stat-keyed shape as the credential cache, and for the same reason:
        an operator bans someone with `manage_users.py` in another process, and
        it has to take effect on the next request without a server restart.
        """
        stamp = self._file_stamp()
        with self._lock:
            if stamp == self._stamp:
                return self._bans
        fresh = BanList.load(self.path)
        with self._lock:
            self._stamp = stamp
            self._bans = fresh
        return fresh

    def check_not_banned(self, *, username: str | None, ip: str | None) -> None:
        bans = self.bans()
        if bans.is_banned(username=username, ip=ip):
            reason = bans.reason(username=username, ip=ip)
            raise BannedIdentity(reason or "this identity is banned from this server")

    def check_registration_allowed(self, ip: str | None) -> None:
        """Throttle BEFORE creating an account, and count only on success.

        Counting attempts here rather than in `note_registration` would let a
        request that fails for an unrelated reason consume a volunteer's quota.
        """
        if not ip:
            return
        now = time.time()
        with self._lock:
            if self._registrations.would_exceed(ip, now):
                raise RateLimited(
                    f"too many new accounts from this address "
                    f"({self._registrations.limit} per "
                    f"{int(self._registrations.window_s / 60)} minutes)"
                )

    def note_registration(self, ip: str | None) -> None:
        if not ip:
            return
        with self._lock:
            self._registrations.record(ip, time.time())

    def check_auth_attempt_allowed(self, ip: str | None) -> None:
        if not ip:
            return
        now = time.time()
        with self._lock:
            if self._failed_auth.would_exceed(ip, now):
                raise RateLimited(
                    f"too many failed sign-ins from this address; wait "
                    f"{int(self._failed_auth.window_s / 60) or 1} minutes"
                )

    def note_auth_failure(self, ip: str | None) -> None:
        if not ip:
            return
        with self._lock:
            self._failed_auth.record(ip, time.time())

    def note_auth_success(self, ip: str | None) -> None:
        """Clear the failure counter, so a worker that fixed its credential is
        not held out for the rest of the window by its own earlier retries."""
        if not ip:
            return
        with self._lock:
            self._failed_auth.clear(ip)

    def failed_auth_count(self, ip: str) -> int:
        with self._lock:
            return self._failed_auth.count(ip, time.time())
