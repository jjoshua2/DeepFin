from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_anti_engine.utils.atomic import atomic_write_text


@dataclass
class UserRecord:
    username: str
  # PBKDF2-SHA256
    salt_b64: str
    iterations: int
    hash_b64: str

    disabled: bool = False

  # Aggregate stats
    uploads: int = 0
    total_bytes: int = 0
    total_positions: int = 0
    last_upload_at_unix: int | None = None

  # Per-machine stats: machine_id -> {uploads, positions, last_upload_at_unix}
    machines: dict = field(default_factory=dict)


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def _pbkdf2(password: str, *, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations), dklen=32)


def hash_password(password: str, *, iterations: int = 200_000) -> tuple[str, str, int]:
    salt = os.urandom(16)
    h = _pbkdf2(password, salt=salt, iterations=int(iterations))
    return (_b64e(salt), _b64e(h), int(iterations))


def verify_password(password: str, rec: UserRecord) -> bool:
    salt = _b64d(rec.salt_b64)
    want = _b64d(rec.hash_b64)
    got = _pbkdf2(password, salt=salt, iterations=int(rec.iterations))
    return hmac.compare_digest(want, got)


def load_users(path: str | Path) -> dict[str, UserRecord]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("users db must be a dict")

    return {
        str(username): UserRecord(username=str(username), **v)
        for username, v in data.items()
        if isinstance(v, dict)
    }


def save_users(
    path: str | Path, users: dict[str, UserRecord], *, durable: bool = True,
) -> None:
    """Write the users DB.

    ``durable=False`` is for the upload-stats path only, which rewrites this
    file on EVERY accepted shard and changes nothing but counters
    (``uploads`` / ``total_bytes`` / ``machines``). Credential changes —
    :func:`add_user`, :func:`upsert_user`, :func:`set_disabled` — keep the
    default: losing a just-created password to an unclean shutdown locks a
    worker out of the fleet, which is not a cost worth ~11 ms per upload.
    """
    data: dict[str, Any] = {}
    for u, rec in users.items():
  # exclude username field (key is username)
        d = rec.__dict__.copy()
        d.pop("username", None)
        data[u] = d
    atomic_write_text(
        Path(path), json.dumps(data, indent=2, sort_keys=True), durable=durable,
    )


def record_upload(
    users: dict[str, UserRecord],
    *,
    username: str,
    bytes_uploaded: int,
    positions: int | None,
    machine_id: str | None = None,
) -> None:
    rec = users.get(username)
    if rec is None:
        return
    now = int(time.time())
    rec.uploads = int(rec.uploads) + 1
    rec.total_bytes = int(rec.total_bytes) + int(bytes_uploaded)
    if positions is not None:
        rec.total_positions = int(rec.total_positions) + int(positions)
    rec.last_upload_at_unix = now

    if machine_id:
        m = rec.machines.setdefault(machine_id, {"uploads": 0, "positions": 0, "last_upload_at_unix": None})
        m["uploads"] = int(m.get("uploads", 0)) + 1
        if positions is not None:
            m["positions"] = int(m.get("positions", 0)) + int(positions)
        m["last_upload_at_unix"] = now


def ensure_user(
    users_path: str | Path,
    *,
    username: str,
    password: str,
    disabled: bool = False,
) -> None:
    users = load_users(users_path)
    if username in users:
        raise ValueError(f"user {username!r} already exists")

    salt_b64, hash_b64, iterations = hash_password(password)
    users[username] = UserRecord(
        username=username,
        salt_b64=salt_b64,
        iterations=iterations,
        hash_b64=hash_b64,
        disabled=bool(disabled),
    )
    save_users(users_path, users)


def upsert_user(
    users_path: str | Path,
    *,
    username: str,
    password: str,
    disabled: bool = False,
) -> None:
    """Create or update (re-hash) a user's password."""
    users = load_users(users_path)
    existing = users.get(username)
    salt_b64, hash_b64, iterations = hash_password(password)
    users[username] = UserRecord(
        username=username,
        salt_b64=salt_b64,
        iterations=iterations,
        hash_b64=hash_b64,
        disabled=bool(disabled),
        uploads=existing.uploads if existing else 0,
        total_bytes=existing.total_bytes if existing else 0,
        total_positions=existing.total_positions if existing else 0,
        last_upload_at_unix=existing.last_upload_at_unix if existing else None,
        machines=dict(existing.machines) if existing else {},
    )
    save_users(users_path, users)


def set_disabled(users_path: str | Path, *, username: str, disabled: bool) -> None:
    users = load_users(users_path)
    if username not in users:
        raise ValueError(f"unknown user {username!r}")
    users[username].disabled = bool(disabled)
    save_users(users_path, users)


@dataclass(frozen=True)
class _DbStamp:
    """Identity of a users-DB file version. Any write changes at least one field.

    `st_ino` is in here because every mutation path goes through `save_users`
    -> `atomic_write_text`, which writes a tmp file and `os.replace`s it: the
    inode is always new, even for a write that somehow lands in the same
    nanosecond at the same size. `st_mtime_ns` rather than `st_mtime` because
    the float loses resolution and this is a cache-invalidation key.
    """

    mtime_ns: int
    size: int
    inode: int
    exists: bool = True

    @staticmethod
    def of(path: Path) -> _DbStamp:
        try:
            st = path.stat()
        except OSError:
            return _DbStamp(0, 0, 0, exists=False)
        return _DbStamp(int(st.st_mtime_ns), int(st.st_size), int(st.st_ino))


class VerifiedCredentialCache:
    """Skips the PBKDF2 recomputation for a credential already verified.

    ⚑ THE COST THIS EXISTS FOR: `_auth_user` re-read the users DB, re-parsed
    it, and recomputed PBKDF2-SHA256 at 200k iterations on EVERY authenticated
    request -- 73.4ms of CPU per call, measured. Workers poll; on a cold start
    they all poll at once, so the herd pins the threadpool and everything
    behind it queues. The work factor is there to make an OFFLINE guess
    expensive, and it still is: a MISS pays it in full, and a wrong password
    is a miss every single time (there is deliberately no negative cache --
    see below). What it should never have been is a per-request tax on a
    credential the process verified sixty seconds ago.

    Two caches, and the split is the whole design:

    * The parsed users DB, keyed by a `_DbStamp`. Re-read only when the file
      changes.
    * Per username, the SHA-256 of the secret that was verified, TOGETHER WITH
      the record material it was verified against (`salt_b64`, `hash_b64`,
      `iterations`).

    A hit requires the presented secret's digest to match AND the current
    record's material to be byte-identical to what was verified. That is what
    makes the invalidation sound rather than merely likely:

    * password changed -> `upsert_user` writes a fresh salt and hash ->
      material mismatch -> full re-verification, and the old password now
      fails as it should;
    * user disabled -> `disabled` is read from the CURRENT record on every
      request, never from the cache, so a revoked user is rejected on the next
      request with no grace period;
    * user deleted -> absent from the current DB -> rejected;
    * an upload -> `record_upload` bumps counters and `save_users` rewrites
      the file, so the STAMP changes but the material does not. This is why
      the cache is not keyed on the file stamp alone: uploads rewrite this
      file constantly, and a stamp-keyed cache would be flushed by traffic
      that cannot possibly have changed a password.

    ⚑ NO NEGATIVE CACHING. A wrong password re-runs PBKDF2 every time. That is
    the expensive direction on purpose: caching rejections would let an
    attacker learn "wrong" cheaply, and would need its own bounded eviction to
    avoid being a memory sink. The corollary is that this class does not
    protect against a credential-stuffing herd -- it protects against the
    legitimate one.

    One entry per username, holding one digest, so the map cannot be grown by
    an attacker: only a SUCCESSFUL verification writes to it. The digest
    comparison is `hmac.compare_digest`, so a hit does not leak the secret
    through comparison timing any more than `verify_password` does.

    A18: this is a `threading.Lock` used from `_auth_user`, which is a sync
    `def` dependency and therefore already runs in Starlette's threadpool.
    Nothing here goes near the event loop. The lock is held only across dict
    reads and writes -- never across PBKDF2, which would serialise the very
    work this class exists to avoid.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._stamp: _DbStamp | None = None
        self._users: dict[str, UserRecord] = {}
  # username -> (sha256(secret), salt_b64, hash_b64, iterations)
        self._verified: dict[str, tuple[bytes, str, str, int]] = {}
        self.pbkdf2_verifications = 0
        self.db_reads = 0

    def users(self) -> dict[str, UserRecord]:
        """The current users DB, re-read only when the file has changed.

        ⚑ Stat, read, re-stat. The read-then-stat order is the one that
        poisons a cache permanently: it can pair OLD content with the NEW
        stamp, and every later request then agrees the stamp is current. Here
        a write racing the read leaves the stamps unequal, so the fresh data
        is returned WITHOUT being cached and the next call re-reads. The cost
        of losing that race is one extra file read.
        """
        stamp = _DbStamp.of(self.path)
        with self._lock:
            if self._stamp == stamp:
                return self._users
        users = {} if not stamp.exists else load_users(self.path)
        self.db_reads += 1
        if _DbStamp.of(self.path) != stamp:
            return users
        with self._lock:
            self._stamp = stamp
            self._users = users
        return users

    def verify(self, username: str, secret: str) -> UserRecord | None:
        """The record for these credentials, or None if they do not authorise.

        `disabled` is deliberately NOT consulted here: the caller distinguishes
        401 from 403, and folding the two together in a cache is how a revoked
        user keeps working.
        """
        rec = self.users().get(str(username))
        if rec is None:
            return None
        digest = hashlib.sha256(str(secret).encode("utf-8")).digest()
        with self._lock:
            cached = self._verified.get(str(username))
        if cached is not None:
            want_digest, salt_b64, hash_b64, iterations = cached
  # Every leg must match: the secret AND the material it was checked
  # against. `compare_digest` on all three so a hit is constant-time in
  # the secret; the material legs are not secret but cost nothing.
            if (
                hmac.compare_digest(want_digest, digest)
                and hmac.compare_digest(salt_b64, str(rec.salt_b64))
                and hmac.compare_digest(hash_b64, str(rec.hash_b64))
                and int(iterations) == int(rec.iterations)
            ):
                return rec
        self.pbkdf2_verifications += 1
        if not verify_password(str(secret), rec):
            return None
        with self._lock:
            self._verified[str(username)] = (
                digest, str(rec.salt_b64), str(rec.hash_b64), int(rec.iterations),
            )
        return rec

    def invalidate(self) -> None:
        """Drop everything. For a caller that has just written the DB itself
        and does not want to depend on stamp resolution to notice."""
        with self._lock:
            self._stamp = None
            self._users = {}
            self._verified.clear()
