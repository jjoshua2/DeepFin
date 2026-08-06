from __future__ import annotations

import base64
import contextlib
import dataclasses
import fcntl
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_anti_engine.utils.atomic import atomic_write_text

_log = logging.getLogger("chess_anti_engine.server")


@dataclass
class UserRecord:
    """A credential. Nothing here changes except by an explicit admin action.

    Upload counters used to live on this record, which meant every accepted
    shard rewrote the credential file. See :class:`UserStats`.
    """

    username: str
  # PBKDF2-SHA256
    salt_b64: str
    iterations: int
    hash_b64: str

    disabled: bool = False


@dataclass
class UserStats:
    """Per-user upload counters. Telemetry — a crash may discard the last write.

    Split out of :class:`UserRecord` because keeping the two in one file forced
    a choice with no good answer: every accepted shard rewrote the whole file,
    so either credentials paid an fsync per upload, or (as shipped in #344)
    credentials were rewritten NON-durably several times an iteration and an
    unclean shutdown could cost a just-created password. Separate files make
    both questions independent — this one is written per upload and
    non-durably, ``users.json`` only on an admin action and always durably.
    """

    uploads: int = 0
    total_bytes: int = 0
    total_positions: int = 0
    last_upload_at_unix: int | None = None
  # Per-machine stats: machine_id -> {uploads, positions, last_upload_at_unix}
    machines: dict[str, dict[str, Any]] = field(default_factory=dict)


# Counter keys that used to live in `users.json`. Read once by
# `migrate_user_stats`, then filtered out of every load so a legacy file still
# parses into the slimmed `UserRecord`.
_LEGACY_STAT_KEYS = frozenset({
    "uploads", "total_bytes", "total_positions", "last_upload_at_unix", "machines",
})


def user_stats_path_for(users_path: str | Path) -> Path:
    """The stats file paired with a users DB (``users.json`` -> ``users.stats.json``).

    Derived from the DB name rather than fixed, so two servers sharing a
    directory with different ``--users-db`` names do not collide.
    """
    p = Path(users_path)
    return p.with_name(f"{p.stem}.stats.json")


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


MIN_PASSWORD_LENGTH = 8
"""Minimum length for a password being SET. User-specified policy, 2026-08-05.

⚑ ENFORCEMENT IS ON SETTING, NOT ON CHECKING. :func:`verify_password` is
deliberately untouched by this: accounts created before the policy keep
authenticating with whatever they have, so raising the bar cannot lock the
fleet out mid-rotation. The bar applies the next time a password is chosen --
`manage_users add` / `set-password`, and self-registration.
"""


class WeakPassword(ValueError):
    """A password being SET does not meet :data:`MIN_PASSWORD_LENGTH`."""


def check_new_password(password: str) -> None:
    """Raise :class:`WeakPassword` unless `password` may be SET on an account.

    The single home for the policy: every route that chooses a password --
    both `manage_users` subcommands across all three input sources, and
    self-registration -- calls this, so there is one place to read and one
    place to change. Callers translate it to their own failure mode (a CLI
    exits, an HTTP route returns 400).
    """
    if not password.strip():
        raise WeakPassword(
            "refusing an empty (or whitespace-only) password: it hashes and "
            "stores fine, and then authenticates a client that sends nothing"
        )
    if len(password) < MIN_PASSWORD_LENGTH:
        raise WeakPassword(
            f"password must be at least {MIN_PASSWORD_LENGTH} characters; "
            f"got {len(password)}"
        )


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

    out: dict[str, UserRecord] = {}
    for username, v in data.items():
        if not isinstance(v, dict):
            continue
  # Drop ONLY the counter keys that used to live here, so a pre-split
  # file still parses. Anything else unexpected still raises TypeError
  # rather than being silently ignored.
        fields = {k: val for k, val in v.items() if k not in _LEGACY_STAT_KEYS}
        out[str(username)] = UserRecord(username=str(username), **fields)
    return out


def save_users(path: str | Path, users: dict[str, UserRecord]) -> None:
    """Write the users DB. Always durable — losing a just-created password to
    an unclean shutdown locks a worker out of the fleet.

    There is deliberately no ``durable=False`` escape hatch any more. The one
    caller that wanted it was the per-upload counter write, and counters now
    live in their own file (:func:`save_user_stats`); re-adding the parameter
    would re-open the hole where an upload rewrites credentials non-durably.
    """
    data: dict[str, Any] = {}
    for u, rec in users.items():
  # exclude username field (key is username)
        d = rec.__dict__.copy()
        d.pop("username", None)
        data[u] = d
  # ⚑ 0600, applied to the tmp file BEFORE the rename. This file holds the
  # PBKDF2 material for every account; at `umask 000` it was created 0666, and a
  # WORLD-WRITABLE users.json is an auth bypass -- any local user can drop in a
  # hash they know and log in as anybody. Disclosure is the lesser half.
    atomic_write_text(
        Path(path), json.dumps(data, indent=2, sort_keys=True), mode=0o600,
    )


def load_user_stats(path: str | Path) -> dict[str, UserStats]:
    """Read the per-user counter file. A missing or corrupt file is empty.

    Corrupt is tolerated HERE, unlike the users DB, because these are counters:
    the alternative is refusing uploads over unreadable telemetry. It is logged
    so the degradation is not silent.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _log.warning(
            "user stats file %s is unreadable (%s: %s); counters restart from "
            "zero — uploads are unaffected", p, type(exc).__name__, exc,
        )
        return {}
    if not isinstance(data, dict):
        _log.warning("user stats file %s is not a dict; counters restart from zero", p)
        return {}
    out: dict[str, UserStats] = {}
    for username, v in data.items():
        if not isinstance(v, dict):
            continue
        out[str(username)] = UserStats(
            uploads=int(v.get("uploads", 0) or 0),
            total_bytes=int(v.get("total_bytes", 0) or 0),
            total_positions=int(v.get("total_positions", 0) or 0),
            last_upload_at_unix=(
                int(v["last_upload_at_unix"])
                if v.get("last_upload_at_unix") is not None else None
            ),
            machines=dict(v.get("machines") or {}),
        )
    return out


def save_user_stats(
    path: str | Path, stats: dict[str, UserStats], *, durable: bool = False,
) -> None:
    """Write the per-user counter file.

    ``durable`` defaults to FALSE: this is written on every accepted shard, an
    fsync costs ~11 ms median on the project's ext4 root (tail past 1.8 s), and
    the content is counters that a crash may discard. Nothing recovery-critical
    lives here — that was the point of splitting it out of ``users.json``.
    """
    data = {u: dataclasses.asdict(s) for u, s in stats.items()}
    atomic_write_text(
        Path(path), json.dumps(data, indent=2, sort_keys=True), durable=durable,
    )


def migrate_user_stats(users_path: str | Path, stats_path: str | Path) -> int:
    """One-time move of legacy counters out of ``users.json``. Returns the
    number of users whose counters were carried over.

    Idempotent by construction: it ASSIGNS the legacy values rather than adding
    them, and it returns immediately once the stats file exists. So neither a
    repeat call nor a crash mid-migration can double-count.

    Order is deliberate — the stats file is written (durably, the one time it
    is) BEFORE the legacy keys are stripped from ``users.json``. A crash
    between the two leaves the counters in both places, which the next load
    resolves in favour of the stats file; the reverse order could lose them.
    """
    users_p, stats_p = Path(users_path), Path(stats_path)
    if stats_p.exists() or not users_p.exists():
        return 0
    try:
        raw = json.loads(users_p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _log.warning(
            "cannot read %s to migrate upload counters (%s: %s); counters start "
            "from zero", users_p, type(exc).__name__, exc,
        )
        return 0
    if not isinstance(raw, dict):
        return 0

    carried: dict[str, UserStats] = {}
    for username, v in raw.items():
        if not isinstance(v, dict) or not (_LEGACY_STAT_KEYS & set(v)):
            continue
        carried[str(username)] = UserStats(
            uploads=int(v.get("uploads", 0) or 0),
            total_bytes=int(v.get("total_bytes", 0) or 0),
            total_positions=int(v.get("total_positions", 0) or 0),
            last_upload_at_unix=(
                int(v["last_upload_at_unix"])
                if v.get("last_upload_at_unix") is not None else None
            ),
            machines=dict(v.get("machines") or {}),
        )
    if not carried:
        return 0

    save_user_stats(stats_p, carried, durable=True)
    stripped = {
        u: {k: val for k, val in v.items() if k not in _LEGACY_STAT_KEYS}
        for u, v in raw.items() if isinstance(v, dict)
    }
  # ⚑ mode=0o600, matching `save_users`. This is the one other writer of the
  # credential file, and it rewrites the WHOLE of it: without the mode it
  # re-created users.json at the umask default, so a server whose credential
  # file was correctly 0600 came back from this one-time migration at 0644 --
  # or 0666 under `umask 000` -- and a world-WRITABLE users.json is an auth
  # bypass by hash replacement. A migration that silently widens permissions is
  # worse than one that fails.
    atomic_write_text(
        users_p, json.dumps(stripped, indent=2, sort_keys=True), mode=0o600,
    )
    _log.info(
        "migrated upload counters for %d user(s) out of %s into %s; the "
        "credential file is no longer written by uploads",
        len(carried), users_p, stats_p,
    )
    return len(carried)


def record_upload(
    stats: dict[str, UserStats],
    *,
    username: str,
    bytes_uploaded: int,
    positions: int | None,
    machine_id: str | None = None,
) -> None:
    """Accumulate one upload into ``stats`` in place.

    Unlike the pre-split version this creates the entry when absent: the stats
    file starts empty after migration and is not seeded with the user list. The
    caller has already authenticated ``username``, so this cannot mint counters
    for an unknown user.
    """
    rec = stats.get(str(username))
    if rec is None:
        rec = UserStats()
        stats[str(username)] = rec
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


USERS_DB_LOCK_SUFFIX = ".lock"
_USERS_DB_LOCK_TIMEOUT_S = 10.0


class UsersDbBusy(RuntimeError):
    """The users-DB lock was held past the deadline. Raised, never bypassed."""


@contextlib.contextmanager
def users_db_lock(
    users_path: str | Path, *, timeout_s: float = _USERS_DB_LOCK_TIMEOUT_S,
) -> Generator[None]:
    """Cross-process mutual exclusion for a read-modify-write of `users.json`.

    ⚑ THE THREADING LOCK IS NOT ENOUGH, AND THE GAP IS NEWLY REACHABLE. Until
    self-registration landed, the server never wrote `users.json` in steady
    state -- the upload counters had moved to their own file -- so `manage_users`
    was the only writer and an unlocked read-modify-write could not lose
    anything. TOFU registration made the SERVER a writer again, and it is a
    different PROCESS from the operator's CLI, which `stats_write_lock` cannot
    see. Measured before this guard existed, with the operator disabling an
    account during a registration:

        alice.disabled = False   (operator set it True)
        LOST UPDATE: the revocation was silently reverted

    That is the dangerous direction: the server writes back the copy it loaded
    before the disable, and a revoked worker keeps uploading. The reverse
    interleave loses a just-registered account instead, which is merely
    annoying. Both are the same bug.

    `flock` rather than an O_EXCL lock file, and the difference is the whole
    reason A17 needed a staleness test: the kernel drops an flock when the
    holder dies, so a crashed process cannot leave a lock nobody can clear.
    There is nothing to steal and no staleness heuristic to get wrong.

    Blocks up to `timeout_s`, then raises `UsersDbBusy` rather than proceeding
    unlocked -- an RMW that gives up on the lock and writes anyway is the
    original bug with extra steps. Callers in a request path turn that into a
    503; the CLI turns it into a message naming the other holder.

    A18: this is a blocking file lock, so it must only ever be taken from a
    sync (threadpool) context, never from the event loop.
    """
    path = Path(users_path)
    lock_path = path.with_name(path.name + USERS_DB_LOCK_SUFFIX)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
  # 0600: the lock's mere existence is not a secret, but it sits beside the
  # credential file and inherits its blast radius if a mode is ever added.
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + float(timeout_s)
    announced = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if not announced:
  # ⚑ SAY WHY, ONCE, THE MOMENT WE START WAITING. The operator-visible
  # symptom of contention is `manage_users` sitting there doing nothing,
  # and the only thing that can explain it is this line -- the holder's
  # pid is recorded in the lock file and nowhere else. Emitted on first
  # contention rather than at the deadline, because a CLI that pauses for
  # ten seconds and THEN explains itself has already been killed.
                    announced = True
                    with contextlib.suppress(Exception):
                        _log.warning(
                            "waiting for the users-db lock %s, held by pid %s",
                            lock_path,
                            os.pread(fd, 64, 0).decode("utf-8", "replace").strip()
                            or "unknown",
                        )
                if time.monotonic() >= deadline:
                    holder = ""
                    with contextlib.suppress(Exception):
                        holder = os.pread(fd, 64, 0).decode("utf-8", "replace").strip()
                    raise UsersDbBusy(
                        f"{lock_path} held by pid {holder or 'unknown'} for more "
                        f"than {timeout_s:g}s; refusing to write {path.name} "
                        f"unlocked"
                    ) from None
                time.sleep(0.02)
        with contextlib.suppress(Exception):
            os.ftruncate(fd, 0)
            os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
        yield
    finally:
        with contextlib.suppress(Exception):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(Exception):
            os.close(fd)


def ensure_user(
    users_path: str | Path,
    *,
    username: str,
    password: str,
    disabled: bool = False,
) -> None:
  # ⚑ HASH OUTSIDE THE LOCK. PBKDF2 at 200k iterations is ~50ms; holding a
  # cross-process lock across it would serialise the KDF between the server
  # and the CLI for no benefit -- only the load-mutate-save has to be atomic.
    salt_b64, hash_b64, iterations = hash_password(password)
    with users_db_lock(users_path):
        users = load_users(users_path)
        if username in users:
            raise ValueError(f"user {username!r} already exists")
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
    """Create or update (re-hash) a user's password.

    No longer has to carry counters across the rewrite: they live in the stats
    file and this function does not touch it, so a password change cannot zero
    a contributor's upload history by forgetting a field.
    """
    salt_b64, hash_b64, iterations = hash_password(password)
    with users_db_lock(users_path):
        users = load_users(users_path)
        users[username] = UserRecord(
            username=username,
            salt_b64=salt_b64,
            iterations=iterations,
            hash_b64=hash_b64,
            disabled=bool(disabled),
        )
        save_users(users_path, users)


def set_disabled(users_path: str | Path, *, username: str, disabled: bool) -> None:
    with users_db_lock(users_path):
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
    * an upload -> counters live in their own file since the users.json split,
      so an upload does not touch this file at all and the stamp does not
      move. The material check is kept anyway: it is what makes invalidation
      SOUND rather than merely likely, and it is the leg that survives if some
      future caller starts rewriting users.json for an unrelated reason.

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
  # Per-PROCESS random key, so what this cache retains is useless outside
  # this process. A bare sha256(password) is an offline-crackable digest of
  # a live credential that outlives the plaintext (which is freed with the
  # request) for the process lifetime; keyed, a memory dump taken later
  # yields nothing without the key, which dies with the process. Behaviour
  # is unchanged -- it is still a constant-length constant-time comparison.
        self._digest_key = os.urandom(32)
  # username -> (keyed digest of secret, salt_b64, hash_b64, iterations)
        self._verified: dict[str, tuple[bytes, str, str, int]] = {}
        self.pbkdf2_verifications = 0
        self.db_reads = 0

    def _secret_digest(self, secret: str) -> bytes:
        return hmac.new(self._digest_key, str(secret).encode("utf-8"), hashlib.sha256).digest()

    def users(self) -> dict[str, UserRecord]:
        """The current users DB, re-read only when the file has changed.

        ⚑ Stat, read, re-stat. The read-then-stat order is the one that
        poisons a cache permanently: it can pair OLD content with the NEW
        stamp, and every later request then agrees the stamp is current. Here
        a write racing the read leaves the stamps unequal, so the fresh data
        is returned WITHOUT being cached and the next call re-reads. The cost
        of losing that race is one extra file read.

        ⚑ Returns a COPY. Handing out `self._users` let a caller mutate the
        cached DB in place -- `record_upload(cache.users(), ...)` was the
        obvious one before the counter split -- which poisons the cache
        without touching the file, and the stamp design is structurally blind
        to that: no write, no new inode, no re-read, wrong data forever. A
        shallow copy is enough for the dict itself; `verify` still returns the
        live record because the caller only reads `disabled` off it.
        """
        stamp = _DbStamp.of(self.path)
        with self._lock:
            if self._stamp == stamp:
                return dict(self._users)
        users = {} if not stamp.exists else load_users(self.path)
        self.db_reads += 1
        if _DbStamp.of(self.path) != stamp:
            return users
        with self._lock:
            self._stamp = stamp
            self._users = users
        return dict(users)

    def verify(self, username: str, secret: str) -> UserRecord | None:
        """The record for these credentials, or None if they do not authorise.

        `disabled` is deliberately NOT consulted here: the caller distinguishes
        401 from 403, and folding the two together in a cache is how a revoked
        user keeps working.
        """
        rec = self.users().get(str(username))
        if rec is None:
            return None
        digest = self._secret_digest(secret)
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
