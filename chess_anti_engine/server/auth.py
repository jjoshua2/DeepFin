from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class UserRecord:
    username: str
    # PBKDF2-SHA256
    salt_b64: str
    iterations: int
    hash_b64: str

    disabled: bool = False

    # Stats (for leaderboards/moderation later)
    uploads: int = 0
    total_bytes: int = 0
    total_positions: int = 0
    last_upload_at_unix: int | None = None


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

    out: dict[str, UserRecord] = {}
    for username, v in data.items():
        if not isinstance(v, dict):
            continue
        out[str(username)] = UserRecord(username=str(username), **v)
    return out


def save_users(path: str | Path, users: dict[str, UserRecord]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    data: dict[str, Any] = {}
    for u, rec in users.items():
        # exclude username field (key is username)
        d = rec.__dict__.copy()
        d.pop("username", None)
        data[u] = d
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def record_upload(
    users: dict[str, UserRecord], *, username: str, bytes_uploaded: int, positions: int | None
) -> None:
    rec = users.get(username)
    if rec is None:
        return
    rec.uploads = int(rec.uploads) + 1
    rec.total_bytes = int(rec.total_bytes) + int(bytes_uploaded)
    if positions is not None:
        rec.total_positions = int(rec.total_positions) + int(positions)
    rec.last_upload_at_unix = int(time.time())


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


def set_disabled(users_path: str | Path, *, username: str, disabled: bool) -> None:
    users = load_users(users_path)
    if username not in users:
        raise ValueError(f"unknown user {username!r}")
    users[username].disabled = bool(disabled)
    save_users(users_path, users)
