"""Preflight only: immutable byte snapshots, never pickle/model loading.

Expected hashes must come from independently qualified receipts. Matching them
proves byte identity, not receipt authenticity or historical holdout independence.
The returned snapshots, not paths reopened later, bind any future qualified load.
The 64 MiB per-file limit is a bounded primitive, not operational readiness for
real checkpoints. All retained metadata remains caller-supplied declarations.
"""
from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import cast

HISTORY = "lc0_root_legacy_meta"
FEATURES = "v2_threats"
MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class Receipt:
    sha256: str
    authority_reference: str


@dataclass(frozen=True)
class LoadPlan:
    data: bytes
    donor: bytes
    candidate: bytes
    training_games: tuple[frozenset[str], frozenset[str]]
    receipts_declared: tuple[Receipt, ...]
    history_declared: str
    features_declared: str
    architectures_declared: tuple[str, ...]
    expected_architecture_declared: str
    holdout_games_declared: frozenset[str]
    memberships_complete_declared: tuple[bool, ...]
    inference_ready: bool = False
    provenance_authenticated: bool = False


def snapshot(path: Path, receipt: Receipt) -> bytes:
    """Read a bounded regular file once; validate against externally supplied hash."""
    if (len(receipt.sha256) != 64 or
            any(c not in "0123456789abcdef" for c in receipt.sha256) or
            not receipt.authority_reference.strip()):
        raise ValueError("independent expected SHA256 and authority reference required")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_BYTES:
            raise ValueError("bounded nonempty regular artifact required")
        content = stream.read(MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, f) != getattr(after, f) for f in fields):
        raise ValueError("artifact changed during snapshot")
    if len(content) != before.st_size or hashlib.sha256(content).hexdigest() != receipt.sha256:
        raise ValueError("artifact identity mismatch")
    return content


def prepare(
    paths: tuple[Path, ...], receipts: tuple[Receipt, ...], *,
    history: str, features: str, architectures: tuple[str, ...], expected_architecture: str,
    training_games: tuple[object, ...],
    memberships_complete: tuple[object, ...], holdout_games: object,
) -> LoadPlan:
    """Gate supplied declarations; does not decode data or inspect checkpoint metadata.

    Architecture is an exact canonical architecture JSON/digest supplied from an
    independently inspected checkpoint receipt, not guessed from filename. Both
    checkpoints must have the same supplied architecture. A real loader must
    separately inspect embedded metadata, require complete strict state loading,
    and verify WDL head/POV. This plan intentionally cannot run inference.
    """
    paths, receipts, architectures = tuple(paths), tuple(receipts), tuple(architectures)
    training_games = tuple(training_games)
    memberships_complete = tuple(memberships_complete)
    if history != HISTORY or features != FEATURES:
        raise ValueError("unsupported encoding; no fallback")
    if (not expected_architecture.strip() or len(architectures) != 2 or
            any(a != expected_architecture for a in architectures)):
        raise ValueError("explicit matching architecture receipt required")
    if len(paths) != 3 or len(receipts) != 3 or len(training_games) != 2:
        raise ValueError("data/donor/candidate and two memberships required")
    if memberships_complete != (True, True) or any(type(v) is not bool for v in memberships_complete):
        raise ValueError("complete membership declarations required")
    for members in (*training_games, holdout_games):
        if not isinstance(members, frozenset) or any(
            not isinstance(g, str) or not g.strip() for g in members
        ):
            raise ValueError("immutable composite source-game membership required")
    training = cast(tuple[frozenset[str], frozenset[str]], training_games)
    holdout = cast(frozenset[str], holdout_games)
    if len(holdout) < 2 or holdout & (training[0] | training[1]):
        raise ValueError("holdout overlap or insufficient source games")
    data, donor, candidate = (snapshot(p, r) for p, r in zip(paths, receipts, strict=True))
    return LoadPlan(data, donor, candidate, training, receipts, history, features,
                    architectures, expected_architecture, holdout,
                    cast(tuple[bool, ...], memberships_complete))
