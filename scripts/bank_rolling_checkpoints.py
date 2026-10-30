"""One identity-aware rolling-bank pass; the shell entrypoint owns polling.

A bank is a first copy of an immutable checkpoint location, not a live mirror.
Source stat signatures detect ordinary rewrites, not identical-stat content
mutation or symlink-target rewrites. trainer.pt must be a regular, non-symlink
file; optional sidecar links retain copy semantics, not independent snapshots.
Retention uses captured SOURCE directory mtime, not independent trial
indices or copy time. It is source-clock ordering, not proof of training progress.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass

_PROVENANCE = ".bank-provenance.json"
_CHECKPOINT = re.compile(r"checkpoint_(\d+)\Z")


class TransientSourceError(RuntimeError):
    """A new source was incomplete/unstable before any bank publication."""


@dataclass(frozen=True)
class Entry:
    identity: str
    index: int
    source: Path
    source_mtime_ns: int
    signature: str

    @property
    def name(self) -> str:
        return f"bank_{self.identity}__checkpoint_{self.index:06d}"

    def metadata(self) -> dict:
        return {"format_version": 1, "source_path": str(self.source),
                "checkpoint_index": self.index, "source_mtime_ns": self.source_mtime_ns,
                "source_signature": self.signature}


def _identity(source: Path) -> str:
    return hashlib.sha256(os.fsencode(source)).hexdigest()


def _signature(source: Path) -> str:
    """Bound ordinary local source changes without rereading large weights."""
    records = []
    for path in [source, *sorted(source.rglob("*"))]:
        s = path.lstat()
        records.append((str(path.relative_to(source)), s.st_mode, s.st_dev, s.st_ino,
                        s.st_size, s.st_mtime_ns, s.st_ctime_ns,
                        os.readlink(path) if stat.S_ISLNK(s.st_mode) else None))
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def _managed(path: Path) -> Entry | None:
    if (path.is_symlink() or not path.is_dir() or not path.name.startswith("bank_")
            or (path / "trainer.pt").is_symlink() or not (path / "trainer.pt").is_file()
            or (path / _PROVENANCE).is_symlink() or not (path / _PROVENANCE).is_file()):
        return None
    try:
        m = json.loads((path / _PROVENANCE).read_text(encoding="utf-8"))
        if not isinstance(m, dict) or type(m.get("format_version")) is not int or m["format_version"] != 1:
            return None
        source = Path(m["source_path"])
        index, clock, signature = m["checkpoint_index"], m["source_mtime_ns"], m["source_signature"]
        if (not source.is_absolute() or type(index) is not int or index < 0
                or type(clock) is not int or not isinstance(signature, str)
                or re.fullmatch(r"[0-9a-f]{64}", signature) is None):
            return None
        source_name = _CHECKPOINT.fullmatch(source.name)
        if source_name is None or int(source_name[1]) != index:
            return None
        entry = Entry(_identity(source), index, source, clock, signature)
        return entry if entry.name == path.name else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _sources(tune: Path, every: int, managed: dict[str, Entry]) -> dict[str, Entry]:
    entries: dict[str, Entry] = {}
    for trial in sorted(tune.glob("train_trial_*")):
        if not trial.is_dir():
            continue
        for checkpoint in sorted(trial.glob("checkpoint_0*")):
            match = _CHECKPOINT.fullmatch(checkpoint.name)
            if not checkpoint.is_dir() or match is None:
                continue
            index = int(match[1])
            if index % every:
                continue
            # A source can be pruned between is_dir and resolve. Keep its
            # canonical identity without requiring every component to survive.
            source = checkpoint.resolve()
            identity = _identity(source)
            source_name = _CHECKPOINT.fullmatch(source.name)
            if source_name is None or int(source_name[1]) != index:
                raise ValueError(f"ambiguous checkpoint-directory alias: {checkpoint}")
            try:
                if (source / _PROVENANCE).exists() or (source / _PROVENANCE).is_symlink():
                    raise ValueError(f"reserved bank metadata in source checkpoint: {source}")
                weights = source / "trainer.pt"
                if not weights.exists() and not weights.is_symlink():
                    if identity not in managed:
                        raise TransientSourceError(f"new source checkpoint is incomplete or disappeared: {source}")
                    if not source.exists():
                        continue  # Already-banked, now pruned: retain its bank.
                    raise ValueError(f"already-banked source lost trainer.pt: {source}")
                if weights.is_symlink() or not weights.is_file():
                    raise ValueError(f"source checkpoint requires regular non-symlink trainer.pt: {source}")
                signature = _signature(source)
                entry = Entry(identity, index, source, source.stat().st_mtime_ns, signature)
            except FileNotFoundError as exc:
                if identity not in managed:
                    raise TransientSourceError(f"new source disappeared during discovery: {source}") from exc
                if not source.exists():
                    continue
                raise ValueError(f"already-banked source changed during discovery: {source}") from exc
            if entry.identity in entries and entries[entry.identity] != entry:
                raise ValueError(f"ambiguous source checkpoint aliases: {source}")
            entries[entry.identity] = entry
    return entries


def _destination_outside_sources(tune: Path, destination: Path) -> Path:
    """Resolve aliases before creating even the destination or lock file."""
    protected = [tune.resolve(strict=True)]
    for trial in tune.glob("train_trial_*"):
        if trial.is_dir():
            protected.append(trial.resolve())
            protected.extend(checkpoint.resolve()
                             for checkpoint in trial.glob("checkpoint_0*")
                             if checkpoint.is_dir())
    resolved = destination.resolve()
    if any(resolved.is_relative_to(root) for root in protected):
        raise ValueError(f"BANK_DIR must be outside source tune/trial/checkpoint roots: {destination}")
    return resolved


def _require_unchanged_new_source(entry: Entry) -> None:
    try:
        signature = _signature(entry.source)
    except FileNotFoundError as exc:
        raise TransientSourceError(f"new source disappeared during checkpoint copy: {entry.source}") from exc
    if signature != entry.signature:
        raise TransientSourceError(f"new source changed during checkpoint copy: {entry.source}")


def bank_once(tune: Path, destination: Path, *, every: int, keep: int) -> None:
    """Rank before copying; trim only format/provenance-validated bank entries."""
    if every <= 0 or keep <= 0:
        raise ValueError("EVERY and KEEP must be positive integers")
    if not tune.is_dir():
        raise ValueError(f"no tune dir at {tune}")
    destination = _destination_outside_sources(tune, destination)
    destination.mkdir(parents=True, exist_ok=True)
    # All cooperating new writers serialize discovery, publication and trim.
    # Old versions ignore this lock and must not run against the same bank.
    lock = destination / ".bank.lock"
    if lock.exists() and not stat.S_ISREG(lock.lstat().st_mode):
        raise ValueError("reserved .bank.lock must be a regular file")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("reserved .bank.lock must be a regular file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        if os.fstat(fd).st_size:
            raise ValueError("nonempty reserved .bank.lock; refusing to treat it as bank control state")
        managed: dict[str, Entry] = {}
        legacy: list[Path] = []
        for path in destination.iterdir():
            if path.name == ".bank.lock":
                continue
            entry = _managed(path)
            if entry is None:
                legacy.append(path)
            else:
                managed[entry.identity] = entry
        capacity = keep - len(legacy)
        if capacity <= 0:
            raise ValueError(
                f"{len(legacy)} unidentified legacy entries consume KEEP={keep}; "
                "no capacity for managed copies (existing bank may be over budget). "
                "Legacy entries were not validated, migrated or deleted; choose BANK_DIR "
                "or explicitly resolve the legacy bank."
            )
        candidates = dict(managed)  # Retain provenance even after source pruning.
        for identity, entry in _sources(tune, every, managed).items():
            previous = managed.get(identity)
            if previous is not None and previous != entry:
                raise ValueError(f"source signature/provenance changed for already banked {entry.source}; "
                                 "immutable checkpoint-location contract violated")
            if previous is None and ((destination / entry.name).exists() or (destination / entry.name).is_symlink()):
                raise ValueError(f"untrusted existing bank entry blocks {entry.name}")
            candidates[identity] = entry
        selected = sorted(candidates.values(), key=lambda e: (e.source_mtime_ns, e.identity), reverse=True)[:capacity]
        selected_ids = {entry.identity for entry in selected}
        staged: list[tuple[Path, Entry]] = []
        published = 0
        try:
            # Stage only ranked winners. A failed copy cannot trigger eviction.
            for entry in selected:
                if entry.identity in managed:
                    continue
                temp = Path(tempfile.mkdtemp(prefix=".bank-stage-", dir=destination))
                staged.append((temp, entry))
                try:
                    shutil.copytree(entry.source, temp, dirs_exist_ok=True, symlinks=True)
                except OSError:
                    # Do not turn arbitrary destination/permission errors into
                    # retries. Only a proven source change is transient here.
                    _require_unchanged_new_source(entry)
                    raise
                _require_unchanged_new_source(entry)
                (temp / _PROVENANCE).write_text(json.dumps(entry.metadata(), sort_keys=True) + "\n", encoding="utf-8")
            for temp, entry in staged:
                target = destination / entry.name
                if target.exists() or target.is_symlink():
                    raise ValueError(f"bank target appeared during publication: {target}")
                temp.rename(target)
                published += 1
                print(f"[bank] banked checkpoint_{entry.index:06d} from {entry.source} -> {target.name}", flush=True)
            for identity, entry in managed.items():
                if identity not in selected_ids:
                    print(f"[bank] trimming managed {entry.name}", flush=True)
                    shutil.rmtree(destination / entry.name)
            # No claim that failures/temporary staging also obey this final cap.
            if len(list(destination.iterdir())) - 1 > keep:
                raise ValueError("bank contents changed concurrently; global KEEP was not realized")
        except BaseException:
            if published:
                print("[bank] ERROR: pass failed after publishing valid copies; "
                      "copies retained, global KEEP not confirmed; no rollback", file=sys.stderr)
            raise
        finally:
            for temp, _entry in staged:
                if temp.exists():
                    shutil.rmtree(temp)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--every", type=int, required=True)
    parser.add_argument("--keep", type=int, required=True)
    args = parser.parse_args()
    try:
        bank_once(args.tune_dir, args.destination, every=args.every, keep=args.keep)
    except TransientSourceError as exc:
        # bank_once's finally blocks must finish successfully to reach this.
        print(f"[bank] TEMPORARY SOURCE: {exc}; no new copies published", file=sys.stderr)
        raise SystemExit(75) from exc
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[bank] ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
