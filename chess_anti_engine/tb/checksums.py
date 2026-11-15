"""Full-content checksum verification of a Syzygy set (``deepfin --selfcheck``).

Header validation at ``SyzygyPath`` load (``uci/tb_root.validate_loaded``)
cannot see corruption inside a table's compressed data: in a fuzz of 300
randomly corrupted tables it refused about 167, and about 65 passed and then
answered WRONG WDL/DTZ with no warning (docs/TCEC_SETUP.md, section 7). Only
hashing every byte against a known-good list finds those, so this module does
that, and only from the self-check: hashing a 6-man set reads ~150 GB, which
must never sit in engine startup.

Checksum lists are the format the standard Syzygy distribution ships (the
lichess mirror's ``md5`` / ``sha256`` files, or a ``*.md5`` beside the
tables): GNU coreutils ``<hex>  <name>`` or ``<hex> *<name>`` (binary-mode
marker), one per line, CRLF tolerated, ``#`` comments and blank lines skipped;
BSD ``MD5 (name) = <hex>`` lines are read too. The algorithm follows from the
digest length (32 md5, 40 sha1, 64 sha256, 128 sha512). Entries are keyed by
BASENAME: Syzygy file names are unique per material, and distributions differ
in directory layout (``6-wdl/KQRvKR.rtbw`` and ``KQRvKR.rtbw`` are the same
table). Entries for anything but ``.rtbw`` / ``.rtbz`` are ignored.

Where the lists come from (:func:`resolve_lists`): the operator's explicit
lists if given; else every list file found inside the SyzygyPath directories
(:func:`discover_lists`); else the built-in manifest of the standard
3-4-5-6-man set shipped beside this module (:data:`BUILTIN_MANIFEST`, its
provenance in its header).
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

TABLE_SUFFIXES = (".rtbw", ".rtbz")
#: Shipped beside this module; see its header for where it came from.
BUILTIN_MANIFEST = "syzygy_standard_3-4-5-6.md5"
BUILTIN_LABEL = "built-in standard 3-4-5-6-man manifest"
_ALGO_BY_LEN = {32: "md5", 40: "sha1", 64: "sha256", 128: "sha512"}
_GNU_LINE = re.compile(r"^(?P<hex>[0-9A-Fa-f]+) (?P<mode>[ *])(?P<name>.+)$")
_BSD_LINE = re.compile(r"^(?P<algo>MD5|SHA1|SHA256|SHA512) \((?P<name>.+)\) = (?P<hex>[0-9A-Fa-f]+)$")
#: List files picked up inside a SyzygyPath directory: ``*.md5`` etc., any
#: ``checksum*`` file, and the lichess mirror's bare ``md5`` / ``sha256`` names.
_LIST_SUFFIXES = (".md5", ".sha1", ".sha256", ".sha512")
_LIST_BARE_NAMES = ("md5", "sha1", "sha256", "sha512")
#: Default hashing threads: the work is disk-bound, more threads only thrash.
DEFAULT_THREADS = 4
_CHUNK = 4 << 20


@dataclass(frozen=True)
class Entry:
    algo: str
    digest: str  # lowercase hex
    source: str  # "<list>:<line>"


@dataclass
class ChecksumList:
    """The merged table entries of one or more lists, plus what was wrong with them."""

    entries: dict[str, Entry] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _parse_line(line: str) -> tuple[str, str, str] | None:
    """(algo, digest, basename) of one list line, or None if it is malformed."""
    m = _GNU_LINE.match(line)
    if m is not None:
        name = m.group("name")
    else:
        m = _BSD_LINE.match(line)
        if m is None:
            return None
        name = m.group("name")
    digest = m.group("hex").lower()
    algo = _ALGO_BY_LEN.get(len(digest))
    if algo is None:
        return None
    if "algo" in m.groupdict() and m.group("algo").lower() != algo:
        return None
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    if not base:
        return None
    return algo, digest, base


def parse_lists(sources: Iterable[tuple[str, str]]) -> ChecksumList:
    """Merge ``(label, text)`` checksum lists into one table-name -> digest map.

    A malformed line, or the same table given two different digests of the
    same algorithm (in one list or across lists), is a problem: a list that
    cannot be read completely is not evidence for anything it covers.
    """
    out = ChecksumList()
    for label, text in sources:
        out.sources.append(label)
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.rstrip("\r\n").rstrip("\r")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parsed = _parse_line(line)
            if parsed is None:
                out.problems.append(f"{label}:{lineno}: malformed checksum line {line[:80]!r}")
                continue
            algo, digest, base = parsed
            if not base.endswith(TABLE_SUFFIXES):
                continue
            where = f"{label}:{lineno}"
            prior = out.entries.get(base)
            if prior is None:
                out.entries[base] = Entry(algo, digest, where)
            elif prior.algo == algo and prior.digest != digest:
                out.problems.append(
                    f"{base}: conflicting {algo} checksums ({prior.source} {prior.digest}, "
                    f"{where} {digest})")
            # A second algorithm for the same table adds nothing: one is checked.
    return out


def discover_lists(dirs: Sequence[str]) -> list[Path]:
    """Checksum list files inside the SyzygyPath directories (not recursive)."""
    found: list[Path] = []
    for d in dirs:
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for name in names:
            low = name.lower()
            if low.endswith(TABLE_SUFFIXES):
                continue
            if low.endswith(_LIST_SUFFIXES) or low.startswith("checksum") or low in _LIST_BARE_NAMES:
                p = Path(d) / name
                if p.is_file():
                    found.append(p)
    return found


def builtin_text() -> str | None:
    try:
        return (resources.files("chess_anti_engine.tb") / BUILTIN_MANIFEST).read_text(encoding="utf-8")
    except (OSError, ModuleNotFoundError):
        return None


def resolve_lists(
    dirs: Sequence[str], explicit: Sequence[str] = (),
) -> tuple[ChecksumList, str]:
    """The lists to verify against and a phrase saying where they came from.

    ``explicit`` (``--syzygy-checksums``, repeatable) replaces discovery; the
    word ``builtin`` there names :data:`BUILTIN_MANIFEST`. With no explicit
    list, lists found in the directories are used, else the built-in manifest.
    An explicit list that cannot be read is a problem, never skipped.
    """
    texts: list[tuple[str, str]] = []
    problems: list[str] = []

    def _read(p: str) -> None:
        try:
            texts.append((p, Path(p).read_text(encoding="utf-8", errors="strict")))
        except (OSError, UnicodeDecodeError) as exc:
            problems.append(f"checksum list {p}: cannot be read ({exc})")

    def _builtin() -> None:
        text = builtin_text()
        if text is None:
            problems.append(f"the {BUILTIN_LABEL} is missing from this install")
        else:
            texts.append((BUILTIN_LABEL, text))

    if explicit:
        for p in explicit:
            if p == "builtin":
                _builtin()
            else:
                _read(p)
        origin = "the given list(s)"
    else:
        found = discover_lists(dirs)
        if found:
            for p in found:
                _read(str(p))
            origin = "list(s) found in the path"
        else:
            text = builtin_text()
            if text is None:
                # Nothing to verify against: reported as NOT VERIFIED (a SKIP,
                # or a FAIL under --syzygy-require-checksums), never a pass.
                origin = f"no checksum list (none in the path; the {BUILTIN_LABEL} is missing)"
            else:
                texts.append((BUILTIN_LABEL, text))
                origin = f"the {BUILTIN_LABEL} (no checksum list found in the path)"
    merged = parse_lists(texts)
    merged.problems[:0] = problems
    return merged, origin


def table_files(dirs: Sequence[str]) -> list[Path]:
    """Every ``.rtbw`` / ``.rtbz`` entry in the directories, broken symlinks included."""
    files: list[Path] = []
    for d in dirs:
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        files += [Path(d) / n for n in names if n.endswith(TABLE_SUFFIXES)]
    return files


@dataclass
class VerifyReport:
    lists: ChecksumList
    origin: str
    files: int = 0
    total_bytes: int = 0
    seconds: float = 0.0
    threads: int = 0
    ok: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    unlisted: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        """Evidence of a bad set or an unreadable list: always a FAIL."""
        return bool(self.mismatched or self.unreadable or self.lists.problems)

    @property
    def unverified(self) -> bool:
        """Some table file has no checksum to check against (or no list at all)."""
        return bool(self.unlisted)


def _hash_file(path: Path, algo: str, on_bytes: Callable[[int], None]) -> str:
    h = hashlib.new(algo)
    with open(path, "rb", buffering=0) as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)  # releases the GIL on large buffers: threads overlap
            on_bytes(len(chunk))
    return h.hexdigest()


ProgressFn = Callable[[int, int, int, int], None]  # (files done, files, bytes done, bytes)


def verify(
    dirs: Sequence[str], lists: ChecksumList, origin: str, *,
    threads: int = DEFAULT_THREADS, progress: ProgressFn | None = None,
    progress_interval_s: float = 5.0,
) -> VerifyReport:
    """Hash every table file in ``dirs`` against ``lists``.

    Each file is ok, mismatched, unreadable (including a broken symlink) or
    unlisted; each listed table with no file of that name in any directory is
    missing. Unlisted files are not hashed. ``threads`` is clamped to
    ``[1, 32]``; reads are sequential within a file.
    """
    report = VerifyReport(lists=lists, origin=origin, threads=max(1, min(32, int(threads))))
    files = table_files(dirs)
    report.files = len(files)
    present = {p.name for p in files}
    report.missing = sorted(set(lists.entries) - present)
    work: list[tuple[Path, Entry, int]] = []
    for p in files:
        entry = lists.entries.get(p.name)
        if entry is None:
            report.unlisted.append(str(p))
            continue
        try:
            size = p.stat().st_size  # follows symlinks: a broken one raises
        except OSError as exc:
            report.unreadable.append(f"{p} ({exc.strerror or exc})")
            continue
        work.append((p, entry, size))
    report.total_bytes = sum(size for _, _, size in work)
    lock = threading.Lock()
    done_bytes = 0
    done_files = 0
    last = time.monotonic()

    def _tick(n: int) -> None:
        nonlocal done_bytes, last
        with lock:
            done_bytes += n
            now = time.monotonic()
            if progress is not None and now - last >= progress_interval_s:
                last = now
                progress(done_files, len(work), done_bytes, report.total_bytes)

    def _one(item: tuple[Path, Entry, int]) -> tuple[Path, Entry, str | None, str | None]:
        nonlocal done_files
        p, entry, _ = item
        try:
            got: str | None = _hash_file(p, entry.algo, _tick)
            err = None
        except OSError as exc:
            got, err = None, str(exc.strerror or exc)
        with lock:
            done_files += 1
        return p, entry, got, err

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=report.threads, thread_name_prefix="tbsum") as pool:
        for p, entry, got, err in pool.map(_one, work):
            if err is not None:
                report.unreadable.append(f"{p} ({err})")
            elif got != entry.digest:
                report.mismatched.append(f"{p} ({entry.algo} {got}, list {entry.digest} at {entry.source})")
            else:
                report.ok.append(str(p))
    report.seconds = time.monotonic() - start
    if progress is not None:
        progress(done_files, len(work), done_bytes, report.total_bytes)
    return report


def _names(items: Sequence[str], cap: int | None = None) -> str:
    if cap is None or len(items) <= cap:
        return ", ".join(items)
    return ", ".join(items[:cap]) + f" and {len(items) - cap} more"


#: Listed-but-absent names shown before "and N more" (a list usually covers
#: more than one install holds, e.g. the mirror's list includes 7-man tables).
MISSING_SHOWN = 20


def describe(report: VerifyReport) -> str:
    """One self-check line. Every FAIL-class and unverified file is named in full."""
    parts: list[str] = []
    if report.lists.problems:
        parts.append(f"checksum list problems: {_names(report.lists.problems)}")
    if report.mismatched:
        parts.append(f"{len(report.mismatched)} CHECKSUM MISMATCH(ES), replace these files: "
                     f"{_names(report.mismatched)}")
    if report.unreadable:
        parts.append(f"{len(report.unreadable)} unreadable table file(s): {_names(report.unreadable)}")
    if report.files and not report.lists.sources:
        # No list at all: every file is unverified; naming 1000+ paths adds nothing.
        parts.append(f"NOT VERIFIED: no checksum list ({report.files} table files unchecked)")
    elif report.unlisted:
        parts.append(f"NOT VERIFIED: {len(report.unlisted)} table file(s) in no checksum list: "
                     f"{_names(report.unlisted)}")
    gb = report.total_bytes / 1e9
    rate = (report.total_bytes / 1e6 / report.seconds) if report.seconds > 0 else 0.0
    parts.append(
        f"{len(report.ok)} of {report.files} table files match {report.origin} "
        f"({len(report.lists.entries)} entries); {gb:.1f} GB hashed in {report.seconds:.0f} s "
        f"({rate:.0f} MB/s, {report.threads} threads)")
    if report.missing:
        parts.append(f"{len(report.missing)} listed table(s) not in the path: "
                     f"{_names(report.missing, MISSING_SHOWN)}")
    return "; ".join(parts)
