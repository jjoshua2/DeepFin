"""Crash-resilient per-game persistence + resume for the match drivers.

Both match drivers (``scripts/arena_standard.py``, ``scripts/match_vs_uci.py``)
persist ONE JSONL record per FINISHED game, flushed immediately, and can resume
from that file. The motivating failure (2026-08-21): a 128-game compiled arena
died of CUDA OOM at ply 20 with ZERO games persisted, and the relaunch lost its
first minutes the same way. Hours of GPU time produced nothing scoreable
because the only durable artifact was written at the END of the run.

Shape of a log::

    {"kind": "header", "version": 1, "driver": "arena_standard", "settings": {...},
     "fingerprint": "a1b2c3d4e5f6", "created": "2026-08-21T10:00:00"}
    {"kind": "game", "pair_id": 0, "half": 0, ...}
    {"kind": "game", "pair_id": 0, "half": 1, ...}

Two properties do the work:

* **Flush per game.** ``write_game`` flushes, so a SIGKILL/OOM loses at most
  the games still in flight. Deliberately NOT ``fsync``: the failure mode this
  guards is a dead PROCESS (OOM, ``timeout -k``), and the page cache survives
  that. Power loss is out of scope and paying an fsync per game to pretend
  otherwise would be a cost bought for nothing.
* **A settings fingerprint in the header.** A resume onto different settings
  would silently mix two populations into one Elo, so ``fingerprint_differences``
  names every setting that moved and the caller refuses. Accepting the resume
  and quietly averaging two rulers is this repo's signature defect.

The log is append-only and may therefore contain a game index MORE THAN ONCE
(a resume replays whatever the crash left half-finished). ``latest_rows_by_key``
is how a reader collapses that: last write wins.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

# The key a reader collapses repeated game rows on: ``(pair_id, half)`` for the
# paired arena, a plain game index for the UCI driver. Generic so each caller
# keeps its own key TYPE and can unpack it — a bare ``Hashable`` would make
# every downstream use an unchecked cast.
_KeyT = TypeVar("_KeyT", bound=Hashable)

GAME_LOG_VERSION = 1
KIND_HEADER = "header"
KIND_GAME = "game"


def settings_fingerprint(settings: Mapping[str, Any]) -> str:
    """12 hex chars of sha256 over the canonical JSON of ``settings``.

    Strict on purpose: a value JSON cannot serialize raises here rather than
    being coerced with ``default=str``. A coerced object would usually
    stringify to a repr carrying its memory address, so the fingerprint would
    differ on every run and ``--resume`` would refuse a log it wrote itself —
    a guard that always fires is as useless as one that never does.
    """
    payload = json.dumps(dict(settings), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def fingerprint_differences(
    recorded: Mapping[str, Any], current: Mapping[str, Any],
) -> list[str]:
    """``key: recorded -> current`` for every setting that is not identical.

    Includes keys present on only one side (shown as ``<absent>``), because a
    driver that GREW a resume-critical setting must not silently treat the old
    log as compatible.
    """
    out: list[str] = []
    for key in sorted(set(recorded) | set(current)):
        was = recorded.get(key, "<absent>")
        now = current.get(key, "<absent>")
        if was != now:
            out.append(f"  {key}: {was!r} -> {now!r}")
    return out


@dataclass(frozen=True)
class GameLog:
    """A parsed game log: its header and every game row, in file order."""

    header: dict[str, Any] = field(default_factory=dict)
    games: list[dict[str, Any]] = field(default_factory=list)
    truncated_tail: bool = False

    @property
    def settings(self) -> dict[str, Any]:
        settings = self.header.get("settings")
        return dict(settings) if isinstance(settings, dict) else {}

    @property
    def fingerprint(self) -> str:
        return str(self.header.get("fingerprint", ""))


def read_game_log(path: str | Path) -> GameLog:
    """Parse a game log, tolerating a half-written FINAL line.

    A crash mid-``write`` can leave the last line truncated; that game is
    dropped (it was in flight, so it is exactly the game a resume must replay).
    A malformed line ANYWHERE ELSE is corruption, not a crash artifact, and
    raises — silently skipping it would drop finished games from the score.
    """
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    header: dict[str, Any] = {}
    games: list[dict[str, Any]] = []
    truncated_tail = False
    last = len(lines) - 1
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if i == last:
                truncated_tail = True
                continue
            raise ValueError(
                f"{p}: line {i + 1} is not valid JSON ({exc}); the log is "
                "corrupt beyond a crash-truncated tail — do not resume it"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"{p}: line {i + 1} is not a JSON object")
        kind = row.get("kind")
        if kind == KIND_HEADER:
            if header:
                raise ValueError(f"{p}: line {i + 1} is a second header row")
            header = row
        elif kind == KIND_GAME:
            games.append(row)
        else:
            raise ValueError(f"{p}: line {i + 1} has unknown kind {kind!r}")
    if not header:
        raise ValueError(
            f"{p}: no header row — this is not a game log written by "
            "GameLogWriter (refusing to guess its settings)"
        )
    return GameLog(header=header, games=games, truncated_tail=truncated_tail)


def latest_rows_by_key(
    rows: Iterable[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], _KeyT],
) -> dict[_KeyT, dict[str, Any]]:
    """Collapse repeated game rows to the LAST write of each key.

    A resumed run replays the games the crash left unfinished, so the same
    ``(pair_id, half)`` / ``game_index`` can appear twice. The later row is the
    one that was actually played to completion in the surviving run.
    """
    out: dict[_KeyT, dict[str, Any]] = {}
    for row in rows:
        out[key(row)] = dict(row)
    return out


class GameLogWriter:
    """Append-only JSONL of finished games, flushed after every record."""

    def __init__(
        self,
        path: str | Path,
        *,
        driver: str,
        settings: Mapping[str, Any],
        resuming: bool = False,
    ) -> None:
        self.path = Path(path)
        self.driver = driver
        self.settings = dict(settings)
        self.fingerprint = settings_fingerprint(self.settings)
        self.games_written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        had_content = self.path.exists() and self.path.stat().st_size > 0
        self._fh = self.path.open("a", encoding="utf-8")
        if not (resuming and had_content):
            self._write({
                "kind": KIND_HEADER,
                "version": GAME_LOG_VERSION,
                "driver": driver,
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "fingerprint": self.fingerprint,
                "settings": self.settings,
            })

    def __enter__(self) -> GameLogWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def write_game(self, record: Mapping[str, Any]) -> None:
        row: dict[str, Any] = {"kind": KIND_GAME}
        row.update(record)
        row.setdefault("ts", datetime.datetime.now().isoformat(timespec="seconds"))
        self._write(row)
        self.games_written += 1

    def _write(self, row: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()


def sanitize_path_component(name: str, *, fallback: str, max_len: int = 48) -> str:
    """Filesystem-safe, readable component for a default log filename."""
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in str(name)]
    out = "".join(keep).strip("._") or fallback
    return out[:max_len]


def default_game_log_path(
    directory: str | Path, *, label: str | None, fingerprint: str,
) -> Path:
    """``<directory>/<label>.<fingerprint>.games.jsonl``.

    The fingerprint is IN THE NAME, which is what makes an always-on game log
    safe for the drivers' shared, append-only result files: two runs that
    differ in any resume-critical setting cannot land in the same log, so they
    cannot mix, and two runs that agree in every one of them land in the same
    log and hit the "already exists" refusal. Deriving the name from the
    driver's ``--out`` instead would give every arena in history ONE shared
    games file (``runs/arena_results.jsonl`` is an append-only aggregate).
    """
    stem = sanitize_path_component(label or "run", fallback="run")
    return Path(directory) / f"{stem}.{fingerprint}.games.jsonl"


def refuse_existing_log_message(path: Path, *, resume_flag: str, out_flag: str) -> str:
    """The message for "a log for these exact settings already exists"."""
    return (
        f"game log already exists: {path}\n"
        f"  Its settings fingerprint matches this invocation, so appending "
        f"would MIX two runs into one score.\n"
        f"  Pass {resume_flag} to continue that run (its finished games are "
        f"kept and only the remainder is played),\n"
        f"  or {out_flag} <path> / change the run label to start a separate one, "
        f"or delete the file."
    )


def refuse_settings_mismatch_message(
    path: Path, *, differences: Sequence[str], resume_flag: str,
) -> str:
    """The message for "``--resume`` onto a log written with other settings"."""
    return (
        f"{resume_flag}: {path} was written with DIFFERENT settings, so its "
        f"games are a different population:\n"
        + "\n".join(differences)
        + "\n  Resuming would average two rulers into one number. Re-run "
        "without the changes to continue that log, or start a new one."
    )
