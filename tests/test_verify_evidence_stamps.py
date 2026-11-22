"""scripts/verify_evidence_stamps.py on a fabricated log and a temp bank.

Every verdict is asserted by EXIT CODE (the gate's contract), with the printed
line checked only to pin which citation produced it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_evidence_stamps.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_evidence_stamps", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # @dataclass resolves annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


VES = _load()

_PREFIX_TABLE = """| Prefix | Expands to |
|---|---|
| `B/` | `~/chess-artifacts/ops/bank one` |
| `~/chess-artifacts/` | `~/chess-artifacts` |
| `~/` | `~/` |
"""


def _sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def _mtime_stamp(path: Path) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(path.stat().st_mtime))


@pytest.fixture
def bank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    root = tmp_path / "artifacts"
    one = root / "ops" / "bank one"  # a path with a space
    (one / "arenas" / "m1").mkdir(parents=True)
    (one / "arenas" / "m1" / "pool.json").write_text('{"elo": 1}\n')
    (one / "arenas" / "m1" / "results.jsonl").write_text('{"games": 2}\n')
    for pin in ("aaa", "bbb"):
        (one / f"guard_{pin}_rep0.json").write_text("zero\n")
        (one / f"guard_{pin}_rep1.json").write_text("one\n")
    living = one / "living.log"
    living.write_text("line\n")
    os.utime(living, (1_700_000_000, 1_700_000_000))
    home = tmp_path / "home"
    (home / "engines").mkdir(parents=True)
    (home / "engines" / "notes.txt").write_text("notes\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CHESS_ARTIFACTS_ROOT", raising=False)
    return {"root": root, "one": one, "home": home, "tmp": tmp_path}


def _run(bank: dict[str, Path], body: str, allow: str | None = None) -> tuple[int, str, Path]:
    log = bank["tmp"] / "log.md"
    log.write_text("# Log\n\n" + _PREFIX_TABLE + "\n" + body)
    if allow is not None:
        (bank["tmp"] / "log.stamp-allowlist.txt").write_text(allow)
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = VES.main([str(log), "--prefix-source", str(log), "--artifacts-root", str(bank["root"])])
    return rc, buf.getvalue(), log


def _status(out: str, status: str) -> list[str]:
    return [ln for ln in out.splitlines() if ln.startswith(status + " ")]


def _ok_row(bank: dict[str, Path]) -> str:
    pool = bank["one"] / "arenas" / "m1" / "pool.json"
    res = bank["one"] / "arenas" / "m1" / "results.jsonl"
    # A row citing two files: a prefixed path, then a bare name resolved beside it.
    return f"| m1 | `B/arenas/m1/pool.json` [`{_sha8(pool)}`], `results.jsonl` [`{_sha8(res)}`] |\n"


def test_ok_sha_mtime_two_files_and_home(bank: dict[str, Path]) -> None:
    living = bank["one"] / "living.log"
    notes = bank["home"] / "engines" / "notes.txt"
    body = (_ok_row(bank)
            + f"| l | `B/living.log` [mtime {_mtime_stamp(living)}] |\n"
            + f"| h | `~/engines/notes.txt` [`{_sha8(notes)}`] |\n")
    rc, out, _ = _run(bank, body)
    assert rc == 0, out
    assert sum(ln.startswith("OK ") for ln in out.splitlines()) == 4, out
    assert "results.jsonl" in out
    assert not any(ln.startswith(("DRIFT", "MISSING")) for ln in out.splitlines()), out


def test_sha_drift_fails(bank: dict[str, Path]) -> None:
    rc, out, _ = _run(bank, "| m1 | `B/arenas/m1/pool.json` [`deadbeef`] |\n")
    assert rc == 1, out
    drift = [ln for ln in out.splitlines() if ln.startswith("DRIFT")]
    assert len(drift) == 1, out
    assert "log.md:9 B/arenas/m1/pool.json" in drift[0], out


def test_drift_in_second_file_of_a_row_fails(bank: dict[str, Path]) -> None:
    pool = bank["one"] / "arenas" / "m1" / "pool.json"
    body = f"| m1 | `B/arenas/m1/pool.json` [`{_sha8(pool)}`], `results.jsonl` [`0badc0de`] |\n"
    rc, out, _ = _run(bank, body)
    assert rc == 1, out
    drift = [ln for ln in out.splitlines() if ln.startswith("DRIFT")]
    assert len(drift) == 1, out
    assert "results.jsonl" in drift[0], out


def test_mtime_drift_fails(bank: dict[str, Path]) -> None:
    rc, out, _ = _run(bank, "| l | `B/living.log` [mtime 01-01 00:00] |\n")
    assert rc == 1, out
    assert len(_status(out, "DRIFT")) == 1, out


def test_new_missing_fails_and_allow_listed_passes(bank: dict[str, Path]) -> None:
    body = _ok_row(bank) + "| x | `B/arenas/gone.json` [`12345678`] |\n"
    rc, out, _ = _run(bank, body)
    assert rc == 1, out
    assert "gone.json" in "".join(_status(out, "MISSING")), out
    rc, out, _ = _run(bank, body, allow="B/arenas/gone.json | deleted, superseded by m1\n")
    assert rc == 0, out
    assert "allow-listed" in out
    # Allow-listing one token does not excuse a second, new MISSING citation.
    rc, out, _ = _run(bank, body + "| y | `B/other.json` |\n", allow="B/arenas/gone.json | deleted\n")
    assert rc == 1, out
    assert [ln for ln in _status(out, "MISSING") if "other.json" in ln], out


def test_allow_list_line_without_reason_is_a_usage_error(bank: dict[str, Path]) -> None:
    rc, _, _ = _run(bank, _ok_row(bank), allow="B/arenas/gone.json\n")
    assert rc == 2


def test_glob_and_brace_per_option_stamps(bank: dict[str, Path]) -> None:
    s0 = _sha8(bank["one"] / "guard_aaa_rep0.json")
    s1 = _sha8(bank["one"] / "guard_aaa_rep1.json")
    rc, out, _ = _run(bank, f"| g | `B/guard_*_rep{{0,1}}.json` [`{s0}`] / [`{s1}`] |\n")
    assert rc == 0, out
    # One file of a byte-equality glob changes: the stamp no longer holds for every match.
    (bank["one"] / "guard_bbb_rep1.json").write_text("changed\n")
    rc, out, _ = _run(bank, f"| g | `B/guard_*_rep{{0,1}}.json` [`{s0}`] / [`{s1}`] |\n")
    assert rc == 1, out
    assert _status(out, "DRIFT"), out
    (bank["one"] / "guard_bbb_rep1.json").write_text("one\n")
    # One stamp must hold for EVERY match: rep0 and rep1 differ, so this is DRIFT.
    rc, out, _ = _run(bank, f"| g | `B/guard_*_rep{{0,1}}.json` [`{s0}`] |\n")
    assert rc == 1, out
    assert _status(out, "DRIFT"), out
    rc, out, _ = _run(bank, f"| g | `B/guard_*_rep0.json` [`{s0}`] |\n")
    assert rc == 0, out
    # A stamp count that fits neither the options (2) nor the matches (4) is DRIFT, not a silent OK.
    rc, out, _ = _run(bank, f"| g | `B/guard_*_rep{{0,1}}.json` [`{s0}`] / [`{s1}`] / [`{s1}`] |\n")
    assert rc == 1, out
    assert _status(out, "DRIFT"), out


def test_prefix_map_is_read_from_the_table(bank: dict[str, Path]) -> None:
    other = bank["root"] / "elsewhere" / "arenas" / "m1"
    other.mkdir(parents=True)
    (other / "pool.json").write_text("different\n")
    log = bank["tmp"] / "log.md"
    table = _PREFIX_TABLE.replace("ops/bank one", "elsewhere")
    pool = bank["one"] / "arenas" / "m1" / "pool.json"
    log.write_text("# Log\n\n" + table + f"\n| m1 | `B/arenas/m1/pool.json` [`{_sha8(pool)}`] |\n")
    rc = VES.main([str(log), "--prefix-source", str(log), "--artifacts-root", str(bank["root"])])
    assert rc == 1  # resolved under `elsewhere/`, whose pool.json has another sha


def test_unstamped_bare_names_are_not_citations(bank: dict[str, Path]) -> None:
    rc, out, _ = _run(bank, _ok_row(bank) + "Edit `chess_anti_engine/foo.py` and `children.npz`.\n")
    assert rc == 0, out


def test_real_log_prefix_table_parses() -> None:
    prefixes = VES.read_prefix_map(VES.DEFAULT_LOG, Path("/artifacts"))
    assert {"RC/", "R/", "F/", "WK/", "~/"} <= set(prefixes)
    assert prefixes["RC/"] == "/artifacts/operations/rl-claude-20261119"
