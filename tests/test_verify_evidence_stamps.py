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
from datetime import datetime
from pathlib import Path
from types import ModuleType
from zoneinfo import ZoneInfo

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


_ZONE = "America/New_York"
_HEADER = f"# Log. Stamp time zone: `{_ZONE}`\n\n"


def _mtime_stamp(path: Path, zone: str = _ZONE) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo(zone)).strftime("%m-%d %H:%M")


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


def _run(bank: dict[str, Path], body: str, allow: str | None = None,
         root_flag: bool = True) -> tuple[int, str, Path]:
    log = bank["tmp"] / "log.md"
    log.write_text(_HEADER + _PREFIX_TABLE + "\n" + body)
    if allow is not None:
        (bank["tmp"] / "log.stamp-allowlist.txt").write_text(allow)
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        argv = [str(log), "--prefix-source", str(log)]
        if root_flag:
            argv += ["--artifacts-root", str(bank["root"])]
        rc = VES.main(argv)
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
    log.write_text(_HEADER + table + f"\n| m1 | `B/arenas/m1/pool.json` [`{_sha8(pool)}`] |\n")
    rc = VES.main([str(log), "--prefix-source", str(log), "--artifacts-root", str(bank["root"])])
    assert rc == 1  # resolved under `elsewhere/`, whose pool.json has another sha


def test_unstamped_bare_names_are_not_citations(bank: dict[str, Path]) -> None:
    rc, out, _ = _run(bank, _ok_row(bank) + "Edit `chess_anti_engine/foo.py` and `children.npz`.\n")
    assert rc == 0, out


def test_real_log_prefix_table_parses() -> None:
    prefixes = VES.read_prefix_map(VES.DEFAULT_LOG, Path("/artifacts"))
    assert {"RC/", "R/", "F/", "WK/", "~/"} <= set(prefixes)
    assert prefixes["RC/"] == "/artifacts/operations/rl-claude-20261119"
    assert VES.read_declared_tz(VES.DEFAULT_LOG) == "America/New_York"


@pytest.mark.parametrize(
    "stamp",
    [
        "[`{h4}`]",  # too short
        "[`{H}`]",  # upper-case hex
        "[{h}]",  # no backticks
        "[mtime 11-21 9:30]",  # malformed mtime
        ": [`{h}`]",  # separated from the token by a colon
    ],
    ids=["short", "uppercase", "no-backticks", "bad-mtime", "after-colon"],
)
def test_malformed_stamps_fail(bank: dict[str, Path], stamp: str) -> None:
    pool = bank["one"] / "arenas" / "m1" / "pool.json"
    h = _sha8(pool)
    text = stamp.format(h=h, h4=h[:4], H=h.upper())
    for token in ("B/arenas/m1/pool.json", "results.jsonl"):  # prefixed and bare
        rc, out, _ = _run(bank, _ok_row(bank) + f"| x | `{token}`{'' if text.startswith(':') else ' '}{text} |\n")
        assert rc == 1, out
        assert _status(out, "MALFORMED"), out


def test_minimum_sha_length_is_eight(bank: dict[str, Path]) -> None:
    pool = bank["one"] / "arenas" / "m1" / "pool.json"
    h = hashlib.sha256(pool.read_bytes()).hexdigest()
    assert VES.MIN_SHA == 8
    rc, out, _ = _run(bank, f"| x | `B/arenas/m1/pool.json` [`{h[:8]}`] |\n")
    assert rc == 0, out
    rc, out, _ = _run(bank, f"| x | `B/arenas/m1/pool.json` [`{h[:7]}`] |\n")
    assert rc == 1, out
    assert _status(out, "MALFORMED"), out
    rc, out, _ = _run(bank, f"| x | `B/arenas/m1/pool.json` [`{h}`] |\n")  # the full digest is fine
    assert rc == 0, out


@pytest.mark.parametrize("token", ["b/arenas/m1/pool.json", "B /arenas/m1/pool.json"])
def test_near_miss_prefix_is_malformed(bank: dict[str, Path], token: str) -> None:
    rc, out, _ = _run(bank, f"| x | `{token}` |\n")
    assert rc == 1, out
    assert _status(out, "MALFORMED"), out


def test_mtime_uses_the_declared_zone_not_the_machine_tz(
        bank: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    living = bank["one"] / "living.log"
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    try:
        rc, out, _ = _run(bank, f"| l | `B/living.log` [mtime {_mtime_stamp(living)}] |\n")
        assert rc == 0, out
        rc, out, _ = _run(bank, f"| l | `B/living.log` [mtime {_mtime_stamp(living, 'UTC')}] |\n")
        assert rc == 1, out
    finally:
        monkeypatch.undo()
        time.tzset()


def test_chess_artifacts_root_env(bank: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHESS_ARTIFACTS_ROOT", str(bank["root"]))
    rc, out, _ = _run(bank, _ok_row(bank), root_flag=False)
    assert rc == 0, out
    monkeypatch.setenv("CHESS_ARTIFACTS_ROOT", str(bank["tmp"] / "nowhere"))
    rc, out, _ = _run(bank, _ok_row(bank), root_flag=False)
    assert rc == 1, out
    assert _status(out, "MISSING"), out


def test_prefix_mentions_in_prose_are_not_citations(bank: dict[str, Path]) -> None:
    body = "`~/chess-artifacts` is the root; `.../` is the previous directory; `B/` is a prefix.\n" + _ok_row(bank)
    rc, out, _ = _run(bank, body)
    assert rc == 0, out
    assert sum(ln.startswith("OK ") for ln in out.splitlines()) == 2, out


def test_stamp_wrapped_onto_the_next_line_still_counts(bank: dict[str, Path]) -> None:
    rc, out, _ = _run(bank, "See `B/arenas/m1/pool.json`\n  [`deadbeef`].\n")
    assert rc == 1, out
    assert _status(out, "DRIFT"), out


@pytest.mark.parametrize(
    "prose",
    [
        "fixed in [0c795fb7a] by the reviewer.",  # commit in brackets
        "see [a0bc34416](https://example.invalid/commit/a0bc34416).",  # markdown commit link
        "merged as [`7fea1b889`].",  # backticked commit after a prose word
        "the [mtime column] is local.",  # prose mtime
        "the run [deadbeef] and [cafe1234 rows].",  # hex words
        "see `docs/README.md` [a0bc34416](https://example.invalid/commit/a0bc34416).",  # link after a path
        "the `F/arenas/` [mtime column] is listed.",  # prose mtime after a path
        "the `v2` [cafe1234 rows] table.",  # hex word after a token that is not path-like
        "see `docs/README.md`, fixed later in [0c795fb7a].",  # bracket later on the line, not after the token
        "`scripts/arena_x.py` [`a0bc34416`] carries the fix.",  # repo code at a commit
        "see `chess_anti_engine/x.py`\n[`a0bc34416`] for the code.",  # same, wrapped
    ],
)
def test_prose_brackets_are_silent(bank: dict[str, Path], prose: str) -> None:
    rc, out, _ = _run(bank, prose + "\n")
    assert rc == 0, out
    assert not _status(out, "MALFORMED"), out


def test_letters_only_hex_after_a_token_is_malformed(bank: dict[str, Path]) -> None:
    rc, out, _ = _run(bank, _ok_row(bank) + "| x | `B/arenas/m1/pool.json` [deadbeef] |\n")
    assert rc == 1, out
    assert _status(out, "MALFORMED"), out
