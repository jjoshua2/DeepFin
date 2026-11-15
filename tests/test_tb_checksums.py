"""Syzygy checksum verification (``chess_anti_engine/tb/checksums.py``) and
the self-check component that runs it (``syzygy-checksums``).

Table-free tests (tiny fake ``.rtbw`` files plus a checksum list) run
everywhere, CI included. The smoke-set tests need ``data/syzygy_3-4man`` and
SKIP with the reason printed where it is absent.
"""
from __future__ import annotations

import hashlib
import io
import os
import random
import sys
from pathlib import Path
from unittest.mock import Mock

import chess.syzygy
import pytest

from chess_anti_engine.tb import checksums
from chess_anti_engine.uci import selfcheck as sc
from chess_anti_engine.utils.syzygy import missing_tablebase_dirs, syzygy_roots


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _tables(d: Path, files: dict[str, bytes]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (d / name).write_bytes(data)
    return d


_FILES = {"KQvK.rtbw": b"wdl-kqk", "KQvK.rtbz": b"dtz-kqk", "KRvK.rtbw": b"wdl-krk"}


def _list_text(files: dict[str, bytes], *, sep: str = "  ", eol: str = "\n",
               prefix: str = "") -> str:
    return "".join(f"{_md5(data)}{sep}{prefix}{name}{eol}" for name, data in files.items())


def _check(path: str | None, lists=(), **kw) -> sc.CheckResult:
    kw.setdefault("progress_stream", io.StringIO())
    return sc._run_check("syzygy-checksums", lambda: sc.check_syzygy_checksums(path, lists, **kw),
                         limit=sc.SYZYGY_CHECKSUM_DETAIL_LIMIT)


# ── parsing ───────────────────────────────────────────────────────────────


def test_gnu_text_and_binary_mode_crlf_and_subpaths_parse() -> None:
    text = (
        f"{_md5(b'a')}  KQvK.rtbw\r\n"            # text mode, CRLF
        f"{_md5(b'b')} *KQvK.rtbz\r\n"            # binary-mode marker
        f"{_md5(b'c')}  6-wdl/KRvK.rtbw\n"       # relative subpath
        f"{_md5(b'd')}  ./3-4-5/KBvK.rtbw\n"
        "# a comment\n\n"
        f"{_md5(b'e')}  README.txt\n"              # not a table: ignored
        f"MD5 (KNvK.rtbw) = {_md5(b'f')}\n"        # BSD style
        f"{hashlib.sha256(b'g').hexdigest()}  KPvK.rtbw\n"
    )
    merged = checksums.parse_lists([("list", text)])
    assert merged.problems == []
    got = {k: (v.algo, v.digest) for k, v in merged.entries.items()}
    assert got == {
        "KQvK.rtbw": ("md5", _md5(b"a")), "KQvK.rtbz": ("md5", _md5(b"b")),
        "KRvK.rtbw": ("md5", _md5(b"c")), "KBvK.rtbw": ("md5", _md5(b"d")),
        "KNvK.rtbw": ("md5", _md5(b"f")), "KPvK.rtbw": ("sha256", hashlib.sha256(b"g").hexdigest()),
    }


@pytest.mark.parametrize("line", [
    "not a checksum line",
    "abc123  KQvK.rtbw",                       # digest of no known length
    f"{_md5(b'a')} KQvK.rtbw",                 # one space, no mode char
    f"{'z' * 32}  KQvK.rtbw",                  # not hex
    f"SHA256 (KQvK.rtbw) = {_md5(b'a')}",      # BSD tag disagrees with length
])
def test_a_malformed_line_is_a_named_problem(line: str) -> None:
    merged = checksums.parse_lists([("my.md5", f"{_md5(b'x')}  KRvK.rtbw\n{line}\n")])
    assert merged.problems
    assert merged.problems[0].startswith("my.md5:2: malformed")
    assert "KRvK.rtbw" in merged.entries


def test_conflicting_entries_are_a_problem() -> None:
    merged = checksums.parse_lists([
        ("a.md5", f"{_md5(b'1')}  KQvK.rtbw\n"), ("b.md5", f"{_md5(b'2')}  sub/KQvK.rtbw\n"),
    ])
    assert len(merged.problems) == 1
    assert "KQvK.rtbw: conflicting md5" in merged.problems[0]


# ── verification outcomes (fake tables) ───────────────────────────────────


def test_all_files_match_passes(tmp_path: Path) -> None:
    d = _tables(tmp_path / "tb", _FILES)
    lst = tmp_path / "sums.md5"
    lst.write_text(_list_text(_FILES))
    res = _check(str(d), [str(lst)])
    assert res.status == "PASS", res.detail
    assert res.detail.startswith("3 of 3 table files match the given list(s)")


def test_a_mismatch_fails_naming_the_file(tmp_path: Path) -> None:
    d = _tables(tmp_path / "tb", _FILES)
    lst = tmp_path / "sums.md5"
    lst.write_text(_list_text(_FILES))
    (d / "KQvK.rtbz").write_bytes(b"dtz-kqk-CORRUPT")
    res = _check(str(d), [str(lst)])
    assert res.status == "FAIL"
    assert "1 CHECKSUM MISMATCH(ES), replace these files: " in res.detail
    assert f"{d / 'KQvK.rtbz'} (md5 {_md5(b'dtz-kqk-CORRUPT')}" in res.detail
    assert "KRvK" not in res.detail.split("; ")[0]


def test_a_listed_file_absent_is_named_but_not_a_failure(tmp_path: Path) -> None:
    d = _tables(tmp_path / "tb", {k: v for k, v in _FILES.items() if k != "KRvK.rtbw"})
    lst = tmp_path / "sums.md5"
    lst.write_text(_list_text(_FILES))
    res = _check(str(d), [str(lst)])
    assert res.status == "PASS", res.detail
    assert "1 listed table(s) not in the path: KRvK.rtbw" in res.detail


def test_an_extra_file_is_not_verified_and_require_makes_it_fail(tmp_path: Path) -> None:
    d = _tables(tmp_path / "tb", {**_FILES, "KBvK.rtbw": b"extra"})
    lst = tmp_path / "sums.md5"
    lst.write_text(_list_text(_FILES))
    res = _check(str(d), [str(lst)])
    assert res.status == "SKIP"
    assert f"NOT VERIFIED: 1 table file(s) in no checksum list: {d / 'KBvK.rtbw'}" in res.detail
    res = _check(str(d), [str(lst)], require=True)
    assert res.status == "FAIL"
    assert "NOT VERIFIED" in res.detail
    assert "--syzygy-require-checksums" in res.detail


def test_no_list_at_all_is_not_verified_never_a_pass(tmp_path: Path, monkeypatch) -> None:
    d = _tables(tmp_path / "tb", _FILES)
    monkeypatch.setattr(checksums, "builtin_text", lambda: None)
    res = _check(str(d))
    assert res.status == "SKIP", res.detail
    assert res.detail.startswith("NOT VERIFIED: no checksum list (3 table files unchecked)")
    assert _check(str(d), require=True).status == "FAIL"
    res = _check(str(d), ["builtin"])  # asked for by name: its absence is a problem
    assert res.status == "FAIL"
    assert "built-in standard 3-4-5-6-man manifest is missing" in res.detail


def test_a_malformed_list_fails_even_when_every_file_matches(tmp_path: Path) -> None:
    d = _tables(tmp_path / "tb", _FILES)
    lst = tmp_path / "sums.md5"
    lst.write_text(_list_text(_FILES) + "garbage line\n")
    res = _check(str(d), [str(lst)])
    assert res.status == "FAIL"
    assert f"{lst}:4: malformed checksum line 'garbage line'" in res.detail


def test_crlf_binary_marker_and_subpath_lists_verify(tmp_path: Path) -> None:
    d = _tables(tmp_path / "tb", _FILES)
    lst = tmp_path / "sums.md5"
    lst.write_bytes(_list_text(_FILES, sep=" *", eol="\r\n", prefix="3-4-5/").encode())
    res = _check(str(d), [str(lst)])
    assert res.status == "PASS", res.detail


def test_an_unreadable_explicit_list_fails(tmp_path: Path) -> None:
    d = _tables(tmp_path / "tb", _FILES)
    res = _check(str(d), [str(tmp_path / "nope.md5")])
    assert res.status == "FAIL"
    assert "nope.md5: cannot be read" in res.detail


def test_a_broken_symlink_table_is_unreadable_and_fails(tmp_path: Path) -> None:
    d = _tables(tmp_path / "tb", _FILES)
    os.symlink(tmp_path / "gone.rtbw", d / "KBvK.rtbw")
    lst = tmp_path / "sums.md5"
    lst.write_text(_list_text({**_FILES, "KBvK.rtbw": b"x"}))
    res = _check(str(d), [str(lst)])
    assert res.status == "FAIL"
    assert f"1 unreadable table file(s): {d / 'KBvK.rtbw'}" in res.detail


# ── where the lists come from ─────────────────────────────────────────────


def test_lists_in_the_path_are_discovered_across_directories(tmp_path: Path) -> None:
    """The production layout: one list in the second directory covers both."""
    first = _tables(tmp_path / "a", {"KQvK.rtbw": b"1", "KQvK.rtbz": b"2"})
    second = _tables(tmp_path / "b", {"KRvK.rtbw": b"3"})
    (second / "3-4-5-6.md5").write_text(
        _list_text({"KQvK.rtbw": b"1", "KQvK.rtbz": b"2", "KRvK.rtbw": b"3"}))
    res = _check(f"{first}{os.pathsep}{second}")
    assert res.status == "PASS", res.detail
    assert "match list(s) found in the path" in res.detail
    for name in ("checksum.md5", "checksums.txt", "md5", "sha256", "x.sha256"):
        assert checksums.discover_lists([str(tmp_path / "none")]) == []
        (first / name).write_text("")
        assert first / name in checksums.discover_lists([str(first)])


def test_an_explicit_list_replaces_discovery(tmp_path: Path) -> None:
    d = _tables(tmp_path / "tb", _FILES)
    (d / "bad.md5").write_text(_list_text({"KQvK.rtbw": b"WRONG"}))
    assert _check(str(d)).status == "FAIL"  # the discovered list disagrees
    good = tmp_path / "good.md5"
    good.write_text(_list_text(_FILES))
    res = _check(str(d), [str(good)])
    assert res.status == "PASS", res.detail


def test_without_any_list_the_builtin_manifest_is_used(tmp_path: Path, monkeypatch) -> None:
    d = _tables(tmp_path / "tb", _FILES)
    monkeypatch.setattr(checksums, "builtin_text", lambda: _list_text(_FILES))
    res = _check(str(d))
    assert res.status == "PASS", res.detail
    assert "built-in standard 3-4-5-6-man manifest (no checksum list found in the path)" in res.detail


def test_verification_off_is_reported_not_verified(tmp_path: Path) -> None:
    d = _tables(tmp_path / "tb", _FILES)
    res = _check(str(d), enabled=False)
    assert res.status == "SKIP"
    assert res.detail.startswith("NOT VERIFIED: checksum verification turned off")
    assert _check(str(d), enabled=False, require=True).status == "FAIL"
    assert _check(None).status == "SKIP"


def test_hashing_runs_on_bounded_threads_and_reports_progress(tmp_path: Path) -> None:
    files = {f"K{p}vK{q}.rtbw": os.urandom(3 << 20) for p in "QRBN" for q in "QRBN"}
    d = _tables(tmp_path / "tb", files)
    merged = checksums.parse_lists([("l", _list_text(files))])
    seen: list[tuple[int, int, int, int]] = []
    report = checksums.verify([str(d)], merged, "l", threads=99,
                              progress=lambda *a: seen.append(a), progress_interval_s=0.0)
    assert report.threads == 32
    assert len(report.ok) == len(files)
    assert seen[-1] == (len(files), len(files), sum(map(len, files.values())), report.total_bytes)


def test_selfcheck_flags_reach_the_check(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake(path, lists=(), **kw):
        seen.update(path=path, lists=tuple(lists), **kw)
        return "x" * 1000  # the line names every file: never cut at the generic 400

    monkeypatch.setattr(sc, "check_syzygy_checksums", fake)
    opts, _ = sc.parse_options([
        "--syzygy", "/tb/a", "--syzygy-checksums", "x.md5", "--syzygy-checksums", "builtin",
        "--syzygy-require-checksums", "--syzygy-checksum-threads", "2", "--no-uci",
    ])
    assert opts.engine_args == ()
    monkeypatch.setattr(sc, "check_syzygy", lambda path: "ok")
    results = sc.run_selfcheck(opts)
    names = [r.name for r in results]
    assert names.index("syzygy-checksums") == names.index("syzygy") - 1
    assert results[names.index("syzygy-checksums")].detail == "x" * 1000
    assert seen == {"path": "/tb/a", "lists": ("x.md5", "builtin"), "enabled": True,
                    "require": True, "threads": 2}
    opts, _ = sc.parse_options(["--syzygy", "/tb/a", "--no-syzygy-checksums"])
    assert opts.verify_syzygy_checksums is False


def _stub_non_table_selfchecks(monkeypatch: pytest.MonkeyPatch) -> list[Mock]:
    checks = []
    for name in ("check_python", "check_torch", "check_extensions", "check_freshness",
                 "check_encoder", "check_movegen", "check_search", "check_checkpoint"):
        check = Mock(return_value="unrelated component stub")
        monkeypatch.setattr(sc, name, check)
        checks.append(check)
    monkeypatch.setattr(sc, "resolve_devices", lambda requested: ["cpu"])
    return checks


_SELFCHECK_GATE_ENGINE = """
import sys
with open(sys.argv[1], 'x') as log:
    for line in sys.stdin:
        log.write(line)
        log.flush()
        cmd = line.strip()
        if cmd == 'uci':
            print('uciok', flush=True)
        elif cmd == 'isready':
            print('readyok', flush=True)
        elif cmd.startswith('go '):
            print('bestmove e2e4', flush=True)
        elif cmd == 'quit':
            break
"""


def _fake_uci_for_checksum_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Exercise the real check_uci consumer using a table-free protocol child."""
    log = tmp_path / "fake-uci-input.txt"
    real_check_uci = sc.check_uci

    def command(checkpoint: str, opts: sc.SelfcheckOptions, cache: str) -> list[str]:
        assert checkpoint == opts.checkpoint == "unused"
        assert Path(cache).is_absolute()
        return [sys.executable, "-c", _SELFCHECK_GATE_ENGINE, str(log)]

    monkeypatch.setattr(sc, "check_uci", lambda opts: real_check_uci(opts, command=command))
    return log


@pytest.mark.parametrize("failure", (
    "mismatch", "malformed-list", "unverified-required", "disabled-required",
))
def test_selfcheck_checksum_gate_fail_blocks_both_table_loading_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    checks = _stub_non_table_selfchecks(monkeypatch)
    d = _tables(tmp_path / "tb", _FILES)
    lst = tmp_path / "sums.md5"
    lst.write_text("" if failure == "unverified-required" else _list_text(_FILES))
    if failure == "mismatch":
        (d / "KQvK.rtbz").write_bytes(b"corrupted fake table")
    elif failure == "malformed-list":
        with lst.open("a") as f:
            f.write("malformed checksum line\n")
    probe = Mock(side_effect=AssertionError("rejected tables reached in-process probe"))
    popen = Mock(side_effect=AssertionError("rejected tables reached UCI child"))
    monkeypatch.setattr(sc, "check_syzygy", probe)
    # Keep check_uci real: without the gate it would reach this Popen sentinel.
    monkeypatch.setattr(sc.subprocess, "Popen", popen)
    opts = sc.SelfcheckOptions(
        checkpoint="unused", devices=("cpu",), syzygy_path=str(d),
        syzygy_checksums=(str(lst),), syzygy_checksum_threads=1,
        verify_syzygy_checksums=failure != "disabled-required",
        require_syzygy_checksums=failure.endswith("required"),
    )
    results = sc.run_selfcheck(opts)
    by_name = {r.name: r for r in results}
    assert sc.overall(results) == "FAIL"
    assert by_name["syzygy-checksums"].status == "FAIL"
    expected_cause = {
        "mismatch": "CHECKSUM MISMATCH", "malformed-list": "malformed checksum line",
        "unverified-required": "NOT VERIFIED", "disabled-required": "turned off",
    }[failure]
    assert expected_cause in by_name["syzygy-checksums"].detail
    for name in ("syzygy", "uci"):
        assert by_name[name].status == "SKIP"
        assert "blocked by syzygy-checksums FAIL" in by_name[name].detail
    probe.assert_not_called()
    popen.assert_not_called()
    for check in checks:
        check.assert_called_once()


@pytest.mark.parametrize("verification", ("pass", "disabled", "unverified"))
def test_selfcheck_checksum_gate_pass_and_skip_keep_existing_loading_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verification: str,
) -> None:
    _stub_non_table_selfchecks(monkeypatch)
    d = _tables(tmp_path / "tb", _FILES)
    lst = tmp_path / "sums.md5"
    lst.write_text("" if verification == "unverified" else _list_text(_FILES))
    probe = Mock(return_value="stand-in probe; no real table loading")
    monkeypatch.setattr(sc, "check_syzygy", probe)
    log = _fake_uci_for_checksum_gate(monkeypatch, tmp_path)
    opts = sc.SelfcheckOptions(
        checkpoint="unused", devices=("cpu",), syzygy_path=str(d), fast=True,
        syzygy_checksums=(str(lst),), syzygy_checksum_threads=1,
        verify_syzygy_checksums=verification != "disabled",
    )
    results = sc.run_selfcheck(opts)
    by_name = {r.name: r for r in results}
    assert sc.overall(results) == "PASS"
    assert by_name["syzygy-checksums"].status == ("PASS" if verification == "pass" else "SKIP")
    assert by_name["syzygy"].status == "PASS"
    assert by_name["uci"].status == "PASS"
    probe.assert_called_once_with(str(d))
    wire = log.read_text().splitlines()
    assert f"setoption name SyzygyPath value {d}" in wire
    assert wire[-1] == "quit"


def test_selfcheck_checksum_gate_no_table_path_keeps_uci_without_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_non_table_selfchecks(monkeypatch)
    log = _fake_uci_for_checksum_gate(monkeypatch, tmp_path)
    results = sc.run_selfcheck(sc.SelfcheckOptions(checkpoint="unused", devices=("cpu",), fast=True))
    by_name = {r.name: r for r in results}
    assert sc.overall(results) == "PASS"
    assert by_name["syzygy-checksums"].status == "SKIP"
    assert by_name["syzygy"].status == "SKIP"
    assert by_name["uci"].status == "PASS"
    assert not any(line.startswith("setoption name SyzygyPath") for line in log.read_text().splitlines())


@pytest.mark.parametrize("corrupt", (False, True))
def test_selfcheck_checksum_gate_no_uci_retains_explicit_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool,
) -> None:
    _stub_non_table_selfchecks(monkeypatch)
    d = _tables(tmp_path / "tb", _FILES)
    lst = tmp_path / "sums.md5"
    lst.write_text(_list_text(_FILES))
    if corrupt:
        (d / "KQvK.rtbz").write_bytes(b"corrupted fake table")
    probe = Mock(return_value="stand-in probe")
    uci = Mock(side_effect=AssertionError("--no-uci started a child"))
    monkeypatch.setattr(sc, "check_syzygy", probe)
    monkeypatch.setattr(sc, "check_uci", uci)
    results = sc.run_selfcheck(sc.SelfcheckOptions(
        devices=("cpu",), syzygy_path=str(d), run_uci=False,
        syzygy_checksums=(str(lst),), syzygy_checksum_threads=1,
    ))
    by_name = {r.name: r for r in results}
    assert sc.overall(results) == ("FAIL" if corrupt else "PASS")
    assert by_name["uci"] == sc.CheckResult("uci", "SKIP", "--no-uci given")
    assert probe.call_count == (0 if corrupt else 1)
    uci.assert_not_called()


def test_the_full_list_is_not_cut_at_the_generic_400_chars(tmp_path: Path) -> None:
    files = {f"K{a}{b}vK.rtbw": a.encode() + b.encode() for a in "QRBNP" for b in "QRBNP"}
    d = _tables(tmp_path / "tb", files)
    lst = tmp_path / "s.md5"
    lst.write_text("")
    res = _check(str(d), [str(lst)])
    assert res.status == "SKIP"
    for name in files:
        assert str(d / name) in res.detail


# ── the built-in manifest ─────────────────────────────────────────────────


def test_the_builtin_manifest_is_the_standard_3_to_6_man_set() -> None:
    text = checksums.builtin_text()
    assert text is not None
    merged = checksums.parse_lists([("builtin", text)])
    assert merged.problems == []
    names = set(merged.entries)
    assert len(names) == 1020
    stems = {n.rsplit(".", 1)[0] for n in names}
    assert len(stems) == 510
    assert names == {f"{s}{ext}" for s in stems for ext in (".rtbw", ".rtbz")}
    assert all(chess.syzygy.is_tablename(s) for s in stems)
    by_men: dict[int, int] = {}
    for s in stems:
        by_men[len(s) - 1] = by_men.get(len(s) - 1, 0) + 1
    assert by_men == {3: 5, 4: 30, 5: 110, 6: 365}
    assert all(e.algo == "md5" for e in merged.entries.values())


# ── smoke tables: data-level corruption the header check cannot see ───────


def _smoke() -> Path | None:
    for root in syzygy_roots():
        smoke = root / "data" / "syzygy_3-4man"
        if not missing_tablebase_dirs(str(smoke)):
            return smoke
    return None


_SMOKE = _smoke()
needs_smoke = pytest.mark.skipif(
    _SMOKE is None,
    reason="3-4 man smoke Syzygy set (data/syzygy_3-4man) not present (CI ships none): the "
           "real-table checksum tests are NOT run",
)


@needs_smoke
def test_the_builtin_manifest_matches_the_smoke_set() -> None:
    assert _SMOKE is not None
    res = _check(str(_SMOKE), ["builtin"])
    assert res.status == "PASS", res.detail
    n = len(list(_SMOKE.glob("*.rtb[wz]")))
    assert res.detail.startswith(f"{n} of {n} table files match")


@needs_smoke
def test_a_byte_flipped_inside_table_data_fails_the_checksum(tmp_path: Path) -> None:
    assert _SMOKE is not None
    d = tmp_path / "tb"
    d.mkdir()
    for stem in ("KQvK", "KRvK", "KQvKR"):
        for ext in (".rtbw", ".rtbz"):
            (d / f"{stem}{ext}").write_bytes((_SMOKE / f"{stem}{ext}").read_bytes())
    f = d / "KQvKR.rtbw"
    data = bytearray(f.read_bytes())
    pos = random.Random(20261114).randrange(len(data) // 2, len(data) - 64)
    data[pos] ^= 0x5A
    f.write_bytes(bytes(data))
    res = _check(str(d))  # no list in the path: the built-in manifest
    assert res.status == "FAIL", res.detail
    assert f"1 CHECKSUM MISMATCH(ES), replace these files: {f} " in res.detail
    print(f"flipped byte {pos} of {len(data)}", file=sys.stderr)


def test_the_selfcheck_thread_default_is_the_modules() -> None:
    opts, _ = sc.parse_options([])
    assert opts.syzygy_checksum_threads == checksums.DEFAULT_THREADS == sc.SelfcheckOptions().syzygy_checksum_threads
