"""docs/TCEC_SETUP.md must describe the engine that exists.

Fails when the page names a UCI option, command-line flag, exit code, UCI
command or `info string` the code does not have, when its UCI options table
differs from what the engine prints for `uci`, or when its flag table misses a
flag or states a default the argument parser does not have.
"""
from __future__ import annotations

import ast
import re
from itertools import pairwise
from pathlib import Path

import pytest

from chess_anti_engine.uci.__main__ import _build_parser
from chess_anti_engine.uci.protocol import _GO_KEYWORDS, _INT_KEYS, CmdGo, CmdUnknown, parse_command
from chess_anti_engine.uci.selfcheck import build_parser as build_selfcheck_parser
from chess_anti_engine.uci.subprocess_client import LineReader, send_line
from tests.test_uci_smoke import _make_tiny_checkpoint, _spawn_engine

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "TCEC_SETUP.md"
TEXT = DOC.read_text(encoding="utf-8")
SPANS = re.findall(r"`([^`\n]+)`", TEXT)


def _section(title_prefix: str) -> str:
    start = TEXT.index(f"## {title_prefix}")
    end = TEXT.find("\n## ", start + 1)
    return TEXT[start:] if end < 0 else TEXT[start:end]


def _table_rows(section: str) -> list[list[str]]:
    rows = []
    for line in section.splitlines():
        if not line.startswith("| ") or set(line) <= {"|", "-", " "}:
            continue
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows[1:]  # drop the header row


# --- UCI options: the table is the engine's own `uci` output ----------------


@pytest.fixture(scope="module")
def handshake(tmp_path_factory) -> dict[str, dict[str, str]]:
    """`option name ...` lines the real engine prints (default command line)."""
    proc = _spawn_engine(_make_tiny_checkpoint(tmp_path_factory.mktemp("ckpt")))
    reader = LineReader(proc)
    try:
        send_line(proc, "uci")
        lines = reader.read_until("uciok", timeout_s=60.0)
        send_line(proc, "quit")
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
    options: dict[str, dict[str, str]] = {}
    for line in lines:
        m = re.match(r"option name (\S+) type (\S+)(.*)$", line)
        if m is None:
            continue
        fields = dict(re.findall(r"(default|min|max) (\S+)", m.group(3)))
        fields["type"] = m.group(2)
        fields["vars"] = " / ".join(re.findall(r"var (\S+)", m.group(3)))
        options[m.group(1)] = fields
    assert options, lines
    return options


def _doc_options() -> dict[str, list[str]]:
    rows = _table_rows(_section("4. UCI options"))
    return {row[0].strip("`"): row for row in rows}


def test_the_options_table_lists_exactly_the_options_the_engine_prints(handshake) -> None:
    assert sorted(_doc_options()) == sorted(handshake)


def test_each_option_row_matches_the_printed_type_range_and_default(handshake) -> None:
    for name, row in _doc_options().items():
        printed = handshake[name]
        _, kind, default, effect, rebuilds = row
        assert effect, name
        assert rebuilds, name
        assert kind.split()[0] == printed["type"], (name, kind, printed)
        assert default == printed["default"].strip() or default == f"`{printed['default']}`", (
            name, default, printed)
        if printed["type"] == "spin":
            assert f"{printed['min']}..{printed['max']}" in kind, (name, kind, printed)
        if printed["type"] == "combo":
            assert printed["vars"].replace(" / ", "` / `") in kind, (name, kind, printed)


# --- command-line flags -----------------------------------------------------


def _engine_flags() -> dict[str, object]:
    out: dict[str, object] = {}
    for action in _build_parser()._actions:
        for opt in action.option_strings:
            if opt.startswith("--") and opt != "--help":
                out[opt] = action.default
    return out


def _script_flags(path: str) -> set[str]:
    src = (ROOT / path).read_text(encoding="utf-8")
    return set(re.findall(r"""["'](--[a-z][a-z0-9-]*)["']""", src)) | set(
        re.findall(r"^\s*(--[a-z][a-z0-9-]*)\)", src, flags=re.MULTILINE))


def test_every_flag_the_page_names_exists() -> None:
    known = set(_engine_flags())
    known |= {o for a in build_selfcheck_parser()._actions for o in a.option_strings}
    known |= _script_flags("scripts/uci_prewarm.py") | _script_flags("scripts/tcec_autotune.sh")
    named = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", TEXT))
    assert named, "no flags found"
    assert named <= known, sorted(named - known)


def _flag_rows() -> dict[str, str]:
    rows = _table_rows(_section("3. Command lines"))
    out: dict[str, str] = {}
    for row in rows:
        for flag in re.findall(r"`(--[a-z0-9-]+)`", row[0]):
            out[flag] = row[1]
    return out


def test_the_flag_table_covers_every_engine_flag() -> None:
    assert set(_engine_flags()) <= set(_flag_rows()), sorted(
        set(_engine_flags()) - set(_flag_rows()))


@pytest.mark.parametrize("flag", sorted(_engine_flags()))
def test_each_flag_default_matches_the_parser(flag: str) -> None:
    default = _engine_flags()[flag]
    cell = _flag_rows()[flag]
    if isinstance(default, bool) or default is None:
        return  # on/off and environment-backed defaults are worded, not values
    if flag == "--compile-cache-dir":
        assert "~/.cache/deepfin/worker_cache" in cell
        return
    if isinstance(default, str):
        assert cell == f"`{default}`", (flag, cell, default)
    else:
        assert cell == str(default), (flag, cell, default)


# --- exit codes -------------------------------------------------------------


def test_documented_exit_codes_exist_in_the_code() -> None:
    rows = _table_rows(_section("8. Positions, failures"))
    codes = {row[0].strip("`") for row in rows if re.fullmatch(r"`\d+`", row[0])}
    assert codes == {"0", "1", "2"}
    main_src = (ROOT / "chess_anti_engine/uci/__main__.py").read_text(encoding="utf-8")
    engine_src = (ROOT / "chess_anti_engine/uci/engine.py").read_text(encoding="utf-8")
    selfcheck_src = (ROOT / "chess_anti_engine/uci/selfcheck.py").read_text(encoding="utf-8")
    assert "rc = 1" in main_src  # startup failure / stuck search at quit
    assert "self._exit_code = 1" in engine_src  # failed boundary health check
    assert "return 2" in selfcheck_src  # extension refusal
    assert "p.error(" in main_src  # argparse: bad command line -> 2


# --- UCI commands -----------------------------------------------------------

_UCI_HEADS = {"uci", "isready", "ucinewgame", "position", "go", "stop", "ponderhit",
              "quit", "setoption", "searchconfig"}


def test_every_uci_command_the_page_names_parses() -> None:
    seen = 0
    for span in SPANS:
        words = span.split()
        if not words or words[0] not in _UCI_HEADS:
            continue
        if any(w in span for w in ("...", "<", "N ", " N")) and words[0] != "go":
            continue
        seen += 1
        cmd = parse_command(span)
        assert not isinstance(cmd, CmdUnknown), span
        if words[0] == "go" and len(words) > 1:
            assert isinstance(cmd, CmdGo), span
            # Upper-case words (N) are the page's placeholders for numbers, and
            # the word after a numeric keyword is its (possibly bad) value.
            keywords = {
                w for prev, w in pairwise(words)
                if w.isalpha() and not w.isupper() and prev not in _INT_KEYS
            }
            assert keywords <= set(_GO_KEYWORDS), span
    assert seen >= 5


def test_setoption_examples_name_real_options(handshake) -> None:
    for span in SPANS:
        m = re.match(r"setoption name (\S+)", span)
        if m:
            assert m.group(1) in handshake, span


# --- info strings: every quoted message exists in the engine's source -------


def _source_strings() -> list[str]:
    out: list[str] = []
    for path in sorted((ROOT / "chess_anti_engine/uci").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                out.append(node.value)
            elif isinstance(node, ast.JoinedStr):
                out.append("".join(
                    v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "\x00"
                    for v in node.values
                ))
    return out


def _message_spans() -> list[str]:
    return [s for s in SPANS if s.startswith(("info string ", "deepfin: "))]


def test_every_quoted_engine_message_exists_in_the_source() -> None:
    corpus = _source_strings()
    spans = _message_spans()
    assert len(spans) >= 15, spans
    for span in spans:
        for fragment in span.split("..."):
            fragment = fragment.strip()
            if len(fragment) < 6:
                continue
            bare = fragment.removeprefix("info string ").removeprefix("deepfin: ")
            assert any(fragment in s or (bare and bare in s) for s in corpus), (span, fragment)
