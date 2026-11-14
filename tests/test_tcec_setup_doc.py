"""docs/TCEC_SETUP.md must describe the engine that exists.

Fails when the page's UCI options table differs from what the engine prints for
`uci`, or a string option's range from the code; when the page names a
command-line flag the code does not have or states a default the parser does
not have; when its exit codes differ from what the entry points return; when
its option-class table differs from how the running engine treats a bad value;
when a quoted UCI command does not parse; when a quoted engine message exists
neither in the code nor in the engine's real output; and when its watchdog,
`stop` or `Threads` statements differ from the code.
"""
from __future__ import annotations

import ast
import io
import math
import os
import re
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest

from chess_anti_engine.mcts.search_options import SEARCH_OPTIONS
from chess_anti_engine.uci import cli
from chess_anti_engine.uci import engine as engine_module
from chess_anti_engine.uci.__main__ import _build_parser
from chess_anti_engine.uci.protocol import _GO_KEYWORDS, _INT_KEYS, CmdGo, CmdUnknown, parse_command
from chess_anti_engine.uci.selfcheck import build_parser as build_selfcheck_parser
from chess_anti_engine.uci.selfcheck import enforce_startup_gate
from chess_anti_engine.uci.subprocess_client import LineReader, send_line
from tests.test_uci_smoke import _make_tiny_checkpoint, _spawn_engine

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "TCEC_SETUP.md"
TEXT = DOC.read_text(encoding="utf-8")
FLAT = " ".join(TEXT.split())
SPANS = re.findall(r"`([^`\n]+)`", TEXT)
SEARCH = {o.name: o for o in SEARCH_OPTIONS}


def _section(title_prefix: str) -> str:
    start = TEXT.index(f"## {title_prefix}")
    end = TEXT.find("\n## ", start + 1)
    return TEXT[start:] if end < 0 else TEXT[start:end]


def _tables(section: str) -> list[list[list[str]]]:
    """Every markdown table in ``section``: rows of cells, header row dropped."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] | None = None
    for line in section.splitlines():
        if line.startswith("|") and set(line) <= {"|", "-", " "}:
            continue  # the separator under a header
        if not line.startswith("| "):
            current = None
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if current is None:
            current = []
            tables.append(current)
            continue  # header row
        current.append(cells)
    return tables


def _names(cell: str) -> set[str]:
    return set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", cell))


# --- the real engine: handshake and setoption probes --------------------------

_DOC_PROBES = (
    # The concrete replies the page quotes.
    "setoption name Hash value 512",
    "setoption name Hash value abc",
    "setoption name ClockBatchMarginSigmas value -1",
    "setoption name Topk value 1",
    "setoption name RPGOpenBudgetFrac value 2",
    "setoption name UCI_ShowWDL value maybe",
    "setoption name SearchParallel value bogus",
    "setoption name Foo value 1",
    "setoption name Hash",
    "setoption name Threads value 2",
)


def _option_lines(lines: list[str]) -> dict[str, dict[str, str]]:
    options: dict[str, dict[str, str]] = {}
    for line in lines:
        m = re.match(r"option name (\S+) type (\S+)(.*)$", line)
        if m is None:
            continue
        fields = dict(re.findall(r"(default|min|max) (\S+)", m.group(3)))
        fields["type"] = m.group(2)
        fields["vars"] = " / ".join(re.findall(r"var (\S+)", m.group(3)))
        options[m.group(1)] = fields
    return options


def _bad_value(name: str, fields: dict[str, str]) -> str:
    """A value outside the option's range, or one that cannot parse."""
    kind = fields["type"]
    if kind == "spin":
        return str(int(fields["min"]) - 1)
    if kind == "check":
        return "maybe"
    if kind == "combo":
        return "bogus"
    if name in SEARCH:
        hi = SEARCH[name].hi
        assert hi is not None
        return repr(float(hi) * 2 + 1)
    return {"ClockBatchMarginSigmas": "-1", "RPGOpenBudgetFrac": "2"}.get(name, "")


class _Run:
    def __init__(self, options: dict[str, dict[str, str]], replies: dict[str, list[str]],
                 lines: list[str], rc: int) -> None:
        self.options = options
        self.replies = replies  # setoption command -> lines before readyok
        self.lines = lines      # every line the engine printed
        self.rc = rc


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> _Run:
    """The real engine (default command line, CPU): `uci`, then one bad value
    per option and the page's own examples, then `quit`."""
    proc = _spawn_engine(_make_tiny_checkpoint(tmp_path_factory.mktemp("ckpt")))
    reader = LineReader(proc)
    lines: list[str] = []
    replies: dict[str, list[str]] = {}
    try:
        send_line(proc, "uci")
        lines += reader.read_until("uciok", timeout_s=60.0)
        options = _option_lines(lines)
        assert options, lines
        send_line(proc, "isready")
        lines += reader.read_until("readyok", timeout_s=120.0)
        probes = [f"setoption name {n} value {_bad_value(n, f)}" for n, f in options.items()
                  if _bad_value(n, f)]
        for cmd in (*probes, *_DOC_PROBES):
            send_line(proc, cmd)
            send_line(proc, "isready")
            got = reader.read_until("readyok", timeout_s=120.0)
            replies[cmd] = got[:-1]
            lines += got
        send_line(proc, "quit")
        rc = proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
    return _Run(options, replies, lines, rc)


def _reply(run: _Run, name: str) -> list[str]:
    fields = run.options[name]
    return run.replies[f"setoption name {name} value {_bad_value(name, fields)}"]


def _clamped(run: _Run, name: str) -> bool:
    return any(ln.startswith(f"info string {name} ") and "; using " in ln
               for ln in _reply(run, name))


def _refused(run: _Run, name: str) -> bool:
    return any(ln.startswith(f"info string {name}: ") and "; keeping " in ln
               for ln in _reply(run, name))


# --- UCI options: the table is the engine's own `uci` output ------------------


def _doc_options() -> dict[str, list[str]]:
    rows = _tables(_section("4. UCI options"))[0]
    return {row[0].strip("`"): row for row in rows}


def test_the_options_table_lists_exactly_the_options_the_engine_prints(run: _Run) -> None:
    assert sorted(_doc_options()) == sorted(run.options)


def test_each_option_row_matches_the_printed_type_range_and_default(run: _Run) -> None:
    for name, row in _doc_options().items():
        printed = run.options[name]
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


def test_each_string_search_option_states_the_range_the_code_enforces() -> None:
    rows = _doc_options()
    for opt in SEARCH_OPTIONS:
        if opt.kind != "string":
            continue
        assert f"string {opt.lo}..{opt.hi}" == rows[opt.name][1], (opt.name, rows[opt.name][1])


def test_threads_is_stated_to_have_no_effect_and_has_none(run: _Run) -> None:
    assert "**No effect**" in _doc_options()["Threads"][3]
    (reply,) = run.replies["setoption name Threads value 2"]
    assert "NO EFFECT" in reply, reply


# --- what a bad value does, per option class ----------------------------------


def _class_rows() -> list[list[str]]:
    (rows,) = [t for t in _tables(_section("4. UCI options"))[1:]
               if t and len(t[0]) == 3]
    return rows


def test_the_option_class_table_covers_every_option_once(run: _Run) -> None:
    named = [n for row in _class_rows() for n in _names(row[0])]
    assert len(named) == len(set(named)), named
    assert set(named) == set(run.options) - {"SyzygyPath", "LogFile"}


def test_clamped_spins_are_exactly_the_spins_the_engine_clamps(run: _Run) -> None:
    row = _class_rows()[0]
    assert row[1].startswith("clamped")
    spins = {n for n, f in run.options.items() if f["type"] == "spin"}
    assert _names(row[0]) == {n for n in spins if _clamped(run, n)}
    assert _names(row[0]).isdisjoint(SEARCH)


def test_the_search_options_row_is_the_registry_and_each_refuses(run: _Run) -> None:
    (row,) = [r for r in _class_rows() if r[0].startswith("The search options")]
    assert _names(row[0]) == set(SEARCH)
    assert row[1].startswith("refused")
    for name in SEARCH:
        assert _refused(run, name), (name, _reply(run, name))


def test_the_remaining_class_rows_match_the_engine(run: _Run) -> None:
    rows = {frozenset(_names(r[0])): r for r in _class_rows()}
    clamp = rows[frozenset({"ClockBatchMarginSigmas"})]
    assert clamp[1].startswith("below 0 is clamped")
    assert _clamped(run, "ClockBatchMarginSigmas")
    assert rows[frozenset({"RPGOpenBudgetFrac"})][1].startswith("refused")
    assert _refused(run, "RPGOpenBudgetFrac")
    checks = {n for n, f in run.options.items() if f["type"] == "check"}
    combos = {n for n, f in run.options.items() if f["type"] == "combo"}
    assert rows[frozenset(checks)][2].startswith("kept")
    assert rows[frozenset(combos)][2].startswith("refused")
    for name in checks | combos:
        assert _refused(run, name), (name, _reply(run, name))


# --- command-line flags -------------------------------------------------------


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
    rows = _tables(_section("3. Command lines"))[0]
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


# --- exit codes: what the entry points really return --------------------------


def _engine_rc(args: list[str], stdin: str = "") -> tuple[int, str]:
    env = {k: v for k, v in os.environ.items() if k != "DEEPFIN_CKPT"}
    env.update(PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, "-m", "chess_anti_engine.uci", *args], input=stdin,
        capture_output=True, text=True, timeout=300, env=env, cwd=ROOT, check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_documented_exit_codes_are_what_the_entry_points_return(
    run: _Run, tmp_path, monkeypatch, capsys,
) -> None:
    rows = _tables(_section("8. Positions, failures"))[0]
    documented = {row[0].strip("`") for row in rows if re.fullmatch(r"`\d+`", row[0])}
    observed = {str(run.rc)}  # a normal quit
    # 2: a bad command line (no --checkpoint) ...
    rc, _ = _engine_rc([])
    assert rc == 2
    observed.add(str(rc))
    # ... a C-extension refusal at startup, and an engine that cannot be imported.
    assert enforce_startup_gate(problems_fn=lambda: ["stale build"], stream=io.StringIO()) == 2

    def _no_engine(name: str) -> object:
        raise ImportError(f"no {name}")

    with monkeypatch.context() as patch:
        patch.setattr(cli.importlib, "import_module", _no_engine)
        assert cli.main([]) == 2
    assert capsys.readouterr().out == ""  # nothing on stdout
    # 1: an out-of-range search flag, and a startup that fails before readyok.
    ckpt = _make_tiny_checkpoint(tmp_path)
    rc, out = _engine_rc(["--checkpoint", str(ckpt), "--topk", "999"])
    assert rc == 1, out
    observed.add(str(rc))
    rc, out = _engine_rc(["--checkpoint", str(tmp_path / "missing.pt"), "--device", "cpu",
                          "--no-compile"], stdin="uci\nisready\nquit\n")
    assert rc == 1, out
    assert "engine startup failed" in out
    assert documented == observed == {"0", "1", "2"}


# --- watchdog and stop: the numbers are the code's ----------------------------


def test_watchdog_and_stop_figures_are_the_code_constants() -> None:
    k = engine_module._WATCHDOG_STALL_CHUNKS
    floor_ms = engine_module._WATCHDOG_STALL_FLOOR_MS
    ceil_s = engine_module._WATCHDOG_STALL_CEIL_MS / 1000.0
    last_ms = engine_module._WATCHDOG_LAST_MOMENT_MS
    stop_s = min(engine_module._JOIN_TIMEOUT_S, engine_module._GO_WAIT_NO_CLOCK_S)
    assert (f"no chunk for {k:g} chunk times (at least {floor_ms:g} ms, at most "
            f"{ceil_s:g} s)") in FLAT
    assert f"unconditionally {last_ms:g} ms before the real budget" in FLAT
    assert f"The {last_ms:g} ms is the watchdog's wake-up latency" in FLAT
    assert f"ignored `stop` for {stop_s:g} s" in FLAT
    assert f"the {stop_s:g} s `stop` bound" in FLAT
    assert f"{engine_module._GO_WAIT_MIN_S:g} s minimum `go` wait" in FLAT
    assert math.isclose(engine_module._GO_WAIT_MIN_S, 0.1)


# --- UCI commands -------------------------------------------------------------

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


def test_setoption_examples_name_real_options(run: _Run) -> None:
    for span in SPANS:
        m = re.match(r"setoption name (\S+)", span)
        if m:
            assert m.group(1) in run.options, span


# --- engine messages: in the code, or in the engine's real output -------------


class _Corpus:
    """The string pieces of chess_anti_engine/uci/*.py. A message built from an
    f-string whose hole is filled by another string (``f"info string
    position: {note}"`` with ``note = f"normalised castling rights ..."``)
    matches as the piece before the hole followed by the start of that string."""

    def __init__(self) -> None:
        self.pieces: list[str] = []      # every literal piece
        self.before_hole: list[str] = []  # pieces an f-string continues with a value
        self.starts: list[str] = []       # what a value can begin with
        for path in sorted((ROOT / "chess_anti_engine/uci").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.pieces.append(node.value)
                    self.starts.append(node.value)
                elif isinstance(node, ast.JoinedStr):
                    parts = node.values
                    for i, v in enumerate(parts):
                        if isinstance(v, ast.Constant) and isinstance(v.value, str):
                            nxt = parts[i + 1] if i + 1 < len(parts) else None
                            if isinstance(nxt, ast.FormattedValue):
                                self.before_hole.append(v.value)

    def has(self, fragment: str) -> bool:
        if any(fragment in s for s in self.pieces):
            return True
        for head in self.before_hole:
            for k in range(1, len(fragment)):
                if head.endswith(fragment[:k]) and any(
                        s.startswith(fragment[k:]) for s in self.starts):
                    return True
        return False


def _message_spans() -> list[str]:
    return [s for s in SPANS if s.startswith(("info string ", "deepfin: "))]


def test_every_quoted_engine_message_exists(run: _Run) -> None:
    corpus = _Corpus()
    spans = _message_spans()
    assert len(spans) >= 15, spans
    for span in spans:
        pattern = ".*".join(re.escape(part.strip()) for part in span.split("..."))
        if any(re.search(pattern, line) for line in run.lines):
            continue  # printed verbatim by the real engine
        for fragment in span.split("..."):
            fragment = fragment.strip()
            if len(fragment) < 6:
                continue
            bare = fragment.removeprefix("info string ").removeprefix("deepfin: ")
            assert corpus.has(fragment) or (bare and corpus.has(bare)), (span, fragment)
