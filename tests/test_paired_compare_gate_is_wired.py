"""The stamp gate must be able to FIRE on the path that actually invokes it.

`paired_compare.require_same_stamp` opens with
``if not a.stamp or not b.stamp: warn; return``. So the gate's reach is exactly
the set of writers that stamp. Before the #442 review that set was
`audit_targets.py` alone — while `scripts/monitor_fen.sh`, the ONLY automated
caller of `paired_compare`, feeds it dumps from `blindspot_panel.py` and
`value_regret.py`, neither of which stamped. The gate was therefore inert on
every automated production comparison: correct code, correctly tested, and
never reached. That is this codebase's signature defect wearing the fix's own
clothes.

And the second half: `paired_compare`'s refusal (a non-zero exit) and its
warning (stdout) were both redirected into a per-cycle log that
`monitor_fen.sh` only ever grepped for "paired delta". A gate whose verdict
lands where nobody reads it has not been wired either.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from chess_anti_engine.utils.audit_cache_format import (
    AUDIT_CACHE_FORMAT,
    ROW_COUNT_KEY,
    STAMP_FORMAT_KEY,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
MONITOR = SCRIPTS / "monitor_fen.sh"


def _dump_writers() -> list[Path]:
    """Every script exposing `--dump-per-position`, discovered not listed.

    A hand-written list stops covering the next writer the moment someone adds
    one, which is the failure this whole file is about.
    """
    found = [p for p in sorted(SCRIPTS.glob("*.py"))
             if '"--dump-per-position"' in p.read_text(encoding="utf-8")]
    assert len(found) >= 3, f"discovery found only {found} — the sweep is vacuous"
    return found


@pytest.mark.parametrize("script", _dump_writers(), ids=lambda p: p.name)
def test_every_dump_per_position_writer_stamps(script: Path) -> None:
    """A writer that does not stamp makes the gate short-circuit to a warning.

    ⚑ Asserts the writer is CALLED, via the AST — not that the name appears in
    the file. A substring check stays green when the call is removed and only
    the import is left behind, which is the same "present, therefore fine"
    reasoning that produced the defect being fixed one level down.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "write_audit_cache" in called, (
        f"{script.name} writes a --dump-per-position file without CALLING "
        "write_audit_cache, so the file carries no provenance stamp. "
        "paired_compare.require_same_stamp returns early on an unstamped side, "
        "so every comparison of this dump silently skips the ruler check."
    )


def test_the_non_validating_dump_readers_skip_the_header() -> None:
    """⚑ #442 review B1: the format change must not leave a reader behind.

    `git grep` over the tree finds exactly two in-repo consumers of a
    `--dump-per-position` file: `audit_compare_buckets.py`, which already goes
    through the VALIDATING `read_audit_cache_by_key`, and `tail_stats.py`,
    which parsed every line itself. `tail_stats` was immune only by accident —
    the stamp carries no `value` and no `cand`, so its numeric test happened to
    drop the header — and an accident is not a property. Asserted through the
    AST, so deleting the call and leaving the import cannot pass.
    """
    script = SCRIPTS / "tail_stats.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "iter_data_rows" in called, (
        f"{script.name} parses a per-position dump without the shared "
        "header-skipping reader, so it counts the provenance stamp as a scored "
        "position the moment the stamp grows a field its rows also carry."
    )


def _prov_shell(script_body: str) -> str:
    """Extract `_prov` from monitor_fen.sh so it can be exercised directly."""
    m = re.search(r"^    _prov\(\)[^\n]*\n(?:.*?\n)*?    \}\n", script_body, re.M)
    assert m, "monitor_fen.sh no longer defines a _prov helper"
    return m.group(0).replace("\n    ", "\n").lstrip()


def _run_prov(log: Path, status: int) -> str:
    """`_prov` as monitor_fen.sh defines it, run over one real log."""
    body = _prov_shell(MONITOR.read_text(encoding="utf-8"))
    script = f'PROV=""\n{body}\n_prov "vs_boot" "{status}" "{log}"\nprintf "%s" "$PROV"\n'
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True,
    ).stdout


# One data row per dump is enough: `_prov` reads the WARNINGS, not the verdict.
def _rows(*, batch_size: bool = True) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for i, fen in enumerate((
        "8/8/6r1/1np5/p2k4/P7/8/2K5 w - - 8 1",
        "8/7k/2K4p/8/2b3P1/5p2/7q/8 w - - 0 1",
        "8/6k1/p7/4pp1p/8/5K1P/3B1PP1/8 w - - 0 1",
    )):
        row: dict[str, object] = {
            "fen": fen, "phase": 0, "value": 10.0 + i, "input_encoding": "fen_only",
        }
        if batch_size:
            row["batch_size"] = 128
        out.append(row)
    return out


def _dump(
    path: Path,
    *,
    stamp: dict[str, object] | None,
    rows: list[dict[str, object]] | None = None,
) -> Path:
    """Write a dump by hand — no torch, no checkpoint, no GPU.

    `stamp=None` writes the LEGACY unstamped shape, which is what all three
    banked production baselines still are.
    """
    body = _rows() if rows is None else rows
    lines = []
    if stamp is not None:
        lines.append(json.dumps({**stamp, ROW_COUNT_KEY: len(body)}, sort_keys=True))
    lines += [json.dumps(r) for r in body]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


_STAMP: dict[str, object] = {
    STAMP_FORMAT_KEY: AUDIT_CACHE_FORMAT,
    "policy_map_version": "aaaaaaaaaaaaaaaa",
    "audit_ruler_version": "bbbbbbbbbbbbbbbb",
    "producer": "value_regret.py --dump-per-position",
    "input_encoding": "fen_only",
    "audit_set": "data/audit_set_v1.jsonl",
    "audit_set_digest": "cccccccccccccccc",
}


def _paired_compare(a: Path, b: Path, log: Path) -> int:
    """Run the REAL script and bank its log exactly as monitor_fen.sh does."""
    proc = subprocess.run(
        ["python3", str(SCRIPTS / "paired_compare.py"), str(a), str(b),
         "--label-a", "boot512", "--label-b", "ck_0", "--n-boot", "200"],
        capture_output=True, text=True, check=False, cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return proc.returncode


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        pytest.param("clean", "", id="clean-is-prov-ok"),
        pytest.param("unstamped", "vs_boot:UNVERIFIED(UNSTAMPED)", id="unstamped"),
        pytest.param("partial", "vs_boot:UNVERIFIED(PARTIAL)", id="partial"),
        pytest.param("ruler", "vs_boot:UNVERIFIED(RULER)", id="ruler-only"),
        pytest.param("refused", "vs_boot:REFUSED(1)", id="refused"),
    ],
)
def test_prov_names_each_outcome_of_a_real_paired_compare(
    tmp_path: Path, case: str, expected: str,
) -> None:
    """⚑ END-TO-END, through the real script — and BOTH branches are exercised.

    The previous version of this test fed `_prov` SYNTHETIC log text, and its
    "clean" case was a line no production log has ever looked like. That is how
    the first `_prov` shipped grepping for the bare word "WARNING", which also
    matches `require_same_ruler`'s row-level warnings: measured against the real
    `paired_*_vs_boot.log`, every deep cycle emitted "'batch_size' not declared
    by boot512", so the monitor printed UNVERIFIED forever and `prov:ok` was
    UNREACHABLE — for a reason unrelated to the provenance stamp (#442 review
    B2). Driving the real `paired_compare` means these greps cannot drift out
    from under its message strings without this test going red.

    `clean` is the passing control. A gate that cannot PASS is the same defect
    as one that cannot FAIL, in the other direction.
    """
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    if case == "clean":
        _dump(a, stamp=_STAMP)
        _dump(b, stamp=_STAMP)
    elif case == "unstamped":
        # The production shape after the B4 re-dump is skipped: a legacy
        # baseline whose ROWS still declare the ruler, against a stamped dump.
        _dump(a, stamp=None)
        _dump(b, stamp=_STAMP)
    elif case == "partial":
        _dump(a, stamp=_STAMP)
        _dump(b, stamp={**_STAMP, "topk": 5})
    elif case == "ruler":
        _dump(a, stamp=_STAMP, rows=_rows(batch_size=False))
        _dump(b, stamp=_STAMP)
    else:
        _dump(a, stamp=_STAMP)
        _dump(b, stamp={**_STAMP, "audit_ruler_version": "dddddddddddddddd"})
    log = tmp_path / "paired.log"
    status = _paired_compare(a, b, log)
    assert status == (1 if case == "refused" else 0), log.read_text(encoding="utf-8")
    out = _run_prov(log, status)
    assert out.strip() == expected, f"PROV={out!r}\n--- log ---\n{log.read_text()}"


# Verbatim from scratchpad/live_read/monitor/paired_0_vs_boot.log on the live
# training box, 2026-08-16 — the shape every deep cycle actually produces
# today. Inlined rather than read from scratchpad/ so CI, which has no such
# file, tests the same thing the box does.
_REAL_PRODUCTION_LOG = """\
[paired-compare] ruler input_encoding="fen_only" (one side inferred from an unstamped dump)
[paired-compare] WARNING: 'batch_size' not declared by boot512 — cannot verify both sides used the same ruler. Re-dump with a current scorer to make this checkable.
paired positions: 1723
  A: 2000 rows, 0 unusable, 277 unmatched   B: 1723 rows, 0 unusable, 0 unmatched
A = boot512: mean 74.38
B = ck_0: mean 55.31
paired delta (A-B): +19.07  [95% CI +8.60 .. +30.87]
verdict at 95%: B better   (A better 19.3% / B better 27.1% / tied 53.6%)
  endgame     n= 1055 delta +25.48 [+10.09 .. +43.43]
  middlegame  n=  668 delta +8.95 [-3.31 .. +23.02]
"""


def test_the_real_production_log_is_not_read_as_a_stamp_failure(tmp_path: Path) -> None:
    """The calibration case: what the monitor sees on the live box, today.

    Under the first `_prov` this log yielded a bare `vs_boot:UNVERIFIED`, which
    an operator reads as "the provenance stamp could not be checked". It is not
    that — there is no stamp warning in it at all. The token must name the
    check that actually spoke.
    """
    log = tmp_path / "paired_0_vs_boot.log"
    log.write_text(_REAL_PRODUCTION_LOG, encoding="utf-8")
    assert _run_prov(log, 0).strip() == "vs_boot:UNVERIFIED(RULER)"


def test_prov_ok_survives_a_log_that_merely_mentions_a_warning(tmp_path: Path) -> None:
    """⚑ THE UNREACHABILITY GUARD. Fails if `prov:ok` stops being reachable.

    `_prov` grepping the bare word "WARNING" is what made `prov:ok` dead, and
    the shape that revives that bug is any grep loose enough to match text a
    clean run can carry. A label, a filename or a `verdict` line mentioning the
    word must not consume the passing state.
    """
    log = tmp_path / "paired.log"
    log.write_text(
        "[paired-compare] ruler input_encoding=\"fen_only\" (both sides)\n"
        "[paired-compare] ruler batch_size=128 (both sides)\n"
        "paired delta (A-B): -1.0  [95% CI -2.0 .. +0.1]\n"
        "verdict at 95%: NOT significant\n"
        "  WARNING-vs-baseline  n=  35 delta -0.01\n",
        encoding="utf-8",
    )
    assert _run_prov(log, 0).strip() == ""


def test_the_monitor_line_carries_the_provenance_verdict() -> None:
    """⚑ The verdict must reach the line an operator actually reads.

    `$PROV` on the deep-cycle echo is the whole point: the per-cycle log is
    grepped only for "paired delta", so a refusal used to read as an empty
    delta field and a warning as nothing at all.
    """
    text = MONITOR.read_text(encoding="utf-8")
    deep_lines = [
        line for line in text.splitlines()
        if 'echo "[monitor' in line and "vs_boot:" in line
    ]
    assert deep_lines, "the deep-cycle monitor line moved; this guard is stale"
    for line in deep_lines:
        assert "$PROV" in line or "${PROV" in line, (
            f"the deep monitor line does not carry the provenance verdict: {line}"
        )
        # ⚑ And the PASSING state must be printed, not merely left blank. An
        # empty `$PROV` rendering as an empty field is indistinguishable from
        # the monitor never having run the check — the operator cannot tell a
        # clean provenance read from a missing one.
        assert "prov:ok" in line, (
            "the deep monitor line has no explicit clean-provenance token, so a "
            f"passing check is invisible: {line}"
        )


def test_every_paired_compare_call_is_followed_by_a_prov_check() -> None:
    """A new comparison must not be able to land unwatched."""
    lines = MONITOR.read_text(encoding="utf-8").splitlines()
    calls = [i for i, line in enumerate(lines) if "scripts/paired_compare.py" in line]
    assert calls, "no paired_compare invocation found; this guard is stale"
    for i in calls:
        window = "\n".join(lines[i:i + 6])
        assert "_prov " in window, (
            f"paired_compare invoked at line {i + 1} with no _prov check within "
            f"5 lines — its refusal would land in a log nobody greps:\n{window}"
        )
