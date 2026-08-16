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
import re
import subprocess
from pathlib import Path

import pytest

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


def _prov_shell(script_body: str) -> str:
    """Extract `_prov` from monitor_fen.sh so it can be exercised directly."""
    m = re.search(r"^    _prov\(\)[^\n]*\n(?:.*?\n)*?    \}\n", script_body, re.M)
    assert m, "monitor_fen.sh no longer defines a _prov helper"
    return m.group(0).replace("\n    ", "\n").lstrip()


@pytest.mark.parametrize(
    ("status", "log_text", "expected"),
    [
        pytest.param("0", "paired delta (A-B): -1.0\n", "", id="clean"),
        pytest.param("1", "disagree on stamp key\n", "REFUSED(1)", id="refused"),
        pytest.param(
            "0",
            "[paired-compare] WARNING: A carries no provenance stamp\n",
            "UNVERIFIED",
            id="unverified",
        ),
    ],
)
def test_prov_reports_refusal_and_warning(
    tmp_path: Path, status: str, log_text: str, expected: str,
) -> None:
    """The three outcomes a paired_compare run can have, as the monitor sees them."""
    log = tmp_path / "paired.log"
    log.write_text(log_text, encoding="utf-8")
    body = _prov_shell(MONITOR.read_text(encoding="utf-8"))
    script = f'PROV=""\n{body}\n_prov "vs_boot" "{status}" "{log}"\nprintf "%s" "$PROV"\n'
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True,
    ).stdout
    if expected:
        assert expected in out, out
        assert "vs_boot" in out
    else:
        assert out.strip() == "", out


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
