"""Structural guards for the live seed-monitor orchestration."""
from __future__ import annotations

from pathlib import Path


def test_monitor_promotes_vetted_seeds_after_retirement_and_gate() -> None:
    script = Path("scripts/monitor_fen.sh").read_text(encoding="utf-8")

    retire_at = script.index("scripts/blindspot_retire_step.py")
    gate_at = script.index("scripts/harvest_gate_step.py")
    feed_at = script.index("scripts/blindspot_feed_step.py")

    assert retire_at < gate_at < feed_at
    assert '--batch "$STAGED_SEEDS"' in script
    assert '[ ! -f "$AUTO_FEED_DISABLED" ]' in script
    assert 'FEED_TAG="auto_ck${N}_$(date -u +%Y%m%dT%H%M%SZ)"' in script
    assert "The feed step is idempotent against the cumulative staging file" in script


def test_the_gate_does_not_derive_its_engine_or_tablebases_from_the_checkout() -> None:
    """⚑⚑ #441 review F2: neither exists IN the checkout, so neither may be
    derived from it.

    `e2e_server/publish/` is untracked runtime output and `data/syzygy_*` is
    151G of tablebases; both live only in the checkout that published them, so
    `$REPO_ROOT/...` is correct in the live tree and empty in every worktree.
    A missing engine is at least visible (the `-x` test skips the gate); a
    missing tablebase is not -- Stockfish answers `readyok` and exits 0 with
    `Found 0 WDL and 0 DTZ`. Both go through the shared discovery, which reaches
    the MAIN checkout, and `$REPO_ROOT` survives only as a last-resort fallback
    AFTER it.
    """
    script = Path("scripts/monitor_fen.sh").read_text(encoding="utf-8")

    for module in ("chess_anti_engine.utils.engine_discovery",
                   "chess_anti_engine.utils.syzygy"):
        assert f"python3 -m {module}" in script, f"{module} is not consulted"

    for var, discovery in (("SF_BIN", "engine_discovery"), ("SYZYGY_PATH", "syzygy")):
        assignments = [ln for ln in script.splitlines() if ln.startswith(f"{var}=")]
        assert len(assignments) == 2, (var, assignments)
        # The DISCOVERY assignment comes first; $REPO_ROOT is only the fallback
        # for the value discovery could not produce.
        assert discovery in assignments[0], (var, assignments[0])
        assert "$REPO_ROOT" not in assignments[0], (var, assignments[0])
        assert "$REPO_ROOT" in assignments[1], (var, assignments[1])
        assert assignments[1].startswith(f'{var}="${{{var}:-'), assignments[1]

    # And the env seams the operator uses are still ahead of both.
    assert 'SF_BIN="${HARVEST_SF_BIN:-' in script
    assert 'SYZYGY_PATH="${HARVEST_SYZYGY_PATH:-' in script
