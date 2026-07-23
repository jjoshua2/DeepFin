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
