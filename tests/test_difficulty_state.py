"""Regression tests for DifficultyState — the single per-iteration source
of truth for opponent difficulty.

These tests pin down the invariant that ``ds.sf_nodes``, the manifest's
``sf_nodes``, and the local Stockfish's ``sf.nodes`` cannot diverge after
PID state is restored from a checkpoint — the failure mode flagged by the
adversarial review of commit eca3123.
"""
from __future__ import annotations

from chess_anti_engine.stockfish.pid import DifficultyPID
from chess_anti_engine.tune.trial_config import DifficultyState, TrialConfig


class _FakeSF:
    """Minimal stand-in for StockfishUCI — only .nodes + set_nodes()."""

    def __init__(self, nodes: int) -> None:
        self.nodes = int(nodes)

    def set_nodes(self, n: int) -> None:
        self.nodes = int(n)


def _build_pid(initial_nodes: int) -> DifficultyPID:
    return DifficultyPID(
        initial_nodes=initial_nodes,
        min_nodes=50,
        max_nodes=50_000,
        min_games_between_adjust=1,
        target_winrate=0.60,
        ema_alpha=0.35,
        initial_wdl_regret=0.3,
        wdl_regret_min=0.01,
        wdl_regret_max=1.0,
        wdl_regret_stage_end=0.0,
    )


def _fresh_tc() -> TrialConfig:
    return TrialConfig.from_dict({"device": "cpu", "sf_nodes": 5000})


def test_difficulty_state_prefers_pid_over_sf() -> None:
    """When pid exists, ds fields must come from pid — not sf.nodes, not tc."""
    pid = _build_pid(initial_nodes=1234)
    sf = _FakeSF(nodes=9999)  # deliberately divergent from pid
    tc = _fresh_tc()

    ds = DifficultyState.from_pid(pid, sf, tc)

    assert ds.sf_nodes == 1234
    assert ds.wdl_regret == 0.3


def test_difficulty_state_falls_back_to_sf_when_no_pid() -> None:
    """Gate-only config (pid disabled): ds.sf_nodes comes from sf."""
    sf = _FakeSF(nodes=2000)
    tc = _fresh_tc()

    ds = DifficultyState.from_pid(None, sf, tc)

    assert ds.sf_nodes == 2000
    assert ds.wdl_regret == -1.0


def test_difficulty_state_falls_back_to_tc_when_no_pid_no_sf() -> None:
    """Pure distributed (no local sf): ds.sf_nodes comes from tc.sf_nodes."""
    tc = _fresh_tc()

    ds = DifficultyState.from_pid(None, None, tc)

    assert ds.sf_nodes == 5000


def test_pid_restore_plus_sf_sync_keeps_manifest_reporting_aligned() -> None:
    """End-to-end: simulate the resume-with-gate-games path that the
    adversarial review flagged as split-brain.

    On resume:
      1. sf is created with nodes = tc.sf_nodes (default, e.g. 5000)
      2. pid is restored to a different value via load_state_dict (e.g. 2500)
      3. trainable.py syncs sf.set_nodes(pid.nodes)
      4. ds = DifficultyState.from_pid(pid, sf, tc)

    The manifest, training weights, and reporting all read from ds. Workers
    play at ds.sf_nodes. sf (used for gate games) runs at sf.nodes. Both
    must equal pid.nodes after the sync — otherwise workers play at one
    difficulty while gate games play at another.
    """
    tc = _fresh_tc()

    # Step 1: sf created with startup default.
    sf = _FakeSF(nodes=tc.sf_nodes)  # 5000

    # Step 2: pid exists with different restored nodes.
    pid = _build_pid(initial_nodes=tc.sf_nodes)
    restored_state = pid.state_dict()
    restored_state["nodes"] = 2500  # simulate restore to checkpointed value
    pid.load_state_dict(restored_state)
    assert pid.nodes == 2500
    assert sf.nodes == 5000  # pre-sync: divergent

    # Step 3: the sync trainable.py performs after pid.load_state_dict.
    sf.set_nodes(int(pid.nodes))

    # Step 4: ds snapshot.
    ds = DifficultyState.from_pid(pid, sf, tc)

    # Invariant: all three agree. Manifest (ds.sf_nodes) → workers,
    # sf.nodes → gate games, pid.nodes → PID observe. No split-brain.
    assert ds.sf_nodes == pid.nodes == sf.nodes == 2500


def test_difficulty_state_is_frozen() -> None:
    """ds must be immutable so nothing can mutate the iteration snapshot."""
    import dataclasses

    ds = DifficultyState.from_pid(None, None, _fresh_tc())
    try:
        setattr(ds, "sf_nodes", 9999)
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("DifficultyState should be frozen")
