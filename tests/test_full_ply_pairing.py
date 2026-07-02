"""Scheduler tests for SearchConfig.full_ply_pair_fraction (paired full plies).

The sf_p0 policy-teacher (one-ply label shift) only exists where ply t-1 is
also a full labeled ply; pairing doubles that adjacency without changing the
total full-ply fraction (= search cost and SF label volume).
"""
from __future__ import annotations

import numpy as np

from chess_anti_engine.selfplay.network_turn import _draw_is_full


def _simulate(pair_fraction: float, *, turns: int = 40_000, f: float = 0.25):
    rng = np.random.default_rng(7)
    force = [False]
    seq = np.zeros(turns, dtype=bool)
    for t in range(turns):
        seq[t] = _draw_is_full(
            rng, [0],
            playout_cap_fraction=f,
            pair_fraction=pair_fraction,
            force_flags=force,
        )[0]
    return seq


def test_pair_zero_preserves_iid_schedule() -> None:
    seq = _simulate(0.0)
    frac = seq.mean()
    assert 0.24 <= frac <= 0.26
    # adjacency at iid = the base rate itself
    adj = seq[1:][seq[:-1] & seq[1:]].size / max(1, seq[1:][seq[1:]].size)
    assert 0.21 <= adj <= 0.29


def test_pairing_keeps_full_fraction_and_doubles_adjacency() -> None:
    seq = _simulate(1.0)
    frac = seq.mean()
    # total full fraction is invariant: base rate is rescaled to the
    # stationary-corrected f / (1 + p*(1-f))
    assert 0.24 <= frac <= 0.26
    # P(prev full | full) at p=1, f=0.25: forced fulls (half of all fulls)
    # are always adjacent; a base full's prev can only be a FORCED full
    # (every base full forces at p=1), giving 0.5*1 + 0.5*P(F)/P(N) ~= 0.57
    fulls = np.flatnonzero(seq[1:]) + 1
    adj = seq[fulls - 1].mean()
    assert 0.50 <= adj <= 0.65


def test_forced_followup_consumes_flag_and_never_chains() -> None:
    rng = np.random.default_rng(0)
    force = [True, False]
    out = _draw_is_full(
        rng, [0, 1],
        playout_cap_fraction=0.0,  # no base fulls possible
        pair_fraction=1.0,
        force_flags=force,
    )
    assert bool(out[0]) is True      # forced slot fires
    assert bool(out[1]) is False
    assert force == [False, False]   # consumed, and a forced full never re-arms
