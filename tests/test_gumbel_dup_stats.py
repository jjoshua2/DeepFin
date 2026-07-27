"""Pins the duplicate-rate telemetry against an exact, independent count.

The telemetry exists to answer one question -- what share of production's NN
forwards carry no information -- and it is only worth having if it agrees with
ground truth. It was wrong twice before it was right: a per-prepare-call C
counter read 0.0% where the truth was 77.5% (duplicates accumulate ACROSS the
reps that fill one GPU batch), and a strided fingerprint read 52.3% where the
truth was 0.0% (the encoded planes are mostly zero, so subsamples collide).
"""
from __future__ import annotations

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts import gumbel_c as gumbel_c_mod
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import (
    duplicate_rate,
    duplicate_stats,
    reset_duplicate_stats,
    run_gumbel_root_many_c,
)

from tests.test_gumbel_c_leaf_duplication import _FENS, _CountingHashEvaluator


def _run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_batch: int = 0,
    vloss_weight: int = 0,
    vloss_mode: int = 0,
) -> tuple[float | None, float]:
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    monkeypatch.setattr(gumbel_c_mod, "_DUP_STATS", True)
    monkeypatch.setattr(gumbel_c_mod, "_DUP_STRIDE", 1)
    monkeypatch.setattr(gumbel_c_mod, "_DUP_SAMPLE", 1)
    reset_duplicate_stats()
    ev = _CountingHashEvaluator()
    # DISTINCT positions only: repeating a FEN would make the boards themselves
    # duplicates and inflate the rate for a reason that has nothing to do with
    # the search.
    boards = [chess.Board(f) for f in _FENS]
    cfg = GumbelConfig(
        simulations=256, topk=16, c_scale=0.1, temperature=0.0, add_noise=False,
    )
    run_gumbel_root_many_c(
        None, boards, device="cpu", rng=np.random.default_rng(0), cfg=cfg,
        evaluator=ev, target_batch=target_batch,
        vloss_weight=vloss_weight, vloss_mode=vloss_mode,
    )
    return duplicate_rate(), 1.0 - ev.distinct_rows / ev.rows


def test_telemetry_matches_an_exact_independent_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production batching: heavy duplication, and we measure the real amount."""
    got, truth = _run(monkeypatch, target_batch=0)
    assert got is not None
    assert truth > 0.5, f"fixture stopped duplicating ({truth:.3f}); test is void"
    assert abs(got - truth) < 0.01, f"telemetry {got:.3f} vs exact {truth:.3f}"


@pytest.mark.parametrize(
    ("target_batch", "vloss_weight", "vloss_mode"),
    [(1, 0, 0), (0, 1, 0), (0, 1, 1)],
)
def test_telemetry_reads_zero_when_duplication_is_removed(
    monkeypatch: pytest.MonkeyPatch,
    target_batch: int, vloss_weight: int, vloss_mode: int,
) -> None:
    """A zero here is a measurement, not an absence of one."""
    got, truth = _run(
        monkeypatch, target_batch=target_batch,
        vloss_weight=vloss_weight, vloss_mode=vloss_mode,
    )
    assert got is not None, "reported no observations where it should have data"
    assert truth == 0.0
    assert got < 0.01, f"telemetry {got:.3f} on an arm with no duplication"


def test_no_observations_is_distinguishable_from_no_duplication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure that motivated `duplicate_rate` returning None.

    Sampling can hash nothing at all -- a search issues ~18 batches, so
    CAE_DUP_SAMPLE=20 observes none of them. `duplicate_stats()` then holds
    0 rows and 0 duplicates, and a naive ratio reports 0.0 = "no duplication"
    for a run whose true rate is over 50%.
    """
    monkeypatch.setattr(gumbel_c_mod, "_DUP_STATS", True)
    monkeypatch.setattr(gumbel_c_mod, "_DUP_SAMPLE", 10_000)
    monkeypatch.setattr(gumbel_c_mod, "_COMPILED_BATCH_BUCKETS", ())
    reset_duplicate_stats()
    boards = [chess.Board(f) for f in _FENS]
    cfg = GumbelConfig(
        simulations=64, topk=16, c_scale=0.1, temperature=0.0, add_noise=False,
    )
    run_gumbel_root_many_c(
        None, boards, device="cpu", rng=np.random.default_rng(0), cfg=cfg,
        evaluator=_CountingHashEvaluator(), target_batch=0,
    )
    rows, dupes, hashed = duplicate_stats()
    assert hashed == 0
    assert rows == 0
    assert dupes == 0
    assert duplicate_rate() is None, "0/0 must not read as 0.0"


def test_disabled_telemetry_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It hashes rows on the hot path (~31% on a CPU benchmark), so it must be
    genuinely inert when off -- not merely default-off."""
    monkeypatch.setattr(gumbel_c_mod, "_DUP_STATS", False)
    reset_duplicate_stats()
    gumbel_c_mod._record_batch_dup(np.zeros((64, 175 * 64), dtype=np.float32))
    assert duplicate_stats() == (0, 0, 0)
    assert duplicate_rate() is None
