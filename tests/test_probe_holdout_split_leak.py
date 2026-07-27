"""Row-carving and CI machinery of scripts/probe_holdout_split_leak.py.

The probe's whole claim is that set B shares no game with the training window
and that the two sets differ in nothing but that. Those are properties of the
carving, not of the GPU pass, so they are testable on CPU — including a
mutation test that breaks the disjointness and checks the assertion goes red,
because a check that has never failed is not known to work.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts import probe_holdout_split_leak as probe
from scripts.paired_compare import paired_bootstrap_ci


def _index(game_sizes: list[int], *, wdl_of_game: list[int]) -> probe.RowIndex:
    """A RowIndex of whole games; wdl is constant within a game, as in production."""
    game_id = np.concatenate([
        np.full((n,), g, dtype=np.int64) for g, n in enumerate(game_sizes)
    ])
    wdl = np.concatenate([
        np.full((n,), wdl_of_game[g], dtype=np.int64) for g, n in enumerate(game_sizes)
    ])
    ply = np.concatenate([np.arange(n, dtype=np.int64) for n in game_sizes])
    total = int(game_id.shape[0])
    return probe.RowIndex(
        shard_pos=np.zeros((total,), dtype=np.int64),
        row=np.arange(total, dtype=np.int64),
        game_id=game_id, wdl=wdl, ply=ply,
        is_selfplay=np.ones((total,), dtype=np.int64),
    )


def _carve_and_check(
    index: probe.RowIndex, train_games: np.ndarray, *, target_rows: int, seed: int = 0,
) -> np.ndarray:
    """The two lines of the probe under test: carve, then verify disjointness."""
    sel = probe.carve_game_disjoint(
        index, train_games=train_games, target_rows=target_rows,
        rng=np.random.default_rng(seed),
    )
    probe.assert_game_disjoint(index.game_id[sel], train_games)
    return sel


# --------------------------------------------------------------------------
# game-disjointness
# --------------------------------------------------------------------------


def test_game_disjoint_carve_shares_no_game_with_training() -> None:
    rng = np.random.default_rng(7)
    index = _index(list(rng.integers(8, 30, size=60)), wdl_of_game=list(rng.integers(0, 3, size=60)))
    train_games = np.arange(0, 40, dtype=np.int64)

    sel = _carve_and_check(index, train_games, target_rows=200)

    assert sel.size >= 200
    assert np.intersect1d(np.unique(index.game_id[sel]), train_games).size == 0
    # whole games only: every row of every picked game is present
    for game in np.unique(index.game_id[sel]):
        assert int((index.game_id[sel] == game).sum()) == int((index.game_id == game).sum())


def test_broken_disjointness_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation test: drop the train_games filter and the check must go red."""
    rng = np.random.default_rng(7)
    index = _index(list(rng.integers(8, 30, size=60)), wdl_of_game=list(rng.integers(0, 3, size=60)))
    train_games = np.arange(0, 40, dtype=np.int64)

    def _broken(index, *, target_rows, **_ignored):
        """The mutant: pick rows without ever consulting ``train_games``."""
        return np.arange(min(target_rows, len(index)), dtype=np.int64)

    monkeypatch.setattr(probe, "carve_game_disjoint", _broken)
    with pytest.raises(AssertionError, match="contaminated"):
        _carve_and_check(index, train_games, target_rows=200)


def test_assert_game_disjoint_accepts_a_clean_set() -> None:
    probe.assert_game_disjoint(np.array([5, 5, 6]), np.array([1, 2, 3]))


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------


def test_matched_sets_have_identical_count_and_wdl_mix() -> None:
    rng = np.random.default_rng(3)
    wdl_a = rng.choice([0, 1, 2], size=900, p=[0.5, 0.2, 0.3])
    wdl_b = rng.choice([0, 1, 2], size=1500, p=[0.2, 0.5, 0.3])

    sel_a, sel_b = probe.match_wdl_mix(wdl_a, wdl_b, rng=np.random.default_rng(0))

    assert sel_a.size == sel_b.size
    hist_a = np.bincount(wdl_a[sel_a], minlength=3)
    hist_b = np.bincount(wdl_b[sel_b], minlength=3)
    assert hist_a.tolist() == hist_b.tolist()
    # per class it keeps everything it can, i.e. it subsamples the larger side
    for cls in range(3):
        expected = min(int((wdl_a == cls).sum()), int((wdl_b == cls).sum()))
        assert int(hist_a[cls]) == expected
    # subsampled, not reweighted: the kept rows are real, distinct rows
    assert np.unique(sel_a).size == sel_a.size
    assert np.unique(sel_b).size == sel_b.size
    # sorted, which is what score_rows requires to keep the row order aligned
    assert np.all(np.diff(sel_a) > 0)
    assert np.all(np.diff(sel_b) > 0)


# --------------------------------------------------------------------------
# the per-row split
# --------------------------------------------------------------------------


def test_per_row_draw_reproduces_the_production_split() -> None:
    """Same expression, same generator sequence as ``_ingest_train_arrays``."""
    counts = [2000, 2000, 685]
    frac = 0.02

    got = probe.per_row_holdout_mask(counts, holdout_fraction=frac, seed=1234)

    rng = np.random.default_rng(1234)
    want = np.concatenate([rng.random(n) < frac for n in counts])
    assert np.array_equal(got, want)
    # and the statistic the audit quotes: ~2% of rows, drawn independently
    assert abs(got.mean() - frac) < 0.006
    assert got.shape[0] == sum(counts)


def test_sibling_stats_count_same_game_training_rows() -> None:
    # two games of 10 plies; hold out row 0 of game 0 and rows 0,1 of game 1
    index = _index([10, 10], wdl_of_game=[0, 2])
    mask = np.zeros((20,), dtype=bool)
    mask[[0, 10, 11]] = True

    stats = probe.sibling_stats(index.game_id, mask)

    assert stats["n_holdout"] == 3.0
    # game 0 leaves 9 training rows, game 1 leaves 8 -> mean over held rows
    assert stats["expected_siblings"] == pytest.approx((9 + 8 + 8) / 3)
    assert stats["p_no_sibling"] == 0.0
    assert stats["plies_per_game"] == pytest.approx(10.0)


def test_sibling_stats_flags_a_game_with_no_training_rows() -> None:
    index = _index([3, 10], wdl_of_game=[0, 1])
    mask = np.zeros((13,), dtype=bool)
    mask[[0, 1, 2]] = True  # all of game 0

    stats = probe.sibling_stats(index.game_id, mask)

    assert stats["p_no_sibling"] == 1.0
    assert stats["expected_siblings"] == 0.0


# --------------------------------------------------------------------------
# the CI
# --------------------------------------------------------------------------


def _clustered_sample(
    rng: np.random.Generator, *, n_games: int, rows_per_game: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rows whose value is dominated by a per-GAME effect, as losses are."""
    game_effect = rng.normal(0.0, 1.0, size=n_games)
    game_id = np.repeat(np.arange(n_games, dtype=np.int64), rows_per_game)
    values = game_effect[game_id] + rng.normal(0.0, 0.2, size=game_id.shape[0])
    return values, game_id


def test_cluster_bootstrap_covers_at_the_nominal_rate() -> None:
    """Coverage check: the CI must contain the truth ~95% of the time.

    Hand-rolled rather than reused because scripts/paired_compare.py bootstraps
    ROWS, and set B's rows come ~16 to a game. The second assertion is the
    reason the hand-rolled version exists: the row bootstrap under-covers badly
    on exactly this data.
    """
    n_games, rows_per_game, replicates = 40, 16, 120
    weights = {0: 1.0}
    covered_cluster = 0
    covered_rows = 0
    for rep in range(replicates):
        rng = np.random.default_rng(1000 + rep)
        values, game_id = _clustered_sample(rng, n_games=n_games, rows_per_game=rows_per_game)
        wdl = np.zeros_like(game_id)
        boots = probe.cluster_bootstrap_means(
            values, game_id, wdl, stratum_weights=weights, n_boot=200,
            rng=np.random.default_rng(rep),
        )
        lo, hi = np.percentile(boots, [2.5, 97.5])
        covered_cluster += int(lo <= 0.0 <= hi)
        rlo, rhi = paired_bootstrap_ci(values, n_boot=200, seed=rep)
        covered_rows += int(rlo <= 0.0 <= rhi)

    assert covered_cluster / replicates > 0.88, covered_cluster
    assert covered_rows / replicates < 0.60, covered_rows


def test_class_aligned_order_pairs_like_with_like() -> None:
    """Row i of one set must be the same W/D/L class as row i of the other."""
    wdl_a = np.array([0, 2, 1, 0, 2])
    wdl_b = np.array([2, 0, 0, 2, 1])

    ord_a, ord_b = probe.class_aligned_order(wdl_a, wdl_b)

    assert np.array_equal(wdl_a[ord_a], wdl_b[ord_b])
    # stable, so it is a deterministic pairing rather than an arbitrary one
    assert ord_a.tolist() == [0, 3, 2, 1, 4]


def test_class_aligned_order_rejects_unmatched_mixes() -> None:
    with pytest.raises(AssertionError, match="W/D/L counts differ"):
        probe.class_aligned_order(np.array([0, 0, 1]), np.array([0, 1, 1]))


def test_stratified_mean_uses_the_supplied_weights() -> None:
    values = np.array([1.0, 1.0, 3.0, 3.0])
    wdl = np.array([0, 0, 1, 1])

    assert probe.stratified_mean(values, wdl, {0: 1.0, 1: 1.0}) == pytest.approx(2.0)
    assert probe.stratified_mean(values, wdl, {0: 3.0, 1: 1.0}) == pytest.approx(1.5)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


class _StubNet(torch.nn.Module):
    """A two-head net just big enough for compute_loss; no checkpoint needed."""

    def __init__(self, *, planes: int, policy_size: int) -> None:
        super().__init__()
        self.policy = torch.nn.Linear(planes * 64, policy_size)
        self.wdl = torch.nn.Linear(planes * 64, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = x.reshape(x.shape[0], -1)
        return {"policy": self.policy(flat), "wdl": self.wdl(flat)}


def _memory_source(n: int, *, planes: int = 3, policy_size: int = 8) -> probe.RowSource:
    rng = np.random.default_rng(11)
    policy = rng.random((n, policy_size)).astype(np.float32)
    policy /= policy.sum(axis=1, keepdims=True)
    arrays = {
        "x": rng.random((n, planes, 8, 8)).astype(np.float32),
        "policy_target": policy,
        "wdl_target": rng.integers(0, 3, size=n).astype(np.int8),
    }
    index = probe.RowIndex(
        shard_pos=np.full((n,), -1, dtype=np.int64), row=np.arange(n, dtype=np.int64),
        game_id=np.arange(n, dtype=np.int64) // 4,
        wdl=arrays["wdl_target"].astype(np.int64),
        ply=np.arange(n, dtype=np.int64), is_selfplay=np.ones((n,), dtype=np.int64),
    )
    return probe.RowSource("stub", index, [], arrays=arrays)


def test_score_rows_per_row_losses_reconstruct_the_batched_loss() -> None:
    torch.manual_seed(0)
    source = _memory_source(12)
    model = _StubNet(planes=3, policy_size=8).eval()

    scored = probe.score_rows(
        model, source, np.arange(12, dtype=np.int64), loss_kwargs={},
        device="cpu", batch_size=5, history_encoding="legacy",
    )

    assert scored["wdl_ce"].shape == (12,)
    assert scored["total"].shape == (12,)
    assert np.all(np.isfinite(scored["wdl_ce"]))
    # the per-row decomposition is checked against compute_loss over the chunk
    assert scored["max_reconstruction_dev"][0] < 1e-5


def test_score_rows_rejects_an_unsorted_selection() -> None:
    source = _memory_source(6)
    model = _StubNet(planes=3, policy_size=8).eval()

    with pytest.raises(ValueError, match="strictly increasing"):
        probe.score_rows(
            model, source, np.array([3, 1, 0]), loss_kwargs={},
            device="cpu", batch_size=4, history_encoding="legacy",
        )


def test_gather_returns_rows_in_selection_order() -> None:
    source = _memory_source(10)
    sel = np.array([1, 4, 7], dtype=np.int64)

    got = source.gather(sel)

    assert np.array_equal(got["wdl_target"], source.index.wdl[sel].astype(np.int8))
