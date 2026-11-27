"""Synthetic CPU checks only: no models, data, search, or GPU."""
from typing import Any

import numpy as np
import pytest

from scripts.stored_wdl_compare import LAYOUT, compare, rows_digest


def fixture():
    x = np.zeros((3, 175, 8, 8), dtype=np.float32)
    x[:, 17, 0, 0] = [1, 2, 3]  # history plane signal, not a bare FEN
    target = np.tile([1., 0., 0.], (3, 1))
    kwargs: dict[str, Any] = {
        "row_ids": ["corpus/g1/p1", "corpus/g1/p2", "corpus/g2/p1"],
        "game_ids": ["corpus/g1", "corpus/g1", "corpus/g2"],
        "donor_sha256": "a" * 64, "candidate_sha256": "b" * 64, "layout": LAYOUT,
        "donor_training_games": frozenset({"training/g1"}),
        "candidate_training_games": frozenset({"training/g2"}),
        "target_description": "synthetic outcome, stored-row side-to-move W,D,L",
        "bootstraps": 100, "seed": 7,
        "donor": lambda a: np.tile([.5, .25, .25], (len(a), 1)),
        "candidate": lambda a: np.tile([.75, .125, .125], (len(a), 1)),
    }
    kwargs["expected_rows_sha256"] = rows_digest(x, kwargs["row_ids"], kwargs["game_ids"])
    return x, target, kwargs


def test_known_paired_deltas_and_repeatability():
    x, target, kw = fixture()
    result = compare(x, target, **kw)
    assert result == compare(x, target, **kw)
    assert result["metrics"]["wdl_log_loss"]["delta"] == pytest.approx(np.log(2 / 3))
    assert result["metrics"]["wdl_brier"]["delta"] == pytest.approx(-.28125)
    assert result["source_games"] == 2


def test_forwards_consume_same_history_and_do_not_share_mutable_arrays():
    x, target, kw = fixture()
    seen = []
    def adapter(a):
        seen.append(a.copy())
        assert not a.flags.writeable
        return np.tile([.5, .25, .25], (len(a), 1))
    kw.update(donor=adapter, candidate=adapter)
    result = compare(x, target, **kw)
    assert all(np.array_equal(a, x) for a in seen)
    assert result["metrics"]["wdl_brier"]["paired_percentile_95"] == [0., 0.]


def test_equal_game_weight_not_equal_row_weight():
    x, target, kw = fixture()
    kw["candidate"] = lambda a: np.array([[1., 0., 0.], [1., 0., 0.], [0., 0., 1.]])
    result = compare(x, target, **kw)
    assert result["metrics"]["wdl_brier"]["candidate"] == 1.
    assert result["metrics"]["wdl_brier"]["delta"] == .625


@pytest.mark.parametrize("change", [
    {"layout": "fen-only"}, {"donor_sha256": "not-a-hash"},
    {"candidate_training_games": frozenset({"corpus/g1"})},
    {"expected_rows_sha256": "c" * 64}, {"bootstraps": 0}, {"seed": -1},
    {"target_description": ""}, {"donor_training_games": set()},
])
def test_contract_rejections_before_inference(change):
    x, target, kw = fixture()
    def forbidden(_):
        pytest.fail("inference reached")
    kw.update(donor=forbidden, candidate=forbidden)
    kw.update(change)
    with pytest.raises(ValueError, match=r"."):
        compare(x, target, **kw)


@pytest.mark.parametrize("value", [
    [[1., 1., 0.]] * 3, [[float("nan"), 0., 1.]] * 3,
    [[-.1, .1, 1.]] * 3, [[1., 0.]] * 3,
])
def test_invalid_predictions(value):
    x, target, kw = fixture()
    kw["candidate"] = lambda _: np.array(value)
    with pytest.raises(ValueError, match=r"."):
        compare(x, target, **kw)


def test_mutating_adapter_rejected():
    x, target, kw = fixture()
    def mutate(a):
        a.flags.writeable = True
        a[0, 17, 0, 0] = 8
        return np.tile([1., 0., 0.], (len(a), 1))
    kw["donor"] = mutate
    with pytest.raises(ValueError, match="mutated"):
        compare(x, target, **kw)
    assert x[0, 17, 0, 0] == 1


def test_duplicate_rows_and_one_game_refused():
    x, target, kw = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        rows_digest(x, ["same"] * 3, kw["game_ids"])
    kw["game_ids"] = ["same"] * 3
    kw["expected_rows_sha256"] = rows_digest(x, kw["row_ids"], kw["game_ids"])
    with pytest.raises(ValueError, match="two independent"):
        compare(x, target, **kw)


@pytest.mark.parametrize("replacement", [[0., 0., 1.], [float("nan"), -1., 2.]])
def test_callback_cannot_change_target_or_identity_snapshot(replacement):
    x, target, kw = fixture()
    expected = compare(x, target, **kw)
    def mutate_external(a):
        target[:] = replacement
        kw["game_ids"][:] = ["tampered"] * len(a)
        kw["row_ids"][:] = ["tampered"] * len(a)
        return np.tile([.75, .125, .125], (len(a), 1))
    kw["candidate"] = mutate_external
    assert compare(x, target, **kw) == expected
