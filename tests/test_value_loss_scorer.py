"""Tests for the losing-position value-error scorer.

Each test names the mutation it catches. The standing bias for this file: the
defect to catch is not an arithmetic slip, it is a number that stops meaning
what its name says — a stratum secretly chosen by the net, a CI that ignores
the game clustering, or a negative control that cannot fail.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from chess_anti_engine.eval.value_optimism import cluster_bootstrap_ci
from scripts.value_loss_scorer import (
    NO_OUTCOME,
    QUANTILES,
    RowSet,
    _synthetic_rows,
    build_report,
    load_rows_from_npz,
    main,
    select_oracle_lost,
    shuffle_model_weights,
    to_probs,
    write_dump,
)

SLOPE = 0.01
DRAW_WIDTH = 60.0


def _rows(n_games: int = 24, plies: int = 10) -> RowSet:
    return _synthetic_rows(n_games=n_games, plies=plies, seed=5)


def test_tool_self_test_passes() -> None:
    """The shipped negative controls must actually run and pass.

    MUTATION: any of the six self-test checks breaks -> ``main`` returns 1.
    This is what makes "the control ships as a test" true rather than a claim
    in a docstring.
    """
    assert main(["--self-test"]) == 0


def test_selection_cannot_see_the_net() -> None:
    """THE structural rule: the stratum filter takes no model-derived input.

    MUTATION: add a ``net_probs``/``q`` parameter to ``select_oracle_lost`` and
    let any criterion read it. The signature check below fails immediately,
    before the arithmetic gets a chance to look reasonable.

    Conditioning the denominator on the thing under test is the failure this
    guards: the resulting "error rate among positions the net thinks it is
    losing" is uncomparable to anything, including itself at another
    checkpoint.
    """
    params = set(inspect.signature(select_oracle_lost).parameters)
    assert params == {
        "p_sf", "loss_prob_min", "sf_cp_max", "slope", "draw_width_cp",
    }, params

    rows = _rows()
    mask = select_oracle_lost(
        rows.p_sf, loss_prob_min=0.75, sf_cp_max=None,
        slope=SLOPE, draw_width_cp=DRAW_WIDTH,
    )
    assert 0 < int(mask.sum()) < len(rows)
    # Every selected row genuinely clears the bar, and no unselected one does.
    assert np.all(rows.p_sf[mask, 2] >= 0.75)
    assert np.all(rows.p_sf[~mask, 2] < 0.75)


def test_selection_threshold_binds_in_both_directions() -> None:
    """MUTATION: use ``>`` where the docstring says "at or above", or compare
    against ``p_sf[:, 0]``. Both change which rows are in the denominator."""
    p_sf = np.array([
        [0.00, 0.00, 1.00],
        [0.20, 0.05, 0.75],  # exactly at the bar -> IN
        [0.30, 0.00, 0.70],  # below -> OUT
        [1.00, 0.00, 0.00],
    ])
    mask = select_oracle_lost(
        p_sf, loss_prob_min=0.75, sf_cp_max=None,
        slope=SLOPE, draw_width_cp=DRAW_WIDTH,
    )
    assert mask.tolist() == [True, True, False, False]


def test_cp_bar_narrows_the_stratum_and_ands_with_the_prob_bar() -> None:
    """MUTATION: OR the two criteria instead of ANDing them, or ignore
    ``sf_cp_max`` entirely -- the stratum silently widens."""
    rows = _rows()
    kw = {"loss_prob_min": 0.75, "slope": SLOPE, "draw_width_cp": DRAW_WIDTH}
    wide = select_oracle_lost(rows.p_sf, sf_cp_max=None, **kw)
    narrow = select_oracle_lost(rows.p_sf, sf_cp_max=-300.0, **kw)
    assert int(narrow.sum()) < int(wide.sum())
    assert np.all(wide[narrow]), "the cp bar must only ever remove rows"


def test_ci_resamples_games_not_rows() -> None:
    """MUTATION: pass row indices as ``game_id`` (i.e. drop the clustering).

    Rows inside one game are consecutive plies of one position sequence. The
    scorer's CIs are only honest if the cluster is the game; a row-level
    bootstrap on the same values reports a materially tighter interval, and a
    tighter interval is how a null result gets published as a finding.
    """
    rng = np.random.default_rng(0)
    n_games, plies = 20, 25
    per_game = rng.normal(0.0, 1.0, size=n_games)
    values = np.repeat(per_game, plies) + rng.normal(0.0, 0.05, size=n_games * plies)
    gid = np.repeat(np.arange(n_games), plies)

    clustered = cluster_bootstrap_ci(
        values, gid, n_boot=2000, rng=np.random.default_rng(1),
    )
    row_level = cluster_bootstrap_ci(
        values, np.arange(values.size), n_boot=2000, rng=np.random.default_rng(1),
    )
    width = lambda ci: ci[1] - ci[0]  # noqa: E731
    assert width(clustered) > 3.0 * width(row_level), (clustered, row_level)


def test_contrast_null_is_zero_only_for_the_control_column() -> None:
    """The lesson this instrument is built around, as an assertion.

    Under a destroyed position<->prediction association ONLY ``net_score`` has
    a null of zero. ``d_score`` is a difference against a reference that varies
    by stratum, so its shuffle null is large and positive -- reading it against
    zero calls a destroyed association a finding.

    MUTATION: point the negative control at ``d_score`` (i.e. assert it too
    contains zero here) and the test fails, which is the point.
    """
    rows = _rows(n_games=40, plies=12)
    mask = select_oracle_lost(
        rows.p_sf, loss_prob_min=0.75, sf_cp_max=None,
        slope=SLOPE, draw_width_cp=DRAW_WIDTH,
    )
    q = rows.p_sf.copy()
    q[mask, 0] += 0.2
    q[mask, 2] -= 0.2
    q = np.clip(q, 1e-6, None)
    q /= q.sum(axis=1, keepdims=True)

    perm = np.random.default_rng(4).permutation(len(rows))
    report, _ = build_report(
        rows, q[perm], loss_prob_min=0.75, sf_cp_max=None,
        slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=600, seed=0,
    )
    by_col = {c.column: c for c in report.contrasts}
    assert not by_col["net_score"].excludes_zero, by_col["net_score"]
    assert by_col["d_score"].excludes_zero, by_col["d_score"]
    assert by_col["d_score"].delta > 0.3


def test_shuffle_model_weights_permutes_and_reports_how_many() -> None:
    """MUTATION: return a constant, or skip tensors so nothing moves.

    A control that silently permutes nothing is a control that cannot fail;
    the CLI refuses to run when the count comes back below 2.
    """
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 3))
    before = [p.detach().clone() for p in net.parameters()]
    touched = shuffle_model_weights(net, seed=3)
    after = list(net.parameters())

    assert touched == sum(1 for p in before if p.numel() >= 2)
    changed = [i for i, (a, b) in enumerate(zip(before, after))
               if not torch.equal(a, b.detach())]
    assert len(changed) >= 2
    # A permutation preserves the multiset of values in every tensor.
    for a, b in zip(before, after):
        assert torch.allclose(
            torch.sort(a.reshape(-1)).values, torch.sort(b.detach().reshape(-1)).values,
        )


def test_to_probs_detects_logits_and_probabilities() -> None:
    """MUTATION: assume one convention. A logit head read as probabilities (or
    the reverse) silently reports a different head."""
    probs = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    assert np.allclose(to_probs(probs), probs)

    # Exact softmax values, not just "normalised and correctly ranked": a
    # clamp-then-renormalise read of these logits also sums to 1 and ranks the
    # same way, so a shape-only assertion cannot tell the two apart.
    logits = np.array([[2.0, 0.0, -1.0], [-3.0, 1.0, 4.0]])
    expected = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    out = to_probs(logits)
    assert np.allclose(out, expected)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.argmax(out, axis=1).tolist() == [0, 2]
    # ...and the clamp-then-renormalise reading is genuinely different here.
    clamped = np.clip(logits, 0.0, None)
    clamped = clamped / clamped.sum(axis=1, keepdims=True)
    assert not np.allclose(out, clamped)

    with pytest.raises(ValueError, match="expected"):
        to_probs(np.zeros((4, 2)))


def test_npz_row_source_refuses_a_dump_without_planes(tmp_path: Path) -> None:
    """MUTATION: default ``x`` to zeros when absent.

    A dump with no planes cannot be re-scored; silently substituting zeros
    would report a NEW checkpoint's name over the OLD checkpoint's numbers.
    """
    path = tmp_path / "nox.npz"
    np.savez_compressed(path, p_sf=np.full((3, 3), 1 / 3, np.float32))
    with pytest.raises(SystemExit, match="re-dump with --dump-x"):
        load_rows_from_npz(str(path))


def test_dump_round_trips_as_a_row_source(tmp_path: Path) -> None:
    """MUTATION: drop a column from ``write_dump``, or write ``x`` only under a
    flag the loader does not require -- the banked file stops being re-usable."""
    rows = _rows(n_games=6, plies=5)
    q = rows.p_sf.copy()
    mask = select_oracle_lost(
        rows.p_sf, loss_prob_min=0.75, sf_cp_max=None,
        slope=SLOPE, draw_width_cp=DRAW_WIDTH,
    )
    path = tmp_path / "dump.npz"
    write_dump(str(path), rows, q, mask, with_x=True, meta={"k": 1})

    again = load_rows_from_npz(str(path))
    assert len(again) == len(rows)
    assert np.allclose(again.x, rows.x)
    assert np.allclose(again.p_sf, rows.p_sf)
    assert np.array_equal(again.game_id, rows.game_id)
    assert np.array_equal(again.outcome, rows.outcome)
    meta = json.loads(Path(str(path) + ".meta.json").read_text("utf-8"))
    assert meta["k"] == 1


def test_rows_without_an_outcome_are_excluded_from_outcome_columns_only() -> None:
    """The only rows dropped from a column are dropped by a POSITION-level
    fact (this row carries no result), never by anything the net emits.

    MUTATION: drop the whole row when the outcome is missing -- the sf-relative
    columns would then be computed on a different, outcome-selected sample.
    """
    rows = _rows(n_games=8, plies=6)
    half = np.zeros(len(rows), bool)
    half[::2] = True
    outcome = rows.outcome.copy()
    outcome[half] = NO_OUTCOME
    partial = RowSet(
        x=rows.x, p_sf=rows.p_sf, outcome=outcome, game_id=rows.game_id,
        ply=rows.ply, is_selfplay=rows.is_selfplay, source="t",
    )
    report, _ = build_report(
        partial, rows.p_sf.copy(), loss_prob_min=0.75, sf_cp_max=None,
        slope=SLOPE, draw_width_cp=DRAW_WIDTH, n_boot=200, seed=0,
    )
    total = report.lost.n + report.rest.n
    assert total == len(rows)
    assert report.lost.n_outcome < report.lost.n
    # THE assertion: the sf-relative columns keep EVERY eligible row, and only
    # the outcome columns shrink. Without the per-column denominator this test
    # could not tell "computed on all rows" from "computed on the outcome-
    # carrying half", since both produce a finite mean.
    for st in (report.lost, report.rest):
        assert st.n_used["e2_sf"] == st.n
        assert st.n_used["d_score"] == st.n
        assert st.n_used["net_score"] == st.n
        assert st.n_used["e2_out"] == st.n_outcome
        assert st.n_used["d_score_out"] == st.n_outcome
        assert 0 < st.n_used["e2_out"] < st.n
    assert np.isfinite(report.lost.means["e2_sf"])
    assert np.isfinite(report.lost.means["e2_out"])
    assert len(report.lost.quantiles["e2_out"]) == len(QUANTILES)
