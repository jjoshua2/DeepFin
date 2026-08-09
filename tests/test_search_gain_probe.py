"""Tests for the search-gain probe's shape arithmetic.

The probe's whole value rests on one thing: that the `SearchShape` it prints in
its header is the search the number came from. Two ways that can silently go
wrong, and one test for each:

  * the probe reimplements the sigma(q) resolution and drifts from the real one
    (`gumbel._root_sigma_scale`), so its "realized q_scale" column describes a
    search nobody runs;
  * the `live_selfplay` shape stops matching what `selfplay/network_turn.py`
    actually builds, so a "production" number is measured on something else.

The second is the one this codebase keeps producing -- a value accepted and
then silently ignored -- so it is asserted against the production call site's
own source rather than against a copy of the constants.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from chess_anti_engine.mcts.gumbel import (
    PLAY_SEARCH_DEFAULTS,
    GumbelConfig,
    _root_sigma_scale,
)
from scripts.search_gain_probe import SHAPES, SearchShape, apply_overrides


def _cfg_for(shape: SearchShape) -> GumbelConfig:
    return GumbelConfig(
        topk=shape.topk,
        c_scale=shape.c_scale,
        c_visit=shape.c_visit,
        c_visit_root=shape.c_visit_root,
        c_scale_root=shape.c_scale_root,
        q_visit_exp=shape.q_visit_exp,
        q_visit_exp_root=shape.q_visit_exp_root,
        halving_div=shape.halving_div,
        policy_temp=shape.policy_temp,
    )


@pytest.mark.parametrize("name", sorted(SHAPES))
@pytest.mark.parametrize("max_visit", [0, 1, 5, 57, 118, 1559])
def test_root_q_scale_matches_the_real_transform(name: str, max_visit: int) -> None:
    """The probe's q_scale must equal what the search itself would compute.

    Differential, not a golden number: if `_root_sigma_scale` changes, this
    fails rather than silently reporting a stale formula.
    """
    shape = SHAPES[name]
    assert shape.root_q_scale(float(max_visit)) == pytest.approx(
        _root_sigma_scale(max_visit=max_visit, cfg=_cfg_for(shape)), rel=1e-12,
    )


def test_live_selfplay_shape_is_a_linear_root_and_play_is_log() -> None:
    """The two shapes must differ in the root transform, not just in constants.

    This is the fact the probe exists to measure; if the fixtures ever agree,
    every root-transform contrast it prints becomes a null by construction.
    """
    assert SHAPES["live_selfplay"].root_transform_name() == "LINEAR"
    assert SHAPES["play"].root_transform_name() == "LOG"
    assert SHAPES["live_selfplay_logroot"].root_transform_name() == "LOG"
    assert SHAPES["live_selfplay_sqrtroot"].root_transform_name() == "POWER(0.5)"
    # And the gap has to be an order of magnitude at the production budget, or
    # the headline contrast is noise. max_visit 57 is the measured root
    # max-visit at 256 sims / topk 32 / halving_div 2.
    lin = SHAPES["live_selfplay"].root_q_scale(57.0)
    log = SHAPES["play"].root_q_scale(57.0)
    assert log > 10.0 * lin


def test_play_shape_tracks_PLAY_SEARCH_DEFAULTS() -> None:
    """The play fixture must not drift from the tuned constants it names."""
    play = SHAPES["play"]
    for key, value in PLAY_SEARCH_DEFAULTS.items():
        assert getattr(play, key) == value, key


def _omitted_gumbel_kwargs() -> set[str]:
    """Fields `network_turn.py` does NOT pass when it builds its GumbelConfig.

    Parsed from the production source, so the day someone plumbs a knob through
    to selfplay this test fails and the probe's fixture gets updated with it,
    instead of the probe quietly measuring the old shape.
    """
    from chess_anti_engine.selfplay import network_turn

    src = inspect.getsource(network_turn)
    tree = ast.parse(src)
    passed: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "GumbelConfig"
        ):
            passed |= {kw.arg for kw in node.keywords if kw.arg}
    assert passed, "no GumbelConfig(...) call found in network_turn.py"
    fields = set(GumbelConfig.__dataclass_fields__)
    return fields - passed


def test_selfplay_leaves_the_root_transform_at_its_sentinels() -> None:
    """Production selfplay cannot select a root transform, and the fixture says so.

    `network_turn.py` builds GumbelConfig from an explicit keyword list. The
    three root-transform fields are not in it and there is no yaml key for them
    either, so selfplay always runs the sentinel fall-back -- the LINEAR root.
    The `live_selfplay` fixture must encode exactly that.
    """
    omitted = _omitted_gumbel_kwargs()
    for field in ("c_visit", "c_visit_root", "c_scale_root", "q_visit_exp_root", "halving_div"):
        assert field in omitted, f"{field} is now plumbed into selfplay; update SHAPES"

    live = SHAPES["live_selfplay"]
    defaults = GumbelConfig()
    assert live.c_visit == defaults.c_visit
    assert live.c_visit_root == defaults.c_visit_root
    assert live.c_scale_root == defaults.c_scale_root
    assert live.q_visit_exp_root == defaults.q_visit_exp_root
    assert live.halving_div == defaults.halving_div


def test_no_yaml_can_set_the_root_transform() -> None:
    """No config file names the root-transform knobs.

    If one ever does, the "selfplay runs a linear root" claim stops being
    unconditional and this probe's shape fixtures need a config reader.
    """
    root = Path(__file__).resolve().parents[1] / "configs"
    hits = [
        p.name
        for p in sorted(root.glob("*.yaml"))
        if any(k in p.read_text(encoding="utf-8")
               for k in ("c_scale_root", "c_visit_root", "q_visit_exp_root"))
    ]
    assert hits == [], f"configs now set the root transform: {hits}"


def test_apply_overrides_rejects_unknown_and_records_provenance() -> None:
    base = SHAPES["live_selfplay"]
    out = apply_overrides(base, "c_scale_root=3.0,topk=16")
    assert out.c_scale_root == 3.0
    assert out.topk == 16
    assert isinstance(out.topk, int)
    assert "3.0" in out.provenance
    assert "OVERRIDDEN" in out.provenance
    with pytest.raises(SystemExit):
        apply_overrides(base, "not_a_field=1")
    with pytest.raises(SystemExit):
        apply_overrides(base, "label=sneaky")
    assert apply_overrides(base, None) is base


def test_spearman_averages_ties_so_a_flat_block_cannot_invent_an_order() -> None:
    """Unvisited root children all share the mixed value.

    Ranking them by argsort order would manufacture a correlation out of array
    position, which is exactly the kind of statistic that cannot fail.
    """
    import numpy as np

    from scripts.search_gain_probe import _spearman

    flat = np.zeros(8)
    assert np.isnan(_spearman(flat, np.arange(8.0)))

    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert _spearman(a, a) == pytest.approx(1.0)
    assert _spearman(a, -a) == pytest.approx(-1.0)

    # Half the vector tied: the tied block must contribute no ordering.
    tied = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    forward = _spearman(tied, np.arange(8.0))
    backward = _spearman(tied[::-1].copy(), np.arange(8.0))
    assert forward == pytest.approx(-backward)
