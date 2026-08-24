"""An out-of-band search knob must be REFUSED, never recorded as realized.

``policy_temp`` has a semantic band (``[POLICY_TEMP_MIN, POLICY_TEMP_MAX]``,
endpoints inclusive) that ``policy_temp_active`` -- THE definition of "tempering
is on" -- reads as OFF outside. The yaml loader rejects out-of-band values
(``trial_config._policy_temperature``); the CLI surfaces did not, because they
reach ``dataclasses.replace`` without passing it. So
``arena_standard.py --cand-gumbel policy_temp=1e300`` ran an UNTEMPERED search
and banked ``policy_temp: 1e300`` into the JSONL as that side's realized
setting, which makes a sweep over out-of-band temperatures a set of IDENTICAL
arms recorded as different ones -- the c_puct Swiss (play-path audit 2026-08-03
F2) with a live knob instead of a dead one.

Every test here drives the REAL entry point (``apply_search_overrides``,
``build_profile_search_shape``, ``build_gumbel_config_from_args``,
``_volatility_kwargs_from_args``), not a reimplementation of it: a guard proven
only on the helper is a guard whose INVOCATION nothing covers.

Mutants run against this file (each reverted after):
  * drop the ``validate_gumbel_config`` call from ``SideSearch.__post_init__``
  * drop the finiteness loop from ``validate_gumbel_config``
  * drop its ``policy_temp`` clause / its ``halving_div`` clause
  * make the ``policy_temp`` clause refuse 1.0 as well (over-refusal)
  * drop the call from ``_refuse_dead_overrides`` /
    ``build_profile_search_shape`` / ``build_gumbel_config_from_args`` /
    ``_volatility_kwargs_from_args``
Each killed at least one test below; the matrix is in the PR body.
"""
from __future__ import annotations

import argparse
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel import (
    MIN_HALVING_DIV,
    PLAY_SEARCH_DEFAULTS,
    POLICY_TEMP_MAX,
    POLICY_TEMP_MIN,
    GumbelConfig,
    apply_policy_temp,
    policy_temp_active,
    validate_gumbel_config,
)

# ⚑ Read off `mcts.gumbel`, never written as literals here. A second copy of
# 0.05/20.0 in this file would keep passing the day the band moves, which is
# how a guard stops sharing the criterion's instrument.
_DEAD_TEMPS = (
    0.0, -1.0, POLICY_TEMP_MIN / 2.0, POLICY_TEMP_MAX * 2.0, 1e300,
    float("inf"), float("nan"),
)
_LIVE_TEMPS = (POLICY_TEMP_MIN, POLICY_TEMP_MAX, 1.0, 2.0, 1.5)


# --- the validator itself ----------------------------------------------------


def test_the_three_shipped_shapes_pass() -> None:
    """A band edit must not be able to strand production or the arena.

    The default ``GumbelConfig`` (the RL/selfplay shape by construction), the
    PLAY/EVAL shape, and the production TRAINING shape the arena resolves from
    the config all have to validate, or the guard is a way to break the run.
    """
    from scripts.arena_standard import resolve_search_shape

    validate_gumbel_config(GumbelConfig(), where="default")
    validate_gumbel_config(
        dataclasses.replace(GumbelConfig(), **PLAY_SEARCH_DEFAULTS), where="play",
    )
    training = resolve_search_shape("training").gumbel
    validate_gumbel_config(
        dataclasses.replace(GumbelConfig(), **training), where="training",
    )


@pytest.mark.parametrize("value", _LIVE_TEMPS)
def test_a_policy_temp_the_search_reads_is_accepted(value: float) -> None:
    """NEGATIVE CONTROL. Without it, "refuse everything" passes every refusal
    test below and the whole ``policy_temp`` surface dies silently.

    The acceptance is checked against the ARITHMETIC, not against the predicate
    the validator consults: 1.0 is the documented identity and must stay
    requestable, everything else here must really move the priors.
    """
    cfg = dataclasses.replace(GumbelConfig(), policy_temp=value)
    validate_gumbel_config(cfg, where="test")

    pol = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    tempered = apply_policy_temp(pol, cfg=cfg)
    if value == 1.0:
        assert np.array_equal(tempered, pol)
    else:
        assert not np.array_equal(tempered, pol), (
            f"policy_temp={value} was accepted but is a no-op"
        )


@pytest.mark.parametrize("value", _DEAD_TEMPS)
def test_a_policy_temp_the_search_will_not_read_is_refused(value: float) -> None:
    """...and the reason the refusal is right, from the arithmetic.

    `apply_policy_temp` returns the priors UNTOUCHED for every value here, so
    accepting one produces a certified-realized null.
    """
    cfg = dataclasses.replace(GumbelConfig(), policy_temp=value)
    with pytest.raises(ValueError, match="outside the band"):
        validate_gumbel_config(cfg, where="test")

    pol = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    assert np.array_equal(apply_policy_temp(pol, cfg=cfg), pol), (
        f"policy_temp={value} is NOT a no-op, so refusing it is wrong"
    )


@pytest.mark.parametrize(
    "field", ["c_scale", "c_visit", "q_visit_exp_root", "volatility_q_scale"],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_knob_is_refused(field: str, value: float) -> None:
    """Not hygiene: every sentinel in ``gumbel.py`` is a COMPARISON.

    ``nan`` is neither ``< 90`` nor ``>= 90``, so ``q_visit_exp_root=nan``
    silently reverts the root transform to ``q_visit_exp`` -- the fallback arm,
    taken without a word, under a record naming the operator's value.
    """
    cfg = dataclasses.replace(GumbelConfig(), **{field: value})
    with pytest.raises(ValueError, match=field):
        validate_gumbel_config(cfg, where="test")


def test_the_sentinel_arm_a_non_finite_value_takes_is_the_silent_one() -> None:
    """The premise of the test above, MEASURED on the real transform.

    ``c_scale_root=nan`` fails ``>= 0.0``, so ``_root_sigma_scale`` -- the
    Python mirror of the C root site -- takes the "root-only knob unset"
    fallback and returns exactly the scale the SENTINEL would have produced.
    Not an error, not a nan: a different search, silently.
    """
    from chess_anti_engine.mcts.gumbel import _root_sigma_scale

    sentinel = dataclasses.replace(GumbelConfig(), c_scale_root=-1.0, c_scale=0.1)
    poisoned = dataclasses.replace(
        GumbelConfig(), c_scale_root=float("nan"), c_scale=0.1,
    )
    scale = _root_sigma_scale(max_visit=60, cfg=sentinel)
    assert _root_sigma_scale(max_visit=60, cfg=poisoned) == pytest.approx(scale)
    assert not policy_temp_active(float("nan"))


@pytest.mark.parametrize("value", [1, 0, -3])
def test_a_halving_div_the_c_would_clamp_is_refused(value: int) -> None:
    """``_mcts_tree.c:3978``: ``g->halving_div = (halving_div >= 2) ? d : 2``.

    So ``halving_div=1`` runs STANDARD halving and every record of the run says
    it did not halve. Same defect as the temperature, different knob.
    """
    cfg = dataclasses.replace(GumbelConfig(), halving_div=value)
    with pytest.raises(ValueError, match="halving_div"):
        validate_gumbel_config(cfg, where="test")


def test_the_c_clamp_this_refusal_is_about_is_still_there() -> None:
    """Pin the C line the refusal quotes, so a future C change cannot leave the
    Python refusing a value the search would have honoured."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "chess_anti_engine" / "mcts" / "_mcts_tree.c"
    ).read_text(encoding="utf-8")
    assert f"(halving_div >= {MIN_HALVING_DIV}) ? halving_div : {MIN_HALVING_DIV}" in src


def test_the_validator_reports_every_offending_knob_not_just_the_first() -> None:
    """An operator fixing one knob and re-running for the next is how a sweep
    loses an afternoon."""
    cfg = dataclasses.replace(
        GumbelConfig(), policy_temp=0.0, c_scale=float("nan"), halving_div=1,
    )
    with pytest.raises(ValueError, match="policy_temp") as exc:
        validate_gumbel_config(cfg, where="test")
    message = str(exc.value)
    assert "policy_temp" in message
    assert "c_scale" in message
    assert "halving_div" in message


# --- the arena CLI, through the function the CLI itself calls ----------------


def _play_side():
    from scripts.arena_standard import resolve_search_shape

    return resolve_search_shape("play")


def test_the_arena_override_path_refuses_an_out_of_band_policy_temp() -> None:
    """THE finding: ``--cand-gumbel policy_temp=1e300``.

    Driven through ``apply_search_overrides``, the function ``main()`` parses
    ``--cand-gumbel`` / ``--ref-gumbel`` with, before any checkpoint is loaded.
    """
    from scripts.arena_standard import apply_search_overrides

    with pytest.raises(SystemExit) as exc:
        apply_search_overrides(_play_side(), spec="policy_temp=1e300")
    message = str(exc.value)
    assert "policy_temp" in message
    assert "1e+300" in message
    assert str(POLICY_TEMP_MIN) in message
    assert str(POLICY_TEMP_MAX) in message
    assert "UNTEMPERED" in message


@pytest.mark.parametrize("spec", ["policy_temp=nan", "c_scale=inf", "halving_div=1"])
def test_the_arena_override_path_refuses_the_rest_of_the_dead_set(spec: str) -> None:
    from scripts.arena_standard import apply_search_overrides

    with pytest.raises(SystemExit):
        apply_search_overrides(_play_side(), spec=spec)


@pytest.mark.parametrize("value", _LIVE_TEMPS)
def test_the_arena_override_path_still_accepts_the_band(value: float) -> None:
    """NEGATIVE CONTROL for the two tests above, including both band ENDS."""
    from scripts.arena_standard import apply_search_overrides

    side = apply_search_overrides(_play_side(), spec=f"policy_temp={value!r}")
    assert side.realized_gumbel()["policy_temp"] == pytest.approx(value)


def test_the_check_is_on_the_side_constructor_not_the_override_parser() -> None:
    """Placement, pinned. EVERY side goes through this constructor.

    ``apply_search_overrides`` would have been the smaller diff and would have
    covered only the two call sites in ``main()``: a resolved shape, or a
    programmatic caller (``scripts/elo_vs_sims.py``), would have walked past it.
    """
    from scripts.arena_standard import SideSearch

    with pytest.raises(SystemExit, match="policy_temp"):
        SideSearch(
            shape="training", source="not the CLI",
            gumbel={"policy_temp": 1e300}, vloss_weight=1, target_batch=0,
        )


def test_what_the_record_would_have_said_without_the_refusal() -> None:
    """WHY refusing beats letting the hot path no-op it.

    Built the way ``apply_search_overrides`` used to, bypassing the constructor
    check: ``realized_gumbel()`` -- the dict ``as_record()`` banks into the
    JSONL and ``describe()`` prints at startup -- reports 1e300 as this side's
    REALIZED search, while the search it describes runs at T=1.0. Two arms of a
    sweep over 1e300 and 1e-300 are then bit-identical searches banked as
    different settings.
    """
    from scripts.arena_standard import SideSearch

    side = SideSearch.__new__(SideSearch)
    for key, value in {
        "shape": "play", "source": "counterfactual", "gumbel": {"policy_temp": 1e300},
        "vloss_weight": 3, "target_batch": 0, "tree_reuse": "cold",
    }.items():
        object.__setattr__(side, key, value)

    assert side.realized_gumbel()["policy_temp"] == 1e300
    assert side.as_record()["gumbel"]["policy_temp"] == 1e300
    assert not policy_temp_active(1e300)
    pol = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    assert np.array_equal(
        apply_policy_temp(pol, cfg=GumbelConfig(policy_temp=1e300)), pol,
    )


def test_the_arena_volatility_flags_are_validated_too() -> None:
    """``--volatility-*`` are GumbelConfig fields that ride as kwargs.

    They never enter ``SideSearch.gumbel``, so the constructor check cannot see
    them -- and ``nan`` reads as ENABLED there (``nan != 0.0``), forcing the
    Python path with a nan sigma, banked into the record as
    ``volatility_candidate``.
    """
    from scripts.arena_standard import _volatility_kwargs_from_args

    def _args(**kw) -> SimpleNamespace:
        base = {
            "volatility_q_scale": 0.0, "volatility_fpu": 0.0,
            "volatility_anchor": None, "mode": "matched_sims",
        }
        base.update(kw)
        return SimpleNamespace(**base)

    assert _volatility_kwargs_from_args(_args()) is None
    ok = _volatility_kwargs_from_args(_args(volatility_q_scale=0.5))
    assert ok is not None
    assert ok["volatility_q_scale"] == pytest.approx(0.5)
    with pytest.raises(SystemExit, match="volatility_q_scale"):
        _volatility_kwargs_from_args(_args(volatility_q_scale=float("nan")))


# --- the other externally-built search configs -------------------------------


def test_audit_targets_policy_temp_flag_is_refused_like_its_sibling() -> None:
    """``--policy-temp`` reaches a ``GumbelConfig`` outside
    ``parse_gumbel_overrides``, so the parse-time refusal never covered it: the
    same script refused ``--gumbel policy_temp=0`` out loud and ran
    ``--policy-temp 0`` untempered under a header printing 0.
    """
    import scripts.audit_targets as at
    from chess_anti_engine.eval.production_shape import production_input_encoding
    from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file
    from scripts.arena_standard import IN_TREE_CONFIG

    flat = flatten_run_config_defaults(load_yaml_file(IN_TREE_CONFIG))
    enc = production_input_encoding(flat)
    profiles = at.build_search_profiles(flat, play_sims=32, play_topk=None)

    def _build(temp: float):
      # Production's own encoding: those four fields are exempt from the shape
      # check, so foreign values would let the exempt map do the work.
        return at.build_profile_search_shape(
            "search", profiles["search"],
            hist=str(enc["input_history_encoding"]),
            extra=str(enc["input_extra_features"]),
            pol_enc="lc0_1858", use_rel=False, play_policy_temp=temp,
        )

    # Positive control: the shipped default builds a shape.
    assert _build(1.0).cfg.policy_temp == pytest.approx(1.0)
    with pytest.raises(SystemExit, match="policy_temp"):
        _build(1e300)


def test_audit_targets_gumbel_overrides_refuse_a_non_finite_value() -> None:
    """The pre-flight parser widened from ``policy_temp``-only to the whole
    validator: ``--gumbel c_scale=nan`` used to parse."""
    import scripts.audit_targets as at

    assert at.parse_gumbel_overrides(["topk=8", "c_scale=0.5"]) == (
        ("topk", 8.0), ("c_scale", 0.5),
    )
    with pytest.raises(SystemExit, match="c_scale"):
        at.parse_gumbel_overrides(["c_scale=nan"])


def test_eval_puzzles_gumbel_flags_are_refused() -> None:
    """``--gumbel-halving-div 1`` measures standard halving and logs 1."""
    from scripts.eval_puzzles import build_gumbel_config_from_args

    def _args(**kw) -> argparse.Namespace:
        base = {
            "gumbel_topk": 32, "gumbel_c_scale": 0.025, "gumbel_c_visit": 50.0,
            "gumbel_qexp": 1.0, "gumbel_global_scale": False,
            "gumbel_qfloor": -1.0, "gumbel_halving_div": 2,
            "gumbel_cvisit_root": 900.0, "gumbel_cscale_root": 7.0,
            "gumbel_qexp_root": -1.0,
        }
        base.update(kw)
        return argparse.Namespace(**base)

    assert build_gumbel_config_from_args(_args()).halving_div == 2
    with pytest.raises(SystemExit, match="halving_div"):
        build_gumbel_config_from_args(_args(gumbel_halving_div=1))
    with pytest.raises(SystemExit, match="c_scale"):
        build_gumbel_config_from_args(_args(gumbel_c_scale=float("nan")))
