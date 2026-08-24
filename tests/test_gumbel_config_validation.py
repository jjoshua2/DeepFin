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

The same defect turned out to be live on three more knobs, all covered here:
``topk`` (silently raised to 2 in PYTHON on both search paths -- it never
reaches the C at all), a fractional ``halving_div``/``topk`` (truncated by the
``int()`` every consumer applies), and ``--volatility-*`` under
``--search-shape training`` (banked into the record, then overwritten by the
shape's own zeros before the search sees it).

Every test here drives the REAL entry point (``resolve_sides_from_args``,
``apply_search_overrides``, ``build_profile_search_shape``,
``build_gumbel_config_from_args``, ``build_probe_gumbel_config``,
``_build_shape``), not a reimplementation of it: a guard proven only on the
helper is a guard whose INVOCATION nothing covers -- which is exactly how the
training-row call in ``build_profile_search_shape`` survived a full suite run
with the call deleted.

The mutation matrix (16 mutants, each applied, run, reverted) is in the PR body.
"""
from __future__ import annotations

import argparse
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel import (
    MIN_HALVING_DIV,
    MIN_TOPK,
    PLAY_SEARCH_DEFAULTS,
    POLICY_TEMP_MAX,
    POLICY_TEMP_MIN,
    GumbelConfig,
    apply_policy_temp,
    halving_keep_count,
    halving_rounds_left,
    halving_visits_per_action,
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
    """``g->halving_div = (halving_div >= 2) ? halving_div : 2`` (_mcts_tree.c).

    So ``halving_div=1`` runs STANDARD halving and every record of the run says
    it did not halve. Same defect as the temperature, different knob.
    """
    cfg = dataclasses.replace(GumbelConfig(), halving_div=value)
    with pytest.raises(ValueError, match="halving_div"):
        validate_gumbel_config(cfg, where="test")


@pytest.mark.parametrize("value", [1, 0, -5])
def test_a_topk_the_python_would_clamp_is_refused(value: int) -> None:
    """⚑ ``topk`` NEVER REACHES THE C -- it is clamped in Python, on both paths.

    An earlier revision of ``validate_gumbel_config`` excluded ``topk`` on the
    stated ground that "the C signature rejects those downstream, loudly". It
    does not: the C is handed the already-chosen candidate LIST and the string
    ``topk`` does not occur in ``_mcts_tree.c`` at all. Both Python selection
    sites do ``m = max(2, ...)``, so 0 / 1 / -5 every one of them runs a
    2-candidate root -- and ``topk`` is in ``PLAY_SEARCH_DEFAULTS``, so the
    arena banks it on every single row.
    """
    cfg = dataclasses.replace(GumbelConfig(), topk=value)
    with pytest.raises(ValueError, match="topk"):
        validate_gumbel_config(cfg, where="test")


def test_the_clamps_these_refusals_are_about_are_still_there() -> None:
    """Pin the CODE each refusal quotes (not its line number), so a future
    change cannot leave Python refusing a value the search would have honoured.

    The ``topk`` half also pins the claim that made the original exclusion
    wrong: the C never sees the knob.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "chess_anti_engine" / "mcts"
    c_src = (root / "_mcts_tree.c").read_text(encoding="utf-8")
    assert (
        f"(halving_div >= {MIN_HALVING_DIV}) ? halving_div : {MIN_HALVING_DIV}" in c_src
    )
    assert "topk" not in c_src, (
        "topk now appears in the C; re-derive where it is clamped before "
        "trusting the MIN_TOPK message"
    )
    c_path = (root / "gumbel_c.py").read_text(encoding="utf-8")
    assert f"m = max({MIN_TOPK}, m)" in c_path
    py_path = (root / "gumbel.py").read_text(encoding="utf-8")
    assert f"m = max({MIN_TOPK}, int(min(int(topk)" in py_path


@pytest.mark.parametrize(("field", "value"), [
    ("halving_div", 2.9), ("halving_div", 4.5), ("topk", 16.5),
])
def test_a_fractional_int_knob_is_refused(field: str, value: float) -> None:
    """Every consumer applies ``int()``, which TRUNCATES: ``halving_div=2.9``
    runs 2 and ``topk=16.5`` runs 16, under a record naming the fraction.

    Reachable because ``audit_targets`` carries every override as a float.
    """
    cfg = dataclasses.replace(GumbelConfig(), **{field: value})
    with pytest.raises(ValueError, match=field) as exc:
        validate_gumbel_config(cfg, where="test")
    assert "int()" in str(exc.value)


# --- the Python search path must HONOUR halving_div, not just accept it ------


def test_the_shared_halving_arithmetic_is_bit_identical_at_the_default() -> None:
    """div=2 must reproduce the hardcoded formulas the Python path used to run.

    Exhaustive over the candidate counts a root can have and a spread of
    budgets, against the OLD expressions written out literally here
    (``ceil(log2(n))`` rounds, ``max(1, (n+1)//2)`` survivors, ``budget //
    (n*rounds)`` visits). This is the "default stays bit-identical" guarantee
    for moving the reference search onto the shared helpers.
    """
    for n in range(1, 129):
        assert halving_keep_count(n, MIN_HALVING_DIV) == max(1, (n + 1) // 2)
        if n <= 1:
            assert halving_visits_per_action(n, 37, MIN_HALVING_DIV) == 37
            continue
        old_rounds = int(np.ceil(np.log2(n)))
        assert halving_rounds_left(n, MIN_HALVING_DIV) == old_rounds
        for budget in (1, 7, 32, 256, 2048, 100_000):
            assert halving_visits_per_action(n, budget, MIN_HALVING_DIV) == max(
                1, int(budget // max(1, n * old_rounds)),
            )


def test_a_bigger_divisor_really_changes_the_schedule() -> None:
    """NEGATIVE CONTROL for the test above: div=4 must not be div=2."""
    assert halving_keep_count(32, 4) != halving_keep_count(32, MIN_HALVING_DIV)
    assert halving_rounds_left(32, 4) != halving_rounds_left(32, MIN_HALVING_DIV)


def test_the_python_reference_search_reads_halving_div() -> None:
    """⚑ THE FIX FOR THE DEFECT THIS PR ALMOST SHIPPED.

    ``run_gumbel_root_many`` hardcoded the div-2 schedule and never read
    ``cfg.halving_div``. That path is reached whenever volatility search is on
    (``selfplay.match.pick_moves_for_boards``) or the C extension is missing, so
    ``--cand-gumbel halving_div=4 --volatility-q-scale 0.5`` validated, banked 4
    and searched at 2 -- falsifying ``validate_gumbel_config``'s own promise
    that it refuses "a value the search will not run".

    Measured end to end on the real search: the divisor changes the played move.
    """
    import chess
    import torch

    from chess_anti_engine.mcts.gumbel import run_gumbel_root_many
    from chess_anti_engine.model import ModelConfig, build_model

    torch.manual_seed(0)
    model = build_model(ModelConfig(
        input_extra_features="v1", kind="tiny", embed_dim=64, num_layers=1,
        num_heads=4, ffn_mult=2, use_smolgen=False, use_nla=False,
    ))
    model.eval()

    def _search(div: int):
        probs, actions, _values, _masks = run_gumbel_root_many(
            model, [chess.Board()], device="cpu", rng=np.random.default_rng(0),
            cfg=GumbelConfig(
                input_extra_features="v1", simulations=64, topk=16,
                temperature=0.0, add_noise=False, halving_div=div,
            ),
        )
        return probs[0], actions[0]

    probs_2, action_2 = _search(MIN_HALVING_DIV)
    probs_4, action_4 = _search(4)
    assert action_2 != action_4
    assert not np.array_equal(probs_2, probs_4)


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


def test_audit_targets_training_row_guard_is_wired_not_just_present() -> None:
    """MUTANT COVER: the TRAINING branch's call had no test of its own.

    ``build_profile_search_shape`` has two exits and only the PLAY one was
    covered, so deleting the training-row call passed the whole suite -- a guard
    whose invocation nothing covers is indistinguishable from a deleted one.

    The override rides in as a raw pair (the shape ``--gumbel-training-rows``
    delivers), which is exactly the case ``parse_gumbel_overrides`` cannot
    catch for a caller that builds profiles directly.
    """
    import scripts.audit_targets as at
    from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file
    from scripts.arena_standard import IN_TREE_CONFIG

    flat = flatten_run_config_defaults(load_yaml_file(IN_TREE_CONFIG))

    def _build(profile):
        return at.build_profile_search_shape(
            "train", profile, hist="legacy", extra="v2_threats",
            pol_enc="lc0_1858", use_rel=False, play_policy_temp=1.0,
        )

    clean = at.build_search_profiles(flat, play_sims=32, play_topk=None)
    _build(clean["train"])

    poisoned = at.build_search_profiles(
        flat, play_sims=32, play_topk=None,
        gumbel_overrides=(("policy_temp", 1e300),),
        override_training_rows=True,
    )
    with pytest.raises(SystemExit, match="policy_temp"):
        _build(poisoned["train"])


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


def test_search_gain_probe_shape_is_refused_before_the_checkpoint_load() -> None:
    """Extracted from a closure inside ``run_probe`` so it is drivable at all.

    Inside the closure the guard sat behind a ~700MB checkpoint load, so nothing
    in the suite could prove it runs and an operator paid the load before being
    told the shape was unrunnable. ``main()`` now calls this builder before the
    load; here it is called with no checkpoint in sight.
    """
    import dataclasses as _dc

    from scripts.search_gain_probe import SHAPES, build_probe_gumbel_config

    base = SHAPES["production"] if "production" in SHAPES else next(iter(SHAPES.values()))
    assert build_probe_gumbel_config(base, sims=64).topk == int(base.topk)
    with pytest.raises(SystemExit, match="policy_temp"):
        build_probe_gumbel_config(_dc.replace(base, policy_temp=1e300), sims=64)
    with pytest.raises(SystemExit, match="topk"):
        build_probe_gumbel_config(_dc.replace(base, topk=1), sims=64)


def test_rare_sound_shape_is_refused_at_the_flag_boundary() -> None:
    """``_build_shape`` is the one place the flags become a ``SimShape`` and
    every subcommand goes through it, so the refusal lands before any load."""
    import scripts.rare_sound_move_coverage as rs

    def _args(**kw) -> argparse.Namespace:
        base = {
            "c_scale": 0.1, "policy_temp": 1.5, "topk": 32, "c_visit": 50.0,
            "halving_div": 2, "vloss_weight": 1, "no_noise": True,
            "gumbel_scale": 1.0, "sims": 32,
        }
        base.update(kw)
        return argparse.Namespace(**base)

    assert rs._build_shape(_args()).policy_temp == pytest.approx(1.5)
    with pytest.raises(SystemExit, match="policy_temp"):
        rs._build_shape(_args(policy_temp=0.0))
    with pytest.raises(SystemExit, match="halving_div"):
        rs._build_shape(_args(halving_div=1))


# --- flags the arena would accept, bank, and then not use --------------------


def _arena_args(**kw) -> SimpleNamespace:
    base = {
        "volatility_q_scale": 0.0, "volatility_fpu": 0.0,
        "volatility_anchor": None, "mode": "matched_sims", "temperature": 0.1,
        "search_shape": "play", "cand_gumbel": None, "ref_gumbel": None,
        "cand_vloss_weight": None, "ref_vloss_weight": None,
        "cand_target_batch": None, "ref_target_batch": None,
        "no_gumbel_noise": True,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_refusal_runs_on_the_path_main_resolves_sides_through() -> None:
    """``resolve_sides_from_args`` is what ``main()`` calls for matched_sims.

    Pinning only ``refuse_flags_the_arena_would_discard`` would leave the
    "proved the function, never proved the call" gap -- the same one that let a
    guard sit uninvoked in ``build_profile_search_shape``'s training branch.
    """
    from scripts.arena_standard import resolve_sides_from_args

    cand, ref = resolve_sides_from_args(_arena_args())
    assert cand.shape == "play"
    assert ref.shape == "play"
    with pytest.raises(SystemExit, match="policy_temp"):
        resolve_sides_from_args(_arena_args(cand_gumbel="policy_temp=1e300"))
    with pytest.raises(SystemExit, match="volatility"):
        resolve_sides_from_args(
            _arena_args(search_shape="training", volatility_q_scale=0.5),
        )
    with pytest.raises(SystemExit, match="temperature"):
        resolve_sides_from_args(_arena_args(temperature=float("nan")))


def test_volatility_flags_under_the_training_shape_are_refused() -> None:
    """They are legal, in range, banked as ``volatility_candidate`` -- and
    overwritten by the training shape's own zeros before the search sees them
    (``match.py`` applies ``side.gumbel`` AFTER the volatility kwargs).

    The PLAY shape carries no volatility keys, so there the flags survive: that
    combination must stay legal, or the refusal has removed the experiment
    instead of fixing the record.
    """
    from scripts.arena_standard import (
        refuse_flags_the_arena_would_discard,
        resolve_search_shape,
    )

    play = resolve_search_shape("play")
    training = resolve_search_shape("training")
    assert "volatility_q_scale" in training.gumbel
    assert "volatility_q_scale" not in play.gumbel

    args = _arena_args(volatility_q_scale=0.5)
    refuse_flags_the_arena_would_discard(play, args)
    refuse_flags_the_arena_would_discard(training, _arena_args())
    with pytest.raises(SystemExit, match="volatility"):
        refuse_flags_the_arena_would_discard(training, args)


def test_a_non_finite_move_temperature_is_refused() -> None:
    """``sample_action_with_temperature`` gates on ``temperature > 0``, False
    for ``nan``, so the arena plays argmax and banks ``temperature: nan``."""
    from chess_anti_engine.mcts.sampling import sample_action_with_temperature
    from scripts.arena_standard import (
        refuse_flags_the_arena_would_discard,
        resolve_search_shape,
    )

    play = resolve_search_shape("play")
    refuse_flags_the_arena_would_discard(play, _arena_args(temperature=0.1))
    with pytest.raises(SystemExit, match="temperature"):
        refuse_flags_the_arena_would_discard(play, _arena_args(temperature=float("nan")))

    # ...and the reason: nan really does take the argmax arm.
    actions = np.array([11, 22, 33], dtype=np.int64)
    weights = np.array([0.1, 0.7, 0.2], dtype=np.float64)
    argmax_idx = int(np.argmax(weights))
    assert sample_action_with_temperature(
        np.random.default_rng(0), actions, weights, float("nan"),
        argmax_idx=argmax_idx,
    ) == int(actions[argmax_idx])


def test_the_arena_int_override_parser_refuses_a_fraction_in_its_own_style() -> None:
    """``int("2.5")`` used to escape as a raw ValueError traceback."""
    from scripts.arena_standard import apply_search_overrides, resolve_search_shape

    with pytest.raises(SystemExit, match="not an integer"):
        apply_search_overrides(resolve_search_shape("play"), spec="halving_div=2.5")
    with pytest.raises(SystemExit, match="not a number"):
        apply_search_overrides(resolve_search_shape("play"), spec="c_scale=abc")


def test_the_uci_startup_guard_refuses_a_non_finite_unbounded_option() -> None:
    """The startup half's ``lo is None or hi is None`` early return came BEFORE
    any finiteness test, so an option registered without bounds would reopen the
    exact ``nan`` hole the ``setoption`` half had. Every shipped float option
    has bounds, so this drives a synthetic one."""
    from chess_anti_engine.mcts.search_options import SearchOption
    from chess_anti_engine.uci.__main__ import _refuse_out_of_range_startup_value

    unbounded = SearchOption(
        "Synthetic", "string", "c_scale", 1.0, frozenset({"gumbel"}), None, None,
    )
    _refuse_out_of_range_startup_value(unbounded, "c_scale", 0.5)
    with pytest.raises(SystemExit, match="finite"):
        _refuse_out_of_range_startup_value(unbounded, "c_scale", float("nan"))
