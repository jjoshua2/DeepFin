"""`--gumbel k=v` must reach the searched config, or the run must refuse.

The defect this file exists for is not "the override is wrong". It is the
guard: ``audit_targets`` checked its overrides under ``if p.overrides:``, i.e.
conditioned on the very value that goes missing. Deleting
``gumbel_overrides=gumbel_overrides`` from ``main()``'s
``build_search_profiles`` call therefore left every profile with ``overrides
= ()``, the guard's loop body never ran, and ``--gumbel policy_temp=2.2``
parsed, printed in the report header, and then audited the DEFAULT search
shape. Measured: that mutant passed 141 tests across the four audit suites.

A guard that can only fire while the value is present cannot detect the value
becoming absent. So the tests below pin BOTH halves:

  * the wiring (``profiles_for_audit``) actually carries the parsed request
    into the profile the runner is handed -- this is what goes red when the
    keyword is dropped; and
  * the dispatch guard (``_assert_overrides_dispatched``) fires on a
    request/profile MISMATCH, including the all-important mismatch where the
    profiles carry nothing at all.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from chess_anti_engine.mcts.gumbel import GumbelConfig
from scripts import audit_targets as at
from scripts.net_source import NetSource


def _args(**over: Any) -> SimpleNamespace:
    """The `main()` arguments `profiles_for_audit` actually reads."""
    base: dict[str, Any] = {
        "gumbel": ["policy_temp=2.2"], "sims": 64, "gumbel_topk": None,
        "rl_sims": None, "gumbel_training_rows": False,
      # ⚑ `None`, the real CLI default, meaning "whatever production runs".
      # These are read by SUBSCRIPT-equivalent attribute access in
      # `profiles_for_audit`, deliberately: a `getattr(args, ..., None)` here
      # would keep passing if main() stopped forwarding them, which is the
      # dropped-keyword failure this whole file exists for.
        "vloss_weight": None, "target_batch": None,
    }
    base.update(over)
    return SimpleNamespace(**base)


# --- the wiring: parsed request -> the profile the runner is handed ----------


def test_the_parsed_gumbel_request_reaches_the_play_profile() -> None:
    """THE regression test for the dropped keyword.

    Goes red if `profiles_for_audit` stops forwarding the parsed overrides into
    `build_search_profiles` -- the exact edit that survived the suite before.
    """
    profiles, requested = at.profiles_for_audit(_args(), {})

    assert requested == (("policy_temp", 2.2),)
    assert profiles["search"].overrides == (("policy_temp", 2.2),), (
        "--gumbel was parsed but never reached the PLAY search profile: the "
        "audit would score the DEFAULT shape under a header naming 2.2"
    )


def test_training_rows_are_untouched_unless_asked_and_follow_when_asked() -> None:
    """`--gumbel` is PLAY-only by default; `--gumbel-training-rows` adds them.

    Both directions, so neither the default nor the flag can rot into the other.
    """
    play_only, _ = at.profiles_for_audit(_args(), {})
    assert play_only["train"].overrides == ()
    assert play_only["train_fast"].overrides == ()

    both, _ = at.profiles_for_audit(_args(gumbel_training_rows=True), {})
    assert both["train"].overrides == (("policy_temp", 2.2),)
    assert both["train_fast"].overrides == (("policy_temp", 2.2),)


def test_no_gumbel_flag_is_an_empty_request_not_a_missing_one() -> None:
    """The null: absence of `--gumbel` must be `()`, and must not trip a guard."""
    profiles, requested = at.profiles_for_audit(_args(gumbel=None), {})
    assert requested == ()
    assert all(p.overrides == () for p in profiles.values())
    at._assert_overrides_dispatched(
        {n: GumbelConfig() for n in profiles}, profiles, requested=(),
    )


# --- the dispatch guard ------------------------------------------------------


def test_the_guard_fires_when_the_profiles_carry_nothing() -> None:
    """The case the old `if p.overrides:` guard structurally could not see."""
    profiles, _ = at.profiles_for_audit(_args(gumbel=None), {})
    assert profiles["search"].overrides == ()

    with pytest.raises(SystemExit) as excinfo:
        at._assert_overrides_dispatched(
            {n: GumbelConfig() for n in profiles}, profiles,
            requested=(("policy_temp", 2.2),),
        )
    assert "dropped between the command line" in str(excinfo.value)


def test_the_guard_fires_when_the_profile_carries_a_different_value() -> None:
    """Half-dropped is as wrong as fully dropped, and less visible."""
    profiles, _ = at.profiles_for_audit(_args(gumbel=["policy_temp=1.1"]), {})

    with pytest.raises(SystemExit):
        at._assert_overrides_dispatched(
            {n: GumbelConfig() for n in profiles}, profiles,
            requested=(("policy_temp", 2.2),),
        )


def test_the_guard_still_catches_an_override_that_never_reached_the_config() -> None:
    """The original dispatch check must survive the rewrite.

    Profile carries the request, but the built config does not -- e.g. a
    `_SearchProfile` field list that silently drops it.
    """
    profiles, requested = at.profiles_for_audit(_args(), {})

    with pytest.raises(SystemExit) as excinfo:
        at._assert_overrides_dispatched(
            {n: GumbelConfig() for n in profiles},  # policy_temp still 1.0
            profiles, requested=requested,
        )
    assert "did not reach the search config" in str(excinfo.value)


def test_the_guard_passes_when_the_override_really_landed() -> None:
    """POSITIVE CONTROL. Without it every raise above could be unconditional."""
    profiles, requested = at.profiles_for_audit(_args(), {})
    cfgs = {
        n: GumbelConfig(policy_temp=2.2 if p.overrides else 1.0)
        for n, p in profiles.items()
    }
    at._assert_overrides_dispatched(cfgs, profiles, requested=requested)


# --- the expectation cannot be omitted ---------------------------------------


class _RanPastTheGuard(AssertionError):
    """`_net_candidates` reached the search with a mismatched request."""


class _GuardTripwire:
    """An evaluator that must never be called.

    Makes the kill EXPLICIT rather than incidental: with the guard's call site
    removed, `_net_candidates` walks on to the forward pass, and the failure
    then names the reason instead of being an incidental `AttributeError` on a
    thin stub that a later, fatter stub would silence.
    """

    def evaluate_encoded(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise _RanPastTheGuard(
            "_net_candidates evaluated positions while the requested --gumbel "
            "overrides were absent from every profile: the "
            "_assert_overrides_dispatched call site is gone"
        )


def test_net_candidates_actually_invokes_the_dispatch_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard's INVOCATION, not its logic.

    Every other test in this file calls `_assert_overrides_dispatched` by hand,
    so replacing the call inside `_net_candidates` with `pass` passed 8/8 --
    the guard could be deleted from the production path in silence, which is
    the same shape as the `if p.overrides:` defect one level up.

    This drives the real `_net_candidates` far enough to reach the guard. The
    model load and the evaluator are the only things between the entry point
    and the check, so stubbing exactly those two keeps the call site itself
    unstubbed: with the call removed the function walks straight past the
    mismatch and no SystemExit is raised.
    """
    import chess

    import chess_anti_engine.inference as inference
    import chess_anti_engine.uci.model_loader as model_loader

    model = SimpleNamespace(
        eval=lambda: None,
        input_history_encoding="legacy",
        input_extra_features="v1",
        policy_encoding="lc0_1858",
        use_dynamic_relations=False,
    )
    monkeypatch.setattr(
        model_loader, "load_model_from_checkpoint", lambda *a, **k: model,
    )
    monkeypatch.setattr(
        inference, "LocalModelEvaluator", lambda *a, **k: _GuardTripwire(),
    )

  # Profiles carry NOTHING while the request asks for policy_temp=2.2: the
  # dropped-keyword case, which is the one the guard exists for.
    profiles, _ = at.profiles_for_audit(_args(gumbel=None), {})
    assert all(p.overrides == () for p in profiles.values())

    with pytest.raises(SystemExit) as excinfo:
        at._net_candidates(
            [chess.Board()],
            net=NetSource(checkpoint="unused-the-loader-is-stubbed"),
            device="cpu",
            batch_size=1,
            seed=0,
            profiles=profiles,
            requested_gumbel_overrides=(("policy_temp", 2.2),),
        )
    assert "dropped between the command line" in str(excinfo.value)


def test_net_candidates_requires_the_request_rather_than_defaulting_it() -> None:
    """`requested_gumbel_overrides` must have NO default.

    A default would let a caller omit it, and the guard would then compare the
    profiles against `()` -- satisfied by absence again, which is the whole bug.
    """
    import inspect

    param = inspect.signature(at._net_candidates).parameters["requested_gumbel_overrides"]
    assert param.default is inspect.Parameter.empty, (
        "requested_gumbel_overrides grew a default; the dispatch guard can now "
        "be silenced by omitting it"
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


# --- a live knob's DEAD values ------------------------------------------------
#
# The two rules above refuse a dead KNOB (a non-field, or one of
# INERT_GUMBEL_KNOBS). They said nothing about a dead VALUE of a live knob,
# and that hole was the same defect one level down: `--gumbel policy_temp=0`
# parsed, reached `cfg.policy_temp`, SATISFIED `_assert_overrides_realized`
# (the value really is on the config) and printed
# `[audit] <candidate>: --gumbel realized policy_temp=0.0` -- while
# `apply_policy_temp` returned the priors untouched. A gate that cannot fail,
# certifying a value that is silently ignored.


def _dead_and_live_policy_temps() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Read the band off `mcts.gumbel`, never off a literal in this file.

    A second copy of 0.05/20.0 here would keep passing the day the band moves,
    which is how a guard stops sharing the criterion's instrument.
    """
    from chess_anti_engine.mcts.gumbel import POLICY_TEMP_MAX, POLICY_TEMP_MIN

    dead = (
        0.0, -1.0, POLICY_TEMP_MIN / 2.0, POLICY_TEMP_MAX * 2.0, 1e300,
        float("inf"), float("nan"),
    )
    live = (POLICY_TEMP_MIN, POLICY_TEMP_MAX, 1.5, 2.2, 0.4)
    return dead, live


@pytest.mark.parametrize("value", _dead_and_live_policy_temps()[0])
def test_a_policy_temp_the_search_will_not_read_is_refused(value: float) -> None:
    """THE F1 regression test: a dead value must not become a certified one.

    The kill condition is deliberately not "SystemExit is raised" alone -- the
    positive control below proves the parser still accepts live values, so a
    parser that refused EVERYTHING could not pass both.
    """
    from chess_anti_engine.mcts.gumbel import apply_policy_temp

    with pytest.raises(SystemExit) as exc:
        at.parse_gumbel_overrides([f"policy_temp={value!r}"])
    assert "outside the band" in str(exc.value)

    # ...and the reason the refusal is right: the value really is a no-op, so
    # accepting it would have produced a certified-realized null. Asserted from
    # the arithmetic, not from the predicate the parser consults.
    import numpy as np

    pol = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    cfg = GumbelConfig(policy_temp=value)
    assert np.array_equal(apply_policy_temp(pol, cfg=cfg), pol), (
        f"policy_temp={value} is NOT a no-op, so refusing it is wrong"
    )


@pytest.mark.parametrize("value", _dead_and_live_policy_temps()[1])
def test_a_policy_temp_the_search_does_read_is_accepted(value: float) -> None:
    """NEGATIVE CONTROL for the test above.

    Without this, "refuse everything" passes the refusal test and the whole
    `--gumbel policy_temp` surface dies silently.
    """
    from chess_anti_engine.mcts.gumbel import apply_policy_temp

    assert at.parse_gumbel_overrides([f"policy_temp={value!r}"]) == (
        ("policy_temp", value),
    )

    import numpy as np

    pol = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    cfg = GumbelConfig(policy_temp=value)
    tempered = apply_policy_temp(pol, cfg=cfg)
    if value == 1.0:
        assert np.array_equal(tempered, pol)
    else:
        assert not np.array_equal(tempered, pol), (
            f"policy_temp={value} was accepted but is a no-op"
        )


def test_the_untempered_prior_is_still_requestable() -> None:
    """`policy_temp=1.0` is `policy_temp_active(1.0) == False` and NOT dead.

    It is the shipped default and "run the untempered prior explicitly" is a
    real request, so the band check must not swallow it. A refusal built on
    `policy_temp_active` alone would.
    """
    assert at.parse_gumbel_overrides(["policy_temp=1.0"]) == (("policy_temp", 1.0),)


def test_the_band_check_does_not_leak_onto_other_knobs() -> None:
    """Only `policy_temp` has a shipped activity predicate.

    `topk=0` is a different question (the C signature rejects it downstream);
    this guard must not start silently policing fields whose dead zones nobody
    has measured.
    """
    assert at.parse_gumbel_overrides(["topk=8", "c_scale=0.5"]) == (
        ("topk", 8.0), ("c_scale", 0.5),
    )


def test_the_refusal_reaches_the_real_cli_entry_point() -> None:
    """The guard must sit on the path `main()` actually parses through.

    `parse_gumbel_overrides` is called from `main()` (validation) and from
    `profiles_for_audit` (the profiles). Pinning only the helper would leave
    the same "proved the function, never proved the call" gap round 1 found.
    """
    with pytest.raises(SystemExit):
        at.profiles_for_audit(_args(gumbel=["policy_temp=0"]), {})


# --- the deep-SF reference sets ----------------------------------------------


def test_the_top10_reference_set_includes_ties_at_the_cutoff() -> None:
    """A move equal in cp to the 10th-best is IN the top 10.

    Slicing `_ranked[:10]` picked among equals by mapping iteration order, so
    `out_of_top10` measured MultiPV tie ordering rather than move quality.
    `sf_top1_set` has always been built by score; this is the same rule at the
    other cutoff.
    """
    move_cp = {f"m{i}": float(100 - i) for i in range(9)}   # ranks 1..9, strictly better
    move_cp.update({"tie_a": 10.0, "tie_b": 10.0, "tie_c": 10.0})  # ranks 10,11,12
    move_cp["worse"] = 1.0

    top1, top10 = at.sf_reference_sets(move_cp)

    assert top1 == {"m0"}
    assert {"tie_a", "tie_b", "tie_c"} <= top10, (
        "a move scored exactly as well as the 10th-best was ranked out of the "
        "top 10 by dict order"
    )
    assert "worse" not in top10
    assert len(top10) == 12


def test_the_top10_set_is_insensitive_to_input_ordering() -> None:
    """The property the slice violated, stated directly.

    Two dicts holding the SAME scores in different insertion orders must give
    the same set — otherwise the statistic is a function of MultiPV emission
    order.
    """
    scores = {f"m{i}": float(100 - i) for i in range(9)}
    scores.update({"tie_a": 10.0, "tie_b": 10.0})
    reordered = dict(reversed(list(scores.items())))

    assert at.sf_reference_sets(scores)[1] == at.sf_reference_sets(reordered)[1]


def test_a_short_multipv_list_yields_no_top10_claim() -> None:
    """Fewer than 10 listings cannot support an "outside the top 10" verdict.

    An empty set is the caller's `None` sentinel; a non-empty one would be
    scored as a success for every candidate.
    """
    top1, top10 = at.sf_reference_sets({"a": 5.0, "b": 5.0, "c": 1.0})
    assert top1 == {"a", "b"}
    assert top10 == set()
    assert at.sf_reference_sets({}) == (set(), set())


def test_the_score_form_is_a_no_op_at_the_shipped_multipv_width() -> None:
    """NEGATIVE CONTROL: no banked audit number moves.

    The frozen set is MultiPV=10 exactly, so there is never an 11th listing to
    tie with and the score form must agree with the old slice element for
    element. If this ever fails, the audit set got wider and the CHANGE — not
    the old behaviour — is what the numbers now reflect.
    """
    for n in range(10, 0, -1):
        move_cp = {f"m{i}": float(100 - i) for i in range(n)}
        ranked = sorted(move_cp.items(), key=lambda kv: -kv[1])
        old = {u for u, _ in ranked[:10]} if len(ranked) >= 10 else set()
        assert at.sf_reference_sets(move_cp)[1] == old, f"diverged at {n} listings"


def test_the_audit_loop_builds_its_top10_through_the_helper() -> None:
    """The F6 fix's CALL SITE, not just its logic.

    Same shape as the `--gumbel` dispatch guard's own history: the helper is
    pinned by the four tests above, its INVOCATION by nothing, so reverting
    `audit_targets.main()` to the inline `_ranked[:10]` slice passed the whole
    suite. Logic pinned, wiring not — one level down from R5-B2.

    ⚑ This check is STRUCTURAL, and that is a compromise, not a preference.
    The call site lives inside `audit_targets.main()`, which needs a checkpoint,
    a Stockfish binary and the frozen audit set before it reaches the line; the
    behavioural observation (drive the loop, assert a tied 11th-ranked move is
    scored `out_of_top10=false`) is not reachable in a unit test. So this
    asserts the two things a revert cannot avoid tripping: `main` must reference
    the helper, and no top-10 set may be built by slicing anywhere in the
    module.
    """
    import inspect

    assert "sf_reference_sets" in at.main.__code__.co_names, (
        "audit_targets.main() no longer calls sf_reference_sets — the top-10 "
        "reference set is being built somewhere else, and the tie handling the "
        "helper's tests pin is not what the audit actually runs"
    )

  # The revert reintroduces a slice. Catch the CONSTRUCTION via the AST, not a
  # substring: `sf_reference_sets`'s own docstring quotes `_ranked[:10]` when
  # explaining why it is wrong, and a text scan cannot tell prose from code.
    import ast

  # ⚑ `lower is None` alone matched only the `[:10]` spelling, so the identical
  # `_ranked[0:10]` reintroduced the tie bug with `co_names` still satisfied and
  # this scan still green. An explicit zero lower bound is the SAME slice.
    def _starts_at_zero(lower: ast.expr | None) -> bool:
        return lower is None or (
            isinstance(lower, ast.Constant)
            and not isinstance(lower.value, bool)
            and isinstance(lower.value, int)
            and lower.value == 0
        )

    tree = ast.parse(inspect.getsource(at))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Slice)
        and _starts_at_zero(node.slice.lower)
        and isinstance(node.slice.upper, ast.Constant)
        and node.slice.upper.value == 10
    ]
    assert not offenders, (
        "a top-10 set is being built by slicing, which picks among moves tied "
        f"at the tenth-ranked cp by mapping order (audit_targets.py lines "
        f"{offenders})"
    )


# --- the runner ARGUMENTS that are not GumbelConfig fields (#443 F1) --------


def test_the_guarded_vloss_weight_actually_reaches_the_c_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTANT (#443 F1): certify production's `vloss_weight`, dispatch a literal.

    ⚑ THIS IS THE SURVIVOR. Replacing
    ``tb_kwargs[name]["vloss_weight"] = int(shapes[name].vloss_weight)`` with
    ``= 0`` left an 18-mutant battery green: every other test checks the SHAPE,
    and the shape stayed production's while the search ran the duplicate-leaf
    arm. "A value accepted and then silently ignored" is this repo's signature
    defect, and the fix for it had shipped a fresh instance one line
    downstream of its own guard.

    So this drives the real ``_net_candidates`` and reads the keyword the
    runner was actually handed FOR THE TRAINING ROW — matched by config
    identity, because the PLAY row deliberately runs a different value and a
    test that read the first call would pass against the bug.

    BOTH directions: production's value by default, and an operator's explicit
    value when asked for. Pinning only "== 1" would pass against a constant.
    """
    import chess
    import numpy as np

    import chess_anti_engine.inference as inference
    import chess_anti_engine.mcts.gumbel_c as gumbel_c
    import chess_anti_engine.uci.model_loader as model_loader
    from chess_anti_engine.eval.production_shape import production_search_shape
    from chess_anti_engine.moves import POLICY_SIZE

    model = SimpleNamespace(
        eval=lambda: None,
        input_history_encoding="legacy",
        input_extra_features="v1",
        policy_encoding="lc0_1858",
        use_dynamic_relations=False,
    )

    class _Evaluator:
        def evaluate_encoded(self, x: Any, relations: Any = None) -> Any:
            del relations
            n = len(x)
            return (
                np.zeros((n, 1858), dtype=np.float32),
                np.zeros((n, 3), dtype=np.float32),
            )

    calls: list[tuple[Any, dict[str, Any]]] = []

    def _fake_runner(**kwargs: Any) -> tuple[Any, ...]:
        calls.append((kwargs["cfg"], dict(kwargs)))
        n = len(kwargs["boards"])
        probs = np.full((n, POLICY_SIZE), 1.0 / POLICY_SIZE, dtype=np.float64)
        return probs, np.zeros(n, dtype=np.int64), np.zeros(n), np.ones((n, 1))

    monkeypatch.setattr(
        model_loader, "load_model_from_checkpoint", lambda *a, **k: model,
    )
    monkeypatch.setattr(inference, "LocalModelEvaluator", lambda *a, **k: _Evaluator())
    monkeypatch.setattr(gumbel_c, "run_gumbel_root_many_c", _fake_runner)

    flat: dict[str, object] = {"gumbel_vloss_weight": 1, "gumbel_target_batch": 0}
    prod = production_search_shape(flat, simulations=8)
    assert prod.vloss_weight == 1, "the fixture does not exercise the case"

    def _dispatched(**profile_kw: Any) -> dict[str, Any]:
        calls.clear()
        profiles = at.build_search_profiles(
            flat, play_sims=8, play_topk=None, **profile_kw,
        )
        *_rest, shapes = at._net_candidates(
            [chess.Board()],
            net=NetSource(checkpoint="unused-the-loader-is-stubbed"),
            device="cpu", batch_size=1, seed=0,
            profiles=profiles, requested_gumbel_overrides=(),
        )
      # ⚑ Matched by IDENTITY to the training row's config. The PLAY row runs a
      # different vloss_weight on purpose, so "the first captured call" would
      # be the wrong object and would pass against the bug.
        train_cfg = shapes["train"].cfg
        for cfg, kwargs in calls:
            if cfg is train_cfg:
                return kwargs
        raise AssertionError("the training row never reached the C runner")

    handed = _dispatched()
    assert handed["vloss_weight"] == 1, (
        "the runner was handed a vloss_weight the guard did not certify — the "
        "audit would search the duplicate-leaf shape under a header reading "
        "'production training target'"
    )
    assert handed["target_batch"] == int(prod.target_batch)

  # ...and the operator arm still reaches the runner, so the fix is not a
  # constant wearing the costume of a lookup.
    handed_arm = _dispatched(vloss_weight=0, target_batch=1)
    assert handed_arm["vloss_weight"] == 0
    assert handed_arm["target_batch"] == 1
