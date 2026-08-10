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


def _args(**over: Any) -> SimpleNamespace:
    """The `main()` arguments `profiles_for_audit` actually reads."""
    base: dict[str, Any] = {
        "gumbel": ["policy_temp=2.2"], "sims": 64, "gumbel_topk": None,
        "rl_sims": None, "gumbel_training_rows": False,
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
            checkpoint="unused-the-loader-is-stubbed",
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
