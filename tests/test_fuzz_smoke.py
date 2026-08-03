"""Smoke-run the CBoard differential fuzzer (scripts/fuzz_cboard_diff.py).

A handful of lockstep games asserting state parity AND the C-vs-Python encoded
planes in the production regime — enough to catch gross divergence regressions
in CI without fuzz-scale runtime. The real budgets live in
scripts/fuzz/run_fuzz.sh.
"""
from __future__ import annotations

import pytest

from chess_anti_engine.encoding import input_plane_count, rep_fix
from tests.script_loading import load_script_module

_MOD = load_script_module("fuzz_cboard_diff.py")

# configs/pbt2_small.yaml: input_extra_features v2_threats (:71) -> 175 planes,
# input_history_encoding lc0_root_legacy_meta (:70), history_rep_fix true (:197).
PROD_EXTRA_FEATURES = "v2_threats"
PROD_PLANES = 175


@pytest.fixture(autouse=True)
def _restore_rep_fix():
    """``run()`` restores the flag itself; this is belt-and-braces so a failure
    inside the fuzzer cannot leak a process-global into the rest of the suite.

    ``run()`` can only restore a value that existed; when nothing had set the
    flag yet it leaves whatever the run used, so pin that case to the default
    off rather than letting it depend on test ordering.
    """
    prior = rep_fix.current()
    yield
    want = False if prior is None else prior
    if rep_fix.current() != want:
        rep_fix.apply(want, boards_discarded=True)


def test_prod_regime_is_what_the_fuzzer_defaults_to() -> None:
    """The script's defaults must BE production, not merely be able to reach it.

    The oracle was previously off by default, pinned to v1/146 planes, and never
    applied history_rep_fix — so the regime carrying all production traffic was
    the one nothing checked (encoding audit E1). Pin the defaults here: a silent
    revert to v1 or to encode_every=0 fails this test rather than quietly
    reducing the fuzzer to a state-parity check again.
    """
    assert _MOD.PROD_EXTRA_FEATURES == PROD_EXTRA_FEATURES
    assert _MOD.PROD_HISTORY_REP_FIX is True
    assert _MOD.DEFAULT_ENCODE_EVERY > 0
    assert int(input_plane_count(_MOD.PROD_EXTRA_FEATURES)) == PROD_PLANES
    assert "lc0_root_legacy_meta" in _MOD.HISTORY_ENCODINGS

    # The realized CLI, not just the module constants: run_fuzz.sh and any
    # manual invocation get these.
    defaults = vars(_MOD.build_arg_parser().parse_args([]))
    assert defaults["encode_every"] == _MOD.DEFAULT_ENCODE_EVERY
    assert defaults["extra_features"] == PROD_EXTRA_FEATURES
    assert defaults["history_rep_fix"] is True


def test_oracle_pins_the_planes_it_actually_compared() -> None:
    """The threaded feature version must REACH the encoder, not just be a default.

    Review F1: pinning the module constants and the argparse defaults does not
    observe whether ``input_extra_features`` survives the trip to
    ``encode_cboard``/``encode_position``. One line inside ``_check_encode``
    pinning it to ``"v1"`` used to drop the 29 v2_threats planes out of the only
    C-vs-Python differential in the repo — every test green, CLI still printing
    ``v=v2_threats``. That is audit E1 verbatim.

    This drives ``_check_encode`` directly with a mismatched ``expect_planes``,
    which is the state that mutation produces, and requires a Failure naming the
    realized width. It goes red if the assertion is removed OR if the encoders
    stop being asked for the version the caller requested.
    """
    import chess

    from chess_anti_engine.encoding._lc0_ext import CBoard

    b = chess.Board()
    cb = CBoard.from_board(b)
    # v1 planes reaching a caller that asked for v2_threats — exactly what a
    # shadowed input_extra_features inside the loop produces.
    failure = _MOD._check_encode(
        cb, b, "unit", [], input_extra_features="v1",
        expect_planes=input_plane_count(PROD_EXTRA_FEATURES),
    )
    assert failure is not None, (
        "the oracle compared 146 planes while the caller asked for 175 and "
        "reported no failure — the feature version is accepted then ignored"
    )
    assert "did not reach the encoder" in str(failure)
    assert "146" in str(failure)
    assert "175" in str(failure)

    # And the matched case must still pass, so the assertion is not a blanket no.
    assert _MOD._check_encode(
        cb, b, "unit", [], input_extra_features=PROD_EXTRA_FEATURES,
        expect_planes=input_plane_count(PROD_EXTRA_FEATURES),
    ) is None


def test_the_expectation_does_not_come_from_the_value_under_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review F1, last rung: pinning ``input_extra_features`` at the top of
    ``_run`` used to stay GREEN.

    ``_run`` computed ``expect_planes`` from the same argument it passed to the
    encoders, so a mutation moved both together and the oracle happily compared
    146 planes to 146 planes while the caller had asked for 175. An expectation
    derived from the value under test is not an expectation. ``run`` now
    computes it and passes it down; this test performs that exact mutation and
    requires a Failure.
    """
    real_run = _MOD._run

    def sabotaged(*, input_extra_features: str, **kwargs: object) -> object:
        # The mutation: the version is accepted and then ignored inside _run.
        assert input_extra_features == PROD_EXTRA_FEATURES
        return real_run(input_extra_features="v1", **kwargs)

    monkeypatch.setattr(_MOD, "_run", sabotaged)
    failure = _MOD.run(
        games=1, seed=5, max_plies=6, encode_every=1,
        input_extra_features=PROD_EXTRA_FEATURES, history_rep_fix=True,
    )
    monkeypatch.setattr(_MOD, "_run", real_run)
    assert failure is not None, (
        "pinning input_extra_features inside _run produced no failure — "
        "expect_planes is still being derived from the mutated value"
    )
    assert "did not reach the encoder" in str(failure)
    assert str(PROD_PLANES) in str(failure)

    # Control: unsabotaged, the same call must pass.
    assert _MOD.run(
        games=1, seed=5, max_plies=6, encode_every=1,
        input_extra_features=PROD_EXTRA_FEATURES, history_rep_fix=True,
    ) is None


def test_differential_fuzz_state_parity_smoke() -> None:
    """State parity (legal moves, bitboards, clocks, terminal flags) is a hard
    invariant between the C board and python-chess — assert it over many plies."""
    failure = _MOD.run(games=8, seed=1234, max_plies=200, encode_every=0)
    assert failure is None, str(failure)


def test_encode_oracle_production_regime_smoke() -> None:
    """C-vs-Python encoded planes, 175 planes, history_rep_fix on, real history.

    This is the standing gate the audit's one-off measurement becomes: a future
    encoder divergence in the shipped regime fails CI instead of silently
    corrupting training data, search and every ruler at once. Long games are
    what the oracle needs (history depth, castling, EP, promotions, repetitions
    older than the C hash-stack window), so games are played out to 200 plies
    and compared every 4th ply across both production history modes — a couple
    of seconds. The fuzz-scale budget is scripts/fuzz/run_fuzz.sh diff.
    """
    failure = _MOD.run(
        games=25, seed=20260803, max_plies=200, encode_every=4,
        input_extra_features=PROD_EXTRA_FEATURES, history_rep_fix=True,
    )
    assert failure is None, str(failure)


def test_encode_oracle_catches_the_prefix_divergence() -> None:
    """Negative control: the oracle must FAIL with history_rep_fix off.

    A gate that passes in both phases would prove nothing about whether the flag
    reaches the planes. The pre-fix C encoder reconstructs per-slot repetitions
    from a hash_stack cleared on every irreversible move, so it under-reports
    repetitions older than the kept window; python-chess sees them. That is the
    divergence this oracle exists to catch, and it is exactly what turning the
    flag off must reproduce.
    """
    failure = _MOD.run(
        games=30, seed=0xDEEFF15, max_plies=200, encode_every=4,
        input_extra_features=PROD_EXTRA_FEATURES, history_rep_fix=False,
    )
    assert failure is not None, (
        "the encode oracle passed with history_rep_fix OFF — either the flag no "
        "longer reaches the repetition planes or the oracle stopped comparing "
        "them; both make the production-regime gate meaningless"
    )
    assert "encode planes diverge" in str(failure)


def test_run_restores_the_process_global_flag() -> None:
    """``history_rep_fix`` is process-global in the C encoders; a fuzz run that
    leaked it would silently change the encoding for every later test."""
    rep_fix.apply(False, boards_discarded=True)
    _MOD.run(games=1, seed=7, max_plies=20, encode_every=4, history_rep_fix=True)
    assert rep_fix.current() is False


def test_root_encode_parity_smoke() -> None:
    """The encoders must agree at the root (no history) for production modes."""
    import chess
    import numpy as np

    from chess_anti_engine.encoding import encode_position
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.encoding.cboard_encode import encode_cboard

    b = chess.Board()
    cb = CBoard.from_board(b)
    for mode in ("lc0_root", "lc0_root_legacy_meta"):
        c_planes = encode_cboard(
            cb, input_history_encoding=mode,
            input_extra_features=PROD_EXTRA_FEATURES,
        )
        py_planes = encode_position(
            b, input_history_encoding=mode,
            input_extra_features=PROD_EXTRA_FEATURES,
        )
        assert c_planes.shape[0] == PROD_PLANES
        assert np.array_equal(c_planes, py_planes), f"root encode diverged for {mode!r}"
