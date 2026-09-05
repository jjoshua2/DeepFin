"""Tier-14 arm (a): ``search_wdl_draw_mode = parametric_q``.

``search_wdl`` is 0.31 of the trained value target. Its production construction
(``_mcts_tree.c``) reads the net's own RAW root draw output as the D axis and
then uses that same number to CLAMP the searched q, so ~31% of the target's draw
mass is the net grading itself and the target's confidence is capped on decisive
rows (measured: the clamp binds on 12.27% of the lowest-d_raw quartile).
``parametric_q`` builds the whole triple from the searched q through the
cp-logistic family's own implied draw curve — no net-WDL input at all.

The standing question here is not "is the arithmetic right" but "does the knob
take effect on the PRODUCTION path, and what observation proves it did". So the
parametric assertions go through the real C extension and the real record-append
functions, never through a Python re-derivation that the loop does not run.

The four ways this deploy could silently not happen, one section each:
  * the curve itself (wrong family, wrong parameters, not actually a simplex),
  * the flag not being bit-identical when off,
  * the two ply paths (C fast path / Python fallback) disagreeing,
  * the config route (schema -> TrialConfig -> reco -> GameConfig -> the C call).
"""
from __future__ import annotations

import dataclasses
import itertools
import math
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.encoding.features import EXTRA_FEATURES_V1, extra_feature_plane_count
from chess_anti_engine.model import ModelConfig
from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.network_turn import (
    _SWDL_DRAW_MODE_TO_C,
    _append_records_via_c,
    _append_records_via_python,
    _search_wdl_draw_mode,
)
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.state import SelfplayState
from chess_anti_engine.stockfish.wdl import (
    SEARCH_WDL_DRAW_MODES,
    SEARCH_WDL_DRAW_NET_RAW,
    SEARCH_WDL_DRAW_PARAMETRIC_Q,
    cp_to_wdl,
    parametric_draw_from_q,
    q_to_wdl_parametric,
)
from chess_anti_engine.tune.distributed_runtime import build_recommended_worker
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.utils.config_yaml import SELFPLAY_CONFIG_KEYS
from chess_anti_engine.worker import WorkerSession

try:
    from chess_anti_engine.mcts._mcts_tree import (
        SWDL_DRAW_NET_RAW,
        SWDL_DRAW_PARAMETRIC_Q,
    )
    from chess_anti_engine.mcts._mcts_tree import batch_process_ply as _bpp
    HAS_C = True
except ImportError:  # pragma: no cover - the dev env hard-requires the extension
    _bpp = None
    SWDL_DRAW_NET_RAW = 0
    SWDL_DRAW_PARAMETRIC_Q = 1
    HAS_C = False

requires_c = pytest.mark.skipif(not HAS_C, reason="C extension not available")

_PRODUCTION_YAML = Path(__file__).resolve().parents[1] / "configs" / "pbt2_small.yaml"

# Production cp-logistic parameters (configs/pbt2_small.yaml sf_wdl_cp_*), the
# curve the SF component of the value blend already uses.
PROD_SLOPE = 0.006
PROD_WIDTH = 120.0

# The `n_extra` the positional-argument tail needs: once ANY optional arg after
# it is supplied it must be spelled out, and 0 is not a legal plane count.
_N_EXTRA_V1 = extra_feature_plane_count(EXTRA_FEATURES_V1)


def _bpp_call():
    assert _bpp is not None
    return _bpp


# ---------------------------------------------------------------------------
# The curve: D(q) really is the cp-logistic family's own draw mass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("slope", "width"),
    [(PROD_SLOPE, PROD_WIDTH), (0.010, 60.0), (0.0045, 145.0), (0.02, 30.0)],
)
def test_draw_curve_matches_the_cp_logistic_implied_curve(
    slope: float, width: float,
) -> None:
    """The reason to trust the closed form: it is not a fit.

    Push a cp grid through the PRODUCTION ``cp_to_wdl`` — the same function that
    builds the SF component — read off the implied ``(q, D)`` pair, and check
    ``parametric_draw_from_q(q)`` reproduces D. The residual is float32
    quantisation in ``cp_to_wdl``'s own output (measured max 3.7e-8 at the
    production parameters), not curve error, so the tolerance is tight enough
    that a refit or a rearranged formula would have to be exact to pass.
    """
    cps = np.concatenate([
        np.linspace(-5000.0, 5000.0, 20001),
        np.linspace(-40000.0, 40000.0, 4001),
    ])
    worst = 0.0
    for cp in cps:
        triple = cp_to_wdl(float(cp), None, slope=slope, draw_width_cp=width)
        q = float(triple[0] - triple[2])
        got = parametric_draw_from_q(q, slope=slope, draw_width_cp=width)
        worst = max(worst, abs(got - float(triple[1])))
    assert worst < 1e-6, f"max |D(q) - cp_logistic D| = {worst:.3e}"


def test_draw_curve_endpoints() -> None:
    """``D(0)`` is the cp-logistic's draw mass at cp = 0, ``D(+-1)`` is 0.

    Both exact, not approximate: ``D(0) = tanh(w/2)`` and ``csch^2 + 1 =
    coth^2``. The cp = 0 pin is what "matches the production draw mass at
    cp = 0" means, and it is the value a constant-D mutant would have to
    reproduce at every other q as well (it cannot — see the monotonicity test).
    """
    w = PROD_SLOPE * PROD_WIDTH
    d0 = parametric_draw_from_q(0.0, slope=PROD_SLOPE, draw_width_cp=PROD_WIDTH)
    assert d0 == pytest.approx(math.tanh(w / 2.0), abs=1e-12)
    assert d0 == pytest.approx(
        float(cp_to_wdl(0.0, None, slope=PROD_SLOPE, draw_width_cp=PROD_WIDTH)[1]),
        abs=1e-7,
    )
    for q in (1.0, -1.0):
        assert parametric_draw_from_q(
            q, slope=PROD_SLOPE, draw_width_cp=PROD_WIDTH,
        ) == 0.0


def test_draw_curve_is_even_and_strictly_decreasing_in_abs_q() -> None:
    qs = np.linspace(0.0, 1.0, 401)
    ds = [parametric_draw_from_q(float(q), slope=PROD_SLOPE, draw_width_cp=PROD_WIDTH)
          for q in qs]
    for a, b in itertools.pairwise(ds):
        assert b <= a
    # strict over any non-trivial step, so a constant curve fails
    assert ds[0] - ds[-1] > 0.3
    for q, d in zip(qs, ds, strict=True):
        mirrored = parametric_draw_from_q(
            -float(q), slope=PROD_SLOPE, draw_width_cp=PROD_WIDTH,
        )
        assert mirrored == pytest.approx(d, abs=1e-15)


def test_triple_is_a_simplex_point_by_construction() -> None:
    """Non-negative and summing to 1 on the whole q range, with no clamp doing
    the work — the property the ``d_raw`` clamp exists to enforce in the other
    mode and that this construction gets for free."""
    for q in np.linspace(-1.0, 1.0, 2001):
        t = q_to_wdl_parametric(float(q), slope=PROD_SLOPE, draw_width_cp=PROD_WIDTH)
        assert t.shape == (3,)
        assert float(t.min()) >= 0.0, (q, t)
        assert float(t.sum()) == pytest.approx(1.0, abs=1e-6)
        # W - L recovers q exactly: the searched value is preserved, which is
        # the half of the target this arm is NOT trying to change.
        assert float(t[0] - t[2]) == pytest.approx(float(q), abs=1e-6)


def test_draw_curve_refuses_a_degenerate_zone() -> None:
    """``cp_to_wdl`` accepts ``draw_width_cp = 0``; the implied curve is
    singular there. Refuse rather than emit a silently different curve."""
    for kwargs in (
        {"slope": 0.0, "draw_width_cp": 120.0},
        {"slope": 0.006, "draw_width_cp": 0.0},
        {"slope": -0.006, "draw_width_cp": 120.0},
    ):
        with pytest.raises(ValueError, match="parametric_draw_from_q"):
            parametric_draw_from_q(0.0, **cast(Any, kwargs))


# ---------------------------------------------------------------------------
# Default off is bit-identical
# ---------------------------------------------------------------------------


def _c_batch(
    *, wdl_logits: np.ndarray, values: np.ndarray, extra: tuple[Any, ...] = (),
) -> np.ndarray:
    """Run ``batch_process_ply`` on fresh start positions; return search WDL."""
    from chess_anti_engine.encoding.cboard_encode import cboard_from_board_fast

    import chess

    n = int(wdl_logits.shape[0])
    boards = [chess.Board() for _ in range(n)]
    cboards = [cboard_from_board_fast(b) for b in boards]
    legal = cboards[0].legal_move_indices()
    rng = np.random.default_rng(11)
    pol = rng.standard_normal((n, POLICY_SIZE)).astype(np.float32)
    probs = np.abs(rng.standard_normal((n, POLICY_SIZE))).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    actions = np.array([int(legal[i % len(legal)]) for i in range(n)], dtype=np.int32)
    result = cast(Any, _bpp_call()(
        cboards, pol, wdl_logits.astype(np.float32), actions,
        values.astype(np.float64), probs,
        1, 4.8, 3.8, 0.09, 1.0, *extra,
    ))
    return np.asarray(result[3])


@requires_c
def test_omitting_the_new_args_is_bit_identical_to_passing_net_raw() -> None:
    """The ``df_norm_scale == 0`` contract, applied to this knob: a caller that
    never heard of the mode and a caller that explicitly selects the default
    must produce the SAME BYTES, not merely close numbers."""
    rng = np.random.default_rng(3)
    wdl = rng.standard_normal((16, 3)).astype(np.float32)
    values = rng.uniform(-0.99, 0.99, size=16)

    old = _c_batch(wdl_logits=wdl, values=values)
    new_default = _c_batch(
        wdl_logits=wdl, values=values,
        extra=(0, _N_EXTRA_V1, 0, 0.0, 0.0, 0.0, SWDL_DRAW_NET_RAW, PROD_SLOPE, PROD_WIDTH),
    )
    assert old.tobytes() == new_default.tobytes()
    # ...and the parametric arm on the SAME inputs is genuinely different, so
    # the equality above is not a vacuous "the flag does nothing" pass.
    parametric = _c_batch(
        wdl_logits=wdl, values=values,
        extra=(0, _N_EXTRA_V1, 0, 0.0, 0.0, 0.0, SWDL_DRAW_PARAMETRIC_Q, PROD_SLOPE, PROD_WIDTH),
    )
    assert parametric.tobytes() != old.tobytes()


@requires_c
def test_net_raw_branch_is_bit_for_bit_the_original_formula() -> None:
    """The byte comparison above is between two runs of the SAME build, so on
    its own it would survive a rewrite of the net_raw arithmetic. Pin the
    formula itself instead, in float32 and exactly:

        d_raw = net softmax draw;  rem = max(0, 1 - d_raw)
        w = 0.5*(rem + clamp(best_q, +-rem));  l = rem - w

    Taking ``d_raw`` from the C's own output makes libm's ``expf`` cancel, so
    the remaining comparison is bit-exact rather than tolerance-based; the
    softmax itself is checked separately against float64 numpy.
    """
    rng = np.random.default_rng(5)
    wdl = rng.standard_normal((12, 3)).astype(np.float32)
    values = rng.uniform(-1.4, 1.4, size=12)   # spans the clamp on both sides
    out = _c_batch(wdl_logits=wdl, values=values)

    e = np.exp(wdl.astype(np.float64) - wdl.astype(np.float64).max(axis=1, keepdims=True))
    np.testing.assert_allclose(
        out[:, 1], (e / e.sum(axis=1, keepdims=True))[:, 1], atol=1e-6,
    )

    d_raw = out[:, 1].astype(np.float32)
    rem = np.maximum(np.float32(0.0), np.float32(1.0) - d_raw)
    q = np.clip(values.astype(np.float32), -rem, rem)
    w = np.float32(0.5) * (rem + q)
    assert out[:, 0].tobytes() == w.tobytes()
    assert out[:, 2].tobytes() == (rem - w).tobytes()
    # the fixture must actually exercise the clamp, or this pins nothing
    assert bool((np.abs(values.astype(np.float32)) > rem).any())


# ---------------------------------------------------------------------------
# The two ply paths
# ---------------------------------------------------------------------------

_BATCH = 6


def _session_state(game: GameConfig) -> SelfplayState:
    """A real ``SelfplayState``, built the way ``play_batch`` builds one."""
    evaluator = Mock(spec=["evaluate_encoded"])
    stockfish = Mock(spec=["search", "nodes"])
    stockfish.nodes = 0
    return SelfplayState.create(
        model=None,
        device="cpu",
        rng=np.random.default_rng(0),
        stockfish=stockfish,
        evaluator=evaluator,
        batch_size=_BATCH,
        continuous=False,
        target=_BATCH,
        opponent=OpponentConfig(),
        temp=TemperatureConfig(),
        search=SearchConfig(),
        opening=OpeningConfig(),
        diff_focus=DiffFocusConfig(),
        game=game,
    )


def _ply_inputs(state: SelfplayState) -> dict[str, Any]:
    """One deterministic ply batch, shared by both append paths."""
    n = _BATCH
    rng = np.random.default_rng(7)
    pol = (rng.standard_normal((n, POLICY_SIZE)) * 2.0).astype(np.float32)
    wdl_logits = rng.standard_normal((n, 3)).astype(np.float32) * 1.5
    probs = np.zeros((n, POLICY_SIZE), dtype=np.float32)
    actions: list[int | None] = []
    values: list[float | None] = []
    for i in range(n):
        legal = np.flatnonzero(legal_move_mask(state.boards[i]))
        k = max(2, len(legal) // 4)
        chosen = rng.choice(legal, size=k, replace=False)
        visits = rng.integers(1, 40, size=k).astype(np.float32)
        probs[i, chosen] = visits / visits.sum()
        actions.append(int(chosen[0]))
        # Spread across the range, including near +-1 where the d_raw clamp bites.
        values.append(float(np.clip(rng.uniform(-1.05, 1.05), -1.0, 1.0)))
    return {
        "pol_logits": pol,
        "wdl_logits": wdl_logits,
        "probs_list": [probs[i] for i in range(n)],
        "actions": actions,
        "values_list": values,
    }


def _via_c(game: GameConfig) -> np.ndarray:
    state = _session_state(game)
    inp = _ply_inputs(state)
    _append_records_via_c(
        state, list(range(_BATCH)),
        cb_encode_list=list(state.cboards[:_BATCH]),
        pol_logits=inp["pol_logits"], wdl_logits_raw=inp["wdl_logits"],
        actions=inp["actions"], values_list=inp["values_list"],
        probs_list=inp["probs_list"],
        gumbel_diags=[None] * _BATCH, is_full_py=[True] * _BATCH,
        sample_weights=[1.0] * _BATCH, diff_focus=DiffFocusConfig(),
    )
    return np.stack([
        np.asarray(state.samples_per_game[i][0].search_wdl_est)
        for i in range(_BATCH)
    ])


def _via_python(game: GameConfig) -> np.ndarray:
    state = _session_state(game)
    inp = _ply_inputs(state)
    logits = inp["wdl_logits"].astype(np.float64)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    wdl_est = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
    planes = input_plane_count(state.game.input_extra_features)
    _append_records_via_python(
        state, list(range(_BATCH)),
        xs_batch=np.zeros((_BATCH, planes, 8, 8), dtype=np.float32),
        pol_logits=inp["pol_logits"], wdl_est=wdl_est,
        probs_list=inp["probs_list"], actions=inp["actions"],
        values_list=inp["values_list"],
        gumbel_diags=[None] * _BATCH, masks_list=[None] * _BATCH,
        is_full=np.ones(_BATCH, dtype=bool), sample_weights=[1.0] * _BATCH,
        diff_focus=DiffFocusConfig(),
    )
    return np.stack([
        np.asarray(state.samples_per_game[i][0].search_wdl_est)
        for i in range(_BATCH)
    ])


@requires_c
@pytest.mark.parametrize("mode", SEARCH_WDL_DRAW_MODES)
@pytest.mark.parametrize(
    ("slope", "width"), [(PROD_SLOPE, PROD_WIDTH), (0.012, 45.0)],
)
def test_c_and_python_ply_paths_agree(mode: str, slope: float, width: float) -> None:
    """Both REAL append paths, same inputs, both modes.

    ``_append_records_via_python`` only runs when the C extension is missing, so
    a fallback that quietly builds a different training target is the defect and
    not the safety net — the same rule ``_policy_kl`` carries. This is also the
    test that proves the mode reaches the C ARGUMENT LIST: nothing else here
    drives ``_append_records_via_c``'s call assembly.
    """
    game = GameConfig(
        search_wdl_draw_mode=mode,
        sf_wdl_use_cp_logistic=True,
        sf_wdl_cp_slope=slope,
        sf_wdl_cp_draw_width=width,
    )
    np.testing.assert_allclose(_via_c(game), _via_python(game), atol=1e-6)


@requires_c
def test_the_two_modes_produce_different_stored_targets() -> None:
    """Both parity runs above would pass if the mode were ignored on BOTH
    paths. It is not: end to end through the record-append path, the stored
    ``search_wdl`` differs."""
    base = GameConfig(
        sf_wdl_use_cp_logistic=True,
        sf_wdl_cp_slope=PROD_SLOPE,
        sf_wdl_cp_draw_width=PROD_WIDTH,
    )
    net_raw = _via_c(
        dataclasses.replace(base, search_wdl_draw_mode=SEARCH_WDL_DRAW_NET_RAW),
    )
    par = _via_c(
        dataclasses.replace(base, search_wdl_draw_mode=SEARCH_WDL_DRAW_PARAMETRIC_Q),
    )
    assert not np.allclose(net_raw, par, atol=1e-4)
    # and the parametric rows are exactly the shared definition of the curve
    for row in par:
        q = float(row[0] - row[2])
        np.testing.assert_allclose(
            row,
            q_to_wdl_parametric(q, slope=PROD_SLOPE, draw_width_cp=PROD_WIDTH),
            atol=1e-6,
        )


@requires_c
def test_parametric_mode_ignores_the_net_draw_output_entirely() -> None:
    """The point of the arm: with the searched q held fixed, moving the net's
    root WDL must not move the stored target at all. Under ``net_raw`` the same
    perturbation moves BOTH axes."""
    values = np.full(4, 0.42)
    wdl_a = np.tile(np.array([0.2, 0.5, 0.3], dtype=np.float32), (4, 1))
    wdl_b = np.tile(np.array([2.0, -1.5, 0.7], dtype=np.float32), (4, 1))

    par_kwargs: dict[str, Any] = {
        "extra": (0, _N_EXTRA_V1, 0, 0.0, 0.0, 0.0,
                  SWDL_DRAW_PARAMETRIC_Q, PROD_SLOPE, PROD_WIDTH),
    }
    par_a = _c_batch(wdl_logits=wdl_a, values=values, **par_kwargs)
    par_b = _c_batch(wdl_logits=wdl_b, values=values, **par_kwargs)
    assert par_a.tobytes() == par_b.tobytes()

    raw_a = _c_batch(wdl_logits=wdl_a, values=values)
    raw_b = _c_batch(wdl_logits=wdl_b, values=values)
    assert not np.allclose(raw_a, raw_b, atol=1e-3)


@requires_c
def test_parametric_mode_removes_the_confidence_cap_on_decisive_rows() -> None:
    """The second defect, measured on the exact shape the ledger reports.

    A near-won row (searched q = 0.96) whose net still holds 30% draw mass:
    ``net_raw`` clamps the target's win probability to ``1 - d_raw`` and stores
    W = 0.70 however winning the search says the position is. The parametric
    curve sends D -> 0 as |q| -> 1, so the searched confidence survives.
    """
    values = np.full(1, 0.96)
    # softmax([0.0, 0.847, 0.0]) ~ (0.3, 0.4, 0.3): a fat raw draw channel
    wdl = np.array([[0.0, 0.847, 0.0]], dtype=np.float32)

    raw = _c_batch(wdl_logits=wdl, values=values)[0]
    par = _c_batch(
        wdl_logits=wdl, values=values,
        extra=(0, _N_EXTRA_V1, 0, 0.0, 0.0, 0.0, SWDL_DRAW_PARAMETRIC_Q, PROD_SLOPE, PROD_WIDTH),
    )[0]

    assert float(raw[0]) < 0.75, "the d_raw clamp is not binding — fixture is wrong"
    assert float(raw[0] - raw[2]) < 0.65, "net_raw did not truncate the searched q"
    assert float(par[0]) > 0.95
    assert float(par[0] - par[2]) == pytest.approx(0.96, abs=1e-4)


@requires_c
def test_the_c_path_reads_the_curve_knobs_it_is_handed() -> None:
    """MUTATION RESIDUAL, and the signature defect of this codebase.

    Every other parametric assertion here runs at the PRODUCTION curve, so a C
    path that ignored ``swdl_cp_slope``/``swdl_cp_draw_width`` and hardcoded
    ``0.006 * 120.0`` passed all of them — the value accepted and then silently
    ignored, one level below the config plumbing this file spends its second
    half checking. Drive a curve that is NOT production and pin the output to
    the shared definition AT THAT CURVE.
    """
    alt_slope, alt_width = 0.012, 45.0
    values = np.linspace(-0.95, 0.95, 8)
    wdl = np.tile(np.array([0.1, 0.4, 0.2], dtype=np.float32), (8, 1))

    alt = _c_batch(
        wdl_logits=wdl, values=values,
        extra=(0, _N_EXTRA_V1, 0, 0.0, 0.0, 0.0,
               SWDL_DRAW_PARAMETRIC_Q, alt_slope, alt_width),
    )
    prod = _c_batch(
        wdl_logits=wdl, values=values,
        extra=(0, _N_EXTRA_V1, 0, 0.0, 0.0, 0.0,
               SWDL_DRAW_PARAMETRIC_Q, PROD_SLOPE, PROD_WIDTH),
    )
    # the two curves are genuinely distinguishable at this fixture
    assert not np.allclose(alt, prod, atol=1e-3)
    for row, v in zip(alt, values, strict=True):
        np.testing.assert_allclose(
            row,
            q_to_wdl_parametric(float(v), slope=alt_slope, draw_width_cp=alt_width),
            atol=1e-6,
        )


@requires_c
def test_the_c_path_emits_no_negative_mass_at_the_endpoints() -> None:
    """MUTATION RESIDUAL: the fp range guard had no test.

    `D(q) = coth - sqrt(csch^2 + q^2)` is exactly 0 at `|q| = 1` in real
    arithmetic and **-2.2e-16** in doubles, so without the clamp a fully
    decisive row stores a NEGATIVE probability in a soft-CE target. Every other
    fixture here samples the interior, where the guard is inactive; drive the
    endpoints, where it is the only thing standing between the loop and a
    negative target entry.
    """
    values = np.array([1.0, -1.0, 1.0 - 1e-9, -(1.0 - 1e-9)])
    wdl = np.tile(np.array([0.3, 0.5, 0.2], dtype=np.float32), (4, 1))
    out = _c_batch(
        wdl_logits=wdl, values=values,
        extra=(0, _N_EXTRA_V1, 0, 0.0, 0.0, 0.0,
               SWDL_DRAW_PARAMETRIC_Q, PROD_SLOPE, PROD_WIDTH),
    )
    assert float(out.min()) >= 0.0, out
    # D is exactly zero at the endpoints, not merely small-and-signed
    assert float(out[0][1]) == 0.0
    assert float(out[1][1]) == 0.0
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)
    # and the Python twin agrees on the same endpoints
    for row, v in zip(out, values, strict=True):
        np.testing.assert_allclose(
            row,
            q_to_wdl_parametric(float(v), slope=PROD_SLOPE, draw_width_cp=PROD_WIDTH),
            atol=1e-7,
        )


@requires_c
@pytest.mark.parametrize(
    ("slope", "width"), [(0.02, 500.0), (0.01, 300.0), (PROD_SLOPE, PROD_WIDTH)],
)
def test_the_c_path_stays_non_negative_at_interior_q_in_a_wide_draw_zone(
    slope: float, width: float,
) -> None:
    """REVIEW FINDING S1: the endpoint test above could not see this.

    ``D <= 1 - |q|`` is proved in EXACT arithmetic and enforced in DOUBLE, but
    the triple is rebuilt in float32; rounding ``D`` up across that cast leaves
    ``1 - D < |q|`` and stores a negative W. It fires at INTERIOR q with a wide
    draw zone, where the endpoint fixture never looks: measured on the built
    extension before the fix, 35 of these 8000 rows at ``w = 10`` stored
    ``W = -7.45e-09`` while the Python twin stayed non-negative. The margin
    shrinks monotonically in ``w`` and hits 0 at ``w ~ 5``, so the wide curve
    here is the reachable-but-non-production case, and the production curve
    (``w = 0.72``) rides along to prove the fix costs it nothing.

    Bitwise, not ``atol``: an ``atol=1e-6`` comparison cannot distinguish
    ``-7.45e-09`` from ``+0.0``, which is exactly why the existing parity test
    was blind to a genuine C/Python divergence. Both sides now do the same
    float32 operations in the same order, so anything less than byte equality
    is a real disagreement.
    """
    n = 8000
    # float32-exact so `(double)best_q` in C and `float(v)` here are the same
    # number; otherwise the two would differ legitimately in the last ulp and
    # the byte comparison below would be measuring the fixture, not the code.
    values = np.linspace(-0.9999, 0.9999, n, dtype=np.float64).astype(
        np.float32,
    ).astype(np.float64)
    wdl = np.tile(np.array([0.3, 0.5, 0.2], dtype=np.float32), (n, 1))
    out = _c_batch(
        wdl_logits=wdl, values=values,
        extra=(0, _N_EXTRA_V1, 0, 0.0, 0.0, 0.0,
               SWDL_DRAW_PARAMETRIC_Q, slope, width),
    )
    assert float(out.min()) >= 0.0, (slope, width, out[out.min(axis=1) < 0.0][:5])
    expected = np.stack([
        q_to_wdl_parametric(float(v), slope=slope, draw_width_cp=width)
        for v in values
    ])
    assert out.tobytes() == expected.tobytes(), (
        "the C and Python twins disagree bitwise on the parametric triple"
    )


def test_the_python_twin_survives_out_of_contract_q() -> None:
    """N3: the ``[-1, 1]`` clamp on ``q`` had no test.

    ⚑ Honest scope. Once W is clamped to ``[0, rem]`` (S1), the q clamp became
    REDUNDANT for every finite out-of-range q — mutating it away and feeding
    1.5 or inf produces byte-identical output, so a test written that way would
    be vacuous. What the clamp still buys is the invariant itself under a
    non-finite q: without it, ``NaN`` propagates into W and L and the function
    returns a non-simplex triple. Pin the invariant, not the clamp.
    """
    for q in (1.5, -1.5, 1e9, float("inf"), float("-inf"), float("nan")):
        t = q_to_wdl_parametric(q, slope=PROD_SLOPE, draw_width_cp=PROD_WIDTH)
        assert float(t.min()) >= 0.0, (q, t)
        assert float(t.sum()) == pytest.approx(1.0, abs=1e-6), (q, t)
    # and the in-contract endpoints are what the saturating values collapse to
    for over, edge in ((1.5, 1.0), (-1.5, -1.0)):
        assert q_to_wdl_parametric(
            over, slope=PROD_SLOPE, draw_width_cp=PROD_WIDTH,
        ).tobytes() == q_to_wdl_parametric(
            edge, slope=PROD_SLOPE, draw_width_cp=PROD_WIDTH,
        ).tobytes()


@requires_c
def test_the_c_call_refuses_a_bad_mode_or_an_unusable_curve() -> None:
    values = np.zeros(2)
    wdl = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="swdl_draw_mode"):
        _c_batch(wdl_logits=wdl, values=values,
                 extra=(0, _N_EXTRA_V1, 0, 0.0, 0.0, 0.0, 7, PROD_SLOPE, PROD_WIDTH))
    for slope, width in ((0.0, PROD_WIDTH), (PROD_SLOPE, 0.0)):
        with pytest.raises(ValueError, match="cp_slope") as exc:
            _c_batch(wdl_logits=wdl, values=values,
                     extra=(0, _N_EXTRA_V1, 0, 0.0, 0.0, 0.0,
                            SWDL_DRAW_PARAMETRIC_Q, slope, width))
        assert str(exc.value).endswith(f"got {slope:g} and {width:g}")
    for slope, width in ((1e-300, 1e-300), (1.0, 1e3)):
        with pytest.raises(ValueError, match="outside the representable range of sinh") as exc:
            _c_batch(wdl_logits=wdl, values=values,
                     extra=(0, _N_EXTRA_V1, 0, 0.0, 0.0, 0.0,
                            SWDL_DRAW_PARAMETRIC_Q, slope, width))
        assert f"slope*draw_width_cp={slope * width:g} is" in str(exc.value)


@requires_c
def test_a_stale_so_is_refused_by_the_abi_gate() -> None:
    """REVIEW FINDING S3: nothing pinned the ABI bump.

    ``tests/test_mcts_c_tree.py`` asserts ``ABI_VERSION >= _REQUIRED_MCTS_ABI``,
    which ``4 >= 3`` satisfies — so reverting the constant to 3 survived the
    whole suite while silently removing this PR's stale-``.so`` protection. Pin
    the gate by its BEHAVIOUR: an extension at the pre-PR ABI must be refused,
    with the rebuild command in the message. ``_REQUIRED_MCTS_ABI`` is imported
    inside the function so the module object patched below is the one the guard
    reads.
    """
    from types import SimpleNamespace

    import torch

    from chess_anti_engine.mcts import gumbel_c
    from chess_anti_engine.mcts._mcts_tree import ABI_VERSION
    from chess_anti_engine.mcts.gumbel import GumbelConfig

    assert ABI_VERSION == gumbel_c._REQUIRED_MCTS_ABI, (
        "the built extension and the required marker have drifted apart"
    )
    stale = SimpleNamespace(ABI_VERSION=ABI_VERSION - 1)
    original = gumbel_c._mcts_tree_ext
    gumbel_c._mcts_tree_ext = cast(Any, stale)
    try:
        with pytest.raises(RuntimeError, match="build_production_extensions"):
            gumbel_c.run_gumbel_root_many_c(
                cast(torch.nn.Module | None, None), [],
                device="cpu", rng=np.random.default_rng(0), cfg=GumbelConfig(),
            )
    finally:
        gumbel_c._mcts_tree_ext = original


def test_the_stale_worker_gate_is_min_worker_version_not_the_protocol() -> None:
    """REVIEW FINDING S5, and the decision it records: **do NOT bump
    ``PROTOCOL_VERSION``** — the deploy step is a PACKAGE version bump.

    The hazard is real: a pre-#409 worker drops the unknown reco key, is not
    restart-keyed by it, and keeps uploading ``net_raw`` rows into the SAME
    replay window as ``parametric_q`` rows with no column distinguishing them.
    The first fix attempted here was ``PROTOCOL_VERSION`` 2 -> 3. It was WRONG,
    and only the merge check against a moved ``main`` caught it:
    ``_check_worker_compat`` requires EXACT protocol equality, so a bump 426s
    the fleet in BOTH directions of a rolling deploy — ``main`` documents
    exactly this in ``version.py`` and names ``min_worker_version`` as the
    cutover mechanism instead, because it is a ``>=`` comparison.

    So the guard has to be the one that CAN fire without taking selfplay to
    zero. Pin that it is wired: the manifest a worker polls carries
    ``min_worker_version``, and it carries ``PACKAGE_VERSION`` — otherwise the
    "bump the package version in the same deploy" step is decorative and the
    prereg's deploy gating rests on nothing.
    """
    import inspect

    from chess_anti_engine.tune import distributed_runtime

    # The manifest is assembled inline inside `_publish_distributed_trial_state`,
    # which needs a live Trainer — so pin the source, the same way the
    # session-start log line above is pinned.
    src = inspect.getsource(distributed_runtime._publish_distributed_trial_state)
    assert '"min_worker_version": str(PACKAGE_VERSION)' in src, (
        "the published manifest no longer carries min_worker_version derived "
        "from PACKAGE_VERSION — the only stale-worker gate that can fire "
        "without 426-ing the whole fleet, and the one this arm's deploy "
        "gating depends on"
    )


def test_the_mode_int_encoding_has_one_home() -> None:
    """A second copy of the string->int mapping is how the Python and C sides
    would end up selecting different modes from the same config."""
    if not HAS_C:  # pragma: no cover - the dev env hard-requires the extension
        pytest.skip("C extension not available")
    assert _SWDL_DRAW_MODE_TO_C == {
        SEARCH_WDL_DRAW_NET_RAW: SWDL_DRAW_NET_RAW,
        SEARCH_WDL_DRAW_PARAMETRIC_Q: SWDL_DRAW_PARAMETRIC_Q,
    }
    assert set(_SWDL_DRAW_MODE_TO_C) == set(SEARCH_WDL_DRAW_MODES)


# ---------------------------------------------------------------------------
# The config route
# ---------------------------------------------------------------------------


def test_game_config_validates_the_mode_and_the_curve() -> None:
    with pytest.raises(ValueError, match="search_wdl_draw_mode"):
        GameConfig(search_wdl_draw_mode="parametric")
    with pytest.raises(ValueError, match="sf_wdl_cp_slope"):
        GameConfig(
            search_wdl_draw_mode=SEARCH_WDL_DRAW_PARAMETRIC_Q, sf_wdl_cp_slope=0.0,
        )
    with pytest.raises(ValueError, match="sf_wdl_cp_slope"):
        GameConfig(
            search_wdl_draw_mode=SEARCH_WDL_DRAW_PARAMETRIC_Q,
            sf_wdl_cp_draw_width=0.0,
        )
    # the default construction is legal and is the production mode
    assert GameConfig().search_wdl_draw_mode == SEARCH_WDL_DRAW_NET_RAW


def test_ply_path_resolver_defaults_but_never_guesses() -> None:
    from types import SimpleNamespace

    assert _search_wdl_draw_mode(GameConfig()) == SEARCH_WDL_DRAW_NET_RAW
    assert _search_wdl_draw_mode(
        GameConfig(search_wdl_draw_mode=SEARCH_WDL_DRAW_PARAMETRIC_Q),
    ) == SEARCH_WDL_DRAW_PARAMETRIC_Q
    # a GameConfig pickled before the field existed keeps the old semantics
    legacy = cast(Any, SimpleNamespace(sf_wdl_cp_slope=PROD_SLOPE))
    assert _search_wdl_draw_mode(legacy) == SEARCH_WDL_DRAW_NET_RAW
    with pytest.raises(ValueError, match="search_wdl_draw_mode"):
        _search_wdl_draw_mode(cast(Any, SimpleNamespace(search_wdl_draw_mode="raw")))


def test_the_key_is_in_the_yaml_schema() -> None:
    assert "search_wdl_draw_mode" in SELFPLAY_CONFIG_KEYS


def test_trial_config_carries_the_key() -> None:
    assert TrialConfig.from_dict(
        {"search_wdl_draw_mode": "parametric_q"},
    ).search_wdl_draw_mode == "parametric_q"
    assert TrialConfig.from_dict({}).search_wdl_draw_mode == SEARCH_WDL_DRAW_NET_RAW


def test_trial_config_rejects_a_typo_on_the_driver() -> None:
    """REVIEW FINDING S4. ``GameConfig.__post_init__`` validates this too, but
    the driver only builds a ``GameConfig`` when ``eval_games > 0`` and
    production runs 0 — so without a check here a typo is published to the reco
    and kills selfplay in EVERY worker while the driver looks healthy. This is
    the CLAUDE.md category-(b) conversion: fail the trial once, on the driver,
    naming the value."""
    for bad in ("parametric", "net-raw", "", "NET_RAW"):
        with pytest.raises(ValueError, match="search_wdl_draw_mode"):
            TrialConfig.from_dict({"search_wdl_draw_mode": bad})


def _reco(config: dict[str, Any]) -> dict[str, Any]:
    return dict(build_recommended_worker(
        config=config, model_cfg=ModelConfig(), sf_nodes=5000, mcts_simulations=32,
    ))


def _bare_session() -> WorkerSession:
    import logging
    import threading
    from types import SimpleNamespace

    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.search_wdl_draw_mode")
    session.args = cast(Any, SimpleNamespace())
    session.opening_book_path = None
    session.opening_book_path_2 = None
    session.opening_fen_list_path = None
    session._dole_lock = threading.Lock()
    return session


def test_the_mode_is_published_and_reaches_game_config() -> None:
    """The E13 class, end to end: yaml value -> published reco -> GameConfig.

    Also pins that the two curve knobs ride the SAME reco, because the mode is
    meaningless without them.
    """
    cfg = {
        "search_wdl_draw_mode": "parametric_q",
        "sf_wdl_use_cp_logistic": True,
        "sf_wdl_cp_slope": PROD_SLOPE,
        "sf_wdl_cp_draw_width": PROD_WIDTH,
    }
    reco = _reco(cfg)
    assert reco["search_wdl_draw_mode"] == "parametric_q"
    assert reco["sf_wdl_cp_slope"] == PROD_SLOPE
    assert reco["sf_wdl_cp_draw_width"] == PROD_WIDTH

    cfgs, _sf = WorkerSession._build_selfplay_configs(_bare_session(), reco)
    game = cast(GameConfig, cfgs["game"])
    assert game.search_wdl_draw_mode == "parametric_q"
    assert game.search_wdl_draw_mode != GameConfig().search_wdl_draw_mode
    assert game.sf_wdl_cp_slope == PROD_SLOPE
    assert game.sf_wdl_cp_draw_width == PROD_WIDTH


def test_the_published_default_is_off() -> None:
    """REVIEW FINDING S2: the publisher is the ONLY hop that writes a default
    for this key, and nothing pinned it. ``test_the_production_config_does_not_
    ship_the_arm`` reads the YAML; the reco is what actually reaches a worker,
    so a flipped default there would hand ``parametric_q`` to the whole fleet
    from a yaml that never mentions the key."""
    assert _reco({})["search_wdl_draw_mode"] == SEARCH_WDL_DRAW_NET_RAW
    cfgs, _sf = WorkerSession._build_selfplay_configs(_bare_session(), _reco({}))
    assert cast(GameConfig, cfgs["game"]).search_wdl_draw_mode == SEARCH_WDL_DRAW_NET_RAW


def test_an_old_manifest_falls_back_to_the_dataclass_default() -> None:
    cfgs, _sf = WorkerSession._build_selfplay_configs(
        _bare_session(), {"sf_nodes": 5000},
    )
    assert cast(GameConfig, cfgs["game"]).search_wdl_draw_mode == (
        GameConfig().search_wdl_draw_mode
    )


def test_worker_classifies_the_key_as_restart_and_resume_incompatible() -> None:
    """Restart-keyed (baked into the frozen GameConfig at session start) AND
    resume-fingerprinted (a resumed game's early plies carry the other
    construction, and finalize cannot re-derive ``search_wdl``)."""
    assert "search_wdl_draw_mode" in WorkerSession._RECO_RESTART_KEYS
    assert "search_wdl_draw_mode" not in WorkerSession._RECO_LIVE_KEYS
    assert "search_wdl_draw_mode" in WorkerSession._RESUME_COMPAT_KEYS


def test_resume_fingerprint_separates_the_two_modes() -> None:
    session = _bare_session()
    a = WorkerSession._resume_compat_fingerprint(session, {"search_wdl_draw_mode": "net_raw"})
    b = WorkerSession._resume_compat_fingerprint(
        session, {"search_wdl_draw_mode": "parametric_q"},
    )
    assert a != b


def test_session_start_log_line_names_the_draw_mode() -> None:
    """The session-start reco line is the deploy-verification instrument; the
    format string is inline in ``_run_selfplay``, so pin the source."""
    import inspect

    from chess_anti_engine import worker

    src = inspect.getsource(worker)
    start = src.index("session-start reco applied")
    assert "swdl_draw=%s" in src[start:start + 400], (
        "the session-start reco line no longer reports the search_wdl draw mode "
        "— the deploy proof for search_wdl_draw_mode"
    )


def test_the_production_config_does_not_ship_the_arm() -> None:
    """This PR ships the capability, not the deploy. The arm is a training-target
    change and needs its own ledger prereg + restart; the production yaml must
    stay on ``net_raw`` (by absence or by value) until then."""
    raw = yaml.safe_load(_PRODUCTION_YAML.read_text(encoding="utf-8"))
    selfplay = raw.get("selfplay", {}) or {}
    assert selfplay.get("search_wdl_draw_mode", SEARCH_WDL_DRAW_NET_RAW) == (
        SEARCH_WDL_DRAW_NET_RAW
    )
