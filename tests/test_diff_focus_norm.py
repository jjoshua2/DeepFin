"""Scale-free diff-focus normalization (task #171) and Python/C KL parity (#173).

The standing question for every test here is not "is this arithmetic right" but
"does this take effect on the PRODUCTION path, and what observation proves it
did". Production runs the C extension, so the negative control and the
keep_prob/priority assertions all go through ``batch_process_ply`` itself rather
than through a Python re-derivation of it.
"""
from __future__ import annotations

import re
import sys
import threading
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.cboard_encode import cboard_from_board_fast
from chess_anti_engine.model import ModelConfig
from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.diff_focus_norm import DiffFocusNormalizer
from chess_anti_engine.selfplay.network_turn import (
    _append_records_via_c,
    _policy_kl,
    _raw_difficulty,
)
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.state import SelfplayState
from chess_anti_engine.tune.distributed_runtime import build_recommended_worker
from chess_anti_engine.utils.config_yaml import SELFPLAY_CONFIG_KEYS
from chess_anti_engine.worker import WorkerSession

try:
    from chess_anti_engine.mcts._mcts_tree import batch_process_ply as _bpp
    HAS_C = True
except ImportError:  # pragma: no cover - the dev env hard-requires the extension
    _bpp = None
    HAS_C = False

requires_c = pytest.mark.skipif(not HAS_C, reason="C extension not available")

NORM_KEYS = (
    "diff_focus_norm_enabled", "diff_focus_norm_window", "diff_focus_norm_warmup",
    "diff_focus_norm_quantile", "diff_focus_norm_slope", "diff_focus_norm_clip",
    "diff_focus_norm_shared",
)
# Deliberately none of them a default, so a publisher line that hard-codes the
# default instead of reading the config still fails.
NORM_VALUES: tuple[Any, ...] = (True, 4096, 512, 0.4, 2.5, 6.0, True)


def _reco() -> dict[str, object]:
    """The manifest the server would publish for a config that sets all six."""
    return build_recommended_worker(
        config=dict(zip(NORM_KEYS, NORM_VALUES, strict=True)),
        model_cfg=ModelConfig(),
        sf_nodes=5000,
        mcts_simulations=32,
    )


# ── fixtures: real positions, real legal masks ──────────────────────────────

def _positions(n: int = 24) -> list[chess.Board]:
    """A spread of real positions, reached by a fixed pseudo-random walk."""
    rng = np.random.default_rng(11)
    out: list[chess.Board] = []
    board = chess.Board()
    while len(out) < n:
        if board.is_game_over() or len(board.move_stack) > 60:
            board = chess.Board()
        moves = list(board.legal_moves)
        board.push(moves[int(rng.integers(len(moves)))])
        if len(board.move_stack) >= 4 and board.legal_moves.count() > 3:
            out.append(board.copy())
    return out


def _batch(boards: list[chess.Board], *, sparse_visits: bool, seed: int = 5) -> dict:
    """Build one ply batch: priors from random logits, visits from a real-shaped
    sparse visit distribution over a subset of legal moves."""
    rng = np.random.default_rng(seed)
    n = len(boards)
    pol = rng.standard_normal((n, POLICY_SIZE)).astype(np.float32) * 2.0
    wdl = rng.standard_normal((n, 3)).astype(np.float32)
    probs = np.zeros((n, POLICY_SIZE), dtype=np.float32)
    actions = np.zeros(n, dtype=np.int32)
    values = rng.uniform(-0.9, 0.9, size=n).astype(np.float64)
    for i, b in enumerate(boards):
        legal = np.flatnonzero(legal_move_mask(b))
        # Gumbel search visits only a handful of candidates, so most legal moves
        # carry exactly zero visit mass -- the case where the two KL
        # implementations used to disagree.
        k = max(2, len(legal) // 4) if sparse_visits else len(legal)
        chosen = rng.choice(legal, size=k, replace=False)
        v = rng.integers(1, 40, size=k).astype(np.float32)
        probs[i, chosen] = v / v.sum()
        actions[i] = int(chosen[0])
    # NOT cboards: batch_process_ply PUSHES the chosen move onto every board it
    # is given, so a shared list would leave the second call of a paired
    # comparison looking at different positions than the first.
    return {
        "boards": boards,
        "pol": pol, "wdl": wdl, "probs": probs, "actions": actions, "values": values,
    }


def _run_c(bt: dict, *, q_w=6.0, pol_s=3.5, df_min=0.025, slope=3.0,
           norm_scale=0.0, norm_slope=0.0, norm_clip=0.0, extra_args=True) -> dict:
    assert _bpp is not None
    args: list[Any] = [
        [cboard_from_board_fast(b) for b in bt["boards"]],
        bt["pol"], bt["wdl"], bt["actions"], bt["values"], bt["probs"],
        1, q_w, pol_s, df_min, slope,
    ]
    if extra_args:
        args += [0, 34, 0, norm_scale, norm_slope, norm_clip]
    res = cast(Any, _bpp(*args))
    return {
        "priority": np.asarray(res[4], dtype=np.float64),
        "kl": np.asarray(res[5], dtype=np.float64),
        "q_delta": np.asarray(res[6], dtype=np.float64),
        "keep": np.asarray(res[7], dtype=np.float64),
        "mask": np.asarray(res[8]),
    }


# ── the estimator itself ────────────────────────────────────────────────────

def test_normalizer_is_scale_free() -> None:
    """The whole point: d and c*d must produce the same normalized values."""
    rng = np.random.default_rng(3)
    d = np.abs(rng.lognormal(0.0, 1.2, size=4000))
    for c in (0.01, 0.1, 10.0, 100.0, 1000.0):
        a = DiffFocusNormalizer(window=4096, warmup=512, quantile=0.5)
        b = DiffFocusNormalizer(window=4096, warmup=512, quantile=0.5)
        a.observe(d)
        b.observe(d * c)
        assert a.armed
        assert b.armed
        assert b.scale == pytest.approx(a.scale * c, rel=1e-12)
        np.testing.assert_allclose(d / a.scale, (d * c) / b.scale, rtol=1e-12)


def test_warmup_reports_zero_and_is_distinguishable_from_armed() -> None:
    n = DiffFocusNormalizer(window=256, warmup=100, quantile=0.5)
    assert not n.armed
    assert n.scale == 0.0
    assert n.count == 0
    n.observe(np.full(99, 2.0))
    # 0.0 is the caller's "normalization off" sentinel: warm-up takes the
    # ORIGINAL unnormalized branch, not a third half-configured behaviour.
    assert not n.armed
    assert n.scale == 0.0
    assert n.count == 99
    n.observe(np.full(1, 2.0))
    assert n.armed
    assert n.scale == pytest.approx(2.0)
    assert n.count == 100


def test_ring_buffer_forgets_the_old_scale() -> None:
    """A scale change must be tracked, not averaged with history forever."""
    n = DiffFocusNormalizer(window=1000, warmup=100, quantile=0.5)
    n.observe(np.full(1000, 1.0))
    assert n.scale == pytest.approx(1.0)
    n.observe(np.full(1000, 10.0))
    assert n.scale == pytest.approx(10.0)


def test_degenerate_window_stays_unarmed_rather_than_dividing_by_zero() -> None:
    n = DiffFocusNormalizer(window=256, warmup=10, quantile=0.5)
    n.observe(np.zeros(256))
    assert not n.armed
    assert n.scale == 0.0


def test_non_finite_difficulties_are_dropped_not_folded_in() -> None:
    n = DiffFocusNormalizer(window=256, warmup=4, quantile=0.5)
    n.observe(np.array([1.0, np.inf, 1.0, np.nan, 1.0, 1.0]))
    assert n.armed
    assert n.scale == pytest.approx(1.0)


@pytest.mark.parametrize(("window", "quantile"), [(0, 0.5), (-1, 0.5), (10, 0.0), (10, 1.0)])
def test_invalid_estimator_config_is_rejected_loudly(window: int, quantile: float) -> None:
    with pytest.raises(ValueError, match="diff_focus_norm_"):
        DiffFocusNormalizer(window=window, warmup=1, quantile=quantile)


# ── the production (C) path ─────────────────────────────────────────────────

@requires_c
def test_default_is_bit_identical_through_the_c_path() -> None:
    """Default OFF must not perturb a single float, including for callers that
    never pass the new arguments at all."""
    bt = _batch(_positions(), sparse_visits=True)
    old = _run_c(bt, extra_args=False)
    new = _run_c(bt, norm_scale=0.0, norm_slope=1.62, norm_clip=8.0)
    for key in ("priority", "kl", "q_delta", "keep"):
        np.testing.assert_array_equal(old[key], new[key], err_msg=key)


@requires_c
def test_normalization_actually_changes_keep_prob_and_priority_in_c() -> None:
    """Execution proof that the scalar reaches the production path and is USED.

    A knob that is accepted and then ignored is this codebase's signature
    defect; this is the observation that rules it out.
    """
    bt = _batch(_positions(), sparse_visits=True)
    off = _run_c(bt)
    scale = float(np.median(off["priority"]))
    on = _run_c(bt, norm_scale=scale, norm_slope=1.62, norm_clip=8.0)

    assert not np.allclose(off["keep"], on["keep"]), "keep_prob did not move"
    assert not np.allclose(off["priority"], on["priority"]), "priority did not move"
    # priority is exactly difficulty/scale, capped.
    np.testing.assert_allclose(
        on["priority"], np.minimum(off["priority"] / scale, 8.0), rtol=1e-5,
    )
    np.testing.assert_allclose(
        on["keep"],
        np.clip(off["priority"] / scale * 1.62, 0.025, 1.0),
        rtol=1e-5,
    )


@requires_c
def test_norm_clip_bounds_the_stored_priority() -> None:
    """The clip is what makes the armed state provable from progress.csv:
    `diff_focus_priority_max <= diff_focus_norm_clip` cannot hold unarmed."""
    bt = _batch(_positions(), sparse_visits=True)
    off = _run_c(bt)
    scale = float(np.median(off["priority"])) / 50.0  # force the cap to bind
    on = _run_c(bt, norm_scale=scale, norm_slope=1.62, norm_clip=8.0)
    assert on["priority"].max() == pytest.approx(8.0)
    uncapped = _run_c(bt, norm_scale=scale, norm_slope=1.62, norm_clip=0.0)
    assert uncapped["priority"].max() > 8.0


@requires_c
def test_zero_norm_slope_with_an_armed_scale_is_refused() -> None:
    """A half-populated config would clamp every keep_prob to min_keep and
    silently discard ~97.5% of plies. It must fail loudly instead."""
    bt = _batch(_positions(), sparse_visits=True)
    with pytest.raises(ValueError, match="df_norm_slope"):
        _run_c(bt, norm_scale=1.0, norm_slope=0.0)


# ── THE NEGATIVE CONTROL ────────────────────────────────────────────────────

@requires_c
def test_negative_control_scaling_difficulty_leaves_the_normalized_regime_invariant() -> None:
    """Shift the SCALE of `difficulty` by 10x (and 100x) on identical positions
    and require the realized keep-rate to be INVARIANT.

    This is the entire point of the change: the 2026-08-09 incident was an
    unrelated search change moving the scale of `kl` under a fixed clamp. The
    scale shift is applied the same way the incident applied it -- by changing
    what `difficulty` evaluates to on the same data -- and the un-normalized arm
    is asserted to MOVE, so a fix that failed the control could not pass by
    accident of an inert test.
    """
    boards = _positions(48)
    bt = _batch(boards, sparse_visits=True)

    raw_keep, norm_keep, norm_prio = [], [], []
    # Spans 5000x and includes the 10x the brief names. It has to bracket 1.0
    # DOWNWARD as well: the raw clamp saturates at the top, so an upward-only
    # sweep would leave the un-normalized arm pinned near 1.0 and the control
    # would look invariant for the wrong reason.
    for c in (0.02, 1.0, 10.0, 100.0):
        # Scaling both weights scales `difficulty` by exactly c.
        off = _run_c(bt, q_w=6.0 * c, pol_s=3.5 * c)
        est = DiffFocusNormalizer(window=4096, warmup=8, quantile=0.5)
        est.observe(off["priority"])
        on = _run_c(
            bt, q_w=6.0 * c, pol_s=3.5 * c,
            norm_scale=est.scale, norm_slope=1.62, norm_clip=8.0,
        )
        raw_keep.append(float(off["keep"].mean()))
        norm_keep.append(float(on["keep"].mean()))
        norm_prio.append(float(on["priority"].mean()))

    # The un-normalized arm must move, or this control proves nothing.
    assert max(raw_keep) - min(raw_keep) > 0.5, (
        f"the raw keep-rate did not move under a 5000x scale shift ({raw_keep}); "
        "the control is inert and cannot discriminate"
    )
    # The normalized arm must not move at all.
    for k, pr in zip(norm_keep[1:], norm_prio[1:], strict=True):
        assert k == pytest.approx(norm_keep[0], rel=1e-6), norm_keep
        assert pr == pytest.approx(norm_prio[0], rel=1e-6), norm_prio


# ── task #173: Python and C must compute the SAME KL ────────────────────────

@requires_c
def test_python_and_c_kl_agree_on_sparse_targets() -> None:
    """Production runs C. The Python fallback must not compute a different
    training signal on the rows where a search target is sparse."""
    boards = _positions(32)
    bt = _batch(boards, sparse_visits=True)
    c = _run_c(bt)
    for i, b in enumerate(boards):
        mask = legal_move_mask(b)
        lg = bt["pol"][i].astype(np.float64).copy()
        lg[~mask] = -1e9
        lg -= lg.max()
        e = np.exp(lg)
        e[~mask] = 0.0
        prior = e / e.sum()
        assert _policy_kl(prior, bt["probs"][i], mask) == pytest.approx(
            float(c["kl"][i]), rel=2e-4, abs=1e-5,
        )


@requires_c
def test_the_old_floored_kl_really_did_diverge() -> None:
    """Guards the test above from being vacuous.

    If the pre-fix formula also agreed with C, `test_python_and_c_kl_agree...`
    would pass on a broken implementation too. Measured on 2,400 real
    production rows the two disagreed on 51-60% of them, by up to 23.29 nats.
    """
    boards = _positions(32)
    bt = _batch(boards, sparse_visits=True)
    c = _run_c(bt)
    diverged = 0
    for i, b in enumerate(boards):
        mask = legal_move_mask(b)
        lg = bt["pol"][i].astype(np.float64).copy()
        lg[~mask] = -1e9
        lg -= lg.max()
        e = np.exp(lg)
        e[~mask] = 0.0
        prior = e / e.sum()
        # The pre-fix network_turn.py formula: floor BOTH at 1e-12, sum over ALL.
        imp = np.maximum(bt["probs"][i].astype(np.float64), 1e-12)
        pc = np.maximum(prior, 1e-12)
        old = float(np.sum(pc * (np.log(pc) - np.log(imp))))
        if abs(old - float(c["kl"][i])) > 1e-3:
            diverged += 1
    assert diverged > len(boards) // 2, (
        f"only {diverged}/{len(boards)} rows diverged under the OLD formula; "
        "the sparse fixture no longer reproduces the defect, so the parity "
        "test above has stopped proving anything"
    )


@requires_c
def test_python_reconstruction_matches_the_c_priority() -> None:
    """The estimator is fed a Python re-derivation of the raw difficulty from
    C's raw kl/q_delta columns. If that drifts from what C used, the normalizer
    silently normalizes by the wrong population."""
    bt = _batch(_positions(32), sparse_visits=True)
    c = _run_c(bt)
    df = DiffFocusConfig()
    np.testing.assert_allclose(
        _raw_difficulty(c["q_delta"], c["kl"], df), c["priority"], rtol=1e-6, atol=1e-7,
    )


@requires_c
def test_raw_kl_and_q_delta_columns_stay_raw_under_normalization() -> None:
    """`replay_pmass_kl_raw_mean` (task #172) and the estimator both read these
    columns expecting config-free units."""
    bt = _batch(_positions(), sparse_visits=True)
    off = _run_c(bt)
    on = _run_c(bt, norm_scale=0.37, norm_slope=1.62, norm_clip=8.0)
    np.testing.assert_array_equal(off["kl"], on["kl"])
    np.testing.assert_array_equal(off["q_delta"], on["q_delta"])


# ── the knob must actually reach the worker ─────────────────────────────────

def test_every_norm_key_is_declared_and_published_to_the_worker() -> None:
    """The five ORIGINAL diff_focus keys are famously not read by the worker
    (tests/test_reco_coverage.py). These six must be, or the fix is a knob that
    never reaches the process that computes keep_prob."""
    reco = _reco()
    for key in NORM_KEYS:
        assert key in SELFPLAY_CONFIG_KEYS, f"{key} is not a declared selfplay key"
        assert key in reco, f"{key} never reaches the worker manifest"
    assert reco["diff_focus_norm_enabled"] is True
    assert reco["diff_focus_norm_window"] == 4096
    assert reco["diff_focus_norm_warmup"] == 512
    assert reco["diff_focus_norm_quantile"] == pytest.approx(0.4)
    assert reco["diff_focus_norm_slope"] == pytest.approx(2.5)
    assert reco["diff_focus_norm_clip"] == pytest.approx(6.0)
    assert reco["diff_focus_norm_shared"] is True


def test_the_worker_turns_the_reco_into_a_diff_focus_config() -> None:
    """Closes the last link by EXECUTION: the worker's own config builder, run
    on a manifest, must yield a DiffFocusConfig carrying these values."""
    stub = types.SimpleNamespace(
        args=types.SimpleNamespace(),
        opening_book_path=None, opening_book_path_2=None,
        opening_fen_list_path=None,
    )
    for name in ("_resolve_reco", "_require_reco"):
        setattr(stub, name, types.MethodType(getattr(WorkerSession, name), stub))
    reco = _reco()
    build = cast(Any, WorkerSession)._build_selfplay_configs
    cfgs, _ = build(stub, reco)
    df = cfgs["diff_focus"]
    assert isinstance(df, DiffFocusConfig)
    assert df.norm_enabled is True
    assert df.norm_window == 4096
    assert df.norm_warmup == 512
    assert df.norm_quantile == pytest.approx(0.4)
    assert df.norm_slope == pytest.approx(2.5)
    assert df.norm_clip == pytest.approx(6.0)
    assert df.norm_shared is True
    # The five ORIGINAL keys stay at their dataclass defaults -- this PR does
    # not newly arm them, and a future change that does needs its own entry.
    assert (df.enabled, df.q_weight, df.pol_scale, df.slope, df.min_keep) == (
        DiffFocusConfig().enabled, DiffFocusConfig().q_weight,
        DiffFocusConfig().pol_scale, DiffFocusConfig().slope,
        DiffFocusConfig().min_keep,
    )


def test_worker_reads_each_norm_key_by_name() -> None:
    """A refactor that drops one `reco.get` line would leave that knob dead
    while every other test still passes."""
    src = Path("chess_anti_engine/worker.py").read_text(encoding="utf-8")
    block = src[src.index('"diff_focus": DiffFocusConfig('):]
    block = block[: block.index('"game": GameConfig(')]
    for key in NORM_KEYS:
        assert re.search(rf'"{key}"', block), f"worker.py never reads {key}"


# ── the LAST link: config -> SelfplayState -> the stored record ─────────────
#
# ⚑ Everything above stops at ``DiffFocusConfig``. Nothing above observes the
# selfplay session turning that config into a live estimator, or the estimator's
# scale reaching ``batch_process_ply``'s argument list. That gap was real: an
# independent review mutated ``SelfplayState.create`` to pass
# ``diff_focus_norm=None`` unconditionally -- i.e. ``norm_enabled`` accepted and
# silently ignored, this codebase's signature defect, in the one function that
# decides whether the feature exists -- and the ENTIRE suite stayed green. These
# three tests are that mutation's kill.

_APPEND_BATCH = 6


def _session_state(
    diff_focus: DiffFocusConfig,
    *,
    diff_focus_norm: DiffFocusNormalizer | None = None,
) -> SelfplayState:
    """A real ``SelfplayState``, built the way ``play_batch`` builds one."""
    evaluator = Mock(spec=["evaluate_encoded"])
    stockfish = Mock(spec=["search", "nodes"])
    stockfish.nodes = 0
    return SelfplayState.create(
        diff_focus_norm=diff_focus_norm,
        model=None,
        device="cpu",
        rng=np.random.default_rng(0),
        stockfish=stockfish,
        evaluator=evaluator,
        batch_size=_APPEND_BATCH,
        continuous=False,
        target=_APPEND_BATCH,
        opponent=OpponentConfig(),
        temp=TemperatureConfig(),
        search=SearchConfig(),
        opening=OpeningConfig(),
        diff_focus=diff_focus,
        game=GameConfig(),
    )


def _append_one_ply(
    state: SelfplayState, diff_focus: DiffFocusConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the REAL C record-append path once; return stored priority/keep_prob.

    Not ``_run_c``: this goes through ``_append_records_via_c``, which is the
    function that reads ``state.diff_focus_norm`` and assembles the C argument
    list. A test that calls ``batch_process_ply`` directly passes the scale in
    by hand and therefore cannot see the plumbing between the two.
    """
    n = _APPEND_BATCH
    rng = np.random.default_rng(7)
    pol = (rng.standard_normal((n, POLICY_SIZE)) * 2.0).astype(np.float32)
    wdl = rng.standard_normal((n, 3)).astype(np.float32)
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
        values.append(float(rng.uniform(-0.9, 0.9)))
    _append_records_via_c(
        state, list(range(n)),
        cb_encode_list=list(state.cboards[:n]),
        pol_logits=pol, wdl_logits_raw=wdl,
        actions=actions, values_list=values,
        probs_list=[probs[i] for i in range(n)],
        gumbel_diags=[None] * n, is_full_py=[True] * n,
        sample_weights=[1.0] * n, diff_focus=diff_focus,
    )
    recs = [state.samples_per_game[i][0] for i in range(n)]
    return (
        np.array([float(r.priority) for r in recs]),
        np.array([float(r.keep_prob) for r in recs]),
    )


def test_selfplay_state_builds_the_normalizer_only_when_the_knob_is_on() -> None:
    """``norm_enabled`` must decide whether the estimator EXISTS, and the three
    estimator knobs must reach the object rather than its defaults."""
    assert _session_state(DiffFocusConfig()).diff_focus_norm is None
    est = _session_state(
        DiffFocusConfig(
            norm_enabled=True, norm_window=64, norm_warmup=8, norm_quantile=0.4,
        ),
    ).diff_focus_norm
    assert est is not None
    est.observe(np.arange(1, 65, dtype=np.float64))
    # q0.40 of 1..64 is 26.2; q0.50 (the default the knob must NOT have fallen
    # back to) is 32.5, and a 8192-wide default ring would never have armed.
    assert est.armed
    assert est.scale == pytest.approx(26.2)


def test_selfplay_state_refuses_an_armed_group_with_a_zero_slope() -> None:
    """``_mcts_tree.c`` refuses this combination, but only on the first ply
    batch after warm-up -- thousands of games in, naming a C argument rather
    than the yaml key. `a dead knob can arm a crash`: refuse at construction."""
    with pytest.raises(ValueError, match="diff_focus_norm_slope"):
        _session_state(DiffFocusConfig(norm_enabled=True, norm_slope=0.0))


@requires_c
def test_the_record_append_path_stores_the_normalized_priority_and_keep_prob() -> None:
    """THE end-to-end observation: an armed session stores different numbers.

    Reads the ``_NetRecord``s that finalize.py later consumes -- ``priority`` is
    the sampler's column and ``keep_prob`` is the drop probability -- so this
    fails if the scale never reaches the C call, whatever the config objects say.
    """
    off = DiffFocusConfig()
    raw_prio, _raw_keep = _append_one_ply(_session_state(off), off)

    on = DiffFocusConfig(
        norm_enabled=True, norm_window=64, norm_warmup=8,
        norm_slope=1.62, norm_clip=8.0,
    )
    state = _session_state(on)
    assert state.diff_focus_norm is not None
    state.diff_focus_norm.observe(np.full(32, 0.5))
    assert state.diff_focus_norm.scale == pytest.approx(0.5)
    prio, keep = _append_one_ply(state, on)

    assert not np.allclose(raw_prio, prio), "the estimator never reached the C call"
    np.testing.assert_allclose(prio, np.minimum(raw_prio / 0.5, 8.0), rtol=1e-5)
    np.testing.assert_allclose(
        keep, np.clip(raw_prio / 0.5 * 1.62, 0.025, 1.0), rtol=1e-5,
    )
    # The window is also FED by this path, in RAW units: 32 pre-loaded + the
    # policy-bearing rows of this ply. A normalized value fed back would pin the
    # reference at 1.0 for ever.
    assert state.diff_focus_norm.count == 32 + _APPEND_BATCH
    assert state.diff_focus_norm.scale == pytest.approx(0.5)


@requires_c
def test_an_unarmed_session_records_exactly_the_pre_fix_numbers() -> None:
    """Warm-up is the OLD behaviour, not a third one -- asserted on the stored
    record, not on the sentinel that is supposed to produce it."""
    off = DiffFocusConfig()
    raw_prio, raw_keep = _append_one_ply(_session_state(off), off)
    on = DiffFocusConfig(norm_enabled=True, norm_window=64, norm_warmup=64)
    state = _session_state(on)
    prio, keep = _append_one_ply(state, on)
    np.testing.assert_array_equal(prio, raw_prio)
    np.testing.assert_array_equal(keep, raw_keep)
    assert state.diff_focus_norm is not None
    assert not state.diff_focus_norm.armed


def test_production_yaml_declares_the_group_default_off() -> None:
    import yaml
    raw = yaml.safe_load(Path("configs/pbt2_small.yaml").read_text(encoding="utf-8"))
    sp = raw["selfplay"]
    for key in NORM_KEYS:
        assert key in sp, f"{key} missing from the production yaml"
    assert sp["diff_focus_norm_enabled"] is False, (
        "this must ship OFF: enabling it changes which plies are recorded"
    )
    assert sp["diff_focus_norm_shared"] is False, (
        "this must ship OFF too: it moves the post-restart transient's rows off "
        "the raw-unit branch, which is a data-affecting change"
    )


# ── W4: the warm-up transient, and the per-THREAD estimator behind it ────────
#
# ⚑ The cost this section is about is invisible in the yaml. `norm_warmup` is
# per ESTIMATOR and one estimator is built per `SelfplayState`, i.e. per
# `play_batch` call — and the worker runs one `play_batch` per
# `--selfplay-threads` (32 live, 4 workers => 128 estimators => 131,072 plies of
# warm-up per restart). Measured 2026-08-12 on the live replay window: ~160k of
# 1,498,168 rows written on the UNARMED branch across the two restart transients
# it held — raw units with `diff_focus_slope` against an 11.7x-moved KL scale and
# an unclipped priority (max 17.59 against `norm_clip` 8.0), each transient not
# clear until ~37 minutes in. progress.csv shows the same events independently:
# `diff_focus_priority_max` above the clip for iters 1-8 after one restart, 1-23
# after the previous one, and again mid-run at iter ~140.
# `diff_focus_norm_shared` divides both costs by the thread count.


def test_play_batch_adopts_a_supplied_estimator_instead_of_building_one() -> None:
    """`play_batch(diff_focus_norm=...)` must reach the state's estimator slot.

    A parameter accepted and then dropped on the floor is this codebase's
    signature defect; the observation that rules it out is object IDENTITY on
    the state the session actually uses.
    """
    df = DiffFocusConfig(norm_enabled=True, norm_window=64, norm_warmup=8)
    shared = DiffFocusNormalizer(window=64, warmup=8, quantile=0.5)
    assert _session_state(df, diff_focus_norm=shared).diff_focus_norm is shared

    # ...and the default still builds a private one, so the off path is
    # unchanged rather than merely equivalent.
    a = _session_state(df).diff_focus_norm
    b = _session_state(df).diff_focus_norm
    assert a is not None
    assert b is not None
    assert a is not b


def test_play_batch_forwards_the_estimator_to_the_state_it_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ The test above calls ``SelfplayState.create`` directly, so it cannot see
    ``play_batch`` accepting the parameter and handing ``None`` down — which is
    precisely this codebase's signature defect and which a mutation proved that
    test survives. Observe the argument at the seam instead, and stop the session
    there rather than booting a whole selfplay batch for one kwarg.
    """
    import chess_anti_engine.selfplay.manager as manager_mod

    class _Stop(Exception):
        pass

    seen: list[object] = []
    real_create = SelfplayState.create

    def _spy(**kwargs: Any) -> SelfplayState:
        seen.append(kwargs.get("diff_focus_norm"))
        raise _Stop

    monkeypatch.setattr(manager_mod.SelfplayState, "create", staticmethod(_spy))
    assert real_create is not None  # the real one is restored by monkeypatch

    df = DiffFocusConfig(norm_enabled=True, norm_window=64, norm_warmup=8)
    shared = DiffFocusNormalizer(window=64, warmup=8, quantile=0.5)
    common: dict[str, Any] = {
        "device": "cpu", "rng": np.random.default_rng(0),
        "stockfish": Mock(spec=["search", "nodes"]),
        "evaluator": Mock(spec=["evaluate_encoded"]),
        "games": 2, "target_games": 2, "diff_focus": df,
    }
    with pytest.raises(_Stop):
        manager_mod.play_batch(None, diff_focus_norm=shared, **common)
    with pytest.raises(_Stop):
        manager_mod.play_batch(None, **common)
    assert seen == [shared, None]


def test_a_supplied_estimator_with_the_group_off_is_refused() -> None:
    """An estimator that is fed and never read is worse than none: it looks
    armed in a debugger while the C path takes the unnormalized branch."""
    est = DiffFocusNormalizer(window=8, warmup=2, quantile=0.5)
    with pytest.raises(ValueError, match="diff_focus_norm_enabled"):
        _session_state(DiffFocusConfig(norm_enabled=False), diff_focus_norm=est)


@requires_c
def test_a_shared_estimator_arms_once_for_all_its_states() -> None:
    """THE point of the knob, read off the stored records.

    Two states sharing one estimator: the second state's rows are NORMALIZED
    even though that state has appended nothing of its own, because the first
    state's plies already armed the shared window. With private estimators the
    second state is still in warm-up and stores RAW priorities.
    """
    df = DiffFocusConfig(
        norm_enabled=True, norm_window=64, norm_warmup=_APPEND_BATCH,
        norm_slope=1.62, norm_clip=8.0,
    )
    shared = DiffFocusNormalizer(window=64, warmup=_APPEND_BATCH, quantile=0.5)
    first = _session_state(df, diff_focus_norm=shared)
    raw_prio, raw_keep = _append_one_ply(first, df)          # arms the window
    assert shared.armed
    second = _session_state(df, diff_focus_norm=shared)
    shared_prio, shared_keep = _append_one_ply(second, df)

    priv_prio, priv_keep = _append_one_ply(_session_state(df), df)

    # The private estimator's first ply is still warming up => raw, unchanged.
    np.testing.assert_array_equal(priv_prio, raw_prio)
    np.testing.assert_array_equal(priv_keep, raw_keep)
    # The shared one is armed => normalized, and demonstrably different.
    assert not np.allclose(shared_prio, priv_prio)
    np.testing.assert_allclose(
        shared_prio, np.minimum(raw_prio / shared.scale, 8.0), rtol=1e-5,
    )
    np.testing.assert_allclose(
        shared_keep,
        np.clip(raw_prio / shared.scale * 1.62, 0.025, 1.0), rtol=1e-5,
    )


def test_concurrent_observe_does_not_lose_ring_writes() -> None:
    """`norm_shared` means 32 selfplay threads call `observe` on one instance.

    Without the lock two calls read the same `_pos`, write the SAME slots and
    both advance `_count`, so the ring keeps its 0.0 fill in the tail while the
    estimator believes it is full — a depressed reference quantile, i.e.
    INFLATED priorities, which is the failure this module exists to prevent.
    Parameters are not arbitrary. Exactly ``window`` rows are observed in total,
    so a lost ring write leaves a slot at its 0.0 fill with nothing to overwrite
    it; the chunk is small and the threads many so the read-modify-write on
    ``_count`` is contended. Measured on the unlocked ring at these settings:
    ``count`` lands at 10,784-12,816 of 16,384 on 5 of 5 runs (so the estimator
    never even arms), against 16,384 every time with the lock.
    """
    window, n_threads, chunk_rows = 16384, 16, 8
    est = DiffFocusNormalizer(window=window, warmup=window, quantile=0.5)
    chunk = np.full(chunk_rows, 3.0, dtype=np.float64)
    per_thread = window // (n_threads * chunk_rows)

    def _hammer() -> None:
        for _ in range(per_thread):
            est.observe(chunk)

    # The unlocked race is a genuine interleaving, not a certainty: force the
    # interpreter to preempt inside the ring update rather than hoping it does.
    # Restored in `finally` so nothing else in the session runs at 1us.
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=_hammer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old_interval)

    assert est.count == window
    # Every observed value is 3.0, so any slot the ring failed to write is a 0.0
    # and drags the median below 3.0.
    assert est.scale == pytest.approx(3.0)


def test_the_worker_shares_one_estimator_across_every_selfplay_thread() -> None:
    """The last link, on the worker's own code path.

    `_build_shared_diff_focus_norm` is what turns `diff_focus_norm_shared` into
    an object; if it returns None with the knob on, the knob is accepted and
    silently ignored — and every other test in this file still passes, because
    they all stop at `DiffFocusConfig`.
    """
    stub = types.SimpleNamespace(
        args=types.SimpleNamespace(threaded_selfplay=True, selfplay_threads=32),
        log=Mock(),
    )
    build = cast(Any, WorkerSession)._build_shared_diff_focus_norm

    on = DiffFocusConfig(norm_enabled=True, norm_shared=True, norm_warmup=1024)
    assert isinstance(build(stub, {"diff_focus": on}, 384), DiffFocusNormalizer)

    # OFF (the shipped default) must yield None, i.e. every state builds its own.
    for df in (
        DiffFocusConfig(),
        DiffFocusConfig(norm_enabled=True, norm_shared=False),
        # shared without the group enabled: nothing to share, and a warning
        # rather than a silently dead knob.
        DiffFocusConfig(norm_enabled=False, norm_shared=True),
    ):
        assert build(stub, {"diff_focus": df}, 384) is None
    assert stub.log.warning.called

    # The validation the shared build must NOT route around.
    with pytest.raises(ValueError, match="diff_focus_norm_slope"):
        build(
            stub,
            {"diff_focus": DiffFocusConfig(
                norm_enabled=True, norm_shared=True, norm_slope=0.0)},
            384,
        )


def test_the_threaded_dispatch_hands_every_thread_the_same_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_run_selfplay_threaded` is the site that multiplies the warm-up by the
    thread count, so it is the site that has to be observed. Captures the
    `diff_focus_norm` each thread's `play_batch` receives."""
    import chess_anti_engine.worker as worker_mod

    seen: list[object] = []
    seen_lock = threading.Lock()

    def _fake_play_batch(_model: Any, **kwargs: Any) -> tuple[list[Any], Any]:
        with seen_lock:
            seen.append(kwargs.get("diff_focus_norm"))
        return [], "stats"

    monkeypatch.setattr(worker_mod, "play_batch", _fake_play_batch)
    stub = types.SimpleNamespace(
        args=types.SimpleNamespace(selfplay_threads=4),
        rng=np.random.default_rng(0),
        log=Mock(),
        device="cpu",
        model=None,
        _upload_buf_lock=None,
        _on_completed_game=None, _record_selfplay_phase_timing=None,
        _check_model_update=None, _register_live_state=None,
        _resume_inflight_enabled=False, _suspend_inflight_games=None,
        _stop_fn=None, _pause_fn=None,
        _clear_live_states=lambda: None,
        _aggregate_thread_stats=lambda stats: stats,
    )
    run = cast(Any, WorkerSession)._run_selfplay_threaded
    shared = DiffFocusNormalizer(window=64, warmup=8, quantile=0.5)

    run(stub, games_per_batch=8, sf=None, eval_=None, cfgs={},
        diff_focus_norm=shared)
    assert len(seen) == 4
    assert all(s is shared for s in seen), "a thread got its own estimator"
    seen.clear()
    run(stub, games_per_batch=8, sf=None, eval_=None, cfgs={})
    assert seen == [None] * 4, "the default path must not share"
