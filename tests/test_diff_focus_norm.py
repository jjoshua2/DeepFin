"""Scale-free diff-focus normalization (task #171) and Python/C KL parity (#173).

The standing question for every test here is not "is this arithmetic right" but
"does this take effect on the PRODUCTION path, and what observation proves it
did". Production runs the C extension, so the negative control and the
keep_prob/priority assertions all go through ``batch_process_ply`` itself rather
than through a Python re-derivation of it.
"""
from __future__ import annotations

import re
import types
from pathlib import Path
from typing import Any, cast

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.cboard_encode import cboard_from_board_fast
from chess_anti_engine.model import ModelConfig
from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask
from chess_anti_engine.selfplay.config import DiffFocusConfig
from chess_anti_engine.selfplay.diff_focus_norm import DiffFocusNormalizer
from chess_anti_engine.selfplay.network_turn import _policy_kl, _raw_difficulty
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
)
# Deliberately none of them a default, so a publisher line that hard-codes the
# default instead of reading the config still fails.
NORM_VALUES: tuple[Any, ...] = (True, 4096, 512, 0.4, 2.5, 6.0)


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


def test_production_yaml_declares_the_group_default_off() -> None:
    import yaml
    raw = yaml.safe_load(Path("configs/pbt2_small.yaml").read_text(encoding="utf-8"))
    sp = raw["selfplay"]
    for key in NORM_KEYS:
        assert key in sp, f"{key} missing from the production yaml"
    assert sp["diff_focus_norm_enabled"] is False, (
        "this must ship OFF: enabling it changes which plies are recorded"
    )
