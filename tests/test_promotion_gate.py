"""Tests for the anchored promotion gate.

Every test here is written to fail if the gate stops TAKING EFFECT, not merely
if its arithmetic drifts. Each one names the mutation it catches; the PR body
records the mutation run that proved each name honest.
"""
from __future__ import annotations

import ast
import csv
import dataclasses
import inspect
import json
import logging
import math
import random
import statistics as st
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch

from chess_anti_engine.model import ModelConfig
from chess_anti_engine.moves.encode import POLICY_SIZE
from chess_anti_engine.replay import ArrayReplayBuffer
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import LOCAL_SHARD_SUFFIX, save_local_shard_arrays
from chess_anti_engine.tune.distributed_runtime import (
    _empty_ingest_summary,
    _ingest_distributed_selfplay,
    _publish_distributed_trial_state,
)
from chess_anti_engine.tune import promotion_gate
from chess_anti_engine.tune.promotion_gate import (
    _CADENCE_RATIO_MAX,
    _CADENCE_RATIO_MIN,
    _DEGENERATE_SE_ULPS,
    _window_is_degenerate,
    DECISION_DEMOTE,
    DECISION_NOT_RUN,
    DECISION_PROMOTE,
    ELO_PER_SCORE_AT_HALF,
    MODE_ENFORCE,
    MODE_OFF,
    MODE_SHADOW,
    OFFLINE,
    READOUT_HOLD,
    READOUT_KILL,
    READOUT_PROMOTE,
    _READOUT_MIN_ROWS,
    ShadowReadout,
    AnchoredSample,
    GateConfig,
    GateDecision,
    GateHoldController,
    PromotionGate,
    _t_quantile,
    _Z_ONE_SIDED,
    gate_config_from_dict,
    gate_metrics,
    read_anchor_stamp,
    read_shard_arms,
    readout_exit_code,
    rederive_reference_from_shards,
    rederive_reference_with_phase_sweep,
    resolve_gate_hold_path,
    shadow_readout_from_csv,
    shadow_readout_verdict,
    write_anchor_stamp,
    ShardArm,
    READOUT_EXIT_CONFOUND_UNMEASURED,
    READOUT_EXIT_HOLD,
    READOUT_EXIT_IDENTITY_UNEVALUATED,
    READOUT_EXIT_KILL,
    READOUT_EXIT_PROMOTE,
    LEG_PASS,
    LEG_SKIPPED,
    LEG_UNMEASURED,
)
from chess_anti_engine.tune.trainable_phases import (
    _publish_iteration_model,
    _run_training_and_gating,
)
from chess_anti_engine.tune.trial_config import (
    DifficultyState,
    RestoreResult,
    SelfplayResult,
    TrialConfig,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _sample(i: int, cur_score: float, prev_score: float, n: int = 100) -> AnchoredSample:
    """Draw-free anchored sample with exactly ``n`` games a side."""
    cw = round(cur_score * n)
    pw = round(prev_score * n)
    return AnchoredSample(
        iteration=i,
        cur_w=cw, cur_d=0, cur_l=n - cw,
        prev_w=pw, prev_d=0, prev_l=n - pw,
    )


# Eight iterations whose deltas average exactly ``shift`` with a fixed spread.
_WOBBLE = (0.00, 0.02, -0.02, 0.01, -0.01, 0.00, 0.02, -0.02)


def _window(shift: float) -> list[AnchoredSample]:
    return [
        _sample(i, cur_score=0.50 + shift + w, prev_score=0.50)
        for i, w in enumerate(_WOBBLE)
    ]


def _gate(
    mode: str = MODE_ENFORCE,
    *,
    window_iters: int = 24,
    min_iters: int = 8,
    min_games_per_side: int = 40,
    demote_delta_elo: float = -25.0,
    demote_step_elo: float = -125.0,
    alpha: float = 0.05,
    max_hold_iters: int = 12,
) -> PromotionGate:
    return PromotionGate(cfg=GateConfig(
        mode=mode, window_iters=window_iters, min_iters=min_iters,
        min_games_per_side=min_games_per_side,
        demote_delta_elo=demote_delta_elo,
        demote_step_elo=min(demote_step_elo, demote_delta_elo),
        alpha=alpha,
        max_hold_iters=max_hold_iters,
    ))


def _demoting_gate(promoted: Path) -> tuple[PromotionGate, GateDecision]:
    """A gate carrying a real DEMOTE verdict, with a fallback on disk."""
    g = _gate(demote_delta_elo=-50.0)
    d = _feed(g, _window(-0.30))
    assert d.acted is True
    promoted.write_bytes(b"promoted-export")
    return g, d


def _feed(gate: PromotionGate, samples: list[AnchoredSample]) -> GateDecision:
    for s in samples:
        gate.observe(s)
    return gate.apply(gate.decide())


# --------------------------------------------------------------------------
# 1. "gate_passed: 1 with gate_games: 0" must be unrepresentable
# --------------------------------------------------------------------------
def test_promote_with_zero_games_cannot_be_reported() -> None:
    """MUTATION: drop the assertion in ``gate_metrics``.

    This is the exact shape the loop emitted for 200+ iterations: a pass with
    no games behind it. It must raise, not round-trip.
    """
    bogus = GateDecision(decision=DECISION_PROMOTE, reason="promote_no_regression",
                         mode=MODE_ENFORCE, iters=8, games_cur=0, games_prev=0)
    with pytest.raises(AssertionError, match="zero anchored games"):
        gate_metrics(bogus)


def test_disabled_gate_reports_not_run_and_never_a_pass() -> None:
    """MUTATION: make DECISION_NOT_RUN equal DECISION_PROMOTE, or default the
    decision to 1. An off gate must be distinguishable in the metrics from a
    gate that ran and passed.

    THE WINDOW IS FULL ON PURPOSE, and an earlier version of this test got that
    wrong. It called ``.decide()`` on an EMPTY gate, which returns
    ``DECISION_NOT_RUN`` via the ``min_iters`` branch whether or not the
    ``MODE_OFF`` check exists -- so ``if cfg.mode == MODE_OFF:`` -> ``if False:``
    survived it, and the OFF SWITCH, the only thing standing between this
    feature and production, was untested. Reviewer round 6.

    With a full clean window the mutation returns ``DECISION_PROMOTE``; with a
    full regressing window it returns ``DECISION_DEMOTE``. Both are asserted,
    because "off" must beat both halves of the rule, not just the quiet one.
    """
    for window in (_window(0.0), _window(-0.30)):
        g = _gate(mode=MODE_OFF)
        for s in window:
            g.observe(s)
        d = g.decide()
        m = gate_metrics(d)
        assert d.reason == "disabled"
        assert m["gate_decision"] == float(DECISION_NOT_RUN)
        assert m["gate_decision"] != float(DECISION_PROMOTE)
        assert m["gate_decision"] != float(DECISION_DEMOTE)
        assert m["gate_would_demote"] == 0.0
        # The window aggregates stay empty: an off gate reports no verdict AND
        # no numbers behind one. The per-iteration sample is still filled, which
        # is what makes the shadow readout run at gate_mode: off.
        assert m["gate_games_cur"] == 0.0
        assert m["gate_games_prev"] == 0.0
        assert m["gate_sample_games_cur"] == 100.0
        # The old metric name must be gone: a dashboard still charting it would
        # silently show a constant again.
        assert "gate_passed" not in m


def test_training_result_default_is_not_a_pass() -> None:
    """MUTATION: restore ``gate_passed: bool = True``. The dataclass default is
    what got emitted every iteration; it must default to "did not run"."""
    from chess_anti_engine.tune.trial_config import TrainingResult

    assert TrainingResult().gate_decision.decision == DECISION_NOT_RUN


# --------------------------------------------------------------------------
# 2. the decision rule
# --------------------------------------------------------------------------
def test_promotes_when_no_regression_is_proven() -> None:
    d = _feed(_gate(demote_delta_elo=-50.0), _window(0.0))
    assert d.decision == DECISION_PROMOTE
    assert d.reason == "promote_no_regression"
    assert d.iters == 8
    assert d.games_cur == 800
    assert d.games_prev == 800
    assert d.delta_elo == pytest.approx(0.0, abs=1e-9)


def test_demotes_only_when_the_upper_bound_clears_the_line() -> None:
    """MUTATION: test ``delta_elo`` (or ``elo_lo``) against the demote line
    instead of ``elo_hi``.

    -0.08 score is -55.6 Elo -- worse than the -50 line on the POINT estimate,
    but its 95% upper bound is -48.2, so the regression is not proven and the
    gate must promote. -0.10 (upper bound -62.1) must demote. A rule reading
    the point estimate flips the first case; a rule reading the lower bound
    flips both.
    """
    near = _feed(_gate(demote_delta_elo=-50.0), _window(-0.08))
    assert near.delta_elo < -50.0          # point estimate is past the line
    assert near.elo_hi > -50.0             # ...but the interval is not
    assert near.decision == DECISION_PROMOTE
    assert near.reason == "promote_no_regression"

    clear = _feed(_gate(demote_delta_elo=-50.0), _window(-0.10))
    assert clear.elo_hi < -50.0
    assert clear.decision == DECISION_DEMOTE
    assert clear.reason == "demote_regression"


def test_ci_is_a_real_interval_not_a_point() -> None:
    """MUTATION: set the t-quantile to 0 (or se to 0). A zero-width interval
    claiming certainty from 8 iterations is audit row L11 in a new place."""
    d = _feed(_gate(demote_delta_elo=-50.0), _window(-0.10))
    assert d.elo_lo < d.delta_elo < d.elo_hi
    assert (d.elo_hi - d.elo_lo) > 5.0


def test_identical_deltas_refuse_to_decide() -> None:
    """A stuck counter produces zero spread. Refuse rather than divide by it."""
    g = _gate()
    d = _feed(g, [_sample(i, 0.50, 0.50) for i in range(8)])
    assert d.decision == DECISION_NOT_RUN
    assert d.reason == "degenerate_variance"


def test_short_window_and_thin_iterations_do_not_decide() -> None:
    """MUTATION: drop the min_iters / min_games_per_side guards. A verdict off
    3 iterations, or off 5 games a side, is noise wearing a decision's clothes.

    The shift is -0.10 (about -70 Elo/iteration), not -0.30: a -0.30 sample is
    STEP-scale and now demotes on the single-iteration leg from one row, which
    is the whole point of that leg
    (``test_a_step_leg_fires_on_a_single_iteration_step``). Dropping min_iters
    still fails this test -- at 3 iterations of -0.10 the window mean is
    -69.5 Elo with ``elo_hi`` -46, well under the -25 line.
    """
    short = _feed(_gate(), _window(-0.10)[:3])
    assert short.decision == DECISION_NOT_RUN
    assert short.reason == "insufficient_iters"

    thin = _feed(_gate(), [
        _sample(i, cur_score=0.50 - 0.30 + w, prev_score=0.50, n=10)
        for i, w in enumerate(_WOBBLE)
    ])
    assert thin.decision == DECISION_NOT_RUN
    assert thin.reason == "insufficient_games"


def test_shadow_mode_computes_the_verdict_but_never_acts() -> None:
    """MUTATION: let shadow mode return DECISION_DEMOTE. Shadow exists to
    measure the anchored delta's null distribution on live data BEFORE the
    unmeasured PID-drift bias is allowed to hold the fleet back."""
    shadow = _feed(_gate(mode=MODE_SHADOW), _window(-0.30))
    assert shadow.reason == "shadow_would_demote"
    assert shadow.decision == DECISION_PROMOTE
    assert shadow.acted is False
    # ...and it still reports the number it would have acted on.
    assert shadow.elo_hi < -50.0

    enforced = _feed(_gate(mode=MODE_ENFORCE), _window(-0.30))
    assert enforced.acted is True


def test_hold_counter_advances_and_the_brake_releases() -> None:
    """MUTATION: make ``apply`` leave ``holds`` at 0, or drop the
    max_hold_iters check. A hold stops new anchored samples arriving (the fleet
    is on one sha), so nothing but this counter can ever release it -- a brake
    that cannot release re-creates the 2026-03 freeze one level up."""
    g = _gate()
    for s in _window(-0.30):
        g.observe(s)
    for expected in range(1, 4):
        d = g.apply(g.decide())
        assert d.acted is True
        assert d.holds == expected

    g.holds = g.cfg.max_hold_iters
    released = g.apply(g.decide())
    assert released.reason == "hold_expired"
    assert released.acted is False
    assert released.holds == 0


def test_gate_never_demotes_for_absence_of_improvement() -> None:
    """A positive demote line would turn the gate into the 2026-03 gate: it
    would reject every step that failed to PROVE improvement, which at 0.02
    Elo/iteration is every step forever."""
    with pytest.raises(ValueError, match="must be negative"):
        GateConfig(demote_delta_elo=0.0).validate()


# --------------------------------------------------------------------------
# 3. the removed gate must not be reachable from a config edit
# --------------------------------------------------------------------------
def test_nonzero_gate_games_is_refused() -> None:
    """MUTATION: accept and ignore ``gate_games``. "Just turn it back on" is
    the forbidden shortcut; nothing implements it, so it must raise rather than
    become a knob that never reaches the code."""
    with pytest.raises(ValueError, match="no longer implemented"):
        gate_config_from_dict({"gate_games": 100, "gate_mode": MODE_SHADOW})

    # Inert leftovers in a live yaml must still validate: deleting a key from a
    # live yaml is itself a reload risk.
    cfg = gate_config_from_dict(
        {"gate_games": 0, "gate_threshold": 0.5, "gate_mcts_sims": 1},
    )
    assert cfg.mode == MODE_OFF


# Every yaml key -> GateConfig field, with a NON-DEFAULT value for each. The
# value must differ from the default, or the test cannot tell "the key was read"
# from "the key was ignored and the default happened to match" -- which is the
# exact shape this whole module is about.
_YAML_TO_GATE_FIELD = (
    ("gate_mode", "mode", MODE_ENFORCE),
    ("gate_window_iters", "window_iters", 31),
    ("gate_min_iters", "min_iters", 7),
    ("gate_min_games_per_side", "min_games_per_side", 23),
    ("gate_demote_delta_elo", "demote_delta_elo", -37.5),
    ("gate_alpha", "alpha", 0.01),
    ("gate_max_hold_iters", "max_hold_iters", 9),
)


@pytest.mark.parametrize(("yaml_key", "field", "value"), _YAML_TO_GATE_FIELD)
def test_every_yaml_gate_key_reaches_the_gate(
    yaml_key: str, field: str, value: object,
) -> None:
    """MUTATION: hardcode ANY of the seven fields in ``gate_config_from_dict``
    to its default -- i.e. simulate that yaml key being unreadable.

    All seven survived the whole 61-test suite before this test existed, and
    ``gate_config_from_dict`` is the ONLY path from the yaml to the gate
    (one call site, ``trainable.py``). The production yaml deliberately ships
    no gate keys, so the only way this feature is EVER enabled is an operator
    adding ``gate_mode: shadow|enforce`` at a restart -- and if that key does
    not land, ``gate_decision`` stays -1/``disabled`` forever, which is
    indistinguishable in the metrics from "the window is not full yet". A knob
    accepted and silently ignored, in the module whose docstrings are about
    that failure mode.
    """
    default = getattr(GateConfig(), field)
    assert value != default, (
        f"{yaml_key} must be tested at a NON-DEFAULT value; {value!r} is the "
        "default, so this row could not distinguish read from ignored"
    )
    cfg = gate_config_from_dict({yaml_key: value})
    assert getattr(cfg, field) == value, (
        f"{yaml_key} did not reach GateConfig.{field}: got "
        f"{getattr(cfg, field)!r}, wanted {value!r}"
    )
    # ...and every other field keeps its default, so no key writes a neighbour.
    for _, other, _ in _YAML_TO_GATE_FIELD:
        if other != field:
            assert getattr(cfg, other) == getattr(GateConfig(), other)


def test_all_seven_yaml_gate_keys_reach_the_gate_together() -> None:
    """The same seven in ONE dict, as an operator would actually write them,
    and the result must be a config the gate accepts rather than a shape that
    only survives because nothing validated it."""
    cfg = gate_config_from_dict({k: v for k, _, v in _YAML_TO_GATE_FIELD})
    for _, field, value in _YAML_TO_GATE_FIELD:
        assert getattr(cfg, field) == value, field
    cfg.validate()
    assert PromotionGate(cfg=cfg).cfg.mode == MODE_ENFORCE


def test_gate_mode_default_is_off() -> None:
    assert gate_config_from_dict({}).mode == MODE_OFF
    # TrialConfig must NOT carry a second copy of the gate knobs: the copy
    # nothing reads is the one that rots (it held the pre-review 40 / -50.0
    # long after both were corrected everywhere else).
    assert not [f for f in dataclasses.fields(TrialConfig)
                if f.name.startswith("gate_") and f.name not in {
                    "gate_games", "gate_threshold", "gate_interval",
                    "gate_mcts_sims"}]


# --------------------------------------------------------------------------
# 4. the ingest split is what makes the games free -- and it must be REAL
# --------------------------------------------------------------------------
def _shard(path: Path, *, sha: str, w: int, d: int, l: int,
           regret: float | None = None) -> None:
    n = 2
    meta: dict = {
        "model_sha256": sha, "games": w + d + l, "positions": n,
        "wins": w, "draws": d, "losses": l,
    }
    if regret is not None:
        meta["opponent_wdl_regret_limit"] = regret
    save_local_shard_arrays(
        path,
        arrs={
            "x": np.zeros((n, 146, 8, 8), dtype=np.float32),
            "policy_target": np.eye(POLICY_SIZE, dtype=np.float32)[np.zeros(n, dtype=int)],
            "wdl_target": np.zeros((n,), dtype=np.int8),
            "priority": np.ones((n,), dtype=np.float32),
            "has_policy": np.ones((n,), dtype=np.uint8),
        },
        meta=meta,
    )


def test_ingest_splits_anchored_counts_by_publishing_model(tmp_path: Path) -> None:
    """MUTATION: bucket every accepted shard as "cur" (ignore prev_model_sha),
    or stop threading prev_model_sha through _process_shard.

    Without the split the gate sees zero prev-model games forever and every
    verdict silently becomes "insufficient_games" -- the failure mode this
    repository is built out of.
    """
    inbox = tmp_path / "inbox"
    processed = tmp_path / "processed"
    _shard(inbox / "w0" / f"a{LOCAL_SHARD_SUFFIX}", sha="cur-sha", w=7, d=2, l=1,
           regret=0.0900)
    _shard(inbox / "w0" / f"b{LOCAL_SHARD_SUFFIX}", sha="prev-sha", w=3, d=4, l=5,
           regret=0.0850)
    _shard(inbox / "w0" / f"c{LOCAL_SHARD_SUFFIX}", sha="ancient-sha", w=9, d=9, l=9,
           regret=0.5000)
    # A LEGACY shard (predates ShardMeta.opponent_wdl_regret_limit): its games
    # count toward the arm, but contribute NOTHING to the difficulty mean --
    # absent is unknown, and 0.0 would read as unhandicapped Stockfish.
    _shard(inbox / "w0" / f"d{LOCAL_SHARD_SUFFIX}", sha="cur-sha", w=1, d=1, l=1)

    summary = _ingest_distributed_selfplay(
        buf=DiskReplayBuffer(
            256, shard_dir=tmp_path / "replay",
            rng=np.random.default_rng(0), shuffle_cap=64, shard_size=8,
  # This test INGESTS, so it is a writer (audit G12's required keyword).
            read_only=False,
        ),
        holdout_buf=ArrayReplayBuffer(32, rng=np.random.default_rng(1)),
        holdout_frac=0.0, holdout_frozen=False,
        inbox_dir=inbox, processed_dir=processed,
        target_games=100,
        accepted_model_shas={"cur-sha", "prev-sha"},
        prev_model_sha="prev-sha",
        prev_model_max_fraction=1.0,
        wait_timeout_s=0.5, poll_seconds=0.01,
        rng=np.random.default_rng(2), on_poll=None, min_games_fraction=0.0,
    )

    assert (summary["gate_cur_w"], summary["gate_cur_d"], summary["gate_cur_l"]) == (8, 3, 2)
    assert (summary["gate_prev_w"], summary["gate_prev_d"], summary["gate_prev_l"]) == (3, 4, 5)
    # The difficulty each arm played at rides the same split (MUTATION: delete
    # the gate_*_regret_* accumulation from _process_shard). Games-weighted;
    # the STALE shard's 0.50 must contaminate neither arm; and the legacy
    # shard's 3 games count toward the arm but not toward the difficulty mean.
    assert summary["gate_cur_regret_games"] == 10
    assert summary["gate_cur_regret_weighted"] == pytest.approx(0.0900 * 10)
    assert summary["gate_prev_regret_games"] == 12
    assert summary["gate_prev_regret_weighted"] == pytest.approx(0.0850 * 12)
    # Stale (neither published model) contributes to NEITHER side: it was played
    # by a net no longer under comparison.
    assert summary["gate_cur_w"] + summary["gate_prev_w"] == summary["matching_w"]
    assert summary["stale_games"] == 27
    # THE IDENTITY, at its source. ``_process_shard`` increments one gate arm
    # and the pooled ``matching_w/d/l`` in the SAME branch, so the two arms and
    # the pool must add up exactly -- and the pool is what ships to
    # ``progress.csv`` as ``pid_curriculum_w/d/l``. This is the check with no
    # statistics in it: shard loss between the split and the pool, or a game
    # bucketed to neither arm, breaks it by an exact integer.
    assert (
        sum(summary[f"gate_{side}_{o}"] for side in ("cur", "prev") for o in "wdl")
        == sum(summary[f"matching_{o}"] for o in "wdl")
    )


# --------------------------------------------------------------------------
# 5. the actuator: a demote must change what the FLEET plays
# --------------------------------------------------------------------------
class _MarkerTrainer:
    """Exports a distinguishable blob each time, so a publish that ignored the
    hold is visible in the published bytes."""

    def __init__(self) -> None:
        self.exports = 0

    def export_swa(self, path: Path) -> None:
        self.exports += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"trainer-export-{self.exports}".encode("ascii"))


def _tiny_model_cfg() -> ModelConfig:
    return ModelConfig(
        kind="transformer", embed_dim=32, num_layers=1, num_heads=2,
        ffn_mult=2, use_smolgen=False, use_nla=False,
        use_qk_rmsnorm=False, use_gradient_checkpointing=False,
    )


def _publish(tmp_path: Path, trainer, override: Path | None) -> Path:
    _publish_distributed_trial_state(
        trainer=trainer,
        config={"selfplay_batch": 4, "max_plies": 40, "mcts": "gumbel",
                "fast_simulations": 8, "sf_move_nodes": 100, "sf_hash_mb": 16},
        model_cfg=_tiny_model_cfg(),
        server_root=tmp_path, trial_id="t0",
        training_iteration=1, trainer_step=1, sf_nodes=10, mcts_simulations=4,
        override_model_path=override,
    )
    return tmp_path / "trials" / "t0" / "publish" / "latest_model.pt"


def test_hold_publishes_the_promoted_export_not_the_trained_weights(
    tmp_path: Path,
) -> None:
    """MUTATION: ignore ``override_model_path`` in the publish path.

    This is the whole actuator. If the override is dropped, the gate keeps
    reporting DEMOTE while the fleet keeps getting the demoted net -- a
    decision that is accepted and then silently ignored.
    """
    trainer = _MarkerTrainer()
    published = _publish(tmp_path, trainer, None)
    assert published.read_bytes() == b"trainer-export-1"

    promoted = tmp_path / "promoted.pt"
    promoted.write_bytes(b"promoted-export")

    held = _publish(tmp_path, trainer, promoted)
    assert held.read_bytes() == b"promoted-export"
    assert trainer.exports == 1, "a hold must not export the trainer's weights"

    manifest = json.loads(
        (tmp_path / "trials" / "t0" / "publish" / "manifest.json").read_text("utf-8"),
    )
    import hashlib
    assert manifest["model"]["sha256"] == hashlib.sha256(b"promoted-export").hexdigest()


def test_hold_with_a_missing_fallback_raises_rather_than_publishing_anything(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        _publish(tmp_path, _MarkerTrainer(), tmp_path / "does-not-exist.pt")


# --------------------------------------------------------------------------
# 5b. the ACTUATOR PATH, end to end -- a decision that never reaches publish
#     is the defect this whole PR is about
# --------------------------------------------------------------------------
def test_a_demote_verdict_resolves_to_the_promoted_path(tmp_path: Path) -> None:
    """MUTATION (reviewer N2): `if tr.gate_decision.acted and ...` -> `if False and ...`.

    ``_publish_distributed_trial_state`` honouring ``override_model_path`` is
    necessary and NOT sufficient: nothing downstream complains if the decision
    never becomes a path. This is the missing half of M15.
    """
    promoted = tmp_path / "gate_promoted_model.pt"
    _g, demote = _demoting_gate(promoted)
    assert resolve_gate_hold_path(
        demote, gate_promoted_model_path=promoted,
    ) == promoted

    promote = _feed(_gate(demote_delta_elo=-50.0), _window(0.0))
    assert promote.acted is False
    assert resolve_gate_hold_path(
        promote, gate_promoted_model_path=promoted,
    ) is None

    # A demote with no fallback on disk publishes normally rather than crashing.
    assert resolve_gate_hold_path(
        demote, gate_promoted_model_path=tmp_path / "absent.pt",
    ) is None
    assert resolve_gate_hold_path(demote, gate_promoted_model_path=None) is None


def _publish_iter(tmp_path: Path, trainer, *, hold: Path | None,
                  promoted: Path | None,
                  controller: GateHoldController | None = None) -> Path:
    publish_dir = tmp_path / "trials" / "t0" / "publish"
    if controller is None:
        controller = GateHoldController(
            gate=_gate(mode=MODE_SHADOW), promoted_model_path=promoted,
            hold_path=hold,
        )
    _publish_iteration_model(
        trainer=trainer,
        config={"selfplay_batch": 4, "max_plies": 40, "mcts": "gumbel",
                "fast_simulations": 8, "sf_move_nodes": 100, "sf_hash_mb": 16},
        model_cfg=_tiny_model_cfg(),
        server_root=tmp_path, publish_dir=publish_dir, trial_id="t0",
        iteration_idx=1, ds=DifficultyState(wdl_regret=0.089, sf_nodes=50_000),
        sims=4, reuse_existing_model_for_same_step=False, hold=controller,
    )
    return publish_dir / "latest_model.pt"


def test_hold_reaches_the_publish_call_and_the_anchor_survives(
    tmp_path: Path,
) -> None:
    """MUTATIONS (reviewer N1 + N3): drop ``override_model_path=`` at the call
    site; drop the ``gate_hold_model_path is None`` guard on the snapshot.

    N1 means the gate reports DEMOTE forever while the fleet keeps receiving
    the demoted net. N3 means the anchor is overwritten with the very weights
    being held back, so the brake releases itself on the next iteration and
    nothing reports that it did.
    """
    trainer = _MarkerTrainer()
    promoted = tmp_path / "gate_promoted_model.pt"

    # Normal iteration: trainer's export goes out AND becomes the anchor.
    published = _publish_iter(tmp_path, trainer, hold=None, promoted=promoted)
    assert published.read_bytes() == b"trainer-export-1"
    assert promoted.read_bytes() == b"trainer-export-1"

    # Held iteration: the ANCHOR goes out, the trainer does not export, and the
    # anchor is NOT refreshed with the held-back weights.
    published = _publish_iter(tmp_path, trainer, hold=promoted, promoted=promoted)
    assert published.read_bytes() == b"trainer-export-1"
    assert trainer.exports == 1
    assert promoted.read_bytes() == b"trainer-export-1"

    # Release: the next unheld publish exports again, and re-anchors ONLY
    # because a genuine promote says so. `_publish_iter` builds a controller
    # with no verdict, and a controller with no verdict does NOT re-anchor an
    # anchor that already exists (audit G3-8 / review R4) -- otherwise every
    # process restart would re-anchor to the current export once.
    ctrl = GateHoldController(gate=_gate(mode=MODE_SHADOW),
                              promoted_model_path=promoted)
    ctrl.on_decision(GateDecision(
        decision=DECISION_PROMOTE, reason="promote_no_regression",
        mode=MODE_SHADOW, games_cur=197, games_prev=38))
    published = _publish_iter(tmp_path, trainer, hold=None, promoted=promoted,
                              controller=ctrl)
    assert published.read_bytes() == b"trainer-export-2"
    assert promoted.read_bytes() == b"trainer-export-2"

    # N3 in the shape where it bites. In production the hold source IS the
    # anchor, so refreshing during a hold is a no-op and the guard is purely
    # defensive -- which is exactly why it needs a test that can tell: give the
    # hold a DIFFERENT source and the missing guard overwrites the anchor with
    # bytes that were never promoted.
    other = tmp_path / "some_other_export.pt"
    other.write_bytes(b"not-the-anchor")
    ctrl.hold_path = other
    published = _publish_iter(tmp_path, trainer, hold=other, promoted=promoted,
                              controller=ctrl)
    assert published.read_bytes() == b"not-the-anchor"
    assert promoted.read_bytes() == b"trainer-export-2", (
        "the promoted anchor must never be refreshed while a hold is on"
    )


def test_disabled_gate_copies_nothing(tmp_path: Path) -> None:
    """MUTATION: pass a promoted path while the gate is off. The export is
    ~252 MB; a disabled feature must not copy it every iteration."""
    published = _publish_iter(tmp_path, _MarkerTrainer(), hold=None, promoted=None)
    assert published.read_bytes() == b"trainer-export-1"
    assert not (tmp_path / "gate_promoted_model.pt").exists()
    assert list(tmp_path.glob("**/*.tmp")) == []


def test_hold_survives_a_restart_and_ages_through_retries() -> None:
    """MUTATIONS (reviewer N7 + N8): ``gate.apply(gate.decide())`` -> ``gate.decide()``;
    ``state_dict`` drops the latch.

    N7 means ``holds`` never advances in production, so ``max_hold_iters`` never
    releases -- the brake-that-cannot-release the module docstring names as the
    thing to avoid. Without the persisted latch a restart mid-hold publishes the
    held-back weights and overwrites the anchor with them.
    """
    g = _gate()
    d = _feed(g, _window(-0.30))
    assert d.acted is True
    assert g.hold_active is True
    assert g.holds == 1

    revived = _gate()
    revived.load_state_dict(json.loads(json.dumps(g.state_dict())))
    assert revived.hold_active is True
    assert revived.holds == 1

    # An iteration that produced no verdict (sp.should_retry) still ages the
    # hold, or a run stuck in retry holds forever with the counter frozen.
    for expected in range(2, revived.cfg.max_hold_iters):
        assert revived.advance_hold_without_decision() is True
        assert revived.holds == expected
    assert revived.advance_hold_without_decision() is False
    assert revived.hold_active is False
    assert revived.holds == 0
    assert revived.advance_hold_without_decision() is False


# --------------------------------------------------------------------------
# 5c. the shadow readout must be a READABLE series
# --------------------------------------------------------------------------
def test_per_iteration_sample_is_reported_separately_from_the_window() -> None:
    """MUTATION: emit only the window aggregates.

    Consecutive windows overlap by ~95%, so the sd of the reported
    ``gate_delta_elo`` column understates the per-iteration sd by ~10x
    (measured: 4.56 vs 45.56 Elo over 56 live iterations). A promote-to-enforce
    rule keyed to the window column therefore CANNOT FAIL -- the repo's
    signature defect, inside the gate's own decision protocol.
    """
    g = _gate(mode=MODE_SHADOW)
    for s in _window(0.0):
        g.observe(s)
    d = g.apply(g.decide())
    m = gate_metrics(d)

    for key in ("gate_sample_delta_elo", "gate_sample_delta_score",
                "gate_sample_games_cur", "gate_sample_games_prev"):
        assert key in m

    # The last window sample is cur 0.48 / prev 0.50 -> -0.02 score.
    assert m["gate_sample_delta_score"] == pytest.approx(-0.02)
    assert m["gate_sample_delta_elo"] == pytest.approx(-0.02 * ELO_PER_SCORE_AT_HALF)
    assert m["gate_sample_games_cur"] == 100.0
    assert m["gate_sample_games_prev"] == 100.0
    # ...and it is a DIFFERENT number from the window mean, which is 0 here.
    assert m["gate_delta_elo"] == pytest.approx(0.0, abs=1e-9)
    assert m["gate_sample_delta_elo"] != pytest.approx(m["gate_delta_elo"])


def test_sample_is_reported_even_when_the_window_declines_to_decide() -> None:
    """A readout that only exists on iterations the gate happened to judge is a
    biased sample of the gate's own inputs -- and at the shipped floors the
    window declines on a large minority of live iterations."""
    g = _gate(mode=MODE_SHADOW)
    g.observe(_sample(0, 0.55, 0.50))
    m = gate_metrics(g.apply(g.decide()))
    assert m["gate_decision"] == float(DECISION_NOT_RUN)
    assert m["gate_sample_games_cur"] == 100.0
    assert m["gate_sample_delta_score"] == pytest.approx(0.05)


def test_measured_live_shape_admits_most_iterations() -> None:
    """MUTATION: restore ``gate_min_games_per_side: 40``.

    MEASURED over live iters 164-219: the prev arm realizes ~38 games/iteration
    (median 33), NOT the 143 that ``distributed_prev_model_max_fraction: 0.60``
    implies -- the cap is a ceiling that never binds
    (``distributed_stale_games == 0`` throughout). A floor of 40 sits at the
    median and disqualifies ~70% of iterations, pinning the effective window at
    K=9-12 and making ``gate_delta_elo`` repeat verbatim across rows.
    """
    assert GateConfig().min_games_per_side <= 20
    live_prev_counts = (1, 21, 25, 33, 52, 68, 99)  # measured quantiles
    admitted = sum(
        1 for n in live_prev_counts
        if AnchoredSample(0, cur_w=197, prev_w=n).usable(
            min_games_per_side=GateConfig().min_games_per_side,
        )
    )
    assert admitted >= 6, "the shipped floor disqualifies most live iterations"


def test_one_sided_sample_is_never_usable() -> None:
    """MUTATION (reviewer N5): ``usable`` ``and`` -> ``or``.

    During a hold every accepted shard carries one sha, so ``cur_games`` is 0.
    An ``or`` admits that, ``delta`` becomes NaN, and NaN poisons the window
    mean into a silent ``promote_no_regression``. The ``and`` is load-bearing.
    """
    held = AnchoredSample(0, prev_w=100, prev_d=40, prev_l=60)
    assert held.cur_games == 0
    assert held.usable(min_games_per_side=5) is False

    g = _gate()
    for i in range(8):
        g.observe(AnchoredSample(i, prev_w=100, prev_d=40, prev_l=60))
    d = g.apply(g.decide())
    assert d.decision == DECISION_NOT_RUN
    assert not math.isnan(d.games_prev)
    assert d.games_cur == 0


def test_promote_with_a_one_sided_zero_is_refused(tmp_path: Path) -> None:
    """MUTATION (reviewer N4): ``min(...)`` -> ``max(...)`` in ``gate_metrics``.

    The realistic zero-game shape is ``(0, 143)`` during a hold, not ``(0, 0)``
    -- and ``max`` cannot be distinguished from ``min`` by a ``(0, 0)`` case.
    """
    del tmp_path
    for cur, prev in ((0, 143), (143, 0), (0, 0)):
        with pytest.raises(AssertionError, match="zero anchored games"):
            gate_metrics(GateDecision(
                decision=DECISION_PROMOTE, reason="promote_no_regression",
                mode=MODE_ENFORCE, iters=8, games_cur=cur, games_prev=prev,
            ))


def test_small_window_t_quantile_is_inflated_over_the_normal() -> None:
    """MUTATION (reviewer N6): drop the small-df inflation.

    The gate runs at df 7-23, where the normal quantile is anti-conservative.
    """
    from chess_anti_engine.tune.promotion_gate import _t_quantile

    assert _t_quantile(0.05, 7) > 1.80
    assert _t_quantile(0.05, 7) > _t_quantile(0.05, 23) > 1.6449
    assert _t_quantile(0.05, 10_000) == pytest.approx(1.6449, abs=1e-3)


def test_window_trim_honours_window_iters() -> None:
    """MUTATION (reviewer N9): ignore ``window_iters`` when trimming.

    An unbounded window turns a rolling alarm into a lifetime average, which
    cannot detect a step at all."""
    g = _gate(mode=MODE_SHADOW)
    for i in range(200):
        g.observe(_sample(i, 0.50, 0.50))
    assert len(g.samples) == g.cfg.window_iters
    revived = _gate(mode=MODE_SHADOW)
    revived.load_state_dict({"samples": [
        {"iteration": i, "cur_w": 50, "cur_l": 50, "prev_w": 50, "prev_l": 50}
        for i in range(200)
    ]})
    assert len(revived.samples) == revived.cfg.window_iters


def test_a_verdict_off_a_handful_of_games_is_refused_by_validate() -> None:
    """``validate()`` is the only thing between an operator and a gate that
    holds the fleet on four coin flips."""
    with pytest.raises(ValueError, match="gate_min_iters"):
        GateConfig(min_iters=2).validate()
    with pytest.raises(ValueError, match="gate_min_games_per_side"):
        GateConfig(min_games_per_side=1).validate()


# --------------------------------------------------------------------------
# 6. the gate must run on every iteration, including ones that do not train
# --------------------------------------------------------------------------
class _NullWriter:
    def add_scalar(self, *_a, **_k) -> None:
        return None


class _StubTrainer:
    writer = _NullWriter()

    def train_steps(self, *_a, **_k):  # pragma: no cover - not reached here
        raise AssertionError("training must be skipped in this test")


def test_gate_observes_the_iteration_even_when_training_is_skipped(
    tmp_path: Path,
) -> None:
    """MUTATION: move the ``_run_net_gating`` call back inside ``if not
    skip_train``. An iteration that ingested games but had too few rows to
    train still produced an anchored A/B; dropping it silently shortens the
    window and delays every verdict."""
    gate = _gate(mode=MODE_SHADOW)
    tc = TrialConfig(batch_size=512)
    sp = SelfplayResult(
        gate_cur_w=40, gate_cur_d=10, gate_cur_l=50,
        gate_prev_w=50, gate_prev_d=10, gate_prev_l=40,
    )
    result = _run_training_and_gating(
        tc=tc, trainer=_StubTrainer(), buf=[], holdout_buf=[], holdout_frozen=True,
        config={}, model_cfg=None, device="cpu",
        ds=DifficultyState(wdl_regret=0.089, sf_nodes=50_000),
        sims=64, sp=sp,
        positions_ingested=0, imported_samples_this_iter=0,
        gate=gate, gate_match_idx=0,
        gate_state_path=tmp_path / "gate_state.json", gate_hold=None,
        distributed_server_root=tmp_path, iteration_idx=5,
        iteration_zero_based=5, trial_id="t0", restore=RestoreResult(),
    )
    assert result.steps == 0                      # training really was skipped
    assert len(gate.samples) == 1                 # ...and the gate still saw it
    assert gate.samples[0].cur_games == 100
    assert result.gate_decision.reason == "insufficient_iters"
    assert (tmp_path / "gate_state.json").is_file()


def test_running_the_gating_phase_advances_the_hold_latch(tmp_path: Path) -> None:
    """MUTATION (reviewer N7): ``gate.apply(gate.decide())`` -> ``gate.decide()``
    inside ``_run_net_gating``.

    ``apply`` is what advances ``holds`` and sets ``hold_active``. Called only
    from the phase, so a unit test that calls ``apply`` directly cannot see it
    being dropped -- and without it ``gate_max_hold_iters`` never releases in
    production: the brake-that-cannot-release the docstring names as the thing
    to avoid.
    """
    gate = _gate(mode=MODE_ENFORCE)
    for s_ in _window(-0.30)[:-1]:
        gate.observe(s_)
    last = _window(-0.30)[-1]
    sp = SelfplayResult(
        gate_cur_w=last.cur_w, gate_cur_d=last.cur_d, gate_cur_l=last.cur_l,
        gate_prev_w=last.prev_w, gate_prev_d=last.prev_d, gate_prev_l=last.prev_l,
    )
    result = _run_training_and_gating(
        tc=TrialConfig(batch_size=512), trainer=_StubTrainer(),
        buf=[], holdout_buf=[], holdout_frozen=True, config={}, model_cfg=None,
        device="cpu",
        ds=DifficultyState(wdl_regret=0.089, sf_nodes=50_000), sims=64, sp=sp,
        positions_ingested=0, imported_samples_this_iter=0,
        gate=gate, gate_match_idx=0,
        gate_state_path=tmp_path / "gate_state.json", gate_hold=None,
        distributed_server_root=tmp_path, iteration_idx=9,
        iteration_zero_based=9, trial_id="t0", restore=RestoreResult(),
    )
    assert result.gate_decision.acted is True
    assert gate.hold_active is True
    assert gate.holds == 1
    assert result.gate_decision.holds == 1
    persisted = json.loads((tmp_path / "gate_state.json").read_text("utf-8"))
    assert persisted["hold_active"] is True
    assert persisted["holds"] == 1


def test_gate_state_round_trips_across_a_restart() -> None:
    """MUTATION: drop ``samples`` from ``state_dict``. Losing the window on
    every restart means the gate is permanently in "insufficient_iters" on a
    run that restarts more often than ``min_iters`` -- silently never fires."""
    g = _gate(mode=MODE_SHADOW)
    for s in _window(-0.30):
        g.observe(s)
    g.apply(g.decide())

    revived = _gate(mode=MODE_SHADOW)
    revived.load_state_dict(json.loads(json.dumps(g.state_dict())))
    assert len(revived.samples) == 8
    assert revived.decide().elo_hi == pytest.approx(g.decide().elo_hi)


def _confounded_sample(i: int) -> AnchoredSample:
    """One anchored row carrying MEASURED confound inputs, all three distinct."""
    return AnchoredSample(
        iteration=i,
        cur_w=120, cur_d=40, cur_l=37,
        prev_w=20, prev_d=8, prev_l=10,
        cur_wdl_regret=0.0890 + 0.0001 * i,
        prev_wdl_regret=0.0975 - 0.0002 * i,
        regret_fit_slope=-3.25 + 0.01 * i,
    )


def test_gate_state_dict_covers_every_anchored_sample_field() -> None:
    """THE STRUCTURAL RULE: ``state_dict`` persists EVERY ``AnchoredSample``
    field, enumerated from the dataclass rather than listed by hand.

    MUTATION: delete any one key from the ``samples`` dict comprehension in
    ``PromotionGate.state_dict``. A hand-written key list is exactly how the
    three confound fields came to be recorded at runtime and dropped at every
    restart, and a test that names them one by one would not have caught the
    NEXT field added to the dataclass. This one fails on that field's first
    commit.
    """
    g = _gate(mode=MODE_SHADOW)
    g.observe(_confounded_sample(0))
    persisted = g.state_dict()["samples"]
    assert isinstance(persisted, list)
    assert persisted
    expected = {f.name for f in dataclasses.fields(AnchoredSample)}
    for row in persisted:
        assert isinstance(row, dict)
        assert set(row) == expected, f"missing {expected - set(row)}"


def test_gate_state_round_trips_the_confound_fields_through_json() -> None:
    """MUTATION: drop ``cur_wdl_regret``/``prev_wdl_regret``/
    ``regret_fit_slope`` from ``state_dict``, or read them back with a 0.0
    default in ``load_state_dict``.

    ``confound_elo`` is the product of all three, so a single dropped field
    silently turns a measured PID-lag offset into NaN (dropped) or into a
    fabricated 0.0 (defaulted) on every restored row. Round-tripping through
    real ``json.dumps``/``json.loads`` is the point: the gate's state reaches
    disk as JSON text, not as a Python object.
    """
    g = _gate(mode=MODE_SHADOW)
    for i in range(4):
        g.observe(_confounded_sample(i))

    text = json.dumps(g.state_dict())
    revived = _gate(mode=MODE_SHADOW)
    revived.load_state_dict(json.loads(text))

    assert len(revived.samples) == len(g.samples)
    for before, after in zip(g.samples, revived.samples):
        assert after == before
        assert after.confound_elo == pytest.approx(before.confound_elo)
    # ...and the offsets are actually non-trivial, so "all NaN" cannot pass.
    assert all(math.isfinite(s.confound_elo) for s in revived.samples)
    assert len({round(s.confound_elo, 6) for s in revived.samples}) == 4


def test_gate_state_writes_strict_json_and_keeps_nan_as_nan() -> None:
    """NaN means "not measured" and must survive as NaN, in STRICT JSON.

    MUTATION A: return ``float(v)`` from ``_json_float`` instead of ``None``
    for non-finite input -- ``json.dumps`` then emits the bare ``NaN`` token,
    which is not JSON and which the strict parse below rejects.
    MUTATION B: default the ``_f`` reader to 0.0 -- an unmeasured difficulty
    gap becomes a measured zero, i.e. the gate would report "this sample
    carries no PID-lag confound" about a sample nobody measured.
    """
    def _strict(_name: str) -> float:
        raise AssertionError("gate_state.json must not contain NaN/Infinity tokens")

    g = _gate(mode=MODE_SHADOW)
    g.observe(_confounded_sample(0))
    g.observe(AnchoredSample(iteration=1, cur_w=1, prev_w=1))  # confound all NaN

    text = json.dumps(g.state_dict())
    revived = _gate(mode=MODE_SHADOW)
    revived.load_state_dict(json.loads(text, parse_constant=_strict))

    assert revived.samples[0] == g.samples[0]
    unmeasured = revived.samples[1]
    assert math.isnan(unmeasured.cur_wdl_regret)
    assert math.isnan(unmeasured.prev_wdl_regret)
    assert math.isnan(unmeasured.regret_fit_slope)
    assert math.isnan(unmeasured.confound_elo)


def test_gate_state_written_before_the_confound_fields_loads_as_unmeasured() -> None:
    """A pre-existing ``gate_state.json`` has no confound keys at all.

    It must load as NaN ("nobody measured this"), never as 0.0 ("measured, and
    the gap was zero"). The counts still restore, so an in-flight run's window
    is not thrown away by the upgrade.
    """
    legacy = {
        "holds": 2, "hold_active": True,
        "samples": [{
            "iteration": 7,
            "cur_w": 100, "cur_d": 40, "cur_l": 57,
            "prev_w": 20, "prev_d": 8, "prev_l": 10,
        }],
    }
    g = _gate(mode=MODE_SHADOW)
    g.load_state_dict(json.loads(json.dumps(legacy)))
    assert g.holds == 2
    assert g.hold_active is True
    s = g.samples[0]
    assert (s.cur_w, s.cur_d, s.cur_l) == (100, 40, 57)
    assert math.isnan(s.cur_wdl_regret)
    assert math.isnan(s.prev_wdl_regret)
    assert math.isnan(s.regret_fit_slope)


# --------------------------------------------------------------------------
# 7. the documented resolving power must stay true
# --------------------------------------------------------------------------
def test_documented_per_iteration_resolution_holds_at_the_live_shape() -> None:
    """The per-iteration standard error, recomputed from the live rows.

    The module docstring, the config comment and the ledger's kill rule are all
    denominated in it. An earlier version of this test hard-coded
    ``observed_sd, null_mean = 45.56, -4.33`` as literals, so despite its
    docstring it guarded the CONFIG CONSTANTS against retuning and not the live
    shape against drifting. It now derives both from ``_LIVE_WINDOWS``.

    NOT 95/143. ``distributed_prev_model_max_fraction: 0.60`` is a ceiling that
    never binds -- ``distributed_stale_games`` is 0 throughout, and a binding cap
    would have to discard prev shards into ``stale_*``. Sizing the gate off the
    cap gives 31 Elo and is the wrong side of the split.
    """
    usable = [s for s in _live_samples() if s.cur_games and s.prev_games]
    deltas = [s.delta * ELO_PER_SCORE_AT_HALF for s in usable]
    observed_sd, null_mean = st.stdev(deltas), st.mean(deltas)
    assert observed_sd == pytest.approx(OFFLINE.sd_delta_elo, abs=0.01)
    assert null_mean == pytest.approx(OFFLINE.mean_delta_elo, abs=0.01)
    assert st.mean([s.cur_games for s in usable]) == pytest.approx(196.8, abs=0.1)
    assert st.mean([s.prev_games for s in usable]) == pytest.approx(38.3, abs=0.1)

    # "45.6 observed vs 41.5 predicted from binomial alone, so there is real
    # variance beyond binomial" was WRONG and shipped in this file's comments.
    # 41.5 uses 1/mean(n); the honest per-window quantity is the RMS of the
    # per-window binomial se, which at this shape is 61.1 Elo. The observed sd
    # is BELOW it, not above, and the standardized residuals sit at 1.011 where
    # 1.0 is pure independent binomial. There is no anchor-drift variance to
    # absorb, and that is a SECOND reason the delta cannot decide anything
    # about attribution -- a pure-noise series is what a broken split produces
    # too. See ``shadow_readout_verdict``.
    pooled = st.pstdev(
        [1.0] * sum(s.cur_w + s.prev_w for s in usable)
        + [0.5] * sum(s.cur_d + s.prev_d for s in usable)
        + [0.0] * sum(s.cur_l + s.prev_l for s in usable)
    )
    assert pooled == pytest.approx(0.3447, abs=0.0005)
    ses = [pooled * math.sqrt(1 / s.cur_games + 1 / s.prev_games)
           * ELO_PER_SCORE_AT_HALF for s in usable]
    rms_se = math.sqrt(sum(x * x for x in ses) / len(ses))
    assert rms_se == pytest.approx(61.1, abs=0.5)
    assert observed_sd < rms_se, (
        "the observed spread is BELOW independent binomial for this shape; any "
        "claim of variance beyond binomial is backwards"
    )
    residual_sd = st.stdev([(d - null_mean) / e for d, e in zip(deltas, ses)])
    assert residual_sd == pytest.approx(1.011, abs=0.02)

    # The shipped demote line is a FALSE-BRAKE BUDGET, not a sigma count. It
    # must stay above 2 window-sigma (spurious holds stay unmeasurably rare:
    # 0 in 8000 simulated null iterations) and below 3.5 (a -50 Elo/iteration
    # break stays detectable at >90% power inside one window -- at 4 sigma it
    # is 24%). See ``test_documented_power_at_the_shipped_line_reproduces``.
    window_se = observed_sd / math.sqrt(GateConfig().window_iters)
    sigmas = (null_mean - GateConfig().demote_delta_elo) / window_se
    assert 2.0 < sigmas < 3.5, (
        f"demote line sits {sigmas:.1f} window-sigma from the measured null"
    )

    # The loop gains ~0.02 Elo/iteration. The correction makes the central
    # negative STRONGER: more noise, not less, so "no gate can ratchet" holds a
    # fortiori. Guard the order of magnitude so nobody re-reads this as a ratchet.
    assert observed_sd / (0.02 / 2.8) > 1e3


def test_t_quantile_is_never_narrower_than_the_exact_t() -> None:
    """MUTATION: revert to the one-term Cornish-Fisher `z + (z**3+z)/(4 df)`.

    That form was shipped with a docstring claiming it was "conservative
    (wider)" below df=8. It is the opposite: 1.7% NARROWER at df=7 and 8.5%
    narrower at df=3 -- and df=3 is reachable, because `validate()` admits
    `min_iters >= 4`. A CI narrower than its stated alpha demotes more readily
    than advertised.

    The reference column is `scipy.stats.t.ppf(1 - alpha, df)`, recorded here
    as literals rather than imported: scipy is not a declared dependency of
    this package, and a test may not add one.
    """
    exact = {
        # df: (alpha 0.10, 0.05, 0.025, 0.01, 0.005) -- scipy.stats.t.ppf
        3: (1.63774, 2.35336, 3.18245, 4.54070, 5.84091),
        5: (1.47588, 2.01505, 2.57058, 3.36493, 4.03214),
        7: (1.41492, 1.89458, 2.36462, 2.99795, 3.49948),
        8: (1.39682, 1.85955, 2.30600, 2.89646, 3.35539),
        12: (1.35622, 1.78229, 2.17881, 2.68100, 3.05454),
        15: (1.34061, 1.75305, 2.13145, 2.60248, 2.94671),
        23: (1.31946, 1.71387, 2.06866, 2.49987, 2.80734),
        50: (1.29871, 1.67591, 2.00856, 2.40327, 2.67779),
        200: (1.28580, 1.65251, 1.97190, 2.34513, 2.60063),
    }
    old_form = {}
    for df, ref in exact.items():
        for alpha, e in zip((0.10, 0.05, 0.025, 0.01, 0.005), ref):
            got = _t_quantile(alpha, df)
            assert got >= e, (
                f"alpha={alpha} df={df}: {got:.5f} is NARROWER than the exact "
                f"t {e:.5f} -- the interval claims more confidence than it has"
            )
            assert got <= e * 1.001, (
                f"alpha={alpha} df={df}: {got:.5f} is {100*(got/e-1):.2f}% "
                "wider than exact; conservative is right, sloppy is not"
            )
            z = _Z_ONE_SIDED[alpha]
            old_form[(alpha, df)] = z + (z ** 3 + z) / (4.0 * df)

    # ...and pin that the OLD form really was anti-conservative, so the
    # rationale in the docstring cannot rot into folklore.
    assert old_form[(0.05, 3)] < exact[3][1] * 0.92
    assert old_form[(0.05, 7)] < exact[7][1] * 0.99

    # df=3 is reachable from a config edit, which is why this matters.
    GateConfig(mode=MODE_SHADOW, min_iters=4, window_iters=4).validate()


def test_elo_scale_matches_the_logistic_derivative_at_even_score() -> None:
    """MUTATION: change ELO_PER_SCORE_AT_HALF. Every threshold in the config is
    denominated in it."""
    eps = 1e-6
    def elo(p: float) -> float:
        return -400.0 * math.log10(1.0 / p - 1.0)
    numeric = (elo(0.5 + eps) - elo(0.5 - eps)) / (2 * eps)
    assert pytest.approx(numeric, rel=1e-6) == ELO_PER_SCORE_AT_HALF


def test_gate_reads_nothing_from_the_frozen_holdout() -> None:
    """MUTATION: feed ``test_metrics`` / ``test_policy_loss`` / ``best_loss``
    into the decision.

    Audit L16: ``policy_target`` is self-generated MCTS visits, so a frozen
    holdout is a DECAYING RULER -- ``test_policy_loss`` rises on a net that is
    getting better (+0.0392 frozen vs -0.0129 fresh, ckpt157->ckpt192). A gate
    on it rejects every model forever, which is the 2026-03 freeze reached by a
    second road. ``_update_best_model`` already has this defect and is why
    ``best_loss`` has not moved in 50+ iterations; the gate must not share it.
    """
    import ast
    import inspect

    import chess_anti_engine.tune.promotion_gate as pg

    import chess_anti_engine.tune.trainable_phases as phases

    tree = ast.parse(
        inspect.getsource(pg)
        + "\n"
        + inspect.getsource(phases._run_net_gating)
        + "\n"
        + inspect.getsource(phases._publish_iteration_model),
    )
    names = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
    } | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    } | {
        c.value for c in ast.walk(tree)
        if isinstance(c, ast.Constant) and isinstance(c.value, str)
    }
    # Docstrings ARE in ``names`` as constants, so strip the ones that are
    # module/class/function docs -- the prose above legitimately discusses the
    # decaying ruler; only an actual read of it is a defect.
    docstrings = {
        ast.get_docstring(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
    }
    names -= docstrings
    for banned in ("test_loss", "test_policy_loss", "test_wdl_loss",
                   "best_loss", "holdout_buf", "TrainMetrics", "test_metrics"):
        assert banned not in names, (
            f"the gate must not read {banned}: a frozen-holdout ruler decays "
            "(audit L16) and would reject every model forever"
        )


def test_no_torch_state_is_touched_by_the_gate() -> None:
    """The 2026-03 gate called ``model.load_state_dict(pre_train_state)``. The
    replacement must have no path that writes weights: a gate that can freeze
    the trunk is the trap this rebuild exists to avoid."""
    import inspect

    import chess_anti_engine.tune.promotion_gate as pg
    import chess_anti_engine.tune.trainable_phases as phases

    src = inspect.getsource(pg) + inspect.getsource(phases._run_net_gating)
    assert "load_state_dict(pre_train" not in src
    assert "pre_train_state" not in inspect.getsource(phases)
    assert torch is not None  # the gate module itself imports no torch
    assert "import torch" not in inspect.getsource(pg)


# --------------------------------------------------------------------------
# 9. THE SHADOW READOUT AND ITS NEGATIVE CONTROL
# --------------------------------------------------------------------------
# ``_LIVE_WINDOWS`` is the real offline reconstruction, not a plausible-looking
# table: every row is (iteration, cur_w, cur_d, cur_l, prev_w, prev_d, prev_l)
# produced by binning all 418 ``processed/_compacted/*.zarr/.zattrs`` by
# ``generated_at_unix`` against the live ``progress.csv`` (trial 13a9f, iters
# 163-219) and splitting each window's curriculum games by ``model_sha256``.
# ``w + d + l == curriculum_games`` on 418 of 418 shards, and 56 of 57 windows
# contain exactly two shas -- the signature the publish cadence predicts. The
# first five rows are the post-restart ramp and genuinely have no second arm;
# an earlier revision of this file carried an invented table whose first rows
# were (203, 33), (198, 41), ... which the real data does not remotely match.
#
# Reproduce with the recipe in the PR body; the derived quantities are asserted
# against ``OfflineReference`` by the first test below, so neither can rot
# without the other noticing.
_LIVE_WINDOWS = (
    (163, 0, 0, 0, 0, 0, 0),
    (164, 0, 0, 0, 0, 0, 0),
    (165, 0, 0, 0, 0, 0, 0),
    (166, 1, 0, 0, 0, 0, 0),
    (167, 7, 2, 0, 1, 0, 0),
    (168, 173, 16, 7, 2, 0, 0),
    (169, 162, 38, 20, 19, 2, 1),
    (170, 120, 76, 28, 21, 7, 2),
    (171, 84, 93, 32, 10, 13, 2),
    (172, 70, 144, 40, 13, 16, 3),
    (173, 56, 108, 33, 4, 23, 7),
    (174, 56, 138, 59, 9, 19, 8),
    (175, 50, 107, 43, 5, 23, 9),
    (176, 32, 81, 39, 18, 56, 25),
    (177, 42, 100, 38, 5, 13, 5),
    (178, 47, 125, 56, 5, 19, 10),
    (179, 34, 113, 54, 8, 18, 8),
    (180, 60, 105, 39, 8, 11, 5),
    (181, 72, 109, 41, 9, 31, 19),
    (182, 66, 111, 42, 4, 13, 3),
    (183, 65, 115, 43, 6, 13, 7),
    (184, 35, 107, 35, 6, 8, 8),
    (185, 40, 75, 31, 19, 37, 18),
    (186, 39, 129, 52, 13, 35, 10),
    (187, 51, 121, 51, 5, 13, 9),
    (188, 38, 101, 35, 7, 23, 5),
    (189, 32, 100, 38, 13, 45, 24),
    (190, 40, 104, 41, 13, 42, 20),
    (191, 51, 112, 53, 5, 8, 3),
    (192, 36, 103, 45, 8, 15, 4),
    (193, 47, 119, 50, 12, 25, 12),
    (194, 51, 94, 40, 6, 14, 4),
    (195, 41, 99, 57, 9, 15, 4),
    (196, 29, 83, 36, 15, 20, 6),
    (197, 63, 125, 38, 17, 32, 11),
    (198, 27, 75, 40, 8, 15, 8),
    (199, 55, 101, 39, 13, 29, 11),
    (200, 35, 136, 43, 14, 26, 16),
    (201, 52, 111, 45, 2, 17, 9),
    (202, 48, 106, 39, 10, 17, 7),
    (203, 54, 98, 38, 14, 34, 20),
    (204, 49, 122, 51, 6, 16, 5),
    (205, 56, 132, 50, 7, 21, 5),
    (206, 52, 89, 46, 9, 20, 7),
    (207, 27, 103, 37, 3, 15, 3),
    (208, 36, 72, 44, 16, 38, 17),
    (209, 43, 111, 44, 13, 25, 14),
    (210, 37, 84, 46, 12, 54, 19),
    (211, 45, 120, 66, 11, 41, 16),
    (212, 49, 115, 39, 7, 16, 8),
    (213, 47, 120, 55, 7, 15, 3),
    (214, 51, 118, 47, 5, 20, 10),
    (215, 43, 96, 38, 6, 15, 4),
    (216, 47, 118, 42, 10, 15, 8),
    (217, 52, 110, 57, 6, 10, 5),
    (218, 54, 124, 45, 8, 14, 4),
    (219, 50, 103, 51, 1, 11, 3),
)
_N_LIVE = len(_LIVE_WINDOWS)


def _live_samples() -> list[AnchoredSample]:
    return [
        AnchoredSample(it, cur_w=cw, cur_d=cd, cur_l=cl,
                       prev_w=pw, prev_d=pd, prev_l=pl)
        for it, cw, cd, cl, pw, pd, pl in _LIVE_WINDOWS
    ]


def _rows(samples: list[AnchoredSample]) -> list[tuple[int, int, float]]:
    """The three per-iteration columns the readout is written against."""
    return [
        (s.cur_games, s.prev_games,
         s.delta * ELO_PER_SCORE_AT_HALF
         if (s.cur_games and s.prev_games) else float("nan"))
        for s in samples
    ]


def _redeal(rng: random.Random, s: AnchoredSample, mode: str) -> AnchoredSample:
    """Destroy this window's cur/prev attribution, keeping the games themselves.

    ``coin``  -- every game is relabelled by a fair coin. This is what a random
                 shard-level mis-attribution produces, and it is the negative
                 control: the label carries no information at all afterwards.
    ``swap``  -- the two shas are exchanged.
    ``allcur``-- the prev sha stops being recognised and everything buckets to
                 cur (the arm sizes' most extreme failure).
    ``sized`` -- review round 2's control: relabel at random but CONDITION on
                 the realized arm sizes. See the test that uses it.
    """
    games = ([0] * (s.cur_w + s.prev_w) + [1] * (s.cur_d + s.prev_d)
             + [2] * (s.cur_l + s.prev_l))
    rng.shuffle(games)
    if mode == "coin":
        cur, prev = [], []
        for g in games:
            (cur if rng.random() < 0.5 else prev).append(g)
    elif mode == "sized":
        cur, prev = games[:s.cur_games], games[s.cur_games:]
    elif mode == "swap":
        cur, prev = games[:s.prev_games], games[s.prev_games:]
    elif mode == "allcur":
        cur, prev = games, []
    else:
        raise AssertionError(mode)
    return AnchoredSample(
        s.iteration,
        cur_w=cur.count(0), cur_d=cur.count(1), cur_l=cur.count(2),
        prev_w=prev.count(0), prev_d=prev.count(1), prev_l=prev.count(2),
    )


def test_offline_reference_matches_the_committed_reconstruction() -> None:
    """``OfflineReference``'s six numbers must be recomputable from the rows.

    MUTATION: edit any field of ``OfflineReference``. The shadow readout's
    deciding legs are all stated relative to it, so a reference that drifts
    away from the data silently retunes every leg at once.
    """
    usable = [s for s in _live_samples() if s.cur_games and s.prev_games]
    deltas = [s.delta * ELO_PER_SCORE_AT_HALF for s in usable]
    assert len(usable) == OFFLINE.n_usable == 53
    assert st.mean([s.cur_games for s in usable]) == pytest.approx(
        OFFLINE.mean_games_cur, abs=0.05)
    assert st.mean([s.prev_games for s in usable]) == pytest.approx(
        OFFLINE.mean_games_prev, abs=0.05)
    share = (sum(s.prev_games for s in usable)
             / sum(s.cur_games + s.prev_games for s in usable))
    assert share == pytest.approx(OFFLINE.prev_share, abs=0.0005)
    assert st.mean(deltas) == pytest.approx(OFFLINE.mean_delta_elo, abs=0.01)
    assert st.stdev(deltas) == pytest.approx(OFFLINE.sd_delta_elo, abs=0.01)


def test_offline_reference_is_internally_consistent() -> None:
    """MUTATION: store ``games_per_second`` as a literal again.

    It WAS a literal, 0.3411, and it made the reference disagree with itself by
    4.6% -- because 0.3411 was the mean of the per-row RATES while
    ``mean_games_*`` are means of per-row COUNTS, and ``E[g/s] != E[g]/E[s]``::

        mean_games_cur + mean_games_prev  = 235.1
        games_per_second * mean_iter_secs = 245.9   (4.6% apart)
        mean_games_prev                   =  38.3
        refresh_lag_seconds * games_per_s =  40.1   (4.6% apart)

    Two legs of one rule were then measuring the same window against two
    different reference loops, and 4.6% of the margin on each was being spent
    on the disagreement rather than on noise. Both identities must close.
    """
    total = OFFLINE.mean_games_cur + OFFLINE.mean_games_prev
    assert OFFLINE.games_per_second * OFFLINE.mean_iter_seconds == pytest.approx(
        total, rel=1e-9), "cadence x rate must reproduce the anchored game count"
    assert OFFLINE.refresh_lag_seconds * OFFLINE.games_per_second == pytest.approx(
        OFFLINE.mean_games_prev, abs=0.01), (
        "the refresh lag, priced at the reference rate, IS the prev arm -- that "
        "is the whole content of the cadence model"
    )
    assert OFFLINE.games_per_second == pytest.approx(0.32607, abs=1e-5)
    # ...and the share the model is built on is the one the counts imply.
    assert OFFLINE.mean_games_prev / total == pytest.approx(
        OFFLINE.prev_share, abs=0.0005)


def test_the_live_reconstruction_promotes() -> None:
    """The rule must PASS on the data it was written against.

    A negative control is only meaningful next to a positive one: a rule that
    killed everything would "pass" every shuffle test below for the wrong
    reason.
    """
    full = shadow_readout_verdict(_rows(_live_samples()), last_n=_N_LIVE)
    assert full.verdict == READOUT_PROMOTE, full
    # 53 windows have a two-sided split; 51 clear the floor of 15 a side.
    assert full.n_usable == 51
    assert full.usable_frac == pytest.approx(51 / 57, abs=0.005)

    # ...and on the last-40 slice the shipped command actually reads.
    last40 = shadow_readout_verdict(_rows(_live_samples()))
    assert last40.verdict == READOUT_PROMOTE, last40
    assert last40.mean_delta_elo == pytest.approx(0.11, abs=0.05)
    assert last40.sd_delta_elo == pytest.approx(43.83, abs=0.05)


def test_negative_control_reshuffled_attribution_is_killed() -> None:
    """THE NEGATIVE CONTROL. Destroy the attribution; the verdict must flip.

    This is the check two review rounds did not have. Round 1 found the
    deciding leg was 10x too loose and it was widened correctly; round 2 then
    re-ran the corrected rule with every game's ``model_sha256`` attribution
    randomly destroyed and watched all three legs pass on 176 of 200 reshuffles.
    A gate that passes with its signal destroyed is not testing what it claims,
    and only a negative control can see that -- correctness review cannot.

    So: 200 reshuffles of the REAL games, relabelled by a fair coin, which is
    what a random shard-level mis-attribution produces. Every one must KILL,
    and must KILL on a COUNT leg -- the attribution-sensitive quantity -- not
    incidentally on the spread.
    """
    live = _live_samples()
    killed, count_leg_fired = 0, 0
    for seed in range(200):
        rng = random.Random(seed)
        shuffled = [_redeal(rng, s, "coin") for s in live]
        r = shadow_readout_verdict(_rows(shuffled), last_n=_N_LIVE)
        killed += r.verdict == READOUT_KILL
        count_leg_fired += any(
            leg.startswith(("refresh_lag", "mean_games_")) for leg in r.failed_legs
        )
    assert killed == 200, "the readout must reject a destroyed attribution EVERY time"
    assert count_leg_fired == 200, (
        "it must reject on the counts -- the quantity attribution moves -- and "
        "not incidentally on a spread leg that a shuffle happens to disturb"
    )


@pytest.mark.parametrize("mode", ["swap", "allcur"])
def test_negative_control_covers_the_real_attribution_failure_modes(
    mode: str,
) -> None:
    """The two non-random ways the split actually breaks.

    ``swap``: cur and prev exchanged -- the sign inverts and every verdict
    means its opposite. ``allcur``: the prev sha stops being recognised, so the
    window empties. The second one used to raise ``statistics.StatisticsError``
    out of the ledger's shell command instead of reporting a verdict, i.e. the
    rule had no defined behaviour for its most important failure mode.
    """
    rng = random.Random(3)
    broken = [_redeal(rng, s, mode) for s in _live_samples()]
    r = shadow_readout_verdict(_rows(broken), last_n=_N_LIVE)
    assert r.verdict == READOUT_KILL, r
    assert r.failed_legs


def test_size_preserving_reshuffle_is_NOT_detectable_and_the_rule_says_so(
) -> None:
    """Review round 2's exact control, and the honest answer to it.

    Pool each window's games and redeal them into the SAME realized arm sizes.
    The attribution is destroyed, but the destruction is conditioned on the
    counts -- so the only channel left is the delta, and the true anchored
    effect (-4.33 Elo, 95% CI [-16.6, +7.9]) is not distinguishable from zero
    at n=53. No rule can pass this control, and claiming one does would be the
    dishonest way out.

    Two things are therefore asserted. First, that the DELTA legs really are
    blind to it -- which is why they must not be the deciding legs. Second,
    that this control is the ONLY one the shipped rule cannot flip: it leaves
    every count identical to the truth, which no real bug does, because shards
    are attributed whole and by sha.
    """
    live = _live_samples()
    true_r = shadow_readout_verdict(_rows(live), last_n=_N_LIVE)
    delta_blind = 0
    for seed in range(200):
        rng = random.Random(1000 + seed)
        shuffled = [_redeal(rng, s, "sized") for s in live]
        r = shadow_readout_verdict(_rows(shuffled), last_n=_N_LIVE)
        # the counts are untouched by construction -- that is the whole point
        assert r.mean_games_cur == pytest.approx(true_r.mean_games_cur)
        assert r.mean_games_prev == pytest.approx(true_r.mean_games_prev)
        delta_blind += abs(r.mean_delta_elo) <= 25.0 and 20.0 < r.sd_delta_elo < 70.0
    assert delta_blind > 150, (
        "if this ever FAILS, the delta legs have become informative about "
        "attribution -- a real finding, and a reason to revisit the rule. As "
        "written they are not, and the rule must not lean on them."
    )


def test_benign_cadence_change_is_not_reported_as_an_attribution_bug() -> None:
    """MUTATION: compare `prev_share` against the RAW reference instead of the
    cadence-adjusted expectation (i.e. `expected = ref.prev_share`).

    `prev` games are the fleet's model-refresh lag, roughly constant in
    SECONDS, so a longer iteration shrinks their share with nothing wrong:
    `corr(time_this_iter_s, prev_share) = -0.332` over the reference window,
    and a 1.5x cadence change *inside that window* moves the leg 0.0509 against
    a 0.06 tolerance. An earlier revision claimed the leg was
    "throughput-invariant". It is not, and a benign restart at a different
    cadence would have reported `kill` -- which the ledger reads as "the split
    is mis-attributing shards".

    THE SWEEP RUNS TO THE DECLARED FLOOR, AND UNDER BOTH CONSTRUCTIONS. It used
    to start at 0.5x, which is why nobody saw that every ratio in
    [0.40, ~0.46] -- INSIDE the then-declared trusted band of 0.4 -- returned
    `kill` on the ATTRIBUTION leg. `_CADENCE_RATIO_MIN` is now 0.6, and the
    reason it is 0.6 rather than 0.5 is that "healthy at k x cadence" has two
    defensible readings that diverge as k falls:

      pinned_prev  -- cur alone scales, prev keeps its absolute count (what
                      this test used to assume, and the only one swept)
      pinned_rate  -- TOTAL ingest scales, the refresh lag stays constant in
                      seconds, cur = total - prev (the model's own picture)

    They differ by 0.0000 at 1.0x, 0.0265 at 0.6x, 0.0455 at 0.5x and 0.0800 at
    0.4x, against a 0.06 tolerance. Sweeping only one of them cannot see that,
    so both are swept, at every 0.05 across the whole declared band.
    """
    def rows_for(mult: float, construction: str) -> list[tuple[int, int, float, float]]:
        secs = OFFLINE.mean_iter_seconds * mult
        prev = round(OFFLINE.mean_games_prev)
        if construction == "pinned_prev":
            cur = round(OFFLINE.mean_games_cur * mult)
        elif construction == "pinned_counts":
            cur = round(OFFLINE.mean_games_cur)
        else:
            total = (OFFLINE.mean_games_cur + OFFLINE.mean_games_prev) * mult
            cur = round(total - OFFLINE.mean_games_prev)
        return [(cur, prev, 8.0 * (1 if i % 2 else -1) * 5.5, secs) for i in range(40)]

    # THE BAND IS CONSTRAINED FROM BOTH SIDES. Round 5 mutated the two
    # constants and found only 0.4 caught: a 0.9 floor or a 1.5 ceiling passed
    # the whole suite, because this sweep is generated FROM the constants and
    # therefore shrinks with them. A 0.9 floor would already false-kill an
    # observed live window -- the current progress.csv's rolling-8 mean cadence
    # bottoms out at 0.893x -- so the band is asserted against the live data,
    # not just swept over itself.
    assert _CADENCE_RATIO_MIN <= 0.7, (
        "the floor must sit below the observed live rolling-8 minimum of "
        "0.893x with margin; 0.9 false-kills a window that has happened"
    )
    assert _CADENCE_RATIO_MAX >= 2.0, (
        "the ceiling must clear the observed live rolling-8 maximum "
        "(1.377x current, 2.909x on the oldest rotated file)"
    )

    steps = round((_CADENCE_RATIO_MAX - _CADENCE_RATIO_MIN) / 0.05)
    mults = [round(_CADENCE_RATIO_MIN + 0.05 * i, 10) for i in range(steps + 1)]
    assert mults[0] == pytest.approx(_CADENCE_RATIO_MIN)
    assert mults[-1] == pytest.approx(_CADENCE_RATIO_MAX)
    for construction in ("pinned_prev", "pinned_rate"):
        for mult in mults:
            r = shadow_readout_verdict(rows_for(mult, construction), last_n=40)
            assert r.verdict == READOUT_PROMOTE, (
                f"{construction} at cadence x{mult:.2f} is a HEALTHY loop and "
                f"must not be reported as an attribution bug: {r}"
            )

    # ...and outside the trusted band the CADENCE leg fires BY NAME and ALONE,
    # so an operator reads "your cadence moved", not "your attribution is
    # broken". Below the band the share model is the thing that stopped
    # holding, so its leg is not evaluated at all rather than extrapolated.
    for mult in (0.4, 3.5):
        r = shadow_readout_verdict(rows_for(mult, "pinned_prev"), last_n=40)
        assert r.verdict == READOUT_KILL, mult
        assert len(r.failed_legs) == 1, f"x{mult}: {r.failed_legs}"
        assert r.failed_legs[0].startswith("cadence"), r
        assert "CADENCE finding" in r.failed_legs[0]

    # THE THIRD CONSTRUCTION, which the comment used to call "the two
    # defensible pictures". `pinned_counts`: cadence moves because the TRAINING
    # phase changes length and selfplay is paused for it, so neither anchored
    # count moves. prev_share is then flat while the leg expects
    # refresh_lag/cadence, and a healthy loop IS false-killed inside the band.
    # It is not widened for -- it is fail-safe (kill only, never a false
    # promote), its mechanism is off in production
    # (`distributed_pause_selfplay_during_training: false`), and the reference
    # data disfavours it: it predicts corr(time, prev_share) = 0 where the 51
    # reference rows give -0.332. It is PINNED, so that if the band or the
    # tolerance moves, the documented range has to move with it.
    killed = [m for m in mults
              if shadow_readout_verdict(rows_for(m, "pinned_counts"),
                                        last_n=40).verdict != READOUT_PROMOTE]
    assert killed, "pinned_counts false-kills inside the band; say so if it stops"
    assert min(killed) == pytest.approx(0.60), killed
    assert max(killed) == pytest.approx(_CADENCE_RATIO_MAX), killed
    # ...and the range is contiguous at each end, 0.60-0.70 and 1.65-3.00.
    low = [m for m in killed if m < 1.0]
    high = [m for m in killed if m > 1.0]
    assert (min(low), max(low)) == pytest.approx((0.60, 0.70)), low
    assert (min(high), max(high)) == pytest.approx((1.65, 3.00)), high
    assert len(low) + len(high) == len(killed), killed


def test_a_window_shorter_than_the_preregistered_length_can_never_promote() -> None:
    """MUTATION: delete the ``window_too_short`` hold leg.

    The ledger pre-registers this readout as "run it after >=40 iterations
    carrying the ``gate_sample_*`` columns". Until ``_READOUT_MIN_ROWS``
    existed that precondition lived ONLY in that sentence: the rule returned
    ``promote_to_enforce`` off as few as TWO rows, and the deciding action of a
    promote is to set ``gate_mode: enforce`` on the live run.

    That is not hypothetical. ``harness._rotate_progress_csv_if_schema_
    changed`` starts a FRESH ``progress.csv`` whenever the reported key set
    changes -- three rotations in four days in this repo -- so an operator
    running the pre-committed command a few iterations after a rotation reads a
    3-row window.

    Every row below is the reference shape, so EVERY OTHER LEG PASSES: the only
    thing between these windows and a promote is the length leg, which is the
    whole point. A short window is a hold, not a kill -- nothing is broken, the
    window is simply not finished -- and it says so by name.
    """
    def rows(n: int) -> list[tuple[int, int, float, float]]:
        return [(round(OFFLINE.mean_games_cur), round(OFFLINE.mean_games_prev),
                 44.0 * (1 if i % 2 else -1), OFFLINE.mean_iter_seconds)
                for i in range(n)]

    for n in (2, 3, 5, 8, 12, 24, _READOUT_MIN_ROWS - 1):
        r = shadow_readout_verdict(rows(n), last_n=n)
        assert r.failed_legs == (), f"n={n} must break no leg: {r}"
        assert r.verdict == READOUT_HOLD, f"n={n} promoted on a short window: {r}"
        assert any(leg.startswith("window_too_short") for leg in r.hold_legs), r
        # The operator has to be able to see WHY, not just "not promoted".
        assert "HOLD: window_too_short" in str(r), str(r)

    # At the pre-registered length, and only there, the same shape promotes.
    r = shadow_readout_verdict(rows(_READOUT_MIN_ROWS), last_n=_READOUT_MIN_ROWS)
    assert r.verdict == READOUT_PROMOTE, r
    assert r.hold_legs == (), r
    assert _READOUT_MIN_ROWS == 40, "the ledger pre-registered 40; move both"

    # The other hold leg still works and is still named, so `hold_in_shadow`
    # never means "we did not say".
    offset = [(c, p, d + 24.0, s) for c, p, d, s in rows(_READOUT_MIN_ROWS)]
    r = shadow_readout_verdict(offset, last_n=_READOUT_MIN_ROWS)
    assert r.verdict == READOUT_HOLD, r
    assert r.failed_legs == (), r
    assert any(leg.startswith("|mean_delta_elo|") for leg in r.hold_legs), r


def test_partial_attribution_leak_sensitivity_floor() -> None:
    """The rule's BLIND SPOT, pinned so it stays known rather than surprising.

    "attribution failures move prev_share by 0.34-0.67" is true of TOTAL
    failures. A partial one-sided leak of fraction f is a different story, and
    a 10-20% leak from prev into cur is invisible to every leg. That is
    consistent with the whole-shard argument -- shards are attributed entire by
    sha, so a partial leak has no mechanism -- but a blind spot that nobody
    wrote down is how the last two review rounds went.
    """
    live = _live_samples()

    def leaked(f: float, direction: str) -> list[tuple[int, int, float, float]]:
        out = []
        for s_ in live:
            c = [s_.cur_w, s_.cur_d, s_.cur_l]
            pv = [s_.prev_w, s_.prev_d, s_.prev_l]
            src, dst = (pv, c) if direction == "prev->cur" else (c, pv)
            for i in range(3):
                n = round(src[i] * f)
                src[i] -= n
                dst[i] += n
            moved = AnchoredSample(s_.iteration, cur_w=c[0], cur_d=c[1], cur_l=c[2],
                                   prev_w=pv[0], prev_d=pv[1], prev_l=pv[2])
            out.append((moved.cur_games, moved.prev_games,
                        moved.delta * ELO_PER_SCORE_AT_HALF
                        if (moved.cur_games and moved.prev_games) else float("nan"),
                        OFFLINE.mean_iter_seconds))
        return out

    def verdict(f: float, direction: str) -> str:
        return shadow_readout_verdict(leaked(f, direction), last_n=_N_LIVE).verdict

    # prev -> cur: invisible up to 20%, and at 30% it is `usable_frac` that
    # notices (the prev arm drops under the floor), NOT `prev_share`.
    for f in (0.10, 0.20):
        assert verdict(f, "prev->cur") == READOUT_PROMOTE, f
    r30 = shadow_readout_verdict(leaked(0.30, "prev->cur"), last_n=_N_LIVE)
    assert r30.verdict == READOUT_KILL
    assert any(x.startswith("usable_frac") for x in r30.failed_legs), r30

    # cur -> prev is caught much earlier, on the refresh_lag/attribution leg
    # (named `prev_share` until audit G3-2), because the prev arm is small and
    # a leak into it moves the share a long way.
    assert verdict(0.06, "cur->prev") == READOUT_PROMOTE
    r08 = shadow_readout_verdict(leaked(0.08, "cur->prev"), last_n=_N_LIVE)
    assert r08.verdict == READOUT_KILL
    assert any(x.startswith("refresh_lag") for x in r08.failed_legs), r08


def test_every_deciding_leg_is_individually_load_bearing() -> None:
    """MUTATION: loosen ANY ONE leg's bound.

    The negative control above kills on the counts, but it moves several legs
    at once -- so widening a single tolerance escapes it while another leg
    happens to cover. That is the "leg that cannot fail" shape one level down,
    and it is how a rule rots: nobody removes a leg, they widen it.

    Each row below trips exactly ONE leg and leaves every other inside its
    bound, so each bound is asserted to be load-bearing on its own.
    """
    def series(cur: int, prev: int, mean: float, sd: float, *, secs: float = 721.0,
               n: int = 40, dead: int = 0) -> list[tuple[int, int, float, float]]:
        # deltas with exactly the requested mean and sample sd, by construction
        half = [mean - sd / math.sqrt(2), mean + sd / math.sqrt(2)]
        rows = [(cur, prev, half[i % 2], secs) for i in range(n)]
        return rows + [(0, prev, float("nan"), secs)] * dead

    # Each row trips exactly ONE leg. `secs` is chosen so the cadence-adjusted
    # `prev_share` expectation still matches, which is what makes the throughput
    # and share legs separable at all.
    cases: list[tuple[str, int, int, float, float, float, int]] = [
        # leg,               cur, prev,  mean,   sd,   secs, dead
        ("games_per_second",  92,   18,   0.0, 44.0,  721.0, 0),
        ("refresh_lag",      140,   56,   0.0, 44.0,  721.0, 0),
        ("sd_delta_elo",     197,   38,   0.0,  4.56, 721.0, 0),
        ("|mean_delta_elo|", 197,   38,  30.0, 44.0,  721.0, 0),
        ("usable_frac",      197,   38,   0.0, 44.0,  721.0, 10),
        ("cadence",          762,   38,   0.0, 44.0, 2500.0, 0),
    ]
    for leg, cur, prev, mean, sd, secs, dead in cases:
        r = shadow_readout_verdict(series(cur, prev, mean, sd, secs=secs, dead=dead),
                                   last_n=60)
        assert r.verdict == READOUT_KILL, f"{leg}: {r}"
        assert [x for x in r.failed_legs if x.startswith(leg)], (
            f"{leg} must be the leg that fires, got {r.failed_legs}"
        )
        assert len(r.failed_legs) == 1, (
            f"{leg} case must isolate ONE leg, got {r.failed_legs}"
        )

    # ...and the same construction on-shape passes, so the cases above fail for
    # the stated reason and not because the constructor is unrealistic.
    assert shadow_readout_verdict(series(197, 38, 0.0, 44.0),
                                  last_n=60).verdict == READOUT_PROMOTE


def test_readout_kills_when_wired_to_the_overlapping_window_column() -> None:
    """MUTATION: point the readout at ``gate_delta_elo`` instead of
    ``gate_sample_delta_elo``.

    That was round 1's finding 1: consecutive windows share ~95% of their
    samples, so the column's sd is 4.56 where the truth is 45.56, and a rule
    keyed to it cannot fail. The spread leg exists to make that visible rather
    than silent.
    """
    rows = [(200, 40, 3.42 + 0.9 * math.sin(i)) for i in range(40)]
    r = shadow_readout_verdict(rows)
    assert r.verdict == READOUT_KILL
    assert any(leg.startswith("sd_delta_elo") for leg in r.failed_legs), r


def test_readout_reports_a_verdict_instead_of_raising_on_a_dead_series() -> None:
    """No input may make the deciding command raise instead of deciding."""
    for rows in ([], [(0, 0, float("nan"))] * 40, [(200, 0, float("nan"))] * 40):
        r = shadow_readout_verdict(rows)
        assert r.verdict == READOUT_KILL
        assert r.failed_legs == ("usable_rows>=2",)


def test_usable_frac_denominator_counts_every_iteration_that_ran() -> None:
    """A rule stated twice is stated inconsistently, and this is where it bit.

    ``usable_frac`` has two defensible denominators: windows with a non-empty
    split, or every progress row. There is now one implementation, and the
    hold-shaped rows -- ``games_cur == 0`` -- stay in the denominator, because
    an iteration that produced no anchored sample is exactly what the leg is
    meant to count.

    THE WORKED EXAMPLE, so the two numbers exist somewhere a reader can check
    rather than only in prose: on the committed reconstruction, 51 windows
    clear the floor, 53 have a non-empty split, and 57 rows ran. 51/53 = 0.962
    against 51/57 = 0.895, with the kill line at 0.85 -- seven points apart
    with four points of margin, which is close enough that a slightly worse
    window has the two denominators returning opposite verdicts. The shipped
    denominator is every row that ran.
    """
    live = _live_samples()
    rows = _rows(live)
    assert len(rows) == 57
    r = shadow_readout_verdict(rows, last_n=_N_LIVE)
    assert r.n_rows == 57
    assert r.usable_frac == pytest.approx(51 / 57, abs=0.005)

    split_rows = [s for s in live if s.cur_games and s.prev_games]
    assert len(split_rows) == 53
    assert r.n_usable == 51
    rejected_denominator = 51 / len(split_rows)   # windows with a split: 0.962
    shipped_denominator = 51 / len(rows)          # every row that ran:   0.895
    assert rejected_denominator == pytest.approx(0.962, abs=0.001)
    assert shipped_denominator == pytest.approx(0.895, abs=0.001)
    # Both clear the line on THIS window -- the defect was that they are seven
    # points apart with four points of margin, not that they disagree here.
    assert min(rejected_denominator, shipped_denominator) > 0.85
    assert rejected_denominator - shipped_denominator == pytest.approx(0.067, abs=0.001)

    # Six more hold-shaped iterations drag it under the line and flip the
    # verdict; they must not be silently dropped.
    padded = shadow_readout_verdict(rows + [(0, 40, float("nan"))] * 6,
                                    last_n=_N_LIVE + 6)
    assert padded.n_rows == 63
    assert padded.verdict == READOUT_KILL
    assert any(leg.startswith("usable_frac") for leg in padded.failed_legs)


def test_the_deciding_command_reads_the_sample_columns_from_a_real_csv(
    tmp_path: Path,
) -> None:
    """MUTATION: emit the window aggregates under the sample column names.

    ``shadow_readout_from_csv`` IS the ledger's deciding command -- the ledger
    quotes a one-line call to it rather than restating the arithmetic, so the
    worked example and the shipped command cannot disagree again.
    """
    path = tmp_path / "progress.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "training_iteration", "gate_delta_elo", "gate_sample_games_cur",
            "gate_sample_games_prev", "gate_sample_delta_elo"])
        w.writeheader()
        for s in _live_samples():
            w.writerow({
                "training_iteration": s.iteration,
                "gate_delta_elo": 3.42,          # the window column: a decoy
                "gate_sample_games_cur": s.cur_games,
                "gate_sample_games_prev": s.prev_games,
                "gate_sample_delta_elo": (
                    s.delta * ELO_PER_SCORE_AT_HALF
                    if (s.cur_games and s.prev_games) else float("nan")),
            })
    r = shadow_readout_from_csv(path, last_n=_N_LIVE)
    assert r.verdict == READOUT_PROMOTE, r
    assert r.n_rows == 57
    assert r.sd_delta_elo == pytest.approx(OFFLINE.sd_delta_elo, abs=1.0)


def test_the_pooled_count_identity_is_a_deciding_leg(tmp_path: Path) -> None:
    """MUTATION: drop the ``anchored_games_vs_pooled`` leg, or stop reading
    ``pid_curriculum_*`` in ``shadow_readout_rows_from_csv``.

    ``gate_sample_games_cur + gate_sample_games_prev
      == pid_curriculum_w + pid_curriculum_d + pid_curriculum_l``
    is exact BY CONSTRUCTION -- ``_process_shard`` increments one arm and the
    pool in the same branch -- and both sides are already progress.csv columns.
    Every other leg in this rule is a band around an estimate; this one is an
    integer identity, so it catches shard loss and unrecognised-sha bucketing
    outright, with no statistics and no tolerance to widen later.
    """
    def write(path: Path, *, lose: int) -> None:
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "training_iteration", "gate_sample_games_cur",
                "gate_sample_games_prev", "gate_sample_delta_elo",
                "time_this_iter_s", "pid_curriculum_w", "pid_curriculum_d",
                "pid_curriculum_l"])
            w.writeheader()
            for s in _live_samples():
                w.writerow({
                    "training_iteration": s.iteration,
                    # the SPLIT loses `lose` games that the POOL still counts
                    "gate_sample_games_cur": max(s.cur_games - lose, 0),
                    "gate_sample_games_prev": s.prev_games,
                    "gate_sample_delta_elo": (
                        s.delta * ELO_PER_SCORE_AT_HALF
                        if (s.cur_games and s.prev_games) else float("nan")),
                    "time_this_iter_s": OFFLINE.mean_iter_seconds,
                    "pid_curriculum_w": s.cur_w + s.prev_w,
                    "pid_curriculum_d": s.cur_d + s.prev_d,
                    "pid_curriculum_l": s.cur_l + s.prev_l,
                })

    clean = tmp_path / "clean.csv"
    write(clean, lose=0)
    r = shadow_readout_from_csv(clean, last_n=_N_LIVE)
    assert r.verdict == READOUT_PROMOTE, r

    # ONE lost game per iteration -- far too small for any band leg to see, and
    # nowhere near the 0.06 prev_share tolerance -- must still kill.
    leaky = tmp_path / "leaky.csv"
    write(leaky, lose=1)
    r = shadow_readout_from_csv(leaky, last_n=_N_LIVE)
    assert r.verdict == READOUT_KILL, r
    assert r.failed_legs == tuple(
        x for x in r.failed_legs if x.startswith("anchored_games_vs_pooled")), r

    # ...and a window with no pid_curriculum columns SKIPS the leg rather than
    # passing it, so a schema change cannot quietly retire the check.
    assert shadow_readout_verdict(
        [(197, 38, 8.0 * (1 if i % 2 else -1) * 5.5, 721.0) for i in range(40)],
    ).verdict == READOUT_PROMOTE


def test_the_documented_yardstick_is_actually_runnable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """MUTATION: delete ``scripts/gate_shadow_readout.py``.

    ``shadow_readout_from_csv`` was documented as "the ledger's ONE deciding
    command, as a function" while no script or CLI invoked it -- a yardstick
    pre-committed as an exact command that could not be run as one. The rule
    and the command must be the same code, and the command must exist.

    Driven through ``main()`` rather than by reading the file, so this fails if
    the script stops working, not merely if it stops existing.

    EVERY DOCUMENTED EXIT CODE IS DRIVEN, because round 5 found exit 1 was not.
    Mutating the verdict->code mapping so that ``hold_in_shadow`` reported as 0
    escaped all 73 tests, which silently converts "extend the window, do not
    promote" into the deciding action of the whole readout.

    THE CONFOUND COLUMN IS NOW WRITTEN BY ``write`` (audit G3-2 x K1). Before
    it was absent, so every window this test drove had ``n_confound == 0`` and
    the command still exited 0 -- which is exactly the exit-code trap the
    ledger pre-registered against: fixing the attribution leg alone makes the
    same run exit 0 with the PID-confound leg measuring nothing. Exit 5 is
    driven below on a csv without the column.
    """
    from scripts import gate_shadow_readout
    from scripts.gate_shadow_readout import main

    codes = (READOUT_EXIT_PROMOTE, READOUT_EXIT_HOLD, READOUT_EXIT_KILL,
             READOUT_EXIT_CONFOUND_UNMEASURED, READOUT_EXIT_IDENTITY_UNEVALUATED)
    assert sorted(codes) == [0, 1, 2, 5, 6], codes
    # 3 (did not run) and 4 (no such file) must stay reserved for the
    # script's own preconditions.
    assert 3 not in codes, codes
    assert 4 not in codes, codes
    # --help IS the module docstring, so every code an operator can branch on
    # has to be named there.
    help_text = gate_shadow_readout.__doc__ or ""
    for documented in ("promote_to_enforce (0)", "hold_in_shadow     (1)",
                       "kill               (2)", "not run            (3)",
                       "no such file       (4)",
                       "confound unmeasured (5)",
                       "identity not evaluated (6)"):
        assert documented in help_text, documented

    def write(p: Path, samples: list[AnchoredSample], *, shift: float = 0.0,
              confound: bool = True, pooled: bool = True) -> None:
        cols = ["training_iteration", "gate_delta_elo", "gate_sample_games_cur",
                "gate_sample_games_prev", "gate_sample_delta_elo",
                "time_this_iter_s"]
        if pooled:
            cols += ["pid_curriculum_w", "pid_curriculum_d", "pid_curriculum_l"]
        if confound:
            cols.append("gate_sample_confound_elo")
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for i, s in enumerate(samples):
                row = {
                    "training_iteration": s.iteration,
                    "gate_delta_elo": 3.42,      # the window column: a decoy
                    "gate_sample_games_cur": s.cur_games,
                    "gate_sample_games_prev": s.prev_games,
                    "gate_sample_delta_elo": (
                        s.delta * ELO_PER_SCORE_AT_HALF + shift
                        if (s.cur_games and s.prev_games) else float("nan")),
                    "time_this_iter_s": OFFLINE.mean_iter_seconds,
                }
                if pooled:
                    # The exact-integer identity: an accepted shard increments
                    # an arm AND the pool in the same branch of _process_shard.
                    row["pid_curriculum_w"] = s.cur_w + s.prev_w
                    row["pid_curriculum_d"] = s.cur_d + s.prev_d
                    row["pid_curriculum_l"] = s.cur_l + s.prev_l
                if confound:
                    # A predicted PID-lag offset with spread and no relation to
                    # the delta beside it: the fit is well posed and the slope
                    # is nowhere near the 0.5 the hold leg fires on.
                    row["gate_sample_confound_elo"] = _WOBBLE[i % len(_WOBBLE)] * 100.0
                w.writerow(row)

    path = tmp_path / "progress.csv"
    write(path, _live_samples())

    monkeypatch.setattr("sys.argv", ["gate_shadow_readout.py", str(path),
                                     "--last-n", str(_N_LIVE), "--verbose"])
    assert main() == 0, "the reference reconstruction must exit 0 = promote"
    out = capsys.readouterr().out
    assert READOUT_PROMOTE in out
    assert "prev_share=0.16" in out, out

    # EXIT 1 = hold_in_shadow, documented since the script shipped and never
    # driven. Mutating `_EXIT` to report a hold as a promote escaped the whole
    # suite, which turns "extend the window, do not promote" into the deciding
    # action. Reached two ways, because there are two hold legs:
    #
    #   (a) every leg passes but the anchored offset is larger than expected
    hold = tmp_path / "hold.csv"
    write(hold, _live_samples(), shift=24.0)
    monkeypatch.setattr("sys.argv", ["gate_shadow_readout.py", str(hold),
                                     "--last-n", str(_N_LIVE)])
    assert main() == 1, "hold_in_shadow must not share an exit code with promote"
    out = capsys.readouterr().out
    assert out.startswith(READOUT_HOLD), out
    assert "FAILED:" not in out, out
    assert "HOLD: |mean_delta_elo|" in out, out

    #   (b) the window is shorter than the pre-registered 40 iterations. A
    #       fresh progress.csv after a report-schema rotation is exactly this,
    #       and it must never read as "promote to enforce".
    #       The three rows are the REFERENCE SHAPE, so every other leg passes
    #       and the length is the only thing standing between this file and
    #       exit 0.
    short = tmp_path / "short.csv"
    with short.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "gate_sample_games_cur", "gate_sample_games_prev",
            "gate_sample_delta_elo", "time_this_iter_s",
            "pid_curriculum_w", "pid_curriculum_d", "pid_curriculum_l",
            "gate_sample_confound_elo"])
        w.writeheader()
        for i in range(3):
            cur, prev = round(OFFLINE.mean_games_cur), round(OFFLINE.mean_games_prev)
            w.writerow({
                "gate_sample_games_cur": cur,
                "gate_sample_games_prev": prev,
                "gate_sample_delta_elo": 44.0 * (1 if i % 2 else -1),
                "time_this_iter_s": OFFLINE.mean_iter_seconds,
                "pid_curriculum_w": cur + prev, "pid_curriculum_d": 0,
                "pid_curriculum_l": 0,
                "gate_sample_confound_elo": _WOBBLE[i] * 100.0,
            })
    monkeypatch.setattr("sys.argv", ["gate_shadow_readout.py", str(short)])
    assert main() == 1, "a 3-row window must hold, never promote"
    out = capsys.readouterr().out
    assert out.startswith(READOUT_HOLD), out
    assert "HOLD: window_too_short" in out, out

    # A destroyed attribution must exit NON-ZERO, or the command cannot be a
    # kill rule regardless of what it prints.
    rng = random.Random(7)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "gate_sample_games_cur", "gate_sample_games_prev",
            "gate_sample_delta_elo", "time_this_iter_s",
            "pid_curriculum_w", "pid_curriculum_d", "pid_curriculum_l",
            "gate_sample_confound_elo"])
        w.writeheader()
        for i, s in enumerate(_redeal(rng, x, "coin") for x in _live_samples()):
            w.writerow({
                "gate_sample_games_cur": s.cur_games,
                "gate_sample_games_prev": s.prev_games,
                "gate_sample_delta_elo": (
                    s.delta * ELO_PER_SCORE_AT_HALF
                    if (s.cur_games and s.prev_games) else float("nan")),
                "time_this_iter_s": OFFLINE.mean_iter_seconds,
                # A coin-shuffle moves games BETWEEN arms and conserves the
                # pool, which is exactly why the identity leg cannot catch it
                # and the share leg can.
                "pid_curriculum_w": s.cur_w + s.prev_w,
                "pid_curriculum_d": s.cur_d + s.prev_d,
                "pid_curriculum_l": s.cur_l + s.prev_l,
                "gate_sample_confound_elo": _WOBBLE[i % len(_WOBBLE)] * 100.0,
            })
    monkeypatch.setattr("sys.argv", ["gate_shadow_readout.py", str(path),
                                     "--last-n", str(_N_LIVE)])
    assert main() == 2
    assert "FAILED:" in capsys.readouterr().out

    # EXIT 5 = the PID-confound axis has no measurement. THE POINT OF THE
    # WHOLE PER-AXIS RULE: this csv is the reference reconstruction that exits
    # 0 above, byte-for-byte, minus the confound column -- which is the shape
    # every live progress.csv row written so far has, because the server
    # compactor rebuilt ShardMeta without opponent_wdl_regret_limit (#323 fixes
    # the producer; it is deploy-gated on a server restart). The verdict is
    # still promote_to_enforce; the command must NOT say 0.
    noconf = tmp_path / "noconfound.csv"
    write(noconf, _live_samples(), confound=False)
    monkeypatch.setattr("sys.argv", ["gate_shadow_readout.py", str(noconf),
                                     "--last-n", str(_N_LIVE)])
    assert main() == 5, (
        "a window with no confound measurement must not share an exit code "
        "with promote OR with hold"
    )
    out = capsys.readouterr().out
    assert READOUT_PROMOTE in out, "the verdict itself is unchanged"
    assert "CONFOUND UNMEASURED" in out, out
    assert LEG_UNMEASURED in out, out

    # EXIT 6 = the pooled-identity axis was never evaluated. Same OR-over-axes
    # rule as 5, on the leg the module calls "the one with no statistics in
    # it", and reachable on any csv rotated from an earlier report schema.
    nopooled = tmp_path / "nopooled.csv"
    write(nopooled, _live_samples(), pooled=False)
    monkeypatch.setattr("sys.argv", ["gate_shadow_readout.py", str(nopooled),
                                     "--last-n", str(_N_LIVE)])
    assert main() == 6, (
        "a window whose exact-integer identity was skipped must not share an "
        "exit code with promote"
    )
    out = capsys.readouterr().out
    assert READOUT_PROMOTE in out, "the verdict itself is unchanged"
    assert "POOLED IDENTITY NOT EVALUATED" in out, out

    # ...and --refresh-lag-seconds 0 is REFUSED rather than dividing by zero
    # inside the leg's own message: no input may make this command raise.
    monkeypatch.setattr("sys.argv", ["gate_shadow_readout.py", str(path),
                                     "--refresh-lag-seconds", "0"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2, "argparse refuses, the module does not divide"
    capsys.readouterr()

    # A missing file exits 4 and says where to look, rather than traceback-ing
    # out of a command the ledger pre-committed to. FOUR, not 2: it used to
    # share `kill`'s code, which re-created for anything branching on the exit
    # status the exact did-not-run/verdict confusion exit 3 was added to fix.
    monkeypatch.setattr("sys.argv", ["gate_shadow_readout.py",
                                     str(tmp_path / "nope.csv")])
    assert main() == 4
    assert "no such file" in capsys.readouterr().out

    # ...and a csv with NO gate columns -- every progress.csv written before
    # this module shipped -- reports NOT RUN, not `kill`. An empty window looks
    # identical to a failed rule from inside the readout, and "did not run
    # reported as a verdict" is the defect this whole module exists to remove.
    bare = tmp_path / "bare.csv"
    with bare.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["training_iteration", "train_loss"])
        w.writeheader()
        w.writerow({"training_iteration": 1, "train_loss": 1.0})
    monkeypatch.setattr("sys.argv", ["gate_shadow_readout.py", str(bare)])
    assert main() == 3
    out = capsys.readouterr().out
    assert out.startswith("NOT RUN")
    assert not out.startswith(READOUT_KILL)


def test_a_hold_erases_its_own_evidence(tmp_path: Path) -> None:
    """The enforce-mode brake is PARTIAL, and this is the mechanism.

    §7 of the PR quoted "~16% of iterations held" -- measured at the RETIRED
    -45 line, and against a model its own author flagged as an underestimate.
    At the shipped -25 it is 55-66%. That is a materially different brake, and
    the reason is not where the line sits:

    during a hold the fleet is on ONE sha, so every accepted shard buckets to
    prev, `cur_games == 0`, and `_run_net_gating` observes a row of ZEROS.
    Those rows occupy slots in the 24-long window. Twelve of them evict half
    the real samples, `len(usable)` drops under `min_iters`, the verdict goes
    NOT_RUN -- `acted=False` -- and the brake releases. `delta_elo` is frozen
    at its pre-hold value throughout, because no new sample can form.

    The hold erases its own evidence. Fail-open, so nothing is unsafe; but it
    is the thing to decide before `enforce`, and it must not be re-quoted at
    16% again.
    """
    anchor = tmp_path / "gate_promoted_model.pt"
    anchor.write_bytes(b"anchor")
    gate = _gate(min_games_per_side=5)
    ctrl = GateHoldController(gate=gate, promoted_model_path=anchor)

    held = zero_rows = not_run = 0
    longest = run = 0
    deltas_during_hold: list[float] = []
    for i in range(120):
        ctrl.note_published()
        holding = ctrl.hold_path is not None
        if holding:
            gate.observe(AnchoredSample(i))       # the zero row
            zero_rows += 1
        else:
  # A sustained regression WITH per-iteration spread. It used to be a
  # constant 0.35 vs 0.50, i.e. the same delta every iteration -- a window
  # whose spread is zero, which `degenerate_variance` is supposed to
  # refuse. It reached a DEMOTE anyway, on a confidence interval of width
  # zero, because the old guard tested `se <= 0.0` and the float
  # round-trip left se at ~1e-17 (audit G3-3). So this test was driving
  # the actuator through the bug it was measuring around. `_WOBBLE` is the
  # same fixed-spread idiom `_window` uses; the mean regression is
  # unchanged at -0.15 (-208 Elo).
            gate.observe(_sample(i, 0.35 + _WOBBLE[i % len(_WOBBLE)], 0.50))
        d = gate.apply(gate.decide())
        not_run += d.decision == DECISION_NOT_RUN
        if holding and not math.isnan(d.delta_elo):
            deltas_during_hold.append(d.delta_elo)
        ctrl.on_decision(d)
        if ctrl.hold_path is not None:
            held += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0

    assert held > 0, "the gate must be able to act at all"
    assert held < 120, "and the brake must be PARTIAL, not 100%"
    assert 0.3 < held / 120 < 0.9, f"held {held}/120 -- restate §7 if this moved"
    assert longest <= gate.cfg.max_hold_iters, "the release path must bound a hold"
    # A hold takes effect on the NEXT publish, so the zero rows trail the
    # held count by at most one.
    assert abs(zero_rows - held) <= 1, (
        f"every held iteration must contribute a zero row (held {held}, "
        f"zero rows {zero_rows}) -- that is the eviction mechanism"
    )
    assert not_run > 0, (
        "the window must actually go blind: zero rows evicting real samples "
        "is why the brake releases, not the line position"
    )
    # ...and the reported delta is frozen while held, because no new sample
    # can form. If this ever starts varying, the mechanism has changed.
    assert len({round(x, 6) for x in deltas_during_hold}) < len(deltas_during_hold)


def test_documented_power_at_the_shipped_line_reproduces() -> None:
    """The power numbers in the module docstring, recomputed from scratch.

    An earlier revision of the PR published 14% / 37% for the -45 line. Those
    do not reproduce; the -25 rows do. A docstring number nobody recomputes is
    a future citation -- this repo has "63M not 78.8M" in its memory index for
    exactly that reason.
    """
    sd, null = OFFLINE.sd_delta_elo, OFFLINE.mean_delta_elo

    def power(line: float, k: int, mu: float) -> float:
        """P(the window mean's upper bound clears ``line``) when the true
        per-iteration mean is ``mu``."""
        se = sd / math.sqrt(k)
        crit = line - _t_quantile(0.05, k - 1) * se
        return 0.5 * (1.0 + math.erf(((crit - mu) / se) / math.sqrt(2.0)))

    # A -50 Elo/iteration break lands ON TOP of the measured null. These moved
    # by ~0.5 point when `_t_quantile` was made exact -- the old approximation
    # was anti-conservative, so it OVERSTATED power. They now agree with
    # scipy.stats.t to two decimals; the -45 rows are the ones that matter,
    # because the retired line looks even weaker than first corrected.
    assert power(-45.0, 8, null - 50.0) == pytest.approx(0.0942, abs=0.002)
    assert power(-45.0, 24, null - 50.0) == pytest.approx(0.2386, abs=0.002)
    assert power(-25.0, 8, null - 50.0) == pytest.approx(0.4706, abs=0.002)
    assert power(-25.0, 24, null - 50.0) == pytest.approx(0.9250, abs=0.002)
    # The shipped line is the one that delivers the documented capability.
    assert GateConfig().demote_delta_elo == -25.0
    assert power(-25.0, GateConfig().window_iters, null - 50.0) > 0.90

    # The warm-start class (-6.7 Elo/iteration) stays out of reach at every
    # reachable K and line, under either reading of "break": the worst case
    # over K in {8, 12, 16, 24} x line in {-25, -45} is 0.29%, at K=8 / -25.
    worst = max(power(line, k, mu)
                for line in (-25.0, -45.0) for k in (8, 12, 16, 24)
                for mu in (null - 6.7, -6.7))
    assert worst == pytest.approx(0.0029, abs=0.0003), worst
    assert round(100 * worst, 3) == 0.287


def test_the_documented_power_numbers_are_quoted_consistently_everywhere() -> None:
    """MUTATION: leave a stale power number in ANY of the ELEVEN copies.

    These numbers live in four places -- the module docstring (a table AND the
    prose under it), the comment sitting next to the shipped
    ``demote_delta_elo`` constant (a table AND its ``Exactly:`` line), the
    production yaml, and the test above. Review round 4 found the GateConfig
    comment still carrying ``10.0%`` / ``48.3%``, computed with the RETIRED
    anti-conservative ``_t_quantile``, while the module docstring three hundred
    lines up explicitly said those two numbers were wrong. The copy nobody
    recomputes is the one that gets cited; that is why "63M not 78.8M" is in
    this repo's memory index.

    Round 5 then found that THIS test's first revision pinned only 5 of the 11
    copies: it inspected the GateConfig comment's two table lines and two
    warm-start percentages, and nothing at all in ``promotion_gate.__doc__``'s
    own table -- the copy the PR body quotes and the first thing any reader
    hits -- nor the yaml's ``92.5%`` / ``23.9%``. Six mutations escaped the
    whole suite. Every copy is now pinned against the source text.

    The values are ``scipy.stats.t``-exact: 9.42 / 23.87 / 47.06 / 92.51, and
    every copy agrees to the precision it displays. The shipped ``_t_quantile``
    is deliberately conservative and returns 9.42 / 23.86 / 47.06 / 92.50 --
    within 0.01 point, and never optimistic -- which is why the ``Exactly:``
    line quotes both.
    """
    src = inspect.getsource(promotion_gate)
    doc = promotion_gate.__doc__ or ""
    yaml_src = (Path(__file__).resolve().parents[1]
                / "configs" / "pbt2_small.yaml").read_text()

    # -- copy 1: the MODULE DOCSTRING's table (the one the PR body quotes) ---
    doc_table = [ln.strip() for ln in doc.splitlines() if "demote line" in ln]
    assert len(doc_table) == 2, doc_table
    for quoted in ("K=8 -> **47.1%**", "K=24 -> **92.5%**"):
        assert quoted in doc_table[0], doc_table[0]
    for quoted in ("K=8 ->  9.4%", "K=24 -> 23.9%"):
        assert quoted in doc_table[1], doc_table[1]

    # -- copy 2: the prose under it, which repeats two of the four ----------
    assert "the numbers are 9.4% / 47.1% and agree with" in doc
    for retired in ("10.0%", "48.3%", "14% / 37%"):
        # Named on purpose as retired; they must stay OUT of the tables above.
        assert not [ln for ln in doc_table if retired in ln], doc_table

    # -- copy 3: the GateConfig comment's table -----------------------------
    cfg_comment = src.split("demote_delta_elo: float")[0].split(
        "# -25, and the derivation is a FALSE-BRAKE BUDGET")[1]
    # The TABLE rows only: the prose below them names the retired numbers on
    # purpose, and a check that cannot tell the table from the note explaining
    # why the table changed is a check nobody can keep green.
    table = [ln.strip() for ln in cfg_comment.splitlines() if "line -" in ln]
    assert len(table) == 2, table
    for quoted in ("K=8 ->  9.4%", "K=24 -> 23.9%"):
        assert quoted in table[0], table[0]
    for quoted in ("K=8 -> 47.1%", "K=24 -> 92.5%"):
        assert quoted in table[1], table[1]
    for retired in ("10.0%", "48.3%"):
        assert not [ln for ln in table if retired in ln], (
            f"{retired} came from the retired anti-conservative _t_quantile"
        )

    # -- copy 4: the GateConfig comment's two-decimal `Exactly:` line -------
    exact = "".join(cfg_comment.split("# Exactly, against")[1].splitlines()[:4])
    assert "9.42 / 23.87 / 47.06 / 92.51" in exact, exact
    assert "9.42 / 23.86 / 47.06 / 92.50" in exact, exact

    # -- copy 5: the warm-start worst case, 0.287% -------------------------
    # asserted as 0.0029 by the test above and rounded to 0.29% in the yaml.
    # The docstring used to round it UP to 0.32%.
    assert "0.287%" in doc
    assert "0.32%" not in doc

    # -- copy 6: the PRODUCTION YAML, all three of its quoted percentages ---
    assert "caught at 92.5% power" in yaml_src
    assert "only 23.9% at -45" in yaml_src
    assert "worst-case power 0.29%" in yaml_src


# --------------------------------------------------------------------------
# 10. the loop's hold state machine, driven end to end
# --------------------------------------------------------------------------
def _controller(tmp_path: Path, *, mode: str = MODE_ENFORCE,
                hold_active: bool = False, anchor: bool = True,
                stamp: bool = True, stamp_iteration: int = 5,
                current_iteration: int | None = 5) -> GateHoldController:
    g = _gate(mode=mode)
    g.hold_active = hold_active
    if anchor and mode != MODE_OFF:
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / "gate_promoted_model.pt"
        path.write_bytes(b"anchor")
        # Production writes the stamp in the same try as the copy, so a
        # stamped anchor is the ONLY shape a real refresh leaves behind.
        if stamp:
            write_anchor_stamp(
                path, iteration=stamp_iteration, trainer_step=1,
                model_sha256="deadbeef", trial_id="t0",
            )
    return GateHoldController.create(
        g, durable_dir=tmp_path, current_iteration=current_iteration,
    )


def test_hold_controller_restores_a_hold_across_a_restart(tmp_path: Path) -> None:
    """MUTATION (reviewer R1): startup restore always returns None.

    A restart mid-hold would lift the brake, and the next publish would
    overwrite the anchor with the very weights being held back.
    """
    c = _controller(tmp_path, hold_active=True)
    assert c.hold_path == tmp_path / "gate_promoted_model.pt"
    assert c.gate.hold_active is True

    # No anchor on disk -> release rather than trust a hold we cannot serve.
    c2 = _controller(tmp_path / "empty", hold_active=True, anchor=False)
    assert c2.hold_path is None
    assert c2.gate.hold_active is False


def test_hold_controller_ages_and_releases_through_aborted_iterations(
    tmp_path: Path,
) -> None:
    """MUTATIONS (reviewer R2 + R3): drop the release, or drop the ageing.

    R3 freezes the counter, so a run stuck in `sp.should_retry` holds forever.
    R2 leaves `hold_path` set after `max_hold_iters` ages out, so the fleet is
    held forever *after* the brake was supposed to release.
    """
    c = _controller(tmp_path, hold_active=True)
    c.gate.holds = 1
    for _ in range(c.gate.cfg.max_hold_iters - 2):
        c.on_aborted_iteration()
        assert c.hold_path is not None
    c.on_aborted_iteration()
    assert c.hold_path is None, "the brake must release, not just stop counting"
    assert c.gate.hold_active is False


def test_hold_controller_threads_a_verdict_to_the_next_publish(
    tmp_path: Path,
) -> None:
    """MUTATION (reviewer R4): the verdict never becomes the next hold path."""
    c = _controller(tmp_path)
    promoted = tmp_path / "gate_promoted_model.pt"
    _g, demote = _demoting_gate(promoted)
    c.on_decision(demote)
    assert c.hold_path == promoted
    c.on_decision(_feed(_gate(demote_delta_elo=-50.0), _window(0.0)))
    assert c.hold_path is None


def test_disabled_gate_has_no_anchor_and_therefore_no_copy(tmp_path: Path) -> None:
    """MUTATION (reviewer R6): the anchor path becomes unconditional and the
    252 MB copy returns at `gate_mode: off`."""
    c = _controller(tmp_path, mode=MODE_OFF, hold_active=True)
    assert c.promoted_model_path is None
    assert c.hold_path is None


def test_sample_is_invalid_across_a_hold_transition(tmp_path: Path) -> None:
    """MUTATION: always report the sample as valid.

    The sign INVERTS on the iteration that first serves the anchor:
    `prev_published_model_sha` still names the demoted net, so `_process_shard`
    labels the OLDER net "cur" and the NEWER one "prev". A -139 Elo regression
    records as +139, at exactly the moment the gate acts on it.
    """
    c = _controller(tmp_path)
    c.note_published()                      # normal publish
    assert c.sample_is_valid is True

    c.hold_path = tmp_path / "gate_promoted_model.pt"
    c.note_published()                      # the transition INTO a hold
    assert c.sample_is_valid is False

    c.note_published()                      # sustained hold
    assert c.sample_is_valid is False

    c.hold_path = None
    c.note_published()                      # the release publish
    assert c.sample_is_valid is False, (
        "on release, 'prev' is the anchor, so the delta spans the whole hold "
        "rather than one training iteration"
    )
    c.note_published()                      # back to normal
    assert c.sample_is_valid is True


def test_invalid_sample_is_recorded_but_excluded_from_the_window(
    tmp_path: Path,
) -> None:
    """An invalid sample must still emit a row -- with zero games, so it is
    visibly excluded -- rather than vanish and silently shorten the window."""
    gate = _gate(mode=MODE_SHADOW)
    sp = SelfplayResult(
        gate_cur_w=100, gate_cur_d=0, gate_cur_l=0,
        gate_prev_w=0, gate_prev_d=0, gate_prev_l=100,
        gate_sample_valid=False,
    )
    result = _run_training_and_gating(
        tc=TrialConfig(batch_size=512), trainer=_StubTrainer(),
        buf=[], holdout_buf=[], holdout_frozen=True, config={}, model_cfg=None,
        device="cpu",
        ds=DifficultyState(wdl_regret=0.089, sf_nodes=50_000), sims=64, sp=sp,
        positions_ingested=0, imported_samples_this_iter=0,
        gate=gate, gate_match_idx=0,
        gate_state_path=tmp_path / "gate_state.json", gate_hold=None,
        distributed_server_root=tmp_path, iteration_idx=5,
        iteration_zero_based=5, trial_id="t0", restore=RestoreResult(),
    )
    m = gate_metrics(result.gate_decision)
    assert m["gate_sample_games_cur"] == 0.0
    assert m["gate_sample_games_prev"] == 0.0
    assert gate.samples[-1].usable(min_games_per_side=5) is False


def test_report_dict_carries_the_sample_columns(tmp_path: Path) -> None:
    """MUTATION (reviewer R9): drop the four `gate_sample_*` keys from the
    report dict.

    That silently deletes the entire fix for round 1's worst finding while the
    rest of the suite stays green -- the readout is the product here, and
    nothing was asserting it reached the CSV.
    """
    del tmp_path
    from chess_anti_engine.tune.promotion_gate import gate_metrics as gm

    emitted = set(gm(GateDecision(mode=MODE_SHADOW)))
    required = {"gate_sample_delta_score", "gate_sample_delta_elo",
                "gate_sample_games_cur", "gate_sample_games_prev"}
    assert required <= emitted

    import inspect

    import chess_anti_engine.tune.trainable_report as rep
    src = inspect.getsource(rep._build_report_dict)
    assert "gate_metrics(tr.gate_decision" in src, (
        "the report dict must go through gate_metrics, or the sample columns "
        "never reach progress.csv"
    )


def test_window_keeps_the_NEWEST_samples(tmp_path: Path) -> None:
    """MUTATION (reviewer R8): trim `[:-keep]` -> `[keep:]`.

    Length is unchanged either way, so a length assertion cannot see it -- but
    the window freezes at its first 24 samples and the alarm goes permanently
    deaf while looking healthy.
    """
    del tmp_path
    g = _gate(mode=MODE_SHADOW)
    for i in range(100):
        g.observe(_sample(i, 0.50, 0.50))
    assert [s.iteration for s in g.samples] == list(range(76, 100))

    revived = _gate(mode=MODE_SHADOW)
    revived.load_state_dict({"samples": [
        {"iteration": i, "cur_w": 50, "cur_l": 50, "prev_w": 50, "prev_l": 50}
        for i in range(100)
    ]})
    assert [s.iteration for s in revived.samples] == list(range(76, 100))


def test_reporting_path_degrades_instead_of_killing_the_trial() -> None:
    """`gate_metrics` guards a state no shipped path can reach. Raising from
    inside `_build_report_dict` would take the whole trial down to protect
    against something that cannot happen."""
    bogus = GateDecision(decision=DECISION_PROMOTE, reason="promote_no_regression",
                         mode=MODE_ENFORCE, iters=8, games_cur=0, games_prev=143)
    with pytest.raises(AssertionError):
        gate_metrics(bogus)
    degraded = gate_metrics(bogus, strict=False)
    assert degraded["gate_decision"] == float(DECISION_NOT_RUN)


def test_selfplay_phase_forwards_the_hold_to_the_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTATION (reviewer R5): `hold=None` on the way into the publish.

    `_run_selfplay_phase` is the only frame between the loop's hold state and
    the fleet. Everything downstream of a dropped argument here reports a
    perfectly consistent DEMOTE while the workers keep receiving the demoted
    net -- the "accepted and then silently ignored" shape this repo is built
    out of.
    """
    import chess_anti_engine.tune.trainable_phases as phases

    seen: dict[str, object] = {}

    def _fake_publish(**kw):
        seen.update(kw)
        h = kw.get("hold")
        if h is not None:
            h.note_published()
        return "sha-published"

    monkeypatch.setattr(phases, "_publish_iteration_model", _fake_publish)
    monkeypatch.setattr(phases, "_ensure_distributed_workers", lambda **kw: [])
    monkeypatch.setattr(phases, "_ensure_inference_broker", lambda **kw: None)
    monkeypatch.setattr(
        phases, "_ingest_distributed_selfplay",
        lambda **kw: {**_empty_ingest_summary(), "matching_games": 0},
    )

    class _Buf:
        _shard_paths: ClassVar[list[Path]] = []
        def flush(self) -> None: ...
        def __len__(self) -> int: return 0
        def _snapshot_shards(self) -> list[Path]: return list(self._shard_paths)

    controller = _controller(tmp_path)
    controller.hold_path = tmp_path / "gate_promoted_model.pt"

    sp, _sha, _win = phases._run_selfplay_phase(
        tc=TrialConfig(batch_size=512), config={}, trainer=_StubTrainer(),
        model_cfg=None, buf=_Buf(), holdout_buf=[], holdout_frozen=False,
        rng=np.random.default_rng(0),
        distributed_dirs={"publish_dir": tmp_path, "inbox_dir": tmp_path,
                          "processed_dir": tmp_path},
        distributed_server_root=tmp_path, distributed_worker_procs=[],
        broker_proc_box=[None], prev_published_model_sha="sha-prev",
        ds=DifficultyState(wdl_regret=0.089, sf_nodes=50_000), sims=64,
        iteration_idx=7, iteration_zero_based=7, trial_id="t0",
        trial_dir=tmp_path, selfplay_shards_dir=tmp_path,
        replay_shard_dir=tmp_path, current_window=1000,
        in_salvage_startup_grace=False, hold=controller,
    )
    assert seen["hold"] is controller, "the hold must reach the publish call"
    assert sp.should_retry is True
    assert sp.gate_sample_valid is False, (
        "a publish that served the anchor cannot produce a valid anchored sample"
    )


# --------------------------------------------------------------------------
# 11. A SINGLE-ITERATION STEP: the mean-CI rule cannot see one at all
#     (review round 6, F2)
# --------------------------------------------------------------------------
def _one_shot_window(step_elo: float, k: int) -> list[float]:
    """K score-deltas: k-1 exact zeros and one worth ``step_elo``."""
    return [0.0] * (k - 1) + [step_elo / ELO_PER_SCORE_AT_HALF]


def test_a_one_iteration_step_cannot_move_the_mean_ci_rule() -> None:
    """THE ARITHMETIC, PINNED. Not a mutation test -- a proof that no tuning of
    this rule can catch a step, which is why a second leg exists.

    For a window of K deltas with one ``-M`` and the rest 0:
        mean = -M/K ; var = M^2/K ; se = M/K ; elo_hi = (M/K)(t-1) > 0
    M cancels. If this test ever fails, the mean-CI rule has become
    step-sensitive and the step leg's justification needs re-reading.
    """
    for k in (8, 12, 24, 48):
        t = _t_quantile(0.05, k - 1)
        assert t > 1.0
        for m in (100.0, 300.0, 600.0, 1_000_000.0):
            d = _one_shot_window(-m, k)
            mean = sum(d) / k
            var = sum((x - mean) ** 2 for x in d) / (k - 1)
            se = math.sqrt(var / k)
            # se equals |mean| exactly, which is the whole result.
            assert se == pytest.approx(abs(mean), rel=1e-12)
            elo_hi = ELO_PER_SCORE_AT_HALF * (mean + t * se)
            assert elo_hi > 0.0, (k, m, elo_hi)
            # ...and it scales with M rather than shrinking, so a bigger break
            # reads as a bigger POSITIVE upper bound.
            assert elo_hi == pytest.approx((m / k) * (t - 1.0), rel=1e-9)

    # The measured headline in the docstring and the yaml.
    k, m = 24, 1_000_000.0
    d = _one_shot_window(-m, k)
    mean = sum(d) / k
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (k - 1) / k)
    hi = ELO_PER_SCORE_AT_HALF * (mean + _t_quantile(0.05, k - 1) * se)
    assert hi == pytest.approx(29_754.0, abs=5.0)

    # And the same thing through the REAL gate, with the step leg turned off by
    # setting its line beyond reach: the window rule alone promotes a -300 step.
    g = _gate(demote_step_elo=-100_000.0, min_games_per_side=40)
    for i in range(23):
        g.observe(_sample(i, 0.50, 0.50))
    g.observe(_sample(23, cur_score=0.50 - 300.0 / ELO_PER_SCORE_AT_HALF,
                      prev_score=0.50))
    d_ = g.decide()
    assert d_.decision == DECISION_PROMOTE
    assert d_.reason == "promote_no_regression"
    assert d_.elo_hi > 0.0


def test_a_step_leg_fires_on_a_single_iteration_step() -> None:
    """MUTATION: delete ``PromotionGate._step_regressed``'s comparison (return
    False), or drop ``step`` from ``decide()``'s ``_resolve`` call.

    THE LEG THE WHOLE OF SECTION 11 EXISTS FOR. One iteration, at the realized
    live arm shape, with a step big enough to clear ``demote_step_elo`` plus its
    own binomial noise -> DEMOTE, from a window that the mean-CI rule promotes.
    """
    g = _gate(min_games_per_side=15)
    for i in range(23):
        g.observe(_sample(i, 0.50, 0.50))
    # -300 Elo in one iteration, at n_cur=197 / n_prev=38.
    shift = -300.0 / ELO_PER_SCORE_AT_HALF
    cur_w = round((0.50 + shift) * 197)
    g.observe(AnchoredSample(
        23, cur_w=cur_w, cur_d=0, cur_l=197 - cur_w,
        prev_w=19, prev_d=0, prev_l=19,
    ))
    d = g.decide()
    assert d.decision == DECISION_DEMOTE
    assert d.reason == "demote_step"
    assert d.would_demote is True
    # The window mean alone would NOT have demoted -- that is the point.
    assert d.elo_hi > GateConfig().demote_delta_elo
    # The leg's own input travels with the verdict.
    assert d.sample_elo_hi < GateConfig().demote_step_elo
    assert gate_metrics(d)["gate_sample_elo_hi"] == pytest.approx(d.sample_elo_hi)
    assert gate_metrics(d)["gate_would_demote"] == 1.0
    assert gate_metrics(d)["gate_reason_code"] == 8.0


def test_the_step_leg_fires_before_the_window_is_full() -> None:
    """MUTATION: move ``step = self._step_regressed()`` below the ``min_iters``
    early return.

    A bad merge three iterations after a restart is exactly when a step leg has
    to work, and every window path (short window, thin rows, degenerate
    variance) returns NOT_RUN -- which RELEASES the brake.
    """
    g = _gate(min_games_per_side=15)
    # The current model lost EVERY game while the previous one held 50%:
    # a -347 Elo step at this sample shape, well past demote_step_elo + noise.
    broken = AnchoredSample(
        0, cur_w=0, cur_d=0, cur_l=197, prev_w=19, prev_d=0, prev_l=19,
    )
    g.observe(broken)
    d = g.decide()
    assert len(g.samples) == 1 < GateConfig().min_iters
    assert d.decision == DECISION_DEMOTE
    assert d.reason == "demote_step"

    # ...and a DEGENERATE window (every delta identical) does not shelter a
    # step either: that path also used to return NOT_RUN unconditionally.
    g2 = _gate(min_games_per_side=15)
    for i in range(8):
        g2.observe(dataclasses.replace(broken, iteration=i))
    d2 = g2.decide()
    assert d2.reason != "degenerate_variance"
    assert d2.decision == DECISION_DEMOTE


def test_the_step_leg_cannot_create_a_promote() -> None:
    """MUTATION: make ``decide()`` return the STEP verdict unconditionally, or
    turn the ``or`` in ``regressed=window_regressed or step`` into an ``and``.

    The leg is allowed to ADD a demote and nothing else. Two directions:

    * a window that already demotes must still demote, with the WINDOW's name,
      whatever the step leg says (an ``and`` would silently disarm the
      sustained-break rule on every iteration whose single sample is quiet);
    * no configuration of the step leg may turn a demote into a promote.
    """
    # window demotes, step quiet -> demote_regression, not promote
    g = _gate(demote_delta_elo=-50.0, min_games_per_side=40)
    d = _feed(g, _window(-0.10))
    assert d.decision == DECISION_DEMOTE
    assert d.reason == "demote_regression"

    # window demotes AND step fires -> still named for the window
    g2 = _gate(demote_delta_elo=-50.0, min_games_per_side=40)
    d2 = _feed(g2, _window(-0.30))
    assert d2.decision == DECISION_DEMOTE
    assert d2.reason == "demote_regression"

    # Exhaustive over the step line: a window-demoting series can never become
    # a promote by moving demote_step_elo.
    for line in (-1.0, -50.0, -125.0, -1000.0, -100_000.0):
        gk = _gate(demote_delta_elo=-50.0, demote_step_elo=min(line, -50.0),
                   min_games_per_side=40)
        assert _feed(gk, _window(-0.30)).decision == DECISION_DEMOTE, line


def test_step_leg_false_brake_budget_and_power_reproduce() -> None:
    """The numbers the shipped ``demote_step_elo`` was chosen against, in the
    docstring, the GateConfig comment and the yaml.

    MUTATION: change ``AnchoredSample.score_se`` to use a per-arm variance with
    no pooled FLOOR. An arm that drew every game then has se 0 and the leg
    fires on certainty it does not have.
    """
    z = _Z_ONE_SIDED[0.05]

    def se_elo(nc: int, npv: int) -> float:
        # All-draw arms have observed variance 0, so this reads the POOLED
        # floor exactly -- the narrowest se the leg can ever act on, hence the
        # WORST case for the false-brake budget. Real draw-light arms have
        # observed variance up to 0.25 > the floor, so their se is wider and
        # they fire less readily.
        return AnchoredSample(
            0, cur_d=nc, prev_d=npv,
        ).score_se * ELO_PER_SCORE_AT_HALF

    realized, worst = se_elo(197, 38), se_elo(197, 15)
    assert realized == pytest.approx(42.4, abs=1.5)
    assert worst == pytest.approx(64.2, abs=2.0)

    def norm_sf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    line = GateConfig().demote_step_elo
    for se, budget, p50, p90 in ((realized, 0.05, 195.0, 249.0),
                                 (worst, 1.5, 231.0, 313.0)):
        fires_at = line - z * se           # sample delta must be below this
        p_null = norm_sf(fires_at / se)
        assert p_null * 8000 < budget, (se, p_null * 8000)
        assert -fires_at == pytest.approx(p50, abs=6.0)
        assert -(fires_at - 1.2816 * se) == pytest.approx(p90, abs=8.0)

    # THE STATED BLIND SPOT: a -100 step does not fire at the realized shape.
    g = _gate(min_games_per_side=15)
    shift = -100.0 / ELO_PER_SCORE_AT_HALF
    cur_w = round((0.50 + shift) * 197)
    g.observe(AnchoredSample(0, cur_w=cur_w, cur_d=0, cur_l=197 - cur_w,
                             prev_w=19, prev_d=0, prev_l=19))
    assert g.decide().decision != DECISION_DEMOTE, (
        "the docs must not claim the step leg catches -100"
    )

    # The pooled floor is load-bearing: an all-draws arm must not read se 0.
    all_draws = AnchoredSample(0, cur_d=197, prev_d=38)
    assert all_draws.score_se > 0.0
    assert all_draws.score_se == pytest.approx(
        math.sqrt(0.3447 ** 2 / 197 + 0.3447 ** 2 / 38), rel=1e-9,
    )


def test_documented_time_to_trip_is_the_steady_state_number() -> None:
    """MUTATION: restore the docstring's "trips inside the 8-iteration floor".

    Eight is the COLD-window latency. Production always carries 24 healthy
    rows, whose means drag the statistic toward zero until they age out, and
    the two regimes disagree by 50% at -100 Elo/iteration. Both are computed
    here so the docs cannot quote one as the other.
    """
    sd, line, k, min_iters, alpha = 45.56, -25.0, 24, 8, 0.05

    def trip(break_elo: float, warm: bool, rng: random.Random) -> int | None:
        w = [rng.gauss(0.0, sd) for _ in range(k)] if warm else []
        for i in range(1, 201):
            w.append(rng.gauss(break_elo, sd))
            w = w[-k:]
            if len(w) < min_iters:
                continue
            n = len(w)
            mean = sum(w) / n
            se = math.sqrt(sum((x - mean) ** 2 for x in w) / (n - 1) / n)
            if se > 0.0 and mean + _t_quantile(alpha, n - 1) * se < line:
                return i
        return None

    rng = random.Random(7)
    med: dict[tuple[float, bool], float] = {}
    for warm in (True, False):
        for b in (-50.0, -100.0, -200.0, -300.0):
            hits = [t for t in (trip(b, warm, rng) for _ in range(400))
                    if t is not None]
            assert len(hits) == 400
            med[(b, warm)] = st.median(hits)

    # Steady state: the numbers the corrected docstring and yaml quote.
    assert med[(-50.0, True)] == pytest.approx(20, abs=2)
    assert med[(-100.0, True)] == pytest.approx(12, abs=2)
    assert med[(-200.0, True)] == pytest.approx(8, abs=2)
    assert med[(-300.0, True)] == pytest.approx(6, abs=2)
    # Cold: pinned at the min_iters floor, which is where "8" came from.
    for b in (-100.0, -200.0, -300.0):
        assert med[(b, False)] == pytest.approx(min_iters, abs=1)
    # ...and the two regimes genuinely differ, or the correction is vacuous.
    assert med[(-100.0, True)] >= med[(-100.0, False)] + 3

    doc = promotion_gate.__doc__ or ""
    assert "EIGHT IS THE COLD-WINDOW" in doc.upper()
    assert "trips within the ``min_iters`` floor of 8 iterations" not in doc


# --------------------------------------------------------------------------
# 12. EVERY SHIPPED CONSTANT PINNED BY VALUE, not merely by consistency
#     (review round 6, F4)
# --------------------------------------------------------------------------
# yaml key -> (GateConfig field, shipped value). The technique that already
# worked for ``demote_delta_elo``: assert the VALUE, so changing it in both the
# dataclass default and ``gate_config_from_dict`` still fails. The previous
# suite only had a CONSISTENCY pin, and every one of these survived being
# changed in both copies at once (window_iters 24->48, min_iters 8->4,
# min_games_per_side 15->5, alpha 0.05->0.10, max_hold_iters 12->100).
_SHIPPED_GATE_CONSTANTS = (
    ("gate_mode", "mode", MODE_OFF),
    ("gate_window_iters", "window_iters", 24),
    ("gate_min_iters", "min_iters", 8),
    ("gate_min_games_per_side", "min_games_per_side", 15),
    ("gate_demote_delta_elo", "demote_delta_elo", -25.0),
    ("gate_demote_step_elo", "demote_step_elo", -125.0),
    ("gate_alpha", "alpha", 0.05),
    ("gate_max_hold_iters", "max_hold_iters", 12),
)


@pytest.mark.parametrize(("yaml_key", "field", "shipped"), _SHIPPED_GATE_CONSTANTS)
def test_every_shipped_gate_constant_is_pinned_by_value(
    yaml_key: str, field: str, shipped: object,
) -> None:
    """MUTATION: change the constant in BOTH the dataclass default and
    ``gate_config_from_dict``'s fallback. Consistency pins pass; this does not.

    Every headline number in this module's docs is quoted AT these values --
    the 9.3 Elo window se, the 92.5% power, the 0.02/8000 false-brake budget,
    the 42.4 Elo sample se, the "96% of iterations admitted" floor. A silent
    change to any of them makes the whole document a description of a different
    gate.
    """
    assert getattr(GateConfig(), field) == shipped, (
        f"{yaml_key}'s dataclass default drifted from the documented value"
    )
    # ...and the yaml fallback is the same value, so an operator who does not
    # set the key gets what the docs describe.
    assert getattr(gate_config_from_dict({}), field) == shipped, (
        f"{yaml_key}'s gate_config_from_dict fallback drifted"
    )


def test_min_games_per_side_is_pinned_from_BOTH_sides() -> None:
    """MUTATION: 15 -> 5, which the old suite did NOT kill (only 15 -> 60 did).

    Five games a side is ~150 Elo of binomial se on ONE anchored delta -- six
    times the demote line -- so admitting a 5-vs-5 iteration puts pure noise
    into the window mean with the same weight as a 197-game row.
    """
    floor = GateConfig().min_games_per_side
    assert floor == 15
    # Upper pin (the pre-review 40 disqualified ~70% of live iterations).
    assert floor <= 20
    # Lower pin, by consequence rather than by taste: one iteration at the
    # floor must not carry more than ~4x the demote line's worth of noise
    # (at 15/15 the pooled-floor se is 87.5 Elo).
    at_floor = AnchoredSample(
        0, cur_d=floor, prev_d=floor,
    ).score_se * ELO_PER_SCORE_AT_HALF
    assert at_floor < 4.0 * abs(GateConfig().demote_delta_elo), (
        f"a floor of {floor} admits an iteration whose own se is "
        f"{at_floor:.0f} Elo"
    )
    assert floor >= 10, "below ~10 games a side an iteration is a coin flip"
    # And the arithmetic that makes 5 unacceptable, so the bound above is not
    # just a number: 5 games a side is ~150 Elo of se on ONE row even at the
    # pooled floor, i.e. 6x the line.
    five = AnchoredSample(0, cur_d=5, prev_d=5).score_se * ELO_PER_SCORE_AT_HALF
    assert five > 140.0
    assert five > 5.0 * abs(GateConfig().demote_delta_elo)


def test_the_no_cadence_fallback_band_is_pinned_by_value() -> None:
    """MUTATION: widen the cadence-unknown count band from 0.25-4.0x to
    0.001-1000x, which the old suite did not kill.

    Without ``time_this_iter_s`` the count carries no information except gross
    breakage, which is why the band is loose -- but "loose" is 4x, not 1000x.
    At 1000x the leg cannot fail, and it is the ONLY count leg on a csv with no
    cadence column.
    """
    ref_total = OFFLINE.mean_games_cur + OFFLINE.mean_games_prev

    def verdict_at(scale: float) -> ShadowReadout:
        rows = [
            (int(OFFLINE.mean_games_cur * scale),
             int(OFFLINE.mean_games_prev * scale),
             -4.0 + 40.0 * math.sin(i))
            for i in range(_READOUT_MIN_ROWS)
        ]
        return shadow_readout_verdict(rows, min_games_per_side=5)

    inside = verdict_at(1.0)
    assert not any("anchored games/iteration" in leg
                   for leg in inside.failed_legs)
    for scale in (0.20, 5.0):
        out = verdict_at(scale)
        assert any("anchored games/iteration" in leg for leg in out.failed_legs), (
            f"a {scale}x count swing with no cadence column must fire the "
            f"fallback band; total {ref_total * scale:.0f} vs {ref_total:.0f}"
        )
        assert out.verdict == READOUT_KILL



# --------------------------------------------------------------------------
# 13. THE SIX PRODUCTION CALL SITES (review round 6, F3)
# --------------------------------------------------------------------------
def _train_trial_body() -> ast.FunctionDef:
    """The ``ast`` of ``trainable.train_trial``.

    WHY AST AND NOT EXECUTION, stated plainly: ``train_trial`` needs Ray, a GPU,
    a published manifest and a live replay window, so its body cannot be driven
    in a unit test. But the defect class here is WIRING -- six calls that a
    reviewer deleted, all six of which escaped the entire suite -- and wiring is
    exactly what a structural assertion can pin. Each site below ALSO has a
    behavioural test of what the call does; this test is what fails when the
    call stops being made.
    """
    import chess_anti_engine.tune.trainable as trainable

    tree = ast.parse(inspect.getsource(trainable))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "train_trial":
            return node
    raise AssertionError("train_trial not found in chess_anti_engine.tune.trainable")


def _calls_on(node: ast.AST, obj: str, attr: str) -> list[ast.Call]:
    return [
        n for n in ast.walk(node)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == obj
        )
    ]


def test_the_loop_body_calls_on_decision_with_the_verdict_and_keeps_it() -> None:
    """MUTATION: ``tr.gate_decision = gate_hold.on_decision(tr.gate_decision)``
    -> ``pass``.

    THIS IS THE ACTUATOR. Without it ``hold_path`` never changes, so the gate
    reports DEMOTE forever and the fleet keeps receiving the demoted net -- a
    verdict accepted and then silently ignored, in the module whose docstrings
    are about that failure mode. It survived the whole 74-test suite.
    """
    body = _train_trial_body()
    calls = _calls_on(body, "gate_hold", "on_decision")
    assert len(calls) == 1, f"expected exactly one on_decision call, got {len(calls)}"
    call = calls[0]
    src = ast.unparse(call)
    assert "tr.gate_decision" in src, src
    # The RESULT must be kept: on_decision returns the decision with the
    # anchor's refresh health attached, and dropping it puts
    # gate_anchor_refresh_failures back to living only inside the controller.
    assigns = [
        n for n in ast.walk(body)
        if isinstance(n, ast.Assign) and n.value is call
    ]
    assert assigns, "on_decision's return value must be assigned, not discarded"
    assert ast.unparse(assigns[0].targets[0]) == "tr.gate_decision"


def test_the_loop_body_ages_a_hold_on_an_aborted_iteration() -> None:
    """MUTATION: delete ``gate_hold.on_aborted_iteration()`` from the
    ``sp.should_retry`` branch.

    ``should_retry`` aborts before the gate observes anything, so the release
    counter freezes while the fleet stays held: a run stuck in retry during a
    hold HOLDS FOREVER. The call must be inside that branch and before the
    ``continue``, or it is unreachable.
    """
    body = _train_trial_body()
    retry_ifs = [
        n for n in ast.walk(body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "sp.should_retry"
    ]
    assert len(retry_ifs) == 1, "expected exactly one `if sp.should_retry:`"
    inner = retry_ifs[0].body
    assert _calls_on(retry_ifs[0], "gate_hold", "on_aborted_iteration"), (
        "the retry branch must age an active hold"
    )
    # ...and before the continue, or it never runs.
    idx_call = next(
        i for i, st_ in enumerate(inner)
        if _calls_on(st_, "gate_hold", "on_aborted_iteration")
    )
    idx_cont = next(
        i for i, st_ in enumerate(inner) if isinstance(st_, ast.Continue)
    )
    assert idx_call < idx_cont


def test_the_startup_publish_honours_a_restored_hold() -> None:
    """MUTATION: ``override_model_path=gate_hold.hold_path`` -> ``None`` at the
    startup publish.

    A resume mid-hold would lift the brake for one publish AND re-anchor on the
    held-back weights, so an enforce-mode hold would be bounded by restart
    cadence rather than by ``gate_max_hold_iters``.
    """
    body = _train_trial_body()
    publishes = [
        n for n in ast.walk(body)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_publish_distributed_trial_state"
    ]
    assert len(publishes) == 1, (
        "train_trial should have exactly one direct publish (the startup one); "
        "the per-iteration publish goes through _publish_iteration_model"
    )
    kw = {k.arg: ast.unparse(k.value) for k in publishes[0].keywords}
    assert kw.get("override_model_path") == "gate_hold.hold_path", kw


def test_the_publish_helper_rolls_the_sign_validity_history(tmp_path: Path) -> None:
    """MUTATION: ``hold.note_published()`` -> ``pass`` in
    ``_publish_iteration_model``.

    Behavioural, not structural. Without the roll, ``_prev_publish_held`` never
    advances, so ``sample_is_valid`` stays True across a hold transition and the
    SIGN-INVERTED sample -- a -139 Elo regression recorded as +139, at exactly
    the moment the gate acts on it -- enters the window.
    """
    trainer = _MarkerTrainer()
    promoted = tmp_path / "gate_promoted_model.pt"
    ctrl = GateHoldController(
        gate=_gate(mode=MODE_ENFORCE), promoted_model_path=promoted,
    )

    _publish_iter(tmp_path, trainer, hold=None, promoted=promoted,
                  controller=ctrl)      # normal publish
    assert ctrl.sample_is_valid is True
    ctrl.hold_path = promoted
    _publish_iter(tmp_path, trainer, hold=promoted, promoted=promoted,
                  controller=ctrl)      # the iteration that STARTS the hold
    assert ctrl.sample_is_valid is False, (
        "the hold-transition sample has an inverted sign and must be excluded"
    )
    ctrl.hold_path = None
    _publish_iter(tmp_path, trainer, hold=None, promoted=promoted,
                  controller=ctrl)      # the RELEASE iteration
    assert ctrl.sample_is_valid is False, (
        "the release sample spans the whole hold, not one training step"
    )
    _publish_iter(tmp_path, trainer, hold=None, promoted=promoted,
                  controller=ctrl)
    assert ctrl.sample_is_valid is True


def test_gate_mode_is_restart_required_and_the_reload_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """MUTATION: drop ``"gate_mode"`` from ``_LIVE_RELOAD_SKIPPED_KEYS``.

    The list's own comment claims membership "turns that silence into a
    WARNING". Nothing checked it, and the claim was only half true: the guard
    read ``k in config and config[k] != v``, so a live ADD -- the ONLY way
    ``gate_mode`` ever changes, because ``configs/pbt2_small.yaml`` ships no
    ``gate_*`` key at all -- took the silent-overlay path. The value landed in
    ``config``, ``PromotionGate`` had been built from the launch config
    iterations earlier, and the operator got a knob that reads back correctly
    and does nothing.
    """
    import logging

    from chess_anti_engine.tune.trainable_config_ops import (
        _LIVE_RELOAD_SKIPPED_KEYS,
        _reload_yaml_into_config,
        restart_required_config_keys,
    )

    for key in ("gate_mode", "gate_window_iters", "gate_min_iters",
                "gate_min_games_per_side", "gate_demote_delta_elo",
                "gate_demote_step_elo", "gate_alpha", "gate_max_hold_iters",
                "gate_games"):
        assert key in _LIVE_RELOAD_SKIPPED_KEYS, key
        assert key in restart_required_config_keys(), key

    yaml_path = tmp_path / "live.yaml"
    yaml_path.write_text("gate_mode: enforce\nbatch_size: 512\n", encoding="utf-8")

    # (a) the live ADD case -- the one the old guard could not see.
    cfg_add: dict = {"batch_size": 512}
    with caplog.at_level(logging.WARNING):
        _reload_yaml_into_config(cfg_add, str(yaml_path), live_reload=True)
    assert cfg_add.get("gate_mode") != "enforce", (
        "a live gate_mode edit must NOT reach the config: the gate is built "
        "once, at launch, so an applied value here is a knob that cannot act"
    )
    assert "gate_mode" in caplog.text, caplog.text
    assert "requires restart" in caplog.text, caplog.text

    # (b) the CHANGE case.
    caplog.clear()
    cfg_chg: dict = {"batch_size": 512, "gate_mode": "off"}
    with caplog.at_level(logging.WARNING):
        _reload_yaml_into_config(cfg_chg, str(yaml_path), live_reload=True)
    assert cfg_chg["gate_mode"] == "off"
    assert "requires restart" in caplog.text
    assert "gate_mode" in caplog.text

    # (c) and a NON-live reload (startup/resume) still applies it, or a restart
    # could never pick the new value up.
    cfg_start: dict = {"batch_size": 512, "gate_mode": "off"}
    _reload_yaml_into_config(cfg_start, str(yaml_path), live_reload=False)
    assert cfg_start["gate_mode"] == "enforce"


# --------------------------------------------------------------------------
# 14. THE ANCHOR REFRESH MUST NOT FAIL SILENTLY (review round 6, F5)
# --------------------------------------------------------------------------
def test_a_failing_anchor_refresh_is_visible_and_stops_the_brake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """MUTATION: restore ``with contextlib.suppress(OSError):`` around the
    anchor refresh in ``_publish_iteration_model``.

    The two halves of the anchor contract disagreed. The hold path RAISES
    ``FileNotFoundError`` rather than publish an unheld model under a hold
    decision, while the refresh -- a ~252 MB copy into ``durable_dir``, every
    iteration -- swallowed every OSError. A full disk therefore froze the
    fallback at an arbitrarily old export with no log line and no metric, and a
    hold N iterations later published that export to the whole fleet.

    It must still not RAISE (an optional alarm may not kill a training run), so
    the required end state is: loud, counted, reported, and eventually
    consequential.
    """
    import logging

    import chess_anti_engine.tune.trainable_phases as phases

    gate = _gate(mode=MODE_ENFORCE)
    anchor = tmp_path / "gate_promoted_model.pt"
    anchor.write_bytes(b"good-old-export")
    ctrl = GateHoldController(gate=gate, promoted_model_path=anchor)
    # A refresh is only DUE after a genuine promote once an anchor exists
    # (G3-8 / review R4), which is the state this scenario is about: the gate
    # is promoting and the copy is what fails.
    ctrl.on_decision(GateDecision(
        decision=DECISION_PROMOTE, reason="promote_no_regression",
        mode=MODE_ENFORCE, games_cur=197, games_prev=38))
    trainer = _MarkerTrainer()

    real_copyfile = phases.shutil.copyfile

    def _boom(src, dst, *a, **k):
        # Only the ANCHOR copy fails; the publish's own export must succeed,
        # because the scenario is "durable_dir is full", not "publishing broke".
        if "gate_promoted_model" in str(dst):
            raise OSError(28, "No space left on device")
        return real_copyfile(src, dst, *a, **k)

    monkeypatch.setattr(phases.shutil, "copyfile", _boom)

    with caplog.at_level(logging.ERROR):
        for _ in range(3):
            _publish_iter(tmp_path, trainer, hold=None, promoted=anchor,
                          controller=ctrl)
    assert ctrl.anchor_refresh_failures == 3
    assert "anchor refresh FAILED" in caplog.text
    assert "No space left on device" in caplog.text

    # ...and the number reaches progress.csv rather than living in the object.
    demote = GateDecision(decision=DECISION_DEMOTE, reason="demote_regression",
                          mode=MODE_ENFORCE, games_cur=197, games_prev=38)
    reported = ctrl.on_decision(demote)
    assert gate_metrics(reported)["gate_anchor_refresh_failures"] == 3.0
    # Three failures is inside gate_max_hold_iters, so the brake still works.
    assert ctrl.hold_path == anchor

    # Past the cap the anchor is older than the longest hold the gate is
    # allowed to impose, so serving it would roll the fleet back further than
    # the mechanism is designed to. It declines instead -- fail-open, which is
    # the same refusal the hold path's raise makes about a MISSING anchor.
    ctrl.anchor_refresh_failures = gate.cfg.max_hold_iters + 1
    assert ctrl.anchor_is_trustworthy is False
    with caplog.at_level(logging.ERROR):
        stale = ctrl.on_decision(demote)
    assert ctrl.hold_path is None, (
        "a hold must not publish an export older than gate_max_hold_iters"
    )
    assert "publishing normally" in caplog.text
    assert gate_metrics(stale)["gate_anchor_refresh_failures"] == float(
        gate.cfg.max_hold_iters + 1,
    )

    # A recovered refresh clears the counter and re-arms the brake.
    monkeypatch.undo()
    ctrl.note_anchor_refreshed()
    assert ctrl.anchor_refresh_failures == 0
    assert ctrl.anchor_is_trustworthy is True


# --------------------------------------------------------------------------
# 15. THE PID LAG DOES NOT CANCEL (review round 6, F1)
# --------------------------------------------------------------------------
def test_the_module_no_longer_claims_the_pid_cancels() -> None:
    """MUTATION: restore "the PID's tracking cancels in the difference".

    The claim is false by construction -- model and difficulty ship in one
    manifest and are applied together, so old-model-at-new-difficulty is not an
    observable state and there is nothing for the subtraction to cancel. It was
    in the module docstring, the yaml and the PR body simultaneously, which is
    why it is pinned as text in all the copies a reader might reach for.
    """
    doc = promotion_gate.__doc__ or ""
    yaml_src = (Path(__file__).resolve().parents[1]
                / "configs" / "pbt2_small.yaml").read_text("utf-8")

    for banned in ("tracking cancels", "PID's tracking cancels",
                   "the PID cancels"):
        assert banned not in doc, banned
        assert banned not in yaml_src, banned

    assert "THE PID LAG DOES NOT CANCEL" in doc
    assert "IT DOES NOT" in yaml_src
    # The -4.3 bound must be labelled as not applying during a break, in both.
    assert "Do not quote -4.3 Elo" in doc
    assert "does NOT\n  # apply during a break" in yaml_src


def test_the_retired_prev_share_is_labelled_retired_and_not_quotable() -> None:
    """MUTATION: re-promote ``16.3%`` to a live figure, or drop the instrument
    stamp, and this fails.

    The ``16.3%`` prev share sat in this docstring for weeks as a bare
    "MEASURED" number.  It was measured by binning shard ``.zattrs``
    ``generated_at_unix`` against iteration timestamps -- a PROXY for the
    ``model_sha256`` partition the gate actually uses -- and on a lineage the
    2026-08-04 fresh boot replaced.  The gate's own counters read 53.1% on
    2026-08-11.  Anyone sizing a window off 16.3% would size it off a number
    that is wrong by 3x, from the module that owns the correct one.

    So the test pins the SHAPE of the correction, not just the digits: the old
    figure must still be present (retired numbers are kept, not deleted --
    deleting one invites its rediscovery), it must be labelled superseded, and
    the replacement must name the instrument it came from.
    """
    doc = promotion_gate.__doc__ or ""

    # The retired figure is kept and explicitly disowned.
    assert "16.3%" in doc
    assert "Do not quote that split" in doc
    assert "Superseded" in doc or "superseded" in doc
    assert "PROXY instrument" in doc

    # The replacement names the instrument -- the gate's own counters, by name.
    for counter in ("gate_iters", "gate_games_cur", "gate_games_prev"):
        assert counter in doc, counter
    assert "53.1%" in doc

    # And the reason the two disagree is stated, not left for the reader.
    assert "model_sha256" in doc

    # The downstream consequence is spelled out rather than silently ignored:
    # the shipped power table was measured under the OLD shape.
    assert "OPTIMISTIC" in doc.upper()


def test_shard_meta_records_the_difficulty_each_arm_played_at() -> None:
    """MUTATION: drop ``opponent_wdl_regret_limit`` from ``ShardMeta``, or drop
    the per-arm accumulation from ``_process_shard``.

    Without this NOTHING -- not the loop, not an offline reconstruction -- can
    check which difficulty each anchored arm played at, so the PID-lag term is
    unmeasurable and the false cancellation claim is unfalsifiable. This is the
    prerequisite that makes F1 answerable with production data.
    """
    from chess_anti_engine.replay.shard import ShardMeta

    # The field exists, and absent means UNKNOWN rather than 0.0.
    assert ShardMeta().opponent_wdl_regret_limit is None
    assert ShardMeta().sf_nodes is None
    assert ShardMeta(opponent_wdl_regret_limit=0.0875).opponent_wdl_regret_limit == 0.0875
    legacy = {k: v for k, v in dataclasses.asdict(ShardMeta(positions=2)).items()
              if k not in ("opponent_wdl_regret_limit", "sf_nodes")}
    assert ShardMeta(**legacy).opponent_wdl_regret_limit is None

    # ...and the ingest split carries a games-weighted mean per arm.
    summary = _empty_ingest_summary()
    summary["gate_cur_regret_weighted"] = 0.08 * 100 + 0.09 * 100
    summary["gate_cur_regret_games"] = 200
    summary["gate_prev_regret_weighted"] = 0.10 * 40
    summary["gate_prev_regret_games"] = 40
    assert phases_arm_mean(summary, "cur") == pytest.approx(0.085)
    assert phases_arm_mean(summary, "prev") == pytest.approx(0.10)
    # No shard said -> NaN, never 0.0 (0.0 regret is UNHANDICAPPED Stockfish,
    # the opposite end of the range from "no data").
    assert math.isnan(phases_arm_mean(_empty_ingest_summary(), "cur"))


def phases_arm_mean(summary: dict, side: str) -> float:
    from chess_anti_engine.tune.trainable_phases import _arm_mean_regret

    return _arm_mean_regret(summary, side)


def test_worker_refuses_to_mix_two_difficulties_into_one_shard() -> None:
    """MUTATION: drop ``_difficulty_matches`` from the buffer identity check.

    ``opponent_wdl_regret_limit`` and ``sf_nodes`` are LIVE reco keys
    (``worker._RECO_LIVE_KEYS``), so they can move without the model sha moving
    -- and then one shard would carry games played at two difficulties under a
    single recorded value. A shard whose stated difficulty is neither of the two
    it played is worse than no field at all.
    """
    from chess_anti_engine.worker_buffer import (
        _buffer_add_completed_game,
        _BufferedUpload,
    )

    class _GB:
        positions = 2
        games = 1
        w = 1
        d = 0
        l = 0
        samples: ClassVar[list[object]] = [object(), object()]
        outcome_stats: ClassVar[dict[str, int]] = {}

    buf = _BufferedUpload()

    def add(regret: float | None = None, nodes: int | None = None,
            *, tracked: bool = True) -> None:
        _buffer_add_completed_game(
            buf=buf, game_batch=_GB(), now_s=0.0,
            model_sha="aa", model_step=1,
            opponent_wdl_regret_limit=regret if tracked else None,
            sf_nodes=nodes if tracked else None,
        )

    add(0.08, 500_000)
    assert buf.opponent_wdl_regret_limit == 0.08
    assert buf.sf_nodes == 500_000

    # same difficulty -> accumulates
    add(0.08, 500_000)
    assert buf.games == 2

    # regret moved -> flush, not silent mixing
    with pytest.raises(ValueError, match="model metadata mismatch"):
        add(0.09, 500_000)
    # nodes moved -> same
    with pytest.raises(ValueError, match="model metadata mismatch"):
        add(0.08, 600_000)
    # None vs a value is a DIFFERENT opponent, not the same one written twice
    with pytest.raises(ValueError, match="model metadata mismatch"):
        add(None, 500_000)
    # ...and a caller that tracks neither (local/bench paths) never flushes.
    add(tracked=False)
    assert buf.games == 3


def test_worker_reads_difficulty_from_the_reco_it_actually_applied() -> None:
    """MUTATION: make ``WorkerSession._active_difficulty`` read the manifest
    instead of ``_active_reco``, or default absent values to 0.

    ``_active_reco`` is the snapshot the worker APPLIED (set at session start
    and on every live apply); the manifest is what the server most recently
    published. The difference between the two is the refresh lag -- exactly the
    thing the field exists to record -- so reading the manifest would erase the
    quantity being measured. And 0.0 regret means UNHANDICAPPED Stockfish, so
    "unknown" must be None, never 0.
    """
    from typing import cast

    from chess_anti_engine.worker import WorkerSession

    def difficulty_of(stub: object) -> tuple[float | None, int | None]:
        # The method only reads ``_active_reco`` via getattr, so a stub self is
        # the honest harness -- constructing a real WorkerSession needs a
        # server, argparse namespace and a GPU broker.
        return WorkerSession._active_difficulty(cast("WorkerSession", stub))

    class _Stub:
        _active_reco: ClassVar[dict[str, object]] = {
            "opponent_wdl_regret_limit": 0.0875, "sf_nodes": 698_000}

    assert difficulty_of(_Stub()) == (0.0875, 698_000)

    class _NoReco:
        pass

    assert difficulty_of(_NoReco()) == (None, None)

    class _PartialReco:
        _active_reco: ClassVar[dict[str, object]] = {"sf_nodes": 5_000}

    assert difficulty_of(_PartialReco()) == (None, 5_000)

    class _Junk:
        # A corrupt value still reports UNKNOWN difficulty -- but it is no
        # longer SILENT about it, so the stub needs the counter/logger the
        # warning path touches (a real WorkerSession sets both in __init__).
        # See tests/test_swallowed_error_hardening.py for the corrupt-vs-absent
        # distinction itself; what is pinned HERE is only that corrupt still
        # reports unknown rather than a fabricated number.
        _active_reco: ClassVar[dict[str, object]] = {
            "opponent_wdl_regret_limit": "not-a-number", "sf_nodes": 5_000}
        _reco_corrupt_count = 0
        _reco_corrupt_last = None
        log = logging.getLogger("test_promotion_gate_junk")

    assert difficulty_of(_Junk()) == (None, None)


def test_the_predicted_confound_is_emitted_beside_the_delta() -> None:
    """MUTATION: drop ``sample_confound_elo`` from ``_base_decision``, or make
    ``AnchoredSample.confound_elo`` return 0.0 when the inputs are NaN.

    The confound column is the falsifier: it is the MEASURED difficulty gap
    between the two arms times the PID's own dWR/dregret, in the same Elo units
    as the delta beside it. A 0.0 where the answer is unknown would read as
    "the controller contributed nothing", which is the claim under dispute.
    """
    s = AnchoredSample(
        0, cur_w=100, cur_d=0, cur_l=97, prev_w=19, prev_d=0, prev_l=19,
        cur_wdl_regret=0.0900, prev_wdl_regret=0.0850, regret_fit_slope=10.886,
    )
    expected = 0.0050 * 10.886 * ELO_PER_SCORE_AT_HALF
    assert s.confound_elo == pytest.approx(expected)
    assert expected == pytest.approx(37.8, abs=0.5)   # ~1.5x the demote line

    g = _gate(mode=MODE_SHADOW, min_games_per_side=15)
    g.observe(s)
    m = gate_metrics(g.decide())
    assert m["gate_sample_confound_elo"] == pytest.approx(expected)

    # Unknown stays unknown.
    unknown = dataclasses.replace(s, regret_fit_slope=float("nan"))
    assert math.isnan(unknown.confound_elo)
    g2 = _gate(mode=MODE_SHADOW, min_games_per_side=15)
    g2.observe(unknown)
    assert math.isnan(gate_metrics(g2.decide())["gate_sample_confound_elo"])


def test_the_readout_regresses_the_delta_on_the_confound() -> None:
    """MUTATION: make the confound leg a bare report with no slope, or drop the
    ``slope_se`` term so it fires on noise.

    THE READOUT THAT SETTLES F1. If the anchored delta is a controller output
    the regression slope is ~1; if the controller is irrelevant it is ~0. The
    leg is one-sided and HOLD-only, so it can never manufacture a promote.
    """
    rng = random.Random(11)
    n = 400   # far past the 40-row window, deliberately: see the last assert

    def rows(passthrough: float, noise_sd: float) -> list[tuple]:
        out = []
        for _ in range(n):
            c = rng.gauss(0.0, 12.0)                      # predicted confound
            d = passthrough * c + rng.gauss(0.0, noise_sd)
            out.append((197, 38, d, OFFLINE.mean_iter_seconds, 235.0, c))
        return out

    # (a) the confound passes straight through -> hold, by name
    hit = shadow_readout_verdict(rows(1.0, 45.56), min_games_per_side=15,
                                 last_n=n)
    assert hit.confound_slope == pytest.approx(1.0, abs=0.25)
    assert hit.verdict == READOUT_HOLD
    assert any("confound_slope" in leg for leg in hit.hold_legs), hit.hold_legs
    assert not hit.failed_legs, "the confound leg must HOLD, never kill"

    # (b) the controller is irrelevant -> the leg is silent
    miss = shadow_readout_verdict(rows(0.0, 45.56), min_games_per_side=15,
                                  last_n=n)
    assert miss.confound_slope == pytest.approx(0.0, abs=0.25)
    assert not any("confound_slope" in leg for leg in miss.hold_legs)

    # (c) AND THE HONEST LIMIT, which is why se is reported. At the
    # pre-registered 40-row window the slope cannot separate 0 from 1, so the
    # readout must say how many rows it needs instead of ruling.
    short = shadow_readout_verdict(rows(1.0, 45.56)[:40], min_games_per_side=15,
                                   last_n=40)
    assert short.confound_slope_se > 0.4
    assert short.confound_rows_needed > 100
    assert not any("confound_slope" in leg for leg in short.hold_legs), (
        "at 40 rows this leg must not rule; se is ~0.6 against a hypothesis "
        "of 1.0"
    )

    # A csv with no confound column at all reports n=0 and fires nothing.
    plain = shadow_readout_verdict(
        [(197, 38, -4.0 + 40.0 * math.sin(i), OFFLINE.mean_iter_seconds, 235.0)
         for i in range(_READOUT_MIN_ROWS)],
        min_games_per_side=15,
    )
    assert plain.n_confound == 0
    assert not any("confound_slope" in leg for leg in plain.hold_legs)


def test_the_confound_column_survives_the_csv_round_trip(tmp_path: Path) -> None:
    """MUTATION: drop ``gate_sample_confound_elo`` from
    ``shadow_readout_rows_from_csv``.

    A column emitted into progress.csv that the deciding command does not read
    is decoration, which is the defect class this whole module is about.
    """
    from chess_anti_engine.tune.promotion_gate import shadow_readout_rows_from_csv

    path = tmp_path / "progress.csv"
    cols = ["gate_sample_games_cur", "gate_sample_games_prev",
            "gate_sample_delta_elo", "gate_sample_confound_elo",
            "time_this_iter_s", "pid_curriculum_w", "pid_curriculum_d",
            "pid_curriculum_l"]
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerow({"gate_sample_games_cur": 197, "gate_sample_games_prev": 38,
                     "gate_sample_delta_elo": -7.5,
                     "gate_sample_confound_elo": 4.25,
                     "time_this_iter_s": 721.0, "pid_curriculum_w": 100,
                     "pid_curriculum_d": 100, "pid_curriculum_l": 35})
    with path.open(newline="") as fh:
        got = shadow_readout_rows_from_csv(csv.DictReader(fh))
    assert got == [(197, 38, -7.5, 721.0, 235.0, 4.25)]

    # A row from before the column existed reads NaN, not 0.0.
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols[:3])
        wr.writeheader()
        wr.writerow({"gate_sample_games_cur": 197, "gate_sample_games_prev": 38,
                     "gate_sample_delta_elo": -7.5})
    with path.open(newline="") as fh:
        old = shadow_readout_rows_from_csv(csv.DictReader(fh))
    assert math.isnan(old[0][5])


def test_the_pid_exposes_its_regret_fit_slope_to_the_gate() -> None:
    """MUTATION: delete ``self.last_regret_fit_slope = ...`` from
    ``DifficultyPID.observe``, or drop it from ``DifficultyState.from_pid``.

    The gate runs EARLIER in the iteration than the PID does, so it cannot read
    a ``PIDUpdate``. Without the slope the confound column is permanently NaN --
    a falsifier that cannot fire, which is worse than not shipping it.
    """
    from chess_anti_engine.stockfish.pid import DifficultyPID

    pid = DifficultyPID(
        initial_nodes=500_000, min_nodes=100_000, max_nodes=1_000_000,
        target_winrate=0.50, initial_wdl_regret=0.09,
        wdl_regret_min=0.001, wdl_regret_max=0.70,
    )
    assert pid.last_regret_fit_slope is None
    assert math.isnan(
        DifficultyState.from_pid(pid, None, TrialConfig()).regret_fit_slope,
    )

    # Seed the lever with a clean positive-slope history (higher regret ->
    # higher winrate, the physical direction), then one out-of-deadband
    # observation drives the FIT path in ``_step_lever``.
    pid.regret_lever.history.extend([
        (0.05, 0.40, 0.02), (0.07, 0.45, 0.02),
    ])
    pid.observe(wins=240, draws=0, losses=160, force=True)   # raw 0.60 at 0.09
    assert pid.last_regret_fit_slope is not None, (
        "the PID must expose the slope the gate needs"
    )
    assert pid.last_regret_fit_slope > 0.0
    ds = DifficultyState.from_pid(pid, None, TrialConfig())
    assert ds.regret_fit_slope == pytest.approx(pid.last_regret_fit_slope)

    # ...and it survives a restart, or the column blanks after every resume.
    revived = DifficultyPID(
        initial_nodes=500_000, min_nodes=100_000, max_nodes=1_000_000,
        target_winrate=0.50, initial_wdl_regret=0.09,
        wdl_regret_min=0.001, wdl_regret_max=0.70,
    )
    revived.load_state_dict(json.loads(json.dumps(pid.state_dict())))
    assert revived.last_regret_fit_slope == pytest.approx(pid.last_regret_fit_slope)


def test_the_gating_phase_puts_the_measured_difficulty_gap_on_the_sample(
    tmp_path: Path,
) -> None:
    """MUTATION: drop ``cur_wdl_regret``/``prev_wdl_regret``/
    ``regret_fit_slope`` from the ``AnchoredSample`` built in
    ``_run_net_gating``.

    End to end through the phase the loop actually calls: shard-measured
    difficulty per arm plus the PID's slope, arriving as a reported Elo column.
    """
    gate = _gate(mode=MODE_SHADOW, min_games_per_side=15)
    sp = SelfplayResult(
        gate_cur_w=100, gate_cur_d=0, gate_cur_l=97,
        gate_prev_w=19, gate_prev_d=0, gate_prev_l=19,
        gate_cur_wdl_regret=0.0900, gate_prev_wdl_regret=0.0850,
    )
    result = _run_training_and_gating(
        tc=TrialConfig(batch_size=512), trainer=_StubTrainer(),
        buf=[], holdout_buf=[], holdout_frozen=True, config={}, model_cfg=None,
        device="cpu",
        ds=DifficultyState(wdl_regret=0.09, sf_nodes=500_000,
                           regret_fit_slope=10.886),
        sims=64, sp=sp, positions_ingested=0, imported_samples_this_iter=0,
        gate=gate, gate_match_idx=0,
        gate_state_path=tmp_path / "gate_state.json", gate_hold=None,
        distributed_server_root=tmp_path, iteration_idx=5,
        iteration_zero_based=5, trial_id="t0", restore=RestoreResult(),
    )
    s = gate.samples[-1]
    assert s.cur_wdl_regret == pytest.approx(0.0900)
    assert s.prev_wdl_regret == pytest.approx(0.0850)
    assert s.regret_fit_slope == pytest.approx(10.886)
    assert result.gate_decision.sample_confound_elo == pytest.approx(
        0.0050 * 10.886 * ELO_PER_SCORE_AT_HALF,
    )


# --------------------------------------------------------------------------
# 16. gate_decision == 1 MEANS FOUR THINGS (review round 6, F8)
# --------------------------------------------------------------------------
def test_would_demote_separates_a_shadow_fire_from_a_clean_pass() -> None:
    """MUTATION: derive ``gate_would_demote`` from ``decision`` instead of from
    ``reason``, or drop the metric.

    ``gate_decision == 1`` is emitted for FOUR situations. In shadow mode -- the
    only mode this ships anywhere near -- "the gate wanted to fire" is THE
    event, and it used to be legible only as ``gate_reason_code == 6``, a number
    mentioned in no yaml and in no script.
    """
    clean = _feed(_gate(mode=MODE_SHADOW, min_games_per_side=40), _window(0.0))
    assert clean.decision == DECISION_PROMOTE
    assert clean.reason == "promote_no_regression"
    assert clean.would_demote is False
    assert gate_metrics(clean)["gate_would_demote"] == 0.0

    window_fire = _feed(
        _gate(mode=MODE_SHADOW, demote_delta_elo=-50.0, min_games_per_side=40),
        _window(-0.10),
    )
    assert window_fire.decision == DECISION_PROMOTE   # shadow never acts
    assert window_fire.reason == "shadow_would_demote"
    assert window_fire.would_demote is True
    assert gate_metrics(window_fire)["gate_would_demote"] == 1.0
    assert gate_metrics(window_fire)["gate_reason_code"] == 6.0

    # A shadow-suppressed STEP demote has its own reason code, so the two legs
    # are distinguishable in a dashboard rather than merged.
    g = _gate(mode=MODE_SHADOW, min_games_per_side=15)
    for i in range(23):
        g.observe(_sample(i, 0.50, 0.50))
    # The current model lost every game: a -347 Elo one-iteration step.
    g.observe(AnchoredSample(23, cur_w=0, cur_d=0, cur_l=197,
                             prev_w=19, prev_d=0, prev_l=19))
    step_fire = g.decide()
    assert step_fire.decision == DECISION_PROMOTE
    assert step_fire.reason == "shadow_would_demote_step"
    assert step_fire.would_demote is True
    assert gate_metrics(step_fire)["gate_reason_code"] == 9.0

    # hold_expired is ALSO a 1 that means the rule fired.
    g2 = _gate(demote_delta_elo=-50.0, min_games_per_side=40)
    for s in _window(-0.30):
        g2.observe(s)
    g2.holds = g2.cfg.max_hold_iters
    expired = g2.decide()
    assert expired.decision == DECISION_PROMOTE
    assert expired.reason == "hold_expired"
    assert expired.would_demote is True

    # ...and both the yaml and the shipped command name the column.
    yaml_src = (Path(__file__).resolve().parents[1]
                / "configs" / "pbt2_small.yaml").read_text("utf-8")
    script = (Path(__file__).resolve().parents[1]
              / "scripts" / "gate_shadow_readout.py").read_text("utf-8")
    for text in (yaml_src, script):
        assert "gate_would_demote" in text
        assert "shadow_would_demote_step" in text


# --------------------------------------------------------------------------
# 17. AUDIT WAVE 3 -- the gate internals and the readout's per-axis rule
#     (docs/experiment_ledger.md, scratchpad/code_audit_20260803/GATE_AUDIT.md)
# --------------------------------------------------------------------------
def test_the_degenerate_variance_floor_separates_float_noise_from_real_spread(
) -> None:
    """MUTATION (audit G3-3): restore ``se <= 0.0`` as the degenerate test.

    It MISSED 51.2% of the windows it was written for. For K copies of one
    float ``d`` the computed mean is only sometimes exactly ``d``; when the
    summation does not round-trip, ``se`` comes out at ~1e-17 -- strictly
    positive -- and the gate then DEMOTED on ``elo_lo == elo_hi``, a confidence
    interval of width zero. That is precisely the certainty claim the guard's
    own comment cites the L11 lesson against.

    Three things asserted, because the fix is a threshold and a threshold
    without its margins is a future mystery:

      1. the fuzz from the audit -- every identical-delta window is caught;
      2. the round-trip residue sits far BELOW the floor;
      3. the smallest spread REAL data can produce sits far ABOVE it.
    """
    rng = random.Random(7)
    bypassed = 0
    residues: list[float] = []
    for _ in range(20_000):
        nc, npv = rng.randint(120, 220), rng.randint(30, 100)
        cw = rng.randint(0, nc)
        cd = rng.randint(0, nc - cw)
        pw = rng.randint(0, npv)
        pd = rng.randint(0, npv - pw)
        s = AnchoredSample(
            iteration=0, cur_w=cw, cur_d=cd, cur_l=nc - cw - cd,
            prev_w=pw, prev_d=pd, prev_l=npv - pw - pd,
        )
        for k in (8, 24):
            ds = [s.delta] * k
            m = sum(ds) / k
            var = sum((x - m) ** 2 for x in ds) / (k - 1)
            se = math.sqrt(var / k)
            residues.append(se)
            if not _window_is_degenerate(ds, se):
                bypassed += 1
    assert bypassed == 0, (
        f"{bypassed}/40000 identical-delta windows escaped the guard; "
        "`se <= 0.0` let 20466 of them through"
    )
    # The old test would have failed here: the residue is POSITIVE half the
    # time, which is the whole finding.
    assert max(residues) > 0.0, "the fuzz must actually produce nonzero se"

    # MARGIN 1 -- float noise is ~3 orders of magnitude under the floor.
    floor = _DEGENERATE_SE_ULPS * math.ulp(0.15)
    assert max(residues) < floor / 100.0, (max(residues), floor)
    # MARGIN 2 -- the smallest spread two DISTINCT w/d/l splits can produce is
    # ~10 orders of magnitude over it. One game moved in the larger arm at the
    # live shape is the finest step real data has.
    a = AnchoredSample(iteration=0, cur_w=70, cur_d=0, cur_l=130,
                       prev_w=15, prev_d=0, prev_l=25)
    b = AnchoredSample(iteration=0, cur_w=71, cur_d=0, cur_l=129,
                       prev_w=15, prev_d=0, prev_l=25)
    real = [a.delta, b.delta] * 12
    real_se = math.sqrt(
        sum((x - sum(real) / len(real)) ** 2 for x in real) / (len(real) - 1)
        / len(real))
    assert real_se > floor * 1e8, (real_se, floor)
    assert not _window_is_degenerate(real, real_se), "real spread must decide"

    # ...and end to end: the window the audit reproduced demoted on a
    # zero-width interval, and must now refuse.
    g = _gate(min_games_per_side=5)
    for i in range(12):
        g.observe(_sample(i, 0.35, 0.50))
    d = g.decide()
    assert d.decision == DECISION_NOT_RUN, d
    assert d.reason == "degenerate_variance", d
    # A refused window must not publish an interval at all.
    assert math.isnan(d.elo_lo), d
    assert math.isnan(d.elo_hi), d


def test_a_hold_that_published_anyway_is_visible_in_the_metrics(
    tmp_path: Path,
) -> None:
    """MUTATION (audit G3-4): drop ``gate_hold_effective`` from ``gate_metrics``.

    "held, fallback present" and "held, fallback MISSING -> published anyway"
    were BYTE-IDENTICAL in every metric row: ``gate_decision=0``,
    ``gate_reason_code=5``, ``gate_holds=1`` in both. The only witness was a
    stdout line in the Ray actor log that no csv consumer reads -- so
    progress.csv could answer "did the gate fire?" and could not answer "was
    the fleet actually held back?", which is the question the standing bias
    says to ask. A hold that never reached the publish path burns the
    ``max_hold_iters`` budget and reports as a successful brake.
    """
    present = tmp_path / "gate_promoted_model.pt"
    _g, demote = _demoting_gate(present)

    with_anchor = GateHoldController(
        gate=_gate(mode=MODE_ENFORCE), promoted_model_path=present,
    ).on_decision(demote)
    without = GateHoldController(
        gate=_gate(mode=MODE_ENFORCE),
        promoted_model_path=tmp_path / "absent.pt",
    ).on_decision(demote)

    m_with, m_without = gate_metrics(with_anchor), gate_metrics(without)
    differing = {k for k in m_with if m_with[k] != m_without[k]}
    assert "gate_hold_effective" in differing, differing
    assert m_with["gate_hold_effective"] == 1.0
    assert m_without["gate_hold_effective"] == 0.0
    assert m_with["gate_fallback_missing"] == 0.0
    assert m_without["gate_fallback_missing"] == 1.0
    # The verdict columns are identical in BOTH -- that is the finding.
    for same in ("gate_decision", "gate_reason_code", "gate_holds"):
        assert m_with[same] == m_without[same], same


def test_anchor_refresh_failures_and_the_anchor_stamp_survive_a_restart(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """MUTATION (audit G3-5): drop the controller state from gate_state.json,
    or accept the anchor on ``.is_file()`` alone.

    ``anchor_is_trustworthy`` is the only thing standing between a stale
    ~252 MB export and the whole selfplay fleet, and it was derived from an
    in-memory counter that died with the process -- so the ``train.sh``
    auto-resume that follows an ENOSPC crash reset it to 0 and re-armed the
    stale anchor as trustworthy. The anchor itself carried no age stamp at
    all, so an ``off -> on`` cycle re-armed whatever a previous era had left
    in ``durable_dir``.
    """
    import logging

    gate = _gate(mode=MODE_ENFORCE)
    ctrl = GateHoldController(
        gate=gate, promoted_model_path=tmp_path / "gate_promoted_model.pt",
    )
    for _ in range(20):
        ctrl.note_anchor_refresh_failed(OSError(28, "No space left on device"))
    assert ctrl.anchor_refresh_failures == 20
    assert ctrl.anchor_is_trustworthy is False

    # THE RESTART. The counter must come back, not reset to zero.
    state = json.loads(json.dumps(ctrl.state_dict()))
    revived = GateHoldController.create(
        _gate(mode=MODE_ENFORCE), durable_dir=tmp_path, state=state,
    )
    assert revived.anchor_refresh_failures == 20
    assert revived.anchor_is_trustworthy is False

    # THE STAMP. An anchor with no stamp is a file from before stamping or
    # from a previous era; either way it cannot say when it was written.
    anchor = tmp_path / "gate_promoted_model.pt"
    anchor.write_bytes(b"an-anchor-from-a-previous-era")
    g2 = _gate(mode=MODE_ENFORCE)
    g2.hold_active = True
    with caplog.at_level(logging.ERROR):
        unstamped = GateHoldController.create(
            g2, durable_dir=tmp_path, current_iteration=500,
        )
    assert unstamped.anchor_stamp_ok is False
    assert unstamped.hold_path is None, "an unstamped anchor must not be served"
    assert unstamped.anchor_is_trustworthy is False
    assert "stamp is MISSING" in caplog.text

    # A stamp older than the longest hold the gate may impose is refused for
    # the same reason a failed refresh is.
    write_anchor_stamp(anchor, iteration=100, trainer_step=1,
                       model_sha256="abc", trial_id="t0")
    stale = GateHoldController.create(
        _gate(mode=MODE_ENFORCE), durable_dir=tmp_path, current_iteration=500,
    )
    assert stale.anchor_stamp_ok is False
    assert stale.anchor_is_trustworthy is False

    # ...and a stamp from this era is accepted, with its provenance readable.
    write_anchor_stamp(anchor, iteration=498, trainer_step=88_000,
                       model_sha256="deadbeef", trial_id="t0")
    g3 = _gate(mode=MODE_ENFORCE)
    g3.hold_active = True
    fresh = GateHoldController.create(
        g3, durable_dir=tmp_path, current_iteration=500,
    )
    assert fresh.anchor_stamp_ok is True
    assert fresh.hold_path == anchor
    assert fresh.anchor_is_trustworthy is True
    stamp = read_anchor_stamp(anchor)
    assert stamp is not None
    assert (stamp.iteration, stamp.trainer_step, stamp.model_sha256) == (
        498, 88_000, "deadbeef")


def test_a_failing_gate_state_write_is_counted_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """MUTATION (audit G3-6): restore ``contextlib.suppress(Exception)`` around
    the ``gate_state.json`` write.

    Two functions above it sits the anchor refresh, whose comment reads "THE
    FAILURE HERE USED TO BE SILENT ... no log line, no metric and no way to
    notice ... Instead the outcome is RECORDED". The identical reasoning was
    never carried across to the state write, which is broader still
    (``Exception``, not ``OSError``). A failed write leaves the persisted
    window and hold latch one iteration behind, and the restart that follows
    silently resumes on it.
    """
    import logging

    import chess_anti_engine.tune.trainable_phases as phases

    def _boom(*_a: object, **_kw: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(phases, "atomic_write_text", _boom)
    gate = _gate(mode=MODE_SHADOW)
    sp = SelfplayResult(
        gate_cur_w=40, gate_cur_d=10, gate_cur_l=50,
        gate_prev_w=50, gate_prev_d=10, gate_prev_l=40,
    )
    result = None
    with caplog.at_level(logging.ERROR):
        for _ in range(3):
            result = _run_training_and_gating(
                tc=TrialConfig(batch_size=512), trainer=_StubTrainer(),
                buf=[], holdout_buf=[], holdout_frozen=True, config={}, model_cfg=None,
        device="cpu",
                ds=DifficultyState(wdl_regret=0.089, sf_nodes=50_000),
                sims=64, sp=sp, positions_ingested=0,
                imported_samples_this_iter=0, gate=gate, gate_match_idx=0,
                gate_state_path=tmp_path / "gate_state.json", gate_hold=None,
                distributed_server_root=tmp_path, iteration_idx=5,
                iteration_zero_based=5, trial_id="t0", restore=RestoreResult(),
            )
    assert gate.state_write_failures == 3
    assert "state write FAILED" in caplog.text
    assert "No space left on device" in caplog.text
    # ...and it reaches progress.csv rather than living in the object.
    assert result is not None
    assert gate_metrics(result.gate_decision)["gate_state_write_failures"] == 3.0
    # The trial keeps training: an optional alarm may not take a run down.
    assert result.gate_decision.reason != ""


def test_a_hold_then_a_release_does_not_move_the_anchor(tmp_path: Path) -> None:
    """MUTATION (audit G3-8): restore ``if gate_hold_model_path is None`` as the
    whole refresh condition.

    The anchor is documented as "the last promoted export" and read as a
    ratchet. It was refreshed on every iteration that was not itself held --
    including ``DECISION_NOT_RUN``, including a shadow-suppressed demote, and
    including the RELEASE iteration, where it was overwritten with the very
    weights the preceding hold existed to keep off the fleet. "Last promoted"
    was really "last iteration's", and the documented 55-66% partial braking
    was a one-iteration-lag filter that re-poisoned itself at each release.
    """
    trainer = _MarkerTrainer()
    anchor = tmp_path / "gate_promoted_model.pt"
    ctrl = GateHoldController(gate=_gate(mode=MODE_ENFORCE),
                              promoted_model_path=anchor)

    def publish(decision: GateDecision | None) -> None:
        if decision is not None:
            ctrl.on_decision(decision)
        _publish_iter(tmp_path, trainer, hold=ctrl.hold_path,
                      promoted=anchor, controller=ctrl)

    promote = GateDecision(decision=DECISION_PROMOTE,
                           reason="promote_no_regression", mode=MODE_ENFORCE,
                           games_cur=197, games_prev=38)
    not_run = GateDecision(decision=DECISION_NOT_RUN, reason="insufficient_iters",
                           mode=MODE_ENFORCE)
    shadow_fire = GateDecision(decision=DECISION_PROMOTE, mode=MODE_ENFORCE,
                               reason="shadow_would_demote",
                               games_cur=197, games_prev=38)
    released = GateDecision(decision=DECISION_PROMOTE, reason="hold_expired",
                            mode=MODE_ENFORCE, games_cur=197, games_prev=38)
    _g, demote = _demoting_gate(tmp_path / "unused.pt")

    # A genuine promote creates and then tracks the anchor.
    publish(promote)
    assert anchor.read_bytes() == b"trainer-export-1"
    publish(promote)
    assert anchor.read_bytes() == b"trainer-export-2"

    # A NOT_RUN verdict does not vouch for anything, so it must not re-anchor.
    publish(not_run)
    assert anchor.read_bytes() == b"trainer-export-2"
    # Nor does a demote rule that fired and was merely suppressed.
    publish(shadow_fire)
    assert anchor.read_bytes() == b"trainer-export-2"

    # THE HOLD, then THE RELEASE. Neither may move the anchor: the release
    # iteration publishes exactly the weights the hold was rejecting.
    publish(demote)
    assert ctrl.hold_path == anchor, "the hold must actually engage"
    assert anchor.read_bytes() == b"trainer-export-2"
    publish(released)
    assert anchor.read_bytes() == b"trainer-export-2", (
        "the release iteration must not overwrite the fallback with the "
        "weights the hold was holding back"
    )
    # ...and the next genuine promote is what re-anchors, with a fresh stamp.
    publish(promote)
    assert anchor.read_bytes().startswith(b"trainer-export-")
    stamp = read_anchor_stamp(anchor)
    assert stamp is not None
    assert stamp.iteration == 1


def test_the_attribution_axis_names_the_refresh_lag_and_reports_both(
) -> None:
    """MUTATION (audit G3-2): report the leg as ``prev_share`` again.

    On the production fleet of 2026-08-03 this leg exited 2 -- the
    pre-registered KILL -- with a message an operator is explicitly told to
    read as "your split is mis-attributing shards", at 0.88x the reference
    cadence, so the cadence leg stayed silent and the whole discrepancy landed
    on a leg whose name blames the split. Whether the split is at fault is NOT
    established: the pooled-count identity passes 109 of 109 live rows (which
    proves no game was lost, not that the labels are right -- a coin-shuffle
    preserves that sum exactly), and an independent shard re-derivation of the
    same trial puts the refresh lag back at the calibration value.

    The two causes are NOT separable from progress.csv -- ``refresh_lag =
    prev_share * cadence`` is an identity -- so the fix is not a new test. It
    is that the axis is named for the quantity it actually measures, reports
    that quantity in seconds against the reference, and says what to do.
    """
    live_lag = 194.2
    rows = [(180, 62, 30.0 * (1 if i % 2 else -1),
             live_lag / (62 / 242.0)) for i in range(40)]
    r = shadow_readout_verdict(rows, last_n=40)
    assert r.verdict == READOUT_KILL, r
    leg = [x for x in r.failed_legs if x.startswith("refresh_lag")]
    assert leg, r.failed_legs
    assert "vs reference" in leg[0]
    assert "re-derive the reference from shards" in leg[0], leg[0]
    assert "before reading this as an attribution failure" in leg[0]
    # Both readings are on the object, in the cadence-free form.
    assert r.refresh_lag_seconds == pytest.approx(live_lag, abs=1.0)
    assert r.ref_refresh_lag_seconds == pytest.approx(
        OFFLINE.refresh_lag_seconds, abs=1e-9)
    # ...and the cadence leg is NOT what fired, which is the whole reason the
    # false kill was legible as an attribution finding.
    assert not [x for x in r.failed_legs if x.startswith("cadence")], r


def test_rebasing_the_refresh_lag_does_not_disarm_the_negative_control(
) -> None:
    """The safety property of ``--refresh-lag-seconds`` / a re-derivation.

    Re-deriving the reference at the fleet's current lag is the sanctioned way
    to make this leg pass again, so the thing to prove is that it does NOT
    also make the leg unable to fail. A reference re-based at the live lag
    must still kill a coin-shuffled attribution -- otherwise "re-derive the
    constants" would be a way to condition the control on its own outcome.
    """
    from dataclasses import replace as _replace

    live_lag = 194.2
    cadence = OFFLINE.mean_iter_seconds
    rebased = _replace(OFFLINE, prev_share=live_lag / cadence)
    assert rebased.refresh_lag_seconds == pytest.approx(live_lag, abs=1e-6)

    healthy = [(180, 62, 30.0 * (1 if i % 2 else -1),
                live_lag / (62 / 242.0)) for i in range(40)]
    assert shadow_readout_verdict(
        healthy, last_n=40, ref=rebased).verdict == READOUT_PROMOTE

    rng = random.Random(11)
    killed = 0
    for _ in range(50):
        shuffled = []
        for cur, prev, delta, secs in healthy:
            total = cur + prev
            c = sum(1 for _ in range(total) if rng.random() < 0.5)
            shuffled.append((c, total - c, delta, secs))
        r = shadow_readout_verdict(shuffled, last_n=40, ref=rebased)
        killed += r.verdict == READOUT_KILL
    assert killed == 50, f"the rebased reference killed only {killed}/50"


def test_the_readout_reports_every_axis_state_not_only_the_failures() -> None:
    """⚑ AN EXIT CODE IS AN *OR* OVER AXES -- so the axes must be printable.

    ``failed_legs`` said which legs fired and nothing said which legs were
    EVALUATED, and "not evaluated" is the state the confound axis has been in
    for 109 of 109 live rows.
    """
    rows = [(197, 38, 44.0 * (1 if i % 2 else -1), 721.0, 235.0, 5.0 * ((i % 5) - 2))
            for i in range(40)]
    r = shadow_readout_verdict(rows, last_n=40)
    names = {leg.name for leg in r.legs}
    for axis in ("cadence", "anchored_games_vs_pooled", "usable_frac",
                 "games_per_second", "refresh_lag", "sd_delta_elo",
                 "|mean_delta_elo|", "window_length"):
        assert axis in names, (axis, names)
    assert {leg.state for leg in r.legs} == {LEG_PASS}, r.per_leg_report()
    report = r.per_leg_report()
    assert "confound" in report
    assert LEG_PASS in report
    assert r.confound_is_measured is True
    assert readout_exit_code(r) == READOUT_EXIT_PROMOTE


def test_a_window_with_no_confound_measurement_can_never_exit_zero() -> None:
    """⚑ THE EXIT-CODE TRAP, pre-registered in the ledger before this ran.

    Fixing the attribution leg alone makes the reference window exit 0 --
    ``promote_to_enforce``, the signal to set ``gate_mode: enforce`` -- while
    the PID-confound leg, the deciding KILL rule the ledger registered for
    exactly this promotion, measures nothing at all.
    ``gate_sample_confound_elo`` is NaN on 109 of 109 rows written so far
    because the server compactor rebuilt ``ShardMeta`` without
    ``opponent_wdl_regret_limit`` (fixed by #323, deploy-gated on a server
    restart -- so this leg stays unmeasured until one happens AND a full window
    of new shards lands).
    """
    # Every OTHER axis measured, including the pooled identity -- otherwise
    # this would exit 6 and test the wrong refusal (review R3).
    healthy = [(197, 38, 44.0 * (1 if i % 2 else -1), 721.0, 235.0)
               for i in range(40)]
    r = shadow_readout_verdict(healthy, last_n=40)
    assert r.verdict == READOUT_PROMOTE, "every other axis passes"
    assert r.n_confound == 0
    assert r.confound_is_measured is False
    assert readout_exit_code(r) == READOUT_EXIT_CONFOUND_UNMEASURED
    assert LEG_UNMEASURED in r.per_leg_report()

    # A hold does not launder it either.
    short = shadow_readout_verdict(healthy[:5], last_n=40)
    assert short.verdict == READOUT_HOLD
    assert readout_exit_code(short) == READOUT_EXIT_CONFOUND_UNMEASURED

    # A KILL still outranks it: a failing leg is a plumbing fact about THIS
    # window and must be read first.
    broken = [(197, 250, 44.0 * (1 if i % 2 else -1), 721.0, 447.0)
              for i in range(40)]
    r_kill = shadow_readout_verdict(broken, last_n=40)
    assert r_kill.verdict == READOUT_KILL
    assert readout_exit_code(r_kill) == READOUT_EXIT_KILL

    # ...and once the column carries numbers, the axis is measured and the
    # command can promote.
    with_conf = [(*row, 5.0 * ((i % 5) - 2)) for i, row in enumerate(healthy)]
    r_ok = shadow_readout_verdict(with_conf, last_n=40)
    assert r_ok.n_confound == 40
    assert readout_exit_code(r_ok) == READOUT_EXIT_PROMOTE


def test_the_reference_rederivation_recovers_an_injected_refresh_lag() -> None:
    """The re-derivation must measure the lag it is told to measure.

    ⚑ AND IT MUST NOT READ THE GATE'S OWN COLUMNS. ``OfflineReference`` was
    built from shard ``.zattrs``, independently of the in-loop splitter it is
    then used to check; re-deriving it from ``gate_sample_games_*`` would
    condition the control on its own outcome and a splitter defect present in
    both windows would cancel exactly.

    Positive control: synthesise a fleet whose previous-model shards stop
    arriving ``lag`` seconds into each iteration and check the derived lag.
    Negative control: destroy the model-step labelling and watch the derived
    share move, so the re-derivation is sensitive to the thing it claims to
    measure.
    """
    cadence, lag, n_iters = 600.0, 180.0, 30
    iterations = [(1_000_000.0 + (i + 1) * cadence, cadence) for i in range(n_iters)]
    shards: list[ShardArm] = []
    for i, (end, secs) in enumerate(iterations):
        start = end - secs
        # prev-model shards land in the first `lag` seconds of the iteration,
        # cur-model shards over the whole of it, at one shard per 30 s.
        shards.extend(
            ShardArm(start + t, model_step=i, model_sha256="p",
                     wins=5, draws=1, losses=4)
            for t in range(0, int(lag), 30)
        )
        shards.extend(
            ShardArm(start + t + 1, model_step=i + 1, model_sha256="c",
                     wins=5, draws=1, losses=4)
            for t in range(int(lag), int(secs), 30)
        )
    r = rederive_reference_from_shards(iterations, shards)
    assert r.n_usable == n_iters
    assert r.mean_iter_seconds == pytest.approx(cadence)
    assert r.refresh_lag_seconds == pytest.approx(lag, rel=0.1), r
    assert r.prev_share == pytest.approx(lag / cadence, rel=0.1), r
    # Both arms score identically by construction, so the anchored delta is 0.
    assert r.mean_delta_elo == pytest.approx(0.0, abs=1e-9)

    # NEGATIVE CONTROL: relabel every shard's model_step at random and the
    # derived share must move away from the injected lag.
    rng = random.Random(3)
    scrambled = [
        ShardArm(s.generated_at_unix,
                 model_step=rng.choice([s.model_step, s.model_step + 1]),
                 model_sha256=s.model_sha256, wins=s.wins, draws=s.draws,
                 losses=s.losses)
        for s in shards
    ]
    bad = rederive_reference_from_shards(iterations, scrambled)
    assert abs(bad.prev_share - lag / cadence) > 0.05, bad


def test_the_readout_help_names_the_rederivation_and_the_exit_code() -> None:
    """The re-derivation obligation is in ``--help``, not only in a ledger entry.

    ``OfflineReference`` was measured in 2026-06 and every leg is stated
    relative to it. An operator running the pre-committed command has to be
    told, by the command, that the constants must be re-derived from
    current-lag shards before ``gate_mode`` leaves ``off``.
    """
    from scripts import gate_shadow_readout

    doc = gate_shadow_readout.__doc__ or ""
    assert "--rederive-reference" in doc
    assert "RE-DERIVED BEFORE ``gate_mode`` LEAVES ``off``" in doc
    assert "confound unmeasured (5)" in doc
    assert "n_confound" in doc or "3 measurements" in doc
    # The flag exists and is wired, not merely documented.
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "gate_shadow_readout.py").read_text("utf-8")
    assert "--refresh-lag-seconds" in src
    assert "rederive_reference_with_phase_sweep" in src


def test_the_loop_body_hands_the_controller_to_the_gating_phase() -> None:
    """MUTATION: ``gate_hold=gate_hold`` -> ``gate_hold=None`` at the call site.

    The controller's counters (``anchor_refresh_failures``,
    ``fallback_missing``) are persisted by ``_run_net_gating``, which is the one
    place ``gate_state.json`` is written. Passing ``None`` keeps every test
    green -- the argument is required, so it cannot be forgotten, but it CAN be
    satisfied with nothing -- while the counters silently go back to dying with
    the process, which is audit G3-5 restored.

    ``on_decision`` is also told the iteration, or ``gate_anchor_age_iters``
    stays NaN forever and the AGE half of ``anchor_is_trustworthy`` becomes a
    guard that cannot fire.
    """
    body = _train_trial_body()
    phase = [
        n for n in ast.walk(body)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_run_training_and_gating"
    ]
    assert len(phase) == 1, phase
    passed = {
        kw.arg: ast.unparse(kw.value) for kw in phase[0].keywords if kw.arg
    }
    assert passed.get("gate_hold") == "gate_hold", passed

    decision_call = _calls_on(body, "gate_hold", "on_decision")[0]
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in decision_call.keywords if kw.arg}
    assert "iteration" in kwargs, ast.unparse(decision_call)
    assert "iteration_idx" in kwargs["iteration"], kwargs

    # ...and the controller is built AFTER the restore, because the stamp check
    # needs the iteration this process is resuming at.
    create = [
        n for n in ast.walk(body)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "create"
        and ast.unparse(n.func).endswith("GateHoldController.create")
    ]
    assert len(create) == 1, create
    create_kwargs = {kw.arg: ast.unparse(kw.value) for kw in create[0].keywords if kw.arg}
    assert "current_iteration" in create_kwargs, create_kwargs
    assert "state" in create_kwargs, create_kwargs


# --------------------------------------------------------------------------
# 18. REVIEW ROUND (PR #324): the re-derivation's own uncertainty, the
#     mandatory-axis rule, and the anchor's first publish after a restart
# --------------------------------------------------------------------------
def test_the_rederivation_refuses_a_phase_unstable_split() -> None:
    """MUTATION (review R1): drop the phase sweep and emit the point estimate.

    THE RECONSTRUCTION CANNOT PIN ITS OWN BIN EDGES. ``generated_at_unix`` on a
    ``_compacted`` shard is the SERVER's flush stamp, while the loop attributes
    at INGEST -- a shard flushed at the end of iteration N is ingested in N+1,
    where its sha is ``prev``. On live data a quarter-iteration shift moves
    ``prev_share`` from 0.177 to 0.428 with the reconstruction's internal
    structure unchanged, so its own uncertainty SPANS both readings of the
    disagreement the ledger pre-registers it to adjudicate.

    A tool that hands an operator a constant it cannot support is worse than no
    tool: the pre-registered step "resolve this from shards" would be satisfied
    by an artifact. So the sweep is the output, and the constants block refuses.
    """
    # A fleet whose prev-arm shards arrive in the first `lag` seconds. Shifting
    # the bin edges re-assigns them wholesale, which is the instability.
    cadence, lag, n = 600.0, 180.0, 30
    iterations = [(1_000_000.0 + (i + 1) * cadence, cadence) for i in range(n)]
    shards: list[ShardArm] = []
    for i, (end, secs) in enumerate(iterations):
        start = end - secs
        shards.extend(
            ShardArm(start + t, model_step=i, model_sha256="p",
                     wins=5, draws=1, losses=4)
            for t in range(0, int(lag), 30)
        )
        shards.extend(
            ShardArm(start + t + 1, model_step=i + 1, model_sha256="c",
                     wins=5, draws=1, losses=4)
            for t in range(int(lag), int(secs), 30)
        )

    swept = rederive_reference_with_phase_sweep(iterations, shards)
    lo, hi = swept.band("prev_share")
    assert hi - lo > 0.06, (
        f"this construction must BE phase-unstable ({lo:.4f}..{hi:.4f}), or the "
        "test cannot see the refusal"
    )
    assert swept.prev_share_is_phase_stable is False
    body = swept.as_offline_reference_source()
    assert body.strip().startswith("REFUSED"), body
    assert "prev_share:" not in body, "no constant may be emitted"
    assert "phase-unstable" in body
    # The bands are REPORTED for every swept field, so the operator can see
    # which of the seven numbers this command can and cannot deliver.
    for name in ("mean_games_cur", "mean_games_prev", "prev_share",
                 "refresh_lag_seconds", "mean_delta_elo", "sd_delta_elo",
                 "mean_iter_seconds"):
        band_lo, band_hi = swept.band(name)
        assert band_hi >= band_lo, name
    # ⚑ NOT EVEN `mean_iter_seconds` IS PHASE-INVARIANT (review B1). An earlier
    # revision asserted its spread was exactly 0 "because the seconds come from
    # progress.csv", and pinned that with a fleet where every bin is usable at
    # every shift -- a test that confirmed its own premise. The mean is taken
    # over the USABLE bins, and usability is precisely what the phase moves, so
    # the claim is asserted here against a fleet where it varies. It is checked
    # in the `!= 0` direction on purpose: a future change that makes the
    # cadence genuinely phase-free must come with the doc lines it invalidates.
    # Iterations of two different lengths, where the LONG ones hold their prev
    # arm late in the bin: a leftward shift pushes that arm out, those bins stop
    # being usable, and the surviving mean is over the short ones alone.
    vary_iters: list[tuple[float, float]] = []
    clock = 1_000_000.0
    for i in range(30):
        secs = 600.0 if i < 20 else 1200.0
        clock += secs
        vary_iters.append((clock, secs))
    varying: list[ShardArm] = []
    for i, (end, secs) in enumerate(vary_iters):
        start = end - secs
        prev_at = start + (0.9 if i >= 20 else 0.1) * secs
        varying.append(ShardArm(start + 0.1 * secs, model_step=i + 1,
                                model_sha256="c", wins=5, draws=1, losses=4))
        varying.append(ShardArm(prev_at + 1.0, model_step=i,
                                model_sha256="p", wins=5, draws=1, losses=4))
    vary_swept = rederive_reference_with_phase_sweep(vary_iters, varying)
    usable_counts = {n for _, n, _ in vary_swept.shift_usable}
    assert len(usable_counts) > 1, (
        f"this fleet must have phase-dependent usability ({usable_counts}), or "
        "the assertion below cannot see what it is for"
    )
    assert vary_swept.spread("mean_iter_seconds") != pytest.approx(0.0, abs=1e-9), (
        "the cadence is averaged over the usable bins, so when usability moves "
        "with the phase so does it -- the invariance claim was false"
    )

    # ...and a fleet whose split does NOT move with the phase still emits.
    steady = [
        ShardArm(1_000_000.0 + (i + 1) * cadence - secs / 2.0,
                 model_step=i + 1, model_sha256="c", wins=5, draws=1, losses=4)
        for i, (_, secs) in enumerate(iterations)
    ] + [
        ShardArm(1_000_000.0 + (i + 1) * cadence - secs / 2.0 + 1.0,
                 model_step=i, model_sha256="p", wins=5, draws=1, losses=4)
        for i, (_, secs) in enumerate(iterations)
    ]
    calm = rederive_reference_with_phase_sweep(iterations, steady)
    assert calm.prev_share_is_phase_stable is True, calm.band("prev_share")
    assert "prev_share:" in calm.as_offline_reference_source()


def test_a_collapsed_shift_does_not_set_the_band_and_cannot_buy_stability() -> None:
    """MUTATION (review B2): band over ALL shifts, or drop the thin-sweep guard.

    A shift whose bins mostly hold ONE model measures the binning falling apart,
    not the split, and it was setting a band endpoint the refusal was then
    banked on. Live: the +0.25 shift kept 13 of 68 bins and produced the quoted
    upper bound 0.8232. The verdict does not change (the surviving shifts still
    span eight times the tolerance) -- but a number that goes in the ledger has
    to come from the population it claims to describe.

    ⚑ THE EXCLUSION IS BY SAMPLE SIZE, NEVER BY THE VALUE. `n_usable` is fixed
    by the binning before any share is read; filtering on the share itself would
    be conditioning the control on its own outcome. And because narrowing a band
    can only ever move a verdict TOWARD "stable", the second half of this test
    is the one that matters: the exclusion must not be able to buy stability.
    """
    # Every bin publishes a new model 25% of the way in -- so a +0.25 shift
    # aligns each bin with exactly one model and collapses it. Every 8th bin
    # publishes at 35% instead, leaving a handful of two-model bins at that
    # shift: a small, unrepresentative sample with an extreme share.
    cadence, n = 600.0, 40
    iterations = [(1_000_000.0 + (i + 1) * cadence, cadence) for i in range(n)]
    shards: list[ShardArm] = []
    for i, (end, secs) in enumerate(iterations):
        start = end - secs
        frac = 0.35 if i % 8 == 0 else 0.25
        for k in range(20):
            t = k * secs / 20.0
            step = i if t < frac * secs else i + 1
            shards.append(ShardArm(
                start + t, model_step=step,
                model_sha256="p" if step == i else "c",
                wins=5, draws=1, losses=4,
            ))

    swept = rederive_reference_with_phase_sweep(iterations, shards)
    counts = {shift: (n_use, degen) for shift, n_use, degen in swept.shift_usable}
    assert counts[0.25][1] is True, f"+0.25 must collapse: {counts}"
    assert all(not degen for shift, (_, degen) in counts.items() if shift != 0.25), counts
    assert swept.n_band_shifts == 4

    # The collapsed shift's own value is OUTSIDE the band it no longer sets.
    collapsed = rederive_reference_from_shards(
        iterations, shards, bin_shift_fraction=0.25)
    lo, hi = swept.band("prev_share")
    assert not lo <= collapsed.prev_share <= hi, (
        f"the collapsed shift ({collapsed.prev_share:.4f}) must fall outside "
        f"[{lo:.4f}, {hi:.4f}], or this test cannot see the exclusion"
    )
    assert collapsed.n_usable < swept.n_usable / 2

    # ...and the verdict is unchanged: the surviving shifts are still wide.
    assert swept.prev_share_is_phase_stable is False
    assert "non-degenerate" in swept.as_offline_reference_source()

    # THE GUARD THAT MATTERS. A band computed over two surviving shifts is not
    # evidence that the phase does not matter, however narrow it is -- otherwise
    # a sweep could be declared stable BECAUSE most of its points collapsed.
    thin = dataclasses.replace(
        swept,
        bands={**swept.bands, "prev_share": (0.20, 0.20)},
        shift_usable=((-0.5, 40, False), (-0.25, 2, True), (0.0, 40, False),
                      (0.25, 2, True), (0.5, 2, True)),
    )
    assert thin.spread("prev_share") == pytest.approx(0.0), "a zero-width band"
    assert thin.prev_share_is_phase_stable is False, (
        "two surviving shifts cannot establish stability"
    )
    body = thin.as_offline_reference_source()
    assert body.strip().startswith("REFUSED")
    assert "too thin to judge" in body
    # The same band over enough shifts DOES emit, so the guard is not a
    # permanent refusal wearing a reason.
    wide = dataclasses.replace(thin, shift_usable=tuple(
        (s, 40, False) for s in (-0.5, -0.25, 0.0, 0.25, 0.5)))
    assert wide.prev_share_is_phase_stable is True
    assert "prev_share:" in wide.as_offline_reference_source()


def test_the_rederivation_excludes_quarantined_shards(tmp_path: Path) -> None:
    """MUTATION (review R2): rglob the whole tree again.

    ``--help`` tells the operator to pass ``.../trials/<trial>/processed``,
    which holds ``_compacted/`` AND ``_quarantine/``. The quarantined shards
    are ones the loop REJECTED and never counted -- on the live trial, 5 shards
    / 136 curriculum games -- so including them makes the "independent
    reconstruction" a mixture of what the loop used and what it threw away.
    """
    root = tmp_path / "processed"
    for sub, wins in (("_compacted", 7), ("_quarantine", 99)):
        d = root / sub / "s.zarr"
        d.mkdir(parents=True)
        (d / ".zattrs").write_text(json.dumps({
            "generated_at_unix": 1_000_000.0, "model_step": 5,
            "model_sha256": "abc", "wins": wins, "draws": 0, "losses": 0,
        }), encoding="utf-8")

    arms, counts = read_shard_arms(root)
    assert [a.wins for a in arms] == [7], "the quarantined shard must not count"
    assert counts["_compacted"] == 1
    assert counts["EXCLUDED _quarantine"] == 1, counts

    # The exclusion is a named set, not a hardcoded string buried in a filter.
    everything, _ = read_shard_arms(root, exclude_dirs=frozenset())
    assert sorted(a.wins for a in everything) == [7, 99]


def test_the_rederivation_refusal_does_not_exit_zero(tmp_path: Path, capsys) -> None:
    """MUTATION (review R1): ``return 0`` at the end of ``_rederive``.

    The refusal prints, so a human reading the terminal is safe either way. The
    caller this protects is the one that captures the output: a wrapper testing
    the status would be told the constants are good in exactly the case where
    the tool declined to emit any. ⚑ An exit code is an OR over axes -- assert
    the AXIS STATE, and "the sweep resolved" is the axis here.

    7 is deliberately outside the verdict range: this mode judges no window, so
    mapping non-zero onto "the gate says kill" would be wrong twice over.
    """
    from scripts import gate_shadow_readout

    cadence, lag, n = 600.0, 180.0, 12
    csv_path = tmp_path / "progress.csv"
    rows = ["timestamp,time_this_iter_s,training_iteration"]
    rows += [
        f"{1_000_000.0 + (i + 1) * cadence},{cadence},{i + 1}" for i in range(n)
    ]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    root = tmp_path / "processed" / "_compacted"
    root.mkdir(parents=True)
    k = 0
    for i in range(n):
        start = 1_000_000.0 + i * cadence
        for t in range(0, int(cadence), 30):
            prev = t < lag
            d = root / f"s{k}.zarr"
            k += 1
            d.mkdir()
            (d / ".zattrs").write_text(json.dumps({
                "generated_at_unix": start + t + (0.0 if prev else 1.0),
                "model_step": i if prev else i + 1,
                "model_sha256": "p" if prev else "c",
                "wins": 5, "draws": 1, "losses": 4,
            }), encoding="utf-8")

    code = gate_shadow_readout._rederive(csv_path, tmp_path / "processed")
    assert code == gate_shadow_readout._REDERIVE_UNRESOLVED, code
    # ⚑ A BAND ENDPOINT IS ONLY AS GOOD AS ITS n (review B2). The per-shift
    # usable-bin counts are PRINTED, so a reader can see how many bins each end
    # of the band was computed from before banking either.
    printed = capsys.readouterr().out
    assert "usable bins per shift" in printed
    for shift in (-0.5, -0.25, 0.0, 0.25, 0.5):
        assert f"{shift:+.2f}:" in printed, printed
    assert "DEGENERATE" in printed, "the flag must be legible in the output"
    assert code != 0
    # ...and it is not confusable with a verdict or with a missing axis.
    assert code not in {
        READOUT_EXIT_PROMOTE, READOUT_EXIT_HOLD, READOUT_EXIT_KILL,
        READOUT_EXIT_CONFOUND_UNMEASURED, READOUT_EXIT_IDENTITY_UNEVALUATED,
        gate_shadow_readout._NOT_RUN, gate_shadow_readout._NO_FILE,
    }
    # The documented table names it, so an operator can look 7 up.
    assert "rederive unresolved (7)" in (gate_shadow_readout.__doc__ or "")

    # ...and the code is CONDITIONAL, not a constant refusal: a fleet whose
    # split survives the sweep still exits 0. Without this half, hardcoding 7
    # would pass -- an instrument that always refuses is as useless as one that
    # never does, and only this direction gets noticed by nobody.
    calm_root = tmp_path / "calm" / "_compacted"
    calm_root.mkdir(parents=True)
    for i in range(n):
        mid = 1_000_000.0 + i * cadence + cadence / 2.0
        for j, (step, sha) in enumerate(((i + 1, "c"), (i, "p"))):
            d = calm_root / f"s{i}_{j}.zarr"
            d.mkdir()
            (d / ".zattrs").write_text(json.dumps({
                "generated_at_unix": mid + j,
                "model_step": step, "model_sha256": sha,
                "wins": 5, "draws": 1, "losses": 4,
            }), encoding="utf-8")
    assert gate_shadow_readout._rederive(csv_path, tmp_path / "calm") == 0


def test_a_skipped_pooled_identity_cannot_exit_zero() -> None:
    """MUTATION (review R3): let ``readout_exit_code`` ignore the axis states.

    Same OR-over-axes trap this PR closes for ``confound``, one axis over, on
    the leg the module itself calls "THE ONE LEG WITH NO STATISTICS IN IT". A
    csv with no ``pid_curriculum_*`` -- a file rotated from an earlier report
    schema, which this repo produces every few days -- skips the exact-integer
    identity, and exit 0 would say the instrument was checked.
    """
    rows = [(197, 38, 44.0 * (1 if i % 2 else -1), 721.0, float("nan"),
             5.0 * ((i % 5) - 2)) for i in range(40)]
    r = shadow_readout_verdict(rows, last_n=40)
    assert r.verdict == READOUT_PROMOTE
    assert r.confound_is_measured is True, "the OTHER mandatory axis is fine"
    assert [leg.state for leg in r.legs
            if leg.name == "anchored_games_vs_pooled"] == [LEG_SKIPPED]
    assert readout_exit_code(r) == READOUT_EXIT_IDENTITY_UNEVALUATED

    # Both axes unmeasured: the identity is the more fundamental fact and is
    # reported first. Pinned so the precedence is a decision, not an accident.
    both = [(197, 38, 44.0 * (1 if i % 2 else -1), 721.0) for i in range(40)]
    assert readout_exit_code(shadow_readout_verdict(both, last_n=40)) == (
        READOUT_EXIT_IDENTITY_UNEVALUATED)

    # A SKIPPED cadence axis is explicitly EXEMPT, and the docstring says why:
    # the attribution axis is still evaluated in that case, against the raw
    # reference share, and says so on a failure.
    no_cadence = [(197, 38, 44.0 * (1 if i % 2 else -1), float("nan"), 235.0,
                   5.0 * ((i % 5) - 2)) for i in range(40)]
    r_nc = shadow_readout_verdict(no_cadence, last_n=40)
    assert [leg.state for leg in r_nc.legs if leg.name == "cadence"] == [LEG_SKIPPED]
    assert readout_exit_code(r_nc) == READOUT_EXIT_PROMOTE
    assert "cadence" in (readout_exit_code.__doc__ or "")


def test_the_first_publish_of_a_process_does_not_re_anchor(tmp_path: Path) -> None:
    """MUTATION (review R4): ``if d is None: return True``.

    That branch is only reachable when an anchor ALREADY EXISTS -- the
    ``not is_file()`` branch above it covers bootstrap -- so returning True
    meant the first unheld publish of every process rewrote the anchor from the
    current (possibly demoted, possibly just-refused-as-stale) export and
    re-stamped it at the current iteration. ``train.sh`` auto-resume restarts
    on every crash, so that is G3-8 restored once per restart; and because
    ``note_anchor_refreshed`` clears the stamp flag and zeroes the age, the AGE
    guard could never observe staleness that predates a restart.
    """
    anchor = tmp_path / "gate_promoted_model.pt"
    ctrl = GateHoldController(gate=_gate(mode=MODE_ENFORCE),
                              promoted_model_path=anchor)
    # Bootstrap: no file yet, so the first publish creates it.
    assert ctrl.anchor_refresh_is_due() is True
    anchor.write_bytes(b"an-anchor")
    write_anchor_stamp(anchor, iteration=100, trainer_step=1,
                       model_sha256="abc", trial_id="t0")

    # A fresh process: an anchor on disk and no verdict yet.
    revived = GateHoldController.create(
        _gate(mode=MODE_ENFORCE), durable_dir=tmp_path, current_iteration=101,
    )
    assert revived.last_decision is None
    assert revived.anchor_refresh_is_due() is False, (
        "the first publish after a restart must not re-anchor to the current "
        "export"
    )
    # ...and the age guard can still see staleness that predates the restart.
    stale = GateHoldController.create(
        _gate(mode=MODE_ENFORCE), durable_dir=tmp_path, current_iteration=200,
    )
    assert stale.anchor_is_trustworthy is False

    # The next genuine promote is what re-anchors, and nothing else does.
    revived.on_decision(GateDecision(
        decision=DECISION_NOT_RUN, reason="insufficient_iters",
        mode=MODE_ENFORCE))
    assert revived.anchor_refresh_is_due() is False
    revived.on_decision(GateDecision(
        decision=DECISION_PROMOTE, reason="promote_no_regression",
        mode=MODE_ENFORCE, games_cur=197, games_prev=38))
    assert revived.anchor_refresh_is_due() is True


# --------------------------------------------------------------------------
# audit L3: the selfplay phase must read the shard list through the buffer's
#           LOCKED accessor, not the raw deque.
# --------------------------------------------------------------------------
def test_the_selfplay_phase_never_touches_the_raw_shard_deque(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED on main, which iterates ``buf._shard_paths`` directly, twice.

    ``_shard_paths`` is guarded by ``DiskReplayBuffer._prefetch_lock`` and the
    buffer's own prefetch thread iterates it under that lock; ``_snapshot_shards()``
    is the accessor that exists for exactly this read. Unlocked iteration is
    safe today only because the sole mutator happens to run on this same
    thread -- the A18 shape. This buffer makes the unlocked read OBSERVABLE
    rather than asserting on source text: the attribute raises, so the test
    fails for the reason it exists.
    """
    import chess_anti_engine.tune.trainable_phases as phases

    monkeypatch.setattr(phases, "_publish_iteration_model", lambda **kw: "sha-pub")
    monkeypatch.setattr(phases, "_ensure_distributed_workers", lambda **kw: [])
    monkeypatch.setattr(phases, "_ensure_inference_broker", lambda **kw: None)
    monkeypatch.setattr(
        phases, "_ingest_distributed_selfplay",
        lambda **kw: {**_empty_ingest_summary(), "matching_games": 12},
    )
  # The sibling export walks the *new* shard paths; nothing about this test
  # concerns it, and it would try to read files that do not exist.
    monkeypatch.setattr(phases, "_export_selfplay_shards_for_siblings", lambda **kw: None)

    class _LockedOnlyBuf:
        capacity = 1000

        def __init__(self) -> None:
            self._paths = [tmp_path / "shard_0000.zarr"]
            self.snapshot_calls = 0

        @property
        def _shard_paths(self) -> list[Path]:
            raise AssertionError(
                "unlocked read of the raw _shard_paths deque (audit L3); "
                "use _snapshot_shards()"
            )

        def _snapshot_shards(self) -> list[Path]:
            self.snapshot_calls += 1
            return list(self._paths)

        def flush(self) -> None: ...
        def enforce_window(self) -> None: ...
        def __len__(self) -> int: return 500

    buf = _LockedOnlyBuf()
    sp, _sha, _win = phases._run_selfplay_phase(
        tc=TrialConfig(batch_size=512), config={}, trainer=_StubTrainer(),
        model_cfg=None, buf=buf, holdout_buf=[], holdout_frozen=True,
        rng=np.random.default_rng(0),
        distributed_dirs={"publish_dir": tmp_path, "inbox_dir": tmp_path,
                          "processed_dir": tmp_path},
        distributed_server_root=tmp_path, distributed_worker_procs=[],
        broker_proc_box=[None], prev_published_model_sha="sha-prev",
        ds=DifficultyState(wdl_regret=0.089, sf_nodes=50_000), sims=64,
        iteration_idx=7, iteration_zero_based=7, trial_id="t0",
        trial_dir=tmp_path, selfplay_shards_dir=tmp_path,
        replay_shard_dir=tmp_path, current_window=1000,
        in_salvage_startup_grace=False, hold=None,
    )

    assert sp.should_retry is False, "the phase must run past the retry early-out"
    assert buf.snapshot_calls == 2, (
        "both shard-list reads (before ingest, and the new-shard diff after) "
        f"must go through the locked accessor; saw {buf.snapshot_calls}"
    )
