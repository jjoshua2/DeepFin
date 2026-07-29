"""Tests for the anchored promotion gate.

Every test here is written to fail if the gate stops TAKING EFFECT, not merely
if its arithmetic drifts. Each one names the mutation it catches; the PR body
records the mutation run that proved each name honest.
"""
from __future__ import annotations

import csv
import dataclasses
import json
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
from chess_anti_engine.tune.promotion_gate import (
    DECISION_DEMOTE,
    DECISION_NOT_RUN,
    DECISION_PROMOTE,
    ELO_PER_SCORE_AT_HALF,
    MODE_ENFORCE,
    MODE_OFF,
    MODE_SHADOW,
    OFFLINE,
    READOUT_KILL,
    READOUT_PROMOTE,
    AnchoredSample,
    GateConfig,
    GateDecision,
    GateHoldController,
    PromotionGate,
    _t_quantile,
    gate_config_from_dict,
    gate_metrics,
    resolve_gate_hold_path,
    shadow_readout_from_csv,
    shadow_readout_verdict,
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
    alpha: float = 0.05,
    max_hold_iters: int = 12,
) -> PromotionGate:
    return PromotionGate(cfg=GateConfig(
        mode=mode, window_iters=window_iters, min_iters=min_iters,
        min_games_per_side=min_games_per_side,
        demote_delta_elo=demote_delta_elo, alpha=alpha,
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
    gate that ran and passed."""
    m = gate_metrics(_gate(mode=MODE_OFF).decide())
    assert m["gate_decision"] == float(DECISION_NOT_RUN)
    assert m["gate_decision"] != float(DECISION_PROMOTE)
    assert m["gate_games_cur"] == 0.0
    assert m["gate_games_prev"] == 0.0
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
    3 iterations, or off 5 games a side, is noise wearing a decision's clothes."""
    short = _feed(_gate(), _window(-0.30)[:3])
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
def _shard(path: Path, *, sha: str, w: int, d: int, l: int) -> None:
    n = 2
    save_local_shard_arrays(
        path,
        arrs={
            "x": np.zeros((n, 146, 8, 8), dtype=np.float32),
            "policy_target": np.eye(POLICY_SIZE, dtype=np.float32)[np.zeros(n, dtype=int)],
            "wdl_target": np.zeros((n,), dtype=np.int8),
            "priority": np.ones((n,), dtype=np.float32),
            "has_policy": np.ones((n,), dtype=np.uint8),
        },
        meta={
            "model_sha256": sha, "games": w + d + l, "positions": n,
            "wins": w, "draws": d, "losses": l,
        },
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
    _shard(inbox / "w0" / f"a{LOCAL_SHARD_SUFFIX}", sha="cur-sha", w=7, d=2, l=1)
    _shard(inbox / "w0" / f"b{LOCAL_SHARD_SUFFIX}", sha="prev-sha", w=3, d=4, l=5)
    _shard(inbox / "w0" / f"c{LOCAL_SHARD_SUFFIX}", sha="ancient-sha", w=9, d=9, l=9)

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

    assert (summary["gate_cur_w"], summary["gate_cur_d"], summary["gate_cur_l"]) == (7, 2, 1)
    assert (summary["gate_prev_w"], summary["gate_prev_d"], summary["gate_prev_l"]) == (3, 4, 5)
    # Stale (neither published model) contributes to NEITHER side: it was played
    # by a net no longer under comparison.
    assert summary["gate_cur_w"] + summary["gate_prev_w"] == summary["matching_w"]
    assert summary["stale_games"] == 27


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

    # Release: the next unheld publish exports again and re-anchors.
    published = _publish_iter(tmp_path, trainer, hold=None, promoted=promoted)
    assert published.read_bytes() == b"trainer-export-2"
    assert promoted.read_bytes() == b"trainer-export-2"

    # N3 in the shape where it bites. In production the hold source IS the
    # anchor, so refreshing during a hold is a no-op and the guard is purely
    # defensive -- which is exactly why it needs a test that can tell: give the
    # hold a DIFFERENT source and the missing guard overwrites the anchor with
    # bytes that were never promoted.
    other = tmp_path / "some_other_export.pt"
    other.write_bytes(b"not-the-anchor")
    published = _publish_iter(tmp_path, trainer, hold=other, promoted=promoted)
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
        tc=tc, trainer=_StubTrainer(), buf=[], holdout_buf=[],
        config={}, model_cfg=None, device="cpu",
        ds=DifficultyState(wdl_regret=0.089, sf_nodes=50_000),
        sims=64, sp=sp,
        positions_ingested=0, imported_samples_this_iter=0,
        gate=gate, gate_match_idx=0,
        gate_state_path=tmp_path / "gate_state.json",
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
        buf=[], holdout_buf=[], config={}, model_cfg=None, device="cpu",
        ds=DifficultyState(wdl_regret=0.089, sf_nodes=50_000), sims=64, sp=sp,
        positions_ingested=0, imported_samples_this_iter=0,
        gate=gate, gate_match_idx=0,
        gate_state_path=tmp_path / "gate_state.json",
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
            leg.startswith(("prev_share", "mean_games_")) for leg in r.failed_legs
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


def test_every_deciding_leg_is_individually_load_bearing() -> None:
    """MUTATION: loosen ANY ONE leg's bound.

    The negative control above kills on the counts, but it moves several legs
    at once -- so widening a single tolerance escapes it while another leg
    happens to cover. That is the "leg that cannot fail" shape one level down,
    and it is how a rule rots: nobody removes a leg, they widen it.

    Each row below trips exactly ONE leg and leaves every other inside its
    bound, so each bound is asserted to be load-bearing on its own.
    """
    def series(cur: int, prev: int, mean: float, sd: float, *, n: int = 40,
               dead: int = 0) -> list[tuple[int, int, float]]:
        # deltas with exactly the requested mean and sample sd, by construction
        half = [mean - sd / math.sqrt(2), mean + sd / math.sqrt(2)]
        rows = [(cur, prev, half[i % 2]) for i in range(n)]
        return rows + [(0, prev, float("nan"))] * dead

    cases: list[tuple[str, int, int, float, float, int]] = [
        # leg,              cur, prev,  mean,   sd, dead
        ("mean_games_cur",  300,   38,   0.0, 44.0, 0),
        ("mean_games_prev", 250,   60,   0.0, 44.0, 0),
        ("prev_share",      140,   56,   0.0, 44.0, 0),
        ("sd_delta_elo",    197,   38,   0.0,  4.56, 0),
        ("|mean_delta_elo|", 197,  38,  30.0, 44.0, 0),
        ("usable_frac",     197,   38,   0.0, 44.0, 10),
    ]
    for leg, cur, prev, mean, sd, dead in cases:
        r = shadow_readout_verdict(series(cur, prev, mean, sd, dead=dead),
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
    """The ledger's worked example and its command must share a denominator.

    The example computed 51/53 = 0.96 (windows with a non-empty split) while
    the command computed 51/57 = 0.89 (all rows), against a KILL line of 0.85.
    Seven points apart with four points of margin. There is now one
    implementation, and the hold-shaped rows -- ``games_cur == 0`` -- stay in
    the denominator, because an iteration that produced no anchored sample is
    exactly what the leg is meant to count.
    """
    rows = _rows(_live_samples())
    assert len(rows) == 57
    r = shadow_readout_verdict(rows, last_n=_N_LIVE)
    assert r.n_rows == 57
    assert r.usable_frac == pytest.approx(51 / 57, abs=0.005)

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

    # A -50 Elo/iteration break lands ON TOP of the measured null.
    assert power(-45.0, 8, null - 50.0) == pytest.approx(0.100, abs=0.005)
    assert power(-45.0, 24, null - 50.0) == pytest.approx(0.239, abs=0.005)
    assert power(-25.0, 8, null - 50.0) == pytest.approx(0.483, abs=0.005)
    assert power(-25.0, 24, null - 50.0) == pytest.approx(0.925, abs=0.005)
    # The shipped line is the one that delivers the documented capability.
    assert GateConfig().demote_delta_elo == -25.0
    assert power(-25.0, GateConfig().window_iters, null - 50.0) > 0.90

    # The warm-start class (-6.7 Elo/iteration) stays out of reach at every
    # reachable K and line, under either reading of "break": the worst case
    # over K in {8, 12, 16, 24} x line in {-25, -45} is 0.32%, at K=8 / -25.
    worst = max(power(line, k, mu)
                for line in (-25.0, -45.0) for k in (8, 12, 16, 24)
                for mu in (null - 6.7, -6.7))
    assert worst == pytest.approx(0.0032, abs=0.0003), worst


# --------------------------------------------------------------------------
# 10. the loop's hold state machine, driven end to end
# --------------------------------------------------------------------------
def _controller(tmp_path: Path, *, mode: str = MODE_ENFORCE,
                hold_active: bool = False, anchor: bool = True) -> GateHoldController:
    g = _gate(mode=mode)
    g.hold_active = hold_active
    if anchor and mode != MODE_OFF:
        (tmp_path / "gate_promoted_model.pt").write_bytes(b"anchor")
    return GateHoldController.create(g, durable_dir=tmp_path)


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
        buf=[], holdout_buf=[], config={}, model_cfg=None, device="cpu",
        ds=DifficultyState(wdl_regret=0.089, sf_nodes=50_000), sims=64, sp=sp,
        positions_ingested=0, imported_samples_this_iter=0,
        gate=gate, gate_match_idx=0,
        gate_state_path=tmp_path / "gate_state.json",
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
