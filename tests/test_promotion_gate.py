"""Tests for the anchored promotion gate.

Every test here is written to fail if the gate stops TAKING EFFECT, not merely
if its arithmetic drifts. Each one names the mutation it catches; the PR body
records the mutation run that proved each name honest.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from chess_anti_engine.model import ModelConfig
from chess_anti_engine.moves.encode import POLICY_SIZE
from chess_anti_engine.replay import ArrayReplayBuffer
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import LOCAL_SHARD_SUFFIX, save_local_shard_arrays
from chess_anti_engine.tune.distributed_runtime import (
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
    AnchoredSample,
    GateConfig,
    GateDecision,
    PromotionGate,
    gate_config_from_dict,
    gate_metrics,
)
from chess_anti_engine.tune.trainable_phases import (
    _publish_iteration_model,
    _run_training_and_gating,
    resolve_gate_hold_path,
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


def _gate(mode: str = MODE_ENFORCE, **kw) -> PromotionGate:
    cfg = GateConfig(mode=mode, window_iters=24, min_iters=8,
                     min_games_per_side=40, demote_delta_elo=-50.0,
                     alpha=0.05, max_hold_iters=12, **kw)
    return PromotionGate(cfg=cfg)


def _demoting_gate(promoted: Path) -> tuple[PromotionGate, GateDecision]:
    """A gate carrying a real DEMOTE verdict, with a fallback on disk."""
    g = _gate()
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
    d = _feed(_gate(), _window(0.0))
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
    near = _feed(_gate(), _window(-0.08))
    assert near.delta_elo < -50.0          # point estimate is past the line
    assert near.elo_hi > -50.0             # ...but the interval is not
    assert near.decision == DECISION_PROMOTE
    assert near.reason == "promote_no_regression"

    clear = _feed(_gate(), _window(-0.10))
    assert clear.elo_hi < -50.0
    assert clear.decision == DECISION_DEMOTE
    assert clear.reason == "demote_regression"


def test_ci_is_a_real_interval_not_a_point() -> None:
    """MUTATION: set the t-quantile to 0 (or se to 0). A zero-width interval
    claiming certainty from 8 iterations is audit row L11 in a new place."""
    d = _feed(_gate(), _window(-0.10))
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
    assert TrialConfig.gate_mode == MODE_OFF


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

    promote = _feed(_gate(), _window(0.0))
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
                  promoted: Path | None) -> Path:
    publish_dir = tmp_path / "trials" / "t0" / "publish"
    _publish_iteration_model(
        trainer=trainer,
        config={"selfplay_batch": 4, "max_plies": 40, "mcts": "gumbel",
                "fast_simulations": 8, "sf_move_nodes": 100, "sf_hash_mb": 16},
        model_cfg=_tiny_model_cfg(),
        server_root=tmp_path, publish_dir=publish_dir, trial_id="t0",
        iteration_idx=1, ds=DifficultyState(wdl_regret=0.089, sf_nodes=50_000),
        sims=4, reuse_existing_model_for_same_step=False,
        gate_hold_model_path=hold, gate_promoted_model_path=promoted,
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
    """The module docstring, the config comment and the ledger's kill rule are
    all denominated in the per-iteration standard error. If the Elo scale or the
    realized game shape drifts, every threshold derived from it is wrong.

    MEASURED over live iters 164-219 by binning 418 processed shard `.zattrs`
    by `generated_at_unix` against `progress.csv` timestamps (every window
    contains exactly two shas):

        n_cur ~= 197, n_prev ~= 38   (prev share 16.3%)
        per-iteration delta: mean -4.33 Elo, sd 45.56 Elo (n=53)

    NOT 95/143. `distributed_prev_model_max_fraction: 0.60` is a ceiling that
    never binds -- `distributed_stale_games` is 0 throughout, and a binding cap
    would have to discard prev shards into `stale_*`. Sizing the gate off the
    cap gives 31 Elo and is the wrong side of the split.
    """
    n_total, w, d = 237.7, 54.2, 129.0
    score = (w + 0.5 * d) / n_total
    var = (w * (1 - score) ** 2 + d * (0.5 - score) ** 2
           + (n_total - w - d) * score ** 2) / n_total
    n_cur, n_prev = 196.8, 38.3            # MEASURED, not derived from the cap
    se_score = math.sqrt(var) * math.sqrt(1.0 / n_cur + 1.0 / n_prev)
    se_elo = se_score * ELO_PER_SCORE_AT_HALF

    assert 38.0 < se_elo < 45.0, f"binomial se moved to {se_elo:.1f} Elo"
    # The observed between-iteration sd is larger still (45.56), i.e. there is
    # real variance beyond binomial. The shipped demote line must sit several
    # window-sigma below the measured null of -4.3 Elo, or the alarm fires on
    # noise -- which is the 2026-03 freeze relocated to the publish layer.
    observed_sd, null_mean = 45.56, -4.33
    window_se = observed_sd / math.sqrt(GateConfig().window_iters)
    sigmas = (null_mean - GateConfig().demote_delta_elo) / window_se
    assert 3.5 < sigmas < 6.0, (
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
        + inspect.getsource(phases.resolve_gate_hold_path),
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
