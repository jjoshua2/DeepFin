"""The live sf_p0 target blend: math, config plumbing, and the wiring proof.

The blend math tests are ported from the dose ladder's
``tests/test_retarget_retrain.py`` (prereg 2026-08-19); the wiring tests drive
the REAL ``_run_training_and_gating`` body with module collaborators
monkeypatched, so the mutant "call ``train_steps(buf)`` instead of the wrapped
buffer" fails here — a helper-only test would survive exactly that mutant.

⚑ Fixtures are float16 because production storage is float16 for BOTH
``policy_target`` and ``sf_p0_policy_target`` — a float32 fixture's tight
row-sum assertion does not even hold at the dtype the wrapper actually
receives (review F3).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml

import chess_anti_engine.tune.trainable_phases as tp
from chess_anti_engine.replay.sfp0_blend import SfP0BlendBuffer
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults


class _FakeSfP0Buffer:
    """Minimal sample_batch_arrays stand-in with controllable columns."""

    def __init__(self, arrs: dict, n: int = 10_000):
        self._arrs = arrs
        self._n = n
        self.rng = np.random.default_rng(0)

    def sample_batch_arrays(self, _batch_size: int, **_kw) -> dict:
        return {k: v.copy() for k, v in self._arrs.items()}

    def __len__(self) -> int:
        return self._n


def _sfp0_arrs(n: int = 4, width: int = 8) -> dict:
    rng = np.random.default_rng(0)
    t0 = rng.random((n, width)).astype(np.float32)
    t0 /= t0.sum(axis=1, keepdims=True)
    q = rng.random((n, width)).astype(np.float32)
    q /= q.sum(axis=1, keepdims=True)
    has = np.array([1, 0, 1, 0][:n], dtype=np.int8)
    # Production storage dtype for BOTH sources (replay/shard.py): float16.
    return {
        "policy_target": t0.astype(np.float16),
        "sf_p0_policy_target": q.astype(np.float16),
        "has_sf_p0": has,
    }


# ── blend math (ported from the ladder; the semantics must not drift) ────────


def test_blend_is_the_convex_combination_on_eligible_rows_only() -> None:
    arrs = _sfp0_arrs()
    buf = SfP0BlendBuffer(_FakeSfP0Buffer(arrs), 0.7)
    out = buf.sample_batch_arrays(4)
    t0, q = arrs["policy_target"], arrs["sf_p0_policy_target"]
    exp = t0.astype(np.float32).copy()
    exp[[0, 2]] = 0.3 * t0[[0, 2]].astype(np.float32) + 0.7 * q[[0, 2]].astype(np.float32)
    assert out["policy_target"].dtype == np.float16
    np.testing.assert_allclose(
        out["policy_target"].astype(np.float32), exp, rtol=2e-3, atol=1e-4,
    )
    # non-eligible rows BITWISE untouched — a blend leaking onto them would
    # train curriculum rows toward a teacher that does not exist for them
    assert (out["policy_target"][[1, 3]] == t0[[1, 3]]).all()
    # convexity of normalized inputs: still ~a distribution, unrenormalized;
    # fp16 storage drift means "== 1.0" only holds at fp16-appropriate atol
    np.testing.assert_allclose(
        out["policy_target"].sum(axis=1, dtype=np.float64), 1.0, atol=2e-3,
    )


def test_blend_counts_realized_coverage_from_its_own_masks() -> None:
    """The log line's f must come from the wrapper's OWN counters, never
    spliced from the live run's has_sf_p0_frac (different population)."""
    buf = SfP0BlendBuffer(_FakeSfP0Buffer(_sfp0_arrs()), 0.25)
    buf.sample_batch_arrays(4)
    buf.sample_batch_arrays(4)
    assert buf.total_rows == 8
    assert buf.blended_rows == 4


def test_blend_refuses_a_pool_without_the_teacher_columns() -> None:
    """Missing columns must be a REFUSAL, not a silent pure-t0 run that reads
    as a null — the exact 'value accepted then ignored' shape. (The realistic
    outage — columns present, flags all zero — is gated per iteration in
    trainable_phases._finish_sfp0_blend; tested below.)"""
    arrs = _sfp0_arrs()
    del arrs["sf_p0_policy_target"]
    buf = SfP0BlendBuffer(_FakeSfP0Buffer(arrs), 0.5)
    with pytest.raises(RuntimeError, match="predates the sf_p0 teacher"):
        buf.sample_batch_arrays(4)


def test_blend_alpha_zero_and_out_of_band_are_refused_by_the_class() -> None:
    """a=0 must run UNWRAPPED (bitwise by construction); accepting it here
    would put OFF through a copy+mask path that only coincidentally matches."""
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="activation needs"):
            SfP0BlendBuffer(_FakeSfP0Buffer(_sfp0_arrs()), bad)


def test_blend_alpha_one_serves_q_exactly_on_eligible_rows() -> None:
    arrs = _sfp0_arrs()
    buf = SfP0BlendBuffer(_FakeSfP0Buffer(arrs), 1.0)
    out = buf.sample_batch_arrays(4)
    np.testing.assert_allclose(
        out["policy_target"][[0, 2]], arrs["sf_p0_policy_target"][[0, 2]], rtol=1e-6,
    )


def test_blend_does_NOT_renormalize_a_within_tolerance_row(
) -> None:
    """⚑ Pins the no-renormalization stance with a mutant that can fail
    (review F2: a renormalizing mutant passed all prior tests, because every
    fixture row was exactly normalized and renorm was a no-op on them).

    The eligible q rows here sum to ~1.005 — inside the corruption tolerance,
    so the wrapper must pass them through. The blended eligible row then sums
    to (1-a)*sum_t0 + a*sum_q != 1.0; a renormalizing mutant forces exactly
    1.0 and fails the second assertion."""
    arrs = _sfp0_arrs()
    q = arrs["sf_p0_policy_target"].astype(np.float32)
    q[[0, 2]] *= 1.005
    arrs["sf_p0_policy_target"] = q.astype(np.float16)
    a = 0.7
    buf = SfP0BlendBuffer(_FakeSfP0Buffer(arrs), a)
    out = buf.sample_batch_arrays(4)
    sum_t0 = arrs["policy_target"][[0, 2]].sum(axis=1, dtype=np.float64)
    sum_q = arrs["sf_p0_policy_target"][[0, 2]].sum(axis=1, dtype=np.float64)
    got = out["policy_target"][[0, 2]].sum(axis=1, dtype=np.float64)
    expected = (1.0 - a) * sum_t0 + a * sum_q
    np.testing.assert_allclose(got, expected, atol=1e-3)
    assert (np.abs(got - 1.0) > 2e-3).all(), (
        "blended sums landed back at 1.0 — the blend renormalized"
    )


def test_blend_refuses_a_corrupt_active_row_before_blending() -> None:
    """Review D: soft_cross_entropy renormalizes downstream, so a corrupt
    stored sum this wrapper lets through is silently repaired one frame
    later. Active rows of BOTH sources are validated; a corrupt row on an
    INELIGIBLE row is out of scope (nothing consumes it as a teacher)."""
    arrs = _sfp0_arrs()
    q = arrs["sf_p0_policy_target"].astype(np.float32)
    q[0] *= 0.5
    arrs["sf_p0_policy_target"] = q.astype(np.float16)
    buf = SfP0BlendBuffer(_FakeSfP0Buffer(arrs), 0.7)
    with pytest.raises(RuntimeError, match="not normalized"):
        buf.sample_batch_arrays(4)

    # the same corruption on an ineligible row does not refuse
    arrs2 = _sfp0_arrs()
    t = arrs2["policy_target"].astype(np.float32)
    t[1] *= 0.5
    arrs2["policy_target"] = t.astype(np.float16)
    out = SfP0BlendBuffer(_FakeSfP0Buffer(arrs2), 0.7).sample_batch_arrays(4)
    assert out["policy_target"].shape == (4, 8)


def test_a_half_built_wrapper_does_not_recurse() -> None:
    """Review F9: attribute reads on a partially-constructed instance
    (constructor raised, __new__ without __init__, copy-protocol probes) used
    to recurse forever through __getattr__ looking for _inner."""
    half_built = SfP0BlendBuffer.__new__(SfP0BlendBuffer)
    assert not hasattr(half_built, "capacity")
    with pytest.raises(AttributeError):
        _ = half_built._inner


# ── config plumbing: schema round-trip and the [0, 1] validator ──────────────


def test_yaml_key_survives_flatten_and_from_dict() -> None:
    cfg = yaml.safe_load(io.StringIO("tune:\n  sf_p0_blend_alpha: 0.7\n"))
    tc = TrialConfig.from_dict(flatten_run_config_defaults(cfg))
    assert tc.sf_p0_blend_alpha == 0.7


def test_default_is_off() -> None:
    assert TrialConfig.from_dict({}).sf_p0_blend_alpha == 0.0


def test_out_of_band_and_non_numeric_values_are_refused() -> None:
    """Category (b) by design: a target-affecting knob validates, accepting
    that a bad live edit kills the trial loudly instead of training on a
    nonsense target silently."""
    with pytest.raises(ValueError, match="sf_p0_blend_alpha"):
        TrialConfig.from_dict({"sf_p0_blend_alpha": 1.5})
    with pytest.raises(ValueError, match=r"could not convert|sf_p0_blend_alpha"):
        TrialConfig.from_dict({"sf_p0_blend_alpha": "x"})


# ── the wiring proof, through the REAL _run_training_and_gating body ─────────


class _CapturingTrainer:
    """Captures the buffer train_steps received AND samples it once, so the
    wiring tests distinguish "trainer samples the wrapped buffer" from
    "trainer merely receives it" (review G)."""

    w_sf_own = 0.0
    rebuild_sf_targets = False

    def __init__(self) -> None:
        self.seen_bufs: list = []
        self.sampled: list[dict] = []

    def train_steps(self, buf, *, batch_size: int, steps: int):
        assert batch_size > 0
        assert steps > 0
        self.seen_bufs.append(buf)
        self.sampled.append(buf.sample_batch_arrays(batch_size))
        return "metrics-sentinel"


def _run_phase(monkeypatch, tmp_path: Path, *, tc: TrialConfig, trainer, buf):
    monkeypatch.setattr(tp, "_run_net_gating", lambda *a, **k: (None, 0))
    monkeypatch.setattr(
        tp, "_compute_train_step_budget",
        lambda **k: {"steps": 2, "target_sample_budget": 0, "window_target_samples": 0},
    )
    monkeypatch.setattr(tp, "_apply_salvage_step_caps", lambda steps, **k: steps)
    monkeypatch.setattr(tp, "_run_holdout_evaluation", lambda **k: (None, None))
    return tp._run_training_and_gating(
        tc=tc, trainer=trainer, buf=buf, holdout_buf=object(),
        holdout_frozen=False, config={}, model_cfg=object(), device="cpu",
        ds=cast(Any, object()), sims=1, sp=cast(Any, object()),
        positions_ingested=0,
        imported_samples_this_iter=0, gate=cast(Any, object()), gate_match_idx=0,
        gate_state_path=tmp_path / "gate.json", gate_hold=None,
        distributed_server_root=tmp_path, iteration_idx=7,
        iteration_zero_based=6, trial_id="t", restore=cast(Any, object()),
        async_test_eval=None,
    )


def test_alpha_zero_passes_the_raw_buffer_identity_object(monkeypatch, tmp_path) -> None:
    """OFF is the UNWRAPPED path: train_steps must receive the very same
    object, not an equivalent wrapper — bitwise identity by construction."""
    tc = TrialConfig.from_dict({"batch_size": 4})
    trainer = _CapturingTrainer()
    buf = _FakeSfP0Buffer(_sfp0_arrs())
    result = _run_phase(monkeypatch, tmp_path, tc=tc, trainer=trainer, buf=buf)
    assert trainer.seen_bufs == [buf]
    assert trainer.seen_bufs[0] is buf
    assert result.metrics == "metrics-sentinel"


def test_positive_alpha_wraps_the_buffer_train_steps_consumes(
    monkeypatch, tmp_path, capsys,
) -> None:
    """THE WIRING TEST. Mutant 'train_steps(buf) instead of train_buf' fails
    here; so does dropping the wrap entirely. The trainer SAMPLES the buffer
    it received, and the sampled content must be the blended target — not
    merely a wrapper handed over unconsumed. The log line is asserted from
    the same captured object — announced from the consumer's own parameter."""
    tc = TrialConfig.from_dict({"batch_size": 4, "sf_p0_blend_alpha": 0.7})
    trainer = _CapturingTrainer()
    arrs = _sfp0_arrs()
    buf = _FakeSfP0Buffer(arrs)
    _run_phase(monkeypatch, tmp_path, tc=tc, trainer=trainer, buf=buf)
    (seen,) = trainer.seen_bufs
    assert isinstance(seen, SfP0BlendBuffer)
    assert seen._inner is buf
    assert seen.alpha == 0.7
    (sampled,) = trainer.sampled
    t0, q = arrs["policy_target"], arrs["sf_p0_policy_target"]
    exp = 0.3 * t0[[0, 2]].astype(np.float32) + 0.7 * q[[0, 2]].astype(np.float32)
    np.testing.assert_allclose(
        sampled["policy_target"][[0, 2]].astype(np.float32), exp, rtol=2e-3, atol=1e-4,
    )
    assert not np.array_equal(sampled["policy_target"], t0), (
        "the trainer sampled the RAW target — the blend never reached it"
    )
    out = capsys.readouterr().out
    assert "[trial] sf_p0_blend: alpha=0.7" in out
    assert "(iteration 7)" in out


def test_a_reloaded_alpha_reaches_the_next_iterations_wrapper(
    monkeypatch, tmp_path,
) -> None:
    """The mid-run reload contract: the wrapper is rebuilt per iteration from
    that iteration's freshly reloaded tc, so an alpha edit lands on the very
    next training phase — in EITHER direction, 0 <-> positive included.
    Mutant 'memoize the first wrapper' fails here."""
    trainer = _CapturingTrainer()
    buf = _FakeSfP0Buffer(_sfp0_arrs())
    for alpha in (0.25, 0.5, 0.0):
        tc = TrialConfig.from_dict({"batch_size": 4, "sf_p0_blend_alpha": alpha})
        _run_phase(monkeypatch, tmp_path, tc=tc, trainer=trainer, buf=buf)
    a_first, a_second, a_off = trainer.seen_bufs
    assert isinstance(a_first, SfP0BlendBuffer)
    assert a_first.alpha == 0.25
    assert isinstance(a_second, SfP0BlendBuffer)
    assert a_second.alpha == 0.5
    assert a_second is not a_first
    assert a_off is buf


def test_an_all_zero_flag_iteration_raises_the_outage_gate(
    monkeypatch, tmp_path,
) -> None:
    """⚑ Review F1, the realistic outage: the codec always materializes
    has_sf_p0 (zeros) and the gatherer zero-fills over the union of keys, so
    a real sf_p0 outage arrives as columns-present-all-zero — the wrapper's
    missing-column refusal cannot see it. An iteration that trained rows and
    blended none must RAISE, not print a line someone has to read."""
    arrs = _sfp0_arrs()
    arrs["has_sf_p0"] = np.zeros(4, dtype=np.int8)
    tc = TrialConfig.from_dict({"batch_size": 4, "sf_p0_blend_alpha": 0.7})
    trainer = _CapturingTrainer()
    with pytest.raises(RuntimeError, match="blended ZERO"):
        _run_phase(
            monkeypatch, tmp_path, tc=tc, trainer=trainer,
            buf=_FakeSfP0Buffer(arrs),
        )
    # the same pool at alpha=0 is NOT an outage — nothing asked for a teacher
    tc0 = TrialConfig.from_dict({"batch_size": 4})
    _run_phase(
        monkeypatch, tmp_path, tc=tc0, trainer=_CapturingTrainer(),
        buf=_FakeSfP0Buffer(arrs),
    )


def test_w_sf_own_and_the_blend_together_are_refused(monkeypatch, tmp_path) -> None:
    """Review C: the additive CE term is separately masked_mean-normalized,
    so combining it with the blend double-doses eligible rows — the exact
    row-upweighting confound the target blend exists to avoid. The check
    reads the TRAINER'S OWN attribute (the live-synced value), so a live
    w_sf_own edit is caught the same iteration it lands."""
    tc = TrialConfig.from_dict({"batch_size": 4, "sf_p0_blend_alpha": 0.7})
    trainer = _CapturingTrainer()
    trainer.w_sf_own = 0.1
    with pytest.raises(RuntimeError, match="double-dose"):
        _run_phase(
            monkeypatch, tmp_path, tc=tc, trainer=trainer,
            buf=_FakeSfP0Buffer(_sfp0_arrs()),
        )
    assert trainer.seen_bufs == [], "training ran despite the refusal"


def test_rebuild_sf_targets_and_the_blend_together_are_refused(
    monkeypatch, tmp_path,
) -> None:
    """Review B: the rebuild runs AFTER the blend and clears has_sf_p0 but
    cannot restore the already-blended policy_target — a stale teacher would
    train while coverage reports it masked."""
    tc = TrialConfig.from_dict({"batch_size": 4, "sf_p0_blend_alpha": 0.7})
    trainer = _CapturingTrainer()
    trainer.rebuild_sf_targets = True
    with pytest.raises(RuntimeError, match="rebuild_sf_targets"):
        _run_phase(
            monkeypatch, tmp_path, tc=tc, trainer=trainer,
            buf=_FakeSfP0Buffer(_sfp0_arrs()),
        )
    assert trainer.seen_bufs == [], "training ran despite the refusal"
