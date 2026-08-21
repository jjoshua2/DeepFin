"""The live sf_p0 target blend: math, config plumbing, and the wiring proof.

The blend math tests are ported from the dose ladder's
``tests/test_retarget_retrain.py`` (prereg 2026-08-19); the wiring tests drive
the REAL ``_run_training_and_gating`` body with module collaborators
monkeypatched, so the mutant "call ``train_steps(buf)`` instead of the wrapped
buffer" fails here — a helper-only test would survive exactly that mutant.
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
    return {"policy_target": t0, "sf_p0_policy_target": q, "has_sf_p0": has}


# ── blend math (ported from the ladder; the semantics must not drift) ────────


def test_blend_is_the_convex_combination_on_eligible_rows_only() -> None:
    arrs = _sfp0_arrs()
    buf = SfP0BlendBuffer(_FakeSfP0Buffer(arrs), 0.7)
    out = buf.sample_batch_arrays(4)
    t0, q = arrs["policy_target"], arrs["sf_p0_policy_target"]
    exp = t0.copy()
    exp[[0, 2]] = 0.3 * t0[[0, 2]] + 0.7 * q[[0, 2]]
    np.testing.assert_allclose(out["policy_target"], exp, rtol=1e-6)
    # non-eligible rows BITWISE untouched — a blend leaking onto them would
    # train curriculum rows toward a teacher that does not exist for them
    assert (out["policy_target"][[1, 3]] == t0[[1, 3]]).all()
    # convexity of normalized inputs: still a distribution, unrenormalized
    np.testing.assert_allclose(out["policy_target"].sum(axis=1), 1.0, rtol=1e-5)


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
    as a null — the exact 'value accepted then ignored' shape."""
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
    def __init__(self) -> None:
        self.seen_bufs: list = []

    def train_steps(self, buf, *, batch_size: int, steps: int):
        assert batch_size > 0
        assert steps > 0
        self.seen_bufs.append(buf)
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
    here; so does dropping the wrap entirely. The log line is asserted from
    the same captured object — announced from the consumer's own parameter."""
    tc = TrialConfig.from_dict({"batch_size": 4, "sf_p0_blend_alpha": 0.7})
    trainer = _CapturingTrainer()
    buf = _FakeSfP0Buffer(_sfp0_arrs())
    _run_phase(monkeypatch, tmp_path, tc=tc, trainer=trainer, buf=buf)
    (seen,) = trainer.seen_bufs
    assert isinstance(seen, SfP0BlendBuffer)
    assert seen._inner is buf
    assert seen.alpha == 0.7
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
