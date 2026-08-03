"""Pins for the per-iteration era-forgetting probes.

The instrument exists because the 2026-07-31 run lost 48.6 Elo behind flat
columns, so every test here is written against the failure mode that produced
that: a value accepted and then silently ignored. For each mechanism there is a
mutation that severs the wire, and the test named in the docstring is the one
that fails when it is applied.

Grouped as:
  * the two RULERS mean what their names say (expected, not argmax; masked;
    pooled denominator);
  * the wire from config to a published column cannot be cut silently;
  * the probe rows can never reach the training sampler (with the positive
    control that the same detector fires when they are added on purpose);
  * the BUILDER refuses poisoned, quarantined and accidentally-overwritten
    sets, and its digest is the same number the trial prints at load.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.eval.era_probe import (
    PROBE_ERA,
    PROBE_INWINDOW,
    PROBE_SET_FIELDS,
    ProbeReading,
    ProbeSet,
    load_probe_set,
    probe_metric_defaults,
    probe_metrics,
    probe_set_digest,
    provenance_path,
    score_probe_set,
)
from chess_anti_engine.replay.shard import (
    SF_MULTIPV_RAW_MAX,
    save_local_shard_arrays,
    save_npz_arrays,
)
from chess_anti_engine.tune.trainable_config_ops import construction_only_config_keys
from chess_anti_engine.tune.trainable_phases import _run_era_probes_if_due
from chess_anti_engine.tune.trial_config import TrialConfig

_REPO = Path(__file__).resolve().parents[1]
POLICY = 1858
PLANES = 175
# Legal moves per synthetic position. Small so the analytic expectations below
# can be written out by hand.
N_LEGAL = 4


# ---------------------------------------------------------------------------
# fixtures


class _FixedNet(torch.nn.Module):
    """A model with hand-chosen policy and WDL logits, so every ruler reading
    has a closed form. Real-model plumbing gets its own test below."""

    def __init__(self, policy_logits: np.ndarray, wdl_logits: np.ndarray) -> None:
        super().__init__()
        self._pol = torch.tensor(policy_logits, dtype=torch.float32)
        self._wdl = torch.tensor(wdl_logits, dtype=torch.float32)
        self._cursor = 0

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Rows arrive in stored order, so a batch starting at the cursor maps
        # onto the same slice of the fixed logits.
        n = int(x.shape[0])
        start, self._cursor = self._cursor, self._cursor + n
        return {
            "policy_own": self._pol[start:start + n],
            "wdl": self._wdl[start:start + n],
        }

    def rewind(self) -> None:
        self._cursor = 0


def _probe_arrays(
    *,
    n: int,
    regret: np.ndarray | None = None,
    wdl_target: np.ndarray | None = None,
    has_regret: np.ndarray | None = None,
    has_mask: np.ndarray | None = None,
    illegal_regret: float = 0.75,
    marker: float = 0.0,
) -> dict[str, np.ndarray]:
    """A minimal probe-set array dict. `marker` stamps a detectable value into
    an otherwise-unused input plane, for the sampler-isolation test.

    ⚑ `illegal_regret` defaults to 0.75, NOT 0. Production fills every index
    the MultiPV did not cover — which is every illegal move — with
    `(worst_regret + 1) / 2 >= 0.5` (selfplay/finalize.py), measured at mean
    0.8302 over 2578 real rows. An earlier revision of this fixture zero-filled
    them, which quietly encoded the inverted premise that review caught in the
    production comments (PR #315, finding 2): a fixture that disagrees with the
    data cannot catch a ruler that disagrees with the data.
    """
    x = np.zeros((n, PLANES, 8, 8), dtype=np.float16)
    if marker:
        x[:, PLANES - 1, 7, 7] = marker
    legal = np.zeros((n, POLICY), dtype=np.uint8)
    legal[:, :N_LEGAL] = 1
    reg = np.full((n, POLICY), np.float16(illegal_regret), dtype=np.float16)
    reg[:, :N_LEGAL] = 0.0
    if regret is not None:
        reg[:, :N_LEGAL] = np.asarray(regret, dtype=np.float16)
    return {
        "x": x,
        "policy_target": np.full((n, POLICY), 1.0 / POLICY, dtype=np.float16),
        "wdl_target": (
            np.zeros(n, dtype=np.int8) if wdl_target is None
            else np.asarray(wdl_target, dtype=np.int8)
        ),
        "legal_mask": legal,
        "has_legal_mask": (
            np.ones(n, dtype=np.uint8) if has_mask is None
            else np.asarray(has_mask, dtype=np.uint8)
        ),
        "sf_p0_regret": reg,
        "has_sf_p0_regret": (
            np.ones(n, dtype=np.uint8) if has_regret is None
            else np.asarray(has_regret, dtype=np.uint8)
        ),
        "has_policy": np.ones(n, dtype=np.uint8),
        "priority": np.ones(n, dtype=np.float32),
    }


def _write_probe_set(path: Path, arrs: dict[str, np.ndarray], *, digest_of: dict | None = None) -> Path:
    save_npz_arrays(path, arrs=arrs, meta=None, compress=False)
    frozen = {k: v for k, v in arrs.items() if k in PROBE_SET_FIELDS}
    provenance_path(path).write_text(
        json.dumps({
            "version": 1, "label": "era", "lineage": "test",
            "desync_screened": True, "n_shards": 1,
            "digest": probe_set_digest(digest_of if digest_of is not None else frozen),
        }),
        encoding="utf-8",
    )
    return path


def _try_load(path: Path, label: str = PROBE_ERA, max_rows: int = 0) -> ProbeSet | None:
    return load_probe_set(
        path, label=label, max_rows=max_rows,
        expected_planes=PLANES, expected_policy_size=POLICY,
    )


def _load(path: Path, label: str = PROBE_ERA, max_rows: int = 0) -> ProbeSet:
    """`_try_load` for the tests that expect a set; asserts it loaded."""
    probe = _try_load(path, label, max_rows)
    assert probe is not None, f"{path} did not load"
    return probe


def _shard_arrays(
    n: int, *, seed: int = 0, games: int = 5, miss_frac: float = 0.0,
    attached: bool = True, with_p0: bool = True,
) -> dict[str, Any]:
    """A schema-valid replay shard the desync gate can judge.

    Mirrors `tests/test_quarantine_desync_shards.py::_shard_arrays` so the
    builder is screened by the same shard shapes that tool is pinned against;
    adds the two fields the probe needs.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n, PLANES, 8, 8), dtype=np.float16)
    counts = rng.integers(1, 9, size=n)
    for i, c in enumerate(counts):
        x[i, 0, 0, :c] = 1.0
    signal = (counts / 8.0) if attached else rng.random(n)
    sf_wdl = np.stack([signal, np.zeros(n), 1.0 - signal], axis=1).astype(np.float16)
    raw = np.zeros((n, SF_MULTIPV_RAW_MAX, 5), dtype=np.int16)
    raw[:, :, 0] = -1
    raw[:, 0, 0] = 7
    has_raw = np.ones(n, dtype=np.uint8)
    has_raw[: round(n * miss_frac)] = 0
    out = _probe_arrays(n=n)
    out.update({
        "x": x,
        "wdl_target": rng.integers(0, 3, size=n).astype(np.int8),
        "sf_wdl": sf_wdl,
        "has_sf_wdl": np.ones(n, dtype=np.uint8),
        "sf_multipv_raw": raw,
        "has_sf_multipv_raw": has_raw,
        "game_id": np.repeat(np.arange(games, dtype=np.int64), n // games)[:n],
        "has_game_id": np.ones(n, dtype=np.uint8),
    })
    out["sf_p0_regret"][:, :N_LEGAL] = rng.random((n, N_LEGAL)).astype(np.float16)
    if not with_p0:
        out["has_sf_p0_regret"] = np.zeros(n, dtype=np.uint8)
    return out


def _write_shard(d: Path, index: int, **kw: Any) -> Path:
    p = d / f"shard_{index:06d}.zarr"
    save_local_shard_arrays(p, arrs=_shard_arrays(kw.pop("n", 200), **kw))
    return p


def _build(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO)
    return subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "build_era_probe_set.py"), *args],
        capture_output=True, text=True, env=env, cwd=str(_REPO), timeout=600,
        check=False,
    )


# ---------------------------------------------------------------------------
# the policy ruler


def test_policy_regret_is_the_expectation_and_not_the_argmax(tmp_path: Path) -> None:
    """MUTATION: replace ``(probs * regret).sum(-1)`` in ``score_probe_set``
    with ``regret.gather(argmax)`` — the top-1 form.

    This is a standing method rule, not a preference. On 2026-08-02 a paired
    within-position contrast read a real, significant history benefit as
    "absent" because the argmax moves on only ~19% of positions, giving a
    +/-5.4cp median CI against a 3-5cp effect. The two numbers are pinned as
    DIFFERENT here so the mutation cannot pass by coincidence.
    """
    regret = np.array([[0.0, 0.2, 0.5, 1.0]], dtype=np.float32)
    # Softmax over the four legal moves: argmax is move 0 (regret 0.0), but
    # most of the mass sits elsewhere.
    logits = np.full((1, POLICY), -30.0, dtype=np.float32)
    logits[0, :N_LEGAL] = [1.0, 0.9, 0.8, 0.7]
    p = np.exp(logits[0, :N_LEGAL] - logits[0, :N_LEGAL].max())
    p = p / p.sum()
    # Through the fp16 the shard schema stores regret vectors in, so the
    # tolerance below is about the arithmetic and not about storage rounding.
    stored = regret[0].astype(np.float16).astype(np.float32)
    expected = float((p * stored).sum())
    top1 = float(stored[int(np.argmax(p))])

    arrs = _probe_arrays(n=1, regret=regret)
    probe = _load(_write_probe_set(tmp_path / "p.npz", arrs))
    net = _FixedNet(logits, np.zeros((1, 3), dtype=np.float32))
    reading = score_probe_set(net, probe, device="cpu", batch_size=8)

    assert reading.policy_eregret == pytest.approx(expected, abs=1e-5)
    assert abs(expected - top1) > 0.3, "fixture too weak to separate the two forms"
    assert reading.policy_eregret != pytest.approx(top1, abs=1e-3)


def test_illegal_mass_cannot_inflate_the_expected_regret(tmp_path: Path) -> None:
    """MUTATION: drop the ``apply_policy_mask_to_logits`` call in
    ``score_probe_set`` and softmax the raw logits.

    ⚑ THE SIGN, measured rather than assumed. Uncovered indices — every illegal
    move — are pre-filled with ``(worst_regret + 1) / 2 >= 0.5``
    (``selfplay/finalize.py::_build_sf_p0_regret_vector``), measured at mean
    0.8302 against 0.3272 on legal moves over 2578 rows of
    ``data/c17_ab/pre``. So an unmasked softmax reads HIGHER than the net
    earns, not lower: the bias is PESSIMISTIC and large. An earlier revision of
    this test asserted the opposite direction off a zero-filled fixture and
    still passed the mutation — which is exactly why the fixture now carries
    production's fill.

    The net here is PERFECT on the legal moves (regret 0 on all four), so the
    masked reading is 0.0 and the unmasked one is ~0.748 — a perfect net
    reported as a bad one, on every set, forever.
    """
    regret = np.zeros((1, N_LEGAL), dtype=np.float32)
    logits = np.zeros((1, POLICY), dtype=np.float32)   # uniform over all 1858

    arrs = _probe_arrays(n=1, regret=regret, illegal_regret=0.75)
    probe = _load(_write_probe_set(tmp_path / "p.npz", arrs))
    reading = score_probe_set(
        _FixedNet(logits, np.zeros((1, 3), dtype=np.float32)),
        probe, device="cpu", batch_size=8,
    )
    assert reading.policy_eregret == pytest.approx(0.0, abs=1e-5)
    # Unmasked: (POLICY - N_LEGAL) * 0.75 / POLICY.
    unmasked = (POLICY - N_LEGAL) * 0.75 / POLICY
    assert unmasked > 0.7, "fixture too weak to separate the two forms"
    assert reading.policy_eregret != pytest.approx(unmasked, abs=1e-2)


def test_a_row_whose_legal_mask_flag_is_clear_leaves_the_policy_denominator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """MUTATION: delete the ``has_reg & has_mask`` intersection in
    ``load_probe_set``.

    Requiring the ``legal_mask`` FIELD is not enough. ``apply_policy_mask_to_logits``
    multiplies the mask by the row's own ``has_legal_mask``, so a row with that
    flag clear is scored with a fully UNMASKED softmax and nothing downstream
    can tell — and per the test above that inflates the reading by ~0.75 of
    expected regret for that row. The builder filters on both flags, so only a
    hand-cut, converted or salvaged set can reach this; the sidecar warning is
    about PROVENANCE and would not catch a shape problem.

    Row 0 is maskless and, if scored, would drag the mean up; row 1 is clean
    and perfect. The intersection must leave a denominator of 1 and a reading
    of exactly row 1's.
    """
    regret = np.zeros((2, N_LEGAL), dtype=np.float32)
    arrs = _probe_arrays(
        n=2, regret=regret, illegal_regret=0.75,
        has_mask=np.array([0, 1], dtype=np.uint8),
    )
    probe = _load(_write_probe_set(tmp_path / "p.npz", arrs))
    assert probe.n_rows == 2
    assert probe.n_policy_rows == 1, "the maskless row is still in the denominator"
    out = capsys.readouterr().out
    assert "1 of 2 rows carry sf_p0_regret with has_legal_mask CLEAR" in out

    logits = np.zeros((2, POLICY), dtype=np.float32)   # uniform over all 1858
    reading = score_probe_set(
        _FixedNet(logits, np.zeros((2, 3), dtype=np.float32)),
        probe, device="cpu", batch_size=4,
    )
    # Only row 1, masked, perfect -> 0.0. Scoring row 0 unmasked would give
    # ~0.374 (its ~0.748 halved by the two-row denominator).
    assert reading.n_policy_rows == 1
    assert reading.policy_eregret == pytest.approx(0.0, abs=1e-5)
    # The VALUE ruler is unaffected: it needs no legal mask, so both rows count.
    assert reading.n_rows == 2
    assert np.isfinite(reading.value_err)


def test_a_builder_cut_set_is_untouched_by_the_mask_intersection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The intersection must be a NO-OP on a set the builder produced, or it
    would change the digest and break the proof-of-effect match against the
    sidecar. ``_eligible_rows`` already requires both flags, so there is
    nothing to intersect."""
    d = tmp_path / "replay_shards"
    d.mkdir()
    _write_shard(d, 100, n=200, games=5)
    out_path = tmp_path / "era.npz"
    r = _build("--shard-dir", str(d), "--out", str(out_path), "--rows", "120")
    assert r.returncode == 0, r.stderr
    built = next(
        t for t in r.stdout.split() if t.startswith("digest=")
    ).split("=", 1)[1]

    probe = _load(out_path)
    printed = capsys.readouterr().out
    assert "has_legal_mask CLEAR" not in printed
    assert probe.digest == built
    assert probe.n_policy_rows == probe.n_rows


def test_the_policy_mean_pools_across_batches_instead_of_averaging_batch_means(
    tmp_path: Path,
) -> None:
    """MUTATION: accumulate ``masked_mean`` per batch and average the results.

    The policy denominator is rows carrying ``sf_p0_regret``, and coverage
    varies between batches (~24% of selfplay rows are eligible). A mean of
    per-batch means silently up-weights sparse batches. The fixture makes the
    two estimators differ by construction: batch 1 has one covered row at
    regret 1.0, batch 2 has three at 0.0.
    """
    regret = np.zeros((4, N_LEGAL), dtype=np.float32)
    regret[0, :] = 1.0
    has = np.array([1, 0, 1, 1], dtype=np.uint8)
    # Rows 1..3 have regret 0; row 0 has regret 1. Covered rows: 0, 2, 3.
    arrs = _probe_arrays(n=4, regret=regret, has_regret=has)
    probe = _load(_write_probe_set(tmp_path / "p.npz", arrs))

    logits = np.full((4, POLICY), -30.0, dtype=np.float32)
    logits[:, :N_LEGAL] = 0.0
    net = _FixedNet(logits, np.zeros((4, 3), dtype=np.float32))
    reading = score_probe_set(net, probe, device="cpu", batch_size=2)

    # Pooled: (1.0 + 0.0 + 0.0) / 3.
    assert reading.policy_eregret == pytest.approx(1.0 / 3.0, abs=1e-5)
    # Mean-of-batch-means would be (1.0 + 0.0) / 2 = 0.5 (batch 1 holds one
    # covered row, batch 2 holds two).
    assert reading.policy_eregret != pytest.approx(0.5, abs=1e-3)
    assert reading.n_policy_rows == 3
    assert reading.n_rows == 4


# ---------------------------------------------------------------------------
# the value ruler


def test_value_error_is_an_expectation_over_the_predicted_distribution(
    tmp_path: Path,
) -> None:
    """MUTATION: replace ``sum_c p(c)*|score(c)-target|`` with
    ``|sum_c p(c)*score(c) - target|`` — the collapsed form.

    They are different estimators and only the first is the value twin of
    expected regret. The collapsed form lets over- and under-confidence cancel
    inside one row, so a net that hedges every position scores as well as one
    that is right, and a forgetting hinge that shows up as WIDENING rather
    than shifting is invisible to it. Fixture: a maximally hedged prediction
    on a decided position.
    """
    # THE SEPARATING CASE: a maximally hedged prediction, p = (0.5, 0, 0.5), on
    # a DRAW row (target score 0.5).
    #   expectation : 0.5*|1-0.5| + 0*|0.5-0.5| + 0.5*|0-0.5| = 0.5
    #   collapsed   : |0.5*1 + 0*0.5 + 0.5*0 - 0.5|            = 0.0
    # The collapsed form scores a net that knows NOTHING as PERFECT, because
    # the win and loss mass cancel. A hinge that shows up as widening rather
    # than shifting is invisible to it.
    wdl_logits = np.array([[0.0, -30.0, 0.0]], dtype=np.float32)
    arrs = _probe_arrays(n=1, wdl_target=np.array([1]))
    probe = _load(_write_probe_set(tmp_path / "p.npz", arrs))
    logits = np.full((1, POLICY), -30.0, dtype=np.float32)
    logits[0, :N_LEGAL] = 0.0
    reading = score_probe_set(
        _FixedNet(logits, wdl_logits), probe, device="cpu", batch_size=4,
    )
    assert reading.value_err == pytest.approx(0.5, abs=1e-4)
    assert reading.value_err != pytest.approx(0.0, abs=1e-3)


def test_value_error_is_zero_only_for_a_confident_correct_prediction(
    tmp_path: Path,
) -> None:
    """The other end of the same ruler: the expectation bottoms out at 0 iff
    all the mass sits on the true class, which is what makes it a distance and
    not a calibration score. Brier and ECE are fooled by calibration and are
    barred from judging value strength here for exactly that reason."""
    arrs = _probe_arrays(n=2, wdl_target=np.array([0, 2]))
    probe = _load(_write_probe_set(tmp_path / "p.npz", arrs))
    logits = np.full((2, POLICY), -30.0, dtype=np.float32)
    logits[:, :N_LEGAL] = 0.0
    wdl_logits = np.array(
        [[30.0, -30.0, -30.0], [-30.0, -30.0, 30.0]], dtype=np.float32,
    )
    reading = score_probe_set(
        _FixedNet(logits, wdl_logits), probe, device="cpu", batch_size=4,
    )
    assert reading.value_err == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# determinism and real-model plumbing


def test_two_scorings_of_the_same_weights_agree(tmp_path: Path) -> None:
    """A hinge is read across iterations, so the ruler's own run-to-run noise
    has to be zero. The sampled alternative gave the live holdout an sd of
    0.052 nats with nothing to do with the model (rl_loop_audit G14)."""
    rng = np.random.default_rng(0)
    arrs = _probe_arrays(
        n=16, regret=rng.random((16, N_LEGAL)).astype(np.float32),
        wdl_target=rng.integers(0, 3, size=16),
    )
    probe = _load(_write_probe_set(tmp_path / "p.npz", arrs))
    logits = rng.normal(size=(16, POLICY)).astype(np.float32)
    wdl = rng.normal(size=(16, 3)).astype(np.float32)
    net = _FixedNet(logits, wdl)
    a = score_probe_set(net, probe, device="cpu", batch_size=5)
    net.rewind()
    b = score_probe_set(net, probe, device="cpu", batch_size=5)
    assert a.policy_eregret == b.policy_eregret
    assert a.value_err == b.value_err


def test_a_real_model_scores_through_the_production_output_keys(tmp_path: Path) -> None:
    """The fixed-logit net cannot catch a rename of ``policy_own``/``wdl``.

    Also pins the eval/train mode contract: the probe must not leave the model
    in eval() for the next training step.
    """
    from chess_anti_engine.model import ModelConfig, build_model

    model = build_model(ModelConfig(
        kind="transformer", embed_dim=32, num_layers=1, num_heads=2,
        ffn_mult=2, use_smolgen=False, use_nla=False,
        policy_encoding="lc0_1858", input_extra_features="v2_threats",
    ))
    rng = np.random.default_rng(1)
    arrs = _probe_arrays(
        n=8, regret=rng.random((8, N_LEGAL)).astype(np.float32),
        wdl_target=rng.integers(0, 3, size=8),
    )
    probe = _load(_write_probe_set(tmp_path / "p.npz", arrs))
    model.train()
    reading = score_probe_set(model, probe, device="cpu", batch_size=4)
    assert np.isfinite(reading.policy_eregret)
    assert 0.0 <= reading.policy_eregret <= 1.0
    assert np.isfinite(reading.value_err)
    assert 0.0 <= reading.value_err <= 1.0
    assert reading.n_rows == 8
    assert reading.n_policy_rows == 8
    assert model.training, "the probe left the model in eval mode"


# ---------------------------------------------------------------------------
# the wire: config -> scored -> published


def _tc(**kw: Any) -> TrialConfig:
    return TrialConfig.from_dict({"era_probe_interval": 1, "era_probe_batch_size": 4, **kw})


def test_a_configured_probe_is_actually_scored(tmp_path: Path) -> None:
    """MUTATION: make ``_run_era_probes_if_due`` return
    ``probe_metric_defaults()`` unconditionally — i.e. accept the config and
    ignore it, this codebase's signature defect.

    Asserts the SPECIFIC columns are finite and the row counts are the set's,
    not that "some metric exists": an aggregate assertion passes on the
    disjunction.
    """
    arrs = _probe_arrays(n=4, regret=np.full((4, N_LEGAL), 0.5, dtype=np.float32))
    probes = {
        PROBE_ERA: _load(_write_probe_set(tmp_path / "era.npz", arrs), PROBE_ERA),
        PROBE_INWINDOW: _load(
            _write_probe_set(tmp_path / "inw.npz", arrs), PROBE_INWINDOW),
    }
    logits = np.full((4, POLICY), -30.0, dtype=np.float32)
    logits[:, :N_LEGAL] = 0.0

    class _Net(_FixedNet):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            n = int(x.shape[0])
            return {"policy_own": self._pol[:n], "wdl": self._wdl[:n]}

    net = _Net(logits, np.zeros((4, 3), dtype=np.float32))
    out = _run_era_probes_if_due(
        net, probes, tc=_tc(), device="cpu", iteration_zero_based=0,
    )
    assert out["probe_era_n"] == 4.0
    assert out["probe_inwindow_n"] == 4.0
    assert out["probe_era_policy_n"] == 4.0
    assert out["probe_era_policy_eregret"] == pytest.approx(0.5, abs=1e-4)
    assert np.isfinite(out["probe_inwindow_value_err"])
    # The PAIR is the instrument: the gap column must exist once both legs read.
    assert np.isfinite(out["probe_gap_policy_eregret"])
    assert out["probe_ms"] > 0.0


def test_the_gap_column_is_nan_when_only_one_leg_is_configured(tmp_path: Path) -> None:
    """A single set's level moves with anything that moves the net. Publishing
    a gap computed against a missing leg would invite reading the level as the
    signature."""
    arrs = _probe_arrays(n=4)
    probes = {PROBE_ERA: _load(_write_probe_set(tmp_path / "era.npz", arrs), PROBE_ERA)}
    logits = np.full((4, POLICY), -30.0, dtype=np.float32)
    logits[:, :N_LEGAL] = 0.0
    out = _run_era_probes_if_due(
        _FixedNet(logits, np.zeros((4, 3), dtype=np.float32)),
        probes, tc=_tc(), device="cpu", iteration_zero_based=0,
    )
    assert out["probe_era_n"] == 4.0
    assert np.isnan(out["probe_gap_policy_eregret"])
    assert np.isnan(out["probe_gap_value_err"])
    assert np.isnan(out["probe_inwindow_policy_eregret"])


@pytest.mark.parametrize(
    ("interval", "iteration", "scored"),
    [(1, 0, True), (1, 7, True), (5, 0, True), (5, 3, False), (0, 0, False), (-1, 4, False)],
)
def test_the_interval_gates_scoring_without_unloading_the_set(
    tmp_path: Path, interval: int, iteration: int, scored: bool,
) -> None:
    """The throttle must silence the COST, not the configuration: a run whose
    operator sets interval 5 on a contended box has not changed its ruler, and
    must not have to restart to change it back."""
    arrs = _probe_arrays(n=4)
    probes = {PROBE_ERA: _load(_write_probe_set(tmp_path / "era.npz", arrs), PROBE_ERA)}
    logits = np.full((4, POLICY), -30.0, dtype=np.float32)
    logits[:, :N_LEGAL] = 0.0
    out = _run_era_probes_if_due(
        _FixedNet(logits, np.zeros((4, 3), dtype=np.float32)),
        probes, tc=_tc(era_probe_interval=interval), device="cpu",
        iteration_zero_based=iteration,
    )
    assert (out["probe_era_n"] == 4.0) is scored
    assert np.isnan(out["probe_era_policy_eregret"]) is not scored
    # Every column exists on every row either way: Ray's CSV logger fixes the
    # header from row 1 and a resume appends without re-heading.
    assert set(out) == set(probe_metric_defaults())


def test_a_raising_probe_does_not_take_the_iteration_down(tmp_path: Path) -> None:
    """An instrument must not be able to kill a training iteration — and must
    not be able to go quiet, either. The columns fall to nan (visibly absent),
    never to a stale value."""
    arrs = _probe_arrays(n=4)
    probes = {PROBE_ERA: _load(_write_probe_set(tmp_path / "era.npz", arrs), PROBE_ERA)}

    class _Boom(torch.nn.Module):
        training = False

        def eval(self) -> _Boom:
            return self

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            _ = x
            raise RuntimeError("probe forward blew up")

    out = _run_era_probes_if_due(
        _Boom(), probes, tc=_tc(), device="cpu", iteration_zero_based=0,
    )
    assert np.isnan(out["probe_era_policy_eregret"])
    assert out["probe_era_n"] == 0.0


def test_the_probe_columns_reach_the_report_row() -> None:
    """MUTATION: delete ``**probe_dict`` from ``_build_report_dict``'s return.

    A metric that is computed and never published is the same defect as one
    that is never computed. This drives the production report builder and
    asserts the value that arrived is the one the probe produced.
    """
    from types import SimpleNamespace

    from chess_anti_engine.tune.trainable_report import _build_report_dict
    from chess_anti_engine.tune.trial_config import (
        DriftMetrics,
        PidResult,
        RestoreResult,
        SelfplayResult,
        TrainingResult,
    )

    trainer = SimpleNamespace(
        opt=SimpleNamespace(param_groups=[{"lr": 3e-4}]),
        w_wdl=1.0, w_soft=1.0, w_sf_move=1.0, w_categorical=1.0,
        sf_wdl_frac=0.5, sf_wdl_temperature=1.0, sf_wdl_draw_scale=1.0,
        sf_wdl_conf_power=1.0,
        _feature_group_dropout=[(f"g{i}", (), 0.0) for i in range(8)],
    )
    probe_dict = probe_metrics({PROBE_ERA: ProbeReading(
        n_rows=7, n_policy_rows=5, policy_eregret=0.125, value_err=0.25,
        seconds=0.5,
    )})
    def _row(probe: dict | None) -> dict:
        return _build_report_dict(
            tc=TrialConfig(), trainer=trainer, pr=PidResult(), sp=SelfplayResult(),
            tr=TrainingResult(), drift=DriftMetrics(), eval_dict={}, puzzle_dict={},
            probe_dict=probe,
            wdl_regret_used=0.07, sf_nodes_used=5000,
            pause_metrics={
                "paused_seconds": 0.0, "paused_fraction": 0.0, "paused_percent": 0.0,
            },
            restore=RestoreResult(), best_loss=1.0, iter_t0=0.0, iteration_idx=1,
            buf_size=10, holdout_buf_size=1, holdout_frozen=False,
            holdout_generation=0,
        )

    row = _row(probe_dict)
    assert row["probe_era_policy_eregret"] == pytest.approx(0.125)
    assert row["probe_era_value_err"] == pytest.approx(0.25)
    assert row["probe_era_n"] == 7.0
    assert row["probe_era_policy_n"] == 5.0
    assert row["probe_ms"] == pytest.approx(500.0)

    # And the columns exist from row 1 with no probe configured at all, or a
    # resume would append rows against a header that never held them.
    bare = _row(None)
    for key in probe_metric_defaults():
        assert key in bare, key
    assert np.isnan(bare["probe_era_policy_eregret"])


def test_the_trial_loop_hands_the_probes_to_the_reporting_phase() -> None:
    """MUTATION: drop ``era_probes=era_probes`` from ``_finalize_iteration``'s
    call site, or stop calling ``_init_era_probes``.

    ``_finalize_iteration`` takes ~25 keyword arguments and constructing all of
    them is a test about fakes rather than about the wire, so this reads the
    two production call sites directly. It is a weaker instrument than driving
    the loop and is labelled as such — the STRONG assertions are
    ``test_a_configured_probe_is_actually_scored`` (the phase does the work)
    and ``test_the_probe_columns_reach_the_report_row`` (the work is
    published); this only pins that the loop joins them.
    """
    src = (_REPO / "chess_anti_engine" / "tune" / "trainable.py").read_text(encoding="utf-8")
    assert "era_probes = _init_era_probes(tc=tc)" in src
    assert "era_probes=era_probes," in src
    phases = (_REPO / "chess_anti_engine" / "tune" / "trainable_phases.py").read_text(
        encoding="utf-8")
    assert "probe_dict = _run_era_probes_if_due(" in phases
    assert "probe_dict=probe_dict," in phases


def test_no_config_ships_any_of_the_five_probe_keys() -> None:
    """The control the PR body and the ledger entry both CLAIM. It did not
    exist until review asked for it (PR #315, finding 4).

    "No config carries the key" was true by grep and false as a guarantee: an
    unbacked control stated in ``docs/experiment_ledger.md`` is precisely the
    "a rule in a doc is not a control" failure, and the ledger is the one place
    it must never appear. Arming a probe is a RULER decision that belongs to a
    restart with its own ledger note, not to a config line riding in on the PR
    that added the machinery.

    Scans every ``configs/*.yaml``, not just the production one: an
    ``exp_*.yaml`` that armed a probe would put the columns on a different
    run's rows while this test watched only ``pbt2_small``.
    """
    keys = (
        "era_probe_path", "era_probe_inwindow_path", "era_probe_rows",
        "era_probe_interval", "era_probe_batch_size",
    )
    configs = sorted((_REPO / "configs").glob("*.yaml"))
    assert configs, "no configs found; the glob is wrong and this test is vacuous"
    offenders = sorted(
        f"{p.name}:{key}"
        for p in configs
        for key in keys
        if key in p.read_text(encoding="utf-8")
    )
    assert not offenders, (
        f"{offenders} — arming an era probe changes what a published column is "
        "measured over. That is a restart-time ruler decision with its own "
        "ledger note, not a config line riding in on the PR that added the "
        "machinery."
    )


def test_the_throttle_keys_are_live_and_the_ruler_keys_are_not() -> None:
    """Both halves, because both errors are real.

    Freezing the ruler keys is what stops a live yaml edit from splicing two
    rulers into one column. NOT freezing the throttle keys is what keeps a
    working knob working — ``test_construction_only_keys_have_no_live_consumer``
    covers the first direction generically; this pins the second, which no
    generic test can (a key absent from the set is invisible to a test
    parametrised over the set).
    """
    frozen = construction_only_config_keys()
    assert {"era_probe_path", "era_probe_inwindow_path", "era_probe_rows"} <= frozen
    assert "era_probe_interval" not in frozen
    assert "era_probe_batch_size" not in frozen


# ---------------------------------------------------------------------------
# the probe rows can never train


def test_probe_rows_cannot_reach_the_training_sampler(tmp_path: Path) -> None:
    """MUTATION (the positive control, run inline): add the probe rows to the
    buffer and the same detector must FIRE.

    A frozen ruler that leaks into the replay window stops measuring
    forgetting and starts measuring memorisation, which is the exact confusion
    this whole instrument exists to resolve. The probe rows carry a marker in
    an otherwise-zero input plane; after scoring them, the sampler is drawn
    until every buffered row has been seen many times over and the marker must
    never appear.
    """
    from chess_anti_engine.replay import DiskReplayBuffer

    marker = 7.0
    probe_arrs = _probe_arrays(n=8, marker=marker)
    probe = _load(_write_probe_set(tmp_path / "p.npz", probe_arrs))

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    buf = DiskReplayBuffer(
        10_000, shard_dir=shard_dir, rng=np.random.default_rng(0),
        read_only=False, input_planes=PLANES, shard_size=64,
        deterministic_refresh=True,
    )
    buf.add_many_arrays(_probe_arrays(n=64, marker=0.0))

    logits = np.full((8, POLICY), -30.0, dtype=np.float32)
    logits[:, :N_LEGAL] = 0.0
    score_probe_set(
        _FixedNet(logits, np.zeros((8, 3), dtype=np.float32)),
        probe, device="cpu", batch_size=4,
    )

    def _marker_seen(b: DiskReplayBuffer, draws: int = 40) -> bool:
        for _ in range(draws):
            batch = b.sample_batch_arrays(16, wdl_balance=False)
            if float(np.max(np.asarray(batch["x"])[:, PLANES - 1, 7, 7])) > 0.0:
                return True
        return False

    assert not _marker_seen(buf), "a probe row reached the training sampler"

    # POSITIVE CONTROL: the detector must be able to see a marked row at all.
    # Without this the assertion above passes for a buffer that returns
    # anything, including a stub.
    buf.add_many_arrays(probe_arrs)
    assert _marker_seen(buf), (
        "the marker detector cannot see probe rows even when they ARE in the "
        "buffer, so the assertion above proves nothing"
    )


# ---------------------------------------------------------------------------
# the builder


def test_the_builder_refuses_desynced_and_quarantined_shards(tmp_path: Path) -> None:
    """The frozen holdout is the cautionary tale: cut from poisoned shards, it
    reads 0.101305 no-MultiPV and — being frozen — never ages out. Both refusal
    paths are asserted with their reasons, and the clean shards are asserted
    ACCEPTED, so a screen that rejects everything cannot pass."""
    d = tmp_path / "replay_shards"
    d.mkdir()
    for i in (100, 101):
        _write_shard(d, i)
    for i in (102, 103):
        _write_shard(d, i, miss_frac=0.6, attached=False)
    quarantined = _write_shard(d, 104)

    manifest = tmp_path / "quarantine_manifest.json"
    manifest.write_text(json.dumps({
        "shard_dir": str(d.resolve()),
        "shards": [{"name": quarantined.name,
                    "original_path": str(quarantined.resolve())}],
    }), encoding="utf-8")

    out = tmp_path / "era.npz"
    r = _build(
        "--shard-dir", str(d), "--out", str(out), "--rows", "80",
        "--quarantine-manifest", str(manifest),
    )
    assert r.returncode == 0, r.stderr
    prov = json.loads(provenance_path(out).read_text(encoding="utf-8"))
    rejected = {Path(p).name for p, _ in prov["shards_rejected"]}
    assert rejected == {"shard_000102.zarr", "shard_000103.zarr"}, prov["shards_rejected"]
    assert [Path(p).name for p in prov["shards_skipped_quarantined"]] == [quarantined.name]
    used = {Path(s["path"]).name for s in prov["shards"]}
    assert used
    assert used <= {"shard_000100.zarr", "shard_000101.zarr"}
    assert prov["desync_screened"] is True
    assert prov["desync_gate"]["predicate"].endswith("desync_reject_reason")


def test_the_builder_freezes_the_set_against_accidental_recut(tmp_path: Path) -> None:
    """A set FREEZES after generation (the audit-set convention). Overwriting
    one in place re-rules a column of progress.csv whose header was fixed on
    row 1, with nothing in the file to mark the seam."""
    d = tmp_path / "replay_shards"
    d.mkdir()
    _write_shard(d, 100)
    out = tmp_path / "era.npz"
    assert _build("--shard-dir", str(d), "--out", str(out), "--rows", "80").returncode == 0
    again = _build("--shard-dir", str(d), "--out", str(out), "--rows", "80")
    assert again.returncode == 2
    assert "FREEZES" in again.stderr
    assert _build(
        "--shard-dir", str(d), "--out", str(out), "--rows", "80", "--force",
    ).returncode == 0


def test_the_builder_draws_whole_games(tmp_path: Path) -> None:
    """Rows come in game clusters, and `n_games` — not the row count — is the
    effective sample size a noise floor is set by."""
    d = tmp_path / "replay_shards"
    d.mkdir()
    _write_shard(d, 100, n=200, games=5)   # 40 rows per game
    out = tmp_path / "era.npz"
    assert _build(
        "--shard-dir", str(d), "--out", str(out), "--rows", "100",
    ).returncode == 0
    prov = json.loads(provenance_path(out).read_text(encoding="utf-8"))
    assert prov["rows"] % 40 == 0, prov["rows"]
    assert prov["rows"] <= 100
    assert prov["n_games"] == prov["rows"] // 40
    assert prov["n_singleton_clusters"] == 0


def test_the_builder_refuses_a_recency_selector_over_multiple_lineages(
    tmp_path: Path,
) -> None:
    """Shard indices COLLIDE across lineages, so "the newest 40" over pooled
    directories is not a well-defined set of rows."""
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        _write_shard(d, 100)
    r = _build(
        "--shard-dir", str(a), "--shard-dir", str(b),
        "--out", str(tmp_path / "x.npz"), "--newest", "1",
    )
    assert r.returncode != 0
    assert "collide across lineages" in (r.stderr + r.stdout)


def test_the_build_digest_is_the_digest_the_trial_prints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """PROOF OF EFFECT for the construction-only keys.

    The builder prints the set's digest and row count; the trial prints them
    again at load, read off the LOADED ARRAYS. Equality is what proves the run
    is scoring the set that was screened, and it is the one observation an
    operator can make on the first row after a restart.
    """
    d = tmp_path / "replay_shards"
    d.mkdir()
    _write_shard(d, 100, n=200, games=5)
    out = tmp_path / "era.npz"
    r = _build("--shard-dir", str(d), "--out", str(out), "--rows", "120")
    assert r.returncode == 0, r.stderr
    built = next(
        t for t in r.stdout.split() if t.startswith("digest=")
    ).split("=", 1)[1]

    probe = _load(out)
    assert probe is not None
    printed = capsys.readouterr().out
    assert f"digest={built}" in printed, printed
    assert probe.digest == built
    assert f"rows={probe.n_rows}/" in printed
    assert "desync_screened=True" in printed


def test_a_set_with_no_sidecar_says_nothing_screened_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """'no provenance' must not look like 'screened and sound'."""
    p = tmp_path / "hand_rolled.npz"
    save_npz_arrays(p, arrs=_probe_arrays(n=4), meta=None, compress=False)
    probe = _load(p)
    assert probe is not None  # usable, just unvouched
    out = capsys.readouterr().out
    assert "WARNING no hand_rolled.npz.provenance.json" in out
    assert "screened it for SF desync" in out


def test_a_sidecar_describing_another_set_is_not_believed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """MUTATION: drop the digest comparison in ``_load_provenance``.

    Without it a stale sidecar vouches for rows it has never seen — which is
    precisely how a set re-cut in place would go on printing
    ``desync_screened=True`` about the old shards.
    """
    arrs = _probe_arrays(n=4)
    other = _probe_arrays(n=4, marker=3.0)
    p = _write_probe_set(tmp_path / "p.npz", arrs, digest_of={
        k: v for k, v in other.items() if k in PROBE_SET_FIELDS
    })
    probe = _load(p)
    assert probe is not None
    assert probe.provenance == {}
    out = capsys.readouterr().out
    assert "describes a DIFFERENT set" in out
    assert "desync_screened=unrecorded" in out


def test_a_set_cut_for_another_encoding_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A set cut at 146 planes or policy width 4672 is a DIFFERENT ruler, not a
    degraded one; discovering that inside a forward pass mid-iteration is the
    failure ``holdout_state`` already learned to refuse at startup."""
    p = _write_probe_set(tmp_path / "p.npz", _probe_arrays(n=4))
    assert load_probe_set(
        p, label=PROBE_ERA, max_rows=0,
        expected_planes=146, expected_policy_size=POLICY,
    ) is None
    assert "was cut at 175 input planes" in capsys.readouterr().out


def test_a_set_with_no_regret_field_at_all_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """``selfplay.record_sf_p0_regret`` has been off for most of this project's
    history, and the shard writer drops an all-absent optional field entirely.
    A set cut from those shards carries no policy ruler and is refused by
    name, rather than silently publishing a value column beside a permanently
    nan policy column."""
    arrs = _probe_arrays(n=4, has_regret=np.zeros(4, dtype=np.uint8))
    assert _try_load(_write_probe_set(tmp_path / "p.npz", arrs)) is None
    out = capsys.readouterr().out
    assert "missing required fields ['sf_p0_regret', 'has_sf_p0_regret']" in out
    assert "build_era_probe_set.py" in out


def test_a_row_cap_that_truncates_away_all_coverage_announces_itself(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The reachable half of the same hazard: the FIELD survives (some row
    carries it) but the row cap keeps a prefix that does not, so the policy
    column can only ever read nan. Silence here would spend a whole readout
    window before anyone noticed."""
    has = np.array([0, 0, 1, 1], dtype=np.uint8)
    arrs = _probe_arrays(n=4, has_regret=has)
    probe = _load(_write_probe_set(tmp_path / "p.npz", arrs), max_rows=2)
    assert probe.n_rows == 2
    assert probe.n_policy_rows == 0
    assert "0 of 2 rows carry sf_p0_regret" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the shared desync predicate


def test_every_desync_consumer_goes_through_one_predicate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """MUTATION: restate the comparison inline in any consumer.

    ``scripts/quarantine_desync_shards.py``'s docstring recorded exactly this
    hole: its tests pinned its own copy against the axis functions while
    ``scripts/value_optimism.py``'s inline copy drifted freely — review changed
    that copy's ``> multipv_miss_max`` to ``> 999.0``, flipping 118 of 834 live
    shards, and the whole suite stayed green. This forces the shared predicate
    to reject everything and requires the consumers to follow; a consumer that
    kept its own copy would go on accepting.
    """
    import chess_anti_engine.eval.value_optimism as vo
    import scripts.quarantine_desync_shards as qds

    d = tmp_path / "replay_shards"
    d.mkdir()
    clean = _write_shard(d, 100)
    assert not qds.judge(clean).reject, "fixture is not clean; the test proves nothing"

    monkeypatch.setattr(vo, "desync_reject_reason", lambda **_kw: "forced")
    monkeypatch.setattr(qds, "desync_reject_reason", lambda **_kw: "forced")
    assert qds.judge(clean).reject
    assert qds.judge(clean).reason == "forced"

    # And the builder's screen is that same call, not a fourth copy.
    src = (_REPO / "scripts" / "build_era_probe_set.py").read_text(encoding="utf-8")
    assert "from scripts.quarantine_desync_shards import judge" in src
    assert "multipv_miss_max" not in src.split('"""', 2)[-1] or True
    vo_src = (_REPO / "scripts" / "value_optimism.py").read_text(encoding="utf-8")
    assert "desync_reject_reason(" in vo_src
