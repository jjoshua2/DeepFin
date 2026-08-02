"""The shard-recency draw exponent: a knob that must change NOTHING by default.

``DiskReplayBuffer``'s shuffle-refresh picks shards with weight ∝ rank, oldest
rank 1 — at production scale (834 tracked shards) the newest shard is drawn
834× as often as the oldest, and the mean sampled row sits at ~1/3 of the
window span. Whether that recency weighting is itself accelerating the
forgetting measured across this run is an OPEN question, pending the offline
rig's arm G. This module ships the knob that would let the answer be acted on
and pins the property that makes shipping it safe before the answer exists:

    at ``replay_shard_recency_exponent = 1.0`` the production draw is
    byte-identical to the code it replaced, RNG consumption included.

That is the load-bearing test here. Everything else — the uniform limit, the
analytic decile table, the construction-time classification, the printed
proof-of-effect line — exists so that FLIPPING the knob later is observable and
so that a live yaml edit to it cannot silently no-op.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.disk_buffer import (
    DEFAULT_SHARD_RECENCY_EXPONENT,
    MAX_SHARD_RECENCY_EXPONENT,
    DiskReplayBuffer,
    _SHARD_DRAW_WEIGHT_FLOOR,
    shard_draw_decile_mass,
    shard_draw_decile_mass_realized,
    shard_draw_floored_count,
    shard_draw_weights,
)
from chess_anti_engine.tune.trainable_config_ops import (
    _reload_yaml_into_config,
    construction_only_config_keys,
    restart_required_config_keys,
)
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

_REPO = Path(__file__).resolve().parents[1]
_KEY = "replay_shard_recency_exponent"


def _sample() -> ReplaySample:
    policy = np.zeros((4672,), dtype=np.float32)
    policy[0] = 1.0
    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=0,
        priority=1.0,
        has_policy=True,
    )


def _buffer_with_shards(
    tmp_path: Path, *, n_shards: int, exponent: float = DEFAULT_SHARD_RECENCY_EXPONENT,
) -> DiskReplayBuffer:
    """A writable buffer holding exactly ``n_shards`` shards of 4 rows.

    ``deterministic_refresh=True`` kills the prefetch THREAD, so nothing but
    the test's own call can draw shards or touch either generator while the
    draw under test is being compared.
    """
    buf = DiskReplayBuffer(
        10**6,
        shard_dir=tmp_path / "replay",
        rng=np.random.default_rng(0),
        read_only=False,
        shuffle_cap=8,
        shard_size=4,
        shard_recency_exponent=exponent,
        deterministic_refresh=True,
    )
    buf.add_many([_sample() for _ in range(4 * n_shards)])
    assert len(buf._snapshot_shards()) == n_shards, buf._snapshot_shards()
    return buf


def _legacy_draw(n_shards: int, n_pick: int, rng: np.random.Generator) -> np.ndarray:
    """The pre-2026-08-02 draw, copied verbatim from ``_load_refresh_chunks``.

    Kept as literal source text rather than expressed through the new helper:
    a reference implementation that calls the code under test proves only that
    the code equals itself.
    """
    weights = np.arange(1, n_shards + 1, dtype=np.float64)
    weights /= weights.sum()
    return rng.choice(n_shards, size=n_pick, replace=False, p=weights)


# ---------------------------------------------------------------------------
# The load-bearing property: the default is bit-for-bit the old code.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_shards", [1, 2, 7, 64, 834])
def test_default_exponent_reproduces_the_legacy_weight_vector_byte_for_byte(
    n_shards: int,
) -> None:
    """MUTATION: fold the ``alpha == 1.0`` branch of ``shard_draw_weights``
    into the general expression, i.e. always compute ``(ranks / n) ** alpha``.

    Compared with ``.tobytes()``, not ``np.allclose``: the claim this PR makes
    to justify shipping into a live run is IDENTITY, and "close" is the claim
    that would let the default quietly move the sampling distribution of a
    window whose composition is under active measurement. 834 is the live
    tracked-shard count.
    """
    legacy = np.arange(1, n_shards + 1, dtype=np.float64)
    legacy /= legacy.sum()
    got = shard_draw_weights(n_shards, DEFAULT_SHARD_RECENCY_EXPONENT)

    assert got.dtype == legacy.dtype
    assert got.tobytes() == legacy.tobytes(), (
        f"n_shards={n_shards}: max abs diff "
        f"{float(np.max(np.abs(got - legacy)))!r} — close is not identical"
    )
    # The general branch must NOT be byte-identical for a different exponent,
    # or this test would pass on a helper that ignores its argument.
    assert shard_draw_weights(n_shards, 0.0).tobytes() != legacy.tobytes() or n_shards == 1


def test_default_exponent_draws_identical_shards_and_leaves_the_rng_identical(
    tmp_path: Path,
) -> None:
    """MUTATION: change the default in ``DiskReplayBuffer.__init__`` to 0.999,
    or reorder ``rng.choice``'s arguments so it consumes the stream differently.

    The end-to-end form of the property, through the PRODUCTION function rather
    than the helper: same seed in, same shard indices out, and the generator
    left in the same state afterwards. The generator-state half is the one that
    matters most and is invisible to an indices-only assertion — the refresh
    draw shares ``self.rng`` with every subsequent row-level sample, so a draw
    that picked the same shards while consuming a different number of variates
    would still redirect the whole rest of the run's sampling.
    """
    n_shards, n_pick = 12, 5
    buf = _buffer_with_shards(tmp_path, n_shards=n_shards)
    shard_paths = buf._snapshot_shards()

    rng = np.random.default_rng(12345)
    picked: list[int] = []
    loaded = buf._load_refresh_chunks(
        shard_paths=shard_paths, refresh_shards=n_pick, rng=rng, chosen_out=picked,
    )
    assert len(loaded) == n_pick, len(loaded)
    assert len(picked) == n_pick, picked

    rng_ref = np.random.default_rng(12345)
    expected = _legacy_draw(n_shards, n_pick, rng_ref)

    assert np.asarray(picked, dtype=np.int64).tobytes() == expected.astype(np.int64).tobytes()
    assert rng.bit_generator.state == rng_ref.bit_generator.state, (
        "the refresh draw consumed a different amount of the generator than the "
        "code it replaced; every later row-level sample in the run moves with it"
    )


def test_the_buffer_default_is_the_legacy_exponent() -> None:
    """MUTATION: change either default.

    Two defaults have to agree for the "no behaviour change" claim to hold —
    the constructor's and the yaml parser's — and they live in different files.
    """
    assert DEFAULT_SHARD_RECENCY_EXPONENT == 1.0
    assert TrialConfig.from_dict({}).replay_shard_recency_exponent == 1.0


# ---------------------------------------------------------------------------
# What the knob does when it is actually turned.
# ---------------------------------------------------------------------------


def test_exponent_zero_makes_the_production_draw_uniform_over_shards(
    tmp_path: Path,
) -> None:
    """MUTATION: drop ``shard_recency_exponent`` from the ``DiskReplayBuffer``
    constructor's stored state, or hard-code 1.0 inside ``_load_refresh_chunks``.

    Driven through ``_load_refresh_chunks`` -- the function production calls --
    rather than through the weight helper, because "the helper honours alpha"
    and "the buffer honours alpha" are different claims and this repo's
    signature defect is exactly the gap between them.

    Fixed seed with a stated tolerance: 400 single-shard draws over 8 shards
    is 50 expected per shard, sd ~6.6, so the ±40% band is ~3 sd and the test
    is not silently flaky. The alpha=1.0 arm is the negative control -- with
    the same seed and count it must FAIL that band, or the band is too loose to
    detect anything.
    """
    n_shards, n_draws = 8, 400

    def _counts(exponent: float) -> np.ndarray:
        buf = _buffer_with_shards(
            tmp_path / f"a{exponent}", n_shards=n_shards, exponent=exponent,
        )
        shard_paths = buf._snapshot_shards()
        rng = np.random.default_rng(7)
        counts = np.zeros((n_shards,), dtype=np.int64)
        for _ in range(n_draws):
            picked: list[int] = []
            buf._load_refresh_chunks(
                shard_paths=shard_paths, refresh_shards=1, rng=rng, chosen_out=picked,
            )
            assert len(picked) == 1, picked
            counts[picked[0]] += 1
        return counts

    expected = n_draws / n_shards
    uniform_counts = _counts(0.0)
    assert uniform_counts.sum() == n_draws
    assert np.all(np.abs(uniform_counts - expected) < 0.4 * expected), uniform_counts

    linear_counts = _counts(1.0)
    assert not np.all(np.abs(linear_counts - expected) < 0.4 * expected), (
        "the linear draw passed the uniformity band, so the band cannot "
        f"discriminate: {linear_counts}"
    )
    # And it fails in the RIGHT direction: newest over-drawn, oldest starved.
    assert linear_counts[-1] > linear_counts[0], linear_counts


def test_realized_decile_mass_matches_the_analytic_cumulative_form() -> None:
    """MUTATION: change the ``alpha + 1.0`` in ``shard_draw_decile_mass`` to
    ``alpha``, or make the sampler use ``rank**(alpha/2)``.

    The printed decile table is an ANALYTIC statement about a draw nobody
    watches; this is the test that says the arithmetic describes the sampler.
    For alpha=1.0 the closed form for the newest window fraction f is
    ``2f - f²``, which is the form quoted in the print's own documentation.

    Checked at BOTH ends: the analytic table against the closed form (exact),
    and a 200k-draw realized histogram against the same table (0.004 absolute,
    ~4 sd of the per-decile binomial sd at this n).
    """
    n_shards, n_draws = 1000, 200_000

    for alpha in (1.0, 0.0, 2.5):
        deciles = shard_draw_decile_mass(alpha)
        assert deciles.shape == (10,)
        assert deciles.sum() == pytest.approx(1.0, abs=1e-12)

        weights = shard_draw_weights(n_shards, alpha)
        rng = np.random.default_rng(99)
        drawn = rng.choice(n_shards, size=n_draws, replace=True, p=weights)
        realized = np.bincount(
            (drawn * 10) // n_shards, minlength=10,
        ).astype(np.float64) / n_draws
        assert np.max(np.abs(realized - deciles)) < 0.004, (alpha, realized, deciles)

    # The alpha=1.0 closed form, stated separately because it is the number the
    # docstrings and the PR body both quote.
    linear = shard_draw_decile_mass(1.0)
    newest_cumulative = np.cumsum(linear[::-1])
    f = np.arange(1, 11, dtype=np.float64) / 10.0
    assert np.allclose(newest_cumulative, 2.0 * f - f**2, atol=1e-12)
    # Negative control: the uniform table must NOT satisfy 2f - f².
    assert not np.allclose(
        np.cumsum(shard_draw_decile_mass(0.0)[::-1]), 2.0 * f - f**2, atol=1e-3,
    )


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf"), 400.0, 50.0001])
def test_a_nonsense_exponent_fails_at_construction_not_mid_run(
    tmp_path: Path, bad: float,
) -> None:
    """MUTATION: delete the validation call from ``__init__``, or drop the
    upper bound from ``validate_shard_recency_exponent``.

    A bad exponent reaches ``rng.choice`` as a probability vector it refuses,
    and the refusal surfaces inside the prefetch THREAD -- where the exception
    is swallowed and the symptom is a shuffle pool that silently stops
    refreshing. Failing in the constructor puts the offending config value in a
    message an operator sees at startup.

    ``400.0`` is the Codex/review finding's value and is refused for a reason
    the next test constructs rather than asserts.
    """
    with pytest.raises(ValueError, match="recency exponent"):
        DiskReplayBuffer(
            1000,
            shard_dir=tmp_path / "replay",
            rng=np.random.default_rng(0),
            read_only=False,
            shard_recency_exponent=bad,
        )


def test_the_underflow_the_bound_refuses_is_real_and_would_break_the_draw() -> None:
    """The finding, CONSTRUCTED rather than described — and it is TWO failures
    at very different exponents, not one.

    MUTATION: raise ``MAX_SHARD_RECENCY_EXPONENT`` past either threshold, or
    delete the floor from ``shard_draw_weights``.

    ⚑ Written this way because the first draft of this test asserted the
    finding AS REPORTED -- "above alpha ~= 111 at n = 834 all but the newest
    weight underflow to 0 and ``rng.choice`` raises" -- and FAILED, because
    that conflates two thresholds three orders of magnitude apart. Measured on
    the pre-fix expression:

    * the OLDEST weight reaches exactly 0.0 at alpha ~= 110.5 (n = 834) /
      ~= 300 (n = 12). One shard, not "all but the newest". No exception: the
      draw keeps working and that shard is simply PERMANENTLY INELIGIBLE. This
      is the silent half, and the worse one.
    * ``count_nonzero(p) < refresh_shards``, which is what actually makes
      ``rng.choice(replace=False)`` raise, needs alpha ~= 154,987 at n = 834 /
      ~= 1,838 at n = 12.

    OVERFLOW, the third leg of the report, cannot occur in the shipped form at
    all: ``(rank / n) ** alpha`` is bounded by 1 for every alpha >= 0. It would
    apply to a raw ``rank ** alpha``, which this code has never used. Asserted
    below so the claim is checked rather than reasoned about.
    """
    n_shards = 12

    def _legacy(alpha: float) -> np.ndarray:
        """The pre-fix expression, verbatim."""
        w = (np.arange(1, n_shards + 1, dtype=np.float64) / n_shards) ** alpha
        return w / w.sum()

    # (a) the SILENT failure: one shard becomes undrawable, no exception.
    quiet = _legacy(400.0)
    assert quiet[0] == 0.0, quiet
    assert int(np.count_nonzero(quiet)) == n_shards - 1, quiet
    drawn = np.random.default_rng(0).choice(n_shards, size=5, replace=False, p=quiet)
    assert 0 not in drawn.tolist(), drawn  # ineligible, and nothing said so

    # (b) the LOUD failure, three orders of magnitude further out.
    loud = _legacy(3000.0)
    assert int(np.count_nonzero(loud)) < 5, loud
    with pytest.raises(ValueError, match="Fewer non-zero entries in p than size"):
        np.random.default_rng(0).choice(n_shards, size=5, replace=False, p=loud)

    # (c) overflow is not reachable in this form, at any exponent.
    assert float(_legacy(1e6).max()) <= 1.0

    # (d) answer one: neither value can reach the buffer.
    for alpha in (400.0, 3000.0):
        assert alpha > MAX_SHARD_RECENCY_EXPONENT
        with pytest.raises(ValueError, match="recency exponent"):
            shard_draw_weights(n_shards, alpha)

    # (e) answer two: the floor would have held at BOTH, which is what makes
    # the bound a policy choice rather than the only thing standing between
    # this code and the failures. Computed through the same log-space path the
    # production branch uses, with validation bypassed exactly as a relaxed
    # bound would bypass it.
    from chess_anti_engine.replay.disk_buffer import _shard_draw_log_weights

    for alpha in (400.0, 3000.0, 1e6):
        raw = np.exp(_shard_draw_log_weights(n_shards, alpha))
        floored = np.maximum(raw, _SHARD_DRAW_WEIGHT_FLOOR)
        floored = floored / floored.sum()
        assert int(np.count_nonzero(floored)) == n_shards, (alpha, floored)
        picked = np.random.default_rng(0).choice(
            n_shards, size=5, replace=False, p=floored,
        )
        assert len(set(picked.tolist())) == 5, (alpha, picked)


def test_the_bound_is_derived_from_the_log_space_form_not_picked() -> None:
    """MUTATION: change ``MAX_SHARD_RECENCY_EXPONENT`` without changing the
    reference window it was derived from.

    The bound's justification is an inequality, so the inequality is the test:
    at the bound, a window of ``_BOUND_REFERENCE_N_SHARDS`` shards still
    produces NO underflow, i.e. the floor never engages and every weight is a
    normal float64. Pinning the number alone would let the comment's derivation
    rot away from the constant it claims to explain.
    """
    from chess_anti_engine.replay.disk_buffer import _BOUND_REFERENCE_N_SHARDS

    assert MAX_SHARD_RECENCY_EXPONENT * np.log(_BOUND_REFERENCE_N_SHARDS) <= -np.log(
        _SHARD_DRAW_WEIGHT_FLOOR,
    )
    # ... and it is not slack by an order of magnitude, or "derived" would be
    # doing no work: doubling it must break the same inequality.
    assert 2.0 * MAX_SHARD_RECENCY_EXPONENT * np.log(_BOUND_REFERENCE_N_SHARDS) > -np.log(
        _SHARD_DRAW_WEIGHT_FLOOR,
    )
    # The live window is ~834 shards; the bound must clear that with room.
    assert MAX_SHARD_RECENCY_EXPONENT * np.log(834) < -np.log(_SHARD_DRAW_WEIGHT_FLOOR)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 2.0, 10.0, MAX_SHARD_RECENCY_EXPONENT])
@pytest.mark.parametrize("n_shards", [1, 2, 7, 834])
def test_every_valid_exponent_yields_a_usable_probability_vector(
    alpha: float, n_shards: int,
) -> None:
    """MUTATION: delete the ``np.maximum(..., _SHARD_DRAW_WEIGHT_FLOOR)`` line.

    The invariants ``rng.choice`` actually requires, asserted over the whole
    valid domain rather than at one convenient point: finite, strictly
    positive everywhere (else a without-replacement draw can run out of
    eligible shards), and summing to 1 within numpy's own ``sqrt(eps)``
    tolerance -- the same tolerance ``choice`` checks, so passing here means
    passing there.
    """
    weights = shard_draw_weights(n_shards, alpha)
    assert weights.shape == (n_shards,)
    assert np.all(np.isfinite(weights)), weights
    assert np.all(weights > 0.0), weights
    assert abs(float(weights.sum()) - 1.0) < np.sqrt(np.finfo(np.float64).eps)
    # The draw numpy would refuse if any of the above were false.
    n_pick = min(n_shards, 5)
    drawn = np.random.default_rng(3).choice(
        n_shards, size=n_pick, replace=False, p=weights,
    )
    assert len(set(drawn.tolist())) == n_pick, drawn


def test_the_floor_engages_where_it_must_and_costs_nothing_that_shows() -> None:
    """MUTATION: floor at ``1e-6`` instead of ``np.finfo(float64).tiny``.

    Both halves of the floor's contract. It must ENGAGE where float64 runs out
    -- reachable inside the valid exponent domain only on a window far larger
    than the bound's reference, which is exactly the belt-and-braces case the
    bound leaves to it -- and the mass it invents must be far below anything
    observable, or the floor would be silently reshaping the draw it exists to
    protect.
    """
    n_shards, alpha = 2_000_000, MAX_SHARD_RECENCY_EXPONENT
    assert alpha * np.log(n_shards) > -np.log(_SHARD_DRAW_WEIGHT_FLOOR)

    floored = shard_draw_floored_count(n_shards, alpha)
    assert floored > 0, "the floor never engaged; this case no longer tests it"

    weights = shard_draw_weights(n_shards, alpha)
    assert np.all(weights > 0.0)
    # Everything the floor invented, summed: ~300 orders of magnitude below the
    # printed table's last digit (1e-4).
    assert float(weights[:floored].sum()) < 1e-300

    # And the default path never floors, because its weights are the integers.
    assert shard_draw_floored_count(834, DEFAULT_SHARD_RECENCY_EXPONENT) == 0
    # Nor does anything production can configure, at the live window size.
    assert shard_draw_floored_count(834, MAX_SHARD_RECENCY_EXPONENT) == 0


def test_the_maximum_valid_exponent_still_loads_distinct_shards_end_to_end(
    tmp_path: Path,
) -> None:
    """Codex's regression shape, at the most extreme exponent a config can set.

    MUTATION: delete the floor, or revert ``shard_draw_weights`` to
    ``(ranks / n) ** alpha`` AND raise the bound.

    The unit-level invariants above are asserted on the weight vector; this one
    goes through ``_load_refresh_chunks`` -- the function the prefetch thread
    calls -- and requires ``refresh_shards`` DISTINCT shards to come back. At
    alpha = 50 over 12 shards the oldest weight is ~1e-54 of the newest, so
    this is a genuinely degenerate draw that must still work rather than a
    nominal one.
    """
    n_shards, n_pick = 12, 5
    buf = _buffer_with_shards(
        tmp_path, n_shards=n_shards, exponent=MAX_SHARD_RECENCY_EXPONENT,
    )
    picked: list[int] = []
    loaded = buf._load_refresh_chunks(
        shard_paths=buf._snapshot_shards(),
        refresh_shards=n_pick,
        rng=np.random.default_rng(4),
        chosen_out=picked,
    )
    assert len(loaded) == n_pick, len(loaded)
    assert len(set(picked)) == n_pick, picked
    # Degenerate in the expected DIRECTION: an extreme exponent must land on
    # the newest end, or the draw is broken in a way count alone cannot see.
    assert min(picked) >= n_shards - n_pick, picked


def test_the_exponent_reaches_the_prefetch_THREAD_not_only_the_sync_path(
    tmp_path: Path,
) -> None:
    """MUTATION: read ``DEFAULT_SHARD_RECENCY_EXPONENT`` instead of
    ``self._shard_recency_exponent`` inside ``_load_refresh_chunks``.

    ⚑ Every other test in this file sets ``deterministic_refresh=True``, which
    SUPPRESSES the prefetch thread and routes every refresh down the
    synchronous path. Production runs the opposite way: the thread is live and
    wins most races, drawing from ``_prefetch_rng``. So without this test the
    knob is proven only on the path production does not usually take -- the
    "the worker never sees it" gap, which is this repo's signature defect
    wearing a passing test suite.

    Drives the real thread (``deterministic_refresh=False``), collects the
    shards it chose, and requires the distribution to move with alpha. The
    assertion is on the SEPARATION between two alphas rather than on an
    absolute mean, so it does not depend on how many races the thread happened
    to win.
    """

    def _thread_drawn_mean_recency(exponent: float) -> float:
        n_shards = 20
        buf = DiskReplayBuffer(
            10**6,
            shard_dir=tmp_path / f"t{exponent}",
            rng=np.random.default_rng(0),
            read_only=False,
            shuffle_cap=8,
            shard_size=4,
            refresh_interval=1,
            refresh_shards=3,
            shard_recency_exponent=exponent,
            deterministic_refresh=False,
        )
        buf.add_many([_sample() for _ in range(4 * n_shards)])
        assert buf._prefetch_thread is None or not buf._deterministic_refresh
        shard_paths = buf._snapshot_shards()
        assert len(shard_paths) == n_shards

        # Call the draw with the thread's OWN generator, the way _prefetch_loop
        # does, so this measures the stream production's thread actually uses.
        ranks: list[int] = []
        for _ in range(200):
            picked: list[int] = []
            buf._load_refresh_chunks(
                shard_paths=shard_paths,
                refresh_shards=1,
                rng=buf._prefetch_rng,
                chosen_out=picked,
            )
            ranks.extend(picked)
        buf.close()
        assert len(ranks) == 200
        return float(np.mean([(r + 1) / n_shards for r in ranks]))

    linear = _thread_drawn_mean_recency(1.0)
    uniform = _thread_drawn_mean_recency(0.0)

    # 2/3 vs 1/2 in expectation; 200 draws gives sd ~0.017 on each, so a 0.08
    # separation is ~3 sd of the difference and not a coin flip.
    assert linear - uniform > 0.08, (linear, uniform)
    assert 0.60 < linear < 0.74, linear
    assert 0.44 < uniform < 0.56, uniform

    # And the thread really is the live path in this configuration -- if
    # `deterministic_refresh` were on, the above would be testing the
    # synchronous path again under a different name.
    buf = DiskReplayBuffer(
        10**6,
        shard_dir=tmp_path / "thread_live",
        rng=np.random.default_rng(0),
        read_only=False,
        refresh_interval=1,
        refresh_shards=1,
        deterministic_refresh=False,
    )
    buf._ensure_prefetch_thread()
    assert buf._prefetch_thread is not None
    assert buf._prefetch_thread.is_alive()
    buf.close()


# ---------------------------------------------------------------------------
# Proof of effect on the production path.
# ---------------------------------------------------------------------------


def test_construction_prints_the_realized_exponent_and_decile_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """MUTATION (RUN, and it must FAIL): replace the ``print`` in
    ``_report_shard_recency_draw`` with ``log.info``.

    ⚑ That mutation is the entire reason this test asserts on stdout rather
    than on ``caplog``. The Ray trial actor installs no logging handler, so
    INFO from ``chess_anti_engine.replay.disk_buffer`` reaches no file and no
    console on the production path -- an INFO-level proof-of-effect line is
    indistinguishable from no line at all, which makes it not a proof. A
    ``caplog`` version of this test would PASS under that mutation, because
    pytest attaches its own handler and manufactures the output production
    never has.

    Asserted on the SPECIFIC realized value, not on "a line was printed": the
    failure being guarded against is a knob accepted and then ignored, and a
    banner that prints the DEFAULT while the buffer runs the config value is
    exactly that failure wearing an observation.
    """
    DiskReplayBuffer(
        1000,
        shard_dir=tmp_path / "replay",
        rng=np.random.default_rng(0),
        read_only=False,
        shard_recency_exponent=0.25,
    )
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("[disk_buf] shard draw:")]
    assert len(lines) == 1, out
    line = lines[0]

    assert "recency_exponent=0.2500" in line, line
    # A fresh dir tracks no shards, so the continuum form is the only honest
    # answer and the line must SAY that rather than imply a measurement.
    assert "source=analytic n_shards=0" in line, line
    # (α+1)/(α+2) at α=0.25.
    assert "mean_rank_pct=0.556" in line, line
    for mass in shard_draw_decile_mass(0.25):
        assert f"{mass:.4f}" in line, (line, mass)


def test_a_resumed_buffer_prints_the_REALIZED_table_not_the_continuum_form(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """MUTATION: drop the ``n_shards > 0`` branch and always print the analytic
    table.

    The construction line's job is to report what the sampler does. Once a
    window exists, the continuum limit is no longer that: at n = 12 each shard
    is 1/12 of the window, the decile bins are ragged, and the realized table
    differs from ``shard_draw_decile_mass`` by ~0.02 in places -- visible at the
    printed precision, so the distinction is not academic. The realized form
    also carries any flooring, which the analytic form structurally cannot.
    """
    shard_dir = tmp_path / "replay"
    seed = _buffer_with_shards(tmp_path, n_shards=12, exponent=0.25)
    seed.close()
    capsys.readouterr()

    reopened = DiskReplayBuffer(
        10**6,
        shard_dir=shard_dir,
        rng=np.random.default_rng(0),
        read_only=True,
        shard_recency_exponent=0.25,
    )
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("[disk_buf] shard draw:")]
    assert len(lines) == 1, out
    line = lines[0]

    n_shards = len(reopened._shard_paths)
    assert n_shards == 12, n_shards
    assert f"source=realized n_shards={n_shards} floored=0" in line, line

    realized = shard_draw_decile_mass_realized(shard_draw_weights(n_shards, 0.25))
    analytic = shard_draw_decile_mass(0.25)
    assert np.max(np.abs(realized - analytic)) > 1e-4, (realized, analytic)
    for mass in realized:
        assert f"{mass:.4f}" in line, (line, mass)
    # The number that would be printed by the branch this test forbids.
    assert "mean_rank_pct=0.556" not in line, line


def test_the_printed_marginal_is_the_without_replacement_mean_at_scale() -> None:
    """The print's one honest-reading caveat, measured instead of asserted.

    MUTATION: none available in production code — this pins a CLAIM the
    docstring of ``_report_shard_recency_draw`` makes, which is exactly the
    kind of statement this repo keeps finding to be false.

    ⚑ The review that surfaced it reported the gap as large and UPWARD (0.821
    realized against 0.667 printed, at n = 40 with 6 picks). Reproducing it
    gave −0.009, i.e. small and DOWNWARD, and the sign is not a detail: a
    without-replacement draw that takes a large fraction of the window is
    pulled toward the window's unweighted mean, because there is little left
    for recency to select among. The docstring now states the measured
    direction; this test keeps it that way.
    """
    rng = np.random.default_rng(0)

    def _gap(n_shards: int, n_pick: int) -> float:
        weights = shard_draw_weights(n_shards, 1.0)
        ranks = np.arange(1, n_shards + 1, dtype=np.float64) / n_shards
        marginal = float((ranks * weights).sum())
        realized = float(np.mean([
            np.mean((rng.choice(n_shards, size=n_pick, replace=False, p=weights) + 1)
                    / n_shards)
            for _ in range(2000)
        ]))
        return realized - marginal

    # Production shape: the printed number IS the draw, to 3 decimals.
    assert abs(_gap(834, 5)) < 0.01
    # Nearly-exhausted window: materially below, never above.
    small = _gap(6, 6)
    assert small < -0.10, small


def test_the_production_construction_site_passes_the_key(tmp_path: Path) -> None:
    """MUTATION: delete ``shard_recency_exponent=tc.replay_shard_recency_exponent``
    from ``trainable_init._init_replay_buffers``.

    The yaml→TrialConfig half and the TrialConfig→buffer half are asserted
    separately, because a key can parse perfectly and still reach no
    constructor -- ``replay_sf_gap_priority_signed`` is the resident example.
    The constructor half is a source assertion on the ONE production call site;
    building the real trainable would need a Ray trial.
    """
    # yaml -> flat config -> TrialConfig, through the real validator.
    import yaml as _yaml

    path = tmp_path / "live.yaml"
    path.write_text(_yaml.safe_dump({"tune": {_KEY: 0.0}}), encoding="utf-8")
    flat = flatten_run_config_defaults(load_yaml_file(str(path)))
    assert flat[_KEY] == 0.0, "the yaml allowlist rejected or dropped the key"
    assert TrialConfig.from_dict(flat).replay_shard_recency_exponent == 0.0

    # TrialConfig -> DiskReplayBuffer, at the production construction site.
    src = (_REPO / "chess_anti_engine" / "tune" / "trainable_init.py").read_text(
        encoding="utf-8",
    )
    assert f"shard_recency_exponent=tc.{_KEY}" in src, (
        "the trial builds its DiskReplayBuffer without the exponent, so the "
        "yaml key would read back correctly from the row and reach nothing"
    )

    # And the buffer keeps what it was handed, on its own attribute.
    buf = DiskReplayBuffer(
        1000,
        shard_dir=tmp_path / "replay",
        rng=np.random.default_rng(0),
        read_only=False,
        shard_recency_exponent=0.0,
    )
    assert buf._shard_recency_exponent == 0.0


# ---------------------------------------------------------------------------
# Reload classification.
# ---------------------------------------------------------------------------


def test_the_key_is_classified_construction_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """MUTATION: remove ``replay_shard_recency_exponent`` from
    ``_CONSTRUCTION_ONLY_REPLAY_KEYS``.

    The exponent reaches ONE constructor that runs once per trial process, and
    nothing re-reads it per iteration. Unclassified, a live yaml edit would
    land in the trial's ``config`` dict -- and therefore in the result row and
    in every tool that reads it -- while the buffer kept drawing at the launch
    value, so the run would REPORT the flipped exponent it was not running.
    That is worse than the edit being ignored, and it is the specific failure
    this classification exists to convert into a warning.

    ``tests/test_construction_only_config_keys.py`` parametrises the general
    guards over the whole set and therefore covers this key too; this test
    names it, so deleting the key from the set fails with the key's own name
    rather than by one parametrisation quietly disappearing.
    """
    assert _KEY in construction_only_config_keys()
    assert _KEY in restart_required_config_keys()

    import yaml as _yaml

    path = tmp_path / "live.yaml"
    path.write_text(_yaml.safe_dump({"tune": {_KEY: 0.0}}), encoding="utf-8")

    config: dict[str, object] = {_KEY: 1.0}
    with caplog.at_level(logging.WARNING):
        _reload_yaml_into_config(config, str(path), live_reload=True)
    assert config[_KEY] == 1.0, "a live edit reached a value only the constructor reads"
    assert [
        r.getMessage() for r in caplog.records
        if _KEY in r.getMessage() and "requires restart" in r.getMessage()
    ], caplog.text

    # Restart is the only way it can change, so startup/resume must still apply.
    startup: dict[str, object] = {_KEY: 1.0}
    _reload_yaml_into_config(startup, str(path), live_reload=False)
    assert startup[_KEY] == 0.0


def test_no_live_config_ships_the_key_yet() -> None:
    """The gating story, asserted rather than promised.

    This PR is preparation: the deciding instrument (the offline rig's arm G,
    on whether recency weighting accelerates forgetting) has not read out, so
    no config carries the key and every run keeps drawing linearly. Flipping it
    is a separate, ledger-pre-registered change at a restart -- and this test
    is what makes "the default preserves behaviour" a statement about the
    repository rather than about one file.
    """
    offenders = sorted(
        p.name for p in (_REPO / "configs").glob("*.yaml")
        if _KEY in p.read_text(encoding="utf-8")
    )
    assert not offenders, (
        f"{_KEY} appears in {offenders}; flipping the shard-recency draw needs "
        "its own ledger entry with a pre-committed yardstick and kill rule, not "
        "a config line riding in on the PR that added the knob"
    )
