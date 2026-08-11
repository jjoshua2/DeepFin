"""``target_untempered_prior`` takes ``policy_temp`` out of the STORED target.

The Gumbel improved policy is ``softmax(log_prior + sigma*Qbar)``, and
``log_prior`` is the TEMPERED prior: ``apply_policy_temp`` divides the policy
logits by ``policy_temp`` before they seed the tree, and the improved-policy
build then reads ``log(pri)`` off that same tempered prior. Production runs
``gumbel_policy_temp: 1.5``, and the stored improved policy is the next
generation's TRAINING TARGET, so the division compounds through the loop::

    L -> L/T -> target = L/T + sigma*Qbar -> the head learns it -> L/T again

whose fixed point is ``L* = sigma*Qbar * T/(T-1)`` -- ``3*sigma*Qbar`` at
T=1.5. The policy head's independently learned information is discounted
geometrically toward a target determined ENTIRELY by Q.

lc0 runs PolicyTemperature 1.45 safely because it trains on VISIT COUNTS, which
carry no additive ``log_prior`` term; Gumbel-MuZero's completed-Q target does.
Same knob, different target algebra, opposite safety.

The fix keeps the tempered prior for CANDIDATE SELECTION -- putting more moves
into the search is what temperature is actually for -- and undoes it in the
stored target's ``log_prior`` term only.

What is tested here, all by execution:

  1. OFF is bit-identical, on both search paths.
  2. The reconstruction is EXACT, checked against the raw logits rather than
     against the algebra that produced it.
  3. ON measurably changes the STORED target on BOTH paths. A knob one path
     honours and the other drops is this repo's documented failure mode.
  4. ON does NOT change the played move -- including at a positive move
     temperature INSIDE the search, where the sample is drawn from the
     unmodified arm. The one place it would leak (the re-draw in
     ``network_turn``) is refused at config load, and that refusal is tested.
  5. The mechanism itself. ⚑ NOT as "``KL(prior||target)`` falls": that is
     regime-dependent and is pinned BOTH ways here, because the target's
     displacement from the prior is ``S`` with the fix and ``S - (1-1/T)L``
     without it, so removing the tempering closes the gap only where search
     DISAGREES with the prior. What is unconditional is the sigma=0 case --
     with no search information the target must BE the head's policy, and
     without the fix it is the head's policy divided by T -- plus the loop
     map's contraction ratio, which is 1/T without the fix and 1 with it.
"""
from __future__ import annotations

from typing import Any, cast

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel import (
    GumbelConfig,
    _softmax,
    target_log_prior,
)

_T = 1.5  # production's gumbel_policy_temp


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p||q) over a shared support, both renormalised."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    keep = p > 0.0
    return float((p[keep] * (np.log(p[keep]) - np.log(np.maximum(q[keep], 1e-300)))).sum())


# ---------------------------------------------------------------------------
# 1 + 2. The helper: off is identity, and on is EXACT against the raw logits
# ---------------------------------------------------------------------------


def test_the_default_is_off_and_the_helper_returns_the_same_object() -> None:
    """A new knob must cost nothing until someone sets it.

    Object identity, not equality: both call sites branch on ``is`` to skip the
    second softmax entirely, so an equal-but-new array would silently cost a
    softmax per root on the production path.
    """
    assert GumbelConfig().target_untempered_prior is False
    log_prior = np.log(np.array([0.5, 0.3, 0.2]))
    cfg = GumbelConfig(policy_temp=_T)
    assert target_log_prior(log_prior, cfg=cfg) is log_prior


@pytest.mark.parametrize("temp", [1.0, 0.0, -1.0, 1e300, 1e-300, float("nan"), float("inf")])
def test_the_helper_never_undoes_a_temperature_that_was_not_applied(temp: float) -> None:
    """The gate must be ``policy_temp_active``, not ``!= 1.0`` or ``isfinite``.

    ``apply_policy_temp`` leaves the priors UNTOUCHED for every value here, so
    "undoing" one would multiply the log-prior by that value -- 1e300 turns the
    prior into a one-hot and nothing would raise. A guard that does not share
    the criterion's instrument is guarding a different question.
    """
    log_prior = np.log(np.array([0.5, 0.3, 0.2]))
    cfg = GumbelConfig(policy_temp=temp, target_untempered_prior=True)
    assert target_log_prior(log_prior, cfg=cfg) is log_prior


def test_the_untempered_prior_is_exact_against_the_raw_logits() -> None:
    """The claim is EXACT recovery, not an approximation -- so check the source.

    The implementation multiplies the tempered log-prior by T rather than
    plumbing the raw logits down to the improved-policy sites. That is exact
    because tempering is literally ``logits / T`` and the improved policy is a
    softmax, which absorbs the constant the algebra leaves behind. This test
    refuses to take that on trust: it starts from raw logits, tempers them the
    way the search does, and asserts the recovered distribution equals the
    softmax of the ORIGINAL logits to float64.
    """
    rng = np.random.default_rng(11)
    for _ in range(20):
        raw = rng.normal(scale=4.0, size=17)
        # exactly what `_masked_priors` / the gumbel_c root loop build
        tempered = raw / _T
        tempered = tempered - tempered.max()
        pri = np.exp(tempered)
        pri /= pri.sum()

        log_prior = np.log(np.maximum(pri, 1e-12))
        cfg = GumbelConfig(policy_temp=_T, target_untempered_prior=True)
        recovered = _softmax(target_log_prior(log_prior, cfg=cfg))

        np.testing.assert_allclose(recovered, _softmax(raw), rtol=1e-12, atol=1e-14)


def test_undoing_the_temperature_sharpens_toward_the_head() -> None:
    """T > 1 softens, so undoing it must SHARPEN -- direction, not just change."""
    rng = np.random.default_rng(3)
    raw = rng.normal(scale=3.0, size=25)
    pri = _softmax(raw / _T)
    log_prior = np.log(pri)
    cfg = GumbelConfig(policy_temp=_T, target_untempered_prior=True)
    untempered = _softmax(target_log_prior(log_prior, cfg=cfg))

    def _h(p: np.ndarray) -> float:
        return float(-(p * np.log(p)).sum())

    assert _h(untempered) < _h(pri)


# ---------------------------------------------------------------------------
# 3 + 4 + 5. Through a REAL search, on both paths
# ---------------------------------------------------------------------------


class _Evaluator:
    """Deterministic stand-in for a net; records the logits it handed back.

    Recording matters: the mechanism test needs the UNTEMPERED prior, and the
    only honest source for it is the logits the model actually produced -- not
    a value re-derived from the same arithmetic the code under test uses.
    """

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def evaluate_encoded(self, xs: np.ndarray, **_kw: object):
        n = int(np.asarray(xs).shape[0])
        rng = np.random.default_rng(1234)
        pol = rng.normal(scale=2.0, size=(n, 4672)).astype(np.float32)
        wdl = np.tile(np.array([[0.4, 0.3, 0.3]], np.float32), (n, 1))
        self.calls.append(pol.copy())
        return pol, wdl


_BOARD_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


def _run(
    path: str,
    *,
    untempered: bool,
    policy_temp: float = _T,
    temperature: float = 0.0,
    c_scale: float = 0.1,
) -> tuple[np.ndarray, int, np.ndarray]:
    """One search; returns (stored policy, played action, the ROOT raw logits)."""
    from chess_anti_engine.mcts.gumbel import run_gumbel_root_many
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c

    runner = run_gumbel_root_many if path == "python" else run_gumbel_root_many_c
    ev = _Evaluator()
    cfg = GumbelConfig(
        simulations=32, topk=8, c_scale=c_scale, temperature=temperature,
        add_noise=False, policy_temp=policy_temp,
        target_untempered_prior=untempered,
    )
    out = runner(
        None, [chess.Board(_BOARD_FEN)], device="cpu",
        rng=np.random.default_rng(7), cfg=cfg,
        evaluator=cast("Any", ev),
    )
    # The root is the FIRST evaluation on both paths (phase 1 resolves the root
    # logits before any leaf batch exists).
    return out[0][0], int(out[1][0]), ev.calls[0][0]


@pytest.mark.parametrize("path", ["python", "c"])
def test_off_is_bit_identical_and_on_moves_the_stored_target(path: str) -> None:
    """PRODUCTION selfplay runs the C path, so parity here is the real check."""
    base_probs, base_action, _ = _run(path, untempered=False)
    again_probs, again_action, _ = _run(path, untempered=False)
    np.testing.assert_array_equal(again_probs, base_probs)
    assert again_action == base_action

    probs, action, _ = _run(path, untempered=True)

    assert not np.array_equal(probs, base_probs), (
        f"[{path}] the knob changed NOTHING -- it is dropped on this path, "
        f"which is exactly the silent-null shape PY_ONLY_GUMBEL_KNOBS guards "
        f"against"
    )
    assert action == base_action, (
        f"[{path}] the knob changed the PLAYED move ({base_action} -> {action})"
    )
    # H(stored target) is the quantity the live ramp moved (0.9849 -> 1.2136
    # against a pre-registered 1.14 bar), so the direction is pinned at
    # production's sigma too. Exactness lives in the sigma=0 test; here it is an
    # observation that the sign is what the mechanism predicts.
    legal = np.flatnonzero(base_probs > 0.0)

    def _h(p: np.ndarray) -> float:
        q = np.asarray(p, dtype=np.float64)
        q = q[q > 0.0] / q.sum()
        return float(-(q * np.log(q)).sum())

    assert _h(probs[legal]) < _h(base_probs[legal]), (
        f"[{path}] undoing a T>1 division on the prior term SOFTENED the "
        f"stored target ({_h(base_probs[legal]):.4f} -> {_h(probs[legal]):.4f})"
    )


@pytest.mark.parametrize("path", ["python", "c"])
def test_the_played_move_is_unchanged_at_a_positive_search_temperature(path: str) -> None:
    """The isolation is structural INSIDE the search, not an artifact of T=0.

    At ``temperature > 0`` the search samples its own action from ``imp_all``,
    the arm the knob does not touch, so the played move must still be identical
    under the same seed. (The separate leak -- ``network_turn`` re-drawing from
    the RETURNED policy -- is refused at config load; see the test below. These
    are two different code paths and only one of them lives here.)
    """
    off_probs, off_action, _ = _run(path, untempered=False, temperature=0.7)
    on_probs, on_action, _ = _run(path, untempered=True, temperature=0.7)

    assert on_action == off_action, (
        f"[{path}] the target-only knob moved the sampled move at temperature "
        f"0.7 ({off_action} -> {on_action}); it is not target-only"
    )
    assert not np.array_equal(on_probs, off_probs)


@pytest.mark.parametrize("path", ["python", "c"])
def test_the_knob_is_inert_when_there_is_no_temperature_to_undo(path: str) -> None:
    """Negative control. At policy_temp 1.0 there is nothing to undo, so the
    stored row must be byte-identical -- otherwise the "change" measured above
    is the knob doing something other than removing the temperature."""
    off_probs, _off_a, _ = _run(path, untempered=False, policy_temp=1.0)
    on_probs, _on_a, _ = _run(path, untempered=True, policy_temp=1.0)
    np.testing.assert_array_equal(on_probs, off_probs)


@pytest.mark.parametrize("path", ["python", "c"])
def test_with_no_search_information_the_target_is_the_head_itself(path: str) -> None:
    """THE mechanism test, in the one regime where it is unconditional.

    ``KL(prior||target)`` is the stored column that rose 11.8x at the median
    (``_mcts_tree.c:4838`` / ``network_turn.py:641``; ``prior`` there is the
    softmax of the RAW model logits over the legal moves, which is what this
    test uses -- taken from the recorded evaluator output, so the test cannot
    pass by agreeing with the code under test).

    Set ``c_scale = 0`` and sigma(q) is exactly 0: the search contributes NO
    information, so an honest target is the head's own policy and
    ``KL(prior||target)`` must be 0. With the knob OFF it is NOT -- the stored
    target is the head's policy divided by T, i.e. the target moved away from
    the prior for a reason that has nothing to do with search, and that
    displacement is precisely the term that compounds through the loop. With
    the knob ON the KL is zero to float32.

    This is the isolation of the defect. See
    ``test_the_kl_direction_depends_on_whether_search_agrees_with_the_prior``
    for why the same comparison at production's sigma is NOT sign-definite --
    that is a real property of the quantity, not a shortcoming of the fix.
    """
    off_probs, _a_off, raw = _run(path, untempered=False, c_scale=0.0)
    on_probs, _a_on, raw_on = _run(path, untempered=True, c_scale=0.0)
    np.testing.assert_array_equal(raw, raw_on)  # same tree, same net

    legal = np.flatnonzero(off_probs > 0.0)
    assert legal.size > 1
    np.testing.assert_array_equal(legal, np.flatnonzero(on_probs > 0.0))

    prior = _softmax(raw[legal].astype(np.float64))
    kl_off = _kl(prior, off_probs[legal])
    kl_on = _kl(prior, on_probs[legal])

    assert kl_on < 1e-6, (
        f"[{path}] with sigma=0 the stored target should BE the head's policy, "
        f"but KL(prior||target) = {kl_on:.3e}"
    )
    assert kl_off > 0.05, (
        f"[{path}] the OFF arm should be displaced from the prior by the whole "
        f"of the 1/T division, but KL(prior||target) = {kl_off:.4f}"
    )
    # ...and the displacement is exactly the tempering, not something else.
    np.testing.assert_allclose(
        _kl(prior, _softmax(raw[legal].astype(np.float64) / _T)),
        kl_off, rtol=2e-3,
    )
    # The OFF target is also strictly SOFTER, which is the direction the live
    # H(policy_target) ramp moved (0.9849 -> 1.2136 against a 1.14 bar).
    def _h(p: np.ndarray) -> float:
        q = np.asarray(p, dtype=np.float64)
        q = q[q > 0.0] / q.sum()
        return float(-(q * np.log(q)).sum())

    assert _h(off_probs[legal]) > _h(on_probs[legal]) + 0.05


def test_the_loop_map_contracts_the_heads_own_information_only_when_off() -> None:
    """The compounding claim itself, as arithmetic rather than as a story.

    Treat one generation as: the head learns the stored target, so its next
    logits are that target's log, and the next search adds the same ``S =
    sigma*Qbar`` on top::

        OFF   L_{n+1} = L_n / T + S
        ON    L_{n+1} = L_n     + S

    OFF is a contraction with ratio 1/T on EVERYTHING the head knows that the
    search did not tell it, so that component decays geometrically to zero and
    the map has a fixed point ``L* = S * T/(T-1)`` -- at T=1.5, ``3*S``, a
    policy determined ENTIRELY by Q. ON has ratio 1: the head's own information
    is carried, not discounted.

    This is what makes ``policy_temp`` safe in lc0 and unsafe here. lc0 trains
    on VISIT COUNTS, which have no additive ``log_prior`` term, so no version of
    this recursion exists there.
    """
    rng = np.random.default_rng(5)
    s = rng.normal(size=40)
    s -= s.mean()
    l0 = rng.normal(scale=3.0, size=40)
    l0 -= l0.mean()
    # The head's OWN information is what the search did not supply, so make it
    # literally orthogonal to S -- otherwise "the head's contribution" and "the
    # search's contribution" are not separable and the coefficient below would
    # be measuring a mixture.
    l0 -= (l0 @ s) / (s @ s) * s

    def _own(vec: np.ndarray) -> float:
        """How much of L0 survives in ``vec``, with S's contribution removed."""
        resid = vec - (vec @ s) / (s @ s) * s
        return float(resid @ l0 / (l0 @ l0))

    off = l0.copy()
    on = l0.copy()
    for n in range(1, 31):
        off = off / _T + s
        on = on + s
        off -= off.mean()
        on -= on.mean()
        assert _own(off) == pytest.approx(_T ** -n, rel=1e-6)
        assert _own(on) == pytest.approx(1.0, rel=1e-9)

    # ...and OFF has converged onto a pure function of Q at the predicted scale.
    np.testing.assert_allclose(off, s * _T / (_T - 1.0), rtol=1e-4, atol=1e-4)


def test_the_kl_direction_depends_on_whether_search_agrees_with_the_prior() -> None:
    """⚑ Written down because the obvious acceptance test is WRONG as stated.

    "With the fix on, ``KL(prior||target)`` must be lower" is not a theorem. In
    logit space the target's displacement from the prior is ``S`` with the fix
    on and ``S - (1 - 1/T)*L`` with it off, so removing the tempering moves the
    target CLOSER to the prior only when the search DISAGREES with the prior. On
    rows where search agrees -- the majority, which is why the live median KL is
    only 0.0238 -- the ``-(1-1/T)L`` term was cancelling part of ``S``, and
    taking it away moves the target FURTHER from the prior.

    Both directions are pinned here so nobody later "fixes" the sign by making
    the knob do something other than remove the temperature. The unconditional
    statement of the mechanism is the sigma=0 test above; the deciding
    measurement for whether the change is GOOD is target quality against the
    frozen deep-SF audit set, never this KL. Change RATE is not change QUALITY
    -- the ``sims=1`` control changes 34.9% of moves and is +7.60 cp WORSE.
    """
    l = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
    prior = _softmax(l)

    def _kls(s: np.ndarray) -> tuple[float, float]:
        return _kl(prior, _softmax(l / _T + s)), _kl(prior, _softmax(l + s))

    agree_off, agree_on = _kls(np.array([3.0, 1.5, 0.0, 0.0, 0.0]))
    assert agree_on > agree_off

    disagree_off, disagree_on = _kls(np.array([0.0, 0.0, 0.0, 1.5, 3.0]))
    assert disagree_on < disagree_off


# ---------------------------------------------------------------------------
# The limitation, written down and enforced
# ---------------------------------------------------------------------------


def test_it_is_refused_alongside_a_positive_move_temperature() -> None:
    """Same exposure, same refusal as ``gumbel_target_max_visit_cap``.

    ``_resample_actions_with_temperature`` re-draws the played move from the
    RETURNED policy, which is the modified one. Production runs every move
    temperature at 0.0, so this refuses a pairing nobody runs rather than
    breaking one somebody does -- and it refuses rather than clamping, because
    a clamp re-creates the accepted-and-silently-ignored shape.
    """
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    from tests.test_target_sigma_decoupling import _COLD

    # ⚑ The safe pairing is the WHOLE cold block, not `temperature: 0.0` alone.
    # An ABSENT key is not 0.0 -- it realizes as the loader's default, so an
    # omitted `temperature_endgame` is 0.6 and an omitted
    # `temperature_decay_moves` is 60. Imported from the parent knob's suite
    # rather than copied, because two hand-maintained copies of the cold block
    # is how one of them ends up wrong.
    flatten_run_config_defaults(
        {"selfplay": {**_COLD, "gumbel_target_untempered_prior": True}},
    )

    with pytest.raises(ValueError, match="positive move temperature"):
        flatten_run_config_defaults(
            {"selfplay": {**_COLD, "gumbel_target_untempered_prior": True,
                          "temperature": 0.35}},
        )
    # ...and OFF must leave a hot temperature alone, or the guard is just a
    # temperature ban wearing a different name.
    flatten_run_config_defaults(
        {"selfplay": {**_COLD, "gumbel_target_untempered_prior": False,
                      "temperature": 0.35}},
    )


def test_the_flag_is_refused_on_a_yaml_that_simply_OMITS_the_temperature_block() -> None:
    """The absent-key half of the guard, for THIS knob rather than the cap's.

    ``flatten_run_config_defaults`` copies only the keys the yaml contains, so a
    config with no ``temperature*`` lines at all realizes ``temperature = 1.0``,
    ``temperature_endgame = 0.6`` and ``temperature_decay_moves = 60`` -- every
    ply re-drawn from the stored (untempered-prior) policy. The guard has to
    refuse it. See ``test_target_sigma_decoupling`` for the ⟺ table that pins
    every key and every default against the loader.
    """
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    with pytest.raises(ValueError, match="positive move temperature"):
        flatten_run_config_defaults(
            {"selfplay": {"gumbel_target_untempered_prior": True}},
        )


def test_the_shared_guard_uses_each_knobs_own_coercion() -> None:
    """The two target-only knobs now share one guard; they must not share one
    truthiness test.

    ``TrialConfig.from_dict`` realizes the cap as ``max(0, int(v))``, so a yaml
    ``0.5`` is OFF -- refusing it alongside a hot temperature would reject a
    pairing the run was never going to be in. The flag realizes as ``bool(v)``,
    so ``0.5`` is ON. A guard has to share the criterion's instrument, and here
    that means one coercion per knob rather than a generic ``> 0``.
    """
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    flatten_run_config_defaults(
        {"selfplay": {"gumbel_target_max_visit_cap": 0.5, "temperature": 0.35}},
    )
    with pytest.raises(ValueError, match="positive move temperature"):
        flatten_run_config_defaults(
            {"selfplay": {
                "gumbel_target_untempered_prior": 0.5, "temperature": 0.35,
            }},
        )


def test_production_leaves_the_knob_off() -> None:
    """Merging this must not change the live run. The ledger entry decides when
    it goes on, and CLAUDE.md rule 1 says no entry, no launch."""
    from pathlib import Path

    from chess_anti_engine.utils.config_yaml import (
        flatten_run_config_defaults,
        load_yaml_file,
    )

    repo = Path(__file__).resolve().parents[1]
    flat = flatten_run_config_defaults(
        load_yaml_file(str(repo / "configs" / "pbt2_small.yaml")),
    )
    assert not flat.get("gumbel_target_untempered_prior", False)
