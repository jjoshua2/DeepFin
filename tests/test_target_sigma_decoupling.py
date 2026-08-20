"""``target_max_visit_cap`` softens the STORED target and nothing else.

The knob exists because sigma(q) in ``softmax(log_prior + sigma*Qbar)`` does two
jobs with one number: it sets how far search-Q may outrank the prior for the
move actually PLAYED, and how sharp the row written to the shard is. Those want
different values -- a move the target never puts mass on is a move the next
generation's prior stops proposing, so the loop never revisits it, whereas a
play-time error costs one game. So the strength-optimal sigma is an upper bound
on the training-optimal one.

Three claims, each tested by execution rather than by reading the source:

  1. OFF (0) is bit-identical to the code before the knob existed.
  2. ON softens the stored target, monotonically in how hard the cap bites.
  3. ON does NOT move the played move -- which is also why an arena is
     structurally blind to this knob, and why the yardstick has to be target
     quality against the deep-SF ruler instead.

Claim 3 is the one worth having. "Accepted and then silently ignored" is this
codebase's signature defect, and its mirror image -- a target-only knob that
quietly leaks into play -- would invalidate every strength readout taken while
it was on.
"""
from __future__ import annotations

from typing import Any, cast

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel import (
    GumbelConfig,
    _completed_q_transform,
    _sigma_scale,
)


def _entropy(p: np.ndarray) -> float:
    q = p[p > 0.0]
    return float(-(q * np.log(q)).sum())


# A root with a clear best move by Q but a flat-ish prior, so sigma has
# something to do: shrinking sigma must visibly pull the target back toward the
# prior rather than merely renormalising noise.
_PRIORS = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
_QVALUES = np.array([0.90, 0.10, -0.20, -0.50, -0.80])
_VISITS = np.array([59.0, 20.0, 12.0, 6.0, 3.0])
_ACTIONS = np.arange(5)


def _target(cap: int, *, cfg: GumbelConfig | None = None) -> np.ndarray:
    cfg = cfg or GumbelConfig(c_scale=0.1, c_visit=50.0)
    logits = np.log(_PRIORS) + _completed_q_transform(
        actions=_ACTIONS, priors=_PRIORS, visits=_VISITS, qvalues=_QVALUES,
        raw_value=0.0, cfg=cfg, root=True, max_visit_cap=cap,
    )
    e = np.exp(logits - logits.max())
    return e / e.sum()


def test_the_default_is_off_and_bit_identical() -> None:
    """A new knob must cost nothing until someone sets it."""
    assert GumbelConfig().target_max_visit_cap == 0
    assert _target(0).tolist() == _target(0).tolist()
    # A cap ABOVE the observed max_visit cannot bite, so it must reproduce OFF
    # exactly -- not approximately. This is what pins the clamp to `min`.
    np.testing.assert_array_equal(_target(0), _target(int(_VISITS.max()) + 1))


def test_the_cap_softens_the_target_monotonically() -> None:
    """Harder cap -> smaller sigma -> target closer to the prior -> higher H."""
    caps = [0, 40, 20, 12, 4, 1]
    ents = [_entropy(_target(c)) for c in caps]
    assert ents == sorted(ents), f"entropy not monotone in the cap: {ents}"
    assert ents[-1] > ents[0] + 0.05, (
        f"a cap of 1 barely moved the target (H {ents[0]:.4f} -> {ents[-1]:.4f}); "
        f"the knob is not reaching sigma"
    )
    # And it moves TOWARD the prior, not just toward uniform: KL to the prior
    # must fall. Entropy alone cannot tell those apart.
    def _kl_to_prior(p: np.ndarray) -> float:
        return float((p * (np.log(p) - np.log(_PRIORS))).sum())
    kls = [_kl_to_prior(_target(c)) for c in caps]
    assert kls == sorted(kls, reverse=True), f"KL(target||prior) not falling: {kls}"


def test_the_cap_shrinks_sigma_and_ONLY_sigma() -> None:
    """The cap's units, and the exact boundary of the "shallow search" analogy.

    Capping max_visit at N gives the sigma a search whose max_visit really was N
    would have produced -- that much is exact, and if it stops holding the
    knob's units stop meaning anything.

    ⚑ But "the cap IS a smaller search budget" is FALSE, and the second half of
    this test is what stops that phrasing coming back. A genuinely shallower
    search also changes the visit-weighted ``mixed_value``, hence the value
    imputed to every UNVISITED child. The cap does not touch it. The two agree
    only where the mix-value branch is untaken, i.e. every child visited -- and
    at production's sims/legal-move ratio that is the exception. The knob is
    "search deep, TRUST the result like a shallow search", nothing more.
    """
    cfg = GumbelConfig(c_scale=0.1, c_visit=50.0)
    for n in (1, 12, 30):
        assert _sigma_scale(max_visit=n, cfg=cfg) == pytest.approx(
            0.1 * (50.0 + n),
        )
        capped = _completed_q_transform(
            actions=_ACTIONS, priors=_PRIORS, visits=_VISITS, qvalues=_QVALUES,
            raw_value=0.0, cfg=cfg, root=True, max_visit_cap=n,
        )
        # A search that genuinely peaked at n visits. Every child here is
        # visited, so the mix-value branch is untaken and the normalised
        # completed-Q vector is identical -- the transforms can therefore
        # differ ONLY through sigma, and must not differ at all.
        native = _completed_q_transform(
            actions=_ACTIONS, priors=_PRIORS,
            visits=np.minimum(_VISITS, float(n)), qvalues=_QVALUES,
            raw_value=0.0, cfg=cfg, root=True,
        )
        np.testing.assert_allclose(capped, native, rtol=0.0, atol=0.0)

    # ...and the boundary itself. With two children UNVISITED the mix-value
    # branch is live, and the capped transform is NOT the shallow one. Asserting
    # the DIFFERENCE is what keeps the docstring honest: if someone later makes
    # the cap rewrite `visits` as well, this goes red and they have to decide
    # that deliberately rather than by phrasing.
    unvisited = np.array([59.0, 20.0, 12.0, 0.0, 0.0])
    capped = _completed_q_transform(
        actions=_ACTIONS, priors=_PRIORS, visits=unvisited, qvalues=_QVALUES,
        raw_value=0.0, cfg=cfg, root=True, max_visit_cap=12,
    )
    native = _completed_q_transform(
        actions=_ACTIONS, priors=_PRIORS, visits=np.minimum(unvisited, 12.0),
        qvalues=_QVALUES, raw_value=0.0, cfg=cfg, root=True,
    )
    assert np.abs(capped - native).max() > 1e-3, (
        "the capped and the genuinely-shallow transforms agreed even with "
        "unvisited children -- either the mix-value branch stopped depending on "
        "the visit counts, or the cap grew a second effect"
    )


class _Evaluator:
    """Deterministic stand-in for a net: stable logits for a given batch size."""

    def evaluate_encoded(self, xs: np.ndarray):
        n = int(np.asarray(xs).shape[0])
        rng = np.random.default_rng(1234)
        pol = rng.normal(size=(n, 4672)).astype(np.float32)
        wdl = np.tile(np.array([[0.4, 0.3, 0.3]], np.float32), (n, 1))
        return pol, wdl


_BOARD_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


@pytest.mark.parametrize("path", ["python", "c"])
def test_both_search_paths_honour_the_cap(path: str) -> None:
    """PRODUCTION selfplay runs the C path, so parity here is the real check.

    A knob the Python reference honours and the C path drops is this repo's
    documented failure mode (PY_ONLY_GUMBEL_KNOBS exists because of it). The
    cap happens to be safe -- both paths build the improved policy in the SAME
    Python arithmetic, the C side only supplies the tree -- but "happens to be"
    is not a guarantee, so pin it by running both.
    """
    from chess_anti_engine.mcts.gumbel import run_gumbel_root_many
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c

    runner = run_gumbel_root_many if path == "python" else run_gumbel_root_many_c

    def _run(c: int):
        cfg = GumbelConfig(
            simulations=32, topk=8, c_scale=0.1, temperature=0.0,
            add_noise=False, target_max_visit_cap=c,
        )
        out = runner(
            None, [chess.Board(_BOARD_FEN)], device="cpu",
            rng=np.random.default_rng(7), cfg=cfg,
            evaluator=cast("Any", _Evaluator()),
        )
        return out[0][0], int(out[1][0])

    base_probs, base_action = _run(0)
    # OFF must be bit-identical through a REAL search, not just through the
    # transform in isolation: the two-arm build must not perturb rng draws.
    again_probs, again_action = _run(0)
    np.testing.assert_array_equal(again_probs, base_probs)
    assert again_action == base_action

    probs, action = _run(6)

    assert action == base_action, (
        f"[{path}] the cap changed the PLAYED move ({base_action} -> {action})"
    )
    assert not np.array_equal(probs, base_probs), (
        f"[{path}] the cap changed NOTHING -- it is dropped on this path, which "
        f"is exactly the silent-null shape PY_ONLY_GUMBEL_KNOBS guards against"
    )
    assert _entropy(probs) > _entropy(base_probs), (
        f"[{path}] the cap sharpened the stored target instead of softening it"
    )


# ---------------------------------------------------------------------------
# The limitation, written down and enforced
# ---------------------------------------------------------------------------


# The all-zero temperature block the production yaml pins, as a dict. Every one
# of the ten lines is written out because that is the point: an ABSENT key is
# not 0.0.
_COLD = {
    "temperature": 0.0,
    "temperature_drop_plies": 0,
    "temperature_after": 0.0,
    "temperature_decay_start_move": 20,
    "temperature_decay_moves": 0,
    "temperature_endgame": 0.0,
    "selfplay_temperature": 0.0,
    "selfplay_temperature_decay_start_move": 1,
    "selfplay_temperature_decay_moves": 60,
    "selfplay_temperature_endgame": 0.0,
}


def test_the_cap_is_refused_alongside_a_positive_move_temperature() -> None:
    """The isolation claim holds ONLY at temperature 0, so the pair is refused.

    Found by this repo's own arena coverage guard while building the knob, not
    by reading the code: ``_resample_actions_with_temperature`` re-draws the
    played move from the RETURNED policy, which is the capped one. Production
    runs every move temperature at 0.0 so nothing today can hit it -- but a
    latent trap that silently converts a target-only knob into a search change
    is precisely the shape that invalidates a ledger verdict after the fact.
    """
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    # ⚑ The safe pairing is the WHOLE cold block, not `temperature: 0.0` alone:
    # an omitted `temperature_endgame` realizes 0.6 and an omitted
    # `temperature_decay_moves` realizes 60, so `{"temperature": 0.0}` on its own
    # is a config that plays at 0.6 from move 80 onward. This test used to assert
    # that config LOADED.
    base = {"selfplay": {**_COLD, "gumbel_target_max_visit_cap": 12}}
    flatten_run_config_defaults(base)  # the safe pairing must still load

    with pytest.raises(ValueError, match="positive move temperature"):
        flatten_run_config_defaults(
            {"selfplay": {"gumbel_target_max_visit_cap": 12, "temperature": 0.35}},
        )
    # ...and the cap being OFF must leave a hot temperature alone, or the guard
    # is just a temperature ban wearing a different name.
    flatten_run_config_defaults(
        {"selfplay": {"gumbel_target_max_visit_cap": 0, "temperature": 0.35}},
    )


def _realized_max_temperature(flat: dict[str, object]) -> float:
    """The hottest ply the RUN would actually play, from the loader's own values.

    Deliberately routed through ``TrialConfig.from_dict`` and
    ``temperature_for_ply`` -- the two things the run itself uses -- rather than
    re-reading ``flat``. A guard has to share the criterion's instrument, so the
    test that judges the guard has to share it too, or it is just a second copy
    of the guard agreeing with the first.
    """
    from chess_anti_engine.selfplay.temperature import temperature_for_ply
    from chess_anti_engine.tune.trial_config import TrialConfig

    tc = TrialConfig.from_dict(dict(flat))
    hottest = 0.0
    # Both arms: curriculum games read `temperature*`, pure-selfplay games read
    # `selfplay_temperature*` and fall back to the former when it is None.
    for sp in (False, True):
        for ply in range(1, 121):
            hottest = max(hottest, temperature_for_ply(
                ply=ply,
                temperature=(
                    float(tc.selfplay_temperature)
                    if sp and tc.selfplay_temperature is not None
                    else float(tc.temperature)
                ),
                drop_plies=int(tc.temperature_drop_plies),
                after=float(tc.temperature_after),
                decay_start_move=(
                    int(tc.selfplay_temperature_decay_start_move)
                    if sp and tc.selfplay_temperature_decay_start_move is not None
                    else int(tc.temperature_decay_start_move)
                ),
                decay_moves=(
                    int(tc.selfplay_temperature_decay_moves)
                    if sp and tc.selfplay_temperature_decay_moves is not None
                    else int(tc.temperature_decay_moves)
                ),
                endgame=(
                    float(tc.selfplay_temperature_endgame)
                    if sp and tc.selfplay_temperature_endgame is not None
                    else float(tc.temperature_endgame)
                ),
            ))
    return hottest


@pytest.mark.parametrize(
    "selfplay",
    [
        # ⚑⚑ THE REGRESSION. A yaml that simply OMITS the temperature block --
        # the deletion `configs/pbt2_small.yaml` warns about in its own comment
        # -- realizes temperature 1.0 / endgame 0.6 / decay_moves 60, i.e. EVERY
        # ply is re-drawn from the stored (capped) policy. The guard used to
        # read the absent keys as 0.0 and wave it through.
        pytest.param({}, id="no temperature block at all"),
        # ...and the same for each key on its own, on top of the cold block, so
        # shortening `_MOVE_TEMPERATURE_KEYS` cannot pass either.
        pytest.param({**_COLD, "temperature": 0.35}, id="temperature"),
        pytest.param(
            {**_COLD, "temperature_drop_plies": 4, "temperature_after": 0.35},
            id="temperature_after",
        ),
        pytest.param(
            {**_COLD, "temperature_decay_moves": 60, "temperature_endgame": 0.35},
            id="temperature_endgame",
        ),
        pytest.param({**_COLD, "selfplay_temperature": 0.35}, id="selfplay_temperature"),
        pytest.param(
            {**_COLD, "selfplay_temperature_decay_moves": 60,
             "selfplay_temperature_endgame": 0.35},
            id="selfplay_temperature_endgame",
        ),
        # The cold block itself must stay ACCEPTED, or the guard is a
        # temperature ban and production could not run the cap at all.
        pytest.param(dict(_COLD), id="the production cold block"),
    ],
)
def test_the_guard_agrees_with_the_temperature_the_run_would_actually_play(
    selfplay: dict[str, object],
) -> None:
    """⟺, not ⟸. The guard must refuse EXACTLY the configs that are hot.

    Both halves matter and each catches a different mutant:

    * refuse-when-hot catches a SHORTENED ``_MOVE_TEMPERATURE_KEYS`` and the
      absent-key default reading 0.0 when the loader reads 1.0;
    * accept-when-cold catches a guard that just bans the cap.

    The hot/cold verdict is computed from ``TrialConfig.from_dict`` +
    ``temperature_for_ply``, so this cannot be satisfied by a second copy of the
    guard's own arithmetic.
    """
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    with_cap = {"selfplay": {**selfplay, "gumbel_target_max_visit_cap": 12}}
    without = {"selfplay": dict(selfplay)}

    # The cap being OFF must never refuse anything, hot or cold.
    hot = _realized_max_temperature(flatten_run_config_defaults(without))

    if hot > 0.0:
        with pytest.raises(ValueError, match="positive move temperature"):
            flatten_run_config_defaults(with_cap)
    else:
        flatten_run_config_defaults(with_cap)


def test_the_guards_absent_key_defaults_are_the_loaders_defaults() -> None:
    """Pin the table against ``TrialConfig.from_dict``, key by key.

    The table above is a hand-written copy of the loader's defaults, and a
    hand-written copy is exactly the thing that drifts. This reads each default
    back out of the loader on an EMPTY config, so changing one there without
    changing it here goes red.
    """
    from chess_anti_engine.tune.trial_config import TrialConfig
    from chess_anti_engine.utils.config_yaml import _MOVE_TEMPERATURE_KEYS

    empty = TrialConfig.from_dict({})
    for key, absent in _MOVE_TEMPERATURE_KEYS:
        realized = getattr(empty, key)
        if absent is None:
            assert realized is None, (
                f"{key} is listed as 'absent means fall back to another key', "
                f"but the loader realizes it as {realized!r}"
            )
        else:
            assert float(realized) == pytest.approx(absent), (
                f"{key}: the guard assumes an absent key realizes as {absent}, "
                f"the loader realizes {realized!r}"
            )


def test_production_leaves_the_cap_off() -> None:
    """Merging this must not change the live run.

    #390 pins its own knob this way; without the same test here, merging THIS
    branch on its own leaves nothing asserting that the committed production
    config still runs the cap at 0. The ledger entry decides when it goes on,
    and CLAUDE.md rule 1 says no entry, no launch -- and the entry records that
    the one screen taken so far is NEGATIVE.
    """
    from pathlib import Path

    from chess_anti_engine.utils.config_yaml import (
        flatten_run_config_defaults,
        load_yaml_file,
    )

    repo = Path(__file__).resolve().parents[1]
    flat = flatten_run_config_defaults(
        load_yaml_file(str(repo / "configs" / "pbt2_small.yaml")),
    )
    assert int(flat.get("gumbel_target_max_visit_cap", 0) or 0) == 0

# ---------------------------------------------------------------------------
# target_q_rescale: sigma's UNITS (mctx rescale_values), store-arm only
# ---------------------------------------------------------------------------


def test_rescale_default_is_on_and_matches_mctx_exactly() -> None:
    """True is the default, and the default path IS the mctx formula, verified
    against an in-test reimplementation rather than a banked golden."""
    assert GumbelConfig().target_q_rescale is True
    cfg = GumbelConfig(c_scale=0.1, c_visit=50.0)
    got = _completed_q_transform(
        actions=_ACTIONS, priors=_PRIORS, visits=_VISITS, qvalues=_QVALUES,
        raw_value=0.0, cfg=cfg, root=True,
    )
    completed = _QVALUES.astype(np.float64)  # every child visited
    norm = (completed - completed.min()) / max(
        completed.max() - completed.min(), 1e-8,
    )
    expected = 0.1 * (50.0 + _VISITS.max()) * norm
    np.testing.assert_array_equal(got, expected)


def test_absolute_q_gives_authority_proportional_to_the_real_gap() -> None:
    """THE indictment, stated as an assertion: under the rescale, a decisive Q
    spread and 1/100th of it produce the IDENTICAL sigma vector -- noise gets
    the full c_scale*(c_visit+max_visit) logits. Absolute units restore
    proportionality exactly."""
    cfg = GumbelConfig(c_scale=0.1, c_visit=50.0)
    noise_q = _QVALUES * 0.01  # same pattern, 1/100th the real distinction

    def _sigma(q: np.ndarray, *, rescale: bool) -> np.ndarray:
        return _completed_q_transform(
            actions=_ACTIONS, priors=_PRIORS, visits=_VISITS, qvalues=q,
            raw_value=0.0, cfg=cfg, root=True, rescale=rescale,
        )

    # mathematically identical (min-max wipes the scale); ulp-level noise from
    # the 0.01 factor is not the claim, so allclose rather than array_equal
    np.testing.assert_allclose(
        _sigma(_QVALUES, rescale=True), _sigma(noise_q, rescale=True),
        rtol=1e-12,
    )
    sharp = _sigma(_QVALUES, rescale=False)
    noisy = _sigma(noise_q, rescale=False)
    np.testing.assert_allclose(noisy, sharp * 0.01, rtol=1e-12)
    # and the absolute arm keeps sigma in Q units: the full-spread vector spans
    # exactly scale * (max_q - min_q), not scale * 1.0
    span = float(sharp.max() - sharp.min())
    assert span == pytest.approx(0.1 * (50.0 + _VISITS.max()) * 1.7)


@pytest.mark.parametrize("path", ["python", "c"])
def test_rescale_off_is_store_only_on_both_paths(path: str) -> None:
    """Same contract as the cap: the flag must change the STORED row and must
    not move the played move, on the path production actually runs.

    Run at cap=0 deliberately: the off-arm shortcut reuses q_play whenever the
    cap alone is off, so this leg goes red if the reuse condition forgets
    `target_q_rescale` -- the accepted-then-ignored shape, killed by execution.
    """
    from chess_anti_engine.mcts.gumbel import run_gumbel_root_many
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c

    runner = run_gumbel_root_many if path == "python" else run_gumbel_root_many_c

    def _run(rescale: bool):
        cfg = GumbelConfig(
            simulations=32, topk=8, c_scale=0.1, temperature=0.0,
            add_noise=False, target_max_visit_cap=0, target_q_rescale=rescale,
        )
        out = runner(
            None, [chess.Board(_BOARD_FEN)], device="cpu",
            rng=np.random.default_rng(7), cfg=cfg,
            evaluator=cast("Any", _Evaluator()),
        )
        return out[0][0], int(out[1][0])

    base_probs, base_action = _run(True)
    probs, action = _run(False)

    assert action == base_action, (
        f"[{path}] target_q_rescale changed the PLAYED move "
        f"({base_action} -> {action})"
    )
    assert not np.array_equal(probs, base_probs), (
        f"[{path}] target_q_rescale=False changed NOTHING -- the q_play reuse "
        f"shortcut swallowed it (accepted-then-ignored)"
    )
    assert _entropy(probs) > _entropy(base_probs), (
        f"[{path}] absolute-Q sigma sharpened the stored target; at 32 sims the "
        f"real Q spread carries less authority than the rescaled one, so the "
        f"target must move toward the prior"
    )
