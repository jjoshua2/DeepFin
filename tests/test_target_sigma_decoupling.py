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


def test_the_cap_is_exactly_a_smaller_search_budget() -> None:
    """'Trust it like a shallow search' must be literally true, not a fudge.

    The claim in the config comment is that capping max_visit at N gives the
    same sigma a search whose max_visit really was N would have produced. If
    that ever stops holding, the knob's units stop meaning anything and the
    ledger entry describing it becomes wrong.
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

    base = {"selfplay": {"gumbel_target_max_visit_cap": 12, "temperature": 0.0}}
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
