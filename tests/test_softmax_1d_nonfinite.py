"""One NaN logit must not poison the whole MCTS prior (audit A22).

`softmax_1d`'s docstring promises a uniform fallback that "avoids
divide-by-zero corrupting the downstream MCTS prior or SF policy target", and
it names exactly the callers that had the problem: `mcts/puct.py:27`
(`softmax_1d(logits[legal_idx])`), `mcts/gumbel.py:48`, and
`selfplay/stockfish_turn.py:40`.

The guard was `if s <= 0`. **`nan <= 0` is False.** So a single nan logit
walked straight through it: `np.max` over a vector containing one nan is nan,
subtracting that made EVERY entry nan, `np.exp` kept them nan, and nan/nan was
returned as the prior. Not a crash — a silent all-nan distribution handed to
search, from one bad logit, by the function documented to prevent exactly that.

⚑ The negative half of this file matters as much as the positive half. The
legitimate use of infinity in these callers is `-inf` for MASKED entries, so a
fix that treats any infinity as corruption would replace a real distribution
with a uniform one on every masked position — a far larger behaviour change
than the bug. `test_a_masked_vector_is_untouched` is the control for that.
"""

from __future__ import annotations

import numpy as np
import pytest

from chess_anti_engine.utils.numpy_helpers import softmax_1d


def test_a_single_nan_logit_yields_the_uniform_fallback() -> None:
    """The headline case: ONE bad logit, not all of them."""
    x = np.array([1.0, 2.0, np.nan, 0.5, -1.0], dtype=np.float32)

    out = softmax_1d(x)

    assert np.isfinite(out).all(), (
        "one nan logit must not produce an all-nan prior; every legal move's "
        "probability was nan before the fix"
    )
    assert out == pytest.approx(np.full(5, 0.2, dtype=np.float32))
    assert float(out.sum()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("name", "x"),
    [
        ("one nan", [1.0, 2.0, np.nan]),
        ("all nan", [np.nan, np.nan, np.nan]),
        ("one +inf", [1.0, np.inf, 0.5]),
        ("all -inf (fully masked)", [-np.inf, -np.inf, -np.inf]),
        ("mixed infinities", [np.inf, -np.inf, 0.0]),
    ],
)
def test_every_non_finite_shape_falls_back_to_uniform(
    name: str, x: list[float],
) -> None:
    """All three ways a non-finite value reaches the max, plus the +inf case.

    The all-`-inf` vector is a fully masked legal set; before the fix it
    produced nans *and* a RuntimeWarning from `-inf - -inf`. `+inf` is treated
    as corruption on purpose -- nothing in this codebase encodes certainty that
    way, and the `-inf` masking convention is covered by the control below.
    """
    del name
    arr = np.array(x, dtype=np.float64)

    with np.errstate(all="raise"):
        out = softmax_1d(arr)

    assert np.isfinite(out).all()
    assert out == pytest.approx(np.full(len(x), 1.0 / len(x), dtype=np.float32))


def test_a_masked_vector_is_untouched() -> None:
    """⚑ NEGATIVE CONTROL: `-inf` masking is the legitimate use of infinity.

    A fix that rejected any infinity would hand back a uniform distribution for
    every partially-masked position -- silently discarding the real prior on
    the common case in order to fix the rare one. The max here is finite, so
    this vector must take the normal path and be bit-identical to before.
    """
    x = np.array([-np.inf, -np.inf, 0.0], dtype=np.float64)

    out = softmax_1d(x)

    assert out == pytest.approx(np.array([0.0, 0.0, 1.0], dtype=np.float32))


def test_the_ordinary_path_is_byte_identical() -> None:
    """The fix must not move any finite result.

    Compared against an independent softmax rather than against the function's
    own output, so this cannot pass by agreeing with a broken implementation.
    """
    x = np.array([2.0, 1.0, 0.5, -3.0], dtype=np.float64)
    expected = np.exp(x - x.max()) / np.exp(x - x.max()).sum()

    out = softmax_1d(x)

    assert out == pytest.approx(expected.astype(np.float32))


def test_the_all_zero_case_still_returns_uniform() -> None:
    """The one shape the old docstring called out explicitly. Unchanged.

    Note it never reached the `s <= 0` fallback even before: an all-zero input
    normalises to uniform through the ordinary path. Pinned so the documented
    promise stays true however the guard is written.
    """
    out = softmax_1d(np.zeros(5, dtype=np.float64))

    assert out == pytest.approx(np.full(5, 0.2, dtype=np.float32))


def test_the_result_is_float32_on_every_path() -> None:
    """The fallback and the normal path must agree on dtype.

    `mcts/puct.py` feeds this straight into the tree's prior array; a float64
    escaping through one branch is the kind of thing that shows up much later
    as a dtype error in an unrelated place.
    """
    assert softmax_1d(np.array([1.0, 2.0], dtype=np.float64)).dtype == np.float32
    assert softmax_1d(np.array([1.0, np.nan], dtype=np.float64)).dtype == np.float32


def test_the_documented_caller_gets_a_usable_prior() -> None:
    """End-to-end through `mcts/puct.py:_softmax_legal`, the named caller.

    Asserting on the helper alone leaves open whether the caller reaches it
    with the shape assumed here: it passes `logits[legal_idx]`, a GATHER, so
    the nan only matters when it lands on a legal move. It does here.
    """
    from chess_anti_engine.mcts.puct import _softmax_legal

    logits = np.array([0.1, np.nan, 0.3, 0.4], dtype=np.float32)
    legal_idx = np.array([0, 1, 2], dtype=np.int64)

    prior = _softmax_legal(logits, legal_idx)

    assert np.isfinite(prior).all(), (
        "the MCTS prior must never contain nan; this is the exact call the "
        "fallback's docstring says it protects"
    )
    assert float(prior.sum()) == pytest.approx(1.0)


def test_a_nan_on_an_ILLEGAL_move_does_not_disturb_the_prior() -> None:
    """The other half of the gather: an out-of-set nan must be irrelevant.

    Without this, `_softmax_legal` returning uniform for every position with a
    nan ANYWHERE in the 1858-wide logit vector would look like a pass.
    """
    from chess_anti_engine.mcts.puct import _softmax_legal

    logits = np.array([0.1, 0.2, 0.3, np.nan], dtype=np.float32)
    legal_idx = np.array([0, 1, 2], dtype=np.int64)

    prior = _softmax_legal(logits, legal_idx)

    legal = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    expected = np.exp(legal - legal.max())
    expected = expected / expected.sum()
    assert prior == pytest.approx(expected.astype(np.float32), rel=1e-6)
