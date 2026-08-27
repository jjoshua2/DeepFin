"""Pentanomial GSPRT — a preregistered sequential stop rule for paired arenas.

WHY THIS EXISTS, AND WHY IT IS NOT "PEEKING". ``scripts/arena_standard.py`` is
fixed-N and final-only on purpose: reading a rolling arena and stopping when it
looked good manufactured **+112 Elo out of a true null** (see the ledger's
``rolling_arena_optional_stopping`` entry). That is the optional-stopping
fallacy, and the remedy the repo adopted was "never look early".

A GSPRT is the OTHER remedy: look as often as you like, but at a boundary
declared BEFORE the first game, whose crossing probabilities are what alpha and
beta name. The four numbers (``elo0``, ``elo1``, ``alpha``, ``beta``) are the
hypothesis, and this module deliberately supplies none of them by default — an
unstated hypothesis is not a hypothesis, it is a post-hoc reading with a test
statistic stapled to it.

⚑ The VERDICT is the deliverable. The Elo point estimate of a sequentially
stopped sample is BIASED AWAY FROM ZERO (a run stops early exactly when the
sample looks extreme), so it is descriptive colour, not the reading. See
``BIAS_CAVEAT``.

THE MATH — Michel Van den Bergh's generalized SPRT over the pentanomial
pair-outcome distribution, i.e. what fishtest computes.

  Source: fishtest ``server/fishtest/stats/LLRcalc.py`` (``regularize``,
  ``results_to_pdf``, ``MLE_expected``, ``LLR``, ``LLR_alt``, ``LLR_logistic``),
  which cites Van den Bergh's note
  "Maximum likelihood estimation of a multinomial distribution with a given
  expectation", http://hardy.uhasselt.be/Fishtest/support_MLE_multinomial.pdf
  (Proposition 1.1). Reimplemented here rather than vendored: the fishtest file
  pulls in ``scipy.optimize.brentq`` and a Monte-Carlo harness we do not want in
  an arena's import graph, and the part we need is one root-find.

  The unit is the PAIR, not the game — the arena plays every opening twice with
  colours swapped, so the two games of a pair are correlated and a trinomial
  (per-game W/D/L) SPRT would understate the variance and stop too early. The
  pair's normalized score is x in {0, 0.25, 0.5, 0.75, 1}.

  Given the observed pair counts n_i over those five outcomes, write
  ``phat_i = n_i / N`` for the empirical distribution. For a hypothesized mean
  score ``s``, the maximum-likelihood multinomial CONSTRAINED to expectation s is

      p_i(s) = phat_i / (1 + t (a_i - s))

  where ``a_i`` is the outcome value and ``t`` is the unique root of

      g(t) = sum_i phat_i (a_i - s) / (1 + t (a_i - s)) = 0

  on ``t in (-1/(w - s), 1/(s - v))`` with v/w the smallest/largest outcome
  value carrying mass. (The mean constraint implies normalization: expanding
  ``sum_i phat_i = 1`` gives ``sum_i p_i + t sum_i p_i (a_i - s) = 1``.) ``g`` is
  strictly decreasing on that interval and runs from ``+inf`` to ``-inf``, so a
  plain bisection is both sufficient and deterministic.

  The generalized log-likelihood ratio for ``s = s1`` against ``s = s0`` is then

      LLR = sum_i n_i log( p_i(s1) / p_i(s0) )

  with ``s0 = L(elo0)``, ``s1 = L(elo1)`` and ``L(e) = 1 / (1 + 10^(-e/400))``
  (logistic Elo, the same scale ``arena_standard._elo_from_score`` inverts).

  ``regularize`` mixes a 1e-3 prior into empty bins, exactly as fishtest does:
  without it an all-draws sample is a point mass, the constrained MLE for any
  other mean does not exist, and the LLR is undefined at precisely the sample
  a null test spends its life looking at.

Stop boundaries are Wald's:

    accept H1 (the candidate is at least elo1) when LLR >= log((1-beta)/alpha)
    accept H0 (the candidate is at most elo0)  when LLR <= log(beta/(1-alpha))

Neither crossing by the game cap is INCONCLUSIVE, and it is reported as such —
never silently rewritten into a fixed-N verdict, which would be the
optional-stopping fallacy wearing a lab coat.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# Pentanomial outcome values, ASCENDING (worst for the candidate first), which
# is fishtest's ``results_to_pdf`` convention: index i carries value i/(l-1).
#
# ⚑ ``scripts/arena_standard.PAIR_SCORES`` is DESCENDING and on the 0..2 point
# scale. The two orders are exact reverses of each other and confusing them
# silently INVERTS the test — a candidate that is winning would accept H0. Nothing
# in the arithmetic can catch that, so the conversion happens in exactly one
# place (``pentanomial_ascending``) and ``tests/test_arena_sprt.py`` pins the
# reversal against the arena's own binning.
PAIR_OUTCOME_VALUES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

# Bin labels in the same ASCENDING order, from the candidate's point of view.
PAIR_OUTCOME_LABELS: tuple[str, ...] = ("LL", "LD_DL", "DD_WL", "WD_DW", "WW")

# fishtest's LLRcalc.regularize: an empty bin becomes this, so the constrained
# MLE exists for every hypothesized mean. It is a prior, not an epsilon guard —
# it changes the answer slightly and is part of the definition.
REGULARIZATION_MASS = 1e-3

BIAS_CAVEAT = (
    "a sequentially stopped Elo point estimate is BIASED AWAY FROM ZERO: the "
    "run stopped at the moment the sample looked extreme enough to cross, so "
    "the magnitude is inflated (by more the earlier it stopped). The SPRT "
    "VERDICT is the deliverable; the Elo and its CI are descriptive only, and "
    "the CI in particular has no nominal coverage after a sequential stop."
)

VERDICT_H1 = "H1"
VERDICT_H0 = "H0"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


def logistic_score(elo: float) -> float:
    """Expected score for a logistic-Elo advantage — fishtest's ``L_``."""
    return 1.0 / (1.0 + 10.0 ** (-float(elo) / 400.0))


def pentanomial_ascending(pair_scores: Sequence[float]) -> tuple[int, int, int, int, int]:
    """Bin raw pair scores (0/0.5/1/1.5/2, candidate POV) into ASCENDING counts.

    Raw pair scores are the candidate's points over the two games of one
    opening, which is what every play loop in ``arena_standard`` returns. The
    normalized outcome is ``score / 2``.
    """
    counts = [0, 0, 0, 0, 0]
    for raw in pair_scores:
        x = float(raw) / 2.0
        try:
            counts[PAIR_OUTCOME_VALUES.index(x)] += 1
        except ValueError:
            raise ValueError(
                f"pair score must be one of (0.0, 0.5, 1.0, 1.5, 2.0), got {raw!r}"
            ) from None
    return (counts[0], counts[1], counts[2], counts[3], counts[4])


def regularize(counts: Sequence[float]) -> tuple[float, ...]:
    """fishtest ``LLRcalc.regularize``: an empty bin gets a small prior mass."""
    return tuple(REGULARIZATION_MASS if c == 0 else float(c) for c in counts)


def constrained_mle(phat: Sequence[float], s: float) -> tuple[float, ...]:
    """MLE of the pentanomial with expectation ``s`` given empirical ``phat``.

    Van den Bergh Proposition 1.1 (see the module docstring): the solution is
    ``p_i = phat_i / (1 + t (a_i - s))`` for the ``t`` that zeroes the mean
    residual. Solved by bisection because the residual is strictly decreasing on
    the feasible interval and runs from +inf to -inf across it, so bisection
    cannot fail and — unlike a Brent/Newton hybrid — needs no derivative and has
    no iteration-order sensitivity to reproduce.
    """
    a = PAIR_OUTCOME_VALUES
    if len(phat) != len(a):
        raise ValueError(f"need {len(a)} bins, got {len(phat)}")
    support = [i for i, p in enumerate(phat) if p > 0.0]
    if not support:
        raise ValueError("empirical distribution has no mass")
    v, w = a[support[0]], a[support[-1]]
    if not v < s < w:
        raise ValueError(
            f"hypothesized mean score {s!r} is outside the observed support "
            f"({v}, {w}); the constrained MLE does not exist there"
        )

    def residual(t: float) -> float:
        return sum(phat[i] * (a[i] - s) / (1.0 + t * (a[i] - s)) for i in support)

    # Feasible open interval: every 1 + t (a_i - s) must stay positive.
    lo, hi = -1.0 / (w - s), 1.0 / (s - v)
    # Step inside the open ends by a relative amount, so the bracket is valid
    # for any s and the endpoint residuals keep their +/- signs.
    pad = 1e-12 * max(1.0, hi - lo)
    lo, hi = lo + pad, hi - pad
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break  # bracket has collapsed to adjacent floats
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return tuple(p / (1.0 + t * (a[i] - s)) for i, p in enumerate(phat))


def gsprt_llr(counts: Sequence[int], *, s0: float, s1: float) -> float:
    """Pentanomial GSPRT log-likelihood ratio for ``s1`` against ``s0``.

    ``counts`` is ASCENDING (``PAIR_OUTCOME_LABELS``). Positive favours ``s1``.
    """
    if not 0.0 < s0 < 1.0 or not 0.0 < s1 < 1.0:
        raise ValueError(f"hypothesis scores must lie in (0, 1); got {s0}, {s1}")
    reg = regularize(counts)
    n = sum(reg)
    phat = tuple(c / n for c in reg)
    p0 = constrained_mle(phat, s0)
    p1 = constrained_mle(phat, s1)
    return n * sum(
        ph * math.log(q1 / q0) for ph, q0, q1 in zip(phat, p0, p1) if ph > 0.0
    )


def gsprt_llr_elo(counts: Sequence[int], *, elo0: float, elo1: float) -> float:
    """``gsprt_llr`` in logistic Elo — fishtest's ``LLR_logistic``."""
    return gsprt_llr(counts, s0=logistic_score(elo0), s1=logistic_score(elo1))


# Largest |Elo| whose logistic score stays strictly inside (0, 1) in float64
# with room for the MLE bracket. L(2800) = 1 - 6e-8; beyond ~5000 it rounds to
# exactly 1.0 and the constrained MLE stops existing. Refused rather than
# clamped: a clamp would accept the number and test a different hypothesis.
MAX_ABS_ELO = 2000.0


@dataclass(frozen=True)
class SprtSpec:
    """The preregistered hypothesis. All four fields are required, no defaults."""

    elo0: float
    elo1: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        for name in ("elo0", "elo1", "alpha", "beta"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"--sprt {name} must be finite, got {value!r}")
        if not self.elo0 < self.elo1:
            raise ValueError(
                f"--sprt needs elo0 < elo1 (H0 is 'at most elo0', H1 is 'at "
                f"least elo1'); got elo0={self.elo0}, elo1={self.elo1}"
            )
        for name in ("alpha", "beta"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"--sprt {name} must be in (0, 1), got {value!r}")
        if self.alpha + self.beta >= 1.0:
            # Otherwise log((1-beta)/alpha) <= log(beta/(1-alpha)): the accept-H1
            # boundary sits at or below the accept-H0 one and the FIRST check
            # crosses both. A test that cannot fail is this repo's signature defect.
            raise ValueError(
                f"--sprt needs alpha + beta < 1 (got {self.alpha} + {self.beta} = "
                f"{self.alpha + self.beta}); otherwise the H1 boundary is not "
                "above the H0 boundary and the test decides on the first look"
            )
        for name in ("elo0", "elo1"):
            value = float(getattr(self, name))
            if abs(value) > MAX_ABS_ELO:
                raise ValueError(
                    f"--sprt {name}={value} is beyond +/-{MAX_ABS_ELO:.0f} Elo, "
                    "where the logistic score rounds to 0 or 1 and the "
                    "constrained MLE does not exist"
                )

    @property
    def s0(self) -> float:
        return logistic_score(self.elo0)

    @property
    def s1(self) -> float:
        return logistic_score(self.elo1)

    @property
    def bound_h1(self) -> float:
        """Wald upper boundary: LLR >= this accepts H1."""
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def bound_h0(self) -> float:
        """Wald lower boundary: LLR <= this accepts H0."""
        return math.log(self.beta / (1.0 - self.alpha))

    @classmethod
    def from_cli(cls, spec: str) -> SprtSpec:
        """Parse ``elo0=E0,elo1=E1,alpha=A,beta=B``.

        Every key is required and there is no default anywhere in this path: a
        hidden default would make the boundary something the operator did not
        state, and the whole point of a sequential test is that the boundary was
        declared in advance.
        """
        required = ("elo0", "elo1", "alpha", "beta")
        seen: dict[str, float] = {}
        for item in str(spec).split(","):
            field = item.strip()
            if not field:
                continue
            if "=" not in field:
                raise ValueError(
                    f"--sprt: {field!r} is not k=v; expected "
                    f"'{','.join(f'{k}=<float>' for k in required)}'"
                )
            key, _, raw = field.partition("=")
            key, raw = key.strip(), raw.strip()
            if key not in required:
                raise ValueError(
                    f"--sprt: unknown key {key!r}; expected exactly "
                    f"{', '.join(required)}"
                )
            if key in seen:
                raise ValueError(f"--sprt: {key!r} given more than once")
            try:
                seen[key] = float(raw)
            except ValueError:
                raise ValueError(
                    f"--sprt: {key}={raw!r} is not a number"
                ) from None
        missing = [k for k in required if k not in seen]
        if missing:
            raise ValueError(
                f"--sprt: missing {', '.join(missing)}. All four of "
                f"{', '.join(required)} are REQUIRED — there is no default, "
                "because an unstated hypothesis is not a hypothesis."
            )
        return cls(**seen)

    def describe(self) -> str:
        return (
            f"H0: elo <= {self.elo0:+.2f} (score {self.s0:.5f})  "
            f"H1: elo >= {self.elo1:+.2f} (score {self.s1:.5f})  "
            f"alpha={self.alpha:.4g} beta={self.beta:.4g}  "
            f"boundaries: H0 <= {self.bound_h0:+.4f}, H1 >= {self.bound_h1:+.4f}"
        )

    def as_record(self) -> dict[str, float]:
        return {
            "elo0": float(self.elo0),
            "elo1": float(self.elo1),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "s0": float(self.s0),
            "s1": float(self.s1),
            "bound_h0": float(self.bound_h0),
            "bound_h1": float(self.bound_h1),
        }


class SprtMonitor:
    """Running GSPRT over the pairs an arena has COMPLETED.

    The LLR is recomputed from every banked pair score on every look — resumed
    pairs included — rather than being accumulated incrementally. That is not an
    efficiency oversight: the GSPRT statistic is a function of the whole
    empirical distribution (the constrained MLE re-fits on all of it), so an
    incremental form would be a different number, and a resumed run would carry
    a stale one.

    ``granularity`` is recorded and printed because it is a real property of the
    reading: the rolling loop can look after every completed pair, the chunked
    loop only between chunks. Both are valid stopping times; the coarser one
    just looks less often.
    """

    def __init__(
        self,
        spec: SprtSpec,
        *,
        prior_pair_scores: Sequence[float] = (),
        pairs_cap: int,
        granularity: str,
    ) -> None:
        self.spec = spec
        self.pairs_cap = int(pairs_cap)
        self.granularity = str(granularity)
        self._prior: list[float] = [float(s) for s in prior_pair_scores]
        self.pairs: int = 0
        self.counts: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
        self.llr: float = 0.0
        self.llr_first: float = 0.0
        self.looks: int = 0
        # (pairs, llr) sampled once per DISTINCT pair count, so a rolling loop
        # that looks every ply does not bank thousands of identical rows.
        self.trajectory: list[tuple[int, float]] = []
        self.verdict: str | None = None
        self.stop_reason: str | None = None
        # The resumed pairs are evidence like any other: a run that already
        # crossed must not play more games just because the process restarted.
        self.update(())
        self.llr_first = self.llr

    def update(self, new_pair_scores: Sequence[float]) -> str | None:
        """Fold in this loop's own completed pairs and re-decide. Returns the verdict."""
        scores = self._prior + [float(s) for s in new_pair_scores]
        pairs = len(scores)
        self.looks += 1
        if pairs == self.pairs and self.trajectory:
            return self.verdict  # nothing new completed; the statistic cannot move
        self.pairs = pairs
        self.counts = pentanomial_ascending(scores)
        self.llr = gsprt_llr(self.counts, s0=self.spec.s0, s1=self.spec.s1)
        self.trajectory.append((pairs, self.llr))
        if self.verdict is None:
            if self.llr >= self.spec.bound_h1:
                self.verdict, self.stop_reason = VERDICT_H1, "boundary"
            elif self.llr <= self.spec.bound_h0:
                self.verdict, self.stop_reason = VERDICT_H0, "boundary"
        return self.verdict

    def crossed(self) -> bool:
        return self.verdict is not None

    def finalize(self, *, stop_reason: str) -> str:
        """Settle the verdict once play has ended. ``INCONCLUSIVE`` if uncrossed."""
        if self.verdict is None:
            self.verdict = VERDICT_INCONCLUSIVE
            self.stop_reason = stop_reason
        return self.verdict

    def verdict_line(self) -> str:
        if self.verdict == VERDICT_H1:
            claim = f"ACCEPT H1 — the candidate is at least {self.spec.elo1:+.2f} Elo"
        elif self.verdict == VERDICT_H0:
            claim = f"ACCEPT H0 — the candidate is at most {self.spec.elo0:+.2f} Elo"
        else:
            claim = (
                "INCONCLUSIVE — neither boundary was crossed; this is NOT a "
                "fixed-N verdict and must not be reported as one"
            )
        return (
            f"SPRT VERDICT: {self.verdict or VERDICT_INCONCLUSIVE}  {claim}\n"
            f"[arena] SPRT: LLR {self.llr_first:+.4f} -> {self.llr:+.4f} over "
            f"{len(self.trajectory)} distinct sample(s) at {self.granularity} "
            f"granularity, {self.looks} consultation(s) "
            f"(H0 <= {self.spec.bound_h0:+.4f}, H1 >= {self.spec.bound_h1:+.4f})\n"
            f"[arena] SPRT: {self.pairs} pair(s) of a {self.pairs_cap} pair cap "
            f"({2 * self.pairs} of {2 * self.pairs_cap} games); "
            f"stop_reason={self.stop_reason}\n"
            f"[arena] SPRT ⚑ {BIAS_CAVEAT}"
        )

    def as_record(self) -> dict[str, Any]:
        """The banked reading — the hypothesis, the trajectory, and the caveat.

        Typed ``dict[str, Any]`` to match ``build_result_record``'s bare ``dict``:
        this block is a JSON payload with heterogeneous values, and a reader
        that has to cast every field before comparing it is a reader that stops
        checking.
        """
        return {
            **self.spec.as_record(),
            "verdict": self.verdict or VERDICT_INCONCLUSIVE,
            "stop_reason": self.stop_reason,
            "stopped_early": bool(
                self.stop_reason == "boundary" and self.pairs < self.pairs_cap
            ),
            "llr": self.llr,
            "llr_first": self.llr_first,
            "llr_trajectory": [[p, llr] for p, llr in self.trajectory],
            "pairs": self.pairs,
            "pairs_cap": self.pairs_cap,
            "games": 2 * self.pairs,
            "games_cap": 2 * self.pairs_cap,
            # Two different counts, and the second is the meaningful one. A
            # rolling loop consults the boundary every ply, but most plies
            # complete no pair, so the statistic is unchanged and the repeat
            # carries no extra multiplicity. `distinct_samples` is the number of
            # DIFFERENT samples the boundary was actually tested against.
            "looks": self.looks,
            "distinct_samples": len(self.trajectory),
            "check_granularity": self.granularity,
            "pentanomial_ascending": dict(zip(PAIR_OUTCOME_LABELS, self.counts)),
            "resumed_pairs": len(self._prior),
            "elo_estimate_biased_away_from_zero": True,
            "caveat": BIAS_CAVEAT,
        }
