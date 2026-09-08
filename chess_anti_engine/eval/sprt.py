"""Pentanomial GSPRT on canonical, complete opening-pair prefixes.

The constrained multinomial likelihood follows Fishtest's LLR_logistic:
https://github.com/official-stockfish/fishtest/blob/master/server/fishtest/stats/LLRcalc.py
Pair outcomes are normalized scores in {0, .25, .5, .75, 1}. For empirical
probabilities phat_i and hypothesized mean s, the constrained MLE is
p_i = phat_i / (1 + t * (a_i - s)), with t solving the mean constraint.

This is a generalized likelihood ratio with estimated nuisance parameters.
Its error/efficiency guarantees are asymptotic, not exact finite-sample Wald
bounds. Alpha/beta set nominal boundaries. Fishtest now ordinarily uses
normalized Elo; this module explicitly uses logistic Elo.

A boundary favors one separated hypothesis over the other. Accepting H1 does
not establish a lower confidence bound at elo1; the indifference region can
resolve either way. Stopped point estimates are selection-biased and ordinary
fixed-N confidence intervals have no nominal sequential coverage.

Every declared look is evaluated in canonical pair order, independently of
which games finish first. A cap/deadline without crossing is inconclusive.
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
    "a sequentially stopped Elo estimate is subject to selection bias; "
    "the ordinary fixed-N CI has no nominal coverage after stopping. "
    "Report the declared hypotheses, stopping prefix and SPRT verdict. "
    "H1 is not a lower confidence bound at elo1; H0 is not equivalence."
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
    first_pairs: int = 1
    step_pairs: int = 1

    def __post_init__(self) -> None:
        for name in ("first_pairs", "step_pairs"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"--sprt {name} must be a positive integer")
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
            if key not in (*required, "first_pairs", "step_pairs"):
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
        look_fields = {}
        for key in ("first_pairs", "step_pairs"):
            if key in seen:
                value = seen.pop(key)
                if not math.isfinite(value) or not value.is_integer():
                    raise ValueError(f"--sprt {key} must be a positive integer")
                look_fields[key] = int(value)
        return cls(**seen, **look_fields)

    def describe(self) -> str:
        return (
            f"H0: elo <= {self.elo0:+.2f} (score {self.s0:.5f})  "
            f"H1: elo >= {self.elo1:+.2f} (score {self.s1:.5f})  "
            f"alpha={self.alpha:.4g} beta={self.beta:.4g}  "
            f"first_pairs={self.first_pairs} step_pairs={self.step_pairs}  "
            f"boundaries: H0 <= {self.bound_h0:+.4f}, H1 >= {self.bound_h1:+.4f}"
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "sampling": "canonical_pair_prefix_v1",
            "first_pairs": self.first_pairs,
            "step_pairs": self.step_pairs,
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
    """GSPRT on a declared prefix of canonical opening-pair IDs.

    Completion can arrive out of order. Only IDs 0..N-1 may contribute, and
    every declared look newly released by a late pair is checked in order.
    Completed suffix pairs remain banked but cannot change a crossed verdict.
    Storage is bounded by the registered pair cap.
    """

    def __init__(
        self, spec: SprtSpec, *, prior_pair_scores: Sequence[float] = (),
        prior_pair_ids: Sequence[int] | None = None, pairs_cap: int,
        granularity: str,
    ) -> None:
        if type(pairs_cap) is not int or pairs_cap < spec.first_pairs:
            raise ValueError("SPRT pair cap must be at least first_pairs")
        self.spec = spec
        self.pairs_cap = pairs_cap
        self.granularity = granularity
        self._prior = list(prior_pair_scores)
        self._complete: dict[int, float] = {}
        self._sample: list[float] = []
        self.pairs = 0
        self.counts: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
        self.llr = 0.0
        self.llr_first = 0.0
        self.looks = 0
        self.trajectory: list[tuple[int, float]] = []
        self.verdict: str | None = None
        self.stop_reason: str | None = None
        self._next_look = spec.first_pairs
        self.inflight_games: list[tuple[int, int]] = []
        self.not_started_games = 0
        ids = list(range(len(self._prior))) if prior_pair_ids is None else prior_pair_ids
        self.update(self._prior, pair_ids=ids)
        self.llr_first = self.llr

    @property
    def next_look_pairs(self) -> int:
        """Next unconsumed declared sample size, bounded by the pair cap."""
        return min(self._next_look, self.pairs_cap)

    @property
    def pair_scores(self) -> list[float]:
        return list(self._sample)

    @property
    def complete_pairs(self) -> dict[int, float]:
        return dict(self._complete)

    def _set_sample(self, scores: Sequence[float]) -> None:
        self._sample = list(scores)
        self.pairs = len(scores)
        self.counts = pentanomial_ascending(scores)
        self.llr = gsprt_llr(self.counts, s0=self.spec.s0, s1=self.spec.s1)

    def update(
        self, new_pair_scores: Sequence[float], *, pair_ids: Sequence[int] | None = None,
    ) -> str | None:
        """Merge cumulative completed observations; repeated IDs must agree."""
        ids = (list(range(len(self._prior), len(self._prior) + len(new_pair_scores)))
               if pair_ids is None else list(pair_ids))
        if len(ids) != len(new_pair_scores) or len(set(ids)) != len(ids):
            raise ValueError("SPRT pair IDs must be unique and score-aligned")
        for pair_id, score in zip(ids, new_pair_scores):
            if type(pair_id) is not int or not 0 <= pair_id < self.pairs_cap:
                raise ValueError("SPRT pair ID is outside the registered cap")
            value = float(score)
            if value not in (0.0, 0.5, 1.0, 1.5, 2.0):
                raise ValueError("invalid SPRT pair score")
            if pair_id in self._complete and self._complete[pair_id] != value:
                raise ValueError("SPRT completed pair score changed")
            self._complete[pair_id] = value
        self.looks += 1
        if self.verdict is not None:
            return self.verdict
        prefix = []
        while len(prefix) in self._complete:
            prefix.append(self._complete[len(prefix)])
        while self._next_look <= len(prefix):
            self._set_sample(prefix[:self._next_look])
            self.trajectory.append((self.pairs, self.llr))
            if self.llr >= self.spec.bound_h1:
                self.verdict, self.stop_reason = VERDICT_H1, "boundary"
            elif self.llr <= self.spec.bound_h0:
                self.verdict, self.stop_reason = VERDICT_H0, "boundary"
            if self.verdict is not None:
                return self.verdict
            if self._next_look == self.pairs_cap:
                self._next_look += 1
            else:
                self._next_look = min(self.pairs_cap, self._next_look + self.spec.step_pairs)
        # A deadline may land between looks. Bank the available prefix and its
        # descriptive LLR, without turning an undeclared look into a decision.
        self._set_sample(prefix)
        return None

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
            claim = f"FAVORS H1 ({self.spec.elo1:+.2f}) over H0 ({self.spec.elo0:+.2f}); not an effect lower bound"
        elif self.verdict == VERDICT_H0:
            claim = f"FAVORS H0 ({self.spec.elo0:+.2f}) over H1 ({self.spec.elo1:+.2f}); not equivalence"
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
            "scored_pair_ids": list(range(self.pairs)),
            "completed_pair_ids": sorted(self._complete),
            "speculative_completed_pair_ids": sorted(i for i in self._complete if i >= self.pairs),
            "inflight_games": [list(item) for item in self.inflight_games],
            "not_started_games": self.not_started_games,
            "last_decision_look_pairs": self.trajectory[-1][0] if self.trajectory else 0,
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
            "elo_estimate_selection_biased": True,
            "caveat": BIAS_CAVEAT,
        }
