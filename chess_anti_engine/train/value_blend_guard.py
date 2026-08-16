"""Fail loudly when the SF share of the value blend silently becomes the outcome.

⚑ THE DEFECT THIS CLOSES IS A SILENT ONE, WHICH IS WHY A PROSE REQUIREMENT
DOES NOT CLOSE IT.

``compute_loss`` builds the WDL value target as

    target = game_frac * game_outcome
           + sf_wdl_frac * sf_component
           + search_wdl_frac * search_component

and ``sf_component`` is

    sf_effective * sf_wdl_probs + (1 - sf_effective) * game_outcome

with ``sf_effective = has_sf_wdl * keep``. So on a row that carries no
Stockfish label the SF component **is the raw one-hot game outcome**. There is
no error, no warning, and no metric whose name says so: the run trains, the
losses look ordinary, and ``sf_wdl_frac`` of the value weight has quietly moved
onto a target nobody chose.

lc0-derived shards (``scripts/lc0_data_to_rows.py``) carry no ``sf_wdl`` at
all, so EVERY row takes that branch. At the production
``sf_wdl_frac: 0.50`` / ``search_wdl_frac: 0.20`` the intended 0.30 outcome
share becomes 0.80 — a different experiment from the one on paper, reached
without a single line of output.

The mitigation ``sf_wdl_frac: 0.0`` already lives in the converter's manifest.
A manifest is documentation. This module is the check, and it is designed
around two rules this codebase keeps re-learning:

1. **A configured value is not an applied value.** ``value_blend_readout``
   takes the fracs that were actually handed to ``compute_loss`` on a real
   step and the batch's own ``sf_wdl_rows`` count, never a yaml field.
2. **A guard must share the criterion's instrument.** The frac normalisation
   comes from ``losses.normalize_value_blend_fracs`` — the same function
   ``compute_loss`` calls — rather than a second copy that could drift.

The check can FAIL: point it at lc0 shards with a non-zero ``sf_wdl_frac`` and
it raises. ``tests/test_value_blend_guard.py`` mutates exactly that way.
"""
from __future__ import annotations

from dataclasses import dataclass

from chess_anti_engine.train.losses import normalize_value_blend_fracs


class ValueBlendMisconfigured(RuntimeError):
    """The SF share of the value blend is landing on the game outcome."""


@dataclass(frozen=True)
class ValueBlendReadout:
    """What the value blend REALIZED on one batch.

    ``sf_wdl_frac`` / ``search_wdl_frac`` / ``game_frac`` are post-clamp,
    post-renormalisation: the numbers ``compute_loss`` multiplied its
    components by, not the numbers in the yaml.
    """

    sf_wdl_frac: float
    search_wdl_frac: float
    game_frac: float
    sf_wdl_rows: float
    batch_rows: float

    @property
    def sf_labelled_frac(self) -> float:
        """Share of batch rows that actually carry a usable ``sf_wdl``."""
        if self.batch_rows <= 0.0:
            return 0.0
        return self.sf_wdl_rows / self.batch_rows

    @property
    def leaked_to_outcome(self) -> float:
        """Value weight that fell from the SF component onto the raw outcome.

        Zero when every row is SF-labelled (production) OR when
        ``sf_wdl_frac`` is 0 (the lc0 override). It is the product, so either
        one alone closes the hole — which is why the guard reports both.
        """
        return self.sf_wdl_frac * (1.0 - self.sf_labelled_frac)

    @property
    def outcome_borne_frac(self) -> float:
        """TOTAL value weight on the raw one-hot outcome, intended or not."""
        return self.game_frac + self.leaked_to_outcome

    def as_table(self) -> list[tuple[str, float]]:
        """Ordered rows for a human-readable realized-weight table."""
        return [
            ("sf_wdl_frac (realized)", self.sf_wdl_frac),
            ("search_wdl_frac (realized)", self.search_wdl_frac),
            ("game_frac (intended outcome share)", self.game_frac),
            ("sf_labelled_frac (rows with sf_wdl)", self.sf_labelled_frac),
            ("leaked_to_outcome", self.leaked_to_outcome),
            ("outcome_borne_frac (game_frac + leak)", self.outcome_borne_frac),
        ]


def value_blend_readout(
    *,
    sf_wdl_frac: float,
    search_wdl_frac: float,
    sf_wdl_rows: float,
    batch_rows: float,
) -> ValueBlendReadout:
    """Build the readout from ONE STEP's realized fracs and row counts.

    ``sf_wdl_rows`` and ``batch_rows`` are ``compute_loss``'s own returned
    scalars of those names — the batch's count of rows carrying ``has_sf_wdl``
    and the batch size. Passing anything else (a shard-level flag, a yaml key,
    a replay-window average) defeats the point: the question is what THIS step
    trained on.
    """
    sf, search, game = normalize_value_blend_fracs(sf_wdl_frac, search_wdl_frac)
    return ValueBlendReadout(
        sf_wdl_frac=sf,
        search_wdl_frac=search,
        game_frac=game,
        sf_wdl_rows=float(sf_wdl_rows),
        batch_rows=float(batch_rows),
    )


# Production's `game_frac` = 1 - sf_wdl_frac(0.50) - search_wdl_frac(0.20).
# The control holds it at exactly this by construction, so it is the bar the
# TOTAL outcome share is judged against rather than a number picked here.
PRODUCTION_GAME_FRAC = 0.30


def assert_outcome_is_not_the_whole_target(
    readout: ValueBlendReadout,
    *,
    max_outcome_borne: float = PRODUCTION_GAME_FRAC,
    context: str = "",
) -> None:
    """Raise when too much of the value target is the RAW ONE-HOT OUTCOME.

    ⚑ THIS IS A SECOND DOOR TO THE SAME ROOM, AND ``leaked_to_outcome`` CANNOT
    SEE IT. The leak is ``sf_wdl_frac x (1 - sf_labelled_frac)``, so it is 0
    whenever ``sf_wdl_frac`` is 0 — including the configuration
    ``sf_wdl_frac: 0.0`` / ``search_wdl_frac: 0.0``, which trains **100% of the
    value target on the raw game outcome** with a leak of exactly 0.00. PR
    #438's review reached that state through all three of the old guards.

    ``outcome_borne_frac`` is the number that answers "how much of the value
    target is the raw outcome", intended or leaked, and it is the one to gate
    on. The default bar is production's own ``game_frac``: the control's whole
    justification for moving the SF share to search is that this number does
    not move.
    """
    if readout.outcome_borne_frac <= max_outcome_borne + 1e-9:
        return
    where = f" [{context}]" if context else ""
    raise ValueBlendMisconfigured(
        f"the value target puts {readout.outcome_borne_frac:.4f} of its mass on "
        f"the RAW GAME OUTCOME{where}, above the bar of {max_outcome_borne:.4f} "
        f"(production's game_frac). Realized fracs: sf={readout.sf_wdl_frac:.4f} "
        f"search={readout.search_wdl_frac:.4f} game={readout.game_frac:.4f}, "
        f"leaked={readout.leaked_to_outcome:.4f}, with "
        f"{readout.sf_wdl_rows:.0f}/{readout.batch_rows:.0f} rows SF-labelled. "
        "⚑ A leak of 0.00 does not clear this: sf_wdl_frac 0 with "
        "search_wdl_frac 0 leaks nothing and trains everything on the outcome."
    )


def assert_no_silent_outcome_fallback(
    readout: ValueBlendReadout,
    *,
    max_leak: float = 0.0,
    max_outcome_borne: float = PRODUCTION_GAME_FRAC,
    context: str = "",
) -> None:
    """Raise on either an SF-to-outcome leak or an all-outcome value target.

    ``max_leak`` defaults to 0.0 — an exact bar, because on lc0 shards the
    quantity is exactly ``sf_wdl_frac`` and on production shards it is exactly
    0.0 apart from the handful of rows a partial shard leaves unlabelled. A
    caller running against a corpus with known partial labelling should raise
    the bar deliberately and say why, rather than have the guard pick a
    tolerance nobody chose.

    Both bars are checked, in that order, because they fail on disjoint
    configurations — see ``assert_outcome_is_not_the_whole_target``.
    """
    if readout.leaked_to_outcome <= max_leak:
        assert_outcome_is_not_the_whole_target(
            readout, max_outcome_borne=max_outcome_borne, context=context,
        )
        return
    where = f" [{context}]" if context else ""
    raise ValueBlendMisconfigured(
        f"value blend is training {readout.leaked_to_outcome:.4f} of the WDL "
        f"target on the RAW GAME OUTCOME through the SF fallback{where}: "
        f"realized sf_wdl_frac={readout.sf_wdl_frac:.4f} while only "
        f"{readout.sf_wdl_rows:.0f}/{readout.batch_rows:.0f} rows carry an "
        f"sf_wdl label. Intended outcome share was game_frac="
        f"{readout.game_frac:.4f}; the target actually being trained puts "
        f"{readout.outcome_borne_frac:.4f} there. losses.py does this without "
        "an error or a warning. On lc0-derived shards set sf_wdl_frac: 0.0 "
        "and sf_wdl_frac_floor: 0.0, and give the freed share to "
        "search_wdl_frac (lc0's own best_q/best_d)."
    )


def assert_pid_cannot_reassert_sf_wdl(
    *, sf_wdl_frac: float, sf_wdl_frac_floor: float, context: str = "",
) -> None:
    """Raise if the difficulty controller could raise ``sf_wdl_frac`` mid-run.

    ⚑ A launch-time override that the controller can undo is not an override,
    it is a head start. ``tune/trainable_config_ops.py`` recomputes the weight
    EVERY iteration via ``trainable_metrics._dynamic_sf_wdl_weight`` and
    assigns it to the live trainer whenever the result is not ``None``. That
    function returns ``None`` only when ``sf_wdl_start <= 0``, and
    ``sf_wdl_start`` is ``TrialConfig.sf_wdl_frac``. So ``sf_wdl_frac: 0.0``
    disables the ramp at its source and ``sf_wdl_frac_floor`` is never read —
    but a config that set the floor without zeroing the start would ramp back
    up as regret fell, silently expiring the override partway through a run.

    The floor is checked too even though it is inert while the start is zero:
    a non-zero floor next to a zero start is a config that means to reassert
    the weight and is one edit away from doing so.
    """
    problems = [
        f"{name}={value!r} must be 0.0"
        for name, value in (
            ("sf_wdl_frac", float(sf_wdl_frac)),
            ("sf_wdl_frac_floor", float(sf_wdl_frac_floor)),
        )
        if value > 0.0
    ]
    if not problems:
        return
    where = f" [{context}]" if context else ""
    raise ValueBlendMisconfigured(
        f"the PID sf_wdl ramp is not disabled{where}: {'; '.join(problems)}. "
        "_dynamic_sf_wdl_weight returns None (leaving the trainer's weight "
        "alone) only while sf_wdl_frac <= 0, so a non-zero start lets the "
        "controller re-raise the SF share every iteration and the lc0 "
        "override expires mid-run without a log line."
    )
