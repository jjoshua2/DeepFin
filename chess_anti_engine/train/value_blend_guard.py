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

⚑⚑ AND THE SEARCH COMPONENT DOES EXACTLY THE SAME THING. ``search_component``
is built by the identical expression against ``has_search_wdl``, which
``_get_mask`` also defaults to 0.0 when the column is absent. Until 2026-08-16
this module measured only the SF half, so a batch could train **100% of its
value target on the raw outcome** through the search half with every guard here
reporting ``outcome_borne_frac 0.3000`` (PR #438 review F1, reproduced through
the real ``compute_loss``). That is this repo's signature defect — a metric
that does not mean what its name says — inside the module written to close an
instance of it. Both halves are measured now, and
``tests/test_value_blend_guard.py`` drives the search-side state.

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

    ``sf_effective_rows`` / ``search_effective_rows`` are ``compute_loss``'s
    own ``sf_wdl_effective_rows`` / ``search_wdl_effective_rows`` — the mass of
    each component that actually came from its LABEL rather than from the
    one-hot fallback. They are not label counts: they carry the
    ``sf_search_dampen_*`` shortfall and read 0 when the label COLUMN is absent.
    """

    sf_wdl_frac: float
    search_wdl_frac: float
    game_frac: float
    sf_effective_rows: float
    search_effective_rows: float
    batch_rows: float

    def _frac(self, rows: float) -> float:
        if self.batch_rows <= 0.0:
            return 0.0
        return rows / self.batch_rows

    @property
    def sf_effective_frac(self) -> float:
        """Share of the SF component that came from a real ``sf_wdl``."""
        return self._frac(self.sf_effective_rows)

    @property
    def search_effective_frac(self) -> float:
        """Share of the SEARCH component that came from a real ``search_wdl``."""
        return self._frac(self.search_effective_rows)

    @property
    def leaked_from_sf(self) -> float:
        """Value weight that fell from the SF component onto the raw outcome."""
        return self.sf_wdl_frac * (1.0 - self.sf_effective_frac)

    @property
    def leaked_from_search(self) -> float:
        """Value weight that fell from the SEARCH component onto the outcome.

        ⚑ THE HALF THAT HAD NO INSTRUMENT UNTIL 2026-08-16, and the LARGER one
        for this arm: the lc0 control puts 0.70 of its value target here. PR
        #438's review reproduced, through the real ``compute_loss``, a target
        that was 100% raw game outcome while every guard read clean, because
        the readout only ever looked at the SF side.
        """
        return self.search_wdl_frac * (1.0 - self.search_effective_frac)

    @property
    def leaked_to_outcome(self) -> float:
        """TOTAL weight that fell onto the raw outcome through EITHER fallback.

        Zero when every row carries both labels (production) or when the
        corresponding frac is 0. Each term is a product, so either factor alone
        closes its own hole — which is why the table reports every factor.
        """
        return self.leaked_from_sf + self.leaked_from_search

    @property
    def outcome_borne_frac(self) -> float:
        """TOTAL value weight on the raw one-hot outcome, intended or not.

        ⚑ Excludes ``wdl_terminal_outcome_*``, which moves part of the search
        share onto the outcome DELIBERATELY, near-terminally, and publishes its
        own ``wdl_terminal_outcome_frac`` column. That transfer is a chosen
        target; this number is about the unchosen one. Both are 0.0 in the
        control (``wdl_terminal_outcome_plies: 0``).
        """
        return self.game_frac + self.leaked_to_outcome

    def as_table(self) -> list[tuple[str, float]]:
        """Ordered rows for a human-readable realized-weight table."""
        return [
            ("sf_wdl_frac (realized)", self.sf_wdl_frac),
            ("search_wdl_frac (realized)", self.search_wdl_frac),
            ("game_frac (intended outcome share)", self.game_frac),
            ("sf_effective_frac (labelled sf mass)", self.sf_effective_frac),
            ("search_effective_frac (labelled search mass)", self.search_effective_frac),
            ("leaked_from_sf", self.leaked_from_sf),
            ("leaked_from_search", self.leaked_from_search),
            ("leaked_to_outcome", self.leaked_to_outcome),
            ("outcome_borne_frac (game_frac + leak)", self.outcome_borne_frac),
        ]


def value_blend_readout(
    *,
    sf_wdl_frac: float,
    search_wdl_frac: float,
    sf_effective_rows: float,
    search_effective_rows: float,
    batch_rows: float,
) -> ValueBlendReadout:
    """Build the readout from ONE STEP's realized fracs and row counts.

    The three row counts are ``compute_loss``'s own returned scalars
    ``sf_wdl_effective_rows`` / ``search_wdl_effective_rows`` / ``batch_rows``.
    Passing anything else (a shard-level flag, a yaml key, a replay-window
    average) defeats the point: the question is what THIS step trained on.

    ⚑ Both effective counts are REQUIRED keyword arguments with no default. A
    default of 0.0 would read as "nothing was labelled" and a default of
    ``batch_rows`` as "everything was" — and the second is the one a caller
    updating this signature would reach for, which is how the search half went
    unmeasured in the first place.
    """
    sf, search, game = normalize_value_blend_fracs(sf_wdl_frac, search_wdl_frac)
    return ValueBlendReadout(
        sf_wdl_frac=sf,
        search_wdl_frac=search,
        game_frac=game,
        sf_effective_rows=float(sf_effective_rows),
        search_effective_rows=float(search_effective_rows),
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
    SEE IT. Every leak term is a product ``frac x (1 - effective_frac)``, so it
    is 0 whenever the frac is 0 — including the configuration
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
        f"leaked={readout.leaked_to_outcome:.4f} "
        f"(sf {readout.leaked_from_sf:.4f} + search "
        f"{readout.leaked_from_search:.4f}), with effective label mass "
        f"sf={readout.sf_effective_rows:.0f}/{readout.batch_rows:.0f} "
        f"search={readout.search_effective_rows:.0f}/{readout.batch_rows:.0f}. "
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
    """Raise on either a label-to-outcome leak or an all-outcome value target.

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
        f"target on the RAW GAME OUTCOME through the label fallback{where}: "
        f"SF leaked {readout.leaked_from_sf:.4f} (realized sf_wdl_frac="
        f"{readout.sf_wdl_frac:.4f}, effective label mass "
        f"{readout.sf_effective_rows:.0f}/{readout.batch_rows:.0f}) and SEARCH "
        f"leaked {readout.leaked_from_search:.4f} (realized search_wdl_frac="
        f"{readout.search_wdl_frac:.4f}, effective label mass "
        f"{readout.search_effective_rows:.0f}/{readout.batch_rows:.0f}). "
        f"Intended outcome share was game_frac={readout.game_frac:.4f}; the "
        f"target actually being trained puts {readout.outcome_borne_frac:.4f} "
        "there. losses.py does this without an error or a warning. On "
        "lc0-derived shards set sf_wdl_frac: 0.0 and sf_wdl_frac_floor: 0.0, "
        "and give the freed share to search_wdl_frac (lc0's own "
        "best_q/best_d) — which the shards must then actually carry."
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
