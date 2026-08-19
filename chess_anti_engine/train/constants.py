from __future__ import annotations

REGRET_TO_Q_SCALE = 2.0

# Max cp-regret used to normalize the per-move SF regret vector (``sf_p0_regret``).
# A move 1000+ cp worse than SF's best becomes 1.0 (maximally bad); the best move
# is 0.0. Legal moves absent from the MultiPV default to the midpoint between the
# worst surfaced regret and 1.0, which is >= 0.5 always.
#
# ⚑ SINGLE HOME. It lives in this dependency-free module because THREE stages
# now divide by it: `selfplay/finalize.py` (which WRITES the vector and re-exports
# this name, so every existing `finalize.SF_OWN_REGRET_CAP_CP` reader is
# unchanged) and `train/losses.py` (which READS it, to turn a cp window into the
# vector's units). A mirrored copy in the loss would be a second definition of
# the unit the two stages have to agree on -- and a cp window compared against a
# vector normalized by a DIFFERENT cap is silently wrong, never loud. `train/`
# rather than `selfplay/` because the loss must not import the selfplay package
# (it pulls in the C extension).
SF_OWN_REGRET_CAP_CP = 1000.0

# Root candidate width of the Gumbel search (`gumbel_topk`), and the ONE place
# the raw yaml value is turned into the integer both consumers use.
#
# It lives in this dependency-free module because two unrelated stages now
# normalize it: `tune/trial_config.py` (which publishes it to the search) and
# `train/trainer.py` (which derives the `sf_policy_floor_tau` default `1/topk`
# from it). A second `max(1, int(...))` written at the second site is the
# ordinary way those two drift into disagreeing about what the running topk is,
# and the loss would then floor priors against a width the search does not use --
# a number that stays plausible while guaranteeing nothing.
DEFAULT_GUMBEL_TOPK = 16


def normalize_gumbel_topk(value: object) -> int:
    """Root candidate width as the search actually uses it: an int >= 1."""
    return max(1, int(value))  # pyright: ignore[reportArgumentType]


FUTURE_REGRET_FIELDS = {
    "sum": ("future_sf_regret_sum", "has_future_sf_regret_sum"),
    "d95": ("future_sf_regret_d95", "has_future_sf_regret_d95"),
    "d98": ("future_sf_regret_d98", "has_future_sf_regret_d98"),
    "max": ("future_sf_regret_max", "has_future_sf_regret_max"),
    "h4": ("future_sf_regret_h4", "has_future_sf_regret_h4"),
    "h6": ("future_sf_regret_h6", "has_future_sf_regret_h6"),
    "h12": ("future_sf_regret_h12", "has_future_sf_regret_h12"),
    "h24": ("future_sf_regret_h24", "has_future_sf_regret_h24"),
    "h50": ("future_sf_regret_h50", "has_future_sf_regret_h50"),
}


def future_regret_field_names(source: str) -> tuple[str, str]:
    source_key = str(source)
    if source_key not in FUTURE_REGRET_FIELDS:
        allowed = ", ".join(FUTURE_REGRET_FIELDS)
        raise ValueError(
            f"unknown adjusted_wdl_regret_source {source_key!r}; expected one of: {allowed}"
        )
    return FUTURE_REGRET_FIELDS[source_key]
