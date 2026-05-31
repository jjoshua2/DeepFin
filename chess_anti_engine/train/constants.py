from __future__ import annotations

REGRET_TO_Q_SCALE = 2.0
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
