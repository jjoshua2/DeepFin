from __future__ import annotations


# Runtime-mutable trainer attributes sourced from config each iteration.
# Shared by YAML allowlisting, live trainer sync, and salvage-donor overlay.
TRAINER_WEIGHT_KEYS: tuple[str, ...] = (
    "w_policy",
    "w_soft",
    "w_future",
    "w_wdl",
    "w_sf_move",
    "w_sf_eval",
    "w_categorical",
    "w_volatility",
    "w_sf_volatility",
    "w_moves_left",
    "sf_wdl_frac",
    "search_wdl_frac",
    "sf_wdl_conf_power",
    "sf_wdl_draw_scale",
    "sf_wdl_temperature",
    "sf_search_dampen_sf_low",
    "sf_search_dampen_sf_high",
)
