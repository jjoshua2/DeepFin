"""Selfplay TB contract validation, without importing native/model code."""
from __future__ import annotations

from collections.abc import Mapping


def syzygy_outcome_mode(value: object) -> str:
    if not isinstance(value, str) or value not in ("theoretical_wdl", "rule50_match_v1"):
        raise ValueError("syzygy_outcome_mode must be theoretical_wdl or rule50_match_v1")
    return value


def validate_selfplay_syzygy(config: Mapping[str, object]) -> None:
    """Validate effective strict options; legacy settings retain their behavior."""
    mode = syzygy_outcome_mode(config.get("syzygy_outcome_mode", "theoretical_wdl"))
    if mode != "rule50_match_v1":
        return
    path = config.get("syzygy_path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("rule50_match_v1 requires syzygy_path")
    sf_path = config.get("stockfish_syzygy_path") or path
    if sf_path != path:
        raise ValueError("rule50_match_v1 requires the same full Stockfish/Syzygy path")
    for key in ("syzygy_adjudicate", "syzygy_in_search"):
        if config.get(key, False) is not True:
            raise ValueError(f"rule50_match_v1 requires {key}=true")
    if config.get("syzygy_adjudicate_fraction", 1.0) != 1.0:
        raise ValueError("rule50_match_v1 requires syzygy_adjudicate_fraction=1")
    if type(config.get("syzygy_rescore_policy", False)) is not bool:
        raise ValueError("syzygy_rescore_policy must be a bool")
    if isinstance(config.get("syzygy_adjudicate_fraction"), bool):
        raise ValueError("syzygy_adjudicate_fraction must be numeric, not a bool")



def refuse_strict_worker_session(config: Mapping[str, object]) -> None:
    """The initial strict producer is finite play_batch, without disk resume."""
    validate_selfplay_syzygy(config)
    if config.get("syzygy_outcome_mode", "theoretical_wdl") == "rule50_match_v1":
        raise ValueError(
            "rule50_match_v1 supports finite play_batch only; continuous/resumed "
            "Tune/Worker sessions cannot preserve the strict outcome witness"
        )
