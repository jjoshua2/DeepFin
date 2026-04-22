from __future__ import annotations

from dataclasses import dataclass

from chess_anti_engine.train.targets import DEFAULT_CATEGORICAL_BINS


@dataclass(frozen=True)
class OpponentConfig:
    wdl_regret_limit: float | None = None


@dataclass(frozen=True)
class TemperatureConfig:
    temperature: float = 1.0
    drop_plies: int = 0
    after: float = 0.0
    decay_start_move: int = 20
    decay_moves: int = 60
    endgame: float = 0.6


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 50
    mcts_type: str = "puct"
    playout_cap_fraction: float = 0.25
    fast_simulations: int = 8
    fpu_reduction: float = 1.2
    fpu_at_root: float = 1.0


@dataclass(frozen=True)
class DiffFocusConfig:
    enabled: bool = True
    q_weight: float = 6.0
    pol_scale: float = 3.5
    slope: float = 3.0
    min_keep: float = 0.025


@dataclass(frozen=True)
class GameConfig:
    max_plies: int = 240
    selfplay_fraction: float = 0.0
    sf_policy_temp: float = 0.25
    sf_policy_label_smooth: float = 0.05
    soft_policy_temp: float = 2.0
    timeout_adjudication_threshold: float = 0.90
    volatility_source: str = "raw"
    syzygy_path: str | None = None
    syzygy_policy: bool = False
    # If true, end the game as soon as the position becomes TB-eligible and
    # use the TB-proven WDL as the outcome. Saves the rest of the MCTS work
    # that would have been spent playing out a known-result endgame.
    syzygy_adjudicate: bool = False
    # If true, override leaf WDL logits during MCTS with TB truth (UCI-style
    # in-tree probing). Pinned Q values collapse noisy endgame search paths.
    syzygy_in_search: bool = False
    categorical_bins: int = DEFAULT_CATEGORICAL_BINS
    hlgauss_sigma: float = 0.04


__all__ = [
    "DiffFocusConfig",
    "GameConfig",
    "OpponentConfig",
    "SearchConfig",
    "TemperatureConfig",
]
