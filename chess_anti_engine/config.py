from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StockfishConfig:
    path: str
    nodes: int = 2000
    multipv: int = 5
    hash_mb: int = 16

    # Adaptive difficulty PID (spec)
    pid_enabled: bool = True
    pid_target_winrate: float = 0.53
    pid_ema_alpha: float = 0.03
    pid_deadzone: float = 0.04
    pid_rate_limit: float = 0.10
    pid_min_games_between_adjust: int = 30
    pid_kp: float = 1.5
    pid_ki: float = 0.10
    pid_kd: float = 0.0
    pid_integral_clamp: float = 1.0
    pid_min_nodes: int = 250
    pid_max_nodes: int = 1_000_000


@dataclass
class SelfPlayConfig:
    games_per_iter: int = 10
    max_plies: int = 200

    # AlphaZero/Leela-style exploration: sample from MCTS visits with temperature.
    # Optionally drop temperature after a fixed number of (full) moves.
    # (In our selfplay loop, one iteration advances both sides, so this is effectively “move”.)
    temperature: float = 1.0

    # Step schedule (legacy / simple): after N moves, set temperature to `temperature_after`.
    temperature_drop_plies: int = 0  # 0 disables
    temperature_after: float = 0.0

    # LC0-like linear decay schedule: starting at move `temperature_decay_start_move`,
    # linearly decay from `temperature` down to `temperature_endgame` over `temperature_decay_moves`.
    # If temperature_decay_moves <= 0, this schedule is disabled.
    temperature_decay_start_move: int = 20
    temperature_decay_moves: int = 60
    temperature_endgame: float = 0.6

    # Opening diversification
    opening_book_path: str | None = None
    opening_book_max_plies: int = 4
    opening_book_max_games: int = 200_000
    opening_book_prob: float = 1.0
    random_start_plies: int = 0

    # Syzygy tablebase rescoring path (None to disable)
    syzygy_path: str | None = None
    syzygy_policy: bool = False  # also rescore policy with DTZ-optimal best move

    # Stockfish WDL confidence required to adjudicate max_plies timeouts as decisive.
    timeout_adjudication_threshold: float = 0.90

    # SF-policy target shaping on SF turns (MultiPV-derived + label smoothing)
    sf_policy_temp: float = 0.25
    sf_policy_label_smooth: float = 0.05


@dataclass
class TrainConfig:
    device: str = "cpu"  # "cuda" if available
    optimizer: str = "nadamw"  # nadamw | adamw | muon | soap
    lr: float = 3e-4
    batch_size: int = 128
    train_steps_per_iter: int = 200
    grad_clip: float = 10.0

    # Gradient accumulation: effective batch = batch_size * accum_steps
    accum_steps: int = 1

    # LR schedule: linear warmup then cosine annealing with warm restarts
    warmup_steps: int = 1500
    lr_eta_min: float = 1e-5
    lr_T0: int = 5000
    lr_T_mult: int = 2

    # torch.compile for training throughput (spec cites ~75% speedup)
    use_compile: bool = False

    # Stochastic Weight Averaging: average weights periodically for smoother exports.
    # swa_start=0 disables SWA.
    swa_start: int = 0
    swa_freq: int = 50


@dataclass
class EvalConfig:
    games_per_iter: int = 10
    sf_nodes: int = 2000


@dataclass
class RunConfig:
    work_dir: Path = Path("runs")
    seed: int = 0

    stockfish: StockfishConfig | None = None
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    iterations: int = 50
    replay_capacity: int = 200_000
