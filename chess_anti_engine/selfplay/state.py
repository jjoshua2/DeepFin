"""Mutable state bundle for ``play_batch``.

Groups the parallel per-slot arrays/lists, shared caches, capability flags,
and stats accumulators that ``play_batch`` threads through every helper.

The C fast paths (``batch_process_ply``, ``batch_encode_146``,
``classify_games``, ``temperature_resample``) consume plain Python lists of
``CBoard`` objects and ``np.int8`` arrays — those layouts are preserved
verbatim so extraction stays zero-cost. Capability handles are
``Callable | None``; callers guard on the boolean ``has_c_ply`` /
``has_classify_c`` flags.

``_StatsAcc`` mirrors ``BatchStats``'s counter field names and exposes
``.to_batch_stats(...)`` to produce the immutable return value.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import chess
import numpy as np

from chess_anti_engine.inference import BatchEvaluator, LocalModelEvaluator
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.opening import OpeningConfig, make_starting_board
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishUCI
from chess_anti_engine.tablebase import SyzygyProbe

if TYPE_CHECKING:
    import torch

    from chess_anti_engine.encoding._lc0_ext import CBoard


# Soft-resign thresholds: if the network's win probability stays below
# SOFT_RESIGN_THRESHOLD for SOFT_RESIGN_CONSECUTIVE consecutive plies we
# downweight the sample (but still play it out — anti-engine training
# benefits from learning losing positions).
SOFT_RESIGN_THRESHOLD: float = 0.05
SOFT_RESIGN_CONSECUTIVE: int = 5


# ── Public stats types ──────────────────────────────────────────────────────
# Moved here from manager.py so finalize/network-turn extraction modules can
# construct them without a circular import back through manager.  Manager
# re-exports both names for backward compatibility with external callers
# (worker.py, bench scripts, tests).

@dataclass
class BatchStats:
    games: int
    positions: int
    w: int
    d: int
    l: int
    total_game_plies: int = 0
    adjudicated_games: int = 0
    tb_adjudicated_games: int = 0
    total_draw_games: int = 0
    selfplay_games: int = 0
    selfplay_adjudicated_games: int = 0
    selfplay_draw_games: int = 0
    curriculum_games: int = 0
    curriculum_adjudicated_games: int = 0
    curriculum_draw_games: int = 0
    checkmate_games: int = 0
    stalemate_games: int = 0
    plies_win: int = 0
    plies_draw: int = 0
    plies_loss: int = 0

    # Adaptive difficulty PID diagnostics (optional)
    sf_nodes: int | None = None
    sf_nodes_next: int | None = None
    pid_ema_winrate: float | None = None

    # Log-only: mean abs delta of SF's winrate-like eval over 6 plies.
    # When training only on network turns, this is computed between SF reply
    # evals attached to samples t and t+3 (i.e. 6 plies apart).
    sf_eval_delta6: float = 0.0
    sf_eval_delta6_n: int = 0


@dataclass
class CompletedGameBatch:
    """One completed game's worth of samples + per-game counters.

    Delivered to ``play_batch``'s ``on_game_complete`` callback.  Counters
    mirror the accumulator fields in ``BatchStats`` so the driver can sum
    them up to reconstruct the aggregate stats for the whole batch.
    """

    samples: list[ReplaySample]
    games: int = 1
    positions: int = 0
    w: int = 0
    d: int = 0
    l: int = 0
    total_game_plies: int = 0
    adjudicated_games: int = 0
    tb_adjudicated_games: int = 0
    total_draw_games: int = 0
    selfplay_games: int = 0
    selfplay_adjudicated_games: int = 0
    selfplay_draw_games: int = 0
    curriculum_games: int = 0
    curriculum_adjudicated_games: int = 0
    curriculum_draw_games: int = 0
    checkmate_games: int = 0
    stalemate_games: int = 0
    plies_win: int = 0
    plies_draw: int = 0
    plies_loss: int = 0


class _NetRecord:
    """Per-ply sample record.  Uses ``__slots__`` to avoid dict overhead."""
    __slots__ = (
        "x", "policy_probs", "net_wdl_est", "search_wdl_est",
        "pov_color", "ply_index", "has_policy", "priority",
        "sample_weight", "keep_prob", "legal_mask",
        "sf_policy_target", "sf_move_index", "sf_wdl",
        "sf_legal_mask",
    )

    x: np.ndarray
    policy_probs: np.ndarray
    net_wdl_est: np.ndarray
    search_wdl_est: np.ndarray
    pov_color: bool
    ply_index: int
    has_policy: bool
    priority: float
    sample_weight: float
    keep_prob: float
    legal_mask: np.ndarray | None
    sf_policy_target: np.ndarray | None
    sf_move_index: int | None
    sf_wdl: np.ndarray | None
    sf_legal_mask: np.ndarray | None

    def __init__(
        self, x, policy_probs, net_wdl_est, search_wdl_est,
        pov_color, ply_index, has_policy, priority,
        sample_weight, keep_prob, legal_mask=None,
        sf_policy_target=None, sf_move_index=None, sf_wdl=None,
    ):
        self.x = x
        self.policy_probs = policy_probs
        self.net_wdl_est = net_wdl_est
        self.search_wdl_est = search_wdl_est
        self.pov_color = pov_color
        self.ply_index = ply_index
        self.has_policy = has_policy
        self.priority = priority
        self.sample_weight = sample_weight
        self.keep_prob = keep_prob
        self.legal_mask = legal_mask
        self.sf_policy_target = sf_policy_target
        self.sf_move_index = sf_move_index
        self.sf_wdl = sf_wdl
        self.sf_legal_mask = None


@dataclass
class _StatsAcc:
    """Mutable accumulator for per-game counters.

    Field names match ``BatchStats`` so ``to_batch_stats`` is a near-direct
    copy.  Replaces the ``_st_*`` locals + 8 ``nonlocal`` declarations that
    previously threaded these counters through ``_finalize_game``.
    """

    w: int = 0
    d: int = 0
    l: int = 0
    total_game_plies: int = 0
    adjudicated_games: int = 0
    tb_adjudicated_games: int = 0
    total_draw_games: int = 0
    selfplay_games: int = 0
    selfplay_adjudicated_games: int = 0
    selfplay_draw_games: int = 0
    curriculum_games: int = 0
    curriculum_adjudicated_games: int = 0
    curriculum_draw_games: int = 0
    checkmate_games: int = 0
    stalemate_games: int = 0
    plies_win: int = 0
    plies_draw: int = 0
    plies_loss: int = 0

    # Log-only volatility metric: SF winrate delta across 6 plies.
    sf_d6_sum: float = 0.0
    sf_d6_n: int = 0

    def to_batch_stats(
        self, *, games: int, positions: int, sf_nodes: int | None,
    ) -> BatchStats:
        """Snapshot the accumulator into an immutable ``BatchStats``.

        ``games`` and ``positions`` come from the driver (``play_batch``);
        ``sf_nodes`` is read from the stockfish object at return time.
        The PID diagnostics are always ``None`` for the base run — the
        outer worker sets them after its adaptive-difficulty controller
        finishes its update.
        """
        mean_sf_d6 = (
            float(self.sf_d6_sum / max(1, self.sf_d6_n)) if self.sf_d6_n > 0 else 0.0
        )
        return BatchStats(
            games=int(games),
            positions=int(positions),
            w=self.w,
            d=self.d,
            l=self.l,
            total_game_plies=int(self.total_game_plies),
            adjudicated_games=int(self.adjudicated_games),
            tb_adjudicated_games=int(self.tb_adjudicated_games),
            total_draw_games=int(self.total_draw_games),
            selfplay_games=int(self.selfplay_games),
            selfplay_adjudicated_games=int(self.selfplay_adjudicated_games),
            selfplay_draw_games=int(self.selfplay_draw_games),
            curriculum_games=int(self.curriculum_games),
            curriculum_adjudicated_games=int(self.curriculum_adjudicated_games),
            curriculum_draw_games=int(self.curriculum_draw_games),
            checkmate_games=int(self.checkmate_games),
            stalemate_games=int(self.stalemate_games),
            plies_win=int(self.plies_win),
            plies_draw=int(self.plies_draw),
            plies_loss=int(self.plies_loss),
            sf_nodes=sf_nodes if (sf_nodes is not None and sf_nodes > 0) else None,
            sf_nodes_next=None,
            pid_ema_winrate=None,
            sf_eval_delta6=mean_sf_d6,
            sf_eval_delta6_n=int(self.sf_d6_n),
        )


@dataclass
class SelfplayState:
    """Bundle of mutable state driving one ``play_batch`` invocation.

    Attributes are grouped by concern:

    * **Config** — frozen dataclasses from the caller.
    * **Per-slot arrays** — numpy ``int8`` arrays consumed by the C fast
      paths; must stay contiguous and in-place for zero-copy.
    * **Per-slot lists** — Python-level state (boards, move history, samples).
    * **Shared caches** — NN-eval cache and persistent MCTS tree.
    * **C capability handles** — ``None`` when the extension is unavailable.
    * **Control counters** — rolling-batch recycling bookkeeping.
    * **Stats** — see ``_StatsAcc``.
    """

    # ── Config (frozen, not mutated) ─────────────────────────────────────────
    device: str
    rng: np.random.Generator
    stockfish: StockfishUCI | StockfishPool
    evaluator: BatchEvaluator
    # Raw module reference kept alongside the evaluator because the MCTS
    # entry points (``run_mcts_many`` / ``run_gumbel_root_many``) still take
    # ``model`` as a positional argument — the evaluator is passed via the
    # ``evaluator=`` kwarg.  ``None`` is only valid when a pre-built
    # evaluator is supplied and no non-evaluator MCTS path is triggered.
    model: torch.nn.Module | None
    opponent: OpponentConfig
    temp: TemperatureConfig
    search: SearchConfig
    opening: OpeningConfig
    diff_focus: DiffFocusConfig
    game: GameConfig
    batch_size: int
    continuous: bool
    target: int
    volatility_source: str  # "raw" or "search"
    base_nodes: int
    terminal_eval_nodes: int

    # ── Per-slot parallel arrays (C fast paths require np.int8) ──────────────
    done_arr: np.ndarray
    finalized_arr: np.ndarray
    net_color_arr: np.ndarray
    selfplay_arr: np.ndarray

    # ── Tablebase adjudication state (per-slot + shared probe) ───────────────
    tb_probe: SyzygyProbe | None
    tb_result_arr: list[str | None]
    tb_adj_roll_arr: np.ndarray

    # ── Per-slot lists ───────────────────────────────────────────────────────
    boards: list[chess.Board]
    cboards: list[CBoard]
    starting_boards: list[chess.Board] | None
    move_idx_history: list[list[int]]
    samples_per_game: list[list[Any]]  # list[_NetRecord]; avoid import cycle
    consecutive_low_winrate: list[int]
    last_net_full: list[bool]
    root_ids: list[int]

    # ── Shared caches (None when C tree unavailable) ─────────────────────────
    nn_cache: Any = None
    mcts_tree: Any = None

    # ── C capability flags + handles ─────────────────────────────────────────
    has_c_ply: bool = False
    has_classify_c: bool = False
    c_process_ply: Callable[..., Any] | None = None
    batch_enc_146: Callable[..., Any] | None = None
    c_classify: Callable[..., Any] | None = None
    c_temp_resample: Callable[..., Any] | None = None

    # ── Control counters (mutated by recycle_slot / main loop) ───────────────
    games_started: int = 0
    games_completed: int = 0

    # ── Stats accumulator ────────────────────────────────────────────────────
    stats: _StatsAcc = field(default_factory=_StatsAcc)

    @classmethod
    def create(
        cls,
        *,
        model: torch.nn.Module | None,
        device: str,
        rng: np.random.Generator,
        stockfish: StockfishUCI | StockfishPool,
        evaluator: BatchEvaluator | None,
        batch_size: int,
        continuous: bool,
        target: int,
        opponent: OpponentConfig,
        temp: TemperatureConfig,
        search: SearchConfig,
        opening: OpeningConfig,
        diff_focus: DiffFocusConfig,
        game: GameConfig,
    ) -> SelfplayState:
        """Build a ``SelfplayState`` from ``play_batch``'s arguments.

        Performs the same setup work (opening boards, CBoards, parallel
        arrays, cache construction, capability probing) that previously
        lived inline at the top of ``play_batch``.
        """
        # Local import to avoid import cycle; _lc0_ext is the C CBoard module.
        from chess_anti_engine.encoding._lc0_ext import CBoard as _CBoard

        if evaluator is None:
            if model is None:
                raise ValueError("play_batch requires model or evaluator")
            evaluator = LocalModelEvaluator(model, device=device)

        boards = [make_starting_board(rng=rng, cfg=opening) for _ in range(batch_size)]
        # Use from_board (not from_raw) to preserve ply count + history from openings.
        cboards = [_CBoard.from_board(b) for b in boards]
        starting_boards = [b.copy() for b in boards] if game.syzygy_path else None

        done_arr = np.zeros(batch_size, dtype=np.int8)
        finalized_arr = np.zeros(batch_size, dtype=np.int8)
        # Alternate which color the network plays so it sees both perspectives.
        net_color_arr = np.array(
            [1 if (i % 2 == 0) else 0 for i in range(batch_size)],
            dtype=np.int8,
        )
        sp_frac = max(0.0, min(1.0, float(game.selfplay_fraction)))
        selfplay_arr = np.array(
            [1 if rng.random() < sp_frac else 0 for _ in range(batch_size)],
            dtype=np.int8,
        )

        # TB state. When ``syzygy_adjudicate`` is on, we end games early at the
        # first TB-eligible position and stash the TB-proven result — saves the
        # compute that would otherwise be spent on MCTS sims through a
        # known-result endgame. When ``syzygy_in_search`` is on, the probe is
        # also passed to MCTS to override NN wdl logits at TB-eligible leaves.
        tb_probe: SyzygyProbe | None = None
        if game.syzygy_path and (game.syzygy_adjudicate or game.syzygy_in_search):
            tb_probe = SyzygyProbe(game.syzygy_path)
        tb_result_arr: list[str | None] = [None] * batch_size
        # Per-game adjudication roll. 1 = this game gets TB-adjudicated when
        # eligible; 0 = play through. Re-rolled on slot recycle below from
        # ``syzygy_adjudicate_fraction``, which lets the NN keep training on
        # endgame positions instead of silently losing endgame skill.
        tb_adj_roll_arr = np.zeros(batch_size, dtype=np.int8)
        if tb_probe is not None and game.syzygy_adjudicate:
            tb_adj_frac = max(0.0, min(1.0, float(game.syzygy_adjudicate_fraction)))
            for _i in range(batch_size):
                if rng.random() < tb_adj_frac:
                    tb_adj_roll_arr[_i] = 1

        move_idx_history: list[list[int]] = [[] for _ in range(batch_size)]
        samples_per_game: list[list[Any]] = [[] for _ in range(batch_size)]
        consecutive_low_winrate = [0] * batch_size
        last_net_full = [True] * batch_size
        root_ids = [-1] * batch_size

        # NN eval cache + persistent MCTS tree (C-only).
        nn_cache = None
        mcts_tree = None
        try:
            from chess_anti_engine.mcts._mcts_tree import (
                MCTSTree as _MCTSTree,
            )
            from chess_anti_engine.mcts._mcts_tree import (
                NNCache as _NNCache,
            )

            nn_cache = _NNCache(1 << 17)  # 131072 slots
            mcts_tree = _MCTSTree()
        except ImportError:
            pass

        # Probe C per-ply fast path.  ``batch_process_ply`` and ``batch_encode_146``
        # live in the same C module; importing together keeps the two
        # ``None``-fallback paths coupled (prevents drift where only one is
        # available, which would break the xs_batch invariant in the network
        # turn).
        c_process_ply: Callable[..., Any] | None = None
        batch_enc_146: Callable[..., Any] | None = None
        has_c_ply = False
        try:
            from chess_anti_engine.mcts._mcts_tree import (
                batch_encode_146 as _batch_enc_146,
            )
            from chess_anti_engine.mcts._mcts_tree import (
                batch_process_ply as _c_process_ply,
            )

            c_process_ply = _c_process_ply
            batch_enc_146 = _batch_enc_146
            has_c_ply = True
        except ImportError:
            logging.getLogger("chess_anti_engine.selfplay").warning(
                "C per-ply fast path unavailable (batch_process_ply/batch_encode_146 "
                "not importable from _mcts_tree); falling back to Python. Rebuild "
                "the C extension for production.",
            )

        # Probe C classify/resample fast paths.
        c_classify: Callable[..., Any] | None = None
        c_temp_resample: Callable[..., Any] | None = None
        has_classify_c = False
        try:
            from chess_anti_engine.mcts._mcts_tree import (
                classify_games as _c_classify,
            )
            from chess_anti_engine.mcts._mcts_tree import (
                temperature_resample as _c_temp_resample,
            )

            c_classify = _c_classify
            c_temp_resample = _c_temp_resample
            has_classify_c = True
        except ImportError:
            pass

        # Normalize volatility source.
        vs = str(game.volatility_source).lower().strip()
        if vs not in ("raw", "search"):
            vs = "raw"

        base_nodes = int(getattr(stockfish, "nodes", 0) or 0)
        terminal_eval_nodes = (5 * base_nodes) if base_nodes > 0 else 1000

        return cls(
            device=device,
            rng=rng,
            stockfish=stockfish,
            evaluator=evaluator,
            model=model,
            opponent=opponent,
            temp=temp,
            search=search,
            opening=opening,
            diff_focus=diff_focus,
            game=game,
            batch_size=batch_size,
            continuous=continuous,
            target=target,
            volatility_source=vs,
            base_nodes=base_nodes,
            terminal_eval_nodes=terminal_eval_nodes,
            done_arr=done_arr,
            finalized_arr=finalized_arr,
            net_color_arr=net_color_arr,
            selfplay_arr=selfplay_arr,
            tb_probe=tb_probe,
            tb_result_arr=tb_result_arr,
            tb_adj_roll_arr=tb_adj_roll_arr,
            boards=boards,
            cboards=cboards,
            starting_boards=starting_boards,
            move_idx_history=move_idx_history,
            samples_per_game=samples_per_game,
            consecutive_low_winrate=consecutive_low_winrate,
            last_net_full=last_net_full,
            root_ids=root_ids,
            nn_cache=nn_cache,
            mcts_tree=mcts_tree,
            has_c_ply=has_c_ply,
            has_classify_c=has_classify_c,
            c_process_ply=c_process_ply,
            batch_enc_146=batch_enc_146,
            c_classify=c_classify,
            c_temp_resample=c_temp_resample,
            games_started=batch_size,
        )

    def net_color(self, i: int) -> chess.Color:
        """Return ``chess.Color`` for slot ``i``'s network-assigned color."""
        return chess.WHITE if self.net_color_arr[i] else chess.BLACK

    def classify_active_slots(
        self,
    ) -> tuple[list[int], list[int], list[int], bool]:
        """Partition active slots by whose move is next.

        Returns ``(net_idxs, selfplay_opp_idxs, curriculum_opp_idxs,
        all_finalized)``:

        * ``net_idxs`` — slots whose side to move is the network-assigned
          color (played via MCTS / the network).
        * ``selfplay_opp_idxs`` — selfplay slots where the "opponent" side
          to move is still the network (these also run through MCTS,
          merged into the network turn).
        * ``curriculum_opp_idxs`` — non-selfplay slots where the side to
          move is the curriculum opponent (handled by Stockfish).
        * ``all_finalized`` — ``True`` when every slot has finalized and
          the main loop can break early.

        Dispatches to the C ``classify_games`` fast path when available;
        otherwise falls back to the Python per-slot loop (which also
        marks timed-out slots as done, matching the original semantics).
        """
        if self.has_classify_c:
            assert self.c_classify is not None
            _c_net, _c_sp, _c_cur = self.c_classify(
                self.cboards,
                self.net_color_arr,
                self.done_arr,
                self.finalized_arr,
                self.selfplay_arr,
                int(self.game.max_plies),
            )
            net_idxs = _c_net.tolist()
            sp_idxs = _c_sp.tolist()
            cur_idxs = _c_cur.tolist()
            all_done = (
                not net_idxs
                and not sp_idxs
                and not cur_idxs
                and not np.any(self.finalized_arr == 0)
            )
            return net_idxs, sp_idxs, cur_idxs, all_done

        # Python fallback: mark timed-out slots + classify by turn.
        active_idxs = [
            i for i in range(self.batch_size) if not self.finalized_arr[i]
        ]
        if not active_idxs:
            return [], [], [], True
        for i in active_idxs:
            if not self.done_arr[i] and (
                self.cboards[i].is_game_over()
                or self.cboards[i].ply >= int(self.game.max_plies)
            ):
                self.done_arr[i] = 1
        live = [
            i
            for i in range(self.batch_size)
            if not self.finalized_arr[i] and not self.done_arr[i]
        ]
        net_idxs = [i for i in live if self.cboards[i].turn == self.net_color(i)]
        sp_idxs = [
            i
            for i in live
            if self.cboards[i].turn != self.net_color(i) and self.selfplay_arr[i]
        ]
        cur_idxs = [
            i
            for i in live
            if self.cboards[i].turn != self.net_color(i) and not self.selfplay_arr[i]
        ]
        return net_idxs, sp_idxs, cur_idxs, False

    def recycle_slot(self, i: int) -> None:
        """Reset slot ``i`` to a fresh opening position.

        Called when a game finalizes in continuous mode (or when more
        games remain in the target).  Mirrors the original inline
        ``_recycle_slot`` closure exactly.
        """
        # Local import mirrors create() above.
        from chess_anti_engine.encoding._lc0_ext import CBoard as _CBoard

        self.boards[i] = make_starting_board(rng=self.rng, cfg=self.opening)
        self.cboards[i] = _CBoard.from_board(self.boards[i])
        if self.starting_boards is not None:
            self.starting_boards[i] = self.boards[i].copy()
        self.move_idx_history[i] = []
        self.done_arr[i] = 0
        self.finalized_arr[i] = 0
        self.net_color_arr[i] = 1 if (self.games_started % 2 == 0) else 0
        sp_frac = max(0.0, min(1.0, float(self.game.selfplay_fraction)))
        self.selfplay_arr[i] = 1 if self.rng.random() < sp_frac else 0
        # Clear TB adjudication stash from the previous game in this slot;
        # without this, the stale non-None value would make the adjudicator
        # skip the new game forever. Re-roll the per-game "do I adjudicate"
        # flag from the configured fraction.
        self.tb_result_arr[i] = None
        if self.tb_probe is not None and self.game.syzygy_adjudicate:
            self.tb_adj_roll_arr[i] = 1 if self.rng.random() < max(
                0.0, min(1.0, float(self.game.syzygy_adjudicate_fraction)),
            ) else 0
        self.samples_per_game[i] = []
        self.consecutive_low_winrate[i] = 0
        self.last_net_full[i] = True
        self.root_ids[i] = -1  # Reset tree reuse for new game.
        self.games_started += 1

    def replay_board(self, i: int) -> chess.Board:
        """Return a ``chess.Board`` at the current position of slot ``i``.

        When the C per-ply fast path is active ``self.boards[i]`` is kept at
        the opening position and we replay ``self.move_idx_history[i]``.
        In the Python fallback the board has already been mutated in place,
        so it's returned as-is.
        """
        from chess_anti_engine.moves import index_to_move

        if not self.has_c_ply:
            return self.boards[i]
        b = self.boards[i].copy(stack=False)
        for mi in self.move_idx_history[i]:
            b.push(index_to_move(mi, b))
        return b


__all__ = [
    "SOFT_RESIGN_CONSECUTIVE",
    "SOFT_RESIGN_THRESHOLD",
    "BatchStats",
    "CompletedGameBatch",
    "SelfplayState",
    "_NetRecord",
    "_StatsAcc",
]
