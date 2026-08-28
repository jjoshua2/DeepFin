from __future__ import annotations

import dataclasses


from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.mcts import GumbelConfig, MCTSConfig
from chess_anti_engine.mcts.gumbel import (
    PLAY_SEARCH_DEFAULTS,
    run_gumbel_root_many,
    volatility_search_enabled,
    warn_volatility_python_path,
)
from chess_anti_engine.mcts.puct import run_mcts_many
from chess_anti_engine.moves import index_to_move_fast, index_to_move_strict
from chess_anti_engine.selfplay.opening import OpeningConfig, make_starting_board

try:
    from chess_anti_engine.mcts.puct_c import run_mcts_many_c as _run_mcts_many_c
    _HAS_C_TREE = True
except ImportError:
    _HAS_C_TREE = False

try:
    from chess_anti_engine.mcts.gumbel_c import (
        run_gumbel_root_many_c as _run_gumbel_root_many_c,
    )
    _HAS_GUMBEL_C = True
except ImportError:
    _HAS_GUMBEL_C = False

if TYPE_CHECKING:
    from chess_anti_engine.inference import BatchEvaluator
    from chess_anti_engine.mcts.gumbel_c import (
        run_gumbel_root_many_c as _run_gumbel_root_many_c,
    )
    from chess_anti_engine.mcts.puct_c import (
        run_mcts_many_c as _run_mcts_many_c,
    )


@dataclass(frozen=True)
class MatchStats:
    games: int
    max_plies: int

  # From model_a perspective
    a_win: int
    a_draw: int
    a_loss: int

    a_as_white: int
    a_as_black: int


def result_from_a_pov(result: str, *, a_is_white: bool) -> int:
    """Map game result to model-a outcome.

    Returns:
        1 for model-a win, 0 for draw, -1 for model-a loss.

    python-chess returns "*" when a game is truncated before reaching a
    terminal result (for example when max_plies is hit). Treat this as draw
    rather than a decisive result.
    """
    if result in {"1/2-1/2", "*"}:
        return 0

    white_won = result == "1-0"
    a_won = (white_won and a_is_white) or ((not white_won) and (not a_is_white))
    return 1 if a_won else -1


def pick_moves_for_boards(
    model: torch.nn.Module, sub_boards: list[chess.Board], *,
    device: str, rng: np.random.Generator,
    mcts_type: str, mcts_simulations: int, temperature: float, c_puct: float,
    gumbel_add_noise: bool,
    volatility_q_scale: float = 0.0,
    volatility_fpu: float = 0.0,
    volatility_anchor: float | None = None,
    gumbel_overrides: dict[str, float] | None = None,
    gumbel_vloss_weight: int = 0,
    gumbel_target_batch: int = 0,
    evaluator: BatchEvaluator | None = None,
) -> list[int]:
    """Run gumbel- or PUCT-MCTS for one model on a list of boards.

    Non-zero ``volatility_q_scale``/``volatility_fpu`` switch on
    volatility-aware Gumbel search, which forces the Python search path
    (logged once). ``volatility_anchor`` overrides the GumbelConfig default
    dataset-mean anchor when given.

    ``gumbel_vloss_weight`` / ``gumbel_target_batch`` are the two C-path search
    controls that are NOT ``GumbelConfig`` fields, so ``gumbel_overrides``
    cannot carry them. Until they were added here this function passed neither,
    both took the C defaults of 0, and the whole arena path (arena_standard
    matched_sims) ran the pre-C17 duplicate-leaf search while production
    selfplay ran ``gumbel_vloss_weight: 1`` — production played a different move
    from the arena on 10% of positions at 32 sims and 27.5% at 256
    (docs/experiment_ledger.md 2026-07-28). The defaults stay 0 so no existing
    caller changes behaviour; callers that must match production pass the
    values from the production ``SearchConfig``.

    They apply to the C path only. The Python reference search
    (``run_gumbel_root_many``, used when the extension is missing or volatility
    search is on) has no equivalent, so a non-zero value there is a request the
    search cannot honour and is rejected rather than dropped.

    ``evaluator`` is an optional LONG-LIVED batch evaluator to run the forwards
    through. ``None`` (the default) is today's behaviour and every other caller
    keeps it: each search entry point then builds a THROWAWAY
    ``LocalModelEvaluator`` per call (``mcts/gumbel_c.run_gumbel_root_many_c``
    and the three other entries reached below). On CUDA that is a fresh, lazily
    created stream per side per ply; torch hands streams out of a fixed
    round-robin pool of 32 per device and the caching allocator partitions
    segments BY STREAM, so a two-model arena cycles the whole pool in 16 plies
    and every stream retains a full forward's working set — reserved VRAM
    inflates until the card OOMs. A hoisted evaluator also carries
    ``_max_batch``, which is the ONLY thing that caps the leaf batch
    (``gumbel_c`` mins its leaf cap against it); a throwaway
    ``LocalModelEvaluator`` has no such attribute, so the leaf batch grows with
    concurrency without bound.

    It is forwarded to whichever search path this call takes — all four entry
    points accept ``evaluator`` — so it is never accepted here and quietly
    dropped below. ⚑ A ``DirectGPUEvaluator`` RAISES when a submit exceeds its
    ``max_batch`` (``get_input_buffer`` / ``evaluate_encoded``) and the C gumbel
    ROOT submit is NOT bucketed against it, so size ``max_batch`` at or above
    the largest board count any one call will be handed.
    """
    input_history_encoding = str(getattr(model, "input_history_encoding", "legacy"))
    input_extra_features = str(getattr(model, "input_extra_features", "v1"))
    policy_encoding = str(getattr(model, "policy_encoding", "lc0_1858"))
    use_dynamic_relations = bool(getattr(model, "use_dynamic_relations", False))
    # The repetition-plane fix is process-global in the C encoders, so apply
    # THIS model's value before encoding its moves: same-process arenas
    # (scripts/arena_standard.py matched_sims) alternate models per move
    # cycle, and the last build_model otherwise wins for both sides.
    # Idempotent and cheap when unchanged.
    # boards_discarded: every C search entry point rebuilds its CBoards from the
    # python board at the start of each call (gumbel_c/puct_c ``CBoard.from_board``),
    # so no board pushed under the other model's flag value outlives this apply.
    # A future long-lived-CBoard optimisation on this path invalidates that and
    # must revisit the guard rather than keep the keyword.
    rep_fix.apply(
        bool(getattr(model, "history_rep_fix", False)), boards_discarded=True,
    )
    if str(mcts_type) == "gumbel":
        gumbel_cfg = GumbelConfig(
            simulations=int(mcts_simulations), temperature=float(temperature),
            add_noise=bool(gumbel_add_noise),
            input_history_encoding=input_history_encoding,
            input_extra_features=input_extra_features,
            policy_encoding=policy_encoding,
            compute_relations=use_dynamic_relations,
            volatility_q_scale=float(volatility_q_scale),
            volatility_fpu=float(volatility_fpu),
        )
        if volatility_anchor is not None:
            gumbel_cfg = dataclasses.replace(
                gumbel_cfg, volatility_anchor=float(volatility_anchor),
            )
        if gumbel_overrides:
            # Per-side gumbel knob overrides for config sweeps (arena_standard):
            # c_scale, c_visit, c_visit_root, c_scale_root, topk, halving_div, etc.
            # NOT c_puct/fpu_reduction/cpuct_*: inert in a Gumbel search, and
            # arena_standard now refuses them (mcts.gumbel.INERT_GUMBEL_KNOBS).
            gumbel_cfg = dataclasses.replace(gumbel_cfg, **gumbel_overrides)
        if volatility_search_enabled(gumbel_cfg):
            warn_volatility_python_path()
        if _HAS_GUMBEL_C and not volatility_search_enabled(gumbel_cfg):
            result = _run_gumbel_root_many_c(
                model, sub_boards, device=device, rng=rng, cfg=gumbel_cfg,
                evaluator=evaluator,
                allow_terminal_root_shortcuts=True,
                vloss_weight=int(gumbel_vloss_weight),
                target_batch=int(gumbel_target_batch),
            )
        else:
            if int(gumbel_vloss_weight) or int(gumbel_target_batch):
                # Fail loud instead of quietly searching without them: silently
                # dropping these is the exact defect this plumbing fixes, and a
                # dropped vloss_weight changes a quarter of the moves at 256 sims.
                raise ValueError(
                    "gumbel_vloss_weight/gumbel_target_batch are C-path only "
                    "(run_gumbel_root_many has no equivalent); got "
                    f"vloss_weight={int(gumbel_vloss_weight)} "
                    f"target_batch={int(gumbel_target_batch)} on the Python path "
                    f"(_HAS_GUMBEL_C={_HAS_GUMBEL_C}, volatility="
                    f"{volatility_search_enabled(gumbel_cfg)}). Pass 0 for both, "
                    "or run the C path. In an arena, zero them on BOTH sides "
                    "(--cand-vloss-weight 0 AND --ref-vloss-weight 0): zeroing "
                    "only the candidate leaves the reference on the C path at "
                    "the shape's vloss_weight, so the volatility A/B is "
                    "confounded by virtual loss."
                )
            result = run_gumbel_root_many(
                model, sub_boards, device=device, rng=rng, cfg=gumbel_cfg,
                evaluator=evaluator,
            )
        _probs, actions, _values, _masks = result[:4]
    else:
        puct_fn = _run_mcts_many_c if _HAS_C_TREE else run_mcts_many
        _probs, actions, _values, _masks = puct_fn(
            model, sub_boards, device=device, rng=rng,
            cfg=MCTSConfig(
                simulations=int(mcts_simulations), temperature=float(temperature),
                c_puct=float(c_puct), dirichlet_eps=0.0,
                input_history_encoding=input_history_encoding,
                input_extra_features=input_extra_features,
                policy_encoding=policy_encoding,
                compute_relations=use_dynamic_relations,
            ),
            evaluator=evaluator,
        )
    return [int(a) for a in actions]


def apply_actions_to_boards(
    boards: list[chess.Board], idxs: list[int], actions: list[int], *, strict: bool,
) -> None:
    """Push each chosen action onto its board.

    ``strict`` is REQUIRED and deliberately has no default, because the two
    modes answer different questions and the wrong one is silent:

    * MEASUREMENT (``True``) — an id that names no legal move raises
      ``ActionDecodeError``. An arena that instead plays an unrelated legal move
      keeps scoring games, and the action-space regression that caused it
      reaches the operator as unattributable Elo loss.
    * GAME GENERATION (``False``) — substitute the first legal move and keep
      playing, so one bad id costs a move rather than a session. The
      substitution is counted by ``decode_fallback_count()``.

    There is no legality re-check here on purpose. Every return path of
    ``index_to_move_fast`` is already a legal move — it substitutes internally —
    so the ``mv not in legal_moves`` guard that used to sit in this loop could
    never fire, and its presence is what made the substitution look handled.
    """
    for i, a in zip(idxs, actions, strict=True):
        board = boards[i]
        board.push(
            index_to_move_strict(int(a), board)
            if strict
            else index_to_move_fast(int(a), board),
        )


def split_active_by_side_to_move(
    active: list[int], boards: list[chess.Board], a_plays_white: list[bool],
) -> tuple[list[int], list[int]]:
    """Partition active slots by which model (a/b) is to move."""
    a_to_move: list[int] = []
    b_to_move: list[int] = []
    for i in active:
        a_is_white = bool(a_plays_white[i])
        a_moves_now = (
            (boards[i].turn == chess.WHITE and a_is_white)
            or (boards[i].turn == chess.BLACK and not a_is_white)
        )
        (a_to_move if a_moves_now else b_to_move).append(i)
    return a_to_move, b_to_move


def _tally_match_results(
    boards: list[chess.Board], a_plays_white: list[bool],
) -> tuple[int, int, int]:
    """Count (a_win, a_draw, a_loss) over the finished boards."""
    a_win = a_draw = a_loss = 0
    for i, b in enumerate(boards):
        outcome = result_from_a_pov(
            b.result(claim_draw=True), a_is_white=bool(a_plays_white[i]),
        )
        if outcome == 0:
            a_draw += 1
        elif outcome > 0:
            a_win += 1
        else:
            a_loss += 1
    return a_win, a_draw, a_loss


def play_match_batch(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    *,
    device: str,
    rng: np.random.Generator,
    games: int,
    max_plies: int,
    a_plays_white: list[bool] | None = None,
    mcts_type: str = "puct",
    mcts_simulations: int = 200,
    mcts_simulations_a: int | None = None,
    mcts_simulations_b: int | None = None,
    temperature: float = 0.1,
    c_puct: float = 2.5,
    gumbel_add_noise: bool = True,
    opening_cfg: OpeningConfig | None = None,
    gumbel_overrides: dict[str, float] | None = None,
) -> MatchStats:
    """Play model-vs-model matches.

    `a_plays_white[i]` controls which side model_a plays in game i.

    `opening_cfg` controls opening diversification — pass an OpeningConfig with
    a book path or random_start_plies so games don't all start from the same position.
    Defaults to 2 random start plies if not provided.

    `gumbel_overrides` tunes the Gumbel search knobs. It defaults to
    `PLAY_SEARCH_DEFAULTS`, because every caller of this function is an
    arena/gate/eval path and none of them records training rows: the in-loop
    gate (`chess_anti_engine/arena.py`), the worker's arena task, and
    `scripts/match_checkpoints.py`. Passing `None` previously meant a bare
    `GumbelConfig()` — the SELFPLAY shape (`c_scale` 0.1, `topk` 16, linear
    root, `fpu_reduction` 1.2) — so those paths silently measured at a search
    shape nobody tuned, while `PLAY_SEARCH_DEFAULTS` claimed to be referenced
    "from every such entry point" and named the training-gate match by name.
    Pass an explicit dict to sweep, or `{}` to force the dataclass defaults.

    Note `PLAY_SEARCH_DEFAULTS` deliberately carries no `simulations`, so the
    caller's sim budget is never overridden here.

    ⚑ Raises `ActionDecodeError` when an action id names no legal move: this is
    a measurement, and playing a substitute move on would score a corrupted
    match. Any caller running INSIDE a training process must therefore catch it
    and record the measurement VOID rather than let it propagate — an uncaught
    raise takes the host process down over one unscorable match.
    """
    g = int(games)
    if g <= 0:
        raise ValueError("games must be > 0")
    if a_plays_white is None:
        a_plays_white = [True] * g
    if len(a_plays_white) != g:
        raise ValueError("a_plays_white length must match games")

    if opening_cfg is None:
        opening_cfg = OpeningConfig(random_start_plies=2)
    boards = [make_starting_board(rng=rng, cfg=opening_cfg) for _ in range(g)]
    done = [False] * g
    sims_a = int(mcts_simulations if mcts_simulations_a is None else mcts_simulations_a)
    sims_b = int(mcts_simulations if mcts_simulations_b is None else mcts_simulations_b)
    if sims_a <= 0 or sims_b <= 0:
        raise ValueError("mcts simulations must be > 0")

    play_overrides = (
        dict(PLAY_SEARCH_DEFAULTS) if gumbel_overrides is None else dict(gumbel_overrides)
    )

    def _pick(model: torch.nn.Module, idxs: list[int], *, sims: int) -> list[int]:
        if not idxs:
            return []
        return pick_moves_for_boards(
            model, [boards[i] for i in idxs],
            device=device, rng=rng,
            mcts_type=mcts_type, mcts_simulations=sims,
            temperature=temperature, c_puct=c_puct,
            gumbel_add_noise=bool(gumbel_add_noise),
            gumbel_overrides=play_overrides,
        )

    for _ply in range(int(max_plies)):
        for i in range(g):
            if not done[i] and boards[i].is_game_over(claim_draw=True):
                done[i] = True
        active = [i for i in range(g) if not done[i]]
        if not active:
            break

        a_to_move, b_to_move = split_active_by_side_to_move(
            active, boards, a_plays_white,
        )
        # strict: every caller of `play_match_batch` is an arena/gate/eval path
        # (see the docstring above), so a decode failure corrupts a measurement
        # rather than costing a training row.
        apply_actions_to_boards(
            boards, a_to_move, _pick(model_a, a_to_move, sims=sims_a), strict=True,
        )
        apply_actions_to_boards(
            boards, b_to_move, _pick(model_b, b_to_move, sims=sims_b), strict=True,
        )

    a_win, a_draw, a_loss = _tally_match_results(boards, a_plays_white)

    return MatchStats(
        games=g,
        max_plies=int(max_plies),
        a_win=int(a_win),
        a_draw=int(a_draw),
        a_loss=int(a_loss),
        a_as_white=int(sum(1 for v in a_plays_white if bool(v))),
        a_as_black=int(sum(1 for v in a_plays_white if not bool(v))),
    )
