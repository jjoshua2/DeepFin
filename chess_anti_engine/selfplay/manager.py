from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, wait
from typing import Any

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.inference import BatchEvaluator
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.finalize import finalize_game
from chess_anti_engine.selfplay.network_turn import run_network_turn
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.state import (
    BatchStats,
    CompletedGameBatch,
    SelfplayState,
)
from chess_anti_engine.selfplay.stockfish_turn import (
    finish_sf_annotation_and_moves,
    finish_pending_curriculum_moves,
    has_pending_sf_labels_for_records,
    poll_async_sf_labels,
    submit_async_curriculum_move_queries,
    submit_async_sf_labels_from_curriculum_moves,
    submit_async_sf_label_queries,
)
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishUCI
from chess_anti_engine.tablebase import tb_adjudicate_result


def _tb_adjudicate_active_games(state: SelfplayState) -> int:
    """Scan active games; if any current position is TB-eligible and
    probable AND this game's per-game roll said "adjudicate", mark the
    game done and stash the TB-proven result. Runs once per step;
    per-active-game cost is ~10µs after the popcount prefilter rejects
    non-endgame positions.

    Games whose roll landed on "play through" (controlled by
    ``syzygy_adjudicate_fraction``) are skipped here and finish naturally,
    so the NN continues training on endgame positions with its own labels
    rather than losing endgame skill entirely.
    """
    assert state.tb_probe is not None
    assert state.game.syzygy_path is not None
    max_p = state.tb_probe.max_pieces
    adjudicated = 0
    for i in range(state.batch_size):
        if state.done_arr[i] or state.finalized_arr[i] or state.tb_result_arr[i] is not None:
            continue
        if not state.tb_adj_roll_arr[i]:
            continue
        if i in state.pending_sf_moves:
            continue
        cb = state.cboards[i]
        occ = int(cb.occ_white) | int(cb.occ_black)
        if occ.bit_count() > max_p or int(cb.castling) != 0:
            continue
        # FEN-seeded games must get at least one NET decision before TB
        # adjudication, else a TB-eligible endgame seed is finalized with zero
        # samples. The helper covers both seatings — not just ply 0 — so a
        # curriculum seed where the opponent replies first (net_side_to_move
        # False) is also protected until the net moves (Codex review, PR #108).
        if state._fenlist_awaiting_net_move(i):
            continue
        board = chess.Board(cb.fen())
        result = tb_adjudicate_result(board, state.game.syzygy_path)
        if result is not None:
            state.tb_result_arr[i] = result
            state.done_arr[i] = 1
            adjudicated += 1
    return adjudicated


def _run_step(
    state: SelfplayState,
    *,
    net_idxs: list[int],
    sp_opp_idxs: list[int],
    cur_opp_idxs: list[int],
    stockfish: StockfishUCI | StockfishPool,
    on_timing: Callable[[str, float], None] | None = None,
) -> tuple[float, float]:
    """Network + SF turn for one step. Returns ``(net_seconds, sf_seconds)``.

    Pipelines SF I/O with GPU work: curriculum SF queries dispatch first
    (disjoint boards), then the combined net+sp_opp MCTS turn runs while
    those queries fly, then selfplay-opp SF dispatches before collecting
    curriculum responses, etc.
    """
    t_net = 0.0
    t_sf = 0.0
    is_pool = isinstance(stockfish, StockfishPool)

    # Curriculum SF first — boards are disjoint, so we overlap SF I/O with
    # the combined network turn below.
    cur_futures: dict[int, Any] | None = None
    if cur_opp_idxs and is_pool:
        t0 = time.perf_counter()
        submit_async_curriculum_move_queries(state, cur_opp_idxs)
        dt = time.perf_counter() - t0
        t_sf += dt
        if on_timing is not None:
            on_timing("sf_submit_curriculum", dt)
        t0 = time.perf_counter()
        submit_async_sf_labels_from_curriculum_moves(state, cur_opp_idxs)
        dt = time.perf_counter() - t0
        t_sf += dt
        if on_timing is not None:
            on_timing("sf_submit_label", dt)

    # Combined network turn: merge net_idxs + sp_opp_idxs into one MCTS call
    # for GPU batch-size doubling. Disjoint + independent.
    combined_net_idxs = net_idxs + sp_opp_idxs
    if combined_net_idxs:
        t0 = time.perf_counter()
        run_network_turn(state, combined_net_idxs)
        dt = time.perf_counter() - t0
        t_net += dt
        if on_timing is not None:
            on_timing("network", dt)

    # Selfplay games still need SF labels on every network turn. Curriculum
    # games get their label from the next SF-opponent query, but selfplay slots
    # would otherwise advance another network turn before annotation and lose
    # the previous record as "latest".
    sp_label_idxs = [
        i for i in combined_net_idxs
        if bool(state.selfplay_arr[i]) and not state.done_arr[i]
    ]
    if sp_label_idxs and is_pool:
        t0 = time.perf_counter()
        submit_async_sf_label_queries(state, sp_label_idxs)
        dt = time.perf_counter() - t0
        t_sf += dt
        if on_timing is not None:
            on_timing("sf_submit_label", dt)

    if cur_opp_idxs:
        t0 = time.perf_counter()
        if is_pool:
            finish_pending_curriculum_moves(state, block=False)
        else:
            finish_sf_annotation_and_moves(
                state, cur_opp_idxs, play_curriculum_moves=True,
                attach_labels=True,
                for_move=False,
                futures=cur_futures,
            )
        dt = time.perf_counter() - t0
        t_sf += dt
        if on_timing is not None:
            on_timing("sf_finish_curriculum", dt)

    if sp_label_idxs and not is_pool:
        t0 = time.perf_counter()
        finish_sf_annotation_and_moves(
            state, sp_label_idxs,
            play_curriculum_moves=False, futures=None,
        )
        dt = time.perf_counter() - t0
        t_sf += dt
        if on_timing is not None:
            on_timing("sf_finish_label", dt)

    return t_net, t_sf


def _finalize_completed_slots(
    state: SelfplayState,
    *,
    all_samples: list[ReplaySample],
    on_game_complete: Callable[[CompletedGameBatch], None] | None,
    batch_size: int,
    continuous: bool,
    target: int,
    defer_pending_labels: bool = False,
) -> None:
    """Finalize done-but-not-yet-finalized games; recycle slots while target allows."""
    for i in range(batch_size):
        if state.done_arr[i] and not state.finalized_arr[i]:
            if (
                defer_pending_labels
                and has_pending_sf_labels_for_records(state, state.samples_per_game[i])
            ):
                continue
            finalize_game(state, i, all_samples, on_game_complete)
            state.finalized_arr[i] = 1
            state.games_completed += 1
            if continuous or state.games_started < target:
                state.recycle_slot(i)


def _wait_for_starved_sf(
    state: SelfplayState, *, control_poll: bool,
) -> None:
    """Wait on SF with a bounded control or background interval."""
    timeout_s = 0.05 if control_poll else 0.25
    if state.pending_sf_moves:
        finish_pending_curriculum_moves(
            state, block=True, block_timeout_s=timeout_s,
        )
    elif state.pending_sf_labels:
        wait(
            tuple(pending.future for pending in state.pending_sf_labels),
            timeout=timeout_s,
            return_when=FIRST_COMPLETED,
        )


def play_batch(
    model: torch.nn.Module | None,
    *,
    device: str,
    rng: np.random.Generator,
    stockfish: StockfishUCI | StockfishPool,
    evaluator: BatchEvaluator | None = None,
    games: int,
    target_games: int = 0,
    on_game_complete: Callable[[CompletedGameBatch], None] | None = None,
    on_timing: Callable[[str, float], None] | None = None,
    on_step: Callable[[], None] | None = None,
    on_state_ready: Callable[[SelfplayState], None] | None = None,
    stop_fn: Callable[[], bool] | None = None,
    pause_fn: Callable[[], bool] | None = None,
    slot_oversubscribe: float = 1.0,
    # Config groups (frozen dataclasses with sensible defaults).
    opponent: OpponentConfig = OpponentConfig(),
    temp: TemperatureConfig = TemperatureConfig(),
    search: SearchConfig = SearchConfig(),
    opening: OpeningConfig = OpeningConfig(),
    diff_focus: DiffFocusConfig = DiffFocusConfig(),
    game: GameConfig = GameConfig(),
    fen_dole_queue: list[str] | None = None,
    fen_sf_refute_queue: list[str] | None = None,
) -> tuple[list[ReplaySample], BatchStats]:
    """Play a batch of games.

    Design goals:
    - keep GPU busy via batched inference
    - keep SF queries minimal (one per SF ply)
    - compute volatility targets from a consistent network-side WDL series without per-ply overhead

    Continuous mode (stop_fn provided, target_games=0): runs forever with all
    finished slots recycled immediately, until stop_fn() returns True.  Samples
    are delivered incrementally via on_game_complete; the returned sample list
    is empty to avoid unbounded memory growth.

    Finite mode (target_games > 0 or stop_fn is None): plays exactly
    target_games (or `games` if target_games=0) then returns all samples.

    ``fen_dole_queue`` (opt-in, opening_fen_dole_per_iter>0): a mutable FIFO of
    seed lines the server doled for this iteration. Selfplay slots drain it (via
    the SelfplayState, deterministic even coverage). Mid-session refill must
    mutate the SHARED list in place; never rebind ``state.fen_dole_queue``
    (that orphans the worker's live queue — PR #154). None = no doled seeds.

    ``fen_sf_refute_queue`` (opt-in SF-refute channel): curriculum slots drain
    this FIFO for short SF-as-opponent openings (``opening_fen_sf_refute_*``),
    then hand off to selfplay after N SF plies. Same in-place-refill rule as
    the selfplay dole queue.

    ``pause_fn`` (continuous mode): while it returns True the loop HOLDS —
    no new work is scheduled but every in-flight game keeps its state — and
    resumes where it left off once it returns False. This is how the
    train-phase selfplay pause preserves partial games instead of abandoning
    them (pre-2026-07 behavior was a full session teardown per iteration,
    discarding ~all in-flight slots and censoring any game longer than one
    selfplay window out of the replay buffer). ``on_step`` keeps being called
    during the hold so the caller's manifest poll can clear the pause and
    hot-swap the freshly trained model before play resumes.

    ``slot_oversubscribe`` multiplies concurrent slots only in continuous
    mode. Finite calls still play exactly their requested target. The default
    1.0 preserves current scheduling; values above one provide independent
    network work while curriculum slots wait on asynchronous Stockfish moves.
    """
    requested_batch = int(games)
    continuous = stop_fn is not None and int(target_games) <= 0
    target = int(target_games) if int(target_games) > 0 else requested_batch
    batch_size = min(requested_batch, target)
    slot_factor = float(slot_oversubscribe)
    if not math.isfinite(slot_factor) or slot_factor < 1.0:
        raise ValueError(
            f"slot_oversubscribe must be finite and >= 1.0, got {slot_oversubscribe!r}"
        )
    if continuous:
        batch_size = max(batch_size, math.ceil(requested_batch * slot_factor))
    if batch_size <= 0:
        raise ValueError("play_batch requires at least one game")

    # Apply the gated repetition-plane fix (process-global in the C encoders)
    # to match the configured value before any encoding happens this batch.
    rep_fix.apply(bool(game.history_rep_fix))

    state = SelfplayState.create(
        model=model,
        device=device,
        rng=rng,
        stockfish=stockfish,
        evaluator=evaluator,
        batch_size=batch_size,
        continuous=continuous,
        target=target,
        opponent=opponent,
        temp=temp,
        search=search,
        opening=opening,
        diff_focus=diff_focus,
        game=game,
        fen_dole_queue=fen_dole_queue,
        fen_sf_refute_queue=fen_sf_refute_queue,
    )

    # Hand the live state to the caller so it can apply live-safe reco changes
    # (selfplay_fraction, opponent regret, SF node budgets) by swapping the
    # config dataclasses in place — every consumer reads them fresh per
    # step/move/recycle, so no session restart is needed.
    if on_state_ready is not None:
        on_state_ready(state)

    all_samples: list[ReplaySample] = []

    # Continuous mode: stop_fn controls exit.  Finite mode: safety bound
    # lets us abort if something goes sideways and no game ever finalizes.
    max_steps = (
        2**62 if continuous else int(target) * (int(game.max_plies) // 2 + 2)
    )
    t_net = 0.0
    t_sf = 0.0

    for _step in range(max_steps):  # skylos: ignore (loop var unused by convention)
        if on_step is not None:
            t0 = time.perf_counter()
            on_step()
            if on_timing is not None:
                on_timing("check_model", time.perf_counter() - t0)
        if stop_fn is not None and stop_fn():
            # Continuous scheduling may have deferred a completed game while
            # its label was still running. Preserve those finished games at a
            # true teardown; only genuinely in-flight boards remain abandoned.
            t0 = time.perf_counter()
            if state.pending_sf_labels:
                poll_async_sf_labels(state)
            _finalize_completed_slots(
                state,
                all_samples=all_samples,
                on_game_complete=on_game_complete,
                batch_size=batch_size,
                continuous=continuous,
                target=target,
                defer_pending_labels=False,
            )
            if on_timing is not None:
                on_timing("finalize", time.perf_counter() - t0)
            break

        if pause_fn is not None and pause_fn():
            t0 = time.perf_counter()
            while pause_fn() and not (stop_fn is not None and stop_fn()):
                if on_step is not None:
                    on_step()
                time.sleep(0.25)
            if on_timing is not None:
                on_timing("pause_hold", time.perf_counter() - t0)
            continue  # re-check stop_fn before scheduling any work

        if state.pending_sf_labels:
            t0 = time.perf_counter()
            poll_async_sf_labels(state)
            if on_timing is not None:
                on_timing("sf_label_poll", time.perf_counter() - t0)

        if state.pending_sf_moves:
            t0 = time.perf_counter()
            finish_pending_curriculum_moves(state, block=False)
            if on_timing is not None:
                on_timing("sf_finish_curriculum", time.perf_counter() - t0)

        # Tablebase adjudication before classify_active_slots. Any TB-eligible
        # game gets marked done; classify then skips it and finalize uses the
        # stashed TB result. Runs at most once per game (state.tb_result_arr
        # is the idempotency key).
        # state.game (not closed-over session-start ``game``) — matches recycle.
        if state.tb_probe is not None and state.game.syzygy_adjudicate:
            t0 = time.perf_counter()
            _tb_adjudicate_active_games(state)
            if on_timing is not None:
                on_timing("tb_adjudicate", time.perf_counter() - t0)

        t0 = time.perf_counter()
        net_idxs, sp_opp_idxs, cur_opp_idxs, all_done = state.classify_active_slots()
        classified_count = len(net_idxs) + len(sp_opp_idxs) + len(cur_opp_idxs)
        if state.pending_sf_moves:
            # Pending moves originate in cur_opp_idxs, and their boards do not
            # advance into another partition until the future is removed.
            cur_opp_idxs = [idx for idx in cur_opp_idxs if idx not in state.pending_sf_moves]
        if on_timing is not None:
            pending_excluded = (
                classified_count - len(net_idxs) - len(sp_opp_idxs) - len(cur_opp_idxs)
            )
            runnable_count = len(net_idxs) + len(sp_opp_idxs) + len(cur_opp_idxs)
            on_timing("classify", time.perf_counter() - t0)
            on_timing("slot_observations", 1.0)
            on_timing("runnable_net_sum", float(len(net_idxs)))
            on_timing("runnable_sp_opp_sum", float(len(sp_opp_idxs)))
            on_timing("runnable_cur_opp_sum", float(len(cur_opp_idxs)))
            on_timing(
                "sf_pending_excluded_sum",
                float(pending_excluded),
            )
            if pending_excluded > 0 and runnable_count > 0:
                on_timing("sf_pending_with_runnable_steps", 1.0)
            on_timing(
                "games_in_flight_sum",
                float(max(0, state.games_started - state.games_completed)),
            )
        if all_done:
            break
        if (
            not net_idxs
            and not sp_opp_idxs
            and not cur_opp_idxs
            and (state.pending_sf_moves or state.pending_sf_labels)
        ):
            t0 = time.perf_counter()
            _wait_for_starved_sf(state, control_poll=on_step is not None)
            if on_timing is not None:
                on_timing("sf_block_starved", time.perf_counter() - t0)
            t0 = time.perf_counter()
            if state.pending_sf_labels:
                poll_async_sf_labels(state)
            _finalize_completed_slots(
                state, all_samples=all_samples,
                on_game_complete=on_game_complete,
                batch_size=batch_size, continuous=continuous, target=target,
                defer_pending_labels=continuous,
            )
            if on_timing is not None:
                on_timing("finalize", time.perf_counter() - t0)
            continue

        net_dt, sf_dt = _run_step(
            state, net_idxs=net_idxs, sp_opp_idxs=sp_opp_idxs,
            cur_opp_idxs=cur_opp_idxs, stockfish=stockfish,
            on_timing=on_timing,
        )
        t_net += net_dt
        t_sf += sf_dt

        t0 = time.perf_counter()
        if state.pending_sf_labels:
            poll_async_sf_labels(state)
        _finalize_completed_slots(
            state, all_samples=all_samples,
            on_game_complete=on_game_complete,
            batch_size=batch_size, continuous=continuous, target=target,
            defer_pending_labels=continuous,
        )
        if on_timing is not None:
            on_timing("finalize", time.perf_counter() - t0)

        # Reset tree when it grows unbounded. Live root_ids can't be preserved
        # across reset, but the next ply's MCTS root-init rebuilds them with no
        # algorithmic loss — only the few-ply tree-reuse window vanishes.
        # Without this, continuous selfplay leaks tree nodes for the lifetime
        # of the worker (the prior `all(rid < 0)` guard never fired with
        # >1 slot since at least one root_id stays >= 0 at all times).
        if (
            state.mcts_tree is not None
            and state.mcts_tree.node_count() > 500_000
        ):
            t0 = time.perf_counter()
            reset_compact = getattr(state.mcts_tree, "reset_compact", None)
            if reset_compact is not None:
                reset_compact()
            else:  # pragma: no cover - compatibility with old extension builds
                state.mcts_tree.reset()
            for _i in range(len(state.root_ids)):
                state.root_ids[_i] = -1
            if on_timing is not None:
                on_timing("tree_reset", time.perf_counter() - t0)

    logging.getLogger("chess_anti_engine.worker").info(
        "play_batch timing: net=%.1fs sf=%.1fs (net %.0f%%, sf %.0f%%)",
        t_net, t_sf,
        t_net / max(0.001, t_net + t_sf) * 100,
        t_sf / max(0.001, t_net + t_sf) * 100,
    )
    if continuous:
        # Quantifies the session-teardown waste: in-flight games at exit are
        # unfinalized and their compute is discarded. With the pause hold this
        # should only be paid at REAL teardowns (restart-keyed reco change,
        # task/trial reassignment, shutdown), not every iteration.
        _in_flight = max(0, int(state.games_started) - int(state.games_completed))
        logging.getLogger("chess_anti_engine.worker").info(
            "play_batch exit: completed=%d started=%d in_flight_abandoned=%d",
            int(state.games_completed), int(state.games_started), _in_flight,
        )
    sf_nodes = int(getattr(stockfish, "nodes", 0) or 0)
    return all_samples, state.stats.to_batch_stats(
        games=state.games_completed,
        positions=len(all_samples),
        sf_nodes=sf_nodes if sf_nodes > 0 else None,
    )
