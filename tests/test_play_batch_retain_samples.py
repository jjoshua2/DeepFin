"""play_batch(retain_samples=False): finite mode stops holding every row of the call.

The default keeps every ReplaySample in the returned list for the whole call
(about 9 MB per 100-row game); the simple_rl collector discards that list and
consumes rows through on_game_complete, so a 9k-game process OOMed on it
(2026-11-26). With retain_samples=False the rows are delivered and dropped, the
returned list is empty, BatchStats.positions still counts them, and the games
themselves are unchanged.
"""
from __future__ import annotations

import gc
import weakref
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.manager import CompletedGameBatch
from chess_anti_engine.selfplay.opening import OpeningConfig
from tests.selfplay_helpers import FakeStockfish, UniformPolicyValueModel


def _run(*, retain: bool, seed: int = 11) -> tuple[list[Any], Any, list[CompletedGameBatch]]:
    completed: list[CompletedGameBatch] = []
    samples, stats = play_batch(
        UniformPolicyValueModel().eval(), device="cpu",
        rng=np.random.default_rng(seed),
        stockfish=FakeStockfish([0.0, 1.0, 0.0]),
        games=2, target_games=5,
        on_game_complete=completed.append,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=6, selfplay_fraction=1.0),
        retain_samples=retain,
    )
    return samples, stats, completed


def _rows(completed: list[CompletedGameBatch]) -> list[tuple[int | None, bytes]]:
    return [(s.ply_index, np.asarray(s.x).tobytes() + np.asarray(s.policy_target).tobytes())
            for b in completed for s in b.samples]


def test_retain_false_returns_empty_list_and_counts_positions() -> None:
    samples, stats, completed = _run(retain=False)
    delivered = sum(len(b.samples) for b in completed)
    assert stats.games == 5
    assert len(completed) == 5
    assert delivered > 0
    assert samples == []
    assert stats.positions == delivered


def test_retain_false_plays_the_same_games_as_the_default() -> None:
    kept, kept_stats, kept_done = _run(retain=True)
    _, drop_stats, drop_done = _run(retain=False)
    assert len(kept) == kept_stats.positions == drop_stats.positions
    # Same rows, byte for byte, in the same order: only the list's lifetime changed.
    assert _rows(drop_done) == _rows(kept_done)
    assert _rows(drop_done) == [(s.ply_index, np.asarray(s.x).tobytes()
                                 + np.asarray(s.policy_target).tobytes()) for s in kept]


def test_retain_false_frees_each_games_rows_before_the_next_game_finishes() -> None:
    # The memory bound itself: the callback keeps only weakrefs to the rows it was
    # handed, so once a game is delivered nothing but play_batch can hold them. At
    # every later game every earlier game's rows must already be dead -- a clear
    # deferred to the end of the call (or no clear) keeps them alive until return.
    # One exception, not an accumulation: a finished game whose slot is NOT reused
    # (the target is already dealt) keeps its rows' inputs alive through the slot's
    # own records until return -- at most one game per slot. Here that is game 4
    # (5 games on 2 slots: games 1-3 free their slot for a new game, game 4 does not).
    earlier: list[weakref.ref[Any]] = []
    alive_at: list[int] = []
    rows_per_game: list[int] = []
    games = 0

    def on_game(batch: CompletedGameBatch) -> None:
        nonlocal games
        gc.collect()
        alive_at.append(sum(1 for r in earlier if r() is not None))
        earlier.extend(weakref.ref(s.x) for s in batch.samples)
        rows_per_game.append(len(batch.samples))
        games += 1

    _, stats = play_batch(
        UniformPolicyValueModel().eval(), device="cpu",
        rng=np.random.default_rng(11),
        stockfish=FakeStockfish([0.0, 1.0, 0.0]),
        games=2, target_games=5,
        on_game_complete=on_game,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=6, selfplay_fraction=1.0),
        retain_samples=False,
    )
    assert games == 5
    assert len(earlier) == stats.positions > 0
    assert alive_at[:4] == [0, 0, 0, 0]
    assert alive_at[4] <= rows_per_game[3]


def test_retain_false_without_callback_is_refused() -> None:
    with pytest.raises(ValueError, match="retain_samples=False needs on_game_complete"):
        play_batch(
            UniformPolicyValueModel().eval(), device="cpu",
            rng=np.random.default_rng(0), stockfish=FakeStockfish([0.0, 1.0, 0.0]),
            games=1, target_games=1,
            search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
            opening=OpeningConfig(random_start_plies=0),
            game=GameConfig(max_plies=2, selfplay_fraction=1.0),
            retain_samples=False,
        )
