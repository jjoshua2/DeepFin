"""In-flight selfplay resume: byte-exact re-encode, and a gate that can fail.

The load-bearing claim is that a suspended game can be rebuilt from its move
list alone. That is only true because the replay reconstructs a REAL move
stack — the history planes are silently wrong for a board built from a bare
FEN or a stack-less copy, which has produced two retracted findings in this
repo. So the first test does not check that resume "works": it captures every
in-flight ply's encoded inputs from the live C play path, resumes, and demands
``np.array_equal`` on each one.

The second half is the negative control. A resume path that only ever succeeds
proves nothing, so the persisted state is corrupted three ways (a dropped move,
a truncated record column, shredded bytes) and each must be REJECTED with a
counted reason rather than silently producing a shorter or misaligned game.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, cast

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding.lc0 import LC0_HISTORY_LEGACY
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.manager import CompletedGameBatch
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.resume import (
    RESUME_FILE_SUFFIX,
    RESUME_FORMAT_VERSION,
    count_unclaimed_resume_files,
    initial_resume_counts,
    resume_inflight_games,
    should_resume_game,
    suspend_inflight_games,
)
from chess_anti_engine.selfplay.state import SelfplayState
from chess_anti_engine.stockfish import StockfishUCI
from chess_anti_engine.utils.config_yaml import (
    flatten_run_config_defaults,
    load_yaml_file,
)
from tests.stockfish_binary import find_stockfish

SF_PATH = find_stockfish()
FINGERPRINT = "test-fingerprint"
TRIAL_ID = "trial-a"

_PROD = flatten_run_config_defaults(load_yaml_file("configs/pbt2_small.yaml"))


def _prod_game_kwargs() -> dict[str, Any]:
    """Encoding-relevant GameConfig fields, taken from the PRODUCTION yaml.

    Plane layout differs by input_history_encoding, so a byte-exactness claim
    made under a different encoding would not be a claim about production.
    Reading the live config (rather than hardcoding) means the test follows a
    production switch instead of quietly testing a dead encoding.
    """
    assert _PROD["input_history_encoding"] == "lc0_root_legacy_meta"
    assert _PROD["input_extra_features"] == "v2_threats"
    return {
        "input_history_encoding": str(_PROD["input_history_encoding"]),
        "input_extra_features": str(_PROD["input_extra_features"]),
        "record_lc0_root_input": bool(_PROD["record_lc0_root_input"]),
        "history_rep_fix": bool(_PROD["history_rep_fix"]),
        "record_relations": bool(_PROD.get("record_relations") or False),
    }


def _game_config(**overrides: Any) -> GameConfig:
    kwargs: dict[str, Any] = {
        "max_plies": 200,
        "selfplay_fraction": 0.0,  # curriculum: SF labels on every recorded ply
        "sf_policy_temp": 0.012,
        "sf_policy_label_smooth": 0.01,
        "sf_wdl_use_cp_logistic": True,
        "sf_wdl_cp_slope": 0.010,
        "sf_wdl_cp_draw_width": 60.0,
        **_prod_game_kwargs(),
    }
    kwargs.update(overrides)
    return GameConfig(**kwargs)


_SEARCH = SearchConfig(
    simulations=4, fast_simulations=2, mcts_type="gumbel", playout_cap_fraction=1.0,
)
_OPENING = OpeningConfig(random_start_plies=0)
_DIFF_FOCUS = DiffFocusConfig(min_keep=1.0)

# The stand-in for `WorkerSession._build_selfplay_configs`'s return value, used
# by `_wired_session` to drive `_run_selfplay` without the manifest-resolution
# machinery. It must carry the SAME KEYS as the real one: `_run_selfplay` and
# its callees index this dict (`cfgs["opening"]`, `cfgs["search"]`,
# `cfgs["game"]`) and hand it to `play_batch` as `**cfgs`, so a key the double
# omits is a code path this file silently stops exercising —
# `_await_broker_ready` was already degrading to its "probe construction
# failed" branch on the missing `game` key, and the `cfgs["search"]` read added
# for the selfplay policy-temp log line turned the same drift into a KeyError.
# `test_the_selfplay_configs_double_matches_the_real_contract` pins the match.
_SELFPLAY_CFGS_DOUBLE: dict[str, Any] = {
    "slot_oversubscribe": 1.0,
    "opponent": OpponentConfig(),
    "temp": TemperatureConfig(),
    "search": _SEARCH,
    "opening": _OPENING,
    "diff_focus": _DIFF_FOCUS,
    "game": _game_config(),
}

# Production plays `opening_book_prob: 1.0` / `opening_book_max_plies: 4`, so a
# real game NEVER starts from a bare startpos with an empty move stack — it
# starts from a board that already carries moves, and the LC0 history planes at
# the early plies are filled from them. A byte-exactness proof run only at
# random_start_plies=0 therefore proves it for the one construction production
# does not use, and "a board without a real move stack" is precisely the shape
# that has produced two retracted findings in this repo. random_start_plies is
# the cheap way to get the same property (_random_playout_from_start pushes real
# moves, so the board has a real _stack); 4 is production's book depth.
#
# The "fen" arm covers the third production shape: a blind-spot fenlist seed,
# whose ROOT is a mid-game FEN with a nonzero halfmove clock and fullmove
# number. Both survive suspension only through `root_fen` — the zobrist
# excludes the halfmove clock, so no pos_hash can catch a wrong clock, yet
# rule50 is an input plane. The seed's history moves also put a repetition
# inside the 8-position history window, so the repetition plane is exercised.
_START_MODE_IDS = {
    0: "startpos-empty-stack", 4: "prior-moves-depth-4", "fen": "midgame-fen-root",
}

_MIDGAME_ROOT_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
# Knights out and back: the root position recurs after move 4 (a twofold, so
# the seed is not immediately claim-drawable), and no pawn move or capture
# means the halfmove clock keeps counting — 2 at the root, 6 at the live
# game's first ply.
_MIDGAME_SEED_MOVES = "b1c3 g8f6 c3b1 f6g8"


def _opening_cfg(random_start_plies: int) -> OpeningConfig:
    return OpeningConfig(random_start_plies=int(random_start_plies))


def _tiny_model() -> Any:
    from chess_anti_engine.model import ModelConfig, build_model

    torch.manual_seed(0)
    return build_model(
        ModelConfig(embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False),
    ).eval()


def _play_session(
    *,
    sf: StockfishUCI,
    game: GameConfig,
    games: int,
    max_steps: int,
    on_state_ready: Any = None,
    on_suspend: Any = None,
    on_game_complete: Any = None,
    seed: int = 0,
    opening: OpeningConfig | None = None,
) -> SelfplayState:
    """Run a continuous session for ``max_steps`` loop steps; return its state."""
    captured: list[SelfplayState] = []

    def _ready(state: SelfplayState) -> None:
        captured.append(state)
        if on_state_ready is not None:
            on_state_ready(state)

    steps = {"n": 0}

    def _stop() -> bool:
        steps["n"] += 1
        return steps["n"] > max_steps

    play_batch(
        _tiny_model(), device="cpu", rng=np.random.default_rng(seed),
        stockfish=sf,
        games=games, target_games=0,
        stop_fn=_stop,
        on_state_ready=_ready,
        on_suspend=on_suspend,
        on_game_complete=on_game_complete,
        opponent=OpponentConfig(),
        temp=TemperatureConfig(temperature=1.0),
        search=_SEARCH,
        opening=(_OPENING if opening is None else opening),
        diff_focus=_DIFF_FOCUS,
        game=game,
    )
    assert captured
    return captured[0]


def _inflight_records(state: SelfplayState) -> dict[tuple[int, ...], list[Any]]:
    """In-flight games keyed by their move list (stable across suspend/resume)."""
    out: dict[tuple[int, ...], list[Any]] = {}
    for i in range(state.batch_size):
        if state.finalized_arr[i] or state.done_arr[i]:
            continue
        if not state.samples_per_game[i]:
            continue
        out[tuple(int(m) for m in state.move_idx_history[i])] = list(
            state.samples_per_game[i]
        )
    return out


def _fresh_state(
    game: GameConfig, *, batch_size: int, seed: int = 5,
    opening: OpeningConfig | None = None,
) -> SelfplayState:
    """A state shaped like the one play_batch builds, without playing.

    ``rep_fix.apply`` is process-global and normally called by play_batch;
    resume re-encodes through the same globals, so a direct state must set it
    too or the repetition planes would depend on test ordering.
    """
    rep_fix.apply(bool(game.history_rep_fix), boards_discarded=True)
    return SelfplayState.create(
        model=None, device="cpu", rng=np.random.default_rng(seed),
        stockfish=cast("Any", _NullStockfish()),
        evaluator=cast("Any", _NullEvaluator()),
        batch_size=batch_size, continuous=True, target=batch_size,
        opponent=OpponentConfig(), temp=TemperatureConfig(), search=_SEARCH,
        opening=(_OPENING if opening is None else opening),
        diff_focus=_DIFF_FOCUS, game=game,
    )


class _NullEvaluator:
    """Evaluator stand-in: resume never runs the net, only the encoders."""

    def evaluate_encoded(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = int(xs.shape[0])
        return np.zeros((n, 1858), np.float32), np.zeros((n, 3), np.float32)


class _NullStockfish:
    nodes = 1


def _bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact-representation equality, including -0.0 vs +0.0 and NaN payloads."""
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.tobytes() == np.ascontiguousarray(b).tobytes()
    )


def _assert_record_identical(before: Any, after: Any, where: str) -> None:
    assert _bytes_equal(np.ascontiguousarray(before.x), after.x), f"{where}: x"
    for name in ("x_lc0_root", "relations", "legal_mask", "policy_probs",
                 "net_wdl_est", "search_wdl_est", "sf_policy_target", "sf_wdl",
                 "sf_wdl_original", "sf_multipv_raw", "sf_label_meta",
                 "sf_legal_mask"):
        want = getattr(before, name)
        got = getattr(after, name)
        if want is None:
            assert got is None, f"{where}: {name} appeared from nowhere"
            continue
        assert got is not None, f"{where}: {name} was lost"
        assert _bytes_equal(np.ascontiguousarray(want), got), f"{where}: {name}"
    for name in ("pov_color", "ply_index", "move_offset", "pos_hash", "has_policy",
                 "priority", "priority_policy_kl", "priority_q_delta",
                 "sample_weight", "keep_prob", "sf_move_index",
                 "sf_played_move_index", "sf_played_rank", "sf_played_regret",
                 "prior_top1_index", "prior_top1_prob",
                 "gumbel_policy_diag", "is_sf_refute_opp"):
        assert getattr(before, name) == getattr(after, name), f"{where}: {name}"


def _suspend_all(state: SelfplayState, out_dir: Path) -> int:
    report = suspend_inflight_games(
        state, out_dir=out_dir, compat_fingerprint=FINGERPRINT,
        model_sha="a" * 64, model_step=7, trial_id=TRIAL_ID,
    )
    return report.persisted


def _state_files(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob(f"*{RESUME_FILE_SUFFIX}"))


def _load_arrays(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as npz:
        return {k: np.array(npz[k]) for k in npz.files}


def _rewrite(path: Path, mutate: Any) -> None:
    """Load an npz state file, let ``mutate`` edit the dict, write it back."""
    arrays = _load_arrays(path)
    mutate(arrays)
    with path.open("wb") as fh:
        np.savez(fh, **arrays)


def _meta_of(arrays: dict[str, Any]) -> dict[str, Any]:
    return json.loads(bytes(arrays["meta_json"]).decode("utf-8"))


def _set_meta(arrays: dict[str, Any], meta: dict[str, Any]) -> None:
    arrays["meta_json"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8,
    )


# ── the byte-exactness proof ────────────────────────────────────────────────


@pytest.mark.skipif(SF_PATH is None, reason="Stockfish not found")
@pytest.mark.parametrize(
    "encoding_overrides",
    [
        pytest.param({}, id="production-encoding"),
        # The alternate-input columns are None under the production encoding
        # (lc0-root history already IS the root input), so a second arm is the
        # only way x_lc0_root and relations get exercised at all.
        pytest.param(
            {
                "input_history_encoding": LC0_HISTORY_LEGACY,
                "record_lc0_root_input": True,
                "record_relations": True,
            },
            id="legacy-encoding-with-alt-inputs",
        ),
    ],
)
@pytest.mark.parametrize(
    "start_mode",
    [pytest.param(n, id=_START_MODE_IDS[n]) for n in (0, 4, "fen")],
)
def test_resume_reencodes_every_inflight_ply_byte_exactly(
    tmp_path: Path, encoding_overrides: dict[str, Any], start_mode: int | str,
) -> None:
    """Replay + re-encode must reproduce the live C path's inputs bit for bit.

    Run at an empty starting move stack, at one carrying prior moves
    (production's `opening_book_prob: 1.0` / `opening_book_max_plies: 4` means
    every real book game is that shape, and the history planes at the early
    plies are the whole hazard the replay exists to get right), and from a
    mid-game FEN root (a blind-spot fenlist seed: nonzero halfmove clock and
    fullmove number, which only `root_fen` carries — the zobrist excludes the
    clock, so pos_hash passes on a wrong clock while the rule50 PLANE differs).
    """
    assert SF_PATH is not None
    game = _game_config(**encoding_overrides)
    if start_mode == "fen":
        seeds = tmp_path / "seeds.txt"
        seeds.write_text(
            f"{_MIDGAME_ROOT_FEN} | {_MIDGAME_SEED_MOVES}\n", encoding="utf-8",
        )
        opening = OpeningConfig(
            opening_fen_list_path=str(seeds), opening_fen_prob=1.0,
            opening_book_prob=0.0,
        )
        # A fenlist TARGET slot is never overwritten by resume (the fresh slot
        # already consumed a seed), so the post-restart session uses a
        # non-seeded opening — which is also production's shape: the SUSPENDED
        # game keeps its own opening; only fresh slots use the new config.
        target_opening = _OPENING
    else:
        opening = _opening_cfg(int(start_mode))
        target_opening = opening
    out_dir = tmp_path / "resume"
    sf = StockfishUCI(SF_PATH, nodes=80, multipv=4)
    try:
        state = _play_session(
            sf=sf, game=game, games=3, max_steps=10, opening=opening,
        )
        if start_mode == 4:
            # The arm is only meaningful if the games really do start from a
            # board with prior moves — a silently-ignored opening config would
            # otherwise make this arm a duplicate of the startpos one.
            assert state.starting_boards is not None
            assert all(
                len(state.starting_boards[i].move_stack) == start_mode
                for i in range(state.batch_size)
            )
        if start_mode == "fen":
            # ...and the fen arm only if the roots really carry the seed's
            # clock/fullmove and the opening really contains a repetition.
            assert state.starting_boards is not None
            for i in range(state.batch_size):
                start = state.starting_boards[i]
                root = start.root()
                assert root.fen() == _MIDGAME_ROOT_FEN
                assert root.halfmove_clock >= 1
                assert root.fullmove_number > 1
                assert start.is_repetition(2), (
                    "the seed history must put a repetition in the window"
                )
        # The rule50 clock of every live in-flight board, keyed by move list;
        # the restored boards must carry the same clock (no hash covers it).
        live_clock = {
            tuple(int(m) for m in state.move_idx_history[i]):
                int(state.cboards[i].halfmove_clock)
            for i in range(state.batch_size)
            if not state.finalized_arr[i] and not state.done_arr[i]
            and state.samples_per_game[i]
        }
        # The claim under test is about the PRODUCTION C path
        # (batch_process_ply). Without the extension the records would come
        # from the Python fallback, which encodes through the same helper the
        # resume uses — the comparison would degrade toward self-comparison
        # while still passing.
        assert state.has_c_ply, "byte-exactness must be proven against the C path"
        before = _inflight_records(state)
        assert before, "no in-flight games to resume"
        assert sum(len(recs) for recs in before.values()) >= 3
        assert any(
            rec.sf_policy_target is not None
            for recs in before.values() for rec in recs
        ), "no SF labels captured — the expensive fields would go untested"
        # ⚑ NON-VACUITY. `_assert_record_identical` compares prior_top1_* by
        # equality, and `None == None` passes — so the two fields are in that
        # list for nothing unless the session actually captured them. Assert on
        # the SOURCE records before the round-trip, not on what came back.
        assert any(
            rec.prior_top1_index is not None
            for recs in before.values() for rec in recs
        ), "no prior_top1 captured — the round-trip assertion would be vacuous"

        assert _suspend_all(state, out_dir) == len(before)

        target = _fresh_state(
            game, batch_size=state.batch_size, opening=target_opening,
        )
        report = resume_inflight_games(
            target, in_dir=out_dir, compat_fingerprint=FINGERPRINT,
            trial_id=TRIAL_ID,
        )
    finally:
        sf.close()

    assert report.discarded == 0
    assert report.resumed == len(before)
    after = _inflight_records(target)
    assert set(after) == set(before)
    for moves, want in before.items():
        got = after[moves]
        assert len(got) == len(want)
        for k, (rec_before, rec_after) in enumerate(zip(want, got, strict=True)):
            _assert_record_identical(rec_before, rec_after, f"game {moves[:4]} ply {k}")
    # The restored LIVE boards carry the suspended games' rule50 clocks — the
    # one field of the tail position no hash covers, and an input plane for
    # every ply still to be played.
    for i in range(target.batch_size):
        if not target.resumed_from_disk[i]:
            continue
        key = tuple(int(m) for m in target.move_idx_history[i])
        assert int(target.cboards[i].halfmove_clock) == live_clock[key]
        if start_mode == "fen":
            assert target.starting_boards is not None
            assert target.starting_boards[i].root().fen() == _MIDGAME_ROOT_FEN
    # ...and the planes are not trivially equal because they are all zero.
    any_x = next(iter(after.values()))[0].x
    assert np.count_nonzero(any_x) > 0
    assert any_x.shape[0] == (175 if game.input_extra_features == "v2_threats" else 146)


@pytest.mark.skipif(SF_PATH is None, reason="Stockfish not found")
def test_resume_restores_slot_state_and_counts_completed_games(
    tmp_path: Path,
) -> None:
    """End-to-end through the play_batch hooks: suspended games come back, keep
    playing, and land in outcome_stats as `resumed_inflight_games` when they
    finish — the number that proves resumption reached a shard."""
    assert SF_PATH is not None
    out_dir = tmp_path / "resume"
    sf = StockfishUCI(SF_PATH, nodes=80, multipv=4)
    completed: list[CompletedGameBatch] = []
    try:
        first = _play_session(
            sf=sf, game=_game_config(max_plies=200), games=2, max_steps=8,
            on_suspend=lambda st: _suspend_all(st, out_dir),
        )
        inflight = _inflight_records(first)
        assert inflight
        played = max(len(m) for m in inflight)
        at_resume: list[dict[str, Any]] = []

        def _resume(state: SelfplayState) -> None:
            report = resume_inflight_games(
                state, in_dir=out_dir, compat_fingerprint=FINGERPRINT,
                trial_id=TRIAL_ID,
            )
            # Snapshot NOW: a resumed game that finishes gets its slot
            # recycled, which clears the per-slot flag by design.
            at_resume.append({
                "resumed": int(report.resumed),
                "flags": list(state.resumed_from_disk),
                "plies": [
                    len(state.move_idx_history[i]) for i in range(state.batch_size)
                ],
                "records": [
                    len(state.samples_per_game[i]) for i in range(state.batch_size)
                ],
                "not_done": [
                    not state.done_arr[i] and not state.finalized_arr[i]
                    for i in range(state.batch_size)
                ],
            })

        # A ply budget just past what the resumed games have already played, so
        # they finish quickly instead of running a full game in a unit test.
        _play_session(
            sf=sf, game=_game_config(max_plies=played + 2), games=2, max_steps=60,
            on_state_ready=_resume, on_game_complete=completed.append, seed=1,
        )
    finally:
        sf.close()

    snap = at_resume[0]
    assert snap["resumed"] == len(inflight)
    assert any(snap["flags"]), "no slot was marked as resumed"
    for i, was_resumed in enumerate(snap["flags"]):
        if not was_resumed:
            continue
        # A resumed slot arrives mid-game: moves already played, records
        # already carried, and not marked done/finalized.
        assert snap["plies"][i] > 0
        assert snap["records"][i] > 0
        assert snap["not_done"][i]
    resumed_completions = sum(
        int(cg.outcome_stats.get("resumed_inflight_games", 0)) for cg in completed
    )
    assert resumed_completions >= 1, (
        "a resumed game finished but outcome_stats never said so"
    )
    assert resumed_completions <= len(completed)
    # Files are consumed, not replayed forever.
    assert not _state_files(out_dir)


@pytest.mark.skipif(SF_PATH is None, reason="Stockfish not found")
def test_discarded_games_reach_outcome_stats_on_the_production_path(
    tmp_path: Path,
) -> None:
    """The failure counter must ride the SAME plumbing as the success.

    `resume_discarded_games` is parked on the state at session start and has to
    survive finalize -> CompletedGameBatch -> shard meta -> result.json. Parking
    it is easy to get right and easy to have no effect; this asserts it actually
    comes out of a completed game batch.
    """
    assert SF_PATH is not None
    out_dir = tmp_path / "resume"
    sf = StockfishUCI(SF_PATH, nodes=80, multipv=4)
    completed: list[CompletedGameBatch] = []
    try:
        first = _play_session(
            sf=sf, game=_game_config(max_plies=200), games=2, max_steps=8,
            on_suspend=lambda st: _suspend_all(st, out_dir),
        )
        assert _inflight_records(first)
        files = _state_files(out_dir)
        assert files
        for path in files:  # every one unreadable => every one discarded
            raw = path.read_bytes()
            path.write_bytes(raw[: len(raw) // 2])

        def _resume(state: SelfplayState) -> None:
            resume_inflight_games(
                state, in_dir=out_dir, compat_fingerprint=FINGERPRINT,
                trial_id=TRIAL_ID,
            )

        # Short games so fresh slots actually finalize inside the step budget.
        _play_session(
            sf=sf, game=_game_config(max_plies=6), games=2, max_steps=60,
            on_state_ready=_resume, on_game_complete=completed.append, seed=1,
        )
    finally:
        sf.close()

    assert completed, "no game finalized, so nothing could carry the counter"
    assert sum(
        int(cg.outcome_stats.get("resume_discarded_games", 0)) for cg in completed
    ) == len(files), "discards never reached a CompletedGameBatch"


# ── negative controls: the gate must be able to fail ────────────────────────


def _fill_slot(
    state: SelfplayState, i: int, *, plies: int = 6, refute_opp_plies: bool = False,
) -> None:
    """Hand-build one in-flight game on slot ``i`` (no Stockfish).

    Records land on the plies the slot's OWN ``net_color`` is to move — the live
    rule (``_classify_live_slots_python`` partitions by exactly that), which is
    also what the net_color invariant on the resume side checks. With
    ``refute_opp_plies`` the opponent's plies carry SF-refute opponent rows, the
    only record kind whose ``pov_color`` is legitimately the other colour.
    """
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.moves import POLICY_SIZE, index_to_move
    from chess_anti_engine.selfplay.state import _NetRecord

    net_color = bool(state.net_color_arr[i])
    board = chess.Board()
    cb = CBoard.from_board(board)
    state.boards[i] = board.copy()
    assert state.starting_boards is not None
    state.starting_boards[i] = board.copy()
    state.samples_per_game[i] = []
    state.move_idx_history[i] = []
    for _ply in range(plies):
        legal = np.asarray(cb.legal_move_indices(), dtype=np.int64)
        probs = np.zeros(POLICY_SIZE, dtype=np.float32)
        probs[legal] = 1.0 / float(legal.size)
        is_net_turn = bool(cb.turn) == net_color
        if is_net_turn or refute_opp_plies:
            rec = _NetRecord(
                x=np.zeros((1,), np.float32),  # replaced by the re-encode
                policy_probs=probs,
                net_wdl_est=np.array([0.2, 0.5, 0.3], np.float32),
                search_wdl_est=np.array([0.25, 0.5, 0.25], np.float32),
                pov_color=chess.WHITE if cb.turn else chess.BLACK,
                ply_index=int(cb.ply),
                has_policy=True,
                priority=1.0,
                sample_weight=1.0,
                keep_prob=1.0,
                legal_mask=_mask_of(cb),
                move_offset=len(state.move_idx_history[i]),
                pos_hash=int(cb.zobrist_hash),
            )
            rec.is_sf_refute_opp = not is_net_turn
            state.samples_per_game[i].append(rec)
        action = int(legal[0])
        board.push(index_to_move(action, board))
        cb.push_index(action)
        state.move_idx_history[i].append(action)
    state.cboards[i] = cb
    state.starting_ply_arr[i] = 0
    state.done_arr[i] = 0
    state.finalized_arr[i] = 0


def _one_suspended_game(tmp_path: Path) -> tuple[Path, GameConfig, SelfplayState]:
    """A single hand-built in-flight game persisted to disk.

    Built without Stockfish so the corruption tests stay fast and deterministic:
    what they exercise is the LOADER's validation, which never sees an engine.
    """
    game = _game_config()
    state = _fresh_state(game, batch_size=2)
    _fill_slot(state, 0, plies=6)
    state.done_arr[1] = 1  # keep slot 1 out of the way

    out_dir = tmp_path / "resume"
    assert _suspend_all(state, out_dir) == 1
    return _state_files(out_dir)[0], game, state


def _mask_of(cb: Any) -> np.ndarray:
    from chess_anti_engine.moves import POLICY_SIZE

    mask = np.zeros(POLICY_SIZE, dtype=np.uint8)
    mask[np.asarray(cb.legal_move_indices(), dtype=np.int64)] = 1
    return mask


def _resume_one(path: Path, game: GameConfig) -> Any:
    target = _fresh_state(game, batch_size=2)
    report = resume_inflight_games(
        target, in_dir=path.parent, compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID,
    )
    return target, report


def test_prior_top1_survives_suspend_and_resume(tmp_path: Path) -> None:
    """The round-trip RESUME_FORMAT_VERSION was bumped 1 -> 2 to buy.

    ⚑ This is the assertion the format bump exists for. `prior_top1_index` /
    `prior_top1_prob` are captured from the ply's raw policy logits, so — unlike
    ``x`` / ``relations`` / the legal mask — they CANNOT be recomputed from the
    replayed board. If they do not survive the npz they are gone, and the PR
    that bumped the format paid a session of dropped in-flight games for
    nothing. Deleting the two `_rebuild_record` restore lines used to leave
    every suite green; it fails here.

    Values are chosen to make the weak passes impossible:

    * distinct per record, so a constant fill or a row permutation cannot pass;
    * dyadic probabilities, so "equal" means bit-equal through the float64
      column rather than equal-after-rounding;
    * one record deliberately carries NO prior, so absence must round-trip as
      absence — a decoder that dropped the presence flag and restored 0 / 0.0
      would otherwise look correct on the other rows.
    """
    game = _game_config()
    state = _fresh_state(game, batch_size=2)
    _fill_slot(state, 0, plies=6)
    state.done_arr[1] = 1  # keep slot 1 out of the way

    recs = state.samples_per_game[0]
    assert len(recs) >= 3, "need a row with a prior, a second one, and an absent one"
    want: list[tuple[int | None, float | None]] = [
        (101 + 7 * k, 0.125 + 0.0625 * k) for k in range(len(recs))
    ]
    want[-1] = (None, None)
    for rec, (idx, prob) in zip(recs, want, strict=True):
        rec.prior_top1_index = idx
        rec.prior_top1_prob = prob
    present = [idx for idx, _ in want if idx is not None]
    assert len(present) >= 2
    assert len(set(present)) == len(present)

    out_dir = tmp_path / "resume"
    assert _suspend_all(state, out_dir) == 1
    target, report = _resume_one(_state_files(out_dir)[0], game)
    assert report.resumed == 1
    assert report.discarded == 0

    got = [r for i in range(target.batch_size) for r in target.samples_per_game[i]]
    assert len(got) == len(recs)
    for k, (rec, (idx, prob)) in enumerate(zip(got, want, strict=True)):
        assert rec.prior_top1_index == idx, f"record {k}: prior_top1_index"
        assert rec.prior_top1_prob == prob, f"record {k}: prior_top1_prob"
    # ...and the restored objects are not the ones we wrote (a resume that
    # handed back the SOURCE records would pass everything above by identity).
    assert all(a is not b for a, b in zip(recs, got, strict=True))


def test_negative_control_dropped_move_is_rejected(tmp_path: Path) -> None:
    """Drop one move from the persisted list. Every later record then belongs
    to a different position — but NOT a different colour (the side to move at a
    given offset is fixed by parity), and the shifted move indices can still be
    legal. This control is why the record carries a position hash: without it
    the game resumed silently with mislabeled planes.
    """
    path, game, _src = _one_suspended_game(tmp_path)

    def _drop(arrays: dict[str, np.ndarray]) -> None:
        meta = _meta_of(arrays)
        meta["moves"] = list(meta["moves"])[1:]
        _set_meta(arrays, meta)

    _rewrite(path, _drop)
    target, report = _resume_one(path, game)

    assert report.resumed == 0
    assert report.discarded == 1
    assert report.reasons.get("position_mismatch") == 1
    assert not any(target.samples_per_game[i] for i in range(target.batch_size))
    assert not any(target.resumed_from_disk)


def test_negative_control_dropped_trailing_move_is_rejected(tmp_path: Path) -> None:
    """Drop the move AFTER the last record. Every record still lands on its own
    position, so the per-record check cannot see it — the suspended position's
    hash is what catches a game that would resume one ply behind reality."""
    path, game, _src = _one_suspended_game(tmp_path)

    def _drop_last(arrays: dict[str, np.ndarray]) -> None:
        meta = _meta_of(arrays)
        meta["moves"] = list(meta["moves"])[:-1]
        _set_meta(arrays, meta)

    _rewrite(path, _drop_last)
    _target, report = _resume_one(path, game)

    assert report.resumed == 0
    assert report.reasons.get("final_position_mismatch") == 1


def test_negative_control_truncated_record_column_is_rejected(tmp_path: Path) -> None:
    """Drop the last row of ONE per-record column. The header still claims
    n_records, so a loader that trusted array lengths would silently build a
    game one ply short with every later field shifted."""
    path, game, _src = _one_suspended_game(tmp_path)

    def _truncate(arrays: dict[str, np.ndarray]) -> None:
        arrays["ply_index"] = arrays["ply_index"][:-1]

    _rewrite(path, _truncate)
    _target, report = _resume_one(path, game)

    assert report.resumed == 0
    assert report.reasons.get("record_count_mismatch") == 1


def test_negative_control_shredded_bytes_are_rejected(tmp_path: Path) -> None:
    """A half-written file (worker killed mid-suspend) must be unreadable, not
    partially decoded."""
    path, game, _src = _one_suspended_game(tmp_path)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])

    _target, report = _resume_one(path, game)

    assert report.resumed == 0
    assert report.discarded == 1


def test_config_mismatch_and_version_mismatch_are_discarded(tmp_path: Path) -> None:
    """The single decision point rejects a game recorded under different
    record-shaping config, or a different on-disk format."""
    path, game, _src = _one_suspended_game(tmp_path)
    target = _fresh_state(game, batch_size=2)
    report = resume_inflight_games(
        target, in_dir=path.parent, compat_fingerprint="a-different-fingerprint",
        trial_id=TRIAL_ID,
    )
    assert report.resumed == 0
    assert report.reasons.get("config_mismatch") == 1

    assert should_resume_game(
        {"format_version": RESUME_FORMAT_VERSION + 1,
         "compat_fingerprint": FINGERPRINT},
        compat_fingerprint=FINGERPRINT,
    ) == "version_mismatch"
    assert should_resume_game(
        {"format_version": RESUME_FORMAT_VERSION, "compat_fingerprint": "other"},
        compat_fingerprint=FINGERPRINT,
    ) == "config_mismatch"
    assert should_resume_game(
        {"format_version": RESUME_FORMAT_VERSION, "compat_fingerprint": FINGERPRINT,
         "trial_id": "other"},
        compat_fingerprint=FINGERPRINT, trial_id=TRIAL_ID,
    ) == "trial_mismatch"
    ok = {
        "format_version": RESUME_FORMAT_VERSION,
        "compat_fingerprint": FINGERPRINT,
        "trial_id": TRIAL_ID,
        "root_fen": chess.STARTING_FEN,
        "moves": [], "opening_moves": [],
        "final_pos_hash": 12345,
        "saved_at": time.time(),
    }
    assert should_resume_game(
        ok, compat_fingerprint=FINGERPRINT, trial_id=TRIAL_ID,
    ) is None
    assert should_resume_game(
        {**ok, "final_pos_hash": 0}, compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID,
    ) == "malformed_meta"


def test_negative_control_zeroed_tail_hash_is_rejected(tmp_path: Path) -> None:
    """The tail guard must not be switchable off by the file it guards. Blanking
    `final_pos_hash` is rejected outright, so the dropped-trailing-move control
    above cannot be defeated by also clearing the field."""
    path, game, _src = _one_suspended_game(tmp_path)

    def _blank(arrays: dict[str, Any]) -> None:
        meta = _meta_of(arrays)
        meta["final_pos_hash"] = 0
        meta["moves"] = list(meta["moves"])[:-1]
        _set_meta(arrays, meta)

    _rewrite(path, _blank)
    _target, report = _resume_one(path, game)

    assert report.resumed == 0
    assert report.reasons.get("malformed_meta") == 1


def test_negative_control_tampered_ply_index_is_rejected(tmp_path: Path) -> None:
    """`ply_index` is not derivable from the moves, and finalize keys its
    sf_p0 one-ply shift off it — a wrong value shifts a teacher instead of
    failing. On the C play path it must equal the replayed CBoard ply."""
    path, game, _src = _one_suspended_game(tmp_path)

    def _shift(arrays: dict[str, Any]) -> None:
        arrays["ply_index"] = arrays["ply_index"] + 100

    _rewrite(path, _shift)
    _target, report = _resume_one(path, game)

    assert report.resumed == 0
    assert report.reasons.get("ply_index_mismatch") == 1


def test_a_game_from_another_trial_is_discarded(tmp_path: Path) -> None:
    """A trial reassignment also tears the session down. A game played under
    another trial's hyperparameters must not be finished and uploaded here."""
    path, game, _src = _one_suspended_game(tmp_path)
    target = _fresh_state(game, batch_size=2)
    report = resume_inflight_games(
        target, in_dir=path.parent, compat_fingerprint=FINGERPRINT,
        trial_id="some-other-trial",
    )
    assert report.resumed == 0
    assert report.reasons.get("trial_mismatch") == 1


def test_stale_games_are_discarded_and_swept(tmp_path: Path) -> None:
    """A game the fleet never came back for is from another era of the run —
    and nothing else on disk would ever delete it."""
    path, game, _src = _one_suspended_game(tmp_path)

    def _age(arrays: dict[str, Any]) -> None:
        meta = _meta_of(arrays)
        meta["saved_at"] = time.time() - 100_000  # ~27.8h, past the 6h default
        _set_meta(arrays, meta)

    _rewrite(path, _age)
    target = _fresh_state(game, batch_size=2)
    report = resume_inflight_games(
        target, in_dir=path.parent, compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID,
    )
    # Expiry runs through the SINGLE decision point, so it is counted with a
    # reason rather than silently swept.
    assert report.resumed == 0
    assert report.reasons.get("stale") == 1
    assert not _state_files(path.parent)

    # The sweep removes the debris a killed suspend/resume leaves behind.
    orphan_tmp = path.parent / f"x{RESUME_FILE_SUFFIX}.tmp"
    orphan_claimed = path.parent / f"y{RESUME_FILE_SUFFIX}.claimed"
    orphan_tmp.write_bytes(b"junk")
    orphan_claimed.write_bytes(b"junk")
    old = time.time() - 86_400
    os.utime(orphan_tmp, (old, old))
    os.utime(orphan_claimed, (old, old))
    _fresh = _fresh_state(game, batch_size=2)
    resume_inflight_games(
        _fresh, in_dir=path.parent, compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID, max_age_s=3600.0,
    )
    assert not orphan_tmp.exists()
    assert not orphan_claimed.exists()


def test_resume_never_overwrites_a_seeded_slot(tmp_path: Path) -> None:
    """A fenlist slot has already CONSUMED a doled blind-spot seed at create
    time and the seed line cannot be pushed back, so resuming over it would
    destroy the seed silently — the dole's historical failure mode."""
    path, game, _src = _one_suspended_game(tmp_path)
    target = _fresh_state(game, batch_size=2)
    # Both slots look seeded => nothing may be overwritten.
    target.opening_source_arr[0] = "fenlist"
    target.opening_source_arr[1] = "fenlist_dole"
    report = resume_inflight_games(
        target, in_dir=path.parent, compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID,
    )
    assert (report.resumed, report.discarded) == (0, 0)
    assert not any(target.resumed_from_disk)
    # The state file is left for the next session rather than claimed and lost.
    assert _state_files(path.parent)

    # One free slot => exactly that slot is used, and the seeded one is intact.
    target2 = _fresh_state(game, batch_size=2)
    target2.opening_source_arr[0] = "fenlist"
    target2.opening_source_arr[1] = "book"
    report2 = resume_inflight_games(
        target2, in_dir=path.parent, compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID,
    )
    assert report2.resumed == 1
    assert target2.resumed_from_disk == [False, True]
    assert target2.opening_source_arr[0] == "fenlist"


def test_negative_control_flipped_net_color_is_rejected(tmp_path: Path) -> None:
    """`net_color` is not derivable from the MOVE LIST — but it is derivable
    from the RECORDS. On a curriculum game every non-refute record is a net
    turn, so net_color must equal the single pov_color among them.

    Flipping it is silent otherwise: no crash, no metric move. It decides which
    seat the net keeps playing (`_classify_live_slots_python` compares the board
    turn against it) and finalize scores the game's w/d/l from that seat, so a
    flipped header hands the PID a win the net actually lost.
    """
    path, game, _src = _one_suspended_game(tmp_path)
    before = _meta_of(_load_arrays(path))
    assert int(before["is_selfplay"]) == 0, "the invariant only binds curriculum games"

    def _flip(arrays: dict[str, Any]) -> None:
        meta = _meta_of(arrays)
        meta["net_color"] = 1 - int(meta["net_color"])
        _set_meta(arrays, meta)

    _rewrite(path, _flip)
    target, report = _resume_one(path, game)

    assert report.resumed == 0
    assert report.reasons.get("net_color_mismatch") == 1
    assert not any(target.resumed_from_disk)


def test_net_color_is_not_checked_on_a_selfplay_game(tmp_path: Path) -> None:
    """The net plays BOTH seats in selfplay, so pov_color alternates and the
    records constrain nothing. Asserting there would reject valid games."""
    path, game, _src = _one_suspended_game(tmp_path)

    def _flip_selfplay(arrays: dict[str, Any]) -> None:
        meta = _meta_of(arrays)
        meta["is_selfplay"] = 1
        meta["net_color"] = 1 - int(meta["net_color"])
        _set_meta(arrays, meta)

    _rewrite(path, _flip_selfplay)
    _target, report = _resume_one(path, game)
    assert (report.resumed, report.discarded) == (1, 0)


def test_a_resumed_refute_opp_record_gains_no_alternate_inputs(
    tmp_path: Path,
) -> None:
    """`_emit_sf_refute_opp_record` ALWAYS builds its row with x_lc0_root=None
    and relations=None, whatever record_lc0_root_input/record_relations say. A
    resumed refute row that re-encoded them would carry columns its non-resumed
    twin does not — the same row shape differing by whether a restart happened.
    """
    game = _game_config(
        input_history_encoding=LC0_HISTORY_LEGACY,
        record_lc0_root_input=True,
        record_relations=True,
    )
    state = _fresh_state(game, batch_size=2)
    _fill_slot(state, 0, plies=6, refute_opp_plies=True)
    state.done_arr[1] = 1
    out_dir = tmp_path / "resume"
    assert _suspend_all(state, out_dir) == 1

    target, report = _resume_one(_state_files(out_dir)[0], game)
    assert report.resumed == 1
    records = next(iter(_inflight_records(target).values()))
    net_rows = [r for r in records if not r.is_sf_refute_opp]
    opp_rows = [r for r in records if r.is_sf_refute_opp]
    assert net_rows, "the fixture must produce net rows"
    assert opp_rows, "the fixture must produce refute-opp rows"
    # The config IS on: the net rows prove the alternate columns were available.
    for rec in net_rows:
        assert rec.x_lc0_root is not None
        assert rec.relations is not None
    for rec in opp_rows:
        assert rec.x_lc0_root is None, "a refute-opp row must not gain x_lc0_root"
        assert rec.relations is None, "a refute-opp row must not gain relations"


def test_a_failed_write_costs_exactly_that_game(tmp_path: Path) -> None:
    """`_write_game` json.dumps(meta) has no `default=`, so a non-JSON value
    raises TypeError, not OSError. Under a narrow `except OSError` it escapes
    suspend_inflight_games entirely, hits the worker's blanket handler and
    abandons EVERY remaining in-flight game — including ones already written.
    """
    game = _game_config()
    state = _fresh_state(game, batch_size=3)
    for i in range(3):
        _fill_slot(state, i, plies=4 + 2 * i)
    # Slot 1's diag is not JSON-serializable => TypeError inside _write_game.
    state.samples_per_game[1][0].gumbel_policy_diag = cast("Any", object())

    out_dir = tmp_path / "resume"
    report = suspend_inflight_games(
        state, out_dir=out_dir, compat_fingerprint=FINGERPRINT,
        model_sha="a" * 64, model_step=7, trial_id=TRIAL_ID,
    )
    assert report.persisted == 2, "one bad game must not abort the whole suspend"
    assert report.reasons.get("write_failed") == 1
    assert len(_state_files(out_dir)) == 2
    # ...and no half-written archive was left behind for the next session.
    assert not list(out_dir.glob(f"*{RESUME_FILE_SUFFIX}.tmp"))


def test_a_game_with_no_trial_id_is_neither_written_nor_resumed(
    tmp_path: Path,
) -> None:
    """`"" == ""` would make the trial guard match EVERYTHING. An unknown trial
    is not a matching trial, so an empty id must refuse on both sides — and a
    game that could never be resumed must not be written at all."""
    game = _game_config()
    state = _fresh_state(game, batch_size=2)
    _fill_slot(state, 0, plies=6)
    state.done_arr[1] = 1
    out_dir = tmp_path / "resume"
    report = suspend_inflight_games(
        state, out_dir=out_dir, compat_fingerprint=FINGERPRINT,
        model_sha="", model_step=0, trial_id="",
    )
    assert report.persisted == 0
    assert report.reasons.get("no_trial_id") == 1
    assert not _state_files(out_dir)

    # ...and the read side refuses independently, so a file written by an older
    # build cannot be matched by an empty id either.
    ok = {
        "format_version": RESUME_FORMAT_VERSION,
        "compat_fingerprint": FINGERPRINT,
        "root_fen": chess.STARTING_FEN,
        "moves": [], "opening_moves": [],
        "final_pos_hash": 12345,
        "saved_at": time.time(),
    }
    assert should_resume_game(
        {**ok, "trial_id": ""}, compat_fingerprint=FINGERPRINT, trial_id="",
    ) == "no_trial_id"
    assert should_resume_game(
        {**ok, "trial_id": TRIAL_ID}, compat_fingerprint=FINGERPRINT, trial_id="",
    ) == "no_trial_id"
    assert should_resume_game(
        {**ok, "trial_id": ""}, compat_fingerprint=FINGERPRINT, trial_id=TRIAL_ID,
    ) == "no_trial_id"


def test_missing_state_directory_is_a_silent_no_op(tmp_path: Path) -> None:
    game = _game_config()
    target = _fresh_state(game, batch_size=2)
    report = resume_inflight_games(
        target, in_dir=tmp_path / "does-not-exist", compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID,
    )
    assert (report.resumed, report.discarded) == (0, 0)
    assert not any(target.resumed_from_disk)


def test_a_state_file_is_claimed_once(tmp_path: Path) -> None:
    """Two sessions (or two selfplay threads) must not both resume one game."""
    path, game, _src = _one_suspended_game(tmp_path)
    first_target, first = _resume_one(path, game)
    second_target, second = _resume_one(path, game)

    assert first.resumed == 1
    assert (second.resumed, second.discarded) == (0, 0)
    assert any(first_target.resumed_from_disk)
    assert not any(second_target.resumed_from_disk)


def test_empty_slots_are_not_persisted(tmp_path: Path) -> None:
    """A slot that has played nothing is indistinguishable from a fresh game;
    persisting it would churn files for no gain."""
    game = _game_config()
    state = _fresh_state(game, batch_size=3)
    out_dir = tmp_path / "resume"
    report = suspend_inflight_games(
        state, out_dir=out_dir, compat_fingerprint=FINGERPRINT,
        model_sha="", model_step=0, trial_id=TRIAL_ID,
    )
    assert report.persisted == 0
    assert report.reasons.get("empty_slot") == 3
    assert not _state_files(out_dir)
    # An empty slot is not a LOSS: `_finalize_completed_slots` recycles every
    # finished slot just before teardown, so most of a real table is empty here
    # and counting it as `skipped` would bury serialize_failed / write_failed.
    assert (report.skipped, report.empty_slots) == (0, 3)


@pytest.mark.parametrize("untracked", ["move_offset", "pos_hash"])
def test_record_without_replay_bookkeeping_is_not_persisted(
    tmp_path: Path, untracked: str,
) -> None:
    """move_offset says WHERE a record's position is and pos_hash proves the
    replay found it. A record missing either takes its whole game out of the
    resume path — the alternative is a guessed position."""
    from chess_anti_engine.moves import POLICY_SIZE
    from chess_anti_engine.selfplay.state import _NetRecord

    # dict[str, Any]: only ever holds move_offset/pos_hash ints, but pyright
    # checks a ** unpack against EVERY keyword param, including the
    # ndarray-typed ones.
    tracked: dict[str, Any] = {"move_offset": 0, "pos_hash": 12345}
    del tracked[untracked]  # left at its "not tracked" default
    game = _game_config()
    state = _fresh_state(game, batch_size=1)
    state.move_idx_history[0] = [0]
    state.samples_per_game[0] = [
        _NetRecord(
            x=np.zeros((1,), np.float32),
            policy_probs=np.zeros(POLICY_SIZE, np.float32),
            net_wdl_est=np.zeros(3, np.float32),
            search_wdl_est=np.zeros(3, np.float32),
            pov_color=chess.WHITE, ply_index=0, has_policy=True,
            priority=1.0, sample_weight=1.0, keep_prob=1.0,
            **tracked,
        ),
    ]
    out_dir = tmp_path / "resume"
    report = suspend_inflight_games(
        state, out_dir=out_dir, compat_fingerprint=FINGERPRINT,
        model_sha="", model_step=0, trial_id=TRIAL_ID,
    )
    assert report.persisted == 0
    assert report.reasons.get("serialize_failed") == 1
    assert report.skipped == 1  # a real loss, counted as one


# ── worker wiring ───────────────────────────────────────────────────────────


def test_every_restart_key_is_classified_for_resume() -> None:
    """Completeness guard: a new restart key must be declared either as
    record-shaping (games recorded under the old value are discarded) or as
    exempt. Unclassified would silently mean "exempt", which is how a game ends
    up carrying two different record schemas into one shard."""
    from chess_anti_engine.worker import WorkerSession

    restart = set(WorkerSession._RECO_RESTART_KEYS)
    compat = set(WorkerSession._RESUME_COMPAT_KEYS)
    exempt = set(WorkerSession._RESUME_COMPAT_EXEMPT_KEYS)
    assert not compat & exempt, sorted(compat & exempt)
    assert not restart - (compat | exempt), sorted(restart - (compat | exempt))
    assert not (compat | exempt) - restart, sorted((compat | exempt) - restart)


def test_resume_flag_is_watched_and_defaults_off() -> None:
    """The key must exist in code BEFORE it can appear in a live yaml (the
    validator rejects the whole reload on an unknown key), and it must reach a
    worker rather than sit in the trial config."""
    from chess_anti_engine.model import ModelConfig
    from chess_anti_engine.tune.distributed_runtime import build_recommended_worker
    from chess_anti_engine.tune.trial_config import TrialConfig
    from chess_anti_engine.utils.config_yaml import SELFPLAY_CONFIG_KEYS
    from chess_anti_engine.worker import WorkerSession

    key = "selfplay_resume_inflight_games"
    assert key in SELFPLAY_CONFIG_KEYS
    assert key in WorkerSession._RECO_WATCH_KEYS
    assert getattr(TrialConfig(), key) is False
    flat = flatten_run_config_defaults({"selfplay": {key: True}})
    assert flat[key] is True
    assert getattr(TrialConfig.from_dict(flat), key) is True
    # ...and it is actually PUBLISHED: the manifest block is the only channel
    # to a worker, so a key missing here would leave the worker on its default
    # while the yaml claimed otherwise.
    published = build_recommended_worker(
        config=flat, model_cfg=ModelConfig(), sf_nodes=1000, mcts_simulations=32,
    )
    assert published[key] is True
    assert build_recommended_worker(
        config={}, model_cfg=ModelConfig(), sf_nodes=1000, mcts_simulations=32,
    )[key] is False


def _bare_session(tmp_path: Path) -> Any:
    """A WorkerSession with only the fields the resume plumbing touches."""
    import logging

    from chess_anti_engine.worker import WorkerSession

    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("test-worker")
    session.resume_dir = tmp_path / "selfplay_resume"
    session.resume_dir.mkdir(parents=True, exist_ok=True)
    session.fixed_trial_id = ""
    session.leased_trial_id = ""
    return session


def test_trial_id_is_snapshot_at_session_start_not_read_at_suspend(
    tmp_path: Path,
) -> None:
    """The ordering bug, not just a mismatched id at rest.

    `_negotiate_lease` assigns `self.leased_trial_id = new_trial_id`
    (worker.py) and only THEN does its caller notice the change and set
    `_stop_selfplay`. The selfplay thread runs `on_suspend` after that, so a
    LIVE read inside the suspend hook returns trial B while every ply in the
    game was played under trial A. Those games would then pass the trial guard
    and be uploaded as trial B's data — and `config_mismatch` cannot catch it,
    since sibling GPBT trials differ on LR and loss weights, not on the 11
    record-shaping keys.
    """
    session = _bare_session(tmp_path)
    session.leased_trial_id = "trial-a"
    session._begin_resume_session({"selfplay_resume_inflight_games": True})
    assert session._resume_trial_id() == "trial-a"

    # The reassignment lands BEFORE the suspend hook runs. This is the mutation.
    session.leased_trial_id = "trial-b"
    assert session._current_trial_id() == "trial-b"
    assert session._resume_trial_id() == "trial-a", (
        "the suspend hook must stamp the trial the GAMES were played under"
    )

    # ...and end to end: a game stamped trial-a is refused by a trial-b session.
    game = _game_config()
    state = _fresh_state(game, batch_size=2)
    _fill_slot(state, 0, plies=6)
    state.done_arr[1] = 1
    out_dir = tmp_path / "resume"
    assert suspend_inflight_games(
        state, out_dir=out_dir, compat_fingerprint=FINGERPRINT,
        model_sha="", model_step=0, trial_id=session._resume_trial_id(),
    ).persisted == 1

    target = _fresh_state(game, batch_size=2)
    report = resume_inflight_games(
        target, in_dir=out_dir, compat_fingerprint=FINGERPRINT,
        trial_id="trial-b",
    )
    assert report.resumed == 0
    assert report.reasons.get("trial_mismatch") == 1


def test_orphan_sweep_runs_with_the_resume_flag_off(tmp_path: Path) -> None:
    """Turning the flag OFF is the documented revert, and it disables the drain
    along with the suspend — so a flag-gated sweep would never run in the one
    case the backstop is named for, and the last teardown's files would sit in
    selfplay_resume/ forever."""
    session = _bare_session(tmp_path)
    session.leased_trial_id = "trial-a"
    stale_game = session.resume_dir / f"old{RESUME_FILE_SUFFIX}"
    orphan_tmp = session.resume_dir / f"x{RESUME_FILE_SUFFIX}.tmp"
    fresh_game = session.resume_dir / f"new{RESUME_FILE_SUFFIX}"
    for p in (stale_game, orphan_tmp, fresh_game):
        p.write_bytes(b"junk")
    old = time.time() - 20 * 86_400
    for p in (stale_game, orphan_tmp):
        os.utime(p, (old, old))

    session._begin_resume_session({"selfplay_resume_inflight_games": False})

    assert session._resume_inflight_enabled is False
    assert not stale_game.exists(), "flag-off residue must still be swept"
    assert not orphan_tmp.exists()
    assert fresh_game.exists(), "a recent file is for the next session, not litter"


def test_discarded_games_are_parked_for_the_next_finalize(tmp_path: Path) -> None:
    """`resumed_inflight_games` (the success) reaches result.json; the FAILURE
    must reach the same place, not just a worker log. A discard has no game of
    its own, so resume parks it and the next finalize drains it."""
    path, game, _src = _one_suspended_game(tmp_path)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])  # unreadable => discarded

    target = _fresh_state(game, batch_size=2)
    report = resume_inflight_games(
        target, in_dir=path.parent, compat_fingerprint=FINGERPRINT,
        trial_id=TRIAL_ID,
    )
    assert report.discarded == 1
    assert target.pending_outcome_stats == {"resume_discarded_games": 1}


def test_compat_fingerprint_moves_only_with_record_shaping_keys() -> None:
    from chess_anti_engine.worker import WorkerSession

    session = object.__new__(WorkerSession)
    base: dict[str, Any] = {
        k: i for i, k in enumerate(WorkerSession._RECO_RESTART_KEYS)
    }
    fp = session._resume_compat_fingerprint(base)
    for key in WorkerSession._RESUME_COMPAT_EXEMPT_KEYS:
        moved = dict(base)
        moved[key] = "changed"
        assert session._resume_compat_fingerprint(moved) == fp, key
    for key in WorkerSession._RESUME_COMPAT_KEYS:
        moved = dict(base)
        moved[key] = "changed"
        assert session._resume_compat_fingerprint(moved) != fp, key


# ── refusals about OUR state must not destroy the file ──────────────────────


def test_a_refusal_about_our_state_preserves_the_file(tmp_path: Path) -> None:
    """A leased worker whose model-sha change cleared `leased_trial_id` starts
    its next session with trial_id="" — every suspended game then refuses
    `no_trial_id`. That is a fact about OUR side, not a defect of the FILE, so
    the file must survive for a later, matching session; unlinking it would
    make the feature DESTROY the previous session's games on exactly the
    leased workers it was built for. Same for `trial_mismatch` and
    `config_mismatch`.
    """
    path, game, _src = _one_suspended_game(tmp_path)

    for fingerprint, trial_id, reason in (
        (FINGERPRINT, "", "no_trial_id"),
        (FINGERPRINT, "some-other-trial", "trial_mismatch"),
        ("another-fingerprint", TRIAL_ID, "config_mismatch"),
    ):
        target = _fresh_state(game, batch_size=2)
        report = resume_inflight_games(
            target, in_dir=path.parent, compat_fingerprint=fingerprint,
            trial_id=trial_id,
        )
        assert (report.resumed, report.discarded) == (0, 0), reason
        assert report.preserved == 1, reason
        assert report.reasons.get(reason) == 1
        # The file survives, un-claimed, for the next session...
        assert _state_files(path.parent), f"{reason} must not destroy the file"
        assert not list(path.parent.glob("*.claimed")), reason
        # ...and nothing is parked as a LOSS, because the game is not lost.
        assert not target.pending_outcome_stats

    # A matching session can still resume the very same file.
    target, report = _resume_one(path, game)
    assert report.resumed == 1
    assert any(target.resumed_from_disk)


def test_no_trial_id_at_session_start_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The leased-worker gap must be loud: with the flag on and no trial id,
    nothing can be persisted or resumed this session, and without a log line
    `resumed_inflight_games: 0` looks like the feature simply not working."""
    import logging

    session = _bare_session(tmp_path)
    session.leased_trial_id = ""
    with caplog.at_level(logging.WARNING, logger="test-worker"):
        session._begin_resume_session({"selfplay_resume_inflight_games": True})
    assert any("no trial id" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="test-worker"):
        session._begin_resume_session({"selfplay_resume_inflight_games": False})
    assert not any("no trial id" in r.message for r in caplog.records), (
        "flag off must not warn — nothing is armed to be surprised about"
    )


# ── ply-index convention guard ───────────────────────────────────────────────


def test_a_game_from_the_other_ply_convention_is_rejected(tmp_path: Path) -> None:
    """`ply_index` means CBoard.ply on the C path but len(move_stack) on the
    Python fallback — different origins for a seeded opening. `has_c_ply` says
    which convention the stored values use; a session on the OTHER convention
    cannot validate them (the replay's ply check keys off the file's own flag)
    and finalize keys its sf_p0 one-ply shift off ply_index, so the game must
    be refused, not resumed under a reinterpreted index."""
    path, game, _src = _one_suspended_game(tmp_path)

    def _flip(arrays: dict[str, Any]) -> None:
        meta = _meta_of(arrays)
        meta["has_c_ply"] = not bool(meta["has_c_ply"])
        _set_meta(arrays, meta)

    _rewrite(path, _flip)
    target, report = _resume_one(path, game)

    assert report.resumed == 0
    assert report.reasons.get("ply_convention_mismatch") == 1
    assert not any(target.resumed_from_disk)


# ── the worker->play_batch wirings ───────────────────────────────────────────


def _wired_session(tmp_path: Path) -> Any:
    """A WorkerSession with enough state to drive the REAL `_run_selfplay`
    through the REAL `_dispatch_selfplay_one_shard`, with `play_batch` faked.

    Everything the wiring under test touches is real: `_begin_resume_session`,
    the on_suspend conditional, `_register_live_state` and the resume hook.
    Everything else (engine sync, evaluators, upload) is stubbed out.
    """
    import threading
    from types import SimpleNamespace

    session = _bare_session(tmp_path)
    session.leased_trial_id = TRIAL_ID
    session.device = "cpu"
    session.model_sha = "f" * 64
    session.model_step = 3
    session.model = None
    session.inference_client = object()  # broker mode: no local model needed
    session._direct_evaluator = None
    session.games_per_batch_local = 2
    session.rng = np.random.default_rng(0)
    session.args = SimpleNamespace(
        threaded_selfplay=False, selfplay_threads=1, poll_seconds=0.0,
        stockfish_from_server=False,
    )
    session.sf = None
    session._live_states_lock = threading.Lock()
    session._live_states = []
    session._pending_live_override = None
    session._dole_lock = threading.Lock()
    session._live_dole_queue = None
    session._live_sf_refute_queue = None
    session._pending_fen_dole = []
    session._pending_sf_refute = []
    session._resume_counts_lock = threading.Lock()
    session._resume_counts = {
        **initial_resume_counts(),
    }
    session._resume_skip_reasons = {}
    # Collaborators outside the wiring under test.
    session._build_selfplay_configs = lambda reco: (
        dict(_SELFPLAY_CFGS_DOUBLE), (1000, 4, 16, None),
    )
    session._sync_stockfish = lambda *a, **k: setattr(session, "sf", object())
    session._promote_pending_dole = lambda: ([], [])
    session._note_selfplay_progress = lambda: None
    session._start_selfplay_stall_watchdog = lambda: None
    session._start_model_watch_thread = lambda: None
    session._flush_and_upload_after_shard = lambda *a, **k: None
    return session


def test_the_selfplay_configs_double_matches_the_real_contract(
    tmp_path: Path,
) -> None:
    """`_wired_session`'s `_build_selfplay_configs` double must not drift.

    A double that returns FEWER keys than the collaborator it replaces does not
    fail loudly — it makes the code under test take a different branch. Both
    halves of that happened here: the missing `game` key had been quietly
    sending `_await_broker_ready` down its "probe construction failed" skip, and
    the `cfgs["search"]` read added for the selfplay policy-temp log line turned
    the same drift into a KeyError. Compared against the REAL return value, so a
    key added to production shows up as a failure here rather than as a hole.
    """
    from types import SimpleNamespace

    session = _bare_session(tmp_path)
    session.args = SimpleNamespace()
    session.opening_book_path = None
    session.opening_book_path_2 = None
    session.opening_fen_list_path = None

    real, real_sf_args = session._build_selfplay_configs(
        {"sf_nodes": 1000, "sf_multipv": 4, "sf_hash_mb": 16},
    )
    double, double_sf_args = _wired_session(tmp_path)._build_selfplay_configs({})

    assert set(double) == set(real)
    # The TYPES too: every `cfgs[...]` reader and `play_batch(**cfgs)` is
    # written against the dataclasses, so a bare dict standing in for a
    # SearchConfig would satisfy a key-set check and still not be the contract.
    assert {k: type(v) for k, v in double.items()} == {
        k: type(v) for k, v in real.items()
    }
    assert len(double_sf_args) == len(real_sf_args)


@pytest.mark.parametrize(
    "threaded", [False, True], ids=["single-path", "threaded-path"],
)
def test_run_selfplay_wires_resume_hooks_into_play_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threaded: bool,
) -> None:
    """The three worker-side wirings, driven through the real `_run_selfplay`:

    1. `_run_selfplay` calls `_begin_resume_session(reco)` — delete it and the
       flag sits at its class default False while the yaml says true;
    2. `_dispatch_selfplay_one_shard` / `_run_selfplay_threaded` hand
       `on_suspend=self._suspend_inflight_games` to play_batch;
    3. `_register_live_state` (play_batch's on_state_ready) calls
       `_resume_inflight_games`, which really restores a persisted game.

    The end-to-end tests supply their own hooks, so they prove play_batch's
    side of the contract, not the worker's — this is the worker's side. The
    concrete failure this pins: yaml says true, a hook silently unwired,
    `resumed_inflight_games` stays 0 forever and nothing crashes.
    """
    import chess_anti_engine.worker as worker_mod
    from chess_anti_engine.selfplay.state import BatchStats

    game = _game_config()
    session = _wired_session(tmp_path)
    session.args.threaded_selfplay = threaded
    reco = {"selfplay_resume_inflight_games": True}

    # One real suspended game, written by the WORKER's own suspend hook so the
    # file carries the same fingerprint/trial the session's resume side reads.
    source = _fresh_state(game, batch_size=2)
    _fill_slot(source, 0, plies=6)
    source.done_arr[1] = 1
    session._begin_resume_session(reco)
    session._suspend_inflight_games(source)
    assert session._resume_counts["suspended"] == 1
    # Reset the session-fixed state to its class defaults: only _run_selfplay's
    # OWN _begin_resume_session call may re-arm it, otherwise the direct call
    # above would mask a deleted wiring (the exact mutation this test pins).
    session._resume_inflight_enabled = False
    session._resume_compat_fingerprint_active = ""
    session._resume_trial_id_active = ""

    captured: list[dict[str, Any]] = []
    states: list[SelfplayState] = []

    def _fake_play_batch(_model: Any, **kwargs: Any) -> Any:
        captured.append(kwargs)
        state = _fresh_state(game, batch_size=2, seed=9)
        states.append(state)
        on_ready = kwargs.get("on_state_ready")
        if on_ready is not None:
            on_ready(state)
        return [], BatchStats(games=0, positions=0, w=0, d=0, l=0)

    monkeypatch.setattr(worker_mod, "play_batch", _fake_play_batch)
    session._run_selfplay({"recommended_worker": reco})

    # Wiring 1: the session began a resume session from the reco.
    assert session._resume_inflight_enabled is True
    assert session._resume_trial_id_active == TRIAL_ID
    # Wiring 2: play_batch got the worker's real suspend hook, not None.
    (kw,) = captured
    assert kw["on_suspend"] == session._suspend_inflight_games
    assert kw["on_state_ready"] == session._register_live_state
    # Wiring 3: on_state_ready really brought the persisted game back.
    assert session._resume_counts["resumed"] == 1
    (state,) = states
    assert any(state.resumed_from_disk)
    assert not _state_files(session.resume_dir)


def test_run_selfplay_keeps_hooks_dark_with_the_flag_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag off = exactly today's behaviour: no suspend hook reaches play_batch
    and no resume runs at state registration."""
    import chess_anti_engine.worker as worker_mod
    from chess_anti_engine.selfplay.state import BatchStats

    game = _game_config()
    session = _wired_session(tmp_path)
    captured: list[dict[str, Any]] = []

    def _fake_play_batch(_model: Any, **kwargs: Any) -> Any:
        captured.append(kwargs)
        state = _fresh_state(game, batch_size=2, seed=9)
        on_ready = kwargs.get("on_state_ready")
        if on_ready is not None:
            on_ready(state)
        return [], BatchStats(games=0, positions=0, w=0, d=0, l=0)

    monkeypatch.setattr(worker_mod, "play_batch", _fake_play_batch)
    session._run_selfplay(
        {"recommended_worker": {"selfplay_resume_inflight_games": False}},
    )

    assert session._resume_inflight_enabled is False
    (kw,) = captured
    assert kw["on_suspend"] is None
    assert session._resume_counts["resumed"] == 0


def test_surplus_suspended_games_are_stranded_and_only_the_new_counter_sees_them(
    tmp_path: Path,
) -> None:
    """⚑ SUPPLY > DEMAND, which is the only regime in which this can fire.

    `resume_inflight_games` walks `sorted(glob(...))` and breaks at
    `report.resumed >= len(slots)`. It is DEMAND-driven: it restores as many
    games as the restarting threads have slots for, and stops. When the previous
    session suspended MORE games than the new one asks for, the surplus is never
    claimed, never decoded, and so never reaches `note()` or `note_preserved()`.

    Both pre-existing counters therefore read a TRUTHFUL ZERO on a real loss:
    `discarded` counts files the resume examined and rejected, and
    `suspend_skipped` counts games suspend failed to write. Neither is wrong;
    neither can see this. The stranded files then expire at DEFAULT_MAX_AGE_S
    and the sweep deletes them.

    MEASURED in production first (2026-08-14 arm-B pause/resume): suspend 3046,
    resume 3017, and exactly 29 *.game.npz left in worker_02's directory --
    the entire gap, in one worker, with discarded=0 and suspend_skipped=0
    everywhere. This test reproduces that shape in miniature.

    ⚑ A version of this test with batch_size >= the number of suspended games
    would pass with the new counter hard-wired to 0, which is why the asserted
    numbers are pinned exactly rather than as "> 0".
    """
    game = _game_config()
    source = _fresh_state(game, batch_size=4)
    for slot in range(3):
        _fill_slot(source, slot, plies=6)
    source.done_arr[3] = 1  # keep the fourth slot out of the way
    out_dir = tmp_path / "resume"
    assert _suspend_all(source, out_dir) == 3, "the SUPPLY side must be 3"
    assert count_unclaimed_resume_files(out_dir) == 3

    # DEMAND is one slot. Two games cannot be placed anywhere.
    target = _fresh_state(game, batch_size=1, seed=11)
    report = resume_inflight_games(
        target, in_dir=out_dir, compat_fingerprint=FINGERPRINT, trial_id=TRIAL_ID,
    )

    assert report.resumed == 1, "demand was one slot"
    # THE BLIND SPOT, asserted as such: a real loss with both counters clean.
    assert report.discarded == 0, "nothing was examined-and-rejected"
    assert report.preserved == 0, "nothing was refused about our state either"
    assert report.reasons == {}, "no reason is recorded, because none applies"
    # THE COUNTER THAT SEES IT.
    assert count_unclaimed_resume_files(out_dir) == 2, (
        "two suspended games are stranded on disk and every pre-existing "
        "counter reports zero loss"
    )
    # And they are whole games, not claim/tmp debris the sweep owns.
    assert len(_state_files(out_dir)) == 2
    assert not list(out_dir.glob("*.claimed"))

    # ⚑ The counter must count STRANDED GAMES, not directory entries. Debris
    # from an interrupted suspend/resume belongs to `sweep_orphan_state_files`,
    # and counting it here would inflate a loss metric with files that are not
    # lost games -- turning the one instrument that sees this loss into one that
    # cries wolf. Added because a mutant returning `glob("*")` instead of
    # `glob(f"*{RESUME_FILE_SUFFIX}")` SURVIVED the assertions above.
    (out_dir / f"deadbeef{RESUME_FILE_SUFFIX}.claimed").write_bytes(b"x")
    (out_dir / f"deadbeef{RESUME_FILE_SUFFIX}.tmp").write_bytes(b"x")
    (out_dir / "unrelated.txt").write_bytes(b"x")
    assert count_unclaimed_resume_files(out_dir) == 2, (
        "claim/tmp debris and foreign files must not be counted as stranded games"
    )




def _resume_ready_session(tmp_path: Path, in_dir: Path, *, hooks: int = 1) -> Any:
    """`_bare_session` plus the fields `_resume_inflight_games` itself reads.

    Deliberately a real `WorkerSession` and the real methods: the whole point of
    these tests is the WIRING between the leftover count, the log line and
    `pending_outcome_stats`, which a hand-rolled stand-in would not exercise.

    ``hooks`` is the settle barrier's expected registration count -- the number
    of selfplay threads this session would run.
    """
    import threading

    session = _bare_session(tmp_path)
    session.resume_dir = in_dir
    session._resume_counts_lock = threading.Lock()
    session._resume_counts = initial_resume_counts()
    session._resume_skip_reasons = {}
    session._resume_compat_fingerprint_active = FINGERPRINT
    session._resume_trial_id_active = TRIAL_ID
    session._resume_hooks_expected = int(hooks)
    session._resume_hooks_done = 0
    session.model_sha = ""
    session.model_step = 0
    return session


def _settled_line(caplog: pytest.LogCaptureFixture) -> dict[str, int]:
    """The `selfplay resume settled:` line, parsed into its named fields.

    Parsing BY NAME rather than by position is what makes an argument-order
    mutation detectable: a swapped `%d` pair still prints a well-formed line,
    and a positional read would happily agree with it.
    """
    lines = [
        r.getMessage() for r in caplog.records
        if "selfplay resume settled:" in r.getMessage()
    ]
    assert len(lines) == 1, f"expected exactly one settled line, got {lines}"
    body = lines[0].split("selfplay resume settled:", 1)[1].split(" dir=", 1)[0]
    return {k: int(v) for k, v in (tok.split("=") for tok in body.split())}


def test_leftover_count_is_taken_once_after_the_last_hook_not_per_hook(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The settle barrier, which is the whole reason this number is trustworthy.

    The hooks run one per selfplay thread against a SHARED directory, so a
    reading taken by any hook but the last counts files another thread is about
    to claim. Here hook 1 resumes one game and leaves two on disk; if the count
    were emitted per hook it would report 2 as leftovers -- and hook 2 then
    claims them both, so the true answer is 0.

    Exactly one settled line must exist, and it must say 0.
    """
    caplog.set_level("INFO", logger="test-worker")
    game = _game_config()
    source = _fresh_state(game, batch_size=4)
    for slot in range(3):
        _fill_slot(source, slot, plies=6)
    source.done_arr[3] = 1
    in_dir = tmp_path / "resume"
    assert _suspend_all(source, in_dir) == 3

    session = _resume_ready_session(tmp_path, in_dir, hooks=2)
    session._resume_inflight_games(_fresh_state(game, batch_size=1, seed=11))
    # Mid-flight the directory really does hold two unclaimed games...
    assert count_unclaimed_resume_files(in_dir) == 2
    assert not [
        r for r in caplog.records if "selfplay resume settled:" in r.getMessage()
    ], "no settled line may be emitted before the last hook has run"

    session._resume_inflight_games(_fresh_state(game, batch_size=4, seed=12))

    fields = _settled_line(caplog)
    assert fields["resumed"] == 3, "all three games were placed, across two hooks"
    assert fields["left_on_disk"] == 0, (
        "nothing was stranded -- a per-hook count would have cried wolf with 2"
    )
    assert count_unclaimed_resume_files(in_dir) == 0, "and the disk agrees"


def test_settled_line_reports_the_stranded_games_from_the_real_resume_dir(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The wiring, end to end, in the regime where the loss is real.

    Supply 3, demand 1, one hook: two games are stranded and no pre-existing
    counter can see them. Four mutations of this method survived the whole suite
    (`tests/test_selfplay_resume.py` plus all of `tests/test_worker*.py`) when
    only `count_unclaimed_resume_files` itself was covered: hard-wiring the
    count to 0, globbing a directory that is not `self.resume_dir`, swapping
    printf arguments, and reverting the emit guard. Each is a way for a live
    worker to report no loss while games rot on disk -- the house defect exactly
    (a value computed, then silently not delivered).

    Values are pinned and ASYMMETRIC so an argument swap cannot reproduce them.
    """
    caplog.set_level("INFO", logger="test-worker")
    game = _game_config()
    source = _fresh_state(game, batch_size=4)
    for slot in range(3):
        _fill_slot(source, slot, plies=6)
    source.done_arr[3] = 1
    in_dir = tmp_path / "resume"
    assert _suspend_all(source, in_dir) == 3

    target = _fresh_state(game, batch_size=1, seed=11)
    session = _resume_ready_session(tmp_path, in_dir, hooks=1)
    session._resume_inflight_games(target)

    fields = _settled_line(caplog)
    assert fields["resumed"] == 1
    assert fields["discarded"] == 0, "nothing was examined-and-rejected"
    assert fields["preserved"] == 0, "nothing was refused about our state either"
    assert fields["left_on_disk"] == 2, (
        "the two stranded games must reach the settled line, counted from the "
        "session's real resume_dir"
    )
    # The loss must also leave the worker's log file: a log line is not part of
    # the experiment metric stream, so a suspend-vs-resume reconciliation that
    # reads only result.json would still see an unexplained gap.
    assert target.pending_outcome_stats["resume_left_on_disk"] == 2
    assert target.pending_outcome_stats["resume_preserved_games"] == 0
    # And it is loud, because 0.95% of a restart's in-flight games went missing
    # the last time this happened and nothing said so.
    assert [
        r for r in caplog.records
        if r.levelname == "WARNING" and "never claimed" in r.getMessage()
    ], "an unexplained leftover must warn, not just appear in an INFO line"


def test_left_on_disk_counts_preserved_files_and_does_not_warn_about_them(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """⚑ `left_on_disk` is NOT a stranded count, and must never be read as one.

    A file refused for a `_PRESERVE_FILE_REASONS` reason is renamed BACK into
    the directory on purpose, so it still matches the glob. Demand here is ample
    (4 slots for 3 games), so NOTHING is stranded -- and `left_on_disk` still
    reads 3. `no_trial_id` is documented as routine, which puts the
    maximum-magnitude false alarm in the most ordinary state there is.

    Hence `preserved` on the same line, and hence the warning keying on
    `left_on_disk > preserved` rather than on `left_on_disk` alone.
    """
    caplog.set_level("INFO", logger="test-worker")
    game = _game_config()
    source = _fresh_state(game, batch_size=4)
    for slot in range(3):
        _fill_slot(source, slot, plies=6)
    source.done_arr[3] = 1
    in_dir = tmp_path / "resume"
    assert _suspend_all(source, in_dir) == 3

    session = _resume_ready_session(tmp_path, in_dir, hooks=1)
    session._resume_trial_id_active = "some-other-trial"
    session._resume_inflight_games(_fresh_state(game, batch_size=4, seed=11))

    fields = _settled_line(caplog)
    assert fields["resumed"] == 0
    assert fields["discarded"] == 0, "a preserved file is refused, not discarded"
    assert fields["preserved"] == 3, (
        "the preserved count must appear on the line, or left_on_disk=3 reads "
        "as three lost games when nothing was lost"
    )
    assert fields["left_on_disk"] == 3, (
        "preserved files are renamed back into the directory and DO match the "
        "glob -- documented behaviour, not a bug in the counter"
    )
    assert not [
        r for r in caplog.records
        if r.levelname == "WARNING" and "never claimed" in r.getMessage()
    ], "routine preservation must not raise the lost-games alarm"


def test_a_raising_resume_still_settles_the_leftover_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A hook that raised still consumed one of the registrations.

    If the failure path returned before touching the barrier, a session whose
    restore blew up would never emit a settled line at all -- and a failed
    restore is exactly when files pile up. The worst case would be the one case
    with no count.
    """
    caplog.set_level("INFO", logger="test-worker")
    game = _game_config()
    source = _fresh_state(game, batch_size=4)
    for slot in range(3):
        _fill_slot(source, slot, plies=6)
    source.done_arr[3] = 1
    in_dir = tmp_path / "resume"
    assert _suspend_all(source, in_dir) == 3

    session = _resume_ready_session(tmp_path, in_dir, hooks=1)
    # A state the resume cannot work with: decode/placement raises inside
    # resume_inflight_games, which the hook swallows and logs.
    session._resume_inflight_games(cast(Any, object()))

    fields = _settled_line(caplog)
    assert fields["left_on_disk"] == 3, (
        "every suspended game is still on disk, and the settled line must say so "
        "even though the restore raised"
    )


@pytest.mark.parametrize(
    ("threaded", "games", "expected"),
    [(False, 64, 1), (True, 64, None), (True, 1, None)],
)
def test_settle_barrier_is_sized_from_the_path_the_session_actually_takes(
    tmp_path: Path, threaded: bool, games: int, expected: int | None,
) -> None:
    """The barrier's denominator, read on the production path.

    The tests above set `_resume_hooks_expected` directly, so a defect in the
    line that COMPUTES it would not show up there -- the settled count would
    simply never fire (denominator too high) or fire on a mid-flight reading
    (too low), and every assertion above would still pass. That is the house
    defect shape, so the arithmetic is read here rather than presence-checked.
    """
    from types import SimpleNamespace

    session = _bare_session(tmp_path)
    session.args = cast(Any, SimpleNamespace(
        threaded_selfplay=threaded, selfplay_threads=32,
    ))
    got = session._resume_hooks_for_session(games)

    if expected is not None:
        assert got == expected, "the single path builds exactly one state"
    else:
        assert got == session._selfplay_state_count(games), (
            "the threaded path must expect one hook per selfplay thread"
        )
    assert got >= 1, "a zero denominator would settle on the very first hook"


def test_the_barrier_denominator_equals_the_hooks_the_threaded_run_fires(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The barrier's denominator against the hook count the session REALLY fires.

    ⚑ The test above cannot see this. On the threaded branch its assertion is
    `got == session._selfplay_state_count(games)` -- and `_resume_hooks_for_session`
    RETURNS `self._selfplay_state_count(games)`, so both sides move together for
    any mutation of the shared helper, and neither side moves at all for a
    mutation of `_run_selfplay_threaded`'s `n_threads`. MEASURED: rewriting
    `n_threads` to `state_count - 1` leaves that test GREEN. It catches a changed
    RETURN (sizing off `games_per_batch` fails it) and nothing else. The number
    that has to match is not `_selfplay_state_count`, it is how many times
    `on_state_ready` actually runs -- decided in a different method. Two methods,
    one number, no test joining them.

    Drift there is silent in the worst direction. Too high and `_settle_resume_leftovers`
    never reaches its denominator: no settled line, no `left_on_disk > preserved`
    warning, and no `resume_left_on_disk` in `pending_outcome_stats` -- a worker
    stranding games reports exactly what a healthy one reports. Too low and it
    settles on a mid-flight reading and reports a number about a race. Both are
    this repo's signature defect: computed, accepted, then silently not delivered.

    So this drives the REAL `_dispatch_selfplay_one_shard` into the REAL
    `_run_selfplay_threaded` with `play_batch` faked, counts the registrations
    independently, and demands the settle fire exactly once on that count.
    """
    import threading
    from types import SimpleNamespace

    import chess_anti_engine.worker as worker_mod

    caplog.set_level("INFO", logger="test-worker")
    in_dir = tmp_path / "resume"
    in_dir.mkdir()
    # Two stranded files so the settled line is emitted at all (it is guarded on
    # a nonzero count) and so `left_on_disk` is asymmetric to the hook count.
    for i in range(2):
        (in_dir / f"g{i}{RESUME_FILE_SUFFIX}").write_bytes(b"x")

    registrations: list[Any] = []
    reg_lock = threading.Lock()

    def _fake_play_batch(_model: Any, **kwargs: Any) -> tuple[list[Any], Any]:
        state = SimpleNamespace(pending_outcome_stats={})
        with reg_lock:
            registrations.append(state)
        kwargs["on_state_ready"](state)
        return [], "stats"

    monkeypatch.setattr(worker_mod, "play_batch", _fake_play_batch)

    session = _resume_ready_session(tmp_path, in_dir, hooks=1)
    # Deliberately left at the constructor default of 1: the dispatch must SIZE
    # the barrier itself. Seeding it here would hide the very line under test.
    session.args = cast(Any, SimpleNamespace(
        threaded_selfplay=True, selfplay_threads=4,
    ))
    session.rng = np.random.default_rng(0)
    session.device = "cpu"
    session.model = None
    session.sf = object()
    session.inference_client = object()
    session._direct_evaluator = None
    session._resume_inflight_enabled = True
    session._suspend_inflight_games = None
    session._stop_fn = None
    session._pause_fn = None
    session._on_completed_game = None
    session._record_selfplay_phase_timing = None
    session._check_model_update = None
    session._live_states_lock = threading.Lock()
    session._live_states = []
    session._pending_live_override = None
    session._aggregate_thread_stats = lambda stats: stats
    session._build_shared_diff_focus_norm = lambda cfgs, gpb: None

    # 16 games over 4 threads: the state count is the THREAD count here, so a
    # barrier sized off `games_per_batch` would be 16 and never settle.
    session._dispatch_selfplay_one_shard(
        games_per_batch=16, cfgs={}, need_local_model=False,
    )

    assert len(registrations) == 4, (
        "the threaded run must fire one on_state_ready per selfplay thread"
    )
    assert session._resume_hooks_expected == len(registrations), (
        "the settle barrier's denominator and the hooks the session really "
        "fires are computed in two different methods and must agree"
    )
    fields = _settled_line(caplog)  # asserts EXACTLY one settled line
    assert fields["hooks"] == len(registrations)
    assert fields["left_on_disk"] == 2, "counted from the session's real resume_dir"
    # Taken by the LAST hook: an earlier reading describes a race, and only the
    # state that settled carries the number out to result.json.
    settled = [s for s in registrations if s.pending_outcome_stats]
    assert len(settled) == 1, "the leftover count must be published exactly once"
    assert settled[0].pending_outcome_stats["resume_left_on_disk"] == 2
