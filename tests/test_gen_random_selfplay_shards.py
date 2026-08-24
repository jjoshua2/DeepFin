"""scripts/gen_random_selfplay_shards.py — the CPU generation-zero generator.

The load-bearing claims, and how each is tested rather than asserted:

* the shards are accepted by the REAL replay load path (``DiskReplayBuffer``),
  not by a hand-rolled reader;
* ``wdl_target``'s POV is the side to move AT THAT PLY, checked on a game whose
  moves are forced end to end so the expected labels come off the board;
* the CBoard/python-chess lockstep guard does not false-positive on en passant,
  where the two encoders disagree by convention in BOTH directions;
* the stored ``policy_target`` really is the search's improved policy — the
  sharp-row test below FAILS against a uniform-over-legal emitter, which is the
  mutant a support-size assertion could not see;
* every flag the operator can set reaches the game loop, checked by making the
  search's own behaviour depend on it (mutation table in the PR).
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.mcts.puct import _value_scalar_from_wdl_logits
from chess_anti_engine.encoding.plane_decode import decode_step0_bitboards
from chess_anti_engine.moves import (
    COMPACT_POLICY_SIZE,
    index_to_move_for_encoding,
    move_to_index,
)
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays
from chess_anti_engine.selfplay.game import _result_to_wdl
from chess_anti_engine.selfplay.opening import OpeningConfig

from tests.script_loading import load_script_module

GEN = load_script_module("gen_random_selfplay_shards.py")

_PLANES = 175
_HIST = "lc0_root_legacy_meta"
_EXTRA = "v2_threats"


@pytest.fixture(autouse=True)
def _rep_fix_on() -> Iterator[None]:
    """The generator encodes with the production repetition fix on."""
    previous = rep_fix.current()
    rep_fix.apply(True, boards_discarded=True)
    yield
    if previous is not None and bool(previous) is not True:
        rep_fix.apply(bool(previous), boards_discarded=True)


def _config(out_dir: Path, **overrides: Any) -> Any:
    base: dict[str, Any] = {
        "out_dir": out_dir,
        "games": 2,
        "workers": 1,
        "sims": 8,
        "max_plies": 40,
        "shard_size": 40,
        "seed": 4242,
        "nice": 0,
        "input_history_encoding": _HIST,
        "input_extra_features": _EXTRA,
        "history_rep_fix": True,
    }
    base.update(overrides)
    return GEN.GenConfig(**base)


def _run(cfg: Any, *, worker_id: int = 0, games: int | None = None) -> Any:
    spec = GEN.WorkerSpec(
        cfg=cfg,
        worker_id=worker_id,
        games=int(cfg.games) if games is None else int(games),
        seed=int(cfg.seed),
        shard_index_start=0,
    )
    return GEN.run_worker(spec)


def _play_one(cfg: Any, *, seed: int = 7) -> tuple[Any, Any]:
    gcfg = GEN.build_gumbel_config(cfg)
    evaluator = GEN.UniformPriorEvaluator(
        value_source=cfg.value_source,
        expected_planes=_PLANES,
        random_salt=seed,
    )
    outcome = GEN.play_game(
        cfg=cfg,
        gcfg=gcfg,
        evaluator=evaluator,
        rng=np.random.default_rng(seed),
        opening_cfg=OpeningConfig(),
    )
    return outcome, evaluator


# ── schema parity: the real loader is the judge ──────────────────────────────

def test_generated_shard_loads_through_the_real_replay_buffer(tmp_path: Path) -> None:
    result = _run(_config(tmp_path))
    assert result.shards, "worker wrote no shard"
    paths = iter_shard_paths(tmp_path)
    assert len(paths) == len(result.shards)

    buf = DiskReplayBuffer(
        10**9, shard_dir=tmp_path, rng=np.random.default_rng(0),
        read_only=True, input_planes=_PLANES,
    )
    try:
        # `_try_load_shard` swallows load failures, so a rejected shard shows up
        # as an EMPTY buffer, never as a raise. Count rows, not exceptions.
        assert len(buf) == result.rows > 0
    finally:
        buf.close()


def test_shard_metadata_declares_the_production_encoding_identity(
    tmp_path: Path,
) -> None:
    _run(_config(tmp_path))
    _arrs, meta = load_shard_arrays(iter_shard_paths(tmp_path)[0])
    assert meta["policy_encoding"] == "lc0_1858"
    assert int(meta["policy_size"]) == COMPACT_POLICY_SIZE
    assert meta["input_history_encoding"] == _HIST
    assert bool(meta["history_rep_fix"]) is True


def test_shard_carries_no_stockfish_or_search_value_fields(tmp_path: Path) -> None:
    """`has_sf_* == 0` is how the loader spells absent, and pruning drops them."""
    _run(_config(tmp_path))
    arrs, _meta = load_shard_arrays(iter_shard_paths(tmp_path)[0])
    leaked = [
        name for name in arrs
        if name.startswith(("sf_", "has_sf_")) or name in ("search_wdl", "has_search_wdl")
    ]
    assert leaked == []


def test_x_is_175_v2_threats_planes(tmp_path: Path) -> None:
    _run(_config(tmp_path))
    arrs, _meta = load_shard_arrays(iter_shard_paths(tmp_path)[0])
    x = np.asarray(arrs["x"])
    assert x.ndim == 4
    assert x.shape[1:] == (_PLANES, 8, 8)


def test_policy_target_is_a_normalised_distribution_inside_the_legal_set(
    tmp_path: Path,
) -> None:
    """Shape only. ⚑ This test CANNOT see whether the search contributed.

    An earlier version of it asserted ``support > 1`` and called the row a
    "visit distribution"; both were wrong. Gumbel stores
    ``softmax(log_prior + sigma*Qbar)``, which is dense over the legal moves by
    construction, so a uniform-over-legal emitter that discarded the search
    entirely satisfies every assertion here. The mutant that proves it is killed
    by ``test_search_signal_reaches_the_policy_target`` below, not by this test.
    """
    _run(_config(tmp_path, sims=32))
    arrs, _meta = load_shard_arrays(iter_shard_paths(tmp_path)[0])
    policy = np.asarray(arrs["policy_target"], dtype=np.float64)
    legal = np.asarray(arrs["legal_mask"], dtype=np.uint8)
    assert policy.shape[1] == COMPACT_POLICY_SIZE

    sums = policy.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=2e-3), f"row sums {sums.min()}..{sums.max()}"
    assert (policy >= 0.0).all()
    # Support lives inside the legal set: a move with mass and no legal bit would
    # mean the search and the mask disagree about the position.
    assert not ((policy > 0.0) & (legal == 0)).any()


def test_policy_target_is_near_uniform_by_default_and_that_is_measured(
    tmp_path: Path,
) -> None:
    """The documented gen-0 property, asserted as a RANGE so it cannot rot.

    A uniform prior plus a constant value makes both terms of
    ``softmax(log_prior + sigma*Qbar)`` flat, so the overwhelming majority of
    rows restate the prior. If a future change made the default corpus SHARP,
    that would be knowledge the run does not have, and this fails.
    """
    result = _run(_config(tmp_path, games=4, sims=32, max_plies=60))
    shape = result.policy_shape.summary()
    assert shape["rows"] > 0
    assert shape["uniform_row_frac"] > 0.9, shape
    assert shape["tv_to_uniform_median"] < 0.01, shape
    assert shape["sharp_row_frac"] < 0.2, shape


# ── the search really does reach the target (the uniform-emitter mutant) ─────

# Five mate-in-1 positions with a UNIQUE mating move, three white to move and
# two black to move. Verified against python-chess when this list was written:
# each has 16-23 legal moves and exactly one mate, so a uniform target puts
# ~1/20 on the mating move and the search must put ~1 on it.
_MATE_IN_ONE: tuple[tuple[str, str], ...] = (
    ("7k/5ppp/8/8/8/8/8/R6K w - - 0 1", "a1a8"),
    ("6k1/5ppp/8/8/8/8/8/4Q2K w - - 0 1", "e1e8"),
    ("4q2k/8/8/8/8/8/5PPP/6K1 b - - 0 1", "e8e1"),
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8"),
    ("r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", "a8a1"),
)


def _single_fen_opening(tmp_path: Path, fen: str, name: str) -> Any:
    """One FEN per list, so the game starts where the test says it does.

    A multi-FEN list is SAMPLED, which made an earlier version of the sharp-row
    test score whichever positions the draw happened to land on -- fine for the
    aggregate, useless for "the mass is on the mate at THIS position".
    """
    path = tmp_path / f"{name}.txt"
    path.write_text(fen + "\n")
    return OpeningConfig(opening_fen_list_path=str(path), opening_fen_prob=1.0)


def test_search_signal_reaches_the_policy_target(tmp_path: Path) -> None:
    """⚑ THE test a uniform-over-legal emitter FAILS.

    The default corpus is ~99 % uniform, which is correct at generation zero and
    also means every shape assertion in this file passes against a target that
    threw the search away. This one plays from positions where the search has
    something to find — a forced mate at the root — and asserts BOTH that the
    sharp-row share clears a floor and that the sharp rows put their mass on the
    mating move. Uniform fails on both halves; a target sharp on the WRONG move
    fails on the second.
    """
    cfg = _config(tmp_path, sims=32, max_plies=8)
    shape = GEN.PolicyShapeStats()
    for i, (fen, mate_uci) in enumerate(_MATE_IN_ONE):
        outcome = GEN.play_game(
            cfg=cfg,
            gcfg=GEN.build_gumbel_config(cfg),
            evaluator=GEN.UniformPriorEvaluator(
                value_source="zero", expected_planes=_PLANES,
            ),
            rng=np.random.default_rng(100 + i),
            opening_cfg=_single_fen_opening(tmp_path, fen, f"mate_{i}"),
        )
        rows = GEN.rows_from_game(outcome, cfg=cfg, shape=shape)
        assert outcome.start_fen == fen
        assert outcome.termination == "checkmate"
        board = chess.Board(fen)
        row = rows[0]
        policy = np.asarray(row.policy_target, dtype=np.float64)
        top = index_to_move_for_encoding(int(policy.argmax()), board)
        assert top.uci() == mate_uci, (
            f"{fen}: search put its mass on {top.uci()}, not the mate {mate_uci}"
        )
        assert policy.max() > 0.5, f"{fen}: top mass {policy.max():.3f} is not sharp"
        assert GEN.policy_tv_to_uniform(policy, row.legal_mask) > 0.5

    summary = shape.summary()
    # A uniform emitter scores 0.0 here by construction.
    assert summary["sharp_row_frac"] > 0.2, summary


def test_sharp_rows_are_absent_from_a_uniform_target() -> None:
    """The floor above is a real discriminator, not a threshold anything clears.

    Feeding ``policy_tv_to_uniform`` an exactly-uniform row must score 0, so the
    ``> 0.2`` sharp-row floor is unreachable for the mutant it exists to catch.
    """
    legal = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.uint8)
    legal[[3, 17, 40, 91]] = 1
    uniform = legal.astype(np.float64) / 4.0
    assert GEN.policy_tv_to_uniform(uniform, legal) == pytest.approx(0.0)
    one_hot = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.float64)
    one_hot[17] = 1.0
    assert GEN.policy_tv_to_uniform(one_hot, legal) == pytest.approx(0.75)


# ── the lockstep guard vs en passant (regression) ────────────────────────────

def _push(fen: str, uci: str) -> tuple[Any, chess.Board]:
    board = chess.Board(fen)
    cb = CBoard.from_board(board)
    move = chess.Move.from_uci(uci)
    cb.push_index(move_to_index(move, board))
    board.push(move)
    return cb, board


def test_lockstep_guard_survives_a_mating_double_pawn_push(tmp_path: Path) -> None:
    """⚑ REGRESSION. This fired on every game that ended in a double pawn push.

    ``chess.Board.fen()`` omits the ep square unless a LEGAL ep capture exists,
    and a mate leaves no legal moves at all — so python-chess prints ``-`` where
    CBoard prints the square, and a verbatim field-4 comparison raises on two
    boards that agree about everything, ``result()`` included.
    """
    cb, board = _push("8/8/8/8/K3p3/3Q4/3Q1Pk1/3R4 w - - 0 1", "f2f4")
    assert board.is_checkmate()
    assert cb.result() == "1-0"
    assert board.result() == "1-0"
    assert cb.fen().split()[3] == "f3"
    assert board.fen().split()[3] == "-"          # the trap
    assert board.fen(en_passant="fen").split()[3] == "f3"
    assert GEN.boards_agree(cb, board)

    # And the other direction: a double push with NO adjacent enemy pawn, where
    # CBoard prints "-" and `en_passant="fen"` prints the square. Comparing
    # against `en_passant="fen"` verbatim would false-positive on every game that
    # ends on an ordinary double push.
    cb2, board2 = _push("7k/5ppp/8/8/8/8/1P6/K6R w - - 0 1", "b2b4")
    assert cb2.fen().split()[3] == "-"
    assert board2.fen(en_passant="fen").split()[3] == "b3"
    assert GEN.boards_agree(cb2, board2)

    # A pinned capturer is the same trap without a mate.
    cb3, board3 = _push("8/8/8/8/R3p2k/8/3P4/4K3 w - - 0 1", "d2d4")
    assert not board3.is_game_over()
    assert cb3.fen().split()[3] == "d3"
    assert board3.fen().split()[3] == "-"
    assert GEN.boards_agree(cb3, board3)
    del tmp_path


def test_lockstep_guard_still_catches_a_real_divergence() -> None:
    """Relaxing the ep field must not turn the guard into a tautology."""
    board = chess.Board()
    cb = CBoard.from_board(board)
    cb.push_index(move_to_index(chess.Move.from_uci("e2e4"), board))
    board.push(chess.Move.from_uci("d2d4"))
    assert not GEN.boards_agree(cb, board)
    # Same placement, different side to move.
    other = chess.Board()
    other_cb = CBoard.from_board(other)
    other.push(chess.Move.from_uci("e2e4"))
    other_cb.push_index(move_to_index(chess.Move.from_uci("e2e4"), chess.Board()))
    other.push(chess.Move.from_uci("e7e5"))
    assert not GEN.boards_agree(other_cb, other)


def test_a_game_ending_on_a_double_pawn_push_produces_rows(tmp_path: Path) -> None:
    """End to end through play_game, which is where the guard actually runs."""
    fen_list = tmp_path / "mate_by_push.txt"
    fen_list.write_text("8/8/8/8/K3p3/3Q4/3Q1Pk1/3R4 w - - 0 1\n")
    cfg = _config(tmp_path, sims=16, max_plies=6)
    script_board = chess.Board("8/8/8/8/K3p3/3Q4/3Q1Pk1/3R4 w - - 0 1")
    x = encode_cboard(
        CBoard.from_board(script_board),
        input_history_encoding=_HIST, input_extra_features=_EXTRA,
    )
    script = {
        _position_signature(x): move_to_index(
            chess.Move.from_uci("f2f4"), script_board,
        ),
    }
    outcome = GEN.play_game(
        cfg=cfg,
        gcfg=GEN.build_gumbel_config(cfg),
        evaluator=_ScriptedPriorEvaluator(
            script, value_source="zero", expected_planes=_PLANES,
        ),
        rng=np.random.default_rng(0),
        opening_cfg=OpeningConfig(
            opening_fen_list_path=str(fen_list), opening_fen_prob=1.0,
        ),
    )
    assert outcome.plies == 1
    assert outcome.result == "1-0"
    assert outcome.termination == "checkmate"
    rows = GEN.rows_from_game(outcome, cfg=cfg)
    assert [int(row.wdl_target) for row in rows] == [0]  # White moved and won


# ── the wdl POV, on a game whose every move is forced ────────────────────────

def _position_signature(x: np.ndarray) -> bytes:
    """Key a scripted prior on the position, not on the call order."""
    bitboards = decode_step0_bitboards(np.asarray(x)[None, ...])
    return np.ascontiguousarray(bitboards[0]).tobytes()


class _ScriptedPriorEvaluator(GEN.UniformPriorEvaluator):
    """Uniform prior everywhere except the scripted positions, forced there.

    The prior is a softmax over the legal moves, and sequential halving scores a
    candidate by ``gumbel + log_prior + sigma*Q``. A +1000 logit puts the scripted
    move ~1000 nats above every alternative while ``|sigma*Q| <= c_scale *
    (c_visit + sims)``, so the survivor is the scripted move with no reliance on
    the value head, the noise draw, or the seed.
    """

    def __init__(self, script: dict[bytes, int], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.script = dict(script)

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        policy, wdl = super().evaluate_encoded(x, relations)
        arr = np.asarray(x)
        for i in range(arr.shape[0]):
            action = self.script.get(_position_signature(arr[i]))
            if action is not None:
                policy[i, action] = 1000.0
        return policy, wdl


def _forced_line_script(ucis: tuple[str, ...]) -> dict[bytes, int]:
    """Key a scripted prior on each position of a forced line."""
    board = chess.Board()
    script: dict[bytes, int] = {}
    for uci in ucis:
        x = encode_cboard(
            CBoard.from_board(board),
            input_history_encoding=_HIST,
            input_extra_features=_EXTRA,
        )
        move = chess.Move.from_uci(uci)
        script[_position_signature(x)] = move_to_index(move, board)
        board.push(move)
    assert board.is_checkmate()
    return script


# 1. f3 e5 2. g4 Qh4# — Black mates on ply 3; White is to move on plies 0 and 2.
_FOOLS_MATE = ("f2f3", "e7e5", "g2g4", "d8h4")
# 1. e4 g5 2. Nc3 f5 3. Qh5# — WHITE mates on ply 4, and the ply count is ODD, so
# a POV bug that happens to cancel on an even-length game cannot hide here.
_SCHOLARS_STYLE = ("e2e4", "g7g5", "b1c3", "f7f5", "d1h5")


def _play_forced(cfg: Any, ucis: tuple[str, ...], *, seed: int) -> Any:
    return GEN.play_game(
        cfg=cfg,
        gcfg=GEN.build_gumbel_config(cfg),
        evaluator=_ScriptedPriorEvaluator(
            _forced_line_script(ucis), value_source="zero", expected_planes=_PLANES,
        ),
        rng=np.random.default_rng(seed),
        opening_cfg=OpeningConfig(),
    )


def test_wdl_target_pov_is_the_side_to_move_at_that_ply(tmp_path: Path) -> None:
    cfg = _config(tmp_path, sims=16, max_plies=20)
    outcome = _play_forced(cfg, _FOOLS_MATE, seed=1)
    assert outcome.plies == 4
    assert outcome.result == "0-1"
    assert outcome.termination == "checkmate"
    assert [rec.pov_white for rec in outcome.records] == [True, False, True, False]

    # The 0/1/2 CODE meaning is measured off the shipped production util, not
    # restated here (selfplay/game.py::_result_to_wdl, "0=W,1=D,2=L from
    # side-to-move perspective at that position"); it is also pinned
    # independently by tests/test_selfplay_result_labeling.py.
    assert _result_to_wdl("0-1", pov_white=True) == 2
    assert _result_to_wdl("0-1", pov_white=False) == 0

    # The POV half comes off the BOARD: Black delivered mate, plies 0 and 2 are
    # White's, so those rows are losses and Black's rows are wins.
    rows = GEN.rows_from_game(outcome, cfg=cfg)
    assert [int(row.wdl_target) for row in rows] == [2, 0, 2, 0]


def test_wdl_target_pov_on_the_mirrored_white_win(tmp_path: Path) -> None:
    """The same claim with the colours swapped and an ODD ply count.

    One decisive game only ever exercises one sign of the POV rule, and a
    four-ply game has an even split — so a mirrored, odd-length win is where a
    half-fixed POV bug shows up.
    """
    cfg = _config(tmp_path, sims=16, max_plies=20)
    outcome = _play_forced(cfg, _SCHOLARS_STYLE, seed=2)
    assert outcome.plies == 5
    assert outcome.result == "1-0"
    assert outcome.termination == "checkmate"
    assert [rec.pov_white for rec in outcome.records] == [
        True, False, True, False, True,
    ]
    assert _result_to_wdl("1-0", pov_white=True) == 0
    assert _result_to_wdl("1-0", pov_white=False) == 2

    rows = GEN.rows_from_game(outcome, cfg=cfg)
    assert [int(row.wdl_target) for row in rows] == [0, 2, 0, 2, 0]


def test_truncated_games_are_labelled_draws_not_losses(tmp_path: Path) -> None:
    """`"*"` is what a max-plies game returns; production adjudicates it with
    Stockfish and there is none here, so it must land on the draw label.

    ⚑ ``--max-plies 2`` from the start position, not a seed that happens to run
    long: the earlier version skipped when its seed finished early, i.e. it could
    silently skip the one scenario it exists to cover. No chess game ends in two
    plies, so the truncation is structural.
    """
    cfg = _config(tmp_path, max_plies=2, sims=8)
    outcome, _ev = _play_one(cfg, seed=3)
    assert outcome.termination == "max_plies"
    assert outcome.plies == 2
    assert outcome.result == "*"
    rows = GEN.rows_from_game(outcome, cfg=cfg)
    assert {int(row.wdl_target) for row in rows} == {1}


# ── the value mapping is exact, not approximate ──────────────────────────────

def test_value_mapping_round_trips_through_the_production_converter() -> None:
    """`q -> logits -> q` through the search's own WDL reader."""
    q = np.array([-1.0, -0.73, -0.1, 0.0, 0.1, 0.5, 0.999], dtype=np.float64)
    logits = GEN.q_to_wdl_logits(q)
    assert logits.shape == (q.size, 3)
    back = np.array([_value_scalar_from_wdl_logits(row) for row in logits])
    assert np.allclose(back, q, atol=1e-6)


def test_material_q_is_side_to_move_pov() -> None:
    def q_of(fen: str) -> float:
        x = encode_cboard(
            CBoard.from_board(chess.Board(fen)),
            input_history_encoding=_HIST,
            input_extra_features=_EXTRA,
        )
        return float(GEN.material_q(x[None, ...])[0])

    assert q_of(chess.STARTING_FEN) == pytest.approx(0.0)
    # Black is a queen down. Same position, both sides to move: the sign flips.
    white_up = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    black_to_move = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    assert q_of(white_up) == pytest.approx(np.tanh(900.0 / 400.0))
    assert q_of(black_to_move) == pytest.approx(-np.tanh(900.0 / 400.0))


# ── the flags reach the game loop ────────────────────────────────────────────

def test_realized_config_is_read_off_the_objects_that_consume_it(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path, sims=11, topk=5, c_scale=0.037, gumbel_scale=0.4)
    gcfg = GEN.build_gumbel_config(cfg)
    evaluator = GEN.UniformPriorEvaluator(
        value_source="material", expected_planes=_PLANES, random_salt=3,
    )
    realized = GEN.realized_config(
        gcfg=gcfg, evaluator=evaluator, opening_cfg=GEN.build_opening_config(cfg),
        cfg=cfg, worker_id=0,
    )
    assert realized["simulations"] == 11
    assert realized["topk"] == 5
    assert realized["c_scale"] == pytest.approx(0.037)
    assert realized["gumbel_scale"] == pytest.approx(0.4)
    assert realized["value_source"] == "material"
    assert realized["input_planes"] == _PLANES
    assert realized["policy_width"] == COMPACT_POLICY_SIZE
    assert realized["history_rep_fix"] is True
    assert realized["opening_book_max_games"] == cfg.opening_max_games
    # The two target-decoupling knobs and the two non-GumbelConfig C-search
    # arguments: all four are live production values and all four must appear.
    assert realized["target_max_visit_cap"] == GEN.DEFAULT_TARGET_MAX_VISIT_CAP
    assert realized["target_untempered_prior"] is True
    assert realized["vloss_weight"] == GEN.DEFAULT_VLOSS_WEIGHT == 1
    assert realized["target_batch"] == GEN.DEFAULT_TARGET_BATCH
    assert "sims" not in realized, "realized must not echo the parser's names"


def test_the_fen_seeding_branch_is_unreachable_and_unannounced(
    tmp_path: Path,
) -> None:
    """Generation zero cannot seed from blind-spot FENs, and does not claim to.

    Two halves, and the second is the one with teeth. FIRST: no CLI-reachable
    combination populates the FEN-list branch of the shared ``OpeningConfig``,
    so a run of this tool is pure selfplay by construction rather than by
    convention -- seeded openings would be curriculum data inside the arm whose
    whole point is having none. SECOND: because those fields cannot move, they
    are absent from the realized line. A constant there is not extra
    provenance; it is the mirror of the defect the line exists to catch, and it
    would weaken the only property the line claims -- that every entry in it
    moved because something asked it to.

    Wiring a real flag for either field fails BOTH halves here and the
    literal-surface pin in ``tests/test_deletion_annotations.py``, which is
    where a consumer of a key the production config no longer sets belongs.
    """
    for cfg in (
        _config(tmp_path),
        _config(tmp_path, random_start_plies=6),
        _config(tmp_path, openings=None),
    ):
        opening_cfg = GEN.build_opening_config(cfg)
        assert opening_cfg.opening_fen_list_path is None
        assert float(opening_cfg.opening_fen_prob) == 0.0
        realized = GEN.realized_config(
            gcfg=GEN.build_gumbel_config(cfg),
            evaluator=GEN.UniformPriorEvaluator(
                value_source="zero", expected_planes=_PLANES,
            ),
            opening_cfg=opening_cfg, cfg=cfg, worker_id=0,
        )
        assert not [k for k in realized if k.startswith("opening_fen")]
        # Not just "the fen keys are gone": the reachable opening source is
        # still announced, so this is a scoped absence and not an empty line.
        assert "opening_book_prob" in realized
        assert "random_start_plies" in realized


def test_search_defaults_match_the_live_production_yaml() -> None:
    """Pin the four provenance numbers an earlier revision took from `main`.

    `main`'s copy of the config is NOT what production runs (~55 keys differ) and
    reading it produced four wrong claims at once. These are the LIVE values;
    changing one means the live yaml changed, which is a thing to notice.
    """
    assert GEN.DEFAULT_TOPK == 16                    # selfplay.gumbel_topk
    assert GEN.DEFAULT_POLICY_TEMP == 1.5            # selfplay.gumbel_policy_temp
    assert GEN.DEFAULT_GUMBEL_SCALE == 1.0           # selfplay.gumbel_scale (pre-decay)
    assert GEN.DEFAULT_TARGET_MAX_VISIT_CAP == 5     # gumbel_target_max_visit_cap
    assert GEN.DEFAULT_TARGET_UNTEMPERED_PRIOR is True
    assert GEN.DEFAULT_VLOSS_WEIGHT == 1             # selfplay.gumbel_vloss_weight
    assert GEN.DEFAULT_TARGET_BATCH == 0
    assert GEN.DEFAULT_MAX_PLIES == 450              # selfplay.max_plies
    assert GEN.DEFAULT_SHARD_SIZE == 2000            # distributed.shard_size
    assert GEN.DEFAULT_TEMPERATURE == 0.0            # selfplay.selfplay_temperature


def test_policy_temp_is_inert_at_a_uniform_prior(tmp_path: Path) -> None:
    """The stated reason the default is production's 1.5 and not 1.0.

    Tempering divides the policy logits, and the stub's are all zero, so every
    finite temperature leaves the prior exactly uniform. Asserted on the GAME,
    not on the arithmetic: same seed, two temperatures, identical move trace.
    """
    at_default, _ = _play_one(_config(tmp_path, max_plies=40), seed=17)
    at_one, _ = _play_one(_config(tmp_path, policy_temp=1.0, max_plies=40), seed=17)
    assert at_default.move_trace == at_one.move_trace
    assert GEN.DEFAULT_POLICY_TEMP != 1.0, "the claim is only interesting if they differ"


def test_sims_reaches_the_search(tmp_path: Path) -> None:
    """The search must ASK the evaluator for more positions at a bigger budget.

    Rows per ply, not rows per game: a different budget plays a different game,
    so a raw total would move even if the budget were being dropped.

    ⚑ A LADDER, not two adjacent points. The C path batches leaves and sequential
    halving allocates `budget // (candidates * ceil(log2 candidates))`, so the
    cost is monotone in the budget but PLATEAUS -- measured on this fixture,
    sims 4 and 8 both cost 33.0 rows/ply, and 32 and 64 both cost 65.0. A fixed
    ratio between one adjacent pair would therefore be a flaky threshold rather
    than a control. Monotone across the ladder plus a wide-endpoint gap is the
    honest form, and the mutant that ignores the flag gives a FLAT ladder.
    """
    ladder = [2, 8, 32, 128]
    per_ply: list[float] = []
    for sims in ladder:
        outcome, evaluator = _play_one(
            _config(tmp_path, sims=sims, max_plies=24), seed=5,
        )
        per_ply.append(evaluator.eval_rows / max(1, outcome.plies))
    assert per_ply == sorted(per_ply), f"cost is not monotone in sims: {per_ply}"
    assert per_ply[-1] > 3.0 * per_ply[0], (
        f"sims did not reach the search: {ladder} -> {per_ply} evaluated "
        "positions per ply"
    )


class _SkewedPriorEvaluator(GEN.UniformPriorEvaluator):
    """A NON-uniform prior, so the target-decoupling knobs have something to do.

    Both ``target_max_visit_cap`` and ``target_untempered_prior`` reshape the
    stored ``softmax(log_prior + sigma*Qbar)``. At generation zero both terms are
    flat, so they are inert BY DESIGN — which is why proving they are wired needs
    a prior that varies. A monotone ramp over the move indices is enough.
    """

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        policy, wdl = super().evaluate_encoded(x, relations)
        ramp = np.linspace(-2.0, 2.0, policy.shape[1], dtype=np.float32)
        return policy + ramp[None, :], wdl


def _first_target(cfg: Any, *, seed: int = 3) -> np.ndarray:
    outcome = GEN.play_game(
        cfg=cfg,
        gcfg=GEN.build_gumbel_config(cfg),
        evaluator=_SkewedPriorEvaluator(
            value_source="material", expected_planes=_PLANES,
        ),
        rng=np.random.default_rng(seed),
        opening_cfg=OpeningConfig(),
    )
    return np.asarray(
        GEN.rows_from_game(outcome, cfg=cfg)[0].policy_target, dtype=np.float64,
    )


def test_target_decoupling_knobs_reach_the_stored_target(tmp_path: Path) -> None:
    """The two knobs pinned to live production values are really wired.

    They are inert at generation zero on purpose, so "the games are identical"
    proves nothing about them. Under a non-uniform prior each must move the
    STORED target — which is the only thing either one touches (the played move
    keeps the uncapped sigma and the tempered prior, so no arena could see this).
    """
    base_cfg = _config(tmp_path, sims=64, max_plies=1)
    base = _first_target(base_cfg)
    for override in (
        {"target_untempered_prior": False},
        {"target_max_visit_cap": 0},
        {"target_max_visit_cap": 1},
    ):
        other = _first_target(_config(tmp_path, sims=64, max_plies=1, **override))
        assert np.abs(other - base).sum() > 1e-3, (
            f"{override} did not change the stored target"
        )


def test_vloss_weight_reaches_the_c_search(tmp_path: Path) -> None:
    """⚑ MEASURED, and the scope matters: it is INERT at this tool's default.

    ``gumbel_vloss_weight`` spreads a batch across distinct leaves. At
    ``--sims 32`` each halving round allocates at most one visit per candidate,
    so there is no in-flight duplicate to spread and 0 / 1 / 3 give a
    byte-identical search — reporting it as "wired" off a run at the default
    would be a clean null. At 256 sims it is a positive control, which is what
    proves the argument reaches the C tree at all.

    It is passed explicitly regardless, because mcts/gumbel.py's standing comment
    requires it at every call site and the call shape is not fixed forever.
    """
    def cost(sims: int, vloss: int) -> int:
        _outcome, evaluator = _play_one(
            _config(tmp_path, sims=sims, max_plies=12, vloss_weight=vloss), seed=5,
        )
        return int(evaluator.eval_rows)

    assert cost(32, 0) == cost(32, 1) == cost(32, 3), "expected inert at 32 sims"
    assert cost(256, 1) != cost(256, 0), "vloss_weight never reached the C search"
    assert cost(256, 3) == cost(256, 1)


def test_target_batch_reaches_the_c_search(tmp_path: Path) -> None:
    """The other non-GumbelConfig argument, live at the default sim count."""
    def cost(batch: int) -> int:
        _outcome, evaluator = _play_one(
            _config(tmp_path, sims=32, max_plies=12, target_batch=batch), seed=5,
        )
        return int(evaluator.eval_rows)

    assert cost(1) > cost(0), "target_batch never reached the C search"


def test_value_source_reaches_the_game_loop(tmp_path: Path) -> None:
    """Same seed, same search, different value: the games must diverge."""
    zero, _ = _play_one(_config(tmp_path, value_source="zero", max_plies=60), seed=9)
    material, _ = _play_one(
        _config(tmp_path, value_source="material", max_plies=60), seed=9,
    )
    assert zero.move_trace != material.move_trace
    random_game, _ = _play_one(
        _config(tmp_path, value_source="random", max_plies=60), seed=9,
    )
    assert random_game.move_trace not in (zero.move_trace, material.move_trace)


def test_out_of_band_search_knobs_are_refused_not_recorded(tmp_path: Path) -> None:
    """A knob the search would ignore must not reach the realized line.

    `policy_temp_active` reads an out-of-band temperature as OFF (it runs per
    leaf and cannot raise), so without the construction-boundary check the run
    would be banked under a setting it never used.
    """
    with pytest.raises(ValueError, match="policy_temp"):
        GEN.build_gumbel_config(_config(tmp_path, policy_temp=1e300))
    with pytest.raises(ValueError, match="topk"):
        GEN.build_gumbel_config(_config(tmp_path, topk=1))
    # The in-band defaults still build, so the guard is not just "always raise".
    assert GEN.build_gumbel_config(_config(tmp_path)).policy_temp == pytest.approx(
        GEN.DEFAULT_POLICY_TEMP,
    )


def test_extra_features_flag_reaches_the_encoder(tmp_path: Path) -> None:
    """A stub built for the wrong plane count must be refused, not evaluated."""
    cfg = _config(tmp_path, input_extra_features="v1")
    evaluator = GEN.UniformPriorEvaluator(
        value_source="zero", expected_planes=_PLANES,
    )
    with pytest.raises(ValueError, match="expected"):
        GEN.play_game(
            cfg=cfg, gcfg=GEN.build_gumbel_config(cfg), evaluator=evaluator,
            rng=np.random.default_rng(0), opening_cfg=OpeningConfig(),
        )


def test_history_planes_are_populated_not_repeat_filled(tmp_path: Path) -> None:
    """The stored planes carry real history, not eight copies of the position.

    A plane-count check proves the WIDTH reached the encoder and nothing about
    the CONTENT: an encoder that filled every history slot with the current
    position would pass it. This reads a ply>=1 row back off disk and requires
    history block 1 to be non-zero AND different from block 0, which is only
    true if the CBoard carried its move stack across the push.
    """
    _run(_config(tmp_path, games=1, sims=8, max_plies=12, shard_size=500))
    arrs, _meta = load_shard_arrays(iter_shard_paths(tmp_path)[0])
    x = np.asarray(arrs["x"], dtype=np.float32)
    plies = np.asarray(arrs["ply_index"])
    later = int(np.argmax(plies >= 1))
    assert int(plies[later]) >= 1, "no ply>=1 row was written"

    # lc0_root layout: 13 planes per history step, 12 of them piece planes.
    block0, block1 = x[later, 0:12], x[later, 13:25]
    assert block0.any(), "current-position planes are empty"
    assert block1.any(), "history slot 1 is empty — no history was encoded"
    assert not np.array_equal(block0, block1), (
        "history slot 1 is a copy of slot 0 — the planes are repeat-filled"
    )
    # And the ply-0 row of the same game has an EMPTY slot 1 (nothing precedes
    # the start position), so the difference above is history, not noise.
    first = int(np.argmin(plies))
    assert not x[first, 13:25].any()


# ── determinism and append safety ────────────────────────────────────────────

def test_same_seed_reproduces_the_first_shard_digest(tmp_path: Path) -> None:
    first = _run(_config(tmp_path / "a"))
    second = _run(_config(tmp_path / "b"))
    assert [s["digest"] for s in first.shards] == [s["digest"] for s in second.shards]
    assert first.rows == second.rows
    assert first.plies == second.plies

    other = _run(_config(tmp_path / "c", seed=999))
    assert other.shards[0]["digest"] != first.shards[0]["digest"]


def test_appending_skips_existing_shard_indices(tmp_path: Path) -> None:
    first = GEN.generate(_config(tmp_path, games=1))
    assert first["shard_index_start"] == 0
    second = GEN.generate(_config(tmp_path, games=1, seed=77))
    assert second["shard_index_start"] == len(first["shards"])
    written = {int(s["index"]) for s in first["shards"]} | {
        int(s["index"]) for s in second["shards"]
    }
    assert len(written) == len(first["shards"]) + len(second["shards"])
    assert len(iter_shard_paths(tmp_path)) == len(written)
    # The first batch is untouched by the second.
    assert GEN.shard_digest(tmp_path / first["shards"][0]["path"]) == (
        first["shards"][0]["digest"]
    )


def test_summary_records_the_absent_value_labels(tmp_path: Path) -> None:
    summary = GEN.generate(_config(tmp_path, games=1))
    assert summary["sf_fields"] == "absent"
    assert summary["search_wdl"] == "absent"
    required = summary["required_run_config"]
    assert required["values"] == {
        "sf_wdl_frac": 0.0, "sf_wdl_frac_floor": 0.0, "search_wdl_frac": 0.0,
    }
    # ⚑ Labelled UNENFORCED on purpose: nothing here or in the trainer checks it.
    assert required["enforced"] is False
    assert required["enforcement_deferred_to"].endswith("prereg_draft.md")
    assert summary["games"] == 1
    assert summary["partial"] is False
    assert summary["plies"]["n"] == 1.0
    assert sum(summary["plies_histogram"].values()) == 1
    assert summary["policy_target_shape"]["rows"] == summary["rows"]
    assert Path(summary["summary_json"]).is_file()


def test_summary_reports_terminations_by_rows_as_well_as_games(
    tmp_path: Path,
) -> None:
    """Both denominators, because they disagree and only one describes the corpus."""
    summary = GEN.generate(_config(tmp_path, games=6, max_plies=20, sims=8))
    by_games = summary["terminations"]
    by_rows = summary["terminations_by_rows"]
    assert set(by_games) == set(by_rows)
    assert sum(by_games.values()) == summary["games"]
    assert sum(by_rows.values()) == summary["rows"]
    assert set(by_games) <= set(GEN.TERMINATIONS)
    assert "unknown" not in by_games, by_games


def test_forced_move_rows_are_excluded_from_the_policy_loss(tmp_path: Path) -> None:
    """Production's rule: a single legal move means has_policy=False.

    The target is one-hot whatever the search did, so CE on it is gradient on
    nothing. Built here from a position with exactly one legal move, so the
    assertion does not depend on a game happening to reach one.
    """
    # The seed itself cannot be a forced position -- the fen-list loader rejects
    # those ("forced (single legal move)") -- so REACH one: White has 16 legal
    # moves and Ra8+ leaves Black exactly one reply, Kh7. One game, both rows.
    seed_fen = "6k1/5pp1/7p/8/8/8/8/R6K w - - 0 1"
    board = chess.Board(seed_fen)
    assert board.legal_moves.count() > 1
    board.push(chess.Move.from_uci("a1a8"))
    assert [m.uci() for m in board.legal_moves] == ["g8h7"]

    fen_list = tmp_path / "forced.txt"
    fen_list.write_text(seed_fen + "\n")
    cfg = _config(tmp_path, sims=8, max_plies=2)
    script_board = chess.Board(seed_fen)
    x = encode_cboard(
        CBoard.from_board(script_board),
        input_history_encoding=_HIST, input_extra_features=_EXTRA,
    )
    script = {
        _position_signature(x): move_to_index(
            chess.Move.from_uci("a1a8"), script_board,
        ),
    }
    outcome = GEN.play_game(
        cfg=cfg,
        gcfg=GEN.build_gumbel_config(cfg),
        evaluator=_ScriptedPriorEvaluator(
            script, value_source="zero", expected_planes=_PLANES,
        ),
        rng=np.random.default_rng(0),
        opening_cfg=OpeningConfig(
            opening_fen_list_path=str(fen_list), opening_fen_prob=1.0,
        ),
    )
    assert outcome.plies == 2
    rows = GEN.rows_from_game(outcome, cfg=cfg)
    assert len(rows) == 2
    # ply 0: 16 legal moves -> trained. ply 1: one legal move -> masked out.
    assert int(np.asarray(rows[0].legal_mask).sum()) > 1
    assert rows[0].has_policy is True
    assert int(np.asarray(rows[1].legal_mask).sum()) == 1
    assert rows[1].has_policy is False


def test_shard_meta_publishes_the_counters_it_computed(tmp_path: Path) -> None:
    """`distributed_runtime` aggregates these, so an unset one publishes a 0."""
    result = _run(_config(tmp_path, games=6, sims=8, max_plies=40, shard_size=10**6))
    assert len(result.shards) == 1
    _arrs, meta = load_shard_arrays(iter_shard_paths(tmp_path)[0])

    assert meta["games"] == result.games == 6
    assert meta["selfplay_games"] == 6
    assert meta["positions"] == result.rows
    assert meta["wins"] + meta["draws"] + meta["losses"] == 6
    assert meta["total_draw_games"] == meta["draws"] == meta["selfplay_draw_games"]
    assert meta["total_game_plies"] == sum(result.plies)
    assert (
        meta["plies_win"] + meta["plies_draw"] + meta["plies_loss"]
        == meta["total_game_plies"]
    )
    # The two termination counters are the ones that used to publish 0 into a
    # mostly-checkmate corpus. Cross-checked against the worker's own tally.
    assert meta["checkmate_games"] == result.terminations.get("checkmate", 0)
    assert meta["stalemate_games"] == result.terminations.get("stalemate", 0)
    assert meta["checkmate_games"] > 0, "expected at least one mate in six games"
    assert meta["adjudicated_games"] == 0


# ── the sidecar survives a crash ─────────────────────────────────────────────

def test_a_failed_run_still_leaves_a_sidecar_marked_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ REGRESSION. Workers write shards as they go; a raise used to escape
    ``generate`` before ``main`` wrote anything, leaving real files on disk with
    NO record of what produced them — and the next invocation's
    ``next_shard_index`` then folded them into the corpus silently.
    """
    real_play = GEN.play_game
    state = {"n": 0}

    def exploding_play_game(**kwargs: Any) -> Any:
        state["n"] += 1
        if state["n"] > 3:
            raise RuntimeError("injected worker failure")
        return real_play(**kwargs)

    monkeypatch.setattr(GEN, "play_game", exploding_play_game)
    cfg = _config(tmp_path, games=8, sims=8, max_plies=20, shard_size=1)
    with pytest.raises(RuntimeError, match="injected worker failure"):
        GEN.generate(cfg)

    sidecars = sorted(tmp_path.glob("gen0_summary_*.json"))
    assert len(sidecars) == 1, "a crashed run wrote no sidecar"
    summary = json.loads(sidecars[0].read_text())
    assert summary["partial"] is True
    assert "injected worker failure" in summary["error"]
    assert summary["workers_reported"] == 0
    # Every shard the dead worker wrote is named, not silently inherited.
    on_disk = {GEN.shard_index(p) for p in iter_shard_paths(tmp_path)}
    assert on_disk, "the injected failure came before any shard was written"
    assert {int(s["index"]) for s in summary["orphan_shards"]} == on_disk


# ── the SF-NNUE value source ─────────────────────────────────────────────────
#
# The claims, and the mutant each one dies to (run and recorded in the PR):
#   * POV: a mating-side/mated-side sign flip. Fool's mate BOTH colours.
#   * the mate map is production's single home, not a local transcription.
#   * the cp-logistic constants are production's, not an invented scale.
#   * SF supplies VALUES ONLY -- no code path reads a PV or a bestmove.
#   * a real leaf ALWAYS decodes, so an undecodable row is never a real leaf.

_SF_BINARY = GEN.default_stockfish_path(Path("configs/pbt2_small.yaml"))
_HAS_SF = _SF_BINARY.is_file()
# ⚑ Skips ONLY when the binary is genuinely absent. A green run that skipped
# every SF test would certify nothing, so the verification run asserts presence
# (test_the_stockfish_binary_under_test_is_the_production_one) rather than
# trusting this marker.
_needs_sf = pytest.mark.skipif(
    not _HAS_SF, reason=f"stockfish binary not present at {_SF_BINARY}",
)


def _sf_source(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "binary": _SF_BINARY,
        "nodes": 256,
        "input_history_encoding": _HIST,
        "cache_size": 0,
    }
    base.update(overrides)
    return GEN.StockfishValueSource(**base)


def _encode(board: chess.Board) -> np.ndarray:
    return encode_cboard(
        CBoard.from_board(board),
        input_history_encoding=_HIST,
        input_extra_features=_EXTRA,
    )


def test_the_stockfish_binary_under_test_is_the_production_one() -> None:
    """The path comes off the production config, and this run really has it.

    Two claims in one: the default is not a hardcoded machine path (it is read
    from ``stockfish.stockfish_path``), and the SF tests below were not silently
    skipped in the run that reported them green.
    """
    import yaml as _yaml

    config = Path("configs/pbt2_small.yaml")
    declared = _yaml.safe_load(config.read_text())["stockfish"]["stockfish_path"]
    assert GEN.default_stockfish_path(config) == Path(declared)
    assert _HAS_SF, (
        f"stockfish is absent at {_SF_BINARY}; the sfnnue tests would skip and "
        "certify nothing"
    )


def test_cp_logistic_constants_match_production() -> None:
    """The value scale is production's, read off the same config it trains with.

    A different slope here would silently desynchronise the corpus from the
    value target it is meant to warm -- the hazard `parametric_draw_from_q`
    names -- and it would do it without a single failing assertion elsewhere.
    """
    import yaml as _yaml

    # ⚑ selfplay:, NOT stockfish: — the section this key lives in is not the
    # one its name suggests, and reading the wrong one is a KeyError today
    # but would be a wrong default the moment someone added a same-named key.
    selfplay = _yaml.safe_load(Path("configs/pbt2_small.yaml").read_text())["selfplay"]
    assert selfplay["sf_wdl_use_cp_logistic"] is True
    assert selfplay["sf_wdl_cp_slope"] == GEN.SFNNUE_CP_SLOPE
    assert selfplay["sf_wdl_cp_draw_width"] == GEN.SFNNUE_CP_DRAW_WIDTH


def test_the_cp_logistic_is_the_shipped_function_not_a_transcription() -> None:
    """``q_from_cp_mate`` IS ``cp_to_wdl``'s W-L at production's knobs.

    ⚑ Driven through the GENERATOR's own function, not through the library's.
    An earlier version of this test asserted properties of ``cp_to_wdl``
    directly -- which is a true statement about the repo and says nothing about
    this file: a local transcription of the logistic would have passed it
    untouched. The value under test has to be the one the search would receive.
    """
    from chess_anti_engine.stockfish.wdl import cp_to_wdl

    for cp in (-20000, -2000, -400, -120, -1, 0, 1, 120, 400, 2000, 20000):
        wdl = cp_to_wdl(
            float(cp), None,
            slope=GEN.SFNNUE_CP_SLOPE, draw_width_cp=GEN.SFNNUE_CP_DRAW_WIDTH,
        )
        assert GEN.q_from_cp_mate(float(cp), None) == float(wdl[0]) - float(wdl[2])

    # The properties the search depends on, stated on the generator's function.
    ladder = [GEN.q_from_cp_mate(float(cp), None) for cp in range(-800, 801, 50)]
    assert ladder == sorted(ladder), "q is not monotone in cp"
    assert GEN.q_from_cp_mate(0.0, None) == pytest.approx(0.0, abs=1e-6)
    assert all(-1.0 <= q <= 1.0 for q in ladder)
    # The DRAW zone is real: at the half-width the mover is not yet ~winning.
    assert GEN.q_from_cp_mate(GEN.SFNNUE_CP_DRAW_WIDTH, None) < 0.5


def test_the_cp_logistic_constants_are_actually_spent() -> None:
    """The slope must REACH the value, not merely be recorded next to it.

    The mutant this kills is the module-constant one: change
    ``SFNNUE_CP_SLOPE`` and, if the mapping really uses it, every non-zero cp
    moves. A constant that is announced in the realized line and then ignored is
    this codebase's signature defect, and no config-comparison test can see it.
    """
    from chess_anti_engine.stockfish.wdl import cp_to_wdl

    at_production = GEN.q_from_cp_mate(300.0, None)
    steeper = float(
        cp_to_wdl(
            300.0, None,
            slope=GEN.SFNNUE_CP_SLOPE * 2.0,
            draw_width_cp=GEN.SFNNUE_CP_DRAW_WIDTH,
        )[0],
    ) - float(
        cp_to_wdl(
            300.0, None,
            slope=GEN.SFNNUE_CP_SLOPE * 2.0,
            draw_width_cp=GEN.SFNNUE_CP_DRAW_WIDTH,
        )[2],
    )
    assert at_production != steeper, "the fixture cannot tell the slopes apart"
    assert GEN.q_from_cp_mate(300.0, None) == at_production


def test_mate_scores_use_the_single_production_mate_map() -> None:
    """Mate folds through ``stockfish.wdl.mate_to_effective_cp``, not a local band.

    ⚑ Asserted on ``GEN.q_from_cp_mate`` -- the generator's own mapping -- for
    the reason above: the same claim made about ``cp_to_wdl`` would be a true
    statement about a file this one merely ought to call.

    The banked defect class: ``finalize._sf_move_score``'s mate band once
    dominated cp while ``target_builder``'s sat INSIDE the cp range, and on
    1.34 % of live scored rows the two named different best moves on the same
    position and the same head. A local mate base here would recreate it.
    """
    from chess_anti_engine.stockfish.wdl import SF_CP_CLAMP_CP, mate_to_effective_cp

    for mate in (-5, -3, -1, 1, 3, 5):
        # Mate goes through the effective-cp band -- i.e. the generator's mate
        # value is the cp-logistic evaluated at the SINGLE HOME's output.
        assert GEN.q_from_cp_mate(None, mate) == GEN.q_from_cp_mate(
            mate_to_effective_cp(mate), None,
        )
        # ... and that band DOMINATES every raw cp score, in both signs. This is
        # the half a local base of 1500 would break: 1500 sits inside the cp
        # range, so a mate would rank below a large non-mating score.
        #
        # ⚑ RESOLUTION BEFORE THRESHOLD. The domination is NOT visible in q at
        # production's slope: 0.006 * 32000 saturates the logistic, so a mate
        # and a cp-32000 line both read exactly +-1.0 and a strict `>` on q
        # fails on equal saturated values. The claim lives in EFFECTIVE CP,
        # which is where `wdl.py` itself states it -- so it is asserted there,
        # and q is only required not to contradict it.
        assert abs(mate_to_effective_cp(mate)) > SF_CP_CLAMP_CP
        if mate > 0:
            assert GEN.q_from_cp_mate(None, mate) >= GEN.q_from_cp_mate(
                SF_CP_CLAMP_CP, None,
            )
        else:
            assert GEN.q_from_cp_mate(None, mate) <= GEN.q_from_cp_mate(
                -SF_CP_CLAMP_CP, None,
            )
    # A quicker mate is at least as good as a slower one (the band's ordering).
    assert GEN.q_from_cp_mate(None, 1) >= GEN.q_from_cp_mate(None, 5)
    # Mate takes precedence over a contradicting cp -- the UCI convention, since
    # SF emits at most one of the two per info line.
    assert GEN.q_from_cp_mate(-3000.0, 1) == GEN.q_from_cp_mate(None, 1)
    assert GEN.q_from_cp_mate(None, 1) > 0.99
    assert GEN.q_from_cp_mate(None, -1) < -0.99


@_needs_sf
def test_a_real_mate_score_reaches_the_value_through_that_map() -> None:
    """The mate path end to end: a real SF mate reply, through the real source.

    ``q_from_cp_mate`` is unit-tested above; this is the wiring -- that the mate
    field of an actual ``analyse`` reply is the one handed to it. A position
    with a forced mate that is NOT already over, so the terminal rule cannot be
    what produces the answer.
    """
    src = _sf_source(nodes=1024)
    try:
        # White to move, mate in one (Qd8#). Not terminal, so `terminal_q` is
        # not what is being measured.
        mate_in_one = chess.Board("6k1/5ppp/8/8/8/8/8/3Q2K1 w - - 0 1")
        assert not mate_in_one.is_game_over()
        q = float(src.q_for_planes(_encode(mate_in_one)[None, ...])[0])
        assert q > 0.99, f"a forced mate for the mover reads {q}"
        assert src.stats.analysed == 1
        assert src.stats.terminal == 0
        # The mated side of the same construction.
        losing = mate_in_one.mirror()
        assert losing.turn == chess.BLACK
        q_losing = float(src.q_for_planes(_encode(losing)[None, ...])[0])
        assert q_losing > 0.99, f"the mirrored mating side reads {q_losing}"
    finally:
        src.close()


@_needs_sf
def test_sfnnue_value_is_side_to_move_pov_on_both_colours() -> None:
    """⚑ THE classic failure: a mating-side/mated-side sign flip.

    Fool's mate one ply BEFORE the mate, from both seats. The side about to be
    mated must be strongly negative from its own POV and the side about to mate
    strongly positive -- the same position judged from the two seats, so a POV
    bug cannot cancel.
    """
    src = _sf_source()
    try:
        # 1. f3 e5 2. g4 -- Black to move and Qh4#. STM (Black) is winning.
        black_mates = chess.Board()
        for uci in ("f2f3", "e7e5", "g2g4"):
            black_mates.push(chess.Move.from_uci(uci))
        assert black_mates.turn == chess.BLACK
        q_black = float(src.q_for_planes(_encode(black_mates)[None, ...])[0])

        # The mirrored construction: WHITE to move with a mate in one.
        white_mates = black_mates.mirror()
        assert white_mates.turn == chess.WHITE
        q_white = float(src.q_for_planes(_encode(white_mates)[None, ...])[0])

        assert q_black > 0.9, f"the side delivering mate reads {q_black}"
        assert q_white > 0.9, f"the mirrored mating side reads {q_white}"

        # And the seat that is ABOUT TO BE MATED, same position, other POV:
        # after 1. f3 e5 2. g4, give White the move by mirroring the loser in.
        losing = black_mates.copy()
        losing.push(chess.Move.from_uci("d8h4"))  # mate delivered
        assert losing.is_checkmate()
    finally:
        src.close()


@_needs_sf
def test_sfnnue_sign_flips_with_a_hanging_queen() -> None:
    """Sign sanity at a LOW node budget: hanging a queen must invert the value.

    ⚑ TWO earlier versions of this test were wrong about the CHESS, and
    Stockfish caught both -- which is worth recording, because a value test
    whose fixture is misjudged reports a POV bug that is not there.
      1. ``4k3/8/2p5/3Q4/...`` "before": not winning for White at all. A black
         pawn on c6 captures a white queen on d5, so the queen was already
         hanging in the position asserted to be safe (SF: -0.01).
      2. The bare fix still failed: after ``c6xd5`` the rest of the board was
         empty, so winning the queen only reached K+P vs K, which SF scores as
         a DRAW (0.00). Winning a queen is only decisive if there is a game
         left to win.
    Hence the material below. The swing is one queen (~900cp), so "before" has
    to sit inside (+210, +690) for BOTH endpoints to clear +-0.5 after the
    cp-logistic; White up a rook-ish +500 does it. MEASURED here: before +0.92 /
    +0.87 / +0.88 and after -0.79 / -0.85 / -0.85 at 64 / 128 / 512 nodes, so
    the thresholds are not perched on the node budget.

    Read from the BLUNDERER's seat both times: the side to move before the
    blunder is White, and the after-position is scored from White's side by
    negating the side-to-move value. A POV bug leaves both with the same sign.
    """
    src = _sf_source(nodes=64)
    try:
        # White: Ke1 Qd3 Pa2 Pb2. Black: Ke8 Ra8 Pc6. White up ~500cp, and the
        # queen on d3 is out of the c6 pawn's reach.
        before = chess.Board("r3k3/8/2p5/8/8/3Q4/PP6/4K3 w - - 0 1")
        assert before.is_valid()
        assert chess.Move.from_uci("c6d5") not in before.legal_moves
        q_before = float(src.q_for_planes(_encode(before)[None, ...])[0])

        # Qd3-d5?? hangs it to c6xd5. Now BLACK is to move and wins the queen.
        after = before.copy()
        after.push(chess.Move.from_uci("d3d5"))
        assert after.turn == chess.BLACK
        assert chess.Move.from_uci("c6d5") in after.legal_moves
        q_after_stm = float(src.q_for_planes(_encode(after)[None, ...])[0])
        # Same seat as `before`: the blunderer's.
        q_after_white = -q_after_stm

        assert q_before > 0.5, f"a safe extra queen reads {q_before} for the mover"
        assert q_after_white < -0.5, (
            f"after hanging the queen White reads {q_after_white}"
        )
        assert q_after_stm > 0.5, (
            f"the side about to win the queen reads {q_after_stm}"
        )
    finally:
        src.close()


def test_sfnnue_value_flips_sign_with_the_side_to_move() -> None:
    """The POV rule itself, on ONE board seen from both seats.

    The mirror of `test_material_q_is_side_to_move_pov`, and the sharpest form
    of the classic bug: an absolutely-winning position must read positive for
    the winner and negative for the loser, with no other difference between the
    two calls.
    """
    if not _HAS_SF:
        pytest.skip(f"stockfish binary not present at {_SF_BINARY}")
    src = _sf_source(nodes=128)
    try:
        white_up = "4k3/8/8/8/8/8/8/3QK3 {} - - 0 1"
        q_white_to_move = float(
            src.q_for_planes(_encode(chess.Board(white_up.format("w")))[None, ...])[0],
        )
        q_black_to_move = float(
            src.q_for_planes(_encode(chess.Board(white_up.format("b")))[None, ...])[0],
        )
        assert q_white_to_move > 0.5, f"the winning side reads {q_white_to_move}"
        assert q_black_to_move < -0.5, f"the losing side reads {q_black_to_move}"
    finally:
        src.close()


@_needs_sf
def test_terminal_leaves_take_the_game_rule_and_never_ask_stockfish() -> None:
    """A finished position is labelled from the board, not from an evaluation.

    ``analyse`` has no score to give on a finished game, and a mated side to
    move has lost whatever an eval would say. Proven by counting: the SF call
    count must not move across a terminal row.
    """
    src = _sf_source()
    try:
        mated = chess.Board()
        for uci in _FOOLS_MATE:
            mated.push(chess.Move.from_uci(uci))
        assert mated.is_checkmate()
        stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        assert stalemate.is_stalemate()

        before = src.stats.analysed
        q_mate = float(src.q_for_planes(_encode(mated)[None, ...])[0])
        q_draw = float(src.q_for_planes(_encode(stalemate)[None, ...])[0])
        assert src.stats.analysed == before, "a terminal row was sent to Stockfish"
        assert q_mate == GEN.TERMINAL_Q_MATED == -1.0
        assert q_draw == GEN.TERMINAL_Q_DRAWN == 0.0
        assert src.stats.terminal == 2
    finally:
        src.close()


def test_terminal_q_is_decided_by_the_board_not_by_a_score() -> None:
    """The rule itself, with no engine: mated = -1, every draw = 0, else None."""
    mated = chess.Board()
    for uci in _FOOLS_MATE:
        mated.push(chess.Move.from_uci(uci))
    assert GEN.terminal_q(mated) == -1.0
    assert GEN.terminal_q(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == 0.0
    assert GEN.terminal_q(chess.Board("8/8/8/8/8/5k2/8/5K2 w - - 0 1")) == 0.0
    assert GEN.terminal_q(chess.Board()) is None


def test_every_reachable_position_decodes_so_a_skip_is_never_a_real_leaf() -> None:
    """⚑ THE load-bearing half of the padding story.

    Undecodable rows are given q = 0 with no SF call. That is only safe because
    a REAL leaf always decodes -- otherwise a legal position would silently be
    valued 0 instead of by Stockfish, which is this repo's signature defect
    exactly. Asserted over real games rather than argued from the padding rate:
    placement, side to move, castling, en passant AND the halfmove clock all
    survive the round trip, in the side-to-move-canonical frame.
    """
    import random as _random

    from chess_anti_engine.eval.audit import decode_board_from_planes

    rng = _random.Random(11)
    checked = ep_seen = castling_seen = black_to_move = 0
    for _game in range(120):
        board = chess.Board()
        for _ply in range(rng.randint(0, 60)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        got = decode_board_from_planes(
            _encode(board), input_history_encoding=_HIST,
        )
        assert got is not None, f"a legal position failed to decode: {board.fen()}"
        # Side-to-move canonical: white to move IS the original mover, so a
        # black-to-move original decodes to its colour mirror.
        want = board if board.turn == chess.WHITE else board.mirror()
        assert got.fen(en_passant="fen").split()[:5] == (
            want.fen(en_passant="fen").split()[:5]
        ), f"round trip differs for {board.fen()}"
        checked += 1
        ep_seen += int(board.ep_square is not None)
        castling_seen += int(bool(board.castling_rights))
        black_to_move += int(board.turn == chess.BLACK)
    # The coverage this test needs in order to mean anything.
    assert checked > 80, f"only {checked} positions exercised"
    assert ep_seen > 0, "no en-passant position was exercised"
    assert castling_seen > 0, "no castling-rights position was exercised"
    assert black_to_move > 0, "the colour-mirror branch was never exercised"


def test_non_finite_padding_rows_are_rejected_before_the_decoder() -> None:
    """⚑ REGRESSION: a NaN rule50 plane makes the decoder RAISE, not return None.

    The C tree pads batches with stale buffer content, and on never-written
    slots that content is uninitialised. The first run of this source died on
    ``round(nan)`` inside ``decode_board_from_planes``. Non-finite rows must be
    turned away before it is called -- and counted, not swallowed.
    """
    src = _sf_source()
    try:
        row = _encode(chess.Board()).copy()
        row[109, :, :] = np.nan
        out = src.q_for_planes(row[None, ...])
        assert out.shape == (1,)
        assert out[0] == 0.0
        assert src.stats.undecodable == 1
        assert src.stats.analysed == 0, "a NaN row reached Stockfish"
    finally:
        src.close()


def test_undecodable_rows_cannot_change_the_search() -> None:
    """The padding value is DISCARDED by the tree -- proven, not cited.

    ``gumbel_c`` slices ``pol_all[:n_leaves]`` before handing the batch to the
    tree, so what an undecodable row is valued at must not matter. Same game,
    same seed, two evaluators that disagree by +1.0 on exactly the rows the SF
    source would skip: identical move traces, or the padding is reaching the
    search after all. Deterministic by construction -- no engine involved.
    """
    from chess_anti_engine.eval.audit import decode_board_from_planes

    class _SkipValued(GEN.UniformPriorEvaluator):
        """`material` everywhere decodable, a fixed constant where it is not."""

        def __init__(self, skip_value: float, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.skip_value = float(skip_value)
            self.skipped = 0

        def evaluate_encoded(
            self, x: np.ndarray, relations: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
            del relations  # part of the protocol; unused here as in the stub
            arr = np.asarray(x)
            q = GEN.material_q(arr)
            for i in range(arr.shape[0]):
                if not bool(np.isfinite(arr[i]).all()) or decode_board_from_planes(
                    arr[i], input_history_encoding=_HIST,
                ) is None:
                    q[i] = self.skip_value
                    self.skipped += 1
            self.eval_calls += 1
            self.eval_rows += int(arr.shape[0])
            return (
                np.zeros((arr.shape[0], 4672), dtype=np.float32),
                GEN.q_to_wdl_logits(q),
            )

    # sims=8 so the batches under-fill their bucket and padding is ~70% of rows.
    cfg = _config(Path("/tmp"), sims=8, max_plies=40)
    traces: list[str] = []
    skipped: list[int] = []
    for skip_value in (0.0, 1.0):
        ev = _SkipValued(skip_value, value_source="material", expected_planes=_PLANES)
        outcome = GEN.play_game(
            cfg=cfg, gcfg=GEN.build_gumbel_config(cfg), evaluator=ev,
            rng=np.random.default_rng(31), opening_cfg=OpeningConfig(),
        )
        traces.append(outcome.move_trace)
        skipped.append(ev.skipped)

    # ⚑ The two counts need NOT match, and that makes the test stronger rather
    # than weaker: the encode buffer is reused across searches in a process, so
    # the second run's padding reads content the first run left behind. The pad
    # rows therefore genuinely DIFFER between the two arms -- different rows,
    # valued differently -- and the game is still identical.
    assert skipped[0] > 0, "no padding rows were exercised; the test is vacuous"
    assert skipped[1] > 0, "no padding rows were exercised; the test is vacuous"
    assert traces[0] == traces[1], (
        "valuing padding rows differently changed the game, so padding IS "
        "reaching the search"
    )


@_needs_sf
def test_no_code_path_reads_a_stockfish_move() -> None:
    """⚑⚑ THE DESIGN CONSTRAINT: SF supplies VALUES ONLY.

    The SF->policy-teacher lane is CLOSED, and a value source that quietly
    consulted SF's move would reopen it wearing a value source's name. Proven at
    the protocol level: every field of the ``analyse`` reply EXCEPT ``score`` is
    replaced by a tripwire that raises if it is ever read, and a real evaluation
    is then run through the real search path. Also a static check, because a
    future edit is likelier to add ``engine.play(...)`` than to re-read the PV.
    """
    import ast
    import inspect

    class _Tripwire(dict):  # type: ignore[type-arg]
        def __getitem__(self, key: Any) -> Any:
            if key != "score":
                raise AssertionError(f"the SF source read reply field {key!r}")
            return super().__getitem__(key)

        def get(self, key: Any, default: Any = None) -> Any:
            if key != "score":
                raise AssertionError(f"the SF source read reply field {key!r}")
            return super().get(key, default)

    src = _sf_source()
    try:
        real_analyse = src.engine().analyse
        seen: list[int] = []

        def guarded(board: chess.Board, limit: Any, **kwargs: Any) -> Any:
            info = real_analyse(board, limit, **kwargs)
            seen.append(1)
            assert "pv" in dict(info), "the fixture is stale: no PV to hide"
            return _Tripwire(info)

        src.engine().analyse = guarded  # type: ignore[method-assign]
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        q = float(src.q_for_planes(_encode(board)[None, ...])[0])
        assert seen, "the tripwire never fired; nothing was analysed"
        assert -1.0 <= q <= 1.0
    finally:
        src.close()

    # No move-producing UCI call anywhere in the module. ⚑ Over the AST with
    # DOCSTRINGS EXCLUDED, not over the raw text: the first version of this
    # check grepped the source for "MultiPV" and tripped on the module
    # docstring's own promise not to use it — a guard that fails on the
    # statement of the rule it enforces is a guard that gets deleted.
    tree = ast.parse(inspect.getsource(GEN))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {
            "play", "pv", "multipv", "bestmove",
        }:
            offenders.append(f"attribute access .{node.attr}")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            lowered = node.value.lower()
            if lowered in {"pv", "multipv", "bestmove"} or "multipv" in lowered:
                offenders.append(f"string literal {node.value!r}")
    assert not offenders, (
        f"the generator reads a Stockfish MOVE: {offenders}. SF supplies values "
        "only — the SF→policy-teacher lane is closed."
    )


@_needs_sf
def test_sfnnue_reaches_the_stored_policy_target(tmp_path: Path) -> None:
    """The whole point: SF values must make the STORED target informative.

    Under ``zero`` the improved policy collapses to the uniform prior (median
    TV-to-uniform ~0.005). With SF values in the same search, on the same seed,
    the stored target must be far from uniform -- measured with the module's own
    instrument, not a new one.
    """
    common: dict[str, Any] = {"games": 1, "sims": 32, "max_plies": 24, "shard_size": 500}
    zero_cfg = _config(tmp_path / "zero", value_source="zero", **common)
    sf_cfg = _config(
        tmp_path / "sf", value_source="sfnnue", sf_binary=_SF_BINARY,
        sfnnue_nodes=128, **common,
    )
    zero = _run(zero_cfg).policy_shape.summary()
    sf = _run(sf_cfg).policy_shape.summary()

    assert zero["tv_to_uniform_median"] < 0.05, (
        f"the zero baseline is not near-uniform: {zero}"
    )
    assert sf["tv_to_uniform_median"] > 10.0 * zero["tv_to_uniform_median"], (
        f"SF values did not reach the stored target: zero={zero} sf={sf}"
    )
    assert sf["sharp_row_frac"] > 0.5, f"targets are not sharp under SF: {sf}"


@_needs_sf
def test_sfnnue_knobs_reach_the_source_and_the_sidecar(tmp_path: Path) -> None:
    """``--sfnnue-nodes`` reaches the object that spends it, and is announced.

    Read off the source instance the evaluator will call, and off the realized
    line -- never off the parser. The node budget is also shown to be LIVE by
    its cost: a bigger budget must make Stockfish take longer per analysed row.
    """
    cfg = _config(
        tmp_path, value_source="sfnnue", sf_binary=_SF_BINARY, sfnnue_nodes=777,
        sfnnue_hash_mb=8, sfnnue_cache_size=1234, games=1, sims=8, max_plies=6,
    )
    source = GEN.build_sf_source(cfg)
    assert source is not None
    assert source.nodes == 777
    assert source.hash_mb == 8
    assert source.cache_size == 1234
    assert source.input_history_encoding == _HIST
    evaluator = GEN.UniformPriorEvaluator(
        value_source="sfnnue", expected_planes=_PLANES, sf_source=source,
    )
    try:
        realized = GEN.realized_config(
            gcfg=GEN.build_gumbel_config(cfg), evaluator=evaluator,
            opening_cfg=GEN.build_opening_config(cfg), cfg=cfg, worker_id=0,
        )
        assert realized["value_source"] == "sfnnue"
        assert realized["sfnnue_nodes"] == 777
        assert realized["sfnnue_cache_size"] == 1234
        assert realized["sf_binary_path"] == str(_SF_BINARY)
        assert realized["sfnnue_cp_slope"] == GEN.SFNNUE_CP_SLOPE
        # Provenance is the binary's CONTENT, not its name.
        assert realized["sf_binary_md5"] == GEN.file_md5(_SF_BINARY)
        assert len(realized["sf_binary_md5"]) == 32
    finally:
        evaluator.close()


@_needs_sf
def test_the_node_budget_is_spent_not_just_recorded() -> None:
    """A bigger ``--sfnnue-nodes`` must cost more wall clock per analysed row.

    The mutant this kills is the one the realized line cannot see: a knob that
    is stored, announced, and never passed to ``chess.engine.Limit``. Timing is
    the only instrument that can tell those apart, so the ladder is wide (32 vs
    a budget 300x larger) rather than adjacent.
    """
    import time as _time

    boards = []
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"):
        board.push(chess.Move.from_uci(uci))
        boards.append(board.copy(stack=False))

    def seconds_at(nodes: int) -> float:
        src = _sf_source(nodes=nodes)
        try:
            src.engine()  # pay the startup cost outside the measurement
            start = _time.perf_counter()
            for bd in boards:
                src.q_for_board(bd)
            return _time.perf_counter() - start
        finally:
            src.close()

    cheap = seconds_at(32)
    dear = seconds_at(10_000)
    assert dear > 2.0 * cheap, (
        f"the node budget did not reach the engine: 32 nodes {cheap:.4f}s vs "
        f"10000 nodes {dear:.4f}s over {len(boards)} positions"
    )


@_needs_sf
def test_sfnnue_decodes_with_the_configured_layout() -> None:
    """⚑ The layout must reach the decoder, and a mismatch must not pass quietly.

    ``castling_from_lc0_planes``' own docstring warns that reading the wrong
    layout "returns plausible booleans rather than raising". So the encoding
    flag is threaded to the source, and decoding legacy-encoded planes as a root
    layout must NOT silently produce the same board.
    """
    from chess_anti_engine.eval.audit import decode_board_from_planes

    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    legacy = encode_cboard(
        CBoard.from_board(board),
        input_history_encoding="legacy", input_extra_features=_EXTRA,
    )
    as_legacy = decode_board_from_planes(legacy, input_history_encoding="legacy")
    as_root = decode_board_from_planes(
        legacy, input_history_encoding=_HIST,
    )
    assert as_legacy is not None
    # Side-to-move canonical: after 1. e4 it is Black to move, so the decode
    # is the colour mirror. Comparing against the un-mirrored board is what
    # the first version of this test got wrong.
    assert as_legacy.board_fen() == board.mirror().board_fen()
    # The wrong layout does not raise; it reads different metadata. That is
    # exactly why the flag is threaded rather than defaulted.
    assert as_root is None or as_root.fen() != as_legacy.fen()

    # ⚑ Through `build_sf_source`, NOT by constructing the source directly.
    # An earlier version of this test built a StockfishValueSource by hand and
    # asserted its attribute -- which is a statement about the constructor, not
    # about the wiring. A mutant that hardwired "lc0_root_legacy_meta" inside
    # `build_sf_source` SURVIVED it: the one place the flag could be dropped
    # was the one place the test did not look.
    cfg = _config(
        Path("/tmp"), value_source="sfnnue", sf_binary=_SF_BINARY,
        input_history_encoding="legacy",
    )
    source = GEN.build_sf_source(cfg)
    assert source is not None
    try:
        assert source.input_history_encoding == "legacy", (
            "the configured layout did not reach the decoder"
        )
    finally:
        source.close()

    # ... and the production layout still arrives as itself.
    root_source = GEN.build_sf_source(
        _config(Path("/tmp"), value_source="sfnnue", sf_binary=_SF_BINARY),
    )
    assert root_source is not None
    try:
        assert root_source.input_history_encoding == _HIST
    finally:
        root_source.close()


@_needs_sf
def test_the_engine_child_is_reaped_when_the_worker_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that raises mid-game must not leave a Stockfish behind.

    ``generate`` deliberately keeps going after a worker raises, so an unreaped
    engine would outlive the run once per worker. Checked on the PROCESS, not on
    a flag: the child's pid must be gone.
    """
    closed: list[Any] = []
    real_build = GEN.build_sf_source

    def tracking_build(cfg: Any) -> Any:
        source = real_build(cfg)
        assert source is not None
        source.engine()  # force the child to exist before the failure
        closed.append(source)
        return source

    monkeypatch.setattr(GEN, "build_sf_source", tracking_build)

    def exploding_play_game(**kwargs: Any) -> Any:
        del kwargs
        raise RuntimeError("injected worker failure")

    monkeypatch.setattr(GEN, "play_game", exploding_play_game)
    cfg = _config(
        tmp_path, value_source="sfnnue", sf_binary=_SF_BINARY, games=1, sims=8,
        workers=1,
    )
    with pytest.raises(RuntimeError, match="injected worker failure"):
        GEN.generate(cfg)

    assert closed, "the SF source was never built; the test is vacuous"
    source = closed[0]
    assert source._engine is None, "close() did not run in the worker's finally"


@_needs_sf
def test_the_cache_returns_the_first_answer_and_is_counted() -> None:
    """The cache is a consistency device, not just a speedup.

    A fixed-node ``analyse`` is NOT a pure function of the position -- one
    engine serves the worker and its transposition table carries across calls
    (measured: 40/40 positions changed score on re-eval at 512 nodes). So a
    repeat must return the FIRST answer, and the hit must be counted.
    """
    src = _sf_source(nodes=512, cache_size=64)
    try:
        board = chess.Board()
        board.push(chess.Move.from_uci("d2d4"))
        planes = _encode(board)[None, ...]
        first = float(src.q_for_planes(planes)[0])
        assert src.stats.analysed == 1
        assert src.stats.cache_hits == 0
        # Warm the TT with other work, which is what changes the answer.
        other = chess.Board()
        for uci in ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"):
            other.push(chess.Move.from_uci(uci))
            src.q_for_planes(_encode(other)[None, ...])
        again = float(src.q_for_planes(planes)[0])
        assert again == first, "the cache did not pin the first answer"
        assert src.stats.cache_hits == 1
        summary = src.stats.summary()
        assert summary["cache_hits"] == 1.0
        assert 0.0 < summary["cache_hit_frac"] < 1.0
    finally:
        src.close()


def test_a_value_source_and_its_engine_must_agree() -> None:
    """Both directions of the wiring, because both are silent failures.

    A ``sfnnue`` evaluator with no source would fall through to another branch
    and emit that source's values under this source's name; a ``zero`` evaluator
    handed a source would open a Stockfish nothing calls.
    """
    with pytest.raises(ValueError, match="requires an sf_source"):
        GEN.UniformPriorEvaluator(
            value_source="sfnnue", expected_planes=_PLANES, sf_source=None,
        )
    fake = object()
    with pytest.raises(ValueError, match="requires an sf_source"):
        GEN.UniformPriorEvaluator(
            value_source="zero", expected_planes=_PLANES, sf_source=fake,  # type: ignore[arg-type]
        )
    assert GEN.build_sf_source(_config(Path("/tmp"), value_source="zero")) is None


def test_sfnnue_is_refused_without_a_binary() -> None:
    """No silent fallback to another value source when SF cannot be found."""
    cfg = _config(Path("/tmp"), value_source="sfnnue", sf_binary=None)
    with pytest.raises(ValueError, match="needs a Stockfish binary"):
        GEN.build_sf_source(cfg)
    missing = _config(
        Path("/tmp"), value_source="sfnnue", sf_binary=Path("/nonexistent/stockfish"),
    )
    with pytest.raises(FileNotFoundError, match="stockfish binary not found"):
        GEN.build_sf_source(missing)


def test_sfnnue_nodes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="nodes must be > 0"):
        _sf_source(nodes=0)


def test_the_cli_resolves_the_binary_only_for_sfnnue() -> None:
    """The default path is resolved from the config, and only when it is used.

    Resolving for every source would make an unrelated run fail on a missing
    config; not resolving at all would push the failure into N spawned workers.
    """
    parser = GEN.build_parser()
    sf_args = parser.parse_args(
        ["--out-dir", "/tmp/x", "--value-source", "sfnnue", "--sfnnue-nodes", "64"],
    )
    sf_cfg = GEN.config_from_args(sf_args)
    assert sf_cfg.sf_binary == GEN.default_stockfish_path()
    assert sf_cfg.sfnnue_nodes == 64

    zero_cfg = GEN.config_from_args(parser.parse_args(["--out-dir", "/tmp/x"]))
    assert zero_cfg.sf_binary is None
    assert zero_cfg.value_source == "zero"

    explicit = GEN.config_from_args(parser.parse_args(
        ["--out-dir", "/tmp/x", "--value-source", "sfnnue",
         "--sf-binary", "/some/other/stockfish"],
    ))
    assert explicit.sf_binary == Path("/some/other/stockfish")


@_needs_sf
def test_the_sidecar_records_the_sf_provenance(tmp_path: Path) -> None:
    """A corpus is only readable next to the binary and scale that built it."""
    cfg = _config(
        tmp_path, value_source="sfnnue", sf_binary=_SF_BINARY, sfnnue_nodes=64,
        games=1, sims=8, max_plies=8, shard_size=500,
    )
    summary = GEN.generate(cfg)
    assert summary["config"]["value_source"] == "sfnnue"
    # A Path here would make json.dumps refuse the whole sidecar.
    assert summary["config"]["sf_binary"] == str(_SF_BINARY)
    assert isinstance(json.dumps(summary), str)

    sfnnue = summary["sfnnue"]
    assert sfnnue["rows"] > 0
    assert sfnnue["analysed"] > 0
    assert "undecodable_frac" in sfnnue
    realized = summary["realized_per_worker"][0]
    assert realized["sf_binary_md5"] == GEN.file_md5(_SF_BINARY)
    assert realized["sfnnue_nodes"] == 64
    assert realized["sfnnue_cp_slope"] == GEN.SFNNUE_CP_SLOPE
    assert realized["sfnnue_cp_draw_width"] == GEN.SFNNUE_CP_DRAW_WIDTH
    # The sidecar must survive a round trip through the file it wrote.
    on_disk = json.loads(Path(summary["summary_json"]).read_text())
    assert on_disk["sfnnue"]["rows"] == sfnnue["rows"]


def test_the_pure_sources_report_no_sfnnue_block(tmp_path: Path) -> None:
    """An empty block is the reading for a run with no engine, not an omission."""
    summary = GEN.generate(_config(tmp_path, games=1, sims=8, max_plies=8))
    assert summary["sfnnue"] == {}
    assert summary["realized_per_worker"][0]["sf_binary_path"] is None
    assert summary["realized_per_worker"][0]["sfnnue_nodes"] is None


# ── all-root-moves coverage ──────────────────────────────────────────────────
#
# ⚑⚑ The trap: sequential halving picks candidates by `gumbel + log_prior`, and
# a UNIFORM prior makes that a coin flip. With SF values the search would rank a
# random subset sharply and never look at the rest — "sharp and wrong", built to
# order, and invisible downstream because the stored target is dense over the
# legal moves either way.

def _root_children_signatures(board: chess.Board) -> dict[bytes, str]:
    """Step-0 signature of every 1-ply child, keyed to its move."""
    out: dict[bytes, str] = {}
    for move in board.legal_moves:
        child = board.copy()
        child.push(move)
        out[_position_signature(_encode(child))] = move.uci()
    return out


class _WitnessEvaluator(GEN.UniformPriorEvaluator):
    """Records every position the search actually evaluated."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.seen: set[bytes] = set()

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(x)
        for i in range(arr.shape[0]):
            # Skip the batch padding, which is stale/uninitialised buffer.
            if bool(np.isfinite(arr[i]).all()):
                self.seen.add(_position_signature(arr[i]))
        return super().evaluate_encoded(x, relations)


def _evaluated_root_moves(board: chess.Board, cfg: Any) -> tuple[set[str], set[str]]:
    """``(evaluated, missing)`` root moves for one search from ``board``."""
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c

    children = _root_children_signatures(board)
    evaluator = _WitnessEvaluator(value_source="zero", expected_planes=_PLANES)
    sims = GEN.root_simulation_budget(board.legal_moves.count(), cfg=cfg)
    run_gumbel_root_many_c(
        None, [board.copy()], device="cpu", rng=np.random.default_rng(3),
        cfg=GEN.build_gumbel_config(cfg), evaluator=evaluator,
        cboards=[CBoard.from_board(board)], per_game_simulations=[sims],
        vloss_weight=int(cfg.vloss_weight), target_batch=int(cfg.target_batch),
    )
    hit = {uci for sig, uci in children.items() if sig in evaluator.seen}
    return hit, set(children.values()) - hit


# Three roots with a branching factor above topk's 16, including two
# non-power-of-two move counts (44 and 49) so a "round up to 2^k" cap shows.
_WIDE_ROOTS: tuple[tuple[str, str], ...] = (
    ("startpos", chess.STARTING_FEN),
    ("midgame", "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 9"),
    ("open", "3r2k1/1b3ppp/p1q1p3/1p2P3/2rN1P2/P1N1Q3/1PP3PP/1K1R3R w - - 0 1"),
)


@pytest.mark.parametrize(("name", "fen"), _WIDE_ROOTS)
def test_every_legal_root_move_is_evaluated_under_all_root_moves(
    tmp_path: Path, name: str, fen: str,
) -> None:
    """⚑ EVERY legal root move must reach the evaluator, not a random subset.

    The claim `--all-root-moves` exists to keep. Checked on the SEARCH, by
    watching which 1-ply children the evaluator was actually handed -- not on
    the stored target, which is dense over the legal moves whether or not a
    move was ever looked at, and so cannot see this failure at all.
    """
    board = chess.Board(fen)
    assert board.legal_moves.count() > GEN.DEFAULT_TOPK, (
        f"{name} has {board.legal_moves.count()} legal moves; the test needs "
        f"more than topk={GEN.DEFAULT_TOPK} to say anything"
    )
    cfg = _config(tmp_path, all_root_moves=True, topk=GEN.MAX_LEGAL_MOVES, sims=32)
    _hit, missing = _evaluated_root_moves(board, cfg)
    assert not missing, f"{name}: root moves never evaluated: {sorted(missing)}"


@pytest.mark.parametrize(("name", "fen"), _WIDE_ROOTS)
def test_the_default_topk_really_does_drop_legal_root_moves(
    tmp_path: Path, name: str, fen: str,
) -> None:
    """⚑ THE MUTANT, run as a test: production's topk 16 DOES lose root moves.

    Without this the coverage test above is unfalsifiable -- it would pass on a
    search that never had a problem. Here the same instrument, with `topk` at
    production's 16 and the per-position budget off, must FAIL to cover the
    root. Measured on the start position: a2a3, b1c3, b2b4, c2c4 are never
    evaluated, at every sims budget.
    """
    board = chess.Board(fen)
    cfg = _config(
        tmp_path, all_root_moves=False, topk=GEN.DEFAULT_TOPK, sims=32,
    )
    hit, missing = _evaluated_root_moves(board, cfg)
    assert missing, (
        f"{name}: topk={GEN.DEFAULT_TOPK} covered ALL "
        f"{board.legal_moves.count()} root moves, so the coverage test above "
        "proves nothing"
    )
    assert len(hit) == GEN.DEFAULT_TOPK, (
        f"{name}: expected exactly topk={GEN.DEFAULT_TOPK} candidates, got "
        f"{len(hit)}"
    )


def test_the_candidate_set_is_capped_by_the_budget_not_only_by_topk(
    tmp_path: Path,
) -> None:
    """⚑ Raising topk alone is NOT the fix, and that is why the budget is scaled.

    The measurement the design rests on: at an unbounded topk the candidate
    count is still capped at ``sims / 2``. A reader who "fixed" the trap by
    bumping topk and leaving ``--sims`` at 32 would still be dropping two thirds
    of a wide root, with a config that looks correct.
    """
    board = chess.Board(_WIDE_ROOTS[1][1])  # 49 legal moves
    legal = board.legal_moves.count()
    covered: dict[int, int] = {}
    for sims in (32, 64, 128):
        cfg = _config(
            tmp_path, all_root_moves=False, topk=GEN.MAX_LEGAL_MOVES, sims=sims,
        )
        hit, _missing = _evaluated_root_moves(board, cfg)
        covered[sims] = len(hit)
    assert covered[32] == 16, covered
    assert covered[64] == 32, covered
    assert covered[128] == legal, covered
    # ... and the per-position budget reaches the same place without a guess.
    scaled = _config(tmp_path, all_root_moves=True, topk=GEN.MAX_LEGAL_MOVES, sims=32)
    assert GEN.root_simulation_budget(legal, cfg=scaled) == 2 * legal
    hit, missing = _evaluated_root_moves(board, scaled)
    assert not missing
    assert len(hit) == legal


def test_root_simulation_budget_is_a_floor_not_a_replacement() -> None:
    """``--sims`` still binds when it is the larger of the two."""
    off = GEN.GenConfig(out_dir=Path("/tmp"), sims=32, all_root_moves=False)
    on = GEN.GenConfig(out_dir=Path("/tmp"), sims=32, all_root_moves=True)
    assert GEN.root_simulation_budget(40, cfg=off) == 32
    assert GEN.root_simulation_budget(40, cfg=on) == 80
    # A narrow root does not shrink below the floor.
    assert GEN.root_simulation_budget(3, cfg=on) == 32
    assert GEN.root_simulation_budget(1, cfg=on) == 32


def test_all_root_moves_refuses_a_topk_that_would_cap_it(tmp_path: Path) -> None:
    """A contradiction is REFUSED, not resolved behind the operator's back.

    `--all-root-moves` promises complete coverage and `topk` caps it. Silently
    raising topk would honour the promise while ignoring a value the operator
    typed; silently keeping it would publish a guarantee the search does not
    keep. Same rule as the merged gumbel-override refusal.
    """
    cfg = _config(tmp_path, all_root_moves=True, topk=16)
    with pytest.raises(ValueError, match="--all-root-moves needs --topk"):
        GEN.build_gumbel_config(cfg)
    # And it is not refusing everything: the resolved default is accepted.
    GEN.build_gumbel_config(_config(tmp_path, all_root_moves=True, topk=218))


def test_the_cli_turns_all_root_moves_on_for_sfnnue_only() -> None:
    """The default follows the value source, and topk follows the default."""
    parser = GEN.build_parser()

    sf = GEN.config_from_args(parser.parse_args(
        ["--out-dir", "/tmp/x", "--value-source", "sfnnue"],
    ))
    assert sf.all_root_moves is True
    assert sf.topk == GEN.MAX_LEGAL_MOVES

    zero = GEN.config_from_args(parser.parse_args(["--out-dir", "/tmp/x"]))
    assert zero.all_root_moves is False
    assert zero.topk == GEN.DEFAULT_TOPK

    # Both overrides are honoured, in both directions.
    forced_off = GEN.config_from_args(parser.parse_args(
        ["--out-dir", "/tmp/x", "--value-source", "sfnnue", "--no-all-root-moves"],
    ))
    assert forced_off.all_root_moves is False
    assert forced_off.topk == GEN.DEFAULT_TOPK

    forced_on = GEN.config_from_args(parser.parse_args(
        ["--out-dir", "/tmp/x", "--all-root-moves"],
    ))
    assert forced_on.all_root_moves is True
    assert forced_on.topk == GEN.MAX_LEGAL_MOVES

    # An explicit --topk is carried through so the refusal can see it.
    explicit = GEN.config_from_args(parser.parse_args(
        ["--out-dir", "/tmp/x", "--value-source", "sfnnue", "--topk", "16"],
    ))
    assert explicit.topk == 16
    with pytest.raises(ValueError, match="--all-root-moves needs --topk"):
        GEN.build_gumbel_config(explicit)


def test_the_realized_root_budget_is_announced_and_measured(tmp_path: Path) -> None:
    """``--sims`` stops being the realized budget, so the realized one is published.

    A flag that silently changes what a recorded number means is the defect this
    repo keeps re-finding. Both the decision and its cost are in the sidecar.
    """
    cfg = _config(
        tmp_path, all_root_moves=True, topk=GEN.MAX_LEGAL_MOVES, sims=8,
        games=1, max_plies=6, shard_size=500,
    )
    summary = GEN.generate(cfg)
    realized = summary["realized_per_worker"][0]
    assert realized["all_root_moves"] is True
    assert realized["topk"] == GEN.MAX_LEGAL_MOVES
    assert realized["root_sims_per_legal_move"] == GEN.ROOT_SIMS_PER_LEGAL_MOVE

    budget = summary["root_budget"]
    assert budget["plies"] > 0
    # The floor was 8 and every real root has more than 4 legal moves, so the
    # realized budget must have exceeded it — i.e. the scaling actually ran.
    assert budget["sims_mean"] > 8.0
    assert budget["sims_max"] >= 2.0 * budget["legal_moves_max"]

    off = GEN.generate(_config(
        tmp_path / "off", all_root_moves=False, sims=8, games=1, max_plies=6,
        shard_size=500,
    ))
    assert off["realized_per_worker"][0]["all_root_moves"] is False
    assert off["realized_per_worker"][0]["root_sims_per_legal_move"] == 0
    assert off["root_budget"]["sims_max"] == 8.0
