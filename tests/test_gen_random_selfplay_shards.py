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
