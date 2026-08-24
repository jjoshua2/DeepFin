"""scripts/gen_random_selfplay_shards.py — the CPU generation-zero generator.

The load-bearing claims, and how each is tested rather than asserted:

* the shards are accepted by the REAL replay load path (``DiskReplayBuffer``),
  not by a hand-rolled reader;
* ``wdl_target``'s POV is the side to move AT THAT PLY, checked on a game whose
  moves are forced end to end so the expected labels come off the board;
* every flag the operator can set reaches the game loop, checked by making the
  search's own behaviour depend on it (mutation table in the PR).
"""
from __future__ import annotations

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
from chess_anti_engine.moves import COMPACT_POLICY_SIZE, move_to_index
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


def test_policy_target_is_a_visit_distribution_over_the_legal_moves(
    tmp_path: Path,
) -> None:
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
    # A search with a budget is not a one-hot: with 32 sims most rows must place
    # mass on more than one move, or the "visit distribution" name is a lie.
    support = (policy > 0.0).sum(axis=1)
    assert float((support > 1).mean()) > 0.5, f"median support {np.median(support)}"


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


def _fools_mate_script() -> dict[bytes, int]:
    """1. f3 e5 2. g4 Qh4# — four plies, mate, White to move on plies 0 and 2."""
    board = chess.Board()
    script: dict[bytes, int] = {}
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
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


def test_wdl_target_pov_is_the_side_to_move_at_that_ply(tmp_path: Path) -> None:
    cfg = _config(tmp_path, sims=16, max_plies=20)
    outcome = GEN.play_game(
        cfg=cfg,
        gcfg=GEN.build_gumbel_config(cfg),
        evaluator=_ScriptedPriorEvaluator(
            _fools_mate_script(), value_source="zero", expected_planes=_PLANES,
        ),
        rng=np.random.default_rng(1),
        opening_cfg=OpeningConfig(),
    )
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


def test_truncated_games_are_labelled_draws_not_losses(tmp_path: Path) -> None:
    """`"*"` is what a max-plies game returns; production adjudicates it with
    Stockfish and there is none here, so it must land on the draw label."""
    cfg = _config(tmp_path, max_plies=6, sims=8)
    outcome, _ev = _play_one(cfg, seed=3)
    if outcome.termination != "max_plies":  # pragma: no cover - seed-dependent
        pytest.skip("this seed finished the game inside the ply cap")
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
    assert "sims" not in realized, "realized must not echo the parser's names"


def test_sims_reaches_the_search(tmp_path: Path) -> None:
    """The search must ASK the evaluator for more positions at a bigger budget.

    Rows per ply, not rows per game: a different budget plays a different game,
    so a raw total would move even if the budget were being dropped.
    """
    small, ev_small = _play_one(_config(tmp_path, sims=4, max_plies=24), seed=5)
    large, ev_large = _play_one(_config(tmp_path, sims=64, max_plies=24), seed=5)
    per_ply_small = ev_small.eval_rows / max(1, small.plies)
    per_ply_large = ev_large.eval_rows / max(1, large.plies)
    assert per_ply_large > 3.0 * per_ply_small, (
        f"sims did not reach the search: {per_ply_small:.1f} vs {per_ply_large:.1f} "
        "evaluated positions per ply"
    )


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
    assert GEN.build_gumbel_config(_config(tmp_path)).policy_temp == 1.0


def test_history_encoding_reaches_the_encoder(tmp_path: Path) -> None:
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
    assert summary["required_run_config"] == {
        "sf_wdl_frac": 0.0, "sf_wdl_frac_floor": 0.0, "search_wdl_frac": 0.0,
    }
    assert summary["games"] == 1
    assert summary["plies"]["n"] == 1.0
    assert sum(summary["plies_histogram"].values()) == 1
