"""The native NNUE arms inside the CPU shard generator.

The load-bearing claims, and how each is tested rather than asserted:

* the shards are SCHEMA-IDENTICAL to the ones the pure gen-0 source writes --
  compared array by array, dtype by dtype, and meta field by meta field, so the
  deep-SF ruler can score a native cell and a banked UCI anchor in one pass;
* the boards the arm evaluates really are the batch the tree encoded -- proved
  by re-deriving every leaf's planes from the board the tree handed back, which
  is the ONE thing that cannot be checked from the values;
* every arm knob is live, each with an observation that CHANGES when the knob
  does -- the mutation table is in the PR;
* the mate band is not laundered through the centipawn slope, which is audit
  N1's defect class in a new scale.

⚑ THE SYNTHETIC PACK MAKES THE VALUES READABLE. ``bucket_pack`` is all zeros
except for the final bias of each layer stack, so a static evaluation collapses
to ``(bucket + 1) * 100`` with ``bucket = (piece_count - 1) // 4``. The value is
therefore a hand-computable function of piece count -- position-dependent, so it
moves the search, and predictable, so a test can say what it should be. The real
big net is a runtime artifact and is never required by a test.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.mcts import _mcts_tree
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.moves import move_to_index
from chess_anti_engine.nnue import _nnue_ext
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.stockfish.wdl import cp_to_wdl
from scripts import nnue_parse
from tests.script_loading import load_script_module
from tests.test_nnue_native_eval import write_synthetic_pack

GEN = load_script_module("gen_random_selfplay_shards.py")

_PLANES = 175
_HIST = "lc0_root_legacy_meta"
_EXTRA = "v2_threats"

# A quiet middlegame with a wide root (the all-root-moves coverage test needs
# more than 16 legal moves) and no side to move in check.
WIDE_FEN = "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 6 9"
# ⚑ FIVE pieces, with a capture available. Under the bucket pack the evaluation
# is (piece_count - 1) // 4 + 1 hundreds, so a capture here crosses a bucket
# (5 -> 4 is bucket 1 -> 0) and the leaf values genuinely DIFFER. A four-piece
# position would not: every leaf would score the same and any test that needed
# the value to move the target would pass on a constant.
CAPTURE_FEN = "8/8/4k3/4p3/3P4/7P/8/4K3 w - - 0 1"
# Mate in one (Ra8#): quiescence that is allowed to try CHECKS finds it and
# returns a mate score, where the static arm stands pat on the bucket value.
CHECK_CHAIN_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
# ⚑ A root with a legal move that DRAWS IMMEDIATELY, for the terminal-shortcut
# test. K+N vs K+B: Nxe5 leaves K+N vs K, which is insufficient material, so the
# child is game-over-drawn. The root itself is NOT (KN vs KB is not insufficient
# under python-chess), it has 12 legal moves, and the bucket pack scores it
# positive for the mover -- the three conditions `allow_terminal_root_shortcuts`
# needs before it prunes anything.
DRAW_CAPTURE_FEN = "4k3/8/8/4b3/8/5N2/8/4K3 w - - 0 1"
DRAW_CAPTURE_MOVE = "f3e5"


@pytest.fixture(autouse=True)
def _rep_fix_on() -> Iterator[None]:
    """The generator encodes with the production repetition fix on."""
    previous = rep_fix.current()
    rep_fix.apply(True, boards_discarded=True)
    yield
    if previous is not None and bool(previous) is not True:
        rep_fix.apply(bool(previous), boards_discarded=True)


@pytest.fixture(scope="module")
def bucket_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """fc2 bias = (bucket + 1) * 1600, everything else zero.

    Redefined here rather than imported: a fixture is resolved by NAME within a
    module, so importing one across files shadows the import and reads as dead
    code to every linter that looks.
    """
    path = tmp_path_factory.mktemp("gennnue") / "bucket.pack"
    write_synthetic_pack(
        path,
        {"fc2_bias": [(b, (b + 1) * 1600) for b in range(nnue_parse.PSQT_BUCKETS)]},
    )
    return path


def _cfg(out_dir: Path, pack: Path | None = None, **overrides: Any) -> Any:
    """A GenConfig for a short run, native arm by default when a pack is given."""
    base: dict[str, Any] = {
        "out_dir": out_dir,
        "games": 1,
        "workers": 1,
        "sims": 8,
        "max_plies": 24,
        "shard_size": 10_000,
        "seed": 4242,
        "nice": 0,
        "input_history_encoding": _HIST,
        "input_extra_features": _EXTRA,
        "history_rep_fix": True,
    }
    if pack is not None:
        base.update(
            value_source=GEN.VALUE_SOURCE_NNUE_STATIC,
            nnue_pack=pack,
            all_root_moves=True,
            topk=GEN.MAX_LEGAL_MOVES,
            nnue_resolver_max_depth=_nnue_ext.RESOLVER_MAX_DEPTH,
        )
    base.update(overrides)
    # ⚑ ARM-SCOPED, mirroring `resolve_arm_knob_defaults`: the quiescence pair
    # is a number only under `nnue-qsearch` and stays None under `nnue-static`,
    # which never reads it. A fixture that filled both for either arm would
    # make every static-arm test run against the shape the P1-1 fix removed.
    if base.get("value_source") == GEN.VALUE_SOURCE_NNUE_QSEARCH:
        base.setdefault("nnue_qsearch_max_ply", _nnue_ext.QSEARCH_MAX_PLY)
        base.setdefault("nnue_qsearch_check_plies", _nnue_ext.QSEARCH_CHECK_PLIES)
    return GEN.GenConfig(**base)


def _run(cfg: Any, *, worker_id: int = 0) -> Any:
    return GEN.run_worker(
        GEN.WorkerSpec(
            cfg=cfg, worker_id=worker_id, games=int(cfg.games),
            seed=int(cfg.seed), shard_index_start=0,
        ),
    )


def _evaluator(cfg: Any) -> Any:
    return GEN.UniformPriorEvaluator(
        value_source=cfg.value_source,
        expected_planes=_PLANES,
        random_salt=int(cfg.seed),
        nnue_source=GEN.build_nnue_source(cfg),
        input_history_encoding=str(cfg.input_history_encoding),
        input_extra_features=str(cfg.input_extra_features),
    )


def _fen_openings(tmp_path: Path, fen: str) -> OpeningConfig:
    """Start every game from one FEN, through the shared opening sampler.

    The generator itself cannot reach this branch (``build_opening_config``
    explains why), so a test builds the ``OpeningConfig`` directly. That makes
    the FEN a FIXTURE rather than a setting.
    """
    seeds = tmp_path / "seeds.txt"
    seeds.write_text(fen + "\n")
    return OpeningConfig(
        opening_book_prob=0.0,
        opening_fen_list_path=str(seeds),
        opening_fen_prob=1.0,
    )


def _play(cfg: Any, *, opening: OpeningConfig | None = None, seed: int = 7) -> Any:
    """One game through the real loop, returning (outcome, evaluator)."""
    evaluator = _evaluator(cfg)
    outcome = GEN.play_game(
        cfg=cfg,
        gcfg=GEN.build_gumbel_config(cfg),
        evaluator=evaluator,
        rng=np.random.default_rng(seed),
        opening_cfg=OpeningConfig() if opening is None else opening,
        budget=GEN.RootBudgetStats(),
    )
    return outcome, evaluator


def _cboard(fen: str) -> CBoard:
    return CBoard.from_board(chess.Board(fen))


# ===========================================================================
# Schema: a native shard and a pure gen-0 shard must be the same thing
# ===========================================================================


def _one_shard(cfg: Any) -> tuple[dict[str, np.ndarray], Any]:
    _run(cfg)
    paths = sorted(iter_shard_paths(cfg.out_dir))
    assert paths, "the run wrote no shard"
    return load_shard_arrays(paths[0])


# ⚑⚑ THE BANKED ANCHOR'S SCHEMA, FROZEN AS A LITERAL. Read off
# data/gen0/bench_anchors/gen0_bench_32/shard_000000.zarr (the sfnnue nodes-32
# UCI anchor, 2026-08) and written down here on purpose.
#
# The comparison test below generates a reference shard from this same script
# and diffs the native one against it. That catches an arm-specific divergence
# and NOTHING ELSE: both sides come out of the same `samples_to_arrays`, so a
# change to the shared writer moves the reference and the native shard
# together, the diff stays empty, and the shards silently stop matching the
# corpus they must be scored beside. A frozen copy is the only side of the
# comparison that cannot move with the code.
#
# 23 arrays: 19 per-row columns and 4 shard-level SCALARS (the underscored
# ones, which are 0-d and describe the whole shard). Changing this literal means
# the native cells are no longer schema-compatible with the banked anchors,
# which is a readout-invalidating event and must be a deliberate edit with a new
# anchor set behind it.
#
# Value is (dtype string, ndim, per-row shape). ndim is carried explicitly
# because it is the axis a scalar and a length-1 column differ on, and a schema
# that recorded only the trailing shape would call them equal.
ANCHOR_SCHEMA: dict[str, tuple[str, int, tuple[int, ...]]] = {
    "_history_rep_fix": ("<U4", 0, ()),
    "_input_history_encoding": ("<U20", 0, ()),
    "_policy_encoding": ("<U8", 0, ()),
    "_policy_size": ("int32", 0, ()),
    "game_id": ("int64", 1, ()),
    "has_game_id": ("uint8", 1, ()),
    "has_is_network_turn": ("uint8", 1, ()),
    "has_is_selfplay": ("uint8", 1, ()),
    "has_legal_mask": ("uint8", 1, ()),
    "has_moves_left": ("uint8", 1, ()),
    "has_opening_source_code": ("uint8", 1, ()),
    "has_ply_index": ("uint8", 1, ()),
    "has_policy": ("uint8", 1, ()),
    "is_network_turn": ("uint8", 1, ()),
    "is_selfplay": ("uint8", 1, ()),
    "legal_mask": ("uint8", 2, (1858,)),
    "moves_left": ("float16", 1, ()),
    "opening_source_code": ("uint8", 1, ()),
    "ply_index": ("int32", 1, ()),
    "policy_target": ("float16", 2, (1858,)),
    "priority": ("float32", 1, ()),
    "wdl_target": ("int8", 1, ()),
    "x": ("float16", 4, (175, 8, 8)),
}


def test_native_shard_matches_the_frozen_banked_anchor_schema(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """The native shard against a schema that CANNOT move when the code does.

    ⚑ This is the half the generated-reference comparison structurally cannot
    do. There, both sides are produced by the writer under test; here the
    expectation is a literal transcribed from the banked UCI anchor the deep-SF
    ruler already scores. A change to `samples_to_arrays` that renames a column
    or widens a dtype passes the comparison test and fails this one.

    The string dtypes are compared by KIND and width because numpy renders a
    fixed-width unicode dtype byte-order-dependently; everything else is exact.
    """
    native, _meta = _one_shard(_cfg(tmp_path / "nat", bucket_pack))
    assert set(native) == set(ANCHOR_SCHEMA), (
        "the native shard's array set differs from the banked anchor's: "
        f"extra={sorted(set(native) - set(ANCHOR_SCHEMA))} "
        f"missing={sorted(set(ANCHOR_SCHEMA) - set(native))}"
    )
    for name, (dtype_str, ndim, row_shape) in sorted(ANCHOR_SCHEMA.items()):
        arr = np.asarray(native[name])
        expected = np.dtype(dtype_str)
        assert (arr.dtype.kind, arr.dtype.itemsize) == (
            expected.kind, expected.itemsize,
        ), f"{name}: {arr.dtype} != {expected}"
        assert arr.ndim == ndim, name
        assert arr.shape[1:] == row_shape, name
        if ndim:
            assert arr.shape[0] > 0, name


def test_the_frozen_schema_is_the_banked_anchors_own() -> None:
    """The literal above, checked against the real anchor when it is present.

    ⚑ A frozen literal is only worth what its transcription is worth, and
    nothing in CI can read a 100 GB corpus. So: skipped when the anchor set is
    not on this machine, and an exact comparison when it is. That makes the
    transcription falsifiable on the box the readout runs on, which is the box
    that matters, without making the suite depend on a runtime artifact.
    """
    anchor = Path("data/gen0/bench_anchors/gen0_bench_32")
    if not anchor.is_dir():
        pytest.skip("banked UCI anchor set is not present on this machine")
    paths = sorted(iter_shard_paths(anchor))
    if not paths:
        pytest.skip("banked UCI anchor directory holds no shard")
    arrays, _meta = load_shard_arrays(paths[0])
    observed = {
        name: (
            np.asarray(arr).dtype.str, np.asarray(arr).ndim,
            np.asarray(arr).shape[1:],
        )
        for name, arr in arrays.items()
    }
    expected = {
        name: (np.dtype(dtype).str, ndim, shape)
        for name, (dtype, ndim, shape) in ANCHOR_SCHEMA.items()
    }
    assert observed == expected


def test_native_shard_schema_matches_a_gen0_reference_shard(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """Field by field, not "it loads": the ruler reads both sets in one pass.

    A missing optional array, a widened dtype or a changed row shape would all
    still load; what breaks is the JOINT scoring pass over the native cells and
    the banked UCI anchors, and it breaks silently by dropping a column.

    ⚑ Both sides of THIS comparison are generated by the code under test, so it
    can only find an arm-specific divergence. The frozen-literal test above is
    what pins the schema against the banked corpus.
    """
    reference, ref_meta = _one_shard(
        _cfg(tmp_path / "ref", value_source=GEN.VALUE_SOURCE_ZERO),
    )
    native, nat_meta = _one_shard(_cfg(tmp_path / "nat", bucket_pack))

    assert set(native) == set(reference)
    for name in sorted(reference):
        ref_arr, nat_arr = np.asarray(reference[name]), np.asarray(native[name])
        assert nat_arr.dtype == ref_arr.dtype, name
        assert nat_arr.shape[1:] == ref_arr.shape[1:], name
        assert nat_arr.ndim == ref_arr.ndim, name
    for field in (
        "version", "input_history_encoding", "history_rep_fix", "policy_encoding",
        "policy_size",
    ):
        assert nat_meta.get(field) == ref_meta.get(field), field
    # ⚑ The pack sha does NOT go into model_sha256: downstream that field names
    # the trained net whose rows these are, and a corpus no net produced must
    # not claim one. Provenance lives in the sidecar.
    assert nat_meta.get("model_sha256") is None
    assert nat_meta.get("model_step") is None


def test_native_shard_loads_through_the_real_replay_buffer(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    cfg = _cfg(tmp_path / "out", bucket_pack)
    result = _run(cfg)
    buf = DiskReplayBuffer(
        10**9, shard_dir=cfg.out_dir, rng=np.random.default_rng(0),
        read_only=True, input_planes=_PLANES,
    )
    try:
        # `_try_load_shard` swallows load failures, so a rejected shard shows up
        # as an EMPTY buffer, never as a raise. Count rows, not exceptions.
        assert len(buf) == result.rows > 0
    finally:
        buf.close()


def test_native_shard_carries_no_stockfish_or_search_value_fields(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """The arm supplies SEARCH values; none of them are stored as labels.

    Writing the arm's leaf value into ``search_wdl`` would launder an
    evaluation into a field every reader treats as the search's own estimate --
    the same rejection the pure sources already carry, and it matters more here
    because the value is now a real one.
    """
    arrays, _meta = _one_shard(_cfg(tmp_path / "out", bucket_pack))
    for name in arrays:
        assert not name.startswith("sf_"), name
        assert "search_wdl" not in name, name


# ===========================================================================
# Binding: the boards the arm scores ARE the batch the tree encoded
# ===========================================================================


def test_pending_leaf_cboards_are_the_encoded_batch(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """Re-derive EVERY leaf's planes from the board the tree handed back.

    This is the one property no value can reveal: an evaluator fed a plausible
    but wrong set of positions returns plausible numbers. The generator itself
    checks row 0 once per worker; this checks every row of every batch, which is
    what makes the cheap runtime check a sample of a verified property rather
    than a hope.
    """
    cfg = _cfg(tmp_path / "out", bucket_pack)
    evaluator = _evaluator(cfg)
    checked = {"batches": 0, "rows": 0}
    inner = evaluator._leaf_boards

    def spy(n_rows: int) -> list[CBoard]:
        boards = inner(n_rows)
        checked["batches"] += 1
        checked["rows"] += len(boards)
        return boards

    evaluator._leaf_boards = spy
    captured: list[tuple[list[CBoard], np.ndarray]] = []
    original_eval = evaluator.evaluate_encoded

    def record(x: np.ndarray, relations: np.ndarray | None = None) -> Any:
        boards = evaluator._tree.pending_leaf_cboards()
        captured.append((boards, np.asarray(x)[: len(boards)].copy()))
        return original_eval(x, relations)

    evaluator.evaluate_encoded = record
    GEN.play_game(
        cfg=cfg, gcfg=GEN.build_gumbel_config(cfg), evaluator=evaluator,
        rng=np.random.default_rng(7), opening_cfg=OpeningConfig(),
    )
    assert checked["batches"] > 0
    assert checked["rows"] > 0
    total = 0
    for boards, planes in captured:
        for i, board in enumerate(boards):
            expected = encode_cboard(
                board,
                input_history_encoding=_HIST,
                input_extra_features=_EXTRA,
            )
            assert np.array_equal(
                np.asarray(expected, dtype=np.float32), planes[i],
            ), f"leaf {i} of a {len(boards)}-board batch is not the encoded row"
            total += 1
    assert total == checked["rows"]


def test_leaf_binding_guard_fires_when_the_boards_are_not_the_batch(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """The one-shot runtime check must be able to FAIL, not just to pass."""
    cfg = _cfg(tmp_path / "out", bucket_pack)
    evaluator = _evaluator(cfg)
    boards = [_cboard(WIDE_FEN)]
    wrong = np.zeros((1, _PLANES, 8, 8), dtype=np.float32)
    with pytest.raises(RuntimeError, match="binding check FAILED"):
        evaluator._check_binding_once(boards, wrong)


def test_leaf_binding_check_passes_on_the_real_planes(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """...and must PASS on the matching ones, or the guard is just a raise."""
    cfg = _cfg(tmp_path / "out", bucket_pack)
    evaluator = _evaluator(cfg)
    board = _cboard(WIDE_FEN)
    planes = np.asarray(
        encode_cboard(
            board, input_history_encoding=_HIST, input_extra_features=_EXTRA,
        ),
        dtype=np.float32,
    )[None, ...]
    evaluator._check_binding_once([board], planes)
    assert evaluator.nnue_source.stats.binding_checks == 1


def test_evaluator_refuses_a_batch_with_no_bound_tree(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    cfg = _cfg(tmp_path / "out", bucket_pack)
    evaluator = _evaluator(cfg)
    with pytest.raises(RuntimeError, match="no bound tree"):
        evaluator.evaluate_encoded(np.zeros((2, _PLANES, 8, 8), dtype=np.float32))


def test_evaluator_refuses_a_tree_with_no_pending_batch(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """A tree that is not mid-search cannot have produced these planes."""
    cfg = _cfg(tmp_path / "out", bucket_pack)
    evaluator = _evaluator(cfg)
    evaluator.bind_tree(MCTSTree())
    with pytest.raises(RuntimeError, match="phase != 1"):
        evaluator.evaluate_encoded(np.zeros((2, _PLANES, 8, 8), dtype=np.float32))


def test_pending_leaf_cboards_keeps_the_leaves_tb_filtering_drops(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """⚑ The reason this method exists rather than reusing the Syzygy one.

    ``get_pending_tb_leaves`` filters to ``castling == 0`` and few enough
    pieces. From the start position every leaf still has castling rights, so
    that method returns NOTHING while the batch is full -- an evaluator built on
    it would have scored an empty set and left every real leaf at the padding
    value.
    """
    cfg = _cfg(tmp_path / "out", bucket_pack)
    evaluator = _evaluator(cfg)
    seen: dict[str, int] = {}
    original_eval = evaluator.evaluate_encoded

    def record(x: np.ndarray, relations: np.ndarray | None = None) -> Any:
        tree = evaluator._tree
        if not seen:
            seen["all"] = len(tree.pending_leaf_cboards())
            seen["tb"] = len(tree.get_pending_tb_leaves(32)[1])
        return original_eval(x, relations)

    evaluator.evaluate_encoded = record
    GEN.play_game(
        cfg=cfg, gcfg=GEN.build_gumbel_config(cfg), evaluator=evaluator,
        rng=np.random.default_rng(7), opening_cfg=OpeningConfig(),
    )
    assert seen["all"] > 0
    assert seen["tb"] < seen["all"]


# ===========================================================================
# Knob liveness — each knob, with an observation that moves when it does
# ===========================================================================


def _arm_values(cfg: Any, fens: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    source = GEN.build_nnue_source(cfg)
    q = source.q_for_boards([_cboard(f) for f in fens])
    source.refresh_context_stats()
    return q, dict(source.stats.context)


def test_arm_choice_reaches_the_evaluation(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """The two arms are two different evaluations, not two labels for one.

    ⚑ The position is the MATE-IN-ONE, not a plain capture. Under the bucket
    pack every evaluation is positive for whoever is to move, so a capture
    always looks worse than standing pat and quiescence over captures alone
    returns the static value -- a test built on one would report "the arms
    agree" about a knob that works.
    """
    fens = [CHECK_CHAIN_FEN, WIDE_FEN]
    static_q, static_ctx = _arm_values(
        _cfg(tmp_path / "s", bucket_pack, value_source=GEN.VALUE_SOURCE_NNUE_STATIC),
        fens,
    )
    qsearch_q, qsearch_ctx = _arm_values(
        _cfg(tmp_path / "q", bucket_pack, value_source=GEN.VALUE_SOURCE_NNUE_QSEARCH),
        fens,
    )
    assert static_ctx["qnodes"] == 0
    assert qsearch_ctx["qnodes"] > 0
    assert not np.array_equal(static_q, qsearch_q)


def test_qsearch_max_ply_zero_is_the_arms_own_negative_control(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """With quiescence off the two arms agree exactly; with it on they do not.

    ⚑⚑ AND THE VALUE CHANNEL ALONE CANNOT SEE THIS KNOB, which is why the work
    channel is asserted too. Under the bucket pack every evaluation is positive
    for whoever is to move, so a capture always scores WORSE than standing pat
    and quiescence-over-captures can never change a value. Measured: at
    ``check_plies=0`` the values at ``max_ply`` 0 and 4 are identical (300/800/
    200 on these three positions) while the quiescence node count goes 3 -> 248.
    A version of this test that compared only values passed against a mutant
    that pinned ``qsearch_max_ply`` to 4 -- it was in the sweep and it was
    MISSED, which is how this paragraph exists.
    """
    fens = [CHECK_CHAIN_FEN, WIDE_FEN, CAPTURE_FEN]
    static_q, static_ctx = _arm_values(
        _cfg(tmp_path / "s", bucket_pack, value_source=GEN.VALUE_SOURCE_NNUE_STATIC),
        fens,
    )
    off_q, off_ctx = _arm_values(
        _cfg(
            tmp_path / "q0", bucket_pack,
            value_source=GEN.VALUE_SOURCE_NNUE_QSEARCH,
            nnue_qsearch_max_ply=0, nnue_qsearch_check_plies=0,
        ),
        fens,
    )
    _capture_q, capture_ctx = _arm_values(
        _cfg(
            tmp_path / "q4c0", bucket_pack,
            value_source=GEN.VALUE_SOURCE_NNUE_QSEARCH,
            nnue_qsearch_max_ply=4, nnue_qsearch_check_plies=0,
        ),
        fens,
    )
    on_q, _on_ctx = _arm_values(
        _cfg(
            tmp_path / "q4", bucket_pack,
            value_source=GEN.VALUE_SOURCE_NNUE_QSEARCH,
            nnue_qsearch_max_ply=4, nnue_qsearch_check_plies=1,
        ),
        fens,
    )
    assert static_ctx["qnodes"] == 0
    assert np.array_equal(static_q, off_q)
    assert not np.array_equal(static_q, on_q)
    # The ply budget is live even where it cannot move a value with this pack.
    assert off_ctx["qsearch_max_ply"] == 0
    assert capture_ctx["qsearch_max_ply"] == 4
    assert capture_ctx["qnodes"] > off_ctx["qnodes"]


def test_qsearch_check_plies_changes_the_work_the_arm_does(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """Measured on the real net at 16.3x the quiescence nodes; here: it moves."""
    fens = [CHECK_CHAIN_FEN, WIDE_FEN, CAPTURE_FEN]
    _q0, ctx0 = _arm_values(
        _cfg(
            tmp_path / "c0", bucket_pack,
            value_source=GEN.VALUE_SOURCE_NNUE_QSEARCH,
            nnue_qsearch_max_ply=4, nnue_qsearch_check_plies=0,
        ),
        fens,
    )
    _q2, ctx2 = _arm_values(
        _cfg(
            tmp_path / "c2", bucket_pack,
            value_source=GEN.VALUE_SOURCE_NNUE_QSEARCH,
            nnue_qsearch_max_ply=4, nnue_qsearch_check_plies=2,
        ),
        fens,
    )
    assert ctx0["qsearch_check_plies"] == 0
    assert ctx2["qsearch_check_plies"] == 2
    assert ctx2["qnodes"] > ctx0["qnodes"]


def test_resolver_max_depth_reaches_the_context_and_can_bind(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """A cap of 1 must actually CUT OFF, or the counter cannot fire at all.

    ``depth_cutoffs`` is the backstop's own alarm. A test that only read the
    configured number back would pass against a knob that was stored and never
    consulted -- this one makes the cap change what the resolver produces.
    """
    fens = [CHECK_CHAIN_FEN, WIDE_FEN, CAPTURE_FEN]
    _deep_q, deep_ctx = _arm_values(
        _cfg(
            tmp_path / "d32", bucket_pack,
            value_source=GEN.VALUE_SOURCE_NNUE_QSEARCH,
            nnue_resolver_max_depth=32, nnue_qsearch_max_ply=4,
            nnue_qsearch_check_plies=2,
        ),
        fens,
    )
    # ⚑ qsearch_max_ply is itself bounded by resolver_max_depth, so the shallow
    # arm has to lower both -- the point is the CAP binding, not the ply budget.
    _shallow_q, shallow_ctx = _arm_values(
        _cfg(
            tmp_path / "d1", bucket_pack,
            value_source=GEN.VALUE_SOURCE_NNUE_QSEARCH,
            nnue_resolver_max_depth=1, nnue_qsearch_max_ply=1,
            nnue_qsearch_check_plies=1,
        ),
        fens,
    )
    assert deep_ctx["resolver_max_depth"] == 32
    assert shallow_ctx["resolver_max_depth"] == 1
    assert deep_ctx["depth_cutoffs"] == 0
    assert shallow_ctx["depth_cutoffs"] > 0


def test_arm_knobs_are_read_back_from_the_context_not_the_setter(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """The realized triple is the CONTEXT's, and the ORDER that makes it so.

    ⚑ ``set_arm_config`` is read at ``init()``. A source that opened its context
    and only then set the configuration would report the triple it asked for
    while the batches ran on whatever was in the process globals -- this repo's
    signature defect with a correct-looking realized line on top. The test
    leaves a DIFFERENT triple in the globals first, so that mistake changes
    what ``arm_stats`` says.

    ⚑ Honest limit: with the setter VALIDATING rather than clamping, the
    producer's copy and the consumer's agree whenever the order is right, so
    this cannot separate "read from the setter" from "read from the context"
    on its own. What it does pin is that they must AGREE -- and under the
    ordering defect they do not.
    """
    _nnue_ext.set_arm_config(20, 2, 1)  # a stale triple, deliberately different
    cfg = _cfg(
        tmp_path / "out", bucket_pack,
        value_source=GEN.VALUE_SOURCE_NNUE_QSEARCH,
        nnue_resolver_max_depth=7, nnue_qsearch_max_ply=3,
        nnue_qsearch_check_plies=2,
    )
    source = GEN.build_nnue_source(cfg)
    live = dict(_nnue_ext.arm_stats(source._handle))
    assert live["resolver_max_depth"] == 7
    assert live["qsearch_max_ply"] == 3
    assert live["qsearch_check_plies"] == 2
    assert source.realized == {key: live[key] for key in source.realized}
    assert source.realized == {
        "resolver_max_depth": 7, "qsearch_max_ply": 3, "qsearch_check_plies": 2,
    }
    # ...and a LATER global change does not retune the running context, which is
    # exactly what makes reading the globals the wrong answer.
    _nnue_ext.set_arm_config(31, 8, 5)
    assert dict(_nnue_ext.arm_stats(source._handle))["qsearch_max_ply"] == 3
    source.refresh_context_stats()
    assert source.stats.context["qsearch_max_ply"] == 3


def test_pack_reaches_the_arm_and_is_named_by_two_distinct_hashes(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """⚑ The FILE digest and the EMBEDDED source digest are different numbers.

    ``source_sha256`` returns what the packer wrote into the header -- the
    ``.nnue`` the pack was built FROM. It is provenance, not identity: it
    survives a truncated, corrupted or hand-edited payload unchanged. Only a
    hash of the pack's own bytes says which weights ran, so both are reported
    and this test pins that they are not the same value.
    """
    cfg = _cfg(tmp_path / "out", bucket_pack)
    source = GEN.build_nnue_source(cfg)
    embedded = _nnue_ext.source_sha256(_nnue_ext.load(str(bucket_pack)))
    assert source.pack_source_sha256 == embedded
    assert source.pack_file_sha256 == hashlib.sha256(
        bucket_pack.read_bytes(),
    ).hexdigest()
    assert source.pack_file_sha256 != embedded
    assert source.kernel in ("avx2", "scalar")


def test_the_file_hash_moves_when_the_bytes_do_and_the_source_hash_does_not(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """The whole point of P2-9, as an observation rather than an argument.

    Append a byte to a copy of the pack: the header's embedded ``.nnue`` digest
    is unchanged (it names the upstream net), the file digest moves. Under the
    old code both packs reported the same ``nnue_pack_sha256`` and the manifest
    could not tell them apart.
    """
    # ⚑ A byte FLIPPED, not appended: the loader checks the declared payload
    # size, so an appended byte is caught and the corruption this test is about
    # -- same length, different weights -- would never be reached.
    raw = bytearray(bucket_pack.read_bytes())
    raw[-1] ^= 0xFF
    edited = tmp_path / "edited.pack"
    edited.write_bytes(bytes(raw))
    cfg = _cfg(tmp_path / "out", edited)
    source = GEN.build_nnue_source(cfg)
    assert source.pack_source_sha256 == _nnue_ext.source_sha256(
        _nnue_ext.load(str(bucket_pack)),
    )
    # ⚑ Both directions. "!= the original file's hash" alone is satisfied by a
    # field that is not a file hash at all -- including the embedded source
    # digest, which is the very thing this test exists to distinguish. Naming
    # the EDITED file's hash is what makes it a file hash.
    assert source.pack_file_sha256 == hashlib.sha256(
        edited.read_bytes(),
    ).hexdigest()
    assert source.pack_file_sha256 != hashlib.sha256(
        bucket_pack.read_bytes(),
    ).hexdigest()


def test_a_missing_pack_fails_loudly(tmp_path: Path, bucket_pack: Path) -> None:
    cfg = _cfg(tmp_path / "out", bucket_pack, nnue_pack=tmp_path / "absent.pack")
    with pytest.raises((ValueError, OSError, RuntimeError)):
        GEN.build_nnue_source(cfg)


def test_cp_per_unit_changes_the_stored_policy_target(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """The value SCALE reaches the target, and the target notices.

    This is the knob the prereg does not pin, so it is the one most likely to be
    accepted and ignored: nothing downstream would look different if it were.

    ⚑ The game starts from the FIVE-piece position on purpose. Started from
    the opening the bucket pack scores every leaf of a short game identically,
    and a constant value makes the stored target SCALE-INVARIANT -- measured,
    and it made the first version of this test pass on a dead knob.
    """
    opening = _fen_openings(tmp_path, CAPTURE_FEN)
    targets = []
    for i, scale in enumerate((0.02, 4.0)):
        cfg = _cfg(
            tmp_path / f"s{i}", bucket_pack, nnue_cp_per_unit=scale, max_plies=1,
        )
        outcome, _evaluator = _play(cfg, opening=opening)
        record = outcome.records[0]
        legal = np.flatnonzero(np.asarray(record.legal_mask))
        targets.append(np.asarray(record.policy_probs)[legal])
    assert not np.allclose(targets[0], targets[1])


def test_sims_reaches_the_native_arm(tmp_path: Path, bucket_pack: Path) -> None:
    """A bigger budget must ask the ARM for more positions, not just be stored."""
    leaves = []
    for i, sims in enumerate((8, 64)):
        cfg = _cfg(
            tmp_path / f"n{i}", bucket_pack, sims=sims,
            all_root_moves=False, topk=GEN.DEFAULT_TOPK, max_plies=6,
        )
        _outcome, evaluator = _play(cfg)
        leaves.append(evaluator.nnue_source.stats.leaves)
    assert leaves[1] > leaves[0]


def test_all_root_moves_covers_every_legal_root_move(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """⚑ THE TRAP. Under a uniform prior the candidate set is the Gumbel draw.

    ⚑ THE INSTRUMENT IS NOT "does the move carry target mass". The improved
    policy is ``softmax(log_prior + sigma*Qbar)`` and is DENSE over the legal
    set by construction -- an unevaluated move gets the completed value, not a
    zero -- so a mass test reports full coverage under both settings and would
    certify the trap as absent. Measured: 44 of 44 either way.

    What actually differs is whether the arm was ever ASKED about the position
    each legal move leads to. So this checks the evaluated set at the consumer:
    with the flag on, every root child's position is evaluated; with production's
    topk it is not.
    """
    opening = _fen_openings(tmp_path, WIDE_FEN)
    root = chess.Board(WIDE_FEN)
    n_legal = root.legal_moves.count()
    assert n_legal > GEN.DEFAULT_TOPK, "the coverage test needs a wide root"
    children = set()
    for move in root.legal_moves:
        root.push(move)
        children.add(root.fen())
        root.pop()

    covered = {}
    for label, flags in (
        ("on", {"all_root_moves": True, "topk": GEN.MAX_LEGAL_MOVES}),
        ("off", {"all_root_moves": False, "topk": GEN.DEFAULT_TOPK}),
    ):
        cfg = _cfg(tmp_path / label, bucket_pack, max_plies=1, sims=8, **flags)
        evaluator = _evaluator(cfg)
        seen: set[str] = set()
        inner = evaluator._leaf_boards

        def spy(n_rows: int, _inner: Any = inner, _seen: set[str] = seen) -> Any:
            boards = _inner(n_rows)
            _seen.update(b.fen() for b in boards)
            return boards

        evaluator._leaf_boards = spy
        GEN.play_game(
            cfg=cfg, gcfg=GEN.build_gumbel_config(cfg), evaluator=evaluator,
            rng=np.random.default_rng(7), opening_cfg=opening,
        )
        covered[label] = len(children & seen)
    assert covered["on"] == n_legal
    assert covered["off"] < n_legal


def test_all_root_moves_refuses_a_topk_that_would_drop_moves(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    cfg = _cfg(tmp_path / "out", bucket_pack, all_root_moves=True, topk=16)
    with pytest.raises(ValueError, match="all-root-moves needs --topk"):
        GEN.build_gumbel_config(cfg)


def _target_and_mask_at(cfg: Any, opening: OpeningConfig) -> tuple[Any, Any]:
    outcome, _evaluator = _play(cfg, opening=opening)
    record = outcome.records[0]
    return np.asarray(record.policy_probs), np.asarray(record.legal_mask)


def test_all_root_moves_turns_off_the_terminal_root_shortcut(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """⚑⚑ P1-3: THE COVERAGE HOLE THE FLAG WAS SUPPOSED TO CLOSE.

    ``allow_terminal_root_shortcuts`` prunes every immediately-DRAWING legal
    move from a root whose value is positive, BEFORE candidate selection. The
    move then gets no prior, no visit and zero target probability -- while the
    stored ``legal_mask`` still advertises it. ``--all-root-moves`` promises
    every legal root move is a candidate, and the generator was passing
    ``True`` anyway, so the promise was broken silently and only on the roots
    where a draw was available.

    The two arms differ in NOTHING ELSE: ``topk`` is 218 on both sides and
    ``--sims`` is high enough that ``max(sims, 2*n_legal)`` equals ``sims``,
    so the realized per-root budget is identical and the shortcut flag is the
    only input that moves. ``DRAW_CAPTURE_FEN``'s Nxe5 leaves K+N vs K.
    """
    opening = _fen_openings(tmp_path, DRAW_CAPTURE_FEN)
    board = chess.Board(DRAW_CAPTURE_FEN)
    drawing = move_to_index(chess.Move.from_uci(DRAW_CAPTURE_MOVE), board)
    n_legal = board.legal_moves.count()
    assert n_legal > 1

    covered, pruned = {}, {}
    for label, all_root in (("on", True), ("off", False)):
        cfg = _cfg(
            tmp_path / f"trs_{label}", bucket_pack, max_plies=1, sims=64,
            topk=GEN.MAX_LEGAL_MOVES, all_root_moves=all_root,
        )
        # Identical realized budget on both sides -- otherwise this compares
        # two searches rather than one flag.
        assert GEN.root_simulation_budget(n_legal, cfg=cfg) == 64
        probs, mask = _target_and_mask_at(cfg, opening)
        assert bool(mask[drawing]), "the legal mask must advertise the move"
        covered[label] = float(probs[drawing])
        pruned[label] = float(probs[drawing]) == 0.0

    # With --all-root-moves the drawing move is a candidate and carries mass.
    assert covered["on"] > 0.0
    assert not pruned["on"]
    # Without it, production's shortcut prunes it: zero mass on a move the mask
    # says is legal. This half is what makes the fix a real change rather than a
    # setting that happened to already be right.
    assert pruned["off"]


def test_the_shortcut_flag_reaching_the_search_is_the_one_the_config_implies(
    tmp_path: Path, bucket_pack: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read at the CONSUMER's own call, not off the realized dict.

    The realized line reports `allow_terminal_root_shortcuts` by recomputing
    `not cfg.all_root_moves`; that is the producer restating its own intent. The
    value that matters is the keyword `run_gumbel_root_many_c` actually
    received, so the test intercepts the call.
    """
    seen: list[bool] = []
    real = GEN.run_gumbel_root_many_c

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(bool(kwargs["allow_terminal_root_shortcuts"]))
        return real(*args, **kwargs)

    for all_root in (True, False):
        cfg = _cfg(
            tmp_path / f"spy{int(all_root)}", bucket_pack, max_plies=1, sims=8,
            topk=GEN.MAX_LEGAL_MOVES, all_root_moves=all_root,
        )
        with monkeypatch.context() as patch:
            patch.setattr(GEN, "run_gumbel_root_many_c", spy)
            _play(cfg)
    assert seen == [False, True], (
        "all_root_moves=True must DISABLE the shortcut and vice versa"
    )


def test_a_root_wider_than_the_c_candidate_cap_is_counted_and_announced() -> None:
    """⚑ P1-4: the 218-move promise meets the C sampler's 64-candidate ceiling.

    ``gss_score_and_halve`` clamps the scored candidate set at
    ``GSS_MAX_CANDS``. ``--topk 218`` therefore cannot be honoured on a root
    with more than 64 legal moves: the surplus is dropped UNSCORED while the
    legal mask still lists it. The decision recorded in the source is that the
    C limit stays where it is and the generator REPORTS every root that
    exceeded it, so a readout can price the effect instead of discovering it.

    ⚑ The cap is read off the extension, not restated: a build whose ceiling
    moved must move this test's expectation with it.
    """
    assert GEN.GSS_MAX_CANDS == _mcts_tree.GSS_MAX_CANDS
    stats = GEN.RootBudgetStats()
    stats.add(legal_moves=61, sims=122, candidate_cap=GEN.GSS_MAX_CANDS)
    stats.add(
        legal_moves=GEN.GSS_MAX_CANDS + 7, sims=142,
        candidate_cap=GEN.GSS_MAX_CANDS,
    )
    out = stats.summary()
    assert out["candidate_cap"] == float(GEN.GSS_MAX_CANDS)
    assert out["roots_over_candidate_cap"] == 1.0
    assert out["roots_over_candidate_cap_frac"] == pytest.approx(0.5)
    assert out["legal_moves_over_cap_max"] == float(GEN.GSS_MAX_CANDS + 7)


def test_the_exported_candidate_cap_is_the_compiled_one() -> None:
    """⚑⚑ ONE NUMBER, TWO HOMES -- and only one of them sizes the buffer.

    ``PyModule_AddIntConstant(m, "GSS_MAX_CANDS", ...)`` is what Python reads;
    ``double scores_buf[GSS_MAX_CANDS]`` in ``gss_score_and_halve`` is what
    actually clamps. Nothing makes them agree, and an export that drifted would
    make the sidecar's `candidate_cap`, the overflow counter and the operator
    warning all describe a ceiling the search does not have -- silently, since
    the clamp itself would keep working.

    The behavioural route to this cannot see it: the clamp's effect (which
    candidates get ranked) is identical whatever Python was told. So the check
    is against the `#define` in the C source, which is the other home.

    ⚑ It compares against the source that BUILT the loaded module only if the
    tree has not been edited since; that is the normal state, and a stale .so is
    a rebuild away from being caught here rather than in the readout.
    """
    csrc = Path(_mcts_tree.__file__).with_suffix("")
    source = csrc.parent / "_mcts_tree.c"
    if not source.is_file():
        pytest.skip("C source not present beside the built extension")
    defines = re.findall(
        r"^#define\s+GSS_MAX_CANDS\s+(\d+)\s*$", source.read_text(), re.MULTILINE,
    )
    assert len(defines) == 1, f"expected exactly one #define, got {defines}"
    assert int(defines[0]) == _mcts_tree.GSS_MAX_CANDS
    # And the buffer really is sized by that name, not by a literal that happens
    # to match -- a `double scores_buf[64]` would pass the check above forever.
    assert "double scores_buf[GSS_MAX_CANDS];" in source.read_text()


def test_a_truncated_leaf_batch_is_refused_rather_than_scored_as_draws(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """⚑⚑ A SHORT BATCH IS THE ONE MISWIRING THE PLANE CHECK CANNOT SEE.

    ``_leaf_boards``'s row-count check is one-directional by necessity: the
    caller pads the encode buffer to a bucket size, so more rows than boards is
    normal. And the binding check re-encodes a board and compares planes, which
    a truncation from the END passes -- every remaining board is still on its
    correct row. What happens instead is silent: ``evaluate_encoded`` fills
    ``q[:len(boards)]`` and every unfilled row keeps q = 0, the value of a drawn
    position, which then backprops into real leaves.

    So the count is cross-checked against ``get_pending_legal_indices``, a
    different C entry point reporting the same batch. This test forges the
    disagreement by dropping the last board.
    """
    cfg = _cfg(tmp_path / "out", bucket_pack, sims=32, max_plies=1)
    evaluator = _evaluator(cfg)
    inner = evaluator._leaf_boards

    def truncating(n_rows: int, _inner: Any = inner) -> Any:
        boards = _inner(n_rows)
        return boards[:-1] if len(boards) > 1 else boards

    saw_short_batch = {"n": 0}

    class ShortTree:
        """The tree, with `pending_leaf_cboards` one board shorter."""

        def __init__(self, tree: Any) -> None:
            self._tree = tree

        def __getattr__(self, name: str) -> Any:
            return getattr(self._tree, name)

        def pending_leaf_cboards(self) -> Any:
            boards = self._tree.pending_leaf_cboards()
            if len(boards) > 1:
                saw_short_batch["n"] += 1
                return boards[:-1]
            return boards

    del truncating, inner
    tree = MCTSTree()
    evaluator.bind_tree(ShortTree(tree))
    board = chess.Board(WIDE_FEN)
    cb = _cboard(WIDE_FEN)
    arm = evaluator.nnue_source
    pre_pol, pre_wdl = GEN.native_root_logits(cb, source=arm)
    with pytest.raises(RuntimeError, match="batch is truncated"):
        GEN.run_gumbel_root_many_c(
            None, [board], device="cpu", rng=np.random.default_rng(3),
            cfg=GEN.build_gumbel_config(cfg), evaluator=evaluator, cboards=[cb],
            tree=tree, pre_pol_logits=pre_pol, pre_wdl_logits=pre_wdl,
            per_game_simulations=[32], allow_terminal_root_shortcuts=False,
            vloss_weight=1, target_batch=0,
        )
    assert saw_short_batch["n"] >= 1, "the forged truncation never happened"
    evaluator.bind_tree(None)


def test_a_run_within_the_candidate_cap_reports_zero_overflow(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """The negative control: the counter must not be always-on.

    A counter that reads 1 on every run says nothing. `WIDE_FEN` has 44 legal
    moves, comfortably under the ceiling, and the sidecar must say so.
    """
    cfg = _cfg(tmp_path / "out", bucket_pack, games=1, max_plies=6)
    budget = _summary(cfg)["root_budget"]
    assert budget["candidate_cap"] == float(GEN.GSS_MAX_CANDS)
    assert budget["roots_over_candidate_cap"] == 0.0
    assert budget["legal_moves_max"] <= GEN.GSS_MAX_CANDS


def test_root_budget_is_per_position_and_reported(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """``--sims`` is a floor under the flag, and the sidecar says what it cost."""
    cfg = _cfg(tmp_path / "out", bucket_pack, sims=8, all_root_moves=True)
    wide = chess.Board(WIDE_FEN).legal_moves.count()
    assert GEN.root_simulation_budget(wide, cfg=cfg) == 2 * wide
    assert GEN.root_simulation_budget(2, cfg=cfg) == 8
    plain = _cfg(tmp_path / "p", bucket_pack, sims=8, all_root_moves=False,
                 topk=GEN.DEFAULT_TOPK)
    assert GEN.root_simulation_budget(wide, cfg=plain) == 8

    result = _run(cfg)
    budget = result.root_budget.summary()
    assert budget["plies"] > 0
    assert budget["sims_mean"] > cfg.sims


def test_default_topk_and_coverage_follow_the_value_source() -> None:
    """The two defaults are one decision, resolved together in the parser."""
    parser = GEN.build_parser()
    native = GEN.config_from_args(parser.parse_args([
        "--out-dir", "x", "--value-source", GEN.VALUE_SOURCE_NNUE_STATIC,
        "--nnue-pack", "p.pack",
    ]))
    assert native.all_root_moves is True
    assert native.topk == GEN.MAX_LEGAL_MOVES
    pure = GEN.config_from_args(parser.parse_args(["--out-dir", "x"]))
    assert pure.all_root_moves is False
    assert pure.topk == GEN.DEFAULT_TOPK
    assert pure.nnue_resolver_max_depth is None


def test_arm_knob_defaults_come_from_the_extension_not_a_local_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ Asserted by MOVING the extension's constants, not by comparing to them.

    ``assert default == _nnue_ext.QSEARCH_MAX_PLY`` passes just as happily
    against a Python copy of the same number -- and a copy is exactly what
    drifts. Patching the extension's exported value makes a local copy fail.
    """
    parser = GEN.build_parser()
    monkeypatch.setattr(_nnue_ext, "RESOLVER_MAX_DEPTH", 29, raising=True)
    monkeypatch.setattr(_nnue_ext, "QSEARCH_MAX_PLY", 6, raising=True)
    monkeypatch.setattr(_nnue_ext, "QSEARCH_CHECK_PLIES", 3, raising=True)
    cfg = GEN.config_from_args(parser.parse_args([
        "--out-dir", "x", "--value-source", GEN.VALUE_SOURCE_NNUE_QSEARCH,
        "--nnue-pack", "p.pack",
    ]))
    assert cfg.nnue_resolver_max_depth == 29
    assert cfg.nnue_qsearch_max_ply == 6
    assert cfg.nnue_qsearch_check_plies == 3


def test_the_static_arm_resolves_no_quiescence_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ P1-1 at the resolver: the gate is the ARM, not "is there an arm".

    The extension's defaults are moved first, so a resolution that leaked
    through would show up as 6/3 rather than as a value that could be confused
    with "nothing was set".
    """
    parser = GEN.build_parser()
    monkeypatch.setattr(_nnue_ext, "RESOLVER_MAX_DEPTH", 29, raising=True)
    monkeypatch.setattr(_nnue_ext, "QSEARCH_MAX_PLY", 6, raising=True)
    monkeypatch.setattr(_nnue_ext, "QSEARCH_CHECK_PLIES", 3, raising=True)
    cfg = GEN.config_from_args(parser.parse_args([
        "--out-dir", "x", "--value-source", GEN.VALUE_SOURCE_NNUE_STATIC,
        "--nnue-pack", "p.pack",
    ]))
    assert cfg.nnue_resolver_max_depth == 29  # consumed by both arms
    assert cfg.nnue_qsearch_max_ply is None
    assert cfg.nnue_qsearch_check_plies is None


# ===========================================================================
# The value mapping
# ===========================================================================


def test_mate_band_values_do_not_go_through_the_centipawn_slope(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """Audit N1's defect class in a new scale, refused here.

    A mate score is a distance to mate, not an evaluation. Multiplying it by the
    cp-per-unit slope would produce hundreds of thousands of "centipawns" --
    harmless only because the logistic saturates, and one slope change away from
    ranking a slow mate below a good position.
    """
    source = GEN.build_nnue_source(_cfg(tmp_path / "out", bucket_pack))
    base = _nnue_ext.RESOLVER_MATE_BASE
    step = _nnue_ext.RESOLVER_MATE_PLY_STEP
    values = np.array([base, base - 4 * step, -(base - 4 * step), -base, 3000.0])
    q, is_mate = source.q_from_values(values)
    assert is_mate.tolist() == [True, True, True, True, False]
    # Mates are decisive and ordered by distance; a plain evaluation is not.
    assert q[0] == pytest.approx(1.0, abs=1e-6)
    assert q[3] == pytest.approx(-1.0, abs=1e-6)
    assert q[0] >= q[1] > q[4]
    assert q[2] < 0.0 < q[4]


def test_evaluation_band_matches_the_production_cp_logistic(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """An evaluation is scaled and then mapped by the SHARED converter.

    ⚑⚑ EVERY NUMBER HERE IS A LITERAL, AND NONE OF THEM IS A DEFAULT. The first
    version of this test configured the source with ``GEN.NNUE_CP_SLOPE`` and
    then built the expected value out of ``GEN.NNUE_CP_SLOPE`` -- so both sides
    moved together and a source that IGNORED ``self.cp_slope`` and hardcoded
    0.006 passed, which is precisely the defect class the file exists to catch.
    A test whose expectation is derived from the code under test's own constants
    is a tautology dressed as an assertion.

    So: three off-default settings, chosen to differ from the production trio in
    every component, and an expectation built from those same literals.
    """
    source = GEN.build_nnue_source(
        _cfg(
            tmp_path / "out", bucket_pack,
            nnue_cp_per_unit=0.4, nnue_cp_slope=0.011, nnue_cp_draw_width=55.0,
        ),
    )
    assert (
        GEN.NNUE_CP_PER_INTERNAL_UNIT, GEN.NNUE_CP_SLOPE, GEN.NNUE_CP_DRAW_WIDTH,
    ) != (0.4, 0.011, 55.0), "the whole point is that these are NOT the defaults"
    q, is_mate = source.q_from_values(np.array([1000.0]))
    assert not bool(is_mate[0])
    expected = cp_to_wdl(1000.0 * 0.4, None, slope=0.011, draw_width_cp=55.0)
    assert q[0] == pytest.approx(float(expected[0]) - float(expected[2]), abs=1e-9)


def test_the_cp_defaults_are_productions_own_numbers() -> None:
    """The defaults, pinned against independent literals rather than themselves.

    Separate from the mapping test on purpose: that one proves the knobs are
    READ, this one proves the values they default to are the live ones. Written
    as literals so a change to `selfplay.sf_wdl_cp_slope` here has to be a
    deliberate edit of this line rather than a silent follow-the-constant.
    """
    assert GEN.NNUE_CP_SLOPE == 0.006
    assert GEN.NNUE_CP_DRAW_WIDTH == 120.0
    assert GEN.NNUE_CP_PER_INTERNAL_UNIT == 0.28


def test_the_cp_knobs_reach_the_mapping_one_at_a_time(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """Each of the three moves the answer ON ITS OWN.

    A test that varies all three together cannot tell which one is live: a
    source that read the scale and hardcoded the slope would still produce a
    different number. One knob per comparison, the other two held at the same
    off-default values in both arms.
    """
    base = {"nnue_cp_per_unit": 0.4, "nnue_cp_slope": 0.011, "nnue_cp_draw_width": 55.0}
    values = np.array([1000.0])

    def q_at(tag: str, **changes: float) -> float:
        cfg = _cfg(tmp_path / tag, bucket_pack, **{**base, **changes})
        return float(GEN.build_nnue_source(cfg).q_from_values(values)[0][0])

    reference = q_at("ref")
    assert q_at("scale", nnue_cp_per_unit=0.9) != pytest.approx(reference, abs=1e-9)
    assert q_at("slope", nnue_cp_slope=0.003) != pytest.approx(reference, abs=1e-9)
    assert q_at("width", nnue_cp_draw_width=400.0) != pytest.approx(
        reference, abs=1e-9,
    )


def test_q_from_values_is_monotone(tmp_path: Path, bucket_pack: Path) -> None:
    source = GEN.build_nnue_source(_cfg(tmp_path / "out", bucket_pack))
    q, _ = source.q_from_values(np.array([-3000.0, -300.0, 0.0, 300.0, 3000.0]))
    assert list(q) == sorted(q)


# ===========================================================================
# Determinism
# ===========================================================================


def test_same_seed_reproduces_the_native_shard_digests(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    digests = []
    for run in ("a", "b"):
        cfg = _cfg(tmp_path / run, bucket_pack, games=2, max_plies=20, seed=31337)
        result = _run(cfg)
        digests.append([(s["index"], s["rows"], s["digest"]) for s in result.shards])
    assert digests[0] == digests[1]
    assert digests[0], "the determinism test must have something to compare"


def test_a_different_seed_changes_the_native_shards(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """...so the determinism test above is not passing on a constant corpus."""
    digests = []
    for run, seed in (("a", 31337), ("b", 31338)):
        cfg = _cfg(tmp_path / run, bucket_pack, games=2, max_plies=20, seed=seed)
        result = _run(cfg)
        digests.append([s["digest"] for s in result.shards])
    assert digests[0] != digests[1]


# ===========================================================================
# The manifest the readout reads
# ===========================================================================


def _summary(cfg: Any) -> dict[str, Any]:
    GEN.generate(cfg)
    path = GEN.summary_path_for(cfg, shard_index_start=0)
    return json.loads(path.read_text())


def test_summary_records_the_arm_provenance_and_the_throughput_gate_inputs(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """Everything the >=813,000 rows/h gate and the cell table are read off."""
    cfg = _cfg(tmp_path / "out", bucket_pack, games=1, max_plies=16)
    summary = _summary(cfg)
    assert summary["partial"] is False
    assert summary["wall_seconds"] > 0.0
    assert summary["rows_per_hour"] > 0.0
    assert summary["games_per_hour"] > 0.0

    provenance = summary["provenance"]
    assert provenance["code_sha"]
    assert "code_dirty" in provenance
    # Both endpoints, so a tree that moved mid-run is visible. On a run this
    # short they agree; the point is that both are recorded.
    assert provenance["code_sha_at_finish"] == provenance["code_sha"]
    assert provenance["seed"] == cfg.seed
    assert provenance["value_source"] == GEN.VALUE_SOURCE_NNUE_STATIC
    assert provenance["worker_seeds"] == [
        {"worker_id": 0, "seed": int(GEN.build_worker_specs(cfg, shard_index_start=0)[0].seed)},
    ]

    realized = summary["realized_per_worker"][0]
    assert realized["nnue_arm"] == GEN.VALUE_SOURCE_NNUE_STATIC
    # The FILE digest under `nnue_pack_sha256`, the header's embedded `.nnue`
    # digest under its own name -- see the two-hash test.
    assert realized["nnue_pack_sha256"] == hashlib.sha256(
        bucket_pack.read_bytes(),
    ).hexdigest()
    assert realized["nnue_pack_source_sha256"] == _nnue_ext.source_sha256(
        _nnue_ext.load(str(bucket_pack)),
    )
    assert realized["nnue_cp_per_internal_unit"] == cfg.nnue_cp_per_unit
    assert realized["nnue_resolver_max_depth"] == _nnue_ext.RESOLVER_MAX_DEPTH
    assert realized["all_root_moves"] is True

    arm = summary["nnue"]
    # ⚑ The prereg's two columns, CROSS-CHECKED against the counters they are
    # derived from. `0.0 <= frac <= 1.0` was the assertion here before and it is
    # definitionally true of any ratio of two non-negative counters -- it passes
    # on a numerator wired to the wrong field, on a hardcoded 0.5, and on a
    # denominator that is `leaves` rather than `calls`. The exact-value test is
    # `test_the_prereg_rates_are_the_ratios_they_claim_to_be`, on forged
    # counters; here the same identity is checked on a real run.
    assert arm["ctx_calls"] > 0
    assert arm["in_check_call_frac"] == pytest.approx(
        arm["ctx_calls_in_check"] / arm["ctx_calls"], abs=1e-12,
    )
    assert arm["resolver_expansion_factor"] == pytest.approx(
        arm["ctx_nodes"] / arm["ctx_resolved_leaves"], abs=1e-12,
    )
    # `calls` is the arm's own count of positions, and it must account for every
    # leaf AND every root this run put through it -- a denominator that silently
    # dropped one population is what makes the rate above meaningless.
    assert arm["ctx_calls"] == arm["leaves"] + arm["roots"]
    # Roots are counted apart from leaves: one root per ply also goes through
    # the arm, and a rate that pooled them would move with the sim budget.
    assert arm["roots"] == summary["rows"]
    assert arm["resolver_expansion_factor"] >= 1.0
    assert arm["leaves"] > 0
    assert arm["binding_checks"] == 1
    assert arm["ctx_config_conflicts"] == 0.0
    assert summary["root_budget"]["sims_mean"] > 0
    assert summary["failed_workers"] == []
    # OFF by default, and the sidecar says so rather than omitting the block.
    assert summary["leaf_bank"] == {"enabled": False, "rows": 0, "files": []}


def test_a_static_run_publishes_no_quiescence_knob_anywhere(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """⚑⚑ P1-1: THE SIGNATURE DEFECT WITH A RECEIPT ON TOP.

    ``cae_arm_static_eval`` reads ``resolver_max_depth`` and nothing else. The
    generator nonetheless resolved, stored, snapshotted and PUBLISHED
    ``qsearch_max_ply=4`` / ``qsearch_check_plies=1`` for a static run -- a value
    accepted and then silently ignored, which is exactly the failure this repo
    is built to catch, except that it came with a realized line asserting the
    opposite.

    The test reads the whole realized line and the config block, not just the
    two keys, so a future field that reintroduces the number under a new name
    fails here too.
    """
    cfg = _cfg(tmp_path / "out", bucket_pack, games=1, max_plies=8)
    assert cfg.value_source == GEN.VALUE_SOURCE_NNUE_STATIC
    summary = _summary(cfg)
    realized = summary["realized_per_worker"][0]
    assert realized["nnue_resolver_max_depth"] == _nnue_ext.RESOLVER_MAX_DEPTH
    assert realized["nnue_qsearch_max_ply"] is None
    assert realized["nnue_qsearch_check_plies"] is None
    assert realized["nnue_consumed_knobs"] == ["resolver_max_depth"]
    assert realized["nnue_arm_config_requested"] == {
        "resolver_max_depth": _nnue_ext.RESOLVER_MAX_DEPTH,
    }
    assert summary["config"]["nnue_qsearch_max_ply"] is None
    assert summary["config"]["nnue_qsearch_check_plies"] is None
    # ⚑ And no OTHER field carries it either. The default is 4/1, so a stray
    # republication would show up as one of those integers under a qsearch name.
    for key, value in realized.items():
        if "qsearch" in key:
            assert value is None, f"{key} republished a knob the static arm ignores"


def test_the_qsearch_run_does_publish_both_because_it_reads_both(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """The other half of P1-1: absence must be arm-specific, not blanket.

    A "fix" that simply stopped reporting the quiescence pair for every arm
    would pass the static test above while removing real provenance from the
    arm that does consume it. Same assertions, opposite expectation.
    """
    cfg = _cfg(
        tmp_path / "out", bucket_pack, games=1, max_plies=8,
        value_source=GEN.VALUE_SOURCE_NNUE_QSEARCH,
        nnue_qsearch_max_ply=3, nnue_qsearch_check_plies=2,
    )
    realized = _summary(cfg)["realized_per_worker"][0]
    assert realized["nnue_qsearch_max_ply"] == 3
    assert realized["nnue_qsearch_check_plies"] == 2
    assert realized["nnue_consumed_knobs"] == [
        "resolver_max_depth", "qsearch_max_ply", "qsearch_check_plies",
    ]


def test_banked_leaf_observations_carry_the_raw_score_and_its_settings(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """⚑ P2-5: the RAW arm score, so a scale change is a reanalysis not a rerun.

    ``--nnue-cp-per-unit`` and the cp-logistic are a pure function of the
    internal value, but only the RESULT reaches ``policy_target``. Without the
    bank, correcting the scale means playing every game again. With it, the
    correction is arithmetic over these rows.

    The row must carry enough to redo that arithmetic and to know it is the
    right corpus: the position, the raw value, the pack the value came from,
    the cp settings in force, and the cluster key that joins it to a shard row.
    """
    cfg = _cfg(
        tmp_path / "out", bucket_pack, games=1, max_plies=6,
        bank_leaf_observations=True,
    )
    summary = _summary(cfg)
    bank = GEN.leaf_bank_path(cfg.out_dir, 0)
    assert bank.exists()
    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    assert rows
    assert summary["leaf_bank"]["enabled"] is True
    assert summary["leaf_bank"]["rows"] == len(rows)
    assert summary["leaf_bank"]["files"] == [bank.name]

    row = rows[0]
    assert row["schema"] == GEN.LEAF_BANK_SCHEMA
    assert row["arm"] == GEN.VALUE_SOURCE_NNUE_STATIC
    assert row["role"] in ("leaf", "root")
    assert chess.Board(row["fen"])  # parses, i.e. it is a position not a label
    assert row["cp_per_internal_unit"] == cfg.nnue_cp_per_unit
    assert row["cp_slope"] == cfg.nnue_cp_slope
    assert row["cp_draw_width"] == cfg.nnue_cp_draw_width
    assert row["resolver_max_depth"] == _nnue_ext.RESOLVER_MAX_DEPTH
    # ⚑ Arm-scoped here too: a static run's bank row must not carry a
    # quiescence setting the arm never read (P1-1).
    assert "qsearch_max_ply" not in row
    assert row["pack_file_sha256"] == hashlib.sha256(
        bucket_pack.read_bytes(),
    ).hexdigest()
    assert row["game"] >= 0
    assert row["ply"] >= 0

    # The value is RAW: it must be reproducible by evaluating that position
    # through the arm, and it must NOT already be a q in [-1, 1].
    source = GEN.build_nnue_source(_cfg(tmp_path / "re", bucket_pack))
    again = _nnue_ext.arm_handle_eval(source._handle, [_cboard(row["fen"])])
    assert int(again[0]) == int(row["value"])
    assert abs(int(row["value"])) > 1


def test_the_bank_is_off_by_default_and_writes_nothing(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """The default costs nothing -- and the sidecar states that it was off."""
    cfg = _cfg(tmp_path / "out", bucket_pack, games=1, max_plies=6)
    assert cfg.bank_leaf_observations is False
    summary = _summary(cfg)
    assert not GEN.leaf_bank_path(cfg.out_dir, 0).exists()
    assert summary["leaf_bank"] == {"enabled": False, "rows": 0, "files": []}
    assert summary["realized_per_worker"][0]["nnue_leaf_bank"] is None


def test_a_worker_that_dies_still_reports_the_work_it_did(
    tmp_path: Path, bucket_pack: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ P2-8: the partial counters travel with the failure.

    A worker that raises after hours of evaluation used to lose its
    ``WorkerResult`` entirely: shards on disk with no owner, listed as orphans,
    beside an ``nnue`` block reporting that the arm evaluated nothing. The
    exception now carries the partial result, and the sidecar folds it in.

    The failure is injected at the SECOND game so there is real work to lose.
    """
    cfg = _cfg(
        tmp_path / "out", bucket_pack, games=3, max_plies=6, shard_size=1,
    )
    real_play = GEN.play_game
    calls = {"n": 0}

    def boom(**kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected worker failure")
        return real_play(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(GEN, "play_game", boom)
        with pytest.raises(RuntimeError, match="injected worker failure"):
            GEN.generate(cfg)

    summary = json.loads(
        GEN.summary_path_for(cfg, shard_index_start=0).read_text(),
    )
    assert summary["partial"] is True
    assert "injected worker failure" in str(summary["error"])
    # The dead worker reported: its game, its rows, its shard and its arm work.
    assert summary["workers_reported"] == 1
    assert summary["games"] == 1
    assert summary["rows"] > 0
    assert summary["shards_written"] == 1
    assert summary["orphan_shards"] == []
    assert summary["nnue"]["leaves"] > 0
    assert summary["failed_workers"] == [
        {"worker_id": 0, "error": "RuntimeError: injected worker failure"},
    ]


def test_the_code_identity_is_captured_before_the_workers_start(
    tmp_path: Path, bucket_pack: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ P2-10: WHEN the sha is read is part of what it means.

    Read at summary time it names the tree as it stands AFTER a run that can
    span a night -- so a branch switch, a rebase or a stash anywhere in that
    window relabels a corpus with code that produced none of it.

    ⚑ THE ORDERING IS NOT VISIBLE IN THE VALUES, and the first version of this
    test proved it the hard way: with `git_sha` made a counter, a summary-time
    capture still yields `code_sha == "sha001"` and a later
    `code_sha_at_finish`, because "first call" is relative to whenever the
    capturing started. The mutant that moved the capture past the workers
    survived. So the instrument is the ORDER OF TWO EVENTS: how many sha reads
    had happened by the time the first game was played. At launch capture that
    is >= 1; at summary capture it is 0.
    """
    calls = {"n": 0}
    at_first_game: list[int] = []
    real_play = GEN.play_game

    def counting_sha() -> str:
        calls["n"] += 1
        return f"sha{calls['n']:03d}"

    def spy_play(**kwargs: Any) -> Any:
        at_first_game.append(calls["n"])
        return real_play(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(GEN, "git_sha", counting_sha)
        patch.setattr(GEN, "play_game", spy_play)
        cfg = _cfg(tmp_path / "out", bucket_pack, games=1, max_plies=6)
        summary = _summary(cfg)

    assert at_first_game, "the spy never saw a game"
    assert at_first_game[0] >= 1, (
        "the code identity must be read BEFORE the first game is played"
    )
    assert summary["provenance"]["code_sha"] == "sha001"
    assert summary["provenance"]["code_sha_at_finish"] != "sha001"
    assert calls["n"] >= 2


def test_summary_of_a_pure_source_reports_no_arm(tmp_path: Path) -> None:
    """Empty is the READING for a source with no arm, not a missing key."""
    summary = _summary(
        _cfg(tmp_path / "out", value_source=GEN.VALUE_SOURCE_ZERO, max_plies=16),
    )
    assert summary["nnue"] == {}
    assert summary["realized_per_worker"][0]["nnue_arm"] is None
    assert summary["provenance"]["nnue_pack"] is None


# ===========================================================================
# Refusals — a value accepted and ignored is the defect this repo has
# ===========================================================================


def test_a_pack_without_an_arm_is_refused(tmp_path: Path, bucket_pack: Path) -> None:
    with pytest.raises(SystemExit, match="would ignore it"):
        GEN.main([
            "--out-dir", str(tmp_path / "out"), "--games", "1",
            "--value-source", GEN.VALUE_SOURCE_ZERO,
            "--nnue-pack", str(bucket_pack),
        ])


def test_an_arm_without_a_pack_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="needs --nnue-pack"):
        GEN.main([
            "--out-dir", str(tmp_path / "out"), "--games", "1",
            "--value-source", GEN.VALUE_SOURCE_NNUE_STATIC,
        ])


def test_an_evaluator_whose_source_disagrees_with_its_label_is_refused(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """The corpus must not be labelled with the arm that did not produce it."""
    source = GEN.build_nnue_source(
        _cfg(tmp_path / "out", bucket_pack, value_source=GEN.VALUE_SOURCE_NNUE_STATIC),
    )
    with pytest.raises(ValueError, match="labelled with the arm"):
        GEN.UniformPriorEvaluator(
            value_source=GEN.VALUE_SOURCE_NNUE_QSEARCH,
            expected_planes=_PLANES,
            nnue_source=source,
        )


def test_a_pure_source_refuses_an_arm_and_an_arm_refuses_none(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    source = GEN.build_nnue_source(_cfg(tmp_path / "out", bucket_pack))
    with pytest.raises(ValueError, match="disagree"):
        GEN.UniformPriorEvaluator(
            value_source=GEN.VALUE_SOURCE_ZERO,
            expected_planes=_PLANES,
            nnue_source=source,
        )
    with pytest.raises(ValueError, match="disagree"):
        GEN.UniformPriorEvaluator(
            value_source=GEN.VALUE_SOURCE_NNUE_STATIC,
            expected_planes=_PLANES,
            nnue_source=None,
        )


def test_a_non_positive_cp_per_unit_is_refused(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        GEN.build_nnue_source(
            _cfg(tmp_path / "out", bucket_pack, nnue_cp_per_unit=0.0),
        )


@pytest.mark.parametrize(
    "knob", ["nnue_cp_per_unit", "nnue_cp_slope", "nnue_cp_draw_width"],
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_cp_knob_is_refused(
    tmp_path: Path, bucket_pack: Path, knob: str, bad: float,
) -> None:
    """⚑⚑ NaN PASSES EVERY COMPARISON-BASED GUARD, because every comparison
    against NaN is False. ``nan > 0`` is False and ``nan <= 0`` is False, so a
    positivity check written either way LETS IT THROUGH -- and it then reaches
    ``cp_to_wdl_array``, makes every W and L NaN, makes ``q`` NaN, and lands in
    ``policy_target`` as a silent corruption of the whole corpus. The infinities
    are here because they take the opposite route through the same hole: they
    pass a positivity check honestly and saturate the logistic to a constant.
    """
    with pytest.raises(ValueError, match="must be finite"):
        GEN.build_nnue_source(_cfg(tmp_path / "out", bucket_pack, **{knob: bad}))


def test_a_nan_scale_would_have_reached_the_target(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """The negative control for the guard above: prove the hole was real.

    A refusal test alone cannot show the refusal matters -- it passes just as
    well if the value was harmless. This drives ``q_from_values`` on a source
    whose scale is NaN (set past the constructor, which is what the missing
    check amounted to) and watches the target go NaN.
    """
    source = GEN.build_nnue_source(_cfg(tmp_path / "out", bucket_pack))
    source.cp_slope = float("nan")
    q, _is_mate = source.q_from_values(np.array([1000.0]))
    assert bool(np.isnan(q[0])), "a NaN slope must in fact poison the value"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--nnue-qsearch-max-ply", "0"),
        ("--nnue-qsearch-check-plies", "0"),
    ],
)
def test_a_quiescence_flag_on_the_static_arm_is_refused(
    tmp_path: Path, bucket_pack: Path, flag: str, value: str,
) -> None:
    """⚑ P2-7 + P1-1: NOT SILENTLY NULLED, REFUSED.

    ``--nnue-qsearch-max-ply 0 --value-source nnue-static`` reads like the
    quiescence arm's own negative control. It used to parse, be replaced with
    None, and produce a default-configured static run -- a command line that
    documents an experiment the corpus is not.
    """
    with pytest.raises(SystemExit, match="read only by"):
        GEN.main([
            "--out-dir", str(tmp_path / "out"), "--games", "1",
            "--value-source", GEN.VALUE_SOURCE_NNUE_STATIC,
            "--nnue-pack", str(bucket_pack), flag, value,
        ])


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--nnue-resolver-max-depth", "8"),
        ("--nnue-qsearch-max-ply", "2"),
        ("--nnue-cp-per-unit", "0.5"),
        ("--nnue-cp-slope", "0.01"),
        ("--nnue-cp-draw-width", "80"),
    ],
)
def test_an_arm_flag_on_a_pure_source_is_refused(
    tmp_path: Path, flag: str, value: str,
) -> None:
    """Every arm flag, not just the pack. The pack was the only one that refused
    before, and only because a missing file would have crashed anyway."""
    with pytest.raises(SystemExit, match="would ignore it"):
        GEN.main([
            "--out-dir", str(tmp_path / "out"), "--games", "1",
            "--value-source", GEN.VALUE_SOURCE_ZERO, flag, value,
        ])


def test_supplying_a_cp_flag_at_its_own_default_is_still_refused(
    tmp_path: Path,
) -> None:
    """⚑ The reason the parser holds a sentinel rather than the constant.

    Typing the default value is still typing the flag, and a run whose command
    line names a knob its value source cannot read is mislabelled whatever the
    number was. argparse cannot tell "typed the default" from "typed nothing"
    unless the parser default is a sentinel -- so this test is what stops that
    indirection from being simplified away.
    """
    with pytest.raises(SystemExit, match="would ignore it"):
        GEN.main([
            "--out-dir", str(tmp_path / "out"), "--games", "1",
            "--value-source", GEN.VALUE_SOURCE_ZERO,
            "--nnue-cp-slope", str(GEN.NNUE_CP_SLOPE),
        ])


def test_banking_without_an_arm_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="has no arm"):
        GEN.main([
            "--out-dir", str(tmp_path / "out"), "--games", "1",
            "--value-source", GEN.VALUE_SOURCE_ZERO, "--bank-leaf-observations",
        ])


# ===========================================================================
# Merging across workers — a peak is not a sum
# ===========================================================================


def _stats(**context: int) -> Any:
    return GEN.NnueArmStats(leaves=1, context=dict(context))


def test_merging_workers_sums_counters_and_takes_the_max_of_peaks() -> None:
    """⚑⚑ P1-2: `max_depth_seen` IS A PEAK, AND SUMMING IT INVENTS A CUTOFF.

    The C side merges these with an atomic MAX inside one context, so the value
    a worker reports is already "the deepest this worker went". Adding four
    workers' peaks together reports 128 for a run whose depth cap is 32 -- a
    number that cannot occur, sitting next to `ctx_depth_cutoffs` which is what
    a reader would check against it. Forged counters, because the point is the
    arithmetic and a real run cannot be made to produce a chosen peak.
    """
    a = _stats(calls=10, nodes=100, max_depth_seen=31, qmax_ply_seen=4)
    b = _stats(calls=7, nodes=70, max_depth_seen=12, qmax_ply_seen=2)
    a.merge(b)
    assert a.context["calls"] == 17
    assert a.context["nodes"] == 170
    assert a.context["max_depth_seen"] == 31
    assert a.context["qmax_ply_seen"] == 4


def test_merging_workers_reports_a_configuration_conflict_rather_than_hiding_it(
) -> None:
    """Config keys are neither summed nor maxed: they must AGREE.

    Two workers at different resolver depths are not one cell, and summing
    32 + 8 = 40 or maxing to 32 would publish a configuration neither ran.
    """
    a = _stats(resolver_max_depth=32, calls=1)
    b = _stats(resolver_max_depth=8, calls=1)
    a.merge(b)
    assert a.context["resolver_max_depth"] == 8  # the shallowest, not the sum
    assert a.context_conflicts == {"resolver_max_depth": [8, 32]}
    assert a.summary()["ctx_config_conflicts"] == 1.0


def test_every_arm_stats_key_is_classified(bucket_pack: Path) -> None:
    """⚑⚑ THE KEY SET COMES OFF A LIVE HANDLE, NOT OFF THE TRANSCRIPTION.

    The three `_CTX_*_KEYS` sets are a Python copy of what `arm_stats_dict`
    emits, and a copy only has to be incomplete once. It was: `qterminal_draw`
    and `qply_cutoffs` were missing, every forged-dict unit test passed, and the
    first real two-worker qsearch run died in `merge` -- correctly, but in
    production rather than in CI.

    So the expectation is the extension's own answer. A new C counter fails
    here, before it can stop a generation run.
    """
    handle = _nnue_ext.arm_open(
        GEN.VALUE_SOURCE_NNUE_QSEARCH, str(bucket_pack),
    )
    keys = set(dict(_nnue_ext.arm_stats(handle)))
    classified = (
        GEN._CTX_COUNTER_KEYS | GEN._CTX_PEAK_KEYS | GEN._CTX_CONFIG_KEYS
    )
    assert keys, "arm_stats returned nothing"
    assert keys <= classified, f"unclassified arm_stats keys: {sorted(keys - classified)}"
    # And nothing classified that the extension does not emit -- a stale entry
    # here is a key that silently never merges.
    assert classified <= keys, f"classified but absent: {sorted(classified - keys)}"
    # The three sets must be disjoint, or a key's merge rule depends on which
    # `elif` happens to come first.
    assert GEN._CTX_COUNTER_KEYS.isdisjoint(GEN._CTX_PEAK_KEYS)
    assert GEN._CTX_COUNTER_KEYS.isdisjoint(GEN._CTX_CONFIG_KEYS)
    assert GEN._CTX_PEAK_KEYS.isdisjoint(GEN._CTX_CONFIG_KEYS)


def test_an_unclassified_context_key_refuses_to_merge() -> None:
    """A new C counter must be classified before it can be merged.

    ⚑ Not a default-to-sum: whichever rule is the fallback becomes the silent
    answer for every future key, and summing was how `max_depth_seen` came to
    read 128. The refusal is the whole mechanism.
    """
    a = _stats(brand_new_counter=1)
    b = _stats(brand_new_counter=1)
    with pytest.raises(ValueError, match="not classified"):
        a.merge(b)


def test_the_prereg_rates_are_the_ratios_they_claim_to_be() -> None:
    """The two reported columns, at values a real run cannot be made to produce.

    ⚑ The assertion this replaces was ``0.0 <= in_check_call_frac <= 1.0``,
    which is definitionally satisfied by any ratio of non-negative counters --
    true of a hardcoded 0.5, of the wrong numerator, and of a denominator that
    used `leaves` instead of `calls`. Forged counters give it an exact value.
    """
    stats = GEN.NnueArmStats(
        leaves=90, roots=10,
        context={
            "calls": 100, "calls_in_check": 25,
            "nodes": 700, "resolved_leaves": 350,
        },
    )
    out = stats.summary()
    assert out["in_check_call_frac"] == pytest.approx(0.25)
    assert out["resolver_expansion_factor"] == pytest.approx(2.0)
