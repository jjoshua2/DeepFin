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
from chess_anti_engine.mcts._mcts_tree import MCTSTree
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
            nnue_qsearch_max_ply=_nnue_ext.QSEARCH_MAX_PLY,
            nnue_qsearch_check_plies=_nnue_ext.QSEARCH_CHECK_PLIES,
        )
    base.update(overrides)
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


def test_native_shard_schema_matches_a_gen0_reference_shard(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    """Field by field, not "it loads": the ruler reads both sets in one pass.

    A missing optional array, a widened dtype or a changed row shape would all
    still load; what breaks is the JOINT scoring pass over the native cells and
    the banked UCI anchors, and it breaks silently by dropping a column.
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


def test_pack_reaches_the_arm_and_is_named_by_its_hash(
    tmp_path: Path, bucket_pack: Path,
) -> None:
    cfg = _cfg(tmp_path / "out", bucket_pack)
    source = GEN.build_nnue_source(cfg)
    assert source.pack_sha256 == _nnue_ext.source_sha256(
        _nnue_ext.load(str(bucket_pack)),
    )
    assert source.kernel in ("avx2", "scalar")


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
    """An evaluation is scaled and then mapped by the SHARED converter."""
    source = GEN.build_nnue_source(
        _cfg(tmp_path / "out", bucket_pack, nnue_cp_per_unit=0.28),
    )
    q, is_mate = source.q_from_values(np.array([1000.0]))
    assert not bool(is_mate[0])
    expected = cp_to_wdl(
        1000.0 * 0.28, None,
        slope=GEN.NNUE_CP_SLOPE, draw_width_cp=GEN.NNUE_CP_DRAW_WIDTH,
    )
    assert q[0] == pytest.approx(float(expected[0]) - float(expected[2]), abs=1e-9)


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
    assert provenance["seed"] == cfg.seed
    assert provenance["value_source"] == GEN.VALUE_SOURCE_NNUE_STATIC
    assert provenance["worker_seeds"] == [
        {"worker_id": 0, "seed": int(GEN.build_worker_specs(cfg, shard_index_start=0)[0].seed)},
    ]

    realized = summary["realized_per_worker"][0]
    assert realized["nnue_arm"] == GEN.VALUE_SOURCE_NNUE_STATIC
    assert realized["nnue_pack_sha256"] == _nnue_ext.source_sha256(
        _nnue_ext.load(str(bucket_pack)),
    )
    assert realized["nnue_cp_per_internal_unit"] == cfg.nnue_cp_per_unit
    assert realized["nnue_resolver_max_depth"] == _nnue_ext.RESOLVER_MAX_DEPTH
    assert realized["all_root_moves"] is True

    arm = summary["nnue"]
    # The two columns the prereg asks every cell to report.
    assert 0.0 <= arm["in_check_call_frac"] <= 1.0
    # Roots are counted apart from leaves: one root per ply also goes through
    # the arm, and a rate that pooled them would move with the sim budget.
    assert arm["roots"] == summary["rows"]
    assert arm["resolver_expansion_factor"] >= 1.0
    assert arm["leaves"] > 0
    assert arm["binding_checks"] == 1
    assert summary["root_budget"]["sims_mean"] > 0


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
