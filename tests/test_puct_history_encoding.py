"""Exact native leaf-buffer parity and checkpoint-mode transport through PUCT."""
from __future__ import annotations

import threading
from functools import partial

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.lc0 import c_input_history_mode
from chess_anti_engine.mcts import _mcts_tree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.puct_history import puct_history_args
from chess_anti_engine.mcts.puct_vl import PucvChunker, _alloc_buffers
from chess_anti_engine.uci.multi_gpu_pucv_pool import MultiGpuPucvConfig, MultiGpuPucvPool
from chess_anti_engine.uci.root_parallel_gumbel import RootParallelGumbelConfig, RootParallelGumbelPool
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.walker_pool import WalkerPool, WalkerPoolConfig

MODES = ["legacy", "lc0_root", "lc0_root_legacy_meta"]


def _position(case: str) -> tuple[chess.Board, chess.Move]:
    board = chess.Board()
    if case == "black_history":
        for move in ["e2e4", "e7e5", "g1f3"]:
            board.push_uci(move)
        move = "b8c6"
    elif case == "white_repetition":
        for move in ["g1f3", "g8f6", "f3g1", "f6g8", "e2e4", "e7e5"]:
            board.push_uci(move)
        # Repeated states in the retained history, but the current position
        # must remain searchable (native search treats a current repeat as draw).
        assert not board.is_repetition(2)
        move = "g1f3"
    else:
        board = chess.Board("r3k2r/8/8/8/2p5/8/3P4/R3K2R w KQkq - 17 42")
        move = "d2d4"
    chosen = chess.Move.from_uci(move)
    assert chosen in board.legal_moves
    return board, chosen


def _seed(board: chess.Board, move: chess.Move):
    cb = CBoard.from_board(board)
    child_board = board.copy(stack=True)
    child_board.push(move)
    child = CBoard.from_board(child_board)
    legal = np.asarray(cb.legal_move_indices(), dtype=np.int32)
    action = -1
    for a in legal:
        candidate = cb.copy()
        candidate.push_index(int(a))
        if candidate.fen() == child.fen():
            action = int(a)
            break
    assert action >= 0
    tree = _mcts_tree.MCTSTree()
    tree.reserve(1024, 8192)
    root = tree.add_root(1, 0.0, cb)
    priors = np.zeros(legal.size, np.float64)
    priors[np.flatnonzero(legal == action)] = 1.0
    tree.expand(root, legal, priors)
    return tree, root, cb, child, action, child_board


def _batch(tree, root, cb, mode, planes=146, *, explicit=True, buffers=None):
    b = _alloc_buffers(1, planes, with_relations=True) if buffers is None else buffers
    b["enc_rows"].fill(-77)
    args = (mode,) if explicit else ()
    tree.batch_descend_puct(
        root, cb, 1, 1.5, 0.0, 0.0, 0, b["enc_rows"],
        b["leaf_ids"], b["path_buf"], b["path_lens"],
        b["legal_buf"], b["legal_lens"], b["term_qs"], b["is_term"],
        0, b["cache_keys"], b["rel"], *args,
    )
    return b


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", ["black_history", "white_repetition", "ep_castling"])
@pytest.mark.parametrize("planes", [146, 175])
@pytest.mark.parametrize("entry", ["walker", "batch"])
def test_native_leaf_exact_history_and_relation_parity(mode, case, planes, entry):
    board, move = _position(case)
    tree, root, cb, child, action, _ = _seed(board, move)
    mode_id = c_input_history_mode(mode)
    expected = child.encode_full(mode_id, planes - 112)
    root_before = cb.encode_full(mode_id, planes - 112).copy()
    if entry == "walker":
        enc = np.full((1, planes, 8, 8), -77, np.float32)
        rel = np.full((1, 5, 64, 64), 255, np.uint8)
        leaf, path, _, terminal = tree.walker_descend_puct(
            root, cb, 1.5, 0.0, 0.0, 0, enc, rel, mode_id,
        )
        assert terminal is None
        assert len(path) == 2
        assert leaf == tree.find_child(root, action)
    else:
        b = _batch(tree, root, cb, mode_id, planes)
        assert b["is_term"][0] == 0
        assert b["path_lens"][0] == 2
        assert b["leaf_ids"][0] == tree.find_child(root, action)
        enc, rel = b["enc_rows"], b["rel"]
    np.testing.assert_array_equal(enc[0], expected)
    np.testing.assert_array_equal(rel[0], child.compute_relations())
    np.testing.assert_array_equal(cb.encode_full(mode_id, planes - 112), root_before)
    assert cb.fen() == CBoard.from_board(board).fen()


def test_original_960_element_witness_and_cache_identity():
    board, move = _position("black_history")
    keys, rows = [], []
    for mode in range(3):
        tree, root, cb, child, _, _ = _seed(board, move)
        b = _batch(tree, root, cb, mode)
        rows.append(b["enc_rows"][0].copy())
        keys.append(b["cache_keys"][0].copy())
        np.testing.assert_array_equal(rows[-1], child.encode_full(mode, 34))
    assert [np.count_nonzero(rows[0] != row) for row in rows] == [0, 960, 960]
    assert not np.array_equal(keys[0], keys[1])
    # Rule50 uses different scales even when root-side piece history agrees.
    assert np.count_nonzero(rows[1] != rows[2]) == 64
    assert not np.array_equal(keys[1], keys[2])


@pytest.mark.parametrize("entry", ["walker", "batch"])
def test_omitted_native_history_argument_preserves_legacy(entry):
    board, move = _position("white_repetition")
    tree, root, cb, child, _, _ = _seed(board, move)
    if entry == "walker":
        enc = np.empty((1, 146, 8, 8), np.float32)
        tree.walker_descend_puct(root, cb, 1.5, 0.0, 0.0, 0, enc)
    else:
        enc = _batch(tree, root, cb, 0, explicit=False)["enc_rows"]
    np.testing.assert_array_equal(enc[0], child.encode_full(0, 34))


@pytest.mark.parametrize("entry", ["walker", "batch"])
@pytest.mark.parametrize("mode", [-1, 3])
def test_invalid_native_mode_rejected_before_tree_or_output_mutation(entry, mode):
    board, move = _position("black_history")
    tree, root, cb, _, _, _ = _seed(board, move)
    before = tree.node_count()
    b = _alloc_buffers(1, with_relations=True)
    enc = b["enc_rows"]
    enc.fill(-77)
    if entry == "walker":
        call = partial(tree.walker_descend_puct, root, cb, 1.5, 0.0, 0.0, 3,
                       enc, None, mode)
    else:
        call = partial(_batch, tree, root, cb, mode, buffers=b)
    with pytest.raises(ValueError, match="input_history_mode"):
        call()
    assert tree.node_count() == before
    assert np.all(enc == -77)
    assert all(tree.get_virtual_loss(n) == 0 for n in range(before))


class _RecordingEvaluator:
    n_slots = 2

    def __init__(self):
        self.buffers = [np.empty((4, 146, 8, 8), np.float32) for _ in range(2)]
        self.rows = []
        self.relations = []
        self.lock = threading.Lock()

    def _record(self, x, relations):
        with self.lock:
            self.rows.extend(row.copy() for row in x)
            if relations is not None:
                self.relations.extend(row.copy() for row in relations)
        return np.zeros((len(x), 4672), np.float32), np.zeros((len(x), 3), np.float32)

    def evaluate_encoded(self, x, relations=None):
        return self._record(x, relations)

    def get_input_buffer(self, bsz, slot=0):
        return self.buffers[slot][:bsz]

    def evaluate_inplace_async(self, bsz, *, slot=0, relations=None):
        pol, wdl = self._record(self.buffers[slot][:bsz], relations)
        return pol, wdl, None


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", ["black_history", "white_repetition"])
@pytest.mark.parametrize("route", ["walker", "pucv", "multi", "rpg"])
def test_checkpoint_history_reaches_real_searchworker_puct_callers(mode, case, route):
    board, move = _position(case)
    tree, root, cb, child, action, _ = _seed(board, move)
    evaluator = _RecordingEvaluator()
    worker = SearchWorker(
        evaluator, device="cpu", n_walkers=2 if route == "walker" else 1,
        gumbel_cfg=GumbelConfig(
            input_extra_features="v1", input_history_encoding=mode,
            compute_relations=True, topk=1, add_noise=False, temperature=0.0,
        ),
    )
    try:
        if route == "walker":
            assert worker._walker_pool is not None
            worker._walker_pool.run(tree=tree, root_id=root, root_cboard=cb,
                                    target_sims=1, stop_event=threading.Event())
        elif route == "pucv":
            worker.set_use_pucv(True)
            assert worker._pucv is not None
            worker._pucv.run(tree, root, cb, 1)
        elif route == "multi":
            worker.install_multi_gpu_pucv([evaluator], gather=1)
            assert worker._pucv_pool is not None
            worker._pucv_pool.run(tree=tree, root_id=root, root_cboard=cb,
                                 target_sims=1, stop_event=threading.Event())
        else:
            worker.install_root_parallel_gumbel([evaluator], gather=1)
            assert worker._rpg_pool is not None
            pol = np.full(4672, -100, np.float32)
            pol[action] = 100
            worker._rpg_pool.prepare_root(
                tree=tree, board=board, pol_logits=pol, wdl_logits=np.zeros(3, np.float32),
            )
            worker._rpg_pool.run(target_sims=6, stop_event=threading.Event())
        assert evaluator.rows
        mode_id = c_input_history_mode(mode)
        expected = {child.encode_full(mode_id, 34).tobytes(): child.compute_relations()}
        if route == "rpg":
            # Sequential halving keeps at least two candidates, even for topk=1.
            # Build independent full-history oracles for each actual candidate.
            assert worker._rpg_pool is not None
            assert worker._rpg_pool._state is not None
            arenas = worker._rpg_pool._state.arenas
            assert action in arenas
            expected = {}
            candidate_rows = set()
            for arena in arenas.values():
                assert arena.cboard is not None
                candidate_board = None
                for candidate_move in list(board.legal_moves):
                    possible = board.copy(stack=True)
                    possible.push(candidate_move)
                    if CBoard.from_board(possible).fen() == arena.cboard.fen():
                        candidate_board = possible
                        break
                assert candidate_board is not None
                candidate = CBoard.from_board(candidate_board)
                encoded = candidate.encode_full(mode_id, 34).tobytes()
                candidate_rows.add(encoded)
                expected[encoded] = candidate.compute_relations()
                for reply in list(candidate_board.legal_moves):
                    leaf_board = candidate_board.copy(stack=True)
                    leaf_board.push(reply)
                    leaf = CBoard.from_board(leaf_board)
                    expected[leaf.encode_full(mode_id, 34).tobytes()] = leaf.compute_relations()
            assert any(row.tobytes() not in candidate_rows for row in evaluator.rows), (
                "RPG did not evaluate a deeper PUCT leaf"
            )
        assert len(evaluator.relations) == len(evaluator.rows)
        for row, rel in zip(evaluator.rows, evaluator.relations, strict=True):
            if row.tobytes() not in expected:
                closest = min(expected, key=lambda key: np.count_nonzero(
                    row != np.frombuffer(key, np.float32).reshape(row.shape)))
                diff = row != np.frombuffer(closest, np.float32).reshape(row.shape)
                pytest.fail(f"input row mismatch: {np.count_nonzero(diff)} elements; "
                            f"planes {np.flatnonzero(diff.any(axis=(1, 2))).tolist()}")
            np.testing.assert_array_equal(rel, expected[row.tobytes()])
    finally:
        worker.clear_root_parallel_gumbel()
        worker.clear_multi_gpu_pucv(restore_fallback=False)
        if worker._walker_pool is not None:
            worker._walker_pool.close()


@pytest.mark.parametrize(("alias", "expected"), [(" current ", ()), (" ROOT_POV ", (1,)),
                                           (" lc0_root_meta ", (2,))])
def test_existing_history_aliases_are_normalized(alias, expected):
    assert puct_history_args(alias) == expected


@pytest.mark.parametrize("route", ["walker", "pucv", "multi", "rpg"])
def test_rooted_mode_stale_binary_rejected_before_threads_or_evaluator(route, monkeypatch):
    monkeypatch.setattr(_mcts_tree, "ABI_VERSION", 8)
    assert puct_history_args("legacy") == ()
    evaluator = _RecordingEvaluator()
    if route == "walker":
        build = partial(WalkerPool, WalkerPoolConfig(1, 1.5, 0.0, 0.2,
                         input_history_encoding="lc0_root"), evaluator)
    elif route == "pucv":
        build = partial(PucvChunker, evaluator, input_history_encoding="lc0_root_legacy_meta")
    elif route == "multi":
        build = partial(MultiGpuPucvPool,
                        MultiGpuPucvConfig(1, input_history_encoding="lc0_root"),
                        evaluators=[evaluator])
    else:
        build = partial(RootParallelGumbelPool, RootParallelGumbelConfig(1),
                        GumbelConfig(input_history_encoding="lc0_root"), evaluators=[evaluator])
    with pytest.raises(RuntimeError, match=r"PUCT input-history modes.*rebuild"):
        build()
    assert not evaluator.rows


def test_unknown_mode_fails_closed():
    with pytest.raises(ValueError, match="Unsupported input_history_encoding"):
        puct_history_args("unknown")
