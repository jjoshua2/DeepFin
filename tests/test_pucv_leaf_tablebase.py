"""Native descendants with immutable fake NN outputs, no models or real tables.

Parallel cases require the safe-handle/no-eviction prerequisite (PR1244); source
preparation does not authorize executing these tests with stale native images.
"""

from __future__ import annotations

import dis
import linecache
import sys
import threading
from types import SimpleNamespace

import chess
import chess.syzygy
import numpy as np
import pytest

from chess_anti_engine import tablebase as tbmod
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts import puct_vl
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.puct_tb import descend_batch
from chess_anti_engine.moves.encode import move_to_index
from chess_anti_engine.uci import multi_gpu_pucv_pool as multi
from chess_anti_engine.uci.root_parallel_gumbel import (
    RootParallelGumbelConfig,
    RootParallelGumbelPool,
    _CandidateArena,
    _SearchState,
    _WorkItem,
)
from chess_anti_engine.uci.search import SearchWorker


class Event:
    def __init__(self):
        self.waits = 0

    def synchronize(self):
        self.waits += 1


class RawEvaluator:
    n_slots = 2

    def __init__(self, gather=4):
        self.raw = np.zeros((gather, 3), np.float32)
        self.raw.flags.writeable = False
        self.policy = np.zeros((gather, 4672), np.float32)
        self.policy.flags.writeable = False
        self.inputs = [np.zeros((gather, 146, 8, 8), np.float32) for _ in range(2)]
        self.receipts = []

    def get_input_buffer(self, n, slot=0):
        return self.inputs[slot][:n]

    def evaluate_inplace_async(self, n, *, slot=0):
        del slot
        receipt = (self.policy[:n], self.raw[:n], Event())
        self.receipts.append(receipt)
        return receipt


class Tables:
    def __init__(self, value):
        self.wdl = {"KPvK": object()}
        self.dtz = {}
        self.value = value
        self.positions = []

    def probe_wdl(self, board):
        self.positions.append(board.fen())
        if self.value is None:
            raise chess.syzygy.MissingTableError("controlled missing material")
        return self.value


def probe_for(monkeypatch, value):
    tables = Tables(value)
    monkeypatch.setattr(tbmod, "get_tablebase", lambda _path: tables)
    return tbmod.SyzygyProbe("controlled", max_pieces=3), tables


def branch(board, moves):
    tree = MCTSTree()
    tree.reserve(1024, 8192)
    root = tree.add_root(0, 0.0, CBoard.from_board(board))
    actions = np.array(
        [move_to_index(chess.Move.from_uci(m), board) for m in moves], np.int32
    )
    tree.expand(root, actions, np.full(len(actions), 1 / len(actions), np.float64))
    return tree, root, actions


def run(kind, evaluator, tree, root, board, probe, *, cache=0, sims=1, gather=1):
    if kind == "single":
        helper = puct_vl.PucvChunker(evaluator, gather=gather, eval_cache_entries=cache)
        return helper.run(tree, root, CBoard.from_board(board), sims, tb_probe=probe)
    with multi.MultiGpuPucvPool(
        multi.MultiGpuPucvConfig(n_gpus=1, gather=gather, eval_cache_entries=cache),
        evaluators=[evaluator],
    ) as pool:
        return pool.run(
            tree=tree,
            root_id=root,
            root_cboard=CBoard.from_board(board),
            target_sims=sims,
            stop_event=threading.Event(),
            tb_probe=probe,
        )


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("cache", [0, 8])
@pytest.mark.parametrize("turn", [chess.WHITE, chess.BLACK])
@pytest.mark.parametrize(("wdl", "status"), [(2, 1), (-2, -1), (0, 2), (1, 2), (-1, 2)])
def test_actual_descendant_value_solved_sign_and_raw_cache(
    monkeypatch, kind, cache, turn, wdl, status
):
    board = chess.Board(
        "7k/8/8/8/8/8/P7/K7 w - - 0 1" if turn else "k7/p7/8/8/8/8/8/7K b - - 0 1"
    )
    move = "a2a3" if turn else "a7a6"
    expected = board.copy(stack=True)
    expected.push_uci(move)
    probe, tables = probe_for(monkeypatch, wdl)
    evaluator = RawEvaluator()
    tree, root, actions = branch(board, [move])
    assert run(kind, evaluator, tree, root, board, probe, cache=cache) == 1
    child = tree.find_child(root, int(actions[0]))
    assert tables.positions == [expected.fen()]
    assert tree.get_solved_status(child) == status
    assert tree.get_solved_status(root) == (-status if status != 2 else 2)
    assert tree.node_q(root) == pytest.approx(-status if status != 2 else 0)
    assert np.array_equal(evaluator.raw, np.zeros_like(evaluator.raw))
    assert not evaluator.raw.flags.writeable
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize(("move", "value"), [("a1b1", 2), ("a2a3", None)])
def test_positive_clock_and_missing_legacy_tables_keep_nn(
    monkeypatch, kind, move, value
):
    board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
    probe, tables = probe_for(monkeypatch, value)
    tree, root, actions = branch(board, [move])
    run(kind, RawEvaluator(), tree, root, board, probe)
    assert len(tables.positions) == int(value is None)
    assert tree.get_solved_status(tree.find_child(root, int(actions[0]))) == 0
    assert probe.hits == 0
    assert tree.node_q(root) == pytest.approx(0)
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))


def test_native_sparse_snapshots_are_owned_and_off_abi_unchanged():
    board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
    tree, root, _ = branch(board, ["a2a3", "a1b1"])
    bufs = puct_vl._alloc_buffers(2)
    probe = tbmod.SyzygyProbe("not opened", max_pieces=3)
    snapshot = descend_batch(
        tree,
        root,
        CBoard.from_board(board),
        2,
        1.4,
        0,
        0.2,
        3,
        np.empty((2, 146, 8, 8), np.float32),
        bufs,
        0,
        None,
        None,
        (),
        probe,
    )
    rows, boards = snapshot
    assert rows.tolist() == [0]
    expected = board.copy(stack=True)
    expected.push_uci("a2a3")
    assert boards[0].fen() == expected.fen()
    saved = boards[0].fen()
    puct_vl.release_batch_rows(tree, bufs, 2, vloss_weight=3)
    # A different descend cannot replace the prior call's copied board.
    other, rid, _ = branch(chess.Board(), ["e2e4"])
    assert (
        descend_batch(
            other,
            rid,
            CBoard.from_board(chess.Board()),
            1,
            1.4,
            0,
            0.2,
            0,
            np.empty((1, 146, 8, 8), np.float32),
            bufs,
            0,
            None,
            None,
            (),
            probe,
        )[1]
        == []
    )
    assert boards[0].fen() == saved
    old = tree.batch_descend_puct(
        root,
        CBoard.from_board(board),
        1,
        1.4,
        0,
        0.2,
        0,
        np.empty((1, 146, 8, 8), np.float32),
        bufs["leaf_ids"],
        bufs["path_buf"],
        bufs["path_lens"],
        bufs["legal_buf"],
        bufs["legal_lens"],
        bufs["term_qs"],
        bufs["is_term"],
    )
    assert old == 1


@pytest.mark.parametrize("kind", ["single", "multi"])
def test_sparse_rows_keep_positive_clock_pending_slot_separate(monkeypatch, kind):
    board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
    probe, tables = probe_for(monkeypatch, -2)
    tree, root, actions = branch(board, ["a2a3", "a1b1"])
    run(kind, RawEvaluator(), tree, root, board, probe, sims=2, gather=1)
    assert len(tables.positions) == 1
    assert tree.get_solved_status(tree.find_child(root, int(actions[0]))) == -1
    assert tree.get_solved_status(tree.find_child(root, int(actions[1]))) == 0
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("cache", [0, 8])
def test_cache_hit_is_raw_and_reprobed_per_fresh_tree(monkeypatch, kind, cache):
    board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
    probe, tables = probe_for(monkeypatch, 2)
    ev = RawEvaluator()
    if kind == "single":
        helper = puct_vl.PucvChunker(ev, gather=1, eval_cache_entries=cache)
    else:
        helper = multi.MultiGpuPucvPool(
            multi.MultiGpuPucvConfig(1, gather=1, eval_cache_entries=cache),
            evaluators=[ev],
        )
    try:
        for value in (2, -2):
            tables.value = value
            tree, root, actions = branch(board, ["a2a3"])
            if kind == "single":
                helper.run(tree, root, CBoard.from_board(board), 1, tb_probe=probe)
            else:
                helper.run(
                    tree=tree,
                    root_id=root,
                    root_cboard=CBoard.from_board(board),
                    target_sims=1,
                    stop_event=threading.Event(),
                    tb_probe=probe,
                )
            assert (
                tree.get_solved_status(tree.find_child(root, int(actions[0])))
                == value // 2
            )
        assert len(ev.receipts) == (1 if cache else 2)
        assert len(tables.positions) == 2
        assert np.array_equal(ev.raw, np.zeros_like(ev.raw))
    finally:
        if kind == "multi":
            helper.close()


@pytest.mark.parametrize("pawn_move", [True, False])
@pytest.mark.parametrize("turn", [chess.WHITE, chess.BLACK])
@pytest.mark.parametrize(
    ("value", "status"), [(2, 1), (-2, -1), (1, 2), (-1, 2), (None, 0)]
)
def test_rpg_first_forced_candidate_is_probed_before_backprop(
    monkeypatch, turn, value, status, pawn_move
):
    board = chess.Board(
        "7k/8/8/8/8/8/P7/K7 w - - 0 1" if turn else "k7/p7/8/8/8/8/8/7K b - - 0 1"
    )
    move = ("a2a3" if turn else "a7a6") if pawn_move else ("a1b1" if turn else "a8b8")
    status = status if pawn_move else 0
    probe, tables = probe_for(monkeypatch, value)
    tree, root, actions = branch(board, [move])
    cid = tree.find_child(root, int(actions[0]))
    arena = _CandidateArena(int(actions[0]), cid)
    st = _SearchState(
        tree,
        root,
        CBoard.from_board(board),
        board.fen(),
        np.zeros(4672),
        actions,
        0.0,
        tb_probe=probe,
    )
    pool = object.__new__(RootParallelGumbelPool)
    pool._cfg = RootParallelGumbelConfig(1)
    pool._gcfg = GumbelConfig(input_extra_features="v1")
    ev = RawEvaluator()
    assert pool._expand_candidate(ev, arena, st) == 1
    expected = board.copy(stack=True)
    expected.push_uci(move)
    assert tables.positions == ([expected.fen()] if pawn_move else [])
    assert arena.expanded
    assert tree.get_solved_status(cid) == status
    assert tree.get_solved_status(root) == (-status if status not in (0, 2) else status)
    assert arena.terminal_value == (
        None if status == 0 else 0 if status == 2 else status
    )
    assert np.array_equal(ev.raw, np.zeros_like(ev.raw))


def test_rpg_descendant_solved_path_includes_global_root(monkeypatch):
    board = chess.Board("7k/8/8/8/1p6/8/P7/K7 w - - 0 1")
    probe, tables = probe_for(monkeypatch, 2)
    ev = RawEvaluator()
    tree, root, actions = branch(board, ["a2a3"])
    cid = tree.find_child(root, int(actions[0]))
    cb = CBoard.from_board(board)
    cb.push_index(int(actions[0]))
    child_board = board.copy(stack=True)
    child_board.push_uci("a2a3")
    reply = move_to_index(chess.Move.from_uci("b4a3"), child_board)
    tree.expand(cid, np.array([reply], np.int32), np.array([1.0]))
    arena = _CandidateArena(int(actions[0]), cid, cb, expanded=True)
    with RootParallelGumbelPool(
        RootParallelGumbelConfig(1, gather=1),
        GumbelConfig(input_extra_features="v1"),
        evaluators=[ev],
    ) as pool:
        pool._state = _SearchState(
            tree,
            root,
            CBoard.from_board(board),
            board.fen(),
            np.zeros(4672),
            actions,
            0,
            tb_probe=probe,
        )
        chunker = pool._chunkers[0]
        assert chunker is not None
        assert (
            pool._run_item(0, ev, chunker, _WorkItem(arena, 1, 0), threading.Event())
            == 1
        )
    child_board.push_uci("b4a3")
    assert tables.positions == [child_board.fen()]
    assert tree.get_solved_status(cid) == -1
    assert tree.get_solved_status(root) == 1
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("boundary", ["registration", "probe", "mark"])
def test_tb_fault_drains_receipts_before_once_only_vl_release(
    monkeypatch, kind, boundary
):
    board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
    probe, _ = probe_for(monkeypatch, 2)
    tree, root, _ = branch(board, ["a2a3"])
    ev = RawEvaluator()
    released, fired = [], []
    original_release = puct_vl.release_batch_rows

    def release(tree_arg, bufs, n, **kwargs):
        assert all(receipt[2].waits >= 1 for receipt in ev.receipts)
        released.append(n)
        return original_release(tree_arg, bufs, n, **kwargs)

    monkeypatch.setattr(puct_vl, "release_batch_rows", release)

    def fail(*_args, **_kwargs):
        raise MemoryError("controlled TB boundary")

    if boundary == "probe":
        monkeypatch.setattr(probe, "apply", fail)
    if boundary == "mark":
        monkeypatch.setattr(
            puct_vl if kind == "single" else multi, "mark_batch_solved", fail
        )
    pipeline = (
        puct_vl.PucvChunker._run_pipelined
        if kind == "single"
        else multi.MultiGpuPucvPool._pipeline_until_done
    )
    instructions = {i.offset: i for i in dis.get_instructions(pipeline)}

    def trace(frame, event, arg):
        del arg
        if frame.f_code is pipeline.__code__:
            frame.f_trace_opcodes = True
            ins = instructions.get(frame.f_lasti)
            line = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
            if (
                boundary == "registration"
                and not fired
                and len(ev.receipts) == 2
                and event == "opcode"
                and ins is not None
                and ins.opname == "STORE_SUBSCR"
                and line.startswith("receipts[")
            ):
                fired.append(True)
                raise MemoryError("controlled TB boundary")
        return trace

    previous, previous_threads = sys.gettrace(), threading.gettrace()
    sys.settrace(trace)
    threading.settrace(trace)
    try:
        with pytest.raises(MemoryError, match="controlled TB boundary"):
            run(kind, ev, tree, root, board, probe, sims=2, gather=1)
    finally:
        sys.settrace(previous)
        threading.settrace(previous_threads)
    assert len(ev.receipts) == 2
    assert sum(released) == (1 if boundary == "mark" else 2)
    assert bool(fired) == (boundary == "registration")
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))


@pytest.mark.parametrize("route", ["single", "multi", "rpg"])
def test_search_dispatch_passes_the_live_probe_to_each_route(route):
    worker = object.__new__(SearchWorker)
    board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
    worker._tree, worker._root_id, _ = branch(board, ["a2a3"])
    worker._pucv_cboard = worker._pucv_pool_cboard = CBoard.from_board(board)
    worker._walker_pool = worker._pucv = worker._pucv_pool = worker._rpg_pool = None
    observed = []

    def consume(**kwargs):
        observed.append(kwargs["tb_probe"])
        return (0.0, -1) if route == "rpg" else 1

    helper = SimpleNamespace(
        run=consume, last_stats=lambda: SimpleNamespace(phases=(), root_finished=False)
    )
    setattr(
        worker,
        {"single": "_pucv", "multi": "_pucv_pool", "rpg": "_rpg_pool"}[route],
        helper,
    )
    probe = tbmod.SyzygyProbe("not opened", max_pieces=3)
    worker._run_one_chunk(1, board, threading.Event(), probe, None, True)
    assert observed == [probe]


def test_multi_workers_keep_distinct_owned_snapshots(monkeypatch):
    board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
    probe, tables = probe_for(monkeypatch, 2)
    tree, root, actions = branch(board, ["a2a3", "a2a4"])
    barrier = threading.Barrier(2)

    class ConcurrentEvaluator(RawEvaluator):
        def evaluate_inplace_async(self, n, *, slot=0):
            barrier.wait(timeout=2)
            return super().evaluate_inplace_async(n, slot=slot)

    evaluators = [ConcurrentEvaluator(), ConcurrentEvaluator()]
    with multi.MultiGpuPucvPool(
        multi.MultiGpuPucvConfig(2, gather=1), evaluators=evaluators
    ) as pool:
        assert (
            pool.run(
                tree=tree,
                root_id=root,
                root_cboard=CBoard.from_board(board),
                target_sims=2,
                stop_event=threading.Event(),
                tb_probe=probe,
            )
            == 2
        )
    expected = []
    for move in ("a2a3", "a2a4"):
        child = board.copy(stack=True)
        child.push_uci(move)
        expected.append(child.fen())
    assert sorted(tables.positions) == sorted(expected)
    assert all(len(ev.receipts) == 1 for ev in evaluators)
    assert all(
        tree.get_solved_status(tree.find_child(root, int(action))) == 1
        for action in actions
    )
    assert tree.get_solved_status(root) == -1
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))


@pytest.mark.parametrize(
    ("fen", "move", "capacity", "terminal"),
    [
        (chess.STARTING_FEN, "e2e4", 3, False),
        ("4k3/8/8/8/8/8/P7/4K2R w K - 0 1", "a2a3", 4, False),
        ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", "f7g7", 3, True),
    ],
)
def test_native_batch_filters_material_castling_and_terminal(
    fen, move, capacity, terminal
):
    board = chess.Board(fen)
    tree, root, _ = branch(board, [move])
    bufs = puct_vl._alloc_buffers(1)
    snapshots = descend_batch(
        tree,
        root,
        CBoard.from_board(board),
        1,
        1.4,
        0,
        0.2,
        3,
        np.empty((1, 146, 8, 8), np.float32),
        bufs,
        0,
        None,
        None,
        (),
        tbmod.SyzygyProbe("not opened", max_pieces=capacity),
    )
    rows, boards = snapshots
    assert rows.size == 0
    assert boards == []
    assert bool(bufs["is_term"][0]) == terminal
    puct_vl.release_batch_rows(tree, bufs, 1, vloss_weight=3)
    assert all(tree.get_virtual_loss(i) == 0 for i in range(tree.node_count()))


@pytest.mark.parametrize("planes", [146, 175])
@pytest.mark.parametrize("history_args", [(), (1,), (2,)])
def test_snapshot_api_preserves_old_encoding_and_full_history(planes, history_args):
    board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
    board.push_uci("a1b1")
    board.push_uci("h8g8")
    expected = board.copy(stack=True)
    expected.push_uci("a2a3")
    outputs = []
    for probe in (None, tbmod.SyzygyProbe("not opened", max_pieces=3)):
        tree, root, _ = branch(board, ["a2a3"])
        bufs = puct_vl._alloc_buffers(1, planes)
        enc = np.empty((1, planes, 8, 8), np.float32)
        snapshot = descend_batch(
            tree,
            root,
            CBoard.from_board(board),
            1,
            1.4,
            0,
            0.2,
            0,
            enc,
            bufs,
            0,
            None,
            None,
            history_args,
            probe,
        )
        if probe is not None:
            assert snapshot[0].tolist() == [0]
            assert snapshot[1][0].fen() == expected.fen()
        outputs.append((enc, bufs))
    assert np.array_equal(outputs[0][0], outputs[1][0])
    for name in ("leaf_ids", "path_lens", "legal_lens", "is_term"):
        assert np.array_equal(outputs[0][1][name], outputs[1][1][name])
    for name, length in (
        ("path_buf", int(outputs[0][1]["path_lens"][0])),
        ("legal_buf", int(outputs[0][1]["legal_lens"][0])),
    ):
        assert np.array_equal(
            outputs[0][1][name][:length], outputs[1][1][name][:length]
        )
