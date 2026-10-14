"""Carried solved roots must play a proof, even outside the fresh shortlist."""
from __future__ import annotations
from typing import cast

import chess
import numpy as np
import pytest
from numpy.typing import NDArray

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.moves.encode import move_to_index


class _NoEvaluation:
    def evaluate_encoded(
        self, x: NDArray[np.float32], relations: NDArray[np.uint8] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        del x, relations
        raise AssertionError("a solved root must not evaluate leaves")


@pytest.mark.parametrize("status", [1, 2])
@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_carried_solved_root_preserves_proof(
    status: int, topk: int, temperature: float,
) -> None:
    board = chess.Board()
    cb = CBoard.from_board(board)
    legal = cb.legal_move_indices().astype(np.int32)
    proven = int(legal[-1])
    tree = MCTSTree()
    root = tree.add_root(1, 0.0, cb)
    tree.expand(root, legal, np.full(legal.size, 1.0 / legal.size))
    for action in legal:
        child_status = (-1 if status == 1 else 2) if int(action) == proven else 1
        tree.mark_solved_path(
            np.array([root, tree.find_child(root, int(action))], dtype=np.int32),
            child_status,
        )
    assert tree.get_solved_status(root) == status
    logits = np.zeros((1, 4672), dtype=np.float32)
    logits[0, proven] = -20.0

    probs, actions, values, _, returned_tree, roots = run_gumbel_root_many_c(
        None, [board], device="cpu", rng=np.random.default_rng(7),
        cfg=GumbelConfig(
            simulations=8, topk=topk, temperature=temperature, add_noise=False,
        ),
        evaluator=_NoEvaluation(), tree=tree, root_node_ids=[root],
        pre_pol_logits=logits, pre_wdl_logits=np.zeros((1, 3), dtype=np.float32),
    )
    assert returned_tree is tree
    assert roots == [root]
    assert actions == [proven]
    assert tree.get_gumbel_remaining() == [[proven]]
    assert probs[0][proven] == 1.0
    assert np.count_nonzero(probs[0]) == 1
    assert values == [1.0 if status == 1 else 0.0]


def test_root_solved_during_halving_recovers_the_mating_child() -> None:
    board = chess.Board()
    for san in ("f3", "e5", "g4"):
        board.push_san(san)
    mate = int(move_to_index(chess.Move.from_uci("d8h4"), board))
    other = int(move_to_index(chess.Move.from_uci("g8f6"), board))
    logits = np.full((1, 4672), -30.0, dtype=np.float32)
    logits[0, mate] = 0.0
    logits[0, other] = 20.0

    class Evaluator:
        def evaluate_encoded(
            self, x: NDArray[np.float32], relations: NDArray[np.uint8] | None = None,
        ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
            del relations
            return (
                np.zeros((x.shape[0], 4672), dtype=np.float32),
                np.zeros((x.shape[0], 3), dtype=np.float32),
            )

    probs, actions, values, _, tree, roots = run_gumbel_root_many_c(
        None, [board], device="cpu", rng=np.random.default_rng(7),
        cfg=GumbelConfig(
            simulations=8, topk=2, temperature=0.0, add_noise=False, c_scale=0.0,
        ),
        evaluator=Evaluator(), pre_pol_logits=logits,
        pre_wdl_logits=np.zeros((1, 3), dtype=np.float32),
        allow_terminal_root_shortcuts=False,
    )
    assert tree.get_solved_status(roots[0]) == 1
    assert actions == [mate]
    assert tree.get_gumbel_remaining() == [[mate]]
    assert probs[0][mate] == 1.0
    assert values == [1.0]


def _tablebase_carried_root() -> tuple[chess.Board, MCTSTree, int]:
    board = chess.Board("8/8/8/5k2/8/8/8/K2Q4 b - - 0 1")
    cb = CBoard.from_board(board)
    legal = cb.legal_move_indices().astype(np.int32)
    action = int(move_to_index(chess.Move.from_uci("f5e4"), board))
    tree = MCTSTree()
    parent = tree.add_root(1, 0.0, cb)
    tree.expand(parent, legal, np.full(legal.size, 1.0 / legal.size))
    prior = np.zeros(4672, dtype=np.float64)
    prior[legal] = 1.0 / legal.size
    pending = tree.start_gumbel_sims(
        [cb], np.array([parent], dtype=np.int32), [[action]],
        [np.zeros(4672, dtype=np.float64)], [prior],
        np.ones(1, dtype=np.int32), np.zeros(1, dtype=np.float64),
        0.1, 50.0, 2.5, 1.2, True,
        np.empty((32, 146, 8, 8), dtype=np.float32), 0, 1, 0,
    )
    assert pending == 1
    tree.mark_tb_solved(np.array([0], dtype=np.int32), np.array([1], dtype=np.int8))
    board.push_uci("f5e4")
    board.fullmove_number = 1
    child_logits = np.full((1, 4672), -30.0, dtype=np.float32)
    child_logits[0, int(move_to_index(chess.Move.from_uci("d1d4"), board))] = 20.0
    child_logits[0, int(move_to_index(chess.Move.from_uci("d1d8"), board))] = 19.0
    assert tree.continue_gumbel_sims(
        child_logits,
        np.array([[30.0, 0.0, -30.0]], dtype=np.float32),
    ) is None
    root = tree.find_child(parent, action)
    assert board.fen() == "8/8/8/8/4k3/8/8/K2Q4 w - - 1 1"
    assert tree.get_solved_status(root) == 1
    actions, _ = tree.get_children_visits(root)
    assert all(tree.get_solved_status(tree.find_child(root, int(a))) == 0 for a in actions)
    return board, tree, root


@pytest.mark.parametrize("simulations", [1, 4])
@pytest.mark.parametrize("shortcuts", [False, True])
@pytest.mark.parametrize("bad_logit", [20.0, 1000.0])
def test_witnessless_tablebase_win_continues_outside_fresh_shortlist(
    simulations: int, shortcuts: bool, bad_logit: float,
) -> None:
    board, tree, root = _tablebase_carried_root()
    bad = int(move_to_index(chess.Move.from_uci("d1d4"), board))
    bad_board = board.copy()
    bad_board.push_uci("d1d4")
    bad_board.push_uci("e4d4")
    assert bad_board.is_insufficient_material()
    class Probe:
        max_pieces = 3

        def apply(self, cboards, wdl, *, indices=None, solved_out=None):
            if indices is None and solved_out is None:
                assert cboards[0].fen().split()[:5] == board.fen().split()[:5]
                wdl[0] = [30.0, 0.0, -30.0]
                return
            assert solved_out is not None
            assert len(cboards) == 1
            # KQvK at these short move clocks is won unless black can capture
            # the queen or is stalemated. Native terminal handling runs first.
            child = chess.Board(cboards[0].fen())
            matching = []
            for move in board.legal_moves:
                expected = board.copy()
                expected.push(move)
                if child.fen().split()[:5] == expected.fen().split()[:5]:
                    matching.append(move)
            assert len(matching) == 1
            can_capture = any(
                child.piece_at(move.to_square) == chess.Piece(chess.QUEEN, chess.WHITE)
                for move in child.legal_moves
            )
            assert can_capture is (self.calls == 0)
            assert not child.is_stalemate()
            status = 2 if self.calls == 0 else -1
            solved_out[0] = status
            row = 0 if indices is None else indices[0]
            wdl[row] = [0.0, 30.0, 0.0] if status == 2 else [-30.0, 0.0, 30.0]
            self.calls += 1
            if status == -1:
                self.winning_action = int(move_to_index(matching[0], board))

        calls = 0
        winning_action = -1

    class Evaluator:
        def evaluate_encoded(self, x, relations=None):
            del relations
            return np.zeros((x.shape[0], 4672), np.float32), np.zeros((x.shape[0], 3), np.float32)

    logits = np.zeros((1, 4672), dtype=np.float32)
    logits[0, bad] = bad_logit
    probe = Probe()
    def run(budget: int):
        return run_gumbel_root_many_c(
            None, [board], device="cpu", rng=np.random.default_rng(7),
            cfg=GumbelConfig(simulations=budget, topk=1, temperature=0.0, add_noise=False),
            evaluator=Evaluator(), pre_pol_logits=logits,
            pre_wdl_logits=np.zeros((1, 3), dtype=np.float32),
            tb_probe=probe, target_batch=1,
            tree=tree, root_node_ids=[root], allow_terminal_root_shortcuts=shortcuts,
        )

    result = run(simulations)
    probs, actions, values, _, _, roots = result
    good = probe.winning_action
    assert roots == [root]
    assert probe.calls == 2
    assert actions == [good]
    assert good != bad
    assert tree.get_gumbel_remaining() == [[good]]
    assert probs[0][good] == 1.0
    assert values == [1.0]


def test_solved_loss_value_does_not_average_earlier_estimates() -> None:
    board = chess.Board("7q/8/8/8/2k5/8/8/K7 w - - 0 1")
    cb = CBoard.from_board(board)
    legal = cb.legal_move_indices().astype(np.int32)
    tree = MCTSTree()
    root = tree.add_root(0, 0.0, cb)
    tree.expand(root, legal, np.full(legal.size, 1.0 / legal.size))
    for action in legal:
        child = tree.find_child(root, int(action))
        path = np.array([root, child], dtype=np.int32)
        for _ in range(9):
            tree.backprop(path, 0.0)
        tree.backprop(path, 1.0)
        tree.mark_solved_path(path, 1)
    assert tree.get_solved_status(root) == -1
    assert tree.node_q(tree.find_child(root, int(legal[0]))) == 1.0
    _, _, values, _, _, _ = run_gumbel_root_many_c(
        None, [board], device="cpu", rng=np.random.default_rng(7),
        cfg=GumbelConfig(simulations=4, topk=1, temperature=0.0, add_noise=False),
        evaluator=_NoEvaluation(), tree=tree, root_node_ids=[root],
        pre_pol_logits=np.zeros((1, 4672), dtype=np.float32),
        pre_wdl_logits=np.zeros((1, 3), dtype=np.float32),
    )
    assert values == [-1.0]


def test_ordinary_mate_search_returns_exact_solved_loss() -> None:
    board = chess.Board("4r3/8/8/8/q7/8/3k4/K7 w - - 0 1")
    assert board.legal_moves.count() == 2
    tree = MCTSTree()

    class Evaluator:
        def evaluate_encoded(self, x, relations=None):
            del relations
            policy = np.full((x.shape[0], 4672), -30.0, np.float32)
            indices, leaves = tree.get_pending_tb_leaves(4)
            assert len(leaves) > 0
            for index, cb in zip(indices, leaves, strict=True):
                leaf = chess.Board(cb.fen())
                mates = []
                for move in leaf.legal_moves:
                    child = leaf.copy()
                    child.push(move)
                    if child.is_checkmate():
                        mates.append(int(move_to_index(move, leaf)))
                assert mates
                policy[index, mates] = 30.0
            return policy, np.zeros((x.shape[0], 3), np.float32)

    _, _, values, _, _, roots = run_gumbel_root_many_c(
        None, [board], device="cpu", rng=np.random.default_rng(7),
        cfg=GumbelConfig(simulations=16, topk=2, temperature=0.0, add_noise=False),
        evaluator=Evaluator(), tree=tree,
        pre_pol_logits=np.zeros((1, 4672), np.float32),
        pre_wdl_logits=np.zeros((1, 3), np.float32), target_batch=1,
    )
    assert tree.get_solved_status(roots[0]) == -1
    actions, visits, child_q = tree.get_children_q(roots[0], 0.0)
    assert all(tree.get_solved_status(tree.find_child(roots[0], int(a))) == 1 for a in actions)
    np.testing.assert_array_equal(visits, [8, 8])
    np.testing.assert_array_equal(child_q, [-1.0, -1.0])
    assert values == [-1.0]

    # Exercise the optional pool's real final-value method with this searched
    # CPU tree, without constructing device workers or claiming GPU coverage.
    from chess_anti_engine.uci.root_parallel_gumbel import RootParallelGumbelPool, _SearchState

    pool = object.__new__(RootParallelGumbelPool)
    state = _SearchState(
        tree=tree, root_id=roots[0], root_cb=CBoard.from_board(board), fen=board.fen(),
        pri=np.zeros(4672), search_legal=actions, root_q_init=0.0,
    )
    assert pool._final_value(state, int(actions[0])) == -1.0


@pytest.mark.parametrize("status", [-1, 1, 2])
def test_native_q_queries_report_proofs_even_without_visits(status: int) -> None:
    tree = MCTSTree()
    root = tree.add_root(0, 0.0)
    tree.expand(root, np.array([0, 1], np.int32), np.array([0.5, 0.5]))
    child = tree.find_child(root, 0)
    tree.mark_solved_path(np.array([root, child], np.int32), status)
    expected = 0.0 if status == 2 else float(status)
    assert tree.node_q(child) == expected
    actions, visits, qs = tree.get_children_q(root, 0.25)
    np.testing.assert_array_equal(actions, [0, 1])
    np.testing.assert_array_equal(visits, [0, 0])
    np.testing.assert_array_equal(qs, [-expected, 0.25])


def test_previous_extension_is_refused_before_solved_search(monkeypatch) -> None:
    import chess_anti_engine.mcts.gumbel_c as runner

    monkeypatch.setattr(runner._mcts_tree_ext, "ABI_VERSION", 6)
    with pytest.raises(RuntimeError, match="ABI_VERSION=6 < required 8"):
        run_gumbel_root_many_c(
            None, [chess.Board()], device="cpu", rng=np.random.default_rng(7),
            cfg=GumbelConfig(simulations=1, topk=1, temperature=0.0, add_noise=False),
            evaluator=_NoEvaluation(),
        )


def test_native_halving_and_parallel_policy_agree_on_proven_draw() -> None:
    from chess_anti_engine.mcts.gumbel import _completed_q_transform
    from chess_anti_engine.uci.root_parallel_gumbel import (
        RootParallelGumbelConfig, RootParallelGumbelPool, _SearchState,
    )

    board = chess.Board("7k/8/6KP/8/8/8/8/8 w - - 0 1")
    cb = CBoard.from_board(board)
    legal = cb.legal_move_indices().astype(np.int32)
    draw = int(move_to_index(chess.Move.from_uci("h6h7"), board))
    other = 3379
    assert other in legal
    tree = MCTSTree()
    root = tree.add_root(1, 0.0, cb)
    pri = np.zeros(4672, dtype=np.float64)
    pri[legal] = 0.1 / (legal.size - 2)
    pri[[draw, other]] = 0.45
    tree.expand(root, legal, pri[legal])
    for action, value in ((draw, 0.9), (other, 0.5)):
        path = np.array([root, tree.find_child(root, action)], np.int32)
        for _ in range(20):
            tree.backprop(path, -value)
    tree.mark_solved_path(np.array([root, tree.find_child(root, draw)], np.int32), 2)
    assert tree.get_solved_status(root) == 0
    pending = tree.start_gumbel_sims(
        [cb], np.array([root], np.int32), [[draw, other]],
        [np.zeros(4672)], [pri], np.array([2], np.int32), np.zeros(1),
        0.1, 50.0, 2.5, 1.2, True, np.empty((32, 146, 8, 8), np.float32), 0, 1, 0,
    )
    while pending is not None:
        pending = tree.continue_gumbel_sims(
            np.zeros((pending, 4672), np.float32),
            np.tile(np.log(np.array([[0.1, 0.3, 0.6]], np.float32)), (pending, 1)),
        )
    assert tree.get_gumbel_remaining() == [[other]]
    actions, visits, q = tree.get_children_q(root, 0.0)
    assert q[actions == draw][0] == 0.0
    transform = _completed_q_transform(
        actions=actions, priors=pri[actions], visits=visits, qvalues=q,
        raw_value=0.0, cfg=GumbelConfig(), root=True,
    )
    policy_preferred = int(actions[np.argmax(np.log(pri[actions]) + transform)])
    assert policy_preferred == other
    state = _SearchState(
        tree=tree, root_id=root, root_cb=cb, fen=board.fen(),
        pri=pri, search_legal=legal, root_q_init=0.0,
    )
    pool = object.__new__(RootParallelGumbelPool)
    pool._gcfg = GumbelConfig()
    pool._cfg = RootParallelGumbelConfig(n_groups=1)
    assert pool._score_and_halve(state, [draw, other], {}) == [other]


class _MissingChildTables:
    max_pieces = 3
    _path = ""
    hits = 0

    def reset_counts(self):
        self.hits = 0

    def __init__(self):
        self.extra = 0
        self.native = 0

    def apply(self, cboards, wdl, *, indices=None, solved_out=None):
        if solved_out is None:
            return
        if indices is None:
            self.extra += len(cboards)
            return
        for j, (index, cb) in enumerate(zip(indices, cboards, strict=True)):
            board = chess.Board(cb.fen())
            assert any(board.piece_at(m.to_square) == chess.Piece(chess.QUEEN, chess.WHITE)
                       for m in board.legal_moves)
            solved_out[j] = 2
            wdl[index] = [0.0, 30.0, 0.0]
            self.native += 1


class _NeutralEvaluator:
    def __init__(self, bad: int):
        self.bad = bad

    def evaluate_encoded(self, x, relations=None):
        del relations
        pol = np.zeros((x.shape[0], 4672), np.float32)
        pol[:, self.bad] = 20.0
        return pol, np.zeros((x.shape[0], 3), np.float32)


def _exhausted_call(board, tree, root, probe, stop_fn=None):
    bad = int(move_to_index(chess.Move.from_uci("d1d4"), board))
    pol = np.zeros((1, 4672), np.float32)
    pol[0, bad] = 20.0
    return run_gumbel_root_many_c(
        None, [board], device="cpu", rng=np.random.default_rng(7),
        cfg=GumbelConfig(simulations=1, topk=1, temperature=0.0, add_noise=False),
        evaluator=_NeutralEvaluator(bad), tree=tree, root_node_ids=[root],
        pre_pol_logits=pol, pre_wdl_logits=np.zeros((1, 3), np.float32),
        tb_probe=probe, target_batch=1, witness_stop_fn=stop_fn,
    )


@pytest.mark.parametrize("stop", [False, True])
def test_exhaustion_is_typed_and_lookup_allowance_is_bounded(monkeypatch, stop):
    import chess_anti_engine.mcts.gumbel_c as runner

    monkeypatch.setattr(runner, "_SOLVED_ROOT_TB_CHECK_LIMIT", 2)
    board, tree, root = _tablebase_carried_root()
    probe = _MissingChildTables()
    with pytest.raises(runner.SolvedRootWitnessUnavailable) as caught:
        _exhausted_call(board, tree, root, probe, lambda: stop)
    error = caught.value
    assert error.outcomes == {0: 1}
    assert error.child_checks == {0: 0 if stop else 2}
    assert probe.extra == (0 if stop else 2)
    assert probe.native == 1
    assert tree.get_children_visits(root)[1].sum() == 1
    assert tree.get_gumbel_remaining(True) == [[]]
    with pytest.raises(RuntimeError, match="no outcome-preserving"):
        tree.get_gumbel_remaining()
    assert error.partial_result[1] == [-1]
    assert not error.partial_result[0][0].any()
    bad = int(move_to_index(chess.Move.from_uci("d1d4"), board))
    assert error.fallback_actions[0] != bad


def test_uci_native_exhaustion_avoids_known_blunder_and_reports_uncertainty(monkeypatch, capsys):
    from unittest.mock import MagicMock
    import chess_anti_engine.mcts.gumbel_c as runner
    from chess_anti_engine.uci.engine import Engine
    from chess_anti_engine.uci.time_manager import SearchLimits

    monkeypatch.setattr(runner, "_SOLVED_ROOT_TB_CHECK_LIMIT", 0)
    board, tree, root = _tablebase_carried_root()
    probe = _MissingChildTables()
    worker = MagicMock()
    # Real native failure travels through the real Engine envelope. The root
    # policy deliberately prefers the known blunder and must not be consulted.
    worker.run.side_effect = lambda *_a, **_k: _exhausted_call(board, tree, root, probe)
    worker.root_policy_bestmove.return_value = "d1d4"
    engine = Engine(worker=worker)
    result = engine._run_one_phase(
        SearchLimits(deadline_ms=None, max_nodes=1, searchmoves=()),
        is_ponder=False, board=board,
    )
    assert result.bestmove_uci != "d1d4"
    assert result.bestmove_uci in {m.uci() for m in board.legal_moves}
    assert tree.get_solved_status(tree.find_child(root, int(move_to_index(
        chess.Move.from_uci(result.bestmove_uci), board)))) == 0
    worker.root_policy_bestmove.assert_not_called()
    assert result.search_incomplete == "solved_root_without_move_witness"
    assert result.score_cp == 0
    assert result.score_mate is None
    assert "outcome_preserved=UNVERIFIED" in capsys.readouterr().out


def _carrier_engine(board, tree, root, probe):
    from chess_anti_engine.uci.engine import Engine
    from chess_anti_engine.uci.search import SearchWorker

    bad = int(move_to_index(chess.Move.from_uci("d1d4"), board))
    worker = SearchWorker(
        _NeutralEvaluator(bad), device="cpu",
        gumbel_cfg=GumbelConfig(simulations=1, topk=1, temperature=0.0, add_noise=False),
        chunk_sims=1, n_walkers=1,
    )
    worker._tree, worker._root_id, worker._tree_fen = tree, root, board.fen()
    worker._root_pol_logits = np.zeros((1, 4672), np.float32)
    worker._root_pol_logits[0, bad] = 20.0
    worker._root_wdl_logits = np.zeros((1, 3), np.float32)
    worker._tb_probe = probe
    return Engine(worker=worker)


class _LaterNativeWitness(_MissingChildTables):
    def __init__(self, *, resolve=True):
        super().__init__()
        self.resolve = resolve

    def apply(self, cboards, wdl, *, indices=None, solved_out=None):
        if solved_out is None:
            return
        if indices is None:
            self.extra += len(cboards)
            return
        for j, index in enumerate(indices):
            if self.native == 0:
                solved_out[j] = 2
                wdl[index] = [0.0, 30.0, 0.0]
            elif self.resolve:
                child = chess.Board(cboards[j].fen())
                assert not child.is_stalemate()
                assert not any(child.piece_at(m.to_square) == chess.Piece(chess.QUEEN, chess.WHITE)
                               for m in child.legal_moves)
                solved_out[j] = -1
                wdl[index] = [-30.0, 0.0, 30.0]
            self.native += 1


@pytest.mark.parametrize("limit", ["nodes", "depth"])
def test_small_chunks_use_remaining_go_budget_but_honor_depth(limit, capsys):
    from chess_anti_engine.uci.time_manager import SearchLimits

    board, tree, root = _tablebase_carried_root()
    probe = _LaterNativeWitness()
    engine = _carrier_engine(board, tree, root, probe)
    result = engine._run_one_phase(
        SearchLimits(deadline_ms=None, max_nodes=2 if limit == "nodes" else None,
                     max_depth=1 if limit == "depth" else None,
                     searchmoves=tuple(m.uci() for m in board.legal_moves)),
        is_ponder=False, board=board,
    )
    assert result.bestmove_uci != "d1d4"
    if limit == "nodes":
        assert probe.native == 2
        assert result.nodes == 2
        assert result.search_incomplete is None
        assert result.score_cp > 0
        assert engine.bestmove_fallback_used == 0
    else:
        assert probe.native == 1
        assert result.search_incomplete == "solved_root_without_move_witness"
        assert "outcome_preserved=UNVERIFIED" in capsys.readouterr().out
    assert tree.get_children_visits(root)[1].sum() == probe.native


def test_witness_check_allowance_is_shared_across_go_chunks(capsys):
    from chess_anti_engine.uci.time_manager import SearchLimits

    board, tree, root = _tablebase_carried_root()
    probe = _LaterNativeWitness(resolve=False)
    engine = _carrier_engine(board, tree, root, probe)
    result = engine._run_one_phase(
        SearchLimits(deadline_ms=None, max_nodes=3,
                     searchmoves=tuple(m.uci() for m in board.legal_moves)),
        is_ponder=False, board=board,
    )
    assert probe.native == 3
    assert probe.extra == 32
    assert tree.get_children_visits(root)[1].sum() == 3
    assert result.search_incomplete == "solved_root_without_move_witness"
    assert "child_checks=32" in capsys.readouterr().out
    assert engine.bestmove_fallback_used == 1


def test_selfplay_exhaustion_adjudicates_only_failed_member_without_a_move(monkeypatch, caplog):
    from unittest.mock import MagicMock
    import chess_anti_engine.mcts.gumbel_c as runner
    from chess_anti_engine.selfplay import network_turn
    from chess_anti_engine.selfplay.config import (
        DiffFocusConfig, GameConfig, OpponentConfig, SearchConfig, TemperatureConfig,
    )
    from chess_anti_engine.selfplay.finalize import _resolve_terminal_result
    from chess_anti_engine.selfplay.opening import OpeningConfig
    from chess_anti_engine.selfplay.state import SelfplayState
    from chess_anti_engine.tablebase import SyzygyProbe

    monkeypatch.setattr(runner, "_SOLVED_ROOT_TB_CHECK_LIMIT", 0)
    board, tree, root = _tablebase_carried_root()
    bad = int(move_to_index(chess.Move.from_uci("d1d4"), board))
    state = SelfplayState.create(
        model=None, device="cpu", rng=np.random.default_rng(7), stockfish=MagicMock(),
        evaluator=_NeutralEvaluator(bad), batch_size=2, continuous=False, target=2,
        opponent=OpponentConfig(), temp=TemperatureConfig(),
        search=SearchConfig(simulations=1, fast_simulations=1, mcts_type="gumbel",
                            gumbel_topk=1, gumbel_target_batch=1),
        opening=OpeningConfig(), diff_focus=DiffFocusConfig(),
        game=GameConfig(syzygy_in_search=True, syzygy_adjudicate=True),
    )
    state.boards[0] = board
    state.cboards[0] = CBoard.from_board(board)
    state.mcts_tree = tree
    state.root_ids[0] = root
    state.tb_probe = cast(SyzygyProbe, cast(object, _MissingChildTables()))
    before = state.cboards[0].fen()
    other_before = state.cboards[1].fen()
    network_turn.run_network_turn(state, [0, 1])
    assert state.done_arr[0] == 1
    assert state.tb_result_arr == ["1-0", None]
    assert state.cboards[0].fen() == before
    assert state.cboards[1].fen() != other_before
    assert "adjudicating slot 0 result=1-0" in caplog.text
    assert _resolve_terminal_result(state, 0)[:3] == ("1-0", False, True)


@pytest.mark.parametrize("stop_reason", ["event", "deadline"])
def test_real_search_worker_passes_stop_to_child_lookup_and_retains_tree(capsys, stop_reason):
    from chess_anti_engine.uci.engine import Engine
    from chess_anti_engine.uci.search import SearchWorker
    from chess_anti_engine.uci.time_manager import SearchLimits

    board, tree, root = _tablebase_carried_root()
    bad = int(move_to_index(chess.Move.from_uci("d1d4"), board))
    worker = SearchWorker(
        _NeutralEvaluator(bad), device="cpu",
        gumbel_cfg=GumbelConfig(simulations=1, topk=1, temperature=0.0, add_noise=False),
        chunk_sims=1, n_walkers=1,
    )
    worker._tree, worker._root_id, worker._tree_fen = tree, root, board.fen()
    worker._root_pol_logits = np.zeros((1, 4672), np.float32)
    worker._root_pol_logits[0, bad] = 20.0
    worker._root_wdl_logits = np.zeros((1, 3), np.float32)
    probe = _MissingChildTables()
    worker._tb_probe = probe
    engine = Engine(worker=worker)
    if stop_reason == "event":
        engine._stop_event.set()
    # Full-support searchmoves skips the ordinary DTZ shortcut while allowing
    # the same carried root. Existing first-chunk behavior is unchanged.
    result = engine._run_one_phase(
        SearchLimits(deadline_ms=0 if stop_reason == "deadline" else None, max_nodes=1,
                     searchmoves=tuple(m.uci() for m in board.legal_moves)),
        is_ponder=False, board=board,
    )
    assert result.search_incomplete == "solved_root_without_move_witness"
    assert result.bestmove_uci != "d1d4"
    assert probe.extra == 0
    assert worker._tree is tree
    assert worker._root_id == root
    assert tree.get_children_visits(root)[1].sum() == 1
    assert "outcome_preserved=UNVERIFIED" in capsys.readouterr().out
