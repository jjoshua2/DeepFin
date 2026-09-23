"""CPU contracts for retaining a BT4 root output during future generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.encoding.lc0 import x_to_lc0_planes
from chess_anti_engine.eval.rvg_surgery import position_fingerprints
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from scripts.bt4_generation_evaluator import BT4OnnxEvaluator
from scripts.bt4_policy_dump import compact_legal_policy
from scripts.bt4_raw_corpus_sidecar import compact_index_for_move
from scripts.gen_sf_rooted_corpus import input_tensor_key

HISTORY = "lc0_root_legacy_meta"
FEATURES = "v2_threats"
MODEL_SHA = "a" * 64


@dataclass
class Output:
    name: str
    shape: list[int | str]
    type: str


class FakeSession:
    def __init__(self, *, kind: str = "probabilities", value_dtype: np.dtype[Any] = np.dtype("float64")) -> None:
        self.kind = kind
        self.value_dtype = np.dtype(value_dtype)
        self.calls: list[tuple[list[str], np.ndarray]] = []
        native_type = {"float16": "tensor(float16)", "float32": "tensor(float)",
                       "float64": "tensor(double)"}[self.value_dtype.name]
        # Reversed order proves resolution is by explicit output name/width.
        self.outputs = [Output("native_wdl", ["batch", 3], native_type),
                        Output("native_policy", ["batch", 1858], "tensor(float)")]

    def get_outputs(self) -> list[Output]:
        return self.outputs

    def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert names == ["native_policy", "native_wdl"]
        assert list(feed) == ["planes"]
        planes = feed["planes"].copy()
        assert planes.shape[1:] == (112, 8, 8)
        self.calls.append((names, planes))
        policy = np.broadcast_to(np.linspace(-3, 3, 1858, dtype=np.float32),
                                 (len(planes), 1858)).copy()
        if self.kind == "probabilities":
            values = np.array([0.625, 0.375, 0.0], dtype=self.value_dtype)
        else:
            values = np.array([1e308, 0.0, -1e308], dtype=self.value_dtype)
        wdl = np.broadcast_to(values, (len(planes), 3)).copy()
        return [policy, wdl]


def make_evaluator(sess: FakeSession, *, history: str = HISTORY, features: str = FEATURES) -> BT4OnnxEvaluator:
    return BT4OnnxEvaluator(
        sess, input_name="planes", input_dtype=np.dtype("float32"),
        policy_output=None, wdl_output="native_wdl", wdl_kind=sess.kind,
        input_history_encoding=history, input_extra_features=features,
        model_sha256=MODEL_SHA,
    )


def encoded(board: chess.Board, *, history: str = HISTORY, features: str = FEATURES) -> np.ndarray:
    return encode_cboard(CBoard.from_board(board), input_history_encoding=history,
                         input_extra_features=features)


@pytest.mark.parametrize("fen", [
    chess.STARTING_FEN,
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
])
def test_root_retains_native_output_and_legal_mapping(fen: str) -> None:
    board = chess.Board(fen)
    sess = FakeSession()
    evaluator = make_evaluator(sess)
    x = encoded(board)
    root = evaluator.evaluate_root(board, x)
    assert evaluator.root_calls == 1
    assert evaluator.leaf_calls == 0
    assert evaluator.leaf_rows == 0
    assert len(sess.calls) == 1
    np.testing.assert_array_equal(sess.calls[0][1],
                                  x_to_lc0_planes(x[None], input_history_encoding=HISTORY))
    expected_logits = np.linspace(-3, 3, 1858, dtype=np.float32)
    moves, probabilities, dense = compact_legal_policy(board, expected_logits)
    np.testing.assert_array_equal(root.policy_t1, dense)
    assert root.policy_t1.tobytes() == dense.tobytes()
    assert root.policy_t1.dtype == np.dtype("float32")
    assert [root.policy_t1[compact_index_for_move(board, move)] for move in moves] == list(probabilities)
    assert root.wdl_raw.dtype == np.dtype("float64")
    np.testing.assert_array_equal(root.wdl_raw, np.array([0.625, 0.375, 0.0]))
    assert root.wdl_output == "native_wdl"
    assert root.wdl_kind == "probabilities"
    assert root.model_sha256 == MODEL_SHA
    assert root.input_name == "planes"
    assert root.input_dtype == "float32"
    assert root.input_history_encoding == HISTORY
    assert root.input_key == input_tensor_key(x)
    assert root.source_key == position_fingerprints(x[None], input_history_encoding=HISTORY)[0]
    assert not root.policy_t1.flags.writeable
    assert not root.wdl_raw.flags.writeable
    search_policy, search_wdl = root.search_inputs()
    assert search_policy.shape == (1, 1858)
    assert search_wdl.shape == (1, 3)
    assert np.isfinite(search_policy).all()
    assert np.isfinite(search_wdl).all()
    assert search_wdl[0, 2] == -1e9
    assert np.all(search_policy[0, dense == 0] == -1e9)
    search_policy[0, :] = 5
    np.testing.assert_array_equal(root.policy_t1, dense)


def test_root_only_sampling_consumer_never_evaluates_leaves() -> None:
    board = chess.Board()
    sess = FakeSession()
    evaluator = make_evaluator(sess)
    rng = np.random.default_rng(27)
    # A caller can sample directly from the retained T=1 root prior. Choice of
    # production sampling temperature is deliberately outside this adapter.
    for _ in range(2):
        root = evaluator.evaluate_root(board, encoded(board))
        moves = list(board.legal_moves)
        weights = np.array([root.policy_t1[compact_index_for_move(board, move)]
                            for move in moves], dtype=np.float64)
        weights /= weights.sum()
        board.push(moves[int(rng.choice(len(moves), p=weights))])
    assert evaluator.root_calls == len(sess.calls) == 2
    assert evaluator.leaf_calls == evaluator.leaf_rows == 0


def test_root_rejects_history_mismatch_before_inference() -> None:
    played = chess.Board()
    played.push_san("Nf3")
    played.push_san("Nf6")
    same_fen_no_history = chess.Board(played.fen())
    sess = FakeSession()
    evaluator = make_evaluator(sess)
    with pytest.raises(ValueError, match="board/history"):
        evaluator.evaluate_root(same_fen_no_history, encoded(played))
    assert sess.calls == []
    root = evaluator.evaluate_root(played, encoded(played))
    assert root.fen == played.fen()


def test_extreme_native_logits_remain_float64_and_search_logits_finite() -> None:
    sess = FakeSession(kind="logits")
    root = make_evaluator(sess).evaluate_root(chess.Board(), encoded(chess.Board()))
    np.testing.assert_array_equal(root.wdl_raw, np.array([1e308, 0.0, -1e308]))
    assert root.wdl_raw.dtype == np.dtype("float64")
    _, search_wdl = root.search_inputs()
    assert np.isfinite(search_wdl).all()
    assert search_wdl[0, 0] == 0
    assert np.all(search_wdl[0, 1:] == -1e9)


def test_float16_native_probabilities_are_retained_without_promotion() -> None:
    board = chess.Board()
    sess = FakeSession(value_dtype=np.dtype("float16"))
    root = make_evaluator(sess).evaluate_root(board, encoded(board))
    assert root.wdl_raw.dtype == np.dtype("float16")
    assert root.wdl_raw.tobytes() == np.array([0.625, 0.375, 0.0], dtype=np.float16).tobytes()
    assert root.search_inputs()[1].dtype == np.dtype("float32")


def test_leaf_refuses_unbound_tree_and_relations() -> None:
    sess = FakeSession()
    evaluator = make_evaluator(sess)
    x = encoded(chess.Board())[None]
    with pytest.raises(RuntimeError, match="not bound"):
        evaluator.evaluate_encoded(x)
    with pytest.raises(ValueError, match="relation inputs"):
        evaluator.evaluate_encoded(x, relations=np.zeros((1, 5, 64, 64), dtype=np.uint8))
    assert sess.calls == []


@pytest.mark.parametrize("bad_leaf", ["history", "legal"])
def test_leaf_refuses_stale_history_or_legal_mapping_before_inference(bad_leaf: str) -> None:
    played = chess.Board()
    played.push_san("Nf3")
    played.push_san("Nf6")
    cb = CBoard.from_board(played)
    legal = cb.legal_move_indices()

    class PendingTree:
        def pending_leaf_cboards(self) -> list[CBoard]:
            return [cb]

        def get_pending_legal_indices(self) -> tuple[np.ndarray, np.ndarray]:
            listed = legal[:-1] if bad_leaf == "legal" else legal
            return listed, np.array([len(listed)], dtype=np.int32)

    sess = FakeSession()
    evaluator = make_evaluator(sess)
    evaluator.bind_tree(PendingTree())
    x = encoded(chess.Board(played.fen())) if bad_leaf == "history" else encoded(played)
    with pytest.raises(ValueError, match=r"board/history|legal indices disagree"):
        evaluator.evaluate_encoded(x[None])
    assert sess.calls == []


def test_actual_c_search_maps_leaves_and_counts_only_real_rows() -> None:
    board = chess.Board()
    cb = CBoard.from_board(board)
    sess = FakeSession()
    evaluator = make_evaluator(sess)
    root = evaluator.evaluate_root(board, encoded(board))
    tree = MCTSTree()
    evaluator.bind_tree(tree)
    observed: list[tuple[int, int]] = []
    evaluate_encoded = evaluator.evaluate_encoded

    def recording_eval(x: np.ndarray, relations: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        leaves = tree.pending_leaf_cboards()
        observed.append((len(leaves), len(x)))
        policy, wdl = evaluate_encoded(x, relations)
        for idx, leaf in enumerate(leaves):
            leaf_board = chess.Board(leaf.fen())
            _, _, expected_prior = compact_legal_policy(
                leaf_board, np.linspace(-3, 3, 1858, dtype=np.float32),
            )
            positive = expected_prior > 0
            np.testing.assert_array_equal(policy[idx, positive],
                                          np.log(expected_prior[positive]).astype(np.float32))
            assert np.all(policy[idx, ~positive] == -1e9)
            assert np.isfinite(wdl[idx]).all()
        assert np.all(policy[len(leaves):] == -1e9)
        assert np.all(wdl[len(leaves):] == 0)
        return policy, wdl

    evaluator.evaluate_encoded = recording_eval  # type: ignore[method-assign]
    cfg = GumbelConfig(simulations=8, topk=4, add_noise=False,
                       input_history_encoding=HISTORY, input_extra_features=FEATURES)
    search_policy, search_wdl = root.search_inputs()
    result = run_gumbel_root_many_c(
        None, [board], device="cpu", rng=np.random.default_rng(2), cfg=cfg,
        evaluator=evaluator, pre_pol_logits=search_policy, pre_wdl_logits=search_wdl,
        cboards=[cb], tree=tree, target_batch=1,
    )
    assert len(result[0]) == 1
    assert len(result[1]) == 1
    assert evaluator.root_calls == 1
    assert evaluator.leaf_calls == len(observed) > 0
    assert evaluator.leaf_rows == sum(real for real, _ in observed)
    assert all(0 < real <= padded for real, padded in observed)
    assert any(real < padded for real, padded in observed)
    assert len(sess.calls) == 1 + evaluator.leaf_calls
    assert [len(call[1]) for call in sess.calls[1:]] == [real for real, _ in observed]
    _, _, expected = compact_legal_policy(board, np.linspace(-3, 3, 1858, dtype=np.float32))
    np.testing.assert_array_equal(root.policy_t1, expected)
    evaluator.bind_tree(None)
