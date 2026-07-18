from typing import Any, cast

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.encoding import encode_positions_batch
from chess_anti_engine.inference import _policy_output
from chess_anti_engine.mcts.gumbel import (
    GumbelConfig,
    _completed_q,
    _select_top_m_with_gumbel,
    run_gumbel_root_many,
)
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.puct import Node, _select_child
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves import (
    POLICY_ENCODING_LC0_1858,
    POLICY_SIZE,
    legal_move_mask,
    move_to_index,
    policy_batch_to_encoding,
)

try:
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
except ImportError:  # pragma: no cover - extension absent
    run_gumbel_root_many_c = None


def _require_run_gumbel_root_many_c():
    assert run_gumbel_root_many_c is not None
    return run_gumbel_root_many_c


def test_gumbel_scale_zero_matches_noise_disabled() -> None:
    legal = np.array([1, 2, 3, 4], dtype=np.int64)
    pri = np.zeros(POLICY_SIZE, dtype=np.float64)
    pri[legal] = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    with_noise_zero, g_zero = _select_top_m_with_gumbel(
        legal=legal,
        pri=pri,
        sim_budget=8,
        topk=2,
        add_noise=True,
        gumbel_scale=0.0,
        rng=np.random.default_rng(1),
    )
    no_noise, g_off = _select_top_m_with_gumbel(
        legal=legal,
        pri=pri,
        sim_budget=8,
        topk=2,
        add_noise=False,
        gumbel_scale=1.0,
        rng=np.random.default_rng(2),
    )

    assert with_noise_zero == no_noise
    assert all(v == 0.0 for v in g_zero.values())
    assert all(v == 0.0 for v in g_off.values())


def test_gumbel_root_many_empty_diagnostics_shape() -> None:
    result = run_gumbel_root_many(
        None,
        [],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=GumbelConfig(input_extra_features="v1", ),
        return_diagnostics=True,
    )

    assert result == ([], [], [], [], [])


def _tiny_model() -> torch.nn.Module:
    model = build_model(
        ModelConfig(input_extra_features="v1",
            kind="tiny",
            embed_dim=64,
            num_layers=1,
            num_heads=4,
            ffn_mult=2,
            use_smolgen=False,
            use_nla=False,
        )
    )
    model.eval()
    return model


class _ZeroEvaluator:
    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations  # interface conformance for BatchEvaluator
        n = int(x.shape[0])
        return np.zeros((n, POLICY_SIZE), dtype=np.float32), np.zeros((n, 3), dtype=np.float32)


class _CompactZeroEvaluator:
    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations  # interface conformance for BatchEvaluator
        n = int(x.shape[0])
        full = np.zeros((n, POLICY_SIZE), dtype=np.float32)
        compact = policy_batch_to_encoding(full, policy_encoding=POLICY_ENCODING_LC0_1858)
        return compact, np.zeros((n, 3), dtype=np.float32)


class _CompactRootParityEvaluator:
    """Return identical BF16-representable logits through dense/compact APIs."""

    supports_input_bf16_bits = True
    supports_legal_bf16 = True

    def __init__(self, *, compact_root: bool) -> None:
        self.supports_compact_root_policy = compact_root
        logits = (torch.arange(POLICY_SIZE, dtype=torch.float32) % 31 - 15) / 8
        self._policy = logits.to(torch.bfloat16).float().numpy()

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        n = int(x.shape[0])
        return np.broadcast_to(self._policy, (n, POLICY_SIZE)).copy(), np.zeros(
            (n, 3), dtype=np.float32,
        )

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        del legal_counts
        compact = self._policy[np.asarray(legal_flat, dtype=np.intp)]
        bits = torch.from_numpy(compact).to(torch.bfloat16).view(torch.uint16).numpy()
        return bits, np.zeros((int(x.shape[0]), 3), dtype=np.float32)


_MATE_VS_STALEMATE_FEN = "5k2/1R6/P4BB1/P7/2P5/8/3K3P/8 w - - 5 70"
_STALEMATE_ONLY_FEN = "7k/8/6KP/8/8/8/8/8 w - - 0 1"


@pytest.mark.skipif(run_gumbel_root_many_c is None, reason="C tree extension not available")
def test_gumbel_c_compact_root_policy_matches_dense_root_mapping() -> None:
    run_c = _require_run_gumbel_root_many_c()
    boards = [
        chess.Board(),
        chess.Board(),
        chess.Board("8/8/8/8/8/8/4k3/4K3 w - - 0 1"),
        # Checkmated root (fool's mate): game-over board with ZERO legal moves
        # in a mixed batch — exercises the count-0 compact-transport row.
        chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"),
    ]
    boards[1].push_san("e4")
    assert boards[3].is_checkmate()
    allowed = [
        None,
        {
            move_to_index(chess.Move.from_uci("c7c5"), boards[1]),
            move_to_index(chess.Move.from_uci("e7e5"), boards[1]),
        },
        None,
        None,
    ]
    cfg = GumbelConfig(
        input_extra_features="v1",
        simulations=8,
        topk=4,
        temperature=0.0,
        add_noise=False,
    )

    dense = run_c(
        None,
        boards,
        device="cpu",
        rng=np.random.default_rng(7),
        cfg=cfg,
        evaluator=_CompactRootParityEvaluator(compact_root=False),
        allowed_root_indices_batch=allowed,
    )
    compact = run_c(
        None,
        boards,
        device="cpu",
        rng=np.random.default_rng(7),
        cfg=cfg,
        evaluator=_CompactRootParityEvaluator(compact_root=True),
        allowed_root_indices_batch=allowed,
    )

    for dense_arrs, compact_arrs in zip(
        (dense[0], dense[3]), (compact[0], compact[3]), strict=True,
    ):
        for dense_arr, compact_arr in zip(dense_arrs, compact_arrs, strict=True):
            assert np.array_equal(dense_arr, compact_arr)
    assert dense[1] == compact[1]
    assert dense[2] == compact[2]


def _mate_vs_stalemate_logits(board: chess.Board) -> tuple[np.ndarray, np.ndarray, int, int]:
    mate_idx = int(move_to_index(chess.Move.from_uci("b7b8"), board))
    stalemate_idx = int(move_to_index(chess.Move.from_uci("g6f7"), board))
    pre_pol = np.full((1, POLICY_SIZE), -100.0, dtype=np.float32)
    pre_pol[0, stalemate_idx] = 100.0
    pre_wdl = np.zeros((1, 3), dtype=np.float32)
    return pre_pol, pre_wdl, mate_idx, stalemate_idx


def test_gumbel_python_takes_immediate_mate_over_high_prior_stalemate():
    board = chess.Board(_MATE_VS_STALEMATE_FEN)
    pre_pol, pre_wdl, mate_idx, stalemate_idx = _mate_vs_stalemate_logits(board)

    probs, actions, values, masks = run_gumbel_root_many(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=GumbelConfig(input_extra_features="v1", simulations=1, topk=1, temperature=0.0, add_noise=False),
        evaluator=cast(Any, object()),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
    )

    assert actions == [mate_idx]
    assert values == [1.0]
    assert probs[0][mate_idx] == 1.0
    assert probs[0][stalemate_idx] == 0.0
    assert np.array_equal(masks[0], legal_move_mask(board))


@pytest.mark.skipif(run_gumbel_root_many_c is None, reason="C tree extension not available")
def test_gumbel_c_takes_immediate_mate_over_high_prior_stalemate():
    board = chess.Board(_MATE_VS_STALEMATE_FEN)
    pre_pol, pre_wdl, mate_idx, stalemate_idx = _mate_vs_stalemate_logits(board)

    run_gumbel_root_many_c = _require_run_gumbel_root_many_c()
    probs, actions, values, masks = run_gumbel_root_many_c(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=GumbelConfig(input_extra_features="v1", simulations=1, topk=1, temperature=0.0, add_noise=False),
        evaluator=cast(Any, object()),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
    )[:4]

    assert actions == [mate_idx]
    assert values == [1.0]
    assert probs[0][mate_idx] == 1.0
    assert probs[0][stalemate_idx] == 0.0
    assert np.array_equal(masks[0], legal_move_mask(board))


@pytest.mark.skipif(run_gumbel_root_many_c is None, reason="C tree extension not available")
def test_gumbel_c_can_disable_terminal_root_shortcuts():
    board = chess.Board(_MATE_VS_STALEMATE_FEN)
    pre_pol, pre_wdl, mate_idx, stalemate_idx = _mate_vs_stalemate_logits(board)

    run_gumbel_root_many_c = _require_run_gumbel_root_many_c()
    probs, actions, values, masks = run_gumbel_root_many_c(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=GumbelConfig(input_extra_features="v1", simulations=1, topk=1, temperature=0.0, add_noise=False),
        evaluator=cast(Any, object()),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
        allow_terminal_root_shortcuts=False,
    )[:4]

    assert actions == [stalemate_idx]
    assert actions[0] != mate_idx
    assert values[0] != 1.0
    assert probs[0][stalemate_idx] == 1.0
    assert np.array_equal(masks[0], legal_move_mask(board))


def _stalemate_only_logits(
    board: chess.Board,
    *,
    root_wdl_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    stalemate_move = None
    for move in board.legal_moves:
        after = board.copy(stack=False)
        after.push(move)
        if after.is_stalemate():
            stalemate_move = move
            break
    assert stalemate_move is not None
    stalemate_idx = int(move_to_index(stalemate_move, board))
    pre_pol = np.full((1, POLICY_SIZE), -100.0, dtype=np.float32)
    pre_pol[0, stalemate_idx] = 100.0
    return pre_pol, root_wdl_logits.astype(np.float32), stalemate_idx


def test_gumbel_python_avoids_high_prior_stalemate_when_root_is_winning():
    board = chess.Board(_STALEMATE_ONLY_FEN)
    pre_pol, pre_wdl, stalemate_idx = _stalemate_only_logits(
        board,
        root_wdl_logits=np.array([[10.0, -10.0, -10.0]], dtype=np.float32),
    )

    probs, actions, values, masks = run_gumbel_root_many(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=GumbelConfig(input_extra_features="v1", simulations=1, topk=1, temperature=0.0, add_noise=False),
        evaluator=_ZeroEvaluator(),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
    )

    assert actions[0] != stalemate_idx
    assert probs[0][stalemate_idx] == 0.0
    assert np.isfinite(values[0])
    assert np.array_equal(masks[0], legal_move_mask(board))


@pytest.mark.skipif(run_gumbel_root_many_c is None, reason="C tree extension not available")
def test_gumbel_c_avoids_high_prior_stalemate_when_root_is_winning():
    board = chess.Board(_STALEMATE_ONLY_FEN)
    pre_pol, pre_wdl, stalemate_idx = _stalemate_only_logits(
        board,
        root_wdl_logits=np.array([[10.0, -10.0, -10.0]], dtype=np.float32),
    )

    run_gumbel_root_many_c = _require_run_gumbel_root_many_c()
    probs, actions, values, masks = run_gumbel_root_many_c(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=GumbelConfig(input_extra_features="v1", simulations=1, topk=1, temperature=0.0, add_noise=False),
        evaluator=_ZeroEvaluator(),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
    )[:4]

    assert actions[0] != stalemate_idx
    assert probs[0][stalemate_idx] == 0.0
    assert np.isfinite(values[0])
    assert np.array_equal(masks[0], legal_move_mask(board))


@pytest.mark.skipif(run_gumbel_root_many_c is None, reason="C tree extension not available")
def test_gumbel_c_rebuilds_reused_root_missing_allowed_action():
    board = chess.Board(_STALEMATE_ONLY_FEN)
    pre_pol, pre_wdl, stalemate_idx = _stalemate_only_logits(
        board,
        root_wdl_logits=np.array([[10.0, -10.0, -10.0]], dtype=np.float32),
    )
    cfg = GumbelConfig(input_extra_features="v1", simulations=1, topk=1, temperature=0.0, add_noise=False)
    tree = MCTSTree()

    run_gumbel_root_many_c = _require_run_gumbel_root_many_c()
    first = run_gumbel_root_many_c(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=cfg,
        evaluator=_ZeroEvaluator(),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
        tree=tree,
    )
    _probs, actions, _values, _masks, tree, root_ids = first[:6]
    old_root = root_ids[0]
    assert actions[0] != stalemate_idx
    assert tree.find_child(old_root, stalemate_idx) == -1

    second = run_gumbel_root_many_c(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=cfg,
        evaluator=_ZeroEvaluator(),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
        tree=tree,
        root_node_ids=[old_root],
        allowed_root_indices_batch=[{stalemate_idx}],
    )
    _probs, actions, _values, _masks, tree, root_ids = second[:6]
    new_root = root_ids[0]

    assert actions == [stalemate_idx]
    assert new_root != old_root
    assert tree.find_child(new_root, stalemate_idx) != -1


@pytest.mark.skipif(run_gumbel_root_many_c is None, reason="C tree extension not available")
def test_gumbel_c_allows_high_prior_stalemate_when_root_is_losing():
    board = chess.Board(_STALEMATE_ONLY_FEN)
    pre_pol, pre_wdl, stalemate_idx = _stalemate_only_logits(
        board,
        root_wdl_logits=np.array([[-10.0, -10.0, 10.0]], dtype=np.float32),
    )

    run_gumbel_root_many_c = _require_run_gumbel_root_many_c()
    probs, actions, values, masks = run_gumbel_root_many_c(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=GumbelConfig(input_extra_features="v1", simulations=1, topk=1, temperature=0.0, add_noise=False),
        evaluator=cast(Any, object()),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
    )[:4]

    assert actions[0] == stalemate_idx
    assert probs[0][stalemate_idx] == 1.0
    assert values[0] == pytest.approx(0.0)
    assert np.array_equal(masks[0], legal_move_mask(board))


def test_gumbel_root_many_returns_values_for_edge_cases():
    # Tiny CPU model is enough; we just want shape/length invariants.
    m = build_model(
        ModelConfig(input_extra_features="v1",
            kind="tiny",
            embed_dim=64,
            num_layers=1,
            num_heads=4,
            ffn_mult=2,
            use_smolgen=False,
            use_nla=False,
        )
    )
    m.eval()

    # 0 legal moves: checkmate.
    b0 = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    assert b0.is_checkmate()

    # 1 legal move: king capture.
    b1 = chess.Board("7k/6Q1/4K3/8/8/8/8/8 b - - 0 1")
    assert len(list(b1.legal_moves)) == 1

    rng = np.random.default_rng(0)
    probs_list, actions, values, _masks = run_gumbel_root_many(
        m,
        [b0, b1, chess.Board()],
        device="cpu",
        rng=rng,
        cfg=GumbelConfig(input_extra_features="v1", simulations=8, topk=8, temperature=1.0),
    )

    assert len(probs_list) == 3
    assert len(actions) == 3
    assert len(values) == 3

    for probs in probs_list:
        assert probs.shape[0] > 1000
        assert float(probs.sum()) <= 1.01
        assert float(probs.sum()) >= -1e-6

    assert np.array_equal(_masks[2], legal_move_mask(chess.Board()))
    assert np.count_nonzero(probs_list[2][~_masks[2]]) == 0


def test_gumbel_root_many_empty_batch_returns_four_lists():
    probs_list, actions, values, masks = run_gumbel_root_many(
        None,
        [],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=GumbelConfig(input_extra_features="v1", simulations=8, topk=8, temperature=1.0),
        evaluator=cast(Any, lambda x: (x, x)),
    )

    assert probs_list == []
    assert actions == []
    assert values == []
    assert masks == []


def test_gumbel_root_many_precomputed_logits_matches_direct_path():
    model = _tiny_model()

    boards = [chess.Board()]
    b2 = chess.Board()
    for san in ["d4", "Nf6", "c4", "e6", "Nc3"]:
        b2.push_san(san)
    boards.append(b2)

    xs = encode_positions_batch(boards, add_features=True)
    with torch.no_grad():
        out = model(torch.from_numpy(xs))
    policy_out = _policy_output(out)
    pre_pol = policy_out.detach().float().cpu().numpy()
    pre_wdl = out["wdl"].detach().float().cpu().numpy()

    cfg = GumbelConfig(input_extra_features="v1",
        simulations=8,
        topk=8,
        temperature=0.0,
        add_noise=False,
    )

    direct = run_gumbel_root_many(
        model,
        boards,
        device="cpu",
        rng=np.random.default_rng(123),
        cfg=cfg,
    )
    precomputed = run_gumbel_root_many(
        model,
        boards,
        device="cpu",
        rng=np.random.default_rng(123),
        cfg=cfg,
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
    )

    probs_direct, actions_direct, values_direct, masks_direct = direct
    probs_pre, actions_pre, values_pre, masks_pre = precomputed

    assert actions_direct == actions_pre
    np.testing.assert_allclose(values_direct, values_pre, atol=1e-7)
    for p0, p1, m0, m1, board in zip(probs_direct, probs_pre, masks_direct, masks_pre, boards, strict=True):
        np.testing.assert_allclose(p0, p1, atol=1e-7)
        assert np.array_equal(m0, m1)
        assert np.array_equal(m0, legal_move_mask(board))
        assert np.count_nonzero(p0[~m0]) == 0


def test_gumbel_root_many_accepts_compact_precomputed_logits():
    board = chess.Board()
    full_pol = np.zeros((1, POLICY_SIZE), dtype=np.float32)
    compact_pol = policy_batch_to_encoding(full_pol, policy_encoding=POLICY_ENCODING_LC0_1858)
    pre_wdl = np.zeros((1, 3), dtype=np.float32)

    probs, actions, values, masks = run_gumbel_root_many(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(123),
        cfg=GumbelConfig(input_extra_features="v1",
            simulations=1,
            topk=8,
            temperature=0.0,
            add_noise=False,
            policy_encoding=POLICY_ENCODING_LC0_1858,
        ),
        evaluator=cast(Any, _ZeroEvaluator()),
        pre_pol_logits=compact_pol,
        pre_wdl_logits=pre_wdl,
    )

    assert len(actions) == 1
    assert len(values) == 1
    assert probs[0].shape == (POLICY_SIZE,)
    assert np.array_equal(masks[0], legal_move_mask(board))
    assert np.count_nonzero(probs[0][~masks[0]]) == 0


def test_gumbel_root_many_honors_per_game_simulation_budgets():
    boards = [chess.Board(), chess.Board()]
    pre_pol = np.zeros((2, POLICY_SIZE), dtype=np.float32)
    pre_wdl = np.zeros((2, 3), dtype=np.float32)

    result = run_gumbel_root_many(
        None,
        boards,
        device="cpu",
        rng=np.random.default_rng(123),
        cfg=GumbelConfig(input_extra_features="v1", simulations=8, topk=8, temperature=0.0, add_noise=False),
        evaluator=cast(Any, _ZeroEvaluator()),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
        per_game_simulations=[1, 8],
        return_diagnostics=True,
    )

    _probs, _actions, _values, _masks, diagnostics = result
    assert diagnostics[0] is not None
    assert diagnostics[1] is not None
    assert diagnostics[0]["candidate_count"] == 1.0
    assert diagnostics[1]["candidate_count"] == 4.0


class _FakeRootTbProbe:
    max_pieces = 32

    def __init__(self) -> None:
        self.root_calls = 0
        self.leaf_calls = 0
        self.hits = 0

    def apply(self, leaf_cboards, wdl, indices=None, solved_out=None):
        if indices is not None:
            self.leaf_calls += 1
            return 0
        self.root_calls += 1
        wdl[:] = np.array([12.0, -12.0, -12.0], dtype=np.float32)
        self.hits += len(leaf_cboards)
        if solved_out is not None:
            solved_out[:] = 1
        return len(leaf_cboards)


@pytest.mark.skipif(run_gumbel_root_many_c is None, reason="C tree extension not available")
def test_gumbel_c_applies_tb_root_override_with_precomputed_logits():
    board = chess.Board("7k/6Q1/4K3/8/8/8/8/8 b - - 0 1")
    assert len(list(board.legal_moves)) == 1

    probe = _FakeRootTbProbe()
    pre_pol = np.zeros((1, POLICY_SIZE), dtype=np.float32)
    pre_wdl = np.zeros((1, 3), dtype=np.float32)

    run_gumbel_root_many_c = _require_run_gumbel_root_many_c()
    _, _, values, _, _, _ = run_gumbel_root_many_c(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=GumbelConfig(input_extra_features="v1", simulations=4, temperature=1.0, add_noise=False),
        evaluator=cast(Any, object()),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
        tb_probe=probe,
    )

    assert probe.root_calls == 1
    assert probe.leaf_calls == 0
    assert values[0] > 0.999


@pytest.mark.skipif(run_gumbel_root_many_c is None, reason="C tree extension not available")
def test_gumbel_c_can_skip_cached_tb_root_override():
    board = chess.Board("7k/6Q1/4K3/8/8/8/8/8 b - - 0 1")
    probe = _FakeRootTbProbe()
    pre_pol = np.zeros((1, POLICY_SIZE), dtype=np.float32)
    pre_wdl = np.zeros((1, 3), dtype=np.float32)

    run_gumbel_root_many_c = _require_run_gumbel_root_many_c()
    _, _, values, _, _, _ = run_gumbel_root_many_c(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=GumbelConfig(input_extra_features="v1", simulations=4, temperature=1.0, add_noise=False),
        evaluator=cast(Any, object()),
        pre_pol_logits=pre_pol,
        pre_wdl_logits=pre_wdl,
        tb_probe=probe,
        pre_wdl_logits_tb_probed=True,
    )

    assert probe.root_calls == 0
    assert probe.leaf_calls == 0
    assert values[0] == 0.0


@pytest.mark.skipif(run_gumbel_root_many_c is None, reason="C tree extension not available")
def test_gumbel_c_accepts_compact_precomputed_logits():
    board = chess.Board()
    full_pol = np.zeros((1, POLICY_SIZE), dtype=np.float32)
    compact_pol = policy_batch_to_encoding(full_pol, policy_encoding=POLICY_ENCODING_LC0_1858)
    pre_wdl = np.zeros((1, 3), dtype=np.float32)

    run_gumbel_root_many_c = _require_run_gumbel_root_many_c()
    probs, actions, values, masks = run_gumbel_root_many_c(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(123),
        cfg=GumbelConfig(input_extra_features="v1",
            simulations=1,
            topk=8,
            temperature=0.0,
            add_noise=False,
            policy_encoding=POLICY_ENCODING_LC0_1858,
        ),
        evaluator=cast(Any, _ZeroEvaluator()),
        pre_pol_logits=compact_pol,
        pre_wdl_logits=pre_wdl,
    )[:4]

    assert len(actions) == 1
    assert len(values) == 1
    assert probs[0].shape == (POLICY_SIZE,)
    assert np.array_equal(masks[0], legal_move_mask(board))
    assert np.count_nonzero(probs[0][~masks[0]]) == 0


def test_gumbel_python_widens_compact_evaluator_logits():
    board = chess.Board()

    probs, actions, values, masks = run_gumbel_root_many(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(123),
        cfg=GumbelConfig(input_extra_features="v1",
            simulations=1,
            topk=8,
            temperature=0.0,
            add_noise=False,
            policy_encoding=POLICY_ENCODING_LC0_1858,
        ),
        evaluator=cast(Any, _CompactZeroEvaluator()),
    )

    assert len(actions) == 1
    assert len(values) == 1
    assert probs[0].shape == (POLICY_SIZE,)
    assert np.array_equal(masks[0], legal_move_mask(board))


@pytest.mark.skipif(run_gumbel_root_many_c is None, reason="C tree extension not available")
def test_gumbel_c_widens_compact_evaluator_logits():
    board = chess.Board()

    run_gumbel_root_many_c = _require_run_gumbel_root_many_c()
    probs, actions, values, masks = run_gumbel_root_many_c(
        None,
        [board],
        device="cpu",
        rng=np.random.default_rng(123),
        cfg=GumbelConfig(input_extra_features="v1",
            simulations=1,
            topk=8,
            temperature=0.0,
            add_noise=False,
            policy_encoding=POLICY_ENCODING_LC0_1858,
        ),
        evaluator=cast(Any, _CompactZeroEvaluator()),
    )[:4]

    assert len(actions) == 1
    assert len(values) == 1
    assert probs[0].shape == (POLICY_SIZE,)
    assert np.array_equal(masks[0], legal_move_mask(board))


def test_completed_q_uses_parent_perspective_for_visited_child():
    root = Node(chess.Board(), parent=None, prior=1.0)
    child_board = root.board.copy(stack=False)
    child_board.push(next(iter(child_board.legal_moves)))
    child = Node(child_board, parent=root, prior=0.5)
    child.N = 4
    child.W = 2.0  # child.Q = +0.5 from child's side-to-move perspective
    root.children[0] = child

    q = _completed_q(root_q=0.1, root=root, action=0)
    assert q == -0.5


def test_puct_select_child_uses_parent_perspective_for_visited_children():
    root = Node(chess.Board(), parent=None, prior=1.0)
    root.N = 10
    root.W = 0.0
    root.expanded = True

    bad_for_root = Node(None, parent=root, prior=0.5, action_idx=0)
    bad_for_root.N = 5
    bad_for_root.W = 4.0  # child.Q = +0.8 => root-perspective Q = -0.8

    good_for_root = Node(None, parent=root, prior=0.5, action_idx=1)
    good_for_root.N = 5
    good_for_root.W = -4.0  # child.Q = -0.8 => root-perspective Q = +0.8

    root.children = {0: bad_for_root, 1: good_for_root}

    action, chosen = _select_child(root, c_puct=0.0, fpu_reduction=1.2)
    assert action == 1
    assert chosen is good_for_root


def test_gumbel_root_many_draw_terminal_root_returns_zero_policy():
    model = _tiny_model()

    bare_kings = chess.Board("8/8/8/8/8/8/4k3/4K3 w - - 0 1")
    assert bare_kings.is_game_over()
    assert bare_kings.result(claim_draw=True) == "1/2-1/2"

    probs, actions, values, masks = run_gumbel_root_many(
        model,
        [bare_kings],
        device="cpu",
        rng=np.random.default_rng(0),
        cfg=GumbelConfig(input_extra_features="v1", simulations=8, temperature=0.0, add_noise=False),
    )

    assert actions == [0]
    assert values == [0.0]
    assert float(probs[0].sum()) == 0.0
    assert not masks[0].any()


@pytest.mark.skipif(run_gumbel_root_many_c is None, reason="gumbel_c extension not available")
def test_gumbel_c_matches_python_on_history_and_terminal_draws():
    model = _tiny_model()

    repeated = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]:
        repeated.push(chess.Move.from_uci(uci))

    opening = chess.Board()
    for san in ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4"]:
        opening.push_san(san)

    bare_kings = chess.Board("8/8/8/8/8/8/4k3/4K3 w - - 0 1")
    boards = [repeated, opening, bare_kings]

    cfg = GumbelConfig(input_extra_features="v1", simulations=16, temperature=0.0, add_noise=False)
    run_gumbel_root_many_c = _require_run_gumbel_root_many_c()
    py = run_gumbel_root_many(
        model,
        boards,
        device="cpu",
        rng=np.random.default_rng(123),
        cfg=cfg,
    )
    c = run_gumbel_root_many_c(
        model,
        boards,
        device="cpu",
        rng=np.random.default_rng(123),
        cfg=cfg,
    )

    _probs_py, actions_py, _values_py, masks_py = py[:4]
    probs_c, actions_c, values_c, masks_c = c[:4]

    # CBoard encodes history planes differently from python-chess, so
    # actions/values can diverge with random untrained models.
    # Check structural agreement only: shapes, masks, game-over handling.
    assert len(probs_c) == len(boards)
    assert len(actions_c) == len(boards)
    assert len(values_c) == len(boards)

    # Game-over board (bare kings) must be detected by both paths
    assert actions_py[2] == 0  # python path: game over
    assert actions_c[2] == 0   # c path: game over

    # Legal masks must agree (encoding doesn't affect legal moves)
    for m_py, m_c in zip(masks_py, masks_c, strict=True):
        assert np.array_equal(m_py, m_c)

    # Probs must be valid distributions
    for p_c in probs_c:
        assert p_c.shape == (4672,)
        s = float(p_c.sum())
        if s > 0:
            assert abs(s - 1.0) < 0.01

    # Self-consistency: gumbel_c produces valid distributions on second run
    c2 = run_gumbel_root_many_c(
        model, boards, device="cpu", rng=np.random.default_rng(123), cfg=cfg,
    )
    for p2 in c2[0]:
        assert p2.shape == (4672,)
        s = float(p2.sum())
        if s > 0:
            assert abs(s - 1.0) < 0.01


@pytest.mark.skipif(run_gumbel_root_many_c is None, reason="gumbel_c extension not available")
def test_gumbel_c_pipeline_path():
    """Test the pipelined simulation path using a mock async evaluator."""
    model = _tiny_model()

    boards = [chess.Board() for _ in range(64)]
    # Add some variety
    for i, san_seq in enumerate([
        ["e4"], ["d4", "d5"], ["Nf3", "d5", "c4"],
        ["e4", "e5", "Nf3", "Nc6"],
    ]):
        b = chess.Board()
        for san in san_seq:
            b.push_san(san)
        boards[i] = b

    cfg = GumbelConfig(input_extra_features="v1", simulations=16, temperature=1.0, add_noise=True)
    run_gumbel_root_many_c = _require_run_gumbel_root_many_c()

    # Sequential path (sync evaluator)
    _probs_seq, _act_seq, _val_seq, masks_seq = run_gumbel_root_many_c(
        model, boards, device="cpu", rng=np.random.default_rng(42), cfg=cfg,
    )[:4]

    # Mock async evaluator that wraps sync eval but exposes evaluate_encoded_async
    from chess_anti_engine.inference import LocalModelEvaluator

    class MockAsyncEvaluator:
        def __init__(self, m):
            self._inner = LocalModelEvaluator(m, device="cpu")

        def evaluate_encoded(self, x, relations=None):
            return self._inner.evaluate_encoded(x, relations)

        def evaluate_encoded_async(self, x, relations=None):
            pol, wdl = self._inner.evaluate_encoded(x, relations)
            import torch
            return torch.from_numpy(pol), torch.from_numpy(wdl), None

    async_ev = MockAsyncEvaluator(model)

    # Pipeline path (async evaluator + >=64 boards triggers pipeline)
    probs_pipe, act_pipe, _val_pipe, masks_pipe = run_gumbel_root_many_c(
        model, boards, device="cpu", rng=np.random.default_rng(42),
        cfg=cfg, evaluator=async_ev,
    )[:4]

    # Both must produce valid results
    assert len(probs_pipe) == 64
    assert len(act_pipe) == 64

    for i in range(32):
        assert probs_pipe[i].shape == (4672,)
        s = float(probs_pipe[i].sum())
        if s > 0:
            assert abs(s - 1.0) < 0.01, f"game {i}: probs sum = {s}"
        assert probs_pipe[i][act_pipe[i]] > 0 or s == 0, f"game {i}: action not in probs"

    # Masks must agree (legal moves don't depend on path)
    for i in range(32):
        assert np.array_equal(masks_seq[i], masks_pipe[i]), f"game {i}: mask mismatch"


@pytest.mark.skipif(run_gumbel_root_many_c is None, reason="gumbel_c extension not available")
def test_gumbel_c_pipeline_routes_bf16_bit_inputs() -> None:
    class RecordingAsyncEvaluator:
        supports_input_bf16_bits = True

        def __init__(self) -> None:
            self.dtypes: list[np.dtype] = []

        def evaluate_encoded(self, x, relations=None):
            del relations
            self.dtypes.append(x.dtype)
            n = int(x.shape[0])
            return np.zeros((n, POLICY_SIZE), dtype=np.float32), np.zeros((n, 3), dtype=np.float32)

        def evaluate_encoded_async(self, x, relations=None):
            policy, wdl = self.evaluate_encoded(x, relations)
            return torch.from_numpy(policy), torch.from_numpy(wdl), None

    evaluator = RecordingAsyncEvaluator()
    run_gumbel_root_many_c = _require_run_gumbel_root_many_c()
    run_gumbel_root_many_c(
        None,
        [chess.Board() for _ in range(64)],
        device="cpu",
        rng=np.random.default_rng(42),
        cfg=GumbelConfig(
            input_extra_features="v1", simulations=4, topk=2, add_noise=False,
        ),
        evaluator=evaluator,
    )

    assert len(evaluator.dtypes) > 1  # root plus at least one pipelined leaf batch
    assert set(evaluator.dtypes) == {np.dtype(np.uint16)}
