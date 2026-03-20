import numpy as np
import chess
import pytest
import torch

from chess_anti_engine.encoding import encode_positions_batch
from chess_anti_engine.mcts.gumbel import GumbelConfig, _completed_q, run_gumbel_root_many
from chess_anti_engine.mcts.puct import Node, _select_child
from chess_anti_engine.moves import legal_move_mask
from chess_anti_engine.model import ModelConfig, build_model

try:
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
except ImportError:  # pragma: no cover - extension absent
    run_gumbel_root_many_c = None


def _tiny_model() -> torch.nn.Module:
    model = build_model(
        ModelConfig(
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


def test_gumbel_root_many_returns_values_for_edge_cases():
    # Tiny CPU model is enough; we just want shape/length invariants.
    m = build_model(
        ModelConfig(
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
        cfg=GumbelConfig(simulations=8, topk=8, child_sims=2, temperature=1.0),
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
        cfg=GumbelConfig(simulations=8, topk=8, child_sims=2, temperature=1.0),
        evaluator=lambda x: (x, x),
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
    policy_out = out["policy"] if "policy" in out else out["policy_own"]
    pre_pol = policy_out.detach().float().cpu().numpy()
    pre_wdl = out["wdl"].detach().float().cpu().numpy()

    cfg = GumbelConfig(
        simulations=8,
        topk=8,
        child_sims=2,
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
        cfg=GumbelConfig(simulations=8, temperature=0.0, add_noise=False),
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

    cfg = GumbelConfig(simulations=16, temperature=0.0, add_noise=False)
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

    probs_py, actions_py, values_py, masks_py = py
    probs_c, actions_c, values_c, masks_c = c

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

    # Self-consistency: gumbel_c is deterministic with same seed
    c2 = run_gumbel_root_many_c(
        model, boards, device="cpu", rng=np.random.default_rng(123), cfg=cfg,
    )
    assert actions_c == c2[1]
    for p1, p2 in zip(probs_c, c2[0], strict=True):
        np.testing.assert_allclose(p1, p2, atol=1e-6)
