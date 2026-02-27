import numpy as np
import chess

from chess_anti_engine.mcts.gumbel import GumbelConfig, run_gumbel_root_many
from chess_anti_engine.model import ModelConfig, build_model


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
    probs_list, actions, values = run_gumbel_root_many(
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
