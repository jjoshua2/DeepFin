import numpy as np
import torch

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.selfplay.match import play_match_batch


class DummyModel(torch.nn.Module):
    def forward(self, x):
        bs = int(x.shape[0])
        # deterministic, uniform-ish policy after masking; neutral WDL
        return {
            "policy_own": torch.zeros((bs, POLICY_SIZE), device=x.device, dtype=torch.float32),
            "wdl": torch.zeros((bs, 3), device=x.device, dtype=torch.float32),
        }


def test_play_match_batch_smoke():
    rng = np.random.default_rng(0)
    m = DummyModel().eval()

    stats = play_match_batch(
        m,
        m,
        device="cpu",
        rng=rng,
        games=2,
        max_plies=6,
        a_plays_white=[True, False],
        mcts_type="puct",
        mcts_simulations=1,
        temperature=0.0,
    )

    assert stats.games == 2
    assert stats.a_win + stats.a_draw + stats.a_loss == 2
    assert stats.a_as_white == 1
    assert stats.a_as_black == 1
