import numpy as np
import torch

from chess_anti_engine.moves import POLICY_ENCODING_LC0_1858, POLICY_SIZE
from chess_anti_engine.selfplay import match as match_mod
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


def test_pick_moves_uses_model_input_history_encoding(monkeypatch):
    seen = {}

    def fake_gumbel(_model, _boards, *, cfg, **_kwargs):
        seen["input_history_encoding"] = cfg.input_history_encoding
        seen["policy_encoding"] = cfg.policy_encoding
        return [np.zeros(POLICY_SIZE, dtype=np.float32)], [0], [0.0], [np.ones(POLICY_SIZE, dtype=bool)]

    model = DummyModel().eval()
    model.input_history_encoding = "lc0_root_legacy_meta"  # pyright: ignore[reportArgumentType]
    setattr(model, "policy_encoding", POLICY_ENCODING_LC0_1858)
    monkeypatch.setattr(match_mod, "_HAS_GUMBEL_C", False)
    monkeypatch.setattr(match_mod, "run_gumbel_root_many", fake_gumbel)

    match_mod._pick_moves_for_boards(
        model,
        [__import__("chess").Board()],
        device="cpu",
        rng=np.random.default_rng(0),
        mcts_type="gumbel",
        mcts_simulations=1,
        temperature=0.0,
        c_puct=2.5,
        gumbel_add_noise=False,
    )

    assert seen["input_history_encoding"] == "lc0_root_legacy_meta"
    assert seen["policy_encoding"] == POLICY_ENCODING_LC0_1858


def test_pick_moves_uses_model_policy_encoding_for_puct(monkeypatch):
    seen = {}

    def fake_puct(_model, _boards, *, cfg, **_kwargs):
        seen["policy_encoding"] = cfg.policy_encoding
        return [np.zeros(POLICY_SIZE, dtype=np.float32)], [0], [0.0], [np.ones(POLICY_SIZE, dtype=bool)]

    model = DummyModel().eval()
    setattr(model, "policy_encoding", POLICY_ENCODING_LC0_1858)
    monkeypatch.setattr(match_mod, "_HAS_C_TREE", False)
    monkeypatch.setattr(match_mod, "run_mcts_many", fake_puct)

    match_mod._pick_moves_for_boards(
        model,
        [__import__("chess").Board()],
        device="cpu",
        rng=np.random.default_rng(0),
        mcts_type="puct",
        mcts_simulations=1,
        temperature=0.0,
        c_puct=2.5,
        gumbel_add_noise=False,
    )

    assert seen["policy_encoding"] == POLICY_ENCODING_LC0_1858
