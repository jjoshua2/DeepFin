from __future__ import annotations

import numpy as np

from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.opening import OpeningConfig
from tests.selfplay_helpers import FakeStockfish, UniformPolicyValueModel


def test_sf_wdl_target_is_flipped_to_network_turn_pov_for_both_colors():
    """SF eval from opponent turn should be flipped when attached to network-turn sample."""
    model = UniformPolicyValueModel().eval()
    rng = np.random.default_rng(0)

    samples, _stats = play_batch(
        model, device="cpu", rng=rng,
        stockfish=FakeStockfish([1.0, 0.0, 0.0]),
        games=2,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=4),
    )

    sf_wdls = [s.sf_wdl for s in samples if s.sf_wdl is not None]
    assert sf_wdls, "Expected at least one sample with sf_wdl"

    # SF reports [1,0,0] for side-to-move at t+1 (opponent turn).
    # Attached target on sample t (network turn) must be flipped => [0,0,1].
    expected = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    assert any(np.allclose(wdl, expected) for wdl in sf_wdls)
