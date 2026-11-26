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
        games=16,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1),
        opening=OpeningConfig(random_start_plies=0),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=6),
    )

    # SF reports [1,0,0] for side-to-move at t+1 (opponent turn).
    # Attached target on sample t (network turn) must be flipped => [0,0,1].
    expected = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # Games start from the initial position with no random plies, so an even
    # ply is a white network move and an odd ply a black one. EVERY labelled row
    # of EACH colour is checked: `any(...)` over the pooled rows passed with one
    # colour unflipped, or with one colour absent from the batch altogether.
    by_colour: dict[str, list[np.ndarray]] = {"white": [], "black": []}
    for s in samples:
        if s.sf_wdl is not None:
            assert s.ply_index is not None
            by_colour["white" if int(s.ply_index) % 2 == 0 else "black"].append(np.asarray(s.sf_wdl))
    for colour, wdls in by_colour.items():
        assert wdls, f"no SF-labelled network row with {colour} to move; the fixture no longer covers both colours"
        for wdl in wdls:
            np.testing.assert_allclose(wdl, expected, err_msg=f"{colour}-to-move row is not in the mover's POV")
