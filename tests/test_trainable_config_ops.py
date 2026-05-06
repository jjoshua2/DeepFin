from __future__ import annotations

from chess_anti_engine.tune.trainable_config_ops import _play_batch_kwargs
from chess_anti_engine.tune.trial_config import TrialConfig


def test_play_batch_kwargs_preserves_syzygy_adjudication_knobs() -> None:
    tc = TrialConfig.from_dict(
        {
            "syzygy_path": "/tmp/tb",
            "syzygy_rescore_policy": True,
            "syzygy_adjudicate": True,
            "syzygy_adjudicate_fraction": 0.5,
            "syzygy_in_search": True,
        }
    )

    game = _play_batch_kwargs(tc)["game"]

    assert game.syzygy_path == "/tmp/tb"
    assert game.syzygy_rescore_policy is True
    assert game.syzygy_adjudicate is True
    assert game.syzygy_adjudicate_fraction == 0.5
    assert game.syzygy_in_search is True
