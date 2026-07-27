"""Bootstrap shards must be written in the model's encoding, not the default.

docs/rl_loop_audit.md M11/M12: `scripts/generate_bootstrap.py` was the one live
producer of legacy/v1 replay rows, and a legacy row entering an
`lc0_root_legacy_meta` window is precisely the case the remap cannot convert
without losing side-to-move.
"""
from __future__ import annotations

import argparse

import numpy as np

from chess_anti_engine.encoding.lc0 import LC0_FULL
from scripts.generate_bootstrap import EncodingSpec, _encoding_spec, play_one_random_game

_SIDE_TO_MOVE_PLANE = LC0_FULL.root_metadata_base + 4  # 108


def _args(
    *, history: str | None = None, extra: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        config="configs/pbt2_small.yaml",
        input_history_encoding=history,
        input_extra_features=extra,
    )


def test_defaults_come_from_the_production_config() -> None:
    spec = _encoding_spec(_args())
    assert spec.input_history_encoding == "lc0_root_legacy_meta"
    assert spec.input_extra_features == "v2_threats"
    assert spec.history_rep_fix is True


def test_explicit_overrides_win() -> None:
    spec = _encoding_spec(_args(history="legacy", extra="v1"))
    assert spec.input_history_encoding == "legacy"
    assert spec.input_extra_features == "v1"


def test_generated_samples_carry_the_encoding_and_a_real_side_to_move() -> None:
    spec = EncodingSpec(
        input_history_encoding="lc0_root_legacy_meta",
        input_extra_features="v2_threats",
        history_rep_fix=True,
    )
    samples = play_one_random_game(7, spec)
    assert samples

    xs = np.stack([np.asarray(s.x) for s in samples])
    assert xs.shape[1] == 175
    assert all(s.input_history_encoding == "lc0_root_legacy_meta" for s in samples)

    # Both colours must appear in the side-to-move plane. Under the old legacy
    # default this plane was absent, and the salvage remap would have set it to
    # 0 for every row.
    flags = xs[:, _SIDE_TO_MOVE_PLANE].reshape(len(samples), -1).max(axis=1)
    assert set(np.unique(flags)) == {0.0, 1.0}


def test_legacy_override_still_produces_v1_planes() -> None:
    spec = EncodingSpec(
        input_history_encoding="legacy",
        input_extra_features="v1",
        history_rep_fix=False,
    )
    samples = play_one_random_game(7, spec)
    assert np.asarray(samples[0].x).shape[0] == 146
