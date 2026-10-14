"""Seed metadata uses the same BOM normalization as the accepted seed body."""
from pathlib import Path

import pytest

from chess_anti_engine.selfplay.opening import _load_fen_backed_bodies, _load_fen_list
from tests.test_worker_model_update import _dole_session_with_list

FEN = "k7/7p/1K6/7P/8/8/2P1PP2/2R5 b - - 0 1"


@pytest.mark.parametrize("bom", [False, True])
@pytest.mark.parametrize("first_comment", [False, True])
@pytest.mark.parametrize("weight", [0, 3])
def test_worker_dole_preserves_seed_weights_and_backed_tag(tmp_path: Path, bom, first_comment, weight):
    session = _dole_session_with_list(tmp_path)
    path = session.opening_fen_list_path
    assert path is not None
    prefix = "# exported seeds\n" if first_comment else ""
    Path(path).write_text(
        prefix + FEN + f" # weight={weight} dropped=1\n",
        encoding="utf-8-sig" if bom else "utf-8",
    )
    session._maybe_ingest_dole_flag({
        "dole_fen_seeds": True, "training_iteration": 1,
        "recommended_worker": {"opening_fen_dole_per_iter": 1,
                               "opening_fen_sf_refute_frac": 1.0,
                               "opening_fen_sf_refute_plies": 3},
    })
    assert _load_fen_list(path) == (FEN,)
    assert session._pending_fen_dole == [FEN] * weight
    assert session._pending_sf_refute == ([FEN] if weight else [])
    assert FEN in _load_fen_backed_bodies(path)
