"""Gated lc0-root repetition-plane fix (selfplay.history_rep_fix).

Default off must keep the C encoding byte-identical to before; on must make the
per-history-slot repetition planes bit-match python-chess, including deep
positions where an irreversible move cleared the C hash_stack.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import _lc0_ext, encode_position, rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.moves import move_to_index

PROD_MODES = ("lc0_root", "lc0_root_legacy_meta")

# A 180-ply game whose ply-174 position is a 3-fold repetition older than the
# C hash_stack window after a later pawn move — the reconstruction encoder
# misses it (plane 90), python-chess catches it.
_DEEP_REP_GAME = (
    "h2h3 a7a5 f2f4 a5a4 g2g3 g7g5 b2b4 e7e6 b4b5 g5g4 h3g4 e8e7 d2d3 a8a6 "
    "c1b2 c7c5 c2c4 h7h5 f1h3 d8e8 e1f2 a6a8 b2e5 d7d5 d1a4 h5g4 e5b8 e8d7 "
    "h3g4 h8h6 b1d2 e7f6 h1h3 g8e7 a1e1 d7c7 a4a7 h6h8 b5b6 a8b8 e1d1 c7d7 "
    "e2e3 e7g8 d1c1 h8h3 a7b8 d7e7 d2b3 h3h4 c1d1 h4h3 g4h5 d5d4 f2f3 h3h5 "
    "d1a1 h5h4 f3f2 h4h2 f2e1 f8h6 b8a7 e6e5 b3c5 h2g2 c5d7 c8d7 g1h3 f6g6 "
    "a1b1 e7d8 g3g4 g8e7 f4e5 d8f8 g4g5 e7d5 b1c1 d5c3 c1c2 f8e8 e5e6 g2g5 "
    "c2f2 g5b5 f2f6 g6h7 e6d7 b5g5 a7a3 h6g7 h3g1 g5d5 d7e8q d5d7 e8e4 c3e4 "
    "f6f5 h7g8 g1e2 g7f6 f5f4 e4g3 a3f8 g8h7 f4f2 g3f1 f8d6 d7d8 d6f6 f1h2 "
    "f6g7 h7g7 f2f4 d8a8 e2g3 f7f5 a2a4 a8a7 f4f1 g7f6 f1g1 f5f4 g1f1 f6g7 "
    "g3h1 a7a4 e3f4 a4a7 f4f5 h2f3 f1f3 a7a4 f3h3 a4b4 h3h4 b4c4 d3c4 g7f8 "
    "h4h3 f8f7 h3a3 f7g8 a3a1 g8g7 h1f2 g7f8 c4c5 f8e7 f2d1 e7d8 e1f1 d8e7 "
    "d1b2 e7f8 b2c4 f8e8 c4e5 e8d8 a1a2 d8c8 a2e2 c8d8 e2e4 d8c8 e5g6 c8d8 "
    "g6e7 d8d7 e7d5 d7c8 d5e7 c8d7 e7g8 d7d8 c5c6 d8c8 e4e2 c8b8"
).split()


@pytest.fixture(autouse=True)
def _reset_flag():
    rep_fix.apply(False)
    yield
    rep_fix.apply(False)


def _build(moves: Sequence[str]) -> tuple[CBoard, chess.Board]:
    b = chess.Board()
    cb = CBoard.from_board(b)
    for u in moves:
        m = chess.Move.from_uci(u)
        cb.push_index(move_to_index(m, b))
        b.push(m)
    return cb, b


def test_default_off_unchanged_and_diverges_like_before():
    """Default off: the deep-repetition game still diverges from python-chess on
    the history-slot repetition plane (documents the bug the fix targets)."""
    _lc0_ext.set_history_rep_fix(False)
    cb, b = _build(_DEEP_REP_GAME)
    c = encode_cboard(cb, input_history_encoding="lc0_root_legacy_meta", input_extra_features="v1")
    p = encode_position(b, input_history_encoding="lc0_root_legacy_meta", input_extra_features="v1")
    diff = [pl for pl in range(c.shape[0]) if not np.array_equal(c[pl], p[pl])]
    assert diff == [90], f"expected the known plane-90 divergence, got {diff}"


def test_fix_on_matches_python_on_deep_repetition():
    _lc0_ext.set_history_rep_fix(True)
    cb, b = _build(_DEEP_REP_GAME)
    for mode in PROD_MODES:
        c = encode_cboard(cb, input_history_encoding=mode, input_extra_features="v1")
        p = encode_position(b, input_history_encoding=mode, input_extra_features="v1")
        assert np.array_equal(c, p), f"fix-on diverged for {mode!r}"


def test_fix_on_matches_python_over_random_games():
    import random

    # Before any boards exist: per-slot recording follows the ordering
    # contract (apply before construction/push), like the production paths.
    rep_fix.apply(True)
    rng = random.Random(2024)
    for g in range(30):
        b = chess.Board()
        cb = CBoard.from_board(b)
        for ply in range(rng.randint(40, 160)):
            moves = list(b.legal_moves)
            if not moves:
                break
            m = rng.choice(moves)
            cb.push_index(move_to_index(m, b))
            b.push(m)
            if ply % 5:
                continue
            for mode in PROD_MODES:
                c = encode_cboard(cb, input_history_encoding=mode, input_extra_features="v1")
                p = encode_position(b, input_history_encoding=mode, input_extra_features="v1")
                assert np.array_equal(c, p), f"g{g} ply{ply} {mode}: fix-on diverged"


def test_exp_config_passes_yaml_selfplay_allowlist():
    """configs/exp_repetition_fix.yaml must flatten — history_rep_fix has to be
    in the selfplay: key allowlist or the advertised gated config can't run."""
    from pathlib import Path

    from chess_anti_engine.utils.config_yaml import (
        flatten_run_config_defaults,
        load_yaml_file,
    )

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "exp_repetition_fix.yaml"
    flat = flatten_run_config_defaults(load_yaml_file(str(cfg_path)))
    assert flat["history_rep_fix"] is True


def test_model_config_persists_history_rep_fix():
    """The flag is model identity: it must survive the manifest round trip and
    be applied to the process-global encoder state when the model is built."""
    from chess_anti_engine.model import (
        ModelConfig,
        build_model,
        model_config_from_manifest_dict,
        model_config_to_manifest_dict,
    )

    cfg = ModelConfig(kind="tiny", history_rep_fix=True)
    md = model_config_to_manifest_dict(cfg)
    assert md["history_rep_fix"] is True
    assert model_config_from_manifest_dict(md).history_rep_fix is True
    # Absent key (pre-fix manifest) means off.
    assert model_config_from_manifest_dict({}).history_rep_fix is False

    model = build_model(cfg)
    assert model.history_rep_fix is True
    # build_model must have applied the flag to the C encoders, so a
    # rep-fix-trained checkpoint evaluates on the planes it was trained on.
    cb, b = _build(_DEEP_REP_GAME)
    c = encode_cboard(cb, input_history_encoding="lc0_root_legacy_meta", input_extra_features="v1")
    p = encode_position(b, input_history_encoding="lc0_root_legacy_meta", input_extra_features="v1")
    assert np.array_equal(c, p), "build_model did not apply history_rep_fix"


def test_shard_meta_and_upload_buffer_persist_history_rep_fix():
    """Shard identity: the flag is recorded in ShardMeta and mixed-flag game
    batches must not silently merge into one upload buffer."""
    from chess_anti_engine.replay.shard import ShardMeta
    from chess_anti_engine.selfplay.state import CompletedGameBatch
    from chess_anti_engine.worker_buffer import (
        _buffer_add_completed_game,
        _BufferedUpload,
    )

    assert ShardMeta(history_rep_fix=True).history_rep_fix is True
    # Pre-field shard dicts load with the flag unset (provably off).
    legacy_meta: dict[str, Any] = {"positions": 1}
    assert ShardMeta(**legacy_meta).history_rep_fix is None

    def batch(flag: bool) -> CompletedGameBatch:
        return CompletedGameBatch(
            samples=cast("list[Any]", [object()]),  # only len() is used
            input_history_encoding="lc0_root_legacy_meta",
            history_rep_fix=flag,
            positions=1,
        )

    buf = _BufferedUpload()
    _buffer_add_completed_game(
        buf=buf, game_batch=batch(True), now_s=0.0, model_sha="sha", model_step=1,
    )
    assert buf.history_rep_fix is True
    with pytest.raises(ValueError, match="metadata mismatch"):
        _buffer_add_completed_game(
            buf=buf, game_batch=batch(False), now_s=0.0, model_sha="sha", model_step=1,
        )


def test_server_accumulator_rejects_mixed_history_rep_fix():
    from chess_anti_engine.server.app import _BufferedUploadAccumulator

    acc = _BufferedUploadAccumulator(
        trial_id=None, model_sha256="sha", created_at_unix=0.0, last_update_unix=0.0,
    )
    meta = {"input_history_encoding": "lc0_root_legacy_meta", "history_rep_fix": True}
    acc.add_upload(samples=[], meta=meta, now_unix=0.0)
    assert acc.history_rep_fix is True
    with pytest.raises(ValueError, match="mixed history_rep_fix"):
        acc.add_upload(samples=[], meta={**meta, "history_rep_fix": False}, now_unix=0.0)


def test_from_board_with_history_sets_per_slot_flags():
    """from_board (opening with move history) must also produce fix-on parity,
    exercising the from_board per-slot population path rather than push."""
    rep_fix.apply(True)
    # Build a board with a real repetition, then reconstruct via from_board so
    # the history comes from python's _stack, not from C pushes.
    b = chess.Board()
    for u in "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8".split():
        b.push(chess.Move.from_uci(u))
    cb = CBoard.from_board(b)
    for mode in PROD_MODES:
        c = encode_cboard(cb, input_history_encoding=mode, input_extra_features="v1")
        p = encode_position(b, input_history_encoding=mode, input_extra_features="v1")
        assert np.array_equal(c, p), f"from_board fix-on diverged for {mode!r}"
