"""``scripts/fit_value_recal.py``: the (T, b) the arena's ``--<seat>-value-recal`` applies."""
from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest

from scripts.fit_value_recal import _softmax, cross_entropy, fit_temp_drawbias, main, recal_probs
from tests.test_arena_hybrid_value import _save


def test_the_fit_recovers_known_parameters() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 1.5, (3000, 3))
    t_true, b_true = 0.6, -0.55
    ref = recal_probs(logits, t_true, b_true)          # the reference IS the recalibrated net
    fit = fit_temp_drawbias(logits, ref)
    assert fit["T"] == pytest.approx(t_true, rel=2e-3)
    assert fit["b"] == pytest.approx(b_true, abs=2e-3)
    assert cross_entropy(logits, ref, fit["T"], fit["b"]) <= cross_entropy(logits, ref, 1.0, 0.0)


def test_the_identity_is_recovered_when_the_nets_agree() -> None:
    logits = np.random.default_rng(1).normal(0.0, 1.0, (500, 3))
    fit = fit_temp_drawbias(logits, _softmax(logits))
    assert fit["T"] == pytest.approx(1.0, abs=2e-3)
    assert fit["b"] == pytest.approx(0.0, abs=2e-3)


def test_bad_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="need"):
        fit_temp_drawbias(np.zeros((4, 2)), np.zeros((4, 2)))


def test_the_cli_fits_two_checkpoints_on_rows_and_records_them(tmp_path: Path) -> None:
    from chess_anti_engine.encoding.model_inputs import encode_position_for_model
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    net, ref = _save(tmp_path / "net.pt", 0), _save(tmp_path / "ref.pt", 1)
    model = load_model_from_checkpoint(net)
    boards = []
    board = chess.Board()
    for mv in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"):
        board.push_uci(mv)
        boards.append(board.copy())
    rows = tmp_path / "rows.npz"
    np.savez(rows, x=np.stack([encode_position_for_model(model, b) for b in boards]).astype(np.float32))
    out = tmp_path / "fit.json"
    assert main(["--net", str(net), "--reference", str(ref), "--rows", str(rows), "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert rec["rows"] == len(boards)
    assert rec["T"] > 0
    assert rec["cli"] == f"{rec['T']!r},{rec['b']!r}"
    assert rec["cross_entropy_fitted"] <= rec["cross_entropy_identity"]
    assert len(rec["net_sha256"]) == 64
    assert rec["net_sha256"] != rec["reference_sha256"]
    # The net fitted against itself is the identity.
    out2 = tmp_path / "self.json"
    main(["--net", str(net), "--reference", str(net), "--rows", str(rows), "--out", str(out2)])
    rec2 = json.loads(out2.read_text())
    assert rec2["T"] == pytest.approx(1.0, abs=5e-3)
    assert rec2["b"] == pytest.approx(0.0, abs=5e-3)


def test_rows_must_come_from_exactly_one_source(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="exactly one"):
        main(["--net", "a.pt", "--reference", "b.pt", "--out", str(tmp_path / "o.json")])
