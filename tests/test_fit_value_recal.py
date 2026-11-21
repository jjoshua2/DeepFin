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
    assert rec["logits_dtype"] == {"net": "float32", "reference": "float32"}   # CPU: no autocast
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


def test_the_holdout_fraction_is_the_runs_pinned_one(tmp_path: Path) -> None:
    """Review of #1457: a --holdout-frac that disagrees with the run's would fit on trained rows."""
    from scripts.fit_value_recal import pinned_holdout_frac

    collect = tmp_path / "run" / "gen_0001" / "collect"
    collect.mkdir(parents=True)
    (collect / "manifest.json").write_text("{}")
    assert pinned_holdout_frac(collect, 0.05) == (0.05, "flag, unpinned")  # nothing pinned: the flag, unchecked
    with pytest.raises(SystemExit, match="pins no holdout_frac"):
        pinned_holdout_frac(collect, None)
    (tmp_path / "run" / "run_config.json").write_text(json.dumps({"settings": {"holdout_frac": 0.1}}))
    assert pinned_holdout_frac(collect, None) == (0.1, "run_config")
    assert pinned_holdout_frac(collect, 0.1) == (0.1, "run_config")
    with pytest.raises(SystemExit, match=r"disagrees with the run's pinned holdout_frac 0\.1"):
        main(["--net", "a.pt", "--reference", "b.pt", "--collect-dir", str(collect), "--holdout-frac", "0.05",
              "--out", str(tmp_path / "o.json")])


def test_rows_that_overlap_the_match_openings_are_refused(tmp_path: Path) -> None:
    from scripts.fit_value_recal import check_rows_disjoint

    after_e4 = chess.Board()
    after_e4.push_uci("e2e4")
    openings = tmp_path / "eval.fen"
    openings.write_text(after_e4.fen() + "\n")
    clean = tmp_path / "clean.npz"
    np.savez(clean, x=np.zeros((2, 1, 8, 8)), root_fen=np.array([chess.STARTING_FEN] * 2),
             history=np.array(["d2d4", "d2d4 d7d5"]))
    with np.load(clean) as z:
        assert check_rows_disjoint(z, clean, openings)["rows_overlapping"] == 0
    dirty = tmp_path / "dirty.npz"   # the second row's game passed through the opening's position
    np.savez(dirty, x=np.zeros((2, 1, 8, 8)), root_fen=np.array([chess.STARTING_FEN] * 2),
             history=np.array(["d2d4", "e2e4 e7e5"]))
    with pytest.raises(SystemExit, match="1 rows started at or passed through"):
        main(["--net", "a.pt", "--reference", "b.pt", "--rows", str(dirty), "--exclude-openings", str(openings),
              "--out", str(tmp_path / "o.json")])
    bare = tmp_path / "bare.npz"
    np.savez(bare, x=np.zeros((1, 1, 8, 8)))
    with pytest.raises(SystemExit, match="needs root_fen and history"):
        main(["--net", "a.pt", "--reference", "b.pt", "--rows", str(bare), "--exclude-openings", str(openings),
              "--out", str(tmp_path / "o.json")])


def test_a_fen_list_line_with_moves_is_keyed_by_the_position_the_match_starts_from(tmp_path: Path) -> None:
    """Review of #1459: '<root> | <moves>' keys the replayed TERMINAL board, not the root.

    The root here is not the start position, so keying the root would silently accept
    a row whose game passes through the real match start.
    """
    from scripts.fit_value_recal import check_rows_disjoint, opening_keys

    root = chess.Board()
    for u in ("e2e4", "e7e5"):
        root.push_uci(u)
    start = root.copy()
    for u in ("g1f3", "b8c6"):
        start.push_uci(u)
    openings = tmp_path / "eval.fen"
    openings.write_text(f"# provenance comment\n{root.fen()} | g1f3 b8c6\n\n")
    keys = opening_keys(openings)
    assert len(keys) == 1
    assert keys == {start.epd()}
    rows = tmp_path / "rows.npz"   # passes through the match start, never through the root's own line end
    np.savez(rows, x=np.zeros((1, 1, 8, 8)), root_fen=np.array([chess.STARTING_FEN]),
             history=np.array(["g1f3 b8c6 e2e4 e7e5 d2d4"]))
    with np.load(rows) as z, pytest.raises(SystemExit, match="1 rows started at or passed through"):
        check_rows_disjoint(z, rows, openings)
    with pytest.raises(SystemExit, match="no fixed opening set"):
        opening_keys(tmp_path / "book.pgn.zip")


def test_a_pgn_line_is_keyed_where_the_arena_starts_it_after_opening_plies(tmp_path: Path) -> None:
    """Review of #1459 (note): the arena truncates a PGN line at --opening-plies, so the guard keys
    the position after min(len, opening_plies) plies, and refuses a FEN header the arena ignores."""
    from scripts.fit_value_recal import opening_keys

    pgn = tmp_path / "eval.pgn"
    pgn.write_text('[Event "x"]\n\n1. e4 e5 2. Nf3 Nc6 *\n')
    after4, after2 = chess.Board(), chess.Board()
    for u in ("e2e4", "e7e5", "g1f3", "b8c6"):
        after4.push_uci(u)
    for u in ("e2e4", "e7e5"):
        after2.push_uci(u)
    assert opening_keys(pgn) == {after4.epd()}                       # 4 plies < the default 16
    assert opening_keys(pgn, opening_plies=2) == {after2.epd()}      # truncated as the arena would
    fen_game = tmp_path / "fen.pgn"
    fen_game.write_text(f'[Event "x"]\n[SetUp "1"]\n[FEN "{after2.fen()}"]\n\n2. Nf3 *\n')
    with pytest.raises(SystemExit, match="FEN header"):
        opening_keys(fen_game)


def test_the_cli_opening_plies_reaches_the_overlap_guard_and_its_record(tmp_path: Path) -> None:
    """Review of #1466: --opening-plies must not be accepted and then ignored. A row that passes
    through only the 2-ply prefix of a 4-ply line is refused at --opening-plies 2, not at 16."""
    from scripts.fit_value_recal import check_rows_disjoint

    pgn = tmp_path / "eval.pgn"
    pgn.write_text('[Event "x"]\n\n1. e4 e5 2. Nf3 Nc6 *\n')
    rows = tmp_path / "rows.npz"
    np.savez(rows, x=np.zeros((1, 1, 8, 8)), root_fen=np.array([chess.STARTING_FEN]),
             history=np.array(["e2e4 e7e5 d2d4"]))
    base = ["--net", "a.pt", "--reference", "b.pt", "--rows", str(rows), "--exclude-openings", str(pgn),
            "--out", str(tmp_path / "o.json")]
    with pytest.raises(SystemExit, match="1 rows started at or passed through"):
        main([*base, "--opening-plies", "2"])
    with pytest.raises(FileNotFoundError, match=r"no trainer\.pt at a\.pt"):   # past the guard: fails on the absent nets instead
        main(base)
    with np.load(rows) as z:
        assert check_rows_disjoint(z, rows, pgn)["opening_plies"] == 16
        assert check_rows_disjoint(z, rows, pgn, 3)["opening_plies"] == 3
