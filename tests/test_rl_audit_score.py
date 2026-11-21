"""scripts/rl_audit_score.py: the RL run 1 plan's offline scorer, dose rule and A2 sampler."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import rl_audit_score as ras

N_MOVES = 1858


def test_clustered_se_matches_the_hand_formula_and_widens_with_correlated_rows() -> None:
    v = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    singles = np.arange(6)
    mean, se = ras.clustered_mean_se(v, singles)
    # one row per cluster: sqrt(G/(G-1) * sum((v - mean)^2)) / n
    assert mean == pytest.approx(0.5)
    assert se == pytest.approx(np.sqrt(6 / 5 * ((v - 0.5) ** 2).sum()) / 6)
    # the same values in two perfectly correlated games: the SE is larger
    _, se_pairs = ras.clustered_mean_se(v, np.array([0, 0, 1, 1, 2, 2]))
    assert se_pairs > se


def _legal(n: int, k: int = 5) -> np.ndarray:
    legal = np.zeros((n, N_MOVES), dtype=bool)
    legal[:, :k] = True
    return legal


def test_score_reads_top1_gap_and_entropy_off_the_priors() -> None:
    n = 8
    legal = _legal(n)
    ruler = np.zeros(n, dtype=np.int64)  # the ruler always picks move 0
    logits = np.full((n, N_MOVES), -50.0)
    logits[:, :5] = 0.0
    logits[: n // 2, 0] = 5.0  # agrees on half the rows
    logits[n // 2:, 1] = 5.0
    game = np.repeat(np.arange(4), 2)
    s = ras.score_one(logits, legal, ruler, game)
    p_top = np.exp(5) / (np.exp(5) + 4)
    assert s["top1"] == pytest.approx(0.5)
    assert s["calibration_gap"] == pytest.approx(p_top - 0.5)
    lp = np.log(np.array([np.exp(5), 1, 1, 1, 1]) / (np.exp(5) + 4))
    assert s["entropy"] == pytest.approx(float(-(np.exp(lp) * lp).sum()))
    flat = np.zeros((n, N_MOVES))
    d = ras.paired_top1(logits, flat, legal, ruler, game)  # flat argmax is move 0 too (first legal)
    assert d["diff"] == pytest.approx(-0.5)


def test_rulers_refuse_an_illegal_move_and_accept_index_or_distribution(tmp_path: Path) -> None:
    legal = _legal(3)
    np.savez(tmp_path / "r.npz", idx=np.array([0, 1, 2]), dist=np.eye(N_MOVES)[[4, 3, 2]].astype(np.float32),
             bad=np.array([0, 9, 1]))
    got = ras._rulers([f"i={tmp_path / 'r.npz'}:idx", f"d={tmp_path / 'r.npz'}:dist"], legal)
    assert got["i"].tolist() == [0, 1, 2]
    assert got["d"].tolist() == [4, 3, 2]
    with pytest.raises(SystemExit, match="illegal move"):
        ras._rulers([f"b={tmp_path / 'r.npz'}:bad"], legal)


def _paired(diffs: dict[tuple[int, int], tuple[float, float]], n: int) -> dict[tuple[int, int], dict[str, float]]:
    out = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d, se = diffs.get((i, j), (0.0, 1.0)) if (i, j) in diffs else (-diffs.get((j, i), (0.0, 1.0))[0],
                                                                              diffs.get((j, i), (0.0, 1.0))[1])
                out[(i, j)] = {"diff": d, "se": se}
    return out


def test_dose_rule_best_in_band_then_earliest_tie_then_nearest() -> None:
    scores = [{"top1": 0.60, "calibration_gap": -0.05}, {"top1": 0.62, "calibration_gap": -0.02},
              {"top1": 0.63, "calibration_gap": 0.05}, {"top1": 0.61, "calibration_gap": 0.00}]
    band = (-0.10, 0.02)
    # 2 is out of band; 1 is the in-band best; 0 is tied with 1 (|0.02| <= SE 0.03); 3 is not (SE 0.005).
    paired = _paired({(1, 0): (0.02, 0.03), (1, 3): (0.01, 0.005), (1, 2): (-0.01, 0.01)}, 4)
    pick, why = ras.choose_dose(scores, paired, band)
    assert pick == 0
    assert "earliest" in why
    paired = _paired({(1, 0): (0.02, 0.005), (1, 3): (0.01, 0.005), (1, 2): (-0.01, 0.01)}, 4)
    assert ras.choose_dose(scores, paired, band)[0] == 1
    out = [{"top1": 0.7, "calibration_gap": 0.10}, {"top1": 0.6, "calibration_gap": 0.04}]
    pick, why = ras.choose_dose(out, _paired({}, 2), band)
    assert pick == 1
    assert "nearest" in why


def _tiny_run_with_snapshots(tmp_path: Path) -> Path:
    from chess_anti_engine.simple_rl.learn import TrainSettings, train_generation
    from tests.simple_rl_helpers import save_tiny_checkpoint
    from tests.test_simple_rl_learn import _write_generation

    run = tmp_path / "cell"
    (run / "gen_0000").mkdir(parents=True)
    save_tiny_checkpoint(run / "gen_0000" / "model.pt")
    _write_generation(run, 1, 48)
    train_generation(run_dir=run, generation=1, init_checkpoint=run / "gen_0000" / "model.pt",
                     prev_trainer_checkpoint=None,
                     settings=TrainSettings(device="cpu", batch_size=8, holdout_frac=0.0, seed=3, lr=1e-3,
                                            snapshot_updates=(2, 4)))
    return run / "gen_0001" / "train"


def _rows_npz(path: Path, n: int = 24) -> Path:
    rng = np.random.default_rng(0)
    legal = _legal(n, 12)
    np.savez(path, x=(rng.standard_normal((n, 175, 8, 8)) * 0.1).astype(np.float16), legal=legal,
             game_id=np.repeat(np.arange(n // 3), 3))
    return path


def test_dose_falls_back_to_the_final_model_when_the_ruler_labels_have_not_landed(tmp_path: Path) -> None:
    train = _tiny_run_with_snapshots(tmp_path)
    rows = _rows_npz(tmp_path / "a1.npz")
    out = tmp_path / "dose.json"
    assert ras.main(["dose", "--train-dir", str(train), "--rows", str(rows),
                     "--ruler", f"bt4s256={tmp_path / 'not_yet.npz'}:move", "--band-lo", "-0.2",
                     "--out", str(out), "--out-model", str(tmp_path / "played.pt")]) == 0
    rec = json.loads(out.read_text())
    assert rec["played"]["name"] == "final"
    assert "have not landed" in rec["why"]
    assert rec["played_model_sha256"] == ras.sha256_file(train / "model.pt")
    assert [c["name"] for c in rec["candidates"]] == ["step_2", "step_4", "final"]


def test_dose_scores_every_snapshot_with_the_forwarded_priors(tmp_path: Path) -> None:
    train = _tiny_run_with_snapshots(tmp_path)
    rows = _rows_npz(tmp_path / "a1.npz")
    rng = np.random.default_rng(1)
    np.savez(tmp_path / "ruler.npz", move=rng.integers(0, 12, 24))
    out = tmp_path / "dose.json"
    assert ras.main(["dose", "--train-dir", str(train), "--rows", str(rows),
                     "--ruler", f"r={tmp_path / 'ruler.npz'}:move", "--band-lo", "-1", "--band-hi", "1",
                     "--out", str(out), "--out-model", str(tmp_path / "played.pt")]) == 0
    rec = json.loads(out.read_text())
    assert len(rec["scores"]) == 3
    # The recorded scores are the snapshot models' own: recompute one through `forward`.
    pri = tmp_path / "pri.npz"
    assert ras.main(["forward", "--rows", str(rows), "--model", f"s2={train / 'snapshots' / 'step_2' / 'model.pt'}",
                     "--out", str(pri)]) == 0
    with np.load(pri) as z:
        s2 = ras.score_one(z["pol_s2"], np.load(rows)["legal"], np.load(tmp_path / "ruler.npz")["move"],
                           np.load(rows)["game_id"])
    assert rec["scores"][0]["top1"] == pytest.approx(s2["top1"])
    assert rec["scores"][0]["calibration_gap"] == pytest.approx(s2["calibration_gap"])
    assert rec["played_model_sha256"] == ras.sha256_file(Path(rec["played"]["model"]))


def test_a2_sample_replays_every_row_and_excludes_training_positions(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import simple_rl_loop
    from tests.test_simple_rl_resume import _setup
    from tests.test_simple_rl_snapshots_arms import _argv, _stage

    openings, model_config = _setup(tmp_path, monkeypatch)
    run = tmp_path / "a2games"
    assert simple_rl_loop.main(_stage(_argv(run, openings, model_config), "collect")) == 0
    collect = run / "gen_0001" / "collect"
    out = tmp_path / "a2.npz"
    assert ras.main(["a2-sample", "--collect-dir", str(collect), "--per-game", "2", "--out", str(out)]) == 0
    summary: dict[str, Any] = json.loads(Path(str(out) + ".json").read_text())
    with np.load(out) as z:
        assert z["x"].shape[0] == summary["rows"] > 0
        per_game = np.unique(z["game_id"], return_counts=True)[1]
        assert per_game.max() <= 2
        import chess

        from chess_anti_engine.moves.encode import legal_move_mask_for_encoding
        for root, hist, fen, legal in zip(z["root_fen"], z["history"], z["fen"], z["legal"], strict=True):
            b = chess.Board(str(root))
            for mv in str(hist).split():
                b.push_uci(mv)
            assert b.fen() == str(fen)
            assert np.array_equal(legal_move_mask_for_encoding(b), legal)
    assert summary["games_truncated_at_a_row_gap"] == 0
    # P2: excluding the collect's own positions leaves nothing.
    out2 = tmp_path / "a2x.npz"
    assert ras.main(["a2-sample", "--collect-dir", str(collect), "--exclude-collect", str(collect),
                     "--out", str(out2)]) == 0
    assert json.loads(Path(str(out2) + ".json").read_text())["rows"] == 0


def test_a2_openings_takes_exactly_the_block(tmp_path: Path) -> None:
    ev = tmp_path / "eval.jsonl"
    ev.write_text("".join(json.dumps({"id": f"e{i}", "book": 2, "line": f"L{i}"}) + "\n" for i in range(20)))
    out = tmp_path / "a2open"
    assert ras.main(["a2-openings", "--eval-jsonl", str(ev), "--lines", "5:12", "--out-dir", str(out)]) == 0
    rows = [json.loads(ln) for ln in (out / "train.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == [f"e{i}" for i in range(5, 12)]
    assert {r["book"] for r in rows} == {1}
    with pytest.raises(SystemExit, match="exists"):
        ras.main(["a2-openings", "--eval-jsonl", str(ev), "--lines", "5:12", "--out-dir", str(out)])
    with pytest.raises(SystemExit, match="does not fit"):
        ras.main(["a2-openings", "--eval-jsonl", str(ev), "--lines", "15:25", "--out-dir", str(tmp_path / "x")])
