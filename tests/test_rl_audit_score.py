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
                     "--ruler", f"bt4s256={tmp_path / 'not_yet.npz'}:move", "--band-lo", "-0.2", "--no-mask",
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
    spec = f"r={tmp_path / 'ruler.npz'}:move"
    val = _validation(tmp_path, spec, passed=True, rows=rows)
    assert ras.main(["dose", "--train-dir", str(train), "--rows", str(rows), "--ruler", spec, "--band-lo", "-1",
                     "--band-hi", "1", "--no-mask", "--ruler-validation", str(val),
                     "--out", str(out), "--out-model", str(tmp_path / "played.pt")]) == 0
    rec = json.loads(out.read_text())
    assert rec["ruler_validated"] is True
    assert rec["ruler"]["sha256"] == ras.sha256_file(tmp_path / "ruler.npz")
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
    src = json.loads(Path(str(out2) + ".json").read_text())["excluded_sources"]
    assert src[0]["manifest_sha256"] == ras.sha256_file(collect / "manifest.json")
    # An audit-rows file (root_fen + history) excludes the positions it holds, by key.
    out3 = tmp_path / "a2r.npz"
    assert ras.main(["a2-sample", "--collect-dir", str(collect), "--per-game", "2", "--exclude-rows", str(out),
                     "--out", str(out3)]) == 0
    s3 = json.loads(Path(str(out3) + ".json").read_text())
    assert s3["rows_dropped_as_training_positions"] >= summary["rows"]
    assert s3["excluded_sources"][0]["sha256"] == ras.sha256_file(out)
    with np.load(out3) as z3, np.load(out) as z1:
        assert not set(map(str, z3["position_key"])) & set(map(str, z1["position_key"]))


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


def _validation(tmp_path: Path, spec: str, *, passed: bool, rows: Path | None = None,
                mask: str | None = None, tag: str = "") -> Path:
    """A validate record for ``spec`` (its exact file sha), passing or not, run on ``rows`` / ``mask``."""
    rec: dict[str, Any] = {"ruler": ras._ruler_record(spec), "passed": passed,
                           "conditions": {"3_positive_control": {"pass": passed}}}
    if rows is not None:
        rec["input"] = ras.load_rows(rows, mask)["_record"]
    out = tmp_path / f"validation_{int(passed)}{tag}.json"
    out.write_text(json.dumps(rec))
    return out


def test_dose_plays_the_final_model_unless_this_exact_ruler_validated(tmp_path: Path) -> None:
    """Review of #1448 (1): an existing but unvalidated ruler never selects a dose."""
    train = _tiny_run_with_snapshots(tmp_path)
    rows = _rows_npz(tmp_path / "a1.npz")
    np.savez(tmp_path / "ruler.npz", move=np.zeros(24, dtype=np.int64))
    np.savez(tmp_path / "other.npz", move=np.ones(24, dtype=np.int64))
    spec = f"r={tmp_path / 'ruler.npz'}:move"
    base = ["dose", "--train-dir", str(train), "--rows", str(rows), "--ruler", spec, "--band-lo", "-1",
            "--band-hi", "1", "--no-mask", "--out-model", str(tmp_path / "played.pt")]
    cases = {"none": [], "failed": ["--ruler-validation", str(_validation(tmp_path, spec, passed=False))],
             "other_file": ["--ruler-validation",
                            str(_validation(tmp_path, f"r={tmp_path / 'other.npz'}:move", passed=True))]}
    for name, extra in cases.items():
        out = tmp_path / f"dose_{name}.json"
        assert ras.main([*base, "--out", str(out), *extra]) == 0
        rec = json.loads(out.read_text())
        assert rec["played"]["name"] == "final", name
        assert rec["ruler_validated"] is False
        assert "ruler not validated" in rec["why"]
        assert "scores" not in rec  # nothing was selected by the ruler
    assert "another ruler file" in json.loads((tmp_path / "dose_other_file.json").read_text())["why"]


def test_dose_refuses_rows_without_a_mask_decision(tmp_path: Path) -> None:
    train = _tiny_run_with_snapshots(tmp_path)
    rows = _rows_npz(tmp_path / "a1.npz")
    with pytest.raises(SystemExit, match="--mask"):
        ras.main(["dose", "--train-dir", str(train), "--rows", str(rows), "--ruler", f"r={tmp_path / 'x.npz'}:m",
                  "--band-lo", "-1", "--out", str(tmp_path / "d.json"), "--out-model", str(tmp_path / "p.pt")])


def _correlated(n_games: int = 12, per: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows whose agreement is constant within a game (maximal within-game correlation)."""
    n = n_games * per
    legal = _legal(n)
    game = np.repeat(np.arange(n_games), per)
    ruler = np.zeros(n, dtype=np.int64)
    logits = np.zeros((n, N_MOVES))
    logits[:, 0] = np.where(game % 2 == 0, 5.0, -5.0)  # agrees on every row of even games
    return logits, legal, ruler, game


def test_score_one_and_paired_top1_cluster_by_game_at_the_call_site() -> None:
    """Review of #1448 (3): the SE used is the game-clustered one, not the row one."""
    logits, legal, ruler, game = _correlated()
    agree = (np.argmax(np.where(legal, logits, -np.inf), axis=1) == ruler).astype(float)
    s = ras.score_one(logits, legal, ruler, game)
    clustered = ras.clustered_mean_se(agree, game)[1]
    by_row = ras.clustered_mean_se(agree, np.arange(agree.size))[1]
    assert s["top1_se"] == pytest.approx(clustered)
    assert clustered > 1.5 * by_row
    flat = np.zeros_like(logits)
    flat[:, 1] = 1.0  # never agrees
    d = ras.paired_top1(logits, flat, legal, ruler, game)
    assert d["se"] == pytest.approx(clustered)
    assert d["se"] > 1.5 * by_row


def _scoring_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    n = 24
    rows = _rows_npz(tmp_path / "rows.npz", n)
    leak = np.zeros(n, dtype=bool)
    leak[:6] = True
    np.savez(tmp_path / "leak.npz", leak=leak)
    ruler = np.zeros(n, dtype=np.int64)
    np.savez(tmp_path / "ruler.npz", move=ruler)
    pol_a = np.zeros((n, N_MOVES))
    pol_a[:6, 0] = 5.0  # agrees only on the leaked rows
    pol_a[6:, 1] = 5.0
    wdl = np.tile(np.array([[1.0, 0.5, -1.0]]), (n, 1))
    np.savez(tmp_path / "pri.npz", pol_a=pol_a, wdl_a=wdl, pol_b=np.zeros((n, N_MOVES)), wdl_b=np.zeros((n, 3)),
             rows_sha256=np.array(ras.sha256_file(rows)))
    return rows, tmp_path / "leak.npz", tmp_path / "ruler.npz", tmp_path / "pri.npz"


def test_score_records_the_mask_and_dropping_leaked_rows_changes_the_result(tmp_path: Path) -> None:
    """Review of #1448 (2): rows sha, mask spec / sha / polarity and the kept count are recorded."""
    rows, leak, ruler, pri = _scoring_inputs(tmp_path)
    common = ["score", "--rows", str(rows), "--priors", str(pri), "--ruler", f"r={ruler}:move", "--nets", "a"]
    assert ras.main([*common, "--out", str(tmp_path / "all.json")]) == 0
    assert ras.main([*common, "--mask", f"{leak}:leak", "--mask-invert", "--out", str(tmp_path / "clean.json")]) == 0
    full, clean = (json.loads((tmp_path / f).read_text()) for f in ("all.json", "clean.json"))
    assert full["input"]["mask"] is None
    assert full["input"]["rows_kept"] == 24
    assert clean["input"]["rows_kept"] == 18
    assert clean["input"]["rows_dropped"] == 6
    assert clean["input"]["mask"]["invert"] is True
    assert clean["input"]["mask"]["sha256"] == ras.sha256_file(leak)
    assert clean["input"]["rows_sha256"] == ras.sha256_file(rows)
    assert full["nets"]["a"]["r"]["top1"] == pytest.approx(0.25)
    assert clean["nets"]["a"]["r"]["top1"] == pytest.approx(0.0)  # the leaked rows were dropped, not kept
    assert clean["rulers_validated"] is False


def test_priors_forwarded_on_other_rows_are_refused(tmp_path: Path) -> None:
    _, _, ruler, pri = _scoring_inputs(tmp_path)
    other = _rows_npz(tmp_path / "other.npz", 24)
    with np.load(other) as z:
        arrs: dict[str, Any] = {k: z[k] for k in z.files}
    arrs["game_id"] = np.arange(24)  # same N, different rows
    np.savez(other, x=arrs["x"], legal=arrs["legal"], game_id=arrs["game_id"])
    with pytest.raises(SystemExit, match="re-run forward"):
        ras.main(["score", "--rows", str(other), "--priors", str(pri), "--ruler", f"r={ruler}:move",
                  "--out", str(tmp_path / "x.json")])


def test_value_shape_columns_and_regret_with_coverage(tmp_path: Path) -> None:
    rows, _, _, pri = _scoring_inputs(tmp_path)
    moves = np.full((24, 2), -1, dtype=np.int64)
    moves[:, 0] = 1
    moves[:12, 1] = 0
    cp = np.zeros((24, 2))
    cp[:, 0] = 30.0
    np.savez(tmp_path / "regret.npz", moves=moves, regret_cp=cp)
    out = tmp_path / "v.json"
    assert ras.main(["score", "--rows", str(rows), "--priors", str(pri), "--reference", "b",
                     "--regret", f"sf={tmp_path / 'regret.npz'}", "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    p = np.exp([1.0, 0.5, -1.0]) / np.exp([1.0, 0.5, -1.0]).sum()
    va = rec["value"]["a"]
    assert va["p_draw"] == pytest.approx(p[1])
    assert va["abs_wl"] == pytest.approx(abs(p[0] - p[2]))
    assert va["entropy"] == pytest.approx(float(-(p * np.log(p)).sum()))
    assert va["vs_reference"]["abs_wl"]["diff"] == pytest.approx(abs(p[0] - p[2]))  # b is uniform: |W-L| 0
    assert sum(b["n"] for b in va["p_draw_by_ref_bucket"].values()) == 24
    ra = rec["nets"]["a"]["regret:sf"]
    assert ra["covered"] == 24  # net a's move (0 on rows 0-5, 1 elsewhere) is in the table on every row
    assert ra["regret_cp"] == pytest.approx((6 * 0.0 + 18 * 30.0) / 24)
    assert "never a dose selector" in ra["caveat"]
    rb = rec["nets"]["b"]["regret:sf"]  # b's argmax is move 0: covered on rows 0-11 only
    assert rb["covered"] == 12


def test_validate_implements_the_three_conditions() -> None:
    n_games, per = 40, 3
    n = n_games * per
    legal = _legal(n)
    game = np.repeat(np.arange(n_games), per)
    rng = np.random.default_rng(5)
    ruler = np.zeros(n, dtype=np.int64)

    def net(agree_frac: float) -> np.ndarray:
        lg = np.zeros((n, N_MOVES))
        hit = rng.random(n) < agree_frac
        lg[np.arange(n), np.where(hit, 0, 1)] = 5.0
        return lg

    pols = {"base": net(0.7), "p1": net(0.5), "f12": net(0.5), "q50": net(0.2)}
    search_good = np.zeros(n, dtype=np.int64)  # always the ruler's move
    res = ras.validate_ruler(ruler, pols, legal, game, base="base", others=["p1", "f12", "q50"],
                             search_move=search_good, independent=True)
    assert res["passed"] is True
    bad_search = np.argmax(np.where(legal, pols["base"], -np.inf), axis=1)  # = the raw argmax: no gain
    res = ras.validate_ruler(ruler, pols, legal, game, base="base", others=["p1", "f12", "q50"],
                             search_move=bad_search, independent=True)
    assert res["passed"] is False
    assert res["conditions"]["3_positive_control"]["pass"] is False
    flipped = {**pols, "p1": net(0.95)}  # a trained net outranks the base
    res = ras.validate_ruler(ruler, flipped, legal, game, base="base", others=["p1", "f12", "q50"],
                             search_move=search_good, independent=True)
    assert res["conditions"]["2_negative_control"]["pass"] is False
    res = ras.validate_ruler(ruler, pols, legal, game, base="base", others=["p1", "f12", "q50"],
                             search_move=search_good, independent=False)
    assert res["conditions"]["1_independent"]["pass"] is False


def test_a2_position_key_is_history_free_so_a_transposition_is_excluded(tmp_path: Path) -> None:
    """Review of #1448 (5): the exclusion key must map transposed move orders together."""
    import chess

    a, b = chess.Board(), chess.Board()
    for mv in ("e2e4", "e7e5", "g1f3", "b8c6"):
        a.push_uci(mv)
    for mv in ("g1f3", "b8c6", "e2e4", "e7e5"):
        b.push_uci(mv)
    assert a.fen() != b.fen()  # the halfmove clocks differ
    assert ras._position_key(a) == ras._position_key(b)
    np.savez(tmp_path / "a1.npz", root_fen=np.asarray([chess.STARTING_FEN]),
             history=np.asarray(["e2e4 e7e5 g1f3 b8c6"]))
    assert ras.rows_position_keys(tmp_path / "a1.npz") == {ras._position_key(b)}


def test_dose_refuses_a_pass_recorded_on_other_rows_or_another_mask(tmp_path: Path) -> None:
    """Review of #1448 (delta): a PASS on another row set or mask must not unlock selection."""
    train = _tiny_run_with_snapshots(tmp_path)
    rows = _rows_npz(tmp_path / "a1.npz")
    other_rows = tmp_path / "other_rows.npz"
    with np.load(rows) as z:
        arrays: dict[str, np.ndarray] = {k: z[k] for k in z.files}
    arrays["game_id"] = np.arange(24)
    np.savez(other_rows, **arrays)  # pyright: ignore[reportArgumentType]
    np.savez(tmp_path / "ruler.npz", move=np.zeros(24, dtype=np.int64))
    leak = np.zeros(24, dtype=bool)
    leak[:3] = True
    np.savez(tmp_path / "leak.npz", leak=leak)
    spec = f"r={tmp_path / 'ruler.npz'}:move"
    mask = f"{tmp_path / 'leak.npz'}:leak"
    base = ["dose", "--train-dir", str(train), "--rows", str(rows), "--ruler", spec, "--band-lo", "-1",
            "--band-hi", "1", "--out-model", str(tmp_path / "played.pt")]
    cases = {
        "other_rows": (["--no-mask"], _validation(tmp_path, spec, passed=True, rows=other_rows, tag="o"),
                       "other rows"),
        "mask_vs_none": (["--mask", mask, "--mask-invert"],
                         _validation(tmp_path, spec, passed=True, rows=rows, tag="n"), "another mask"),
        "polarity": (["--mask", mask], _validation(tmp_path, spec, passed=True, rows=rows, mask=mask, tag="p"),
                     None),
    }
    for name, (flags, val, why) in cases.items():
        out = tmp_path / f"dose_{name}.json"
        assert ras.main([*base, *flags, "--ruler-validation", str(val), "--out", str(out)]) == 0
        rec = json.loads(out.read_text())
        if why is None:     # same rows, same mask and polarity: selection is unlocked
            assert rec["ruler_validated"] is True, name
            continue
        assert rec["ruler_validated"] is False, name
        assert rec["played"]["name"] == "final", name
        assert why in rec["ruler_validation_reason"], (name, rec["ruler_validation_reason"])
    # the same mask with the other polarity is another row set
    out = tmp_path / "dose_inverted.json"
    val = _validation(tmp_path, spec, passed=True, rows=rows, mask=mask, tag="i")
    assert ras.main([*base, "--mask", mask, "--mask-invert", "--ruler-validation", str(val), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["ruler_validated"] is False


def test_the_draw_buckets_split_by_the_reference_nets_wl_not_the_nets_own() -> None:
    n = 24
    game = np.repeat(np.arange(8), 3)
    ref_wl = np.r_[np.full(8, 0.05), np.full(8, 0.2), np.full(8, 0.6)]
    # the net's own |W - L| is large on every row (all in the >0.3 bucket by its own split),
    # its draw mass differs by the REFERENCE's bucket
    p_draw = np.r_[np.full(8, 0.30), np.full(8, 0.20), np.full(8, 0.10)]
    w = 1.0 - p_draw - 0.001
    p = np.stack([w, p_draw, np.full(n, 0.001)], axis=1)
    out = ras.value_shape(np.log(p), game, ref_wl)
    buckets = out["p_draw_by_ref_bucket"]
    assert [buckets[name]["n"] for _, _, name in ras.WL_BUCKETS] == [8, 8, 8]
    assert [buckets[name]["mean"] for _, _, name in ras.WL_BUCKETS] == pytest.approx([0.30, 0.20, 0.10])


def test_the_negative_control_counts_a_pair_within_one_paired_se_as_a_tie(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 12
    legal = _legal(n, 4)
    game = np.repeat(np.arange(4), 3)
    pols = {k: np.zeros((n, N_MOVES)) for k in ("base", "x", "last")}
    ruler = np.zeros(n, dtype=np.int64)

    def run(diff_over_se: float) -> dict[str, Any]:
        def fake(a: np.ndarray, b: np.ndarray, *_args: Any) -> dict[str, float]:
            # base vs x: the tested pair; everything else comfortably ordered
            if a is pols["base"] and b is pols["x"]:
                return {"diff": diff_over_se * 0.01, "se": 0.01}
            return {"diff": 0.5, "se": 0.01}
        monkeypatch.setattr(ras, "paired_top1", fake)
        return ras.validate_ruler(ruler, pols, legal, game, base="base", others=["x", "last"],
                                  search_move=ruler, independent=True)["conditions"]["2_negative_control"]

    assert run(-0.9)["pass"] is True      # x above base by less than one SE: a tie
    assert run(-1.1)["pass"] is False     # by more than one SE: an error
    assert run(+0.5)["pass"] is True


def test_regret_drops_uncovered_rows_rather_than_zero_filling() -> None:
    moves = np.array([[3, -1], [5, -1], [4, -1]])
    cp = np.array([[100.0, 0.0], [40.0, 0.0], [20.0, 0.0]])
    game = np.array([0, 1, 2])
    col = ras.regret_column(np.array([3, 7, 4]), {"moves": moves, "regret_cp": cp}, game)  # row 1 uncovered
    assert col["covered"] == 2
    assert col["regret_cp"] == pytest.approx(60.0)          # (100 + 20) / 2, not / 3
    pair = ras.paired_regret(np.array([3, 5, 7]), np.array([3, 7, 4]), {"moves": moves, "regret_cp": cp}, game)
    assert pair["rows_both_covered"] == 1                   # only row 0 is covered for both
    assert pair["diff_cp"] == pytest.approx(0.0)


def _sf_line(depth: int, rank: int, mv: str, score: str, *, bound: str = "") -> str:
    return f"info depth {depth} seldepth 4 multipv {rank} score {score}{bound} nodes 9 nps 1 pv {mv} e7e5"


def test_the_sf_converter_follows_the_sf_parse_convention(tmp_path: Path) -> None:
    import chess

    rows = tmp_path / "rows.npz"
    root = chess.STARTING_FEN
    np.savez(rows, x=np.zeros((2, 175, 8, 8), dtype=np.float16), legal=np.ones((2, N_MOVES), dtype=bool),
             game_id=np.array([0, 1]), root_fen=np.array([root, root]), history=np.array(["", "e2e4"]))
    lines = [
        _sf_line(10, 1, "e2e4", "cp 40"), _sf_line(10, 2, "d2d4", "cp 35"),       # depth 10 complete
        _sf_line(11, 1, "e2e4", "cp 60", bound=" lowerbound"),                   # a bound: skipped
        _sf_line(11, 1, "g1f3", "cp 30"), _sf_line(11, 2, "e2e4", "cp 28"),      # depth 11 complete (first wins)
        _sf_line(11, 1, "e2e4", "cp 99"),                                        # a later rank-1 line: ignored
        _sf_line(12, 1, "g1f3", "cp 31"),                                        # depth 12 incomplete
    ]
    rec = {"row": 0, "fen": chess.Board().fen(), "multipv_a": 2, "a": {"lines": lines},
           "b": {"searchmoves": ["h2h4"], "lines": [_sf_line(9, 1, "h2h4", "mate -3")]}}
    sfj = tmp_path / "sf.jsonl"
    sfj.write_text(json.dumps(rec) + "\n")
    out = tmp_path / "regret.npz"
    assert ras.main(["sf-regret-table", "--rows", str(rows), "--sf-jsonl", str(sfj), "--out", str(out)]) == 0
    from chess_anti_engine.moves.encode import FULL_TO_COMPACT_POLICY, move_to_index

    board = chess.Board()
    idx = {u: int(FULL_TO_COMPACT_POLICY[move_to_index(chess.Move.from_uci(u), board)])
           for u in ("g1f3", "e2e4", "h2h4")}
    with np.load(out) as z:
        moves, cp = z["moves"], z["regret_cp"]
    got = {int(m): float(c) for m, c in zip(moves[0], cp[0]) if m >= 0}
    # best g1f3 30 (depth 11); e2e4 28 -> regret 2; h2h4 mated in 3 -> -9970 -> regret clipped to 1000
    assert got == {idx["g1f3"]: 0.0, idx["e2e4"]: 2.0, idx["h2h4"]: 1000.0}
    assert (moves[1] == -1).all()            # row 1 has no label: uncovered, not zero regret
    meta = json.loads(Path(str(out) + ".json").read_text())
    assert meta["rows_labelled"] == 1
    assert meta["cap_cp"] == 1000


def test_the_paired_regret_difference_is_net_minus_reference() -> None:
    """Review of #1465 (R8): a flipped sign would invert every vs_reference conclusion."""
    moves = np.array([[3, 4], [3, 4]])
    cp = np.array([[0.0, 100.0], [0.0, 60.0]])
    tab = {"moves": moves, "regret_cp": cp}
    pair = ras.paired_regret(np.array([4, 4]), np.array([3, 3]), tab, np.array([0, 1]))
    assert pair["diff_cp"] == pytest.approx(80.0)            # the net (a) plays the worse move: positive


def test_mate_scores_keep_the_distance_and_searchmoves_only_fill_gaps() -> None:
    """Review of #1465 (R3, R5): mate in 2 beats mate in 5 by 30 cp, not a flat 10000 tie; the
    searchmoves pass never overrides a MultiPV score."""
    a_lines = [_sf_line(10, 1, "e2e4", "mate 2"), _sf_line(10, 2, "d2d4", "mate 5")]
    rec = {"row": 0, "multipv_a": 2, "a": {"lines": a_lines},
           "b": {"searchmoves": ["d2d4", "h2h4"],
                 "lines": [_sf_line(9, 1, "d2d4", "cp 5"), _sf_line(9, 2, "h2h4", "cp -50")]}}
    scores, best = ras.sf_record_scores(rec)
    assert best == 9980
    assert scores["d2d4"] == 9950                            # MultiPV's mate 5, not the b pass's cp 5
    assert scores["h2h4"] == -50                             # the gap the b pass fills


def test_a_duplicate_fen_in_the_labels_is_refused(tmp_path: Path) -> None:
    import chess

    rows = tmp_path / "rows.npz"
    np.savez(rows, x=np.zeros((1, 175, 8, 8), dtype=np.float16), legal=np.ones((1, N_MOVES), dtype=bool),
             game_id=np.array([0]), root_fen=np.array([chess.STARTING_FEN]), history=np.array([""]))
    rec = {"row": 0, "fen": chess.Board().fen(), "multipv_a": 1, "a": {"lines": [_sf_line(9, 1, "e2e4", "cp 9")]}}
    sfj = tmp_path / "sf.jsonl"
    sfj.write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n")
    with pytest.raises(SystemExit, match="duplicate FEN"):
        ras.main(["sf-regret-table", "--rows", str(rows), "--sf-jsonl", str(sfj), "--out", str(tmp_path / "r.npz")])


def test_dose_refuses_a_pass_whose_mask_file_was_rewritten_at_the_same_path(tmp_path: Path) -> None:
    """Review of #1465 (R9): same mask spec, different contents -> another row set."""
    train = _tiny_run_with_snapshots(tmp_path)
    rows = _rows_npz(tmp_path / "a1.npz")
    np.savez(tmp_path / "ruler.npz", move=np.zeros(24, dtype=np.int64))
    leak = np.zeros(24, dtype=bool)
    leak[:3] = True
    np.savez(tmp_path / "leak.npz", leak=leak)
    spec = f"r={tmp_path / 'ruler.npz'}:move"
    mask = f"{tmp_path / 'leak.npz'}:leak"
    val = _validation(tmp_path, spec, passed=True, rows=rows, mask=mask, tag="m")
    leak[:6] = True
    np.savez(tmp_path / "leak.npz", leak=leak)                # rewritten in place
    out = tmp_path / "dose.json"
    assert ras.main(["dose", "--train-dir", str(train), "--rows", str(rows), "--ruler", spec, "--band-lo", "-1",
                     "--band-hi", "1", "--mask", mask, "--out-model", str(tmp_path / "played.pt"),
                     "--ruler-validation", str(val), "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert rec["ruler_validated"] is False
    assert "another mask" in rec["ruler_validation_reason"]
