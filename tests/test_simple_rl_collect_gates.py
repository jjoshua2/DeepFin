"""scripts/simple_rl_collect_gates.py on a real (tiny, CPU) sealed collect."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import simple_rl_collect_gates as gates
from scripts import simple_rl_loop
from tests.test_simple_rl_resume import _setup
from tests.test_simple_rl_snapshots_arms import _argv, _stage


@pytest.fixture
def collect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    openings, model_config = _setup(tmp_path, monkeypatch)
    src = tmp_path / "src"
    assert simple_rl_loop.main(_stage(_argv(src, openings, model_config), "collect")) == 0
    return src / "gen_0001" / "collect"


def test_spearman_uses_average_ranks() -> None:
    # Hand-computed: x ranks (0, 1.5, 1.5, 3) vs y ranks (0, 1, 2, 3) -> rho = 4.5 / sqrt(4.5 * 5).
    assert gates.spearman(np.array([1.0, 2.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(
        4.5 / np.sqrt(4.5 * 5.0))
    assert gates.spearman(np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(-1.0)
    assert np.isnan(gates.spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])))


def test_report_on_a_sealed_collect(collect: Path, tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    assert gates.main([str(collect), "--train-target-temp", "1.0", "--train-target-temp", "2.0",
                       "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    games = [json.loads(x) for x in (collect / "games.jsonl").read_text().splitlines() if x.strip()]
    score = {"1-0": (1.0, 0.0), "0-1": (0.0, 1.0), "1/2-1/2": (0.5, 0.5)}
    # The book lines end with white to move (2 plies from the start position): the mover is white.
    assert r["book_end_mover_white_share"] == 1.0
    assert r["book_end_mover_score"] == pytest.approx(np.mean([score[g["result"]][0] for g in games]))
    stats = json.loads((collect / "manifest.json").read_text())["stats"]
    assert r["rows"] == stats["rows"] == r["sampled_rows"]
    assert r["rows_per_game"] == pytest.approx(stats["rows"] / stats["games"])
    assert r["draw_rate"] == pytest.approx(stats["draw_rate"])
    assert r["target_entropy"]["overall"] == pytest.approx(stats["target_entropy"])
    assert r["kl_target_prior"]["overall"] == pytest.approx(stats["kl_target_prior"])
    assert r["buffer_full_flushes"] == stats["buffer_full_flushes"]
    # Re-temper at 1.0 is the identity; 2.0 flattens.
    assert r["retempered_target_entropy"]["1"]["overall"] == pytest.approx(r["target_entropy"]["overall"])
    assert r["retempered_target_entropy"]["2"]["overall"] > r["target_entropy"]["overall"]
    cells = r["spearman_childq_prior"]
    assert sum(c["rows"] for c in cells.values()) == r["root_stats_rows"]
    sides = r["target_entropy"]
    assert min(sides["book_end_mover"], sides["reply"]) <= sides["overall"] <= max(sides["book_end_mover"],
                                                                                   sides["reply"])


def test_sample_is_uniform_over_every_shard_and_seeded(collect: Path) -> None:
    shards = sorted((collect / "shards").glob("shard_*.meta.npz"))
    assert len(shards) > 2
    sizes = [len(np.load(s)["game_id"]) for s in shards]
    _, meta, total = gates.sample_rows(collect, n=sum(sizes) // 2, seed=5)
    _, again, _ = gates.sample_rows(collect, n=sum(sizes) // 2, seed=5)
    assert total == sum(sizes)
    assert np.array_equal(meta["game_id"], again["game_id"])
    last_games = set(np.load(shards[-1])["game_id"].tolist())
    assert last_games & set(meta["game_id"].tolist())  # the tail shards are sampled, not just the head
    _, other, _ = gates.sample_rows(collect, n=sum(sizes) // 2, seed=6)
    assert _keys(meta) != _keys(other)


def _keys(meta: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    return list(zip(meta["game_id"].tolist(), meta["ply_index"].tolist(), strict=True))


def test_thresholds_exit_3_on_a_failed_gate(collect: Path, tmp_path: Path) -> None:
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"rows_per_game": {"min": 0.0}, "draw_rate": {"min": 0.0, "max": 1.0}}))
    assert gates.main([str(collect), "--thresholds", str(ok)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"rows_per_game": {"min": 1e9}}))
    assert gates.main([str(collect), "--thresholds", str(bad)]) == gates.EXIT_GATE
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"no.such.metric": {"min": 0.0}}))
    assert gates.main([str(collect), "--thresholds", str(missing)]) == gates.EXIT_GATE


def test_book_end_mover_is_the_side_to_move_at_start_ply() -> None:
    games = [{"result": "1-0", "start_ply": 2}, {"result": "1-0", "start_ply": 3},
             {"result": "1/2-1/2", "start_ply": 5}, {"result": "0-1", "start_ply": 4}]
    r = gates.game_metrics(games, rows=40)
    # white mover wins (1), black mover loses (0), black mover draws (0.5), white mover loses (0)
    assert r["book_end_mover_score"] == pytest.approx(1.5 / 4)
    assert r["white_score"] == pytest.approx(2.5 / 4)
    assert r["book_end_mover_white_share"] == 0.5
    assert (r["rows_per_game"], r["draw_rate"], r["decisive_rate"]) == (10.0, 0.25, 0.75)
