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
    # The stats.json fields under their own names, recomputed over every row.
    for k in ("target_entropy", "kl_target_prior", "mean_survivor_minus_node_q", "repeat_rate_within"):
        assert r[k] == pytest.approx(stats[k], abs=1e-12), k
    # All rows sampled here, so the sample's overall values are the same numbers.
    assert r["target_entropy_by_side"]["overall"] == pytest.approx(stats["target_entropy"])
    assert r["kl_target_prior_by_side"]["overall"] == pytest.approx(stats["kl_target_prior"])
    assert r["buffer_full_flushes"] == stats["buffer_full_flushes"]
    # Re-temper at 1.0 is the identity; 2.0 flattens.
    assert r["retempered_target_entropy"]["1"]["overall"] == pytest.approx(r["target_entropy"])
    assert r["retempered_target_entropy"]["2"]["overall"] > r["target_entropy"]
    cells = r["spearman_childq_prior"]
    assert sum(c["rows"] for c in cells.values()) == r["root_stats_rows"]
    sides = r["target_entropy_by_side"]
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


def test_a_nan_metric_fails_its_gate() -> None:
    report = {"a": float("nan"), "b": 0.5, "c": True, "d": {"e": None}, "f": float("nan"), "g": float("inf")}
    # An empty bound is a presence gate: the metric must exist and be finite.
    out = {g["metric"]: g["pass"] for g in gates.check_gates(
        report, {"a": {"min": 0.0}, "b": {"min": 0.0, "max": 1.0}, "c": {"min": 0.0}, "d.e": {"max": 1.0},
                 "f": {}, "g": {}, "b2": {}})}
    assert out == {"a": False, "b": True, "c": False, "d.e": False, "f": False, "g": False, "b2": False}
    assert gates.check_gates(report, {"b": {}})[0]["pass"] is True


def test_g_c2_equality_and_stats_fields_are_gateable(collect: Path, tmp_path: Path) -> None:
    import json as _json

    m = _json.loads((collect / "manifest.json").read_text())
    m["stats"].update(eval_max_batch_realised=128, leaf_rows_required=128)
    (collect / "manifest.json").write_text(_json.dumps(m))
    th = tmp_path / "t.json"
    th.write_text(_json.dumps({"repeat_rate_within": {"max": 0.05}, "target_entropy": {"min": 0.0},
                               "kl_target_prior": {"min": 0.0}, "mean_survivor_minus_node_q": {"min": -1.0}}))
    out = tmp_path / "r.json"
    assert gates.main([str(collect), "--thresholds", str(th), "--out", str(out)]) == 0
    r = _json.loads(out.read_text())
    assert r["eval_max_batch_equals_leaf_rows"] is True
    assert all(g["pass"] for g in r["gates"])
    m["stats"]["eval_max_batch_realised"] = 256
    (collect / "manifest.json").write_text(_json.dumps(m))
    assert gates.main([str(collect), "--out", str(out)]) == 0
    r = _json.loads(out.read_text())
    assert (r["eval_max_batch_equals_leaf_rows"], r["eval_cap_covers_leaf_rows"]) == (False, True)


def test_content_digest_ignores_shard_file_bytes_but_not_rows(collect: Path, tmp_path: Path) -> None:
    import shutil

    from chess_anti_engine.replay.shard import load_shard_arrays, save_local_shard_arrays

    base = gates.content_digest(collect)
    copy = tmp_path / "copy" / "gen_0001" / "collect"
    shutil.copytree(collect, copy)
    shard = sorted((copy / "shards").glob("shard_*.zarr"))[0]
    arrays, meta = load_shard_arrays(shard, lazy=False, validate=True)
    from chess_anti_engine.replay.shard import ShardMeta

    shutil.rmtree(shard)
    fields = {k: v for k, v in meta.items() if k in ShardMeta.__dataclass_fields__}
    fields["generated_at_unix"] = int(fields.get("generated_at_unix") or 0) + 12345  # a different file, same rows
    save_local_shard_arrays(shard, arrs=arrays, meta=ShardMeta(**fields))
    again = gates.content_digest(copy)
    assert again["content_sha256"] == base["content_sha256"]
    assert again["games_sha256"] == base["games_sha256"]
    arrays["wdl_target"] = (np.asarray(arrays["wdl_target"]) + 1) % 3
    shutil.rmtree(shard)
    save_local_shard_arrays(shard, arrs=arrays, meta=ShardMeta(**fields))
    assert gates.content_digest(copy)["content_sha256"] != base["content_sha256"]
