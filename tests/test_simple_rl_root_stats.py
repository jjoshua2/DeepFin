"""Root search statistics: schema round trip, mirror, and end-to-end rebuild.

The new optional shard arrays (root_log_prior / root_child_q /
root_child_visits / root_net_value / root_net_wdl / root_survivor_q /
root_node_q / root_start_visits / root_survivor_index / played_move_index /
search_budget) must round-trip, mirror
consistently with the policy, stay absent by default, and -- through the real
``play_batch`` -> finalize -> shard path -- reproduce every stored policy
target with ``rebuild_improved_policy``.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.encode import mirror_policy
from chess_anti_engine.replay.augment import maybe_mirror_batch_arrays, mirror_sample
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import (
    arrays_to_samples,
    load_shard_arrays,
    samples_to_arrays,
    save_local_shard_arrays,
)
from chess_anti_engine.simple_rl import collect as collect_mod
from chess_anti_engine.simple_rl.collect import CollectSettings, collect_generation, load_generation_rows
from tests.simple_rl_helpers import QuietStockfish, save_tiny_checkpoint

ROOT_FIELDS = ("root_log_prior", "root_child_q", "root_child_visits")
SCALAR_FIELDS = ("root_net_value", "root_survivor_q", "root_node_q", "root_start_visits",
                 "root_survivor_index", "played_move_index", "search_budget")


def _sample(rng: np.random.Generator, *, with_root: bool) -> ReplaySample:
    n = COMPACT_POLICY_SIZE
    legal = np.zeros(n, dtype=np.uint8)
    support = rng.choice(n, size=20, replace=False)
    legal[support] = 1
    target = np.zeros(n, dtype=np.float32)
    target[support] = rng.dirichlet(np.ones(20)).astype(np.float32)
    s = ReplaySample(x=rng.standard_normal((175, 8, 8)).astype(np.float32), policy_target=target,
                     wdl_target=int(rng.integers(0, 3)), legal_mask=legal)
    if with_root:
        lp = np.zeros(n, dtype=np.float32)
        lp[support] = np.log(rng.dirichlet(np.ones(20))).astype(np.float32)
        q = np.zeros(n, dtype=np.float32)
        q[support] = rng.uniform(-1, 1, 20).astype(np.float32)
        v = np.full(n, -1.0, dtype=np.float32)
        v[support] = rng.integers(0, 3000, 20).astype(np.float32)  # > 2048: float16 would round
        s.root_log_prior, s.root_child_q, s.root_child_visits = lp, q, v
        s.root_net_value = float(rng.uniform(-1, 1))
        s.root_net_wdl = rng.dirichlet(np.ones(3)).astype(np.float32)
        s.root_survivor_q = float(rng.uniform(-1, 1))
        s.root_node_q = float(rng.uniform(-1, 1))
        s.root_start_visits = int(rng.integers(0, 5000))
        s.root_survivor_index = int(support[0])
        s.played_move_index = int(support[1])
        s.search_budget = int(rng.integers(8, 4000))
    return s


def test_root_fields_round_trip_through_a_shard(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    samples = [_sample(rng, with_root=True), _sample(rng, with_root=False), _sample(rng, with_root=True)]
    save_local_shard_arrays(tmp_path / "s.zarr", arrs=samples_to_arrays(samples))
    arrays, _ = load_shard_arrays(tmp_path / "s.zarr", lazy=False, validate=True)
    for name in ROOT_FIELDS:
        assert arrays[name].dtype == np.float32
    back = arrays_to_samples(arrays)
    for orig, got in zip(samples, back, strict=True):
        for name in (*ROOT_FIELDS, "root_net_wdl", *SCALAR_FIELDS):
            a, b = getattr(orig, name), getattr(got, name)
            if a is None:
                assert b is None, name
            elif isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b, err_msg=name)
            else:
                assert b == pytest.approx(a, abs=1e-6), name


def test_default_samples_write_no_root_arrays(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    save_local_shard_arrays(tmp_path / "s.zarr", arrs=samples_to_arrays([_sample(rng, with_root=False)]))
    arrays, _ = load_shard_arrays(tmp_path / "s.zarr", lazy=False, validate=True)
    assert not {*ROOT_FIELDS, "root_net_wdl", *SCALAR_FIELDS} & set(arrays)


def test_root_fields_mirror_with_the_policy() -> None:
    rng = np.random.default_rng(2)
    s = _sample(rng, with_root=True)
    m = mirror_sample(s)
    for name in ROOT_FIELDS:
        np.testing.assert_array_equal(getattr(m, name), mirror_policy(getattr(s, name)))
    for name in ("root_net_value", "root_survivor_q", "root_node_q", "root_start_visits", "search_budget"):
        assert getattr(m, name) == getattr(s, name), name
    assert m.root_net_wdl is not None
    assert s.root_net_wdl is not None
    np.testing.assert_array_equal(m.root_net_wdl, s.root_net_wdl)
    # the move indices follow the same permutation as the policy target
    for index_name in ("root_survivor_index", "played_move_index"):
        onehot = np.zeros(COMPACT_POLICY_SIZE, dtype=np.float32)
        onehot[getattr(s, index_name)] = 1.0
        assert int(np.argmax(mirror_policy(onehot))) == getattr(m, index_name)
    back = mirror_sample(m)
    for name in ROOT_FIELDS:
        np.testing.assert_array_equal(getattr(back, name), getattr(s, name))
    # the batch mirror applies the identical column permutation
    arrays = samples_to_arrays([s, s])
    out = maybe_mirror_batch_arrays(arrays, rng=np.random.default_rng(0), prob=1.0, input_history_encoding=None)
    for name in ROOT_FIELDS:
        np.testing.assert_array_equal(out[name][0], getattr(m, name))
    assert int(out["played_move_index"][0]) == m.played_move_index


def _collect(tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch, **overrides) -> Path:
    monkeypatch.setattr(collect_mod, "_open_stockfish", lambda _s: QuietStockfish([0.0, 1.0, 0.0]))
    ckpt = tmp_path / "model.pt"
    if not ckpt.exists():
        save_tiny_checkpoint(ckpt, seed=5)
    openings = tmp_path / "train.jsonl"
    if not openings.exists():
        lines = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | e2e4 e7e5",
                 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | d2d4 d7d5 c2c4",
                 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | g1f3"]
        openings.write_text("".join(json.dumps({"id": f"o{i}", "book": 1, "line": ln}) + "\n"
                                    for i, ln in enumerate(lines)))
    settings = replace(CollectSettings(games=3, concurrency=3, sims=8, max_plies=24, seed=9, book_mix=(1.0,)),
                       **overrides)
    out = tmp_path / name / "gen_0001" / "collect"
    collect_generation(gen_dir=out, generation=1, checkpoint=ckpt, train_jsonl=openings, settings=settings)
    return out


def _rebuild_all(collect_dir: Path) -> tuple[int, dict[int, np.ndarray]]:
    from chess_anti_engine.mcts.gumbel_c import rebuild_improved_policy
    from chess_anti_engine.selfplay.config import GameConfig, SearchConfig
    from chess_anti_engine.selfplay.network_turn import build_selfplay_gumbel_config

    manifest = json.loads((collect_dir / "manifest.json").read_text())
    search = SearchConfig(**manifest["config"]["dataclasses"]["search"])
    cfg = build_selfplay_gumbel_config(search=search, game=GameConfig(), simulations=search.simulations)
    arrays, meta = load_generation_rows(collect_dir)
    has = arrays["has_root_log_prior"].astype(bool)
    checked = 0
    by_hash: dict[int, np.ndarray] = {}
    policy = arrays["has_policy"].astype(bool)
    for i in np.flatnonzero(has & policy & ~meta["tb_policy_override"]):
        support = arrays["root_child_visits"][i] >= 0
        rebuilt = rebuild_improved_policy(log_prior=arrays["root_log_prior"][i][support],
                                          completed_q=arrays["root_child_q"][i][support],
                                          visits=arrays["root_child_visits"][i][support],
                                          net_value=float(arrays["root_net_value"][i]), cfg=cfg)
        stored = arrays["policy_target"][i].astype(np.float64)
        np.testing.assert_allclose(stored[support], rebuilt, atol=2e-3)  # float16 storage
        assert stored[~support].sum() == 0
        assert support[arrays["played_move_index"][i]]
        assert support[arrays["root_survivor_index"][i]]
        assert arrays["legal_mask"][i][support].all()
        # the three root values are what their names say
        wdl = arrays["root_net_wdl"][i].astype(np.float64)
        assert float(arrays["root_net_value"][i]) == pytest.approx(wdl[0] - wdl[2], abs=1e-5)
        assert float(arrays["root_survivor_q"][i]) == pytest.approx(
            float(arrays["root_child_q"][i][arrays["root_survivor_index"][i]]), abs=1e-6)
        assert int(arrays["search_budget"][i]) == int(search.simulations)
        by_hash[int(meta["x_hash"][i])] = stored
        checked += 1
    return checked, by_hash


def test_stored_targets_rebuild_from_root_stats_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _collect(tmp_path, "base", monkeypatch)
    checked, base_targets = _rebuild_all(base)
    stats = json.loads((base / "stats.json").read_text())
    assert checked > 20
    assert stats["root_stats_rows"] >= checked
    # A pass-through knob, through the whole play_batch path, moves stored rows.
    moved = _collect(tmp_path, "cvisit", monkeypatch, c_visit=3.0)
    checked2, moved_targets = _rebuild_all(moved)
    assert checked2 > 20
    common = set(base_targets) & set(moved_targets)
    assert common, "no shared position between the two collections"
    assert max(float(np.abs(base_targets[h] - moved_targets[h]).max()) for h in common) > 1e-3


def test_tree_reuse_flag_controls_inherited_visits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse on (production): some roots start with carried visits, and the
    stored child visits exceed the ply's own budget. Off: every root starts
    empty and holds at most the budget."""
    # 32 sims over 4 candidates: the survivor's subtree holds visits below it,
    # which is what a carried root inherits. (At 8 sims over 8 candidates each
    # candidate gets one visit and there is nothing below it to carry.)
    reuse = _collect(tmp_path, "reuse", monkeypatch, sims=32, topk=4)
    cold = _collect(tmp_path, "cold", monkeypatch, sims=32, topk=4, tree_reuse=False)
    for d, expect_reuse in ((reuse, True), (cold, False)):
        arrays, _ = load_generation_rows(d)
        has = arrays["has_root_log_prior"].astype(bool)
        start = arrays["root_start_visits"][has]
        total = np.where(arrays["root_child_visits"] >= 0, arrays["root_child_visits"], 0).sum(axis=1)[has]
        manifest = json.loads((d / "manifest.json").read_text())
        assert manifest["config"]["dataclasses"]["search"]["gumbel_tree_reuse"] is expect_reuse
        if expect_reuse:
            assert (start > 0).mean() > 0.3
            assert (total > 32).any()
            np.testing.assert_array_equal(total[start > 0] >= start[start > 0], True)
        else:
            assert (start == 0).all()
            assert (total <= 32).all()
    stats = json.loads((reuse / "stats.json").read_text())
    assert stats["reused_root_share"] > 0.3


def test_root_full_width_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through collect -> play_batch -> finalize -> shard, with --root-full-width
    and the playout cap on: full rows hold a visit on every searched child and
    store their round-0 simulations as root_prelude_visits; fast rows skip round
    0 and spend exactly --fast-sims; search_budget stays the budget given.
    Mutants: fast plies widened (mask dropped) / prelude not stored."""
    sims, fast = 12, 4
    on = _collect(tmp_path, "fw", monkeypatch, sims=sims, full_prob=0.5, fast_sims=fast, fast_value_rows=True,
                  root_full_width=True)
    arrays, _ = load_generation_rows(on)
    has = arrays["has_root_log_prior"].astype(bool)
    policy = arrays["has_policy"].astype(bool)
    assert arrays["has_root_prelude_visits"][has].all()
    rcv = arrays["root_child_visits"]
    total = np.where(rcv >= 0, rcv, 0).sum(axis=1).astype(np.int64)
    start = arrays["root_start_visits"].astype(np.int64)
    budget = arrays["search_budget"].astype(np.int64)
    prelude = arrays["root_prelude_visits"].astype(np.int64)
    full, fastr = has & policy, has & ~policy
    assert full.sum() > 10
    assert fastr.sum() > 10
    assert set(np.unique(budget[full]).tolist()) == {sims}
    assert set(np.unique(budget[fastr]).tolist()) == {fast}
    np.testing.assert_array_equal(((rcv >= 0).sum(axis=1))[full], ((rcv > 0).sum(axis=1))[full])
    assert (prelude[full] > 0).mean() > 0.9
    assert (prelude[fastr] == 0).all()
    # A root proven mid-search stops early; every other row spends exactly this.
    assert (total[has] <= (start + prelude + budget)[has]).all()
    assert (total[has] == (start + prelude + budget)[has]).mean() > 0.95
    stats = json.loads((on / "stats.json").read_text())
    assert stats["mean_root_visited_children"] == pytest.approx(stats["mean_root_support_children"])
    assert stats["mean_root_prelude_visits"] == pytest.approx(float(prelude[full & arrays["has_root_prelude_visits"].astype(bool)].mean()))
    assert set(stats["rows_by_search_budget"]) == {str(fast), str(sims)}
    manifest = json.loads((on / "manifest.json").read_text())
    assert manifest["config"]["dataclasses"]["search"]["gumbel_root_full_width"] is True
    # Knob off: nothing new is stored and the width stats show unvisited children.
    off = _collect(tmp_path, "nofw", monkeypatch)
    off_arrays, _ = load_generation_rows(off)
    assert "root_prelude_visits" not in off_arrays or not off_arrays["has_root_prelude_visits"].any()
    off_stats = json.loads((off / "stats.json").read_text())
    assert "mean_root_prelude_visits" not in off_stats
    assert off_stats["mean_root_visited_children"] < off_stats["mean_root_support_children"]


def test_playout_cap_records_the_budget_and_stores_policy_only_at_full(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = _collect(tmp_path, "pcr", monkeypatch, sims=12, full_prob=0.5, fast_sims=4, fast_value_rows=True,
                   tree_reuse=False)
    arrays, _ = load_generation_rows(out)
    budget = arrays["search_budget"]
    assert arrays["has_search_budget"].all()
    assert set(np.unique(budget).tolist()) == {4, 12}
    policy = arrays["has_policy"].astype(bool)
    assert (budget[policy] == 12).all()
    assert (~policy[budget == 4]).all()
    # fast rows keep their (4-sim) search statistics for value targets
    fast_root = arrays["has_root_log_prior"].astype(bool)[budget == 4]
    assert fast_root.mean() > 0.9
    visits = np.where(arrays["root_child_visits"] >= 0, arrays["root_child_visits"], 0).sum(axis=1)
    assert (visits[(budget == 4) & arrays["has_root_log_prior"].astype(bool)] <= 4).all()
    stats = json.loads((out / "stats.json").read_text())
    assert set(stats["rows_by_search_budget"]) == {"4", "12"}


def _bit_squares(bb: int) -> list[int]:
    return [s for s in range(64) if (int(bb) >> s) & 1]


def test_played_move_is_the_sampled_move_and_replays_to_the_next_row(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At positive temperature the played move differs from the halving
    survivor on some rows, and the stored ``played_move_index`` is the move
    actually pushed: applying it to row i's pieces gives row i+1's pieces
    (seen from the other side)."""
    from chess_anti_engine.encoding.plane_decode import decode_step0_bitboards
    from chess_anti_engine.moves.encode import _PLANE_TO_DELTA, COMPACT_TO_FULL_POLICY, PLANE_COUNT

    out = _collect(tmp_path, "temp", monkeypatch, temp_start=1.5, temp_floor=1.0, tree_reuse=False)
    arrays, meta = load_generation_rows(out)
    has = arrays["has_root_log_prior"].astype(bool)
    played, survivor = arrays["played_move_index"], arrays["root_survivor_index"]
    assert (played[has] != survivor[has]).sum() >= 5
    bbs = decode_step0_bitboards(arrays["x"])
    key = {(int(g), int(p)): i for i, (g, p) in enumerate(zip(meta["game_id"], meta["ply_index"], strict=True))}
    checked = differing = 0
    for (g, p), i in key.items():
        j = key.get((g, p + 1))
        if j is None or not has[i]:
            continue
        full = int(COMPACT_TO_FULL_POLICY[int(played[i])])
        frm, plane = full // PLANE_COUNT, full % PLANE_COUNT
        if plane >= 64:
            continue  # underpromotion
        df, dr = _PLANE_TO_DELTA[plane]
        to = frm + 8 * dr + df
        us = [int(b) for b in bbs[i][:6]]
        piece = next(k for k in range(6) if (us[k] >> frm) & 1)
        if (piece == 0 and to >= 56) or (piece == 5 and abs(df) == 2):
            continue  # promotion / castling move more than one piece
        us[piece] = (us[piece] & ~(1 << frm)) | (1 << to)
        them_next = [int(b) for b in bbs[j][6:12]]
        flipped = [sum(1 << (s ^ 56) for s in _bit_squares(b)) for b in us]
        assert flipped == them_next, (g, p)
        checked += 1
        differing += int(played[i] != survivor[i])
    assert checked > 20
    assert differing >= 3  # the replay check covers rows where played != survivor
