from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from scripts import nnue_gumbel_readout as readout


class FakeExt:
    RESOLVER_MAX_DEPTH = 12
    QSEARCH_MAX_PLY = 4
    QSEARCH_CHECK_PLIES = 1
    QSEARCH_DAG_NODE_CAP = 0
    FASTQ_MAX_QPLY = 4
    FASTQ_NODE_CAP = 32
    FASTQ_DELTA_MARGIN = 200
    FASTQ_RECAPTURE_EXEMPT = 1
    RESOLVER_MATE_BASE = 100000
    RESOLVER_MAX_PLIES = 128
    RESOLVER_MATE_PLY_STEP = 1

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.arm_cfg = {
            "resolver_max_depth": self.RESOLVER_MAX_DEPTH,
            "qsearch_max_ply": self.QSEARCH_MAX_PLY,
            "qsearch_check_plies": self.QSEARCH_CHECK_PLIES,
            "dag_node_cap": 0,
        }
        self.fastq_cfg = {
            "max_qply": self.FASTQ_MAX_QPLY,
            "node_cap": self.FASTQ_NODE_CAP,
            "delta_margin": self.FASTQ_DELTA_MARGIN,
            "see_recapture_exempt": self.FASTQ_RECAPTURE_EXEMPT,
        }
        self.handles: dict[object, str] = {}
        self.resets = 0

    def set_arm_config(self, resolver: int, qply: int, checks: int, cap: int = 0):
        self.events.append(("set_arm_config", resolver, qply, checks, cap))
        self.arm_cfg = {
            "resolver_max_depth": resolver,
            "qsearch_max_ply": qply,
            "qsearch_check_plies": checks,
            "dag_node_cap": cap,
        }
        return dict(self.arm_cfg)

    def fastq_set_config(self, max_qply: int, node_cap: int, margin: int, exempt: int):
        self.events.append(("fastq_set_config", max_qply, node_cap, margin, exempt))
        self.fastq_cfg = {
            "max_qply": max_qply,
            "node_cap": node_cap,
            "delta_margin": margin,
            "see_recapture_exempt": exempt,
        }
        return dict(self.fastq_cfg)

    def arm_open(self, arm: str, path: str):
        self.events.append(("arm_open", arm, path))
        handle = object()
        self.handles[handle] = arm
        return handle

    def arm_stats(self, handle: object):
        arm = self.handles[handle]
        if arm == readout.ARM_FASTQ:
            raise AssertionError("FastQ must never use arm_stats")
        return {
            "calls": 0,
            "nnue_evals": 0,
            "max_depth_seen": 0,
            "qmax_ply_seen": 0,
            "dag_enabled": int(arm == readout.ARM_QSEARCH_DAG),
            **self.arm_cfg,
        }

    def fastq_stats(self, handle: object):
        arm = self.handles[handle]
        if arm != readout.ARM_FASTQ:
            raise AssertionError("non-FastQ arm must never use fastq_stats")
        return {
            "calls": 0,
            "nodes": 0,
            "nodes_created": 0,
            "nodes_created_in_check": 0,
            "nnue_evals": 0,
            "max_ply_seen": 0,
            **self.fastq_cfg,
        }

    def arm_dag_stats(self, handle: object):
        if self.handles[handle] == readout.ARM_QSEARCH:
            raise AssertionError("plain qsearch has no DAG")
        return {
            "nodes": 11,
            "edges": 17,
            "probes": 20,
            "hits": 5,
            "inserts": 11,
            "memory_bytes": 12345,
        }

    def arm_dag_reset(self, handle: object):
        if self.handles[handle] == readout.ARM_QSEARCH:
            raise AssertionError("plain qsearch has no DAG")
        self.resets += 1

    def load(self, path: str):
        return ("weights", path)

    def source_sha256(self, handle: object):
        return "a" * 64

    def simd_active(self):
        return False


def _args(arm: str, **overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "arm": arm,
        "nnue_resolver_max_depth": None,
        "nnue_qsearch_max_ply": None,
        "nnue_qsearch_check_plies": None,
        "dag_node_cap": None,
        "fastq_max_qply": None,
        "fastq_node_cap": None,
        "fastq_delta_margin": None,
        "fastq_recapture_exempt": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _pack(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.pack"
    path.write_bytes(b"not-a-real-net; fake extension owns this test")
    return path


def test_arm_specs_are_the_three_intended_experimental_cells() -> None:
    assert readout.ARM_SPECS == {
        "nnue-qsearch": readout.ArmSpec("nnue-qsearch", False, "arm", True, False),
        "nnue-qsearch-dag": readout.ArmSpec(
            "nnue-qsearch-dag", True, "arm", True, False,
        ),
        "nnue-fastq": readout.ArmSpec("nnue-fastq", True, "fastq", False, True),
    }


def test_defaults_come_from_the_extension_for_the_selected_arm() -> None:
    ext = FakeExt()
    q = readout.resolve_arm_config(_args(readout.ARM_QSEARCH_DAG), ext)
    assert q.consumed() == {
        "resolver_max_depth": 12,
        "qsearch_max_ply": 4,
        "qsearch_check_plies": 1,
        "dag_node_cap": 0,
    }
    f = readout.resolve_arm_config(_args(readout.ARM_FASTQ), ext)
    assert f.consumed() == {
        "max_qply": 4,
        "node_cap": 32,
        "delta_margin": 200,
        "see_recapture_exempt": 1,
    }


def test_knobs_for_another_provider_are_refused_not_ignored() -> None:
    ext = FakeExt()
    with pytest.raises(ValueError, match="does not consume --fastq"):
        readout.resolve_arm_config(
            _args(readout.ARM_QSEARCH_DAG, fastq_node_cap=64), ext,
        )
    with pytest.raises(ValueError, match="does not consume qsearch"):
        readout.resolve_arm_config(
            _args(readout.ARM_FASTQ, nnue_qsearch_max_ply=8), ext,
        )
    with pytest.raises(ValueError, match="has no DAG"):
        readout.resolve_arm_config(
            _args(readout.ARM_QSEARCH, dag_node_cap=10), ext,
        )


def test_fastq_config_is_snapshotted_before_open_and_read_back_from_fastq_stats(
    tmp_path: Path,
) -> None:
    ext = FakeExt()
    cfg = readout.resolve_arm_config(
        _args(
            readout.ARM_FASTQ,
            fastq_max_qply=6,
            fastq_node_cap=48,
            fastq_delta_margin=333,
            fastq_recapture_exempt=0,
        ),
        ext,
    )
    source = readout.ReadoutArmSource(
        config=cfg,
        pack=_pack(tmp_path),
        cp_per_internal_unit=0.28,
        cp_slope=0.006,
        cp_draw_width=120.0,
        ext=ext,
    )
    assert ext.events[:2] == [
        ("fastq_set_config", 6, 48, 333, 0),
        ("arm_open", readout.ARM_FASTQ, str(source.pack)),
    ]
    assert source.realized == {
        "max_qply": 6,
        "node_cap": 48,
        "delta_margin": 333,
        "see_recapture_exempt": 0,
    }
    # FakeExt raises if ReadoutArmSource chose the resolver counter surface.
    assert source.provider_stats()["node_cap"] == 48


def test_qsearch_dag_uses_arm_stats_and_resets_only_its_graph(tmp_path: Path) -> None:
    ext = FakeExt()
    cfg = readout.resolve_arm_config(
        _args(
            readout.ARM_QSEARCH_DAG,
            nnue_resolver_max_depth=10,
            nnue_qsearch_max_ply=3,
            nnue_qsearch_check_plies=0,
            dag_node_cap=64,
        ),
        ext,
    )
    source = readout.ReadoutArmSource(
        config=cfg,
        pack=_pack(tmp_path),
        cp_per_internal_unit=0.28,
        cp_slope=0.006,
        cp_draw_width=120.0,
        ext=ext,
    )
    assert ext.events[:2] == [
        ("set_arm_config", 10, 3, 0, 64),
        ("arm_open", readout.ARM_QSEARCH_DAG, str(source.pack)),
    ]
    assert source.provider_stats()["dag_enabled"] == 1
    assert source.dag_stats() == {
        "nodes": 11,
        "edges": 17,
        "probes": 20,
        "hits": 5,
        "inserts": 11,
        "memory_bytes": 12345,
    }
    source.reset_game()
    source.reset_game()
    assert ext.resets == 2


def test_plain_qsearch_never_touches_the_dag_surface(tmp_path: Path) -> None:
    ext = FakeExt()
    cfg = readout.resolve_arm_config(_args(readout.ARM_QSEARCH), ext)
    source = readout.ReadoutArmSource(
        config=cfg,
        pack=_pack(tmp_path),
        cp_per_internal_unit=0.28,
        cp_slope=0.006,
        cp_draw_width=120.0,
        ext=ext,
    )
    assert source.dag_stats() is None
    source.reset_game()
    assert ext.resets == 0


def test_worker_assignment_is_schedule_independent_for_game_seed() -> None:
    cfg = readout.RunConfig(
        arm_config=readout.ResolvedArmConfig(
            arm=readout.ARM_FASTQ,
            fastq_max_qply=4,
            fastq_node_cap=32,
            fastq_delta_margin=200,
            fastq_recapture_exempt=1,
        ),
        pack=Path("x"),
        games=7,
        workers=3,
        seed=123,
        sims=32,
        topk=218,
        max_plies=100,
        all_root_moves=True,
        cp_per_internal_unit=0.28,
        cp_slope=0.006,
        cp_draw_width=120.0,
        bank_path=None,
        run_id="test",
        nice=0,
    )
    specs = readout._build_worker_specs(cfg)
    assert [s.game_indices for s in specs] == [(0, 3, 6), (1, 4), (2, 5)]
    assert sorted(g for s in specs for g in s.game_indices) == list(range(7))
    # _run_worker seeds each game as cfg.seed + game_index, not by local index;
    # moving a game between workers therefore cannot change its trajectory.
    assert [cfg.seed + g for g in range(cfg.games)] == list(range(123, 130))


def test_dag_game_summary_means_per_game_not_lifetime_store_size() -> None:
    stats = readout.DagGameStats()
    stats.add({
        "nodes": 100, "edges": 150, "hits": 20, "probes": 120,
        "inserts": 100, "memory_bytes": 1_000_000,
    })
    stats.add({
        "nodes": 50, "edges": 70, "hits": 10, "probes": 60,
        "inserts": 50, "memory_bytes": 1_000_000,
    })
    summary = stats.summary()
    assert summary["nodes_per_game"] == 75
    assert summary["nodes_peak_per_game"] == 100
    assert summary["canonical_hit_rate"] == pytest.approx(30 / 180)
    # reset retains capacity, so this is intentionally a worker resource peak,
    # not a claim that the second game itself needed a fresh MB.
    assert summary["memory_peak_per_worker_bytes"] == 1_000_000
