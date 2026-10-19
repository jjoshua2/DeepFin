"""Actual factory/option bodies with inert devices; no project or GPU imports."""
from __future__ import annotations

from types import MethodType
from typing import Any

import pytest

from tests.test_uci_topology_reuse import (
    Direct, Refusal, Wrapper, definitions, engine_fixture, leaves,
)


def _fixture(*, enabled: bool = False, parallel: str = "pucv", devices: int = 2) -> tuple[Any, Any, list]:
    worker: Any
    engine, worker, _models, messages, builds = engine_fixture(enabled, parallel, devices)
    options = engine._options
    options.use_vl = False
    options.rpg_open_vpa = -1
    options.rpg_min_vpa = -1
    options.rpg_min_keep = 2
    options.rpg_open_budget_frac = -1.0
    engine._search_devices = tuple("inert:" + str(i) for i in range(devices))
    methods = definitions(
        "chess_anti_engine/uci/engine.py",
        ["Engine._set_search_parallel", "Engine._try_install_root_parallel_gumbel",
         "Engine._reinstall_configured_search_path"],
        {"_println": messages.append, "_emit_info_string": messages.append},
    )
    for name, method in methods.items():
        setattr(engine, name.rsplit(".", 1)[-1], MethodType(method, engine))

    def clear() -> None:
        if worker._rpg_pool is not None:
            worker._rpg_pool.close()
            worker._rpg_pool = None

    old_set = worker.set_evaluator

    def set_evaluator(evaluator: Any) -> None:
        clear()
        old_set(evaluator)

    def install(factories: list, **_kwargs: Any) -> None:
        clear()
        worker.clear_multi_gpu_pucv()
        worker._rpg_pool = Wrapper([factory() for factory in factories])

    worker.clear_root_parallel_gumbel = clear
    worker.set_evaluator = set_evaluator
    worker.install_root_parallel_gumbel = install
    worker.realized_rpg_vloss_mode = lambda: 0
    worker.restore_classic_search_path = lambda: None
    return engine, worker, builds


@pytest.mark.parametrize(
    ("enabled", "parallel", "devices", "expected"),
    [(False, "pucv", 2, 2), (True, "pucv", 2, 1),
     (False, "gumbel", 2, 1), (False, "gumbel", 1, 1)],
    ids=["classic", "pucv-pool", "rpg", "single-device"],
)
def test_startup_primary_matches_selected_topology(enabled, parallel, devices, expected):
    _engine, worker, _builds = _fixture(enabled=enabled, parallel=parallel, devices=devices)
    assert len(leaves(worker.primary)) == expected


@pytest.mark.parametrize("enabled", [False, True], ids=["classic", "pucv-pool"])
def test_enter_rpg_rebuilds_primary_before_pool_install(enabled):
    engine, worker, builds = _fixture(enabled=enabled)
    old = leaves(worker.primary)
    engine._set_search_parallel("gumbel")
    assert builds[-1] == (enabled, "gumbel")
    assert len(leaves(worker.primary)) == 1
    assert len(leaves(worker._rpg_pool)) == 2
    assert all(leaf.closed == 1 for leaf in old)
    assert sum(leaf.closed == 0 for leaf in Direct.instances) == 3
    assert engine._options.search_parallel == "gumbel"
    assert engine._options.use_multi_gpu_pucv is enabled


@pytest.mark.parametrize("enabled", [False, True], ids=["classic", "pucv-pool"])
def test_leave_rpg_restores_primary_topology(enabled):
    engine, worker, _builds = _fixture(enabled=enabled)
    engine._set_search_parallel("gumbel")
    engine._set_search_parallel("pucv")
    assert worker._rpg_pool is None
    assert len(leaves(worker.primary)) == (1 if enabled else 2)
    assert engine._options.search_parallel == "pucv"
    assert (worker._pucv_pool is not None) is enabled


@pytest.mark.parametrize("available", [None, 1], ids=["missing", "single"])
def test_unavailable_rpg_does_not_rebuild_primary(available):
    engine, worker, builds = _fixture()
    engine._rebuild_multi_gpu_pucv_factories = (
        None if available is None else lambda *_args: [lambda: None]
    )
    primary = worker.primary
    engine._set_search_parallel("gumbel")
    assert worker.primary is primary
    assert len(builds) == 1
    assert worker._rpg_pool is None
    assert engine._options.search_parallel == "pucv"


def test_failed_primary_construction_restores_requested_mode():
    engine, worker, _builds = _fixture()
    primary = worker.primary

    def refuse(*_args, **_kwargs):
        assert engine._options.search_parallel == "gumbel"
        raise Refusal("inert factory refusal")

    engine._rebuild_evaluator = refuse
    with pytest.raises(Refusal):
        engine._set_search_parallel("gumbel")
    assert engine._options.search_parallel == "pucv"
    assert worker.primary is primary
    assert all(leaf.closed == 0 for leaf in leaves(primary))
    assert worker._rpg_pool is None


def test_rpg_reinstall_retains_primary_and_pucv_off_keeps_it_small():
    engine, worker, builds = _fixture()
    engine._set_search_parallel("gumbel")
    primary = worker.primary
    engine._reinstall_configured_search_path()
    assert worker.primary is primary
    assert len(builds) == 2
    engine._set_use_multi_gpu_pucv("false")
    assert len(leaves(worker.primary)) == 1
    assert len(leaves(worker._rpg_pool)) == 2
    assert sum(leaf.closed == 0 for leaf in Direct.instances) == 3
    assert engine._options.search_parallel == "gumbel"
