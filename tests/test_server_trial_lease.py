from __future__ import annotations

import json
from pathlib import Path

from chess_anti_engine.server.lease import (
    assign_trial_lease,
    available_trial_ids,
    pick_trial_for_lease,
)


def _write_manifest(root: Path, trial_id: str) -> None:
    publish = root / "trials" / trial_id / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    (publish / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "server_version": "0.0.1",
                "min_worker_version": "0.0.1",
                "trial_id": trial_id,
                "task": {"type": "selfplay"},
            }
        ),
        encoding="utf-8",
    )


def test_lease_trial_is_sticky_and_balanced(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "trial_a")
    _write_manifest(tmp_path, "trial_b")
    leases_root = tmp_path / "leases"

    def _load_manifest(trial_id: str | None):
        if trial_id is None:
            return None
        mf = tmp_path / "trials" / trial_id / "publish" / "manifest.json"
        return json.loads(mf.read_text(encoding="utf-8"))

    trials = available_trial_ids(server_root=tmp_path)
    assert trials == ["trial_a", "trial_b"]

    lease1 = assign_trial_lease(
        leases_root=leases_root,
        username="worker",
        worker_info={"hostname": "box1"},
        available_trials=trials,
        manifest_loader=_load_manifest,
        now_unix=100,
    )
    assert lease1["trial_id"] == "trial_a"
    assert lease1["api_prefix"] == "/v1/trials/trial_a"

    lease2 = assign_trial_lease(
        leases_root=leases_root,
        username="worker",
        worker_info={"hostname": "box2"},
        available_trials=trials,
        manifest_loader=_load_manifest,
        now_unix=101,
    )
    assert lease2["trial_id"] == "trial_b"

    lease3 = assign_trial_lease(
        leases_root=leases_root,
        username="worker",
        worker_info={"hostname": "box1"},
        available_trials=trials,
        manifest_loader=_load_manifest,
        requested_lease_id=str(lease1["lease_id"]),
        requested_trial_id=str(lease1["trial_id"]),
        now_unix=102,
    )
    assert lease3["lease_id"] == lease1["lease_id"]
    assert lease3["trial_id"] == lease1["trial_id"]


def test_pick_trial_prefers_under_floor_before_faster_trials() -> None:
    manifests = {
        "trial_a": {"training_iteration": 10},
        "trial_b": {"training_iteration": 10},
        "trial_c": {"training_iteration": 9},
    }
    active_leases = [
        {"lease_id": "a1", "trial_id": "trial_a", "last_heartbeat_unix": 100, "worker_info": {"worker_id": "wa1"}},
        {"lease_id": "a2", "trial_id": "trial_a", "last_heartbeat_unix": 101, "worker_info": {"worker_id": "wa2"}},
        {"lease_id": "b1", "trial_id": "trial_b", "last_heartbeat_unix": 102, "worker_info": {"worker_id": "wb1"}},
        {"lease_id": "b2", "trial_id": "trial_b", "last_heartbeat_unix": 103, "worker_info": {"worker_id": "wb2"}},
        {"lease_id": "c1", "trial_id": "trial_c", "last_heartbeat_unix": 104, "worker_info": {"worker_id": "wc1"}},
    ]
    throughput = {
        "trial_a": {"ema_positions_per_s": 2.0},
        "trial_b": {"ema_positions_per_s": 1.0},
        "trial_c": {"ema_positions_per_s": 20.0},
    }

    chosen = pick_trial_for_lease(
        available_trials=["trial_a", "trial_b", "trial_c"],
        active_leases=active_leases,
        manifest_loader=lambda tid: manifests[tid],
        trial_throughput_loader=lambda tid: throughput.get(tid, {}),
        min_workers_per_trial=2,
        max_worker_delta_per_rebalance=1,
    )

    assert chosen == "trial_c"


def test_pick_trial_prefers_slower_trial_when_worker_floor_is_satisfied() -> None:
    manifests = {
        "trial_a": {"training_iteration": 12},
        "trial_b": {"training_iteration": 12},
    }
    active_leases = [
        {"lease_id": "a1", "trial_id": "trial_a", "last_heartbeat_unix": 100, "worker_info": {"worker_id": "wa1"}},
        {"lease_id": "a2", "trial_id": "trial_a", "last_heartbeat_unix": 101, "worker_info": {"worker_id": "wa2"}},
        {"lease_id": "b1", "trial_id": "trial_b", "last_heartbeat_unix": 102, "worker_info": {"worker_id": "wb1"}},
        {"lease_id": "b2", "trial_id": "trial_b", "last_heartbeat_unix": 103, "worker_info": {"worker_id": "wb2"}},
    ]
    throughput = {
        "trial_a": {"ema_positions_per_s": 12.0},
        "trial_b": {"ema_positions_per_s": 4.0},
    }

    chosen = pick_trial_for_lease(
        available_trials=["trial_a", "trial_b"],
        active_leases=active_leases,
        manifest_loader=lambda tid: manifests[tid],
        trial_throughput_loader=lambda tid: throughput.get(tid, {}),
        min_workers_per_trial=2,
        max_worker_delta_per_rebalance=1,
    )

    assert chosen == "trial_b"


def test_pick_trial_prioritizes_iteration_lag_over_throughput() -> None:
    manifests = {
        "trial_a": {"training_iteration": 5},
        "trial_b": {"training_iteration": 3},
    }
    active_leases = [
        {"lease_id": "a1", "trial_id": "trial_a", "last_heartbeat_unix": 100, "worker_info": {"worker_id": "wa1"}},
        {"lease_id": "a2", "trial_id": "trial_a", "last_heartbeat_unix": 101, "worker_info": {"worker_id": "wa2"}},
        {"lease_id": "b1", "trial_id": "trial_b", "last_heartbeat_unix": 102, "worker_info": {"worker_id": "wb1"}},
        {"lease_id": "b2", "trial_id": "trial_b", "last_heartbeat_unix": 103, "worker_info": {"worker_id": "wb2"}},
    ]
    throughput = {
        "trial_a": {"ema_positions_per_s": 1.0},
        "trial_b": {"ema_positions_per_s": 100.0},
    }

    chosen = pick_trial_for_lease(
        available_trials=["trial_a", "trial_b"],
        active_leases=active_leases,
        manifest_loader=lambda tid: manifests[tid],
        trial_throughput_loader=lambda tid: throughput.get(tid, {}),
        min_workers_per_trial=2,
        max_worker_delta_per_rebalance=1,
    )

    assert chosen == "trial_b"


def test_assign_trial_keeps_worker_on_source_when_source_is_at_floor(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "trial_a")
    _write_manifest(tmp_path, "trial_b")
    leases_root = tmp_path / "leases"

    def _load_manifest(trial_id: str | None):
        if trial_id is None:
            return None
        mf = tmp_path / "trials" / trial_id / "publish" / "manifest.json"
        return json.loads(mf.read_text(encoding="utf-8"))

    trials = available_trial_ids(server_root=tmp_path)
    assert trials == ["trial_a", "trial_b"]

    assign_trial_lease(
        leases_root=leases_root,
        username="worker",
        worker_info={"worker_id": "wa1", "hostname": "box1"},
        available_trials=trials,
        manifest_loader=_load_manifest,
        now_unix=100,
        min_workers_per_trial=2,
    )
    assign_trial_lease(
        leases_root=leases_root,
        username="worker",
        worker_info={"worker_id": "wa2", "hostname": "box2"},
        available_trials=trials,
        manifest_loader=_load_manifest,
        now_unix=101,
        min_workers_per_trial=2,
    )
    assign_trial_lease(
        leases_root=leases_root,
        username="worker",
        worker_info={"worker_id": "wb1", "hostname": "box3"},
        available_trials=trials,
        manifest_loader=_load_manifest,
        now_unix=102,
        min_workers_per_trial=2,
    )
    assign_trial_lease(
        leases_root=leases_root,
        username="worker",
        worker_info={"worker_id": "wb2", "hostname": "box4"},
        available_trials=trials,
        manifest_loader=_load_manifest,
        now_unix=103,
        min_workers_per_trial=2,
    )

    moved = assign_trial_lease(
        leases_root=leases_root,
        username="worker",
        worker_info={"worker_id": "wa1", "hostname": "box1"},
        available_trials=trials,
        manifest_loader=_load_manifest,
        now_unix=104,
        min_workers_per_trial=2,
        max_worker_delta_per_rebalance=1,
    )

    assert moved["trial_id"] == "trial_a"


def test_requested_stale_lease_is_not_reused_when_trial_is_unavailable(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "trial_a")
    _write_manifest(tmp_path, "trial_b")
    leases_root = tmp_path / "leases"

    def _load_manifest(trial_id: str | None):
        if trial_id is None:
            return None
        mf = tmp_path / "trials" / trial_id / "publish" / "manifest.json"
        return json.loads(mf.read_text(encoding="utf-8"))

    trials = available_trial_ids(server_root=tmp_path)
    original = assign_trial_lease(
        leases_root=leases_root,
        username="worker",
        worker_info={"worker_id": "wa1", "hostname": "box1"},
        available_trials=trials,
        manifest_loader=_load_manifest,
        requested_trial_id="trial_a",
        now_unix=100,
    )
    assert original["trial_id"] == "trial_a"

    reassigned = assign_trial_lease(
        leases_root=leases_root,
        username="worker",
        worker_info={"worker_id": "wa1", "hostname": "box1"},
        available_trials=["trial_b"],
        manifest_loader=_load_manifest,
        requested_lease_id=str(original["lease_id"]),
        now_unix=101,
    )

    assert reassigned["trial_id"] == "trial_b"
    assert reassigned["lease_id"] == original["lease_id"]
    assert len(list(leases_root.glob("*.json"))) == 1
