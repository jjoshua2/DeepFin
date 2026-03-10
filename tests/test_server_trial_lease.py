from __future__ import annotations

import json
from pathlib import Path

from chess_anti_engine.server.lease import assign_trial_lease, available_trial_ids


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
