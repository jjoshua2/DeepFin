from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


def normalize_trial_id(trial_id: str | None) -> str | None:
    if trial_id is None:
        return None
    tid = str(trial_id).strip()
    return tid or None


def lease_path(*, leases_root: Path, lease_id: str) -> Path:
    return leases_root / f"{lease_id}.json"


def load_lease(*, leases_root: Path, lease_id: str) -> dict[str, Any] | None:
    p = lease_path(leases_root=leases_root, lease_id=lease_id)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def save_lease(*, leases_root: Path, lease: dict[str, Any]) -> None:
    lease_id = str(lease.get("lease_id") or "").strip()
    if not lease_id:
        raise ValueError("lease_id required")
    p = lease_path(leases_root=leases_root, lease_id=lease_id)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(lease, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def prune_expired_leases(*, leases_root: Path, now_unix: int) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for p in sorted(leases_root.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            p.unlink(missing_ok=True)
            continue
        if not isinstance(data, dict):
            p.unlink(missing_ok=True)
            continue
        expires_at = int(data.get("expires_at_unix") or 0)
        if expires_at <= now_unix:
            p.unlink(missing_ok=True)
            continue
        active.append(data)
    return active


def available_trial_ids(*, server_root: Path, publish_dir: str = "publish") -> list[str]:
    trials_root = server_root / "trials"
    entries: list[tuple[str, int]] = []
    if not trials_root.exists():
        return []
    for mf in sorted(trials_root.glob(f"*/{publish_dir}/manifest.json")):
        trial_id = normalize_trial_id(mf.parent.parent.name)
        if trial_id is None:
            continue
        try:
            payload = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        try:
            server_time = int(payload.get("server_time_unix") or 0)
        except Exception:
            server_time = 0
        entries.append((trial_id, server_time))
    if not entries:
        return []
    newest_trial_id, _ = max(entries, key=lambda item: int(item[1]))
    current_prefix = str(newest_trial_id).split("_", 1)[0]
    return [
        trial_id
        for trial_id, _ in entries
        if str(trial_id).split("_", 1)[0] == current_prefix
    ]


def lease_counts_by_trial(*, active_leases: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for lease in active_leases:
        tid = normalize_trial_id(lease.get("trial_id"))
        if tid is None:
            continue
        counts[tid] = int(counts.get(tid, 0)) + 1
    return counts


def pick_trial_for_lease(
    *,
    available_trials: list[str],
    active_leases: list[dict[str, Any]],
    manifest_loader,
) -> str | None:
    if not available_trials:
        return None
    counts = lease_counts_by_trial(active_leases=active_leases)
    manifests: dict[str, dict[str, Any]] = {}
    max_iter = 0
    for tid in available_trials:
        mf = manifest_loader(tid) or {}
        manifests[tid] = mf
        try:
            max_iter = max(max_iter, int(mf.get("training_iteration") or 0))
        except Exception:
            pass

    def _score(tid: str) -> tuple[float, int, str]:
        mf = manifests.get(tid) or {}
        try:
            iter_idx = int(mf.get("training_iteration") or 0)
        except Exception:
            iter_idx = 0
        lag = max(0, int(max_iter) - int(iter_idx))
        active = int(counts.get(tid, 0))
        lag_bias = min(1.0, 0.25 * float(lag))
        load_ratio = float(active) / float(1.0 + lag_bias)
        return (load_ratio, -lag, tid)

    return min(available_trials, key=_score)


def assign_trial_lease(
    *,
    leases_root: Path,
    username: str,
    worker_info: dict[str, Any],
    available_trials: list[str],
    manifest_loader,
    requested_lease_id: str | None = None,
    requested_trial_id: str | None = None,
    lease_seconds: int = 3600,
    now_unix: int | None = None,
) -> dict[str, Any]:
    leases_root.mkdir(parents=True, exist_ok=True)
    now_unix = int(time.time()) if now_unix is None else int(now_unix)
    active_leases = prune_expired_leases(leases_root=leases_root, now_unix=now_unix)
    requested_trial_id = normalize_trial_id(requested_trial_id)
    requested_lease_id = str(requested_lease_id or "").strip()

    if requested_lease_id:
        existing = load_lease(leases_root=leases_root, lease_id=requested_lease_id)
        if existing is not None and str(existing.get("username") or "") == str(username):
            existing["last_heartbeat_unix"] = now_unix
            existing["expires_at_unix"] = now_unix + int(lease_seconds)
            existing["worker_info"] = worker_info
            save_lease(leases_root=leases_root, lease=existing)
            return existing

    chosen_trial_id = requested_trial_id if requested_trial_id in available_trials else None
    if chosen_trial_id is None and available_trials:
        chosen_trial_id = pick_trial_for_lease(
            available_trials=available_trials,
            active_leases=active_leases,
            manifest_loader=manifest_loader,
        )

    api_prefix = "/v1" if chosen_trial_id is None else f"/v1/trials/{chosen_trial_id}"
    lease = {
        "lease_id": uuid.uuid4().hex,
        "username": str(username),
        "trial_id": chosen_trial_id,
        "api_prefix": api_prefix,
        "issued_at_unix": now_unix,
        "last_heartbeat_unix": now_unix,
        "expires_at_unix": now_unix + int(lease_seconds),
        "worker_info": worker_info,
    }
    save_lease(leases_root=leases_root, lease=lease)
    return lease
