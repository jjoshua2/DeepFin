from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from chess_anti_engine.utils.atomic import atomic_write_text


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
    atomic_write_text(p, json.dumps(lease, indent=2, sort_keys=True))


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
    active_prefix_path = Path(server_root) / "active_run_prefix.txt"
    current_prefix: str | None = None
    try:
        active_prefix = active_prefix_path.read_text(encoding="utf-8").strip()
        if active_prefix:
            current_prefix = str(active_prefix)
    except Exception:
        current_prefix = None
    if not current_prefix:
        newest_trial_id, _ = max(entries, key=lambda item: int(item[1]))
        current_prefix = str(newest_trial_id).split("_", 1)[0]
    return [
        trial_id
        for trial_id, _ in entries
        if str(trial_id).split("_", 1)[0] == current_prefix
    ]


def active_run_prefix(*, server_root: Path, publish_dir: str = "publish") -> str | None:
    active_prefix_path = Path(server_root) / "active_run_prefix.txt"
    try:
        active_prefix = active_prefix_path.read_text(encoding="utf-8").strip()
        if active_prefix:
            return str(active_prefix)
    except Exception:
        pass

    trials_root = Path(server_root) / "trials"
    entries: list[tuple[str, int]] = []
    if not trials_root.exists():
        return None
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
        return None
    newest_trial_id, _ = max(entries, key=lambda item: int(item[1]))
    return str(newest_trial_id).split("_", 1)[0]


def prune_non_active_run_leases(*, leases_root: Path, active_prefix: str | None) -> None:
    if not active_prefix:
        return
    for p in sorted(leases_root.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            p.unlink(missing_ok=True)
            continue
        if not isinstance(data, dict):
            p.unlink(missing_ok=True)
            continue
        trial_id = normalize_trial_id(data.get("trial_id"))
        if trial_id is None:
            continue
        if str(trial_id).split("_", 1)[0] != str(active_prefix):
            p.unlink(missing_ok=True)


def _worker_key(lease: dict[str, Any]) -> str:
    worker_info = lease.get("worker_info")
    if isinstance(worker_info, dict):
        worker_id = str(worker_info.get("worker_id") or "").strip()
        if worker_id:
            return worker_id
    lease_id = str(lease.get("lease_id") or "").strip()
    return lease_id or uuid.uuid4().hex


def active_worker_assignments(*, active_leases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Deduplicate active leases down to one current assignment per worker."""
    by_worker: dict[str, dict[str, Any]] = {}
    for lease in active_leases:
        key = _worker_key(lease)
        prev = by_worker.get(key)
        if prev is None:
            by_worker[key] = lease
            continue
        prev_hb = int(prev.get("last_heartbeat_unix") or 0)
        cur_hb = int(lease.get("last_heartbeat_unix") or 0)
        if cur_hb >= prev_hb:
            by_worker[key] = lease
    return by_worker


def lease_counts_by_trial(*, active_leases: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for lease in active_worker_assignments(active_leases=active_leases).values():
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
    trial_throughput_loader=None,
    min_workers_per_trial: int = 1,
    max_worker_delta_per_rebalance: int = 1,
) -> str | None:
    if not available_trials:
        return None
    counts = lease_counts_by_trial(active_leases=active_leases)
    manifests: dict[str, dict[str, Any]] = {}
    throughputs: dict[str, float] = {}
    max_iter = 0
    for tid in available_trials:
        mf = manifest_loader(tid) or {}
        manifests[tid] = mf
        try:
            max_iter = max(max_iter, int(mf.get("training_iteration") or 0))
        except Exception:
            pass
        tp = trial_throughput_loader(tid) if callable(trial_throughput_loader) else {}
        if not isinstance(tp, dict):
            tp = {}
        try:
            throughputs[tid] = max(0.0, float(tp.get("ema_positions_per_s", tp.get("avg_positions_per_s", 0.0)) or 0.0))
        except Exception:
            throughputs[tid] = 0.0

    min_workers_per_trial = max(0, int(min_workers_per_trial))
    max_worker_delta_per_rebalance = max(0, int(max_worker_delta_per_rebalance))
    if available_trials:
        min_active = min(int(counts.get(tid, 0)) for tid in available_trials)
    else:
        min_active = 0

    under_floor = [tid for tid in available_trials if int(counts.get(tid, 0)) < min_workers_per_trial]
    if under_floor:
        def _underfloor_score(tid: str) -> tuple[int, float, int, str]:
            mf = manifests.get(tid) or {}
            try:
                iter_idx = int(mf.get("training_iteration") or 0)
            except Exception:
                iter_idx = 0
            lag = max(0, int(max_iter) - int(iter_idx))
            return (
                int(counts.get(tid, 0)),
                float(throughputs.get(tid, 0.0)),
                -lag,
                tid,
            )

        return min(under_floor, key=_underfloor_score)

    eligible = [
        tid
        for tid in available_trials
        if int(counts.get(tid, 0)) <= int(min_active + max_worker_delta_per_rebalance)
    ]
    if not eligible:
        eligible = list(available_trials)

    def _score(tid: str) -> tuple[int, int, float, str]:
        mf = manifests.get(tid) or {}
        try:
            iter_idx = int(mf.get("training_iteration") or 0)
        except Exception:
            iter_idx = 0
        lag = max(0, int(max_iter) - int(iter_idx))
        active = int(counts.get(tid, 0))
        throughput = float(throughputs.get(tid, 0.0))
  # Primary goal: keep active trials close in iteration progress.
  # Only use throughput as a secondary tiebreaker once lag is equal.
        return (-lag, active, throughput, tid)

    return min(eligible, key=_score)


def _try_renew_existing_lease(
    *,
    leases_root: Path,
    requested_lease_id: str,
    username: str,
    worker_info: dict[str, Any],
    available_trials: list[str],
    now_unix: int,
    lease_seconds: int,
) -> dict[str, Any] | None:
    """Heartbeat path: extend the worker's existing lease if still valid.

    Returns the renewed lease, or None if there's nothing to renew (no such
    lease, owned by another user, or its trial is gone — we delete those).
    """
    if not requested_lease_id:
        return None
    existing = load_lease(leases_root=leases_root, lease_id=requested_lease_id)
    if existing is None or str(existing.get("username") or "") != str(username):
        return None
    if normalize_trial_id(existing.get("trial_id")) not in available_trials:
  # Trial was retired; drop the stale lease so the caller can pick a new trial.
        lease_path(leases_root=leases_root, lease_id=requested_lease_id).unlink(missing_ok=True)
        return None
    existing["last_heartbeat_unix"] = now_unix
    existing["expires_at_unix"] = now_unix + int(lease_seconds)
    existing["worker_info"] = worker_info
    save_lease(leases_root=leases_root, lease=existing)
    return existing


def _find_same_worker_leases(
    active_leases: list[dict[str, Any]], *, username: str, worker_id: str,
) -> list[dict[str, Any]]:
    """All active leases owned by the same (username, worker_id) tuple."""
    if not worker_id:
        return []
    out: list[dict[str, Any]] = []
    for lease in active_leases:
        if str(lease.get("username") or "") != str(username):
            continue
        info = lease.get("worker_info")
        if isinstance(info, dict) and str(info.get("worker_id") or "").strip() == worker_id:
            out.append(lease)
    return out


def _choose_trial_for_lease(
    *,
    requested_trial_id: str | None,
    current_trial_id: str | None,
    available_trials: list[str],
    active_leases: list[dict[str, Any]],
    manifest_loader,
    trial_throughput_loader,
    min_workers_per_trial: int,
    max_worker_delta_per_rebalance: int,
) -> str | None:
    """Pick which trial to assign. Order: requested → current (if not over-served) → balancer."""
    if requested_trial_id in available_trials:
        return requested_trial_id
    counts = lease_counts_by_trial(active_leases=active_leases)
    if (
        current_trial_id in available_trials
        and int(counts.get(str(current_trial_id), 0)) <= int(max(0, min_workers_per_trial))
    ):
        return current_trial_id
    if available_trials:
        return pick_trial_for_lease(
            available_trials=available_trials,
            active_leases=active_leases,
            manifest_loader=manifest_loader,
            trial_throughput_loader=trial_throughput_loader,
            min_workers_per_trial=min_workers_per_trial,
            max_worker_delta_per_rebalance=max_worker_delta_per_rebalance,
        )
    return None


def assign_trial_lease(
    *,
    leases_root: Path,
    username: str,
    worker_info: dict[str, Any],
    available_trials: list[str],
    manifest_loader,
    trial_throughput_loader=None,
    requested_lease_id: str | None = None,
    requested_trial_id: str | None = None,
    lease_seconds: int = 3600,
    min_workers_per_trial: int = 1,
    max_worker_delta_per_rebalance: int = 1,
    now_unix: int | None = None,
) -> dict[str, Any]:
    leases_root.mkdir(parents=True, exist_ok=True)
    now_unix = int(time.time()) if now_unix is None else int(now_unix)
    active_leases = prune_expired_leases(leases_root=leases_root, now_unix=now_unix)
    requested_trial_id = normalize_trial_id(requested_trial_id)
    requested_lease_id = str(requested_lease_id or "").strip()
    # Defense-in-depth: app.py already normalizes worker_info, but a non-HTTP
    # caller (test, future internal path) passing None or a non-dict payload
    # would otherwise 500 with AttributeError on the .get() below.
    if not isinstance(worker_info, dict):
        worker_info = {}
    worker_id = str(worker_info.get("worker_id") or "").strip()

    renewed = _try_renew_existing_lease(
        leases_root=leases_root,
        requested_lease_id=requested_lease_id,
        username=username,
        worker_info=worker_info,
        available_trials=available_trials,
        now_unix=now_unix,
        lease_seconds=lease_seconds,
    )
    if renewed is not None:
        return renewed

    same_worker_leases = _find_same_worker_leases(
        active_leases, username=username, worker_id=worker_id,
    )
    same_worker_leases.sort(
        key=lambda lease: int(lease.get("last_heartbeat_unix") or 0), reverse=True,
    )
    current_trial_id = (
        normalize_trial_id(same_worker_leases[0].get("trial_id"))
        if same_worker_leases else None
    )

    chosen_trial_id = _choose_trial_for_lease(
        requested_trial_id=requested_trial_id,
        current_trial_id=current_trial_id,
        available_trials=available_trials,
        active_leases=active_leases,
        manifest_loader=manifest_loader,
        trial_throughput_loader=trial_throughput_loader,
        min_workers_per_trial=min_workers_per_trial,
        max_worker_delta_per_rebalance=max_worker_delta_per_rebalance,
    )

    reuse_lease_id = (
        str(same_worker_leases[0].get("lease_id") or "").strip()
        if same_worker_leases else ""
    )
    lease = {
        "lease_id": reuse_lease_id or uuid.uuid4().hex,
        "username": str(username),
        "trial_id": chosen_trial_id,
        "api_prefix": "/v1" if chosen_trial_id is None else f"/v1/trials/{chosen_trial_id}",
        "issued_at_unix": now_unix,
        "last_heartbeat_unix": now_unix,
        "expires_at_unix": now_unix + int(lease_seconds),
        "worker_info": worker_info,
    }
    save_lease(leases_root=leases_root, lease=lease)
    for old in same_worker_leases:
        old_lease_id = str(old.get("lease_id") or "").strip()
        if old_lease_id and old_lease_id != str(lease["lease_id"]):
            lease_path(leases_root=leases_root, lease_id=old_lease_id).unlink(missing_ok=True)
    return lease
