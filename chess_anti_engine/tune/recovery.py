from __future__ import annotations

import json
import math
import re
from pathlib import Path

from chess_anti_engine.tune._utils import stable_seed_u32


def _trial_index_from_name(trial_dir: Path) -> int:
    """Best-effort numeric trial index from Tune trial directory name."""
    name = str(Path(trial_dir).name)
    # Expected: train_trial_<runid>_<trial_idx>_<params>_<timestamp>
    # Use the explicit <trial_idx> field, not trailing timestamp digits.
    m = re.match(r"^train_trial_[^_]+_(\d+)_", name)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _select_salvage_seed_slot(
    *,
    seed_pool_dir: Path,
    trial_dir: Path,
    trial_id: str,
) -> tuple[Path | None, int, int]:
    """Select deterministic salvage seed dir for a fresh trial."""
    manifest_path = Path(seed_pool_dir) / "manifest.json"
    if not manifest_path.exists():
        return None, -1, 0

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None, -1, 0
    if not isinstance(manifest, dict):
        return None, -1, 0

    entries_raw = manifest.get("entries")
    if not isinstance(entries_raw, list):
        return None, -1, 0

    slots: list[tuple[int, Path]] = []
    for i, e in enumerate(entries_raw):
        if not isinstance(e, dict):
            continue
        slot_v = e.get("slot", i)
        try:
            slot_i = int(slot_v) if isinstance(slot_v, (int, float)) else int(i)
        except Exception:
            slot_i = int(i)

        seed_rel = e.get("seed_dir")
        if isinstance(seed_rel, str) and seed_rel.strip():
            sd = Path(seed_pool_dir) / seed_rel
        else:
            sd = Path(seed_pool_dir) / "seeds" / f"slot_{slot_i:03d}"
        if sd.is_dir():
            slots.append((slot_i, sd))

    if not slots:
        return None, -1, 0

    slots.sort(key=lambda x: x[0])
    num_slots = len(slots)
    trial_idx = _trial_index_from_name(trial_dir)
    if trial_idx < 0:
        trial_idx = int(stable_seed_u32("salvage", trial_id))
    pick_idx = int(trial_idx) % int(num_slots)
    picked_slot, picked_dir = slots[pick_idx]
    return picked_dir, int(picked_slot), int(num_slots)


def _load_salvage_manifest_entry(
    *,
    seed_pool_dir: Path,
    slot: int,
) -> dict | None:
    manifest_path = Path(seed_pool_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(manifest, dict):
        return None
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return None

    slot_i = int(slot)
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        raw = e.get("slot", i)
        try:
            cur_slot = int(raw) if isinstance(raw, (int, float)) else int(i)
        except Exception:
            cur_slot = int(i)
        if cur_slot == slot_i:
            return e
    return None


def _merge_pid_state_from_result_row(
    *,
    pid_state: dict | None,
    result_row: dict | None,
) -> tuple[dict | None, list[str]]:
    if not isinstance(result_row, dict):
        return pid_state, []

    out: dict = dict(pid_state) if isinstance(pid_state, dict) else {}
    applied: list[str] = []

    def _set(dst_key: str, src_keys: tuple[str, ...]) -> None:
        nonlocal out
        for sk in src_keys:
            v = result_row.get(sk)
            if isinstance(v, (int, float)):
                fv = float(v)
                if math.isfinite(fv):
                    out[dst_key] = fv
                    applied.append(f"{dst_key}<-{sk}")
                    return

    _set("random_move_prob", ("random_move_prob_next", "random_move_prob"))
    _set("nodes", ("sf_nodes_next", "sf_nodes"))
    _set("skill_level", ("skill_level_next", "skill_level"))
    _set("ema_winrate", ("pid_ema_winrate",))
    return (out if out else pid_state), applied
