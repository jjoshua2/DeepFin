from __future__ import annotations

# Optional dependency module (Ray Tune). Kept import-light so the core package
# works without installing `.[tune]`.

from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

import math

from chess_anti_engine.model import ModelConfig, build_model, zero_policy_head_parameters_
from chess_anti_engine.moves.encode import POLICY_SIZE
from chess_anti_engine.replay import DiskReplayBuffer, ReplayBuffer
from chess_anti_engine.replay.shard import load_npz
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.budget import progressive_mcts_simulations
from chess_anti_engine.stockfish import DifficultyPID, StockfishPool, StockfishUCI
from chess_anti_engine.train import Trainer
from chess_anti_engine.version import PACKAGE_VERSION, PROTOCOL_VERSION


def _stable_seed_u32(*parts: object) -> int:
    """Deterministic 32-bit seed from arbitrary parts."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "little") & 0xFFFFFFFF


def _count_jsonl_rows(path: Path) -> int:
    """Count non-empty rows in a JSONL file (best effort)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return 0


def _latest_trial_result_row(trial_dir: Path) -> dict | None:
    """Read the latest result row from Ray Tune's JSONL stream."""
    result_path = Path(trial_dir) / "result.json"
    if not result_path.exists():
        return None

    last_row: dict | None = None
    try:
        with result_path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except Exception:
                    continue
                if isinstance(row, dict):
                    last_row = row
    except Exception:
        return None

    if not isinstance(last_row, dict):
        return None
    return last_row


def _all_trial_result_rows(trial_dir: Path) -> list[dict]:
    """Read all result rows from Ray Tune's JSONL stream."""
    result_path = Path(trial_dir) / "result.json"
    if not result_path.exists():
        return []

    rows: list[dict] = []
    try:
        with result_path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return []
    return rows


def _metric_from_result_row(row: dict) -> float | None:
    # Use the PB2 optimization metric directly.
    v = row.get("opponent_strength")
    if isinstance(v, (int, float)):
        fv = float(v)
        if math.isfinite(fv):
            return fv
    return None


def _to_nonnegative_int(v: object, default: int = 0) -> int:
    try:
        iv = int(v)  # type: ignore[arg-type]
        return iv if iv >= 0 else default
    except Exception:
        return default


def _latest_trial_snapshot(trial_dir: Path) -> dict | None:
    """Best-effort latest trial snapshot with metric + generation size."""
    row = _latest_trial_result_row(trial_dir)
    if row is None:
        return None
    metric = _metric_from_result_row(row)
    if metric is None:
        return None
    iter_idx = _to_nonnegative_int(row.get("training_iteration", row.get("iter", -1)), default=-1)
    positions_added = _to_nonnegative_int(row.get("positions_added", 0), default=0)
    return {
        "trial_dir": Path(trial_dir),
        "metric": float(metric),
        "iter": int(iter_idx),
        "positions_added": int(positions_added),
    }


def _all_trial_snapshots(trial_dir: Path) -> list[dict]:
    """Best-effort per-iteration snapshots with metric + generation size."""
    snapshots: list[dict] = []
    seen_iters: set[int] = set()
    for row in _all_trial_result_rows(trial_dir):
        metric = _metric_from_result_row(row)
        if metric is None:
            continue
        iter_idx = _to_nonnegative_int(row.get("training_iteration", row.get("iter", -1)), default=-1)
        if iter_idx < 0 or iter_idx in seen_iters:
            continue
        seen_iters.add(iter_idx)
        positions_added = _to_nonnegative_int(row.get("positions_added", 0), default=0)
        snapshots.append(
            {
                "trial_dir": Path(trial_dir),
                "metric": float(metric),
                "iter": int(iter_idx),
                "positions_added": int(positions_added),
            }
        )
    snapshots.sort(key=lambda s: int(s["iter"]))
    return snapshots


def _estimate_recent_shard_count(
    *,
    positions_added: int,
    shard_size: int,
    holdout_fraction: float,
) -> int:
    """Estimate shard count for the latest generation from positions_added."""
    pa = max(0, int(positions_added))
    if pa <= 0:
        return 0
    ss = max(1, int(shard_size))
    # Positions entering train replay are roughly (1-holdout_fraction) of total.
    hf = max(0.0, min(0.99, float(holdout_fraction)))
    train_positions = max(1, int(math.ceil(float(pa) * (1.0 - hf))))
    return max(1, int(math.ceil(float(train_positions) / float(ss))))


def _load_shard_samples_with_retry(
    shard_path: Path,
    *,
    retries: int = 4,
    sleep_s: float = 0.15,
) -> list | None:
    """Best-effort shard read with short retries for in-flight writes."""
    attempts = max(1, int(retries))
    for i in range(attempts):
        try:
            samples, _meta = load_npz(shard_path)
            return samples
        except Exception:
            if i + 1 < attempts and sleep_s > 0:
                time.sleep(float(sleep_s))
    return None


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
        trial_idx = int(_stable_seed_u32("salvage", trial_id))
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


def _resolve_pause_marker_path(*, config: dict, trial_dir: Path) -> Path:
    raw = config.get("pause_file")
    if isinstance(raw, str) and raw.strip():
        p = Path(raw.strip())
        if not p.is_absolute():
            p = trial_dir.parent / p
        return p
    return trial_dir.parent / "pause.txt"


def _wait_if_paused(
    *,
    pause_marker_path: Path,
    poll_seconds: int,
    trial_id: str,
    iteration: int,
) -> None:
    poll_s = max(1, int(poll_seconds))
    announced = False
    while pause_marker_path.exists():
        if not announced:
            print(
                f"[trial] pause marker detected: {pause_marker_path} "
                f"(trial={trial_id}, next_iter={iteration})"
            )
            announced = True
        time.sleep(float(poll_s))
    if announced:
        print(
            f"[trial] pause marker cleared: {pause_marker_path} "
            f"(trial={trial_id}, resuming_iter={iteration})"
        )


def _select_top_trial_snapshots(
    *,
    recipient_trial_dir: Path,
    top_k_trials: int,
    within_best_frac: float,
    min_metric: float,
) -> list[dict]:
    """Pick top sibling trials constrained by best-relative threshold."""
    parent = Path(recipient_trial_dir).parent
    if not parent.is_dir():
        return []

    try:
        recipient_resolved = recipient_trial_dir.resolve()
    except Exception:
        recipient_resolved = recipient_trial_dir

    snapshots: list[dict] = []
    for td in sorted(parent.glob("train_trial_*")):
        if not td.is_dir():
            continue
        try:
            td_resolved = td.resolve()
        except Exception:
            td_resolved = td
        if td_resolved == recipient_resolved:
            continue
        snap = _latest_trial_snapshot(td)
        if snap is None:
            continue
        snapshots.append(snap)

    if not snapshots:
        return []

    best_metric = max(float(s["metric"]) for s in snapshots)
    frac = max(0.0, min(1.0, float(within_best_frac)))
    cutoff = max(float(min_metric), best_metric * (1.0 - frac))
    eligible = [s for s in snapshots if float(s["metric"]) >= cutoff]
    eligible.sort(key=lambda s: float(s["metric"]), reverse=True)

    if top_k_trials > 0:
        eligible = eligible[:top_k_trials]
    return eligible


def _refresh_replay_shards_on_exploit(
    *,
    replay_shard_dir: Path,
    recipient_trial_dir: Path,
    donor_trial_dir: Path | None,
    keep_recent_fraction: float,
    keep_older_fraction: float,
    donor_shards: int,
    donor_skip_newest: int,
    shard_size: int,
    holdout_fraction: float,
) -> dict[str, int]:
    """Refresh recipient replay after PB2 exploit.

    Strategy:
    - Keep a small fraction of recipient's *most recent* local generation
      (likely poor just before exploit) while retaining more of older history.
    - Copy a configurable slice of the donor's recent replay shards so exploit
      recipients actually inherit some stronger data instead of only weights.
    """
    replay_shard_dir = Path(replay_shard_dir)
    replay_shard_dir.mkdir(parents=True, exist_ok=True)

    keep_recent_fraction = max(0.0, min(1.0, float(keep_recent_fraction)))
    keep_older_fraction = max(0.0, min(1.0, float(keep_older_fraction)))
    donor_shards = max(0, int(donor_shards))
    donor_skip_newest = max(0, int(donor_skip_newest))
    shard_size = max(1, int(shard_size))
    holdout_fraction = max(0.0, min(0.99, float(holdout_fraction)))

    summary = {
        "local_before": 0,
        "local_deleted": 0,
        "local_recent_deleted": 0,
        "local_older_deleted": 0,
        "local_after_keep": 0,
        "local_final": 0,
        "donor_available": 0,
        "donor_selected": 0,
        "donor_copied": 0,
    }

    local_shards = sorted(replay_shard_dir.glob("shard_*.npz"))
    summary["local_before"] = len(local_shards)

    recipient_snap = _latest_trial_snapshot(recipient_trial_dir)
    recipient_recent_shards = 0
    if recipient_snap is not None:
        recipient_recent_shards = _estimate_recent_shard_count(
            positions_added=int(recipient_snap.get("positions_added", 0)),
            shard_size=shard_size,
            holdout_fraction=holdout_fraction,
        )
    recipient_recent_shards = max(0, min(len(local_shards), int(recipient_recent_shards)))

    if local_shards:
        if recipient_recent_shards > 0:
            local_recent = local_shards[-recipient_recent_shards:]
            local_older = local_shards[:-recipient_recent_shards]
        else:
            local_recent = []
            local_older = local_shards

        keep_recent_n = int(math.ceil(len(local_recent) * keep_recent_fraction))
        keep_older_n = int(math.ceil(len(local_older) * keep_older_fraction))
        keep_recent = set(local_recent[-keep_recent_n:]) if keep_recent_n > 0 else set()
        keep_older = set(local_older[-keep_older_n:]) if keep_older_n > 0 else set()
        keep_set = keep_recent | keep_older

        # Keep at least one local shard to avoid an empty replay on corner cases.
        if not keep_set and local_shards:
            keep_set.add(local_shards[-1])

        for sp in local_shards:
            if sp in keep_set:
                continue
            is_recent = sp in local_recent
            try:
                sp.unlink(missing_ok=True)
                summary["local_deleted"] += 1
                if is_recent:
                    summary["local_recent_deleted"] += 1
                else:
                    summary["local_older_deleted"] += 1
            except Exception:
                pass

    local_after_keep = sorted(replay_shard_dir.glob("shard_*.npz"))
    summary["local_after_keep"] = len(local_after_keep)

    donor_dir = Path(donor_trial_dir) if donor_trial_dir is not None else None
    donor_replay_dir = (donor_dir / "replay_shards") if donor_dir is not None else None
    if donor_shards > 0 and donor_replay_dir is not None and donor_replay_dir.is_dir():
        donor_files = sorted(donor_replay_dir.glob("shard_*.npz"))
        summary["donor_available"] = len(donor_files)
        if donor_skip_newest > 0 and len(donor_files) > donor_skip_newest:
            donor_files = donor_files[:-donor_skip_newest]
        elif donor_skip_newest > 0:
            donor_files = []
        if donor_files:
            donor_files = donor_files[-donor_shards:]
            summary["donor_selected"] = len(donor_files)

            existing = sorted(replay_shard_dir.glob("shard_*.npz"))
            next_idx = 0
            for sp in existing:
                try:
                    next_idx = max(next_idx, int(sp.stem.split("_")[1]) + 1)
                except Exception:
                    continue

            for src in donor_files:
                dst = replay_shard_dir / f"shard_{next_idx:06d}.npz"
                next_idx += 1
                try:
                    shutil.copy2(str(src), str(dst))
                    summary["donor_copied"] += 1
                except Exception:
                    pass

    summary["local_final"] = len(list(replay_shard_dir.glob("shard_*.npz")))
    return summary


def _share_top_replay_each_iteration(
    *,
    recipient_trial_dir: Path,
    replay_shard_dir: Path,
    buf: DiskReplayBuffer,
    top_k_trials: int,
    within_best_frac: float,
    min_metric: float,
    source_skip_newest: int,
    shard_size: int,
    holdout_fraction: float,
    max_unseen_iters_per_source: int,
) -> dict[str, int]:
    """Ingest recent unseen top-trial generations into the live replay buffer."""

    source_skip_newest = max(0, int(source_skip_newest))
    shard_size = max(1, int(shard_size))
    holdout_fraction = max(0.0, min(0.99, float(holdout_fraction)))
    max_unseen_iters_per_source = max(1, int(max_unseen_iters_per_source))

    source_snaps = _select_top_trial_snapshots(
        recipient_trial_dir=recipient_trial_dir,
        top_k_trials=int(top_k_trials),
        within_best_frac=float(within_best_frac),
        min_metric=float(min_metric),
    )

    summary = {
        "source_trials_selected": len(source_snaps),
        "source_trials_ingested": 0,
        "source_trials_skipped_repeat": 0,
        "source_shards_loaded": 0,
        "source_samples_ingested": 0,
    }

    if not source_snaps:
        return summary

    import_state_path = Path(replay_shard_dir) / "_import_state.json"
    import_state: dict[str, int] = {}
    if import_state_path.exists():
        try:
            obj = json.loads(import_state_path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                import_state = {
                    str(k): int(v) for k, v in obj.items() if isinstance(v, (int, float))
                }
        except Exception:
            import_state = {}

    for snap in source_snaps:
        td = Path(snap["trial_dir"])
        source_key = str(td.resolve()) if td.exists() else str(td)
        last_imported_iter = int(import_state.get(source_key, -1))

        unseen_snaps = [
            s for s in _all_trial_snapshots(td)
            if int(s.get("iter", -1)) > last_imported_iter
        ]
        if not unseen_snaps:
            summary["source_trials_skipped_repeat"] += 1
            continue
        if len(unseen_snaps) > max_unseen_iters_per_source:
            unseen_snaps = unseen_snaps[-max_unseen_iters_per_source:]

        src_dir = td / "replay_shards"
        if not src_dir.is_dir():
            continue
        shards = sorted(src_dir.glob("shard_*.npz"))
        if source_skip_newest > 0:
            if len(shards) > source_skip_newest:
                shards = shards[:-source_skip_newest]
            else:
                shards = []
        if not shards:
            continue

        ingested_any = False
        imported_iters: list[int] = []
        shard_end = len(shards)

        # Import the newest unseen generations, but keep per-generation shard
        # slices distinct so we can backfill missed iterations instead of only
        # ever seeing the latest sibling generation.
        for unseen_snap in reversed(unseen_snaps):
            n_recent = _estimate_recent_shard_count(
                positions_added=int(unseen_snap.get("positions_added", 0)),
                shard_size=shard_size,
                holdout_fraction=holdout_fraction,
            )
            if n_recent <= 0:
                continue
            shard_start = max(0, shard_end - n_recent)
            iter_shards = shards[shard_start:shard_end]
            shard_end = shard_start
            if not iter_shards:
                continue
            for sp in iter_shards:
                samples = _load_shard_samples_with_retry(sp, retries=5, sleep_s=0.12)
                if samples is None:
                    continue
                if not samples:
                    continue
                buf.add_many(samples)
                summary["source_shards_loaded"] += 1
                summary["source_samples_ingested"] += int(len(samples))
                ingested_any = True
            if ingested_any:
                imported_iters.append(int(unseen_snap.get("iter", -1)))

        if ingested_any:
            summary["source_trials_ingested"] += 1
            if imported_iters:
                import_state[source_key] = max(imported_iters)

    try:
        import_state_path.write_text(
            json.dumps(import_state, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    return summary


def _opponent_strength(
    *,
    random_move_prob: float,
    sf_nodes: int,
    skill_level: int,
    ema_winrate: float,
    min_nodes: int,
    max_nodes: int,
) -> float:
    """Composite metric capturing full difficulty progression.

    Returns a single scalar (higher = harder opponent = better model):
      Stage 1 (score   0-100): random_move_prob 1.0 → 0.0
      Stage 2 (score 100-200): sf_nodes min → max (log-scaled)
      Stage 3 (score 200-400): skill_level 0 → 20

    The PID controller holds winrate at ~0.53, so the only differentiator
    between trials is how hard an opponent they can maintain that against.
    """
    rand_prob = float(random_move_prob)
    nodes = int(sf_nodes)
    skill = int(skill_level)

    min_nodes = int(min_nodes)
    max_nodes = int(max_nodes)

    # Stage 1: random_move_prob 1.0→0.0 maps to score 0→100
    stage1 = (1.0 - rand_prob) * 100.0

    # Stage 2: sf_nodes on log scale, maps to score 100→200
    if min_nodes < max_nodes and nodes > 0:
        log_frac = (math.log(max(nodes, min_nodes)) - math.log(max(1, min_nodes))) / (
            math.log(max(1, max_nodes)) - math.log(max(1, min_nodes))
        )
        log_frac = max(0.0, min(1.0, log_frac))
    else:
        log_frac = 0.0
    stage2 = log_frac * 100.0

    # Stage 3: skill_level 0→20 maps to score 200→400
    stage3 = (float(skill) / 20.0) * 200.0

    # Winrate tiebreaker: when all trials are at the same difficulty
    # (e.g. everyone stuck at random_move_prob=1.0), the winrate term
    # differentiates who is closest to breaking through the PID threshold.
    # Range 0-10, negligible once opponent_strength starts climbing (0-400).
    winrate_bonus = float(ema_winrate) * 10.0

    return stage1 + stage2 + stage3 + winrate_bonus



def _gate_check(
    model: torch.nn.Module,
    *,
    device: str,
    rng: np.random.Generator,
    sf: object,
    gate_games: int,
    opponent_random_move_prob: float,
    config: dict,
) -> tuple[float, int, int, int]:
    """Play gate games to measure winrate. Returns (winrate, W, D, L)."""
    _gate_samples, gate_stats = play_batch(
        model,
        device=device,
        rng=rng,
        stockfish=sf,
        games=gate_games,
        opponent_random_move_prob=opponent_random_move_prob,
        opponent_topk_stage_end=float(config.get("sf_pid_random_move_stage_end", 0.5)),
        temperature=0.3,  # Low temperature for gating (exploit, don't explore)
        temperature_drop_plies=0,
        temperature_after=0.0,
        temperature_decay_start_move=10,
        temperature_decay_moves=30,
        temperature_endgame=0.1,
        max_plies=int(config.get("max_plies", 120)),
        mcts_simulations=int(config.get("gate_mcts_sims", 1)),  # 1 = raw policy + value
        mcts_type=str(config.get("mcts", "puct")),
        playout_cap_fraction=1.0,
        fast_simulations=0,
        sf_policy_temp=float(config.get("sf_policy_temp", 0.25)),
        sf_policy_label_smooth=float(config.get("sf_policy_label_smooth", 0.05)),
        timeout_adjudication_threshold=float(config.get("timeout_adjudication_threshold", 0.90)),
        volatility_source=str(config.get("volatility_source", "raw")),
        opening_book_path=config.get("opening_book_path"),
        opening_book_max_plies=int(config.get("opening_book_max_plies", 4)),
        opening_book_max_games=int(config.get("opening_book_max_games", 200_000)),
        opening_book_prob=float(config.get("opening_book_prob", 1.0)),
        random_start_plies=int(config.get("random_start_plies", 0)),
        fpu_reduction=float(config.get("fpu_reduction", 1.2)),
        fpu_at_root=float(config.get("fpu_at_root", 1.0)),
    )
    w, d, l = gate_stats.w, gate_stats.d, gate_stats.l
    total = max(1, w + d + l)
    winrate = (w + 0.5 * d) / total
    return winrate, w, d, l


def _prune_trial_checkpoints(*, trial_dir: Path, keep_last: int) -> None:
    """Best-effort deletion of old checkpoint_* dirs inside a Tune trial.

    This complements Ray's `CheckpointConfig(num_to_keep=...)`.
    In particular, it helps when resuming an older experiment whose RunConfig did
    not have checkpoint retention enabled.
    """

    import shutil

    keep_last = int(keep_last)
    if keep_last <= 0:
        return

    ckpts = sorted(
        [p for p in trial_dir.glob("checkpoint_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if len(ckpts) <= keep_last:
        return

    for p in ckpts[:-keep_last]:
        shutil.rmtree(p, ignore_errors=True)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{int(time.time() * 1000)}")
    try:
        tmp.write_text(text, encoding=encoding)
        tmp.replace(path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _trial_server_dirs(*, server_root: Path, trial_id: str) -> dict[str, Path]:
    trial_root = Path(server_root) / "trials" / str(trial_id)
    return {
        "trial_root": trial_root,
        "publish_dir": trial_root / "publish",
        "inbox_dir": trial_root / "inbox",
        "processed_dir": trial_root / "processed",
    }


def _publish_distributed_trial_state(
    *,
    trainer: Trainer,
    config: dict,
    model_cfg: ModelConfig,
    server_root: Path,
    trial_id: str,
    training_iteration: int,
    trainer_step: int,
    sf_nodes: int,
    random_move_prob: float,
    skill_level: int,
    mcts_simulations: int,
) -> str:
    dirs = _trial_server_dirs(server_root=server_root, trial_id=trial_id)
    publish_dir = dirs["publish_dir"]
    publish_dir.mkdir(parents=True, exist_ok=True)

    model_path = publish_dir / "latest_model.pt"
    trainer.export_swa(model_path)
    model_sha = _sha256_file(model_path)
    api_prefix = f"/v1/trials/{trial_id}"
    published_stockfish_path: Path | None = None
    published_worker_wheel_path: Path | None = None

    stockfish_raw = str(config.get("stockfish_path", "")).strip()
    if stockfish_raw:
        stockfish_src = Path(stockfish_raw)
        if stockfish_src.exists() and stockfish_src.is_file():
            dst = publish_dir / ("stockfish" + stockfish_src.suffix)
            try:
                shutil.copy2(str(stockfish_src), str(dst))
                published_stockfish_path = dst
            except Exception:
                published_stockfish_path = None

    worker_wheel_raw = str(config.get("worker_wheel_path", "")).strip()
    if worker_wheel_raw:
        worker_wheel_src = Path(worker_wheel_raw)
        if worker_wheel_src.exists() and worker_wheel_src.is_file():
            dst = publish_dir / "worker.whl"
            try:
                shutil.copy2(str(worker_wheel_src), str(dst))
                published_worker_wheel_path = dst
            except Exception:
                published_worker_wheel_path = None

    recommended_worker = {
        "games_per_batch": int(config.get("selfplay_batch", 4)),
        "max_plies": int(config.get("max_plies", 120)),
        "mcts": str(config.get("mcts", "puct")),
        "mcts_simulations": int(mcts_simulations),
        "playout_cap_fraction": float(config.get("playout_cap_fraction", 0.25)),
        "fast_simulations": int(config.get("fast_simulations", 8)),
        "opening_book_max_plies": int(config.get("opening_book_max_plies", 4)),
        "opening_book_max_games": int(config.get("opening_book_max_games", 200_000)),
        "opening_book_prob": float(config.get("opening_book_prob", 1.0)),
        "random_start_plies": int(config.get("random_start_plies", 0)),
        "selfplay_fraction": float(config.get("selfplay_fraction", 0.0)),
        "sf_nodes": int(sf_nodes),
        "sf_multipv": int(config.get("sf_multipv", 1)),
        "sf_policy_temp": float(config.get("sf_policy_temp", 0.25)),
        "sf_policy_label_smooth": float(config.get("sf_policy_label_smooth", 0.05)),
        "sf_skill_level": int(skill_level),
        "opponent_random_move_prob": float(random_move_prob),
        "opponent_topk_stage_end": float(config.get("sf_pid_random_move_stage_end", 0.5)),
        "temperature": float(config.get("temperature", 1.0)),
        "temperature_decay_start_move": int(config.get("temperature_decay_start_move", 20)),
        "temperature_decay_moves": int(config.get("temperature_decay_moves", 60)),
        "temperature_endgame": float(config.get("temperature_endgame", 0.6)),
        "timeout_adjudication_threshold": float(config.get("timeout_adjudication_threshold", 0.90)),
    }

    manifest: dict[str, object] = {
        "server_time_unix": int(time.time()),
        "protocol_version": int(PROTOCOL_VERSION),
        "server_version": str(PACKAGE_VERSION),
        "min_worker_version": str(PACKAGE_VERSION),
        "trial_id": str(trial_id),
        "training_iteration": int(training_iteration),
        "trainer_step": int(trainer_step),
        "task": {"type": "selfplay"},
        "recommended_worker": recommended_worker,
        "encoding": {
            "input_planes": 146,
            "policy_size": int(POLICY_SIZE),
            "policy_encoding": "lc0_4672",
        },
        "model": {
            "sha256": str(model_sha),
            "endpoint": api_prefix + "/model",
            "filename": "latest_model.pt",
            "format": "torch_state_dict",
        },
        "model_config": {
            "kind": str(model_cfg.kind),
            "embed_dim": int(model_cfg.embed_dim),
            "num_layers": int(model_cfg.num_layers),
            "num_heads": int(model_cfg.num_heads),
            "ffn_mult": int(model_cfg.ffn_mult),
            "use_smolgen": bool(model_cfg.use_smolgen),
            "use_nla": bool(model_cfg.use_nla),
            "use_qk_rmsnorm": bool(getattr(model_cfg, "use_qk_rmsnorm", False)),
            "gradient_checkpointing": bool(model_cfg.use_gradient_checkpointing),
        },
    }

    opening_book_path = config.get("opening_book_path")
    if isinstance(opening_book_path, str) and opening_book_path.strip():
        p = Path(opening_book_path.strip())
        if p.exists():
            manifest["opening_book"] = {
                "endpoint": "/v1/opening_book",
                "filename": p.name,
                "sha256": _sha256_file(p),
            }

    if published_stockfish_path is not None and published_stockfish_path.exists():
        manifest["stockfish"] = {
            "endpoint": api_prefix + "/stockfish",
            "filename": published_stockfish_path.name,
            "sha256": _sha256_file(published_stockfish_path),
        }

    if published_worker_wheel_path is not None and published_worker_wheel_path.exists():
        manifest["worker_wheel"] = {
            "endpoint": api_prefix + "/worker_wheel",
            "filename": published_worker_wheel_path.name,
            "sha256": _sha256_file(published_worker_wheel_path),
            "version": str(PACKAGE_VERSION),
        }

    _atomic_write_text(
        publish_dir / "manifest.json",
        json.dumps(manifest, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return model_sha


def _launch_distributed_worker(
    *,
    config: dict,
    trial_dir: Path,
    trial_id: str,
    worker_index: int,
) -> subprocess.Popen[bytes]:
    worker_root = trial_dir / "distributed_workers" / f"worker_{worker_index:02d}"
    worker_root.mkdir(parents=True, exist_ok=True)
    worker_log = worker_root / "worker.log"
    worker_out = worker_root / "worker.out"
    device = str(config.get("distributed_worker_device") or config.get("device", "cpu"))
    shared_cache_raw = str(config.get("distributed_worker_shared_cache_dir") or "").strip()
    if shared_cache_raw:
        shared_cache_root = Path(shared_cache_raw).expanduser()
    else:
        shared_cache_root = Path(str(config["distributed_server_root"])) / "worker_cache"
    shared_cache_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "chess_anti_engine.worker",
        "--server-url",
        str(config["distributed_server_url"]),
        "--username",
        str(config["distributed_worker_username"]),
        "--password-file",
        str(config["distributed_worker_password_file"]),
        "--stockfish-path",
        str(config["stockfish_path"]),
        "--work-dir",
        str(worker_root),
        "--shared-cache-dir",
        str(shared_cache_root),
        "--device",
        device,
        "--sf-workers",
        str(int(config.get("distributed_worker_sf_workers", 1))),
        "--poll-seconds",
        str(float(config.get("distributed_worker_poll_seconds", 1.0))),
        "--seed",
        str(_stable_seed_u32("dist-worker", trial_id, worker_index, config.get("seed", 0))),
        "--log-file",
        str(worker_log),
        "--log-level",
        "info",
    ]

    if bool(config.get("distributed_worker_auto_tune", False)):
        cmd.extend(
            [
                "--auto-tune",
                "--target-batch-seconds",
                str(float(config.get("distributed_worker_target_batch_seconds", 30.0))),
                "--min-games-per-batch",
                str(int(config.get("distributed_worker_min_games_per_batch", 1))),
                "--max-games-per-batch",
                str(int(config.get("distributed_worker_max_games_per_batch", 64))),
            ]
        )

    out_fh = worker_out.open("ab")
    try:
        return subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=out_fh,
            stderr=subprocess.STDOUT,
        )
    finally:
        out_fh.close()


def _stop_worker_processes(procs: list[subprocess.Popen[bytes]]) -> None:
    for proc in procs:
        if proc.poll() is not None:
            continue
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except Exception:
                pass


def _ensure_distributed_workers(
    *,
    config: dict,
    trial_dir: Path,
    trial_id: str,
    procs: list[subprocess.Popen[bytes]],
) -> list[subprocess.Popen[bytes]]:
    want = max(0, int(config.get("distributed_workers_per_trial", 0)))
    out = list(procs)
    for idx in range(want):
        if idx < len(out) and out[idx].poll() is None:
            continue
        if idx < len(out) and out[idx].poll() is not None:
            print(
                f"[trial] restarting distributed worker idx={idx} "
                f"exit_code={out[idx].returncode} trial={trial_id}"
            )
            out[idx] = _launch_distributed_worker(
                config=config,
                trial_dir=trial_dir,
                trial_id=trial_id,
                worker_index=idx,
            )
        elif idx >= len(out):
            out.append(
                _launch_distributed_worker(
                    config=config,
                    trial_dir=trial_dir,
                    trial_id=trial_id,
                    worker_index=idx,
                )
            )
    return out[:want]


def _ingest_distributed_selfplay(
    *,
    buf: DiskReplayBuffer,
    holdout_buf: ReplayBuffer,
    holdout_frac: float,
    holdout_frozen: bool,
    inbox_dir: Path,
    processed_dir: Path,
    target_games: int,
    target_model_sha: str,
    wait_timeout_s: float,
    poll_seconds: float,
    rng: np.random.Generator,
) -> dict[str, int]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    target_games = max(1, int(target_games))
    deadline = time.time() + float(wait_timeout_s)
    summary = {
        "matching_games": 0,
        "matching_positions": 0,
        "matching_w": 0,
        "matching_d": 0,
        "matching_l": 0,
        "matching_total_game_plies": 0,
        "matching_timeout_games": 0,
        "matching_total_draw_games": 0,
        "positions_replay_added": 0,
        "stale_games": 0,
        "stale_positions": 0,
        "matching_shards": 0,
        "stale_shards": 0,
    }

    while summary["matching_games"] < target_games:
        shard_paths = sorted(inbox_dir.glob("*/*.npz"))
        if not shard_paths:
            if time.time() >= deadline:
                raise RuntimeError(
                    f"timed out waiting for distributed selfplay shards for model {target_model_sha[:12]}"
                )
            time.sleep(float(poll_seconds))
            continue

        processed_any = False
        for sp in shard_paths:
            processed_any = True
            rel = sp.relative_to(inbox_dir)
            out = processed_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                samples, meta = load_npz(sp)
            except Exception:
                bad = processed_dir / "bad" / rel.name
                bad.parent.mkdir(parents=True, exist_ok=True)
                try:
                    sp.replace(bad)
                except Exception:
                    sp.unlink(missing_ok=True)
                continue

            model_sha = str(meta.get("model_sha256") or "")
            wins = int(meta.get("wins", 0) or 0)
            draws = int(meta.get("draws", 0) or 0)
            losses = int(meta.get("losses", 0) or 0)
            games = int(meta.get("games", wins + draws + losses) or 0)
            positions = int(meta.get("positions", len(samples)) or len(samples))
            total_game_plies = int(meta.get("total_game_plies", 0) or 0)
            timeout_games = int(meta.get("timeout_games", 0) or 0)
            total_draw_games = int(meta.get("total_draw_games", draws) or draws)

            train_samples: list = []
            for s in samples:
                if holdout_frac > 0.0 and (not holdout_frozen) and (rng.random() < holdout_frac):
                    holdout_buf.add_many([s])
                else:
                    train_samples.append(s)
            if train_samples:
                buf.add_many(train_samples)
            summary["positions_replay_added"] += int(positions)

            if model_sha == target_model_sha:
                summary["matching_games"] += int(games)
                summary["matching_positions"] += int(positions)
                summary["matching_w"] += int(wins)
                summary["matching_d"] += int(draws)
                summary["matching_l"] += int(losses)
                summary["matching_total_game_plies"] += int(total_game_plies)
                summary["matching_timeout_games"] += int(timeout_games)
                summary["matching_total_draw_games"] += int(total_draw_games)
                summary["matching_shards"] += 1
            else:
                summary["stale_games"] += int(games)
                summary["stale_positions"] += int(positions)
                summary["stale_shards"] += 1

            try:
                sp.replace(out)
            except Exception:
                sp.unlink(missing_ok=True)

            if summary["matching_games"] >= target_games:
                break

        if not processed_any:
            if time.time() >= deadline:
                raise RuntimeError(
                    f"timed out waiting for distributed selfplay shards for model {target_model_sha[:12]}"
                )
            time.sleep(float(poll_seconds))

    return summary


def _games_per_iter_for_iteration(config: dict, iteration_idx: int) -> int:
    target = max(1, int(config.get("games_per_iter", 1)))
    start = int(config.get("games_per_iter_start", target))
    ramp_iters = max(0, int(config.get("games_per_iter_ramp_iters", 0)))

    if ramp_iters <= 0 or iteration_idx >= ramp_iters:
        return int(target)

    frac = float(max(0, iteration_idx - 1)) / float(ramp_iters)
    value = float(start) + (float(target) - float(start)) * frac
    return max(1, int(round(value)))


def train_trial(config: dict):
    """Ray Tune trainable.

    Reports metrics per outer-loop iteration. Supports Ray AIR checkpoint restore.
    """

    from ray.air import session
    from ray.train import Checkpoint

    base_seed = int(config.get("seed", 0))
    trial_id = str(getattr(session, "get_trial_id", lambda: "trial")())
    trial_seed = _stable_seed_u32(base_seed, trial_id)
    active_seed = int(trial_seed)
    rng = np.random.default_rng(active_seed)
    torch.manual_seed(active_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(active_seed)

    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model_cfg = ModelConfig(
        kind=str(config.get("model", "transformer")),
        embed_dim=int(config.get("embed_dim", 256)),
        num_layers=int(config.get("num_layers", 6)),
        num_heads=int(config.get("num_heads", 8)),
        ffn_mult=int(config.get("ffn_mult", 2)),
        use_smolgen=bool(config.get("use_smolgen", True)),
        use_nla=bool(config.get("use_nla", False)),
        use_gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
    )
    model = build_model(model_cfg)

    # Use Ray-provided trial directory for ALL per-trial state (checkpoints,
    # replay shards, gate state, best model, TensorBoard logs).
    # IMPORTANT: Do NOT use config["work_dir"] here — it points to the shared
    # runs/pbt2_small/ directory. Using it caused all 10 trials to write
    # checkpoints to the same directory, making PB2 unable to clone checkpoints
    # ("no checkpoint for trial X. Skip exploit.").
    trial_dir = Path(session.get_trial_dir())
    work_dir = trial_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    # Gate match counter (so we can log gate scalars against match #, not iteration).
    gate_state_path = work_dir / "gate_state.json"
    gate_match_idx = 0
    if gate_state_path.exists():
        try:
            d = json.loads(gate_state_path.read_text(encoding="utf-8"))
            gate_match_idx = int(d.get("matches", 0))
        except Exception:
            gate_match_idx = 0

    # Best-model tracking (per trial)
    best_state_path = work_dir / "best.json"
    best_dir = work_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    if best_state_path.exists():
        try:
            d = json.loads(best_state_path.read_text(encoding="utf-8"))
            best_loss = float(d.get("best_loss", d.get("loss", best_loss)))
        except Exception:
            pass

    trainer = Trainer(
        model,
        device=device,
        lr=float(config.get("lr", 3e-4)),
        zclip_z_thresh=float(config.get("zclip_z_thresh", 2.5)),
        zclip_alpha=float(config.get("zclip_alpha", 0.97)),
        zclip_max_norm=float(config.get("zclip_max_norm", 1.0)),
        log_dir=work_dir / "tb",
        use_amp=bool(config.get("use_amp", True)),
        feature_dropout_p=float(config.get("feature_dropout_p", 0.3)),
        w_volatility=float(config.get("w_volatility", 0.05)),
        accum_steps=int(config.get("accum_steps", 1)),
        warmup_steps=int(config.get("warmup_steps", 1500)),
        warmup_lr_start=config.get("warmup_lr_start", None),
        lr_eta_min=float(config.get("lr_eta_min", 1e-5)),
        lr_T0=int(config.get("lr_T0", 5000)),
        lr_T_mult=int(config.get("lr_T_mult", 2)),
        use_compile=bool(config.get("use_compile", False)),
        optimizer=str(config.get("optimizer", "nadamw")),
        cosmos_rank=int(config.get("cosmos_rank", 64)),
        cosmos_gamma=float(config.get("cosmos_gamma", 0.2)),
        swa_start=int(config.get("swa_start", 0)),
        swa_freq=int(config.get("swa_freq", 50)),
        # Tunable loss weights (Ray Tune ablations)
        w_policy=float(config.get("w_policy", 1.0)),
        w_soft=float(config.get("w_soft", 0.5)),
        w_future=float(config.get("w_future", 0.15)),
        w_wdl=float(config.get("w_wdl", 1.0)),
        w_sf_move=float(config.get("w_sf_move", 0.15)),
        w_sf_eval=float(config.get("w_sf_eval", 0.15)),
        w_categorical=float(config.get("w_categorical", 0.10)),
        w_sf_volatility=float(config.get("w_sf_volatility", config.get("w_volatility", 0.05))),
        w_moves_left=float(config.get("w_moves_left", 0.02)),
        w_sf_wdl=float(config.get("w_sf_wdl", 1.0)),
        sf_wdl_conf_power=float(config.get("sf_wdl_conf_power", 0.0)),
        sf_wdl_draw_scale=float(config.get("sf_wdl_draw_scale", 1.0)),
    )

    salvage_restore_donor_config = bool(config.get("salvage_restore_donor_config", False))
    salvage_restore_full_trainer_state = bool(config.get("salvage_restore_full_trainer_state", False))
    salvage_startup_no_share_iters = max(0, int(config.get("salvage_startup_no_share_iters", 0)))
    salvage_startup_max_train_steps = max(0, int(config.get("salvage_startup_max_train_steps", 0)))
    salvage_startup_post_share_ramp_iters = max(
        0, int(config.get("salvage_startup_post_share_ramp_iters", 0))
    )
    salvage_startup_post_share_max_train_steps = max(
        0, int(config.get("salvage_startup_post_share_max_train_steps", 0))
    )

    # Dynamic sf_wdl weight schedule: start at w_sf_wdl (default 1.0, equal to w_wdl)
    # and decline linearly as random_move_prob drops.  When random_move_prob reaches
    # sf_wdl_floor_at (default 0.1), the weight is sf_wdl_floor (default 0.1).
    # This bootstraps the value head from SF evaluations early on, then fades to
    # game-outcome WDL once the model is strong enough to generate meaningful games.
    sf_wdl_start = float(config.get("w_sf_wdl", 1.0))
    sf_wdl_floor = float(config.get("sf_wdl_floor", 0.1))
    sf_wdl_floor_at = float(config.get("sf_wdl_floor_at", 0.1))  # random_move_prob at which we hit the floor

    # Restore from checkpoint if provided by Ray.
    # NOTE: we restore PID state later (after PID is constructed).
    restored_pid_state = None
    restored_rng_state = None
    restored_trial_meta = None
    seed_warmstart_used = False
    seed_warmstart_slot = -1
    seed_warmstart_slots_total = 0
    seed_warmstart_dir: Path | None = None
    seed_warmstart_replay_dir: Path | None = None
    seed_warmstart_manifest_row: dict | None = None
    salvage_origin_used = False
    salvage_origin_slot = -1
    salvage_origin_slots_total = 0
    salvage_origin_dir = ""
    startup_source = "fresh"
    restored_owner_optimizer = ""
    ckpt = session.get_checkpoint()
    if ckpt is not None:
        ckpt_dir = Path(ckpt.to_directory())
        maybe = ckpt_dir / "trainer.pt"
        pid_path = ckpt_dir / "pid_state.json"
        if pid_path.exists():
            try:
                restored_pid_state = json.loads(pid_path.read_text(encoding="utf-8"))
            except Exception:
                restored_pid_state = None
        rng_path = ckpt_dir / "rng_state.json"
        if rng_path.exists():
            try:
                restored_rng_state = json.loads(rng_path.read_text(encoding="utf-8"))
            except Exception:
                restored_rng_state = None
        trial_meta_path = ckpt_dir / "trial_meta.json"
        if trial_meta_path.exists():
            try:
                restored_trial_meta = json.loads(trial_meta_path.read_text(encoding="utf-8"))
            except Exception:
                restored_trial_meta = None
        if isinstance(restored_trial_meta, dict):
            restored_owner_optimizer = str(restored_trial_meta.get("optimizer", "") or "")
        current_optimizer = str(config.get("optimizer", "nadamw")).lower()
        if maybe.exists():
            model_only_restore = False
            if isinstance(restored_trial_meta, dict):
                owner_trial_id = str(restored_trial_meta.get("owner_trial_id", ""))
                if owner_trial_id and owner_trial_id != trial_id:
                    if restored_owner_optimizer:
                        model_only_restore = restored_owner_optimizer.lower() != current_optimizer
                    elif bool(config.get("search_optimizer", False)):
                        model_only_restore = True
            if model_only_restore:
                ckpt_data = torch.load(str(maybe), map_location=device)
                trainer.model.load_state_dict(ckpt_data["model"])
                del ckpt_data
                startup_source = "checkpoint_model_only"
            else:
                trainer.load(maybe)
                startup_source = "checkpoint"
    elif isinstance(config.get("salvage_seed_pool_dir"), str) and str(config.get("salvage_seed_pool_dir", "")).strip():
        seed_pool_dir = Path(str(config.get("salvage_seed_pool_dir"))).expanduser()
        if not seed_pool_dir.is_dir():
            raise RuntimeError(f"salvage_seed_pool_dir not found: {seed_pool_dir}")

        picked_dir, picked_slot, num_slots = _select_salvage_seed_slot(
            seed_pool_dir=seed_pool_dir,
            trial_dir=trial_dir,
            trial_id=trial_id,
        )
        if picked_dir is None:
            raise RuntimeError(
                f"salvage requested but no seed slot could be selected for "
                f"trial_id={trial_id} from {seed_pool_dir}"
            )

        maybe = Path(picked_dir) / "trainer.pt"
        if not maybe.exists():
            raise RuntimeError(f"salvage seed missing trainer.pt: {maybe}")

        startup_source = "salvage"
        seed_warmstart_used = True
        seed_warmstart_slot = int(picked_slot)
        seed_warmstart_slots_total = int(num_slots)
        seed_warmstart_dir = Path(picked_dir)
        seed_warmstart_replay_dir = seed_warmstart_dir / "replay_shards"
        salvage_origin_used = True
        salvage_origin_slot = int(seed_warmstart_slot)
        salvage_origin_slots_total = int(seed_warmstart_slots_total)
        salvage_origin_dir = str(seed_warmstart_dir.resolve())
        seed_entry = _load_salvage_manifest_entry(
            seed_pool_dir=seed_pool_dir,
            slot=seed_warmstart_slot,
        )
        if isinstance(seed_entry, dict):
            rr = seed_entry.get("result_row")
            if isinstance(rr, dict):
                seed_warmstart_manifest_row = rr
                if salvage_restore_donor_config:
                    donor_cfg = rr.get("config")
                    if isinstance(donor_cfg, dict):
                        # Start this fresh lineage from the donor's effective
                        # tunable trainer settings rather than the new run's
                        # freshly sampled values.
                        for k in (
                            "lr",
                            "cosmos_gamma",
                            "w_soft",
                            "w_future",
                            "w_wdl",
                            "w_sf_move",
                            "w_sf_eval",
                            "w_categorical",
                            "w_volatility",
                            "w_sf_wdl",
                            "sf_wdl_conf_power",
                            "sf_wdl_draw_scale",
                        ):
                            if k in donor_cfg:
                                config[k] = donor_cfg[k]
                        if "lr" in config:
                            trainer.set_peak_lr(float(config["lr"]), rescale_current=False)
                        if "cosmos_gamma" in config and hasattr(trainer.opt, "gamma"):
                            trainer.opt.gamma = float(config["cosmos_gamma"])
                        for wk in (
                            "w_soft",
                            "w_future",
                            "w_wdl",
                            "w_sf_move",
                            "w_sf_eval",
                            "w_categorical",
                            "w_volatility",
                            "w_sf_wdl",
                            "sf_wdl_conf_power",
                            "sf_wdl_draw_scale",
                        ):
                            if wk in config:
                                setattr(trainer, wk, float(config[wk]))

        if salvage_restore_full_trainer_state:
            trainer.load(maybe)
        else:
            # Salvage starts a fresh Tune population from strong model weights, but
            # not from the old optimizer/scheduler/step state. Carrying those over
            # is appropriate for true in-place resume, not for a new population with
            # a copied replay window and fresh perturbation lifecycle.
            salvage_ckpt = torch.load(str(maybe), map_location=device)
            trainer.model.load_state_dict(salvage_ckpt["model"])
            del salvage_ckpt
        print(
            f"[trial] salvage warmstart loaded slot={seed_warmstart_slot} "
            f"of {seed_warmstart_slots_total} from {seed_warmstart_dir}"
        )
        pid_path = seed_warmstart_dir / "pid_state.json"
        if pid_path.exists():
            try:
                restored_pid_state = json.loads(pid_path.read_text(encoding="utf-8"))
            except Exception:
                restored_pid_state = None
        restored_pid_state, pid_manifest_overrides = _merge_pid_state_from_result_row(
            pid_state=restored_pid_state,
            result_row=seed_warmstart_manifest_row,
        )
        if pid_manifest_overrides:
            print(
                "[trial] salvage PID overrides from manifest row: "
                + ", ".join(pid_manifest_overrides)
            )

    restored_owner_trial_id = ""
    restored_owner_trial_dir = ""
    global_iter = 0
    if isinstance(restored_trial_meta, dict):
        restored_owner_trial_id = str(restored_trial_meta.get("owner_trial_id", ""))
        restored_owner_trial_dir = str(restored_trial_meta.get("owner_trial_dir", ""))
        restored_owner_optimizer = str(restored_trial_meta.get("optimizer", restored_owner_optimizer))
        salvage_origin_used = bool(restored_trial_meta.get("salvage_origin_used", salvage_origin_used))
        salvage_origin_slot = int(restored_trial_meta.get("salvage_origin_slot", salvage_origin_slot))
        salvage_origin_slots_total = int(
            restored_trial_meta.get("salvage_origin_slots_total", salvage_origin_slots_total)
        )
        salvage_origin_dir = str(restored_trial_meta.get("salvage_origin_dir", salvage_origin_dir or ""))
        global_iter = int(restored_trial_meta.get("global_iter", 0))
    cross_trial_restore = bool(
        ckpt is not None and restored_owner_trial_id and restored_owner_trial_id != trial_id
    )
    if cross_trial_restore and startup_source == "checkpoint":
        startup_source = "exploit_restore"
    elif cross_trial_restore and startup_source == "checkpoint_model_only":
        startup_source = "exploit_restore_model_only"

    if restored_rng_state is not None and not cross_trial_restore:
        try:
            rng.bit_generator.state = restored_rng_state
        except Exception:
            pass

    if ckpt is not None and cross_trial_restore:
        # PB2 exploit clone: fork RNG stream so recipients do not replay donor opening sequences.
        restore_rows = _count_jsonl_rows(trial_dir / "result.json")
        fork_seed = _stable_seed_u32(
            base_seed,
            trial_id,
            "exploit",
            restore_rows,
            int(getattr(trainer, "step", 0)),
        )
        active_seed = int(fork_seed)
        rng = np.random.default_rng(active_seed)
        torch.manual_seed(active_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(active_seed)
        print(
            f"[trial] PB2 exploit restore detected: owner={restored_owner_trial_id} "
            f"recipient={trial_id} fork_seed={active_seed} "
            f"owner_optimizer={restored_owner_optimizer or 'unknown'} "
            f"recipient_optimizer={str(config.get('optimizer', 'nadamw')).lower()} "
            f"restore_mode={startup_source}"
        )

    # Growing sliding window: start small, grow as the net matures.
    window_start = int(config.get("replay_window_start", 100_000))
    window_max = int(config.get("replay_window_max", int(config.get("replay_capacity", 1_000_000))))
    window_growth = int(config.get("replay_window_growth", 10_000))
    current_window = window_start

    shuffle_cap = int(config.get("shuffle_buffer_size", 20_000))
    shard_size = int(config.get("shard_size", 1000))
    replay_shard_dir = work_dir / "replay_shards"

    # Optional warmstart replay from salvage seed slot (fresh trials only).
    if (
        seed_warmstart_used
        and (not cross_trial_restore)
        and seed_warmstart_replay_dir is not None
        and seed_warmstart_replay_dir.is_dir()
        and (not any(replay_shard_dir.glob("shard_*.npz")))
    ):
        replay_shard_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for sp in sorted(seed_warmstart_replay_dir.glob("shard_*.npz")):
            shutil.copy2(str(sp), str(replay_shard_dir / sp.name))
            copied += 1
        if copied:
            print(
                f"[trial] Copied {copied} replay shards from salvage slot "
                f"{seed_warmstart_slot} ({seed_warmstart_replay_dir})"
            )

    shared_summary = {
        "source_trials_selected": 0,
        "source_trials_ingested": 0,
        "source_trials_skipped_repeat": 0,
        "source_shards_loaded": 0,
        "source_samples_ingested": 0,
    }

    # Seed replay buffer with shared iter-0 data (played once from bootstrap net).
    # Only copy if this is a fresh trial (no existing shards in replay_shard_dir).
    shared_shards_dir = config.get("shared_shards_dir")
    if shared_shards_dir and not any(replay_shard_dir.glob("shard_*.npz")) and (not cross_trial_restore):
        src = Path(shared_shards_dir)
        if src.is_dir():
            replay_shard_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for sp in sorted(src.glob("shard_*.npz")):
                shutil.copy2(str(sp), str(replay_shard_dir / sp.name))
                copied += 1
            if copied:
                shared_summary["source_shards_loaded"] = int(copied)
                print(f"[trial] Copied {copied} shared iter-0 shards from {src}")

    if cross_trial_restore and bool(config.get("exploit_replay_refresh_enabled", True)):
        donor_trial_dir = Path(restored_owner_trial_dir).expanduser() if restored_owner_trial_dir else None
        refresh_summary = _refresh_replay_shards_on_exploit(
            replay_shard_dir=replay_shard_dir,
            recipient_trial_dir=trial_dir,
            donor_trial_dir=donor_trial_dir,
            keep_recent_fraction=float(config.get("exploit_replay_local_keep_recent_fraction", 0.20)),
            keep_older_fraction=float(config.get("exploit_replay_local_keep_older_fraction", 0.65)),
            donor_shards=int(config.get("exploit_replay_donor_shards", 0)),
            donor_skip_newest=int(config.get("exploit_replay_skip_newest", 0)),
            shard_size=int(shard_size),
            holdout_fraction=float(config.get("holdout_fraction", 0.02)),
        )
        print(
            "[trial] replay refresh after exploit: "
            f"local_before={refresh_summary['local_before']} "
            f"deleted={refresh_summary['local_deleted']} "
            f"local_recent_deleted={refresh_summary['local_recent_deleted']} "
            f"local_older_deleted={refresh_summary['local_older_deleted']} "
            f"after_keep={refresh_summary['local_after_keep']} "
            f"donor_available={refresh_summary['donor_available']} "
            f"donor_selected={refresh_summary['donor_selected']} "
            f"donor_copied={refresh_summary['donor_copied']} "
            f"final={refresh_summary['local_final']}"
        )

    buf = DiskReplayBuffer(
        current_window,
        shard_dir=replay_shard_dir,
        rng=rng,
        shuffle_cap=shuffle_cap,
        shard_size=shard_size,
    )

    # Preserve intentionally seeded replay (resume, salvage warmstart, shared-shard
    # bootstrap), but keep plain fresh starts at replay_window_start so easy early
    # games evict promptly instead of inheriting stale local shards.
    seeded_replay_start = bool(ckpt is not None or seed_warmstart_used or shared_summary["source_shards_loaded"] > 0)
    if seeded_replay_start:
        current_window = max(int(current_window), int(len(buf)))
    buf.capacity = int(current_window)
    holdout_buf = ReplayBuffer(int(config.get("holdout_capacity", 50_000)), rng=rng)
    holdout_frac = float(config.get("holdout_fraction", 0.02))

    # Load pre-trained bootstrap checkpoint (trained offline via scripts/train_bootstrap.py).
    # This gives the value head a working signal so first MCTS searches are better than random.
    # IMPORTANT: Only load MODEL WEIGHTS — do NOT restore optimizer/scheduler/step.
    # The bootstrap was trained for ~13k steps with its own LR schedule; carrying that
    # state into the trainable causes: (1) step=13323 skips warmup entirely,
    # (2) scheduler resumes mid-cosine-cycle with near-zero LR then spikes on restart,
    # (3) optimizer momentum buffers from bootstrap's data distribution cause wrong
    # gradient directions on selfplay data, (4) PB2's lr perturbation has no effect
    # because scheduler's base_lr is locked to bootstrap's lr (0.0003).
    bootstrap_ckpt = config.get("bootstrap_checkpoint")
    if bootstrap_ckpt and ckpt is None and (not seed_warmstart_used):
        # Only load bootstrap if Ray didn't restore a trial checkpoint (i.e. fresh start).
        bp = Path(bootstrap_ckpt)
        if bp.exists():
            print(f"[trial] Loading pre-trained bootstrap model weights: {bp}")
            ckpt_data = torch.load(str(bp), map_location=device)
            trainer.model.load_state_dict(ckpt_data["model"])
            if bool(config.get("bootstrap_zero_policy_heads", False)):
                zeroed = zero_policy_head_parameters_(trainer.model)
                if zeroed:
                    print(f"[trial] Zeroed bootstrap policy heads: {', '.join(zeroed)}")
            # Deliberately skip: optimizer, scheduler, step — start fresh.
            del ckpt_data
        else:
            print(f"[trial] WARNING: bootstrap checkpoint not found: {bp}")

    # Net gating config
    gate_games = int(config.get("gate_games", 0))  # 0 = disabled
    gate_threshold = float(config.get("gate_threshold", 0.50))
    gate_interval = int(config.get("gate_interval", 1))  # gate every N iters

    # Holdout management: optionally freeze once it reaches a target size, and optionally
    # reset it if the training distribution drifts too far.
    freeze_holdout_at = int(config.get("freeze_holdout_at", 0))
    reset_holdout_on_drift = bool(config.get("reset_holdout_on_drift", False))
    drift_threshold = float(config.get("drift_threshold", 0.0))
    drift_sample_size = int(config.get("drift_sample_size", 256))
    holdout_frozen = False
    holdout_generation = 0

    distributed_workers_per_trial = max(0, int(config.get("distributed_workers_per_trial", 0)))
    use_distributed_selfplay = (
        distributed_workers_per_trial > 0
        and isinstance(config.get("distributed_server_root"), str)
        and str(config.get("distributed_server_root", "")).strip()
        and isinstance(config.get("distributed_server_url"), str)
        and str(config.get("distributed_server_url", "")).strip()
    )

    sf_workers = int(config.get("sf_workers", 1))
    sf_multipv = int(config.get("sf_multipv", 1))
    sf_hash_mb = int(config.get("sf_hash_mb", 16))
    need_local_sf = (not use_distributed_selfplay) or (gate_games > 0)
    sf = None
    if need_local_sf:
        if sf_workers > 1:
            sf = StockfishPool(
                path=str(config["stockfish_path"]),
                nodes=int(config.get("sf_nodes", 500)),
                num_workers=sf_workers,
                multipv=sf_multipv,
                hash_mb=sf_hash_mb,
            )
        else:
            sf = StockfishUCI(
                str(config["stockfish_path"]),
                nodes=int(config.get("sf_nodes", 500)),
                multipv=sf_multipv,
                hash_mb=sf_hash_mb,
            )

    distributed_server_root = (
        Path(str(config["distributed_server_root"])).expanduser()
        if use_distributed_selfplay
        else None
    )
    distributed_dirs = (
        _trial_server_dirs(server_root=distributed_server_root, trial_id=trial_id)
        if distributed_server_root is not None
        else None
    )
    distributed_worker_procs: list[subprocess.Popen[bytes]] = []

    eval_games = int(config.get("eval_games", 0))
    eval_sf_nodes = int(config.get("eval_sf_nodes", config.get("sf_nodes", 500)))
    eval_mcts_sims = int(config.get("eval_mcts_simulations", config.get("mcts_simulations", 50)))

    eval_sf = None
    if eval_games > 0:
        # For fixed-strength evaluation, use a dedicated engine instance with its own node limit.
        eval_sf = StockfishUCI(
            str(config["stockfish_path"]),
            nodes=eval_sf_nodes,
            multipv=1,
            hash_mb=sf_hash_mb,
        )

    pid = None
    if bool(config.get("sf_pid_enabled", True)):
        pid = DifficultyPID(
            initial_nodes=int(config.get("sf_nodes", 500)),
            target_winrate=float(config.get("sf_pid_target_winrate", 0.52)),
            ema_alpha=float(config.get("sf_pid_ema_alpha", 0.03)),
            deadzone=float(config.get("sf_pid_deadzone", 0.05)),
            rate_limit=float(config.get("sf_pid_rate_limit", 0.10)),
            min_games_between_adjust=int(config.get("sf_pid_min_games_between_adjust", 30)),
            kp=float(config.get("sf_pid_kp", 1.5)),
            ki=float(config.get("sf_pid_ki", 0.10)),
            kd=float(config.get("sf_pid_kd", 0.0)),
            integral_clamp=float(config.get("sf_pid_integral_clamp", 1.0)),
            min_nodes=int(config.get("sf_pid_min_nodes", 250)),
            max_nodes=int(config.get("sf_pid_max_nodes", 1000000)),
            initial_skill_level=int(config.get("sf_pid_initial_skill_level", 0)),
            skill_min=int(config.get("sf_pid_skill_min", 0)),
            skill_max=int(config.get("sf_pid_skill_max", 20)),
            skill_promote_nodes=int(config.get("sf_pid_skill_promote_nodes", 200)),
            skill_demote_nodes=int(config.get("sf_pid_skill_demote_nodes", 100)),
            skill_nodes_on_promote=int(config.get("sf_pid_skill_nodes_on_promote", 100)),
            skill_nodes_on_demote=int(config.get("sf_pid_skill_nodes_on_demote", 150)),
            initial_random_move_prob=float(config.get("sf_pid_random_move_prob_start", 1.0)),
            random_move_prob_min=float(config.get("sf_pid_random_move_prob_min", 0.0)),
            random_move_prob_max=float(config.get("sf_pid_random_move_prob_max", 1.0)),
            random_move_stage_end=float(config.get("sf_pid_random_move_stage_end", 0.5)),
            max_rand_step=float(config.get("sf_pid_max_rand_step", 0.01)),
        )
        if restored_pid_state is not None:
            try:
                pid.load_state_dict(restored_pid_state)
            except Exception:
                pass

    # Optional puzzle evaluation suite.
    puzzle_suite = None
    puzzle_interval = int(config.get("puzzle_interval", 1))
    puzzle_sims = int(config.get("puzzle_simulations", 200))
    _puzzle_epd = config.get("puzzle_epd")
    if _puzzle_epd and puzzle_interval > 0:
        from chess_anti_engine.eval import load_epd
        try:
            puzzle_suite = load_epd(_puzzle_epd)
        except FileNotFoundError:
            puzzle_suite = None

    pause_marker_path = _resolve_pause_marker_path(config=config, trial_dir=trial_dir)
    pause_poll_seconds = int(config.get("pause_poll_seconds", 60))

    if use_distributed_selfplay:
        current_rand_init = (
            float(pid.random_move_prob)
            if pid is not None
            else float(config.get("sf_pid_random_move_prob_start", 0.0))
        )
        current_nodes_init = int(pid.nodes) if pid is not None else int(config.get("sf_nodes", 500))
        current_skill_init = int(pid.skill_level) if pid is not None else 0
        sims_init = int(config.get("mcts_simulations", 50))
        if bool(config.get("progressive_mcts", True)):
            sims_init = progressive_mcts_simulations(
                int(getattr(trainer, "step", 0)),
                start=int(config.get("mcts_start_simulations", 50)),
                max_sims=int(config.get("mcts_simulations", 50)),
                ramp_steps=int(config.get("mcts_ramp_steps", 10_000)),
                exponent=float(config.get("mcts_ramp_exponent", 2.0)),
            )
        _publish_distributed_trial_state(
            trainer=trainer,
            config=config,
            model_cfg=model_cfg,
            server_root=distributed_server_root,
            trial_id=trial_id,
            training_iteration=0,
            trainer_step=int(getattr(trainer, "step", 0)),
            sf_nodes=current_nodes_init,
            random_move_prob=current_rand_init,
            skill_level=current_skill_init,
            mcts_simulations=int(sims_init),
        )
        distributed_worker_procs = _ensure_distributed_workers(
            config=config,
            trial_dir=trial_dir,
            trial_id=trial_id,
            procs=distributed_worker_procs,
        )

    try:
        iterations = int(config.get("iterations", 10))
        for it in range(iterations):
            iteration_idx = int(it) + 1
            global_iter += 1
            in_salvage_startup_grace = (
                startup_source == "salvage"
                and bool(salvage_origin_used)
                and int(it) < salvage_startup_no_share_iters
            )
            _wait_if_paused(
                pause_marker_path=pause_marker_path,
                poll_seconds=pause_poll_seconds,
                trial_id=trial_id,
                iteration=iteration_idx,
            )

            # Difficulty knobs used for this iteration's selfplay (kept fixed across
            # selfplay chunks). PID is updated once per iteration AFTER training so
            # changes align to net updates rather than chunk noise.
            current_rand = float(pid.random_move_prob) if pid is not None else float(config.get("sf_pid_random_move_prob_start", 0.0))
            sf_nodes_used = (
                int(getattr(sf, "nodes", 0) or 0)
                if sf is not None
                else (int(pid.nodes) if pid is not None else int(config.get("sf_nodes", 500)))
            )
            skill_level_used = int(getattr(pid, "skill_level", 0) or 0) if pid is not None else 0

            base_sims = int(config.get("mcts_simulations", 50))
            sims = base_sims
            if bool(config.get("progressive_mcts", True)):
                sims = progressive_mcts_simulations(
                    int(getattr(trainer, "step", 0)),
                    start=int(config.get("mcts_start_simulations", 50)),
                    max_sims=base_sims,
                    ramp_steps=int(config.get("mcts_ramp_steps", 10_000)),
                    exponent=float(config.get("mcts_ramp_exponent", 2.0)),
                )

            # Play games in mini-batches to keep memory low (each mini-batch
            # frees its MCTS trees / board objects before the next starts).
            total_games = _games_per_iter_for_iteration(config, iteration_idx)
            selfplay_batch = int(config.get("selfplay_batch", 10))
            games_remaining = total_games

            # Accumulators for stats across mini-batches.
            total_w = total_d = total_l = 0
            total_games_generated = 0
            total_game_plies = 0
            total_timeout_games = 0
            total_draw_games = 0
            total_positions = 0
            total_sf_d6 = 0.0
            total_sf_d6_n = 0
            distributed_stale_positions = 0
            distributed_stale_games = 0

            if use_distributed_selfplay:
                distributed_worker_procs = _ensure_distributed_workers(
                    config=config,
                    trial_dir=trial_dir,
                    trial_id=trial_id,
                    procs=distributed_worker_procs,
                )
                published_model_sha = _publish_distributed_trial_state(
                    trainer=trainer,
                    config=config,
                    model_cfg=model_cfg,
                    server_root=distributed_server_root,
                    trial_id=trial_id,
                    training_iteration=int(iteration_idx),
                    trainer_step=int(getattr(trainer, "step", 0)),
                    sf_nodes=int(pid.nodes) if pid is not None else int(config.get("sf_nodes", 500)),
                    random_move_prob=float(current_rand),
                    skill_level=int(skill_level_used),
                    mcts_simulations=int(sims),
                )
                ingest_summary = _ingest_distributed_selfplay(
                    buf=buf,
                    holdout_buf=holdout_buf,
                    holdout_frac=float(holdout_frac),
                    holdout_frozen=bool(holdout_frozen),
                    inbox_dir=distributed_dirs["inbox_dir"],
                    processed_dir=distributed_dirs["processed_dir"],
                    target_games=int(total_games),
                    target_model_sha=str(published_model_sha),
                    wait_timeout_s=float(config.get("distributed_wait_timeout_seconds", 900.0)),
                    poll_seconds=float(config.get("distributed_worker_poll_seconds", 1.0)),
                    rng=rng,
                )
                total_w = int(ingest_summary["matching_w"])
                total_d = int(ingest_summary["matching_d"])
                total_l = int(ingest_summary["matching_l"])
                total_games_generated = int(ingest_summary["matching_games"])
                total_game_plies = int(ingest_summary["matching_total_game_plies"])
                total_timeout_games = int(ingest_summary["matching_timeout_games"])
                total_draw_games = int(ingest_summary["matching_total_draw_games"])
                total_positions = int(ingest_summary["positions_replay_added"])
                distributed_stale_positions = int(ingest_summary["stale_positions"])
                distributed_stale_games = int(ingest_summary["stale_games"])
            else:
                selfplay_kwargs = dict(
                    device=device,
                    rng=rng,
                    stockfish=sf,
                    opponent_random_move_prob=current_rand,
                    opponent_topk_stage_end=float(config.get("sf_pid_random_move_stage_end", 0.5)),
                    selfplay_fraction=float(config.get("selfplay_fraction", 0.0)),
                    temperature=float(config.get("temperature", 1.0)),
                    temperature_drop_plies=int(config.get("temperature_drop_plies", 0)),
                    temperature_after=float(config.get("temperature_after", 0.0)),
                    temperature_decay_start_move=int(config.get("temperature_decay_start_move", 20)),
                    temperature_decay_moves=int(config.get("temperature_decay_moves", 60)),
                    temperature_endgame=float(config.get("temperature_endgame", 0.6)),
                    max_plies=int(config.get("max_plies", 120)),
                    mcts_simulations=int(sims),
                    mcts_type=str(config.get("mcts", "puct")),
                    playout_cap_fraction=float(config.get("playout_cap_fraction", 0.25)),
                    fast_simulations=int(config.get("fast_simulations", 8)),
                    sf_policy_temp=float(config.get("sf_policy_temp", 0.25)),
                    sf_policy_label_smooth=float(config.get("sf_policy_label_smooth", 0.05)),
                    timeout_adjudication_threshold=float(config.get("timeout_adjudication_threshold", 0.90)),
                    volatility_source=str(config.get("volatility_source", "raw")),
                    opening_book_path=config.get("opening_book_path"),
                    opening_book_max_plies=int(config.get("opening_book_max_plies", 4)),
                    opening_book_max_games=int(config.get("opening_book_max_games", 200_000)),
                    opening_book_prob=float(config.get("opening_book_prob", 1.0)),
                    random_start_plies=int(config.get("random_start_plies", 0)),
                    syzygy_path=config.get("syzygy_path"),
                    syzygy_policy=bool(config.get("syzygy_policy", False)),
                    diff_focus_enabled=bool(config.get("diff_focus_enabled", True)),
                    diff_focus_q_weight=float(config.get("diff_focus_q_weight", 6.0)),
                    diff_focus_pol_scale=float(config.get("diff_focus_pol_scale", 3.5)),
                    diff_focus_slope=float(config.get("diff_focus_slope", 3.0)),
                    diff_focus_min=float(config.get("diff_focus_min", 0.025)),
                    categorical_bins=int(config.get("categorical_bins", 32)),
                    hlgauss_sigma=float(config.get("hlgauss_sigma", 0.04)),
                    fpu_reduction=float(config.get("fpu_reduction", 1.2)),
                    fpu_at_root=float(config.get("fpu_at_root", 1.0)),
                    soft_policy_temp=float(config.get("soft_policy_temp", 2.0)),
                )

                while games_remaining > 0:
                    chunk = min(selfplay_batch, games_remaining)
                    samples, stats = play_batch(trainer.model, games=chunk, **selfplay_kwargs)
                    games_remaining -= chunk

                    total_games_generated += int(stats.games)
                    total_w += stats.w
                    total_d += stats.d
                    total_l += stats.l
                    total_game_plies += int(getattr(stats, "total_game_plies", 0))
                    total_timeout_games += int(getattr(stats, "timeout_games", 0))
                    total_draw_games += int(getattr(stats, "total_draw_games", 0))
                    total_positions += stats.positions
                    total_sf_d6 += float(getattr(stats, "sf_eval_delta6", 0.0)) * int(getattr(stats, "sf_eval_delta6_n", 0))
                    total_sf_d6_n += int(getattr(stats, "sf_eval_delta6_n", 0))

                    train_samples: list = []
                    for s in samples:
                        if holdout_frac > 0.0 and (not holdout_frozen) and (rng.random() < holdout_frac):
                            holdout_buf.add_many([s])
                        else:
                            train_samples.append(s)
                    if train_samples:
                        buf.add_many(train_samples)
                    del samples

            # Growing window: expand buffer capacity each iteration before
            # cross-trial imports so we do not prune newly shared data against
            # the previous (smaller) capacity.
            if current_window < window_max:
                current_window = min(current_window + window_growth, window_max)
                if buf.capacity < current_window:
                    buf.capacity = current_window

            # Pull fresh top-trial data after selfplay so we train on the
            # newest available generations in asynchronous runs.
            shared_summary = {
                "source_trials_selected": 0,
                "source_trials_ingested": 0,
                "source_trials_skipped_repeat": 0,
                "source_shards_loaded": 0,
                "source_samples_ingested": 0,
            }
            if bool(config.get("exploit_replay_share_top_enabled", False)) and (not in_salvage_startup_grace):
                shared_summary = _share_top_replay_each_iteration(
                    recipient_trial_dir=trial_dir,
                    replay_shard_dir=replay_shard_dir,
                    buf=buf,
                    top_k_trials=int(config.get("exploit_replay_top_k_trials", 5)),
                    within_best_frac=float(config.get("exploit_replay_top_within_best_frac", 0.10)),
                    min_metric=float(config.get("exploit_replay_top_min_metric", -1e9)),
                    source_skip_newest=int(config.get("exploit_replay_skip_newest", 1)),
                    shard_size=int(shard_size),
                    holdout_fraction=float(holdout_frac),
                    max_unseen_iters_per_source=int(config.get("exploit_replay_max_unseen_iters_per_source", 2)),
                )
                if shared_summary["source_samples_ingested"] > 0:
                    print(
                        "[trial] replay share this iter: "
                        f"sources_selected={shared_summary['source_trials_selected']} "
                        f"sources_ingested={shared_summary['source_trials_ingested']} "
                        f"sources_skipped_repeat={shared_summary['source_trials_skipped_repeat']} "
                        f"source_shards_loaded={shared_summary['source_shards_loaded']} "
                        f"source_samples_ingested={shared_summary['source_samples_ingested']}"
                    )
            elif bool(config.get("exploit_replay_share_top_enabled", False)) and in_salvage_startup_grace:
                print(
                    "[trial] replay share skipped during salvage startup grace: "
                    f"iter={it} grace_iters={salvage_startup_no_share_iters}"
                )

            imported_samples_this_iter = int(shared_summary.get("source_samples_ingested", 0))

            train_samples = []  # Already flushed to buf above.

            if (not holdout_frozen) and freeze_holdout_at > 0 and len(holdout_buf) >= freeze_holdout_at:
                holdout_frozen = True

            # Drift estimates (cheap heuristics for monitoring when data distribution moves).
            drift_input_l2 = None
            drift_wdl_js = None
            drift_policy_entropy_diff = None
            drift_policy_entropy_train = None
            drift_policy_entropy_holdout = None

            if len(buf) >= drift_sample_size and len(holdout_buf) >= drift_sample_size:
                train_batch = buf.sample_batch(drift_sample_size)
                hold_batch = holdout_buf.sample_batch(drift_sample_size)

                # (1) Input drift: L2 distance between mean input plane tensors.
                train_x = np.stack([s.x for s in train_batch], axis=0).astype(np.float32, copy=False)
                hold_x = np.stack([s.x for s in hold_batch], axis=0).astype(np.float32, copy=False)
                drift_input_l2 = float(np.linalg.norm(train_x.mean(axis=0) - hold_x.mean(axis=0)))

                # (2) Target drift: WDL label distribution (JS divergence).
                def _wdl_hist(ss: list) -> np.ndarray:
                    h = np.zeros((3,), dtype=np.float64)
                    for s in ss:
                        t = int(getattr(s, "wdl_target", 1))
                        if 0 <= t <= 2:
                            h[t] += 1.0
                    h /= max(1.0, float(h.sum()))
                    return h

                p = _wdl_hist(train_batch)
                q = _wdl_hist(hold_batch)
                m = 0.5 * (p + q)
                eps = 1e-12
                drift_wdl_js = float(
                    0.5 * np.sum(p * (np.log(p + eps) - np.log(m + eps)))
                    + 0.5 * np.sum(q * (np.log(q + eps) - np.log(m + eps)))
                )

                # (3) Policy drift: mean entropy of stored policy targets.
                def _mean_entropy(ss: list) -> float:
                    ent = 0.0
                    n = 0
                    for s in ss:
                        pt = getattr(s, "policy_target", None)
                        if pt is None:
                            continue
                        p = np.asarray(pt, dtype=np.float64)
                        ps = float(p.sum())
                        if ps <= 0:
                            continue
                        p = p / ps
                        ent += float(-np.sum(p * np.log(p + eps)))
                        n += 1
                    return float(ent / max(1, n))

                drift_policy_entropy_train = _mean_entropy(train_batch)
                drift_policy_entropy_holdout = _mean_entropy(hold_batch)
                drift_policy_entropy_diff = float(drift_policy_entropy_train - drift_policy_entropy_holdout)

                # Optional holdout reset based on input drift threshold.
                if (
                    reset_holdout_on_drift
                    and (drift_threshold > 0.0)
                    and (drift_input_l2 is not None)
                    and (drift_input_l2 > drift_threshold)
                ):
                    holdout_buf.clear()
                    holdout_frozen = False
                    holdout_generation += 1

            # Re-read loss weights from config each iteration so PB2 perturbations
            # take effect immediately (PB2 mutates the config dict in-place).
            if "lr" in config:
                trainer.set_peak_lr(float(config["lr"]), rescale_current=True)
            if "cosmos_gamma" in config and hasattr(trainer.opt, "gamma"):
                trainer.opt.gamma = float(config["cosmos_gamma"])
            for wk in ("w_soft", "w_future", "w_wdl", "w_sf_move", "w_sf_eval",
                        "w_categorical", "w_volatility", "w_sf_wdl",
                        "sf_wdl_conf_power", "sf_wdl_draw_scale"):
                if wk in config:
                    setattr(trainer, wk, float(config[wk]))

            # Also update sf_wdl_start for the dynamic schedule below.
            sf_wdl_start = float(config.get("w_sf_wdl", sf_wdl_start))

            # Dynamic sf_wdl weight: interpolate between sf_wdl_start and sf_wdl_floor
            # based on how far random_move_prob has dropped from 1.0.
            # At random_move_prob=1.0: full SF bootstrapping (sf_wdl_start).
            # At random_move_prob=sf_wdl_floor_at: SF weight reaches sf_wdl_floor.
            if sf_wdl_start > 0:
                rp = current_rand  # 1.0 → 0.0 as model improves
                # Linear interp: rp=1.0 → sf_wdl_start, rp=sf_wdl_floor_at → sf_wdl_floor
                if rp >= 1.0:
                    cur_sf_wdl = sf_wdl_start
                elif rp <= sf_wdl_floor_at:
                    cur_sf_wdl = sf_wdl_floor
                else:
                    t = (rp - sf_wdl_floor_at) / (1.0 - sf_wdl_floor_at)
                    cur_sf_wdl = sf_wdl_floor + t * (sf_wdl_start - sf_wdl_floor)
                trainer.w_sf_wdl = cur_sf_wdl

            batch_size = int(config.get("batch_size", 128))
            accum_steps = max(1, int(config.get("accum_steps", 1)))
            effective_batch_size = batch_size * accum_steps
            skip_train = len(buf) < batch_size
            steps = 0
            target_sample_budget = 0
            window_target_samples = 0
            if skip_train:
                metrics = None
                gate_passed = True
            else:
                # Training steps: target the larger of:
                #   (1) newly-added positions this iteration, and
                #   (2) a configured fraction of the current replay size.
                # This keeps early training conservative while scaling updates
                # upward naturally as the replay grows.
                base_max_steps = int(config.get("train_steps", 25))
                train_window_fraction = max(0.0, float(config.get("train_window_fraction", 0.0)))
                effective_positions = int(total_positions) + int(imported_samples_this_iter)
                window_target_samples = int(math.ceil(train_window_fraction * max(0, len(buf))))
                target_sample_budget = max(int(effective_positions), int(window_target_samples))
                target_steps = max(1, int(math.ceil(target_sample_budget / max(1, effective_batch_size))))
                # If we imported shared games this iteration, use proportional steps
                # so top-1 vs top-5 import volume differences are reflected.
                if imported_samples_this_iter > 0:
                    steps = target_steps
                else:
                    steps = min(target_steps, base_max_steps)
                if startup_source == "salvage" and bool(salvage_origin_used):
                    if int(it) < salvage_startup_no_share_iters and salvage_startup_max_train_steps > 0:
                        steps = min(steps, salvage_startup_max_train_steps)
                    elif (
                        salvage_startup_post_share_ramp_iters > 0
                        and int(it) < (salvage_startup_no_share_iters + salvage_startup_post_share_ramp_iters)
                        and salvage_startup_post_share_max_train_steps > 0
                    ):
                        steps = min(steps, salvage_startup_post_share_max_train_steps)

                # Save model state for potential rollback (net gating).
                gate_passed = True
                pre_train_state = None
                if gate_games > 0 and (it % gate_interval == 0):
                    pre_train_state = {
                        k: v.clone() for k, v in trainer.model.state_dict().items()
                    }

                metrics = trainer.train_steps(
                    buf,
                    batch_size=batch_size,
                    steps=steps,
                )

                # Net gating: play gate_games and reject if winrate < threshold.
                if pre_train_state is not None and gate_games > 0:
                    gate_wr, gate_w, gate_d, gate_l = _gate_check(
                        trainer.model,
                        device=device,
                        rng=rng,
                        sf=sf,
                        gate_games=gate_games,
                        opponent_random_move_prob=current_rand,
                        config=config,
                    )

                    # Gate match index is separate from iteration because gates are rare.
                    gate_match_idx += 1
                    try:
                        gate_state_path.write_text(
                            json.dumps(
                                {"matches": int(gate_match_idx)},
                                indent=2,
                                sort_keys=True,
                            ),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

                    # TensorBoard logging for gating (x-axis = gate match #).
                    # Useful when gate checks happen infrequently.
                    try:
                        trainer.writer.add_scalar("gate/winrate", float(gate_wr), int(gate_match_idx))
                        trainer.writer.add_scalar("gate/win", float(gate_w), int(gate_match_idx))
                        trainer.writer.add_scalar("gate/draw", float(gate_d), int(gate_match_idx))
                        trainer.writer.add_scalar("gate/loss", float(gate_l), int(gate_match_idx))
                        trainer.writer.add_scalar("gate/passed", float(1.0 if gate_wr >= gate_threshold else 0.0), int(gate_match_idx))
                    except Exception:
                        pass

                    if gate_wr < gate_threshold:
                        # Revert model — training made it worse.
                        trainer.model.load_state_dict(pre_train_state)
                        gate_passed = False

            test_metrics = None
            if len(holdout_buf) >= int(config.get("batch_size", 128)):
                test_metrics = trainer.eval_steps(
                    holdout_buf,
                    batch_size=int(config.get("batch_size", 128)),
                    steps=int(config.get("test_steps", 10)),
                )

            eval_dict = {}
            if eval_games > 0 and eval_sf is not None:
                # Evaluation: fixed-strength games (no training data generated, only W/D/L).
                _eval_samples, eval_stats = play_batch(
                    trainer.model,
                    device=device,
                    rng=rng,
                    stockfish=eval_sf,
                    games=eval_games,
                    opponent_topk_stage_end=float(config.get("sf_pid_random_move_stage_end", 0.5)),
                    temperature=float(config.get("eval_temperature", 0.25)),
                    max_plies=int(config.get("eval_max_plies", config.get("max_plies", 120))),
                    mcts_simulations=eval_mcts_sims,
                    mcts_type=str(config.get("mcts", "puct")),
                    playout_cap_fraction=1.0,
                    fast_simulations=0,
                    sf_policy_temp=float(config.get("sf_policy_temp", 0.25)),
                    sf_policy_label_smooth=float(config.get("sf_policy_label_smooth", 0.05)),
                    timeout_adjudication_threshold=float(config.get("timeout_adjudication_threshold", 0.90)),
                    volatility_source=str(config.get("volatility_source", "raw")),
                )
                denom = float(max(1, eval_stats.w + eval_stats.d + eval_stats.l))
                eval_dict = {
                    "eval_win": eval_stats.w,
                    "eval_draw": eval_stats.d,
                    "eval_loss": eval_stats.l,
                    "eval_winrate": (float(eval_stats.w) + 0.5 * float(eval_stats.d)) / denom,
                }

            # Flush any remaining samples to disk before checkpointing.
            buf.flush()

            # Save a lightweight checkpoint (model+optimizer+step + PID state).
            ckpt_dir = work_dir / "ckpt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "trainer.pt"
            trainer.save(ckpt_path)
            try:
                (ckpt_dir / "rng_state.json").write_text(
                    json.dumps(rng.bit_generator.state, sort_keys=True),
                    encoding="utf-8",
                )
            except Exception:
                pass
            try:
                (ckpt_dir / "trial_meta.json").write_text(
                    json.dumps(
                        {
                            "owner_trial_id": str(trial_id),
                            "owner_trial_dir": str(trial_dir.resolve()),
                            "optimizer": str(config.get("optimizer", "nadamw")).lower(),
                            "base_seed": int(base_seed),
                            "active_seed": int(active_seed),
                            "startup_source": str(startup_source),
                            "salvage_origin_used": bool(salvage_origin_used),
                            "salvage_origin_slot": int(salvage_origin_slot),
                            "salvage_origin_slots_total": int(salvage_origin_slots_total),
                            "salvage_origin_dir": str(salvage_origin_dir),
                            "global_iter": int(global_iter),
                        },
                        sort_keys=True,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
            checkpoint = Checkpoint.from_directory(str(ckpt_dir))

            test_dict = {
                "holdout_frozen": int(1 if holdout_frozen else 0),
                "holdout_generation": int(holdout_generation),
            }
            if drift_input_l2 is not None:
                test_dict["data_drift_input_l2"] = float(drift_input_l2)
            if drift_wdl_js is not None:
                test_dict["data_drift_wdl_js"] = float(drift_wdl_js)
            if drift_policy_entropy_diff is not None:
                test_dict["data_drift_policy_entropy_diff"] = float(drift_policy_entropy_diff)
            if drift_policy_entropy_train is not None:
                test_dict["data_drift_policy_entropy_train"] = float(drift_policy_entropy_train)
            if drift_policy_entropy_holdout is not None:
                test_dict["data_drift_policy_entropy_holdout"] = float(drift_policy_entropy_holdout)

            if test_metrics is not None:
                test_dict.update(
                    {
                        "test_size": len(holdout_buf),
                        "test_loss": test_metrics.loss,
                        "test_policy_loss": test_metrics.policy_loss,
                        "test_soft_policy_loss": test_metrics.soft_policy_loss,
                        "test_future_policy_loss": test_metrics.future_policy_loss,
                        "test_wdl_loss": test_metrics.wdl_loss,
                        "test_sf_move_loss": test_metrics.sf_move_loss,
                        "test_sf_move_acc": test_metrics.sf_move_acc,
                        "test_sf_eval_loss": test_metrics.sf_eval_loss,
                        "test_categorical_loss": test_metrics.categorical_loss,
                        "test_volatility_loss": test_metrics.volatility_loss,
                        "test_sf_volatility_loss": test_metrics.sf_volatility_loss,
                        "test_moves_left_loss": test_metrics.moves_left_loss,
                    }
                )

            # Best-model tracking: prefer holdout loss when available, skip if no training yet.
            cur_loss = float(test_metrics.loss) if test_metrics is not None else (float(metrics.loss) if metrics is not None else float("inf"))
            if cur_loss < best_loss - 1e-12:
                best_loss = cur_loss
                trainer.save(best_dir / "trainer.pt")
                trainer.export_swa(best_dir / "best_model.pt")
                best_state_path.write_text(
                    json.dumps(
                        {
                            "best_loss": float(best_loss),
                            "iter": int(iteration_idx),
                            "trainer_step": int(getattr(trainer, "step", 0)),
                            "source": "test_loss" if test_metrics is not None else "train_loss",
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )

            # Update PID ONCE per iteration (after training) so difficulty changes
            # line up with net updates rather than intra-iteration selfplay noise.
            pid_update = None
            pid_ema_wr = 0.0
            total_games_played = max(1, int(total_w + total_d + total_l))
            selfplay_winrate_raw = (float(total_w) + 0.5 * float(total_d)) / float(total_games_played)
            avg_game_plies = float(total_game_plies) / float(max(1, int(total_games_generated)))
            timeout_rate = float(total_timeout_games) / float(max(1, int(total_games_generated)))
            game_draw_rate = float(total_draw_games) / float(max(1, int(total_games_generated)))
            sf_nodes_next = int(sf_nodes_used)
            random_move_prob_next = float(current_rand)
            skill_level_next = int(skill_level_used)
            if pid is not None and (total_w + total_d + total_l) > 0:
                pid_update = pid.observe(wins=total_w, draws=total_d, losses=total_l)
                pid_ema_wr = float(pid_update.ema_winrate)
                sf_nodes_next = int(pid.nodes)
                random_move_prob_next = float(pid.random_move_prob)
                skill_level_next = int(pid.skill_level)
                if sf is not None:
                    if hasattr(sf, "set_nodes"):
                        sf.set_nodes(int(sf_nodes_next))
                    else:
                        setattr(sf, "nodes", int(sf_nodes_next))

            # Persist PID state AFTER observe() so checkpoints/salvage carry the
            # post-iteration difficulty that produced random_move_prob_next.
            if pid is not None:
                try:
                    (ckpt_dir / "pid_state.json").write_text(
                        json.dumps(pid.state_dict(), sort_keys=True, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass

            opp_strength = _opponent_strength(
                random_move_prob=float(current_rand),
                sf_nodes=int(sf_nodes_used),
                skill_level=int(skill_level_used),
                ema_winrate=float(pid_ema_wr),
                min_nodes=int(getattr(pid, "min_nodes", 50)) if pid is not None else 50,
                max_nodes=int(getattr(pid, "max_nodes", 50000)) if pid is not None else 50000,
            )

            # Puzzle evaluation (overspecialization canary).
            puzzle_dict = {}
            if puzzle_suite is not None and puzzle_interval > 0 and (it % puzzle_interval == 0):
                from chess_anti_engine.eval import run_puzzle_eval
                pr = run_puzzle_eval(
                    trainer.model, puzzle_suite,
                    device=device, mcts_simulations=puzzle_sims, rng=rng,
                )
                puzzle_dict = {
                    "puzzle_accuracy": pr.accuracy,
                    "puzzle_correct": pr.correct,
                    "puzzle_total": pr.total,
                }

            iteration_step = int(global_iter)
            try:
                trainer.writer.add_scalar("difficulty/opponent_strength", float(opp_strength), iteration_step)
                trainer.writer.add_scalar("difficulty/random_move_prob", float(current_rand), iteration_step)
                trainer.writer.add_scalar("difficulty/random_move_prob_next", float(random_move_prob_next), iteration_step)
                trainer.writer.add_scalar("difficulty/selfplay_winrate_raw", float(selfplay_winrate_raw), iteration_step)
                trainer.writer.add_scalar("difficulty/pid_ema_winrate", float(pid_ema_wr), iteration_step)
                trainer.writer.add_scalar("selfplay/avg_game_plies", float(avg_game_plies), iteration_step)
                trainer.writer.add_scalar("selfplay/timeout_rate", float(timeout_rate), iteration_step)
                trainer.writer.add_scalar("selfplay/game_draw_rate", float(game_draw_rate), iteration_step)
                trainer.writer.add_scalar("meta/salvage_warmstart_used", float(1 if seed_warmstart_used else 0), iteration_step)
                trainer.writer.add_scalar("meta/salvage_warmstart_slot", float(seed_warmstart_slot), iteration_step)
            except Exception:
                pass

            session.report(
                {
                    "iter": int(iteration_idx),
                    "global_iter": int(global_iter),
                    "replay": len(buf),
                    "test_replay": len(holdout_buf),
                    "positions_added": total_positions,
                    "games_generated": int(total_games_generated),
                    "avg_game_plies": float(avg_game_plies),
                    "timeout_rate": float(timeout_rate),
                    "game_draw_rate": float(game_draw_rate),
                    "shared_samples_ingested": int(imported_samples_this_iter),
                    "shared_trials_selected": int(shared_summary["source_trials_selected"]),
                    "shared_trials_ingested": int(shared_summary["source_trials_ingested"]),
                    "shared_trials_skipped_repeat": int(shared_summary["source_trials_skipped_repeat"]),
                    "shared_shards_loaded": int(shared_summary["source_shards_loaded"]),
                    "distributed_selfplay": int(1 if use_distributed_selfplay else 0),
                    "distributed_workers_per_trial": int(distributed_workers_per_trial),
                    "distributed_stale_games": int(distributed_stale_games),
                    "distributed_stale_positions": int(distributed_stale_positions),
                    "startup_source": str(startup_source),
                    "salvage_warmstart_used": int(1 if seed_warmstart_used else 0),
                    "salvage_warmstart_slot": int(seed_warmstart_slot),
                    "salvage_warmstart_slots_total": int(seed_warmstart_slots_total),
                    "salvage_origin_used": int(1 if salvage_origin_used else 0),
                    "salvage_origin_slot": int(salvage_origin_slot),
                    "salvage_origin_slots_total": int(salvage_origin_slots_total),
                    "train_steps_used": int(steps),
                    "train_target_samples": int(target_sample_budget),
                    "train_window_target_samples": int(window_target_samples),
                    "win": total_w,
                    "draw": total_d,
                    "loss": total_l,
                    "selfplay_winrate_raw": float(selfplay_winrate_raw),
                    "sf_eval_delta6": float(total_sf_d6 / max(1, total_sf_d6_n)) if total_sf_d6_n > 0 else 0.0,
                    "sf_eval_delta6_n": total_sf_d6_n,
                    "sf_nodes": int(sf_nodes_used),
                    "sf_nodes_next": int(sf_nodes_next),
                    "pid_ema_winrate": float(pid_ema_wr),
                    "random_move_prob": float(current_rand),
                    "random_move_prob_next": float(random_move_prob_next),
                    "skill_level": int(skill_level_used),
                    "skill_level_next": int(skill_level_next),
                    "opponent_strength": float(opp_strength),
                    "opt_lr": float(trainer.opt.param_groups[0]["lr"]),
                    "peak_lr": float(getattr(trainer, "_peak_lr", 0.0)),
                    "w_wdl": float(trainer.w_wdl),
                    "w_soft": float(trainer.w_soft),
                    "w_sf_move": float(trainer.w_sf_move),
                    "w_sf_wdl": float(trainer.w_sf_wdl),
                    "optimizer_name": str(config.get("optimizer", "adamw")),
                    "sf_wdl_conf_power": float(trainer.sf_wdl_conf_power),
                    "sf_wdl_draw_scale": float(trainer.sf_wdl_draw_scale),
                    "train_loss": float(metrics.loss) if metrics is not None else 999.0,
                    "train_time_s": float(metrics.train_time_s) if metrics is not None else 0.0,
                    "optimizer_step_time_s": float(metrics.opt_step_time_s) if metrics is not None else 0.0,
                    "trainer_steps_done": int(metrics.train_steps_done) if metrics is not None else 0,
                    "train_samples_seen": int(metrics.train_samples_seen) if metrics is not None else 0,
                    "trainer_steps_per_s": float(
                        metrics.train_steps_done / max(metrics.train_time_s, 1e-9)
                    ) if metrics is not None and metrics.train_time_s > 0.0 else 0.0,
                    "trainer_samples_per_s": float(
                        metrics.train_samples_seen / max(metrics.train_time_s, 1e-9)
                    ) if metrics is not None and metrics.train_time_s > 0.0 else 0.0,
                    "optimizer_steps_per_s": float(
                        metrics.train_steps_done / max(metrics.opt_step_time_s, 1e-9)
                    ) if metrics is not None and metrics.opt_step_time_s > 0.0 else 0.0,
                    "best_loss": float(best_loss),
                    "policy_loss": float(metrics.policy_loss) if metrics is not None else 0.0,
                    "soft_policy_loss": float(metrics.soft_policy_loss) if metrics is not None else 0.0,
                    "future_policy_loss": float(metrics.future_policy_loss) if metrics is not None else 0.0,
                    "wdl_loss": float(metrics.wdl_loss) if metrics is not None else 0.0,
                    "sf_move_loss": float(metrics.sf_move_loss) if metrics is not None else 0.0,
                    "sf_move_acc": float(metrics.sf_move_acc) if metrics is not None else 0.0,
                    "sf_eval_loss": float(metrics.sf_eval_loss) if metrics is not None else 0.0,
                    "categorical_loss": float(metrics.categorical_loss) if metrics is not None else 0.0,
                    "volatility_loss": float(metrics.volatility_loss) if metrics is not None else 0.0,
                    "sf_volatility_loss": float(metrics.sf_volatility_loss) if metrics is not None else 0.0,
                    "moves_left_loss": float(metrics.moves_left_loss) if metrics is not None else 0.0,
                    "gate_passed": int(1 if gate_passed else 0),
                    **eval_dict,
                    **test_dict,
                    **puzzle_dict,
                },
                checkpoint=checkpoint,
            )

            # Best-effort: keep disk usage bounded even when resuming an older
            # experiment that did not have checkpoint retention configured.
            _prune_trial_checkpoints(
                trial_dir=trial_dir,
                keep_last=int(config.get("tune_num_to_keep", 2)),
            )
    finally:
        _stop_worker_processes(distributed_worker_procs)
        if sf is not None:
            sf.close()
        if eval_sf is not None:
            eval_sf.close()
