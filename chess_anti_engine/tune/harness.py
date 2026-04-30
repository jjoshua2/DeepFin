from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from chess_anti_engine.tune._utils import (
    resolve_local_override_root,
    terminate_process as _terminate_process,
)
from chess_anti_engine.utils.atomic import atomic_write_text


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


# Atomic-write tmp prefixes used by save_local_shard_arrays + server compaction
# (mirrors `is_tmp_shard_name` in chess_anti_engine.replay.shard). Anchored
# globs only — Ray's _ExcludingLocalFilesystem matches via fnmatch on the path
# relative to the sync root, so an unanchored "._tmp_*" wouldn't fire on
# nested `selfplay_shards/._tmp_*.zarr/...` paths.
#
# We also exclude the entire `selfplay_shards/` tree. Ray's experiment-state
# sync iterates these in pyarrow's `_copy_files_selector`, which races against
# `enforce_window` deletes — even completed `shard_000123.zarr/.zattrs` paths
# get raced if window enforcement deletes them between enumerate and open.
# Selfplay shards are local-only (worker writes → trainable consumes); Ray's
# experiment state has no business with them.
_TMP_SHARD_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "**/._tmp_*",
    "**/tmp_*",
    "._tmp_*",
    "tmp_*",
    "**/selfplay_shards/**",
    "selfplay_shards/**",
    # `chess_anti_engine.utils.atomic.atomic_write` uses
    # `<final>.tmp.<pid>.<uuid>` then atomic rename. Ray's enumerator races
    # the rename → FileNotFoundError on .tmp.* paths.
    "**/*.tmp.*",
    "*.tmp.*",
)


def _patch_ray_artifact_sync_excludes() -> None:
    """Make Ray's experiment-state syncer skip in-flight atomic-rename shards.

    Ray's experiment-state checkpoint syncs the trial dir to persistent storage
    via `pyarrow.fs.copy_files`, which enumerates files then opens each one.
    When the enumeration captures a `._tmp_<pid>_*.zarr` path that is then
    atomic-renamed away by `save_local_shard_arrays`, the open fails with
    `FileNotFoundError`. The data is fine (the rename is atomic; the next sync
    sees the final shard) but the syncer raises and pollutes logs.

    Ray's `SyncConfig` does not expose excludes, but `_upload_to_fs_path`
    accepts an `exclude` kwarg that switches to a fsspec-based walker which
    filters before enumerating — eliminating the race entirely. Wrap that
    function so every call carries our temp prefixes in `exclude`.
    """
    try:
        from ray.train._internal import storage as _ray_storage
    except Exception:
        return
    orig = _ray_storage._upload_to_fs_path
    if getattr(orig, "_chess_tmp_exclude_patched", False):
        return

    def _upload_to_fs_path_with_excludes(local_path, fs, fs_path, exclude=None):
        merged = list(exclude or [])
        merged.extend(_TMP_SHARD_EXCLUDE_PATTERNS)
        return orig(local_path, fs, fs_path, exclude=merged)

    _upload_to_fs_path_with_excludes._chess_tmp_exclude_patched = True  # pyright: ignore[reportFunctionMemberAccess]
    _ray_storage._upload_to_fs_path = _upload_to_fs_path_with_excludes


def _wait_for_server_ready(
    *,
    base_url: str,
    proc: subprocess.Popen[bytes],
    timeout_s: float = 20.0,
) -> None:
    deadline = time.time() + float(timeout_s)
    url = str(base_url).rstrip("/") + "/v1/update_info"
    last_err = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"distributed Tune server exited early with code {proc.returncode}")
        try:
            with urllib_request.urlopen(url, timeout=1.0) as resp:
                if int(getattr(resp, "status", 200)) in (200, 404):
                    return
        except urllib_error.HTTPError as e:
            if int(getattr(e, "code", 0)) in (200, 404):
                return
            last_err = f"HTTP {e.code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.25)
    raise RuntimeError(f"distributed Tune server did not become ready at {url}: {last_err}")


def _prepare_distributed_worker_auth(
    *, server_root: Path, config: dict | None = None
) -> tuple[str, Path]:
    from chess_anti_engine.server.auth import load_users, upsert_user

    cfg = config or {}
    username = str(cfg.get("distributed_worker_username", "") or "").strip()
    password = str(cfg.get("distributed_worker_password", "") or "").strip()

    if not username or not password:
        raise RuntimeError(
            "distributed_worker_username and distributed_worker_password must be set in config. "
            "Add a user first: python -m chess_anti_engine.server.manage_users add <username>"
        )

  # First-time provisioning only — if the user already exists, leave
  # their record alone. This preserves the disabled flag (so an operator
  # can disable a compromised account without startup silently re-
  # enabling it) and any rotated password from manage_users.py (so yaml
  # drift can't overwrite the authoritative admin change).
  # (Codex adversarial review.)
    users_path = server_root / "users.json"
    user_existed = username in load_users(users_path)
    if not user_existed:
        upsert_user(users_path, username=username, password=password)

    password_file = server_root / f"{username}.password"
  # Symmetric to the upsert: only write the .password file on first
  # provisioning. If admin rotated the password via manage_users, the
  # yaml password is stale; rewriting it would cause workers to read the
  # stale value and fail auth against the live users.json.
    if not user_existed and not password_file.exists():
        password_file.write_text(password + "\n", encoding="utf-8")
        try:
            password_file.chmod(0o600)
        except OSError:
            pass  # Windows / non-POSIX filesystem — chmod is best-effort
    return username, password_file


def _delete_obsolete_tune_files(*, tune_dir: Path, ts: str) -> None:
    """Remove train_trial dirs + state aux files for one experiment timestamp."""
    import shutil

    for d in tune_dir.glob(f"train_trial_*{ts}"):
        if d.is_dir():
            print(f"[run_tune] Pruning old Ray Tune trial dir: {d}")
            shutil.rmtree(d, ignore_errors=True)
    for p in (
        tune_dir / f"experiment_state-{ts}.json",
        tune_dir / f"basic-variant-state-{ts}.json",
    ):
        if p.exists():
            print(f"[run_tune] Pruning old Ray Tune state file: {p}")
            try:
                p.unlink()
            except Exception:
                pass


def _prune_orphan_pb2_policy_logs(*, tune_dir: Path) -> None:
    """Drop pbt_policy_<prefix>_<id>.txt files whose prefix has no surviving
    train_trial_<prefix>_* directory."""
    import re

    live_prefixes: set[str] = set()
    for d in tune_dir.glob("train_trial_*"):
        if not d.is_dir():
            continue
        parts = d.name.split("_", 2)
        if len(parts) >= 3:
            live_prefixes.add(parts[1])

    for p in tune_dir.glob("pbt_policy_*.txt"):
        m = re.match(r"^pbt_policy_(?P<prefix>[^_]+)_\d+\.txt$", p.name)
        if m and m.group("prefix") not in live_prefixes:
            print(f"[run_tune] Pruning old PB2 policy log: {p}")
            try:
                p.unlink()
            except Exception:
                pass


def _cleanup_old_tune_experiments(*, tune_dir: Path, keep_last: int) -> None:
    """Best-effort pruning of old Ray Tune experiments.

    Keeps the newest `keep_last` experiment_state-*.json files and deletes
    train_trial_* directories associated with older experiment timestamps.
    Only intended to run on fresh starts (not --resume).
    """
    import re

    keep_last = int(keep_last)
    if keep_last <= 0 or not tune_dir.exists():
        return

    state_files = sorted(tune_dir.glob("experiment_state-*.json"))
    if len(state_files) <= keep_last:
        return

    keep = set(state_files[-keep_last:])
    delete_states = [p for p in state_files if p not in keep]

    ts_re = re.compile(r"^experiment_state-(?P<ts>.+)\.json$")
    delete_ts: set[str] = set()
    for p in delete_states:
        m = ts_re.match(p.name)
        if m:
            delete_ts.add(m.group("ts"))

    for ts in sorted(delete_ts):
        _delete_obsolete_tune_files(tune_dir=tune_dir, ts=ts)

    _prune_orphan_pb2_policy_logs(tune_dir=tune_dir)


# NB: harness's ``work_dir`` is the run root, while the canonical
# ``resolve_local_override_root`` in ``tune/_utils.py`` expects the *tune*
# subdir and derives ``run_root = tune_dir.parent``. Pass ``work_dir/"tune"``
# to bridge the conventions.


def _load_trial_config_from_state_entry(entry: object) -> tuple[dict[str, object] | None, bool]:
    if isinstance(entry, (list, tuple)) and entry:
        payload = entry[0]
    elif isinstance(entry, dict):
        payload = entry
    else:
        return None, False
    if isinstance(payload, str):
        trial = json.loads(payload)
        return trial if isinstance(trial, dict) else None, True
    return payload if isinstance(payload, dict) else None, False


def _patch_experiment_state_for_resume(
    *,
    state_file: Path,
    param_space: dict[str, object],
) -> tuple[set[str], set[str], set[str]]:
    """Patch saved trial configs with current YAML keys so Ray's resume validation passes.

    Returns ``(added_keys, skipped_keys, post_patch_saved_keys)``. The third
    value is the union of trial-config keys after patching — callers use it to
    drop param_space keys that aren't in any saved trial config (Ray rejects
    those on restore).
    """
    with state_file.open("r", encoding="utf-8") as fh:
        experiment_state = json.load(fh)

    trial_data = experiment_state.get("trial_data", [])
    if not isinstance(trial_data, list):
        return set(), set(), set()

    added_keys: set[str] = set()
    skipped_keys: set[str] = set()
    saved_keys: set[str] = set()
    changed = False
    for idx, entry in enumerate(trial_data):
        trial, payload_is_json = _load_trial_config_from_state_entry(entry)
        if not isinstance(trial, dict):
            continue
        cfg = trial.get("config")
        if not isinstance(cfg, dict):
            continue

        trial_changed = False
        for key, value in param_space.items():
            if key in cfg:
                continue
            try:
                json.dumps(value)
            except TypeError:
                skipped_keys.add(str(key))
                continue
            cfg[key] = value
            added_keys.add(str(key))
            trial_changed = True

        saved_keys.update(str(k) for k in cfg.keys())
        if not trial_changed:
            continue
        changed = True
        if isinstance(entry, list) and entry and payload_is_json:
            entry[0] = json.dumps(trial, separators=(",", ":"))
        elif isinstance(entry, dict):
            trial_data[idx] = trial

    if changed:
        atomic_write_text(state_file, json.dumps(experiment_state) + "\n")
    return added_keys, skipped_keys, saved_keys


def _launch_distributed_server(
    *, base_config: dict, work_dir: Path
) -> subprocess.Popen[bytes] | None:
    """Launch the trial server subprocess if distributed mode is enabled.

    Mutates ``base_config`` in place with resolved server URLs and credentials
    so trainable actors can connect.
    """
    if int(base_config.get("distributed_workers_per_trial", 0)) <= 0:
        return None

    server_root_override = str(base_config.get("distributed_server_root_override", "")).strip()
    if server_root_override:
        server_root = resolve_local_override_root(
            raw_root=server_root_override,
            tune_work_dir=work_dir / "tune",
            suffix="server",
        ).resolve()
    else:
        server_root = work_dir.resolve() / "server"
    server_root.mkdir(parents=True, exist_ok=True)
    server_log = server_root / "server.log"
    port = int(base_config.get("distributed_server_port", 0)) or _pick_free_port()
    host = str(base_config.get("distributed_server_host", "127.0.0.1")).strip() or "127.0.0.1"
    public_url = str(base_config.get("distributed_server_public_url", "")).strip()
    ready_host = "127.0.0.1" if host == "0.0.0.0" else host
    ready_url = f"http://{ready_host}:{port}"
    base_url = public_url or ready_url
    username, password_file = _prepare_distributed_worker_auth(server_root=server_root, config=base_config)

    cmd = [
        sys.executable,
        "-m",
        "chess_anti_engine.server.run_server",
        "--host",
        host,
        "--port",
        str(port),
        "--server-root",
        str(server_root),
        "--min-workers-per-trial",
        str(int(base_config.get("distributed_min_workers_per_trial", 1))),
        "--max-worker-delta-per-rebalance",
        str(int(base_config.get("distributed_max_worker_delta_per_rebalance", 1))),
        "--upload-compact-shard-size",
        str(int(base_config.get("distributed_upload_compact_shard_size", base_config.get("shard_size", 2000)))),
        "--upload-compact-max-age-seconds",
        str(float(base_config.get("distributed_upload_compact_max_age_seconds", 90.0))),
    ]
    for cfg_key, flag in (
        ("opening_book_path", "--opening-book-path"),
        ("opening_book_path_2", "--opening-book-path-2"),
    ):
        book = base_config.get(cfg_key)
        if isinstance(book, str) and book.strip():
            cmd.extend([flag, book.strip()])

    from chess_anti_engine.tune.distributed_runtime import _spawn_with_reap
    server_proc = _spawn_with_reap(
        cmd=cmd,
        log_path=server_log,
        reap_module="chess_anti_engine.server.run_server",
        reap_terms=["--server-root", str(server_root)],
        reap_label="distributed Tune servers",
    )

    try:
        _wait_for_server_ready(base_url=ready_url, proc=server_proc)
    except Exception:
        _terminate_process(server_proc)
        raise

    base_config.update({
        "distributed_server_root": str(server_root),
        "distributed_server_url": str(ready_url),
        "distributed_server_public_url": str(base_url),
        "distributed_worker_username": str(username),
        "distributed_worker_password_file": str(password_file),
    })
    return server_proc


def _maybe_launch_shared_broker(
    *, base_config: dict, work_dir: Path
) -> subprocess.Popen[bytes] | None:
    """Launch shared broker. Must be called *after* ``_launch_distributed_server``
    populates ``base_config['distributed_server_root']``.
    """
    if not (
        int(base_config.get("distributed_workers_per_trial", 0)) > 0
        and bool(base_config.get("distributed_inference_broker_enabled", False))
        and bool(base_config.get("distributed_inference_shared_broker", False))
    ):
        return None
    from chess_anti_engine.tune.distributed_runtime import launch_shared_inference_broker
    return launch_shared_inference_broker(
        config=base_config,
        server_root=Path(base_config.get("distributed_server_root", str(work_dir / "server"))),
    )


def _build_scheduler_and_param_space(
    *, scheduler_name: str, base_config: dict, metric: str, mode: str
):
    """Dispatch to the per-scheduler builder. Returns (param_space, scheduler, search_alg)."""
    if scheduler_name == "pb2":
        param_space, scheduler = _build_pb2(base_config, metric=metric, mode=mode)
        return param_space, scheduler, None
    if scheduler_name == "pbt":
        param_space, scheduler = _build_pbt(base_config, metric=metric, mode=mode)
        return param_space, scheduler, None
    if scheduler_name == "gpbt_pl":
        param_space, scheduler = _build_gpbt_pl(base_config, metric=metric, mode=mode)
        return param_space, scheduler, None
    if scheduler_name == "asha":
        return _build_asha(base_config, metric=metric, mode=mode)
    if scheduler_name == "none":
        return {**base_config}, None, None
    raise ValueError(
        f"Unsupported tune_scheduler={scheduler_name!r}. "
        "Expected one of: 'pb2', 'pbt', 'gpbt_pl', 'asha', 'none'."
    )


def _find_valid_experiment_state(experiment_path: Path) -> Path | None:
    """Newest non-corrupt experiment_state-*.json. Renames corrupt ones to .corrupt.

    Ray silently falls back to a broken 1-trial restart if state is corrupt,
    so we validate JSON parseability before handing the file to Tuner.restore.
    """
    import glob as _glob
    state_files = sorted(_glob.glob(str(experiment_path / "experiment_state-*.json")))
    for sf in reversed(state_files):  # newest first
        try:
            with open(sf, encoding="utf-8") as fh:
                json.loads(fh.read())
            return Path(sf)
        except Exception:
            corrupt_name = sf + ".corrupt"
            os.rename(sf, corrupt_name)
            print(f"[run_tune] Renamed corrupt state file: {sf} -> {corrupt_name}")
    return None


def _filter_param_space_against_tuner_pkl(
    param_space: dict, *, experiment_path: Path
) -> dict:
    """Strip/pad keys to match ``__flattened_param_space_keys`` in tuner.pkl.

    Ray's _validate_param_space_on_restore reads from the pickle, not the
    experiment_state JSON we patched. Any key not in the pkl triggers
    ValueError; missing keys must be padded with None.
    """
    import pickle as _pickle
    out = dict(param_space)
    try:
        with (experiment_path / "tuner.pkl").open("rb") as fh:
            tuner_state = _pickle.load(fh)
    except (FileNotFoundError, _pickle.UnpicklingError, EOFError) as exc:
        print(f"[run_tune] Could not read tuner.pkl for key filtering (non-fatal): {exc}")
        return out
    ray_keys = set(tuner_state.get("__flattened_param_space_keys") or [])
    if not ray_keys:
        return out
    drop = set(out.keys()) - ray_keys
    if drop:
        out = {k: v for k, v in out.items() if k not in drop}
        print(f"[run_tune] Stripping keys absent from tuner.pkl for Ray validation: {sorted(drop)}")
    missing = ray_keys - set(out.keys())
    if missing:
        for k in missing:
            out[k] = None
        print(f"[run_tune] Padding removed keys with None for Ray validation: {sorted(missing)}")
    return out


def _resolve_resume_param_space(
    *, param_space: dict, valid_state_file: Path, experiment_path: Path
) -> dict:
    """Patch param_space so Ray's restore validation accepts our current keys."""
    added_keys, skipped_keys, saved_keys = _patch_experiment_state_for_resume(
        state_file=valid_state_file,
        param_space=param_space,
    )
    if added_keys:
        print(f"[run_tune] Added {len(added_keys)} new config keys to restored trial state: {sorted(added_keys)}")
    out = _filter_param_space_against_tuner_pkl(param_space, experiment_path=experiment_path)
    if skipped_keys:
        out = {k: v for k, v in out.items() if k not in skipped_keys}
        print(f"[run_tune] Skipping non-JSON param_space keys for resume: {sorted(skipped_keys)}")
    extra = set(out.keys()) - saved_keys
    if extra:
        out = {k: v for k, v in out.items() if k not in extra}
        print(f"[run_tune] Stripping unresolved param_space keys for resume: {sorted(extra)}")
    return out


def _hotpatch_scheduler_bounds(*, tuner, scheduler) -> None:
    """Overwrite restored scheduler's hyperparam_bounds with current YAML.

    Ray unpickles the old scheduler (with old bounds baked in) on restore, so
    any pb2_bounds_* changes in the YAML are silently ignored. We reach into
    the live scheduler and overwrite its bounds dict so the very next
    exploit/explore step uses the updated ranges.
    """
    if scheduler is None:
        return
    try:
  # Ray's `Tuner._local_tuner._tune_config.scheduler` is a private surface
  # — pyright has no typings for these attrs.
        live_sched = tuner._local_tuner._tune_config.scheduler
    except AttributeError as exc:
        print(f"[run_tune] Could not hotpatch scheduler bounds (non-fatal): {exc}")
        return
    if not (hasattr(live_sched, "_hyperparam_bounds") and hasattr(scheduler, "_hyperparam_bounds")):
        return
    old = dict(live_sched._hyperparam_bounds)
    live_sched._hyperparam_bounds = dict(scheduler._hyperparam_bounds)
    changed = {k: (old.get(k), v) for k, v in scheduler._hyperparam_bounds.items() if old.get(k) != v}
    if changed:
        print(f"[run_tune] Hotpatched scheduler bounds (old→new): {changed}")
    else:
        print("[run_tune] Scheduler bounds unchanged after restore.")


def run_tune(
    *,
    base_config: dict,
    work_dir: Path,
    num_samples: int,
    metric: str = "train_loss",
    mode: str = "min",
    resume: bool = False,
):
    """Run the Ray Tune harness.

    Supports four schedulers, selectable via base_config["tune_scheduler"]:

    "pb2" (default, recommended for RL)
        Population-Based Bandits 2 (Parker-Holder et al., NeurIPS 2020).
        Uses a Gaussian Process bandit to guide the exploit+explore step, so
        it's more sample-efficient than vanilla PBT with random perturbation.
        Designed for non-stationary objectives — exactly what self-play RL
        produces.  Continuously adapts hyperparameters during training rather
        than searching once upfront.

        Requires: all trials running simultaneously (max_concurrent_trials ≥
        num_samples), otherwise PB2 degenerates to sequential random search.

    "pbt"
        Vanilla Population-Based Training. Uses the same mutable parameter
        bounds as PB2, but perturbs them with exploit/explore heuristics rather
        than a GP bandit. Useful when you want the same config surface as PB2
        with simpler, fully built-in Ray behavior.

    "gpbt_pl"
        Ray-native pairwise-learning adaptation inspired by gpbt-pl. Keeps the
        current PBT checkpoint/exploit lifecycle, but uses weighted pairwise
        donor selection and momentum-based pairwise hyperparameter updates.

    "asha" (legacy)
        ASHA + Optuna search.  Assumes a stationary objective; valid for
        binary architecture ablations on a fixed dataset, but wrong for
        RL-coupled hyperparameters (diff_focus_*, temperature, LR schedule).

    GPU allocation:
        Set gpus_per_trial < 1 (e.g. 0.1) to run multiple trials on the same
        GPU.  On an RTX 5090 with a small model, 10 trials at gpu=0.1 each
        fits comfortably in 32 GB VRAM.
    """
    try:
        import ray
        from ray import tune
        from ray.tune import CheckpointConfig, RunConfig
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Ray Tune is required. Install with `pip install -e '.[tune]'`."
        ) from e

    _patch_ray_artifact_sync_excludes()

    from chess_anti_engine.tune.trainable import train_trial

    work_dir.mkdir(parents=True, exist_ok=True)
    base_config = dict(base_config)
    mode = str(mode)
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

    server_proc = _launch_distributed_server(base_config=base_config, work_dir=work_dir)
    shared_broker_proc = _maybe_launch_shared_broker(base_config=base_config, work_dir=work_dir)

    want_cuda = str(base_config.get("device", "")) == "cuda"
    resources = {
        "cpu": int(base_config.get("cpus_per_trial", 4)),
        "gpu": float(base_config.get("gpus_per_trial", 1.0 if want_cuda else 0.0)),
    }
    trainable = tune.with_resources(train_trial, resources)

    scheduler_name = str(base_config.get("tune_scheduler", "pb2")).lower()
    param_space, scheduler, search_alg = _build_scheduler_and_param_space(
        scheduler_name=scheduler_name, base_config=base_config, metric=metric, mode=mode,
    )

    tune_config_kwargs: dict = dict(
        num_samples=int(num_samples),
        scheduler=scheduler,
        max_concurrent_trials=int(base_config.get("max_concurrent_trials", num_samples)),
  # The tune path uses the function-style trainable API, which does not
  # implement reset_config(). Reusing actors under GPBT/PBT causes Ray to
  # error during perturb/exploit when it attempts an in-place config reset.
        reuse_actors=False,
    )
    if search_alg is not None:
        tune_config_kwargs["search_alg"] = search_alg

    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("RAY_memory_usage_threshold", "0.98")

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    experiment_path = work_dir.resolve() / "tune"
    if not resume:
        keep_last_exps = int(base_config.get("tune_keep_last_experiments", 2))
  # Prune to (N-1) before the new experiment is created so we end at <=N total.
        _cleanup_old_tune_experiments(
            tune_dir=experiment_path,
            keep_last=max(0, keep_last_exps - 1),
        )

    valid_state_file = (
        _find_valid_experiment_state(experiment_path)
        if resume and experiment_path.exists()
        else None
    )

    if valid_state_file is not None:
        restore_param_space = _resolve_resume_param_space(
            param_space=param_space,
            valid_state_file=valid_state_file,
            experiment_path=experiment_path,
        )
        tuner = tune.Tuner.restore(
            str(experiment_path),
            trainable=trainable,
            param_space=restore_param_space,
            resume_errored=True,
            resume_unfinished=True,
        )
        _hotpatch_scheduler_bounds(tuner=tuner, scheduler=scheduler)
    else:
        if resume and experiment_path.exists():
            print("[run_tune] All experiment state files corrupt; starting fresh run.")
        ckpt_keep = int(base_config.get("tune_num_to_keep", 2))
        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(**tune_config_kwargs),
            run_config=RunConfig(
                name="tune",
                storage_path=str(work_dir.resolve()),
                checkpoint_config=CheckpointConfig(num_to_keep=ckpt_keep),
            ),
        )

    try:
        return tuner.fit()
    finally:
        _terminate_process(shared_broker_proc)
        _terminate_process(server_proc)


# ---------------------------------------------------------------------------
# PB2 scheduler + search space
# ---------------------------------------------------------------------------

def _collect_mutation_bounds(base_config: dict) -> dict[str, list[float]]:
    """Build scheduler bounds from pb2_bounds_* config keys.

    We intentionally keep one scheduler-neutral bounds surface so the user can
    switch between PB2 and vanilla PBT without maintaining separate configs.
    """

    bounds_out: dict[str, list[float]] = {}
    for cfg_key, bounds in base_config.items():
        if not str(cfg_key).startswith("pb2_bounds_"):
            continue
        param_name = str(cfg_key)[len("pb2_bounds_"):]
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            bounds_out[param_name] = [float(bounds[0]), float(bounds[1])]
    return bounds_out


def _optimizer_candidates_from_config(base_config: dict, *, include_nadamw: bool = True) -> list[str]:
    default = ["nadamw", "adamw", "muon", "cosmos", "cosmos_fast", "soap"]
    if not include_nadamw:
        default = ["adamw", "muon", "cosmos", "cosmos_fast", "soap"]
    raw = base_config.get("search_optimizer_choices")
    if isinstance(raw, (list, tuple)):
        vals = [str(v).strip().lower() for v in raw if str(v).strip()]
        vals = [v for i, v in enumerate(vals) if v in default and v not in vals[:i]]
        if vals:
            return vals
    return list(default)


def _build_mutation_param_space(base_config: dict, *, bounds: dict[str, list[float]]):
    from ray import tune

    optimizer_candidates = _optimizer_candidates_from_config(base_config, include_nadamw=True)
    param_space = {**dict(base_config)}
    for param_name, param_bounds in bounds.items():
        param_space[param_name] = tune.uniform(*param_bounds)

    if base_config.get("search_smolgen", False):
        param_space["use_smolgen"] = tune.choice([True, False])
    if base_config.get("search_nla", False):
        param_space["use_nla"] = tune.choice([True, False])
    if base_config.get("search_optimizer", False):
        param_space["optimizer"] = tune.choice(optimizer_candidates)

    return param_space


def _build_pbt_mutations(base_config: dict, *, bounds: dict[str, list[float]]):
    from ray import tune

    optimizer_candidates = _optimizer_candidates_from_config(base_config, include_nadamw=True)
    mutations: dict[str, object] = {}
    for param_name, param_bounds in bounds.items():
        mutations[param_name] = tune.uniform(*param_bounds)

    if base_config.get("search_smolgen", False):
        mutations["use_smolgen"] = [True, False]
    if base_config.get("search_nla", False):
        mutations["use_nla"] = [True, False]
    if base_config.get("search_optimizer", False):
        mutations["optimizer"] = list(optimizer_candidates)

    return mutations


def _pbt_family_setup(base_config: dict) -> tuple[int, dict, dict, dict]:
    """Shared setup for PBT-family schedulers.

    Returns ``(perturbation_interval, hyperparam_bounds, param_space, mutations)``.
    PB2 ignores ``mutations`` (it perturbs along ``hyperparam_bounds``
    directly) but it's cheap to compute, and lets the three builders
    share a single setup block.
    """
    perturbation_interval = int(base_config.get("pb2_perturbation_interval", 10))
    hyperparam_bounds = _collect_mutation_bounds(base_config)
    param_space = _build_mutation_param_space(base_config, bounds=hyperparam_bounds)
    mutations = _build_pbt_mutations(base_config, bounds=hyperparam_bounds)
    return perturbation_interval, hyperparam_bounds, param_space, mutations


def _build_pb2(base_config: dict, *, metric: str, mode: str):
    """PB2: Population-Based Bandits 2 for RL hyperparameter search."""
    try:
        from ray.tune.schedulers.pb2 import PB2
    except ImportError as e:
        raise ImportError(
            "PB2 requires `ray[tune]`. Install with: pip install -e '.[tune]'"
        ) from e

    perturbation_interval, hyperparam_bounds, param_space, _ = _pbt_family_setup(base_config)
    scheduler = PB2(
        time_attr="training_iteration",
        metric=metric,
        mode=mode,
        perturbation_interval=perturbation_interval,
        hyperparam_bounds=hyperparam_bounds,
    )
    return param_space, scheduler


def _build_pbt(base_config: dict, *, metric: str, mode: str):
    """Vanilla Population-Based Training using the same mutable config surface as PB2."""
    try:
        from ray.tune.schedulers import PopulationBasedTraining
    except ImportError as e:
        raise ImportError(
            "PBT requires `ray[tune]`. Install with: pip install -e '.[tune]'"
        ) from e

    perturbation_interval, _, param_space, mutations = _pbt_family_setup(base_config)
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric=metric,
        mode=mode,
        perturbation_interval=perturbation_interval,
        hyperparam_mutations=mutations,
        synch=bool(base_config.get("pbt_synch", False)),
    )
    return param_space, scheduler


def _build_gpbt_pl(base_config: dict, *, metric: str, mode: str):
    """Pairwise-learning PBT adaptation using the same mutable config surface as PB2."""
    try:
        from chess_anti_engine.tune.gpbt import GPBTPairwiseScheduler
    except ImportError as e:
        raise ImportError(
            "GPBT-PL scheduler could not be imported from chess_anti_engine.tune.gpbt."
        ) from e

    perturbation_interval, hyperparam_bounds, param_space, mutations = _pbt_family_setup(base_config)
    scheduler = GPBTPairwiseScheduler(
        time_attr="training_iteration",
        metric=metric,
        mode=mode,
        perturbation_interval=perturbation_interval,
        hyperparam_mutations=mutations,
        hyperparam_bounds=hyperparam_bounds,
        trial_inertia_weight=float(base_config.get("gpbt_inertia_weight", 1.0)),
        trial_winner_weight=float(base_config.get("gpbt_winner_weight", 1.0)),
        quantile_fraction=float(base_config.get("gpbt_quantile_fraction", 0.25)),
        resample_probability=float(base_config.get("gpbt_resample_probability", 0.05)),
        synch=bool(base_config.get("pbt_synch", False)),
    )
    return param_space, scheduler


# ---------------------------------------------------------------------------
# ASHA scheduler + search space (legacy, for architecture ablations)
# ---------------------------------------------------------------------------

def _build_asha(base_config: dict, *, metric: str, mode: str):
    """ASHA + Optuna: best for binary architecture ablations on a fixed objective.

    NOT recommended for RL-coupled hyperparameters (diff_focus_*, temperature,
    loss weights) because ASHA assumes a stationary objective.
    """
    try:
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.search.optuna import OptunaSearch
    except ImportError as e:
        raise ImportError(
            "ASHA requires `ray[tune]` with optuna. "
            "Install with: pip install -e '.[tune]'"
        ) from e

    optimizer_candidates = _optimizer_candidates_from_config(base_config, include_nadamw=False)
    if base_config.get("asha_optimizer_only", False):
        param_space = {**dict(base_config)}
        if base_config.get("search_optimizer", False):
            param_space["optimizer"] = tune.grid_search(list(optimizer_candidates))
        repeats = max(1, int(base_config.get("asha_optimizer_repeats", 1)))
        base_seed = int(base_config.get("seed", 0))
        if repeats > 1:
            param_space["seed"] = tune.grid_search([base_seed + i for i in range(repeats)])
        search_alg = None
    else:
        param_space = {
            **dict(base_config),
            "lr": tune.loguniform(1e-5, 3e-3),
            "embed_dim": tune.choice([128, 192, 256, 320]),
            "num_layers": tune.choice([4, 6, 8]),
            "num_heads": tune.choice([4, 8]),
            "ffn_mult": tune.choice([2, 3, 4]),
            "use_smolgen": tune.choice([True, False]),
            "use_nla": tune.choice([True, False]),
            "temperature": tune.uniform(0.8, 1.6),
            "playout_cap_fraction": tune.uniform(0.1, 0.5),
            "sf_policy_temp": tune.uniform(0.15, 0.8),
            "sf_policy_label_smooth": tune.uniform(0.0, 0.15),
        }

        if base_config.get("search_feature_dropout_p", False):
            param_space["feature_dropout_p"] = tune.choice([0.0, 0.3])
        if base_config.get("search_w_volatility", False):
            param_space["w_volatility"] = tune.choice([0.0, 0.05, 0.10])
        if base_config.get("search_volatility_source", False):
            param_space["volatility_source"] = tune.choice(["raw", "search"])
        if base_config.get("search_optimizer", False):
            param_space["optimizer"] = tune.choice(optimizer_candidates)
        search_alg = OptunaSearch(metric=metric, mode=mode)

    scheduler = ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=int(base_config.get("iterations", 10)),
        grace_period=max(1, int(base_config.get("iterations", 10)) // 5),
        reduction_factor=2,
    )

    return param_space, scheduler, search_alg
