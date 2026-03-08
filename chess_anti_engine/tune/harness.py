from __future__ import annotations

from pathlib import Path


def _cleanup_old_tune_experiments(*, tune_dir: Path, keep_last: int) -> None:
    """Best-effort pruning of old Ray Tune experiments.

    We keep the newest `keep_last` experiment_state-*.json files and delete
    train_trial_* directories associated with older experiment timestamps.

    This is only intended to run when starting a *fresh* run (i.e. not --resume).
    """

    import re
    import shutil

    keep_last = int(keep_last)
    if keep_last <= 0:
        return
    if not tune_dir.exists():
        return

    # Ray writes these per experiment run.
    state_files = sorted(tune_dir.glob("experiment_state-*.json"))
    if len(state_files) <= keep_last:
        return

    keep = set(state_files[-keep_last:])
    delete_states = [p for p in state_files if p not in keep]

    # Extract timestamps like 2026-02-26_18-51-01 from filenames.
    ts_re = re.compile(r"^experiment_state-(?P<ts>.+)\.json$")
    delete_ts: set[str] = set()
    for p in delete_states:
        m = ts_re.match(p.name)
        if m:
            delete_ts.add(m.group("ts"))

    # Delete associated trial directories and any small aux files.
    for ts in sorted(delete_ts):
        # Remove trial dirs with this timestamp suffix.
        for d in tune_dir.glob(f"train_trial_*{ts}"):
            if d.is_dir():
                print(f"[run_tune] Pruning old Ray Tune trial dir: {d}")
                shutil.rmtree(d, ignore_errors=True)

        # Remove state files for this timestamp.
        for p in [
            tune_dir / f"experiment_state-{ts}.json",
            tune_dir / f"basic-variant-state-{ts}.json",
        ]:
            if p.exists():
                print(f"[run_tune] Pruning old Ray Tune state file: {p}")
                try:
                    p.unlink()
                except Exception:
                    pass

    # Also prune PB2 policy logs that correspond to now-missing trial prefixes.
    # These are small, but they can accumulate.
    prefixes: set[str] = set()
    for d in tune_dir.glob("train_trial_*"):
        if not d.is_dir():
            continue
        # train_trial_<prefix>_...
        parts = d.name.split("_", 2)
        if len(parts) >= 3:
            prefixes.add(parts[1])

    for p in tune_dir.glob("pbt_policy_*.txt"):
        # pbt_policy_<prefix>_<trialid>.txt
        m = re.match(r"^pbt_policy_(?P<prefix>[^_]+)_\d+\.txt$", p.name)
        if m and m.group("prefix") not in prefixes:
            print(f"[run_tune] Pruning old PB2 policy log: {p}")
            try:
                p.unlink()
            except Exception:
                pass


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
        from ray.air import RunConfig
        from ray.air.config import CheckpointConfig
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Ray Tune is required. Install with `pip install -e '.[tune]'`."
        ) from e

    from chess_anti_engine.tune.trainable import train_trial

    work_dir.mkdir(parents=True, exist_ok=True)

    scheduler_name = str(base_config.get("tune_scheduler", "pb2")).lower()
    mode = str(mode)
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

    # ------------------------------------------------------------------
    # GPU / CPU resource allocation
    # ------------------------------------------------------------------
    want_cuda = str(base_config.get("device", "")) == "cuda"
    cpus_per_trial = int(base_config.get("cpus_per_trial", 4))
    gpus_per_trial = float(base_config.get("gpus_per_trial", 1.0 if want_cuda else 0.0))
    resources = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    trainable = tune.with_resources(train_trial, resources)

    # ------------------------------------------------------------------
    # Build param_space
    # ------------------------------------------------------------------
    if scheduler_name == "pb2":
        param_space, scheduler = _build_pb2(base_config, metric=metric, mode=mode)
        search_alg = None  # Scheduler handles search internally.
    elif scheduler_name == "pbt":
        param_space, scheduler = _build_pbt(base_config, metric=metric, mode=mode)
        search_alg = None  # Scheduler handles search internally.
    elif scheduler_name == "gpbt_pl":
        param_space, scheduler = _build_gpbt_pl(base_config, metric=metric, mode=mode)
        search_alg = None  # Scheduler handles search internally.
    elif scheduler_name == "asha":
        param_space, scheduler, search_alg = _build_asha(base_config, metric=metric, mode=mode)
    else:
        raise ValueError(
            f"Unsupported tune_scheduler={scheduler_name!r}. "
            "Expected one of: 'pb2', 'pbt', 'gpbt_pl', 'asha'."
        )

    # ------------------------------------------------------------------
    # Assemble and run Tuner
    # ------------------------------------------------------------------
    tune_config_kwargs: dict = dict(
        num_samples=int(num_samples),
        scheduler=scheduler,
        max_concurrent_trials=int(base_config.get("max_concurrent_trials", num_samples)),
        reuse_actors=True,
    )
    if search_alg is not None:
        tune_config_kwargs["search_alg"] = search_alg

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    experiment_path = work_dir.resolve() / "tune"

    # If we're starting a fresh run (not restoring), prune old experiments to keep
    # disk usage bounded.
    if not resume:
        keep_last_exps = int(base_config.get("tune_keep_last_experiments", 2))
        # We prune to (N-1) before starting a fresh run so that after the new
        # experiment is created we have at most N total.
        _cleanup_old_tune_experiments(
            tune_dir=experiment_path,
            keep_last=max(0, keep_last_exps - 1),
        )

    restored = False
    if resume and experiment_path.exists():
        # Validate experiment state files before attempting restore.
        # Ray silently falls back to a broken 1-trial restart if state is corrupt.
        import glob as _glob
        state_files = sorted(_glob.glob(str(experiment_path / "experiment_state-*.json")))
        state_ok = False
        for sf in reversed(state_files):  # try newest first
            try:
                with open(sf, "r", encoding="utf-8") as fh:
                    import json as _json
                    _json.loads(fh.read())
                state_ok = True
                break
            except Exception:
                import os
                corrupt_name = sf + ".corrupt"
                os.rename(sf, corrupt_name)
                print(f"[run_tune] Renamed corrupt state file: {sf} -> {corrupt_name}")

        if state_ok:
            tuner = tune.Tuner.restore(
                str(experiment_path),
                trainable=trainable,
                param_space=param_space,
                resume_errored=True,
                resume_unfinished=True,
            )
            restored = True
        else:
            print("[run_tune] All experiment state files corrupt; starting fresh run.")

    if not restored:
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

    return tuner.fit()


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


def _build_mutation_param_space(base_config: dict, *, bounds: dict[str, list[float]]):
    from ray import tune

    param_space = {**dict(base_config)}
    for param_name, param_bounds in bounds.items():
        param_space[param_name] = tune.uniform(*param_bounds)

    if bool(base_config.get("search_smolgen", False)):
        param_space["use_smolgen"] = tune.choice([True, False])
    if bool(base_config.get("search_nla", False)):
        param_space["use_nla"] = tune.choice([True, False])
    if bool(base_config.get("search_optimizer", False)):
        param_space["optimizer"] = tune.choice(["nadamw", "soap"])

    return param_space


def _build_pbt_mutations(base_config: dict, *, bounds: dict[str, list[float]]):
    from ray import tune

    mutations: dict[str, object] = {}
    for param_name, param_bounds in bounds.items():
        mutations[param_name] = tune.uniform(*param_bounds)

    if bool(base_config.get("search_smolgen", False)):
        mutations["use_smolgen"] = [True, False]
    if bool(base_config.get("search_nla", False)):
        mutations["use_nla"] = [True, False]
    if bool(base_config.get("search_optimizer", False)):
        mutations["optimizer"] = ["nadamw", "soap"]

    return mutations


def _build_pb2(base_config: dict, *, metric: str, mode: str):
    """PB2: Population-Based Bandits 2 for RL hyperparameter search."""
    try:
        from ray.tune.schedulers.pb2 import PB2
    except ImportError as e:
        raise ImportError(
            "PB2 requires `ray[tune]`. Install with: pip install -e '.[tune]'"
        ) from e

    perturbation_interval = int(base_config.get("pb2_perturbation_interval", 10))
    hyperparam_bounds = _collect_mutation_bounds(base_config)
    param_space = _build_mutation_param_space(base_config, bounds=hyperparam_bounds)

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

    perturbation_interval = int(base_config.get("pb2_perturbation_interval", 10))
    hyperparam_bounds = _collect_mutation_bounds(base_config)
    param_space = _build_mutation_param_space(base_config, bounds=hyperparam_bounds)
    hyperparam_mutations = _build_pbt_mutations(base_config, bounds=hyperparam_bounds)

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric=metric,
        mode=mode,
        perturbation_interval=perturbation_interval,
        hyperparam_mutations=hyperparam_mutations,
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

    perturbation_interval = int(base_config.get("pb2_perturbation_interval", 10))
    hyperparam_bounds = _collect_mutation_bounds(base_config)
    param_space = _build_mutation_param_space(base_config, bounds=hyperparam_bounds)
    hyperparam_mutations = _build_pbt_mutations(base_config, bounds=hyperparam_bounds)

    scheduler = GPBTPairwiseScheduler(
        time_attr="training_iteration",
        metric=metric,
        mode=mode,
        perturbation_interval=perturbation_interval,
        hyperparam_mutations=hyperparam_mutations,
        hyperparam_bounds=hyperparam_bounds,
        pairwise_lr=float(base_config.get("gpbt_pairwise_lr", 0.35)),
        pairwise_momentum=float(base_config.get("gpbt_pairwise_momentum", 0.5)),
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
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.search.optuna import OptunaSearch
        from ray import tune
    except ImportError as e:
        raise ImportError(
            "ASHA requires `ray[tune]` with optuna. "
            "Install with: pip install -e '.[tune]'"
        ) from e

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

    if bool(base_config.get("search_feature_dropout_p", False)):
        param_space["feature_dropout_p"] = tune.choice([0.0, 0.3])
    if bool(base_config.get("search_w_volatility", False)):
        param_space["w_volatility"] = tune.choice([0.0, 0.05, 0.10])
    if bool(base_config.get("search_volatility_source", False)):
        param_space["volatility_source"] = tune.choice(["raw", "search"])

    scheduler = ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=int(base_config.get("iterations", 10)),
        grace_period=max(1, int(base_config.get("iterations", 10)) // 5),
        reduction_factor=2,
    )
    search_alg = OptunaSearch(metric=metric, mode=mode)

    return param_space, scheduler, search_alg
