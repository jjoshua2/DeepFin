#!/usr/bin/env python3
"""Plain self-play RL loop: split-openings, init, collect, train, evaluate, run.

See docs/simple_rl_loop.md. Every stage is resumable at its boundary: a stage
directory with a manifest.json is complete and is skipped by ``run``; one
without is incomplete and is redone whole.

    PYTHONPATH=. python scripts/simple_rl_loop.py split-openings --out <dir> \
        --book <8move.pgn.zip>:16 --book <2move.pgn.zip>:4
    PYTHONPATH=. python scripts/simple_rl_loop.py run --run-dir <dir> \
        --openings-dir <dir> --init-checkpoint <sl_base.pt> --generations 10 \
        --sf-path <stockfish> --syzygy <tb paths> --device cuda
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_anti_engine.simple_rl.collect import (
    CollectSettings,
    collect_generation,
    default_collect_shape,
    validate_settings,
)
from chess_anti_engine.simple_rl.common import (
    blocking_markers,
    input_ref,
    read_manifest,
    sha256_tree,
    stage_done,
    write_json_atomic,
    write_manifest,
)
from chess_anti_engine.simple_rl.evaluate import (
    TRIPWIRE_FIELDS,
    EvalSettings,
    check_tripwires,
    evaluate_generation,
    format_table,
    tripwire_table,
)
from chess_anti_engine.simple_rl.learn import (
    MATRIX_OPTIMIZER_SCOPES, MODEL_FILE, TrainSettings, check_snapshot_plan, lineage_capped_collects,
    planned_phase_updates, planned_snapshot_updates, preflight_init_optimizer, refuse_capped_collects,
    train_generation, trainer_kwargs,
)
from chess_anti_engine.simple_rl.lineage import later_sealed, unseal_after, verify_before, verify_stage
from chess_anti_engine.simple_rl.openings import split_openings
from chess_anti_engine.simple_rl.reference import precheck_source, reference_collect

EXIT_TRIPWIRE = 3
EXIT_BLOCKED = 4
RUN_CONFIG = "run_config.json"


def gen_dir(run_dir: Path, g: int) -> Path:
    return Path(run_dir) / f"gen_{g:04d}"


def model_path(run_dir: Path, g: int) -> Path:
    return gen_dir(run_dir, 0) / MODEL_FILE if g == 0 else gen_dir(run_dir, g) / "train" / MODEL_FILE


def sealed_model(run_dir: Path, g: int) -> Path:
    """Generation g's model, only if its stage manifest is present and its
    recorded sha256 is the file on disk (a kill between the weights and the
    manifest of a redo, or a stray overwrite, is refused rather than used)."""
    path = model_path(run_dir, g)
    manifest = read_manifest(path.parent)
    if manifest is None:
        raise SystemExit(f"generation {g} model is not sealed (no manifest beside {path}); redo that stage")
    want = manifest["outputs"]["model"]["sha256"]
    if not path.is_file() or sha256_tree(path) != want:
        raise SystemExit(f"{path} does not match the sha256 its manifest recorded ({want}); redo that stage")
    return path


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def init_run(run_dir: Path, *, init_checkpoint: str | None, random_init: bool,
             model_config: str | None, arch_from: str | None, seed: int) -> Path:
    """Create gen_0000/model.pt (the generation-0 actor), once."""
    import torch

    from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig
    from chess_anti_engine.simple_rl.learn import build_trainer
    from chess_anti_engine.uci.model_loader import model_config_from_arch

    out = gen_dir(run_dir, 0)
    if stage_done(out):
        return out / MODEL_FILE
    if bool(init_checkpoint) == bool(random_init):
        raise SystemExit("exactly one of --init-checkpoint / --random-init is required to create generation 0")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    target = out / MODEL_FILE
    if init_checkpoint:
        src = Path(init_checkpoint)
        payload = torch.load(str(src), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "model" not in payload or "arch" not in payload:
            raise SystemExit(f"{src} is not a self-describing Trainer checkpoint (needs 'model' and 'arch')")
        model_config_from_arch(dict(payload["arch"]))  # strict schema check
        shutil.copyfile(src, target)
        inputs = {"init_checkpoint": input_ref(src)}
        config: dict[str, Any] = {"kind": "checkpoint"}
    else:
        if arch_from:
            base = dict(torch.load(arch_from, map_location="cpu", weights_only=True)["arch"])
        else:
            base = {"_schema_version": ARCH_SCHEMA_VERSION, **asdict(ModelConfig())}
        overrides = json.loads(Path(model_config).read_text()) if model_config else {}
        unknown = set(overrides) - {f.name for f in fields(ModelConfig)}
        if unknown:
            raise SystemExit(f"--model-config has unknown ModelConfig keys {sorted(unknown)}")
        arch = {**base, **overrides, "_schema_version": ARCH_SCHEMA_VERSION}
        cfg = model_config_from_arch(arch)
        torch_seed = int(seed) & 0xFFFFFFFF
        torch.manual_seed(torch_seed)
        # The Trainer only serialises the fresh weights in the standard
        # self-describing format (model + arch) every loader here reads.
        trainer = build_trainer(cfg, TrainSettings(device="cpu"), log_dir=None)
        trainer.save(target)
        inputs = {**({"arch_from": input_ref(Path(arch_from))} if arch_from else {}),
                  **({"model_config": input_ref(Path(model_config))} if model_config else {})}
        config = {"kind": "random_init", "arch": arch, "torch_seed": torch_seed}
    write_manifest(out, stage="init", config=config, inputs=inputs, outputs={"model": input_ref(target)},
                   counts={}, seeds={"init": int(seed), "init_torch_seed_realised": int(seed) & 0xFFFFFFFF})
    return target


# ---------------------------------------------------------------------------
# settings from flags
# ---------------------------------------------------------------------------

def collect_settings(a: argparse.Namespace) -> CollectSettings:
    return CollectSettings(
        games=a.games, concurrency=resolve_collect_shape(a)[1], sims=a.sims, full_prob=a.full_prob, fast_sims=a.fast_sims,
        fast_value_rows=a.fast_value_rows, tree_reuse=not a.no_tree_reuse, topk=a.topk, c_scale=a.c_scale,
        c_visit=a.c_visit, target_q_rescale=not a.no_target_q_rescale, policy_temp=a.policy_temp,
        target_untempered_prior=a.target_untempered_prior, vloss_weight=a.vloss_weight,
        temp_start=a.temp_start, temp_floor=a.temp_floor, temp_decay_plies=a.temp_decay_plies,
        root_noise_scale=a.root_noise, root_noise_after=a.root_noise_after,
        root_noise_decay_start_move=a.root_noise_decay_start_move,
        root_noise_decay_moves=a.root_noise_decay_moves, max_plies=a.max_plies,
        syzygy=a.syzygy, sf_path=a.sf_path or "", sf_nodes=a.sf_nodes,
        book_mix=tuple(float(x) for x in a.book_mix.split(",")), rows_per_shard=a.rows_per_shard,
        opening_sampling=a.opening_sampling, dup_sketch_budget=a.dup_sketch_budget,
        exact_repeat_vs_previous=a.exact_repeat_vs_previous,
        device=a.device, eval_max_batch=a.eval_max_batch, collect_procs=resolve_collect_procs(a), seed=a.seed,
    )


def train_settings(a: argparse.Namespace, run_dir: Path) -> TrainSettings:
    ref = a.kl_reference or (str(model_path(run_dir, 0)) if (a.kl_policy > 0 or a.kl_wdl > 0) else None)
    mix_ref = a.prior_mix_reference or (
        str(model_path(run_dir, 0)) if a.policy_prior_mix < 1.0 and a.prior_mix_source == "reference" else None)
    settings = TrainSettings(
        presentations=a.presentations, sample_with_replacement=a.sample_with_replacement,
        window=a.window, batch_size=a.batch_size, updates=a.updates, passes=a.passes, lr=a.lr,
        lr_floor=a.lr_floor, warmup_steps=a.warmup_steps, warmup_start=a.warmup_start,
        optimizer=a.optimizer, optimizer_state=a.optimizer_state,
        reapply_optimizer_hparams=a.reapply_optimizer_hparams, init_optimizer_state=a.init_optimizer_state,
        adopt_checkpoint_optimizer_hparams=a.adopt_checkpoint_optimizer_hparams,
        matrix_optimizer_scope=a.matrix_optimizer_scope, mirror_prob=a.mirror_prob,
        w_policy=a.w_policy, w_wdl=a.w_wdl, value_outcome_weight=a.value_outcome_weight,
        value_search_weight=a.value_search_weight, value_search_source=a.value_search_source,
        value_draw_source=a.value_draw_source, policy_prior_mix=a.policy_prior_mix,
        prior_mix_source=a.prior_mix_source, prior_mix_reference=mix_ref,
        policy_target_temp=a.train_target_temp,
        kl_policy=a.kl_policy, kl_wdl=a.kl_wdl, kl_reference=ref, sl_mix_dir=a.sl_mix_dir,
        sl_mix_frac=a.sl_mix_frac, sl_mix_max_rows=a.sl_mix_max_rows, holdout_frac=a.holdout_frac,
        probe_rows=a.probe_rows, device=a.device, use_amp=str(a.device).startswith("cuda"), seed=a.seed,
        allow_capped_collect_rows=a.allow_capped_collect_rows,
        snapshot_updates=parse_snapshot_updates(a.snapshot_updates),
        accum_steps=a.accum_steps, snapshot_rows=parse_snapshot_updates(a.snapshot_rows, flag="--snapshot-rows"),
    )
    settings.validate()
    trainer_kwargs(settings)  # the Trainer's own range checks (e.g. --train-target-temp)
    return settings


def parse_snapshot_updates(spec: str | None, *, flag: str = "--snapshot-updates") -> tuple[int, ...]:
    if not spec:
        return ()
    try:
        return tuple(int(x) for x in str(spec).split(",") if x.strip())
    except ValueError as exc:
        raise SystemExit(f"{flag} wants comma-separated counts, got {spec!r}") from exc


def eval_opening_plies(a: argparse.Namespace) -> int:
    """The arena must replay every EVAL line to its end: read the split's
    longest line and refuse a smaller explicit value."""
    summary_path = Path(a.openings_dir) / "summary.json"
    longest = int(json.loads(summary_path.read_text())["eval_max_plies"]) if summary_path.is_file() else None
    if a.eval_opening_plies is None:
        if longest is None:
            raise SystemExit(f"{summary_path} missing; pass --eval-opening-plies")
        return longest
    if longest is not None and int(a.eval_opening_plies) < longest:
        raise SystemExit(f"--eval-opening-plies {a.eval_opening_plies} < the split's longest EVAL line ({longest}): "
                         "arena games would start mid-line")
    return int(a.eval_opening_plies)


def eval_settings(a: argparse.Namespace) -> EvalSettings:
    return EvalSettings(games=a.eval_games, sims=a.eval_sims, seed=a.eval_seed,
                        device=a.eval_device or a.device,
                        syzygy=None if a.eval_no_syzygy else a.syzygy, max_plies=a.eval_max_plies,
                        opening_plies=eval_opening_plies(a), max_concurrent_games=a.eval_concurrency,
                        root_noise=a.eval_root_noise,
                        extra_args=tuple(a.arena_arg or ()))


def thresholds(a: argparse.Namespace) -> dict[str, float | None]:
    return {name: getattr(a, name) for name in TRIPWIRE_FIELDS}


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def do_collect(a: argparse.Namespace, run_dir: Path, g: int) -> None:
    ckpt = sealed_model(run_dir, g - 1)
    if a.collect_from:
        try:
            stats = reference_collect(collect_dir=gen_dir(run_dir, g) / "collect", generation=g,
                                      source_run_dir=Path(a.collect_from), actor_checkpoint=ckpt,
                                      gen0_model=sealed_model(run_dir, 0),
                                      train_jsonl=Path(a.openings_dir) / "train.jsonl", settings=collect_settings(a))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps({"stage": "collect", "generation": g, "reference_to": str(Path(a.collect_from).resolve()),
                          **{k: stats.get(k) for k in ("games", "rows", "decisive_rate", "draw_rate")}},
                         default=str), flush=True)
        return
    stats = collect_generation(gen_dir=gen_dir(run_dir, g) / "collect", generation=g, checkpoint=ckpt,
                               train_jsonl=Path(a.openings_dir) / "train.jsonl", settings=collect_settings(a),
                               pause_markers=tuple(Path(m) for m in (a.pause_marker or ())))
    print(json.dumps({"stage": "collect", "generation": g, **{k: stats[k] for k in (
        "games", "rows", "decisive_rate", "draw_rate", "repeat_rate_within", "repeat_rate_vs_previous",
        "target_entropy", "games_per_hour", "cuda_graphs", "evaluator_mode", "collect_procs",
        "paused_seconds", "timing_independent")}},
                     default=str), flush=True)


def do_train(a: argparse.Namespace, run_dir: Path, g: int) -> None:
    prev_trainer = sealed_model(run_dir, g - 1) if g > 1 else None
    summary = train_generation(run_dir=run_dir, generation=g, init_checkpoint=sealed_model(run_dir, g - 1),
                               prev_trainer_checkpoint=prev_trainer, settings=train_settings(a, run_dir))
    print(json.dumps({"stage": "train", "generation": g, "updates": summary["updates"],
                      "heldout_after": summary["heldout_after"], "probe": summary["probe_vs_previous"],
                      "anchor": summary["anchor"], "sl_frac_realised": summary["sl_frac_realised"]},
                     default=str), flush=True)


def do_evaluate(a: argparse.Namespace, run_dir: Path, g: int) -> None:
    ref = Path(a.reference_checkpoint) if a.reference_checkpoint else sealed_model(run_dir, 0)
    results = evaluate_generation(run_dir=run_dir, generation=g, candidate=sealed_model(run_dir, g),
                                  previous=sealed_model(run_dir, g - 1), reference=ref,
                                  openings_pgn=Path(a.openings_dir) / "eval.pgn", settings=eval_settings(a))
    print(json.dumps({"stage": "evaluate", "generation": g, **results}, default=str), flush=True)


def tripwire_exit(a: argparse.Namespace, run_dir: Path) -> int:
    rows = tripwire_table(run_dir)
    print(format_table(rows), flush=True)
    if not a.tripwire:
        return 0
    try:
        breaches = check_tripwires(rows, thresholds(a))
    except ValueError as exc:
        print(f"tripwire: {exc}", file=sys.stderr)
        return 2
    for line in breaches:
        print(f"TRIPWIRE {line}", file=sys.stderr)
    return EXIT_TRIPWIRE if breaches else 0


# Every parsed setting is pinned except these, which only say where/how the
# process runs or when to stop. A deny-list, so a newly added flag is pinned by
# default instead of being changeable mid-run without a record.
RUN_CONFIG_EXEMPT = frozenset({
    # `device` is NOT exempt: it switches bf16 autocast in collect and train.
    # `eval_device` only places the arena (whose AMP choice is its own record).
    "command", "run_dir", "generations", "generation", "redo", "redo_downstream", "allow_config_change", "threads",
    "pause_marker", "eval_device", "tripwire", *TRIPWIRE_FIELDS,
})


# Flags added after runs were already pinned, with the value every earlier run
# implicitly had. A run_config.json without the key reads as that default, so
# resuming an older run at the default is not a "change" -- any other value is.
ADDED_SINCE_PINNING = {"init_optimizer_state": "fresh",
                       "matrix_optimizer_scope": "default", "adopt_checkpoint_optimizer_hparams": False,
                       "allow_capped_collect_rows": False, "snapshot_updates": None, "collect_from": None,
                       # Every collect before --collect-procs ran as one process.
                       "collect_procs": 1, "accum_steps": 1, "snapshot_rows": None}


def _pinned_value(path: Path, key: str, legacy: Any) -> Any:
    """``key`` from a run_config.json, ``legacy`` when the run predates it, None when unpinned."""
    if not path.is_file():
        return None
    return json.loads(path.read_text())["settings"].get(key, legacy)


def _collect_manifest_value(source_run_dir: Path, key: str, legacy: Any) -> Any:
    """``key`` from the earliest sealed collect manifest of a source run without a
    run_config.json (e.g. a ``retarget`` output), None when it has none."""
    for manifest in sorted(source_run_dir.glob("gen_*/collect/manifest.json")):
        return json.loads(manifest.read_text()).get("config", {}).get("settings", {}).get(key, legacy)
    return None


def resolve_collect_shape(a: argparse.Namespace) -> tuple[int, int]:
    """(``--collect-procs``, ``--concurrency``) as the integers that are pinned and used.

    Explicit values are kept. A flag not given takes, in order: the value the run
    already pinned (1 process for a run pinned before --collect-procs existed); for
    an arm, the source run's value (its rows were made with it; from its collect
    manifest when the source has no run_config, as a retarget output); else the
    sims-dependent default table ``collect.CUDA_COLLECT_SHAPES`` (CPU: 1 x 64)."""
    pinned, source = Path(a.run_dir) / RUN_CONFIG, Path(a.collect_from) / RUN_CONFIG if a.collect_from else None
    default = default_collect_shape(a.device, a.sims)
    # A defaulted K never exceeds --games (a tiny run would otherwise be refused).
    default = (min(default[0], max(1, int(a.games))), default[1])
    out = []
    for key, given, legacy, fallback in (("collect_procs", a.collect_procs, 1, default[0]),
                                         ("concurrency", a.concurrency, 64, default[1])):
        if given is not None:
            if int(given) < 1:
                raise SystemExit(f"--{key.replace('_', '-')} must be >= 1, got {given}")
            out.append(int(given))
            continue
        value = _pinned_value(pinned, key, legacy)
        if value is None and source is not None:
            value = _pinned_value(source, key, legacy)
        if value is None and a.collect_from:
            value = _collect_manifest_value(Path(a.collect_from), key, legacy)
        out.append(int(fallback if value is None else value))
    return out[0], out[1]


def resolve_collect_procs(a: argparse.Namespace) -> int:
    return resolve_collect_shape(a)[0]


def validate_before_pinning(a: argparse.Namespace, run_dir: Path, stages: set[str]) -> None:
    """Raise every settings-only refusal BEFORE any side effect.

    Both ``run`` and a standalone stage call this ahead of ``check_run_config``
    (which pins run_config.json) and, for a stage, ahead of any unsealing or
    manifest removal, so a refused setting is neither pinned (dropping it again
    would need --allow-config-change) nor leaves lineage half-rewound.

    Every flag group is pinned whichever command runs, so every group's
    settings-only checks run always (a standalone ``collect`` pins the train
    flags too). ``stages`` names what will execute and adds the checks that read
    run state: "collect" generation 0's init flags and an arm's
    ``--collect-from`` source, "train" the init-optimizer source, the capped
    lineage and the snapshot plan against the phase's planned updates.
    """
    settings = train_settings(a, run_dir)
    try:
        validate_settings(collect_settings(a))
    except ValueError as exc:
        raise SystemExit(f"collect settings refused: {exc}") from exc
    # Without the split's summary.json and an explicit --eval-opening-plies the
    # opening plies are unknowable, which only an executing `eval` must refuse.
    knowable = a.eval_opening_plies is not None or (Path(a.openings_dir) / "summary.json").is_file()
    if "eval" in stages or (not a.no_eval and knowable):
        eval_settings(a)
    if getattr(a, "generation", None) is not None:
        gens = [int(a.generation)]
    else:
        gens = list(range(1, int(getattr(a, "generations", 0) or 0) + 1))
    if "train" in stages:
        if settings.init_optimizer_state == "checkpoint" and not stage_done(gen_dir(run_dir, 1) / "train"):
            preflight_init_optimizer_source(a, run_dir, settings)
        to_train = gens if getattr(a, "generation", None) is not None else [
            g for g in gens if not stage_done(gen_dir(run_dir, g) / "train")]
        # The whole lineage, not just the windows: a model descends from every
        # earlier generation's rows (lineage_capped_collects).
        try:
            refuse_capped_collects(lineage_capped_collects(run_dir, max(to_train, default=0)), settings)
            for g in to_train:
                # Knowable only once the phase's collects are sealed; otherwise
                # train_generation refuses before its first update.
                planned = planned_phase_updates(run_dir, g, settings)
                snaps = planned_snapshot_updates(run_dir, g, settings)
                if planned is not None and snaps is not None:
                    check_snapshot_plan(snaps, planned)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if "collect" in stages:
        if not stage_done(gen_dir(run_dir, 0)) and bool(a.init_checkpoint) == bool(a.random_init):
            raise SystemExit("exactly one of --init-checkpoint / --random-init is required to create generation 0")
        if a.collect_from:
            to_collect = gens if getattr(a, "generation", None) is not None else [
                g for g in gens if not stage_done(gen_dir(run_dir, g) / "collect")]
            try:
                for g in to_collect:
                    precheck_source(Path(a.collect_from), g, collect_settings(a))
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc


def preflight_init_optimizer_source(a: argparse.Namespace, run_dir: Path, settings: TrainSettings) -> None:
    """--init-optimizer-state checkpoint refusals that need only the init
    checkpoint's header: generation 1 trains from gen_0000/model.pt, a copy of
    --init-checkpoint (or the random init, which has no moments to start from)."""
    gen0 = gen_dir(run_dir, 0)
    if stage_done(gen0):
        kind = json.loads((gen0 / "manifest.json").read_text()).get("config", {}).get("kind")
        source = None if kind == "random_init" else gen0 / MODEL_FILE
    else:
        source = None if a.random_init else (Path(a.init_checkpoint) if a.init_checkpoint else None)
    if source is None:
        raise SystemExit("--init-optimizer-state checkpoint needs an --init-checkpoint with optimizer state; a "
                         "random init has none (use --init-optimizer-state fresh)")
    try:
        preflight_init_optimizer(source, settings)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def run_stages_left(a: argparse.Namespace, run_dir: Path) -> set[str]:
    """The stage kinds a ``run`` would still execute (a sealed stage is skipped)."""
    kinds = ["collect", "train"] + ([] if a.no_eval else ["eval"])
    return {k for g in range(1, int(a.generations) + 1) for k in kinds
            if not stage_done(gen_dir(run_dir, g) / k)}


def check_run_config(a: argparse.Namespace, run_dir: Path) -> None:
    """Pin the loop's settings at the first ``run``; refuse a silent change on resume."""
    current = {k: v for k, v in sorted(vars(a).items()) if k not in RUN_CONFIG_EXEMPT}
    path = run_dir / RUN_CONFIG
    if not path.exists():
        write_json_atomic(path, {"settings": current, "changes": []})
        return
    stored = json.loads(path.read_text())
    # Only CURRENT flags are compared, so a key a later version removed (e.g.
    # collect_cuda_graphs, retired 2026-11-21: CUDA graphs are the only collect
    # path on CUDA) still loads from an older run_config.json, whatever its value.
    pinned = {**ADDED_SINCE_PINNING, **stored["settings"]}
    diff = {k: {"was": pinned.get(k), "now": v} for k, v in current.items() if pinned.get(k) != v}
    if not diff:
        return
    if not a.allow_config_change:
        raise SystemExit(f"run settings differ from {path}: {json.dumps(diff, default=str)}; "
                         "pass --allow-config-change to adopt them from the next stage on")
    stored["changes"].append({"diff": diff, "argv": sys.argv})
    stored["settings"] = current
    write_json_atomic(path, stored)


def cmd_run(a: argparse.Namespace) -> int:
    run_dir = Path(a.run_dir)
    markers = [Path(m) for m in (a.pause_marker or ())]
    blocked = blocking_markers(run_dir, markers)
    if blocked:
        print("not starting: " + "; ".join(blocked), file=sys.stderr)
        return EXIT_BLOCKED
    # Validate BEFORE creating anything: a refused setting leaves no run dir behind.
    validate_before_pinning(a, run_dir, {"collect", "train"} | run_stages_left(a, run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    check_run_config(a, run_dir)
    init_run(run_dir, init_checkpoint=a.init_checkpoint, random_init=a.random_init,
             model_config=a.model_config, arch_from=a.arch_from, seed=a.seed)
    stages = [("collect", do_collect), ("train", do_train)]
    if not a.no_eval:
        stages.append(("eval", do_evaluate))
    verify_stage(run_dir, 0, "init")
    for g in range(1, int(a.generations) + 1):
        for name, fn in stages:
            if stage_done(gen_dir(run_dir, g) / name):
                verify_stage(run_dir, g, name)  # a sealed stage is reused only if its upstream is unchanged
                continue
            blocked = blocking_markers(run_dir, markers)
            if blocked:
                print(f"stopping before gen {g} {name}: " + "; ".join(blocked), file=sys.stderr)
                return EXIT_BLOCKED
            fn(a, run_dir, g)
        code = tripwire_exit(a, run_dir)
        if code:
            return code
    return 0


def cmd_stage(a: argparse.Namespace) -> int:
    run_dir = Path(a.run_dir)
    blocked = blocking_markers(run_dir, [Path(m) for m in (a.pause_marker or ())])
    if blocked:
        print("not starting: " + "; ".join(blocked), file=sys.stderr)
        return EXIT_BLOCKED
    stage = "eval" if a.command == "evaluate" else a.command
    # Before pinning, unsealing or unlinking anything: a refused setting must
    # leave run_config.json and the lineage exactly as they were.
    validate_before_pinning(a, run_dir, {stage})
    if (run_dir / RUN_CONFIG).exists():
        check_run_config(a, run_dir)  # the same pinned settings a `run` would be held to
    stage_dir = gen_dir(run_dir, a.generation) / stage
    if stage_done(stage_dir) and not a.redo:
        verify_before(run_dir, a.generation, stage, inclusive=True)
        print(f"{stage_dir} is already complete (manifest present); --redo to replace it", file=sys.stderr)
        return 0
    if a.redo_downstream and not a.redo:
        raise SystemExit("--redo-downstream needs --redo")
    downstream = later_sealed(run_dir, a.generation, stage)
    if downstream and not a.redo_downstream:
        raise SystemExit(f"gen {a.generation} {stage}: later stages are sealed "
                         f"({', '.join(f'gen {g} {s}' for g, s in downstream)}) and were built on this one; "
                         "pass --redo-downstream to unseal them (moved to <run>/superseded/), or redo a later stage")
    verify_before(run_dir, a.generation, stage, inclusive=False)
    if downstream:
        print(json.dumps({"redo_downstream": unseal_after(run_dir, a.generation, stage, include_self=True)}),
              flush=True)
    elif stage_dir.exists() and a.redo and stage_done(stage_dir):
        unseal_after(run_dir, a.generation, stage, include_self=True)  # rewind this stage's own side files
    if stage_dir.exists() and a.redo:
        (stage_dir / "manifest.json").unlink(missing_ok=True)  # unsealed until the redo's manifest lands
    if a.command == "collect":
        init_run(run_dir, init_checkpoint=a.init_checkpoint, random_init=a.random_init,
                 model_config=a.model_config, arch_from=a.arch_from, seed=a.seed)
    {"collect": do_collect, "train": do_train, "evaluate": do_evaluate}[a.command](a, run_dir, a.generation)
    return tripwire_exit(a, run_dir) if a.command == "evaluate" else 0


def cmd_split(a: argparse.Namespace) -> int:
    books = []
    for spec in a.book:
        path, _, plies = spec.rpartition(":")
        if not path or not plies.isdigit():
            raise SystemExit(f"--book wants PATH:MAX_PLIES, got {spec!r}")
        books.append((Path(path), int(plies)))
    out = Path(a.out)
    existing = read_manifest(out)
    if existing is not None and not a.force:
        want = {"eval_lines": a.eval_lines, "salt": a.salt,
                "books": [{"path": str(p), "max_plies": m, "book": i + 1} for i, (p, m) in enumerate(books)]}
        have = {k: existing.get("config", {}).get(k) for k in want}
        if have != want:
            raise SystemExit(f"{out} was split with different settings {have}; asked for {want}. "
                             "Use another --out, or --force to replace it")
        print(f"{out} already split with these settings (manifest present); --force to redo", file=sys.stderr)
        return 0
    print(json.dumps(split_openings(books=books, out_dir=out, eval_lines=a.eval_lines, salt=a.salt)), flush=True)
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-dir", required=True)
    p.add_argument("--openings-dir", required=True, help="output of split-openings")
    p.add_argument("--seed", type=int, default=20261119)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = torch default)")
    p.add_argument("--pause-marker", action="append",
                   help="refuse to start a stage while this file exists (repeatable)")
    g = p.add_argument_group("generation 0")
    g.add_argument("--init-checkpoint", default=None, help="start from this Trainer checkpoint (e.g. the SL base)")
    g.add_argument("--random-init", action="store_true", help="start from random weights")
    g.add_argument("--model-config", default=None, help="JSON of ModelConfig overrides for --random-init")
    g.add_argument("--arch-from", default=None, help="checkpoint whose embedded arch --random-init starts from")

    c = p.add_argument_group("collect")
    c.add_argument("--games", type=int, default=2000)
    c.add_argument("--concurrency", type=int, default=None,
                   help="concurrent game slots PER collect process (default: the run's pinned value, else by "
                        "--sims from collect.CUDA_COLLECT_SHAPES on CUDA -- 64 at <= 16 sims, 128 above -- "
                        "and 64 on CPU)")
    c.add_argument("--sims", type=int, default=64,
                   help="simulations per TARGET (full) ply; the arena's budget is --eval-sims, independent")
    c.add_argument("--full-prob", type=float, default=1.0,
                   help="playout cap randomisation: share of plies searched at --sims and stored as policy "
                        "rows; the rest search --fast-sims and store no policy target (1.0 = off)")
    c.add_argument("--fast-sims", type=int, default=16)
    c.add_argument("--fast-value-rows", action="store_true",
                   help="keep fast plies as value-only rows (default: fast plies store no row)")
    c.add_argument("--no-tree-reuse", action="store_true",
                   help="search every ply from a fresh root (default: carry the chosen subtree, as production)")
    c.add_argument("--topk", type=int, default=16)
    c.add_argument("--c-scale", type=float, default=0.1)
    c.add_argument("--c-visit", type=float, default=50.0)
    c.add_argument("--no-target-q-rescale", action="store_true",
                   help="stored target uses sigma in absolute Q units (GumbelConfig.target_q_rescale=False)")
    c.add_argument("--policy-temp", type=float, default=1.0, help="search prior temperature (gumbel_policy_temp)")
    c.add_argument("--target-untempered-prior", action="store_true")
    c.add_argument("--vloss-weight", type=int, default=1)
    c.add_argument("--temp-start", type=float, default=0.5, help="move temperature on the first ply after the book")
    c.add_argument("--temp-floor", type=float, default=0.1)
    c.add_argument("--temp-decay-plies", type=int, default=10)
    c.add_argument("--root-noise", type=float, default=1.0, help="root Gumbel noise scale (production 1.0)")
    c.add_argument("--root-noise-after", type=float, default=0.5)
    c.add_argument("--root-noise-decay-start-move", type=int, default=12)
    c.add_argument("--root-noise-decay-moves", type=int, default=3)
    c.add_argument("--max-plies", type=int, default=450)
    c.add_argument("--syzygy", default=None, help="Syzygy path(s); enables the strict rule50 TB route")
    c.add_argument("--sf-path", default=None, help="Stockfish binary (required by play_batch; labels unused)")
    c.add_argument("--sf-nodes", type=int, default=1)
    c.add_argument("--book-mix", default="0.95,0.05", help="per-book game share, book order of split-openings")
    c.add_argument("--opening-sampling", choices=("cycle", "independent"), default="cycle",
                   help="cycle (default): walk each book's seeded permutation across generations, reusing no line "
                        "before the book is used up; independent: a fresh draw every generation")
    c.add_argument("--dup-sketch-budget", type=int, default=4_000_000,
                   help="hashes kept by the constant-memory cross-generation duplicate estimator (~8 bytes each)")
    c.add_argument("--exact-repeat-vs-previous", action="store_true",
                   help="compute repeat_rate_vs_previous exactly over every earlier row (memory grows with the run)")
    c.add_argument("--rows-per-shard", type=int, default=20000)
    c.add_argument("--eval-max-batch", type=int, default=None,
                   help="CUDA evaluator batch cap (default: the search's leaf-buffer size at this "
                        "--concurrency / --topk; a smaller value is refused -- it would drop leaf evaluations)")
    c.add_argument("--collect-procs", type=int, default=None, metavar="K",
                   help="collect processes sharing the device, each playing a disjoint slice of the openings "
                        "with its own seed (pinned: it sets the RNG streams). Default: the run's pinned value, "
                        "else by --sims from collect.CUDA_COLLECT_SHAPES on CUDA (8 at <= 16 sims, 3 above) and "
                        "1 on CPU. Refused, never stepped down, when it would not fit in usable GPU memory")
    c.add_argument("--collect-from", default=None, metavar="SOURCE_RUN_DIR",
                   help="training arm: seal each collect as a verified reference to SOURCE_RUN_DIR's sealed "
                        "collect of the same generation instead of playing games (same gen-0 model, actor, "
                        "TRAIN openings and collect flags required; see docs/simple_rl_loop.md)")

    t = p.add_argument_group("train")
    t.add_argument("--presentations", type=int, default=1,
                   help="times each trainable row is presented in its lifetime, shuffled, without replacement: "
                        "1 = in its own generation's phase; 2 (the cap) = that phase and the next")
    t.add_argument("--sample-with-replacement", action="store_true",
                   help="NON-DEFAULT: uniform sampling with replacement over --window generations "
                        "for --updates / --passes (no per-row presentation guarantee)")
    t.add_argument("--window", type=int, default=None, help="with --sample-with-replacement: last K generations (4)")
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--updates", type=int, default=0, help="with --sample-with-replacement: optimizer updates")
    t.add_argument("--passes", type=float, default=None,
                   help="with --sample-with-replacement: updates = passes * window rows / batch (1.0)")
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--lr-floor", type=float, default=None, help="cosine floor (default = --lr, i.e. flat)")
    t.add_argument("--warmup-steps", type=int, default=0)
    t.add_argument("--warmup-start", type=float, default=0.0)
    t.add_argument("--optimizer", default="nadamw")
    t.add_argument("--optimizer-state", choices=("fresh", "continue"), default="fresh")
    t.add_argument("--init-optimizer-state", choices=("fresh", "checkpoint"), default="fresh",
                   help="generation 1: 'checkpoint' starts from the init checkpoint's optimizer moments and "
                        "step counters (refused if it has none, or its parameter layout / hyperparameters differ; "
                        "its scheduler is not restored -- this run's --lr schedule drives the rate)")
    t.add_argument("--matrix-optimizer-scope", default="default", choices=MATRIX_OPTIMIZER_SCOPES,
                   help="Aurora/Muon (and SOAP) only, refused with any other optimizer: which matrices the "
                        "matrix group owns (the SL base used mlp_out)")
    t.add_argument("--reapply-optimizer-hparams", action="store_true",
                   help="on a restore (--optimizer-state continue, --init-optimizer-state checkpoint): apply the "
                        "requested lr / lr floor and this run's configured param-group hyperparameters onto "
                        "the restored optimizer (default: refuse when any differs from the checkpoint's)")
    t.add_argument("--adopt-checkpoint-optimizer-hparams", action="store_true",
                   help="on a restore: KEEP the checkpoint's param-group hyperparameters (weight decay, betas, "
                        "Aurora polar knobs, ...) where they differ from this run's configured ones; recorded. "
                        "Exclusive with --reapply-optimizer-hparams; never adopts lr / lr floor")
    t.add_argument("--allow-capped-collect-rows", action="store_true",
                   help="train on rows from collects whose evaluator cap could absorb search leaves (default: "
                        "refused; when allowed the phase records trained_on_capped_rows)")
    t.add_argument("--snapshot-updates", default=None, metavar="N[,N...]",
                   help="exact-once schedule: after these optimizer update counts of each phase write "
                        "train/snapshots/step_<N>/ (weights-only model.pt + metrics.json); strictly increasing, "
                        "each <= the phase's planned updates; training is unchanged (bit-identical on CPU)")
    t.add_argument("--snapshot-rows", default=None, metavar="N[,N...]",
                   help="exact-once schedule, instead of --snapshot-updates: snapshot after the first optimizer "
                        "step at which N rows have been presented (matched samples across effective batches; "
                        "the resolved update counts are recorded)")
    t.add_argument("--accum-steps", type=int, default=1,
                   help="gradient accumulation: one optimizer step per N micro-batches of --batch-size rows "
                        "(effective batch = batch-size x N; every row's loss coefficient is 1/effective batch, "
                        "so the step's gradient is the native big batch's). Warmup, the LR schedule and "
                        "--snapshot-updates count optimizer steps. Refused with a KL anchor, an SL share, or phase rows "
                        "missing a trained head's target (e.g. value-only playout-cap rows). "
                        "Default 1 (no accumulation)")
    t.add_argument("--mirror-prob", type=float, default=0.5)
    t.add_argument("--w-policy", type=float, default=1.0)
    t.add_argument("--w-wdl", type=float, default=1.0, help="0 = policy only")
    t.add_argument("--value-outcome-weight", type=float, default=1.0)
    t.add_argument("--value-search-weight", type=float, default=0.0)
    t.add_argument("--value-search-source", choices=("survivor_q", "stored_search_wdl"), default="survivor_q",
                   help="search value: the halving survivor's completed Q (default) or the legacy search_wdl")
    t.add_argument("--value-draw-source", choices=("net", "zero"), default="net",
                   help="draw channel of the survivor-Q WDL: the raw net's root draw, or 0")
    t.add_argument("--policy-prior-mix", type=float, default=1.0,
                   help="policy target = lam * stored target + (1 - lam) * prior (1.0 = off)")
    t.add_argument("--prior-mix-source", choices=("reference", "actor"), default="reference",
                   help="prior for the mix: a frozen reference net, or the searching net's stored prior")
    t.add_argument("--prior-mix-reference", default=None, help="reference net for the prior mix (default: gen 0)")
    t.add_argument("--train-target-temp", type=float, default=1.0, help="policy target temperature at train time")
    t.add_argument("--kl-policy", type=float, default=0.0)
    t.add_argument("--kl-wdl", type=float, default=0.0)
    t.add_argument("--kl-reference", default=None, help="frozen anchor net (default: generation 0)")
    t.add_argument("--sl-mix-dir", default=None)
    t.add_argument("--sl-mix-frac", type=float, default=0.0)
    t.add_argument("--sl-mix-max-rows", type=int, default=1_000_000)
    t.add_argument("--holdout-frac", type=float, default=0.05, help="share of games held out of training")
    t.add_argument("--probe-rows", type=int, default=2048)

    e = p.add_argument_group("evaluate")
    e.add_argument("--no-eval", action="store_true", help="run: skip the arena stage")
    e.add_argument("--eval-games", type=int, default=400)
    e.add_argument("--eval-sims", type=int, default=64,
                   help="arena simulations per move; independent of the collection budget")
    e.add_argument("--eval-seed", type=int, default=20261119)
    e.add_argument("--eval-device", default=None)
    e.add_argument("--eval-max-plies", type=int, default=300)
    e.add_argument("--eval-no-syzygy", action="store_true",
                   help="arena without --syzygy. With it (the default when --syzygy is set) the arena is "
                        "strict: one game still unresolved at --eval-max-plies invalidates the match, which "
                        "a weak/random net at low sims can hit")
    e.add_argument("--eval-opening-plies", type=int, default=None,
                   help="book plies the arena replays (default: the split's longest EVAL line; smaller is refused)")
    e.add_argument("--eval-concurrency", type=int, default=64)
    e.add_argument("--eval-root-noise", action="store_true",
                   help="arena with root Gumbel noise (default off, as the RL-takeover yardstick)")
    e.add_argument("--reference-checkpoint", default=None, help="new-vs-reference opponent (default: generation 0)")
    e.add_argument("--arena-arg", action="append", help="extra argument passed to arena_standard (repeatable)")
    e.add_argument("--tripwire", action="store_true", help="exit 3 when an armed threshold is crossed")
    for name, (column, kind) in TRIPWIRE_FIELDS.items():
        e.add_argument("--" + name.replace("_", "-"), type=float, default=None,
                       help=f"tripwire: {kind} allowed {column}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("split-openings", help="deterministic TRAIN / EVAL opening split")
    s.add_argument("--book", action="append", required=True, help="PATH:MAX_PLIES (repeatable, in book order)")
    s.add_argument("--out", required=True)
    s.add_argument("--eval-lines", type=int, default=4000)
    s.add_argument("--salt", default="simple-rl-v1")
    s.add_argument("--force", action="store_true")
    for name in ("collect", "train", "evaluate"):
        sp = sub.add_parser(name)
        _common(sp)
        sp.add_argument("--generation", type=int, required=True)
        sp.add_argument("--allow-config-change", action="store_true")
        sp.add_argument("--redo", action="store_true", help="replace this stage even if it is complete")
        sp.add_argument("--redo-downstream", action="store_true",
                        help="with --redo: also unseal every later sealed stage (moved to <run>/superseded/) and "
                             "rewind the run-level files derived from them")
    rt = sub.add_parser("retarget", help="write a NEW run dir whose generation-G collect is the source's with "
                                         "the policy target rebuilt from its stored root statistics (offline; "
                                         "an arm then trains on it with --collect-from)")
    rt.add_argument("--source-run-dir", required=True)
    rt.add_argument("--generation", type=int, required=True)
    rt.add_argument("--out-run-dir", required=True, help="must not exist")
    rt.add_argument("--no-target-q-rescale", dest="target_q_rescale", action="store_false",
                    help="rebuild without the min-max Q rescale (the only retarget so far; required)")
    r = sub.add_parser("run", help="generations 1..G of collect -> train -> evaluate")
    _common(r)
    r.add_argument("--generations", type=int, required=True)
    r.add_argument("--allow-config-change", action="store_true")
    return p


def cmd_retarget(a: argparse.Namespace) -> int:
    from chess_anti_engine.simple_rl.retarget import retarget_collect, retarget_summary

    if a.target_q_rescale:
        raise SystemExit("retarget: pass --no-target-q-rescale (the only rebuild this step offers)")
    try:
        record = retarget_collect(source_run_dir=Path(a.source_run_dir), generation=int(a.generation),
                                  out_run_dir=Path(a.out_run_dir), target_q_rescale=False)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(retarget_summary(record))
    return 0


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    if hasattr(a, "collect_procs"):
        a.collect_procs, a.concurrency = resolve_collect_shape(a)
    if getattr(a, "threads", 0):
        import torch

        torch.set_num_threads(int(a.threads))
    if a.command == "split-openings":
        return cmd_split(a)
    if a.command == "retarget":
        return cmd_retarget(a)
    if a.command == "run":
        return cmd_run(a)
    return cmd_stage(a)


if __name__ == "__main__":
    raise SystemExit(main())
