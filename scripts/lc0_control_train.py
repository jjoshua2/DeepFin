#!/usr/bin/env python3
"""Supervised training driver for the lc0 positive control.

lc0-derived shards -> DiskReplayBuffer -> Trainer.train_steps -> checkpoint.
No selfplay, no MCTS, no PID, no curriculum, no server, no Ray. The model, the
optimizer and every loss weight come from `configs/lc0_positive_control.yaml`,
which is the production config with the value blend re-pointed (see that
file's header for the full measured diff).

⚑⚑ THIS SCRIPT'S REASON TO EXIST IS THE GUARD, NOT THE LOOP.

lc0 shards carry no `sf_wdl`. `train/losses.py` falls the SF component of the
value blend back to the RAW ONE-HOT GAME OUTCOME for any row without one — no
error, no warning, no metric named for it. At the production `sf_wdl_frac:
0.50` this arm would silently train 0.80 of its value target on the deep game
outcome instead of the 0.30 on paper, and every number it produced would be
about an experiment nobody chose.

Three checks, at three different levels, because the first two can be
satisfied by a config that still does the wrong thing:

  1. LAUNCH, config-level. `assert_pid_cannot_reassert_sf_wdl` — an override
     the difficulty controller can undo is not an override.
  2. LAUNCH, corpus-level. The converter's own `run_config_problems` is REUSED
     (not restated) against the actual shard directory, so the "do these
     shards carry an SF label" question is answered by reading the shards.
  3. ⚑ REALIZED, per-step. `compute_loss` is wrapped for the duration of the
     run and the fracs it is ACTUALLY CALLED WITH — together with the batch's
     own `sf_wdl_rows` count — are fed to `value_blend_guard`. A configured
     value is not an applied value; this is the only one of the three that
     measures the applied one.

Check 3 fails loudly (non-zero exit, no checkpoint written). `--allow-leak`
exists so the failure can be demonstrated deliberately, and prints a banner
saying the run is not a valid control.

Usage
-----
    PYTHONPATH=. python3 scripts/lc0_control_train.py \\
        --config configs/lc0_positive_control.yaml \\
        --shards <converted-dir> [<converted-dir> ...] \\
        --steps 20 --out-dir <run-dir>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import numpy as np
import torch

from chess_anti_engine.model import build_model, model_config_from_flat_config
from chess_anti_engine.replay.buffer import ReplayBuffer
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import iter_shard_paths
from chess_anti_engine.train import trainer as trainer_module
from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config
from chess_anti_engine.train.value_blend_guard import (
    ValueBlendMisconfigured,
    ValueBlendReadout,
    assert_no_silent_outcome_fallback,
    assert_pid_cannot_reassert_sf_wdl,
    value_blend_readout,
)
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

# Reused, not restated: the converter owns the "these shards have no SF label"
# question and already ships the config check for it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lc0_data_to_rows import run_config_problems, shard_dir_has_sf_wdl


class _LossCapture:
    """Records what ``compute_loss`` was ACTUALLY called with, every call.

    ⚑ Wrapping the call is the point. Reading `trainer.sf_wdl_frac` before the
    step would report the attribute, and the attribute is exactly what a live
    reload, a PID push or a PB2 mutation can change between the read and the
    step. The kwargs recorded here are the ones the trained objective used.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.kwargs: dict[str, Any] = {}
        self.readouts: list[ValueBlendReadout] = []

    def observe(self, kwargs: dict[str, Any], result: dict[str, torch.Tensor]) -> None:
        self.calls += 1
        self.kwargs = dict(kwargs)
        self.readouts.append(value_blend_readout(
            sf_wdl_frac=float(kwargs.get("sf_wdl_frac", 0.0)),
            search_wdl_frac=float(kwargs.get("search_wdl_frac", 0.0)),
            sf_wdl_rows=float(result["sf_wdl_rows"].detach().item()),
            batch_rows=float(result["batch_rows"].detach().item()),
        ))

    @property
    def worst(self) -> ValueBlendReadout:
        """The call with the largest leak — the one a guard must judge.

        Not the first and not the mean: the failure this guards against can
        begin partway through a run (a live edit, a controller push), and a
        mean over 800 steps would dilute a leak that started at step 700 into
        something under any tolerance.
        """
        return max(self.readouts, key=lambda r: r.leaked_to_outcome)


class CaptureRealizedLosses:
    """Wrap ``trainer.compute_loss`` for the duration of the block.

    A class rather than ``@contextmanager`` so the captured object has a real
    type at the call site: the whole point of this block is that a reviewer can
    see what the guard is judging.
    """

    def __init__(self) -> None:
        self.capture = _LossCapture()
        self._original = trainer_module.compute_loss

    def __enter__(self) -> _LossCapture:
        capture, original = self.capture, self._original

        def wrapped(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            result = original(*args, **kwargs)
            capture.observe(kwargs, result)
            return result

        trainer_module.compute_loss = wrapped
        return capture

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        trainer_module.compute_loss = self._original


def _as_replay_buffer(sampler: Any) -> ReplayBuffer:
    """Duck-typed stand-in, exactly as ``scripts/offline_replay_epoch.py`` does.

    ``train_steps`` needs only ``sample_batch_arrays`` + ``rng``, which
    ``DiskReplayBuffer`` has, but the two classes share no declared base.
    """
    return cast(ReplayBuffer, sampler)


def stage_shards(shard_dirs: list[Path], staging: Path) -> int:
    """Symlink every shard into ONE flat directory with unique indices.

    ``DiskReplayBuffer`` reads a single directory and ``iter_shard_paths``
    does not recurse, while each conversion run numbers its shards from
    ``shard_000000`` — so N converted hours cannot be pointed at directly
    without colliding. Symlinks keep the buffer on the production ingest path
    (rather than a bespoke loader that would no longer be the stack under
    test) and copy no data.

    ⚑ The buffer is opened ``read_only=True`` downstream regardless. That
    class DELETES SHARDS from a writable directory whose capacity differs from
    the live one; through a symlink farm it would delete the CONVERTED
    ORIGINALS, which cost ~3.5 minutes per 400 games to rebuild.
    """
    staging.mkdir(parents=True, exist_ok=True)
    for stale in staging.iterdir():
        if stale.is_symlink():
            stale.unlink()
    index = 0
    for shard_dir in shard_dirs:
        paths = iter_shard_paths(shard_dir)
        if not paths:
            raise ValueError(f"no shards under {shard_dir}")
        for path in paths:
            (staging / f"shard_{index:06d}.zarr").symlink_to(path.resolve())
            index += 1
    if index == 0:
        raise ValueError("no shards found in any --shards directory")
    return index


def preflight(cfg: dict[str, Any], shard_dirs: list[Path], *, allow_leak: bool) -> None:
    """Both LAUNCH-level guards.

    ``allow_leak`` downgrades them to a banner instead of skipping them. The
    point of the flag is to let the run REACH the realized (per-step) check so
    the leak can be observed as a number, which is the only way to demonstrate
    that the guard is measuring something rather than restating the config.
    """
    def fail(message: str) -> None:
        if allow_leak:
            print(f"⚑ --allow-leak: IGNORING launch guard — {message}")
            return
        raise SystemExit(f"REFUSING TO LAUNCH — {message}")

    try:
        assert_pid_cannot_reassert_sf_wdl(
            sf_wdl_frac=float(cfg.get("sf_wdl_frac", 0.0)),
            sf_wdl_frac_floor=float(cfg.get("sf_wdl_frac_floor", 0.0)),
            context="launch config",
        )
    except ValueBlendMisconfigured as exc:
        fail(str(exc))
    else:
        print("[preflight] PID sf_wdl ramp disabled at source (sf_wdl_frac <= 0)")

    have_sf = any(shard_dir_has_sf_wdl(Path(d)) for d in shard_dirs)
    print(f"[preflight] shards carry an sf_wdl label: {have_sf}")
    problems = run_config_problems(cfg, shards_have_sf_wdl=have_sf)
    if problems:
        fail("the config is wrong for these shards:\n  " + "\n  ".join(problems))
    else:
        print("[preflight] converter run-config gate: no problems")


def _metric_fields(metrics: Any, predicate: Any) -> list[tuple[str, Any]]:
    return [
        (field.name, getattr(metrics, field.name))
        for field in sorted(dataclasses.fields(type(metrics)), key=lambda f: f.name)
        if predicate(field.name)
    ]


def print_realized(capture: _LossCapture, metrics: Any) -> None:
    """The realized-vs-configured table, head losses, and head denominators."""
    readout = capture.worst
    print("\n=== REALIZED VALUE BLEND (read off compute_loss, worst of "
          f"{capture.calls} calls) ===")
    for name, value in readout.as_table():
        print(f"  {name:38s} {value:.6f}")
    print("\n=== REALIZED LOSS WEIGHTS (the kwargs compute_loss received) ===")
    for key in sorted(capture.kwargs):
        if key.startswith(("w_", "sf_wdl_", "search_wdl_", "policy_target_temp")):
            print(f"  {key:38s} {capture.kwargs[key]!r}")
    print("\n=== HEAD-BY-HEAD LOSSES ===")
    for name, value in _metric_fields(
        metrics, lambda n: ("loss" in n or n.endswith(("_ce", "_acc"))) and "rows" not in n,
    ):
        if isinstance(value, (int, float)):
            print(f"  {name:38s} {value:.6f}")
    print("\n=== HEAD DENOMINATORS (rows that actually trained each head) ===")
    print("  ⚑ A weight on a head whose target the shards do not carry is "
          "INERT, not applied.\n     These counts are the proof, summed over "
          "the run's microbatches.")
    for name, value in _metric_fields(
        metrics, lambda n: n.endswith(("_rows", "_frac")) or n.startswith("has_"),
    ):
        print(f"  {name:38s} {value!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/lc0_positive_control.yaml"))
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=0, help="0 = config batch_size")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-compile", action="store_true",
        help="skip torch.compile; changes throughput, not the objective",
    )
    parser.add_argument(
        "--allow-leak", action="store_true",
        help="⚑ proceed even if the SF share is landing on the game outcome. "
             "For demonstrating the guard. The run is NOT a valid control.",
    )
    args = parser.parse_args(argv)

    cfg = flatten_run_config_defaults(load_yaml_file(str(args.config)))
    shard_dirs = [Path(d) for d in args.shards]
    preflight(cfg, shard_dirs, allow_leak=bool(args.allow_leak))

    torch.manual_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_shards(shard_dirs, out_dir / "staged_shards")
    print(f"[data] staged {staged} shard(s) from {len(shard_dirs)} directory(ies)")

    kwargs = trainer_kwargs_from_config(cfg)
    if args.device:
        kwargs["device"] = args.device
    if args.no_compile:
        kwargs["use_compile"] = False
    batch_size = int(args.batch_size) or int(cfg.get("batch_size", 512))

    model_cfg = model_config_from_flat_config(cfg)
    model = build_model(model_cfg)
  # ⚑ model_config is not decoration: the trainer derives its input-history
  # encoding from it, and without it `select_input_history_arrays` refuses
  # every LC0-root row in the corpus. Same construction as tune/trainable.py.
    trainer = Trainer(model, model_config=model_cfg, **kwargs)
    params = sum(
        p.numel() for p in {id(p): p for p in model.parameters() if p.requires_grad}.values()
    )
    print(f"[model] {params} trainable params on {kwargs['device']}")

    buf = DiskReplayBuffer(
        capacity=10**9,
        shard_dir=out_dir / "staged_shards",
        rng=np.random.default_rng(int(args.seed)),
  # ⚑ NEVER False here. See stage_shards: a writable buffer enforces its
  # window in __init__ and would delete the converted shards through the
  # symlinks.
        read_only=True,
  # Deviations from the production buffer, both because the corpus is FIXED
  # rather than streaming:
  #  - recency exponent 0.0 (production 1.0) makes the shard draw UNIFORM.
  #    Production weights newest-first because new data is the scarce thing;
  #    here every hour is equally old and a recency weight would just
  #    over-train the last converted hour and call it an epoch.
  #  - deterministic_refresh makes the draw a pure function of the seed. The
  #    production default lets a background prefetch thread race the
  #    synchronous path, and the two consume different generators, so the same
  #    seed gives different data. An offline ruler needs the opposite.
        shard_recency_exponent=0.0,
        deterministic_refresh=True,
        input_planes=int(cfg.get("input_planes", 0)) or None,
    )
    print(f"[data] replay buffer: {len(buf)} rows in the hot shuffle pool "
          f"over {staged} shard(s)")

    with CaptureRealizedLosses() as capture:
        metrics = trainer.train_steps(
            _as_replay_buffer(buf), batch_size=batch_size, steps=int(args.steps),
        )

    if capture.calls == 0:
        raise SystemExit("compute_loss was never called — no step ran")

    print_realized(capture, metrics)

    readout = capture.worst
    if args.allow_leak:
        print("\n⚑⚑ --allow-leak: the value-blend guard is BYPASSED. This run "
              "is not a valid positive control.")
    else:
        assert_no_silent_outcome_fallback(readout, context="realized training step")
        print("\n[guard] PASS: zero SF-to-outcome leak on every observed step")

    ckpt = out_dir / "checkpoint.pt"
    trainer.save(ckpt)
    summary = {
        "steps": int(args.steps),
        "batch_size": batch_size,
        "trainable_params": params,
        "compute_loss_calls": capture.calls,
        "realized": dict(readout.as_table()),
        "metrics": {
            f.name: getattr(metrics, f.name)
            for f in dataclasses.fields(type(metrics))
            if isinstance(getattr(metrics, f.name), (int, float, bool, type(None)))
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[checkpoint] {ckpt}")
    print(f"[summary]    {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
