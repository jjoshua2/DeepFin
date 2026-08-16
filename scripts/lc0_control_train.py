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

Four checks, at four different levels, because each can be satisfied by a run
that still does the wrong thing:

  0. LAUNCH, architecture. `assert_control_matches_live_architecture` — the
     arm's premise is "test THE STACK WE RUN", and the in-tree
     `configs/pbt2_small.yaml` is not the file the live run reads. This one
     judges against the LIVE yaml when `$CHESS_LIVE_PRODUCTION_CONFIG` names
     it, and against a recorded pin otherwise.
  1. LAUNCH, config-level. `assert_pid_cannot_reassert_sf_wdl` — an override
     the difficulty controller can undo is not an override.
  2. LAUNCH, corpus-level. The converter's own `run_config_problems` is REUSED
     (not restated) against the actual shard directories, driven by the
     MEASURED coverage of every directory — of BOTH value labels, `sf_wdl` and
     `search_wdl` — rather than `any()`.
  3. ⚑ REALIZED, per-step. `compute_loss` is wrapped for the duration of the
     run and the fracs it is ACTUALLY CALLED WITH — together with the batch's
     own EFFECTIVE label mass for both components — are fed to
     `value_blend_guard`. A configured value is not an applied value; this is
     the only one of the four that measures the applied one.

Check 3 fails loudly (non-zero exit, no checkpoint written).

⚑ `--allow-leak` DOWNGRADES ONLY THE LAUNCH GUARDS (1 and 2). It used to skip
check 3 as well, which made check 3 — the headline of this script —
UNREACHABLE: guard 1 refuses every `sf_wdl_frac > 0` config outright, nothing
downstream can raise the frac, and the only way past guard 1 also skipped the
assert. So `--allow-leak` on a production-blend config now runs the steps,
prints the realized leak, and then EXITS 1 WITHOUT A CHECKPOINT. That is the
demonstration; `scratchpad/lc0_positive_control/RIG_VERIFICATION_20260816.md`
records it.

⚑ AND UNDER `--allow-leak` THE PARTIAL-CORPUS REFUSAL IS A PRINT, SO A MIXED
CORPUS *IS* REACHABLE — check 3 is the only thing left, and it is what actually
holds (measured 14/14 seeds, PR #438's re-review). Do not read "the launch
guards make the states below unreachable" as a proof about the flag path; it is
a proof about the default path only.

`--allow-arch-drift` does the same for check 0, and exists because the arm's
plumbing has to be smoke-testable on a branch whose code cannot yet build the
live architecture. Both flags stamp `valid_control: false` into summary.json,
as does omitting `--purity-receipt` — with the reason named in
`validity_problems`.

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

from chess_anti_engine.eval.lc0_control_arch import (
    ControlArchitectureDrift,
    assert_control_matches_live_architecture,
    live_production_config_path,
    unique_storage_param_count,
)
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
from scripts.lc0_data_to_rows import (
    run_config_problems,
    shard_dir_search_wdl_coverage,
    shard_dir_sf_wdl_coverage,
)


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
  # ⚑ EFFECTIVE, not `sf_wdl_rows`. `sf_wdl_rows` is the LABEL count: it
  # ignores the `sf_search_dampen_*` shortfall and reads non-zero on a batch
  # with no `sf_wdl` column at all. And the search count is not optional — it
  # is 0.70 of this arm's value target and had no reader until review F1.
            sf_effective_rows=float(result["sf_wdl_effective_rows"].detach().item()),
            search_effective_rows=float(
                result["search_wdl_effective_rows"].detach().item(),
            ),
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


def purity_receipt_problems(
    receipt_path: Path, shard_dirs: list[Path],
) -> tuple[dict[str, Any], list[str]]:
    """The banked purity check, and every reason it does not cover THIS corpus.

    ⚑ REVIEW F5. ``purity --train-shards`` and ``--shards`` here were two
    unconnected CLI arguments: nothing in ``summary.json`` recorded which
    directories were trained on, so the arm's held-out slope had no
    machine-checkable evidence that the corpus it trained on is the corpus
    purity cleared. Set equality against the RESOLVED directories is the check
    — not a count, not a "looks right", and not a flag somebody sets by hand.
    """
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    covered = {str(Path(d)) for d in receipt.get("train_shards", ())}
    used = {str(Path(d).resolve()) for d in shard_dirs}
    problems: list[str] = []
    if not receipt.get("pure", False):
        problems.append(
            f"the purity receipt {receipt_path} records pure=False "
            f"({receipt.get('exposed_inputs')} exposed inputs). The held-out "
            "set measures exposure recency, not generalisation.",
        )
    if used - covered:
        problems.append(
            "these --shards directories are NOT covered by the purity receipt: "
            + ", ".join(sorted(used - covered))
            + ". The receipt cleared: " + (", ".join(sorted(covered)) or "<none>"),
        )
    return receipt, problems


def preflight(
    cfg: dict[str, Any], shard_dirs: list[Path], *, allow_leak: bool,
) -> dict[str, dict[str, int]]:
    """The two LAUNCH-level value-blend guards. Returns the measured coverage.

    The coverage is RETURNED rather than only printed so ``summary.json`` can
    record what the run actually read — a corpus identity nobody can reconstruct
    from a log line that scrolled past (review F5).

    ``allow_leak`` downgrades them to a banner instead of skipping them. The
    point of the flag is to let the run REACH the realized (per-step) check so
    the leak can be observed as a number — and, since that check now also
    runs under the flag, so it can be observed RAISING.
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

  # ⚑ COVERAGE, not `any()`. A single SF-labelled row anywhere in a mixed
  # --shards list used to set have_sf=True, which made `run_config_problems`
  # return [] and waved the whole corpus through (Codex #2). The question the
  # gate needs answered is "do ALL these rows carry a label", so measure it.
  #
  # ⚑⚑ AND BOTH LABELS, not just the SF one (review F1). `search_wdl_frac` is
  # 0.70 of this arm's value target and `compute_loss` falls the search
  # component back to the raw one-hot on an unlabelled row exactly as it falls
  # the SF component back. Measuring one and assuming the other is how the
  # larger term went unchecked.
    coverage: dict[str, tuple[int, int]] = {}
    for flag, measure in (
        ("sf_wdl", shard_dir_sf_wdl_coverage),
        ("search_wdl", shard_dir_search_wdl_coverage),
    ):
        labelled = rows = 0
        for shard_dir in shard_dirs:
            dir_labelled, dir_rows = measure(Path(shard_dir))
            labelled += dir_labelled
            rows += dir_rows
            print(f"[preflight] {Path(shard_dir).name}: {flag} label coverage "
                  f"{dir_labelled}/{dir_rows}")
        coverage[flag] = (labelled, rows)

    complete: dict[str, bool] = {}
    for flag, (labelled, rows) in coverage.items():
        complete[flag] = rows > 0 and labelled == rows
        print(f"[preflight] {flag} coverage over all shards: {labelled}/{rows} "
              f"-> treating the corpus as {flag}-labelled: {complete[flag]}")
        if 0 < labelled < rows:
            fail(
                f"the corpus is PARTIALLY {flag}-labelled ({labelled}/{rows} "
                "rows). Every value-blend decision downstream assumes one "
                "regime or the other; split the run rather than averaging two "
                "corpora.",
            )
    have_sf = complete["sf_wdl"]
    problems = run_config_problems(
        cfg,
        shards_have_sf_wdl=have_sf,
        shards_have_search_wdl=complete["search_wdl"],
    )
    if problems:
        fail("the config is wrong for these shards:\n  " + "\n  ".join(problems))
    else:
        print("[preflight] converter run-config gate: no problems")
    return {
        flag: {"labelled_rows": labelled, "rows": rows}
        for flag, (labelled, rows) in coverage.items()
    }


def preflight_architecture(
    config_path: Path, *, allow_drift: bool, model: torch.nn.Module | None = None,
) -> str:
    """LAUNCH guard 0 — the arm must build the architecture PRODUCTION RUNS.

    Called TWICE, and the second call is the one that means something new.
    Before the model exists this can only compare ``model:`` sections; once it
    exists, ``model=`` puts the pinned 61,444,448 against the net this tree's
    code actually built from those keys. Two different failures — a config that
    says the wrong thing, and code that builds the wrong thing from the right
    config — and only the second is visible after ``build_model``.

    ⚑ The env var is resolved HERE and passed in, not read inside the assert:
    a library function whose verdict depends on the caller's shell fails
    differently in the operator's terminal than in the test suite (review F6).
    """
    raw = load_yaml_file(str(config_path))
    stage = "built model" if model is not None else "config"
    try:
        provenance = assert_control_matches_live_architecture(
            raw, model=model, live_config=live_production_config_path(),
            context=f"lc0 control launch ({stage})",
        )
    except ControlArchitectureDrift as exc:
        if not allow_drift:
            raise SystemExit(f"REFUSING TO LAUNCH — {exc}") from exc
        print(f"⚑⚑ --allow-arch-drift: THIS IS NOT A VALID POSITIVE CONTROL — {exc}")
        return "DRIFTED (--allow-arch-drift)"
    print(f"[preflight] architecture ({stage}) matches {provenance}")
    return provenance


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
        help="⚑ downgrade the LAUNCH guards to a banner so the run reaches the "
             "REALIZED per-step guard and that guard can be observed raising. "
             "It does NOT skip the realized guard. The run is NOT a valid control.",
    )
    parser.add_argument(
        "--allow-arch-drift", action="store_true",
        help="⚑⚑ proceed even though the config does not build production's "
             "live architecture. PLUMBING SMOKE RUNS ONLY — no number from such "
             "a run may be quoted as 'our stack'.",
    )
    parser.add_argument(
        "--purity-receipt", type=Path, default=None,
        help="the JSON written by `lc0_control_heldout.py purity --receipt`. "
             "Refuses to launch unless it covers every --shards directory, and "
             "banks its frozen sha256 in summary.json. WITHOUT IT the run is "
             "recorded as not a valid control: nothing then ties the trained "
             "corpus to the held-out purity check.",
    )
    args = parser.parse_args(argv)

    arch_provenance = preflight_architecture(
        Path(args.config), allow_drift=bool(args.allow_arch_drift),
    )
    cfg = flatten_run_config_defaults(load_yaml_file(str(args.config)))
    shard_dirs = [Path(d) for d in args.shards]
    coverage = preflight(cfg, shard_dirs, allow_leak=bool(args.allow_leak))

    receipt: dict[str, Any] | None = None
    receipt_path = Path(args.purity_receipt) if args.purity_receipt else None
    if receipt_path is not None:
        receipt, receipt_problems = purity_receipt_problems(
            receipt_path, shard_dirs,
        )
        if receipt_problems:
            raise SystemExit(
                "REFUSING TO LAUNCH — the purity receipt does not cover this "
                "corpus:\n  " + "\n  ".join(receipt_problems),
            )
        print(f"[preflight] purity receipt covers every --shards directory "
              f"(frozen sha256 {receipt.get('frozen_sha256')})")
    else:
        print("⚑ no --purity-receipt: NOTHING ties this corpus to a held-out "
              "purity check, and summary.json will record valid_control: false")

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
  # ⚑ Unique STORAGE, not sum(numel) over the state_dict: the 16
  # `layer_smolgens.N.gen_weight.weight` keys are one shared tensor (CLAUDE.md).
    params = unique_storage_param_count(model)
    print(f"[model] {params} trainable params on {kwargs['device']}")
  # ⚑ GUARD 0b. The pinned count gated NOTHING until 2026-08-16: no caller ever
  # passed `model=`, so the arch fix's own headline number (61,444,448) was a
  # decoration inside the module that fixed a decoration (review F4). This is
  # the only check that can see this tree's code building a different net from
  # the same `model:` keys — which is exactly what the bt4heads promotion does.
    preflight_architecture(
        Path(args.config), allow_drift=bool(args.allow_arch_drift), model=model,
    )

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
  # ⚑ NOT gated on --allow-leak. Guard 1 refuses every sf_wdl_frac > 0 config
  # and nothing on this call graph can raise the frac afterward, so skipping
  # the assert here left it with no reachable input that could fire it. The
  # flag opens the LAUNCH gate; this one still closes.
    try:
        assert_no_silent_outcome_fallback(readout, context="realized training step")
    except ValueBlendMisconfigured as exc:
        raise SystemExit(
            f"REALIZED VALUE-BLEND GUARD FAILED — no checkpoint written.\n{exc}",
        ) from exc
    print("\n[guard] PASS: no SF-to-outcome leak and no all-outcome value "
          "target on any observed step")

    ckpt = out_dir / "checkpoint.pt"
    trainer.save(ckpt)
  # ⚑ EVERY reason the run is not a valid control, NAMED. A bare
  # `valid_control: false` says a run is disqualified without saying what for,
  # which is the same "a flag instead of a measurement" shape as the rest of
  # this review's findings.
    validity_problems = [
        message for flag, message in (
            (args.allow_leak, "--allow-leak: the LAUNCH value-blend guards were "
                              "downgraded to banners"),
            (args.allow_arch_drift, "--allow-arch-drift: this is NOT production's "
                                    "architecture"),
            (receipt is None, "no --purity-receipt: the trained corpus is not "
                              "tied to any held-out purity check"),
        ) if flag
    ]
    summary = {
        "steps": int(args.steps),
        "batch_size": batch_size,
        "trainable_params": params,
        "architecture_judged_against": arch_provenance,
        "valid_control": not validity_problems,
        "validity_problems": validity_problems,
  # ⚑ REVIEW F5 — CORPUS IDENTITY. `--shards` was a CLI argument that left no
  # trace in the artifact, so the run's headline number could not be tied to
  # the rows that produced it or to the purity check that cleared them.
        "corpus": {
            "shard_dirs": [str(Path(d).resolve()) for d in shard_dirs],
            "staged_shards": staged,
            "label_coverage": coverage,
        },
        "purity_receipt": (
            None if receipt is None or receipt_path is None else {
                "path": str(receipt_path.resolve()),
                "frozen": receipt.get("frozen"),
                "frozen_sha256": receipt.get("frozen_sha256"),
                "train_shards": receipt.get("train_shards"),
                "exposed_inputs": receipt.get("exposed_inputs"),
                "train_rows": receipt.get("train_rows"),
            }
        ),
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
