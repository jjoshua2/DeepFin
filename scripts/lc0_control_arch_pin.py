#!/usr/bin/env python3
"""Read the LIVE production config and compare the control against it.

⚑ The point of this script is that it reads a file OUTSIDE this working tree.
No committed test can: the live yaml lives in the live working tree, on the
live branch, and is re-read every iteration, while this tree carries `main`'s
copy — stale by the whole bt4heads bundle as of 2026-08-16. The in-tree tests
judge the architecture against `LIVE_ARCH_PIN` and the trainer against
`LIVE_TRAINER_PIN`, and THIS SCRIPT is what makes those pins honest: `--emit`
regenerates one from the live file and `--check` compares the control straight
against the live file. See `chess_anti_engine/eval/lc0_control_arch.py` and
`chess_anti_engine/eval/lc0_control_trainer.py`.

⚑ TWO AXES, because "our EXACT net and trainer" is two claims and they went
stale independently — the architecture axis was re-pointed at the live file on
2026-08-16 while the trainer axis stayed on `main`'s copy, thirteen
`trainer_kwargs_from_config` entries behind it, for the rest of the same day.
`--check` does BOTH by default.

    # print a pin block to paste into the matching module
    PYTHONPATH=. python3 scripts/lc0_control_arch_pin.py --axis arch \
        --live-config /path/to/live/configs/pbt2_small.yaml
    PYTHONPATH=. python3 scripts/lc0_control_arch_pin.py --axis trainer \
        --live-config /path/to/live/configs/pbt2_small.yaml

    # gate: exit 1 if the control's model: section or trainer recipe is not live's
    PYTHONPATH=. python3 scripts/lc0_control_arch_pin.py --check \
        --live-config /path/to/live/configs/pbt2_small.yaml

`--live-config` may be omitted if `$CHESS_LIVE_PRODUCTION_CONFIG` is set. With
neither, `--check` judges against the recorded pin and says so loudly, because
"the live file was not read" is a materially weaker claim than "it was".

⚑ `--emit` must be run from a tree whose code can BUILD the live config. If the
live yaml carries a `model:` key this tree's schema does not know, the flattener
raises — that is category (a) from CLAUDE.md and it is the answer, not an error.
This tree could not build the live config until PR #439 (2026-08-16) put the
bt4heads keys into `main`'s schema; it can now, and the pin was regenerated
here to prove it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chess_anti_engine.eval.lc0_control_arch import (
    LIVE_ARCH_PIN,
    ControlArchitectureDrift,
    assert_control_matches_live_architecture,
    live_production_config_path,
    model_section,
    unique_storage_param_count,
)
from chess_anti_engine.eval.lc0_control_trainer import (
    LIVE_TRAINER_PIN,
    ControlTrainerDrift,
    assert_control_matches_live_trainer,
    trainer_kwargs_signature,
)
from chess_anti_engine.utils import load_yaml_file

AXES = ("arch", "trainer", "both")


def _emit_arch(live_config: Path) -> int:
    from chess_anti_engine.model import build_model, model_config_from_flat_config
    from chess_anti_engine.utils import flatten_run_config_defaults

    raw = load_yaml_file(str(live_config))
    flat = flatten_run_config_defaults(raw)
    model = build_model(model_config_from_flat_config(flat))
    state = model.state_dict()
    pin = {
        "recorded": "<today>",
        "live_branch": "<branch>",
        "live_commit": "<commit>",
        "provenance": "<why this changed>",
        "trainable_params_unique_storage": unique_storage_param_count(model),
        "state_dict_numel_sum": int(sum(v.numel() for v in state.values())),
        "model": model_section(raw),
    }
    print(json.dumps(pin, indent=4, sort_keys=True))
    return 0


def _emit_trainer(live_config: Path) -> int:
    from chess_anti_engine.utils import flatten_run_config_defaults

    flat = flatten_run_config_defaults(load_yaml_file(str(live_config)))
    pin = {
        "recorded": "<today>",
        "live_branch": "<branch>",
        "live_commit": "<commit>",
        "provenance": "<why this changed>",
        "kwargs": trainer_kwargs_signature(flat),
    }
    print(json.dumps(pin, indent=4, sort_keys=True))
    return 0


def _check(control_config: Path, live_config: Path | None, axis: str) -> int:
    raw = load_yaml_file(str(control_config))
    failed = False
    if axis in ("arch", "both"):
        try:
            provenance = assert_control_matches_live_architecture(
                raw, live_config=live_config, context="lc0_control_arch_pin --check",
            )
        except ControlArchitectureDrift as exc:
            print(str(exc), file=sys.stderr)
            failed = True
        else:
            print(f"OK: {control_config} architecture matches {provenance}")
    if axis in ("trainer", "both"):
        from chess_anti_engine.utils import flatten_run_config_defaults

        try:
            provenance = assert_control_matches_live_trainer(
                flatten_run_config_defaults(raw), live_config=live_config,
                context="lc0_control_arch_pin --check",
            )
        except ControlTrainerDrift as exc:
            print(str(exc), file=sys.stderr)
            failed = True
        else:
            print(f"OK: {control_config} trainer matches {provenance}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--live-config", type=Path, default=None)
    parser.add_argument(
        "--control-config", type=Path,
        default=Path("configs/lc0_positive_control.yaml"),
    )
    parser.add_argument("--check", action="store_true", help="gate instead of print")
    parser.add_argument(
        "--axis", choices=AXES, default="both",
        help="which pin to emit or check. ⚑ `both` is the DEFAULT for --check "
             "because the two axes failed independently once already: the "
             "architecture was re-pointed at the live file on 2026-08-16 and "
             "the trainer stayed 13 kwargs behind it for the rest of the day. "
             "A gate you can pass by checking half of it is half a gate.",
    )
    args = parser.parse_args(argv)

    live = args.live_config or live_production_config_path()
    if args.check:
        return _check(Path(args.control_config), live, str(args.axis))
    if live is None:
        pins = {"arch": LIVE_ARCH_PIN, "trainer": LIVE_TRAINER_PIN}
        shown = pins if args.axis == "both" else {args.axis: pins[args.axis]}
        print(
            "no --live-config and no $CHESS_LIVE_PRODUCTION_CONFIG; the recorded "
            f"pin(s):\n{json.dumps(shown, indent=4, sort_keys=True)}",
        )
        return 0
    if args.axis == "both":
        raise SystemExit(
            "--axis both cannot EMIT: the two pins are separate literals in "
            "separate modules and printing them together would invite pasting "
            "one into the other. Emit them one at a time.",
        )
    return _emit_arch(Path(live)) if args.axis == "arch" else _emit_trainer(Path(live))


if __name__ == "__main__":
    sys.exit(main())
