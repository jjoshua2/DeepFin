#!/usr/bin/env python3
"""Read the LIVE production architecture and compare the control against it.

⚑ The point of this script is that it reads a file OUTSIDE this working tree.
`tests/test_lc0_control_config.py` diffs the control against the in-tree
`configs/pbt2_small.yaml`, which is the right *design* and cannot see the live
run: the live yaml lives in the live working tree, on the live branch, and is
re-read every iteration. See `chess_anti_engine/eval/lc0_control_arch.py`.

    # print the pin block to paste into lc0_control_arch.LIVE_ARCH_PIN
    PYTHONPATH=. python3 scripts/lc0_control_arch_pin.py \
        --live-config /path/to/live/configs/pbt2_small.yaml

    # gate: exit 1 if the control's model: section is not the live one
    PYTHONPATH=. python3 scripts/lc0_control_arch_pin.py --check \
        --live-config /path/to/live/configs/pbt2_small.yaml

`--live-config` may be omitted if `$CHESS_LIVE_PRODUCTION_CONFIG` is set. With
neither, `--check` judges against the recorded pin and says so loudly, because
"the live file was not read" is a materially weaker claim than "it was".

⚑ `--emit` must be run from a tree whose code can BUILD the live config. If the
live yaml carries a `model:` key this tree's schema does not know, the flattener
raises — that is category (a) from CLAUDE.md and it is the answer, not an error.
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
from chess_anti_engine.utils import load_yaml_file


def _emit(live_config: Path) -> int:
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


def _check(control_config: Path, live_config: Path | None) -> int:
    raw = load_yaml_file(str(control_config))
    try:
        provenance = assert_control_matches_live_architecture(
            raw, live_config=live_config, context="lc0_control_arch_pin --check",
        )
    except ControlArchitectureDrift as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"OK: {control_config} matches {provenance}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--live-config", type=Path, default=None)
    parser.add_argument(
        "--control-config", type=Path,
        default=Path("configs/lc0_positive_control.yaml"),
    )
    parser.add_argument("--check", action="store_true", help="gate instead of print")
    args = parser.parse_args(argv)

    live = args.live_config or live_production_config_path()
    if args.check:
        return _check(Path(args.control_config), live)
    if live is None:
        print(
            "no --live-config and no $CHESS_LIVE_PRODUCTION_CONFIG; the recorded "
            f"pin is:\n{json.dumps(LIVE_ARCH_PIN, indent=4, sort_keys=True)}",
        )
        return 0
    return _emit(Path(live))


if __name__ == "__main__":
    sys.exit(main())
