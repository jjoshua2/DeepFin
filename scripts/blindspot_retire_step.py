"""Live auto-retirement step for the cadenced FEN-seed monitor.

Re-scores the CURRENT live seed list (from the yaml's opening_fen_list_path) with
a checkpoint, tracks per-seed consecutive-AWARE reads in a state file, and RETIRES
seeds that read AWARE (net_q <= --resolved-below) TWICE IN A ROW — so a single-read
fluke never drops a seed, and a seed that regresses resets its streak.

Retiring writes a NEW versioned seed file (minus the retired seeds — required, since
selfplay/opening.py::_load_fen_list is lru_cached by PATH, so an in-place edit is
ignored), points the live yaml's opening_fen_list_path at it, and VALIDATES the
strict reload — reverting the yaml on ANY error so a bad write can never freeze the
live config. Prints one summary line for the monitor log. Fail-soft: never raises.

Intended to be called once per monitor cadence with the same checkpoint the panels
use. Off unless --yaml + --checkpoint are given; --dry-run scores + reports only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from chess_anti_engine.selfplay.opening import seed_board_from_line
from scripts.blindspot_resolution import load_seed_lines, score_seeds


def _placement(line: str) -> str:
    """Position key = piece placement only (ignore move counters / ep)."""
    return seed_board_from_line(line.split("#", 1)[0].strip()).fen().split()[0]


def update_streaks(
    keys: list[str], net_q: list[float], state: dict[str, int], *,
    resolved_below: float, min_consecutive: int,
) -> tuple[set[str], int]:
    """Advance each seed's consecutive-AWARE streak (mutates ``state``) and return
    (retire_keys, resolved_count). A seed AWARE this read (net_q <= resolved_below)
    increments its streak; otherwise the streak RESETS to 0. A seed is retired once
    its streak reaches ``min_consecutive`` — i.e. it read AWARE that many reads in a
    row, so a single-read fluke never drops it and a regression un-arms it."""
    resolved = 0
    for k, qi in zip(keys, net_q):
        if qi <= resolved_below:
            state[k] = int(state.get(k, 0)) + 1
            resolved += 1
        else:
            state[k] = 0
    retire_keys = {k for k in keys if state.get(k, 0) >= min_consecutive}
    return retire_keys, resolved


def _current_seed_path(yaml_path: str) -> str:
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults, load_yaml_file
    return str(flatten_run_config_defaults(load_yaml_file(yaml_path))["opening_fen_list_path"])


def _write_seed_file(path: str, lines: list[str], *, note: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {note}\n")
        for ln in lines:
            fh.write(ln + "\n")


def _repoint_yaml(yaml_path: str, new_seed_path: str) -> bool:
    """Point opening_fen_list_path at new_seed_path; validate the strict reload and
    REVERT on any failure. Returns True only if the live yaml now loads cleanly."""
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults, load_yaml_file

    with open(yaml_path, encoding="utf-8") as fh:
        original = fh.read()
    pat = re.compile(r"^(\s*opening_fen_list_path:\s*)\S+.*$", re.MULTILINE)
    if len(pat.findall(original)) != 1:
        print("[retire] ABORT: opening_fen_list_path not uniquely found in yaml", file=sys.stderr)
        return False
    updated = pat.sub(lambda m: m.group(1) + new_seed_path, original, count=1)
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(updated)
    try:
        flat = flatten_run_config_defaults(load_yaml_file(yaml_path))  # strict validator
        assert str(flat["opening_fen_list_path"]) == new_seed_path
        return True
    except Exception as e:  # bad write -> restore so the live reload never freezes
        with open(yaml_path, "w", encoding="utf-8") as fh:
            fh.write(original)
        print(f"[retire] ABORT: yaml validation failed ({type(e).__name__}); reverted", file=sys.stderr)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--yaml", default="configs/pbt2_small.yaml")
    ap.add_argument("--state", default="scratchpad/live_read/retire_state.json")
    ap.add_argument("--resolved-below", type=float, default=-0.4,
                    help="net_q <= this = AWARE/solved (the panel's AWARE bar)")
    ap.add_argument("--min-consecutive", type=int, default=2,
                    help="retire only after this many consecutive AWARE reads")
    ap.add_argument("--min-pool", type=int, default=20, help="never retire below this many seeds")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--tag", default="", help="filename tag for the new list (e.g. checkpoint num)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        seed_path = _current_seed_path(args.yaml)
        lines = load_seed_lines(seed_path)
        q = score_seeds(args.checkpoint, lines, device=args.device,
                        gpu_mem_fraction=args.gpu_mem_fraction)
        keys = [_placement(ln) for ln in lines]

        state: dict[str, int] = {}
        if os.path.exists(args.state):
            with open(args.state, encoding="utf-8") as fh:
                state = json.load(fh)

        retire_keys, resolved_now = update_streaks(
            keys, q.tolist(), state,
            resolved_below=args.resolved_below, min_consecutive=args.min_consecutive)
        keep = [ln for ln, k in zip(lines, keys) if k not in retire_keys]

        did_retire = False
        new_base = os.path.basename(seed_path)
        if retire_keys and len(keep) >= args.min_pool and len(keep) < len(lines):
            if args.dry_run:
                did_retire = True  # report intent
            else:
                tag = args.tag or "auto"
                new_path = os.path.join(args.out_dir, f"blindspot_fens_retire_{tag}.txt")
                _write_seed_file(new_path, keep,
                                 note=f"auto-retired {len(retire_keys)} seeds (2x-consecutive AWARE "
                                      f"@net_q<={args.resolved_below}) from {new_base}")
                # only repoint if the new file loads and the yaml stays valid
                from chess_anti_engine.selfplay.opening import _load_fen_list
                if len(_load_fen_list(new_path)) >= args.min_pool and _repoint_yaml(args.yaml, os.path.abspath(new_path)):
                    for k in retire_keys:
                        state.pop(k, None)
                    new_base = os.path.basename(new_path)
                    did_retire = True

        if not args.dry_run:
            os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)
            with open(args.state, "w", encoding="utf-8") as fh:
                json.dump(state, fh)

        pool_now = len(keep) if did_retire else len(lines)
        print(f"retire: pool={len(lines)} resolved={resolved_now}/{len(lines)} "
              f"(net<={args.resolved_below}) retired={'DRY:' if args.dry_run else ''}{len(retire_keys) if did_retire else 0} "
              f"(2x) -> pool={pool_now} list={new_base}")
    except Exception as e:  # side tool — must never break the monitor
        print(f"retire: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
