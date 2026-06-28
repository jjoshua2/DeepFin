#!/usr/bin/env python3
"""Self-play time-control validation harness for the DeepFin UCI engine.

Sweeps the time-management knobs (``--abort-factor`` / ``--time-budget-scale``)
by playing paired-opening games at a real game clock between a baseline config
and each candidate config, then reports per config:

  - **flag-safety**: how many games each side LOST ON TIME. This must be 0 — a
    flag is a whole point and the catastrophic failure mode of any
    time-management change, so the harness exits non-zero if any side flags.
  - **relative strength**: candidate score + crude Elo (with a 95% CI) vs the
    baseline.
  - **clock usage**: min time left at game end and average move time per side
    (when a ``--move-log-out`` CSV is produced).

It drives the existing, tested ``scripts/match_vs_uci.py`` once per candidate
(both engines are the same checkpoint, only the knob flags differ), writes a
PGN, and aggregates the PGN ``Result`` / ``Termination`` headers. The pure
aggregation logic is unit-tested; the game-playing itself needs a GPU + a
checkpoint, so run this on the target box.

Because it exits non-zero on any flag, it doubles as a pre-ship / CI gate for
time-management changes.

Usage::

    PYTHONPATH=. python scripts/validate_time_mgmt.py \\
        --checkpoint <path> --device cuda \\
        --clock-base-ms 10000 --clock-inc-ms 100 --games 50 \\
        --openings openings.epd \\
        --baseline "abort_factor=1.0,time_budget_scale=1.0" \\
        --candidate "abort_factor=0.7,time_budget_scale=1.0" \\
        --candidate "abort_factor=0.6,time_budget_scale=1.5"
"""
from __future__ import annotations

import argparse
import io
import json
import math
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# name -> deepfin CLI flag. The sweep only knows these knobs; an unknown knob in
# a --candidate/--baseline spec is rejected so a typo fails loudly.
_KNOB_FLAGS: dict[str, str] = {
    "abort_factor": "--abort-factor",
    "time_budget_scale": "--time-budget-scale",
    "optimum_fraction": "--optimum-fraction",
    "moves_horizon": "--moves-horizon",
  # Not a time knob, but the lever that couples NPS and stop-granularity, so the
  # sweep must reach it; only credited in --opponent-engine (absolute) mode.
    "chunk_sims": "--chunk-sims",
}


def parse_knobs(spec: str) -> dict[str, float]:
    """``"abort_factor=0.7,time_budget_scale=1.5"`` -> ``{...}``. Empty spec is
    the engine defaults. Unknown knobs raise rather than being silently dropped."""
    knobs: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"bad knob {part!r}: expected name=value")
        name, _, value = part.partition("=")
        name = name.strip()
        if name not in _KNOB_FLAGS:
            raise ValueError(f"unknown knob {name!r}; known: {sorted(_KNOB_FLAGS)}")
        knobs[name] = float(value)
    return knobs


def knobs_label(knobs: dict[str, float]) -> str:
    """A short, filesystem/PGN-safe label for a knob config."""
    if not knobs:
        return "default"
    return "_".join(f"{name}={knobs[name]:g}" for name in sorted(knobs))


def knob_flags(knobs: dict[str, float]) -> list[str]:
    """Knob config -> deepfin CLI flags, in a deterministic order."""
    out: list[str] = []
    for name in sorted(knobs):
        out += [_KNOB_FLAGS[name], f"{knobs[name]:g}"]
    return out


def build_engine_cmd(checkpoint: str, device: str, knobs: dict[str, float]) -> str:
    """The shell command that launches a deepfin UCI engine with ``knobs``."""
    parts = [
        sys.executable, "-m", "chess_anti_engine.uci",
        "--checkpoint", checkpoint, "--device", device,
        *knob_flags(knobs),
    ]
    return " ".join(shlex.quote(p) for p in parts)


def build_match_argv(
    match_script: Path | str,
    *,
    engine_baseline: str,
    engine_candidate: str,
    label_baseline: str,
    label_candidate: str,
    clock_base_ms: int,
    clock_inc_ms: int,
    games: int,
    max_plies: int,
    pgn_out: Path | str,
    openings: Path | str | None = None,
    move_log_out: Path | str | None = None,
) -> list[str]:
    """argv for one baseline-vs-candidate clock match via match_vs_uci.py.

    Engine A is the baseline, engine B the candidate, so the PGN's White/Black
    headers carry the config labels and ``aggregate_pgn`` can attribute flags."""
    argv = [
        sys.executable, str(match_script),
        "--engine-a", engine_baseline,
        "--engine-b", engine_candidate,
        "--label-a", label_baseline,
        "--label-b", label_candidate,
        "--clock-base-ms", str(clock_base_ms),
        "--clock-inc-ms", str(clock_inc_ms),
        "--games", str(games),
        "--max-plies", str(max_plies),
        "--pgn-out", str(pgn_out),
    ]
    if openings is not None:
        argv += ["--openings", str(openings)]
    if move_log_out is not None:
        argv += ["--move-log-out", str(move_log_out)]
    return argv


def _elo_from_score(score: float) -> float | None:
    """Crude logistic Elo from a win-rate score; None at a saturated 0/1."""
    if not 0.0 < score < 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def _score_ci95(points: list[float]) -> tuple[float, float] | None:
    """95% CI on the mean score (normal approx). None for <2 games."""
    n = len(points)
    if n < 2:
        return None
    mean = sum(points) / n
    var = sum((p - mean) ** 2 for p in points) / (n - 1)
    half = 1.96 * math.sqrt(var / n)
    return (max(0.0, mean - half), min(1.0, mean + half))


@dataclass
class ConfigResult:
    """Aggregated outcome of one candidate config vs the baseline."""
    label: str
    knobs: dict[str, float]
    games: int = 0
    candidate_points: float = 0.0
    cand_wins: int = 0
    draws: int = 0
    cand_losses: int = 0
    baseline_flags: int = 0
    candidate_flags: int = 0
  # When the baseline is a fixed external opponent (absolute-strength mode), an
  # opponent flag is not OUR bug, so flag-safety gates on the candidate only.
    opponent_mode: bool = False
    _points: list[float] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.candidate_points / self.games if self.games else 0.0

    @property
    def elo(self) -> float | None:
        return _elo_from_score(self.score)

    @property
    def elo_ci95(self) -> tuple[float | None, float | None] | None:
        ci = _score_ci95(self._points)
        if ci is None:
            return None
        return (_elo_from_score(ci[0]), _elo_from_score(ci[1]))

    @property
    def flag_safe(self) -> bool:
        if self.opponent_mode:
            return self.candidate_flags == 0
        return self.candidate_flags == 0 and self.baseline_flags == 0


def _flagger_label(result: str, white: str, black: str) -> str | None:
    """Which engine lost on time, given the game result. match_vs_uci returns
    result ``0-1`` when White flags and ``1-0`` when Black flags."""
    if result == "1-0":
        return black
    if result == "0-1":
        return white
    return None  # a timed-out draw shouldn't happen, but don't guess


def aggregate_pgn(
    pgn_text: str, *, label_baseline: str, label_candidate: str,
    label: str | None = None, knobs: dict[str, float] | None = None,
    opponent_mode: bool = False,
) -> ConfigResult:
    """Fold a match PGN into a ConfigResult from the candidate's perspective.

    Reads each game's White/Black/Result/Termination headers: scores the
    candidate, and on ``Termination == "time"`` attributes the flag to whichever
    engine lost on the clock. ``opponent_mode`` marks the baseline as a fixed
    external opponent so the flag-safety gate ignores its (non-ours) flags."""
    import chess.pgn  # heavy-ish; deferred so --help stays light

    res = ConfigResult(
        label=label if label is not None else label_candidate,
        knobs=knobs if knobs is not None else {},
        opponent_mode=opponent_mode,
    )
    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        h = game.headers
        white, black, result = h.get("White", ""), h.get("Black", ""), h.get("Result", "*")
        if result not in ("1-0", "0-1", "1/2-1/2"):
            continue  # unfinished / unknown; not a counted game
        res.games += 1
        cand_is_white = white == label_candidate
        if result == "1/2-1/2":
            point = 0.5
            res.draws += 1
        else:
            cand_won = (result == "1-0") == cand_is_white
            point = 1.0 if cand_won else 0.0
            res.cand_wins += int(cand_won)
            res.cand_losses += int(not cand_won)
        res.candidate_points += point
        res._points.append(point)
        if h.get("Termination") == "time":
            flagger = _flagger_label(result, white, black)
            if flagger == label_candidate:
                res.candidate_flags += 1
            elif flagger == label_baseline:
                res.baseline_flags += 1
    return res


def validation_passed(results: list[ConfigResult]) -> bool:
    """The harness's gate: every config must be flag-safe (no side lost on time)."""
    return all(r.flag_safe for r in results)


def format_summary(results: list[ConfigResult], *, baseline_label: str) -> str:
    lines = [f"baseline: {baseline_label}", ""]
    header = f"{'candidate':28} {'games':>5} {'score':>7} {'elo':>9} {'cand_flag':>9} {'base_flag':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        elo = "n/a" if r.elo is None else f"{r.elo:+.0f}"
        flag = "" if r.flag_safe else "  <-- FLAGGED"
        lines.append(
            f"{r.label:28} {r.games:>5} {r.score:>7.3f} {elo:>9} "
            f"{r.candidate_flags:>9} {r.baseline_flags:>9}{flag}"
        )
    lines.append("")
    lines.append(
        "PASS — no flags" if validation_passed(results)
        else "FAIL — a config lost on time (see <-- FLAGGED)"
    )
    return "\n".join(lines)


def _result_record(r: ConfigResult, *, baseline_label: str) -> dict[str, object]:
    ci = r.elo_ci95
    return {
        "candidate": r.label,
        "baseline": baseline_label,
        "knobs": r.knobs,
        "games": r.games,
        "score": r.score,
        "elo": r.elo,
        "elo_ci95": list(ci) if ci is not None else None,
        "cand_wins": r.cand_wins,
        "draws": r.draws,
        "cand_losses": r.cand_losses,
        "candidate_flags": r.candidate_flags,
        "baseline_flags": r.baseline_flags,
        "flag_safe": r.flag_safe,
    }


def _baseline_engine_and_label(
    args: argparse.Namespace, baseline_knobs: dict[str, float],
) -> tuple[str, str, bool]:
    """``(engine_cmd, label, opponent_mode)`` for the baseline side. A fixed
    external opponent (``--opponent-engine``) gives absolute-strength numbers;
    otherwise the baseline is our own engine at ``baseline_knobs`` (relative)."""
    if args.opponent_engine:
        return str(args.opponent_engine), str(args.opponent_label), True
    return build_engine_cmd(args.checkpoint, args.device, baseline_knobs), \
        knobs_label(baseline_knobs), False


def _run_one(
    args: argparse.Namespace, baseline_knobs: dict[str, float], cand_spec: str,
) -> ConfigResult:
    cand_knobs = parse_knobs(cand_spec)
    label = knobs_label(cand_knobs)
    engine_baseline, baseline_label, opponent_mode = _baseline_engine_and_label(
        args, baseline_knobs,
    )
    engine_candidate = build_engine_cmd(args.checkpoint, args.device, cand_knobs)
    with tempfile.TemporaryDirectory() as tmp:
        pgn_out = Path(tmp) / "match.pgn"
        argv = build_match_argv(
            args.match_script,
            engine_baseline=engine_baseline,
            engine_candidate=engine_candidate,
            label_baseline=baseline_label,
            label_candidate=label,
            clock_base_ms=args.clock_base_ms,
            clock_inc_ms=args.clock_inc_ms,
            games=args.games,
            max_plies=args.max_plies,
            pgn_out=pgn_out,
            openings=args.openings,
        )
        if args.dry_run:
            print("[dry-run] " + " ".join(shlex.quote(a) for a in argv), flush=True)
            return ConfigResult(label=label, knobs=cand_knobs, opponent_mode=opponent_mode)
        print(f"[validate] {label}: running {args.games} paired games…", flush=True)
        subprocess.run(argv, check=True)
        pgn_text = pgn_out.read_text()
    return aggregate_pgn(
        pgn_text,
        label_baseline=baseline_label,
        label_candidate=label,
        label=label,
        knobs=cand_knobs,
        opponent_mode=opponent_mode,
    )


def main() -> int:
    p = argparse.ArgumentParser(prog="validate_time_mgmt")
    p.add_argument("--checkpoint", required=True, help="path to trainer.pt / checkpoint dir")
    p.add_argument("--device", default="cuda", help="cpu|cuda|cuda:N (default: cuda)")
    p.add_argument("--baseline", default="abort_factor=1.0,time_budget_scale=1.0",
                   help="baseline knob config for our engine (relative mode; ignored when "
                        "--opponent-engine is set)")
    p.add_argument("--opponent-engine", default="",
                   help="absolute-strength mode: a fixed external opponent command (e.g. a "
                        "Cheese/SF binary). Each candidate plays THIS opponent, so the score is "
                        "absolute Elo, not a self-play delta. Required to credit chunk_sims/NPS "
                        "changes (self-play shares NPS and can't see them).")
    p.add_argument("--opponent-label", default="opponent",
                   help="PGN/summary label for the --opponent-engine side (default: opponent)")
    p.add_argument("--candidate", action="append", default=[],
                   help="candidate knob config, e.g. 'abort_factor=0.6,time_budget_scale=1.5' "
                        "(repeatable to sweep)")
    p.add_argument("--clock-base-ms", type=int, default=10000, help="base clock per side (ms)")
    p.add_argument("--clock-inc-ms", type=int, default=100, help="increment per move (ms)")
    p.add_argument("--games", type=int, default=50, help="paired games per candidate")
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--openings", type=Path, default=None, help="FEN/EPD opening file (paired)")
    p.add_argument("--match-script", type=Path,
                   default=Path(__file__).with_name("match_vs_uci.py"),
                   help="path to match_vs_uci.py (default: sibling)")
    p.add_argument("--out", type=Path, default=None, help="append JSONL result records here")
    p.add_argument("--dry-run", action="store_true",
                   help="print the match commands without running them (no GPU needed)")
    args = p.parse_args()

    if not args.candidate:
        p.error("at least one --candidate config is required")
    baseline_knobs = parse_knobs(args.baseline)
    for spec in args.candidate:
        parse_knobs(spec)  # validate all up front so a typo fails before any game

    results = [_run_one(args, baseline_knobs, spec) for spec in args.candidate]
    if args.dry_run:
        return 0

    baseline_label = (
        str(args.opponent_label) if args.opponent_engine else knobs_label(baseline_knobs)
    )
    print("\n" + format_summary(results, baseline_label=baseline_label), flush=True)
    if args.out is not None:
        with args.out.open("a") as fh:
            for r in results:
                fh.write(json.dumps(_result_record(r, baseline_label=baseline_label)) + "\n")
    return 0 if validation_passed(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
