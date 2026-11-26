"""Arena matches on held-out EVAL openings, and the per-generation tripwires.

Matches run ``scripts/arena_standard.py`` in a subprocess (``matched_sims``,
``--search-shape training``, fixed seed, fixed N, no SPRT) and only the FINAL
record each arena appends is read and copied to the run ledger. A partial
arena (no manifest) is deleted and replayed whole.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chess_anti_engine.simple_rl.common import REPO_ROOT, append_jsonl, input_ref, read_manifest, write_manifest

LEDGER = "ledger.jsonl"


@dataclass(frozen=True)
class EvalSettings:
    games: int = 200
    sims: int = 64
    seed: int = 20261119
    device: str = "cuda"
    syzygy: str | None = None
    max_plies: int = 300
    opening_plies: int = 16
    max_concurrent_games: int = 64
    root_noise: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)


def _engine_name(checkpoint: Path) -> str:
    """``gen_0003`` for a loop checkpoint, else the file stem (PGN player names)."""
    p = Path(checkpoint)
    for part in (p.parent.name, p.parent.parent.name):
        if part.startswith("gen_"):
            return part
    return p.stem


def arena_command(*, candidate: Path, reference: Path, openings_pgn: Path, out: Path, label: str,
                  settings: EvalSettings) -> list[str]:
    if settings.games <= 0 or settings.games % 2:
        raise ValueError("arena games must be a positive even number (paired openings)")
    cmd = [sys.executable, "-u", str(REPO_ROOT / "scripts" / "arena_standard.py"),
           "--candidate", str(candidate), "--reference", str(reference),
           "--mode", "matched_sims", "--search-shape", "training",
           "--sims", str(settings.sims), "--games", str(settings.games),
           "--openings", str(openings_pgn), "--opening-plies", str(settings.opening_plies),
           "--max-plies", str(settings.max_plies), "--seed", str(settings.seed),
           "--device", settings.device, "--compile", "off",
           "--max-concurrent-games", str(settings.max_concurrent_games),
           "--label", label, "--out", str(out), "--pgn-out", str(out.with_suffix(".pgn")),
           "--games-out", str(out.with_suffix(".games.jsonl")),
           "--pgn-candidate-name", f"candidate_{_engine_name(candidate)}",
           "--pgn-reference-name", f"reference_{_engine_name(reference)}"]
    if not settings.root_noise:
        # The RL-takeover yardstick's setting: the arena cannot express the
        # per-ply self-play noise schedule, so it plays without root noise.
        cmd.append("--no-gumbel-noise")
    if settings.syzygy:
        cmd += ["--syzygy", settings.syzygy]
    return cmd + list(settings.extra_args)


def run_arena(cmd: list[str], *, out: Path, log: Path) -> dict[str, Any]:
    out.unlink(missing_ok=True)
    with log.open("w") as fh:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, check=False,
                              env=_child_env())
    if proc.returncode != 0:
        tail = log.read_text().strip().splitlines()[-1:] or [""]
        hint = (" (strict Syzygy arena: raise --eval-max-plies or pass --eval-no-syzygy)"
                if "unresolved game" in tail[0] else "")
        raise RuntimeError(f"arena exited {proc.returncode}: {tail[0]}{hint}; see {log}")
    rows = [json.loads(r) for r in out.read_text().splitlines() if r.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one final arena record in {out}, found {len(rows)}")
    return check_complete(rows[0], out)


def check_complete(record: dict[str, Any], out: Path) -> dict[str, Any]:
    """A fixed-N yardstick: a truncated or short arena is not this generation's result."""
    games, requested = record.get("games"), record.get("games_requested")
    if record.get("truncated") or games is None or requested is None or int(games) != int(requested):
        raise RuntimeError(f"arena record in {out} is incomplete (games={games}, games_requested={requested}, "
                           f"truncated={record.get('truncated')}); a fixed-N result needs every game")
    return record


def _child_env() -> dict[str, str]:
    import os

    env = dict(os.environ)  # carries CAE_GPU_GATE_PASSED from the driver's gate (scripts/simple_rl_loop.py run)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def evaluate_generation(*, run_dir: Path, generation: int, candidate: Path, previous: Path,
                        reference: Path | None, openings_pgn: Path, settings: EvalSettings,
                        pause_markers: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Run the generation's matches. ``<run>/PAUSE`` and ``pause_markers`` pause
    each arena cooperatively (``--pause-marker``): it holds its in-flight games and
    continues when they are removed; paused time is excluded from its timings."""
    from chess_anti_engine.simple_rl.common import PAUSE_FILE

    run_dir = Path(run_dir)
    pause_args = [x for m in (run_dir / PAUSE_FILE, *pause_markers) for x in ("--pause-marker", str(m))]
    out_dir = run_dir / f"gen_{generation:04d}" / "eval"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    matches = [("vs_previous", previous)]
    if reference is not None and Path(reference).resolve() != Path(previous).resolve():
        matches.append(("vs_reference", reference))
    results: dict[str, Any] = {}
    ledger_rows: list[dict[str, Any]] = []
    for name, opponent in matches:
        label = f"simple_rl_gen{generation:04d}_{name}"
        out = out_dir / f"arena_{name}.jsonl"
        cmd = arena_command(candidate=candidate, reference=opponent, openings_pgn=openings_pgn,
                            out=out, label=label, settings=settings) + pause_args
        record = run_arena(cmd, out=out, log=out_dir / f"arena_{name}.log")
        results[name] = {"opponent": str(opponent), "score": record.get("score"), "elo": record.get("elo"),
                         "elo_ci95": record.get("elo_ci95"), "games": record.get("games", settings.games),
                         "pentanomial": record.get("pentanomial"),
                         # games scored as draws at the ply cap (arena --on-max-plies); None from an older arena
                         "max_plies_unresolved": record.get("max_plies_unresolved")}
        ledger_rows.append({"generation": generation, "match": name, "candidate": str(candidate),
                            "opponent": str(opponent), "record": record})
    # Idempotent: this generation's rows replace any an interrupted attempt
    # left, written together after both matches finished.
    replace_ledger_rows(run_dir / LEDGER, generation, ledger_rows)
    write_manifest(out_dir, stage="evaluate", config={"settings": asdict(settings)},
                   inputs={"candidate": input_ref(candidate), "previous": input_ref(previous),
                           "openings": input_ref(openings_pgn),
                           **({"reference": input_ref(reference)} if reference is not None else {})},
                   outputs={f"arena_{n}": input_ref(out_dir / f"arena_{n}.jsonl") for n, _ in matches},
                   counts={"matches": len(matches)}, seeds={"arena": settings.seed},
                   extra={"generation": generation, "results": results})
    return results


def replace_ledger_rows(path: Path, generation: int, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    kept = []
    if path.exists():
        kept = [r for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())
                if int(r.get("generation", -1)) != int(generation)]
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.unlink(missing_ok=True)
    for r in [*kept, *rows]:
        append_jsonl(tmp, r)
    if not kept and not rows:
        tmp.write_text("")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# tripwires
# ---------------------------------------------------------------------------

TRIPWIRE_FIELDS: dict[str, tuple[str, str]] = {
    # flag name -> (table column, "min" | "max")
    "min_policy_entropy": ("policy_entropy", "min"),
    "min_target_entropy": ("target_entropy", "min"),
    "max_draw_rate": ("draw_rate", "max"),
    "min_decisive_rate": ("decisive_rate", "min"),
    "max_repeat_rate": ("repeat_rate_within", "max"),
    "max_repeat_rate_vs_previous": ("repeat_rate_vs_previous", "max"),
    "max_kl_target_prior": ("kl_target_prior", "max"),
}


def tripwire_table(run_dir: Path) -> list[dict[str, Any]]:
    """One row per generation with a finished collect (and train, if done)."""
    rows = []
    for gen_dir in sorted(Path(run_dir).glob("gen_[0-9][0-9][0-9][0-9]")):
        g = int(gen_dir.name.split("_")[1])
        if g == 0:
            continue
        collect = read_manifest(gen_dir / "collect")
        if collect is None:
            continue
        s = collect["stats"]
        row: dict[str, Any] = {"generation": g, "target_entropy": s.get("target_entropy"),
                               "prior_entropy": s.get("prior_entropy"), "kl_target_prior": s.get("kl_target_prior"),
                               "draw_rate": s.get("draw_rate"), "decisive_rate": s.get("decisive_rate"),
                               "repeat_rate_within": s.get("repeat_rate_within"),
                               "repeat_rate_vs_previous": s.get("repeat_rate_vs_previous"),
                               "policy_entropy": None}
        train = read_manifest(gen_dir / "train")
        if train is not None:
            row["policy_entropy"] = train["summary"]["heldout_after"].get("policy_entropy")
        rows.append(row)
    return rows


def check_tripwires(rows: list[dict[str, Any]], thresholds: dict[str, float | None]) -> list[str]:
    """Breaches in the NEWEST generation's row. Raises when nothing is armed."""
    armed = {k: v for k, v in thresholds.items() if v is not None}
    unknown = set(armed) - set(TRIPWIRE_FIELDS)
    if unknown:
        raise ValueError(f"unknown tripwire thresholds {sorted(unknown)}")
    if not armed:
        raise ValueError("--tripwire needs at least one threshold flag; refusing a check that cannot fail")
    bad = {k: v for k, v in armed.items() if not math.isfinite(float(v))}
    if bad:
        raise ValueError(f"tripwire thresholds must be finite (a NaN/inf limit can never trip): {bad}")
    if not rows:
        raise ValueError("no finished generation to check")
    row = rows[-1]
    breaches = []
    for name, limit in armed.items():
        column, kind = TRIPWIRE_FIELDS[name]
        value = row.get(column)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            breaches.append(f"gen {row['generation']}: {column} unavailable for --{name.replace('_', '-')}")
            continue
        if (kind == "min" and value < limit) or (kind == "max" and value > limit):
            breaches.append(f"gen {row['generation']}: {column}={value:.4f} crosses --{name.replace('_', '-')} {limit}")
    return breaches


def format_table(rows: list[dict[str, Any]]) -> str:
    cols = ["generation", "policy_entropy", "target_entropy", "prior_entropy", "kl_target_prior",
            "draw_rate", "decisive_rate", "repeat_rate_within", "repeat_rate_vs_previous"]

    def cell(v: Any) -> str:
        if v is None:
            return "-"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = ["  ".join(cols)]
    lines += ["  ".join(cell(r.get(c)) for c in cols) for r in rows]
    return "\n".join(lines)
