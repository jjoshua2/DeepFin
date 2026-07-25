"""Audit CONFIGURED ratio knobs against what the loop actually REALIZED.

Every bug this exists to catch has the same shape: a number in the yaml that
is not the number the pipeline produced, with nothing anywhere asserting the
two agree. Four of them shipped to production undetected:

  - `train_views_per_position` 2.5 divided by matching-only positions while 4.5-6.5x
    more positions were ingested, so TRUE reuse was ~0.46 and over half of all
    data was never trained on once (fixed 2026-07-24, PR #225).
  - `selfplay_fraction` 0.35 against a realized completed-game mix of 83.5%,
    because session restarts abandoned ~58% of games and curriculum games are
    the long ones (fixed PR #224).
  - `opening_fen_dole_max_fraction` unbounded in practice: the seeded share of
    selfplay reached 100%, crowding normal openings to zero.
  - Workers frozen on a stale `model_sha` so ~80% of games were discarded as
    stale while both counters looked individually sane (fixed PR #228).

This is a point-in-time audit, NOT a loop guard — `loop_health.py` owns the
per-iteration alerting. Run it after a deploy, before trusting a throughput
number, or when a ratio knob is suspected of being inert. Exit 1 if any knob
diverges beyond tolerance, so it is still cron-safe if you want it periodic.

Usage:
  PYTHONPATH=. python3 scripts/audit_realized_config.py
  PYTHONPATH=. python3 scripts/audit_realized_config.py --last 20
  PYTHONPATH=. python3 scripts/audit_realized_config.py --result-json PATH
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from collections.abc import Callable

from chess_anti_engine.tune.result_keys import row_counter, row_counter_opt
from scripts.loop_health import load_rows, parse_outcome_stats
from scripts.trial_paths import latest_result_path

# Relative tolerance before a knob is called divergent. Generous on purpose:
# these are ratios of small per-iteration counts, so Poisson noise alone moves
# them several percent iteration to iteration. The failures this catches were
# 2x-6x wrong, not 15% wrong.
_DEFAULT_TOL = 0.15


class Knob:
    """One configured ratio and the realized value to compare it against."""

    def __init__(
        self,
        name: str,
        config_key: str,
        realized: Callable[[dict, dict], float | None],
        *,
        note: str = "",
        tol: float = _DEFAULT_TOL,
    ) -> None:
        self.name = name
        self.config_key = config_key
        self.realized = realized
        self.note = note
        self.tol = tol


def _ratio(num: float | None, den: float | None) -> float | None:
    """num/den, or None when the denominator says nothing (no data this iter)."""
    if num is None or den is None:
        return None
    den_f = float(den)
    if den_f <= 0.0:
        return None
    return float(num) / den_f


def _selfplay_share(row: dict, _stats: dict) -> float | None:
    return _ratio(row.get("selfplay_games"), row_counter_opt(row, "matching_games"))


def _views(row: dict, _stats: dict) -> float | None:
    v = row.get("train_views_actual")
    return float(v) if v is not None else None


def _seeded_share_of_selfplay(row: dict, stats: dict) -> float | None:
    """Seeded selfplay games / selfplay games — what the dole cap actually caps.

    The cap is deliberately a share of SELFPLAY, not of all games, so the
    denominator here must be selfplay_games and not matching_games.
    """
    seeded = int(stats.get("selfplay_fenlist_games", 0)) + int(
        stats.get("selfplay_fenlist_sf_refute_games", 0)
    )
    return _ratio(seeded, row.get("selfplay_games"))


_KNOBS: tuple[Knob, ...] = (
    Knob(
        "selfplay share of completed games",
        "selfplay_fraction",
        _selfplay_share,
        note="realized = selfplay_games / matching_games (COMPLETED, not slots)",
    ),
    Knob(
        "replay views per ingested position",
        "train_views_per_ingested_position",
        _views,
        note="realized = train_views_actual; <1.0 means data is never trained on once",
    ),
    Knob(
        "seeded share of selfplay",
        "opening_fen_dole_max_fraction",
        _seeded_share_of_selfplay,
        note="a CAP: realized should be <= configured, not equal to it",
    ),
)

# DELIBERATELY NOT AUDITED: holdout_fraction. It gates the share of each
# iteration's INGEST that is diverted to the holdout buffer, but the only
# emitted metric is `test_replay` — the buffer's current SIZE, which is also
# shaped by holdout_capacity, freeze_holdout_at and drift resets. Comparing
# test_replay/replay makes it look 15x low (0.0013 vs 0.02) when nothing is
# wrong. A check whose denominator we cannot justify teaches people to ignore
# the tool, which is worse than not checking. Add it back with a real
# per-iteration "holdout rows added" metric, not a guessed denominator.


def _median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def audit_knobs(rows: list[dict], stats: list[dict]) -> list[str]:
    """Print the configured-vs-realized table. Returns divergence findings."""
    findings: list[str] = []
    cfg = rows[-1].get("config") or {}
    print("=== configured vs REALIZED (median over window) ===")
    for knob in _KNOBS:
        if knob.config_key not in cfg:
            print(f"  {knob.name:38s} SKIP  ({knob.config_key} not in effective config)")
            continue
        configured = float(cfg[knob.config_key])
        vals = [v for v in (knob.realized(r, s) for r, s in zip(rows, stats, strict=True))
                if v is not None]
        realized = _median(vals)
        if realized is None:
            print(f"  {knob.name:38s} n/a   configured={configured:.4g} "
                  f"(no iteration in the window reported it)")
            findings.append(
                f"{knob.config_key}: configured {configured:.4g} but NOTHING realized it "
                f"over {len(rows)} iters — the knob may be inert or the metric missing"
            )
            continue
        # A cap is satisfied by being under it; everything else should match.
        is_cap = "CAP" in knob.note or knob.note.startswith("a CAP")
        if is_cap:
            ok = realized <= configured * (1.0 + knob.tol)
        else:
            ok = abs(realized - configured) <= max(knob.tol * abs(configured), 1e-9)
        flag = "ok  " if ok else "DIVERGENT"
        ratio = realized / configured if configured else float("inf")
        print(f"  {knob.name:38s} {flag} configured={configured:.4g} "
              f"realized={realized:.4g} (x{ratio:.2f}, n={len(vals)})")
        print(f"      {knob.note}")
        if not ok:
            findings.append(
                f"{knob.config_key}: configured {configured:.4g} vs realized "
                f"{realized:.4g} (x{ratio:.2f}) — {knob.note}"
            )
    return findings


def audit_counters(rows: list[dict]) -> list[str]:
    """Cross-check counter pairs whose names invite the wrong denominator."""
    findings: list[str] = []
    print()
    print("=== counter cross-checks ===")

    stale = [float(r.get("distributed_stale_games") or 0) for r in rows]
    games = [float(row_counter(r, "matching_games")) for r in rows]
    frozen = [s > g > 0 for s, g in zip(stale, games, strict=True)]
    n_frozen = sum(frozen)
    print(f"  stale_games > matching_games on {n_frozen}/{len(rows)} iters")
    print("      matching_games counts CURRENT-MODEL games only; a frozen worker fleet "
          "outproduces it")
    if n_frozen >= 2:
        findings.append(
            f"stale_games exceeded matching_games on {n_frozen}/{len(rows)} iters — "
            "workers frozen on an old model_sha; most selfplay is being discarded"
        )

    # matching_positions is current-model only; replay_positions_ingested is the truth.
    ratios = [r for r in (_ratio(row.get("replay_positions_ingested"),
                                 row_counter_opt(row, "matching_positions")) for row in rows)
              if r is not None]
    med = _median(ratios)
    if med is not None:
        print(f"  replay_positions_ingested / matching_positions = {med:.2f} (median)")
        print("      matching_positions is CURRENT-MODEL only. This ratio IS the views-denominator "
              "error factor: any per-position budget keyed off matching_positions is off by it")
        if med > 1.0 + _DEFAULT_TOL:
            findings.append(
                f"matching_positions undercounts true ingest by {med:.2f}x — verify no budget "
                "divides by it (this was the views bug, PR #225)"
            )
    return findings


def audit_wall_clock(rows: list[dict]) -> list[str]:
    """Reported iteration time vs actual wall clock.

    Ray's time_this_iter_s / time_total_s / time_since_restore are measured
    since the last RESTORE, so they silently exclude downtime and any attempt
    discarded by a restart. On 2026-07-24 iter 229 reported 1,208s against a
    6,687s timestamp gap — the difference was a watchdog auto-recover restart.
    """
    findings: list[str] = []
    print()
    print("=== wall clock: reported vs actual ===")
    prev_ts = None
    prev_isr = None
    for row in rows:
        ts = row.get("timestamp")
        it = row.get("training_iteration", "?")
        reported = float(row.get("time_this_iter_s") or 0.0)
        isr = row.get("iterations_since_restore")
        if prev_ts is not None and ts is not None:
            gap = float(ts) - float(prev_ts)
            unaccounted = gap - reported
            # A restore resets Ray's per-iteration timers, so a repeated
            # iterations_since_restore is the direct tell for a restart we did
            # not ask for.
            restarted = isr is not None and prev_isr is not None and int(isr) <= int(prev_isr)
            if unaccounted > max(0.25 * gap, 300.0):
                mark = "RESTART" if restarted else "UNACCOUNTED"
                print(f"  iter {it}: reported={reported:.0f}s wall={gap:.0f}s "
                      f"{mark} {unaccounted:.0f}s")
                findings.append(
                    f"iter {it}: {unaccounted:.0f}s of wall clock is not in "
                    f"time_this_iter_s ({reported:.0f}s reported vs {gap:.0f}s actual)"
                    + (" — iterations_since_restore reset, i.e. an unrequested restart"
                       if restarted else "")
                )
        prev_ts = ts
        prev_isr = isr
    if not findings:
        print("  ok   every iteration's reported time matches its timestamp delta")
    return findings


def audit_live_yaml(rows: list[dict], yaml_path: Path | None) -> list[str]:
    """Effective config vs the yaml on disk.

    The live-yaml validator is all-or-nothing: one unknown key rejects the
    WHOLE reload, silently leaving every knob at its pre-edit value. Comparing
    the trial's effective config against the file catches that.
    """
    findings: list[str] = []
    cfg = rows[-1].get("config") or {}
    path = yaml_path or Path(str(cfg.get("_yaml_config_path") or ""))
    print()
    print("=== effective config vs yaml on disk ===")
    if not path or not path.exists():
        print(f"  SKIP  yaml not found ({path})")
        return findings
    try:
        import yaml as _yaml

        raw = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"  SKIP  could not parse {path}: {exc}")
        return findings

    # Flatten one level: the production yaml groups knobs under sections.
    flat: dict[str, object] = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            flat.update(val)
        else:
            flat[key] = val

    checked = 0
    for knob in _KNOBS:
        if knob.config_key not in flat or knob.config_key not in cfg:
            continue
        checked += 1
        on_disk = flat[knob.config_key]
        effective = cfg[knob.config_key]
        try:
            same = abs(float(str(on_disk)) - float(str(effective))) < 1e-9
        except (TypeError, ValueError):
            same = on_disk == effective
        if not same:
            print(f"  DIVERGENT {knob.config_key}: yaml={on_disk} effective={effective}")
            findings.append(
                f"{knob.config_key}: yaml on disk says {on_disk} but the trial is running "
                f"{effective} — a live reload was likely REJECTED (one unknown key rejects "
                "the whole file)"
            )
    if not findings:
        print(f"  ok   {checked} ratio knobs match the yaml on disk")
    return findings


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--result-json", type=Path, default=None,
                    help="trial result.json (default: newest under $TRAIN_WORK_DIR/tune)")
    ap.add_argument("--last", type=int, default=10,
                    help="iterations to scan (default 10, min 2 — wall-clock and "
                         "frozen-fleet checks both need a previous row)")
    ap.add_argument("--yaml", type=Path, default=None,
                    help="config yaml to compare against (default: the trial's own "
                         "_yaml_config_path)")
    args = ap.parse_args()
    if args.last < 2:
        sys.exit("--last must be >= 2")

    path = args.result_json or latest_result_path(required=True)
    assert path is not None
    rows = load_rows(path, args.last)
    if len(rows) < 2:
        sys.exit(f"need >=2 iterations in {path}, found {len(rows)}")
    stats = [parse_outcome_stats(str(r.get("outcome_stats") or "")) for r in rows]

    first = rows[0].get("training_iteration", "?")
    last = rows[-1].get("training_iteration", "?")
    print(f"trial={path.parent.name}")
    print(f"window: iters {first}..{last} ({len(rows)} rows)")
    print()

    findings = audit_knobs(rows, stats)
    findings += audit_counters(rows)
    findings += audit_wall_clock(rows)
    findings += audit_live_yaml(rows, args.yaml)

    print()
    if not findings:
        print("AUDIT OK — every checked knob matches what the loop realized.")
        return
    print(f"AUDIT FOUND {len(findings)} DIVERGENCE(S):")
    for f in findings:
        print(f"  - {f}")
    sys.exit(1)


if __name__ == "__main__":
    main()
