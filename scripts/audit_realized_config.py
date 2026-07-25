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
    """One configured ratio and the realized value to compare it against.

    ``semantics`` decides what "agrees" means, and it is an explicit field
    rather than something inferred from the note text -- a reworded note must
    not silently change the comparison:

      "target"       two-sided; realized should equal configured.
      "cap"          realized must not EXCEED configured. 0 disables the cap.
      "floor_target" realized should equal configured but is allowed to come
                     out HIGH, because the loop clamps it up from below.
    """

    def __init__(
        self,
        name: str,
        config_key: str,
        realized: Callable[[dict, dict], float | None],
        *,
        note: str = "",
        tol: float = _DEFAULT_TOL,
        semantics: str = "target",
    ) -> None:
        if semantics not in ("target", "cap", "floor_target"):
            raise ValueError(f"unknown knob semantics: {semantics!r}")
        self.name = name
        self.config_key = config_key
        self.realized = realized
        self.note = note
        self.tol = tol
        self.semantics = semantics


def _ratio(num: float | None, den: float | None) -> float | None:
    """num/den, or None when the denominator says nothing (no data this iter)."""
    if num is None or den is None:
        return None
    den_f = float(den)
    if den_f <= 0.0:
        return None
    return float(num) / den_f


def _views(row: dict, _stats: dict) -> float | None:
    v = row.get("train_views_actual")
    return float(v) if v is not None else None


def _seeded_share_of_games_per_iter(row: dict, stats: dict) -> float | None:
    """Seeded games / games_per_iter -- the base the cap is actually computed on.

    distributed_runtime does ``ceil(opening_fen_dole_max_fraction *
    games_per_iter)`` and hands that down as an absolute per-iteration seeded
    game bound, so the denominator is TOTAL games, not selfplay games. Dividing
    by selfplay_games instead inflates the reading by 1/selfplay_fraction --
    2x at the production 0.50, 5x at 0.20 -- and reports a cap that is being
    honoured exactly as a 2x breach. The ledger records the operator picking
    0.25 in full knowledge of this base ("seeded <= half of SELFPLAY, not half
    of total games", at selfplay_fraction 0.50).

    All four seeded channels count. `fenlist_backed` and
    `fenlist_sf_refute_backed` are seeded games too -- the backed variants are
    the GOAL channel -- so summing only the two unbacked ones would let a
    blame-backup-heavy pool breach the cap while reading compliant.
    """
    seeded = sum(
        int(v)
        for k, v in stats.items()
        if k.startswith("selfplay_fenlist") and k.endswith("_games")
    )
    cfg = row.get("config") or {}
    try:
        games_per_iter = float(cfg["games_per_iter"])
    except (KeyError, TypeError, ValueError):
        return None
    return _ratio(seeded, games_per_iter)


_KNOBS: tuple[Knob, ...] = (
    Knob(
        "replay views per ingested position",
        "train_views_per_ingested_position",
        _views,
        semantics="floor_target",
        note=("realized = train_views_actual; <1.0 means data is never trained on "
              "once. Reads HIGH legitimately when the fresh-samples floor or the "
              "ingest-drought fallback binds, so only the LOW side is a finding"),
    ),
    Knob(
        "seeded share of games_per_iter",
        "opening_fen_dole_max_fraction",
        _seeded_share_of_games_per_iter,
        semantics="cap",
        note=("a CAP on a share of games_per_iter (NOT of selfplay); sums all four "
              "selfplay_fenlist* channels"),
    ),
)

# DELIBERATELY NOT AUDITED: selfplay_fraction. It is not a share of completed
# games -- it is the probability rolled PER SLOT when a game starts
# (selfplay/state.py _init_color_and_selfplay_arrays, recycle_slot). The
# completed-game share only equals it when selfplay and curriculum games take
# the same wall time, and they emphatically do not: curriculum games run long
# against ~700k-node SF and ~89% of them get abandoned. So the steady-state
# share is f/d_sp / (f/d_sp + (1-f)/d_cur), and comparing selfplay_games /
# matching_games against the knob reads DIVERGENT forever while the code does
# exactly what it says. No emitted metric measures slot rolls (`games_started`
# exists in selfplay/state.py but is not reported). The honest reading is
# printed as context by `report_selfplay_mix` below; make it a pass/fail check
# only once a games-started counter exists.
#
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


def _configured(row: dict, key: str) -> tuple[float | None, str | None]:
    """This row's OWN configured value, or (None, why-not).

    Per-row and not "the newest row's config" on purpose: a window that spans a
    deploy mixes iterations that ran different values, and a median across the
    join is a number no configuration ever asked for. That mistake produced two
    confidently wrong verdicts on this tool's first real run -- a dole cap
    reported as entirely inert when the five pre-deploy iterations dominated
    the median, and a selfplay mix measured mostly on rows that ran a different
    value.
    """
    cfg = row.get("config") or {}
    if key not in cfg:
        return None, f"{key} is not in this iteration's effective config"
    raw = cfg[key]
    try:
        return float(raw), None
    except (TypeError, ValueError):
        return None, f"{key} is not a number in the config: {raw!r}"


def audit_knobs(rows: list[dict], stats: list[dict]) -> list[str]:
    """Print the configured-vs-realized table. Returns divergence findings."""
    findings: list[str] = []
    print("=== configured vs REALIZED (median over window) ===")
    compared = 0
    for knob in _KNOBS:
        configured, why = _configured(rows[-1], knob.config_key)
        if configured is None:
            # NOT a silent skip. "Nothing can check this knob" is exactly the
            # state every bug in the docstring was hiding in.
            print(f"  {knob.name:38s} UNCHECKABLE  ({why})")
            findings.append(
                f"{knob.config_key}: cannot audit — {why}. Either the knob was "
                "renamed and this script was not updated, or the config value is "
                "malformed; both leave the knob unverified."
            )
            continue

        # Only iterations that ran the CURRENT value are comparable.
        pairs = [
            (r, st) for r, st in zip(rows, stats, strict=True)
            if _configured(r, knob.config_key)[0] == configured
        ]
        vals = [v for v in (knob.realized(r, st) for r, st in pairs) if v is not None]
        realized = _median(vals)

        if knob.semantics == "cap" and configured == 0.0:
            print(f"  {knob.name:38s} UNCAPPED     (configured 0 disables the cap; "
                  f"realized={realized if realized is None else round(realized, 4)})")
            print(f"      {knob.note}")
            compared += 1
            continue

        if realized is None:
            print(f"  {knob.name:38s} n/a   configured={configured:.4g} "
                  f"(no iteration in the window reported it)")
            findings.append(
                f"{knob.config_key}: configured {configured:.4g} but NOTHING realized it "
                f"over {len(pairs)} iters — the knob may be inert or the metric missing"
            )
            continue

        if knob.semantics == "cap":
            ok = realized <= configured * (1.0 + knob.tol)
        elif knob.semantics == "floor_target":
            ok = realized >= configured * (1.0 - knob.tol)
        else:
            ok = abs(realized - configured) <= max(knob.tol * abs(configured), 1e-9)
        compared += 1
        flag = "ok  " if ok else "DIVERGENT"
        ratio = realized / configured if configured else float("inf")
        print(f"  {knob.name:38s} {flag} configured={configured:.4g} "
              f"realized={realized:.4g} (x{ratio:.2f}, n={len(vals)})")
        if len(pairs) < len(rows):
            print(f"      NOTE: {knob.config_key} changed inside this window; judged on "
                  f"the {len(pairs)}/{len(rows)} iters that ran {configured:.4g}")
        print(f"      {knob.note}")
        if not ok:
            findings.append(
                f"{knob.config_key}: configured {configured:.4g} vs realized "
                f"{realized:.4g} (x{ratio:.2f}) — {knob.note}"
            )
    if compared == 0:
        findings.append(
            "no knob could be compared at all — this run proves nothing; do not "
            "read a clean exit as agreement"
        )
    return findings


def report_selfplay_mix(rows: list[dict], _stats: list[dict]) -> None:
    """Print the realized selfplay data mix as CONTEXT, never as pass/fail.

    See the selfplay_fraction note above: the knob sets a per-slot roll
    probability, so the completed-game share is expected to exceed it whenever
    curriculum games run longer than selfplay games. Printed because the mix
    itself matters (76% selfplay was the value-poison condition) -- but a
    number with no comparable configured value is not a divergence.
    """
    print()
    print("=== realized selfplay mix (context, not a check) ===")
    shares = [
        v for v in (
            _ratio(r.get("selfplay_games"), row_counter_opt(r, "matching_games"))
            for r in rows
        ) if v is not None
    ]
    med = _median(shares)
    configured, _ = _configured(rows[-1], "selfplay_fraction")
    if med is None:
        print("  n/a   no iteration reported selfplay_games")
        return
    cfg_txt = "unknown" if configured is None else f"{configured:.4g}"
    print(f"  selfplay share of completed games = {med:.4g} "
          f"(slot-roll knob = {cfg_txt}, n={len(shares)})")
    print("      Expected to run HIGHER than the knob: the gap is the "
          "curriculum/selfplay duration ratio, not a bug.")


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
    if checked == 0:
        print("  UNCHECKABLE  no audited knob appears in both the yaml and the "
              "effective config")
        findings.append(
            "yaml-vs-effective check compared 0 knobs — a live-reload rejection "
            "would look identical to agreement here"
        )
    elif not findings:
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
    report_selfplay_mix(rows, stats)
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
