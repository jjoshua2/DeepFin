"""Audit CONFIGURED ratio knobs against what the loop actually REALIZED.

Every bug this exists to catch has the same shape: a number in the yaml that
is not the number the pipeline produced, with nothing anywhere asserting the
two agree. Seven of them shipped to production undetected:

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
  - `soft_policy_temp` 3.0 in the yaml against a realized 2.0 for five months,
    because the key was published by nobody and consumed by nobody. The
    reco-vs-yaml diff that should have caught it compared only the keys the two
    sides SHARE, so an absent key was structurally invisible (rl_loop_audit
    E13/A7). `--reco-diff` now runs that comparison over the UNION.
  - `params.json` read as though it were the realized config. It is the trial's
    LAUNCH config, and Ray rewrites it on every checkpoint, so the mtime tracks
    the run while the contents never move (rl_loop_audit J5). The provenance
    section names the authoritative source per key and lists what is stale.
  - this script itself, on `shuffle_buffer_size` and its four siblings: the
    reloader overlaid them into the trial config every iteration while the
    `DiskReplayBuffer` that consumes them was built once at startup, so the
    provenance section printed them under "the running value is correct"
    about a value nothing was running. It was structurally incapable of any
    other verdict — the check compared the yaml against the overlaid dict.
    Fixed 2026-08-01 by classifying them restart-required
    (`construction_only_config_keys()`), which makes the live edit warn and
    reads out here as PENDING-RESTART.

This is a point-in-time audit, NOT a loop guard — `loop_health.py` owns the
per-iteration alerting. Run it after a deploy, before trusting a throughput
number, or when a ratio knob is suspected of being inert. Exit 1 if any knob
diverges beyond tolerance, so it is still cron-safe if you want it periodic.

Usage:
  PYTHONPATH=. python3 scripts/audit_realized_config.py
  PYTHONPATH=. python3 scripts/audit_realized_config.py --last 20
  PYTHONPATH=. python3 scripts/audit_realized_config.py --result-json PATH
  PYTHONPATH=. python3 scripts/audit_realized_config.py --reco-diff
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping

from chess_anti_engine.tune.result_keys import row_counter, row_counter_opt
from chess_anti_engine.utils.config_yaml import SELFPLAY_CONFIG_KEYS
from scripts.loop_health import load_rows, parse_outcome_stats
from scripts.trial_paths import default_run_dir, latest_result_path

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


def flatten_yaml_config(raw: Mapping[str, object]) -> dict[str, object]:
    """Flatten one level: the production yaml groups knobs under sections."""
    flat: dict[str, object] = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            flat.update(val)
        else:
            flat[key] = val
    return flat


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

    flat = flatten_yaml_config(raw)

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


# ---------------------------------------------------------------------------
# J5: is the source you read a "realized" config from actually realized?
#
# Three files claim to describe the running trial's config and only one of them
# does:
#
#   params.json                Ray's snapshot of the config the trial was
#                              CONSTRUCTED with. Rewritten on every checkpoint,
#                              so its mtime tracks the run and it looks current;
#                              the contents never move. Authoritative only for
#                              keys a live reload refuses to apply.
#   configs/*.yaml (live)      re-read every iteration; wins for every key the
#                              reloader will apply, but it is an INPUT — a
#                              rejected reload leaves it disagreeing with what
#                              runs, which is the failure this section detects.
#   result.json row["config"]  the trial's own config dict AFTER the reload, one
#                              snapshot per iteration. The realized source.
#
# Measured on the live trial 2026-07-26: params.json read
# opening_fen_dole_per_iter=1 and .../retire_307.txt while the yaml and the
# realized row both read 0 and retire_59. Nothing in the file says so.
# ---------------------------------------------------------------------------

# Keys whose yaml value moves on its own between iterations, so a yaml-vs-row
# difference is the healthy state rather than a rejected reload. Each entry must
# say what rewrites it — an unexplained entry silences a real finding.
_PROVENANCE_ROTATING_KEYS: dict[str, str] = {
    "opening_fen_list_path": (
        "the blind-spot retire loop rewrites the yaml with a new retire_N file "
        "between iterations, so the newest row legitimately lags by one rotation"
    ),
}


def classify_config_provenance(
    params: Mapping[str, object],
    flat_yaml: Mapping[str, object],
    realized: Mapping[str, object],
    *,
    restart_keys: Iterable[str],
    searched_keys: Iterable[str] = (),
    construction_only_keys: Iterable[str] = (),
    yaml_is_newer_than_row: bool = False,
) -> tuple[list[str], list[str]]:
    """Label every shared config key by which source is authoritative for it.

    Returns ``(report_lines, findings)``. Four outcomes per key:

      * restart-required and yaml != realized — the yaml was edited and the
        running trial has not taken it. A FINDING: this is the
        "my change may not be in effect" trap, and it is silent.
      * construction-only, yaml == row, params.json != row — a NOTE, never a
        finding: the row is what the constructor was handed (the reloader
        refuses to overlay these), and params.json is the trial's ORIGINAL
        creation config, so a difference here means a restart picked the yaml
        up. Reading it the other way round is the J5 error itself.
      * live-reloadable and yaml != realized — the reload did not land. A
        FINDING: the live-yaml validator is all-or-nothing, so one unknown key
        rejects every other key's change too.
      * live-reloadable and params.json != realized — the reloader overlaid the
        yaml value, reported by count and by name because reading that file is
        how the trap fires.

    Compared against ``realized`` rather than against each other on purpose: a
    ``params.json``-vs-yaml diff cannot distinguish "the reload landed" from
    "the reload was rejected and both sources are wrong".

    ``restart_keys`` is injected rather than imported so this stays a pure
    function testable without a built C extension; the caller supplies
    ``trainable_config_ops.restart_required_config_keys()``.

    ``construction_only_keys`` is the subset of those the reloader freezes
    because a live edit could not act, not because acting would be unsafe
    (``trainable_config_ops.construction_only_config_keys()``). It is passed
    SEPARATELY, and checked alongside ``restart_keys`` rather than through it,
    so the "params.json is merely the launch value" verdict cannot be reached
    for a key with no live consumer even if ``restart_keys`` is wrong — which
    is exactly the state this function was in until 2026-08-01, when it printed
    ``shuffle_buffer_size: params.json=25000 running=100000`` under the header
    "the running value is correct" about a cap the ``DiskReplayBuffer`` had
    been built with at 25000 and never re-read.

    ``yaml_is_newer_than_row`` downgrades every yaml-vs-realized difference to a
    printed note. The realized row is minutes old by construction, so a yaml
    edited after it has simply not been read yet — reporting that as a rejected
    reload would make "edit the yaml, then audit" fire a false alarm every time,
    and a monitor that cries wolf is a monitor nobody reads.
    """
    restart = set(restart_keys)
    construction_only = set(construction_only_keys)
    searched = set(searched_keys)
    stale: list[str] = []
    report: list[str] = []
    findings: list[str] = []

    def _diverged(kind: str, key: str, on_disk: object, live: object, why: str) -> None:
        if yaml_is_newer_than_row:
            report.append(
                f"  {kind}-UNRESOLVED {key}: yaml={on_disk!r} running={live!r} "
                "(yaml edited after the newest row — re-run after the next iteration)"
            )
            return
        if key in _PROVENANCE_ROTATING_KEYS:
            report.append(
                f"  {kind}-EXPECTED {key}: yaml={on_disk!r} running={live!r} "
                f"({_PROVENANCE_ROTATING_KEYS[key]})"
            )
            return
        report.append(f"  {kind} {key}: yaml={on_disk!r} running={live!r}")
        findings.append(why)

    for key in sorted(set(flat_yaml) & set(realized)):
        if key in searched or key.startswith("pb2_bounds_"):
            continue
        on_disk = flat_yaml[key]
        live = realized[key]
        if key in restart or key in construction_only:
            if not _same_value(on_disk, live):
                extra = (
                    " (it is a constructor argument — the object using it was "
                    "built at startup and never re-reads config)"
                    if key in construction_only else ""
                )
                _diverged(
                    "PENDING-RESTART", key, on_disk, live,
                    f"{key}: the yaml says {on_disk!r} but the trial is running "
                    f"{live!r} and a live reload will never apply it{extra} — "
                    "restart required, or the experiment is not the one you think",
                )
            elif key in construction_only and key in params \
                    and not _same_value(params[key], live):
                # NOT a finding, and specifically not "the object is running
                # params.json's value" -- that inference is the J5 error this
                # script exists to warn about. params.json is the trial's
                # ORIGINAL creation config; the startup reload runs BEFORE the
                # constructor (trainable.py:520 precedes :634), so every restart
                # since the yaml moved handed the constructor the yaml value
                # while params.json kept the creation-time one. Measured on the
                # live trial 2026-08-01: params.json said shuffle_buffer_size
                # 25000 across seven restarts that each built the buffer at the
                # yaml's 100000.
                report.append(
                    f"  ok(ctor)  {key}: row={live!r} matches the yaml; "
                    f"params.json={params[key]!r} is the trial's ORIGINAL "
                    "creation config and does not describe this process"
                )
            continue
        if not _same_value(on_disk, live):
            _diverged(
                "RELOAD-NOT-APPLIED", key, on_disk, live,
                f"{key}: live-reloadable, but the yaml value {on_disk!r} has not "
                f"reached the running trial ({live!r}) — one unknown key rejects "
                "the WHOLE yaml reload",
            )
        elif key in params and not _same_value(params[key], live):
            stale.append(f"      {key}: params.json={params[key]!r} running={live!r}")
    if stale:
        report.append(
            f"  STALE-IN-PARAMS-JSON  {len(stale)} live-reloadable key(s) where "
            "params.json holds the LAUNCH value and the reloader has overlaid "
            "the yaml value:"
        )
        report.extend(stale)
        report.append(
            "      That the RELOADER applied it is all this shows. Whether the "
            "component re-read it is a per-key fact about consumers; for the "
            "ones already proven inert see construction_only_config_keys()."
        )
    return report, findings


def audit_config_provenance(
    rows: list[dict], yaml_path: Path | None, params_path: Path | None,
) -> list[str]:
    """Name the authoritative source per key, and check the yaml actually landed."""
    # Imported here, not at module scope: trainable_config_ops pulls in the
    # selfplay package (and its C extension), and this script must stay
    # importable wherever the extension is not built.
    from chess_anti_engine.tune.trainable_config_ops import (
        construction_only_config_keys,
        restart_required_config_keys,
    )

    print()
    print("=== config provenance: params.json vs live yaml vs realized row ===")
    cfg = rows[-1].get("config") or {}
    path = yaml_path or Path(str(cfg.get("_yaml_config_path") or ""))
    if not cfg:
        print("  SKIP  the newest result row carries no config block")
        return []
    if not path or not path.exists():
        print(f"  SKIP  yaml not found ({path})")
        return []
    try:
        import yaml as _yaml

        raw = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # yaml.YAMLError is not an OSError/ValueError
        print(f"  SKIP  could not parse {path}: {exc}")
        return []

    params: dict[str, object] = {}
    params_stamp = ""
    if params_path is not None and params_path.exists():
        try:
            params = json.loads(params_path.read_text(encoding="utf-8"))
            params_stamp = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(params_path.stat().st_mtime)
            )
        except (OSError, ValueError) as exc:
            print(f"  WARN  could not read {params_path}: {exc}")
    # The realized row is by construction one iteration old, so a yaml edited
    # since then has not been read yet and every difference it introduces is
    # unresolved rather than rejected.
    row_ts = rows[-1].get("timestamp")
    yaml_mtime = path.stat().st_mtime
    yaml_is_newer = row_ts is not None and yaml_mtime > float(row_ts)

    print(f"  realized:    iter {rows[-1].get('training_iteration', '?')} "
          "(result.json row config — the only per-iteration source)")
    print(f"  live yaml:   {path}  (authoritative for live-reloadable keys"
          + (", EDITED SINCE THE NEWEST ROW" if yaml_is_newer else "") + ")")
    if params:
        print(f"  params.json: {params_path}  (LAUNCH config; rewritten "
              f"{params_stamp}, so the mtime tracks the run and the CONTENTS DO NOT)")
    else:
        print("  params.json: not read — the launch-vs-live staleness list is skipped")

    searched = {k.removeprefix("pb2_bounds_") for k in cfg if k.startswith("pb2_bounds_")}
    report, findings = classify_config_provenance(
        params, flatten_yaml_config(raw), cfg,
        restart_keys=restart_required_config_keys(),
        searched_keys=searched,
        construction_only_keys=construction_only_config_keys(),
        yaml_is_newer_than_row=yaml_is_newer,
    )
    for line in report:
        print(line)
    if not findings:
        print("  ok   every shared yaml key has reached the running trial")
    return findings


# ---------------------------------------------------------------------------
# A4 / A7: does every worker-affecting yaml key actually reach the worker?
#
# A4 used to diff the published `recommended_worker` against the yaml over the
# INTERSECTION of the two key sets and report "63 of 64 exact". A key that is
# absent from the reco entirely never enters that comparison, so the check was
# structurally incapable of seeing the largest live config defect in the loop:
# `soft_policy_temp` read 3.0 in the yaml for five months while every worker
# built the target at GameConfig's default of 2.0, because nobody published it
# and nobody consumed it (rl_loop_audit E13/A7). The diff below runs over the
# UNION, so "absent" is a reportable state rather than an invisible one.
# ---------------------------------------------------------------------------

# Worker-affecting yaml keys the server deliberately does not publish under
# their own name. Each entry must say WHY, because "it isn't published" is the
# defect this section exists to find — an unexplained entry here is a bug in
# disguise.
_RECO_SERVER_RESOLVED: dict[str, str] = {
    "games_per_iter": "server-only game budget; workers are handed games_per_batch",
    "games_per_iter_start": "server-only ramp input to games_per_iter",
    "games_per_iter_ramp_iters": "server-only ramp input to games_per_iter",
    "selfplay_batch": "published under the name games_per_batch",
    "mcts_start_simulations": "server resolves the ramp into mcts_simulations",
    "mcts_ramp_steps": "server resolves the ramp into mcts_simulations",
    "mcts_ramp_exponent": "server resolves the ramp into mcts_simulations",
    "opening_fen_dole_max_fraction": (
        "server resolves it against games_per_iter into opening_fen_dole_max_games"
    ),
    "opening_book_path": "served as a manifest download endpoint, not a reco value",
    "opening_fen_list_path": "served as a manifest download endpoint, not a reco value",
}

# Keys the worker intentionally leaves at its own dataclass default, mapped to
# the default it must equal. Deliberately self-invalidating: the moment the
# yaml moves OFF the default the allowlist stops covering the key and it
# becomes a finding, because at that point the yaml is asking for something the
# worker will never do. `test_reco_worker_defaults_match_dataclasses` pins
# these numbers to the real dataclass fields so the table cannot rot.
_RECO_WORKER_DEFAULT: dict[str, object] = {
    "fpu_reduction": 1.2,          # SearchConfig.fpu_reduction
    "fpu_at_root": 1.0,            # SearchConfig.fpu_at_root
    "temperature_drop_plies": 0,   # TemperatureConfig.drop_plies
    "temperature_after": 0.0,      # TemperatureConfig.after
    "categorical_bins": 32,        # GameConfig.categorical_bins
    "hlgauss_sigma": 0.04,         # GameConfig.hlgauss_sigma
  # diff_focus moved here 2026-07-28. The worker never builds a DiffFocusConfig
  # from the reco, so these five have always run the dataclass defaults. Until
  # 2026-07-26 the yaml asked for the Run-4 sweep winners instead, which made
  # them a live DIVERGENCE tracked separately; that entry pinned the yaml to the
  # realized values as config honesty, and a divergence that no longer diverges
  # belongs in this table. Self-invalidating is the point: if anyone "tunes"
  # diff_focus in the yaml it stops matching and becomes a finding again,
  # instead of silently doing nothing.
    "diff_focus_enabled": True,    # DiffFocusConfig.enabled
    "diff_focus_q_weight": 6.0,    # DiffFocusConfig.q_weight
    "diff_focus_pol_scale": 3.5,   # DiffFocusConfig.pol_scale
    "diff_focus_slope": 3.0,       # DiffFocusConfig.slope
    "diff_focus_min": 0.025,       # DiffFocusConfig.min_keep
}

# Keys whose reco value is EXPECTED to differ from the yaml value, with why.
_RECO_INTENTIONAL_VALUE_DIVERGENCE: dict[str, str] = {
    "sf_nodes": (
        "the yaml key is the PID FLOOR; the published value is the live "
        "base_nodes the controller has ramped to"
    ),
}


def _same_value(a: object, b: object) -> bool:
    """Value equality that survives the yaml->json round trip (1 vs 1.0)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
    return str(a) == str(b)


def diff_reco_coverage(
    reco: Mapping[str, object],
    flat_cfg: Mapping[str, object],
    *,
    selfplay_keys: Iterable[str] = SELFPLAY_CONFIG_KEYS,
) -> tuple[list[str], list[str]]:
    """Diff the published reco against the yaml over the UNION of key sets.

    Returns ``(report_lines, findings)``. Four outcomes per key:

      both sides      -> values compared (see _RECO_INTENTIONAL_VALUE_DIVERGENCE)
      yaml only       -> FINDING unless allowlisted; this is the E13 class
      reco only       -> context: the worker runs the server's default for it
      neither side    -> context: nobody configures it, worker default stands

    ``selfplay_keys`` is the canonical enumeration of worker-affecting yaml
    keys, so a key that no yaml currently sets is still in the basis — a knob
    that is unpublished AND unset is one yaml edit away from being the next
    silent no-op, and it should be visible before that edit, not after.
    """
    report: list[str] = []
    findings: list[str] = []
    basis = set(selfplay_keys) | set(reco)
    unset: list[str] = []

    for key in sorted(basis):
        in_reco = key in reco
        in_yaml = key in flat_cfg
        if in_reco and in_yaml:
            why = _RECO_INTENTIONAL_VALUE_DIVERGENCE.get(key)
            if _same_value(reco[key], flat_cfg[key]):
                continue
            if why is not None:
                report.append(
                    f"  ok(intent) {key}: reco={reco[key]!r} yaml={flat_cfg[key]!r} — {why}"
                )
                continue
            report.append(f"  DIVERGENT {key}: reco={reco[key]!r} yaml={flat_cfg[key]!r}")
            findings.append(
                f"{key}: published reco says {reco[key]!r} but the yaml says "
                f"{flat_cfg[key]!r} — workers are running the published value"
            )
        elif in_yaml:
            if key in _RECO_SERVER_RESOLVED:
                report.append(f"  ok(server) {key}: {_RECO_SERVER_RESOLVED[key]}")
            elif key in _RECO_WORKER_DEFAULT:
                default = _RECO_WORKER_DEFAULT[key]
                if _same_value(flat_cfg[key], default):
                    report.append(
                        f"  ok(default) {key}={flat_cfg[key]!r}: unpublished on purpose, "
                        f"and equal to the worker default"
                    )
                else:
                    report.append(
                        f"  ABSENT    {key}: yaml={flat_cfg[key]!r} but the worker "
                        f"default is {default!r} and the key is not published"
                    )
                    findings.append(
                        f"{key}: yaml asks for {flat_cfg[key]!r} but the key is absent "
                        f"from recommended_worker, so every worker uses {default!r}. "
                        "Either publish+consume it or put the yaml back on the default"
                    )
            else:
                report.append(f"  ABSENT    {key}={flat_cfg[key]!r}: never published")
                findings.append(
                    f"{key}: set in the yaml but absent from recommended_worker — "
                    "no worker can ever see it, and the selfplay dataclass silently "
                    "keeps its own default (this is the soft_policy_temp/E13 class)"
                )
        elif in_reco:
            report.append(
                f"  ok(unset)  {key}={reco[key]!r}: published, but no yaml key sets it"
            )
        else:
            unset.append(key)

    if unset:
        report.append(
            f"  ok(inert)  {len(unset)} selfplay key(s) neither published nor set in "
            f"the yaml: {', '.join(unset)}"
        )
    return report, findings


def _find_live_manifest(rows: list[dict], explicit: Path | None) -> Path | None:
    """Newest publish/manifest.json for this run, or None."""
    if explicit is not None:
        return explicit if explicit.exists() else None
    cfg = rows[-1].get("config") or {}
    roots: list[Path] = []
    work_dir = str(cfg.get("work_dir") or "").strip()
    if work_dir:
        roots.append(Path(work_dir) / "server")
    roots.append(default_run_dir() / "server")
    for root in roots:
        found = sorted(
            root.glob("trials/*/publish/manifest.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if found:
            return found[0]
    return None


def audit_reco_coverage(
    rows: list[dict], yaml_path: Path | None, manifest_path: Path | None,
) -> list[str]:
    """Published reco vs the yaml, over the UNION of their key sets (A4/A7)."""
    print()
    print("=== published reco vs yaml (UNION of key sets) ===")
    cfg = rows[-1].get("config") or {}
    path = yaml_path or Path(str(cfg.get("_yaml_config_path") or ""))
    if not path or not path.exists():
        print(f"  SKIP  yaml not found ({path})")
        return []
    manifest = _find_live_manifest(rows, manifest_path)
    if manifest is None:
        print("  SKIP  no publish/manifest.json found — this check needs a "
              "distributed run's published manifest")
        return []
    try:
        import yaml as _yaml

        raw = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        published = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:  # yaml.YAMLError is not an OSError/ValueError
        print(f"  SKIP  could not read {path} / {manifest}: {exc}")
        return []
    reco = published.get("recommended_worker")
    if not isinstance(reco, dict):
        print(f"  SKIP  {manifest} has no recommended_worker block")
        return []

    flat = flatten_yaml_config(raw)
    report, findings = diff_reco_coverage(reco, flat)
    print(f"  manifest: {manifest} ({len(reco)} published keys)")
    print(f"  yaml:     {path}")
    for line in report:
        print(line)
    if not findings:
        print("  ok   every worker-affecting yaml key is published or explained")
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
    ap.add_argument("--reco-diff", action="store_true",
                    help="also diff the published recommended_worker against the yaml "
                         "over the UNION of key sets (rl_loop_audit A4/A7). Off by "
                         "default because it needs a distributed run's manifest")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="publish/manifest.json for --reco-diff (default: newest under "
                         "the run's server/trials)")
    ap.add_argument("--params-json", type=Path, default=None,
                    help="the trial's params.json for the provenance section (default: "
                         "sibling of the result.json). It is the LAUNCH config, never a "
                         "realized one — the section exists to say so per key")
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
    findings += audit_config_provenance(
        rows, args.yaml, args.params_json or path.parent / "params.json",
    )
    if args.reco_diff:
        findings += audit_reco_coverage(rows, args.yaml, args.manifest)

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
